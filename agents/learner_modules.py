# -*- coding: utf-8 -*-
"""
@module agents.learner_modules
@new 2026-05-02

异质学习型主体（heterogeneous learning agents）模块。

这个模块解决的核心问题：
    项目当前所有农户走同一套 prospect_theory + bayesian_belief 决策公式，
    仅参数随机抽样不同。这违反"中国农村存在多种结构性异质学习路径"的现实，
    也是 Nature 子刊审稿的关键软肋。

设计理念（与 modules/farmer_characteristics.py 的 8 类 FarmerType 对应）：
    - HabitualLearner       → TRADITIONAL_SUBSISTENCE
        老一辈、习惯依赖、口碑驱动、慢响应
    - AdaptiveLearner       → PART_TIME_AGRI_MAIN / PART_TIME_OFF_FARM_MAIN / WEEKEND_FARMER
        风险分散、收益跟踪、机会成本敏感
    - StrategicLearner      → COMMERCIAL_FARMER / NEW_GENERATION_FARMER
        长期规划、Q-learning、信息搜寻能力强
    - RentMaximizingLearner → URBAN_MIGRANT / ABSENTEE_LANDLORD
        脱离生产、租金最大化、被动维护

每个 learner 维护自己的 ExperienceBuffer，记录过去 10 年的关键决策与结果。
通过 mechanism_switches.use_learning_farmer = True 启用。

@theory:
    - Bandura (1977): 观察学习
    - Sutton & Barto (2018): 强化学习
    - Kahneman (2011): System 1/2 双系统决策
    - Brock & Hommes (1998): 异质期望
    - Manski (1993): 反身性社会效应
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.reproducibility import np_random_value, np_randint


# ============================================================================
# 经验记录
# ============================================================================

@dataclass
class FarmerExperience:
    """
    单年农户经验记录，进入 ExperienceBuffer 用于学习。

    与 prospect theory 的 reference_point 不同：
    这里记录的是离散事件（选择 + 结果 + 上下文），
    更像强化学习的 (s, a, r) 三元组。
    """
    year: int
    crop_choice: str             # 'grain' | 'cash_crop'
    transfer_choice: str         # 'keep' | 'rent_in' | 'rent_out' | 'abandon'
    income: float
    subsidy_received: float
    grain_price: float
    cash_price: float
    neighbor_grain_share: float  # 邻居中种粮户比例 [0,1]
    weather_shock: bool
    rent_offer_received: float   # 当年获得的最高出租报价（元/亩）
    reward_realized: float       # 综合 reward signal（agent type 自定义口径）

    def is_grain(self) -> bool:
        return self.crop_choice == 'grain'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'year': self.year,
            'crop_choice': self.crop_choice,
            'transfer_choice': self.transfer_choice,
            'income': self.income,
            'subsidy_received': self.subsidy_received,
            'grain_price': self.grain_price,
            'cash_price': self.cash_price,
            'neighbor_grain_share': self.neighbor_grain_share,
            'weather_shock': self.weather_shock,
            'rent_offer_received': self.rent_offer_received,
            'reward_realized': self.reward_realized,
        }


# ============================================================================
# 基类
# ============================================================================

class BaseLearner(ABC):
    """所有 learner module 的基类。"""

    def __init__(self, farmer, buffer_size: int = 10):
        self.farmer = farmer
        self.experience_buffer: deque = deque(maxlen=buffer_size)
        self.last_state: Optional[Tuple] = None
        self.last_action_crop: Optional[str] = None
        self.last_action_transfer: Optional[str] = None
        self.learner_type: str = self.__class__.__name__

    def observe(self, experience: FarmerExperience) -> None:
        """记录单年经验。"""
        self.experience_buffer.append(experience)

    def buffer_size(self) -> int:
        return len(self.experience_buffer)

    def recent_grain_freq(self, window: int = 5) -> float:
        """过去 window 年里种粮的频率，无历史时返回 0.5（中性先验）。"""
        if not self.experience_buffer:
            return 0.5
        recent = list(self.experience_buffer)[-window:]
        return sum(1 for e in recent if e.is_grain()) / len(recent)

    def recent_avg_income(self, crop: Optional[str] = None) -> Optional[float]:
        """过去经验里指定作物的平均收入。"""
        if not self.experience_buffer:
            return None
        if crop is None:
            xs = [e.income for e in self.experience_buffer]
        else:
            xs = [e.income for e in self.experience_buffer if e.crop_choice == crop]
        return float(np.mean(xs)) if xs else None

    @abstractmethod
    def decide_crop(self, env_signals: Dict[str, Any]) -> str:
        """返回 'grain' 或 'cash_crop'。"""
        ...

    @abstractmethod
    def decide_transfer(self, env_signals: Dict[str, Any]) -> str:
        """返回 'keep' | 'rent_in' | 'rent_out' | 'abandon'。"""
        ...

    def export_state(self) -> Dict[str, Any]:
        """用于审计：导出当前学习状态。"""
        return {
            'learner_type': self.learner_type,
            'buffer_size': self.buffer_size(),
            'recent_grain_freq': self.recent_grain_freq(),
            'last_state': self.last_state,
            'last_action_crop': self.last_action_crop,
            'last_action_transfer': self.last_action_transfer,
        }

    def export_sar(self, env_signals: Dict[str, Any] = None) -> Dict[str, Any]:
        """导出当前 (state, action) 用于 SAR 日志。

        Ponytail: 只返回已有字段，不为 SAR 新建计算。
        policy_mode 由调用方（SAR logger）根据 mechanism_switches 注入。
        注意：Habitual/Adaptive/RentMaximizing 没有 update_after_outcome，
        因此不产生 reward 或 next_state——它们不是 RL agent。
        """
        return {
            'state': self.last_state,
            'action': {
                'crop': self.last_action_crop,
                'transfer': self.last_action_transfer,
            },
            'learner_type': self.learner_type,
            'is_rl_agent': isinstance(self, StrategicLearner),
        }


# ============================================================================
# HabitualLearner — 习惯依赖型（传统生存型小农）
# ============================================================================

class HabitualLearner(BaseLearner):
    """
    适用 FarmerType: TRADITIONAL_SUBSISTENCE

    决策心理学基础：System 1 习惯决策（Kahneman），结合社会规范
    （Coleman, Foundations of Social Theory）。

    决策机制：
        score_grain = 0.70 × 自身过去 5 年种粮频率
                    + 0.25 × 邻居当年种粮比例
                    + 0.05 × 价格信号
    新信息只占 5%，习惯+口碑占 95%。这反映老一辈农户的真实行为。
    """

    OWN_HISTORY_WEIGHT = 0.70
    NEIGHBOR_WEIGHT = 0.25
    PRICE_WEIGHT = 0.05
    DEFAULT_PRIOR_GRAIN = 0.7  # 习惯农户先验偏向种粮

    def decide_crop(self, env_signals: Dict[str, Any]) -> str:
        own_grain_freq = self.recent_grain_freq(window=5)
        if self.buffer_size() < 3:
            own_grain_freq = self.DEFAULT_PRIOR_GRAIN

        neighbor_grain_share = float(env_signals.get('neighbor_grain_share', 0.5))
        price_signal = float(env_signals.get('grain_premium', 0.0))
        # price_signal 截断到 [-0.5, 0.5] 避免极端价格偏离主导决策
        price_signal = max(-0.5, min(0.5, price_signal))
        # 平移到 [0, 1] 以便加权
        price_score = (price_signal + 0.5)

        score_grain = (
            self.OWN_HISTORY_WEIGHT * own_grain_freq
            + self.NEIGHBOR_WEIGHT * neighbor_grain_share
            + self.PRICE_WEIGHT * price_score
        )
        decision = 'grain' if score_grain > 0.5 else 'cash_crop'

        # 记录状态用于审计
        self.last_state = ('habitual', round(own_grain_freq, 2), round(neighbor_grain_share, 2))
        self.last_action_crop = decision
        return decision

    def decide_transfer(self, env_signals: Dict[str, Any]) -> str:
        # 习惯型几乎不主动流转；仅在年龄>=70 或健康问题时被动出租
        age = getattr(self.farmer, 'age', 50)
        if age >= 70:
            decision = 'rent_out'
        elif age >= 65 and float(env_signals.get('rent_offer_per_mu', 0)) > 250:
            # 老年人遇到高价才考虑出租
            decision = 'rent_out'
        else:
            decision = 'keep'
        self.last_action_transfer = decision
        return decision


# ============================================================================
# AdaptiveLearner — 适应型（兼业农户）
# ============================================================================

class AdaptiveLearner(BaseLearner):
    """
    适用 FarmerType: PART_TIME_AGRI_MAIN, PART_TIME_OFF_FARM_MAIN, WEEKEND_FARMER

    决策心理学基础：适应性预期（Cyert & March, 1963）+ Brock-Hommes 异质期望。
    通过 ExperienceBuffer 计算两类作物的历史平均收益，并扣除非农机会成本。

    决策机制：
        adjusted_return(crop) = avg_income(crop)
                              - (1 - agri_dependence) × off_farm_wage × 0.1
        choose argmax across {grain, cash_crop}
    """

    OFF_FARM_OPPORTUNITY_WEIGHT = 0.1

    def decide_crop(self, env_signals: Dict[str, Any]) -> str:
        grain_returns = [e.income for e in self.experience_buffer if e.crop_choice == 'grain']
        cash_returns = [e.income for e in self.experience_buffer if e.crop_choice == 'cash_crop']

        # 经验不足时落到价格信号驱动
        if len(grain_returns) < 2 and len(cash_returns) < 2:
            premium = float(env_signals.get('grain_premium', 0.0))
            decision = 'grain' if premium >= -0.05 else 'cash_crop'
            self.last_state = ('adaptive', 'cold_start')
            self.last_action_crop = decision
            return decision

        grain_mean = float(np.mean(grain_returns)) if grain_returns else 0.0
        cash_mean = float(np.mean(cash_returns)) if cash_returns else 0.0

        # 加入非农机会成本
        off_farm_wage = float(env_signals.get('off_farm_wage', 0.0))
        agri_dep = self._get_agri_dependence()
        opportunity_cost = (1.0 - agri_dep) * off_farm_wage * self.OFF_FARM_OPPORTUNITY_WEIGHT

        adjusted_grain = grain_mean - opportunity_cost
        adjusted_cash = cash_mean - opportunity_cost

        decision = 'grain' if adjusted_grain >= adjusted_cash else 'cash_crop'
        self.last_state = (
            'adaptive',
            round(grain_mean, 0),
            round(cash_mean, 0),
            round(agri_dep, 2),
        )
        self.last_action_crop = decision
        return decision

    def decide_transfer(self, env_signals: Dict[str, Any]) -> str:
        rent_offer = float(env_signals.get('rent_offer_per_mu', 0.0))
        agri_dep = self._get_agri_dependence()

        # 兼业农户：根据农业依赖度+租金双向决策
        if rent_offer > 200 and agri_dep < 0.4:
            decision = 'rent_out'
        elif rent_offer < 100 and agri_dep > 0.7:
            decision = 'rent_in'
        else:
            decision = 'keep'
        self.last_action_transfer = decision
        return decision

    def _get_agri_dependence(self) -> float:
        """从 farmer.characteristics 取 agricultural_dependence，缺失时默认 0.5。"""
        characteristics = getattr(self.farmer, 'characteristics', None)
        if characteristics is None:
            return 0.5
        return float(getattr(characteristics, 'agricultural_dependence', 0.5))


# ============================================================================
# StrategicLearner — 策略型（家庭农场+新生代）
# ============================================================================

class StrategicLearner(BaseLearner):
    """
    适用 FarmerType: COMMERCIAL_FARMER, NEW_GENERATION_FARMER

    决策心理学基础：System 2 + 强化学习（Sutton & Barto, 2018）。
    维护一个 Q-table，状态由 (规模档, 价格档, 资本化档) 构成，
    动作分作物（2 选项）+ 流转（3 选项），共 6 个 action slots。

    每年年末调用 update_after_outcome() 用 realized reward 更新 Q-table。
    """

    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.1  # ε-greedy 探索

    # action 索引：0=grain, 1=cash | 2=keep, 3=rent_in, 4=rent_out, 5=abandon
    N_CROP_ACTIONS = 2
    N_TRANSFER_ACTIONS = 4
    TOTAL_ACTIONS = 6

    def __init__(self, farmer, buffer_size: int = 10):
        super().__init__(farmer, buffer_size=buffer_size)
        self.q_table: Dict[Tuple, np.ndarray] = {}

    def _get_state(self, env_signals: Dict[str, Any]) -> Tuple[int, int, int]:
        land = float(getattr(self.farmer, 'land_area_mu', None) or
                     getattr(self.farmer, 'land_area', 0))
        scale_tier = 0 if land < 5 else 1 if land < 30 else 2

        premium = float(env_signals.get('grain_premium', 0.0))
        premium_tier = 0 if premium < -0.05 else 1 if premium < 0.05 else 2

        cap_rate = float(env_signals.get('capitalization_rate', 0.0))
        cap_tier = 0 if cap_rate < 0.2 else 1 if cap_rate < 0.4 else 2

        return (scale_tier, premium_tier, cap_tier)

    def _ensure_q(self, state: Tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.TOTAL_ACTIONS, dtype=float)
        return self.q_table[state]

    def decide_crop(self, env_signals: Dict[str, Any]) -> str:
        state = self._get_state(env_signals)
        q = self._ensure_q(state)

        # ε-greedy
        if np_random_value(self.farmer) < self.EPSILON:
            decision = 'grain' if np_random_value(self.farmer) < 0.5 else 'cash_crop'
        else:
            decision = 'grain' if q[0] >= q[1] else 'cash_crop'

        self.last_state = state
        self.last_action_crop = decision
        return decision

    def decide_transfer(self, env_signals: Dict[str, Any]) -> str:
        state = self._get_state(env_signals)
        q = self._ensure_q(state)

        actions = ['keep', 'rent_in', 'rent_out', 'abandon']
        if np_random_value(self.farmer) < self.EPSILON:
            decision = actions[int(np_randint(self.farmer, self.N_TRANSFER_ACTIONS))]
        else:
            transfer_q = q[self.N_CROP_ACTIONS:self.N_CROP_ACTIONS + self.N_TRANSFER_ACTIONS]
            decision = actions[int(np.argmax(transfer_q))]
        self.last_action_transfer = decision
        return decision

    def update_after_outcome(
        self,
        reward: float,
        next_env_signals: Dict[str, Any],
    ) -> None:
        """
        每年末调用：根据 realized reward 更新 Q-table。

        Args:
            reward: 综合 reward（建议口径：净收入 / land_area_mu，归一化）
            next_env_signals: 当前年末的环境信号（用于估计 max Q(s')）
        """
        if self.last_state is None or self.last_action_crop is None:
            return

        crop_action_idx = 0 if self.last_action_crop == 'grain' else 1
        transfer_action_map = {'keep': 2, 'rent_in': 3, 'rent_out': 4, 'abandon': 5}
        transfer_action_idx = transfer_action_map.get(self.last_action_transfer or 'keep', 2)

        next_state = self._get_state(next_env_signals)
        q_next = self._ensure_q(next_state)
        q_curr = self._ensure_q(self.last_state)

        max_q_next_crop = float(np.max(q_next[:self.N_CROP_ACTIONS]))
        max_q_next_transfer = float(np.max(q_next[self.N_CROP_ACTIONS:]))

        # Bellman 更新（作物动作）
        old_crop = q_curr[crop_action_idx]
        q_curr[crop_action_idx] = old_crop + self.LEARNING_RATE * (
            reward + self.DISCOUNT_FACTOR * max_q_next_crop - old_crop
        )

        # Bellman 更新（流转动作）
        old_transfer = q_curr[transfer_action_idx]
        q_curr[transfer_action_idx] = old_transfer + self.LEARNING_RATE * (
            reward + self.DISCOUNT_FACTOR * max_q_next_transfer - old_transfer
        )

    def export_state(self) -> Dict[str, Any]:
        base = super().export_state()
        base['n_q_states'] = len(self.q_table)
        return base

    def export_sar(self, env_signals: Dict[str, Any] = None) -> Dict[str, Any]:
        """StrategicLearner 的 SAR 导出：包含 Q-table 统计和 next_state 估计。"""
        base = super().export_sar(env_signals)
        base['n_q_states'] = len(self.q_table)
        if env_signals:
            base['next_state'] = self._get_state(env_signals)
        # Q-table 摘要: 每个 state 的 max Q 均值和方差
        if self.q_table:
            max_qs = [float(np.max(q)) for q in self.q_table.values()]
            base['q_stats'] = {
                'n_states': len(self.q_table),
                'max_q_mean': float(np.mean(max_qs)),
                'max_q_std': float(np.std(max_qs)),
            }
        return base


# ============================================================================
# RentMaximizingLearner — 租金型（地主+城市迁移）
# ============================================================================

class RentMaximizingLearner(BaseLearner):
    """
    适用 FarmerType: URBAN_MIGRANT, ABSENTEE_LANDLORD

    决策心理学基础：纯财务最优化，脱离生产语境。

    决策机制：
        decide_crop: 始终返回占位 'grain'（因为不实际种植，由 transfer 决定）
        decide_transfer:
            - 若 rent_offer > rent_floor: rent_out
            - 若 rent_offer 太低且土地质量差: abandon
            - 否则: keep（持等出价）
        rent_floor 基于过去 5 年自身获得的最高出价 + 通胀
    """

    INFLATION_RATE = 0.03
    MIN_RENT_FLOOR = 30.0  # 元/亩，最低保留价

    def decide_crop(self, env_signals: Dict[str, Any]) -> str:
        # 不实际经营，决策由 transfer 主导
        decision = 'grain'  # 占位
        self.last_action_crop = decision
        return decision

    def decide_transfer(self, env_signals: Dict[str, Any]) -> str:
        rent_offer = float(env_signals.get('rent_offer_per_mu', 0.0))
        rent_floor = self._compute_rent_floor()

        soil_quality = float(env_signals.get('soil_quality', 0.5))

        if rent_offer >= rent_floor:
            decision = 'rent_out'
        elif soil_quality < 0.4 and rent_offer < self.MIN_RENT_FLOOR:
            # 土地差且无出价，撂荒
            decision = 'abandon'
        else:
            decision = 'keep'

        self.last_state = ('rent_max', round(rent_offer, 0), round(rent_floor, 0))
        self.last_action_transfer = decision
        return decision

    def _compute_rent_floor(self) -> float:
        """rent_floor = max(过去5年最高出价) × (1 + inflation)，无历史时用 MIN_RENT_FLOOR。"""
        recent = list(self.experience_buffer)[-5:]
        if not recent:
            return self.MIN_RENT_FLOOR
        max_recent = max((e.rent_offer_received for e in recent), default=0.0)
        if max_recent <= 0:
            return self.MIN_RENT_FLOOR
        return max_recent * (1.0 + self.INFLATION_RATE)


# ============================================================================
# Factory
# ============================================================================

class FarmerLearnerFactory:
    """
    根据 FarmerType 选择对应的 learner module。

    使用方式：
        from agents.learner_modules import FarmerLearnerFactory
        learner = FarmerLearnerFactory.create_for_farmer(farmer)
    """

    @staticmethod
    def create_for_farmer(farmer) -> BaseLearner:
        """根据 farmer 实例的 farmer_type 字段或 characteristics 返回对应 learner。"""
        ftype_name = FarmerLearnerFactory._resolve_type_name(farmer)
        return FarmerLearnerFactory._create_by_type_name(farmer, ftype_name)

    @staticmethod
    def _resolve_type_name(farmer) -> str:
        """容错地解析 farmer 类型字符串。"""
        ftype = getattr(farmer, 'farmer_type', None)
        if ftype is None and hasattr(farmer, 'characteristics'):
            characteristics = farmer.characteristics
            ftype = getattr(characteristics, 'farmer_type', None)
        if ftype is None:
            return 'UNKNOWN'
        if hasattr(ftype, 'value'):
            return str(ftype.value)
        if hasattr(ftype, 'name'):
            return str(ftype.name)
        return str(ftype)

    @staticmethod
    def _create_by_type_name(farmer, ftype_name: str) -> BaseLearner:
        # 大小写不敏感+包含匹配，兼容多种类型字符串
        upper = ftype_name.upper()

        if 'TRADITIONAL' in upper or 'SUBSISTENCE' in upper:
            return HabitualLearner(farmer)
        if any(tag in upper for tag in ('PART_TIME', 'PART-TIME', 'PARTTIME', 'WEEKEND', 'PLURIACTIVE')):
            return AdaptiveLearner(farmer)
        if 'COMMERCIAL' in upper or 'NEW_GEN' in upper or 'NEWGEN' in upper or 'NEW GENERATION' in upper:
            return StrategicLearner(farmer)
        if 'URBAN_MIGRANT' in upper or 'MIGRANT' in upper or 'ABSENTEE' in upper or 'LANDLORD' in upper:
            return RentMaximizingLearner(farmer)

        # 默认回落到 AdaptiveLearner（最通用的兼业型）
        return AdaptiveLearner(farmer)


# ============================================================================
# 便捷函数：构建 env_signals
# ============================================================================

def build_default_env_signals(farmer) -> Dict[str, Any]:
    """
    从 farmer.model.environment 构建标准 env_signals。

    设计原则：所有 learner 用同一份信号 schema，便于跨 learner 比较和审计。
    """
    env = getattr(farmer.model, 'environment', None)
    if env is None:
        return {
            'grain_premium': 0.0,
            'neighbor_grain_share': 0.5,
            'capitalization_rate': 0.0,
            'rent_offer_per_mu': 100.0,
            'off_farm_wage': 0.0,
            'soil_quality': 0.5,
            'subsidy_signal': 0.0,
        }

    grain_price = _safe_call(lambda: env.get_market_price('crop_price'), default=1.0)
    cash_price = _safe_call(lambda: env.get_market_price('cash_crop'), default=1.0)
    grain_premium = (grain_price - cash_price) / max(cash_price, 1e-6)

    wage_rate = _safe_call(lambda: env.get_market_price('wage_rate'), default=0.0)
    subsidy_signal = _safe_call(lambda: env.get_policy_param('grain_subsidies_rate'), default=0.0)

    cap_rate = float(getattr(env, 'annual_rent_capitalization_rate', 0.0) or 0.0)
    rent_offer = float(getattr(env, 'avg_rent_offer', 100.0) or 100.0)

    # 邻居种粮比例
    neighbor_grain_share = _compute_neighbor_grain_share(farmer)

    # 自身土地土壤质量
    soil_quality = _compute_own_soil_quality(farmer)

    return {
        'grain_premium': float(grain_premium),
        'neighbor_grain_share': float(neighbor_grain_share),
        'capitalization_rate': float(cap_rate),
        'rent_offer_per_mu': float(rent_offer),
        'off_farm_wage': float(wage_rate),
        'soil_quality': float(soil_quality),
        'subsidy_signal': float(subsidy_signal),
    }


def _safe_call(fn, default):
    try:
        v = fn()
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _compute_neighbor_grain_share(farmer) -> float:
    """从 farmer.social_network 抽取邻居最近的作物选择，计算种粮比例。"""
    network = getattr(farmer, 'social_network', None) or []
    if not network:
        return 0.5  # 中性先验

    model = getattr(farmer, 'model', None)
    if model is None:
        return 0.5

    grain_count = 0
    total = 0
    for neighbor_id in network:
        neighbor = None
        # 尝试多种获取方式
        if hasattr(model, 'get_agent_by_id'):
            try:
                neighbor = model.get_agent_by_id(neighbor_id)
            except Exception:
                neighbor = None
        if neighbor is None and hasattr(model, 'agent_registry'):
            neighbor = model.agent_registry.get(neighbor_id)
        if neighbor is None:
            continue

        history = getattr(neighbor, 'crop_choice_history', None) or []
        if not history:
            continue
        latest = history[-1]
        choice = latest.get('choice') if isinstance(latest, dict) else latest
        choice_str = str(getattr(choice, 'value', choice)).lower()
        total += 1
        if 'grain' in choice_str or '粮' in choice_str:
            grain_count += 1

    if total == 0:
        return 0.5
    return grain_count / total


def _compute_own_soil_quality(farmer) -> float:
    """从 land_patches 求平均 soil_quality。"""
    patches = getattr(farmer, 'land_patches', None) or []
    if not patches:
        return 0.5
    qualities = []
    for p in patches:
        q = getattr(p, 'soil_quality', None)
        if q is not None:
            try:
                qualities.append(float(q))
            except (TypeError, ValueError):
                continue
    if not qualities:
        return 0.5
    return float(np.mean(qualities))


# ============================================================================
# 公共 API
# ============================================================================

__all__ = [
    'FarmerExperience',
    'BaseLearner',
    'HabitualLearner',
    'AdaptiveLearner',
    'StrategicLearner',
    'RentMaximizingLearner',
    'FarmerLearnerFactory',
    'build_default_env_signals',
]
