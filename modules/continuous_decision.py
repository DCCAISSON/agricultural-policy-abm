# -*- coding: utf-8 -*-
"""
@theory 基于连续变量的决策函数模块
@critical 2025-12: 用连续变量替代离散标签

核心思想：
- 不再使用 if agent.type == "兼业II型" 这样的离散判断
- 而是基于连续特征变量的函数（logit、阈值函数等）
- 大幅提升模型真实性和连续性

参考文献：
- McFadden (1974) 离散选择的条件Logit模型
- Train (2009) 离散选择方法及模拟

政策敏感度矩阵（参考用户提供的表格）：
农户类型           价格补贴  种粮直补  流转补贴  社会化服务
传统生存型小农      ★★★★★   ★★★     ★        ★★★★
现代商业型农户      ★★★     ★★★★★   ★★★★    ★
兼业I型            ★★★★    ★★★★    ★★★     ★★
兼业II型           ★        ★★       ★★★★★   ★★★★★
"离土不离权"地主   ★        ★        ★★★★★   ★★★★
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set, Any, Union


@dataclass
class ContinuousDecisionParams:
    """连续决策模型参数"""
    # Logit模型温度参数（越大选择越随机）
    temperature: float = 1.0
    
    # 阈值函数的斜率参数
    threshold_slope: float = 5.0
    
    # 噪声水平
    noise_std: float = 0.1


class ContinuousDecisionFunctions:
    """
    基于连续变量的决策函数集合
    
    所有函数输入农户特征（连续变量），输出决策概率或效用值
    """
    
    @staticmethod
    def sigmoid(x: float, slope: float = 1.0, center: float = 0.0) -> float:
        """Sigmoid函数，用于将连续变量映射到[0,1]概率"""
        return 1.0 / (1.0 + np.exp(-slope * (x - center)))
    
    @staticmethod
    def logit_probability(utilities: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """
        Logit选择概率
        
        P(选择i) = exp(U_i / T) / Σ exp(U_j / T)
        """
        if not utilities:
            return {}
        
        # 防止数值溢出
        max_u = max(utilities.values())
        exp_utils = {k: np.exp((v - max_u) / temperature) for k, v in utilities.items()}
        total = sum(exp_utils.values())
        
        return {k: v / total for k, v in exp_utils.items()}
    
    # ==================== 政策敏感度计算 ====================
    
    @staticmethod
    def calculate_price_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        commercialization_rate: float,
        aging_rate: float
    ) -> float:
        """
        计算对粮食价格补贴的敏感度
        
        逻辑：
        - 农业依赖度高 → 敏感度高（主要收入来源）
        - 规模小 → 敏感度高（边际效用大）
        - 商品化率低 → 敏感度高（主要自用，价格影响粮食安全感）
        - 老龄化高 → 敏感度高（风险承受低）
        
        Returns: [0, 1] 敏感度
        """
        # 各因素权重
        w_agri = 0.35
        w_scale = 0.25
        w_comm = 0.20
        w_aging = 0.20
        
        # 农业依赖度正向影响
        s_agri = agricultural_dependence
        
        # 规模负向影响（小规模更敏感）
        s_scale = 1.0 - min(1.0, land_scale / 50.0)
        
        # 商品化率负向影响（低商品化更敏感）
        s_comm = 1.0 - commercialization_rate
        
        # 老龄化正向影响
        s_aging = aging_rate
        
        sensitivity = w_agri * s_agri + w_scale * s_scale + w_comm * s_comm + w_aging * s_aging
        return np.clip(sensitivity, 0.0, 1.0)
    
    @staticmethod
    def calculate_direct_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        commercialization_rate: float,
        technology_adoption: float
    ) -> float:
        """
        计算对种粮直接补贴的敏感度
        
        逻辑：
        - 规模大 → 敏感度高（补贴总额大）
        - 商品化率高 → 敏感度高（专业化经营）
        - 技术采纳高 → 敏感度高（现代农业经营者）
        - 农业依赖度中等时最敏感（既专业又依赖）
        """
        w_scale = 0.35
        w_comm = 0.30
        w_tech = 0.20
        w_agri = 0.15
        
        # 规模正向影响（大规模更敏感）
        s_scale = min(1.0, land_scale / 50.0)
        
        # 商品化率正向影响
        s_comm = commercialization_rate
        
        # 技术采纳正向影响
        s_tech = technology_adoption
        
        # 农业依赖度倒U型（中等依赖度最敏感）
        s_agri = 1.0 - abs(agricultural_dependence - 0.6) * 2
        s_agri = max(0.0, s_agri)
        
        sensitivity = w_scale * s_scale + w_comm * s_comm + w_tech * s_tech + w_agri * s_agri
        return np.clip(sensitivity, 0.0, 1.0)
    
    @staticmethod
    def calculate_rental_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        land_leased_out_ratio: float,
        land_leased_in_ratio: float
    ) -> float:
        """
        计算对流转租金补贴的敏感度
        
        逻辑：
        - 已流出土地多 → 敏感度高（租金是收入来源）
        - 农业依赖度低 → 敏感度高（已退出农业）
        - 已流入土地多 → 敏感度高（租金是成本）
        """
        w_out = 0.40
        w_agri = 0.30
        w_in = 0.30
        
        # 流出比例正向影响
        s_out = land_leased_out_ratio
        
        # 农业依赖度负向影响（低依赖度更敏感）
        s_agri = 1.0 - agricultural_dependence
        
        # 流入比例正向影响（租入者关心租金成本）
        s_in = land_leased_in_ratio
        
        sensitivity = w_out * s_out + w_agri * s_agri + w_in * s_in
        return np.clip(sensitivity, 0.0, 1.0)
    
    # ==================== 核心决策函数 ====================
    
    @staticmethod
    def land_transfer_out_probability(
        agricultural_dependence: float,
        land_scale: float,
        aging_rate: float,
        off_farm_income_ratio: float,
        rental_price: float,
        reservation_price: float,
        social_capital: float = 5.0,
        age: int = 50,
        loss_aversion: float = 1.95,  # @updated 2025-12-28: Brown et al. (2024) AER meta-analysis
        years_owned: int = 10
    ) -> float:
        """
        土地流出概率（基于前景理论的有限理性决策）
        
        @enhanced 2025-12-28: 添加前景理论核心要素
        
        @literature:
        - Kahneman & Tversky (1979): 前景理论，损失厌恶λ≈2.5
        - Thaler (1980): 禀赋效应，拥有物品后估值上升
        - Samuelson & Zeckhauser (1988): 现状偏好，倾向维持现状
        - MDPI (2023): 年龄U型非线性
        
        新增机制：
        1. 损失厌恶：失去土地的"痛苦"是获得租金"快乐"的2.5倍
        2. 禀赋效应：拥有土地越久，越不愿流出
        3. 现状偏好：默认维持现状，需要足够激励才会改变
        
        Returns: [0, 1] 流出概率
        """
        # === 1. 基础效用 ===
        u_agri = -2.0 * agricultural_dependence
        u_scale = -0.02 * land_scale
        
        # 年龄U型非线性
        age_centered = age - 60
        if age < 60:
            u_age = -0.01 * age_centered
        else:
            u_age = 0.05 * age_centered
        
        u_aging = 0.8 * aging_rate
        u_off_farm = 1.0 * off_farm_income_ratio
        
        # 租金吸引力
        price_ratio = (rental_price - reservation_price) / max(reservation_price, 100)
        u_price = 0.5 * price_ratio
        
        u_social = -0.05 * social_capital
        
        # === 2. 前景理论：损失厌恶 ===
        # @new 2025-12-28: 流出土地被感知为"损失"
        # 损失厌恶使得农户要求更高的租金补偿
        # 只有租金显著高于保留价时才愿意流出
        if rental_price < reservation_price * 1.2:
            # 租金不够高，损失厌恶起作用
            u_loss_aversion = -0.8 * (loss_aversion - 1)  # λ=2.5时约-1.2
        else:
            u_loss_aversion = 0
        
        # === 3. 禀赋效应：拥有越久越不愿放弃 ===
        # @new 2025-12-28: 土地不仅是生产资料，还有情感价值
        endowment_effect = -0.03 * min(years_owned, 30)  # 最多30年影响
        
        # === 4. 现状偏好：默认维持现状 ===
        # @new 2025-12-28: 需要额外激励才能改变现状
        status_quo_bias = -0.5  # 固定负效用，偏好维持现状
        
        # === 5. 总效用 ===
        utility = (u_agri + u_scale + u_age + u_aging + u_off_farm + 
                   u_price + u_social + u_loss_aversion + endowment_effect + status_quo_bias)
        
        prob = ContinuousDecisionFunctions.sigmoid(utility, slope=1.0, center=0.0)
        
        return np.clip(prob, 0.05, 0.95)
    
    @staticmethod
    def land_transfer_in_probability(
        agricultural_dependence: float,
        land_scale: float,
        capital: float,
        technology_adoption: float,
        target_scale: float,
        rental_price: float,
        expected_profit_per_mu: float,
        age: int = 50,
        optimal_scale: float = 50.0
    ) -> float:
        """
        土地流入概率（扩大经营规模的意愿）
        
        @enhanced 2025-12-28: 根据学术文献优化
        
        @literature:
        - MDPI (2023): <60岁流入率显著>流出率
        - ResearchGate: 年龄与流入呈倒U型关系（先促进后抑制）
        - crnews.net: 专业大户最优规模30-100亩
        - 中国农业统计: 东部40-48亩，中部50-75亩，东北88-117亩
        
        Args:
            optimal_scale: 最优经营规模（亩），基于地区设定
        
        Returns: [0, 1] 流入概率
        """
        # 农业依赖度正向影响（依赖农业则想扩大）
        u_agri = 1.5 * agricultural_dependence
        
        # @new 2025-12-28: 最优规模约束
        # 已达到或超过最优规模时，流入意愿显著降低
        # 这解释了为什么农户不会无限扩张
        if land_scale >= optimal_scale:
            # 超过最优规模：流入意愿大幅降低
            scale_factor = -1.5 * (land_scale - optimal_scale) / optimal_scale
        elif land_scale >= optimal_scale * 0.7:
            # 接近最优规模：流入意愿开始降低
            scale_factor = 0.3
        else:
            # 远低于最优规模：有扩张动力
            scale_gap = (optimal_scale - land_scale) / optimal_scale
            scale_factor = 1.2 * scale_gap
        
        u_scale = scale_factor
        
        # @enhanced 2025-12-28: 年龄倒U型非线性影响
        # 35-55岁是流入高峰期，<35岁可能更关注非农，>55岁体力下降
        age_peak = 45  # 流入高峰年龄
        age_centered = age - age_peak
        # 倒U型：二次函数形式
        u_age = -0.002 * (age_centered ** 2)  # 偏离峰值越远，意愿越低
        
        # 60岁后额外抑制效应
        if age >= 60:
            u_age -= 0.5 * (age - 60) / 10
        
        # 资本充足度
        capital_adequacy = min(1.0, capital / (land_scale * rental_price + 10000))
        u_capital = 0.8 * capital_adequacy
        
        # 技术采纳能力
        u_tech = 0.5 * technology_adoption
        
        # 预期利润
        profit_attractiveness = expected_profit_per_mu / 500  # 500元/亩作为参考
        u_profit = 0.6 * min(1.0, max(-1.0, profit_attractiveness))
        
        utility = u_agri + u_scale + u_age + u_capital + u_tech + u_profit
        prob = ContinuousDecisionFunctions.sigmoid(utility, slope=0.8, center=1.0)
        
        return np.clip(prob, 0.02, 0.90)
    
    @staticmethod
    def crop_choice_grain_preference(
        agricultural_dependence: float,
        land_scale: float,
        risk_tolerance: float,
        price_subsidy_sensitivity: float,
        grain_expected_return: float,
        cash_crop_expected_return: float,
        grain_risk: float,
        cash_crop_risk: float
    ) -> float:
        """
        粮食作物选择偏好（基于连续变量）
        
        Returns: [0, 1] 选择粮食的概率
        """
        # 风险调整收益
        grain_utility = grain_expected_return - (1 - risk_tolerance) * grain_risk
        cash_utility = cash_crop_expected_return - (1 - risk_tolerance) * cash_crop_risk
        
        # 补贴敏感度调整（对粮食补贴敏感则偏好粮食）
        grain_utility += price_subsidy_sensitivity * 0.3 * grain_expected_return
        
        # 农业依赖度高→风险厌恶→偏好稳定的粮食
        if agricultural_dependence > 0.7:
            grain_utility += 0.1 * grain_expected_return
        
        # 规模大→可能偏好经济作物（追求利润）
        if land_scale > 30:
            cash_utility += 0.05 * cash_crop_expected_return
        
        # Logit选择
        utilities = {'grain': grain_utility, 'cash': cash_utility}
        probs = ContinuousDecisionFunctions.logit_probability(utilities, temperature=0.5)
        
        return probs.get('grain', 0.5)
    
    @staticmethod
    def subsidy_response_adjustment(
        base_decision: float,
        subsidy_change: float,
        sensitivity: float,
        response_type: str = 'linear'
    ) -> float:
        """
        补贴变化对决策的调整
        
        Args:
            base_decision: 基础决策值
            subsidy_change: 补贴变化率（正为增加）
            sensitivity: 敏感度 [0, 1]
            response_type: 响应类型 'linear' 或 'diminishing'
        
        Returns: 调整后的决策值
        """
        if response_type == 'linear':
            adjustment = sensitivity * subsidy_change
        elif response_type == 'diminishing':
            # 边际递减响应
            adjustment = sensitivity * np.tanh(subsidy_change)
        else:
            adjustment = 0
        
        return np.clip(base_decision + adjustment, 0.0, 1.0)


# ==================== 前景理论参数的类型默认值 ====================

PROSPECT_THEORY_DEFAULTS = {
    # 基于农业依赖度区间的默认值
    'high_dependence': {  # 农业依赖度 > 0.7
        'loss_aversion_lambda': 2.8,
        'value_curvature_alpha': 0.50,
        'probability_weighting_gamma': 0.65,
    },
    'medium_dependence': {  # 0.3 < 农业依赖度 <= 0.7
        'loss_aversion_lambda': 2.5,
        'value_curvature_alpha': 0.55,
        'probability_weighting_gamma': 0.70,
    },
    'low_dependence': {  # 农业依赖度 <= 0.3
        'loss_aversion_lambda': 2.2,
        'value_curvature_alpha': 0.60,
        'probability_weighting_gamma': 0.75,
    },
}


def get_prospect_params_by_dependence(agricultural_dependence: float) -> Dict[str, float]:
    """
    根据农业依赖度返回前景理论参数默认值
    
    简化：不再为每个农户单独抽样，而是按类型给默认值
    """
    if agricultural_dependence > 0.7:
        return PROSPECT_THEORY_DEFAULTS['high_dependence']
    elif agricultural_dependence > 0.3:
        return PROSPECT_THEORY_DEFAULTS['medium_dependence']
    else:
        return PROSPECT_THEORY_DEFAULTS['low_dependence']


# ==================== 简化的社会网络关系判断 ====================

def check_acquaintance_simple(
    agent1_network: Set[int],
    agent2_id: int,
    agent2_network: Optional[Set[int]] = None,
    max_distance: int = 2
) -> Tuple[bool, Union[int, float]]:
    """
    简化的熟人关系判断
    
    原来的规则太复杂，现在简化为：
    - 距离1：直接邻居（在对方网络中）
    - 距离2：有共同邻居
    
    Args:
        agent1_network: agent1的直接邻居集合
        agent2_id: agent2的ID
        agent2_network: agent2的直接邻居集合（可选）
        max_distance: 最大网络距离
    
    Returns:
        (is_acquaintance, distance)
    """
    # 距离1：直接邻居
    if agent2_id in agent1_network:
        return True, 1
    
    # 距离2：共同邻居（如果提供了agent2的网络）
    if max_distance >= 2 and agent2_network is not None:
        common_neighbors = agent1_network.intersection(agent2_network)
        if len(common_neighbors) > 0:
            return True, 2
    
    return False, float('inf')


def calculate_acquaintance_discount(distance: int, base_discount: float = 0.15) -> float:
    """
    计算熟人折扣率
    
    距离1：折扣15%
    距离2：折扣8%
    非熟人：无折扣
    """
    if distance == 1:
        return base_discount
    elif distance == 2:
        return base_discount * 0.5
    else:
        return 0.0


# ==================== Stone-Geary 效用函数集成 ====================
# @new 2025-12-19: Priority 0 修复 - 决策与评估对齐

@dataclass
class StoneGearyDecisionConfig:
    """
    Stone-Geary 效用函数决策配置
    
    @theory 生存约束下的效用最大化
    
    核心公式:
    U = Σ α_i * log(c_i - γ_i)
    
    其中:
    - c_i: 实际消费量
    - γ_i: 生存必需量（Stone-Geary 门槛）
    - α_i: 边际效用权重
    
    Decision_Utility = Expected_Profit - Risk_Penalty - Subsistence_Penalty
    """
    # 生存必需量（元/年）
    gamma_food: float = 5000.0       # 食品最低消费（口粮价值）
    gamma_housing: float = 3000.0    # 住房最低支出
    gamma_medical: float = 2000.0    # 医疗最低支出
    gamma_education: float = 1000.0  # 教育（子女）最低支出
    
    # 效用函数权重
    alpha_food: float = 0.40
    alpha_housing: float = 0.25
    alpha_medical: float = 0.20
    alpha_education: float = 0.15
    
    # 风险厌恶参数
    risk_aversion_coefficient: float = 2.0
    
    # 生存惩罚指数（越高惩罚越陡峭）
    survival_penalty_exponent: float = 3.0
    
    @property
    def total_subsistence(self) -> float:
        """总生存必需消费"""
        return self.gamma_food + self.gamma_housing + self.gamma_medical + self.gamma_education


def calculate_stone_geary_decision_utility(
    expected_income: float,
    income_variance: float,
    config: Optional[StoneGearyDecisionConfig] = None,
    household_size: int = 1
) -> Tuple[float, Dict[str, float]]:
    """
    计算基于 Stone-Geary 的决策效用
    
    @theory 决策-评估对齐原则（Priority 0 核心修复）
    
    Args:
        expected_income: 预期年收入（元）
        income_variance: 收入方差
        config: Stone-Geary 配置参数
        household_size: 家庭人口
    
    Returns:
        (decision_utility, breakdown_dict)
    """
    if config is None:
        config = StoneGearyDecisionConfig()
    
    # 按家庭规模调整生存需求
    adjusted_subsistence = config.total_subsistence * household_size
    
    # 1. 计算预期效用（基础项）
    base_utility = expected_income
    
    # 2. 计算风险惩罚（CRRA形式）
    # Penalty_risk = 0.5 * λ * σ² / μ
    if expected_income > 0:
        cv = np.sqrt(income_variance) / expected_income  # 变异系数
        risk_penalty = 0.5 * config.risk_aversion_coefficient * (cv ** 2) * expected_income
    else:
        risk_penalty = 0.0
    
    # 3. 计算生存惩罚（Stone-Geary 核心）
    if expected_income >= adjusted_subsistence:
        # 收入充足，无生存惩罚
        subsistence_penalty = 0.0
        survival_ratio = 1.0
    else:
        # 收入不足，施加惩罚
        # 惩罚公式：penalty = K * (1 - income/subsistence)^n
        survival_ratio = expected_income / adjusted_subsistence
        K = adjusted_subsistence * 2  # 惩罚系数
        subsistence_penalty = K * ((1 - survival_ratio) ** config.survival_penalty_exponent)
    
    # 4. 汇总决策效用
    decision_utility = base_utility - risk_penalty - subsistence_penalty
    
    breakdown = {
        'base_utility': base_utility,
        'risk_penalty': risk_penalty,
        'subsistence_penalty': subsistence_penalty,
        'survival_ratio': survival_ratio,
        'adjusted_subsistence': adjusted_subsistence,
        'final_utility': decision_utility
    }
    
    return decision_utility, breakdown


def create_decision_evaluator_bridge(
    stone_geary_config: Optional[StoneGearyDecisionConfig] = None
):
    """
    创建决策-评估桥接器
    
    @theory Priority 0 修复 - 确保决策函数与事后评估使用相同效用框架
    
    Returns:
        bridge_function: 可在决策和评估阶段调用的统一效用计算函数
    """
    config = stone_geary_config or StoneGearyDecisionConfig()
    
    def unified_utility_calculator(
        income: float,
        variance: float = 0.0,
        household_size: int = 1,
        stage: str = 'decision'  # 'decision' or 'evaluation'
    ) -> Tuple[float, Dict]:
        """
        统一效用计算器
        
        stage='decision': 用于事前决策（使用预期值和方差）
        stage='evaluation': 用于事后评估（使用实际值，方差=0）
        """
        if stage == 'evaluation':
            # 事后评估：使用实际收入，无方差
            variance = 0.0
        
        return calculate_stone_geary_decision_utility(
            expected_income=income,
            income_variance=variance,
            config=config,
            household_size=household_size
        )
    
    return unified_utility_calculator
