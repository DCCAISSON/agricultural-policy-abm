# -*- coding: utf-8 -*-
"""
@theory
MainModel - 基于最终版设计规范的核心模型
实现严格的年度决策周期和关键输出指标收集
@refactor: 简化MainModel，将初始化逻辑迁移到ModelBuilder

@new 2025-12-04: 
- 集成墓地统计系统（避免幸存者偏差）
- 添加服务市场统计数据收集
- 集成异质性预期机制
"""

from mesa import Model
from core.fixed_rule_runtime import (
    action_income_enabled,
    evaluate_fixed_rule_operators,
    fixed_rule_enabled,
    submit_action_income_bids,
    preproduction_cash_enabled,
    prepare_preproduction_cash,
    production_area,
    production_cash_quote,
)
from core.income_utils import get_farmer_net_income, is_professional_large_operator
try:
    from mesa.time import RandomActivation
except ImportError:
    # Mesa 3.0+ compatibility: define a complete RandomActivation
    # Mesa 3.0 removed mesa.time module, using AgentSet instead
    class RandomActivation:
        """Mesa 3.x compatible RandomActivation scheduler."""
        def __init__(self, model):
            self.model = model
            self._agents = []  # Internal list
            self.steps = 0     # Step counter (required by many components)
            self.time = 0      # Time counter
        
        @property
        def agents(self):
            """Property to access agents list (required by unified_indicators, etc)."""
            return self._agents
        
        def add(self, agent):
            """Add an agent to the scheduler."""
            self._agents.append(agent)
            
        def remove(self, agent):
            """Remove an agent from the scheduler."""
            if agent in self._agents:
                self._agents.remove(agent)
                
        def step(self):
            """Execute step for all agents in random order."""
            rng_shuffle(self, self._agents)
            for agent in self._agents:
                agent.step()
            self.steps += 1
            self.time += 1
        
        def get_agent_count(self):
            """Return count of agents."""
            return len(self._agents)

from mesa.space import MultiGrid
import numpy as np

from core.reproducibility import (
    RandomController,
    build_numpy_rng,
    np_choice,
    np_lognormal,
    np_randint,
    np_uniform,
    rng_random,
    rng_randint,
    rng_shuffle,
    rng_uniform,
)
from modules.land_clearing_house import LandClearingHouse
from modules.land_patch import LandPatch
# @new 2025-12-04: 导入新模块
from modules.ecological_feedback import EcologicalFeedbackMechanism  # @new 2025-01-02: 生态反馈集成
from modules.graveyard import Graveyard, ExitReason
# @new 2025-01-02: P1-P3 Nature级模块集成
from modules.climate_module import ClimateManager  # P3: 空间气候风险
from modules.environmental_accountant import EnvironmentalAccountant  # P2: 环境核算
# @new 2025-12-22: 集成市场协调器（FIX_REPORT_20251222改进）
from core.market_coordinator import MarketCoordinator

class MainModel(Model):
    def __init__(self, config, environment, seed=None):
        """
        @refactor: __init__ 大幅简化，只接收必要对象
        """
        super().__init__(seed=seed)
        effective_seed = seed if seed is not None else RandomController.instance().master_seed
        self.np_random = build_numpy_rng(effective_seed)
        if environment is not None:
            environment.model = self
            environment.random = getattr(self, "random", None)
            environment.np_random = self.np_random
        
        self.config = config
        self.environment = environment # @refactor: 接收环境对象
        self.running = True
        self.step_count = 0
        self.simulation_base_year = self._resolve_simulation_base_year()
        if self._fixed_rule_enabled():
            self.v4_8_fixed_rule_trace = []
            self.v4_8_market_stage_trace = []

        # 核心组件
        # @fix 2024-11-20: 使用RandomActivation替代BaseScheduler，消除先发优势
        # 专家审阅意见：BaseScheduler严格按顺序执行，带来严重的先发优势
        # 例如ID为0的农户总是先租地，ID为100的农户总是捡剩下的
        # RandomActivation每步随机打乱agent顺序，消除人工造成的先发优势
        # 注意：使用seed保证可复现性
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(
            width=config['scale']['grid_width'],
            height=config['scale']['grid_height'],
            torus=True
        )
        self.land_clearing_house = LandClearingHouse()
        
        # @new 2025-12: 撂荒记录
        self.abandonment_records = []
        
        # @new 2025-12-17: 年龄退出记录
        self.age_exit_records = []
        
        # 智能体列表 - 由ModelBuilder填充
        self.farmers = []
        self.agricultural_enterprises = []
        self.government = None
        self.village_collective = None
        self.village_collectives = []
        self.village_collective_network = {
            'num_nodes': 0,
            'edges': [],
            'density': 0.0,
        }
        self.grain_reserve_bureau = None
        self.land_patches = []
        self.land_tenure_version = 0
        self._operated_area_by_village_cache_version = None
        self._operated_area_by_village_cache = {}
        
        # @fix 2024-11-20: 添加agent_registry字典，优化get_agent_by_id性能（O(N)->O(1)）
        # 专家审阅意见：get_agent_by_id是O(N)操作，频繁调用会导致性能问题
        self.agent_registry = {}  # {agent_id: agent}
        
        # @new 2025-11-29: 劳动力市场模块
        # 理论：实现农村劳动力市场的完整机制（大户雇散户、企业雇农户）
        self.labor_market = None  # 由ModelBuilder初始化
        
        # @new 2025-12: 农业社会化服务市场
        # @theory 服务规模经济 - "隐性的农业革命"
        # 核心洞见：土地没有流转，但服务在流转
        # 通过托管服务实现规模经济，无需土地产权变动
        self.service_market = None  # 由ModelBuilder初始化
        self.service_providers = []  # 服务提供商列表
        
        # @new 2025-12-22: 市场协调器 - 解决市场时序耦合问题（FIX_REPORT改进）
        # @theory 跨市场资本约束与蛛网模型阻尼
        # 1. 资本预留机制：参与土地竞价前冻结部分资金
        # 2. 预期形成联动：出价考虑劳动力可得性和服务价格
        # 3. 蛛网阻尼：适应性预期混合，防止市场震荡发散
        self.market_coordinator = None  # 由ModelBuilder初始化
        
        # @new 2025-12-04: 墓地统计系统 - 避免幸存者偏差
        # 理论：仅分析存活智能体会产生严重幸存者偏差
        # 墓地系统记录所有退出智能体的完整信息，支持事后分析
        self.graveyard = Graveyard()
        
        # @new 2026-01-07: 人口动态历史记录
        # 记录每年的人口变动统计（退出/进入/土地重分配）
        self.population_dynamics_history = []
        self.land_area_history = []
        
        # @new 2025-01-02: 生态反馈机制 - CHANS建模核心组件
        # 理论：Nature级论文需要人类-自然耦合系统的建模
        # 实现：土壤质量动态退化、临界点效应、休耕恢复机制
        self.ecological_feedback_mechanism = EcologicalFeedbackMechanism(config)
        
        # @new 2025-01-02: P1-P3 Nature级模块初始化
        # P2: 环境核算系统 (Li et al., 2020, PNAS)
        self.environmental_accountant = EnvironmentalAccountant(self)
        
        # P3: 空间气候风险管理器 (CLIMADA / spatial-climate-ABM)
        climate_config = config.get('environment', {}).get('climate', {})
        if climate_config.get('spatial_hazards_enabled', True):
            grid_width = config['scale']['grid_width']
            grid_height = config['scale']['grid_height']
            self.climate_manager = ClimateManager(grid_width, grid_height, climate_config)
        else:
            self.climate_manager = None
        
        # @new 2025-12-04: 服务市场统计
        self.service_market_stats = {
            'total_contracts': 0,
            'total_service_area': 0,
            'total_service_fee': 0,
            'contracts_by_type': {},
            'provider_utilization': [],
            'provider_profit_total': 0.0,
            'provider_profit_mean': 0.0,
            'service_managed_retained_grain_area': 0.0,
            'service_managed_retained_grain_contracts': 0,
        }
        
        # @new 2025-12-27: 土地流转率年度追踪（实现max_annual_transfer_rate约束）
        # 解决问题：配置中的流转率上限参数未被实际使用，导致流转率爆发式增长
        # 机制：每step清零annual，但检查是否超过年度上限后才允许新增流转
        transfer_cfg = (
            config.get('agents', {})
            .get('farmer', {})
            .get('land_transfer', {})
        )
        if not transfer_cfg:
            transfer_cfg = config.get('farmer', {}).get('land_transfer', {})
        self.transfer_rate_control = {
            'annual_new_transferred_area': 0.0,      # 当年新增流转面积
            'cumulative_transferred_area': 0.0,      # 累计流转面积
            'max_annual_rate': transfer_cfg.get('max_annual_transfer_rate', 0.05),  # 年度上限5%
            'priced_auction_reserve_share': transfer_cfg.get('priced_auction_reserve_share', 0.15),  # 竞拍预留
            'cumulative_ceiling': transfer_cfg.get('cumulative_transfer_ceiling', 0.70),  # 硬上限兜底
            'endogenous_soft_floor': transfer_cfg.get('endogenous_soft_floor', 0.42),
            'service_capacity_bonus': transfer_cfg.get('service_capacity_bonus', 0.04),
            'profitability_capacity_bonus': transfer_cfg.get('profitability_capacity_bonus', 0.09),
            'aging_capacity_bonus': transfer_cfg.get('aging_capacity_bonus', 0.02),
            'professional_capacity_bonus': transfer_cfg.get(
                'professionalization_capacity_bonus',
                transfer_cfg.get('professional_capacity_bonus', 0.08),
            ),
            'village_coordination_bonus': transfer_cfg.get(
                'village_facilitation_bonus',
                transfer_cfg.get('village_coordination_bonus', 0.04),
            ),
            'policy_support_capacity_bonus': transfer_cfg.get('policy_support_capacity_bonus', 0.0),
            'capacity_signal_curvature': transfer_cfg.get('capacity_signal_curvature', 1.20),
            'ceiling_unlock_share': transfer_cfg.get('ceiling_unlock_share', 0.22),
            'unlock_signal_curvature': transfer_cfg.get('unlock_signal_curvature', 1.0),
            'policy_unlock_bonus': transfer_cfg.get('policy_unlock_bonus', 0.10),
            'service_unlock_bonus': transfer_cfg.get('service_unlock_bonus', 0.04),
            'village_unlock_bonus': transfer_cfg.get('village_unlock_bonus', 0.04),
            'professional_unlock_bonus': transfer_cfg.get('professional_unlock_bonus', 0.03),
            'soft_throttle_min_acceptance': transfer_cfg.get('soft_throttle_min_acceptance', 0.08),
            'collective_overflow_base_acceptance': transfer_cfg.get('collective_overflow_base_acceptance', 0.22),
            'cross_village_overflow_acceptance': transfer_cfg.get('cross_village_overflow_acceptance', 0.88),
            'profit_reference_income': transfer_cfg.get('profit_reference_income', 80000),
            'operator_share_reference': transfer_cfg.get('operator_share_reference', 0.14),
            'service_reference_share': transfer_cfg.get('service_reference_share', 0.75),
            'aging_reference_share': transfer_cfg.get('aging_reference_share', 0.55),
            'policy_reference_intensity': transfer_cfg.get('policy_reference_intensity', 0.20),
            'effective_cumulative_ceiling': transfer_cfg.get('endogenous_soft_floor', 0.42),
            'effective_cumulative_ceiling_upper_bound': transfer_cfg.get('cumulative_transfer_ceiling', 0.70),
            'transfer_capacity_signal': 0.0,
            'capacity_unlock_signal': 0.0,
            'village_governance_signal': 0.0,
            'village_congestion_signal': 0.0,
            'service_slack_signal': 1.0,
            'soft_ceiling_acceptance_prob': 1.0,
            'baseline_total_land_area': None,        # 初始总面积（固定基准）
            'observed_total_land_area': 0.0,         # 每年观测面积
            'observed_current_transferred_area': 0.0,
            'land_area_drift': 0.0,                  # 观测与基准差值
            'total_land_area': 0.0,                  # 总土地面积（用于约束基准）
        }
        
        # 数据收集器 - 由ModelBuilder设置
        self.datacollector = None

    def _resolve_simulation_base_year(self) -> int:
        """Resolve the single simulation year anchor shared by runtime modules."""
        config = self.config if isinstance(self.config, dict) else {}
        population_cfg = config.get('population_dynamics', {}) if isinstance(config, dict) else {}
        macro_anchor_cfg = population_cfg.get('macro_anchor', {}) if isinstance(population_cfg, dict) else {}
        base_year = macro_anchor_cfg.get('base_year')
        if base_year is None:
            base_year = config.get('experiment', {}).get('base_year')
        try:
            return int(base_year or 2010)
        except (TypeError, ValueError):
            return 2010

    def _refresh_total_land_area(self):
        """刷新土地总面积基准，确保年度约束使用固定基准。"""
        if not hasattr(self, 'transfer_rate_control'):
            return
        if not self.land_patches:
            return

        control = self.transfer_rate_control
        observed = sum(getattr(patch, 'area', 0) for patch in self.land_patches)
        baseline = control.get('baseline_total_land_area')
        if baseline is None:
            baseline = getattr(self, 'initial_total_land_area', None) or observed
            control['baseline_total_land_area'] = baseline

        control['observed_total_land_area'] = observed
        control['land_area_drift'] = observed - baseline
        control['total_land_area'] = baseline

        if hasattr(self, 'land_area_history'):
            self.land_area_history.append({
                'step': self.step_count,
                'observed_total_land_area': observed,
                'baseline_total_land_area': baseline,
                'drift': control['land_area_drift'],
            })

    def invalidate_land_tenure_caches(self):
        """Invalidate caches that depend on current lease/operator assignments."""
        self.land_tenure_version = int(getattr(self, 'land_tenure_version', 0) or 0) + 1
        self._operated_area_by_village_cache_version = None
        self._operated_area_by_village_cache = {}

    def _reset_annual_capitalization_accounting(self):
        """Reset annual capitalization bookkeeping before a new simulation year starts."""
        environment = getattr(self, 'environment', None)
        if environment is None:
            return

        environment.annual_capitalization_loss = 0.0
        environment.annual_raw_basic_subsidy = 0.0

    def _get_village_collective_list(self):
        """Return village collectives as a normalized list."""
        villages = getattr(self, 'village_collectives', None)
        if villages:
            return [village for village in villages if village is not None]

        village_collective = getattr(self, 'village_collective', None)
        if village_collective is None:
            return []
        if isinstance(village_collective, (list, tuple, set)):
            return [village for village in village_collective if village is not None]
        return [village_collective]

    def get_village_count(self):
        return len(self._get_village_collective_list())

    def get_average_registered_farmers_per_village(self):
        villages = self._get_village_collective_list()
        if not villages:
            return 0.0
        return sum(len(getattr(village, 'registered_farmers', [])) for village in villages) / len(villages)

    def get_village_network_density(self):
        network = getattr(self, 'village_collective_network', {}) or {}
        return float(network.get('density', 0.0) or 0.0)

    def get_village_facilitated_transfer_area(self):
        return sum(
            float(getattr(getattr(village, 'economy_state', None), 'annual_transferred_area', 0.0) or 0.0)
            for village in self._get_village_collective_list()
        )

    def get_village_governance_capacity(self):
        villages = self._get_village_collective_list()
        if not villages:
            return 0.0

        signals = []
        for village in villages:
            if hasattr(village, 'get_governance_capacity_signal'):
                try:
                    signals.append(float(village.get_governance_capacity_signal()))
                    continue
                except Exception:
                    pass

            infrastructure = max(0.0, min(1.0, float(getattr(village, 'infrastructure_condition', 1.0) or 1.0)))
            custody_land = max(0.0, float(getattr(village, 'custody_land_area', 0.0) or 0.0))
            total_area = max(1.0, float(getattr(village, 'total_village_area', 1.0) or 1.0))
            max_custody_ratio = max(0.05, float(getattr(village, 'max_custody_ratio', 0.4) or 0.4))
            custody_pressure = min(1.0, (custody_land / total_area) / max_custody_ratio)
            collapse_penalty = 0.4 if bool(getattr(village, 'is_collapsed', False)) else 0.0
            signal = 0.55 * infrastructure + 0.45 * (1.0 - custody_pressure) - collapse_penalty
            signals.append(max(0.0, min(1.0, signal)))

        return float(np.mean(signals)) if signals else 0.0

    def get_village_transfer_congestion(self):
        villages = self._get_village_collective_list()
        if not villages:
            return 0.0

        congestion_values = []
        for village in villages:
            if hasattr(village, 'get_transfer_congestion_signal'):
                try:
                    signal = float(village.get_transfer_congestion_signal())
                    congestion_values.append(1.0 - max(0.0, min(1.0, signal)))
                    continue
                except Exception:
                    pass

            custody_land = max(0.0, float(getattr(village, 'custody_land_area', 0.0) or 0.0))
            total_area = max(1.0, float(getattr(village, 'total_village_area', 1.0) or 1.0))
            max_custody_ratio = max(0.05, float(getattr(village, 'max_custody_ratio', 0.4) or 0.4))
            congestion = min(1.0, (custody_land / total_area) / max_custody_ratio)
            if bool(getattr(village, 'is_collapsed', False)):
                congestion = max(congestion, 0.9)
            congestion_values.append(congestion)

        return float(np.mean(congestion_values)) if congestion_values else 0.0

    def get_primary_village_for_patch(self, patch):
        """Assign a land patch to the nearest village collective."""
        villages = self._get_village_collective_list()
        if not villages:
            return None

        coordinates = getattr(patch, 'coordinates', None)
        if coordinates is None:
            owner = getattr(patch, 'owner', None)
            if owner is not None and getattr(owner, 'pos', None) is not None:
                coordinates = owner.pos
            elif getattr(self, 'grid', None) is not None:
                coordinates = (self.grid.width // 2, self.grid.height // 2)
            else:
                coordinates = (0, 0)

        best_village = None
        best_score = None
        for village in villages:
            village_pos = getattr(village, 'pos', None) or getattr(village, 'position', None)
            if village_pos is None:
                continue
            distance = abs(village_pos[0] - coordinates[0]) + abs(village_pos[1] - coordinates[1])
            score = (distance, str(village.unique_id))
            if best_score is None or score < best_score:
                best_score = score
                best_village = village
        return best_village

    def get_operated_area_by_village(self, agent):
        """Map an operator's current operated area to village jurisdictions."""
        area_by_village = {}
        if agent is None:
            return area_by_village

        cache_version = getattr(self, 'land_tenure_version', 0)
        if cache_version != self._operated_area_by_village_cache_version:
            self._operated_area_by_village_cache = {}
            self._operated_area_by_village_cache_version = cache_version

        agent_key = str(getattr(agent, 'unique_id', id(agent)))
        cached = self._operated_area_by_village_cache.get(agent_key)
        if cached is not None:
            return dict(cached)

        seen_patch_ids = set()
        candidate_patches = []

        for patch in getattr(agent, 'land_patches', []) or []:
            leased_to = getattr(patch, 'leased_to', None)
            current_operator = getattr(patch, 'current_operator', None)
            if (
                getattr(patch, 'owner', None) is agent
                and (leased_to is None or leased_to is agent)
                and (current_operator is None or current_operator is agent)
            ):
                candidate_patches.append(patch)

        for attr_name in ('leased_in_patches', 'leased_patches'):
            for patch in getattr(agent, attr_name, []) or []:
                if getattr(patch, 'leased_to', None) is agent or getattr(patch, 'current_operator', None) is agent:
                    candidate_patches.append(patch)

        for patch in getattr(self, 'land_patches', []) or []:
            if getattr(patch, 'current_operator', None) is agent:
                candidate_patches.append(patch)

        for patch in candidate_patches:
            if (getattr(patch, 'operational_transfer_channel', '') == 'author_paid_initial_lease'
                    and patch.current_operator is not agent):
                continue
            patch_key = id(patch)
            if patch_key in seen_patch_ids:
                continue
            seen_patch_ids.add(patch_key)
            if not (
                getattr(patch, 'current_operator', None) is agent
                or getattr(patch, 'leased_to', None) is agent
                or (
                    getattr(patch, 'owner', None) is agent
                    and getattr(patch, 'leased_to', None) is None
                )
            ):
                continue
            village = self.get_primary_village_for_patch(patch)
            if village is None:
                continue
            area_by_village[village.unique_id] = area_by_village.get(village.unique_id, 0.0) + float(getattr(patch, 'area', 0.0) or 0.0)

        self._operated_area_by_village_cache[agent_key] = dict(area_by_village)
        return area_by_village

    def get_owned_area_by_village(self, agent):
        """Map an owner's contracted land area to village jurisdictions."""
        area_by_village = {}
        if agent is None:
            return area_by_village

        for patch in getattr(agent, 'land_patches', []) or []:
            if getattr(patch, 'owner', None) is not agent:
                continue
            village = self.get_primary_village_for_patch(patch)
            if village is None:
                continue
            area_by_village[village.unique_id] = area_by_village.get(village.unique_id, 0.0) + float(getattr(patch, 'area', 0.0) or 0.0)
        return area_by_village

    def get_primary_village_for_agent(self, agent):
        """Return the agent's primary village based on assignment or nearest jurisdiction."""
        if agent is None:
            return None

        primary_id = getattr(agent, 'primary_village_collective_id', None)
        if primary_id is not None:
            for village in self._get_village_collective_list():
                if village.unique_id == primary_id:
                    return village

        area_by_village = self.get_operated_area_by_village(agent)
        if not area_by_village:
            area_by_village = self.get_owned_area_by_village(agent)
        if area_by_village:
            best_village_id = max(
                area_by_village.items(),
                key=lambda item: (float(item[1] or 0.0), str(item[0])),
            )[0]
            for village in self._get_village_collective_list():
                if village.unique_id == best_village_id:
                    return village

        agent_pos = getattr(agent, 'pos', None)
        if agent_pos is None:
            return None

        best_village = None
        best_score = None
        for village in self._get_village_collective_list():
            village_pos = getattr(village, 'pos', None) or getattr(village, 'position', None)
            if village_pos is None:
                continue
            distance = abs(village_pos[0] - agent_pos[0]) + abs(village_pos[1] - agent_pos[1])
            score = (distance, str(village.unique_id))
            if best_score is None or score < best_score:
                best_score = score
                best_village = village
        return best_village

    def get_village_cross_village_transfer_area(self):
        return sum(
            float(getattr(village, 'annual_cross_village_transfer_area', 0.0) or 0.0)
            for village in self._get_village_collective_list()
        )

    def get_village_cross_village_contract_count(self):
        return sum(
            int(getattr(village, 'annual_cross_village_contracts', 0) or 0)
            for village in self._get_village_collective_list()
        )

    def get_village_collaboration_rate(self):
        villages = self._get_village_collective_list()
        if not villages:
            return 0.0
        participating = sum(
            1
            for village in villages
            if len(getattr(village, 'annual_cross_village_partners', set()) or set()) > 0
        )
        return participating / len(villages)

    def get_village_subsidy_pre_review_coverage(self):
        villages = self._get_village_collective_list()
        reviewed = sum(float(getattr(village, 'annual_reviewed_claims', 0.0) or 0.0) for village in villages)
        received = sum(float(getattr(village, 'annual_claims_received', 0.0) or 0.0) for village in villages)
        return reviewed / received if received > 0 else 0.0

    def get_village_subsidy_pre_review_approval_rate(self):
        villages = self._get_village_collective_list()
        approved = sum(float(getattr(village, 'annual_approved_claims', 0.0) or 0.0) for village in villages)
        reviewed = sum(float(getattr(village, 'annual_reviewed_claims', 0.0) or 0.0) for village in villages)
        return approved / reviewed if reviewed > 0 else 0.0

    def _iter_phase_managed_agents(self):
        """Yield institutional agents that should be run in explicit model phases."""
        government = getattr(self, "government", None)
        if government is not None:
            yield government

        for village in self._get_village_collective_list():
            yield village

        bureau = getattr(self, "grain_reserve_bureau", None)
        if bureau is not None:
            yield bureau

    def _ensure_phase_managed_agents_removed_from_schedule(self):
        """Avoid double execution when institutional actors are mistakenly scheduled."""
        scheduled_agents = getattr(self.schedule, "agents", None)
        if scheduled_agents is None:
            return

        for agent in self._iter_phase_managed_agents():
            if agent in scheduled_agents:
                self.schedule.remove(agent)

    def _run_governance_stage(self):
        """Government acts ex ante as the governance-stage leader."""
        government = getattr(self, "government", None)
        if government is None:
            return

        if hasattr(government, "begin_annual_cycle"):
            government.begin_annual_cycle()
            return
        government.step()

    def _finalize_governance_stage(self):
        """Government evaluates outcomes ex post after production is realized."""
        government = getattr(self, "government", None)
        if government is None or not hasattr(government, "end_annual_cycle"):
            return
        government.end_annual_cycle()

    def _run_village_collective_stage(self):
        """Village collective organizes meso-level land coordination before clearing."""
        villages = self._get_village_collective_list()
        for village in villages:
            if hasattr(village, "begin_annual_cycle"):
                village.begin_annual_cycle()
            village.step()
        self._run_cross_village_governance_stage(villages)

    def _run_actor_land_market_planning_stage(self):
        """Let producers refresh current-year crop plans before land-market clearing."""
        self._run_fixed_rule_decisions()
        for farmer in sorted(self.farmers, key=lambda item: str(getattr(item, 'unique_id', item))):
            if hasattr(farmer, "prepare_land_market_decisions"):
                farmer.prepare_land_market_decisions()

        for enterprise in sorted(self.agricultural_enterprises, key=lambda item: str(getattr(item, 'unique_id', item))):
            if hasattr(enterprise, "prepare_annual_decisions"):
                enterprise.prepare_annual_decisions()

    def _run_actor_labor_planning_stage(self):
        """Refresh farmer labor allocation after land transfers but before labor-market clearing."""
        for farmer in sorted(self.farmers, key=lambda item: str(getattr(item, 'unique_id', item))):
            if hasattr(farmer, "prepare_labor_market_decisions"):
                farmer.prepare_labor_market_decisions()

    def _register_collective_supply_into_clearing_house(self):
        """Re-register collective-integrated supply after the annual clearing house is rebuilt."""
        clearing_house = getattr(self, 'land_clearing_house', None)
        if clearing_house is None:
            return

        existing_patch_ids = set()
        for offer in getattr(clearing_house, 'supply_offers', []) or []:
            patch = offer.get('patch') if isinstance(offer, dict) else None
            if patch is None:
                continue
            existing_patch_ids.add(getattr(patch, 'patch_id', None) or id(patch))

        for village in self._get_village_collective_list():
            register_supply = getattr(village, '_register_integrated_supply', None)
            if register_supply is None:
                continue
            for patch in getattr(village, 'integrated_land_pool', []) or []:
                if patch is None or getattr(patch, 'leased_to', None) is not None:
                    continue
                patch_id = getattr(patch, 'patch_id', None) or id(patch)
                if patch_id in existing_patch_ids:
                    continue
                register_supply(patch)
                existing_patch_ids.add(patch_id)

    def _run_cross_village_governance_stage(self, villages):
        """Execute cross-village subcontracting and collaborative-governance bargaining."""
        if not villages or len(villages) <= 1:
            return

        for village in villages:
            if hasattr(village, "prepare_cross_village_market_state"):
                village.prepare_cross_village_market_state()

        processed_pairs = set()
        for village in villages:
            for neighbor in getattr(village, "network_neighbors", []) or []:
                if neighbor is None:
                    continue
                pair_key = tuple(sorted((str(village.unique_id), str(neighbor.unique_id))))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                if hasattr(village, "attempt_cross_village_coordination_with"):
                    village.attempt_cross_village_coordination_with(neighbor)

    def _fixed_rule_enabled(self):
        return fixed_rule_enabled(self.config if isinstance(self.config, dict) else {})

    def _run_fixed_rule_decisions(self):
        if not self._fixed_rule_enabled():
            return
        experiment = self.config.get("experiment", {}) if isinstance(self.config, dict) else {}
        smoke_years = experiment.get("smoke_years", {}) if isinstance(experiment, dict) else {}
        start_year = int(smoke_years.get("start", self.simulation_base_year) or self.simulation_base_year)
        year = int(start_year + self.step_count - 1)
        if action_income_enabled(self.config):
            year = int(self.simulation_base_year + self.step_count)
        trace_rows, exited_farmers = evaluate_fixed_rule_operators(
            self.config,
            self.farmers,
            self.agricultural_enterprises,
            year,
        )
        self.v4_8_fixed_rule_trace.extend(trace_rows)
        for farmer in exited_farmers:
            self._remove_fixed_rule_exited_farmer(farmer)

    def _remove_fixed_rule_exited_farmer(self, farmer):
        if farmer not in self.farmers:
            return
        self.record_agent_exit(
            farmer,
            ExitReason.VOLUNTARY_EXIT,
            {"source": "v4_8_fixed_rule", "action": "exit"},
        )
        population_manager = getattr(self, 'population_manager', None)
        if population_manager is not None:
            current_year = int(self.simulation_base_year + self.step_count)
            population_manager.current_year = current_year
            population_manager.land_bank.current_year = current_year
            population_manager._process_agent_exit(farmer, 'fixed_rule_exit')
        self.farmers.remove(farmer)
        if hasattr(self, 'schedule'):
            self.schedule.remove(farmer)
        if hasattr(self, 'grid') and getattr(farmer, 'pos', None) is not None:
            self.grid.remove_agent(farmer)
        self.agent_registry.pop(str(getattr(farmer, 'unique_id', '')), None)

    def _finalize_fixed_rule_exits(self):
        if action_income_enabled(self.config):
            if getattr(self, 'initial_paid_lease_accounts', ()):
                for enterprise in list(self.agricultural_enterprises):
                    if getattr(enterprise, 'fixed_rule_exit_pending', False):
                        self._remove_enterprise_agent(enterprise, ExitReason.VOLUNTARY_EXIT, 'paid_lease_production_exit')
            return
        if not self._fixed_rule_enabled():
            return
        for farmer in list(self.farmers):
            if bool(getattr(farmer, "fixed_rule_exit_pending", False)):
                self._remove_fixed_rule_exited_farmer(farmer)

    def _record_market_stage(self, stage):
        if not self._fixed_rule_enabled():
            return
        self.v4_8_market_stage_trace.append({"step": int(self.step_count), "stage": stage})

    def _run_fixed_rule_production_stage(self):
        paid_accounts = getattr(self, 'initial_paid_lease_accounts', ())
        cash_mode = preproduction_cash_enabled(self.config)
        if paid_accounts or cash_mode:
            # B* rent is debited before the household recognizes actual receipts.
            enterprises = list(self.agricultural_enterprises)
            for account in paid_accounts:
                if account.tenant not in enterprises:
                    enterprises.append(account.tenant)
            if cash_mode:
                for row in getattr(self, 'preproduction_rent_ledger', []):
                    tenant = row['tenant']
                    if row['total_due_yuan'] > 0 and hasattr(tenant, 'enterprise_type') and tenant not in enterprises:
                        enterprises.append(tenant)
            for enterprise in sorted(enterprises, key=lambda item: str(item.unique_id)):
                if cash_mode and enterprise not in self.agricultural_enterprises:
                    enterprise.hired_labor_days = enterprise.hired_labor_cost = 0.0
                    enterprise.service_expense = enterprise.fixed_rule_conversion_expense = 0.0
                    enterprise.fixed_rule_labor_settled = True
                enterprise._execute_post_market_cycle()
        for farmer in sorted(self.farmers, key=lambda item: str(getattr(item, "unique_id", item))):
            if bool(getattr(farmer, "_skip_remaining_step_this_cycle", False)):
                continue
            production_cycle = getattr(farmer, "_execute_post_market_cycle", None)
            if callable(production_cycle):
                production_cycle()
        for enterprise in sorted(
            () if paid_accounts or cash_mode else self.agricultural_enterprises,
            key=lambda item: str(getattr(item, "unique_id", item)),
        ):
            production_cycle = getattr(enterprise, "_execute_post_market_cycle", None)
            if callable(production_cycle):
                production_cycle()
        for household in getattr(self, 'initial_household_cohort', ()):
            if household in self.farmers:
                continue
            # Demographic exit ends production, not the initial household's account.
            household.age += 1
            household.fixed_rule_exit_pending = True
            household.planned_crop = None
            household.expected_yield = household.current_production_area = 0.0
            household.service_fee_expense = household.fixed_rule_conversion_expense = 0.0
            household.labor_sale_income = household.labor_sold_days = 0.0
            household.hired_labor_cost = household.hired_labor_days = 0.0
            household.harvest_and_settle()
            household.make_consumption_decision()
            household.livelihood_strategy_evolution()

    def _finalize_village_collective_stage(self):
        """Run ex-post village verification after agents submit realized claims."""
        for village in self._get_village_collective_list():
            if hasattr(village, "finalize_annual_cycle"):
                village.finalize_annual_cycle()

    def _begin_grain_reserve_stage(self):
        """Reset reserve-bureau bookkeeping before the grain market clears."""
        bureau = getattr(self, "grain_reserve_bureau", None)
        if bureau is not None and hasattr(bureau, "begin_annual_cycle"):
            bureau.begin_annual_cycle()

    def _finalize_grain_reserve_stage(self):
        """Finalize reserve-bureau accounting after market intervention is executed."""
        bureau = getattr(self, "grain_reserve_bureau", None)
        if bureau is not None and hasattr(bureau, "finalize_annual_cycle"):
            bureau.finalize_annual_cycle()

    def step(self):
        """
        @theory
        模型年度决策周期
        严格按照设计规范的阶段顺序执行
        
        @fix 2025-12: 添加多年期合同年限递减逻辑
        @fix 2025-11-29: 添加劳动力市场出清步骤
        @new 2025-12-04: 添加服务市场统计重置
        """
        if getattr(self, 'area_economy', None) is not None:
            return self._step_area_economy()
        self.step_count += 1
        from core.p3_project_contracts import configured_program
        if configured_program(self) is not None:
            current_year = int(self.simulation_base_year + self.step_count)
            for provider in self.service_providers:
                provider.active_contracts = [
                    contract for contract in provider.active_contracts
                    if not contract.is_expired(current_year)
                ]
                provider.service_area_used = sum(contract.area for contract in provider.active_contracts)
        self._refresh_total_land_area()
        self._reset_annual_capitalization_accounting()
        self._ensure_phase_managed_agents_removed_from_schedule()
        
        # @new 2025-12-22: 市场协调器年度周期开始（FIX_REPORT改进）
        # 在任何市场交易前执行资本预留和预期更新
        if self.market_coordinator is not None:
            self.market_coordinator.begin_annual_cycle()
        
        # @new 2025-12-04: 重置当步服务市场统计
        self._reset_service_market_stats()
        
        # @new 2025-12-27: 重置年度土地流转追踪（每step=1年）
        # 机制：年度新增流转面积每年清零，但累计流转面积保留
        if hasattr(self, 'transfer_rate_control'):
            self.transfer_rate_control['annual_new_transferred_area'] = 0.0
        
        # 0. 更新土地租赁合同状态（多年期合同机制）
        self._update_land_lease_contracts()
        
        # 1. 环境更新 (政策和市场价格)
        self._record_market_stage("government")
        self._run_governance_stage()
        self._begin_grain_reserve_stage()
        self._record_market_stage("environment_price_update")
        self.environment.update_market_prices(self)
        
        # @new 2025-01-02: 技术进步更新
        # 在生产决策之前更新技术水平
        self._record_market_stage("technology_update")
        self.environment.update_technology_progress(self)

        # @new 2025-01-02: P3 空间气候风险更新
        # 在土地市场之前更新气候场，影响土地价值预期
        self._record_market_stage("climate_update")
        if hasattr(self, 'climate_manager') and self.climate_manager is not None:
            self.climate_manager.step()
        
        # @new 2025-01-02: P1 土地置换机制(每5年执行)
        # 理论: Deng et al. (2024, Nature Food) - 主动整合碎片化土地
        if self.step_count % 5 == 0 and self.step_count > 0:
            swaps = self.land_clearing_house.facilitate_land_swapping(self)
            if swaps > 0:
                pass  # 可选: 记录置换次数
        
        # 阶段二：土地租赁市场
        self._record_market_stage("land_market_planning")
        self._run_actor_land_market_planning_stage()
        self._record_market_stage("village_collective")
        self._run_village_collective_stage()
        self._record_market_stage("land_rental_market")
        self.conduct_land_rental_market()

        # @new 2025-11-29: 阶段三：先做劳动力规划
        # 服务托管会改变后续真实用工需求，因此先规划、再托管、后劳动力撮合。
        self._record_market_stage("labor_planning")
        self._run_actor_labor_planning_stage()
        
        # @new 2025-12: 阶段四：农业社会化服务市场出清
        # 理论：服务规模经济 - 让小农户通过购买托管服务实现规模效益
        self._record_market_stage("service_market")
        prepare_preproduction_cash(self)
        self._conduct_service_market()

        # @new 2026-03-20: 服务合同会降低劳动需求，必须在托管后再清算劳动力市场。
        self._record_market_stage("labor_market")
        self._conduct_labor_market()
        
        # 2. 智能体决策
        self._record_market_stage("agent_production")
        if self._fixed_rule_enabled():
            self._run_fixed_rule_production_stage()
        else:
            self.schedule.step()
        self._finalize_grain_reserve_stage()
        self._finalize_village_collective_stage()
        self._finalize_governance_stage()
        
        # @new 2025-01-02: 生态反馈更新
        # 在智能体生产决策后，更新地块生态状态
        self._update_ecological_feedback()
        self._settle_service_providers_end_of_year()

        # Annual production indicators must be collected before end-of-year
        # demographic turnover. Otherwise farmers/enterprises that produced in
        # the current year can be removed before reporting, while next-year
        # entrants are assigned land after production and appear as inactive
        # current-year operators.
        if hasattr(self, 'environmental_accountant') and self.environmental_accountant is not None:
            self.environmental_accountant.calculate_annual_footprint()

        self._record_market_stage("annual_collection")
        self.datacollector.collect(self)
        annual_snapshot_hook = getattr(self, '_completed_production_year_snapshot_hook', None)
        if callable(annual_snapshot_hook):
            annual_snapshot_hook(self)
        if self.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode') in ('B_star_1y', 'B_star_5y'):
            self._update_land_lease_contracts(after_production=True)
        self._finalize_fixed_rule_exits()
        
        # @new 2025-12-26: 新农户进入机制
        # @deprecated 2026-01-14: 禁用此方法，改用PopulationManagerV2的土地守恒机制
        # 问题诊断：此方法创建空土地列表，导致土地面积不守恒
        # 解决方案：使用下方population_manager.step()统一处理
        # self._process_farmer_entry()  # 已禁用，由PopulationManagerV2接管
        
        # @new 2026-01-06: 人口动态管理器年度步进
        # @update 2026-01-07: 支持V2版本（土地守恒）
        # 理论: Macro-Anchored Micro-Evolution Framework
        # 功能: 执行城镇化退出、死亡退出、新进入者引入、土地重分配
        # 数据: 使用NBS农村户数约束总量，CFPS提供概率和特征
        if hasattr(self, 'population_manager') and self.population_manager is not None:
            self._record_market_stage("population_dynamics")
            # 计算当前模拟年份（2010年为基期）
            current_sim_year = self.simulation_base_year + self.step_count
            previous_farmers = list(self.farmers)
            updated_farmers, pop_stats = self.population_manager.step(
                current_sim_year,
                self.farmers,
                self.agricultural_enterprises
            )
            
            # 处理返回的农户列表（可能包含字典形式的新进入者）
            converted_farmers = []
            for f in updated_farmers:
                if isinstance(f, dict):
                    # 新进入者是字典，需要转换为FarmerAgent
                    new_farmer = self._create_farmer_from_features(f)
                    if new_farmer:
                        converted_farmers.append(new_farmer)
                    elif self.population_manager is not None:
                        for parcel in f.get('land_parcels') or []:
                            self.population_manager.land_bank.deposit(parcel)
                else:
                    converted_farmers.append(f)
            
            # 更新农户列表并同步调度器/注册表
            self.farmers = converted_farmers
            self._synchronize_farmers_after_population_update(
                previous_farmers=previous_farmers,
                updated_farmers=self.farmers
            )
                
            # 记录人口动态统计
            if hasattr(self, 'population_dynamics_history'):
                self.population_dynamics_history.append({
                    'year': current_sim_year,
                    'stats': pop_stats
                })

        if not (hasattr(self, 'population_manager') and self.population_manager is not None):
            self._record_market_stage("population_dynamics")
        self._record_market_stage("enterprise_demography")
        self._update_enterprise_demography()
        if self.government is not None and hasattr(self.government, 'finish_government_q_year'):
            self.government.finish_government_q_year()
    
    def _conduct_labor_market(self):
        """
        @new 2025-11-29: 劳动力市场出清
        
        理论：
        1. 农户既是劳动力供给者，也可能是需求者（大户）
        2. 企业使用大型机械 + 雇佣农户劳动力
        3. 劳动力市场匹配后，确定雇佣关系和工资
        """
        if self.labor_market is not None:
            if not action_income_enabled(self.config):
                self.labor_market.update_market(self.farmers, self.agricultural_enterprises)
                return
            market = self.labor_market
            market._reset_market_accounts(self.farmers, self.agricultural_enterprises)
            market.labor_supply_pool.clear()
            market.labor_demand_pool.clear()
            for farmer in self.farmers:
                profile = farmer.estimate_production_labor_profile()
                farmer.family_labor_used = profile['family_labor_used']
                farmer.agricultural_labor = profile['family_labor_used']
                farmer.off_farm_labor = max(0.0, farmer.total_labor_units - farmer.family_labor_used)
                farmer.requested_hired_labor_days = profile['labor_gap']
                farmer.fixed_rule_labor_need = profile['total_labor_need']
                supply = market.calculate_farmer_labor_supply(farmer)
                supply['available_for_sale'] = min(supply['available_for_sale'], farmer.off_farm_labor)
                if supply['available_for_sale'] > 0:
                    market.labor_supply_pool[farmer.unique_id] = supply
                if profile['labor_gap'] > 0:
                    market.labor_demand_pool[f"farmer_{farmer.unique_id}"] = {
                        'farmer': farmer, 'labor_gap': profile['labor_gap'],
                        'max_wage': market.base_agricultural_wage * (0.9 + 0.2 * farmer.farming_ability),
                    }
            for enterprise in self.agricultural_enterprises:
                account = enterprise.expected_fixed_rule_account(enterprise.fixed_rule_current_action)
                hired_days = account.hired_labor_days
                if preproduction_cash_enabled(self.config):
                    _, hired_days, _ = production_cash_quote(enterprise, production_area(enterprise))
                enterprise.fixed_rule_labor_need = hired_days
                if hired_days > 0:
                    market.labor_demand_pool[enterprise.unique_id] = {
                        'enterprise': enterprise, 'hired_labor_need': hired_days,
                        'max_wage': market.base_agricultural_wage * enterprise.efficiency_factor * 1.2,
                    }
            if preproduction_cash_enabled(self.config):
                for demand in market.labor_demand_pool.values():
                    operator = demand.get('farmer', demand.get('enterprise'))
                    budget = operator.last_preproduction_cash_account
                    key = 'labor_gap' if 'farmer' in demand else 'hired_labor_need'
                    demand[key] = min(demand[key], budget['labor_reserved_yuan'] / demand['max_wage'])
            market._clear_market()
            market._settle_employment_records()
            market._update_equilibrium_wage()
            market._record_market_state()
            for operator in [*self.farmers, *self.agricultural_enterprises]:
                operator.fixed_rule_labor_settled = True
                need = operator.fixed_rule_labor_need
                supplied = getattr(operator, 'family_labor_used', 0.0) + operator.hired_labor_days
                operator.fixed_rule_labor_fulfillment = min(1.0, supplied / need) if need > 0 else 1.0
                if preproduction_cash_enabled(self.config):
                    budget = operator.last_preproduction_cash_account
                    paid = float(operator.hired_labor_cost)
                    if paid > budget['labor_reserved_yuan'] + 1e-6:
                        raise RuntimeError('Labor match exceeds reserved cash')
                    budget['labor_contracted_yuan'] = paid
                    budget['labor_commitment_released_yuan'] = budget['labor_reserved_yuan'] - paid
                    budget['committed_yuan'] -= budget['labor_commitment_released_yuan']
                    budget['uncommitted_yuan'] += budget['labor_commitment_released_yuan']
    
    def _conduct_service_market(self):
        """
        @new 2025-12: 农业社会化服务市场出清
        
        @theory 服务规模经济 - "隐性的农业革命"
        
        核心洞见（陈义媛、高原研究）：
        1. 中国农业现代化不仅是土地流转的故事
        2. 更大的故事是"服务规模"——通过托管实现规模效益
        3. 小农户保留土地承包权，但将生产服务外包给专业化服务商
        
        机制设计：
        1. 服务提供商有能力上限和服务范围
        2. 农户根据劳动力状况、年龄、成本效益决定是否购买服务
        3. 购买服务后，农户可以保留土地但减少投入劳动
        4. 服务商实现规模经济，单位成本下降
        
        与土地流转的替代关系：
        - 有服务可用时，老年农户可以继续持有土地
        - 服务市场发达地区，土地流转需求可能下降
        """
        if self.service_market is None or not self.service_providers:
            return

        # 1. 收集服务需求
        service_demands = []
        for farmer in self.farmers:
            if bool(getattr(farmer, '_skip_remaining_step_this_cycle', False)):
                continue
            demand = self._get_farmer_service_demand(farmer)
            if demand is not None and preproduction_cash_enabled(self.config):
                budget = farmer.last_preproduction_cash_account
                demand['area'] = min(demand['area'], production_area(farmer))
                if demand['area'] <= 0 or budget['uncommitted_yuan'] <= 0:
                    demand = None
                else:
                    demand['max_price_per_mu'] = min(
                        demand['max_price_per_mu'], budget['uncommitted_yuan'] / demand['area'])
            if demand is not None:
                service_demands.append(demand)

        demand_area = sum(float(d.get('area', 0.0) or 0.0) for d in service_demands)
        retained_demand_area = sum(
            float(d.get('area', 0.0) or 0.0)
            for d in service_demands
            if bool(d.get('service_managed_retained_grain', False))
        )
        self.service_market_stats['service_demand_count'] = len(service_demands)
        self.service_market_stats['service_demand_area'] = float(demand_area)
        self.service_market_stats['service_retained_grain_demand_area'] = float(retained_demand_area)
        matches = []
        
        # 2. 服务市场匹配
        if service_demands:
            matches = self.service_market.match_service_demands(
                service_demands, 
                self.service_providers
            )
            
            # 3. 应用匹配结果
            for match in matches:
                self._apply_service_contract(match)

        matched_area = sum(
            float(getattr(match.get('contract'), 'area', 0.0) or 0.0)
            for match in matches
        )
        retained_matched_area = sum(
            float(getattr(match.get('contract'), 'area', 0.0) or 0.0)
            for match in matches
            if bool(match.get('service_managed_retained_grain', False))
        )
        self.service_market_stats['service_matched_area'] = float(matched_area)
        self.service_market_stats['service_unmatched_demand_area'] = max(0.0, float(demand_area - matched_area))
        self.service_market_stats['service_unmatched_demand_count'] = max(
            0,
            int(len(service_demands) - len(matches)),
        )
        self.service_market_stats['service_retained_grain_matched_area'] = float(retained_matched_area)
        self.service_market_stats['service_retained_grain_unmatched_area'] = max(
            0.0,
            float(retained_demand_area - retained_matched_area),
        )

    def _settle_service_providers_end_of_year(self):
        """Finalize service-provider accounts with the current simulation year."""
        from core.p3_project_contracts import configured_program
        p3_program = configured_program(self)
        unified_cfg = self.config.get('unified_paper_model', {}) if isinstance(self.config, dict) else {}
        if (
            isinstance(unified_cfg, dict)
            and unified_cfg.get('enabled') is True
            and unified_cfg.get('service_environment_only') is True
            and p3_program is None
        ):
            self.service_market_stats['provider_profit_total'] = 0.0
            self.service_market_stats['provider_profit_mean'] = 0.0
            return
        if not self.service_providers:
            return

        current_year = int(self.simulation_base_year + self.step_count)
        provider_profits = []
        for provider in self.service_providers:
            if p3_program is not None:
                provider_profits.append(float(provider.end_of_year_settlement(current_year)))
                continue
            settle_fn = getattr(provider, 'end_of_year_settlement', None)
            if not callable(settle_fn):
                continue
            try:
                provider_profits.append(float(settle_fn(current_year) or 0.0))
            except TypeError:
                provider_profits.append(float(settle_fn() or 0.0))
            except Exception:
                continue

        self.service_market_stats['provider_profit_total'] = float(sum(provider_profits)) if provider_profits else 0.0
        self.service_market_stats['provider_profit_mean'] = float(np.mean(provider_profits)) if provider_profits else 0.0

    def _get_enterprise_population_dynamics_config(self):
        enterprise_cfg = self.config.get('agents', {}).get('enterprise', {}) or {}
        dynamics_cfg = enterprise_cfg.get('population_dynamics', {}) or {}
        return dynamics_cfg if isinstance(dynamics_cfg, dict) else {}

    def _resolve_enterprise_count_bounds(self, dynamics_cfg):
        scale_cfg = self.config.get('scale', {}) or {}
        current_farmers = max(1, len(self.farmers))
        initial_enterprise_count = int(
            getattr(self, '_enterprise_demography_initial_count', len(self.agricultural_enterprises))
            or 0
        )
        if not hasattr(self, '_enterprise_demography_initial_count'):
            self._enterprise_demography_initial_count = initial_enterprise_count
        anchor_farmers = int(
            getattr(self, '_enterprise_demography_anchor_farmers', current_farmers)
            or current_farmers
        )
        if not hasattr(self, '_enterprise_demography_anchor_farmers'):
            self._enterprise_demography_anchor_farmers = anchor_farmers
        count_anchor = str(dynamics_cfg.get('count_anchor', 'initial_enterprise_count') or 'initial_enterprise_count')
        farmer_count_for_target = anchor_farmers if count_anchor == 'initial_farmer_count' else current_farmers
        base_ratio = float(
            dynamics_cfg.get(
                'target_ratio',
                scale_cfg.get('enterprise_ratio', 0.0),
            ) or 0.0
        )
        min_ratio = float(dynamics_cfg.get('min_ratio', base_ratio) or 0.0)
        max_ratio = float(dynamics_cfg.get('max_ratio', base_ratio if base_ratio > 0 else 1.0) or 0.0)

        configured_min_count = int(dynamics_cfg.get('min_count', 0) or 0)
        ratio_count_base = anchor_farmers if count_anchor == 'initial_farmer_count' else current_farmers
        ratio_min_count = int(round(ratio_count_base * min_ratio)) if min_ratio > 0 else 0
        ratio_max_count = int(round(ratio_count_base * max_ratio)) if max_ratio > 0 else 0
        if ratio_max_count > 0 and configured_min_count > ratio_max_count:
            # `min_count` is calibrated for the production-scale sample. In
            # 100-farmer smoke/paired runs, keep the ratio semantics instead of
            # trying to force dozens of enterprises into a small world.
            configured_min_count = 0
        min_count = max(configured_min_count, ratio_min_count)
        max_count = int(dynamics_cfg.get('max_count', max(min_count, ratio_max_count or 0)) or 0)
        if ratio_max_count > 0:
            max_count = min(max_count, max(min_count, ratio_max_count))
        max_count = max(min_count, max_count)

        target_count = int(round(farmer_count_for_target * base_ratio)) if base_ratio > 0 else len(self.agricultural_enterprises)
        if count_anchor == 'initial_enterprise_count':
            target_count = max(target_count, initial_enterprise_count)
            min_count = max(min_count, initial_enterprise_count)
            max_count = max(max_count, min_count)
        target_count = max(min_count, min(max_count, target_count))
        return min_count, max_count, target_count

    @staticmethod
    def _get_enterprise_recent_metric(enterprise, attr_name, history_attr, window=2):
        history = list(getattr(enterprise, history_attr, None) or [])
        if history:
            tail = [float(value or 0.0) for value in history[-window:]]
            return float(sum(tail) / len(tail))
        return float(getattr(enterprise, attr_name, 0.0) or 0.0)

    @staticmethod
    def _get_enterprise_loss_streak(enterprise):
        streak = 0
        history = list(getattr(enterprise, 'profit_history', None) or [])
        if history:
            for value in reversed(history):
                if float(value or 0.0) < 0:
                    streak += 1
                else:
                    break
            return streak
        return 1 if float(getattr(enterprise, 'profit', 0.0) or 0.0) < 0 else 0

    def _identify_enterprise_exit_candidates(self, dynamics_cfg):
        exit_loss_years = max(1, int(dynamics_cfg.get('exit_consecutive_loss_years', 3) or 3))
        exit_min_operated_area = max(0.0, float(dynamics_cfg.get('exit_min_operated_area', 3.0) or 3.0))
        exit_capital_floor = float(dynamics_cfg.get('exit_capital_floor', 30000.0) or 30000.0)
        exit_roi_floor = float(dynamics_cfg.get('exit_roi_floor', -0.03) or -0.03)

        candidates = []
        for enterprise in self.agricultural_enterprises:
            operated_area = max(0.0, float(enterprise.get_operated_land_area() or 0.0))
            capital = float(getattr(enterprise, 'capital', 0.0) or 0.0)
            recent_profit = self._get_enterprise_recent_metric(enterprise, 'profit', 'profit_history')
            recent_roi = self._get_enterprise_recent_metric(enterprise, 'roi', 'roi_history')
            loss_streak = self._get_enterprise_loss_streak(enterprise)

            exit_reason = None
            if loss_streak >= exit_loss_years and recent_roi <= exit_roi_floor:
                exit_reason = ExitReason.PROFIT_NEGATIVE
            elif capital <= exit_capital_floor:
                exit_reason = ExitReason.CAPITAL_DEPLETED
            elif operated_area <= exit_min_operated_area and loss_streak >= max(1, exit_loss_years - 1):
                exit_reason = ExitReason.SCALE_TOO_SMALL

            if exit_reason is None:
                continue

            notes = (
                f"area={operated_area:.2f}, capital={capital:.2f}, "
                f"recent_profit={recent_profit:.2f}, recent_roi={recent_roi:.4f}, "
                f"loss_streak={loss_streak}"
            )
            candidates.append((enterprise, exit_reason, notes, operated_area, capital, recent_roi))

        candidates.sort(key=lambda item: (item[3] > 0.0, item[3], item[4], item[5], str(item[0].unique_id)))
        return candidates

    def _identify_enterprise_rebalance_candidates(self, excluded_ids=None):
        excluded_ids = set(excluded_ids or [])
        candidates = []
        for enterprise in self.agricultural_enterprises:
            unique_id = str(getattr(enterprise, 'unique_id', '') or '')
            if unique_id in excluded_ids:
                continue
            operated_area = max(0.0, float(enterprise.get_operated_land_area() or 0.0))
            capital = float(getattr(enterprise, 'capital', 0.0) or 0.0)
            recent_profit = self._get_enterprise_recent_metric(enterprise, 'profit', 'profit_history')
            recent_roi = self._get_enterprise_recent_metric(enterprise, 'roi', 'roi_history')
            notes = (
                f"structural_rebalance area={operated_area:.2f}, capital={capital:.2f}, "
                f"recent_profit={recent_profit:.2f}, recent_roi={recent_roi:.4f}"
            )
            candidates.append((enterprise, ExitReason.MARKET_SQUEEZE, notes, operated_area, capital, recent_roi))
        candidates.sort(key=lambda item: (item[3], item[5], item[4], str(item[0].unique_id)))
        return candidates

    def _release_enterprise_land(self, enterprise):
        land_bank = getattr(getattr(self, 'population_manager', None), 'land_bank', None)
        current_year = int(self.simulation_base_year + self.step_count)
        seen_patch_ids = set()
        candidate_patches = []

        if hasattr(enterprise, '_get_active_leased_patches'):
            try:
                candidate_patches.extend(enterprise._get_active_leased_patches(include_current_operator=True))
            except TypeError:
                candidate_patches.extend(enterprise._get_active_leased_patches())
            except Exception:
                pass

        for attr_name in ('leased_in_patches', 'leased_patches'):
            candidate_patches.extend(getattr(enterprise, attr_name, None) or [])

        for patch in candidate_patches:
            patch_key = id(patch)
            if patch_key in seen_patch_ids:
                continue
            seen_patch_ids.add(patch_key)

            owner_is_enterprise = getattr(patch, 'owner', None) is enterprise
            leased_to_enterprise = getattr(patch, 'leased_to', None) is enterprise
            patch_area = float(getattr(patch, 'area', 0.0) or 0.0)

            if owner_is_enterprise:
                if land_bank is not None and patch_area > 0:
                    from modules.population_dynamics_v2 import LandParcel

                    coordinates = getattr(patch, 'coordinates', None) or (0.0, 0.0)
                    land_bank.deposit(
                        LandParcel(
                            area_mu=patch_area,
                            quality=float(getattr(patch, 'soil_quality', 1.0) or 1.0),
                            location_x=float(coordinates[0]),
                            location_y=float(coordinates[1]),
                            previous_owner_id=enterprise.unique_id,
                            exit_year=current_year,
                            exit_type='enterprise_exit',
                            patches=[patch],
                        )
                    )

                patch.owner = None
                patch.leased_to = None
                patch.current_operator = None
                patch.rental_price = 0.0
                patch.contracted_rental_price = 0.0
                patch.lease_contract_years = 0
                setattr(patch, 'operational_transfer_active', False)
                setattr(patch, 'operational_transfer_channel', '')
            elif leased_to_enterprise:
                try:
                    patch.end_lease()
                except Exception:
                    patch.leased_to = None
                    patch.current_operator = getattr(patch, 'owner', None)

        enterprise.leased_patches = []
        enterprise.leased_in_patches = []
        enterprise.total_leased_area = 0.0
        enterprise._leased_patch_cache = []
        enterprise._leased_patch_cache_version = None
        enterprise._operated_land_area_cache = 0.0
        enterprise._operated_land_area_cache_version = None
        self.invalidate_land_tenure_caches()

    def _rollback_spawned_enterprise(self, enterprise):
        if enterprise in self.agricultural_enterprises:
            self.agricultural_enterprises.remove(enterprise)
        if hasattr(self.schedule, 'remove'):
            try:
                self.schedule.remove(enterprise)
            except Exception:
                pass
        if getattr(enterprise, 'pos', None) is not None:
            try:
                self.grid.remove_agent(enterprise)
            except Exception:
                pass
        self.agent_registry.pop(getattr(enterprise, 'unique_id', None), None)

    def _remove_enterprise_agent(self, enterprise, exit_reason, notes):
        if any(a.tenant is enterprise for a in getattr(self, 'initial_paid_lease_accounts', ())):
            from core.fixed_rule_runtime import enact_fixed_rule_action
            enact_fixed_rule_action(enterprise, 'exit', 'scale')
        else:
            self._release_enterprise_land(enterprise)
        self.graveyard.record_exit(
            enterprise,
            exit_reason=exit_reason,
            step=self.step_count,
            notes=notes,
        )
        self._rollback_spawned_enterprise(enterprise)

    def _spawn_dynamic_enterprise(self, entry_index):
        from agents.enterprise_agent import AgriculturalEnterpriseAgent, EnterpriseType

        enterprise_cfg = self.config.get('agents', {}).get('enterprise', {}) or {}
        type_distribution = enterprise_cfg.get('type_distribution', {}) or {
            'AGRI_CORPORATION': 0.6,
            'COOPERATIVE': 0.4,
        }
        type_names = list(type_distribution.keys())
        probabilities = np.array([float(type_distribution[name] or 0.0) for name in type_names], dtype=float)
        probabilities[probabilities < 0] = 0.0
        if probabilities.sum() <= 0:
            probabilities = np.array([0.6, 0.4], dtype=float)
            type_names = ['AGRI_CORPORATION', 'COOPERATIVE']
        probabilities = probabilities / probabilities.sum()

        type_mapping = {
            'AGRI_CORPORATION': EnterpriseType.AGRI_CORPORATION,
            'COOPERATIVE': EnterpriseType.COOPERATIVE,
        }
        enterprise_type_name = str(np_choice(self, type_names, p=probabilities.tolist()))
        enterprise_type = type_mapping.get(enterprise_type_name, EnterpriseType.AGRI_CORPORATION)

        type_cfg = enterprise_cfg.get('types', {}).get(enterprise_type_name, {}) or {}
        capital_dist = type_cfg.get('initial_capital_dist', {}) or {}
        if capital_dist.get('type') == 'lognormal':
            initial_capital = float(
                np_lognormal(
                    self,
                    capital_dist.get('mean', 13.0),
                    capital_dist.get('std', 0.8),
                )
            )
        else:
            initial_capital = float(type_cfg.get('initial_capital', 500000.0) or 500000.0)

        unique_id = f"enterprise_dyn_{self.step_count}_{entry_index}_{len(self.agricultural_enterprises)}"
        enterprise = AgriculturalEnterpriseAgent(
            unique_id,
            self,
            enterprise_type=enterprise_type,
            initial_capital=initial_capital,
        )
        enterprise.is_new_entrant = True
        enterprise.entry_year = int(self.simulation_base_year + self.step_count)
        self.agricultural_enterprises.append(enterprise)
        self.schedule.add(enterprise)
        self.agent_registry[enterprise.unique_id] = enterprise

        x = rng_randint(self, self.grid.width // 4, 3 * self.grid.width // 4)
        y = rng_randint(self, self.grid.height // 4, 3 * self.grid.height // 4)
        self.grid.place_agent(enterprise, (x, y))
        return enterprise

    def _allocate_household_operational_land_to_enterprise(self, enterprise, target_area, dynamics_cfg):
        if not bool(dynamics_cfg.get('startup_household_leasing_enabled', True)):
            return 0.0

        target_area = max(0.0, float(target_area or 0.0))
        if target_area <= 0:
            return 0.0

        min_owner_retained_area = max(
            0.0,
            float(dynamics_cfg.get('startup_owner_retained_area_mu',
                                   dynamics_cfg.get('initial_owner_retained_area_mu', 2.0)) or 2.0),
        )
        contract_years = max(
            1,
            int(dynamics_cfg.get('startup_household_contract_years',
                                 dynamics_cfg.get('initial_operational_contract_years', 30)) or 30),
        )
        transfer_cfg = self.config.get('agents', {}).get('farmer', {}).get('land_transfer', {}) or {}
        base_rent = float(transfer_cfg.get('base_rent_price', 420.0) or 420.0)

        candidates = []
        for patch in list(getattr(self, 'land_patches', []) or []):
            area = float(getattr(patch, 'area', 0.0) or 0.0)
            owner = getattr(patch, 'owner', None)
            if area <= 0 or owner is None or owner is enterprise:
                continue
            if getattr(patch, 'leased_to', None) is not None:
                continue

            retained_area_ok = True
            if hasattr(owner, 'get_owned_land_area'):
                try:
                    retained_area_ok = (
                        float(owner.get_owned_land_area(include_leased_out=True) or 0.0) - area
                        >= min_owner_retained_area
                    )
                except TypeError:
                    retained_area_ok = (
                        float(owner.get_owned_land_area() or 0.0) - area
                        >= min_owner_retained_area
                    )
                except Exception:
                    retained_area_ok = True
            if not retained_area_ok:
                continue

            livelihood = str(getattr(getattr(owner, 'livelihood_strategy', None), 'value', '') or '')
            willingness = float(getattr(owner, 'land_transfer_willingness', 0.0) or 0.0)
            pending_area = float(getattr(owner, 'land_pending_transfer', 0.0) or 0.0)
            age = float(getattr(owner, 'age', 0.0) or 0.0)
            priority = (
                0 if livelihood == 'Non_Farmer' else 1,
                -pending_area,
                -willingness,
                -age,
                area,
            )
            candidates.append((priority, patch, owner, area))

        candidates.sort(key=lambda item: item[0])
        assigned_total = 0.0
        for _, patch, owner, area in candidates:
            if assigned_total >= target_area:
                break
            if assigned_total > 0 and assigned_total + area > target_area * 1.25:
                continue
            gate = getattr(patch, '_passes_operational_financial_gate', None)
            if callable(gate) and not gate(enterprise, base_rent):
                continue

            patch.leased_to = enterprise
            patch.current_operator = enterprise
            patch.lease_contract_years = max(int(getattr(patch, 'lease_contract_years', 0) or 0), contract_years)
            patch.rental_price = base_rent
            patch.contracted_rental_price = base_rent
            setattr(patch, 'operational_transfer_active', True)
            setattr(patch, 'operational_transfer_channel', 'dynamic_enterprise_startup')

            for attr_name in ('leased_patches', 'leased_in_patches'):
                if hasattr(enterprise, attr_name):
                    patches = getattr(enterprise, attr_name)
                    if patch not in patches:
                        patches.append(patch)
            if hasattr(owner, 'leased_out_patches') and patch not in owner.leased_out_patches:
                owner.leased_out_patches.append(patch)

            assigned_total += area

        if assigned_total > 0:
            leased_area = sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in (getattr(enterprise, 'leased_patches', []) or [])
            )
            setattr(enterprise, 'total_leased_area', leased_area)
            setattr(enterprise, '_leased_patch_cache_version', None)
            setattr(enterprise, '_operated_land_area_cache_version', None)
            setattr(enterprise, '_operated_land_area_cache', leased_area)
            self.invalidate_land_tenure_caches()

        return float(assigned_total)

    def _allocate_startup_land_to_enterprise(self, enterprise, dynamics_cfg):
        land_bank = getattr(getattr(self, 'population_manager', None), 'land_bank', None)

        entry_land_trigger = max(5.0, float(dynamics_cfg.get('entry_land_trigger_mu', 60.0) or 60.0))
        expansion_target_cfg = (
            self.config.get('agents', {})
            .get('enterprise', {})
            .get('expansion_decision', {})
            .get('expansion_target', {})
        )
        enterprise_type_name = getattr(getattr(enterprise, 'enterprise_type', None), 'name', 'AGRI_CORPORATION')
        starter_area = float(
            expansion_target_cfg.get('starter_area_by_type', {}).get(
                enterprise_type_name,
                entry_land_trigger,
            ) or entry_land_trigger
        )
        starter_area = max(5.0, starter_area)

        allocated_area = 0.0
        if land_bank is not None and float(getattr(land_bank, 'total_area_mu', 0.0) or 0.0) > 0:
            land_bank.random_context = self
            distributed = land_bank.distribute_to_neighbors(
                [enterprise],
                max_per_agent=starter_area,
                max_total_area=starter_area,
                preserve_order=True,
                rent_per_mu=float(
                    self.config.get('agents', {})
                    .get('farmer', {})
                    .get('land_transfer', {})
                    .get('base_rent_price', 420.0) or 420.0
                ),
            )
            allocated_area = float(distributed.get('total_distributed', 0.0) or 0.0)

        remaining_target = max(0.0, starter_area - allocated_area)
        if remaining_target > 0:
            allocated_area += self._allocate_household_operational_land_to_enterprise(
                enterprise,
                remaining_target,
                dynamics_cfg,
            )
        return float(allocated_area)

    def _update_enterprise_demography(self):
        dynamics_cfg = self._get_enterprise_population_dynamics_config()
        if not dynamics_cfg.get('enabled', False):
            return

        min_count, max_count, target_count = self._resolve_enterprise_count_bounds(dynamics_cfg)
        annual_exit_cap = max(0, int(dynamics_cfg.get('annual_exit_cap', 0) or 0))
        annual_entry_cap = max(0, int(dynamics_cfg.get('annual_entry_cap', 0) or 0))
        target_rebalance_enabled = bool(dynamics_cfg.get('target_rebalance_enabled', True))
        target_rebalance_band = max(0, int(dynamics_cfg.get('target_rebalance_band', 2) or 2))
        exited_ids = set()
        exits_completed = 0

        if annual_exit_cap > 0 and len(self.agricultural_enterprises) > min_count:
            exit_candidates = self._identify_enterprise_exit_candidates(dynamics_cfg)
            max_exits = min(annual_exit_cap, max(0, len(self.agricultural_enterprises) - min_count))
            for enterprise, exit_reason, notes, *_ in exit_candidates[:max_exits]:
                exited_ids.add(str(getattr(enterprise, 'unique_id', '') or ''))
                self._remove_enterprise_agent(enterprise, exit_reason, notes)
                exits_completed += 1

        if (
            annual_exit_cap > exits_completed
            and target_rebalance_enabled
            and len(self.agricultural_enterprises) > max(min_count, target_count + target_rebalance_band)
        ):
            structural_gap = len(self.agricultural_enterprises) - max(min_count, target_count)
            max_structural_exits = min(annual_exit_cap - exits_completed, max(0, structural_gap))
            structural_candidates = self._identify_enterprise_rebalance_candidates(exited_ids)
            for enterprise, exit_reason, notes, *_ in structural_candidates[:max_structural_exits]:
                self._remove_enterprise_agent(enterprise, exit_reason, notes)
                exits_completed += 1

        land_bank = getattr(getattr(self, 'population_manager', None), 'land_bank', None)
        available_land = float(getattr(land_bank, 'total_area_mu', 0.0) or 0.0)
        entry_land_trigger = max(5.0, float(dynamics_cfg.get('entry_land_trigger_mu', 60.0) or 60.0))
        entry_profit_anchor = float(dynamics_cfg.get('entry_profit_anchor', 0.02) or 0.02)
        startup_household_leasing_enabled = bool(dynamics_cfg.get('startup_household_leasing_enabled', True))

        recent_rois = [
            self._get_enterprise_recent_metric(enterprise, 'roi', 'roi_history')
            for enterprise in self.agricultural_enterprises
        ]
        mean_recent_roi = float(np.mean(recent_rois)) if recent_rois else 0.0

        desired_entries = 0
        if (
            len(self.agricultural_enterprises) < target_count
            and (available_land >= entry_land_trigger or startup_household_leasing_enabled)
        ):
            desired_entries = min(annual_entry_cap, max(0, target_count - len(self.agricultural_enterprises)))
        elif (
            len(self.agricultural_enterprises) < max_count
            and available_land >= entry_land_trigger * 1.5
            and mean_recent_roi >= entry_profit_anchor
        ):
            desired_entries = min(annual_entry_cap, 1)

        for entry_index in range(desired_entries):
            if len(self.agricultural_enterprises) >= max_count:
                break
            if (
                (land_bank is None or float(getattr(land_bank, 'total_area_mu', 0.0) or 0.0) < 5.0)
                and not startup_household_leasing_enabled
            ):
                break

            enterprise = self._spawn_dynamic_enterprise(entry_index)
            allocated_area = self._allocate_startup_land_to_enterprise(enterprise, dynamics_cfg)
            if allocated_area <= 0:
                self._rollback_spawned_enterprise(enterprise)
                break
    
    def _get_farmer_service_demand(self, farmer):
        """
        @theory 判断农户是否有服务需求
        
        触发条件：
        1. 劳动力不足（如老年农户、兼业户）
        2. 有足够资金购买服务
        3. 成本效益可行
        """
        if action_income_enabled(self.config) and farmer.planned_crop is None:
            return None
        from core.p3_project_contracts import configured_program
        p3_mode = getattr(farmer, "p3_service_mode", None) if configured_program(self) is not None else None
        if p3_mode in ("self", "transfer"):
            return None
        # 获取服务决策配置 (在 production.agricultural_service 下)
        prod_config = self.config.get('production', {})
        service_config = prod_config.get('agricultural_service', {})
        if not service_config.get('enabled', False):
            return None
            
        farmer_config = service_config.get('farmer_decision', {})
        operated_area = production_area(farmer)
        if operated_area < 1:  # 少于1亩不需要
            return None

        livelihood_value = str(
            getattr(getattr(farmer, 'livelihood_strategy', None), 'value', getattr(farmer, 'livelihood_strategy', ''))
            or ''
        )
        crop_config = self.config.get('crop', {}) if isinstance(self.config, dict) else {}
        retained_service_enabled = bool(
            farmer_config.get('non_farmer_retained_service_enabled', True)
        )
        retained_service_min_area = float(
            farmer_config.get(
                'non_farmer_retained_service_min_area_mu',
                crop_config.get('non_farmer_retained_land_max_area_mu', 12.0),
            )
            or 12.0
        )
        non_farmer_retained_service = (
            retained_service_enabled
            and livelihood_value == 'Non_Farmer'
            and operated_area > retained_service_min_area
        )
        if action_income_enabled(self.config):
            non_farmer_retained_service = non_farmer_retained_service and str(getattr(farmer.planned_crop, 'value', '')) == 'Grain'

        pending_transfer_area = 0.0
        if hasattr(farmer, 'get_pending_transfer_area'):
            try:
                pending_transfer_area = max(0.0, float(farmer.get_pending_transfer_area() or 0.0))
            except (TypeError, ValueError):
                pending_transfer_area = 0.0
        custody_preference = bool(getattr(farmer, 'prefers_cooperative_custody', False)) or bool(
            getattr(farmer, 'is_weekend_farmer', False)
        )
        
        # 条件1: 劳动力稀缺性
        labor_scarcity_threshold = farmer_config.get('labor_scarcity_threshold', 0.5)
        age_threshold = farmer_config.get('age_threshold', 55)
        
        # 计算劳动力充裕度（简化模型）
        labor_sufficiency = 1.0
        if hasattr(farmer, 'age'):
            if farmer.age > age_threshold:
                labor_sufficiency *= 0.5  # 高龄降低劳动能力
        if hasattr(farmer, 'livelihood_strategy'):
            if farmer.livelihood_strategy.value == 'Pluriactive':
                labor_sufficiency *= 0.6  # 兼业户时间分配有限
        
        retention_cfg = (
            self.config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        scale_service_threshold = float(
            farmer_config.get(
                'scale_service_threshold_mu',
                retention_cfg.get('min_operated_area', 30),
            ) or 30.0
        )
        scale_service_max_bonus = float(
            farmer_config.get('scale_service_max_bonus', 0.18) or 0.18
        )
        grain_specialization_bonus = float(
            farmer_config.get('grain_specialization_service_bonus', 0.08) or 0.08
        )
        grain_policy_service_bonus = float(
            farmer_config.get('grain_policy_service_bonus', 0.0) or 0.0
        )
        policy_service_sensitivity_weight = float(
            farmer_config.get('policy_service_sensitivity_weight', 0.0) or 0.0
        )
        policy_price_floor_service_bonus = float(
            farmer_config.get('policy_price_floor_service_bonus', 0.0) or 0.0
        )
        policy_service_min_scale = float(
            farmer_config.get('policy_service_min_scale_mu', 8.0) or 8.0
        )

        scale_service_bonus = 0.0
        if operated_area >= scale_service_threshold:
            scale_span = max(scale_service_threshold, 20.0)
            scale_progress = min(1.0, max(0.0, (operated_area - scale_service_threshold) / scale_span))
            scale_service_bonus += scale_service_max_bonus * (0.45 + 0.55 * scale_progress)

        agricultural_ratio = 0.0
        if hasattr(farmer, '_get_livelihood_agricultural_income_ratio'):
            try:
                agricultural_ratio = float(farmer._get_livelihood_agricultural_income_ratio() or 0.0)
            except (TypeError, ValueError):
                agricultural_ratio = 0.0

        grain_share = 0.0
        try:
            grain_area = float(getattr(farmer, 'grain_area', 0.0) or 0.0)
            if operated_area > 0:
                grain_share = max(0.0, min(1.0, grain_area / operated_area))
        except (TypeError, ValueError):
            grain_share = 0.0

        if getattr(farmer, "household_planned_grain_share", None) is not None:
            grain_share = farmer.household_planned_grain_share
        if agricultural_ratio >= 0.85 and grain_share >= 0.50:
            scale_service_bonus += grain_specialization_bonus

        characteristics = getattr(farmer, 'characteristics', None)
        service_policy_sensitivity = float(
            getattr(characteristics, 'service_policy_sensitivity', 0.5) or 0.5
        )
        active_policy = getattr(getattr(self, 'environment', None), 'active_policy', {}) or {}
        try:
            grain_policy_signal = max(0.0, min(1.0, float(active_policy.get('grain_subsidies_rate', 0.0) or 0.0)))
        except (TypeError, ValueError):
            grain_policy_signal = 0.0
        try:
            grain_price_floor = float(active_policy.get('grain_price_floor', 0.0) or 0.0)
        except (TypeError, ValueError):
            grain_price_floor = 0.0
        if operated_area >= policy_service_min_scale and grain_share >= 0.50:
            policy_specialization = max(0.0, min(1.0, 0.5 * grain_share + 0.5 * agricultural_ratio))
            scale_service_bonus += (
                grain_policy_service_bonus
                * grain_policy_signal
                * policy_specialization
            )
            scale_service_bonus += (
                policy_service_sensitivity_weight
                * max(0.0, min(1.0, service_policy_sensitivity))
                * grain_policy_signal
            )
            if grain_price_floor > 0.0:
                scale_service_bonus += (
                    policy_price_floor_service_bonus
                    * policy_specialization
                )

        effective_labor_sufficiency = max(0.0, labor_sufficiency - scale_service_bonus)

        # 判断是否需要服务：不仅看劳动力短缺，也看规模化经营对专业服务的内生需求
        needs_service = effective_labor_sufficiency < labor_scarcity_threshold
        service_area = operated_area
        if non_farmer_retained_service:
            needs_service = True
            custody_preference = True
            effective_labor_sufficiency = min(
                effective_labor_sufficiency,
                float(
                    farmer_config.get(
                        'non_farmer_retained_service_labor_sufficiency_cap',
                        0.20,
                    )
                    or 0.20
                ),
            )
            service_area = min(
                operated_area,
                float(
                    farmer_config.get(
                        'non_farmer_retained_service_area_cap_mu',
                        operated_area,
                    )
                    or operated_area
                ),
            )
        pending_ratio = 0.0
        if pending_transfer_area > 0 and operated_area > 0:
            pending_ratio = max(0.0, min(1.0, pending_transfer_area / operated_area))
            service_area = min(operated_area, max(1.0, pending_transfer_area))
            if custody_preference or pending_ratio >= 0.15:
                needs_service = True
                effective_labor_sufficiency = min(
                    effective_labor_sufficiency,
                    0.22 if pending_ratio >= 0.55 else 0.42,
                )
        
        if not needs_service and p3_mode != "outsource":
            return None
        
        # 条件2: 确定服务类型和需求
        from modules.agricultural_service import ServiceType
        
        if pending_ratio >= 0.55 or effective_labor_sufficiency < 0.3:
            # 劳动力极度匮乏，需要全程托管
            service_type = ServiceType.FULL_TRUSTEESHIP
        elif pending_ratio >= 0.20 or effective_labor_sufficiency < 0.6:
            # 中度匮乏，关键环节托管
            service_type = ServiceType.PARTIAL_TRUSTEESHIP
        else:
            # 轻度匮乏，单一环节服务
            service_type = ServiceType.SINGLE_SERVICE
        
        return {
            'farmer': farmer,
            'area': service_area,
            'service_type': service_type,
            'max_price_per_mu': self._calculate_max_service_price(farmer, service_type),
            'service_managed_retained_grain': bool(non_farmer_retained_service),
        }
    
    def _calculate_max_service_price(self, farmer, service_type):
        """
        @theory 计算农户愿意支付的最高服务价格
        
        基于机会成本：
        - 自己种地的劳动成本
        - 外出务工的机会收入
        """
        prod_config = self.config.get('production', {})
        service_config = prod_config.get('agricultural_service', {})
        service_types = service_config.get('service_types', {})
        farmer_config = service_config.get('farmer_decision', {})
        
        if service_type.value == 'Full_Trusteeship':
            type_config = service_types.get('full_trusteeship', {})
        elif service_type.value == 'Partial_Trusteeship':
            type_config = service_types.get('partial_trusteeship', {})
        else:
            type_config = service_types.get('single_service', {})

        max_fee = float(type_config.get('max_fee', 600) or 600)
        operated_area = max(0.0, float(getattr(farmer, 'get_operated_land_area', lambda: 0.0)() or 0.0))
        grain_area = max(0.0, float(getattr(farmer, 'grain_area', 0.0) or 0.0))
        grain_share = min(1.0, grain_area / operated_area) if operated_area > 0 else 0.0
        if getattr(farmer, "household_planned_grain_share", None) is not None:
            grain_share = farmer.household_planned_grain_share
        active_policy = getattr(getattr(self, 'environment', None), 'active_policy', {}) or {}
        try:
            grain_policy_signal = max(0.0, min(1.0, float(active_policy.get('grain_subsidies_rate', 0.0) or 0.0)))
        except (TypeError, ValueError):
            grain_policy_signal = 0.0
        affordability_bonus = float(
            farmer_config.get('policy_affordability_bonus', 0.0) or 0.0
        )
        if grain_share >= 0.50 and grain_policy_signal > 0.0:
            max_fee *= (1.0 + affordability_bonus * grain_policy_signal * grain_share)

        # 使用配置的最大价格
        return max_fee
    
    def _apply_service_contract(self, match):
        """
        @theory 应用服务合同，更新农户和服务商状态
        @new 2025-12-04: 添加服务市场统计数据收集
        """
        farmer = match.get('farmer')
        provider = match.get('provider')
        contract = match.get('contract')
        
        if farmer is None or provider is None or contract is None:
            return

        if preproduction_cash_enabled(self.config):
            budget = farmer.last_preproduction_cash_account
            fee = float(contract.fee_per_mu * contract.area)
            if fee < 0 or not np.isfinite(fee) or fee > budget['uncommitted_yuan'] + 1e-6:
                raise RuntimeError('Service contract exceeds preproduction cash offer')
            budget['service_reserved_yuan'] += fee
            budget['committed_yuan'] += fee
            budget['uncommitted_yuan'] -= fee

        from core.p3_project_contracts import P3AccountingError, configured_program
        p3_program = configured_program(self)
        p3_service_result = None
        if p3_program is not None:
            from core.p3_service_settlement import prepare_service
            prepare_service(farmer, provider, contract)
            p3_service_result = provider.provide_service(contract)
            expected_fee = contract.fee_per_mu * contract.area
            if not p3_service_result or abs(float(p3_service_result.get('service_fee', -1)) - expected_fee) > 1e-6:
                raise P3AccountingError("service execution has no matching fulfilled fee")
            contract.p3_fulfilled = True

        service_managed_retained_grain = bool(match.get('service_managed_retained_grain', False))
        if service_managed_retained_grain:
            try:
                from agents.farmer_agent import CropType
            except ImportError:
                CropType = None
            if CropType is not None:
                farmer.planned_crop = CropType.GRAIN
            farmer.service_managed_retained_grain = True
        
        # 1. 更新农户状态
        if hasattr(farmer, 'service_contracts'):
            farmer.service_contracts.append(contract)
        else:
            farmer.service_contracts = [contract]
        
        # 2. 应用服务效果
        # - 减少劳动投入需求
        # - 可能提高产量
        if hasattr(farmer, 'labor_reduction_ratio'):
            farmer.labor_reduction_ratio = contract.coverage_ratio * 0.85
            if action_income_enabled(self.config):
                farmer.labor_reduction_ratio *= min(1.0, contract.area / max(production_area(farmer), 1e-8))
        service_result = p3_service_result or {}
        provide_service = getattr(provider, 'provide_service', None)
        if p3_service_result is None and callable(provide_service):
            try:
                service_result = provide_service(contract) or {}
            except Exception:
                service_result = {}
        raw_yield_bonus = float(service_result.get('yield_bonus', 1.0) or 1.0)
        service_config = (
            self.config.get('production', {})
            .get('agricultural_service', {})
            if isinstance(getattr(self, 'config', {}), dict)
            else {}
        )
        apply_direct_yield_bonus = bool(
            service_config.get('apply_direct_yield_bonus_multiplier', False)
        )
        effective_yield_bonus = 1.0
        if apply_direct_yield_bonus:
            effective_yield_bonus = 1.0 + min(0.15, max(0.0, raw_yield_bonus - 1.0)) * float(
                getattr(contract, 'coverage_ratio', 0.0) or 0.0
            )
        current_yield_bonus = float(getattr(farmer, 'service_yield_bonus_multiplier', 1.0) or 1.0)
        farmer.service_yield_bonus_multiplier = min(
            1.18,
            max(1.0, current_yield_bonus, effective_yield_bonus),
        )
        farmer.service_direct_yield_bonus_suppressed = bool(
            not apply_direct_yield_bonus and raw_yield_bonus > 1.0
        )
        farmer.service_direct_yield_bonus_raw = raw_yield_bonus
        if self.labor_market is not None and hasattr(self.labor_market, 'register_service_substitution'):
            from core.labor_market_v2 import ServiceSubstitutionInfo

            raw_service_type = str(getattr(contract, 'service_type', '') or '').lower()
            if 'full' in raw_service_type:
                normalized_service_type = 'full'
            elif 'partial' in raw_service_type:
                normalized_service_type = 'partial'
            else:
                normalized_service_type = 'single'

            service_info = ServiceSubstitutionInfo(
                farmer_id=str(farmer.unique_id),
                service_coverage_ratio=float(getattr(contract, 'coverage_ratio', 0.0) or 0.0),
                service_type=normalized_service_type,
                labor_saved_days=0.0,
            )
            self.labor_market.register_service_substitution(str(farmer.unique_id), service_info)
        
        # 3. 扣除服务费用
        service_fee = contract.fee_per_mu * contract.area
        if p3_program is None and hasattr(farmer, 'cash'):
            farmer.cash -= service_fee
        farmer.service_fee_expense = float(
            getattr(farmer, 'service_fee_expense', 0.0) or 0.0
        ) + float(service_fee or 0.0)
        
        # 4. 更新服务商状态
        if hasattr(provider, 'add_service_contract'):
            provider.add_service_contract(contract)
        
        # @new 2025-12-04: 收集统计数据
        self._update_service_market_stats(
            contract,
            provider,
            service_managed_retained_grain=service_managed_retained_grain,
        )
    
    def _update_service_market_stats(self, contract, provider, *, service_managed_retained_grain=False):
        """
        @new 2025-12-04: 更新服务市场统计数据
        
        跟踪指标：
        - 总合同数
        - 总服务面积
        - 总服务费用
        - 按服务类型分类的合同数
        - 服务提供商利用率
        """
        # 更新基本统计
        contract_area = float(getattr(contract, 'area', 0) or 0.0)
        self.service_market_stats['total_contracts'] += 1
        self.service_market_stats['total_service_area'] += contract_area
        service_fee = getattr(contract, 'fee_per_mu', 0) * contract_area
        self.service_market_stats['total_service_fee'] += service_fee
        if service_managed_retained_grain:
            self.service_market_stats['service_managed_retained_grain_area'] = (
                float(self.service_market_stats.get('service_managed_retained_grain_area', 0.0) or 0.0)
                + contract_area
            )
            self.service_market_stats['service_managed_retained_grain_contracts'] = (
                int(self.service_market_stats.get('service_managed_retained_grain_contracts', 0) or 0)
                + 1
            )
        
        # 按类型统计
        service_type = str(getattr(contract, 'service_type', 'Unknown'))
        if service_type not in self.service_market_stats['contracts_by_type']:
            self.service_market_stats['contracts_by_type'][service_type] = 0
        self.service_market_stats['contracts_by_type'][service_type] += 1
        
        # 更新提供商利用率（在步骤结束时汇总）
    
    def get_service_market_statistics(self):
        """
        @new 2025-12-04: 获取服务市场统计数据
        
        返回：
        - 当前步骤的服务市场统计摘要
        - 用于数据收集器和分析
        """
        # 计算提供商利用率
        provider_utilization = []
        for provider in self.service_providers:
            if hasattr(provider, 'get_utilization_rate'):
                util_rate = provider.get_utilization_rate()
            else:
                # 备用计算
                capacity = getattr(provider, 'max_service_area', 1000)
                current = getattr(provider, 'current_service_area', 0)
                util_rate = current / capacity if capacity > 0 else 0
            provider_utilization.append(util_rate)
        
        avg_utilization = sum(provider_utilization) / len(provider_utilization) if provider_utilization else 0
        
        return {
            'total_contracts': self.service_market_stats['total_contracts'],
            'total_service_area': self.service_market_stats['total_service_area'],
            'total_service_fee': self.service_market_stats['total_service_fee'],
            'contracts_by_type': self.service_market_stats['contracts_by_type'].copy(),
            'provider_count': len(self.service_providers),
            'average_provider_utilization': avg_utilization,
            'provider_utilization_list': provider_utilization,
            'provider_profit_total': self.service_market_stats.get('provider_profit_total', 0.0),
            'provider_profit_mean': self.service_market_stats.get('provider_profit_mean', 0.0),
            'service_managed_retained_grain_area': self.service_market_stats.get(
                'service_managed_retained_grain_area', 0.0
            ),
            'service_managed_retained_grain_contracts': self.service_market_stats.get(
                'service_managed_retained_grain_contracts', 0
            ),
            'service_demand_count': self.service_market_stats.get('service_demand_count', 0),
            'service_demand_area': self.service_market_stats.get('service_demand_area', 0.0),
            'service_matched_area': self.service_market_stats.get('service_matched_area', 0.0),
            'service_unmatched_demand_area': self.service_market_stats.get(
                'service_unmatched_demand_area', 0.0
            ),
            'service_unmatched_demand_count': self.service_market_stats.get(
                'service_unmatched_demand_count', 0
            ),
            'service_retained_grain_demand_area': self.service_market_stats.get(
                'service_retained_grain_demand_area', 0.0
            ),
            'service_retained_grain_matched_area': self.service_market_stats.get(
                'service_retained_grain_matched_area', 0.0
            ),
            'service_retained_grain_unmatched_area': self.service_market_stats.get(
                'service_retained_grain_unmatched_area', 0.0
            ),
        }
    
    def _reset_service_market_stats(self):
        """
        @new 2025-12-04: 在每个步骤开始时重置服务市场统计
        """
        for farmer in getattr(self, 'farmers', []) or []:
            farmer.service_contracts = []
            farmer.service_managed_retained_grain = False
            farmer.labor_reduction_ratio = 0.0
            farmer.service_yield_bonus_multiplier = 1.0
            farmer.service_direct_yield_bonus_suppressed = False
            farmer.service_direct_yield_bonus_raw = 1.0
            farmer.service_fee_expense = 0.0
        if self.labor_market is not None and hasattr(self.labor_market, 'clear_service_substitution_cache'):
            self.labor_market.clear_service_substitution_cache()

        self.service_market_stats = {
            'total_contracts': 0,
            'total_service_area': 0,
            'total_service_fee': 0,
            'contracts_by_type': {},
            'provider_utilization': [],
            'service_managed_retained_grain_area': 0.0,
            'service_managed_retained_grain_contracts': 0,
            'service_demand_count': 0,
            'service_demand_area': 0.0,
            'service_matched_area': 0.0,
            'service_unmatched_demand_area': 0.0,
            'service_unmatched_demand_count': 0,
            'service_retained_grain_demand_area': 0.0,
            'service_retained_grain_matched_area': 0.0,
            'service_retained_grain_unmatched_area': 0.0,
        }
    
    def _process_farmer_entry(self):
        """
        @new 2025-12-26: 新农户进入机制
        
        @theory 返乡创业与代际更替
        
        问题诊断：
        - 模型中只有农户退出，没有新农户进入
        - 导致30年后纯农户比例→0%，不符合现实
        
        理论依据：
        - 《乡村振兴战略规划(2018-2022)》鼓励返乡创业
        - 2015-2022年返乡入乡创业人员累计超1200万人
        - 每年约2%的农村劳动力返乡从事农业经营
        
        实现逻辑：
        1. 根据配置的年度进入率计算新增农户数
        2. 创建新农户（年轻、高能力、偏向纯农户）
        3. 分配初始土地（可能来自撂荒地恢复）
        """
        # 获取配置
        farmer_config = self.config.get('agents', {}).get('farmer', {})
        evolution_config = farmer_config.get('livelihood_evolution', {})
        entry_config = evolution_config.get('farmer_entry', {})
        
        if not entry_config.get('enabled', False):
            return
        
        # 计算新增农户数
        annual_entry_rate = entry_config.get('annual_entry_rate', 0.02)
        min_farmer_count = entry_config.get('min_farmer_count', 100)
        
        current_farmer_count = len(self.farmers)
        
        # 确保农户数量不低于下限
        natural_entry = int(current_farmer_count * annual_entry_rate)
        
        # 如果农户数量低于下限，强制补充
        if current_farmer_count < min_farmer_count:
            forced_entry = min_farmer_count - current_farmer_count
            new_farmer_count = max(natural_entry, forced_entry)
        else:
            new_farmer_count = natural_entry
        
        if new_farmer_count <= 0:
            return
        
        # 创建新农户
        from agents.farmer_agent import FarmerAgent, LivelihoodStrategy
        
        # 年龄范围
        age_range = entry_config.get('entry_age_range', [28, 45])
        
        # 策略分布
        strategy_dist = entry_config.get('entry_strategy_distribution', {
            'Pure_Farmer': 0.40,
            'Pluriactive': 0.50,
            'Non_Farmer': 0.10
        })
        
        for i in range(new_farmer_count):
            # 生成唯一ID
            new_id = f"farmer_{len(self.farmers)}_{self.step_count}_{i}"
            
            # 年龄
            age = rng_randint(self, age_range[0], age_range[1])
            
            # 生计策略（加权随机选择）
            strategy_choice = rng_random(self)
            cumulative = 0
            chosen_strategy = LivelihoodStrategy.PLURIACTIVE  # 默认
            
            for strat_name, prob in strategy_dist.items():
                cumulative += prob
                if strategy_choice <= cumulative:
                    if strat_name == 'Pure_Farmer':
                        chosen_strategy = LivelihoodStrategy.PURE_FARMER
                    elif strat_name == 'Pluriactive':
                        chosen_strategy = LivelihoodStrategy.PLURIACTIVE
                    else:
                        chosen_strategy = LivelihoodStrategy.NON_FARMER
                    break
            
            # 初始资本（返乡创业者通常有一定积蓄）
            initial_capital = float(np_lognormal(self, 9.5, 0.4))  # 略高于平均水平
            
            # 农业能力（年轻人受教育程度高，学习能力强）
            farming_ability = float(np_uniform(self, 1.0, 1.3))  # 略高于平均
            
            # 创建空土地列表（新进入者需要通过土地市场获取土地）
            land_patches = []
            
            # 创建农户
            new_farmer = FarmerAgent(
                new_id,
                self,
                land_patches,
                initial_capital=initial_capital,
                farming_ability=farming_ability,
                config=self.config
            )
            
            new_farmer.age = age
            new_farmer.livelihood_strategy = chosen_strategy
            new_farmer._initialize_labor_allocation()
            
            # 添加到模型
            self.farmers.append(new_farmer)
            self.schedule.add(new_farmer)
            self.agent_registry[new_farmer.unique_id] = new_farmer
            
            # 放置在网格中
            x = rng_randint(self, 0, self.grid.width - 1)
            y = rng_randint(self, 0, self.grid.height - 1)
            self.grid.place_agent(new_farmer, (x, y))
    
    def _update_ecological_feedback(self):
        """
        @new 2025-01-02: 生态反馈机制更新
        
        @theory CHANS (Coupled Human and Natural Systems)
        Nature级论文需要人类-自然耦合系统的建模
        
        核心机制：
        1. 高强度农业生产导致土壤质量退化
        2. 土壤退化导致边际产量递减
        3. 存在不可逆临界点（土壤质量<0.3恢复极慢）
        4. 休耕和轮作可以恢复土壤质量
        
        这产生关键涌现：
        - 短期：高补贴 → 高投入 → 高产量
        - 长期：过度投入 → 土壤退化 → 财政投入增加但效果递减
        """
        if not self.land_patches:
            return
        
        # 获取生态配置参数
        ecological_config = self.config.get('production', {}).get('ecological', {})
        if not ecological_config:
            ecological_config = (
                self.config.get('agents', {})
                .get('fiscal_system', {})
                .get('ecological', {})
            )
        enabled = ecological_config.get('enabled', True)
        
        if not enabled:
            return
        
        # 获取配置参数
        # Prefer the environment-climate runtime value so sensitivity shocks on
        # `environment.climate.soil_degradation_rate` propagate into patch quality.
        env_degradation_rate = None
        if hasattr(self, 'environment') and self.environment is not None:
            try:
                env_degradation_rate = float(getattr(self.environment, 'soil_degradation_rate', None))
            except (TypeError, ValueError):
                env_degradation_rate = None
        if env_degradation_rate is not None and env_degradation_rate > 0:
            base_degradation_rate = env_degradation_rate
        else:
            base_degradation_rate = ecological_config.get('base_degradation_rate', 0.02)  # 2%/年
        overuse_degradation_coef = ecological_config.get('fertilizer_degradation_coef', 0.15)
        continuous_cropping_penalty = ecological_config.get('continuous_cropping_penalty', 0.03)
        continuous_penalty_cap_years = int(
            ecological_config.get('continuous_penalty_cap_years', 6) or 6
        )
        continuous_penalty_max = float(
            ecological_config.get(
                'continuous_penalty_max',
                max(0.0, continuous_cropping_penalty * max(0, continuous_penalty_cap_years - 3)),
            )
            or 0.0
        )
        active_management_recovery_rate = float(
            ecological_config.get('active_management_recovery_rate', 0.0) or 0.0
        )
        baseline_regeneration_rate = float(
            ecological_config.get('baseline_regeneration_rate', 0.0) or 0.0
        )
        recovery_gap_bonus = float(
            ecological_config.get('recovery_gap_bonus', 0.0) or 0.0
        )
        service_management_recovery_bonus = float(
            ecological_config.get('service_management_recovery_bonus', 0.0) or 0.0
        )
        professional_management_recovery_bonus = float(
            ecological_config.get('professional_management_recovery_bonus', 0.0) or 0.0
        )
        managed_quality_ceiling_bonus = float(
            ecological_config.get('managed_quality_ceiling_bonus', 0.0) or 0.0
        )
        fallow_recovery_rate = ecological_config.get('fallow_recovery_rate', 0.08)  # 8%/年
        critical_threshold = ecological_config.get('critical_threshold', 0.3)
        critical_recovery_penalty = ecological_config.get('critical_recovery_penalty', 0.5)
        patch_quality_floor = max(
            0.10,
            float(
                ecological_config.get(
                    'soil_quality_floor',
                    getattr(getattr(self, 'environment', None), 'soil_quality_floor', 0.20),
                ) or 0.20
            ),
        )
        
        # 遍历所有地块，更新生态状态
        for patch in self.land_patches:
            # 获取当前土壤质量
            current_quality = getattr(patch, 'soil_quality', 1.0)
            initial_quality = getattr(patch, 'initial_soil_quality', current_quality)
            fertilizer_intensity = getattr(patch, 'fertilizer_intensity', 1.0)
            consecutive_years = getattr(patch, 'consecutive_cropping_years', 0)
            
            # 检查是否休耕
            is_fallow = getattr(patch, 'abandoned', False)
            
            if is_fallow:
                # 休耕恢复逻辑
                recovery_rate = fallow_recovery_rate
                
                # 临界退化后恢复减速
                if current_quality < critical_threshold:
                    recovery_rate *= critical_recovery_penalty
                 
                # 向初始质量恢复（但不能超过初始值）
                quality_ceiling = min(
                    1.0,
                    max(float(initial_quality or current_quality), float(current_quality)) + managed_quality_ceiling_bonus,
                )
                new_quality = min(quality_ceiling, max(patch_quality_floor, current_quality + recovery_rate))
                
                # 重置连作计数
                patch.consecutive_cropping_years = 0
            else:
                # 正常耕作：计算退化
                degradation = base_degradation_rate
                
                # 过量施肥退化
                if fertilizer_intensity > 1.0:
                    overuse_ratio = fertilizer_intensity - 1.0
                    degradation += overuse_degradation_coef * overuse_ratio
                
                # 连作惩罚（3年以上）
                if consecutive_years > 3:
                    capped_years = min(
                        max(consecutive_years, 0),
                        max(continuous_penalty_cap_years, 3),
                    )
                    incremental_penalty = continuous_cropping_penalty * max(0, capped_years - 3)
                    degradation += min(continuous_penalty_max, incremental_penalty)

                # 低质量土壤退化加速（正反馈）
                if current_quality < 0.5:
                    degradation *= 1.5

                operator = (
                    getattr(patch, 'current_operator', None)
                    or getattr(patch, 'leased_to', None)
                    or getattr(patch, 'owner', None)
                )
                quality_gap = max(0.0, float(initial_quality or current_quality) - float(current_quality))
                management_recovery = (
                    baseline_regeneration_rate
                    + active_management_recovery_rate
                    + recovery_gap_bonus * quality_gap
                )

                if operator is not None and hasattr(operator, 'enterprise_type'):
                    management_recovery += professional_management_recovery_bonus
                elif operator is not None and is_professional_large_operator(operator):
                    management_recovery += professional_management_recovery_bonus

                service_coverage = 0.0
                for contract in getattr(operator, 'service_contracts', []) or []:
                    try:
                        coverage = float(getattr(contract, 'coverage_ratio', 0.0) or 0.0)
                    except (TypeError, ValueError):
                        coverage = 0.0
                    service_coverage = max(service_coverage, coverage)
                management_recovery += service_management_recovery_bonus * max(0.0, min(1.0, service_coverage))

                net_change = management_recovery - degradation
                quality_ceiling = max(float(initial_quality or current_quality), float(current_quality))
                if operator is not None:
                    quality_ceiling = min(1.0, quality_ceiling + managed_quality_ceiling_bonus)
                new_quality = min(
                    quality_ceiling,
                    max(patch_quality_floor, current_quality + net_change),
                )
                
                # 更新连作年数
                patch.consecutive_cropping_years = consecutive_years + 1
            
            # 更新土壤质量
            patch.soil_quality = new_quality
            
            # 记录历史（保留最近10年）
            if hasattr(patch, 'soil_quality_history'):
                patch.soil_quality_history.append(new_quality)
                if len(patch.soil_quality_history) > 10:
                    patch.soil_quality_history = patch.soil_quality_history[-10:]
            else:
                patch.soil_quality_history = [new_quality]

        if hasattr(self, 'environment') and self.environment is not None:
            patch_qualities = [
                float(getattr(patch, 'soil_quality', 0.0) or 0.0)
                for patch in self.land_patches
                if getattr(patch, 'soil_quality', None) is not None
            ]
            if patch_qualities:
                self.environment.soil_quality = float(np.mean(patch_qualities))

    def _create_farmer_from_features(self, features: dict):
        """
        @new 2026-01-07: 从特征字典创建新农户Agent
        
        用于人口动态中新进入者的创建
        
        Args:
            features: 包含农户属性的字典（由PopulationManagerV2生成）
        
        Returns:
            FarmerAgent实例或None（创建失败时）
        """
        try:
            from agents.farmer_agent import FarmerAgent

            new_id = str(features.get('unique_id') or self._generate_entrant_id())

            livelihood_raw = (
                features.get('livelihood')
                or features.get('livelihood_type')
                or features.get('livelihood_original')
            )
            agri_ratio = features.get('agri_ratio')
            if agri_ratio is None:
                agri_ratio = self._infer_agri_ratio_from_livelihood(livelihood_raw)
            agri_ratio = float(np.clip(agri_ratio, 0.0, 1.0))

            initial_capital = float(features.get('initial_capital', 50000.0) or 50000.0)
            total_income = float(features.get('total_income', 0.0) or 0.0)
            if total_income <= 0:
                total_income = max(12000.0, initial_capital * 0.6)

            land_area = float(features.get('land_area_mu', features.get('land_area', 10.0)) or 10.0)
            land_area = max(0.1, land_area)

            microdata_features = {
                'land_area_mu': land_area,
                'total_income': total_income,
                'agri_ratio': agri_ratio,
                'livelihood': self._normalize_livelihood_label(livelihood_raw, agri_ratio),
                'family_size': int(features.get('household_size', features.get('family_size', 3)) or 3),
                'region': features.get('region', '中部')
            }

            farmer = FarmerAgent.from_microdata(
                unique_id=new_id,
                model=self,
                microdata_features=microdata_features,
                config=self.config
            )

            allocated_parcels = features.get('land_parcels') or []
            allocated_patches = [
                patch
                for parcel in allocated_parcels
                for patch in (getattr(parcel, 'patches', []) or [])
            ]
            if allocated_patches:
                farmer.land_patches = allocated_patches

            # 用入口特征覆盖关键人口学属性
            if 'age' in features:
                farmer.age = int(np.clip(features['age'], 20, 85))
                if hasattr(farmer, 'characteristics'):
                    farmer.characteristics.household_head_age = farmer.age
            if 'education_level' in features:
                farmer.education_level = int(np.clip(features['education_level'], 0, 6))
            if 'household_size' in features:
                farmer.household_size = max(1, int(features['household_size']))

            farmer.capital = max(0.0, initial_capital)
            farmer.initial_capital = max(0.0, initial_capital)

            # 标记为新进入者
            farmer.is_new_entrant = True
            farmer.entry_year = features.get('entry_year', self.simulation_base_year + self.step_count)
            farmer.representation_weight_q = float(
                features.get('representation_weight_q', 1.0) or 1.0
            )

            # 注册地块
            for patch in getattr(farmer, 'land_patches', []):
                patch.owner = farmer
                if getattr(patch, 'leased_to', None) is None:
                    patch.current_operator = farmer
                else:
                    patch.current_operator = patch.leased_to
                if patch not in self.land_patches:
                    self.land_patches.append(patch)
            if hasattr(farmer, 'sync_owned_land_area'):
                farmer.sync_owned_land_area()
            self.invalidate_land_tenure_caches()
            
            # 添加到调度器和注册表
            if hasattr(self.schedule, 'agents') and farmer not in self.schedule.agents:
                self.schedule.add(farmer)
            elif not hasattr(self.schedule, 'agents'):
                self.schedule.add(farmer)
            self.agent_registry[new_id] = farmer

            # 放置到网格（若未放置）
            if getattr(farmer, 'pos', None) is None:
                x = int(np_randint(self, 0, self.grid.width))
                y = int(np_randint(self, 0, self.grid.height))
                self.grid.place_agent(farmer, (x, y))

            return farmer
            
        except Exception as e:
            import logging
            logging.warning(f"创建新农户失败: {e}")
            return None

    def _generate_entrant_id(self):
        """为新进入农户生成唯一ID。"""
        base = f"entrant_{self.simulation_base_year + self.step_count}"
        suffix = 0
        while True:
            candidate = f"{base}_{suffix}"
            if candidate not in self.agent_registry:
                return candidate
            suffix += 1

    def _normalize_livelihood_label(self, livelihood_raw, agri_ratio: float):
        """统一生计类型标签到FarmerAgent.from_microdata可识别格式。"""
        if isinstance(livelihood_raw, str):
            label = livelihood_raw.strip().lower()
            mapping = {
                'pure_farmer': 'Pure_Farmer',
                'pure farmer': 'Pure_Farmer',
                'pluriactive': 'Pluriactive',
                'non_farmer': 'Non_Farmer',
                'non farmer': 'Non_Farmer',
                '纯农型': 'Pure_Farmer',
                '兼业型': 'Pluriactive',
                '非农型': 'Non_Farmer'
            }
            if label in mapping:
                return mapping[label]

        if agri_ratio >= 0.6:
            return 'Pure_Farmer'
        if agri_ratio >= 0.2:
            return 'Pluriactive'
        return 'Non_Farmer'

    def _infer_agri_ratio_from_livelihood(self, livelihood_raw):
        """从生计标签推断农业收入占比，避免使用固定硬编码分组。"""
        if isinstance(livelihood_raw, str):
            label = livelihood_raw.strip().lower()
            mapping = {
                'pure_farmer': 0.85,
                'pure farmer': 0.85,
                'pluriactive': 0.45,
                'non_farmer': 0.10,
                'non farmer': 0.10,
                '纯农型': 0.85,
                '兼业型': 0.45,
                '非农型': 0.10
            }
            if label in mapping:
                return mapping[label]
        return 0.45

    def _synchronize_farmers_after_population_update(self, previous_farmers, updated_farmers):
        """
        同步人口动态后的农户状态，避免调度器/注册表残留退出主体。
        """
        def _record_sync_error(stage, agent, exc):
            if not hasattr(self, '_population_sync_errors'):
                self._population_sync_errors = []
            self._population_sync_errors.append({
                'step': int(getattr(self, 'step_count', 0) or 0),
                'stage': stage,
                'agent_id': getattr(agent, 'unique_id', None),
                'error': str(exc),
                'error_type': exc.__class__.__name__,
            })

        previous_by_id = {
            getattr(f, 'unique_id'): f
            for f in previous_farmers
            if hasattr(f, 'unique_id') and getattr(f, 'unique_id') is not None
        }
        current_ids = {
            getattr(f, 'unique_id')
            for f in updated_farmers
            if hasattr(f, 'unique_id') and getattr(f, 'unique_id') is not None
        }

        exited_ids = set(previous_by_id.keys()) - current_ids
        for exited_id in exited_ids:
            exited_agent = previous_by_id[exited_id]

            if hasattr(self.schedule, 'remove'):
                try:
                    self.schedule.remove(exited_agent)
                except (ValueError, KeyError):
                    pass
                except Exception as exc:
                    _record_sync_error('schedule_remove', exited_agent, exc)
                    raise RuntimeError(
                        f"population farmer sync failed during schedule_remove for {exited_id}"
                    ) from exc

            if hasattr(self, 'grid') and getattr(exited_agent, 'pos', None) is not None:
                try:
                    self.grid.remove_agent(exited_agent)
                except (ValueError, KeyError):
                    pass
                except Exception as exc:
                    _record_sync_error('grid_remove', exited_agent, exc)
                    raise RuntimeError(
                        f"population farmer sync failed during grid_remove for {exited_id}"
                    ) from exc

            self.agent_registry.pop(exited_id, None)

        for farmer in updated_farmers:
            farmer_id = getattr(farmer, 'unique_id', None)
            if hasattr(farmer, 'sync_owned_land_area'):
                try:
                    farmer.sync_owned_land_area()
                except Exception as exc:
                    _record_sync_error('sync_owned_land_area', farmer, exc)
                    raise RuntimeError(
                        f"population farmer sync failed during sync_owned_land_area for {farmer_id}"
                    ) from exc
            if farmer_id is not None:
                self.agent_registry[farmer_id] = farmer
                if hasattr(self.schedule, 'agents') and farmer not in self.schedule.agents:
                    self.schedule.add(farmer)
    
    def record_agent_exit(self, agent, reason: ExitReason, details: dict = None):
        """
        @new 2025-12-04: 记录智能体退出到墓地系统
        
        当智能体因各种原因退出模拟时调用此方法
        避免幸存者偏差，保留完整的退出信息
        
        参数：
        - agent: 退出的智能体
        - reason: 退出原因（ExitReason枚举）
        - details: 额外的退出详情
        """
        notes = ""
        if details is not None:
            notes = f"details={details!r}"

        self.graveyard.record_exit(
            agent=agent,
            exit_reason=reason,
            step=self.step_count,
            notes=notes,
        )
    
    def get_graveyard_summary(self):
        """
        @new 2025-12-04: 获取墓地统计摘要
        """
        return self.graveyard.get_exit_summary()
    
    def _update_land_lease_contracts(self, *, after_production=False):
        """
        @critical 2025-12: 多年期租赁合同管理
        专家意见：真实土地流转是3-5年期合同，非逐年议价
        
        每年度更新：
        1. 递减剩余合同年限
        2. 合同到期时处理续约或解约
        """
        expired_contracts = []
        
        for patch in self.land_patches:
            if patch.leased_to is not None:
                paid_initial = (
                    self.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode') in ('B_star_1y', 'B_star_5y')
                    and getattr(patch, 'operational_transfer_channel', '') == 'author_paid_initial_lease'
                )
                if paid_initial != after_production:
                    continue
                # 递减合同年限
                contract_still_valid = patch.update_lease_contract()
                
                if not contract_still_valid:
                    # 合同到期，需要决定是否续约
                    expired_contracts.append(patch)
        
        # 处理到期合同
        for patch in expired_contracts:
            self._handle_expired_lease_contract(patch)
    
    def _handle_expired_lease_contract(self, patch):
        """
        @theory 合同到期处理逻辑
        
        基于三权分置理论：
        - 承包权人（owner）可选择收回或续约
        - 经营权人（leased_to）可选择续约
        - 续约需双方同意
        
        简化模型：
        - 若双方都有续约意愿（基于收益比较），则续约
        - 否则合同终止，土地经营权回归承包权人
        """
        owner = patch.owner
        tenant = patch.leased_to
        if (self.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode') in ('B_star_1y', 'B_star_5y')
                and getattr(patch, 'operational_transfer_channel', '') == 'author_paid_initial_lease'):
            patch.end_lease()
            return
        
        if owner is None or tenant is None:
            patch.end_lease()
            return
        
        # 检查双方续约意愿
        owner_wants_renew = self._check_owner_renewal_willingness(patch, owner)
        tenant_wants_renew = self._check_tenant_renewal_willingness(patch, tenant)
        
        if owner_wants_renew and tenant_wants_renew:
            # 续约：重置合同年限，可能调整租金
            new_contract_years = 3  # 默认3年
            new_rent = self._negotiate_renewal_rent(patch)
            patch.set_leased(tenant, contract_years=new_contract_years, rental_price=new_rent)  # @fix: 正确的方法名
        else:
            # 不续约：合同终止
            patch.end_lease()
    
    def _check_owner_renewal_willingness(self, patch, owner):
        """
        @theory 土地所有者续约意愿
        
        考虑因素：
        - 租金收入是否满意
        - 是否有自己种植的需求
        """
        # 简化逻辑：如果租金高于保留价，愿意续约
        from agents.farmer_agent import FarmerAgent
        if isinstance(owner, FarmerAgent):
            # 农户的续约意愿取决于生计策略
            if owner.livelihood_strategy.value == 'Non_Farmer':
                return True  # 非农户愿意继续出租
            elif owner.livelihood_strategy.value == 'Pure_Farmer':
                return False  # 纯农户倾向收回自己种
            else:
                # 兼业户根据收益比较
                return patch.rental_price >= patch.area * 200  # 200元/亩以上续约
        return True  # 默认愿意续约
    
    def _check_tenant_renewal_willingness(self, patch, tenant):
        """
        @theory 承租人续约意愿
        
        考虑因素：
        - 经营这块地的利润是否满意
        - 是否有其他更好的选择
        """
        # 简化逻辑：如果有经营能力且利润满意，愿意续约
        from agents.enterprise_agent import AgriculturalEnterpriseAgent  # @fix: 正确的模块名
        from agents.farmer_agent import FarmerAgent
        
        if isinstance(tenant, AgriculturalEnterpriseAgent):  # @fix: 正确的类名
            recent_profit = float(getattr(tenant, 'profit', 0.0) or 0.0)
            profit_history = getattr(tenant, 'profit_history', None) or []
            if profit_history:
                recent_profit = float(sum(profit_history[-2:]) / min(2, len(profit_history)))

            recent_roi = float(getattr(tenant, 'roi', 0.0) or 0.0)
            roi_history = getattr(tenant, 'roi_history', None) or []
            if roi_history:
                recent_roi = float(sum(roi_history[-2:]) / min(2, len(roi_history)))

            max_rent = 0.0
            if hasattr(tenant, '_calculate_bid_price'):
                max_rent = float(tenant._calculate_bid_price(patch) or 0.0)
            if max_rent <= 0:
                return False

            target_roi = float(getattr(tenant, 'target_return_rate', 0.0) or 0.0)
            if recent_profit < 0 and recent_roi < max(0.02, target_roi * 0.5):
                return patch.rental_price <= max_rent * 0.9
            return patch.rental_price <= max_rent * 1.05
        elif isinstance(tenant, FarmerAgent):
            recent_net_income = 0.0
            if hasattr(tenant, '_get_recent_net_income_signal'):
                recent_net_income = float(tenant._get_recent_net_income_signal() or 0.0)

            max_rent = 0.0
            if hasattr(tenant, 'calculate_max_willing_to_pay_rent'):
                max_rent = float(tenant.calculate_max_willing_to_pay_rent(patch) or 0.0)
            if max_rent <= 0:
                return False

            if tenant.livelihood_strategy.value == 'Pure_Farmer':
                rent_tolerance = 0.95 if recent_net_income < 0 else 1.05
                return patch.rental_price <= max_rent * rent_tolerance
            return recent_net_income > 0 and patch.rental_price <= max_rent
        return True
    
    def _negotiate_renewal_rent(self, patch):
        """
        @theory 续约租金协商
        
        考虑因素：
        - 原租金水平
        - 市场租金变化
        - 土地质量变化
        """
        original_rent = patch.rental_price
        # 续约时根据市场情况调整5%-10%
        adjustment = rng_uniform(self, -0.05, 0.10)
        new_rent = original_rent * (1 + adjustment)
        return max(new_rent, patch.area * 150)  # 至少150元/亩

    def conduct_land_rental_market(self):
        """
        @critical 2025-01: 改进土地流转市场机制
        
        问题诊断：之前的实现中，供需覆盖的地块不重叠：
        - 供给方只提交愿意出租的地块
        - 需求方对所有可租赁地块提交需求
        - 导致大量供需无法匹配
        
        修复：
        1. 农户先提交供给
        2. 收集已供给的地块列表
        3. 企业只对已供给的地块提交需求
        """
        # 清空清算所
        self.land_clearing_house = LandClearingHouse()
        self.land_clearing_house.model = self
        if hasattr(self.land_clearing_house, 'build_spatial_index'):
            self.land_clearing_house.build_spatial_index(self.land_patches)
        
        # 阶段1：让所有农户提交供给意愿
        for farmer in sorted(self.farmers, key=lambda item: str(getattr(item, 'unique_id', item))):
            farmer.participate_land_rental_market()

        # 村集体整合地在 collective stage 已形成，但 clearing house 在此处被重建，需要补登记一次。
        self._register_collective_supply_into_clearing_house()
        
        # 阶段2：收集已提交供给的地块，供企业参考
        # @critical: 这是关键改进 - 让企业知道哪些地块在市场上
        supplied_patches = {}
        for offer in self.land_clearing_house.supply_offers:
            patch = offer['patch']
            if hasattr(patch, 'get_patch_id'):
                patch_id = patch.get_patch_id()
            else:
                patch_id = str(getattr(patch, 'patch_id', getattr(patch, 'coordinates', 'patch')))
            supplied_patches[patch_id] = patch
        available_patches = [supplied_patches[patch_id] for patch_id in sorted(supplied_patches)]
        if action_income_enabled(self.config):
            for farmer in sorted(self.farmers, key=lambda actor: str(actor.unique_id)):
                area = farmer.get_operated_land_area()
                target = getattr(farmer, 'target_operated_area', farmer._compute_target_operated_area())
                max_added = min(max(0.0, target - area), farmer._get_expansion_area_cap(area))
                submit_action_income_bids(farmer, max_added)
        
        # 阶段3：企业对已供给的地块提交需求
        for enterprise in sorted(self.agricultural_enterprises, key=lambda item: str(getattr(item, 'unique_id', item))):
            enterprise.participate_land_rental_market(available_patches=available_patches)
        
        # 市场出清 - @fix 2026-01-14: 传入config以支持消融实验机制开关
        self.land_clearing_house.clear_market(model_config=self.config)
        
        # @fix 2025-12-18: 更新环境中的土地租金市场价格
        # 用于农户决策时参考市场租金
        market_stats = self.land_clearing_house.get_market_statistics()
        average_rent_per_mu = float(
            market_stats.get('average_rent_per_mu', market_stats.get('average_price', 0.0)) or 0.0
        )
        if average_rent_per_mu > 0:
            self.environment.land_rental_price = average_rent_per_mu
            if hasattr(self.environment, 'market_prices'):
                self.environment.market_prices['land_rent'] = average_rent_per_mu
                self.environment.market_prices['land_rental'] = average_rent_per_mu
        else:
            average_rent_per_mu = self._get_operational_transfer_rent_per_mu()
            if average_rent_per_mu > 0:
                self.environment.land_rental_price = average_rent_per_mu
                if hasattr(self.environment, 'market_prices'):
                    self.environment.market_prices['land_rent'] = average_rent_per_mu
                    self.environment.market_prices['land_rental'] = average_rent_per_mu

        total_area = float(sum(getattr(patch, 'area', 0.0) for patch in self.land_patches) or 0.0)
        leased_area = self._get_market_transferred_area()
        baseline_area = float(
            self.transfer_rate_control.get('baseline_total_land_area', total_area) or total_area or 0.0
        )
        observed_transfer_rate = leased_area / max(baseline_area, 1.0)
        self.transfer_rate_control['observed_total_land_area'] = total_area
        self.transfer_rate_control['observed_current_transferred_area'] = leased_area
        self.transfer_rate_control['current_transfer_rate'] = observed_transfer_rate
        if hasattr(self, 'environment') and self.environment is not None:
            self.environment.current_transfer_rate = observed_transfer_rate

    def get_agent_by_id(self, agent_id):
        """
        @theory
        根据ID获取智能体 - 支持社会学习机制
        
        @fix 2024-11-20: 使用agent_registry字典优化性能（O(N)->O(1)）
        专家审阅意见：原来的O(N)查找在1000+ agents时会极慢
        """
        # 使用字典查找，O(1)时间复杂度
        return self.agent_registry.get(agent_id, None)

    def get_region_label(self, actor):
        """
        @theory
        基于网格Y坐标划分南北，供政府和政策靶向共享，确保地域判断一致。
        """
        if actor is None or not hasattr(actor, 'pos') or actor.pos is None:
            return 'global'
        _, y = actor.pos
        height = max(1, getattr(self.grid, 'height', 1))
        if y < height / 3:
            return 'north'
        if y < (2 * height) / 3:
            return 'central'
        return 'south'

    def get_available_patches(self):
        """获取可租赁的地块"""
        available = []
        for patch in self.land_patches:
            if patch.owner and patch.leased_to is None:
                available.append(patch)
        return available

    # 数据收集函数 - 保持不变
    def get_land_operated_by_enterprises(self):
        """企业经营土地总面积"""
        total_area = 0
        for enterprise in self.agricultural_enterprises:
            total_area += enterprise.get_operated_land_area()
        return total_area

    def get_land_distribution_by_livelihood(self):
        """按农户生计类型划分的土地经营面积分布"""
        distribution = {
            'Pure_Farmer': 0,
            'Pluriactive': 0,
            'Non_Farmer': 0
        }
        
        for farmer in self.farmers:
            area = farmer.get_operated_land_area()
            strategy = farmer.livelihood_strategy.value
            distribution[strategy] += area
        
        return distribution

    def get_gini_coefficient_land(self):
        """土地经营规模的基尼系数"""
        areas = [farmer.get_operated_land_area() for farmer in self.farmers]
        areas.extend([enterprise.get_operated_land_area() for enterprise in self.agricultural_enterprises])
        
        if not areas or sum(areas) == 0:
            return 0
        
        areas = sorted(areas)
        n = len(areas)
        cumsum = np.cumsum(areas)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _safe_capitalization_ratio(self, capitalization_loss: float, raw_subsidy: float) -> float:
        """Convert formal bookkeeping into a bounded capitalization ratio."""
        if raw_subsidy <= 0:
            return 0.0
        return float(min(1.0, max(0.0, capitalization_loss / raw_subsidy)))

    def get_annual_subsidy_incidence_ratio(self):
        """Annual grain-subsidy capitalization rate."""
        environment = getattr(self, 'environment', None)
        if environment is None:
            return 0.0

        annual_loss = float(getattr(environment, 'annual_capitalization_loss', 0.0) or 0.0)
        annual_raw_subsidy = float(getattr(environment, 'annual_raw_basic_subsidy', 0.0) or 0.0)
        return self._safe_capitalization_ratio(annual_loss, annual_raw_subsidy)

    def get_cumulative_subsidy_incidence_ratio(self):
        """Cumulative grain-subsidy capitalization rate."""
        environment = getattr(self, 'environment', None)
        if environment is None:
            return 0.0

        cumulative_loss = float(getattr(environment, 'cumulative_capitalization_loss', 0.0) or 0.0)
        cumulative_raw_subsidy = float(getattr(environment, 'cumulative_raw_basic_subsidy', 0.0) or 0.0)
        return self._safe_capitalization_ratio(cumulative_loss, cumulative_raw_subsidy)

    def get_subsidy_incidence_ratio(self):
        """补贴资本化程度（兼容接口：返回年度口径）。"""
        return self.get_annual_subsidy_incidence_ratio()

    def get_income_gini_coefficient(self):
        """农村内部的收入基尼系数"""
        incomes = [get_farmer_net_income(farmer) for farmer in self.farmers]
        
        if not incomes:
            return 0

        min_income = min(incomes)
        if min_income < 0:
            incomes = [income - min_income for income in incomes]

        if sum(incomes) == 0:
            return 0
        
        incomes = sorted(incomes)
        n = len(incomes)
        cumsum = np.cumsum(incomes)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def get_number_of_pure_farmers(self):
        """纯农户数量"""
        from agents.farmer_agent import LivelihoodStrategy
        count = 0
        for farmer in self.farmers:
            if farmer.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                count += 1
        return count

    def get_smallholder_exit_rate(self):
        """小农转为非农户的年均速率"""
        from agents.farmer_agent import LivelihoodStrategy
        # 简化计算：当前非农户比例
        non_farmer_count = 0
        for farmer in self.farmers:
            if farmer.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                non_farmer_count += 1
        return non_farmer_count / len(self.farmers) if self.farmers else 0

    def get_land_rental_transactions(self):
        """土地租赁交易数量"""
        return self.land_clearing_house.get_market_statistics()['transaction_count']

    def get_average_land_rental_price(self):
        """平均土地租金"""
        market_stats = self.land_clearing_house.get_market_statistics()
        value = float(market_stats.get('average_rent_per_mu', market_stats.get('average_price', 0.0)) or 0.0)
        return value if value > 0.0 else self._get_operational_transfer_rent_per_mu()

    def _normalize_transfer_rent_total(self, rent, area):
        rent = float(rent or 0.0)
        area = float(area or 0.0)
        if rent <= 0.0 or area <= 0.0:
            return 0.0
        # ponytail: legacy paths mix total contract rent and per-mu rent.
        return rent * area if area > 1.0 and rent < 2000.0 else rent

    def get_operational_transfer_rent_ledger(self):
        """Patch-state rent ledger for all active transfer channels."""
        rows = []
        for patch in getattr(self, 'land_patches', []) or []:
            if not self._is_patch_effectively_transferred(patch):
                continue
            raw_rent = float(
                getattr(patch, 'rental_price', 0.0)
                or getattr(patch, 'contracted_rental_price', 0.0)
                or 0.0
            )
            area = float(getattr(patch, 'area', 0.0) or 0.0)
            rent_total = self._normalize_transfer_rent_total(raw_rent, area)
            if rent_total <= 0.0 or area <= 0.0:
                continue
            channel = self._get_patch_transfer_channel(patch) or 'unknown'
            if hasattr(patch, 'get_patch_id'):
                patch_id = patch.get_patch_id()
            else:
                patch_id = str(getattr(patch, 'patch_id', getattr(patch, 'coordinates', 'patch')))
            rows.append({
                'patch_id': patch_id,
                'channel': channel,
                'area_mu': area,
                'rent_value_raw': raw_rent,
                'rent_total': rent_total,
                'rent_per_mu': rent_total / area,
                'rent_semantics': (
                    'legacy_per_mu_value'
                    if area > 1.0 and raw_rent < 2000.0
                    else 'total_contract_rent'
                ),
                'ledger_source': 'patch_state_operational_transfer',
            })
        return rows

    def _get_operational_transfer_rent_per_mu(self):
        """Average rent from all active transfer channels with positive contract rent."""
        ledger = self.get_operational_transfer_rent_ledger()
        rent_total = sum(float(row.get('rent_total', 0.0) or 0.0) for row in ledger)
        area_total = sum(float(row.get('area_mu', 0.0) or 0.0) for row in ledger)
        return rent_total / area_total if area_total > 0.0 else 0.0

    _POPULATION_SUCCESSION_TRANSFER_CHANNELS = {
        'population_reallocation',
        'population_exit_reassignment',
        'population_entry_assignment',
        'land_bank_reallocation',
    }
    _COLLECTIVE_TRANSFER_CHANNELS = {
        'collective_overflow',
        'cross_village_overflow',
        'village_collective',
        'cross_village_collective',
    }

    def _get_patch_transfer_channel(self, patch):
        return str(
            getattr(patch, 'transfer_channel', None)
            or getattr(patch, 'operational_transfer_channel', '')
            or ''
        ).strip().lower()

    def _is_patch_market_transferred(self, patch):
        if patch is None:
            return False
        if getattr(patch, 'leased_to', None) is not None:
            return True
        transfer_channel = self._get_patch_transfer_channel(patch)
        return (
            bool(getattr(patch, 'operational_transfer_active', False))
            and transfer_channel in self._COLLECTIVE_TRANSFER_CHANNELS
        )

    def _is_patch_population_succession_transfer(self, patch):
        if patch is None:
            return False
        transfer_channel = self._get_patch_transfer_channel(patch)
        return (
            bool(getattr(patch, 'operational_transfer_active', False))
            and transfer_channel in self._POPULATION_SUCCESSION_TRANSFER_CHANNELS
        )

    def _is_patch_effectively_transferred(self, patch):
        if patch is None:
            return False
        if self._is_patch_market_transferred(patch):
            return True
        return self._is_patch_population_succession_transfer(patch)

    def _get_market_transferred_area(self):
        return float(
            sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in getattr(self, 'land_patches', []) or []
                if self._is_patch_market_transferred(patch)
            )
            or 0.0
        )

    def _get_effective_transferred_area(self):
        return float(
            sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in getattr(self, 'land_patches', []) or []
                if self._is_patch_effectively_transferred(patch)
            )
            or 0.0
        )

    def get_land_transfer_rate(self):
        """市场化土地流转率（租赁/集体流转面积/总面积）。

        Population-exit land-bank reassignment is a succession/operation-closure
        mechanism, not a land-rental-market expansion. It remains visible through
        successor-chain diagnostics and `_get_effective_transferred_area()`.
        """
        total_area = sum(patch.area for patch in self.land_patches)
        if total_area <= 0:
            return 0
        transferred_area = self._get_market_transferred_area()
        return transferred_area / total_area
    
    def get_land_abandonment_rate(self):
        """
        @new 2025-12: 土地撂荒率（撂荒面积/总面积）
        专家审阅意见：撂荒是中国农村重要现象，需要跟踪
        实证数据：全国平均10-15%，部分山区可达20-30%
        """
        total_area = sum(patch.area for patch in self.land_patches)
        if total_area <= 0:
            return 0
        abandoned_area = sum(patch.area for patch in self.land_patches if getattr(patch, 'abandoned', False))
        return abandoned_area / total_area
    
    def get_abandonment_statistics(self):
        """
        @new 2025-12: 详细撂荒统计
        """
        total_patches = len(self.land_patches)
        abandoned_patches = sum(1 for p in self.land_patches if getattr(p, 'abandoned', False))
        total_area = sum(p.area for p in self.land_patches)
        abandoned_area = sum(p.area for p in self.land_patches if getattr(p, 'abandoned', False))
        
        return {
            'total_patches': total_patches,
            'abandoned_patches': abandoned_patches,
            'patch_abandonment_rate': abandoned_patches / total_patches if total_patches > 0 else 0,
            'total_area': total_area,
            'abandoned_area': abandoned_area,
            'area_abandonment_rate': abandoned_area / total_area if total_area > 0 else 0,
            'current_step_abandonments': len([r for r in self.abandonment_records if r.get('step') == self.step_count])
        }

    def get_land_concentration_top10(self):
        """衡量土地集中度：前10%经营主体占比。

        经营主体包括仍在经营的农户、企业/合作社。优先按当前地块经营权
        (`current_operator`/`leased_to`/未流转 owner) 汇总，避免只统计
        `self.farmers` 而漏掉规模经营主体。
        """
        area_by_operator = {}
        for patch in getattr(self, 'land_patches', []) or []:
            try:
                area = float(getattr(patch, 'area', 0.0) or 0.0)
            except (TypeError, ValueError):
                area = 0.0
            if area <= 0.0:
                continue
            operator = (
                getattr(patch, 'current_operator', None)
                or getattr(patch, 'leased_to', None)
            )
            if operator is None and getattr(patch, 'leased_to', None) is None:
                operator = getattr(patch, 'owner', None)
            if operator is None:
                continue
            key = id(operator)
            area_by_operator[key] = area_by_operator.get(key, 0.0) + area

        if area_by_operator:
            areas = [area for area in area_by_operator.values() if area > 0.0]
        else:
            areas = []
            for farmer in getattr(self, 'farmers', []) or []:
                if not hasattr(farmer, 'get_operated_land_area'):
                    continue
                try:
                    area = float(farmer.get_operated_land_area() or 0.0)
                except (TypeError, ValueError):
                    area = 0.0
                if area > 0.0:
                    areas.append(area)
            for enterprise in getattr(self, 'agricultural_enterprises', []) or []:
                if not hasattr(enterprise, 'get_operated_land_area'):
                    continue
                try:
                    area = float(enterprise.get_operated_land_area() or 0.0)
                except (TypeError, ValueError):
                    area = 0.0
                if area > 0.0:
                    areas.append(area)
        if not areas:
            return 0
        areas.sort(reverse=True)
        top_n = max(1, int(len(areas) * 0.1))
        top_sum = sum(areas[:top_n])
        total_sum = sum(areas)
        return top_sum / total_sum if total_sum > 0 else 0

    def get_land_fragmentation_index(self):
        """土地碎片化指数：拥有土地的农户比例"""
        if not self.farmers:
            return 0
        operators = sum(1 for farmer in self.farmers if farmer.get_operated_land_area() > 0.1)
        return operators / len(self.farmers)

    def get_livelihood_strategy_distribution(self):
        """获取生计类型分布"""
        distribution = {}
        for farmer in self.farmers:
            strategy = farmer.livelihood_strategy.value
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution

    # 生计策略转换指标计算方法
    def get_livelihood_transition_rate(self):
        """计算生计策略转换率"""
        transitions = 0
        total_farmers = len(self.farmers)
        
        for farmer in self.farmers:
            if hasattr(farmer, 'strategy_history') and len(farmer.strategy_history) >= 2:
                if farmer.strategy_history[-1] != farmer.strategy_history[-2]:
                    transitions += 1
        
        return transitions / total_farmers if total_farmers > 0 else 0

    def get_average_transition_probability(self):
        """计算平均转换概率"""
        probabilities = []
        for farmer in self.farmers:
            if hasattr(farmer, 'last_transition_probability'):
                probabilities.append(farmer.last_transition_probability)
        
        return np.mean(probabilities) if probabilities else 0

    def get_consecutive_high_non_farm_ratio(self):
        """计算连续高非农收入占比的农户比例"""
        count = 0
        total_farmers = len(self.farmers)
        
        for farmer in self.farmers:
            if hasattr(farmer, 'consecutive_high_non_farm_ratio'):
                if farmer.consecutive_high_non_farm_ratio >= 1:
                    count += 1
        
        return count / total_farmers if total_farmers > 0 else 0

    def get_consecutive_low_agricultural_ratio(self):
        """计算连续低农业收入占比的农户比例"""
        count = 0
        total_farmers = len(self.farmers)
        
        for farmer in self.farmers:
            if hasattr(farmer, 'consecutive_low_agricultural_ratio'):
                if farmer.consecutive_low_agricultural_ratio >= 1:
                    count += 1
        
        return count / total_farmers if total_farmers > 0 else 0

    # 企业策略调整指标计算方法
    def get_average_target_return_rate(self):
        """计算平均目标回报率"""
        return_rates = []
        for enterprise in self.agricultural_enterprises:
            if hasattr(enterprise, 'target_return_rate'):
                return_rates.append(enterprise.target_return_rate)
        
        return np.mean(return_rates) if return_rates else 0

    def get_average_performance_trend(self):
        """计算平均业绩趋势"""
        trends = []
        for enterprise in self.agricultural_enterprises:
            if hasattr(enterprise, 'performance_trend'):
                trends.append(enterprise.performance_trend)
        
        return np.mean(trends) if trends else 0

    def get_average_learning_rate(self):
        """计算平均学习率"""
        learning_rates = []
        for enterprise in self.agricultural_enterprises:
            if hasattr(enterprise, 'learning_rate'):
                learning_rates.append(enterprise.learning_rate)
        
        return np.mean(learning_rates) if learning_rates else 0

    def get_enterprise_strategy_adaptation_rate(self):
        """计算企业策略调整率"""
        adaptations = 0
        total_enterprises = len(self.agricultural_enterprises)
        
        for enterprise in self.agricultural_enterprises:
            if hasattr(enterprise, 'target_return_rate_history') and len(enterprise.target_return_rate_history) >= 2:
                if abs(enterprise.target_return_rate_history[-1] - enterprise.target_return_rate_history[-2]) > 0.001:
                    adaptations += 1
        
        return adaptations / total_enterprises if total_enterprises > 0 else 0

    # ========== @new 2025-01-02: Nature级韧性指标 (Folke et al., 2010, Nature) ==========
    
    def get_grain_production_cv(self):
        """
        计算粮食产量变异系数(Coefficient of Variation)
        
        CV = σ/μ，衡量系统产出的稳定性
        - CV < 0.2: 高度稳定
        - 0.2 ≤ CV < 0.4: 中等稳定
        - CV ≥ 0.4: 低稳定性/高脆弱性
        
        参考: Lobell et al. (2011), Nature Climate Change
        """
        # 从历史数据获取产量序列
        if not hasattr(self, 'production_history'):
            self.production_history = []
        
        # 当前年度总产量
        current_production = sum(
            getattr(f, 'current_grain_production', 0) 
            for f in self.farmers
        )
        self.production_history.append(current_production)
        
        # 需要至少3期数据才能计算有意义的CV
        if len(self.production_history) < 3:
            return 0.0
        
        # 计算最近10期的CV (避免过长历史稀释近期波动)
        recent_production = self.production_history[-min(10, len(self.production_history)):]
        mean_production = np.mean(recent_production)
        std_production = np.std(recent_production)
        
        if mean_production <= 0:
            return 0.0
        
        return std_production / mean_production
    
    def get_rental_price_change_rate(self):
        """
        计算土地租金变化率
        
        用于分析补贴对土地租金的资本化效应(Rent Displacement)
        - 正值表示租金上涨，可能反映补贴被地主/规模户获取
        - 参考: Kirwan (2009), American Economic Review
        """
        if not hasattr(self, 'rental_price_history'):
            self.rental_price_history = []
        
        current_rental = self.get_average_land_rental_price()
        self.rental_price_history.append(current_rental)
        
        # 需要至少2期数据
        if len(self.rental_price_history) < 2:
            return 0.0
        
        # 前一期租金
        prev_rental = self.rental_price_history[-2]
        if prev_rental <= 0:
            return 0.0
        
        return (current_rental - prev_rental) / prev_rental
    
    def get_recovery_capacity_index(self):
        """
        计算恢复能力指数 (Recovery Capacity Index)
        
        衡量系统从冲击中恢复的能力：
        - 基于产量偏离趋势值后回归的速度
        - RCI = 1 - (当前偏离/最大历史偏离)
        - RCI > 0.8: 高恢复能力
        - RCI < 0.5: 低恢复能力
        
        参考: Carpenter et al. (2001), Ecology Letters
        """
        if not hasattr(self, 'production_history') or len(self.production_history) < 5:
            return 1.0  # 数据不足时返回正常值
        
        recent_production = self.production_history[-min(10, len(self.production_history)):]
        
        # 计算趋势值 (简单移动平均)
        trend = np.mean(recent_production)
        
        # 当前偏离
        current_deviation = abs(recent_production[-1] - trend) / trend if trend > 0 else 0
        
        # 历史最大偏离
        max_deviation = max(abs(p - trend) / trend for p in recent_production) if trend > 0 else 0
        
        if max_deviation <= 0:
            return 1.0
        
        # 恢复能力 = 1 - 相对偏离
        rci = 1.0 - (current_deviation / max_deviation)
        return max(0, min(1, rci))
    
    def get_ecological_threshold_distance(self):
        """
        计算生态阈值距离 (Ecological Threshold Distance)
        
        衡量当前状态与关键生态阈值的距离：
        - 基于环境核算模块的氮流失和碳排放
        - 阈值参考: Steffen et al. (2015), Science - 行星边界理论
        
        距离 = min(氮阈值距离, 碳阈值距离)
        - > 0.5: 安全区
        - 0.2-0.5: 警戒区
        - < 0.2: 危险区
        """
        # 定义行星边界阈值 (适应中国农业体系)
        # 氮流失阈值: 150 kg N/ha (参考中国农业面源污染限值)
        NITROGEN_THRESHOLD = 150  # kg N/ha
        # 碳排放阈值: 5000 kg CO2eq/ha (参考低碳农业标准)
        CARBON_THRESHOLD = 5000  # kg CO2eq/ha
        
        # 获取当前环境指标
        if not hasattr(self, 'environmental_accountant') or not self.environmental_accountant:
            return 1.0  # 未启用环境核算模块
        
        if not self.environmental_accountant.history:
            return 1.0
        
        latest = self.environmental_accountant.history[-1]
        total_nitrogen = latest.get('total_nitrogen_loss', 0)
        total_carbon = latest.get('total_carbon_emission', 0)
        
        # 计算有效农业面积 (ha)
        total_area_mu = sum(p.area for p in self.land_patches)
        total_area_ha = total_area_mu / 15  # 亩转公顷
        
        if total_area_ha <= 0:
            return 1.0
        
        # 单位面积排放
        nitrogen_per_ha = total_nitrogen / total_area_ha
        carbon_per_ha = total_carbon / total_area_ha
        
        # 计算阈值距离 (归一化到0-1)
        nitrogen_distance = max(0, 1 - nitrogen_per_ha / NITROGEN_THRESHOLD)
        carbon_distance = max(0, 1 - carbon_per_ha / CARBON_THRESHOLD)
        
        # 返回最小距离 (最受约束的维度)
        return min(nitrogen_distance, carbon_distance)
    
    def get_subsidy_net_benefit(self):
        """
        计算补贴净收益 (扣除租金上涨后的实际收益)
        
        用于分析租金挤出效应:
        净收益 = 补贴金额 - 租金增量
        - 正值: 农户真正受益
        - 负值: 补贴被土地所有者获取(租金挤出)
        
        参考: Ciaian & Swinnen (2009), Journal of Agricultural Economics
        """
        if not hasattr(self, 'rental_price_history') or len(self.rental_price_history) < 2:
            return 0.0
        
        # 计算租金增量
        rental_increase = self.rental_price_history[-1] - self.rental_price_history[0]

        # 与租金增量统一成“元/亩”口径，避免把每户总补贴直接减每亩租金。
        subsidy_per_mu_samples = []
        for farmer in self.farmers:
            subsidy_income = max(0.0, float(getattr(farmer, 'subsidy_income', 0.0) or 0.0))
            grain_area = max(0.0, float(getattr(farmer, 'grain_area', 0.0) or 0.0))
            if grain_area <= 0 and hasattr(farmer, 'get_operated_land_area'):
                grain_area = max(0.0, float(farmer.get_operated_land_area() or 0.0))
            if subsidy_income > 0 and grain_area > 0:
                subsidy_per_mu_samples.append(subsidy_income / grain_area)

        avg_subsidy_per_mu = float(np.mean(subsidy_per_mu_samples)) if subsidy_per_mu_samples else 0.0
        return avg_subsidy_per_mu - rental_increase
    def _step_area_economy(self):
        """Household and organization plans, one regional land market, harvest, then payment."""
        from collections import defaultdict
        import numpy as np
        from core.fixed_rule_runtime import continuous_farm_plan, crop_history_group
        e=self.area_economy
        self.step_count+=1
        year=2010+self.step_count
        inputs=e['inputs']
        actors=e['actors']
        previous=e['state']
        rate=e['rate']
        annual_budget=e['budget']
        sar=self.government.announce_area_support(year,previous,annual_budget)
        orders=[]
        rows={}
        leases=[c for c in e['contracts'] if c['end']>=year]
        def position(contracts):
            amounts=defaultdict(lambda:[0.,0.,0.,0.])
            for c in contracts:
                q,p=c['area'],c['rent']
                amounts[c['seller']][0]+=q
                amounts[c['buyer']][1]+=q
                amounts[c['seller']][2]+=q*p
                amounts[c['buyer']][3]+=q*p
            return amounts
        carry=position(leases)
        def factors(a):
            args=(*inputs[a.zone],e['wage'],rate)
            if 'v4' in e:
                yield_ratio=getattr(a,'area_grain_yield',e['yields'][a.zone])/e['yields'][a.zone] if e['v4']>=3 else 1.
                return (args[0]*yield_ratio,*args[1:])
            return args
        v4=e.get('v4')
        if v4:
            from core.v4_household import farm_batch, uniforms
            families=[a for a in actors if a.area_kind=='household']
            ids=[a.unique_id for a in families]
            index={k:i for i,k in enumerate(ids)}
            u=uniforms(e['seed'],year,ids)
            fitted=e['v4_fit'][min(2012,year-1)]
            diagnostic=e.get('behavior_diagnostic')
            prior=np.array([g/(g+c) if g+c>1e-7 else np.nan for g,c in (a.prior_crops for a in families)])
            if 'entry_shares' in fitted:
                had_operating_land=np.array([a.area_previous_operated>1e-7 or a.lease_right>1e-7 for a in families])
            intensity=[a.sowing_intensity for a in families]
            if 'development_intensity' in fitted:
                profile=fitted['development_intensity']
                opening_area=np.array([a.base_area-carry[k][0]+carry[k][1] for a,k in zip(families,ids)])
                bands=np.digitize(opening_area,[5.,15.,30.])
                intensity=[profile[e['macro_region'][a.zone],band] for a,band in zip(families,bands)]
            opening_data=e.get('development_opening')
            def batch(positions, trading, fees=None, observed_opening=False):
                stock=np.array([positions[k] for k in ids])
                reserves=np.array([a.reserve for a in families])
                funds=np.maximum(0.,np.array([a.balance for a in families])-reserves-stock[:,3]
                    -(np.array([fees[k] for k in ids]) if fees is not None else 0.))
                data=dict(land=np.array([a.base_area for a in families])-stock[:,0]+stock[:,1],
                    rights=np.array([a.lease_right for a in families])-stock[:,0],funds=funds,
                    rent=[e['rents'][a.zone]*(1.+e.get('rent_shift',0.)) for a in families],intensity=intensity,
                    labor=[a.family_days for a in families],workers=[a.v1_laborers for a in families],
                    transfer=[a.area_prior_transfer for a in families],prior=prior,
                    region=np.array([e['macro_region'][a.zone] for a in families]),
                    inputs=[factors(a) for a in families],limit=[a.area_limit for a in families])
                if 'entry_shares' in fitted:
                    data['had_operating_land']=had_operating_land
                if observed_opening:
                    for field in ('prior','intensity','had_operating_land'):
                        if field in data:
                            data[field]=np.array(data[field],copy=True)
                    for j,k in enumerate(ids):
                        record=opening_data.get(k)
                        if record is None:
                            continue
                        opening_land=record['area_b']
                        data['land'][j]=opening_land
                        data['rights'][j]=min(data['rights'][j],opening_land)
                        sown=record['sow_food']+record['sow_cash']
                        data['prior'][j]=record['sow_food']/sown if sown>1e-7 else np.nan
                        if 'had_operating_land' in data:
                            data['had_operating_land'][j]=opening_land>1e-7
                        if 'development_intensity' in fitted:
                            data['intensity'][j]=profile[e['macro_region'][families[j].zone],
                                np.digitize(opening_land,[5.,15.,30.])]
                if e.get('scalar_reference'):
                    one=[farm_batch({k:np.asarray(x)[i:i+1] for k,x in data.items()},fitted,u[i:i+1],
                             stage=v4,trade=trading,incentive=e.get('incentive',1.),diagnostic=diagnostic) for i in range(len(ids))]
                    return {k:np.concatenate([x[k] for x in one]) for k in one[0]}
                return farm_batch(data,fitted,u,stage=v4,trade=trading,incentive=e.get('incentive',1.),diagnostic=diagnostic)
            first=batch(carry,True)
            if opening_data is not None:
                conditioned=batch(carry,False,observed_opening=True)
        for a in actors:
            k=a.unique_id
            out,ins,receivable,payable=carry[k]
            land=a.base_area-out+ins
            rights=a.lease_right-out
            funds=max(0.,a.balance-a.reserve-payable-a.overhead)
            rent=e['rents'][a.zone]*(1.+e.get('rent_shift',0.)) if v4 else e['rents'][a.zone]
            args=factors(a)
            mix={}
            if v4 and a.area_kind=='household':
                i=index[k]
                offer,demand=first['offer'][i],first['demand'][i]
                if max(offer,demand)>1e-8:
                    ask=first['ask'][i]
                    if offer and first['cultivated'][i]<max(0.,rights-offer)-1e-8:
                        ask=0.
                    orders.append(dict(id=k,region=a.zone,side='sell' if offer else 'buy',
                         quantity=max(offer,demand),price=ask if offer else first['bid'][i]))
                rows[k]=dict(year=year,id=k,kind=a.area_kind,region=a.zone,opening=a.balance,base_area=a.base_area,
                    rights=a.lease_right,expected_rent=rent,offer=offer,demand=demand,working_funds=funds,
                    planned_cultivated=first['cultivated'][i],cash_shadow=float(first['shadow_cash'][i]),
                    planned_grain=first['grain'][i],planned_crop=first['crop'][i],prior_share=prior[i],
                    target_share=first['target'][i],choice_u=u[i,1],family_capacity=a.family_days,
                    p_down=first['probabilities'][i,0],p_hold=first['probabilities'][i,1],
                    p_up=first['probabilities'][i,2],p_stay=first['stay_probability'][i])
                if diagnostic and diagnostic.get('trace'):
                    rows[k].update({f'pre_{j}':first[j][i] for j in (
                        'stay_u','plan_sown','plan_land','max_sown_land','max_sown_affordable','plan_cash',
                        'plan_hired','plan_labor','classifier_funds','labor_binding','land_binding','value')})
                if 'endpoint_rescue' in first:
                    rows[k]['pre_endpoint_rescue']=int(first['endpoint_rescue'][i])
                if 'development_intensity' in fitted:
                    rows[k]['development_intensity']=intensity[i]
                if opening_data is not None and k in opening_data:
                    rows[k].update(opencheck_grain=conditioned['grain'][i],
                        opencheck_crop=conditioned['crop'][i],
                        opencheck_land=opening_data[k]['area_b'])
                continue
            if 'transition' in e and a.area_kind=='household':
                import hashlib
                g0,c0=a.prior_crops
                prior=crop_history_group(g0,c0)
                token=hashlib.sha256(f'crop-share:{e["seed"]}:{year}:{k}'.encode()).digest()
                u=int.from_bytes(token[:8],'big')/2**64
                # Missing history has no estimated row: use the unconstrained old choice.
                a.crop_mix_target=None
                if 'regional_history' in e:
                    from core.fixed_rule_runtime import continuous_crop_target
                    a.crop_adjustment_yuan=e['actor_crop_penalty'][k] if 'actor_crop_penalty' in e else e['crop_penalty']
                    share=g0/(g0+c0) if g0+c0>1e-7 else None
                    a.crop_mix_target=continuous_crop_target(e['regional_history'][e['macro_region'][a.zone]],prior,share,u)
                elif 'continuous_crop_history' in e:
                    from core.fixed_rule_runtime import continuous_crop_target
                    share=g0/(g0+c0) if g0+c0>1e-7 else None
                    a.crop_mix_target=continuous_crop_target(e['continuous_crop_history'],prior,share,u)
                elif prior in e['transition']:
                    cumulative,targets=e['transition'][prior]
                    j=min(int(np.searchsorted(cumulative,u,side='right')),len(targets)-1)
                    a.crop_mix_target=float(targets[j])
                    assert 0<=a.crop_mix_target<=1
                mix=dict(prior_group=prior,prior_share=g0/(g0+c0) if g0+c0>1e-7 else None,
                         choice_u=u,target_share=a.crop_mix_target,margin_difference=args[0]-args[2]-args[1]+args[3],
                         family_capacity=a.family_days)
            base=continuous_farm_plan(a,land,rights,funds,rent,args,False)
            plan=continuous_farm_plan(a,land,rights,funds,rent,args,True) if a.active_area_business else base
            # A potential entrant may bid below last year's price. Do not censor its order.
            if a.active_area_business and plan['offer']+plan['demand']<1e-8 and land<a.area_limit:
                probe=min(1e-4,a.area_limit-land)
                extra=continuous_farm_plan(a,land+probe,rights,funds,0.,args,False)
                marginal=(extra['expected']-base['expected'])/probe/(1.+extra['cash_shadow'])
                if 1e-6<marginal<=rent+1e-6:
                    trial=continuous_farm_plan(a,land,rights,funds,max(0.,marginal-1e-5),args,True)
                    if trial['demand']>1e-8: plan=trial
            q=plan['offer'] or plan['demand']
            side='sell' if plan['offer'] else 'buy'
            blocked_by_overhead=False
            if q>1e-8:
                # Reprice the last unit, not average surplus on the entire order.
                eps=min(q,1e-4)
                if side=='sell':
                    sold=continuous_farm_plan(a,land-q,rights-q,funds,0.,args,False)
                    back=continuous_farm_plan(a,land-q+eps,rights-q+eps,funds,0.,args,False)
                    price=(back['expected']-sold['expected'])/eps
                else:
                    headroom=max(0.,funds-plan['cost'])/q
                    work=max(0.,funds-min(rent,headroom)*q)
                    before=continuous_farm_plan(a,land+q-eps,rights,work,0.,args,False)
                    after=continuous_farm_plan(a,land+q,rights,work,0.,args,False)
                    price=min(headroom,(after['expected']-before['expected'])/eps/(1.+after['cash_shadow']))
                    if v4 and v4>=4 and a.area_kind!='household':
                        # No bid whose full proposed operation cannot cover the business's fixed costs.
                        price=min(price,(after['expected']-a.overhead)/q)
                        blocked_by_overhead=price<0.
                if price>=-1e-7:
                    orders.append(dict(id=k,region=a.zone,side=side,quantity=q,price=max(0.,price)))
            rows[k]=dict(year=year,id=k,kind=a.area_kind,region=a.zone,opening=a.balance,
                         base_area=a.base_area,rights=a.lease_right,expected_rent=rent,
                         offer=plan['offer'],demand=plan['demand'],working_funds=funds,
                         planned_cultivated=plan['cultivated'],cash_shadow=plan['cash_shadow'])
            if v4 and v4>=4:
                rows[k]['bid_rejected_overhead']=blocked_by_overhead
            if mix:
                rows[k].update(**mix,planned_grain=plan['grain'],planned_crop=plan['crop'],
                               labor_shadow=plan['labor_shadow'],land_shadow=plan['land_value'])
        if v4 and 'v4_space' in e:
            external=e.get('scale_pool')
            if external is not None:
                external={z:dict(p,reference_rent=e['rents'][z]) for z,p in external.items()}
                deals=self.land_clearing_house.clear_area_orders(orders,year,e['term'],space=e['v4_space'],
                        external=external,household_ids=set(ids))
            else:
                deals=self.land_clearing_house.clear_area_orders(orders,year,e['term'],space=e['v4_space'])
        else:
            deals=self.land_clearing_house.clear_area_orders(orders,year,e['term'])
        leases.extend(deals)
        e['contracts']=leases
        for d in deals: e['rents'][d['region']]=d['rent']
        final=position(leases)
        claims={}
        if v4:
            fees=defaultdict(float)
            for d in deals: fees[d['buyer']]+=d.get('search_fee',0.)*d['area']
            last=batch(final,False,fees)
            weather={}
            for z in set(a.zone for a in actors):
                rng=np.random.default_rng([e['seed'],year,int(z)])
                r=rng.random()
                weather[z]=(1.-rng.uniform(.05,.10) if r<.05 else
                            1.+rng.uniform(.0275,.055) if r<.13 else 1.)
            hv=np.array([getattr(a,'area_grain_yield',e['yields'][a.zone]) if v4>=3 else e['yields'][a.zone] for a in families])
            hf=np.array([weather[a.zone] for a in families])
            gain=np.asarray([factors(a)[:2] for a in families])
            actual_qty=last['grain']*hv*hf
            actual_sales=(last['grain']*gain[:,0]+last['crop']*gain[:,1])*hf
        for a in actors:
            k=a.unique_id
            out,ins,rental_income,rental_cost=final[k]
            land=a.base_area-out+ins
            rights=a.lease_right-out
            overhead=a.overhead if a.active_area_business else 0.
            funds=max(0.,a.balance-a.reserve-rental_cost-overhead)
            args=factors(a)
            if v4:
                funds=max(0.,funds-fees[k])
            if v4 and a.area_kind=='household':
                i=index[k]
                p={c:last[c][i] for c in ('grain','crop','cultivated','eligible','hired','cost')}
            else:
                p=continuous_farm_plan(a,land,rights,funds,0.,args,False)
            if not a.active_area_business:
                assert land<1e-7
            # Same region-year draw regardless of market participation or actor count.
            rng=np.random.default_rng([e['seed'],year,int(a.zone)])
            u=rng.random()
            shock=(1.-rng.uniform(.05,.10) if u<.05 else
                   1.+rng.uniform(.0275,.055) if u<.13 else 1.)
            qty=p['grain']*e['yields'][a.zone]*shock
            sales=(p['grain']*args[0]+p['crop']*args[1])*shock
            if v4:
                rows[k].update(transaction_cost=fees[k],preplant_funds=funds,weather_factor=shock,
                               crop_kg=p['crop']*e['cash_yield']*shock,nonlabor_cost=p['grain']*args[2]+p['crop']*args[3],
                               hired_cost=p['hired']*args[6],agricultural_labor_requirement=p['grain']*args[4]+p['crop']*args[5],
                               demand_limited=max(0.,rows[k]['demand']-ins))
                if a.area_kind=='household':
                    qty,sales=actual_qty[i],actual_sales[i]
                    a.prior_crops=p['grain'],p['crop']
                    if 'entry_shares' in fitted:
                        a.area_previous_operated=land
                    a.area_prior_transfer=int(ins>1e-7)-int(out>1e-7)
                    rows[k].update(family_used=last['family'][i],final_cash_shadow=float(last['shadow_cash'][i]),
                        final_p_down=last['probabilities'][i,0],final_p_hold=last['probabilities'][i,1],
                        final_p_up=last['probabilities'][i,2],final_choice=int(last['direction'][i]))
                    if diagnostic and diagnostic.get('trace'):
                        rows[k].update({f'post_{j}':last[j][i] for j in (
                            'stay_u','stay_probability','plan_sown','plan_land','max_sown_land','max_sown_affordable',
                            'plan_cash','plan_hired','plan_labor','classifier_funds','labor_binding','land_binding','value')})
                    if 'endpoint_rescue' in last:
                        rows[k]['post_endpoint_rescue']=int(last['endpoint_rescue'][i])
            claims[k]=rate*p['eligible']
            if 'transition' in e and a.area_kind=='household':
                a.prior_crops=p['grain'],p['crop']
                rows[k].update(family_used=p['grain']*args[4]+p['crop']*args[5]-p['hired'],
                               final_cash_shadow=p['cash_shadow'],final_labor_shadow=p['labor_shadow'],
                               final_land_shadow=p['land_value'],preplant_funds=funds,weather_factor=shock)
            assert -1e-7<=out<=a.lease_right+1e-7 and p['cost']<=funds+1e-5
            assert p['cultivated']<=land+1e-6
            rows[k].update(operated=land,rented_in=ins,rented_out=out,rent_paid=rental_cost,
                rent_received=rental_income,grain=p['grain'],crop=p['crop'],cultivated=p['cultivated'],
                idle=max(0.,land-p['cultivated']),grain_kg=qty,sales=sales,input_hire_cost=p['cost'],
                hired_days=p['hired'],overhead=overhead,nonfarm=a.wage_income,claim=claims[k])
        paid=self.government.settle_area_support(sar,claims,{a.unique_id:a.zone for a in actors})
        self.area_external_last=[]
        if 'scale_pool' in e:
            for z,pool in sorted(e['scale_pool'].items()):
                out,ins,income,rent=final[f'_area_pool_{z}']
                assert out<=pool['supply']+1e-7 and ins<=pool['demand']+1e-7
                left=pool['supply']-out
                held=left+ins
                m=pool['intensity']
                share=pool['share']
                grain,crop=held*m*share,held*m*(1.-share)
                gy=pool['grain_yield']
                g_price=inputs[z][0]/e['yields'][z]
                sales=(grain*gy*g_price+crop*inputs[z][1])*weather[z]
                cost=grain*pool['grain_cost']+crop*inputs[z][3]
                self.area_external_last.append(dict(year=year,region=z,opening_area=pool['supply'],
                    demand_capacity=pool['demand'],rented_in=ins,rented_out=out,operated=held,retained=left,
                    grain=grain,crop=crop,grain_kg=grain*gy*weather[z],crop_kg=crop*e['cash_yield']*weather[z],
                    acquired_grain_kg=ins*m*share*gy*weather[z],released_grain_kg=out*m*share*gy*weather[z],
                    rent_paid=rent,rent_received=income,sales=sales,cost=cost,net=sales-cost+income-rent,paid=0.,
                    reference_rent=external[z]['reference_rent'],weather_factor=weather[z]))
        state={}
        for a in actors:
            r=rows[a.unique_id]
            surplus=r['sales']-r['input_hire_cost']-r['rent_paid']-r['overhead']+r['rent_received']
            if v4:
                surplus-=r['transaction_cost']
            net=surplus+paid[a.unique_id]+a.wage_income
            # Consumption uses available balances; a loss-making organization may exit.
            consumption=min(a.consumption,max(0.,a.balance+net))
            a.balance+=net-consumption
            r.update(paid=paid[a.unique_id],surplus=surplus,net=net,consumption=consumption,
                     closing=a.balance,exited=False)
            if a.area_kind!='household' and a.active_area_business and a.balance<a.overhead:
                a.active_area_business=False
                r['exited']=True
            s=state.setdefault(a.zone,dict(grain_kg=0.,grain_mu=0.,leased_mu=0.,paid=0.))
            for col,val in [('grain_kg',r['grain_kg']),('grain_mu',r['grain']),
                            ('leased_mu',r['rented_in']),('paid',r['paid'])]: s[col]+=val
            if v4:
                for key,value in [('physical_mu',r['operated']),('rent_paid',r['rent_paid']),
                                  ('organization_mu',r['operated'] if a.area_kind!='household' else 0.),
                                  ('household_income',r['net'] if a.area_kind=='household' else 0.),
                                  ('households',1 if a.area_kind=='household' else 0)]:
                    s[key]=s.get(key,0.)+value
        for z,s in state.items():
            s['grain_gap']=e['target'][z]-s['grain_kg']
            s['budget_fraction_national']=s['paid']/annual_budget
            if v4:
                s['lease_rate']=s['leased_mu']/max(s['physical_mu'],1e-6)
                s['average_rent']=s['rent_paid']/s['leased_mu'] if s['leased_mu']>0 else None
                s['rent_change']=s['average_rent']-previous[z]['average_rent'] if (
                    s['average_rent'] is not None and previous[z].get('average_rent') is not None) else None
                s['organization_share']=s['organization_mu']/max(s['physical_mu'],1e-6)
                s['mean_household_income']=s['household_income']/s['households'] if s['households']>0 else None
        sar.update(next_state=state,reward=sum(s['grain_kg']-previous[z]['grain_kg'] for z,s in state.items()))
        if v4:
            sar['budget_remaining']=annual_budget-sum(s['paid'] for s in state.values())
            sar['action_mask']={a:bool(annual_budget>0 or a==self.government.AREA_ACTIONS[0]) for a in self.government.AREA_ACTIONS}
        e['state']=state
        self.area_last=(list(rows.values()),orders,deals,sar)
