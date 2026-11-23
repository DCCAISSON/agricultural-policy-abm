# -*- coding: utf-8 -*-
"""
@theory
FarmerAgent - 基于多层次博弈设计规范的农户主体
核心特征：生计类型动态变化，清晰的收支定义，基于前景理论的决策
新增：作物选择博弈、多层次博弈机制、生计策略演化
理论一致性修正：完全实现前景理论决策框架

@refactor 2025-12: 
- 新增连续变量农户分类系统（FarmerCharacteristics）
- 实现6-7分法精细农户类型划分
- 添加家庭生命周期和代际接替机制
- 实现政策敏感度异质性

@fix 2026-01-04:
- 集成FarmerContinuousDecisionMixin，启用连续决策功能
- 实现get_agricultural_weight()等连续决策API
"""

from mesa import Agent
import numpy as np
from enum import Enum
import random
from copy import copy
from core.fixed_rule_runtime import ActionIncomeAccount, action_income_enabled, expected_land_cash, expected_support, rental_rights_holder
from core.fixed_rule_runtime import production_area, preproduction_cash_enabled, settle_preproduction_cash
from core.config_utils import DEFAULT_BASE_NON_FARM_WAGE, resolve_base_non_farm_wage
from core.reproducibility import (
    np_choice,
    np_normal,
    paired_yield_noise,
    np_randint,
    np_random_value,
    np_uniform,
    rng_random,
    rng_randint,
    rng_sample,
    rng_shuffle,
)
from utils.ai_runtime import mechanism_enabled, productivity_params
from utils.runtime_fallback_audit import record_runtime_fallback
from utils.yield_stack_runtime import finalize_yield_stack
from utils.yield_growth_runtime import resolve_crop_annual_yield_growth
from core.mechanisms import ProspectTheoryMechanism  # 导入前景理论机制
from modules.household_consumption import HouseholdConsumptionModel  # 导入消费模块
# @new 2025-12: 导入连续变量农户特征系统
from modules.farmer_characteristics import (
    FarmerCharacteristics, 
    FarmerType, 
    FarmerTypeClassifier,
    LifecycleManager,
    initialize_farmer_characteristics
)
# @new 2025-12: 导入连续变量决策函数（替代离散类型判断）
from modules.continuous_decision import (
    ContinuousDecisionFunctions,
    get_prospect_params_by_dependence,
    check_acquaintance_simple,
    calculate_acquaintance_discount
)
# @fix 2026-01-04: 导入连续决策Mixin
from modules.agent_decision_fix import FarmerContinuousDecisionMixin
from modules.crop_components import assign_grain_crop_components
from utils.constants import JIN_PER_KG

class LivelihoodStrategy(Enum):
    """生计类型枚举"""
    PURE_FARMER = "Pure_Farmer"      # 纯农户：农业收入 > 90%
    PLURIACTIVE = "Pluriactive"      # 兼业农户：农业收入 10%-90%
    NON_FARMER = "Non_Farmer"        # 非农户：农业收入 < 10%

class FarmerScale(Enum):
    """
    农户经营规模枚举
    
    @new 2025-12: 新增农户规模分类
    理论依据：家庭农场归并到农户类型后，需要区分不同规模的农户
    
    规模划分标准（参考农业农村部《家庭农场培育计划》）：
    - 小农户：3-15亩，以自给自足为主
    - 中等规模农户：15-50亩，商品化生产
    - 家庭农场：50亩以上，规模化经营（可通过租入土地达到）
    
    注意：家庭农场是"大农户"，不是"企业"，以家庭劳动力为主
    """
    SMALLHOLDER = "Smallholder"          # 小农户：3-15亩
    MEDIUM_FARMER = "Medium_Farmer"      # 中等规模农户：15-50亩
    FAMILY_FARM = "Family_Farm"          # 家庭农场：50亩以上（含租入）

class CropType(Enum):
    """作物类型枚举"""
    GRAIN = "Grain"                  # 粮食作物
    CASH_CROP = "Cash_Crop"          # 经济作物
    MIXED = "Mixed"


class FarmerAgent(Agent, FarmerContinuousDecisionMixin):
    """
    @fix 2026-01-04: 集成FarmerContinuousDecisionMixin
    
    现在FarmerAgent具有连续决策能力：
    - get_agricultural_weight(): 获取农业生计权重（替代 if PURE_FARMER 判断）
    - get_non_farm_weight(): 获取非农生计权重
    - get_transfer_willingness('out'/'in'): 获取土地流转意愿
    - get_land_rental_price_factor(): 获取租金出价因子
    
    @new 2026-01-07: 支持微观数据驱动初始化
    - from_microdata(): 类方法，从CFPS/NFPS微观特征初始化
    """
    
    @property
    def _national_v1(self):
        return self.model.config.get('runtime_fixed_rule', {}).get('national_v1') is True

    def initialize_area_household(self, row):
        self.area_kind='household'
        self.zone=int(row.smgen)
        self.base_area=float(row.area_e)
        self.lease_right=float(row.verified_right)
        self.sowing_intensity=float(row.intensity)
        self.balance=float(row.opening_cash)
        self.reserve=float(row.held_reserve)
        self.consumption=float(row.consumption)
        self.wage_income=float(row.wage_income)
        self.overhead=0.
        self.family_days=self.agricultural_labor
        self.area_limit=float(row.region_land)
        self.active_area_business=True

    def set_v1_labor(self, laborers, paid_nonfarm_proxy):
        """Use a baseline household count and an annual CFPS group proxy, not matched persons."""
        valid = (laborers is not None and np.isfinite(laborers) and 0 <= laborers <= self.household_size
                 and float(laborers).is_integer())
        self.v1_laborers = int(laborers) if valid else int(self.household_size > 0)
        self.worker_count_source = 'input_count' if valid else 'one_worker_assumption'
        # scripts/audit_one_mu_hired_days.py:audit_capacity; rural 2010 qh1*qh2, grouped by qh4.
        self.agricultural_days_per_worker = {True: 120., False: 180., None: 176.}[paid_nonfarm_proxy]
        self.total_labor_units = 365. * self.v1_laborers
        self.agricultural_labor = self.v1_laborers * self.agricultural_days_per_worker
        self.off_farm_labor = self.total_labor_units - self.agricultural_labor

    @classmethod
    def from_microdata(cls, unique_id, model, microdata_features: dict, config=None):
        """
        从微观数据特征初始化农户智能体
        
        @new 2026-01-07: 微观数据驱动的ABM智能体初始化
        
        Args:
            unique_id: 智能体ID
            model: 模型实例
            microdata_features: 微观数据特征字典，包含：
                - land_area_mu: 土地面积（亩）
                - agri_income: 农业收入
                - total_income: 家庭总收入
                - agri_ratio: 农业收入占比
                - livelihood: 生计类型 (Pure_Farmer/Pluriactive/Non_Farmer)
                - farmer_scale: 农户规模
                - family_size: 家庭人口
                - region: 区域
                - is_rent_out: 是否转出土地
                - is_rent_in: 是否转入土地
            config: 配置字典
        
        Returns:
            FarmerAgent实例
        """
        from modules.land_patch import LandPatch
        
        # 从微观数据提取关键特征
        land_area = microdata_features.get('land_area_mu', 10.0)
        total_income = microdata_features.get('total_income', 30000)
        agri_income = microdata_features.get('agri_income', None)
        net_income = microdata_features.get('net_income', total_income)
        agri_ratio = microdata_features.get('agri_ratio', 0.3)
        farmer_scale_label = microdata_features.get('farmer_scale', None)
        land_rent_in_area = microdata_features.get('land_rent_in_area', 0.0)
        land_rent_out_area = microdata_features.get('land_rent_out_area', 0.0)
        livelihood = microdata_features.get('livelihood', 'Pluriactive')
        family_size = microdata_features.get('family_size', 4)
        region = microdata_features.get('region', '中部')
        official_macro_regions = {'东部', '中部', '西部', '东北'}
        region_scope = 'official_macro_region' if str(region) in official_macro_regions else None

        try:
            land_area = max(0.1, float(land_area or 0.1))
        except (TypeError, ValueError):
            land_area = 10.0
        try:
            total_income = float(total_income or 0.0)
        except (TypeError, ValueError):
            total_income = 30000.0
        try:
            agri_income = float(agri_income) if agri_income is not None else total_income * float(agri_ratio or 0.0)
        except (TypeError, ValueError):
            agri_income = total_income * float(agri_ratio or 0.0)
        try:
            net_income = float(net_income or total_income)
        except (TypeError, ValueError):
            net_income = total_income
        try:
            land_rent_in_area = max(0.0, float(land_rent_in_area or 0.0))
        except (TypeError, ValueError):
            land_rent_in_area = 0.0
        try:
            land_rent_out_area = max(0.0, float(land_rent_out_area or 0.0))
        except (TypeError, ValueError):
            land_rent_out_area = 0.0
        
        # 创建土地地块
        # 假设每个农户的土地分成若干小块
        n_patches = max(1, int(land_area / 3))  # 每块约3亩
        patch_size = land_area / n_patches if n_patches > 0 else land_area
        
        # 默认产量剖面（与ModelBuilder一致）
        default_yield_profiles = {
            'wheat': {'base_yield': 400, 'price': 2.5},
            'rice': {'base_yield': 500, 'price': 2.8},
            'corn': {'base_yield': 450, 'price': 2.2},
            'soybean': {'base_yield': 180, 'price': 4.5},
            'vegetables': {'base_yield': 3000, 'price': 1.5},
            'fruits': {'base_yield': 2000, 'price': 3.0}
        }
        
        land_patches = []
        for i in range(n_patches):
            # LandPatch参数: (area, soil_quality, climate_zone, yield_profiles, coordinates, model)
            patch = LandPatch(
                area=patch_size,
                soil_quality=float(np_uniform(model, 0.8, 1.2)),  # 地力
                climate_zone='temperate',  # 默认温带
                yield_profiles=default_yield_profiles,
                coordinates=(int(np_randint(model, 0, 100)), int(np_randint(model, 0, 100))),
                model=model
            )
            patch.patch_id = f"{unique_id}_patch_{i}"
            patch.region = region
            patch.region_scope = region_scope
            if region_scope == 'official_macro_region':
                patch.official_region = region
                patch.official_region_scope = region_scope
            land_patches.append(patch)
        
        # Explicit liquid stocks take precedence; zero is an observed input, not missing.
        explicit_cash = ('initial_working_capital_yuan' in microdata_features
                         or 'household_cash_reserve_yuan' in microdata_features)
        if explicit_cash:
            try:
                initial_capital = float(microdata_features['initial_working_capital_yuan'])
                household_cash_reserve = float(microdata_features['household_cash_reserve_yuan'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Explicit household cash requires finite L and R fields') from exc
            if (not np.isfinite(initial_capital) or initial_capital < 0
                    or not np.isfinite(household_cash_reserve) or household_cash_reserve < 0):
                raise ValueError('Explicit household cash L and R must be finite and nonnegative')
        else:
            initial_capital = max(5000, total_income * 0.3)
        
        # 根据农业收入占比估算耕作能力
        # 高农业收入占比 -> 高耕作能力
        if model.config.get('runtime_fixed_rule', {}).get('national_v1') is True:
            np_uniform(model, 0.0, 1.0)  # Consume the legacy draw without assigning an income-based skill.
            farming_ability = 1.0
        elif agri_ratio > 0.5:
            farming_ability = float(np_uniform(model, 0.9, 1.3))
        elif agri_ratio > 0.2:
            farming_ability = float(np_uniform(model, 0.8, 1.1))
        else:
            farming_ability = float(np_uniform(model, 0.6, 1.0))
        
        # 创建智能体
        agent = cls(
            unique_id=unique_id,
            model=model,
            land_patches=land_patches,
            initial_capital=initial_capital,
            farming_ability=farming_ability,
            config=config
        )
        if explicit_cash:
            agent.initial_working_capital_yuan = initial_capital
            agent.household_cash_reserve_yuan = household_cash_reserve
        
        # 设置生计策略
        if livelihood == 'Pure_Farmer':
            agent.livelihood_strategy = LivelihoodStrategy.PURE_FARMER
        elif livelihood == 'Pluriactive':
            agent.livelihood_strategy = LivelihoodStrategy.PLURIACTIVE
        else:
            agent.livelihood_strategy = LivelihoodStrategy.NON_FARMER
        
        # 设置家庭规模
        agent.household_size = family_size
        
        # 设置区域（用于政策响应差异）
        agent.region = region
        
        # 设置土地流转状态
        agent.is_initial_rent_out = microdata_features.get('is_rent_out', 0) == 1
        agent.is_initial_rent_in = microdata_features.get('is_rent_in', 0) == 1

        # 将微观收入/流转信息注入运行时字段，避免初始化阶段丢失经验口径。
        land_transfer_cfg = (
            (config or {})
            .get('agents', {})
            .get('farmer', {})
            .get('land_transfer', {})
        )
        base_rent_price = float(land_transfer_cfg.get('base_rent_price', 420) or 420)
        estimated_rental_income = min(
            max(0.0, total_income - agri_income),
            land_rent_out_area * base_rent_price,
        )
        estimated_off_farm_income = max(0.0, total_income - agri_income - estimated_rental_income)
        operated_area_hint = max(0.0, land_area + land_rent_in_area - land_rent_out_area)

        agent.agricultural_income = max(0.0, agri_income)
        agent.planting_gross_income = agent.agricultural_income
        agent.planting_net_income = agent.agricultural_income
        agent.land_rental_income = max(0.0, estimated_rental_income)
        agent.off_farm_income = max(0.0, estimated_off_farm_income)
        agent.gross_income = max(0.0, total_income)
        agent.total_income = agent.gross_income
        agent.income_total = agent.gross_income
        agent.income_agricultural = agent.agricultural_income
        agent.net_income = net_income
        agent.cash_net_income = net_income
        agent.production_cost = max(0.0, agent.gross_income - agent.net_income)
        agent.land_area = land_area
        agent.land_rent_in_area = land_rent_in_area
        agent.land_rent_out_area = land_rent_out_area
        agent.initial_operated_area_hint = operated_area_hint
        agent._microdata_operated_area_hint = operated_area_hint

        scale_map = {
            'smallholder': FarmerScale.SMALLHOLDER,
            'small_holder': FarmerScale.SMALLHOLDER,
            'small': FarmerScale.SMALLHOLDER,
            'medium_farmer': FarmerScale.MEDIUM_FARMER,
            'medium': FarmerScale.MEDIUM_FARMER,
            'family_farm': FarmerScale.FAMILY_FARM,
            'family': FarmerScale.FAMILY_FARM,
        }
        raw_scale = str(farmer_scale_label or '').strip().lower()
        if raw_scale in scale_map:
            agent.farmer_scale = scale_map[raw_scale]
        elif operated_area_hint > 0:
            if operated_area_hint < 15:
                agent.farmer_scale = FarmerScale.SMALLHOLDER
            elif operated_area_hint < 50:
                agent.farmer_scale = FarmerScale.MEDIUM_FARMER
            else:
                agent.farmer_scale = FarmerScale.FAMILY_FARM
        
        # 保存原始微观数据引用（用于验证）
        agent._microdata_source = microdata_features
        
        # 根据生计类型调整劳动力分配
        if livelihood == 'Pure_Farmer':
            agent.agricultural_labor = int(agent.total_labor_units * np_uniform(agent, 0.7, 0.9))
            agent.off_farm_labor = agent.total_labor_units - agent.agricultural_labor
            agent.off_farm_wage_level = float(np_uniform(agent, 0.6, 0.9))
        elif livelihood == 'Pluriactive':
            agent.agricultural_labor = int(agent.total_labor_units * np_uniform(agent, 0.3, 0.6))
            agent.off_farm_labor = agent.total_labor_units - agent.agricultural_labor
            agent.off_farm_wage_level = float(np_uniform(agent, 0.8, 1.2))
        else:  # Non_Farmer
            agent.agricultural_labor = int(agent.total_labor_units * np_uniform(agent, 0.05, 0.2))
            agent.off_farm_labor = agent.total_labor_units - agent.agricultural_labor
            agent.off_farm_wage_level = float(np_uniform(agent, 1.0, 1.5))

        if agent._national_v1:
            agent.set_v1_labor(microdata_features.get('labor_count'), microdata_features.get('paid_nonfarm_proxy'))
            agent.working_capital_source = ('frozen_liquid_stock_less_reserve' if explicit_cash
                                            else 'income_proxy_not_observed_liquidity')

        has_micro_income_profile = (
            agent.gross_income > 0
            or agent.agricultural_income > 0
            or agent.land_rent_in_area > 0
            or agent.land_rent_out_area > 0
        )
        agent._sync_characteristics_income_fields(use_strategy_anchor=not has_micro_income_profile)
        agent._align_characteristics_to_livelihood_strategy(
            blend=0.25 if has_micro_income_profile else 0.85
        )
        agent.farmer_type, agent.farmer_type_confidence = FarmerTypeClassifier.classify(
            agent.characteristics
        )
        
        return agent
    
    def __init__(self, unique_id, model, land_patches, initial_capital=10000, farming_ability=None, config=None):
        # Mesa 3.x compatibility: Agent.__init__ may pass args to object incorrectly
        # We manually set required attributes first, then try super() but ignore failures
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # Required by Mesa grid system
        try:
            super().__init__(unique_id, model)
        except TypeError:
            # Mesa 3.x issue: don't call super().__init__() at all
            # Required attributes (unique_id, model, pos) already set above
            pass
        # @fix: 保存config引用以便后续使用
        self.agent_config = config if config is not None else model.config if hasattr(model, 'config') else {}
        productivity_shock = productivity_params(self.agent_config)
        self.productivity_shock_enabled = mechanism_enabled(self.agent_config, "productivity_shock")
        self.productivity_non_farm_job_shock = productivity_shock.get('non_farm_job_shock', 0.0)
        self.productivity_wage_penalty = productivity_shock.get('wage_penalty', 0.0)
        self.productivity_adoption_lag = productivity_shock.get('adoption_lag', 0.0)
        
        # 核心属性
        self.land_patches = land_patches  # 拥有的地块列表
        self.land_area_mu = float(sum(getattr(patch, 'area', 0.0) for patch in land_patches))
        self.capital = initial_capital    # 资本（缓冲能力）
        self.initial_capital = initial_capital  # 初始资本（用于财富效应计算）
        # @new 2026-01-27: 作物面积与产量（供统一指标收集）
        self.grain_area = 0
        self.cash_crop_area = 0
        self.grain_production = 0
        self.cash_crop_production = 0
        self.crop_component_area_mu = {}
        self.crop_component_production_kg = {}
        self.crop_component_status = "not_initialized"
        
        # 生计相关属性
        self.livelihood_strategy = LivelihoodStrategy.PURE_FARMER  # 生计类型
        self.previous_livelihood_strategy = self.livelihood_strategy
        self.current_livelihood_strategy = self.livelihood_strategy
        self.transition_reason = "初始状态"
        self.transition_probability = 0.0
        self.transition_year = None
        self.livelihood_transition_history = []
        
        # @new 2025-12: 农户经营规模分类
        # 初始规模根据拥有土地面积确定
        self.farmer_scale = self._determine_farmer_scale()
        
        # @fix: 接受从外部传入的 farming_ability 参数
        if farming_ability is not None:
            self.farming_ability = farming_ability
        else:
            self.farming_ability = float(np_uniform(self, 0.8, 1.2))       # 耕作能力（学习能力）
        
        if self._national_v1:
            self.farming_ability = 1.0
            self.v1_laborers = 1
            self.worker_count_source = 'one_worker_assumption'
            self.agricultural_days_per_worker = 176.
        self.off_farm_wage_level = float(np_uniform(self, 0.8, 1.2))   # 非农工资水平（人力资本）
        
        # 动态风险偏好机制（第二阶段新增）
        self.base_risk_aversion = float(np_uniform(self, 1.5, 2.5))    # 基础风险规避系数
        self.risk_aversion = self.base_risk_aversion             # 当前风险规避系数（动态调整）
        self.success_streak = 0      # 连续成功次数
        self.failure_streak = 0      # 连续失败次数
        self.risk_learning_rate = 0.1  # 风险偏好学习速率
        
        # 前景理论决策机制
        # @improvement: P2优先级 - 前景理论参数差异化
        # 根据生计策略和收入水平差异化参数
        prospect_config = self._get_differentiated_prospect_config()
        self.prospect_theory = ProspectTheoryMechanism(prospect_config)
        
        # 动态参考点机制
        self.reference_point = initial_capital * 0.8  # 初始参考点设为初始资本的80%
        self.reference_point_type = "上年收入"  # 参考点类型
        self.income_memory = []  # 收入记忆（用于更新参考点）
        self.reference_point_history = []  # 参考点历史
        
        # 价格历史与预期形成（解决黑箱问题）
        self.price_history = {
            'grain': [],      # 粮食价格历史
            'cash_crop': []   # 经济作物价格历史
        }
        self.memory_length = 5  # 价格记忆长度（年）
        
        # 社会网络与学习机制（第二阶段新增）
        self.social_network = []  # 社会网络中的邻居农户ID列表
        self.network_strength = {}  # 与每个邻居的关系强度 {neighbor_id: strength}
        self.social_learning_weight = float(np_uniform(self, 0.2, 0.6))  # 社会学习权重（个体异质性）
        self.decision_mode = "hybrid"  # 决策模式：rational/social/hybrid
        self.neighbor_observation_history = []  # 邻居观察历史
        self.influence_received = []  # 受到的影响记录
        self.transaction_neighbors = []  # 土地交易可行性网络
        self.transaction_link_strength = {}  # 交易网络强度（局部匹配时刻画信任）
        self.weak_tie_contacts = []  # 跨圈层弱连接（供信息扩散与流转市场识别）
        
        # 信息传播与扩散（第二阶段新增）
        self.information_receptivity = float(np_uniform(self, 0.3, 0.8))  # 信息接受度
        self.success_sharing_propensity = float(np_uniform(self, 0.4, 0.9))  # 成功案例分享倾向
        self.conformity_tendency = float(np_uniform(self, 0.2, 0.7))  # 从众倾向
        
        # 社会资本相关
        self.is_village_elite = rng_random(self) < 0.2  # 20%概率为村精英（社会资本）
        self.social_network_strength = float(np_uniform(self, 0.5, 1.5))  # 社会网络强度
        
        # 劳动力相关
        self.total_labor_units = 365  # 总劳动力单位（天）
        self.agricultural_labor = 0   # 农业劳动分配
        self.off_farm_labor = 0      # 非农劳动分配
        
        # 收入结构（年度）
        self.agricultural_income = 0      # 农业收入
        self.planting_gross_income = 0.0  # 作物销售毛收入（验证口径）
        self.planting_net_income = 0.0    # 作物净收入（NFPS对齐口径）
        self.land_rental_income = 0      # 土地租金收入
        self.subsidy_income = 0          # 补贴收入
        self.scale_subsidy_income = 0    # @new 2025-12-08: 规模种粮补贴收入
        self.scale_subsidy_qualified = False  # @new 2025-12-08: 是否达到规模补贴门槛
        self.off_farm_income = 0         # 非农务工收入
        self.gross_income = 0            # 总收入（未减去支出）
        self.total_income = 0            # 总收入（向后兼容，等同于gross_income）
        self.in_kind_support_balance = 0.0  # 实物补贴折算价值
        self.in_kind_support_breakdown = {}
        self.subsidy_payment_history = []
        self.pending_subsidy_claim = None
        self.arrears_received = 0.0
        self.subsidy_arrears_owed = 0.0
        
        # 支出结构（年度）
        self.production_cost = 0         # 现金性农业生产成本（向后兼容）
        self.cash_production_cost = 0.0  # 现金性农业生产成本
        self.imputed_family_labor_cost = 0.0  # 家庭劳动力机会成本（不进入现金净收入）
        self.land_rental_expense = 0     # 土地租金支出
        self.total_expense = 0           # 现金总支出
        self.economic_total_expense = 0.0  # 含家庭劳动力机会成本的经济总支出
        
        # @new 2025-11-29: 劳动力市场相关
        # 理论：农户既是劳动力供给者，也可能是需求者（大户）
        self.hired_labor_cost = 0        # 雇佣劳动力支出（大户雇佣散户）
        self.labor_sale_income = 0       # 出售劳动力收入（被其他农户/企业雇佣）
        self.hired_labor_days = 0        # 雇佣的劳动天数
        self.requested_hired_labor_days = 0.0  # 当年生产计划推导出的雇工需求
        self.labor_sold_days = 0         # 出售的劳动天数
        self.family_labor_used = 0       # 自家使用的劳动力天数
        self.realized_off_farm_labor_days = 0.0  # 扣除本地卖工后的真实非农劳动天数
        self.labor_constraint_binding = False  # 劳动力约束是否绑定
        self.labor_reduction_ratio = 0.0
        self.service_contracts = []
        self.service_fee_expense = 0.0
        
        # @new 2025-12: 信贷约束相关
        # 专家审阅意见：规模户扩张依赖贷款，应有信贷约束
        self._initialize_farmer_credit_constraints()
        
        # @new 2025-12-28: P2满意化原则 (Simon, 1955) + P3情绪状态
        # @theory: 有限理性决策者不追求最优，只追求"足够好"
        # @literature: Simon (1955) "满意化"、Aspiration level理论
        self.satisfaction = 0.5  # 满意度 [0,1]，初始中等
        # 期望水平：基于初始收入×1.1（略高于当前，符合适应性期望理论）
        # 而非硬编码，因各农户初始收入差异大
        # @fix 2025-12-29: 使用已定义的变量计算初始收入估计
        _initial_land_area = sum(p.area for p in land_patches) if land_patches else 10.0
        _base_yield = 400  # kg/亩，合理默认值
        _grain_price = 2.58  # 元/kg，默认粮价
        _off_farm_modifier = 0.5  # 默认非农收入修正系数
        initial_income_estimate = (_initial_land_area * _base_yield * _grain_price * 0.5 + 
                                   _off_farm_modifier * 30000 * 0.5)
        self.aspiration_income = max(15000, initial_income_estimate * 1.1)
        self.recent_loss = False  # 近期是否经历损失
        self.consecutive_loss_years = 0  # 连续亏损年数
        self.risk_mood = 'neutral'  # 风险情绪：cautious/neutral/optimistic
        
        # 净收入与经济利润
        self.net_income = 0              # 现金净收入 = 总收入 - 现金支出
        self.cash_net_income = 0.0       # 显式保留现金净收入口径
        self.economic_profit = 0.0       # 经济利润 = 总收入 - (现金支出 + 家庭劳动力机会成本)
        self.disposable_income = 0       # 可支配收入（用于消费）
        
        # 消费和福利相关（新增）
        self.consumption_model = HouseholdConsumptionModel(config)  # 消费决策模型
        self.household_size = max(1, int(np_normal(self, 3.5, 0.8)))  # 家庭人口数（根据2020年农村平均家庭规模）

        age_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("age_dist", {})
        if age_cfg.get("type") == "normal":
            mean_age = age_cfg.get("mean", 50.0)
            std_age = age_cfg.get("std", 10.0)
            self.age = int(np.clip(np_normal(self, mean_age, std_age), 20, 80))
        else:
            self.age = int(np.clip(np_normal(self, 50.0, 10.0), 20, 80))
        
        # @new 2025-12: 初始化连续变量农户特征系统
        # 用连续变量替代离散标签，实现更精细的异质性刻画
        land_area = sum(p.area for p in self.land_patches) if self.land_patches else 10.0
        generation = self._determine_generation_type()
        self.characteristics = initialize_farmer_characteristics(
            age=self.age,
            land_area=land_area,
            initial_capital=initial_capital,
            generation=generation,
            random_state=np.random
        )
        
        # 精细农户类型（6-7分法）
        self.farmer_type, self.farmer_type_confidence = FarmerTypeClassifier.classify(self.characteristics)
        
        # @new 2025-11-30: 异质性预期机制
        # 理论：不同类型农户形成预期的方式不同
        # - 老年小农：适应性预期（后向看）
        # - 年轻规模户：有限理性预期（前向看）
        self._initialize_expectation_mechanism()
        
        self.target_operated_area = self._compute_target_operated_area()
        self.consumption_bundle = {}     # 消费束
        self.food_consumption_details = {}  # 食物消费详情
        self.welfare_indicators = {}     # 福利指标
        self.grain_consumption_kg = 0    # 粮食消费量（kg）
        
        # @new 2025-12-23: 行为惯性机制（FIX_REPORT要求）
        # 理论依据：Bocquého et al. (2014) 农户种植决策的路径依赖
        # 农户每期决策不独立，受历史行为影响
        self.behavioral_inertia = float(np_uniform(self, 0.3, 0.7))  # 行为惯性系数
        self.consecutive_same_crop_years = 0  # 连续种植同一作物年数
        self.crop_history = []  # 作物种植历史
        self.grain_expenditure = 0       # 粮食支出
        self.total_consumption_expenditure = 0  # 总消费支出
        self.engel_coefficient = 0       # 恩格尔系数
        self.food_security_index = 0     # 食物安全指数
        self.consumer_surplus = 0        # 消费者剩余
        self.is_in_poverty = False       # 是否贫困
        self.malnutrition_risk = False   # 营养不良风险
        
        # 土地租赁相关
        self.leased_out_patches = []     # 出租的地块
        self.leased_in_patches = []      # 租入的地块
        self.leased_patches = []         # 与企业口径对齐的租入地块别名
        self.reservation_prices = {}     # 保留价格
        self._operated_land_area_cache = 0.0
        self._operated_land_area_cache_version = None
        
        # 作物选择博弈相关（新增）
        self.planned_crop = None         # 本年计划种植的作物
        self.crop_choice_history = []    # 作物选择历史
        self.fertility_subsidy_negative_list = []  # 丧失地力补贴资格的经济作物列表
        
        # 生计策略演化相关（新增）
        self.income_history = []         # 收入历史（最近5年）
        self.agricultural_income_ratio_history = []  # 农业收入占比历史
        self.strategy_history = []       # 策略历史
        self.consecutive_high_non_farm_ratio = 0  # 连续高非农收入占比年数
        self.consecutive_low_agricultural_ratio = 0  # 连续低农业收入占比年数
        self.cohort_urbanized = False
        self.cohort_urbanization_reason = None
        self.cohort_urbanization_step = None
        self.strategy_switch_cooldown = 0  # @fix: 切换冷却期（步数）
        
        # 贝叶斯学习机制（新增）
        # @fix: 从配置读取贝叶斯学习参数
        bayesian_config = self.agent_config.get('agents', {}).get('farmer', {}).get('bayesian_learning', {})
        prior_alpha = bayesian_config.get('prior_alpha', 5.0)
        prior_beta = bayesian_config.get('prior_beta', 5.0)
        self.crop_belief = {
            'grain': (prior_alpha, prior_beta),  # (α, β) - 先验信念，代表有利证据和不利证据
            'cash_crop': (prior_alpha, prior_beta)
        }
        self.belief_history = []  # 信念历史记录
        
        # @fix: 从配置读取参考点参数
        reference_config = self.agent_config.get('agents', {}).get('farmer', {}).get('reference_point', {})
        internal_orientation_dist = reference_config.get('internal_orientation_dist', {})
        if internal_orientation_dist.get('type') == 'uniform':
            min_io = internal_orientation_dist.get('min', 0.5)
            max_io = internal_orientation_dist.get('max', 0.9)
            self.internal_orientation = float(np_uniform(self, min_io, max_io))
        else:
            self.internal_orientation = float(np_uniform(self, 0.5, 0.9))  # 默认值
        
        self.safety_net_factor = reference_config.get('safety_net_factor', 0.9)  # 心理安全网系数
        self.last_beliefs = None  # 上一期的信念状态

        # @fix: 40%概率初始为经济作物（基于CLES数据，粮食作物约60%）
        if rng_random(self) < 0.40:
            self.planned_crop = CropType.CASH_CROP
        else:
            self.planned_crop = CropType.GRAIN
        self._land_market_plan_step = -1
        self._labor_market_plan_step = -1
        self._skip_remaining_step_this_cycle = False
        
        # @fix 2025-11-26: 根据初始生计策略设置合理的劳动力分配
        # 确保初始状态的收入结构与生计策略一致
        self._initialize_labor_allocation()
        self._sync_characteristics_income_fields(use_strategy_anchor=True)
        self._align_characteristics_to_livelihood_strategy(blend=0.85)

        # @new 2026-05-02: 异质学习型主体（learner module）插件式集成
        # 通过 mechanism_switches.use_learning_farmer = True 启用，
        # 替代默认的 prospect_theory + bayesian_belief 决策路径。
        # 详细文档：agents/learner_modules.py
        self.learner = None
        try:
            mechanism_switches = self.model.config.get('mechanism_switches', {}) or {}
        except Exception:
            mechanism_switches = {}
        if mechanism_switches.get('use_learning_farmer', False):
            try:
                from agents.learner_modules import FarmerLearnerFactory
                self.learner = FarmerLearnerFactory.create_for_farmer(self)
            except Exception as exc:
                # 学习模块加载失败不应阻塞主流程，但要记录到 model 上以便审计
                if not hasattr(self.model, '_learner_init_errors'):
                    self.model._learner_init_errors = []
                self.model._learner_init_errors.append({
                    'farmer_id': getattr(self, 'unique_id', None),
                    'error': str(exc),
                })
                self.learner = None

    def _initialize_labor_allocation(self):
        """
        根据初始生计策略设置合理的劳动力分配
        @fix 2025-11-26: 解决初始化时劳动力分配为0的问题
        
        按照生计策略设置劳动力分配和非农工资水平：
        - 纯农户：75-90%农业劳动，较低非农工资水平（0.6-0.9）
        - 兼业农户：30-60%农业劳动，中等非农工资水平（0.8-1.2）
        - 非农户：5-20%农业劳动，较高非农工资水平（1.0-1.5）
        
        @critical 2025-12: 改用连续变量替代离散判断
        专家意见："用连续变量替代离散标签"
        农业劳动分配基于 agricultural_dependence 连续变量
        """
        # @critical 2025-12: 使用连续变量进行决策（而非离散的if-else）
        # 如果有 characteristics，使用连续变量计算
        if hasattr(self, 'characteristics') and self.characteristics is not None:
            agri_dep = self.characteristics.agricultural_dependence
            
            # @fix 2026-01-14: 修正劳动力分配映射
            # 问题：原[0.60, 0.95]导致非农户(agri_dep=0.1)农业劳动占64%，与收入结构矛盾
            # 修正：[0.15, 0.95] - 非农户仅15%时间维持自留地，纯农户95%
            # 实证依据：NBS 2022外出务工者平均离家9.5个月/年
            # 农业劳动比例：agri_dep ∈ [0, 1] → agri_labor_ratio ∈ [0.15, 0.95]
            base_labor_ratio = 0.15 + 0.80 * agri_dep  # 线性映射到 [0.15, 0.95]
            noise = float(np_uniform(self, -0.03, 0.03))  # ±3%随机噪声（减小波动）
            # @fix 2026-01-14: clip下限从0.55改为0.12，与[0.15, 0.95]设计保持一致
            labor_ratio = np.clip(base_labor_ratio + noise, 0.12, 0.98)
            
            self.agricultural_labor = self.total_labor_units * labor_ratio
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            
            # 非农工资水平：与农业依赖度负相关
            # agri_dep高 → 非农能力低 → 工资水平低
            base_wage_level = 1.5 - 0.8 * agri_dep  # 映射到 [0.7, 1.5]
            wage_noise = float(np_uniform(self, -0.1, 0.1))
            self.off_farm_wage_level = np.clip(base_wage_level + wage_noise, 0.5, 1.6)
            return
        
        # 向后兼容：如果没有 characteristics，使用旧的离散逻辑
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户：75-90%的劳动力用于农业
            self.agricultural_labor = self.total_labor_units * np_uniform(self, 0.75, 0.90)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 调整非农工资水平（纯农户通常非农收入能力较低）
            self.off_farm_wage_level = min(self.off_farm_wage_level, float(np_uniform(self, 0.6, 0.9)))
        
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业农户：30-60%的劳动力用于农业
            self.agricultural_labor = self.total_labor_units * np_uniform(self, 0.30, 0.60)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 保持默认的工资水平（0.8-1.2）
        
        else:  # NON_FARMER
            # 非农户：5-20%的劳动力用于农业（仅维持自给自足）
            self.agricultural_labor = self.total_labor_units * np_uniform(self, 0.05, 0.20)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 提高非农工资水平（非农户通常有更好的非农就业能力）
            self.off_farm_wage_level = max(self.off_farm_wage_level, float(np_uniform(self, 1.0, 1.5)))
    
    def prepare_land_market_decisions(self):
        """Prepare current-year livelihood and crop plans before land-market clearing."""
        current_step = int(getattr(self.model, 'step_count', 0) or 0)
        if self._land_market_plan_step == current_step:
            return
        fixed_action = getattr(self, "fixed_rule_current_action", None)
        if fixed_action in {"maintain_grain", "expand_cash", "exit"} or (fixed_action or "").startswith("portfolio_"):
            self._land_market_plan_step = current_step
            if fixed_action == "exit":
                self._skip_remaining_step_this_cycle = not action_income_enabled(self.model.config)
                self._labor_market_plan_step = current_step
            return

        self._skip_remaining_step_this_cycle = False
        self.age += 1
        if not self._defer_age_exit_to_population_manager() and self._check_age_exit():
            self._skip_remaining_step_this_cycle = True
            self._land_market_plan_step = current_step
            self._labor_market_plan_step = current_step
            return

        self.update_farmer_characteristics()
        self.update_farmer_scale()
        self.attempt_land_reclaim()
        self.crop_choice_game()
        self._land_market_plan_step = current_step

    def _defer_age_exit_to_population_manager(self) -> bool:
        """
        PopulationManagerV2 owns structural farmer exit and land reallocation.

        When it is active, a FarmerAgent should not independently retire at the
        beginning of the production year: doing so skips current-year production
        while leaving land tenure to be closed only at year end.
        """
        return getattr(getattr(self, 'model', None), 'population_manager', None) is not None

    def prepare_labor_market_decisions(self):
        """Refresh abandonment and labor allocation after land transfers."""
        current_step = int(getattr(self.model, 'step_count', 0) or 0)
        self.prepare_land_market_decisions()
        if self._skip_remaining_step_this_cycle or self._labor_market_plan_step == current_step:
            return

        if action_income_enabled(self.model.config):
            profile = self.estimate_production_labor_profile()
            self.agricultural_labor = profile['family_labor_used']
            self.off_farm_labor = max(0.0, self.total_labor_units - self.agricultural_labor)
            self._labor_market_plan_step = current_step
            return

        self.make_abandonment_decision()
        self.allocate_labor()
        self._labor_market_plan_step = current_step

    def _execute_post_market_cycle(self):
        """Run production, settlement, and adaptation after factor markets clear."""
        self.make_production_decisions()
        self.harvest_and_settle()
        self.make_consumption_decision()
        self.livelihood_strategy_evolution()
        self._update_social_network_dynamics()
        self._update_satisfaction_and_mood()
        self._record_year_experience_to_learner()

    def _legacy_pre_market_cycle(self):
        """
        @theory
        Legacy one-pass farmer cycle kept only as a readable reference.
        MainModel now runs land and labor planning explicitly before markets clear.
        """
        # 阶段-1：年龄增长与退出机制
        self.age += 1  # 年龄增长1岁
        if self._check_age_exit():
            return  # 退出农业后不再执行后续决策
        
        # 阶段0：更新连续变量农户特征（新增）
        self.update_farmer_characteristics()
        
        # 阶段一：更新农户规模分类（基于当前经营面积）
        self.update_farmer_scale()
        
        # 阶段1.5：土地回收检查（非农转农时可能收回土地）
        self.attempt_land_reclaim()
        
        # 阶段二：首要博弈 - 作物选择
        self.crop_choice_game()
        
        # 阶段2.5：撂荒决策（新增）- 未流转土地在低收益时可能撂荒
        self.make_abandonment_decision()
        
        # @fix: 阶段三：土地租赁市场参与 - 已在模型层面统一调用，这里不再重复调用
        # self.participate_land_rental_market()  # 已移除，由model.conduct_land_rental_market()统一调用
        
        # 阶段四：劳动力分配决策
        self.allocate_labor()
        
        # 阶段五：生产决策
        self.make_production_decisions()
        
        # 阶段六：收获与结算
        self.harvest_and_settle()
        
        # 阶段6.5：消费决策（新增）
        self.make_consumption_decision()
        
        # 阶段七：动态博弈 - 生计策略演化
        self.livelihood_strategy_evolution()
        
        # @new 2025-12-28: 社会网络动态重构
        # @theory: 进城务工导致村庄空心化，原有熟人网络断裂
        # 非农户的网络影响力随时间衰减
        self._update_social_network_dynamics()
        
        # @new 2025-12-28: P2满意化原则 + P3情绪状态更新
        self._update_satisfaction_and_mood()

        # @new 2026-05-02: learner module 经验记录与 Q-learning 更新
        # 在所有市场清算与收入计算完成后，把当年决策与结果送入 buffer
        self._record_year_experience_to_learner()

    def _record_year_experience_to_learner(self):
        """
        @new 2026-05-02: 把当年的 (state, action, reward) 送入 learner module。

        仅在启用 learner_module 时执行。对 StrategicLearner 还会触发
        Q-table 的 Bellman 更新，使下年决策真正"从经验中学习"。
        """
        if getattr(self, 'learner', None) is None:
            return
        try:
            from agents.learner_modules import (
                FarmerExperience,
                build_default_env_signals,
                StrategicLearner,
            )

            env_signals = build_default_env_signals(self)

            # 构造单年经验
            crop_choice = (
                'grain'
                if getattr(self, 'last_crop_choice', None) == CropType.GRAIN
                else 'cash_crop'
            )
            transfer_choice = (self.learner.last_action_transfer or 'keep')
            current_step = getattr(self.model, 'step_count', 0)
            initial_year = getattr(self.model, 'initial_year', 2010)
            current_year = int(initial_year) + int(current_step)

            net_income = float(getattr(self, 'net_income', 0.0) or 0.0)
            subsidy = float(getattr(self, 'subsidy_income', 0.0) or 0.0)
            land_area_mu = max(
                1.0,
                float(getattr(self, 'land_area_mu', None)
                      or getattr(self, 'land_area', 1.0))
            )
            # reward 标准化为单亩净收入（对各类规模农户公平）
            reward = net_income / land_area_mu

            experience = FarmerExperience(
                year=current_year,
                crop_choice=crop_choice,
                transfer_choice=transfer_choice,
                income=net_income,
                subsidy_received=subsidy,
                grain_price=float(env_signals.get('grain_premium', 0.0)),
                cash_price=0.0,  # 在 grain_premium 已蕴含相对信息
                neighbor_grain_share=float(env_signals.get('neighbor_grain_share', 0.5)),
                weather_shock=bool(getattr(self.model.environment, 'extreme_weather_this_year', False)),
                rent_offer_received=float(env_signals.get('rent_offer_per_mu', 0.0)),
                reward_realized=reward,
            )
            self.learner.observe(experience)

            # Strategic learner 触发 Q-table 更新
            if isinstance(self.learner, StrategicLearner):
                self.learner.update_after_outcome(reward, env_signals)
        except Exception as exc:
            if not hasattr(self.model, '_learner_runtime_errors'):
                self.model._learner_runtime_errors = []
            self.model._learner_runtime_errors.append({
                'farmer_id': getattr(self, 'unique_id', None),
                'error': str(exc),
                'phase': 'record_experience',
            })
    
    def step(self):
        """
        Run the annual farmer cycle using the explicit planning phases managed by MainModel.
        """
        if getattr(self, "fixed_rule_current_action", None) in {"maintain_grain", "expand_cash", "exit"} or getattr(self, "household_planned_grain_share", None) is not None:
            if getattr(self, "fixed_rule_current_action", None) == "exit":
                return
            return self._execute_post_market_cycle()
        self.prepare_labor_market_decisions()
        if self._skip_remaining_step_this_cycle:
            return
        return self._execute_post_market_cycle()

    def make_abandonment_decision(self):
        """
        @theory 撂荒决策机制
        @critical 2025-12: 专家审阅意见 - 撂荒机制缺失
        
        理论依据：
        1. 撂荒是中国农村普遍现象，特别是在山区、丘陵地区
        2. 数据：全国撂荒率约10-15%（部分地区高达20-30%）
        3. 主要发生在劳动力外流的Non_Farmer群体
        
        @fix 2025-12-01: 添加补贴对撂荒的抑制效果
        参考：Zhang & Xiong (2025) 补贴显著抑制耕地撂荒
        
        撂荒触发条件：
        1. 农户类型为Non_Farmer（主要收入来自非农）
        2. 土地无法成功流转（无人愿意承租）
        3. 自耕收益低于务工机会成本
        4. 地块细碎化严重（fragmentation_index低）
        5. 土地质量差（soil_quality低）
        
        抑制因素：
        - 有粮食补贴时，撂荒概率降低
        """
        # 获取撂荒配置参数
        abandonment_config = self.agent_config.get('agents', {}).get('farmer', {}).get('land_abandonment', {})
        base_rate = abandonment_config.get('base_rate', 0.02)  # @fix: 基础撂荒率从10%降到2%
        max_rate = abandonment_config.get('max_rate', 0.10)  # @fix: 最高撂荒率从25%降到10%
        low_quality_threshold = abandonment_config.get('low_quality_threshold', 0.3)
        fragmentation_threshold = abandonment_config.get('fragmentation_threshold', 0.5)
        subsidy_suppression = abandonment_config.get('subsidy_suppression_factor', 0.5)
        
        # @fix 2025-12-01: 检查是否有补贴，有补贴则抑制撂荒
        has_subsidy = getattr(self, 'subsidy_income', 0) > 0
        
        # 对每个自有且未流转的地块进行撂荒决策
        for patch in self.land_patches:
            if patch.owner != self:
                continue
            if patch.leased_to is not None:
                continue  # 已流转的地块不考虑撂荒
            if patch.abandoned:
                continue  # 已撂荒的继续撂荒
            
            # 计算撂荒概率
            abandonment_prob = self._calculate_abandonment_probability(
                patch, base_rate, max_rate, 
                low_quality_threshold, fragmentation_threshold
            )
            
            # @fix 2025-12-01: 应用补贴抑制因子
            if has_subsidy:
                abandonment_prob *= (1 - subsidy_suppression)
            
            # 随机决定是否撂荒
            if rng_random(self) < abandonment_prob:
                patch.set_abandoned()
                # 记录撂荒事件
                if hasattr(self.model, 'abandonment_records'):
                    self.model.abandonment_records.append({
                        'step': self.model.step_count,
                        'farmer_id': self.unique_id,
                        'patch_id': patch.get_patch_id(),
                        'area': patch.area,
                        'reason': 'low_return_opportunity_cost'
                    })
    
    def _calculate_abandonment_probability(self, patch, base_rate, max_rate, 
                                            low_quality_threshold, fragmentation_threshold):
        """
        @theory 计算地块撂荒概率
        
        @fix 2026-01-14: 改用乘法模型替代累加模型
        问题：原累加模型导致撂荒率6-8%，远高于统计数据1-2%
        修正：使用概率乘法模型，增强补贴抑制效果
        数据来源：农业农村部(2020)全国撂荒率1.14%
        
        影响因素（乘法模型）：
        - 基础概率：0.005 (0.5%)
        - 各因素以乘数形式调整
        """
        # 基础概率设为0.5%（对标统计数据）
        prob = 0.005
        
        # 1. 生计策略调整（乘数）
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # agri_dep=0 → 3x, agri_dep=1 → 0.5x
            strategy_multiplier = 0.5 + 2.5 * (1 - agri_dep)
        else:
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_multiplier = 3.0
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                strategy_multiplier = 1.5
            else:
                strategy_multiplier = 0.5
        
        prob *= strategy_multiplier
        
        # 2. 土地质量影响（乘数）
        if hasattr(patch, 'soil_quality') and patch.soil_quality < low_quality_threshold:
            quality_penalty = 1 + (low_quality_threshold - patch.soil_quality) / low_quality_threshold
            prob *= quality_penalty
        
        # 3. 细碎化程度影响（乘数）
        if hasattr(patch, 'fragmentation_index') and patch.fragmentation_index < fragmentation_threshold:
            frag_penalty = 1 + 0.5 * (fragmentation_threshold - patch.fragmentation_index) / fragmentation_threshold
            prob *= frag_penalty
        
        # 4. 地块面积影响（乘数）
        if patch.area < 1.0:  # 小于1亩
            prob *= 1.3
        elif patch.area < 3.0:  # 1-3亩
            prob *= 1.1
        
        # 5. 年龄影响（乘数）
        if self.age >= 65:
            prob *= 1.5
        elif self.age >= 55:
            prob *= 1.2
        
        # 6. 机会成本考虑（乘数）
        off_farm_wage = self.model.environment.get_market_price('wage_rate')
        expected_agri_return = self._estimate_patch_return(patch)
        if expected_agri_return < off_farm_wage * 0.3:
            prob *= 1.5
        
        # 限制在最大撂荒率以内（但max_rate也需调低）
        return min(prob, min(max_rate, 0.05))
    
    def _estimate_patch_return(self, patch):
        """
        @theory 估算地块预期年收益（每亩）
        用于撂荒决策的机会成本比较
        @fix 2025-12-23: 从配置文件读取基础产量
        """
        # 从配置读取基础产量
        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        base_yield = base_yields.get('grain', 420)  # 默认420kg/亩
        
        grain_price = self._get_output_price_per_kg('crop_price')
        
        # 考虑土地质量
        quality_factor = getattr(patch, 'soil_quality', 0.7)
        adjusted_yield = base_yield * quality_factor
        
        # 毛收入
        gross_income = adjusted_yield * grain_price
        
        # 成本（简化估算：从配置读取或默认400元/亩）
        cost_config = production_config.get('cost_structure', {})
        cost = cost_config.get('input_requirements', {}).get('other_cost_per_mu', {}).get('grain', 400)
        
        # 补贴
        subsidy = self.model.environment.get_subsidy_for_farmer(self, 1.0) if hasattr(self.model.environment, 'get_subsidy_for_farmer') else 100
        
        # 净收益
        net_return = gross_income - cost + subsidy
        
        return max(0, net_return)

    def crop_choice_game(self):
        """
        @theory
        首要博弈：作物选择博弈 - 前景理论与社会学习的混合决策框架
        第二阶段核心改进：从完全理性转向有限理性+社会学习
        新增：贝叶斯学习机制
        决策过程：
        1. 更新动态参考点和风险偏好
        2. 进行贝叶斯学习
        3. 观察邻居行为（社会学习）
        4. 结合前景理论效用、贝叶斯信念和社会影响
        5. 做出最终决策
        
        @fix 2026-01-14: 添加消融实验机制开关支持
        - crop_choice_enabled=False 时强制种粮
        - forced_grain=True 时强制种粮
        """
        # @fix 2026-01-14: 消融实验机制开关检查
        # 如果配置中crop_choice_enabled=False或forced_grain=True，强制种粮
        mechanism_switches = self.model.config.get('mechanism_switches', {})
        crop_config = self.model.config.get('crop', {})
        
        crop_choice_enabled = mechanism_switches.get('crop_choice_enabled', True)
        forced_grain = crop_config.get('forced_grain', False)
        
        if not crop_choice_enabled or forced_grain:
            # 消融实验：强制种粮，跳过作物选择博弈
            self.planned_crop = CropType.GRAIN
            return

        # @new 2026-05-02: 学习型主体短路决策
        # 当 mechanism_switches.use_learning_farmer = True 时，
        # 由 learner module 直接做出决策，跳过 prospect theory 路径。
        # 这是路径A保守版的核心创新：把"同公式不同参数"的伪异质性
        # 改为"按 FarmerType 不同决策模块"的真异质性。
        if getattr(self, 'learner', None) is not None:
            from agents.learner_modules import build_default_env_signals
            env_signals = build_default_env_signals(self)
            try:
                crop_decision = self.learner.decide_crop(env_signals)
                self.planned_crop = (
                    CropType.GRAIN if crop_decision == 'grain' else CropType.CASH_CROP
                )
                # 记录 learner-driven 决策到 history 用于后续验证与审计
                decision_record = {
                    'period': len(self.crop_choice_history) + 1,
                    'choice': self.planned_crop,
                    'decision_source': self.learner.learner_type,
                    'last_state': self.learner.last_state,
                }
                self.crop_choice_history.append(decision_record)
                # 同步行为惯性追踪
                if len(self.crop_history) > 0 and self.crop_history[-1] == self.planned_crop:
                    self.consecutive_same_crop_years += 1
                else:
                    self.consecutive_same_crop_years = 1
                self.crop_history.append(self.planned_crop)
                if len(self.crop_history) > 10:
                    self.crop_history = self.crop_history[-10:]
                return
            except Exception as exc:
                # learner 失败时降级到 prospect theory 路径，但记录错误
                if not hasattr(self.model, '_learner_runtime_errors'):
                    self.model._learner_runtime_errors = []
                self.model._learner_runtime_errors.append({
                    'farmer_id': getattr(self, 'unique_id', None),
                    'error': str(exc),
                    'phase': 'decide_crop',
                })

        # @refactor 2025-12: 用连续变量判断是否跳过作物选择。
        # @fix 2026-06: 农村非农户保留的少量自营地不应继续走完整
        # "粮食 vs 经作"利润最大化路径。现实中这类土地更接近口粮地、
        # 低劳动投入地或等待流转地；让它选择劳动密集型经作会人为制造
        # 粮食面积下滑。大规模非农经营者仍保留常规作物选择。
        skip_crop_choice = False
        skip_crop_choice_reason = None
        operated_area_for_crop_choice = 0.0
        try:
            operated_area_for_crop_choice = float(self.get_operated_land_area() or 0.0)
        except (TypeError, ValueError):
            operated_area_for_crop_choice = 0.0
        non_farmer_retained_land_fallback = bool(
            crop_config.get("non_farmer_retained_land_grain_fallback_enabled", True)
        )
        retained_land_max_area = float(
            crop_config.get("non_farmer_retained_land_max_area_mu", 12.0) or 12.0
        )
        retained_land_tolerance = max(
            0.0,
            float(crop_config.get("non_farmer_retained_land_threshold_tolerance_mu", 0.25) or 0.0),
        )
        retained_land_effective_max_area = retained_land_max_area + retained_land_tolerance
        if hasattr(self, 'characteristics') and self.characteristics:
            # 农业依赖度极低时（< 0.1），跳过作物选择
            if self.characteristics.agricultural_dependence < 0.1:
                skip_crop_choice = True
                skip_crop_choice_reason = "skip_low_agricultural_dependence"
            elif (
                non_farmer_retained_land_fallback
                and self.livelihood_strategy == LivelihoodStrategy.NON_FARMER
                and operated_area_for_crop_choice <= retained_land_effective_max_area
            ):
                skip_crop_choice = True
                skip_crop_choice_reason = "non_farmer_retained_land_grain_fallback"
        else:
            # 回退：使用离散判断
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                skip_crop_choice = True
                skip_crop_choice_reason = "non_farmer_retained_land_grain_fallback"
        
        if skip_crop_choice:
            # 保留地/低农业依赖农户采用低劳动、政策合规的种粮 fallback。
            self.planned_crop = CropType.GRAIN
            if len(self.crop_history) > 0 and self.crop_history[-1] == self.planned_crop:
                self.consecutive_same_crop_years += 1
            else:
                self.consecutive_same_crop_years = 1
            self.crop_history.append(self.planned_crop)
            if len(self.crop_history) > 10:
                self.crop_history = self.crop_history[-10:]
            decision_record = {
                'period': len(self.crop_choice_history) + 1,
                'choice': self.planned_crop,
                'decision_source': skip_crop_choice_reason or 'skip_low_agricultural_dependence',
                'skip_crop_choice': True,
                'operated_area': operated_area_for_crop_choice,
                'agricultural_dependence': (
                    getattr(self.characteristics, 'agricultural_dependence', None)
                    if hasattr(self, 'characteristics') and self.characteristics
                    else None
                ),
            }
            self.crop_choice_history.append(decision_record)
            return
        
        # ================================================================
        # @refactor 2025-12: 简化决策模型
        # 专家建议：移除贝叶斯学习，以前景理论为核心
        # 社会学习仅作为"信息获取"渠道（影响概率估计），不直接加成效用
        # ================================================================
        
        # 阶段1：更新动态参考点（隐含适应性预期）
        self.update_reference_point()
        self.update_risk_preference()
        self.update_prospect_theory_parameters()
        
        # 阶段2：获取市场信息
        grain_price = self.model.environment.get_market_price('crop_price')
        cash_crop_price = self.model.environment.get_market_price('cash_crop')
        grain_subsidies_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
        
        # 阶段3：社会学习 - 仅作为信息获取渠道
        # 观察邻居的种植结果来更新对概率分布的认知
        neighbor_crop_outcomes = self._collect_neighbor_crop_outcomes()
        
        # 阶段4：生成情景并进行前景理论分析
        # 情景概率受社会学习影响（而非效用直接加成）
        grain_scenarios = self._generate_informed_scenarios(
            'grain', grain_price, grain_subsidies_rate, neighbor_crop_outcomes
        )
        cash_crop_scenarios = self._generate_informed_scenarios(
            'cash_crop', cash_crop_price, 0, neighbor_crop_outcomes
        )
        
        # 使用前景理论计算效用（核心决策机制）
        grain_utility = self.prospect_theory.decision_utility(
            outcomes=[s['profit'] for s in grain_scenarios],
            probabilities=[s['probability'] for s in grain_scenarios],
            reference_point=self.reference_point
        )
        
        cash_crop_utility = self.prospect_theory.decision_utility(
            outcomes=[s['profit'] for s in cash_crop_scenarios],
            probabilities=[s['probability'] for s in cash_crop_scenarios],
            reference_point=self.reference_point
        )
        
        # 阶段5：基于效用比较做出决策
        # @new 2025-12-23: 添加行为惯性加成（FIX_REPORT要求）
        grain_inertia_bonus = self._calculate_behavioral_inertia_bonus(CropType.GRAIN)
        cash_crop_inertia_bonus = self._calculate_behavioral_inertia_bonus(CropType.CASH_CROP)
        
        adjusted_grain_utility = grain_utility + grain_inertia_bonus
        adjusted_cash_crop_utility = cash_crop_utility + cash_crop_inertia_bonus
        
        if adjusted_grain_utility > adjusted_cash_crop_utility:
            self.planned_crop = CropType.GRAIN
        else:
            self.planned_crop = CropType.CASH_CROP
        
        # 阶段6：有限探索（防止系统锁定）
        # @fix: 探索概率降低到3%，企业级别的稳定性
        if rng_random(self) < 0.03:
            self.planned_crop = CropType.CASH_CROP if self.planned_crop == CropType.GRAIN else CropType.GRAIN

        profit_guard_applied = False
        expected_grain_profit = sum(
            float(s.get('profit', 0.0) or 0.0) * float(s.get('probability', 0.0) or 0.0)
            for s in grain_scenarios
        )
        expected_cash_crop_profit = sum(
            float(s.get('profit', 0.0) or 0.0) * float(s.get('probability', 0.0) or 0.0)
            for s in cash_crop_scenarios
        )
        profitability_guard_enabled = bool(
            crop_config.get("cash_crop_profitability_guard_enabled", True)
        )
        margin_per_mu = float(
            crop_config.get("cash_crop_profit_guard_margin_yuan_per_mu", 50.0) or 50.0
        )
        if profitability_guard_enabled and self.planned_crop == CropType.CASH_CROP:
            area_basis = max(1.0, float(operated_area_for_crop_choice or 0.0))
            allowed_gap = max(0.0, margin_per_mu) * area_basis
            if expected_grain_profit - expected_cash_crop_profit > allowed_gap:
                self.planned_crop = CropType.GRAIN
                profit_guard_applied = True
        
        # @new 2025-12-23: 更新作物种植历史（用于行为惯性计算）
        if len(self.crop_history) > 0 and self.crop_history[-1] == self.planned_crop:
            self.consecutive_same_crop_years += 1
        else:
            self.consecutive_same_crop_years = 1
        self.crop_history.append(self.planned_crop)
        if len(self.crop_history) > 10:  # 保留最近10年记录
            self.crop_history = self.crop_history[-10:]
        
        # 阶段7：记录决策数据（简化版）
        decision_record = {
            'period': len(self.crop_choice_history) + 1,
            'choice': self.planned_crop,
            'reference_point': self.reference_point,
            'risk_aversion': self.risk_aversion,
            'grain_utility': grain_utility,
            'cash_crop_utility': cash_crop_utility,
            'utility_diff': grain_utility - cash_crop_utility,
            'grain_price': grain_price,
            'cash_crop_price': cash_crop_price,
            'inertia_bonus_grain': grain_inertia_bonus,
            'inertia_bonus_cash': cash_crop_inertia_bonus,
            'expected_grain_profit': expected_grain_profit,
            'expected_cash_crop_profit': expected_cash_crop_profit,
            'cash_crop_profitability_guard_applied': profit_guard_applied,
            'cash_crop_profit_guard_margin_yuan_per_mu': margin_per_mu,
        }
        
        self.crop_choice_history.append(decision_record)

    def _apply_final_crop_profitability_guard(self, operated_area, phase="production"):
        """Re-check cash-crop plans against realized operated area before production."""
        if self.planned_crop != CropType.CASH_CROP:
            return False

        try:
            area = max(0.0, float(operated_area or 0.0))
        except (TypeError, ValueError):
            area = 0.0
        if area <= 0:
            return False

        crop_config = self.model.config.get('crop', {}) if hasattr(self.model, 'config') else {}
        if not bool(crop_config.get("cash_crop_profitability_guard_enabled", True)):
            return False

        margin_per_mu = float(
            crop_config.get("cash_crop_profit_guard_margin_yuan_per_mu", 50.0) or 50.0
        )
        env = getattr(self.model, 'environment', None)
        grain_price = (
            env.get_market_price('crop_price')
            if env is not None and hasattr(env, 'get_market_price')
            else 0.0
        )
        cash_crop_price = (
            env.get_market_price('cash_crop')
            if env is not None and hasattr(env, 'get_market_price')
            else 0.0
        )
        grain_subsidy_rate = (
            env.get_policy_param('grain_subsidies_rate')
            if env is not None and hasattr(env, 'get_policy_param')
            else 0.0
        )

        grain_profit = self.calculate_grain_expected_profit(
            grain_price,
            grain_subsidy_rate,
            area,
        )
        cash_crop_profit = self.calculate_cash_crop_expected_profit(
            cash_crop_price,
            area,
        )
        allowed_gap = max(0.0, margin_per_mu) * max(1.0, area)
        if grain_profit - cash_crop_profit <= allowed_gap:
            return False

        self.planned_crop = CropType.GRAIN
        self.crop_choice_history.append({
            'period': len(self.crop_choice_history) + 1,
            'choice': self.planned_crop,
            'decision_source': 'final_crop_profitability_guard',
            'phase': phase,
            'operated_area': area,
            'expected_grain_profit': grain_profit,
            'expected_cash_crop_profit': cash_crop_profit,
            'cash_crop_profitability_guard_applied': True,
            'cash_crop_profit_guard_margin_yuan_per_mu': margin_per_mu,
        })
        return True

    def _calculate_behavioral_inertia_bonus(self, crop_type):
        """
        @theory 行为惯性加成：连续种植同一作物会产生学习效应和心理舒适区
        @new 2025-12-23: 实现FIX_REPORT要求的行为惯性机制
        
        理论依据：
        - Bocquého et al. (2014): 农户种植决策的路径依赖
        - 学习效应：连续种植同一作物积累经验，降低风险感知
        - 心理舒适区：避免变化带来的不确定性
        
        Returns:
            float: 惯性加成值（0-0.15之间）
        """
        if len(self.crop_history) == 0:
            return 0.0
        
        if crop_type == self.crop_history[-1]:
            years = self.consecutive_same_crop_years
            # 惯性加成：5-15%，随年数递增但边际递减
            # 使用指数衰减函数：1 - e^(-0.5 * years)
            max_bonus = 0.15 * self.behavioral_inertia
            bonus = max_bonus * (1 - np.exp(-0.5 * years))
            return bonus
        
        return 0.0

    def _collect_neighbor_crop_outcomes(self):
        """
        @refactor 2025-12: 简化社会学习 - 仅收集邻居种植结果作为信息
        
        @fix 2026-01-14: 添加消融实验开关支持
        - social_learning_enabled=False 时返回空结果
        - acquaintance_network_enabled=False 时禁用熟人网络
        
        专家建议：社会学习应作为信息获取渠道，影响对概率的估计
        而非直接作为效用函数的加成项
        """
        outcomes = {'grain': [], 'cash_crop': []}
        
        # @fix 2026-01-14: 消融实验机制开关检查
        mechanism_switches = self.model.config.get('mechanism_switches', {})
        social_config = self.model.config.get('agents', {}).get('farmer', {}).get('social', {})
        
        acquaintance_enabled = mechanism_switches.get('acquaintance_network_enabled', True)
        social_learning_enabled = social_config.get('social_learning_enabled', True)
        
        if not acquaintance_enabled or not social_learning_enabled:
            # 消融实验：禁用社会学习，返回空结果
            return outcomes
        
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor and hasattr(neighbor, 'crop_choice_history') and neighbor.crop_choice_history:
                last_record = neighbor.crop_choice_history[-1]
                choice = last_record.get('choice')
                # 获取邻居的收入情况
                income = (
                    neighbor._get_income_signal()
                    if hasattr(neighbor, '_get_income_signal')
                    else getattr(neighbor, 'net_income', getattr(neighbor, 'total_income', 0))
                )
                if choice == CropType.GRAIN:
                    outcomes['grain'].append(income)
                elif choice == CropType.CASH_CROP:
                    outcomes['cash_crop'].append(income)
        
        return outcomes

    def _generate_informed_scenarios(self, crop_type, base_price, subsidy_rate, neighbor_outcomes):
        """
        @refactor 2025-12: 基于社会信息的情景生成
        
        邻居的种植结果会影响对概率分布的认知（而非直接加成效用）
        这符合专家建议：社会学习作为信息获取渠道
        
        @fix 2025-12-01: 修复收益参数，使粮食和经济作物收益差距符合中国实际情况
        参考数据：《中国农村统计年鉴》2022、CLES调查数据
        - 粮食：产量约500kg/亩，价格2.6元/kg，收入1300元/亩，成本约800元/亩
        - 经济作物：收益约2000-4000元/亩，但劳动密集、风险高、成本高
        
        @fix 2025-12-01 v2: 纳入劳动力成本差异
        - 粮食：机械化程度高，12天/亩（农户）
        - 经济作物：劳动密集，30天/亩
        """
        # 从环境获取价格统计
        price_stats = self._get_output_price_statistics_per_kg(
            'grain' if crop_type == 'grain' else 'cash_crop'
        )
        price_mean = price_stats['mean']
        price_std = price_stats['std']
        
        # 获取劳动力市场参数
        labor_config = self.model.config.get('labor_market', {}).get('labor_requirements', {})
        daily_wage = self.model.environment.get_market_price('wage_rate')
        
        # ================================================================
        # @fix 2025-12-01: 调整产量和成本参数，纳入劳动力成本差异
        # @fix 2025-12-23: 从配置文件读取基础产量，保持一致性
        # ================================================================
        operated_area = max(self.get_operated_land_area(), 1.0)
        
        if crop_type == 'grain':
            base_yield = self._calculate_deterministic_expected_yield(
                CropType.GRAIN,
                operated_area,
            )
            
            # 物资成本率40%（种子、化肥、农药等）
            material_cost_ratio = 0.40
            
            # 劳动力成本：12天/亩（农户不完全机械化）
            labor_days_per_mu = labor_config.get('grain', {}).get('base', 12)
            labor_cost_per_mu = labor_days_per_mu * daily_wage

            # @new: 粮食的风险波动较小（政府托市）
            price_volatility_factor = 0.8  # 波动降低20%
            pessimistic_yield_factor = 0.80
            optimistic_yield_factor = 1.20
        else:
            base_yield = self._calculate_deterministic_expected_yield(
                CropType.CASH_CROP,
                operated_area,
            )
            
            # 物资成本率45%（需要更多农药、肥料等）
            material_cost_ratio = 0.45
            
            # 劳动力成本：30天/亩（劳动密集型）
            labor_days_per_mu = labor_config.get('cash_crop', {}).get('base', 30)
            labor_cost_per_mu = labor_days_per_mu * daily_wage

            # 经济作物风险更高
            price_volatility_factor = 1.2  # 波动增加20%
            pessimistic_yield_factor = 0.80
            optimistic_yield_factor = 1.20

        # @fix 2025-12: 正确计算补贴金额（使用environment的标准接口）
        if crop_type == 'grain' and hasattr(self.model, 'environment'):
            subsidy_result = self.model.environment.calculate_subsidy_for_actor(operated_area, actor_type='farmer')
            grain_subsidy_amount = subsidy_result.get('cash', 0) + subsidy_result.get('in_kind_value', 0)
        else:
            grain_subsidy_amount = 0
        grain_security_premium = (
            self._get_grain_security_decision_premium(operated_area)
            if crop_type == 'grain'
            else 0.0
        )
        
        # 基础情景概率
        pessimistic_prob = 0.2
        baseline_prob = 0.6
        optimistic_prob = 0.2

        # @fix 2026-01-14: 增强社会学习机制 - 使用个体异质性权重进行贝叶斯更新
        # 理论依据: 社会学习权重(social_learning_weight)表示个体对邻居信息的依赖程度
        # 公式: P_subjective = (1-w) * P_objective + w * P_neighbor_success
        neighbor_data = neighbor_outcomes.get(crop_type, [])
        if neighbor_data:
            avg_income = np.mean(neighbor_data)
            expected_income = self.reference_point
            
            # 使用个体社会学习权重（个体异质性）
            w = self.social_learning_weight  # 范围[0.2, 0.6]
            
            # 计算邻居成功率（相对于参考点）
            neighbor_success_rate = sum(1 for inc in neighbor_data if inc > expected_income) / len(neighbor_data)
            
            # 客观概率
            obj_optimistic = 0.2
            obj_pessimistic = 0.2
            
            # 贝叶斯更新：主观概率 = (1-w)*客观 + w*邻居信号
            if neighbor_success_rate > 0.6:
                # 邻居表现好 -> 乐观概率提高
                adjustment = w * 0.2 * (neighbor_success_rate - 0.5)
                optimistic_prob += adjustment
                pessimistic_prob -= adjustment * 0.5
            elif neighbor_success_rate < 0.4:
                # 邻居表现差 -> 悲观概率提高
                adjustment = w * 0.2 * (0.5 - neighbor_success_rate)
                pessimistic_prob += adjustment
                optimistic_prob -= adjustment * 0.5
        
        # 确保概率归一化
        total = pessimistic_prob + baseline_prob + optimistic_prob
        pessimistic_prob /= total
        baseline_prob /= total
        optimistic_prob /= total
        
        # @fix 2025-12-01: 应用波动因子
        adjusted_price_std = price_std * price_volatility_factor
        
        # 生成情景（纳入劳动力成本）
        scenarios = []

        # 悲观情景
        pessimistic_price = max(0.1, price_mean - 1.5 * adjusted_price_std)
        pessimistic_yield = base_yield * pessimistic_yield_factor
        pessimistic_revenue = pessimistic_yield * pessimistic_price
        pessimistic_material_cost = pessimistic_revenue * material_cost_ratio
        pessimistic_labor_cost = labor_cost_per_mu * operated_area
        pessimistic_total_cost = pessimistic_material_cost + pessimistic_labor_cost
        pessimistic_subsidy = grain_subsidy_amount if crop_type == 'grain' else 0
        scenarios.append({
            'name': '悲观情景',
            'probability': pessimistic_prob,
            'profit': pessimistic_revenue - pessimistic_total_cost + pessimistic_subsidy + grain_security_premium
        })
        
        # 基准情景
        baseline_yield = base_yield
        baseline_revenue = baseline_yield * price_mean
        baseline_material_cost = baseline_revenue * material_cost_ratio
        baseline_labor_cost = labor_cost_per_mu * operated_area
        baseline_total_cost = baseline_material_cost + baseline_labor_cost
        baseline_subsidy = grain_subsidy_amount if crop_type == 'grain' else 0
        scenarios.append({
            'name': '基准情景',
            'probability': baseline_prob,
            'profit': baseline_revenue - baseline_total_cost + baseline_subsidy + grain_security_premium
        })
        
        # 乐观情景
        optimistic_price = price_mean + 1.5 * adjusted_price_std
        optimistic_yield = base_yield * optimistic_yield_factor
        optimistic_revenue = optimistic_yield * optimistic_price
        optimistic_material_cost = optimistic_revenue * material_cost_ratio
        optimistic_labor_cost = labor_cost_per_mu * operated_area
        optimistic_total_cost = optimistic_material_cost + optimistic_labor_cost
        optimistic_subsidy = grain_subsidy_amount if crop_type == 'grain' else 0
        scenarios.append({
            'name': '乐观情景',
            'probability': optimistic_prob,
            'profit': optimistic_revenue - optimistic_total_cost + optimistic_subsidy + grain_security_premium
        })
        
        return scenarios

    def update_reference_point(self):
        """
        @theory
        动态参考点更新机制 - 增强版
        基于前景理论的参考点依赖性，动态调整农户的决策基准
        新增：整合个人历史、预期趋势和社会比较
        """
        if not hasattr(self, 'income_history') or len(self.income_history) == 0:
            if self.income_memory:
                self.reference_point = self.income_memory[-1]
            return
            
        # 1. 个人历史参考点：基于过去收入的线性趋势预测
        if len(self.income_history) >= 3:
            # 使用简单线性回归预测下一期收入
            x = np.array(range(len(self.income_history)))
            y = np.array(self.income_history)
            slope, intercept = np.polyfit(x, y, 1)
            personal_expectation = intercept + slope * len(self.income_history)
            
            # 如果是下降趋势，则适当调低预期（损失厌恶）
            if slope < 0:
                personal_expectation = max(self.income_history[-1] * 0.9, personal_expectation)
        else:
            # 数据不足，使用最近收入
            personal_expectation = self.income_history[-1]
        
        # 2. 社会参考点：基于社会网络
        social_incomes = []
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor:
                social_incomes.append(
                    neighbor._get_income_signal()
                    if hasattr(neighbor, '_get_income_signal')
                    else getattr(neighbor, 'net_income', getattr(neighbor, 'total_income', 0))
                )
        
        social_expectation = np.mean(social_incomes) if social_incomes else personal_expectation
        
        # 3. 融合个人和社会参考点
        # internal_orientation决定了农户是"向内看"还是"向外看"
        self.reference_point = (self.internal_orientation * personal_expectation + 
                              (1 - self.internal_orientation) * social_expectation)
        
        # 确保参考点不会太低（心理安全网）
        # @fix: 从配置读取安全网系数
        safety_net_factor = getattr(self, 'safety_net_factor', 0.9)
        min_income = min(self.income_history) if self.income_history else self.reference_point * (1 - safety_net_factor)
        self.reference_point = max(self.reference_point, min_income * safety_net_factor)
        
        # 记录参考点历史
        self.reference_point_history.append(self.reference_point)
    
    def update_price_history(self, grain_price, cash_crop_price):
        """
        @theory
        价格历史更新 - 解决内生化风险感知问题
        农户基于观察到的价格历史形成对未来风险的判断
        """
        # 更新价格历史
        self.price_history['grain'].append(grain_price)
        self.price_history['cash_crop'].append(cash_crop_price)
        
        # 保持记忆长度限制
        if len(self.price_history['grain']) > self.memory_length:
            self.price_history['grain'] = self.price_history['grain'][-self.memory_length:]
        if len(self.price_history['cash_crop']) > self.memory_length:
            self.price_history['cash_crop'] = self.price_history['cash_crop'][-self.memory_length:]
    
    def bayesian_learning_update(self):
        """
        @theory
        基于贝叶斯更新的学习机制
        将观察到的证据（自身和邻居的收益）转化为对不同作物收益的信念更新
        """
        # 保存上一期的信念状态
        if self.last_beliefs is None:
            self.last_beliefs = {
                'grain': self.crop_belief['grain'],
                'cash_crop': self.crop_belief['cash_crop']
            }
        else:
            self.last_beliefs = {
                'grain': self.crop_belief['grain'],
                'cash_crop': self.crop_belief['cash_crop']
            }
        
        # 收集证据：自身和邻居的收入数据
        grain_evidence = []
        cash_crop_evidence = []
        
        # 自己的数据
        if hasattr(self, 'crop_choice_history') and self.crop_choice_history:
            last_choice = self.crop_choice_history[-1]
            if 'choice' in last_choice:
                income_signal = self._get_income_signal()
                if last_choice['choice'] == CropType.GRAIN:
                    grain_evidence.append((income_signal, 1.0))  # (收入, 权重)
                else:
                    cash_crop_evidence.append((income_signal, 1.0))
        
        # 邻居的数据（通过社会学习获得）
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor and hasattr(neighbor, 'crop_choice_history') and neighbor.crop_choice_history:
                last_choice = neighbor.crop_choice_history[-1]
                if 'choice' in last_choice:
                    neighbor_income_signal = (
                        neighbor._get_income_signal()
                        if hasattr(neighbor, '_get_income_signal')
                        else getattr(neighbor, 'net_income', getattr(neighbor, 'total_income', 0))
                    )
                    # 网络强度作为权重
                    weight = self.network_strength.get(neighbor_id, 0.5)
                    if last_choice['choice'] == CropType.GRAIN:
                        grain_evidence.append((neighbor_income_signal, weight))
                    else:
                        cash_crop_evidence.append((neighbor_income_signal, weight))
        
        # 贝叶斯更新
        # 有利证据：收入高于期望
        # 不利证据：收入低于期望
        expected_income = self.reference_point * 1.1  # 期望收入略高于参考点
        
        # 提取当前信念参数
        alpha_grain, beta_grain = self.crop_belief['grain']
        alpha_cash, beta_cash = self.crop_belief['cash_crop']
        
        # 更新粮食信念
        for income, weight in grain_evidence:
            if income >= expected_income:
                alpha_grain += weight  # 增加有利证据
            else:
                beta_grain += weight   # 增加不利证据
        
        # 更新经济作物信念
        for income, weight in cash_crop_evidence:
            if income >= expected_income:
                alpha_cash += weight   # 增加有利证据
            else:
                beta_cash += weight    # 增加不利证据
        
        # 衰减历史信息（避免累积太多历史信息导致新证据难以改变信念）
        # @fix: 从配置读取衰减率
        bayesian_config = self.agent_config.get('agents', {}).get('farmer', {}).get('bayesian_learning', {})
        decay_rate = bayesian_config.get('decay_rate', 0.9)
        prior_alpha = bayesian_config.get('prior_alpha', 5.0)
        prior_beta = bayesian_config.get('prior_beta', 5.0)
        alpha_grain = prior_alpha + (alpha_grain - prior_alpha) * decay_rate
        beta_grain = prior_beta + (beta_grain - prior_beta) * decay_rate
        alpha_cash = prior_alpha + (alpha_cash - prior_alpha) * decay_rate
        beta_cash = prior_beta + (beta_cash - prior_beta) * decay_rate
        
        # 更新信念
        self.crop_belief['grain'] = (alpha_grain, beta_grain)
        self.crop_belief['cash_crop'] = (alpha_cash, beta_cash)
        
        # 根据后验信念计算每种作物的预期收益率
        # E[X] = α/(α+β) for Beta(α,β)
        grain_expectation = alpha_grain / (alpha_grain + beta_grain)
        cash_crop_expectation = alpha_cash / (alpha_cash + beta_cash)
        
        return grain_expectation, cash_crop_expectation
    
    def generate_crop_scenarios(self, crop_type, base_price, subsidy_rate):
        """
        @theory
        生成作物收益情景分析
        基于价格历史和农户认知产生多种可能的收益结果
        替代固定方差的"黑箱"机制
        增强：从环境中获取价格统计数据
        @fix 2025-12-23: 从配置文件读取基础产量，保持一致性
        """
        scenarios = []
        
        # @fix: 风险感知内生化，基于环境历史价格
        price_stats = self._get_output_price_statistics_per_kg(
            'grain' if crop_type == 'grain' else 'cash_crop'
        )
        price_mean = price_stats['mean']
        price_std = price_stats['std']
        operated_area = max(self.get_operated_land_area(), 1.0)
        
        # 从配置读取基础产量
        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        
        if crop_type == 'grain':
            yield_per_mu = base_yields.get('grain', 420)  # 默认420kg/亩
            base_yield = yield_per_mu * self.farming_ability * operated_area
            cost_ratio = 0.6
        else:
            yield_per_mu = base_yields.get('cash_crop', 1000)  # 默认1000kg/亩
            base_yield = yield_per_mu * self.farming_ability * operated_area
            cost_ratio = 0.4  # @更激进修复：从0.5降到0.4
        
        # 生成三种主要情景
        scenarios.append({
            'name': '悲观情景',
            'probability': 0.2,
            'price': max(0.1, price_mean - 1.5 * price_std),
            'yield_factor': 0.8
        })
        
        scenarios.append({
            'name': '基准情景',
            'probability': 0.6,
            'price': price_mean,
            'yield_factor': 1.0
        })
        
        scenarios.append({
            'name': '乐观情景',
            'probability': 0.2,
            'price': price_mean + 1.5 * price_std,
            'yield_factor': 1.2
        })
        
        # 计算每种情景的利润
        for scenario in scenarios:
            actual_yield = base_yield * scenario['yield_factor']
            revenue = actual_yield * scenario['price']
            
            # @fix: 采用正确的补贴决策逻辑: 利润 = (市场收入 - 成本) + 补贴
            cost = revenue * cost_ratio
            subsidy = 0
            if crop_type == 'grain':
                base_subsidy_per_mu = 0.0
                if hasattr(self.model, 'environment') and self.model.environment.active_policy:
                    base_subsidy_per_mu = self.model.environment.active_policy.get('base_subsidy_per_mu', 0.0) or 0.0
                elif hasattr(self.model, 'config'):
                    base_subsidy_per_mu = (
                        self.model.config.get('policy', {})
                        .get('government', {})
                        .get('base_subsidy_per_mu', 0.0)
                    ) or 0.0
                subsidy = operated_area * float(base_subsidy_per_mu) * subsidy_rate

            scenario['profit'] = (revenue - cost) + subsidy
        
        return scenarios

    def expected_fixed_rule_account(self, action: str, area: float | None = None) -> ActionIncomeAccount:
        terminal = action == "exit"
        area = 0.0 if terminal else max(0.0, self.get_operated_land_area() if area is None else area)
        crop = CropType.CASH_CROP if action == "expand_cash" else CropType.GRAIN
        share = getattr(self, "household_candidate_grain_shares", {}).get(action)
        if share is not None and 0 < share < 1:
            crop = CropType.MIXED
        preview = copy(self)
        preview._fixed_rule_preview_area = area
        preview._get_average_soil_quality = self._get_average_soil_quality
        preview._get_technology_level = self._get_technology_level
        preview.planned_crop = None if terminal else crop
        preview.household_planned_grain_share = share
        preview.fixed_rule_labor_settled = False
        preview.hired_labor_cost = 0.0
        preview.labor_reduction_ratio = 0.0
        preview.service_contracts = []
        preview.service_yield_bonus_multiplier = 1.0
        grain_share = float(crop == CropType.GRAIN) if share is None else share
        preview.grain_area = area * grain_share
        service_fee = service_area = 0.0
        demand = self.model._get_farmer_service_demand(preview) if area > 0 else None
        if demand is not None and self.model.service_market is not None:
            providers = [p for p in self.model.service_providers if p.can_serve(self.pos, demand["area"])]
            quotes = [p.calculate_service_fee(demand["area"], demand["service_type"],
                      market_price_multiplier=self.model.service_market._current_price_multiplier) for p in providers]
            affordable = [quote for quote in quotes if quote <= demand["max_price_per_mu"]]
            if affordable:
                service_area = float(demand["area"])
                service_fee = min(affordable) * service_area
                service_cfg = self.model.service_market._get_service_type_config(demand["service_type"])
                coverage = float(service_cfg.get("coverage_ratio", 0.5))
                preview.labor_reduction_ratio = coverage * 0.85 * service_area / area
        cost = preview._calculate_detailed_cost(crop, area)
        family_days = float(preview.family_labor_used)
        preview.fixed_rule_family_labor_days = family_days
        free_days = max(0.0, self.total_labor_units - family_days)
        off_farm_days = preview._get_max_off_farm_days(free_days)
        off_farm = preview._calculate_off_farm_income_from_labor_days(free_days)
        if crop == CropType.MIXED:
            grain_output = grain_share * preview._calculate_deterministic_expected_yield(CropType.GRAIN, area)
            cash_output = (1 - grain_share) * preview._calculate_deterministic_expected_yield(CropType.CASH_CROP, area)
            output = grain_output + cash_output
            sales = grain_output * self._get_output_price_per_kg("crop_price") + cash_output * self._get_output_price_per_kg("cash_crop")
        else:
            output = preview._calculate_deterministic_expected_yield(crop, area)
            grain_output = output if crop == CropType.GRAIN else 0.0
            cash_output = output - grain_output
            sales = output * self._get_output_price_per_kg("cash_crop" if crop == CropType.CASH_CROP else "crop_price")
        if self._national_v1:
            sales = self._national_crop_sales(grain_output, cash_output)['sales_yuan']
        rent_out, rent_in = expected_land_cash(self, area, terminal)
        available_labor = getattr(self, "fixed_rule_hired_labor_available_days", None)
        if available_labor is None:
            available_labor = sum(max(0.0, f.off_farm_labor) for f in self.model.farmers if f is not self)
        conversion = float(getattr(self, "fixed_rule_conversion_cost_yuan", 0.0)) if action != getattr(self, "fixed_rule_previous_action", "maintain_grain") else 0.0
        risk = getattr(self, "fixed_rule_risk_penalty_yuan", {})
        if share is not None and self.household_previous_grain_share is not None:
            conversion = float(getattr(self, "fixed_rule_conversion_cost_yuan", 0.0)) if share != self.household_previous_grain_share else 0.0
        grain_service_fee = service_fee
        if crop == CropType.MIXED:
            # With unknown plot crops, only whole-farm coverage identifies the grain service.
            grain_service_fee = service_fee * grain_share if service_area >= area - 1e-8 else 0.0
        return ActionIncomeAccount(
            action=action, crop="none" if terminal else crop.value, area_mu=area, output_kg=output,
            sales_yuan=sales,
            support_yuan=expected_support(self, preview.grain_area, "farmer",
                                         service_cost_yuan=grain_service_fee,
                                         service_area_mu=service_area),
            rent_out_yuan=rent_out, off_farm_yuan=off_farm,
            common_income_yuan=float(getattr(self, "fixed_rule_common_income_yuan", 0.0)),
            production_cost_yuan=cost["cash_total_cost"], hired_labor_cost_yuan=cost["hired_labor_cost"],
            service_cost_yuan=service_fee, rent_in_yuan=rent_in, management_cost_yuan=0.0,
            conversion_cost_yuan=conversion,
            risk_penalty_yuan=(grain_share * risk.get("maintain_grain", 0.0) + (1-grain_share) * risk.get("expand_cash", 0.0)
                               if crop == CropType.MIXED else float(risk.get(action, 0.0))),
            family_labor_days=family_days, hired_labor_days=preview.requested_hired_labor_days,
            off_farm_days=off_farm_days, service_area_mu=service_area,
            working_capital_available=self.balance_sheet_snapshot()["working_capital"],
            hired_labor_available_days=float(available_labor),
            grain_area_mu=preview.grain_area, grain_output_kg=grain_output, cash_output_kg=cash_output,
        )

    def _calculate_deterministic_expected_yield(self, crop_type, area):
        """Mirror the structural yield function without climate/random shocks."""
        area = max(float(area or 0.0), 0.0)
        if area <= 0:
            return 0.0

        crop_name = str(getattr(crop_type, 'value', crop_type)).lower()
        is_grain = crop_type == CropType.GRAIN or crop_name == 'grain'

        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        yield_factors = production_config.get('yield_factors', {})
        tech_progress = production_config.get('technology_progress', {})

        if is_grain:
            base_yield_per_mu = base_yields.get('grain', 420)
            if self._national_v1:
                base_yield_per_mu = self._v1_grain_base_yield(base_yield_per_mu)
        else:
            base_yield_per_mu = base_yields.get('cash_crop', 1000)

        deterministic_yield = base_yield_per_mu * area

        annual_growth = resolve_crop_annual_yield_growth(tech_progress, crop_type)
        current_step = getattr(self.model, 'step_count', 0)
        deterministic_yield *= (1 + annual_growth) ** current_step

        scale_config = tech_progress.get('scale_efficiency_bonus', {})
        threshold = scale_config.get('threshold_area', 30)
        if area >= 100:
            deterministic_yield *= scale_config.get('large_scale', 1.20)
        elif area >= threshold:
            deterministic_yield *= scale_config.get('medium_scale', 1.10)
        else:
            deterministic_yield *= scale_config.get('small_scale', 1.0)

        avg_soil_quality = self._get_average_soil_quality()
        soil_config = yield_factors.get('soil_quality', {})
        min_effect = soil_config.get('min_effect', 0.5)
        max_effect = soil_config.get('max_effect', 1.0)
        exponent = soil_config.get('exponent', 0.8)
        soil_factor = min_effect + (max_effect - min_effect) * (avg_soil_quality ** exponent)
        deterministic_yield *= soil_factor

        ability_config = yield_factors.get('farming_ability', {})
        min_mult = ability_config.get('min_multiplier', 0.8)
        max_mult = ability_config.get('max_multiplier', 1.2)
        ability_multiplier = min_mult + (max_mult - min_mult) * ((self.farming_ability - 0.8) / 0.4)
        if self._national_v1:
            ability_multiplier = 1.0
        deterministic_yield *= ability_multiplier

        tech_config = yield_factors.get('technology', {})
        base_tech_level = tech_config.get('base_level', 1.0)
        max_improvement = tech_config.get('max_improvement', 0.2)
        tech_level = self._get_technology_level()
        deterministic_yield *= 1.0 + max_improvement * (tech_level - base_tech_level)

        service_contracts = getattr(self, 'service_contracts', []) or []
        service_signal = max(0.0, min(1.0, float(getattr(self, 'labor_reduction_ratio', 0.0) or 0.0)))
        if isinstance(service_contracts, list) and service_contracts:
            service_signal = max(service_signal, min(1.0, 0.20 + 0.18 * len(service_contracts)))

        mechanization_potential = 0.7
        if hasattr(self, '_get_regional_land_transfer_params'):
            try:
                regional_params = self._get_regional_land_transfer_params()
                mechanization_potential = float(regional_params.get('mechanization_potential', 0.7) or 0.7)
            except Exception:
                mechanization_potential = 0.7
        mechanization_signal = max(0.0, min(1.0, mechanization_potential))

        retention_cfg = (
            self.model.config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        professional_threshold = float(retention_cfg.get('min_operated_area', 30) or 30)
        scale_progress = max(0.0, min(1.0, (area - 10.0) / max(20.0, professional_threshold * 2.0)))
        strategy_name = str(
            getattr(getattr(self, 'livelihood_strategy', None), 'value', getattr(self, 'livelihood_strategy', ''))
        ).lower()
        professional_signal = 0.0
        if area >= professional_threshold:
            professional_signal += 0.55
        if not self._national_v1 and 'pure' in strategy_name:
            professional_signal += 0.25
        if float(getattr(self, 'farming_ability', 1.0) or 1.0) >= 1.05:
            professional_signal += 0.20
        professional_signal = max(0.0, min(1.0, professional_signal))
        structural_cfg = yield_factors.get('structural_efficiency', {}) or {}

        if is_grain:
            deterministic_yield *= (
                1.0
                + float(structural_cfg.get('farmer_grain_scale_mechanization_weight', 0.08) or 0.0)
                * scale_progress
                * mechanization_potential
                + float(structural_cfg.get('farmer_grain_service_weight', 0.06) or 0.0)
                * service_signal
                + float(structural_cfg.get('farmer_grain_service_mechanization_weight', 0.05) or 0.0)
                * service_signal
                * mechanization_potential
                + float(structural_cfg.get('farmer_grain_professional_weight', 0.04) or 0.0)
                * professional_signal
            )
        else:
            deterministic_yield *= (
                1.0
                + float(structural_cfg.get('farmer_cash_scale_weight', 0.03) or 0.0)
                * scale_progress
                + float(structural_cfg.get('farmer_cash_service_weight', 0.02) or 0.0)
                * service_signal
                + float(structural_cfg.get('farmer_cash_professional_weight', 0.02) or 0.0)
                * professional_signal
            )

        service_yield_bonus_multiplier = max(
            1.0,
            min(1.18, float(getattr(self, 'service_yield_bonus_multiplier', 1.0) or 1.0)),
        )
        if is_grain:
            deterministic_yield *= service_yield_bonus_multiplier
        else:
            deterministic_yield *= 1.0 + 0.5 * (service_yield_bonus_multiplier - 1.0)

        return max(0.0, deterministic_yield)

    def _get_grain_security_decision_premium(self, operated_area):
        env = getattr(self.model, 'environment', None)
        active_policy = getattr(env, 'active_policy', {}) if env is not None else {}
        if not isinstance(active_policy, dict) or active_policy.get('disable_grain_intervention'):
            return 0.0

        grain_req_config = (
            getattr(self.model, 'config', {})
            .get('agents', {})
            .get('village_collective', {})
            .get('grain_requirement', {})
        )
        if not grain_req_config.get('enabled', True):
            return 0.0

        base_premium = max(
            0.0,
            float(grain_req_config.get('farmer_security_premium_per_mu', 0.0) or 0.0),
        )
        if base_premium <= 0:
            return 0.0

        history = getattr(env, 'grain_market_history', None) or []
        recent_supply = 0.0
        if history:
            try:
                recent_supply = float(history[-1].get('grain_supply', 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                recent_supply = 0.0

        targets = (
            getattr(self.model, 'config', {})
            .get('policy', {})
            .get('government_objectives', {})
            .get('targets', {})
        )
        try:
            target_output = float(targets.get('grain_output', 0.0) or 0.0)
        except (TypeError, ValueError):
            target_output = 0.0
        gap_ratio = max(0.0, (target_output - recent_supply) / target_output) if target_output > 0 else 0.0
        gap_multiplier = max(
            0.0,
            float(grain_req_config.get('farmer_security_gap_multiplier', 1.0) or 1.0),
        )
        return max(0.0, float(operated_area or 0.0)) * base_premium * (1.0 + gap_multiplier * gap_ratio)

    def calculate_grain_expected_profit(self, grain_price_floor, grain_subsidies_rate, operated_area,
                                       *, expected_subsidy_income: float | None = None):
        """
        @theory
        计算粮食作物的预期利润
        @fix 校准：基于《全国农产品成本收益资料汇编2023》
        - 产量: 从配置读取（默认750kg/亩）
        - 成本率: 87% (接近实证92%: 1252/1362)
        - 2022年实证: 总产值1442元, 成本1252元, 净利润189元
        @fix 2025-12-23: 从配置文件读取基础产量
        """
        # 基础产量（考虑耕作能力）
        operated_area = max(operated_area, 0.0)
        if operated_area == 0:
            return 0.0
        
        expected_output_kg = self._calculate_deterministic_expected_yield(
            CropType.GRAIN,
            operated_area,
        )

        # 环境中的作物价格以元/斤存储，收益结算统一换算为元/kg。
        grain_price_per_kg = max(0.0, float(grain_price_floor or 0.0)) * JIN_PER_KG
        expected_revenue = expected_output_kg * grain_price_per_kg
        
        # 成本: 87%成本率 (接近实证92%)
        # 实证: 2022年粮食总成本1252元(不含土地), 总产值1442元, 成本率1252/1442=87%
        cost_grain = expected_revenue * 0.87
        
        subsidy_income = 0.0
        env = getattr(self.model, 'environment', None)
        if expected_subsidy_income is not None:
            subsidy_income = expected_subsidy_income
        elif env is not None and hasattr(env, 'get_subsidy_for_farmer'):
            subsidy_income = max(0.0, float(env.get_subsidy_for_farmer(self, operated_area) or 0.0))
        else:
            base_subsidy_per_mu = 0.0
            active_policy = getattr(env, 'active_policy', {}) if env is not None else {}
            if active_policy:
                base_subsidy_per_mu = float(active_policy.get('base_subsidy_per_mu', 0.0) or 0.0)
            subsidy_income = max(0.0, float(grain_subsidies_rate or 0.0)) * base_subsidy_per_mu * operated_area

        security_premium = self._get_grain_security_decision_premium(operated_area)

        # 净利润（显式计入政策收益与粮食安全导向的预期稳定性溢价）
        net_profit = expected_revenue + subsidy_income + security_premium - cost_grain
        
        return net_profit

    def calculate_cash_crop_expected_profit(self, market_price_cash_crop, operated_area):
        """
        @theory
        计算经济作物的预期利润
        @fix 校准: 与企业保持一致
        @fix 2025-12-23: 从配置文件读取基础产量
        @fix 2026-06-07: 与作物选择情景口径一致，显式计入劳动密集型成本
        """
        # 基础产量（考虑耕作能力）
        operated_area = max(operated_area, 0.0)
        if operated_area == 0:
            return 0.0
        
        base_yield_kg = self._calculate_deterministic_expected_yield(
            CropType.CASH_CROP,
            operated_area,
        )

        cash_crop_price_per_kg = max(0.0, float(market_price_cash_crop or 0.0)) * JIN_PER_KG
        expected_revenue = base_yield_kg * cash_crop_price_per_kg

        # 与 _generate_informed_scenarios 保持一致：
        # 经济作物为劳动密集型，不能只用收入比例成本率，否则会系统性高估经作利润。
        material_cost_ratio = 0.45
        labor_config = getattr(self.model, 'config', {}).get('labor_market', {}).get('labor_requirements', {})
        labor_days_per_mu = float(labor_config.get('cash_crop', {}).get('base', 30) or 30)
        daily_wage = self._get_current_non_farm_daily_wage()
        material_cost = expected_revenue * material_cost_ratio
        labor_cost = labor_days_per_mu * daily_wage * operated_area

        return expected_revenue - material_cost - labor_cost

    def _get_base_non_farm_wage(self):
        canonical_fallback = resolve_base_non_farm_wage(
            getattr(self.model, 'config', {}),
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_base_non_farm_wage_rate'):
            wage = env.get_base_non_farm_wage_rate()
            if wage is not None:
                return self._normalize_non_farm_daily_wage(wage, fallback=canonical_fallback)
        return self._normalize_non_farm_daily_wage(canonical_fallback, fallback=canonical_fallback)

    def _normalize_non_farm_daily_wage(self, wage_value, fallback=None):
        canonical_fallback = resolve_base_non_farm_wage(
            getattr(self.model, 'config', {}),
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        if fallback is None:
            fallback = canonical_fallback
        try:
            candidate = float(wage_value)
        except (TypeError, ValueError):
            candidate = float(fallback)

        if candidate <= 0:
            candidate = float(fallback)

        if candidate > 1000:
            candidate = candidate / 260.0

        labor_cfg = getattr(self.model, 'config', {}).get('labor_market', {}) or {}
        floor = float(labor_cfg.get('wage_floor', 80.0) or 80.0)
        ceiling = float(labor_cfg.get('wage_ceiling', max(300.0, floor)) or max(300.0, floor))
        if ceiling < floor:
            ceiling = floor
        return float(min(ceiling, max(floor, candidate)))

    def _get_current_non_farm_daily_wage(self):
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_market_price'):
            wage = env.get_market_price('wage_rate')
            if wage is not None:
                return self._normalize_non_farm_daily_wage(wage)
        return self._get_base_non_farm_wage()

    def _get_effective_non_farm_daily_wage(self):
        daily_wage = self._get_current_non_farm_daily_wage()
        effective_wage = daily_wage
        if self.productivity_shock_enabled:
            strategy_modifier = 1.0
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_modifier = max(0.2, 1 - self.productivity_adoption_lag)
            wage_factor = max(0.2, 1 - self.productivity_wage_penalty * strategy_modifier)
            effective_wage *= wage_factor
        return max(0.0, float(effective_wage))

    def _get_max_off_farm_days(self, candidate_labor_days):
        effective_labor = max(0.0, float(candidate_labor_days or 0.0))
        if self.productivity_shock_enabled:
            strategy_modifier = 1.0
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_modifier = max(0.2, 1 - self.productivity_adoption_lag)
            labor_factor = max(0.0, 1 - self.productivity_non_farm_job_shock * strategy_modifier)
            effective_labor *= labor_factor

        if action_income_enabled(self.model.config):
            caps = {LivelihoodStrategy.PURE_FARMER: 30.0, LivelihoodStrategy.PLURIACTIVE: 100.0, LivelihoodStrategy.NON_FARMER: 220.0}
            return min(effective_labor, caps[self.livelihood_strategy])

        self_operated_area = self.get_operated_land_area()
        agri_busy_days = self_operated_area * 15
        max_off_farm_days_by_strategy = {
            LivelihoodStrategy.PURE_FARMER: 30,
            LivelihoodStrategy.PLURIACTIVE: 100,
            LivelihoodStrategy.NON_FARMER: 220,
        }
        absolute_max = max_off_farm_days_by_strategy.get(self.livelihood_strategy, 100)

        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            return min(effective_labor, max(0.0, self.total_labor_units * 0.4 - agri_busy_days), absolute_max)
        if self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            actual_available_days = max(0.0, self.total_labor_units - agri_busy_days)
            return min(effective_labor, actual_available_days * 0.85, absolute_max)
        return min(effective_labor, absolute_max)

    def _calculate_off_farm_income_from_labor_days(self, candidate_labor_days, apply_minimum_guarantee=False):
        actual_off_farm_labor = self._get_max_off_farm_days(candidate_labor_days)
        effective_wage = self._get_effective_non_farm_daily_wage()
        off_farm_income = actual_off_farm_labor * self.off_farm_wage_level * effective_wage

        if apply_minimum_guarantee and self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            if off_farm_income <= 0 and candidate_labor_days > 0:
                daily_wage = self._get_current_non_farm_daily_wage()
                off_farm_income = max(off_farm_income, daily_wage * 0.02 * max(0.0, float(candidate_labor_days)))
            elif off_farm_income <= 0:
                min_work_days = 5
                daily_wage = self._get_current_non_farm_daily_wage()
                off_farm_income = min_work_days * self.off_farm_wage_level * daily_wage

        return max(0.0, float(off_farm_income))

    def _estimate_incremental_off_farm_income(self, released_labor_days):
        released_labor_days = max(0.0, float(released_labor_days or 0.0))
        if released_labor_days <= 0:
            return 0.0
        current_labor_days = max(0.0, float(getattr(self, 'off_farm_labor', 0.0) or 0.0))
        baseline_income = self._calculate_off_farm_income_from_labor_days(current_labor_days)
        expanded_income = self._calculate_off_farm_income_from_labor_days(current_labor_days + released_labor_days)
        return max(0.0, expanded_income - baseline_income)

    def _get_recent_net_income_signal(self):
        for attr_name in ('cash_net_income', 'net_income'):
            try:
                value = float(getattr(self, attr_name, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if abs(value) > 1e-6:
                return value

        income_history = getattr(self, 'income_history', None) or []
        if income_history:
            try:
                return float(income_history[-1] or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _get_expansion_area_cap(self, current_area):
        current_area = max(0.0, float(current_area or 0.0))
        recent_net_income = self._get_recent_net_income_signal()
        service_signal = self._get_service_readiness_signal()
        professional_signal = self._get_professional_operator_signal(current_area)
        strong_absorber = professional_signal >= 0.55 or service_signal >= 0.65
        if recent_net_income <= 0:
            if strong_absorber and recent_net_income > -5000:
                return max(1.5, min(5.0, current_area * 0.12 + 1.0))
            return 0.0
        if recent_net_income < 8000:
            upper_cap = 6.0 if strong_absorber else 4.0
            return max(1.5, min(upper_cap, current_area * 0.12 + 0.8))
        if recent_net_income < 20000:
            upper_cap = 12.0 if strong_absorber else 8.0
            return max(3.0, min(upper_cap, current_area * 0.22 + 1.5))
        upper_cap = 18.0 if strong_absorber else 15.0
        return max(6.0, min(upper_cap, current_area * 0.34 + 2.5))

    def _get_service_readiness_signal(self):
        service_contracts = getattr(self, 'service_contracts', []) or []
        service_signal = max(0.0, min(1.0, float(getattr(self, 'labor_reduction_ratio', 0.0) or 0.0)))
        if isinstance(service_contracts, list) and service_contracts:
            service_signal = max(service_signal, min(1.0, 0.20 + 0.18 * len(service_contracts)))
        if bool(getattr(self, 'prefers_cooperative_custody', False)):
            service_signal = max(service_signal, 0.75)
        if bool(getattr(self, 'is_weekend_farmer', False)):
            service_signal = max(service_signal, 0.65)
        return service_signal

    def _get_professional_operator_signal(self, current_area=None):
        current_area = self.get_operated_land_area() if current_area is None else max(0.0, float(current_area or 0.0))
        retention_cfg = (
            self.model.config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        professional_threshold = float(retention_cfg.get('min_operated_area', 30.0) or 30.0)
        scale_signal = max(
            0.0,
            min(1.0, (current_area - 10.0) / max(20.0, professional_threshold * 1.6)),
        )
        agri_ratio = self._get_livelihood_agricultural_income_ratio()
        ability_signal = max(0.0, min(1.0, (float(getattr(self, 'farming_ability', 1.0) or 1.0) - 0.8) / 0.4))
        return max(0.0, min(1.0, 0.45 * scale_signal + 0.35 * agri_ratio + 0.20 * ability_signal))

    def get_pending_transfer_area(self):
        available_owned_patches = [
            patch for patch in self.land_patches
            if patch.owner == self and patch.leased_to is None
        ]
        available_owned_area = float(sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in available_owned_patches))
        if available_owned_area <= 0:
            return 0.0

        explicit_pending = max(0.0, float(getattr(self, 'land_pending_transfer', 0.0) or 0.0))
        partial_ratio = getattr(self, 'partial_succession_ratio', None)
        if partial_ratio is not None:
            try:
                partial_ratio = max(0.0, min(1.0, float(partial_ratio or 0.0)))
            except (TypeError, ValueError):
                partial_ratio = None

        if partial_ratio is not None:
            total_owned_area = float(
                sum(
                    float(getattr(patch, 'area', 0.0) or 0.0)
                    for patch in self.land_patches
                    if patch.owner == self
                )
            )
            target_self_cultivation_area = total_owned_area * partial_ratio
            explicit_pending = max(
                explicit_pending,
                max(0.0, available_owned_area - target_self_cultivation_area),
            )

        return max(0.0, min(available_owned_area, explicit_pending))

    def _submit_pending_transfer_supply(self):
        pending_area = self.get_pending_transfer_area()
        if pending_area <= 0:
            return
        if not (
            bool(getattr(self, 'prefers_cooperative_custody', False))
            or bool(getattr(self, 'is_weekend_farmer', False))
            or float(getattr(self, 'land_transfer_willingness', 0.0) or 0.0) >= 0.60
        ):
            return

        candidate_patches = [
            patch for patch in self.land_patches
            if patch.owner == self
            and patch.leased_to is None
            and patch not in self.reservation_prices
        ]
        if not candidate_patches:
            return

        def supply_priority(patch):
            patch_area = max(1.0e-6, float(getattr(patch, 'area', 0.0) or 0.0))
            try:
                profit_per_mu = float(self.calculate_self_cultivation_profit(patch) or 0.0) / patch_area
            except Exception:
                profit_per_mu = 0.0
            return (profit_per_mu, -patch_area)

        released_area = 0.0
        rent_discount = 0.82 if bool(getattr(self, 'prefers_cooperative_custody', False)) else 0.88
        for patch in sorted(candidate_patches, key=supply_priority):
            if released_area >= pending_area:
                break
            if self.act_as_land_supplier_for_patch(patch, rent_discount=rent_discount):
                released_area += float(getattr(patch, 'area', 0.0) or 0.0)

    def _is_profitable_rent_in_opportunity(self, patch):
        patch_area = max(0.0, float(getattr(patch, 'area', 0.0) or 0.0))
        if patch_area <= 0:
            return False

        self_cultivation_profit = self.calculate_self_cultivation_profit(patch)
        quality_multiplier = max(0.6, min(1.4, float(getattr(patch, 'soil_quality', 1.0) or 1.0)))
        expected_rent = patch_area * self._get_average_rental_price() * quality_multiplier
        post_rent_surplus = self_cultivation_profit - expected_rent
        recent_net_income = self._get_recent_net_income_signal()
        service_signal = self._get_service_readiness_signal()
        professional_signal = self._get_professional_operator_signal()
        strong_absorber = professional_signal >= 0.55 or service_signal >= 0.65

        if post_rent_surplus <= 0:
            return False
        if recent_net_income < 0:
            if strong_absorber:
                return post_rent_surplus >= max(40.0 * patch_area, 0.02 * abs(recent_net_income))
            return post_rent_surplus >= max(120.0 * patch_area, 0.05 * abs(recent_net_income))
        if recent_net_income < 8000:
            if strong_absorber:
                return post_rent_surplus >= 40.0 * patch_area
            return post_rent_surplus >= 60.0 * patch_area
        return True

    def _get_agriculture_linked_income(self):
        agricultural_income = float(getattr(self, 'agricultural_income', 0.0) or 0.0)
        subsidy_income = float(getattr(self, 'subsidy_income', 0.0) or 0.0)
        agricultural_wage_income = float(getattr(self, 'labor_sale_income', 0.0) or 0.0)
        return max(0.0, agricultural_income + subsidy_income + agricultural_wage_income)

    def _get_livelihood_agricultural_income_ratio(self):
        agriculture_linked_income = self._get_agriculture_linked_income()
        total_income = float(getattr(self, 'gross_income', getattr(self, 'total_income', 0.0)) or 0.0)
        if total_income <= 0:
            off_farm_income = max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
            property_income = max(0.0, float(getattr(self, 'land_rental_income', 0.0) or 0.0))
            transfer_income = max(0.0, float(getattr(self, 'transfer_income', 0.0) or 0.0))
            non_farm_business_income = max(
                0.0,
                float(getattr(self, 'non_farm_business_income', 0.0) or 0.0),
            )
            total_income = (
                agriculture_linked_income
                + off_farm_income
                + property_income
                + transfer_income
                + non_farm_business_income
            )
        if total_income > 0:
            return max(0.0, min(1.0, agriculture_linked_income / total_income))
        if agriculture_linked_income > 0:
            return 1.0
        non_ag_income = (
            max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
            + max(0.0, float(getattr(self, 'land_rental_income', 0.0) or 0.0))
            + max(0.0, float(getattr(self, 'transfer_income', 0.0) or 0.0))
            + max(0.0, float(getattr(self, 'non_farm_business_income', 0.0) or 0.0))
        )
        if non_ag_income > 0:
            return 0.0
        return 0.5

    def _get_off_farm_income_ratio(self):
        return max(0.0, min(1.0, 1.0 - self._get_livelihood_agricultural_income_ratio()))

    def _get_strategy_agricultural_dependence_anchor(self):
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            return float(np_uniform(self, 0.78, 0.92))
        if self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            return float(np_uniform(self, 0.38, 0.62))
        return float(np_uniform(self, 0.02, 0.18))

    def _sync_characteristics_income_fields(self, use_strategy_anchor=False):
        if not hasattr(self, 'characteristics') or self.characteristics is None:
            return

        agricultural_income = max(0.0, float(getattr(self, 'agricultural_income', 0.0) or 0.0))
        subsidy_income = max(0.0, float(getattr(self, 'subsidy_income', 0.0) or 0.0))
        agricultural_wage_income = max(0.0, float(getattr(self, 'labor_sale_income', 0.0) or 0.0))
        wage_income = max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
        property_income = max(0.0, float(getattr(self, 'land_rental_income', 0.0) or 0.0))

        farm_income = agricultural_income + subsidy_income + agricultural_wage_income
        transfer_income = 0.0
        non_farm_business_income = 0.0

        total_income = farm_income + wage_income + property_income + transfer_income + non_farm_business_income
        if use_strategy_anchor and total_income <= 0:
            anchor = self._get_strategy_agricultural_dependence_anchor()
            baseline_income = max(
                12000.0,
                float(getattr(self, 'initial_capital', 0.0) or 0.0) * 0.6
                + float(self.get_operated_land_area()) * 800.0,
            )
            farm_income = baseline_income * anchor
            non_ag_income = max(0.0, baseline_income - farm_income)
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                property_share = 0.20
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                property_share = 0.08
            else:
                property_share = 0.03
            property_income = non_ag_income * property_share
            wage_income = max(0.0, non_ag_income - property_income)
            total_income = farm_income + wage_income + property_income

        self.characteristics.farm_income = float(farm_income)
        self.characteristics.wage_income = float(wage_income)
        self.characteristics.property_income = float(property_income)
        self.characteristics.transfer_income = float(transfer_income)
        self.characteristics.non_farm_business_income = float(non_farm_business_income)

        if total_income > 0:
            self.characteristics.update_from_income()

    def _align_characteristics_to_livelihood_strategy(self, blend=0.65):
        if not hasattr(self, 'characteristics') or self.characteristics is None:
            return

        anchor = self._get_strategy_agricultural_dependence_anchor()
        current = float(getattr(self.characteristics, 'agricultural_dependence', anchor) or anchor)
        blend = max(0.0, min(1.0, float(blend)))
        self.characteristics.agricultural_dependence = max(
            0.0,
            min(1.0, blend * anchor + (1.0 - blend) * current),
        )

    def _get_income_signal(self):
        """Decision-facing income signal aligned with policy evaluation semantics."""
        try:
            net_income = float(getattr(self, 'net_income', None))
            return net_income
        except (TypeError, ValueError):
            pass
        try:
            return float(getattr(self, 'gross_income', getattr(self, 'total_income', 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_transfer_in_willingness(self):
        """
        @theory
        计算土地转入意愿
        
        理论基础：Deininger & Jin (2005), 钱忠好 (2003)
        
        转入意愿受以下因素影响：
        1. 年龄：年轻农户更愿意扩大经营
        2. 非农收入占比：非农收入占比低的更愿意转入
        3. 农业能力：农业能力强的更愿意扩大规模
        4. 当前规模：接近家庭农场规模的更愿意继续扩大
        5. 生计策略：纯农户更愿意转入
        
        @refactor 2025-12: 增加农户规模和家庭农场因素
        """
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        m_age = float(transfer_cfg.get("m_age", 45.0))
        off_ratio = self._get_off_farm_income_ratio()
        age_term = (float(self.age) - m_age) / 10.0
        
        # 基础转入意愿公式
        num = 16.0 * np.exp(age_term - 5.0 * off_ratio)
        denom = (1.0 + np.exp(age_term)) ** 2 * (1.0 + np.exp(-5.0 * off_ratio)) ** 2
        if denom <= 0.0:
            base_prob = 0.0
        else:
            base_prob = num / denom
        
        # @new 2025-12: 农业能力加成
        # 农业能力强的农户更愿意扩大规模（规模效益）
        ability_bonus = (self.farming_ability - 1.0) * 0.2  # 能力每高0.1，意愿增加2%
        
        # @new 2025-12: 规模经济动机
        # 如果接近家庭农场规模（50亩），转入意愿增加
        operated_area = self.get_operated_land_area()
        if 30 <= operated_area < 50:
            # 接近家庭农场规模，有动力继续扩大
            scale_bonus = 0.15
        elif 15 <= operated_area < 30:
            # 中等规模，有一定扩张动力
            scale_bonus = 0.10
        else:
            scale_bonus = 0.0
        
        # @new 2025-12: 生计策略调整
        # @refactor 2025-12: 用连续变量替代离散类型判断
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越高，转入意愿越强
            strategy_bonus = 0.3 * (agri_dep - 0.5)  # agri_dep=1 → +0.15, agri_dep=0 → -0.15
        else:
            # 回退：使用离散判断
            if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                strategy_bonus = 0.15
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                strategy_bonus = 0.05
            else:
                strategy_bonus = -0.20
        
        prob = base_prob + ability_bonus + scale_bonus + strategy_bonus
        
        # @new 2025-12-23: 最优规模与转入意愿深度耦合
        try:
            scale_info = self.calculate_optimal_operated_scale()
            if scale_info['decision'] == 'expand':
                # 需扩张时，转入意愿显著提升
                gap_bonus = min(0.30, scale_info['scale_gap'] / 100)  # 每差距100亩+30%意愿
                prob += gap_bonus
            elif scale_info['decision'] == 'contract':
                # 已超规模时，转入意愿降低
                prob -= 0.20
        except Exception as exc:
            record_runtime_fallback(
                self.model,
                component="FarmerAgent",
                context="get_transfer_in_willingness.calculate_optimal_operated_scale",
                error=exc,
                agent_id=getattr(self, "unique_id", None),
                fallback="skip_optimal_scale_adjustment",
            )

        recent_net_income = self._get_recent_net_income_signal()
        if recent_net_income < 0:
            prob *= 0.35
        elif recent_net_income < 5000:
            prob *= 0.75

        # Ensure strategy-specific transfer parameters feed the continuous
        # willingness channel even when characteristics are available.
        prob += self._get_strategy_transfer_willingness_adjustment(direction="in")

        return float(max(0.0, min(1.0, prob)))

    def get_transfer_out_willingness(self):
        """
        @theory
        计算土地转出意愿
        
        理论基础：Deininger & Jin (2005), 钱忠好 (2003)
        
        转出意愿受以下因素影响：
        1. 年龄：老年农户更愿意转出
        2. 非农收入占比：非农收入占比高的更愿意转出
        3. 健康状况：（简化为年龄代理）
        4. 非农就业机会：有稳定非农工作的更愿意转出
        5. 生计策略：非农户更愿意转出
        
        @refactor 2025-12: 增加非农转型和生计策略因素
        """
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        n_age = float(transfer_cfg.get("n_age", 30.0))
        off_ratio = self._get_off_farm_income_ratio()
        age_term = (float(self.age) - n_age) / 10.0
        
        # 基础转出意愿公式
        num = 16.0 * np.exp(age_term + 5.0 * off_ratio - 5.0)
        denom = (1.0 + np.exp(age_term)) ** 2 * (1.0 + np.exp(5.0 * off_ratio - 5.0)) ** 2
        if denom <= 0.0:
            base_prob = 0.0
        else:
            base_prob = num / denom
        
        # @new 2025-12: 老龄化加成
        # 60岁以上农户转出意愿显著增加
        if self.age >= 60:
            age_bonus = 0.18
        elif self.age >= 55:
            age_bonus = 0.10
        else:
            age_bonus = 0.0
        
        # @new 2025-12: 非农就业稳定性
        # 如果有稳定的非农收入（连续多年非农收入占比高），更愿意转出
        if hasattr(self, 'consecutive_high_non_farm_ratio') and self.consecutive_high_non_farm_ratio >= 3:
            stability_bonus = 0.12
        elif off_ratio > 0.7:
            stability_bonus = 0.06
        else:
            stability_bonus = 0.0
        
        # @new 2025-12: 生计策略调整
        # @refactor 2025-12: 用连续变量替代离散类型判断
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越低，转出意愿越强
            strategy_bonus = 0.5 * (0.5 - agri_dep)  # agri_dep=0 → +0.25, agri_dep=1 → -0.25
        else:
            # 回退：使用离散判断
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_bonus = 0.25
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                strategy_bonus = 0.10
            else:
                strategy_bonus = -0.10
        
        prob = base_prob + age_bonus + stability_bonus + strategy_bonus
        
        # @new 2025-12-23: 土地面积与转出意愿深度耦合
        try:
            transfer_area, _ = self.calculate_transfer_out_area()
            operated_area = self.get_operated_land_area()
            if operated_area > 0:
                area_transfer_ratio = transfer_area / operated_area
                # 建议转出面积越大，意愿越强
                prob += 0.08 * area_transfer_ratio
            
            # === 代际事件触发叠加影响 ===
            if self.age >= 65 and operated_area > 50:
                # 65岁+面积>50亩，流出意愿额外提升30%
                prob += 0.15
        except Exception as exc:
            record_runtime_fallback(
                self.model,
                component="FarmerAgent",
                context="get_transfer_out_willingness.calculate_transfer_out_area",
                error=exc,
                agent_id=getattr(self, "unique_id", None),
                fallback="skip_transfer_area_adjustment",
            )

        # Ensure strategy-specific transfer parameters feed the continuous
        # willingness channel even when characteristics are available.
        prob += self._get_strategy_transfer_willingness_adjustment(direction="out")
        
        return float(max(0.0, min(1.0, prob)))

    def _get_strategy_transfer_threshold_modifier(self, direction: str) -> float:
        """Resolve strategy-specific threshold modifiers from config."""
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        direction_key = str(direction or "").strip().lower()

        modifier = 1.0
        if direction_key == "out":
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                modifier = transfer_cfg.get("non_farm_out_multiplier", 0.6)
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                modifier = transfer_cfg.get("pluriactive_out_multiplier", 0.85)
        elif direction_key == "in":
            if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                modifier = transfer_cfg.get("pure_farmer_in_multiplier", 0.9)
            elif self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                modifier = transfer_cfg.get("non_farm_in_multiplier", 1.2)

        try:
            return max(0.1, float(modifier))
        except (TypeError, ValueError):
            return 1.0

    def _get_strategy_transfer_willingness_adjustment(self, direction: str) -> float:
        """
        Translate threshold-style strategy modifiers into a damped additive
        willingness adjustment so they affect the main transmission chain.
        """
        modifier = self._get_strategy_transfer_threshold_modifier(direction)
        direction_key = str(direction or "").strip().lower()
        scale = 0.18 if direction_key == "in" else 0.20
        normalized_effect = max(-1.0, min(1.0, 1.0 - modifier))
        return scale * normalized_effect

    def calculate_transfer_out_area(self):
        """
        @theory 土地流转决策与面积挂钩 - 转出面积计算
        @new 2025-12-23: 实现年龄×土地总面积×非农收入占比的三维耦合计算
        
        理论依据：
        - Deininger & Jin (2005): 年龄与流转意愿正相关
        - 罗必良 (2022): 不同规模农户流转行为差异显著
        - 代际接班模式影响保留面积
        
        区域异质性：基于spatial_module.py的区域参数调整
        - 东北平原: 地块大(10亩)，细碎化低(0.2)，倾向大规模流转
        - 西南山区: 地块小(2亩)，细碎化高(0.8)，倾向保守流转
        
        Returns:
            tuple: (transfer_area, retained_area) 建议转出面积和保留面积（亩）
        """
        total_land_area = sum(p.area for p in self.land_patches if p.owner == self)
        off_farm_ratio = self._get_off_farm_income_ratio()
        
        # === 1. 获取区域参数 ===
        # 尝试从环境获取空间异质性参数
        regional_params = self._get_regional_land_transfer_params()
        
        # === 2. 土地面积区间系数（区域异质化） ===
        # 基准阈值 (参考全国平均3.5亩户均)
        small_threshold = regional_params.get('small_land_threshold', 5.0)  # <5亩
        medium_threshold = regional_params.get('medium_land_threshold', 30.0)  # 5-30亩
        
        if total_land_area < small_threshold:
            area_coefficient = 0.3  # 小规模: 保守转出30%
        elif total_land_area <= medium_threshold:
            area_coefficient = 0.5  # 中等规模: 适度转出50%
        else:
            area_coefficient = 0.7  # 大规模: 积极转出70%
        
        # === 3. 年龄系数 ===
        if self.age >= 65:
            age_coefficient = 1.0  # 65岁以上: 完全转出意愿
        elif self.age >= 55:
            age_coefficient = 0.8  # 55-65岁: 高转出意愿
        elif self.age >= 45:
            age_coefficient = 0.5  # 45-55岁: 中等意愿
        elif self.age >= 40:
            age_coefficient = 0.3  # 40-45岁: 低意愿
        else:
            age_coefficient = 0.1  # <40岁: 倾向保留/扩大
        
        # === 4. 非农收入占比系数 ===
        non_farm_coefficient = off_farm_ratio  # 非农收入越高越愿意转出
        
        # === 5. 代际接班模式联动 ===
        generation_type = self._determine_generation_type()
        if generation_type == 1:  # 第一代农民(55+)
            # 检查是否有继承人
            if hasattr(self, 'has_successor') and self.has_successor:
                # 部分接班: 保留面积 = 人均0.5亩 × 家庭人口
                min_retained = max(0.5 * self.household_size, 1.0)
            else:
                # 不接班: 可以全部转出
                min_retained = 0.0
        elif generation_type == 2:  # 第二代农民(40-54)
            # 自己种+考虑未来接班
            min_retained = max(1.0 * self.household_size, 3.0)
        else:  # 第三代农民(<40)
            # 年轻人倾向扩大规模，保留较多
            min_retained = min(total_land_area, 10.0)
        
        # === 6. 区域细碎化修正 ===
        # 细碎化程度高的地区，倾向于整体流转或完全不流转
        fragmentation = regional_params.get('fragmentation_index', 0.5)
        if fragmentation > 0.6:  # 高细碎化（如西南山区）
            # 要么全部转出，要么保留
            if area_coefficient > 0.5:
                area_coefficient = 0.8  # 倾向全部转出
            else:
                area_coefficient *= 0.5  # 更加保守
        
        # === 7. 综合计算 ===
        base_transfer_ratio = area_coefficient * age_coefficient * (0.5 + 0.5 * non_farm_coefficient)
        base_transfer_ratio = min(1.0, max(0.0, base_transfer_ratio))
        
        suggested_transfer_area = total_land_area * base_transfer_ratio
        retained_area = max(min_retained, total_land_area - suggested_transfer_area)
        actual_transfer_area = max(0, total_land_area - retained_area)
        
        return actual_transfer_area, retained_area

    def calculate_optimal_operated_scale(self):
        """
        @theory 计算农户最优经营规模
        @new 2025-12-23: 将当前面积与最优规模的差值作为流转决策核心触发条件
        
        理论依据：
        - 规模经济理论: 存在最优经营规模
        - 黄季焜等(2023): 中国家庭农场规模50-200亩, 小农户5-30亩
        - 区域异质性: 东北大规模(100亩+), 西南小规模(5-10亩)
        
        差值阈值：
        - 差值>20亩: 积极转入
        - 差值<-10亩: 积极转出
        
        Returns:
            dict: {
                'optimal_scale': 最优规模(亩),
                'current_scale': 当前经营面积(亩),
                'scale_gap': 差距(正=需扩张, 负=需收缩),
                'decision': 'expand'/'contract'/'maintain',
                'expand_threshold': 扩张阈值,
                'contract_threshold': 收缩阈值
            }
        """
        current_scale = self.get_operated_land_area()
        
        # === 1. 获取区域参数 ===
        regional_params = self._get_regional_land_transfer_params()
        
        # === 2. 基于农户类型确定基础最优规模 ===
        if self.farmer_scale == FarmerScale.FAMILY_FARM:
            # 家庭农场: 50-200亩
            base_optimal = regional_params.get('optimal_family_farm', 100.0)
        elif self.farmer_scale == FarmerScale.MEDIUM_FARMER:
            # 中等规模户: 20-50亩
            base_optimal = regional_params.get('optimal_medium_farmer', 30.0)
        else:
            # 小农户: 5-15亩
            base_optimal = regional_params.get('optimal_smallholder', 10.0)
        
        # === 3. 耕作能力修正 ===
        optimal_scale = base_optimal * self.farming_ability
        
        # === 4. 年龄修正 ===
        if self.age >= 60:
            optimal_scale *= 0.7  # 老龄农户最优规模降低30%
        elif self.age < 40:
            optimal_scale *= 1.2  # 年轻农户可以经营更大规模
        
        # === 5. 土地类型修正 ===
        if hasattr(self, 'land_patches') and self.land_patches:
            # 基本农田增加最优规模（机械化条件好）
            basic_farmland_count = sum(
                1 for p in self.land_patches 
                if getattr(p, 'land_type', None) == 'basic_farmland'
            )
            basic_farmland_ratio = basic_farmland_count / len(self.land_patches) if self.land_patches else 0
            
            # 机械化潜力修正
            mech_potential = regional_params.get('mechanization_potential', 0.7)
            optimal_scale *= (1 + 0.2 * basic_farmland_ratio * mech_potential)
        
        # === 6. 计算差距和决策阈值（区域异质化） ===
        expand_threshold = regional_params.get('expand_threshold', 20.0)
        contract_threshold = regional_params.get('contract_threshold', 10.0)
        
        scale_gap = optimal_scale - current_scale
        
        if scale_gap > expand_threshold:  # 差距>阈值时积极转入
            decision = 'expand'
        elif scale_gap < -contract_threshold:  # 超出阈值时考虑转出
            decision = 'contract'
        else:
            decision = 'maintain'
        
        return {
            'optimal_scale': optimal_scale,
            'current_scale': current_scale,
            'scale_gap': scale_gap,
            'decision': decision,
            'expand_threshold': expand_threshold,
            'contract_threshold': contract_threshold
        }

    def _get_regional_land_transfer_params(self):
        """
        @new 2025-12-23: 获取区域化的土地流转参数
        
        基于spatial_module.py中的区域划分:
        - 华北平原: avg_plot_size=5亩, fragmentation=0.4
        - 东北平原: avg_plot_size=10亩, fragmentation=0.2
        - 长江中下游: avg_plot_size=3.5亩, fragmentation=0.5
        - 西南山区: avg_plot_size=2亩, fragmentation=0.8
        - 西北干旱: avg_plot_size=6亩, fragmentation=0.35
        
        Returns:
            dict: 区域化参数字典
        """
        # 默认参数（全国平均）
        default_params = {
            'small_land_threshold': 5.0,       # <5亩为小规模
            'medium_land_threshold': 30.0,     # 5-30亩为中等规模
            'optimal_family_farm': 100.0,      # 家庭农场最优规模
            'optimal_medium_farmer': 30.0,     # 中等规模户最优规模
            'optimal_smallholder': 10.0,       # 小农户最优规模
            'expand_threshold': 20.0,          # 扩张触发阈值
            'contract_threshold': 10.0,        # 收缩触发阈值
            'fragmentation_index': 0.5,        # 细碎化程度
            'mechanization_potential': 0.7,    # 机械化潜力
        }
        
        # 尝试从模型环境获取空间异质性管理器
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'spatial_manager'):
            spatial_manager = self.model.environment.spatial_manager
            if spatial_manager and hasattr(spatial_manager, 'params'):
                sp = spatial_manager.params
                # 根据区域调整参数
                avg_plot = getattr(sp, 'avg_plot_size', 5.0)
                frag = getattr(sp, 'fragmentation_index', 0.5)
                mech = getattr(sp, 'mechanization_potential', 0.7)
                
                # 动态调整阈值（基于区域地块大小）
                # 东北(10亩): 阈值放大2倍; 西南(2亩): 阈值缩小0.5倍
                scale_factor = avg_plot / 5.0  # 相对于5亩标准
                
                return {
                    'small_land_threshold': 5.0 * scale_factor,
                    'medium_land_threshold': 30.0 * scale_factor,
                    'optimal_family_farm': 100.0 * scale_factor,
                    'optimal_medium_farmer': 30.0 * scale_factor,
                    'optimal_smallholder': 10.0 * scale_factor,
                    'expand_threshold': 20.0 * scale_factor,
                    'contract_threshold': 10.0 * scale_factor,
                    'fragmentation_index': frag,
                    'mechanization_potential': mech,
                }
        
        # 尝试从配置文件读取
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        if transfer_cfg:
            for key in default_params:
                if key in transfer_cfg:
                    default_params[key] = transfer_cfg[key]
        
        return default_params

    def get_land_reclaim_willingness(self):
        """
        @new 2025-12: 计算土地回收意愿（从租出状态收回土地）
        
        理论基础：中国农村土地流转实践
        
        土地回收通常发生在：
        1. 生计策略从非农转向农业（返乡农民）
        2. 非农就业不稳定时想回归农业
        3. 年轻一代接班想扩大经营
        4. 租金收益不理想想自己经营
        
        注意：土地回收需要合同到期或协商解除
        """
        # 基础回收意愿为0（一般情况下不会主动收回）
        base_prob = 0.0
        
        # 情况1：生计策略转向农业
        # 如果从非农户变成兼业户或纯农户，可能想收回土地
        if hasattr(self, 'previous_livelihood_strategy'):
            if (self.previous_livelihood_strategy == LivelihoodStrategy.NON_FARMER and
                self.livelihood_strategy in [LivelihoodStrategy.PURE_FARMER, LivelihoodStrategy.PLURIACTIVE]):
                base_prob += 0.30
        
        # 情况2：非农收入大幅下降（失业风险）
        if hasattr(self, 'off_farm_income_history') and len(self.off_farm_income_history) >= 2:
            recent_income = self.off_farm_income_history[-1]
            previous_income = self.off_farm_income_history[-2]
            if previous_income > 0 and recent_income / previous_income < 0.7:
                # 非农收入下降超过30%，想回归农业
                base_prob += 0.25
        
        # 情况3：年轻且农业能力强
        if self.age < 40 and self.farming_ability > 1.1:
            base_prob += 0.15
        
        # 情况4：租金收益不理想
        # 检查租金收入与预期的差距
        leased_out_area = sum(p.area for p in self.land_patches 
                            if p.owner == self and p.leased_to is not None)
        if leased_out_area > 0:
            avg_rent = sum(getattr(p, 'rental_price', 0) for p in self.land_patches 
                          if p.owner == self and p.leased_to is not None) / leased_out_area
            expected_rent = 800  # 预期租金（元/亩）
            if avg_rent < expected_rent * 0.7:
                base_prob += 0.10
        
        return float(max(0.0, min(1.0, base_prob)))

    def attempt_land_reclaim(self):
        """
        @new 2025-12: 尝试收回已流转出去的土地
        
        土地回收机制：
        1. 检查回收意愿
        2. 检查是否有已流转的土地
        3. 与承租方协商（简化：按概率成功）
        4. 执行回收
        """
        reclaim_willingness = self.get_land_reclaim_willingness()
        
        if reclaim_willingness < 0.2:
            return  # 意愿太低，不尝试回收
        
        # 获取已流转出去的土地
        leased_out_patches = [p for p in self.land_patches 
                             if p.owner == self and p.leased_to is not None]
        
        if not leased_out_patches:
            return  # 没有可回收的土地
        
        for patch in leased_out_patches:
            # 回收成功概率 = 回收意愿 × 协商成功率
            # 协商成功率取决于承租方的依赖程度
            lessee = patch.leased_to
            if lessee is None:
                continue
            
            # 承租方如果是企业，协商更困难
            if hasattr(lessee, 'enterprise_type'):
                negotiation_rate = 0.3  # 企业更难协商
            else:
                # 农户承租方，看其对这块地的依赖
                lessee_total_area = lessee.get_operated_land_area() if hasattr(lessee, 'get_operated_land_area') else 50
                dependency = patch.area / max(1, lessee_total_area)
                negotiation_rate = 0.6 - dependency * 0.3  # 依赖程度越高越难协商
            
            success_prob = reclaim_willingness * max(0.2, negotiation_rate)
            
            if rng_random(self) < success_prob:
                # 回收成功
                self._execute_land_reclaim(patch, lessee)

    def _execute_land_reclaim(self, patch, lessee):
        """
        @new 2025-12: 执行土地回收
        """
        # 清除流转状态
        patch.leased_to = None
        patch.current_operator = self
        patch.rental_price = 0
        patch.contracted_rental_price = 0
        patch.lease_contract_years = 0
        setattr(patch, 'operational_transfer_active', False)
        setattr(patch, 'operational_transfer_channel', '')
        
        # 更新双方的流转列表
        if patch in self.leased_out_patches:
            self.leased_out_patches.remove(patch)
        
        if hasattr(lessee, 'leased_in_patches') and patch in lessee.leased_in_patches:
            lessee.leased_in_patches.remove(patch)
        if hasattr(lessee, 'leased_patches') and patch in lessee.leased_patches:
            lessee.leased_patches.remove(patch)
        for holder in (self, lessee):
            if holder is None:
                continue
            for attr_name in ('_leased_patch_cache_version', '_operated_land_area_cache_version'):
                if hasattr(holder, attr_name):
                    setattr(holder, attr_name, None)
        if hasattr(self.model, 'invalidate_land_tenure_caches'):
            self.model.invalidate_land_tenure_caches()
        
        # 记录回收事件
        if not hasattr(self.model, 'land_reclaim_records'):
            self.model.land_reclaim_records = []
        self.model.land_reclaim_records.append({
            'step': getattr(self.model, 'step_count', 0),
            'reclaimer': self.unique_id,
            'former_lessee': lessee.unique_id if lessee else None,
            'area': patch.area,
            'reason': 'livelihood_change'
        })

    def _should_transfer_out_locally(self, willingness=None):
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        threshold = float(transfer_cfg.get("threshold_out", 0.2))
        if willingness is None:
            willingness = self.get_transfer_out_willingness()

        # @refactor 2025-12: 将连续变量与配置中的主体乘数同时纳入
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越低，转出门槛越低（modifier越小）
            continuous_modifier = 0.5 + 0.5 * agri_dep  # agri_dep=0 → 0.5, agri_dep=1 → 1.0
        else:
            continuous_modifier = 1.0
        strategy_modifier = self._get_strategy_transfer_threshold_modifier(direction="out")
        modifier = continuous_modifier * strategy_modifier

        if self.age >= transfer_cfg.get("elderly_age_threshold", 60) and self._qualifies_for_elderly_land_release():
            return True
        return willingness >= threshold * modifier

    def _should_transfer_in_locally(self, willingness=None):
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        threshold = float(transfer_cfg.get("threshold_in", 0.6))
        if willingness is None:
            willingness = self.get_transfer_in_willingness()

        # @refactor 2025-12: 将连续变量与配置中的主体乘数同时纳入
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越高，转入门槛越低（modifier越小）
            continuous_modifier = 0.8 + 0.4 * (1 - agri_dep)  # agri_dep=1 → 0.8, agri_dep=0 → 1.2
        else:
            continuous_modifier = 1.0
        strategy_modifier = self._get_strategy_transfer_threshold_modifier(direction="in")
        modifier = continuous_modifier * strategy_modifier

        if self.check_off_farm_opportunity() and self.off_farm_income > 0:
            modifier *= 1.1
        # 与转出逻辑保持同一口径：modifier 越小，转入门槛越低。
        return willingness >= threshold * modifier

    def _select_patch_for_local_transfer(self, min_retained_area_per_capita=0.3):
        """选择可供流转的边际地块，保留家庭自用面积"""
        available = [p for p in self.land_patches if p.owner == self and p.leased_to is None]
        if not available:
            return None
        min_retained_area = self.household_size * float(min_retained_area_per_capita)
        retained_area = sum(p.area for p in available)
        available.sort(key=lambda p: p.area)
        for patch in available:
            if retained_area - patch.area >= min_retained_area:
                return patch
        return None

    def _apply_local_transfer_transaction(self, patch, lessee, rent_price):
        """执行局部土地流转的状态更新"""
        lease_applied = patch.set_leased(
            lessee,
            rental_price=rent_price,
            transfer_channel='direct_farmer'
        )
        if lease_applied is False:
            return False
        if patch not in self.leased_out_patches:
            self.leased_out_patches.append(patch)
        if patch not in lessee.leased_in_patches:
            lessee.leased_in_patches.append(patch)
        if hasattr(lessee, 'leased_patches'):
            if patch not in lessee.leased_patches:
                lessee.leased_patches.append(patch)
        # 记录局部交易供政府或分析模块使用
        if not hasattr(self.model, 'local_transfer_records'):
            self.model.local_transfer_records = []
        self.model.local_transfer_records.append({
            'step': getattr(self.model, 'step_count', 0),
            'lessor': self.unique_id,
            'lessee': lessee.unique_id,
            'area': patch.area,
            'price': rent_price,
            'price_per_mu': float(rent_price) / max(float(getattr(patch, 'area', 0.0) or 0.0), 1e-6),
            'channel': 'network'
        })

        return True

    def _get_pairwise_link_strength(self, neighbor):
        """根据信息网络与交易网络强度，构建二人之间的信任度"""
        own_strength = self.transaction_link_strength.get(neighbor.unique_id)
        neighbor_strength = neighbor.transaction_link_strength.get(self.unique_id) if hasattr(neighbor, 'transaction_link_strength') else None
        fallback = self.network_strength.get(neighbor.unique_id, 0.5)
        neighbor_fallback = neighbor.network_strength.get(self.unique_id, 0.5) if hasattr(neighbor, 'network_strength') else 0.5
        scores = [s for s in [own_strength, neighbor_strength, fallback, neighbor_fallback] if s is not None]
        if not scores:
            return 0.4
        return min(1.0, max(0.1, sum(scores) / len(scores)))

    def _attempt_local_transfer(self, lessor, lessee, mechanism, link_strength):
        """调用机制协调器评估局部匹配并执行"""
        transfer_cfg = lessor.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        min_retained_area = transfer_cfg.get("min_retained_area_per_capita", 0.3)
        patch = lessor._select_patch_for_local_transfer(min_retained_area)
        if patch is None:
            return False
        clearing_house = getattr(self.model, 'land_clearing_house', None)
        if (
            clearing_house is not None and
            hasattr(clearing_house, '_check_transfer_rate_limit') and
            not clearing_house._check_transfer_rate_limit(
                self.model,
                getattr(patch, 'area', 0.0),
                transfer_channel='direct_farmer',
                demander=lessee,
            )
        ):
            return False
        land_quality = getattr(patch, 'soil_quality', 0.6)
        subsidy_level = self.model.environment.get_policy_param('grain_subsidies_rate')
        social_strength = min(1.0, max(0.05, link_strength))
        lessee_area_gap = max(0.0, getattr(lessee, 'target_operated_area', lessee._compute_target_operated_area()) - lessee.get_operated_land_area())
        area_factor = min(1.0, lessee_area_gap / max(1.0, patch.area))
        transfer_prob = mechanism.land_transfer.calculate_transfer_probability(
            land_quality,
            getattr(lessee, 'farming_ability', 1.0),
            getattr(lessor, 'farming_ability', 1.0),
            subsidy_level,
            social_strength
        )
        transfer_prob *= (0.5 + 0.5 * area_factor)
        transfer_prob = min(1.0, transfer_prob)
        if rng_random(self) > transfer_prob:
            return False
        base_price = float(transfer_cfg.get("base_rent_price", 650))
        subsidy_level = self.model.environment.get_policy_param('grain_subsidies_rate') or 0.0
        price_per_mu = mechanism.land_transfer.calculate_transfer_price(
            land_quality,
            getattr(lessee, 'farming_ability', 1.0),
            base_price,
            subsidy_level
        )
        total_price = float(price_per_mu) * float(getattr(patch, 'area', 0.0) or 0.0)
        lessor._apply_local_transfer_transaction(patch, lessee, total_price)
        return True

    def run_local_land_transfer_round(self):
        """
        @theory
        基于交易网络的局部土地流转：参考Deininger & Jin (2005)，先在熟人网络中撮合，
        再进入中央清算所，捕捉“先友后市”的土地配置过程。
        """
        mechanism = getattr(self.model.environment, 'mechanism_coordinator', None)
        if mechanism is None:
            return
        neighbors = list(self.transaction_neighbors or self.social_network)
        if not neighbors:
            return
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        max_matches = int(transfer_cfg.get("max_local_matches", 2))
        if max_matches <= 0:
            return
        rng_shuffle(self, neighbors)
        matches = 0
        for neighbor_id in neighbors:
            if matches >= max_matches:
                break
            partner = self.model.get_agent_by_id(neighbor_id)
            if partner is None or partner.unique_id == self.unique_id:
                continue
            if not hasattr(partner, 'land_patches'):
                continue
            link_strength = self._get_pairwise_link_strength(partner)
            paired = False
            if self._should_transfer_out_locally() and partner._should_transfer_in_locally():
                paired = self._attempt_local_transfer(self, partner, mechanism, link_strength)
            elif self._should_transfer_in_locally() and partner._should_transfer_out_locally():
                paired = self._attempt_local_transfer(partner, self, mechanism, link_strength)
            if paired:
                matches += 1

    def _determine_generation_type(self) -> int:
        """
        @new 2025-12: 根据年龄判断农户所属的代际类型
        
        Returns:
            int: 1=第一代农民(60-70后, 年龄55+), 
                 2=第二代农民(50-60后, 年龄40-54), 
                 3=新生代农民(80-90后, 年龄<40)
        
        @theory
        代际差异是理解"谁来种粮"的关键分析维度：
        - 第一代：对土地有深厚感情，倾向自己种地直到体力不允许
        - 第二代：在外务工多年，对种地兴趣降低，但会考虑社会保障
        - 新生代：多不愿从事农业，除非是"新型职业农民"
        """
        if self.age >= 55:
            return 1  # 第一代农民
        elif self.age >= 40:
            return 2  # 第二代农民
        else:
            return 3  # 新生代农民
    
    def _initialize_expectation_mechanism(self):
        """
        @new 2025-11-30: 初始化异质性预期机制
        
        理论依据：
        - 适应性预期（Adaptive Expectations）：小农户根据历史经验形成预期
        - 有限理性预期（Bounded Rational）：规模农户/家庭农场有部分前瞻能力
        - 理性预期（Rational Expectations）：大型经营主体使用市场信息
        
        不同类型农户使用不同的预期形成机制：
        - 老年小农（>55岁）：适应性预期，adjustment_speed较低
        - 中年兼业（40-55岁）：适应性预期，adjustment_speed中等
        - 年轻规模户（<40岁）：有限理性预期
        """
        try:
            from modules.expectation_mechanism import (
                ExpectationFactory, 
                ExpectationManager,
                AdaptiveExpectation
            )
            
            # 确定农户类型
            farmer_type = getattr(self.livelihood_strategy, 'value', 'Pure_Farmer') if hasattr(self, 'livelihood_strategy') else 'Pure_Farmer'
            experience_years = max(1, self.age - 18) if hasattr(self, 'age') else 10
            
            # 获取更多农户特征用于异质性预期参数
            education_years = getattr(self, 'education_years', 9)
            land_scale = self.get_operated_land_area() if hasattr(self, 'get_operated_land_area') else 10.0
            risk_preference = getattr(self, 'risk_preference', 0.5)
            social_capital = getattr(self, 'social_capital', 0.5)
            
            # 如果有characteristics对象，优先使用其中的值
            if hasattr(self, 'characteristics') and self.characteristics:
                education_years = getattr(self.characteristics, 'education_years', education_years)
                risk_preference = getattr(self.characteristics, 'risk_preference', risk_preference)
                social_capital = getattr(self.characteristics, 'social_capital', social_capital)
            
            # 使用工厂创建预期机制 - 传入完整的特征参数
            base_mechanism = ExpectationFactory.create_for_farmer(
                farmer_type=farmer_type,
                age=getattr(self, 'age', 45),
                experience_years=experience_years,
                education_years=education_years,
                land_scale=land_scale,
                risk_preference=risk_preference,
                social_capital=social_capital
            )
            
            # 创建预期管理器
            self.expectation_manager = ExpectationManager(base_mechanism)
            
            # 初始化历史数据（使用默认值，因为environment可能还未准备好）
            grain_price = 2.5
            cash_price = 5.0
            wage_rate = self._get_base_non_farm_wage()
            
            # 尝试从environment获取价格
            if hasattr(self, 'model') and self.model:
                env = getattr(self.model, 'environment', None)
                if env and hasattr(env, 'get_market_price'):
                    try:
                        gp = env.get_market_price('grain')
                        if gp is not None:
                            grain_price = gp
                        cp = env.get_market_price('cash_crop')
                        if cp is not None:
                            cash_price = cp
                        wr = env.get_market_price('wage_rate')
                        if wr is not None:
                            wage_rate = wr
                    except Exception:
                        pass  # 使用默认值
            
            # 添加初始历史
            for _ in range(3):
                self.expectation_manager.update_history(
                    grain_price=grain_price,
                    cash_price=cash_price,
                    wage=wage_rate
                )
                        
        except ImportError as e:
            # 如果模块不可用，使用简单预期
            self.expectation_manager = None
        except Exception as e:
            self.expectation_manager = None

    def update_farmer_characteristics(self):
        """
        @new 2025-12: 更新农户的连续变量特征
        
        每一步更新农户的特征变量，反映其动态变化：
        - 年龄增长（老龄化率变化）
        - 土地规模变化（通过流转）
        - 收入结构变化（农业依赖度）
        - 社会资本积累
        """
        if not hasattr(self, 'characteristics') or self.characteristics is None:
            return
        
        # 更新土地规模
        current_land_area = self.get_operated_land_area()
        self.characteristics.land_scale = current_land_area

        previous_ag_dependence = float(
            getattr(self.characteristics, 'agricultural_dependence', 0.5) or 0.5
        )
        self._sync_characteristics_income_fields(use_strategy_anchor=False)
        
        # 更新农业依赖度（基于收入结构）
        total_income = self._get_agriculture_linked_income() + max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
        if total_income > 0:
            new_agri_dep = self._get_livelihood_agricultural_income_ratio()
            # 平滑更新，避免剧烈波动
            self.characteristics.agricultural_dependence = (
                0.35 * previous_ag_dependence +
                0.65 * new_agri_dep
            )
        
        # 更新老龄化率（基于年龄变化）
        self.characteristics.aging_rate = max(0, min(1, (self.age - 40) / 40))
        
        # 更新生计多样性指数
        income_sources = []
        agriculture_linked_income = self._get_agriculture_linked_income()
        if agriculture_linked_income > 0:
            income_sources.append(agriculture_linked_income)
        if self.off_farm_income > 0:
            income_sources.append(self.off_farm_income)
        rental_income = getattr(self, 'land_rental_income', 0)
        if rental_income > 0:
            income_sources.append(rental_income)
        
        if len(income_sources) > 1:
            total = sum(income_sources)
            shares = [s / total for s in income_sources]
            hhi = sum(s ** 2 for s in shares)
            self.characteristics.livelihood_diversity = 1 - hhi
        else:
            self.characteristics.livelihood_diversity = 0.0
        
        # 重新分类农户类型
        self.farmer_type, self.farmer_type_confidence = FarmerTypeClassifier.classify(self.characteristics)

    def _check_generational_succession(self):
        """
        @critical 2025-12: 代际接替机制
        
        理论依据：中国农村土地流转爆发式增长的根本驱动力是：
        - 60后户主65岁前后集中出租（2020-2023年数据）
        - 90后拒绝回乡务农
        
        参考：
        - 王济民等 (2023) 《农村土地流转的代际效应》
        - 中国农村固定观察点数据 (2020-2023)
        """
        if not hasattr(self, 'characteristics') or self.characteristics is None:
            return
        
        # 导入LifecycleManager
        from modules.farmer_characteristics import LifecycleManager
        
        # 更新特征中的年龄
        self.characteristics.household_head_age = self.age
        
        # 检查代际接替事件
        event = LifecycleManager.check_aging_event(self.characteristics)
        
        if event is None:
            return
        
        # 处理不同类型的事件
        if event == 'land_outflow':
            # 土地流出：老龄户主倾向于出租土地
            self._trigger_land_outflow()
        
        elif event == 'land_inflow':
            # 土地流入：子女返乡创业，扩大经营
            self._trigger_land_inflow()
        
        elif event == 'health_shock':
            # 健康冲击：可能导致劳动力下降
            self._handle_health_shock()
        
        elif event == 'succession':
            # 代际接替完成：户主更换为下一代
            self._complete_succession()
        
        elif event == 'urban_migration':
            # 全家城市迁移：成为"离土不离权"地主
            self._complete_urban_migration()
    
    def _trigger_land_outflow(self):
        """
        @critical 2025-12: 触发土地流出
        老龄户主开始出租土地
        """
        # 增加土地流出意愿
        self.land_transfer_willingness = min(1.0, getattr(self, 'land_transfer_willingness', 0.5) + 0.3)
        
        # 降低对土地的保留价格（更愿意出租）
        self.reservation_price_adjustment = getattr(self, 'reservation_price_adjustment', 1.0) * 0.9
        
        # 可能触发生计策略向Non_Farmer转变
        if hasattr(self, 'characteristics'):
            self.characteristics.agricultural_dependence *= 0.8
        
        # 记录事件
        if hasattr(self, 'lifecycle_events'):
            self.lifecycle_events.append(('land_outflow', self.model.schedule.steps))
    
    def _trigger_land_inflow(self):
        """
        @critical 2025-12: 触发土地流入
        子女返乡创业，扩大经营规模
        """
        # 增加土地需求
        if hasattr(self, 'target_operated_area'):
            self.target_operated_area *= 1.5
        
        # 提高对土地的出价意愿
        self.bid_price_adjustment = getattr(self, 'bid_price_adjustment', 1.0) * 1.1
        
        # 提高农业依赖度
        if hasattr(self, 'characteristics'):
            self.characteristics.agricultural_dependence = min(1.0, 
                self.characteristics.agricultural_dependence * 1.3)
            self.characteristics.technology_adoption = min(1.0,
                self.characteristics.technology_adoption * 1.2)
        
        # 更新代际类型（接班的是新生代）
        if hasattr(self, 'characteristics'):
            self.characteristics.generation_type = 3  # 80后
            self.characteristics.successor_willing = True
        
        # 记录事件
        if hasattr(self, 'lifecycle_events'):
            self.lifecycle_events.append(('land_inflow', self.model.schedule.steps))
    
    def _handle_health_shock(self):
        """
        @critical 2025-12: 处理健康冲击
        劳动力下降，可能需要减少经营面积
        """
        # 减少有效劳动力
        self.household_labor_capacity = max(0.3, 
            getattr(self, 'household_labor_capacity', 1.0) * 0.7)
        
        # 增加流出意愿
        self.land_transfer_willingness = min(1.0, 
            getattr(self, 'land_transfer_willingness', 0.5) + 0.2)
        
        # 记录事件
        if hasattr(self, 'lifecycle_events'):
            self.lifecycle_events.append(('health_shock', self.model.schedule.steps))
    
    def _complete_succession(self):
        """
        @critical 2025-12: 完成代际接替（含部分接班机制）
        
        专家审阅意见：
        - 现有模型只有"全接"或"不接"两种情况
        - 现实中存在大量"部分接班"情形：子女保留30-50%自耕，其余交给合作社
        - "周末农民"模式：城市工作+周末回村管理
        
        三种接班模式：
        1. 全面接班(~25%)：子女完全回乡，扩大经营
        2. 部分接班(~35%)：保留部分自耕，大部分土地托管
        3. 不接班(~40%)：全部流出，成为缺席地主
        
        参考：
        - 张忠根等 (2022) 《代际分化与农地流转》
        - 费孝通 "离土不离乡" 概念的现代演绎
        """
        import numpy as np
        
        if not hasattr(self, 'characteristics') or self.characteristics is None:
            return
        
        # 根据子女条件决定接班类型
        succession_type = self._determine_succession_type()
        
        if succession_type == 'full':
            # 全面接班：子女完全回乡
            self._execute_full_succession()
            
        elif succession_type == 'partial':
            # 部分接班：周末农民模式
            self._execute_partial_succession()
            
        else:
            # 不接班：成为缺席地主
            self._execute_no_succession()
        
        # 记录事件
        if hasattr(self, 'lifecycle_events'):
            self.lifecycle_events.append(('succession', self.model.schedule.steps))
    
    def _determine_succession_type(self):
        """
        @new 2025-12: 确定接班类型
        
        影响因素：
        - successor_willing: 子女是否有意愿
        - 经营规模：大规模更容易吸引子女回乡
        - 区位：近郊更容易实现"部分接班"
        - 补贴水平：高补贴增加接班激励
        """
        import numpy as np
        
        if not self.characteristics.successor_willing:
            return 'none'
        
        # 基础概率
        prob_full = 0.25
        prob_partial = 0.35
        prob_none = 0.40
        
        # 经营规模效应
        if hasattr(self, 'operated_land') and self.operated_land > 30:
            # 规模大于30亩，全面接班概率提高
            prob_full += 0.15
            prob_partial -= 0.05
            prob_none -= 0.10
        
        # 区位效应（假设distance_to_city越小越近郊）
        if hasattr(self.characteristics, 'distance_to_city'):
            if self.characteristics.distance_to_city < 20:
                # 近郊更适合部分接班
                prob_partial += 0.10
                prob_full -= 0.05
                prob_none -= 0.05
        
        # 补贴预期效应
        if hasattr(self, 'received_subsidies') and self.received_subsidies:
            recent_subsidy = sum(list(self.received_subsidies.values())[-3:]) if self.received_subsidies else 0
            if recent_subsidy > 3000:  # 年均补贴超过3000元
                prob_full += 0.10
                prob_partial += 0.05
                prob_none -= 0.15
        
        # 归一化
        total = prob_full + prob_partial + prob_none
        prob_full /= total
        prob_partial /= total
        prob_none /= total
        
        # 随机抽取
        rand = np_random_value(self)
        if rand < prob_full:
            return 'full'
        elif rand < prob_full + prob_partial:
            return 'partial'
        else:
            return 'none'
    
    def _execute_full_succession(self):
        """
        @new 2025-12: 执行全面接班
        子女完全回乡，扩大经营规模
        """
        import numpy as np
        
        # 户主年龄重置为接班人年龄
        self.age = int(np_randint(self, 30, 40))
        self.characteristics.household_head_age = self.age
        
        # 更新代际类型
        self.characteristics.generation_type = 3  # 80后/90后
        
        # 更新老龄化率
        self.characteristics.aging_rate = max(0, self.characteristics.aging_rate - 0.3)
        
        # 新生代倾向于规模化经营
        if hasattr(self, 'target_operated_area'):
            self.target_operated_area *= 1.3
        
        # 技术采纳能力提升
        self.characteristics.technology_adoption = min(1.0,
            self.characteristics.technology_adoption * 1.5)
        
        # 农业依赖度恢复
        self.characteristics.agricultural_dependence = min(1.0,
            self.characteristics.agricultural_dependence * 1.3)
        
        # 生计策略可能转为规模农户（全面接班后成为纯农户）
        self.livelihood_strategy = LivelihoodStrategy.PURE_FARMER  # @fix: 使用正确的枚举值
        
        print(f"[代际] 农户 {self.unique_id}: 全面接班，新户主年龄 {self.age}")
    
    def _execute_partial_succession(self):
        """
        @new 2025-12: 执行部分接班 - "周末农民"模式
        
        核心特征：
        - 子女城市工作，但保留30-50%土地自耕
        - 其余土地交给合作社或专业户托管
        - 形成"周末回村"或"农忙返乡"模式
        - 农业依赖度降至0.3-0.5
        """
        import numpy as np
        
        # 保留自耕比例
        self_cultivation_ratio = float(np_uniform(self, 0.30, 0.50))
        
        # 记录部分接班状态
        self.partial_succession_ratio = self_cultivation_ratio
        self.is_weekend_farmer = True
        
        # 户主年龄重置（部分接班也是新生代）
        self.age = int(np_randint(self, 30, 45))
        self.characteristics.household_head_age = self.age
        self.characteristics.generation_type = 3
        
        # 农业依赖度降低但不完全脱离
        self.characteristics.agricultural_dependence = float(np_uniform(self, 0.30, 0.50))
        
        # 计算需要托管/流出的土地
        if hasattr(self, 'operated_land') and self.operated_land > 0:
            land_to_transfer = self.operated_land * (1 - self_cultivation_ratio)
            self.land_pending_transfer = land_to_transfer
            
            # 增加土地流出意愿（针对超出部分）
            self.land_transfer_willingness = 0.7
        
        # 生计策略变为兼业（部分接班者成为兼业农户）
        self.livelihood_strategy = LivelihoodStrategy.PLURIACTIVE  # @fix: 使用正确的枚举值
        
        # 更新农户类型
        from modules.farmer_characteristics import FarmerType
        self.farmer_type = FarmerType.WEEKEND_FARMER
        self.farmer_type_confidence = 0.85
        
        # 记录托管意向（优先考虑合作社托管）
        self.prefers_cooperative_custody = True
        
        print(f"[代际] 农户 {self.unique_id}: 部分接班（周末农民），"
              f"自耕比例 {self_cultivation_ratio:.1%}，待流转 {getattr(self, 'land_pending_transfer', 0):.1f}亩")
    
    def _execute_no_succession(self):
        """
        @new 2025-12: 无人接班
        全部土地流出，成为缺席地主
        """
        # 转变为Non_Farmer
        self.livelihood_strategy = LivelihoodStrategy.NON_FARMER
        
        # 更新类型为缺席地主
        from modules.farmer_characteristics import FarmerType
        self.farmer_type = FarmerType.ABSENTEE_LANDLORD
        self.farmer_type_confidence = 0.9
        
        # 农业依赖度归零
        self.characteristics.agricultural_dependence = 0.0
        
        # 所有土地流出
        self.land_transfer_willingness = 1.0
        self.reservation_price_adjustment = 0.8
        
        print(f"[代际] 农户 {self.unique_id}: 无人接班，成为缺席地主")
    
    def _complete_urban_migration(self):
        """
        @critical 2025-12: 全家城市迁移
        成为"离土不离权"的缺席地主（Absentee Landlord）
        """
        # 转变为Non_Farmer
        self.livelihood_strategy = LivelihoodStrategy.NON_FARMER
        
        # 更新精细类型为缺席地主
        from modules.farmer_characteristics import FarmerType
        self.farmer_type = FarmerType.ABSENTEE_LANDLORD
        self.farmer_type_confidence = 0.9
        
        # 农业依赖度降至最低
        if hasattr(self, 'characteristics'):
            self.characteristics.agricultural_dependence = 0.0
        
        # 所有土地都倾向于流出
        self.land_transfer_willingness = 1.0
        
        # 降低保留价格（更愿意出租）
        self.reservation_price_adjustment = 0.8
        
        # 记录事件
        if hasattr(self, 'lifecycle_events'):
            self.lifecycle_events.append(('urban_migration', self.model.schedule.steps))
        self.cohort_urbanized = True
        self.cohort_urbanization_reason = 'urban_migration'
        self.cohort_urbanization_step = int(
            getattr(self.model, 'step_count', getattr(self.model.schedule, 'steps', 0)) or 0
        )

    def _mark_structural_urbanization(self, transition_reason, agricultural_income_ratio):
        """
        Some households remain in the land-rights system as absentee owners after
        shifting to a durable non-farm livelihood. For validation they should be
        treated as urbanized rather than kept in the rural livelihood pool.
        """
        if getattr(self, 'cohort_urbanized', False):
            return
        if self.livelihood_strategy != LivelihoodStrategy.NON_FARMER:
            return

        # Only count explicit migration-style paths here. Most Non_Farmer
        # transitions are rural non-farm households, not urbanized households.
        if transition_reason not in {"城镇化吸引", "urban_migration"}:
            return

        evolution_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('livelihood_evolution', {})
        transition_cfg = evolution_cfg.get('transition_probabilities', {})
        non_farmer_threshold = float(evolution_cfg.get('non_farmer_threshold', 0.20) or 0.20)
        stable_offfarm_income_threshold = float(
            transition_cfg.get('stable_offfarm_income_threshold', 15000) or 15000
        )
        off_farm_ratio = self._get_off_farm_income_ratio()
        durable_non_farm_signal = (
            self.off_farm_income >= stable_offfarm_income_threshold * 0.85
            or off_farm_ratio >= 0.55
            or getattr(self, 'consecutive_high_non_farm_ratio', 0) >= 2
            or agricultural_income_ratio <= min(non_farmer_threshold, 0.20)
        )
        explicit_city_link = bool(getattr(self, 'has_urban_family', False))
        if transition_reason == "urban_migration":
            self.cohort_urbanized = True
            self.cohort_urbanization_reason = transition_reason
            self.cohort_urbanization_step = int(
                getattr(self.model, 'step_count', getattr(self.model.schedule, 'steps', 0)) or 0
            )
            return
        if explicit_city_link and durable_non_farm_signal and off_farm_ratio >= 0.70 and self.age < 60:
            self.cohort_urbanized = True
            self.cohort_urbanization_reason = transition_reason
            self.cohort_urbanization_step = int(
                getattr(self.model, 'step_count', getattr(self.model.schedule, 'steps', 0)) or 0
            )

    def _qualifies_for_elderly_land_release(self, agricultural_income_ratio=None):
        """
        Elderly households should not automatically become annual suppliers.
        Only households already on a durable non-farm path should bypass the
        normal profitability screen.
        """
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            return True
        if agricultural_income_ratio is None:
            agricultural_income_ratio = self._get_livelihood_agricultural_income_ratio()
        agricultural_income_ratio = max(0.0, min(1.0, float(agricultural_income_ratio or 0.0)))
        off_farm_ratio = max(0.0, min(1.0, float(self._get_off_farm_income_ratio() or 0.0)))

        if self.age >= 70 and (off_farm_ratio >= 0.45 or agricultural_income_ratio <= 0.45):
            return True
        if (
            self.age >= 65
            and getattr(self, 'consecutive_high_non_farm_ratio', 0) >= 2
            and off_farm_ratio >= 0.35
        ):
            return True
        return False

    def _compute_target_operated_area(self):
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        base_target = transfer_cfg.get("base_target_operated_area")
        if base_target is not None:
            return float(base_target)
        per_capita = transfer_cfg.get("target_area_per_capita", 5.0)
        return float(self.household_size) * float(per_capita)

    def participate_land_rental_market(self):
        """
        @theory
        核心博弈：土地租赁市场的价格战
        在确定了种植意向后，主体进入市场，通过价格战来调整其经营规模
        @fix: 修复地块状态管理，确保已流转地块状态被正确保留
        """
        fixed_action = getattr(self, "fixed_rule_current_action", None)
        if action_income_enabled(self.model.config):
            self.reservation_prices = {}
            if fixed_action == "exit":
                for parcel in self.model.land_patches:
                    if parcel.owner is self and parcel.leased_to is None:
                        rent = parcel.area * self.model.environment.land_rental_price
                        self.reservation_prices[parcel] = rent
                        self.model.land_clearing_house.add_supply_offer(parcel, self, rent)
            return
        if fixed_action == "exit":
            self._submit_pending_transfer_supply()
            return
        # @fix: 不要清空已流转的地块列表，而是从地块状态中恢复
        # 只清空本期的报价列表，保留已流转的地块状态
        self.leased_out_patches = [p for p in self.land_patches 
                                   if p.owner == self and p.leased_to is not None]
        self.leased_in_patches = []  # 租入地块需要重新计算
        self.leased_patches = []
        self.reservation_prices = {}
        # 先在交易网络内完成熟人撮合，再进入中央市场
        self.run_local_land_transfer_round()
        self._submit_pending_transfer_supply()
        
        # @fix: 只有未流转的地块才能参与市场
        # 已流转的地块（leased_to is not None）不应该再次参与市场
        
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            # Non_Farmer作为纯粹的供给方，只对未流转的地块提交供给意愿
            self.act_as_land_supplier()
        else:
            # 其他农户根据收益比较决定角色，只对未流转的地块参与市场
            self.determine_market_role_and_participate()

    def act_as_land_supplier(self):
        """
        @theory
        Non_Farmer作为土地供给方
        
        @critical 2025-01: 粮食安全战略要求土地不能闲置
        Non_Farmer应该积极出租土地给规模户/企业种粮
        
        @fix: 大幅降低保留面积要求，让更多土地进入流转市场
        @fix 2025-01: 老年农户应该更积极出租土地
        """
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        elderly_threshold = transfer_cfg.get("elderly_age_threshold", 60)
        base_threshold = float(transfer_cfg.get("profit_threshold_base", 0.4) or 0.4)
        rural_nonfarmer_profit_threshold = float(
            transfer_cfg.get("rural_nonfarmer_profit_threshold", max(0.05, base_threshold - 0.10))
            or max(0.05, base_threshold - 0.10)
        )
        urbanized_min_retained_area_per_capita = float(
            transfer_cfg.get("urbanized_non_farmer_min_retained_area_per_capita", 0.10) or 0.10
        )
        rural_min_retained_area_per_capita = float(
            transfer_cfg.get("non_farmer_min_retained_area_per_capita", 0.28) or 0.28
        )

        farmer_type_name = str(
            getattr(getattr(self, 'farmer_type', None), 'name', getattr(self, 'farmer_type', ''))
        ).upper()
        aggressive_release = bool(getattr(self, 'cohort_urbanized', False)) or 'ABSENTEE_LANDLORD' in farmer_type_name
        if aggressive_release:
            min_retained_area = self.household_size * urbanized_min_retained_area_per_capita
        else:
            min_retained_area = self.household_size * rural_min_retained_area_per_capita

        # @fix 2025-01: 老年农户应该更积极出租土地！
        is_elderly = self.age >= elderly_threshold
        if aggressive_release:
            rent_discount = 0.6 if is_elderly else 0.8
        else:
            rent_discount = 0.9 if is_elderly else 1.0
        
        # 计算当前自留土地面积（未流转的地块）
        remaining_retained_area = sum(
            patch.area for patch in self.land_patches 
            if patch.owner == self and patch.leased_to is None
        )
        
        # 跟踪供给地块数量（用于调试）
        supply_count = 0
        
        for patch in self.land_patches:
            # Non_Farmer应该积极出租所有土地
            if patch.owner == self and patch.leased_to is None:
                if aggressive_release:
                    effective_min_retained = min_retained_area * (0.3 if is_elderly else 0.5)
                    if remaining_retained_area - patch.area < effective_min_retained:
                        # 显性城镇化/缺席地主允许以更高保留价出租最后一块地。
                        pass
                else:
                    if remaining_retained_area - patch.area < min_retained_area:
                        continue
                    self_cultivation_profit = self.calculate_self_cultivation_profit(patch)
                    rental_plus_wage_income = self.calculate_rental_plus_wage_income(patch)
                    if rental_plus_wage_income < self_cultivation_profit * (1 - rural_nonfarmer_profit_threshold):
                        continue
                
                min_acceptable_rent = self.calculate_min_acceptable_rent(patch)
                
                # 显性城镇化/缺席地主更接近“离土不离权”出租人，农村非农户则保留更强的土地黏性。
                min_acceptable_rent = min_acceptable_rent * rent_discount
                
                # @critical: 设置保留价格下限和上限，确保价格在合理范围
                # 实证：全国平均土地租金280元/亩，范围150-500元
                min_rent_per_mu = 100  # 最低100元/亩（保障农户基本权益）
                max_rent_per_mu = 400  # 最高400元/亩（避免价格过高阻碍流转）
                
                rent_per_mu = min_acceptable_rent / max(patch.area, 0.01)
                rent_per_mu = max(min_rent_per_mu, min(max_rent_per_mu, rent_per_mu))
                min_acceptable_rent = rent_per_mu * patch.area
                
                self.reservation_prices[patch] = min_acceptable_rent
                
                # 提交供给意愿
                self.model.land_clearing_house.add_supply_offer(
                    patch, self, min_acceptable_rent
                )
                supply_count += 1
                remaining_retained_area = max(
                    0.0,
                    remaining_retained_area - float(getattr(patch, 'area', 0.0) or 0.0),
                )
        
        # 调试信息（可选）
        # if supply_count > 0:
        #     print(f"[供给] {self.unique_id} (Non_Farmer) 提交 {supply_count} 块地出租")

    def determine_market_role_and_participate(self):
        """
        @theory
        其他农户根据收益比较决定市场角色
        @fix: 增加更多农户愿意出租土地的条件，提高土地供给
        @fix: 只对未流转的地块参与市场
        @fix 2024-11-19: 添加最低土地保留机制，确保农户保留少量土地用于自给自足
        @fix 2025-01: 修正老年农户逻辑 - 老年农户作为供给方而非被排除
        """
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        min_retained_area_per_capita = float(
            transfer_cfg.get("min_retained_area_per_capita", 0.3) or 0.3
        )
        # @fix 2024-11-19: 最低保留面积必须和配置口径一致，不能再硬编码0.5亩/人
        min_retained_area = self.household_size * min_retained_area_per_capita

        # 计算当前自留土地面积（未流转的地块）
        remaining_retained_area = sum(
            patch.area for patch in self.land_patches 
            if patch.owner == self and patch.leased_to is None
        )

        threshold_in = transfer_cfg.get("threshold_in", 0.6)
        threshold_out = transfer_cfg.get("threshold_out", 0.2)
        elderly_threshold = transfer_cfg.get("elderly_age_threshold", 60)
        current_operated_area = self.get_operated_land_area()
        target_area = getattr(self, "target_operated_area", self._compute_target_operated_area())
        willingness_in = self.get_transfer_in_willingness()
        willingness_out = self.get_transfer_out_willingness()
        recent_net_income = self._get_recent_net_income_signal()
        base_threshold = float(transfer_cfg.get("profit_threshold_base", 0.4) or 0.4)
        global_profit_threshold = base_threshold
        strong_absorber = (
            self._get_professional_operator_signal(current_operated_area) >= 0.55
            or self._get_service_readiness_signal() >= 0.65
        )
        if current_operated_area < target_area and willingness_in >= threshold_in:
            global_profit_threshold = max(0.1, base_threshold - 0.1)
        elif current_operated_area > target_area and willingness_out >= threshold_out:
            global_profit_threshold = min(0.8, base_threshold + 0.1)
        
        # @fix 2025-01: 老年农户应该倾向于出租土地，而不是被排除
        is_elderly = self.age >= elderly_threshold
        elderly_release_path = is_elderly and self._qualifies_for_elderly_land_release()
        
        for patch in self.land_patches:
            # @fix: 只对未流转的地块（leased_to is None）参与市场
            if patch.owner == self and patch.leased_to is None:
                if patch in self.reservation_prices:
                    continue
                # @fix 2025-01: 老年农户直接作为供给方，不需要计算收益比较
                if elderly_release_path:
                    # 老年农户降低保留面积要求（0.5→0.25），更愿意流转
                    effective_min_retained = min_retained_area * 0.5
                    if remaining_retained_area - patch.area < effective_min_retained:
                        continue
                    if self.act_as_land_supplier_for_patch(patch, rent_discount=0.7):
                        remaining_retained_area = max(
                            0.0,
                            remaining_retained_area - float(getattr(patch, 'area', 0.0) or 0.0),
                        )
                    continue
                
                # @fix 2024-11-19: 如果流转这块地后，剩余土地低于最低保留面积，不允许流转
                if remaining_retained_area - patch.area < min_retained_area:
                    continue  # 跳过这块地，保留用于自给自足
                # 计算自耕的预期收益
                self_cultivation_profit = self.calculate_self_cultivation_profit(patch)
                
                # 计算出租+务工的预期收益
                rental_plus_wage_income = self.calculate_rental_plus_wage_income(patch)
                
                # @fix: 增加一个"接近"的判断，如果出租收益接近自耕收益（60%以上），也选择出租
                # 这样可以增加土地供给，提高流转率
                profit_threshold = global_profit_threshold  # 调整：从0.3提高到0.4，如果出租收益 >= 自耕收益的60%，也选择出租
                
                if rental_plus_wage_income >= self_cultivation_profit * (1 - profit_threshold):
                    # 选择出租（出租收益接近或高于自耕收益）
                    if self.act_as_land_supplier_for_patch(patch):
                        remaining_retained_area = max(
                            0.0,
                            remaining_retained_area - float(getattr(patch, 'area', 0.0) or 0.0),
                        )
                else:
                    # 选择自耕，保留土地
                    pass
        
        # @new 2025-01: Pure_Farmer和部分Pluriactive主动寻找可租入土地
        # @fix 2026-03: 给扩张加入盈利和现金流刹车，避免“亏损还扩张”
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            if (
                recent_net_income > (2000 if strong_absorber else 3000)
                and current_operated_area < target_area
                and willingness_in >= max(0.55, threshold_in * 0.9)
            ):
                self._seek_land_to_rent_in(target_area, current_operated_area)
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业户：只有在经营扩张仍有收益时才参与
            if recent_net_income > (6000 if strong_absorber else 8000) and willingness_in >= max(0.60, threshold_in):
                self._seek_land_to_rent_in(target_area, current_operated_area)

    def calculate_min_acceptable_rent(self, patch):
        """
        @theory 保留价 = 机会成本（基于经济学定义）
        参考实证数据：2022年全国平均土地成本280元/亩

        机会成本 = 自耕净收益

        关键：农户流转土地后会失去补贴（补贴给了实际种地者），
        所以保留价必须包含补贴损失！

        注：不是两者相加！机会成本是放弃的最佳选择
        """
        # calculate_self_cultivation_profit() 已返回包含补贴的完整净收益，
        # 这里不再重复叠加补贴，避免系统性抬高保留价。
        self_cultivation_profit_per_mu = self.calculate_self_cultivation_profit(patch) / max(patch.area, 0.01)

        # 4. 保留价 = 自耕总收益（土地的机会成本）
        # 保留价只反映这块地继续自种能获得多少，不再混入其他路径收益。
        opportunity_cost_per_mu = self_cultivation_profit_per_mu
        
        # 4. 调整：参考市场均衡租金（280元/亩基准）
        # 实证显示：农户保留价通常在200-400元之间
        market_benchmark = 280  # 元/亩
        
        # 如果计算的机会成本过高或过低，向市场基准调整
        if opportunity_cost_per_mu > 500:
            # 过高时适当降低（农户需要流转土地获得现金收入）
            reservation_price_per_mu = opportunity_cost_per_mu * 0.7 + market_benchmark * 0.3
        elif opportunity_cost_per_mu < 150:
            # 过低时适当提高（维持基本权益）
            reservation_price_per_mu = max(150, opportunity_cost_per_mu)
        else:
            reservation_price_per_mu = opportunity_cost_per_mu
        
        # 5. 最终保留价（总额）
        return reservation_price_per_mu * patch.area

    def calculate_self_cultivation_profit(self, patch):
        """
        @theory
        计算自耕的预期利润
        """
        if self.planned_crop == CropType.GRAIN:
            # @fix: 决策时正确计入补贴
            return self.calculate_grain_expected_profit(
                self.model.environment.get_market_price('crop_price'),
                self.model.environment.get_policy_param('grain_subsidies_rate'),
                patch.area
            )
        elif self.planned_crop == CropType.CASH_CROP:
            # @fix: 使用 environment 获取动态市场价格
            return self.calculate_cash_crop_expected_profit(
                self.model.environment.get_market_price('cash_crop'),
                patch.area
            )
        else:
            return 0

    def calculate_rental_plus_wage_income(self, patch):
        """
        @theory
        计算出租+务工的预期收益
        """
        rental_price_per_mu = self._get_average_rental_price()
        quality_multiplier = max(0.6, min(1.4, float(getattr(patch, 'soil_quality', 1.0) or 1.0)))
        rental_income = float(getattr(patch, 'area', 0.0) or 0.0) * rental_price_per_mu * quality_multiplier

        production_cfg = self.model.config.get('production', {}) if hasattr(self.model, 'config') else {}
        input_requirements = production_cfg.get('input_requirements', {})
        labor_days_cfg = input_requirements.get('labor_days', {})
        if self.planned_crop == CropType.CASH_CROP:
            labor_days_per_mu = labor_days_cfg.get('cash_crop_farmer', labor_days_cfg.get('cash_crop', 25))
        else:
            labor_days_per_mu = labor_days_cfg.get('grain_farmer', labor_days_cfg.get('grain', 10))
        released_labor_days = float(getattr(patch, 'area', 0.0) or 0.0) * float(labor_days_per_mu)
        wage_income = self._estimate_incremental_off_farm_income(released_labor_days)

        return rental_income + wage_income

    def act_as_land_demander(self, patch):
        """
        @theory
        作为土地需求方（规模户/Pure_Farmer扩张经营）
        
        @new 2025-01: 实现农户间土地流转
        Pure_Farmer和部分Pluriactive农户可以租入土地扩大经营规模
        """
        # 检查是否有扩张意愿
        current_area = self.get_operated_land_area()
        target_area = getattr(self, 'target_operated_area', self._compute_target_operated_area())
        
        # 只有当前面积小于目标面积时才有租入意愿
        if current_area >= target_area:
            return
        
        # 检查资金能力（需要有足够资金支付租金）
        available_capital = self.capital * 0.3  # 最多用30%资金支付租金
        
        # 计算最高愿付租金
        max_rent = self.calculate_max_willing_to_pay_rent(patch)
        
        if max_rent <= 0:
            return
        
        # 考虑资金约束
        max_affordable_rent = available_capital / max(1, (target_area - current_area) / patch.area)
        bid_price = min(max_rent, max_affordable_rent)
        
        if bid_price > 0:
            # 提交需求报价
            self.model.land_clearing_house.add_demand_offer(
                patch=patch,
                demander=self,
                bid_price=bid_price
            )

    def _calculate_reservation_profit_rate(self):
        """
        @new 计算农户保留利润率（内生决定，不是外生参数）
        
        @theory 保留利润率由农户特征决定：
        1. 风险偏好：风险厌恶型保留更多利润（高保留率）
        2. 资金约束：资金紧张时愿意出更高租金获取土地（低保留率）
        3. 年龄：年轻农户更激进，愿意支付更高租金（低保留率）
        4. 经营能力：高能力农户更自信，愿意出更高租金（低保留率）
        
        @return float 保留利润率 [0.15, 0.35]
        """
        # 基础保留率：25%（农户至少要保留1/4利润作为激励）
        base_rate = 0.25
        
        # 1. 风险偏好影响 [-0.05, +0.05]
        risk_adjustment = (getattr(self, 'risk_aversion', 0.5) - 0.5) * 0.10
        
        # 2. 资金约束影响 [-0.08, 0]
        # 资金越紧张，越愿意降低保留利润率换取土地
        capital_ratio = self.capital / max(1, self.get_operated_land_area() * 1000)
        if capital_ratio < 0.5:  # 资金紧张
            capital_adjustment = -0.08 * (0.5 - capital_ratio) * 2  # 最多降低8%
        else:
            capital_adjustment = 0
        
        # 3. 年龄影响 [-0.05, +0.05]
        # 年轻农户（<50岁）更激进，年老农户（>60岁）更保守
        age = getattr(self, 'age', 50)
        if age < 50:
            age_adjustment = -0.05 * (50 - age) / 20  # 最多降低5%
        elif age > 60:
            age_adjustment = 0.05 * (age - 60) / 20   # 最多增加5%
        else:
            age_adjustment = 0
        
        # 4. 经营能力影响 [-0.05, +0.02]
        ability_adjustment = (self.farming_ability - 1.0) * 0.05
        
        # 综合计算
        reservation_rate = base_rate + risk_adjustment + capital_adjustment + age_adjustment + ability_adjustment
        
        # 限制在合理范围[15%, 35%]
        reservation_rate = max(0.15, min(0.35, reservation_rate))
        
        return reservation_rate

    def calculate_max_willing_to_pay_rent(self, patch):
        """
        @theory
        计算最高愿付租金（基于边际利润）
        
        @fix 2025-12-18: 补贴-租金联动的涌现机制
        关键逻辑：补贴↑ → 种粮利润↑ → 农户愿付租金↑ → 市场租金↑
        
        理论依据：
        1. 补贴资本化理论 (Lence & Mishra 2003): 政府补贴会被土地所有者俘获
        2. 租金支付意愿 = 预期边际利润 (收入 + 补贴 - 成本)
        3. 市场均衡租金由供需双方报价决定，补贴提高需求方出价
        
        实证基础：
        - 黄季焜等(2015): 中国粮食补贴约30-40%转化为租金
        - Ciaian & Kancs(2012): EU直接支付20-30%资本化为地租
        """
        patch_area = max(0.0, float(getattr(patch, 'area', 0.0) or 0.0))
        if patch_area <= 0:
            return 0.0

        # 在新增土地上种植 planned_crop 的整块预期边际利润
        if self.planned_crop == CropType.GRAIN:
            from core.p3_project_contracts import configured_program
            expected_subsidy_income = None
            if configured_program(self.model) is not None:
                current_area = self.get_operated_land_area()
                already_operated = (
                    patch.current_operator is self or patch.leased_to is self
                    or (patch.owner is self and patch.leased_to is None
                        and patch.current_operator is None)
                )
                without_area = max(0.0, current_area - patch_area) if already_operated else current_area
                retained = self.expected_fixed_rule_account("maintain_grain", without_area + patch_area)
                released = self.expected_fixed_rule_account("maintain_grain", without_area)
                expected_subsidy_income = retained.support_yuan - released.support_yuan
            marginal_profit = self.calculate_grain_expected_profit(
                self.model.environment.get_market_price('crop_price'),
                self.model.environment.get_policy_param('grain_subsidies_rate'),
                patch_area,
                expected_subsidy_income=expected_subsidy_income,
            )
        elif self.planned_crop == CropType.CASH_CROP:
            cash_price = self.model.environment.get_market_price('cash_crop')
            marginal_profit = self.calculate_cash_crop_expected_profit(cash_price, patch_area)
        else:
            marginal_profit = 0.0
        
        # @key 涌现式租金决定机制：
        # 1. 农户支付意愿 = marginal_profit（扣除保留利润）
        # 2. 保留利润率由农户个体特征决定（风险偏好、资金约束）
        # 3. 实际租金由土地清算所供需匹配决定，不是农户单方面决定
        
        # 保留利润率（农户想保留多少利润作为激励）
        # 由农户特征内生决定，而非外生参数
        reservation_profit_rate = self._calculate_reservation_profit_rate()
        recent_net_income = self._get_recent_net_income_signal()
        risk_buffer = 20.0 * patch_area
        if recent_net_income < 0:
            risk_buffer += 100.0 * patch_area
        elif recent_net_income < 8000:
            risk_buffer += 40.0 * patch_area

        # 最高愿付租金 = 边际利润 - 保留利润
        max_rent = max(0.0, (marginal_profit - risk_buffer) * (1 - reservation_profit_rate))
        
        return max_rent

    def act_as_land_supplier_for_patch(self, patch, rent_discount=1.0):
        """
        @theory
        作为土地供给方（针对特定地块）
        @param rent_discount: 租金折扣系数，用于老年农户降价（默认1.0无折扣）
        """
        if patch is None:
            return False
        if patch in self.reservation_prices:
            return False
        min_acceptable_rent = self.calculate_min_acceptable_rent(patch)
        # 应用折扣
        min_acceptable_rent = min_acceptable_rent * rent_discount
        self.reservation_prices[patch] = min_acceptable_rent
        
        # 提交供给意愿
        self.model.land_clearing_house.add_supply_offer(
            patch, self, min_acceptable_rent
        )
        return True

    def _seek_land_to_rent_in(self, target_area, current_area):
        """
        @new 2025-01: 农户主动寻找可租入的土地
        
        实现农户间土地流转：
        - Pure_Farmer 扩大经营规模
        - Pluriactive 有扩张意愿时也可租入
        
        @theory
        参考：《中国农村经济形势分析与预测》
        规模户（经营50亩以上）约占15%，是土地流转的主要需求方
        """
        # 计算需要租入的面积，并施加年度扩张上限
        expansion_cap = self._get_expansion_area_cap(current_area)
        need_area = min(target_area - current_area, expansion_cap)
        if need_area <= 0:
            return

        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        # 检查资金能力
        available_capital_share = float(transfer_cfg.get("rent_in_capital_share", 0.36) or 0.36)
        available_capital = self.capital * max(0.0, min(0.8, available_capital_share))

        # 优先使用本期已进入清算所的真实供给地块，避免把搜索预算浪费在不可交易地块上
        available_patches = []
        clearing_house = getattr(self.model, 'land_clearing_house', None)
        if clearing_house is not None:
            seen_patch_ids = set()
            for offer in getattr(clearing_house, 'supply_offers', []) or []:
                patch = offer.get('patch') if isinstance(offer, dict) else None
                if patch is None or patch.owner == self or patch.leased_to is not None:
                    continue
                patch_id = getattr(patch, 'patch_id', None) or id(patch)
                if patch_id in seen_patch_ids:
                    continue
                seen_patch_ids.add(patch_id)
                available_patches.append(patch)
        if not available_patches:
            available_patches = self.model.get_available_patches()
        if not available_patches:
            return
        
        # 优先选择距离近的地块（空间连片）
        if hasattr(self, 'pos') and self.pos is not None:
            def distance_to_self(patch):
                if hasattr(patch, 'pos') and patch.pos is not None:
                    return abs(patch.pos[0] - self.pos[0]) + abs(patch.pos[1] - self.pos[1])
                return float('inf')
            available_patches = sorted(available_patches, key=distance_to_self)
        
        # 遍历可用地块，提交需求报价
        rented_area = 0
        max_patches = 15  # @fix 2025-11-29: 从5增加到15，增加农户可竞争的地块数量
        avg_rent_per_mu = self._get_average_rental_price()
        
        for patch in available_patches[:max_patches]:
            if rented_area >= need_area:
                break
            
            if not self._is_profitable_rent_in_opportunity(patch):
                continue

            # 计算最高愿付租金
            max_rent = self.calculate_max_willing_to_pay_rent(patch)
            if max_rent <= 0:
                continue
            
            # 考虑资金约束
            remaining_capital = available_capital - (rented_area * avg_rent_per_mu)
            if remaining_capital <= 0:
                break
            
            max_affordable = remaining_capital
            bid_price = min(max_rent, max_affordable)

            min_bid_per_mu = float(transfer_cfg.get("rent_in_min_bid_per_mu", 90.0) or 90.0)
            if bid_price > min_bid_per_mu * patch.area and self._passes_investment_sunk_cost_gate(patch, bid_price):
                self.model.land_clearing_house.add_demand_offer(
                    patch=patch,
                    demander=self,
                    bid_price=bid_price
                )
                rented_area += patch.area

    def allocate_labor(self):
        """
        @theory
        劳动力分配决策
        @improvement: P0优先级 - 实现内生劳动力分配
        参考：三部门模型（农业-农村非农-城市非农）
        基于经济信号动态调整，而非固定比例
        """
        # 检查是否启用内生分配
        labor_config = self.model.config.get('agents', {}).get('farmer', {}).get('labor_allocation', {})
        use_endogenous = labor_config.get('use_endogenous_allocation', True)
        
        if not use_endogenous:
            # 使用固定比例（向后兼容）
            self._allocate_labor_fixed()
            return
        
        # 内生劳动力分配：基于经济信号动态调整
        # 1. 获取市场信号
        off_farm_wage = self.model.environment.get_market_price('wage_rate')
        avg_rental_price = self._get_average_rental_price()
        
        # 2. 计算不同活动的预期收益
        agri_expected_income = self._calculate_expected_agricultural_income_per_labor_unit()
        off_farm_expected_income = self.off_farm_wage_level * off_farm_wage
        rental_expected_income = self._calculate_expected_rental_income_per_labor_unit()
        
        # 3. 计算收入方差（用于风险调整）
        agri_income_variance = self._estimate_agricultural_income_variance()
        off_farm_income_variance = 0.1  # 非农收入相对稳定，方差较小
        
        # 4. 基于相对收益和风险偏好进行分配（使用效用函数）
        # 效用 = 预期收益 - 风险厌恶系数 × 收益方差
        risk_weight = labor_config.get('elasticity', {}).get('risk_aversion_weight', 0.1)
        agri_utility = agri_expected_income - self.risk_aversion * risk_weight * agri_income_variance
        off_farm_utility = off_farm_expected_income - self.risk_aversion * risk_weight * off_farm_income_variance
        
        # 5. 考虑租金收入的影响（如果出租土地，可以释放更多劳动力）
        if rental_expected_income > 0:
            # 租金收入增加了非农劳动的吸引力
            off_farm_utility += rental_expected_income * labor_config.get('elasticity', {}).get('rental_price_elasticity', 0.2)
        
        # 6. 计算最优分配比例（使用Logit模型）
        # @fix: 使用数值稳定的softmax计算，避免overflow
        import numpy as np
        
        # 归一化以避免overflow
        max_utility = max(agri_utility, off_farm_utility)
        if max_utility > 100:  # 避免极端值
            agri_utility = agri_utility - max_utility + 100
            off_farm_utility = off_farm_utility - max_utility + 100
        
        # 使用数值稳定的softmax
        try:
            exp_agri = np.exp(np.clip(agri_utility, -500, 500))
            exp_off = np.exp(np.clip(off_farm_utility, -500, 500))
            total_exp = exp_agri + exp_off
            
            if total_exp > 0 and not np.isnan(total_exp) and not np.isinf(total_exp):
                agri_ratio = exp_agri / total_exp
                off_farm_ratio = 1.0 - agri_ratio
            else:
                # 如果计算失败，使用默认分配
                agri_ratio, off_farm_ratio = self._get_default_labor_allocation()
        except (RuntimeWarning, FloatingPointError):
            # 如果发生数值错误，使用默认分配
            agri_ratio, off_farm_ratio = self._get_default_labor_allocation()
        
        # 7. 应用工资弹性调整
        wage_elasticity = labor_config.get('elasticity', {}).get('wage_elasticity', 0.3)
        base_wage = self._get_base_non_farm_wage()
        wage_ratio = off_farm_wage / base_wage if base_wage > 0 else 1.0
        wage_adjustment = (wage_ratio - 1.0) * wage_elasticity
        off_farm_ratio = max(0.0, min(1.0, off_farm_ratio + wage_adjustment))
        agri_ratio = 1.0 - off_farm_ratio
        
        # 8. 应用调整速度（避免过度波动）
        adjustment_speed = labor_config.get('adjustment_speed', 0.2)
        if hasattr(self, '_last_agri_ratio'):
            agri_ratio = adjustment_speed * agri_ratio + (1 - adjustment_speed) * self._last_agri_ratio
            off_farm_ratio = 1.0 - agri_ratio
        self._last_agri_ratio = agri_ratio
        
        # 9. 应用转换成本（如果变化太大）
        switching_cost = labor_config.get('switching_cost', 0.05)
        if hasattr(self, '_last_agri_ratio'):
            ratio_change = abs(agri_ratio - self._last_agri_ratio)
            if ratio_change > switching_cost:
                # 限制变化幅度
                if agri_ratio > self._last_agri_ratio:
                    agri_ratio = min(agri_ratio, self._last_agri_ratio + switching_cost)
                else:
                    agri_ratio = max(agri_ratio, self._last_agri_ratio - switching_cost)
                off_farm_ratio = 1.0 - agri_ratio
        
        # 10. 确保比例在合理范围内（基于生计策略）
        min_agri, max_agri = self._get_labor_allocation_bounds()
        agri_ratio = max(min_agri, min(max_agri, agri_ratio))
        off_farm_ratio = 1.0 - agri_ratio
        
        # 11. 分配劳动力
        self.agricultural_labor = self.total_labor_units * agri_ratio
        self.off_farm_labor = self.total_labor_units * off_farm_ratio
    
    def _allocate_labor_fixed(self):
        """固定比例分配（向后兼容）"""
        labor_config = self.model.config.get('agents', {}).get('farmer', {}).get('labor_allocation', {})
        defaults = labor_config.get('default_by_strategy', {})
        
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # @fix: 调整Pure_Farmer的默认分配，从70%/30%改为60%/40%
            # 确保有足够的非农收入，防止正反馈循环
            default = defaults.get('PURE_FARMER', {'agricultural': 0.6, 'off_farm': 0.4})
            self.agricultural_labor = self.total_labor_units * default['agricultural']
            self.off_farm_labor = self.total_labor_units * default['off_farm']
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            default = defaults.get('PLURIACTIVE', {'agricultural': 0.4, 'off_farm': 0.6})
            self.agricultural_labor = self.total_labor_units * default['agricultural']
            self.off_farm_labor = self.total_labor_units * default['off_farm']
        else:  # NON_FARMER
            # @fix: Non_Farmer应该接近0%农业劳动，100%非农劳动
            # 因为Non_Farmer把土地都流转了，不从事农业生产
            default = defaults.get('NON_FARMER', {'agricultural': 0.01, 'off_farm': 0.99})  # 从5%降到1%
            self.agricultural_labor = self.total_labor_units * default['agricultural']
            self.off_farm_labor = self.total_labor_units * default['off_farm']
    
    def _get_default_labor_allocation(self):
        """获取默认劳动力分配比例"""
        labor_config = self.model.config.get('agents', {}).get('farmer', {}).get('labor_allocation', {})
        defaults = labor_config.get('default_by_strategy', {})
        
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # @fix: 调整Pure_Farmer的默认分配，从70%/30%改为60%/40%
            # 确保有足够的非农收入，防止正反馈循环
            default = defaults.get('PURE_FARMER', {'agricultural': 0.6, 'off_farm': 0.4})
            return default['agricultural'], default['off_farm']
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            default = defaults.get('PLURIACTIVE', {'agricultural': 0.4, 'off_farm': 0.6})
            return default['agricultural'], default['off_farm']
        else:  # NON_FARMER
            default = defaults.get('NON_FARMER', {'agricultural': 0.05, 'off_farm': 0.95})
            return default['agricultural'], default['off_farm']
    
    def _get_labor_allocation_bounds(self):
        """获取劳动力分配的合理范围（基于生计策略和自耕土地面积）"""
        # @fix 2024-11-18: 根据自耕土地面积动态调整非农劳动时间上限
        # 理论依据：兼业农户既从事农业又外出务工，自耕土地越多，农忙期越长，可用非农劳动时间越少
        # 参考：Schmitzberger et al. (2005), "How farming styles influence biodiversity in Austrian agricultural landscapes"
        
        # 计算自耕土地面积（排除流转出去的）
        self_operated_area = self.get_operated_land_area()
        
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户：几乎全年务农，农忙期（春耕+秋收）约150天，农闲期也需田间管理
            # 可用非农劳动时间：农闲期约100-150天
            # 农业劳动占比：60%-85%（非农劳动15%-40%）
            return 0.6, 0.85
        
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业农户：根据自耕土地面积动态调整
            # 假设每亩地需要约15天农忙劳动（春耕5天+田间管理5天+秋收5天）
            agri_days_per_mu = 15
            required_agri_days = self_operated_area * agri_days_per_mu
            
            # 农业劳动下限：满足基本农业生产需求
            min_agri_ratio = min(0.70, required_agri_days / self.total_labor_units)
            
            # 农业劳动上限：考虑农闲期也可以部分务工
            # 如果自耕土地多，农业劳动占比高；如果流转了部分土地，农业劳动占比低
            max_agri_ratio = min(0.75, (required_agri_days + 100) / self.total_labor_units)
            
            # 确保合理范围
            min_agri_ratio = max(0.20, min(0.70, min_agri_ratio))
            max_agri_ratio = max(min_agri_ratio + 0.05, min(0.75, max_agri_ratio))
            
            return min_agri_ratio, max_agri_ratio
        
        else:  # NON_FARMER
            # 非农户：流转全部土地或只保留少量自给地，几乎不从事农业生产
            # 可全年外出务工（250-300天）
            # 农业劳动占比：0%-5%（自给自足）
            return 0.0, 0.05
    
    def _calculate_expected_agricultural_income_per_labor_unit(self):
        """计算单位农业劳动的预期收入"""
        operated_area = self.get_operated_land_area()
        if operated_area == 0:
            return 0.0
        
        # 计算预期农业收入
        if self.planned_crop == CropType.GRAIN:
            expected_profit = self.calculate_grain_expected_profit(
                self.model.environment.get_market_price('crop_price'),
                self.model.environment.get_policy_param('grain_subsidies_rate'),
                operated_area
            )
        elif self.planned_crop == CropType.CASH_CROP:
            expected_profit = self.calculate_cash_crop_expected_profit(
                self.model.environment.get_market_price('cash_crop'),
                operated_area
            )
        else:
            return 0.0
        
        # @fix: 修正单位农业劳动收入计算
        # 估算所需农业劳动（基于面积，假设每10亩需要1个劳动单位，但应该用实际分配的农业劳动）
        # 使用总劳动单位来估算，而不是只基于面积
        total_labor = self.total_labor_units if hasattr(self, 'total_labor_units') else 365.0
        # 假设农业劳动至少占总劳动的30%（基于生计策略）
        estimated_agri_labor = max(1.0, total_labor * 0.3)
        
        # 单位劳动收入 = 总利润 / 估算农业劳动天数
        return expected_profit / estimated_agri_labor
    
    def _calculate_expected_rental_income_per_labor_unit(self):
        """计算单位劳动的预期租金收入（如果出租土地）"""
        # 计算如果出租所有土地能获得的租金收入
        total_rental_income = 0.0
        for patch in self.land_patches:
            if patch.owner == self and patch.leased_to is None:
                # 估算租金收入
                base_rental = self._get_average_rental_price()
                quality_multiplier = self.model.config.get('environment', {}).get('market', {}).get('quality_multiplier', 1.4)
                rental_price = base_rental * (0.5 + 0.5 * patch.soil_quality) * quality_multiplier
                total_rental_income += rental_price * patch.area
        
        # 如果出租土地，释放的劳动力可以用于非农工作
        # 假设每出租1亩土地，可以释放约10天的农业劳动
        labor_released = total_rental_income / 100.0  # 简化计算
        return total_rental_income / max(1.0, labor_released) if labor_released > 0 else 0.0
    
    def _estimate_agricultural_income_variance(self):
        """估算农业收入的方差（用于风险调整）"""
        # 基于历史数据或假设
        # 农业收入波动较大，假设方差为预期收入的30%
        operated_area = self.get_operated_land_area()
        if operated_area == 0:
            return 0.0
        
        if self.planned_crop == CropType.GRAIN:
            expected_profit = self.calculate_grain_expected_profit(
                self.model.environment.get_market_price('crop_price'),
                self.model.environment.get_policy_param('grain_subsidies_rate'),
                operated_area
            )
        elif self.planned_crop == CropType.CASH_CROP:
            expected_profit = self.calculate_cash_crop_expected_profit(
                self.model.environment.get_market_price('cash_crop'),
                operated_area
            )
        else:
            return 0.0
        
        return expected_profit * 0.3  # 假设方差为30%
    
    def _get_average_rental_price(self):
        """获取平均土地租金价格"""
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_market_price'):
            market_rent = env.get_market_price('land_rent')
            if market_rent is not None and market_rent > 0:
                return float(market_rent)
        if env is not None and hasattr(env, 'land_rental_price'):
            if env.land_rental_price is not None and env.land_rental_price > 0:
                return float(env.land_rental_price)
        return float(
            self.model.config.get('environment', {}).get('market', {}).get('base_land_rent', 280.0)
        )

    def _get_output_price_per_kg(self, item_name):
        """获取作物销售价格（元/kg）。环境内部价格口径是元/斤。"""
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_market_price_per_kg'):
            price = env.get_market_price_per_kg(item_name)
            if price is not None:
                return float(price)
        if env is not None and hasattr(env, 'get_market_price'):
            price = env.get_market_price(item_name)
            if price is not None:
                return float(price) * JIN_PER_KG
        return 0.0

    def _get_output_price_statistics_per_kg(self, crop_type, periods=5):
        """获取作物价格统计（元/kg），兼容环境侧旧的元/斤接口。"""
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_price_statistics_per_kg'):
            stats = env.get_price_statistics_per_kg(crop_type, periods=periods)
            if stats:
                return stats

        if env is not None and hasattr(env, 'get_price_statistics'):
            legacy_stats = env.get_price_statistics(crop_type, periods=periods)
            return {
                'mean': float(legacy_stats['mean']) * JIN_PER_KG,
                'std': float(legacy_stats['std']) * JIN_PER_KG,
                'min': float(legacy_stats['min']) * JIN_PER_KG,
                'max': float(legacy_stats['max']) * JIN_PER_KG,
                'history': [float(item) * JIN_PER_KG for item in legacy_stats.get('history', [])],
            }

        fallback_price = self._get_output_price_per_kg('crop_price' if crop_type == 'grain' else 'cash_crop')
        fallback_std = fallback_price * (0.1 if crop_type == 'grain' else 0.2)
        return {
            'mean': fallback_price,
            'std': fallback_std,
            'min': fallback_price * 0.9,
            'max': fallback_price * 1.1,
            'history': [fallback_price],
        }

    def make_production_decisions(self):
        """
        @theory
        生产决策
        @improvement: P1优先级 - 产量计算精细化
        考虑土壤质量、耕作能力、技术水平、气候冲击等多因素
        参考：APSIM模型、农业生产效率研究
        """
        operated_area = production_area(self)
        self.current_production_area = 0.0
        self.grain_area = 0.0
        self.cash_crop_area = 0.0
        new_account = action_income_enabled(self.model.config)
        if new_account and getattr(self, "fixed_rule_exit_pending", False):
            self.expected_yield = 0.0
            return
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER and not new_account:
            if not self._allows_non_farmer_retained_land_production(operated_area):
                # Non_Farmer without retained-plot or service-management support
                # has transferred out or idled operational agriculture.
                self.expected_yield = 0
                return

        if not new_account:
            self._apply_final_crop_profitability_guard(
                operated_area,
                phase="production_after_land_and_service_market",
            )

        if self.planned_crop and operated_area > 0:
            self.current_production_area = max(0.0, float(operated_area or 0.0))
            if self.planned_crop == CropType.MIXED:
                share = self.household_planned_grain_share
                fulfillment = getattr(self, "fixed_rule_labor_fulfillment", 1.0)
                self.household_expected_grain_kg = self._calculate_detailed_yield(
                    CropType.GRAIN, operated_area, area_share=share) * fulfillment
                shocks = self.last_yield_component_trace
                cash = self._calculate_detailed_yield(CropType.CASH_CROP, operated_area, area_share=1-share,
                    shared_shocks=(shocks["climate_shock"], shocks["random_factor"]))
                self.expected_yield = self.household_expected_grain_kg + cash * fulfillment
                return
            # 使用精细化产量计算
            self.expected_yield = self._calculate_detailed_yield(self.planned_crop, operated_area)
            if new_account:
                self.expected_yield *= getattr(self, "fixed_rule_labor_fulfillment", 1.0)
        else:
            self.expected_yield = 0

    def _allows_non_farmer_retained_land_production(self, operated_area):
        """Whether a Non_Farmer's retained operational land can produce this year.

        This keeps the model's story explicit: non-farm households do not
        generally farm, but small retained plots and service-managed plots can
        still generate low-labor grain production instead of becoming fake idle
        land after crop-choice fallback.
        """
        if self.livelihood_strategy != LivelihoodStrategy.NON_FARMER:
            return True
        if self.planned_crop is None or operated_area <= 0:
            return False

        crop_config = self.model.config.get('crop', {}) if hasattr(self, 'model') else {}
        if not bool(crop_config.get("non_farmer_retained_land_grain_fallback_enabled", True)):
            return False

        retained_land_max_area = float(
            crop_config.get("non_farmer_retained_land_max_area_mu", 12.0) or 12.0
        )
        retained_land_tolerance = max(
            0.0,
            float(crop_config.get("non_farmer_retained_land_threshold_tolerance_mu", 0.25) or 0.0),
        )
        if operated_area <= retained_land_max_area + retained_land_tolerance:
            return True

        service_threshold = float(
            crop_config.get("non_farmer_retained_land_service_min_signal", 0.65) or 0.65
        )
        try:
            return self._get_service_readiness_signal() >= service_threshold
        except Exception:
            return False
    
    def _calculate_detailed_yield(self, crop_type, area, *, area_share=1.0, shared_shocks=None):
        """
        @theory
        精细化产量计算：考虑多因素影响
        @improvement: P1优先级
        
        @fix 2025-12-01: 添加技术进步和规模化效率因子
        解决问题：粮食产量下降不符合中国粮食安全战略
        现实：技术进步+规模化使亩产持续提升，总产量增加
        """
        # 获取配置参数
        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        yield_factors = production_config.get('yield_factors', {})
        random_noise = production_config.get('random_noise', {})
        tech_progress = production_config.get('technology_progress', {})
        
        area = max(0.0, float(area or 0.0))
        # 1. 基础产量（按作物类型）
        if crop_type == CropType.GRAIN:
            base_yield_per_mu = base_yields.get('grain', 420)
            if self._national_v1:
                base_yield_per_mu = self._v1_grain_base_yield(base_yield_per_mu)
        else:  # CASH_CROP
            base_yield_per_mu = base_yields.get('cash_crop', 1000)
        
        base_yield = base_yield_per_mu * area * area_share
        raw_base_yield = base_yield
        
        # ================================================================
        # @new 2025-12-01: 技术进步因子
        # 参考：中国粮食亩产年均增长约1.3%（2015-2022）
        # ================================================================
        annual_growth = resolve_crop_annual_yield_growth(tech_progress, crop_type)
        current_step = getattr(self.model, 'step_count', 0)
        tech_progress_factor = (1 + annual_growth) ** current_step
        base_yield = base_yield * tech_progress_factor
        
        # ================================================================
        # @new 2025-12-01: 规模化效率加成
        # 参考：规模户亩产比小农高10-20%
        # ================================================================
        scale_config = tech_progress.get('scale_efficiency_bonus', {})
        threshold = scale_config.get('threshold_area', 30)
        if area >= 100:
            scale_factor = scale_config.get('large_scale', 1.20)
        elif area >= threshold:
            scale_factor = scale_config.get('medium_scale', 1.10)
        else:
            scale_factor = scale_config.get('small_scale', 1.0)
        base_yield = base_yield * scale_factor
        
        # 2. 土壤质量影响（非线性）
        avg_soil_quality = self._get_average_soil_quality()
        soil_config = yield_factors.get('soil_quality', {})
        min_effect = soil_config.get('min_effect', 0.5)
        max_effect = soil_config.get('max_effect', 1.0)
        exponent = soil_config.get('exponent', 0.8)
        
        soil_factor = min_effect + (max_effect - min_effect) * (avg_soil_quality ** exponent)
        soil_effect = base_yield * soil_factor
        
        # 3. 耕作能力影响
        ability_config = yield_factors.get('farming_ability', {})
        min_mult = ability_config.get('min_multiplier', 0.8)
        max_mult = ability_config.get('max_multiplier', 1.2)
        ability_multiplier = min_mult + (max_mult - min_mult) * ((self.farming_ability - 0.8) / 0.4)
        if self._national_v1:
            ability_multiplier = 1.0
        ability_effect = soil_effect * ability_multiplier
        
        # 4. 技术水平影响（从地块或环境获取）
        tech_config = yield_factors.get('technology', {})
        base_tech_level = tech_config.get('base_level', 1.0)
        max_improvement = tech_config.get('max_improvement', 0.2)
        
        tech_level = self._get_technology_level()
        tech_multiplier = 1.0 + max_improvement * (tech_level - base_tech_level)
        tech_effect = ability_effect * tech_multiplier

        service_contracts = getattr(self, 'service_contracts', []) or []
        service_signal = max(0.0, min(1.0, float(getattr(self, 'labor_reduction_ratio', 0.0) or 0.0)))
        if isinstance(service_contracts, list) and service_contracts:
            service_signal = max(service_signal, min(1.0, 0.20 + 0.18 * len(service_contracts)))

        mechanization_potential = 0.7
        if hasattr(self, '_get_regional_land_transfer_params'):
            try:
                regional_params = self._get_regional_land_transfer_params()
                mechanization_potential = float(regional_params.get('mechanization_potential', 0.7) or 0.7)
            except Exception:
                mechanization_potential = 0.7
        mechanization_signal = max(0.0, min(1.0, mechanization_potential))

        retention_cfg = (
            self.model.config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        professional_threshold = float(retention_cfg.get('min_operated_area', 30) or 30)
        scale_progress = max(0.0, min(1.0, (area - 10.0) / max(20.0, professional_threshold * 2.0)))
        strategy_name = str(
            getattr(getattr(self, 'livelihood_strategy', None), 'value', getattr(self, 'livelihood_strategy', ''))
        ).lower()
        professional_signal = 0.0
        if area >= professional_threshold:
            professional_signal += 0.55
        if not self._national_v1 and 'pure' in strategy_name:
            professional_signal += 0.25
        if float(getattr(self, 'farming_ability', 1.0) or 1.0) >= 1.05:
            professional_signal += 0.20
        professional_signal = max(0.0, min(1.0, professional_signal))
        structural_cfg = yield_factors.get('structural_efficiency', {}) or {}

        if crop_type == CropType.GRAIN:
            structural_multiplier = (
                1.0
                + float(structural_cfg.get('farmer_grain_scale_mechanization_weight', 0.08) or 0.0)
                * scale_progress
                * mechanization_potential
                + float(structural_cfg.get('farmer_grain_service_weight', 0.06) or 0.0)
                * service_signal
                + float(structural_cfg.get('farmer_grain_service_mechanization_weight', 0.05) or 0.0)
                * service_signal
                * mechanization_potential
                + float(structural_cfg.get('farmer_grain_professional_weight', 0.04) or 0.0)
                * professional_signal
            )
        else:
            structural_multiplier = (
                1.0
                + float(structural_cfg.get('farmer_cash_scale_weight', 0.03) or 0.0)
                * scale_progress
                + float(structural_cfg.get('farmer_cash_service_weight', 0.02) or 0.0)
                * service_signal
                + float(structural_cfg.get('farmer_cash_professional_weight', 0.02) or 0.0)
                * professional_signal
            )
        tech_effect *= structural_multiplier
        
        # 5. 气候冲击
        service_yield_bonus_multiplier = max(
            1.0,
            min(1.18, float(getattr(self, 'service_yield_bonus_multiplier', 1.0) or 1.0)),
        )
        if crop_type == CropType.GRAIN:
            service_bonus_effective_multiplier = service_yield_bonus_multiplier
        else:
            service_bonus_effective_multiplier = 1.0 + 0.5 * (service_yield_bonus_multiplier - 1.0)
        tech_effect *= service_bonus_effective_multiplier

        climate_config = yield_factors.get('climate', {})
        climate_volatility = climate_config.get('volatility', 0.1)
        climate_mean = climate_config.get('mean', 0.0)
        
        climate_shock = self._get_climate_shock(climate_mean, climate_volatility) if shared_shocks is None else shared_shocks[0]
        climate_resilience_cfg = yield_factors.get('climate_resilience', {}) or {}
        resilience_buffer = 0.0
        if climate_resilience_cfg.get('enabled', True) and climate_shock < 0:
            resilience_buffer = (
                float(climate_resilience_cfg.get('service_buffer', 0.0) or 0.0) * service_signal
                + float(climate_resilience_cfg.get('professional_buffer', 0.0) or 0.0) * professional_signal
                + float(climate_resilience_cfg.get('technology_buffer', 0.0) or 0.0)
                * max(0.0, min(1.0, tech_level - base_tech_level))
                + float(climate_resilience_cfg.get('soil_buffer', 0.0) or 0.0) * avg_soil_quality
                + float(climate_resilience_cfg.get('scale_buffer', 0.0) or 0.0) * scale_progress
            )
            resilience_buffer = min(
                float(climate_resilience_cfg.get('max_total_buffer', 0.45) or 0.45),
                max(0.0, resilience_buffer),
            )
            effective_climate_modifier = 1.0 + climate_shock * (1.0 - resilience_buffer)
            minimum_modifier = float(climate_resilience_cfg.get('minimum_yield_modifier', 0.72) or 0.72)
            effective_climate_modifier = max(minimum_modifier, effective_climate_modifier)
        else:
            effective_climate_modifier = 1.0 + climate_shock
        # 6. 随机波动
        noise_std = random_noise.get('std', 0.1)
        if shared_shocks is not None:
            random_factor = shared_shocks[1]
        elif self._national_v1:
            random_factor = paired_yield_noise(self, 'production', 1.0, noise_std)
        else:
            random_factor = float(np_normal(self, 1.0, noise_std))
        final_yield, yield_trace = finalize_yield_stack(
            model=self.model,
            actor_type='farmer',
            crop_type=crop_type,
            area_mu=area * area_share,
            base_yield_per_mu=base_yield_per_mu,
            raw_base_yield_kg=raw_base_yield,
            non_climate_yield_kg=tech_effect,
            tech_progress_factor=tech_progress_factor,
            scale_factor=scale_factor,
            soil_factor=soil_factor,
            ability_or_efficiency_multiplier=ability_multiplier,
            technology_multiplier=tech_multiplier,
            structural_multiplier=structural_multiplier,
            service_bonus_multiplier=service_bonus_effective_multiplier,
            service_signal=service_signal,
            mechanization_signal=mechanization_signal,
            scale_progress=scale_progress,
            professional_signal=professional_signal,
            climate_shock=climate_shock,
            climate_modifier=effective_climate_modifier,
            resilience_buffer=resilience_buffer,
            random_factor=random_factor,
        )
        trace_key = 'grain' if crop_type == CropType.GRAIN else 'cash_crop'
        if not hasattr(self, 'last_yield_component_traces') or not isinstance(
            getattr(self, 'last_yield_component_traces', None), dict
        ):
            self.last_yield_component_traces = {}
        self.last_yield_component_traces[trace_key] = yield_trace
        self.last_yield_component_trace = yield_trace

        # @new 2026-06-24: Cobb-Douglas direct-measurement shadow
        # Uses actual factor inputs (land, labor, capital) instead of proxy multipliers.
        # NFPS-estimated elasticities from calibration/nfps_production_function.py
        _yield_mode = "legacy_stack"
        try:
            _yield_mode = str(self.model.config.get("production", {}).get("yield_mode", "legacy_stack"))
        except Exception:
            pass
        if _yield_mode == "production_function_shadow":
            try:
                cd_cfg = self.model.config.get("production", {}).get("cobb_douglas", {})
                cd_alpha = float(cd_cfg.get("alpha", 0.7026))
                cd_beta = float(cd_cfg.get("beta", 0.2547))
                cd_gamma = float(cd_cfg.get("gamma", 0.0173))
                # Direct factor measurements
                cd_land = max(0.01, float(area))
                cd_labor_days = max(0.01, float(getattr(self, 'total_labor_days', 1.0) or 1.0))
                cd_capital = max(0.01, float(getattr(self, 'cash_production_cost', 0.0) or getattr(self, 'total_expenditure', 0.0) or 1.0))
                cd_tfp = float(base_yield_per_mu)
                cd_direct_per_mu = cd_tfp * (cd_land ** cd_alpha) * (cd_labor_days ** cd_beta) * (cd_capital ** cd_gamma)
                cd_direct_yield = max(0.0, cd_direct_per_mu)
                yield_trace["shadow_cd_direct_input_yield_kg"] = float(cd_direct_yield)
                yield_trace["shadow_cd_direct_land_mu"] = float(cd_land)
                yield_trace["shadow_cd_direct_labor_days"] = float(cd_labor_days)
                yield_trace["shadow_cd_direct_capital_yuan"] = float(cd_capital)
            except Exception:
                pass

        return final_yield
    
    def _get_average_soil_quality(self):
        """获取平均土壤质量"""
        operated_patches = []
        for patch in self.land_patches:
            # 自有且未流转的地块
            if patch.owner == self and patch.leased_to is None:
                operated_patches.append(patch)
        
        # 租入的地块
        for patch in self.model.land_patches:
            if patch.leased_to == self:
                operated_patches.append(patch)
        
        if len(operated_patches) == 0:
            return 0.75  # 默认值
        
        avg_quality = sum(patch.soil_quality for patch in operated_patches) / len(operated_patches)
        return avg_quality
    
    def _get_technology_level(self):
        """获取技术水平"""
        # 如果有技术扩散模块，从那里获取
        # 否则，基于地块的技术水平计算平均值
        operated_patches = []
        for patch in self.land_patches:
            if patch.owner == self and patch.leased_to is None:
                operated_patches.append(patch)
        for patch in self.model.land_patches:
            if patch.leased_to == self:
                operated_patches.append(patch)
        
        if len(operated_patches) == 0:
            return 1.0  # 默认技术水平
        
        avg_tech = sum(getattr(patch, 'technology_level', 1.0) for patch in operated_patches) / len(operated_patches)
        return avg_tech
    
    def _get_climate_shock(self, mean, volatility):
        """
        获取气候冲击
        
        @enhanced 2025-01-02: 集成P3空间化气候风险
        优先使用ClimateManager的空间异质性灾害场，
        回退到environment的简化版或随机生成
        """
        # 1. Paired/shock experiments must use the deterministic worldline shock.
        env = getattr(self.model, 'environment', None)
        if (
            env is not None
            and getattr(env, '_worldline', None) is not None
            and hasattr(env, 'get_climate_shock')
        ):
            return env.get_climate_shock()

        # 2. 优先：P3空间化气候风险 (ClimateManager)
        if hasattr(self.model, 'climate_manager') and self.model.climate_manager is not None:
            # 获取农户位置
            pos = getattr(self, 'pos', None)
            if pos is not None:
                # 返回产量修正系数-1 (转换为冲击格式)
                yield_modifier = self.model.climate_manager.get_yield_modifier(pos)
                return yield_modifier - 1.0  # 1.0 → 0冲击, 0.6 → -0.4冲击
        
        # 3. 回退：environment简化版气候
        if env is not None and hasattr(env, 'get_climate_shock'):
            return env.get_climate_shock()
        
        # 4. 回退：随机生成
        if self._national_v1:
            return paired_yield_noise(self, 'climate_fallback', mean, volatility)
        return float(np_normal(self, mean, volatility))
    
    def _initialize_farmer_credit_constraints(self):
        """
        @new 2025-12: 初始化农户信贷约束相关参数
        专家审阅意见：规模户扩张依赖贷款，应有信贷约束
        
        理论依据：
        1. 农村金融约束：农户获得正规信贷困难（信贷可得性约40-60%）
        2. 非正规信贷：民间借贷利率高（10-20%）
        3. 抵押物要求：土地经营权抵押贷款政策
        4. 信贷歧视：小农户更难获得贷款
        
        农户信贷特点（与企业不同）：
        - 贷款规模小（几千到几万元）
        - 利率较高（农户6-10%）
        - 信贷可得性差（受规模、社会资本影响）
        - 主要用于生产性投资和扩大规模
        """
        # 贷款余额
        self.debt_balance = 0.0
        
        # 贷款利率（年化，农户通常较高）
        # 小农户依赖非正规信贷，利率更高
        operated_area = self.get_operated_land_area()
        if operated_area >= 50:  # 家庭农场
            self.loan_interest_rate = float(np_uniform(self, 0.06, 0.08))
            self.max_leverage_ratio = 0.4  # 最大负债率40%
        elif operated_area >= 15:  # 中等规模
            self.loan_interest_rate = float(np_uniform(self, 0.07, 0.10))
            self.max_leverage_ratio = 0.3  # 最大负债率30%
        else:  # 小农户
            self.loan_interest_rate = float(np_uniform(self, 0.08, 0.15))
            self.max_leverage_ratio = 0.2  # 最大负债率20%
        
        # 信贷可得性（银行愿意放贷的概率）
        self.credit_accessibility = self._calculate_farmer_credit_accessibility()
        
        # 贷款历史
        self.loan_history = []
    
    def _calculate_farmer_credit_accessibility(self):
        """
        @theory 计算农户信贷可得性
        
        影响因素：
        1. 经营规模（规模越大越容易）
        2. 社会资本（村精英更容易）
        3. 生计策略（纯农户略占优势）
        4. 土地资产（可抵押）
        """
        operated_area = self.get_operated_land_area()
        
        # 基于规模的基础可得性
        if operated_area >= 50:  # 家庭农场
            base_accessibility = 0.7
        elif operated_area >= 15:  # 中等规模
            base_accessibility = 0.5
        else:  # 小农户
            base_accessibility = 0.3
        
        # 社会资本加成（村精英）
        if self.is_village_elite:
            elite_bonus = 0.15
        else:
            elite_bonus = 0.0
        
        # 生计策略调整
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            strategy_bonus = 0.05  # 银行更信任专业农户
        else:
            strategy_bonus = 0.0
        
        return min(0.85, base_accessibility + elite_bonus + strategy_bonus)
    
    def apply_for_farmer_loan(self, amount):
        """
        @theory 农户申请贷款
        
        信贷约束模型：
        1. 检查资产负债率约束
        2. 检查银行放贷意愿
        3. 计算可批准金额
        
        Args:
            amount: 申请金额
            
        Returns:
            approved_amount: 批准的贷款金额（0表示拒绝）
        """
        # 计算当前资产（资本 + 土地价值）
        land_value = self.get_operated_land_area() * 300 * 10  # 每亩年租金300元，10倍估值
        total_assets = self.capital + land_value
        
        # 检查杠杆率约束
        max_additional_debt = total_assets * self.max_leverage_ratio - self.debt_balance
        if max_additional_debt <= 0:
            return 0  # 杠杆率已达上限
        
        # 银行放贷意愿（随机因素）
        if rng_random(self) > self.credit_accessibility:
            return 0  # 银行拒绝
        
        # 批准金额（不超过申请额和杠杆上限）
        approved_amount = min(amount, max_additional_debt)
        
        # 记录贷款
        if approved_amount > 0:
            self.debt_balance += approved_amount
            self.loan_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'amount': approved_amount,
                'interest_rate': self.loan_interest_rate
            })
        
        return approved_amount
    
    def repay_farmer_loan(self, amount):
        """偿还贷款"""
        actual_repay = min(amount, self.debt_balance)
        self.debt_balance -= actual_repay
        return actual_repay
    
    def get_farmer_annual_interest_expense(self):
        """计算农户年度利息支出"""
        return self.debt_balance * self.loan_interest_rate
    
    def get_farmer_available_capital_for_expansion(self):
        """
        @theory 计算农户可用于扩张的资金（考虑信贷）
        
        可用资金 = 可支配现金 + 可获得贷款
        """
        # 可支配现金（假设30%可用于新增投资）
        if self._national_v1:
            return self.balance_sheet_snapshot()['working_capital']
        available_cash = max(0, self.capital * 0.3)
        
        # 估算可获得贷款
        land_value = self.get_operated_land_area() * 300 * 10
        total_assets = self.capital + land_value
        potential_loan = max(0, total_assets * self.max_leverage_ratio - self.debt_balance)
        
        # 考虑信贷可得性折扣
        expected_loan = potential_loan * self.credit_accessibility
        
        return available_cash + expected_loan

    def balance_sheet_snapshot(self, annual_investment=0.0, added_area=0.0):
        """Minimal farm balance sheet used by land-expansion gates."""
        operated_area = max(0.0, float(self.get_operated_land_area() or 0.0))
        fixed_assets = (operated_area + max(0.0, float(added_area or 0.0))) * 500.0
        cash = max(0.0, float(getattr(self, 'capital', 0.0) or 0.0))
        household_reserve = float(getattr(self, 'household_cash_reserve_yuan', 0.0))
        debt = max(0.0, float(getattr(self, 'debt_balance', 0.0) or 0.0))
        investment = max(0.0, float(annual_investment or 0.0))
        sunk_exit_cost = fixed_assets * 0.15
        equity = cash + fixed_assets - debt
        return {
            'cash': cash,
            'working_capital': max(0.0, cash - household_reserve),
            'household_cash_reserve_yuan': household_reserve,
            'fixed_assets': fixed_assets,
            'debt': debt,
            'equity': equity,
            'net_worth': equity,
            'annual_investment': investment,
            'depreciation': fixed_assets * 0.08,
            'sunk_exit_cost': sunk_exit_cost,
        }

    def _passes_investment_sunk_cost_gate(self, patch, bid_price):
        """Reject expansion bids that would overrun simple balance-sheet capacity."""
        self.financial_gate_call_count = int(getattr(self, 'financial_gate_call_count', 0) or 0) + 1
        patch_area = max(0.0, float(getattr(patch, 'area', 0.0) or 0.0))
        snapshot = self.balance_sheet_snapshot(annual_investment=bid_price, added_area=patch_area)
        investment = snapshot['annual_investment']
        if investment <= 0.0:
            self.financial_gate_failure_count = int(getattr(self, 'financial_gate_failure_count', 0) or 0) + 1
            return False
        operated_area = max(0.0, float(self.get_operated_land_area() or 0.0))
        sunk_buffer = snapshot['sunk_exit_cost'] * min(1.0, patch_area / max(patch_area + operated_area, 1.0))
        equity = max(snapshot['equity'], 1.0)
        passed = (
            investment + sunk_buffer <= equity * 0.45
            and snapshot['cash'] - investment >= -0.05 * equity
        )
        if not passed:
            self.financial_gate_failure_count = int(getattr(self, 'financial_gate_failure_count', 0) or 0) + 1
        return passed
    
    def _check_age_exit(self):
        """
        @new 2025-12-17: 年龄相关退出机制
        @theory: 基于中国农村老龄化现实
        
        退出规则(专家咨询意见):
        - 55岁: 40%概率退出农业(主要为女性劳动力)
        - 65岁: 60%概率退出农业(男性劳动力逐步退出)
        - 75岁: 100%退出农业(完全丧失劳动能力)
        
        退出后果:
        - 所有土地流转出去(租给其他农户或企业)
        - 不再从事农业生产
        - 保留土地承包权,获得租金收入
        - 依靠转移性收入(养老金、子女赡养)生活
        
        返回:
            bool: True表示退出农业,False表示继续从事农业
        """
        age_exit_config = self.agent_config.get('agents', {}).get('farmer', {}).get('age_exit', {
            'age_55_exit_rate': 0.40,
            'age_65_exit_rate': 0.60,
            'age_75_exit_rate': 1.00,
            'enabled': True
        })
        
        if not age_exit_config.get('enabled', True):
            return False
        
        # 强制退出(75岁以上)
        if self.age >= 75:
            self._execute_age_exit()
            return True
        
        # 65岁: 60%概率退出
        if self.age >= 65:
            exit_rate = age_exit_config.get('age_65_exit_rate', 0.60)
            if np_random_value(self) < exit_rate:
                self._execute_age_exit()
                return True
        
        # 55岁: 40%概率退出
        if self.age >= 55:
            exit_rate = age_exit_config.get('age_55_exit_rate', 0.40)
            if np_random_value(self) < exit_rate:
                self._execute_age_exit()
                return True
        
        return False
    
    def _execute_age_exit(self):
        """
        @new 2025-12-17: 执行年龄退出
        
        退出操作:
        1. 将所有自有土地流转出去
        2. 标记为Non_Farmer(非农户)
        3. 农业收入置零,主要依靠租金和转移收入
        """
        # 流转所有自有土地
        for patch in self.land_patches:
            if patch.owner == self and patch.leased_to is None:
                # 将土地挂到流转市场,价格设为市场均价
                if hasattr(self.model, 'environment'):
                    rental_price_per_mu = float(self.model.environment.land_rental_price or 0.0)
                else:
                    rental_price_per_mu = 800.0  # 默认价格
                total_rental_price = rental_price_per_mu * max(float(getattr(patch, 'area', 0.0) or 0.0), 0.0)

                # 寻找承租方(优先企业,其次大农户)
                potential_tenants = []
                for agent in self.model.schedule.agents:
                    if hasattr(agent, 'enterprise_type'):  # 企业
                        potential_tenants.append((agent, 2))  # 权重2
                    elif hasattr(agent, 'farmer_scale') and agent.farmer_scale == FarmerScale.FAMILY_FARM:
                        potential_tenants.append((agent, 1))  # 权重1
                
                if potential_tenants:
                    # 加权随机选择承租方
                    weights = [w for _, w in potential_tenants]
                    tenant = np_choice(self, [t for t, _ in potential_tenants], p=np.array(weights)/sum(weights))
                    if patch.set_leased(
                        to_agent=tenant,
                        rental_price=total_rental_price,
                        transfer_channel='direct_farmer'
                    ) is False:
                        continue
        
        # 更新生计策略
        self.livelihood_strategy = LivelihoodStrategy.NON_FARMER
        
        # 清零农业相关变量
        self.grain_area = 0
        self.cash_crop_area = 0
        self.grain_production = 0
        self.cash_crop_production = 0
        self.farming_income = 0
        
        # 记录退出事件
        if hasattr(self.model, 'age_exit_records'):
            self.model.age_exit_records.append({
                'step': self.model.schedule.steps,
                'farmer_id': self.unique_id,
                'age': self.age,
                'land_area': sum(p.area for p in self.land_patches if p.owner == self)
            })
    
    def _determine_farmer_scale(self):
        """
        @new 2025-12: 根据土地经营面积确定农户规模类型
        
        规模划分标准（参考农业农村部《家庭农场培育计划》）：
        - 小农户(Smallholder)：经营面积 < 15亩
        - 中等规模农户(Medium_Farmer)：15亩 <= 经营面积 < 50亩
        - 家庭农场(Family_Farm)：经营面积 >= 50亩
        
        返回:
            FarmerScale: 农户规模枚举值
        """
        operated_area = self.get_operated_land_area()
        
        if operated_area < 15:
            return FarmerScale.SMALLHOLDER
        elif operated_area < 50:
            return FarmerScale.MEDIUM_FARMER
        else:
            return FarmerScale.FAMILY_FARM
    
    def update_farmer_scale(self):
        """
        @new 2025-12: 更新农户规模分类（每年调用）
        
        农户规模可能因土地流转而动态变化：
        - 租入更多土地 → 规模扩大 → 可能升级为家庭农场
        - 流转出土地 → 规模缩小 → 可能降级为小农户
        
        这体现了中国农村"种粮大户"的形成过程
        """
        old_scale = self.farmer_scale
        self.farmer_scale = self._determine_farmer_scale()
        
        # 如果规模发生变化，记录日志
        if old_scale != self.farmer_scale and not self._national_v1:
            # 规模升级：更愿意扩大规模，提高效率
            if (old_scale == FarmerScale.SMALLHOLDER and 
                self.farmer_scale in [FarmerScale.MEDIUM_FARMER, FarmerScale.FAMILY_FARM]):
                # 规模扩大后，农业能力略有提升（规模经济效应）
                self.farming_ability = min(1.5, self.farming_ability * 1.05)
            
            # 如果成为家庭农场，进一步提升效率
            if self.farmer_scale == FarmerScale.FAMILY_FARM:
                self.farming_ability = min(1.5, self.farming_ability * 1.03)
    
    def is_family_farm(self):
        """
        @new 2025-12: 判断是否为家庭农场
        
        返回:
            bool: 如果是家庭农场返回True
        """
        return self.farmer_scale == FarmerScale.FAMILY_FARM
    
    def get_scale_based_efficiency_bonus(self):
        """
        @new 2025-12: 获取基于规模的效率加成
        
        规模经济效应：
        - 小农户：无加成
        - 中等规模：5%加成
        - 家庭农场：10%加成
        
        返回:
            float: 效率乘数
        """
        if self.farmer_scale == FarmerScale.SMALLHOLDER:
            return 1.0
        elif self.farmer_scale == FarmerScale.MEDIUM_FARMER:
            return 1.05
        else:  # FAMILY_FARM
            return 1.10
    
    def _get_differentiated_prospect_config(self):
        """
        @theory
        获取差异化的前景理论参数
        @improvement: P2优先级 - 按生计策略和收入水平差异化
        """
        # 获取配置
        farmer_config = self.model.config.get('agents', {}).get('farmer', {})
        prospect_config_all = farmer_config.get('prospect_theory', {})
        
        # 根据生计策略获取参数（初始时使用默认生计策略）
        # 注意：在初始化时，livelihood_strategy可能还未确定，使用默认值
        strategy_name = getattr(self, 'livelihood_strategy', LivelihoodStrategy.PLURIACTIVE).name
        
        by_strategy = prospect_config_all.get('by_livelihood_strategy', {})
        strategy_params = by_strategy.get(strategy_name)
        
        if strategy_params:
            # 使用策略特定参数
            lambda_dist = strategy_params.get('loss_aversion_lambda_dist', {})
            curvature_dist = strategy_params.get('value_function_curvature_dist', {})
            prob_weight_dist = strategy_params.get('probability_weighting_dist', {})
        else:
            # 使用默认参数
            default = prospect_config_all.get('default', {})
            lambda_dist = default.get('loss_aversion_lambda_dist', {'type': 'normal', 'mean': 3.0, 'std': 0.5})
            curvature_dist = default.get('value_function_curvature_dist', {'type': 'normal', 'mean': 0.55, 'std': 0.1})
            prob_weight_dist = default.get('probability_weighting_dist', {'type': 'normal', 'mean': 0.7, 'std': 0.05})
        
        # 从分布中采样
        if lambda_dist.get('type') == 'normal':
            lambda_val = float(np_normal(self, lambda_dist.get('mean', 3.0), lambda_dist.get('std', 0.5)))
        else:
            lambda_val = float(np_normal(self, 3.0, 0.5))
        
        if curvature_dist.get('type') == 'normal':
            curvature_val = float(np_normal(self, curvature_dist.get('mean', 0.55), curvature_dist.get('std', 0.1)))
        else:
            curvature_val = float(np_normal(self, 0.55, 0.1))
        
        if prob_weight_dist.get('type') == 'normal':
            prob_weight_val = float(np_normal(self, prob_weight_dist.get('mean', 0.7), prob_weight_dist.get('std', 0.05)))
        else:
            prob_weight_val = float(np_normal(self, 0.7, 0.05))
        
        # 构建配置字典
        prospect_config = {
            "loss_aversion_lambda": lambda_val,
            "value_function_curvature_sigma": curvature_val,
            "probability_weighting_alpha": prob_weight_val,
        }
        
        return prospect_config
    
    def update_prospect_theory_by_income(self):
        """
        @theory
        根据收入水平动态调整前景理论参数
        @improvement: P2优先级 - 按收入水平差异化
        """
        # 获取收入分位数配置
        farmer_config = self.model.config.get('agents', {}).get('farmer', {})
        prospect_config_all = farmer_config.get('prospect_theory', {})
        by_income = prospect_config_all.get('by_income_quartile', {})
        
        current_income_signal = self._get_income_signal()
        if not by_income or current_income_signal <= 0:
            return  # 如果没有配置或没有收入数据，不调整
        
        # 计算收入分位数（需要所有农户的收入数据）
        # 简化：基于历史收入记忆或当前收入估算分位数
        if hasattr(self, 'income_memory') and len(self.income_memory) > 0:
            avg_income = np.mean(self.income_memory)
        else:
            avg_income = current_income_signal
        
        # 估算分位数（简化：基于当前收入与初始资本的比较）
        initial_capital = getattr(self, 'initial_capital', 10000)
        income_ratio = avg_income / initial_capital if initial_capital > 0 else 1.0
        
        # 确定分位数（简化规则）
        if income_ratio < 0.5:
            quartile = 'q1'
        elif income_ratio < 1.0:
            quartile = 'q2'
        elif income_ratio < 1.5:
            quartile = 'q3'
        else:
            quartile = 'q4'
        
        # 获取该分位数的参数调整
        quartile_params = by_income.get(quartile, {})
        lambda_mean_adjustment = quartile_params.get('loss_aversion_lambda_mean')
        
        if lambda_mean_adjustment:
            # 调整损失厌恶系数（平滑调整，避免突变）
            adjustment_factor = lambda_mean_adjustment / 3.0  # 相对于基准值3.0
            current_lambda = self.prospect_theory.loss_aversion_lambda
            target_lambda = current_lambda * adjustment_factor
            
            # 平滑调整（每次调整10%）
            self.prospect_theory.loss_aversion_lambda = 0.9 * current_lambda + 0.1 * target_lambda

    def _v1_grain_base_yield(self, fallback):
        # Runner: 2010 NFPS median of household prod_food / sow_food in kg per sown mu.
        table = self.model.config['runtime_fixed_rule'].get('grain_kg_mu_by_region', {})
        region = getattr(self, 'region', None)
        if region not in table:
            region = getattr(self, '_microdata_source', {}).get('region')
        return table.get(region, fallback)

    def _national_crop_sales(self, grain_kg, cash_kg):
        # Marketed fractions are scenario assumptions, not observed farm-level transactions.
        ratios = self.model.config['runtime_fixed_rule'].get('household_sales_ratio', {})
        grain, cash = float(ratios.get('grain', .8)), float(ratios.get('cash_crop', .85))
        if not (0 <= grain <= 1 and 0 <= cash <= 1):
            raise ValueError('household_sales_ratio must be between zero and one')
        pg, pc = self._get_output_price_per_kg('crop_price'), self._get_output_price_per_kg('cash_crop')
        sold_g, sold_c = grain_kg * grain, cash_kg * cash
        return dict(grain_sold_kg=sold_g, cash_crop_sold_kg=sold_c, sold_output_kg=sold_g + sold_c,
                    unsold_output_kg=grain_kg - sold_g + cash_kg - sold_c,
                    output_value_yuan=grain_kg * pg + cash_kg * pc, sales_yuan=sold_g * pg + sold_c * pc)

    def harvest_and_settle(self):
        """
        @theory
        收获与收入结算
        新增：更新收入记忆以支持前景理论的动态参考点机制
        """
        capital_before_harvest = float(self.capital)
        # 计算实际产量（加入随机波动）
        production_config = self.model.config.get('production', {})
        settlement_noise_raw = production_config.get('random_noise', {}).get('settlement_std', 0.02)
        try:
            settlement_noise_std = max(0.0, float(settlement_noise_raw))
        except (TypeError, ValueError):
            settlement_noise_std = 0.02
        if self.expected_yield > 0:
            if settlement_noise_std > 0:
                factor = (paired_yield_noise(self, 'settlement', 1.0, settlement_noise_std)
                          if self._national_v1 else float(np_normal(self, 1.0, settlement_noise_std)))
                actual_yield = self.expected_yield * (max(0., factor) if self._national_v1 else factor)
            else:
                actual_yield = float(self.expected_yield)
        else:
            actual_yield = 0
        
        # @fix: 记录本期的实际产出和作物选择，供下一期市场更新使用
        self.last_yield = actual_yield
        self.last_crop_choice = self.planned_crop
        # @new 2026-01-27: 记录粮食/经济作物产量（便于统一指标收集）
        self.grain_production = 0
        self.cash_crop_production = 0
        production_area = max(0.0, float(getattr(self, 'current_production_area', 0.0) or 0.0))
        self.grain_area = 0.0
        self.cash_crop_area = 0.0
        if self.planned_crop == CropType.MIXED:
            self.grain_area = production_area * self.household_planned_grain_share
            self.cash_crop_area = production_area - self.grain_area
            self.grain_production = actual_yield * self.household_expected_grain_kg / self.expected_yield if self.expected_yield > 0 else 0.0
            self.cash_crop_production = actual_yield - self.grain_production
        elif self.planned_crop == CropType.GRAIN:
            self.grain_production = actual_yield
            self.grain_area = production_area
        elif self.planned_crop == CropType.CASH_CROP:
            self.cash_crop_production = actual_yield
            self.cash_crop_area = production_area
        if hasattr(self, "household_candidate_grain_shares"):
            self.household_actual_grain_share = self.grain_area / production_area if production_area > 0 else None
        assign_grain_crop_components(
            self,
            model=self.model,
            grain_area_mu=self.grain_area,
            grain_output_kg=self.grain_production,
            actor_type="farmer",
        )
        
        # @fix: Non_Farmer的收入完全来源于非农收入
        # 但可能有土地情节，保留少量土地用于自给自足小农生产，满足自身必须食物消费
        # 所以Non_Farmer可以有少量农业收入（自给自足），但主要收入是非农收入
        if self.planned_crop == CropType.MIXED:
            self.agricultural_income = (self.grain_production * self._get_output_price_per_kg('crop_price')
                                        + self.cash_crop_production * self._get_output_price_per_kg('cash_crop'))
        elif self.planned_crop == CropType.GRAIN:
            self.agricultural_income = actual_yield * self._get_output_price_per_kg('crop_price')
        elif self.planned_crop == CropType.CASH_CROP:
            self.agricultural_income = actual_yield * self._get_output_price_per_kg('cash_crop')
        else:
            # @fix: 如果planned_crop为None，确保有默认值，避免收入为0
            if self.planned_crop is None and not (action_income_enabled(self.model.config) and getattr(self, "fixed_rule_exit_pending", False)):
                # 默认选择粮食作物，确保有收入
                self.planned_crop = CropType.GRAIN
                if actual_yield > 0:
                    self.agricultural_income = actual_yield * self._get_output_price_per_kg('crop_price')
                else:
                    self.agricultural_income = 0
            else:
                self.agricultural_income = 0
        if self._national_v1:
            harvest_sales = self._national_crop_sales(self.grain_production, self.cash_crop_production)
            self.agricultural_income = harvest_sales['sales_yuan']
        self.planting_gross_income = max(0.0, float(self.agricultural_income or 0.0))
        
        # 计算其他收入
        self.calculate_land_rental_income()
        self.calculate_subsidy_income() # @fix: 移到此处，在结算时计算
        self.calculate_off_farm_income()
        
        # 计算支出
        self.calculate_production_cost()
        self.calculate_land_rental_expense()
        
        # 更新总收入（未减去支出）
        self.gross_income = (
            self.agricultural_income +
            self.land_rental_income +
            self.subsidy_income +
            self.off_farm_income +
            self.labor_sale_income
        )
        if action_income_enabled(self.model.config):
            self.gross_income += float(getattr(self, "fixed_rule_common_income_yuan", 0.0))
        self.total_income = self.gross_income  # 向后兼容
        self.total_expense = (
            self.cash_production_cost +
            self.land_rental_expense +
            float(getattr(self, 'service_fee_expense', 0.0) or 0.0)
        )
        if action_income_enabled(self.model.config):
            self.total_expense += float(getattr(self, "fixed_rule_conversion_expense", 0.0))
        self.economic_total_expense = self.total_expense + self.imputed_family_labor_cost
        self.planting_net_income = max(
            0.0,
            float(self.agricultural_income or 0.0)
            - float(self.cash_production_cost or 0.0)
            - float(getattr(self, 'service_fee_expense', 0.0) or 0.0),
        )
        
        # 现金净收入与经济利润分离：前者用于福利/行为，后者保留机会成本分析
        self.net_income = self.gross_income - self.total_expense
        self.cash_net_income = self.net_income
        self.economic_profit = self.gross_income - self.economic_total_expense
        
        # 计算可支配收入（假设储蓄率为20%）
        savings_rate = 0.2
        self.annual_savings_rate = savings_rate
        self.disposable_income = self.net_income * (1 - savings_rate)
        
        # 更新资本
        # @fix: 确保资本不会变成极端负值，限制在合理范围内
        settle_preproduction_cash(self, self.total_expense)
        self.capital += self.net_income
        if action_income_enabled(self.model.config):
            self.last_income_settlement = {
                "step": self.model.step_count,
                "action": getattr(self, "fixed_rule_current_action", None),
                "capital_before_harvest_yuan": capital_before_harvest,
                "capital_after_harvest_yuan": float(self.capital),
                "output_kg": float(actual_yield),
                "sold_output_kg": float(actual_yield),
                "sales_yuan": float(self.agricultural_income),
                "rent_received_yuan": float(self.land_rental_income),
                "off_farm_income_yuan": float(self.off_farm_income),
                "labor_sale_income_yuan": float(self.labor_sale_income),
                "common_income_yuan": float(getattr(self, "fixed_rule_common_income_yuan", 0.0)),
                "support_at_harvest_yuan": float(self.subsidy_income),
                "verified_support_received_yuan": 0.0,
                "production_cost_yuan": float(self.cash_production_cost),
                "hired_labor_cost_yuan": float(self.hired_labor_cost),
                "input_cost_before_in_kind_yuan": (
                    float(self.cash_production_cost + self.production_in_kind_offset - self.hired_labor_cost)
                    if self.model.config.get("production", {}).get("cost_structure", {}).get("use_detailed_cost", True)
                    else None
                ),
                "in_kind_cost_offset_yuan": float(self.production_in_kind_offset),
                "service_cost_yuan": float(getattr(self, "service_fee_expense", 0.0)),
                "rent_paid_yuan": float(self.land_rental_expense),
                "conversion_cost_yuan": float(getattr(self, "fixed_rule_conversion_expense", 0.0)),
                "family_labor_imputed_yuan": float(self.imputed_family_labor_cost),
                "net_income_at_harvest_yuan": float(self.net_income),
                "family_labor_days": float(self.estimate_production_labor_profile()["family_labor_used"]),
                "hired_labor_days": float(self.hired_labor_days),
                "labor_sold_days": float(self.labor_sold_days),
                "off_farm_days": float(self.realized_off_farm_labor_days),
            }
            if self._national_v1:
                self.last_income_settlement.update(harvest_sales)
        
        # 计算消费和效用（新增）
        self.calculate_consumption_and_utility()
        self.submit_subsidy_claim()
        # 限制资本在合理范围内（避免极端值）
        if not action_income_enabled(self.model.config):
            self.capital = max(-50000, min(self.capital, 10000000))
        
        # 新增：更新收入记忆以支持前景理论动态参考点机制
        self.update_income_memory()
    
    def update_income_memory(self):
        """
        @theory
        更新收入记忆，支持前景理论的动态参考点机制
        保持有限记忆长度，确保参考点的动态更新
        """
        # 将本期净收入信号加入记忆
        self.income_memory.append(self._get_income_signal())
        
        # 保持记忆长度限制（默认5年）
        if len(self.income_memory) > self.memory_length:
            self.income_memory = self.income_memory[-self.memory_length:]
    
    def build_social_network(self, all_farmers):
        """
        @theory
        构建社会网络 - ABM的核心特征
        基于地理邻近性和社会资本形成网络连接
        """
        self.social_network = []
        self.network_strength = {}
        self.transaction_neighbors = []
        self.transaction_link_strength = {}
        self.weak_tie_contacts = []
        
        social_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('social', {})
        max_network_size_cfg = int(social_cfg.get('average_degree', 8))
        transaction_degree_cfg = int(social_cfg.get('transaction_avg_degree', max(3, max_network_size_cfg // 2)))
        max_network_size = min(max_network_size_cfg, max(0, len(all_farmers) - 1))
        
        # 计算与其他农户的"距离"（地理+社会）
        neighbor_scores = []
        for other_farmer in all_farmers:
            if other_farmer.unique_id != self.unique_id:
                # 地理距离（基于土地位置的简化计算）
                geo_distance = self.calculate_social_distance(other_farmer)
                
                # 社会相似性
                social_similarity = self.calculate_social_similarity(other_farmer)
                
                # 综合得分（距离越小，相似性越高，得分越高）
                score = social_similarity / (1 + geo_distance)
                neighbor_scores.append((other_farmer, score))
        
        # 选择得分最高的几个作为社会网络邻居
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        selected_neighbors = neighbor_scores[:max_network_size]
        
        # @new 2025-12-28: 小世界网络弱连接机制
        # @theory Granovetter (1973) "The Strength of Weak Ties"
        # 问题：同质化选择导致回音室效应，技术扩散被阻断
        # 解决：10%概率添加随机长程连接，打破圈层壁垒
        weak_tie_probability = float(social_cfg.get('weak_tie_prob', 0.10) or 0.0)
        weak_tie_probability = max(0.0, min(1.0, weak_tie_probability))
        weak_tie_strength_floor = float(social_cfg.get('weak_tie_strength_floor', 0.18) or 0.18)
        weak_tie_strength_floor = max(0.1, min(1.0, weak_tie_strength_floor))
        weak_tie_count = max(0, int(round(max_network_size * weak_tie_probability)))
        
        # 从未被选中的农户中随机选择弱连接
        selected_ids = {n[0].unique_id for n in selected_neighbors}
        remaining_farmers = [ns for ns in neighbor_scores if ns[0].unique_id not in selected_ids]
        
        weak_tie_ids = set()
        available_weak_tie_slots = min(
            weak_tie_count,
            len(selected_neighbors),
            len(remaining_farmers),
        )
        if available_weak_tie_slots > 0:
            weak_ties = rng_sample(self, remaining_farmers, available_weak_tie_slots)
            weak_tie_ids = {neighbor.unique_id for neighbor, _ in weak_ties}
            # 用弱连接替换部分强连接；候选不足时仍保留可用的弱连接份额
            if available_weak_tie_slots >= len(selected_neighbors):
                selected_neighbors = weak_ties
            else:
                selected_neighbors = selected_neighbors[:-available_weak_tie_slots] + weak_ties
        
        for neighbor, score in selected_neighbors:
            self.social_network.append(neighbor.unique_id)
            # 网络强度基于综合得分，归一化到[0.1, 1.0]
            if neighbor.unique_id in weak_tie_ids:
                self.weak_tie_contacts.append(neighbor.unique_id)
                self.network_strength[neighbor.unique_id] = min(
                    1.0,
                    max(weak_tie_strength_floor, score),
                )
            else:
                self.network_strength[neighbor.unique_id] = min(1.0, max(0.1, score))

        # 交易可行性网络：优先选择得分最高、关系稳固的邻居
        if selected_neighbors:
            weak_candidates = [
                pair for pair in selected_neighbors
                if pair[0].unique_id in weak_tie_ids
            ]
            strong_candidates = [
                pair for pair in selected_neighbors
                if pair[0].unique_id not in weak_tie_ids
            ]
            transaction_target = min(transaction_degree_cfg, len(selected_neighbors))
            weak_slots = 0
            if weak_candidates and transaction_target > 0:
                weak_slots = min(len(weak_candidates), 1)
            strong_slots = max(0, transaction_target - weak_slots)
            transaction_subset = strong_candidates[:strong_slots] + weak_candidates[:weak_slots]
            if len(transaction_subset) < transaction_target:
                remaining_candidates = strong_candidates[strong_slots:] + weak_candidates[weak_slots:]
                transaction_subset.extend(remaining_candidates[: transaction_target - len(transaction_subset)])
            for neighbor, score in transaction_subset:
                self.transaction_neighbors.append(neighbor.unique_id)
                # 交易网络强调互信，保证最低强度阈值
                baseline_strength = self.network_strength.get(neighbor.unique_id, score)
                self.transaction_link_strength[neighbor.unique_id] = min(1.0, max(0.2, baseline_strength))
    
    def calculate_social_distance(self, other_farmer):
        """
        @theory
        计算与其他农户的社会距离（简化版）
        """
        # 基于土地位置的简化距离计算
        if self.land_patches and other_farmer.land_patches:
            # 取第一个地块作为代表
            my_patch = self.land_patches[0]
            other_patch = other_farmer.land_patches[0]
            if hasattr(my_patch, 'coordinates') and hasattr(other_patch, 'coordinates'):
                dx = my_patch.coordinates[0] - other_patch.coordinates[0]
                dy = my_patch.coordinates[1] - other_patch.coordinates[1]
                return np.sqrt(dx**2 + dy**2)
        
        # 如果没有坐标信息，返回随机距离
        return float(np_uniform(self, 1, 10))
    
    def calculate_social_similarity(self, other_farmer):
        """
        @theory
        计算社会相似性 - 基于共同特征
        """
        similarity = 0.5  # 基础相似性
        
        # 生计策略相似性
        if self.livelihood_strategy == other_farmer.livelihood_strategy:
            similarity += 0.3
        
        # 村精英状态相似性  
        if self.is_village_elite == other_farmer.is_village_elite:
            similarity += 0.2
        
        # 能力水平相似性（农业能力和非农工资水平）
        ability_diff = abs(self.farming_ability - other_farmer.farming_ability)
        wage_diff = abs(self.off_farm_wage_level - other_farmer.off_farm_wage_level)
        ability_similarity = max(0, 1 - (ability_diff + wage_diff) / 2)
        similarity += 0.3 * ability_similarity
        
        return min(1.0, similarity)
    
    def observe_neighbors(self):
        """
        @theory
        观察邻居的决策和收益 - 社会学习的信息收集阶段
        农户通过社会网络获取关于不同作物选择效果的信息
        """
        if not self.social_network:
            return {}
        
        neighbor_info = {
            'grain_adopters': [],
            'cash_crop_adopters': [],
            'grain_avg_profit': 0,
            'cash_crop_avg_profit': 0,
            'grain_success_rate': 0,
            'cash_crop_success_rate': 0
        }
        
        grain_profits = []
        cash_crop_profits = []
        grain_count = 0
        cash_crop_count = 0
        
        # 遍历社会网络中的邻居
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor and hasattr(neighbor, 'crop_choice_history') and neighbor.crop_choice_history:
                # 获取邻居最近的作物选择和收益
                recent_choice = neighbor.crop_choice_history[-1] if neighbor.crop_choice_history else None
                recent_income = neighbor.income_memory[-1] if neighbor.income_memory else 0
                
                if recent_choice and 'choice' in recent_choice:
                    network_strength = self.network_strength.get(neighbor_id, 0.5)
                    
                    if recent_choice['choice'] == CropType.GRAIN:
                        neighbor_info['grain_adopters'].append({
                            'id': neighbor_id,
                            'profit': recent_income,
                            'strength': network_strength
                        })
                        grain_profits.append(recent_income * network_strength)  # 加权收益
                        grain_count += 1
                    elif recent_choice['choice'] == CropType.CASH_CROP:
                        neighbor_info['cash_crop_adopters'].append({
                            'id': neighbor_id,
                            'profit': recent_income,
                            'strength': network_strength
                        })
                        cash_crop_profits.append(recent_income * network_strength)
                        cash_crop_count += 1
        
        # 计算加权平均收益
        if grain_profits:
            neighbor_info['grain_avg_profit'] = np.mean(grain_profits)
            neighbor_info['grain_success_rate'] = grain_count / len(self.social_network)
        
        if cash_crop_profits:
            neighbor_info['cash_crop_avg_profit'] = np.mean(cash_crop_profits)
            neighbor_info['cash_crop_success_rate'] = cash_crop_count / len(self.social_network)
        
        # 记录观察历史
        self.neighbor_observation_history.append(neighbor_info)
        if len(self.neighbor_observation_history) > self.memory_length:
            self.neighbor_observation_history = self.neighbor_observation_history[-self.memory_length:]
        
        return neighbor_info
    
    def calculate_social_influence(self, neighbor_info):
        """
        @theory
        计算社会影响 - 社会学习的核心机制
        基于邻居的成功经验调整对不同作物的倾向
        """
        social_influence = {
            'grain_influence': 0.0,
            'cash_crop_influence': 0.0,
            'herding_effect': 0.0
        }
        
        if not neighbor_info or not self.social_network:
            return social_influence
        
        # 获取自己的历史收益作为比较基准
        my_recent_income = self.income_memory[-1] if self.income_memory else self.reference_point
        
        # 1. 相对收益影响
        if neighbor_info['grain_avg_profit'] > my_recent_income * 1.1:
            # 邻居种粮收益显著高于自己，产生正向影响
            social_influence['grain_influence'] = (neighbor_info['grain_avg_profit'] / my_recent_income - 1) * self.information_receptivity
        
        if neighbor_info['cash_crop_avg_profit'] > my_recent_income * 1.1:
            # 邻居种经济作物收益显著高于自己
            social_influence['cash_crop_influence'] = (neighbor_info['cash_crop_avg_profit'] / my_recent_income - 1) * self.information_receptivity
        
        # 2. 羊群效应（从众行为）
        total_neighbors = len(self.social_network)
        if total_neighbors > 0:
            grain_adoption_rate = neighbor_info['grain_success_rate']
            cash_crop_adoption_rate = neighbor_info['cash_crop_success_rate']
            
            # 当某种作物采用率很高时，产生从众压力
            if grain_adoption_rate > 0.6:  # 60%以上邻居种粮
                social_influence['herding_effect'] += (grain_adoption_rate - 0.5) * self.conformity_tendency
                social_influence['grain_influence'] += social_influence['herding_effect']
            
            if cash_crop_adoption_rate > 0.6:  # 60%以上邻居种经济作物
                herding_cash = (cash_crop_adoption_rate - 0.5) * self.conformity_tendency
                social_influence['cash_crop_influence'] += herding_cash
        
        # 3. 网络强度调整
        # 考虑与成功邻居的关系强度
        for adopter in neighbor_info['grain_adopters']:
            if adopter['profit'] > my_recent_income:
                strength_bonus = adopter['strength'] * 0.2
                social_influence['grain_influence'] += strength_bonus
        
        for adopter in neighbor_info['cash_crop_adopters']:
            if adopter['profit'] > my_recent_income:
                strength_bonus = adopter['strength'] * 0.2
                social_influence['cash_crop_influence'] += strength_bonus
        
        # 限制影响范围，避免过度影响
        social_influence['grain_influence'] = min(2.0, max(-0.5, social_influence['grain_influence']))
        social_influence['cash_crop_influence'] = min(2.0, max(-0.5, social_influence['cash_crop_influence']))
        
        return social_influence
    
    def update_risk_preference(self):
        """
        @theory
        动态风险偏好更新 - 财富效应与经验学习
        基于财富变化和经验学习调整风险偏好
        """
        # 1. 财富效应
        # 确保capital是实数，避免复数比较
        capital_real = float(np.real(self.capital)) if isinstance(self.capital, complex) else float(self.capital)
        
        if self.initial_capital > 0 and capital_real > 0:
            wealth_ratio = max(0.01, capital_real / self.initial_capital)  # 防止除零和负数
            # 财富增加 → 风险偏好降低（更愿意冒险）
            # 财富减少 → 风险偏好增加（更保守）
            wealth_adjustment = min(2.0, max(0.5, 1.0 / wealth_ratio))
        else:
            wealth_adjustment = 1.0
        
        # 2. 经验学习
        experience_adjustment = 1.0
        
        # 判断最近的成功/失败
        if len(self.income_memory) >= 2:
            recent_income = self.income_memory[-1]
            previous_income = self.income_memory[-2]
            
            if recent_income > previous_income * 1.1:  # 收入增长超过10%视为成功
                self.success_streak += 1
                self.failure_streak = 0
            elif recent_income < previous_income * 0.9:  # 收入下降超过10%视为失败
                self.failure_streak += 1
                self.success_streak = 0
            else:
                # 平稳状态，重置计数
                self.success_streak = 0
                self.failure_streak = 0
        
        # 连续成功 → 降低风险偏好（更冒险）
        if self.success_streak >= 2:
            experience_adjustment = 1.0 - (self.success_streak * self.risk_learning_rate)
            
        # 连续失败 → 提高风险偏好（更保守）
        elif self.failure_streak >= 2:
            experience_adjustment = 1.0 + (self.failure_streak * self.risk_learning_rate)
        
        # 3. 更新风险偏好
        experience_adjustment = min(2.0, max(0.5, experience_adjustment))
        self.risk_aversion = self.base_risk_aversion * wealth_adjustment * experience_adjustment
        
        # 确保风险偏好在合理范围内
        self.risk_aversion = min(5.0, max(0.5, self.risk_aversion))
    
    def update_prospect_theory_parameters(self):
        """
        @theory
        基于经验动态调整前景理论参数
        @improvement: P2优先级 - 增加收入水平差异化调整
        """
        # 连续成功可能降低损失厌恶（更愿意承担风险）
        if self.success_streak >= 3:
            adjustment_factor = 1.0 - (self.success_streak * 0.05)
            adjustment_factor = max(0.8, adjustment_factor)  # 最多降低20%
            
            # 调整损失厌恶系数
            original_lambda = self.prospect_theory.loss_aversion_lambda
            self.prospect_theory.loss_aversion_lambda = original_lambda * adjustment_factor
        
        # 连续失败可能增加损失厌恶（更保守）
        elif self.failure_streak >= 3:
            adjustment_factor = 1.0 + (self.failure_streak * 0.05)
            adjustment_factor = min(1.3, adjustment_factor)  # 最多增加30%
            
            original_lambda = self.prospect_theory.loss_aversion_lambda
            self.prospect_theory.loss_aversion_lambda = original_lambda * adjustment_factor
        
        # @improvement: P2优先级 - 根据收入水平动态调整
        self.update_prospect_theory_by_income()

    def _derive_livelihood_strategy_from_continuous(self) -> LivelihoodStrategy:
        """
        @critical 2025-12: 从连续变量派生生计策略标签
        
        专家意见："用连续变量替代离散标签"
        
        生计策略不再是独立状态，而是由 agricultural_dependence 连续变量计算得出
        这样做的好处：
        1. 消除参数等效性风险（不同参数组合产生相同结果）
        2. 行为连续变化，而非阶跃
        3. 可追溯：标签变化可解释为收入结构变化
        
        映射规则（基于CLES数据校准）：
        - agricultural_dependence > 0.65 → PURE_FARMER
        - agricultural_dependence < 0.35 → NON_FARMER
        - otherwise → PLURIACTIVE
        """
        if hasattr(self, 'characteristics') and self.characteristics is not None:
            agri_dep = self.characteristics.agricultural_dependence
        else:
            # 回退：从收入计算
            core_income = self._get_agriculture_linked_income() + max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
            if core_income > 0:
                agri_dep = self._get_livelihood_agricultural_income_ratio()
            else:
                agri_dep = 0.5  # 默认值
        
        # 阈值判断（与原逻辑保持一致）
        if agri_dep > 0.65:
            return LivelihoodStrategy.PURE_FARMER
        elif agri_dep < 0.35:
            return LivelihoodStrategy.NON_FARMER
        else:
            return LivelihoodStrategy.PLURIACTIVE

    def livelihood_strategy_evolution(self):
        """
        @theory
        动态博弈：生计策略演化
        农户根据博弈结果动态调整自身类型和未来策略
        
        @critical 2025-12: 简化重构 - 生计策略由连续变量自动派生
        专家意见："生计策略不应是独立变量，应由连续特征自动计算"
        
        @fix 2025-12-04 v2: 修复纯农户异常增多问题
        核心问题：
        1. 缺乏城镇化结构趋势压力
        2. 从兼业户到纯农户的转换条件太宽松
        
        修复思路：
        1. 大幅降低"农业专业化"转换概率（仅5%）
        2. 提高城镇化流出概率（年轻人20%，老年人年龄退出20%）
        3. 限制成为纯农户的条件（需要规模经营+极低非农收入）
        
        中国农村现实：
        - 每年约500万农村劳动力转移到城镇
        - 纯农户占比从2000年的60%下降到2020年的20%左右
        - 35岁以下农业从业者仅占10%
        """
        # 从配置读取演化参数
        evolution_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('livelihood_evolution', {})
        pure_farmer_threshold = evolution_cfg.get('pure_farmer_threshold', 0.50)
        non_farmer_threshold = evolution_cfg.get('non_farmer_threshold', 0.20)
        cooldown_steps = evolution_cfg.get('cooldown_steps', 1)
        
        # 城镇化结构趋势参数 - @fix 2025-12: 大幅降低流出率，防止纯农户过快减少
        # 原设置base_outflow_prob=0.12导致每步12%流出，20步后纯农户几乎消失
        # 新设置：每步2%流出，符合中国年化劳动力转移约3%的现实
        urbanization_cfg = evolution_cfg.get('urbanization_trend', {})
        base_outflow_prob = urbanization_cfg.get('base_outflow_prob', 0.02)  # @fix: 从0.12降到0.02
        young_multiplier = urbanization_cfg.get('young_multiplier', 1.5)    # @fix: 从2.0降到1.5
        elderly_exit_prob = urbanization_cfg.get('elderly_exit_prob', 0.05) # @fix: 从0.20降到0.05
        
        # 更新历史记录
        self.update_history()
        
        # 计算农业收入占比
        core_income = self._get_agriculture_linked_income() + max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
        if core_income > 0:
            agricultural_income_ratio = self._get_livelihood_agricultural_income_ratio()
            agricultural_income_ratio = max(0.0, min(1.0, agricultural_income_ratio))
        else:
            if self.agricultural_income > 0:
                agricultural_income_ratio = 1.0
            elif self.off_farm_income > 0:
                agricultural_income_ratio = 0.0
            else:
                agricultural_income_ratio = 0.5
        
        # 更新连续指标
        self.update_consecutive_indicators(agricultural_income_ratio)

        if self._is_initial_cohort_baseline_observation(
            evolution_cfg.get('livelihood_persistence', {}) or {}
        ):
            self.last_livelihood_transition_skipped_reason = "initial_cohort_baseline_observation"
            return

        # @critical 2025-12: 检查代际接替事件
        self._check_generational_succession()
        
        # 检查切换冷却期
        if self.strategy_switch_cooldown > 0:
            self.strategy_switch_cooldown -= 1
            return
        
        old_strategy = self.livelihood_strategy
        new_strategy = old_strategy
        transition_reason = None
        transition_probability = 0.0
        self.last_nonfarmer_reentry_audit = None
        
        # 获取年龄
        age = getattr(self, 'age', 45)
        operated_area = self.get_operated_land_area()
        professional_profile = self._assess_professional_farmer_profile(
            evolution_cfg=evolution_cfg,
            agricultural_income_ratio=agricultural_income_ratio,
            operated_area=operated_area,
            age=age,
        )
        neighbor_influence_weight = evolution_cfg.get('neighbor_influence_weight', 0.08)
        transition_cfg = evolution_cfg.get('transition_probabilities', {})
        stable_offfarm_to_nonfarmer_prob = float(
            transition_cfg.get('stable_offfarm_to_nonfarmer_prob', 0.20) or 0.20
        )
        stable_offfarm_income_threshold = float(
            transition_cfg.get('stable_offfarm_income_threshold', 15000) or 15000
        )
        stable_offfarm_agri_ratio_cap = float(
            transition_cfg.get('stable_offfarm_agri_ratio_cap', 0.40) or 0.40
        )
        pure_farmer_diversification_prob = float(
            transition_cfg.get('pure_farmer_diversification_prob', 0.15) or 0.15
        )
        pure_farmer_direct_nonfarmer_prob = float(
            transition_cfg.get('pure_farmer_direct_nonfarmer_prob', 0.0) or 0.0
        )
        pure_farmer_direct_nonfarmer_income_threshold = float(
            transition_cfg.get('pure_farmer_direct_nonfarmer_income_threshold', stable_offfarm_income_threshold)
            or stable_offfarm_income_threshold
        )
        smallholder_pressure_area_threshold = float(
            transition_cfg.get('smallholder_pressure_area_threshold', 0.0) or 0.0
        )
        smallholder_pure_to_pluriactive_prob = float(
            transition_cfg.get('smallholder_pure_to_pluriactive_prob', 0.0) or 0.0
        )
        smallholder_direct_nonfarmer_prob = float(
            transition_cfg.get('smallholder_direct_nonfarmer_prob', 0.0) or 0.0
        )
        smallholder_direct_nonfarmer_income_threshold = float(
            transition_cfg.get(
                'smallholder_direct_nonfarmer_income_threshold',
                pure_farmer_direct_nonfarmer_income_threshold,
            )
            or pure_farmer_direct_nonfarmer_income_threshold
        )
        pluriactive_structural_nonfarmer_prob = float(
            transition_cfg.get('pluriactive_structural_nonfarmer_prob', 0.0) or 0.0
        )
        strict_scale_reentry_prob = float(
            transition_cfg.get('strict_scale_reentry_prob', 0.02) or 0.02
        )
        nonfarmer_to_pluriactive_prob = float(
            transition_cfg.get('nonfarmer_to_pluriactive_prob', 0.0) or 0.0
        )
        nonfarmer_reentry_min_agri_ratio = float(
            transition_cfg.get('nonfarmer_reentry_min_agri_ratio', non_farmer_threshold)
            or non_farmer_threshold
        )
        nonfarmer_reentry_min_operated_area = float(
            transition_cfg.get('nonfarmer_reentry_min_operated_area', 1.0) or 1.0
        )
        nonfarmer_reentry_max_age = int(
            transition_cfg.get('nonfarmer_reentry_max_age', 60) or 60
        )
        persistence_cfg = evolution_cfg.get('livelihood_persistence', {}) or {}
        pure_farmer_persistence_multiplier = float(
            persistence_cfg.get('pure_farmer_outflow_multiplier', 1.0) or 1.0
        )
        new_entrant_transition_multiplier = float(
            persistence_cfg.get('new_entrant_transition_multiplier', 1.0) or 1.0
        )
        new_entrant_retention_years = int(
            persistence_cfg.get('new_entrant_retention_years', 0) or 0
        )
        transition_persistence_multiplier = 1.0
        if old_strategy == LivelihoodStrategy.PURE_FARMER:
            transition_persistence_multiplier *= max(0.0, min(1.0, pure_farmer_persistence_multiplier))
        model_obj = getattr(self, 'model', None)
        base_year = int(getattr(model_obj, 'simulation_base_year', 2010) or 2010)
        current_year_for_persistence = int(
            base_year + getattr(model_obj, 'step_count', 0)
        )
        if bool(getattr(self, 'is_new_entrant', False)) and new_entrant_retention_years > 0:
            try:
                entrant_tenure = max(
                    0,
                    current_year_for_persistence - int(
                        getattr(self, 'entry_year', current_year_for_persistence)
                    ),
                )
            except (TypeError, ValueError):
                entrant_tenure = 0
            if entrant_tenure < new_entrant_retention_years:
                transition_persistence_multiplier *= max(0.0, min(1.0, new_entrant_transition_multiplier))
        elif not bool(getattr(self, 'is_new_entrant', False)):
            initial_retention_years = int(
                persistence_cfg.get('initial_cohort_retention_years', 0) or 0
            )
            initial_transition_multiplier = float(
                persistence_cfg.get('initial_cohort_transition_multiplier', 1.0) or 1.0
            )
            initial_tenure = max(
                0,
                current_year_for_persistence - int(getattr(self, 'entry_year', base_year) or base_year),
            )
            if initial_tenure < initial_retention_years:
                transition_persistence_multiplier *= max(
                    0.0,
                    min(1.0, initial_transition_multiplier),
                )

        timing_multipliers = {
            'stable_offfarm_to_nonfarmer': 1.0,
            'pluriactive_structural_nonfarmer': 1.0,
            'pure_farmer_diversification': 1.0,
            'pure_farmer_direct_nonfarmer': 1.0,
            'smallholder_pure_to_pluriactive': 1.0,
            'smallholder_direct_nonfarmer': 1.0,
            'pure_outflow': 1.0,
            'pluriactive_elderly_exit': 1.0,
            'dominant_nonfarm': 1.0,
            'pluriactive_neighbor_pressure': 1.0,
        }
        timing_cfg = evolution_cfg.get('transition_timing_adjustments', {}) or {}
        if bool(timing_cfg.get('enabled', False)):
            current_year = int(
                getattr(getattr(self, 'model', None), 'simulation_base_year', 2010)
                + getattr(getattr(self, 'model', None), 'step_count', 0)
            )
            for period_cfg in timing_cfg.get('periods', []) or []:
                try:
                    start_year = int(period_cfg.get('start_year', current_year))
                    end_year = int(period_cfg.get('end_year', current_year))
                except (TypeError, ValueError):
                    continue
                if start_year <= current_year <= end_year:
                    for key in timing_multipliers:
                        cfg_key = f'{key}_multiplier'
                        if cfg_key in period_cfg:
                            try:
                                timing_multipliers[key] *= max(0.0, float(period_cfg[cfg_key]))
                            except (TypeError, ValueError):
                                continue

        stable_offfarm_to_nonfarmer_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['stable_offfarm_to_nonfarmer']
        )
        pure_farmer_diversification_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['pure_farmer_diversification']
        )
        pure_farmer_direct_nonfarmer_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['pure_farmer_direct_nonfarmer']
        )
        smallholder_pure_to_pluriactive_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['smallholder_pure_to_pluriactive']
        )
        smallholder_direct_nonfarmer_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['smallholder_direct_nonfarmer']
        )
        pluriactive_structural_nonfarmer_prob *= (
            transition_persistence_multiplier
            * timing_multipliers['pluriactive_structural_nonfarmer']
        )
        neighbor_influence_weight *= transition_persistence_multiplier
        
        # ========== 结构趋势1: 城镇化吸引（单向流出） ==========
        # 这是中国农村的主导趋势：劳动力从农业流向非农
        
        # 年龄调整后的流出概率
        if age < 35:
            outflow_prob = base_outflow_prob * young_multiplier  # 年轻人流出概率高
        elif age < 45:
            outflow_prob = base_outflow_prob * 1.5
        elif age > 60:
            outflow_prob = base_outflow_prob * 0.2  # 老年人流动性低
        else:
            outflow_prob = base_outflow_prob
        outflow_prob *= professional_profile['outflow_multiplier']
        outflow_prob *= transition_persistence_multiplier
        if old_strategy == LivelihoodStrategy.PURE_FARMER:
            outflow_prob *= timing_multipliers['pure_outflow']

        if old_strategy == LivelihoodStrategy.PURE_FARMER:
            pressure_start = float(transition_cfg.get('pure_nonfarmer_pressure_start', 1.0) or 1.0)
            pressure_multiplier = float(transition_cfg.get('pure_nonfarmer_pressure_multiplier', 0.0) or 0.0)
            pressure_cap = float(transition_cfg.get('pure_nonfarmer_pressure_cap', 1.0) or 1.0)
            if pressure_multiplier > 0 and pressure_cap > 1.0:
                current_farmers = list(getattr(getattr(self, 'model', None), 'farmers', []) or [])
                if current_farmers:
                    current_nonfarmer_share = sum(
                        1
                        for farmer in current_farmers
                        if getattr(farmer, 'livelihood_strategy', None) == LivelihoodStrategy.NON_FARMER
                    ) / len(current_farmers)
                    if current_nonfarmer_share > pressure_start:
                        outflow_prob *= min(
                            pressure_cap,
                            1.0 + pressure_multiplier * (current_nonfarmer_share - pressure_start),
                        )

        pluriactive_nonfarm_saturation_factor = 1.0
        if old_strategy == LivelihoodStrategy.PLURIACTIVE:
            saturation_start = float(transition_cfg.get('nonfarmer_saturation_start', 1.0) or 1.0)
            saturation_cap = float(transition_cfg.get('nonfarmer_saturation_cap', 1.0) or 1.0)
            saturation_min_factor = float(
                transition_cfg.get('nonfarmer_saturation_min_factor', 0.25) or 0.25
            )
            if saturation_cap > saturation_start:
                current_farmers = list(getattr(getattr(self, 'model', None), 'farmers', []) or [])
                if current_farmers:
                    current_nonfarmer_share = sum(
                        1
                        for farmer in current_farmers
                        if getattr(farmer, 'livelihood_strategy', None) == LivelihoodStrategy.NON_FARMER
                    ) / len(current_farmers)
                    if current_nonfarmer_share > saturation_start:
                        progress = min(
                            1.0,
                            (current_nonfarmer_share - saturation_start)
                            / max(1e-9, saturation_cap - saturation_start),
                        )
                        pluriactive_nonfarm_saturation_factor = max(
                            saturation_min_factor,
                            1.0 - progress * (1.0 - saturation_min_factor),
                        )
        
        # Pure_Farmer → Pluriactive（城镇化吸引）
        if old_strategy == LivelihoodStrategy.PURE_FARMER:
            if rng_random(self) < outflow_prob:
                new_strategy = LivelihoodStrategy.PLURIACTIVE
                transition_reason = "城镇化吸引"
                transition_probability = outflow_prob
        
        # Pluriactive → Non_Farmer（非农收入稳定后完全转型）
        elif old_strategy == LivelihoodStrategy.PLURIACTIVE:
            if (
                self.off_farm_income >= stable_offfarm_income_threshold
                and agricultural_income_ratio <= stable_offfarm_agri_ratio_cap
            ):
                stable_nonfarm_prob = stable_offfarm_to_nonfarmer_prob
                if getattr(self, 'consecutive_high_non_farm_ratio', 0) >= 2:
                    stable_nonfarm_prob = min(0.95, stable_nonfarm_prob + 0.10)
                stable_nonfarm_prob *= pluriactive_nonfarm_saturation_factor
                if rng_random(self) < stable_nonfarm_prob:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "非农收入稳定"
                    transition_probability = stable_nonfarm_prob
            if (
                new_strategy == old_strategy
                and pluriactive_structural_nonfarmer_prob > 0
                and not professional_profile['retain_agriculture']
            ):
                structural_nonfarm_prob = pluriactive_structural_nonfarmer_prob
                if age < 45:
                    structural_nonfarm_prob *= 1.15
                if agricultural_income_ratio <= stable_offfarm_agri_ratio_cap:
                    structural_nonfarm_prob *= 1.20
                if self.off_farm_income >= stable_offfarm_income_threshold * 0.5:
                    structural_nonfarm_prob *= 1.15
                structural_nonfarm_prob *= pluriactive_nonfarm_saturation_factor
                structural_nonfarm_prob = min(0.95, max(0.0, structural_nonfarm_prob))
                if rng_random(self) < structural_nonfarm_prob:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "兼业非农化"
                    transition_probability = structural_nonfarm_prob

        elif old_strategy == LivelihoodStrategy.NON_FARMER:
            agriculture_linked_income = self._get_agriculture_linked_income()
            reentry_area_ok = operated_area >= nonfarmer_reentry_min_operated_area
            reentry_agri_ratio_ok = agricultural_income_ratio >= nonfarmer_reentry_min_agri_ratio
            reentry_income_ok = agriculture_linked_income > 0.0
            reentry_age_ok = age <= nonfarmer_reentry_max_age
            has_reentered_agriculture = (
                reentry_area_ok
                and reentry_agri_ratio_ok
                and reentry_income_ok
                and reentry_age_ok
            )
            self.last_nonfarmer_reentry_audit = {
                'year': int(
                    getattr(getattr(self, 'model', None), 'simulation_base_year', 2010)
                    + getattr(getattr(self, 'model', None), 'step_count', 0)
                ),
                'eligible_state': True,
                'operated_area_mu': operated_area,
                'minimum_operated_area_mu': nonfarmer_reentry_min_operated_area,
                'agricultural_income_ratio': agricultural_income_ratio,
                'minimum_agricultural_income_ratio': nonfarmer_reentry_min_agri_ratio,
                'agriculture_linked_income': agriculture_linked_income,
                'age': age,
                'maximum_age': nonfarmer_reentry_max_age,
                'area_condition': reentry_area_ok,
                'agricultural_ratio_condition': reentry_agri_ratio_ok,
                'agriculture_income_condition': reentry_income_ok,
                'age_condition': reentry_age_ok,
                'candidate': has_reentered_agriculture,
                'probability': min(1.0, max(0.0, nonfarmer_to_pluriactive_prob)),
                'realized': False,
            }
            if has_reentered_agriculture and nonfarmer_to_pluriactive_prob > 0.0:
                reentry_probability = min(1.0, max(0.0, nonfarmer_to_pluriactive_prob))
                if rng_random(self) < reentry_probability:
                    new_strategy = LivelihoodStrategy.PLURIACTIVE
                    transition_reason = "农业经营恢复"
                    transition_probability = reentry_probability
                    self.last_nonfarmer_reentry_audit['realized'] = True
        
        # ========== 结构趋势2: 老年退出 ==========
        if age > 60:
            if old_strategy == LivelihoodStrategy.PURE_FARMER:
                pure_farmer_exit_prob = (
                    elderly_exit_prob
                    * professional_profile['elderly_exit_multiplier']
                    * transition_persistence_multiplier
                )
                if rng_random(self) < pure_farmer_exit_prob:
                    new_strategy = LivelihoodStrategy.PLURIACTIVE
                    transition_reason = "年龄退出"
                    transition_probability = pure_farmer_exit_prob
            elif old_strategy == LivelihoodStrategy.PLURIACTIVE:
                pluriactive_exit_prob = elderly_exit_prob * 0.5
                pluriactive_exit_prob *= transition_persistence_multiplier
                pluriactive_exit_prob *= timing_multipliers['pluriactive_elderly_exit']
                if professional_profile['retain_agriculture']:
                    pluriactive_exit_prob *= professional_profile['elderly_exit_multiplier']
                if rng_random(self) < pluriactive_exit_prob:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "老年完全退出"
                    transition_probability = pluriactive_exit_prob
        
        # ========== 限制性逆向转换：成为纯农户条件极其严格 ==========
        # @fix 2025-12-01: 进一步收紧纯农户转换条件
        # 理由：中国农村纯农户占比持续下降，从2000年的60%降到2020年的20%
        # 成为纯农户必须同时满足：
        # 1. 规模经营(>50亩，真正的家庭农场规模)
        # 2. 极低非农收入(<2000元)
        # 3. 极高农业占比(>95%)
        # 4. 年龄在35-55岁（务农黄金期）
        # 即使满足所有条件，概率也仅2%
        if old_strategy == LivelihoodStrategy.PLURIACTIVE and new_strategy == old_strategy:
            if professional_profile['is_reentry_candidate']:
                if rng_random(self) < professional_profile['reentry_probability']:
                    new_strategy = LivelihoodStrategy.PURE_FARMER
                    transition_reason = "职业化回流"
                    transition_probability = professional_profile['reentry_probability']
            elif (agricultural_income_ratio > 0.95 and 
                  operated_area >= 50 and 
                  self.off_farm_income < 2000 and
                  35 <= age <= 55):
                if rng_random(self) < strict_scale_reentry_prob:
                    new_strategy = LivelihoodStrategy.PURE_FARMER
                    transition_reason = "规模化专业化"
                    transition_probability = strict_scale_reentry_prob
        
        # ========== 收入驱动的策略调整（补充机制） ==========
        if (
            old_strategy == LivelihoodStrategy.PURE_FARMER
            and new_strategy == old_strategy
            and smallholder_pressure_area_threshold > 0
            and operated_area <= smallholder_pressure_area_threshold
            and not professional_profile['retain_agriculture']
        ):
            scale_pressure_prob = smallholder_pure_to_pluriactive_prob
            if age < 45:
                scale_pressure_prob *= 1.15
            if self.off_farm_income >= smallholder_direct_nonfarmer_income_threshold:
                if rng_random(self) < smallholder_direct_nonfarmer_prob:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "smallholder_pressure"
                    transition_probability = smallholder_direct_nonfarmer_prob
            elif rng_random(self) < scale_pressure_prob:
                new_strategy = LivelihoodStrategy.PLURIACTIVE
                transition_reason = "smallholder_pressure"
                transition_probability = scale_pressure_prob

        if new_strategy == old_strategy:
            # 非农收入占主导时转为Non_Farmer
            if agricultural_income_ratio < non_farmer_threshold:
                if self.off_farm_income >= stable_offfarm_income_threshold:
                    dominant_nonfarm_prob = stable_offfarm_to_nonfarmer_prob
                    if getattr(self, 'consecutive_high_non_farm_ratio', 0) >= 2:
                        dominant_nonfarm_prob = min(0.95, dominant_nonfarm_prob + 0.10)
                    if old_strategy == LivelihoodStrategy.PLURIACTIVE:
                        dominant_nonfarm_prob *= timing_multipliers['dominant_nonfarm']
                    if rng_random(self) < dominant_nonfarm_prob:
                        new_strategy = LivelihoodStrategy.NON_FARMER
                        transition_reason = "非农主导"
                        transition_probability = dominant_nonfarm_prob
            
            # 中间状态趋向兼业（从Pure_Farmer流出）
            elif old_strategy == LivelihoodStrategy.PURE_FARMER:
                if (
                    agricultural_income_ratio < non_farmer_threshold
                    and self.off_farm_income >= pure_farmer_direct_nonfarmer_income_threshold
                    and rng_random(self) < pure_farmer_direct_nonfarmer_prob
                ):
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "非农主导"
                    transition_probability = pure_farmer_direct_nonfarmer_prob
                elif agricultural_income_ratio < pure_farmer_threshold:
                    income_diversification_prob = pure_farmer_diversification_prob
                    if professional_profile['retain_agriculture']:
                        income_diversification_prob *= professional_profile['income_diversification_multiplier']
                    if rng_random(self) < income_diversification_prob:
                        new_strategy = LivelihoodStrategy.PLURIACTIVE
                        transition_reason = "收入分散"
                        transition_probability = income_diversification_prob
        
        # ========== 邻居策略传染效应 ==========
        if new_strategy == old_strategy and self.social_network:
            neighbor_strategies = []
            for nid in self.social_network:
                neighbor = self.model.get_agent_by_id(nid)
                if neighbor and hasattr(neighbor, 'livelihood_strategy'):
                    neighbor_strategies.append(neighbor.livelihood_strategy)
            
            if neighbor_strategies:
                non_farmer_ratio = neighbor_strategies.count(LivelihoodStrategy.NON_FARMER) / len(neighbor_strategies)
                
                # 邻居多为非农户，增加流出压力
                if non_farmer_ratio > 0.40 and old_strategy != LivelihoodStrategy.NON_FARMER:
                    neighbor_transition_prob = neighbor_influence_weight
                    if professional_profile['retain_agriculture']:
                        neighbor_transition_prob *= professional_profile['neighbor_pressure_multiplier']
                    if old_strategy == LivelihoodStrategy.PLURIACTIVE:
                        neighbor_transition_prob *= timing_multipliers['pluriactive_neighbor_pressure']
                    if agricultural_income_ratio < 0.50 and rng_random(self) < neighbor_transition_prob:
                        if old_strategy == LivelihoodStrategy.PURE_FARMER:
                            new_strategy = LivelihoodStrategy.PLURIACTIVE
                        else:
                            new_strategy = LivelihoodStrategy.NON_FARMER
                        transition_reason = "邻居影响"
                        transition_probability = neighbor_transition_prob
        
        # ========== 执行策略切换 ==========
        if old_strategy != new_strategy:
            self.livelihood_strategy = new_strategy
            self.strategy_switch_cooldown = cooldown_steps
            print(f"[生计演化] 农户{self.unique_id}: {old_strategy.value}→{new_strategy.value}, "
                  f"农业占比:{agricultural_income_ratio:.2f}, 年龄:{age}, 原因:{transition_reason}")
            
            # @new 2025-11-30: 记录到墓地系统 - 如果转为Non_Farmer
            if new_strategy == LivelihoodStrategy.NON_FARMER:
                self._mark_structural_urbanization(transition_reason, agricultural_income_ratio)
                self._record_exit_to_graveyard(old_strategy, transition_reason, age, agricultural_income_ratio)

        transition_probability = min(1.0, max(0.0, float(transition_probability or 0.0)))
        transition_year = int(
            getattr(getattr(self, 'model', None), 'simulation_base_year', 2010)
            + getattr(getattr(self, 'model', None), 'step_count', 0)
        )
        self.previous_livelihood_strategy = old_strategy
        self.current_livelihood_strategy = new_strategy
        self.transition_reason = transition_reason or "状态保持"
        self.transition_probability = transition_probability
        self.last_transition_probability = transition_probability
        self.transition_year = transition_year
        if not hasattr(self, 'livelihood_transition_history'):
            self.livelihood_transition_history = []
        self.livelihood_transition_history.append({
            'year': transition_year,
            'from_state': old_strategy.value,
            'to_state': new_strategy.value,
            'transition_reason': self.transition_reason,
            'transition_probability': transition_probability,
            'stay_probability': 1.0 - transition_probability,
            'realized_transition': old_strategy != new_strategy,
            'agricultural_income_ratio': agricultural_income_ratio,
            'operated_area_mu': operated_area,
            'nonfarmer_reentry_audit': dict(
                getattr(self, 'last_nonfarmer_reentry_audit', None) or {}
            ),
        })
        
        # @critical 2025-12: 最终更新 - 同步连续变量到 characteristics
        if hasattr(self, 'characteristics') and self.characteristics is not None:
            self._sync_characteristics_income_fields(use_strategy_anchor=False)
            if old_strategy != new_strategy:
                self._align_characteristics_to_livelihood_strategy(blend=0.60)
            else:
                self.characteristics.agricultural_dependence = agricultural_income_ratio
            from modules.farmer_characteristics import FarmerTypeClassifier
            self.farmer_type, self.farmer_type_confidence = FarmerTypeClassifier.classify(self.characteristics)
            sensitivity = FarmerTypeClassifier.get_type_policy_sensitivity(self.farmer_type)
            self.characteristics.price_subsidy_sensitivity = sensitivity.get('price_subsidy', 0.5)
            self.characteristics.direct_subsidy_sensitivity = sensitivity.get('direct_subsidy', 0.5)
            self.characteristics.rental_subsidy_sensitivity = sensitivity.get('rental_subsidy', 0.5)
            self.characteristics.service_policy_sensitivity = sensitivity.get('service', 0.5)

    def _assess_professional_farmer_profile(self, evolution_cfg, agricultural_income_ratio, operated_area, age):
        """Identify a shrinking but persistent professional-farmer core."""
        retention_cfg = evolution_cfg.get('professional_farmer_retention', {})
        default_profile = {
            'retain_agriculture': False,
            'is_reentry_candidate': False,
            'outflow_multiplier': 1.0,
            'elderly_exit_multiplier': 1.0,
            'income_diversification_multiplier': 1.0,
            'neighbor_pressure_multiplier': 1.0,
            'reentry_probability': 0.0,
        }
        if not retention_cfg.get('enabled', False):
            return default_profile

        min_operated_area = retention_cfg.get('min_operated_area', 20)
        min_ag_ratio = retention_cfg.get('min_agricultural_income_ratio', 0.78)
        max_off_farm_income = retention_cfg.get('max_off_farm_income', 12000)
        min_farming_ability = retention_cfg.get('min_farming_ability', 0.95)
        preferred_age_min = retention_cfg.get('preferred_age_min', 28)
        preferred_age_max = retention_cfg.get('preferred_age_max', 58)
        legacy_farmer_age_max = retention_cfg.get('legacy_farmer_age_max', 65)

        farming_ability = float(getattr(self, 'farming_ability', 1.0) or 1.0)
        in_core_age_band = preferred_age_min <= age <= preferred_age_max
        within_legacy_age_band = age <= legacy_farmer_age_max
        has_professional_scale = operated_area >= min_operated_area
        retains_agriculture = (
            has_professional_scale
            and agricultural_income_ratio >= min_ag_ratio
            and self.off_farm_income <= max_off_farm_income
            and farming_ability >= min_farming_ability
            and within_legacy_age_band
        )

        reentry_candidate = (
            operated_area >= retention_cfg.get('pluriactive_reentry_min_area', 18)
            and agricultural_income_ratio >= retention_cfg.get('pluriactive_reentry_ratio', 0.85)
            and self.off_farm_income <= retention_cfg.get('pluriactive_reentry_max_off_farm_income', 8000)
            and farming_ability >= min_farming_ability
            and in_core_age_band
        )

        return {
            'retain_agriculture': retains_agriculture,
            'is_reentry_candidate': reentry_candidate,
            'outflow_multiplier': (
                retention_cfg.get('outflow_resistance_multiplier', 0.35)
                if retains_agriculture else 1.0
            ),
            'elderly_exit_multiplier': (
                retention_cfg.get('elderly_exit_resistance_multiplier', 0.60)
                if retains_agriculture else 1.0
            ),
            'income_diversification_multiplier': (
                retention_cfg.get('income_diversification_resistance_multiplier', 0.45)
                if retains_agriculture else 1.0
            ),
            'neighbor_pressure_multiplier': (
                retention_cfg.get('neighbor_pressure_resistance_multiplier', 0.40)
                if retains_agriculture else 1.0
            ),
            'reentry_probability': (
                retention_cfg.get('pluriactive_reentry_prob', 0.06)
                if reentry_candidate else 0.0
            ),
        }
    
    def _update_social_network_dynamics(self):
        """
        @new 2025-12-28: 社会网络动态重构
        
        @theory: 费孝通"差序格局" + Granovetter弱连接理论
        
        现实机制：
        1. 进城务工后，与村庄的社会联系逐渐减弱
        2. 长期不在村的农户，其信息影响力下降
        3. 返乡后，网络影响力可以部分恢复
        
        实现：
        - NON_FARMER: 网络强度每年衰减10%
        - PLURIACTIVE: 网络强度每年衰减5%
        - PURE_FARMER: 网络强度保持或增强
        """
        if not hasattr(self, 'social_network_strength'):
            self.social_network_strength = 1.0  # 初始化网络影响力
        if not hasattr(self, 'years_away_from_village'):
            self.years_away_from_village = 0
        
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            # 非农户：逐步脱离村庄网络
            self.years_away_from_village += 1
            decay_rate = 0.10  # 每年衰减10%
            self.social_network_strength *= (1 - decay_rate)
            self.social_network_strength = max(0.1, self.social_network_strength)  # 下限10%
            
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业户：部分脱离
            decay_rate = 0.05
            self.social_network_strength *= (1 - decay_rate)
            self.social_network_strength = max(0.3, self.social_network_strength)
            
        else:  # PURE_FARMER
            # 纯农户：网络强度恢复/增强
            if self.years_away_from_village > 0:
                # 返乡农户，逐步恢复
                recovery_rate = 0.15
                self.social_network_strength = min(1.0, self.social_network_strength + recovery_rate)
                self.years_away_from_village = max(0, self.years_away_from_village - 1)
            else:
                # 一直在村，网络强度维持或略增
                self.social_network_strength = min(1.0, self.social_network_strength + 0.02)
    
    def _update_satisfaction_and_mood(self):
        """
        @new 2025-12-28: P2满意化原则 + P3情绪状态更新
        
        @literature:
        - Simon (1955): 满意化原则，有限理性
        - Kahneman (2011): 快思慢想，情绪影响决策
        - Aspiration level理论：期望水平动态调整
        
        机制：
        1. 满意度：基于实际收入与期望的比较
        2. 期望水平：适应性调整（成功上调，失败下调）
        3. 风险情绪：连续成功→乐观，连续失败→谨慎
        """
        # === 1. 计算满意度 ===
        if self.aspiration_income > 0:
            ratio = self.net_income / self.aspiration_income
            # 满意度在收入达到期望80%时开始上升
            if ratio >= 1.2:
                self.satisfaction = 0.9
            elif ratio >= 1.0:
                self.satisfaction = 0.7 + 0.2 * (ratio - 1.0) / 0.2
            elif ratio >= 0.8:
                self.satisfaction = 0.5 + 0.2 * (ratio - 0.8) / 0.2
            else:
                self.satisfaction = 0.3 * ratio / 0.8
        
        # === 2. 更新连续亏损/盈利状态 ===
        if self.net_income < self.aspiration_income * 0.8:
            self.recent_loss = True
            self.consecutive_loss_years += 1
        else:
            self.recent_loss = False
            self.consecutive_loss_years = 0
        
        # === 3. 更新风险情绪 ===
        # @theory: 连续失败导致风险厌恶增加
        if self.consecutive_loss_years >= 2:
            self.risk_mood = 'cautious'
        elif self.satisfaction > 0.7 and not self.recent_loss:
            self.risk_mood = 'optimistic'
        else:
            self.risk_mood = 'neutral'
        
        # === 4. 适应性期望调整（Aspiration level） ===
        # 成功后期望上调，失败后期望下调
        adaptation_rate = 0.1
        if self.net_income > self.aspiration_income:
            # 超过期望：提高期望
            self.aspiration_income *= (1 + adaptation_rate * 0.5)
        elif self.net_income < self.aspiration_income * 0.7:
            # 远低于期望：降低期望
            self.aspiration_income *= (1 - adaptation_rate * 0.3)
        
        # 期望上下限
        self.aspiration_income = max(10000, min(200000, self.aspiration_income))

    def _calculate_neighbor_strategy_pressure(self):
        """
        @enhancement 2025-11-30: 计算邻居策略压力
        返回邻居中占主导的策略类型
        """
        if not self.social_network:
            return self.livelihood_strategy
        
        strategy_counts = {
            LivelihoodStrategy.PURE_FARMER: 0,
            LivelihoodStrategy.PLURIACTIVE: 0,
            LivelihoodStrategy.NON_FARMER: 0
        }
        
        for nid in self.social_network:
            neighbor = self.model.get_agent_by_id(nid)
            if neighbor and hasattr(neighbor, 'livelihood_strategy'):
                strategy_counts[neighbor.livelihood_strategy] += 1
        
        # 返回占比最高的策略
        dominant_strategy = max(strategy_counts, key=strategy_counts.get)
        return dominant_strategy
    
    def _apply_external_shock(self):
        """
        @enhancement 2025-11-30: 应用外部冲击
        模拟宏观经济变化、政策调整等外部因素
        返回是否发生冲击
        """
        # 基础冲击概率：每年10%的农户受到冲击
        base_shock_prob = 0.10
        
        # 年龄因素：年轻农户更容易受城镇化吸引
        age_factor = 1.0
        if hasattr(self, 'age'):
            if self.age < 40:
                age_factor = 1.5  # 年轻人更容易转型
            elif self.age > 55:
                age_factor = 0.6  # 老年人更稳定
        
        # 非农收入因素：已有高非农收入的更可能完全转型
        income_factor = 1.0
        if self.off_farm_income > 20000:
            income_factor = 1.3
        
        shock_prob = base_shock_prob * age_factor * income_factor
        return rng_random(self) < shock_prob
    
    def _record_exit_to_graveyard(self, old_strategy, transition_reason, age, agricultural_income_ratio):
        """
        @new 2025-11-30: 记录农户退出到墓地系统
        
        当农户转为Non_Farmer时，记录其最终状态
        这有助于避免幸存者偏差，追踪谁离开了农业以及为什么
        
        Args:
            old_strategy: 原生计策略
            transition_reason: 转换原因
            age: 年龄
            agricultural_income_ratio: 农业收入占比
        """
        if not hasattr(self.model, 'graveyard') or self.model.graveyard is None:
            return
        
        try:
            from modules.graveyard import ExitReason
            
            # 根据转换原因映射到退出原因
            reason_mapping = {
                "年龄退出": ExitReason.AGE_RETIREMENT,
                "老年完全退出": ExitReason.AGE_RETIREMENT,
                "城镇化吸引": ExitReason.URBANIZATION,
                "非农收入稳定": ExitReason.VOLUNTARY_EXIT,
                "非农主导": ExitReason.VOLUNTARY_EXIT,
                "规模化专业化": ExitReason.VOLUNTARY_EXIT,
                "收入分散": ExitReason.VOLUNTARY_EXIT,
                "邻居影响": ExitReason.URBANIZATION,
                "代际接替": ExitReason.GENERATIONAL_EXIT,
                "无人接班": ExitReason.GENERATIONAL_EXIT,
            }
            
            exit_reason = reason_mapping.get(transition_reason, ExitReason.OTHER)
            
            # 构建注释
            notes = (f"从{old_strategy.value}转为Non_Farmer，"
                    f"原因:{transition_reason}，"
                    f"年龄:{age}，"
                    f"农业收入占比:{agricultural_income_ratio:.2%}")
            
            self.model.graveyard.record_exit(
                agent=self,
                exit_reason=exit_reason,
                step=self.model.step_count,
                notes=notes
            )
        except Exception as e:
            # 不因墓地记录失败而影响主流程
            pass

    def update_history(self):
        """
        @theory
        更新历史记录
        """
        # 更新收入历史
        self.income_history.append(self._get_income_signal())
        if len(self.income_history) > 5:
            self.income_history.pop(0)
        
        # 更新农业收入占比历史
        # @fix: 使用与livelihood_strategy_evolution()一致的计算方式（只考虑农业收入和非农收入）
        core_income = self._get_agriculture_linked_income() + max(0.0, float(getattr(self, 'off_farm_income', 0.0) or 0.0))
        if core_income > 0:
            agricultural_income_ratio = self._get_livelihood_agricultural_income_ratio()
        else:
            if self.agricultural_income > 0:
                agricultural_income_ratio = 1.0
            elif self.off_farm_income > 0:
                agricultural_income_ratio = 0.0
            else:
                agricultural_income_ratio = 0.5
        self.agricultural_income_ratio_history.append(agricultural_income_ratio)
        if len(self.agricultural_income_ratio_history) > 5:
            self.agricultural_income_ratio_history.pop(0)
        
        # 更新策略历史
        self.strategy_history.append(self.livelihood_strategy)
        if len(self.strategy_history) > 5:
            self.strategy_history.pop(0)

    def update_consecutive_indicators(self, agricultural_income_ratio):
        """
        @theory
        更新连续指标
        @fix: 使用新的阈值（20%/50%），与当前CFPS运行口径一致
        """
        evolution_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('livelihood_evolution', {})
        non_farmer_threshold = evolution_cfg.get('non_farmer_threshold', 0.20)
        pure_farmer_threshold = evolution_cfg.get('pure_farmer_threshold', 0.50)
        if agricultural_income_ratio < non_farmer_threshold:
            self.consecutive_high_non_farm_ratio += 1
            self.consecutive_low_agricultural_ratio = 0
        elif agricultural_income_ratio > pure_farmer_threshold:
            self.consecutive_low_agricultural_ratio += 1
            self.consecutive_high_non_farm_ratio = 0
        else:  # Pluriactive：农业收入占比 30%-70%
            self.consecutive_high_non_farm_ratio = 0
            self.consecutive_low_agricultural_ratio = 0

    def calculate_transition_probability(self, agricultural_income_ratio):
        """
        @theory
        计算生计策略转换概率
        """
        # 基础转换概率
        base_prob = 0.1
        
        # 收入占比漂移因子
        drift_factor = self.calculate_income_share_drift_factor(agricultural_income_ratio)
        
        # 资本因子
        capital_factor = self.calculate_capital_factor()
        
        # 人力资本因子
        human_capital_factor = self.calculate_human_capital_factor()
        
        # 社会资本因子
        social_capital_factor = self.calculate_social_capital_factor()
        
        # 综合转换概率
        transition_prob = base_prob * drift_factor * capital_factor * human_capital_factor * social_capital_factor
        
        return min(transition_prob, 0.5)  # 最大转换概率为50%

    def calculate_income_share_drift_factor(self, agricultural_income_ratio):
        """
        @theory
        计算收入占比漂移因子
        """
        evolution_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('livelihood_evolution', {})
        non_farmer_threshold = evolution_cfg.get('non_farmer_threshold', 0.20)
        pure_farmer_threshold = evolution_cfg.get('pure_farmer_threshold', 0.50)
        if agricultural_income_ratio < non_farmer_threshold and self.consecutive_high_non_farm_ratio >= 2:
            # 连续2年非农收入占比高，向Non_Farmer转换
            return 2.0
        elif agricultural_income_ratio > pure_farmer_threshold and self.consecutive_low_agricultural_ratio >= 2:
            # 连续2年农业收入占比高，向Pure_Farmer转换
            return 2.0
        else:
            return 1.0

    def calculate_capital_factor(self):
        """
        @theory
        计算资本因子
        """
        if self.capital > 20000:
            return 1.2  # 资本充足，转换概率增加
        elif self.capital < 5000:
            return 0.8  # 资本不足，转换概率降低
        else:
            return 1.0

    def calculate_human_capital_factor(self):
        """
        @theory
        计算人力资本因子
        """
        if self.off_farm_wage_level > 1.5:
            return 1.3  # 人力资本高，转换概率增加
        elif self.off_farm_wage_level < 0.8:
            return 0.7  # 人力资本低，转换概率降低
        else:
            return 1.0

    def calculate_social_capital_factor(self):
        """
        @theory
        计算社会资本因子
        """
        if self.is_village_elite:
            return 1.2  # 村精英，转换概率增加
        else:
            return 1.0

    def determine_new_strategy(self, agricultural_income_ratio):
        """
        @theory
        确定新的生计策略
        @fix: 使用新的阈值（20%/50%），与当前CFPS运行口径一致
        """
        evolution_cfg = self.agent_config.get('agents', {}).get('farmer', {}).get('livelihood_evolution', {})
        non_farmer_threshold = evolution_cfg.get('non_farmer_threshold', 0.20)
        pure_farmer_threshold = evolution_cfg.get('pure_farmer_threshold', 0.50)
        if agricultural_income_ratio < non_farmer_threshold:
            return LivelihoodStrategy.NON_FARMER
        elif agricultural_income_ratio > pure_farmer_threshold:
            return LivelihoodStrategy.PURE_FARMER
        else:  # Pluriactive：农业收入占比 30%-70%
            return LivelihoodStrategy.PLURIACTIVE

    def _is_initial_cohort_baseline_observation(self, persistence_cfg=None):
        """Return True when baseline-year livelihood evolution must be frozen.

        The first recorded model year should preserve the empirically
        initialized CFPS/NFPS livelihood distribution. Annual transition rules
        start after that baseline observation; otherwise the 2010 audit row is
        no longer an initialization check.
        """
        cfg = persistence_cfg or {}
        if not bool(cfg.get('freeze_initial_cohort_baseline_year', True)):
            return False
        if bool(getattr(self, 'is_new_entrant', False)):
            return False

        model_obj = getattr(self, 'model', None)
        try:
            base_year = int(getattr(model_obj, 'simulation_base_year', 2010) or 2010)
            step_count = int(getattr(model_obj, 'step_count', 0) or 0)
            entry_year = int(getattr(self, 'entry_year', base_year) or base_year)
            baseline_max_step = int(
                cfg.get('initial_cohort_baseline_observation_max_step', 1) or 1
            )
        except (TypeError, ValueError):
            return False

        return 0 <= step_count <= baseline_max_step and entry_year <= base_year

    def calculate_land_rental_income(self):
        """
        @theory
        计算土地租金收入
        @fix: 从地块状态中恢复已流转的地块，确保租金收入正确计算
        
        @fix 2025-11-29: 修复租金重复计算bug
        问题：patch.rental_price存储的是总价（bid_price_per_mu * area），
              但之前又乘了patch.area，导致租金被重复计算
        修复：rental_price已经是总价，直接累加即可
        """
        # @fix: 从地块状态中恢复已流转的地块列表
        # 确保即使leased_out_patches被清空，也能正确计算租金收入
        patches = (self.model.land_patches if hasattr(self.model, 'initial_household_cohort')
                   else self.land_patches)
        leased_patches = [p for p in patches
                         if rental_rights_holder(p) is self and p.leased_to not in (None, self)]
        
        # 更新leased_out_patches以保持一致性
        self.leased_out_patches = leased_patches
        
        # @fix 2025-11-29: 修正租金计算
        # rental_price已经是该地块的总租金（已包含面积），无需再乘面积
        self.land_rental_income = sum(
            patch.rental_price 
            for patch in leased_patches 
            if patch.rental_price > 0
        )
        if self.model.step_count > 0 and getattr(self.model, "initial_paid_lease_accounts", ()):
            initial_patches = {id(a.patch) for a in self.model.initial_paid_lease_accounts
                               if self.model.step_count <= a.end_step}
            self.land_rental_income -= sum(p.rental_price for p in leased_patches if id(p) in initial_patches)
            self.land_rental_income += sum(
                a.amount_paid_yuan for a in self.model.initial_paid_lease_accounts
                if a.recipient is self and a.settlement_step == self.model.step_count
            )
        if preproduction_cash_enabled(self.model.config):
            self.land_rental_income = sum(
                row['payment_yuan'] for row in getattr(self.model, 'preproduction_rent_ledger', [])
                if row['recipient'] is self and row['step'] == self.model.step_count
                and (row['status'] == 'paid' or row['tenant'] in self.model.farmers)
            ) + sum(a.amount_paid_yuan for a in getattr(self.model, 'initial_paid_lease_accounts', ())
                    if a.recipient is self and a.settlement_step == self.model.step_count)

    def calculate_subsidy_income(self):
        """
        @theory
        计算补贴收入（普通种粮补贴 + 规模种粮补贴）
        
        @new 2025-12-08: 增加规模种粮补贴支持
        - 普通补贴：按种粮面积，所有种粮主体均可获得
        - 规模补贴：种粮面积达到30亩以上的主体额外获得
        
        政策依据：
        1. 浙江省2023年度省级规模种粮动态补贴：50亩以上，20元/亩
        2. 四川省荣县2025年种粮大户补贴：30亩以上，200元/亩
        """
        from core.p3_project_contracts import configured_program
        if configured_program(self.model) is not None:
            self.p3_capital_before_harvest = float(self.capital)
        self.subsidy_income = 0
        self.scale_subsidy_income = 0  # @new: 规模补贴收入
        self.scale_subsidy_qualified = False  # @new: 是否达到规模补贴门槛
        if getattr(self.model.environment, "manual_subsidy_dispatch", False):
            # Government now pays ex post after verification, so settlement starts from zero.
            return
        
        if self.planned_crop in (CropType.GRAIN, CropType.MIXED):
            grain_area = self.grain_area if self.planned_crop == CropType.MIXED else production_area(self)
            if grain_area > 0:
                # 使用新的综合补贴计算接口
                subsidy_info = self.model.environment.get_total_subsidy_for_farmer(self, grain_area)
                self.subsidy_income = subsidy_info['total']
                self.scale_subsidy_income = subsidy_info['scale_subsidy']
                self.scale_subsidy_qualified = subsidy_info['scale_qualified']

    def build_subsidy_claim(self):
        """Build a realized subsidy claim after harvest is observed."""
        from core.p3_project_contracts import configured_program
        if configured_program(self.model) is not None:
            return {'agent': self, 'agent_id': self.unique_id, 'actor_type': 'farmer'}
        if not getattr(self.model.environment, "manual_subsidy_dispatch", False):
            return None

        active_policy = getattr(self.model.environment, 'active_policy', {}) or {}
        subsidy_target = active_policy.get('subsidy_target', 'operator')
        if subsidy_target == 'owner':
            eligible_area = sum(patch.area for patch in self.land_patches if patch.owner == self)
        else:
            eligible_area = self.get_operated_land_area()

        grain_area = self.get_operated_land_area() if self.planned_crop == CropType.GRAIN else 0.0
        eligible_area_breakdown = {}
        grain_area_breakdown = {}
        primary_village = None
        if hasattr(self.model, 'get_owned_area_by_village') and hasattr(self.model, 'get_operated_area_by_village'):
            if subsidy_target == 'owner':
                eligible_area_breakdown = self.model.get_owned_area_by_village(self)
            else:
                eligible_area_breakdown = self.model.get_operated_area_by_village(self)
            if grain_area > 0:
                grain_area_breakdown = self.model.get_operated_area_by_village(self)
            primary_village = self.model.get_primary_village_for_agent(self)
            if eligible_area_breakdown:
                eligible_area = sum(float(value or 0.0) for value in eligible_area_breakdown.values())
            if grain_area_breakdown:
                grain_area = sum(float(value or 0.0) for value in grain_area_breakdown.values())
        if preproduction_cash_enabled(self.model.config):
            held = self.get_operated_land_area()
            ratio = production_area(self) / held if held > 0 else 0.0
            grain_area = production_area(self) if self.planned_crop == CropType.GRAIN else 0.0
            grain_area_breakdown = {k: v * ratio for k, v in grain_area_breakdown.items()}
            if subsidy_target != 'owner':
                eligible_area = production_area(self)
                eligible_area_breakdown = {k: v * ratio for k, v in eligible_area_breakdown.items()}
        if self.planned_crop == CropType.MIXED:
            grain_area = self.grain_area
            grain_area_breakdown = ({next(iter(eligible_area_breakdown)): grain_area}
                                    if subsidy_target != "owner" and len(eligible_area_breakdown) == 1 else {})
        if eligible_area <= 0 and grain_area <= 0:
            return None

        government = getattr(self.model, 'government', None)
        region = government._get_region_label(self) if government is not None else getattr(self, 'region', 'global')
        if primary_village is not None:
            self.primary_village_collective_id = primary_village.unique_id
        claim = {
            'agent': self,
            'agent_id': self.unique_id,
            'actor_type': 'farmer',
            'eligible_area': eligible_area,
            'grain_area': grain_area,
            'region': region,
            'primary_village_id': primary_village.unique_id if primary_village is not None else getattr(self, 'primary_village_collective_id', None),
            'village_area_breakdown': dict(eligible_area_breakdown),
            'village_grain_area_breakdown': dict(grain_area_breakdown),
            'crop_type': getattr(self.planned_crop, 'value', str(self.planned_crop)) if self.planned_crop is not None else None,
            'realized_output': float(getattr(self, 'grain_production', 0.0) or 0.0),
            'scale_qualified': bool(grain_area >= 30),
            'priority_score': (1000.0 if grain_area >= 30 else 0.0) + grain_area,
        }
        self.pending_subsidy_claim = claim
        return claim

    def submit_subsidy_claim(self):
        """Submit the realized claim to the government once annual settlement is complete."""
        from core.p3_project_contracts import configured_program
        if configured_program(self.model) is not None:
            from core.p3_service_settlement import confirm_service_settlement
            confirm_service_settlement(self)
        government = getattr(self.model, 'government', None)
        if government is None or not hasattr(government, 'register_subsidy_claim'):
            return None
        claim = self.build_subsidy_claim()
        if claim is None:
            self.pending_subsidy_claim = None
            return None
        return government.register_subsidy_claim(claim)

    def apply_verified_subsidy_payment(self, cash_amount, basic_cash=0.0, scale_cash=0.0,
                                       in_kind_value=0.0, in_kind_breakdown=None, payment_record=None):
        """Apply an ex-post verified subsidy payment to the farmer's accounts."""
        self.subsidy_income += cash_amount
        self.scale_subsidy_income += scale_cash
        self.scale_subsidy_qualified = self.scale_subsidy_qualified or scale_cash > 0
        if in_kind_value > 0:
            self.receive_in_kind_support(in_kind_value, in_kind_breakdown)

        self.gross_income += cash_amount
        self.total_income = self.gross_income
        self.net_income += cash_amount
        self.cash_net_income = self.net_income
        self.economic_profit += cash_amount
        savings_rate = getattr(self, 'annual_savings_rate', 0.2)
        self.disposable_income = self.net_income * (1 - savings_rate)
        self.capital += cash_amount
        settlement = getattr(self, "last_income_settlement", None)
        if settlement is not None and settlement["step"] == self.model.step_count:
            settlement["verified_support_received_yuan"] += float(cash_amount)
            settlement["capital_after_verified_support_yuan"] = float(self.capital)
        self.calculate_consumption_and_utility()
        self.subsidy_payment_history.append({
            'step': getattr(self.model, 'step_count', 0),
            'cash_amount': cash_amount,
            'basic_cash': basic_cash,
            'scale_cash': scale_cash,
            'in_kind_value': in_kind_value,
            'claim_id': (payment_record or {}).get('claim_id') if payment_record else None,
        })
        self.pending_subsidy_claim = None

    def calculate_off_farm_income(self):
        """
        @theory
        计算非农务工收入
        @fix: 修正非农收入计算，应该乘以实际日工资
        """
        labor_sold_days = max(0.0, float(getattr(self, 'labor_sold_days', 0.0) or 0.0))
        if action_income_enabled(self.model.config):
            profile = self.estimate_production_labor_profile()
            candidate = max(0.0, self.total_labor_units - profile["family_labor_used"] - labor_sold_days)
            self.realized_off_farm_labor_days = self._get_max_off_farm_days(candidate)
            self.off_farm_income = self._calculate_off_farm_income_from_labor_days(candidate)
            return
        candidate_off_farm_days = max(
            0.0,
            float(getattr(self, 'off_farm_labor', 0.0) or 0.0) - labor_sold_days
        )
        self.realized_off_farm_labor_days = candidate_off_farm_days
        self.off_farm_income = self._calculate_off_farm_income_from_labor_days(
            candidate_off_farm_days,
            apply_minimum_guarantee=(labor_sold_days <= 0),
        )

    def calculate_production_cost(self):
        """
        @theory
        计算农业生产成本
        @improvement: P1优先级 - 成本核算精细化
        基于具体投入品价格和数量，而非简单的收入比例
        """
        # 检查是否使用详细成本计算
        production_config = self.model.config.get('production', {})
        cost_structure = production_config.get('cost_structure', {})
        use_detailed = cost_structure.get('use_detailed_cost', True)
        self.cash_production_cost = 0.0
        self.imputed_family_labor_cost = 0.0
        self.production_in_kind_offset = 0.0
        
        if not use_detailed:
            # 使用简单比例（向后兼容）
            if self.planned_crop == CropType.GRAIN:
                self.production_cost = self.agricultural_income * 0.6
            elif self.planned_crop == CropType.CASH_CROP:
                self.production_cost = self.agricultural_income * 0.4
            elif self.planned_crop == CropType.MIXED:
                self.production_cost = (0.6 * self.grain_production * self._get_output_price_per_kg('crop_price')
                                        + 0.4 * self.cash_crop_production * self._get_output_price_per_kg('cash_crop'))
            else:
                self.production_cost = 0
            self.cash_production_cost = float(self.production_cost or 0.0)
        else:
            # 使用详细成本计算
            operated_area = production_area(self)
            if self.planned_crop and operated_area > 0:
                cost_breakdown = self._calculate_detailed_cost(self.planned_crop, operated_area)
                self.production_cost = cost_breakdown['cash_total_cost']
                self.cash_production_cost = cost_breakdown['cash_total_cost']
                self.imputed_family_labor_cost = cost_breakdown['family_labor_imputed_cost']
            else:
                self.production_cost = 0
                self.cash_production_cost = 0.0
                self.imputed_family_labor_cost = 0.0
        
        # 实物支持抵消
        if self.in_kind_support_balance > 0 and self.cash_production_cost > 0:
            offset = min(self.cash_production_cost, self.in_kind_support_balance)
            self.cash_production_cost -= offset
            self.production_cost = self.cash_production_cost
            self.in_kind_support_balance -= offset
            self.production_in_kind_offset = float(offset)

    def _get_family_labor_capacity_for_production(self):
        """Canonical family-labor capacity used by both cost accounting and labor-market demand."""
        if self._national_v1:
            return min(self.total_labor_units, self.v1_laborers * self.agricultural_days_per_worker)
        total_labor_units = float(getattr(self, 'total_labor_units', 365.0) or 365.0)
        if action_income_enabled(self.model.config):
            total_labor_units = max(0.0, float(self.total_labor_units))
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            return total_labor_units * 0.85
        if self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            return total_labor_units * 0.50
        return total_labor_units * 0.15

    def estimate_production_labor_profile(self, crop_type=None, area=None, labor_days_per_mu=None):
        """
        Build a single-source labor profile for farmer production.

        This keeps hired-labor demand consistent across:
        - farmer-side cost accounting
        - labor-market demand registration
        - service-substitution effects
        """
        crop_type = crop_type if crop_type is not None else getattr(self, 'planned_crop', None)
        operated_area = float(
            production_area(self) if area is None else area
        )
        profile = {
            'crop_type': crop_type,
            'area': max(0.0, operated_area),
            'labor_days_per_mu': 0.0,
            'total_labor_need': 0.0,
            'available_family_labor': 0.0,
            'family_labor_used': 0.0,
            'labor_gap': 0.0,
            'service_substitution_rate': 0.0 if self._national_v1 else max(
                0.0,
                min(0.8, float(getattr(self, 'labor_reduction_ratio', 0.0) or 0.0))
            ),
        }
        if crop_type is None or operated_area <= 0:
            return profile

        if labor_days_per_mu is None:
            if crop_type == CropType.MIXED:
                share = self.household_planned_grain_share
                rate = sum(weight * self.estimate_production_labor_profile(crop, operated_area)['labor_days_per_mu']
                           for crop, weight in ((CropType.GRAIN, share), (CropType.CASH_CROP, 1-share)))
                return self.estimate_production_labor_profile(crop_type, operated_area, rate)
            cost_structure = self.model.config.get('production', {}).get('cost_structure', {})
            input_requirements = cost_structure.get('input_requirements', {})
            labor_days = input_requirements.get('labor_days', {})
            if crop_type == CropType.GRAIN:
                labor_days_per_mu = labor_days.get('grain', 10)
                if action_income_enabled(self.model.config):
                    labor_days_per_mu = labor_days.get('grain_farmer', labor_days.get('grain', 12))
            else:
                labor_days_per_mu = labor_days.get('cash_crop', 25)
                if action_income_enabled(self.model.config):
                    labor_days_per_mu = labor_days.get('cash_crop_farmer', labor_days.get('cash_crop', 30))

            scale_adjustments = self._get_scale_operation_cost_adjustments(crop_type, operated_area)
            labor_days_per_mu *= scale_adjustments['labor_multiplier']

        labor_days_per_mu = max(0.0, float(labor_days_per_mu or 0.0))
        total_labor_need = labor_days_per_mu * operated_area
        available_family_labor = self._get_family_labor_capacity_for_production()
        family_labor_used = min(total_labor_need, available_family_labor)
        labor_gap = max(0.0, total_labor_need - available_family_labor)

        profile.update({
            'labor_days_per_mu': labor_days_per_mu,
            'total_labor_need': total_labor_need,
            'available_family_labor': available_family_labor,
            'family_labor_used': family_labor_used,
            'labor_gap': labor_gap,
        })
        return profile

    def _get_scale_operation_cost_adjustments(self, crop_type, area):
        """
        职业化农户可以通过部分机械化与托管服务降低劳动强度，
        不能永远按纯小农、纯人工的单位成本口径计算。
        """
        operated_area = max(float(area or 0.0), float(production_area(self) or 0.0))
        regional_params = self._get_regional_land_transfer_params()
        mech_potential = float(regional_params.get('mechanization_potential', 0.7) or 0.7)
        farming_ability = float(getattr(self, 'farming_ability', 1.0) or 1.0)
        service_reduction = max(
            0.0,
            min(0.55, float(getattr(self, 'labor_reduction_ratio', 0.0) or 0.0))
        )

        retention_cfg = (
            self.agent_config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        professional_threshold = float(retention_cfg.get('min_operated_area', 30) or 30)

        if crop_type == CropType.GRAIN:
            scale_progress = max(
                0.0,
                min(1.0, (operated_area - 10.0) / max(20.0, professional_threshold * 2.5))
            )
            scale_reduction = 0.55 * scale_progress * mech_potential
            min_labor_multiplier = 0.22
            min_tools_multiplier = 0.45
            min_other_multiplier = 0.88
            min_input_multiplier = 0.86
        else:
            scale_progress = max(
                0.0,
                min(1.0, (operated_area - 10.0) / max(30.0, professional_threshold * 3.0))
            )
            scale_reduction = 0.22 * scale_progress * mech_potential
            min_labor_multiplier = 0.55
            min_tools_multiplier = 0.60
            min_other_multiplier = 0.92
            min_input_multiplier = 0.92

        if operated_area >= professional_threshold:
            scale_reduction += 0.10 * mech_potential
        if operated_area >= professional_threshold * 2.0:
            scale_reduction += 0.15 * mech_potential

        skill_reduction = max(0.0, min(0.08, (farming_ability - 1.0) * 0.20))
        input_multiplier = max(
            min_input_multiplier,
            1.0 - 0.12 * scale_progress * mech_potential - (0.05 if operated_area >= professional_threshold * 2.0 else 0.0)
        )

        labor_multiplier = max(
            min_labor_multiplier,
            1.0 - scale_reduction - service_reduction - skill_reduction
        )
        tools_multiplier = max(
            min_tools_multiplier,
            1.0 - 0.50 * scale_reduction - 0.40 * service_reduction
        )
        other_multiplier = max(
            min_other_multiplier,
            1.0 - 0.10 * scale_progress * mech_potential
        )

        return {
            # Per-mu crop budgets already include mechanized work; no task-specific saving identified.
            'labor_multiplier': 1.0 if self._national_v1 else float(labor_multiplier),
            'tools_multiplier': float(tools_multiplier),
            'other_multiplier': float(other_multiplier),
            'input_multiplier': float(input_multiplier),
        }
    
    def _calculate_detailed_cost(self, crop_type, area):
        """
        @theory
        精细化成本核算：基于具体投入品
        @improvement: P1优先级
        @fix 2025-11-29: 添加劳动力约束检查和雇佣机制
        
        农户生产特点：
        1. 主要使用家庭劳动力
        2. 不使用大型机械，只使用小型农具
        3. 大户（经营面积>30亩）可能需要雇佣帮工
        """
        # 获取配置参数
        cost_structure = self.model.config.get('production', {}).get('cost_structure', {})
        input_prices = cost_structure.get('input_prices', {})
        input_requirements = cost_structure.get('input_requirements', {})
        
        # 1. 获取投入品价格（从市场或环境获取）
        seed_prices = input_prices.get('seed', {})
        fertilizer_price = input_prices.get('fertilizer', 2.5)
        pesticide_price = input_prices.get('pesticide', 15.0)
        # 农户不使用大型机械，使用小型农具成本较低
        small_tools_price = input_prices.get('small_tools', 20.0)  # 小型农具成本
        
        # @fix: 农业劳动力成本应使用农业工资
        if hasattr(self.model, 'labor_market'):
            labor_price = self.model.labor_market.current_agricultural_wage
        elif hasattr(self.model, 'environment'):
            non_farm_wage = self.model.environment.get_market_price('wage_rate')
            labor_price = non_farm_wage * 0.7  # 农业工资约为非农工资的70%
        else:
            base_wage = self._get_base_non_farm_wage()
            labor_price = base_wage * 0.7
        
        # 2. 获取投入品需求量
        if crop_type == CropType.MIXED:
            share = self.household_planned_grain_share
            inputs = 0.0
            for crop, weight in ((CropType.GRAIN, share), (CropType.CASH_CROP, 1-share)):
                cost = copy(self)._calculate_detailed_cost(crop, area)
                inputs += weight * (cost["cash_total_cost"] - cost["hired_labor_cost"])
            profile = self.estimate_production_labor_profile(crop_type, area)
            self.family_labor_used = profile["family_labor_used"]
            self.requested_hired_labor_days = profile["labor_gap"]
            self.labor_constraint_binding = self.requested_hired_labor_days > 0
            if getattr(self, "fixed_rule_labor_settled", False):
                hired = self.hired_labor_cost
            else:
                self.hired_labor_days = self.requested_hired_labor_days
                hired = self.hired_labor_days * labor_price
            family_cost = self.family_labor_used * labor_price * 0.5
            return dict(cash_total_cost=inputs + hired, family_labor_imputed_cost=family_cost,
                        economic_total_cost=inputs + hired + family_cost, hired_labor_cost=hired)
        requirements = input_requirements.get('seed', {})
        labor_days_cfg = input_requirements.get('labor_days', {})
        if crop_type == CropType.GRAIN:
            seed_required = requirements.get('grain', 30)  # 斤/亩
            fertilizer_required = input_requirements.get('fertilizer', {}).get('grain', 100)  # 斤/亩
            pesticide_required = input_requirements.get('pesticide', {}).get('grain', 2)  # 斤/亩
            # 小农默认劳动强度更高，但职业化农户会通过部分机械化/托管降本
            labor_days_required = labor_days_cfg.get('grain_farmer', labor_days_cfg.get('grain', 12))  # 天/亩（比企业高）
            tools_usage = input_requirements.get('small_tools', {}).get('grain', 1)  # 次/亩
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('grain', 200)  # 元/亩
            seed_price = seed_prices.get('grain', 5.0)
        else:  # CASH_CROP
            seed_required = requirements.get('cash_crop', 5)  # 斤/亩
            fertilizer_required = input_requirements.get('fertilizer', {}).get('cash_crop', 150)  # 斤/亩
            pesticide_required = input_requirements.get('pesticide', {}).get('cash_crop', 5)  # 斤/亩
            # 经济作物需要更多人工
            labor_days_required = labor_days_cfg.get('cash_crop_farmer', labor_days_cfg.get('cash_crop', 30))  # 天/亩
            tools_usage = input_requirements.get('small_tools', {}).get('cash_crop', 2)  # 次/亩
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('cash_crop', 300)  # 元/亩
            seed_price = seed_prices.get('cash_crop', 8.0)

        scale_adjustments = self._get_scale_operation_cost_adjustments(crop_type, area)
        seed_required *= scale_adjustments['input_multiplier']
        fertilizer_required *= scale_adjustments['input_multiplier']
        pesticide_required *= scale_adjustments['input_multiplier']
        labor_days_required *= scale_adjustments['labor_multiplier']
        tools_usage *= scale_adjustments['tools_multiplier']
        other_cost *= scale_adjustments['other_multiplier']

        # 3. 劳动力约束检查
        labor_profile = self.estimate_production_labor_profile(
            crop_type=crop_type,
            area=area,
            labor_days_per_mu=labor_days_required,
        )
        total_labor_need = labor_profile['total_labor_need']
        available_family_labor = labor_profile['available_family_labor']
        self.family_labor_used = labor_profile['family_labor_used']

        # 4. 计算劳动力缺口和雇佣成本
        labor_gap = labor_profile['labor_gap']
        self.requested_hired_labor_days = labor_gap

        if labor_gap > 0:
            self.labor_constraint_binding = True
            # 仅在劳动力市场尚未结算时保留回退估算，不覆盖已撮合出的实际雇工量。
            if not getattr(self, "fixed_rule_labor_settled", False) and not (hasattr(self, 'hired_labor_cost') and float(getattr(self, 'hired_labor_cost', 0.0) or 0.0) > 0):
                self.hired_labor_days = labor_gap
        else:
            self.labor_constraint_binding = False
            self.hired_labor_days = 0
        
        # 5. 计算各项成本
        seed_cost = seed_price * seed_required * area
        fertilizer_cost = fertilizer_price * fertilizer_required * area
        pesticide_cost = pesticide_price * pesticide_required * area
        
        # 家庭劳动力隐性成本（机会成本，按市场工资的50%计算）
        family_labor_cost = self.family_labor_used * labor_price * 0.5
        
        # 雇佣劳动力显性成本（如果劳动力市场已结算，使用记录值）
        if getattr(self, "fixed_rule_labor_settled", False) or (hasattr(self, 'hired_labor_cost') and self.hired_labor_cost > 0):
            hired_labor_cost = self.hired_labor_cost
        else:
            # 如果劳动力市场尚未结算，估算雇佣成本
            hired_labor_cost = self.hired_labor_days * labor_price
        
        # 农户使用小型农具而非大型机械
        tools_cost = small_tools_price * tools_usage * area
        other_costs = other_cost * area
        
        cash_total_cost = (
            seed_cost + fertilizer_cost + pesticide_cost +
            hired_labor_cost + tools_cost + other_costs
        )
        economic_total_cost = cash_total_cost + family_labor_cost
        
        return {
            'cash_total_cost': float(cash_total_cost),
            'family_labor_imputed_cost': float(family_labor_cost),
            'economic_total_cost': float(economic_total_cost),
            'hired_labor_cost': float(hired_labor_cost),
        }

    def calculate_land_rental_expense(self):
        """
        @theory
        计算土地租金支出
        @fix: 从地块状态中恢复已租入的地块，确保租金支出正确计算
        
        @fix 2025-11-29: 修复租金重复计算bug
        问题：patch.rental_price存储的是总价（bid_price_per_mu * area），
              但之前又乘了patch.area，导致租金被重复计算
        修复：rental_price已经是总价，直接累加即可
        """
        # @fix: 从地块状态中恢复已租入的地块列表
        leased_in = [p for p in self.model.land_patches 
                    if p.leased_to == self]
        self.leased_in_patches = leased_in
        
        # @fix 2025-11-29: 修正租金计算
        # rental_price已经是该地块的总租金（已包含面积），无需再乘面积
        self.land_rental_expense = sum(
            patch.rental_price 
            for patch in leased_in 
            if patch.rental_price > 0
        )
        budget = getattr(self, 'last_preproduction_cash_account', {})
        if preproduction_cash_enabled(self.model.config) and budget.get('step') == self.model.step_count:
            self.land_rental_expense = budget['ordinary_rent_reserved_yuan']

    def get_operated_land_area(self):
        """
        @theory
        获取经营土地面积
        @fix: 正确计算经营面积，排除已流转的地块，包括租入的地块
        @fix 2026-06: 以显式地块账本为准；current_operator 代表实际经营权，
        microdata operated-area hint 只在没有任何地块账本时兜底，避免全国面积口径膨胀。
        """
        if hasattr(self, "_fixed_rule_preview_area"):
            return self._fixed_rule_preview_area
        model = getattr(self, 'model', None)
        cache_version = getattr(model, 'land_tenure_version', None)
        cached_area = getattr(self, '_operated_land_area_cache', 0.0)
        cached_version = getattr(self, '_operated_land_area_cache_version', None)
        if cache_version is not None and cached_version == cache_version:
            return float(cached_area)

        model_patches = list(getattr(model, 'land_patches', []) or []) if model is not None else []
        local_candidates = []
        local_candidates.extend(getattr(self, 'land_patches', None) or [])
        local_candidates.extend(getattr(self, 'leased_in_patches', None) or [])
        local_candidates.extend(getattr(self, 'leased_patches', None) or [])
        explicit_patch_state_available = bool(model_patches or local_candidates)

        candidates = []
        candidates.extend(local_candidates)
        candidates.extend(model_patches)

        operated_patches = []
        leased_in = []
        leased_out = []
        seen_patch_ids = set()
        for patch in candidates:
            patch_key = id(patch)
            if patch_key in seen_patch_ids:
                continue
            seen_patch_ids.add(patch_key)

            owner = getattr(patch, 'owner', None)
            leased_to = getattr(patch, 'leased_to', None)
            current_operator = getattr(patch, 'current_operator', None)

            if owner is self and leased_to is not None and leased_to is not self and current_operator is not self:
                leased_out.append(patch)

            owned_and_not_leased = (
                owner is self
                and leased_to in (None, self)
                and current_operator in (None, self)
            )
            operated_by_self = current_operator is self or leased_to is self or owned_and_not_leased
            if not operated_by_self:
                continue
            operated_patches.append(patch)
            if leased_to is self:
                leased_in.append(patch)

        operated_area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in operated_patches)
        self.leased_out_patches = list(leased_out)
        self.leased_in_patches = list(leased_in)
        self.leased_patches = list(leased_in)

        # Microdata-driven initial states may encode rent-in/rent-out areas before
        # explicit patch-level contracts are constructed. Once explicit patches exist,
        # patch-level ownership/operator state is the accounting authority.
        if not explicit_patch_state_available:
            try:
                operated_area_hint = float(getattr(self, '_microdata_operated_area_hint', 0.0) or 0.0)
            except (TypeError, ValueError):
                operated_area_hint = 0.0
            if operated_area_hint > 0:
                operated_area = float(operated_area_hint)

        self._operated_land_area_cache = float(operated_area)
        self._operated_land_area_cache_version = cache_version
        return float(operated_area)
    
    def get_land_transfer_ratio(self):
        """
        @theory
        计算土地流转比例（已流转土地面积 / 自有土地总面积）
        用于多维度生计类型演化判断
        """
        total_owned_area = sum(patch.area for patch in self.land_patches if patch.owner == self)
        if total_owned_area <= 0:
            return 0.0
        
        leased_out_area = sum(patch.area for patch in self.leased_out_patches)
        if leased_out_area <= 0:
            try:
                leased_out_area = max(
                    0.0,
                    float(getattr(self, 'land_rent_out_area', 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                leased_out_area = 0.0
        return leased_out_area / total_owned_area
    
    def check_off_farm_opportunity(self):
        """
        @theory
        检查非农就业机会是否充足
        基于劳动力市场的供需关系
        @fix 2024-11-19: 提高工资水平要求，从80%提高到90%，保护纯农户
        """
        # 获取劳动力市场状态
        if hasattr(self.model, 'environment'):
            # 简化：基于工资水平判断（工资高说明需求大）
            current_wage = self.model.environment.get_market_price('wage_rate')
            base_wage = self._get_base_non_farm_wage()
            
            # @fix 2024-11-19: 提高工资水平要求，从80%提高到90%，保护纯农户
            if current_wage >= base_wage * 0.9:  # 工资不低于基础工资的90%
                return True
        
        return False
    
    def check_off_farm_income_sufficient(self):
        """
        @theory
        检查非农收入是否足够维持基本生活
        参考：最低生活保障标准
        @fix 2025-11-26: 进一步提高非农收入门槛，从12000元提高到20000元
        参考2024年农村人均可支配收入约20000元，只有非农收入达到这个水平
        才能认为有足够的能力成为非农户
        """
        min_living_standard = 20000  # 最低生活保障（元/年），从12000提高到20000
        return self.off_farm_income >= min_living_standard

    def get_owned_land_area(self, include_leased_out=True):
        area = sum(patch.area for patch in self.land_patches if patch.owner == self)
        if not include_leased_out:
            area -= sum(patch.area for patch in self.leased_out_patches)
        return max(area, 0)

    def sync_owned_land_area(self):
        """Keep compatibility land-area fields aligned with patch ownership."""
        self.land_area_mu = float(self.get_owned_land_area(include_leased_out=True))
        return self.land_area_mu

    def receive_in_kind_support(self, value, breakdown=None):
        if value <= 0:
            return
        self.in_kind_support_balance += value
        if breakdown:
            for key, amount in breakdown.items():
                self.in_kind_support_breakdown[key] = self.in_kind_support_breakdown.get(key, 0) + amount

    def get_income_breakdown(self):
        """
        @theory
        获取收入构成
        """
        return {
            'agricultural_income': self.agricultural_income,
            'land_rental_income': self.land_rental_income,
            'subsidy_income': self.subsidy_income,
            'off_farm_income': self.off_farm_income,
            'labor_sale_income': self.labor_sale_income,
            'gross_income': self.gross_income,
            'total_income': self.total_income,
            'cash_production_cost': self.cash_production_cost,
            'imputed_family_labor_cost': self.imputed_family_labor_cost,
            'economic_total_expense': self.economic_total_expense,
            'net_income': self.net_income,
            'cash_net_income': self.cash_net_income,
            'economic_profit': self.economic_profit,
            'disposable_income': self.disposable_income
        }
    
    def calculate_consumption_and_utility(self):
        """
        @theory
        计算消费和效用
        基于理论框架：U = U_food(C_essential, C_luxury) + U_leisure(L_leisure)
        """
        paid = getattr(self, "last_consumption_account", {})
        if paid.get("step") == self.model.step_count and paid.get("payment_rule") == "cash_limited_no_credit":
            # A later subsidy changes funds, not consumption already paid in this year.
            grain_price = self.model.environment.get_market_price('crop_price')
            cash_price = self.model.environment.get_market_price('cash_crop')
            self.consumption_essential = self.food_consumption_details['grain_expenditure'] / grain_price if grain_price > 0 else 0.0
            self.consumption_luxury = self.food_consumption_details['other_food_expenditure'] / cash_price if cash_price > 0 else 0.0
            self.food_utility = self.consumption_essential ** 0.7 * self.consumption_luxury ** 0.3 if self.consumption_luxury > 0 else self.consumption_essential ** 0.7
            self.leisure_time = max(0, self.total_labor_units - self.agricultural_labor - self.off_farm_labor)
            leisure_component = 100 * np.log(1 + self.leisure_time) if self.leisure_time > 0 else 0.0
            self.total_utility = self.food_utility + leisure_component
            self.welfare_index = min(1.0, self.total_utility / 10000)
            return
        # 1. 计算闲暇时间
        # 总时间 = 365天，减去农业劳动和非农劳动
        self.leisure_time = max(0, self.total_labor_units - self.agricultural_labor - self.off_farm_labor)
        
        # 2. 消费决策：在可支配收入约束下分配消费预算
        if self.disposable_income <= 0:
            # 如果可支配收入为0或负，只消费基本需求
            self.consumption_essential = 0
            self.consumption_luxury = 0
            self.food_utility = 0
            self.leisure_utility = 0
            self.total_utility = 0
            self.welfare_index = 0
            return
        
        # 获取市场价格
        grain_price = self.model.environment.get_market_price('crop_price')
        cash_crop_price = self.model.environment.get_market_price('cash_crop')
        
        # 基本需求食物价格（假设为粮食价格）
        essential_price = grain_price
        # 高端食物价格（假设为经济作物价格，通常更高）
        luxury_price = cash_crop_price
        
        # 3. 计算高端食物偏好系数
        # 基于户主年龄和城市经历（如果有这些属性）
        # 简化：基于生计策略推断
        if hasattr(self, 'age'):
            age_factor = max(0, 1 - abs(self.age - 45) / 30)  # 45岁左右偏好最高
        else:
            age_factor = 0.5  # 默认值
        
        if hasattr(self, 'urban_experience'):
            urban_factor = min(1.0, self.urban_experience / 10)  # 城市经历增加偏好
        else:
            # 基于生计策略推断：Non_Farmer有更多城市经历
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                urban_factor = 0.8
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                urban_factor = 0.5
            else:
                urban_factor = 0.2
        
        # 高端食物偏好系数
        beta_luxury = 0.3 + 0.4 * urban_factor + 0.2 * age_factor
        beta_luxury = max(0.1, min(0.8, beta_luxury))  # 限制在0.1-0.8之间
        
        # 4. 消费预算分配
        # 基本需求食物：至少满足最低需求，剩余按偏好分配
        min_essential_consumption = 1000  # 最低基本需求（元）
        essential_budget = min_essential_consumption + (self.disposable_income - min_essential_consumption) * (1 - beta_luxury)
        essential_budget = max(min_essential_consumption, min(essential_budget, self.disposable_income * 0.9))
        
        luxury_budget = self.disposable_income - essential_budget
        luxury_budget = max(0, luxury_budget)
        
        # 计算消费量（假设单位价格为1，简化处理）
        self.consumption_essential = essential_budget / essential_price if essential_price > 0 else 0
        self.consumption_luxury = luxury_budget / luxury_price if luxury_price > 0 else 0
        
        # 5. 计算食品消费效用（Cobb-Douglas形式）
        # U_food = (C_essential^alpha) * (C_luxury^(1-alpha))
        alpha = 0.7  # 基本需求食物权重
        if self.consumption_essential > 0 and self.consumption_luxury > 0:
            self.food_utility = (self.consumption_essential ** alpha) * (self.consumption_luxury ** (1 - alpha))
        elif self.consumption_essential > 0:
            self.food_utility = self.consumption_essential ** alpha
        else:
            self.food_utility = 0
        
        # 6. 计算闲暇效用（对数形式）
        # U_leisure = gamma * log(1 + L_leisure)
        gamma = 100  # 闲暇效用系数
        self.leisure_utility = gamma * np.log(1 + self.leisure_time) if self.leisure_time > 0 else 0
        
        # 7. 计算总效用
        # U_total = U_food + U_leisure
        self.total_utility = self.food_utility + self.leisure_utility
        
        # 8. 计算福利指数（归一化到0-1）
        # 基于总效用，使用对数变换归一化
        max_utility = 10000  # 假设的最大效用值
        self.welfare_index = min(1.0, self.total_utility / max_utility) if max_utility > 0 else 0 
    
    def make_consumption_decision(self):
        """
        @theory
        农户消费决策
        基于扩展线性支出系统(ELES)和Stone-Geary效用函数
        在年度结算后执行，根据可支配收入决定消费支出
        
        @new 2025-12: 扩展加入闲暇效用计算（AHM框架）
        @critical 2025-12: 考虑粮食自给部分
        """
        if action_income_enabled(self.model.config) and getattr(self, "_consumption_decision_step", None) == self.model.step_count:
            return
        payment_config = self.model.config.get("agents", {}).get("farmer", {}).get("consumption_payment", {})
        payment_rule = payment_config.get("mode", "full_demand")
        if payment_rule not in {"full_demand", "cash_limited_no_credit"}:
            raise ValueError("Unknown household consumption payment mode: " + str(payment_rule))
        no_credit = payment_rule == "cash_limited_no_credit"
        if no_credit and (not action_income_enabled(self.model.config) or
                payment_config.get("funds_basis") != "capital_as_liquid_operating_balance"):
            raise ValueError("cash_limited_no_credit requires action-income and explicit capital_as_liquid_operating_balance")
        current_capital = float(np.real(self.capital)) if isinstance(self.capital, complex) else float(self.capital)
        if no_credit and (isinstance(self.capital, complex) or not np.isfinite(current_capital)):
            raise ValueError("cash_limited_no_credit requires a finite real signed capital balance")
        # 获取当前粮食价格
        grain_price = self.model.environment.get_market_price('crop_price')
        
        # @critical 2025-12: 计算自产粮食用于自给
        # 只有种粮的农户才有自给粮
        self_produced_grain_kg = 0
        # Action-income harvest books all output as sales; no retained inventory exists.
        if not action_income_enabled(self.model.config) and hasattr(self, 'last_crop_choice') and self.last_crop_choice == CropType.GRAIN:
            if hasattr(self, 'last_yield') and self.last_yield > 0:
                # last_yield记录的是实物产量(kg)，不是货币收入。
                total_grain_kg = float(self.last_yield)
                
                # 自给率：根据户主年龄和家庭规模决定
                # 年轻家庭自给率低（商品化程度高），老年家庭自给率高
                base_self_sufficiency = 0.4  # 基础自给率40%
                age_factor = min(0.3, max(0, (self.age - 40) * 0.01))  # 年龄越大自给率越高
                self_sufficiency_rate = min(0.8, base_self_sufficiency + age_factor)
                
                self_produced_grain_kg = total_grain_kg * self_sufficiency_rate
        
        # 计算消费束和福利指标（传入自给粮）
        self.consumption_bundle, self.food_consumption_details, self.welfare_indicators = \
            self.consumption_model.calculate_consumption_bundle(
                household_income=self.disposable_income,
                household_size=self.household_size,
                grain_price=grain_price,
                self_produced_grain_kg=self_produced_grain_kg  # @critical 2025-12: 传入自给粮
            )
        demand_bundle = dict(self.consumption_bundle)
        consumption_demand = float(sum(demand_bundle.values()))
        payment = min(consumption_demand, max(0.0, current_capital)) if no_credit else None
        if no_credit:
            self.consumption_bundle, self.food_consumption_details, self.welfare_indicators = \
                self.consumption_model.calculate_consumption_bundle(
                    household_income=self.disposable_income, household_size=self.household_size,
                    grain_price=grain_price, self_produced_grain_kg=0.0, cash_budget=payment)
        
        # @new 2025-12: 计算闲暇效用（AHM模型扩展）
        # 获取实际工作天数
        labor_days = getattr(self, 'agricultural_labor_days', 100) + \
                     getattr(self, 'off_farm_labor_days', 100)
        
        # 计算闲暇效用
        self.leisure_indicators = self.consumption_model.calculate_leisure_utility(
            labor_days_worked=labor_days,
            leisure_preference=0.3
        )
        
        # 计算综合效用（消费 + 闲暇）
        self.comprehensive_utility = self.consumption_model.calculate_comprehensive_utility(
            household_income=self.disposable_income,
            household_size=self.household_size,
            grain_price=grain_price,
            labor_days_worked=labor_days,
            consumption_welfare=self.welfare_indicators if no_credit else None,
        )
        
        # 提取关键指标到智能体属性（便于数据收集）
        self.grain_consumption_kg = self.food_consumption_details.get('grain_consumption_kg', 0)
        self.grain_expenditure = self.food_consumption_details.get('grain_expenditure', 0)
        self.total_consumption_expenditure = self.welfare_indicators.get('total_consumption', 0)
        self.engel_coefficient = self.welfare_indicators.get('engel_coefficient', 0)
        self.food_security_index = self.welfare_indicators.get('food_security_index', 0)
        self.consumer_surplus = self.welfare_indicators.get('consumer_surplus', 0)
        self.is_in_poverty = self.welfare_indicators.get('is_in_poverty', False)
        self.malnutrition_risk = self.welfare_indicators.get('malnutrition_risk', False)
        
        # @new 2025-12: 提取闲暇效用指标
        self.leisure_days = self.leisure_indicators.get('leisure_days', 0)
        self.leisure_utility = self.leisure_indicators.get('leisure_utility', 0)
        self.is_overworked = self.leisure_indicators.get('is_overworked', False)
        self.quality_of_life_index = self.comprehensive_utility.get('quality_of_life_index', 50)
        self.work_life_balance = self.comprehensive_utility.get('work_life_balance', '一般')
        
        # 更新资本（消费支出后的剩余）
        # 确保capital和consumer_surplus都是实数，避免复数运算
        surplus = float(np.real(self.consumer_surplus)) if isinstance(self.consumer_surplus, complex) else float(self.consumer_surplus)
        from core.p3_project_contracts import configured_program
        if no_credit:
            self.capital = current_capital - payment
        elif action_income_enabled(self.model.config) or configured_program(self.model) is not None:
            self.capital = current_capital - self.total_consumption_expenditure
        else:
            self.capital = current_capital + surplus
        if action_income_enabled(self.model.config):
            self._consumption_decision_step = self.model.step_count
            # ELES is a demand schedule, not a cash-constrained payment or a loan.
            self.last_consumption_account = {
                "step": self.model.step_count,
                "consumption_demand_yuan": consumption_demand,
                "available_cash_before_yuan": max(0.0, current_capital),
                "signed_balance_before_yuan": current_capital,
                "signed_balance_after_yuan": float(self.capital),
                "consumption_funding_gap_yuan": max(0.0, consumption_demand - max(0.0, current_capital)),
                "actual_cash_payment_yuan": payment,
                "unmet_consumption_yuan": consumption_demand - payment if no_credit else None,
                "credit_received_yuan": 0.0 if no_credit else None,
                "payment_rule": payment_rule,
                "funds_basis": payment_config.get("funds_basis") if no_credit else None,
                "basis": "paid_proportional_demand_no_new_credit" if no_credit else "full_demand_charge_no_payment_or_credit_mechanism",
                "demand_by_category_yuan": {k.value:float(v) for k,v in demand_bundle.items()},
                "paid_by_category_yuan": {k.value:float(v) for k,v in self.consumption_bundle.items()} if no_credit else None,
                "unmet_by_category_yuan": {k.value:float(v - self.consumption_bundle[k]) for k,v in demand_bundle.items()} if no_credit else None,
            }
            if no_credit:
                self.calculate_consumption_and_utility()
        settlement = getattr(self, "last_income_settlement", None)
        if settlement is not None and settlement["step"] == self.model.step_count:
            settlement["consumption_yuan"] = float(self.total_consumption_expenditure)
            settlement["capital_before_consumption_yuan"] = current_capital
            settlement["capital_after_consumption_yuan"] = float(self.capital)
            settlement["consumption_funding_gap_yuan"] = self.last_consumption_account["consumption_funding_gap_yuan"]
            settlement["consumption_actual_payment_yuan"] = payment
            settlement["consumption_unmet_yuan"] = self.last_consumption_account["unmet_consumption_yuan"]
