# @refactor: 新增环境类，作为智能体感知世界的中介，解耦
# --- core/environment.py ---
import numpy as np
from core.config_utils import (
    DEFAULT_BASE_NON_FARM_WAGE,
    resolve_base_non_farm_wage,
    sync_base_non_farm_wage,
)
from core.reproducibility import np_choice, np_random_value, np_uniform
from utils.constants import JIN_PER_KG
from utils.ai_runtime import (
    information_governance_params,
    mechanism_enabled,
    productivity_params,
)
from utils.policy_runtime import normalize_grain_subsidy_aliases, resolve_fiscal_system_config

class Environment:
    """
    环境类，负责管理全局状态和市场信息。
    智能体通过查询环境来获取信息，而不是直接访问其他智能体。
    
    @theory:
    实现真正的市场出清机制，使价格内生化，形成跨期反馈循环
    """
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.model = None
        self.random = None
        self.np_random = None
        
        # @fix 2024-11-20: 注入MechanismCoordinator单例
        # 专家审阅意见：将MechanismCoordinator作为单例注入到Environment中
        # 让所有Agent都能访问统一的机制逻辑，避免重复实例化
        from core.mechanisms import MechanismCoordinator
        # 尝试从config获取机制配置，如果没有则使用默认路径
        mechanism_config = config.get('mechanisms', {})
        if mechanism_config:
            self.mechanism_coordinator = MechanismCoordinator(config_dict=mechanism_config)
        else:
            self.mechanism_coordinator = MechanismCoordinator()
        
        # @fix 2024-11-20: 移除硬编码参数，所有参数必须从config流入
        # 专家审阅意见：硬编码切断了模型与配置文件的联系，导致敏感性分析无效
        # 如果需要在特定实验中固定值，应在runner.py的配置加载阶段处理
        # 确保配置存在（如果不存在则使用默认值，但不强制覆盖）
        if 'environment' not in self.config:
            self.config['environment'] = {}
        if 'market' not in self.config['environment']:
            self.config['environment']['market'] = {}
        canonical_base_wage = sync_base_non_farm_wage(
            self.config, default=DEFAULT_BASE_NON_FARM_WAGE
        )
        # @fix 2025-11-30: 移除需求值强制修正限制
        # 原限制导致配置中的合理需求值被错误覆盖
        # 实际供给：粮食~3000吨，经作~13000吨，需求应与之匹配
        if 'base_grain_demand' not in self.config['environment']['market']:
            # 默认值：根据实际供给量设置，保持供需比约1.5（略供不应求鼓励种粮）
            self.config['environment']['market']['base_grain_demand'] = 5000000  # 5000吨
        
        if 'base_cash_crop_demand' not in self.config['environment']['market']:
            # 默认值：根据实际供给量设置，保持供需比约1.0（供需平衡）
            self.config['environment']['market']['base_cash_crop_demand'] = 12000000  # 12000吨

        # 政策状态
        self.policy = {}
        self.active_policy = {}
        self.active_policy_id = None
        self.payment_lag = 0
        self.manual_subsidy_dispatch = False

        # 市场状态 (内生化)
        # @improvement: P0优先级 - 价格将内生决定，初始值仅作为起点
        market_config = self.config.get('environment', {}).get('market', {})
        base_prices = market_config.get('base_prices', {})
        self.market_prices = {
            # @calibration: 根据审评建议和国家公布的2020年农产品价格统计校准价格
            # 2020年粮食平均收购价每公斤2.5元（即1.25元/斤）完全吻合
            # 注意：这些是初始价格，将被内生价格机制覆盖
            'crop_price': base_prices.get('grain', 1.25),  # 粮食价格 (元/斤)
            # 根据农业细分数据测算，劳动密集型农业（主要种经济作物）亩均产值约为土地密集型（主要种粮食）的2.76倍
            # 据此推算，经济作物价格应在2.4元/斤左右更为合理
            'cash_crop': base_prices.get('cash_crop', 1.8), # 经济作物价格 (元/斤) @fix: 默认值应与配置文件一致(1.8)
            'wage_rate': market_config.get(
                'base_non_farm_wage', DEFAULT_BASE_NON_FARM_WAGE
            )
        }
        self.market_prices['wage_rate'] = canonical_base_wage
        self.market_prices['land_rent'] = market_config.get('base_land_rent', 280.0)
        self.market_prices['land_rental'] = market_config.get('base_land_rent', 280.0)
        
        # @improvement: 价格内生机制开关
        self.use_endogenous_pricing = market_config.get('use_endogenous_pricing', True)
        
        # @new 2025-12-28: 刘易斯拐点机制 - 非农工资内生化
        # @theory: Lewis (1954) "Economic Development with Unlimited Supplies of Labour"
        # 当农村剩余劳动力大量转移时，工资受供给冲击影响
        # 过了拐点后，工资开始快速上涨
        lewis_config = market_config.get('lewis_turning_point', {})
        self.lewis_threshold = lewis_config.get('threshold', 0.35)  # 蔡昉(2013): 35%
        self.lewis_wage_elasticity = lewis_config.get('wage_elasticity', 0.5)  # 工资对供给的弹性
        self.initial_agri_labor_ratio = 0.50  # 初始农业劳动力占比
        self.current_agri_labor_ratio = self.initial_agri_labor_ratio
        self.base_wage = resolve_base_non_farm_wage(
            self.config, default=DEFAULT_BASE_NON_FARM_WAGE
        )
        self.base_wage = self._sanitize_non_farm_daily_wage(self.base_wage)
        self.market_prices['wage_rate'] = self._sanitize_non_farm_daily_wage(
            self.market_prices['wage_rate']
        )
        
        # 市场历史记录（用于形成预期）
        self.market_history = []
        self.labor_market_history = []
        self.grain_market_history = []
        self.cash_crop_market_history = []
        
        # 政府干预记录
        self.policy_intervention_history = []
        self.government_grain_storage = 0

        # Formal bookkeeping for grain-subsidy capitalization diagnostics.
        self.annual_capitalization_loss = 0.0
        self.cumulative_capitalization_loss = 0.0
        self.annual_raw_basic_subsidy = 0.0
        self.cumulative_raw_basic_subsidy = 0.0
        
        # @fix 2025-12-18: 添加土地租金市场价格
        # 用于农户老年退出时参考市场租金
        self.land_rental_price = market_config.get('base_land_rent', 280.0)  # 元/亩，基于2022年全国平均
        
        # @new 2025-12-28: 环境/气候模块（Nature可持续性分析必需）
        # @theory: 气候变化对农业产量的影响、土壤退化与可持续性
        # 文献：IPCC AR6 Chapter 5, Lobell et al. (2011)
        self.market_prices['land_rent'] = self.land_rental_price
        self.market_prices['land_rental'] = self.land_rental_price
        env_config = self.config.get('environment', {}).get('climate', {})
        
        # 土壤质量 (0-1, 1为最佳)
        self.initial_soil_quality = float(env_config.get('initial_soil_quality', 0.85) or 0.85)
        self.soil_quality = self.initial_soil_quality
        self.soil_degradation_rate = float(env_config.get('soil_degradation_rate', 0.005) or 0.005)
        self.soil_recovery_rate_base = float(env_config.get('soil_recovery_rate_base', 0.0010) or 0.0010)
        self.soil_recovery_gap_bonus = float(env_config.get('soil_recovery_gap_bonus', 0.0010) or 0.0010)
        self.soil_quality_floor = float(env_config.get('soil_quality_floor', 0.30) or 0.30)
        self.soil_yield_sensitivity = float(env_config.get('soil_yield_sensitivity', 0.55) or 0.55)
        ceiling_bonus = float(env_config.get('soil_quality_ceiling_bonus', 0.05) or 0.05)
        self.soil_quality_ceiling = max(
            self.initial_soil_quality,
            min(1.0, self.initial_soil_quality + ceiling_bonus),
        )
        self.extreme_weather_probability = float(env_config.get('extreme_weather_prob', 0.10) or 0.10)
        self.extreme_weather_yield_loss = float(env_config.get('extreme_weather_loss', 0.20) or 0.20)
        self.favorable_weather_probability = float(env_config.get('favorable_weather_prob', 0.0) or 0.0)
        self.favorable_weather_yield_gain = float(env_config.get('favorable_weather_gain', 0.0) or 0.0)
        self.minimum_yield_modifier = float(env_config.get('minimum_yield_modifier', 0.30) or 0.30)
        self.maximum_yield_modifier = float(env_config.get('maximum_yield_modifier', 1.10) or 1.10)
        self.current_year_favorable_weather = False
        self.favorable_weather_gain = 0.0
        self.soil_degradation_rate = env_config.get('soil_degradation_rate', 0.005)  # 年退化率0.5%
        
        # 气候冲击参数
        self.extreme_weather_probability = env_config.get('extreme_weather_prob', 0.10)  # 10%概率
        self.extreme_weather_yield_loss = env_config.get('extreme_weather_loss', 0.20)  # 产量损失20%
        
        # 年度气候状态
        self.current_year_climate_shock = False
        self.climate_shock_severity = 0.0  # 当年冲击强度
        
        # @new 2026-01-04: 世界线支持（配对实验必需）
        self._worldline = None  # 预生成的外生冲击序列
        self._worldline_id = None

    def set_worldline(self, worldline, worldline_id=None):
        """
        @new 2026-01-04: 设置世界线（配对实验模式）
        
        配对实验设计：同一世界线在不同政策下运行，消除随机性干扰
        
        Args:
            worldline: WorldLine对象或字典，包含：
                - climate_shocks: List[bool]  # 每年是否有气候冲击
                - climate_severities: List[float]  # 每年冲击强度
                - price_shocks: List[float]  # 每年价格冲击（可选）
            worldline_id: 世界线标识符
        """
        self._worldline = worldline
        self._worldline_id = worldline_id
        
    def get_worldline_id(self):
        """获取当前世界线ID"""
        return self._worldline_id
    
    def clear_worldline(self):
        """清除世界线，恢复随机模式"""
        self._worldline = None
        self._worldline_id = None

    def _resolve_technology_config(self):
        production_config = self.config.get('production', {}) or {}
        legacy_config = production_config.get('technology', {}) or {}
        progress_config = production_config.get('technology_progress', {}) or {}

        if not isinstance(legacy_config, dict):
            legacy_config = {}
        if not isinstance(progress_config, dict):
            progress_config = {}

        resolved_config = dict(progress_config)
        resolved_config.update(legacy_config)
        return resolved_config

    @staticmethod
    def _clamp_rate(value, default):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float(default)
        return max(0.0, min(1.0, numeric_value))

    def _get_patch_operator(self, patch):
        if (getattr(patch, 'operational_transfer_channel', '') == 'author_paid_initial_lease'
                and patch.lease_contract_years > 0 and patch.current_operator is None):
            return None
        leased_to = getattr(patch, 'leased_to', None)
        if leased_to is not None:
            return leased_to

        current_operator = getattr(patch, 'current_operator', None)
        if current_operator is not None:
            return current_operator

        return getattr(patch, 'owner', None)

    def _get_operator_technology_level(self, operator):
        if operator is None:
            return 1.0

        raw_value = getattr(operator, 'technology_level', None)
        if raw_value is None and hasattr(operator, '_get_technology_level'):
            try:
                raw_value = operator._get_technology_level()
            except Exception:
                raw_value = 1.0

        try:
            return max(1.0, min(2.0, float(raw_value if raw_value is not None else 1.0)))
        except (TypeError, ValueError):
            return 1.0

    def _sync_patch_technology_levels(self, model, tech_config):
        if model is None or not hasattr(model, 'land_patches'):
            return

        diffusion_rate = self._clamp_rate(
            tech_config.get(
                'patch_technology_diffusion_rate',
                tech_config.get('diffusion_rate', 1.0),
            ),
            default=1.0,
        )

        tenure_changed = False
        for patch in getattr(model, 'land_patches', []) or []:
            if patch is None:
                continue

            operator = self._get_patch_operator(patch)
            if operator is not None and getattr(patch, 'current_operator', None) is not operator:
                patch.current_operator = operator
                tenure_changed = True

            target_level = self._get_operator_technology_level(operator)
            try:
                current_level = float(getattr(patch, 'technology_level', 1.0) or 1.0)
            except (TypeError, ValueError):
                current_level = 1.0

            updated_level = current_level + (target_level - current_level) * diffusion_rate
            if hasattr(patch, 'update_technology_level'):
                patch.update_technology_level(updated_level)
            else:
                patch.technology_level = max(1.0, min(2.0, updated_level))
        if tenure_changed and hasattr(model, 'invalidate_land_tenure_caches'):
            model.invalidate_land_tenure_caches()

    def _get_non_farm_wage_bounds(self):
        labor_cfg = self.config.get('labor_market', {}) or {}
        floor = float(labor_cfg.get('wage_floor', 80.0) or 80.0)
        ceiling = float(labor_cfg.get('wage_ceiling', max(300.0, floor)) or max(300.0, floor))
        if ceiling < floor:
            ceiling = floor
        return floor, ceiling

    def _sanitize_non_farm_daily_wage(self, wage_value):
        fallback = resolve_base_non_farm_wage(
            self.config,
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        try:
            candidate = float(wage_value)
        except (TypeError, ValueError):
            candidate = float(fallback)

        if candidate <= 0:
            candidate = float(fallback)

        # Guard against annual/monthly wage figures leaking into a daily-wage slot.
        if candidate > 1000:
            candidate = candidate / 260.0

        floor, ceiling = self._get_non_farm_wage_bounds()
        return float(min(ceiling, max(floor, candidate)))

    def update_policy(self, government_agent):
        """由政府智能体在每一步开始时更新政策"""
        # 这个方法现在由 government_agent.announce_policy() 直接调用 set_active_policy
        # 保留此方法以保持接口兼容性
        pass

    def set_active_policy(self, policy_state, policy_id=None):
        """更新当前政策状态"""
        self.active_policy_id = policy_id or self.active_policy_id
        normalized_policy = normalize_grain_subsidy_aliases(
            policy_state or {},
            context=f"environment active policy '{policy_id or self.active_policy_id or 'unknown'}'",
        )
        disable_grain_intervention = self._policy_disables_grain_intervention(
            normalized_policy,
            policy_id=self.active_policy_id,
        )
        normalized_policy["disable_grain_intervention"] = disable_grain_intervention
        if disable_grain_intervention:
            normalized_policy["grain_price_floor"] = 0.0
            normalized_policy["grain_subsidies_rate"] = 0.0
            normalized_policy["subsidy_rate"] = 0.0
            normalized_policy["base_subsidy_per_mu"] = 0.0
        self.active_policy = normalized_policy
        self.policy = self.active_policy  # 兼容旧接口
        self.payment_lag = int(self.active_policy.get("payment_lag", 0))

    def _policy_disables_grain_intervention(self, policy_state=None, policy_id=None):
        scenario_label = str(policy_id or self.active_policy_id or "").strip().lower()
        if scenario_label in {"no_policy", "none", "market_only"}:
            return True

        policy = policy_state or {}
        if not isinstance(policy, dict):
            return False

        explicit_disable = policy.get("disable_grain_intervention")
        if explicit_disable is not None:
            return bool(explicit_disable)

        try:
            floor_value = float(policy.get("grain_price_floor", 0.0) or 0.0)
        except (TypeError, ValueError):
            floor_value = 0.0
        try:
            subsidy_rate = float(
                policy.get(
                    "grain_subsidies_rate",
                    policy.get("subsidy_rate", 0.0),
                ) or 0.0
            )
        except (TypeError, ValueError):
            subsidy_rate = 0.0

        return floor_value <= 0.0 and subsidy_rate <= 0.0

    def update_market_prices(self, model):
        """
        根据供需更新内生价格
        
        @theory:
        实现真正的市场出清机制
        1. 经济作物市场：总供给 = 所有智能体经济作物的总产量，总需求 = 价格弹性函数
        2. 劳动力市场：总供给 = 所有农户的非农劳动力，总需求 = 企业需求 + 非农部门需求
        3. 粮食市场：考虑政府价格干预
        """
        # 市场步骤计数
        self.current_step += 1
        
        # @new 2025-12-28: 年度环境更新（气候冲击 + 土壤退化）
        self._update_environmental_conditions()
        
        # 1. 更新农产品市场
        if model.config.get("runtime_fixed_rule", {}).get("national_v1"):
            # This cohort takes national crop prices as given; rents still clear locally.
            prices = self.config["environment"]["market"]["base_prices"]
            self.market_prices.update(crop_price=prices["grain"], cash_crop=prices["cash_crop"])
        else:
            self.update_agricultural_markets(model)
        
        # 2. 更新劳动力市场
        self.update_labor_market(model)
        
        # 记录当前市场状态
        self.market_history.append({
            'step': self.current_step,
            'crop_price': self.market_prices['crop_price'],
            'cash_crop_price': self.market_prices['cash_crop'],
            'wage_rate': self.market_prices['wage_rate'],
            'soil_quality': self.soil_quality,  # 新增环境指标
            'climate_shock': self.current_year_climate_shock
        })

    def _update_environmental_conditions(self):
        self.current_year_climate_shock = False
        self.climate_shock_severity = 0.0
        self.current_year_favorable_weather = False
        self.favorable_weather_gain = 0.0

        if hasattr(self, '_worldline') and self._worldline is not None:
            try:
                year = self.current_step - 1
                if isinstance(self._worldline, dict):
                    if year < len(self._worldline.get('climate_shocks', [])):
                        self.current_year_climate_shock = bool(self._worldline['climate_shocks'][year])
                        self.climate_shock_severity = float(self._worldline['climate_severities'][year] or 0.0)
                        favorable_flags = (
                            self._worldline.get('favorable_weather')
                            or self._worldline.get('favorable_shocks')
                            or []
                        )
                        favorable_severities = (
                            self._worldline.get('favorable_severities')
                            or self._worldline.get('favorable_gains')
                            or []
                        )
                        if year < len(favorable_flags):
                            self.current_year_favorable_weather = bool(favorable_flags[year])
                        if year < len(favorable_severities):
                            self.favorable_weather_gain = float(favorable_severities[year] or 0.0)
                    else:
                        self._generate_random_climate()
                else:
                    year_data = self._worldline.get_year(year)
                    self.current_year_climate_shock = bool(year_data.get('climate_shock', False))
                    self.climate_shock_severity = float(year_data.get('climate_severity', 0.0) or 0.0)
                    self.current_year_favorable_weather = bool(
                        year_data.get('favorable_weather', year_data.get('favorable_shock', False))
                    )
                    self.favorable_weather_gain = float(
                        year_data.get('favorable_severity', year_data.get('favorable_gain', 0.0)) or 0.0
                    )
            except (IndexError, KeyError, AttributeError, TypeError, ValueError):
                self._generate_random_climate()
        else:
            self._generate_random_climate()

        if self.current_year_climate_shock and self.current_year_favorable_weather:
            if float(self.climate_shock_severity or 0.0) >= float(self.favorable_weather_gain or 0.0):
                self.current_year_favorable_weather = False
                self.favorable_weather_gain = 0.0
            else:
                self.current_year_climate_shock = False
                self.climate_shock_severity = 0.0

        self._update_soil_degradation()
        return
        """
        @new 2025-12-28: 年度环境条件更新
        @fix 2026-01-04: 支持世界线模式，确保配对实验可复现
        
        @theory: 气候变化与农业可持续性
        - 气候冲击：随机极端天气事件（干旱、洪涝）影响产量
        - 土壤退化：连续耕作导致土壤质量下降
        
        文献：IPCC AR6, Lobell et al. (2011) Nature Climate Change
        """
        # @fix 2026-01-04: 优先使用世界线中的气候序列（配对实验模式）
        if hasattr(self, '_worldline') and self._worldline is not None:
            # 从世界线获取当年气候条件
            try:
                year = self.current_step - 1  # step从1开始，年份从0开始
                if isinstance(self._worldline, dict):
                    # 从字典加载
                    if year < len(self._worldline.get('climate_shocks', [])):
                        self.current_year_climate_shock = self._worldline['climate_shocks'][year]
                        self.climate_shock_severity = self._worldline['climate_severities'][year]
                    else:
                        # 超出世界线范围，使用随机
                        self._generate_random_climate()
                else:
                    # 从WorldLine对象获取
                    year_data = self._worldline.get_year(year)
                    self.current_year_climate_shock = year_data['climate_shock']
                    self.climate_shock_severity = year_data['climate_severity']
            except (IndexError, KeyError, AttributeError):
                # 回退到随机生成
                self._generate_random_climate()
        else:
            # 独立运行模式：使用随机生成
            self._generate_random_climate()
        
        # 2. 土壤退化 - 带阈值效应和迟滞效应
        self._update_soil_degradation()
    
    def _generate_random_climate(self):
        climate_draw = np_random_value(self)
        adverse_prob = max(0.0, min(1.0, float(self.extreme_weather_probability or 0.0)))
        favorable_prob = max(0.0, min(1.0 - adverse_prob, float(self.favorable_weather_probability or 0.0)))
        self.current_year_climate_shock = False
        self.climate_shock_severity = 0.0
        self.current_year_favorable_weather = False
        self.favorable_weather_gain = 0.0
        if climate_draw < adverse_prob:
            self.current_year_climate_shock = True
            self.climate_shock_severity = float(self.extreme_weather_yield_loss) * (0.5 + np_random_value(self) * 0.5)
            return
        if climate_draw < (adverse_prob + favorable_prob):
            self.current_year_favorable_weather = True
            self.favorable_weather_gain = float(self.favorable_weather_yield_gain) * (0.5 + np_random_value(self) * 0.5)
            return
        return
    def _update_soil_degradation(self):
        floor = max(0.10, min(float(self.soil_quality_floor or 0.30), float(self.initial_soil_quality or 0.85)))
        ceiling = max(float(self.initial_soil_quality or 0.85), min(1.0, float(self.soil_quality_ceiling or 1.0)))
        current_quality = float(self.soil_quality or 0.0)
        quality_gap = max(0.0, float(self.initial_soil_quality or 0.0) - current_quality)
        excess_quality = max(0.0, current_quality - float(self.initial_soil_quality or 0.0))
        range_above_baseline = max(1e-6, ceiling - float(self.initial_soil_quality or 0.0))
        effective_degradation = float(self.soil_degradation_rate or 0.0) * (
            1.0 + 0.4 * (excess_quality / range_above_baseline)
        )
        effective_recovery = float(self.soil_recovery_rate_base or 0.0) + float(self.soil_recovery_gap_bonus or 0.0) * quality_gap
        if self.current_year_climate_shock:
            effective_recovery *= 0.85
        elif self.current_year_favorable_weather:
            effective_recovery *= 1.10
        self.soil_quality = max(
            floor,
            min(ceiling, current_quality + effective_recovery - effective_degradation),
        )
        return
    
    def get_yield_modifier(self):
        base_modifier = 1.0 + (
            (float(self.soil_quality or 0.0) - float(self.initial_soil_quality or 0.0))
            * float(self.soil_yield_sensitivity or 0.0)
        )
        if self.current_year_climate_shock:
            base_modifier *= (1.0 - float(self.climate_shock_severity or 0.0))
        elif self.current_year_favorable_weather:
            base_modifier *= (1.0 + float(self.favorable_weather_gain or 0.0))
        return max(float(self.minimum_yield_modifier or 0.30), min(float(self.maximum_yield_modifier or 1.10), base_modifier))

    def get_climate_shock(self):
        if self.current_year_climate_shock:
            return -max(0.0, float(self.climate_shock_severity or 0.0))
        if self.current_year_favorable_weather:
            return max(0.0, float(self.favorable_weather_gain or 0.0))
        return 0.0
    def update_technology_progress(self, model):
        """
        @new 2025-01-02: 技术进步机制
        
        @theory 内生增长理论与农业技术进步
        - Romer (1990): 技术进步是经济增长的内生动力
        - Hayami & Ruttan (1985): 诱致性技术变迁
        - 中国农业TFP增长率约2.5-3.5%/年（World Bank, 2020）
        
        机制设计:
        1. 全要素生产率(TFP)年度增长
        2. 技术扩散受规模、年龄、教育等因素影响
        3. 技术前沿与采纳滞后
        
        Args:
            model: 主模型对象
        """
        # 获取技术进步配置
        tech_config = self._resolve_technology_config()
        if not tech_config.get('enabled', True):
            return
        
        # 基础TFP增长率 (默认2.5%/年)
        base_tfp_growth = tech_config.get(
            'annual_tfp_growth',
            tech_config.get('annual_yield_growth_rate', 0.025),
        )
        
        # 技术扩散参数
        diffusion_rate = tech_config.get('diffusion_rate', 0.1)  # 扩散速度
        education_effect = tech_config.get('education_effect', 0.02)  # 教育对采纳的影响
        age_barrier = tech_config.get('age_barrier', 0.01)  # 年龄障碍
        
        # 更新全局技术前沿
        if not hasattr(self, 'technology_frontier'):
            self.technology_frontier = 1.0
        self.technology_frontier *= (1 + base_tfp_growth)
        
        # 遍历农户，更新个体技术水平
        if hasattr(model, 'farmers'):
            from agents.farmer_agent import LivelihoodStrategy
            
            for farmer in model.farmers:
                if not hasattr(farmer, 'technology_level'):
                    farmer.technology_level = 1.0
                
                # 技术差距
                tech_gap = self.technology_frontier - farmer.technology_level
                
                if tech_gap > 0:
                    # 基础采纳率
                    adoption_rate = diffusion_rate
                    
                    # 规模效应：大户更容易采纳新技术
                    total_land = sum(p.area for p in farmer.land_patches) if farmer.land_patches else 0
                    if total_land > 30:  # 大户（30亩以上）
                        adoption_rate *= 1.5
                    elif total_land > 10:  # 中户
                        adoption_rate *= 1.2
                    
                    # 年龄效应：年轻农户更容易采纳
                    if hasattr(farmer, 'age'):
                        if farmer.age < 40:
                            adoption_rate *= 1.3
                        elif farmer.age > 60:
                            adoption_rate *= 0.7
                    
                    # 生计策略效应：纯农户更有激励投资技术
                    if hasattr(farmer, 'livelihood_strategy'):
                        if farmer.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                            adoption_rate *= 1.2
                    
                    # 技术追赶
                    farmer.technology_level += tech_gap * min(1.0, adoption_rate)
        
        # 更新企业技术水平（企业技术采纳更快）
        if hasattr(model, 'agricultural_enterprises'):
            enterprise_adoption = tech_config.get('enterprise_adoption_rate', 0.3)
            
            for enterprise in model.agricultural_enterprises:
                if not hasattr(enterprise, 'technology_level'):
                    enterprise.technology_level = 1.0
                
                tech_gap = self.technology_frontier - enterprise.technology_level
                if tech_gap > 0:
                    enterprise.technology_level += tech_gap * enterprise_adoption

        self._sync_patch_technology_levels(model, tech_config)

    def update_wage_with_lewis_mechanism(self, model):
        """
        @new 2025-12-28: 刘易斯拐点机制 - 非农工资内生化
        
        @theory: Lewis (1954) "Economic Development with Unlimited Supplies of Labour"
        
        @literature:
        - 蔡昉 (2007, 2010) 中国社科院: 中国刘易斯拐点在2004-2010年区间
        - 蔡昉 (2013): 农村劳动力比重约35%时达拐点(类似日本1960/韩国1972)
        - IMF (2012): 中国劳动年龄人口2010年达峰值后开始下降
        
        机制:
        - 拐点前: 农村剩余劳动力充足，工资增长受抑制
        - 拐点后: 剩余劳动力耗尽，工资快速上涨（"民工荒"现象）
        
        阈值校准: 农业劳动力占比35%（基于蔡昉实证研究）

        Args:
            model: 主模型对象，用于获取农户生计状态
        """
        # 1. 计算当前农业劳动力占比
        if hasattr(model, 'farmers') and len(model.farmers) > 0:
            from agents.farmer_agent import LivelihoodStrategy
            
            agri_labor = sum(1 for f in model.farmers 
                           if hasattr(f, 'livelihood_strategy') and 
                           f.livelihood_strategy == LivelihoodStrategy.PURE_FARMER)
            total = len(model.farmers)
            self.current_agri_labor_ratio = agri_labor / total if total > 0 else 0.5
        
        # 2. 刘易斯拐点逻辑
        gdp_growth_rate = self.config.get('simulation', {}).get('gdp_growth_rate', 0.05)
        
        if self.current_agri_labor_ratio > self.lewis_threshold:
            # 拐点前：工资增长受抑制（劳动力过剩）
            # 工资增长 = GDP增长 × 弹性系数（<1）
            wage_growth = gdp_growth_rate * self.lewis_wage_elasticity
            wage_pressure = "拐点前:劳动力过剩"
        else:
            # 拐点后：工资加速增长
            # 劳动力越稀缺，工资涨幅越大
            scarcity_factor = (self.lewis_threshold - self.current_agri_labor_ratio) / self.lewis_threshold
            wage_growth = gdp_growth_rate * (1 + scarcity_factor)
            wage_pressure = "拐点后:劳动力稀缺"
        
        # 3. 劳动力转移的供给冲击
        # 如果农民大量退出农业→压低低端非农工资
        labor_transfer_rate = self.initial_agri_labor_ratio - self.current_agri_labor_ratio
        if labor_transfer_rate > 0.1:  # 转移超过10%
            supply_shock = -0.02 * labor_transfer_rate  # 供给冲击压低工资
            wage_growth += supply_shock
        
        # 4. 更新工资
        self.market_prices['wage_rate'] *= (1 + max(-0.05, min(0.10, wage_growth)))
        self.market_prices['wage_rate'] = self._sanitize_non_farm_daily_wage(
            self.market_prices['wage_rate']
        )
        
        # 记录历史
        self.labor_market_history.append({
            'step': self.current_step,
            'agri_labor_ratio': self.current_agri_labor_ratio,
            'wage_growth': wage_growth,
            'wage_rate': self.market_prices['wage_rate'],
            'pressure': wage_pressure
        })

    def update_agricultural_markets(self, model):
        """更新农产品市场价格"""
        # 经济作物市场
        self._update_cash_crop_market(model)
        
        # @critical 2025-12: 统一市场出清算法
        # 专家意见："储备局不应是一个事后反应的Agent，而应作为需求函数的一部分参与到价格形成的联立方程中"
        # 粮食市场（政府干预内生于价格形成）
        self._update_grain_market_with_intervention(model)

    def _update_cash_crop_market(self, model):
        """
        经济作物市场出清
        @improvement: P0优先级 - 实现内生价格形成机制
        基于供需关系动态调整价格
        """
        # 检查是否启用内生价格机制
        market_config = self.config.get('environment', {}).get('market', {})
        production_config = self.config.get('production', {})
        use_endogenous = market_config.get('use_endogenous_pricing', True)
        
        # 获取单产参数用于初始估计
        base_yield_cash_crop = production_config.get('base_yield', {}).get('cash_crop', 500)  # kg/亩
        
        # @fix 2025-12: 使用上一期产出或期望产量作为供给
        total_supply = 0
        
        # 农户供给
        for farmer in model.farmers:
            if hasattr(farmer, "household_actual_grain_share"):
                total_supply += farmer.cash_crop_production
            elif hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Cash_Crop":
                if hasattr(farmer, 'last_yield') and farmer.last_yield > 0:
                    total_supply += farmer.last_yield
                elif hasattr(farmer, 'expected_yield') and farmer.expected_yield > 0:
                    total_supply += farmer.expected_yield
            elif self.current_step <= 1:
                # 第一步：使用土地面积×单产估计（假设50%种经作）
                operated_area = getattr(farmer, 'operated_land_area', 0)
                if operated_area <= 0 and hasattr(farmer, 'land_patches'):
                    operated_area = sum(p.area for p in farmer.land_patches if p.owner == farmer)
                if operated_area > 0:
                    total_supply += operated_area * base_yield_cash_crop * 0.5

        # 企业供给
        for enterprise in model.agricultural_enterprises:
            if hasattr(enterprise, 'last_yields') and isinstance(enterprise.last_yields, dict):
                cash_crop_yield = enterprise.last_yields.get('cash_crop', 0)
                total_supply += cash_crop_yield
            elif self.current_step <= 1:
                # 第一步使用土地面积×单产×经作比例估计
                land_area = getattr(enterprise, 'land_area', 0)
                grain_weight = getattr(enterprise, 'grain_weight', 0.5)
                if land_area > 0:
                    total_supply += land_area * base_yield_cash_crop * (1 - grain_weight)
        
        # @fix 2024-11-20: 移除向后兼容代码，简化逻辑
        # 专家审阅意见：既然追求"顶级"，就应该删除所有"兼容旧版本"的死代码
        # 配置中use_endogenous_pricing默认为true，固定价格逻辑不再需要
        if use_endogenous:
            # @improvement: 内生价格形成机制
            # 1. 获取基础需求量和价格弹性
            # @fix 2024-11-18: 降低基础需求（5,000,000→200,000）
            # 经济作物需求通常低于粮食，设置为粮食需求的50%
            base_demand = market_config.get('base_cash_crop_demand', 200000)  # 从配置读取
            # @fix 2024-11-26: 修正默认值为-0.4，与master_config.yaml一致
            price_elasticity = market_config.get('cash_crop_price_elasticity', -0.4)  # 从配置读取
            
            # 2. 计算供需比
            if base_demand > 0:
                supply_demand_ratio = total_supply / base_demand
            else:
                supply_demand_ratio = 1.0
            
            # 3. 计算市场出清价格（基于价格弹性）
            # 价格 = 基础价格 × (供需比 ^ (1/弹性))
            # @fix 2024-11-21: 修复经作价格低于粮食价格的问题
            # 问题1：经作价格1.20元/斤低于粮食价格1.25元/斤，不合理
            # 问题2：价格被锁定在1.20元，不再动态变化
            # 原因：供需比极低（企业100%种粮），价格下限1.2元过低且低于粮食价格
            # 修复：提高价格下限，确保经作价格始终高于粮食价格，增加供给不足时的价格上涨机制
            base_price = market_config.get('base_prices', {}).get('cash_crop', 1.8)
            grain_price = self.market_prices.get('crop_price', 1.25)  # 获取当前粮食价格
            
            if supply_demand_ratio > 0 and price_elasticity != 0:
                # @fix 2024-11-26: 从配置读取供需比限制范围，替代硬编码值
                ratio_limits = market_config.get('supply_demand_ratio_limits', {})
                # @fix 2024-11-29: 修正ratio_limits的读取方式（嵌套字典）
                cash_crop_ratio_limits = ratio_limits.get('cash_crop', {})
                min_ratio = cash_crop_ratio_limits.get('min', 0.3)
                max_ratio = cash_crop_ratio_limits.get('max', 3.0)
                clamped_ratio = max(min_ratio, min(max_ratio, supply_demand_ratio))
                # 调整价格弹性：如果绝对值>0.7，限制为-0.5，使价格变化更平滑
                effective_elasticity = price_elasticity
                if abs(effective_elasticity) > 0.7:
                    effective_elasticity = -0.5 if effective_elasticity < 0 else 0.5
                elif abs(effective_elasticity) < 0.3:
                    effective_elasticity = -0.3 if effective_elasticity < 0 else 0.3
                market_clearing_price = base_price * (clamped_ratio ** (1.0 / effective_elasticity))
                
                # @fix 2024-11-21: 增加供给不足时的价格上涨机制
                # 如果供需比<0.5（供给不足），价格应该上涨，不应低于基准价格的80%（1.44元/斤）
                if supply_demand_ratio < 0.5:
                    # 供给不足时，价格应该上涨：基准价格 × (1 + 短缺惩罚)
                    shortage_penalty = (0.5 - supply_demand_ratio) * 0.5  # 短缺越大，价格上涨越多
                    min_price_floor = base_price * (0.8 + shortage_penalty)  # 至少基准价格的80%，最高130%
                    market_clearing_price = max(min_price_floor, market_clearing_price)
                
                # @fix 2024-11-26: 从配置读取价格限制，替代硬编码值
                price_limits = market_config.get('price_limits', {})
                # @fix 2024-11-29: 修正price_limits的读取方式（嵌套字典）
                cash_crop_limits = price_limits.get('cash_crop', {})
                min_cash_crop_price = max(cash_crop_limits.get('min', 1.3), grain_price + 0.05)
                market_clearing_price = max(min_cash_crop_price, market_clearing_price)
                
                # 确保价格在合理范围内（上限从配置读取）
                max_cash_crop_price = cash_crop_limits.get('max', 5.0)
                market_clearing_price = min(max_cash_crop_price, market_clearing_price)
            else:
                market_clearing_price = base_price
            
            # 4. 应用价格粘性（避免过度波动）
            # @fix 2024-11-20: 降低价格粘性（0.85→0.7），让价格更快响应供需变化
            # 原值0.85导致价格变化极慢，价格被锁定
            price_stickiness = market_config.get('price_stickiness', {}).get('cash_crop', 0.7)
            current_price = self.market_prices['cash_crop']
            new_price = price_stickiness * current_price + (1 - price_stickiness) * market_clearing_price
            
            # @debug: 打印价格计算（关键步骤）
            if self.current_step in [0, 1, 5, 10, 20, 50, 99]:
                print(f"[Step {self.current_step}] 经作市场统一出清: 供给={total_supply:.0f}kg, "
                      f"均衡价={new_price:.2f}元/斤 (理论价={market_clearing_price:.2f})")
            
            # 5. 价格上限保护（基于真实市场范围）
            # @fix 2024-11-21: 确保经作价格不低于粮食价格
            # 经济作物价格应始终高于粮食价格（劳动密集型，附加值高）
            # 价格下限动态调整：max(1.3, 粮食价格 + 0.05)
            max_price = 5.5  # 最高价格保护（基础价1.8的3倍，给予更大调节空间）
            min_price = max(1.3, grain_price + 0.05)  # 最低价格保护（至少高于粮食价格0.05元）
            self.market_prices['cash_crop'] = max(min_price, min(max_price, new_price))
        
        # 记录市场数据
        self.cash_crop_market_history.append({
            'time_step': self.current_step,
            'cash_crop_supply': total_supply,
            'cash_crop_price': self.market_prices['cash_crop']
        })

    def _update_grain_market_with_intervention(self, model):
        """
        @critical 2025-12: 统一市场出清算法 - 专家建议实现
        
        核心改进：
        储备局作为需求函数的一部分参与价格形成的联立方程，而非事后干预
        
        理论框架：
        市场均衡：Q_s = Q_d + Q_govt
        价格形成：P* = f(Q_s, Q_d, Q_govt, P_floor)
        
        其中：
        - Q_s: 总供给（农户 + 企业）
        - Q_d: 市场需求（消费者）
        - Q_govt: 政府收购需求（托市机制）
        - P_floor: 最低收购价（政策参数）
        
        均衡条件：
        当 P_market < P_floor 时，Q_govt = max(0, Q_s - Q_d_at_floor)
        政府以P_floor无限量收购，将价格托至P_floor
        
        专家原话："确保价格形成是联立方程解，而不是两步走的补丁"
        """
        # ========== 1. 计算供给 ==========
        market_config = self.config.get('environment', {}).get('market', {})
        production_config = self.config.get('production', {})
        
        total_supply = 0
        farmer_supply = 0
        enterprise_supply = 0
        
        # 获取单产参数用于初始估计
        base_yield_grain = production_config.get('base_yield', {}).get('grain', 450)  # kg/亩
        
        # 农户供给
        for farmer in model.farmers:
            # @fix 2025-12: 使用上一期产量或期望产量（解决Step 1供给为0的问题）
            if hasattr(farmer, "household_actual_grain_share"):
                farmer_supply += farmer.grain_production
            elif hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Grain":
                if hasattr(farmer, 'last_yield') and farmer.last_yield > 0:
                    farmer_supply += farmer.last_yield
                elif hasattr(farmer, 'expected_yield') and farmer.expected_yield > 0:
                    # 使用期望产量作为初始估计
                    farmer_supply += farmer.expected_yield
            elif self.current_step <= 1:
                # 第一步：使用土地面积×单产估计（假设50%种粮）
                operated_area = getattr(farmer, 'operated_land_area', 0)
                if operated_area <= 0 and hasattr(farmer, 'land_patches'):
                    operated_area = sum(p.area for p in farmer.land_patches if p.owner == farmer)
                if operated_area > 0:
                    farmer_supply += operated_area * base_yield_grain * 0.5  # 假设50%种粮
        
        # 企业供给
        for enterprise in model.agricultural_enterprises:
            if hasattr(enterprise, 'last_yields') and isinstance(enterprise.last_yields, dict):
                grain_yield = enterprise.last_yields.get('grain', 0)
                enterprise_supply += grain_yield
            elif self.current_step <= 1:
                # 第一步使用土地面积×单产×粮食比例估计
                land_area = getattr(enterprise, 'land_area', 0)
                grain_weight = getattr(enterprise, 'grain_weight', 0.5)
                if land_area > 0:
                    enterprise_supply += land_area * base_yield_grain * grain_weight
        
        total_supply = farmer_supply + enterprise_supply
        
        # ========== 2. 计算需求函数 ==========
        base_demand = market_config.get('base_grain_demand', 400000)
        price_elasticity = market_config.get('grain_price_elasticity', -0.3)
        base_price = market_config.get('base_prices', {}).get('grain', 1.25)
        
        # ========== 3. 计算政府干预参数 ==========
        # 托市价格（最低收购价）
        active_policy = getattr(self, 'active_policy', {}) or {}
        disable_grain_intervention = self._policy_disables_grain_intervention(
            active_policy,
            policy_id=getattr(self, 'active_policy_id', None),
        )
        grain_price_floor = self.get_policy_param('grain_price_floor')
        if grain_price_floor is None:
            # 从配置获取默认托市价
            grain_price_floor = market_config.get('price_limits', {}).get('grain', {}).get('min', 1.0)
        price_limits = market_config.get('price_limits', {}).get('grain', {})
        min_price_limit = price_limits.get('min', 1.0)
        max_price_limit = price_limits.get('max', 5.0)
        market_only_numeric_floor = float(market_config.get('market_only_numeric_floor', 0.0) or 0.0)
        if disable_grain_intervention:
            grain_price_floor = 0.0
            min_price_limit = max(0.0, market_only_numeric_floor)
        else:
            grain_price_floor = max(min_price_limit, min(max_price_limit, grain_price_floor))
        
        # 抛储价格（市场价高于此价时政府抛储）
        ceiling_multiplier = market_config.get('grain_ceiling_multiplier', 1.3)
        grain_price_ceiling = float('inf') if disable_grain_intervention else grain_price_floor * ceiling_multiplier
        market_price_raw = base_price
        govt_purchase = 0.0
        govt_release = 0.0
        needed_purchase = 0.0
        support_coverage_ratio = 0.0
        intervention_shortfall = 0.0
        fiscal_warning = None
        
        # ========== 4. 联立求解均衡价格 ==========
        if total_supply > 0 and base_demand > 0:
            # 计算无干预时的市场出清价格
            supply_demand_ratio = total_supply / base_demand
            
            # 价格公式：P = P_base × (Q_s/Q_d)^(1/ε)
            if price_elasticity != 0:
                # 限制供需比在合理范围内
                ratio_limits = market_config.get('supply_demand_ratio_limits', {}).get('grain', {})
                min_ratio = ratio_limits.get('min', 0.3)
                max_ratio = ratio_limits.get('max', 3.0)
                safe_ratio = max(min_ratio, min(max_ratio, supply_demand_ratio))
                
                # 确保弹性系数有效
                effective_elasticity = price_elasticity
                if abs(effective_elasticity) < 0.2:
                    effective_elasticity = -0.3 if effective_elasticity < 0 else 0.3
                
                market_price_raw = base_price * (safe_ratio ** (1.0 / effective_elasticity))
            else:
                market_price_raw = base_price
            
            # ========== 5. 政府干预联立方程 ==========
            # 这是核心改进：政府干预内生于均衡
            # @enhanced 2025-12: 调用储备局新接口，集成财政约束和可持续性评估
            govt_purchase = 0
            govt_release = 0
            needed_purchase = 0.0
            support_coverage_ratio = 0.0
            intervention_shortfall = 0.0
            needed_purchase = 0.0
            support_coverage_ratio = 0.0
            intervention_shortfall = 0.0
            govt_storage = getattr(self, 'government_grain_storage', 0)
            if hasattr(model, 'grain_reserve_bureau') and model.grain_reserve_bureau is not None:
                govt_storage = model.grain_reserve_bureau.get_storage_kg()
                self.government_grain_storage = govt_storage
            fiscal_warning = None
            
            if (not disable_grain_intervention) and market_price_raw < grain_price_floor:
                # 托市收购启动
                # 政府需求 = 使市场价格达到托市价所需的收购量
                # Q_govt = Q_s - D(P_floor)
                # 其中 D(P_floor) 是托市价格下的市场需求
                demand_at_floor = base_demand * (grain_price_floor / base_price) ** price_elasticity
                needed_purchase = max(0, total_supply - demand_at_floor)
                
                # 调用储备局新接口获取干预能力
                if hasattr(model, 'grain_reserve_bureau') and model.grain_reserve_bureau is not None:
                    bureau = model.grain_reserve_bureau
                    intervention_info = bureau.calculate_intervention_demand(market_price_raw, total_supply)
                    
                    # 使用储备局计算的有效干预量（已考虑财政悬崖等约束）
                    govt_purchase = min(needed_purchase, intervention_info['quantity'])
                    
                    # 检查财政预警
                    if intervention_info['fiscal_stress_level'] >= 2:
                        fiscal_warning = f"财政紧张(L{intervention_info['fiscal_stress_level']}), 连续干预{intervention_info['consecutive_years']}年"
                    
                    # 执行实际收购
                    bureau.execute_intervention('purchase', govt_purchase, grain_price_floor)
                    self.government_grain_storage = bureau.get_storage_kg()
                else:
                    # 无储备局时的简化处理
                    annual_budget, max_capacity_kg, _ = self._get_fallback_reserve_limits()
                    floor_price_per_kg = max(grain_price_floor * 2.0, 1e-6)
                    budget_limit = annual_budget / floor_price_per_kg
                    capacity_limit = max(0.0, max_capacity_kg - govt_storage)
                    govt_purchase = min(needed_purchase, budget_limit, capacity_limit)
                    # 更新政府粮储
                    self.government_grain_storage = govt_storage + govt_purchase
                
                # 均衡价格 = 托市价（政府收购足够）或略低于托市价（预算约束）
                support_coverage_ratio = min(1.0, govt_purchase / max(needed_purchase, 1e-6)) if needed_purchase > 0 else 1.0
                intervention_shortfall = max(0.0, needed_purchase - govt_purchase)

                if govt_purchase >= needed_purchase:
                    equilibrium_price = grain_price_floor
                else:
                    # 预算/库容受限时，价格支撑只能覆盖真实吸收掉的那部分过剩供给。
                    # 不能在“没收进去多少粮”的情况下仍把价格强行钉在托市价附近。
                    residual_supply = total_supply - govt_purchase
                    if residual_supply > 0 and base_demand > 0:
                        residual_ratio = residual_supply / base_demand
                        residual_market_price = base_price * (residual_ratio ** (1.0 / effective_elasticity))
                    else:
                        residual_market_price = market_price_raw
                    equilibrium_price = residual_market_price + support_coverage_ratio * (
                        grain_price_floor - residual_market_price
                    )
                    
            elif (not disable_grain_intervention) and market_price_raw > grain_price_ceiling:
                # 抛储稳价
                # 政府释放储备，增加供给，压低价格
                target_price = grain_price_ceiling
                
                # 需要释放多少使价格降至目标
                demand_at_ceiling = base_demand * (target_price / base_price) ** price_elasticity
                needed_release = max(0, demand_at_ceiling - total_supply)
                
                # 调用储备局接口
                if hasattr(model, 'grain_reserve_bureau') and model.grain_reserve_bureau is not None:
                    bureau = model.grain_reserve_bureau
                    intervention_info = bureau.calculate_intervention_demand(market_price_raw, total_supply)
                    # @fix 2025-11-30: 确保不会抛售超过库存
                    available_release = bureau.get_available_release_kg()
                    govt_release = min(needed_release, intervention_info['quantity'], available_release)
                    bureau.execute_intervention('release', govt_release, market_price_raw)
                    # 同步粮储到 environment（从bureau读取实际值）
                    self.government_grain_storage = bureau.get_storage_kg()
                else:
                    # 库存约束
                    _, _, safety_stock = self._get_fallback_reserve_limits()
                    available_release = max(0, govt_storage - safety_stock)
                    govt_release = min(needed_release, available_release)
                    # 更新政府粮储（确保不低于0）
                    self.government_grain_storage = max(0, govt_storage - govt_release)
                
                # 计算抛储后均衡价格
                total_supply_with_release = total_supply + govt_release
                if total_supply_with_release > 0 and base_demand > 0:
                    new_ratio = total_supply_with_release / base_demand
                    equilibrium_price = base_price * (new_ratio ** (1.0 / effective_elasticity))
                else:
                    equilibrium_price = market_price_raw
                
                equilibrium_price = max(equilibrium_price, grain_price_floor)  # 不低于托市价
            else:
                # 市场价在[托市价, 抛储价]区间，无需干预
                equilibrium_price = market_price_raw
            
            # ========== 6. 价格粘性与边界约束 ==========
            price_stickiness = market_config.get('price_stickiness', {}).get('grain', 0.8)
            current_price = self.market_prices['crop_price']
            new_price = price_stickiness * current_price + (1 - price_stickiness) * equilibrium_price
            
            # 价格边界
            price_limits = market_config.get('price_limits', {}).get('grain', {})
            min_price = max(0.0, market_only_numeric_floor) if disable_grain_intervention else price_limits.get('min', 1.0)
            max_price = price_limits.get('max', 5.0)
            final_price = max(min_price, min(max_price, new_price))
            
        else:
            # 无供给时使用基准价格
            final_price = base_price
            govt_purchase = 0
            govt_release = 0
            fiscal_warning = None  # @fix: 确保变量初始化
        
        # ========== 7. 更新市场价格 ==========
        self.market_prices['crop_price'] = final_price
        
        # ========== 8. 记录与调试 ==========
        if self.current_step in [0, 1, 5, 10, 20, 50, 99] or govt_purchase > 0 or govt_release > 0:
            intervention_type = "托市收购" if govt_purchase > 0 else ("抛储" if govt_release > 0 else "无干预")
            print(f"[Step {self.current_step}] 粮食市场统一出清: 供给={total_supply:.0f}kg, "
                  f"均衡价={final_price:.2f}元/斤, {intervention_type}")
            if govt_purchase > 0:
                print(f"  → 政府收购={govt_purchase:.0f}kg, 粮储={self.government_grain_storage:.0f}kg")
                if fiscal_warning:
                    print(f"  [WARN] {fiscal_warning}")
            if govt_release > 0:
                print(f"  → 政府抛储={govt_release:.0f}kg, 粮储={self.government_grain_storage:.0f}kg")
        
        # 记录市场数据
        self.grain_market_history.append({
            'time_step': self.current_step,
            'grain_supply': total_supply,
            'market_price_raw': market_price_raw if total_supply > 0 and base_demand > 0 else base_price,
            'grain_price': final_price,
            'govt_purchase': govt_purchase,
            'govt_release': govt_release,
            'govt_storage': self.government_grain_storage,
            'needed_purchase': needed_purchase,
            'support_coverage_ratio': support_coverage_ratio,
            'intervention_shortfall': intervention_shortfall,
            'administrative_grain_price_floor': grain_price_floor,
            'disable_grain_intervention': disable_grain_intervention,
            'intervention_type': 'purchase' if govt_purchase > 0 else ('release' if govt_release > 0 else 'none'),
            'fiscal_warning': fiscal_warning
        })

    def _update_grain_market(self, model):
        """
        @deprecated: 旧版粮食市场出清（两步走）
        保留作为备用，新版使用 _update_grain_market_with_intervention
        
        粮食市场出清（考虑政府价格底线）
        @improvement: P0优先级 - 实现内生价格形成机制
        基于供需关系动态调整价格
        """
        # 保留旧接口名，但统一委托到当前主路径，
        # 避免旧逻辑里的无限收购和硬编码库容再次生效。
        self._update_grain_market_with_intervention(model)
        return

        # 检查是否启用内生价格机制
        market_config = self.config.get('environment', {}).get('market', {})
        use_endogenous = market_config.get('use_endogenous_pricing', True)
        
        # @fix: 使用上一期的实际产出作为本期的供给
        total_supply = 0
        farmer_supply = 0
        enterprise_supply = 0
        
        # 农户供给
        for farmer in model.farmers:
            if hasattr(farmer, "household_actual_grain_share"):
                farmer_supply += farmer.grain_production
            elif hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Grain":
                if hasattr(farmer, 'last_yield'):
                    farmer_supply += farmer.last_yield
        
        # 企业供给
        # @fix 2024-11-20: 简化代码，移除复杂的if-else嵌套
        for enterprise in model.agricultural_enterprises:
            # 企业使用last_yields字典存储上一期产量
            if hasattr(enterprise, 'last_yields') and isinstance(enterprise.last_yields, dict):
                grain_yield = enterprise.last_yields.get('grain', 0)
                enterprise_supply += grain_yield
        
        total_supply = farmer_supply + enterprise_supply
        
        # @fix 2024-11-20: 移除向后兼容代码，简化逻辑
        # 配置中use_endogenous_pricing默认为true，固定价格逻辑不再需要
        if use_endogenous:
            # @improvement: 内生价格形成机制
            # 1. 获取基础需求量和价格弹性
            # @fix 2024-11-18: 大幅降低基础需求（8,000,000→400,000）
            # 计算依据：300农户 × 4人/户 × 280kg/人/年 ≈ 336,000 kg
            # 之前设置800万kg导致supply_demand_ratio极低，价格暴涨触及上限
            base_demand = market_config.get('base_grain_demand', 400000)  # 从配置读取
            price_elasticity = market_config.get('grain_price_elasticity', -0.3)  # 从配置读取
            
            # 2. 计算供需比
            if base_demand > 0:
                supply_demand_ratio = total_supply / base_demand
            else:
                supply_demand_ratio = 1.0
            
            # @debug: 打印供需信息（关键步骤）
            # @fix 2024-11-20: 添加调试输出，检查配置是否生效
            if self.current_step in [0, 1, 10, 50, 99]:
                print(f"[Step {self.current_step}] 粮食供给={total_supply:.0f}kg (农户{farmer_supply:.0f}kg + 企业{enterprise_supply:.0f}kg), 需求={base_demand:.0f}kg, 供需比={supply_demand_ratio:.4f}")
                if self.current_step == 1:
                    # 首次运行时输出配置来源，帮助诊断
                    print(f"[DEBUG] base_grain_demand来源: config['environment']['market']['base_grain_demand'] = {market_config.get('base_grain_demand', '未设置')}")
            
            # 3. 计算市场出清价格（基于价格弹性）
            # 价格 = 基础价格 × (供需比 ^ (1/弹性))
            # @fix 2024-11-20: 移除过度clip，改为警告机制
            # 专家审阅意见：过度clip掩盖了模型结构不稳定的事实
            # 如果供需比经常跑出合理范围，说明产能设定与基础需求在数量级上没对齐
            # 应该回到initial_conditions.py校准初始产能与需求，而不是用代码强行拉回
            base_price = market_config.get('base_prices', {}).get('grain', 1.25)
            if supply_demand_ratio > 0 and price_elasticity != 0:
                # @fix 2024-11-26: 从配置读取供需比限制范围，替代硬编码值
                ratio_limits = market_config.get('supply_demand_ratio_limits', {})
                # @fix 2024-11-29: 修正ratio_limits的读取方式（嵌套字典）
                grain_ratio_limits = ratio_limits.get('grain', {})
                min_ratio = grain_ratio_limits.get('min', 0.3)
                max_ratio = grain_ratio_limits.get('max', 3.0)
                
                # @fix: 如果供需比超出合理范围，记录警告（而非强制clip）
                if supply_demand_ratio < 0.1 or supply_demand_ratio > 10.0:
                    if self.current_step % 10 == 0:  # 每10步警告一次，避免日志爆炸
                        print(f"[WARNING] Step {self.current_step}: 供需比={supply_demand_ratio:.4f}超出合理范围[0.1, 10]")
                        print(f"  这表明初始产能与需求在数量级上未对齐，请检查initial_conditions.py的校准")
                        print(f"  当前供给={total_supply:.0f}kg, 需求={base_demand:.0f}kg")
                
                # @fix 2024-11-26: 限制供需比在配置的范围内，避免价格公式爆炸
                safe_ratio = max(min_ratio, min(max_ratio, supply_demand_ratio))
                # 使用绝对值更大的价格弹性（至少-0.3），使价格变化更平滑
                effective_elasticity = price_elasticity
                if abs(effective_elasticity) < 0.3:
                    effective_elasticity = -0.3 if effective_elasticity < 0 else 0.3
                market_clearing_price = base_price * (safe_ratio ** (1.0 / effective_elasticity))
                
                # @fix 2024-11-26: 从配置读取价格限制，确保价格在合理范围内
                price_limits = market_config.get('price_limits', {})
                # @fix 2024-11-29: 修正price_limits的读取方式（嵌套字典）
                grain_price_limits = price_limits.get('grain', {})
                min_clearing_price = grain_price_limits.get('min', 0.8)
                max_clearing_price = grain_price_limits.get('max', 5.0)
                market_clearing_price = max(min_clearing_price, min(max_clearing_price * 0.5, market_clearing_price))
                
                # 仅在极端情况下（价格>10倍基础价格）进行保护，并记录警告
                max_reasonable_price = base_price * 10.0  # 允许价格波动到10倍（比之前的3倍更宽松）
                if market_clearing_price > max_reasonable_price:
                    if self.current_step % 10 == 0:
                        print(f"[WARNING] Step {self.current_step}: 理论价格={market_clearing_price:.2f}元过高（>10倍基础价格）")
                        print(f"  这表明供需严重失衡，请检查模型初始条件和产能设定")
                    # 仅在极端情况下限制价格，但记录这是异常情况
                    market_clearing_price = max_reasonable_price
            else:
                market_clearing_price = base_price
            
            # 4. 应用价格粘性（避免过度波动）
            # @fix 2024-11-20: 降低价格粘性（0.85→0.8），让价格更快响应供需变化
            # 原值0.85导致价格变化极慢（只有15%的新价格反映），价格被锁定
            # @fix 2024-11-20: 默认值应与配置文件一致(0.8)
            price_stickiness = market_config.get('price_stickiness', {}).get('grain', 0.8)
            current_price = self.market_prices['crop_price']
            
            # @debug: 打印价格计算（关键步骤）
            if self.current_step in [0, 1, 10, 50, 99] or self.current_step % 5 == 0:
                print(f"[Step {self.current_step}] 粮食理论价格={market_clearing_price:.2f}元/斤, 当前价格={current_price:.2f}元/斤")
            
            new_price = price_stickiness * current_price + (1 - price_stickiness) * market_clearing_price
            
            # @critical 2025-12: 实现真正的托市收购机制
            # 理论依据：中国粮食最低收购价政策（2004年起）
            # 粮食价格 = max(托市价, 供需决定的市场价)
            # 这与"如果市场价低于地板价就用地板价"有本质区别：
            # - 旧机制：价格可以在地板价附近波动，政府被动接受
            # - 新机制（托市收购）：当市场价低于托市价时，政府无限量收购，
            #   将过剩供给吸收进政府粮储，从而托起市场价格
            grain_price_floor = self.get_policy_param('grain_price_floor')
            if grain_price_floor and grain_price_floor > 0:
                if new_price < grain_price_floor:
                    # 托市收购启动：政府以托市价无限量收购
                    # 计算政府需要收购的数量（超出需求的供给）
                    oversupply = max(0, total_supply - base_demand)
                    
                    # 政府粮储增加
                    govt_purchase = oversupply * (grain_price_floor - new_price) / grain_price_floor
                    self.government_grain_storage = getattr(self, 'government_grain_storage', 0) + govt_purchase
                    
                    # 托市成功，市场价格被托起到托市价
                    new_price = grain_price_floor
                    
                    if self.current_step % 10 == 0:
                        print(f"[托市收购] Step {self.current_step}: 政府以{grain_price_floor:.2f}元/斤收购，粮储增加{govt_purchase:.0f}kg")
                else:
                    # 市场价高于托市价，政府可能抛储平抑价格
                    # 当市场价高于托市价×1.3时，政府抛储
                    if new_price > grain_price_floor * 1.3:
                        govt_storage = getattr(self, 'government_grain_storage', 0)
                        if govt_storage > 0:
                            # 抛储量：不超过库存的20%
                            sell_quantity = min(govt_storage * 0.2, base_demand * 0.1)
                            self.government_grain_storage = govt_storage - sell_quantity
                            # 抛储压低价格
                            new_price = new_price * 0.95
                            if self.current_step % 10 == 0:
                                print(f"[政府抛储] Step {self.current_step}: 抛售{sell_quantity:.0f}kg平抑价格")
            
            # 6. 价格上下限保护（从配置读取）
            # @fix 2024-11-26: 从配置读取价格限制，替代硬编码值
            price_limits = market_config.get('price_limits', {})
            # @fix 2024-11-29: 修正price_limits的读取方式（嵌套字典）
            grain_price_limits = price_limits.get('grain', {})
            min_price = grain_price_limits.get('min', 1.0)  # 最低价格保护（政府收购价下限）
            max_price = grain_price_limits.get('max', 5.0)  # 最高价格保护
            final_price = max(min_price, min(max_price, new_price))
            self.market_prices['crop_price'] = final_price
            
            # @debug: 打印最终价格（关键步骤）
            if self.current_step in [0, 1, 10, 50, 99]:
                print(
                    f"[Step {self.current_step}] 最终粮价={final_price:.2f}元/斤 "
                    f"{'[WARN] 触及上限!' if final_price >= max_price else ''}\n"
                )
        
        # 记录市场数据
        self.grain_market_history.append({
            'time_step': self.current_step,
            'grain_supply': total_supply,
            'grain_price': self.market_prices['crop_price'],
            'govt_storage': self.government_grain_storage
        })

    def update_labor_market(self, model):
        """
        更新劳动力市场状态
        
        @theory:
        实现劳动力市场出清，考虑农业和非农部门的互动
        
        @critical 2025-12: 增强城市劳动力市场模块
        问题：原模型兼业II型农户非农收入波动太小
        解决：加入工资周期性波动 + 随机冲击
        
        理论依据：
        1. 劳动力需求受宏观经济周期影响（4-5年周期）
        2. 城镇就业存在季节性波动（春节后用工荒、年末裁员）
        3. 经济冲击（如2020年疫情）造成突发性失业
        """
        # ========== 1. 计算经济周期与冲击 ==========
        # @critical 2025-12: 城市劳动力市场周期性波动
        labor_market_config = self.config.get('environment', {}).get('labor_market', {})
        
        # 宏观经济周期（4-5年，振幅±10%）
        cycle_length = labor_market_config.get('business_cycle_length', 5)  # 年
        cycle_amplitude = labor_market_config.get('business_cycle_amplitude', 0.10)
        cycle_phase = 2 * np.pi * (self.current_step % cycle_length) / cycle_length
        cycle_factor = 1 + cycle_amplitude * np.sin(cycle_phase)
        
        # 季节性波动（年内，振幅±5%）
        # 假设一年12步（月），第1-2月（春节）用工荒，第11-12月裁员
        seasonal_amplitude = labor_market_config.get('seasonal_amplitude', 0.05)
        month_in_year = self.current_step % 12
        if month_in_year in [0, 1]:  # 春节后用工荒（1-2月）
            seasonal_factor = 1 + seasonal_amplitude * 1.5  # 工资上涨
        elif month_in_year in [10, 11]:  # 年末裁员（11-12月）
            seasonal_factor = 1 - seasonal_amplitude  # 工资下降
        else:
            seasonal_factor = 1.0
        
        # 随机经济冲击（模拟疫情、金融危机等）
        shock_probability = labor_market_config.get('shock_probability', 0.02)  # 每期2%概率
        shock_magnitude = labor_market_config.get('shock_magnitude', 0.20)  # 冲击幅度±20%
        
        if np_random_value(self) < shock_probability:
            # 发生冲击
            shock_direction = int(np_choice(self, [-1, 1], p=[0.7, 0.3]))  # 70%负面冲击
            shock_factor = 1 + shock_direction * shock_magnitude * float(np_uniform(self, 0.5, 1.0))
            # 记录冲击事件
            self.policy_intervention_history.append({
                'step': self.current_step,
                'type': 'labor_market_shock',
                'direction': 'negative' if shock_direction < 0 else 'positive',
                'magnitude': shock_direction * shock_magnitude
            })
        else:
            shock_factor = 1.0
        
        # 综合调整因子
        market_condition_factor = cycle_factor * seasonal_factor * shock_factor
        
        # ========== 2. 计算劳动力供给 ==========
        total_labor_supply = 0
        for farmer in model.farmers:
            if hasattr(farmer, 'off_farm_labor'):
                total_labor_supply += farmer.off_farm_labor
        
        # ========== 3. 计算劳动力需求 ==========
        # 农业部门劳动力需求
        agricultural_labor_demand = 0
        for enterprise in model.agricultural_enterprises:
            land_area = enterprise.get_operated_land_area()
            agricultural_labor_demand += land_area * 0.2  # 每亩地需要0.2单位劳动力
        
        # 非农部门劳动力需求曲线（受经济周期影响）
        base_demand = self.config['environment']['market'].get('non_farm_labor_base_demand', 100000)
        ai_params = productivity_params(self.config)
        demand_penalty = (
            ai_params.get('non_farm_demand_penalty', 0.0)
            if mechanism_enabled(self.config, "productivity_shock")
            else 0.0
        )
        base_demand *= max(0.0, 1 - demand_penalty)
        
        # @critical 2025-12: 需求受经济周期影响
        base_demand *= market_condition_factor
        
        elasticity = self.config['environment']['market'].get('labor_demand_elasticity', -0.5)
        current_wage = self._sanitize_non_farm_daily_wage(self.market_prices['wage_rate'])
        self.market_prices['wage_rate'] = current_wage
        
        # 计算非农部门需求
        non_farm_labor_demand = base_demand * (current_wage ** elasticity)
        
        # 总劳动力需求
        total_labor_demand = agricultural_labor_demand + non_farm_labor_demand
        
        # ========== 4. 劳动力市场出清 ==========
        adjustment_factor = 0.05  # 调整速度
        
        if self.config.get("runtime_fixed_rule", {}).get("national_v1"):
            self.market_prices['wage_rate'] = self.base_wage
        elif total_labor_demand > 0:
            relative_gap = (total_labor_supply - total_labor_demand) / total_labor_demand
            
            max_change_ratio = 0.15  # 单次最多调整15%
            wage_change = -relative_gap * adjustment_factor * current_wage
            wage_change = max(-max_change_ratio * current_wage, min(max_change_ratio * current_wage, wage_change))
            
            base_wage_floor = self.base_wage
            wage_floor_ratio = ai_params.get('wage_floor', 0.50)
            new_wage = max(base_wage_floor * wage_floor_ratio, current_wage + wage_change)
            
            # 应用工资粘性
            wage_stickiness = 0.75
            self.market_prices['wage_rate'] = self._sanitize_non_farm_daily_wage(
                wage_stickiness * current_wage + (1 - wage_stickiness) * new_wage
            )
        
        # ========== 5. 记录劳动力市场状态 ==========
        self.labor_market_history.append({
            'time_step': self.current_step,
            'labor_supply': total_labor_supply,
            'agricultural_demand': agricultural_labor_demand,
            'non_farm_demand': non_farm_labor_demand,
            'wage': self.market_prices['wage_rate'],
            'gap': (total_labor_supply - total_labor_demand) if total_labor_demand > 0 else 0,
            # @critical 2025-12: 记录周期和冲击因子
            'cycle_factor': cycle_factor,
            'seasonal_factor': seasonal_factor,
            'shock_factor': shock_factor,
            'market_condition': market_condition_factor
        })

    def get_policy_param(self, param_name):
        """获取政策参数"""
        return self.active_policy.get(param_name)

    def get_market_price(self, item_name):
        """获取市场价格"""
        alias_map = {
            'grain': 'crop_price',
            'grain_price': 'crop_price',
            'crop_price': 'crop_price',
            'cash_crop_price': 'cash_crop',
            'land_rental_price': 'land_rent',
            'land_rental': 'land_rent',
            'land_rent': 'land_rent',
            'wage': 'wage_rate',
            'base_non_farm_wage': 'wage_rate',
        }
        canonical_name = alias_map.get(item_name, item_name)
        if canonical_name == 'land_rent':
            return float(getattr(self, 'land_rental_price', self.market_prices.get('land_rent', 0.0)) or 0.0)
        if canonical_name == 'wage_rate':
            sanitized = self._sanitize_non_farm_daily_wage(self.market_prices.get('wage_rate'))
            self.market_prices['wage_rate'] = sanitized
            return sanitized
        return self.market_prices.get(canonical_name)

    def get_market_price_per_kg(self, item_name):
        """获取按元/kg计价的作物价格；非作物价格保持原单位。"""
        alias_map = {
            'grain': 'crop_price',
            'grain_price': 'crop_price',
            'crop_price': 'crop_price',
            'cash_crop_price': 'cash_crop',
        }
        canonical_name = alias_map.get(item_name, item_name)
        price = self.get_market_price(canonical_name)
        if price is None:
            return None
        if canonical_name in {'crop_price', 'cash_crop'}:
            return float(price) * JIN_PER_KG
        return float(price)

    def get_crop_price(self):
        """获取粮食价格（兼容旧API）"""
        return self.market_prices['crop_price']

    def get_base_non_farm_wage_rate(self):
        """获取基础非农工资率（兼容旧API）"""
        self.base_wage = self._sanitize_non_farm_daily_wage(self.base_wage)
        return self.base_wage

    def get_subsidy_rate(self):
        """获取补贴率（兼容旧API）"""
        return self.active_policy.get('grain_subsidies_rate', 0.0)

    def get_expected_subsidy(self, area):
        """获取预期补贴（兼容旧API）"""
        info = self.calculate_subsidy_for_actor(area, actor_type='farmer')
        return info['cash'] + info['in_kind_value']

    def get_subsidy_for_farmer(self, farmer, area):
        """获取农户补贴（普通种粮补贴）"""
        return self.calculate_subsidy_for_actor(area, actor_type='farmer')['cash']

    def get_subsidy_for_enterprise(self, enterprise, area):
        """获取企业补贴（普通种粮补贴）"""
        return self.calculate_subsidy_for_actor(area, actor_type='enterprise')['cash']

    def _get_reserve_config(self):
        agents_cfg = self.config.get('agents', {})
        reserve_cfg = agents_cfg.get('grain_reserve_bureau')
        if isinstance(reserve_cfg, dict) and reserve_cfg:
            return reserve_cfg
        legacy_cfg = agents_cfg.get('grain_reserve', {})
        return legacy_cfg if isinstance(legacy_cfg, dict) else {}

    def _infer_reserve_storage_unit(self, reserve_config):
        explicit_unit = str(reserve_config.get('storage_unit', '') or '').strip().lower()
        if explicit_unit in {'kg', 'kilogram', 'kilograms'}:
            return 'kg'
        if explicit_unit in {'ton', 'tons', 't'}:
            return 'ton'

        numeric_values = []
        for key in ('initial_storage', 'max_storage_capacity', 'min_safety_stock'):
            try:
                numeric_values.append(float(reserve_config.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        if numeric_values and max(numeric_values) <= 1000:
            return 'ton'
        return 'kg'

    def _convert_reserve_storage_to_kg(self, storage_value, storage_unit):
        value = float(storage_value or 0.0)
        return value * 1000.0 if storage_unit == 'ton' else value

    def _get_fallback_reserve_limits(self):
        reserve_cfg = self._get_reserve_config()
        storage_unit = self._infer_reserve_storage_unit(reserve_cfg)
        annual_budget = float(reserve_cfg.get('annual_budget', 1000000) or 1000000)
        max_capacity_kg = self._convert_reserve_storage_to_kg(
            reserve_cfg.get('max_storage_capacity', 500000),
            storage_unit,
        )
        min_safety_stock_kg = self._convert_reserve_storage_to_kg(
            reserve_cfg.get('min_safety_stock', 20000),
            storage_unit,
        )
        return annual_budget, max_capacity_kg, min_safety_stock_kg
    
    def calculate_scale_subsidy(self, grain_area, actor_type='farmer'):
        """
        @new 2025-12-08: 计算规模种粮补贴
        @theory 鼓励规模化、集约化粮食生产的专项补贴
        
        政策依据：
        1. 浙江省2023年度省级规模种粮动态补贴：50亩以上，20元/亩
        2. 四川省荣县2025年种粮大户补贴：30亩以上，200元/亩
        3. 武平县规模种粮补助：30-100亩60元/亩，100亩以上100元/亩
        
        Args:
            grain_area: 实际种粮面积（亩）
            actor_type: 主体类型（farmer/enterprise/cooperative）
        
        Returns:
            dict: {
                'scale_subsidy': 规模补贴金额,
                'qualified': 是否达到门槛,
                'tier': 所属档次,
                'breakdown': 补贴构成明细
            }
        """
        # 从配置获取规模补贴参数
        fiscal_config = resolve_fiscal_system_config(self.config, create=False)
        budget_allocation = fiscal_config.get('budget_allocation', {})
        scale_config = budget_allocation.get('scale_subsidy', {})
        
        # 检查是否启用规模补贴
        if not scale_config.get('enabled', False):
            return {'scale_subsidy': 0.0, 'qualified': False, 'tier': 0, 'breakdown': {}}
        
        # 获取门槛和基础补贴率
        threshold_area = scale_config.get('threshold_area', 30)  # 默认30亩
        base_rate = scale_config.get('base_rate', 200.0)  # 默认200元/亩
        calculation_method = scale_config.get('calculation_method', 'full_area')
        
        # 检查是否达到门槛
        if grain_area < threshold_area:
            return {'scale_subsidy': 0.0, 'qualified': False, 'tier': 0, 'breakdown': {}}
        
        # 计算基础规模补贴
        if calculation_method == 'full_area':
            # 对全部种粮面积发放补贴
            subsidy_area = grain_area
        else:  # 'excess_area'
            # 仅对超过门槛部分发放补贴
            subsidy_area = grain_area - threshold_area
        
        base_subsidy = subsidy_area * base_rate
        tier = 1
        breakdown = {'base': base_subsidy}
        
        # 计算阶梯递增补贴
        tiered_bonus_config = scale_config.get('tiered_bonus', {})
        if tiered_bonus_config.get('enabled', False):
            tiers = tiered_bonus_config.get('tiers', [])
            bonus_total = 0
            for tier_info in tiers:
                tier_threshold = tier_info.get('threshold', 0)
                bonus_rate = tier_info.get('bonus_rate', 0)
                if grain_area >= tier_threshold:
                    # 对超过该档门槛的面积给予额外补贴
                    if calculation_method == 'full_area':
                        bonus_area = grain_area
                    else:
                        bonus_area = grain_area - tier_threshold
                    tier_bonus = bonus_area * bonus_rate
                    bonus_total += tier_bonus
                    tier += 1
                    breakdown[f'tier_{tier_threshold}亩'] = tier_bonus
            
            base_subsidy += bonus_total
        
        # 主体类型调整系数（企业和合作社获得相同待遇）
        subject_adjustment = 1.0
        if actor_type in ['enterprise', 'cooperative']:
            subject_adjustment = 1.0  # 规模补贴不额外区分主体类型
        
        final_subsidy = base_subsidy * subject_adjustment
        
        return {
            'scale_subsidy': final_subsidy,
            'qualified': True,
            'tier': tier,
            'breakdown': breakdown
        }
    
    def get_total_subsidy_for_farmer(self, farmer, grain_area):
        """
        @new 2025-12-08: 获取农户的全部补贴（普通补贴+规模补贴）
        
        Args:
            farmer: 农户智能体
            grain_area: 种粮面积
        
        Returns:
            dict: {
                'basic_subsidy': 普通种粮补贴,
                'scale_subsidy': 规模补贴,
                'total': 补贴总额
            }
        """
        basic_subsidy = self.get_subsidy_for_farmer(farmer, grain_area)
        scale_info = self.calculate_scale_subsidy(grain_area, actor_type='farmer')
        
        return {
            'basic_subsidy': basic_subsidy,
            'scale_subsidy': scale_info['scale_subsidy'],
            'scale_qualified': scale_info['qualified'],
            'total': basic_subsidy + scale_info['scale_subsidy']
        }
    
    def get_total_subsidy_for_enterprise(self, enterprise, grain_area):
        """
        @new 2025-12-08: 获取企业的全部补贴（普通补贴+规模补贴）
        
        Args:
            enterprise: 企业智能体
            grain_area: 种粮面积
        
        Returns:
            dict: {
                'basic_subsidy': 普通种粮补贴,
                'scale_subsidy': 规模补贴,
                'total': 补贴总额
            }
        """
        basic_subsidy = self.get_subsidy_for_enterprise(enterprise, grain_area)
        scale_info = self.calculate_scale_subsidy(grain_area, actor_type='enterprise')
        
        return {
            'basic_subsidy': basic_subsidy,
            'scale_subsidy': scale_info['scale_subsidy'],
            'scale_qualified': scale_info['qualified'],
            'total': basic_subsidy + scale_info['scale_subsidy']
        }

    def calculate_subsidy_for_actor(self, area, actor_type='farmer', policy_override=None):
        """
        计算主体补贴金额
        
        @enhanced 2026-01-03: 添加补贴边际递减机制
        
        理论依据:
        1. 补贴资本化效应 (Ciaian & Swinnen, 2006):
           - 土地市场不完全时，补贴的20-40%被地租吸收
           - 东部土地稀缺，资本化率更高
        2. 满意化阈值 (Simon, 1955):
           - 补贴超过一定水平后，农户敏感度下降
        3. 规模经济约束:
           - 小农户的边际效用更高
        
        Args:
            area: 土地面积
            actor_type: 主体类型 (farmer/enterprise/cooperative)
            policy_override: 可选的政策覆盖配置
        
        Returns:
            Dict: {cash, in_kind_value, in_kind_breakdown, effective_rate, capitalization_loss}
        """
        from core.p3_project_contracts import P3AccountingError, configured_program
        model = getattr(self, 'model', None)
        if model is not None and configured_program(model) is not None:
            raise P3AccountingError("P3 requires a project recipient; actorless capitalization payments are disabled")
        policy = policy_override or self.active_policy or {}
        if area <= 0 or not policy:
            return {'cash': 0.0, 'in_kind_value': 0.0, 'in_kind_breakdown': {}, 
                    'effective_rate': 1.0, 'capitalization_loss': 0.0}

        base_subsidy_per_mu = float(policy.get('base_subsidy_per_mu', 0))
        subsidy_type = policy.get('subsidy_type', 'area_based')

        if subsidy_type == 'tiered_area':
            rate = self._get_tiered_rate(area, policy)
        else:
            rate = float(policy.get('grain_subsidies_rate', 0.0))

        subject_adjustment = 1.0
        active_policy = getattr(self, 'active_policy', {}) or {}
        subject_adjustment_cfg = active_policy.get('subject_adjustment', {})
        if isinstance(subject_adjustment_cfg, dict):
            try:
                subject_adjustment = float(subject_adjustment_cfg.get(actor_type, subject_adjustment) or subject_adjustment)
            except (TypeError, ValueError):
                subject_adjustment = 1.0
        
        # 计算原始补贴金额
        raw_base_value = base_subsidy_per_mu * rate * area * subject_adjustment
        
        # ======= @new 2026-01-03: 补贴边际递减机制 =======
        # 
        # 机制1: 补贴资本化效应 (Ciaian & Swinnen, 2006)
        # 补贴的一部分被地租吸收，有效补贴降低
        capitalization_rate = self._get_subsidy_capitalization_rate(actor_type, area)
        
        # 机制2: 边际效用递减 (满意化阈值)
        # 补贴超过阈值后，效用折扣
        subsidy_per_mu = raw_base_value / area if area > 0 else 0
        utility_discount = self._get_subsidy_utility_discount(subsidy_per_mu, actor_type)
        
        # 计算有效补贴率和资本化损失
        effective_rate = (1 - capitalization_rate) * utility_discount
        capitalization_loss = raw_base_value * capitalization_rate
        
        # 应用边际递减后的实际补贴值
        base_value = raw_base_value * effective_rate
        
        # Record annual/cumulative bookkeeping for formal capitalization outputs.
        if not hasattr(self, 'annual_capitalization_loss'):
            self.annual_capitalization_loss = 0.0
        if not hasattr(self, 'cumulative_capitalization_loss'):
            self.cumulative_capitalization_loss = 0.0
        if not hasattr(self, 'annual_raw_basic_subsidy'):
            self.annual_raw_basic_subsidy = 0.0
        if not hasattr(self, 'cumulative_raw_basic_subsidy'):
            self.cumulative_raw_basic_subsidy = 0.0

        self.annual_capitalization_loss += capitalization_loss
        self.cumulative_capitalization_loss += capitalization_loss
        self.annual_raw_basic_subsidy += raw_base_value
        self.cumulative_raw_basic_subsidy += raw_base_value
        # ======= 边际递减机制结束 =======
        
        delivery_mode = policy.get('delivery_mode', 'cash')
        cash_value = 0.0
        in_kind_value = 0.0
        breakdown = {}

        if delivery_mode == 'cash':
            cash_value = base_value
        elif delivery_mode == 'in_kind':
            in_kind_value, breakdown = self._compute_in_kind_value(policy, area, base_value)
        elif delivery_mode == 'mixed':
            cash_share = float(policy.get('cash_share', 0.5))
            cash_value = base_value * cash_share
            in_kind_comp, breakdown = self._compute_in_kind_value(policy, area, base_value * (1 - cash_share))
            in_kind_value = in_kind_comp
        else:
            cash_value = base_value  # 默认兜底

        return {
            'cash': cash_value, 
            'in_kind_value': in_kind_value, 
            'in_kind_breakdown': breakdown,
            'effective_rate': effective_rate,
            'capitalization_loss': capitalization_loss
        }

    def _get_subsidy_capitalization_rate(self, actor_type: str, area: float) -> float:
        """
        @new 2026-01-03: 计算补贴资本化率
        
        理论: Ciaian & Swinnen (2006), Kirwan (2009)
        - 土地市场越紧张，资本化率越高
        - 大规模主体谈判力更强，资本化率更低
        - 区域差异显著（东部>中部>西部>东北）
        
        Returns:
            float: 资本化率 (0-0.5)
        """
        # @provenance nfps_data_bracketed — NOT a pass-through estimate
        # base_rate=0.22 对应 NFPS p75-p90 区间 (NFPS 2009-2015, N=6329,
        # rent/grain-revenue ratio: p25=0.04, p50=0.08, p75=0.16, p90=0.27, mean=0.126).
        # 文献范围: Ciaian & Swinnen (2006) 0.15-0.30, Kirwan (2009).
        # 优先从 config 读取 (capitalization.base_rate)。
        # 涌现 pass-through 0.01-0.03 (diagnostic) 来自 operational transfer ledger，
        # 尚未替换此参数。统一口径见 data_contracts/parameter_evidence_ledger.csv。
        cap_cfg = self.config.get('capitalization', {}) if hasattr(self, 'config') and self.config else {}
        base_rate = float(cap_cfg.get('base_rate', 0.22))
        
        # 主体类型调整：规模主体谈判力更强
        if actor_type == 'enterprise':
            base_rate *= 0.85
        elif actor_type == 'cooperative':
            base_rate *= 0.90
        
        # 规模调整：面积越大，谈判力越强
        if area > 100:
            scale_factor = 0.7
        elif area > 50:
            scale_factor = 0.85
        elif area > 30:
            scale_factor = 0.95
        else:
            scale_factor = 1.0  # 小农户资本化率最高
        
        # 土地市场紧张度调整（基于流转率）
        transfer_rate = max(0.0, min(1.0, float(getattr(self, 'current_transfer_rate', 0.25) or 0.25)))
        market_factor = 0.92 + 0.28 * transfer_rate
        
        final_rate = base_rate * scale_factor * market_factor
        if mechanism_enabled(self.config, "information_governance"):
            ai_info = information_governance_params(self.config)
            detection_signal = max(
                0.0,
                min(
                    1.0,
                    0.5 * ai_info.get("misallocation_detection_rate", 0.0)
                    + 0.3 * ai_info.get("duplicate_subsidy_detection_rate", 0.0)
                    + 0.2 * ai_info.get("damage_detection_accuracy", 0.0),
                ),
            )
            repair_strength = max(0.0, min(0.8, ai_info.get("leakage_reduction_per_detection", 0.35)))
            final_rate *= 1.0 - detection_signal * repair_strength
        return min(0.45, max(0.08, final_rate))
    
    def _get_subsidy_utility_discount(self, subsidy_per_mu: float, actor_type: str) -> float:
        """
        @new 2026-01-03: 计算补贴效用折扣（满意化阈值）
        
        理论: Simon (1955) 满意化原则, Kahneman敏感度递减
        - 补贴低于阈值时，边际效用高
        - 超过阈值后，边际效用下降
        
        实现: 使用S型函数（logistic）模拟阈值效应
        
        Returns:
            float: 效用折扣系数 (0.6-1.0)
        """
        # 补贴敏感度阈值（元/亩）
        # 基于经济理论：当补贴超过农业纯收入的50%时开始递减
        # 小麦纯收入约500元/亩，阈值设为250元/亩
        if actor_type == 'enterprise':
            threshold = 400  # 企业阈值更高（规模效益）
        else:
            threshold = 250  # 农户阈值较低
        
        # 使用S型衰减函数
        if subsidy_per_mu <= threshold:
            return 1.0  # 阈值以下，完全敏感
        else:
            # 超过阈值部分的效用折扣
            excess_ratio = (subsidy_per_mu - threshold) / threshold
            # 衰减公式：1 / (1 + 0.5 * excess_ratio)
            # 当excess_ratio=1（补贴翻倍）时，效用约67%
            discount = 1 / (1 + 0.5 * excess_ratio)
            return max(0.6, discount)  # 最低60%效用

    def _compute_in_kind_value(self, policy, area, base_value):
        package = policy.get('in_kind_package', {})
        breakdown = {}
        total = base_value
        for key, value in package.items():
            per_mu = float(value)
            component_value = per_mu * area
            breakdown[key] = component_value
            total += component_value
        return total, breakdown

    def _get_tiered_rate(self, area, policy):
        breakpoints = policy.get('tier_breakpoints', [])
        rates = policy.get('tier_rates', [])
        if not rates:
            return float(policy.get('grain_subsidies_rate', 0.0))
        tiers = list(breakpoints)
        tiers.append(float('inf'))
        for idx, threshold in enumerate(tiers):
            if area <= threshold:
                return float(rates[min(idx, len(rates) - 1)])
        return float(rates[-1])

    def get_payment_lag(self):
        return self.payment_lag
            
    def get_market_history(self, market_type='all', steps=5):
        """
        获取市场历史数据
        
        @theory:
        为智能体提供跨期市场数据，支持预期形成和贝叶斯学习
        """
        if market_type == 'labor':
            history = self.labor_market_history[-steps:] if len(self.labor_market_history) > steps else self.labor_market_history
        elif market_type == 'grain':
            history = self.grain_market_history[-steps:] if len(self.grain_market_history) > steps else self.grain_market_history
        elif market_type == 'cash_crop':
            history = self.cash_crop_market_history[-steps:] if len(self.cash_crop_market_history) > steps else self.cash_crop_market_history
        else:
            history = self.market_history[-steps:] if len(self.market_history) > steps else self.market_history
            
        return history
        
    def get_price_statistics(self, crop_type, periods=5):
        """
        获取价格统计数据（均值、标准差等）
        
        @theory:
        为Markowitz投资组合优化提供输入
        """
        if crop_type == 'grain':
            prices = [record['grain_price'] for record in self.grain_market_history[-periods:]] if self.grain_market_history else [self.market_prices['crop_price']]
        else:  # cash_crop
            prices = [record['cash_crop_price'] for record in self.cash_crop_market_history[-periods:]] if self.cash_crop_market_history else [self.market_prices['cash_crop']]
        
        # 计算统计量
        mean = np.mean(prices) if prices else (self.market_prices['crop_price'] if crop_type == 'grain' else self.market_prices['cash_crop'])
        # @fix: 设定不对称的初始风险感知
        if len(prices) > 1:
            std = np.std(prices)
        else:
            if crop_type == 'grain':
                std = mean * 0.1 # 粮食风险低
            else:
                std = mean * 0.2 # 经济作物风险高
        
        return {
            'mean': mean,
            'std': std,
            'min': np.min(prices) if prices else mean * 0.9,
            'max': np.max(prices) if prices else mean * 1.1,
            'history': prices
        } 

    def get_price_statistics_per_kg(self, crop_type, periods=5):
        """获取按元/kg换算后的作物价格统计；保留非作物原单位。"""
        stats = self.get_price_statistics(crop_type, periods=periods)
        if crop_type not in {'grain', 'cash_crop'}:
            return {
                key: (list(value) if isinstance(value, list) else value)
                for key, value in stats.items()
            }

        converted = {}
        for key, value in stats.items():
            if key == 'history':
                converted[key] = [float(item) * JIN_PER_KG for item in value]
            else:
                converted[key] = float(value) * JIN_PER_KG
        return converted
