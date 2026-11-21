# @refactor: 新增构建器，将模型初始化逻辑分离，使MainModel更纯粹
# @new 2025-11-30: 集成配置验证器，消除硬编码
# @update 2025-12-23: 集成NFPS数据驱动的初始条件，替代CLES数据
# --- core/model_builder.py ---

import json
import math
import random
from core.cfps_livelihood_alignment import load_canonical_livelihood_distribution
from pathlib import Path

import numpy as np
from mesa.datacollection import DataCollector

from agents.enterprise_agent import AgriculturalEnterpriseAgent, EnterpriseType
from agents.farmer_agent import FarmerAgent
from agents.government_agent import GovernmentAgent
from agents.grain_reserve_bureau import GrainReserveBureau  # @new 2025-12: 粮食储备局
from core.environment import Environment
from core.fixed_rule_runtime import apply_experiment_overlay, load_experiment_overlay
from core.income_utils import (
    get_farmer_gross_income,
    get_farmer_income_signal,
    get_farmer_livelihood_agriculture_ratio,
    get_farmer_net_income,
)
from core.initial_conditions import apply_data_driven_initial_conditions
# @new 2025-12-23: NFPS数据驱动的初始条件
from core.nfps_initial_conditions import apply_nfps_initial_conditions, NFPSInitialConditions
from core.model import MainModel
from core.reproducibility import (
    RandomController,
    np_choice,
    np_dirichlet,
    np_lognormal,
    np_normal,
    rng_choice,
    rng_randint,
    rng_random,
    rng_shuffle,
    rng_uniform,
)
from modules.land_patch import LandPatch
# @new 2025-11-30: 导入配置验证器
from modules.config_validator import ConfigValidator
# @new 2025-12-22: 导入市场协调器（FIX_REPORT改进）
from core.market_coordinator import MarketCoordinator
from core.config_utils import sync_base_non_farm_wage
from utils.constants import JIN_PER_KG
from utils.policy_runtime import apply_policy_runtime


def _missing_observation() -> float:
    """Missing measurement sentinel for disabled or unavailable modules."""
    return float("nan")


def _latest_environmental_metric(model, key: str) -> float:
    accountant = getattr(model, "environmental_accountant", None)
    history = getattr(accountant, "history", None) if accountant is not None else None
    if not history:
        return _missing_observation()

    latest = history[-1] or {}
    if key not in latest:
        return _missing_observation()
    try:
        return float(latest[key])
    except (TypeError, ValueError):
        return _missing_observation()


def _climate_hazard_coverage(model) -> float:
    climate_manager = getattr(model, "climate_manager", None)
    hazard_map = getattr(climate_manager, "current_hazard_map", None)
    if hazard_map is None:
        return _missing_observation()
    hazard_array = np.asarray(hazard_map)
    if hazard_array.size == 0:
        return _missing_observation()
    return float((hazard_array > 0.1).sum() / hazard_array.size)


def _climate_hazard_mean_severity(model) -> float:
    climate_manager = getattr(model, "climate_manager", None)
    hazard_map = getattr(climate_manager, "current_hazard_map", None)
    if hazard_map is None:
        return _missing_observation()
    hazard_array = np.asarray(hazard_map)
    if hazard_array.size == 0:
        return _missing_observation()
    return float(hazard_array.mean())


def _agent_elite_network_social_capital(agent) -> float:
    if not hasattr(agent, "is_village_elite"):
        return 0.0
    elite_flag = 1.0 if bool(getattr(agent, "is_village_elite", False)) else 0.0
    try:
        network_strength = float(getattr(agent, "social_network_strength", 0.0) or 0.0)
    except (TypeError, ValueError):
        network_strength = 0.0
    return elite_flag * network_strength


def _agent_characteristic_social_capital(agent) -> float:
    characteristics = getattr(agent, "characteristics", None)
    if characteristics is None:
        return 0.0
    try:
        return float(getattr(characteristics, "social_capital", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _agent_social_capital(agent) -> float:
    return max(
        _agent_elite_network_social_capital(agent),
        _agent_characteristic_social_capital(agent),
    )


class ModelBuilder:
    """
    模型构建器类，负责根据配置构建和初始化模型实例。
    将繁琐的初始化逻辑从MainModel中分离出来。
    
    @new 2025-11-30: 集成ConfigValidator，强制配置参数读取
    """
    def __init__(self, config=None, n_farmers=None, n_steps=None, seed=None):
        """
        初始化模型构建器
        支持两种模式：
        1. 传统模式：传入config字典
        2. Mesa模式：传入n_farmers, n_steps, seed参数
        """
        if config is None:
            # Mesa模式：从配置文件加载，但允许参数覆盖
            import yaml
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).resolve().parents[1]
            config_path = PROJECT_ROOT / "config" / "master_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 覆盖参数
            if n_farmers is not None:
                config['scale']['n_farmers'] = n_farmers
            if seed is not None:
                config['experiment']['seed'] = seed
            if n_steps is not None:
                config['experiment']['steps'] = n_steps
        
        experiment_cfg = config.get('experiment', {})
        seed_value = experiment_cfg.get('seed')
        # @fix 2026-06-01: 不再无条件用配置种子覆盖全局 RNG。
        # 改为通过 RandomController.apply_config_seed 协调：
        #   - 若调用方在构造本 builder 前调用过 RandomController.set_seed(X)，
        #     则以 X 为准（外部显式种子优先），避免多种子循环被静默回退成 42；
        #   - 否则按配置种子重置（保持旧行为，向后兼容）。
        # 并把最终生效种子回写进 config，使 build() 阶段创建 MainModel 时
        # 传入的 seed 与全局 RNG 一致。
        effective_seed = RandomController.apply_config_seed(seed_value)
        config.setdefault('experiment', {})
        config['experiment']['seed'] = int(effective_seed)
        overlay_path = config.get('experiment', {}).get('overlay_path')
        if overlay_path:
            initial_tenure_mode = config.get('runtime_fixed_rule', {}).get('initial_tenure_mode')
            preproduction_cash = config.get('runtime_fixed_rule', {}).get('preproduction_cash')
            overlay = load_experiment_overlay(overlay_path)
            apply_experiment_overlay(config, overlay)
            if initial_tenure_mode is not None:
                config.setdefault('runtime_fixed_rule', {})['initial_tenure_mode'] = initial_tenure_mode
            if preproduction_cash is not None:
                config.setdefault('runtime_fixed_rule', {})['preproduction_cash'] = preproduction_cash
        
        # @update 2025-12-23: 优先使用NFPS数据驱动的初始条件
        # 数据源优先级：NFPS(2009-2015) > CLES(2020-2021) > 配置文件默认值
        ic_cfg = config.get('initial_conditions', {})
        data_source = ic_cfg.get('data_source', 'nfps').lower()
        
        if data_source == 'nfps':
            # 使用NFPS全国固定观察点数据（推荐）
            self.config, self.initial_condition_stats = apply_nfps_initial_conditions(config)
            self.data_source = 'NFPS 2009-2015'
        elif data_source == 'cles':
            # 使用CLES数据（仅保留用于江苏局部分析）
            import warnings
            warnings.warn(
                "CLES is Jiangsu-only and MUST NOT be used for national calibration, "
                "validation, or policy claims. Use initial_conditions.data_source=nfps. "
                "See project_management/CURRENT_CODEBASE_PAIN_POINT_INVENTORY_20260623.md",
                UserWarning,
            )
            self.config, self.initial_condition_stats = apply_data_driven_initial_conditions(config)
            self.data_source = 'CLES 2020-2021 (JIANGSU-ONLY, NOT NATIONAL)'
        else:
            # 使用配置文件默认值
            self.config = config
            self.initial_condition_stats = {}
            self.data_source = 'Config defaults'

        # @new 2026-06-23: 加载校准参数并覆盖master_config中的手写值
        # 这是校准→模型的桥接层。之前的校准结果（来自NFPS/CFPS/宏观统计）
        # 保存在 calibrated_params_enhanced.json 中但从未被模型加载。
        # 现在默认启用；可通过 config.calibration.apply_calibrated_params=false 关闭。
        self._apply_calibrated_params()
        
        # @new 2025-11-30: 初始化配置验证器
        self.config_validator = ConfigValidator(self.config, strict_mode=False)

    def _apply_calibrated_params(self) -> None:
        """将 calibrated_params_enhanced.json 的校准值覆盖到 master_config 中。

        Ponytail: 不改变 master_config.yaml 文件本身，只在内存中覆盖。
        这是校准→模型的桥接层。之前的校准结果保存在 JSON 中但从未被模型加载。

        覆盖规则：
          - economic_parameters → config['environment']['market']
          - agent_parameters → config['agents']
          - land_parameters → config['land']
          - behavioral_parameters → config['agents']['farmer']['prospect_theory']

        可通过 config.calibration.apply_calibrated_params=false 禁用。
        """
        cal_cfg = self.config.get('calibration', {})
        if not cal_cfg.get('apply_calibrated_params', True):
            return

        cal_path = cal_cfg.get('calibrated_params_path', 'config/calibrated_params_enhanced.json')
        if not Path(cal_path).is_absolute():
            cal_path = Path(__file__).resolve().parents[1] / cal_path
        else:
            cal_path = Path(cal_path)

        if not cal_path.exists():
            return

        try:
            with open(cal_path, 'r', encoding='utf-8') as f:
                calibrated = json.load(f)
        except Exception:
            return

        applied_count = 0

        # --- economic_parameters ---
        econ = calibrated.get('economic_parameters', {})
        for key, spec in econ.items():
            value = spec.get('value') if isinstance(spec, dict) else spec
            if value is None:
                continue
            if key == 'crop_price':
                market_cfg = self.config.setdefault('environment', {}).setdefault('market', {})
                base_prices = market_cfg.setdefault('base_prices', {})
                # calibrated_params_enhanced stores crop_price in 元/kg; Environment
                # keeps active crop_price/base_prices.grain in 元/斤.
                base_prices['grain'] = float(value) / JIN_PER_KG
                applied_count += 1
            elif key == 'base_non_farm_wage_rate':
                wage_value = float(value)
                self.config.setdefault('labor_market', {})['base_non_farm_wage'] = wage_value
                self.config.setdefault('environment', {}).setdefault('market', {})['base_non_farm_wage'] = wage_value
                sync_base_non_farm_wage(self.config)
                applied_count += 1
            elif key == 'production_cost_per_mu':
                self.config.setdefault('production', {})['base_cost_per_mu'] = float(value)
                applied_count += 1

        # --- agent_parameters ---
        agents_cfg = calibrated.get('agent_parameters', {})
        for key, spec in agents_cfg.items():
            value = spec.get('value') if isinstance(spec, dict) else spec
            if value is None:
                continue
            if key == 'farmer_initial_capital':
                self.config.setdefault('agents', {}).setdefault('farmer', {})['initial_capital'] = float(value)
                applied_count += 1
            elif key == 'enterprise_initial_capital':
                self.config.setdefault('agents', {}).setdefault('enterprise', {})['initial_capital'] = float(value)
                applied_count += 1
            elif key == 'farmer_income_2024':
                self.config.setdefault('agents', {}).setdefault('farmer', {})['base_income'] = float(value)
                applied_count += 1

        # --- land_parameters ---
        land = calibrated.get('land_parameters', {})
        for key, spec in land.items():
            value = spec.get('value') if isinstance(spec, dict) else spec
            if value is None:
                continue
            if key == 'patch_area_range':
                if isinstance(value, list) and len(value) == 2:
                    self.config.setdefault('land', {})['min_patch_area'] = float(value[0])
                    self.config.setdefault('land', {})['max_patch_area'] = float(value[1])
                    applied_count += 1
            elif key == 'patches_per_farmer':
                if isinstance(value, list):
                    mean_val = sum(float(v) for v in value) / len(value)
                else:
                    mean_val = float(value)
                self.config.setdefault('land', {})['patches_per_farmer_mean'] = mean_val
                applied_count += 1
            elif key == 'base_rental_price':
                self.config.setdefault('land', {})['base_rental_price'] = float(value)
                applied_count += 1

        # --- behavioral_parameters ---
        behav = calibrated.get('behavioral_parameters', {})
        if behav and isinstance(behav, dict):
            prospect = self.config.setdefault('agents', {}).setdefault('farmer', {}).setdefault('prospect_theory', {})
            for key, spec in behav.items():
                value = spec.get('value') if isinstance(spec, dict) else spec
                if value is None:
                    continue
                if 'lambda' in key or 'loss_aversion' in key:
                    prospect['loss_aversion'] = float(value)
                    applied_count += 1
                elif 'curvature' in key or 'alpha' in key:
                    prospect['curvature'] = float(value)
                    applied_count += 1
                elif 'weighting' in key or 'probability' in key:
                    prospect['probability_weighting'] = float(value)
                    applied_count += 1

        # --- 记录溯源 ---
        self.config.setdefault('_calibration_bridge', {})
        self.config['_calibration_bridge'] = {
            'source_file': str(cal_path),
            'parameters_applied': applied_count,
            'data_source': calibrated.get('metadata', {}).get('data_source', 'unknown'),
        }

    def _resolve_project_path(self, raw_path, *, default: Path | None = None) -> Path | None:
        if raw_path in (None, ""):
            return default

        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            return path_obj

        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / path_obj,
            project_root / "config" / path_obj,
            Path.cwd() / path_obj,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _get_simulation_base_year(self) -> int:
        population_cfg = self.config.get('population_dynamics', {}) or {}
        macro_anchor_cfg = population_cfg.get('macro_anchor', {}) or {}
        base_year = macro_anchor_cfg.get('base_year')
        if base_year is None:
            base_year = (self.config.get('experiment', {}) or {}).get('base_year')
        try:
            return int(base_year or 2010)
        except (TypeError, ValueError):
            return 2010

    def build(self):
        """根据配置构建并返回一个完整的MainModel实例"""
        
        # @new 2025-11-30: 验证关键配置参数
        self._validate_critical_config()

        policy_cfg = self.config.get('policy', {}) or {}
        apply_policy_runtime(
            self.config,
            scenario_id=policy_cfg.get('active_scenario', 'baseline_cash'),
            overrides=policy_cfg.get('overrides', {}) or {},
        )
        
        seed = self.config['experiment']['seed']
        environment = Environment(self.config)
        
        # 1. 创建模型核心
        model = MainModel(
            config=self.config,
            environment=environment,
            seed=seed
        )

        # 1.1 载入补贴政策库并设定初始场景
        policy_library = self._load_policy_library()
        model.policy_library = policy_library
        active_scenario_id = self.config.get('policy', {}).get('active_scenario', 'baseline_cash')
        policy_overrides = self.config.get('policy', {}).get('overrides', {}) or {}
        resolved_policy = self._resolve_policy_state(policy_library, active_scenario_id, policy_overrides)
        model.active_policy_id = active_scenario_id
        model.policy_overrides = policy_overrides
        model.environment.set_active_policy(resolved_policy, active_scenario_id)
        model.environment.manual_subsidy_dispatch = True

        # 2. 创建政府
        government = self._create_government(model)
        model.government = government

        # 3. 创建农户和土地
        self._create_farmers_and_land(model)
        
        # @new 2025-12-18: 记录初始总土地面积（用于计算流转率）
        if hasattr(model, 'land_patches') and model.land_patches:
            model.initial_total_land_area = sum(p.area for p in model.land_patches)
        else:
            # 从配置估算
            n_farmers = self.config['scale']['n_farmers']
            avg_land = self.config['agents']['farmer'].get('initial_land_mean', 15)
            model.initial_total_land_area = n_farmers * avg_land

        if hasattr(model, 'transfer_rate_control'):
            model.transfer_rate_control['baseline_total_land_area'] = model.initial_total_land_area
            model.transfer_rate_control['observed_total_land_area'] = model.initial_total_land_area
            model.transfer_rate_control['total_land_area'] = model.initial_total_land_area

        # 4. 创建企业
        self._create_enterprises(model)
        
        # 4.5 创建村集体（新增）
        self._create_village_collective(model)
        
        # 4.6 创建粮食储备局（新增）
        self._create_grain_reserve_bureau(model)
        
        # 5. 构建社会网络
        self._build_social_networks(model)

        # 6. 设置数据收集器
        self._setup_data_collector(model)
        
        # @new 2025-11-29: 初始化劳动力市场模块
        self._initialize_labor_market(model)
        
        # @new 2025-12: 初始化农业社会化服务市场
        self._initialize_service_market(model)
        
        # @new 2025-12-22: 初始化市场协调器（FIX_REPORT改进）
        self._initialize_market_coordinator(model)
        
        # @new 2026-01-06: 初始化人口动态管理器（宏观锚定+微观演化）
        # 理论依据：Matthews et al. (2007) JASSS - ABM开放人口系统设计
        self._initialize_population_manager(model)
        
        return model
    
    def _initialize_service_market(self, model):
        """
        @new 2025-12: 初始化农业社会化服务市场
        
        @theory 服务规模经济 - "隐性的农业革命"（陈义媛、高原）
        
        核心洞见：
        1. 中国农业现代化不仅是土地流转的故事
        2. 更大的故事是"服务规模"——通过托管实现规模效益
        3. 小农户保留土地，将生产环节外包给专业化服务商
        
        现实参照（2022年数据）：
        - 全国农业社会化服务组织超过104万个
        - 服务面积18.7亿亩次
        - 服务小农户超过8900万户
        """
        # 配置在 production.agricultural_service 下
        prod_config = self.config.get('production', {})
        service_config = prod_config.get('agricultural_service', {})
        if not service_config.get('enabled', False):
            print("[INFO] 农业社会化服务市场已禁用")
            model.service_market = None
            model.service_providers = []
            return
        
        try:
            from modules.agricultural_service import ServiceMarket, ServiceProvider
            
            # 创建服务市场
            model.service_market = ServiceMarket(model, service_config)
            
            # 创建服务提供商
            n_providers = service_config.get('num_providers', 10)
            provider_config = service_config.get('provider_economics', {})
            
            model.service_providers = []
            for i in range(n_providers):
                # 随机位置
                x = rng_randint(model, 0, model.grid.width - 1)
                y = rng_randint(model, 0, model.grid.height - 1)
                
                # 初始服务能力
                capacity_range = provider_config.get('initial_capacity', {'min': 200, 'max': 1000})
                capacity = rng_uniform(model, capacity_range['min'], capacity_range['max'])
                
                provider = ServiceProvider(
                    unique_id=f"service_provider_{i}",
                    model=model,
                    position=(x, y),
                    initial_capacity=capacity,
                    service_radius=provider_config.get('service_radius', 15)
                )
                if self.config.get('unified_paper_model', {}).get('enabled') is True:
                    provider.service_role = 'capacity_matching_environment'
                model.service_providers.append(provider)
            
            total_capacity = sum(p.current_capacity for p in model.service_providers)
            print(f"[INFO] 农业社会化服务市场初始化成功: {n_providers}个服务商, 总能力{total_capacity:.0f}亩")
            
        except Exception as e:
            print(f"[WARNING] 农业服务市场初始化失败: {e}")
            import traceback
            traceback.print_exc()
            model.service_market = None
            model.service_providers = []
    
    def _initialize_labor_market(self, model):
        """
        @new 2025-11-29: 初始化劳动力市场
        @fix 2026-01-03: 升级到v2版本，支持服务市场替代效应
        
        理论：实现农村劳动力市场的完整机制
        1. 农户既是劳动力供给者，也可能是需求者（大户）
        2. 企业使用大型机械 + 雇佣农户劳动力
        3. 劳动力市场匹配后，确定雇佣关系和工资
        4. @new v2: 服务市场替代 - effective_labor_demand = physical_need * (1 - service_substitution_rate)
        """
        try:
            # @critical_fix 2026-01-03: 使用v2版本，支持服务替代效应
            from core.labor_market_v2 import LaborMarketV2
            model.labor_market = LaborMarketV2(model, self.config)
            print("[INFO] 劳动力市场模块v2初始化成功（支持服务替代效应）")
        except Exception as e:
            print(f"[WARNING] 劳动力市场v2初始化失败，回退到v1: {e}")
            try:
                from core.labor_market import LaborMarket
                model.labor_market = LaborMarket(model, self.config)
                print("[INFO] 劳动力市场模块v1初始化成功（兼容模式）")
            except Exception as e2:
                print(f"[WARNING] 劳动力市场初始化完全失败: {e2}")
                model.labor_market = None
    
    def _initialize_market_coordinator(self, model):
        """
        @new 2025-12-22: 初始化市场协调器（FIX_REPORT改进）
        
        @theory 解决市场时序耦合与资本约束问题
        
        核心问题：
        1. 资本约束失效：农户在土地市场花光现金后，劳动力/服务市场仍可"透支"
        2. 互补要素错配：企业租地后发现劳动力短缺，无法退地
        3. 理性预期伪闭环：缺乏内生反馈导致价格预期系统性偏差
        
        解决方案：
        1. 运营资本预留机制：参与土地竞价前冻结部分资金用于后续投入
        2. 预期形成联动：土地出价需考虑劳动力可得性和服务价格预期
        3. 蛛网模型阻尼：引入适应性预期混合，防止市场震荡发散
        """
        try:
            model.market_coordinator = MarketCoordinator(model, self.config)
            print("[INFO] 市场协调器初始化成功（跨市场资本约束+蛛网阻尼）")
        except Exception as e:
            print(f"[WARNING] 市场协调器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            model.market_coordinator = None

    def _initialize_population_manager(self, model):
        """
        @new 2026-01-06: 初始化人口动态管理器
        @update 2026-01-07: 升级到V2版本，支持土地守恒机制
        
        @theory Macro-Anchored Micro-Evolution Framework
        
        设计原则:
        1. 宏观锚定: 使用国家统计年鉴农村户数作为总量约束
        2. 微观演化: 基于Agent属性计算城镇化概率
        3. 动态平衡: 通过Bootstrap重采样生成新进入者
        4. 土地守恒: 退出者土地进入土地库，分配给邻居或新进入者
        
        参考文献:
        - Matthews et al. (2007) JASSS - ABM人口动态设计
        - Berger (2001) Agricultural Systems - MPMAS框架
        - Parker et al. (2003) PNAS - LUCC ABM设计
        
        数据来源组合:
        - 总量趋势: 国家统计年鉴 (config/nbs_rural_population.json)
        - 退出规则: CFPS面板追踪的城镇化概率
        - 新进入者: CFPS截面的Bootstrap重采样 (config/cfps_feature_library.json)
        """
        # 检查是否启用人口动态
        pop_config = self.config.get('population_dynamics', {})
        if not pop_config.get('enabled', False):
            print("[INFO] 人口动态模块已禁用（使用默认固定人口）")
            model.population_manager = None
            return
        
        try:
            # @update 2026-01-07: 优先使用V2版本（带土地守恒）
            try:
                from modules.population_dynamics_v2 import PopulationManagerV2
                use_v2 = True
                print("[INFO] 使用PopulationManagerV2（土地守恒版）")
            except ImportError:
                from modules.population_dynamics import PopulationManager
                use_v2 = False
                print("[INFO] 使用PopulationManager（基础版）")
            
            from pathlib import Path
            
            # 确定区域配置
            region = pop_config.get('region', 'national')
            
            # 计算初始总土地面积
            initial_total_land = sum(
                getattr(f, 'land_area_mu', 0) for f in model.farmers
            )
            
            if use_v2:
                # 特征库路径（V2使用内置默认值如果文件不存在）
                population_cfg = self.config.get('population_dynamics', {}) or {}
                macro_anchor_cfg = population_cfg.get('macro_anchor', {}) or {}
                entry_cfg = population_cfg.get('entry_mechanism', {}) or {}
                macro_data_path = self._resolve_project_path(
                    macro_anchor_cfg.get('data_source'),
                    default=Path(__file__).parent.parent / 'config' / 'nbs_rural_population.json',
                )
                default_feature_lib = Path(__file__).parent.parent / 'config' / 'cfps_feature_library.json'
                entry_feature_library = entry_cfg.get('entrant_feature_library')
                if not entry_feature_library and entry_cfg.get('feature_library'):
                    entry_feature_library = entry_cfg.get('feature_library')
                    print("[INFO] entry_mechanism.feature_library is treated as a legacy alias of entrant_feature_library")
                feature_lib_path = self._resolve_project_path(
                    entry_feature_library,
                    default=default_feature_lib,
                )
                feature_lib_str = str(feature_lib_path) if feature_lib_path and feature_lib_path.exists() else None
                macro_data_str = str(macro_data_path) if macro_data_path and macro_data_path.exists() else None
                
                # 创建V2人口管理器
                model.population_manager = PopulationManagerV2(
                    config_path=macro_data_str,
                    region=region,
                    initial_total_land=initial_total_land,
                    feature_library_path=feature_lib_str
                )
                model.population_manager.base_year = self._get_simulation_base_year()
                model.population_manager.current_year = model.population_manager.base_year
                # 初始化
                model.population_manager.initialize(model.farmers)
            else:
                # 使用旧版
                from modules.population_dynamics import CFPSFeatureLibrary
                feature_lib_path = Path(__file__).parent.parent / 'config' / 'cfps_feature_library.json'
                feature_library = None
                if feature_lib_path.exists():
                    feature_library = CFPSFeatureLibrary(str(feature_lib_path))
                
                model.population_manager = PopulationManager(
                    feature_library=feature_library,
                    region_profile=region
                )
                core_panel = [f for f in model.farmers if getattr(f, 'is_core_panel', True)]
                model.population_manager.register_core_panel(core_panel)
            
            print(f"[INFO] 人口动态管理器初始化成功: region={region}, "
                  f"初始农户={len(model.farmers)}, 初始土地={initial_total_land:.0f}亩")
            
        except ImportError as e:
            print(f"[WARNING] 人口动态模块导入失败: {e}")
            model.population_manager = None
        except Exception as e:
            print(f"[WARNING] 人口动态管理器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            model.population_manager = None

    def _get_data_driven_livelihood_distribution(self):
        """
        @new 2025-12-23: 获取数据驱动的生计策略分布
        @fix 2026-01-04: 直接使用CFPS 2010基期生计分布
        
        数据来源优先级：
        1. CFPS 2010 rural known-only baseline
        2. calibration_targets_dual_source.json / master_config.yaml
        3. Legacy calibration files only as fallback
        
        @critical 这是初始化科学口径的关键：
        - 不能使用硬编码的1/3均匀分布
        - 必须反映中国农村"非农化"趋势的真实结构
        
        @fix 2026-01-06: 调整优先级，使用CFPS 2010基期数据
        - 优先级1: master_config.yaml 中的 livelihood_strategy_distribution (CFPS 2010)
        - 优先级2: calibration_targets_dual_source.json 中的 abm_initialization
        - 优先级3: 旧的 calibration_results_2009_2015.json (NFPS 2015末期，不推荐)
        """
        # 优先级1：从配置文件读取 (推荐 - CFPS 2010基期数据)
        canonical_dist = load_canonical_livelihood_distribution(year=2010, include_unknown=False)
        if canonical_dist:
            print(
                "[INFO] Canonical CFPS livelihood baseline: "
                f"Pure_Farmer={canonical_dist['Pure_Farmer']:.1%}, "
                f"Pluriactive={canonical_dist['Pluriactive']:.1%}, "
                f"Non_Farmer={canonical_dist['Non_Farmer']:.1%}"
            )
            return canonical_dist

        farmer_config = self.config.get('agents', {}).get('farmer', {})
        config_dist = farmer_config.get('livelihood_strategy_distribution', {})
        if config_dist and config_dist.get('Pure_Farmer', 0) > 0.1:  # 确认是CFPS 2010数据(纯农>10%)
            normalized = self._normalize_abm_livelihood_distribution(config_dist)
            print(f"[INFO] 生计策略分布来自CFPS 2010基期数据：Pure_Farmer={normalized['Pure_Farmer']:.1%}, "
                  f"Pluriactive={normalized['Pluriactive']:.1%}, Non_Farmer={normalized['Non_Farmer']:.1%}")
            return normalized
        
        # 优先级2：从calibration_targets_dual_source.json读取
        dual_config_file = Path(__file__).parent.parent / "config" / "calibration_targets_dual_source.json"
        if dual_config_file.exists():
            try:
                import json
                with open(dual_config_file, 'r', encoding='utf-8') as f:
                    dual_config = json.load(f)
                
                abm_init = dual_config.get('abm_initialization', {})
                liv_dist = abm_init.get('livelihood_distribution', {})
                if liv_dist:
                    dist = self._normalize_abm_livelihood_distribution({
                        'Pure_Farmer': liv_dist.get('Pure_Farmer', 0.277),
                        'Pluriactive': liv_dist.get('Pluriactive', 0.235),
                        'Non_Farmer': liv_dist.get('Non_Farmer', 0.422)
                    })
                    print(f"[INFO] 生计策略分布来自CFPS 2010（dual_source配置）：Pure_Farmer={dist['Pure_Farmer']:.1%}, "
                          f"Pluriactive={dist['Pluriactive']:.1%}, Non_Farmer={dist['Non_Farmer']:.1%}")
                    return dist
            except Exception as e:
                print(f"[WARN] 读取dual_source配置失败: {e}")
        
        # 优先级3：使用NFPS校准结果（⚠️ 不推荐，这是2015年末期数据）
        calib_file = Path(__file__).parent.parent / "config" / "calibration_results_2009_2015.json"
        if calib_file.exists():
            try:
                nfps_conditions = NFPSInitialConditions(calibration_file=str(calib_file))
                dist = self._normalize_abm_livelihood_distribution(
                    nfps_conditions.get_national_livelihood_distribution()
                )
                print(f"[INFO] ????????NFPS?????Pure_Farmer={dist['Pure_Farmer']:.1%}, "
                      f"Pluriactive={dist['Pluriactive']:.1%}, Non_Farmer={dist['Non_Farmer']:.1%}")
                return dist
            except Exception as e:
                print(f"[WARN] 读取NFPS校准结果失败: {e}")
        
        # 优先级4：从initial_condition_stats读取（CLES数据 - 不推荐）
        if hasattr(self, 'initial_condition_stats') and self.initial_condition_stats:
            farmer_stats = self.initial_condition_stats.get('farmer', {})
            strategy_dist = farmer_stats.get('strategy_distribution', {})
            if strategy_dist:
                # 转换键名格式
                dist = self._normalize_abm_livelihood_distribution({
                    'Pure_Farmer': strategy_dist.get('PURE_FARMER', 0.04),
                    'Pluriactive': strategy_dist.get('PLURIACTIVE', 0.04),
                    'Non_Farmer': strategy_dist.get('NON_FARMER', 0.92)
                })
                print(f"[INFO] 生计策略分布来自CLES数据")
                return dist
        
        # 优先级5：从配置文件读取（fallback）
        if config_dist:
            print(f"[INFO] 生计策略分布来自配置文件（fallback）")
            return self._normalize_abm_livelihood_distribution(config_dist)
        
        # 优先级6：基于中国农村实际的默认值
        # 参考：2020年全国农民工3.5亿，农村劳动力4.9亿，非农化率约71%
        print(f"[WARN] 使用默认生计策略分布（基于中国农村非农化率约70%的经验值）")
        return self._normalize_abm_livelihood_distribution({
            'Pure_Farmer': 0.10,    # 10% 纯农户
            'Pluriactive': 0.20,    # 20% 兼业户
            'Non_Farmer': 0.70      # 70% 非农户
        })

    def _get_initial_land_size_distribution(self):
        """Return NFPS-informed land-size bins for t=0 microdata sampling."""
        dual_config_file = Path(__file__).parent.parent / "config" / "calibration_targets_dual_source.json"
        if not dual_config_file.exists():
            return None

        try:
            with open(dual_config_file, "r", encoding="utf-8") as handle:
                dual_config = json.load(handle)
        except Exception as exc:
            print(f"[WARN] Failed to read dual-source land-size targets: {exc}")
            return None

        land_size = (
            dual_config.get("abm_initialization", {}) or {}
        ).get("land_size_distribution", {}) or {}
        target = {
            "small": float((land_size.get("small", {}) or {}).get("prob", 0.0) or 0.0),
            "medium": float((land_size.get("medium", {}) or {}).get("prob", 0.0) or 0.0),
            "large": float((land_size.get("large", {}) or {}).get("prob", 0.0) or 0.0),
            "enterprise": float((land_size.get("enterprise", {}) or {}).get("prob", 0.0) or 0.0),
        }
        total = sum(target.values())
        if total <= 0:
            return None
        return {key: value / total for key, value in target.items()}

    @staticmethod
    def _normalize_abm_livelihood_distribution(distribution):
        """Map empirical livelihood shares onto the three ABM strategies."""
        if not isinstance(distribution, dict):
            distribution = {}

        normalized = {
            'Pure_Farmer': max(0.0, float(distribution.get('Pure_Farmer', 0.0) or 0.0)),
            'Pluriactive': max(0.0, float(distribution.get('Pluriactive', 0.0) or 0.0)),
            'Non_Farmer': max(0.0, float(distribution.get('Non_Farmer', 0.0) or 0.0)),
        }
        total = sum(normalized.values())
        if total <= 0:
            return {
                'Pure_Farmer': 0.297,
                'Pluriactive': 0.252,
                'Non_Farmer': 0.451,
            }
        return {key: value / total for key, value in normalized.items()}

    def _create_government(self, model):
        """创建政府智能体"""
        government = GovernmentAgent("government_0", model)
        government.set_active_policy(model.active_policy_id, model.environment.active_policy)
        # @fix 2024-11-20: 更新agent_registry
        model.agent_registry[government.unique_id] = government
        return government


    def _create_farmers_and_land(self, model):
        """
        创建农户和土地
        
        @update 2026-01-07: 支持微观数据驱动的初始化
        数据源优先级：
        1. 微观数据驱动（CFPS户级特征Bootstrap采样）
        2. 宏观分布驱动（从配置的分布参数生成）
        """
        n_farmers = self.config['scale']['n_farmers']
        print(f"正在创建 {n_farmers} 个农户...")
        
        model.farmers = []
        model.land_patches = []
        
        # 检查是否启用微观数据驱动
        ic_cfg = self.config.get('initial_conditions', {})
        use_microdata = ic_cfg.get('microdata_driven', False)
        
        if use_microdata:
            self._create_farmers_from_microdata(model, n_farmers)
        else:
            self._create_farmers_from_distribution(model, n_farmers)
    
    def _create_farmers_from_microdata(self, model, n_farmers):
        """
        @new 2026-01-07: 从微观数据Bootstrap采样创建农户
        
        核心理念：
        - 每个智能体的特征来自真实数据的采样
        - 保留数据中的相关性结构（土地-收入-生计策略的联合分布）
        - 真正体现ABM的异质性
        """
        from core.microdata_agent_initializer import MicrodataAgentInitializer
        
        print("[INFO] 使用微观数据驱动的农户初始化...")
        
        ic_cfg = self.config.get('initial_conditions', {}) or {}
        fixed_cohort = ic_cfg.get('fixed_cohort_path')
        initializer = None
        if fixed_cohort:
            cohort_path = self._resolve_project_path(fixed_cohort)
            if cohort_path is None or not cohort_path.exists():
                raise FileNotFoundError(f"Fixed cohort not found: {fixed_cohort}")
            payload = json.loads(cohort_path.read_text(encoding='utf-8'))
            farmer_features = payload.get('farmers') if isinstance(payload, dict) else None
            if not isinstance(farmer_features, list) or len(farmer_features) != n_farmers:
                raise ValueError(
                    f"Fixed cohort must contain exactly {n_farmers} farmers"
                )
            if any(
                not math.isclose(
                    float(features.get('representation_weight_q', 0.0) or 0.0),
                    1.0,
                    abs_tol=1e-12,
                )
                for features in farmer_features
            ):
                raise ValueError(
                    "Fixed cohort runtime representation weights must equal one after resampling"
                )
        else:
            # 加载初始化器
            try:
                initial_feature_library = ic_cfg.get('initial_feature_library')
                if not initial_feature_library and ic_cfg.get('feature_library'):
                    initial_feature_library = ic_cfg.get('feature_library')
                    print("[INFO] initial_conditions.feature_library is treated as a legacy alias of initial_feature_library")
                feature_library_path = self._resolve_project_path(
                    initial_feature_library,
                    default=Path(__file__).resolve().parents[1] / "config" / "agent_feature_library.json",
                )
                initializer = MicrodataAgentInitializer(
                    feature_library_path=str(feature_library_path) if feature_library_path else None
                )
            except Exception as e:
                print(f"[警告] 微观数据初始化器加载失败: {e}")
                print("[INFO] 回退到分布驱动初始化...")
                self._create_farmers_from_distribution(model, n_farmers)
                return

            # 批量采样农户特征。Unknown 份额不直接建模，
            # 先按三类策略重新归一化后再采样，避免把缺失样本并入Non_Farmer。
            livelihood_weights = self._get_data_driven_livelihood_distribution()
            land_size_weights = self._get_initial_land_size_distribution()
            if land_size_weights:
                print(
                    "[INFO] NFPS-informed land-size baseline: "
                    f"small={land_size_weights['small']:.1%}, "
                    f"medium={land_size_weights['medium']:.1%}, "
                    f"large={land_size_weights['large']:.1%}, "
                    f"enterprise={land_size_weights['enterprise']:.1%}"
                )
            farmer_features = initializer.sample_farmers_batch(
                n=n_farmers,
                livelihood_weights=livelihood_weights,
                land_size_weights=land_size_weights,
            )
        
        # 统计初始分布
        livelihood_counts = {}
        for f in farmer_features:
            lv = f.get('livelihood', 'Unknown')
            livelihood_counts[lv] = livelihood_counts.get(lv, 0) + 1
        
        print(f"[INFO] 微观数据采样完成:")
        for lv, count in livelihood_counts.items():
            print(f"  - {lv}: {count}人 ({count/n_farmers*100:.1f}%)")
        
        # 创建农户智能体
        for i, features in enumerate(farmer_features):
            farmer = FarmerAgent.from_microdata(
                unique_id=str(features.get('cohort_agent_id') or f"farmer_{i}"),
                model=model,
                microdata_features=features,
                config=self.config
            )

            representation_weight = float(
                features.get(
                    'representation_weight_q',
                    self.config.get('scale', {}).get('agent_representation_weight', 1.0),
                )
                or 1.0
            )
            farmer.representation_weight_q = representation_weight
            farmer.source_representation_weight_q = float(
                features.get('source_representation_weight_q', representation_weight)
            )
            farmer.cohort_probability_q = float(
                features.get('cohort_probability_q', 0.0) or 0.0
            )
            for patch in farmer.land_patches:
                patch.area *= representation_weight
            scaled_land_area = sum(float(patch.area) for patch in farmer.land_patches)
            setattr(farmer, 'land_area', scaled_land_area)
            farmer.land_area_mu = scaled_land_area
            farmer._operated_land_area_cache = scaled_land_area
            farmer._operated_land_area_cache_version = None
            
            # 设置地块所有权和经营权
            for patch in farmer.land_patches:
                patch.owner = farmer
                patch.current_operator = farmer
                farmer_region = getattr(farmer, 'region', None)
                if farmer_region in {'东部', '中部', '西部', '东北'}:
                    patch.region = farmer_region
                    patch.region_scope = 'official_macro_region'
                    patch.official_region = farmer_region
                    patch.official_region_scope = 'official_macro_region'
                model.land_patches.append(patch)
            
            model.farmers.append(farmer)
            model.schedule.add(farmer)
            model.agent_registry[farmer.unique_id] = farmer
            
            # 放置农户到网格中
            x = rng_randint(model, 0, model.grid.width - 1)
            y = rng_randint(model, 0, model.grid.height - 1)
            model.grid.place_agent(farmer, (x, y))
        
        # 验证分布
        if initializer is not None:
            validation = initializer.validate_distribution(farmer_features)
            print(f"[INFO] 分布验证:")
            print(f"  - 土地面积: 采样均值={validation['land_area']['sampled_mean']:.1f}亩, "
                  f"原始均值={validation['land_area']['original_mean']:.1f}亩")
    
    def _create_farmers_from_distribution(self, model, n_farmers):
        """
        从配置的分布参数创建农户（原有逻辑）
        """
        
        # 从配置中获取分布参数
        farmer_config = self.config['agents']['farmer']
        land_config = self.config['environment']['land']
        
        # @fix 2025-12-23: 使用数据驱动的生计策略分布，而非硬编码
        # 数据来源优先级：
        # 1. 校准结果(全国固定观察点)
        # 2. CLES数据的initial_condition_stats
        # 3. 配置文件中的livelihood_strategy_distribution
        # 4. 默认分布(fallback)
        
        target_distribution = self._get_data_driven_livelihood_distribution()
        
        # 计算各策略的农户数量
        n_pure = int(n_farmers * target_distribution.get('Pure_Farmer', 0.04))
        n_pluri = int(n_farmers * target_distribution.get('Pluriactive', 0.04))
        n_non = n_farmers - n_pure - n_pluri  # 剩余的都是非农户
        
        # 创建策略分配列表
        from agents.farmer_agent import LivelihoodStrategy
        strategy_assignments = (
            [LivelihoodStrategy.PURE_FARMER] * n_pure +
            [LivelihoodStrategy.PLURIACTIVE] * n_pluri +
            [LivelihoodStrategy.NON_FARMER] * n_non
        )
        
        # 随机打乱顺序（避免空间聚集）
        rng_shuffle(model, strategy_assignments)
        
        print(f"[INFO] 初始生计策略分布（数据驱动）：纯农户{n_pure}人({n_pure/n_farmers*100:.1f}%)，"
              f"兼业{n_pluri}人({n_pluri/n_farmers*100:.1f}%)，"
              f"非农户{n_non}人({n_non/n_farmers*100:.1f}%)")
        
        for i in range(n_farmers):
            # 为每个农户创建地块
            land_patches = self._create_land_patches_for_farmer(
                model,
                land_config,
                owner_id=f"farmer_{i}",
            )
            
            # 生成初始资本（基于配置的分布）
            capital_dist = farmer_config['initial_capital_dist']
            if capital_dist['type'] == 'lognormal':
                initial_capital = float(np_lognormal(
                    model,
                    capital_dist['mean'], 
                    capital_dist['std']
                ))
            else:
                initial_capital = 10000  # 默认值
            
            # 生成农业能力（基于配置的分布）
            ability_dist = farmer_config['farming_ability_dist']
            if ability_dist['type'] == 'uniform':
                farming_ability = rng_uniform(
                    model,
                    ability_dist['min'], 
                    ability_dist['max']
                )
            elif ability_dist['type'] == 'normal':
                mean = ability_dist.get('mean', 1.0)
                std = ability_dist.get('std', 0.1)
                farming_ability = max(0.1, float(np_normal(model, mean, std)))
            else:
                farming_ability = 1.0  # 默认值
            
            # @fix 2025-11-26: 使用预分配的策略（从strategy_assignments列表获取）
            initial_strategy = strategy_assignments[i]
            
            # @new 2025-01: 根据生计策略生成差异化年龄
            # Non_Farmer应该是年轻进城务工者，Pure_Farmer可能更老
            age_by_strategy = farmer_config.get('age_by_livelihood_strategy', {})
            strategy_name = initial_strategy.name  # 'PURE_FARMER', 'PLURIACTIVE', 'NON_FARMER'
            # 配置中用的是 'Pure_Farmer' 格式
            strategy_key = strategy_name.replace('_', ' ').title().replace(' ', '_')
            if strategy_key in age_by_strategy:
                age_cfg = age_by_strategy[strategy_key]
                mean_age = age_cfg.get('mean', 45.0)
                std_age = age_cfg.get('std', 10.0)
            else:
                # 使用默认年龄分布
                default_age = farmer_config.get('age_dist', {})
                mean_age = default_age.get('mean', 45.0)
                std_age = default_age.get('std', 12.0)
            initial_age = int(np.clip(np_normal(model, mean_age, std_age), 20, 80))
                
            # 创建农户并分配地块
            farmer = FarmerAgent(
                f"farmer_{i}", 
                model, 
                land_patches,
                initial_capital=initial_capital,
                farming_ability=farming_ability,
                config=self.config  # @fix: 传入config以便读取参数
            )
            
            # @fix 2025-11-26: 在_initialize_labor_allocation()调用之前设置生计策略
            # 这样_initialize_labor_allocation()会根据正确的策略调整劳动力分配
            farmer.livelihood_strategy = initial_strategy
            
            # @new 2025-01: 覆盖年龄为按策略生成的值
            farmer.age = initial_age
            
            # 重新调用劳动力分配初始化，使用新的策略
            farmer._initialize_labor_allocation()
            
            # 设置地块所有权和经营权
            # @critical 2025-12: 初始状态下，农户自己经营自己的土地
            # current_operator 是补贴发放和收益计算的关键字段
            for patch in land_patches:
                patch.owner = farmer
                patch.current_operator = farmer  # @fix: 初始经营者就是所有者
            
            model.farmers.append(farmer)
            model.schedule.add(farmer)
            model.agent_registry[farmer.unique_id] = farmer
            
            # 放置农户到网格中
            x = rng_randint(model, 0, model.grid.width - 1)
            y = rng_randint(model, 0, model.grid.height - 1)
            model.grid.place_agent(farmer, (x, y))

    def _create_land_patches_for_farmer(self, model, land_config, owner_id=None):
        """
        为农户创建地块
        
        @critical 2025-12: 使用对数正态分布替代均匀分布
        问题：原模型土地分配是均匀的，高估规模经济效应
        解决：使用对数正态分布，校准到中国耕地基尼系数0.58
        
        理论依据：
        1. 中国耕地基尼系数约0.58（农业农村部2020）
        2. 对数正态分布更符合现实土地分布（少数大户，多数小农）
        3. σ≈1.0时，对数正态分布的基尼系数≈0.52-0.60
        """
        patches = []
        
        # 从配置获取分布参数
        patches_dist = land_config['patches_per_farmer_dist']
        area_dist = land_config['area_dist']
        soil_dist = land_config['soil_quality_dist']
        
        # 新增：获取气候区和产量配置
        climate_zone = land_config.get('climate_zone', 'temperate') 
        yield_profiles = land_config.get('yield_profiles', {
            'Grain': [(800, 0.2), (1000, 0.6), (1200, 0.2)],
            'Cash_Crop': [(600, 0.2), (800, 0.6), (1000, 0.2)]
        })
        
        # 确定地块数量
        if patches_dist['type'] == 'randint':
            num_patches = rng_randint(model, patches_dist['min'], patches_dist['max'])
        else:
            num_patches = rng_randint(model, 1, 3)  # 默认值
            
        # @fix 2026-01-04: 对数正态分布生成农户总土地面积
        # 之前的bug：每个地块独立采样导致总面积过大
        # 修正：先生成总面积，再分配到各地块
        if area_dist['type'] == 'lognormal':
            mu = area_dist.get('mu', 1.5)
            sigma = area_dist.get('sigma', 1.0)
            min_area = area_dist.get('min', 0.5)
            max_area = area_dist.get('max', 200)
            
            # 生成农户总土地面积
            total_area = float(np_lognormal(model, mu, sigma))
            total_area = max(min_area, min(max_area, total_area))
        elif area_dist['type'] == 'uniform':
            total_area = rng_uniform(model, area_dist['min'], area_dist['max'])
        else:
            mu = 1.5
            sigma = 1.0
            total_area = float(np_lognormal(model, mu, sigma))
            total_area = max(0.5, min(200, total_area))

        representation_weight = float(
            self.config.get('scale', {}).get('agent_representation_weight', 1.0) or 1.0
        )
        total_area *= representation_weight
        
        # 将总面积分配到各地块（使用狄利克雷分布）
        if num_patches == 1:
            patch_areas = [total_area]
        else:
            # 狄利克雷分布生成比例，参数为1代表均匀分布
            proportions = np_dirichlet(model, [1] * num_patches)
            patch_areas = [total_area * p for p in proportions]
            
        # 创建地块
        for patch_index, area in enumerate(patch_areas):
            # 确保每个地块至少有0.1亩
            area = max(0.1, area)
            
            # 确定土壤质量
            if soil_dist['type'] == 'uniform':
                soil_quality = rng_uniform(model, soil_dist['min'], soil_dist['max'])
            else:
                soil_quality = rng_uniform(model, 0.5, 1.0)  # 默认值
                
            # 创建地块
            patch = LandPatch(
                area=area,
                soil_quality=soil_quality,
                climate_zone=climate_zone,
                yield_profiles=yield_profiles,
                coordinates=(
                    rng_randint(model, 0, model.grid.width - 1),
                    rng_randint(model, 0, model.grid.height - 1)
                ),
                model=model
            )
            if owner_id is not None:
                patch.patch_id = f"{owner_id}_patch_{patch_index}"
            
            patches.append(patch)
            model.land_patches.append(patch)
            
        return patches

    def _create_enterprises(self, model):
        """
        创建企业智能体
        
        @refactor 2025-12: 重构企业类型系统
        - 移除FAMILY_FARM_LARGE（家庭农场归并到FarmerAgent）
        - 企业仅保留农业公司(AGRI_CORPORATION)和合作社(COOPERATIVE)两种类型
        """
        scale_cfg = self.config.get('scale', {}) or {}
        n_farmers = int(scale_cfg.get('n_farmers', 0) or 0)
        enterprise_ratio = scale_cfg.get('enterprise_ratio', None)
        if enterprise_ratio is not None:
            try:
                n_enterprises = int(round(n_farmers * float(enterprise_ratio)))
            except (TypeError, ValueError):
                n_enterprises = int(scale_cfg.get('n_enterprises', 0) or 0)
        else:
            n_enterprises = int(scale_cfg.get('n_enterprises', 0) or 0)
        n_enterprises = max(0, n_enterprises)
        print(f"正在创建 {n_enterprises} 个农业企业...")
        
        model.agricultural_enterprises = []
        
        # 获取企业类型分布和参数
        enterprise_config = self.config['agents']['enterprise']
        type_dist = enterprise_config['type_distribution']
        types = list(type_dist.keys())
        probabilities = np.array([float(enterprise_config['type_distribution'][t]) for t in types], dtype=float)
        probabilities[probabilities < 0] = 0.0
        total_prob = probabilities.sum()
        if total_prob <= 0:
            raise ValueError("企业类型分布概率总和为0，请检查配置 agents.enterprise.type_distribution")
        if not np.isclose(total_prob, 1.0):
            print(f"[WARN] 企业类型概率和为 {total_prob:.6f}，已自动归一化。")
            probabilities = probabilities / total_prob
        prob_dict = dict(zip(types, probabilities.tolist()))
        print(f"[INFO] 企业类型概率（归一化后）: {prob_dict}，合计 {probabilities.sum():.6f}")
        probabilities = probabilities.tolist()
        
        # @refactor 2025-12: 更新企业类型映射，移除FAMILY_FARM_LARGE
        type_mapping = {
            'AGRI_CORPORATION': EnterpriseType.AGRI_CORPORATION,
            'COOPERATIVE': EnterpriseType.COOPERATIVE
        }
        
        for i in range(n_enterprises):
            # 决定企业类型
            enterprise_type_name = np_choice(model, types, p=probabilities)
            enterprise_type = type_mapping[enterprise_type_name]
            
            # 获取该类型的参数
            type_params = enterprise_config['types'][enterprise_type_name]
            
            # 生成企业参数（基于配置的分布）
            if type_params['initial_capital_dist']['type'] == 'lognormal':
                initial_capital = float(np_lognormal(
                    model,
                    type_params['initial_capital_dist']['mean'],
                    type_params['initial_capital_dist']['std']
                ))
            else:
                initial_capital = 1000000  # 默认值
                
            # 创建企业
            enterprise = AgriculturalEnterpriseAgent(
                f"enterprise_{i}", 
                model,
                enterprise_type=enterprise_type,
                initial_capital=initial_capital
            )
            
            model.agricultural_enterprises.append(enterprise)
            model.schedule.add(enterprise)
            # @fix 2024-11-20: 更新agent_registry
            model.agent_registry[enterprise.unique_id] = enterprise
            
            # 将企业放置在网格中 - 集中在中心区域
            x = rng_randint(model, model.grid.width//4, 3*model.grid.width//4)
            y = rng_randint(model, model.grid.height//4, 3*model.grid.height//4)
            model.grid.place_agent(enterprise, (x, y))

        self._seed_enterprise_operational_land(model)

    def _seed_enterprise_operational_land(self, model):
        """Seed a modest initial operating base for enterprises/cooperatives.

        This represents pre-existing village-level leases/operation contracts.
        It transfers operating rights only; household contract ownership remains
        with farmers. Without this, enterprise agents start at zero land and the
        long-run model mechanically understates their ability to absorb exiting
        households.
        """
        mode = self.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode', 'legacy')
        if mode not in ('legacy', 'A', 'B_star_1y', 'B_star_5y'):
            raise ValueError('Unknown initial_tenure_mode')
        if mode == 'A':
            return
        enterprises = list(getattr(model, 'agricultural_enterprises', []) or [])
        patches = list(getattr(model, 'land_patches', []) or [])
        if not enterprises or not patches:
            return

        enterprise_cfg = self.config.get('agents', {}).get('enterprise', {}) or {}
        dynamics_cfg = enterprise_cfg.get('population_dynamics', {}) or {}
        seed_share = max(0.0, min(0.6, float(dynamics_cfg.get('initial_operated_land_share', 0.0) or 0.0)))
        if seed_share <= 0:
            return

        total_land = float(sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in patches))
        remaining_cap = total_land * seed_share
        if remaining_cap <= 0:
            return

        max_per_enterprise = max(
            1.0,
            float(dynamics_cfg.get('initial_max_area_per_enterprise', 25.0) or 25.0),
        )
        min_owner_retained_area = max(
            0.0,
            float(dynamics_cfg.get('initial_owner_retained_area_mu', 2.0) or 2.0),
        )
        initial_contract_years = max(
            1,
            int(dynamics_cfg.get('initial_operational_contract_years', 30) or 30),
        )
        transfer_cfg = self.config.get('agents', {}).get('farmer', {}).get('land_transfer', {}) or {}
        base_rent = float(transfer_cfg.get('base_rent_price', 420.0) or 420.0)

        def owner_priority(patch):
            owner = getattr(patch, 'owner', None)
            livelihood = getattr(getattr(owner, 'livelihood_strategy', None), 'value', '')
            if livelihood == 'Non_Farmer':
                return 0
            if livelihood == 'Pluriactive':
                return 1
            return 2

        candidate_patches = [
            patch for patch in patches
            if getattr(patch, 'leased_to', None) is None
            and float(getattr(patch, 'area', 0.0) or 0.0) > 0
        ]
        candidate_patches.sort(key=lambda patch: (owner_priority(patch), float(getattr(patch, 'area', 0.0) or 0.0)))

        assigned_by_enterprise = {id(enterprise): 0.0 for enterprise in enterprises}
        enterprise_idx = 0
        assigned_total = 0.0
        for patch in candidate_patches:
            if remaining_cap <= 0:
                break
            area = float(getattr(patch, 'area', 0.0) or 0.0)
            if area <= 0 or area > remaining_cap:
                continue

            owner = getattr(patch, 'owner', None)
            if owner is not None and hasattr(owner, 'get_owned_land_area'):
                try:
                    if float(owner.get_owned_land_area(include_leased_out=True) or 0.0) - area < min_owner_retained_area:
                        continue
                except TypeError:
                    pass
                except Exception:
                    pass

            selected = None
            for _ in range(len(enterprises)):
                enterprise = enterprises[enterprise_idx % len(enterprises)]
                enterprise_idx += 1
                if assigned_by_enterprise[id(enterprise)] + area <= max_per_enterprise:
                    selected = enterprise
                    break
            if selected is None:
                break
            gate = getattr(patch, '_passes_operational_financial_gate', None)
            if callable(gate) and not gate(selected, base_rent):
                continue

            patch.leased_to = selected
            patch.current_operator = selected
            patch.lease_contract_years = max(
                int(getattr(patch, 'lease_contract_years', 0) or 0),
                initial_contract_years,
            )
            patch.rental_price = 0.0
            patch.contracted_rental_price = 0.0
            setattr(patch, 'operational_transfer_active', True)
            setattr(patch, 'operational_transfer_channel', 'initial_enterprise_operational_seed')

            for attr_name in ('leased_patches', 'leased_in_patches'):
                if hasattr(selected, attr_name):
                    target = getattr(selected, attr_name)
                    if patch not in target:
                        target.append(patch)

            assigned_by_enterprise[id(selected)] += area
            assigned_total += area
            remaining_cap -= area

        # Price only after selection, preserving the legacy within-seed mapping.
        if mode in ('B_star_1y', 'B_star_5y'):
            for patch in candidate_patches:
                if getattr(patch, 'operational_transfer_channel', '') == 'initial_enterprise_operational_seed':
                    patch.rental_price = float(patch.area) * base_rent
                    patch.contracted_rental_price = patch.rental_price
                    patch.lease_contract_years = 1 if mode == 'B_star_1y' else 5
                    patch.lease_start_year = 0
                    patch.operational_transfer_channel = 'author_paid_initial_lease'

        for enterprise in enterprises:
            leased_area = sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in (getattr(enterprise, 'leased_patches', []) or [])
            )
            setattr(enterprise, 'total_leased_area', leased_area)
            setattr(enterprise, '_leased_patch_cache_version', None)
            setattr(enterprise, '_operated_land_area_cache_version', None)
            setattr(enterprise, '_operated_land_area_cache', leased_area)

        if assigned_total > 0 and hasattr(model, 'invalidate_land_tenure_caches'):
            model.invalidate_land_tenure_caches()
    
    def _create_village_collective(self, model):
        """
        @new 2025-12: 创建村集体智能体
        专家审阅意见：添加村集体/村委会Actor
        
        理论依据：
        - 约30%的土地流转通过村集体组织
        - 村集体在土地整合、流转监督中发挥重要作用
        """
        from agents.village_collective import VillageCollective, VillageCollectiveRole
        
        # 获取村集体配置
        village_config = self.config.get('agents', {}).get('village_collective', {})
        enable_village = village_config.get('enabled', True)  # 默认启用
        
        if not enable_village:
            model.village_collective = None
            model.village_collectives = []
            model.village_collective_network = {'num_nodes': 0, 'edges': [], 'density': 0.0}
            print("[INFO] 村集体功能已禁用")
            return
        
        num_villages = max(1, int(village_config.get('num_villages', 1)))
        village_names = list(village_config.get('village_names', []) or [])
        positions = self._generate_village_positions(model, num_villages, village_config)
        spatial_config = village_config.get('spatial', {})
        configured_radius = spatial_config.get('service_radius')
        auto_radius = max(
            8,
            int(max(model.grid.width, model.grid.height) / max(2, math.sqrt(num_villages) * 2)),
        )

        villages = []
        for idx in range(num_villages):
            village_name = village_names[idx] if idx < len(village_names) else f"village_{idx + 1}"
            village = VillageCollective(
                f"village_collective_{idx}",
                model,
                config=self.config,
                village_name=village_name
            )
            village.village_index = idx
            if configured_radius is None:
                village.service_radius = auto_radius
            position = positions[idx]
            model.grid.place_agent(village, position)
            village.position = position
            villages.append(village)
            model.agent_registry[village.unique_id] = village

        network_summary = self._configure_village_collective_network(model, villages, village_config)
        model.village_collectives = villages
        model.village_collective = villages if len(villages) > 1 else villages[0]
        model.village_collective_network = network_summary

        print(
            f"[完成] 创建村集体网络: {len(villages)}个节点, "
            f"网络密度={network_summary.get('density', 0.0):.2f}, "
            f"服务半径={villages[0].service_radius if villages else 0}"
        )

    def _generate_village_positions(self, model, num_villages, village_config):
        """Generate stable village positions on the model grid."""
        spatial_config = village_config.get('spatial', {})
        explicit_positions = spatial_config.get('positions', []) or []
        positions = []
        for item in explicit_positions[:num_villages]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                x = max(0, min(model.grid.width - 1, int(item[0])))
                y = max(0, min(model.grid.height - 1, int(item[1])))
                positions.append((x, y))
        if len(positions) == num_villages:
            return positions

        cols = max(1, math.ceil(math.sqrt(num_villages)))
        rows = max(1, math.ceil(num_villages / cols))
        x_spacing = max(1, model.grid.width // (cols + 1))
        y_spacing = max(1, model.grid.height // (rows + 1))
        for idx in range(len(positions), num_villages):
            row = idx // cols
            col = idx % cols
            x = min(model.grid.width - 1, max(0, (col + 1) * x_spacing))
            y = min(model.grid.height - 1, max(0, (row + 1) * y_spacing))
            positions.append((x, y))
        return positions

    def _configure_village_collective_network(self, model, villages, village_config):
        """Build a nearest-neighbor information network among villages."""
        if not villages:
            return {'num_nodes': 0, 'edges': [], 'density': 0.0}

        network_config = village_config.get('network', {})
        nearest_neighbors = max(0, min(len(villages) - 1, int(network_config.get('nearest_neighbors', 2))))
        long_tie_probability = float(network_config.get('long_tie_probability', 0.10))
        edge_keys = set()

        for idx, village in enumerate(villages):
            candidates = []
            for other in villages:
                if other is village:
                    continue
                distance = abs(village.pos[0] - other.pos[0]) + abs(village.pos[1] - other.pos[1])
                candidates.append((distance, str(other.unique_id), other))
            candidates.sort(key=lambda item: (item[0], item[1]))
            neighbors = [item[2] for item in candidates[:nearest_neighbors]]

            if len(villages) > 2 and rng_random(model) < long_tie_probability:
                long_tie_candidates = [item[2] for item in candidates[nearest_neighbors:]]
                if long_tie_candidates:
                    sampled = rng_choice(model, long_tie_candidates)
                    if sampled not in neighbors:
                        neighbors.append(sampled)

            weights = {}
            for neighbor in neighbors:
                distance = abs(village.pos[0] - neighbor.pos[0]) + abs(village.pos[1] - neighbor.pos[1])
                weights[neighbor.unique_id] = 1.0 / max(1.0, float(distance))
                edge_keys.add(tuple(sorted((str(village.unique_id), str(neighbor.unique_id)))))
            village.configure_network(neighbors=neighbors, weights=weights, village_index=idx)

        num_nodes = len(villages)
        possible_edges = max(1, num_nodes * (num_nodes - 1) / 2)
        edges = [{'source': source, 'target': target} for source, target in sorted(edge_keys)]
        return {
            'num_nodes': num_nodes,
            'edges': edges,
            'density': len(edge_keys) / possible_edges,
            'nearest_neighbors': nearest_neighbors,
        }

    def _create_grain_reserve_bureau(self, model):
        """
        @new 2025-12: 创建粮食储备局智能体
        专家审阅意见：补充托市收购机制以支撑最低收购价
        
        理论依据：
        - 2023年托市收购约4000万吨稻谷（约占总产量20%）
        - 当市场价低于最低收购价时，储备局入市收购
        - 财政预算约2000亿元/年，库容上限5000万吨
        
        参考文献：
        - 国家粮食和物资储备局年度报告
        - 粮食最低收购价政策执行情况白皮书
        """
        # 获取粮食储备局配置
        grain_reserve_config = self.config.get('agents', {}).get('grain_reserve_bureau', {})
        enable_reserve = grain_reserve_config.get('enabled', True)  # 默认启用
        
        if not enable_reserve:
            model.grain_reserve_bureau = None
            print("[INFO] 粮食储备局功能已禁用")
            return
        
        # 创建粮食储备局
        bureau = GrainReserveBureau(
            unique_id="grain_reserve_bureau_0",
            model=model,
            config=self.config
        )
        
        model.grain_reserve_bureau = bureau
        model.agent_registry[bureau.unique_id] = bureau
        if getattr(model, 'environment', None) is not None:
            model.environment.government_grain_storage = bureau.get_storage_kg()
        
        print(f"[完成] 创建粮食储备局: 年度预算 {bureau.annual_budget/1e4:.1f}万元, "
              f"最低收购价 {bureau.floor_price:.2f}元/斤, "
              f"最大库容 {bureau.max_storage_capacity:.0f}吨")

    def _build_social_networks(self, model):
        """构建农户社会网络"""
        print("正在构建农户社会网络...")
        for farmer in model.farmers:
            farmer.build_social_network(model.farmers)
        
        # 计算平均网络规模
        if model.farmers:
            avg_size = sum(len(f.social_network) for f in model.farmers) / len(model.farmers)
            print(f"[完成] 社会网络构建完成，平均网络规模: {avg_size:.1f}")
        self._update_network_metrics(model)

    def _update_network_metrics(self, model):
        """根据交易网络计算区域级别的网络密度供政策靶向使用"""
        region_metrics = {}
        for farmer in model.farmers:
            region = model.get_region_label(farmer)
            degree = len(farmer.transaction_neighbors) if getattr(farmer, 'transaction_neighbors', None) else len(farmer.social_network)
            stats = region_metrics.setdefault(region, {'degree_sum': 0.0, 'nodes': 0})
            stats['degree_sum'] += degree
            stats['nodes'] += 1
        network_summary = {}
        for region, stats in region_metrics.items():
            nodes = max(1, stats['nodes'])
            avg_degree = stats['degree_sum'] / nodes
            max_directed_edges = max(1, nodes * (nodes - 1))
            density = min(1.0, stats['degree_sum'] / max_directed_edges)
            network_summary[region] = {
                'avg_transaction_degree': avg_degree,
                'transaction_density': density
            }
        model.network_metrics = network_summary
        if model.government:
            model.government.network_metrics = network_summary

    def _setup_data_collector(self, model):
        """设置数据收集器"""
        model.datacollector = DataCollector(
            model_reporters={
                # 土地结构指标
                "Land_Operated_by_Enterprises": model.get_land_operated_by_enterprises,
                "Land_Distribution_by_Livelihood_Type": model.get_land_distribution_by_livelihood,
                "Gini_Coefficient_of_Land_Operation": model.get_gini_coefficient_land,
                "Land_Transfer_Rate": model.get_land_transfer_rate,
                "Land_Concentration_Top10": model.get_land_concentration_top10,
                "Land_Fragmentation_Index": model.get_land_fragmentation_index,
                
                # 补贴分配与公平性指标
                "Subsidy_Incidence_Ratio": model.get_subsidy_incidence_ratio,
                "Income_Gini_Coefficient_Rural": model.get_income_gini_coefficient,
                
                # 小农生存空间指标
                "Number_of_Pure_Farmers": model.get_number_of_pure_farmers,
                "Smallholder_Exit_Rate": model.get_smallholder_exit_rate,
                
                # 市场统计
                "Land_Rental_Transactions": model.get_land_rental_transactions,
                "Average_Land_Rental_Price": model.get_average_land_rental_price,
                "Village_Node_Count": model.get_village_count,
                "Village_Avg_Registered_Farmers": model.get_average_registered_farmers_per_village,
                "Village_Network_Density": model.get_village_network_density,
                "Village_Facilitated_Transfer_Area": model.get_village_facilitated_transfer_area,
                "Village_CrossVillage_Transfer_Area": model.get_village_cross_village_transfer_area,
                "Village_CrossVillage_Contracts": model.get_village_cross_village_contract_count,
                "Village_Collaboration_Rate": model.get_village_collaboration_rate,
                "Village_Subsidy_PreReview_Coverage": model.get_village_subsidy_pre_review_coverage,
                "Village_Subsidy_PreReview_Approval_Rate": model.get_village_subsidy_pre_review_approval_rate,
                
                # 生计类型分布（Mesa可视化需要）
                "Livelihood_Strategy_Distribution": model.get_livelihood_strategy_distribution,
                "Pure_Farmer": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Pure_Farmer"),
                "Pluriactive": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Pluriactive"),
                "Non_Farmer": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Non_Farmer"),
                
                # 收入指标（Mesa可视化需要）
                "Average_Income": lambda m: sum(get_farmer_income_signal(f) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Gross_Income": lambda m: sum(get_farmer_gross_income(f) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Net_Income": lambda m: sum(get_farmer_net_income(f) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Median_Income": lambda m: sorted(get_farmer_income_signal(f) for f in m.farmers)[len(m.farmers)//2] if m.farmers else 0,
                "Median_Gross_Income": lambda m: sorted(get_farmer_gross_income(f) for f in m.farmers)[len(m.farmers)//2] if m.farmers else 0,
                "Median_Net_Income": lambda m: sorted(get_farmer_net_income(f) for f in m.farmers)[len(m.farmers)//2] if m.farmers else 0,
                
                # 消费和福利指标（新增）
                "Average_Grain_Consumption_kg": lambda m: sum(getattr(f, 'grain_consumption_kg', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Grain_Expenditure": lambda m: sum(getattr(f, 'grain_expenditure', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Total_Consumption": lambda m: sum(getattr(f, 'total_consumption_expenditure', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Engel_Coefficient": lambda m: sum(getattr(f, 'engel_coefficient', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Food_Security_Index": lambda m: sum(getattr(f, 'food_security_index', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Consumer_Surplus": lambda m: sum(getattr(f, 'consumer_surplus', 0) for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Poverty_Rate": lambda m: sum(1 for f in m.farmers if getattr(f, 'is_in_poverty', False)) / len(m.farmers) if m.farmers else 0,
                "Malnutrition_Risk_Rate": lambda m: sum(1 for f in m.farmers if getattr(f, 'malnutrition_risk', False)) / len(m.farmers) if m.farmers else 0,
                "Total_Grain_Consumption": lambda m: sum(getattr(f, 'grain_consumption_kg', 0) for f in m.farmers),
                
                # 基尼系数（Mesa可视化需要）
                "Gini_Coefficient": model.get_income_gini_coefficient,
                
                # 市场价格（Mesa可视化需要）
                "Grain_Price": lambda m: m.environment.get_market_price('crop_price'),
                "Cash_Crop_Price": lambda m: m.environment.get_market_price('cash_crop'),
                "Wage_Rate": lambda m: m.environment.get_market_price('wage_rate'),
                
                # 生计策略转换指标
                "Livelihood_Transition_Rate": model.get_livelihood_transition_rate,
                "Average_Transition_Probability": model.get_average_transition_probability,
                "Consecutive_High_Non_Farm_Ratio": model.get_consecutive_high_non_farm_ratio,
                "Consecutive_Low_Agricultural_Ratio": model.get_consecutive_low_agricultural_ratio,
                
                # 企业策略调整指标
                "Average_Target_Return_Rate": model.get_average_target_return_rate,
                "Average_Performance_Trend": model.get_average_performance_trend,
                "Average_Learning_Rate": model.get_average_learning_rate,
                "Enterprise_Strategy_Adaptation_Rate": model.get_enterprise_strategy_adaptation_rate,
                
                # 新增：作物类型农户生计类型演化指标
                "Grain_Pure_Farmer": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Grain" and f.livelihood_strategy.value == "Pure_Farmer"),
                "Grain_Pluriactive": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Grain" and f.livelihood_strategy.value == "Pluriactive"),
                "Grain_Non_Farmer": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Grain" and f.livelihood_strategy.value == "Non_Farmer"),
                "Cash_Pure_Farmer": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Cash_Crop" and f.livelihood_strategy.value == "Pure_Farmer"),
                "Cash_Pluriactive": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Cash_Crop" and f.livelihood_strategy.value == "Pluriactive"),
                "Cash_Non_Farmer": lambda m: sum(1 for f in m.farmers if hasattr(f, 'planned_crop') and f.planned_crop and f.planned_crop.value == "Cash_Crop" and f.livelihood_strategy.value == "Non_Farmer"),
                
                # @refactor 2025-12: 企业类型演化指标（移除Family_Farms）
                "Agri_Corporations": lambda m: sum(1 for e in m.agricultural_enterprises if hasattr(e, 'enterprise_type') and e.enterprise_type.value == "Agri_Corporation"),
                "Cooperatives": lambda m: sum(1 for e in m.agricultural_enterprises if hasattr(e, 'enterprise_type') and e.enterprise_type.value == "Cooperative"),
                
                # @new 2025-12: 连续变量农户类型分布（6-7分法）
                "Traditional_Subsistence": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "Traditional_Subsistence"),
                "Modern_Commercial": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "Modern_Commercial"),
                "PartTime_Agri_Main": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "PartTime_Agri_Main"),
                "PartTime_NonAgri_Main": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "PartTime_NonAgri_Main"),
                "NewGen_Professional": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "NewGen_Professional"),
                "Migrant_NonFarmer": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "Migrant_NonFarmer"),
                "Absentee_Landlord": lambda m: sum(1 for f in m.farmers if hasattr(f, 'farmer_type') and f.farmer_type.value == "Absentee_Landlord"),
                
                # @new 2025-12: 农户连续变量平均值
                "Avg_Agricultural_Dependence": lambda m: sum(getattr(f, 'characteristics', None).agricultural_dependence if hasattr(f, 'characteristics') and f.characteristics else 0 for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Avg_Livelihood_Diversity": lambda m: sum(getattr(f, 'characteristics', None).livelihood_diversity if hasattr(f, 'characteristics') and f.characteristics else 0 for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Avg_Aging_Rate": lambda m: sum(getattr(f, 'characteristics', None).aging_rate if hasattr(f, 'characteristics') and f.characteristics else 0 for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Avg_Land_Scale": lambda m: sum(getattr(f, 'characteristics', None).land_scale if hasattr(f, 'characteristics') and f.characteristics else 0 for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                
                # @new 2025-12-04: 服务市场统计
                "Service_Market_Contracts": lambda m: m.service_market_stats.get('total_contracts', 0) if hasattr(m, 'service_market_stats') else 0,
                "Service_Market_Area": lambda m: m.service_market_stats.get('total_service_area', 0) if hasattr(m, 'service_market_stats') else 0,
                "Service_Market_Fee": lambda m: m.service_market_stats.get('total_service_fee', 0) if hasattr(m, 'service_market_stats') else 0,
                "Service_Provider_Count": lambda m: len(m.service_providers) if hasattr(m, 'service_providers') else 0,
                "Service_Provider_Avg_Utilization": lambda m: m.get_service_market_statistics().get('average_provider_utilization', 0) if hasattr(m, 'get_service_market_statistics') else 0,
                
                # @new 2025-12-04: 墓地统计（避免幸存者偏差）
                "Graveyard_Total_Exits": lambda m: m.graveyard.get_exit_summary().get('total_exits', 0) if hasattr(m, 'graveyard') and m.graveyard else 0,
                "Graveyard_Farmer_Exits": lambda m: m.graveyard.get_exit_summary().get('farmer_exits', 0) if hasattr(m, 'graveyard') and m.graveyard else 0,
                "Graveyard_Enterprise_Exits": lambda m: m.graveyard.get_exit_summary().get('enterprise_exits', 0) if hasattr(m, 'graveyard') and m.graveyard else 0,
                
                # @new 2025-12-04: 撂荒统计
                "Land_Abandonment_Rate": lambda m: m.get_land_abandonment_rate() if hasattr(m, 'get_land_abandonment_rate') else 0,
                
                # @new 2025-01-02: P2环境核算指标 (Li et al., 2020, PNAS)
                "Total_Carbon_Emission": lambda m: _latest_environmental_metric(m, 'total_carbon_emission'),
                "Total_Nitrogen_Loss": lambda m: _latest_environmental_metric(m, 'total_nitrogen_loss'),
                "Carbon_Intensity": lambda m: _latest_environmental_metric(m, 'carbon_intensity'),
                "Total_Fertilizer_Input": lambda m: _latest_environmental_metric(m, 'total_fertilizer_input'),
                
                # @new 2025-01-02: P3空间气候风险指标
                "Climate_Hazard_Coverage": lambda m: _climate_hazard_coverage(m),
                "Avg_Climate_Shock_Severity": lambda m: _climate_hazard_mean_severity(m),
                
                # @new 2025-01-02: 土地租金指标 (租金挤出效应分析)
                "Avg_Land_Rental_Price": lambda m: m.get_average_land_rental_price() if hasattr(m, 'get_average_land_rental_price') else 0,
                "Rental_Price_Change_Rate": lambda m: m.get_rental_price_change_rate() if hasattr(m, 'get_rental_price_change_rate') else 0,
                
                # @new 2025-01-02: 韧性指标 (系统脆弱性分析)
                "Grain_Production_CV": lambda m: m.get_grain_production_cv() if hasattr(m, 'get_grain_production_cv') else 0,
                "Recovery_Capacity_Index": lambda m: m.get_recovery_capacity_index() if hasattr(m, 'get_recovery_capacity_index') else 0,
                "Ecological_Threshold_Distance": lambda m: m.get_ecological_threshold_distance() if hasattr(m, 'get_ecological_threshold_distance') else 0,
            },
            agent_reporters={
                "Capital": "capital",
                "Total_Income": "total_income",
                "Gross_Income": lambda a: getattr(a, 'gross_income', 0),
                "Net_Income": lambda a: getattr(a, 'net_income', 0),
                "Disposable_Income": lambda a: getattr(a, 'disposable_income', 0),
                "Livelihood_Strategy": lambda a: a.livelihood_strategy.value if hasattr(a, 'livelihood_strategy') else None,
                "Operated_Land_Area": lambda a: a.get_operated_land_area() if hasattr(a, 'get_operated_land_area') else 0,
                "Planned_Crop": lambda a: a.planned_crop.value if hasattr(a, 'planned_crop') and a.planned_crop else None,
                
                # 消费和福利指标（新增）
                "Grain_Consumption_kg": lambda a: getattr(a, 'grain_consumption_kg', 0),
                "Grain_Expenditure": lambda a: getattr(a, 'grain_expenditure', 0),
                "Total_Consumption": lambda a: getattr(a, 'total_consumption_expenditure', 0),
                "Engel_Coefficient": lambda a: getattr(a, 'engel_coefficient', 0),
                "Food_Security_Index": lambda a: getattr(a, 'food_security_index', 0),
                "Consumer_Surplus": lambda a: getattr(a, 'consumer_surplus', 0),
                "Is_In_Poverty": lambda a: getattr(a, 'is_in_poverty', False),
                "Malnutrition_Risk": lambda a: getattr(a, 'malnutrition_risk', False),
                "Household_Size": lambda a: getattr(a, 'household_size', 1),
                
                # 农户生计策略相关指标
                "Agricultural_Income_Ratio": lambda a: get_farmer_livelihood_agriculture_ratio(a),
                "Transition_Probability": lambda a: getattr(a, 'last_transition_probability', 0),
                "Consecutive_High_Non_Farm_Years": lambda a: getattr(a, 'consecutive_high_non_farm_ratio', 0) if hasattr(a, 'consecutive_high_non_farm_ratio') else 0,
                "Elite_Network_Social_Capital": lambda a: _agent_elite_network_social_capital(a),
                
                # 企业策略相关指标
                "Enterprise_Type": lambda a: a.enterprise_type.value if hasattr(a, 'enterprise_type') else None,
                "Target_Return_Rate": lambda a: getattr(a, 'target_return_rate', 0) if hasattr(a, 'target_return_rate') else 0,
                "Performance_Trend": lambda a: getattr(a, 'performance_trend', 0) if hasattr(a, 'performance_trend') else 0,
                "Learning_Rate": lambda a: getattr(a, 'learning_rate', 0) if hasattr(a, 'learning_rate') else 0,
                "ROI": lambda a: getattr(a, 'roi', 0) if hasattr(a, 'roi') else 0,
                
                # @new 2025-12: 连续变量农户特征指标
                "Farmer_Type": lambda a: a.farmer_type.value if hasattr(a, 'farmer_type') and a.farmer_type else None,
                "Farmer_Type_Confidence": lambda a: getattr(a, 'farmer_type_confidence', 0),
                "Agricultural_Dependence": lambda a: a.characteristics.agricultural_dependence if hasattr(a, 'characteristics') and a.characteristics else 0,
                "Livelihood_Diversity": lambda a: a.characteristics.livelihood_diversity if hasattr(a, 'characteristics') and a.characteristics else 0,
                "Aging_Rate": lambda a: a.characteristics.aging_rate if hasattr(a, 'characteristics') and a.characteristics else 0,
                "Land_Scale_Mu": lambda a: a.characteristics.land_scale if hasattr(a, 'characteristics') and a.characteristics else 0,
                "Characteristic_Social_Capital": lambda a: _agent_characteristic_social_capital(a),
                "Social_Capital": lambda a: _agent_social_capital(a),
                "Generation_Type": lambda a: a.characteristics.generation_type if hasattr(a, 'characteristics') and a.characteristics else 0,
            }
        ) 
    
    def _validate_critical_config(self):
        """
        @new 2025-11-30: 验证关键配置参数
        
        使用ConfigValidator确保所有必需参数存在且合理
        消除硬编码，强制从配置读取
        """
        v = self.config_validator
        
        # 验证规模参数
        n_farmers = v.get_validated('scale.n_farmers', default=300, 
                                   type_hint=int, min_value=10, max_value=10000)
        n_enterprises = v.get_validated('scale.n_enterprises', default=80,
                                        type_hint=int, min_value=0, max_value=1000)
        grid_width = v.get_validated('scale.grid_width', default=50,
                                     type_hint=int, min_value=10, max_value=500)
        grid_height = v.get_validated('scale.grid_height', default=50,
                                      type_hint=int, min_value=10, max_value=500)
        
        # 验证生产参数
        grain_yield = v.get_validated('production.grain.yield_per_mu', default=500,
                                      type_hint=(int, float), min_value=100, max_value=2000)
        cash_crop_yield = v.get_validated('production.cash_crop.yield_per_mu', default=800,
                                          type_hint=(int, float), min_value=100, max_value=3000)
        
        # 验证实验参数
        steps = v.get_validated('experiment.steps', default=20,
                               type_hint=int, min_value=1, max_value=1000)
        seed = v.get_validated('experiment.seed', default=42,
                              type_hint=int)
        
        # 记录验证状态
        if v._missing_paths:
            print(f"[CONFIG] 警告: {len(v._missing_paths)}个配置参数使用默认值")
            for path in v._missing_paths[:5]:  # 只显示前5个
                print(f"  - {path}")

    def _load_policy_library(self):
        policy_path = Path(__file__).resolve().parents[1] / "config" / "policy_params.json"
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy configuration not found: {policy_path}")
        with policy_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _resolve_policy_state(self, library, scenario_id, overrides):
        scenarios = library.get("scenarios", {})
        if scenario_id not in scenarios:
            raise KeyError(f"Policy scenario '{scenario_id}' not found in policy_params.json")
        policy_state = dict(scenarios[scenario_id])
        if overrides:
            policy_state.update(overrides)
        scenario_label = str(scenario_id or "").strip().lower()
        explicit_disable = policy_state.get("disable_grain_intervention")
        if explicit_disable is None:
            try:
                floor_value = float(policy_state.get("grain_price_floor", 0.0) or 0.0)
            except (TypeError, ValueError):
                floor_value = 0.0
            try:
                subsidy_rate = float(
                    policy_state.get(
                        "grain_subsidies_rate",
                        policy_state.get("subsidy_rate", 0.0),
                    ) or 0.0
                )
            except (TypeError, ValueError):
                subsidy_rate = 0.0
            disable_grain_intervention = (
                scenario_label in {"no_policy", "none", "market_only"}
                or (floor_value <= 0.0 and subsidy_rate <= 0.0)
            )
        else:
            disable_grain_intervention = bool(explicit_disable) or scenario_label in {
                "no_policy",
                "none",
                "market_only",
            }

        policy_state["disable_grain_intervention"] = disable_grain_intervention
        if disable_grain_intervention:
            policy_state["grain_price_floor"] = 0.0
            policy_state["grain_subsidies_rate"] = 0.0
            policy_state["subsidy_rate"] = 0.0
            policy_state["base_subsidy_per_mu"] = 0.0
            if "annual_subsidy_budget" in policy_state:
                policy_state["annual_subsidy_budget"] = 0.0
        return policy_state
