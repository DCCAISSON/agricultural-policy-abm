# @refactor: 新增构建器，将模型初始化逻辑分离，使MainModel更纯粹
# @new 2025-11-30: 集成配置验证器，消除硬编码
# --- core/model_builder.py ---

import json
import random
from pathlib import Path

import numpy as np
from mesa.datacollection import DataCollector

from agents.enterprise_agent import AgriculturalEnterpriseAgent, EnterpriseType
from agents.farmer_agent import FarmerAgent
from agents.government_agent import GovernmentAgent
from agents.grain_reserve_bureau import GrainReserveBureau  # @new 2025-12: 粮食储备局
from core.environment import Environment
from core.initial_conditions import apply_data_driven_initial_conditions
from core.model import MainModel
from modules.land_patch import LandPatch
# @new 2025-11-30: 导入配置验证器
from modules.config_validator import ConfigValidator

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
        if seed_value is not None:
            np.random.seed(seed_value)
            random.seed(seed_value)
        
        self.config, self.initial_condition_stats = apply_data_driven_initial_conditions(config)
        
        # @new 2025-11-30: 初始化配置验证器
        self.config_validator = ConfigValidator(self.config, strict_mode=False)

    def build(self):
        """根据配置构建并返回一个完整的MainModel实例"""
        
        # @new 2025-11-30: 验证关键配置参数
        self._validate_critical_config()
        
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
                x = random.randint(0, model.grid.width - 1)
                y = random.randint(0, model.grid.height - 1)
                
                # 初始服务能力
                capacity_range = provider_config.get('initial_capacity', {'min': 200, 'max': 1000})
                capacity = random.uniform(capacity_range['min'], capacity_range['max'])
                
                provider = ServiceProvider(
                    unique_id=f"service_provider_{i}",
                    model=model,
                    position=(x, y),
                    initial_capacity=capacity,
                    service_radius=provider_config.get('service_radius', 15)
                )
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
        
        理论：实现农村劳动力市场的完整机制
        1. 农户既是劳动力供给者，也可能是需求者（大户）
        2. 企业使用大型机械 + 雇佣农户劳动力
        3. 劳动力市场匹配后，确定雇佣关系和工资
        """
        try:
            from core.labor_market import LaborMarket
            model.labor_market = LaborMarket(model, self.config)
            print("[INFO] 劳动力市场模块初始化成功")
        except Exception as e:
            print(f"[WARNING] 劳动力市场初始化失败: {e}")
            model.labor_market = None

    def _create_government(self, model):
        """创建政府智能体"""
        government = GovernmentAgent("government_0", model)
        government.set_active_policy(model.active_policy_id, model.environment.active_policy)
        model.schedule.add(government)
        # @fix 2024-11-20: 更新agent_registry
        model.agent_registry[government.unique_id] = government
        return government

    def _create_farmers_and_land(self, model):
        """创建农户和土地"""
        n_farmers = self.config['scale']['n_farmers']
        print(f"正在创建 {n_farmers} 个农户...")
        
        model.farmers = []
        model.land_patches = []
        
        # 从配置中获取分布参数
        farmer_config = self.config['agents']['farmer']
        land_config = self.config['environment']['land']
        
        # @fix 2025-11-26: 使用目标分布初始化生计策略
        # 不依赖随机分配，而是按照预设比例确定性分配
        # @update 2025-11-26: 修改为三种类型各100人（各1/3），观察演化趋势
        target_distribution = {
            'Pure_Farmer': 1/3,     # 100人/300人
            'Pluriactive': 1/3,     # 100人/300人
            'Non_Farmer': 1/3       # 100人/300人
        }
        
        # 计算各策略的农户数量
        n_pure = int(n_farmers * target_distribution['Pure_Farmer'])
        n_pluri = int(n_farmers * target_distribution['Pluriactive'])
        n_non = n_farmers - n_pure - n_pluri  # 剩余的都是非农户
        
        # 创建策略分配列表
        from agents.farmer_agent import LivelihoodStrategy
        strategy_assignments = (
            [LivelihoodStrategy.PURE_FARMER] * n_pure +
            [LivelihoodStrategy.PLURIACTIVE] * n_pluri +
            [LivelihoodStrategy.NON_FARMER] * n_non
        )
        
        # 随机打乱顺序（避免空间聚集）
        random.shuffle(strategy_assignments)
        
        print(f"[INFO] 初始生计策略分布：纯农户{n_pure}人，兼业{n_pluri}人，非农户{n_non}人")
        
        for i in range(n_farmers):
            # 为每个农户创建地块
            land_patches = self._create_land_patches_for_farmer(model, land_config)
            
            # 生成初始资本（基于配置的分布）
            capital_dist = farmer_config['initial_capital_dist']
            if capital_dist['type'] == 'lognormal':
                initial_capital = np.random.lognormal(
                    capital_dist['mean'], 
                    capital_dist['std']
                )
            else:
                initial_capital = 10000  # 默认值
            
            # 生成农业能力（基于配置的分布）
            ability_dist = farmer_config['farming_ability_dist']
            if ability_dist['type'] == 'uniform':
                farming_ability = random.uniform(
                    ability_dist['min'], 
                    ability_dist['max']
                )
            elif ability_dist['type'] == 'normal':
                mean = ability_dist.get('mean', 1.0)
                std = ability_dist.get('std', 0.1)
                farming_ability = max(0.1, np.random.normal(mean, std))
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
            initial_age = int(np.clip(np.random.normal(mean_age, std_age), 20, 80))
                
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
            
            # 放置农户到网格中
            x = random.randint(0, model.grid.width - 1)
            y = random.randint(0, model.grid.height - 1)
            model.grid.place_agent(farmer, (x, y))

    def _create_land_patches_for_farmer(self, model, land_config):
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
            num_patches = random.randint(patches_dist['min'], patches_dist['max'])
        else:
            num_patches = random.randint(1, 3)  # 默认值
            
        # @critical 2025-12: 使用对数正态分布生成土地面积
        # 参数校准说明：
        # - 中国户均耕地约7.5亩（农业农村部2020）
        # - 基尼系数约0.58（接近对数正态分布σ=1.0时的0.52-0.60）
        # - 使用lognormal(μ=1.5, σ=1.0)：均值≈7.5亩，基尼≈0.56
        if area_dist['type'] == 'lognormal':
            # 从配置读取对数正态参数
            mu = area_dist.get('mu', 1.5)      # log尺度均值
            sigma = area_dist.get('sigma', 1.0)  # log尺度标准差
            min_area = area_dist.get('min', 0.5)  # 最小面积
            max_area = area_dist.get('max', 200)  # 最大面积
        else:
            # 默认：校准到基尼系数≈0.58
            mu = 1.5     # E[X] = exp(μ + σ²/2) ≈ 7.5亩
            sigma = 1.0  # Gini ≈ 2Φ(σ/√2) - 1 ≈ 0.56
            min_area = 0.5
            max_area = 200
            
        # 创建地块
        for _ in range(num_patches):
            # @critical 2025-12: 对数正态分布生成面积
            if area_dist['type'] == 'uniform':
                # 保留向后兼容：如果配置指定uniform，使用均匀分布
                area = random.uniform(area_dist['min'], area_dist['max'])
            else:
                # 默认使用对数正态分布
                area = np.random.lognormal(mu, sigma)
                # 截断到合理范围
                area = max(min_area, min(max_area, area))
                
            # 确定土壤质量
            if soil_dist['type'] == 'uniform':
                soil_quality = random.uniform(soil_dist['min'], soil_dist['max'])
            else:
                soil_quality = random.uniform(0.5, 1.0)  # 默认值
                
            # 创建地块
            patch = LandPatch(
                area=area,
                soil_quality=soil_quality,
                climate_zone=climate_zone,
                yield_profiles=yield_profiles,
                coordinates=(
                    random.randint(0, model.grid.width - 1), 
                    random.randint(0, model.grid.height - 1)
                )
            )
            
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
        n_enterprises = self.config['scale']['n_enterprises']
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
            enterprise_type_name = np.random.choice(types, p=probabilities)
            enterprise_type = type_mapping[enterprise_type_name]
            
            # 获取该类型的参数
            type_params = enterprise_config['types'][enterprise_type_name]
            
            # 生成企业参数（基于配置的分布）
            if type_params['initial_capital_dist']['type'] == 'lognormal':
                initial_capital = np.random.lognormal(
                    type_params['initial_capital_dist']['mean'],
                    type_params['initial_capital_dist']['std']
                )
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
            x = random.randint(model.grid.width//4, 3*model.grid.width//4)
            y = random.randint(model.grid.height//4, 3*model.grid.height//4)
            model.grid.place_agent(enterprise, (x, y))
    
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
            print("[INFO] 村集体功能已禁用")
            return
        
        # 创建村集体（简化：一个模型对应一个村集体）
        village = VillageCollective(
            "village_collective_0",
            model,
            config=self.config,
            village_name="示范村"
        )
        
        model.village_collective = village
        model.schedule.add(village)
        model.agent_registry[village.unique_id] = village
        
        print(f"[完成] 创建村集体: {village.village_name}，角色类型: {village.role.value}")

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
        model.schedule.add(bureau)
        model.agent_registry[bureau.unique_id] = bureau
        
        print(f"[完成] 创建粮食储备局: 年度预算 {bureau.annual_budget/1e8:.0f}亿元, "
              f"最低收购价 {bureau.floor_price:.2f}元/kg, "
              f"最大库容 {bureau.max_storage_capacity/1e4:.0f}万吨")

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
                
                # 生计类型分布（Mesa可视化需要）
                "Livelihood_Strategy_Distribution": model.get_livelihood_strategy_distribution,
                "Pure_Farmer": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Pure_Farmer"),
                "Pluriactive": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Pluriactive"),
                "Non_Farmer": lambda m: sum(1 for f in m.farmers if f.livelihood_strategy.value == "Non_Farmer"),
                
                # 收入指标（Mesa可视化需要）
                "Average_Income": lambda m: sum(f.total_income for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Average_Net_Income": lambda m: sum(f.net_income for f in m.farmers) / len(m.farmers) if m.farmers else 0,
                "Median_Income": lambda m: sorted([f.total_income for f in m.farmers])[len(m.farmers)//2] if m.farmers else 0,
                "Median_Net_Income": lambda m: sorted([f.net_income for f in m.farmers])[len(m.farmers)//2] if m.farmers else 0,
                
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
                "Agricultural_Income_Ratio": lambda a: (a.agricultural_income / a.total_income if a.total_income > 0 else 0) if hasattr(a, 'agricultural_income') else 0,
                "Transition_Probability": lambda a: getattr(a, 'last_transition_probability', 0),
                "Consecutive_High_Non_Farm_Years": lambda a: getattr(a, 'consecutive_high_non_farm_ratio', 0) if hasattr(a, 'consecutive_high_non_farm_ratio') else 0,
                "Social_Capital": lambda a: (a.is_village_elite * a.social_network_strength) if hasattr(a, 'is_village_elite') else 0,
                
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
                "Social_Capital": lambda a: a.characteristics.social_capital if hasattr(a, 'characteristics') and a.characteristics else 0,
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
        return policy_state