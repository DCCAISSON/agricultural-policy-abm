# @refactor: 新增环境类，作为智能体感知世界的中介，解耦
# --- core/environment.py ---
import numpy as np

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
            'wage_rate': market_config.get('base_non_farm_wage', 120.0)  # 修复：默认值从15.0更新为120.0
        }
        
        # @improvement: 价格内生机制开关
        self.use_endogenous_pricing = market_config.get('use_endogenous_pricing', True)
        
        # 市场历史记录（用于形成预期）
        self.market_history = []
        self.labor_market_history = []
        self.grain_market_history = []
        self.cash_crop_market_history = []
        
        # 政府干预记录
        self.policy_intervention_history = []
        self.government_grain_storage = 0

    def update_policy(self, government_agent):
        """由政府智能体在每一步开始时更新政策"""
        # 这个方法现在由 government_agent.announce_policy() 直接调用 set_active_policy
        # 保留此方法以保持接口兼容性
        pass

    def set_active_policy(self, policy_state, policy_id=None):
        """更新当前政策状态"""
        self.active_policy_id = policy_id or self.active_policy_id
        self.active_policy = policy_state.copy()
        self.policy = self.active_policy  # 兼容旧接口
        self.payment_lag = int(self.active_policy.get("payment_lag", 0))

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
        
        # 1. 更新农产品市场
        self.update_agricultural_markets(model)
        
        # 2. 更新劳动力市场
        self.update_labor_market(model)
        
        # 记录当前市场状态
        self.market_history.append({
            'step': self.current_step,
            'crop_price': self.market_prices['crop_price'],
            'cash_crop_price': self.market_prices['cash_crop'],
            'wage_rate': self.market_prices['wage_rate']
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
            if hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Cash_Crop":
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
            if hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Grain":
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
        grain_price_floor = self.get_policy_param('grain_price_floor')
        if grain_price_floor is None or grain_price_floor <= 0:
            # 从配置获取默认托市价
            grain_price_floor = market_config.get('price_limits', {}).get('grain', {}).get('min', 1.0)
        
        # 抛储价格（市场价高于此价时政府抛储）
        ceiling_multiplier = market_config.get('grain_ceiling_multiplier', 1.3)
        grain_price_ceiling = grain_price_floor * ceiling_multiplier
        
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
            govt_storage = getattr(self, 'government_grain_storage', 0)
            fiscal_warning = None
            
            if market_price_raw < grain_price_floor:
                # 托市收购启动
                # 政府需求 = 使市场价格达到托市价所需的收购量
                # Q_govt = Q_s - D(P_floor)
                # 其中 D(P_floor) 是托市价格下的市场需求
                demand_at_floor = base_demand * (grain_price_floor / base_price) ** price_elasticity
                needed_purchase = max(0, total_supply - demand_at_floor)
                
                # 调用储备局新接口获取干预能力
                if hasattr(model, 'grain_reserve_bureau'):
                    bureau = model.grain_reserve_bureau
                    intervention_info = bureau.calculate_intervention_demand(market_price_raw, total_supply)
                    
                    # 使用储备局计算的有效干预量（已考虑财政悬崖等约束）
                    govt_purchase = min(needed_purchase, intervention_info['quantity'])
                    
                    # 检查财政预警
                    if intervention_info['fiscal_stress_level'] >= 2:
                        fiscal_warning = f"财政紧张(L{intervention_info['fiscal_stress_level']}), 连续干预{intervention_info['consecutive_years']}年"
                    
                    # 执行实际收购
                    bureau.execute_intervention('purchase', govt_purchase, grain_price_floor)
                else:
                    # 无储备局时的简化处理
                    budget_limit = 500000 / grain_price_floor  # 默认50万预算
                    capacity_limit = 500000 - govt_storage
                    govt_purchase = min(needed_purchase, budget_limit, capacity_limit)
                
                # 更新政府粮储
                self.government_grain_storage = govt_storage + govt_purchase
                
                # 均衡价格 = 托市价（政府收购足够）或略低于托市价（预算约束）
                if govt_purchase >= needed_purchase:
                    equilibrium_price = grain_price_floor
                else:
                    # 预算受限，无法完全托市
                    residual_supply = total_supply - govt_purchase
                    if residual_supply > 0 and base_demand > 0:
                        residual_ratio = residual_supply / base_demand
                        equilibrium_price = base_price * (residual_ratio ** (1.0 / effective_elasticity))
                    else:
                        equilibrium_price = market_price_raw
                    equilibrium_price = max(equilibrium_price, grain_price_floor * 0.95)  # 至少达到托市价95%
                    
            elif market_price_raw > grain_price_ceiling:
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
                    available_release = max(0, bureau.storage - bureau.min_safety_stock)
                    govt_release = min(needed_release, intervention_info['quantity'], available_release)
                    bureau.execute_intervention('release', govt_release, market_price_raw)
                    # 同步粮储到 environment（从bureau读取实际值）
                    self.government_grain_storage = bureau.storage
                else:
                    # 库存约束
                    safety_stock = market_config.get('min_safety_stock', 20000)
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
            min_price = price_limits.get('min', 1.0)
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
                    print(f"  ⚠ {fiscal_warning}")
            if govt_release > 0:
                print(f"  → 政府抛储={govt_release:.0f}kg, 粮储={self.government_grain_storage:.0f}kg")
        
        # 记录市场数据
        self.grain_market_history.append({
            'time_step': self.current_step,
            'grain_supply': total_supply,
            'grain_price': final_price,
            'govt_purchase': govt_purchase,
            'govt_release': govt_release,
            'govt_storage': self.government_grain_storage,
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
        # 检查是否启用内生价格机制
        market_config = self.config.get('environment', {}).get('market', {})
        use_endogenous = market_config.get('use_endogenous_pricing', True)
        
        # @fix: 使用上一期的实际产出作为本期的供给
        total_supply = 0
        farmer_supply = 0
        enterprise_supply = 0
        
        # 农户供给
        for farmer in model.farmers:
            if hasattr(farmer, 'last_crop_choice') and farmer.last_crop_choice and farmer.last_crop_choice.value == "Grain":
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
                print(f"[Step {self.current_step}] 最终粮价={final_price:.2f}元/斤 {'⚠️触及上限!' if final_price >= max_price else ''}\n")
        
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
        
        if np.random.random() < shock_probability:
            # 发生冲击
            shock_direction = np.random.choice([-1, 1], p=[0.7, 0.3])  # 70%负面冲击
            shock_factor = 1 + shock_direction * shock_magnitude * np.random.uniform(0.5, 1.0)
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
        ai_cfg = self.config.get('ai', {}).get('labor_market', {})
        demand_penalty = ai_cfg.get('non_farm_demand_penalty', 0.0)
        base_demand *= max(0.0, 1 - demand_penalty)
        
        # @critical 2025-12: 需求受经济周期影响
        base_demand *= market_condition_factor
        
        elasticity = self.config['environment']['market'].get('labor_demand_elasticity', -0.5)
        current_wage = self.market_prices['wage_rate']
        
        # 计算非农部门需求
        non_farm_labor_demand = base_demand * (current_wage ** elasticity)
        
        # 总劳动力需求
        total_labor_demand = agricultural_labor_demand + non_farm_labor_demand
        
        # ========== 4. 劳动力市场出清 ==========
        adjustment_factor = 0.05  # 调整速度
        
        if total_labor_demand > 0:
            relative_gap = (total_labor_supply - total_labor_demand) / total_labor_demand
            
            max_change_ratio = 0.15  # 单次最多调整15%
            wage_change = -relative_gap * adjustment_factor * current_wage
            wage_change = max(-max_change_ratio * current_wage, min(max_change_ratio * current_wage, wage_change))
            
            base_wage_floor = self.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 120.0)
            wage_floor_ratio = ai_cfg.get('wage_floor', 0.50)
            new_wage = max(base_wage_floor * wage_floor_ratio, current_wage + wage_change)
            
            # 应用工资粘性
            wage_stickiness = 0.75
            self.market_prices['wage_rate'] = wage_stickiness * current_wage + (1 - wage_stickiness) * new_wage
        
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
        return self.market_prices.get(item_name)

    def get_crop_price(self):
        """获取粮食价格（兼容旧API）"""
        return self.market_prices['crop_price']

    def get_base_non_farm_wage_rate(self):
        """获取基础非农工资率（兼容旧API）"""
        return self.market_prices['wage_rate']

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
        fiscal_config = self.config.get('fiscal_system', {})
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
        policy = policy_override or self.active_policy or {}
        if area <= 0 or not policy:
            return {'cash': 0.0, 'in_kind_value': 0.0, 'in_kind_breakdown': {}}

        base_subsidy_per_mu = float(policy.get('base_subsidy_per_mu', 0))
        subsidy_type = policy.get('subsidy_type', 'area_based')

        if subsidy_type == 'tiered_area':
            rate = self._get_tiered_rate(area, policy)
        else:
            rate = float(policy.get('grain_subsidies_rate', 0.0))

        # @fix: 应用主体调整系数（企业补贴比农户高15%）
        subject_adjustment = 1.0
        if actor_type == 'enterprise':
            subject_adjustment = 1.15  # 基于政策文件中的enterprise adjustment
        elif actor_type == 'cooperative':
            subject_adjustment = 1.1
        
        base_value = base_subsidy_per_mu * rate * area * subject_adjustment
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

        return {'cash': cash_value, 'in_kind_value': in_kind_value, 'in_kind_breakdown': breakdown}

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