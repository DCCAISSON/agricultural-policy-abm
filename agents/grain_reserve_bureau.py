# -*- coding: utf-8 -*-
"""
@theory 粮食储备局智能体
@critical 2025-12: 专家终极审阅 - 托市机制的另一半
@enhanced 2025-12: 专家审阅第二轮 - 强化财政约束与可持续性评估
@fix 2026-01-02: 单位统一修复 - 所有接口统一使用kg，内部存储使用吨

理论依据：
1. 中国粮食最低收购价政策自2004年实施
2. 2023年托市收购约4000万吨稻谷，占总产量20%
3. 托市收购是价格地板机制的关键执行手段
4. 没有收储量，模型会低估财政压力，高估"提补贴就能稳粮价"的效果

实现逻辑：
1. 当市场价格低于托市价格时，启动收购
2. 收购量 = min(超额供给量, 预算/托市价)
3. 记录财政支出，影响政府预算约束
4. 储备粮可在价格过高时抛售稳定市场

新增强化功能（专家第二轮审阅）：
1. 财政悬崖机制：连续多年托市导致预算削减
2. 连续干预跟踪：评估政策可持续性
3. 与统一市场出清算法联动：提供干预需求函数

单位约定（P0级约束）：
- 外部接口（calculate_intervention_demand, execute_intervention）：全部使用 **kg**
- 内部存储（self.storage）：使用 **吨**，与现实报告一致
- 转换常量：KG_PER_TON = 1000

参考文献：
- 《国家粮食安全保障规划(2024-2035)》
- Huang & Rozelle (2009), "Agricultural Development and Policy in China"
- 《中国粮食安全白皮书》(2023)
"""

from mesa import Agent
import numpy as np

# @fix 2026-01-02: 单位转换常量
KG_PER_TON = 1000
TON_PER_KG = 0.001
JIN_PER_KG = 2.0


class GrainReserveBureau(Agent):
    """
    粮食储备局智能体
    
    @theory 托市收购 + 储备调节 + 财政可持续性
    
    主要功能：
    1. 最低收购价收购（托市）
    2. 粮食储备管理
    3. 抛储稳定市场
    4. 财政支出核算
    5. [新增] 财政悬崖预警与应对
    6. [新增] 连续干预年数跟踪
    
    政策参数（参考2023年数据）：
    - 稻谷最低收购价：1.29-1.31元/斤（中晚籼稻）
    - 小麦最低收购价：1.18元/斤
    - 托市收购量：约4000万吨/年（稻谷）
    """
    
    def __init__(self, unique_id, model, config=None):
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
        
        self.unique_id = unique_id
        self.config = config or {}
        
        # 储备局配置参数
        agents_config = self.config.get('agents', {})
        reserve_config = (
            agents_config.get('grain_reserve_bureau')
            or agents_config.get('grain_reserve')
            or {}
        )
        
        # 年度收购预算（亿元）
        # 2023年中央储备粮收购资金约2000亿元
        self.base_annual_budget = reserve_config.get('annual_budget', 200000000000)  # 2000亿元
        self.annual_budget = self.base_annual_budget
        self.remaining_budget = self.annual_budget
        self.base_annual_budget = float(self.base_annual_budget or 0.0)
        self.annual_budget = float(self.annual_budget or 0.0)
        self.remaining_budget = float(self.remaining_budget or 0.0)
        if self._policy_disables_grain_intervention():
            self.annual_budget = 0.0
            self.remaining_budget = 0.0
        
        # 托市价格（元/公斤）
        # 2023年中晚籼稻：1.29元/斤 = 2.58元/公斤
        self.floor_price = reserve_config.get('floor_price', 2.58)
        
        # 抛储价格（元/公斤）- 市场价高于此价时抛储
        # 一般高于托市价10-15%
        self.ceiling_price = reserve_config.get('ceiling_price', 2.90)
        
        # 储备粮库存（吨）
        # 全国政策性粮食库存约3亿吨
        self.storage = reserve_config.get('initial_storage', 100000)  # 10万吨初始
        
        # 最大库存容量（吨）
        self.max_storage_capacity = reserve_config.get('max_storage_capacity', 500000)  # 50万吨
        
        # 最低安全库存（吨）
        self.min_safety_stock = reserve_config.get('min_safety_stock', 20000)  # 2万吨
        
        # 储存成本（元/吨/年）
        # 包括仓储、保管、损耗等，约80-100元/吨/年
        self.price_unit = self._infer_price_unit(reserve_config)
        self.floor_price = self._convert_price_to_jin(self.floor_price, self.price_unit)
        self.floor_price_per_kg = self._convert_price_to_kg(self.floor_price, 'yuan_per_jin')
        self.ceiling_price = self._convert_price_to_jin(self.ceiling_price, self.price_unit)
        self.ceiling_price_per_kg = self._convert_price_to_kg(self.ceiling_price, 'yuan_per_jin')
        if self.ceiling_price < self.floor_price:
            self.ceiling_price = self.floor_price
        if self.ceiling_price_per_kg < self.floor_price_per_kg:
            self.ceiling_price_per_kg = self.floor_price_per_kg

        self.storage_unit = self._infer_storage_unit(reserve_config)
        self.storage = self._convert_storage_to_ton(self.storage, self.storage_unit)
        self.max_storage_capacity = self._convert_storage_to_ton(
            self.max_storage_capacity, self.storage_unit
        )
        self.min_safety_stock = self._convert_storage_to_ton(
            self.min_safety_stock, self.storage_unit
        )

        self.storage_cost_per_ton = reserve_config.get('storage_cost', 90)
        self.storage_cost_per_ton = float(self.storage_cost_per_ton or 0.0)
        
        # 累计财政支出
        self.total_expenditure = 0
        self.annual_expenditure = 0
        
        # 收购/抛售记录
        self.purchase_history = []
        self.release_history = []
        
        # 年度统计
        self.annual_purchase_volume = 0.0
        self.annual_release_volume = 0.0
        self.annual_storage_cost = 0
        
        # ========== 新增：连续干预跟踪与财政可持续性 ==========
        # 连续托市收购年数
        self.consecutive_intervention_years = 0
        
        # 财政悬崖参数
        # 连续托市超过阈值年数后，预算开始削减
        self.fiscal_cliff_threshold = reserve_config.get('fiscal_cliff_threshold', 3)  # 连续3年
        self.fiscal_cliff_reduction_rate = reserve_config.get('fiscal_cliff_reduction_rate', 0.15)  # 每年削减15%
        
        # 财政压力状态
        self.fiscal_stress_level = 0  # 0=正常, 1=警戒, 2=紧张, 3=危机
        self.fiscal_cliff_active = False  # 是否已触发财政悬崖
        
        # 历年干预记录（用于趋势分析）
        self.intervention_history = []  # [(year, type, intensity), ...]
        
        # 政策效果评估
        self.policy_effectiveness_score = 1.0  # 1.0=完全有效，递减表示效果衰减
        self._annual_intervention_type = 'none'

    def _policy_disables_grain_intervention(self):
        env = getattr(self.model, 'environment', None)
        active_policy = getattr(env, 'active_policy', {}) if env is not None else {}
        if bool((active_policy or {}).get('disable_grain_intervention', False)):
            return True

        scenario_id = (
            getattr(env, 'active_policy_id', None)
            or getattr(self.model, 'active_policy_id', None)
            or ''
        )
        return str(scenario_id).strip().lower() in {'no_policy', 'none', 'market_only'}

    def _infer_price_unit(self, reserve_config):
        explicit_unit = str(reserve_config.get('price_unit', '') or '').strip().lower()
        if explicit_unit in {'yuan_per_kg', '元/kg', 'kg'}:
            return 'yuan_per_kg'
        if explicit_unit in {'yuan_per_jin', '元/斤', 'jin'}:
            return 'yuan_per_jin'

        numeric_prices = []
        for key in ('floor_price', 'ceiling_price'):
            try:
                numeric_prices.append(float(reserve_config.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        if numeric_prices and max(numeric_prices) > 2.2:
            return 'yuan_per_kg'
        return 'yuan_per_jin'

    def _convert_price_to_jin(self, price_value, price_unit):
        price_value = float(price_value or 0.0)
        if price_unit == 'yuan_per_kg':
            return price_value / JIN_PER_KG
        return price_value

    def _convert_price_to_kg(self, price_value, price_unit):
        price_value = float(price_value or 0.0)
        if price_unit == 'yuan_per_kg':
            return price_value
        return price_value * JIN_PER_KG

    def _infer_storage_unit(self, reserve_config):
        explicit_unit = str(reserve_config.get('storage_unit', '') or '').strip().lower()
        if explicit_unit in {'kg', 'kilogram', 'kilograms', '公斤'}:
            return 'kg'
        if explicit_unit in {'ton', 'tons', '吨', 't'}:
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

    def _convert_storage_to_ton(self, storage_value, storage_unit):
        storage_value = float(storage_value or 0.0)
        if storage_unit == 'kg':
            return storage_value * TON_PER_KG
        return storage_value

    def get_storage_kg(self):
        return max(0.0, self.storage * KG_PER_TON)

    def get_available_release_kg(self):
        return max(0.0, (self.storage - self.min_safety_stock) * KG_PER_TON)

    def _get_active_price_band(self):
        floor_price = float(self.floor_price)
        ceiling_price = float(self.ceiling_price)

        if hasattr(self.model, 'environment') and self.model.environment is not None:
            active_policy = getattr(self.model.environment, 'active_policy', {}) or {}
            if bool(active_policy.get('disable_grain_intervention', False)):
                return 0.0, float('inf')
            market_cfg = self.model.environment.config.get('environment', {}).get('market', {})
            grain_limits = market_cfg.get('price_limits', {}).get('grain', {})
            price_min = float(grain_limits.get('min', floor_price) or floor_price)
            price_max = float(grain_limits.get('max', ceiling_price) or ceiling_price)
            ceiling_multiplier = float(market_cfg.get('grain_ceiling_multiplier', 1.3) or 1.3)

            try:
                floor_price = float(active_policy.get('grain_price_floor', floor_price) or floor_price)
            except (TypeError, ValueError):
                floor_price = float(self.floor_price)
            floor_price = max(price_min, min(price_max, floor_price))

            default_ceiling = max(ceiling_price, floor_price * ceiling_multiplier)
            ceiling_price = max(floor_price, min(price_max, default_ceiling))

        return floor_price, ceiling_price
        
    def step(self):
        """
        储备局年度行动
        
        执行顺序：
        1. 评估财政状态（新增）
        2. 评估市场供需状况
        3. 决定收购/抛售策略
        4. 计算储存成本
        5. 更新财务状况
        6. 更新连续干预记录（新增）
        """
        self.begin_annual_cycle()

        # 兼容旧式直接调度：若独立执行，则按当期价格直接完成一次干预。
        current_price = self._get_market_price()
        floor_price, ceiling_price = self._get_active_price_band()
        if current_price < floor_price:
            self._execute_price_support_purchase(current_price)
        elif current_price > ceiling_price:
            self._execute_stabilization_release(current_price)

        self.finalize_annual_cycle()

    def _mark_intervention(self, intervention_type):
        """Track the realized intervention mode for end-of-year auditing."""
        if intervention_type not in {'purchase', 'release'}:
            return
        if self._annual_intervention_type == 'none':
            self._annual_intervention_type = intervention_type
        elif self._annual_intervention_type != intervention_type:
            self._annual_intervention_type = 'mixed'

    def export_sar(self) -> dict:
        """Export (state, action) for SAR logging.

        Grain reserve bureau is currently rule-based (policy_mode='rule').
        Reward is captured by SAR logger from reserve stability / fiscal cost.
        """
        return {
            'state': {
                'current_storage_ton': float(getattr(self, 'current_storage', 0.0) or 0.0),
                'storage_ceiling_ton': float(getattr(self, 'storage_ceiling', 5000.0) or 5000.0),
                'reserve_ratio': float(getattr(self, '_reserve_ratio', 0.0) or 0.0),
                'fiscal_cliff_active': bool(getattr(self, '_fiscal_cliff_active', False)),
            },
            'action': {
                'purchase_volume_ton': float(getattr(self, 'annual_purchase_volume', 0.0) or 0.0),
                'release_volume_ton': float(getattr(self, 'annual_release_volume', 0.0) or 0.0),
                'intervention_type': str(getattr(self, '_annual_intervention_type', 'none')),
                'price_floor': float(getattr(self, 'current_grain_price_floor', 0.0) or 0.0),
            },
            'policy_mode': 'rule',
            'agent_type': 'grain_reserve_bureau',
        }

    def begin_annual_cycle(self):
        """Reset annual reserve-bureau bookkeeping before grain market clearing."""
        self.annual_expenditure = 0
        self.annual_purchase_volume = 0.0
        self.annual_release_volume = 0.0
        self.annual_storage_cost = 0
        self._annual_intervention_type = 'none'

    def finalize_annual_cycle(self):
        """Finalize annual reserve-bureau accounting after market intervention."""
        self._enforce_storage_ceiling()
        self._calculate_storage_cost()
        self.total_expenditure += self.annual_expenditure
        self._update_intervention_tracking(
            self.annual_purchase_volume > 0,
            self._annual_intervention_type
        )
        self._assess_fiscal_status()
        self._reset_annual_budget()
    
    def _enforce_storage_ceiling(self):
        """
        @fix 2025-12-22: 强制执行库存上限，防止30年模拟库存爆炸
        
        当库存超过最大容量时强制抛售：
        - 超出部分以折价方式抛售
        - 记录强制抛售损失
        - 反映现实中的仓储压力
        """
        if self.storage > self.max_storage_capacity:
            excess = self.storage - self.max_storage_capacity
            excess_kg = excess * KG_PER_TON
            
            # 强制抛售超出部分（折价20%）
            floor_price, _ = self._get_active_price_band()
            forced_sale_price = floor_price * 0.8
            revenue = excess_kg * forced_sale_price * JIN_PER_KG
            
            # 更新库存和记录
            self.storage = self.max_storage_capacity
            self.annual_release_volume += excess_kg
            self._mark_intervention('release')
            
            self.release_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'type': 'forced_ceiling',
                'quantity_ton': excess,
                'quantity_kg': excess_kg,
                'price_jin': forced_sale_price,
                'price_kg': forced_sale_price * JIN_PER_KG,
                'revenue': revenue,
                'reason': 'storage_ceiling_breach'
            })
            
            print(f"  [储备局] 库存超限强制抛售: {excess:.0f}吨，折价20%")
    
    def _assess_fiscal_status(self):
        """
        @theory 评估财政状态
        
        财政悬崖机制：
        - 连续托市超过阈值年数后，财政压力累积
        - 预算开始被动削减，反映地方财政承压
        - 这会导致托市能力下降，价格支撑减弱
        
        现实依据：
        - 2018-2022年，部分地区托市资金拨付延迟
        - 地方配套资金不足导致收购能力下降
        """
        # 更新财政压力等级
        fiscal_pressure = self.get_fiscal_pressure_index()
        
        if fiscal_pressure < 0.5:
            self.fiscal_stress_level = 0  # 正常
        elif fiscal_pressure < 0.8:
            self.fiscal_stress_level = 1  # 警戒
        elif fiscal_pressure < 1.0:
            self.fiscal_stress_level = 2  # 紧张
        else:
            self.fiscal_stress_level = 3  # 危机
        
        # 检查是否触发财政悬崖
        if self.consecutive_intervention_years >= self.fiscal_cliff_threshold:
            if not self.fiscal_cliff_active:
                self.fiscal_cliff_active = True
                # 记录财政悬崖触发事件
                step = self.model.step_count if hasattr(self.model, 'step_count') else 0
                print(f"  [!] 财政悬崖触发: 连续{self.consecutive_intervention_years}年托市干预")
    
    def _update_intervention_tracking(self, purchase_occurred, intervention_type='none'):
        """
        @theory 更新连续干预跟踪
        
        跟踪连续托市年数，用于：
        1. 触发财政悬崖
        2. 评估政策可持续性
        3. 计算政策效果衰减
        """
        step = self.model.step_count if hasattr(self.model, 'step_count') else 0
        total_production = self._get_total_production()
        
        if purchase_occurred:
            self.consecutive_intervention_years += 1
            if intervention_type == 'none':
                intervention_type = 'purchase'
        else:
            # 没有干预，连续干预年数归零
            self.consecutive_intervention_years = 0
            self.fiscal_cliff_active = False  # 解除财政悬崖

        if total_production > 0:
            if intervention_type == 'purchase':
                intervention_intensity = self.annual_purchase_volume / total_production
            elif intervention_type == 'release':
                intervention_intensity = self.annual_release_volume / total_production
            elif intervention_type == 'mixed':
                intervention_intensity = (
                    (self.annual_purchase_volume + self.annual_release_volume) / total_production
                )
            else:
                intervention_intensity = 0
        else:
            intervention_intensity = 0
        
        # 记录历史
        self.intervention_history.append({
            'step': step,
            'type': intervention_type,
            'intensity': intervention_intensity,
            'consecutive_years': self.consecutive_intervention_years,
            'fiscal_stress': self.fiscal_stress_level
        })
        
        # 更新政策效果评分（连续干预会导致效果衰减）
        if self.consecutive_intervention_years > 2:
            # 连续干预超过2年后，每年效果下降10%
            decay = 0.1 * (self.consecutive_intervention_years - 2)
            self.policy_effectiveness_score = max(0.5, 1.0 - decay)
        
    def _get_market_price(self):
        """获取当前粮食市场价格"""
        if hasattr(self.model, 'environment'):
            return self.model.environment.get_market_price('crop_price')
        return 2.5  # 默认价格
    
    def _get_total_production(self):
        """估算当年总产量"""
        if hasattr(self.model, 'farmers'):
            total = 0
            for farmer in self.model.farmers:
                if hasattr(farmer, 'expected_yield') and farmer.expected_yield > 0:
                    # 假设60%是粮食
                    total += farmer.expected_yield * 0.6
            return total
        return 100000  # 默认10万吨
    
    def _estimate_oversupply(self, current_price):
        """
        @theory 估算超额供给量
        
        超额供给 = 当价格低于托市价时，农户愿意卖但市场消化不了的量
        简化估算：基于价格弹性
        """
        total_production = self._get_total_production()
        
        # 价格弹性法估算
        # 当价格低于托市价时，需求不足的比例
        floor_price, _ = self._get_active_price_band()
        price_gap_ratio = (floor_price - current_price) / max(floor_price, 1e-6)
        
        # 超额供给率（经验值：价格每低1%，超额供给约0.5-1%）
        oversupply_ratio = min(0.30, price_gap_ratio * 0.5)  # 最高30%
        
        return total_production * oversupply_ratio
    
    def _execute_price_support_purchase(self, current_price):
        """
        @theory 托市收购
        
        当市场价格低于最低收购价时：
        1. 计算超额供给量
        2. 根据预算决定收购量
        3. 执行收购，增加库存
        4. 记录财政支出
        
        这是托市机制的核心：
        政府作为"最后买家"，吸收过剩供给，维护价格地板
        """
        floor_price, _ = self._get_active_price_band()
        floor_price_per_kg = floor_price * JIN_PER_KG

        # 估算需要收购的量
        oversupply = self._estimate_oversupply(current_price)
        
        # 库存容量约束
        available_capacity = (self.max_storage_capacity - self.storage) * KG_PER_TON
        
        # 预算约束
        budget_capacity = self.remaining_budget / max(floor_price_per_kg, 1e-6)
        
        # 实际收购量 = min(超额供给, 库存容量, 预算容量)
        purchase_qty = min(oversupply, available_capacity, budget_capacity)
        
        if purchase_qty > 0:
            # 执行收购
            purchase_cost = purchase_qty * floor_price_per_kg
            
            self.storage += purchase_qty * TON_PER_KG
            self.remaining_budget -= purchase_cost
            self.annual_expenditure += purchase_cost
            self.annual_purchase_volume += purchase_qty
            self._mark_intervention('purchase')
            
            # 记录
            self.purchase_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'quantity_ton': purchase_qty * TON_PER_KG,
                'quantity_kg': purchase_qty,
                'price_jin': floor_price,
                'price_kg': floor_price_per_kg,
                'cost': purchase_cost,
                'market_price': current_price
            })
            
            # 通知政府财政支出
            if hasattr(self.model, 'government'):
                if hasattr(self.model.government, 'record_expenditure'):
                    self.model.government.record_expenditure('grain_reserve_purchase', purchase_cost)
            
            # 托市收购应该影响市场价格（推高价格）
            self._affect_market_price_after_purchase(purchase_qty)
    
    def _execute_stabilization_release(self, current_price):
        """
        @theory 抛储稳价
        
        当市场价格过高时：
        1. 释放储备粮
        2. 增加市场供给
        3. 平抑价格
        
        这是粮食安全的另一面：
        防止价格过高损害消费者利益和社会稳定
        """
        _, ceiling_price = self._get_active_price_band()

        # 保留安全库存
        available_release = self.get_available_release_kg()
        
        if available_release <= 0:
            return
        
        # 估算需要释放多少才能稳定价格
        # 简化：价格每高于天花板1%，释放库存的0.5%
        price_gap_ratio = (current_price - ceiling_price) / max(ceiling_price, 1e-6)
        release_ratio = min(0.20, price_gap_ratio * 0.5)  # 最高释放20%可用量
        
        release_qty = available_release * release_ratio
        
        if release_qty > 0:
            # 执行抛售（以市场价或略低于市场价）
            release_price = current_price * 0.98  # 略低于市场价促进销售
            revenue = release_qty * release_price * JIN_PER_KG
            
            self.storage -= release_qty * TON_PER_KG
            self.annual_release_volume += release_qty
            self._mark_intervention('release')
            
            # 记录
            self.release_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'quantity_ton': release_qty * TON_PER_KG,
                'quantity_kg': release_qty,
                'price_jin': release_price,
                'price_kg': release_price * JIN_PER_KG,
                'revenue': revenue,
                'market_price': current_price
            })
            
            # 抛储应该压低市场价格
            self._affect_market_price_after_release(release_qty)
    
    def _affect_market_price_after_purchase(self, purchase_qty):
        """
        托市收购后对市场价格的影响
        大量收购会推高价格
        """
        if hasattr(self.model, 'environment'):
            total_production = self._get_total_production()
            if total_production > 0:
                # 收购量占产量的比例越高，价格上涨越多
                purchase_ratio = purchase_qty / total_production
                price_boost = purchase_ratio * 0.5  # 每收购10%产量，价格上涨5%
                
                # 通知环境调整价格（简化处理）
                current_price = self.model.environment.get_market_price('crop_price')
                floor_price, _ = self._get_active_price_band()
                new_price = min(current_price * (1 + price_boost), floor_price * 1.1)
                
                # 记录价格干预
                if hasattr(self.model.environment, 'policy_intervention_history'):
                    self.model.environment.policy_intervention_history.append({
                        'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                        'type': 'grain_reserve_purchase',
                        'quantity': purchase_qty,
                        'price_effect': price_boost
                    })
    
    def _affect_market_price_after_release(self, release_qty):
        """
        抛储后对市场价格的影响
        大量抛售会压低价格
        """
        if hasattr(self.model, 'environment'):
            total_production = self._get_total_production()
            if total_production > 0:
                release_ratio = release_qty / total_production
                price_drop = release_ratio * 0.5
                
                current_price = self.model.environment.get_market_price('crop_price')
                floor_price, _ = self._get_active_price_band()
                new_price = max(current_price * (1 - price_drop), floor_price)
                
                if hasattr(self.model.environment, 'policy_intervention_history'):
                    self.model.environment.policy_intervention_history.append({
                        'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                        'type': 'grain_reserve_release',
                        'quantity': release_qty,
                        'price_effect': -price_drop
                    })
    
    def _calculate_storage_cost(self):
        """计算年度储存成本"""
        self.annual_storage_cost = self.storage * self.storage_cost_per_ton
        self.annual_expenditure += self.annual_storage_cost
    
    def _reset_annual_budget(self):
        """
        @theory 重置年度预算（含财政悬崖调整）
        
        财政悬崖机制：
        - 连续托市超过阈值后，每年预算削减fiscal_cliff_reduction_rate
        - 反映财政可持续性约束
        - 最低保留基础预算的50%
        
        这是专家强调的"财政压力传导机制"的核心
        """
        if self._policy_disables_grain_intervention():
            self.annual_budget = 0.0
            self.remaining_budget = 0.0
            return

        if self.fiscal_cliff_active:
            # 计算削减后的预算
            years_over_threshold = self.consecutive_intervention_years - self.fiscal_cliff_threshold
            reduction = 1 - (self.fiscal_cliff_reduction_rate * years_over_threshold)
            reduction = max(0.5, reduction)  # 最低保留50%
            
            self.annual_budget = self.base_annual_budget * reduction
        else:
            self.annual_budget = self.base_annual_budget
        
        self.remaining_budget = self.annual_budget
    
    def get_statistics(self):
        """获取储备局统计信息"""
        return {
            'storage': self.storage,
            'storage_kg': self.get_storage_kg(),
            'storage_ratio': self.storage / self.max_storage_capacity if self.max_storage_capacity > 0 else 0,
            'annual_purchase': self.annual_purchase_volume,
            'annual_release': self.annual_release_volume,
            'annual_expenditure': self.annual_expenditure,
            'total_expenditure': self.total_expenditure,
            'annual_storage_cost': self.annual_storage_cost,
            'remaining_budget': self.remaining_budget
        }
    
    def get_fiscal_pressure_index(self):
        """
        @theory 财政压力指数
        
        用于评估托市政策的财政可持续性
        指数 = 年度支出 / 年度预算
        
        当指数 > 1时，表示财政压力过大
        这是专家强调的"精确预测财政压力"的关键指标
        """
        if self.annual_budget > 0:
            return self.annual_expenditure / self.annual_budget
        return 0
    
    def get_market_intervention_intensity(self):
        """
        @theory 市场干预强度
        
        收购量 / 总产量
        反映政府对粮食市场的干预程度
        
        2023年数据：约4000万吨/2亿吨 ≈ 20%
        """
        total_production = self._get_total_production()
        if total_production > 0:
            return self.annual_purchase_volume / total_production
        return 0
    
    # ========== 新增：统一市场出清算法联动接口 ==========
    
    def calculate_intervention_demand(self, market_price_jin, total_supply_kg):
        """
        @theory 计算干预需求量（供统一市场出清算法使用）
        @fix 2026-01-02: 统一单位为kg，与environment.py接口一致
        
        这是储备局作为需求函数的一部分参与均衡价格计算
        
        Args:
            market_price: 市场价格（元/kg）
            total_supply_kg: 总供给量（kg）- 注意是kg！
        
        Returns:
            dict containing:
            - intervention_type: 'purchase'（托市收购）, 'release'（抛储）, 'none'
            - quantity: 干预数量（**kg**）- 统一用kg与environment交互
            - effective_budget: 本次可用预算（元）
        
        统一市场出清算法会调用此方法，将干预量纳入供需计算
        """
        intervention_type = 'none'
        intervention_qty_kg = 0  # @fix: 明确标注单位
        effective_budget = self.remaining_budget
        
        # 应用财政悬崖削减后的预算约束
        if self.fiscal_cliff_active:
            years_over = max(0, self.consecutive_intervention_years - self.fiscal_cliff_threshold)
            budget_factor = max(0.5, 1 - self.fiscal_cliff_reduction_rate * years_over)
            effective_budget = self.remaining_budget * budget_factor
        
        floor_price, ceiling_price = self._get_active_price_band()
        if not np.isfinite(ceiling_price):
            return {
                'type': intervention_type,
                'quantity': intervention_qty_kg,
                'effective_budget': effective_budget,
                'fiscal_stress_level': self.fiscal_stress_level,
                'consecutive_years': self.consecutive_intervention_years,
                'policy_effectiveness': self.policy_effectiveness_score
            }
        floor_price_per_kg = floor_price * JIN_PER_KG

        if market_price_jin < floor_price:
            # 托市收购：价格低于地板价
            intervention_type = 'purchase'
            
            # 估算超额供给（输入已经是kg）
            price_gap = (floor_price - market_price_jin) / max(floor_price, 1e-6)
            oversupply_ratio = min(0.30, price_gap * 0.5)
            oversupply_kg = total_supply_kg * oversupply_ratio
            
            # 约束：库存容量（吨→kg）、预算（元/kg）
            available_capacity_kg = (self.max_storage_capacity - self.storage) * KG_PER_TON
            budget_capacity_kg = effective_budget / max(floor_price_per_kg, 1e-6)
            
            intervention_qty_kg = min(oversupply_kg, available_capacity_kg, budget_capacity_kg)
            
            # 应用政策效果衰减
            intervention_qty_kg *= self.policy_effectiveness_score
            
        elif market_price_jin > ceiling_price:
            # 抛储稳价：价格高于天花板
            intervention_type = 'release'
            
            available_release_kg = self.get_available_release_kg()
            if available_release_kg > 0:
                price_gap = (market_price_jin - ceiling_price) / max(ceiling_price, 1e-6)
                release_ratio = min(0.20, price_gap * 0.5)
                intervention_qty_kg = available_release_kg * release_ratio
        
        return {
            'type': intervention_type,
            'quantity': intervention_qty_kg,  # @fix: 现在是kg
            'effective_budget': effective_budget,
            'fiscal_stress_level': self.fiscal_stress_level,
            'consecutive_years': self.consecutive_intervention_years,
            'policy_effectiveness': self.policy_effectiveness_score
        }
    
    def execute_intervention(self, intervention_type, quantity_kg, price_jin):
        """
        @theory 执行市场干预（由统一市场出清算法调用）
        @fix 2026-01-02: 输入quantity_kg现在是kg单位，内部转换为吨存储
        
        在均衡价格确定后，执行实际的收购或抛储操作
        更新库存和财政支出
        
        Args:
            intervention_type: 'purchase' 或 'release'
            quantity_kg: 干预数量（**kg**）- 与environment接口一致
            price: 价格（元/kg）
        """
        step = self.model.step_count if hasattr(self.model, 'step_count') else 0
        
        price_per_kg = float(price_jin or 0.0) * JIN_PER_KG
        # @fix: 转换为吨用于内部存储
        quantity_ton = quantity_kg * TON_PER_KG
        
        if intervention_type == 'purchase' and quantity_kg > 0:
            available_capacity_kg = max(0.0, (self.max_storage_capacity - self.storage) * KG_PER_TON)
            actual_purchase_kg = min(float(quantity_kg or 0.0), available_capacity_kg)
            if actual_purchase_kg <= 0:
                return
            quantity_ton = actual_purchase_kg * TON_PER_KG
            cost = actual_purchase_kg * price_per_kg
            self.storage += quantity_ton  # 内部用吨存储
            self.remaining_budget -= cost
            self.annual_expenditure += cost
            self.annual_purchase_volume += actual_purchase_kg
            self._mark_intervention('purchase')
            
            self.purchase_history.append({
                'step': step,
                'quantity_ton': quantity_ton,
                'quantity_kg': actual_purchase_kg,
                'price_jin': price_jin,
                'price_kg': price_per_kg,
                'cost': cost
            })
            
            # 通知政府
            if hasattr(self.model, 'government') and hasattr(self.model.government, 'record_expenditure'):
                self.model.government.record_expenditure('grain_reserve_purchase', cost)
                
        elif intervention_type == 'release' and quantity_kg > 0:
            # @fix 2025-11-30: 确保抛储量不超过实际库存
            # 内部storage是吨，转换为kg比较
            available_release_kg = self.get_available_release_kg()
            actual_release_kg = min(quantity_kg, available_release_kg)
            actual_release_ton = actual_release_kg * TON_PER_KG
            
            if actual_release_kg > 0:
                release_price_jin = price_jin * 0.98
                release_price_kg = release_price_jin * JIN_PER_KG
                revenue = actual_release_kg * release_price_kg
                self.storage -= actual_release_ton
                self.annual_release_volume += actual_release_kg
                self._mark_intervention('release')
                
                self.release_history.append({
                    'step': step,
                    'quantity_ton': actual_release_ton,
                    'quantity_kg': actual_release_kg,
                    'price_jin': release_price_jin,
                    'price_kg': release_price_kg,
                    'revenue': revenue
                })
    
    def get_sustainability_assessment(self):
        """
        @theory 政策可持续性评估
        
        返回综合指标，用于评估托市政策的长期可行性：
        1. 财政压力指数
        2. 库存压力指数
        3. 连续干预年数
        4. 政策效果评分
        
        这是回答"财政能支撑多久"的关键指标
        """
        fiscal_pressure = self.get_fiscal_pressure_index()
        storage_pressure = self.storage / self.max_storage_capacity if self.max_storage_capacity > 0 else 0
        
        # 综合可持续性得分（0-1，越高越可持续）
        sustainability_score = 1.0
        
        # 财政压力惩罚
        if fiscal_pressure > 0.5:
            sustainability_score -= (fiscal_pressure - 0.5) * 0.4
        
        # 库存压力惩罚
        if storage_pressure > 0.7:
            sustainability_score -= (storage_pressure - 0.7) * 0.3
        
        # 连续干预惩罚
        if self.consecutive_intervention_years > 2:
            sustainability_score -= 0.1 * (self.consecutive_intervention_years - 2)
        
        # 政策效果衰减
        sustainability_score *= self.policy_effectiveness_score
        
        return {
            'sustainability_score': max(0, min(1, sustainability_score)),
            'fiscal_pressure': fiscal_pressure,
            'storage_pressure': storage_pressure,
            'consecutive_intervention_years': self.consecutive_intervention_years,
            'policy_effectiveness': self.policy_effectiveness_score,
            'fiscal_cliff_active': self.fiscal_cliff_active,
            'fiscal_stress_level': self.fiscal_stress_level,
            'warning_level': 'critical' if sustainability_score < 0.3 else 
                            'warning' if sustainability_score < 0.6 else 'normal'
        }
