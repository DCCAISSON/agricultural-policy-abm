# -*- coding: utf-8 -*-
"""
@theory 粮食储备局智能体
@critical 2025-12: 专家终极审阅 - 托市机制的另一半
@enhanced 2025-12: 专家审阅第二轮 - 强化财政约束与可持续性评估

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

参考文献：
- 《国家粮食安全保障规划(2024-2035)》
- Huang & Rozelle (2009), "Agricultural Development and Policy in China"
- 《中国粮食安全白皮书》(2023)
"""

from mesa import Agent
import numpy as np


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
        super().__init__(unique_id, model)  # @fix: Mesa Agent需要unique_id和model两个参数
        
        self.unique_id = unique_id
        self.config = config or {}
        
        # 储备局配置参数
        reserve_config = self.config.get('agents', {}).get('grain_reserve', {})
        
        # 年度收购预算（亿元）
        # 2023年中央储备粮收购资金约2000亿元
        self.base_annual_budget = reserve_config.get('annual_budget', 200000000000)  # 2000亿元
        self.annual_budget = self.base_annual_budget
        self.remaining_budget = self.annual_budget
        
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
        self.storage_cost_per_ton = reserve_config.get('storage_cost', 90)
        
        # 累计财政支出
        self.total_expenditure = 0
        self.annual_expenditure = 0
        
        # 收购/抛售记录
        self.purchase_history = []
        self.release_history = []
        
        # 年度统计
        self.annual_purchase_volume = 0
        self.annual_release_volume = 0
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
        # 重置年度统计
        self.annual_expenditure = 0
        self.annual_purchase_volume = 0
        self.annual_release_volume = 0
        
        # 0. 评估财政状态（在行动前）
        self._assess_fiscal_status()
        
        # 1. 获取市场价格
        current_price = self._get_market_price()
        
        # 2. 决策：托市收购 or 抛储稳价
        intervention_occurred = False
        if current_price < self.floor_price:
            self._execute_price_support_purchase(current_price)
            if self.annual_purchase_volume > 0:
                intervention_occurred = True
        elif current_price > self.ceiling_price:
            self._execute_stabilization_release(current_price)
        
        # 3. 更新连续干预年数
        self._update_intervention_tracking(intervention_occurred)
        
        # 4. 计算储存成本
        self._calculate_storage_cost()
        
        # 5. 更新累计支出
        self.total_expenditure += self.annual_expenditure
        
        # 6. 重置年度预算（考虑财政悬崖削减）
        self._reset_annual_budget()
    
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
    
    def _update_intervention_tracking(self, intervention_occurred):
        """
        @theory 更新连续干预跟踪
        
        跟踪连续托市年数，用于：
        1. 触发财政悬崖
        2. 评估政策可持续性
        3. 计算政策效果衰减
        """
        step = self.model.step_count if hasattr(self.model, 'step_count') else 0
        total_production = self._get_total_production()
        
        if intervention_occurred:
            self.consecutive_intervention_years += 1
            intervention_intensity = (self.annual_purchase_volume / total_production 
                                     if total_production > 0 else 0)
            intervention_type = 'purchase'
        else:
            # 没有干预，连续干预年数归零
            self.consecutive_intervention_years = 0
            self.fiscal_cliff_active = False  # 解除财政悬崖
            intervention_intensity = 0
            intervention_type = 'none'
        
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
        price_gap_ratio = (self.floor_price - current_price) / self.floor_price
        
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
        # 估算需要收购的量
        oversupply = self._estimate_oversupply(current_price)
        
        # 库存容量约束
        available_capacity = self.max_storage_capacity - self.storage
        
        # 预算约束
        budget_capacity = self.remaining_budget / self.floor_price
        
        # 实际收购量 = min(超额供给, 库存容量, 预算容量)
        purchase_qty = min(oversupply, available_capacity, budget_capacity)
        
        if purchase_qty > 0:
            # 执行收购
            purchase_cost = purchase_qty * self.floor_price
            
            self.storage += purchase_qty
            self.remaining_budget -= purchase_cost
            self.annual_expenditure += purchase_cost
            self.annual_purchase_volume = purchase_qty
            
            # 记录
            self.purchase_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'quantity': purchase_qty,
                'price': self.floor_price,
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
        # 保留安全库存
        available_release = self.storage - self.min_safety_stock
        
        if available_release <= 0:
            return
        
        # 估算需要释放多少才能稳定价格
        # 简化：价格每高于天花板1%，释放库存的0.5%
        price_gap_ratio = (current_price - self.ceiling_price) / self.ceiling_price
        release_ratio = min(0.20, price_gap_ratio * 0.5)  # 最高释放20%可用量
        
        release_qty = available_release * release_ratio
        
        if release_qty > 0:
            # 执行抛售（以市场价或略低于市场价）
            release_price = current_price * 0.98  # 略低于市场价促进销售
            revenue = release_qty * release_price
            
            self.storage -= release_qty
            self.annual_release_volume = release_qty
            
            # 记录
            self.release_history.append({
                'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
                'quantity': release_qty,
                'price': release_price,
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
                new_price = min(current_price * (1 + price_boost), self.floor_price * 1.1)
                
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
                new_price = max(current_price * (1 - price_drop), self.floor_price)
                
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
    
    def calculate_intervention_demand(self, market_price, total_supply):
        """
        @theory 计算干预需求量（供统一市场出清算法使用）
        
        这是储备局作为需求函数的一部分参与均衡价格计算
        
        返回：
        - intervention_type: 'purchase'（托市收购）, 'release'（抛储）, 'none'
        - intervention_qty: 干预数量（吨）
        - effective_budget: 本次可用预算
        
        统一市场出清算法会调用此方法，将干预量纳入供需计算
        """
        intervention_type = 'none'
        intervention_qty = 0
        effective_budget = self.remaining_budget
        
        # 应用财政悬崖削减后的预算约束
        if self.fiscal_cliff_active:
            years_over = max(0, self.consecutive_intervention_years - self.fiscal_cliff_threshold)
            budget_factor = max(0.5, 1 - self.fiscal_cliff_reduction_rate * years_over)
            effective_budget = self.remaining_budget * budget_factor
        
        if market_price < self.floor_price:
            # 托市收购：价格低于地板价
            intervention_type = 'purchase'
            
            # 估算超额供给
            price_gap = (self.floor_price - market_price) / self.floor_price
            oversupply_ratio = min(0.30, price_gap * 0.5)
            oversupply = total_supply * oversupply_ratio
            
            # 约束：库存容量、预算
            available_capacity = self.max_storage_capacity - self.storage
            budget_capacity = effective_budget / self.floor_price
            
            intervention_qty = min(oversupply, available_capacity, budget_capacity)
            
            # 应用政策效果衰减
            intervention_qty *= self.policy_effectiveness_score
            
        elif market_price > self.ceiling_price:
            # 抛储稳价：价格高于天花板
            intervention_type = 'release'
            
            available_release = max(0, self.storage - self.min_safety_stock)
            if available_release > 0:
                price_gap = (market_price - self.ceiling_price) / self.ceiling_price
                release_ratio = min(0.20, price_gap * 0.5)
                intervention_qty = available_release * release_ratio
        
        return {
            'type': intervention_type,
            'quantity': intervention_qty,
            'effective_budget': effective_budget,
            'fiscal_stress_level': self.fiscal_stress_level,
            'consecutive_years': self.consecutive_intervention_years,
            'policy_effectiveness': self.policy_effectiveness_score
        }
    
    def execute_intervention(self, intervention_type, quantity, price):
        """
        @theory 执行市场干预（由统一市场出清算法调用）
        
        在均衡价格确定后，执行实际的收购或抛储操作
        更新库存和财政支出
        """
        step = self.model.step_count if hasattr(self.model, 'step_count') else 0
        
        if intervention_type == 'purchase' and quantity > 0:
            cost = quantity * price
            self.storage += quantity
            self.remaining_budget -= cost
            self.annual_expenditure += cost
            self.annual_purchase_volume = quantity
            
            self.purchase_history.append({
                'step': step,
                'quantity': quantity,
                'price': price,
                'cost': cost
            })
            
            # 通知政府
            if hasattr(self.model, 'government') and hasattr(self.model.government, 'record_expenditure'):
                self.model.government.record_expenditure('grain_reserve_purchase', cost)
                
        elif intervention_type == 'release' and quantity > 0:
            # @fix 2025-11-30: 确保抛储量不超过实际库存
            available_release = max(0, self.storage - self.min_safety_stock)
            actual_release = min(quantity, available_release)
            
            if actual_release > 0:
                revenue = actual_release * price * 0.98  # 略低于市场价
                self.storage -= actual_release
                self.annual_release_volume = actual_release
                
                self.release_history.append({
                    'step': step,
                    'quantity': actual_release,
                    'price': price * 0.98,
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
