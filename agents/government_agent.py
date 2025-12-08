# -*- coding: utf-8 -*-
"""
@theory
GovernmentAgent - 基于多层次博弈设计规范的政府主体
核心功能：政策发布和补贴发放
新增：多层次博弈政策设定、价格地板机制
"""

import json
from pathlib import Path

from mesa import Agent
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class SubsidyType(Enum):
    """补贴类型枚举"""
    AREA_BASED = "area_based"      # 按面积补贴
    OUTPUT_BASED = "output_based"  # 按产量补贴

class SubsidyTarget(Enum):
    """补贴目标枚举"""
    OWNER = "owner"        # 补贴给所有者
    OPERATOR = "operator"  # 补贴给耕作者

class GovernmentAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # 政策参数（新增）
        self.grain_price_floor = 1.2                    # 粮食最低收购价
        self.grain_subsidies_rate = 0.15                # 粮食补贴率
        self.market_price_cash_crop = 2.0               # 经济作物市场价格
        self.subsidy_rate = 0.15                        # 一般补贴率
        
        # 补贴政策
        self.subsidy_type = SubsidyType.AREA_BASED      # 补贴类型
        self.subsidy_target = SubsidyTarget.OWNER       # 补贴目标
        
        # 补贴发放记录
        self.total_subsidy_paid = 0                     # 总补贴支出
        self.subsidy_records = []                       # 补贴记录
        
        # ========== @new 2025-11-30: 财政约束系统 ==========
        # 核心理念：补贴预算是外生给定的，来自中央财政转移支付
        # 模型内部只能在预算约束下分配，不能自行创造
        fiscal_cfg = self.model.config.get('agents', {}).get('fiscal_system', {})
        
        # 年度补贴预算（元）- 外生给定
        self.annual_subsidy_budget = fiscal_cfg.get('annual_subsidy_budget', 2000000)  # 200万
        self.remaining_budget = self.annual_subsidy_budget
        
        # 预算约束参数
        budget_constraints = fiscal_cfg.get('budget_constraints', {})
        self.hard_cap = budget_constraints.get('hard_cap', True)  # 硬性约束
        self.overspending_threshold = budget_constraints.get('overspending_threshold', 1.2)
        self.consecutive_overspending_years = budget_constraints.get('consecutive_overspending_years', 3)
        self.fiscal_reduction_rate = budget_constraints.get('reduction_rate', 0.15)
        self.minimum_budget_ratio = budget_constraints.get('minimum_budget_ratio', 0.5)
        
        # 财政状态跟踪
        self.consecutive_overspending_count = 0
        self.fiscal_cliff_active = False
        self.budget_utilization_history = []  # 历年预算使用率
        
        # ========== @new 2025-12-04: 补贴拖欠机制 ==========
        # 核心理念：财政压力大时，政府不是立即削减补贴标准，而是产生拖欠
        # 拖欠会在后续年度逐步清偿，影响农户当期可支配收入
        arrears_cfg = fiscal_cfg.get('arrears_mechanism', {})
        self.enable_arrears = arrears_cfg.get('enable', True)  # 启用拖欠机制
        self.max_arrears_ratio = arrears_cfg.get('max_arrears_ratio', 0.3)  # 最大拖欠比例
        self.arrears_clearance_rate = arrears_cfg.get('clearance_rate', 0.5)  # 年度清偿比例
        self.arrears_interest_rate = arrears_cfg.get('interest_rate', 0.0)  # 拖欠利息（通常为0）
        
        # 拖欠状态跟踪（agent_id -> 拖欠金额）
        self.subsidy_arrears = {}  # 累计拖欠
        self.current_year_arrears = {}  # 本年新增拖欠
        self.arrears_paid_this_year = {}  # 本年清偿的历史拖欠
        self.total_arrears_outstanding = 0  # 总未清偿拖欠
        
        # 政策历史（新增）
        self.policy_history = []                        # 政策历史
        self.price_history = []                         # 价格历史
        self.regional_policy_assignments = {}
        self.latest_regional_summary = {}
        self.active_scenario_id = None
        results_cfg = self.model.config.get('results', {})
        dashboard_rel = results_cfg.get('policy_dashboard_path', 'results/policy_dashboard.json')
        dashboard_path = Path(dashboard_rel)
        if not dashboard_path.is_absolute():
            dashboard_path = PROJECT_ROOT / dashboard_rel
        self.policy_dashboard_path = dashboard_path
        policy_section = self.model.config.get('policy', {})
        self.gov_policy_config = policy_section.get('government', {})

        # 多目标政府优化参数
        policy_cfg = self.model.config.get('policy', {})
        objectives = policy_cfg.get('government_objectives', {})
        weights = objectives.get('weights', {})
        targets = objectives.get('targets', {})
        adjustments = objectives.get('adjustments', {})
        self.objective_weights = {
            'grain_security': weights.get('grain_security', 0.4),
            'farmer_welfare': weights.get('farmer_welfare', 0.35),
            'fiscal_balance': weights.get('fiscal_balance', 0.25)
        }
        self.target_grain_output = targets.get('grain_output', 200000000)
        self.target_avg_income = targets.get('avg_farmer_income', 20000)
        self.subsidy_budget = targets.get('subsidy_budget', 100000000)
        self.subsidy_step = adjustments.get('subsidy_step', 0.01)
        self.price_floor_step = adjustments.get('price_floor_step', 0.05)
        self.rent_incentive_step = adjustments.get('rent_incentive_step', 0.01)

    def step(self):
        """
        @theory
        政府年度决策周期 - 严格按照多层次博弈设计规范的阶段顺序执行
        """
        # @new 2025-11-30: 年初重置预算
        self._reset_annual_budget()
        
        # 阶段一：政策发布
        self.announce_policy()
        
        # 阶段五：补贴发放（遵循预算约束）
        self.distribute_subsidies()
        
        # 阶段六：政策评估与调整
        self.evaluate_and_adjust_policy()
        
        # @new 2025-11-30: 年末财政评估
        self._evaluate_fiscal_status()

    def _reset_annual_budget(self):
        """
        @new 2025-11-30: 年初重置年度预算
        
        财政悬崖机制：
        - 如果连续多年超支，预算会被削减
        - 反映中央财政对地方的约束
        """
        base_budget = self.annual_subsidy_budget
        
        if self.fiscal_cliff_active:
            # 计算削减后的预算
            reduction = 1 - (self.fiscal_reduction_rate * 
                           (self.consecutive_overspending_count - self.consecutive_overspending_years + 1))
            reduction = max(self.minimum_budget_ratio, reduction)
            base_budget = self.annual_subsidy_budget * reduction
            print(f"  [⚠ 财政悬崖] 预算削减至 {reduction*100:.0f}%，本年预算 {base_budget/10000:.1f}万元")
        
        self.remaining_budget = base_budget
    
    def _evaluate_fiscal_status(self):
        """
        @new 2025-11-30: 年末财政状况评估
        
        检查是否超支，更新财政悬崖状态
        """
        utilization = self.total_subsidy_paid / self.annual_subsidy_budget if self.annual_subsidy_budget > 0 else 0
        self.budget_utilization_history.append({
            'step': self.model.schedule.steps,
            'budget': self.annual_subsidy_budget,
            'spent': self.total_subsidy_paid,
            'utilization': utilization
        })
        
        if utilization > self.overspending_threshold:
            self.consecutive_overspending_count += 1
            print(f"  [⚠ 财政预警] 预算超支 {(utilization-1)*100:.1f}%，连续{self.consecutive_overspending_count}年")
            
            if self.consecutive_overspending_count >= self.consecutive_overspending_years:
                if not self.fiscal_cliff_active:
                    self.fiscal_cliff_active = True
                    print(f"  [! 财政悬崖触发] 连续{self.consecutive_overspending_count}年超支，明年预算将削减")
        else:
            # 如果今年没超支，重置计数器
            self.consecutive_overspending_count = 0
            self.fiscal_cliff_active = False

    def announce_policy(self):
        """
        @theory
        阶段一：政策发布
        公布本年度的各项补贴率和价格地板，为所有参与者的决策设定外部支付结构
        支持从 policy_params.json 读取多种补贴场景，并支持 policy.overrides 动态覆盖
        """
        import json
        from pathlib import Path
        
        # 读取配置
        policy_section = self.model.config.get('policy', {})
        policy_config = self.gov_policy_config or policy_section.get('government', {})
        scenario_id = policy_config.get('scenario_id', policy_section.get('active_scenario', 'baseline_cash'))
        policy_overrides = policy_section.get('overrides', {})
        self.active_scenario_id = scenario_id
        
        # 读取 policy_params.json 中的场景配置
        policy_params_path = PROJECT_ROOT / 'config' / 'policy_params.json'
        policy_scenarios = {}
        if policy_params_path.exists():
            try:
                with open(policy_params_path, 'r', encoding='utf-8') as f:
                    policy_data = json.load(f)
                    policy_scenarios = policy_data.get('scenarios', {})
            except Exception as e:
                print(f"[警告] 无法读取政策配置文件: {e}")
        
        # 构建政策参数（优先级：overrides > policy_params.json > 默认值）
        scenario = {}
        if scenario_id in policy_scenarios:
            scenario = dict(policy_scenarios[scenario_id])
        
        # 应用 overrides（关键修复）
        if policy_overrides:
            scenario.update(policy_overrides)
            print(f"[政府] 检测到 policy.overrides，已应用: {list(policy_overrides.keys())}")
        
        # 设置政府参数
        self.grain_price_floor = scenario.get('grain_price_floor', policy_config.get('grain_price_floor', 1.2))
        self.grain_subsidies_rate = scenario.get('grain_subsidies_rate', policy_config.get('grain_subsidies_rate', 0.15))
        self.subsidy_rate = scenario.get('grain_subsidies_rate', self.grain_subsidies_rate)
        
        subsidy_type_str = scenario.get('subsidy_type', policy_config.get('subsidy_type', 'area_based'))
        self.subsidy_type = SubsidyType.AREA_BASED if subsidy_type_str == 'area_based' else SubsidyType.OUTPUT_BASED
        
        subsidy_target_str = scenario.get('subsidy_target', policy_config.get('subsidy_target', 'operator'))
        self.subsidy_target = SubsidyTarget.OPERATOR if subsidy_target_str == 'operator' else SubsidyTarget.OWNER
        
        # 构建完整政策字典传递给环境
        policy_payload = {
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate,
            'base_subsidy_per_mu': scenario.get('base_subsidy_per_mu', 100),
            'subsidy_type': subsidy_type_str,
            'subsidy_target': subsidy_target_str,
            'subsidy_rate': self.subsidy_rate,
            'delivery_mode': scenario.get('delivery_mode', 'cash'),
            'payment_lag': scenario.get('payment_lag', 0),
            'cash_share': scenario.get('cash_share', 1.0),
            'in_kind_package': scenario.get('in_kind_package', {}),
            'tier_breakpoints': scenario.get('tier_breakpoints', []),
            'tier_rates': scenario.get('tier_rates', [])
        }
        
        # 将政策信息传递给环境
        self.model.environment.update_policy(self)
        self.model.environment.set_active_policy(policy_payload, policy_id=scenario_id)
        self.model.environment.manual_subsidy_dispatch = True
        region_summary = self._assign_regional_policies(policy_scenarios)
        
        # 记录政策历史
        self.policy_history.append({
            'step': self.model.schedule.steps,
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate,
            'subsidy_rate': self.subsidy_rate,
            'scenario_id': scenario_id
        })
        
        if region_summary:
            region_display = {region: info.get('scenario_id') for region, info in region_summary.items()}
            print(f"[政府] 区域差异化策略: {region_display}")
        
        print(f"政府发布政策（场景：{scenario_id}）：粮食最低收购价={self.grain_price_floor}，粮食补贴率={self.grain_subsidies_rate}")

    def get_current_policy(self):
        """
        @fix: 新增方法，返回当前政策字典以修复AttributeError
        """
        return {
            'subsidy_type': self.subsidy_type.value,
            'subsidy_target': self.subsidy_target.value,
            'subsidy_rate': self.subsidy_rate,
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate
        }
    
    def set_active_policy(self, policy_id, policy_state):
        """
        设置当前活动政策（由ModelBuilder调用）
        
        参数:
            policy_id: 政策场景ID
            policy_state: 政策状态字典
        """
        if policy_state:
            self.grain_price_floor = policy_state.get('grain_price_floor', 1.2)
            self.grain_subsidies_rate = policy_state.get('grain_subsidies_rate', 0.15)
            self.subsidy_rate = policy_state.get('grain_subsidies_rate', 0.15)
            
            subsidy_type_str = policy_state.get('subsidy_type', 'area_based')
            self.subsidy_type = SubsidyType.AREA_BASED if subsidy_type_str == 'area_based' else SubsidyType.OUTPUT_BASED
            
            subsidy_target_str = policy_state.get('subsidy_target', 'operator')
            self.subsidy_target = SubsidyTarget.OPERATOR if subsidy_target_str == 'operator' else SubsidyTarget.OWNER

    def _get_region_label(self, actor):
        if hasattr(self.model, 'get_region_label'):
            return self.model.get_region_label(actor)
        if not hasattr(actor, 'pos') or actor.pos is None:
            return 'global'
        _, y = actor.pos
        height = max(1, getattr(self.model.grid, 'height', 1))
        if y < height / 3:
            return 'north'
        if y < (2 * height) / 3:
            return 'central'
        return 'south'

    def _compute_region_statistics(self):
        stats = {}
        regions = {}
        for farmer in getattr(self.model, 'farmers', []):
            region = self._get_region_label(farmer)
            regions.setdefault(region, []).append(farmer)
        for region, farmers in regions.items():
            if not farmers:
                continue
            incomes = [getattr(f, 'total_income', 0) for f in farmers if getattr(f, 'total_income', 0) > 0]
            agri_ratios = []
            for f in farmers:
                total = getattr(f, 'total_income', 0)
                if total > 0:
                    agri = getattr(f, 'agricultural_income', 0)
                    agri_ratios.append(agri / total)
            avg_income = sum(incomes) / len(incomes) if incomes else 0.0
            avg_agri_ratio = sum(agri_ratios) / len(agri_ratios) if agri_ratios else 0.0
            stats[region] = {
                'avg_income': avg_income,
                'avg_agri_ratio': avg_agri_ratio,
                'population': len(farmers)
            }
            network_meta = getattr(self.model, 'network_metrics', {}).get(region, {})
            stats[region]['avg_transaction_degree'] = network_meta.get('avg_transaction_degree', 0.0)
            stats[region]['transaction_density'] = network_meta.get('transaction_density', 0.0)
        self.region_stats = stats
        return stats

    def _assign_regional_policies(self, policy_scenarios):
        cfg = self.gov_policy_config or {}
        if not cfg.get('enable_regional_heterogeneity'):
            self.regional_policy_assignments = {}
            self.latest_regional_summary = {}
            return None
        stats = self._compute_region_statistics()
        if not stats:
            self.regional_policy_assignments = {}
            self.latest_regional_summary = {}
            return None
        thresholds = cfg.get('regional_thresholds', {})
        income_low = thresholds.get('income_low', 20000)
        agri_ratio_high = thresholds.get('agri_ratio_high', 0.8)
        density_low = thresholds.get('network_density_low', 0.2)
        scenario_map = cfg.get('regional_scenarios', {})
        vulnerable_scenario = scenario_map.get('vulnerable', self.active_scenario_id)
        transforming_scenario = scenario_map.get('transforming', self.active_scenario_id)
        assignments = {}
        summary = {}
        for region, region_stats in stats.items():
            scenario_choice = transforming_scenario
            density = region_stats.get('transaction_density', 0)
            if (region_stats.get('avg_income', 0) < income_low or
                region_stats.get('avg_agri_ratio', 0) > agri_ratio_high or
                density < density_low):
                scenario_choice = vulnerable_scenario
            scenario_payload = policy_scenarios.get(scenario_choice)
            if not scenario_payload:
                scenario_payload = policy_scenarios.get(self.active_scenario_id, self.model.environment.active_policy)
            assignments[region] = {
                'scenario_id': scenario_choice,
                'policy': scenario_payload
            }
            summary[region] = {
                'scenario_id': scenario_choice,
                'avg_income': region_stats.get('avg_income', 0),
                'avg_agri_ratio': region_stats.get('avg_agri_ratio', 0),
                'population': region_stats.get('population', 0)
            }
        self.regional_policy_assignments = assignments
        self.latest_regional_summary = summary
        return summary

    def _get_region_scenario_id(self, region):
        entry = self.regional_policy_assignments.get(region)
        return entry.get('scenario_id') if entry else self.active_scenario_id

    def _get_policy_for_region(self, region):
        entry = self.regional_policy_assignments.get(region)
        return entry.get('policy') if entry else None
    
    def _find_agent_by_id(self, agent_id):
        """
        @new 2025-12-04: 根据ID查找主体
        用于拖欠清偿时找到对应的农户或企业
        """
        # 先在农户中查找
        for farmer in self.model.farmers:
            if farmer.unique_id == agent_id:
                return farmer
        # 再在企业中查找
        for enterprise in self.model.agricultural_enterprises:
            if enterprise.unique_id == agent_id:
                return enterprise
        return None

    def distribute_subsidies(self):
        """
        @theory
        阶段五：补贴发放
        根据政府设定的subsidy_target精确发放补贴，支持时滞和实物补贴
        
        @new 2025-11-30: 添加财政预算约束
        - 如果启用hard_cap，补贴总额不能超过年度预算
        - 超出预算时按比例缩减每个主体的补贴
        """
        self.total_subsidy_paid = 0
        self.subsidy_records = []
        
        # 获取当前政策配置
        active_policy = self.model.environment.active_policy
        subsidy_target_str = active_policy.get('subsidy_target', 'operator')
        payment_lag = active_policy.get('payment_lag', 0)
        current_step = self.model.schedule.steps
        
        # ========== 第一阶段：计算应发补贴总额 ==========
        pending_subsidies = []  # 待发放补贴列表
        total_planned = 0
        
        if subsidy_target_str == 'owner':
            # 补贴给土地所有者
            for farmer in self.model.farmers:
                owned_area = sum(patch.area for patch in farmer.land_patches if patch.owner == farmer)
                if owned_area > 0:
                    region = self._get_region_label(farmer)
                    policy_override = self._get_policy_for_region(region)
                    subsidy_info = self.model.environment.calculate_subsidy_for_actor(
                        owned_area,
                        actor_type='farmer',
                        policy_override=policy_override
                    )
                    planned_amount = subsidy_info['cash'] + subsidy_info['in_kind_value']
                    total_planned += planned_amount
                    pending_subsidies.append({
                        'agent': farmer,
                        'agent_type': 'farmer',
                        'subsidy_info': subsidy_info,
                        'planned_amount': planned_amount,
                        'region': region
                    })
        
        elif subsidy_target_str == 'operator':
            # 补贴给实际耕作者
            for farmer in self.model.farmers:
                operated_area = farmer.get_operated_land_area()
                if operated_area > 0:
                    region = self._get_region_label(farmer)
                    policy_override = self._get_policy_for_region(region)
                    subsidy_info = self.model.environment.calculate_subsidy_for_actor(
                        operated_area,
                        actor_type='farmer',
                        policy_override=policy_override
                    )
                    planned_amount = subsidy_info['cash'] + subsidy_info['in_kind_value']
                    total_planned += planned_amount
                    pending_subsidies.append({
                        'agent': farmer,
                        'agent_type': 'farmer',
                        'subsidy_info': subsidy_info,
                        'planned_amount': planned_amount,
                        'region': region
                    })
            
            for enterprise in self.model.agricultural_enterprises:
                operated_area = enterprise.get_operated_land_area()
                if operated_area > 0:
                    region = self._get_region_label(enterprise)
                    policy_override = self._get_policy_for_region(region)
                    subsidy_info = self.model.environment.calculate_subsidy_for_actor(
                        operated_area,
                        actor_type='enterprise',
                        policy_override=policy_override
                    )
                    planned_amount = subsidy_info['cash'] + subsidy_info['in_kind_value']
                    total_planned += planned_amount
                    pending_subsidies.append({
                        'agent': enterprise,
                        'agent_type': 'enterprise',
                        'subsidy_info': subsidy_info,
                        'planned_amount': planned_amount,
                        'region': region
                    })
        
        # ========== 第二阶段：先清偿历史拖欠 ==========
        self.arrears_paid_this_year = {}
        arrears_payment_budget = 0
        
        if self.enable_arrears and self.total_arrears_outstanding > 0:
            # 用当期预算的一部分清偿历史拖欠
            arrears_payment_budget = min(
                self.remaining_budget * self.arrears_clearance_rate,
                self.total_arrears_outstanding
            )
            
            if arrears_payment_budget > 0:
                # 按比例清偿各主体的拖欠
                for agent_id, arrears_amount in list(self.subsidy_arrears.items()):
                    if arrears_amount > 0:
                        payment_ratio = arrears_payment_budget / self.total_arrears_outstanding
                        payment = arrears_amount * payment_ratio
                        self.arrears_paid_this_year[agent_id] = payment
                        self.subsidy_arrears[agent_id] -= payment
                        
                        # 将清偿金额发放给主体
                        agent = self._find_agent_by_id(agent_id)
                        if agent:
                            if hasattr(agent, 'subsidy_income'):
                                agent.arrears_received = payment  # 新增属性：收到的拖欠补贴
                            elif hasattr(agent, 'subsidy_revenue'):
                                agent.arrears_received = payment
                
                self.total_arrears_outstanding -= arrears_payment_budget
                self.remaining_budget -= arrears_payment_budget
                print(f"  [拖欠清偿] 清偿历史拖欠 {arrears_payment_budget/10000:.1f}万元，"
                      f"剩余未清偿 {self.total_arrears_outstanding/10000:.1f}万元")
        
        # ========== 第三阶段：计算当期补贴发放方案 ==========
        scaling_factor = 1.0
        arrears_factor = 0.0  # 拖欠比例
        budget_exceeded = False
        
        if total_planned > self.remaining_budget and self.remaining_budget > 0:
            if self.enable_arrears:
                # 启用拖欠机制：先按预算发放，超出部分记为拖欠
                if total_planned > 0:
                    # 计算可发放比例
                    scaling_factor = min(1.0, self.remaining_budget / total_planned)
                    # 超出部分记为拖欠，但不超过最大拖欠比例
                    arrears_factor = min(1.0 - scaling_factor, self.max_arrears_ratio)
                    scaling_factor = max(scaling_factor, 1.0 - self.max_arrears_ratio)
                budget_exceeded = True
                print(f"  [财政约束-拖欠机制] 计划 {total_planned/10000:.1f}万元，"
                      f"预算 {self.remaining_budget/10000:.1f}万元，"
                      f"发放 {scaling_factor*100:.0f}%，拖欠 {arrears_factor*100:.0f}%")
            elif self.hard_cap:
                # 硬性约束：直接按比例缩减
                scaling_factor = self.remaining_budget / total_planned
                budget_exceeded = True
                print(f"  [财政约束] 计划补贴 {total_planned/10000:.1f}万元 > 预算 {self.remaining_budget/10000:.1f}万元，"
                      f"按 {scaling_factor*100:.1f}% 比例发放")
        
        # ========== 第四阶段：发放补贴并记录拖欠 ==========
        self.current_year_arrears = {}
        
        for item in pending_subsidies:
            agent = item['agent']
            subsidy_info = item['subsidy_info']
            region = item['region']
            
            # 如果需要时滞，检查是否到期
            if payment_lag > 0 and current_step < payment_lag:
                continue
            
            # 计算实际发放和拖欠金额
            planned_cash = subsidy_info['cash']
            planned_in_kind = subsidy_info['in_kind_value']
            
            actual_cash = planned_cash * scaling_factor
            actual_in_kind_value = planned_in_kind * scaling_factor
            
            # 记录拖欠
            arrears_amount = (planned_cash + planned_in_kind) * arrears_factor
            if arrears_amount > 0:
                agent_id = agent.unique_id
                self.current_year_arrears[agent_id] = arrears_amount
                self.subsidy_arrears[agent_id] = self.subsidy_arrears.get(agent_id, 0) + arrears_amount
                self.total_arrears_outstanding += arrears_amount
            
            if item['agent_type'] == 'farmer':
                agent.subsidy_income = actual_cash
                agent.subsidy_arrears_owed = self.subsidy_arrears.get(agent.unique_id, 0)  # 告知农户欠款
                if hasattr(agent, 'in_kind_subsidies'):
                    # 实物补贴也按比例缩减
                    scaled_in_kind = {k: v * scaling_factor for k, v in subsidy_info['in_kind_breakdown'].items()}
                    agent.in_kind_subsidies = scaled_in_kind
            else:  # enterprise
                agent.subsidy_revenue = actual_cash
                agent.subsidy_arrears_owed = self.subsidy_arrears.get(agent.unique_id, 0)
                if hasattr(agent, 'in_kind_subsidies'):
                    scaled_in_kind = {k: v * scaling_factor for k, v in subsidy_info['in_kind_breakdown'].items()}
                    agent.in_kind_subsidies = scaled_in_kind
            
            self.total_subsidy_paid += actual_cash + actual_in_kind_value
            self.subsidy_records.append({
                'recipient': agent.unique_id,
                'amount': actual_cash,
                'in_kind_value': actual_in_kind_value,
                'planned_amount': item['planned_amount'],
                'scaling_factor': scaling_factor,
                'arrears_amount': arrears_amount,  # 新增：记录拖欠
                'type': f"{item['agent_type']}_{'owner' if subsidy_target_str == 'owner' else 'operator'}",
                'region': region,
                'regional_policy': self._get_region_scenario_id(region)
            })
        
        # 更新剩余预算
        self.remaining_budget -= self.total_subsidy_paid
        
        if budget_exceeded:
            print(f"  [补贴发放] 实际发放 {self.total_subsidy_paid/10000:.1f}万元，"
                  f"新增拖欠 {sum(self.current_year_arrears.values())/10000:.1f}万元，"
                  f"剩余预算 {self.remaining_budget/10000:.1f}万元")

    def calculate_subsidy_for_farmer(self, farmer):
        """
        @theory
        计算农户补贴金额
        """
        # 根据农户拥有的土地面积计算补贴
        owned_area = sum(patch.area for patch in farmer.land_patches if patch.owner == farmer)
        
        # 基础补贴
        base_subsidy = owned_area * self.subsidy_rate
        
        # 粮食补贴（如果种植粮食作物）
        grain_subsidy = 0
        if hasattr(farmer, 'planned_crop') and farmer.planned_crop:
            if farmer.planned_crop.name == 'GRAIN':
                grain_subsidy = owned_area * self.grain_subsidies_rate
        
        return base_subsidy + grain_subsidy

    def calculate_subsidy_by_area(self, area):
        """
        @theory
        根据面积计算补贴金额
        """
        return area * self.subsidy_rate

    def evaluate_and_adjust_policy(self):
        """
        @theory
        阶段六：政策评估与调整
        根据市场表现调整政策参数
        """
        # 分析市场表现
        market_performance = self.analyze_market_performance()
        
        # 根据市场表现调整政策
        self.adjust_policy_parameters(market_performance)
        
        # 记录价格历史
        self.price_history.append({
            'step': self.model.schedule.steps,
            'grain_price_floor': self.grain_price_floor,
            # @fix: market_price_cash_crop 由环境决定，此处不再重复记录
            # 'market_price_cash_crop': self.market_price_cash_crop,
            'total_subsidy_paid': self.total_subsidy_paid
        })
        self._record_policy_dashboard(market_performance)

    def analyze_market_performance(self):
        """
        @theory
        分析市场表现
        """
        # 计算农户平均收入
        farmer_incomes = [farmer.total_income for farmer in self.model.farmers]
        avg_farmer_income = sum(farmer_incomes) / len(farmer_incomes) if farmer_incomes else 0
        
        # 计算企业平均利润
        enterprise_profits = [enterprise.profit for enterprise in self.model.agricultural_enterprises]
        avg_enterprise_profit = sum(enterprise_profits) / len(enterprise_profits) if enterprise_profits else 0
        
        # 计算土地流转率（按面积）
        total_land_area = sum(patch.area for patch in self.model.land_patches)
        leased_land_area = sum(patch.area for patch in self.model.land_patches if patch.leased_to is not None)
        land_transfer_rate = leased_land_area / total_land_area if total_land_area > 0 else 0

        # 计算粮食总产量
        grain_output = 0
        for farmer in self.model.farmers:
            if getattr(farmer, 'last_crop_choice', None) and getattr(farmer.last_crop_choice, 'value', '') == "Grain":
                grain_output += getattr(farmer, 'last_yield', 0)
        for enterprise in self.model.agricultural_enterprises:
            if getattr(enterprise, 'last_crop_choice', None) and getattr(enterprise.last_crop_choice, 'value', '') == "Grain":
                grain_output += getattr(enterprise, 'last_yield', 0)
        
        return {
            'avg_farmer_income': avg_farmer_income,
            'avg_enterprise_profit': avg_enterprise_profit,
            'land_transfer_rate': land_transfer_rate,
            'total_subsidy_paid': self.total_subsidy_paid,
            'grain_output': grain_output
        }

    def adjust_policy_parameters(self, market_performance):
        """
        @theory
        根据市场表现调整政策参数
        """
        grain_target = self.target_grain_output or 1
        income_target = self.target_avg_income or 1
        subsidy_budget = self.subsidy_budget or 1

        grain_score = market_performance['grain_output'] / grain_target
        income_score = market_performance['avg_farmer_income'] / income_target
        fiscal_ratio = market_performance['total_subsidy_paid'] / subsidy_budget

        # 粮食安全优先：低于目标则提高托市价
        if grain_score < 0.95:
            self.grain_price_floor = min(self.grain_price_floor + self.price_floor_step, 3.0)
        elif grain_score > 1.05 and fiscal_ratio > 1.0:
            self.grain_price_floor = max(self.grain_price_floor - self.price_floor_step, 0.8)

        # 农民福利：低于目标则提高补贴
        if income_score < 0.95:
            self.subsidy_rate = min(self.subsidy_rate + self.subsidy_step, 0.3)
        elif income_score > 1.1 and fiscal_ratio > 1.0:
            self.subsidy_rate = max(self.subsidy_rate - self.subsidy_step, 0.05)

        # 土地流转效率仍作为辅助调整
        land_rate = market_performance['land_transfer_rate']
        if land_rate < 0.3:
            self.subsidy_rate = min(self.subsidy_rate + self.rent_incentive_step, 0.35)
        elif land_rate > 0.7:
            self.subsidy_rate = max(self.subsidy_rate - self.rent_incentive_step, 0.05)

        # @fix 2024-11-19: 不要覆盖grain_subsidies_rate，它应该保持政策配置值（如1.0）
        # 原因：自适应调整的subsidy_rate（约0.15-0.35）会导致企业补贴从403元降到125元
        # 解决：grain_subsidies_rate由政策场景控制，不参与自适应调整
        # self.grain_subsidies_rate = self.subsidy_rate  # ❌ 已禁用
        self._sync_policy_with_environment()

    def _sync_policy_with_environment(self):
        """将调整后的政策参数写回环境，确保后续主体读取的是最新政策。"""
        active_policy = dict(self.model.environment.active_policy or {})
        active_policy['grain_price_floor'] = self.grain_price_floor
        active_policy['grain_subsidies_rate'] = self.grain_subsidies_rate
        active_policy['base_subsidy_per_mu'] = active_policy.get('base_subsidy_per_mu', 100)
        active_policy['subsidy_type'] = self.subsidy_type.value
        active_policy['subsidy_target'] = self.subsidy_target.value
        active_policy['subsidy_rate'] = self.subsidy_rate
        self.model.environment.set_active_policy(active_policy, getattr(self.model, 'active_policy_id', None))

    def get_subsidy_statistics(self):
        """
        @theory
        获取补贴统计信息
        """
        return {
            'total_subsidy_paid': self.total_subsidy_paid,
            'subsidy_records_count': len(self.subsidy_records),
            'current_policy': {
                'grain_price_floor': self.grain_price_floor,
                'grain_subsidies_rate': self.grain_subsidies_rate,
                # @fix: 移除 market_price_cash_crop
                # 'market_price_cash_crop': self.market_price_cash_crop,
                'subsidy_rate': self.subsidy_rate,
                'subsidy_type': self.subsidy_type.value,
                'subsidy_target': self.subsidy_target.value
            }
        } 

    def _record_policy_dashboard(self, market_performance):
        path = self.policy_dashboard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dashboard = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        except json.JSONDecodeError:
            dashboard = []
        entry = {
            'step': self.model.schedule.steps,
            'scenario_id': self.active_scenario_id,
            'regional_assignments': {region: info.get('scenario_id') for region, info in self.regional_policy_assignments.items()},
            'regional_summary': self.latest_regional_summary,
            'kpis': market_performance
        }
        dashboard.append(entry)
        path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding='utf-8')