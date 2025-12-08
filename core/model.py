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
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np

from modules.land_clearing_house import LandClearingHouse
from modules.land_patch import LandPatch
# @new 2025-12-04: 导入新模块
from modules.graveyard import Graveyard, ExitReason

class MainModel(Model):
    def __init__(self, config, environment, seed=None):
        """
        @refactor: __init__ 大幅简化，只接收必要对象
        """
        super().__init__(seed=seed)
        
        self.config = config
        self.environment = environment # @refactor: 接收环境对象
        self.running = True
        self.step_count = 0

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
        
        # 智能体列表 - 由ModelBuilder填充
        self.farmers = []
        self.agricultural_enterprises = []
        self.government = None
        self.land_patches = []
        
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
        
        # @new 2025-12-04: 墓地统计系统 - 避免幸存者偏差
        # 理论：仅分析存活智能体会产生严重幸存者偏差
        # 墓地系统记录所有退出智能体的完整信息，支持事后分析
        self.graveyard = Graveyard()
        
        # @new 2025-12-04: 服务市场统计
        self.service_market_stats = {
            'total_contracts': 0,
            'total_service_area': 0,
            'total_service_fee': 0,
            'contracts_by_type': {},
            'provider_utilization': []
        }
        
        # 数据收集器 - 由ModelBuilder设置
        self.datacollector = None

    def step(self):
        """
        @theory
        模型年度决策周期
        严格按照设计规范的阶段顺序执行
        
        @fix 2025-12: 添加多年期合同年限递减逻辑
        @fix 2025-11-29: 添加劳动力市场出清步骤
        @new 2025-12-04: 添加服务市场统计重置
        """
        self.step_count += 1
        
        # @new 2025-12-04: 重置当步服务市场统计
        self._reset_service_market_stats()
        
        # 0. 更新土地租赁合同状态（多年期合同机制）
        self._update_land_lease_contracts()
        
        # 1. 环境更新 (政策和市场价格)
        self.environment.update_policy(self.government)
        self.environment.update_market_prices(self)

        # 阶段二：土地租赁市场
        self.conduct_land_rental_market()
        
        # @new 2025-11-29: 阶段三：劳动力市场出清
        # 在生产决策之前完成劳动力市场匹配
        self._conduct_labor_market()
        
        # @new 2025-12: 阶段四：农业社会化服务市场出清
        # 理论：服务规模经济 - 让小农户通过购买托管服务实现规模效益
        self._conduct_service_market()
        
        # 2. 智能体决策
        self.schedule.step()

        # 3. 数据收集
        self.datacollector.collect(self)
    
    def _conduct_labor_market(self):
        """
        @new 2025-11-29: 劳动力市场出清
        
        理论：
        1. 农户既是劳动力供给者，也可能是需求者（大户）
        2. 企业使用大型机械 + 雇佣农户劳动力
        3. 劳动力市场匹配后，确定雇佣关系和工资
        """
        if self.labor_market is not None:
            self.labor_market.update_market(self.farmers, self.agricultural_enterprises)
    
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
            demand = self._get_farmer_service_demand(farmer)
            if demand is not None:
                service_demands.append(demand)
        
        # 2. 服务市场匹配
        if service_demands:
            matches = self.service_market.match_service_demands(
                service_demands, 
                self.service_providers
            )
            
            # 3. 应用匹配结果
            for match in matches:
                self._apply_service_contract(match)
    
    def _get_farmer_service_demand(self, farmer):
        """
        @theory 判断农户是否有服务需求
        
        触发条件：
        1. 劳动力不足（如老年农户、兼业户）
        2. 有足够资金购买服务
        3. 成本效益可行
        """
        # 获取服务决策配置 (在 production.agricultural_service 下)
        prod_config = self.config.get('production', {})
        service_config = prod_config.get('agricultural_service', {})
        if not service_config.get('enabled', False):
            return None
            
        farmer_config = service_config.get('farmer_decision', {})
        
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
        
        # 判断是否需要服务
        needs_service = labor_sufficiency < labor_scarcity_threshold
        
        if not needs_service:
            return None
        
        # 条件2: 经营面积（只有有地的人才需要服务）
        operated_area = farmer.get_operated_land_area()
        if operated_area < 1:  # 少于1亩不需要
            return None
        
        # 条件3: 确定服务类型和需求
        from modules.agricultural_service import ServiceType
        
        if labor_sufficiency < 0.3:
            # 劳动力极度匮乏，需要全程托管
            service_type = ServiceType.FULL_TRUSTEESHIP
        elif labor_sufficiency < 0.6:
            # 中度匮乏，关键环节托管
            service_type = ServiceType.PARTIAL_TRUSTEESHIP
        else:
            # 轻度匮乏，单一环节服务
            service_type = ServiceType.SINGLE_SERVICE
        
        return {
            'farmer': farmer,
            'area': operated_area,
            'service_type': service_type,
            'max_price_per_mu': self._calculate_max_service_price(farmer, service_type)
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
        
        if service_type.value == 'Full_Trusteeship':
            type_config = service_types.get('full_trusteeship', {})
        elif service_type.value == 'Partial_Trusteeship':
            type_config = service_types.get('partial_trusteeship', {})
        else:
            type_config = service_types.get('single_service', {})
        
        # 使用配置的最大价格
        return type_config.get('max_fee', 600)
    
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
        
        # 3. 扣除服务费用
        service_fee = contract.fee_per_mu * contract.area
        if hasattr(farmer, 'cash'):
            farmer.cash -= service_fee
        
        # 4. 更新服务商状态
        if hasattr(provider, 'add_service_contract'):
            provider.add_service_contract(contract)
        
        # @new 2025-12-04: 收集统计数据
        self._update_service_market_stats(contract, provider)
    
    def _update_service_market_stats(self, contract, provider):
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
        self.service_market_stats['total_contracts'] += 1
        self.service_market_stats['total_service_area'] += getattr(contract, 'area', 0)
        service_fee = getattr(contract, 'fee_per_mu', 0) * getattr(contract, 'area', 0)
        self.service_market_stats['total_service_fee'] += service_fee
        
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
            'provider_utilization_list': provider_utilization
        }
    
    def _reset_service_market_stats(self):
        """
        @new 2025-12-04: 在每个步骤开始时重置服务市场统计
        """
        self.service_market_stats = {
            'total_contracts': 0,
            'total_service_area': 0,
            'total_service_fee': 0,
            'contracts_by_type': {},
            'provider_utilization': []
        }
    
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
        self.graveyard.record_exit(
            agent=agent,
            step=self.step_count,
            reason=reason,
            additional_details=details
        )
    
    def get_graveyard_summary(self):
        """
        @new 2025-12-04: 获取墓地统计摘要
        """
        return self.graveyard.get_exit_summary()
    
    def _update_land_lease_contracts(self):
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
            # 企业通常愿意续约以保持规模
            return True
        elif isinstance(tenant, FarmerAgent):
            # 规模户愿意续约
            if tenant.livelihood_strategy.value == 'Pure_Farmer':
                return True
            return False
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
        import random
        adjustment = random.uniform(-0.05, 0.10)
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
        
        # 阶段1：让所有农户提交供给意愿
        for farmer in self.farmers:
            farmer.participate_land_rental_market()
        
        # 阶段2：收集已提交供给的地块，供企业参考
        # @critical: 这是关键改进 - 让企业知道哪些地块在市场上
        supplied_patches = set()
        for offer in self.land_clearing_house.supply_offers:
            supplied_patches.add(offer['patch'])
        
        # 阶段3：企业对已供给的地块提交需求
        for enterprise in self.agricultural_enterprises:
            enterprise.participate_land_rental_market(available_patches=list(supplied_patches))
        
        # 市场出清
        self.land_clearing_house.clear_market()

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

    def get_subsidy_incidence_ratio(self):
        """补贴资本化程度"""
        # 简化计算：年度平均租金上涨额 / 单位面积补贴额
        if self.step_count == 0:
            return 0
        
        # 这里需要更复杂的计算逻辑
        return 0.5  # 占位符

    def get_income_gini_coefficient(self):
        """农村内部的收入基尼系数"""
        incomes = [farmer.total_income for farmer in self.farmers]
        
        if not incomes or sum(incomes) == 0:
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
        return self.land_clearing_house.get_market_statistics()['average_price']

    def get_land_transfer_rate(self):
        """土地流转率（流转面积/总面积）"""
        total_area = sum(patch.area for patch in self.land_patches)
        if total_area <= 0:
            return 0
        leased_area = sum(patch.area for patch in self.land_patches if patch.leased_to is not None)
        return leased_area / total_area
    
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
        """衡量土地集中度：前10%经营主体占比"""
        areas = [farmer.get_operated_land_area() for farmer in self.farmers if farmer.get_operated_land_area() > 0]
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
