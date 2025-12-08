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
"""

from mesa import Agent
import numpy as np
from enum import Enum
import random
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

class FarmerAgent(Agent):
    def __init__(self, unique_id, model, land_patches, initial_capital=10000, farming_ability=None, config=None):
        super().__init__(unique_id, model)
        # @fix: 保存config引用以便后续使用
        self.agent_config = config if config is not None else model.config if hasattr(model, 'config') else {}
        ai_config = self.agent_config.get('ai', {})
        farmer_ai = ai_config.get('farmer', {})
        self.ai_enabled = ai_config.get('enabled', False)
        self.ai_non_farm_job_shock = farmer_ai.get('non_farm_job_shock', 0.0)
        self.ai_wage_penalty = farmer_ai.get('wage_penalty', 0.0)
        self.ai_adoption_lag = farmer_ai.get('adoption_lag', 0.0)
        
        # 核心属性
        self.land_patches = land_patches  # 拥有的地块列表
        self.capital = initial_capital    # 资本（缓冲能力）
        self.initial_capital = initial_capital  # 初始资本（用于财富效应计算）
        
        # 生计相关属性
        self.livelihood_strategy = LivelihoodStrategy.PURE_FARMER  # 生计类型
        
        # @new 2025-12: 农户经营规模分类
        # 初始规模根据拥有土地面积确定
        self.farmer_scale = self._determine_farmer_scale()
        
        # @fix: 接受从外部传入的 farming_ability 参数
        if farming_ability is not None:
            self.farming_ability = farming_ability
        else:
            self.farming_ability = np.random.uniform(0.8, 1.2)       # 耕作能力（学习能力）
        
        self.off_farm_wage_level = np.random.uniform(0.8, 1.2)   # 非农工资水平（人力资本）
        
        # 动态风险偏好机制（第二阶段新增）
        self.base_risk_aversion = np.random.uniform(1.5, 2.5)    # 基础风险规避系数
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
        self.social_learning_weight = np.random.uniform(0.2, 0.6)  # 社会学习权重（个体异质性）
        self.decision_mode = "hybrid"  # 决策模式：rational/social/hybrid
        self.neighbor_observation_history = []  # 邻居观察历史
        self.influence_received = []  # 受到的影响记录
        self.transaction_neighbors = []  # 土地交易可行性网络
        self.transaction_link_strength = {}  # 交易网络强度（局部匹配时刻画信任）
        
        # 信息传播与扩散（第二阶段新增）
        self.information_receptivity = np.random.uniform(0.3, 0.8)  # 信息接受度
        self.success_sharing_propensity = np.random.uniform(0.4, 0.9)  # 成功案例分享倾向
        self.conformity_tendency = np.random.uniform(0.2, 0.7)  # 从众倾向
        
        # 社会资本相关
        self.is_village_elite = random.random() < 0.2  # 20%概率为村精英（社会资本）
        self.social_network_strength = np.random.uniform(0.5, 1.5)  # 社会网络强度
        
        # 劳动力相关
        self.total_labor_units = 365  # 总劳动力单位（天）
        self.agricultural_labor = 0   # 农业劳动分配
        self.off_farm_labor = 0      # 非农劳动分配
        
        # 收入结构（年度）
        self.agricultural_income = 0      # 农业收入
        self.land_rental_income = 0      # 土地租金收入
        self.subsidy_income = 0          # 补贴收入
        self.scale_subsidy_income = 0    # @new 2025-12-08: 规模种粮补贴收入
        self.scale_subsidy_qualified = False  # @new 2025-12-08: 是否达到规模补贴门槛
        self.off_farm_income = 0         # 非农务工收入
        self.gross_income = 0            # 总收入（未减去支出）
        self.total_income = 0            # 总收入（向后兼容，等同于gross_income）
        self.in_kind_support_balance = 0.0  # 实物补贴折算价值
        self.in_kind_support_breakdown = {}
        
        # 支出结构（年度）
        self.production_cost = 0         # 农业生产成本
        self.land_rental_expense = 0     # 土地租金支出
        self.total_expense = 0           # 总支出
        
        # @new 2025-11-29: 劳动力市场相关
        # 理论：农户既是劳动力供给者，也可能是需求者（大户）
        self.hired_labor_cost = 0        # 雇佣劳动力支出（大户雇佣散户）
        self.labor_sale_income = 0       # 出售劳动力收入（被其他农户/企业雇佣）
        self.hired_labor_days = 0        # 雇佣的劳动天数
        self.labor_sold_days = 0         # 出售的劳动天数
        self.family_labor_used = 0       # 自家使用的劳动力天数
        self.labor_constraint_binding = False  # 劳动力约束是否绑定
        
        # @new 2025-12: 信贷约束相关
        # 专家审阅意见：规模户扩张依赖贷款，应有信贷约束
        self._initialize_farmer_credit_constraints()
        
        # 净利润和可支配收入
        self.net_income = 0              # 净利润 = 总收入 - 总支出
        self.disposable_income = 0       # 可支配收入（用于消费）
        
        # 消费和福利相关（新增）
        self.consumption_model = HouseholdConsumptionModel(config)  # 消费决策模型
        self.household_size = max(1, int(np.random.normal(3.5, 0.8)))  # 家庭人口数（根据2020年农村平均家庭规模）

        age_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("age_dist", {})
        if age_cfg.get("type") == "normal":
            mean_age = age_cfg.get("mean", 50.0)
            std_age = age_cfg.get("std", 10.0)
            self.age = int(np.clip(np.random.normal(mean_age, std_age), 20, 80))
        else:
            self.age = int(np.clip(np.random.normal(50.0, 10.0), 20, 80))
        
        # @new 2025-12: 初始化连续变量农户特征系统
        # 用连续变量替代离散标签，实现更精细的异质性刻画
        land_area = sum(p.area for p in self.land_patches) if self.land_patches else 10.0
        generation = self._determine_generation_type()
        self.characteristics = initialize_farmer_characteristics(
            age=self.age,
            land_area=land_area,
            initial_capital=initial_capital,
            generation=generation,
            random_state=np.random.default_rng()
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
        self.reservation_prices = {}     # 保留价格
        
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
            self.internal_orientation = np.random.uniform(min_io, max_io)
        else:
            self.internal_orientation = np.random.uniform(0.5, 0.9)  # 默认值
        
        self.safety_net_factor = reference_config.get('safety_net_factor', 0.9)  # 心理安全网系数
        self.last_beliefs = None  # 上一期的信念状态

        # @fix: 40%概率初始为经济作物（基于CLES数据，粮食作物约60%）
        if random.random() < 0.40:
            self.planned_crop = CropType.CASH_CROP
        else:
            self.planned_crop = CropType.GRAIN
        
        # @fix 2025-11-26: 根据初始生计策略设置合理的劳动力分配
        # 确保初始状态的收入结构与生计策略一致
        self._initialize_labor_allocation()

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
            
            # 农业劳动比例：直接由农业依赖度决定（加上一定随机噪声）
            # 映射：agri_dep ∈ [0, 1] → agri_labor_ratio ∈ [0.05, 0.90]
            base_labor_ratio = 0.05 + 0.85 * agri_dep  # 线性映射
            noise = np.random.uniform(-0.05, 0.05)  # ±5%随机噪声
            labor_ratio = np.clip(base_labor_ratio + noise, 0.05, 0.95)
            
            self.agricultural_labor = self.total_labor_units * labor_ratio
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            
            # 非农工资水平：与农业依赖度负相关
            # agri_dep高 → 非农能力低 → 工资水平低
            base_wage_level = 1.5 - 0.8 * agri_dep  # 映射到 [0.7, 1.5]
            wage_noise = np.random.uniform(-0.1, 0.1)
            self.off_farm_wage_level = np.clip(base_wage_level + wage_noise, 0.5, 1.6)
            return
        
        # 向后兼容：如果没有 characteristics，使用旧的离散逻辑
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户：75-90%的劳动力用于农业
            self.agricultural_labor = self.total_labor_units * np.random.uniform(0.75, 0.90)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 调整非农工资水平（纯农户通常非农收入能力较低）
            self.off_farm_wage_level = min(self.off_farm_wage_level, np.random.uniform(0.6, 0.9))
        
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业农户：30-60%的劳动力用于农业
            self.agricultural_labor = self.total_labor_units * np.random.uniform(0.30, 0.60)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 保持默认的工资水平（0.8-1.2）
        
        else:  # NON_FARMER
            # 非农户：5-20%的劳动力用于农业（仅维持自给自足）
            self.agricultural_labor = self.total_labor_units * np.random.uniform(0.05, 0.20)
            self.off_farm_labor = self.total_labor_units - self.agricultural_labor
            # 提高非农工资水平（非农户通常有更好的非农就业能力）
            self.off_farm_wage_level = max(self.off_farm_wage_level, np.random.uniform(1.0, 1.5))

    def step(self):
        """
        @theory
        农户年度决策周期 - 严格按照多层次博弈设计规范的阶段顺序执行
        @fix: 移除土地租赁市场参与调用，因为模型层面已经统一调用
        @new 2025-12: 添加农户规模更新和土地回收机制
        @new 2025-12: 添加连续变量特征更新
        @new 2025-12: 添加撂荒决策机制
        """
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
            if random.random() < abandonment_prob:
                patch.set_abandoned()
                # 记录撂荒事件
                if hasattr(self.model, 'abandonment_records'):
                    self.model.abandonment_records.append({
                        'step': self.model.step_count,
                        'farmer_id': self.unique_id,
                        'patch_id': id(patch),  # @fix: 使用id()作为地块标识
                        'area': patch.area,
                        'reason': 'low_return_opportunity_cost'
                    })
    
    def _calculate_abandonment_probability(self, patch, base_rate, max_rate, 
                                            low_quality_threshold, fragmentation_threshold):
        """
        @theory 计算地块撂荒概率
        
        影响因素权重：
        - 生计策略：Non_Farmer撂荒概率最高
        - 土地质量：低于阈值时增加撂荒概率
        - 细碎化程度：指数低时增加撂荒概率
        - 距离/位置：偏远地块更易撂荒（简化处理）
        - 年龄：老年户更易撂荒
        """
        prob = 0.0
        
        # @refactor 2025-12: 用连续变量替代离散类型判断
        # 不再使用: if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER
        # 而是基于农业依赖度的连续函数
        
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越低，撂荒概率越高（非农化程度高）
            strategy_prob = base_rate * (1.5 - agri_dep)  # agri_dep=0 → 1.5x, agri_dep=1 → 0.5x
        else:
            # 回退：使用离散判断
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_prob = base_rate * 1.5
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                strategy_prob = base_rate * 0.5
            else:
                strategy_prob = base_rate * 0.1
        
        prob += strategy_prob
        
        # 2. 土地质量影响
        if hasattr(patch, 'soil_quality') and patch.soil_quality < low_quality_threshold:
            quality_penalty = (low_quality_threshold - patch.soil_quality) / low_quality_threshold
            prob += base_rate * quality_penalty
        
        # 3. 细碎化程度影响
        if hasattr(patch, 'fragmentation_index') and patch.fragmentation_index < fragmentation_threshold:
            frag_penalty = (fragmentation_threshold - patch.fragmentation_index) / fragmentation_threshold
            prob += base_rate * frag_penalty * 0.5  # 细碎化影响权重0.5
        
        # 4. 地块面积影响（小地块更易撂荒）
        if patch.area < 1.0:  # 小于1亩
            prob += base_rate * 0.3
        elif patch.area < 3.0:  # 1-3亩
            prob += base_rate * 0.1
        
        # 5. 年龄影响（老年户更易撂荒）
        if self.age >= 65:
            prob += base_rate * 0.5
        elif self.age >= 55:
            prob += base_rate * 0.2
        
        # 6. 机会成本考虑（非农收入高时更易撂荒）
        off_farm_wage = self.model.environment.get_market_price('wage_rate')
        expected_agri_return = self._estimate_patch_return(patch)
        if expected_agri_return < off_farm_wage * 0.3:  # 农业收益低于务工30%
            prob += base_rate * 0.5
        
        # 限制在最大撂荒率以内
        return min(prob, max_rate)
    
    def _estimate_patch_return(self, patch):
        """
        @theory 估算地块预期年收益（每亩）
        用于撂荒决策的机会成本比较
        """
        # 基础产量
        base_yield = 500  # 假设平均500公斤/亩
        grain_price = self.model.environment.get_market_price('crop_price')
        
        # 考虑土地质量
        quality_factor = getattr(patch, 'soil_quality', 0.7)
        adjusted_yield = base_yield * quality_factor
        
        # 毛收入
        gross_income = adjusted_yield * grain_price
        
        # 成本（简化估算：400元/亩）
        cost = 400
        
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
        """
        # @refactor 2025-12: 用连续变量判断是否跳过作物选择
        # 不再使用: if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER
        skip_crop_choice = False
        if hasattr(self, 'characteristics') and self.characteristics:
            # 农业依赖度极低时（< 0.1），跳过作物选择
            if self.characteristics.agricultural_dependence < 0.1:
                skip_crop_choice = True
        else:
            # 回退：使用离散判断
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                skip_crop_choice = True
        
        if skip_crop_choice:
            # 不种地，但保持planned_crop有值以避免后续None错误
            if self.planned_crop is None:
                self.planned_crop = CropType.GRAIN
        
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
        if grain_utility > cash_crop_utility:
            self.planned_crop = CropType.GRAIN
        else:
            self.planned_crop = CropType.CASH_CROP
        
        # 阶段6：有限探索（防止系统锁定）
        # @fix: 探索概率降低到3%，企业级别的稳定性
        if random.random() < 0.03:
            self.planned_crop = CropType.CASH_CROP if self.planned_crop == CropType.GRAIN else CropType.GRAIN
        
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
        }
        
        self.crop_choice_history.append(decision_record)

    def _collect_neighbor_crop_outcomes(self):
        """
        @refactor 2025-12: 简化社会学习 - 仅收集邻居种植结果作为信息
        
        专家建议：社会学习应作为信息获取渠道，影响对概率的估计
        而非直接作为效用函数的加成项
        """
        outcomes = {'grain': [], 'cash_crop': []}
        
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor and hasattr(neighbor, 'crop_choice_history') and neighbor.crop_choice_history:
                last_record = neighbor.crop_choice_history[-1]
                choice = last_record.get('choice')
                # 获取邻居的收入情况
                income = getattr(neighbor, 'total_income', 0)
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
        price_stats = self.model.environment.get_price_statistics(
            'grain' if crop_type == 'grain' else 'cash_crop'
        )
        price_mean = price_stats['mean']
        price_std = price_stats['std']
        
        # 获取劳动力市场参数
        labor_config = self.model.config.get('labor_market', {}).get('labor_requirements', {})
        daily_wage = self.model.environment.get_market_price('wage_rate')
        
        # ================================================================
        # @fix 2025-12-01: 调整产量和成本参数，纳入劳动力成本差异
        # ================================================================
        operated_area = max(self.get_operated_land_area(), 1.0)
        
        if crop_type == 'grain':
            # 粮食作物：产量约500kg/亩（稳定）
            base_yield = 500 * self.farming_ability * operated_area
            
            # 物资成本率40%（种子、化肥、农药等）
            material_cost_ratio = 0.40
            
            # 劳动力成本：12天/亩（农户不完全机械化）
            labor_days_per_mu = labor_config.get('grain', {}).get('base', 12)
            labor_cost_per_mu = labor_days_per_mu * daily_wage
            
            # @new: 粮食的风险波动较小（政府托市）
            price_volatility_factor = 0.8  # 波动降低20%
        else:
            # 经济作物：产量约600kg/亩（算价值量）
            base_yield = 600 * self.farming_ability * operated_area
            
            # 物资成本率45%（需要更多农药、肥料等）
            material_cost_ratio = 0.45
            
            # 劳动力成本：30天/亩（劳动密集型）
            labor_days_per_mu = labor_config.get('cash_crop', {}).get('base', 30)
            labor_cost_per_mu = labor_days_per_mu * daily_wage
            
            # 经济作物风险更高
            price_volatility_factor = 1.2  # 波动增加20%
        
        # @fix 2025-12: 正确计算补贴金额（使用environment的标准接口）
        if crop_type == 'grain' and hasattr(self.model, 'environment'):
            subsidy_result = self.model.environment.calculate_subsidy_for_actor(operated_area, actor_type='farmer')
            grain_subsidy_amount = subsidy_result.get('cash', 0) + subsidy_result.get('in_kind_value', 0)
        else:
            grain_subsidy_amount = 0
        
        # 基础情景概率
        pessimistic_prob = 0.2
        baseline_prob = 0.6
        optimistic_prob = 0.2
        
        # 根据邻居结果调整概率（社会学习作为信息渠道）
        neighbor_data = neighbor_outcomes.get(crop_type, [])
        if neighbor_data:
            avg_income = np.mean(neighbor_data)
            expected_income = self.reference_point
            
            if avg_income > expected_income * 1.2:
                optimistic_prob += 0.1
                pessimistic_prob -= 0.1
            elif avg_income < expected_income * 0.8:
                # 邻居表现差 -> 悲观概率提高
                pessimistic_prob += 0.1
                optimistic_prob -= 0.1
        
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
        pessimistic_yield = base_yield * 0.8
        pessimistic_revenue = pessimistic_yield * pessimistic_price
        pessimistic_material_cost = pessimistic_revenue * material_cost_ratio
        pessimistic_labor_cost = labor_cost_per_mu * operated_area
        pessimistic_total_cost = pessimistic_material_cost + pessimistic_labor_cost
        pessimistic_subsidy = grain_subsidy_amount if crop_type == 'grain' else 0
        scenarios.append({
            'name': '悲观情景',
            'probability': pessimistic_prob,
            'profit': pessimistic_revenue - pessimistic_total_cost + pessimistic_subsidy
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
            'profit': baseline_revenue - baseline_total_cost + baseline_subsidy
        })
        
        # 乐观情景
        optimistic_price = price_mean + 1.5 * adjusted_price_std
        optimistic_yield = base_yield * 1.2
        optimistic_revenue = optimistic_yield * optimistic_price
        optimistic_material_cost = optimistic_revenue * material_cost_ratio
        optimistic_labor_cost = labor_cost_per_mu * operated_area
        optimistic_total_cost = optimistic_material_cost + optimistic_labor_cost
        optimistic_subsidy = grain_subsidy_amount if crop_type == 'grain' else 0
        scenarios.append({
            'name': '乐观情景',
            'probability': optimistic_prob,
            'profit': optimistic_revenue - optimistic_total_cost + optimistic_subsidy
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
            if neighbor and hasattr(neighbor, 'total_income'):
                social_incomes.append(neighbor.total_income)
        
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
            if hasattr(self, 'total_income') and 'choice' in last_choice:
                if last_choice['choice'] == CropType.GRAIN:
                    grain_evidence.append((self.total_income, 1.0))  # (收入, 权重)
                else:
                    cash_crop_evidence.append((self.total_income, 1.0))
        
        # 邻居的数据（通过社会学习获得）
        for neighbor_id in self.social_network:
            neighbor = self.model.get_agent_by_id(neighbor_id)
            if neighbor and hasattr(neighbor, 'crop_choice_history') and neighbor.crop_choice_history:
                last_choice = neighbor.crop_choice_history[-1]
                if hasattr(neighbor, 'total_income') and 'choice' in last_choice:
                    # 网络强度作为权重
                    weight = self.network_strength.get(neighbor_id, 0.5)
                    if last_choice['choice'] == CropType.GRAIN:
                        grain_evidence.append((neighbor.total_income, weight))
                    else:
                        cash_crop_evidence.append((neighbor.total_income, weight))
        
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
        """
        scenarios = []
        
        # @fix: 风险感知内生化，基于环境历史价格
        price_stats = self.model.environment.get_price_statistics('grain' if crop_type == 'grain' else 'cash_crop')
        price_mean = price_stats['mean']
        price_std = price_stats['std']
        operated_area = max(self.get_operated_land_area(), 1.0)
        if crop_type == 'grain':
            base_yield = 750 * self.farming_ability * operated_area  # @calibration: 根据审评建议，750斤/亩
            cost_ratio = 0.6
        else:
            base_yield = 1500 * self.farming_ability * operated_area  # @calibration: 根据审评建议，1500斤/亩（更能代表主要经济作物的平均水平）
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
                base_subsidy_per_mu = self.model.config.get('policy', {}).get('government', {}).get('base_subsidy_per_mu', 100)
                if not base_subsidy_per_mu:
                    # 从环境获取
                    if hasattr(self.model, 'environment') and self.model.environment.active_policy:
                        base_subsidy_per_mu = self.model.environment.active_policy.get('base_subsidy_per_mu', 100)
                    else:
                        base_subsidy_per_mu = 100
                subsidy = operated_area * base_subsidy_per_mu * subsidy_rate

            scenario['profit'] = (revenue - cost) + subsidy
        
        return scenarios

    def calculate_grain_expected_profit(self, grain_price_floor, grain_subsidies_rate, operated_area):
        """
        @theory
        计算粮食作物的预期利润
        @fix 校准：基于《全国农产品成本收益资料汇编2023》
        - 产量: 500kg/亩 (1000斤/亩, 与企业一致)
        - 成本率: 87% (接近实证92%: 1252/1362)
        - 2022年实证: 总产值1442元, 成本1252元, 净利润189元
        """
        # 基础产量（考虑耕作能力）
        operated_area = max(operated_area, 0.0)
        if operated_area == 0:
            return 0.0
        
        # 产量: 500kg/亩 (1000斤/亩), 与企业保持一致
        base_yield_kg = 500 * self.farming_ability * operated_area  # kg
        
        # 收入 (不包含补贴)
        expected_revenue = base_yield_kg * grain_price_floor
        
        # 成本: 87%成本率 (接近实证92%)
        # 实证: 2022年粮食总成本1252元(不含土地), 总产值1442元, 成本率1252/1442=87%
        cost_grain = expected_revenue * 0.87
        
        # 净利润 (不包含补贴)
        # 注：补贴是额外收益，不影响机会成本计算
        net_profit = expected_revenue - cost_grain
        
        return net_profit

    def calculate_cash_crop_expected_profit(self, market_price_cash_crop, operated_area):
        """
        @theory
        计算经济作物的预期利润
        @fix 校准: 与企业保持一致
        """
        # 基础产量（考虑耕作能力）
        operated_area = max(operated_area, 0.0)
        if operated_area == 0:
            return 0.0
        
        # 产量: 300kg/亩, 与企业保持一致
        base_yield_kg = 300 * self.farming_ability * operated_area  # kg
        
        # 预期收入
        expected_revenue = base_yield_kg * market_price_cash_crop
        
        # 生产成本: 80%成本率
        cost_cash_crop = expected_revenue * 0.80
        
        return expected_revenue - cost_cash_crop

    def _get_off_farm_income_ratio(self):
        if getattr(self, "total_income", 0) > 0:
            ratio = self.off_farm_income / self.total_income
            if ratio < 0.0:
                return 0.0
            if ratio > 1.0:
                return 1.0
            return ratio
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
            age_bonus = 0.25
        elif self.age >= 55:
            age_bonus = 0.15
        else:
            age_bonus = 0.0
        
        # @new 2025-12: 非农就业稳定性
        # 如果有稳定的非农收入（连续多年非农收入占比高），更愿意转出
        if hasattr(self, 'consecutive_high_non_farm_ratio') and self.consecutive_high_non_farm_ratio >= 3:
            stability_bonus = 0.20
        elif off_ratio > 0.7:
            stability_bonus = 0.10
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
        return float(max(0.0, min(1.0, prob)))

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
            
            if random.random() < success_prob:
                # 回收成功
                self._execute_land_reclaim(patch, lessee)

    def _execute_land_reclaim(self, patch, lessee):
        """
        @new 2025-12: 执行土地回收
        """
        # 清除流转状态
        patch.leased_to = None
        patch.rental_price = 0
        
        # 更新双方的流转列表
        if patch in self.leased_out_patches:
            self.leased_out_patches.remove(patch)
        
        if hasattr(lessee, 'leased_in_patches') and patch in lessee.leased_in_patches:
            lessee.leased_in_patches.remove(patch)
        if hasattr(lessee, 'leased_patches') and patch in lessee.leased_patches:
            lessee.leased_patches.remove(patch)
        
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
        
        # @refactor 2025-12: 用连续变量替代离散类型判断
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越低，转出门槛越低（modifier越小）
            modifier = 0.5 + 0.5 * agri_dep  # agri_dep=0 → 0.5, agri_dep=1 → 1.0
        else:
            # 回退：使用离散判断
            modifier = 1.0
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                modifier = transfer_cfg.get("non_farm_out_multiplier", 0.6)
            elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                modifier = transfer_cfg.get("pluriactive_out_multiplier", 0.85)
        
        if self.age >= transfer_cfg.get("elderly_age_threshold", 60):
            return True
        return willingness >= threshold * modifier

    def _should_transfer_in_locally(self, willingness=None):
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        threshold = float(transfer_cfg.get("threshold_in", 0.6))
        if willingness is None:
            willingness = self.get_transfer_in_willingness()
        
        # @refactor 2025-12: 用连续变量替代离散类型判断
        if hasattr(self, 'characteristics') and self.characteristics:
            agri_dep = self.characteristics.agricultural_dependence
            # 农业依赖度越高，转入门槛越低（modifier越小）
            modifier = 0.8 + 0.4 * (1 - agri_dep)  # agri_dep=1 → 0.8, agri_dep=0 → 1.2
        else:
            # 回退：使用离散判断
            modifier = 1.0
            if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                modifier = transfer_cfg.get("pure_farmer_in_multiplier", 0.9)
            elif self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                modifier = transfer_cfg.get("non_farm_in_multiplier", 1.2)
        
        if self.check_off_farm_opportunity() and self.off_farm_income > 0:
            modifier *= 1.1
        return willingness >= threshold * (1.0 / modifier if modifier > 0 else 1.0)

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
        patch.set_leased(lessee)
        patch.rental_price = rent_price
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
            'channel': 'network'
        })

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
        if random.random() > transfer_prob:
            return False
        base_price = float(transfer_cfg.get("base_rent_price", 650))
        price = mechanism.land_transfer.calculate_transfer_price(
            land_quality,
            getattr(lessee, 'farming_ability', 1.0),
            base_price
        )
        lessor._apply_local_transfer_transaction(patch, lessee, price)
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
        random.shuffle(neighbors)
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
            wage_rate = 120
            
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
        
        # 更新农业依赖度（基于收入结构）
        total_income = self.agricultural_income + self.off_farm_income
        if total_income > 0:
            new_agri_dep = self.agricultural_income / total_income
            # 平滑更新，避免剧烈波动
            self.characteristics.agricultural_dependence = (
                0.7 * self.characteristics.agricultural_dependence + 
                0.3 * new_agri_dep
            )
        
        # 更新老龄化率（基于年龄变化）
        self.characteristics.aging_rate = max(0, min(1, (self.age - 40) / 40))
        
        # 更新生计多样性指数
        income_sources = []
        if self.agricultural_income > 0:
            income_sources.append(self.agricultural_income)
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
        rand = np.random.random()
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
        self.age = np.random.randint(30, 40)
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
        self_cultivation_ratio = np.random.uniform(0.30, 0.50)
        
        # 记录部分接班状态
        self.partial_succession_ratio = self_cultivation_ratio
        self.is_weekend_farmer = True
        
        # 户主年龄重置（部分接班也是新生代）
        self.age = np.random.randint(30, 45)
        self.characteristics.household_head_age = self.age
        self.characteristics.generation_type = 3
        
        # 农业依赖度降低但不完全脱离
        self.characteristics.agricultural_dependence = np.random.uniform(0.30, 0.50)
        
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
        # @fix: 不要清空已流转的地块列表，而是从地块状态中恢复
        # 只清空本期的报价列表，保留已流转的地块状态
        self.leased_out_patches = [p for p in self.land_patches 
                                   if p.owner == self and p.leased_to is not None]
        self.leased_in_patches = []  # 租入地块需要重新计算
        self.reservation_prices = {}
        # 先在交易网络内完成熟人撮合，再进入中央市场
        self.run_local_land_transfer_round()
        
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
        # @critical 2025-01: 降低保留面积要求，促进土地流转
        # 实证：中国农村Non_Farmer家庭平均只保留0.1-0.2亩菜地
        min_retained_area = self.household_size * 0.1  # 每人0.1亩（仅菜地）

        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        elderly_threshold = transfer_cfg.get("elderly_age_threshold", 60)

        # @fix 2025-01: 老年农户应该更积极出租土地！
        is_elderly = self.age >= elderly_threshold
        elderly_rent_discount = 0.6 if is_elderly else 0.8  # 老年农户降价40%，其他降价20%
        
        # 计算当前自留土地面积（未流转的地块）
        current_retained_area = sum(
            patch.area for patch in self.land_patches 
            if patch.owner == self and patch.leased_to is None
        )
        
        # 跟踪供给地块数量（用于调试）
        supply_count = 0
        
        for patch in self.land_patches:
            # Non_Farmer应该积极出租所有土地
            if patch.owner == self and patch.leased_to is None:
                # @critical 2025-01: 放宽保留限制，只保留最基本的自给地
                effective_min_retained = min_retained_area * (0.3 if is_elderly else 0.5)
                if current_retained_area - patch.area < effective_min_retained:
                    # 这是最后一块地，保留但仍可出租（如果价格足够好）
                    pass  # 不再跳过！而是以更高价格出租
                
                min_acceptable_rent = self.calculate_min_acceptable_rent(patch)
                
                # @fix 2025-01: Non_Farmer大幅降低保留价格，鼓励流转
                # 理论：Non_Farmer的主要收入来自非农，土地对其是"鸡肋"
                min_acceptable_rent = min_acceptable_rent * elderly_rent_discount
                
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
        # @fix 2024-11-19: 计算最低土地保留面积（每人0.5亩，用于自给自足）
        min_retained_area = self.household_size * 0.5  # 每人0.5亩
        
        # 计算当前自留土地面积（未流转的地块）
        current_retained_area = sum(
            patch.area for patch in self.land_patches 
            if patch.owner == self and patch.leased_to is None
        )
        
        transfer_cfg = self.agent_config.get("agents", {}).get("farmer", {}).get("land_transfer", {})
        threshold_in = transfer_cfg.get("threshold_in", 0.6)
        threshold_out = transfer_cfg.get("threshold_out", 0.2)
        elderly_threshold = transfer_cfg.get("elderly_age_threshold", 60)
        current_operated_area = self.get_operated_land_area()
        target_area = getattr(self, "target_operated_area", self._compute_target_operated_area())
        willingness_in = self.get_transfer_in_willingness()
        willingness_out = self.get_transfer_out_willingness()
        base_threshold = 0.4
        global_profit_threshold = base_threshold
        if current_operated_area < target_area and willingness_in >= threshold_in:
            global_profit_threshold = max(0.1, base_threshold - 0.1)
        elif current_operated_area > target_area and willingness_out >= threshold_out:
            global_profit_threshold = min(0.8, base_threshold + 0.1)
        
        # @fix 2025-01: 老年农户应该倾向于出租土地，而不是被排除
        is_elderly = self.age >= elderly_threshold
        
        for patch in self.land_patches:
            # @fix: 只对未流转的地块（leased_to is None）参与市场
            if patch.owner == self and patch.leased_to is None:
                # @fix 2025-01: 老年农户直接作为供给方，不需要计算收益比较
                if is_elderly:
                    # 老年农户降低保留面积要求（0.5→0.25），更愿意流转
                    effective_min_retained = min_retained_area * 0.5
                    if current_retained_area - patch.area < effective_min_retained:
                        continue
                    self.act_as_land_supplier_for_patch(patch, rent_discount=0.7)  # 降价30%
                    continue
                
                # @fix 2024-11-19: 如果流转这块地后，剩余土地低于最低保留面积，不允许流转
                if current_retained_area - patch.area < min_retained_area:
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
                    self.act_as_land_supplier_for_patch(patch)
                else:
                    # 选择自耕，保留土地
                    pass
        
        # @new 2025-01: Pure_Farmer和部分Pluriactive主动寻找可租入土地
        # @fix 2025-11-29: 降低门槛，增加农户间流转
        # Pure_Farmer总是积极扩张；Pluriactive有一定意愿时也参与
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户总是尝试扩张
            self._seek_land_to_rent_in(target_area, current_operated_area)
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业户：意愿较高或经营面积较小时参与
            if willingness_in >= threshold_in * 0.7 or current_operated_area < target_area * 0.5:
                self._seek_land_to_rent_in(target_area, current_operated_area)

    def calculate_min_acceptable_rent(self, patch):
        """
        @theory 保留价 = 机会成本（基于经济学定义）
        参考实证数据：2022年全国平均土地成本280元/亩
        
        机会成本 = max(自耕净收益 + 补贴, 外出打工净收益)
        
        关键：农户流转土地后会失去补贴（补贴给了实际种地者），
        所以保留价必须包含补贴损失！
        
        注：不是两者相加！机会成本是放弃的最佳选择
        """
        # 1. 自耕净收益（每亩）= 利润 + 补贴
        self_cultivation_profit_per_mu = self.calculate_self_cultivation_profit(patch) / max(patch.area, 0.01)
        
        # 补贴（如果种粮食）
        subsidy_per_mu = 0
        if self.planned_crop == CropType.GRAIN:
            # 获取每亩补贴金额
            subsidy_per_mu = self.model.environment.get_subsidy_for_farmer(self, 1.0)  # 1亩的补贴
        
        # 自耕总收益（利润 + 补贴）
        self_cultivation_total_per_mu = self_cultivation_profit_per_mu + subsidy_per_mu
        
        # 4. 保留价 = 自耕总收益（土地的机会成本）
        # @fix 移除务工收益的错误计算
        # 原错误：opportunity_cost = max(自耕收益, 务工收益/面积)
        # 问题：务工收益不应按面积分摊，导致保留价虚高14倍！
        # 正确：保留价只反映这块地自己种能获得多少
        opportunity_cost_per_mu = self_cultivation_total_per_mu
        
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
        # 出租收入（基于土地面积和质量估算价值）
        land_value = patch.area * patch.soil_quality * 1000  # 简化的土地价值计算
        rental_income = land_value * 0.1
        
        # 务工收入
        wage_income = self.off_farm_wage_level * 30 * 12  # 全年务工收入
        
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

    def calculate_max_willing_to_pay_rent(self, patch):
        """
        @theory
        计算最高愿付租金（基于边际利润）
        """
        # 在新增土地上种植planned_crop的预期边际利润
        if self.planned_crop == CropType.GRAIN:
            # @fix: 决策时正确计入补贴
            marginal_profit = self.calculate_grain_expected_profit(
                self.model.environment.get_market_price('crop_price'),
                self.model.environment.get_policy_param('grain_subsidies_rate'),
                patch.area
            )
        elif self.planned_crop == CropType.CASH_CROP:
            # @fix: 使用 environment 获取动态市场价格
            marginal_profit = self.calculate_cash_crop_expected_profit(
                self.model.environment.get_market_price('cash_crop'),
                patch.area
            )
        else:
            marginal_profit = 0
        
        return marginal_profit

    def act_as_land_supplier_for_patch(self, patch, rent_discount=1.0):
        """
        @theory
        作为土地供给方（针对特定地块）
        @param rent_discount: 租金折扣系数，用于老年农户降价（默认1.0无折扣）
        """
        min_acceptable_rent = self.calculate_min_acceptable_rent(patch)
        # 应用折扣
        min_acceptable_rent = min_acceptable_rent * rent_discount
        self.reservation_prices[patch] = min_acceptable_rent
        
        # 提交供给意愿
        self.model.land_clearing_house.add_supply_offer(
            patch, self, min_acceptable_rent
        )

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
        # 计算需要租入的面积
        need_area = target_area - current_area
        if need_area <= 0:
            return
        
        # 检查资金能力
        available_capital = self.capital * 0.3  # 最多用30%资金支付租金
        
        # 获取市场上可用的地块
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
        
        for patch in available_patches[:max_patches]:
            if rented_area >= need_area:
                break
            
            # 计算最高愿付租金
            max_rent = self.calculate_max_willing_to_pay_rent(patch)
            if max_rent <= 0:
                continue
            
            # 考虑资金约束
            remaining_capital = available_capital - (rented_area * 300)  # 假设平均租金300元/亩
            if remaining_capital <= 0:
                break
            
            max_affordable = remaining_capital / max(patch.area, 0.1) * patch.area
            bid_price = min(max_rent, max_affordable)
            
            if bid_price > 100 * patch.area:  # 至少100元/亩才出价
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
        base_wage = self.model.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 15.0)
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
                base_rental = self.model.config.get('environment', {}).get('market', {}).get('base_rental_price', 800)
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
        # 从土地流转市场获取平均租金
        if hasattr(self.model, 'land_clearing_house'):
            # 尝试从市场获取平均价格
            return self.model.config.get('environment', {}).get('market', {}).get('base_rental_price', 800)
        return 800.0  # 默认值

    def make_production_decisions(self):
        """
        @theory
        生产决策
        @improvement: P1优先级 - 产量计算精细化
        考虑土壤质量、耕作能力、技术水平、气候冲击等多因素
        参考：APSIM模型、农业生产效率研究
        """
        # @fix: Non_Farmer不应该有预期产量，因为土地都流转了
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            # Non_Farmer把土地都流转了，不从事农业生产
            self.expected_yield = 0
        elif self.planned_crop and self.get_operated_land_area() > 0:
            operated_area = self.get_operated_land_area()
            
            # 使用精细化产量计算
            self.expected_yield = self._calculate_detailed_yield(self.planned_crop, operated_area)
        else:
            self.expected_yield = 0
    
    def _calculate_detailed_yield(self, crop_type, area):
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
        
        # 1. 基础产量（按作物类型）
        if crop_type == CropType.GRAIN:
            base_yield_per_mu = base_yields.get('grain', 750)
        else:  # CASH_CROP
            base_yield_per_mu = base_yields.get('cash_crop', 1500)
        
        base_yield = base_yield_per_mu * area
        
        # ================================================================
        # @new 2025-12-01: 技术进步因子
        # 参考：中国粮食亩产年均增长约1.3%（2015-2022）
        # ================================================================
        annual_growth = tech_progress.get('annual_yield_growth_rate', 0.015)
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
        ability_effect = soil_effect * ability_multiplier
        
        # 4. 技术水平影响（从地块或环境获取）
        tech_config = yield_factors.get('technology', {})
        base_tech_level = tech_config.get('base_level', 1.0)
        max_improvement = tech_config.get('max_improvement', 0.2)
        
        tech_level = self._get_technology_level()
        tech_multiplier = 1.0 + max_improvement * (tech_level - base_tech_level)
        tech_effect = ability_effect * tech_multiplier
        
        # 5. 气候冲击
        climate_config = yield_factors.get('climate', {})
        climate_volatility = climate_config.get('volatility', 0.1)
        climate_mean = climate_config.get('mean', 0.0)
        
        climate_shock = self._get_climate_shock(climate_mean, climate_volatility)
        climate_effect = tech_effect * (1.0 + climate_shock)
        
        # 6. 随机波动
        noise_std = random_noise.get('std', 0.1)
        random_factor = np.random.normal(1.0, noise_std)
        final_yield = climate_effect * random_factor
        
        return max(0, final_yield)
    
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
        """获取气候冲击"""
        # 如果有环境模块提供气候数据，从那里获取
        # 否则，使用随机生成
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_climate_shock'):
            return self.model.environment.get_climate_shock()
        else:
            return np.random.normal(mean, volatility)
    
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
            self.loan_interest_rate = np.random.uniform(0.06, 0.08)
            self.max_leverage_ratio = 0.4  # 最大负债率40%
        elif operated_area >= 15:  # 中等规模
            self.loan_interest_rate = np.random.uniform(0.07, 0.10)
            self.max_leverage_ratio = 0.3  # 最大负债率30%
        else:  # 小农户
            self.loan_interest_rate = np.random.uniform(0.08, 0.15)
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
        if random.random() > self.credit_accessibility:
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
        available_cash = max(0, self.capital * 0.3)
        
        # 估算可获得贷款
        land_value = self.get_operated_land_area() * 300 * 10
        total_assets = self.capital + land_value
        potential_loan = max(0, total_assets * self.max_leverage_ratio - self.debt_balance)
        
        # 考虑信贷可得性折扣
        expected_loan = potential_loan * self.credit_accessibility
        
        return available_cash + expected_loan
    
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
        if old_scale != self.farmer_scale:
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
            lambda_val = np.random.normal(lambda_dist.get('mean', 3.0), lambda_dist.get('std', 0.5))
        else:
            lambda_val = np.random.normal(3.0, 0.5)
        
        if curvature_dist.get('type') == 'normal':
            curvature_val = np.random.normal(curvature_dist.get('mean', 0.55), curvature_dist.get('std', 0.1))
        else:
            curvature_val = np.random.normal(0.55, 0.1)
        
        if prob_weight_dist.get('type') == 'normal':
            prob_weight_val = np.random.normal(prob_weight_dist.get('mean', 0.7), prob_weight_dist.get('std', 0.05))
        else:
            prob_weight_val = np.random.normal(0.7, 0.05)
        
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
        
        if not by_income or not hasattr(self, 'total_income') or self.total_income <= 0:
            return  # 如果没有配置或没有收入数据，不调整
        
        # 计算收入分位数（需要所有农户的收入数据）
        # 简化：基于历史收入记忆或当前收入估算分位数
        if hasattr(self, 'income_memory') and len(self.income_memory) > 0:
            avg_income = np.mean(self.income_memory)
        else:
            avg_income = self.total_income
        
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

    def harvest_and_settle(self):
        """
        @theory
        收获与收入结算
        新增：更新收入记忆以支持前景理论的动态参考点机制
        """
        # 计算实际产量（加入随机波动）
        if self.expected_yield > 0:
            actual_yield = self.expected_yield * np.random.normal(1.0, 0.1)
        else:
            actual_yield = 0
        
        # @fix: 记录本期的实际产出和作物选择，供下一期市场更新使用
        self.last_yield = actual_yield
        self.last_crop_choice = self.planned_crop
        
        # @fix: Non_Farmer的收入完全来源于非农收入
        # 但可能有土地情节，保留少量土地用于自给自足小农生产，满足自身必须食物消费
        # 所以Non_Farmer可以有少量农业收入（自给自足），但主要收入是非农收入
        if self.planned_crop == CropType.GRAIN:
            self.agricultural_income = actual_yield * self.model.environment.get_market_price('crop_price')
        elif self.planned_crop == CropType.CASH_CROP:
            self.agricultural_income = actual_yield * self.model.environment.get_market_price('cash_crop')
        else:
            # @fix: 如果planned_crop为None，确保有默认值，避免收入为0
            if self.planned_crop is None:
                # 默认选择粮食作物，确保有收入
                self.planned_crop = CropType.GRAIN
                if actual_yield > 0:
                    self.agricultural_income = actual_yield * self.model.environment.get_market_price('crop_price')
                else:
                    self.agricultural_income = 0
            else:
                self.agricultural_income = 0
        
        # 计算其他收入
        self.calculate_land_rental_income()
        self.calculate_subsidy_income() # @fix: 移到此处，在结算时计算
        self.calculate_off_farm_income()
        
        # 计算支出
        self.calculate_production_cost()
        self.calculate_land_rental_expense()
        
        # 更新总收入（未减去支出）
        self.gross_income = (self.agricultural_income + self.land_rental_income + 
                           self.subsidy_income + self.off_farm_income)
        self.total_income = self.gross_income  # 向后兼容
        self.total_expense = self.production_cost + self.land_rental_expense
        
        # 计算净利润
        self.net_income = self.gross_income - self.total_expense
        
        # 计算可支配收入（假设储蓄率为20%）
        savings_rate = 0.2
        self.disposable_income = self.net_income * (1 - savings_rate)
        
        # 更新资本
        # @fix: 确保资本不会变成极端负值，限制在合理范围内
        self.capital += self.net_income
        
        # 计算消费和效用（新增）
        self.calculate_consumption_and_utility()
        # 限制资本在合理范围内（避免极端值）
        self.capital = max(-50000, min(self.capital, 10000000))  # 限制在-5万到1000万之间
        
        # 新增：更新收入记忆以支持前景理论动态参考点机制
        self.update_income_memory()
    
    def update_income_memory(self):
        """
        @theory
        更新收入记忆，支持前景理论的动态参考点机制
        保持有限记忆长度，确保参考点的动态更新
        """
        # 将本期总收入加入记忆
        self.income_memory.append(self.total_income)
        
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
        
        for neighbor, score in selected_neighbors:
            self.social_network.append(neighbor.unique_id)
            # 网络强度基于综合得分，归一化到[0.1, 1.0]
            self.network_strength[neighbor.unique_id] = min(1.0, max(0.1, score))
        
        # 交易可行性网络：优先选择得分最高、关系稳固的邻居
        if selected_neighbors:
            transaction_subset = selected_neighbors[:min(transaction_degree_cfg, len(selected_neighbors))]
            for neighbor, score in transaction_subset:
                self.transaction_neighbors.append(neighbor.unique_id)
                # 交易网络强调互信，保证最低强度阈值
                self.transaction_link_strength[neighbor.unique_id] = min(1.0, max(0.2, score))
    
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
        return np.random.uniform(1, 10)
    
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
            core_income = self.agricultural_income + self.off_farm_income
            if core_income > 0:
                agri_dep = self.agricultural_income / core_income
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
        pure_farmer_threshold = evolution_cfg.get('pure_farmer_threshold', 0.80)
        non_farmer_threshold = evolution_cfg.get('non_farmer_threshold', 0.30)
        cooldown_steps = evolution_cfg.get('cooldown_steps', 1)
        
        # 城镇化结构趋势参数 - @fix 2025-12: 大幅降低流出率，防止纯农户过快减少
        # 原设置base_outflow_prob=0.12导致每步12%流出，20步后纯农户几乎消失
        # 新设置：每步2%流出，符合中国年化劳动力转移约3%的现实
        urbanization_cfg = evolution_cfg.get('urbanization_trend', {})
        base_outflow_prob = urbanization_cfg.get('base_outflow_prob', 0.02)  # @fix: 从0.12降到0.02
        young_multiplier = urbanization_cfg.get('young_multiplier', 1.5)    # @fix: 从2.0降到1.5
        elderly_exit_prob = urbanization_cfg.get('elderly_exit_prob', 0.05) # @fix: 从0.20降到0.05
        
        # @critical 2025-12: 检查代际接替事件
        self._check_generational_succession()
        
        # 更新历史记录
        self.update_history()
        
        # 计算农业收入占比
        core_income = self.agricultural_income + self.off_farm_income
        if core_income > 0:
            agricultural_income_ratio = self.agricultural_income / core_income
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
        
        # 检查切换冷却期
        if self.strategy_switch_cooldown > 0:
            self.strategy_switch_cooldown -= 1
            return
        
        old_strategy = self.livelihood_strategy
        new_strategy = old_strategy
        transition_reason = None
        
        # 获取年龄
        age = getattr(self, 'age', 45)
        operated_area = self.get_operated_land_area()
        
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
        
        # Pure_Farmer → Pluriactive（城镇化吸引）
        if old_strategy == LivelihoodStrategy.PURE_FARMER:
            if random.random() < outflow_prob:
                new_strategy = LivelihoodStrategy.PLURIACTIVE
                transition_reason = "城镇化吸引"
        
        # Pluriactive → Non_Farmer（非农收入稳定后完全转型）
        elif old_strategy == LivelihoodStrategy.PLURIACTIVE:
            if self.off_farm_income > 15000 and agricultural_income_ratio < 0.40:
                if random.random() < outflow_prob * 1.2:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "非农收入稳定"
        
        # ========== 结构趋势2: 老年退出 ==========
        if age > 60:
            if old_strategy == LivelihoodStrategy.PURE_FARMER:
                if random.random() < elderly_exit_prob:
                    new_strategy = LivelihoodStrategy.PLURIACTIVE
                    transition_reason = "年龄退出"
            elif old_strategy == LivelihoodStrategy.PLURIACTIVE:
                if random.random() < elderly_exit_prob * 0.5:
                    new_strategy = LivelihoodStrategy.NON_FARMER
                    transition_reason = "老年完全退出"
        
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
            if (agricultural_income_ratio > 0.95 and 
                operated_area >= 50 and 
                self.off_farm_income < 2000 and
                35 <= age <= 55):
                if random.random() < 0.02:  # 仅2%概率（从5%降低）
                    new_strategy = LivelihoodStrategy.PURE_FARMER
                    transition_reason = "规模化专业化"
        
        # ========== 收入驱动的策略调整（补充机制） ==========
        if new_strategy == old_strategy:
            # 非农收入占主导时转为Non_Farmer
            if agricultural_income_ratio < non_farmer_threshold:
                if self.off_farm_income >= 12000:
                    if random.random() < 0.20:
                        new_strategy = LivelihoodStrategy.NON_FARMER
                        transition_reason = "非农主导"
            
            # 中间状态趋向兼业（从Pure_Farmer流出）
            elif old_strategy == LivelihoodStrategy.PURE_FARMER:
                if agricultural_income_ratio < 0.70:
                    if random.random() < 0.15:
                        new_strategy = LivelihoodStrategy.PLURIACTIVE
                        transition_reason = "收入分散"
        
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
                    if agricultural_income_ratio < 0.50 and random.random() < 0.08:
                        if old_strategy == LivelihoodStrategy.PURE_FARMER:
                            new_strategy = LivelihoodStrategy.PLURIACTIVE
                        else:
                            new_strategy = LivelihoodStrategy.NON_FARMER
                        transition_reason = "邻居影响"
        
        # ========== 执行策略切换 ==========
        if old_strategy != new_strategy:
            self.livelihood_strategy = new_strategy
            self.strategy_switch_cooldown = cooldown_steps
            print(f"[生计演化] 农户{self.unique_id}: {old_strategy.value}→{new_strategy.value}, "
                  f"农业占比:{agricultural_income_ratio:.2f}, 年龄:{age}, 原因:{transition_reason}")
            
            # @new 2025-11-30: 记录到墓地系统 - 如果转为Non_Farmer
            if new_strategy == LivelihoodStrategy.NON_FARMER:
                self._record_exit_to_graveyard(old_strategy, transition_reason, age, agricultural_income_ratio)
        
        # @critical 2025-12: 最终更新 - 同步连续变量到 characteristics
        if hasattr(self, 'characteristics') and self.characteristics is not None:
            self.characteristics.agricultural_dependence = agricultural_income_ratio
            from modules.farmer_characteristics import FarmerTypeClassifier
            self.farmer_type, self.farmer_type_confidence = FarmerTypeClassifier.classify(self.characteristics)
            sensitivity = FarmerTypeClassifier.get_type_policy_sensitivity(self.farmer_type)
            self.characteristics.price_subsidy_sensitivity = sensitivity.get('price_subsidy', 0.5)
            self.characteristics.direct_subsidy_sensitivity = sensitivity.get('direct_subsidy', 0.5)
            self.characteristics.rental_subsidy_sensitivity = sensitivity.get('rental_subsidy', 0.5)
            self.characteristics.service_policy_sensitivity = sensitivity.get('service', 0.5)
    
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
        return random.random() < shock_prob
    
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
        self.income_history.append(self.total_income)
        if len(self.income_history) > 5:
            self.income_history.pop(0)
        
        # 更新农业收入占比历史
        # @fix: 使用与livelihood_strategy_evolution()一致的计算方式（只考虑农业收入和非农收入）
        core_income = self.agricultural_income + self.off_farm_income
        if core_income > 0:
            agricultural_income_ratio = self.agricultural_income / core_income
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
        @fix: 使用新的阈值（30%/70%），与CLES数据一致
        """
        if agricultural_income_ratio < 0.30:  # Non_Farmer：农业收入占比 < 30%
            self.consecutive_high_non_farm_ratio += 1
            self.consecutive_low_agricultural_ratio = 0
        elif agricultural_income_ratio > 0.70:  # Pure_Farmer：农业收入占比 > 70%
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
        # @fix: 使用新的阈值（30%/70%），与CLES数据一致
        if agricultural_income_ratio < 0.30 and self.consecutive_high_non_farm_ratio >= 2:  # Non_Farmer：农业收入占比 < 30%
            # 连续2年非农收入占比高，向Non_Farmer转换
            return 2.0
        elif agricultural_income_ratio > 0.70 and self.consecutive_low_agricultural_ratio >= 2:  # Pure_Farmer：农业收入占比 > 70%
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
        @fix: 使用新的阈值（30%/70%），与CLES数据一致
        """
        if agricultural_income_ratio < 0.30:  # Non_Farmer：农业收入占比 < 30%
            return LivelihoodStrategy.NON_FARMER
        elif agricultural_income_ratio > 0.70:  # Pure_Farmer：农业收入占比 > 70%
            return LivelihoodStrategy.PURE_FARMER
        else:  # Pluriactive：农业收入占比 30%-70%
            return LivelihoodStrategy.PLURIACTIVE

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
        leased_patches = [p for p in self.land_patches 
                         if p.owner == self and p.leased_to is not None]
        
        # 更新leased_out_patches以保持一致性
        self.leased_out_patches = leased_patches
        
        # @fix 2025-11-29: 修正租金计算
        # rental_price已经是该地块的总租金（已包含面积），无需再乘面积
        self.land_rental_income = sum(
            patch.rental_price 
            for patch in leased_patches 
            if patch.rental_price > 0
        )

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
        if getattr(self.model.environment, "manual_subsidy_dispatch", False):
            # 政府代理已在外部发放补贴，这里仅保留现有数值
            return

        # @fix: 重构为按面积补贴，在结算时计算
        self.subsidy_income = 0
        self.scale_subsidy_income = 0  # @new: 规模补贴收入
        self.scale_subsidy_qualified = False  # @new: 是否达到规模补贴门槛
        
        if self.planned_crop == CropType.GRAIN:
            # 简化：假设所有经营面积都种了计划作物
            grain_area = self.get_operated_land_area() 
            if grain_area > 0:
                # 使用新的综合补贴计算接口
                subsidy_info = self.model.environment.get_total_subsidy_for_farmer(self, grain_area)
                self.subsidy_income = subsidy_info['total']
                self.scale_subsidy_income = subsidy_info['scale_subsidy']
                self.scale_subsidy_qualified = subsidy_info['scale_qualified']

    def calculate_off_farm_income(self):
        """
        @theory
        计算非农务工收入
        @fix: 修正非农收入计算，应该乘以实际日工资
        """
        # 获取市场日工资（从环境获取，如果没有则使用配置中的基础工资）
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_market_price'):
            daily_wage = self.model.environment.get_market_price('wage_rate')
        else:
            # 从配置获取基础工资
            daily_wage = self.model.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 120.0)
        
        effective_labor = self.off_farm_labor
        effective_wage = daily_wage
        if self.ai_enabled:
            # AI冲击：削弱非农岗位与工资，兼顾不同生计策略的适应能力
            strategy_modifier = 1.0
            if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                strategy_modifier = max(0.2, 1 - self.ai_adoption_lag)
            labor_factor = max(0.0, 1 - self.ai_non_farm_job_shock * strategy_modifier)
            wage_factor = max(0.2, 1 - self.ai_wage_penalty * strategy_modifier)
            effective_labor *= labor_factor
            effective_wage *= wage_factor
        
        # @fix 2024-11-18: 根据自耕土地面积调整实际可用非农劳动时间
        # 理论：兼业农户自耕土地越多，农忙期越长，实际可用于外出务工的时间越少
        self_operated_area = self.get_operated_land_area()
        
        # 计算农忙期时间损失（春耕+秋收+田间管理）
        # 假设每亩地需要15天集中劳动（不能外出务工）
        agri_busy_days = self_operated_area * 15
        
        # 调整后的实际可用非农劳动天数
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户：农忙期很长，实际外出务工时间有限
            max_off_farm_days = min(effective_labor, max(0, self.total_labor_units * 0.4 - agri_busy_days))
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业农户：农忙期占用部分时间，剩余时间可外出务工
            # 实际可用非农天数 = 非农劳动分配天数 - 农忙期损失
            actual_available_days = max(0, self.total_labor_units - agri_busy_days)
            max_off_farm_days = min(effective_labor, actual_available_days * 0.85)  # 85%的剩余时间可用于务工
        else:  # NON_FARMER
            # 非农户：流转土地，几乎全年可外出务工
            max_off_farm_days = effective_labor
        
        # 使用调整后的实际可用天数计算非农收入
        actual_off_farm_labor = min(effective_labor, max_off_farm_days)
        
        # 非农收入 = 实际可用天数 × 工资水平系数 × 有效日工资
        self.off_farm_income = actual_off_farm_labor * self.off_farm_wage_level * effective_wage
        
        # @fix: 确保Non_Farmer有最低非农收入，避免总收入为0
        if self.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
            # Non_Farmer应该主要依靠非农收入，如果非农收入为0，给予最低保障
            if self.off_farm_income <= 0 and effective_labor > 0:
                # 如果有效劳动天数>0但收入为0，可能是工资为0，给予最低保障
                self.off_farm_income = max(self.off_farm_income, daily_wage * 0.1 * effective_labor)
            elif self.off_farm_income <= 0:
                # 如果连劳动天数都为0，给予最低保障收入（相当于每月工作几天）
                min_work_days = 30  # 最低工作30天
                self.off_farm_income = min_work_days * self.off_farm_wage_level * daily_wage

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
        
        if not use_detailed:
            # 使用简单比例（向后兼容）
            if self.planned_crop == CropType.GRAIN:
                self.production_cost = self.agricultural_income * 0.6
            elif self.planned_crop == CropType.CASH_CROP:
                self.production_cost = self.agricultural_income * 0.4
            else:
                self.production_cost = 0
        else:
            # 使用详细成本计算
            operated_area = self.get_operated_land_area()
            if self.planned_crop and operated_area > 0:
                self.production_cost = self._calculate_detailed_cost(self.planned_crop, operated_area)
            else:
                self.production_cost = 0
        
        # 实物支持抵消
        if self.in_kind_support_balance > 0 and self.production_cost > 0:
            offset = min(self.production_cost, self.in_kind_support_balance)
            self.production_cost -= offset
            self.in_kind_support_balance -= offset
    
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
            base_wage = self.model.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 120.0)
            labor_price = base_wage * 0.7
        
        # 2. 获取投入品需求量
        requirements = input_requirements.get('seed', {})
        if crop_type == CropType.GRAIN:
            seed_required = requirements.get('grain', 30)  # 斤/亩
            fertilizer_required = input_requirements.get('fertilizer', {}).get('grain', 100)  # 斤/亩
            pesticide_required = input_requirements.get('pesticide', {}).get('grain', 2)  # 斤/亩
            # 农户不使用机械，劳动力需求更高
            labor_days_required = input_requirements.get('labor_days', {}).get('grain_farmer', 12)  # 天/亩（比企业高）
            tools_usage = input_requirements.get('small_tools', {}).get('grain', 1)  # 次/亩
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('grain', 200)  # 元/亩
            seed_price = seed_prices.get('grain', 5.0)
        else:  # CASH_CROP
            seed_required = requirements.get('cash_crop', 5)  # 斤/亩
            fertilizer_required = input_requirements.get('fertilizer', {}).get('cash_crop', 150)  # 斤/亩
            pesticide_required = input_requirements.get('pesticide', {}).get('cash_crop', 5)  # 斤/亩
            # 经济作物需要更多人工
            labor_days_required = input_requirements.get('labor_days', {}).get('cash_crop_farmer', 30)  # 天/亩
            tools_usage = input_requirements.get('small_tools', {}).get('cash_crop', 2)  # 次/亩
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('cash_crop', 300)  # 元/亩
            seed_price = seed_prices.get('cash_crop', 8.0)
        
        # 3. 劳动力约束检查
        total_labor_need = labor_days_required * area
        available_family_labor = self.total_labor_units * 0.9  # 90%用于农业
        
        # 根据生计策略调整可用劳动力
        if self.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
            available_family_labor = self.total_labor_units * 0.85  # 纯农户85%劳动力可用于自家农业
        elif self.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
            available_family_labor = self.total_labor_units * 0.5   # 兼业户50%
        else:  # NON_FARMER
            available_family_labor = self.total_labor_units * 0.15  # 非农户15%
        
        # 记录家庭劳动力使用
        self.family_labor_used = min(total_labor_need, available_family_labor)
        
        # 4. 计算劳动力缺口和雇佣成本
        labor_gap = max(0, total_labor_need - available_family_labor)
        
        if labor_gap > 0:
            self.labor_constraint_binding = True
            # 需要雇佣外部劳动力（大户情况）
            # 雇佣成本 = 缺口天数 × 市场工资（已在劳动力市场结算）
            # 这里只记录需求，实际雇佣在劳动力市场模块完成
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
        if hasattr(self, 'hired_labor_cost') and self.hired_labor_cost > 0:
            hired_labor_cost = self.hired_labor_cost
        else:
            # 如果劳动力市场尚未结算，估算雇佣成本
            hired_labor_cost = self.hired_labor_days * labor_price
        
        # 农户使用小型农具而非大型机械
        tools_cost = small_tools_price * tools_usage * area
        other_costs = other_cost * area
        
        # 6. 总成本（包含隐性和显性劳动力成本）
        total_cost = (seed_cost + fertilizer_cost + pesticide_cost + 
                     family_labor_cost + hired_labor_cost + tools_cost + other_costs)
        
        return total_cost

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

    def get_operated_land_area(self):
        """
        @theory
        获取经营土地面积
        @fix: 正确计算经营面积，排除已流转的地块，包括租入的地块
        """
        operated_area = 0
        
        # @fix: 从地块状态中恢复已流转的地块列表
        leased_out = [p for p in self.land_patches 
                     if p.owner == self and p.leased_to is not None]
        self.leased_out_patches = leased_out
        
        # 自有且未流转的地块
        for patch in self.land_patches:
            if patch.owner == self and patch.leased_to is None:
                operated_area += patch.area
        
        # 租入的地块（从地块状态中恢复）
        leased_in = [p for p in self.model.land_patches 
                    if p.leased_to == self]
        self.leased_in_patches = leased_in
        
        for patch in leased_in:
            operated_area += patch.area
            
        return operated_area
    
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
            base_wage = self.model.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 120.0)
            
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
            'gross_income': self.gross_income,
            'total_income': self.total_income,
            'net_income': self.net_income,
            'disposable_income': self.disposable_income
        }
    
    def calculate_consumption_and_utility(self):
        """
        @theory
        计算消费和效用
        基于理论框架：U = U_food(C_essential, C_luxury) + U_leisure(L_leisure)
        """
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
        # 获取当前粮食价格
        grain_price = self.model.environment.get_market_price('crop_price')
        
        # @critical 2025-12: 计算自产粮食用于自给
        # 只有种粮的农户才有自给粮
        self_produced_grain_kg = 0
        if hasattr(self, 'last_crop_choice') and self.last_crop_choice == CropType.GRAIN:
            if hasattr(self, 'last_yield') and self.last_yield > 0:
                # last_yield是元，需要转换为kg
                # 假设粮价是元/斤，1斤=0.5kg
                grain_price_per_kg = grain_price * 2 if grain_price > 0 else 2.5
                total_grain_kg = self.last_yield / grain_price_per_kg
                
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
            labor_days_worked=labor_days
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
        current_capital = float(np.real(self.capital)) if isinstance(self.capital, complex) else float(self.capital)
        surplus = float(np.real(self.consumer_surplus)) if isinstance(self.consumer_surplus, complex) else float(self.consumer_surplus)
        self.capital = current_capital + surplus