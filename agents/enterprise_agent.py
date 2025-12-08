# -*- coding: utf-8 -*-
"""
@theory
AgriculturalEnterpriseAgent - 基于Melitz（2003）企业异质性模型的农业企业主体
核心特征：企业生产率异质性分布，基于真实数据校准的分层决策机制
理论支撑：
1. Melitz (2003): 企业异质性与贸易理论
2. Deininger & Jin (2005): 土地市场中的企业异质性
3. 《新型农业经营主体发展报告2023》: 中国农业企业分类与特征

@refactor 2025-12: 重构主体类型
- 移除FAMILY_FARM_LARGE类型，家庭农场归并到FarmerAgent中
- 企业仅包括农业公司（工商资本下乡）和合作社两种类型
- 这更符合中国农村现实：家庭农场本质上是"大农户"而非"企业"
"""

from mesa import Agent
import numpy as np
import random
from enum import Enum
import scipy.optimize as opt

# @new 2025-12: 预期机制集成
from modules.expectation_mechanism import ExpectationFactory, ExpectationManager

class EnterpriseType(Enum):
    """
    农业企业类型枚举
    基于《新型农业经营主体发展报告2023》和农业农村部分类标准
    
    @refactor 2025-12: 
    - 移除FAMILY_FARM_LARGE（归并到FarmerAgent）
    - 企业仅保留工商资本型农业公司和农民合作社两种
    
    理论依据：
    1. 家庭农场以家庭劳动力为主，本质是"大农户"而非"企业"
    2. 农业公司代表工商资本下乡，资本密集、雇工经营
    3. 合作社代表农户联合体，集体决策、风险共担
    """
    AGRI_CORPORATION = "Agri_Corporation"        # 农业公司：工商资本下乡，企业化管理，资本密集
    COOPERATIVE = "Cooperative"                  # 合作社：农户联合体，集体决策，风险共担

class CropType(Enum):
    """作物类型枚举"""
    GRAIN = "Grain"                  # 粮食作物
    CASH_CROP = "Cash_Crop"          # 经济作物

class AgriculturalEnterpriseAgent(Agent):
    def __init__(self, unique_id, model, enterprise_type=None, initial_capital=None):
        super().__init__(unique_id, model)
        
        # 企业类型决定（如果未指定，则随机分配）
        # @refactor 2025-12: 移除FAMILY_FARM_LARGE，家庭农场归并到FarmerAgent
        # 企业现在仅包括农业公司（60%）和合作社（40%）
        if enterprise_type is None:
            rand = random.random()
            if rand < 0.60:
                self.enterprise_type = EnterpriseType.AGRI_CORPORATION  # 农业公司占60%
            else:
                self.enterprise_type = EnterpriseType.COOPERATIVE       # 合作社占40%
        else:
            self.enterprise_type = enterprise_type
        
        # 基于Melitz（2003）理论的异质性参数设计
        self._initialize_heterogeneous_parameters(initial_capital)
        
        # 土地经营相关
        self.leased_patches = []
        self.total_leased_area = 0
        
        # 作物组合投资组合（基于Markowitz理论）
        # @fix 2025-11-29: 中性初始化，不预设偏好
        # 初始50/50，后续完全由利润比较决定
        self.crop_portfolio = {
            CropType.GRAIN: 0.5,       # 粮食作物分配比例
            CropType.CASH_CROP: 0.5    # 经济作物分配比例
        }
        
        # @fix 2025-12-08: 添加种植面积属性，便于外部读取
        self.grain_area = 0            # 种粮面积（亩）
        self.cash_crop_area = 0        # 经作面积（亩）
        self.total_output = 0          # 总产量（kg）
        
        # 决策历史与学习
        self.decision_history = []
        self.performance_history = []
        
        # 博弈决策相关
        self.current_strategy = None
        self.expected_payoff = {}
        
        # 收入和成本相关（初始化）
        self.agricultural_revenue = 0
        self.subsidy_revenue = 0
        self.total_revenue = 0
        self.production_cost = 0
        self.land_rental_cost = 0
        self.management_cost = 0
        self.total_cost = 0
        self.profit = 0
        self.roi = 0
        self.in_kind_support_balance = 0.0
        self.in_kind_support_breakdown = {}
        
        # @new 2025-11-29: 劳动力雇佣相关
        # 理论：企业使用大型机械+雇佣农户劳动力进行生产
        self.hired_labor_cost = 0        # 雇佣劳动力成本
        self.hired_labor_days = 0        # 雇佣的劳动天数
        self.machinery_cost = 0          # 机械成本
        self.machinery_saved_labor = 0   # 机械替代的劳动力天数
        
        # @fix 2025-12-06: 初始化固定资产（用于预期机制计算资本密集度）
        # 基于企业规模估计初始固定资产：机械、设备、仓储等
        self.fixed_assets = self.total_leased_area * 500  # 约500元/亩的固定资产投入
        
        # 业绩历史（初始化）
        self.profit_history = []
        self.roi_history = []
        self.target_return_rate_history = []
        self.operating_area_history = []
        self.performance_trend = 0
        
        # 学习参数（初始化）
        self.learning_rate = 0.1
        self.aggressive_threshold = 1.2
        self.conservative_threshold = 0.8
        
        # 种植决策相关
        self.planned_crop = None
        self.expected_yield = 0
        
        # 作物选择历史
        self.crop_choice_history = []
        
        # @fix: 初始化产量字典，避免首次读取错误
        self.yields = {'grain': 0, 'cash_crop': 0}
        self.last_yields = {'grain': 0, 'cash_crop': 0}
        self.expected_yields = {CropType.GRAIN: 0, CropType.CASH_CROP: 0}  # 使用CropType枚举作为key

        # @fix 2025-11-29: 中性初始化，不预设偏好
        # 初始50/50，随机选择初始作物，后续由利润比较决定
        if random.random() < 0.5:
            self.planned_crop = CropType.GRAIN
        else:
            self.planned_crop = CropType.CASH_CROP

        # AI冲击参数
        ai_config = self.model.config.get('ai', {})
        enterprise_ai = ai_config.get('enterprise', {})
        self.ai_enabled = ai_config.get('enabled', False)
        self.ai_tfp_boost = enterprise_ai.get('tfp_boost', 0.0)
        self.ai_labor_substitution = enterprise_ai.get('labor_substitution', 0.0)
        self.ai_input_saving = enterprise_ai.get('input_saving', 0.0)
        
        # @new 2025-12: 机械化效率加成参数
        # @theory 现实动态4: 规模户机械化比较优势在粮食
        # 大型机械（联合收割机、大型播种机等）对大田作物（粮食）的适配性远高于经作
        # 粮食种植的机械化率可达95%以上，经作机械化率仅50-60%
        # 参考：《中国农业机械化年鉴2022》，《农业规模经营效率分析》
        labor_market_config = self.model.config.get('labor_market', {})
        mechanization_config = labor_market_config.get('mechanization_efficiency_bonus', {})
        self.grain_mechanization_bonus = mechanization_config.get('grain', 0.20)     # 粮食+20%效率
        self.cash_crop_mechanization_bonus = mechanization_config.get('cash_crop', 0.0)  # 经作无加成
        
        # @new 2025-12: 预期机制集成
        # 企业采用更理性的预期形成机制
        self._initialize_expectation_mechanism()

    def _initialize_heterogeneous_parameters(self, initial_capital):
        """
        基于企业类型初始化异质性参数
        
        @refactor 2025-12: 移除FAMILY_FARM_LARGE类型处理
        @new 2025-12: 添加信贷约束相关参数
        
        参数设计基于：
        1. 《农业投资回报率调查报告2023》
        2. 《中国农业规模经济效应研究》DEA分析结果
        3. 《新型农业经营主体发展报告2023》
        
        企业类型特征：
        1. 农业公司(Agri_Corporation)：工商资本下乡，资本密集，效率最高，风险偏好强
        2. 合作社(Cooperative)：农户联合体，集体决策，效率居中，风险偏好保守
        """
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司：工商资本下乡，效率最高，风险偏好较强，资本规模大
            # 特点：企业化管理、雇工经营、机械化程度高、追求利润最大化
            self.capital = initial_capital or np.random.lognormal(13.8, 1.0)  # 约500-2000万元
            self.efficiency_factor = np.random.normal(1.7, 0.3)  # 规模经济效应
            self.efficiency_factor = max(1.2, self.efficiency_factor)
            self.target_return_rate = np.random.normal(0.18, 0.04)  # 目标回报率18%
            self.target_return_rate = max(0.10, self.target_return_rate)
            self.risk_aversion = np.random.normal(1.0, 0.3)  # 更愿意承担风险
            self.risk_aversion = max(0.7, self.risk_aversion)
            
        else:  # COOPERATIVE
            # 合作社：农户联合体，效率居中，集体决策相对保守，资本规模中等
            # 特点：集体决策、风险共担、兼顾成员利益、不单纯追求利润
            self.capital = initial_capital or np.random.lognormal(12.8, 0.6)  # 约200-800万元
            self.efficiency_factor = np.random.normal(1.5, 0.25)  # 介于两者之间
            self.efficiency_factor = max(1.1, self.efficiency_factor)
            self.target_return_rate = np.random.normal(0.16, 0.03)  # 目标回报率16%
            self.target_return_rate = max(0.09, self.target_return_rate)
            self.risk_aversion = np.random.normal(1.3, 0.3)  # 集体决策相对保守
            self.risk_aversion = max(0.9, self.risk_aversion)
        
        # @new 2025-12: 信贷约束相关参数
        self._initialize_credit_parameters()
        
        # 技术采纳能力（基于企业类型差异）
        self.tech_adoption_threshold = self._calculate_tech_adoption_threshold()
        
        # 市场信息获取能力
        self.market_info_quality = self._calculate_market_info_quality()
    
    def _initialize_expectation_mechanism(self):
        """
        @new 2025-12: 初始化企业预期机制
        @updated 2025-12-05: 增加更多特征参数实现异质性
        
        企业主体比农户更接近理性预期：
        1. 农业公司：采用REH（理性预期）或适应性预期，信息较完备
        2. 合作社：采用适应性预期，介于农户和农业公司之间
        
        理论依据：
        1. Muth (1961): 理性预期假说
        2. 规模企业有专业分析团队，信息获取更充分
        3. 但农业市场不确定性高，完全理性预期不现实
        """
        try:
            # 获取企业类型字符串
            enterprise_type_str = self.enterprise_type.value if hasattr(self.enterprise_type, 'value') else str(self.enterprise_type)
            
            # 估计资本密集度（基于固定资产和经营规模）
            capital_intensity = min(1.0, self.fixed_assets / (self.total_leased_area * 1000 + 1))
            
            # 判断市场导向
            market_orientation = 'domestic'  # 默认内销
            if hasattr(self, 'export_share') and self.export_share > 0.3:
                market_orientation = 'export'
            
            # 使用工厂方法创建预期机制 - 传入完整的特征参数
            base_mechanism = ExpectationFactory.create_for_enterprise(
                enterprise_type=enterprise_type_str,
                scale=self.total_leased_area if self.total_leased_area > 0 else 500,
                years_in_operation=getattr(self, 'years_in_operation', random.randint(1, 10)),
                capital_intensity=capital_intensity,
                market_orientation=market_orientation
            )
            
            # 初始化管理器
            self.expectation_manager = ExpectationManager(base_mechanism)
            
            # 初始化历史数据
            if hasattr(self.model, 'config'):
                market_config = self.model.config.get('market', {})
                initial_grain_price = market_config.get('crop_price', 2.6)
                initial_cash_price = market_config.get('cash_crop_price', 5.0)
            else:
                initial_grain_price = 2.6
                initial_cash_price = 5.0
                
            self.expectation_manager.update_history(
                grain_price=initial_grain_price,
                cash_price=initial_cash_price,
                subsidy_rate=1.0
            )
            
        except Exception as e:
            # 预期机制初始化失败时使用默认值
            self.expectation_manager = None
            print(f"[WARN] 企业{self.unique_id}预期机制初始化失败: {e}")

    def _initialize_credit_parameters(self):
        """
        @new 2025-12: 初始化信贷相关参数
        专家审阅意见：规模户扩张依赖贷款，应有信贷约束
        
        理论依据：
        1. 农业贷款利率：一般4-8%，小额信贷可达10-15%
        2. 信贷可得性：企业比农户更容易获得贷款
        3. 资产抵押：土地经营权抵押贷款政策
        4. 杠杆率：农业企业资产负债率约40-60%
        """
        # 贷款余额
        self.debt_balance = 0.0
        
        # 贷款利率（年化）
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            self.loan_interest_rate = np.random.uniform(0.04, 0.06)  # 农业公司4-6%
            self.max_leverage_ratio = 0.6  # 最大负债率60%
            self.credit_score = np.random.uniform(0.7, 1.0)  # 信用评分较高
        else:  # COOPERATIVE
            self.loan_interest_rate = np.random.uniform(0.05, 0.08)  # 合作社5-8%
            self.max_leverage_ratio = 0.5  # 最大负债率50%
            self.credit_score = np.random.uniform(0.5, 0.8)  # 信用评分中等
        
        # 信贷可得性（银行愿意放贷的概率）
        self.credit_accessibility = self._calculate_credit_accessibility()
        
        # 贷款历史
        self.loan_history = []
    
    def _calculate_credit_accessibility(self):
        """
        计算信贷可得性
        影响因素：资产规模、企业类型、历史信用
        """
        # 基于资产规模的基础可得性
        if self.capital > 5000000:  # 500万以上
            base_accessibility = 0.9
        elif self.capital > 1000000:  # 100-500万
            base_accessibility = 0.7
        else:
            base_accessibility = 0.5
        
        # 企业类型调整
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            type_bonus = 0.1
        else:
            type_bonus = 0.0
        
        return min(1.0, base_accessibility + type_bonus)
    
    def apply_for_loan(self, amount):
        """
        @theory 申请贷款
        
        信贷约束模型：
        1. 检查资产负债率约束
        2. 检查银行放贷意愿
        3. 计算可批准金额
        
        Returns:
            approved_amount: 批准的贷款金额（0表示拒绝）
        """
        # 计算当前资产负债率
        total_assets = self.capital + self.get_land_assets_value()
        current_debt_ratio = self.debt_balance / total_assets if total_assets > 0 else 0
        
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
    
    def repay_loan(self, amount):
        """偿还贷款"""
        actual_repay = min(amount, self.debt_balance)
        self.debt_balance -= actual_repay
        return actual_repay
    
    def get_annual_interest_expense(self):
        """计算年度利息支出"""
        return self.debt_balance * self.loan_interest_rate
    
    def get_land_assets_value(self):
        """估算土地经营权资产价值"""
        # 基于年租金的10倍估值
        annual_rent = sum(p.rental_price for p in self.leased_patches) if self.leased_patches else 0
        return annual_rent * 10
    
    def get_available_capital_for_expansion(self):
        """
        @theory 计算可用于扩张的资金（考虑信贷）
        
        可用资金 = 现金 + 可获得贷款
        """
        # 现有资金（假设20%可用于新增投资）
        available_cash = self.capital * 0.2
        
        # 估算可获得贷款
        total_assets = self.capital + self.get_land_assets_value()
        potential_loan = max(0, total_assets * self.max_leverage_ratio - self.debt_balance)
        
        # 考虑信贷可得性折扣
        expected_loan = potential_loan * self.credit_accessibility
        
        return available_cash + expected_loan

    def _calculate_herding_factor(self):
        """
        @new 2025-12: 羊群行为 - 补贴预期外推
        
        专家审阅意见：
        - 现有模型企业扩张过于"理性"，只看当前收益
        - 现实中企业对政策有外推式预期：连续涨补贴 → 预期继续涨 → 过度扩张
        - 形成"补贴竞赛"和"土地囤积"现象
        
        理论依据：
        - Kahneman & Tversky (1974) 代表性启发式
        - Shiller (2000) 非理性繁荣
        - 农业经济学文献中的"政策依赖症"
        
        机制：
        expected_subsidy_growth = f(历史补贴趋势, 同行扩张信号)
        target_land_scale *= (1 + expected_growth * herding_intensity)
        """
        import numpy as np
        
        # 1. 计算补贴趋势（基于历史补贴数据外推）
        subsidy_trend = self._estimate_subsidy_trend()
        
        # 2. 同行扩张信号（"别人在抢地，我也要抢"）
        peer_expansion_signal = self._observe_peer_expansion()
        
        # 3. 企业类型差异
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司更激进：资本逐利性更强
            herding_intensity = np.random.uniform(0.6, 1.0)
        else:  # COOPERATIVE
            # 合作社更保守：农户利益导向
            herding_intensity = np.random.uniform(0.3, 0.6)
        
        # 4. 综合羊群因子
        herding_factor = (subsidy_trend * 0.6 + peer_expansion_signal * 0.4) * herding_intensity
        
        # 限制范围：-0.10 ~ +0.15（最多增加15%扩张概率）
        return max(-0.10, min(0.15, herding_factor))
    
    def _estimate_subsidy_trend(self):
        """
        @new 2025-12: 估算补贴趋势
        
        基于历史补贴数据进行简单外推
        如果补贴连续增长 → 预期继续增长 → 正向趋势
        如果补贴减少或波动 → 预期不确定 → 负向或零趋势
        """
        import numpy as np
        
        # 获取历史补贴数据（假设企业记录了最近几年的补贴）
        if not hasattr(self, 'subsidy_history'):
            self.subsidy_history = []
        
        # 尝试从environment获取当前补贴率
        try:
            current_subsidy_rate = self.model.environment.get_policy_param('grain_subsidies_rate', 0.10)
            # 记录到历史
            if len(self.subsidy_history) == 0 or self.subsidy_history[-1] != current_subsidy_rate:
                self.subsidy_history.append(current_subsidy_rate)
                # 最多保留10期
                if len(self.subsidy_history) > 10:
                    self.subsidy_history = self.subsidy_history[-10:]
        except:
            current_subsidy_rate = 0.10
        
        # 计算趋势（需要至少3期数据）
        if len(self.subsidy_history) >= 3:
            recent = self.subsidy_history[-3:]
            # 简单线性趋势
            if recent[-1] > recent[0]:  # 上涨趋势
                growth_rate = (recent[-1] - recent[0]) / max(recent[0], 0.01)
                # 外推式预期：认为趋势会持续
                expected_growth = growth_rate * 2  # 外推放大效应
                return min(0.20, max(0, expected_growth))
            elif recent[-1] < recent[0]:  # 下降趋势
                return -0.05  # 悲观预期
            else:
                return 0.0  # 稳定
        else:
            # 数据不足，假设稳定
            return 0.0
    
    def _observe_peer_expansion(self):
        """
        @new 2025-12: 观察同行扩张信号
        
        "羊群效应"的核心：别人在扩张，我也要跟上
        - 观察其他企业的土地租入行为
        - 如果发现很多企业在抢地 → 触发"fear of missing out"
        """
        import numpy as np
        
        # 获取所有企业
        try:
            all_enterprises = [a for a in self.model.schedule.agents 
                              if hasattr(a, 'enterprise_type') and a != self]
        except:
            return 0.0
        
        if len(all_enterprises) == 0:
            return 0.0
        
        # 计算平均扩张率（基于上一期的土地租入变化）
        expansion_signals = []
        for ent in all_enterprises:
            if hasattr(ent, 'leased_patches') and hasattr(ent, 'last_period_land_area'):
                current_area = sum(p.area for p in ent.leased_patches) if ent.leased_patches else 0
                last_area = ent.last_period_land_area
                if last_area > 0:
                    expansion_rate = (current_area - last_area) / last_area
                    expansion_signals.append(expansion_rate)
        
        if len(expansion_signals) == 0:
            # 无数据，假设市场平稳
            return 0.0
        
        avg_expansion = np.mean(expansion_signals)
        
        # 如果平均扩张率高于10%，触发跟风信号
        if avg_expansion > 0.10:
            return min(0.15, avg_expansion)  # 跟风扩张
        elif avg_expansion < -0.10:
            return max(-0.10, avg_expansion)  # 跟风收缩
        else:
            return 0.0  # 无明显信号

    def _calculate_tech_adoption_threshold(self):
        """计算技术采纳阈值"""
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            return np.random.uniform(0.3, 0.5)  # 更容易采纳新技术
        else:  # COOPERATIVE
            return np.random.uniform(0.4, 0.6)  # 集体决策，采纳速度中等

    def _calculate_market_info_quality(self):
        """计算市场信息获取质量"""
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            return np.random.uniform(0.8, 1.0)  # 信息获取能力强
        else:  # COOPERATIVE
            return np.random.uniform(0.7, 0.9)  # 集体信息共享优势

    def _get_village_grain_requirement(self):
        """
        @critical 2025-12: 获取企业的种粮约束比例（简化版）
        
        简化规则（基于土地类型）：
        - 永久基本农田（约80%耕地）：流转后≥80%种粮（刚性约束）
        - 一般耕地（约20%耕地）：无强制约束，靠激励政策引导
        
        计算逻辑：
        - 按地块面积加权计算企业整体种粮约束
        - 基本农田面积 × 80% / 总面积 = 企业最低种粮比例
        
        例：企业租入100亩地，其中80亩基本农田，20亩一般耕地
        → 最低种粮比例 = 80亩 × 80% / 100亩 = 64%
        
        返回:
            float: 企业整体最低粮食种植比例 [0, 0.80]
        """
        # 从配置获取村集体粮食要求
        village_config = self.model.config.get('village_collective', {})
        grain_req_config = village_config.get('grain_requirement', {})
        
        # 获取基础比例和执行力度
        base_ratio = grain_req_config.get('ratio', 0.80)  # 基本农田80%种粮
        enforcement = grain_req_config.get('enforcement_strength', 0.9)  # 90%执行率
        
        # 如果配置中比例为0，表示禁用种粮约束
        if base_ratio <= 0:
            return 0.0
        
        # 获取企业租入的所有地块
        leased_patches = getattr(self, 'leased_patches', [])
        
        if not leased_patches:
            # 没有租入土地，返回0（无约束）
            return 0.0
        
        # ================================================================
        # @critical 2025-12: 基于土地类型计算企业整体约束
        # 简化逻辑：只区分基本农田和一般耕地
        # ================================================================
        
        total_area = 0.0
        basic_farmland_area = 0.0
        
        for patch in leased_patches:
            area = getattr(patch, 'area', 1.0)
            total_area += area
            
            # 判断是否为基本农田
            land_type = getattr(patch, 'land_type', None)
            if land_type == 'basic_farmland':
                basic_farmland_area += area
            elif hasattr(patch, 'is_basic_farmland') and patch.is_basic_farmland():
                basic_farmland_area += area
        
        # 计算企业整体种粮约束
        # 公式：基本农田面积 × 基准约束比例 / 总面积
        if total_area > 0:
            avg_requirement = (basic_farmland_area * base_ratio) / total_area
        else:
            avg_requirement = 0.0
        
        # 执行力度影响（随机性）
        if np.random.random() > enforcement:
            # 约束执行不力，降低10-30%
            avg_requirement *= np.random.uniform(0.7, 0.9)
        
        return avg_requirement
    
    def _get_patch_grain_requirement_legacy(self, patch, base_ratio):
        """
        @deprecated 2025-12: 向后兼容的地块约束计算
        已简化为基于土地类型的约束
        
        Args:
            patch: 地块对象
            base_ratio: 基础种粮比例（默认0.80）
        
        Returns:
            float: 该地块的种粮约束比例
        """
        # 优先使用新的土地类型判断
        land_type = getattr(patch, 'land_type', None)
        if land_type == 'basic_farmland':
            return base_ratio  # 基本农田：80%约束
        elif land_type == 'general_farmland':
            return 0.0  # 一般耕地：无约束
        
        # 向后兼容：如果没有land_type属性，使用随机判断
        # 约80%概率被视为基本农田
        if np.random.random() < 0.80:
            return base_ratio
        else:
            return 0.0

    def optimize_crop_portfolio(self, crop_expected_returns, crop_covariance_matrix):
        """
        @theory
        使用Markowitz投资组合理论优化作物组合
        @fix: 从配置读取投资组合优化参数
        
        参数:
        crop_expected_returns: dict, 各作物期望收益率
        crop_covariance_matrix: numpy.array, 作物收益协方差矩阵
        
        返回:
        optimal_weights: dict, 最优作物分配权重
        """
        # @fix: 从配置读取投资组合优化参数
        enterprise_config = self.model.config.get('agents', {}).get('enterprise', {})
        portfolio_config = enterprise_config.get('portfolio_optimization', {})
        default_correlation = portfolio_config.get('default_correlation', 0.3)
        min_weight = portfolio_config.get('min_weight', 0.0)
        max_weight = portfolio_config.get('max_weight', 1.0)
        
        crops = list(crop_expected_returns.keys())
        n_crops = len(crops)
        
        if n_crops == 0:
            return {}
        
        # 期望收益向量
        mu = np.array([crop_expected_returns[crop] for crop in crops])
        
        # 协方差矩阵
        if crop_covariance_matrix is None or crop_covariance_matrix.shape[0] != n_crops:
            # 如果没有协方差矩阵，使用简化假设
            sigma = np.eye(n_crops) * 0.1
        else:
            sigma = crop_covariance_matrix
        
        # 目标函数：最大化 mu'w - (lambda/2) * w'Sigma*w
        # @fix 2024-11-20: 恢复风险项，这是Markowitz投资组合理论的核心
        # 之前被错误地移除了风险项，导致企业只看收益，100%种粮
        # 修复：恢复风险项，但调整尺度（方差缩小1000倍），使风险惩罚与收益在同一数量级
        def objective(w):
            portfolio_return = np.dot(mu, w)
            portfolio_risk = np.dot(w, np.dot(sigma, w))
            # 风险项尺度调整：将方差缩小1000倍，使风险惩罚与收益在同一数量级
            # 原公式：(self.risk_aversion / 2) * portfolio_risk
            # 调整后：(self.risk_aversion / 2) * (portfolio_risk / 1000.0)
            risk_penalty = (self.risk_aversion / 2) * (portfolio_risk / 1000.0)
            return -(portfolio_return - risk_penalty)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为1
        ]
        
        # 边界条件：每个作物权重在min_weight-max_weight之间（从配置读取）
        bounds = [(min_weight, max_weight) for _ in range(n_crops)]
        
        # 初始猜测：等权重
        x0 = np.ones(n_crops) / n_crops
        
        try:
            # 使用SLSQP方法求解
            result = opt.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = {}
                for i, crop in enumerate(crops):
                    optimal_weights[crop] = result.x[i]
                return optimal_weights
            else:
                # 如果优化失败，返回等权重分配
                return {crop: 1.0/n_crops for crop in crops}
                
        except Exception as e:
            # 异常处理：返回基于风险偏好的简单分配
            if self.risk_aversion > 1.5:
                # 风险厌恶者偏好粮食作物
                return {CropType.GRAIN: 0.7, CropType.CASH_CROP: 0.3}
            else:
                # 风险偏好者偏好经济作物
                return {CropType.GRAIN: 0.4, CropType.CASH_CROP: 0.6}

    def step(self):
        """
        @theory
        农业企业年度决策周期 - 严格按照多层次博弈设计规范的阶段顺序执行
        @fix: 移除土地租赁市场参与调用，因为模型层面已经统一调用
        """
        # 阶段二：首要博弈 - 作物选择
        self.crop_choice_game()
        
        # @fix: 阶段三：土地租赁市场参与 - 已在模型层面统一调用，这里不再重复调用
        # self.participate_land_rental_market()  # 已移除，由model.conduct_land_rental_market()统一调用
        
        # 阶段四：生产决策
        self.make_production_decisions()
        
        # 阶段五：收获与结算
        self.harvest_and_settle()
        
        # 阶段六：状态更新与策略调整
        self.update_state_and_adapt_strategy()

    def crop_choice_game(self):
        """
        @theory
        首要博弈：作物选择博弈
        
        @refactor 2025-11-29: 完全基于收益计算的决策机制
        移除预设偏好和最低约束，让企业根据实际利润比较自主决策
        
        决策逻辑：
        1. 计算粮食和经作的边际利润（包含补贴）
        2. 根据利润比阈值进行渐进调整
        3. 没有强制的最低约束，完全由市场决定
        """
        # 获取市场价格和补贴信息
        grain_price = self.model.environment.get_market_price('crop_price')
        cash_crop_price = self.model.environment.get_market_price('cash_crop')
        grain_subsidy_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
        
        # ================================================================
        # @fix 2025-12-01: 区分粮食和经作的成本结构
        # 问题：原来两者都用50%成本率，忽视了劳动力投入差异
        # 修复：粮食机械化程度高、劳动力需求低；经作劳动密集、成本高
        # 参考：《中国农村统计年鉴》、配置文件labor_requirements
        # ================================================================
        
        # 从配置获取劳动力需求参数
        labor_config = self.model.config.get('labor_market', {}).get('labor_requirements', {})
        grain_labor_days = labor_config.get('grain', {}).get('with_machinery', 4)  # 企业用机械：4天/亩
        cash_labor_days = labor_config.get('cash_crop', {}).get('with_machinery', 21)  # 21天/亩
        daily_wage = self.model.environment.get_market_price('wage_rate')
        
        # 机械化效率加成
        grain_mech_bonus = getattr(self, 'grain_mechanization_bonus', 0.20)
        cash_crop_mech_bonus = getattr(self, 'cash_crop_mechanization_bonus', 0.0)
        
        # ========== 粮食作物利润计算 ==========
        # 产量：750kg/亩 × 效率 × 机械化加成
        grain_yield = 750 * self.efficiency_factor * (1 + grain_mech_bonus)
        grain_revenue = grain_yield * grain_price
        
        # 成本：物资成本(40%) + 劳动力成本
        grain_material_cost = grain_revenue * 0.40  # 种子、化肥、农药等
        grain_labor_cost = grain_labor_days * daily_wage  # 劳动力成本/亩
        grain_total_cost = grain_material_cost + grain_labor_cost
        
        # 补贴
        grain_subsidy = self.model.environment.get_subsidy_for_enterprise(self, 1)
        grain_profit = grain_revenue - grain_total_cost + grain_subsidy
        
        # ========== 经济作物利润计算 ==========
        # 产量：1000kg/亩 × 效率（无机械化加成）
        cash_crop_yield = 1000 * self.efficiency_factor * (1 + cash_crop_mech_bonus)
        cash_crop_revenue = cash_crop_yield * cash_crop_price
        
        # 成本：物资成本(45%) + 劳动力成本（更高）
        # 经作劳动密集，物资成本也略高（需要更多农药、肥料等）
        cash_material_cost = cash_crop_revenue * 0.45
        cash_labor_cost = cash_labor_days * daily_wage  # 劳动力成本/亩（更高）
        cash_total_cost = cash_material_cost + cash_labor_cost
        
        cash_crop_profit = cash_crop_revenue - cash_total_cost  # 无补贴
        
        # 计算利润比
        profit_ratio = grain_profit / max(cash_crop_profit, 1)
        
        # 获取当前配置
        current_grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        
        # ================================================================
        # 基于利润比较的渐进调整规则
        # @new 2025-12: 添加村集体/合作社强制种粮约束
        # @theory 现实动态5: 村集体统一流转的土地有粮食种植最低要求
        # ================================================================
        
        # 调整幅度：企业类型决定调整速度
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            adjustment_step = 0.10  # 农业公司调整更激进
        else:  # COOPERATIVE
            adjustment_step = 0.05  # 合作社调整更保守
        
        # 阈值比较调整
        if profit_ratio > 1.1:
            # 粮食利润更高，增加粮食比例
            new_grain_weight = min(1.0, current_grain_weight + adjustment_step)
        elif profit_ratio < 0.9:
            # 经作利润更高，减少粮食比例
            new_grain_weight = max(0.0, current_grain_weight - adjustment_step)
        else:
            # 利润相近，保持当前配置（粘性）
            new_grain_weight = current_grain_weight
        
        # ================================================================
        # @new 2025-12: 村集体强制种粮约束
        # 理论依据：村集体统一流转的土地通常含"禁止非粮化"条款
        # 实施：检查企业是否受村集体约束，如是则强制最低粮食比例
        # ================================================================
        min_grain_ratio = self._get_village_grain_requirement()
        if min_grain_ratio > 0:
            # 执行强制种粮约束
            new_grain_weight = max(min_grain_ratio, new_grain_weight)
        
        new_cash_weight = 1.0 - new_grain_weight
        
        # 更新配置
        self.crop_portfolio = {
            CropType.GRAIN: new_grain_weight, 
            CropType.CASH_CROP: new_cash_weight
        }
        
        # 决定主要作物类型
        if new_grain_weight >= new_cash_weight:
            self.planned_crop = CropType.GRAIN
        else:
            self.planned_crop = CropType.CASH_CROP
        
        # 保存历史记录
        self.last_crop_choice = self.planned_crop
        
        # 记录决策
        choice_record = {
            'period': len(self.crop_choice_history) + 1,
            'grain_profit': grain_profit,
            'cash_crop_profit': cash_crop_profit,
            'profit_ratio': profit_ratio,
            'grain_weight': new_grain_weight,
            'cash_weight': new_cash_weight,
            'planned_crop': self.planned_crop,
        }
        self.crop_choice_history.append(choice_record)

    def participate_land_rental_market(self, available_patches=None):
        """
        @theory
        核心博弈：土地租赁市场参与
        企业作为土地需求方，根据盈利预期和扩张策略决定租入土地。
        @improvement: P0优先级 - 企业扩张决策微观基础化
        基于企业类型、资本规模、效率优势和市场条件进行内生决策
        
        @critical 2025-01: 新增available_patches参数
        让企业只对已有供给报价的地块提交需求，提高匹配效率
        
        @param available_patches: 已有供给报价的地块列表（可选）
                                  如果不提供，使用所有可租赁地块
        """
        # 获取配置参数
        enterprise_config = self.model.config.get('agents', {}).get('enterprise', {})
        expansion_config = enterprise_config.get('expansion_decision', {})
        base_probs = expansion_config.get('base_participation_probability', {})
        factor_weights = expansion_config.get('factor_weights', {})
        
        # 1. 基础参与概率（基于企业类型）
        type_name = self.enterprise_type.name
        base_prob = base_probs.get(type_name, 0.5)  # 默认0.5
        
        # 2. 市场信号影响（价格差异）
        price_diff = self.model.environment.get_market_price('cash_crop') - self.model.environment.get_market_price('crop_price')
        grain_price = self.model.environment.get_market_price('crop_price')
        price_factor_weight = factor_weights.get('price_factor', 0.2)
        price_factor = price_factor_weight * max(0, price_diff / max(1e-3, grain_price))
        
        # 3. 效率优势影响（效率越高，扩张意愿越强）
        efficiency_factor_weight = factor_weights.get('efficiency_factor', 0.3)
        efficiency_advantage = (self.efficiency_factor - 1.0) * efficiency_factor_weight  # 效率因子>1表示有优势
        efficiency_factor = max(0, min(0.3, efficiency_advantage))
        
        # 4. 资本充足度影响（资本越多，扩张意愿越强）
        # @new 2025-12: 考虑信贷约束
        capital_factor_weight = factor_weights.get('capital_factor', 0.35)
        # 估算所需资本（假设每100亩需要10万元）
        required_capital_per_100mu = 100000
        
        # 计算可用资金（现金 + 可获得贷款）
        if hasattr(self, 'get_available_capital_for_expansion'):
            available_capital = self.get_available_capital_for_expansion()
        else:
            available_capital = self.capital if hasattr(self, 'capital') else 0
        
        capital_availability = min(1.0, available_capital / required_capital_per_100mu) if available_capital > 0 else 0
        capital_factor = capital_availability * capital_factor_weight
        
        # 5. 历史盈利影响（盈利越好，扩张意愿越强）
        profit_factor_weight = factor_weights.get('profit_factor', 0.15)
        if hasattr(self, 'profit_history') and len(self.profit_history) > 0:
            recent_profit = self.profit_history[-1]
            profit_factor = profit_factor_weight if recent_profit > 0 else 0
        else:
            profit_factor = 0
        
        # 6. 综合决策概率
        # @fix: 大幅提高企业扩张概率，确保企业能够参与土地流转
        expand_prob = base_prob + price_factor + efficiency_factor + capital_factor + profit_factor
        
        # @new 2025-12: 羊群行为 - 补贴预期外推
        # 专家审阅意见：农业公司扩张仍太"理性"，需添加对补贴政策的外推式预期
        herding_factor = self._calculate_herding_factor()
        expand_prob += herding_factor
        
        expand_prob = min(0.98, max(0.70, expand_prob))  # @fix: 从50%-98%提高到70%-98%，确保企业更积极参与
        
        import random
        should_expand = random.random() < expand_prob
        if not should_expand:
            return

        # 7. 计算扩张目标（基于企业能力和市场条件）
        expansion_target_config = expansion_config.get('expansion_target', {})
        current_area = self.get_operated_land_area()
        
        if current_area == 0:
            # 初始扩张目标（基于企业类型和能力）
            max_areas = expansion_target_config.get('max_area_by_type', {})
            max_area = max_areas.get(type_name, 200)
            min_area = expansion_target_config.get('min_area', 50)
            
            # 基于效率和资本调整目标
            efficiency_multiplier = 0.8 + 0.4 * (self.efficiency_factor - 1.0)  # 效率越高，目标越大
            capital_multiplier = min(1.2, 0.8 + 0.4 * capital_availability)  # 资本越多，目标越大
            
            target_expansion_area = (min_area + max_area) / 2 * efficiency_multiplier * capital_multiplier
            target_expansion_area = max(min_area, min(max_area, target_expansion_area))
        else:
            # 扩张比例（基于当前规模和市场条件）
            ratio_range = expansion_target_config.get('expansion_ratio_range', [0.2, 0.5])
            min_ratio, max_ratio = ratio_range
            
            # 基于市场条件和效率调整扩张比例
            market_condition = price_factor / price_factor_weight if price_factor_weight > 0 else 0.5
            expansion_ratio = min_ratio + (max_ratio - min_ratio) * (0.5 + 0.5 * market_condition)
            expansion_ratio *= (0.9 + 0.2 * efficiency_advantage)  # 效率优势增加扩张比例
            
            target_expansion_area = current_area * expansion_ratio

        # @new 2025-12: 羊群效应放大扩张目标
        # 补贴预期乐观时，企业倾向于"抢先扩张"、"囤地待补贴"
        subsidy_expectation = self._estimate_subsidy_trend()
        if subsidy_expectation > 0:
            # 正向预期：放大扩张目标（最多翻倍）
            herding_multiplier = 1.0 + subsidy_expectation * 2.0
            target_expansion_area *= min(2.0, herding_multiplier)
            
        # 记录当前土地面积（用于下期同行观察）
        self.last_period_land_area = current_area

        # @fix: 恢复已租入的地块列表（从地块状态中恢复）
        # 确保已流转的地块状态被正确保留
        self.leased_patches = [p for p in self.model.land_patches 
                              if p.leased_to == self]
        
        # @critical 2025-01: 使用传入的available_patches（已有供给的地块）
        # 如果没有传入，则使用所有可租赁地块（向后兼容）
        if available_patches is None:
            available_patches = self.model.get_available_patches()
        
        # @fix 2025-12: 改进地块选择策略，避免所有企业都挤在同一批高质量地块
        # 策略1: 随机打乱地块顺序，确保不同企业竞争不同地块
        # 策略2: 按加权分数排序（质量 + 随机噪声 + 与已有地块的邻近性）
        import random
        import numpy as np
        
        # 计算每块地的综合吸引力分数（加入随机噪声打破趋同）
        def calculate_patch_score(patch):
            # 基础质量分（0.5-1.5）
            quality_score = patch.soil_quality
            
            # 随机噪声（0-0.3），确保企业之间有差异化选择
            random_noise = random.uniform(0, 0.3)
            
            # 邻近性奖励：如果与已有地块相邻，优先选择
            contiguity_bonus = 0
            if hasattr(self, 'leased_patches') and self.leased_patches:
                for owned in self.leased_patches:
                    if hasattr(owned, 'pos') and hasattr(patch, 'pos'):
                        if owned.pos and patch.pos:
                            dist = abs(owned.pos[0] - patch.pos[0]) + abs(owned.pos[1] - patch.pos[1])
                            if dist <= 2:
                                contiguity_bonus = 0.3
                                break
            
            return quality_score + random_noise + contiguity_bonus
        
        # 按加权分数排序（而非纯质量）
        scored_patches = [(p, calculate_patch_score(p)) for p in available_patches]
        scored_patches.sort(key=lambda x: x[1], reverse=True)
        sorted_patches = [p for p, _ in scored_patches]

        rented_area = 0
        for patch in sorted_patches:
            # @fix: 检查地块是否已经被流转（避免重复出价）
            if patch.leased_to is not None:
                continue
                
            if rented_area >= target_expansion_area:
                break

            bid_price = self._calculate_bid_price(patch)

            # 只有当预期利润为正时才出价
            if bid_price > 0:
                self.model.land_clearing_house.add_demand_offer(
                    patch=patch,
                    demander=self,
                    bid_price=bid_price
                )
                rented_area += patch.area

    def _calculate_bid_price(self, patch):
        """
        计算对特定地块的最高愿付租金
        
        @theory 基于Ricardo地租理论和实证数据
        参考：《全国农产品成本收益资料汇编2023》
        - 2022年全国平均土地成本：280元/亩
        - 土地租金占总产值：15-25%
        - 区域差异：东北500-600元，中部400-600元，东部1000元
        
        地租 = 土地边际产出 = f(土壤质量, 区位, 基础设施)
        
        @critical 2025-12: 企业异质性增强
        农业公司（Agri_Corporation）：资本驱动，愿意高价抢地
        合作社（Cooperative）：农户联合体，租金敏感度高
        """
        # 获取市场价格（使用10期历史平滑短期波动）
        # @fix 2024-11-19: 从5期增加到10期，避免企业对短期价格波动过度反应
        grain_stats = self.model.environment.get_price_statistics('grain', periods=10)
        cash_crop_stats = self.model.environment.get_price_statistics('cash_crop', periods=10)
        
        # 基于实证数据的产量和产值（2022年数据校准）
        # 粮食：总产值约1442元/亩，产量约1000斤/亩（500kg）
        # 价格：2.88元/kg（1442/500）
        # @new 2025-12: 粮食机械化效率加成
        # @theory 现实动态4: 规模户机械化比较优势在粮食
        grain_mech_bonus = getattr(self, 'grain_mechanization_bonus', 0.20)
        cash_crop_mech_bonus = getattr(self, 'cash_crop_mechanization_bonus', 0.0)
        
        grain_yield_per_mu_kg = 500 * self.efficiency_factor * patch.soil_quality * (1 + grain_mech_bonus)  # kg/亩，+机械化加成
        grain_revenue_per_mu = grain_yield_per_mu_kg * grain_stats['mean']  # 元/亩
        
        # 经济作物：假设产值约2400元/亩（无机械化效率加成）
        cash_crop_yield_per_mu_kg = 300 * self.efficiency_factor * patch.soil_quality * (1 + cash_crop_mech_bonus)  # kg/亩
        cash_crop_revenue_per_mu = cash_crop_yield_per_mu_kg * cash_crop_stats['mean']  # 元/亩
        
        # 加权总产值（根据投资组合）
        grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        cash_crop_weight = self.crop_portfolio.get(CropType.CASH_CROP, 0.5)
        total_revenue_per_mu = (grain_revenue_per_mu * grain_weight + 
                                cash_crop_revenue_per_mu * cash_crop_weight)
        
        # @critical 2025-12: 企业类型差异化租金敏感度
        # 农业公司：资本充裕，可接受较高租金比例（25-30%产值）
        # 合作社：农户联合，对高租金敏感（15-20%产值）
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司：更高的基准地租率，更愿意用高租金抢地
            base_rent_ratio = np.random.uniform(0.25, 0.30)  # 25-30%
            # 资本驱动的竞争溢价
            capital_premium = min(0.15, self.capital / 2000000 * 0.1)  # 资本越多溢价越高
        else:  # COOPERATIVE
            # 合作社：较低的地租率上限，更谨慎
            base_rent_ratio = np.random.uniform(0.15, 0.20)  # 15-20%
            # 合作社没有资本溢价
            capital_premium = 0
        
        # 基准地租（元/亩）
        base_land_rent = total_revenue_per_mu * base_rent_ratio
        
        # 调整因子
        # 1. 土壤质量已经体现在产量中
        # 2. 企业效率优势：效率高的企业愿意支付更多
        efficiency_premium = (self.efficiency_factor - 1.0) * 0.5  # 效率溢价
        
        # 3. 市场竞争程度：如果目标扩张面积大，愿意多出价抢地
        competition_factor = 1.0
        if hasattr(self, 'target_expansion_area') and self.target_expansion_area > 0:
            rented_ratio = self.total_leased_area / max(1, self.target_expansion_area)
            if rented_ratio < 0.5:  # 如果租到的还不到一半，增加出价
                # @critical 2025-12: 农业公司更急于扩张
                if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
                    competition_factor = 1.15  # 农业公司增加15%
                else:
                    competition_factor = 1.05  # 合作社更保守，只增加5%
        
        # 最终出价（元/亩）
        bid_price_per_mu = base_land_rent * (1 + efficiency_premium + capital_premium) * competition_factor
        
        # @critical 2025-12: 根据企业类型设置不同的出价区间
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司：出价区间更高（300-1000元/亩）
            bid_price_per_mu = max(300, min(1000, bid_price_per_mu))
        else:  # COOPERATIVE
            # 合作社：出价区间更保守（200-600元/亩）
            bid_price_per_mu = max(200, min(600, bid_price_per_mu))
        
        # 添加小幅随机扰动（±5%）反映信息不完全
        noise = np.random.uniform(0.95, 1.05)
        bid_price_per_mu = bid_price_per_mu * noise
        
        # 返回总出价（元）
        return bid_price_per_mu * patch.area

    def calculate_crop_statistics(self):
        """
        @theory
        基于历史数据计算作物预期收益和协方差矩阵
        为Markowitz投资组合优化提供输入
        """
        # 如果环境提供了价格统计信息，优先使用
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_price_statistics'):
            # 粮食作物统计数据（使用10期历史平滑）
            # @fix 2024-11-19: 从5期增加到10期，避免企业对短期价格波动过度反应
            grain_stats = self.model.environment.get_price_statistics('grain', periods=10)
            grain_price_mean = grain_stats['mean']
            grain_price_std = grain_stats['std']
            
            # 经济作物统计数据
            cash_crop_stats = self.model.environment.get_price_statistics('cash_crop', periods=10)
            cash_crop_price_mean = cash_crop_stats['mean']
            cash_crop_price_std = cash_crop_stats['std']
            
            # 获取政策信息
            grain_subsidies_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
            
            # @calibration: 根据审评建议校准产量系数
            # 计算预期收益率（考虑补贴和成本）
            # @new 2025-12: 粮食机械化效率加成
            # @theory 现实动态4: 规模户机械化比较优势在粮食
            grain_mech_bonus = getattr(self, 'grain_mechanization_bonus', 0.20)
            cash_crop_mech_bonus = getattr(self, 'cash_crop_mechanization_bonus', 0.0)
            
            grain_yield = 750 * self.efficiency_factor * (1 + grain_mech_bonus)  # 粮食基础产量750kg/亩 + 机械化加成
            cash_crop_yield = 1000 * self.efficiency_factor * (1 + cash_crop_mech_bonus)  # 经作基础产量1000kg/亩
            
            grain_revenue = grain_yield * grain_price_mean
            grain_cost = grain_revenue * 0.5
            subsidy = self.model.environment.get_subsidy_for_enterprise(self, 1)
            grain_profit = (grain_revenue - grain_cost) + subsidy

            cash_crop_revenue = cash_crop_yield * cash_crop_price_mean
            # @fix: 经济作物成本率略高于粮食（机械化程度低）
            cash_crop_cost = cash_crop_revenue * 0.50  # 与粮食相同，简化模型
            cash_crop_profit = cash_crop_revenue - cash_crop_cost
            
            # @fix: 经济作物销售风险（轻微滞销风险）
            cash_crop_sale_success_rate = 0.95  # 5%滞销或低价风险（温和）
            cash_crop_profit *= cash_crop_sale_success_rate
            
            # @fix 2024-11-19: 使用每亩绝对利润而不是ROI
            # 原因：ROI会因补贴的杠杆效应过度偏向粮食（121% vs 95%）
            # 实际：土地是稀缺资源，企业应该最大化每亩利润，而不是资本回报率
            # 修复：直接使用每亩利润作为expected_returns
            
            # 构建预期收益字典（每亩利润）
            expected_returns = {
                CropType.GRAIN: grain_profit,          # 约2,297元/亩
                CropType.CASH_CROP: cash_crop_profit   # 约3,454元/亩
            }
            
            # 构建协方差矩阵
            # @fix 2024-11-19: 因为改用绝对利润，方差也应该是利润的方差，不是收益率的方差
            # 利润方差 ≈ (产量 * 价格std)^2 因为利润主要由价格波动驱动
            if len(grain_stats.get('history', [])) < 3 or len(cash_crop_stats.get('history', [])) < 3:
                # 使用标准差估计方差和协方差
                # 利润波动 = 产量 * 价格波动
                grain_profit_std = grain_yield * grain_price_std
                cash_profit_std = cash_crop_yield * cash_crop_price_std
                
                grain_variance = grain_profit_std ** 2
                cash_crop_variance = cash_profit_std ** 2
                
                # 估计协方差（从配置读取默认相关性）
                enterprise_config = self.model.config.get('agents', {}).get('enterprise', {})
                portfolio_config = enterprise_config.get('portfolio_optimization', {})
                correlation = portfolio_config.get('default_correlation', 0.3)
                covariance = correlation * grain_profit_std * cash_profit_std
                
                covariance_matrix = np.array([
                    [grain_variance, covariance],
                    [covariance, cash_crop_variance]
                ])
            else:
                # 使用历史价格数据计算协方差矩阵
                # @fix 2024-11-19: 转换为利润波动，而不是价格收益率
                grain_prices = np.array(grain_stats['history'])
                cash_crop_prices = np.array(cash_crop_stats['history'])
                
                # 计算利润波动序列（价格变化 * 产量）
                grain_profit_changes = np.diff(grain_prices) * grain_yield
                cash_profit_changes = np.diff(cash_crop_prices) * cash_crop_yield
                
                # 如果序列长度不同，取最短的
                min_length = min(len(grain_profit_changes), len(cash_profit_changes))
                grain_profit_changes = grain_profit_changes[-min_length:] if min_length > 0 else np.array([0])
                cash_profit_changes = cash_profit_changes[-min_length:] if min_length > 0 else np.array([0])
                
                # 计算协方差矩阵
                if min_length > 1:
                    data = np.vstack([grain_profit_changes, cash_profit_changes])
                    covariance_matrix = np.cov(data)
                else:
                    # 如果数据不足，使用默认估计（基于利润标准差）
                    grain_profit_std = grain_yield * grain_price_std
                    cash_profit_std = cash_crop_yield * cash_crop_price_std
                    covariance_matrix = np.array([
                        [grain_profit_std**2, grain_profit_std * cash_profit_std * 0.3],
                        [grain_profit_std * cash_profit_std * 0.3, cash_profit_std**2]
                    ])
        else:
            # 如果环境没有提供统计数据，使用默认估计值
            expected_returns = {
                CropType.GRAIN: 0.15,  # 粮食作物的预期收益率
                CropType.CASH_CROP: 0.22  # 经济作物的预期收益率
            }
            
            # @fix: 调整风险参数，反映经济作物中等风险特征
            covariance_matrix = np.array([
                [0.01, 0.003],   # 粮食: 低风险（有最低收购价保护）
                [0.003, 0.06]    # 经济作物: 中等风险（价格波动适中）
            ])
        
        # 企业特性调整
        # 不同类型的企业可能有不同的预期收益（例如效率因素）
        type_adjustment = 1.0
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            type_adjustment = 1.1  # 农业公司效率更高
        elif self.enterprise_type == EnterpriseType.COOPERATIVE:
            type_adjustment = 1.05  # 合作社效率中等
            
        for crop_type in expected_returns:
            expected_returns[crop_type] *= type_adjustment
            
        return expected_returns, covariance_matrix

    def make_production_decisions(self):
        """
        @theory
        生产决策 - 基于投资组合优化结果
        考虑Markowitz投资组合理论的分散化策略
        """
        total_area = self.get_operated_land_area()
        
        if total_area <= 0:
            self.expected_yield = 0
            # @fix: 即使没有土地也要初始化expected_yields
            self.expected_yields = {
                CropType.GRAIN: 0,
                CropType.CASH_CROP: 0
            }
            return
            
        # 获取最新的投资组合权重
        grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        cash_crop_weight = self.crop_portfolio.get(CropType.CASH_CROP, 0.5)
        
        # 计算不同作物的土地分配
        grain_area = total_area * grain_weight
        cash_crop_area = total_area * cash_crop_weight
        
        # @fix 2025-12-08: 保存种植面积属性，便于外部读取
        self.grain_area = grain_area
        self.cash_crop_area = cash_crop_area
        
        # @improvement: P1优先级 - 产量计算精细化
        # 使用精细化产量计算（考虑土壤质量、技术水平等）
        grain_yield = self._calculate_detailed_yield(CropType.GRAIN, grain_area)
        cash_crop_yield = self._calculate_detailed_yield(CropType.CASH_CROP, cash_crop_area)
        
        # 存储预期产量数据
        self.expected_yields = {
            CropType.GRAIN: grain_yield,
            CropType.CASH_CROP: cash_crop_yield
        }
        
        # 保持与原接口兼容
        if self.planned_crop == CropType.GRAIN:
            self.expected_yield = grain_yield
        else:  # CASH_CROP
            self.expected_yield = cash_crop_yield

    def harvest_and_settle(self):
        """
        @theory
        收获与收入结算 - 基于投资组合
        考虑不同作物的混合种植和收益
        """
        # 获取价格信息
        grain_price = self.model.environment.get_market_price('crop_price')
        cash_crop_price = self.model.environment.get_market_price('cash_crop')
        grain_subsidies_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
        
        # 计算实际产量（加入随机波动）
        grain_actual_yield = 0
        cash_crop_actual_yield = 0
        
        if hasattr(self, 'expected_yields'):
            grain_expected = self.expected_yields.get(CropType.GRAIN, 0)
            if grain_expected > 0:
                grain_actual_yield = grain_expected * np.random.normal(1.0, 0.1)
                
            cash_crop_expected = self.expected_yields.get(CropType.CASH_CROP, 0)
            if cash_crop_expected > 0:
                cash_crop_actual_yield = cash_crop_expected * np.random.normal(1.0, 0.1)
        else:
            # 兼容旧版本
            if self.expected_yield > 0:
                if self.planned_crop == CropType.GRAIN:
                    grain_actual_yield = self.expected_yield * np.random.normal(1.0, 0.1)
                else:  # CASH_CROP
                    cash_crop_actual_yield = self.expected_yield * np.random.normal(1.0, 0.1)
        
        # 计算农业收入
        grain_revenue = grain_actual_yield * grain_price
        cash_crop_revenue = cash_crop_actual_yield * cash_crop_price
        self.agricultural_revenue = grain_revenue + cash_crop_revenue
        
        # 计算其他收入和成本
        self.calculate_subsidy_revenue(grain_actual_yield)
        self.calculate_production_cost(grain_actual_yield, cash_crop_actual_yield)
        self.calculate_land_rental_cost()
        self.calculate_management_cost()
        
        # 计算总收入和总成本
        self.total_revenue = self.agricultural_revenue + self.subsidy_revenue
        self.total_cost = self.production_cost + self.land_rental_cost + self.management_cost
        
        # 计算利润和ROI
        self.profit = self.total_revenue - self.total_cost
        if self.capital > 0:
            self.roi = self.profit / self.capital
        else:
            self.roi = 0
        
        # 更新资本
        self.capital += self.profit
        
        # @fix: 保存上一期的yields供价格计算使用（类似农户的last_yield）
        if hasattr(self, 'yields'):
            self.last_yields = self.yields.copy()
        else:
            self.last_yields = {'grain': 0, 'cash_crop': 0}
        
        # 记录产量信息（当前期）
        self.yields = {
            'grain': grain_actual_yield,
            'cash_crop': cash_crop_actual_yield
        }
        
        # @fix 2025-12-08: 更新总产量属性
        self.total_output = grain_actual_yield + cash_crop_actual_yield
        
        # 记录详细收益信息
        self.revenues = {
            'grain': grain_revenue,
            'cash_crop': cash_crop_revenue,
            'subsidy': self.subsidy_revenue
        }

    def calculate_subsidy_revenue(self, grain_yield=0):
        """
        @theory
        计算补贴收入 - 根据实际粮食产量
        """
        operated_area = self.get_operated_land_area()
        grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        grain_area = operated_area * grain_weight
        
        # 根据政府政策计算补贴
        if grain_area > 0:
            # 按面积补贴
            self.subsidy_revenue = self.model.environment.get_subsidy_for_enterprise(self, grain_area)
        else:
            self.subsidy_revenue = 0

    def calculate_production_cost(self, grain_yield=0, cash_crop_yield=0):
        """
        @theory
        计算农业生产成本 - 基于不同作物产量
        @improvement: P1优先级 - 成本核算精细化
        """
        # 检查是否使用详细成本计算
        production_config = self.model.config.get('production', {})
        cost_structure = production_config.get('cost_structure', {})
        use_detailed = cost_structure.get('use_detailed_cost', True)
        
        if not use_detailed:
            # 使用简单比例（向后兼容）
            grain_cost = (grain_yield * self.model.environment.get_market_price('crop_price')) * 0.5 if grain_yield > 0 else 0
            cash_crop_cost = (cash_crop_yield * self.model.environment.get_market_price('cash_crop')) * 0.4 if cash_crop_yield > 0 else 0
            self.production_cost = grain_cost + cash_crop_cost
        else:
            # 使用详细成本计算
            total_area = self.get_operated_land_area()
            grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
            cash_crop_weight = self.crop_portfolio.get(CropType.CASH_CROP, 0.5)
            
            grain_area = total_area * grain_weight
            cash_crop_area = total_area * cash_crop_weight
            
            grain_cost = self._calculate_detailed_cost(CropType.GRAIN, grain_area) if grain_area > 0 else 0
            cash_crop_cost = self._calculate_detailed_cost(CropType.CASH_CROP, cash_crop_area) if cash_crop_area > 0 else 0
            self.production_cost = grain_cost + cash_crop_cost
        
        # 实物支持抵消
        if self.in_kind_support_balance > 0 and self.production_cost > 0:
            offset = min(self.production_cost, self.in_kind_support_balance)
            self.production_cost -= offset
            self.in_kind_support_balance -= offset
    
    def _calculate_detailed_yield(self, crop_type, area):
        """精细化产量计算（企业版）"""
        # 获取配置参数
        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        yield_factors = production_config.get('yield_factors', {})
        random_noise = production_config.get('random_noise', {})
        
        # 1. 基础产量
        if crop_type == CropType.GRAIN:
            base_yield_per_mu = base_yields.get('grain', 750)
        else:
            base_yield_per_mu = base_yields.get('cash_crop', 1000)  # 修正：从1500改为1000
        
        base_yield = base_yield_per_mu * area
        
        # 2. 效率因子影响（企业特有）
        efficiency_effect = base_yield * self.efficiency_factor
        
        # 3. 土壤质量影响（平均）
        avg_soil_quality = self._get_average_soil_quality()
        soil_config = yield_factors.get('soil_quality', {})
        min_effect = soil_config.get('min_effect', 0.5)
        max_effect = soil_config.get('max_effect', 1.0)
        exponent = soil_config.get('exponent', 0.8)
        soil_factor = min_effect + (max_effect - min_effect) * (avg_soil_quality ** exponent)
        soil_effect = efficiency_effect * soil_factor
        
        # 4. 技术水平影响
        tech_config = yield_factors.get('technology', {})
        base_tech_level = tech_config.get('base_level', 1.0)
        max_improvement = tech_config.get('max_improvement', 0.2)
        tech_level = self._get_technology_level()
        tech_multiplier = 1.0 + max_improvement * (tech_level - base_tech_level)
        if self.ai_enabled:
            tech_multiplier *= (1.0 + self.ai_tfp_boost)
        tech_effect = soil_effect * tech_multiplier
        
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
    
    def _calculate_detailed_cost(self, crop_type, area):
        """
        精细化成本核算（企业版）
        
        @fix 2025-11-29: 实现企业特有的生产方式
        
        企业生产特点：
        1. 使用大型机械，可替代大部分劳动力
        2. 雇佣农户劳动力进行必要的人工作业
        3. 规模经济效应：单位成本更低
        """
        cost_structure = self.model.config.get('production', {}).get('cost_structure', {})
        input_prices = cost_structure.get('input_prices', {})
        input_requirements = cost_structure.get('input_requirements', {})
        
        seed_prices = input_prices.get('seed', {})
        fertilizer_price = input_prices.get('fertilizer', 2.5)
        pesticide_price = input_prices.get('pesticide', 15.0)
        # 企业使用大型机械
        machinery_price = input_prices.get('large_machinery', 80.0)  # 大型机械成本更高
        
        # 获取农业劳动力工资
        if hasattr(self.model, 'labor_market'):
            labor_price = self.model.labor_market.current_agricultural_wage
        elif hasattr(self.model, 'environment'):
            labor_price = self.model.environment.get_market_price('wage_rate')
        else:
            labor_price = self.model.config.get('environment', {}).get('market', {}).get('base_non_farm_wage', 120.0)
        
        requirements = input_requirements.get('seed', {})
        if crop_type == CropType.GRAIN:
            seed_required = requirements.get('grain', 30)
            fertilizer_required = input_requirements.get('fertilizer', {}).get('grain', 100)
            pesticide_required = input_requirements.get('pesticide', {}).get('grain', 2)
            # 企业使用机械后劳动力需求大幅降低
            base_labor_days = input_requirements.get('labor_days', {}).get('grain', 10)
            machinery_substitution_rate = 0.6  # 粮食作物机械可替代60%劳动
            labor_days_required = base_labor_days * (1 - machinery_substitution_rate)
            machinery_hours_required = input_requirements.get('machinery_hours', {}).get('grain_enterprise', 4)  # 企业机械使用更多
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('grain', 200)
            seed_price = seed_prices.get('grain', 5.0)
        else:
            seed_required = requirements.get('cash_crop', 5)
            fertilizer_required = input_requirements.get('fertilizer', {}).get('cash_crop', 150)
            pesticide_required = input_requirements.get('pesticide', {}).get('cash_crop', 5)
            # 经济作物机械化程度较低
            base_labor_days = input_requirements.get('labor_days', {}).get('cash_crop', 25)
            machinery_substitution_rate = 0.3  # 经济作物机械只能替代30%劳动
            labor_days_required = base_labor_days * (1 - machinery_substitution_rate)
            machinery_hours_required = input_requirements.get('machinery_hours', {}).get('cash_crop_enterprise', 5)
            other_cost = input_requirements.get('other_cost_per_mu', {}).get('cash_crop', 300)
            seed_price = seed_prices.get('cash_crop', 8.0)
        
        # AI与规模经济带来的成本改进
        scale_factor = 0.90  # 企业规模效应：10%成本节约
        if self.ai_enabled:
            labor_days_required *= max(0.2, 1 - self.ai_labor_substitution)
            machinery_hours_required *= max(0.5, 1 - self.ai_input_saving)
            seed_required *= max(0.4, 1 - self.ai_input_saving)
            fertilizer_required *= max(0.4, 1 - self.ai_input_saving)
            pesticide_required *= max(0.4, 1 - self.ai_input_saving)
            other_cost *= max(0.3, 1 - self.ai_input_saving)
        
        # 计算各项成本
        seed_cost = seed_price * seed_required * area * scale_factor
        fertilizer_cost = fertilizer_price * fertilizer_required * area * scale_factor
        pesticide_cost = pesticide_price * pesticide_required * area * scale_factor
        
        # 雇佣劳动力成本
        total_labor_days = labor_days_required * area
        self.hired_labor_days = total_labor_days  # 记录雇佣天数
        
        # @fix 2025-12: 正确计算并更新雇佣劳动力成本
        # 根据劳动天数和市场工资计算成本
        labor_cost = labor_price * total_labor_days * scale_factor
        self.hired_labor_cost = labor_cost  # 记录雇佣成本
        
        # 大型机械成本
        machinery_cost = machinery_price * machinery_hours_required * area * scale_factor
        self.machinery_cost = machinery_cost  # 记录机械成本
        
        # 计算机械替代的劳动力天数
        full_labor_days = base_labor_days * area
        self.machinery_saved_labor = full_labor_days * machinery_substitution_rate
        
        other_costs = other_cost * area * scale_factor
        
        total_cost = (seed_cost + fertilizer_cost + pesticide_cost + 
                     labor_cost + machinery_cost + other_costs)
        
        return total_cost
    
    def _get_average_soil_quality(self):
        """获取平均土壤质量"""
        leased_patches = [p for p in self.model.land_patches if p.leased_to == self]
        if len(leased_patches) == 0:
            return 0.75
        return sum(patch.soil_quality for patch in leased_patches) / len(leased_patches)
    
    def _get_technology_level(self):
        """获取技术水平"""
        leased_patches = [p for p in self.model.land_patches if p.leased_to == self]
        if len(leased_patches) == 0:
            return 1.0
        avg_tech = sum(getattr(patch, 'technology_level', 1.0) for patch in leased_patches) / len(leased_patches)
        return avg_tech
    
    def _get_climate_shock(self, mean, volatility):
        """获取气候冲击"""
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_climate_shock'):
            return self.model.environment.get_climate_shock()
        else:
            return np.random.normal(mean, volatility)

    def calculate_land_rental_cost(self):
        """
        @theory
        计算土地租金支出
        @fix: 从地块状态中恢复已租入的地块，确保租金成本正确计算
        
        @fix 2025-11-29: 修复租金重复计算bug
        问题：patch.rental_price存储的是总价（bid_price_per_mu * area），
              但之前又乘了patch.area，导致租金被重复计算
        修复：rental_price已经是总价，直接累加即可
        """
        # @fix: 从地块状态中恢复已租入的地块列表
        # 确保即使leased_patches被清空，也能正确计算租金成本
        leased_patches = [p for p in self.model.land_patches 
                         if p.leased_to == self]
        
        # 更新leased_patches以保持一致性
        self.leased_patches = leased_patches
        
        # @fix 2025-11-29: 修正租金计算
        # rental_price已经是该地块的总租金（已包含面积），无需再乘面积
        self.land_rental_cost = sum(
            patch.rental_price 
            for patch in leased_patches 
            if patch.rental_price > 0
        )

    def calculate_management_cost(self):
        """
        @theory
        计算管理成本
        """
        # 管理成本与经营面积成正比
        operated_area = self.get_operated_land_area()
        self.management_cost = operated_area * 50  # 每亩50元管理成本

    def update_state_and_adapt_strategy(self):
        """
        @theory
        状态更新与策略调整
        """
        # 更新业绩历史
        self.update_performance_history()
        
        # 分析业绩趋势
        self.analyze_performance_trend()
        
        # 调整目标回报率
        self.adapt_target_return_rate()
        
        # 更新学习参数
        self.update_learning_parameters()

    def update_performance_history(self):
        """
        @theory
        更新业绩历史
        """
        # 更新利润历史
        self.profit_history.append(self.profit)
        if len(self.profit_history) > 5:
            self.profit_history.pop(0)
        
        # 更新ROI历史
        self.roi_history.append(self.roi)
        if len(self.roi_history) > 5:
            self.roi_history.pop(0)
        
        # 更新目标回报率历史
        self.target_return_rate_history.append(self.target_return_rate)
        if len(self.target_return_rate_history) > 5:
            self.target_return_rate_history.pop(0)
        
        # 更新经营面积历史
        operated_area = self.get_operated_land_area()
        self.operating_area_history.append(operated_area)
        if len(self.operating_area_history) > 5:
            self.operating_area_history.pop(0)

    def analyze_performance_trend(self):
        """
        @theory
        分析业绩趋势
        """
        if len(self.roi_history) >= 2:
            # 计算ROI趋势
            recent_roi = self.roi_history[-1]
            previous_roi = self.roi_history[-2]
            
            if recent_roi > previous_roi:
                self.performance_trend = min(self.performance_trend + 0.1, 1.0)
            elif recent_roi < previous_roi:
                self.performance_trend = max(self.performance_trend - 0.1, -1.0)
            else:
                self.performance_trend = 0

    def adapt_target_return_rate(self):
        """
        @theory
        调整目标回报率
        """
        # 根据业绩趋势调整目标回报率
        if self.performance_trend > 0.5:
            # 业绩良好，可以设定更高目标
            self.target_return_rate = min(self.target_return_rate * 1.05, 0.25)
        elif self.performance_trend < -0.5:
            # 业绩不佳，降低目标
            self.target_return_rate = max(self.target_return_rate * 0.95, 0.05)
        
        # 根据资本水平调整
        if self.capital > 200000:
            # 资本充足，可以承担更高风险
            self.target_return_rate = min(self.target_return_rate * 1.02, 0.25)
        elif self.capital < 50000:
            # 资本不足，降低风险
            self.target_return_rate = max(self.target_return_rate * 0.98, 0.05)

    def update_learning_parameters(self):
        """
        @theory
        更新学习参数
        """
        # 根据业绩调整学习率
        if self.performance_trend > 0.3:
            self.learning_rate = min(self.learning_rate * 1.1, 0.2)
        elif self.performance_trend < -0.3:
            self.learning_rate = max(self.learning_rate * 0.9, 0.05)
        
        # 根据ROI调整阈值
        if self.roi > self.target_return_rate * self.aggressive_threshold:
            # 业绩超过激进阈值，可以更激进
            self.aggressive_threshold = min(self.aggressive_threshold * 1.05, 1.5)
        elif self.roi < self.target_return_rate * self.conservative_threshold:
            # 业绩低于保守阈值，需要更保守
            self.conservative_threshold = max(self.conservative_threshold * 0.95, 0.5)
        
        # 根据市场环境调整学习率
        price_diff = self.model.environment.get_market_price('cash_crop') - self.model.environment.get_market_price('crop_price')
        if abs(price_diff) > 0.3:  # 价格差异较大时，增加学习率
            self.learning_rate = min(self.learning_rate * 1.05, 0.2)
        
        # 根据企业类型调整学习率
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司学习能力更强
            self.learning_rate = min(self.learning_rate * 1.02, 0.2)
        elif self.enterprise_type == EnterpriseType.COOPERATIVE:
            # 合作社学习相对保守
            self.learning_rate = max(self.learning_rate * 0.98, 0.05)
        
        # 根据资本水平调整学习率
        if self.capital > 500000:
            # 资本充足，可以承担更多学习成本
            self.learning_rate = min(self.learning_rate * 1.03, 0.2)
        elif self.capital < 100000:
            # 资本不足，减少学习成本
            self.learning_rate = max(self.learning_rate * 0.97, 0.05)

    def get_operated_land_area(self):
        """
        @theory
        获取经营土地面积
        @fix: 从地块状态中恢复已租入的地块，确保面积正确计算
        @fix 2025-12: 同时更新total_leased_area属性
        """
        # @fix: 从地块状态中恢复已租入的地块列表
        # 确保即使leased_patches被清空，也能正确计算面积
        leased_patches = [p for p in self.model.land_patches 
                         if p.leased_to == self]
        self.leased_patches = leased_patches
        
        # @fix 2025-12: 同步更新total_leased_area属性
        area = sum(patch.area for patch in leased_patches)
        self.total_leased_area = area
        
        return area

    def get_profit_breakdown(self):
        """
        @theory
        获取利润构成
        """
        return {
            'agricultural_revenue': self.agricultural_revenue,
            'subsidy_revenue': self.subsidy_revenue,
            'total_revenue': self.total_revenue,
            'production_cost': self.production_cost,
            'land_rental_cost': self.land_rental_cost,
            'management_cost': self.management_cost,
            'total_cost': self.total_cost,
            'profit': self.profit,
            'roi': self.roi
        } 

    def receive_in_kind_support(self, value, breakdown=None):
        if value <= 0:
            return
        self.in_kind_support_balance += value
        if breakdown:
            for key, amount in breakdown.items():
                self.in_kind_support_breakdown[key] = self.in_kind_support_breakdown.get(key, 0) + amount