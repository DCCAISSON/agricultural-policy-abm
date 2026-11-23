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
import hashlib
import math
import numpy as np
import random
from copy import copy
from core.fixed_rule_runtime import ActionIncomeAccount, action_income_enabled, expected_land_cash, expected_support, settle_initial_paid_rent, submit_action_income_bids
from core.fixed_rule_runtime import production_area, preproduction_cash_enabled, settle_preproduction_cash
from enum import Enum
import scipy.optimize as opt
from core.reproducibility import (
    paired_yield_noise,
    np_normal,
    np_choice,
    np_lognormal,
    np_random_value,
    np_uniform,
    rng_choice,
    rng_randint,
    rng_random,
    rng_uniform,
)
from core.config_utils import DEFAULT_BASE_NON_FARM_WAGE, resolve_base_non_farm_wage
from utils.constants import JIN_PER_KG
from utils.ai_runtime import mechanism_enabled, productivity_params
from utils.runtime_fallback_audit import record_runtime_fallback
from utils.yield_stack_runtime import finalize_yield_stack
from utils.yield_growth_runtime import resolve_crop_annual_yield_growth
from modules.crop_components import assign_grain_crop_components

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
    def initialize_area_business(self, kind, zone, capital, capacity):
        self.enterprise_type = EnterpriseType[kind]
        self.area_kind = kind
        self.zone = int(zone)
        self.base_area = self.lease_right = 0.
        self.balance, self.area_limit = float(capital), float(capacity)
        self.fixed_area_cost = .01*self.balance
        self.management_area_cost = 10.*self.area_limit
        self.overhead = self.fixed_area_cost+self.management_area_cost
        self.sowing_intensity = 1.
        self.family_days = self.reserve = self.consumption = self.wage_income = 0.
        self.active_area_business = True

    def __init__(self, unique_id, model, enterprise_type=None, initial_capital=None):
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
        
        # 企业类型决定（如果未指定，则随机分配）
        # @refactor 2025-12: 移除FAMILY_FARM_LARGE，家庭农场归并到FarmerAgent
        # 企业现在仅包括农业公司（60%）和合作社（40%）
        if enterprise_type is None:
            rand = rng_random(self)
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
        self.leased_in_patches = []  # 与农户口径对齐，便于统一流转接口更新
        self.total_leased_area = 0
        self._leased_patch_cache = []
        self._leased_patch_cache_version = None
        self._operated_land_area_cache = 0.0
        self._operated_land_area_cache_version = None
        
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
        self.scale_subsidy_revenue = 0  # @new 2025-12-08: 规模种粮补贴收入
        self.scale_subsidy_qualified = False  # @new 2025-12-08: 是否达到规模补贴门槛
        self.total_revenue = 0
        self.production_cost = 0
        self.land_rental_cost = 0
        self.management_cost = 0
        self.total_cost = 0
        self.profit = 0
        self.roi = 0
        self.in_kind_support_balance = 0.0
        self.in_kind_support_breakdown = {}
        self.subsidy_payment_history = []
        self.pending_subsidy_claim = None
        self.arrears_received = 0.0
        self.subsidy_arrears_owed = 0.0
        
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
        self._annual_plan_step = -1
        self.expansion_willingness = 0.0
        self.target_expansion_area = 0.0
        self.last_period_land_area = 0.0
        self.land_expansion_context = {}
        
        # @fix: 初始化产量字典，避免首次读取错误
        self.yields = {'grain': 0, 'cash_crop': 0}
        self.last_yields = {'grain': 0, 'cash_crop': 0}
        self.expected_yields = {CropType.GRAIN: 0, CropType.CASH_CROP: 0}  # 使用CropType枚举作为key
        
        # @fix 2025-12-26: 添加grain_production和grain_area属性
        # CRITICAL: unified_indicators.py第156行读取e.grain_production但属性不存在
        # 这导致企业粮食产量贡献始终为0，是粮食总产量下降的根本原因！
        self.grain_production = 0  # 当期粮食产量(公斤)
        self.grain_area = 0        # 粮食种植面积(亩)
        # @new 2026-01-27: 经济作物产量属性
        self.cash_crop_production = 0  # 当期经济作物产量(公斤)
        self.crop_component_area_mu = {}
        self.crop_component_production_kg = {}
        self.crop_component_status = "not_initialized"

        # @fix 2025-11-29: 中性初始化，不预设偏好
        # 初始50/50，随机选择初始作物，后续由利润比较决定
        if rng_random(self) < 0.5:
            self.planned_crop = CropType.GRAIN
        else:
            self.planned_crop = CropType.CASH_CROP

        # 生产率冲击参数；信息治理和服务协调由独立机制处理。
        productivity_shock = productivity_params(self.model.config)
        self.productivity_shock_enabled = mechanism_enabled(self.model.config, "productivity_shock")
        self.productivity_tfp_boost = productivity_shock.get('tfp_boost', 0.0)
        self.productivity_labor_substitution = productivity_shock.get('labor_substitution', 0.0)
        self.productivity_input_saving = productivity_shock.get('input_saving', 0.0)
        
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
        enterprise_cfg = self.model.config.get('agents', {}).get('enterprise', {})
        type_cfg = enterprise_cfg.get('types', {}).get(self.enterprise_type.name, {})

        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司：工商资本下乡，效率最高，风险偏好较强，资本规模大
            # 特点：企业化管理、雇工经营、机械化程度高、追求利润最大化
            self.capital = initial_capital or float(np_lognormal(self, 13.8, 1.0))  # 约500-2000万元
            self.efficiency_factor = float(np_normal(self, 1.7, 0.3))  # 规模经济效应
            self.efficiency_factor = max(1.2, self.efficiency_factor)
            target_mean = float(type_cfg.get('target_return_rate', 0.10) or 0.10)
            target_std = float(type_cfg.get('target_return_rate_std', 0.02) or 0.02)
            target_floor = float(
                type_cfg.get('min_target_return_rate', max(0.05, target_mean * 0.70))
                or max(0.05, target_mean * 0.70)
            )
            self.target_return_rate = float(np_normal(self, target_mean, target_std))
            self.target_return_rate = max(target_floor, self.target_return_rate)
            self.risk_aversion = float(np_normal(self, 1.0, 0.3))  # 更愿意承担风险
            self.risk_aversion = max(0.7, self.risk_aversion)
            
        else:  # COOPERATIVE
            # 合作社：农户联合体，效率居中，集体决策相对保守，资本规模中等
            # 特点：集体决策、风险共担、兼顾成员利益、不单纯追求利润
            self.capital = initial_capital or float(np_lognormal(self, 12.8, 0.6))  # 约200-800万元
            self.efficiency_factor = float(np_normal(self, 1.5, 0.25))  # 介于两者之间
            self.efficiency_factor = max(1.1, self.efficiency_factor)
            target_mean = float(type_cfg.get('target_return_rate', 0.08) or 0.08)
            target_std = float(type_cfg.get('target_return_rate_std', 0.02) or 0.02)
            target_floor = float(
                type_cfg.get('min_target_return_rate', max(0.04, target_mean * 0.70))
                or max(0.04, target_mean * 0.70)
            )
            self.target_return_rate = float(np_normal(self, target_mean, target_std))
            self.target_return_rate = max(target_floor, self.target_return_rate)
            self.risk_aversion = float(np_normal(self, 1.3, 0.3))  # 集体决策相对保守
            self.risk_aversion = max(0.9, self.risk_aversion)
            
            # @new 2025-12-28: 合作社独有属性
            # @theory: 合作社与农业公司的核心区别
            # 文献：《农民专业合作社法》、Valentinov (2007)
            self._initialize_cooperative_attributes()

        self.base_target_return_rate = float(self.target_return_rate)
        
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
                years_in_operation=getattr(self, 'years_in_operation', rng_randint(self, 1, 10)),
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

    def _initialize_cooperative_attributes(self):
        """
        @new 2025-12-28: 初始化合作社独有属性
        
        @theory: 合作社的本质区别
        文献：《农民专业合作社法》(2017修订)、Valentinov (2007)
        
        合作社核心特征：
        1. 成员所有制 - 农户是合作社的所有者和受益者
        2. 民主决策 - 一人一票或基于交易量投票
        3. 盈余返还 - 利润按交易量返还成员
        4. 服务导向 - 追求成员利益最大化，非利润最大化
        
        与农业公司区别：
        - 农业公司：股东-雇工关系，追求投资回报
        - 合作社：成员-所有者关系，追求服务收益
        """
        import random
        
        # 1. 成员管理
        # 合作社成员数量：根据规模，一般10-100户
        self.member_count = rng_randint(self, 15, 60)
        
        # 成员土地贡献（虚拟连接，实际由农户单独决策）
        self.member_land_contribution = {}  # {farmer_id: contributed_area}
        
        # 成员入社时间
        self.member_join_year = {}  # {farmer_id: year}
        
        # 2. 利润分配机制
        # 盈余返还比例（法定≥60%返还成员）
        self.profit_return_ratio = rng_uniform(self, 0.60, 0.80)
        
        # 公积金提取比例
        self.reserve_fund_ratio = rng_uniform(self, 0.10, 0.20)
        
        # 分配基准：交易量(transaction) vs 股金(equity)
        self.distribution_basis = 'transaction'  # 'transaction' or 'equity'
        
        # 3. 集体决策参数
        # 重大决策需要的票数比例（≥2/3）
        self.major_decision_threshold = 0.67
        
        # 成员异质性容忍度（影响决策效率）
        # 高异质性 → 决策冲突 → 效率损失
        self.member_heterogeneity = rng_uniform(self, 0.3, 0.7)
        self.decision_efficiency = max(0.6, 1.0 - self.member_heterogeneity * 0.5)
        
        # 4. 合作社历史
        self.annual_member_returns = []  # 历年成员分红记录
        self.member_satisfaction = 0.7  # 成员满意度 (0-1)
        
        # 5. 规模不经济的额外约束 - 合作社组织成本
        # @theory: 合作社的组织成本高于农业公司
        # 原因：集体决策慢、搭便车问题、成员利益冲突
        self.coordination_cost_factor = 1.0 + 0.01 * self.member_count  # 成员越多，协调成本越高

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
            self.loan_interest_rate = float(np_uniform(self, 0.04, 0.06))  # 农业公司4-6%
            self.max_leverage_ratio = 0.6  # 最大负债率60%
            self.credit_score = float(np_uniform(self, 0.7, 1.0))  # 信用评分较高
        else:  # COOPERATIVE
            self.loan_interest_rate = float(np_uniform(self, 0.05, 0.08))  # 合作社5-8%
            self.max_leverage_ratio = 0.5  # 最大负债率50%
            self.credit_score = float(np_uniform(self, 0.5, 0.8))  # 信用评分中等
        
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
        if rng_random(self) > self.credit_accessibility:
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
        leased_patches = self._get_active_leased_patches()
        annual_rent = sum(p.rental_price for p in leased_patches) if leased_patches else 0
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

    def balance_sheet_snapshot(self, annual_investment=0.0, added_area=0.0):
        """Minimal enterprise balance sheet used by land-expansion gates.

        @provenance nfps_large_operator_bracketed
        Thresholds come from NFPS 2009-2015 large operators (area>50mu, N=2811 after outlier filter).
        Configurable via master_config.yaml agents.enterprise.financial_gate.
        """
        operated_area = max(0.0, float(self.get_operated_land_area() or 0.0))
        # Read configurable thresholds (robust to missing model in tests)
        cfg = {}
        try:
            cfg = self.model.config.get('agents', {}).get('enterprise', {}).get('financial_gate', {})
        except Exception:
            cfg = {}
        fa_per_mu = float(cfg.get('fixed_assets_per_mu', 600)) if cfg else 600.0
        sunk_share = float(cfg.get('sunk_exit_cost_share', 0.02))
        depr_rate = float(cfg.get('depreciation_rate', 0.10))

        fixed_assets = max(
            float(getattr(self, 'fixed_assets', 0.0) or 0.0),
            (operated_area + max(0.0, float(added_area or 0.0))) * fa_per_mu,
        )
        cash = max(0.0, float(getattr(self, 'capital', 0.0) or 0.0))
        land_assets = max(0.0, float(self.get_land_assets_value() or 0.0))
        debt = max(0.0, float(getattr(self, 'debt_balance', 0.0) or 0.0))
        investment = max(0.0, float(annual_investment or 0.0))
        sunk_exit_cost = fixed_assets * sunk_share
        equity = cash + fixed_assets + land_assets - debt
        return {
            'cash': cash,
            'working_capital': cash,
            'fixed_assets': fixed_assets,
            'land_assets': land_assets,
            'debt': debt,
            'equity': equity,
            'net_worth': equity,
            'annual_investment': investment,
            'depreciation': fixed_assets * depr_rate,
            'sunk_exit_cost': sunk_exit_cost,
        }

    def _passes_investment_sunk_cost_gate(self, patch, bid_price):
        """Reject expansion bids that would overrun simple balance-sheet capacity.

        @provenance nfps_large_operator_bracketed (N=2811 after outlier filter, area>50mu)
        Calibrated from NFPS 2009-2015 large-scale farm operators:
          - fixed_assets_per_mu: p50=600 yuan/mu (was stipulated 500)
          - equity_threshold: 0.21 (1 - profit_margin_p25)
          - sunk_exit_cost_share: 0.02 (expense_to_income p25)
          - depreciation: 0.10 (10-year straight-line, standard accounting)
        Estimation: calibration/nfps_enterprise_financial_profile.py
        Caveat: These are farmer households with >50mu, not registered enterprises.
        """
        self.financial_gate_call_count = int(getattr(self, 'financial_gate_call_count', 0) or 0) + 1
        patch_area = max(0.0, float(getattr(patch, 'area', 0.0) or 0.0))
        snapshot = self.balance_sheet_snapshot(annual_investment=bid_price, added_area=patch_area)
        investment = snapshot['annual_investment']
        if investment <= 0.0:
            self.financial_gate_failure_count = int(getattr(self, 'financial_gate_failure_count', 0) or 0) + 1
            return False
        operated_area = max(0.0, float(self.get_operated_land_area() or 0.0))
        sunk_buffer = snapshot['sunk_exit_cost'] * min(1.0, patch_area / max(patch_area + operated_area, 1.0))
        equity = max(snapshot['equity'], 1.0)
        # Read configurable thresholds
        cfg = {}
        try:
            cfg = self.model.config.get('agents', {}).get('enterprise', {}).get('financial_gate', {})
        except Exception:
            pass
        equity_threshold = float(cfg.get('equity_threshold', 0.21))
        cash_buffer = float(cfg.get('cash_buffer', 0.10))
        passed = (
            investment + sunk_buffer <= equity * equity_threshold
            and snapshot['cash'] - investment >= -cash_buffer * equity
        )
        if not passed:
            self.financial_gate_failure_count = int(getattr(self, 'financial_gate_failure_count', 0) or 0) + 1
        return passed

    def _get_recent_profit_signal(self):
        profit_history = getattr(self, 'profit_history', None) or []
        samples = []
        for value in profit_history[-3:]:
            try:
                samples.append(float(value or 0.0))
            except (TypeError, ValueError):
                continue
        if samples:
            return float(sum(samples) / len(samples))
        return float(getattr(self, 'profit', 0.0) or 0.0)

    def _get_recent_roi_signal(self):
        roi_history = getattr(self, 'roi_history', None) or []
        samples = []
        for value in roi_history[-3:]:
            try:
                samples.append(float(value or 0.0))
            except (TypeError, ValueError):
                continue
        if samples:
            return float(sum(samples) / len(samples))
        return float(getattr(self, 'roi', 0.0) or 0.0)

    def _get_land_rent_demand_multiplier(self):
        config = getattr(getattr(self, 'model', None), 'config', {}) or {}
        expansion_cfg = config.get('agents', {}).get('enterprise', {}).get('expansion_decision', {})
        elasticity = max(0.0, float(expansion_cfg.get('land_rent_demand_elasticity', 0.5) or 0.0))
        reference_rent = max(
            1.0,
            float(config.get('environment', {}).get('market', {}).get('base_land_rent', 280.0) or 280.0),
        )

        environment = getattr(getattr(self, 'model', None), 'environment', None)
        expected_rent = 0.0
        if environment is not None and hasattr(environment, 'get_market_price'):
            expected_rent = float(environment.get_market_price('land_rent') or 0.0)
        if expected_rent <= 0.0 and environment is not None:
            expected_rent = float(getattr(environment, 'land_rental_price', 0.0) or 0.0)
        if expected_rent <= 0.0:
            expected_rent = reference_rent

        multiplier = (expected_rent / reference_rent) ** (-elasticity) if elasticity > 0.0 else 1.0
        multiplier = max(0.20, min(2.0, multiplier))
        self.expected_land_rent_per_mu = expected_rent
        self.land_rent_reference_per_mu = reference_rent
        self.land_rent_demand_multiplier = multiplier
        return multiplier

    def _apply_land_rent_quantity_response(self, target_area):
        return max(0.0, float(target_area or 0.0)) * self._get_land_rent_demand_multiplier()

    def _fixed_rule_allows_land_expansion(self):
        return getattr(self, "fixed_rule_current_action", None) != "contract"

    def _get_land_transfer_capacity_context(self):
        """Preview how much system-level transfer capacity remains for enterprise expansion."""
        model = getattr(self, 'model', None)
        control = getattr(model, 'transfer_rate_control', {}) or {}

        total_area = float(control.get('total_land_area', 0.0) or 0.0)
        if total_area <= 0 and hasattr(model, 'land_patches'):
            total_area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in model.land_patches)

        if total_area <= 0:
            context = {
                'total_land_area': 0.0,
                'current_transfer_rate': 0.0,
                'effective_ceiling_rate': float(control.get('effective_cumulative_ceiling', 0.0) or 0.0),
                'effective_upper_bound_rate': float(control.get('effective_cumulative_ceiling_upper_bound', control.get('cumulative_ceiling', 0.72)) or 0.72),
                'headroom_signal': 1.0,
                'effective_headroom_signal': 1.0,
                'institutional_signal': 0.5,
                'overflow_ratio': 0.0,
                'upper_headroom_area': float('inf'),
            }
            self.land_expansion_context = context
            return context

        current_transferred_area = float(control.get('observed_current_transferred_area', 0.0) or 0.0)
        if current_transferred_area <= 0 and hasattr(model, 'land_patches'):
            current_transferred_area = sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in model.land_patches
                if getattr(patch, 'leased_to', None) is not None
            )

        land_clearing_house = getattr(model, 'land_clearing_house', None)
        effective_ceiling_rate = float(control.get('effective_cumulative_ceiling', 0.0) or 0.0)
        if land_clearing_house is not None and hasattr(land_clearing_house, '_estimate_endogenous_transfer_ceiling'):
            try:
                effective_ceiling_rate = float(
                    land_clearing_house._estimate_endogenous_transfer_ceiling(model, total_area) or effective_ceiling_rate
                )
            except Exception:
                effective_ceiling_rate = float(control.get('effective_cumulative_ceiling', effective_ceiling_rate) or effective_ceiling_rate)

        hard_ceiling_rate = max(
            effective_ceiling_rate,
            float(control.get('cumulative_ceiling', 0.72) or 0.72),
        )
        effective_upper_bound_rate = float(
            control.get('effective_cumulative_ceiling_upper_bound', hard_ceiling_rate) or hard_ceiling_rate
        )
        effective_upper_bound_rate = max(effective_ceiling_rate, min(hard_ceiling_rate, effective_upper_bound_rate))

        current_transfer_rate = current_transferred_area / max(total_area, 1.0)
        upper_headroom_rate = max(0.0, effective_upper_bound_rate - current_transfer_rate)
        if effective_upper_bound_rate > 0:
            headroom_signal = min(1.0, upper_headroom_rate / effective_upper_bound_rate)
        else:
            headroom_signal = 1.0

        overflow_ratio = 0.0
        if effective_upper_bound_rate > effective_ceiling_rate and current_transfer_rate > effective_ceiling_rate:
            overflow_ratio = min(
                1.0,
                (current_transfer_rate - effective_ceiling_rate)
                / max(1e-6, effective_upper_bound_rate - effective_ceiling_rate),
            )

        governance_signal = float(control.get('village_governance_signal', 0.0) or 0.0)
        if governance_signal <= 0 and hasattr(model, 'get_village_governance_capacity'):
            try:
                governance_signal = float(model.get_village_governance_capacity() or 0.0)
            except Exception:
                governance_signal = 0.0

        congestion_signal = float(control.get('village_congestion_signal', 0.0) or 0.0)
        if congestion_signal <= 0 and hasattr(model, 'get_village_transfer_congestion'):
            try:
                congestion_signal = float(model.get_village_transfer_congestion() or 0.0)
            except Exception:
                congestion_signal = 0.0

        service_slack_signal = float(control.get('service_slack_signal', max(0.0, 1.0 - congestion_signal)) or 0.0)
        unlock_signal = float(control.get('capacity_unlock_signal', 0.0) or 0.0)
        governance_signal = max(0.0, min(1.0, governance_signal))
        congestion_signal = max(0.0, min(1.0, congestion_signal))
        service_slack_signal = max(0.0, min(1.0, service_slack_signal))
        unlock_signal = max(0.0, min(1.0, unlock_signal))

        institutional_signal = max(
            0.0,
            min(1.0, 0.45 * governance_signal + 0.35 * unlock_signal + 0.20 * service_slack_signal),
        )
        effective_headroom_signal = max(
            0.0,
            min(1.0, headroom_signal * (0.70 + 0.30 * institutional_signal) * (1.0 - 0.55 * overflow_ratio)),
        )

        context = {
            'total_land_area': total_area,
            'current_transfer_rate': current_transfer_rate,
            'effective_ceiling_rate': effective_ceiling_rate,
            'effective_upper_bound_rate': effective_upper_bound_rate,
            'headroom_signal': headroom_signal,
            'effective_headroom_signal': effective_headroom_signal,
            'institutional_signal': institutional_signal,
            'overflow_ratio': overflow_ratio,
            'upper_headroom_area': upper_headroom_rate * total_area,
        }
        self.land_expansion_context = context
        return context

    def preview_land_expansion_willingness(self):
        """Expose current-year enterprise demand intensity to village coordination before clearing."""
        enterprise_config = self.model.config.get('agents', {}).get('enterprise', {})
        expansion_config = enterprise_config.get('expansion_decision', {})
        base_probs = expansion_config.get('base_participation_probability', {})

        type_name = self.enterprise_type.name
        base_prob = float(base_probs.get(type_name, 0.35) or 0.35)
        current_area = self.get_operated_land_area()
        recent_profit = self._get_recent_profit_signal()
        recent_roi = self._get_recent_roi_signal()
        target_roi = float(getattr(self, 'target_return_rate', 0.0) or 0.0)
        policy_signal = self._get_grain_policy_signal()
        context = self._get_land_transfer_capacity_context()

        required_capital_per_100mu = 100000.0
        available_capital = self.get_available_capital_for_expansion()
        capital_signal = min(1.0, available_capital / required_capital_per_100mu) if available_capital > 0 else 0.0
        efficiency_signal = max(0.0, min(1.0, self.efficiency_factor - 1.0))
        diseconomy_config = enterprise_config.get('diseconomy', {})
        supervision_cost_base = float(diseconomy_config.get('supervision_cost_base', 50.0) or 50.0)
        supervision_trigger_area = max(
            1.0,
            float(diseconomy_config.get('supervision_cost_trigger_area', 100.0) or 100.0),
        )
        target_cfg = expansion_config.get('expansion_target', {})
        entry_min_area = max(
            1.0,
            float(target_cfg.get('min_area', supervision_trigger_area) or supervision_trigger_area),
        )
        entry_max_area = max(
            entry_min_area,
            float(target_cfg.get('max_area_by_type', {}).get(type_name, entry_min_area) or entry_min_area),
        )
        starter_area = max(
            1.0,
            float(target_cfg.get('starter_area_by_type', {}).get(type_name, entry_min_area) or entry_min_area),
        )
        projected_entry_area = current_area if current_area > 0 else min(
            entry_max_area,
            max(entry_min_area, starter_area),
        )
        supervision_diseconomy_signal = 0.0
        if supervision_cost_base > 0:
            import math

            area_pressure = 0.25
            if projected_entry_area > supervision_trigger_area:
                area_pressure = max(area_pressure, math.log(projected_entry_area / supervision_trigger_area))
            supervision_diseconomy_signal = min(
                1.0,
                max(0.0, (supervision_cost_base / 50.0) * area_pressure),
            )

        willingness = (
            base_prob
            + 0.12 * efficiency_signal
            + 0.12 * capital_signal
            + 0.08 * policy_signal
            + 0.12 * float(context.get('effective_headroom_signal', 0.0) or 0.0)
            + 0.10 * float(context.get('institutional_signal', 0.0) or 0.0)
            - 0.16 * supervision_diseconomy_signal
        )
        willingness *= self._get_land_rent_demand_multiplier()

        if current_area > 0 and recent_profit < 0:
            willingness *= 0.40
        elif recent_roi < target_roi:
            willingness *= 0.75

        if float(context.get('upper_headroom_area', 0.0) or 0.0) <= 0:
            willingness = 0.0

        self.expansion_willingness = max(0.0, min(1.0, float(willingness or 0.0)))
        return self.expansion_willingness

    def _estimate_patch_operating_surplus_per_mu(self, patch):
        grain_stats = self._get_output_price_statistics_per_kg('grain', periods=10)
        cash_crop_stats = self._get_output_price_statistics_per_kg('cash_crop', periods=10)

        grain_yield_per_mu_kg = self._estimate_patch_expected_yield_per_mu(CropType.GRAIN, patch=patch)
        cash_crop_yield_per_mu_kg = self._estimate_patch_expected_yield_per_mu(CropType.CASH_CROP, patch=patch)

        grain_revenue_per_mu = grain_yield_per_mu_kg * grain_stats['mean']
        cash_crop_revenue_per_mu = cash_crop_yield_per_mu_kg * cash_crop_stats['mean']

        from core.p3_project_contracts import configured_program
        p3_program = configured_program(self.model)
        subsidy_per_mu = 0.0
        if p3_program is None and hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_subsidy_for_enterprise'):
            subsidy_per_mu = float(self.model.environment.get_subsidy_for_enterprise(self, 1.0) or 0.0)

        preferred_grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        grain_weight = self._apply_grain_requirement_and_policy_bias(
            preferred_grain_weight,
            patch=patch,
        )
        cash_crop_weight = max(0.0, 1.0 - grain_weight)

        grain_cash_cost_per_mu = grain_revenue_per_mu * 0.55
        cash_cash_cost_per_mu = cash_crop_revenue_per_mu * 0.60

        weighted_revenue_per_mu = (
            grain_revenue_per_mu * grain_weight +
            cash_crop_revenue_per_mu * cash_crop_weight
        )
        weighted_surplus_per_mu = (
            (grain_revenue_per_mu + subsidy_per_mu - grain_cash_cost_per_mu) * grain_weight +
            (cash_crop_revenue_per_mu - cash_cash_cost_per_mu) * cash_crop_weight
        )
        if p3_program is not None and patch.area > 0:
            current_area = self.get_operated_land_area()
            held = patch.leased_to is self or patch.current_operator is self
            other_area = current_area - patch.area if held else current_area
            without_grain_area = other_area * preferred_grain_weight
            retained_support = expected_support(self, without_grain_area + patch.area * grain_weight, "enterprise")
            released_support = expected_support(self, without_grain_area, "enterprise")
            weighted_surplus_per_mu += (retained_support - released_support) / patch.area
        return weighted_revenue_per_mu, weighted_surplus_per_mu

    def _get_yield_efficiency_multiplier(self):
        """Dampen enterprise efficiency when it enters the per-mu yield equation."""
        raw_efficiency = float(getattr(self, 'efficiency_factor', 1.0) or 1.0)
        cfg = (
            getattr(self.model, 'config', {})
            .get('production', {})
            .get('yield_factors', {})
            .get('enterprise_efficiency', {})
        )
        if not isinstance(cfg, dict) or not bool(cfg.get('enabled', True)):
            return max(0.0, raw_efficiency)
        baseline = float(cfg.get('baseline', 1.0) or 1.0)
        weight = float(cfg.get('yield_weight', 0.35) or 0.35)
        min_multiplier = float(cfg.get('min_multiplier', 0.90) or 0.90)
        max_multiplier = float(cfg.get('max_multiplier', 1.25) or 1.25)
        if max_multiplier < min_multiplier:
            max_multiplier = min_multiplier
        damped = baseline + weight * (raw_efficiency - baseline)
        return max(min_multiplier, min(max_multiplier, damped))

    def expected_fixed_rule_account(self, action: str, area: float | None = None) -> ActionIncomeAccount:
        paid_initial_mode = self.model.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode') in ('B_star_1y', 'B_star_5y')
        terminal = action == "contract" or (paid_initial_mode and action == "exit")
        held_patches = self._get_active_leased_patches(include_current_operator=True) if paid_initial_mode else []
        available_area = sum(p.area for p in held_patches) if paid_initial_mode else self.get_operated_land_area()
        area = max(0.0, available_area if area is None else area)
        if terminal:
            area = sum(p.area for p in self._get_active_leased_patches(include_current_operator=True) if p.leased_to is not self)
        grain_share = self._get_village_grain_requirement() if action == "expand_cash" else 1.0
        preview = copy(self)
        preview._fixed_rule_preview_area = area
        preview._get_average_soil_quality = self._get_average_soil_quality
        preview._get_technology_level = self._get_technology_level
        production_cost = hired_days = hired_cost = sales = 0.0
        outputs = {}
        for crop, crop_area in ((CropType.GRAIN, area * grain_share), (CropType.CASH_CROP, area * (1.0 - grain_share))):
            forecast = copy(preview)
            forecast._fixed_rule_preview_area = crop_area
            raw_cost = forecast._calculate_detailed_cost(crop, crop_area)
            raw_labor = forecast.hired_labor_cost
            days = forecast.hired_labor_days
            wage = self.model.labor_market.current_agricultural_wage
            hired_days += days
            hired_cost += days * wage
            production_cost += raw_cost - raw_labor + days * wage
            outputs[crop] = forecast._estimate_patch_expected_yield_per_mu(crop) * crop_area
            sales += outputs[crop] * self._get_output_price_per_kg("crop_price" if crop == CropType.GRAIN else "cash_crop")
        _, rent_in = expected_land_cash(self, area, terminal)
        if paid_initial_mode:
            rent_in = sum(p.rental_price for p in held_patches if p.leased_to is self
                          and (not terminal or (p.lease_contract_years > 0
                               and getattr(p, 'operational_transfer_channel', '') == 'author_paid_initial_lease')))
            if not terminal:
                rent_in += max(0.0, area - available_area) * float(self.model.environment.land_rental_price)
        management = float(self.model.config.get("agents", {}).get("enterprise", {}).get("diseconomy", {}).get("supervision_cost_base", 50.0)) * area
        conversion = float(getattr(self, "fixed_rule_conversion_cost_yuan", 0.0)) if action != getattr(self, "fixed_rule_previous_action", "maintain_grain") else 0.0
        available_labor = getattr(self, "fixed_rule_hired_labor_available_days", sum(max(0.0, f.off_farm_labor) for f in self.model.farmers))
        return ActionIncomeAccount(
            action=action, crop="Grain" if grain_share >= 0.5 else "Cash_Crop", area_mu=area,
            output_kg=sum(outputs.values()), sales_yuan=sales,
            support_yuan=expected_support(self, area * grain_share, "enterprise"),
            rent_out_yuan=0.0, off_farm_yuan=0.0, common_income_yuan=0.0,
            production_cost_yuan=production_cost, hired_labor_cost_yuan=hired_cost,
            service_cost_yuan=0.0, rent_in_yuan=rent_in, management_cost_yuan=management,
            conversion_cost_yuan=conversion,
            risk_penalty_yuan=float(getattr(self, "fixed_rule_risk_penalty_yuan", {}).get(action, 0.0)),
            family_labor_days=0.0, hired_labor_days=hired_days, off_farm_days=0.0,
            service_area_mu=0.0, working_capital_available=self.balance_sheet_snapshot()["working_capital"],
            hired_labor_available_days=available_labor, grain_area_mu=area * grain_share,
            grain_output_kg=outputs[CropType.GRAIN], cash_output_kg=outputs[CropType.CASH_CROP],
        )

    def _estimate_patch_expected_yield_per_mu(self, crop_type, patch=None):
        """Estimate deterministic per-mu yield used in land bidding and entry decisions."""
        production_config = self.model.config.get('production', {})
        base_yields = production_config.get('base_yield', {})
        yield_factors = production_config.get('yield_factors', {})
        tech_progress = production_config.get('technology_progress', {})

        if crop_type == CropType.GRAIN:
            base_yield_per_mu = float(base_yields.get('grain', 420) or 420)
        else:
            base_yield_per_mu = float(base_yields.get('cash_crop', 1000) or 1000)

        annual_growth = resolve_crop_annual_yield_growth(tech_progress, crop_type)
        current_step = int(getattr(self.model, 'step_count', 0) or 0)
        expected_yield_per_mu = base_yield_per_mu * ((1 + annual_growth) ** current_step)

        scale_config = tech_progress.get('scale_efficiency_bonus', {})
        threshold = float(scale_config.get('threshold_area', 30) or 30)
        representative_area = max(
            threshold,
            float(self.get_operated_land_area() or 0.0),
            float(getattr(self, 'target_expansion_area', 0.0) or 0.0),
        )
        if action_income_enabled(self.model.config):
            representative_area = self.get_operated_land_area()
        if representative_area >= 100:
            expected_yield_per_mu *= float(scale_config.get('large_scale', 1.20) or 1.20)
        elif representative_area >= threshold:
            expected_yield_per_mu *= float(scale_config.get('medium_scale', 1.10) or 1.10)
        else:
            expected_yield_per_mu *= float(scale_config.get('small_scale', 1.0) or 1.0)

        expected_yield_per_mu *= self._get_yield_efficiency_multiplier()

        soil_quality = (
            float(getattr(patch, 'soil_quality', 0.0) or 0.0)
            if patch is not None
            else float(self._get_average_soil_quality() or 0.75)
        )
        soil_config = yield_factors.get('soil_quality', {})
        min_effect = float(soil_config.get('min_effect', 0.5) or 0.5)
        max_effect = float(soil_config.get('max_effect', 1.0) or 1.0)
        exponent = float(soil_config.get('exponent', 0.8) or 0.8)
        soil_factor = min_effect + (max_effect - min_effect) * (max(0.0, soil_quality) ** exponent)
        expected_yield_per_mu *= soil_factor

        tech_config = yield_factors.get('technology', {})
        base_tech_level = float(tech_config.get('base_level', 1.0) or 1.0)
        max_improvement = float(tech_config.get('max_improvement', 0.2) or 0.2)
        tech_level = self._get_technology_level()
        expected_yield_per_mu *= 1.0 + max_improvement * max(0.0, tech_level - base_tech_level)

        management_signal = max(0.0, min(1.0, (float(self.efficiency_factor or 1.0) - 0.8) / 0.5))
        coordination_signal = (
            1.0
            if str(getattr(self.enterprise_type, 'value', self.enterprise_type)).lower().startswith('cooperative')
            else 0.75
        )
        structural_cfg = yield_factors.get('structural_efficiency', {}) or {}
        if crop_type == CropType.GRAIN:
            mechanization_signal = max(
                0.0,
                min(1.0, float(getattr(self, 'grain_mechanization_bonus', 0.20) or 0.20) / 0.25),
            )
            expected_yield_per_mu *= (
                1.0
                + float(structural_cfg.get('enterprise_grain_mechanization_weight', 0.08) or 0.0)
                * mechanization_signal
                + float(structural_cfg.get('enterprise_grain_management_weight', 0.06) or 0.0)
                * management_signal
                + float(structural_cfg.get('enterprise_grain_coordination_weight', 0.03) or 0.0)
                * coordination_signal
            )
        else:
            mechanization_signal = max(
                0.0,
                min(1.0, float(getattr(self, 'cash_crop_mechanization_bonus', 0.0) or 0.0) / 0.20),
            )
            expected_yield_per_mu *= (
                1.0
                + float(structural_cfg.get('enterprise_cash_mechanization_weight', 0.04) or 0.0)
                * mechanization_signal
                + float(structural_cfg.get('enterprise_cash_management_weight', 0.05) or 0.0)
                * management_signal
            )

        return max(0.0, expected_yield_per_mu)

    def _get_grain_requirement_config(self):
        """Resolve enterprise grain-constraint settings from the canonical config path."""
        config = getattr(self.model, 'config', {}) or {}
        agents_config = config.get('agents', {}) or {}
        village_config = (
            agents_config.get('village_collective')
            or config.get('village_collective', {})
            or {}
        )
        return village_config.get('grain_requirement', {}) or {}

    def _get_grain_policy_signal(self):
        """Map the active grain policy to a 0-1 signal for enterprise crop choice."""
        env = getattr(self.model, 'environment', None)
        active_policy = getattr(env, 'active_policy', {}) if env is not None else {}
        if not isinstance(active_policy, dict) or not active_policy:
            return 0.0

        grain_req_config = self._get_grain_requirement_config()
        policy_reference = max(
            1e-6,
            float(grain_req_config.get('policy_signal_reference', 1.0) or 1.0),
        )
        subsidy_weight = max(
            0.0,
            float(grain_req_config.get('policy_signal_subsidy_weight', 0.65) or 0.65),
        )
        price_weight = max(
            0.0,
            float(grain_req_config.get('policy_signal_price_weight', 0.35) or 0.35),
        )
        weight_total = subsidy_weight + price_weight
        if weight_total <= 0:
            subsidy_weight = 0.65
            price_weight = 0.35
            weight_total = 1.0

        subsidy_rate = max(0.0, float(active_policy.get('grain_subsidies_rate', 0.0) or 0.0))
        subsidy_signal = min(1.0, subsidy_rate / policy_reference)

        price_signal = 0.0
        market_cfg = (
            getattr(self.model, 'config', {})
            .get('environment', {})
            .get('market', {})
        )
        grain_limits = market_cfg.get('price_limits', {}).get('grain', {})
        try:
            min_price = float(grain_limits.get('min', 0.0) or 0.0)
            max_price = float(grain_limits.get('max', min_price) or min_price)
            price_floor = float(active_policy.get('grain_price_floor', min_price) or min_price)
        except (TypeError, ValueError):
            min_price = 0.0
            max_price = 0.0
            price_floor = 0.0
        if max_price > min_price:
            price_signal = max(0.0, min(1.0, (price_floor - min_price) / (max_price - min_price)))

        return min(
            1.0,
            (
                subsidy_weight * subsidy_signal
                + price_weight * price_signal
            ) / weight_total,
        )

    def _get_patch_grain_requirement(self, patch):
        """Estimate patch-level minimum grain share before the bid is placed."""
        grain_req_config = self._get_grain_requirement_config()
        basic_farmland_ratio = max(
            0.0,
            min(1.0, float(grain_req_config.get('basic_farmland_ratio', 0.80) or 0.80)),
        )
        collective_ratio = max(
            0.0,
            min(1.0, float(grain_req_config.get('ratio', 0.0) or 0.0)),
        )
        applies_to = {
            str(item).strip().lower()
            for item in (grain_req_config.get('applies_to', []) or [])
            if str(item).strip()
        }

        patch_requirement = 0.0
        if patch is not None:
            land_type = getattr(patch, 'land_type', None)
            if str(land_type) == 'basic_farmland':
                patch_requirement = basic_farmland_ratio
            elif hasattr(patch, 'is_basic_farmland'):
                try:
                    if patch.is_basic_farmland():
                        patch_requirement = basic_farmland_ratio
                except Exception:
                    pass

            transfer_channel = str(getattr(patch, 'transfer_channel', '') or '').strip().lower()
            if (
                transfer_channel in {'village_collective', 'cross_village_collective'}
                and 'enterprises_on_village_land' in applies_to
            ):
                patch_requirement = max(patch_requirement, collective_ratio)

        if self.enterprise_type == EnterpriseType.COOPERATIVE and 'cooperatives' in applies_to:
            patch_requirement = max(patch_requirement, collective_ratio)

        return max(0.0, min(1.0, patch_requirement))

    def _get_operated_patches_for_grain_requirement(self):
        return self._get_active_leased_patches(include_current_operator=True)

    def _get_active_leased_patches(self, include_current_operator: bool = False):
        """
        Rebuild the enterprise's operated-patch view from local state first and only
        fall back to a full model scan when bootstrap recovery is needed.
        """
        model = getattr(self, 'model', None)
        cache_version = getattr(model, 'land_tenure_version', None)
        cached_version = getattr(self, '_leased_patch_cache_version', None)
        cached_scope_includes_operator = bool(getattr(self, '_leased_patch_cache_include_current_operator', False))
        cached_patches = list(getattr(self, '_leased_patch_cache', []) or [])

        if cache_version is not None and cached_version == cache_version:
            if include_current_operator:
                if cached_scope_includes_operator:
                    return cached_patches
                # A leased-only cache is narrower than the requested operated-patch
                # view; rebuild below so current_operator-only patches are not missed.
            else:
                return [patch for patch in cached_patches if getattr(patch, 'leased_to', None) is self]

        candidate_patches = []
        candidate_patches.extend(getattr(self, 'leased_patches', []) or [])
        candidate_patches.extend(getattr(self, 'leased_in_patches', []) or [])
        if include_current_operator and model is not None:
            candidate_patches.extend(getattr(model, 'land_patches', []) or [])

        active_patches = []
        seen_patch_ids = set()

        for patch in candidate_patches:
            patch_id = id(patch)
            if patch_id in seen_patch_ids:
                continue
            leased_to_self = getattr(patch, 'leased_to', None) is self
            operated_by_self = include_current_operator and getattr(patch, 'current_operator', None) is self
            if not leased_to_self and not operated_by_self:
                continue
            seen_patch_ids.add(patch_id)
            active_patches.append(patch)

        if not active_patches and model is not None:
            for patch in getattr(model, 'land_patches', []) or []:
                patch_id = id(patch)
                if patch_id in seen_patch_ids:
                    continue
                leased_to_self = getattr(patch, 'leased_to', None) is self
                operated_by_self = include_current_operator and getattr(patch, 'current_operator', None) is self
                if not leased_to_self and not operated_by_self:
                    continue
                seen_patch_ids.add(patch_id)
                active_patches.append(patch)

        leased_only = [
            patch for patch in active_patches
            if getattr(patch, 'leased_to', None) is self
        ]
        leased_area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in leased_only)
        operated_area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in active_patches)

        self.leased_patches = list(leased_only)
        self.leased_in_patches = list(leased_only)
        self.total_leased_area = float(leased_area)
        self._leased_patch_cache = list(active_patches)
        self._leased_patch_cache_version = cache_version
        self._leased_patch_cache_include_current_operator = bool(include_current_operator)
        if include_current_operator:
            self._operated_land_area_cache = float(operated_area)
            self._operated_land_area_cache_version = cache_version

        if include_current_operator:
            return list(active_patches)
        return list(leased_only)

    def _get_grain_security_gap_signal(self, grain_req_config):
        env = getattr(self.model, 'environment', None)
        history = getattr(env, 'grain_market_history', None) or []
        if not history:
            return 0.0

        try:
            recent_supply = float(history[-1].get('grain_supply', 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            recent_supply = 0.0

        targets = (
            getattr(self.model, 'config', {})
            .get('policy', {})
            .get('government_objectives', {})
            .get('targets', {})
        )
        try:
            target_output = float(targets.get('grain_output', 0.0) or 0.0)
        except (TypeError, ValueError):
            target_output = 0.0
        if target_output <= 0:
            return 0.0

        gap_ratio = max(0.0, (target_output - recent_supply) / target_output)
        reference = max(1e-6, float(grain_req_config.get('security_gap_reference', 0.20) or 0.20))
        return max(0.0, min(1.0, gap_ratio / reference))

    def _apply_grain_requirement_and_policy_bias(self, preferred_grain_weight, patch=None):
        """Keep crop choice, grain policy, and bid pricing on the same grain-share logic."""
        grain_weight = max(0.0, min(1.0, float(preferred_grain_weight or 0.0)))
        min_grain_ratio = self._get_village_grain_requirement()
        if patch is not None:
            min_grain_ratio = max(min_grain_ratio, self._get_patch_grain_requirement(patch))
        grain_weight = max(grain_weight, min_grain_ratio)

        grain_req_config = self._get_grain_requirement_config()
        policy_bias_strength = max(
            0.0,
            min(1.0, float(grain_req_config.get('policy_bias_strength', 0.18) or 0.18)),
        )
        policy_bias_cap = max(
            grain_weight,
            min(1.0, float(grain_req_config.get('policy_bias_cap', 0.82) or 0.82)),
        )
        policy_signal = self._get_grain_policy_signal()
        if policy_signal > 0 and policy_bias_strength > 0 and policy_bias_cap > grain_weight:
            policy_min_weight = max(
                0.0,
                min(1.0, float(grain_req_config.get('policy_min_grain_weight', 0.0) or 0.0)),
            )
            if policy_min_weight > 0:
                grain_weight = max(grain_weight, min(policy_min_weight, policy_bias_cap))
            grain_weight += (policy_bias_cap - grain_weight) * policy_bias_strength * policy_signal
            gap_signal = self._get_grain_security_gap_signal(grain_req_config)
            gap_strength = max(
                0.0,
                min(1.0, float(grain_req_config.get('security_gap_bias_strength', 0.0) or 0.0)),
            )
            if gap_signal > 0 and gap_strength > 0 and policy_bias_cap > grain_weight:
                grain_weight += (policy_bias_cap - grain_weight) * gap_strength * gap_signal

        return max(0.0, min(1.0, grain_weight))

    def _get_output_price_per_kg(self, item_name):
        """获取作物销售价格（元/kg）。环境侧作物价格按元/斤维护。"""
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_market_price_per_kg'):
            price = env.get_market_price_per_kg(item_name)
            if price is not None:
                return float(price)
        if env is not None and hasattr(env, 'get_market_price'):
            price = env.get_market_price(item_name)
            if price is not None:
                return float(price) * JIN_PER_KG
        return 0.0

    def _get_output_price_statistics_per_kg(self, crop_type, periods=5):
        """获取作物价格统计（元/kg），兼容环境侧旧的元/斤历史口径。"""
        env = getattr(self.model, 'environment', None)
        if env is not None and hasattr(env, 'get_price_statistics_per_kg'):
            stats = env.get_price_statistics_per_kg(crop_type, periods=periods)
            if stats:
                return stats

        if env is not None and hasattr(env, 'get_price_statistics'):
            legacy_stats = env.get_price_statistics(crop_type, periods=periods)
            return {
                'mean': float(legacy_stats['mean']) * JIN_PER_KG,
                'std': float(legacy_stats['std']) * JIN_PER_KG,
                'min': float(legacy_stats['min']) * JIN_PER_KG,
                'max': float(legacy_stats['max']) * JIN_PER_KG,
                'history': [float(item) * JIN_PER_KG for item in legacy_stats.get('history', [])],
            }

        fallback_price = self._get_output_price_per_kg('crop_price' if crop_type == 'grain' else 'cash_crop')
        fallback_std = fallback_price * (0.1 if crop_type == 'grain' else 0.2)
        return {
            'mean': fallback_price,
            'std': fallback_std,
            'min': fallback_price * 0.9,
            'max': fallback_price * 1.1,
            'history': [fallback_price],
        }

    def calculate_optimal_operated_scale(self):
        """
        @theory 基于经济学原理的企业最优经营规模计算
        @enhanced 2025-12-28: 从硬编码改为基于边际分析的动态计算
        
        @literature:
        - 利润最大化条件: MR = MC (边际收益 = 边际成本)
        - 规模经济与规模不经济: 超过一定规模后边际成本上升
        - 要素约束: 资本、劳动力、投入品互相制约
        
        计算逻辑:
        1. 资本约束: 可支付的租金+投入品+劳动力总额
        2. 劳动力约束: 可雇佣/管理的劳动力上限
        3. 边际收益递减: 规模过大后管理效率下降
        4. 取以上约束的最小值作为上限
        
        Returns:
            dict: 最优规模信息
        """
        current_scale = self.get_operated_land_area()
        
        # === 1. 获取关键经济参数 ===
        # 租金成本
        avg_rent_per_mu = 500.0  # 默认500元/亩
        if hasattr(self.model, 'environment'):
            env = self.model.environment
            if hasattr(env, 'get_market_price'):
                avg_rent_per_mu = float(env.get_market_price('land_rent') or 500.0)
            else:
                avg_rent_per_mu = float(getattr(env, 'market_prices', {}).get('land_rent', 500.0) or 500.0)
        
        # 粮价和产量
        grain_price = 2.5  # 元/kg
        if hasattr(self.model, 'environment'):
            grain_price = self._get_output_price_per_kg('crop_price') or 2.5
        base_yield = float(
            self.model.config.get('production', {}).get('base_yield', {}).get('grain', 420) or 420
        )
        
        # 劳动力成本
        wage_per_day = DEFAULT_BASE_NON_FARM_WAGE  # 元/天
        if hasattr(self.model, 'environment'):
            env = self.model.environment
            if hasattr(env, 'get_market_price'):
                wage_per_day = float(env.get_market_price('wage_rate') or DEFAULT_BASE_NON_FARM_WAGE)
            else:
                wage_per_day = float(
                    getattr(env, 'market_prices', {}).get('wage_rate', DEFAULT_BASE_NON_FARM_WAGE)
                    or DEFAULT_BASE_NON_FARM_WAGE
                )
        
        # === 2. 资本约束计算可经营上限 ===
        # 每亩总成本 = 租金 + 投入品(种子/化肥/农药) + 劳动力
        input_cost_per_mu = 300.0  # 投入品成本（种子+化肥+农药）
        labor_days_per_mu = 8.0    # 每亩劳动天数（机械化企业）
        labor_cost_per_mu = labor_days_per_mu * wage_per_day / 10  # 企业雇工效率高
        
        total_cost_per_mu = avg_rent_per_mu + input_cost_per_mu + labor_cost_per_mu
        
        # 资本可支撑的最大规模
        available_capital = self.capital
        if hasattr(self, 'get_available_capital_for_expansion'):
            available_capital = self.get_available_capital_for_expansion()
        
        capital_constrained_scale = available_capital / total_cost_per_mu if total_cost_per_mu > 0 else 1000
        
        # === 3. 劳动力约束计算 ===
        # 管理能力约束：规模越大，监督成本越高
        # 公司可雇佣更多人，合作社依赖成员
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            max_workers = 20 + (self.capital / 500000) * 10  # 资本越多可雇越多人
            labor_productivity = 80  # 每人可管理80亩（机械化）
        else:  # COOPERATIVE
            member_count = getattr(self, 'member_count', 30)
            max_workers = member_count * 0.5  # 一半成员参与农业
            labor_productivity = 50  # 每人可管理50亩
        
        labor_constrained_scale = max_workers * labor_productivity
        
        # === 4. 边际收益递减约束 ===
        # 规模超过一定程度后，管理效率下降
        # 边际成本 MC = 基础成本 × (1 + 监督成本系数 × ln(规模/100))
        # 边际收益 MR = 粮价 × 产量 × 效率衰减
        gross_revenue_per_mu = grain_price * base_yield * self.efficiency_factor
        
        # 找到MR=MC的规模（简化为规模经济拐点）
        # 假设200亩以上开始出现规模不经济
        diseconomy_config = (
            self.model.config.get('agents', {})
            .get('enterprise', {})
            .get('diseconomy', {})
        )
        supervision_cost_base = float(diseconomy_config.get('supervision_cost_base', 50.0) or 50.0)
        supervision_reference = max(
            1.0,
            float(diseconomy_config.get('supervision_cost_reference', 50.0) or 50.0),
        )
        supervision_scale_penalty = max(
            0.45,
            min(1.35, supervision_reference / max(supervision_cost_base, 1e-6)),
        )
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            scale_economy_threshold = 500.0 * supervision_scale_penalty  # 公司500亩开始边际递减
        else:
            scale_economy_threshold = 300.0 * supervision_scale_penalty  # 合作社300亩开始边际递减
        
        # 边际利润 = MR - MC
        # 当规模超过阈值，边际利润开始递减
        # 最优规模约为阈值的1.5-2倍（规模不经济不太严重时）
        efficiency_optimal_scale = scale_economy_threshold * 1.5 * self.efficiency_factor
        
        # === 5. 区域异质性修正 ===
        region_factor = 1.0
        try:
            if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'spatial_manager'):
                spatial_manager = self.model.environment.spatial_manager
                if spatial_manager and hasattr(spatial_manager, 'params'):
                    sp = spatial_manager.params
                    avg_plot = getattr(sp, 'avg_plot_size', 5.0)
                    mech_potential = getattr(sp, 'mechanization_potential', 0.7)
                    region_factor = (avg_plot / 5.0) * mech_potential
        except Exception as exc:
            record_runtime_fallback(
                self.model,
                component="AgriculturalEnterpriseAgent",
                context="calculate_optimal_operated_scale.region_factor",
                error=exc,
                agent_id=getattr(self, "unique_id", None),
                fallback="region_factor=1.0",
            )
        
        efficiency_optimal_scale *= region_factor
        labor_constrained_scale *= region_factor
        capital_constrained_scale *= supervision_scale_penalty
        labor_constrained_scale *= supervision_scale_penalty
        
        # === 6. 取各约束的最小值 ===
        optimal_scale = min(
            capital_constrained_scale,
            labor_constrained_scale,
            efficiency_optimal_scale
        )
        state_space_cfg = self.model.config.get('paper_b_state_space', {}) or {}
        absorptive_capacity_scale = float(state_space_cfg.get('operator_absorptive_capacity_scale', 1.0) or 1.0)
        if not math.isfinite(absorptive_capacity_scale) or absorptive_capacity_scale <= 0.0:
            raise ValueError('operator_absorptive_capacity_scale must be positive and finite')
        optimal_scale *= absorptive_capacity_scale
        optimal_scale = max(50.0, optimal_scale)  # 最低50亩
        
        # === 7. 计算差距和决策 ===
        scale_gap = optimal_scale - current_scale
        
        aggressive_threshold = 200.0
        expand_threshold = 50.0
        contract_threshold = 100.0
        if (current_scale == 0.0 and action_income_enabled(self.model.config)
                and self.model.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode')
                in ('A', 'B_star_1y', 'B_star_5y')):
            expand_threshold = 0.0
        
        if scale_gap > aggressive_threshold:
            decision = 'aggressive_expand'
            price_adjustment = 0.95
        elif scale_gap > expand_threshold:
            decision = 'expand'
            price_adjustment = 1.0
        elif scale_gap < -contract_threshold:
            decision = 'contract'
            price_adjustment = 1.0
        else:
            decision = 'maintain'
            price_adjustment = 1.0
        
        return {
            'optimal_scale': optimal_scale,
            'current_scale': current_scale,
            'scale_gap': scale_gap,
            'decision': decision,
            'price_adjustment': price_adjustment,
            'capital_constrained': capital_constrained_scale,
            'labor_constrained': labor_constrained_scale,
            'efficiency_optimal': efficiency_optimal_scale,
            'binding_constraint': 'capital' if optimal_scale == capital_constrained_scale 
                                 else ('labor' if optimal_scale == labor_constrained_scale 
                                       else 'efficiency')
        }

    def _estimate_production_cost(self, area: float) -> float:
        """
        @theory 事前成本估算（解决先生产后扣钱问题）
        @new 2025-12-23: 实现FIX_REPORT要求的企业预算约束
        
        理论依据：
        - 预算约束理论：企业在春季播种前需要评估资金是否充足
        - 专家审阅意见：企业"先生产后扣钱"，春季资金不足仍可赊账生产
        
        成本组成（每亩）：
        - 种子：150元
        - 化肥：300元
        - 劳动力：400元（可被农业服务替代）
        
        Returns:
            float: 预估生产成本（元）
        """
        if area <= 0:
            return 0.0
        
        # 基础成本估算（每亩）
        seed_cost = 150  # 种子
        fertilizer_cost = 300  # 化肥
        labor_cost = 400  # 劳动力
        
        # 规模经济折扣（采购、机械化）
        if area >= 500:
            scale_economy_factor = 0.75  # 500亩以上享受25%折扣
        elif area >= 200:
            scale_economy_factor = 0.85  # 200-500亩享受15%折扣
        elif area >= 100:
            scale_economy_factor = 0.90  # 100-200亩享受10%折扣
        else:
            scale_economy_factor = 1.0  # 100亩以下无折扣
        
        # @new 2025-12-28: 规模不经济项（监督成本 + 激励相容问题）
        # @theory Inverse Size-Productivity Relationship
        # 文献：Fan & Chan-Kang (2005), Hazell et al. (2010)
        # 现实：大规模农业经营面临监督成本，雇工不如自营卖力
        # 实证：中国家庭农场亩产往往高于大型农业公司
        supervision_cost_per_mu = 0.0
        labor_efficiency_loss = 0.0
        
        diseconomy_config = (
            self.model.config.get('agents', {})
            .get('enterprise', {})
            .get('diseconomy', {})
        )
        supervision_cost_base = float(diseconomy_config.get('supervision_cost_base', 50.0) or 50.0)
        supervision_trigger_area = float(diseconomy_config.get('supervision_cost_trigger_area', 100.0) or 100.0)
        labor_loss_trigger_area = float(diseconomy_config.get('labor_efficiency_loss_trigger_area', 300.0) or 300.0)
        labor_loss_per_100mu = float(diseconomy_config.get('labor_efficiency_loss_per_100mu', 0.02) or 0.02)
        labor_loss_cap = float(diseconomy_config.get('labor_efficiency_loss_cap', 0.20) or 0.20)

        if area > supervision_trigger_area:  # 超过阈值后开始出现监督问题
            import math
            # 监督成本：对数增长，规模越大管理越复杂
            # 每超过阈值面积，监督成本以对数形式增加
            supervision_cost_per_mu = supervision_cost_base * math.log(area / supervision_trigger_area)
            
            # 雇工效率损失：大规模雇工激励不足导致产出效率下降
            # 超过阈值后每增加100亩，效率按配置参数下降
            if area > labor_loss_trigger_area:
                efficiency_loss_rate = min(
                    labor_loss_cap,
                    labor_loss_per_100mu * (area - labor_loss_trigger_area) / 100,
                )
                labor_efficiency_loss = labor_cost * efficiency_loss_rate
        
        # 服务替代效应（如果有农业服务替代，劳动力成本降低）
        service_substitution = 0.0
        if hasattr(self.model, 'service_market') and self.model.service_market:
            # 假设有服务市场，劳动力成本可替代部分
            service_substitution = 0.5  # 50%的劳动力可被服务替代
        
        # 劳动力成本调整
        labor_adjusted = labor_cost * (1 - service_substitution * 0.6)
        
        # 总成本 = (基础成本×规模经济折扣 + 规模不经济项) + 不可预见费用
        per_mu_base = (seed_cost + fertilizer_cost + labor_adjusted) * scale_economy_factor
        per_mu_diseconomy = supervision_cost_per_mu + labor_efficiency_loss
        per_mu_cost = per_mu_base + per_mu_diseconomy
        
        total_cost = per_mu_cost * area * 1.1  # 10%不可预见费用
        
        return total_cost

    def check_production_budget_constraint(self, area: float) -> tuple:
        """
        @new 2025-12-23: 检查生产预算约束
        
        返回：
            tuple: (is_sufficient, affordable_area, estimated_cost)
        """
        estimated_cost = self._estimate_production_cost(area)
        min_prepay_ratio = 0.30  # 至少需要30%预付
        
        available_capital = self.capital
        required_prepay = estimated_cost * min_prepay_ratio
        
        if available_capital >= required_prepay:
            return (True, area, estimated_cost)
        else:
            # 资金不足，计算可负担的面积
            affordable_ratio = available_capital / required_prepay if required_prepay > 0 else 0
            affordable_area = area * min(1.0, affordable_ratio)
            return (False, affordable_area, estimated_cost)

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
            herding_intensity = float(np_uniform(self, 0.6, 1.0))
        else:  # COOPERATIVE
            # 合作社更保守：农户利益导向
            herding_intensity = float(np_uniform(self, 0.3, 0.6))
        
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
    
    def _competitive_learning(self):
        """
        @new 2025-12-28: 竞争学习机制
        
        @literature:
        - Bandura (1977): 社会学习理论，观察模仿
        - Nelson & Winter (1982): 进化经济学，企业能力差异
        - spatial-climate-ABM: 进化压力与策略继承
        
        机制：
        1. 识别成功企业（高利润、规模扩张）
        2. 观察其策略（种植结构、技术采纳、土地策略）
        3. 概率性模仿（效率因子越低，模仿概率越高）
        
        与农户社会学习的区别：
        - 农户：邻居效应，局部信息
        - 企业：行业竞争，全局信息，更理性
        """
        import numpy as np
        
        # 获取同类型企业
        try:
            competitors = [a for a in self.model.schedule.agents 
                          if hasattr(a, 'enterprise_type') 
                          and a.enterprise_type == self.enterprise_type 
                          and a != self]
        except:
            return
        
        if len(competitors) < 2:
            return  # 竞争者太少
        
        # === 1. 识别成功企业（基于利润和规模） ===
        competitor_scores = []
        for comp in competitors:
            profit = getattr(comp, 'net_profit', 0)
            area = getattr(comp, 'operated_land', 0)
            if hasattr(comp, 'leased_patches'):
                area = sum(p.area for p in comp.leased_patches) if comp.leased_patches else 0
            
            # 综合评分：利润权重0.7 + 规模权重0.3
            score = profit * 0.7 + area * 10 * 0.3  # 规模放大便于比较
            competitor_scores.append((comp, score))
        
        # 排序找到最成功的企业
        competitor_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not competitor_scores or competitor_scores[0][1] <= 0:
            return  # 没有成功案例
        
        best_competitor = competitor_scores[0][0]
        my_score = getattr(self, 'net_profit', 0) * 0.7
        
        # === 2. 决定是否模仿（落后者更可能模仿） ===
        if my_score >= competitor_scores[0][1] * 0.8:
            return  # 我表现也不错，不需要模仿
        
        # 模仿概率：落后越多，模仿概率越高
        gap_ratio = 1 - (my_score / max(competitor_scores[0][1], 1))
        imitation_prob = min(0.3, gap_ratio * 0.5)  # 最高30%模仿概率
        
        if np_random_value(self) > imitation_prob:
            return  # 本期不模仿
        
        # === 3. 策略模仿（带噪声） ===
        # 3.1 效率因子向成功者靠拢
        if hasattr(best_competitor, 'efficiency_factor'):
            target_efficiency = best_competitor.efficiency_factor
            # 渐进调整（学习需要时间）
            adjustment = (target_efficiency - self.efficiency_factor) * 0.2
            self.efficiency_factor += adjustment
            self.efficiency_factor = max(0.8, min(1.5, self.efficiency_factor))
        
        # 3.2 风险偏好学习
        if hasattr(best_competitor, 'risk_preference'):
            target_risk = best_competitor.risk_preference
            adjustment = (target_risk - self.risk_preference) * 0.15
            self.risk_preference += adjustment
            self.risk_preference = max(0.3, min(0.9, self.risk_preference))
        
        # 3.3 记录学习事件（用于分析）
        if not hasattr(self, 'learning_history'):
            self.learning_history = []
        self.learning_history.append({
            'period': getattr(self.model, 'current_step', 0),
            'learned_from': best_competitor.unique_id,
            'improvement': adjustment if 'adjustment' in dir() else 0
        })
        if len(self.learning_history) > 20:
            self.learning_history = self.learning_history[-20:]

    def _calculate_tech_adoption_threshold(self):
        """计算技术采纳阈值"""
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            return float(np_uniform(self, 0.3, 0.5))  # 更容易采纳新技术
        else:  # COOPERATIVE
            return float(np_uniform(self, 0.4, 0.6))  # 集体决策，采纳速度中等

    def _calculate_market_info_quality(self):
        """计算市场信息获取质量"""
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            return float(np_uniform(self, 0.8, 1.0))  # 信息获取能力强
        else:  # COOPERATIVE
            return float(np_uniform(self, 0.7, 0.9))  # 集体信息共享优势

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
        grain_req_config = self._get_grain_requirement_config()
        base_ratio = max(
            0.0,
            min(1.0, float(grain_req_config.get('basic_farmland_ratio', 0.80) or 0.80)),
        )
        collective_ratio = max(
            0.0,
            min(1.0, float(grain_req_config.get('ratio', 0.0) or 0.0)),
        )
        enforcement = max(
            0.0,
            min(1.0, float(grain_req_config.get('enforcement_strength', 0.9) or 0.9)),
        )
        applies_to = {
            str(item).strip().lower()
            for item in (grain_req_config.get('applies_to', []) or [])
            if str(item).strip()
        }

        if base_ratio <= 0 and collective_ratio <= 0:
            return 0.0
        
        leased_patches = self._get_operated_patches_for_grain_requirement()
        if not leased_patches:
            return 0.0

        total_area = 0.0
        required_grain_area = 0.0
        for patch in leased_patches:
            area = max(0.0, float(getattr(patch, 'area', 1.0) or 1.0))
            total_area += area
            required_grain_area += area * self._get_patch_grain_requirement(patch)

        if total_area > 0:
            avg_requirement = required_grain_area / total_area
        else:
            avg_requirement = 0.0

        if np_random_value(self) > enforcement:
            non_compliance_floor = max(
                0.0,
                min(1.0, float(grain_req_config.get('non_compliance_floor', 0.75) or 0.75)),
            )
            avg_requirement *= float(np_uniform(self, non_compliance_floor, 1.0))

        return max(0.0, min(1.0, avg_requirement))
    
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
        if np_random_value(self) < 0.80:
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

    def prepare_annual_decisions(self):
        """Refresh the current-year crop portfolio before land-market clearing."""
        current_step = int(getattr(self.model, "step_count", 0) or 0)
        if self._annual_plan_step == current_step:
            return
        fixed_action = getattr(self, "fixed_rule_current_action", None)
        if (fixed_action in {"maintain_grain", "expand_cash", "contract"}
                or (fixed_action == "exit" and getattr(self.model, 'initial_paid_lease_accounts', ()))):
            self._annual_plan_step = current_step
            return
        self.crop_choice_game()
        self.preview_land_expansion_willingness()
        self._annual_plan_step = current_step

    def _execute_post_market_cycle(self):
        """Run production, settlement, and adaptation after factor markets clear."""
        self.make_production_decisions()
        self.harvest_and_settle()
        self.update_state_and_adapt_strategy()

    def step(self):
        """
        Run the annual enterprise cycle using the explicit planning phases managed by MainModel.
        """
        self.prepare_annual_decisions()

        # @new 2026-06-24: Enterprise Q-learning (shadow learner)
        # When mechanism_switches.use_learning_enterprise=True, augment
        # heuristic decisions with Q-table exploration.
        self._enterprise_q_learning_step()

        return self._execute_post_market_cycle()

    # ---- Enterprise Q-learning (ponytail: minimal adaptation of StrategicLearner pattern) ----
    def _enterprise_q_learning_step(self):
        """Q-learning for enterprise land-demand and pricing decisions."""
        try:
            switches = self.model.config.get("mechanism_switches", {}) or {}
        except Exception:
            return
        if not switches.get("use_learning_enterprise", False):
            return

        # Lazy-init Q-table
        if not hasattr(self, "_eq_q_table"):
            self._eq_q_table = {}  # state tuple -> np.array(4 actions)
            self._eq_last_state = None
            self._eq_last_action = None
            self._eq_epsilon = 0.15
            self._eq_lr = 0.1
            self._eq_gamma = 0.9

        # State: (profit_tier, scale_tier, rent_tier)
        profit = float(getattr(self, "annual_profit", 0.0) or 0.0)
        area = float(getattr(self, "get_operated_land_area", lambda: 0.0)() or 0.0)
        rent = float(getattr(self.model.environment, "land_rental_price", 280.0) or 280.0)
        state = (
            0 if profit < 0 else 1 if profit < 50000 else 2,
            0 if area < 30 else 1 if area < 100 else 2,
            0 if rent < 200 else 1 if rent < 500 else 2,
        )

        if state not in self._eq_q_table:
            self._eq_q_table[state] = np.zeros(4, dtype=float)

        # Epsilon-greedy action selection
        if np.random.random() < self._eq_epsilon:
            action = np.random.randint(0, 4)
        else:
            action = int(np.argmax(self._eq_q_table[state]))

        # Actions: 0=maintain, 1=expand aggressively, 2=expand conservatively, 3=contract
        if action == 1:
            setattr(self, "_eq_land_multiplier", 1.5)
            setattr(self, "_eq_bid_multiplier", 1.2)
        elif action == 2:
            setattr(self, "_eq_land_multiplier", 1.1)
            setattr(self, "_eq_bid_multiplier", 1.0)
        elif action == 3:
            setattr(self, "_eq_land_multiplier", 0.7)
            setattr(self, "_eq_bid_multiplier", 0.9)
        else:
            setattr(self, "_eq_land_multiplier", 1.0)
            setattr(self, "_eq_bid_multiplier", 1.0)

        # Update Q-table from last year's outcome
        if self._eq_last_state is not None and self._eq_last_action is not None:
            reward = float(getattr(self, "annual_profit", 0.0) or 0.0) / max(
                float(getattr(self, "get_operated_land_area", lambda: 1.0)() or 1.0), 1.0
            )
            old_q = self._eq_q_table[self._eq_last_state][self._eq_last_action]
            max_future = float(np.max(self._eq_q_table[state]))
            self._eq_q_table[self._eq_last_state][self._eq_last_action] = (
                old_q + self._eq_lr * (reward + self._eq_gamma * max_future - old_q)
            )

        self._eq_last_state = state
        self._eq_last_action = action

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
        grain_price = self._get_output_price_per_kg('crop_price')
        cash_crop_price = self._get_output_price_per_kg('cash_crop')
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
        
        # ========== 粮食作物利润计算 ==========
        # `_estimate_patch_expected_yield_per_mu` already includes technology,
        # scale, soil, management, and mechanization effects. Keep the crop
        # choice forecast on the same yield basis as production settlement.
        grain_yield = self._estimate_patch_expected_yield_per_mu(CropType.GRAIN)
        grain_revenue = grain_yield * grain_price
        
        # 成本：物资成本(40%) + 劳动力成本
        grain_material_cost = grain_revenue * 0.40  # 种子、化肥、农药等
        grain_labor_cost = grain_labor_days * daily_wage  # 劳动力成本/亩
        grain_total_cost = grain_material_cost + grain_labor_cost
        
        # 补贴
        grain_subsidy = self.model.environment.get_subsidy_for_enterprise(self, 1)
        grain_profit = grain_revenue - grain_total_cost + grain_subsidy
        
        # ========== 经济作物利润计算 ==========
        cash_crop_yield = self._estimate_patch_expected_yield_per_mu(CropType.CASH_CROP)
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
        crop_guard_config = (
            self.model.config
            .get('agents', {})
            .get('enterprise', {})
            .get('crop_profitability_guard', {})
        )
        profitability_guard_enabled = bool(crop_guard_config.get('enabled', True))
        unprofitable_cash_min_grain_weight = max(
            0.0,
            min(1.0, float(crop_guard_config.get('unprofitable_cash_min_grain_weight', 0.85) or 0.85)),
        )
        cash_profit_disadvantage_margin = max(
            0.0,
            float(crop_guard_config.get('cash_profit_disadvantage_margin_yuan_per_mu', 50.0) or 50.0),
        )
        
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

        profit_guard_applied = False
        if (
            profitability_guard_enabled
            and grain_profit - cash_crop_profit > cash_profit_disadvantage_margin
            and new_grain_weight < unprofitable_cash_min_grain_weight
        ):
            new_grain_weight = unprofitable_cash_min_grain_weight
            profit_guard_applied = True
        
        # ================================================================
        # @new 2025-12: 村集体强制种粮约束
        # 理论依据：村集体统一流转的土地通常含"禁止非粮化"条款
        # 实施：检查企业是否受村集体约束，如是则强制最低粮食比例
        # ================================================================
        new_grain_weight = self._apply_grain_requirement_and_policy_bias(new_grain_weight)
        
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
            'cash_crop_profitability_guard_applied': profit_guard_applied,
            'unprofitable_cash_min_grain_weight': unprofitable_cash_min_grain_weight,
            'cash_profit_disadvantage_margin_yuan_per_mu': cash_profit_disadvantage_margin,
            'minimum_grain_ratio': self._get_village_grain_requirement(),
            'policy_grain_signal': self._get_grain_policy_signal(),
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
        fixed_action = getattr(self, "fixed_rule_current_action", None)
        if not self._fixed_rule_allows_land_expansion():
            self.expansion_willingness = 0.0
            self.target_expansion_area = 0.0
            return
        if action_income_enabled(self.model.config):
            capacity = self._get_land_transfer_capacity_context()
            max_added = max(0.0, float(capacity['upper_headroom_area']))
            submit_action_income_bids(self, max_added)
            self.target_expansion_area = max(0.0, getattr(self, 'fixed_rule_desired_area', self.get_operated_land_area()) - self.get_operated_land_area())
            return
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
        
        recent_profit = self._get_recent_profit_signal()
        recent_roi = self._get_recent_roi_signal()
        loss_streak = 0
        for value in reversed(getattr(self, 'profit_history', None) or []):
            try:
                numeric_value = float(value or 0.0)
            except (TypeError, ValueError):
                numeric_value = 0.0
            if numeric_value < 0:
                loss_streak += 1
            else:
                break

        current_area = self.get_operated_land_area()
        capacity_context = self._get_land_transfer_capacity_context()
        target_roi = float(getattr(self, 'target_return_rate', 0.0) or 0.0)
        preview_willingness = max(
            0.0,
            min(1.0, float(self.preview_land_expansion_willingness() or 0.0)),
        )
        if current_area > 0 and loss_streak >= 2 and recent_roi < max(0.02, target_roi * 0.5):
            self.expansion_willingness = 0.0
            self.target_expansion_area = 0.0
            return

        # 5. 历史盈利影响（盈利越好，扩张意愿越强）
        profit_factor_weight = factor_weights.get('profit_factor', 0.15)
        if recent_profit > 0:
            profit_factor = profit_factor_weight
        elif recent_profit < 0:
            loss_scale = min(1.0, abs(recent_profit) / max(1.0, float(getattr(self, 'capital', 1.0) or 1.0)))
            profit_factor = -profit_factor_weight * max(0.4, loss_scale)
        else:
            profit_factor = 0
        
        # 6. 综合决策概率
        # @fix: 大幅提高企业扩张概率，确保企业能够参与土地流转
        expand_prob = base_prob + price_factor + efficiency_factor + capital_factor + profit_factor
        expand_prob = 0.50 * expand_prob + 0.50 * preview_willingness

        # @new 2025-12: 羊群行为 - 补贴预期外推
        # 专家审阅意见：农业公司扩张仍太"理性"，需添加对补贴政策的外推式预期
        herding_factor = self._calculate_herding_factor()
        expand_prob += herding_factor
        
        if recent_roi < 0:
            expand_prob *= 0.35
        elif recent_roi < target_roi:
            expand_prob *= 0.7

        headroom_signal = float(capacity_context.get('effective_headroom_signal', 1.0) or 0.0)
        institutional_signal = float(capacity_context.get('institutional_signal', 0.0) or 0.0)
        overflow_ratio = float(capacity_context.get('overflow_ratio', 0.0) or 0.0)
        if float(capacity_context.get('upper_headroom_area', 0.0) or 0.0) <= 0:
            self.expansion_willingness = 0.0
            self.target_expansion_area = 0.0
            return

        expand_prob *= (0.55 + 0.45 * headroom_signal)
        expand_prob *= (0.70 + 0.30 * institutional_signal)
        if overflow_ratio > 0:
            expand_prob *= max(0.25, 1.0 - 0.50 * overflow_ratio)

        expand_prob = min(0.90, max(0.01, expand_prob))
        self.expansion_willingness = max(0.0, min(1.0, expand_prob))
        
        import random
        should_expand = rng_random(self) < expand_prob
        if not should_expand:
            self.target_expansion_area = 0.0
            return

        # 7. 计算扩张目标（基于企业能力和市场条件）
        expansion_target_config = expansion_config.get('expansion_target', {})
        if current_area == 0:
            # 初始扩张目标（基于企业类型和能力）
            max_areas = expansion_target_config.get('max_area_by_type', {})
            max_area = max_areas.get(type_name, 200)
            min_area = expansion_target_config.get('min_area', 50)
            starter_area = expansion_target_config.get('starter_area_by_type', {}).get(type_name, min_area)
            starter_area = max(min_area, min(max_area, float(starter_area or min_area)))
            
            # 基于效率和资本调整目标
            efficiency_multiplier = 0.9 + 0.2 * max(0.0, self.efficiency_factor - 1.0)
            capital_multiplier = min(1.15, 0.9 + 0.2 * capital_availability)
            
            target_expansion_area = starter_area * efficiency_multiplier * capital_multiplier
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
        if subsidy_expectation > 0 and recent_profit >= 0:
            # 正向预期：放大扩张目标（最多翻倍）
            herding_multiplier = 1.0 + subsidy_expectation * 2.0
            target_expansion_area *= min(2.0, herding_multiplier)

        if recent_profit < 0:
            target_expansion_area *= 0.25 if current_area > 0 else 0.5
        elif recent_roi < target_roi:
            target_expansion_area *= 0.6

        target_expansion_area *= (0.40 + 0.60 * preview_willingness)
        target_expansion_area = self._apply_land_rent_quantity_response(target_expansion_area)
        target_expansion_area *= (0.35 + 0.65 * headroom_signal)
        target_expansion_area = min(
            target_expansion_area,
            float(capacity_context.get('upper_headroom_area', target_expansion_area) or target_expansion_area),
        )

        self.target_expansion_area = max(0.0, float(target_expansion_area or 0.0))
        if self.target_expansion_area <= 0:
            return
             
        # 记录当前土地面积（用于下期同行观察）
        self.last_period_land_area = current_area

        # @fix: 恢复已租入的地块列表（从地块状态中恢复）
        # 确保已流转的地块状态被正确保留
        self._get_active_leased_patches()
        
        # @critical 2025-01: 使用传入的available_patches（已有供给的地块）
        # 如果没有传入，则使用所有可租赁地块（向后兼容）
        if available_patches is None:
            available_patches = self.model.get_available_patches()
        available_patches = sorted(
            available_patches,
            key=lambda patch: str(patch.get_patch_id() if hasattr(patch, 'get_patch_id') else getattr(patch, 'patch_id', getattr(patch, 'coordinates', 'patch'))),
        )
        
        # @fix 2026-03: 将原先的随机噪声改为稳定的伪随机偏好，
        # 避免同seed重跑时因地块遍历顺序变化而把噪声分配给不同地块。
        
        # 计算每块地的综合吸引力分数（加入随机噪声打破趋同）
        def calculate_patch_score(patch):
            # 基础质量分（0.5-1.5）
            quality_score = patch.soil_quality
            
            # 稳定伪随机噪声（0-0.3），由企业ID-年份-地块ID唯一决定
            patch_key = patch.get_patch_id() if hasattr(patch, 'get_patch_id') else getattr(patch, 'patch_id', getattr(patch, 'coordinates', 'patch'))
            noise_payload = f"{self.unique_id}|{getattr(self.model, 'step_count', 0)}|{patch_key}"
            noise_digest = hashlib.md5(noise_payload.encode("utf-8")).hexdigest()
            random_noise = (int(noise_digest[:8], 16) / 0xFFFFFFFF) * 0.3
            
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
        scored_patches.sort(
            key=lambda item: (
                -item[1],
                str(item[0].get_patch_id() if hasattr(item[0], 'get_patch_id') else getattr(item[0], 'patch_id', getattr(item[0], 'coordinates', 'patch'))),
            )
        )
        sorted_patches = [p for p, _ in scored_patches]

        rented_area = 0
        for patch in sorted_patches:
            # @fix: 检查地块是否已经被流转（避免重复出价）
            if patch.leased_to is not None:
                continue
                
            if rented_area >= self.target_expansion_area:
                break

            bid_price = self._calculate_bid_price(patch)

            # 只有当预期利润为正时才出价
            if bid_price > 0 and self._passes_investment_sunk_cost_gate(patch, bid_price):
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
        total_revenue_per_mu, operating_surplus_per_mu = self._estimate_patch_operating_surplus_per_mu(patch)
        recent_profit = self._get_recent_profit_signal()
        recent_roi = self._get_recent_roi_signal()
        target_roi = float(getattr(self, 'target_return_rate', 0.0) or 0.0)

        risk_buffer_per_mu = 60.0
        if recent_profit < 0:
            risk_buffer_per_mu += 80.0
        elif recent_roi < target_roi:
            risk_buffer_per_mu += 40.0

        enterprise_cfg = self.model.config.get('agents', {}).get('enterprise', {})
        diseconomy_config = enterprise_cfg.get('diseconomy', {})
        supervision_cost_base = float(diseconomy_config.get('supervision_cost_base', 50.0) or 50.0)
        supervision_trigger_area = max(
            1.0,
            float(diseconomy_config.get('supervision_cost_trigger_area', 100.0) or 100.0),
        )
        patch_area = max(1e-6, float(getattr(patch, 'area', 0.0) or 0.0))
        current_area = max(0.0, float(self.get_operated_land_area() or 0.0))
        projected_area = current_area + patch_area
        incremental_supervision_cost_per_mu = 0.0
        if projected_area > supervision_trigger_area and supervision_cost_base > 0.0:
            import math

            current_supervision_total = 0.0
            if current_area > supervision_trigger_area:
                current_supervision_total = (
                    current_area
                    * supervision_cost_base
                    * math.log(current_area / supervision_trigger_area)
                )
            projected_supervision_total = (
                projected_area
                * supervision_cost_base
                * math.log(projected_area / supervision_trigger_area)
            )
            incremental_supervision_cost_per_mu = max(
                0.0,
                (projected_supervision_total - current_supervision_total) / patch_area,
            )

        required_margin_per_mu = max(40.0, total_revenue_per_mu * max(0.05, target_roi))
        max_affordable_rent_per_mu = (
            operating_surplus_per_mu
            - required_margin_per_mu
            - risk_buffer_per_mu
            - incremental_supervision_cost_per_mu
        )
        if max_affordable_rent_per_mu <= 0:
            return 0.0
        
        # @critical 2025-12: 企业类型差异化租金敏感度
        # 农业公司：资本充裕，可接受较高租金比例（25-30%产值）
        # 合作社：农户联合，对高租金敏感（15-20%产值）
        if self.enterprise_type == EnterpriseType.AGRI_CORPORATION:
            # 农业公司：更高的基准地租率，更愿意用高租金抢地
            base_rent_ratio = float(np_uniform(self, 0.25, 0.30))  # 25-30%
            # 资本驱动的竞争溢价
            capital_premium = min(0.15, self.capital / 2000000 * 0.1)  # 资本越多溢价越高
        else:  # COOPERATIVE
            # 合作社：较低的地租率上限，更谨慎
            base_rent_ratio = float(np_uniform(self, 0.15, 0.20))  # 15-20%
            # 合作社没有资本溢价
            capital_premium = 0
        
        # 基准地租（元/亩）：先由收益潜力决定，再受可承受盈利约束限制
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
        bid_price_per_mu = min(bid_price_per_mu, max_affordable_rent_per_mu)
        
        # 添加小幅随机扰动（±5%）反映信息不完全
        noise = float(np_uniform(self, 0.95, 1.05))
        bid_price_per_mu = max(0.0, bid_price_per_mu * noise)
        if bid_price_per_mu <= 0:
            return 0.0
        
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
            grain_stats = self._get_output_price_statistics_per_kg('grain', periods=10)
            grain_price_mean = grain_stats['mean']
            grain_price_std = grain_stats['std']
            
            # 经济作物统计数据
            cash_crop_stats = self._get_output_price_statistics_per_kg('cash_crop', periods=10)
            cash_crop_price_mean = cash_crop_stats['mean']
            cash_crop_price_std = cash_crop_stats['std']
            
            # 获取政策信息
            grain_subsidies_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
            
            # @calibration: 根据审评建议校准产量系数
            # `_estimate_patch_expected_yield_per_mu` is the single entry point
            # for technology, scale, soil, management, and mechanization effects.
            grain_yield = self._estimate_patch_expected_yield_per_mu(CropType.GRAIN)
            cash_crop_yield = self._estimate_patch_expected_yield_per_mu(CropType.CASH_CROP)
            
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
        total_area = production_area(self)
        
        if total_area <= 0:
            self.expected_yield = 0
            self.grain_area = 0.0
            self.cash_crop_area = 0.0
            self.grain_production = 0.0
            self.cash_crop_production = 0.0
            # @fix: 即使没有土地也要初始化expected_yields
            self.expected_yields = {
                CropType.GRAIN: 0,
                CropType.CASH_CROP: 0
            }
            return
            
        # Refresh the portfolio floor after land-market clearing. Annual planning
        # happens before new leased/current-operator patches are attached, so
        # basic-farmland and village-transfer grain requirements must be applied
        # again against the realized operated patch set before production.
        if action_income_enabled(self.model.config):
            grain_weight = max(self.crop_portfolio.get(CropType.GRAIN, 1.0), self._get_village_grain_requirement())
        else:
            grain_weight = self._apply_grain_requirement_and_policy_bias(
                self.crop_portfolio.get(CropType.GRAIN, 0.5)
            )
            grain_weight = self._apply_final_crop_profitability_guard_to_weight(grain_weight)
        cash_crop_weight = max(0.0, 1.0 - grain_weight)
        self.crop_portfolio = {
            CropType.GRAIN: grain_weight,
            CropType.CASH_CROP: cash_crop_weight,
        }
        self.planned_crop = CropType.GRAIN if grain_weight >= cash_crop_weight else CropType.CASH_CROP
        
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
        if action_income_enabled(self.model.config):
            fulfillment = getattr(self, "fixed_rule_labor_fulfillment", 1.0)
            grain_yield *= fulfillment
            cash_crop_yield *= fulfillment
        
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

    def _apply_final_crop_profitability_guard_to_weight(self, grain_weight):
        """Re-check enterprise crop portfolio after land-market clearing."""
        crop_guard_config = (
            self.model.config
            .get('agents', {})
            .get('enterprise', {})
            .get('crop_profitability_guard', {})
        )
        if not bool(crop_guard_config.get('enabled', True)):
            return grain_weight

        try:
            current_grain_weight = max(0.0, min(1.0, float(grain_weight or 0.0)))
        except (TypeError, ValueError):
            current_grain_weight = 0.5

        min_grain_weight = max(
            0.0,
            min(1.0, float(crop_guard_config.get('unprofitable_cash_min_grain_weight', 0.85) or 0.85)),
        )
        disadvantage_margin = max(
            0.0,
            float(crop_guard_config.get('cash_profit_disadvantage_margin_yuan_per_mu', 50.0) or 50.0),
        )
        if current_grain_weight >= min_grain_weight:
            return current_grain_weight

        env = getattr(self.model, 'environment', None)
        if (
            env is None
            or not hasattr(env, 'get_market_price')
            or not hasattr(env, 'get_subsidy_for_enterprise')
        ):
            return current_grain_weight

        grain_price = self._get_output_price_per_kg('crop_price')
        cash_crop_price = self._get_output_price_per_kg('cash_crop')
        daily_wage = env.get_market_price('wage_rate')
        labor_config = self.model.config.get('labor_market', {}).get('labor_requirements', {})
        grain_labor_days = labor_config.get('grain', {}).get('with_machinery', 4)
        cash_labor_days = labor_config.get('cash_crop', {}).get('with_machinery', 21)

        grain_yield = self._estimate_patch_expected_yield_per_mu(CropType.GRAIN)
        cash_crop_yield = self._estimate_patch_expected_yield_per_mu(CropType.CASH_CROP)
        grain_revenue = grain_yield * grain_price
        cash_crop_revenue = cash_crop_yield * cash_crop_price
        grain_profit = (
            grain_revenue
            - grain_revenue * 0.40
            - grain_labor_days * daily_wage
            + env.get_subsidy_for_enterprise(self, 1)
        )
        cash_crop_profit = (
            cash_crop_revenue
            - cash_crop_revenue * 0.45
            - cash_labor_days * daily_wage
        )
        if grain_profit - cash_crop_profit <= disadvantage_margin:
            return current_grain_weight

        self.crop_choice_history.append({
            'period': len(self.crop_choice_history) + 1,
            'phase': 'production_after_land_market',
            'grain_profit': grain_profit,
            'cash_crop_profit': cash_crop_profit,
            'profit_ratio': grain_profit / max(cash_crop_profit, 1),
            'grain_weight': min_grain_weight,
            'cash_weight': max(0.0, 1.0 - min_grain_weight),
            'cash_crop_profitability_guard_applied': True,
            'unprofitable_cash_min_grain_weight': min_grain_weight,
            'cash_profit_disadvantage_margin_yuan_per_mu': disadvantage_margin,
            'planned_crop': CropType.GRAIN if min_grain_weight >= 0.5 else CropType.CASH_CROP,
        })
        return min_grain_weight

    def harvest_and_settle(self):
        """
        @theory
        收获与收入结算 - 基于投资组合
        考虑不同作物的混合种植和收益
        """
        # 获取价格信息
        grain_price = self._get_output_price_per_kg('crop_price')
        cash_crop_price = self._get_output_price_per_kg('cash_crop')
        grain_subsidies_rate = self.model.environment.get_policy_param('grain_subsidies_rate')
        production_config = self.model.config.get('production', {})
        settlement_noise_std = float(
            production_config.get('random_noise', {}).get('settlement_std', 0.02) or 0.02
        )

        def _apply_settlement_noise(expected_value, crop):
            expected_value = float(expected_value or 0.0)
            if expected_value <= 0:
                return 0.0
            if settlement_noise_std <= 0:
                return expected_value
            if self.model.config.get('runtime_fixed_rule', {}).get('national_v1', False):
                return expected_value * paired_yield_noise(self, f"settlement_{crop}", 1.0, settlement_noise_std)
            return expected_value * float(np_normal(self, 1.0, settlement_noise_std))
        
        # 计算实际产量（加入随机波动）
        grain_actual_yield = 0
        cash_crop_actual_yield = 0
        
        if hasattr(self, 'expected_yields'):
            grain_expected = self.expected_yields.get(CropType.GRAIN, 0)
            if grain_expected > 0:
                grain_actual_yield = _apply_settlement_noise(grain_expected, "grain")
                
            cash_crop_expected = self.expected_yields.get(CropType.CASH_CROP, 0)
            if cash_crop_expected > 0:
                cash_crop_actual_yield = _apply_settlement_noise(cash_crop_expected, "cash")
        else:
            # 兼容旧版本
            if self.expected_yield > 0:
                if self.planned_crop == CropType.GRAIN:
                    grain_actual_yield = _apply_settlement_noise(self.expected_yield, "grain")
                else:  # CASH_CROP
                    cash_crop_actual_yield = _apply_settlement_noise(self.expected_yield, "cash")
        
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
        if action_income_enabled(self.model.config):
            self.total_cost += float(getattr(self, "service_expense", 0.0)) + float(getattr(self, "fixed_rule_conversion_expense", 0.0))
        if getattr(self.model, "initial_paid_lease_accounts", ()):
            initial_rent = sum(p.rental_price for p in self._get_active_leased_patches()
                               if getattr(p, "operational_transfer_channel", "") == "author_paid_initial_lease")
            non_initial_cost = self.total_cost - initial_rent
            rent_cash = self.capital + self.total_revenue - non_initial_cost
            if preproduction_cash_enabled(self.model.config):
                rent_cash = self.last_preproduction_cash_account['initial_rent_reserved_yuan']
            paid = settle_initial_paid_rent(self, rent_cash)
            self.land_rental_cost += paid - initial_rent
            self.total_cost = non_initial_cost + paid
        
        # 计算利润和ROI
        self._annual_capital_base_for_roi = self.capital
        self.profit = self.total_revenue - self.total_cost
        if self.capital > 0:
            self.roi = self.profit / self.capital
        else:
            self.roi = 0
        
        # 更新资本
        settle_preproduction_cash(self, self.total_cost)
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
        
        # @fix 2025-12-18: 添加grain_production属性供统一指标收集使用
        # 这个属性记录企业当期的粮食产量（kg）
        self.grain_production = grain_actual_yield
        # @new 2026-01-27: 经济作物产量（kg）
        self.cash_crop_production = cash_crop_actual_yield
        assign_grain_crop_components(
            self,
            model=self.model,
            grain_area_mu=getattr(self, 'grain_area', 0.0),
            grain_output_kg=self.grain_production,
            actor_type=str(getattr(getattr(self, 'enterprise_type', None), 'value', getattr(self, 'enterprise_type', 'enterprise'))),
        )
        
        # 记录详细收益信息
        self.revenues = {
            'grain': grain_revenue,
            'cash_crop': cash_crop_revenue,
            'subsidy': self.subsidy_revenue
        }
        self.submit_subsidy_claim()

    def calculate_subsidy_revenue(self, grain_yield=0):
        """
        @theory
        计算补贴收入（普通种粮补贴 + 规模种粮补贴）
        
        @new 2025-12-08: 增加规模种粮补贴支持
        企业作为规模经营主体，通常更容易达到规模补贴门槛
        
        政策依据：
        1. 浙江省2023年度省级规模种粮动态补贴：50亩以上，20元/亩
        2. 四川省荣县2025年种粮大户补贴：30亩以上，200元/亩
        """
        operated_area = production_area(self)
        grain_weight = self.crop_portfolio.get(CropType.GRAIN, 0.5)
        grain_area = operated_area * grain_weight
        
        self.subsidy_revenue = 0
        self.scale_subsidy_revenue = 0  # @new: 规模补贴收入
        self.scale_subsidy_qualified = False  # @new: 是否达到规模补贴门槛
        if getattr(self.model.environment, "manual_subsidy_dispatch", False):
            return
        
        # 根据政府政策计算补贴
        if grain_area > 0:
            # 使用新的综合补贴计算接口
            subsidy_info = self.model.environment.get_total_subsidy_for_enterprise(self, grain_area)
            self.subsidy_revenue = subsidy_info['total']
            self.scale_subsidy_revenue = subsidy_info['scale_subsidy']
            self.scale_subsidy_qualified = subsidy_info['scale_qualified']
        else:
            self.subsidy_revenue = 0

    def build_subsidy_claim(self):
        """Build a realized enterprise claim after production is observed."""
        from core.p3_project_contracts import configured_program
        if configured_program(self.model) is not None:
            return {'agent': self, 'agent_id': self.unique_id, 'actor_type': 'enterprise'}
        if not getattr(self.model.environment, "manual_subsidy_dispatch", False):
            return None

        active_policy = getattr(self.model.environment, 'active_policy', {}) or {}
        subsidy_target = active_policy.get('subsidy_target', 'operator')
        if subsidy_target == 'owner':
            return None

        operated_area = self.get_operated_land_area()
        grain_area = float(getattr(self, 'grain_area', 0.0) or 0.0)
        operated_area_breakdown = {}
        grain_area_breakdown = {}
        primary_village = None
        if hasattr(self.model, 'get_operated_area_by_village'):
            operated_area_breakdown = self.model.get_operated_area_by_village(self)
            if grain_area > 0:
                grain_area_breakdown = self.model.get_operated_area_by_village(self)
            primary_village = self.model.get_primary_village_for_agent(self)
            if operated_area_breakdown:
                operated_area = sum(float(value or 0.0) for value in operated_area_breakdown.values())
            if grain_area_breakdown:
                grain_area = sum(float(value or 0.0) for value in grain_area_breakdown.values())
        if preproduction_cash_enabled(self.model.config):
            held = self.get_operated_land_area()
            ratio = production_area(self) / held if held > 0 else 0.0
            operated_area = production_area(self)
            grain_area = float(getattr(self, 'grain_area', 0.0))
            operated_area_breakdown = {k: v * ratio for k, v in operated_area_breakdown.items()}
            grain_ratio = grain_area / held if held > 0 else 0.0
            grain_area_breakdown = {k: v * grain_ratio for k, v in grain_area_breakdown.items()}
        if operated_area <= 0 and grain_area <= 0:
            return None

        government = getattr(self.model, 'government', None)
        region = government._get_region_label(self) if government is not None else 'global'
        claim = {
            'agent': self,
            'agent_id': self.unique_id,
            'actor_type': 'enterprise',
            'eligible_area': operated_area,
            'grain_area': grain_area,
            'region': region,
            'primary_village_id': primary_village.unique_id if primary_village is not None else None,
            'village_area_breakdown': dict(operated_area_breakdown),
            'village_grain_area_breakdown': dict(grain_area_breakdown),
            'realized_output': float(getattr(self, 'grain_production', 0.0) or 0.0),
            'scale_qualified': bool(grain_area >= 30),
            'priority_score': 1000.0 + grain_area,
        }
        self.pending_subsidy_claim = claim
        return claim

    def submit_subsidy_claim(self):
        """Submit the realized enterprise claim to the government."""
        government = getattr(self.model, 'government', None)
        if government is None or not hasattr(government, 'register_subsidy_claim'):
            return None
        claim = self.build_subsidy_claim()
        if claim is None:
            self.pending_subsidy_claim = None
            return None
        return government.register_subsidy_claim(claim)

    def apply_verified_subsidy_payment(self, cash_amount, basic_cash=0.0, scale_cash=0.0,
                                       in_kind_value=0.0, in_kind_breakdown=None, payment_record=None):
        """Apply an ex-post verified subsidy payment to the enterprise accounts."""
        self.subsidy_revenue += cash_amount
        self.scale_subsidy_revenue += scale_cash
        self.scale_subsidy_qualified = self.scale_subsidy_qualified or scale_cash > 0
        if in_kind_value > 0:
            self.receive_in_kind_support(in_kind_value, in_kind_breakdown)

        self.total_revenue += cash_amount
        self.profit += cash_amount
        base_capital = max(float(getattr(self, '_annual_capital_base_for_roi', self.capital) or 1.0), 1.0)
        self.roi = self.profit / base_capital
        self.capital += cash_amount
        if not hasattr(self, 'revenues') or not isinstance(self.revenues, dict):
            self.revenues = {}
        self.revenues['subsidy'] = self.subsidy_revenue
        self.subsidy_payment_history.append({
            'step': getattr(self.model, 'step_count', 0),
            'cash_amount': cash_amount,
            'basic_cash': basic_cash,
            'scale_cash': scale_cash,
            'in_kind_value': in_kind_value,
            'claim_id': (payment_record or {}).get('claim_id') if payment_record else None,
        })
        self.pending_subsidy_claim = None

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
        
        if action_income_enabled(self.model.config):
            nonlabor = 0.0
            required_days = 0.0
            for crop, weight in self.crop_portfolio.items():
                preview = copy(self)
                cost = preview._calculate_detailed_cost(crop, production_area(self) * weight)
                nonlabor += cost - preview.hired_labor_cost
                required_days += preview.hired_labor_days
            labor_cost = self.hired_labor_cost if getattr(self, "fixed_rule_labor_settled", False) else required_days * self.model.labor_market.current_agricultural_wage
            self.production_cost = nonlabor + labor_cost
        elif not use_detailed:
            # 使用简单比例（向后兼容）
            grain_cost = (grain_yield * self._get_output_price_per_kg('crop_price')) * 0.5 if grain_yield > 0 else 0
            cash_crop_cost = (cash_crop_yield * self._get_output_price_per_kg('cash_crop')) * 0.4 if cash_crop_yield > 0 else 0
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
        tech_progress = production_config.get('technology_progress', {})
        
        area = max(0.0, float(area or 0.0))
        # 1. 基础产量
        if crop_type == CropType.GRAIN:
            base_yield_per_mu = base_yields.get('grain', 420)
        else:
            base_yield_per_mu = base_yields.get('cash_crop', 1000)  # 修正：从1500改为1000
        
        base_yield = base_yield_per_mu * area
        raw_base_yield = base_yield

        annual_growth = resolve_crop_annual_yield_growth(tech_progress, crop_type)
        current_step = getattr(self.model, 'step_count', 0)
        tech_progress_factor = (1 + annual_growth) ** current_step
        base_yield *= tech_progress_factor

        scale_config = tech_progress.get('scale_efficiency_bonus', {})
        threshold = scale_config.get('threshold_area', 30)
        if area >= 100:
            scale_factor = scale_config.get('large_scale', 1.20)
        elif area >= threshold:
            scale_factor = scale_config.get('medium_scale', 1.10)
        else:
            scale_factor = scale_config.get('small_scale', 1.0)
        scale_progress = max(0.0, min(1.0, (area - float(threshold or 30)) / 70.0))
        base_yield *= scale_factor
        
        # 2. 效率因子影响（企业特有）
        ability_or_efficiency_multiplier = self._get_yield_efficiency_multiplier()
        efficiency_effect = base_yield * ability_or_efficiency_multiplier
        
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
        if self.productivity_shock_enabled:
            tech_multiplier *= (1.0 + self.productivity_tfp_boost)
        tech_effect = soil_effect * tech_multiplier

        management_signal = max(0.0, min(1.0, (float(self.efficiency_factor or 1.0) - 0.8) / 0.5))
        coordination_signal = (
            1.0
            if str(getattr(self.enterprise_type, 'value', self.enterprise_type)).lower().startswith('cooperative')
            else 0.75
        )
        structural_cfg = yield_factors.get('structural_efficiency', {}) or {}
        if crop_type == CropType.GRAIN:
            mechanization_signal = max(
                0.0,
                min(1.0, float(getattr(self, 'grain_mechanization_bonus', 0.20) or 0.20) / 0.25),
            )
            structural_multiplier = (
                1.0
                + float(structural_cfg.get('enterprise_grain_mechanization_weight', 0.08) or 0.0)
                * mechanization_signal
                + float(structural_cfg.get('enterprise_grain_management_weight', 0.06) or 0.0)
                * management_signal
                + float(structural_cfg.get('enterprise_grain_coordination_weight', 0.03) or 0.0)
                * coordination_signal
            )
        else:
            mechanization_signal = max(
                0.0,
                min(1.0, float(getattr(self, 'cash_crop_mechanization_bonus', 0.0) or 0.0) / 0.20),
            )
            structural_multiplier = (
                1.0
                + float(structural_cfg.get('enterprise_cash_mechanization_weight', 0.04) or 0.0)
                * mechanization_signal
                + float(structural_cfg.get('enterprise_cash_management_weight', 0.05) or 0.0)
                * management_signal
            )
        tech_effect *= structural_multiplier
        
        # 5. 气候冲击
        climate_config = yield_factors.get('climate', {})
        climate_volatility = climate_config.get('volatility', 0.1)
        climate_mean = climate_config.get('mean', 0.0)
        climate_shock = self._get_climate_shock(climate_mean, climate_volatility)
        climate_resilience_cfg = yield_factors.get('climate_resilience', {}) or {}
        resilience_buffer = 0.0
        if climate_resilience_cfg.get('enabled', True) and climate_shock < 0:
            resilience_buffer = (
                float(climate_resilience_cfg.get('management_buffer', 0.0) or 0.0) * management_signal
                + float(climate_resilience_cfg.get('coordination_buffer', 0.0) or 0.0) * coordination_signal
                + float(climate_resilience_cfg.get('scale_buffer', 0.0) or 0.0) * mechanization_signal
                + float(climate_resilience_cfg.get('technology_buffer', 0.0) or 0.0)
                * max(0.0, min(1.0, tech_level - base_tech_level))
                + float(climate_resilience_cfg.get('soil_buffer', 0.0) or 0.0) * avg_soil_quality
            )
            resilience_buffer = min(
                float(climate_resilience_cfg.get('max_total_buffer', 0.45) or 0.45),
                max(0.0, resilience_buffer),
            )
            effective_climate_modifier = 1.0 + climate_shock * (1.0 - resilience_buffer)
            minimum_modifier = float(climate_resilience_cfg.get('minimum_yield_modifier', 0.72) or 0.72)
            effective_climate_modifier = max(minimum_modifier, effective_climate_modifier)
        else:
            effective_climate_modifier = 1.0 + climate_shock
        # 6. 随机波动
        noise_std = random_noise.get('std', 0.1)
        if self.model.config.get('runtime_fixed_rule', {}).get('national_v1', False):
            label = "production_grain" if crop_type == CropType.GRAIN else "production_cash"
            random_factor = paired_yield_noise(self, label, 1.0, noise_std)
        else:
            random_factor = float(np_normal(self, 1.0, noise_std))
        final_yield, yield_trace = finalize_yield_stack(
            model=self.model,
            actor_type='enterprise',
            crop_type=crop_type,
            area_mu=area,
            base_yield_per_mu=base_yield_per_mu,
            raw_base_yield_kg=raw_base_yield,
            non_climate_yield_kg=tech_effect,
            tech_progress_factor=tech_progress_factor,
            scale_factor=scale_factor,
            soil_factor=soil_factor,
            ability_or_efficiency_multiplier=ability_or_efficiency_multiplier,
            technology_multiplier=tech_multiplier,
            structural_multiplier=structural_multiplier,
            service_bonus_multiplier=1.0,
            service_signal=0.0,
            mechanization_signal=mechanization_signal,
            scale_progress=scale_progress,
            management_signal=management_signal,
            coordination_signal=coordination_signal,
            climate_shock=climate_shock,
            climate_modifier=effective_climate_modifier,
            resilience_buffer=resilience_buffer,
            random_factor=random_factor,
        )
        trace_key = 'grain' if crop_type == CropType.GRAIN else 'cash_crop'
        if not hasattr(self, 'last_yield_component_traces') or not isinstance(
            getattr(self, 'last_yield_component_traces', None), dict
        ):
            self.last_yield_component_traces = {}
        self.last_yield_component_traces[trace_key] = yield_trace
        self.last_yield_component_trace = yield_trace
        
        return final_yield
    
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
            labor_price = resolve_base_non_farm_wage(
                getattr(self.model, 'config', {}),
                default=DEFAULT_BASE_NON_FARM_WAGE,
            )
        
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
        
        # 生产率冲击与规模经济带来的成本改进
        scale_factor = 0.90  # 企业规模效应：10%成本节约
        if self.productivity_shock_enabled:
            labor_days_required *= max(0.2, 1 - self.productivity_labor_substitution)
            machinery_hours_required *= max(0.5, 1 - self.productivity_input_saving)
            seed_required *= max(0.4, 1 - self.productivity_input_saving)
            fertilizer_required *= max(0.4, 1 - self.productivity_input_saving)
            pesticide_required *= max(0.4, 1 - self.productivity_input_saving)
            other_cost *= max(0.3, 1 - self.productivity_input_saving)
        
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
        leased_patches = self._get_active_leased_patches()
        if len(leased_patches) == 0:
            return 0.75
        return sum(patch.soil_quality for patch in leased_patches) / len(leased_patches)
    
    def _get_technology_level(self):
        """获取技术水平"""
        leased_patches = self._get_active_leased_patches()
        agent_level = float(getattr(self, 'technology_level', 1.0) or 1.0)
        if len(leased_patches) == 0:
            return agent_level
        avg_tech = sum(getattr(patch, 'technology_level', 1.0) for patch in leased_patches) / len(leased_patches)
        return max(agent_level, avg_tech)
    
    def _get_climate_shock(self, mean, volatility):
        """获取气候冲击"""
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'get_climate_shock'):
            return self.model.environment.get_climate_shock()
        else:
            return float(np_normal(self, mean, volatility))

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
        leased_patches = self._get_active_leased_patches()
        
        # @fix 2025-11-29: 修正租金计算
        # rental_price已经是该地块的总租金（已包含面积），无需再乘面积
        self.land_rental_cost = sum(
            patch.rental_price 
            for patch in leased_patches 
            if patch.rental_price > 0
        )
        budget = getattr(self, 'last_preproduction_cash_account', {})
        if preproduction_cash_enabled(self.model.config) and budget.get('step') == self.model.step_count:
            initial = sum(p.rental_price for p in leased_patches
                          if getattr(p, 'operational_transfer_channel', '') == 'author_paid_initial_lease')
            self.land_rental_cost = budget['ordinary_rent_reserved_yuan'] + initial

    def calculate_management_cost(self):
        """
        @theory
        计算管理成本
        """
        # 管理成本与经营面积成正比
        operated_area = production_area(self)
        diseconomy_config = (
            self.model.config.get('agents', {})
            .get('enterprise', {})
            .get('diseconomy', {})
        )
        supervision_cost_base = float(diseconomy_config.get('supervision_cost_base', 50.0) or 50.0)
        self.management_cost = operated_area * supervision_cost_base

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
        enterprise_cfg = self.model.config.get('agents', {}).get('enterprise', {})
        adapt_cfg = enterprise_cfg.get('return_rate_adaptation', {}) or {}
        base_target = max(
            0.02,
            float(getattr(self, 'base_target_return_rate', getattr(self, 'target_return_rate', 0.1)) or 0.1),
        )
        current_target = max(0.02, float(getattr(self, 'target_return_rate', base_target) or base_target))

        positive_trend_step = float(adapt_cfg.get('positive_trend_step', 0.02) or 0.02)
        negative_trend_step = float(adapt_cfg.get('negative_trend_step', 0.04) or 0.04)
        low_capital_drag = float(adapt_cfg.get('low_capital_drag', 0.02) or 0.02)
        mean_reversion = float(adapt_cfg.get('mean_reversion', 0.12) or 0.12)
        floor_ratio = float(adapt_cfg.get('floor_ratio', 0.70) or 0.70)
        ceiling_ratio = float(adapt_cfg.get('ceiling_ratio', 1.25) or 1.25)

        target = current_target
        if self.performance_trend > 0.5:
            target *= (1.0 + positive_trend_step)
        elif self.performance_trend < -0.5:
            target *= max(0.0, 1.0 - negative_trend_step)

        if self.capital < 50000:
            target *= max(0.0, 1.0 - low_capital_drag)

        target = (1.0 - mean_reversion) * target + mean_reversion * base_target

        lower_bound = max(0.02, base_target * floor_ratio)
        upper_bound = max(lower_bound, base_target * ceiling_ratio)
        self.target_return_rate = min(max(target, lower_bound), upper_bound)

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
        @fix 2026-06: 经营面积以 current_operator/leased_to 地块集合为准，
        不能用 total_leased_area 的旧值提前返回，否则会漏掉经营权已登记但租赁列表未同步的地块。
        """
        if hasattr(self, "_fixed_rule_preview_area"):
            return self._fixed_rule_preview_area
        if self.model.config.get('runtime_fixed_rule', {}).get('initial_tenure_mode') in ('B_star_1y', 'B_star_5y'):
            return sum(float(p.area) for p in self._get_active_leased_patches(include_current_operator=True)
                       if p.current_operator is self
                       or getattr(p, 'operational_transfer_channel', '') != 'author_paid_initial_lease')
        model = getattr(self, 'model', None)
        cache_version = getattr(model, 'land_tenure_version', None)
        cached_version = getattr(self, '_operated_land_area_cache_version', None)
        if cache_version is not None and cached_version == cache_version:
            return float(getattr(self, '_operated_land_area_cache', 0.0) or 0.0)

        operated_patches = self._get_active_leased_patches(include_current_operator=True)
        area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in operated_patches)
        self._operated_land_area_cache = area
        self._operated_land_area_cache_version = cache_version
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

    def export_sar(self) -> dict:
        """Export (state, action) for SAR logging.

        Enterprise agents are currently rule-based (policy_mode='rule').
        SAR export records the profit-driven decision state for future learner integration.
        Reward is not computed here — it must be captured by the SAR logger after market clearing.
        """
        return {
            'state': {
                'operated_area_mu': float(getattr(self, 'get_operated_land_area', lambda: 0.0)() or 0.0),
                'crop_choice': str(getattr(self, 'crop_choice', None)),
                'enterprise_type': str(getattr(self, 'enterprise_type', None)),
                'equity': float(getattr(self, 'equity', 0.0) or 0.0),
                'cash': float(getattr(self, 'cash', 0.0) or 0.0),
                'annual_profit': float(getattr(self, 'annual_profit', 0.0) or 0.0),
            },
            'action': {
                'land_demand_mu': float(getattr(self, 'total_leased_area', 0.0) or 0.0),
                'bid_price': float(getattr(self, '_last_bid_price', 0.0) or 0.0),
            },
            'policy_mode': 'rule',
            'agent_type': 'enterprise',
        }

    def receive_in_kind_support(self, value, breakdown=None):
        if value <= 0:
            return
        self.in_kind_support_balance += value
        if breakdown:
            for key, amount in breakdown.items():
                self.in_kind_support_breakdown[key] = self.in_kind_support_breakdown.get(key, 0) + amount
