# -*- coding: utf-8 -*-
"""
@module expectation_mechanism.py
@description 理性预期机制 - 所有主体采用理性预期（Rational Expectations Hypothesis）
@created 2025-12-04
@updated 2025-12-01: 全面升级为理性预期框架

理论依据：
1. 理性预期假说(Rational Expectations Hypothesis, REH)
   - Muth (1961): "Rational Expectations and the Theory of Price Movements"
   - Lucas (1972): "Expectations and the Neutrality of Money"
   - Sargent & Wallace (1975): 政策无效性命题
   
2. 核心假设：
   - 主体使用所有可获得的信息形成预期
   - 预期是模型一致的（model-consistent）
   - 平均而言预期是正确的：E[P_t+1 | I_t] = P_t+1 + ε
   - 预期误差是随机的，不存在系统性偏差
   
3. 农业市场中的理性预期：
   - Nerlove (1958) 蛛网模型中的预期形成
   - Chavas & Holt (1990): 农产品期货市场的理性预期
   - Gardner (1992): 农业政策与理性预期
   
4. 实现方式：
   - 使用供需模型的均衡解作为预期
   - 纳入政策信号的一阶效应
   - 考虑历史波动性估计预期方差

核心功能：
1. 所有主体统一使用理性预期机制
2. 提供基于模型的价格、产量、补贴预期
3. 追踪预期误差验证理性预期假说
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from core.config_utils import DEFAULT_BASE_NON_FARM_WAGE


class ExpectationType(Enum):
    """预期类型枚举"""
    ADAPTIVE = "adaptive"           # 适应性预期（后向看）
    NAIVE = "naive"                 # 朴素预期（下期=本期）
    RATIONAL = "rational"           # 理性预期（前向看）
    BOUNDED_RATIONAL = "bounded"    # 有限理性
    EXTRAPOLATIVE = "extrapolative" # 外推式预期（趋势延续）


@dataclass
class ExpectationState:
    """预期状态"""
    expected_grain_price: float = 0.0
    expected_cash_crop_price: float = 0.0
    expected_subsidy_rate: float = 0.0
    expected_wage: float = 0.0
    expected_rental_price: float = 0.0
    
    # 预期误差追踪（用于学习调整）
    grain_price_errors: List[float] = field(default_factory=list)
    cash_price_errors: List[float] = field(default_factory=list)
    subsidy_errors: List[float] = field(default_factory=list)
    
    # 学习参数
    learning_rate: float = 0.3
    max_error_history: int = 5


class ExpectationMechanism(ABC):
    """预期形成机制抽象基类"""
    
    @abstractmethod
    def form_expectation(self, 
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """
        形成预期
        
        Args:
            historical_values: 历史值列表（从旧到新）
            current_value: 当前值
            additional_info: 额外信息（如政策信号）
            
        Returns:
            预期值
        """
        pass
    
    def update_learning(self, expected: float, realized: float) -> float:
        """
        更新学习参数
        
        Args:
            expected: 预期值
            realized: 实现值
            
        Returns:
            预期误差
        """
        return realized - expected


class AdaptiveExpectation(ExpectationMechanism):
    """
    适应性预期
    
    理论：
    - 预期根据过去的预期误差进行调整
    - E[P_t+1] = E[P_t] + λ * (P_t - E[P_t])
    - λ 是学习参数，反映主体对新信息的敏感度
    
    适用：小农户、信息受限主体
    """
    
    def __init__(self, adjustment_speed: float = 0.3):
        """
        初始化
        
        Args:
            adjustment_speed: 调整速度 λ，0-1之间
        """
        self.adjustment_speed = adjustment_speed
        self.last_expectation = None
    
    def form_expectation(self,
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """形成适应性预期"""
        if self.last_expectation is None:
            # 首次预期：使用历史均值
            if historical_values:
                self.last_expectation = np.mean(historical_values[-3:])
            else:
                self.last_expectation = current_value
        
        # 适应性调整
        error = current_value - self.last_expectation
        new_expectation = self.last_expectation + self.adjustment_speed * error
        
        self.last_expectation = new_expectation
        return new_expectation


class NaiveExpectation(ExpectationMechanism):
    """
    朴素预期
    
    理论：
    - 预期下一期等于当前期
    - E[P_t+1] = P_t
    
    适用：短期决策、信息极度受限
    """
    
    def form_expectation(self,
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """形成朴素预期"""
        return current_value


class RationalExpectation(ExpectationMechanism):
    """
    完整理性预期机制（Rational Expectations Hypothesis）
    
    理论依据：
    - Muth (1961): 预期与经济模型一致
    - Lucas (1972): 主体利用所有可得信息，预期无系统偏差
    - Sargent (1987): 理性预期的动态宏观模型
    
    核心特征：
    1. 模型一致性：预期基于对经济模型结构的理解
    2. 无偏性：E[P_{t+1} | I_t] = P_{t+1}（平均而言预期正确）
    3. 效率性：使用所有可得信息，不存在可预测的预期误差
    
    农业市场应用：
    - 理解供需弹性对价格的影响
    - 考虑政府干预（最低收购价）的锚定效应
    - 纳入季节性、政策周期性因素
    
    适用：所有主体（符合Lucas批判的要求）
    
    @literature_basis 2025-12-26: 供需弹性参数文献支持
    供给弹性 (supply_elasticity = 0.5):
      - Luo (Nerlove Model Application): China grain short-run supply elasticity = 0.3-0.5
      - Jiang (2018): 中国粮食供给弹性较低，长期略高于短期
      - Wei & Wang (VEC Model): 长期正向关系，短期调整缓慢
      - 本模型取值 0.5：位于文献共识范围上限（体现规模化农业趋势）
    
    需求弹性 (demand_elasticity = -0.4):
      - Huang & Rozelle (1995): aggregate -0.52
      - Hus et al. (2002): Urban -0.16, Rural -0.37
      - 本模型取值 -0.4：略高于农村估计，反映市场化程度提升
    
    DOI References:
      - 10.3390/agriculture11010040 (Supply elasticity meta-analysis)
      - 10.1016/j.foodpol.2002.08.005 (Hus demand elasticity)
    """
    
    def __init__(self, 
                 supply_elasticity: float = 0.5,
                 demand_elasticity: float = -0.4,
                 floor_price_grain: float = 2.58,
                 policy_credibility: float = 0.9,
                 volatility_persistence: float = 0.7):
        """
        初始化理性预期参数
        
        @literature_basis:
        - supply_elasticity=0.5: Luo/Jiang Nerlove model (0.3-0.5)
        - demand_elasticity=-0.4: Hus et al. 2002 China meta (-0.16 to -0.52)
        - floor_price_grain=2.58: NDRC 2023 小麦最低收购价
        
        Args:
            supply_elasticity: 供给价格弹性（正值），文献范围 0.3-0.5
            demand_elasticity: 需求价格弹性（负值），文献范围 -0.16 to -0.52
            floor_price_grain: 粮食最低收购价（政府锚），NDRC公布价
            policy_credibility: 政策可信度（0-1）
            volatility_persistence: 波动性持续性（GARCH效应）
        """
        self.supply_elasticity = supply_elasticity
        self.demand_elasticity = demand_elasticity
        self.floor_price_grain = floor_price_grain
        self.policy_credibility = policy_credibility
        self.volatility_persistence = volatility_persistence
        
        # 预期误差历史（用于验证理性预期假说）
        self.prediction_errors = []
        self.last_prediction = None
    
    def form_expectation(self,
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """
        形成理性预期
        
        理性预期的核心：利用模型结构和所有可得信息
        
        模型：P* = f(供给, 需求, 政策)
        预期：E[P_{t+1}] = E[f(S_{t+1}, D_{t+1}, Policy_{t+1})]
        
        实现步骤：
        1. 估计市场供需缺口
        2. 计算政策干预的影响
        3. 考虑历史波动性（条件方差）
        4. 形成无偏预期
        """
        if not historical_values or len(historical_values) < 2:
            return current_value
        
        # ========== 1. 供需均衡分析 ==========
        # 估计均衡价格（长期均值代理）
        long_term_equilibrium = np.mean(historical_values)
        
        # 当前偏离程度
        current_deviation = (current_value - long_term_equilibrium) / long_term_equilibrium
        
        # ========== 2. 蛛网模型动态 ==========
        # 蛛网比率：|供给弹性 / 需求弹性|
        # < 1: 收敛蛛网；> 1: 发散蛛网
        cobweb_ratio = abs(self.supply_elasticity / self.demand_elasticity)
        
        # 预期价格变化（蛛网动态修正）
        if cobweb_ratio < 1:
            # 收敛蛛网：价格将向均衡回归
            expected_change = -current_deviation * (1 - cobweb_ratio) * 0.5
        else:
            # 发散蛛网：使用阻尼因子防止发散
            damping = 0.3
            expected_change = -current_deviation * damping
        
        # ========== 3. 政策锚定效应 ==========
        policy_effect = 0.0
        if additional_info:
            # 检查是否是粮食市场（有最低收购价）
            is_grain = additional_info.get('crop_type') == 'grain'
            
            if is_grain:
                # 粮食价格受最低收购价锚定
                floor_price = additional_info.get('floor_price', self.floor_price_grain)
                if current_value < floor_price * 1.1:
                    # 接近地板价，政府将托市
                    policy_effect = (floor_price - current_value) * self.policy_credibility
            
            # 补贴政策信号
            subsidy_change = additional_info.get('subsidy_change', 0)
            policy_effect += subsidy_change * 0.05  # 补贴变化的价格传导
        
        # ========== 4. 波动性估计（条件方差） ==========
        if len(historical_values) >= 3:
            returns = np.diff(historical_values) / historical_values[:-1]
            historical_vol = np.std(returns)
            # GARCH(1,1)风格的波动率更新
            expected_vol = self.volatility_persistence * historical_vol
        else:
            expected_vol = 0.05
        
        # ========== 5. 形成理性预期 ==========
        # 基础预期 = 当前值 × (1 + 预期变化率)
        base_expectation = current_value * (1 + expected_change)
        
        # 加入政策效应
        final_expectation = base_expectation + policy_effect
        
        # 记录预期用于事后验证
        self.last_prediction = final_expectation
        
        return float(final_expectation)
    
    def verify_rationality(self, realized_value: float) -> Dict[str, float]:
        """
        验证理性预期假说
        
        如果预期是理性的：
        1. 预期误差均值应为0（无偏性）
        2. 预期误差应不可预测（正交性）
        """
        if self.last_prediction is None:
            return {}
        
        error = realized_value - self.last_prediction
        self.prediction_errors.append(error)
        
        # 限制历史长度
        if len(self.prediction_errors) > 50:
            self.prediction_errors = self.prediction_errors[-50:]
        
        # 计算统计量
        result = {
            'current_error': error,
            'error_pct': abs(error) / realized_value if realized_value != 0 else 0
        }
        
        if len(self.prediction_errors) >= 5:
            errors = np.array(self.prediction_errors)
            result['mean_error'] = float(np.mean(errors))
            result['error_std'] = float(np.std(errors))
            result['is_unbiased'] = abs(result['mean_error']) < 2 * result['error_std'] / np.sqrt(len(errors))
        
        return result


class ExtrapolativeExpectation(ExpectationMechanism):
    """
    外推式预期
    
    理论：
    - 将当前趋势线性外推到未来
    - 容易导致过度乐观或悲观
    
    适用：新进入者、缺乏经验的规模户
    警告：这种预期可能导致"羊群效应"
    """
    
    def __init__(self, extrapolation_factor: float = 1.0):
        """
        初始化
        
        Args:
            extrapolation_factor: 外推系数，>1表示过度外推
        """
        self.extrapolation_factor = extrapolation_factor
    
    def form_expectation(self,
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """形成外推式预期"""
        if not historical_values:
            return current_value
        
        # 计算最近的趋势
        prev_value = historical_values[-1] if historical_values else current_value
        trend = current_value - prev_value
        
        # 外推
        expectation = current_value + trend * self.extrapolation_factor
        
        return expectation


class BoundedRationalExpectation(ExpectationMechanism):
    """
    有限理性预期
    
    理论（Simon, Kahneman）：
    - 使用简单启发式规则
    - 介于适应性和理性之间
    - 受认知限制和信息处理能力约束
    
    适用：家庭农场、中等规模经营者
    """
    
    def __init__(self, 
                 attention_span: int = 3,
                 anchor_weight: float = 0.5):
        """
        初始化
        
        Args:
            attention_span: 注意力跨度（考虑多少期历史）
            anchor_weight: 锚定效应权重
        """
        self.attention_span = attention_span
        self.anchor_weight = anchor_weight
        self.anchor_value = None
    
    def form_expectation(self,
                         historical_values: List[float],
                         current_value: float,
                         additional_info: Dict[str, Any] = None) -> float:
        """形成有限理性预期"""
        # 1. 设置锚点（首次接触的值）
        if self.anchor_value is None:
            self.anchor_value = current_value
        
        # 2. 有限注意力：只看最近几期
        if historical_values:
            relevant_history = historical_values[-self.attention_span:]
            historical_avg = np.mean(relevant_history)
        else:
            historical_avg = current_value
        
        # 3. 锚定与调整
        adjusted_value = (
            self.anchor_weight * self.anchor_value +
            (1 - self.anchor_weight) * historical_avg
        )
        
        # 4. 可得性启发：当前值权重更高（容易回想）
        expectation = 0.6 * current_value + 0.4 * adjusted_value
        
        return expectation


@dataclass
class AgentExpectationProfile:
    """
    主体预期参数配置 - 实现异质性预期
    
    参考spatial-climate-ABM的distance_cost异质性模式
    每个主体根据其特征获得差异化的预期参数
    
    @new 2025-12-05: 细化不同主体类型的预期参数差异
    """
    # 基础经济参数
    supply_elasticity: float = 0.5      # 供给价格弹性
    demand_elasticity: float = -0.4     # 需求价格弹性
    
    # 政策相关参数
    policy_credibility: float = 0.85    # 政策可信度 [0-1]
    policy_attention: float = 0.8       # 政策关注度 [0-1]
    
    # 波动性与风险参数
    volatility_persistence: float = 0.7  # 波动持续性 (GARCH效应)
    risk_aversion: float = 0.5          # 风险厌恶系数 [0-1]
    
    # 信息处理能力
    information_efficiency: float = 0.8  # 信息处理效率 [0-1]
    learning_rate: float = 0.3          # 学习调整速度
    memory_span: int = 5                # 历史记忆跨度（期）
    
    # 社会网络影响
    social_influence: float = 0.3       # 社会网络影响权重
    herding_tendency: float = 0.2       # 羊群效应倾向
    
    # 认知偏差参数（前景理论）
    loss_aversion: float = 1.95         # 损失厌恶系数 (λ) @updated: Brown et al. (2024) AER
    probability_weighting: float = 0.65  # 概率权重参数 (γ)
    reference_dependence: float = 0.5   # 参考点依赖强度


def _calculate_farmer_expectation_profile(
    farmer_type: str,
    age: int,
    education_years: int,
    land_scale: float,
    risk_preference: float,
    social_capital: float = 0.5,
    experience_years: int = 10
) -> AgentExpectationProfile:
    """
    计算农户的异质性预期参数配置
    
    @new 2025-12-05: 实现基于农户特征的参数细化
    
    理论依据：
    1. 年龄效应 (Lusardi & Mitchell, 2014)
       - 老年人金融素养较低，信息处理能力下降
       - 但经验积累可部分弥补
    
    2. 教育效应 (Huffman, 2001)
       - 教育提高信息处理能力和理性程度
       - 但农业实践经验同样重要
    
    3. 规模效应 (Griliches, 1957)
       - 大规模农户更关注市场价格信号
       - 有更强的价格谈判能力
    
    4. 风险偏好 (Binswanger, 1980)
       - 风险厌恶农户更保守，波动容忍度低
       - 风险偏好农户对价格信号更敏感
    
    5. 社会资本 (Coleman, 1988)
       - 高社会资本意味着更好的信息获取
       - 也可能导致更强的羊群效应
    
    Args:
        farmer_type: 农户类型 (Pure_Farmer, Pluriactive, Non_Farmer)
        age: 年龄
        education_years: 受教育年限
        land_scale: 经营规模（亩）
        risk_preference: 风险偏好 (0=极度厌恶, 1=风险中性)
        social_capital: 社会资本水平
        experience_years: 务农经验年限
    
    Returns:
        AgentExpectationProfile: 个性化参数配置
    """
    import numpy as np
    
    profile = AgentExpectationProfile()
    
    # ========== 1. 年龄效应 ==========
    # 年龄对信息处理能力的影响（倒U型曲线）
    age_efficiency_factor = 1.0 - 0.01 * abs(age - 42) ** 1.2 / 10
    age_efficiency_factor = np.clip(age_efficiency_factor, 0.5, 1.0)
    
    # 年龄对波动容忍度的影响（老年更保守）
    if age > 55:
        profile.volatility_persistence = 0.85  # 高持续性 = 更保守
        profile.risk_aversion = min(0.8, 0.5 + (age - 55) * 0.02)
    elif age < 35:
        profile.volatility_persistence = 0.55
        profile.risk_aversion = max(0.3, 0.5 - (35 - age) * 0.015)
    else:
        profile.volatility_persistence = 0.7
        profile.risk_aversion = 0.5
    
    # ========== 2. 教育效应 ==========
    # 教育对信息处理效率的提升
    edu_efficiency = 0.6 + 0.03 * min(education_years, 16)  # 最高12年提升到0.96
    profile.information_efficiency = min(0.98, edu_efficiency * age_efficiency_factor)
    
    # 教育对学习率的影响
    profile.learning_rate = 0.2 + 0.02 * min(education_years, 12)
    
    # 高教育降低羊群效应
    profile.herding_tendency = max(0.1, 0.35 - 0.015 * education_years)
    
    # ========== 3. 规模效应 ==========
    # 大规模农户供给弹性更高（对价格更敏感）
    if land_scale > 50:  # 大规模
        scale_factor = min(1.5, 1.0 + (land_scale - 50) / 200)
        profile.supply_elasticity = 0.5 * scale_factor
        profile.policy_attention = min(0.95, 0.7 + land_scale / 500)
    elif land_scale > 15:  # 中等规模
        profile.supply_elasticity = 0.5 + (land_scale - 15) / 100
        profile.policy_attention = 0.7 + (land_scale - 15) / 150
    else:  # 小规模
        profile.supply_elasticity = 0.35 + land_scale / 100
        profile.policy_attention = 0.6 + land_scale / 50
    
    # ========== 4. 风险偏好效应 ==========
    # 转换risk_preference (0-1) 到前景理论参数
    profile.loss_aversion = 2.75 - 0.5 * risk_preference  # [2.25, 2.75]
    profile.probability_weighting = 0.55 + 0.2 * risk_preference  # [0.55, 0.75]
    profile.reference_dependence = 0.6 - 0.2 * risk_preference  # [0.4, 0.6]
    
    # ========== 5. 农户类型效应 ==========
    if farmer_type == 'Pure_Farmer':
        # 纯农户：高政策关注，高农业经验
        profile.policy_credibility = 0.95
        profile.memory_span = 5 + min(experience_years // 5, 3)  # 经验增加记忆
        profile.social_influence = 0.35  # 村庄社会网络影响较强
    elif farmer_type == 'Pluriactive':
        # 兼业户：注意力分散
        profile.policy_credibility = 0.80
        profile.policy_attention *= 0.85  # 注意力分散
        profile.information_efficiency *= 0.9
        profile.memory_span = 4
        profile.social_influence = 0.25
    else:  # Non_Farmer
        # 非农户：关注非农市场
        profile.policy_credibility = 0.70
        profile.policy_attention *= 0.6
        profile.memory_span = 3
        profile.social_influence = 0.15
    
    # ========== 6. 社会资本效应 ==========
    # 高社会资本提高信息效率但也增加羊群效应
    profile.information_efficiency *= (0.9 + 0.2 * social_capital)
    profile.information_efficiency = min(0.98, profile.information_efficiency)
    
    profile.herding_tendency += 0.15 * social_capital  # 社会资本增加从众倾向
    profile.herding_tendency = min(0.5, profile.herding_tendency)
    
    # ========== 7. 经验效应 ==========
    # 经验改进供给弹性估计准确度
    if experience_years > 20:
        profile.supply_elasticity *= 1.15
    elif experience_years < 5:
        profile.supply_elasticity *= 0.85
    
    return profile


def _calculate_enterprise_expectation_profile(
    enterprise_type: str,
    scale: float,
    years_in_operation: int,
    capital_intensity: float = 0.5,
    market_orientation: str = 'domestic'
) -> AgentExpectationProfile:
    """
    计算企业的异质性预期参数配置
    
    @new 2025-12-05: 实现基于企业特征的参数细化
    
    理论依据：
    1. Melitz (2003) 异质性企业
       - 生产率差异导致市场参与差异
       - 高生产率企业对价格更敏感
    
    2. 企业学习曲线 (Arrow, 1962)
       - 运营时间增加经验积累
       - 降低预期误差
    
    3. 规模经济与信息
       - 大企业有专业分析团队
       - 信息处理能力更强
    """
    import numpy as np
    
    profile = AgentExpectationProfile()
    
    # ========== 1. 规模效应 ==========
    if scale > 1000:  # 大型企业
        profile.supply_elasticity = 0.75
        profile.information_efficiency = 0.95
        profile.policy_attention = 0.95
        profile.learning_rate = 0.4
    elif scale > 500:  # 中型企业
        profile.supply_elasticity = 0.6
        profile.information_efficiency = 0.85
        profile.policy_attention = 0.85
        profile.learning_rate = 0.35
    else:  # 小型企业
        profile.supply_elasticity = 0.5
        profile.information_efficiency = 0.75
        profile.policy_attention = 0.75
        profile.learning_rate = 0.3
    
    # ========== 2. 企业类型效应 ==========
    if enterprise_type in ['AGRI_CORPORATION', 'Agri_Corporation']:
        # 农业公司：更理性，更低风险厌恶
        profile.policy_credibility = 0.95
        profile.risk_aversion = 0.35
        profile.volatility_persistence = 0.55
        profile.herding_tendency = 0.15  # 独立决策
    else:  # COOPERATIVE
        # 合作社：更保守，社会影响更大
        profile.policy_credibility = 0.88
        profile.risk_aversion = 0.55
        profile.volatility_persistence = 0.7
        profile.social_influence = 0.4
        profile.herding_tendency = 0.25  # 成员间有羊群效应
    
    # ========== 3. 运营年限效应 ==========
    experience_bonus = min(0.15, years_in_operation * 0.02)
    profile.information_efficiency = min(0.98, profile.information_efficiency + experience_bonus)
    profile.memory_span = 5 + min(years_in_operation, 5)
    
    # ========== 4. 资本密集度效应 ==========
    # 高资本密集度企业对价格波动更敏感
    profile.loss_aversion = 2.0 + 0.5 * capital_intensity
    
    # ========== 5. 市场导向效应 ==========
    if market_orientation == 'export':
        profile.demand_elasticity = -0.5  # 出口市场更弹性
        profile.volatility_persistence = 0.6  # 更能应对波动
    else:
        profile.demand_elasticity = -0.35
    
    return profile


class ExpectationFactory:
    """
    预期机制工厂
    
    @updated 2025-12-01: 所有主体统一采用理性预期
    @updated 2025-12-05: 增加基于主体特征的参数细化
    
    理论依据（Lucas批判）：
    - 政策分析必须基于理性预期
    - 适应性预期在政策变化时会产生系统性偏差
    - ABM模型作为政策实验室，应满足Lucas批判
    
    中国农业市场的理性预期支撑：
    1. 信息化程度提升：智能手机普及率超95%
    2. 市场意识增强：农户关注期货价格、政策动向
    3. 社会网络传播：信息在村庄内快速扩散
    4. 政府信息公开：最低收购价提前公布
    
    异质性设计（参考spatial-climate-ABM的distance_cost模式）：
    - 每个主体根据特征获得差异化参数
    - 参数差异影响预期形成过程
    - 实现从微观异质性到宏观涌现
    """
    
    @staticmethod
    def create_for_farmer(farmer_type: str,
                          age: int = 45,
                          experience_years: int = 10,
                          education_years: int = 9,
                          land_scale: float = 10.0,
                          risk_preference: float = 0.5,
                          social_capital: float = 0.5) -> ExpectationMechanism:
        """
        为农户创建理性预期机制
        
        @updated 2025-12-01: 所有农户统一使用理性预期
        @updated 2025-12-05: 增加更多特征参数实现异质性
        
        理论依据：
        1. Muth (1961): 理性预期假说
        2. 中国农户市场化程度高，关注价格信息
        3. 政府最低收购价政策提供价格锚
        
        Args:
            farmer_type: 农户类型 (Pure_Farmer, Pluriactive, Non_Farmer)
            age: 年龄（影响波动性容忍度）
            experience_years: 务农年限（影响模型校准）
            education_years: 受教育年限（影响信息处理）
            land_scale: 经营规模（亩）
            risk_preference: 风险偏好 (0-1)
            social_capital: 社会资本水平
            
        Returns:
            RationalExpectation实例（带个性化参数）
        """
        # 计算个性化参数配置
        profile = _calculate_farmer_expectation_profile(
            farmer_type=farmer_type,
            age=age,
            education_years=education_years,
            land_scale=land_scale,
            risk_preference=risk_preference,
            social_capital=social_capital,
            experience_years=experience_years
        )
        
        # 使用计算出的参数创建RationalExpectation实例
        mechanism = RationalExpectation(
            supply_elasticity=profile.supply_elasticity,
            demand_elasticity=profile.demand_elasticity,
            floor_price_grain=2.58,
            policy_credibility=profile.policy_credibility,
            volatility_persistence=profile.volatility_persistence
        )
        
        # 附加额外参数供后续使用
        mechanism.profile = profile
        mechanism.agent_type = 'farmer'
        mechanism.agent_subtype = farmer_type
        
        return mechanism
    
    @staticmethod
    def create_for_enterprise(enterprise_type: str,
                               scale: float = 500,
                               years_in_operation: int = 5,
                               capital_intensity: float = 0.5,
                               market_orientation: str = 'domestic') -> ExpectationMechanism:
        """
        为企业创建理性预期机制
        
        @updated 2025-12-01: 所有企业统一使用理性预期
        @updated 2025-12-05: 增加更多特征参数实现异质性
        
        理论依据：
        1. 企业有更强的信息获取和分析能力
        2. 规模化经营需要准确的市场预判
        3. 资本投入大，决策必须基于理性分析
        
        Args:
            enterprise_type: 企业类型 (AGRI_CORPORATION, COOPERATIVE)
            scale: 经营规模（亩）
            years_in_operation: 运营年限
            capital_intensity: 资本密集度
            market_orientation: 市场导向 (domestic, export)
            
        Returns:
            RationalExpectation实例（带个性化参数）
        """
        # 计算个性化参数配置
        profile = _calculate_enterprise_expectation_profile(
            enterprise_type=enterprise_type,
            scale=scale,
            years_in_operation=years_in_operation,
            capital_intensity=capital_intensity,
            market_orientation=market_orientation
        )
        
        # 使用计算出的参数创建RationalExpectation实例
        mechanism = RationalExpectation(
            supply_elasticity=profile.supply_elasticity,
            demand_elasticity=profile.demand_elasticity,
            floor_price_grain=2.58,
            policy_credibility=profile.policy_credibility,
            volatility_persistence=profile.volatility_persistence
        )
        
        # 附加额外参数供后续使用
        mechanism.profile = profile
        mechanism.agent_type = 'enterprise'
        mechanism.agent_subtype = enterprise_type
        
        return mechanism
    
    @staticmethod
    def create_for_village_collective(
        village_size: int = 300,
        collective_assets: float = 1000000,
        governance_quality: float = 0.7
    ) -> ExpectationMechanism:
        """
        为村集体创建理性预期机制
        
        @new 2025-12-05: 村集体专用预期机制
        
        Args:
            village_size: 村庄规模（户数）
            collective_assets: 集体资产（元）
            governance_quality: 治理质量 (0-1)
        """
        profile = AgentExpectationProfile()
        
        # 大村庄有更好的信息汇聚
        profile.information_efficiency = 0.7 + 0.1 * min(village_size / 500, 1.0)
        
        # 治理质量影响政策可信度
        profile.policy_credibility = 0.75 + 0.2 * governance_quality
        
        # 集体资产影响风险态度
        if collective_assets > 2000000:
            profile.risk_aversion = 0.4  # 资产雄厚，风险承受力强
        else:
            profile.risk_aversion = 0.6
        
        profile.supply_elasticity = 0.5
        profile.volatility_persistence = 0.7
        profile.social_influence = 0.5  # 集体决策受社会因素影响大
        
        mechanism = RationalExpectation(
            supply_elasticity=profile.supply_elasticity,
            demand_elasticity=-0.4,
            floor_price_grain=2.58,
            policy_credibility=profile.policy_credibility,
            volatility_persistence=profile.volatility_persistence
        )
        
        mechanism.profile = profile
        mechanism.agent_type = 'village_collective'
        
        return mechanism
    
    @staticmethod
    def create_by_type(expectation_type: ExpectationType, **kwargs) -> ExpectationMechanism:
        """
        根据类型创建预期机制
        
        Args:
            expectation_type: 预期类型枚举
            **kwargs: 额外参数
            
        Returns:
            ExpectationMechanism实例
        """
        if expectation_type == ExpectationType.ADAPTIVE:
            return AdaptiveExpectation(**kwargs)
        elif expectation_type == ExpectationType.NAIVE:
            return NaiveExpectation()
        elif expectation_type == ExpectationType.RATIONAL:
            return RationalExpectation(**kwargs)
        elif expectation_type == ExpectationType.BOUNDED_RATIONAL:
            return BoundedRationalExpectation(**kwargs)
        elif expectation_type == ExpectationType.EXTRAPOLATIVE:
            return ExtrapolativeExpectation(**kwargs)
        else:
            return AdaptiveExpectation()


class ExpectationManager:
    """
    预期管理器
    
    为主体管理多种预期（价格、补贴、工资等）
    """
    
    def __init__(self, base_mechanism: ExpectationMechanism):
        """
        初始化
        
        Args:
            base_mechanism: 基础预期机制
        """
        self.mechanism = base_mechanism
        self.state = ExpectationState()
        
        # 历史数据
        self.price_history = {'grain': [], 'cash_crop': []}
        self.subsidy_history = []
        self.wage_history = []
        self.rental_history = []
    
    def update_history(self,
                       grain_price: float = None,
                       cash_price: float = None,
                       subsidy_rate: float = None,
                       wage: float = None,
                       rental_price: float = None):
        """更新历史数据"""
        if grain_price is not None:
            self.price_history['grain'].append(grain_price)
        if cash_price is not None:
            self.price_history['cash_crop'].append(cash_price)
        if subsidy_rate is not None:
            self.subsidy_history.append(subsidy_rate)
        if wage is not None:
            self.wage_history.append(wage)
        if rental_price is not None:
            self.rental_history.append(rental_price)
    
    def form_all_expectations(self,
                               current_values: Dict[str, float],
                               additional_info: Dict[str, Any] = None) -> ExpectationState:
        """
        形成所有预期
        
        Args:
            current_values: 当前值字典
            additional_info: 额外信息
            
        Returns:
            ExpectationState
        """
        # 粮食价格预期
        self.state.expected_grain_price = self.mechanism.form_expectation(
            self.price_history['grain'],
            current_values.get('grain_price', 1.5),
            additional_info
        )
        
        # 经作价格预期
        self.state.expected_cash_crop_price = self.mechanism.form_expectation(
            self.price_history['cash_crop'],
            current_values.get('cash_crop_price', 2.5),
            additional_info
        )
        
        # 补贴预期
        self.state.expected_subsidy_rate = self.mechanism.form_expectation(
            self.subsidy_history,
            current_values.get('subsidy_rate', 0.15),
            additional_info
        )
        
        # 工资预期
        self.state.expected_wage = self.mechanism.form_expectation(
            self.wage_history,
            current_values.get('wage', DEFAULT_BASE_NON_FARM_WAGE),
            additional_info
        )
        
        return self.state
    
    def record_prediction_error(self,
                                 predicted: Dict[str, float],
                                 realized: Dict[str, float]):
        """记录预期误差用于学习"""
        if 'grain_price' in predicted and 'grain_price' in realized:
            error = realized['grain_price'] - predicted['grain_price']
            self.state.grain_price_errors.append(error)
            if len(self.state.grain_price_errors) > self.state.max_error_history:
                self.state.grain_price_errors.pop(0)
        
        if 'cash_crop_price' in predicted and 'cash_crop_price' in realized:
            error = realized['cash_crop_price'] - predicted['cash_crop_price']
            self.state.cash_price_errors.append(error)
            if len(self.state.cash_price_errors) > self.state.max_error_history:
                self.state.cash_price_errors.pop(0)
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """获取预期准确度"""
        accuracy = {}
        
        if self.state.grain_price_errors:
            mse = np.mean(np.square(self.state.grain_price_errors))
            accuracy['grain_price_mse'] = mse
        
        if self.state.cash_price_errors:
            mse = np.mean(np.square(self.state.cash_price_errors))
            accuracy['cash_price_mse'] = mse
        
        return accuracy
