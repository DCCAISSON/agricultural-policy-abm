# -*- coding: utf-8 -*-
"""
集成决策逻辑模块
@module: modules/integrated_decision_logic.py
@author: Agricultural ABM Team
@date: 2025-12-22
@priority: 0 - 核心逻辑漏洞修复

@critical_fix:
    解决"决策目标与评估指标错位"问题:
    1. 构建新的决策效用函数: Decision_Utility = Expected_Profit - Risk_Penalty - Subsistence_Penalty
    2. 若农户预期收入低于Stone-Geary效用函数的生存线(γ参数),生存惩罚项趋于无穷大
    3. 确保低收入农户优先满足生存需求,而非单纯最大化利润

@theory:
    Stone-Geary效用函数与前景理论融合:
    U = Σ βᵢ * ln(Cᵢ - γᵢ) 其中γᵢ为生存线(subsistence level)
    当 Cᵢ < γᵢ 时,效用为负无穷,农户无法生存
    
    决策效用 = 期望利润 - 风险惩罚(前景理论) - 生存惩罚(Stone-Geary)

@reference:
    - Stone (1954) Linear Expenditure Systems
    - Geary (1950) A Note on 'A Constant-Utility Index of the Cost of Living'
    - Kahneman & Tversky (1979) Prospect Theory
    - Ray (1998) Development Economics, Ch.8 Poverty Traps
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum


class DecisionType(Enum):
    """决策类型枚举"""
    PRODUCTION = "production"       # 生产决策(种什么、种多少)
    LAND_TRANSFER = "land_transfer" # 土地流转决策
    LABOR_ALLOCATION = "labor_allocation"  # 劳动力配置决策
    SERVICE_PURCHASE = "service_purchase"  # 服务购买决策
    CONSUMPTION = "consumption"     # 消费决策


@dataclass
class SubsistenceParameters:
    """
    生存线参数 (Stone-Geary γ参数)
    
    @theory:
    这些参数定义了农户维持基本生存所需的最低消费/收入水平
    低于这些阈值时,农户会优先保障生存而非利润最大化
    """
    # 年度生存线收入(元/年),低于此收入将触发生存惩罚
    annual_subsistence_income: float = 8000.0
    
    # 人均粮食消费量(公斤/年),自用粮食需求
    per_capita_grain_consumption: float = 200.0
    
    # 最低劳动天数(天/年),低于此无法维持基本农业生产
    min_labor_days: float = 50.0
    
    # 生存惩罚斜率,控制惩罚增长速度
    penalty_slope: float = 10.0
    
    # 生存缓冲系数,收入在1.0-1.5倍生存线之间时仍有轻微惩罚
    buffer_multiplier: float = 1.5


@dataclass
class RiskParameters:
    """
    风险参数 (前景理论)
    """
    # 损失厌恶系数 λ (Tversky & Kahneman: 2.25)
    loss_aversion: float = 1.95  # @updated 2025-12-28: Brown et al. (2024) AER meta-analysis
    
    # 价值函数曲率 α (Tversky & Kahneman: 0.88)
    value_curvature: float = 0.88
    
    # 概率权重函数参数 γ
    probability_weight_gamma: float = 0.61
    
    # 参考点类型 (last_year_income, average_income, expected_income)
    reference_point_type: str = "last_year_income"


@dataclass
class IntegratedDecisionConfig:
    """
    集成决策配置
    """
    subsistence: SubsistenceParameters = field(default_factory=SubsistenceParameters)
    risk: RiskParameters = field(default_factory=RiskParameters)
    
    # 权重参数
    weight_expected_profit: float = 1.0
    weight_risk_penalty: float = 0.3
    weight_subsistence_penalty: float = 2.0  # 生存惩罚权重较高
    
    # 时间偏好(折现率)
    discount_rate: float = 0.1


class IntegratedDecisionEngine:
    """
    集成决策引擎
    
    @critical_fix 2025-12-22:
    消费模块与生产决策的桥接层,确保:
    1. 决策目标与农户实际福利一致
    2. 低收入农户优先满足生存需求
    3. 风险厌恶随收入水平变化
    """
    
    def __init__(self, config: Union[Dict, IntegratedDecisionConfig] = None):
        if config is None:
            self.config = IntegratedDecisionConfig()
        elif isinstance(config, dict):
            subsistence = SubsistenceParameters(**config.get('subsistence', {}))
            risk = RiskParameters(**config.get('risk', {}))
            self.config = IntegratedDecisionConfig(
                subsistence=subsistence,
                risk=risk,
                weight_expected_profit=config.get('weight_expected_profit', 1.0),
                weight_risk_penalty=config.get('weight_risk_penalty', 0.3),
                weight_subsistence_penalty=config.get('weight_subsistence_penalty', 2.0),
                discount_rate=config.get('discount_rate', 0.1)
            )
        else:
            self.config = config
    
    # ==================== 核心效用计算 ====================
    
    def calculate_decision_utility(
        self,
        expected_profit: float,
        profit_variance: float,
        expected_income: float,
        reference_point: float = None,
        farmer_characteristics: Dict = None
    ) -> float:
        """
        @critical_fix: 计算综合决策效用
        
        Decision_Utility = Expected_Profit - Risk_Penalty - Subsistence_Penalty
        
        Args:
            expected_profit: 期望利润
            profit_variance: 利润方差(用于风险评估)
            expected_income: 预期总收入(含非农收入)
            reference_point: 参考点(默认为上年收入)
            farmer_characteristics: 农户特征字典
        
        Returns:
            决策效用值
        """
        # 1. 期望利润项
        profit_component = self.config.weight_expected_profit * expected_profit
        
        # 2. 风险惩罚项 (前景理论)
        risk_penalty = self._calculate_risk_penalty(
            expected_profit, profit_variance, reference_point
        )
        risk_component = self.config.weight_risk_penalty * risk_penalty
        
        # 3. 生存惩罚项 (Stone-Geary)
        subsistence_penalty = self._calculate_subsistence_penalty(
            expected_income, farmer_characteristics
        )
        subsistence_component = self.config.weight_subsistence_penalty * subsistence_penalty
        
        # 综合效用
        utility = profit_component - risk_component - subsistence_component
        
        return utility
    
    def _calculate_risk_penalty(
        self, 
        expected_profit: float, 
        variance: float,
        reference_point: float = None
    ) -> float:
        """
        计算风险惩罚 (基于前景理论)
        
        @theory:
        - 低于参考点的部分按损失处理,惩罚更重(损失厌恶)
        - 方差越大,不确定性越高,惩罚越重
        """
        if variance <= 0:
            return 0.0
        
        std_dev = np.sqrt(variance)
        ref = reference_point if reference_point else expected_profit * 0.8
        
        # 计算预期损失概率(假设正态分布)
        loss_prob = self._normal_cdf(ref, expected_profit, std_dev)
        
        # 预期损失幅度
        if loss_prob > 0.01:
            # 条件期望: E[X | X < ref]
            expected_loss = self._conditional_expected_loss(expected_profit, std_dev, ref)
            loss_magnitude = max(0, ref - expected_loss)
        else:
            loss_magnitude = 0
        
        # 损失厌恶调整
        weighted_loss = self.config.risk.loss_aversion * loss_magnitude
        
        # 概率权重
        weighted_prob = self._probability_weight(loss_prob)
        
        risk_penalty = weighted_prob * weighted_loss
        
        return risk_penalty
    
    def _calculate_subsistence_penalty(
        self,
        expected_income: float,
        farmer_characteristics: Dict = None
    ) -> float:
        """
        @critical_fix: 计算生存惩罚 (Stone-Geary)
        
        @theory:
        当收入低于生存线时,惩罚急剧增加
        这确保低收入农户优先满足生存需求
        
        惩罚公式:
        if income < γ: penalty → +∞ (实际使用大数)
        if income < γ * buffer: penalty = slope * (γ * buffer - income)
        if income >= γ * buffer: penalty = 0
        """
        params = self.config.subsistence
        
        # 调整生存线(根据家庭规模)
        family_size = 3  # 默认
        if farmer_characteristics:
            family_size = farmer_characteristics.get('family_size', 3)
        
        adjusted_subsistence = params.annual_subsistence_income * (family_size / 3)
        buffer_threshold = adjusted_subsistence * params.buffer_multiplier
        
        if expected_income < adjusted_subsistence * 0.5:
            # 极度贫困:惩罚趋于无穷大(实际使用1e6)
            return 1e6
        elif expected_income < adjusted_subsistence:
            # 低于生存线:严重惩罚
            shortfall_ratio = (adjusted_subsistence - expected_income) / adjusted_subsistence
            return params.penalty_slope * 100 * (shortfall_ratio ** 2)
        elif expected_income < buffer_threshold:
            # 缓冲区:轻微惩罚
            shortfall = buffer_threshold - expected_income
            shortfall_ratio = shortfall / (buffer_threshold - adjusted_subsistence)
            return params.penalty_slope * shortfall_ratio
        else:
            # 充裕:无惩罚
            return 0.0
    
    # ==================== 决策评估方法 ====================
    
    def evaluate_production_decision(
        self,
        options: List[Dict],
        farmer: Any,
        market_conditions: Dict = None
    ) -> Dict:
        """
        评估生产决策选项
        
        Args:
            options: 选项列表,每个选项包含 {'crop_type', 'area', 'expected_yield', 'expected_price', 'cost'}
            farmer: 农户Agent
            market_conditions: 市场条件
        
        Returns:
            最优选项及其效用
        """
        results = []
        
        # 获取农户特征
        farmer_chars = self._extract_farmer_characteristics(farmer)
        reference_point = self._get_reference_point(farmer)
        
        for option in options:
            # 计算期望利润
            expected_revenue = option.get('expected_yield', 0) * option.get('expected_price', 0)
            expected_cost = option.get('cost', 0)
            expected_profit = expected_revenue - expected_cost
            
            # 估计利润方差(基于历史波动)
            price_volatility = market_conditions.get('price_volatility', 0.2) if market_conditions else 0.2
            yield_volatility = farmer_chars.get('yield_volatility', 0.15)
            profit_variance = (expected_profit * (price_volatility + yield_volatility)) ** 2
            
            # 计算预期总收入
            off_farm_income = getattr(farmer, 'off_farm_income', 0) if farmer else 0
            expected_income = expected_profit + off_farm_income
            
            # 计算决策效用
            utility = self.calculate_decision_utility(
                expected_profit=expected_profit,
                profit_variance=profit_variance,
                expected_income=expected_income,
                reference_point=reference_point,
                farmer_characteristics=farmer_chars
            )
            
            results.append({
                'option': option,
                'expected_profit': expected_profit,
                'expected_income': expected_income,
                'utility': utility
            })
        
        # 选择效用最高的选项
        if results:
            best = max(results, key=lambda x: x['utility'])
            return {
                'best_option': best['option'],
                'best_utility': best['utility'],
                'all_evaluations': results
            }
        return {'best_option': None, 'best_utility': float('-inf'), 'all_evaluations': []}
    
    def evaluate_land_transfer_decision(
        self,
        transfer_out_area: float,
        expected_rental_income: float,
        alternative_income: float,
        farmer: Any
    ) -> Dict:
        """
        评估土地流转决策
        
        Args:
            transfer_out_area: 拟转出面积
            expected_rental_income: 预期租金收入
            alternative_income: 转出后的替代收入(如务工)
            farmer: 农户Agent
        
        Returns:
            决策评估结果
        """
        farmer_chars = self._extract_farmer_characteristics(farmer)
        
        # 转出情景
        total_income_transfer = expected_rental_income + alternative_income
        utility_transfer = self.calculate_decision_utility(
            expected_profit=expected_rental_income,
            profit_variance=expected_rental_income * 0.1,  # 租金收入波动小
            expected_income=total_income_transfer,
            farmer_characteristics=farmer_chars
        )
        
        # 不转出情景(保持现状)
        current_income = getattr(farmer, 'total_income', 0) if farmer else 0
        current_profit = getattr(farmer, 'agricultural_income', 0) if farmer else 0
        utility_keep = self.calculate_decision_utility(
            expected_profit=current_profit,
            profit_variance=current_profit * 0.25,  # 农业收入波动大
            expected_income=current_income,
            farmer_characteristics=farmer_chars
        )
        
        should_transfer = utility_transfer > utility_keep
        
        return {
            'should_transfer': should_transfer,
            'utility_transfer': utility_transfer,
            'utility_keep': utility_keep,
            'utility_gain': utility_transfer - utility_keep,
            'income_comparison': {
                'transfer': total_income_transfer,
                'keep': current_income
            }
        }
    
    # ==================== 辅助方法 ====================
    
    def _extract_farmer_characteristics(self, farmer: Any) -> Dict:
        """提取农户特征"""
        if farmer is None:
            return {
                'family_size': 3,
                'age': 50,
                'risk_aversion': 1.0,
                'yield_volatility': 0.15
            }
        
        return {
            'family_size': getattr(farmer, 'family_size', 3),
            'age': getattr(farmer, 'age', 50),
            'risk_aversion': getattr(farmer, 'risk_aversion_coefficient', 1.0),
            'yield_volatility': getattr(farmer, 'yield_volatility', 0.15),
            'farming_ability': getattr(farmer, 'farming_ability', 1.0),
            'off_farm_ratio': getattr(farmer, 'off_farm_income_ratio', 0.3)
        }
    
    def _get_reference_point(self, farmer: Any) -> float:
        """获取决策参考点"""
        if farmer is None:
            return 10000.0
        
        ref_type = self.config.risk.reference_point_type
        
        if ref_type == "last_year_income":
            return getattr(farmer, 'last_year_income', 
                          getattr(farmer, 'total_income', 10000))
        elif ref_type == "average_income":
            return getattr(farmer, 'average_historical_income', 10000)
        else:
            return getattr(farmer, 'expected_income', 10000)
    
    def _normal_cdf(self, x: float, mean: float, std: float) -> float:
        """正态分布CDF"""
        from scipy.stats import norm
        return norm.cdf(x, loc=mean, scale=std)
    
    def _conditional_expected_loss(self, mean: float, std: float, threshold: float) -> float:
        """计算条件期望损失 E[X | X < threshold]"""
        from scipy.stats import norm
        z = (threshold - mean) / std
        if norm.cdf(z) < 1e-10:
            return mean
        return mean - std * norm.pdf(z) / norm.cdf(z)
    
    def _probability_weight(self, p: float) -> float:
        """
        概率权重函数 (Prelec 1998)
        w(p) = exp(-(-ln(p))^γ)
        """
        if p <= 0:
            return 0
        if p >= 1:
            return 1
        
        gamma = self.config.risk.probability_weight_gamma
        return np.exp(-((-np.log(p)) ** gamma))


# ==================== 便捷接口 ====================

def create_decision_engine(config: Dict = None) -> IntegratedDecisionEngine:
    """创建决策引擎的工厂函数"""
    return IntegratedDecisionEngine(config)


def calculate_integrated_utility(
    expected_profit: float,
    expected_income: float,
    profit_variance: float = None,
    reference_point: float = None,
    family_size: int = 3,
    config: Dict = None
) -> float:
    """
    便捷函数:计算集成决策效用
    """
    engine = create_decision_engine(config)
    
    if profit_variance is None:
        profit_variance = (expected_profit * 0.2) ** 2
    
    return engine.calculate_decision_utility(
        expected_profit=expected_profit,
        profit_variance=profit_variance,
        expected_income=expected_income,
        reference_point=reference_point,
        farmer_characteristics={'family_size': family_size}
    )
