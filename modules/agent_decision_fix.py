# -*- coding: utf-8 -*-
"""
@theory 智能体决策修复模块
@new 2025-12-22: 解决"决策孤立"问题 - 连续决策函数未被实际调用

问题诊断：
1. ContinuousDecisionFunctions已导入但未调用
2. 仍有20+处使用离散类型判断 (if livelihood_strategy == ...)
3. 决策与评估使用不同逻辑框架

解决方案：
1. 提供统一的连续决策API
2. 自动将离散类型映射到连续特征
3. 确保决策闭环：感知→预期→决策→行动→学习
"""

import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent, LivelihoodStrategy


@dataclass
class ContinuousAgentState:
    """连续智能体状态 - 替代离散类型判断"""
    agricultural_dependence: float  # [0,1] 农业依赖度
    land_scale_normalized: float    # [0,1] 土地规模（归一化）
    age_factor: float              # [0,1] 年龄因子（年轻=0，老年=1）
    risk_aversion: float           # [0,1] 风险厌恶程度
    social_capital: float          # [0,1] 社会资本水平
    labor_capacity: float          # [0,1] 劳动能力
    financial_stress: float        # [0,1] 财务压力指数
    
    @classmethod
    def from_farmer(cls, farmer) -> 'ContinuousAgentState':
        """从农户对象提取连续状态"""
        # 农业依赖度
        if hasattr(farmer, 'characteristics') and farmer.characteristics:
            agri_dep = farmer.characteristics.agricultural_dependence
        else:
            # 从收入结构估计
            total_income = getattr(farmer, 'total_income', 1) or 1
            agri_income = getattr(farmer, 'agricultural_income', 0)
            agri_dep = agri_income / total_income if total_income > 0 else 0.5
        
        # 土地规模归一化（50亩为上限）
        operated_area = farmer.get_operated_land_area() if hasattr(farmer, 'get_operated_land_area') else 0
        land_scale = min(1.0, operated_area / 50)
        
        # 年龄因子（25岁=0, 75岁=1）
        age = getattr(farmer, 'age', 50)
        age_factor = np.clip((age - 25) / 50, 0, 1)
        
        # 风险厌恶（归一化到0-1）
        risk_raw = getattr(farmer, 'risk_aversion', 1.5)
        risk_aversion = np.clip((risk_raw - 0.5) / 3, 0, 1)
        
        # 社会资本
        if hasattr(farmer, 'characteristics') and farmer.characteristics:
            social_capital = getattr(farmer.characteristics, 'social_capital', 0.5)
        else:
            social_capital = 0.5 + (0.2 if getattr(farmer, 'is_village_elite', False) else 0)
        
        # 劳动能力（基于年龄和健康）
        if age < 50:
            labor_capacity = 1.0
        elif age < 60:
            labor_capacity = 0.9 - (age - 50) * 0.02
        elif age < 70:
            labor_capacity = 0.7 - (age - 60) * 0.03
        else:
            labor_capacity = max(0.3, 0.4 - (age - 70) * 0.02)
        
        # 财务压力指数
        capital = getattr(farmer, 'capital', 10000)
        initial = getattr(farmer, 'initial_capital', 10000)
        if initial > 0:
            capital_ratio = capital / initial
            financial_stress = np.clip(1 - capital_ratio, 0, 1)
        else:
            financial_stress = 0.5
        
        return cls(
            agricultural_dependence=agri_dep,
            land_scale_normalized=land_scale,
            age_factor=age_factor,
            risk_aversion=risk_aversion,
            social_capital=social_capital,
            labor_capacity=labor_capacity,
            financial_stress=financial_stress
        )


class ContinuousDecisionBridge:
    """
    连续决策桥接器
    
    @theory 将离散类型判断替换为连续函数
    
    核心转换：
    - if livelihood_strategy == PURE_FARMER → f(agricultural_dependence > 0.7)
    - if livelihood_strategy == NON_FARMER → f(agricultural_dependence < 0.2)
    - if farmer_scale == FAMILY_FARM → f(land_scale > 0.6)
    
    优势：
    1. 渐进变化替代跳变
    2. 边界模糊化，更符合现实
    3. 参数可从数据校准
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # 类型边界参数（可配置）
        boundary_config = self.config.get('decision_boundaries', {})
        self.pure_farmer_threshold = boundary_config.get('pure_farmer', 0.7)
        self.non_farmer_threshold = boundary_config.get('non_farmer', 0.2)
        self.family_farm_threshold = boundary_config.get('family_farm', 0.6)
        
        # 决策函数平滑度
        self.smoothness = boundary_config.get('smoothness', 5.0)
    
    def soft_pure_farmer(self, state: ContinuousAgentState) -> float:
        """
        软化的纯农户判断
        返回0-1之间的概率/程度值
        """
        return self._sigmoid(
            state.agricultural_dependence, 
            center=self.pure_farmer_threshold,
            slope=self.smoothness
        )
    
    def soft_non_farmer(self, state: ContinuousAgentState) -> float:
        """软化的非农户判断"""
        return 1 - self._sigmoid(
            state.agricultural_dependence,
            center=self.non_farmer_threshold,
            slope=self.smoothness
        )
    
    def soft_pluriactive(self, state: ContinuousAgentState) -> float:
        """软化的兼业户判断"""
        # 兼业户是中间类型
        pure_prob = self.soft_pure_farmer(state)
        non_prob = self.soft_non_farmer(state)
        return 1 - pure_prob - non_prob
    
    def land_transfer_willingness(self, state: ContinuousAgentState, 
                                   as_supplier: bool = True) -> float:
        """
        @theory 土地流转意愿（连续函数）
        
        转出意愿因素：
        - 非农依赖度高 → 意愿高
        - 年龄大 → 意愿高（尤其是无接班人时）
        - 劳动能力低 → 意愿高
        - 服务可得性低 → 意愿高（无法托管只能流转）
        
        转入意愿因素：
        - 规模未达最优 → 意愿高
        - 资金充裕 → 意愿高
        - 经营能力强 → 意愿高
        """
        if as_supplier:
            # 转出意愿
            base = 0.1  # 基础意愿
            
            # 非农依赖贡献
            non_farm_factor = (1 - state.agricultural_dependence) * 0.3
            
            # 年龄贡献（老年人更愿意转出）
            age_factor = state.age_factor * 0.25
            
            # 劳动能力贡献（能力低更愿意转出）
            labor_factor = (1 - state.labor_capacity) * 0.2
            
            # 财务压力贡献（压力大可能被迫转出）
            financial_factor = state.financial_stress * 0.15
            
            willingness = base + non_farm_factor + age_factor + labor_factor + financial_factor
            
        else:
            # 转入意愿
            base = 0.05
            
            # 农业依赖贡献（依赖农业更愿意扩大）
            agri_factor = state.agricultural_dependence * 0.25
            
            # 规模缺口贡献（规模小更愿意扩大）
            scale_factor = (1 - state.land_scale_normalized) * 0.3
            
            # 财务能力贡献（有钱才能租地）
            financial_factor = (1 - state.financial_stress) * 0.2
            
            # 劳动能力贡献
            labor_factor = state.labor_capacity * 0.15
            
            willingness = base + agri_factor + scale_factor + financial_factor + labor_factor
        
        return np.clip(willingness, 0, 1)
    
    def service_purchase_propensity(self, state: ContinuousAgentState) -> float:
        """
        @theory 服务购买倾向（连续函数）
        
        购买服务的动机：
        - 劳动能力不足
        - 有一定经济能力
        - 想保留土地但无力自耕
        """
        # 劳动能力不足驱动
        labor_need = (1 - state.labor_capacity) * 0.4
        
        # 年龄驱动（老年人更依赖服务）
        age_need = state.age_factor * 0.25
        
        # 经济能力约束（太穷买不起）
        economic_ability = (1 - state.financial_stress) * 0.25
        
        # 农业依赖（依赖农业才需要服务）
        agri_need = state.agricultural_dependence * 0.1
        
        propensity = labor_need + age_need + economic_ability + agri_need
        
        return np.clip(propensity, 0, 1)
    
    def abandonment_probability(self, state: ContinuousAgentState,
                                 patch_quality: float = 0.7,
                                 service_available: bool = True) -> float:
        """
        @theory 撂荒概率（连续函数）
        
        撂荒驱动因素：
        - 非农化程度高
        - 劳动能力低
        - 土地质量差
        - 服务不可用
        """
        base = 0.02  # 基础撂荒率2%
        
        # 非农化驱动
        non_farm = (1 - state.agricultural_dependence) * 0.08
        
        # 劳动能力驱动
        labor = (1 - state.labor_capacity) * 0.05
        
        # 土地质量驱动
        quality = (1 - patch_quality) * 0.04
        
        # 服务可得性调节（有服务可用则降低撂荒）
        service_factor = 0.6 if service_available else 1.2
        
        prob = (base + non_farm + labor + quality) * service_factor
        
        return np.clip(prob, 0, 0.25)  # 最高25%撂荒概率
    
    def optimal_scale_target(self, state: ContinuousAgentState,
                              base_optimal: float = 30) -> float:
        """
        @theory 最优经营规模目标（亩）
        
        受以下因素影响：
        - 劳动能力
        - 年龄
        - 经营能力（隐含在agricultural_dependence中）
        """
        # 劳动能力调节
        labor_factor = 0.5 + state.labor_capacity * 0.5
        
        # 年龄调节（老年人目标规模小）
        age_factor = 1.0 - state.age_factor * 0.4
        
        # 农业依赖调节（依赖农业的目标规模大）
        agri_factor = 0.6 + state.agricultural_dependence * 0.6
        
        target = base_optimal * labor_factor * age_factor * agri_factor
        
        return max(3, target)  # 最小3亩
    
    def _sigmoid(self, x: float, center: float = 0.5, slope: float = 5.0) -> float:
        """Sigmoid函数用于软化边界"""
        return 1 / (1 + np.exp(-slope * (x - center)))


def get_continuous_decision_for_farmer(farmer, decision_type: str, **kwargs) -> float:
    """
    @api 获取农户的连续决策值
    
    统一入口，替代离散类型判断
    
    Args:
        farmer: 农户对象
        decision_type: 决策类型
            - 'land_supply_willingness': 土地转出意愿
            - 'land_demand_willingness': 土地转入意愿
            - 'service_propensity': 服务购买倾向
            - 'abandonment_prob': 撂荒概率
            - 'optimal_scale': 最优规模目标
    
    Returns:
        连续决策值 (0-1 或具体数值)
    """
    state = ContinuousAgentState.from_farmer(farmer)
    bridge = ContinuousDecisionBridge()
    
    if decision_type == 'land_supply_willingness':
        return bridge.land_transfer_willingness(state, as_supplier=True)
    
    elif decision_type == 'land_demand_willingness':
        return bridge.land_transfer_willingness(state, as_supplier=False)
    
    elif decision_type == 'service_propensity':
        return bridge.service_purchase_propensity(state)
    
    elif decision_type == 'abandonment_prob':
        patch_quality = kwargs.get('patch_quality', 0.7)
        service_available = kwargs.get('service_available', True)
        return bridge.abandonment_probability(state, patch_quality, service_available)
    
    elif decision_type == 'optimal_scale':
        base = kwargs.get('base_optimal', 30)
        return bridge.optimal_scale_target(state, base)
    
    elif decision_type == 'is_pure_farmer':
        return bridge.soft_pure_farmer(state)
    
    elif decision_type == 'is_non_farmer':
        return bridge.soft_non_farmer(state)
    
    elif decision_type == 'is_pluriactive':
        return bridge.soft_pluriactive(state)
    
    else:
        raise ValueError(f"Unknown decision type: {decision_type}")


def replace_discrete_with_continuous(farmer, original_decision: str) -> Tuple[float, str]:
    """
    @utility 将离散判断替换为连续值
    
    用于渐进式迁移现有代码
    
    Returns:
        (continuous_value, interpretation)
    """
    value = get_continuous_decision_for_farmer(farmer, original_decision)
    
    
    # 提供解释
    if original_decision in ['is_pure_farmer', 'is_non_farmer', 'is_pluriactive']:
        if value > 0.7:
            interpretation = "strong"
        elif value > 0.4:
            interpretation = "moderate"
        else:
            interpretation = "weak"
    else:
        interpretation = f"{value:.2f}"
    
    return value, interpretation


class FarmerContinuousDecisionMixin:
    """
    @theory Mixin类：为FarmerAgent提供连续决策能力
    
    使用方法：在FarmerAgent类定义中添加此Mixin
    class FarmerAgent(Agent, FarmerContinuousDecisionMixin):
        ...
    
    然后在需要类型判断的地方使用：
    - self.get_agricultural_weight() 替代 if livelihood_strategy == PURE_FARMER
    - self.get_non_farm_weight() 替代 if livelihood_strategy == NON_FARMER
    - self.get_transfer_willingness('out') 替代硬编码的转出意愿
    """
    
    _decision_bridge: ContinuousDecisionBridge = None
    _continuous_state: ContinuousAgentState = None
    _state_cache_step: int = -1
    
    def _ensure_decision_bridge(self):
        """确保决策桥接器已初始化"""
        if self._decision_bridge is None:
            config = getattr(self.model, 'config', {}) if hasattr(self, 'model') else {}
            self._decision_bridge = ContinuousDecisionBridge(config)
    
    def _ensure_state_updated(self):
        """确保状态是当前步骤的"""
        current_step = self.model.step_count if hasattr(self, 'model') and self.model else 0
        if self._continuous_state is None or self._state_cache_step != current_step:
            self._continuous_state = ContinuousAgentState.from_farmer(self)
            self._state_cache_step = current_step
    
    def get_agricultural_weight(self) -> float:
        """
        @api 获取农业生计权重（替代 if PURE_FARMER 判断）
        
        返回值解释：
        - > 0.7: 相当于"纯农户"
        - 0.3-0.7: 相当于"兼业户"  
        - < 0.3: 相当于"非农户"
        """
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.soft_pure_farmer(self._continuous_state)
    
    def get_non_farm_weight(self) -> float:
        """@api 获取非农生计权重（替代 if NON_FARMER 判断）"""
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.soft_non_farmer(self._continuous_state)
    
    def get_pluriactive_weight(self) -> float:
        """@api 获取兼业权重"""
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.soft_pluriactive(self._continuous_state)
    
    def get_transfer_willingness(self, direction: str = 'out') -> float:
        """
        @api 获取土地流转意愿（连续值）
        
        Args:
            direction: 'out' 转出, 'in' 转入
        """
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.land_transfer_willingness(
            self._continuous_state,
            as_supplier=(direction == 'out')
        )
    
    def get_service_purchase_propensity(self) -> float:
        """@api 获取服务购买倾向（连续值）"""
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.service_purchase_propensity(self._continuous_state)
    
    def get_abandonment_probability(self, patch_quality: float = 0.7,
                                      service_available: bool = True) -> float:
        """@api 获取撂荒概率"""
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.abandonment_probability(
            self._continuous_state, patch_quality, service_available
        )
    
    def get_optimal_scale_target(self, base_optimal: float = 30) -> float:
        """@api 获取最优经营规模目标（亩）"""
        self._ensure_decision_bridge()
        self._ensure_state_updated()
        return self._decision_bridge.optimal_scale_target(self._continuous_state, base_optimal)
    
    def compute_strategy_weighted_value(self, 
                                         pure_value: float,
                                         pluri_value: float,
                                         non_value: float) -> float:
        """
        @api 计算加权决策值（替代if-else分支）
        
        示例：
        原代码：
            if self.livelihood_strategy == PURE_FARMER:
                multiplier = 1.2
            elif self.livelihood_strategy == PLURIACTIVE:
                multiplier = 0.8
            else:  # NON_FARMER
                multiplier = 0.5
        
        新代码：
            multiplier = self.compute_strategy_weighted_value(1.2, 0.8, 0.5)
        """
        w_pure = self.get_agricultural_weight()
        w_non = self.get_non_farm_weight()
        w_pluri = 1 - w_pure - w_non  # 确保权重和为1
        w_pluri = max(0, w_pluri)  # 防止负值
        
        total = w_pure + w_non + w_pluri
        if total > 0:
            w_pure /= total
            w_non /= total
            w_pluri /= total
        
        return w_pure * pure_value + w_pluri * pluri_value + w_non * non_value

