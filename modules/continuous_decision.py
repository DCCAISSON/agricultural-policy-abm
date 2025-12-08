# -*- coding: utf-8 -*-
"""
@theory 基于连续变量的决策函数模块
@critical 2025-12: 用连续变量替代离散标签

核心思想：
- 不再使用 if agent.type == "兼业II型" 这样的离散判断
- 而是基于连续特征变量的函数（logit、阈值函数等）
- 大幅提升模型真实性和连续性

参考文献：
- McFadden (1974) 离散选择的条件Logit模型
- Train (2009) 离散选择方法及模拟

政策敏感度矩阵（参考用户提供的表格）：
农户类型           价格补贴  种粮直补  流转补贴  社会化服务
传统生存型小农      ★★★★★   ★★★     ★        ★★★★
现代商业型农户      ★★★     ★★★★★   ★★★★    ★
兼业I型            ★★★★    ★★★★    ★★★     ★★
兼业II型           ★        ★★       ★★★★★   ★★★★★
"离土不离权"地主   ★        ★        ★★★★★   ★★★★
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set, Any, Union


@dataclass
class ContinuousDecisionParams:
    """连续决策模型参数"""
    # Logit模型温度参数（越大选择越随机）
    temperature: float = 1.0
    
    # 阈值函数的斜率参数
    threshold_slope: float = 5.0
    
    # 噪声水平
    noise_std: float = 0.1


class ContinuousDecisionFunctions:
    """
    基于连续变量的决策函数集合
    
    所有函数输入农户特征（连续变量），输出决策概率或效用值
    """
    
    @staticmethod
    def sigmoid(x: float, slope: float = 1.0, center: float = 0.0) -> float:
        """Sigmoid函数，用于将连续变量映射到[0,1]概率"""
        return 1.0 / (1.0 + np.exp(-slope * (x - center)))
    
    @staticmethod
    def logit_probability(utilities: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """
        Logit选择概率
        
        P(选择i) = exp(U_i / T) / Σ exp(U_j / T)
        """
        if not utilities:
            return {}
        
        # 防止数值溢出
        max_u = max(utilities.values())
        exp_utils = {k: np.exp((v - max_u) / temperature) for k, v in utilities.items()}
        total = sum(exp_utils.values())
        
        return {k: v / total for k, v in exp_utils.items()}
    
    # ==================== 政策敏感度计算 ====================
    
    @staticmethod
    def calculate_price_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        commercialization_rate: float,
        aging_rate: float
    ) -> float:
        """
        计算对粮食价格补贴的敏感度
        
        逻辑：
        - 农业依赖度高 → 敏感度高（主要收入来源）
        - 规模小 → 敏感度高（边际效用大）
        - 商品化率低 → 敏感度高（主要自用，价格影响粮食安全感）
        - 老龄化高 → 敏感度高（风险承受低）
        
        Returns: [0, 1] 敏感度
        """
        # 各因素权重
        w_agri = 0.35
        w_scale = 0.25
        w_comm = 0.20
        w_aging = 0.20
        
        # 农业依赖度正向影响
        s_agri = agricultural_dependence
        
        # 规模负向影响（小规模更敏感）
        s_scale = 1.0 - min(1.0, land_scale / 50.0)
        
        # 商品化率负向影响（低商品化更敏感）
        s_comm = 1.0 - commercialization_rate
        
        # 老龄化正向影响
        s_aging = aging_rate
        
        sensitivity = w_agri * s_agri + w_scale * s_scale + w_comm * s_comm + w_aging * s_aging
        return np.clip(sensitivity, 0.0, 1.0)
    
    @staticmethod
    def calculate_direct_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        commercialization_rate: float,
        technology_adoption: float
    ) -> float:
        """
        计算对种粮直接补贴的敏感度
        
        逻辑：
        - 规模大 → 敏感度高（补贴总额大）
        - 商品化率高 → 敏感度高（专业化经营）
        - 技术采纳高 → 敏感度高（现代农业经营者）
        - 农业依赖度中等时最敏感（既专业又依赖）
        """
        w_scale = 0.35
        w_comm = 0.30
        w_tech = 0.20
        w_agri = 0.15
        
        # 规模正向影响（大规模更敏感）
        s_scale = min(1.0, land_scale / 50.0)
        
        # 商品化率正向影响
        s_comm = commercialization_rate
        
        # 技术采纳正向影响
        s_tech = technology_adoption
        
        # 农业依赖度倒U型（中等依赖度最敏感）
        s_agri = 1.0 - abs(agricultural_dependence - 0.6) * 2
        s_agri = max(0.0, s_agri)
        
        sensitivity = w_scale * s_scale + w_comm * s_comm + w_tech * s_tech + w_agri * s_agri
        return np.clip(sensitivity, 0.0, 1.0)
    
    @staticmethod
    def calculate_rental_subsidy_sensitivity(
        agricultural_dependence: float,
        land_scale: float,
        land_leased_out_ratio: float,
        land_leased_in_ratio: float
    ) -> float:
        """
        计算对流转租金补贴的敏感度
        
        逻辑：
        - 已流出土地多 → 敏感度高（租金是收入来源）
        - 农业依赖度低 → 敏感度高（已退出农业）
        - 已流入土地多 → 敏感度高（租金是成本）
        """
        w_out = 0.40
        w_agri = 0.30
        w_in = 0.30
        
        # 流出比例正向影响
        s_out = land_leased_out_ratio
        
        # 农业依赖度负向影响（低依赖度更敏感）
        s_agri = 1.0 - agricultural_dependence
        
        # 流入比例正向影响（租入者关心租金成本）
        s_in = land_leased_in_ratio
        
        sensitivity = w_out * s_out + w_agri * s_agri + w_in * s_in
        return np.clip(sensitivity, 0.0, 1.0)
    
    # ==================== 核心决策函数 ====================
    
    @staticmethod
    def land_transfer_out_probability(
        agricultural_dependence: float,
        land_scale: float,
        aging_rate: float,
        off_farm_income_ratio: float,
        rental_price: float,
        reservation_price: float,
        social_capital: float = 5.0
    ) -> float:
        """
        土地流出概率（基于连续变量的Logit函数）
        
        不再使用：if livelihood_strategy == NON_FARMER: return 0.8
        而是基于连续特征计算
        
        Returns: [0, 1] 流出概率
        """
        # 效用函数 U_out = β0 + β1*农业依赖度 + β2*规模 + β3*老龄化 + ...
        
        # 农业依赖度负向影响流出（依赖农业则不愿流出）
        u_agri = -2.0 * agricultural_dependence
        
        # 规模负向影响（小规模更易流出）
        u_scale = -0.02 * land_scale
        
        # 老龄化正向影响（老龄化高则流出）
        u_aging = 1.5 * aging_rate
        
        # 非农收入占比正向影响
        u_off_farm = 1.0 * off_farm_income_ratio
        
        # 租金相对保留价的吸引力
        price_ratio = (rental_price - reservation_price) / max(reservation_price, 100)
        u_price = 0.5 * price_ratio
        
        # 社会资本负向影响（关系多则信息多，不急于流出）
        u_social = -0.05 * social_capital
        
        # 总效用
        utility = u_agri + u_scale + u_aging + u_off_farm + u_price + u_social
        
        # Sigmoid转换为概率
        prob = ContinuousDecisionFunctions.sigmoid(utility, slope=1.0, center=0.0)
        
        return np.clip(prob, 0.05, 0.95)  # 保留极端值边界
    
    @staticmethod
    def land_transfer_in_probability(
        agricultural_dependence: float,
        land_scale: float,
        capital: float,
        technology_adoption: float,
        target_scale: float,
        rental_price: float,
        expected_profit_per_mu: float
    ) -> float:
        """
        土地流入概率（扩大经营规模的意愿）
        
        Returns: [0, 1] 流入概率
        """
        # 农业依赖度正向影响（依赖农业则想扩大）
        u_agri = 1.5 * agricultural_dependence
        
        # 当前规模与目标规模的差距
        scale_gap = max(0, target_scale - land_scale) / max(target_scale, 10)
        u_gap = 1.0 * scale_gap
        
        # 资本充足度
        capital_adequacy = min(1.0, capital / (land_scale * rental_price + 10000))
        u_capital = 0.8 * capital_adequacy
        
        # 技术采纳能力
        u_tech = 0.5 * technology_adoption
        
        # 预期利润
        profit_attractiveness = expected_profit_per_mu / 500  # 500元/亩作为参考
        u_profit = 0.6 * min(1.0, max(-1.0, profit_attractiveness))
        
        utility = u_agri + u_gap + u_capital + u_tech + u_profit
        prob = ContinuousDecisionFunctions.sigmoid(utility, slope=0.8, center=1.0)
        
        return np.clip(prob, 0.02, 0.90)
    
    @staticmethod
    def crop_choice_grain_preference(
        agricultural_dependence: float,
        land_scale: float,
        risk_tolerance: float,
        price_subsidy_sensitivity: float,
        grain_expected_return: float,
        cash_crop_expected_return: float,
        grain_risk: float,
        cash_crop_risk: float
    ) -> float:
        """
        粮食作物选择偏好（基于连续变量）
        
        Returns: [0, 1] 选择粮食的概率
        """
        # 风险调整收益
        grain_utility = grain_expected_return - (1 - risk_tolerance) * grain_risk
        cash_utility = cash_crop_expected_return - (1 - risk_tolerance) * cash_crop_risk
        
        # 补贴敏感度调整（对粮食补贴敏感则偏好粮食）
        grain_utility += price_subsidy_sensitivity * 0.3 * grain_expected_return
        
        # 农业依赖度高→风险厌恶→偏好稳定的粮食
        if agricultural_dependence > 0.7:
            grain_utility += 0.1 * grain_expected_return
        
        # 规模大→可能偏好经济作物（追求利润）
        if land_scale > 30:
            cash_utility += 0.05 * cash_crop_expected_return
        
        # Logit选择
        utilities = {'grain': grain_utility, 'cash': cash_utility}
        probs = ContinuousDecisionFunctions.logit_probability(utilities, temperature=0.5)
        
        return probs.get('grain', 0.5)
    
    @staticmethod
    def subsidy_response_adjustment(
        base_decision: float,
        subsidy_change: float,
        sensitivity: float,
        response_type: str = 'linear'
    ) -> float:
        """
        补贴变化对决策的调整
        
        Args:
            base_decision: 基础决策值
            subsidy_change: 补贴变化率（正为增加）
            sensitivity: 敏感度 [0, 1]
            response_type: 响应类型 'linear' 或 'diminishing'
        
        Returns: 调整后的决策值
        """
        if response_type == 'linear':
            adjustment = sensitivity * subsidy_change
        elif response_type == 'diminishing':
            # 边际递减响应
            adjustment = sensitivity * np.tanh(subsidy_change)
        else:
            adjustment = 0
        
        return np.clip(base_decision + adjustment, 0.0, 1.0)


# ==================== 前景理论参数的类型默认值 ====================

PROSPECT_THEORY_DEFAULTS = {
    # 基于农业依赖度区间的默认值
    'high_dependence': {  # 农业依赖度 > 0.7
        'loss_aversion_lambda': 2.8,
        'value_curvature_alpha': 0.50,
        'probability_weighting_gamma': 0.65,
    },
    'medium_dependence': {  # 0.3 < 农业依赖度 <= 0.7
        'loss_aversion_lambda': 2.5,
        'value_curvature_alpha': 0.55,
        'probability_weighting_gamma': 0.70,
    },
    'low_dependence': {  # 农业依赖度 <= 0.3
        'loss_aversion_lambda': 2.2,
        'value_curvature_alpha': 0.60,
        'probability_weighting_gamma': 0.75,
    },
}


def get_prospect_params_by_dependence(agricultural_dependence: float) -> Dict[str, float]:
    """
    根据农业依赖度返回前景理论参数默认值
    
    简化：不再为每个农户单独抽样，而是按类型给默认值
    """
    if agricultural_dependence > 0.7:
        return PROSPECT_THEORY_DEFAULTS['high_dependence']
    elif agricultural_dependence > 0.3:
        return PROSPECT_THEORY_DEFAULTS['medium_dependence']
    else:
        return PROSPECT_THEORY_DEFAULTS['low_dependence']


# ==================== 简化的社会网络关系判断 ====================

def check_acquaintance_simple(
    agent1_network: Set[int],
    agent2_id: int,
    agent2_network: Optional[Set[int]] = None,
    max_distance: int = 2
) -> Tuple[bool, Union[int, float]]:
    """
    简化的熟人关系判断
    
    原来的规则太复杂，现在简化为：
    - 距离1：直接邻居（在对方网络中）
    - 距离2：有共同邻居
    
    Args:
        agent1_network: agent1的直接邻居集合
        agent2_id: agent2的ID
        agent2_network: agent2的直接邻居集合（可选）
        max_distance: 最大网络距离
    
    Returns:
        (is_acquaintance, distance)
    """
    # 距离1：直接邻居
    if agent2_id in agent1_network:
        return True, 1
    
    # 距离2：共同邻居（如果提供了agent2的网络）
    if max_distance >= 2 and agent2_network is not None:
        common_neighbors = agent1_network.intersection(agent2_network)
        if len(common_neighbors) > 0:
            return True, 2
    
    return False, float('inf')


def calculate_acquaintance_discount(distance: int, base_discount: float = 0.15) -> float:
    """
    计算熟人折扣率
    
    距离1：折扣15%
    距离2：折扣8%
    非熟人：无折扣
    """
    if distance == 1:
        return base_discount
    elif distance == 2:
        return base_discount * 0.5
    else:
        return 0.0
