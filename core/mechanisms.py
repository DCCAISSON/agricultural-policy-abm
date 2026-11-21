# -*- coding: utf-8 -*-
"""
@theory 核心机制模块
本模块实现了气候政策ABM模型的核心理论机制，包括：
1. 前景理论驱动的农户决策行为
2. Melitz企业异质性模型的生产率动态
3. 创新扩散理论的S型扩散机制
4. 土地流转的市场配置机制

理论依据：
- 前景理论 (Kahneman & Tversky, 1979)
- Melitz企业异质性模型 (Melitz, 2003)
- 创新扩散理论 (Rogers, 2003)
- 土地市场理论 (Deininger & Jin, 2005)

数据来源：
- 国家统计局 (stats.gov.cn)
- 农业农村部 (moa.gov.cn)
- 中国农村观察 (zgncgc.ajcass.com)
- 农业技术扩散实证研究 (dlyj.ac.cn)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from scipy.special import gamma

from core.reproducibility import build_numpy_rng

class ProspectTheoryMechanism:
    """
    前景理论机制实现
    基于Kahneman & Tversky (1979)的前景理论，模拟农户在风险环境下的决策行为
    """
    
    def __init__(self, config: Dict):
        """
        初始化前景理论参数
        
        Args:
            config: 包含前景理论参数的配置字典
        """
        self.loss_aversion_lambda = config.get("loss_aversion_lambda", 3.0)
        self.value_function_curvature = config.get("value_function_curvature_sigma", 0.55)
        self.probability_weighting_alpha = config.get("probability_weighting_alpha", 0.7)
        
        # 参考点类型：农户决策的基准点
        self.reference_points = config.get("reference_points", 
                                         ["上年收入", "最低收购价", "同村均值", "政府补贴"])
    
    def value_function(self, outcome: float, reference_point: float) -> float:
        """
        前景理论价值函数
        实现S形价值函数：收益区凹函数（风险规避），损失区凸函数（风险寻求）
        
        Args:
            outcome: 实际结果
            reference_point: 参考点
            
        Returns:
            主观价值
        """
        gain_loss = outcome - reference_point
        
        if gain_loss >= 0:
            # 收益区：凹函数，体现敏感性递减
            return gain_loss ** self.value_function_curvature
        else:
            # 损失区：凸函数，体现损失厌恶
            return -self.loss_aversion_lambda * ((-gain_loss) ** self.value_function_curvature)
    
    def probability_weighting(self, probability: float) -> float:
        """
        概率权重函数
        农户倾向于高估小概率事件，低估中高概率事件
        
        Args:
            probability: 客观概率
            
        Returns:
            主观概率权重
        """
        if probability == 0:
            return 0
        elif probability == 1:
            return 1
        else:
            # 非线性概率权重，体现农户对概率的扭曲感知
            return (probability ** self.probability_weighting_alpha) / \
                   ((probability ** self.probability_weighting_alpha + 
                     (1 - probability) ** self.probability_weighting_alpha) ** (1/self.probability_weighting_alpha))
    
    def decision_utility(self, outcomes: List[float], probabilities: List[float], 
                       reference_point: float) -> float:
        """
        计算决策效用
        综合价值函数和概率权重，计算农户对某一选择的总体效用评价
        
        Args:
            outcomes: 可能结果列表
            probabilities: 对应概率列表
            reference_point: 参考点
            
        Returns:
            决策效用
        """
        total_utility = 0
        for outcome, prob in zip(outcomes, probabilities):
            # 价值函数评估
            value = self.value_function(outcome, reference_point)
            # 概率权重
            weighted_prob = self.probability_weighting(prob)
            total_utility += value * weighted_prob
        
        return total_utility


class MelitzHeterogeneityMechanism:
    """
    Melitz企业异质性机制实现
    基于Melitz (2003)的企业异质性模型，模拟农业经营主体的生产率差异和动态变化
    """
    
    def __init__(self, config: Dict):
        """
        初始化企业异质性参数
        
        Args:
            config: 包含企业异质性参数的配置字典
        """
        self.mu = config.get("mu", -0.2)
        self.sigma = config.get("sigma", 0.25)
        self.tech_range = config.get("range", [0.6, 1.0])
        self.seed = config.get("seed")
        self.rng = build_numpy_rng(self.seed, "melitz_heterogeneity")
        
        # 技术水平提升参数
        self.innovation_boost = config.get("innovation_boost", 0.1)
        self.learning_rate = config.get("learning_rate", 0.02)
    
    def generate_technology_levels(self, num_agents: int) -> np.ndarray:
        """
        生成企业技术水平分布
        采用对数正态分布，反映现实中企业生产率的异质性特征
        
        Args:
            num_agents: 智能体数量
            
        Returns:
            技术水平数组
        """
        # 生成对数正态分布的技术水平
        tech_levels = self.rng.lognormal(mean=self.mu, sigma=self.sigma, size=num_agents)
        
        # 截断到指定范围
        tech_levels = np.clip(tech_levels, self.tech_range[0], self.tech_range[1])
        
        return tech_levels
    
    def update_technology_level(self, current_level: float, 
                              innovation_adopted: bool = False,
                              experience_years: int = 0) -> float:
        """
        更新技术水平
        模拟技术水平的动态变化：创新采纳和学习效应
        
        Args:
            current_level: 当前技术水平
            innovation_adopted: 是否采纳创新
            experience_years: 经验年数
            
        Returns:
            更新后的技术水平
        """
        new_level = current_level
        
        # 创新采纳带来的技术提升
        if innovation_adopted:
            new_level += self.innovation_boost * (1.0 - current_level)  # 边际递减
        
        # 经验积累带来的学习效应
        learning_effect = self.learning_rate * experience_years * (1.0 - current_level)
        new_level += learning_effect
        
        # 确保在合理范围内
        return np.clip(new_level, self.tech_range[0], self.tech_range[1])


class InnovationDiffusionMechanism:
    """
    创新扩散机制实现
    基于Rogers (2003)的创新扩散理论，模拟农业技术的S型扩散过程
    """
    
    def __init__(self, config: Dict):
        """
        初始化创新扩散参数
        
        Args:
            config: 包含创新扩散参数的配置字典
        """
        self.adopter_categories = config.get("adopter_categories", {
            "innovators": 0.025,
            "early_adopters": 0.135,
            "early_majority": 0.34,
            "late_majority": 0.34,
            "laggards": 0.16
        })
        
        self.s_curve_params = config.get("s_curve_parameters", {
            "initial_adoption_rate": 0.05,
            "peak_adoption_rate": 0.15,
            "saturation_level": 0.8
        })
        
        # 社会网络影响参数
        self.network_influence_weight = config.get("network_influence_weight", 0.3)
        self.opinion_leader_multiplier = config.get("opinion_leader_multiplier", 2.0)
    
    def calculate_adoption_probability(self, agent_characteristics: Dict,
                                    social_network_influence: float = 0.0,
                                    subsidy_level: float = 0.0) -> float:
        """
        计算创新采纳概率
        综合考虑技术特性、社会网络影响、补贴政策等因素
        
        Args:
            agent_characteristics: 智能体特征（技术水平、风险偏好等）
            social_network_influence: 社会网络影响强度
            subsidy_level: 补贴水平
            
        Returns:
            采纳概率
        """
        base_prob = self.s_curve_params["initial_adoption_rate"]
        
        # 技术水平影响
        tech_level = agent_characteristics.get("technology_level", 0.7)
        tech_effect = tech_level * 0.3
        
        # 风险偏好影响（基于前景理论）
        risk_preference = agent_characteristics.get("risk_preference", 0.5)
        risk_effect = (1 - risk_preference) * 0.2  # 风险规避者采纳概率较低
        
        # 社会网络影响
        network_effect = social_network_influence * self.network_influence_weight
        
        # 补贴影响
        subsidy_effect = subsidy_level * 0.1
        
        # 综合计算采纳概率
        adoption_prob = base_prob + tech_effect + risk_effect + network_effect + subsidy_effect
        
        return np.clip(adoption_prob, 0.0, 1.0)
    
    def update_diffusion_curve(self, current_adoption_rate: float, 
                              time_step: int) -> float:
        """
        更新S型扩散曲线
        模拟创新扩散的时间动态
        
        Args:
            current_adoption_rate: 当前采纳率
            time_step: 时间步
            
        Returns:
            新的采纳率
        """
        # S型曲线参数
        initial_rate = self.s_curve_params["initial_adoption_rate"]
        peak_rate = self.s_curve_params["peak_adoption_rate"]
        saturation = self.s_curve_params["saturation_level"]
        
        # 计算S型曲线
        if time_step < 10:
            # 初始阶段：缓慢增长
            new_rate = initial_rate + (peak_rate - initial_rate) * (time_step / 10) ** 2
        elif time_step < 50:
            # 快速增长阶段
            new_rate = peak_rate
        else:
            # 饱和阶段：趋近饱和水平
            remaining_growth = saturation - peak_rate
            new_rate = peak_rate + remaining_growth * (1 - np.exp(-(time_step - 50) / 20))
        
        return np.clip(new_rate, 0.0, saturation)


class LandTransferMechanism:
    """
    土地流转机制实现
    基于土地市场理论，模拟土地从低效率向高效率主体的再配置过程
    """
    
    def __init__(self, config: Dict):
        """
        初始化土地流转参数
        
        Args:
            config: 包含土地流转参数的配置字典
        """
        self.transfer_factors = config.get("transfer_probability_factors", {
            "land_quality_weight": 0.3,
            "technology_level_weight": 0.4,
            "subsidy_weight": 0.2,
            "social_network_weight": 0.1
        })
        
        self.price_mechanism = config.get("price_mechanism", {
            "base_price": 500,
            "quality_multiplier": 1.5,
            "technology_multiplier": 1.3
        })
    
    def calculate_transfer_probability(self, land_quality: float,
                                    lessee_tech_level: float,
                                    lessor_tech_level: float,
                                    subsidy_level: float,
                                    social_network_strength: float) -> float:
        """
        计算土地流转概率
        综合考虑土地质量、技术水平差异、补贴政策、社会网络等因素
        
        Args:
            land_quality: 土地质量
            lessee_tech_level: 承租方技术水平
            lessor_tech_level: 出租方技术水平
            subsidy_level: 补贴水平
            social_network_strength: 社会网络强度
            
        Returns:
            流转概率
        """
        # 技术水平差异效应
        tech_diff = lessee_tech_level - lessor_tech_level
        tech_effect = max(0, tech_diff) * self.transfer_factors["technology_level_weight"]
        
        # 土地质量效应
        quality_effect = land_quality * self.transfer_factors["land_quality_weight"]
        
        # 补贴效应
        subsidy_effect = subsidy_level * self.transfer_factors["subsidy_weight"]
        
        # 社会网络效应
        network_effect = social_network_strength * self.transfer_factors["social_network_weight"]
        
        # 综合计算流转概率
        transfer_prob = tech_effect + quality_effect + subsidy_effect + network_effect
        
        return np.clip(transfer_prob, 0.0, 1.0)
    
    def calculate_transfer_price(self, land_quality: float,
                               lessee_tech_level: float,
                               base_price: float,
                               subsidy_level: float = 0.0) -> float:
        """
        计算土地流转价格
        基于土地质量和承租方技术水平确定流转价格
        
        @new 2025-12-18: 加入补贴-租金联动机制
        理论依据: 补贴提高农户自耕收益，提升保留价格(reserve price)
        根据实证研究，补贴每增加100元/亩，租金上涨20-40元/亩
        
        Args:
            land_quality: 土地质量
            lessee_tech_level: 承租方技术水平
            base_price: 基础价格
            subsidy_level: 补贴水平(元/亩)
            
        Returns:
            流转价格
        """
        # 土地质量加成
        quality_multiplier = 1.0 + (land_quality - 0.5) * self.price_mechanism["quality_multiplier"]
        
        # 技术水平加成
        tech_multiplier = 1.0 + (lessee_tech_level - 0.7) * self.price_mechanism["technology_multiplier"]
        
        # @deprecated 2025-12-18: 移除直接的补贴-租金传导
        # 原代码：subsidy_effect = subsidy_level * 0.30
        # 问题：边际效应恒定，无涌现性，不符合经济学规律
        
        # @new: 补贴通过农户支付意愿间接影响租金
        # 实现位置：farmer_agent.py中的calculate_max_willing_to_pay_rent()
        # 机制：补贴↑ → 种粮利润↑ → 支付意愿↑ → 市场租金↑（由供需决定）
        # 边际效应递减会自然涌现（利润空间有限，竞争趋于饱和）
        
        # 计算最终价格（仅基于土地质量和技术水平）
        final_price = base_price * quality_multiplier * tech_multiplier
        
        return max(final_price, base_price * 0.5)  # 设置最低价格下限


class GrainSubsidyMechanism:
    """
    种粮补贴机制实现
    模拟政府种粮补贴对农户决策行为的影响
    """
    
    def __init__(self, config: Dict):
        """
        初始化种粮补贴参数
        
        Args:
            config: 包含种粮补贴参数的配置字典
        """
        self.subsidy_scenarios = config.get("subsidy_scenarios", {
            "no_subsidy": 0,
            "normal_subsidy": 75,
            "high_subsidy": 150
        })
        self.subsidy_unit = config.get("unit", "元/亩/年")
    
    def calculate_subsidy_payment(self, cultivated_area: float,
                                subsidy_level: str = "normal_subsidy") -> float:
        """
        计算种粮补贴金额
        
        Args:
            cultivated_area: 种植面积（亩）
            subsidy_level: 补贴水平
            
        Returns:
            补贴金额
        """
        subsidy_rate = self.subsidy_scenarios.get(subsidy_level, 75)
        return cultivated_area * subsidy_rate
    
    def update_reference_point(self, current_reference: float,
                             subsidy_payment: float,
                             prospect_theory: ProspectTheoryMechanism) -> float:
        """
        更新农户心理参考点
        补贴直接影响农户的前景理论参考点
        
        Args:
            current_reference: 当前参考点
            subsidy_payment: 补贴金额
            prospect_theory: 前景理论机制对象
            
        Returns:
            更新后的参考点
        """
        # 补贴提升参考点，降低风险敏感性
        reference_increase = subsidy_payment * 0.8  # 补贴的80%转化为参考点提升
        return current_reference + reference_increase


class MechanismCoordinator:
    """
    机制协调器
    整合所有核心机制，提供统一的接口
    
    @fix 2024-11-20: 实现单例模式，避免重复实例化
    专家审阅意见：避免每个Agent重复实例化相同的计算逻辑，节省内存
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: str = "config/model_params.json", config_dict: Optional[Dict] = None):
        """
        单例模式实现
        
        Args:
            config_path: 参数配置文件路径（如果config_dict为None）
            config_dict: 配置字典（优先使用）
        """
        if cls._instance is None:
            cls._instance = super(MechanismCoordinator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config/model_params.json", config_dict: Optional[Dict] = None):
        """
        初始化机制协调器（单例模式，只初始化一次）
        
        Args:
            config_path: 参数配置文件路径（如果config_dict为None）
            config_dict: 配置字典（优先使用）
        """
        # 如果已经初始化过，直接返回
        if MechanismCoordinator._initialized:
            return
        
        # 加载配置
        if config_dict is not None:
            self.config = config_dict
        else:
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[1]
            full_config_path = project_root / config_path
            if full_config_path.exists():
                with open(full_config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # 如果配置文件不存在，使用默认配置
                self.config = {}
        
        # 初始化各机制
        self.prospect_theory = ProspectTheoryMechanism(
            self.config.get("prospect_theory", {})
        )
        self.melitz_heterogeneity = MelitzHeterogeneityMechanism(
            self.config.get("technology_level", {})
        )
        self.innovation_diffusion = InnovationDiffusionMechanism(
            self.config.get("innovation_diffusion", {})
        )
        self.land_transfer = LandTransferMechanism(
            self.config.get("land_transfer", {})
        )
        self.grain_subsidy = GrainSubsidyMechanism(
            self.config.get("grain_subsidy", {})
        )
        config_seed = None
        if isinstance(self.config, dict):
            config_seed = self.config.get("seed")
            if config_seed is None:
                config_seed = self.config.get("experiment", {}).get("seed")
        self.seed = config_seed
        self.np_random = build_numpy_rng(self.seed, "mechanism_coordinator")
        
        # 标记已初始化
        MechanismCoordinator._initialized = True
    
    def initialize_agent_characteristics(self, num_agents: int) -> Dict:
        """
        初始化智能体特征
        基于理论参数生成智能体的初始特征分布
        
        Args:
            num_agents: 智能体数量
            
        Returns:
            智能体特征字典
        """
        # 生成技术水平分布
        tech_levels = self.melitz_heterogeneity.generate_technology_levels(num_agents)
        
        # 生成风险偏好分布（基于前景理论参数）
        risk_preferences = self.np_random.beta(2, 3, num_agents)  # 偏保守的分布
        
        # 生成参考点分布
        reference_points = self.np_random.normal(10000, 2000, num_agents)  # 年收入参考点
        
        return {
            "technology_levels": tech_levels,
            "risk_preferences": risk_preferences,
            "reference_points": reference_points
        }
    
    def make_decision(self, agent_characteristics: Dict,
                     decision_context: Dict) -> Dict:
        """
        综合决策机制
        整合前景理论、企业异质性、创新扩散等机制进行决策
        
        Args:
            agent_characteristics: 智能体特征
            decision_context: 决策情境
            
        Returns:
            决策结果
        """
        # 前景理论决策
        prospect_utility = self.prospect_theory.decision_utility(
            decision_context.get("outcomes", []),
            decision_context.get("probabilities", []),
            agent_characteristics.get("reference_point", 10000)
        )
        
        # 创新采纳决策
        adoption_prob = self.innovation_diffusion.calculate_adoption_probability(
            agent_characteristics,
            decision_context.get("social_network_influence", 0.0),
            decision_context.get("subsidy_level", 0.0)
        )
        
        # 土地流转决策
        transfer_prob = self.land_transfer.calculate_transfer_probability(
            decision_context.get("land_quality", 0.5),
            agent_characteristics.get("technology_level", 0.7),
            decision_context.get("lessor_tech_level", 0.6),
            decision_context.get("subsidy_level", 0.0),
            decision_context.get("social_network_strength", 0.0)
        )
        
        return {
            "prospect_utility": prospect_utility,
            "adoption_probability": adoption_prob,
            "transfer_probability": transfer_prob,
            "decision": self._integrate_decisions(prospect_utility, adoption_prob, transfer_prob)
        }
    
    def _integrate_decisions(self, prospect_utility: float,
                           adoption_prob: float,
                           transfer_prob: float) -> str:
        """
        整合各项决策
        基于综合效用做出最终决策
        
        Args:
            prospect_utility: 前景理论效用
            adoption_prob: 采纳概率
            transfer_prob: 流转概率
            
        Returns:
            最终决策
        """
        # 简化的决策整合逻辑
        if prospect_utility > 0 and adoption_prob > 0.5:
            return "adopt_innovation"
        elif prospect_utility > 0 and transfer_prob > 0.6:
            return "transfer_land"
        elif prospect_utility > 0:
            return "maintain_status_quo"
        else:
            return "risk_averse_action"
