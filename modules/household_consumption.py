# -*- coding: utf-8 -*-
"""
@module: household_consumption
农户消费和福利模块
基于扩展线性支出系统(ELES)和Stone-Geary效用函数
"""

import numpy as np
from enum import Enum

class ConsumptionCategory(Enum):
    """消费类别"""
    FOOD = "food"              # 食物消费
    HOUSING = "housing"        # 居住
    HEALTHCARE = "healthcare"  # 医疗
    EDUCATION = "education"    # 教育
    OTHER = "other"           # 其他

class HouseholdConsumptionModel:
    """
    农户消费决策模型
    基于扩展线性支出系统(Extended Linear Expenditure System, ELES)
    理论基础：Stone-Geary效用函数
    
    U = Σ βi * ln(qi - γi)
    其中：
    - qi: 第i类商品的消费量
    - γi: 第i类商品的基本需求量（subsistence level）
    - βi: 第i类商品的边际消费倾向
    """
    
    def __init__(self, config=None):
        """
        初始化消费模型
        参数根据《中国统计年鉴》和《中国农村住户调查年鉴》2020年数据校准
        """
        self.config = config or {}
        
        # 基本需求参数（γi）- 单位：元/年/人
        # 根据2021年农村居民支出结构实际数据校准（经济日报2022年报告）
        # 农村居民食品烟酒占比32%，居住占比21%，人均总支出约15000元
        self.subsistence_consumption = {
            ConsumptionCategory.FOOD: 2400,        # 食物基本需求：2400元/年/人（15000*32%*0.5）
            ConsumptionCategory.HOUSING: 1500,     # 居住基本需求：1500元/年/人
            ConsumptionCategory.HEALTHCARE: 900,   # 医疗基本需求：900元/年/人
            ConsumptionCategory.EDUCATION: 600,    # 教育基本需求：600元/年/人
            ConsumptionCategory.OTHER: 800         # 其他基本需求：800元/年/人
        }
        
        # 边际消费倾向（βi）
        # 根据2021年中国农村居民实际消费支出结构校准
        self.marginal_propensity = {
            ConsumptionCategory.FOOD: 0.32,        # 食物边际消费倾向：32%（实际占比）
            ConsumptionCategory.HOUSING: 0.21,     # 居住边际消费倾向：21%（实际占比）
            ConsumptionCategory.HEALTHCARE: 0.13,  # 医疗边际消费倾向：13%
            ConsumptionCategory.EDUCATION: 0.11,   # 教育边际消费倾向：11%
            ConsumptionCategory.OTHER: 0.23        # 其他边际消费倾向：23%
        }
        
        # 食物消费详细参数
        # 粮食需求：根据农业农村部2020年实际数据
        # 来源：《中国农业产业发展报告2021》+ 新京报2024年报道
        self.grain_subsistence_per_capita = 193  # 基本粮食需求：193kg/年/人（2020年实际人均谷物消费）
        
        # @literature_basis 2025-12-26: 粮食需求价格弹性
        # 文献综述:
        #   - Huang & Rozelle (1995): China grain price elasticity = -0.52
        #   - Hus et al. (2002): Urban -0.16, Rural -0.37（农村需求弹性较高）
        #   - Zhuang & Abbott (2005): Wheat -0.298, Rice -0.352, Corn -0.476
        #   - USDA-ERS Meta-analysis: China aggregate -0.16 to -0.37
        #   - 本模型取值 -0.3：位于Hus农村估计(-0.37)与城市(-0.16)中点偏农村侧
        # @validation: 与FAO Global Food Demand Elasticity Database一致
        # DOI: 10.1016/j.foodpol.2002.08.005 (Hus et al.)
        self.grain_price_elasticity = -0.3  # 粮食需求价格弹性（文献支持范围：-0.16 to -0.52）
        
        # 食物恩格尔系数参数
        self.food_engel_coefficient_params = {
            'base': 0.30,           # 基准恩格尔系数（2020年农村居民）
            'income_elasticity': -0.4  # 收入弹性（随收入增加而下降）
        }
    
    def calculate_consumption_bundle(self, household_income, household_size, grain_price, 
                                      self_produced_grain_kg=0, cash_budget=None):
        """
        计算农户消费束
        
        参数:
        - household_income: 家庭总收入（元/年）
        - household_size: 家庭人口数
        - grain_price: 粮食价格（元/斤）
        - self_produced_grain_kg: 自产粮食（kg），用于自给消费
        
        @critical 2025-12: 增加自给粮食考虑
        问题：原模型没考虑粮食自给部分不进入市场，高估粮食供给弹性
        解决：区分自给粮和商品粮
        
        理论依据：
        1. 中国农户平均自给率60-70%（农业农村部2020）
        2. 自给粮不产生市场交易，不影响价格
        3. 只有商品粮才进入市场供给
        
        返回:
        - consumption_bundle: 各类消费支出字典
        - food_consumption_details: 食物消费详情
        - welfare_indicators: 福利指标字典
        """
        # 计算人均收入
        per_capita_income = household_income / household_size if household_size > 0 else 0
        
        # 计算总基本需求支出
        total_subsistence = sum(self.subsistence_consumption.values()) * household_size
        
        # @critical 2025-12: 计算自给粮食价值（不计入货币支出但提供效用）
        # 自给粮食满足的需求价值
        basic_grain_need_kg = self.grain_subsistence_per_capita * household_size
        grain_self_consumption_kg = min(self_produced_grain_kg, basic_grain_need_kg)
        grain_self_consumption_value = grain_self_consumption_kg * 2 * grain_price  # 折算为市场价值
        
        # 调整后的食物基本需求支出（扣除自给部分）
        food_subsistence_original = self.subsistence_consumption[ConsumptionCategory.FOOD] * household_size
        food_subsistence_adjusted = max(0, food_subsistence_original - grain_self_consumption_value)
        
        # 调整后的总基本需求
        total_subsistence_adjusted = total_subsistence - (food_subsistence_original - food_subsistence_adjusted)
        
        # 计算超额收入（可自由支配收入）
        discretionary_income = max(0, household_income - total_subsistence_adjusted)
        
        # 根据ELES模型分配支出
        consumption_bundle = {}
        for category in ConsumptionCategory:
            if category == ConsumptionCategory.FOOD:
                # 食物支出：扣除自给粮价值后的基本需求 + 边际消费
                subsistence_expenditure = food_subsistence_adjusted
            else:
                subsistence_expenditure = self.subsistence_consumption[category] * household_size
            
            # 边际消费部分
            marginal_expenditure = self.marginal_propensity[category] * discretionary_income
            
            # 总支出
            consumption_bundle[category] = subsistence_expenditure + marginal_expenditure

        if cash_budget is not None:
            cash_budget = float(cash_budget)
            if not np.isfinite(cash_budget) or cash_budget < 0:
                raise ValueError("cash_budget must be finite and nonnegative")
            demand = sum(consumption_bundle.values())
            share = min(1.0, cash_budget / demand) if demand > 0 else 0.0
            # Explicit no-credit allocation: preserve demand shares, including basic needs.
            consumption_bundle = {category: amount * share for category, amount in consumption_bundle.items()}
        
        # 计算食物消费详情（包含自给粮信息）
        food_consumption_details = self._calculate_food_consumption(
            consumption_bundle[ConsumptionCategory.FOOD],
            household_size,
            grain_price,
            self_produced_grain_kg=self_produced_grain_kg,
            basic_grain_priority=(cash_budget is not None),
        )
        
        # 计算福利指标
        welfare_indicators = self._calculate_welfare_indicators(
            household_income,
            household_size,
            consumption_bundle,
            food_consumption_details,
            grain_price
        )
        
        return consumption_bundle, food_consumption_details, welfare_indicators
    
    def _calculate_food_consumption(self, food_expenditure, household_size, grain_price,
                                     self_produced_grain_kg=0, basic_grain_priority=False):
        """
        计算食物消费详情
        
        @critical 2025-12: 区分自给粮和购买粮
        
        返回:
        - grain_consumption_total: 总粮食消费量（kg）
        - grain_consumption_self: 自给粮消费量（kg）
        - grain_consumption_purchased: 购买粮消费量（kg）
        - grain_expenditure: 粮食支出（元）—只算购买部分
        - other_food_expenditure: 其他食物支出（元）
        """
        # 基本粮食需求量
        basic_grain_need = self.grain_subsistence_per_capita * household_size
        
        # @critical 2025-12: 分离自给粮和购买粮
        # 先用自产粮满足需求
        grain_self_consumption = min(self_produced_grain_kg, basic_grain_need)
        remaining_grain_need = max(0, basic_grain_need - grain_self_consumption)
        
        # 剩余的粮食需求通过市场购买
        basic_grain_expenditure = remaining_grain_need * 2 * grain_price  # 购买部分的支出
        
        # 如果食物支出不足以满足购买需求
        if basic_grain_priority:
            # New cash-budget condition only: basic grain first, then 22% of surplus food spending.
            grain_expenditure = min(food_expenditure, basic_grain_expenditure) + 0.22 * max(0.0, food_expenditure - basic_grain_expenditure)
            grain_purchased = grain_expenditure / (2 * grain_price) if grain_price > 0 else 0.0
            other_food_expenditure = food_expenditure - grain_expenditure
        elif food_expenditure < basic_grain_expenditure:
            grain_purchased = food_expenditure / (2 * grain_price) if grain_price > 0 else 0
            grain_expenditure = food_expenditure
            other_food_expenditure = 0
        else:
            # 超额食物支出按一定比例分配
            grain_ratio = 0.22  # 粮食占食物支出22%
            grain_expenditure = min(food_expenditure * grain_ratio, 
                                   remaining_grain_need * 2 * grain_price + food_expenditure * 0.05)
            grain_purchased = grain_expenditure / (2 * grain_price) if grain_price > 0 else 0
            other_food_expenditure = food_expenditure - grain_expenditure
        
        # 总粮食消费 = 自给 + 购买
        grain_consumption_total = grain_self_consumption + grain_purchased
        
        return {
            'grain_consumption_kg': grain_consumption_total,
            'grain_consumption_self': grain_self_consumption,  # @new: 自给粮
            'grain_consumption_purchased': grain_purchased,    # @new: 购买粮
            'grain_expenditure': grain_expenditure,            # 只算购买部分支出
            'other_food_expenditure': other_food_expenditure,
            'total_food_expenditure': food_expenditure,
            'grain_self_sufficiency_ratio': grain_self_consumption / (grain_consumption_total + 1e-6),
            # @new: 自给率指标
            'self_sufficiency_rate': grain_self_consumption / (basic_grain_need + 1e-6)
        }
    
    def _calculate_welfare_indicators(self, household_income, household_size, 
                                     consumption_bundle, food_details, grain_price):
        """
        计算福利指标
        
        返回多个福利度量指标：
        1. 恩格尔系数 (Engel Coefficient)
        2. 人均消费支出
        3. 消费者剩余（近似）
        4. 食物安全指标
        5. Stone-Geary效用水平
        """
        # 总消费支出
        total_consumption = sum(consumption_bundle.values())
        
        # 人均消费支出
        per_capita_consumption = total_consumption / household_size if household_size > 0 else 0
        
        # 恩格尔系数
        engel_coefficient = (consumption_bundle[ConsumptionCategory.FOOD] / 
                            total_consumption if total_consumption > 0 else 0)
        
        # 食物安全指标（FAO四维框架）
        # @fix 2024-11-18: 采用FAO食物安全四维度评估
        # 参考：FAO (2020), "The State of Food Security and Nutrition in the World"
        # 维度：可获得性、可及性、利用率、稳定性
        
        per_capita_grain = food_details['grain_consumption_kg'] / household_size if household_size > 0 else 0
        
        # 1. 可获得性（Availability）：粮食消费量充足度
        availability_score = min(1.0, per_capita_grain / self.grain_subsistence_per_capita)
        
        # 2. 可及性（Access）：经济可负担性
        # 粮食支出占收入比例（应<30%）
        food_expenditure_share = consumption_bundle[ConsumptionCategory.FOOD] / household_income if household_income > 0 else 1.0
        access_score = max(0, 1.0 - (food_expenditure_share - 0.30) / 0.20) if food_expenditure_share > 0.30 else 1.0
        
        # 3. 利用率（Utilization）：营养多样性
        # 非粮食物支出占比（蛋白质、蔬菜等）
        non_grain_food_share = food_details['other_food_expenditure'] / food_details['total_food_expenditure'] if food_details['total_food_expenditure'] > 0 else 0
        utilization_score = min(1.0, non_grain_food_share / 0.55)  # 理想55%
        
        # 4. 稳定性（Stability）：基于收入稳定性的代理变量
        # 简化：假设农业收入占比高则稳定性低（受天气影响）
        # 这里暂用availability作为代理，实际需要历史数据
        stability_score = availability_score
        
        # 综合食物安全指数（加权平均）
        food_security_index = (availability_score * 0.35 + access_score * 0.30 + 
                              utilization_score * 0.20 + stability_score * 0.15)
        
        # Stone-Geary效用水平（对数形式）
        utility = 0
        for category in ConsumptionCategory:
            consumption = consumption_bundle[category] / household_size  # 人均消费
            subsistence = self.subsistence_consumption[category]
            if consumption > subsistence:
                marginal_propensity = self.marginal_propensity[category]
                utility += marginal_propensity * np.log(consumption - subsistence + 1e-6)
        
        # 消费者剩余（修正：基于补偿变分）
        # @fix 2024-11-18: 使用Stone-Geary效用函数的补偿变分近似
        # CS ≈ 效用水平 × 边际收入效用的倒数
        # 参考：Deaton & Muellbauer (1980), "Economics and Consumer Behavior"
        per_capita_income = household_income / household_size if household_size > 0 else 1
        # 边际收入效用递减：1 / (per_capita_income^0.5)
        # 确保per_capita_income为正，避免负数开方产生复数
        per_capita_income = max(1.0, abs(per_capita_income))
        marginal_utility_income = 1.0 / (per_capita_income ** 0.5 + 1e-6)
        consumer_surplus = utility / (marginal_utility_income + 1e-6)
        # 确保consumer_surplus为实数
        consumer_surplus = float(np.real(consumer_surplus)) if isinstance(consumer_surplus, complex) else float(consumer_surplus)
        
        # 贫困状态判断（多维贫困指数）
        # @fix 2024-11-18: 采用Alkire-Foster多维贫困方法
        # 参考：Alkire & Santos (2014), "Measuring Acute Poverty in the Developing World"
        # 维度：收入、消费、营养、教育、住房
        
        # 1. 收入贫困（相对贫困线：中位数收入的60%）
        # 使用2020年农村居民人均可支配收入17131元的60% = 10279元
        relative_poverty_line = 10279
        income_deprived = (household_income / household_size) < relative_poverty_line if household_size > 0 else True
        
        # 2. 消费贫困（低于基本需求130%）
        consumption_poverty_line = sum(self.subsistence_consumption.values()) * 1.3
        consumption_deprived = per_capita_consumption < consumption_poverty_line
        
        # 3. 营养贫困（粮食消费不足）
        nutrition_deprived = per_capita_grain < (self.grain_subsistence_per_capita * 0.85)
        
        # 4. 教育贫困（教育支出占比过低）
        education_share = consumption_bundle[ConsumptionCategory.EDUCATION] / total_consumption if total_consumption > 0 else 0
        education_deprived = education_share < 0.05  # 低于5%
        
        # 5. 住房贫困（住房支出占比过低）
        housing_share = consumption_bundle[ConsumptionCategory.HOUSING] / total_consumption if total_consumption > 0 else 0
        housing_deprived = housing_share < 0.10  # 低于10%
        
        # 多维贫困：至少2个维度被剥夺
        deprivation_count = sum([income_deprived, consumption_deprived, nutrition_deprived, 
                                education_deprived, housing_deprived])
        is_in_poverty = deprivation_count >= 2
        poverty_intensity = deprivation_count / 5.0  # 贫困强度
        
        # 营养不良风险（综合评估）
        # @fix 2024-11-18: 基于WHO/FAO营养不良评估标准
        # 参考：WHO (2021), "Malnutrition"
        # 考虑：能量摄入、蛋白质摄入、微量营养素
        
        # 1. 能量摄入不足（粮食消费<基本需求85%）
        energy_inadequate = per_capita_grain < (self.grain_subsistence_per_capita * 0.85)
        
        # 2. 蛋白质摄入不足（非粮食物支出过低）
        protein_inadequate = food_details['other_food_expenditure'] / household_size < 1200 if household_size > 0 else True
        
        # 3. 食物多样性不足（营养不良的关键指标）
        diet_diversity_inadequate = utilization_score < 0.5
        
        # 营养不良风险：满足任意两项
        malnutrition_risk = sum([energy_inadequate, protein_inadequate, diet_diversity_inadequate]) >= 2
        
        return {
            'total_consumption': total_consumption,
            'per_capita_consumption': per_capita_consumption,
            'engel_coefficient': engel_coefficient,
            'food_security_index': food_security_index,
            'food_security_components': {
                'availability': availability_score,
                'access': access_score,
                'utilization': utilization_score,
                'stability': stability_score
            },
            'stone_geary_utility': utility,
            'consumer_surplus': consumer_surplus,
            'is_in_poverty': is_in_poverty,
            'poverty_intensity': poverty_intensity,
            'multidimensional_poverty_components': {
                'income': income_deprived,
                'consumption': consumption_deprived,
                'nutrition': nutrition_deprived,
                'education': education_deprived,
                'housing': housing_deprived
            },
            'malnutrition_risk': malnutrition_risk,
            'per_capita_grain_consumption_kg': per_capita_grain,
            'savings_rate': (household_income - total_consumption) / household_income if household_income > 0 else 0
        }
    
    def calculate_leisure_utility(self, labor_days_worked, total_available_days=365,
                                   leisure_preference=0.3, labor_disutility_param=1.5):
        """
        @new 2025-12: 计算闲暇效用
        
        理论基础：Agricultural Household Model (AHM)
        参考：Singh, Squire & Strauss (1986), "Agricultural Household Models: Extensions, Applications, and Policy"
        
        AHM框架：农户同时是生产者和消费者，需要在劳动供给和闲暇之间权衡
        
        效用函数形式：U = U(C, L) = u(C) + v(L)
        其中：
        - C: 消费
        - L: 闲暇时间 = 总时间 - 劳动时间
        - v(L): 闲暇效用函数，采用对数形式
        
        参数:
        - labor_days_worked: 工作天数（农业劳动 + 非农劳动）
        - total_available_days: 总可用天数（默认365天）
        - leisure_preference: 闲暇偏好系数（0-1，越高越重视闲暇）
        - labor_disutility_param: 劳动负效用参数
        
        返回:
        - dict: 包含闲暇效用和劳动负效用的各项指标
        """
        # 计算闲暇天数
        leisure_days = max(0, total_available_days - labor_days_worked)
        
        # 闲暇比例
        leisure_ratio = leisure_days / total_available_days
        
        # 最低闲暇需求（每年至少需要30天休息）
        min_leisure_days = 30
        
        # 闲暇效用（对数形式）
        # v(L) = β * ln(L - L_min + 1)
        if leisure_days > min_leisure_days:
            leisure_utility = leisure_preference * np.log(leisure_days - min_leisure_days + 1)
        else:
            # 闲暇不足，产生负效用
            leisure_utility = leisure_preference * np.log(1) - \
                             (min_leisure_days - leisure_days) * 0.01
        
        # 劳动负效用（过度劳动的代价）
        # 理论：长时间劳动导致健康下降、生活质量降低
        if labor_days_worked > 300:
            # 超过300天工作，产生显著负效用
            overwork_penalty = -labor_disutility_param * \
                              ((labor_days_worked - 300) / 65) ** 2
        elif labor_days_worked > 250:
            # 250-300天，轻度负效用
            overwork_penalty = -labor_disutility_param * 0.1 * \
                              ((labor_days_worked - 250) / 50)
        else:
            overwork_penalty = 0
        
        # 综合劳动-闲暇效用
        labor_leisure_utility = leisure_utility + overwork_penalty
        
        # 劳动强度指标
        if labor_days_worked < 100:
            labor_intensity = "低"
        elif labor_days_worked < 200:
            labor_intensity = "中等"
        elif labor_days_worked < 280:
            labor_intensity = "较高"
        else:
            labor_intensity = "过度"
        
        return {
            'leisure_days': leisure_days,
            'leisure_ratio': leisure_ratio,
            'leisure_utility': leisure_utility,
            'labor_disutility': overwork_penalty,
            'labor_leisure_utility': labor_leisure_utility,
            'labor_intensity': labor_intensity,
            'is_overworked': labor_days_worked > 280,
            'leisure_adequate': leisure_days >= min_leisure_days * 2
        }
    
    def calculate_comprehensive_utility(self, household_income, household_size, 
                                        grain_price, labor_days_worked,
                                        leisure_weight=0.25, consumption_weight=0.75,
                                        consumption_welfare=None):
        """
        @new 2025-12: 计算综合效用（消费效用 + 闲暇效用）
        
        理论基础：Singh, Squire & Strauss (1986) AHM模型
        
        综合效用：U = α * U_consumption + (1-α) * U_leisure
        
        参数:
        - household_income: 家庭总收入
        - household_size: 家庭人口数
        - grain_price: 粮食价格
        - labor_days_worked: 工作天数
        - leisure_weight: 闲暇在效用中的权重（默认0.25）
        - consumption_weight: 消费在效用中的权重（默认0.75）
        
        返回:
        - dict: 综合效用分析结果
        """
        # 计算消费效用
        if consumption_welfare is None:
            _, _, welfare_indicators = self.calculate_consumption_bundle(household_income, household_size, grain_price)
        else:
            welfare_indicators = consumption_welfare
        
        consumption_utility = welfare_indicators['stone_geary_utility']
        
        # 计算闲暇效用
        leisure_result = self.calculate_leisure_utility(labor_days_worked)
        leisure_utility = leisure_result['labor_leisure_utility']
        
        # 综合效用（加权平均，并标准化）
        # 由于两种效用的量级可能不同，需要进行标准化
        # 简化处理：假设消费效用范围在0-10，闲暇效用范围在-2到3
        normalized_consumption_utility = max(0, min(10, consumption_utility))
        normalized_leisure_utility = (leisure_utility + 2) / 5 * 10  # 映射到0-10
        normalized_leisure_utility = max(0, min(10, normalized_leisure_utility))
        
        comprehensive_utility = (consumption_weight * normalized_consumption_utility + 
                                leisure_weight * normalized_leisure_utility)
        
        # 生活质量指数（0-100）
        quality_of_life_index = comprehensive_utility * 10
        
        # 工作生活平衡评估
        if leisure_result['is_overworked']:
            work_life_balance = "失衡（过度劳动）"
        elif leisure_result['leisure_adequate']:
            work_life_balance = "良好"
        else:
            work_life_balance = "一般"
        
        return {
            'comprehensive_utility': comprehensive_utility,
            'consumption_utility': consumption_utility,
            'leisure_utility': leisure_utility,
            'normalized_consumption_utility': normalized_consumption_utility,
            'normalized_leisure_utility': normalized_leisure_utility,
            'quality_of_life_index': quality_of_life_index,
            'work_life_balance': work_life_balance,
            'consumption_details': welfare_indicators,
            'leisure_details': leisure_result
        }
    
    def calculate_compensating_variation(self, initial_income, final_income, 
                                        household_size, grain_price):
        """
        计算补偿变分（Compensating Variation, CV）
        衡量政策变化对福利的影响
        
        CV表示：在价格或政策变化后，需要给予（或取走）消费者多少收入，
        才能使其达到变化前的效用水平
        
        参数:
        - initial_income: 初始收入
        - final_income: 最终收入
        - household_size: 家庭人口数
        - grain_price: 粮食价格
        
        返回:
        - cv: 补偿变分（正值表示福利提高，负值表示福利降低）
        """
        # 计算初始消费和效用
        _, _, initial_welfare = self.calculate_consumption_bundle(
            initial_income, household_size, grain_price
        )
        initial_utility = initial_welfare['stone_geary_utility']
        
        # 计算最终消费和效用
        _, _, final_welfare = self.calculate_consumption_bundle(
            final_income, household_size, grain_price
        )
        final_utility = final_welfare['stone_geary_utility']
        
        # 补偿变分近似计算
        # CV ≈ (U_final - U_initial) * marginal_utility_of_income
        # 边际收入效用近似为 1/per_capita_income
        per_capita_income = initial_income / household_size if household_size > 0 else 1
        marginal_utility_income = 1.0 / (per_capita_income + 1e-6)
        
        cv = (final_utility - initial_utility) / (marginal_utility_income + 1e-6)
        
        return {
            'compensating_variation': cv,
            'utility_change': final_utility - initial_utility,
            'welfare_improved': cv > 0
        }
