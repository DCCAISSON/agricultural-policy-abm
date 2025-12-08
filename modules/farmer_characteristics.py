# -*- coding: utf-8 -*-
"""
农户特征模块 - 连续变量分类系统
================================

实现基于连续变量的农户分类，替代简单的三分法离散标签。

核心连续变量：
1. 农业依赖度 (agricultural_dependence): 农业收入占比，0-1连续
2. 生计多样性指数 (livelihood_diversity): 基于香农指数的收入来源多样性
3. 土地规模 (land_scale): 经营土地面积（亩）
4. 老龄化率 (aging_rate): 家庭>60岁劳动力占比
5. 社会资本强度 (social_capital): 可动员关系数量

农户类型（6-7分法）：
1. 传统生存型小农 (TRADITIONAL_SUBSISTENCE)
2. 现代商业型农户/家庭农场 (COMMERCIAL_FARMER)
3. 兼业I型-以农为主 (PART_TIME_AGRI_MAIN)
4. 兼业II型-以工为主 (PART_TIME_OFF_FARM_MAIN)
5. 新生代职业农民 (NEW_GENERATION_FARMER)
6. 离乡型非农户 (URBAN_MIGRANT)
7. "离土不离权"地主型 (ABSENTEE_LANDLORD)

@version: 1.0 (2025-12)
@theory: 基于Deininger & Jin (2005)土地市场理论 + 农户分化文献
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np


class FarmerType(Enum):
    """
    精细农户类型（8分法）
    基于中国农村实际情况的更精细分类
    
    @update 2025-12: 新增 WEEKEND_FARMER 类型
    """
    # 1. 传统生存型小农：60岁以上为主，耕地<10亩，商品率<30%
    TRADITIONAL_SUBSISTENCE = "Traditional_Subsistence"
    
    # 2. 现代商业型农户（家庭农场/专业大户）：规模10-200亩，商品率>80%
    COMMERCIAL_FARMER = "Commercial_Farmer"
    
    # 3. 兼业I型（以农为主）：非农收入占比20%-50%
    PART_TIME_AGRI_MAIN = "PartTime_Agri_Main"
    
    # 4. 兼业II型（以工为主）：非农收入占比>70%
    PART_TIME_OFF_FARM_MAIN = "PartTime_OffFarm_Main"
    
    # 5. 新生代职业农民：80后90后返乡或农业专业毕业
    NEW_GENERATION_FARMER = "New_Generation_Farmer"
    
    # 6. 离乡型非农户（彻底退出）：全家已在城市
    URBAN_MIGRANT = "Urban_Migrant"
    
    # 7. "离土不离权"地主型非农：保留承包权，靠出租获租金
    ABSENTEE_LANDLORD = "Absentee_Landlord"
    
    # 8. "周末农民"型（部分接班）：城市工作+保留30-50%自耕
    # @new 2025-12: 代际接替时的"部分接班"模式
    WEEKEND_FARMER = "Weekend_Farmer"


@dataclass
class FarmerCharacteristics:
    """
    农户连续特征变量集合
    用连续变量替代离散标签，实现更精细的异质性刻画
    """
    
    # ==================== 核心连续变量 ====================
    
    # 农业依赖度：农业收入占总收入比例 [0, 1]
    agricultural_dependence: float = 0.5
    
    # 生计多样性指数：基于香农指数 [0, 1]
    # 计算公式：1 - Shannon Index，值越高表示收入来源越单一
    livelihood_diversity: float = 0.5
    
    # 土地规模：经营土地面积（亩）
    land_scale: float = 10.0
    
    # 家庭劳动力老龄化率：>60岁劳动力占家庭总劳动力比例 [0, 1]
    aging_rate: float = 0.3
    
    # 社会资本强度：可动员的关系数量（亲友、村民等）
    social_capital: float = 5.0
    
    # ==================== 补充特征 ====================
    
    # 户主年龄
    household_head_age: int = 50
    
    # 是否有子女进城
    children_in_city: bool = False
    
    # 子女是否愿意接班务农
    successor_willing: bool = False
    
    # 代际类型：1=60后以前, 2=60-70后, 3=80后, 4=90后及以后
    generation_type: int = 2
    
    # @new 2025-12: 进城经历相关特征
    # 是否有进城务工经历（户主本人）
    has_urban_experience: bool = False
    
    # 进城务工年数
    urban_work_years: float = 0.0
    
    # 是否当前在城市务工（季节性或长期）
    currently_working_in_city: bool = False
    
    # @new 2025-12: 家庭劳动力结构
    # 家庭男性劳动力占比 [0, 1]
    male_labor_ratio: float = 0.5
    
    # 家庭劳动力总人数
    total_labor_count: int = 2
    
    # 16-60岁劳动力占比
    prime_age_labor_ratio: float = 0.7
    
    # 商品化率：农产品销售比例 [0, 1]
    commercialization_rate: float = 0.5
    
    # 技术采纳指数：对新技术、新模式的接受程度 [0, 1]
    technology_adoption: float = 0.3
    
    # 风险承受能力 [0, 1]
    risk_tolerance: float = 0.3
    
    # 信息获取能力 [0, 1]
    information_access: float = 0.3
    
    # 土地流出比例：已流出土地占承包地比例 [0, 1]
    land_leased_out_ratio: float = 0.0
    
    # 土地流入比例：流入土地占经营地比例 [0, 1]
    land_leased_in_ratio: float = 0.0
    
    # ==================== 收入结构（用于计算依赖度和多样性）====================
    
    # 农业经营收入
    farm_income: float = 0.0
    
    # 工资性收入（务工收入）
    wage_income: float = 0.0
    
    # 财产性收入（土地租金、分红等）
    property_income: float = 0.0
    
    # 转移性收入（补贴、养老金等）
    transfer_income: float = 0.0
    
    # 经营性非农收入（经商、加工等）
    non_farm_business_income: float = 0.0
    
    # ==================== 政策敏感度（关键！）====================
    
    # 对粮食价格补贴的敏感度 [0, 1]
    price_subsidy_sensitivity: float = 0.5
    
    # 对种粮直接补贴的敏感度 [0, 1]
    direct_subsidy_sensitivity: float = 0.5
    
    # 对流转租金补贴的敏感度 [0, 1]
    rental_subsidy_sensitivity: float = 0.5
    
    # 对社会化服务政策的敏感度 [0, 1]
    service_policy_sensitivity: float = 0.5
    
    def update_from_income(self):
        """根据收入结构更新农业依赖度和生计多样性"""
        total_income = (self.farm_income + self.wage_income + 
                       self.property_income + self.transfer_income + 
                       self.non_farm_business_income)
        
        if total_income > 0:
            # 更新农业依赖度
            self.agricultural_dependence = self.farm_income / total_income
            
            # 计算生计多样性（基于香农指数）
            incomes = [self.farm_income, self.wage_income, self.property_income,
                      self.transfer_income, self.non_farm_business_income]
            shares = [inc / total_income for inc in incomes if inc > 0]
            
            if len(shares) > 0:
                # 香农指数
                shannon = -sum(p * np.log(p + 1e-10) for p in shares if p > 0)
                # 归一化到[0,1]，最大熵为ln(5)≈1.61
                max_entropy = np.log(5)
                self.livelihood_diversity = 1 - (shannon / max_entropy)
            else:
                self.livelihood_diversity = 1.0  # 完全单一
    
    def calculate_transfer_out_willingness(self) -> float:
        """
        计算土地流出意愿
        基于多因素的连续函数
        
        Returns:
            float: 流出意愿 [0, 1]
        """
        # 基础意愿：农业依赖度越低，流出意愿越高
        base = 1.0 - self.agricultural_dependence
        
        # 年龄因素：老龄化程度越高，流出意愿越高
        age_factor = 0.0
        if self.household_head_age >= 65:
            age_factor = 0.3
        elif self.household_head_age >= 55:
            age_factor = 0.15
        
        # 继承因素：无人接班则流出意愿更高
        succession_factor = 0.2 if (self.children_in_city and not self.successor_willing) else 0.0
        
        # 社会资本：社会资本高者可能更倾向于熟人流转而非市场流转
        # 但总体流出意愿不受太大影响
        
        # 土地规模：规模越小，流出边际成本越低
        scale_factor = 0.1 if self.land_scale < 5 else 0.0
        
        willingness = min(1.0, base + age_factor + succession_factor + scale_factor)
        return max(0.0, willingness)
    
    def calculate_transfer_in_willingness(self) -> float:
        """
        计算土地流入意愿
        基于多因素的连续函数
        
        Returns:
            float: 流入意愿 [0, 1]
        """
        # 基础意愿：农业依赖度越高，流入意愿越高
        base = self.agricultural_dependence * 0.5
        
        # 年龄因素：中青年更愿意扩张
        age_factor = 0.0
        if 35 <= self.household_head_age <= 55:
            age_factor = 0.2
        elif self.household_head_age < 35:
            age_factor = 0.25  # 新生代更激进
        
        # 规模因素：现有规模越大，扩张动力越强（规模效应）
        scale_factor = 0.0
        if self.land_scale >= 30:
            scale_factor = 0.2
        elif self.land_scale >= 15:
            scale_factor = 0.1
        
        # 技术采纳：技术能力强者更愿意扩张
        tech_factor = self.technology_adoption * 0.15
        
        # 风险承受：风险承受能力强者更愿意扩张
        risk_factor = self.risk_tolerance * 0.1
        
        willingness = min(1.0, base + age_factor + scale_factor + tech_factor + risk_factor)
        return max(0.0, willingness)
    
    def calculate_rental_offer_price(self, base_rental: float) -> float:
        """
        计算土地租金出价（作为需求方）
        
        Args:
            base_rental: 基准租金（元/亩）
        
        Returns:
            float: 愿意支付的租金价格
        """
        # 效率溢价：技术能力强者愿意支付更高租金
        efficiency_premium = 1.0 + self.technology_adoption * 0.2
        
        # 规模溢价：大户愿意支付更高租金以扩大规模
        scale_premium = 1.0
        if self.land_scale >= 50:
            scale_premium = 1.15
        elif self.land_scale >= 30:
            scale_premium = 1.10
        
        # 竞争压力：信息获取能力强者知道市场行情
        market_factor = 1.0 + self.information_access * 0.1
        
        return base_rental * efficiency_premium * scale_premium * market_factor
    
    def calculate_rental_reserve_price(self, base_rental: float) -> float:
        """
        计算土地租金保留价（作为供给方）
        
        Args:
            base_rental: 基准租金（元/亩）
        
        Returns:
            float: 愿意接受的最低租金
        """
        # 基础保留价
        reserve = base_rental * 0.8
        
        # 社会资本折扣：对熟人可能接受更低价格
        if self.social_capital > 10:
            reserve *= 0.9  # 熟人网络广，更可能低价流转给熟人
        
        # 老龄折扣：年龄大者更急于流出
        if self.household_head_age >= 70:
            reserve *= 0.85
        elif self.household_head_age >= 60:
            reserve *= 0.9
        
        # 财产性收入依赖：如果已有较多财产性收入，可能接受较低价格
        if self.property_income > self.farm_income * 0.5:
            reserve *= 0.95
        
        return max(reserve, base_rental * 0.5)  # 最低不低于50%
    
    def calculate_policy_response(self, policy_type: str, policy_intensity: float) -> float:
        """
        计算对特定政策的响应程度
        不同类型农户对不同政策的敏感度差异化
        
        Args:
            policy_type: 政策类型 ('price_subsidy', 'direct_subsidy', 'rental_subsidy', 'service')
            policy_intensity: 政策强度 [0, 1]
        
        Returns:
            float: 响应程度 [0, 1]
        """
        sensitivity_map = {
            'price_subsidy': self.price_subsidy_sensitivity,
            'direct_subsidy': self.direct_subsidy_sensitivity,
            'rental_subsidy': self.rental_subsidy_sensitivity,
            'service': self.service_policy_sensitivity
        }
        
        sensitivity = sensitivity_map.get(policy_type, 0.5)
        
        # 响应 = 政策强度 × 敏感度 × 能力因子
        ability_factor = 0.5 + 0.3 * self.information_access + 0.2 * self.technology_adoption
        
        return min(1.0, policy_intensity * sensitivity * ability_factor)


class FarmerTypeClassifier:
    """
    农户类型分类器
    基于连续变量进行农户类型判定
    """
    
    @staticmethod
    def classify(chars: FarmerCharacteristics) -> Tuple[FarmerType, float]:
        """
        根据连续特征变量判定农户类型
        返回类型和置信度
        
        Args:
            chars: 农户特征变量集合
        
        Returns:
            Tuple[FarmerType, float]: (农户类型, 置信度)
        """
        scores = {}
        
        # 1. 传统生存型小农
        # 特征：60岁以上，耕地<10亩，商品率<30%
        score_traditional = 0.0
        if chars.household_head_age >= 60:
            score_traditional += 0.3
        if chars.land_scale < 10:
            score_traditional += 0.3
        if chars.commercialization_rate < 0.3:
            score_traditional += 0.3
        if chars.agricultural_dependence > 0.5:
            score_traditional += 0.1
        scores[FarmerType.TRADITIONAL_SUBSISTENCE] = score_traditional
        
        # 2. 现代商业型农户/家庭农场
        # 特征：规模10-200亩，商品率>80%，40-55岁
        score_commercial = 0.0
        if 10 <= chars.land_scale <= 200:
            score_commercial += 0.25
        if chars.land_scale > 50:
            score_commercial += 0.15
        if chars.commercialization_rate > 0.8:
            score_commercial += 0.25
        if 40 <= chars.household_head_age <= 55:
            score_commercial += 0.15
        if chars.technology_adoption > 0.5:
            score_commercial += 0.2
        scores[FarmerType.COMMERCIAL_FARMER] = score_commercial
        
        # 3. 兼业I型（以农为主）
        # 特征：非农收入占比20%-50%，仍以农业为主
        score_part1 = 0.0
        non_agri_ratio = 1 - chars.agricultural_dependence
        if 0.2 <= non_agri_ratio <= 0.5:
            score_part1 += 0.4
        if chars.land_scale >= 5:
            score_part1 += 0.2
        if chars.land_leased_in_ratio > 0:
            score_part1 += 0.2
        if chars.livelihood_diversity > 0.3 and chars.livelihood_diversity < 0.7:
            score_part1 += 0.2
        scores[FarmerType.PART_TIME_AGRI_MAIN] = score_part1
        
        # 4. 兼业II型（以工为主）
        # 特征：非农收入占比>70%，务工稳定
        score_part2 = 0.0
        if non_agri_ratio > 0.7:
            score_part2 += 0.4
        if chars.wage_income > chars.farm_income * 2:
            score_part2 += 0.2
        if chars.land_leased_out_ratio > 0.3:
            score_part2 += 0.2
        if 40 <= chars.household_head_age <= 60:
            score_part2 += 0.2
        scores[FarmerType.PART_TIME_OFF_FARM_MAIN] = score_part2
        
        # 5. 新生代职业农民
        # 特征：80后90后，高度市场化，接受新模式
        score_newgen = 0.0
        if chars.generation_type >= 3:  # 80后及以后
            score_newgen += 0.3
        if chars.household_head_age < 40:
            score_newgen += 0.2
        if chars.technology_adoption > 0.6:
            score_newgen += 0.25
        if chars.commercialization_rate > 0.7:
            score_newgen += 0.15
        if chars.risk_tolerance > 0.5:
            score_newgen += 0.1
        scores[FarmerType.NEW_GENERATION_FARMER] = score_newgen
        
        # 6. 离乡型非农户
        # 特征：全家已在城市，土地长期撂荒或托管
        score_migrant = 0.0
        if chars.children_in_city and not chars.successor_willing:
            score_migrant += 0.3
        if chars.agricultural_dependence < 0.1:
            score_migrant += 0.3
        if chars.land_leased_out_ratio > 0.8:
            score_migrant += 0.3
        if chars.farm_income < 1000:  # 几乎无农业收入
            score_migrant += 0.1
        scores[FarmerType.URBAN_MIGRANT] = score_migrant
        
        # 7. "离土不离权"地主型
        # 特征：在城里工作但保留承包权，靠出租获租金
        score_landlord = 0.0
        if chars.property_income > chars.farm_income:
            score_landlord += 0.3
        if chars.land_leased_out_ratio > 0.5 and chars.land_leased_out_ratio < 1.0:
            score_landlord += 0.3
        if chars.agricultural_dependence < 0.3:
            score_landlord += 0.2
        if chars.wage_income > 0 or chars.non_farm_business_income > 0:
            score_landlord += 0.2
        scores[FarmerType.ABSENTEE_LANDLORD] = score_landlord
        
        # 选择得分最高的类型
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # 归一化置信度
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = scores[best_type] / total_score
        
        return best_type, confidence
    
    @staticmethod
    def get_type_policy_sensitivity(farmer_type: FarmerType) -> Dict[str, float]:
        """
        获取特定农户类型的政策敏感度配置
        
        Returns:
            Dict: 各类政策的敏感度
        """
        sensitivity_matrix = {
            FarmerType.TRADITIONAL_SUBSISTENCE: {
                'price_subsidy': 0.9,      # 对价格补贴最敏感
                'direct_subsidy': 0.6,
                'rental_subsidy': 0.2,
                'service': 0.4
            },
            FarmerType.COMMERCIAL_FARMER: {
                'price_subsidy': 0.6,
                'direct_subsidy': 0.9,     # 对直接补贴最敏感
                'rental_subsidy': 0.7,
                'service': 0.8
            },
            FarmerType.PART_TIME_AGRI_MAIN: {
                'price_subsidy': 0.7,
                'direct_subsidy': 0.7,
                'rental_subsidy': 0.5,
                'service': 0.6
            },
            FarmerType.PART_TIME_OFF_FARM_MAIN: {
                'price_subsidy': 0.2,
                'direct_subsidy': 0.4,
                'rental_subsidy': 0.9,     # 对流转补贴最敏感
                'service': 0.3
            },
            FarmerType.NEW_GENERATION_FARMER: {
                'price_subsidy': 0.5,
                'direct_subsidy': 0.8,
                'rental_subsidy': 0.8,
                'service': 0.9             # 对社会化服务最敏感
            },
            FarmerType.URBAN_MIGRANT: {
                'price_subsidy': 0.1,
                'direct_subsidy': 0.2,
                'rental_subsidy': 0.5,
                'service': 0.2
            },
            FarmerType.ABSENTEE_LANDLORD: {
                'price_subsidy': 0.1,
                'direct_subsidy': 0.1,
                'rental_subsidy': 0.95,    # 对流转租金极其敏感
                'service': 0.2
            },
        }
        
        return sensitivity_matrix.get(farmer_type, {
            'price_subsidy': 0.5,
            'direct_subsidy': 0.5,
            'rental_subsidy': 0.5,
            'service': 0.5
        })


class LifecycleManager:
    """
    家庭生命周期管理器
    处理代际接替、老龄化事件等
    
    @critical 2025-12: 代际接替机制
    理论依据：中国农村土地流转爆发式增长的根本驱动力是：
    - 60后户主65岁前后集中出租（2020-2023年数据）
    - 90后拒绝回乡务农
    
    参考：
    - 王济民等 (2023) 《农村土地流转的代际效应》
    - 中国农村固定观察点数据 (2020-2023)
    """
    
    # @critical 2025-12: 代际接替核心参数
    # 户主65岁+流出概率每年+25%
    BASE_AGING_OUTFLOW_PROB = 0.10  # 基础流出概率
    AGING_OUTFLOW_INCREMENT = 0.25  # 每年增加
    
    # 90后子女城市落户后，流出概率再+40%
    CHILD_URBAN_SETTLED_BONUS = 0.40
    
    # 子女返乡创业，逆转成流入概率
    CHILD_RETURN_INFLOW_PROB = 0.60
    
    @staticmethod
    def calculate_land_outflow_probability(chars: FarmerCharacteristics) -> float:
        """
        @critical 2025-12: 计算土地流出概率
        
        核心公式：
        P(流出) = base + (age-65) × 0.25 + child_urban_bonus + generation_factor
        
        Args:
            chars: 农户特征
        
        Returns:
            float: 土地流出概率 [0, 1]
        """
        prob = 0.0
        age = chars.household_head_age
        
        # 1. 年龄因素：65岁+每年增加25%
        if age >= 65:
            years_over_65 = age - 65
            prob = LifecycleManager.BASE_AGING_OUTFLOW_PROB + years_over_65 * LifecycleManager.AGING_OUTFLOW_INCREMENT
        elif age >= 60:
            # 60-65岁：初现流出倾向
            prob = 0.05 + (age - 60) * 0.02
        
        # 2. 子女城市定居因素：+40%
        if chars.children_in_city and not chars.successor_willing:
            prob += LifecycleManager.CHILD_URBAN_SETTLED_BONUS
        
        # 3. 代际因素
        # 第一代农民（50后60后）：土地情结强，流出概率-10%
        if chars.generation_type == 1:
            prob -= 0.10
        # 新生代（80后90后）：若不愿务农，流出概率+20%
        elif chars.generation_type >= 3 and not chars.successor_willing:
            prob += 0.20
        
        # 4. 健康因素：老龄化率高，流出概率增加
        if chars.aging_rate > 0.5:
            prob += (chars.aging_rate - 0.5) * 0.3
        
        return max(0, min(1, prob))
    
    @staticmethod
    def calculate_land_inflow_probability(chars: FarmerCharacteristics, economic_conditions: dict = None) -> float:
        """
        @critical 2025-12: 计算土地流入概率（扩大经营）
        
        核心条件：
        1. 子女返乡创业
        2. 年轻+技术采纳能力强
        3. 经济条件有利
        
        Args:
            chars: 农户特征
            economic_conditions: 经济条件
        
        Returns:
            float: 土地流入概率 [0, 1]
        """
        if economic_conditions is None:
            economic_conditions = {}
        
        prob = 0.0
        
        # 1. 子女返乡创业：流入概率+60%
        # 这是从"流出"到"流入"的关键逆转
        if chars.successor_willing and chars.generation_type >= 3:
            prob = LifecycleManager.CHILD_RETURN_INFLOW_PROB
        
        # 2. 年轻+技术能力强
        if chars.household_head_age < 45 and chars.technology_adoption > 0.6:
            prob += 0.25
        
        # 3. 已有规模基础
        if chars.land_scale > 30:  # 已有较大规模
            prob += 0.15
        
        # 4. 经济条件有利
        agri_profit_ratio = economic_conditions.get('agri_profit_ratio', 1.0)
        if agri_profit_ratio > 1.3:  # 农业收益比非农高30%+
            prob *= 1.3
        
        # 5. 社会资本强（能够低成本获得土地）
        if chars.social_capital > 8:
            prob += 0.10
        
        return max(0, min(1, prob))
    
    @staticmethod
    def check_aging_event(chars: FarmerCharacteristics, random_state=None) -> Optional[str]:
        """
        检查是否触发老龄化相关事件
        
        @critical 2025-12: 增强代际接替逻辑
        
        Args:
            chars: 农户特征
            random_state: 随机数生成器
        
        Returns:
            Optional[str]: 事件类型，None表示无事件
            - 'land_outflow': 土地流出事件
            - 'land_inflow': 土地流入事件（子女返乡）
            - 'health_shock': 健康冲击
            - 'succession': 代际接替完成
            - 'urban_migration': 全家城市迁移
        """
        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state
        
        # 计算流出和流入概率
        outflow_prob = LifecycleManager.calculate_land_outflow_probability(chars)
        inflow_prob = LifecycleManager.calculate_land_inflow_probability(chars)
        
        # 流出和流入互斥，取较高者
        if outflow_prob > inflow_prob:
            if rng.random() < outflow_prob:
                # 检查是否全家迁移（子女已在城市且农户年龄大）
                if chars.children_in_city and chars.household_head_age >= 70:
                    if rng.random() < 0.3:  # 30%概率全家迁移
                        return 'urban_migration'
                return 'land_outflow'
        else:
            if rng.random() < inflow_prob:
                return 'land_inflow'
        
        # 健康冲击事件（独立）
        if chars.household_head_age >= 55:
            health_shock_prob = 0.02 + (chars.household_head_age - 55) * 0.005
            if rng.random() < health_shock_prob:
                return 'health_shock'
        
        # 代际接替完成事件
        if chars.household_head_age >= 60 and chars.successor_willing:
            succession_prob = 0.08 + (chars.household_head_age - 60) * 0.02
            if rng.random() < succession_prob:
                return 'succession'
        
        return None
    
    @staticmethod
    def check_return_event(chars: FarmerCharacteristics, economic_conditions: dict, random_state=None) -> bool:
        """
        检查是否触发返乡创业事件
        
        Args:
            chars: 农户特征
            economic_conditions: 经济条件（城市就业率、农业收益等）
            random_state: 随机数生成器
        
        Returns:
            bool: 是否触发返乡
        """
        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state
        
        # 基础返乡概率
        base_prob = 0.02
        
        # 城市就业恶化增加返乡概率
        urban_employment = economic_conditions.get('urban_employment_rate', 0.95)
        if urban_employment < 0.90:
            base_prob += (0.90 - urban_employment) * 0.5
        
        # 农业收益好转增加返乡概率
        agri_profit_ratio = economic_conditions.get('agri_profit_ratio', 1.0)
        if agri_profit_ratio > 1.2:
            base_prob += (agri_profit_ratio - 1.2) * 0.1
        
        # 新生代更可能返乡创业
        if chars.generation_type >= 3 and chars.technology_adoption > 0.5:
            base_prob *= 1.5
        
        # 有土地可以收回增加返乡动力
        if chars.land_leased_out_ratio > 0.5:
            base_prob *= 1.2
        
        return rng.random() < base_prob


def initialize_farmer_characteristics(
    age: int = 50,
    land_area: float = 10.0,
    initial_capital: float = 50000,
    generation: int = 2,
    random_state = None
) -> FarmerCharacteristics:
    """
    初始化农户特征变量
    根据基本参数生成完整的特征集
    
    Args:
        age: 户主年龄
        land_area: 土地面积（亩）
        initial_capital: 初始资本
        generation: 代际类型
        random_state: 随机数生成器
    
    Returns:
        FarmerCharacteristics: 初始化的特征对象
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    
    chars = FarmerCharacteristics()
    
    # 基本属性
    chars.household_head_age = age
    chars.land_scale = land_area
    chars.generation_type = generation
    
    # @new 2025-12: 进城经历（根据年龄和代际推断）
    # 年轻人和80后90后更可能有进城务工经历
    if generation >= 3 or age < 45:
        chars.has_urban_experience = rng.random() < 0.85  # 85%有进城经历
        if chars.has_urban_experience:
            chars.urban_work_years = rng.uniform(2, 15)
            chars.currently_working_in_city = rng.random() < 0.60  # 60%当前仍在城市
    elif generation == 2 or (45 <= age < 55):
        chars.has_urban_experience = rng.random() < 0.65  # 65%有进城经历
        if chars.has_urban_experience:
            chars.urban_work_years = rng.uniform(1, 10)
            chars.currently_working_in_city = rng.random() < 0.35  # 35%当前在城市
    else:  # 老年人
        chars.has_urban_experience = rng.random() < 0.30  # 30%有进城经历
        if chars.has_urban_experience:
            chars.urban_work_years = rng.uniform(0.5, 5)
            chars.currently_working_in_city = rng.random() < 0.10  # 10%当前在城市
    
    # @new 2025-12: 家庭劳动力结构
    # 男性劳动力占比（农村家庭男性占比约50-60%）
    chars.male_labor_ratio = rng.uniform(0.45, 0.65)
    
    # 家庭劳动力人数（与户主年龄相关）
    if age < 40:
        chars.total_labor_count = int(rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2]))
    elif age < 55:
        chars.total_labor_count = int(rng.choice([2, 3, 4], p=[0.4, 0.45, 0.15]))
    else:
        chars.total_labor_count = int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
    
    # 16-60岁劳动力占比
    if age >= 60:
        chars.prime_age_labor_ratio = rng.uniform(0.3, 0.5)
    else:
        chars.prime_age_labor_ratio = rng.uniform(0.6, 0.9)
    
    # 根据年龄推断其他特征
    # 老龄化率
    if age >= 60:
        chars.aging_rate = 0.6 + rng.random() * 0.3
    elif age >= 50:
        chars.aging_rate = 0.3 + rng.random() * 0.3
    else:
        chars.aging_rate = rng.random() * 0.2
    
    # 子女是否进城（年龄越大越可能）
    chars.children_in_city = rng.random() < (0.3 + age * 0.01)
    
    # 子女是否愿意接班（年轻代际更不愿意）
    chars.successor_willing = rng.random() < (0.4 - generation * 0.1)
    
    # 技术采纳（年轻人更高）
    if generation >= 3:
        chars.technology_adoption = 0.5 + rng.random() * 0.4
    else:
        chars.technology_adoption = 0.2 + rng.random() * 0.4
    
    # 风险承受能力
    chars.risk_tolerance = max(0, 0.5 - age * 0.005 + rng.random() * 0.3)
    
    # 信息获取能力
    chars.information_access = 0.3 + rng.random() * 0.4
    if generation >= 3:
        chars.information_access += 0.2
    
    # 社会资本（与年龄正相关）
    chars.social_capital = 3 + age * 0.1 + rng.random() * 5
    
    # 商品化率（与规模正相关）
    if land_area > 30:
        chars.commercialization_rate = 0.7 + rng.random() * 0.25
    elif land_area > 10:
        chars.commercialization_rate = 0.4 + rng.random() * 0.3
    else:
        chars.commercialization_rate = 0.1 + rng.random() * 0.3
    
    # 初始收入结构（简化估算）
    base_farm_income = land_area * 1000 * (0.8 + rng.random() * 0.4)
    chars.farm_income = base_farm_income
    
    # 非农收入与年龄和代际相关
    if age < 55:
        chars.wage_income = initial_capital * 0.3 * (1 - age / 100)
    else:
        chars.wage_income = initial_capital * 0.1
    
    chars.property_income = land_area * 50 * rng.random()  # 土地租金等
    chars.transfer_income = 2000 + rng.random() * 3000  # 补贴等
    chars.non_farm_business_income = initial_capital * 0.05 * rng.random()
    
    # 更新依赖度和多样性
    chars.update_from_income()
    
    # 根据类型设置政策敏感度
    farmer_type, _ = FarmerTypeClassifier.classify(chars)
    sensitivity = FarmerTypeClassifier.get_type_policy_sensitivity(farmer_type)
    chars.price_subsidy_sensitivity = sensitivity['price_subsidy']
    chars.direct_subsidy_sensitivity = sensitivity['direct_subsidy']
    chars.rental_subsidy_sensitivity = sensitivity['rental_subsidy']
    chars.service_policy_sensitivity = sensitivity['service']
    
    return chars
