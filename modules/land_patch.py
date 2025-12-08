# -*- coding: utf-8 -*-
"""
@theory
本文件定义了 LandPatch 类，代表具有异质性的地块。
地块具有面积、土壤质量、气候区等属性，支持多种土地利用选择。
新增地理坐标和空间邻域机制，支持空间ABM。

@critical 2025-12: 添加土地细碎化指数
参考：Tan et al. (2006) 中国耕地细碎化的定量测度
实证数据：中国户均6.4块地块，平均每块0.5亩

@critical 2025-01: 添加土地原用途和流转渠道追踪
理论依据（江浙实践经验）：
1. 原种粮地流转后必须种粮 → previous_crop_type='grain' 时有更严格约束
2. 村集体统一流转 vs 农户直接流转有不同的种粮要求
3. 政策目标是"防止非粮化"而非"强制粮化"

流转渠道分类：
- 'village_collective': 村集体统一组织流转（约30%），有严格种粮要求
- 'direct_farmer': 农户直接流转（约70%），约束相对较弱
- None: 未流转，自己经营
"""

import numpy as np

# 流转渠道枚举
class TransferChannel:
    """土地流转渠道类型"""
    VILLAGE_COLLECTIVE = 'village_collective'  # 村集体统一流转
    DIRECT_FARMER = 'direct_farmer'            # 农户直接流转
    NONE = None                                 # 未流转


# @new 2025-12: 土地类型枚举
class LandType:
    """
    土地类型分类（基于国土资源管理实践）
    
    理论依据：
    1. 《基本农田保护条例》：永久基本农田重点保护
    2. 《防止耕地"非粮化"意见》(2020)：基本农田必须种粮
    3. 现实比例：基本农田约占耕地总面积的80%
    """
    BASIC_FARMLAND = 'basic_farmland'      # 永久基本农田（80%种粮刚性约束）
    GENERAL_FARMLAND = 'general_farmland'  # 一般耕地（无强制约束，激励引导）


class LandPatch:
    def __init__(self, area, soil_quality, climate_zone, yield_profiles, coordinates=(0, 0), model=None):
        # 地块基本属性
        self.area = area
        self.soil_quality = soil_quality
        self.climate_zone = climate_zone
        self.yield_profiles = yield_profiles
        
        # 地理坐标
        self.coordinates = coordinates  # (x, y)坐标
        self.neighbors = []  # 空间邻域地块列表
        
        # @critical 2025-12: 土地细碎化指数
        # fragmentation_index ∈ [0.5, 1.0]
        # 1.0 = 完整连片地块（无效率损失）
        # 0.5 = 高度细碎地块（50%效率损失）
        # 影响：机械作业效率、田间管理成本、规模经济实现程度
        # 计算方法：基于地块面积和形状因子
        self.fragmentation_index = self._calculate_fragmentation_index(area)
        
        # 土地利用状态
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = None
        
        # @critical 2025-12: 土地类型（简化版种粮约束核心）
        # land_type: 土地分类，决定种粮约束强度
        # - 'basic_farmland': 永久基本农田（约80%），流转后必须保证≥80%种粮
        # - 'general_farmland': 一般耕地（约20%），无强制约束，靠激励引导
        # 初始化时随机分配，80%概率为基本农田
        self.land_type = LandType.BASIC_FARMLAND if np.random.random() < 0.80 else LandType.GENERAL_FARMLAND
        
        # @deprecated 2025-12: 以下属性保留但简化，主要用于数据记录
        self.previous_crop_type = None  # 'grain', 'cash_crop', 或 None
        self.crop_history = []  # 最近3年作物历史（用于统计）
        self.historical_grain_ratio = 0.5  # 默认50%（用于统计）
        
        # @deprecated 2025-12: 流转渠道不再影响种粮约束
        # 保留属性用于数据追踪，但不影响约束计算
        self.transfer_channel = None  # 'village_collective', 'direct_farmer', 或 None
        
        # 流转和弃耕状态
        self.leased_to = None  # 承租方ID
        self.rental_price = 0.0  # @fix: 初始化租金价格，避免hasattr检查失败
        self.abandoned = False
        
        # @critical 2025-12: 经营者补贴机制
        # current_operator: 当前实际经营者（可能是owner自己，也可能是租入者）
        # 补贴发放时根据subsidy_target判断发给owner还是current_operator
        self.current_operator = None  # 初始为None，创建后由owner设置
        
        # @critical 2025-12: 多年期租赁合同
        # 现实中土地租赁3-5年一签，而非每年重新竞价
        self.lease_contract_years = 0  # 剩余合同年限
        self.lease_start_year = None   # 合同起始年份
        self.contracted_rental_price = 0.0  # 合同约定租金（锁定价格）
        
        # 技术相关
        self.technology_level = 1.0  # 技术水平
        self.owner = None  # 地块所有者（承包权人）
        
        # 新增：模型引用
        self.model = model
    
    def _calculate_fragmentation_index(self, area):
        """
        @critical 2025-12: 计算土地细碎化指数
        
        理论依据：
        - Tan et al. (2006): 中国耕地细碎化测度
        - 实证数据：地块越小，单位面积管理成本越高
        
        计算逻辑：
        - 大地块（≥20亩）：接近1.0，规模经济充分实现
        - 中地块（5-20亩）：0.85-0.95，部分规模经济
        - 小地块（1-5亩）：0.70-0.85，规模经济受限
        - 极小地块（<1亩）：0.50-0.70，严重细碎化
        
        Args:
            area: 地块面积（亩）
        
        Returns:
            float: 细碎化指数 [0.5, 1.0]
        """
        if area >= 20:
            # 大规模地块，规模经济充分
            base_index = 0.95 + 0.05 * min(1.0, (area - 20) / 80)
        elif area >= 5:
            # 中等地块
            base_index = 0.85 + 0.10 * (area - 5) / 15
        elif area >= 1:
            # 小地块
            base_index = 0.70 + 0.15 * (area - 1) / 4
        else:
            # 极小地块（<1亩）
            base_index = 0.50 + 0.20 * area
        
        # 添加随机形状因子扰动（±5%）
        # 反映地块形状不规则性（长条形比方形更难耕作）
        shape_factor = np.random.uniform(0.95, 1.05)
        
        return min(1.0, max(0.5, base_index * shape_factor))

    def generate_land_use_options(self):
        """
        @theory
        生成土地利用选项：种植、流转、弃耕
        考虑地块属性和当前状态
        """
        options = ["种小麦", "种玉米"]
        
        # 如果未被承租且未弃耕，可以流转
        if self.leased_to is None and not self.abandoned:
            options.append("leased")
        
        # 如果未被承租，可以弃耕
        if self.leased_to is None:
            options.append("abandoned")
        
        return options

    def get_prospects(self, crop):
        """
        @theory
        获取作物种植的前景分布
        考虑土壤质量、气候区、技术水平等因素
        
        @critical 2025-12: 加入细碎化指数影响
        细碎化降低规模经济效应，体现为单位面积产量下降
        """
        if crop not in self.yield_profiles:
            return [(0, 1.0)]
        
        base_prospects = self.yield_profiles[crop]
        
        # 考虑土壤质量的影响
        soil_factor = self.soil_quality
        
        # 考虑技术水平的影响
        tech_factor = self.technology_level
        
        # @critical 2025-12: 考虑细碎化的影响
        # 细碎化降低机械作业效率和规模经济
        fragmentation_factor = self.fragmentation_index
        
        # 调整后的前景
        adjusted_prospects = []
        for income, prob in base_prospects:
            adjusted_income = income * soil_factor * tech_factor * fragmentation_factor
            adjusted_prospects.append((adjusted_income, prob))
        
        return adjusted_prospects

    def set_land_use(self, crop):
        """
        @theory
        设置土地利用类型
        更新地块状态和产量
        """
        if crop == "leased":
            self.set_leased(None)
        elif crop == "abandoned":
            self.set_abandoned()
        elif crop in self.yield_profiles:
            self.current_crop = crop
            self.last_land_use = crop
            self.abandoned = False
            self.leased_to = None
            
            # 计算预期产量
            prospects = self.get_prospects(crop)
            self.last_yield = sum([income * prob for income, prob in prospects])
        else:
            raise ValueError(f"不支持的作物类型: {crop}")

    def set_abandoned(self):
        """
        @theory
        设置地块为弃耕状态
        """
        self.abandoned = True
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = "abandoned"
        self.leased_to = None

    def set_leased(self, to_agent, contract_years=3, rental_price=0.0, 
                   transfer_channel=None, via_village=False):
        """
        @theory
        设置地块为流转状态
        
        @critical 2025-12: 完善流转机制
        1. 更新current_operator为实际经营者
        2. 设置多年期合同（默认3年，现实中3-5年一签）
        3. 锁定合同期内租金价格
        
        @critical 2025-01: 记录土地原用途和流转渠道
        4. 保存流转前的作物类型（previous_crop_type）
        5. 记录流转渠道（村集体 vs 农户直接）
        
        Args:
            to_agent: 承租方Agent
            contract_years: 合同年限（默认3年）
            rental_price: 合同约定租金
            transfer_channel: 流转渠道 ('village_collective' 或 'direct_farmer')
            via_village: 是否通过村集体流转（旧参数，向后兼容）
        """
        # @critical 2025-01: 记录流转前的作物类型
        # 在流转前保存当前作物状态，用于后续种粮约束判定
        if self.current_crop is not None:
            # 将作物类型转换为字符串标识
            crop_str = str(self.current_crop).lower()
            if 'grain' in crop_str or '粮' in crop_str or '小麦' in crop_str or '玉米' in crop_str or '水稻' in crop_str:
                self.previous_crop_type = 'grain'
            elif 'cash' in crop_str or '经济' in crop_str or '蔬菜' in crop_str or '水果' in crop_str:
                self.previous_crop_type = 'cash_crop'
            else:
                # 默认按粮食处理（保守原则）
                self.previous_crop_type = 'grain'
        elif self.last_land_use is not None:
            # 如果current_crop为空，尝试从last_land_use推断
            use_str = str(self.last_land_use).lower()
            if 'grain' in use_str or '粮' in use_str:
                self.previous_crop_type = 'grain'
            elif 'cash' in use_str or '经济' in use_str:
                self.previous_crop_type = 'cash_crop'
            else:
                self.previous_crop_type = 'grain'  # 默认
        else:
            # 无法推断，默认为粮食（保守原则：防止非粮化）
            self.previous_crop_type = 'grain'
        
        # @critical 2025-01: 记录流转渠道
        if transfer_channel is not None:
            self.transfer_channel = transfer_channel
        elif via_village:
            self.transfer_channel = 'village_collective'
        else:
            self.transfer_channel = 'direct_farmer'
        
        self.leased_to = to_agent
        self.current_operator = to_agent  # 经营权转移给承租方
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = "leased"
        self.abandoned = False
        
        # 多年期合同
        self.lease_contract_years = contract_years
        self.lease_start_year = self.model.current_step if self.model else 0
        self.contracted_rental_price = rental_price
        self.rental_price = rental_price
    
    def end_lease(self):
        """
        @critical 2025-12: 合同到期，经营权回归owner
        """
        self.leased_to = None
        self.current_operator = self.owner  # 经营权回归承包权人
        self.lease_contract_years = 0
        self.lease_start_year = None
        self.contracted_rental_price = 0.0
        self.rental_price = 0.0
    
    def update_lease_contract(self):
        """
        @critical 2025-12: 每年更新合同状态
        合同年限-1，到期则自动解除
        
        Returns:
            bool: True表示合同仍有效，False表示合同到期
        """
        if self.leased_to is not None and self.lease_contract_years > 0:
            self.lease_contract_years -= 1
            if self.lease_contract_years <= 0:
                # 合同到期，但不自动解除，等待续约或解约决策
                return False
        return True
    
    def is_lease_expiring(self):
        """合同是否即将到期（剩余1年内）"""
        return self.leased_to is not None and self.lease_contract_years <= 1

    def get_spatial_neighbors(self):
        """
        @theory
        获取空间邻域地块
        用于技术扩散和信息传播
        """
        return self.neighbors

    def calculate_distance_to(self, other_patch):
        """
        @theory
        计算到其他地块的距离
        用于空间价格梯度和邻域效应
        """
        if not hasattr(other_patch, 'coordinates'):
            return float('inf')
        
        dx = self.coordinates[0] - other_patch.coordinates[0]
        dy = self.coordinates[1] - other_patch.coordinates[1]
        return np.sqrt(dx**2 + dy**2)

    def is_neighbor_of(self, other_patch, radius=2):
        """
        @theory
        判断是否为邻域地块
        基于距离阈值定义邻域关系
        """
        distance = self.calculate_distance_to(other_patch)
        return distance <= radius

    def update_technology_level(self, new_level):
        """
        @theory
        更新技术水平
        影响产量和决策
        """
        self.technology_level = max(1.0, min(2.0, new_level))

    def get_spatial_price(self, base_price):
        """
        @theory
        空间价格梯度：基于Fujita et al.(1999)空间经济学理论
        
        空间价格函数：
        P(x,y) = P_market - α × d(x,y)
        
        其中：
        - P_market: 市场中心价格
        - α: 运输成本率
        - d(x,y): 到市场中心的距离
        
        运输成本函数：
        TransportCost = distance × costRate × area
        
        基于中国农村交通成本数据，运输成本率在0.03-0.08之间
        """
        market_center = self.model.market_center
        distance = np.sqrt((self.coordinates[0] - market_center[0])**2 + 
                          (self.coordinates[1] - market_center[1])**2)
        
        # 空间价格梯度
        transport_cost_rate = self.model.transport_cost_rate
        spatial_price = max(0, base_price - distance * transport_cost_rate)
        
        return spatial_price

    # ==================== 种粮约束相关方法 ====================
    
    def update_crop_history(self, crop_type):
        """
        @critical 2025-01: 更新作物种植历史
        每年种植结束后调用，记录该年种植的作物类型
        
        Args:
            crop_type: 本年种植的作物类型 ('grain' 或 'cash_crop')
        """
        # 标准化作物类型
        crop_str = str(crop_type).lower() if crop_type else 'unknown'
        if 'grain' in crop_str or '粮' in crop_str or '小麦' in crop_str or '玉米' in crop_str or '水稻' in crop_str:
            normalized_crop = 'grain'
        elif 'cash' in crop_str or '经济' in crop_str or '蔬菜' in crop_str or '水果' in crop_str:
            normalized_crop = 'cash_crop'
        else:
            normalized_crop = 'grain'  # 默认按粮食处理（保守原则）
        
        # 添加到历史记录
        self.crop_history.append(normalized_crop)
        
        # 只保留最近3年
        if len(self.crop_history) > 3:
            self.crop_history = self.crop_history[-3:]
        
        # 更新历史种粮比例
        self._update_historical_grain_ratio()
    
    def _update_historical_grain_ratio(self):
        """
        @critical 2025-01: 计算过去3年的历史种粮比例
        """
        if not self.crop_history:
            self.historical_grain_ratio = 0.5  # 无历史数据，默认50%
            return
        
        grain_count = sum(1 for crop in self.crop_history if crop == 'grain')
        self.historical_grain_ratio = grain_count / len(self.crop_history)
    
    def get_grain_requirement_ratio(self, base_ratio=0.80):
        """
        @critical 2025-12: 获取本地块的种粮约束比例（简化版）
        
        简化规则（基于土地类型）：
        - 永久基本农田（约80%耕地）：流转后≥80%种粮（刚性约束）
        - 一般耕地（约20%耕地）：无强制约束，靠激励政策引导
        
        理论依据：
        1. 《基本农田保护条例》：永久基本农田重点保护
        2. 《防止耕地"非粮化"意见》(2020)：基本农田必须种粮
        3. 符合中国耕地管理实践：约80%耕地为基本农田
        4. 简化了流转渠道的差异化约束，更清晰
        
        Args:
            base_ratio: 基本农田的基准种粮约束（默认0.80）
        
        Returns:
            float: 该地块的种粮约束比例 [0, 0.80]
        """
        # 未流转的土地无外部约束（由经营者自主决定）
        if self.leased_to is None:
            return 0.0
        
        # @critical 2025-12: 基于土地类型的简化约束
        if self.land_type == LandType.BASIC_FARMLAND:
            # 永久基本农田：刚性约束80%种粮
            return base_ratio  # 0.80
        else:
            # 一般耕地：无强制约束
            return 0.0
    
    def requires_grain_planting(self):
        """
        @critical 2025-12: 判断该地块是否必须种粮（简化版）
        
        Returns:
            bool: True表示该地块为基本农田，必须种粮
        """
        return self.land_type == LandType.BASIC_FARMLAND and self.leased_to is not None
    
    def is_basic_farmland(self):
        """
        @critical 2025-12: 判断是否为永久基本农田
        
        Returns:
            bool: True表示是永久基本农田
        """
        return self.land_type == LandType.BASIC_FARMLAND
    
    def is_general_farmland(self):
        """
        @critical 2025-12: 判断是否为一般耕地
        
        Returns:
            bool: True表示是一般耕地
        """
        return self.land_type == LandType.GENERAL_FARMLAND

    # @deprecated 2025-12: 以下方法保留用于向后兼容，但不再影响约束计算
    def is_from_village_collective(self):
        """
        @deprecated 2025-12: 判断是否为村集体流转土地
        保留用于数据统计，不再影响种粮约束
        
        Returns:
            bool: True表示是村集体统一流转的土地
        """
        return self.transfer_channel == 'village_collective'
    
    def was_grain_land(self):
        """
        @deprecated 2025-12: 判断流转前是否为种粮土地
        保留用于数据统计，不再影响种粮约束
        
        Returns:
            bool: True表示流转前种植粮食
        """
        return self.previous_crop_type == 'grain'