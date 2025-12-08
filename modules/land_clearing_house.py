# -*- coding: utf-8 -*-
"""
@theory
LandClearingHouse - 土地租赁中央清算所
实现基于多层次博弈设计规范的市场出清机制
新增：村精英优先权、改进价格形成、动态博弈反馈

@critical 2025-12: 熟人网络折扣机制
参考：Bai et al. (2023, AEJ:Applied) 土地流转网络模型
中国农村固定观察点数据：熟人/村内流转占比55%-70%

@enhanced 2025-12: 空间显式化与连片流转优先
专家审阅建议：使LandPatch距离影响网络形成、优先连片土地流转
参考：Deininger et al. (2012), "Land fragmentation and consolidation in China"

@enhancement 2025-11-30: 合约理论完善
- 新增分成合约类型（Sharecropping）
- 实现公开市场违约反悔机制
- 模拟金钱对传统关系的侵蚀
参考：Stiglitz (1974), "Incentives and Risk Sharing in Sharecropping"

@new 2025-11-30: 向量化匹配优化模式
- 大规模模拟时可启用向量化匹配提升性能
- 使用numpy矩阵运算替代O(N²)循环
"""

import numpy as np
from enum import Enum

class ContractType(Enum):
    """
    @new 2025-11-30: 合约类型枚举
    
    理论依据：
    - 固定租金合约（Fixed Rent）：风险全部由承租方承担
    - 分成合约（Sharecropping）：风险在双方间分担
    
    中国农村现实：
    - 熟人间倾向于分成合约（"这就给你种，给点粮食就行"）
    - 陌生人/企业倾向于固定租金合约（清晰的权责边界）
    """
    FIXED_RENT = "fixed_rent"           # 固定租金：承租方承担全部风险
    SHARECROPPING = "sharecropping"     # 分成合约：按产出比例分成


class LandClearingHouse:
    """
    @theory
    土地租赁中央清算所：实现多层次博弈的市场出清机制
    基于供需匹配的价格形成和交易执行，支持村精英优先权
    
    @critical 2025-12: 熟人网络优先机制
    - 本村/亲缘关系流转：租金打0.75-0.90折，优先匹配
    - 基于social_capital决定是否接受低租金优先给熟人
    
    @enhancement 2025-11-30: 合约类型与违约机制
    - 熟人交易倾向于分成合约
    - 公开市场高价可能触发"违约"（金钱侵蚀关系）
    """
    
    def __init__(self):
        # 供给方报价（农户出租意愿）
        self.supply_offers = []  # [(patch, supplier, reservation_price), ...]
        
        # 需求方报价（企业租入意愿）
        self.demand_offers = []  # [(patch, demander, bid_price), ...]
        
        # 成交记录
        self.transactions = []  # [(patch, supplier, demander, price), ...]
        
        # 市场统计（新增）
        self.market_statistics = {
            'total_transactions': 0,
            'total_area_transferred': 0,
            'average_price': 0,
            'price_volatility': 0,
            'elite_advantage_count': 0,
            # @critical 2025-12: 熟人流转统计
            'acquaintance_transactions': 0,  # 熟人交易数量
            'acquaintance_area': 0,          # 熟人流转面积
            'acquaintance_discount_total': 0, # 熟人折扣总额
            # @new 2025-11-30: 合约类型统计
            'sharecropping_count': 0,         # 分成合约数量
            'fixed_rent_count': 0,            # 固定租金合约数量
            'breach_count': 0,                # 违约事件数量
            'breach_penalty_total': 0         # 违约罚金总额
        }
        
        # @critical 2025-12: 熟人网络参数（可从配置读取）
        self.acquaintance_discount_range = (0.70, 0.90)  # 熟人折扣范围：最低可至70%
        self.acquaintance_priority_weight = 1.5  # 熟人优先权重
        
        # @critical 2025-12 UPGRADE: 双轨制参数
        # 参考：Wang & Zhang (2025, working paper, PKU)
        self.acquaintance_min_discount = 0.70   # 熟人可接受的最低价格（市场价70%）
        self.open_market_max_premium = 1.20    # 公开市场最高溢价（市场价120%）
        self.acquaintance_exclusion = True     # 熟人网络排他性（外人高价也进不来）
        
        # @enhanced 2025-12: 空间显式化参数
        # 参考：Deininger et al. (2012), "Land fragmentation and consolidation in China"
        self.contiguity_bonus = 0.15           # 连片土地溢价比例（15%）
        self.max_spatial_distance = 5.0        # 最大考虑距离（单位：网格）
        self.spatial_priority_weight = 1.3     # 空间邻近优先权重
        
        # @new 2025-11-30: 向量化匹配优化模式
        self.use_vectorized_matching = False   # 是否启用向量化匹配
        self._vectorized_matcher = None        # 延迟初始化
        
    def enable_vectorized_matching(self, enabled: bool = True):
        """
        @new 2025-11-30: 启用/禁用向量化匹配模式
        
        向量化匹配使用numpy矩阵运算，在大规模模拟时性能更优
        """
        self.use_vectorized_matching = enabled
        if enabled and self._vectorized_matcher is None:
            try:
                from modules.vectorized_matching import VectorizedLandMatcher
                self._vectorized_matcher = VectorizedLandMatcher(max_distance=self.max_spatial_distance)
            except ImportError:
                print("[WARNING] 无法导入VectorizedLandMatcher，回退到标准匹配")
                self.use_vectorized_matching = False
        
    def add_supply_offer(self, patch, supplier, reservation_price):
        """
        @theory
        添加供给报价：农户提交出租意愿
        包含最低可接受租金（Min_Acceptable_Rent）
        """
        self.supply_offers.append({
            'patch': patch,
            'supplier': supplier,
            'reservation_price': reservation_price,
            'is_elite': hasattr(supplier, 'is_village_elite') and supplier.is_village_elite
        })
    
    def add_demand_offer(self, patch, demander, bid_price):
        """
        @theory
        添加需求报价：企业提交租入意愿
        包含最高愿付租金（Max_Willing_to_Pay_Rent）
        """
        self.demand_offers.append({
            'patch': patch,
            'demander': demander,
            'bid_price': bid_price,
            'is_elite': hasattr(demander, 'is_village_elite') and demander.is_village_elite
        })
    
    def clear_market(self):
        """
        @theory 双轨制市场出清机制
        @critical 2025-12 UPGRADE: 实现中国土地流转的"半市场化"本质
        
        参考：Wang & Zhang (2025, working paper, PKU)
        
        核心逻辑：
        第一轨：熟人网络内部消化（先到先得 + 价格可以低至70%）
                → 外人即使出高价20%也抢不到！这才是关键！
        第二轨：剩余地块公开竞价（价格可以高至120%）
        
        这就是中国农村"半市场化"土地流转的真实写照：
        - 熟人关系网决定"能不能租到"
        - 价格只决定"租多少钱"
        """
        self.transactions = []
        
        # @fix 2025-01: 在清空前记录供需数量，用于后续统计
        self.last_supply_count = len(self.supply_offers)
        self.last_demand_count = len(self.demand_offers)
        
        # 按地块分组收集报价
        patch_supply = {}  # {patch: [supply_offers]}
        patch_demand = {}  # {patch: [demand_offers]}
        
        for offer in self.supply_offers:
            patch = offer['patch']
            if patch not in patch_supply:
                patch_supply[patch] = []
            patch_supply[patch].append(offer)
        
        for offer in self.demand_offers:
            patch = offer['patch']
            if patch not in patch_demand:
                patch_demand[patch] = []
            patch_demand[patch].append(offer)
        
        # 第一轨：熟人网络内部消化
        matched_patches = self._acquaintance_track_matching(patch_supply, patch_demand)
        
        # 第二轨：公开市场竞价（剩余地块）
        self._open_market_track_matching(patch_supply, patch_demand, matched_patches)
        
        # 更新市场统计
        self._update_market_statistics()
        
        # 清空报价列表
        self.supply_offers = []
        self.demand_offers = []
    
    def _acquaintance_track_matching(self, patch_supply, patch_demand):
        """
        @critical 2025-12: 第一轨 - 熟人网络内部消化
        
        核心特征：
        1. 先到先得：熟人网络内的需求优先匹配
        2. 价格可低：熟人可以70%市场价拿到地
        3. 排他性：一旦有熟人想租，外人即使高价也进不来！
        
        这才是中国农村土地流转的核心逻辑：
        村干部/亲戚会干预，把"自己人"安排上
        """
        matched_patches = set()
        
        for patch, supply_offers in patch_supply.items():
            if patch in matched_patches:
                continue
            if patch not in patch_demand:
                continue
            
            demand_offers = patch_demand[patch]
            if not demand_offers:
                continue
            
            # 取第一个供给方（通常一块地只有一个owner）
            supply_offer = supply_offers[0]
            supplier = supply_offer['supplier']
            reservation_price = supply_offer['reservation_price']
            
            # 获取供给方的熟人网络
            social_network = getattr(supplier, 'social_network', [])
            transaction_neighbors = getattr(supplier, 'transaction_neighbors', [])
            
            # 找出所有熟人需求方
            acquaintance_demands = []
            outsider_demands = []
            
            for demand_offer in demand_offers:
                demander = demand_offer['demander']
                is_acq = self._check_acquaintance(supplier, demander, social_network)
                
                if is_acq:
                    acquaintance_demands.append(demand_offer)
                else:
                    outsider_demands.append(demand_offer)
            
            # 关键逻辑：如果有熟人想租，直接锁定给熟人！
            # 外人即使出更高价也没用（村干部/亲戚会干预）
            if acquaintance_demands and self.acquaintance_exclusion:
                # 在熟人中选出价最高的
                best_acq = max(acquaintance_demands, key=lambda x: x['bid_price'])
                bid = best_acq['bid_price']
                
                # 熟人折扣：供给方愿意接受低至70%的价格
                # 折扣率基于社会资本
                sc = self._get_social_capital(supplier)
                discount_rate = self.acquaintance_min_discount + (1 - self.acquaintance_min_discount) * (1 - sc / 10)
                min_acceptable = reservation_price * discount_rate
                
                if bid >= min_acceptable:
                    # 成交！
                    # 熟人交易价格：偏向需求方（供给方让利）
                    transaction_price = min_acceptable * 0.5 + bid * 0.5
                    transaction_price = max(transaction_price, min_acceptable)
                    
                    self._record_transaction(
                        patch, supply_offer, best_acq, 
                        transaction_price, is_acquaintance=True
                    )
                    matched_patches.add(patch)
                    
                    # 更新统计
                    self.market_statistics['acquaintance_transactions'] += 1
                    self.market_statistics['acquaintance_area'] += patch.area
                    self.market_statistics['acquaintance_discount_total'] += (reservation_price - transaction_price)
                    
                    continue
            
            # 如果熟人出价太低或没有熟人，这块地进入公开市场
            # （但不在这里处理，留给第二轨）
        
        return matched_patches
    
    def _open_market_track_matching(self, patch_supply, patch_demand, matched_patches):
        """
        @critical 2025-12: 第二轨 - 公开市场竞价
        @enhanced 2025-12: 加入空间优先级 - 连片流转优先
        
        核心特征：
        1. 竞价机制：价高者得，但连片经营者有优先权
        2. 价格可高：最高可出到市场价120%
        3. 连片溢价：愿意为相邻地块支付10-15%溢价
        4. 进入门槛：只有熟人网络"剩下"的地块才进入公开市场
        
        这解释了为什么：
        - 农业公司租地价格普遍高于农户间流转
        - 大户/企业愿意为连片土地支付溢价
        """
        for patch, supply_offers in patch_supply.items():
            if patch in matched_patches:
                continue
            if patch not in patch_demand:
                continue
            
            demand_offers = patch_demand[patch]
            if not demand_offers:
                continue
            
            supply_offer = supply_offers[0]
            reservation_price = supply_offer['reservation_price']
            
            # @enhanced 2025-12: 使用空间优先级排序需求
            # 连片经营者获得优先匹配权
            spatial_ranked_demands = self._get_spatial_priority_demands(patch, demand_offers)
            
            for ranked_demand in spatial_ranked_demands:
                demand_offer = ranked_demand['demand_offer']
                proximity_score = ranked_demand['proximity_score']
                bid = demand_offer['bid_price']
                
                # 计算连片溢价（需求方愿意为连片支付额外价格）
                contiguity_bonus = self._calculate_contiguity_bonus(
                    patch, demand_offer['demander'], reservation_price
                )
                effective_bid = bid + contiguity_bonus
                
                # 公开市场允许最高20%溢价
                max_allowed_bid = reservation_price * self.open_market_max_premium
                effective_bid = min(effective_bid, max_allowed_bid)
                
                # 出价需达到保留价的85%
                # 但连片流转可以放宽到80%（供给方也愿意连片出让）
                min_threshold = 0.80 if proximity_score > 0.5 else 0.85
                
                if effective_bid >= reservation_price * min_threshold:
                    # 确定成交价
                    # 公开市场议价：双方力量相对均衡
                    if supply_offer['is_elite']:
                        farmer_power = 0.65  # 村精英供给方议价能力强
                    elif demand_offer['is_elite']:
                        farmer_power = 0.35  # 村精英需求方议价能力强
                    else:
                        farmer_power = 0.50  # 均衡
                    
                    # 连片流转时，供给方更愿意让步（统一管理方便）
                    if proximity_score > 0.5:
                        farmer_power *= 0.9
                    
                    transaction_price = reservation_price * farmer_power + effective_bid * (1 - farmer_power)
                    
                    self._record_transaction(
                        patch, supply_offer, demand_offer,
                        transaction_price, is_acquaintance=False,
                        is_contiguous=(proximity_score > 0.5)
                    )
                    matched_patches.add(patch)
                    
                    # 更新连片流转统计
                    if proximity_score > 0.5:
                        if 'contiguous_transactions' not in self.market_statistics:
                            self.market_statistics['contiguous_transactions'] = 0
                            self.market_statistics['contiguous_area'] = 0
                        self.market_statistics['contiguous_transactions'] += 1
                        self.market_statistics['contiguous_area'] += patch.area
                    
                    break  # 该地块已匹配
    
    def _get_social_capital(self, agent):
        """获取智能体的社会资本值（0-10）"""
        # 尝试从连续变量特征系统获取
        if hasattr(agent, 'characteristics') and agent.characteristics:
            return getattr(agent.characteristics, 'social_capital', 5.0)
        # 尝试从社会网络强度估算
        if hasattr(agent, 'social_network_strength'):
            return agent.social_network_strength * 10
        return 5.0  # 默认值
    
    # ========== 新增：空间显式化方法 ==========
    
    def _calculate_operation_radius(self, demander) -> float:
        """
        @new 2025-12-05: 计算主体的经营半径（借鉴spatial-climate-ABM的work_radius）
        
        理论依据：
        1. 农户经营半径受到交通成本、时间成本约束
        2. 大规模经营主体有更大的经营半径（机械化、雇工）
        3. 参考spatial-climate-ABM中work_radius的实现
        
        Returns:
            float: 经营半径（网格单位）
        """
        # 基础半径
        base_radius = 3.0
        
        # 根据经营规模调整
        leased_area = 0
        if hasattr(demander, 'total_leased_area'):
            leased_area = demander.total_leased_area
        elif hasattr(demander, 'get_operated_land_area'):
            leased_area = demander.get_operated_land_area()
        
        # 规模经济：面积越大，经营半径越大
        # 但存在边际递减
        if leased_area > 100:
            scale_bonus = np.log10(leased_area / 10) * 2.0
        elif leased_area > 30:
            scale_bonus = 1.5
        elif leased_area > 10:
            scale_bonus = 0.5
        else:
            scale_bonus = 0.0
        
        # 主体类型调整（企业比农户半径更大）
        if hasattr(demander, 'enterprise_type'):
            type_bonus = 3.0  # 企业
        elif hasattr(demander, 'is_large_farmer') and demander.is_large_farmer:
            type_bonus = 1.5  # 大户
        else:
            type_bonus = 0.0  # 普通农户
        
        # 社会资本调整（高社会资本可以跨越更大范围）
        social_capital = 5.0
        if hasattr(demander, 'characteristics') and demander.characteristics:
            social_capital = getattr(demander.characteristics, 'social_capital', 5.0)
        elif hasattr(demander, 'social_capital'):
            social_capital = demander.social_capital
        
        social_bonus = (social_capital - 5.0) / 10 * 1.0  # -0.5到0.5的调整
        
        operation_radius = base_radius + scale_bonus + type_bonus + social_bonus
        return max(1.0, min(operation_radius, 15.0))  # 限制在1-15之间
    
    def _calculate_distance_cost(self, demander) -> float:
        """
        @new 2025-12-05: 计算主体的距离成本系数（借鉴spatial-climate-ABM的distance_cost）
        
        理论依据：
        1. 不同主体对距离的敏感性不同
        2. 机械化程度高的主体距离成本较低
        3. 老年农户距离成本较高
        
        Returns:
            float: 距离成本系数 (0.1-0.5)
        """
        # 基础距离成本
        base_cost = 0.3
        
        # 年龄调整（老年人距离成本更高）
        age = 45
        if hasattr(demander, 'age'):
            age = demander.age
        elif hasattr(demander, 'characteristics') and demander.characteristics:
            age = getattr(demander.characteristics, 'aging_rate', 0.5) * 80  # 估算
        
        age_adjustment = 0.1 * (age - 45) / 30  # 75岁比45岁高0.1
        
        # 主体类型调整（企业距离成本低）
        if hasattr(demander, 'enterprise_type'):
            type_adjustment = -0.15  # 企业机械化程度高
        elif hasattr(demander, 'mechanization_level'):
            type_adjustment = -0.1 * demander.mechanization_level
        else:
            type_adjustment = 0.0
        
        # 经济能力调整（有钱可以克服距离）
        income = 50000
        if hasattr(demander, 'total_income'):
            income = demander.total_income
        elif hasattr(demander, 'cash'):
            income = demander.cash
        
        income_adjustment = -0.05 * min(1.0, np.log10(income / 50000 + 1))
        
        distance_cost = base_cost + age_adjustment + type_adjustment + income_adjustment
        return np.clip(distance_cost, 0.1, 0.5)
    
    def _calculate_spatial_proximity_score(self, patch, demander):
        """
        @enhanced 2025-12: 计算空间邻近性评分
        @updated 2025-12-05: 引入operation_radius和distance_cost概念
        
        评估需求方（demander）与目标地块（patch）的空间邻近程度
        邻近的地块能够形成连片经营，提高规模经济效率
        
        理论依据：
        - Deininger et al. (2012): 土地细碎化降低生产效率
        - 连片经营可提高机械化效率10-20%
        - spatial-climate-ABM: utility = wage_offer - distance_cost * dist
        
        改进要点（对比之前版本）：
        1. 引入个体化的经营半径(operation_radius)替代固定的max_spatial_distance
        2. 引入距离成本系数(distance_cost)实现主体异质性
        3. 使用效用函数而非简单线性衰减
        
        Returns:
            float: 空间邻近性得分 [0, 1]，1表示直接相邻
        """
        if not hasattr(patch, 'coordinates'):
            return 0.0
        
        # 获取需求方已有的土地
        demander_patches = []
        if hasattr(demander, 'leased_patches'):
            demander_patches.extend(demander.leased_patches)
        if hasattr(demander, 'land_patches'):
            demander_patches.extend(demander.land_patches)
        if hasattr(demander, 'owned_patches'):
            demander_patches.extend(demander.owned_patches)
        
        if not demander_patches:
            # 没有已有土地时，计算到demander位置的距离
            if hasattr(demander, 'pos') and demander.pos:
                dx = patch.coordinates[0] - demander.pos[0]
                dy = patch.coordinates[1] - demander.pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # 使用operation_radius进行评估
                op_radius = self._calculate_operation_radius(demander)
                if distance <= op_radius:
                    # 在经营半径内，使用距离成本计算效用
                    distance_cost = self._calculate_distance_cost(demander)
                    # 效用函数：1 - distance_cost * (distance / op_radius)
                    utility = 1.0 - distance_cost * (distance / op_radius)
                    return max(0.0, min(1.0, utility))
                else:
                    return 0.0
            else:
                return 0.0  # 无法计算邻近性
        
        # 计算到最近已有地块的距离
        min_distance = float('inf')
        patch_coords = patch.coordinates
        
        # 同时计算"连片程度" - 周围有多少已有地块
        adjacent_count = 0
        near_count = 0
        
        for existing_patch in demander_patches:
            if not hasattr(existing_patch, 'coordinates'):
                continue
            
            dx = patch_coords[0] - existing_patch.coordinates[0]
            dy = patch_coords[1] - existing_patch.coordinates[1]
            distance = np.sqrt(dx**2 + dy**2)
            min_distance = min(min_distance, distance)
            
            # 统计相邻和近邻数量
            if distance <= 1.5:
                adjacent_count += 1
            elif distance <= 3.0:
                near_count += 1
        
        if min_distance == float('inf'):
            return 0.0
        
        # 获取个体化参数
        op_radius = self._calculate_operation_radius(demander)
        distance_cost = self._calculate_distance_cost(demander)
        
        # 计算基础邻近得分
        if min_distance <= 1.5:
            # 直接相邻：最高得分
            base_score = 1.0
        elif min_distance <= op_radius:
            # 在经营半径内：效用函数
            base_score = 1.0 - distance_cost * (min_distance / op_radius)
        else:
            # 超出经营半径
            base_score = 0.0
        
        # 连片程度加成（多个相邻地块比单个相邻价值更高）
        # 这反映了连片经营的规模效应
        contiguity_bonus = 0.0
        if adjacent_count >= 2:
            contiguity_bonus = 0.1 * min(adjacent_count - 1, 3)  # 最多0.3加成
        if near_count >= 2:
            contiguity_bonus += 0.05 * min(near_count - 1, 2)  # 最多0.1加成
        
        final_score = min(1.0, base_score + contiguity_bonus)
        return max(0.0, final_score)
    
    def _calculate_contiguity_bonus(self, patch, demander, base_price):
        """
        @enhanced 2025-12: 计算连片流转溢价
        
        如果地块与需求方已有土地连片，需求方愿意支付溢价
        这反映了规模经济的真实激励
        
        Returns:
            float: 连片溢价金额
        """
        proximity_score = self._calculate_spatial_proximity_score(patch, demander)
        
        if proximity_score > 0.5:
            # 高邻近性：愿意支付全额溢价
            return base_price * self.contiguity_bonus * proximity_score
        elif proximity_score > 0:
            # 中等邻近性：支付部分溢价
            return base_price * self.contiguity_bonus * proximity_score * 0.5
        else:
            return 0.0
    
    def _get_spatial_priority_demands(self, patch, demand_offers):
        """
        @enhanced 2025-12: 按空间优先级排序需求
        
        将需求方按与目标地块的空间邻近性排序
        优先匹配能形成连片经营的需求方
        
        Returns:
            list: 按空间优先级排序的需求列表
        """
        scored_demands = []
        
        for demand_offer in demand_offers:
            demander = demand_offer['demander']
            proximity_score = self._calculate_spatial_proximity_score(patch, demander)
            
            # 计算综合优先级得分
            # 出价权重 + 空间邻近权重
            bid = demand_offer['bid_price']
            
            # 归一化出价（假设保留价为基准）
            max_bid = max(d['bid_price'] for d in demand_offers) if demand_offers else 1
            normalized_bid = bid / max_bid if max_bid > 0 else 0
            
            # 综合得分：出价权重0.6 + 空间邻近权重0.4
            # 但如果是直接相邻，空间权重更高
            if proximity_score >= 0.8:
                priority_score = normalized_bid * 0.5 + proximity_score * 0.5
            else:
                priority_score = normalized_bid * 0.7 + proximity_score * 0.3
            
            scored_demands.append({
                'demand_offer': demand_offer,
                'proximity_score': proximity_score,
                'priority_score': priority_score
            })
        
        # 按优先级得分排序
        scored_demands.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return scored_demands
    
    def _clear_patch_market(self, patch, offers):
        """
        @theory
        单个地块的市场出清
        支持村精英优先权，从最高出价和最低保留价开始匹配
        
        @critical 2025-12: 熟人网络优先机制
        参考：Bai et al. (2023, AEJ:Applied) 土地流转网络模型
        1. 先尝试熟人匹配（同村/亲缘），允许较低租金
        2. 熟人匹配失败后才进入市场化匹配
        """
        supply_offers = offers['supply']
        demand_offers = offers['demand']
        
        # 第一阶段：熟人优先匹配
        # @critical: 中国农村固定观察点数据显示，熟人/村内流转占比55%-70%
        matched_supply = []
        matched_demand = []
        
        for supply_offer in supply_offers:
            if supply_offer in matched_supply:
                continue
                
            supplier = supply_offer['supplier']
            reservation = supply_offer['reservation_price']
            
            # 获取供给方的社会网络
            social_network = getattr(supplier, 'social_network', [])
            
            # 遍历需求方，优先匹配熟人
            best_acquaintance_match = None
            best_acquaintance_bid = 0
            
            for demand_offer in demand_offers:
                if demand_offer in matched_demand:
                    continue
                    
                demander = demand_offer['demander']
                bid = demand_offer['bid_price']
                
                # 判断是否熟人关系
                is_acquaintance = self._check_acquaintance(supplier, demander, social_network)
                
                if is_acquaintance:
                    # 熟人折扣：供给方愿意接受较低租金
                    # 折扣率基于社会资本强度
                    social_capital = getattr(supplier, 'characteristics', None)
                    if social_capital:
                        sc = getattr(social_capital, 'social_capital', 5.0)
                    else:
                        sc = getattr(supplier, 'social_network_strength', 0.5) * 10
                    
                    # 社会资本越高，越愿意接受折扣（关系优先于金钱）
                    # 折扣率：0.75-0.90，社会资本高的接受更多折扣
                    discount_rate = 0.90 - 0.15 * min(1.0, sc / 10)  # sc=10时折扣0.75
                    discounted_reservation = reservation * discount_rate
                    
                    # 熟人出价只需达到折扣后的保留价即可成交
                    if bid >= discounted_reservation:
                        if bid > best_acquaintance_bid:
                            best_acquaintance_match = demand_offer
                            best_acquaintance_bid = bid
            
            # 如果找到熟人匹配，优先完成交易
            if best_acquaintance_match:
                demander = best_acquaintance_match['demander']
                bid = best_acquaintance_match['bid_price']
                
                # 熟人交易价格：取折扣后保留价和出价的中间值
                social_capital = getattr(supplier, 'characteristics', None)
                if social_capital:
                    sc = getattr(social_capital, 'social_capital', 5.0)
                else:
                    sc = getattr(supplier, 'social_network_strength', 0.5) * 10
                discount_rate = 0.90 - 0.15 * min(1.0, sc / 10)
                discounted_reservation = reservation * discount_rate
                
                # 成交价：加权平均，熟人交易中供给方权重较低（愿意让利）
                transaction_price = discounted_reservation * 0.6 + bid * 0.4
                
                # 记录熟人交易
                self._record_transaction(
                    patch, supply_offer, best_acquaintance_match,
                    transaction_price, is_acquaintance=True
                )
                
                matched_supply.append(supply_offer)
                matched_demand.append(best_acquaintance_match)
                
                # 更新熟人交易统计
                self.market_statistics['acquaintance_transactions'] += 1
                self.market_statistics['acquaintance_area'] += patch.area
                self.market_statistics['acquaintance_discount_total'] += (reservation - transaction_price)
                
                continue  # 该地块已匹配，处理下一个
        
        # 第二阶段：市场化匹配（剩余未匹配的地块）
        remaining_supply = [s for s in supply_offers if s not in matched_supply]
        remaining_demand = [d for d in demand_offers if d not in matched_demand]
        
        # 按保留价格排序（从低到高），村精英优先
        remaining_supply.sort(key=lambda x: (x['reservation_price'], not x['is_elite']))
        
        # 按出价排序（从高到低），村精英优先
        remaining_demand.sort(key=lambda x: (x['bid_price'], not x['is_elite']), reverse=True)
        
        # 匹配交易
        for demand_offer in remaining_demand:
            for supply_offer in remaining_supply[:]:
                if demand_offer['patch'] == supply_offer['patch']:
                    bid = demand_offer['bid_price']
                    reservation = supply_offer['reservation_price']
                    
                    # 计算价格偏离度
                    if reservation > 0:
                        price_gap = (bid - reservation) / reservation
                    else:
                        price_gap = 0
                    
                    # 市场化交易：出价不低于保留价的85%就可以交易
                    if price_gap >= -0.15:
                        # 根据双方议价能力确定成交价
                        if supply_offer['is_elite']:
                            farmer_power = 0.70
                        elif demand_offer['is_elite']:
                            farmer_power = 0.40
                        else:
                            farmer_power = 0.50
                        
                        transaction_price = bid * (1 - farmer_power) + reservation * farmer_power
                        
                        # 记录市场化交易
                        self._record_transaction(
                            patch, supply_offer, demand_offer,
                            transaction_price, is_acquaintance=False
                        )
                        
                        remaining_supply.remove(supply_offer)
                        break
    
    def _check_acquaintance(self, supplier, demander, social_network):
        """
        @critical 2025-12: 判断是否熟人关系
        
        熟人定义：
        1. 在供给方的社会网络中
        2. 同村（如果有村庄属性）
        3. 是农户而非企业（农户间更容易熟人交易）
        """
        # 检查社会网络
        if demander in social_network:
            return True
        
        # 检查是否同村
        supplier_village = getattr(supplier, 'village_id', None)
        demander_village = getattr(demander, 'village_id', None)
        if supplier_village and demander_village and supplier_village == demander_village:
            return True
        
        # 检查是否都是农户（农户间熟人概率更高）
        supplier_is_farmer = hasattr(supplier, 'livelihood_strategy')
        demander_is_farmer = hasattr(demander, 'livelihood_strategy')
        
        if supplier_is_farmer and demander_is_farmer:
            # 同为农户，检查社会网络重叠
            demander_network = getattr(demander, 'social_network', [])
            # 如果有共同好友，视为熟人
            common_friends = set(social_network) & set(demander_network)
            if len(common_friends) >= 2:  # 至少2个共同好友
                return True
        
        return False
    
    def _record_transaction(self, patch, supply_offer, demand_offer, price, is_acquaintance=False, is_contiguous=False):
        """
        @critical 2025-12: 记录交易并更新各方状态
        统一处理熟人交易和市场化交易
        
        @enhanced 2025-12: 支持连片流转标记
        @critical 2025-01: 标记流转渠道为农户直接流转
        """
        # 记录交易
        transaction = {
            'patch': patch,
            'supplier': supply_offer['supplier'],
            'demander': demand_offer['demander'],
            'price': price,
            'supplier_reservation': supply_offer['reservation_price'],
            'demander_bid': demand_offer['bid_price'],
            'supplier_is_elite': supply_offer['is_elite'],
            'demander_is_elite': demand_offer['is_elite'],
            'is_acquaintance': is_acquaintance,  # 熟人交易标记
            'is_contiguous': is_contiguous,      # 连片流转标记
            'transfer_channel': 'direct_farmer'  # 流转渠道：农户直接流转
        }
        self.transactions.append(transaction)
        
        # 更新地块状态（使用新的多年期合同机制）
        # @critical 2025-01: 标记为农户直接流转渠道
        contract_years = 3 if not is_acquaintance else 5  # 熟人交易合同期更长（信任）
        patch.set_leased(
            demand_offer['demander'], 
            contract_years=contract_years, 
            rental_price=price,
            transfer_channel='direct_farmer'  # 关键：标记为农户直接流转
        )
        
        # 更新农户和企业状态
        supplier = supply_offer['supplier']
        demander = demand_offer['demander']
        
        if hasattr(supplier, 'leased_out_patches'):
            if patch not in supplier.leased_out_patches:
                supplier.leased_out_patches.append(patch)
        
        if hasattr(demander, 'leased_patches'):
            if patch not in demander.leased_patches:
                demander.leased_patches.append(patch)
        
        # 记录精英优势
        if supply_offer['is_elite'] or demand_offer['is_elite']:
            self.market_statistics['elite_advantage_count'] += 1
    
    def _determine_transaction_price(self, demand_offer, supply_offer):
        """
        @theory
        确定成交价格：基于博弈理论的价格形成机制
        考虑村精英优势和市场力量对比
        """
        bid_price = demand_offer['bid_price']
        reservation_price = supply_offer['reservation_price']
        
        # 基础价格：出价和保留价的加权平均
        base_price = (bid_price + reservation_price) / 2
        
        # 村精英优势调整
        elite_bonus = 0
        if demand_offer['is_elite']:
            elite_bonus += 0.05  # 村精英需求方获得5%价格优势
        if supply_offer['is_elite']:
            elite_bonus += 0.05  # 村精英供给方获得5%价格优势
        
        # 市场力量对比调整
        price_range = bid_price - reservation_price
        market_power_adjustment = price_range * 0.1  # 10%的价格范围作为调整
        
        # 最终价格
        final_price = base_price + elite_bonus + market_power_adjustment
        
        # 确保价格在合理范围内
        return max(reservation_price, min(bid_price, final_price))
    
    def _update_market_statistics(self):
        """
        @theory
        更新市场统计信息
        """
        if self.transactions:
            self.market_statistics['total_transactions'] = len(self.transactions)
            self.market_statistics['total_area_transferred'] = sum(
                t['patch'].area for t in self.transactions
            )
            self.market_statistics['average_price'] = sum(
                t['price'] for t in self.transactions
            ) / len(self.transactions)
            
            # 计算价格波动性
            prices = [t['price'] for t in self.transactions]
            if len(prices) > 1:
                import numpy as np
                self.market_statistics['price_volatility'] = np.std(prices)
    
    def get_market_statistics(self):
        """
        @theory
        获取市场统计信息
        """
        if not self.transactions:
            return {
                'transaction_count': 0,
                'average_price': 0,
                'total_area': 0,
                'elite_advantage_count': 0,
                'price_volatility': 0,
                'supply_count': getattr(self, 'last_supply_count', 0),
                'demand_count': getattr(self, 'last_demand_count', 0)
            }
        
        return {
            'transaction_count': len(self.transactions),
            'average_price': self.market_statistics['average_price'],
            'total_area': self.market_statistics['total_area_transferred'],
            'elite_advantage_count': self.market_statistics['elite_advantage_count'],
            'price_volatility': self.market_statistics['price_volatility'],
            'supply_count': getattr(self, 'last_supply_count', 0),
            'demand_count': getattr(self, 'last_demand_count', 0)
        }
    
    def get_detailed_transactions(self):
        """
        @theory
        获取详细交易记录
        用于动态博弈的反馈分析
        """
        return self.transactions
    
    def get_agent_performance(self):
        """
        @theory
        获取智能体表现统计
        用于生计策略演化和动态博弈
        """
        agent_stats = {}
        
        # 统计农户表现
        for transaction in self.transactions:
            supplier = transaction['supplier']
            if hasattr(supplier, 'unique_id'):
                if supplier.unique_id not in agent_stats:
                    agent_stats[supplier.unique_id] = {
                        'type': 'farmer',
                        'transactions': 0,
                        'total_rental_income': 0,
                        'is_elite': transaction['supplier_is_elite']
                    }
                agent_stats[supplier.unique_id]['transactions'] += 1
                agent_stats[supplier.unique_id]['total_rental_income'] += transaction['price']
        
        # 统计企业表现
        for transaction in self.transactions:
            demander = transaction['demander']
            if hasattr(demander, 'unique_id'):
                if demander.unique_id not in agent_stats:
                    agent_stats[demander.unique_id] = {
                        'type': 'enterprise',
                        'transactions': 0,
                        'total_rental_cost': 0,
                        'is_elite': transaction['demander_is_elite']
                    }
                agent_stats[demander.unique_id]['transactions'] += 1
                agent_stats[demander.unique_id]['total_rental_cost'] += transaction['price']
        
        return agent_stats 