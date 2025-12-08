# -*- coding: utf-8 -*-
"""
@module agricultural_service.py
@description 农业社会化服务模块 (Agricultural Socialized Services, ASS)
@theory 中国农业正在经历"隐性革命"：土地不流转，服务流转
       "小农户+大服务" 模式正在成为规模化的主要路径
@reference 
    - 黄宗智《中国的隐性农业革命》
    - 罗必良《农业社会化服务体系研究》
    - 陈义媛、高原: 服务规模经济研究
@created 2025-12-04

核心机制：
1. 服务商Agent：拥有大型机械和技术，不占有土地
2. 农户购买服务：在"自种"和"出租"外增加"托管"选项
3. 服务规模经济：服务商为多个农户提供服务，实现规模效应
4. 涌现现象：解释为何中国仍存在海量小农户但粮食产量未崩盘
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np


class ServiceType(Enum):
    """服务类型枚举"""
    TILLAGE = "tillage"           # 耕作服务（旋耕、深翻）
    SEEDING = "seeding"           # 播种服务
    FERTILIZATION = "fertilization"  # 施肥服务
    PEST_CONTROL = "pest_control"  # 植保服务（打药）
    HARVESTING = "harvesting"      # 收割服务
    FULL_CUSTODY = "full_custody"  # 全程托管（耕种收防）
    
    # 新增：与model.py中的ServiceType对应
    FULL_TRUSTEESHIP = "Full_Trusteeship"
    PARTIAL_TRUSTEESHIP = "Partial_Trusteeship"
    SINGLE_SERVICE = "Single_Service"


class CustodyType(Enum):
    """托管类型枚举"""
    NONE = "none"                  # 不托管（自耕）
    PARTIAL = "partial"            # 半托管（部分环节）
    FULL = "full"                  # 全托管（全程托管）


@dataclass
class ServiceContract:
    """服务合同"""
    farmer_id: str                 # 农户ID
    service_provider_id: str       # 服务商ID
    land_area: float               # 托管面积（亩）
    crop_type: str                 # 作物类型
    custody_type: CustodyType      # 托管类型
    service_fee: float             # 服务费（元/亩）
    duration: int = 1              # 合同期限（年）
    year_signed: int = 0           # 签约年份
    
    # 新增字段（兼容model.py）
    service_type: Optional[ServiceType] = None
    fee_per_mu: float = 0.0
    coverage_ratio: float = 1.0
    area: float = 0.0
    
    # 收益分成（仅全托管时使用）
    revenue_share_farmer: float = 0.0   # 农户收益分成比例
    revenue_share_provider: float = 0.0  # 服务商收益分成比例
    
    def __post_init__(self):
        """初始化后处理"""
        if self.fee_per_mu == 0.0:
            self.fee_per_mu = self.service_fee
        if self.area == 0.0:
            self.area = self.land_area
    
    def is_expired(self, current_year: int) -> bool:
        """检查合同是否到期"""
        return current_year >= self.year_signed + self.duration


class ServiceProvider:
    """
    农业服务商 (Mesa-compatible Agent)
    @theory 不占有土地，但拥有大型机械和技术
    @new 2025-12: 重构为兼容Mesa框架的Agent类
    """
    
    def __init__(self, unique_id: str, model: Any, position: Tuple[int, int] = None,
                 initial_capacity: float = 500.0, service_radius: int = 15):
        """
        初始化服务提供商
        
        Args:
            unique_id: 唯一标识
            model: Mesa模型引用
            position: 网格位置 (x, y)
            initial_capacity: 初始服务能力（亩/年）
            service_radius: 服务半径（格子单位）
        """
        self.unique_id = unique_id
        self.model = model
        self.pos = position
        
        # 服务能力
        self.initial_capacity = initial_capacity
        self.current_capacity = initial_capacity
        self.max_capacity = initial_capacity * 2  # 最大扩张到2倍
        
        # 机械设备
        self.large_tractors = max(1, int(initial_capacity / 200))
        self.combine_harvesters = max(1, int(initial_capacity / 400))
        self.drone_sprayers = max(1, int(initial_capacity / 300))
        
        # 服务范围
        self.service_radius = service_radius
        
        # 当前服务面积和合同
        self.service_area_used = 0.0
        self.active_contracts: List[ServiceContract] = []
        
        # 财务
        self.capital = 100000.0
        self.annual_revenue = 0.0
        self.annual_cost = 0.0
        self.cumulative_profit = 0.0
        
        # 服务定价（元/亩）- 根据配置动态调整
        self.service_prices = {
            ServiceType.FULL_TRUSTEESHIP: 500.0,
            ServiceType.PARTIAL_TRUSTEESHIP: 280.0,
            ServiceType.SINGLE_SERVICE: 100.0,
            ServiceType.FULL_CUSTODY: 450.0,
            ServiceType.HARVESTING: 120.0,
            ServiceType.TILLAGE: 80.0,
            ServiceType.SEEDING: 50.0,
        }
        
        # 效率因子（规模经济）
        self.efficiency_factor = 1.0
        self._update_efficiency()
    
    def _update_efficiency(self):
        """
        @theory 规模经济效应
        服务面积越大，单位成本越低，效率越高
        """
        utilization = self.service_area_used / max(1, self.current_capacity)
        # 利用率越高，效率越好（但不超过1.3）
        self.efficiency_factor = 1.0 + min(0.3, utilization * 0.4)
    
    def get_remaining_capacity(self) -> float:
        """获取剩余服务能力"""
        return max(0, self.current_capacity - self.service_area_used)
    
    def can_serve(self, farmer_pos: Tuple[int, int], required_area: float) -> bool:
        """
        检查是否能为该农户提供服务
        考虑：距离和容量
        """
        if required_area > self.get_remaining_capacity():
            return False
        
        if farmer_pos is None or self.pos is None:
            return True  # 无位置信息时默认可服务
        
        # 计算距离
        dx = abs(farmer_pos[0] - self.pos[0])
        dy = abs(farmer_pos[1] - self.pos[1])
        distance = max(dx, dy)  # 切比雪夫距离
        
        return distance <= self.service_radius
    
    def calculate_service_fee(self, area: float, service_type: ServiceType) -> float:
        """
        计算服务费用
        @theory 规模折扣 + 效率加成
        """
        base_price = self.service_prices.get(service_type, 300.0)
        
        # 规模折扣
        if area >= 100:
            discount = 0.80  # 20%折扣
        elif area >= 50:
            discount = 0.88  # 12%折扣
        elif area >= 20:
            discount = 0.93  # 7%折扣
        elif area >= 10:
            discount = 0.97  # 3%折扣
        else:
            discount = 1.0
        
        # 效率加成（高效服务商可以给更低价格）
        efficiency_discount = 1.0 - (self.efficiency_factor - 1.0) * 0.5
        
        return base_price * discount * efficiency_discount
    
    def add_service_contract(self, contract: ServiceContract):
        """添加服务合同"""
        self.active_contracts.append(contract)
        self.service_area_used += contract.area
        self._update_efficiency()
    
    def provide_service(self, contract: ServiceContract) -> Dict:
        """
        执行服务并返回结果
        """
        service_fee = contract.fee_per_mu * contract.area
        
        result = {
            'yield_bonus': self.efficiency_factor,
            'labor_saved': contract.area * 200 * contract.coverage_ratio,
            'service_fee': service_fee,
        }
        
        self.annual_revenue += service_fee
        return result
    
    def end_of_year_settlement(self, current_year: int) -> float:
        """年末结算"""
        # 设备折旧
        equipment_depreciation = (
            self.large_tractors * 12000 +
            self.combine_harvesters * 20000 +
            self.drone_sprayers * 3000
        )
        
        # 运营成本
        labor_cost = self.service_area_used * 45
        fuel_cost = self.service_area_used * 25
        maintenance_cost = equipment_depreciation * 0.1
        
        self.annual_cost = equipment_depreciation + labor_cost + fuel_cost + maintenance_cost
        
        # 利润
        profit = self.annual_revenue - self.annual_cost
        self.capital += profit
        self.cumulative_profit += profit
        
        # 清理过期合同
        self.active_contracts = [
            c for c in self.active_contracts
            if not c.is_expired(current_year)
        ]
        
        # 根据利润调整能力
        if profit > 50000 and self.current_capacity < self.max_capacity:
            # 扩张能力
            expansion = min(0.2 * self.current_capacity, self.max_capacity - self.current_capacity)
            self.current_capacity += expansion
            self.large_tractors += 1
        
        # 重置年度计数
        self.service_area_used = sum(c.area for c in self.active_contracts)
        self.annual_revenue = 0.0
        self.annual_cost = 0.0
        
        self._update_efficiency()
        
        return profit
    
    def step(self):
        """Mesa Agent step（如果加入schedule）"""
        pass  # 服务商的行为主要在market层面协调


class ServiceMarket:
    """
    农业社会化服务市场
    @theory 协调农户和服务商之间的服务交易
    @new 2025-12: 为model.py提供接口
    """
    
    def __init__(self, model: Any, config: dict = None):
        self.model = model
        self.config = config or {}
        self.transaction_history: List[Dict] = []
        
        # 统计
        self.total_area_served = 0.0
        self.total_revenue = 0.0
        self.n_contracts_signed = 0
    
    def match_service_demands(self, demands: List[Dict], providers: List[ServiceProvider]) -> List[Dict]:
        """
        匹配服务需求与服务提供商
        
        Args:
            demands: 农户服务需求列表 [{farmer, area, service_type, max_price_per_mu}]
            providers: 服务提供商列表
            
        Returns:
            匹配结果列表 [{farmer, provider, contract}]
        """
        matches = []
        
        # 按需求面积排序（大户优先，更容易找到服务商）
        sorted_demands = sorted(demands, key=lambda d: d.get('area', 0), reverse=True)
        
        for demand in sorted_demands:
            farmer = demand.get('farmer')
            area = demand.get('area', 0)
            service_type = demand.get('service_type', ServiceType.SINGLE_SERVICE)
            max_price = demand.get('max_price_per_mu', 600)
            
            if farmer is None or area <= 0:
                continue
            
            farmer_pos = getattr(farmer, 'pos', None)
            
            # 寻找可用的服务商
            available_providers = [
                p for p in providers
                if p.can_serve(farmer_pos, area)
            ]
            
            if not available_providers:
                continue
            
            # 按价格排序，选择最便宜的
            best_provider = None
            best_price = float('inf')
            
            for provider in available_providers:
                price = provider.calculate_service_fee(area, service_type)
                if price <= max_price * area and price < best_price:
                    best_provider = provider
                    best_price = price
            
            if best_provider is not None:
                # 创建合同
                fee_per_mu = best_price / area
                
                # 确定覆盖比例
                if service_type == ServiceType.FULL_TRUSTEESHIP:
                    coverage = 1.0
                elif service_type == ServiceType.PARTIAL_TRUSTEESHIP:
                    coverage = 0.5
                else:
                    coverage = 0.2
                
                contract = ServiceContract(
                    farmer_id=farmer.unique_id,
                    service_provider_id=best_provider.unique_id,
                    land_area=area,
                    crop_type='grain',
                    custody_type=CustodyType.FULL if coverage == 1.0 else CustodyType.PARTIAL,
                    service_fee=fee_per_mu,
                    service_type=service_type,
                    fee_per_mu=fee_per_mu,
                    coverage_ratio=coverage,
                    area=area,
                    year_signed=self.model.step_count if self.model else 0
                )
                
                # 记录匹配
                matches.append({
                    'farmer': farmer,
                    'provider': best_provider,
                    'contract': contract
                })
                
                # 更新统计
                self.total_area_served += area
                self.total_revenue += best_price
                self.n_contracts_signed += 1
                
                # 记录历史
                self.transaction_history.append({
                    'year': self.model.step_count if self.model else 0,
                    'farmer_id': farmer.unique_id,
                    'provider_id': best_provider.unique_id,
                    'area': area,
                    'service_type': service_type.value,
                    'fee_per_mu': fee_per_mu
                })
        
        return matches
    
    def get_statistics(self) -> Dict:
        """获取市场统计"""
        return {
            'total_area_served': self.total_area_served,
            'total_revenue': self.total_revenue,
            'n_contracts': self.n_contracts_signed,
            'avg_fee_per_mu': self.total_revenue / max(1, self.total_area_served)
        }


# ============================================================
# 以下保留原有的 dataclass 版本 ServiceProvider 作为 LegacyServiceProvider
# 以保持向后兼容
# ============================================================

@dataclass
class LegacyServiceProvider:
    """
    农业服务商 (旧版 dataclass)
    @deprecated 请使用新的 ServiceProvider 类
    """
    provider_id: str
    name: str
    
    large_tractors: int = 2
    combine_harvesters: int = 1
    drone_sprayers: int = 1
    
    annual_service_capacity: float = 3000.0
    current_service_area: float = 0.0
    
    capital: float = 500000.0
    annual_revenue: float = 0.0
    annual_cost: float = 0.0
    
    service_prices: Dict[ServiceType, float] = field(default_factory=lambda: {
        ServiceType.TILLAGE: 80.0,
        ServiceType.SEEDING: 50.0,
        ServiceType.FERTILIZATION: 30.0,
        ServiceType.PEST_CONTROL: 40.0,
        ServiceType.HARVESTING: 120.0,
        ServiceType.FULL_CUSTODY: 450.0,
    })
    
    efficiency_factor: float = 1.3
    active_contracts: List[ServiceContract] = field(default_factory=list)


class AgriculturalServiceMarket:
    """
    农业服务市场 (旧版，保持向后兼容)
    @deprecated 请使用新的 ServiceMarket 类
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.service_providers: List[LegacyServiceProvider] = []
        self.transaction_history: List[Dict] = []
        
        self.default_full_custody_price = self.config.get('full_custody_price', 450)
        self.default_partial_custody_price = self.config.get('partial_custody_price', 200)
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化服务商"""
        n_providers = self.config.get('n_service_providers', 3)
        
        for i in range(n_providers):
            provider = LegacyServiceProvider(
                provider_id=f"sp_{i}",
                name=f"农机服务合作社{i+1}",
                annual_service_capacity=self.config.get('service_capacity', 3000),
            )
            self.service_providers.append(provider)
    
    def farmer_evaluate_custody_decision(self, farmer, crop_type: str = 'grain') -> Tuple[CustodyType, float]:
        """
        农户评估是否托管
        """
        operated_area = farmer.get_operated_land_area()
        
        if operated_area <= 0:
            return CustodyType.NONE, 0.0
        
        age = getattr(farmer, 'age', 45)
        off_farm_income = getattr(farmer, 'off_farm_income', 0)
        farming_ability = getattr(farmer, 'farming_ability', 1.0)
        
        # 估算各选项的净收益
        self_farming_yield = operated_area * 500 * farming_ability
        self_farming_price = 2.5
        self_farming_revenue = self_farming_yield * self_farming_price
        self_farming_material_cost = operated_area * 400
        self_farming_net = self_farming_revenue - self_farming_material_cost
        
        partial_service_fee = operated_area * self.default_partial_custody_price
        partial_yield_bonus = 1.05
        partial_revenue = self_farming_yield * partial_yield_bonus * self_farming_price
        partial_labor_saved = operated_area * 80
        partial_net = partial_revenue - self_farming_material_cost - partial_service_fee + partial_labor_saved
        
        full_service_fee = operated_area * self.default_full_custody_price
        full_yield_bonus = 1.15
        full_revenue = self_farming_yield * full_yield_bonus * self_farming_price
        full_net = full_revenue - full_service_fee
        
        # 调整因子
        age_factor = 1.3 if age > 60 else (1.15 if age > 55 else 1.0)
        opportunity_cost_factor = 1.3 if off_farm_income > 20000 else (1.15 if off_farm_income > 10000 else 1.0)
        scale_factor = 1.4 if operated_area < 5 else (1.2 if operated_area < 10 else 1.0)
        
        partial_net_adjusted = partial_net * age_factor * opportunity_cost_factor * scale_factor
        full_net_adjusted = full_net * age_factor * opportunity_cost_factor * scale_factor
        
        best_option = CustodyType.NONE
        best_net = self_farming_net
        
        if partial_net_adjusted > best_net:
            best_option = CustodyType.PARTIAL
            best_net = partial_net_adjusted
        
        if full_net_adjusted > best_net:
            best_option = CustodyType.FULL
            best_net = full_net_adjusted
        
        return best_option, best_net
    
    def get_market_statistics(self) -> Dict:
        """获取市场统计"""
        total_capacity = sum(p.annual_service_capacity for p in self.service_providers)
        total_utilized = sum(p.current_service_area for p in self.service_providers)
        
        return {
            'n_providers': len(self.service_providers),
            'total_capacity': total_capacity,
            'total_utilized': total_utilized,
            'utilization_rate': total_utilized / total_capacity if total_capacity > 0 else 0,
            'n_active_contracts': sum(len(p.active_contracts) for p in self.service_providers)
        }


# 辅助函数
def integrate_custody_decision(farmer, service_market: AgriculturalServiceMarket, 
                                crop_type: str, current_year: int) -> Dict:
    """
    将托管决策整合到农户的年度决策中
    """
    custody_type, expected_net = service_market.farmer_evaluate_custody_decision(farmer, crop_type)
    
    result = {
        'custody_type': custody_type,
        'contract': None,
        'yield_bonus': 1.0,
        'service_fee': 0.0
    }
    
    if custody_type != CustodyType.NONE:
        # 这里可以扩展匹配逻辑
        result['custody_type'] = custody_type
        result['yield_bonus'] = 1.15 if custody_type == CustodyType.FULL else 1.05
    
    return result
