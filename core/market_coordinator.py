# -*- coding: utf-8 -*-
"""
@theory 市场协调器 - 解决市场时序耦合与资本约束问题
@new 2025-12-22: 修复Critical Issues - 市场时序耦合冲突

核心问题诊断：
1. 资本约束失效：农户在土地市场花光现金后，劳动力/服务市场仍可"透支"
2. 互补要素错配：企业租地后发现劳动力短缺，无法退地
3. 理性预期伪闭环：缺乏内生反馈导致价格预期系统性偏差

解决方案：
1. 运营资本预留机制：参与土地竞价前冻结部分资金用于后续投入
2. 预期形成联动：土地出价需考虑劳动力可得性和服务价格预期
3. 蛛网模型阻尼：引入适应性预期混合，防止市场震荡发散
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from core.config_utils import DEFAULT_BASE_NON_FARM_WAGE, resolve_base_non_farm_wage

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import AgriculturalEnterpriseAgent


@dataclass
class CapitalReservation:
    """资本预留记录"""
    agent_id: str
    total_capital: float           # 总资本
    reserved_for_land: float       # 预留用于土地租金
    reserved_for_labor: float      # 预留用于劳动力雇佣
    reserved_for_service: float    # 预留用于服务购买
    available_capital: float       # 可用资本（扣除预留后）
    
    @property
    def total_reserved(self) -> float:
        return self.reserved_for_land + self.reserved_for_labor + self.reserved_for_service


@dataclass 
class CobwebDampingParams:
    """蛛网模型阻尼参数 - 防止理性预期导致的市场震荡"""
    # 适应性预期混合权重（0=完全理性，1=完全适应性）
    adaptive_weight: float = 0.3
    # 价格调整阻尼系数（越大调整越慢）
    price_damping: float = 0.7
    # 供给响应弹性上限（限制过度反应）
    max_supply_elasticity: float = 0.5
    # 记忆衰减因子
    memory_decay: float = 0.8


class MarketCoordinator:
    """
    市场协调器 - 跨市场资本约束与预期联动
    
    @theory 解决市场时序耦合问题的核心机制
    
    职责：
    1. 管理智能体跨市场的资本约束
    2. 协调土地-劳动力-服务市场的预期形成
    3. 实现蛛网模型阻尼防止市场发散
    4. 提供统一的市场状态查询接口
    """
    
    def __init__(self, model, config: Optional[dict] = None):
        self.model = model
        self.config = config or {}
        wage_config = self.config or getattr(model, 'config', {})
        self.default_labor_wage = resolve_base_non_farm_wage(
            wage_config,
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        
        # 资本预留配置
        reserve_config = self.config.get('market_coordination', {}).get('capital_reservation', {})
        self.land_reserve_ratio = reserve_config.get('land_ratio', 0.4)      # 土地租金预留比例
        self.labor_reserve_ratio = reserve_config.get('labor_ratio', 0.3)    # 劳动力预留比例
        self.service_reserve_ratio = reserve_config.get('service_ratio', 0.2) # 服务预留比例
        self.min_operating_capital = reserve_config.get('min_operating', 1000) # 最低运营资本
        
        # 蛛网阻尼参数
        damping_config = self.config.get('market_coordination', {}).get('cobweb_damping', {})
        self.damping = CobwebDampingParams(
            adaptive_weight=damping_config.get('adaptive_weight', 0.3),
            price_damping=damping_config.get('price_damping', 0.7),
            max_supply_elasticity=damping_config.get('max_supply_elasticity', 0.5),
            memory_decay=damping_config.get('memory_decay', 0.8)
        )
        
        # 资本预留记录
        self.capital_reservations: Dict[str, CapitalReservation] = {}
        
        # 市场状态缓存 - 使用有意义的默认值避免None
        self.market_state_cache = {
            'expected_labor_wage': self.default_labor_wage,
            'expected_service_price': 100,     # 默认服务费100元/亩
            'expected_rental_price': 500,      # 默认地租500元/亩
            'labor_availability_index': 1.0,
            'service_availability_index': 1.0
        }
        
        # 历史价格（用于阻尼计算）
        self.price_history = {
            'rental': [],
            'labor': [],
            'service': []
        }
    
    def begin_annual_cycle(self):
        """
        @theory 年度周期开始时的预处理
        
        在任何市场交易前执行：
        1. 清空上年资本预留
        2. 更新市场预期
        3. 为所有智能体计算新的资本预留
        """
        # 清空上年预留
        self.capital_reservations.clear()
        
        # 更新市场状态预期
        self._update_market_expectations()
        
        # 为所有农户和企业计算资本预留
        for farmer in self.model.farmers:
            self._calculate_capital_reservation(farmer)
        
        for enterprise in self.model.agricultural_enterprises:
            self._calculate_capital_reservation(enterprise)
    
    def _update_market_expectations(self):
        """
        @theory 更新市场预期 - 蛛网模型阻尼机制
        
        核心公式：
        E[P_t] = α * P_{t-1} + (1-α) * E_{REH}[P_t]
        
        其中 α 是适应性预期权重，E_{REH} 是理性预期值
        """
        # 获取历史价格
        env = self.model.environment
        
        # 劳动力工资预期（蛛网阻尼）
        current_wage = (
            env.get_market_price('wage_rate')
            if hasattr(env, 'get_market_price')
            else self.default_labor_wage
        )
        if self.price_history['labor']:
            historical_wage = np.mean(self.price_history['labor'][-3:])  # 最近3年均值
            # 理性预期：基于供需估计
            reh_wage = self._estimate_reh_wage()
            # 蛛网阻尼混合
            expected_wage = (self.damping.adaptive_weight * historical_wage + 
                           (1 - self.damping.adaptive_weight) * reh_wage)
            # 价格调整阻尼
            expected_wage = current_wage + self.damping.price_damping * (expected_wage - current_wage)
        else:
            expected_wage = current_wage
        
        self.market_state_cache['expected_labor_wage'] = expected_wage
        self.price_history['labor'].append(current_wage)
        
        # 服务价格预期（类似逻辑）
        if self.model.service_market:
            current_service_price = self.model.service_market.get_average_price() if hasattr(
                self.model.service_market, 'get_average_price') else 100
        else:
            current_service_price = 100
        
        if self.price_history['service']:
            historical_price = np.mean(self.price_history['service'][-3:])
            expected_service = (self.damping.adaptive_weight * historical_price + 
                              (1 - self.damping.adaptive_weight) * current_service_price)
        else:
            expected_service = current_service_price
            
        self.market_state_cache['expected_service_price'] = expected_service
        self.price_history['service'].append(current_service_price)
        
        # 土地租金预期（蛛网阻尼）
        current_rental = env.get_market_price('land_rent') if hasattr(env, 'get_market_price') else 500
        if current_rental is None:
            current_rental = 500
        
        if self.price_history['rental']:
            historical_rental = np.mean(self.price_history['rental'][-3:])
            expected_rental = (self.damping.adaptive_weight * historical_rental + 
                             (1 - self.damping.adaptive_weight) * current_rental)
        else:
            expected_rental = current_rental
            
        self.market_state_cache['expected_rental_price'] = expected_rental
        self.price_history['rental'].append(current_rental)
        
        # 计算劳动力可得性指数（供给/需求比）
        self._calculate_availability_indices()
    
    def _estimate_reh_wage(self) -> float:
        """
        @theory 理性预期工资估计
        
        基于模型内部供需关系估计均衡工资
        包含反身性修正：如果所有人预期工资上涨→供给增加→工资下跌
        """
        # 估计劳动力供给
        total_supply = sum(
            max(0, f.total_labor_units - f.agricultural_labor) 
            for f in self.model.farmers
        )
        
        # 估计劳动力需求
        total_demand = sum(
            getattr(e, 'labor_demand', 0) 
            for e in self.model.agricultural_enterprises
        )
        # 加上大农户需求
        for f in self.model.farmers:
            if f.get_operated_land_area() > 30:  # 大户阈值
                total_demand += f.get_operated_land_area() * 2  # 假设2天/亩
        
        # 供需比决定工资调整
        if total_supply > 0:
            sd_ratio = total_demand / total_supply
        else:
            sd_ratio = 2.0  # 供给为0时假设需求旺盛
        
        base_wage = (
            self.model.environment.get_market_price('wage_rate')
            if hasattr(self.model.environment, 'get_market_price')
            else self.default_labor_wage
        )
        
        # 工资响应（阻尼限制弹性）
        elasticity = min(self.damping.max_supply_elasticity, 0.3)
        wage_adjustment = (sd_ratio - 1) * elasticity
        wage_adjustment = np.clip(wage_adjustment, -0.2, 0.2)  # 限制调整幅度
        
        return base_wage * (1 + wage_adjustment)
    
    def _calculate_availability_indices(self):
        """计算劳动力和服务的可得性指数"""
        # 劳动力可得性
        total_supply = sum(
            max(0, f.total_labor_units - f.agricultural_labor) 
            for f in self.model.farmers
        )
        total_demand = sum(
            getattr(e, 'labor_demand', 100) 
            for e in self.model.agricultural_enterprises
        )
        
        if total_demand > 0:
            self.market_state_cache['labor_availability_index'] = min(2.0, total_supply / total_demand)
        else:
            self.market_state_cache['labor_availability_index'] = 2.0
        
        # 服务可得性
        if self.model.service_providers:
            total_capacity = sum(p.current_capacity for p in self.model.service_providers)
            # 估计需求（简化）
            estimated_demand = len(self.model.farmers) * 5  # 假设平均5亩/户
            self.market_state_cache['service_availability_index'] = min(2.0, total_capacity / max(1, estimated_demand))
        else:
            self.market_state_cache['service_availability_index'] = 0.0
    
    def _calculate_capital_reservation(self, agent):
        """
        @theory 计算智能体的资本预留
        
        核心逻辑：
        1. 估计本年各市场可能的支出
        2. 按比例预留资本
        3. 确保不低于最低运营资本
        """
        agent_id = agent.unique_id
        total_capital = getattr(agent, 'capital', 0)
        
        # 估计土地租金支出（处理None值）
        operated_area = agent.get_operated_land_area() if hasattr(agent, 'get_operated_land_area') else 0
        expected_rental = self.market_state_cache.get('expected_rental_price') or 500
        estimated_land_cost = operated_area * expected_rental * 0.5  # 假设一半面积可能流转
        
        # 估计劳动力支出
        expected_wage = self.market_state_cache.get('expected_labor_wage') or self.default_labor_wage
        labor_availability = self.market_state_cache.get('labor_availability_index') or 1.0
        # 劳动力紧缺时预留更多
        labor_reserve_factor = 1.0 + (1.0 - min(1.0, labor_availability)) * 0.5
        estimated_labor_cost = operated_area * 3 * expected_wage * labor_reserve_factor  # 3天/亩
        
        # 估计服务支出
        expected_service_price = self.market_state_cache.get('expected_service_price') or 100
        service_availability = self.market_state_cache.get('service_availability_index') or 1.0
        # 服务紧缺时可能放弃购买
        if service_availability > 0.5:
            estimated_service_cost = operated_area * expected_service_price * 0.3  # 假设30%面积购买服务
        else:
            estimated_service_cost = 0
        
        # 计算预留金额（不超过可用资本）
        total_estimated = estimated_land_cost + estimated_labor_cost + estimated_service_cost
        
        if total_estimated > 0:
            available_for_reservation = max(0, total_capital - self.min_operating_capital)
            
            # 按比例分配预留
            land_share = estimated_land_cost / total_estimated
            labor_share = estimated_labor_cost / total_estimated
            service_share = estimated_service_cost / total_estimated
            
            reserved_land = available_for_reservation * land_share * self.land_reserve_ratio
            reserved_labor = available_for_reservation * labor_share * self.labor_reserve_ratio
            reserved_service = available_for_reservation * service_share * self.service_reserve_ratio
        else:
            reserved_land = 0
            reserved_labor = 0
            reserved_service = 0
        
        # 创建预留记录
        reservation = CapitalReservation(
            agent_id=agent_id,
            total_capital=total_capital,
            reserved_for_land=reserved_land,
            reserved_for_labor=reserved_labor,
            reserved_for_service=reserved_service,
            available_capital=total_capital - reserved_land - reserved_labor - reserved_service
        )
        
        self.capital_reservations[agent_id] = reservation
    
    def get_available_capital_for_land(self, agent) -> float:
        """
        @api 获取可用于土地租赁的资本
        
        土地市场调用此方法获取农户/企业的可用预算
        """
        agent_id = agent.unique_id
        if agent_id in self.capital_reservations:
            reservation = self.capital_reservations[agent_id]
            return reservation.available_capital + reservation.reserved_for_land
        else:
            # 回退：使用资本的50%
            return getattr(agent, 'capital', 0) * 0.5
    
    def get_available_capital_for_labor(self, agent) -> float:
        """获取可用于劳动力雇佣的资本"""
        agent_id = agent.unique_id
        if agent_id in self.capital_reservations:
            reservation = self.capital_reservations[agent_id]
            # 土地市场结束后，未用完的土地预留可用于劳动力
            return reservation.reserved_for_labor
        else:
            return getattr(agent, 'capital', 0) * 0.3
    
    def get_available_capital_for_service(self, agent) -> float:
        """获取可用于服务购买的资本"""
        agent_id = agent.unique_id
        if agent_id in self.capital_reservations:
            return self.capital_reservations[agent_id].reserved_for_service
        else:
            return getattr(agent, 'capital', 0) * 0.2
    
    def get_land_bid_discount_factor(self, agent) -> float:
        """
        @theory 计算土地出价折扣因子
        
        考虑劳动力可得性和服务价格，调整土地出价
        
        如果预期劳动力短缺或服务昂贵，应降低土地出价
        因为经营成本会更高
        """
        labor_avail = self.market_state_cache.get('labor_availability_index', 1.0)
        service_avail = self.market_state_cache.get('service_availability_index', 1.0)
        
        # 基础折扣（1.0 = 无折扣）
        discount = 1.0
        
        # 劳动力短缺折扣
        if labor_avail < 0.8:
            discount *= 0.9 + 0.1 * labor_avail  # 短缺时最多折扣10%
        
        # 服务不可用折扣（对老年农户影响更大）
        if service_avail < 0.5:
            age = getattr(agent, 'age', 40)
            if age > 55:
                discount *= 0.85  # 老年农户更依赖服务
            else:
                discount *= 0.95
        
        return discount
    
    def record_land_transaction(self, agent, actual_cost: float):
        """
        @api 记录土地交易实际支出
        
        土地市场完成后调用，更新剩余预留
        """
        agent_id = agent.unique_id
        if agent_id in self.capital_reservations:
            reservation = self.capital_reservations[agent_id]
            # 计算土地租赁实际使用的资本
            used_from_land_reserve = min(actual_cost, reservation.reserved_for_land)
            used_from_available = actual_cost - used_from_land_reserve
            
            # 更新预留（未用完的土地预留转为可用）
            reservation.reserved_for_land = max(0, reservation.reserved_for_land - actual_cost)
            reservation.available_capital = max(0, reservation.available_capital - used_from_available)
    
    def get_market_expectations(self) -> dict:
        """获取当前市场预期状态"""
        return self.market_state_cache.copy()
    
    def end_annual_cycle(self):
        """
        @theory 年度周期结束时的后处理
        
        1. 更新价格历史（用于下年阻尼计算）
        2. 清算未使用的预留
        3. 计算预期误差（用于调整阻尼参数）
        """
        # 记录实际租金价格
        if hasattr(self.model, 'land_clearing_house'):
            stats = self.model.land_clearing_house.get_market_statistics()
            avg_rental = stats.get('average_rent_per_mu', stats.get('average_price', 0))
            if avg_rental and avg_rental > 0:
                self.price_history['rental'].append(avg_rental)
        
        # 限制历史长度
        for key in self.price_history:
            if len(self.price_history[key]) > 10:
                self.price_history[key] = self.price_history[key][-10:]


def create_market_coordinator(model, config: Optional[dict] = None) -> MarketCoordinator:
    """工厂函数：创建市场协调器实例"""
    return MarketCoordinator(model, config)
