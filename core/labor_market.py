# -*- coding: utf-8 -*-
"""
劳动力市场模块
@module: core/labor_market.py
@author: AI Assistant
@date: 2025-11-29
@theory:
    实现农村劳动力市场的完整机制：
    1. 农户劳动力供给：自用 + 出售给其他农户/企业 + 外出务工
    2. 企业劳动力需求：雇佣农户劳动力 + 使用机械替代
    3. 大户劳动力需求：规模经营需要雇佣帮工
    4. 劳动力市场出清：匹配供需，确定工资

@reference:
    - 黄宗智《中国的隐性农业革命》关于农业劳动力市场
    - Deininger & Jin (2005) Land rental markets
    - CLES数据：农业日工资约80-120元/天
"""

import numpy as np
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import EnterpriseAgent


class LaborType(Enum):
    """劳动力类型"""
    FAMILY_LABOR = "family"          # 家庭劳动力（自用）
    HIRED_LABOR = "hired"            # 雇佣劳动力
    SEASONAL_WORKER = "seasonal"     # 季节工
    MACHINERY = "machinery"          # 机械替代


class EmploymentRecord:
    """雇佣记录"""
    def __init__(self, employer, worker, labor_days: float, wage_rate: float, 
                 season: str = "annual", labor_type: LaborType = LaborType.HIRED_LABOR):
        self.employer = employer          # 雇主（农户或企业）
        self.worker = worker              # 劳动者（农户）
        self.labor_days = labor_days      # 雇佣天数
        self.wage_rate = wage_rate        # 日工资率
        self.total_wage = labor_days * wage_rate  # 总工资
        self.season = season              # 季节（spring/summer/autumn/winter/annual）
        self.labor_type = labor_type      # 劳动类型


class LaborMarket:
    """
    农村劳动力市场
    
    @theory:
    实现三层次劳动力配置：
    1. 家庭内部配置：农业劳动 vs 非农务工
    2. 本地农业劳动力市场：大户/企业雇佣散户
    3. 城市劳动力市场：农户外出务工
    
    劳动力市场特点：
    - 季节性波动（农忙期需求高）
    - 信息不对称（熟人网络匹配）
    - 工资刚性（存在最低工资）
    """
    
    def __init__(self, model, config: dict):
        """
        初始化劳动力市场
        
        @param model: GrainSubsidyModel实例
        @param config: 配置字典
        """
        self.model = model
        self.config = config.get('labor_market', {})
        
        # 市场参数
        self.base_agricultural_wage = self.config.get('base_agricultural_wage', 100.0)  # 农业基础日工资
        self.base_non_farm_wage = self.config.get('base_non_farm_wage', 150.0)  # 非农基础日工资
        self.wage_floor = self.config.get('wage_floor', 80.0)  # 最低工资
        self.wage_ceiling = self.config.get('wage_ceiling', 300.0)  # 工资上限
        
        # 季节性参数
        self.seasonal_factors = self.config.get('seasonal_factors', {
            'spring': 1.3,    # 春耕期需求高
            'summer': 1.0,    # 夏季正常
            'autumn': 1.5,    # 秋收期需求最高
            'winter': 0.6     # 冬季需求低
        })
        
        # 机械化参数
        self.machinery_substitution = self.config.get('machinery_substitution', {
            'grain': 0.6,       # 粮食作物机械化程度高，可替代60%劳动
            'cash_crop': 0.3    # 经济作物机械化程度低，只能替代30%劳动
        })
        
        # 劳动力需求参数（天/亩）
        self.labor_requirements = self.config.get('labor_requirements', {
            'grain': {
                'base': 8,           # 粮食基础需求8天/亩
                'with_machinery': 3  # 使用机械后3天/亩
            },
            'cash_crop': {
                'base': 20,          # 经济作物基础需求20天/亩
                'with_machinery': 14 # 使用机械后14天/亩
            }
        })
        
        # 当前市场状态
        self.current_agricultural_wage = self.base_agricultural_wage
        self.current_season = 'annual'
        
        # 雇佣记录
        self.employment_records: List[EmploymentRecord] = []
        self.labor_supply_pool: Dict = {}  # 劳动力供给池
        self.labor_demand_pool: Dict = {}  # 劳动力需求池
        
        # 历史记录
        self.market_history = []
        
    def get_current_season(self) -> str:
        """
        获取当前季节
        
        @return: 季节字符串
        """
        if hasattr(self.model, 'schedule') and hasattr(self.model.schedule, 'steps'):
            step = self.model.schedule.steps
            month = step % 12
            if month in [2, 3, 4]:
                return 'spring'
            elif month in [5, 6, 7]:
                return 'summer'
            elif month in [8, 9, 10]:
                return 'autumn'
            else:
                return 'winter'
        return 'annual'
    
    def get_seasonal_wage_factor(self) -> float:
        """
        获取季节性工资调整因子
        
        @return: 工资调整倍数
        """
        season = self.get_current_season()
        return self.seasonal_factors.get(season, 1.0)
    
    def calculate_labor_requirement(self, area: float, crop_type: str, 
                                   use_machinery: bool = False) -> float:
        """
        计算农业生产所需劳动力
        
        @param area: 种植面积（亩）
        @param crop_type: 作物类型 ('grain' or 'cash_crop')
        @param use_machinery: 是否使用大型机械
        @return: 所需劳动天数
        """
        req = self.labor_requirements.get(crop_type, self.labor_requirements['grain'])
        
        if use_machinery:
            base_days = req.get('with_machinery', req['base'] * 0.4)
        else:
            base_days = req['base']
        
        # 季节性调整
        seasonal_factor = self.get_seasonal_wage_factor()
        
        total_days = base_days * area
        
        return total_days
    
    def calculate_farmer_labor_supply(self, farmer: 'FarmerAgent') -> Dict:
        """
        计算农户的劳动力供给
        
        @param farmer: 农户Agent
        @return: 供给信息字典
        """
        # 农户总劳动力（天/年）
        total_labor = getattr(farmer, 'total_labor_units', 365)
        
        # 自用农业劳动力
        self_operated_area = farmer.get_operated_land_area()
        planned_crop = getattr(farmer, 'planned_crop', None)
        
        # 计算自用劳动力需求
        if planned_crop and self_operated_area > 0:
            from agents.farmer_agent import CropType
            crop_type_str = 'grain' if planned_crop == CropType.GRAIN else 'cash_crop'
            # 农户不使用大型机械
            self_labor_need = self.calculate_labor_requirement(
                self_operated_area, crop_type_str, use_machinery=False
            )
        else:
            self_labor_need = 0
        
        # 可供出售的劳动力
        available_for_sale = max(0, total_labor - self_labor_need)
        
        # 根据生计策略调整
        from agents.farmer_agent import LivelihoodStrategy
        strategy = getattr(farmer, 'livelihood_strategy', LivelihoodStrategy.PLURIACTIVE)
        
        if strategy == LivelihoodStrategy.PURE_FARMER:
            # 纯农户：主要从事自家农业，少量出售劳动力
            max_sell_ratio = 0.3
        elif strategy == LivelihoodStrategy.PLURIACTIVE:
            # 兼业户：自家农业 + 出售劳动力
            max_sell_ratio = 0.6
        else:  # NON_FARMER
            # 非农户：主要出售劳动力
            max_sell_ratio = 0.9
        
        available_for_sale = min(available_for_sale, total_labor * max_sell_ratio)
        
        # 农户的期望工资（基于个人特征）
        wage_level = getattr(farmer, 'off_farm_wage_level', 1.0)
        min_acceptable_wage = self.base_agricultural_wage * wage_level * 0.8
        
        return {
            'farmer': farmer,
            'total_labor': total_labor,
            'self_labor_need': self_labor_need,
            'available_for_sale': available_for_sale,
            'min_acceptable_wage': min_acceptable_wage,
            'strategy': strategy
        }
    
    def calculate_enterprise_labor_demand(self, enterprise: 'EnterpriseAgent') -> Dict:
        """
        计算企业的劳动力需求
        
        @param enterprise: 企业Agent
        @return: 需求信息字典
        """
        operated_area = enterprise.get_operated_land_area()
        
        if operated_area <= 0:
            return {
                'enterprise': enterprise,
                'total_demand': 0,
                'machinery_labor': 0,
                'hired_labor_need': 0,
                'max_wage': 0
            }
        
        # 获取作物组合
        from agents.farmer_agent import CropType
        crop_portfolio = getattr(enterprise, 'crop_portfolio', {CropType.GRAIN: 0.5, CropType.CASH_CROP: 0.5})
        
        grain_area = operated_area * crop_portfolio.get(CropType.GRAIN, 0.5)
        cash_crop_area = operated_area * crop_portfolio.get(CropType.CASH_CROP, 0.5)
        
        # 企业使用大型机械
        grain_labor = self.calculate_labor_requirement(grain_area, 'grain', use_machinery=True)
        cash_crop_labor = self.calculate_labor_requirement(cash_crop_area, 'cash_crop', use_machinery=True)
        
        total_demand = grain_labor + cash_crop_labor
        
        # 机械可替代的劳动力
        full_grain_labor = self.calculate_labor_requirement(grain_area, 'grain', use_machinery=False)
        full_cash_crop_labor = self.calculate_labor_requirement(cash_crop_area, 'cash_crop', use_machinery=False)
        machinery_saved = (full_grain_labor + full_cash_crop_labor) - total_demand
        
        # 企业愿意支付的最高工资
        efficiency_factor = getattr(enterprise, 'efficiency_factor', 1.0)
        max_wage = self.base_agricultural_wage * efficiency_factor * 1.2
        
        return {
            'enterprise': enterprise,
            'total_demand': total_demand,
            'machinery_labor': machinery_saved,  # 机械替代节省的劳动力
            'hired_labor_need': total_demand,    # 仍需雇佣的劳动力
            'max_wage': max_wage,
            'grain_area': grain_area,
            'cash_crop_area': cash_crop_area
        }
    
    def calculate_large_farmer_labor_demand(self, farmer: 'FarmerAgent') -> Dict:
        """
        计算大户的劳动力需求
        
        @param farmer: 农户Agent（大户）
        @return: 需求信息字典
        """
        operated_area = farmer.get_operated_land_area()
        total_labor = getattr(farmer, 'total_labor_units', 365)
        
        # 计算自家劳动力需求
        supply_info = self.calculate_farmer_labor_supply(farmer)
        self_labor_need = supply_info['self_labor_need']
        
        # 如果自家劳动力不足，需要雇佣
        if self_labor_need > total_labor * 0.9:  # 预留10%弹性
            # 大户（一般经营面积>30亩）可能需要雇佣帮工
            labor_gap = self_labor_need - total_labor * 0.9
            
            # 大户的期望支付工资
            farming_ability = getattr(farmer, 'farming_ability', 1.0)
            max_wage = self.base_agricultural_wage * farming_ability
            
            return {
                'farmer': farmer,
                'is_large_farmer': True,
                'operated_area': operated_area,
                'self_labor_need': self_labor_need,
                'available_family_labor': total_labor * 0.9,
                'hired_labor_need': labor_gap,
                'max_wage': max_wage
            }
        else:
            return {
                'farmer': farmer,
                'is_large_farmer': False,
                'operated_area': operated_area,
                'self_labor_need': self_labor_need,
                'available_family_labor': total_labor,
                'hired_labor_need': 0,
                'max_wage': 0
            }
    
    def collect_labor_supply(self, farmers: List['FarmerAgent']):
        """
        收集所有农户的劳动力供给
        
        @param farmers: 农户列表
        """
        self.labor_supply_pool = {}
        
        for farmer in farmers:
            supply_info = self.calculate_farmer_labor_supply(farmer)
            if supply_info['available_for_sale'] > 0:
                self.labor_supply_pool[farmer.unique_id] = supply_info
    
    def collect_labor_demand(self, farmers: List['FarmerAgent'], 
                            enterprises: List['EnterpriseAgent']):
        """
        收集所有主体的劳动力需求
        
        @param farmers: 农户列表
        @param enterprises: 企业列表
        """
        self.labor_demand_pool = {}
        
        # 企业需求
        for enterprise in enterprises:
            demand_info = self.calculate_enterprise_labor_demand(enterprise)
            if demand_info['hired_labor_need'] > 0:
                self.labor_demand_pool[f'E_{enterprise.unique_id}'] = demand_info
        
        # 大户需求
        for farmer in farmers:
            demand_info = self.calculate_large_farmer_labor_demand(farmer)
            if demand_info['hired_labor_need'] > 0:
                self.labor_demand_pool[f'F_{farmer.unique_id}'] = demand_info
    
    def match_labor_market(self) -> List[EmploymentRecord]:
        """
        劳动力市场匹配
        
        @theory:
        采用双边匹配机制：
        1. 按工资出价排序需求方
        2. 按期望工资排序供给方
        3. 从高到低匹配
        4. 确定均衡工资
        
        @return: 雇佣记录列表
        """
        self.employment_records = []
        
        if not self.labor_supply_pool or not self.labor_demand_pool:
            return self.employment_records
        
        # 按最高可支付工资排序需求方（从高到低）
        demand_sorted = sorted(
            self.labor_demand_pool.items(),
            key=lambda x: x[1].get('max_wage', 0),
            reverse=True
        )
        
        # 按最低可接受工资排序供给方（从低到高）
        supply_sorted = sorted(
            self.labor_supply_pool.items(),
            key=lambda x: x[1]['min_acceptable_wage']
        )
        
        # 匹配过程
        remaining_supply = {k: v['available_for_sale'] for k, v in self.labor_supply_pool.items()}
        
        for demand_id, demand_info in demand_sorted:
            remaining_demand = demand_info.get('hired_labor_need', 0)
            max_wage = demand_info.get('max_wage', 0)
            
            if remaining_demand <= 0:
                continue
            
            # 确定雇主
            if demand_id.startswith('E_'):
                employer = demand_info['enterprise']
            else:
                employer = demand_info['farmer']
            
            # 寻找可匹配的劳动力供给者
            for supply_id, supply_info in supply_sorted:
                if remaining_demand <= 0:
                    break
                
                worker = supply_info['farmer']
                
                # 不能雇佣自己
                if hasattr(employer, 'unique_id') and worker.unique_id == employer.unique_id:
                    continue
                
                available = remaining_supply.get(supply_id, 0)
                if available <= 0:
                    continue
                
                min_wage = supply_info['min_acceptable_wage']
                
                # 检查工资是否可接受
                if max_wage < min_wage:
                    continue  # 工资不匹配
                
                # 确定成交工资（取中间值）
                agreed_wage = (max_wage + min_wage) / 2
                
                # 确定雇佣数量
                hired_days = min(remaining_demand, available)
                
                # 创建雇佣记录
                record = EmploymentRecord(
                    employer=employer,
                    worker=worker,
                    labor_days=hired_days,
                    wage_rate=agreed_wage,
                    season=self.get_current_season(),
                    labor_type=LaborType.HIRED_LABOR
                )
                self.employment_records.append(record)
                
                # 更新剩余供需
                remaining_demand -= hired_days
                remaining_supply[supply_id] -= hired_days
        
        # 更新市场工资（基于成交记录）
        if self.employment_records:
            total_wage = sum(r.total_wage for r in self.employment_records)
            total_days = sum(r.labor_days for r in self.employment_records)
            self.current_agricultural_wage = total_wage / total_days if total_days > 0 else self.base_agricultural_wage
        
        return self.employment_records
    
    def settle_labor_payments(self):
        """
        结算劳动力报酬
        
        将工资从雇主转移到劳动者
        """
        for record in self.employment_records:
            employer = record.employer
            worker = record.worker
            wage = record.total_wage
            
            # 雇主支付工资（增加劳动力成本）
            if hasattr(employer, 'hired_labor_cost'):
                employer.hired_labor_cost += wage
            else:
                employer.hired_labor_cost = wage
            
            # 劳动者获得收入
            if hasattr(worker, 'labor_sale_income'):
                worker.labor_sale_income += wage
            else:
                worker.labor_sale_income = wage
    
    def update_market(self, farmers: List['FarmerAgent'], 
                     enterprises: List['EnterpriseAgent']):
        """
        更新劳动力市场（每期调用）
        
        @param farmers: 农户列表
        @param enterprises: 企业列表
        """
        # 1. 重置状态
        self.employment_records = []
        self.labor_supply_pool = {}
        self.labor_demand_pool = {}
        
        # 重置主体的雇佣相关属性
        for farmer in farmers:
            farmer.hired_labor_cost = 0
            farmer.labor_sale_income = 0
            farmer.hired_labor_days = 0  # 雇佣的劳动天数
        
        for enterprise in enterprises:
            enterprise.hired_labor_cost = 0
            enterprise.hired_labor_days = 0
        
        # 2. 收集供需
        self.collect_labor_supply(farmers)
        self.collect_labor_demand(farmers, enterprises)
        
        # 3. 市场匹配
        self.match_labor_market()
        
        # 4. 结算支付
        self.settle_labor_payments()
        
        # 5. 更新主体的雇佣天数
        for record in self.employment_records:
            if hasattr(record.employer, 'hired_labor_days'):
                record.employer.hired_labor_days += record.labor_days
            if hasattr(record.worker, 'labor_sold_days'):
                record.worker.labor_sold_days = getattr(record.worker, 'labor_sold_days', 0) + record.labor_days
        
        # 6. 记录市场历史
        self._record_market_state()
    
    def _record_market_state(self):
        """记录市场状态"""
        total_supply = sum(v['available_for_sale'] for v in self.labor_supply_pool.values())
        total_demand = sum(
            v.get('hired_labor_need', 0) for v in self.labor_demand_pool.values()
        )
        total_hired = sum(r.labor_days for r in self.employment_records)
        avg_wage = self.current_agricultural_wage
        
        self.market_history.append({
            'step': self.model.schedule.steps if hasattr(self.model, 'schedule') else 0,
            'season': self.get_current_season(),
            'total_supply': total_supply,
            'total_demand': total_demand,
            'total_hired': total_hired,
            'average_wage': avg_wage,
            'num_employers': len(self.labor_demand_pool),
            'num_workers': len(self.labor_supply_pool),
            'num_matches': len(self.employment_records)
        })
    
    def get_labor_cost_for_farmer(self, farmer: 'FarmerAgent') -> Dict:
        """
        获取农户的劳动力成本结构
        
        @param farmer: 农户
        @return: 成本结构字典
        """
        supply_info = self.calculate_farmer_labor_supply(farmer)
        demand_info = self.calculate_large_farmer_labor_demand(farmer)
        
        # 家庭劳动力价值（机会成本）
        family_labor_used = min(supply_info['self_labor_need'], supply_info['total_labor'])
        family_labor_value = family_labor_used * self.current_agricultural_wage * 0.5  # 机会成本取一半
        
        # 雇佣劳动力成本
        hired_cost = getattr(farmer, 'hired_labor_cost', 0)
        
        return {
            'family_labor_days': family_labor_used,
            'family_labor_value': family_labor_value,  # 隐性成本
            'hired_labor_days': demand_info['hired_labor_need'],
            'hired_labor_cost': hired_cost,  # 显性成本
            'total_labor_cost': family_labor_value + hired_cost
        }
    
    def get_labor_cost_for_enterprise(self, enterprise: 'EnterpriseAgent') -> Dict:
        """
        获取企业的劳动力成本结构
        
        @param enterprise: 企业
        @return: 成本结构字典
        """
        demand_info = self.calculate_enterprise_labor_demand(enterprise)
        
        # 企业全部使用雇佣劳动力
        hired_cost = getattr(enterprise, 'hired_labor_cost', 0)
        hired_days = demand_info['hired_labor_need']
        
        # 机械成本（替代劳动力）
        machinery_saved_days = demand_info['machinery_labor']
        machinery_cost_per_day = self.config.get('machinery_cost_per_day', 80)  # 机械替代成本
        machinery_cost = machinery_saved_days * machinery_cost_per_day
        
        return {
            'hired_labor_days': hired_days,
            'hired_labor_cost': hired_cost,
            'machinery_saved_days': machinery_saved_days,
            'machinery_cost': machinery_cost,
            'total_labor_cost': hired_cost + machinery_cost
        }
    
    def check_labor_constraint(self, farmer: 'FarmerAgent', planned_area: float, 
                               crop_type: str) -> Tuple[bool, float]:
        """
        检查农户是否满足劳动力约束
        
        @param farmer: 农户
        @param planned_area: 计划种植面积
        @param crop_type: 作物类型
        @return: (是否满足约束, 实际可种植面积)
        """
        total_labor = getattr(farmer, 'total_labor_units', 365)
        
        # 计算劳动力需求
        labor_need = self.calculate_labor_requirement(planned_area, crop_type, use_machinery=False)
        
        # 考虑可雇佣的劳动力
        demand_info = self.calculate_large_farmer_labor_demand(farmer)
        hireable = 0
        if demand_info['is_large_farmer']:
            # 估计可雇佣的劳动力（基于市场供给）
            hireable = min(
                demand_info['hired_labor_need'],
                sum(v['available_for_sale'] for v in self.labor_supply_pool.values()) * 0.2  # 最多占市场20%
            )
        
        available_labor = total_labor * 0.9 + hireable  # 家庭劳动力 + 可雇佣劳动力
        
        if labor_need <= available_labor:
            return True, planned_area
        else:
            # 计算实际可种植面积
            req = self.labor_requirements.get(crop_type, self.labor_requirements['grain'])
            labor_per_mu = req['base']
            max_area = available_labor / labor_per_mu
            return False, max_area
    
    def get_market_summary(self) -> Dict:
        """
        获取劳动力市场摘要
        
        @return: 摘要字典
        """
        return {
            'current_agricultural_wage': self.current_agricultural_wage,
            'current_season': self.get_current_season(),
            'seasonal_factor': self.get_seasonal_wage_factor(),
            'total_supply_pool': len(self.labor_supply_pool),
            'total_demand_pool': len(self.labor_demand_pool),
            'total_employment_records': len(self.employment_records),
            'total_hired_days': sum(r.labor_days for r in self.employment_records),
            'total_wage_paid': sum(r.total_wage for r in self.employment_records)
        }
