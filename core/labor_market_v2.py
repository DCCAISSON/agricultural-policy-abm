# -*- coding: utf-8 -*-
"""
劳动力市场模块 v2.0
@module: core/labor_market_v2.py
@author: Agricultural ABM Team
@date: 2025-12-22
@priority: 0 - 核心逻辑漏洞修复

@critical_fix:
    解决"要素市场孤岛效应"问题:
    1. 在Agent步进逻辑中显式传递service_substitution参数
    2. 实现 effective_labor_demand = physical_labor_need * (1 - service_substitution_rate)
    3. 让劳动力需求充分考虑服务市场的替代效应

@theory:
    实现农村劳动力市场的完整机制,与服务市场联动：
    1. 农户劳动力供给：自用 + 出售给其他农户/企业 + 外出务工
    2. 企业劳动力需求：雇佣农户劳动力 + 使用机械替代
    3. 服务市场替代：购买托管服务后,劳动力需求下降
    4. 劳动力市场出清：匹配供需,确定工资

@reference:
    - 黄宗智《中国的隐性农业革命》关于农业劳动力市场
    - Deininger & Jin (2005) Land rental markets
    - CLES数据：农业日工资约80-120元/天
"""

import numpy as np
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass, field
from core.config_utils import (
    DEFAULT_BASE_NON_FARM_WAGE,
    resolve_base_non_farm_wage,
    sync_base_non_farm_wage,
)

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import EnterpriseAgent


class LaborType(Enum):
    """劳动力类型"""
    FAMILY_LABOR = "family"          # 家庭劳动力（自用）
    HIRED_LABOR = "hired"            # 雇佣劳动力
    SEASONAL_WORKER = "seasonal"     # 季节工
    MACHINERY = "machinery"          # 机械替代


@dataclass
class ServiceSubstitutionInfo:
    """
    @new 2025-12-22: 服务替代信息
    用于传递服务市场对劳动力市场的替代效应
    """
    farmer_id: str
    service_coverage_ratio: float = 0.0  # 服务覆盖比例 [0,1]
    service_type: str = "none"  # none/single/partial/full
    labor_saved_days: float = 0.0  # 节省的劳动力天数
    
    @property
    def substitution_rate(self) -> float:
        """计算劳动力替代率"""
        # 全托管替代80%劳动力,半托管50%,单项服务20%
        if self.service_type == "full":
            return 0.8 * self.service_coverage_ratio
        elif self.service_type == "partial":
            return 0.5 * self.service_coverage_ratio
        elif self.service_type == "single":
            return 0.2 * self.service_coverage_ratio
        return 0.0


@dataclass
class EmploymentRecord:
    """雇佣记录"""
    employer: Any          # 雇主（农户或企业）
    worker: Any            # 劳动者（农户）
    labor_days: float      # 雇佣天数
    wage_rate: float       # 日工资率
    season: str = "annual" # 季节（spring/summer/autumn/winter/annual）
    labor_type: LaborType = LaborType.HIRED_LABOR
    
    @property
    def total_wage(self) -> float:
        """总工资"""
        return self.labor_days * self.wage_rate


class LaborMarketV2:
    """
    农村劳动力市场 v2.0
    
    @critical_fix 2025-12-22:
    - 新增服务市场替代效应的显式处理
    - 实现 effective_labor_demand = physical_labor_need * (1 - service_substitution_rate)
    
    @theory:
    实现三层次劳动力配置：
    1. 家庭内部配置：农业劳动 vs 非农务工
    2. 本地农业劳动力市场：大户/企业雇佣散户
    3. 城市劳动力市场：农户外出务工
    
    劳动力市场特点：
    - 季节性波动（农忙期需求高）
    - 信息不对称（熟人网络匹配）
    - 工资刚性（存在最低工资）
    - 服务替代效应（托管服务减少劳动力需求）
    """
    
    def __init__(self, model, config: dict):
        """
        初始化劳动力市场
        
        @param model: GrainSubsidyModel实例
        @param config: 配置字典
        """
        self.model = model
        sync_base_non_farm_wage(config, default=DEFAULT_BASE_NON_FARM_WAGE)
        self.config = config.get('labor_market', {})
        
        # 市场参数
        self.base_agricultural_wage = self.config.get('base_agricultural_wage', 100.0)
        self.base_non_farm_wage = resolve_base_non_farm_wage(
            config, default=DEFAULT_BASE_NON_FARM_WAGE
        )
        self.wage_floor = self.config.get('wage_floor', 80.0)
        self.wage_ceiling = self.config.get('wage_ceiling', 300.0)
        
        # 季节性参数
        self.seasonal_factors = self.config.get('seasonal_factors', {
            'spring': 1.3,
            'summer': 1.0,
            'autumn': 1.5,
            'winter': 0.6
        })
        
        # 机械化参数
        self.machinery_substitution = self.config.get('machinery_substitution', {
            'grain': 0.6,
            'cash_crop': 0.3
        })
        
        # 劳动力需求参数（天/亩）
        self.labor_requirements = self.config.get('labor_requirements', {
            'grain': {'base': 8, 'with_machinery': 3},
            'cash_crop': {'base': 20, 'with_machinery': 14}
        })
        
        # @new 2025-12-22: 服务替代信息缓存
        self._service_substitution_cache: Dict[str, ServiceSubstitutionInfo] = {}
        
        # 当前市场状态
        self.current_agricultural_wage = self.base_agricultural_wage
        self.current_season = 'annual'
        
        # 雇佣记录
        self.employment_records: List[EmploymentRecord] = []
        self.labor_supply_pool: Dict = {}
        self.labor_demand_pool: Dict = {}
        
        # 历史记录
        self.market_history = []
    
    # ==================== 服务市场联动接口 ====================
    
    def register_service_substitution(self, farmer_id: str, info: ServiceSubstitutionInfo):
        """
        @new 2025-12-22: 注册服务替代信息
        
        由服务市场调用,将农户的服务购买情况传递给劳动力市场
        """
        self._service_substitution_cache[farmer_id] = info
    
    def clear_service_substitution_cache(self):
        """
        @new 2025-12-22: 清空服务替代缓存（每步开始时调用）
        """
        self._service_substitution_cache.clear()
    
    def get_service_substitution_rate(self, farmer_id: str) -> float:
        """
        @new 2025-12-22: 获取农户的服务替代率
        
        Returns: [0,1] 服务替代劳动力的比例
        """
        if farmer_id in self._service_substitution_cache:
            return self._service_substitution_cache[farmer_id].substitution_rate
        return 0.0
    
    # ==================== 核心计算方法 ====================
    
    def get_current_season(self) -> str:
        """获取当前季节"""
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
        """获取季节性工资调整因子"""
        season = self.get_current_season()
        return self.seasonal_factors.get(season, 1.0)
    
    def calculate_physical_labor_requirement(self, area: float, crop_type: str, 
                                             use_machinery: bool = False) -> float:
        """
        计算物理劳动力需求（不考虑服务替代）
        
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
        
        return base_days * area
    
    def calculate_effective_labor_demand(self, farmer_id: str, physical_labor_need: float) -> float:
        """
        @critical_fix 2025-12-22: 计算有效劳动力需求（考虑服务替代）
        
        核心公式: effective_labor_demand = physical_labor_need * (1 - service_substitution_rate)
        
        @param farmer_id: 农户ID
        @param physical_labor_need: 物理劳动力需求（天）
        @return: 有效劳动力需求（天）
        """
        substitution_rate = self.get_service_substitution_rate(farmer_id)
        effective_demand = physical_labor_need * (1 - substitution_rate)
        return max(0, effective_demand)
    
    def calculate_farmer_labor_supply(self, farmer: 'FarmerAgent') -> Dict:
        """
        计算农户的劳动力供给
        
        @param farmer: 农户Agent
        @return: 供给信息字典
        """
        total_labor = getattr(farmer, 'total_labor_units', 365)
        
        # 计算自用劳动力需求
        self_operated_area = farmer.get_operated_land_area()
        planned_crop = getattr(farmer, 'planned_crop', None)
        
        labor_gap = 0.0
        service_substitution_rate = self.get_service_substitution_rate(farmer.unique_id)
        if planned_crop and self_operated_area > 0:
            labor_profile = None
            if hasattr(farmer, 'estimate_production_labor_profile'):
                try:
                    labor_profile = farmer.estimate_production_labor_profile(
                        crop_type=planned_crop,
                        area=self_operated_area,
                    )
                except Exception:
                    labor_profile = None

            if labor_profile is not None:
                self_labor_need = max(0.0, float(labor_profile.get('total_labor_need', 0.0) or 0.0))
                labor_gap = max(0.0, float(labor_profile.get('labor_gap', 0.0) or 0.0))
                service_substitution_rate = max(
                    service_substitution_rate,
                    float(labor_profile.get('service_substitution_rate', 0.0) or 0.0),
                )
            else:
                from agents.farmer_agent import CropType
                crop_type_str = 'grain' if planned_crop == CropType.GRAIN else 'cash_crop'
                physical_need = self.calculate_physical_labor_requirement(
                    self_operated_area, crop_type_str, use_machinery=False
                )
                # @critical_fix 2025-12-22: 应用服务替代效应
                self_labor_need = self.calculate_effective_labor_demand(
                    farmer.unique_id, physical_need
                )
                labor_gap = max(0.0, self_labor_need - total_labor * 0.9)
        else:
            self_labor_need = 0.0
        
        # 可供出售的劳动力
        available_for_sale = max(0, total_labor - self_labor_need)
        planned_market_work_days = max(0.0, float(getattr(farmer, 'off_farm_labor', 0.0) or 0.0))
        
        # 根据生计策略调整
        from agents.farmer_agent import LivelihoodStrategy
        strategy = getattr(farmer, 'livelihood_strategy', LivelihoodStrategy.PLURIACTIVE)
        
        if strategy == LivelihoodStrategy.PURE_FARMER:
            max_sell_ratio = 0.3
        elif strategy == LivelihoodStrategy.PLURIACTIVE:
            max_sell_ratio = 0.6
        else:
            max_sell_ratio = 0.9
        
        available_for_sale = min(
            available_for_sale,
            total_labor * max_sell_ratio,
            planned_market_work_days
        )
        
        # 期望工资
        wage_level = getattr(farmer, 'off_farm_wage_level', 1.0)
        min_acceptable_wage = self.base_agricultural_wage * wage_level * 0.8
        
        return {
            'farmer': farmer,
            'total_labor': total_labor,
            'self_labor_need': self_labor_need,
            'labor_gap': labor_gap,
            'available_for_sale': available_for_sale,
            'min_acceptable_wage': min_acceptable_wage,
            'strategy': strategy,
            'service_substitution_rate': service_substitution_rate
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
        
        from agents.farmer_agent import CropType
        crop_portfolio = getattr(enterprise, 'crop_portfolio', 
                                 {CropType.GRAIN: 0.5, CropType.CASH_CROP: 0.5})
        
        grain_area = operated_area * crop_portfolio.get(CropType.GRAIN, 0.5)
        cash_crop_area = operated_area * crop_portfolio.get(CropType.CASH_CROP, 0.5)
        
        # 企业使用大型机械
        grain_labor = self.calculate_physical_labor_requirement(grain_area, 'grain', use_machinery=True)
        cash_crop_labor = self.calculate_physical_labor_requirement(cash_crop_area, 'cash_crop', use_machinery=True)
        
        total_demand = grain_labor + cash_crop_labor
        
        # 机械替代节省的劳动力
        full_grain_labor = self.calculate_physical_labor_requirement(grain_area, 'grain', use_machinery=False)
        full_cash_crop_labor = self.calculate_physical_labor_requirement(cash_crop_area, 'cash_crop', use_machinery=False)
        machinery_saved = (full_grain_labor + full_cash_crop_labor) - total_demand
        
        efficiency_factor = getattr(enterprise, 'efficiency_factor', 1.0)
        max_wage = self.base_agricultural_wage * efficiency_factor * 1.2
        
        return {
            'enterprise': enterprise,
            'total_demand': total_demand,
            'machinery_labor': machinery_saved,
            'hired_labor_need': total_demand,
            'max_wage': max_wage,
            'grain_area': grain_area,
            'cash_crop_area': cash_crop_area
        }
    
    def update_market(self, farmers: List['FarmerAgent'], 
                     enterprises: List['EnterpriseAgent']):
        """
        更新劳动力市场（市场出清）
        
        @critical_fix 2025-12-22: 
        在市场出清前,应先通过register_service_substitution()
        接收服务市场的替代信息
        """
        self._reset_market_accounts(farmers, enterprises)

        # 收集供给
        self.labor_supply_pool.clear()
        for farmer in farmers:
            supply = self.calculate_farmer_labor_supply(farmer)
            if supply['available_for_sale'] > 0:
                self.labor_supply_pool[farmer.unique_id] = supply
        
        # 收集需求
        self.labor_demand_pool.clear()
        for enterprise in enterprises:
            demand = self.calculate_enterprise_labor_demand(enterprise)
            if demand['hired_labor_need'] > 0:
                self.labor_demand_pool[enterprise.unique_id] = demand
        
        # 大户需求
        for farmer in farmers:
            if farmer.get_operated_land_area() > 30:
                demand = self._calculate_large_farmer_demand(farmer)
                if demand and demand.get('labor_gap', 0) > 0:
                    self.labor_demand_pool[f"farmer_{farmer.unique_id}"] = demand
        
        # 市场出清
        self._clear_market()
        self._settle_employment_records()
        
        # 更新工资
        self._update_equilibrium_wage()
        
        # 记录历史
        self._record_market_state()
    
    def _calculate_large_farmer_demand(self, farmer: 'FarmerAgent') -> Optional[Dict]:
        """计算大户劳动力需求"""
        operated_area = farmer.get_operated_land_area()
        supply = self.calculate_farmer_labor_supply(farmer)
        labor_gap = max(0.0, float(supply.get('labor_gap', 0.0) or 0.0))
        
        if operated_area > 30 and labor_gap > 0:
            farming_ability = getattr(farmer, 'farming_ability', 1.0)
            max_wage = self.base_agricultural_wage * (0.9 + 0.2 * farming_ability)
            
            return {
                'farmer': farmer,
                'labor_gap': labor_gap,
                'max_wage': max_wage,
                'is_large_farmer': True
            }
        return None
    
    def _clear_market(self):
        """市场出清"""
        self.employment_records.clear()
        
        sorted_demand = sorted(
            self.labor_demand_pool.items(),
            key=lambda x: x[1].get('max_wage', 0),
            reverse=True
        )
        
        available_supply = {k: v['available_for_sale'] for k, v in self.labor_supply_pool.items()}
        
        for demander_id, demand in sorted_demand:
            needed = demand.get('hired_labor_need', demand.get('labor_gap', 0))
            max_wage = demand.get('max_wage', self.base_agricultural_wage)
            employer = demand.get('enterprise', demand.get('farmer'))
            
            if needed <= 0 or employer is None:
                continue
            
            sorted_suppliers = sorted(
                [(sid, self.labor_supply_pool[sid]) for sid in available_supply if available_supply[sid] > 0],
                key=lambda x: x[1]['min_acceptable_wage']
            )
            
            for supplier_id, supply in sorted_suppliers:
                if available_supply[supplier_id] <= 0:
                    continue
                if supply['min_acceptable_wage'] > max_wage:
                    continue
                if supply.get('farmer') is employer:
                    continue
                
                hired = min(needed, available_supply[supplier_id])
                wage = (supply['min_acceptable_wage'] + max_wage) / 2
                
                record = EmploymentRecord(
                    employer=employer,
                    worker=supply['farmer'],
                    labor_days=hired,
                    wage_rate=wage,
                    season=self.get_current_season(),
                    labor_type=LaborType.HIRED_LABOR
                )
                self.employment_records.append(record)
                
                available_supply[supplier_id] -= hired
                needed -= hired
                
                if needed <= 0:
                    break

    def _reset_market_accounts(self, farmers: List['FarmerAgent'],
                               enterprises: List['EnterpriseAgent']):
        """Reset annual labor-market accounting before a fresh clearing round."""
        for farmer in farmers:
            farmer.hired_labor_cost = 0.0
            farmer.labor_sale_income = 0.0
            farmer.hired_labor_days = 0.0
            farmer.requested_hired_labor_days = 0.0
            farmer.labor_sold_days = 0.0

        for enterprise in enterprises:
            if hasattr(enterprise, 'hired_labor_cost'):
                enterprise.hired_labor_cost = 0.0
            if hasattr(enterprise, 'hired_labor_days'):
                enterprise.hired_labor_days = 0.0

    def _settle_employment_records(self):
        """Push labor-market transactions into worker income and employer cost accounts."""
        for record in self.employment_records:
            total_wage = float(record.total_wage)
            labor_days = float(record.labor_days)
            worker = record.worker
            employer = record.employer

            if worker is not None:
                worker.labor_sale_income = float(getattr(worker, 'labor_sale_income', 0.0) or 0.0) + total_wage
                worker.labor_sold_days = float(getattr(worker, 'labor_sold_days', 0.0) or 0.0) + labor_days

            if employer is not None and hasattr(employer, 'hired_labor_cost'):
                employer.hired_labor_cost = float(getattr(employer, 'hired_labor_cost', 0.0) or 0.0) + total_wage
            if employer is not None and hasattr(employer, 'hired_labor_days'):
                employer.hired_labor_days = float(getattr(employer, 'hired_labor_days', 0.0) or 0.0) + labor_days
    
    def _update_equilibrium_wage(self):
        """更新均衡工资"""
        if self.employment_records:
            total_labor = sum(r.labor_days for r in self.employment_records)
            weighted_wage = sum(r.labor_days * r.wage_rate for r in self.employment_records)
            self.current_agricultural_wage = weighted_wage / total_labor if total_labor > 0 else self.base_agricultural_wage
        
        self.current_agricultural_wage = max(self.wage_floor, 
                                            min(self.wage_ceiling, self.current_agricultural_wage))
    
    def _record_market_state(self):
        """记录市场状态"""
        total_hired = sum(r.labor_days for r in self.employment_records)
        total_supply = sum(v['available_for_sale'] for v in self.labor_supply_pool.values())
        total_demand = sum(v.get('hired_labor_need', v.get('labor_gap', 0)) 
                          for v in self.labor_demand_pool.values())
        
        # @new 2025-12-22: 记录服务替代统计
        service_affected_farmers = len(self._service_substitution_cache)
        avg_substitution = np.mean([info.substitution_rate for info in self._service_substitution_cache.values()]) \
                          if self._service_substitution_cache else 0
        
        self.market_history.append({
            'step': getattr(self.model, 'step_count', 0),
            'season': self.current_season,
            'wage': self.current_agricultural_wage,
            'total_hired': total_hired,
            'total_supply': total_supply,
            'total_demand': total_demand,
            'n_transactions': len(self.employment_records),
            'service_affected_farmers': service_affected_farmers,
            'avg_service_substitution': avg_substitution
        })
    
    def get_market_summary(self) -> Dict:
        """获取市场摘要"""
        return {
            'current_wage': self.current_agricultural_wage,
            'season': self.current_season,
            'n_suppliers': len(self.labor_supply_pool),
            'n_demanders': len(self.labor_demand_pool),
            'n_employment_records': len(self.employment_records),
            'total_labor_traded': sum(r.labor_days for r in self.employment_records),
            'service_substitution_enabled': len(self._service_substitution_cache) > 0
        }


# 兼容性别名
LaborMarket = LaborMarketV2
