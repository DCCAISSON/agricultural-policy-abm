# -*- coding: utf-8 -*-
"""
经济闭环守恒断言模块
@module: core/conservation_invariants.py
@author: Agricultural ABM Team
@date: 2026-01-04

@purpose:
实现"会计恒等式"检验，确保模型经济逻辑闭环：
1. 土地守恒：初始+转入-转出-转化=期末
2. 预算约束：收入+补贴-成本-租金-投资=期末现金变化
3. 财政守恒：期初+收入-支出=期末
4. 市场清算：交易量平衡

@theory:
模型闭环要靠硬约束验证，而不是"能跑就行"
参考：Godley & Lavoie (2007) "Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth"

@critical:
- 任何守恒失败应直接抛错，而不是静默填默认值
- 输出年度会计平衡表，差额必须接近0
- 若不平衡，必须能定位是哪个模块造成
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import warnings


class ConservationViolation(Exception):
    """守恒约束违反异常"""
    def __init__(self, constraint_name: str, expected: float, actual: float, 
                 details: Optional[Dict] = None):
        self.constraint_name = constraint_name
        self.expected = expected
        self.actual = actual
        self.details = details or {}
        self.discrepancy = actual - expected
        
        message = (
            f"守恒约束违反: {constraint_name}\n"
            f"  期望值: {expected:.4f}\n"
            f"  实际值: {actual:.4f}\n"
            f"  差异: {self.discrepancy:.4f}\n"
        )
        if details:
            message += f"  详情: {json.dumps(details, indent=2, ensure_ascii=False)}"
        
        super().__init__(message)


class InvariantType(Enum):
    """不变量类型"""
    LAND_CONSERVATION = "土地守恒"
    BUDGET_BALANCE = "预算平衡"
    FISCAL_BALANCE = "财政平衡"
    MARKET_CLEARING = "市场清算"
    POPULATION_CONSERVATION = "主体守恒"
    NON_NEGATIVE = "非负约束"
    PRICE_BOUNDS = "价格区间"


@dataclass
class InvariantCheckResult:
    """不变量检查结果"""
    invariant_type: InvariantType
    passed: bool
    expected: float
    actual: float
    tolerance: float
    discrepancy: float
    details: Dict = field(default_factory=dict)
    
    @property
    def relative_error(self) -> float:
        if abs(self.expected) < 1e-10:
            return abs(self.discrepancy)
        return abs(self.discrepancy / self.expected)


@dataclass
class AnnualBalanceSheet:
    """
    年度会计平衡表
    
    记录所有守恒量的收支情况
    """
    year: int
    
    # 土地账户 (亩)
    land_initial: float = 0.0
    land_transfer_in: float = 0.0
    land_transfer_out: float = 0.0
    land_reclaimed: float = 0.0      # 开垦
    land_abandoned: float = 0.0       # 撂荒/退耕
    land_final: float = 0.0
    land_discrepancy: float = 0.0
    
    # 农户现金账户 (元) - 总计
    farmer_cash_initial: float = 0.0
    farmer_income_agricultural: float = 0.0
    farmer_income_off_farm: float = 0.0
    farmer_income_rental: float = 0.0   # 土地租金收入
    farmer_subsidy_received: float = 0.0
    farmer_cost_production: float = 0.0
    farmer_cost_rental: float = 0.0     # 租金支出
    farmer_investment: float = 0.0
    farmer_consumption: float = 0.0
    farmer_cash_final: float = 0.0
    farmer_cash_discrepancy: float = 0.0
    
    # 企业现金账户 (元) - 总计
    enterprise_cash_initial: float = 0.0
    enterprise_revenue: float = 0.0
    enterprise_subsidy: float = 0.0
    enterprise_cost_labor: float = 0.0
    enterprise_cost_rental: float = 0.0
    enterprise_cost_input: float = 0.0
    enterprise_investment: float = 0.0
    enterprise_cash_final: float = 0.0
    enterprise_cash_discrepancy: float = 0.0
    
    # 政府财政账户 (元)
    fiscal_initial: float = 0.0
    fiscal_income: float = 0.0          # 税收等
    fiscal_subsidy_expenditure: float = 0.0
    fiscal_other_expenditure: float = 0.0
    fiscal_final: float = 0.0
    fiscal_discrepancy: float = 0.0
    
    # 市场清算 (交易量)
    land_market_supply: float = 0.0
    land_market_demand: float = 0.0
    land_market_traded: float = 0.0
    labor_market_supply: float = 0.0
    labor_market_demand: float = 0.0
    labor_market_traded: float = 0.0
    
    def compute_discrepancies(self):
        """计算各账户的差额"""
        # 土地守恒
        self.land_discrepancy = (
            self.land_initial + self.land_transfer_in + self.land_reclaimed
            - self.land_transfer_out - self.land_abandoned
            - self.land_final
        )
        
        # 农户现金守恒
        farmer_inflow = (
            self.farmer_income_agricultural +
            self.farmer_income_off_farm +
            self.farmer_income_rental +
            self.farmer_subsidy_received
        )
        farmer_outflow = (
            self.farmer_cost_production +
            self.farmer_cost_rental +
            self.farmer_investment +
            self.farmer_consumption
        )
        self.farmer_cash_discrepancy = (
            self.farmer_cash_initial + farmer_inflow - farmer_outflow
            - self.farmer_cash_final
        )
        
        # 企业现金守恒
        enterprise_inflow = self.enterprise_revenue + self.enterprise_subsidy
        enterprise_outflow = (
            self.enterprise_cost_labor +
            self.enterprise_cost_rental +
            self.enterprise_cost_input +
            self.enterprise_investment
        )
        self.enterprise_cash_discrepancy = (
            self.enterprise_cash_initial + enterprise_inflow - enterprise_outflow
            - self.enterprise_cash_final
        )
        
        # 财政守恒
        self.fiscal_discrepancy = (
            self.fiscal_initial + self.fiscal_income
            - self.fiscal_subsidy_expenditure - self.fiscal_other_expenditure
            - self.fiscal_final
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConservationInvariantChecker:
    """
    守恒不变量检查器
    
    在每步结束时检查所有守恒约束
    """
    
    # 默认容差
    DEFAULT_TOLERANCE = {
        InvariantType.LAND_CONSERVATION: 0.01,      # 土地：0.01亩
        InvariantType.BUDGET_BALANCE: 1.0,          # 现金：1元
        InvariantType.FISCAL_BALANCE: 1.0,          # 财政：1元
        InvariantType.MARKET_CLEARING: 0.1,         # 交易量：0.1
        InvariantType.POPULATION_CONSERVATION: 0,    # 主体数：精确
        InvariantType.NON_NEGATIVE: 0,              # 非负：精确
        InvariantType.PRICE_BOUNDS: 0               # 价格：精确
    }
    
    def __init__(self, model, strict_mode: bool = True):
        """
        Args:
            model: 模型实例
            strict_mode: 严格模式下，违反约束直接抛错
        """
        self.model = model
        self.strict_mode = strict_mode
        self.tolerances = self.DEFAULT_TOLERANCE.copy()
        
        # 历史记录
        self.balance_sheets: List[AnnualBalanceSheet] = []
        self.check_results: List[List[InvariantCheckResult]] = []
        self.violations: List[ConservationViolation] = []
    
    def set_tolerance(self, inv_type: InvariantType, value: float):
        """设置特定不变量的容差"""
        self.tolerances[inv_type] = value
    
    def check_all(self, year: int) -> List[InvariantCheckResult]:
        """
        执行所有守恒检查
        
        Args:
            year: 当前年份
            
        Returns:
            检查结果列表
        """
        results = []
        
        # 收集平衡表数据
        balance = self._collect_balance_sheet(year)
        balance.compute_discrepancies()
        self.balance_sheets.append(balance)
        
        # 1. 土地守恒检查
        results.append(self._check_land_conservation(balance))
        
        # 2. 农户预算平衡
        results.append(self._check_farmer_budget(balance))
        
        # 3. 企业预算平衡
        results.append(self._check_enterprise_budget(balance))
        
        # 4. 财政平衡
        results.append(self._check_fiscal_balance(balance))
        
        # 5. 非负约束
        results.extend(self._check_non_negative())
        
        # 6. 价格区间
        results.extend(self._check_price_bounds())
        
        # 7. 主体守恒
        results.append(self._check_population_conservation())
        
        self.check_results.append(results)
        
        # 处理违反
        for r in results:
            if not r.passed:
                violation = ConservationViolation(
                    constraint_name=r.invariant_type.value,
                    expected=r.expected,
                    actual=r.actual,
                    details=r.details
                )
                self.violations.append(violation)
                
                if self.strict_mode:
                    raise violation
                else:
                    warnings.warn(str(violation))
        
        return results
    
    def _collect_balance_sheet(self, year: int) -> AnnualBalanceSheet:
        """从模型收集平衡表数据"""
        balance = AnnualBalanceSheet(year=year)
        
        model = self.model
        
        # 土地账户
        if hasattr(model, 'total_land_area'):
            balance.land_initial = getattr(model, '_prev_total_land', model.total_land_area)
            balance.land_final = model.total_land_area
        
        # 土地流转统计
        if hasattr(model, 'land_rental_market') and model.land_rental_market:
            lrm = model.land_rental_market
            balance.land_transfer_in = getattr(lrm, 'total_rented_in', 0)
            balance.land_transfer_out = getattr(lrm, 'total_rented_out', 0)
            balance.land_market_supply = getattr(lrm, 'current_supply', 0)
            balance.land_market_demand = getattr(lrm, 'current_demand', 0)
            balance.land_market_traded = getattr(lrm, 'current_traded', 0)
        
        # 撂荒
        if hasattr(model, 'total_abandoned_land'):
            balance.land_abandoned = model.total_abandoned_land
        
        # 农户现金账户 (汇总)
        if hasattr(model, 'farmers'):
            for farmer in model.farmers:
                balance.farmer_cash_final += getattr(farmer, 'capital', 0)
                balance.farmer_income_agricultural += getattr(farmer, 'agricultural_income', 0)
                balance.farmer_income_off_farm += getattr(farmer, 'off_farm_income', 0)
                balance.farmer_income_rental += getattr(farmer, 'land_rental_income', 0)
                balance.farmer_subsidy_received += getattr(farmer, 'subsidy_received', 0)
                balance.farmer_cost_production += getattr(farmer, 'production_cost', 0)
                balance.farmer_cost_rental += getattr(farmer, 'rental_cost_paid', 0)
        
        # 获取上一步的初始值
        if hasattr(model, '_prev_farmer_capital'):
            balance.farmer_cash_initial = model._prev_farmer_capital
        
        # 企业现金账户
        if hasattr(model, 'agricultural_enterprises'):
            for ent in model.agricultural_enterprises:
                balance.enterprise_cash_final += getattr(ent, 'capital', 0)
                balance.enterprise_revenue += getattr(ent, 'annual_revenue', 0)
                balance.enterprise_subsidy += getattr(ent, 'subsidy_received', 0)
                balance.enterprise_cost_labor += getattr(ent, 'labor_cost', 0)
                balance.enterprise_cost_rental += getattr(ent, 'rental_cost', 0)
                balance.enterprise_cost_input += getattr(ent, 'input_cost', 0)
        
        if hasattr(model, '_prev_enterprise_capital'):
            balance.enterprise_cash_initial = model._prev_enterprise_capital
        
        # 政府财政
        if hasattr(model, 'government'):
            gov = model.government
            balance.fiscal_final = getattr(gov, 'fiscal_balance', 0)
            balance.fiscal_subsidy_expenditure = getattr(gov, 'total_subsidy_paid', 0)
            
            if hasattr(model, '_prev_fiscal_balance'):
                balance.fiscal_initial = model._prev_fiscal_balance
        
        # 劳动力市场
        if hasattr(model, 'labor_market') and model.labor_market:
            lm = model.labor_market
            balance.labor_market_supply = sum(
                v.get('available_for_sale', 0) 
                for v in getattr(lm, 'labor_supply_pool', {}).values()
            )
            balance.labor_market_demand = sum(
                v.get('hired_labor_need', v.get('labor_gap', 0))
                for v in getattr(lm, 'labor_demand_pool', {}).values()
            )
            balance.labor_market_traded = sum(
                r.labor_days for r in getattr(lm, 'employment_records', [])
            )
        
        return balance
    
    def _check_land_conservation(self, balance: AnnualBalanceSheet) -> InvariantCheckResult:
        """检查土地守恒"""
        tol = self.tolerances[InvariantType.LAND_CONSERVATION]
        
        expected = 0.0  # 差额应为0
        actual = balance.land_discrepancy
        
        return InvariantCheckResult(
            invariant_type=InvariantType.LAND_CONSERVATION,
            passed=abs(actual) <= tol,
            expected=expected,
            actual=actual,
            tolerance=tol,
            discrepancy=actual,
            details={
                'initial': balance.land_initial,
                'transfer_in': balance.land_transfer_in,
                'transfer_out': balance.land_transfer_out,
                'abandoned': balance.land_abandoned,
                'final': balance.land_final
            }
        )
    
    def _check_farmer_budget(self, balance: AnnualBalanceSheet) -> InvariantCheckResult:
        """检查农户预算平衡"""
        tol = self.tolerances[InvariantType.BUDGET_BALANCE]
        
        expected = 0.0
        actual = balance.farmer_cash_discrepancy
        
        return InvariantCheckResult(
            invariant_type=InvariantType.BUDGET_BALANCE,
            passed=abs(actual) <= tol,
            expected=expected,
            actual=actual,
            tolerance=tol,
            discrepancy=actual,
            details={
                'account': 'farmer',
                'initial': balance.farmer_cash_initial,
                'income_agri': balance.farmer_income_agricultural,
                'income_off_farm': balance.farmer_income_off_farm,
                'income_rental': balance.farmer_income_rental,
                'subsidy': balance.farmer_subsidy_received,
                'cost_prod': balance.farmer_cost_production,
                'cost_rental': balance.farmer_cost_rental,
                'final': balance.farmer_cash_final
            }
        )
    
    def _check_enterprise_budget(self, balance: AnnualBalanceSheet) -> InvariantCheckResult:
        """检查企业预算平衡"""
        tol = self.tolerances[InvariantType.BUDGET_BALANCE]
        
        expected = 0.0
        actual = balance.enterprise_cash_discrepancy
        
        return InvariantCheckResult(
            invariant_type=InvariantType.BUDGET_BALANCE,
            passed=abs(actual) <= tol,
            expected=expected,
            actual=actual,
            tolerance=tol,
            discrepancy=actual,
            details={
                'account': 'enterprise',
                'initial': balance.enterprise_cash_initial,
                'revenue': balance.enterprise_revenue,
                'subsidy': balance.enterprise_subsidy,
                'cost_labor': balance.enterprise_cost_labor,
                'cost_rental': balance.enterprise_cost_rental,
                'cost_input': balance.enterprise_cost_input,
                'final': balance.enterprise_cash_final
            }
        )
    
    def _check_fiscal_balance(self, balance: AnnualBalanceSheet) -> InvariantCheckResult:
        """检查财政平衡"""
        tol = self.tolerances[InvariantType.FISCAL_BALANCE]
        
        expected = 0.0
        actual = balance.fiscal_discrepancy
        
        return InvariantCheckResult(
            invariant_type=InvariantType.FISCAL_BALANCE,
            passed=abs(actual) <= tol,
            expected=expected,
            actual=actual,
            tolerance=tol,
            discrepancy=actual,
            details={
                'initial': balance.fiscal_initial,
                'income': balance.fiscal_income,
                'subsidy_expenditure': balance.fiscal_subsidy_expenditure,
                'other_expenditure': balance.fiscal_other_expenditure,
                'final': balance.fiscal_final
            }
        )
    
    def _check_non_negative(self) -> List[InvariantCheckResult]:
        """检查非负约束"""
        results = []
        model = self.model
        
        checks = []
        
        # 农户资本非负
        if hasattr(model, 'farmers'):
            for i, farmer in enumerate(model.farmers):
                capital = getattr(farmer, 'capital', 0)
                if capital < 0:
                    checks.append((f'farmer_{i}_capital', capital))
        
        # 企业资本非负
        if hasattr(model, 'agricultural_enterprises'):
            for i, ent in enumerate(model.agricultural_enterprises):
                capital = getattr(ent, 'capital', 0)
                if capital < 0:
                    checks.append((f'enterprise_{i}_capital', capital))
        
        # 价格非负
        if hasattr(model, 'environment'):
            env = model.environment
            prices = getattr(env, 'market_prices', {})
            for name, price in prices.items():
                if price < 0:
                    checks.append((f'price_{name}', price))
        
        # 土地面积非负
        if hasattr(model, 'total_land_area'):
            if model.total_land_area < 0:
                checks.append(('total_land_area', model.total_land_area))
        
        for name, value in checks:
            results.append(InvariantCheckResult(
                invariant_type=InvariantType.NON_NEGATIVE,
                passed=False,
                expected=0.0,
                actual=value,
                tolerance=0,
                discrepancy=value,
                details={'variable': name}
            ))
        
        if not checks:
            results.append(InvariantCheckResult(
                invariant_type=InvariantType.NON_NEGATIVE,
                passed=True,
                expected=0.0,
                actual=0.0,
                tolerance=0,
                discrepancy=0.0,
                details={'message': '所有非负约束满足'}
            ))
        
        return results
    
    def _check_price_bounds(self) -> List[InvariantCheckResult]:
        """检查价格区间"""
        results = []
        model = self.model
        
        # 从配置获取价格限制
        config = getattr(model, 'config', {})
        price_limits = config.get('environment', {}).get('market', {}).get('price_limits', {})
        
        if not price_limits:
            price_limits = {
                'grain': {'min': 0.5, 'max': 5.0},
                'cash_crop': {'min': 0.5, 'max': 10.0}
            }
        
        if hasattr(model, 'environment'):
            env = model.environment
            prices = getattr(env, 'market_prices', {})
            
            # 检查粮食价格
            grain_price = prices.get('crop_price', 0)
            grain_limits = price_limits.get('grain', {'min': 0.5, 'max': 5.0})
            if not (grain_limits['min'] <= grain_price <= grain_limits['max']):
                results.append(InvariantCheckResult(
                    invariant_type=InvariantType.PRICE_BOUNDS,
                    passed=False,
                    expected=(grain_limits['min'] + grain_limits['max']) / 2,
                    actual=grain_price,
                    tolerance=0,
                    discrepancy=0,
                    details={
                        'variable': 'grain_price',
                        'min': grain_limits['min'],
                        'max': grain_limits['max']
                    }
                ))
            
            # 检查经作价格
            cash_price = prices.get('cash_crop', 0)
            cash_limits = price_limits.get('cash_crop', {'min': 0.5, 'max': 10.0})
            if not (cash_limits['min'] <= cash_price <= cash_limits['max']):
                results.append(InvariantCheckResult(
                    invariant_type=InvariantType.PRICE_BOUNDS,
                    passed=False,
                    expected=(cash_limits['min'] + cash_limits['max']) / 2,
                    actual=cash_price,
                    tolerance=0,
                    discrepancy=0,
                    details={
                        'variable': 'cash_crop_price',
                        'min': cash_limits['min'],
                        'max': cash_limits['max']
                    }
                ))
        
        if not results:
            results.append(InvariantCheckResult(
                invariant_type=InvariantType.PRICE_BOUNDS,
                passed=True,
                expected=0,
                actual=0,
                tolerance=0,
                discrepancy=0,
                details={'message': '所有价格在合理区间'}
            ))
        
        return results
    
    def _check_population_conservation(self) -> InvariantCheckResult:
        """检查主体数量守恒"""
        model = self.model
        
        expected_farmers = getattr(model, '_initial_n_farmers', 0)
        actual_farmers = len(getattr(model, 'farmers', []))
        
        # 计算退出/新进入数量
        exits = getattr(model, 'farmer_exits', 0)
        entries = getattr(model, 'farmer_entries', 0)
        
        expected_final = expected_farmers - exits + entries
        
        passed = actual_farmers == expected_final
        
        return InvariantCheckResult(
            invariant_type=InvariantType.POPULATION_CONSERVATION,
            passed=passed,
            expected=expected_final,
            actual=actual_farmers,
            tolerance=0,
            discrepancy=actual_farmers - expected_final,
            details={
                'initial': expected_farmers,
                'exits': exits,
                'entries': entries,
                'expected_final': expected_final,
                'actual_final': actual_farmers
            }
        )
    
    def save_state_for_next_step(self):
        """保存当前状态用于下一步的守恒检查"""
        model = self.model
        
        # 保存土地总量
        if hasattr(model, 'total_land_area'):
            model._prev_total_land = model.total_land_area
        
        # 保存农户资本总和
        if hasattr(model, 'farmers'):
            model._prev_farmer_capital = sum(
                getattr(f, 'capital', 0) for f in model.farmers
            )
        
        # 保存企业资本总和
        if hasattr(model, 'agricultural_enterprises'):
            model._prev_enterprise_capital = sum(
                getattr(e, 'capital', 0) for e in model.agricultural_enterprises
            )
        
        # 保存财政余额
        if hasattr(model, 'government'):
            model._prev_fiscal_balance = getattr(model.government, 'fiscal_balance', 0)
        
        # 保存初始农户数
        if not hasattr(model, '_initial_n_farmers'):
            model._initial_n_farmers = len(getattr(model, 'farmers', []))
    
    def get_summary(self) -> Dict:
        """获取检查摘要"""
        total_checks = sum(len(r) for r in self.check_results)
        total_passed = sum(
            sum(1 for c in r if c.passed) 
            for r in self.check_results
        )
        
        return {
            'total_years': len(self.balance_sheets),
            'total_checks': total_checks,
            'total_passed': total_passed,
            'total_violations': len(self.violations),
            'pass_rate': total_passed / total_checks if total_checks > 0 else 1.0,
            'violation_types': [v.constraint_name for v in self.violations]
        }
    
    def export_balance_sheets(self, path: str):
        """导出所有平衡表"""
        import pandas as pd
        
        data = [b.to_dict() for b in self.balance_sheets]
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"[ConservationChecker] 平衡表已导出到 {path}")


# ==================== 快捷函数 ====================

def check_conservation(model, year: int, strict: bool = False) -> List[InvariantCheckResult]:
    """
    快捷函数：执行守恒检查
    
    应在每步结束时调用
    """
    if not hasattr(model, '_conservation_checker'):
        model._conservation_checker = ConservationInvariantChecker(model, strict_mode=strict)
    
    checker = model._conservation_checker
    checker.save_state_for_next_step()
    
    if year > 0:  # 第一年无法检查差额
        return checker.check_all(year)
    
    return []


def get_conservation_summary(model) -> Dict:
    """获取守恒检查摘要"""
    if hasattr(model, '_conservation_checker'):
        return model._conservation_checker.get_summary()
    return {'message': '未启用守恒检查'}


if __name__ == "__main__":
    print("=" * 60)
    print("守恒不变量检查模块测试")
    print("=" * 60)
    
    # 创建模拟模型对象
    class MockModel:
        def __init__(self):
            self.farmers = []
            self.agricultural_enterprises = []
            self.total_land_area = 1000.0
            self.config = {
                'environment': {
                    'market': {
                        'price_limits': {
                            'grain': {'min': 1.0, 'max': 2.0},
                            'cash_crop': {'min': 1.5, 'max': 3.5}
                        }
                    }
                }
            }
            
            class MockEnv:
                market_prices = {'crop_price': 1.25, 'cash_crop': 2.2}
            self.environment = MockEnv()
    
    model = MockModel()
    
    # 测试检查器
    checker = ConservationInvariantChecker(model, strict_mode=False)
    checker.save_state_for_next_step()
    
    # 模拟一步后的变化
    model.total_land_area = 999.5  # 小变化
    
    results = checker.check_all(year=1)
    
    print(f"\n检查结果: {len(results)} 项")
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.invariant_type.value}: 差异={r.discrepancy:.4f}")
    
    print(f"\n摘要: {checker.get_summary()}")
