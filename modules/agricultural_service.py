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
import math
import numpy as np
from utils.ai_runtime import mechanism_enabled, service_coordination_params


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


def _resolve_model_simulation_year(model: Any) -> int:
    """Resolve the best available simulation year without assuming one model shape."""
    if model is None:
        return 0

    explicit_year = getattr(model, "current_simulation_year", None)
    if explicit_year is None:
        explicit_year = getattr(model, "current_year", None)

    if explicit_year is None:
        base_year = getattr(model, "simulation_base_year", None)
        step_count = getattr(model, "step_count", None)
        if base_year is not None and step_count is not None:
            try:
                explicit_year = int(base_year) + int(step_count)
            except (TypeError, ValueError):
                explicit_year = None

    if explicit_year is None:
        population_manager = getattr(model, "population_manager", None)
        if population_manager is not None:
            explicit_year = getattr(population_manager, "current_year", None)

    if explicit_year is None:
        explicit_year = getattr(model, "step_count", 0)

    try:
        return int(explicit_year or 0)
    except (TypeError, ValueError):
        return 0

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
    production_season: str | None = None
    plot_areas_mu: dict[str, float] = field(default_factory=dict)
    plot_allocation_source: str | None = None
    p3_paid_yuan: float = 0.0
    p3_payment_step: int | None = None
    p3_application_step: int | None = None
    p3_fulfilled: bool = False
    
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
        self._provider_economics_config = self._get_provider_economics_config()
        
        # 服务能力
        initial_capacity = self._resolve_initial_capacity(initial_capacity)
        self.initial_capacity = initial_capacity
        self.current_capacity = initial_capacity
        max_capacity_multiplier = self._safe_float(
            self._provider_economics_config.get("max_capacity_multiplier"),
            2.0,
            minimum=1.0,
        )
        self.max_capacity = max(
            initial_capacity,
            self._safe_float(
                self._provider_economics_config.get("max_capacity"),
                initial_capacity * max_capacity_multiplier,
                minimum=initial_capacity,
            ),
        )
        self.max_annual_expansion = self._safe_float(
            self._provider_economics_config.get("max_annual_expansion"),
            0.2,
            minimum=0.0,
        )
        self.expansion_investment_threshold = self._safe_float(
            self._provider_economics_config.get("equipment_investment"),
            50000.0,
            minimum=0.0,
        )
        
        # 机械设备
        self.large_tractors = max(1, int(initial_capacity / 200))
        self.combine_harvesters = max(1, int(initial_capacity / 400))
        self.drone_sprayers = max(1, int(initial_capacity / 300))
        
        # 服务范围
        self.service_radius = service_radius
        if mechanism_enabled(getattr(self.model, "config", {}), "service_coordination"):
            coordination = service_coordination_params(getattr(self.model, "config", {}))
            self.service_radius = int(
                round(self.service_radius * (1.0 + max(0.0, coordination.get("service_radius_boost", 0.0))))
            )
        
        # 当前服务面积和合同
        self.service_area_used = 0.0
        self.active_contracts: List[ServiceContract] = []
        
        # 财务
        self.capital = 100000.0
        self.annual_revenue = 0.0
        self.annual_cost = 0.0
        self.cumulative_profit = 0.0
        self.last_settlement_year: Optional[int] = None
        self.last_settlement_profit = 0.0
        self.last_settlement_account: Optional[Dict[str, Any]] = None
        self.cash_mode = self._provider_economics_config.get("cash_mode", "legacy")
        if self.cash_mode not in ("legacy", "operating_cash"):
            raise ValueError("unknown provider cash_mode")
        if self.cash_mode == "operating_cash":
            from core.p3_project_contracts import configured_program
            if configured_program(model) is None:
                raise ValueError("operating_cash requires configured P3 settlement")
            value = self._provider_economics_config.get("initial_cash_yuan")
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("operating_cash requires explicit finite nonnegative initial_cash_yuan")
            self.cash_balance = float(value)
            self._cash_open = self.cash_balance
            self._cash_receipts = 0.0
            self._cash_operating_paid = 0.0
            self._maintenance_paid = 0.0
            self._cash_orders = {}
        
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
        self._load_configured_prices()
        if mechanism_enabled(getattr(self.model, "config", {}), "service_coordination"):
            coordination = service_coordination_params(getattr(self.model, "config", {}))
            cost_reduction = max(0.0, min(0.8, coordination.get("service_cost_reduction", 0.0)))
            if cost_reduction:
                self.service_prices = {
                    service_type: price * (1.0 - cost_reduction)
                    for service_type, price in self.service_prices.items()
                }
        
        # 效率因子（规模经济）
        self.efficiency_factor = 1.0
        self._update_efficiency()

    def _safe_float(
        self,
        value: Any,
        default: float,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = float(default)

        if minimum is not None:
            result = max(float(minimum), result)
        if maximum is not None:
            result = min(float(maximum), result)
        return result

    def _get_provider_economics_config(self) -> Dict[str, Any]:
        config = getattr(self.model, "config", {}) or {}
        provider_cfg = (
            config.get("production", {})
            .get("agricultural_service", {})
            .get("provider_economics", {})
        )
        return provider_cfg if isinstance(provider_cfg, dict) else {}

    def _resolve_initial_capacity(self, initial_capacity: float) -> float:
        if initial_capacity is not None:
            try:
                resolved = float(initial_capacity)
                if resolved > 0:
                    return resolved
            except (TypeError, ValueError):
                pass

        capacity_cfg = self._provider_economics_config.get("initial_capacity")
        if isinstance(capacity_cfg, dict):
            min_capacity = self._safe_float(capacity_cfg.get("min"), 500.0, minimum=1.0)
            max_capacity = self._safe_float(capacity_cfg.get("max"), min_capacity, minimum=min_capacity)
            return 0.5 * (min_capacity + max_capacity)
        if capacity_cfg is not None:
            return self._safe_float(capacity_cfg, 500.0, minimum=1.0)
        return 500.0

    def _resolve_service_type_key(self, service_type: ServiceType) -> Optional[str]:
        mapping = {
            ServiceType.FULL_TRUSTEESHIP: "full_trusteeship",
            ServiceType.PARTIAL_TRUSTEESHIP: "partial_trusteeship",
            ServiceType.SINGLE_SERVICE: "single_service",
            ServiceType.FULL_CUSTODY: "full_trusteeship",
        }
        return mapping.get(service_type)

    def _load_configured_prices(self):
        config = getattr(self.model, "config", {}) or {}
        service_types = (
            config.get("production", {})
            .get("agricultural_service", {})
            .get("service_types", {})
        )
        if not isinstance(service_types, dict):
            return

        for service_type in list(self.service_prices.keys()):
            config_key = self._resolve_service_type_key(service_type)
            if not config_key:
                continue
            service_cfg = service_types.get(config_key, {}) or {}
            configured_fee = service_cfg.get("base_fee_per_mu")
            if configured_fee is None:
                continue
            try:
                self.service_prices[service_type] = float(configured_fee)
            except (TypeError, ValueError):
                continue
    
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

    def get_utilization_rate(self) -> float:
        """Compatibility helper for model-level service statistics."""
        if self.current_capacity <= 0:
            return 0.0
        return max(0.0, min(1.0, self.service_area_used / self.current_capacity))
    
    def can_serve(self, farmer_pos: Tuple[int, int], required_area: float) -> bool:
        """
        检查是否能为该农户提供服务
        考虑：距离和容量
        """
        if required_area > self.get_remaining_capacity():
            return False
        if self.cash_mode == "operating_cash":
            pending = sum(r["area"] for r in self._cash_orders.values() if not r["added"])
            reserved = sum(r["cost"] for r in self._cash_orders.values() if not r["provided"])
            maintenance = self._maintenance_required() - self._maintenance_paid
            if (required_area > self.get_remaining_capacity() - pending + 1e-8
                    or required_area * 70 + reserved + maintenance > self.cash_balance + 1e-8):
                return False
        
        if farmer_pos is None or self.pos is None:
            return True  # 无位置信息时默认可服务
        
        # 计算距离
        dx = abs(farmer_pos[0] - self.pos[0])
        dy = abs(farmer_pos[1] - self.pos[1])
        distance = max(dx, dy)  # 切比雪夫距离
        
        return distance <= self.service_radius

    def _maintenance_required(self) -> float:
        return (self.large_tractors * 12000 + self.combine_harvesters * 20000
                + self.drone_sprayers * 3000) * 0.1

    def reserve_operating_cash(self, contract: ServiceContract) -> bool:
        """Reserve original labor/fuel costs and annual maintenance before matching."""
        if self.cash_mode == "legacy":
            return True
        from core.p3_project_contracts import P3AccountingError, configured_program
        if configured_program(self.model) is None:
            raise P3AccountingError("operating_cash requires P3")
        if (contract.service_provider_id != self.unique_id or contract.duration != 1
                or not math.isfinite(contract.area) or contract.area <= 0
                or not math.isfinite(contract.fee_per_mu) or contract.fee_per_mu < 0):
            raise P3AccountingError("operating_cash requires a valid annual own-provider contract")
        previous = self._cash_orders.get(id(contract))
        if previous is not None:
            if previous["area"] != contract.area or previous["fee"] != contract.area * contract.fee_per_mu:
                raise P3AccountingError("reserved service contract changed")
            return True
        if not self.can_serve(None, contract.area):
            return False
        self._cash_orders[id(contract)] = {
            "contract": contract, "area": contract.area, "cost": contract.area * 70,
            "fee": contract.area * contract.fee_per_mu,
            "task_sequence": len(self._cash_orders) + 1,
            "provided": False, "added": False, "received": False,
        }
        return True

    def validate_cash_receipt(self, contract: ServiceContract) -> None:
        from core.p3_project_contracts import P3AccountingError
        row = self._cash_orders.get(id(contract))
        if (row is None or not row["provided"] or not row["added"]
                or row["fee"] != contract.area * contract.fee_per_mu):
            raise P3AccountingError("cash receipt requires unchanged fulfilled service")

    def receive_confirmed_payment(self, contract: ServiceContract) -> None:
        """Called only after P3 has reconciled the farmer's actual expense posting."""
        from core.p3_project_contracts import P3AccountingError
        self.validate_cash_receipt(contract)
        row = self._cash_orders[id(contract)]
        if (contract.p3_payment_step is None or not contract.p3_fulfilled
                or not math.isclose(contract.p3_paid_yuan, row["fee"], abs_tol=1e-8)):
            raise P3AccountingError("cash receipt lacks confirmed P3 payment")
        if not row["received"]:
            self.cash_balance += contract.p3_paid_yuan
            self._cash_receipts += contract.p3_paid_yuan
            row["received"] = True
    
    def calculate_service_fee(self, area: float, service_type: ServiceType, market_price_multiplier: float = 1.0) -> float:
        """
        计算服务费用
        @theory 规模折扣 + 效率加成 + 供需动态定价
        
        @fix 2025-12-22: 集成市场供需比动态定价
        
        Args:
            area: 服务面积
            service_type: 服务类型
            market_price_multiplier: 市场供需价格乘数（由ServiceMarket提供）
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
        
        # @fix 2025-12-22: 应用市场供需价格乘数
        return base_price * discount * efficiency_discount * market_price_multiplier
    
    def add_service_contract(self, contract: ServiceContract):
        """添加服务合同"""
        if self.cash_mode == "operating_cash":
            from core.p3_project_contracts import P3AccountingError
            row = self._cash_orders.get(id(contract))
            if row is None or not row["provided"]:
                raise P3AccountingError("cash-mode contract must be fulfilled before registration")
            if row["added"]:
                return
            row["added"] = True
        self.active_contracts.append(contract)
        self.service_area_used += contract.area
        self._update_efficiency()
    
    def provide_service(self, contract: ServiceContract) -> Dict:
        """
        执行服务并返回结果
        """
        service_fee = contract.fee_per_mu * contract.area
        if self.cash_mode == "operating_cash":
            from core.p3_project_contracts import P3AccountingError
            if contract.p3_application_step is None:
                raise P3AccountingError("cash-mode service requires P3 prepare_service")
            if not self.reserve_operating_cash(contract):
                raise P3AccountingError("provider lacks reserved operating cash")
            row = self._cash_orders[id(contract)]
            if row["provided"]:
                return dict(row["result"])
            maintenance = self._maintenance_required() - self._maintenance_paid
            self.cash_balance -= row["cost"] + maintenance
            self._cash_operating_paid += row["cost"] + maintenance
            self._maintenance_paid += maintenance
            row["provided"] = True
        
        result = {
            'yield_bonus': self.efficiency_factor,
            'labor_saved': contract.area * 200 * contract.coverage_ratio,
            'service_fee': service_fee,
        }
        
        self.annual_revenue += service_fee
        if self.cash_mode == "operating_cash":
            row["result"] = dict(result)
        return result
    
    def end_of_year_settlement(self, current_year: int) -> float:
        """年末结算"""
        settlement_year = current_year
        try:
            settlement_year = int(settlement_year)
            base_year = getattr(self.model, "simulation_base_year", None)
            if base_year is not None:
                base_year = int(base_year)
                if settlement_year < base_year:
                    settlement_year = base_year + settlement_year
        except (TypeError, ValueError):
            settlement_year = _resolve_model_simulation_year(self.model)

        if self.last_settlement_year == settlement_year:
            return float(self.last_settlement_profit)

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
        maintenance_required = maintenance_cost
        if self.cash_mode == "operating_cash":
            from core.p3_project_contracts import P3AccountingError
            if any(not r["received"] for r in self._cash_orders.values()):
                raise P3AccountingError("close P3 orders and confirmed payments before annual settlement")
            # Unfunded maintenance means no operation, not credit or an unpaid debt.
            if not self._maintenance_paid and self.cash_balance >= maintenance_cost:
                self.cash_balance -= maintenance_cost
                self._cash_operating_paid += maintenance_cost
                self._maintenance_paid = maintenance_cost
            maintenance_cost = self._maintenance_paid
            if not math.isclose(labor_cost + fuel_cost + maintenance_cost,
                                self._cash_operating_paid, abs_tol=1e-7):
                raise P3AccountingError("service operating cash differs from fulfilled costs")
        
        self.annual_cost = equipment_depreciation + labor_cost + fuel_cost + maintenance_cost
        
        # 利润
        profit = self.annual_revenue - self.annual_cost
        # Preserve accrual components; only the explicit P3 cash mode posts operating cash.
        self.last_settlement_account = {
            "year": settlement_year,
            "revenue_yuan": self.annual_revenue,
            "depreciation_yuan": equipment_depreciation,
            "labor_cost_yuan": labor_cost,
            "fuel_cost_yuan": fuel_cost,
            "maintenance_cost_yuan": maintenance_cost,
            "operating_cost_yuan": labor_cost + fuel_cost + maintenance_cost,
            "profit_yuan": profit,
            "capital_before_yuan": self.capital,
            "capital_after_yuan": self.capital + profit,
            "cash_receipts_yuan": None,
            "operating_cash_paid_yuan": None,
            "equipment_investment_paid_yuan": None,
            "equipment_purchase_posting_implemented": False,
        }
        self.capital += profit
        if self.cash_mode == "operating_cash":
            self.last_settlement_account.update({
                "cash_open_yuan": self._cash_open,
                "cash_receipts_yuan": self._cash_receipts,
                "operating_cash_paid_yuan": self._cash_operating_paid,
                "cash_close_yuan": self.cash_balance,
                "maintenance_required_yuan": maintenance_required,
                "maintenance_not_performed_yuan": maintenance_required - maintenance_cost,
                "operating_counterparty": "external_labor_fuel_maintenance_sector",
                "equipment_investment_paid_yuan": 0.0,
                "expansion_frozen": True,
                "cash_mode": self.cash_mode,
                "orders": [{
                    "farmer_id": str(r["contract"].farmer_id),
                    "provider_id": str(self.unique_id),
                    "year": settlement_year,
                    "task_type": getattr(r["contract"].service_type, "value", None),
                    "task_sequence": r["task_sequence"],
                    "area_mu": r["area"], "invoice_yuan": r["fee"],
                    "labor_paid_yuan": r["area"] * 45,
                    "fuel_paid_yuan": r["area"] * 25,
                    "actual_receipt_yuan": r["contract"].p3_paid_yuan,
                    "payment_step": r["contract"].p3_payment_step,
                } for r in self._cash_orders.values()],
            })
            self._cash_open = self.cash_balance
            self._cash_receipts = self._cash_operating_paid = self._maintenance_paid = 0.0
            self._cash_orders.clear()
        self.cumulative_profit += profit
        
        # 清理过期合同
        self.active_contracts = [
            c for c in self.active_contracts
            if not c.is_expired(settlement_year)
        ]
        
        # 根据利润调整能力
        if self.cash_mode == "legacy" and profit > self.expansion_investment_threshold and self.current_capacity < self.max_capacity:
            # 扩张能力
            expansion = min(
                self.max_annual_expansion * self.current_capacity,
                self.max_capacity - self.current_capacity,
            )
            if expansion > 0:
                self.current_capacity += expansion
                self.large_tractors += 1
        
        # 重置年度计数
        self.service_area_used = sum(c.area for c in self.active_contracts)
        self.annual_revenue = 0.0
        self.annual_cost = 0.0
        self.last_settlement_year = settlement_year
        self.last_settlement_profit = float(profit)
        
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
    
    @critical_fix 2025-12-22:
    引入供需比(S/D Ratio)动态定价机制,替换原有静态定价逻辑
    实现服务价格随供需关系动态调整
    """
    
    def __init__(self, model: Any, config: dict = None):
        self.model = model
        self.config = config or {}
        self.transaction_history: List[Dict] = []
        
        # 统计
        self.total_area_served = 0.0
        self.total_revenue = 0.0
        self.n_contracts_signed = 0
        
        # @new 2025-12-22: 供需比动态定价参数
        self.base_prices = {
            ServiceType.FULL_TRUSTEESHIP: 500.0,
            ServiceType.PARTIAL_TRUSTEESHIP: 280.0,
            ServiceType.SINGLE_SERVICE: 100.0,
            ServiceType.FULL_CUSTODY: 450.0,
            ServiceType.HARVESTING: 120.0,
            ServiceType.TILLAGE: 80.0,
        }
        self._load_configured_base_prices()
        
        # 供需状态
        self._total_demand = 0.0  # 总需求面积
        self._total_supply = 0.0  # 总供给能力
        self._supply_demand_ratio = 1.0  # 供需比
        self._current_price_multiplier = 1.0  # 价格乘数
        
        # 价格调整参数
        self.price_elasticity = self.config.get('price_elasticity', 0.3)  # 价格弹性
        self.min_price_multiplier = self.config.get('min_price_multiplier', 0.7)  # 最低价格倍数
        self.max_price_multiplier = self.config.get('max_price_multiplier', 1.5)  # 最高价格倍数

    def _resolve_service_type_key(self, service_type: ServiceType) -> Optional[str]:
        mapping = {
            ServiceType.FULL_TRUSTEESHIP: "full_trusteeship",
            ServiceType.PARTIAL_TRUSTEESHIP: "partial_trusteeship",
            ServiceType.SINGLE_SERVICE: "single_service",
            ServiceType.FULL_CUSTODY: "full_trusteeship",
        }
        return mapping.get(service_type)

    def _get_service_type_config(self, service_type: ServiceType) -> Dict[str, Any]:
        config_key = self._resolve_service_type_key(service_type)
        if not config_key:
            return {}
        return self.config.get("service_types", {}).get(config_key, {}) or {}

    def _load_configured_base_prices(self):
        for service_type in list(self.base_prices.keys()):
            service_cfg = self._get_service_type_config(service_type)
            configured_fee = service_cfg.get("base_fee_per_mu")
            if configured_fee is None:
                continue
            try:
                self.base_prices[service_type] = float(configured_fee)
            except (TypeError, ValueError):
                continue
    
    def update_supply_demand_ratio(self, demands: List[Dict], providers: List[ServiceProvider]):
        """
        @critical_fix 2025-12-22: 更新供需比并计算动态价格乘数
        
        供需比计算: SD_Ratio = Total_Supply / Total_Demand
        价格乘数: Price_Multiplier = 1 / (SD_Ratio ^ elasticity)
        
        - SD_Ratio > 1: 供过于求,价格下降
        - SD_Ratio < 1: 供不应求,价格上升
        """
        # 计算总需求
        self._total_demand = sum(d.get('area', 0) for d in demands)
        
        # 计算总供给
        self._total_supply = sum(p.get_remaining_capacity() for p in providers)
        if mechanism_enabled(getattr(self.model, "config", {}), "service_coordination"):
            coordination = service_coordination_params(getattr(self.model, "config", {}))
            dispatch_efficiency = coordination.get(
                "service_dispatch_efficiency",
                coordination.get("service_matching_boost", 0.0),
            )
            self._total_supply *= 1.0 + max(0.0, dispatch_efficiency)
        
        # 计算供需比
        if self._total_demand > 0:
            self._supply_demand_ratio = self._total_supply / self._total_demand
        else:
            self._supply_demand_ratio = float('inf') if self._total_supply > 0 else 1.0
        
        # 计算价格乘数
        if self._supply_demand_ratio > 0 and self._supply_demand_ratio != float('inf'):
            raw_multiplier = 1.0 / (self._supply_demand_ratio ** self.price_elasticity)
            self._current_price_multiplier = max(
                self.min_price_multiplier,
                min(self.max_price_multiplier, raw_multiplier)
            )
        else:
            self._current_price_multiplier = 1.0
    
    def get_dynamic_price(self, service_type: ServiceType, area: float = 1.0) -> float:
        """
        @critical_fix 2025-12-22: 获取动态定价
        
        Dynamic_Price = Base_Price * Price_Multiplier * Scale_Discount
        """
        base_price = self.base_prices.get(service_type, 300.0)
        
        # 规模折扣
        if area >= 100:
            scale_discount = 0.80
        elif area >= 50:
            scale_discount = 0.88
        elif area >= 20:
            scale_discount = 0.93
        elif area >= 10:
            scale_discount = 0.97
        else:
            scale_discount = 1.0
        
        return base_price * self._current_price_multiplier * scale_discount
    
    def match_service_demands(self, demands: List[Dict], providers: List[ServiceProvider]) -> List[Dict]:
        """
        匹配服务需求与服务提供商
        
        @critical_fix 2025-12-22: 集成动态定价机制
        
        Args:
            demands: 农户服务需求列表 [{farmer, area, service_type, max_price_per_mu}]
            providers: 服务提供商列表
            
        Returns:
            匹配结果列表 [{farmer, provider, contract}]
        """
        matches = []
        
        # @critical_fix 2025-12-22: 先更新供需比
        self.update_supply_demand_ratio(demands, providers)
        
        # 按需求面积排序（大户优先，更容易找到服务商）
        sorted_demands = sorted(demands, key=lambda d: d.get('area', 0), reverse=True)
        from core.p3_project_contracts import configured_program
        p3_program = configured_program(self.model)
        reserved_area = {}
        reserved_cash = {}
        
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
                and (p3_program is None or p.cash_mode == "operating_cash"
                     or area <= p.get_remaining_capacity() - reserved_area.get(p.unique_id, 0.0))
            ]
            
            if not available_providers:
                continue
            
            # @critical_fix 2025-12-22: 使用动态定价
            # 按价格排序，选择最便宜的
            best_provider = None
            best_price_per_mu = float('inf')
            
            for provider in available_providers:
                # 使用动态定价而非静态定价
                fee_per_mu = provider.calculate_service_fee(
                    area,
                    service_type,
                    market_price_multiplier=self._current_price_multiplier,
                )
                
                if fee_per_mu <= max_price and fee_per_mu < best_price_per_mu:
                    best_provider = provider
                    best_price_per_mu = fee_per_mu
            
            if best_provider is not None:
                if p3_program is not None:
                    own_cash = farmer.balance_sheet_snapshot()["working_capital"]
                    committed = float(getattr(farmer, "service_fee_expense", 0.0)) + reserved_cash.get(farmer.unique_id, 0.0)
                    if best_price_per_mu * area > own_cash - committed:
                        continue
                    reserved_cash[farmer.unique_id] = reserved_cash.get(farmer.unique_id, 0.0) + best_price_per_mu * area
                    reserved_area[best_provider.unique_id] = reserved_area.get(best_provider.unique_id, 0.0) + area
                # 创建合同
                fee_per_mu = best_price_per_mu
                
                # 确定覆盖比例
                service_cfg = self._get_service_type_config(service_type)
                if service_type == ServiceType.FULL_TRUSTEESHIP:
                    default_coverage = 1.0
                elif service_type == ServiceType.PARTIAL_TRUSTEESHIP:
                    default_coverage = 0.5
                else:
                    default_coverage = 0.2
                try:
                    coverage = float(service_cfg.get('coverage_ratio', default_coverage) or default_coverage)
                except (TypeError, ValueError):
                    coverage = default_coverage
                coverage = max(0.0, min(1.0, coverage))
                
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
                    year_signed=_resolve_model_simulation_year(self.model)
                )
                
                # 记录匹配
                if not best_provider.reserve_operating_cash(contract):
                    continue
                match = {
                    'farmer': farmer,
                    'provider': best_provider,
                    'contract': contract
                }
                for passthrough_key in ('service_managed_retained_grain',):
                    if passthrough_key in demand:
                        match[passthrough_key] = demand[passthrough_key]
                matches.append(match)
                
                # 更新统计
                self.total_area_served += area
                self.total_revenue += fee_per_mu * area
                self.n_contracts_signed += 1
                
                # 记录历史
                self.transaction_history.append({
                    'year': _resolve_model_simulation_year(self.model),
                    'farmer_id': farmer.unique_id,
                    'provider_id': best_provider.unique_id,
                    'area': area,
                    'service_type': service_type.value,
                    'fee_per_mu': fee_per_mu
                })
        
        return matches
    
    def get_statistics(self) -> Dict:
        """获取市场统计（包含动态定价信息）"""
        return {
            'total_area_served': self.total_area_served,
            'total_revenue': self.total_revenue,
            'n_contracts': self.n_contracts_signed,
            'avg_fee_per_mu': self.total_revenue / max(1, self.total_area_served),
            # 动态定价统计
            'supply_demand_ratio': self.supply_demand_ratio,
            'price_multiplier': self.price_multiplier,
            'total_supply': self.total_supply,
            'total_demand': self.total_demand,
            'market_tension': 'supply_shortage' if self.supply_demand_ratio < 0.8 else (
                'balanced' if self.supply_demand_ratio < 1.2 else 'supply_surplus'
            )
        }
    
    def get_dynamic_pricing_state(self) -> Dict:
        """获取动态定价状态（供外部模块查询）"""
        return {
            'sd_ratio': self.supply_demand_ratio,
            'price_multiplier': self.price_multiplier,
            'elasticity': self.price_elasticity,
            'min_multiplier': self.min_price_multiplier,
            'max_multiplier': self.max_price_multiplier,
            'effective_price_adjustment': f"{(self.price_multiplier - 1) * 100:+.1f}%"
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
