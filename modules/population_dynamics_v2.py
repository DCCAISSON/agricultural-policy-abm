"""
PopulationManager V2: 宏观锚定+土地守恒人口动态模块
(Macro-Anchored Land-Conserving Population Dynamics)

设计原理:
=========
1. 宏观锚定: N_model(t) = N_model(2010) × Pop_stats(t) / Pop_stats(2010)
2. 土地守恒: 总土地量不变，退出者土地流转给其他农户/企业
3. 微观选择: 基于Agent属性的城镇化概率模型

数据源组合:
| 模块       | 数据源                | 作用                      |
|------------|----------------------|---------------------------|
| 总量趋势   | NBS统计年鉴          | 决定有多少人 (Quantity)    |
| 退出规则   | CFPS面板追踪         | 决定谁离开 (Selection)     |
| 新进入特征 | CFPS当年截面         | 决定新来的人什么样 (Attr)  |
| 土地/技术  | NFPS (2009-2015)     | 决定生产函数 (Production)  |

退出机制:
- 自然退出: 老龄化死亡（基于年龄分布）
- 城镇化退出: P(Exit) = f(Income, NonFarmRatio, Age, Education)

进入机制:
- 触发条件: △N = N_target - (N_current - N_exit) > 0
- 特征生成: Bootstrap抽样从CFPS当年截面

土地守恒:
- 退出者土地 → LandBank
- 新进入者 ← LandBank分配
- 剩余土地 → 分配给现有农户/企业

参考文献:
- Matthews et al. (2007) JASSS - ABM人口动态
- Berger (2001) Agricultural Systems - MPMAS框架

作者: Agricultural Policy ABM Team
日期: 2026-01-06 (V2重构)
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from core.income_utils import (
    get_farmer_agriculture_linked_income,
    get_farmer_gross_income,
    get_farmer_livelihood_agriculture_ratio,
    get_farmer_net_income,
    get_farmer_operated_land_area,
)
try:
    from core.income_utils import (
        get_farmer_livelihood_strategy_name as _income_utils_get_farmer_livelihood_strategy_name,
        is_professional_large_operator as _income_utils_is_professional_large_operator,
    )
except ImportError:
    _income_utils_get_farmer_livelihood_strategy_name = None
    _income_utils_is_professional_large_operator = None
from core.reproducibility import (
    np_lognormal,
    np_normal,
    rng_choice,
    rng_randint,
    rng_random,
    rng_shuffle,
    rng_uniform,
)

logger = logging.getLogger(__name__)


def _capacity_diagnostic_row(
    operator_id: str,
    current_area_mu: float,
    desired_area_mu: float,
    capacity_limit_area_mu: float,
    allocated_area_mu: float,
) -> Dict[str, Any]:  # noqa: DICT_OK
    current = max(0.0, float(current_area_mu or 0.0))
    desired = max(current, float(desired_area_mu or 0.0))
    capacity = max(0.0, float(capacity_limit_area_mu or 0.0))
    allocated = max(0.0, float(allocated_area_mu or 0.0))
    delta_desired = max(desired - current, 0.0)
    delta_feasible = min(delta_desired, max(capacity - current, 0.0))
    return {
        "operator_id": str(operator_id),
        "current_area_mu": current,
        "a_desired_mu": desired,
        "a_bar_mu": capacity,
        "delta_a_desired_mu": delta_desired,
        "delta_a_feasible_mu": delta_feasible,
        "delta_a_allocated_mu": allocated,
        "capacity_gap_mu": max(delta_desired - delta_feasible, 0.0),
        "binding": bool(delta_desired > max(capacity - current, 0.0) + 1e-9),
        "area_unit": "mu",
    }


def _exit_signal_diagnostic_row(
    operator_id: str,
    base_signal: float,
    pressure_scale: float,
    final_action: str,
) -> Dict[str, Any]:  # noqa: DICT_OK
    base = min(1.0, max(0.0, float(base_signal or 0.0)))
    scale = max(0.0, float(pressure_scale or 0.0))
    normalized_action = {
        "Continue": "Continue",
        "maintain_grain": "Continue",
        "Expand": "Expand",
        "expand_cash": "Expand",
        "Shrink": "Shrink",
        "contract": "Shrink",
        "rent_out": "Shrink",
        "Exit": "Exit",
        "exit": "Exit",
    }.get(str(final_action), "Continue")
    return {
        "operator_id": str(operator_id),
        "exit_signal_base": base,
        "exit_pressure_scale": scale,
        "exit_signal_treated": min(1.0, max(0.0, base * scale)),
        "final_action": normalized_action,
    }


@dataclass
class LandParcel:
    """土地地块"""
    area_mu: float
    quality: float = 1.0
    location_x: float = 0.0
    location_y: float = 0.0
    previous_owner_id: Any = None
    exit_year: int = 0
    exit_type: str = ""
    patches: List[Any] = field(default_factory=list)
    release_event_id: str = ""


@dataclass
class LandBank:
    """
    土地银行：管理退出农户释放的土地资源
    
    核心原则: 土地守恒
    - 所有退出土地必须流向：新进入者/现有农户/企业
    - 土地不会凭空消失或增加
    """
    
    parcels: List[LandParcel] = field(default_factory=list)
    total_area_mu: float = 0.0
    cumulative_deposits: float = 0.0  # 累计存入
    cumulative_withdrawals: float = 0.0  # 累计取出
    
    random_context: Any = None
    production_succession_account: Any = None
    current_year: int = 0

    def deposit(self, parcel: LandParcel) -> None:
        """存入退出农户释放的土地"""
        if parcel.area_mu > 0:
            account = self.production_succession_account
            if account is not None and not parcel.release_event_id:
                from modules.production_succession_accounting import (
                    ReleaseRegistration,
                    grain_status,
                )

                patch_ids = []
                origin_statuses = []
                for patch in parcel.patches:
                    patch_id_getter = getattr(patch, 'get_patch_id', None)
                    patch_ids.append(
                        str(patch_id_getter())
                        if callable(patch_id_getter)
                        else str(getattr(patch, 'patch_id', id(patch)))
                    )
                    crop_status = None
                    for crop in (
                        getattr(patch, 'current_crop', None),
                        getattr(patch, 'previous_crop_type', None),
                        getattr(patch, 'production_succession_origin_crop_label', None),
                        getattr(patch, 'last_land_use', None),
                    ):
                        crop_status = grain_status(
                            None if crop is None else str(getattr(crop, 'value', crop))
                        )
                        if crop_status is not None:
                            break
                    origin_statuses.append(crop_status)
                known_origin = [status for status in origin_statuses if status is not None]
                was_grain_land = (
                    known_origin[0]
                    if known_origin and len(known_origin) == len(origin_statuses)
                    and all(status is known_origin[0] for status in known_origin)
                    else None
                )
                release = account.record_release(
                    ReleaseRegistration(
                        release_year=int(parcel.exit_year),
                        source_operator_id=str(parcel.previous_owner_id),
                        source_operator_type=(
                            'enterprise' if parcel.exit_type == 'enterprise_exit' else 'farmer'
                        ),
                        release_reason=str(parcel.exit_type),
                        area_mu=float(parcel.area_mu),
                        patch_ids=tuple(patch_ids),
                        was_grain_land=was_grain_land,
                    )
                )
                parcel.release_event_id = release.release_event_id
                for patch in parcel.patches:
                    setattr(patch, 'production_succession_release_event_id', release.release_event_id)
            self.parcels.append(parcel)
            self.total_area_mu += parcel.area_mu
            self.cumulative_deposits += parcel.area_mu
            logger.debug(f"LandBank存入: {parcel.area_mu:.2f}亩 (from {parcel.exit_type}), "
                        f"累计: {self.total_area_mu:.2f}亩")
    
    def withdraw(self, target_area: float, 
                 prefer_contiguous: bool = True) -> List[LandParcel]:
        """
        为新主体分配土地
        
        Args:
            target_area: 目标面积(亩)
            prefer_contiguous: 是否优先分配连片土地
        
        Returns:
            分配的地块列表
        """
        if self.total_area_mu <= 0 or target_area <= 0:
            return []
        
        allocated = []
        remaining_need = target_area
        
        # 排序策略
        if prefer_contiguous:
            # 优先分配大块土地
            self.parcels.sort(key=lambda x: x.area_mu, reverse=True)
        else:
            # 随机分配
            rng_shuffle(self.random_context, self.parcels)
        
        for parcel in self.parcels[:]:
            if remaining_need <= 0:
                break
            if parcel.area_mu <= remaining_need * 1.2:  # 允许20%超额
                allocated.append(parcel)
                remaining_need -= parcel.area_mu
                self.parcels.remove(parcel)
                self.total_area_mu -= parcel.area_mu
                self.cumulative_withdrawals += parcel.area_mu
        
        return allocated
    
    def distribute_to_neighbors(self, 
                                agents: List[Any],
                                max_per_agent: float = 50.0,
                                max_total_area: Optional[float] = None,
                                max_area_by_recipient: Optional[Dict[int, float]] = None,
                                preserve_order: bool = False,
                                rent_per_mu: float = 0.0) -> Dict:
        """
        将剩余土地分配给现有农户（模拟流转市场）
        
        Args:
            agents: 可接收土地的农户列表
            max_per_agent: 单个农户最大增量(亩)
        
        Returns:
            分配统计
        """
        stats = {'n_recipients': 0, 'total_distributed': 0.0, 'allocations': []}
        
        if not self.parcels or not agents:
            return stats
        
        total_cap = (
            float(max_total_area)
            if max_total_area is not None
            else float(max_per_agent) * max(1, len(agents))
        )
        per_agent_cap = float(max_per_agent)

        # 按土地规模排序（小农户优先获得扩大机会）
        eligible_agents = list(agents)
        
        if not eligible_agents:
            eligible_agents = agents
        
        if not preserve_order:
            rng_shuffle(self.random_context, eligible_agents)
        
        recipient_ids = set()
        additional_area_by_recipient = {}

        for parcel in self.parcels[:]:
            if not eligible_agents:
                break
            if stats['total_distributed'] >= total_cap:
                break
            if stats['total_distributed'] + parcel.area_mu > total_cap:
                continue
            
            chosen_idx = None
            for idx, candidate in enumerate(eligible_agents):
                recipient_key = id(candidate)
                already_added = additional_area_by_recipient.get(recipient_key, 0.0)
                recipient_cap = per_agent_cap
                if max_area_by_recipient is not None:
                    recipient_cap = min(
                        recipient_cap,
                        max(0.0, float(max_area_by_recipient.get(recipient_key, 0.0) or 0.0)),
                    )
                if already_added + parcel.area_mu <= recipient_cap:
                    chosen_idx = idx
                    break

            if chosen_idx is None:
                continue

            recipient = eligible_agents[chosen_idx]
            if not self._passes_parcel_financial_gate(recipient, parcel, rent_per_mu):
                eligible_agents.append(eligible_agents.pop(chosen_idx))
                continue
            self._assign_parcel_to_recipient(recipient, parcel, rental_price=rent_per_mu)
            
            recipient_key = id(recipient)
            additional_area_by_recipient[recipient_key] = (
                additional_area_by_recipient.get(recipient_key, 0.0) + parcel.area_mu
            )
            recipient_ids.add(recipient_key)
            stats['n_recipients'] = len(recipient_ids)
            stats['total_distributed'] += parcel.area_mu
            stats['allocations'].append(
                {
                    'recipient_id': str(
                        recipient.get('unique_id', 'unknown')
                        if isinstance(recipient, dict)
                        else getattr(recipient, 'unique_id', 'unknown')
                    ),
                    'recipient_type': (
                        'enterprise'
                        if (
                            bool(recipient.get('enterprise_type'))
                            if isinstance(recipient, dict)
                            else hasattr(recipient, 'enterprise_type')
                        )
                        else 'farmer'
                    ),
                    'is_new_entrant': (
                        bool(recipient.get('is_new_entrant'))
                        if isinstance(recipient, dict)
                        else bool(getattr(recipient, 'is_new_entrant', False))
                    ),
                    'release_event_id': str(parcel.release_event_id or ''),
                    'absorbed_area_mu': float(parcel.area_mu),
                    'area_unit': 'mu',
                }
            )

            self.parcels.remove(parcel)
            self.total_area_mu -= parcel.area_mu
            self.cumulative_withdrawals += parcel.area_mu

            # 轮换接收者
            eligible_agents.append(eligible_agents.pop(chosen_idx))

        if stats['total_distributed'] > 0:
            model = getattr(self.random_context, 'model', None)
            if model is None and agents:
                model = getattr(agents[0], 'model', None)
            invalidate_fn = getattr(model, 'invalidate_land_tenure_caches', None)
            if callable(invalidate_fn):
                try:
                    invalidate_fn()
                except Exception:
                    pass
        
        return stats

    def _passes_parcel_financial_gate(self, recipient: Any, parcel: LandParcel, rental_price: float = 0.0) -> bool:
        rent = max(0.0, float(rental_price or 0.0))
        if rent <= 0.0:
            return True
        patches = list(getattr(parcel, 'patches', []) or [])
        if not patches:
            return True  # ponytail: synthetic parcels lack patch-level assets; gate them once they carry real patches.
        for patch in patches:
            gate = getattr(patch, '_passes_operational_financial_gate', None)
            patch_rent = rent
            if hasattr(patch, 'initial_rental_rights_holder') and not hasattr(recipient, 'land_patches'):
                patch_rent *= float(patch.area)
            if callable(gate) and not gate(recipient, patch_rent):
                return False
        return True

    def _get_agent_land_area(self, agent: Any) -> float:
        if isinstance(agent, dict):
            parcels = agent.get('land_parcels') or []
            parcel_area = float(sum(
                float(getattr(parcel, 'area_mu', 0.0) or 0.0)
                for parcel in parcels
            ))
            if parcel_area > 0:
                return parcel_area
            return float(agent.get('land_area_mu', agent.get('land_area', 0.0)) or 0.0)
        if hasattr(agent, 'get_owned_land_area'):
            try:
                return float(agent.get_owned_land_area(include_leased_out=True))
            except TypeError:
                return float(agent.get_owned_land_area())
            except Exception:
                pass
        if hasattr(agent, 'get_operated_land_area'):
            try:
                return float(agent.get_operated_land_area())
            except Exception:
                pass
        return float(getattr(agent, 'land_area_mu', 0.0) or 0.0)

    def _assign_parcel_to_recipient(self, recipient: Any, parcel: LandParcel, rental_price: float = 0.0) -> None:
        if isinstance(recipient, dict):
            recipient['land_area_mu'] = float(recipient.get('land_area_mu', 0.0) or 0.0) + float(parcel.area_mu)
            land_parcels = recipient.setdefault('land_parcels', [])
            if parcel not in land_parcels:
                land_parcels.append(parcel)
        else:
            recipient.land_area_mu = float(getattr(recipient, 'land_area_mu', 0.0) or 0.0) + float(parcel.area_mu)

            if hasattr(recipient, 'received_land_transfers'):
                recipient.received_land_transfers.append({
                    'area': parcel.area_mu,
                    'year': parcel.exit_year,
                    'source': parcel.exit_type
                })

            for patch in getattr(parcel, 'patches', []) or []:
                patch.owner = recipient
                if hasattr(recipient, 'land_patches'):
                    if getattr(patch, 'leased_to', None) is None:
                        patch.current_operator = recipient
                        self._mark_patch_operational_transfer(
                            patch,
                            channel='population_exit_reassignment',
                        )
                    else:
                        patch.current_operator = patch.leased_to
                    if patch not in recipient.land_patches:
                        recipient.land_patches.append(patch)
                else:
                    patch.leased_to = recipient
                    patch.current_operator = recipient
                    patch_rent = rental_price
                    if hasattr(patch, 'initial_rental_rights_holder'):
                        patch_rent *= float(patch.area)
                    patch.rental_price = patch_rent
                    patch.contracted_rental_price = patch_rent
                    patch.lease_contract_years = max(int(getattr(patch, 'lease_contract_years', 0) or 0), 3)
                    if hasattr(recipient, 'leased_patches') and patch not in recipient.leased_patches:
                        recipient.leased_patches.append(patch)
                    if hasattr(recipient, 'leased_in_patches') and patch not in recipient.leased_in_patches:
                        recipient.leased_in_patches.append(patch)
                    self._mark_patch_operational_transfer(
                        patch,
                        channel='population_exit_reassignment',
                    )

        account = self.production_succession_account
        if account is not None and parcel.release_event_id:
            from modules.production_succession_accounting import AllocationEvent, Channel

            allocation_year = int(self.current_year or parcel.exit_year)
            destination_id = str(
                recipient.get('unique_id', 'unknown')
                if isinstance(recipient, dict)
                else getattr(recipient, 'unique_id', 'unknown')
            )
            is_enterprise = (
                bool(recipient.get('enterprise_type'))
                if isinstance(recipient, dict)
                else hasattr(recipient, 'enterprise_type')
            )
            entry_year = int(
                recipient.get('entry_year', -1)
                if isinstance(recipient, dict)
                else getattr(recipient, 'entry_year', -1)
                or -1
            )
            is_new = entry_year == allocation_year
            account.record_allocation(
                AllocationEvent(
                    allocation_event_id=(
                        f"allocation_{allocation_year}_{parcel.release_event_id}_{destination_id}"
                    ),
                    release_event_id=parcel.release_event_id,
                    allocation_year=allocation_year,
                    channel=Channel.MARKET,
                    destination_operator_id=destination_id,
                    destination_operator_type='enterprise' if is_enterprise else 'farmer',
                    succession_role=(
                        'new_entrant'
                        if is_new
                        else 'existing_scale_operator' if is_enterprise else 'expanding_farmer'
                    ),
                    absorbed_area_mu=float(parcel.area_mu),
                )
            )

        if hasattr(recipient, 'sync_owned_land_area'):
            try:
                recipient.sync_owned_land_area()
            except Exception:
                pass
        elif hasattr(recipient, 'leased_patches'):
            leased_area = sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in (getattr(recipient, 'leased_patches', []) or [])
            )
            setattr(recipient, 'total_leased_area', leased_area)
            if hasattr(recipient, 'fixed_assets'):
                recipient.fixed_assets = max(
                    float(getattr(recipient, 'fixed_assets', 0.0) or 0.0),
                    leased_area * 500.0,
                )
            setattr(recipient, '_leased_patch_cache_version', None)
            setattr(recipient, '_operated_land_area_cache_version', None)
            setattr(recipient, '_operated_land_area_cache', leased_area)

    def _mark_patch_operational_transfer(self, patch: Any, channel: str) -> None:
        setattr(patch, 'operational_transfer_active', True)
        existing_channel = str(getattr(patch, 'operational_transfer_channel', '') or '').strip()
        if not existing_channel:
            setattr(patch, 'operational_transfer_channel', channel)

    def _detach_patch_from_current_tenure(self, patch: Any) -> None:
        """Release a patch into LandBank without stale lease/operator links."""
        previous_owner = getattr(patch, 'owner', None)
        previous_lessee = getattr(patch, 'leased_to', None)
        previous_operator = getattr(patch, 'current_operator', None)

        for holder in (previous_owner, previous_lessee, previous_operator):
            if holder is None:
                continue
            for attr_name in ('leased_out_patches', 'leased_in_patches', 'leased_patches'):
                patches = getattr(holder, attr_name, None)
                if not isinstance(patches, list):
                    continue
                while patch in patches:
                    patches.remove(patch)

        patch.owner = None
        patch.leased_to = None
        patch.current_operator = None
        patch.current_crop = None
        patch.rental_price = 0.0
        patch.contracted_rental_price = 0.0
        patch.lease_contract_years = 0

        for holder in (previous_owner, previous_lessee, previous_operator):
            if holder is None:
                continue
            for attr_name in (
                '_leased_patch_cache_version',
                '_operated_land_area_cache_version',
            ):
                if hasattr(holder, attr_name):
                    setattr(holder, attr_name, None)
            if hasattr(holder, '_operated_land_area_cache'):
                setattr(holder, '_operated_land_area_cache', 0.0)

    def _release_patch_operation_from_exiting_agent(self, agent: Any, patch: Any) -> bool:
        """End a non-owned lease/operator assignment when the operator exits."""
        if hasattr(patch, 'initial_rental_rights_holder'):
            if patch.initial_rental_rights_holder is agent:
                return False
            if patch.leased_to is agent or patch.current_operator is agent:
                patch.end_lease()
                return True
            return False
        if getattr(patch, 'owner', None) is agent:
            return False

        leased_to_agent = getattr(patch, 'leased_to', None) is agent
        operated_by_agent = getattr(patch, 'current_operator', None) is agent
        if not leased_to_agent and not operated_by_agent:
            return False

        owner = getattr(patch, 'owner', None)
        prior_lessee = getattr(patch, 'leased_to', None)
        prior_operator = getattr(patch, 'current_operator', None)
        affected_holders = [agent, owner, prior_lessee, prior_operator]

        for holder in affected_holders:
            if holder is None:
                continue
            for attr_name in ('leased_out_patches', 'leased_in_patches', 'leased_patches'):
                patches = getattr(holder, attr_name, None)
                if not isinstance(patches, list):
                    continue
                while patch in patches:
                    patches.remove(patch)

        if leased_to_agent:
            patch.leased_to = None
            patch.rental_price = 0.0
            patch.contracted_rental_price = 0.0
            patch.lease_contract_years = 0
            setattr(patch, 'operational_transfer_active', False)
            setattr(patch, 'operational_transfer_channel', '')

        if operated_by_agent:
            replacement_operator = getattr(patch, 'leased_to', None) or owner
            patch.current_operator = replacement_operator

        if getattr(patch, 'current_operator', None) is None and owner is not None:
            patch.current_operator = owner

        for holder in affected_holders:
            if holder is None:
                continue
            for attr_name in (
                '_leased_patch_cache_version',
                '_operated_land_area_cache_version',
            ):
                if hasattr(holder, attr_name):
                    setattr(holder, attr_name, None)

        return True

    def _release_non_owned_operations_from_exiting_agent(self, agent: Any) -> int:
        model = getattr(getattr(self, 'random_context', None), 'model', None)
        if model is None:
            model = getattr(agent, 'model', None)
        candidates = []
        candidates.extend(getattr(agent, 'leased_in_patches', []) or [])
        candidates.extend(getattr(agent, 'leased_patches', []) or [])
        if model is not None:
            candidates.extend(getattr(model, 'land_patches', []) or [])

        released = 0
        seen_patch_ids = set()
        for patch in candidates:
            patch_id = id(patch)
            if patch_id in seen_patch_ids:
                continue
            seen_patch_ids.add(patch_id)
            if self._release_patch_operation_from_exiting_agent(agent, patch):
                released += 1
        if released > 0 and model is not None:
            invalidate_fn = getattr(model, 'invalidate_land_tenure_caches', None)
            if callable(invalidate_fn):
                try:
                    invalidate_fn()
                except Exception as exc:
                    if not hasattr(model, '_land_tenure_invalidation_errors'):
                        model._land_tenure_invalidation_errors = []
                    model._land_tenure_invalidation_errors.append({
                        'stage': 'release_non_owned_operations_from_exiting_agent',
                        'agent_id': getattr(agent, 'unique_id', None),
                        'released_patch_count': released,
                        'error': str(exc),
                        'error_type': exc.__class__.__name__,
                    })
                    raise RuntimeError(
                        "land tenure cache invalidation failed after releasing exiting-agent operations"
                    ) from exc
        return released
    
    @property
    def max_contiguous_area(self) -> float:
        """最大连片土地面积"""
        if not self.parcels:
            return 0.0
        return max(p.area_mu for p in self.parcels)
    
    def get_balance_check(self) -> Dict:
        """土地收支平衡检查"""
        return {
            'current_balance': self.total_area_mu,
            'cumulative_in': self.cumulative_deposits,
            'cumulative_out': self.cumulative_withdrawals,
            'check_passed': abs(self.cumulative_deposits - self.cumulative_withdrawals - self.total_area_mu) < 0.01
        }


@dataclass
class UrbanizationModel:
    """
    城镇化概率模型
    
    基于CFPS面板追踪数据拟合的Logit模型
    P(Exit) = 1 / (1 + exp(-z))
    z = β0 + β1*income_gap + β2*non_farm_ratio + β3*age_factor + β4*education
    """
    
    # 模型系数（基于CFPS拟合，可调整）
    intercept: float = -2.5
    coef_income_gap: float = 1.2      # 与城镇收入差距
    coef_non_farm_ratio: float = 2.0  # 非农收入占比
    coef_age: float = -0.8            # 年龄因子（年龄越大越不愿迁移）
    coef_education: float = 0.5       # 教育水平
    coef_family_urban: float = 0.8    # 有家人在城镇
    
    # 参考值
    urban_avg_income: float = 50000
    
    def calculate_probability(self, agent: Any) -> float:
        """
        计算单个Agent的城镇化概率
        
        Returns:
            城镇化概率 [0, 1]
        """
        # 提取特征
        agri_income = max(0.0, float(get_farmer_agriculture_linked_income(agent) or 0.0))
        total_income = max(0.0, float(get_farmer_gross_income(agent, default=30000.0) or 0.0))
        age = getattr(agent, 'age', 45)
        education = getattr(agent, 'education_level', 2)  # 1-4
        has_urban_family = getattr(agent, 'has_urban_family', False)
        
        # 计算特征值
        income_gap = max(0, self.urban_avg_income - total_income) / self.urban_avg_income
        agri_ratio = get_farmer_livelihood_agriculture_ratio(agent, default=0.5)
        non_farm_ratio = 1.0 - agri_ratio if total_income > 0 or agri_income > 0 else 0.5
        
        # 年龄因子: 20-40岁最高，>60岁趋近0
        if age < 20:
            age_factor = 0.3
        elif age <= 40:
            age_factor = 1.0
        elif age <= 60:
            age_factor = 1.0 - (age - 40) / 40
        else:
            age_factor = 0.1
        
        edu_factor = education / 4.0  # 标准化到0-1
        family_factor = 1.0 if has_urban_family else 0.0
        
        # 线性组合
        z = (self.intercept + 
             self.coef_income_gap * income_gap +
             self.coef_non_farm_ratio * non_farm_ratio +
             self.coef_age * (1 - age_factor) +  # 注意：年龄因子反向
             self.coef_education * edu_factor +
             self.coef_family_urban * family_factor)
        
        # Logistic变换
        prob = 1 / (1 + np.exp(-z))
        
        return min(max(prob, 0.0), 1.0)


def _urbanization_model_calculate_probability(self, agent: Any) -> float:
    """Use runtime income fields and active config to score urbanization exits."""
    population_cfg = {}
    if hasattr(agent, 'model') and getattr(agent, 'model', None) is not None:
        population_cfg = getattr(agent.model, 'config', {}).get('population_dynamics', {}) or {}
    exit_cfg = population_cfg.get('exit_mechanism', {}) or {}
    urban_cfg = exit_cfg.get('urbanization_model', {}) or {}

    strategy_name = getattr(getattr(agent, 'livelihood_strategy', None), 'value', None)
    if strategy_name is None:
        strategy_name = getattr(getattr(agent, 'livelihood_strategy', None), 'name', None)
    strategy_name = str(strategy_name or '').strip()
    if strategy_name == 'Pure_Farmer':
        default_agri_ratio = 0.85
    elif strategy_name == 'Non_Farmer':
        default_agri_ratio = 0.08
    else:
        default_agri_ratio = 0.45

    income_signal = max(
        0.0,
        float(get_farmer_net_income(agent, default=0.0) or 0.0),
        float(get_farmer_gross_income(agent, default=0.0) or 0.0),
        float(getattr(agent, 'income_total', 0.0) or 0.0),
    )
    agri_ratio = get_farmer_livelihood_agriculture_ratio(agent, default=default_agri_ratio)
    non_farm_ratio = max(0.0, min(1.0, 1.0 - agri_ratio))
    age = float(getattr(agent, 'age', 45) or 45)
    education = float(getattr(agent, 'education_level', 2) or 2)
    max_education = max(1.0, float(urban_cfg.get('education_scale_max', 6.0) or 6.0))
    has_urban_family = bool(getattr(agent, 'has_urban_family', False))
    urban_avg_income = max(
        1.0,
        float(urban_cfg.get('urban_avg_income', exit_cfg.get('urban_avg_income', self.urban_avg_income)) or self.urban_avg_income)
    )
    income_gap = max(0.0, urban_avg_income - income_signal) / urban_avg_income

    if age < 20:
        age_factor = 0.3
    elif age <= 40:
        age_factor = 1.0
    elif age <= 60:
        age_factor = 1.0 - (age - 40) / 40
    else:
        age_factor = 0.1

    edu_factor = max(0.0, min(1.0, education / max_education))
    family_factor = 1.0 if has_urban_family else 0.0

    score_weights = exit_cfg.get('urbanization_score_weights', {}) or {}
    weight_age = max(0.0, float(score_weights.get('age', 0.30) or 0.30))
    weight_education = max(0.0, float(score_weights.get('education', 0.25) or 0.25))
    weight_income = max(0.0, float(score_weights.get('income_gap', 0.30) or 0.30))
    weight_social = max(0.0, float(score_weights.get('social_link', 0.15) or 0.15))

    intercept = float(urban_cfg.get('intercept', self.intercept) or self.intercept)
    coef_income_gap = float(urban_cfg.get('coef_income_gap', self.coef_income_gap) or self.coef_income_gap)
    coef_non_farm_ratio = float(urban_cfg.get('coef_non_farm_ratio', self.coef_non_farm_ratio) or self.coef_non_farm_ratio)
    coef_age = float(urban_cfg.get('coef_age', self.coef_age) or self.coef_age)
    coef_education = float(urban_cfg.get('coef_education', self.coef_education) or self.coef_education)
    coef_family_urban = float(urban_cfg.get('coef_family_urban', self.coef_family_urban) or self.coef_family_urban)
    coef_scale_lock = float(urban_cfg.get('coef_scale_lock', 0.15) or 0.15)
    professional_scale_anchor = max(1.0, float(urban_cfg.get('professional_scale_anchor', 55.0) or 55.0))
    operated_area = max(0.0, float(get_farmer_operated_land_area(agent) or 0.0))
    scale_lock = min(1.0, operated_area / professional_scale_anchor) * max(0.0, agri_ratio)

    z = (
        intercept
        + coef_income_gap * weight_income * income_gap
        + coef_non_farm_ratio * non_farm_ratio
        + coef_age * weight_age * (1.0 - age_factor)
        + coef_education * weight_education * edu_factor
        + coef_family_urban * weight_social * family_factor
        - coef_scale_lock * scale_lock
    )

    prob = 1.0 / (1.0 + np.exp(-z))
    min_probability = float(urban_cfg.get('min_probability', 0.0) or 0.0)
    max_probability = float(urban_cfg.get('max_probability', 1.0) or 1.0)
    if max_probability < min_probability:
        max_probability = min_probability
    return min(max(prob, min_probability), max_probability)


UrbanizationModel.calculate_probability = _urbanization_model_calculate_probability


class PopulationManagerV2:
    """
    人口动态管理器 V2
    
    核心设计:
    1. 宏观锚定: 总人口跟随NBS统计趋势
    2. 土地守恒: 退出土地必须流转，不能消失
    3. 微观选择: 基于概率模型决定谁退出/进入
    """
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 feature_library_path: Optional[str] = None,
                 region: str = 'national',
                 initial_total_land: Optional[float] = None):
        """
        初始化
        
        Args:
            config_path: NBS宏观数据配置路径
            feature_library_path: CFPS特征库路径
            region: 区域类型
            initial_total_land: 初始总土地量(亩)，用于守恒检查
        """
        self.region = region
        self.land_bank = LandBank()
        self.urbanization_model = UrbanizationModel()
        self.model = None
        
        # 加载宏观数据
        self.macro_data = self._load_macro_data(config_path)
        
        # 加载特征库（用于Bootstrap新进入者）
        self.feature_library = self._load_feature_library(feature_library_path)

        # 运行状态
        self.base_year = self._resolve_base_year()
        self.current_year = self.base_year
        self.base_n_agents = None  # 基准年Agent数量
        
        # 土地守恒
        self.initial_total_land = initial_total_land
        self.land_conservation_log = []
        
        # 统计追踪
        self.annual_stats = []

        logger.info(f"PopulationManagerV2初始化: region={region}")

    def _resolve_base_year(self) -> int:
        """Keep the population module anchored to the configured macro base year."""
        params = self.macro_data.get('derived_parameters', {}) if isinstance(self.macro_data, dict) else {}
        try:
            return int(params.get('base_year', 2010) or 2010)
        except (TypeError, ValueError):
            return 2010
    
    def _load_macro_data(self, config_path: Optional[str]) -> Dict:
        """加载NBS宏观数据"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'nbs_rural_population.json'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"宏观数据未找到: {config_path}")
            return self._get_default_macro_data()
    
    def _get_default_macro_data(self) -> Dict:
        """默认宏观数据"""
        return {
            'derived_parameters': {
                'base_year': 2010,
                'base_households_100m': 2.19,
                'average_annual_decline_rate': -0.0265
            },
            'annual_data': {}
        }
    
    def _load_feature_library(self, path: Optional[str]) -> Optional[Dict]:
        """加载CFPS特征库"""
        project_root = Path(__file__).parent.parent
        fitting_library = (
            project_root
            / 'data'
            / 'derived_runtime'
            / 'v4_8'
            / 'cfps_entrant_feature_library_2010_2016.json'
        )
        candidate_paths = []
        if path is not None:
            candidate_paths.append(Path(path))
        if fitting_library not in candidate_paths:
            candidate_paths.append(fitting_library)
        expected_entrant_pools = {
            'return_migrant',
            'family_farm',
            'young_household',
            'cooperative',
        }
        fitting_years = {2010, 2012, 2014, 2016}
        holdout_years = {2018, 2020, 2022}
        self.feature_library_source_path = None

        for candidate in candidate_paths:
            if candidate is None or not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception:
                continue

            if isinstance(payload, dict):
                metadata = payload.get('metadata', {})
                years_used = {
                    int(year)
                    for year in metadata.get('years_used', [])
                    if str(year).strip()
                }
                runtime_compliant = (
                    bool(years_used)
                    and years_used.issubset(fitting_years)
                    and years_used.isdisjoint(holdout_years)
                    and metadata.get('runtime_partition') == 'fitting_period_feature_pool'
                    and metadata.get('allowed_in_entrant_sampling') is True
                )
                if not runtime_compliant:
                    logger.warning(
                        "Rejected entrant feature library %s because its runtime partition or donor years are not fitting-only",
                        candidate,
                    )
                    continue
                available_pools = {
                    key
                    for key, value in payload.items()
                    if key != 'metadata' and isinstance(value, list) and value and isinstance(value[0], dict)
                }
                if available_pools & expected_entrant_pools:
                    self.feature_library_source_path = candidate.resolve()
                    return payload
                if available_pools:
                    logger.warning(
                        "Rejected entrant feature library %s because it exposes pools %s rather than entrant pools %s",
                        candidate,
                        sorted(available_pools),
                        sorted(expected_entrant_pools),
                    )

        return None
    
    def initialize(self, agents: List[Any]) -> None:
        """
        初始化人口管理器
        
        Args:
            agents: 初始Agent列表
        """
        self.base_n_agents = len(agents)
        self.model = getattr(agents[0], 'model', None) if agents else None
        self.land_bank.random_context = self.model or self
        
        # 计算初始总土地
        if self.initial_total_land is None:
            if self.model is not None and getattr(self.model, 'land_patches', None):
                self.initial_total_land = float(sum(
                    getattr(patch, 'area', 0.0) or 0.0
                    for patch in self.model.land_patches
                ))
            else:
                self.initial_total_land = sum(
                    self._get_agent_owned_land_area(a) for a in agents
                )
        
        logger.info(f"PopulationManager初始化: {self.base_n_agents}户, "
                   f"总土地={self.initial_total_land:.0f}亩")

    def _get_population_config(self) -> Dict:
        model = getattr(self, 'model', None)
        if model is not None and hasattr(model, 'config'):
            return model.config.get('population_dynamics', {}) or {}
        return {}

    def _get_farmer_entry_preferences(self) -> Dict:
        if self.model is not None and hasattr(self.model, 'config'):
            return (
                self.model.config.get('agents', {})
                .get('farmer', {})
                .get('livelihood_evolution', {})
                .get('farmer_entry', {})
                or {}
            )
        return {}

    def _is_farmer_entry_enabled(self) -> bool:
        """Whether replacement-entry households are allowed to absorb land-bank parcels."""
        pop_cfg = self._get_population_config()
        entry_cfg = pop_cfg.get('entry_mechanism', {}) or {}
        farmer_entry_cfg = self._get_farmer_entry_preferences()
        return bool(farmer_entry_cfg.get('enabled', entry_cfg.get('enabled', True)))

    def _get_farmer_livelihood_evolution_config(self) -> Dict:
        if self.model is not None and hasattr(self.model, 'config'):
            return (
                self.model.config.get('agents', {})
                .get('farmer', {})
                .get('livelihood_evolution', {})
                or {}
            )
        return {}

    def _get_entry_age_range(self) -> Tuple[int, int]:
        entry_cfg = self._get_farmer_entry_preferences()
        age_range = entry_cfg.get('entry_age_range', [30, 55]) or [30, 55]
        if not isinstance(age_range, (list, tuple)) or len(age_range) != 2:
            return 30, 55
        try:
            age_min = int(age_range[0])
            age_max = int(age_range[1])
        except (TypeError, ValueError):
            return 30, 55
        if age_max < age_min:
            age_max = age_min
        return age_min, age_max

    def _sample_entry_livelihood(self) -> Optional[str]:
        if not self._is_farmer_entry_enabled():
            return None
        entry_cfg = self._get_farmer_entry_preferences()
        distribution = entry_cfg.get('entry_strategy_distribution', {}) or {}
        labels = ['Pure_Farmer', 'Pluriactive', 'Non_Farmer']
        weights = [max(0.0, float(distribution.get(label, 0.0) or 0.0)) for label in labels]
        if sum(weights) <= 0:
            return None
        return str(rng_choice(self, labels, weights=weights))

    def _get_agent_owned_land_area(self, agent: Any) -> float:
        if isinstance(agent, dict):
            parcels = agent.get('land_parcels') or []
            parcel_area = float(sum(
                float(getattr(parcel, 'area_mu', 0.0) or 0.0)
                for parcel in parcels
            ))
            if parcel_area > 0:
                return parcel_area
            return float(agent.get('land_area_mu', agent.get('land_area', 0.0)) or 0.0)

        if hasattr(agent, 'sync_owned_land_area'):
            try:
                return float(agent.sync_owned_land_area())
            except Exception:
                pass
        if hasattr(agent, 'get_owned_land_area'):
            try:
                return float(agent.get_owned_land_area(include_leased_out=True))
            except TypeError:
                return float(agent.get_owned_land_area())
            except Exception:
                pass
        if hasattr(agent, 'land_patches'):
            try:
                return float(sum(
                    getattr(patch, 'area', 0.0)
                    for patch in getattr(agent, 'land_patches', [])
                    if getattr(patch, 'owner', None) == agent
                ))
            except Exception:
                pass
        return float(getattr(agent, 'land_area_mu', 0.0) or 0.0)

    def _get_total_land_balance(self,
                                agents: List[Any],
                                enterprises: Optional[List[Any]] = None) -> float:
        total_land = sum(self._get_agent_owned_land_area(agent) for agent in agents)
        total_land += float(self.land_bank.total_area_mu or 0.0)

        if enterprises:
            total_land += sum(self._get_agent_owned_land_area(ent) for ent in enterprises)

        active_agents = agents + list(enterprises or [])
        for patch in getattr(getattr(self, 'model', None), 'land_patches', []):
            if (
                hasattr(patch, 'initial_rental_rights_holder')
                and patch.owner not in active_agents
                and patch.current_operator in active_agents
            ):
                total_land += float(patch.area)

        return float(total_land)

    def _plan_entry_count(self, agents: List[Any], net_exit_needed: int) -> int:
        if not self._is_farmer_entry_enabled():
            return 0
        pop_cfg = self._get_population_config()
        entry_cfg = pop_cfg.get('entry_mechanism', {}) or {}
        farmer_entry_cfg = self._get_farmer_entry_preferences()
        entry_prob = float(
            farmer_entry_cfg.get(
                'annual_entry_rate',
                entry_cfg.get('new_entry_probability', 0.02),
            ) or 0.0
        )
        min_scale = float(entry_cfg.get('min_scale_farming_size', 15.0) or 15.0)
        baseline_churn_rate = float(
            farmer_entry_cfg.get(
                'baseline_replacement_churn_rate',
                entry_cfg.get('baseline_replacement_churn_rate', 0.0),
            ) or 0.0
        )
        replacement_share = float(
            farmer_entry_cfg.get(
                'max_replacement_share',
                entry_cfg.get('max_replacement_share', 0.35),
            ) or 0.35
        )
        min_farmer_count = int(farmer_entry_cfg.get('min_farmer_count', 0) or 0)

        if replacement_share <= 0 or (entry_prob <= 0 and baseline_churn_rate <= 0):
            return 0

        current_n = len(agents)
        development_entries = int(round(current_n * entry_prob)) if net_exit_needed > 0 else 0
        # Baseline replacement churn is not tied to net population decline. It
        # represents generational succession/return migrants replacing retiring
        # operators, so it must still operate when the active-farmer headcount is
        # stable.
        churn_entries = int(round(current_n * baseline_churn_rate))
        desired_entries = development_entries + churn_entries
        projected_post_exit = max(0, current_n - int(net_exit_needed))
        if min_farmer_count > 0 and projected_post_exit < min_farmer_count:
            desired_entries = max(desired_entries, min_farmer_count - projected_post_exit)
        if desired_entries <= 0:
            return 0

        owned_areas = [self._get_agent_owned_land_area(agent) for agent in agents if self._get_agent_owned_land_area(agent) > 0]
        avg_owned_area = float(np.mean(owned_areas)) if owned_areas else min_scale
        effective_exit_pool = max(0, int(net_exit_needed)) + churn_entries
        estimated_available_land = max(0.0, avg_owned_area * effective_exit_pool)
        if estimated_available_land < min_scale:
            if churn_entries > 0 and estimated_available_land >= min_scale * 0.5:
                max_by_land = 1
            else:
                return 0
        else:
            max_by_land = int(estimated_available_land / max(min_scale, 1.0))
        development_cap = max(0, int(np.ceil(max(0, net_exit_needed) * replacement_share)))
        capped_development = min(development_entries, development_cap, max_by_land)
        remaining_land_capacity = max(0, max_by_land - capped_development)
        capped_churn = min(churn_entries, remaining_land_capacity)
        return max(0, capped_development + capped_churn)

    def _get_entry_exit_linkage_share(self) -> float:
        """Only part of replacement entry should mechanically induce incumbent exit."""
        pop_cfg = self._get_population_config()
        entry_cfg = pop_cfg.get('entry_mechanism', {}) or {}
        linkage_share = float(entry_cfg.get('entry_exit_linkage_share', 0.35) or 0.35)
        return max(0.0, min(1.0, linkage_share))

    def _resolve_macro_anchor_ratio(self, year: int) -> float:
        """Resolve the macro anchor ratio used to size the rural household pool."""
        annual_data = self.macro_data.get('annual_data', {})
        params = self.macro_data.get('derived_parameters', {})
        pop_cfg = self._get_population_config()
        anchor_cfg = pop_cfg.get('macro_anchor', {}) or {}

        base_year = int(anchor_cfg.get('base_year', params.get('base_year', 2010)) or 2010)
        base_row = annual_data.get(str(base_year), {}) or {}
        monotonic_smoothing = bool(anchor_cfg.get('monotonic_smoothing', True))

        base_households = float(
            base_row.get('rural_households_100m', params.get('base_households_100m', 2.19)) or 2.19
        )
        base_population = float(base_row.get('rural_population_100m', 0.0) or 0.0)
        decline_rate = float(
            anchor_cfg.get('average_annual_decline_rate', params.get('average_annual_decline_rate', -0.0265))
            or -0.0265
        )

        anchor_metric = str(
            anchor_cfg.get('metric', anchor_cfg.get('anchor_metric', 'households')) or 'households'
        ).strip().lower()
        household_weight = max(0.0, float(anchor_cfg.get('household_weight', 0.4) or 0.4))
        population_weight = max(0.0, float(anchor_cfg.get('population_weight', 0.6) or 0.6))
        weight_sum = household_weight + population_weight
        if weight_sum > 0:
            household_weight /= weight_sum
            population_weight /= weight_sum
        else:
            household_weight = 0.5
            population_weight = 0.5

        def _row_value(target_year: int, key: str, base_value: float) -> float:
            row = annual_data.get(str(target_year), {}) or {}
            raw_value = row.get(key, None)
            if raw_value is not None:
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    pass

            # If the requested year is beyond the observed macro series, extend
            # from the latest available observation instead of jumping back to a
            # base-year exponential path. The previous base-year fallback created
            # an artificial 2022->2023 population drop when NBS data stopped at
            # 2022.
            latest_year = None
            latest_value = None
            for year_key, year_row in annual_data.items():
                try:
                    candidate_year = int(year_key)
                except (TypeError, ValueError):
                    continue
                if candidate_year >= int(target_year):
                    continue
                candidate_value = (year_row or {}).get(key, None)
                if candidate_value is None:
                    continue
                try:
                    candidate_value = float(candidate_value)
                except (TypeError, ValueError):
                    continue
                if latest_year is None or candidate_year > latest_year:
                    latest_year = candidate_year
                    latest_value = candidate_value
            if latest_year is not None and latest_value is not None:
                years_diff = max(0, int(target_year) - latest_year)
                return float(latest_value) * ((1 + decline_rate) ** years_diff)

            years_diff = max(0, int(target_year) - base_year)
            return float(base_value) * ((1 + decline_rate) ** years_diff)

        def _raw_ratio(target_year: int) -> float:
            target_households = _row_value(target_year, 'rural_households_100m', base_households)
            household_ratio = target_households / max(base_households, 1e-6)

            population_ratio = household_ratio
            if base_population > 0:
                target_population = _row_value(target_year, 'rural_population_100m', base_population)
                population_ratio = target_population / max(base_population, 1e-6)

            if anchor_metric == 'population':
                ratio = population_ratio
            elif anchor_metric in {'hybrid', 'hybrid_rural_presence'}:
                ratio = household_weight * household_ratio + population_weight * population_ratio
            elif anchor_metric in {'min_household_population', 'min_household_population_ratio'}:
                ratio = min(household_ratio, population_ratio)
            else:
                ratio = household_ratio
            return max(0.05, float(ratio))

        ratio = _raw_ratio(year)
        if monotonic_smoothing and year > base_year:
            for target_year in range(base_year, int(year) + 1):
                ratio = min(ratio, _raw_ratio(target_year))

        census_cfg = anchor_cfg.get('census_shock_smoothing', {}) or {}
        if bool(census_cfg.get('enabled', False)):
            start_year = int(census_cfg.get('start_year', 2021) or 2021)
            end_year = int(census_cfg.get('end_year', 2022) or 2022)
            release_year = int(census_cfg.get('release_year', end_year) or end_year)
            floor_year = int(census_cfg.get('floor_year', start_year - 1) or (start_year - 1))
            floor_multiplier = max(
                0.0,
                float(census_cfg.get('floor_multiplier', 1.0) or 1.0),
            )
            if start_year <= int(year) <= max(end_year, release_year):
                # The 2020 census introduced a denominator discontinuity in the
                # rural population/household series. Active farm operators should
                # not be forced to absorb that statistical break within the same
                # two-year validation window.
                census_floor = _raw_ratio(floor_year) * floor_multiplier
                if int(year) > end_year and release_year > end_year:
                    release_weight = (
                        (release_year - int(year))
                        / max(1, release_year - end_year)
                    )
                    release_weight = max(0.0, min(1.0, float(release_weight)))
                    census_floor = release_weight * census_floor + (1.0 - release_weight) * ratio
                ratio = max(ratio, census_floor)

        return max(0.05, ratio)

    def _map_macro_ratio_to_active_farmer_ratio(self, macro_ratio: float) -> float:
        """Map rural macro anchors to active-farmer targets while preserving legacy behavior."""
        ratio = max(0.05, float(macro_ratio or 0.0))
        anchor_cfg = self._get_population_config().get('macro_anchor', {}) or {}
        mapping = str(anchor_cfg.get('active_farmer_mapping', 'direct') or 'direct').strip().lower()

        if mapping not in {'softened', 'soften', 'elastic_softened'}:
            return ratio
        if ratio >= 1.0:
            return ratio

        floor_ratio = max(0.0, min(1.0, float(anchor_cfg.get('active_farmer_floor_ratio', 0.0) or 0.0)))
        decline_elasticity = max(
            1.0e-6,
            float(anchor_cfg.get('active_farmer_decline_elasticity', 1.0) or 1.0),
        )
        softened_ratio = floor_ratio + (1.0 - floor_ratio) * (ratio ** decline_elasticity)
        return max(floor_ratio, float(softened_ratio))

    def _get_farmer_livelihood_strategy_name(self, agent: Any, default: str = "") -> str:
        if _income_utils_get_farmer_livelihood_strategy_name is not None:
            try:
                return _income_utils_get_farmer_livelihood_strategy_name(agent, default=default)
            except Exception:
                pass

        if isinstance(agent, dict):
            strategy = agent.get('livelihood') or agent.get('livelihood_strategy')
        else:
            strategy = getattr(getattr(agent, 'livelihood_strategy', None), 'value', None)
            if strategy is None:
                strategy = getattr(getattr(agent, 'livelihood_strategy', None), 'name', None)
            if strategy is None:
                strategy = getattr(agent, 'livelihood_strategy', None)
        if strategy is not None:
            strategy_name = str(strategy).strip()
            if strategy_name:
                return strategy_name

        agri_ratio = get_farmer_livelihood_agriculture_ratio(agent, default=0.5)
        if agri_ratio >= 0.5:
            return 'Pure_Farmer'
        if agri_ratio <= 0.2:
            return 'Non_Farmer'
        return 'Pluriactive'

    def _get_agent_operated_land_area(self, agent: Any) -> float:
        if isinstance(agent, dict):
            parcels = agent.get('land_parcels') or []
            parcel_area = float(sum(float(getattr(parcel, 'area_mu', 0.0) or 0.0) for parcel in parcels))
            if parcel_area > 0:
                return parcel_area
            return float(agent.get('land_area_mu', agent.get('land_area', 0.0)) or 0.0)
        try:
            return float(get_farmer_operated_land_area(agent) or 0.0)
        except Exception:
            return self._get_agent_owned_land_area(agent)

    def _get_professional_retention_config(self) -> Dict:
        evolution_cfg = self._get_farmer_livelihood_evolution_config()
        return evolution_cfg.get('professional_farmer_retention', {}) or {}

    def _is_professional_large_operator(self, agent: Any) -> bool:
        retention_cfg = self._get_professional_retention_config()
        kwargs = {
            'min_operated_area': float(retention_cfg.get('min_operated_area', 30.0) or 30.0),
            'min_agricultural_income_ratio': float(
                retention_cfg.get('min_agricultural_income_ratio', 0.78) or 0.78
            ),
            'max_off_farm_income': float(retention_cfg.get('max_off_farm_income', 12000.0) or 12000.0),
        }
        if _income_utils_is_professional_large_operator is not None:
            try:
                return bool(_income_utils_is_professional_large_operator(agent, **kwargs))
            except Exception:
                pass

        if self._get_farmer_livelihood_strategy_name(agent) == 'Non_Farmer':
            return False

        operated_area = self._get_agent_operated_land_area(agent)
        off_farm_income = max(0.0, float(getattr(agent, 'off_farm_income', 0.0) or 0.0))
        agri_ratio = get_farmer_livelihood_agriculture_ratio(agent, default=0.5)
        return (
            operated_area >= kwargs['min_operated_area']
            and agri_ratio >= kwargs['min_agricultural_income_ratio']
            and off_farm_income <= kwargs['max_off_farm_income']
        )

    def _get_service_readiness_score(self, agent: Any) -> float:
        if isinstance(agent, dict):
            service_contracts = agent.get('service_contracts') or []
            labor_reduction_ratio = float(agent.get('labor_reduction_ratio', 0.0) or 0.0)
            prefers_cooperative_custody = bool(agent.get('prefers_cooperative_custody', False))
            characteristics = agent.get('characteristics', None)
            service_policy_sensitivity = float(agent.get('service_policy_sensitivity', 0.0) or 0.0)
        else:
            service_contracts = getattr(agent, 'service_contracts', []) or []
            labor_reduction_ratio = float(getattr(agent, 'labor_reduction_ratio', 0.0) or 0.0)
            prefers_cooperative_custody = bool(getattr(agent, 'prefers_cooperative_custody', False))
            characteristics = getattr(agent, 'characteristics', None)
            service_policy_sensitivity = float(
                getattr(characteristics, 'service_policy_sensitivity', 0.0) or 0.0
            )

        service_score = max(0.0, min(1.0, labor_reduction_ratio))
        if isinstance(service_contracts, list) and service_contracts:
            service_score = max(service_score, min(1.0, 0.20 + 0.18 * len(service_contracts)))
        if prefers_cooperative_custody:
            service_score = max(service_score, 0.75)

        technology_adoption = float(getattr(characteristics, 'technology_adoption', 0.0) or 0.0)
        information_access = float(getattr(characteristics, 'information_access', 0.0) or 0.0)
        support_readiness = max(technology_adoption, information_access, service_policy_sensitivity)
        return min(1.0, max(service_score, support_readiness))

    def _get_farmer_ability_score(self, agent: Any) -> float:
        if isinstance(agent, dict):
            farming_ability = float(agent.get('farming_ability', 1.0) or 1.0)
            technology_adoption = float(
                agent.get('technology_adoption', agent.get('tech_adoption', 0.0)) or 0.0
            )
            information_access = float(agent.get('information_access', 0.0) or 0.0)
            social_network_strength = float(agent.get('social_network_strength', 0.0) or 0.0)
        else:
            characteristics = getattr(agent, 'characteristics', None)
            farming_ability = float(getattr(agent, 'farming_ability', 1.0) or 1.0)
            technology_adoption = float(getattr(characteristics, 'technology_adoption', 0.0) or 0.0)
            information_access = float(getattr(characteristics, 'information_access', 0.0) or 0.0)
            social_network_strength = float(getattr(agent, 'social_network_strength', 0.0) or 0.0)

        normalized_farming_ability = max(0.0, min(1.0, (farming_ability - 0.8) / 0.5))
        normalized_social_capital = max(0.0, min(1.0, social_network_strength / 1.5))
        return min(
            1.0,
            0.55 * normalized_farming_ability
            + 0.25 * max(0.0, min(1.0, technology_adoption))
            + 0.10 * max(0.0, min(1.0, information_access))
            + 0.10 * normalized_social_capital,
        )

    def _build_land_successor_tiers(
        self,
        agents: List[Any],
        enterprises: Optional[List[Any]] = None,
    ) -> Dict[str, List[Any]]:
        used_agent_ids = set()
        retention_cfg = self._get_professional_retention_config()
        min_farming_ability = float(retention_cfg.get('min_farming_ability', 1.0) or 1.0)

        professional_candidates = []
        service_supported_candidates = []
        general_candidates = []
        fallback_candidates = []

        for agent in agents:
            strategy_name = self._get_farmer_livelihood_strategy_name(agent)
            operated_area = self._get_agent_operated_land_area(agent)
            agri_ratio = get_farmer_livelihood_agriculture_ratio(agent, default=0.5)
            service_score = self._get_service_readiness_score(agent)
            ability_score = self._get_farmer_ability_score(agent)
            farming_ability = float(getattr(agent, 'farming_ability', 1.0) or 1.0)
            prefers_cooperative_custody = bool(getattr(agent, 'prefers_cooperative_custody', False))
            is_weekend_farmer = bool(getattr(agent, 'is_weekend_farmer', False))

            if self._is_professional_large_operator(agent):
                professional_candidates.append(
                    (agent, ability_score + service_score, operated_area, agri_ratio)
                )
                continue

            if (
                strategy_name != 'Non_Farmer'
                and (
                    service_score >= 0.35
                    or prefers_cooperative_custody
                    or is_weekend_farmer
                    or (ability_score >= 0.55 and farming_ability >= min_farming_ability)
                )
            ):
                service_supported_candidates.append(
                    (agent, service_score, ability_score, operated_area, agri_ratio)
                )
                continue

            if (
                strategy_name == 'Non_Farmer'
                and agri_ratio < 0.25
                and service_score < 0.55
                and ability_score < 0.60
            ):
                fallback_candidates.append(
                    (agent, ability_score, service_score, agri_ratio, operated_area)
                )
                continue

            general_candidates.append(
                (agent, ability_score, service_score, agri_ratio, operated_area)
            )

        professional_tier = []
        for agent, *_ in sorted(
            professional_candidates,
            key=lambda item: (item[1], item[2], item[3]),
            reverse=True,
        ):
            professional_tier.append(agent)
            used_agent_ids.add(id(agent))

        service_supported_tier = []
        for agent, *_ in sorted(
            service_supported_candidates,
            key=lambda item: (item[1], item[2], item[3], item[4]),
            reverse=True,
        ):
            if id(agent) in used_agent_ids:
                continue
            service_supported_tier.append(agent)
            used_agent_ids.add(id(agent))

        general_tier = []
        for agent, *_ in sorted(
            general_candidates,
            key=lambda item: (item[1], item[2], item[3], item[4]),
            reverse=True,
        ):
            if id(agent) in used_agent_ids:
                continue
            general_tier.append(agent)
            used_agent_ids.add(id(agent))

        fallback_tier = []
        for agent, *_ in sorted(
            fallback_candidates,
            key=lambda item: (item[1], item[2], item[3], item[4]),
            reverse=True,
        ):
            if id(agent) in used_agent_ids:
                continue
            fallback_tier.append(agent)
            used_agent_ids.add(id(agent))

        cooperative_tier = []
        corporation_tier = []
        for enterprise in enterprises or []:
            enterprise_type_name = str(
                getattr(getattr(enterprise, 'enterprise_type', None), 'name', getattr(enterprise, 'enterprise_type', ''))
            ).upper()
            if 'COOPERATIVE' in enterprise_type_name:
                cooperative_tier.append(enterprise)
            else:
                corporation_tier.append(enterprise)

        cooperative_tier.sort(
            key=lambda ent: (
                float(getattr(ent, 'efficiency_factor', 1.0) or 1.0),
                float(getattr(ent, 'capital', 0.0) or 0.0),
            ),
            reverse=True,
        )
        corporation_tier.sort(
            key=lambda ent: (
                float(getattr(ent, 'efficiency_factor', 1.0) or 1.0),
                float(getattr(ent, 'capital', 0.0) or 0.0),
            ),
            reverse=True,
        )

        return {
            'professional_large_operators': professional_tier,
            'service_supported_households': service_supported_tier,
            'service_ready_pluriactive': service_supported_tier,
            'general_farmers': general_tier,
            'cooperatives': cooperative_tier,
            'agri_corporations': corporation_tier,
            'enterprises': list(enterprises or []),
            'fallback_households': fallback_tier,
        }

    def _get_feasible_expansion_demand_area(self, agent: Any) -> float:
        if isinstance(agent, dict):
            current_area = self._get_agent_operated_land_area(agent)
            target_area = max(
                0.0,
                float(agent.get('target_operated_area', current_area) or 0.0),
            )
            return max(0.0, target_area - current_area)

        fixed_action = str(getattr(agent, 'fixed_rule_current_action', '') or '')
        if fixed_action in {'contract', 'exit'} or getattr(agent, 'fixed_rule_participating', True) is False:
            return 0.0

        current_area = self._get_agent_operated_land_area(agent)
        if hasattr(agent, 'enterprise_type'):
            desired_area = max(0.0, float(getattr(agent, 'target_expansion_area', 0.0) or 0.0))
            willingness = max(0.0, float(getattr(agent, 'expansion_willingness', 0.0) or 0.0))
            if desired_area <= 0.0 or willingness <= 0.3:
                return 0.0
            scale_fn = getattr(agent, 'calculate_optimal_operated_scale', None)
            if not callable(scale_fn):
                return 0.0
            scale_info = scale_fn()
            feasible_gap = max(0.0, float(scale_info.get('scale_gap', 0.0) or 0.0))
            return min(desired_area, feasible_gap)

        target_area = max(0.0, float(getattr(agent, 'target_operated_area', current_area) or 0.0))
        desired_area = max(0.0, target_area - current_area)
        willingness_fn = getattr(agent, 'get_transfer_in_willingness', None)
        if desired_area <= 0.0 or not callable(willingness_fn):
            return 0.0
        willingness = max(0.0, float(willingness_fn() or 0.0))
        if willingness <= 0.3:
            return 0.0

        capital_fn = getattr(agent, 'get_farmer_available_capital_for_expansion', None)
        if not callable(capital_fn):
            return 0.0
        available_capital = max(0.0, float(capital_fn() or 0.0))
        if available_capital <= 0.0:
            return 0.0
        production_cost = max(0.0, float(getattr(agent, 'production_cost', 0.0) or 0.0))
        if production_cost > 0.0 and current_area > 0.0:
            desired_area = min(desired_area, available_capital / (production_cost / current_area))

        labor_profile_fn = getattr(agent, 'estimate_production_labor_profile', None)
        if not callable(labor_profile_fn):
            return 0.0
        labor_profile = labor_profile_fn(area=current_area + desired_area)
        labor_days_per_mu = max(0.0, float(labor_profile.get('labor_days_per_mu', 0.0) or 0.0))
        available_family_labor = max(
            0.0,
            float(labor_profile.get('available_family_labor', 0.0) or 0.0),
        )
        service_substitution = max(
            0.0,
            min(0.8, float(labor_profile.get('service_substitution_rate', 0.0) or 0.0)),
        )
        effective_labor_days_per_mu = labor_days_per_mu * (1.0 - service_substitution)
        if effective_labor_days_per_mu > 0.0:
            labor_capacity_area = available_family_labor / effective_labor_days_per_mu
            desired_area = min(desired_area, max(0.0, labor_capacity_area - current_area))
        return max(0.0, desired_area)

    def _get_land_matching_capacity_scale(self) -> float:
        raw = self._get_population_config().get('land_matching_capacity_scale', 1.0)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError('land matching capacity scale must be numeric') from exc
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError('land matching capacity scale must be in (0, 1]')
        return value

    def _get_expected_land_return_yuan_per_mu(self, agent: Any) -> float:
        if isinstance(agent, dict):
            expected_return = float(
                agent.get('expected_land_return_yuan_per_mu', 0.0) or 0.0
            )
            return expected_return if math.isfinite(expected_return) else 0.0

        current_area = self._get_agent_operated_land_area(agent)
        if current_area <= 0.0:
            return 0.0
        if hasattr(agent, 'enterprise_type'):
            annual_return = float(
                getattr(agent, 'annual_profit', 0.0)
                or getattr(agent, 'profit', 0.0)
                or getattr(agent, 'net_income', 0.0)
                or 0.0
            )
        else:
            annual_return = float(
                getattr(agent, 'planting_net_income', 0.0)
                or getattr(agent, 'agricultural_income', 0.0)
                or 0.0
            )
        return annual_return / current_area if math.isfinite(annual_return) else 0.0

    def _allocate_weighted_released_land(
        self,
        agents: List[Any],
        enterprises: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        released_area_mu = float(self.land_bank.total_area_mu or 0.0)
        candidates = []
        weighted_capacity_by_recipient = {}
        candidate_rows = []
        for candidate in [*agents, *(enterprises or [])]:
            feasible_expansion_mu = self._get_feasible_expansion_demand_area(candidate)
            current_area_mu = self._get_agent_operated_land_area(candidate)
            if hasattr(candidate, 'enterprise_type'):
                target_expansion_mu = max(
                    0.0,
                    float(getattr(candidate, 'target_expansion_area', 0.0) or 0.0),
                )
                willingness = max(0.0, float(getattr(candidate, 'expansion_willingness', 0.0) or 0.0))
                desired_area_mu = current_area_mu + target_expansion_mu if willingness > 0.3 else current_area_mu
                scale_info = candidate.calculate_optimal_operated_scale()
                capacity_limit_area_mu = max(
                    0.0,
                    float(scale_info.get('optimal_scale', current_area_mu) or current_area_mu),
                )
            else:
                desired_area_mu = current_area_mu + feasible_expansion_mu
                capacity_limit_area_mu = max(current_area_mu, current_area_mu + feasible_expansion_mu)
            representation_weight_q = max(
                0.0,
                float(
                    candidate.get('representation_weight_q', 1.0)
                    if isinstance(candidate, dict)
                    else getattr(candidate, 'representation_weight_q', 1.0)
                    or 0.0
                ),
            )
            expected_return_yuan_per_mu = self._get_expected_land_return_yuan_per_mu(candidate)
            weighted_capacity_mu = representation_weight_q * feasible_expansion_mu
            capacity_row = _capacity_diagnostic_row(
                operator_id=(
                    candidate.get('unique_id', 'unknown')
                    if isinstance(candidate, dict)
                    else getattr(candidate, 'unique_id', 'unknown')
                ),
                current_area_mu=current_area_mu,
                desired_area_mu=desired_area_mu,
                capacity_limit_area_mu=capacity_limit_area_mu,
                allocated_area_mu=0.0,
            )
            eligible = (
                math.isfinite(representation_weight_q)
                and math.isfinite(weighted_capacity_mu)
                and expected_return_yuan_per_mu > 0.0
                and weighted_capacity_mu > 0.0
            )
            candidate_rows.append(
                {
                    'year': int(self.current_year),
                    'operator_id': str(
                        candidate.get('unique_id', 'unknown')
                        if isinstance(candidate, dict)
                        else getattr(candidate, 'unique_id', 'unknown')
                    ),
                    'representation_weight_q': representation_weight_q,
                    'feasible_expansion_mu': feasible_expansion_mu,
                    'weighted_capacity_mu': weighted_capacity_mu,
                    'expected_return_yuan_per_mu': expected_return_yuan_per_mu,
                    'eligible_bid': eligible,
                    'area_unit': 'mu',
                    **capacity_row,
                }
            )
            if eligible:
                candidates.append(candidate)
                weighted_capacity_by_recipient[id(candidate)] = weighted_capacity_mu

        candidates.sort(
            key=lambda candidate: (
                -self._get_expected_land_return_yuan_per_mu(candidate),
                str(
                    candidate.get('unique_id', 'unknown')
                    if isinstance(candidate, dict)
                    else getattr(candidate, 'unique_id', 'unknown')
                ),
            )
        )
        allocation_stats = self.land_bank.distribute_to_neighbors(
            candidates,
            max_per_agent=max(weighted_capacity_by_recipient.values(), default=0.0),
            max_total_area=released_area_mu * self._get_land_matching_capacity_scale(),
            max_area_by_recipient=weighted_capacity_by_recipient,
            preserve_order=True,
        )
        allocated_area_mu = float(allocation_stats['total_distributed'])
        allocation_rows = list(allocation_stats['allocations'])
        allocated_by_recipient = {}
        for allocation in allocation_rows:
            operator_id = str(allocation.get('recipient_id', ''))
            allocated_by_recipient[operator_id] = (
                allocated_by_recipient.get(operator_id, 0.0)
                + float(allocation.get('absorbed_area_mu', 0.0) or 0.0)
            )
        for candidate_row in candidate_rows:
            allocated_area = allocated_by_recipient.get(str(candidate_row['operator_id']), 0.0)
            candidate_row['delta_a_allocated_mu'] = allocated_area
        allocated_to_enterprises_mu = float(
            sum(
                row['absorbed_area_mu']
                for row in allocation_rows
                if row['recipient_type'] == 'enterprise'
            )
        )
        allocated_to_new_entrants_mu = float(
            sum(
                row['absorbed_area_mu']
                for row in allocation_rows
                if row['is_new_entrant']
            )
        )
        result = {
            'year': int(self.current_year),
            'area_unit': 'mu',
            'released_area_mu': released_area_mu,
            'eligible_weighted_demand_mu': float(
                sum(weighted_capacity_by_recipient.values())
            ),
            'allocated_area_mu': allocated_area_mu,
            'allocated_to_farmers_mu': allocated_area_mu - allocated_to_enterprises_mu,
            'allocated_to_enterprises_mu': allocated_to_enterprises_mu,
            'allocated_to_new_entrants_mu': allocated_to_new_entrants_mu,
            'unabsorbed_area_mu': float(self.land_bank.total_area_mu or 0.0),
            'capacity_operator_count': len(candidate_rows),
            'capacity_binding_count': sum(bool(row['binding']) for row in candidate_rows),
            'capacity_desired_expansion_mu': float(sum(row['delta_a_desired_mu'] for row in candidate_rows)),
            'capacity_feasible_expansion_mu': float(sum(row['delta_a_feasible_mu'] for row in candidate_rows)),
            'capacity_allocated_expansion_mu': float(sum(row['delta_a_allocated_mu'] for row in candidate_rows)),
            'capacity_gap_mu': float(sum(row['capacity_gap_mu'] for row in candidate_rows)),
            'candidate_trace': candidate_rows,
            'allocation_trace': allocation_rows,
        }
        trace = getattr(self, 'weighted_allocation_trace', None)
        if trace is None:
            self.weighted_allocation_trace = []
            trace = self.weighted_allocation_trace
        trace.extend(candidate_rows)
        self.weighted_allocation_annual_stats = getattr(
            self,
            'weighted_allocation_annual_stats',
            [],
        )
        self.weighted_allocation_annual_stats.append(
            {
                key: value
                for key, value in result.items()
                if key not in {'candidate_trace', 'allocation_trace'}
            }
        )
        return result

    def _get_farmer_entry_floor(self) -> int:
        """Return the structural floor for active farming households."""
        if not self._is_farmer_entry_enabled():
            return 0
        farmer_entry_cfg = self._get_farmer_entry_preferences()
        configured_min_count = int(farmer_entry_cfg.get('min_farmer_count', 0) or 0)
        min_farmer_ratio = float(farmer_entry_cfg.get('min_farmer_ratio', 0.0) or 0.0)
        ratio_floor = 0
        if self.base_n_agents is not None and min_farmer_ratio > 0:
            ratio_floor = int(round(self.base_n_agents * max(0.0, min(1.0, min_farmer_ratio))))
        if self.base_n_agents is None:
            absolute_floor = configured_min_count
        elif configured_min_count > self.base_n_agents:
            absolute_floor = 0
        else:
            # `min_farmer_count` is calibrated for the 1000-household production
            # scale. Small paired-worldline/smoke runs must scale by the ratio
            # floor instead of trying to expand a 100-household model to 760.
            absolute_floor = configured_min_count
        return max(0, absolute_floor, ratio_floor)

    def _get_successor_absorption_config(self) -> Dict[str, float]:
        """Resolve explicit successor absorption settings for exit-land reallocation."""
        farmer_entry_cfg = self._get_farmer_entry_preferences()
        absorption_cfg = farmer_entry_cfg.get('successor_absorption', {}) or {}
        return absorption_cfg if isinstance(absorption_cfg, dict) else {}

    def _get_enterprise_operational_rent_proxy(self) -> float:
        model = getattr(self, 'model', None)
        config = getattr(model, 'config', {}) if model is not None else {}
        transfer_cfg = (
            config.get('agents', {})
            .get('farmer', {})
            .get('land_transfer', {})
            if isinstance(config, dict)
            else {}
        )
        return float(transfer_cfg.get('base_rent_price', 420.0) or 420.0)
    
    def get_target_n_agents(self, year: int) -> int:
        """
        计算目标年份的Agent数量
        
        公式: N_model(t) = N_base × (Pop_stats(t) / Pop_stats(2010))
        """
        if self.base_n_agents is None:
            return 1000  # 默认值
        
        macro_ratio = self._resolve_macro_anchor_ratio(year)
        active_farmer_ratio = self._map_macro_ratio_to_active_farmer_ratio(macro_ratio)
        target_n = int(round(self.base_n_agents * active_farmer_ratio))

        # Do not force active farming households to shrink one-for-one with the
        # macro rural-household anchor; keep an explicit managed-operator floor.
        return max(target_n, self._get_farmer_entry_floor(), 50)  # 最少保留50户
    
    def step(self, 
             current_year: int, 
             agents: List[Any],
             enterprises: List[Any] = None) -> Tuple[List[Any], Dict]:
        """
        执行年度人口动态演化
        
        Args:
            current_year: 当前模拟年份
            agents: 当前农户列表
            enterprises: 企业列表（用于接收土地）
        
        Returns:
            更新后的农户列表, 年度统计
        """
        self.current_year = current_year
        self.land_bank.current_year = current_year
        
        if self.base_n_agents is None:
            self.initialize(agents)
        
        # 计算当前总土地（守恒检查）
        current_total_land = self._get_total_land_balance(agents, enterprises)
        macro_anchor_ratio = self._resolve_macro_anchor_ratio(current_year)
        active_farmer_target_ratio = self._map_macro_ratio_to_active_farmer_ratio(macro_anchor_ratio)
        
        stats = {
            'year': current_year,
            'initial_n': len(agents),
            'target_n': self.get_target_n_agents(current_year),
            'macro_anchor_ratio': macro_anchor_ratio,
            'active_farmer_target_ratio': active_farmer_target_ratio,
            'exits': {'death': 0, 'urbanization': 0},
            'entries': {'new_entrant': 0},
            'land_transfers': {'to_new': 0, 'to_existing': 0, 'to_enterprise': 0},
            'land_bank_balance': self.land_bank.total_area_mu,
            'total_land_check': current_total_land
        }
        
        target_n = stats['target_n']
        current_n = len(agents)
        
        logger.info(f"[Year {current_year}] 当前={current_n}, 目标={target_n}")
        
        # ==================== 1. 退出阶段 ====================
        entry_enabled = self._is_farmer_entry_enabled()
        stats['farmer_entry_enabled'] = entry_enabled
        planned_entries = self._plan_entry_count(agents, max(0, current_n - target_n))
        stats['planned_entries'] = planned_entries
        net_exit_gap = max(0, current_n - target_n)
        entry_exit_linkage_share = self._get_entry_exit_linkage_share()
        induced_replacement_exits = int(round(planned_entries * entry_exit_linkage_share))
        stats['entry_exit_linkage_share'] = entry_exit_linkage_share
        stats['induced_replacement_exits'] = induced_replacement_exits

        if current_n > target_n or planned_entries > 0:
            n_to_exit = net_exit_gap + induced_replacement_exits
            agents, exit_stats = self._execute_exits(agents, n_to_exit)
            stats['exits'] = exit_stats
        
        # ==================== 2. 进入阶段 ====================
        current_n = len(agents)
        n_to_enter = planned_entries
        new_agents = []
        if entry_enabled and current_n < target_n:
            n_to_enter = max(n_to_enter, target_n - current_n)
        if n_to_enter > 0:
            new_agents, entry_stats = self._execute_entries(n_to_enter)
            agents.extend(new_agents)
            stats['entries'] = entry_stats
            stats['land_transfers']['to_new'] = 0.0
        
        # ==================== 3. 土地再分配阶段 ====================
        if self.land_bank.total_area_mu > 0:
            weighted_allocation = self._allocate_weighted_released_land(
                agents,
                enterprises,
            )
            stats['weighted_land_allocation'] = weighted_allocation
            stats['land_transfers']['to_existing'] = weighted_allocation[
                'allocated_to_farmers_mu'
            ] - weighted_allocation['allocated_to_new_entrants_mu']
            stats['land_transfers']['to_enterprise'] = weighted_allocation[
                'allocated_to_enterprises_mu'
            ]
            stats['land_transfers']['to_new'] = weighted_allocation[
                'allocated_to_new_entrants_mu'
            ]

        if new_agents:
            funded_new_agents = [
                candidate
                for candidate in new_agents
                if float(candidate.get('land_area_mu', 0.0) or 0.0) > 0.0
            ]
            unfunded_ids = {id(candidate) for candidate in new_agents} - {
                id(candidate) for candidate in funded_new_agents
            }
            agents = [candidate for candidate in agents if id(candidate) not in unfunded_ids]
            stats['entries']['new_entrant'] = len(funded_new_agents)
            stats['entries']['land_allocated'] = float(
                sum(
                    float(candidate.get('land_area_mu', 0.0) or 0.0)
                    for candidate in funded_new_agents
                )
            )
        
        # ==================== 4. 土地守恒检查 ====================
        final_total_land = self._get_total_land_balance(agents, enterprises)
        
        stats['final_n'] = len(agents)
        stats['final_total_land'] = final_total_land
        stats['land_conservation_check'] = abs(final_total_land - self.initial_total_land) < 1.0
        
        if not stats['land_conservation_check']:
            logger.warning(f"[WARN] 土地守恒异常: 初始={self.initial_total_land:.1f}, "
                          f"当前={final_total_land:.1f}, "
                          f"差异={final_total_land - self.initial_total_land:.1f}亩")
        
        self.annual_stats.append(stats)
        
        return agents, stats
    
    def _execute_exits(self, 
                       agents: List[Any], 
                       n_to_exit: int) -> Tuple[List[Any], Dict]:
        """
        执行退出
        
        退出类型:
        1. 自然死亡: 高龄农户
        2. 城镇化迁移: 基于概率模型
        """
        stats = {'death': 0, 'urbanization': 0}

        paper_b_state = getattr(self.model, 'config', {}).get('paper_b_state_space', {}) or {}
        exit_pressure_scale = max(
            0.0,
            float(paper_b_state.get('exit_pressure', 1.0) or 1.0),
        )
        signal_rows = {
            str(getattr(agent, 'unique_id', '')): _exit_signal_diagnostic_row(
                operator_id=str(getattr(agent, 'unique_id', '')),
                base_signal=self.urbanization_model.calculate_probability(agent),
                pressure_scale=exit_pressure_scale,
                final_action=str(
                    getattr(agent, 'fixed_rule_current_action', 'Continue')
                    or 'Continue'
                ),
            )
            for agent in agents
        }
        
        pop_cfg = self._get_population_config()
        exit_cfg = pop_cfg.get('exit_mechanism', {}) or {}
        paper_exit_mode = pop_cfg.get('paper_model_exit_mode')
        if paper_exit_mode == 'livelihood_exit_only':
            death_ratio = 0.0
            urbanization_ratio = 1.0
        else:
            death_ratio = max(0.0, min(1.0, float(exit_cfg.get('death_ratio', 0.20) or 0.20)))
            urbanization_ratio = max(0.0, min(1.0, float(exit_cfg.get('urbanization_ratio', 1.0 - death_ratio) or (1.0 - death_ratio))))
        ratio_total = death_ratio + urbanization_ratio
        if ratio_total <= 0:
            death_ratio = 0.20
            urbanization_ratio = 0.80
            ratio_total = 1.0
        death_ratio /= ratio_total

        # 1. 自然死亡
        death_quota = min(n_to_exit, int(round(n_to_exit * death_ratio)))
        if (paper_exit_mode != 'livelihood_exit_only' and death_quota <= 0 and n_to_exit > 0
                and any(getattr(a, 'age', 0) > 65 for a in agents)):
            death_quota = 1
        death_candidates = sorted(
            [a for a in agents if getattr(a, 'age', 50) > 65],
            key=lambda x: getattr(x, 'age', 50),
            reverse=True
        )
        
        for agent in death_candidates[:death_quota]:
            signal_rows[str(getattr(agent, 'unique_id', ''))]['final_action'] = 'Exit'
            self._process_agent_exit(agent, 'death')
            if agent in agents:
                agents.remove(agent)
                stats['death'] += 1
        
        # 2. 城镇化退出（剩余配额）
        urban_quota = n_to_exit - stats['death']
        
        # 计算所有Agent的城镇化概率
        remaining_agents = [a for a in agents]
        scored = [
            (
                a,
                float(signal_rows[str(getattr(a, 'unique_id', ''))]['exit_signal_treated']),
            )
            for a in remaining_agents
        ]
        
        # 按概率加权随机选择（不是简单取top-k）
        for _ in range(urban_quota):
            if not scored:
                break
            
            # 轮盘赌选择
            total_prob = sum(p for _, p in scored)
            if total_prob <= 0:
                break
            
            r = rng_random(self) * total_prob
            cumsum = 0
            selected_idx = 0
            
            for i, (_, prob) in enumerate(scored):
                cumsum += prob
                if cumsum >= r:
                    selected_idx = i
                    break
            
            agent, _ = scored.pop(selected_idx)
            signal_rows[str(getattr(agent, 'unique_id', ''))]['final_action'] = 'Exit'
            self._process_agent_exit(agent, 'urbanization')
            if agent in agents:
                agents.remove(agent)
                stats['urbanization'] += 1
        
        logger.info(f"  退出: 死亡={stats['death']}, 城镇化={stats['urbanization']}")
        self.exit_signal_trace = getattr(self, 'exit_signal_trace', [])
        self.exit_signal_trace.extend(
            [{"year": self.current_year, **row} for row in signal_rows.values()]
        )
        return agents, stats
    
    def _process_agent_exit(self, agent: Any, exit_type: str) -> None:
        """处理单个Agent退出，回收土地"""
        self.land_bank._release_non_owned_operations_from_exiting_agent(agent)

        owned_patches = [
            patch for patch in getattr(agent, 'land_patches', [])
            if getattr(patch, 'owner', None) == agent
        ]

        if owned_patches:
            for patch in owned_patches:
                if (
                    hasattr(patch, 'initial_rental_rights_holder')
                    and patch.leased_to is not None and patch.leased_to is not agent
                    and patch.lease_contract_years > 0
                ):
                    continue
                origin_operator = getattr(patch, 'current_operator', None) or agent
                origin_crop = getattr(origin_operator, 'planned_crop', None)
                if origin_crop is not None:
                    setattr(
                        patch,
                        'production_succession_origin_crop_label',
                        str(getattr(origin_crop, 'value', origin_crop)),
                    )
                parcel = LandParcel(
                    area_mu=float(getattr(patch, 'area', 0.0) or 0.0),
                    quality=float(getattr(patch, 'soil_quality', 1.0) or 1.0),
                    previous_owner_id=getattr(agent, 'unique_id', None),
                    exit_year=self.current_year,
                    exit_type=exit_type,
                    patches=[patch]
                )
                self.land_bank._detach_patch_from_current_tenure(patch)
                self.land_bank.deposit(parcel)
        else:
            land_area = self._get_agent_owned_land_area(agent)
            if land_area > 0:
                parcel = LandParcel(
                    area_mu=land_area,
                    quality=getattr(agent, 'land_quality', 1.0),
                    previous_owner_id=getattr(agent, 'unique_id', None),
                    exit_year=self.current_year,
                    exit_type=exit_type
                )
                self.land_bank.deposit(parcel)

        if hasattr(agent, 'land_patches'):
            agent.land_patches = [
                patch for patch in getattr(agent, 'land_patches', [])
                if getattr(patch, 'owner', None) == agent
            ]
        if hasattr(agent, 'land_area_mu'):
            agent.land_area_mu = 0.0
        
        # 标记状态
        if hasattr(agent, 'status'):
            agent.status = f'exited_{exit_type}'
    
    def _execute_entries(self, n_to_enter: int) -> Tuple[List[Dict], Dict]:
        """
        执行新进入
        
        从土地库分配土地，从CFPS特征库Bootstrap采样特征
        """
        stats = {'new_entrant': 0, 'land_allocated': 0.0}
        new_agents = []
        
        for entry_index in range(n_to_enter):
            target_scale = self._sample_entry_scale()
            features = self._sample_entrant_features()
            features['unique_id'] = f"entrant_{self.current_year}_{entry_index}"
            features['land_area_mu'] = 0.0
            features['land_parcels'] = []
            features['target_operated_area'] = target_scale
            features['representation_weight_q'] = float(
                features.get('representation_weight_q', 1.0) or 1.0
            )
            expected_return = features.get('expected_land_return_yuan_per_mu')
            if expected_return is None:
                agricultural_income = float(features.get('agri_income', 0.0) or 0.0)
                expected_return = agricultural_income / target_scale if target_scale > 0.0 else 0.0
            features['expected_land_return_yuan_per_mu'] = float(expected_return or 0.0)
            features['entry_year'] = self.current_year
            features['is_new_entrant'] = True

            new_agents.append(features)
            stats['new_entrant'] += 1

        logger.info(f"  进入候选: {stats['new_entrant']}户, 等待加权土地配置")
        return new_agents, stats
    
    def _sample_entry_scale(self) -> float:
        """采样新进入者目标经营规模"""
        # 基于区域调整
        scale_params = {
            'national': (10, 50),
            'eastern': (15, 80),
            'central': (10, 40),
            'western': (5, 30),
            'northeast': (30, 150)
        }
        min_scale, max_scale = scale_params.get(self.region, (10, 50))
        entry_cfg = self._get_population_config().get('entry_mechanism', {}) or {}
        configured_min_scale = entry_cfg.get('min_scale_farming_size')
        configured_max_scale = entry_cfg.get('max_scale_farming_size')
        if configured_min_scale is not None:
            min_scale = max(1.0, float(configured_min_scale or min_scale))
        if configured_max_scale is not None:
            max_scale = max(min_scale, float(configured_max_scale or max_scale))
        
        # 对数正态分布
        mean_scale = (min_scale + max_scale) / 2
        return max(min_scale, min(max_scale, float(np_lognormal(self, np.log(mean_scale), 0.5))))
    
    def _sample_entrant_features(self) -> Dict:
        """
        Bootstrap采样新进入者特征
        
        优先从CFPS特征库采样，否则使用默认值
        """
        desired_livelihood = self._sample_entry_livelihood()
        age_min, age_max = self._get_entry_age_range()

        if self.feature_library:
            candidate_pools = {}
            for key, value in self.feature_library.items():
                if key == 'metadata':
                    continue
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    candidate_pools[key] = value

            if candidate_pools:
                pool_names = list(candidate_pools.keys())
                population_cfg = self._get_population_config()
                entry_cfg = population_cfg.get('entry_mechanism', {}) or {}
                configured_type_dist = entry_cfg.get('type_distribution', {}) or {}
                configured_pool_weights = [
                    max(0.0, float(configured_type_dist.get(name, 0.0) or 0.0))
                    for name in pool_names
                ]
                if sum(configured_pool_weights) > 0:
                    pool_weights = configured_pool_weights
                else:
                    pool_weights = [len(candidate_pools[name]) for name in pool_names]
                selected_pool_name = rng_choice(self, pool_names, weights=pool_weights)
                selected_pool = list(candidate_pools[selected_pool_name])

                if desired_livelihood:
                    matched_pool = [
                        row for row in selected_pool
                        if self._normalize_livelihood_label(
                            livelihood_raw=(
                                row.get('livelihood')
                                or row.get('livelihood_type')
                                or row.get('livelihood_original')
                            ),
                            pool_name=selected_pool_name
                        ) == desired_livelihood
                    ]
                    if matched_pool:
                        selected_pool = matched_pool

                sample = rng_choice(self, selected_pool).copy()

                if 'age' in sample and sample['age'] is not None:
                    raw_age = int(sample['age'])
                    if age_min <= raw_age <= age_max:
                        sample['age'] = int(np.clip(raw_age + rng_randint(self, -3, 3), age_min, age_max))
                    else:
                        # Entrants are a distinct returnee/succession pool; old
                        # CFPS donors should not all be clipped to the upper
                        # age bound, which erased the under-40 cohort.
                        sample['age'] = rng_randint(self, age_min, age_max)
                else:
                    sample['age'] = rng_randint(self, age_min, age_max)

                sample['education_level'] = int(np.clip(sample.get('education_level', 2) or 2, 0, 6))
                sample['household_size'] = max(1, int(sample.get('household_size', sample.get('family_size', 3)) or 3))

                init_capital = float(sample.get('initial_capital', 80000.0) or 80000.0)
                sample['initial_capital'] = max(5000.0, init_capital * rng_uniform(self, 0.9, 1.1))

                if 'land_area_mu' in sample and sample['land_area_mu'] is not None:
                    sample['land_area_mu'] = max(1.0, float(sample['land_area_mu']))

                livelihood_raw = (
                    sample.get('livelihood')
                    or sample.get('livelihood_type')
                    or sample.get('livelihood_original')
                )
                livelihood = self._normalize_livelihood_label(
                    livelihood_raw=livelihood_raw,
                    pool_name=selected_pool_name
                )
                if desired_livelihood is not None:
                    livelihood = desired_livelihood
                sample['livelihood'] = livelihood
                sample['agri_ratio'] = self._infer_agri_ratio(
                    livelihood=livelihood,
                    livelihood_raw=livelihood_raw
                )
                return sample
        
        # 默认新进入者特征
        return {
            'age': rng_randint(self, age_min, age_max),
            'education_level': rng_randint(self, 2, 4),
            'household_size': rng_randint(self, 2, 4),
            'initial_capital': rng_uniform(self, 50000, 200000),
            'livelihood': desired_livelihood or 'Pluriactive',
            'agri_ratio': self._infer_agri_ratio(
                livelihood=(desired_livelihood or 'Pluriactive'),
                livelihood_raw=(desired_livelihood or 'Pluriactive'),
            ),
            'tech_adoption': rng_uniform(self, 0.3, 0.7)
        }

    def _normalize_livelihood_label(self, livelihood_raw: Any, pool_name: str) -> str:
        """统一不同数据源中的生计标签。"""
        if isinstance(livelihood_raw, str):
            label = livelihood_raw.strip().lower()
            mapping = {
                'pure_farmer': 'Pure_Farmer',
                'pure farmer': 'Pure_Farmer',
                'pluriactive': 'Pluriactive',
                'non_farmer': 'Non_Farmer',
                'non farmer': 'Non_Farmer',
                '纯农型': 'Pure_Farmer',
                '兼业型': 'Pluriactive',
                '非农型': 'Non_Farmer'
            }
            if label in mapping:
                return mapping[label]

        if pool_name == 'family_farm':
            return 'Pure_Farmer'
        if pool_name == 'cooperative':
            return 'Pluriactive'
        if pool_name == 'return_migrant':
            return 'Non_Farmer'
        return 'Pluriactive'

    def _infer_agri_ratio(self, livelihood: str, livelihood_raw: Any) -> float:
        """根据生计标签推断农业收入占比（用于进入者策略初始化）。"""
        if isinstance(livelihood_raw, str):
            raw = livelihood_raw.strip()
            if raw == '纯农型':
                return float(np.clip(np_normal(self, 0.85, 0.08), 0.05, 0.98))
            if raw == '兼业型':
                return float(np.clip(np_normal(self, 0.45, 0.12), 0.05, 0.98))
            if raw == '非农型':
                return float(np.clip(np_normal(self, 0.12, 0.06), 0.01, 0.60))

        if livelihood == 'Pure_Farmer':
            return float(np.clip(np_normal(self, 0.80, 0.10), 0.30, 0.98))
        if livelihood == 'Non_Farmer':
            return float(np.clip(np_normal(self, 0.12, 0.06), 0.01, 0.50))
        return float(np.clip(np_normal(self, 0.45, 0.12), 0.05, 0.90))

    def get_summary_statistics(self) -> Dict:
        """
        Backward-compatible summary interface for workflow/report scripts.

        Reconstruct the richer summary expected by legacy callers from the
        canonical annual stats collected by PopulationManagerV2.
        """
        summary = self.get_summary()
        if not summary:
            current_population = int(self.base_n_agents or 0)
            return {
                'years_simulated': 0,
                'total_exits': 0,
                'total_entries': 0,
                'net_change': 0,
                'initial_population': current_population,
                'current_population': current_population,
                'core_panel_remaining': current_population,
                'land_bank_area': float(self.land_bank.total_area_mu),
                'initial_total_land': self.initial_total_land,
                'land_conservation': self.land_bank.get_balance_check(),
                'annual_stats': []
            }

        last_stats = self.annual_stats[-1] if self.annual_stats else {}
        initial_population = int(self.base_n_agents or last_stats.get('initial_n', 0))
        current_population = int(last_stats.get('final_n', initial_population))

        enriched = dict(summary)
        enriched.update({
            'initial_population': initial_population,
            'current_population': current_population,
            'core_panel_remaining': current_population,
            'land_bank_area': float(self.land_bank.total_area_mu),
        })
        return enriched
    
    def get_summary(self) -> Dict:
        """Return annual population-dynamics summary."""
        if not self.annual_stats:
            return {}

        total_exits = sum(
            s['exits']['death'] + s['exits']['urbanization']
            for s in self.annual_stats
        )
        total_entries = sum(
            s['entries']['new_entrant'] for s in self.annual_stats
        )

        return {
            'years_simulated': len(self.annual_stats),
            'total_exits': total_exits,
            'total_entries': total_entries,
            'net_change': total_entries - total_exits,
            'initial_total_land': self.initial_total_land,
            'land_bank_current': self.land_bank.total_area_mu,
            'land_conservation': self.land_bank.get_balance_check(),
            'annual_stats': self.annual_stats,
        }


def create_population_manager_v2(
    region: str = 'national',
    initial_land: float = None
) -> PopulationManagerV2:
    """Create a PopulationManagerV2 instance."""
    return PopulationManagerV2(
        region=region,
        initial_total_land=initial_land
    )


# ============== ??? ==============

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("PopulationManagerV2 测试")
    print("=" * 60)
    
    # 创建模拟Agent
    class MockAgent:
        def __init__(self, uid, age, edu, income, land):
            self.unique_id = uid
            self.age = age
            self.education_level = edu
            self.income_agricultural = income * 0.5
            self.income_total = income
            self.land_area_mu = land
            self.household_size = 3
            self.has_urban_family = rng_random(None) > 0.7
            self.status = 'active'
            self.received_land_transfers = []
    
    # 创建1000户初始农户
    agents = []
    for i in range(1000):
        agents.append(MockAgent(
            uid=i,
            age=rng_randint(None, 25, 75),
            edu=rng_randint(None, 1, 4),
            income=rng_uniform(None, 10000, 80000),
            land=float(np_lognormal(None, np.log(10), 0.8))  # 对数正态，均值约10亩
        ))
    
    initial_land = sum(a.land_area_mu for a in agents)
    print(f"初始: {len(agents)}户, 总土地={initial_land:.0f}亩")
    
    # 创建人口管理器
    pm = PopulationManagerV2(region='national', initial_total_land=initial_land)
    pm.initialize(agents)
    
    # 模拟12年
    for year in range(2010, 2022):
        agents, stats = pm.step(year, agents)
        
        # 将新进入者字典转换为MockAgent对象
        converted_agents = []
        for a in agents:
            if isinstance(a, dict):
                # 新进入者是字典，需要转换
                ma = MockAgent(
                    uid=a.get('unique_id', f'new_{year}_{len(converted_agents)}'),
                    age=a.get('age', 35),
                    edu=a.get('education_level', 3),
                    income=a.get('income_total', 50000),
                    land=a.get('land_area_mu', 10)
                )
                converted_agents.append(ma)
            else:
                converted_agents.append(a)
        agents = converted_agents
        
        current_land = sum(a.land_area_mu for a in agents) + pm.land_bank.total_area_mu
        check_mark = "OK" if stats['land_conservation_check'] else "FAIL"
        print(f"\n{year}年: 农户={stats['final_n']}, "
              f"退出={stats['exits']['death']}+{stats['exits']['urbanization']}, "
              f"进入={stats['entries']['new_entrant']}, "
              f"土地={current_land:.0f}亩 [{check_mark}]")
    
    # 汇总
    summary = pm.get_summary()
    print("\n=== 汇总 ===")
    print(f"模拟年数: {summary['years_simulated']}")
    print(f"总退出: {summary['total_exits']}")
    print(f"总进入: {summary['total_entries']}")
    print(f"土地守恒: {summary['land_conservation']}")
