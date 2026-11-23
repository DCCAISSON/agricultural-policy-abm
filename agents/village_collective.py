# -*- coding: utf-8 -*-
"""
@theory 村集体智能体 - 集体经济最大化决策框架
@critical 2025-12: 专家审阅意见 - 添加村集体Actor
@updated 2025-12-01: 增强决策智能化，添加集体经济最大化目标函数

理论依据：
1. 中国农村土地集体所有制：土地归村集体所有
2. 三权分置：所有权（集体）、承包权（农户）、经营权（可流转）
3. 集体经济组织理论（Weitzman, 1984; Lin, 1990）
4. 村庄治理与集体行动（Ostrom, 1990）

村集体的多重目标：
1. 集体经济收入最大化（服务费、土地出租收益）
2. 村民福利最大化（就业、收入）
3. 粮食安全保障（响应政府号召）
4. 土地资源优化配置（规模化、减少撂荒）

决策机制：
1. 基于多目标优化的土地流转组织策略
2. 理性预期框架下的价格预判
3. 博弈论视角的谈判定价
"""

from mesa import Agent
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from core.income_utils import get_farmer_net_income
from core.fixed_rule_runtime import action_income_enabled
from core.reproducibility import np_uniform, rng_random


class VillageCollectiveRole(Enum):
    """村集体在土地流转中的角色"""
    PASSIVE = "passive"      # 被动型：只登记备案
    ACTIVE = "active"        # 主动型：积极组织流转
    INTEGRATED = "integrated"  # 整合型：进行土地整合后统一出租


class CollectiveStrategy(Enum):
    """村集体的经营策略"""
    MAXIMIZE_REVENUE = "maximize_revenue"     # 收益最大化：追求最高租金
    BALANCED_WELFARE = "balanced_welfare"     # 均衡福利：兼顾农户利益
    GRAIN_SECURITY = "grain_security"         # 粮食安全：优先种粮主体
    SCALE_PROMOTION = "scale_promotion"       # 规模促进：优先大经营主体


@dataclass
class CollectiveEconomyState:
    """集体经济状态"""
    # 收入
    accumulated_service_fee: float = 0.0      # 累计服务费
    annual_service_fee: float = 0.0           # 年度服务费
    land_rental_income: float = 0.0           # 土地出租收入（集体所有的经营性资产）
    other_income: float = 0.0                 # 其他收入
    
    # 支出
    operating_cost: float = 0.0               # 运营成本
    public_service_expenditure: float = 0.0   # 公共服务支出
    
    # 流转统计
    total_facilitated_transfers: int = 0      # 累计促成流转数
    annual_facilitated_transfers: int = 0     # 年度促成流转数
    total_transferred_area: float = 0.0       # 累计流转面积
    annual_transferred_area: float = 0.0      # 年度流转面积
    
    # 效益指标
    transfer_success_rate: float = 0.0        # 流转成功率
    avg_rental_price: float = 0.0             # 平均租金


class VillageCollective(Agent):
    """
    村集体智能体
    
    @theory 基于集体经济最大化的智能决策框架
    
    核心功能：
    1. 土地流转中介与定价
    2. 土地整合与规模化经营组织
    3. 集体经济收入优化
    4. 村民福利协调
    5. 粮食安全政策响应
    
    决策逻辑：
    - 目标函数：max U = α·收益 + β·村民福利 + γ·粮食安全 + δ·规模效率
    - 约束条件：土地资源、劳动力供给、政策要求
    - 决策变量：租金定价、流转组织力度、整合策略
    """
    
    def __init__(self, unique_id, model, config=None, village_name="default_village"):
        # Mesa 3.x compatibility: Agent.__init__ may pass args to object incorrectly
        # We manually set required attributes first, then try super() but ignore failures
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # Required by Mesa grid system
        try:
            super().__init__(unique_id, model)
        except TypeError:
            # Mesa 3.x issue: don't call super().__init__() at all
            # Required attributes (unique_id, model, pos) already set above
            pass
        
        self.unique_id = unique_id
        self.village_name = village_name
        self.config = config or {}
        
        # 村集体配置参数
        village_config = self.config.get('agents', {}).get('village_collective', {})
        
        # 村集体角色类型
        role_type = village_config.get('role_type', 'active')
        self.role = VillageCollectiveRole(role_type)
        
        # 服务费率（默认3%）
        self.service_fee_rate = village_config.get('service_fee_rate', 0.03)
        grain_requirement_cfg = village_config.get('grain_requirement', {}) or {}
        self.grain_requirement_enabled = bool(grain_requirement_cfg.get('enabled', True))
        self.grain_requirement_ratio = float(grain_requirement_cfg.get('ratio', 0.40) or 0.40)
        self.basic_farmland_grain_requirement_ratio = float(
            grain_requirement_cfg.get('basic_farmland_ratio', 0.80) or 0.80
        )
        self.grain_requirement_enforcement_strength = float(
            grain_requirement_cfg.get('enforcement_strength', 0.70) or 0.70
        )
        
        # 流转中介成功率（主动组织时的成功率提升）
        self.mediation_success_boost = village_config.get('mediation_success_boost', 0.2)
        
        # 土地整合阈值（超过此面积才进行整合）
        self.integration_threshold = village_config.get('integration_threshold', 50)  # 亩
        
        # 最小整合面积（整合后每块地不小于此面积）
        self.min_integrated_area = village_config.get('min_integrated_area', 10)  # 亩
        
        # 村集体参与流转的比例（约30%的流转通过村委会）
        self.participation_rate = village_config.get('participation_rate', 0.30)
        self.policy_reference_rate = village_config.get('policy_reference_rate', 0.20)
        self.dynamic_participation_rate = float(self.participation_rate)
        self.dynamic_mediation_success_boost = float(self.mediation_success_boost)
        self.governance_capacity_signal = 0.5
        self.transfer_congestion_signal = 1.0
        
        # ========== @new 2025-12-22: 兜底崩溃机制 ==========
        # Nature级模型要求：村集体托管超限会导致系统性风险
        collapse_cfg = village_config.get('collapse_mechanism', {})
        self.enable_collapse = collapse_cfg.get('enable', True)
        self.max_custody_ratio = collapse_cfg.get('max_custody_ratio', 0.4)  # 最大托管比例（40%）
        self.collapse_yield_penalty = collapse_cfg.get('yield_penalty', 0.3)  # 崩溃后产量惩罚
        self.infrastructure_decay_rate = collapse_cfg.get('decay_rate', 0.1)  # 基础设施衰退率
        
        # 托管状态跟踪
        self.custody_land_area = 0  # 托管土地面积（亩）
        self.total_village_area = collapse_cfg.get('total_area', 3000)  # 村庄总耕地面积
        self.is_collapsed = False  # 是否已崩溃
        self.collapse_year = None  # 崩溃年份
        self.infrastructure_condition = 1.0  # 基础设施状况 [0, 1]
        
        # ========== 新增：集体经济最大化目标函数参数 ==========
        objective_weights = village_config.get('objective_weights', {})
        self.alpha = objective_weights.get('revenue', 0.35)        # 收益权重
        self.beta = objective_weights.get('welfare', 0.30)         # 村民福利权重
        self.gamma = objective_weights.get('grain_security', 0.20) # 粮食安全权重
        self.delta = objective_weights.get('scale_efficiency', 0.15)  # 规模效率权重
        
        # 当前策略（初始化为均衡福利）
        self.current_strategy = CollectiveStrategy.BALANCED_WELFARE
        
        # 集体经济状态
        self.economy_state = CollectiveEconomyState()
        
        # 内部记录
        self.registered_farmers = []  # 登记在册的农户
        self.land_transfer_records = []  # 流转记录
        self.integrated_land_pool = []  # 整合后的土地池
        self.service_fee_collected = 0  # 累计服务费
        self.step_transfers_facilitated = 0  # 本期促成的流转数
        
        # ========== 新增：市场信息与预期 ==========
        transfer_config = self.config.get('agents', {}).get('farmer', {}).get('land_transfer', {})
        initial_rental_price = float(transfer_config.get('base_rent_price', 300.0) or 300.0)
        self.market_info = {
            'avg_rental_price': initial_rental_price,      # 平均租金
            'grain_price': 2.58,            # 粮食价格
            'cash_crop_price': 5.0,         # 经济作物价格
            'policy_subsidy_rate': 0.15     # 政策补贴率
        }
        
        self.rental_price_history = []      # 租金历史
        self.expected_rental_price = initial_rental_price   # 预期租金
        
        # ========== 新增 2025-12-22：空间集成参数 ==========
        # @theory 空间异质性 - 村集体的服务范围影响土地流转组织效率
        # @fix 解决"position/service_radius未使用"问题
        spatial_config = village_config.get('spatial', {})
        self.service_radius = spatial_config.get('service_radius', 5)  # 服务半径（网格单位）
        self.position = None  # 将在被放置到网格时由Mesa自动设置为self.pos
        network_config = village_config.get('network', {})
        self.network_neighbors = []
        self.network_weights = {}
        self.network_influence_weight = network_config.get('influence_weight', 0.15)
        self.assignment_mode = network_config.get('assignment_mode', 'nearest_only')
        self.village_index = None
        self.network_reference_price = 0.0

        subsidy_review_config = village_config.get('subsidy_review', {})
        self.subsidy_review_enabled = subsidy_review_config.get('enabled', True)
        self.subsidy_review_actor_types = set(
            subsidy_review_config.get('actor_types', ['farmer', 'enterprise'])
        )
        self.subsidy_direct_review_actor_types = set(
            subsidy_review_config.get('direct_review_actor_types', ['enterprise'])
        )
        self.subsidy_claim_area_tolerance = float(subsidy_review_config.get('area_tolerance', 1e-6))
        self.subsidy_claim_grain_tolerance = float(subsidy_review_config.get('grain_tolerance', 1e-6))
        self.subsidy_review_records = []
        self.annual_claims_received = 0
        self.annual_reviewed_claims = 0
        self.annual_approved_claims = 0

        cross_village_config = village_config.get('cross_village_governance', {})
        self.cross_village_governance_enabled = cross_village_config.get('enabled', True)
        self.cross_village_max_contracts_per_link = max(
            0,
            int(cross_village_config.get('max_contracts_per_link', 2)),
        )
        self.cross_village_acceptance_threshold = float(
            cross_village_config.get('acceptance_threshold', 0.55)
        )
        self.cross_village_coordination_bonus = float(
            cross_village_config.get('coordination_bonus', 0.10)
        )
        self.cross_village_contract_years = max(
            1,
            int(cross_village_config.get('contract_years', 3)),
        )
        self.cross_village_rent_markup = float(
            cross_village_config.get('rent_markup', 0.03)
        )
        source_share = float(cross_village_config.get('source_fee_share', 0.6))
        receiver_share = float(cross_village_config.get('receiver_fee_share', 0.4))
        share_total = source_share + receiver_share
        if share_total <= 0:
            source_share, receiver_share, share_total = 0.6, 0.4, 1.0
        self.cross_village_source_fee_share = source_share / share_total
        self.cross_village_receiver_fee_share = receiver_share / share_total
        self.pending_cross_village_supply = []
        self.pending_cross_village_demand = []
        self.cross_village_transfer_records = []
        self.cross_village_agreements = []
        self.annual_cross_village_transfer_area = 0.0
        self.annual_cross_village_contracts = 0
        self.annual_cross_village_partners = set()

        # @new 2026-05-02: 声誉学习系统（Reputation Belief System）
        # 通过 mechanism_switches.use_reputation_village = True 启用，
        # 让村集体从纯阈值规则升级为"在长期撮合中累积信念的学习节点"。
        # 详细文档：agents/village_reputation.py
        self.reputation_system = None
        try:
            mechanism_switches = (
                self.model.config.get('mechanism_switches', {})
                if hasattr(self.model, 'config') else {}
            ) or {}
        except Exception:
            mechanism_switches = {}
        if mechanism_switches.get('use_reputation_village', False):
            try:
                from agents.village_reputation import ReputationBeliefSystem
                rep_cfg = (config or {}).get('agents', {}).get('village_collective', {}).get('reputation', {})
                self.reputation_system = ReputationBeliefSystem(
                    decay_rate=float(rep_cfg.get('decay_rate', 0.95)),
                    prior_alpha=float(rep_cfg.get('prior_alpha', 1.0)),
                    prior_beta=float(rep_cfg.get('prior_beta', 1.0)),
                )
            except Exception as exc:
                if not hasattr(self.model, '_reputation_init_errors'):
                    self.model._reputation_init_errors = []
                self.model._reputation_init_errors.append({
                    'village_id': getattr(self, 'unique_id', None),
                    'error': str(exc),
                })
                self.reputation_system = None
        
        # 空间相关的效率参数
        # 距离衰减系数：距离越远，服务效率越低
        self.distance_decay = spatial_config.get('distance_decay', 0.1)  # η: e^(-η*d)
        # 集中度加成：高密度区域组织效率更高
        self.concentration_bonus = spatial_config.get('concentration_bonus', 0.05)
        
        # ========== 新增：理性预期机制 ==========
        self._initialize_rational_expectation()

    def configure_network(self, neighbors=None, weights=None, village_index=None):
        """Configure the village's position inside a multi-village network."""
        self.network_neighbors = list(neighbors or [])
        self.network_weights = dict(weights or {})
        if village_index is not None:
            self.village_index = village_index

    def begin_annual_cycle(self):
        """Reset annual ledgers before village-level coordination begins."""
        self.step_transfers_facilitated = 0
        self.pending_cross_village_supply = []
        self.pending_cross_village_demand = []
        self.annual_claims_received = 0
        self.annual_reviewed_claims = 0
        self.annual_approved_claims = 0
        self.annual_cross_village_transfer_area = 0.0
        self.annual_cross_village_contracts = 0
        self.annual_cross_village_partners = set()
        self.dynamic_participation_rate = float(self.participation_rate)
        self.dynamic_mediation_success_boost = float(self.mediation_success_boost)

    def export_sar(self) -> dict:
        """Export (state, action) for SAR logging.

        Village collective is currently rule-based (policy_mode='rule').
        Reward is captured by SAR logger from coordination outcomes.
        """
        return {
            'state': {
                'integrated_pool_size': len(getattr(self, 'integrated_land_pool', []) or []),
                'strategy': str(getattr(self, 'current_strategy', None)),
                'coordination_success_rate': float(getattr(self, '_coordination_success_rate', 0.0) or 0.0),
                'cross_village_edges': int(getattr(self, 'network_neighbors', []) and len(self.network_neighbors) or 0),
                'mediation_volume_mu': float(getattr(self, '_mediation_volume_mu', 0.0) or 0.0),
            },
            'action': {
                'mediation_count': int(getattr(self, '_mediation_count', 0) or 0),
                'cross_village_transfers': int(getattr(self, '_cross_village_transfer_count', 0) or 0),
            },
            'policy_mode': 'rule',
            'agent_type': 'village_collective',
        }

    def finalize_annual_cycle(self):
        """Run village-side ex-post verification after agents file claims."""
        self.pre_review_subsidy_claims()

    def _get_governance_coordination_posture(self) -> float:
        posture_map = {
            CollectiveStrategy.MAXIMIZE_REVENUE: 0.62,
            CollectiveStrategy.BALANCED_WELFARE: 0.78,
            CollectiveStrategy.GRAIN_SECURITY: 0.72,
            CollectiveStrategy.SCALE_PROMOTION: 0.82,
        }
        return float(posture_map.get(self.current_strategy, 0.70))

    def _get_cross_village_network_weight(self, other_village) -> float:
        if other_village is None:
            return 0.0
        return float(
            self.network_weights.get(
                other_village.unique_id,
                getattr(other_village, 'network_weights', {}).get(self.unique_id, 0.0),
            ) or 0.0
        )

    def prepare_cross_village_market_state(self):
        """Expose post-local-matching surplus and demand for inter-village bargaining."""
        if not self.cross_village_governance_enabled:
            self.pending_cross_village_supply = []
            self.pending_cross_village_demand = []
            return

        self.pending_cross_village_supply = sorted(
            self._identify_potential_suppliers(),
            key=lambda item: -float(item.get('willingness', 0.0) or 0.0),
        )
        self.pending_cross_village_demand = sorted(
            self._identify_potential_demanders(),
            key=lambda item: -float(item.get('willingness', 0.0) or 0.0),
        )

    def attempt_cross_village_coordination_with(self, other_village):
        """Negotiate bilateral cross-village subcontracting agreements."""
        if (
            not self.cross_village_governance_enabled
            or other_village is None
            or not getattr(other_village, 'cross_village_governance_enabled', False)
        ):
            return []

        agreements = []
        agreements.extend(self._attempt_directional_cross_village_coordination(self, other_village))
        agreements.extend(self._attempt_directional_cross_village_coordination(other_village, self))
        return agreements

    def _attempt_directional_cross_village_coordination(self, source_village, receiver_village):
        """Try to route land from a source village with surplus supply to a receiving village."""
        if source_village is None or receiver_village is None:
            return []
        if not getattr(source_village, 'pending_cross_village_supply', None):
            return []
        if not getattr(receiver_village, 'pending_cross_village_demand', None):
            return []

        agreements = []
        max_contracts = min(
            source_village.cross_village_max_contracts_per_link,
            len(source_village.pending_cross_village_supply),
            len(receiver_village.pending_cross_village_demand),
        )

        while source_village.pending_cross_village_supply and receiver_village.pending_cross_village_demand:
            if len(agreements) >= max_contracts:
                break

            supplier = source_village.pending_cross_village_supply[0]
            demander = receiver_village.pending_cross_village_demand[0]
            agreement = source_village._negotiate_cross_village_coordination(
                receiver_village,
                supplier,
                demander,
            )
            if agreement is None:
                receiver_village.pending_cross_village_demand.pop(0)
                continue

            contract_record = source_village._execute_cross_village_subcontract(
                receiver_village,
                supplier,
                demander,
                agreement,
            )
            if contract_record is None:
                source_village.pending_cross_village_supply.pop(0)
                continue

            agreements.append(contract_record)
            source_village.pending_cross_village_supply.pop(0)
            receiver_village.pending_cross_village_demand.pop(0)

        return agreements

    def _negotiate_cross_village_coordination(self, receiver_village, supplier, demander):
        """Compute whether a village-village agreement is acceptable to both sides."""
        if receiver_village is None or supplier is None or demander is None:
            return None

        supplier_willingness = float(supplier.get('willingness', 0.0) or 0.0)
        demander_willingness = float(demander.get('willingness', 0.0) or 0.0)
        network_weight = self._get_cross_village_network_weight(receiver_village)
        source_posture = self._get_governance_coordination_posture()
        receiver_posture = receiver_village._get_governance_coordination_posture()
        acceptance_threshold = max(
            self.cross_village_acceptance_threshold,
            receiver_village.cross_village_acceptance_threshold,
        )

        bargaining_score = (
            0.30 * source_posture
            + 0.30 * receiver_posture
            + 0.20 * supplier_willingness
            + 0.20 * demander_willingness
            + min(0.10, network_weight * 0.10)
        )
        if bargaining_score < acceptance_threshold:
            return None

        success_probability = min(
            0.98,
            bargaining_score + 0.5 * self.cross_village_coordination_bonus,
        )
        if rng_random(self) > success_probability:
            return None

        source_fee_share = 0.5 * (
            self.cross_village_source_fee_share + receiver_village.cross_village_source_fee_share
        )
        receiver_fee_share = 0.5 * (
            self.cross_village_receiver_fee_share + receiver_village.cross_village_receiver_fee_share
        )
        share_total = source_fee_share + receiver_fee_share
        if share_total <= 0:
            source_fee_share, receiver_fee_share, share_total = 0.6, 0.4, 1.0

        return {
            'bargaining_score': bargaining_score,
            'success_probability': success_probability,
            'source_fee_share': source_fee_share / share_total,
            'receiver_fee_share': receiver_fee_share / share_total,
            'rent_markup': 0.5 * (
                self.cross_village_rent_markup + receiver_village.cross_village_rent_markup
            ),
            'contract_years': max(
                self.cross_village_contract_years,
                receiver_village.cross_village_contract_years,
            ),
            'network_weight': network_weight,
        }

    def _execute_cross_village_subcontract(self, receiver_village, supplier, demander, agreement):
        """Finalize a cross-village subcontract after bilateral agreement is reached."""
        supplier_farmer = supplier.get('farmer')
        demander_agent = demander.get('agent')
        if supplier_farmer is None or demander_agent is None:
            return None

        available_patches = [
            patch
            for patch in supplier.get('patches', [])
            if getattr(patch, 'owner', None) == supplier_farmer
            and getattr(patch, 'leased_to', None) is None
            and not getattr(patch, 'abandoned', False)
        ]
        if not available_patches:
            return None

        total_area = sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in available_patches)
        if total_area <= 0:
            return None

        local_reference = float(self.expected_rental_price or self.market_info.get('avg_rental_price', 300.0) or 300.0)
        receiver_reference = float(
            getattr(receiver_village, 'expected_rental_price', 0.0)
            or getattr(receiver_village, 'market_info', {}).get('avg_rental_price', 0.0)
            or local_reference
        )
        adjusted_rent = 0.5 * (local_reference + receiver_reference)
        adjusted_rent *= 1 + float(agreement.get('rent_markup', 0.0) or 0.0)

        contract_years = int(agreement.get('contract_years', self.cross_village_contract_years))
        leased_area = 0.0
        parcel_ids = [] if self.model.config.get('runtime_fixed_rule', {}).get('national_v1') else None
        for patch in available_patches:
            patch_rent = adjusted_rent * patch.area if action_income_enabled(self.model.config) else adjusted_rent
            lease_applied = True
            if hasattr(patch, 'set_leased'):
                lease_applied = patch.set_leased(
                    to_agent=demander_agent,
                    contract_years=contract_years,
                    rental_price=patch_rent,
                    transfer_channel='cross_village_collective',
                )
            else:
                clearing_house = getattr(self.model, 'land_clearing_house', None)
                if (
                    clearing_house is not None and
                    hasattr(clearing_house, '_check_transfer_rate_limit') and
                    not clearing_house._check_transfer_rate_limit(
                        self.model,
                        float(getattr(patch, 'area', 0.0) or 0.0),
                        transfer_channel='cross_village_collective',
                        demander=demander_agent,
                    )
                ):
                    lease_applied = False
                else:
                    patch.leased_to = demander_agent
                    patch.current_operator = demander_agent
                    patch.rental_price = patch_rent
                    if hasattr(patch, 'transfer_channel'):
                        patch.transfer_channel = 'cross_village_collective'
            if lease_applied is False:
                continue
            leased_area += float(getattr(patch, 'area', 0.0) or 0.0)
            if parcel_ids is not None:
                parcel_ids.append(str(patch.get_patch_id()))
        if leased_area <= 0:
            return None
        if hasattr(self.model, 'invalidate_land_tenure_caches'):
            self.model.invalidate_land_tenure_caches()

        transaction_value = leased_area * adjusted_rent
        total_service_fee = transaction_value * self.service_fee_rate
        source_fee = total_service_fee * float(agreement.get('source_fee_share', self.cross_village_source_fee_share) or 0.0)
        receiver_fee = total_service_fee * float(agreement.get('receiver_fee_share', self.cross_village_receiver_fee_share) or 0.0)

        self.service_fee_collected += source_fee
        receiver_village.service_fee_collected += receiver_fee
        self.economy_state.annual_service_fee += source_fee
        receiver_village.economy_state.annual_service_fee += receiver_fee
        self.economy_state.annual_transferred_area += leased_area
        self.economy_state.annual_facilitated_transfers += 1
        self.step_transfers_facilitated += 1
        self.annual_cross_village_transfer_area += leased_area
        self.annual_cross_village_contracts += 1
        self.annual_cross_village_partners.add(receiver_village.unique_id)
        receiver_village.annual_cross_village_partners.add(self.unique_id)

        contract_id = (
            f"cross_village_{self.model.step_count}_{self.unique_id}_"
            f"{receiver_village.unique_id}_{len(self.cross_village_transfer_records) + 1}"
        )
        source_patch_count = len(available_patches)
        source_avg_patch_size = leased_area / max(1, source_patch_count)
        source_fragmented_area = sum(
            float(getattr(patch, 'area', 0.0) or 0.0)
            for patch in available_patches
            if float(getattr(patch, 'area', 0.0) or 0.0) <= self.min_integrated_area
        )
        basic_farmland_area = sum(
            float(getattr(patch, 'area', 0.0) or 0.0)
            for patch in available_patches
            if hasattr(patch, 'is_basic_farmland') and bool(patch.is_basic_farmland())
        )
        grain_obligation_ratio = (
            self.basic_farmland_grain_requirement_ratio
            if basic_farmland_area > 0 else self.grain_requirement_ratio
        )
        grain_obligation_area = leased_area * grain_obligation_ratio if self.grain_requirement_enabled else 0.0
        receiver_enterprise_type = ''
        receiver_operator_type = demander.get('type')
        if hasattr(demander_agent, 'enterprise_type'):
            raw_type = getattr(demander_agent, 'enterprise_type', None)
            receiver_enterprise_type = str(
                getattr(raw_type, 'value', getattr(raw_type, 'name', raw_type)) or ''
            )
            if 'cooperative' in receiver_enterprise_type.lower():
                receiver_operator_type = 'cooperative'
            else:
                receiver_operator_type = 'agri_corporation'
        chain_label = f"fragmented_land_to_cross_village_collective_to_{receiver_operator_type}"
        record = {
            'contract_id': contract_id,
            'step': getattr(self.model, 'step_count', 0),
            'supplier_id': supplier_farmer.unique_id,
            'demander_id': demander_agent.unique_id,
            'demander_type': demander.get('type'),
            'receiver_operator_type': receiver_operator_type,
            'receiver_enterprise_type': receiver_enterprise_type,
            'source_village_id': self.unique_id,
            'receiver_village_id': receiver_village.unique_id,
            'area': leased_area,
            'source_patch_count': source_patch_count,
            'source_avg_patch_size': source_avg_patch_size,
            'source_fragmented_area': source_fragmented_area,
            'basic_farmland_area': basic_farmland_area,
            'grain_obligation_enabled': self.grain_requirement_enabled,
            'grain_obligation_ratio': grain_obligation_ratio,
            'grain_obligation_area': grain_obligation_area,
            'rental_value': transaction_value,
            'rental_price_per_mu': adjusted_rent,
            'service_fee_total': total_service_fee,
            'service_fee_source': source_fee,
            'service_fee_receiver': receiver_fee,
            'estimated_owner_rent_income': max(0.0, transaction_value - total_service_fee),
            'operator_rent_payment': transaction_value,
            'rent_distribution_farmer_share': max(0.0, transaction_value - total_service_fee) / transaction_value if transaction_value > 0 else 0.0,
            'rent_distribution_collective_share': total_service_fee / transaction_value if transaction_value > 0 else 0.0,
            'transfer_channel': 'cross_village_collective',
            'governance_mode': 'cross_village_subcontract',
            'compliance_status': 'pending',
            'breach_detected': False,
            'monitoring_probability': self.grain_requirement_enforcement_strength,
            'bargaining_score': agreement.get('bargaining_score'),
            'network_weight': agreement.get('network_weight'),
            'chain_label': chain_label,
        }
        if parcel_ids is not None:
            record['contract_patch_ids'] = parcel_ids
        self.cross_village_transfer_records.append(record)
        self.cross_village_agreements.append(dict(record))
        receiver_village.cross_village_agreements.append(dict(record, record_role='receiver'))
        self.land_transfer_records.append(dict(record))
        return record

    def _collect_relevant_subsidy_claims(self):
        """Filter raw claims down to those that fall inside this village's review scope."""
        government = getattr(self.model, 'government', None)
        raw_claims = getattr(government, 'subsidy_claims_submitted', []) or []
        relevant_claims = []
        for claim in raw_claims:
            actor_type = claim.get('actor_type', 'farmer')
            if self.subsidy_review_actor_types and actor_type not in self.subsidy_review_actor_types:
                continue

            area_breakdown = dict(claim.get('village_area_breakdown', {}) or {})
            grain_breakdown = dict(claim.get('village_grain_area_breakdown', {}) or {})
            local_area = float(area_breakdown.get(self.unique_id, 0.0) or 0.0)
            local_grain = float(grain_breakdown.get(self.unique_id, 0.0) or 0.0)
            primary_village_id = claim.get('primary_village_id')

            if local_area > self.subsidy_claim_area_tolerance or local_grain > self.subsidy_claim_grain_tolerance:
                relevant_claims.append(claim)
                continue
            if primary_village_id == self.unique_id:
                relevant_claims.append(claim)
                continue

            agent = claim.get('agent')
            if getattr(agent, 'primary_village_collective_id', None) == self.unique_id:
                relevant_claims.append(claim)
        return relevant_claims

    def pre_review_subsidy_claims(self):
        """Village-side first review: certify local eligible area before government re-review."""
        if not self.subsidy_review_enabled:
            return []

        government = getattr(self.model, 'government', None)
        if government is None:
            return []

        relevant_claims = self._collect_relevant_subsidy_claims()
        self.annual_claims_received = len(relevant_claims)
        review_records = []

        for claim in relevant_claims:
            existing_reviews = claim.setdefault('village_reviews', [])
            if any(review.get('village_id') == self.unique_id for review in existing_reviews):
                continue
            review = self._review_subsidy_claim(claim)
            existing_reviews.append(review)
            claim['village_pre_reviewed_area'] = sum(
                float(item.get('reviewed_eligible_area', 0.0) or 0.0)
                for item in existing_reviews
                if item.get('status') == 'approved'
            )
            claim['village_pre_reviewed_grain_area'] = sum(
                float(item.get('reviewed_grain_area', 0.0) or 0.0)
                for item in existing_reviews
                if item.get('status') == 'approved'
            )
            review_records.append(review)

        self.subsidy_review_records.extend(review_records)
        self.annual_reviewed_claims = len(review_records)
        self.annual_approved_claims = sum(1 for record in review_records if record.get('status') == 'approved')
        if government.current_subsidy_runtime:
            government.current_subsidy_runtime.setdefault('claims_pre_reviewed', []).extend(
                dict(record) for record in review_records
            )
        return review_records

    def _review_subsidy_claim(self, claim):
        """Review the part of a claim that belongs to this village's jurisdiction."""
        agent = claim.get('agent')
        active_policy = getattr(getattr(self.model, 'environment', None), 'active_policy', {}) or {}
        subsidy_target = active_policy.get('subsidy_target', 'operator')
        if subsidy_target == 'owner':
            expected_area_map = self.model.get_owned_area_by_village(agent)
        else:
            expected_area_map = self.model.get_operated_area_by_village(agent)
        operated_area_map = self.model.get_operated_area_by_village(agent)

        area_breakdown = dict(claim.get('village_area_breakdown', {}) or {})
        grain_breakdown = dict(claim.get('village_grain_area_breakdown', {}) or {})
        local_claimed_area = float(area_breakdown.get(self.unique_id, 0.0) or 0.0)
        local_claimed_grain = float(grain_breakdown.get(self.unique_id, 0.0) or 0.0)
        expected_local_area = float(expected_area_map.get(self.unique_id, 0.0) or 0.0)
        expected_local_grain = float(operated_area_map.get(self.unique_id, 0.0) or 0.0)

        if local_claimed_area <= self.subsidy_claim_area_tolerance and claim.get('primary_village_id') == self.unique_id:
            fallback_area = float(claim.get('eligible_area', 0.0) or 0.0)
            local_claimed_area = min(fallback_area, expected_local_area if expected_local_area > 0 else fallback_area)
        if local_claimed_grain <= self.subsidy_claim_grain_tolerance and claim.get('primary_village_id') == self.unique_id:
            fallback_grain = float(claim.get('grain_area', 0.0) or 0.0)
            local_claimed_grain = min(fallback_grain, expected_local_grain if expected_local_grain > 0 else fallback_grain)

        status = 'approved'
        reason = 'area_certified'
        reviewed_area = max(0.0, min(local_claimed_area, expected_local_area))
        reviewed_grain_area = max(0.0, min(local_claimed_grain, expected_local_grain))

        if reviewed_area <= self.subsidy_claim_area_tolerance and reviewed_grain_area <= self.subsidy_claim_grain_tolerance:
            status = 'rejected'
            reason = 'no_local_jurisdiction'
        elif local_claimed_area > expected_local_area + self.subsidy_claim_area_tolerance:
            status = 'rejected'
            reason = 'eligible_area_exceeds_village_registry'
            reviewed_area = 0.0
            reviewed_grain_area = 0.0
        elif local_claimed_grain > expected_local_grain + self.subsidy_claim_grain_tolerance:
            status = 'rejected'
            reason = 'grain_area_exceeds_operated_registry'
            reviewed_area = 0.0
            reviewed_grain_area = 0.0

        return {
            'claim_id': claim.get('claim_id'),
            'step': getattr(self.model, 'step_count', 0),
            'village_id': self.unique_id,
            'village_name': self.village_name,
            'reviewer_id': self.unique_id,
            'status': status,
            'reason': reason,
            'reviewed_eligible_area': reviewed_area,
            'reviewed_grain_area': reviewed_grain_area,
            'actor_type': claim.get('actor_type', 'farmer'),
        }
    
    # ========== 空间集成方法 2025-12-22 ==========
    
    def get_neighbors_in_radius(self, radius: int = None) -> List:
        """
        @theory 获取服务半径内的农户
        
        使用Mesa grid的空间查询功能，基于曼哈顿距离或Moore邻域。
        
        理论依据：
        - 村集体服务具有空间衰减特性（Tobler第一定律）
        - 近距离农户更容易参与村集体组织的流转
        
        Args:
            radius: 查询半径，默认使用self.service_radius
            
        Returns:
            List[FarmerAgent]: 服务半径内的农户列表
        """
        if radius is None:
            radius = self.service_radius
        
        neighbors = []
        
        # 方法1: 使用Mesa grid的pos属性
        if hasattr(self, 'pos') and self.pos is not None and hasattr(self.model, 'grid'):
            # 获取邻域内的所有智能体
            # include_center=False: 不包括自己
            try:
                neighbor_cells = self.model.grid.get_neighborhood(
                    self.pos, 
                    moore=True,  # Moore邻域（8方向）
                    include_center=False,
                    radius=radius
                )
                for cell in neighbor_cells:
                    cell_contents = self.model.grid.get_cell_list_contents([cell])
                    for agent in cell_contents:
                        # 只收集FarmerAgent
                        if hasattr(agent, 'livelihood_strategy'):  # FarmerAgent的特征属性
                            neighbors.append(agent)
            except Exception:
                pass
        
        # 方法2: 回退到registered_farmers
        if not neighbors and self.registered_farmers:
            neighbors = list(self.registered_farmers)
        
        return neighbors
    
    def get_spatial_efficiency(self, farmer) -> float:
        """
        @theory 计算村集体服务特定农户的空间效率
        
        基于距离衰减模型：efficiency = e^(-η * distance)
        
        理论依据：
        - 地理学第一定律（Tobler, 1970）：近事物比远事物更相关
        - 交易成本理论：距离越远，组织协调成本越高
        
        Args:
            farmer: 目标农户
            
        Returns:
            float: 空间效率系数 [0, 1]
        """
        # 计算距离
        distance = self._calculate_distance_to_farmer(farmer)
        
        # 距离衰减公式
        efficiency = np.exp(-self.distance_decay * distance)
        
        # 加入集中度加成（如果该区域农户密度高）
        density_bonus = self._calculate_local_density_bonus(farmer)
        
        return min(1.0, efficiency + density_bonus)
    
    def _calculate_distance_to_farmer(self, farmer) -> float:
        """
        计算到农户的空间距离
        
        优先使用网格位置，回退到随机距离
        """
        # 优先使用网格位置
        if hasattr(self, 'pos') and self.pos is not None:
            farmer_pos = getattr(farmer, 'pos', None)
            if farmer_pos is not None:
                # 曼哈顿距离
                return abs(self.pos[0] - farmer_pos[0]) + abs(self.pos[1] - farmer_pos[1])
        
        # 尝试使用土地位置
        if hasattr(farmer, 'land_patches') and farmer.land_patches:
            patch = farmer.land_patches[0]
            if hasattr(patch, 'coordinates') and patch.coordinates:
                # 假设村集体位于网格中心
                center = (self.model.grid.width // 2, self.model.grid.height // 2) if hasattr(self.model, 'grid') else (5, 5)
                return abs(patch.coordinates[0] - center[0]) + abs(patch.coordinates[1] - center[1])
        
        # 回退：随机距离
        return float(np_uniform(self, 1, self.service_radius * 2))
    
    def _calculate_local_density_bonus(self, farmer) -> float:
        """
        计算局部农户密度带来的效率加成
        
        高密度区域组织成本更低（规模经济）
        """
        if not hasattr(self.model, 'grid') or not hasattr(farmer, 'pos') or farmer.pos is None:
            return 0.0
        
        try:
            # 统计农户周围的农户数量
            local_neighbors = self.model.grid.get_neighbors(
                farmer.pos,
                moore=True,
                include_center=False,
                radius=2
            )
            farmer_count = sum(1 for n in local_neighbors if hasattr(n, 'livelihood_strategy'))
            
            # 密度加成：每多一个农户，增加concentration_bonus
            return min(0.2, farmer_count * self.concentration_bonus)
        except Exception:
            return 0.0
    
    def update_registered_farmers_by_proximity(self):
        """
        @theory 基于空间邻近性更新登记农户列表
        
        只将服务半径内的农户纳入登记，体现空间服务约束。
        """
        self.registered_farmers = []
        
        # 获取服务范围内的农户
        nearby_farmers = self.get_neighbors_in_radius(self.service_radius)
        
        for farmer in nearby_farmers:
            # 计算服务效率
            efficiency = self.get_spatial_efficiency(farmer)
            
            # 效率高于阈值才登记
            if efficiency > 0.3 and self._is_primary_collective_for_farmer(farmer):
                self.registered_farmers.append(farmer)
                farmer.primary_village_collective_id = self.unique_id
                
                # 记录效率用于后续决策
                if not hasattr(farmer, '_village_collective_efficiency'):
                    farmer._village_collective_efficiency = {}
                farmer._village_collective_efficiency[self.unique_id] = efficiency

    def _is_primary_collective_for_farmer(self, farmer) -> bool:
        """Avoid duplicate registration by assigning each farmer to a primary village."""
        if self.assignment_mode != 'nearest_only':
            return True

        villages = getattr(self.model, 'village_collectives', None)
        if not villages or len(villages) <= 1:
            return True

        best_village = None
        best_distance = None
        for village in villages:
            if village is None:
                continue
            distance = village._calculate_distance_to_farmer(farmer)
            score = (distance, str(village.unique_id))
            if best_distance is None or score < best_distance:
                best_distance = score
                best_village = village
        return best_village is None or best_village.unique_id == self.unique_id

    def _blend_network_information(self):
        """Blend neighboring-village signals into local market expectations."""
        if not self.network_neighbors:
            self.network_reference_price = 0.0
            return

        neighbor_prices = [
            float(getattr(neighbor, 'expected_rental_price', 0.0) or 0.0)
            for neighbor in self.network_neighbors
            if neighbor is not None and getattr(neighbor, 'expected_rental_price', 0.0) > 0
        ]
        if not neighbor_prices:
            self.network_reference_price = 0.0
            return

        reference_price = float(np.mean(neighbor_prices))
        influence = min(max(float(self.network_influence_weight or 0.0), 0.0), 0.5)
        self.network_reference_price = reference_price
        self.expected_rental_price = (
            (1 - influence) * self.expected_rental_price + influence * reference_price
        )
        self.market_info['network_reference_rental_price'] = reference_price
        self.market_info['network_degree'] = len(self.network_neighbors)
        
    def _initialize_rational_expectation(self):
        """初始化理性预期机制"""
        try:
            from modules.expectation_mechanism import RationalExpectation
            self.expectation_mechanism = RationalExpectation(
                supply_elasticity=0.4,
                demand_elasticity=-0.3,
                floor_price_grain=2.58,
                policy_credibility=0.9,
                volatility_persistence=0.7
            )
        except ImportError:
            self.expectation_mechanism = None
        
    def step(self):
        """
        村集体年度行动 - 智能决策周期
        
        决策流程：
        1. 信息收集与预期形成
        2. 策略选择（基于多目标优化）
        3. 执行流转组织
        4. 收益核算与反馈
        5. 兜底崩溃检测
        """
        # 阶段0：信息收集与预期更新
        self._collect_market_information()
        self._blend_network_information()
        self._update_rational_expectations()
        
        # 阶段1：策略选择（基于集体经济最大化目标）
        self._select_optimal_strategy()
        self.dynamic_participation_rate = self._get_dynamic_participation_rate()
        self.dynamic_mediation_success_boost = self._get_dynamic_mediation_success_boost()
        
        # 阶段2：更新登记信息
        self._update_registration()
        
        # 阶段3：根据策略执行决策
        if self.role == VillageCollectiveRole.INTEGRATED:
            self._attempt_land_integration()
        elif self.role == VillageCollectiveRole.ACTIVE:
            self._facilitate_land_transfers()
        
        # 阶段4：收益核算
        self._calculate_collective_economy()
        
        # 阶段5：记录年度统计
        self._record_annual_statistics()

        # @new 2026-05-02: 阶段5.5 - 声誉学习年度更新
        # 处理过去一年的合同履约情况、更新对手 belief、年末衰减
        self._update_reputation_from_contracts()
        if self.reputation_system is not None:
            self.reputation_system.annual_decay()

        # @new 2025-12-22: 阶段6：兜底崩溃检测
        self._check_collapse_condition()

    def _update_reputation_from_contracts(self):
        """
        @new 2026-05-02: 从过去一年的合同履约情况更新声誉信念。

        履约判定逻辑：
            - 若 patch.leased_to 仍存在且租约未中断 → 履约成功
            - 若中途解约或租金拖欠（rent_paid < rent_owed × 0.9） → 履约失败

        该方法在 mechanism_switches.use_reputation_village = False 时是 no-op。
        """
        if self.reputation_system is None:
            return

        # 从全局土地清算所或本村注册农户中提取履约证据
        try:
            for record in self.cross_village_transfer_records[-50:]:
                if not isinstance(record, dict):
                    continue
                lessor_id = record.get('source_farmer_id') or record.get('lessor_id')
                lessee_id = record.get('receiver_id') or record.get('lessee_id')
                fulfilled = bool(record.get('contract_fulfilled', True))
                if lessor_id is not None:
                    self.reputation_system.record_outcome(lessor_id, fulfilled)
                if lessee_id is not None:
                    self.reputation_system.record_outcome(lessee_id, fulfilled)
        except Exception as exc:
            if not hasattr(self.model, '_reputation_runtime_errors'):
                self.model._reputation_runtime_errors = []
            self.model._reputation_runtime_errors.append({
                'village_id': getattr(self, 'unique_id', None),
                'error': str(exc),
                'phase': 'update_reputation',
            })

    def get_reputation_priority_boost(self, agent_id) -> float:
        """
        @new 2026-05-02: 撮合优先级修正接口。

        被 land_clearing_house 或本类的撮合方法调用，对 base_score
        做修正：可信对手优先匹配，不可信对手降低优先级。

        Returns:
            浮点修正量。当 reputation_system 未启用时返回 0.0（向后兼容）。
        """
        if self.reputation_system is None:
            return 0.0
        return self.reputation_system.matching_priority_boost(agent_id)
    
    def _check_collapse_condition(self):
        """
        @new 2025-12-22: 检测村集体兜底崩溃条件
        
        核心机制：
        1. 计算托管土地占比
        2. 超过阈值触发崩溃
        3. 崩溃导致：基础设施衰退 → 周边地块yield下降
        
        理论依据：
        - 集体行动理论（Ostrom, 1990）：公共资源管理崩溃
        - 制度经济学：超过承载能力会导致制度失效
        """
        if not self.enable_collapse:
            return
        
        # 1. 更新托管面积统计
        self._update_custody_statistics()
        
        # 2. 计算托管比例
        custody_ratio = self.custody_land_area / max(self.total_village_area, 1)
        
        # 3. 检测崩溃条件
        if custody_ratio > self.max_custody_ratio and not self.is_collapsed:
            self._trigger_collapse()
        
        # 4. 如果已崩溃，持续执行惩罚效应
        if self.is_collapsed:
            self._apply_collapse_effects()
    
    def _update_custody_statistics(self):
        """更新托管土地统计"""
        # 统计托管土地面积
        custody_area = 0
        
        # 从integrated_land_pool统计
        for patch in self.integrated_land_pool:
            custody_area += getattr(patch, 'area', 1)
        
        # 从model中统计无主/抛荒土地
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'land_patches'):
            for patch in self.model.environment.land_patches:
                # 无主土地或抛荒土地视为村集体托管
                if getattr(patch, 'owner', None) is None or getattr(patch, 'is_abandoned', False):
                    custody_area += getattr(patch, 'area', 1)
        
        self.custody_land_area = custody_area
    
    def _trigger_collapse(self):
        """
        触发村集体兜底崩溃
        
        后果：
        1. 基础设施维护停止
        2. 灌溉系统失效
        3. 土地质量下降
        """
        self.is_collapsed = True
        self.collapse_year = getattr(self.model, 'step_count', 0)
        
        print(f"  [🔴 村集体崩溃] {self.village_name}: "
              f"托管面积{self.custody_land_area:.0f}亩 ({self.custody_land_area/self.total_village_area*100:.1f}%)，"
              f"超过承载上限{self.max_custody_ratio*100:.0f}%")
        
        # 立即产生效果
        self.infrastructure_condition = 0.7  # 初始衰退到70%
    
    def _apply_collapse_effects(self):
        """
        应用崩溃后的持续效应
        
        效应传导：
        1. 基础设施持续衰退
        2. 周边地块产量惩罚
        """
        # 1. 基础设施持续衰退
        self.infrastructure_condition = max(0.3, 
            self.infrastructure_condition * (1 - self.infrastructure_decay_rate))
        
        # 2. 对周边地块应用产量惩罚
        yield_penalty = self.collapse_yield_penalty * (1 - self.infrastructure_condition)
        
        if hasattr(self.model, 'environment') and hasattr(self.model.environment, 'land_patches'):
            for patch in self.model.environment.land_patches:
                # 对所有本村地块应用惩罚
                if hasattr(patch, 'yield_multiplier'):
                    # 产量乘数降低
                    patch.yield_multiplier = max(0.5, 
                        getattr(patch, 'yield_multiplier', 1.0) * (1 - yield_penalty))
                
                # 影响土地质量（永久性损害）
                if hasattr(patch, 'soil_quality'):
                    patch.soil_quality = max(0.3, 
                        patch.soil_quality * (1 - yield_penalty * 0.1))
        
        # 记录崩溃状态
        years_since_collapse = getattr(self.model, 'step_count', 0) - (self.collapse_year or 0)
        if years_since_collapse % 5 == 0:
            print(f"  [⚠️ 崩溃持续] {self.village_name}: "
                  f"基础设施状况{self.infrastructure_condition*100:.0f}%，"
                  f"已持续{years_since_collapse}年")
    
    def get_collapse_status(self) -> dict:
        """返回崩溃状态信息"""
        return {
            'is_collapsed': self.is_collapsed,
            'collapse_year': self.collapse_year,
            'custody_ratio': self.custody_land_area / max(self.total_village_area, 1),
            'infrastructure_condition': self.infrastructure_condition,
            'custody_area': self.custody_land_area,
            'total_area': self.total_village_area
        }
    
    def _collect_market_information(self):
        """
        收集市场信息
        
        村集体作为信息枢纽，汇集本村及周边市场信息
        """
        if not hasattr(self.model, 'environment'):
            return
        
        env = self.model.environment
        
        # 获取价格信息
        if hasattr(env, 'get_market_price'):
            grain_price = env.get_market_price('crop_price')
            if grain_price:
                self.market_info['grain_price'] = grain_price
            
            cash_price = env.get_market_price('cash_crop')
            if cash_price:
                self.market_info['cash_crop_price'] = cash_price
        
        # 获取政策信息
        if hasattr(env, 'get_policy_param'):
            subsidy_rate = env.get_policy_param('grain_subsidies_rate')
            if subsidy_rate is not None:
                self.market_info['policy_subsidy_rate'] = subsidy_rate
        
        # 从流转记录估算平均租金
        if self.land_transfer_records:
            recent_records = self.land_transfer_records[-10:]
            total_value = sum(r.get('rental_value', 0) for r in recent_records)
            total_area = sum(r.get('area', 1) for r in recent_records)
            if total_area > 0:
                self.market_info['avg_rental_price'] = total_value / total_area
    
    def _update_rational_expectations(self):
        """
        基于理性预期更新租金预期
        
        使用理性预期假说：E[P_{t+1}] = f(供需, 政策, 历史)
        """
        if self.expectation_mechanism is None:
            # 回退：简单移动平均
            if len(self.rental_price_history) >= 3:
                self.expected_rental_price = np.mean(self.rental_price_history[-3:])
            return
        
        # 使用理性预期机制
        current_price = self.market_info['avg_rental_price']
        additional_info = {
            'crop_type': 'land_rental',  # 土地租赁
            'policy_signal': self.market_info['policy_subsidy_rate']
        }
        
        self.expected_rental_price = self.expectation_mechanism.form_expectation(
            historical_values=self.rental_price_history,
            current_value=current_price,
            additional_info=additional_info
        )
        
        # 记录历史
        self.rental_price_history.append(current_price)
        if len(self.rental_price_history) > 20:
            self.rental_price_history = self.rental_price_history[-20:]

    def _get_policy_support_signal(self) -> float:
        reference_rate = max(1e-6, float(self.policy_reference_rate or 0.20))
        subsidy_rate = max(0.0, float(self.market_info.get('policy_subsidy_rate', 0.0) or 0.0))
        return max(0.0, min(1.0, subsidy_rate / reference_rate))

    def get_transfer_congestion_signal(self) -> float:
        custody_ratio = float(self.custody_land_area or 0.0) / max(float(self.total_village_area or 1.0), 1.0)
        max_ratio = max(0.05, float(self.max_custody_ratio or 0.4))
        congestion_pressure = min(1.0, custody_ratio / max_ratio)
        infrastructure = max(0.0, min(1.0, float(self.infrastructure_condition or 1.0)))
        collapse_penalty = 0.45 if self.is_collapsed else 0.0
        signal = 0.60 * (1.0 - congestion_pressure) + 0.40 * infrastructure - collapse_penalty
        self.transfer_congestion_signal = max(0.0, min(1.0, signal))
        return self.transfer_congestion_signal

    def get_governance_capacity_signal(self) -> float:
        policy_signal = self._get_policy_support_signal()
        congestion_signal = self.get_transfer_congestion_signal()
        network_support = 0.0
        if hasattr(self.model, 'get_village_network_density'):
            try:
                network_support = float(self.model.get_village_network_density() or 0.0)
            except Exception:
                network_support = 0.0
        posture = self._get_governance_coordination_posture()
        signal = (
            0.30 * max(0.0, min(1.0, posture))
            + 0.30 * congestion_signal
            + 0.25 * max(0.0, min(1.0, float(self.infrastructure_condition or 1.0)))
            + 0.10 * max(0.0, min(1.0, network_support))
            + 0.05 * policy_signal
        )
        self.governance_capacity_signal = max(0.0, min(1.0, signal))
        return self.governance_capacity_signal

    def _get_dynamic_participation_rate(self) -> float:
        governance_signal = self.get_governance_capacity_signal()
        policy_signal = self._get_policy_support_signal()
        dynamic_rate = float(self.participation_rate or 0.0) * (0.75 + 0.45 * governance_signal + 0.20 * policy_signal)
        self.dynamic_participation_rate = max(0.05, min(0.85, dynamic_rate))
        return self.dynamic_participation_rate

    def _get_dynamic_mediation_success_boost(self, demander=None) -> float:
        governance_signal = self.get_governance_capacity_signal()
        demand_signal = 0.0
        if isinstance(demander, dict):
            demand_signal = float(demander.get('willingness', 0.0) or 0.0)
        elif demander is not None:
            demand_signal = float(getattr(demander, 'expansion_willingness', 0.0) or 0.0)

        grain_alignment = 0.5
        target_agent = demander.get('agent') if isinstance(demander, dict) else demander
        if target_agent is not None:
            grain_alignment = max(0.2, min(1.0, self._get_agent_grain_share(target_agent)))

        boost = float(self.mediation_success_boost or 0.0) * (
            0.80 + 0.35 * governance_signal + 0.15 * demand_signal + 0.10 * grain_alignment
        )
        self.dynamic_mediation_success_boost = max(0.05, min(0.45, boost))
        return self.dynamic_mediation_success_boost

    def _get_agent_grain_share(self, agent) -> float:
        """Estimate the current operated grain share of a demander/operator."""
        if agent is None:
            return 0.0

        crop_portfolio = getattr(agent, 'crop_portfolio', None)
        if isinstance(crop_portfolio, dict) and crop_portfolio:
            grain_share = 0.0
            for crop_key, weight in crop_portfolio.items():
                crop_name = str(getattr(crop_key, 'value', crop_key)).strip().lower()
                if 'grain' in crop_name:
                    grain_share += float(weight or 0.0)
            return max(0.0, min(1.0, grain_share))

        grain_area = float(getattr(agent, 'grain_area', 0.0) or 0.0)
        if hasattr(agent, 'get_operated_land_area'):
            try:
                operated_area = float(agent.get_operated_land_area() or 0.0)
            except Exception:
                operated_area = 0.0
            if operated_area > 0:
                return max(0.0, min(1.0, grain_area / operated_area))

        if hasattr(agent, 'planned_crop'):
            crop_name = str(getattr(agent.planned_crop, 'value', agent.planned_crop)).strip().lower()
            return 1.0 if crop_name == 'grain' else 0.0

        if hasattr(agent, 'grain_weight'):
            return max(0.0, min(1.0, float(getattr(agent, 'grain_weight', 0.0) or 0.0)))

        return 0.0

    def _get_patch_current_operator(self, patch):
        """Resolve the actor currently making production decisions on a patch."""
        for attr_name in ('current_operator', 'leased_to', 'owner'):
            operator = getattr(patch, attr_name, None)
            if operator is not None:
                return operator
        return None
    
    def _select_optimal_strategy(self):
        """
        选择最优策略 - 基于多目标优化
        
        目标函数：
        U = α·Revenue + β·Welfare + γ·GrainSecurity + δ·ScaleEfficiency
        
        策略选择规则：
        1. 政府强调粮食安全 → GRAIN_SECURITY
        2. 村民收入偏低 → BALANCED_WELFARE
        3. 集体经济薄弱 → MAXIMIZE_REVENUE
        4. 土地细碎严重 → SCALE_PROMOTION
        """
        # 计算各目标的当前状态得分
        scores = self._calculate_objective_scores()
        
        # 识别最需要改善的目标
        min_score = min(scores.values())
        min_objective = [k for k, v in scores.items() if v == min_score][0]
        
        # 策略选择
        if min_objective == 'grain_security':
            self.current_strategy = CollectiveStrategy.GRAIN_SECURITY
        elif min_objective == 'welfare':
            self.current_strategy = CollectiveStrategy.BALANCED_WELFARE
        elif min_objective == 'revenue':
            self.current_strategy = CollectiveStrategy.MAXIMIZE_REVENUE
        else:  # scale_efficiency
            self.current_strategy = CollectiveStrategy.SCALE_PROMOTION
    
    def _calculate_objective_scores(self) -> Dict[str, float]:
        """
        计算各目标的当前得分（0-1）
        
        用于策略选择决策
        """
        scores = {}
        
        # 1. 收益得分
        target_revenue = 50000  # 目标年收益5万元
        actual_revenue = self.economy_state.annual_service_fee + self.economy_state.land_rental_income
        scores['revenue'] = min(1.0, actual_revenue / target_revenue) if target_revenue > 0 else 0.5
        
        # 2. 村民福利得分（用平均收入代理）
        if hasattr(self.model, 'farmers') and self.model.farmers:
            avg_income = np.mean([
                get_farmer_net_income(f) for f in self.model.farmers
            ])
            target_income = 30000  # 目标人均收入3万元
            scores['welfare'] = min(1.0, avg_income / target_income) if target_income > 0 else 0.5
        else:
            scores['welfare'] = 0.5
        
        # 3. 粮食安全得分（种粮面积占比）
        if hasattr(self.model, 'land_patches'):
            total_grain_area = 0
            total_area = 0
            for patch in getattr(self.model, 'land_patches', []) or []:
                operator = self._get_patch_current_operator(patch)
                patch_area = float(getattr(patch, 'area', 0.0) or 0.0)
                if operator is None or patch_area <= 0:
                    continue
                total_area += patch_area
                total_grain_area += patch_area * self._get_agent_grain_share(operator)
            
            grain_ratio = total_grain_area / total_area if total_area > 0 else 0
            # 目标：粮食种植面积占比60%以上
            scores['grain_security'] = min(1.0, grain_ratio / 0.6)
        else:
            scores['grain_security'] = 0.5
        
        # 4. 规模效率得分（平均经营规模）
        if hasattr(self.model, 'farmers') and self.model.farmers:
            avg_scale = np.mean([
                f.get_operated_land_area() for f in self.model.farmers
                if hasattr(f, 'get_operated_land_area')
            ])
            target_scale = 15  # 目标户均15亩
            scores['scale_efficiency'] = min(1.0, avg_scale / target_scale) if target_scale > 0 else 0.5
        else:
            scores['scale_efficiency'] = 0.5
        
        return scores
    
    def calculate_objective_function(self) -> float:
        """
        计算集体经济最大化目标函数值
        
        U = α·Revenue + β·Welfare + γ·GrainSecurity + δ·ScaleEfficiency
        
        返回值：加权综合得分（0-1）
        """
        scores = self._calculate_objective_scores()
        
        U = (
            self.alpha * scores.get('revenue', 0) +
            self.beta * scores.get('welfare', 0) +
            self.gamma * scores.get('grain_security', 0) +
            self.delta * scores.get('scale_efficiency', 0)
        )
        
        return U
    
    def _is_patch_available_for_integration(self, patch):
        """判断地块是否可整合"""
        # 未流转
        if patch.leased_to is not None:
            return False
        # 未撂荒
        if getattr(patch, 'abandoned', False):
            return False
        # 面积较小（细碎）
        if patch.area > self.min_integrated_area:
            return False
        # 土地质量可接受
        if getattr(patch, 'soil_quality', 0.5) < 0.3:
            return False
        
        # 检查owner是否愿意流转
        owner = patch.owner
        if owner is None:
            return False
        
        # @new 基于连续农业权重的流转意愿判断（取代离散策略比较）
        # 农业权重越低（非农收入占比越高），流转意愿越强
        if hasattr(owner, 'get_agricultural_weight'):
            ag_weight = owner.get_agricultural_weight()
            # 非农权重 = 1 - 农业权重，作为流转意愿基础
            non_ag_weight = 1 - ag_weight
            
            # 年龄因素：60岁以上增加流转意愿
            age_bonus = 0.0
            if hasattr(owner, 'age') and owner.age >= 60:
                age_bonus = 0.3
            
            # 综合流转意愿概率
            transfer_prob = min(1.0, non_ag_weight * 0.8 + age_bonus)
            
            return rng_random(self) < transfer_prob
        
        return False
    
    def _perform_integration(self, patches):
        """
        执行土地整合
        简化实现：将patches标记为村集体托管
        """
        # 记录整合信息
        self.integrated_land_pool = patches
        
        # 通知土地清算所有村集体托管的土地
        if hasattr(self.model, 'land_clearing_house'):
            for patch in patches:
                # 在清算所中登记为村集体供给
                self._register_integrated_supply(patch)
    
    def _register_integrated_supply(self, patch):
        """
        将整合土地登记到流转市场
        整合后的土地以村集体名义统一出租
        """
        # 计算整合后的保留价（比分散出租低，吸引规模经营者）
        base_rent = 300  # 基准租金 元/亩
        quality_factor = getattr(patch, 'soil_quality', 0.7)
        
        # 整合土地有规模优势，租金可略低
        discount_for_scale = 0.9  # 90%的价格（降10%）
        reservation_price = base_rent * quality_factor * discount_for_scale * patch.area
        
        # 提交到市场
        if hasattr(self.model, 'land_clearing_house'):
            self.model.land_clearing_house.add_supply_offer(
                patch, 
                self,  # 供给方是村集体
                reservation_price
            )
    
    def _facilitate_land_transfers(self):
        """
        @theory 促成土地流转
        
        村集体作为中介，降低信息搜索成本：
        1. 收集供需信息
        2. 匹配潜在交易双方
        3. 协调价格谈判
        """
        # 识别潜在供给方（愿意出租的农户）
        potential_suppliers = self._identify_potential_suppliers()
        
        # 识别潜在需求方（想要扩大规模的经营者）
        potential_demanders = self._identify_potential_demanders()
        
        # 尝试撮合交易
        self._match_suppliers_demanders(potential_suppliers, potential_demanders)
    
    def _identify_potential_suppliers(self):
        """识别潜在土地供给方"""
        suppliers = []
        v1 = self.model.config.get("runtime_fixed_rule", {}).get("national_v1") is True
        
        for farmer in self.registered_farmers:
            if v1 and getattr(farmer, "fixed_rule_current_action", None) != "exit":
                continue
            # 检查是否有未流转的土地
            unrented_patches = []
            for patch in getattr(farmer, 'land_patches', []):
                if patch.owner == farmer and patch.leased_to is None:
                    if not getattr(patch, 'abandoned', False):
                        unrented_patches.append(patch)
            
            if not unrented_patches:
                continue
            
            # 判断流转意愿
            willingness = getattr(farmer, 'get_transfer_out_willingness', lambda: 0)()
            if willingness > 0.3:  # 意愿较高
                suppliers.append({
                    'farmer': farmer,
                    'patches': unrented_patches,
                    'willingness': willingness
                })
        
        return suppliers
    
    def _identify_potential_demanders(self):
        """识别潜在土地需求方"""
        demanders = []
        
        # 规模农户
        for farmer in self.registered_farmers:
            from agents.farmer_agent import LivelihoodStrategy
            if hasattr(farmer, 'livelihood_strategy'):
                if farmer.livelihood_strategy == LivelihoodStrategy.PURE_FARMER:
                    willingness = getattr(farmer, 'get_transfer_in_willingness', lambda: 0)()
                    if willingness > 0.3:
                        demanders.append({
                            'agent': farmer,
                            'type': 'farmer',
                            'willingness': willingness
                        })
        
        # 农业企业
        if hasattr(self.model, 'agricultural_enterprises'):
            for enterprise in self.model.agricultural_enterprises:
                primary_village = None
                if hasattr(self.model, 'get_primary_village_for_agent'):
                    primary_village = self.model.get_primary_village_for_agent(enterprise)
                if primary_village is not None and primary_village.unique_id != self.unique_id:
                    continue
                willingness = 0.0
                if hasattr(enterprise, 'preview_land_expansion_willingness'):
                    try:
                        willingness = float(enterprise.preview_land_expansion_willingness() or 0.0)
                    except Exception:
                        willingness = float(getattr(enterprise, 'expansion_willingness', 0.0) or 0.0)
                else:
                    willingness = float(getattr(enterprise, 'expansion_willingness', 0.0) or 0.0)
                if willingness > 0.3:
                    demanders.append({
                        'agent': enterprise,
                        'type': 'enterprise',
                        'willingness': willingness
                    })
        
        return demanders
    
    def _match_suppliers_demanders(self, suppliers, demanders):
        """
        撮合供需双方
        """
        if not suppliers or not demanders:
            return
        
        transfers_this_step = 0
        
        # 简单匹配策略：按意愿排序后依次匹配
        sorted_suppliers = sorted(suppliers, key=lambda x: -x['willingness'])
        if self.current_strategy == CollectiveStrategy.GRAIN_SECURITY:
            sorted_demanders = sorted(
                demanders,
                key=lambda x: (
                    -self._get_agent_grain_share(x.get('agent')),
                    -float(x.get('willingness', 0.0) or 0.0),
                ),
            )
        else:
            sorted_demanders = sorted(demanders, key=lambda x: -x['willingness'])
        
        remaining_capacity = {
            id(demander.get('agent')): self._demander_capacity_area(demander)
            for demander in sorted_demanders
        }

        for supplier in sorted_suppliers:
            if not sorted_demanders:
                break

            supplier_area = sum(
                float(getattr(patch, 'area', 0.0) or 0.0)
                for patch in supplier.get('patches', []) or []
            )
            eligible_demanders = [
                demander for demander in sorted_demanders
                if remaining_capacity.get(id(demander.get('agent')), 0.0) >= supplier_area
            ]
            if not eligible_demanders:
                continue
            
            # 村集体参与的概率
            participation_rate = self._get_dynamic_participation_rate()
            if rng_random(self) > participation_rate:
                continue  # 这笔交易不通过村集体
            
            # 选择最匹配的需求方
            demander = eligible_demanders[0]
            
            # 尝试促成交易（提高成功率）
            base_success = 0.5
            boosted_success = min(1.0, base_success + self._get_dynamic_mediation_success_boost(demander))
            
            if rng_random(self) < boosted_success:
                # 交易成功，记录
                leased_area = float(self._record_transfer(supplier, demander) or 0.0)
                if leased_area > 0.0:
                    transfers_this_step += 1
                    demander_id = id(demander.get('agent'))
                    remaining_capacity[demander_id] = max(
                        0.0,
                        remaining_capacity.get(demander_id, 0.0) - leased_area,
                    )
        
        self.step_transfers_facilitated = transfers_this_step

    def _demander_capacity_area(self, demander):
        agent = demander.get('agent') if isinstance(demander, dict) else demander
        demander_type = demander.get('type') if isinstance(demander, dict) else None
        if demander_type == 'enterprise':
            context = getattr(agent, 'land_expansion_context', {}) or {}
            try:
                return max(0.0, float(context.get('upper_headroom_area', 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0
        if demander_type == 'farmer' and hasattr(agent, 'get_operated_land_area'):
            try:
                current_area = max(0.0, float(agent.get_operated_land_area() or 0.0))
                scale_info = agent.calculate_optimal_operated_scale()
                optimal_area = max(0.0, float(scale_info.get('optimal_scale', current_area) or 0.0))
                return max(0.0, optimal_area - current_area)
            except (AttributeError, TypeError, ValueError):
                return 0.0
        return 0.0
    
    def _record_transfer(self, supplier, demander):
        """
        记录流转并收取服务费
        
        @updated 2025-12-01: 基于策略的差异化定价
        @critical 2025-01: 设置流转渠道为村集体流转
        """
        total_area = sum(p.area for p in supplier['patches'])
        
        # 基于策略调整租金
        base_rent = self.expected_rental_price  # 使用预期租金
        
        if self.current_strategy == CollectiveStrategy.MAXIMIZE_REVENUE:
            # 收益最大化：提高租金
            adjusted_rent = base_rent * 1.1
        elif self.current_strategy == CollectiveStrategy.GRAIN_SECURITY:
            # 粮食安全：对种粮主体优惠
            is_grain_producer = self._check_grain_producer(demander['agent'])
            adjusted_rent = base_rent * 0.9 if is_grain_producer else base_rent
        elif self.current_strategy == CollectiveStrategy.SCALE_PROMOTION:
            # 规模促进：对大经营主体优惠
            if demander['type'] == 'enterprise':
                adjusted_rent = base_rent * 0.85  # 企业优惠15%
            else:
                adjusted_rent = base_rent
        else:  # BALANCED_WELFARE
            adjusted_rent = base_rent
        
        transaction_value = total_area * adjusted_rent
        
        # @critical 2025-01: 执行流转并标记为村集体渠道
        # 通过村集体流转的土地会有更严格的种粮约束
        leased_area = 0.0
        parcel_ids = [] if self.model.config.get('runtime_fixed_rule', {}).get('national_v1') else None
        for patch in supplier['patches']:
            patch_rent = adjusted_rent * patch.area if action_income_enabled(self.model.config) else adjusted_rent
            lease_applied = True
            if hasattr(patch, 'set_leased'):
                # 使用新的set_leased接口，标记为村集体流转
                lease_applied = patch.set_leased(
                    to_agent=demander['agent'],
                    contract_years=3,  # 默认3年合同
                    rental_price=patch_rent,
                    transfer_channel='village_collective'  # 关键：标记流转渠道
                )
            else:
                # 向后兼容：旧版LandPatch
                clearing_house = getattr(self.model, 'land_clearing_house', None)
                if (
                    clearing_house is not None and
                    hasattr(clearing_house, '_check_transfer_rate_limit') and
                    not clearing_house._check_transfer_rate_limit(
                        self.model,
                        float(getattr(patch, 'area', 0.0) or 0.0),
                        transfer_channel='village_collective',
                        demander=demander['agent'],
                    )
                ):
                    lease_applied = False
                else:
                    patch.leased_to = demander['agent']
                    patch.current_operator = demander['agent']
                    patch.rental_price = patch_rent
                    if hasattr(patch, 'transfer_channel'):
                        patch.transfer_channel = 'village_collective'
            if lease_applied is False:
                continue
            leased_area += float(getattr(patch, 'area', 0.0) or 0.0)
            if parcel_ids is not None:
                parcel_ids.append(str(patch.get_patch_id()))

        if leased_area <= 0:
            return 0.0
        if hasattr(self.model, 'invalidate_land_tenure_caches'):
            self.model.invalidate_land_tenure_caches()
        
        # 收取服务费
        transaction_value = leased_area * adjusted_rent
        service_fee = transaction_value * self.service_fee_rate
        self.service_fee_collected += service_fee
        
        # 更新经济状态
        self.economy_state.annual_service_fee += service_fee
        self.economy_state.annual_transferred_area += leased_area
        self.economy_state.annual_facilitated_transfers += 1
        
        source_patch_count = len(supplier['patches'])
        source_avg_patch_size = leased_area / max(1, source_patch_count)
        source_fragmented_area = sum(
            float(getattr(patch, 'area', 0.0) or 0.0)
            for patch in supplier['patches']
            if float(getattr(patch, 'area', 0.0) or 0.0) <= self.min_integrated_area
        )
        basic_farmland_area = sum(
            float(getattr(patch, 'area', 0.0) or 0.0)
            for patch in supplier['patches']
            if hasattr(patch, 'is_basic_farmland') and bool(patch.is_basic_farmland())
        )
        grain_obligation_ratio = (
            self.basic_farmland_grain_requirement_ratio
            if basic_farmland_area > 0 else self.grain_requirement_ratio
        )
        grain_obligation_area = leased_area * grain_obligation_ratio if self.grain_requirement_enabled else 0.0
        receiver_enterprise_type = ''
        receiver_operator_type = demander['type']
        if hasattr(demander['agent'], 'enterprise_type'):
            raw_type = getattr(demander['agent'], 'enterprise_type', None)
            receiver_enterprise_type = str(
                getattr(raw_type, 'value', getattr(raw_type, 'name', raw_type)) or ''
            )
            if 'cooperative' in receiver_enterprise_type.lower():
                receiver_operator_type = 'cooperative'
            else:
                receiver_operator_type = 'agri_corporation'
        chain_label = f"fragmented_land_to_village_collective_to_{receiver_operator_type}"

        # 记录流转信息
        self.land_transfer_records.append({
            **({'contract_patch_ids': parcel_ids} if parcel_ids is not None else {}),
            'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
            'supplier_id': supplier['farmer'].unique_id,
            'demander_id': demander['agent'].unique_id,
            'demander_type': demander['type'],
            'receiver_operator_type': receiver_operator_type,
            'receiver_enterprise_type': receiver_enterprise_type,
            'source_village_id': self.unique_id,
            'receiver_village_id': self.unique_id,
            'area': leased_area,
            'source_patch_count': source_patch_count,
            'source_avg_patch_size': source_avg_patch_size,
            'source_fragmented_area': source_fragmented_area,
            'basic_farmland_area': basic_farmland_area,
            'grain_obligation_enabled': self.grain_requirement_enabled,
            'grain_obligation_ratio': grain_obligation_ratio,
            'grain_obligation_area': grain_obligation_area,
            'rental_value': transaction_value,
            'rental_price_per_mu': adjusted_rent,
            'service_fee': service_fee,
            'estimated_owner_rent_income': max(0.0, transaction_value - service_fee),
            'operator_rent_payment': transaction_value,
            'rent_distribution_farmer_share': max(0.0, transaction_value - service_fee) / transaction_value if transaction_value > 0 else 0.0,
            'rent_distribution_collective_share': service_fee / transaction_value if transaction_value > 0 else 0.0,
            'strategy_used': self.current_strategy.value,
            'transfer_channel': 'village_collective',
            'governance_mode': 'within_village',
            'cross_village': False,
            'compliance_status': 'pending',
            'breach_detected': False,
            'monitoring_probability': self.grain_requirement_enforcement_strength,
            'chain_label': chain_label,
        })
        return leased_area
    
    def _check_grain_producer(self, agent) -> bool:
        """检查是否为种粮主体"""
        return self._get_agent_grain_share(agent) >= 0.5
    
    def _calculate_collective_economy(self):
        """
        核算集体经济收支

        收入来源：
        1. 土地流转服务费
        2. 集体资产出租收入
        3. 其他经营收入

        支出项目：
        1. 运营管理成本
        2. 公共服务支出
        3. @new 2026-06-25: 集体盈余向农户的转移支付
        """
        # 更新累计值
        self.economy_state.accumulated_service_fee += self.economy_state.annual_service_fee
        self.economy_state.total_facilitated_transfers += self.economy_state.annual_facilitated_transfers
        self.economy_state.total_transferred_area += self.economy_state.annual_transferred_area

        # 计算成功率
        if hasattr(self.model, 'farmers') and self.model.farmers:
            total_potential = len([f for f in self.model.farmers
                                   if getattr(f, 'get_transfer_out_willingness', lambda: 0)() > 0.3])
            if total_potential > 0:
                self.economy_state.transfer_success_rate = (
                    self.economy_state.annual_facilitated_transfers / total_potential
                )

        # 计算平均租金
        if self.economy_state.annual_transferred_area > 0:
            recent_records = self.land_transfer_records[-10:] if self.land_transfer_records else []
            if recent_records:
                total_value = sum(r.get('rental_value', 0) for r in recent_records)
                total_area = sum(r.get('area', 1) for r in recent_records)
                self.economy_state.avg_rental_price = total_value / total_area if total_area > 0 else 0

        # @new 2026-06-25: 集体盈余向农户的转移支付
        # 集体经济净收益的一部分按户分配，计入农户 transfer_income
        # 这闭合了村集体场景下收入核算残差过大的问题
        self._distribute_collective_surplus_to_farmers()

    def _distribute_collective_surplus_to_farmers(self):
        """将集体经济净盈余的一部分转移支付给注册农户。

        转移支付来源：累计服务费收入 - 运营成本后的净盈余。
        分配方式：按户均分给注册农户，计入 transfer_income。
        这使村集体场景下的收入核算可以追踪福利转移通道。

        @fix 2026-06-25: 使用 accumulated_service_fee (跨年累计) 而非
        annual_service_fee (当年值，短时 smoke 中可能为0)。
        """
        # 使用累计服务费 - 年度运营成本作为净盈余
        net_surplus = max(0.0, (
            self.economy_state.accumulated_service_fee
            - getattr(self, 'annual_operating_cost', 0.0)
        ))
        if net_surplus <= 0.0:
            return

        eligible_farmers = [
            f for f in getattr(self, 'registered_farmers', []) or []
            if hasattr(f, 'transfer_income')
        ]
        if not eligible_farmers:
            return

        # 分配比例：净盈余的 60% 返还农户（现实中的集体分红/福利）
        distribution_share = 0.60
        total_distribution = net_surplus * distribution_share
        per_farmer = total_distribution / len(eligible_farmers)

        for farmer in eligible_farmers:
            current = float(getattr(farmer, 'transfer_income', 0.0) or 0.0)
            farmer.transfer_income = current + per_farmer

        # 记录分配用于审计
        self._last_surplus_distribution = {
            'net_surplus': net_surplus,
            'distribution_share': distribution_share,
            'total_distributed': total_distribution,
            'per_farmer': per_farmer,
            'n_recipients': len(eligible_farmers),
        }
    
    def _update_registration(self):
        """
        更新农户和土地登记信息
        
        @update 2025-12-22: 使用空间邻近性进行登记
        现在只登记服务半径内的农户，体现空间服务约束
        """
        if not hasattr(self.model, 'farmers'):
            return
        
        # @fix 2025-12-22: 使用空间邻近性更新登记
        # 首先尝试基于空间位置的登记
        if hasattr(self, 'pos') and self.pos is not None:
            self.update_registered_farmers_by_proximity()
        else:
            # 回退到全部农户（兼容无空间位置的情况）
            self.registered_farmers = list(self.model.farmers)
        
        # 重置年度统计
        self.economy_state.annual_service_fee = 0.0
        self.economy_state.annual_facilitated_transfers = 0
        self.economy_state.annual_transferred_area = 0.0
    
    def _attempt_land_integration(self):
        """
        @theory 土地整合
        
        整合逻辑：
        1. 收集愿意流转的细碎地块
        2. 将空间相邻的地块整合成大块
        3. 统一对外出租给企业或规模户
        """
        if not hasattr(self.model, 'land_patches'):
            return
        
        # 收集可整合的地块
        available_for_integration = []
        for patch in self.model.land_patches:
            if self._is_patch_available_for_integration(patch):
                available_for_integration.append(patch)
        
        # 如果可整合面积不足，跳过
        total_available_area = sum(p.area for p in available_for_integration)
        if total_available_area < self.integration_threshold:
            return
        
        # 执行整合
        self._perform_integration(available_for_integration)
    
    def _record_annual_statistics(self):
        """记录年度统计"""
        # 计算目标函数值
        objective_value = self.calculate_objective_function()
        
        # 可以在这里添加更多统计逻辑
        if hasattr(self, 'objective_history'):
            self.objective_history.append(objective_value)
        else:
            self.objective_history = [objective_value]
    
    def get_statistics(self):
        """获取村集体统计信息"""
        return {
            'registered_farmers': len(self.registered_farmers),
            'total_transfers': len(self.land_transfer_records),
            'transfers_this_step': self.step_transfers_facilitated,
            'total_service_fee': self.service_fee_collected,
            'integrated_land_pool_size': len(self.integrated_land_pool),
            'integrated_land_area': sum(p.area for p in self.integrated_land_pool) if self.integrated_land_pool else 0,
            'current_strategy': self.current_strategy.value,
            'network_degree': len(self.network_neighbors),
            'network_reference_price': self.network_reference_price,
            'subsidy_pre_review_coverage': (
                self.annual_reviewed_claims / self.annual_claims_received
                if self.annual_claims_received > 0 else 0.0
            ),
            'subsidy_pre_review_approval_rate': (
                self.annual_approved_claims / self.annual_reviewed_claims
                if self.annual_reviewed_claims > 0 else 0.0
            ),
            'cross_village_contracts': self.annual_cross_village_contracts,
            'cross_village_transfer_area': self.annual_cross_village_transfer_area,
            'cross_village_partners': len(self.annual_cross_village_partners),
            'objective_function_value': self.calculate_objective_function(),
            'economy_state': {
                'annual_service_fee': self.economy_state.annual_service_fee,
                'accumulated_service_fee': self.economy_state.accumulated_service_fee,
                'transfer_success_rate': self.economy_state.transfer_success_rate,
                'avg_rental_price': self.economy_state.avg_rental_price
            }
        }
    
    def get_transfer_history(self):
        """获取流转历史记录"""
        return self.land_transfer_records.copy()
