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
@new 2025-12-28: 土地置换机制 (Land Swapping)
参考：Deng et al. (2024, Nature Food) "Managing fragmented croplands"
主动识别并促成地块置换，解决细碎化问题


@enhancement 2025-11-30: 合约理论完善
- 新增分成合约类型（Sharecropping）
- 实现公开市场违约反悔机制
- 模拟金钱对传统关系的侵蚀
参考：Stiglitz (1974), "Incentives and Risk Sharing in Sharecropping"

@new 2025-11-30: 向量化匹配优化模式
- 大规模模拟时可启用向量化匹配提升性能
- 使用numpy矩阵运算替代O(N²)循环

@new 2025-12-22: R-tree/网格空间索引
- 加速邻近地块查询，从O(N²)降至O(log N)
- 支持可选rtree库，无依赖时回退到网格索引
"""

import numpy as np
from enum import Enum
from canonical.constants.policy_constants import BASELINE_SUBSIDY_NORMALIZATION_ANCHOR
from core.reproducibility import rng_random, rng_sample, rng_shuffle
from core.income_utils import (
    get_farmer_net_income,
    get_farmer_operated_land_area,
    is_professional_large_operator,
)

# @new 2025-12-22: 空间索引支持
try:
    from modules.spatial_index import PatchSpatialIndex, build_global_patch_index
    SPATIAL_INDEX_AVAILABLE = True
except ImportError:
    SPATIAL_INDEX_AVAILABLE = False
    PatchSpatialIndex = None

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
        self.last_supply_count = 0
        self.last_demand_count = 0
        self.last_supply_area = 0.0
        self.last_demand_area = 0.0
        self.last_matched_area = 0.0
        self.last_unmatched_supply_area = 0.0
        self.last_unmet_demand_area = 0.0
        self._reset_signal_caches()
        
        # 市场统计（新增）
        self.market_statistics = {
            'total_transactions': 0,
            'total_area_transferred': 0,
            'average_price': 0,
            'average_transaction_price': 0,
            'average_rent_per_mu': 0,
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
        
        # @new 2025-12-22: 空间索引
        self._spatial_index = None
        self._use_spatial_index = SPATIAL_INDEX_AVAILABLE  # 如果可用则自动启用

    def _reset_signal_caches(self):
        """Clear per-market signal caches used by repeated ranking/scoring calls."""
        self._demander_grain_alignment_cache = {}
        self._demander_productive_capacity_cache = {}
        self._grain_policy_support_cache = {}
        self._policy_weighted_bid_score_cache = {}

    def _capture_flow_audit(self):
        supply_by_patch = {
            self._get_patch_key(offer.get('patch')): offer.get('patch')
            for offer in self.supply_offers
            if offer.get('patch') is not None
        }
        demand_by_patch = {
            self._get_patch_key(offer.get('patch')): offer.get('patch')
            for offer in self.demand_offers
            if offer.get('patch') is not None
        }
        matched_by_patch = {
            self._get_patch_key(transaction.get('patch')): transaction.get('patch')
            for transaction in self.transactions
            if transaction.get('patch') is not None
        }

        def area(patches):
            return float(sum(float(getattr(patch, 'area', 0.0) or 0.0) for patch in patches.values()))

        self.last_supply_area = area(supply_by_patch)
        self.last_demand_area = area(demand_by_patch)
        self.last_matched_area = area(matched_by_patch)
        self.last_unmatched_supply_area = area({
            key: patch for key, patch in supply_by_patch.items() if key not in matched_by_patch
        })
        self.last_unmet_demand_area = area({
            key: patch for key, patch in demand_by_patch.items() if key not in matched_by_patch
        })

    def build_spatial_index(self, patches: list):
        """
        @new 2025-12-22: 构建空间索引以加速邻近查询
        
        应在模型初始化时调用，传入所有地块列表
        
        Args:
            patches: LandPatch 对象列表
        """
        if not SPATIAL_INDEX_AVAILABLE:
            return
        
        self._spatial_index = PatchSpatialIndex(patches, cell_size=self.max_spatial_distance)
        self._use_spatial_index = True
        
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

    def _get_agent_key(self, agent) -> str:
        """Return a stable sortable key for an agent."""
        return str(getattr(agent, 'unique_id', agent))

    def _get_patch_key(self, patch) -> str:
        """Return a stable sortable key for a patch."""
        getter = getattr(patch, 'get_patch_id', None)
        if callable(getter):
            return str(getter())
        return str(getattr(patch, 'patch_id', None) or f"patch_{getattr(patch, 'coordinates', None)}")

    def _get_reputation_priority_signal(self, agent) -> float:
        """Return village-reputation priority adjustment for market ranking."""
        cfg = getattr(self, 'model_config', {}) or {}
        if not (cfg.get('mechanism_switches', {}) or {}).get('use_reputation_village', False):
            return 0.0
        model = getattr(self, 'model', None)
        if model is None:
            return 0.0
        agent_id = self._get_agent_key(agent)
        boosts = []
        for village in getattr(model, 'village_collectives', []) or []:
            reputation_system = getattr(village, 'reputation_system', None)
            if reputation_system is None:
                continue
            try:
                boosts.append(float(reputation_system.matching_priority_boost(agent_id)))
            except Exception:
                continue
        return float(sum(boosts) / len(boosts)) if boosts else 0.0

    def _sorted_patch_sequence(self, patch_map):
        """Iterate patch maps in stable patch-id order."""
        return sorted(patch_map.keys(), key=self._get_patch_key)

    def _sort_supply_offers(self, offers):
        return sorted(
            offers,
            key=lambda offer: (
                self._get_agent_key(offer.get('supplier')),
                self._get_patch_key(offer.get('patch')),
                float(offer.get('reservation_price', 0.0) or 0.0),
            ),
        )

    def _sort_demand_offers(self, offers):
        return sorted(
            offers,
            key=lambda offer: (
                -self._get_policy_weighted_bid_score(offer),
                -self._get_demander_grain_effective_capacity_signal(offer.get('demander')),
                -self._get_demander_productive_capacity_signal(offer.get('demander')),
                -self._get_reputation_priority_signal(offer.get('demander')),
                self._get_agent_key(offer.get('demander')),
                self._get_patch_key(offer.get('patch')),
            ),
        )
        
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
        
        @enhanced 2025-12-28: 添加最优规模约束
        企业/合作社超过最优规模时自动抑制流入需求
        """
        # @new 2025-12-28: 企业最优规模约束
        # 防止企业无限扩张，符合规模不经济理论
        from agents.enterprise_agent import AgriculturalEnterpriseAgent
        if isinstance(demander, AgriculturalEnterpriseAgent):
            if hasattr(demander, 'calculate_optimal_operated_scale'):
                scale_info = demander.calculate_optimal_operated_scale()
                scale_gap = scale_info.get('scale_gap', 100)
                decision = scale_info.get('decision', 'expand')
                
                # 无扩张需求时拒绝添加需求
                if decision == 'maintain' or decision == 'contract':
                    return  # 不添加需求，企业已达或超过最优规模
                
                # 接近最优规模时降低出价
                if scale_gap < 50:
                    bid_price *= 0.85  # 降低15%出价意愿
        
        self.demand_offers.append({
            'patch': patch,
            'demander': demander,
            'bid_price': bid_price,
            'is_elite': hasattr(demander, 'is_village_elite') and demander.is_village_elite
        })

    def _get_demander_grain_alignment_signal(self, demander) -> float:
        if demander is None:
            return 0.5

        cache_key = self._get_agent_key(demander)
        cached_value = self._demander_grain_alignment_cache.get(cache_key)
        if cached_value is not None:
            return float(cached_value)

        portfolio = getattr(demander, 'crop_portfolio', None)
        if isinstance(portfolio, dict) and portfolio:
            grain_share = 0.0
            for crop_key, weight in portfolio.items():
                crop_name = str(getattr(crop_key, 'value', crop_key)).lower()
                if 'grain' in crop_name:
                    try:
                        grain_share += float(weight or 0.0)
                    except (TypeError, ValueError):
                        continue
            signal = max(0.20, min(1.0, 0.35 + 0.65 * grain_share))
            self._demander_grain_alignment_cache[cache_key] = signal
            return signal

        planned_crop = getattr(demander, 'planned_crop', None)
        planned_name = str(getattr(planned_crop, 'value', planned_crop)).lower()
        if 'grain' in planned_name:
            self._demander_grain_alignment_cache[cache_key] = 0.95
            return 0.95

        strategy_name = str(
            getattr(getattr(demander, 'livelihood_strategy', None), 'value', getattr(demander, 'livelihood_strategy', ''))
        ).lower()
        if 'pure' in strategy_name:
            signal = 0.85
        elif 'pluri' in strategy_name:
            signal = 0.65
        elif 'non' in strategy_name:
            signal = 0.35
        else:
            signal = 0.5
        self._demander_grain_alignment_cache[cache_key] = signal
        return signal

    def _get_grain_policy_support_signal(self, model) -> float:
        if model is None or not hasattr(model, 'environment') or model.environment is None:
            return 0.0

        cache_key = (id(model), getattr(model, 'step_count', None))
        cached_value = self._grain_policy_support_cache.get(cache_key)
        if cached_value is not None:
            return float(cached_value)

        active_policy = getattr(model.environment, 'active_policy', {}) or {}
        if bool(active_policy.get('disable_grain_intervention', False)):
            self._grain_policy_support_cache[cache_key] = 0.0
            return 0.0

        try:
            floor_value = float(active_policy.get('grain_price_floor', 0.0) or 0.0)
        except (TypeError, ValueError):
            floor_value = 0.0
        try:
            subsidy_rate = float(
                active_policy.get(
                    'grain_subsidies_rate',
                    active_policy.get('subsidy_rate', 0.0),
                ) or 0.0
            )
        except (TypeError, ValueError):
            subsidy_rate = 0.0
        try:
            base_subsidy = float(active_policy.get('base_subsidy_per_mu', 0.0) or 0.0)
        except (TypeError, ValueError):
            base_subsidy = 0.0

        floor_signal = max(0.0, min(1.0, (floor_value - 1.0) / 0.4))
        subsidy_signal = max(
            0.0,
            min(1.0, max(subsidy_rate, base_subsidy / BASELINE_SUBSIDY_NORMALIZATION_ANCHOR)),
        )
        signal = max(0.0, min(1.0, 0.55 * floor_signal + 0.45 * subsidy_signal))
        self._grain_policy_support_cache[cache_key] = signal
        return signal

    def _get_demander_productive_capacity_signal(self, demander) -> float:
        if demander is None:
            return 0.5

        model = getattr(demander, 'model', None)
        cache_key = (
            self._get_agent_key(demander),
            getattr(model, 'step_count', None),
            getattr(model, 'land_tenure_version', None),
        )
        cached_value = self._demander_productive_capacity_cache.get(cache_key)
        if cached_value is not None:
            return float(cached_value)

        operated_area = max(0.0, get_farmer_operated_land_area(demander))
        area_signal = min(1.0, operated_area / 80.0)

        service_signal = 0.0
        service_contracts = getattr(demander, 'service_contracts', None)
        if isinstance(service_contracts, list) and service_contracts:
            service_signal = min(1.0, 0.25 + 0.18 * len(service_contracts))

        labor_reduction = max(0.0, min(1.0, float(getattr(demander, 'labor_reduction_ratio', 0.0) or 0.0)))

        mech_signal = 0.0
        if hasattr(demander, 'grain_mechanization_bonus'):
            mech_signal = max(0.0, min(1.0, float(getattr(demander, 'grain_mechanization_bonus', 0.0) or 0.0) / 0.25))
        elif hasattr(demander, '_get_regional_land_transfer_params'):
            try:
                regional_params = demander._get_regional_land_transfer_params()
                mech_signal = max(0.0, min(1.0, float(regional_params.get('mechanization_potential', 0.7) or 0.7)))
            except Exception:
                mech_signal = 0.0

        efficiency_signal = max(
            0.0,
            min(1.0, (float(getattr(demander, 'efficiency_factor', 1.0) or 1.0) - 0.8) / 0.5),
        )

        professional_signal = 1.0 if is_professional_large_operator(demander) else 0.0
        if professional_signal <= 0.0:
            strategy_name = str(
                getattr(getattr(demander, 'livelihood_strategy', None), 'value', getattr(demander, 'livelihood_strategy', ''))
            ).lower()
            if 'pure' in strategy_name:
                professional_signal = 0.55
            elif 'pluri' in strategy_name:
                professional_signal = 0.35
            elif 'non' in strategy_name:
                professional_signal = 0.10
            elif hasattr(demander, 'enterprise_type'):
                professional_signal = 0.70
            else:
                professional_signal = 0.35

        income_signal = 0.0
        net_income = max(0.0, get_farmer_net_income(demander, default=0.0))
        if net_income > 0:
            income_signal = min(1.0, net_income / 120000.0)

        composite = (
            0.12 * area_signal
            + 0.22 * service_signal
            + 0.18 * labor_reduction
            + 0.18 * mech_signal
            + 0.15 * efficiency_signal
            + 0.10 * professional_signal
            + 0.05 * income_signal
        )
        signal = max(0.10, min(1.0, composite))
        self._demander_productive_capacity_cache[cache_key] = signal
        return signal

    def _get_demander_grain_effective_capacity_signal(self, demander) -> float:
        """Policy should reward grain-oriented operators only when they can actually farm productively."""
        grain_alignment = self._get_demander_grain_alignment_signal(demander)
        productive_capacity = self._get_demander_productive_capacity_signal(demander)
        signal = grain_alignment * (0.35 + 0.65 * productive_capacity)
        if productive_capacity < 0.45:
            signal *= (0.70 + 0.40 * productive_capacity)
        return max(0.10, min(1.0, signal))

    def _get_policy_quality_multiplier(
        self,
        demander,
        *,
        policy_signal: float | None = None,
        grain_effective_capacity: float | None = None,
        productive_capacity: float | None = None,
    ) -> float:
        if demander is None:
            return 1.0

        model = getattr(demander, 'model', None)
        if policy_signal is None:
            policy_signal = self._get_grain_policy_support_signal(model)
        if grain_effective_capacity is None:
            grain_effective_capacity = self._get_demander_grain_effective_capacity_signal(demander)
        if productive_capacity is None:
            productive_capacity = self._get_demander_productive_capacity_signal(demander)

        strategy_name = str(
            getattr(getattr(demander, 'livelihood_strategy', None), 'value', getattr(demander, 'livelihood_strategy', ''))
        ).lower()
        professional_operator = is_professional_large_operator(demander)

        quality_multiplier = 1.0 + policy_signal * (
            0.14 * (grain_effective_capacity - 0.5)
            + 0.20 * (productive_capacity - 0.5)
        )

        if policy_signal > 0.0:
            if professional_operator and productive_capacity >= 0.55:
                quality_multiplier += 0.06 * policy_signal
            elif 'pure' in strategy_name and productive_capacity >= 0.62:
                quality_multiplier += 0.02 * policy_signal
            elif 'non' in strategy_name and grain_effective_capacity < 0.55:
                quality_multiplier -= 0.10 * policy_signal

            if grain_effective_capacity < 0.55:
                quality_multiplier -= 0.18 * policy_signal * (0.55 - grain_effective_capacity)
            if productive_capacity < 0.50:
                quality_multiplier -= 0.18 * policy_signal * (0.50 - productive_capacity)

        return max(0.72, quality_multiplier)

    def _get_policy_weighted_bid_score(self, offer) -> float:
        bid_price = float(offer.get('bid_price', 0.0) or 0.0)
        demander = offer.get('demander')
        model = getattr(demander, 'model', None)
        cache_key = (
            self._get_agent_key(demander),
            self._get_patch_key(offer.get('patch')),
            bid_price,
            getattr(model, 'step_count', None),
            getattr(model, 'land_tenure_version', None),
        )
        cached_value = self._policy_weighted_bid_score_cache.get(cache_key)
        if cached_value is not None:
            return float(cached_value)

        policy_signal = self._get_grain_policy_support_signal(model)
        grain_effective_capacity = self._get_demander_grain_effective_capacity_signal(demander)
        productive_capacity = self._get_demander_productive_capacity_signal(demander)
        quality_multiplier = self._get_policy_quality_multiplier(
            demander,
            policy_signal=policy_signal,
            grain_effective_capacity=grain_effective_capacity,
            productive_capacity=productive_capacity,
        )
        score = bid_price * quality_multiplier
        self._policy_weighted_bid_score_cache[cache_key] = score
        return score

    def _check_transfer_rate_limit(self, model, patch_area: float, transfer_channel: str | None = None, demander=None) -> bool:
        """
        @new 2025-12-27: 检查是否超过土地流转率上限
        
        机制：
        1. 年度上限：每年新增流转不超过总面积的5%
        2. 累计上限：累计流转率不超过70%
        
        返回：True表示允许流转，False表示达到上限
        """
        if model is None or not hasattr(model, 'transfer_rate_control'):
            return True  # 无追踪时默认允许
        
        control = model.transfer_rate_control
        total_area = control.get('total_land_area', 0)
        if total_area <= 0:
            total_area = control.get('observed_total_land_area', 0)
        if total_area <= 0:
            total_area = control.get('baseline_total_land_area', 0)
        
        # 首次调用时设置总面积
        if total_area <= 0 and hasattr(model, 'land_patches'):
            total_area = sum(getattr(p, 'area', 0) for p in model.land_patches)
            control['total_land_area'] = total_area
        
        if total_area <= 0:
            return True
        
        # 检查年度上限
        annual_new = control['annual_new_transferred_area']
        max_annual_rate = float(control.get('max_annual_rate', 0.05) or 0.05)
        # @new 2026-06-24: 竞拍市场保留优先额度
        # priced_auction_reserve_share 从年度上限中为竞拍市场预留份额
        # 非竞拍渠道（village/direct/enterprise）使用标准上限
        # 竞拍渠道（transfer_channel='direct_farmer' from clearing house）享有预留额度
        if transfer_channel and 'direct_farmer' in str(transfer_channel):
            reserve_share = float(control.get('priced_auction_reserve_share', 0.15))
            max_annual_rate = max_annual_rate * (1.0 + reserve_share)
        max_annual = max_annual_rate * total_area
        if annual_new + patch_area > max_annual:
            return False
        
        # 检查当前流转率上限。这里必须基于“当前已流转面积”，
        # 否则历史累计成交会和当前存量流转率脱节，导致上限失效。
        current_transferred_area = float(control.get('observed_current_transferred_area', 0.0) or 0.0)
        if current_transferred_area <= 0.0 and hasattr(model, 'land_patches'):
            current_transferred_area = sum(
                getattr(p, 'area', 0.0)
                for p in model.land_patches
                if getattr(p, 'leased_to', None) is not None
            )
        control['observed_total_land_area'] = total_area
        control['observed_current_transferred_area'] = current_transferred_area

        projected_transferred_area = current_transferred_area + max(0.0, float(patch_area or 0.0))
        effective_ceiling_rate = self._estimate_endogenous_transfer_ceiling(model, total_area)
        effective_upper_bound_rate = float(
            control.get('effective_cumulative_ceiling_upper_bound', effective_ceiling_rate) or effective_ceiling_rate
        )
        hard_ceiling_rate = float(control.get('cumulative_ceiling', 0.70) or 0.70)
        effective_ceiling = effective_ceiling_rate * total_area
        effective_upper_bound = effective_upper_bound_rate * total_area
        hard_ceiling = hard_ceiling_rate * total_area

        if projected_transferred_area <= effective_ceiling:
            return True
        if projected_transferred_area > hard_ceiling:
            return False
        if projected_transferred_area > effective_upper_bound:
            return False

        # @new 2026-06-24: 竞拍市场在有效上限和硬上限之间享有与村集体同等的软上限通过权
        # 这使 priced auction 能在累计流转率接近上限时仍产生交易
        privileged_channels = {'village_collective', 'cross_village_collective', 'direct_farmer'}
        if transfer_channel not in privileged_channels:
            control['soft_ceiling_channel'] = transfer_channel or 'direct_farmer'
            control['soft_ceiling_acceptance_prob'] = 0.0
            return False

        overflow_rate = max(0.0, projected_transferred_area / max(total_area, 1.0) - effective_ceiling_rate)
        overflow_band = max(1e-6, effective_upper_bound_rate - effective_ceiling_rate)
        overflow_ratio = min(1.0, overflow_rate / overflow_band)
        capacity_signal = max(0.0, min(1.0, float(control.get('transfer_capacity_signal', 0.0) or 0.0)))
        unlock_signal = max(0.0, min(1.0, float(control.get('capacity_unlock_signal', capacity_signal) or capacity_signal)))
        governance_signal = max(
            0.0,
            min(1.0, float(control.get('village_governance_signal', control.get('village_capacity_signal', 0.0)) or 0.0)),
        )
        service_slack_signal = max(0.0, min(1.0, float(control.get('service_slack_signal', 1.0) or 1.0)))
        grain_alignment = self._get_demander_grain_effective_capacity_signal(demander)
        # @new 2026-06-24: direct_farmer (priced auction) gets market-based acceptance
        # independent of village governance signals. Market tightness (bid/reservation ratio,
        # tradable share) determines acceptance, not institutional politics.
        if transfer_channel == 'cross_village_collective':
            channel_acceptance = float(control.get('cross_village_overflow_acceptance', 0.88) or 0.0)
        elif transfer_channel == 'direct_farmer':
            channel_acceptance = float(control.get('priced_auction_overflow_acceptance', 0.50) or 0.0)
        else:
            channel_acceptance = float(control.get('collective_overflow_base_acceptance', 0.22) or 0.0)

        min_acceptance = float(control.get('soft_throttle_min_acceptance', 0.04) or 0.04)

        if transfer_channel == 'direct_farmer':
            # Priced auction: market-based acceptance independent of overflow_ratio.
            # The annual cap check (with priced_auction_reserve_share) already
            # controls volume. Soft-ceiling zone acceptance should reflect market
            # conditions (bid/reservation ratio, tradable share), not ceiling fullness.
            market_tightness = max(0.0, min(1.0, float(control.get('preclear_max_bid_reservation_ratio', 1.0) or 1.0) - 0.85) / 0.5)
            tradable_n = max(0, int(control.get('preclear_raw_tradable_085_patch_count', 0) or 0))
            overlap_n = max(1, int(control.get('preclear_overlap_patch_count', 1) or 1))
            tradable_share = max(0.05, min(1.0, float(tradable_n) / float(overlap_n)))
            market_support = max(0.20, min(1.0, 0.55 * market_tightness + 0.45 * tradable_share))
            acceptance_prob = min_acceptance + (1.0 - min_acceptance) * market_support
        else:
            institutional_support_val = max(
                0.0,
                min(1.0, 0.45 * governance_signal + 0.35 * unlock_signal + 0.20 * service_slack_signal),
            )
            acceptance_prob = min_acceptance + (1.0 - min_acceptance) * institutional_support_val * ((1.0 - overflow_ratio) ** 1.6)
        acceptance_prob *= max(0.0, min(1.0, channel_acceptance))
        acceptance_prob *= max(0.25, min(1.0, grain_alignment))
        control['soft_ceiling_channel'] = transfer_channel
        control['soft_ceiling_acceptance_prob'] = acceptance_prob
        return rng_random(model) < acceptance_prob

    def _get_provider_capacity_snapshot(self, model):
        """Summarize provider capacity for ceiling estimation fallbacks."""
        providers = list(getattr(model, 'service_providers', []) or [])
        if not providers:
            return {
                'provider_count': 0,
                'total_capacity_area': 0.0,
                'utilized_capacity_area': 0.0,
                'slack_signal': 1.0,
                'cache_signature': (),
            }

        total_capacity = 0.0
        utilized_capacity = 0.0
        cache_signature = []

        for provider in providers:
            capacity = 0.0
            for attr_name in ('current_capacity', 'annual_service_capacity', 'max_service_area', 'initial_capacity'):
                try:
                    capacity = float(getattr(provider, attr_name, 0.0) or 0.0)
                except (TypeError, ValueError):
                    capacity = 0.0
                if capacity > 0.0:
                    break

            used_area = None
            for attr_name in ('service_area_used', 'current_service_area'):
                if hasattr(provider, attr_name):
                    try:
                        used_area = float(getattr(provider, attr_name, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        used_area = 0.0
                    break

            if used_area is None:
                active_contracts = getattr(provider, 'active_contracts', None)
                if isinstance(active_contracts, list) and active_contracts:
                    used_area = sum(
                        max(
                            0.0,
                            float(
                                getattr(contract, 'area', getattr(contract, 'land_area', 0.0)) or 0.0
                            ),
                        )
                        for contract in active_contracts
                    )
                else:
                    used_area = 0.0

            capacity = max(0.0, capacity)
            used_area = max(0.0, float(used_area or 0.0))
            if capacity > 0.0:
                used_area = min(capacity, used_area)

            total_capacity += capacity
            utilized_capacity += used_area
            cache_signature.append((
                str(getattr(provider, 'unique_id', getattr(provider, 'provider_id', 'provider'))),
                round(capacity, 6),
                round(used_area, 6),
            ))

        slack_signal = 1.0
        if total_capacity > 0.0:
            slack_signal = max(0.0, min(1.0, 1.0 - utilized_capacity / total_capacity))

        return {
            'provider_count': len(providers),
            'total_capacity_area': total_capacity,
            'utilized_capacity_area': utilized_capacity,
            'slack_signal': slack_signal,
            'cache_signature': tuple(cache_signature),
        }

    def _estimate_endogenous_transfer_ceiling(self, model, total_area: float) -> float:
        """Estimate transfer carrying capacity before the hard backstop is hit."""
        if model is None or total_area <= 0:
            return 0.0

        control = getattr(model, 'transfer_rate_control', {}) or {}
        provider_capacity_snapshot = self._get_provider_capacity_snapshot(model)
        service_stats_area = 0.0
        try:
            service_stats_area = float(
                getattr(model, 'service_market_stats', {}).get('total_service_area', 0.0) or 0.0
            )
        except (TypeError, ValueError):
            service_stats_area = 0.0
        cache_key = (
            int(getattr(model, 'step_count', -1) or -1),
            round(float(total_area), 6),
            round(service_stats_area, 6),
            provider_capacity_snapshot.get('cache_signature', ()),
        )
        if control.get('_capacity_cache_key') == cache_key:
            cached_ceiling = float(control.get('effective_cumulative_ceiling', 0.0) or 0.0)
            if cached_ceiling > 0:
                return cached_ceiling
        hard_ceiling = float(control.get('cumulative_ceiling', 0.70) or 0.70)
        soft_floor = float(control.get('endogenous_soft_floor', 0.46) or 0.46)
        service_bonus = float(control.get('service_capacity_bonus', 0.04) or 0.04)
        profitability_bonus = float(control.get('profitability_capacity_bonus', 0.09) or 0.09)
        aging_bonus = float(control.get('aging_capacity_bonus', 0.02) or 0.02)
        professional_bonus = float(
            control.get(
                'professionalization_capacity_bonus',
                control.get('professional_capacity_bonus', 0.08),
            ) or 0.08
        )
        village_bonus = float(
            control.get(
                'village_facilitation_bonus',
                control.get('village_coordination_bonus', 0.04),
            ) or 0.04
        )
        policy_bonus = float(
            control.get('policy_support_capacity_bonus', 0.0) or 0.0
        )
        curvature = max(0.25, float(control.get('capacity_signal_curvature', 1.20) or 1.20))
        unlock_share = max(0.0, min(0.95, float(control.get('ceiling_unlock_share', 0.22) or 0.22)))
        unlock_curvature = max(0.25, float(control.get('unlock_signal_curvature', 1.0) or 1.0))
        policy_unlock_bonus = float(control.get('policy_unlock_bonus', 0.10) or 0.10)
        service_unlock_bonus = float(control.get('service_unlock_bonus', 0.04) or 0.04)
        village_unlock_bonus = float(control.get('village_unlock_bonus', 0.04) or 0.04)
        professional_unlock_bonus = float(control.get('professional_unlock_bonus', 0.03) or 0.03)
        profit_reference_income = max(1.0, float(control.get('profit_reference_income', 80000) or 80000))
        aging_reference_share = max(0.05, float(control.get('aging_reference_share', 0.55) or 0.55))
        operator_share_reference = max(0.01, float(control.get('operator_share_reference', 0.14) or 0.14))
        service_reference_share = max(0.10, float(control.get('service_reference_share', 0.75) or 0.75))
        policy_reference_intensity = max(
            1e-6,
            float(control.get('policy_reference_intensity', 0.20) or 0.20),
        )

        farmers = list(getattr(model, 'farmers', []) or [])
        if not farmers and hasattr(model, 'schedule'):
            farmers = [agent for agent in model.schedule.agents if hasattr(agent, 'land_patches')]

        retention_cfg = (
            getattr(model, 'config', {})
            .get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        min_operated_area = float(retention_cfg.get('min_operated_area', 30) or 30)
        min_ag_ratio = float(retention_cfg.get('min_agricultural_income_ratio', 0.78) or 0.78)
        max_off_farm_income = float(retention_cfg.get('max_off_farm_income', 12000) or 12000)

        professional_large_operators = [
            farmer
            for farmer in farmers
            if is_professional_large_operator(
                farmer,
                min_operated_area=min_operated_area,
                min_agricultural_income_ratio=min_ag_ratio,
                max_off_farm_income=max_off_farm_income,
            )
        ]
        positive_large_income = [
            max(0.0, float(get_farmer_net_income(farmer, default=0.0) or 0.0))
            for farmer in professional_large_operators
        ]
        mean_large_income = float(np.mean(positive_large_income)) if positive_large_income else 0.0
        profitability_signal = 1.0 - np.exp(-mean_large_income / profit_reference_income)
        professional_operated_area = sum(
            max(0.0, float(get_farmer_operated_land_area(farmer) or 0.0))
            for farmer in professional_large_operators
        )
        professional_signal = min(
            1.0,
            professional_operated_area / max(total_area * operator_share_reference, 1.0),
        )

        service_area = 0.0
        if hasattr(model, 'get_service_market_statistics'):
            try:
                service_area = float(model.get_service_market_statistics().get('total_service_area', 0.0) or 0.0)
            except Exception:
                service_area = 0.0
        if service_area <= 0:
            service_area = float(
                getattr(model, 'service_market_stats', {}).get('total_service_area', 0.0) or 0.0
            )
        service_reference_area = max(total_area * service_reference_share, 1.0)
        realized_service_signal = min(1.0, service_area / service_reference_area)
        provider_capacity_area = float(provider_capacity_snapshot.get('total_capacity_area', 0.0) or 0.0)
        provider_used_area = float(provider_capacity_snapshot.get('utilized_capacity_area', 0.0) or 0.0)
        provider_slack_signal = max(0.0, min(1.0, float(provider_capacity_snapshot.get('slack_signal', 1.0) or 1.0)))
        provider_capacity_signal = min(1.0, provider_capacity_area / service_reference_area)
        provider_commitment_signal = min(1.0, provider_used_area / service_reference_area)
        provider_capacity_fallback_signal = provider_capacity_signal * (0.35 + 0.65 * provider_slack_signal)
        if service_area > 0.0:
            service_signal = max(realized_service_signal, provider_commitment_signal)
        else:
            service_signal = max(provider_commitment_signal, provider_capacity_fallback_signal)

        village_transfer_area = 0.0
        village_network_density = 0.0
        if hasattr(model, 'get_village_facilitated_transfer_area'):
            try:
                village_transfer_area = float(model.get_village_facilitated_transfer_area() or 0.0)
            except Exception:
                village_transfer_area = 0.0
        if hasattr(model, 'get_village_network_density'):
            try:
                village_network_density = float(model.get_village_network_density() or 0.0)
            except Exception:
                village_network_density = 0.0
        village_transfer_signal = min(1.0, village_transfer_area / max(total_area, 1.0))
        village_governance_signal = 0.0
        village_congestion_signal = 0.0
        if hasattr(model, 'get_village_governance_capacity'):
            try:
                village_governance_signal = float(model.get_village_governance_capacity() or 0.0)
            except Exception:
                village_governance_signal = 0.0
        if hasattr(model, 'get_village_transfer_congestion'):
            try:
                village_congestion_signal = float(model.get_village_transfer_congestion() or 0.0)
            except Exception:
                village_congestion_signal = 0.0
        village_governance_signal = max(0.0, min(1.0, village_governance_signal))
        village_congestion_signal = max(0.0, min(1.0, village_congestion_signal))
        village_signal = min(
            1.0,
            0.45 * village_transfer_signal + 0.15 * village_network_density + 0.40 * village_governance_signal,
        )
        village_unlock_signal = min(1.0, 0.65 * village_governance_signal + 0.35 * village_signal)
        service_slack_signal = max(0.0, min(1.0, 1.0 - village_congestion_signal))
        if provider_capacity_area > 0.0:
            service_slack_signal = max(
                0.0,
                min(1.0, 0.5 * service_slack_signal + 0.5 * provider_slack_signal),
            )

        older_share = 0.0
        if farmers:
            older_share = sum(1 for farmer in farmers if float(getattr(farmer, 'age', 0.0) or 0.0) >= 55.0) / len(farmers)
        aging_signal = min(1.0, older_share / aging_reference_share)

        policy_signal = 0.0
        env = getattr(model, 'environment', None)
        active_policy = getattr(env, 'active_policy', {}) if env is not None else {}
        if isinstance(active_policy, dict) and active_policy:
            subsidy_rate = max(0.0, float(active_policy.get('grain_subsidies_rate', 0.0) or 0.0))
            subsidy_signal = min(1.0, subsidy_rate / policy_reference_intensity)

            price_support_signal = 0.0
            market_cfg = (
                getattr(model, 'config', {})
                .get('environment', {})
                .get('market', {})
            )
            grain_limits = market_cfg.get('price_limits', {}).get('grain', {})
            try:
                min_price = float(grain_limits.get('min', 0.0) or 0.0)
                max_price = float(grain_limits.get('max', min_price) or min_price)
                price_floor = float(active_policy.get('grain_price_floor', min_price) or min_price)
            except (TypeError, ValueError):
                min_price = 0.0
                max_price = 0.0
                price_floor = 0.0
            if max_price > min_price:
                price_support_signal = max(0.0, min(1.0, (price_floor - min_price) / (max_price - min_price)))

            policy_signal = min(1.0, 0.6 * subsidy_signal + 0.4 * price_support_signal)

        signal_weight_total = (
            service_bonus
            + profitability_bonus
            + aging_bonus
            + professional_bonus
            + village_bonus
            + policy_bonus
        )
        if signal_weight_total <= 0:
            composite_capacity_signal = 0.0
        else:
            composite_capacity_signal = (
                service_bonus * service_signal
                + profitability_bonus * profitability_signal
                + aging_bonus * aging_signal
                + professional_bonus * professional_signal
                + village_bonus * village_signal
                + policy_bonus * policy_signal
            ) / signal_weight_total

        unlock_weight_total = (
            policy_unlock_bonus
            + service_unlock_bonus
            + village_unlock_bonus
            + professional_unlock_bonus
        )
        if unlock_weight_total <= 0:
            capacity_unlock_signal = policy_signal
        else:
            capacity_unlock_signal = (
                policy_unlock_bonus * policy_signal
                + service_unlock_bonus * service_signal
                + village_unlock_bonus * village_unlock_signal
                + professional_unlock_bonus * professional_signal
            ) / unlock_weight_total
        capacity_unlock_signal *= (0.75 + 0.25 * service_slack_signal)
        capacity_unlock_signal = max(0.0, min(1.0, capacity_unlock_signal))

        ceiling_span = max(0.0, hard_ceiling - soft_floor)
        institutional_top_band = ceiling_span * unlock_share
        carrying_band = max(0.0, ceiling_span - institutional_top_band)

        effective_ceiling = soft_floor + carrying_band * (composite_capacity_signal ** curvature)
        effective_ceiling = max(soft_floor, min(soft_floor + carrying_band, effective_ceiling))

        effective_upper_bound = effective_ceiling + institutional_top_band * (capacity_unlock_signal ** unlock_curvature)
        effective_upper_bound = max(effective_ceiling, min(hard_ceiling, effective_upper_bound))

        control['service_capacity_signal'] = service_signal
        control['provider_capacity_signal'] = provider_capacity_signal
        control['provider_committed_service_signal'] = provider_commitment_signal
        control['provider_capacity_area'] = provider_capacity_area
        control['provider_service_area_used'] = provider_used_area
        control['profitability_capacity_signal'] = profitability_signal
        control['aging_capacity_signal'] = aging_signal
        control['professional_capacity_signal'] = professional_signal
        control['village_capacity_signal'] = village_signal
        control['village_governance_signal'] = village_governance_signal
        control['village_congestion_signal'] = village_congestion_signal
        control['policy_capacity_signal'] = policy_signal
        control['service_slack_signal'] = service_slack_signal
        control['transfer_capacity_signal'] = composite_capacity_signal
        control['capacity_unlock_signal'] = capacity_unlock_signal
        control['institutional_top_band'] = institutional_top_band
        control['effective_cumulative_ceiling_upper_bound'] = effective_upper_bound
        control['effective_cumulative_ceiling'] = effective_ceiling
        control['_capacity_cache_key'] = cache_key
        return effective_ceiling
    
    def clear_market(self, model_config: dict = None):
        """
        @theory 双轨制市场出清机制
        @critical 2025-12 UPGRADE: 实现中国土地流转的"半市场化"本质
        
        @fix 2026-01-14: 添加消融实验机制开关支持
        - rent_market_enabled=False 时使用固定租金
        - fixed_rent_value 指定固定租金金额
        
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
        self._reset_signal_caches()
        self.model_config = model_config or {}
        self.last_supply_count = len(self.supply_offers)
        self.last_demand_count = len(self.demand_offers)

        mechanism_switches = self.model_config.get('mechanism_switches', {})
        land_market_config = self.model_config.get('land_market', {})
        cell_size = land_market_config.get('synthetic_locality_cell_size')
        if cell_size is not None and (type(cell_size) is not int or cell_size <= 0):
            raise ValueError('land_market.synthetic_locality_cell_size must be a positive integer')

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
            if cell_size is not None:
                # Synthetic grid blocks only; neither supplier location nor official region enters eligibility.
                coordinates = getattr(patch, 'coordinates', None)
                position = getattr(offer['demander'], 'pos', None)
                if coordinates is None or position is None or (
                    coordinates[0] // cell_size, coordinates[1] // cell_size
                ) != (position[0] // cell_size, position[1] // cell_size):
                    continue
            if patch not in patch_demand:
                patch_demand[patch] = []
            patch_demand[patch].append(offer)

        for patch, offers in list(patch_supply.items()):
            patch_supply[patch] = self._sort_supply_offers(offers)
        for patch, offers in list(patch_demand.items()):
            patch_demand[patch] = self._sort_demand_offers(offers)

        if not mechanism_switches.get('rent_market_enabled', True) or land_market_config.get('fixed_rent', False):
            self._fixed_rent_matching(land_market_config.get('fixed_rent_value', 300), patch_supply, patch_demand)
            return

        # @new 2026-06-24: Write market signals to transfer_rate_control so
        # _check_transfer_rate_limit can use them for priced-auction acceptance
        self._write_market_signals_to_control(patch_supply, patch_demand)

        # 第一轨：熟人网络内部消化
        matched_patches = self._acquaintance_track_matching(patch_supply, patch_demand)
        
        # 第二轨：公开市场竞价（剩余地块）
        self._open_market_track_matching(patch_supply, patch_demand, matched_patches)

        self._capture_flow_audit()
        
        # 更新市场统计
        self._update_market_statistics()
        
        # 清空报价列表
        self.supply_offers = []
        self.demand_offers = []
    
    def _write_market_signals_to_control(self, patch_supply, patch_demand):
        """Write preclear market signals to model.transfer_rate_control.

        These signals are read by _check_transfer_rate_limit to compute
        market-based acceptance probability for priced auction transactions.
        """
        model = getattr(self, 'model', None)
        if model is None or not hasattr(model, 'transfer_rate_control'):
            return
        control = model.transfer_rate_control

        # Overlapping patches (have both supply and demand)
        overlap = set(patch_supply.keys()) & set(patch_demand.keys())
        control['preclear_overlap_patch_count'] = len(overlap)

        # Tradable patches: bid >= 85% of reservation
        tradable = 0
        max_bid_ratio = 1.0
        for patch in overlap:
            supplies = patch_supply[patch]
            demands = patch_demand[patch]
            if supplies and demands:
                reservation = supplies[0].get('reservation_price', 0)
                best_bid = demands[0].get('bid_price', 0)
                if reservation > 0:
                    ratio = best_bid / reservation
                    if ratio > max_bid_ratio:
                        max_bid_ratio = ratio
                    if ratio >= 0.85:
                        tradable += 1
        control['preclear_raw_tradable_085_patch_count'] = tradable
        control['preclear_max_bid_reservation_ratio'] = max_bid_ratio

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
        
        for patch in self._sorted_patch_sequence(patch_supply):
            supply_offers = patch_supply[patch]
            if patch in matched_patches:
                continue
            if patch not in patch_demand:
                continue
            
            demand_offers = self._sort_demand_offers(patch_demand[patch])
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
                best_acq = self._sort_demand_offers(acquaintance_demands)[0]
                bid = best_acq['bid_price']
                
                # @new 2025-12-28: 差序格局定价
                # @theory: 费孝通《乡土中国》"差序格局"
                # @literature: cssn.cn - "差序信任格局"影响土地流转
                # 
                # 定价层级（关系越近折扣越大）：
                # 1. 亲属（宗族）：极低价/象征性 → 折扣30-50%
                # 2. 同组村民（熟人）：低于市场价 → 折扣60-80%
                # 3. 本村外组/普通熟人：接近市场价 → 折扣80-95%
                # 4. 外来企业（公开市场）：市场价或溢价 → 100-120%
                
                demander = best_acq['demander']
                kinship_level = self._get_kinship_level(supplier, demander)
                
                # 根据关系层级确定折扣率
                if kinship_level == 'kinship':  # 亲属/宗族
                    discount_rate = 0.40  # 可低至40%
                elif kinship_level == 'same_group':  # 同组村民
                    discount_rate = 0.65  # 可低至65%
                elif kinship_level == 'same_village':  # 同村外组
                    discount_rate = 0.80  # 可低至80%
                else:  # 普通熟人
                    discount_rate = 0.90  # 接近市场价
                
                min_acceptable = reservation_price * discount_rate
                
                if bid >= min_acceptable:
                    # @new 2025-12-27: 检查流转率上限
                    model = getattr(supplier, 'model', None)
                    if not self._check_transfer_rate_limit(
                        model,
                        patch.area,
                        transfer_channel='direct_farmer',
                        demander=best_acq['demander'],
                    ):
                        continue  # 达到上限，跳过此交易
                    
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
        for patch in self._sorted_patch_sequence(patch_supply):
            supply_offers = patch_supply[patch]
            if patch in matched_patches:
                continue
            if patch not in patch_demand:
                continue
            
            demand_offers = self._sort_demand_offers(patch_demand[patch])
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
                policy_support_signal = float(ranked_demand.get('policy_support_signal', 0.0) or 0.0)
                grain_alignment = float(
                    ranked_demand.get(
                        'grain_effective_capacity',
                        ranked_demand.get('grain_alignment', 0.5),
                    ) or 0.5
                )
                productive_capacity = float(ranked_demand.get('productive_capacity', 0.5) or 0.5)
                
                # 计算连片溢价（需求方愿意为连片支付额外价格）
                contiguity_bonus = self._calculate_contiguity_bonus(
                    patch, demand_offer['demander'], reservation_price
                )
                effective_bid = bid + contiguity_bonus
                demander_actor = demand_offer['demander']
                policy_quality_multiplier = self._get_policy_quality_multiplier(
                    demander_actor,
                    policy_signal=policy_support_signal,
                    grain_effective_capacity=grain_alignment,
                    productive_capacity=productive_capacity,
                )
                effective_bid *= policy_quality_multiplier
                
                # 公开市场允许最高20%溢价
                max_allowed_bid = reservation_price * self.open_market_max_premium
                effective_bid = min(effective_bid, max_allowed_bid)
                
                # 出价需达到保留价的85%
                # 但连片流转可以放宽到80%（供给方也愿意连片出让）
                min_threshold = 0.80 if proximity_score > 0.5 else 0.85
                
                if effective_bid >= reservation_price * min_threshold:
                    # @new 2025-12-27: 检查流转率上限
                    model = getattr(supply_offer['supplier'], 'model', None)
                    if not self._check_transfer_rate_limit(
                        model,
                        patch.area,
                        transfer_channel='direct_farmer',
                        demander=demand_offer['demander'],
                    ):
                        continue  # 达到上限，跳过此交易
                    
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
    
    def _get_kinship_level(self, supplier, demander):
        """
        @new 2025-12-28: 判断差序格局下的关系层级
        
        @theory: 费孝通《乡土中国》"差序格局"
        
        关系层级（由近到远）：
        1. kinship: 亲属/宗族（同姓、血缘）
        2. same_group: 同组村民（日常互动频繁）
        3. same_village: 同村外组（认识但互动较少）
        4. acquaintance: 普通熟人（通过弱连接认识）
        
        Returns:
            str: 关系层级
        """
        # 企业默认为外人
        from agents.enterprise_agent import AgriculturalEnterpriseAgent
        if isinstance(demander, AgriculturalEnterpriseAgent):
            # 企业可能与村集体有关系（如村办企业）
            if hasattr(demander, 'is_village_enterprise') and demander.is_village_enterprise:
                return 'same_village'
            return 'outsider'
        
        # 检查亲属关系（同姓氏或显式亲属标记）
        supplier_surname = getattr(supplier, 'surname', None)
        demander_surname = getattr(demander, 'surname', None)
        if supplier_surname and demander_surname and supplier_surname == demander_surname:
            return 'kinship'
        
        # 检查显式亲属列表
        kinship_list = getattr(supplier, 'kinship_network', [])
        if demander.unique_id in kinship_list:
            return 'kinship'
        
        # 检查同组（基于village_group_id）
        supplier_group = getattr(supplier, 'village_group_id', None)
        demander_group = getattr(demander, 'village_group_id', None)
        if supplier_group and demander_group and supplier_group == demander_group:
            return 'same_group'
        
        # 检查同村（基于village_id或社会网络）
        supplier_village = getattr(supplier, 'village_id', 1)
        demander_village = getattr(demander, 'village_id', 1)
        if supplier_village == demander_village:
            return 'same_village'
        
        # 弱连接熟人
        weak_ties = getattr(supplier, 'weak_tie_contacts', [])
        if demander.unique_id in weak_ties:
            return 'acquaintance'
        
        # 普通社会网络熟人
        social_network = getattr(supplier, 'social_network', [])
        if demander in social_network or demander.unique_id in [getattr(n, 'unique_id', n) for n in social_network]:
            return 'acquaintance'
        
        return 'outsider'
    
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
        @optimized 2025-12-22: 使用空间索引加速邻近查询
        
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
        4. 使用空间索引加速邻近查询（从O(N)降至O(log N)）
        
        Returns:
            float: 空间邻近性得分 [0, 1]，1表示直接相邻
        """
        if not hasattr(patch, 'coordinates') or not patch.coordinates:
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
        
        # @optimized 2025-12-22: 使用空间索引查询邻近地块
        op_radius = self._calculate_operation_radius(demander)
        patch_coords = patch.coordinates
        
        # 如果有空间索引，使用索引查询
        nearby_patches = []
        if self._use_spatial_index and self._spatial_index is not None:
            nearby_patches = self._spatial_index.find_neighbors(patch, radius=op_radius)
            # 过滤出需求方的地块
            demander_patch_ids = {self._get_patch_key(p) for p in demander_patches}
            nearby_patches = [p for p in nearby_patches if self._get_patch_key(p) in demander_patch_ids]
        
        # 计算到最近已有地块的距离
        min_distance = float('inf')
        
        # 同时计算"连片程度" - 周围有多少已有地块
        adjacent_count = 0
        near_count = 0
        
        # 如果使用了空间索引，只检查nearby_patches；否则检查所有demander_patches
        patches_to_check = nearby_patches if nearby_patches else demander_patches
        
        for existing_patch in patches_to_check:
            if not hasattr(existing_patch, 'coordinates') or not existing_patch.coordinates:
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
        weighted_bid_scores = {
            id(demand_offer): self._get_policy_weighted_bid_score(demand_offer)
            for demand_offer in demand_offers
        }
        max_bid = max(weighted_bid_scores.values()) if weighted_bid_scores else 1.0

        for demand_offer in demand_offers:
            demander = demand_offer['demander']
            proximity_score = self._calculate_spatial_proximity_score(patch, demander)
            grain_alignment = self._get_demander_grain_alignment_signal(demander)
            grain_effective_capacity = self._get_demander_grain_effective_capacity_signal(demander)
            productive_capacity = self._get_demander_productive_capacity_signal(demander)
            policy_support_signal = self._get_grain_policy_support_signal(getattr(demander, 'model', None))
            weighted_bid_score = weighted_bid_scores.get(id(demand_offer), 0.0)
            normalized_bid = weighted_bid_score / max_bid if max_bid > 0 else 0.0

            if proximity_score >= 0.8:
                weights = {
                    'bid': 0.52,
                    'proximity': 0.26,
                    'grain': 0.08 + 0.06 * policy_support_signal,
                    'capacity': 0.14 + 0.08 * policy_support_signal,
                }
            else:
                weights = {
                    'bid': 0.70 - 0.08 * policy_support_signal,
                    'proximity': 0.20,
                    'grain': 0.04 + 0.08 * policy_support_signal,
                    'capacity': 0.08 + 0.16 * policy_support_signal,
                }
            weight_total = sum(weights.values()) or 1.0
            priority_score = (
                weights['bid'] * normalized_bid
                + weights['proximity'] * proximity_score
                + weights['grain'] * grain_effective_capacity
                + weights['capacity'] * productive_capacity
            ) / weight_total

            scored_demands.append({
                'demand_offer': demand_offer,
                'proximity_score': proximity_score,
                'priority_score': priority_score,
                'grain_alignment': grain_alignment,
                'grain_effective_capacity': grain_effective_capacity,
                'productive_capacity': productive_capacity,
                'policy_support_signal': policy_support_signal,
            })
        
        # 按优先级得分排序
        scored_demands.sort(
            key=lambda item: (
                -float(item['priority_score']),
                -float(item['proximity_score']),
                -float(item['demand_offer'].get('bid_price', 0.0) or 0.0),
                self._get_agent_key(item['demand_offer'].get('demander')),
                self._get_patch_key(item['demand_offer'].get('patch')),
            )
        )
        
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
        
        # @fix 2026-01-14: 消融实验 - 检查熟人网络开关
        acquaintance_enabled = True
        if hasattr(self, 'model_config') and self.model_config:
            mechanism_switches = self.model_config.get('mechanism_switches', {})
            acquaintance_enabled = mechanism_switches.get('acquaintance_network_enabled', True)
        
        # 第一阶段：熟人优先匹配
        # @critical: 中国农村固定观察点数据显示，熟人/村内流转占比55%-70%
        # @ablation: 若acquaintance_network_enabled=False，跳过熟人匹配，直接市场出清
        matched_supply = []
        matched_demand = []
        
        # @fix 2026-01-14: 消融实验 - 若禁用熟人网络，跳过整个熟人匹配阶段
        if not acquaintance_enabled:
            logger.debug("[消融] 熟人网络已禁用，跳过熟人匹配阶段")
        else:
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
        # 更新地块状态（使用新的多年期合同机制）
        # @critical 2025-01: 标记为农户直接流转渠道
        contract_years = 3 if not is_acquaintance else 5  # 熟人交易合同期更长（信任）
        lease_applied = patch.set_leased(
            demand_offer['demander'],
            contract_years=contract_years,
            rental_price=price,
            transfer_channel='direct_farmer'  # 关键：标记为农户直接流转
        )
        if lease_applied is False:
            return False

        # 记录交易
        transaction = {
            'patch': patch,
            'patch_id': self._get_patch_key(patch),
            'supplier': supply_offer['supplier'],
            'supplier_id': self._get_agent_key(supply_offer['supplier']),
            'demander': demand_offer['demander'],
            'demander_id': self._get_agent_key(demand_offer['demander']),
            'price': price,
            'price_per_mu': float(price) / max(float(getattr(patch, 'area', 0.0) or 0.0), 1e-6),
            'supplier_reservation': supply_offer['reservation_price'],
            'demander_bid': demand_offer['bid_price'],
            'supplier_is_elite': supply_offer['is_elite'],
            'demander_is_elite': demand_offer['is_elite'],
            'is_acquaintance': is_acquaintance,  # 熟人交易标记
            'is_contiguous': is_contiguous,      # 连片流转标记
            'transfer_channel': 'direct_farmer'  # 流转渠道：农户直接流转
        }
        self.transactions.append(transaction)
        
        # 更新农户和企业状态
        supplier = supply_offer['supplier']
        demander = demand_offer['demander']
        
        if hasattr(supplier, 'leased_out_patches'):
            if patch not in supplier.leased_out_patches:
                supplier.leased_out_patches.append(patch)
        
        if hasattr(demander, 'leased_patches'):
            if patch not in demander.leased_patches:
                demander.leased_patches.append(patch)
        if hasattr(demander, 'leased_in_patches'):
            if patch not in demander.leased_in_patches:
                demander.leased_in_patches.append(patch)
        if hasattr(demander, 'total_leased_area'):
            demander.total_leased_area = float(getattr(demander, 'total_leased_area', 0.0) or 0.0) + float(
                getattr(patch, 'area', 0.0) or 0.0
            )
        if hasattr(demander, '_operated_land_area_cache'):
            demander._operated_land_area_cache = float(
                getattr(demander, 'total_leased_area', getattr(demander, '_operated_land_area_cache', 0.0)) or 0.0
            )
        
        # 记录精英优势
        if supply_offer['is_elite'] or demand_offer['is_elite']:
            self.market_statistics['elite_advantage_count'] += 1
        
        return True
    
    def _determine_transaction_price(self, demand_offer, supply_offer):
        """
        @theory
        确定成交价格：基于博弈理论的价格形成机制
        考虑村精英优势和市场力量对比
        
        @new 2025-12-22: 企业类型因子
        专家审阅意见：合作社相比农业公司有信任优势，租金溢价
        参考：Feng & Huang (2023), 合作社社会信任资本研究
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
        
        # @new 2025-12-22: 企业类型调整
        # 合作社作为需求方时，农户更愿意接受稍低的租金（信任溢价）
        # 农业公司需要支付风险溢价来弥补信任缺失
        enterprise_type_adjustment = 0
        demander = demand_offer['demander']
        if hasattr(demander, 'enterprise_type'):
            from agents.enterprise_agent import EnterpriseType
            if demander.enterprise_type == EnterpriseType.COOPERATIVE:
                # 合作社：农户愿意折价10%（信任溢价）
                # 体现为对最终价格的负调整
                enterprise_type_adjustment = -base_price * 0.10
            elif demander.enterprise_type == EnterpriseType.AGRI_CORPORATION:
                # 农业公司：需要支付5%风险溢价
                enterprise_type_adjustment = base_price * 0.05
        
        # 市场力量对比调整
        price_range = bid_price - reservation_price
        market_power_adjustment = price_range * 0.1  # 10%的价格范围作为调整
        
        # 最终价格
        final_price = base_price + elite_bonus + enterprise_type_adjustment + market_power_adjustment
        
        # 确保价格在合理范围内
        return max(reservation_price, min(bid_price, final_price))
    
    def _fixed_rent_matching(self, fixed_rent_value: float, patch_supply, patch_demand):
        """
        @new 2026-01-14: 固定租金匹配（消融实验专用）
        
        当rent_market_enabled=False时调用，跳过市场定价机制，
        所有成交都使用固定租金价格。
        
        Args:
            fixed_rent_value: 固定租金（元/亩）
        """
        # 简单匹配：有供有需就成交，价格固定
        for patch in self._sorted_patch_sequence(patch_supply):
            supply_offers = patch_supply[patch]
            if patch not in patch_demand:
                continue
            demand_offers = patch_demand[patch]
            if not demand_offers:
                continue
            
            supply_offer = supply_offers[0]
            demand_offer = demand_offers[0]  # 取第一个需求方

            # 固定租金配置按元/亩输入，交易记录与地块状态统一保存总租金。
            total_fixed_rent = float(fixed_rent_value) * float(getattr(patch, 'area', 0.0) or 0.0)
            recorded = self._record_transaction(
                patch,
                supply_offer,
                demand_offer,
                total_fixed_rent,
                is_acquaintance=False,
            )
            if recorded is False:
                continue

            if self.transactions:
                self.transactions[-1]['contract_type'] = ContractType.FIXED_RENT
                self.transactions[-1]['ablation_fixed_rent'] = True
            self.market_statistics['fixed_rent_count'] += 1
        
        # 更新统计
        self._capture_flow_audit()
        self._update_market_statistics()
        
        # 清空报价
        self.supply_offers = []
        self.demand_offers = []
    
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
            total_price = sum(t['price'] for t in self.transactions)
            total_area = self.market_statistics['total_area_transferred']
            self.market_statistics['average_transaction_price'] = total_price / len(self.transactions)
            self.market_statistics['average_rent_per_mu'] = (
                total_price / total_area if total_area > 0 else 0.0
            )
            # Backward-compatible alias: downstream consumers expect this to be 元/亩.
            self.market_statistics['average_price'] = self.market_statistics['average_rent_per_mu']
            
            # 计算价格波动性
            prices = [t.get('price_per_mu', 0.0) for t in self.transactions]
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
                'average_transaction_price': 0,
                'average_rent_per_mu': 0,
                'total_area': 0,
                'elite_advantage_count': 0,
                'price_volatility': 0,
                'supply_count': getattr(self, 'last_supply_count', 0),
                'demand_count': getattr(self, 'last_demand_count', 0)
            }
        
        return {
            'transaction_count': len(self.transactions),
            'average_price': self.market_statistics['average_price'],
            'average_transaction_price': self.market_statistics['average_transaction_price'],
            'average_rent_per_mu': self.market_statistics['average_rent_per_mu'],
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

    def facilitate_land_swapping(self, model):
        """
        @new 2025-12-28: 土地置换机制 (Land Swapping)
        @theory: Deng et al. (2024, Nature Food) "Managing fragmented croplands"
        
        机制：
        1. 主动识别"错配"地块：A拥有的地块离B的基地近，且B拥有的地块离A的基地近
        2. 提议置换：如果置换后双方的"连片度"或"平均距离"都有改善，则进行置换
        3. 帕累托改进：置换不改变总面积，只改变空间配置
        
        执行频率：建议每年（每12步）执行一次
        """
        if not hasattr(model, 'farmers'):
            return 0
            
        swap_count = 0
        farmers = sorted(
            [f for f in model.farmers if hasattr(f, 'land_patches') and len(f.land_patches) > 1],
            key=self._get_agent_key,
        )
        rng = getattr(model, 'random', None)
        
        # 随机乱序以避免顺序偏差
        rng_shuffle(model, farmers)
        
        # 遍历农户寻找置换机会
        # 为避免O(N^2)，只检查"邻居"或随机抽样
        for i, farmer_a in enumerate(farmers):
            if i >= len(farmers) - 1:
                break
                
            # 简化：只与列表中的下一个农户尝试置换（随机配对效果）
            # 或者：寻找farmer_a的"远地"周围的农户
            
            # 1. 找出farmer_a最"远"的地块
            a_center = getattr(farmer_a, 'pos', (0,0))
            if not a_center: continue
            
            a_patches = sorted(farmer_a.land_patches, key=self._get_patch_key)
            # 计算每个地块到中心的距离
            a_dists = []
            for p in a_patches:
                if p.owner is not farmer_a:
                    continue
                if not hasattr(p, 'coordinates') or not p.coordinates:
                    continue
                area_p = float(getattr(p, 'area', 0.0) or 0.0)
                if not np.isfinite(area_p) or area_p <= 1e-9:
                    continue
                d = np.sqrt((p.coordinates[0]-a_center[0])**2 + (p.coordinates[1]-a_center[1])**2)
                a_dists.append((p, d))
            
            if not a_dists: continue
            
            # 按距离降序排列，取最远的一个
            a_dists.sort(key=lambda x: (-x[1], self._get_patch_key(x[0])))
            candidate_patch_a = a_dists[0][0]
            dist_a = a_dists[0][1]
            area_a = float(getattr(candidate_patch_a, 'area', 0.0) or 0.0)
            if not np.isfinite(area_a) or area_a <= 1e-9:
                continue
            
            # 2. 寻找潜在交换对象
            # 理想对象：拥有candidate_patch_a附近土地的农户
            # 这里简化：随机尝试5个其他农户
            partner_pool = [candidate for candidate in farmers if candidate is not farmer_a]
            partner_pool = sorted(partner_pool, key=self._get_agent_key)
            sample_size = min(5, len(partner_pool))
            if sample_size <= 0:
                continue
            potential_partners = rng_sample(model, partner_pool, sample_size)
            
            for farmer_b in potential_partners:
                if farmer_b == farmer_a: continue
                
                b_center = getattr(farmer_b, 'pos', (0,0))
                if not b_center: continue
                
                # 检查: patch_a是否离B更近？
                dist_a_to_b = np.sqrt((candidate_patch_a.coordinates[0]-b_center[0])**2 + (candidate_patch_a.coordinates[1]-b_center[1])**2)
                
                if dist_a_to_b >= dist_a:
                    continue # patch_a离B也远，没必要给B
                    
                # 3. 找出B拥有的、离A近的地块
                b_patches = sorted(farmer_b.land_patches, key=self._get_patch_key)
                best_patch_b = None
                best_improvement = 0
                
                for patch_b in b_patches:
                    if patch_b.owner is not farmer_b:
                        continue
                    if not hasattr(patch_b, 'coordinates') or not patch_b.coordinates:
                        continue
                    area_b = float(getattr(patch_b, 'area', 0.0) or 0.0)
                    if not np.isfinite(area_b) or area_b <= 1e-9:
                        continue
                        
                    # 距离计算
                    dist_b_current = np.sqrt((patch_b.coordinates[0]-b_center[0])**2 + (patch_b.coordinates[1]-b_center[1])**2)
                    dist_b_to_a = np.sqrt((patch_b.coordinates[0]-a_center[0])**2 + (patch_b.coordinates[1]-a_center[1])**2)
                    
                    # 只有当patch_b离A更近，且面积相近时才考虑
                    # 面积差异不超过20%
                    if not np.isfinite(area_a) or area_a <= 1e-9:
                        continue
                    try:
                        area_ratio = area_b / area_a
                    except ZeroDivisionError:
                        continue
                    if not np.isfinite(area_ratio):
                        continue
                    if not (0.8 <= area_ratio <= 1.2):
                        continue
                        
                    # 计算双赢幅度
                    # A的收益：缩短距离 (dist_a - dist_b_to_a)
                    # B的收益：缩短距离 (dist_b_current - dist_a_to_b)
                    gain_a = dist_a - dist_b_to_a
                    gain_b = dist_b_current - dist_a_to_b
                    
                    if gain_a > 0 and gain_b > 0:
                        total_gain = gain_a + gain_b
                        if total_gain > best_improvement:
                            best_improvement = total_gain
                            best_patch_b = patch_b
                
                # 4. 执行置换
                if best_patch_b:
                    self._execute_swap(farmer_a, candidate_patch_a, farmer_b, best_patch_b)
                    swap_count += 1
                    break # A已经换了一块，进入下一个A
        
        return swap_count

    def _execute_swap(self, farmer_a, patch_a, farmer_b, patch_b):
        """执行地块所有权置换"""
        if patch_a.owner is not farmer_a or patch_b.owner is not farmer_b:
            return
        area_a = float(getattr(patch_a, 'area', 0.0) or 0.0)
        area_b = float(getattr(patch_b, 'area', 0.0) or 0.0)
        if (not np.isfinite(area_a) or area_a <= 1e-9 or
                not np.isfinite(area_b) or area_b <= 1e-9):
            return
        # 1. 更新地块所有者
        patch_a.owner = farmer_b
        patch_b.owner = farmer_a
        
        # 2. 更新农户地块列表
        if patch_a in farmer_a.land_patches:
            farmer_a.land_patches.remove(patch_a)
        if patch_b not in farmer_a.land_patches:
            farmer_a.land_patches.append(patch_b)
            
        if patch_b in farmer_b.land_patches:
            farmer_b.land_patches.remove(patch_b)
        if patch_a not in farmer_b.land_patches:
            farmer_b.land_patches.append(patch_a)
    @staticmethod
    def clear_area_orders(orders, year, term, *, space=None, external=None, household_ids=()):
        """Uniform marginal midpoint per region; quantities are divisible physical mu."""
        from collections import defaultdict
        groups = defaultdict(list)
        for o in orders:
            if o['quantity'] > 1e-8 and o['region'] is not None:
                groups[o['region']].append(o)
        deals = []
        for region, group in sorted(groups.items()):
            sellers = sorted((o for o in group if o['side']=='sell'),key=lambda o:(o['price'],o['id']))
            buyers = sorted((o for o in group if o['side']=='buy'),key=lambda o:(-o['price'],o['id']))
            if not buyers or not sellers:
                continue
            if space is not None:
                # Candidate edges are produced in one sparse slice; conflict resolution
                # preserves the existing protected ask/bid and regional rent constraints.
                ix=space['index']; anchor=space['anchors']
                si=[ix[o['id']] for o in sellers]
                bi=[ix.get(anchor.get(o['id']),-1) for o in buyers]
                w=space['neighbors']
                sr=[s['quantity'] for s in sellers]
                lower,upper=0.,float('inf')
                local=[]
                for j,buyer in enumerate(buyers):
                    if bi[j]<0:
                        continue
                    if space.get('search',True):
                        connect=w[bi[j],si]
                        mask=set(connect.indices)
                        mask.update(i for i,a in enumerate(si) if a==bi[j])
                    else:
                        mask=range(len(sellers))
                    available=buyer['quantity']
                    for i in sorted(mask):
                        seller=sellers[i]
                        _,bx,by=space['coordinates'][anchor[buyer['id']]]
                        _,sx,sy=space['coordinates'][seller['id']]
                        fee=space.get('fee',0.) if space.get('search',True) and abs(bx-sx)+abs(by-sy)>1 else 0.
                        if available<=1e-8:
                            break
                        ask,bid=seller['price'],buyer['price']-fee
                        if sr[i]<=1e-8 or seller['id']==buyer['id'] or max(lower,ask)>min(upper,bid):
                            continue
                        q=min(sr[i],available)
                        base=seller['quantity']-sr[i]
                        local.append(dict(region=region,seller=seller['id'],buyer=buyer['id'],area=q,
                            ask=ask,bid=bid,start=year,end=year+term-1,plot=seller['id'],
                            slice_start=base,slice_end=base+q,search_fee=fee))
                        sr[i]-=q;available-=q
                        lower,upper=max(lower,ask),min(upper,bid)
                for d in local:
                    d['rent']=(lower+upper)/2.
                deals.extend(local)
                continue
            i=j=0
            s,b=sellers[0]['quantity'],buyers[0]['quantity']
            local=[]
            while i<len(sellers) and j<len(buyers):
                x,y=sellers[i],buyers[j]
                if y['price'] < x['price']:
                    break
                if x['id']==y['id']:
                    raise ValueError("self trade")
                q=min(s,b)
                local.append(dict(region=region,seller=x['id'],buyer=y['id'],area=q,
                                  ask=x['price'],bid=y['price'],start=year,end=year+term-1))
                s,b=s-q,b-q
                if s<1e-8:
                    i+=1
                    if i<len(sellers): s=sellers[i]['quantity']
                if b<1e-8:
                    j+=1
                    if j<len(buyers): b=buyers[j]['quantity']
            if local:
                price=(local[-1]['ask']+local[-1]['bid'])/2
                for d in local:
                    d['rent']=price
                    assert d['ask']<=price<=d['bid']
                deals.extend(local)
        if external is not None:
            assert term==1
            used=defaultdict(float)
            regional_rent={}
            for d in deals:
                used[d['seller'],'sell']+=d['area']
                used[d['buyer'],'buy']+=d['area']
                regional_rent[d['region']]=d['rent']
            for region,profile in sorted(external.items()):
                rent=regional_rent.get(region,profile['reference_rent'])
                pool=f'_area_pool_{region}'
                for side,limit in [('buy','supply'),('sell','demand')]:
                    room=profile[limit]
                    eligible=[o for o in groups.get(region,[]) if o['id'] in household_ids and o['side']==side]
                    for order in sorted(eligible,key=lambda o:(-o['price'] if side=='buy' else o['price'],o['id'])):
                        if room<=1e-8:break
                        if (side=='buy' and order['price']<rent) or (side=='sell' and order['price']>rent):continue
                        actor=order['id']
                        offset=used[actor,side]
                        amount=min(room,order['quantity']-offset)
                        if amount<=1e-8:continue
                        room-=amount
                        selling=side=='sell'
                        deals.append(dict(region=region,seller=actor if selling else pool,buyer=pool if selling else actor,
                            area=amount,ask=order['price'] if selling else rent,bid=rent if selling else order['price'],
                            start=year,end=year,plot=actor if selling else None,slice_start=offset if selling else None,
                            slice_end=offset+amount if selling else None,search_fee=0.,rent=rent,
                            channel='external_demand' if selling else 'external_supply'))
        return deals
