# -*- coding: utf-8 -*-
"""
@theory
本文件定义了 LandPatch 类，代表具有异质性的地块。
地块具有面积、土壤质量、气候区等属性，支持多种土地利用选择。
新增地理坐标和空间邻域机制，支持空间ABM。

@critical 2025-12: 添加土地细碎化指数
参考：Tan et al. (2006) 中国耕地细碎化的定量测度
实证数据：中国户均6.4块地块，平均每块0.5亩

@critical 2025-01: 添加土地原用途和流转渠道追踪
理论依据（江浙实践经验）：
1. 原种粮地流转后必须种粮 → previous_crop_type='grain' 时有更严格约束
2. 村集体统一流转 vs 农户直接流转有不同的种粮要求
3. 政策目标是"防止非粮化"而非"强制粮化"

流转渠道分类：
- 'village_collective': 村集体统一组织流转（约30%），有严格种粮要求
- 'direct_farmer': 农户直接流转（约70%），约束相对较弱
- None: 未流转，自己经营
"""

import hashlib
from dataclasses import asdict, replace
import numpy as np
from core.fixed_rule_runtime import action_income_enabled
from core.reproducibility import np_random_value, np_uniform

# 流转渠道枚举
class TransferChannel:
    """土地流转渠道类型"""
    VILLAGE_COLLECTIVE = 'village_collective'  # 村集体统一流转
    DIRECT_FARMER = 'direct_farmer'            # 农户直接流转
    NONE = None                                 # 未流转


# @new 2025-12: 土地类型枚举
class LandType:
    """
    土地类型分类（基于国土资源管理实践）
    
    理论依据：
    1. 《基本农田保护条例》：永久基本农田重点保护
    2. 《防止耕地"非粮化"意见》(2020)：基本农田必须种粮
    3. 现实比例：基本农田约占耕地总面积的80%
    """
    BASIC_FARMLAND = 'basic_farmland'      # 永久基本农田（80%种粮刚性约束）
    GENERAL_FARMLAND = 'general_farmland'  # 一般耕地（无强制约束，激励引导）


class LandPatch:
    def __init__(self, area, soil_quality, climate_zone, yield_profiles, coordinates=(0, 0), model=None):
        # 地块基本属性
        self.area = area
        self.soil_quality = soil_quality
        self.climate_zone = climate_zone
        self.yield_profiles = yield_profiles
        self.model = model
        
        # 地理坐标
        self.coordinates = coordinates  # (x, y)坐标
        self.neighbors = []  # 空间邻域地块列表
        self.patch_id = None  # Prefer builder-assigned stable IDs for audit-grade reproducibility.

        # Forecast-audit geography. These fields are intentionally separate
        # from grid coordinates: national forecast weighting must trace the
        # official microdata region of the land, not the actor's runtime grid
        # position. They are populated by farmer/microdata initialization when
        # an official macro-region label is available.
        self.region = None
        self.region_scope = None
        self.official_region = None
        self.official_region_scope = None
        
        # @critical 2025-12: 土地细碎化指数
        # fragmentation_index ∈ [0.5, 1.0]
        # 1.0 = 完整连片地块（无效率损失）
        # 0.5 = 高度细碎地块（50%效率损失）
        # 影响：机械作业效率、田间管理成本、规模经济实现程度
        # 计算方法：基于地块面积和形状因子
        self.fragmentation_index = self._calculate_fragmentation_index(area)
        
        # 土地利用状态
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = None
        
        # @critical 2025-12: 土地类型（简化版种粮约束核心）
        # land_type: 土地分类，决定种粮约束强度
        # - 'basic_farmland': 永久基本农田（约80%），流转后必须保证≥80%种粮
        # - 'general_farmland': 一般耕地（约20%），无强制约束，靠激励引导
        # 初始化时随机分配，80%概率为基本农田
        self.land_type = LandType.BASIC_FARMLAND if np_random_value(self) < 0.80 else LandType.GENERAL_FARMLAND
        
        # @deprecated 2025-12: 以下属性保留但简化，主要用于数据记录
        self.previous_crop_type = None  # 'grain', 'cash_crop', 或 None
        self.crop_history = []  # 最近3年作物历史（用于统计）
        self.historical_grain_ratio = 0.5  # 默认50%（用于统计）
        
        # @deprecated 2025-12: 流转渠道不再影响种粮约束
        # 保留属性用于数据追踪，但不影响约束计算
        self.transfer_channel = None  # 'village_collective', 'direct_farmer', 或 None
        
        # 流转和弃耕状态
        self.leased_to = None  # 承租方ID
        self.rental_price = 0.0  # @fix: 初始化租金价格，避免hasattr检查失败
        self.abandoned = False
        
        # @new 2025-12-22: 土地质量内生演化参数
        # 理论依据：agriLOVE模型的土壤退化机制
        # 高强度利用（企业化、规模化）导致地力透支
        self.initial_soil_quality = soil_quality  # 初始土地质量
        self.consecutive_cropping_years = 0  # 连作年数
        self.fertilizer_intensity = 1.0  # 化肥使用强度 (1.0=正常, >1=过度)
        self.soil_quality_history = [soil_quality]  # 土地质量历史
        
        # @critical 2025-12: 经营者补贴机制
        # current_operator: 当前实际经营者（可能是owner自己，也可能是租入者）
        # 补贴发放时根据subsidy_target判断发给owner还是current_operator
        self.current_operator = None  # 初始为None，创建后由owner设置
        
        # @critical 2025-12: 多年期租赁合同
        # 现实中土地租赁3-5年一签，而非每年重新竞价
        self.lease_contract_years = 0  # 剩余合同年限
        self.lease_start_year = None   # 合同起始年份
        self.contracted_rental_price = 0.0  # 合同约定租金（锁定价格）
        
        # 技术相关
        self.technology_level = 1.0  # 技术水平
        self.owner = None  # 地块所有者（承包权人）
        
        # 新增：模型引用
        self.model = model

    def get_patch_id(self):
        """Return a process-stable patch identifier for ledgers and reproducibility tests."""
        if self.patch_id is not None:
            return str(self.patch_id)

        owner = getattr(self, 'owner', None)
        owner_id = getattr(owner, 'unique_id', 'unassigned')
        coords = self.coordinates if self.coordinates else (0, 0)
        payload = (
            f"{owner_id}|{self.climate_zone}|"
            f"{coords[0]}|{coords[1]}|"
            f"{float(self.area):.8f}|{float(self.soil_quality):.8f}"
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
        return f"patch_{digest}"
    
    def _calculate_fragmentation_index(self, area):
        """
        @critical 2025-12: 计算土地细碎化指数
        
        理论依据：
        - Tan et al. (2006): 中国耕地细碎化测度
        - 实证数据：地块越小，单位面积管理成本越高
        
        计算逻辑：
        - 大地块（≥20亩）：接近1.0，规模经济充分实现
        - 中地块（5-20亩）：0.85-0.95，部分规模经济
        - 小地块（1-5亩）：0.70-0.85，规模经济受限
        - 极小地块（<1亩）：0.50-0.70，严重细碎化
        
        Args:
            area: 地块面积（亩）
        
        Returns:
            float: 细碎化指数 [0.5, 1.0]
        """
        if area >= 20:
            # 大规模地块，规模经济充分
            base_index = 0.95 + 0.05 * min(1.0, (area - 20) / 80)
        elif area >= 5:
            # 中等地块
            base_index = 0.85 + 0.10 * (area - 5) / 15
        elif area >= 1:
            # 小地块
            base_index = 0.70 + 0.15 * (area - 1) / 4
        else:
            # 极小地块（<1亩）
            base_index = 0.50 + 0.20 * area
        
        # 添加随机形状因子扰动（±5%）
        # 反映地块形状不规则性（长条形比方形更难耕作）
        shape_factor = float(np_uniform(self, 0.95, 1.05))
        
        return min(1.0, max(0.5, base_index * shape_factor))

    def generate_land_use_options(self):
        """
        @theory
        生成土地利用选项：种植、流转、弃耕
        考虑地块属性和当前状态
        """
        options = ["种小麦", "种玉米"]
        
        # 如果未被承租且未弃耕，可以流转
        if self.leased_to is None and not self.abandoned:
            options.append("leased")
        
        # 如果未被承租，可以弃耕
        if self.leased_to is None:
            options.append("abandoned")
        
        return options

    def get_prospects(self, crop):
        """
        @theory
        获取作物种植的前景分布
        考虑土壤质量、气候区、技术水平等因素
        
        @critical 2025-12: 加入细碎化指数影响
        细碎化降低规模经济效应，体现为单位面积产量下降
        """
        if crop not in self.yield_profiles:
            return [(0, 1.0)]
        
        base_prospects = self.yield_profiles[crop]
        
        # 考虑土壤质量的影响
        soil_factor = self.soil_quality
        
        # 考虑技术水平的影响
        tech_factor = self.technology_level
        
        # @critical 2025-12: 考虑细碎化的影响
        # 细碎化降低机械作业效率和规模经济
        fragmentation_factor = self.fragmentation_index
        
        # 调整后的前景
        adjusted_prospects = []
        for income, prob in base_prospects:
            adjusted_income = income * soil_factor * tech_factor * fragmentation_factor
            adjusted_prospects.append((adjusted_income, prob))
        
        return adjusted_prospects

    def set_land_use(self, crop):
        """
        @theory
        设置土地利用类型
        更新地块状态和产量
        """
        if crop == "leased":
            self.set_leased(None)
        elif crop == "abandoned":
            self.set_abandoned()
        elif crop in self.yield_profiles:
            self.current_crop = crop
            self.last_land_use = crop
            self.abandoned = False
            self.leased_to = None
            self.current_operator = self.owner
            
            # 计算预期产量
            prospects = self.get_prospects(crop)
            self.last_yield = sum([income * prob for income, prob in prospects])
            if self.model is not None and hasattr(self.model, 'invalidate_land_tenure_caches'):
                self.model.invalidate_land_tenure_caches()
        else:
            raise ValueError(f"不支持的作物类型: {crop}")

    def set_abandoned(self):
        """
        @theory
        设置地块为弃耕状态
        """
        self.abandoned = True
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = "abandoned"
        self.leased_to = None
        self.current_operator = None
        if self.model is not None and hasattr(self.model, 'invalidate_land_tenure_caches'):
            self.model.invalidate_land_tenure_caches()
    
    def update_soil_quality(self):
        """
        @new 2025-12-22: 土地质量内生演化机制
        
        理论依据：
        1. agriLOVE模型的土壤退化机制 (Ferraro et al.)
        2. 高强度利用（企业化、规模化）往往导致地力透支
        3. 小农可能精耕细作养地
        
        反馈链条：
        耕作方式 → 土壤退化/恢复 → 产量变化 → 收益受损/改善
        
        关键参数：
        - 恢复力：休耕或轮作带来的恢复
        - 损耗：化肥过度使用、连作带来的退化
        """
        # 基础恢复率
        base_recovery = 0.02  # 每年自然恢复2%
        
        # 基础退化率
        base_degradation = 0.01  # 每年自然退化1%
        
        # === 恢复因素 ===
        recovery = 0.0
        
        # 1. 休耕恢复（撂荒地的生态恢复）
        if self.abandoned or self.current_crop is None:
            recovery += 0.05  # 休耕地恢复5%
            self.consecutive_cropping_years = 0  # 重置连作年数
        else:
            self.consecutive_cropping_years += 1
        
        # 2. 轮作恢复（粮食-经济作物轮作有利于土壤）
        if hasattr(self, 'last_crop_type') and self.current_crop:
            if self.last_crop_type != self.current_crop:
                recovery += 0.02  # 轮作额外恢复2%
        
        # 记录上一期作物
        self.last_crop_type = self.current_crop
        
        # === 退化因素 ===
        degradation = base_degradation
        
        # 1. 连作退化（连续种植同一作物加速土壤退化）
        if self.consecutive_cropping_years > 3:
            degradation += 0.01 * (self.consecutive_cropping_years - 3)  # 连作3年后加速退化
        
        # 2. 化肥过度使用（规模经营者可能为短期利润过度使用化肥）
        if self.current_operator is not None:
            operator = self.current_operator
            # 检查经营者类型
            operator_type = getattr(operator, '__class__', None)
            operator_name = operator_type.__name__ if operator_type else ''
            
            if 'Enterprise' in operator_name:
                # 企业经营者可能有更高的集约化程度
                intensification = getattr(operator, 'intensification_level', 1.0)
                if intensification > 1.2:  # 高强度集约化
                    degradation += 0.02 * (intensification - 1.0)
                    self.fertilizer_intensity = intensification
            elif 'Farmer' in operator_name:
                # 农户经营者
                # 纯农户可能更注重养地
                ag_weight = getattr(operator, 'get_agricultural_weight', lambda: 0.5)()
                if ag_weight > 0.8:  # 纯农户
                    degradation -= 0.005  # 精耕细作减缓退化
        
        # 3. 地块面积因素（大规模地块机械化作业可能导致土壤压实）
        if self.area > 50:
            degradation += 0.005  # 大规模地块额外退化
        
        # === 计算新的土地质量 ===
        delta = base_recovery + recovery - degradation
        new_quality = self.soil_quality + delta
        
        # 边界约束：土地质量在[0.2, 1.0]范围内
        self.soil_quality = max(0.2, min(1.0, new_quality))
        
        # 记录历史
        self.soil_quality_history.append(self.soil_quality)
        
        return self.soil_quality

    def _operational_transfer_investment(self, rental_price):
        area = max(0.0, float(getattr(self, 'area', 0.0) or 0.0))
        rent = max(0.0, float(rental_price or 0.0))
        return area * rent

    def _passes_operational_financial_gate(self, to_agent, rental_price):
        gate = getattr(to_agent, '_passes_investment_sunk_cost_gate', None)
        if not callable(gate):
            return True
        action = getattr(to_agent, 'fixed_rule_current_action', None)
        if action_income_enabled(getattr(self.model, 'config', {})) and action is not None:
            rent = max(0.0, float(rental_price or 0.0))
            before = to_agent.expected_fixed_rule_account(action)
            already_operated = (
                self.leased_to is to_agent or self.current_operator is to_agent
                or (self.owner is to_agent and self.leased_to is None)
            )
            area = before.area_mu + (0.0 if already_operated else self.area)
            candidate = to_agent.expected_fixed_rule_account(action, area)
            prior_patch_rent = self.rental_price if self.leased_to is to_agent and self.owner is not to_agent else 0.0
            new_patch_rent = rent if self.owner is not to_agent else 0.0
            candidate = replace(candidate, rent_in_yuan=before.rent_in_yuan - prior_patch_rent + new_patch_rent)
            reason = candidate.constraint_reason
            if action in {'exit', 'contract'}:
                reason = '|'.join(filter(None, (reason, 'selected_action_no_new_lease')))
            if not reason and rent > 0.0 and not gate(self, rent):
                reason = 'balance_sheet_gate'
            if hasattr(to_agent, 'fixed_rule_bid_accounts'):
                row = {
                    'year': self.model.simulation_base_year + self.model.step_count,
                    'operator_id': str(to_agent.unique_id), 'action': action,
                    'patch_id': str(self.get_patch_id()), 'record_type': 'lease_execution_gate',
                    'bid_yuan': rent, 'lease_gate_passed': not reason, 'reason': reason,
                    'before_working_capital_required': before.working_capital_required,
                    'after_working_capital_required': candidate.working_capital_required,
                    'replaced_patch_rent_yuan': prior_patch_rent,
                    'candidate_land_cost_yuan': new_patch_rent,
                }
                for prefix, account in (('before', before), ('after', candidate)):
                    row.update({f'{prefix}_{key}': value for key, value in asdict(account).items()})
                to_agent.fixed_rule_bid_accounts.append(row)
            return not reason
        investment = self._operational_transfer_investment(rental_price)
        if investment <= 0.0:
            return True  # ponytail: zero-rent succession has no cash outlay; use imputed rent if policy needs it.
        return bool(gate(self, investment))

    def set_leased(self, to_agent, contract_years=3, rental_price=0.0, 
                   transfer_channel=None, via_village=False):
        """
        @theory
        设置地块为流转状态
        
        @critical 2025-12: 完善流转机制
        1. 更新current_operator为实际经营者
        2. 设置多年期合同（默认3年，现实中3-5年一签）
        3. 锁定合同期内租金价格
        
        @critical 2025-01: 记录土地原用途和流转渠道
        4. 保存流转前的作物类型（previous_crop_type）
        5. 记录流转渠道（村集体 vs 农户直接）
        
        Args:
            to_agent: 承租方Agent
            contract_years: 合同年限（默认3年）
            rental_price: 合同约定租金
            transfer_channel: 流转渠道 ('village_collective' 或 'direct_farmer')
            via_village: 是否通过村集体流转（旧参数，向后兼容）
        """
        if (getattr(self, 'operational_transfer_channel', '') == 'author_paid_initial_lease'
                and self.leased_to is not None and self.lease_contract_years > 0):
            return False
        is_new_lease = to_agent is not None and self.leased_to is None
        if self.model is not None:
            clearing_house = getattr(self.model, 'land_clearing_house', None)
            if (
                is_new_lease and
                clearing_house is not None and
                hasattr(clearing_house, '_check_transfer_rate_limit') and
                not clearing_house._check_transfer_rate_limit(
                    self.model,
                    float(getattr(self, 'area', 0.0) or 0.0),
                    transfer_channel=transfer_channel or ('village_collective' if via_village else 'direct_farmer'),
                    demander=to_agent,
                )
            ):
                return False
            needs_financial_gate = is_new_lease or (
                action_income_enabled(self.model.config) and to_agent is not None and to_agent is not self.leased_to
            )
            if needs_financial_gate and not self._passes_operational_financial_gate(to_agent, rental_price):
                return False

        # @critical 2025-01: 记录流转前的作物类型
        # 在流转前保存当前作物状态，用于后续种粮约束判定
        if self.current_crop is not None:
            # 将作物类型转换为字符串标识
            crop_str = str(self.current_crop).lower()
            if 'grain' in crop_str or '粮' in crop_str or '小麦' in crop_str or '玉米' in crop_str or '水稻' in crop_str:
                self.previous_crop_type = 'grain'
            elif 'cash' in crop_str or '经济' in crop_str or '蔬菜' in crop_str or '水果' in crop_str:
                self.previous_crop_type = 'cash_crop'
            else:
                # 默认按粮食处理（保守原则）
                self.previous_crop_type = 'grain'
        elif self.last_land_use is not None:
            # 如果current_crop为空，尝试从last_land_use推断
            use_str = str(self.last_land_use).lower()
            if 'grain' in use_str or '粮' in use_str:
                self.previous_crop_type = 'grain'
            elif 'cash' in use_str or '经济' in use_str:
                self.previous_crop_type = 'cash_crop'
            else:
                self.previous_crop_type = 'grain'  # 默认
        else:
            # 无法推断，默认为粮食（保守原则：防止非粮化）
            self.previous_crop_type = 'grain'
        
        # @critical 2025-01: 记录流转渠道
        if transfer_channel is not None:
            self.transfer_channel = transfer_channel
        elif via_village:
            self.transfer_channel = 'village_collective'
        else:
            self.transfer_channel = 'direct_farmer'
        
        self.leased_to = to_agent
        self.current_operator = to_agent  # 经营权转移给承租方
        self.current_crop = None
        self.last_yield = 0
        self.last_land_use = "leased"
        self.abandoned = False
        
        # 多年期合同
        self.lease_contract_years = contract_years
        # @fix 2026-01-07: 使用step_count替代current_step
        self.lease_start_year = self.model.step_count if self.model else 0
        self.contracted_rental_price = rental_price
        self.rental_price = rental_price
        if is_new_lease and self.model is not None:
            control = getattr(self.model, 'transfer_rate_control', None)
            if isinstance(control, dict):
                patch_area = float(getattr(self, 'area', 0.0) or 0.0)
                control['annual_new_transferred_area'] = (
                    float(control.get('annual_new_transferred_area', 0.0) or 0.0) + patch_area
                )
                control['cumulative_transferred_area'] = (
                    float(control.get('cumulative_transferred_area', 0.0) or 0.0) + patch_area
                )
                current_area = float(control.get('observed_current_transferred_area', 0.0) or 0.0) + patch_area
                control['observed_current_transferred_area'] = current_area
                total_area = float(
                    control.get('baseline_total_land_area', control.get('total_land_area', 0.0)) or 0.0
                )
                if total_area > 0:
                    control['observed_total_land_area'] = total_area
                    control['current_transfer_rate'] = current_area / max(total_area, 1.0)
        if self.model is not None and hasattr(self.model, 'invalidate_land_tenure_caches'):
            self.model.invalidate_land_tenure_caches()
        return True
    
    def end_lease(self):
        """
        @critical 2025-12: 合同到期，经营权回归owner
        """
        if (getattr(self, 'operational_transfer_channel', '') == 'author_paid_initial_lease'
                and self.leased_to is not None and self.lease_contract_years > 0):
            raise RuntimeError('Unexpired paid initial lease cannot be released by production exit')
        former_lessee = self.leased_to
        had_active_lease = former_lessee is not None
        patch_area = float(getattr(self, 'area', 0.0) or 0.0)
        authorized_tenure = hasattr(self, 'initial_rental_rights_holder')
        return_holder = getattr(self, 'initial_rental_rights_holder', self.owner)
        former_owner = self.owner
        former_operator = self.current_operator or former_lessee or self.owner
        bank = None
        if authorized_tenure:
            holder_active = (
                return_holder in getattr(self.model, 'farmers', [])
                and not getattr(return_holder, 'fixed_rule_exit_pending', False)
            )
            if not holder_active:
                bank = getattr(getattr(self.model, 'population_manager', None), 'land_bank', None)
                if bank is None:
                    message = 'Initial tenure return requires an existing LandBank for an inactive holder'
                    raise RuntimeError(message)
                if any(self is patch for parcel in bank.parcels for patch in parcel.patches):
                    return
        self.leased_to = None
        self.current_operator = return_holder if bank is None else None
        self.lease_contract_years = 0
        self.lease_start_year = None
        self.contracted_rental_price = 0.0
        self.rental_price = 0.0
        if former_lessee is not None:
            if hasattr(former_lessee, 'leased_in_patches'):
                while self in former_lessee.leased_in_patches:
                    former_lessee.leased_in_patches.remove(self)
            if hasattr(former_lessee, 'leased_patches'):
                while self in former_lessee.leased_patches:
                    former_lessee.leased_patches.remove(self)
            if hasattr(former_lessee, 'total_leased_area'):
                former_lessee.total_leased_area = max(
                    0.0,
                    float(getattr(former_lessee, 'total_leased_area', 0.0) or 0.0) - patch_area,
                )
            if hasattr(former_lessee, '_operated_land_area_cache'):
                former_lessee._operated_land_area_cache = max(
                    0.0,
                    float(getattr(former_lessee, '_operated_land_area_cache', 0.0) or 0.0) - patch_area,
                )
        if authorized_tenure:
            setattr(self, 'operational_transfer_active', False)
            setattr(self, 'operational_transfer_channel', '')
            if bank is not None:
                from modules.population_dynamics_v2 import LandParcel

                bank._detach_patch_from_current_tenure(self)
                bank.deposit(LandParcel(
                    area_mu=patch_area,
                    quality=self.soil_quality,
                    previous_owner_id=getattr(former_operator, 'unique_id', None),
                    exit_year=bank.current_year,
                    exit_type='lease_return' if had_active_lease else 'operational_return',
                    patches=[self],
                ))
            else:
                self.owner = return_holder
                holder_patches = getattr(return_holder, 'land_patches')
                if self not in holder_patches:
                    holder_patches.append(self)
            if former_owner is not self.owner and hasattr(former_owner, 'land_area_mu'):
                setattr(former_owner, 'land_area_mu', max(
                    0.0, float(getattr(former_owner, 'land_area_mu')) - patch_area,
                ))
        if had_active_lease and self.model is not None:
            control = getattr(self.model, 'transfer_rate_control', None)
            if isinstance(control, dict):
                current_area = max(
                    0.0,
                    float(control.get('observed_current_transferred_area', 0.0) or 0.0) - patch_area,
                )
                control['observed_current_transferred_area'] = current_area
                baseline_area = float(
                    control.get('baseline_total_land_area', control.get('total_land_area', 0.0)) or 0.0
                )
                if baseline_area > 0:
                    control['current_transfer_rate'] = current_area / max(baseline_area, 1.0)
        if (had_active_lease or authorized_tenure) and self.model is not None:
            if hasattr(self.model, 'invalidate_land_tenure_caches'):
                self.model.invalidate_land_tenure_caches()
    
    def update_lease_contract(self):
        """
        @critical 2025-12: 每年更新合同状态
        合同年限-1，到期则自动解除
        
        Returns:
            bool: True表示合同仍有效，False表示合同到期
        """
        if self.leased_to is not None and self.lease_contract_years > 0:
            self.lease_contract_years -= 1
            if self.lease_contract_years <= 0:
                # 合同到期，但不自动解除，等待续约或解约决策
                return False
        return True
    
    def is_lease_expiring(self):
        """合同是否即将到期（剩余1年内）"""
        return self.leased_to is not None and self.lease_contract_years <= 1

    def get_spatial_neighbors(self):
        """
        @theory
        获取空间邻域地块
        用于技术扩散和信息传播
        """
        return self.neighbors

    def calculate_distance_to(self, other_patch):
        """
        @theory
        计算到其他地块的距离
        用于空间价格梯度和邻域效应
        """
        if not hasattr(other_patch, 'coordinates'):
            return float('inf')
        
        dx = self.coordinates[0] - other_patch.coordinates[0]
        dy = self.coordinates[1] - other_patch.coordinates[1]
        return np.sqrt(dx**2 + dy**2)

    def is_neighbor_of(self, other_patch, radius=2):
        """
        @theory
        判断是否为邻域地块
        基于距离阈值定义邻域关系
        """
        distance = self.calculate_distance_to(other_patch)
        return distance <= radius

    def update_technology_level(self, new_level):
        """
        @theory
        更新技术水平
        影响产量和决策
        """
        self.technology_level = max(1.0, min(2.0, new_level))

    def get_spatial_price(self, base_price):
        """
        @theory
        空间价格梯度：基于Fujita et al.(1999)空间经济学理论
        
        空间价格函数：
        P(x,y) = P_market - α × d(x,y)
        
        其中：
        - P_market: 市场中心价格
        - α: 运输成本率
        - d(x,y): 到市场中心的距离
        
        运输成本函数：
        TransportCost = distance × costRate × area
        
        基于中国农村交通成本数据，运输成本率在0.03-0.08之间
        """
        market_center = self.model.market_center
        distance = np.sqrt((self.coordinates[0] - market_center[0])**2 + 
                          (self.coordinates[1] - market_center[1])**2)
        
        # 空间价格梯度
        transport_cost_rate = self.model.transport_cost_rate
        spatial_price = max(0, base_price - distance * transport_cost_rate)
        
        return spatial_price

    # ==================== 种粮约束相关方法 ====================
    
    def update_crop_history(self, crop_type):
        """
        @critical 2025-01: 更新作物种植历史
        每年种植结束后调用，记录该年种植的作物类型
        
        Args:
            crop_type: 本年种植的作物类型 ('grain' 或 'cash_crop')
        """
        # 标准化作物类型
        crop_str = str(crop_type).lower() if crop_type else 'unknown'
        if 'grain' in crop_str or '粮' in crop_str or '小麦' in crop_str or '玉米' in crop_str or '水稻' in crop_str:
            normalized_crop = 'grain'
        elif 'cash' in crop_str or '经济' in crop_str or '蔬菜' in crop_str or '水果' in crop_str:
            normalized_crop = 'cash_crop'
        else:
            normalized_crop = 'grain'  # 默认按粮食处理（保守原则）
        
        # 添加到历史记录
        self.crop_history.append(normalized_crop)
        
        # 只保留最近3年
        if len(self.crop_history) > 3:
            self.crop_history = self.crop_history[-3:]
        
        # 更新历史种粮比例
        self._update_historical_grain_ratio()
    
    def _update_historical_grain_ratio(self):
        """
        @critical 2025-01: 计算过去3年的历史种粮比例
        """
        if not self.crop_history:
            self.historical_grain_ratio = 0.5  # 无历史数据，默认50%
            return
        
        grain_count = sum(1 for crop in self.crop_history if crop == 'grain')
        self.historical_grain_ratio = grain_count / len(self.crop_history)
    
    def get_grain_requirement_ratio(self, base_ratio=0.80):
        """
        @critical 2025-12: 获取本地块的种粮约束比例（简化版）
        
        简化规则（基于土地类型）：
        - 永久基本农田（约80%耕地）：流转后≥80%种粮（刚性约束）
        - 一般耕地（约20%耕地）：无强制约束，靠激励政策引导
        
        理论依据：
        1. 《基本农田保护条例》：永久基本农田重点保护
        2. 《防止耕地"非粮化"意见》(2020)：基本农田必须种粮
        3. 符合中国耕地管理实践：约80%耕地为基本农田
        4. 简化了流转渠道的差异化约束，更清晰
        
        Args:
            base_ratio: 基本农田的基准种粮约束（默认0.80）
        
        Returns:
            float: 该地块的种粮约束比例 [0, 0.80]
        """
        # 未流转的土地无外部约束（由经营者自主决定）
        if self.leased_to is None:
            return 0.0
        
        # @critical 2025-12: 基于土地类型的简化约束
        if self.land_type == LandType.BASIC_FARMLAND:
            # 永久基本农田：刚性约束80%种粮
            return base_ratio  # 0.80
        else:
            # 一般耕地：无强制约束
            return 0.0
    
    def requires_grain_planting(self):
        """
        @critical 2025-12: 判断该地块是否必须种粮（简化版）
        
        Returns:
            bool: True表示该地块为基本农田，必须种粮
        """
        return self.land_type == LandType.BASIC_FARMLAND and self.leased_to is not None
    
    def is_basic_farmland(self):
        """
        @critical 2025-12: 判断是否为永久基本农田
        
        Returns:
            bool: True表示是永久基本农田
        """
        return self.land_type == LandType.BASIC_FARMLAND
    
    def is_general_farmland(self):
        """
        @critical 2025-12: 判断是否为一般耕地
        
        Returns:
            bool: True表示是一般耕地
        """
        return self.land_type == LandType.GENERAL_FARMLAND

    # @deprecated 2025-12: 以下方法保留用于向后兼容，但不再影响约束计算
    def is_from_village_collective(self):
        """
        @deprecated 2025-12: 判断是否为村集体流转土地
        保留用于数据统计，不再影响种粮约束
        
        Returns:
            bool: True表示是村集体统一流转的土地
        """
        return self.transfer_channel == 'village_collective'
    
    def was_grain_land(self):
        """
        @deprecated 2025-12: 判断流转前是否为种粮土地
        保留用于数据统计，不再影响种粮约束
        
        Returns:
            bool: True表示流转前种植粮食
        """
        return self.previous_crop_type == 'grain'
