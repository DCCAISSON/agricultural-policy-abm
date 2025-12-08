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
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


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
        super().__init__(unique_id, model)
        
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
        
        # 流转中介成功率（主动组织时的成功率提升）
        self.mediation_success_boost = village_config.get('mediation_success_boost', 0.2)
        
        # 土地整合阈值（超过此面积才进行整合）
        self.integration_threshold = village_config.get('integration_threshold', 50)  # 亩
        
        # 最小整合面积（整合后每块地不小于此面积）
        self.min_integrated_area = village_config.get('min_integrated_area', 10)  # 亩
        
        # 村集体参与流转的比例（约30%的流转通过村委会）
        self.participation_rate = village_config.get('participation_rate', 0.30)
        
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
        self.market_info = {
            'avg_rental_price': 300.0,      # 平均租金
            'grain_price': 2.58,            # 粮食价格
            'cash_crop_price': 5.0,         # 经济作物价格
            'policy_subsidy_rate': 0.15     # 政策补贴率
        }
        
        self.rental_price_history = []      # 租金历史
        self.expected_rental_price = 300.0   # 预期租金
        
        # ========== 新增：理性预期机制 ==========
        self._initialize_rational_expectation()
        
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
        """
        # 阶段0：信息收集与预期更新
        self._collect_market_information()
        self._update_rational_expectations()
        
        # 阶段1：策略选择（基于集体经济最大化目标）
        self._select_optimal_strategy()
        
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
            if subsidy_rate:
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
                getattr(f, 'total_income', 0) for f in self.model.farmers
            ])
            target_income = 30000  # 目标人均收入3万元
            scores['welfare'] = min(1.0, avg_income / target_income) if target_income > 0 else 0.5
        else:
            scores['welfare'] = 0.5
        
        # 3. 粮食安全得分（种粮面积占比）
        if hasattr(self.model, 'farmers'):
            total_grain_area = 0
            total_area = 0
            for farmer in self.model.farmers:
                for patch in getattr(farmer, 'land_patches', []):
                    if patch.owner == farmer:
                        total_area += patch.area
                        if hasattr(farmer, 'planned_crop') and str(farmer.planned_crop) == 'Grain':
                            total_grain_area += patch.area
            
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
        
        # 简化判断：Non_Farmer和老年农户更愿意
        from agents.farmer_agent import LivelihoodStrategy
        if hasattr(owner, 'livelihood_strategy'):
            if owner.livelihood_strategy == LivelihoodStrategy.NON_FARMER:
                return True
            if hasattr(owner, 'age') and owner.age >= 60:
                return True
            # 兼业户有一定概率
            if owner.livelihood_strategy == LivelihoodStrategy.PLURIACTIVE:
                return random.random() < 0.3
        
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
        
        for farmer in self.registered_farmers:
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
                willingness = getattr(enterprise, 'expansion_willingness', 0.5)
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
        sorted_demanders = sorted(demanders, key=lambda x: -x['willingness'])
        
        for supplier in sorted_suppliers:
            if not sorted_demanders:
                break
            
            # 村集体参与的概率
            if random.random() > self.participation_rate:
                continue  # 这笔交易不通过村集体
            
            # 选择最匹配的需求方
            demander = sorted_demanders[0]
            
            # 尝试促成交易（提高成功率）
            base_success = 0.5
            boosted_success = min(1.0, base_success + self.mediation_success_boost)
            
            if random.random() < boosted_success:
                # 交易成功，记录
                self._record_transfer(supplier, demander)
                transfers_this_step += 1
                
                # 如果需求方已满足，从列表移除
                # （简化：每个需求方每期只接受一笔）
                sorted_demanders.pop(0)
        
        self.step_transfers_facilitated = transfers_this_step
    
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
        for patch in supplier['patches']:
            if hasattr(patch, 'set_leased'):
                # 使用新的set_leased接口，标记为村集体流转
                patch.set_leased(
                    to_agent=demander['agent'],
                    contract_years=3,  # 默认3年合同
                    rental_price=adjusted_rent,
                    transfer_channel='village_collective'  # 关键：标记流转渠道
                )
            else:
                # 向后兼容：旧版LandPatch
                patch.leased_to = demander['agent']
                patch.rental_price = adjusted_rent
                if hasattr(patch, 'transfer_channel'):
                    patch.transfer_channel = 'village_collective'
        
        # 收取服务费
        service_fee = transaction_value * self.service_fee_rate
        self.service_fee_collected += service_fee
        
        # 更新经济状态
        self.economy_state.annual_service_fee += service_fee
        self.economy_state.annual_transferred_area += total_area
        self.economy_state.annual_facilitated_transfers += 1
        
        # 记录流转信息
        self.land_transfer_records.append({
            'step': self.model.step_count if hasattr(self.model, 'step_count') else 0,
            'supplier_id': supplier['farmer'].unique_id,
            'demander_id': demander['agent'].unique_id,
            'demander_type': demander['type'],
            'area': total_area,
            'rental_value': transaction_value,
            'rental_price_per_mu': adjusted_rent,
            'service_fee': service_fee,
            'strategy_used': self.current_strategy.value,
            'transfer_channel': 'village_collective'  # 记录流转渠道
        })
    
    def _check_grain_producer(self, agent) -> bool:
        """检查是否为种粮主体"""
        if hasattr(agent, 'planned_crop'):
            return str(agent.planned_crop) == 'Grain'
        if hasattr(agent, 'grain_weight'):
            return agent.grain_weight > 0.5
        return False
    
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
    
    def _update_registration(self):
        """
        更新农户和土地登记信息
        """
        if not hasattr(self.model, 'farmers'):
            return
        
        # 获取本村农户
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
