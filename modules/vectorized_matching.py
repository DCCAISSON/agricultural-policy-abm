# -*- coding: utf-8 -*-
"""
@module vectorized_matching.py
@description 向量化土地匹配算法 - 使用numpy/pandas优化O(N²)匹配
@created 2025-12-04

性能优化目标：
1. 原始双层循环匹配：O(N×M) 其中N=供给数，M=需求数
2. 向量化后：使用矩阵运算批量计算距离/价格匹配
3. 预期提升：10-100倍性能提升

核心算法：
1. 构建供需矩阵
2. 向量化计算匹配得分
3. 贪心匹配或匈牙利算法
4. 批量更新状态
"""

import numpy as np
import pandas as pd
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from core.reproducibility import build_python_rng


def _stable_numeric_id(value: Any, modulus: int = 10**9) -> int:
    """Create a process-stable numeric encoding for agent ids."""
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


@dataclass
class SupplyOffer:
    """供给报价"""
    patch_id: int
    supplier_id: str
    area: float
    min_price: float          # 保留价
    quality: float            # 土地质量
    position: Tuple[int, int] # 位置


@dataclass
class DemandOffer:
    """需求报价"""
    demander_id: str
    max_price: float          # 最高愿付价
    desired_area: float       # 期望面积
    position: Tuple[int, int] # 位置
    priority: float = 1.0     # 优先级（企业>大户>小户）


@dataclass
class MatchResult:
    """匹配结果"""
    patch_id: int
    supplier_id: str
    demander_id: str
    matched_area: float
    matched_price: float
    surplus: float            # 社会福利剩余


class VectorizedLandMatcher:
    """
    向量化土地匹配器
    
    使用numpy矩阵运算替代双层循环
    """
    
    def __init__(self, 
                 max_distance: float = 15.0,
                 price_weight: float = 0.4,
                 distance_weight: float = 0.3,
                 quality_weight: float = 0.3,
                 use_local_matching: bool = True):
        """
        初始化匹配器
        
        Args:
            max_distance: 最大匹配距离（熟人网络半径）
            price_weight: 价格匹配权重
            distance_weight: 距离权重
            quality_weight: 质量权重
            use_local_matching: 是否优先本地匹配
        """
        self.max_distance = max_distance
        self.price_weight = price_weight
        self.distance_weight = distance_weight
        self.quality_weight = quality_weight
        self.use_local_matching = use_local_matching
        
        # 性能统计
        self.last_match_time = 0
        self.last_matrix_size = (0, 0)
    
    def match(self, 
              supply_offers: List[SupplyOffer],
              demand_offers: List[DemandOffer]) -> List[MatchResult]:
        """
        执行向量化匹配
        
        Args:
            supply_offers: 供给报价列表
            demand_offers: 需求报价列表
            
        Returns:
            匹配结果列表
        """
        start_time = time.time()
        
        if not supply_offers or not demand_offers:
            return []
        
        n_supply = len(supply_offers)
        n_demand = len(demand_offers)
        self.last_matrix_size = (n_supply, n_demand)
        
        # 1. 构建数据矩阵
        supply_matrix = self._build_supply_matrix(supply_offers)
        demand_matrix = self._build_demand_matrix(demand_offers)
        
        # 2. 计算距离矩阵 (N_supply × N_demand)
        distance_matrix = self._compute_distance_matrix(supply_offers, demand_offers)
        
        # 3. 计算价格匹配矩阵
        price_match_matrix = self._compute_price_match_matrix(supply_offers, demand_offers)
        
        # 4. 计算综合得分矩阵
        score_matrix = self._compute_score_matrix(
            distance_matrix, 
            price_match_matrix,
            supply_matrix, 
            demand_matrix
        )
        
        # 5. 应用约束（距离过远、价格不匹配）
        valid_mask = self._compute_valid_mask(distance_matrix, price_match_matrix)
        score_matrix = np.where(valid_mask, score_matrix, -np.inf)
        
        # 6. 贪心匹配
        results = self._greedy_match(
            score_matrix, 
            supply_offers, 
            demand_offers,
            price_match_matrix
        )
        
        self.last_match_time = time.time() - start_time
        
        return results
    
    def _build_supply_matrix(self, offers: List[SupplyOffer]) -> np.ndarray:
        """构建供给矩阵 [n, 5]: patch_id, area, min_price, quality, supplier_hash"""
        n = len(offers)
        matrix = np.zeros((n, 5))
        
        for i, offer in enumerate(offers):
            matrix[i, 0] = offer.patch_id
            matrix[i, 1] = offer.area
            matrix[i, 2] = offer.min_price
            matrix[i, 3] = offer.quality
            matrix[i, 4] = _stable_numeric_id(offer.supplier_id)
        
        return matrix
    
    def _build_demand_matrix(self, offers: List[DemandOffer]) -> np.ndarray:
        """构建需求矩阵 [m, 4]: max_price, desired_area, priority, demander_hash"""
        m = len(offers)
        matrix = np.zeros((m, 4))
        
        for i, offer in enumerate(offers):
            matrix[i, 0] = offer.max_price
            matrix[i, 1] = offer.desired_area
            matrix[i, 2] = offer.priority
            matrix[i, 3] = _stable_numeric_id(offer.demander_id)
        
        return matrix
    
    def _compute_distance_matrix(self, 
                                  supply_offers: List[SupplyOffer],
                                  demand_offers: List[DemandOffer]) -> np.ndarray:
        """
        向量化计算距离矩阵
        
        使用广播机制避免双层循环
        """
        n = len(supply_offers)
        m = len(demand_offers)
        
        # 提取位置坐标
        supply_pos = np.array([o.position for o in supply_offers])  # (n, 2)
        demand_pos = np.array([o.position for o in demand_offers])  # (m, 2)
        
        # 广播计算欧氏距离
        # supply_pos[:, np.newaxis, :] -> (n, 1, 2)
        # demand_pos[np.newaxis, :, :] -> (1, m, 2)
        # 差值 -> (n, m, 2)
        diff = supply_pos[:, np.newaxis, :] - demand_pos[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, m)
        
        return distance_matrix
    
    def _compute_price_match_matrix(self,
                                     supply_offers: List[SupplyOffer],
                                     demand_offers: List[DemandOffer]) -> np.ndarray:
        """
        向量化计算价格匹配矩阵
        
        价格匹配度 = 需求价 - 供给价（剩余越大匹配度越高）
        """
        # 提取价格
        supply_prices = np.array([o.min_price / max(o.area, 1) for o in supply_offers])  # 每亩价格
        demand_prices = np.array([o.max_price for o in demand_offers])  # 最高愿付每亩价格
        
        # 广播计算价格差
        # (n,) -> (n, 1)
        # (m,) -> (1, m)
        price_diff = demand_prices[np.newaxis, :] - supply_prices[:, np.newaxis]  # (n, m)
        
        return price_diff
    
    def _compute_score_matrix(self,
                               distance_matrix: np.ndarray,
                               price_match_matrix: np.ndarray,
                               supply_matrix: np.ndarray,
                               demand_matrix: np.ndarray) -> np.ndarray:
        """
        计算综合得分矩阵
        
        得分 = w1 * 价格匹配度 + w2 * (1 - 距离/max_dist) + w3 * 土地质量
        """
        n, m = distance_matrix.shape
        
        # 1. 价格得分（归一化）
        price_scores = price_match_matrix.copy()
        price_max = np.max(np.abs(price_scores)) + 1e-6
        price_scores = price_scores / price_max
        
        # 2. 距离得分（距离越近越好）
        distance_scores = 1 - distance_matrix / self.max_distance
        distance_scores = np.clip(distance_scores, 0, 1)
        
        # 3. 质量得分（广播）
        quality_scores = supply_matrix[:, 3][:, np.newaxis]  # (n, 1)
        quality_scores = np.broadcast_to(quality_scores, (n, m))
        
        # 4. 需求方优先级加权
        priority_weights = demand_matrix[:, 2][np.newaxis, :]  # (1, m)
        priority_weights = np.broadcast_to(priority_weights, (n, m))
        
        # 5. 综合得分
        scores = (
            self.price_weight * price_scores +
            self.distance_weight * distance_scores +
            self.quality_weight * quality_scores
        ) * priority_weights
        
        return scores
    
    def _compute_valid_mask(self,
                            distance_matrix: np.ndarray,
                            price_match_matrix: np.ndarray) -> np.ndarray:
        """
        计算有效匹配掩码
        
        条件：
        1. 距离在允许范围内
        2. 需求价格 >= 供给价格
        """
        distance_valid = distance_matrix <= self.max_distance
        price_valid = price_match_matrix >= 0
        
        return distance_valid & price_valid
    
    def _greedy_match(self,
                      score_matrix: np.ndarray,
                      supply_offers: List[SupplyOffer],
                      demand_offers: List[DemandOffer],
                      price_match_matrix: np.ndarray) -> List[MatchResult]:
        """
        贪心匹配算法
        
        按得分从高到低匹配，已匹配的供需双方不再参与
        """
        results = []
        
        # 复制得分矩阵用于修改
        scores = score_matrix.copy()
        
        # 记录已匹配的供给和需求
        matched_supply = set()
        matched_demand = set()
        
        # 追踪需求方剩余需求面积
        remaining_demand = {i: offer.desired_area for i, offer in enumerate(demand_offers)}
        
        while True:
            # 找到最高得分的匹配
            max_idx = np.argmax(scores)
            max_score = scores.flat[max_idx]
            
            if max_score == -np.inf:
                break  # 没有有效匹配了
            
            # 转换为二维索引
            supply_idx, demand_idx = np.unravel_index(max_idx, scores.shape)
            
            # 检查是否已匹配
            if supply_idx in matched_supply or demand_idx in matched_demand:
                scores[supply_idx, demand_idx] = -np.inf
                continue
            
            # 获取报价信息
            supply_offer = supply_offers[supply_idx]
            demand_offer = demand_offers[demand_idx]
            
            # 计算成交价格（取中间价）
            supply_price_per_mu = supply_offer.min_price / max(supply_offer.area, 1)
            matched_price = (supply_price_per_mu + demand_offer.max_price) / 2
            
            # 计算成交面积
            matched_area = min(supply_offer.area, remaining_demand[demand_idx])
            
            # 计算剩余
            surplus = (demand_offer.max_price - supply_price_per_mu) * matched_area
            
            # 创建匹配结果
            result = MatchResult(
                patch_id=supply_offer.patch_id,
                supplier_id=supply_offer.supplier_id,
                demander_id=demand_offer.demander_id,
                matched_area=matched_area,
                matched_price=matched_price * matched_area,
                surplus=surplus
            )
            results.append(result)
            
            # 更新状态
            matched_supply.add(supply_idx)
            remaining_demand[demand_idx] -= matched_area
            
            if remaining_demand[demand_idx] <= 0:
                matched_demand.add(demand_idx)
                # 将该需求方的所有匹配设为无效
                scores[:, demand_idx] = -np.inf
            
            # 将该供给方的所有匹配设为无效
            scores[supply_idx, :] = -np.inf
        
        return results


class VectorizedServiceMatcher:
    """
    向量化服务匹配器
    
    用于农业社会化服务市场的匹配
    """
    
    def __init__(self, max_distance: float = 15.0):
        """初始化"""
        self.max_distance = max_distance
    
    def match_services(self,
                        farmers: List[Dict],  # 农户需求信息
                        providers: List[Dict]  # 服务商供给信息
                       ) -> List[Dict]:
        """
        向量化服务匹配
        
        Args:
            farmers: 农户列表，每个包含 {id, position, area, max_price, service_type}
            providers: 服务商列表，每个包含 {id, position, capacity, price, efficiency}
            
        Returns:
            匹配结果列表
        """
        if not farmers or not providers:
            return []
        
        n_farmers = len(farmers)
        n_providers = len(providers)
        
        # 构建位置矩阵
        farmer_pos = np.array([f.get('position', (0, 0)) for f in farmers])
        provider_pos = np.array([p.get('position', (0, 0)) for p in providers])
        
        # 向量化距离计算
        diff = farmer_pos[:, np.newaxis, :] - provider_pos[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # 价格匹配
        farmer_max_price = np.array([f.get('max_price', 500) for f in farmers])
        provider_price = np.array([p.get('price', 400) for p in providers])
        
        price_valid = farmer_max_price[:, np.newaxis] >= provider_price[np.newaxis, :]
        distance_valid = distances <= self.max_distance
        
        # 综合有效性
        valid_mask = price_valid & distance_valid
        
        # 效率得分
        efficiency = np.array([p.get('efficiency', 1.0) for p in providers])
        efficiency_scores = np.broadcast_to(efficiency[np.newaxis, :], (n_farmers, n_providers))
        
        # 综合得分
        scores = np.where(valid_mask, efficiency_scores, -np.inf)
        
        # 贪心匹配
        results = []
        matched_farmers = set()
        provider_remaining_capacity = {i: p.get('capacity', 500) for i, p in enumerate(providers)}
        
        while True:
            max_idx = np.argmax(scores)
            max_score = scores.flat[max_idx]
            
            if max_score == -np.inf:
                break
            
            farmer_idx, provider_idx = np.unravel_index(max_idx, scores.shape)
            
            if farmer_idx in matched_farmers:
                scores[farmer_idx, :] = -np.inf
                continue
            
            farmer = farmers[farmer_idx]
            provider = providers[provider_idx]
            
            required_area = farmer.get('area', 10)
            available_capacity = provider_remaining_capacity[provider_idx]
            
            if available_capacity < required_area:
                scores[farmer_idx, provider_idx] = -np.inf
                continue
            
            # 创建匹配
            result = {
                'farmer_id': farmer.get('id'),
                'provider_id': provider.get('id'),
                'area': required_area,
                'price': provider.get('price', 400) * required_area,
                'service_type': farmer.get('service_type', 'full'),
            }
            results.append(result)
            
            matched_farmers.add(farmer_idx)
            provider_remaining_capacity[provider_idx] -= required_area
            scores[farmer_idx, :] = -np.inf
        
        return results


def benchmark_matching(n_supply: int = 100, n_demand: int = 50, iterations: int = 10):
    """
    性能基准测试
    
    比较向量化匹配和传统双循环匹配的性能
    """
    rng = build_python_rng(42, "vectorized_matching_benchmark")
    
    # 生成测试数据
    supply_offers = [
        SupplyOffer(
            patch_id=i,
            supplier_id=f"farmer_{i}",
            area=rng.uniform(5, 50),
            min_price=rng.uniform(200, 400),
            quality=rng.uniform(0.5, 1.0),
            position=(rng.randint(0, 50), rng.randint(0, 50))
        )
        for i in range(n_supply)
    ]
    
    demand_offers = [
        DemandOffer(
            demander_id=f"enterprise_{i}",
            max_price=rng.uniform(300, 600),
            desired_area=rng.uniform(50, 200),
            position=(rng.randint(0, 50), rng.randint(0, 50)),
            priority=rng.uniform(0.5, 2.0)
        )
        for i in range(n_demand)
    ]
    
    # 向量化匹配
    matcher = VectorizedLandMatcher()
    
    times = []
    for _ in range(iterations):
        start = time.time()
        results = matcher.match(supply_offers, demand_offers)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"向量化匹配性能测试:")
    print(f"  供给数: {n_supply}, 需求数: {n_demand}")
    print(f"  迭代次数: {iterations}")
    print(f"  平均时间: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
    print(f"  匹配成功数: {len(results)}")
    
    return avg_time, len(results)


if __name__ == "__main__":
    # 运行基准测试
    benchmark_matching(100, 50, 10)
    benchmark_matching(500, 200, 5)
    benchmark_matching(1000, 400, 3)
