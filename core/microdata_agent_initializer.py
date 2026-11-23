"""
微观数据驱动的智能体初始化器
==========================================
从户级微观数据库Bootstrap采样初始化ABM智能体

@theory 真正的ABM应该是微观数据驱动的
- 每个智能体的特征来自真实数据的采样
- 保留数据中的相关性结构（土地-收入-生计策略的联合分布）
- 支持分层采样保持分布一致性

@usage
    from core.microdata_agent_initializer import MicrodataAgentInitializer
    
    initializer = MicrodataAgentInitializer()
    features = initializer.sample_farmer(region='东部')
    # features = {
    #     'land_area_mu': 8.5,
    #     'agri_income': 12000,
    #     'livelihood': 'Pluriactive',
    #     ...
    # }
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from core.cfps_livelihood_alignment import align_feature_library_livelihoods
from core.reproducibility import build_numpy_rng, build_python_rng

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVELIHOOD_WEIGHTS = {
    'Pure_Farmer': 0.297,
    'Pluriactive': 0.252,
    'Non_Farmer': 0.451,
}
DEFAULT_LAND_SIZE_WEIGHTS = {
    'small': 0.523,
    'medium': 0.335,
    'large': 0.117,
    'enterprise': 0.025,
}
LAND_SIZE_ALIASES = {
    'small': 'small',
    'small_lt_5mu': 'small',
    '<5': 'small',
    'medium': 'medium',
    'medium_5_15mu': 'medium',
    '5_15': 'medium',
    'large': 'large',
    'large_15_50mu': 'large',
    '15_50': 'large',
    'enterprise': 'enterprise',
    'enterprise_gt_50mu': 'enterprise',
    'gt_50': 'enterprise',
}


class MicrodataAgentInitializer:
    """
    微观数据驱动的智能体初始化器
    
    核心功能：
    1. 从户级特征库Bootstrap采样
    2. 支持分层采样（按区域、生计类型）
    3. 保持联合分布的相关性
    """
    
    def __init__(self, feature_library_path: Optional[str] = None, seed: Optional[int] = None):
        """
        Args:
            feature_library_path: 特征库JSON路径，默认为config/agent_feature_library.json
        """
        if feature_library_path is None:
            feature_library_path = PROJECT_ROOT / "config" / "agent_feature_library.json"
        else:
            feature_library_path = Path(feature_library_path)
        
        self.feature_library_path = feature_library_path
        self.library: Optional[Dict] = None
        self.farmers: List[Dict] = []
        self.seed = seed
        self.random = build_python_rng(seed, "microdata_agent_initializer_py")
        self.np_random = build_numpy_rng(seed, "microdata_agent_initializer_np")
        self._index_by_region: Dict[str, List[int]] = {}
        self._index_by_livelihood: Dict[str, List[int]] = {}
        self._index_by_region_livelihood: Dict[str, Dict[str, List[int]]] = {}
        self._index_by_land_size: Dict[str, List[int]] = {}
        
        self._load_library()

    @staticmethod
    def _land_size_bin(land_area_mu: Any) -> str:
        try:
            land_area = float(land_area_mu or 0.0)
        except (TypeError, ValueError):
            land_area = 0.0
        if land_area < 5:
            return 'small'
        if land_area < 15:
            return 'medium'
        if land_area < 50:
            return 'large'
        return 'enterprise'

    @staticmethod
    def _normalize_land_size_bin(value: Any) -> Optional[str]:
        if value is None:
            return None
        return LAND_SIZE_ALIASES.get(str(value).strip().lower())

    @staticmethod
    def _normalize_weights(
        weights: Optional[Dict[str, float]],
        *,
        aliases: Dict[str, str],
        default: Dict[str, float],
    ) -> Dict[str, float]:
        if not weights:
            return default.copy()

        normalized = {key: 0.0 for key in default}
        for raw_key, raw_value in weights.items():
            key = aliases.get(str(raw_key).strip().lower())
            if key is None:
                continue
            try:
                normalized[key] += max(0.0, float(raw_value or 0.0))
            except (TypeError, ValueError):
                continue

        total = sum(normalized.values())
        if total <= 0:
            return default.copy()
        return {key: value / total for key, value in normalized.items()}

    def _candidate_indices(
        self,
        region: Optional[str] = None,
        livelihood: Optional[str] = None,
        land_size_bin: Optional[str] = None,
    ) -> List[int]:
        if region and livelihood:
            base = self._index_by_region_livelihood.get(region, {}).get(livelihood, [])
        elif region:
            base = self._index_by_region.get(region, [])
        elif livelihood:
            base = self._index_by_livelihood.get(livelihood, [])
        else:
            base = list(range(len(self.farmers)))

        if not base:
            base = list(range(len(self.farmers)))

        land_size_bin = self._normalize_land_size_bin(land_size_bin)
        if land_size_bin is None:
            return list(base)

        filtered = [
            idx for idx in base
            if self._land_size_bin(self.farmers[idx].get('land_area_mu')) == land_size_bin
        ]
        if filtered:
            return filtered

        fallback = self._index_by_land_size.get(land_size_bin, [])
        return list(fallback) if fallback else list(base)

    @staticmethod
    def _constrain_land_area_to_bin(land_area: Any, land_size_bin: Optional[str]) -> float:
        try:
            value = max(0.1, float(land_area or 0.1))
        except (TypeError, ValueError):
            value = 0.1

        land_size_bin = MicrodataAgentInitializer._normalize_land_size_bin(land_size_bin)
        if land_size_bin == 'small':
            return min(max(value, 0.1), 4.95)
        if land_size_bin == 'medium':
            return min(max(value, 5.0), 14.95)
        if land_size_bin == 'large':
            return min(max(value, 15.0), 49.5)
        if land_size_bin == 'enterprise':
            return max(value, 50.0)
        return value
    
    def _load_library(self) -> None:
        """加载特征库并建立索引"""
        if not self.feature_library_path.exists():
            logger.warning(f"特征库不存在: {self.feature_library_path}")
            return
        
        with open(self.feature_library_path, 'r', encoding='utf-8') as f:
            self.library = json.load(f)

        alignment = align_feature_library_livelihoods(self.library, year=2010)
        if alignment.get("matched", 0):
            logger.info(
                "Aligned feature-library livelihood labels to canonical CFPS panel "
                "(matched=%s, changed=%s, missing=%s)",
                alignment.get("matched", 0),
                alignment.get("changed", 0),
                alignment.get("missing", 0),
            )

        self.farmers = self.library.get('farmers', [])
        
        # 建立索引
        for i, farmer in enumerate(self.farmers):
            region = farmer.get('region', '其他')
            livelihood = farmer.get('livelihood', 'Non_Farmer')
            
            # 按区域索引
            if region not in self._index_by_region:
                self._index_by_region[region] = []
            self._index_by_region[region].append(i)
            
            # 按生计类型索引
            if livelihood not in self._index_by_livelihood:
                self._index_by_livelihood[livelihood] = []
            self._index_by_livelihood[livelihood].append(i)
            
            # 按区域+生计类型索引
            if region not in self._index_by_region_livelihood:
                self._index_by_region_livelihood[region] = {}
            if livelihood not in self._index_by_region_livelihood[region]:
                self._index_by_region_livelihood[region][livelihood] = []
            self._index_by_region_livelihood[region][livelihood].append(i)

            land_size = self._land_size_bin(farmer.get('land_area_mu'))
            if land_size not in self._index_by_land_size:
                self._index_by_land_size[land_size] = []
            self._index_by_land_size[land_size].append(i)
        
        logger.info(f"✓ 加载特征库: {len(self.farmers)} 农户")
        logger.info(f"  区域: {list(self._index_by_region.keys())}")
        logger.info(f"  生计类型: {list(self._index_by_livelihood.keys())}")
    
    def sample_farmer(
        self,
        region: Optional[str] = None,
        livelihood: Optional[str] = None,
        scale: Optional[str] = None,
        land_size_bin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从特征库Bootstrap采样一个农户特征
        
        Args:
            region: 指定区域 (东部/中部/西部/东北)，None则随机
            livelihood: 指定生计类型 (Pure_Farmer/Pluriactive/Non_Farmer)，None则随机
            scale: 指定规模 (Smallholder/Medium_Farmer/Large_Farmer/Family_Farm)，None则随机
        
        Returns:
            农户特征字典
        """
        if not self.farmers:
            return self._generate_default_farmer()
        
        # 确定候选集
        candidates = None
        
        if region and livelihood:
            # 同时指定区域和生计类型
            if region in self._index_by_region_livelihood:
                if livelihood in self._index_by_region_livelihood[region]:
                    candidates = self._index_by_region_livelihood[region][livelihood]
        elif region:
            # 仅指定区域
            if region in self._index_by_region:
                candidates = self._index_by_region[region]
        elif livelihood:
            # 仅指定生计类型
            if livelihood in self._index_by_livelihood:
                candidates = self._index_by_livelihood[livelihood]
        
        # 如果无匹配候选，使用全部
        if candidates is None or len(candidates) == 0:
            candidates = list(range(len(self.farmers)))

        normalized_land_size_bin = self._normalize_land_size_bin(land_size_bin)
        if normalized_land_size_bin:
            land_filtered = [
                i for i in candidates
                if self._land_size_bin(self.farmers[i].get('land_area_mu')) == normalized_land_size_bin
            ]
            if not land_filtered:
                land_filtered = self._index_by_land_size.get(normalized_land_size_bin, [])
            if land_filtered:
                candidates = list(land_filtered)
        
        # 如果指定了规模，进一步筛选
        if scale:
            candidates = [i for i in candidates if self.farmers[i].get('farmer_scale') == scale]
            if not candidates:
                candidates = list(range(len(self.farmers)))
        
        # Bootstrap采样
        idx = self.random.choice(candidates)
        farmer = self.farmers[idx].copy()
        
        # 添加一些随机扰动（保持异质性但不完全相同）
        farmer = self._add_perturbation(farmer, land_size_bin=land_size_bin)
        
        return farmer
    
    def _add_perturbation(
        self,
        farmer: Dict,
        noise_level: float = 0.1,
        land_size_bin: Optional[str] = None,
    ) -> Dict:
        """
        对农户特征添加微小随机扰动
        
        保持数据分布特征的同时增加多样性
        """
        result = farmer.copy()
        
        # 连续变量添加扰动
        for key in ['land_area_mu', 'agri_income', 'total_income']:
            if key in result and result[key] is not None and result[key] > 0:
                noise = self.np_random.normal(1.0, noise_level)
                result[key] = max(0.1, result[key] * noise)
        
        # 家庭规模添加整数扰动
        if 'family_size' in result:
            noise = self.random.choice([-1, 0, 0, 0, 1])
            result['family_size'] = max(1, result['family_size'] + noise)

        if land_size_bin is not None and 'land_area_mu' in result:
            result['land_area_mu'] = self._constrain_land_area_to_bin(
                result.get('land_area_mu'),
                land_size_bin,
            )
        
        return result
    
    def _generate_default_farmer(self) -> Dict[str, Any]:
        """生成默认农户特征（当特征库不可用时）"""
        weights = DEFAULT_LIVELIHOOD_WEIGHTS
        livelihood = self.random.choices(
            ['Pure_Farmer', 'Pluriactive', 'Non_Farmer'],
            weights=[weights['Pure_Farmer'], weights['Pluriactive'], weights['Non_Farmer']]
        )[0]
        
        return {
            'region': self.random.choice(['东部', '中部', '西部', '东北']),
            'land_area_mu': self.np_random.lognormal(1.5, 1.0),
            'agri_income': self.np_random.lognormal(8, 1) if livelihood != 'Non_Farmer' else 0,
            'total_income': self.np_random.lognormal(10, 0.5),
            'agri_ratio': 0.6 if livelihood == 'Pure_Farmer' else (0.3 if livelihood == 'Pluriactive' else 0.1),
            'livelihood': livelihood,
            'farmer_scale': 'Smallholder',
            'family_size': self.random.randint(2, 6),
            'is_rent_out': 0,
            'is_rent_in': 0,
        }
    
    def sample_farmers_batch(
        self,
        n: int,
        region_weights: Optional[Dict[str, float]] = None,
        livelihood_weights: Optional[Dict[str, float]] = None,
        land_size_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量采样农户，支持分层采样
        
        Args:
            n: 采样数量
            region_weights: 区域权重，如 {'东部': 0.28, '中部': 0.27, ...}
            livelihood_weights: 生计类型权重
        
        Returns:
            农户特征列表
        """
        farmers = []
        
        # 默认使用数据中的分布
        if region_weights is None:
            region_weights = {
                region: len(indices) / len(self.farmers)
                for region, indices in self._index_by_region.items()
            }
        
        if livelihood_weights is None and self.library:
            stats = self.library.get('statistics', {})
            livelihood_weights = stats.get('livelihood_distribution', DEFAULT_LIVELIHOOD_WEIGHTS)
        if livelihood_weights is None:
            livelihood_weights = DEFAULT_LIVELIHOOD_WEIGHTS.copy()
        else:
            livelihood_weights = {
                key: max(0.0, float(livelihood_weights.get(key, 0.0) or 0.0))
                for key in ['Pure_Farmer', 'Pluriactive', 'Non_Farmer']
            }
            total = sum(livelihood_weights.values())
            if total <= 0:
                livelihood_weights = DEFAULT_LIVELIHOOD_WEIGHTS.copy()
            else:
                livelihood_weights = {
                    key: value / total for key, value in livelihood_weights.items()
                }
        
        # 分层采样
        regions = list(region_weights.keys())
        region_probs = list(region_weights.values())
        
        livelihoods = list(livelihood_weights.keys())
        livelihood_probs = list(livelihood_weights.values())

        normalized_land_size_weights = None
        if land_size_weights:
            normalized_land_size_weights = self._normalize_weights(
                land_size_weights,
                aliases=LAND_SIZE_ALIASES,
                default=DEFAULT_LAND_SIZE_WEIGHTS,
            )
        land_size_bins = list(normalized_land_size_weights.keys()) if normalized_land_size_weights else []
        land_size_probs = list(normalized_land_size_weights.values()) if normalized_land_size_weights else []
        
        for _ in range(n):
            region = self.random.choices(regions, weights=region_probs)[0]
            livelihood = self.random.choices(livelihoods, weights=livelihood_probs)[0]
            land_size_bin = (
                self.random.choices(land_size_bins, weights=land_size_probs)[0]
                if normalized_land_size_weights
                else None
            )
            farmer = self.sample_farmer(
                region=region,
                livelihood=livelihood,
                land_size_bin=land_size_bin,
            )
            farmers.append(farmer)
        
        return farmers
    
    def get_statistics(self) -> Dict[str, Any]:
        """返回特征库统计信息"""
        if self.library:
            return self.library.get('statistics', {})
        return {}
    
    def validate_distribution(self, sampled_farmers: List[Dict]) -> Dict[str, Any]:
        """
        验证采样结果的分布与原始数据一致性
        
        Args:
            sampled_farmers: 采样的农户列表
        
        Returns:
            对比结果
        """
        if not sampled_farmers or not self.library:
            return {}
        
        # 计算采样分布
        sampled_livelihood = {}
        sampled_region = {}
        sampled_land = []
        
        for f in sampled_farmers:
            lv = f.get('livelihood', 'Unknown')
            sampled_livelihood[lv] = sampled_livelihood.get(lv, 0) + 1
            
            rg = f.get('region', 'Unknown')
            sampled_region[rg] = sampled_region.get(rg, 0) + 1
            
            if f.get('land_area_mu') is not None:
                sampled_land.append(f['land_area_mu'])
        
        n = len(sampled_farmers)
        sampled_livelihood = {k: v/n for k, v in sampled_livelihood.items()}
        sampled_region = {k: v/n for k, v in sampled_region.items()}
        
        # 原始分布
        original_stats = self.library.get('statistics', {})
        original_livelihood = original_stats.get('livelihood_distribution', {})
        original_region = original_stats.get('region_distribution', {})
        original_land = original_stats.get('land_area', {})
        
        return {
            'livelihood': {
                'sampled': sampled_livelihood,
                'original': original_livelihood,
            },
            'region': {
                'sampled': sampled_region,
                'original': original_region,
            },
            'land_area': {
                'sampled_mean': np.mean(sampled_land) if sampled_land else 0,
                'original_mean': original_land.get('mean', 0),
            }
        }


class VillageCollectiveInitializer:
    """
    村集体智能体初始化器
    
    基于CFPS村级数据或聚合农户数据
    """
    
    def __init__(self):
        self.village_features = []
    
    def aggregate_from_farmers(self, farmers: List[Dict], village_size: int = 50) -> List[Dict]:
        """
        从农户特征聚合生成村集体特征
        
        Args:
            farmers: 农户特征列表
            village_size: 每村农户数
        
        Returns:
            村集体特征列表
        """
        villages = []
        n_villages = max(1, len(farmers) // village_size)
        
        for i in range(n_villages):
            start = i * village_size
            end = min(start + village_size, len(farmers))
            village_farmers = farmers[start:end]
            
            if not village_farmers:
                continue
            
            # 聚合村级特征
            village = {
                'village_id': i,
                'n_households': len(village_farmers),
                'total_land_mu': sum(f.get('land_area_mu', 0) for f in village_farmers),
                'avg_land_mu': np.mean([f.get('land_area_mu', 0) for f in village_farmers]),
                'avg_income': np.mean([f.get('total_income', 0) for f in village_farmers]),
                'pure_farmer_ratio': sum(1 for f in village_farmers if f.get('livelihood') == 'Pure_Farmer') / len(village_farmers),
                'transfer_rate': sum(1 for f in village_farmers if f.get('is_rent_out', 0) or f.get('is_rent_in', 0)) / len(village_farmers),
                'region': village_farmers[0].get('region', '其他'),  # 取第一个农户的区域
            }
            villages.append(village)
        
        return villages


class EnterpriseInitializer:
    """
    农业企业/合作社智能体初始化器
    
    基于大户特征或独立配置
    """
    
    def __init__(self, enterprise_ratio: float = 0.05, seed: Optional[int] = None):
        """
        Args:
            enterprise_ratio: 企业占农户比例（默认5%）
        """
        self.enterprise_ratio = enterprise_ratio
        self.random = build_python_rng(seed, "enterprise_initializer_py")
        self.np_random = build_numpy_rng(seed, "enterprise_initializer_np")
    
    def identify_from_farmers(self, farmers: List[Dict], land_threshold: float = 50.0) -> List[Dict]:
        """
        从农户中识别潜在企业（规模大户）
        
        Args:
            farmers: 农户特征列表
            land_threshold: 土地面积阈值（亩）
        
        Returns:
            企业特征列表
        """
        enterprises = []
        
        for f in farmers:
            land = f.get('land_area_mu', 0)
            if land >= land_threshold:
                enterprise = {
                    'enterprise_id': f.get('id', self.random.randint(10000, 99999)),
                    'land_area_mu': land,
                    'region': f.get('region', '其他'),
                    'capital': f.get('total_income', 100000) * 2,  # 假设资本是收入的2倍
                    'type': 'Family_Farm' if land < 100 else 'Cooperative',
                }
                enterprises.append(enterprise)
        
        return enterprises
    
    def generate_enterprises(self, n: int, regions: List[str] = None) -> List[Dict]:
        """
        生成企业特征
        
        Args:
            n: 企业数量
            regions: 区域列表
        """
        if regions is None:
            regions = ['东部', '中部', '西部', '东北']
        
        enterprises = []
        for i in range(n):
            enterprise = {
                'enterprise_id': i,
                'land_area_mu': self.np_random.lognormal(4, 0.8),  # 50-500亩
                'region': self.random.choice(regions),
                'capital': self.np_random.lognormal(12, 0.5),  # 10万-100万
                'type': self.random.choice(['Family_Farm', 'Cooperative', 'Company']),
            }
            enterprises.append(enterprise)
        
        return enterprises


# 便捷函数
def load_agent_initializer() -> MicrodataAgentInitializer:
    """加载默认初始化器"""
    return MicrodataAgentInitializer()


def sample_farmer_features(
    n: int = 1,
    region: str = None,
    livelihood: str = None
) -> List[Dict]:
    """
    便捷函数：采样农户特征
    
    Args:
        n: 采样数量
        region: 区域
        livelihood: 生计类型
    """
    initializer = MicrodataAgentInitializer()
    
    if n == 1:
        return [initializer.sample_farmer(region=region, livelihood=livelihood)]
    else:
        return [
            initializer.sample_farmer(region=region, livelihood=livelihood)
            for _ in range(n)
        ]


if __name__ == "__main__":
    # 测试
    print("=" * 70)
    print("测试微观数据驱动的智能体初始化器")
    print("=" * 70)
    
    # 加载初始化器
    init = MicrodataAgentInitializer()
    
    # 采样单个农户
    print("\n单个农户采样:")
    farmer = init.sample_farmer(region='东部', livelihood='Pure_Farmer')
    for k, v in farmer.items():
        print(f"  {k}: {v}")
    
    # 批量采样
    print("\n批量采样 100 农户:")
    farmers = init.sample_farmers_batch(100)
    
    # 验证分布
    validation = init.validate_distribution(farmers)
    print("\n分布验证:")
    print(f"  生计类型 - 采样: {validation['livelihood']['sampled']}")
    print(f"  生计类型 - 原始: {validation['livelihood']['original']}")
    print(f"  土地面积 - 采样均值: {validation['land_area']['sampled_mean']:.1f}")
    print(f"  土地面积 - 原始均值: {validation['land_area']['original_mean']:.1f}")
