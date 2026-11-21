# -*- coding: utf-8 -*-
"""
空间差异化模块
==============

实现中国不同区域的农业异质性
根据地理位置、气候条件、经济发展水平调整模型参数

@new 2025-12-18: 与中国实际结合的空间异质性
理论依据:
1. 自然禀赋差异: 东部平原vs西部山区
2. 经济发展差异: 发达地区vs欠发达地区
3. 政策倾斜差异: 主产区vs非主产区

区域划分:
- 华北平原: 河北、河南、山东
- 东北平原: 黑龙江、吉林、辽宁
- 长江中下游: 江苏、安徽、湖北、湖南
- 西南山区: 四川、云南、贵州
- 西北干旱区: 陕西、甘肃、新疆
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


class Region(Enum):
    """中国农业主要区域"""
    NORTH_CHINA_PLAIN = "华北平原"  # 小麦玉米主产区
    NORTHEAST_PLAIN = "东北平原"    # 水稻玉米大豆主产区
    YANGTZE_RIVER = "长江中下游"    # 水稻油菜主产区
    SOUTHWEST_MOUNTAIN = "西南山区"  # 地形复杂，小规模农业
    NORTHWEST_ARID = "西北干旱区"    # 灌溉农业，特色作物


@dataclass
class RegionalParameters:
    """区域差异化参数"""
    
    # 基础属性
    region: Region
    province: str
    description: str
    
    # 自然条件
    avg_plot_size: float  # 平均地块大小(亩)
    fragmentation_index: float  # 细碎化程度(0-1,越高越细碎)
    soil_quality: float  # 土壤质量(0-1)
    water_availability: float  # 水资源可得性(0-1)
    
    # 生产参数
    grain_yield_multiplier: float  # 单产倍数(相对全国平均)
    mechanization_potential: float  # 机械化潜力(0-1)
    production_cost_multiplier: float  # 生产成本倍数
    
    # 经济参数
    land_price_multiplier: float  # 地价倍数
    wage_level_multiplier: float  # 工资水平倍数
    non_farm_opportunity: float  # 非农就业机会(0-1)
    
    # 政策参数
    subsidy_multiplier: float  # 补贴倍数(主产区更高)
    policy_priority: float  # 政策优先级(0-1)
    infrastructure_level: float  # 基础设施水平(0-1)
    
    # 社会网络
    network_density: float  # 社会网络密度(0-1)
    information_diffusion_speed: float  # 信息传播速度(0-1)


# 区域参数库(基于实际数据)
REGIONAL_PARAMS: Dict[Region, RegionalParameters] = {
    
    Region.NORTH_CHINA_PLAIN: RegionalParameters(
        region=Region.NORTH_CHINA_PLAIN,
        province="河北/河南/山东",
        description="小麦玉米主产区，平原为主，机械化程度高",
        # 自然条件
        avg_plot_size=5.0,  # 较大
        fragmentation_index=0.4,  # 较低
        soil_quality=0.75,
        water_availability=0.6,  # 依赖灌溉
        # 生产参数
        grain_yield_multiplier=1.1,  # 高于全国平均
        mechanization_potential=0.9,  # 平原，机械化条件好
        production_cost_multiplier=0.95,
        # 经济参数
        land_price_multiplier=1.2,  # 人口密集，地价高
        wage_level_multiplier=1.1,
        non_farm_opportunity=0.7,  # 城市化程度高
        # 政策参数
        subsidy_multiplier=1.15,  # 粮食主产区，政策倾斜
        policy_priority=0.9,
        infrastructure_level=0.8,
        # 社会网络
        network_density=0.7,
        information_diffusion_speed=0.8,
    ),
    
    Region.NORTHEAST_PLAIN: RegionalParameters(
        region=Region.NORTHEAST_PLAIN,
        province="黑龙江/吉林/辽宁",
        description="水稻玉米大豆主产区，地广人稀，规模化程度最高",
        # 自然条件
        avg_plot_size=10.0,  # 最大
        fragmentation_index=0.2,  # 最低
        soil_quality=0.85,  # 黑土地
        water_availability=0.7,
        # 生产参数
        grain_yield_multiplier=1.2,  # 最高
        mechanization_potential=0.95,  # 最适合机械化
        production_cost_multiplier=0.9,  # 规模效应
        # 经济参数
        land_price_multiplier=0.8,  # 人口密度低
        wage_level_multiplier=0.9,
        non_farm_opportunity=0.5,  # 非农就业少
        # 政策参数
        subsidy_multiplier=1.2,  # 最高补贴
        policy_priority=1.0,  # 最高优先级
        infrastructure_level=0.75,
        # 社会网络
        network_density=0.5,  # 人口稀疏
        information_diffusion_speed=0.7,
    ),
    
    Region.YANGTZE_RIVER: RegionalParameters(
        region=Region.YANGTZE_RIVER,
        province="江苏/安徽/湖北/湖南",
        description="水稻油菜主产区，水网密布，经济发达",
        # 自然条件
        avg_plot_size=3.5,
        fragmentation_index=0.5,
        soil_quality=0.8,
        water_availability=0.9,  # 水资源丰富
        # 生产参数
        grain_yield_multiplier=1.15,
        mechanization_potential=0.7,  # 水田机械化难度大
        production_cost_multiplier=1.05,
        # 经济参数
        land_price_multiplier=1.5,  # 经济发达，地价高
        wage_level_multiplier=1.3,  # 工资水平高
        non_farm_opportunity=0.85,  # 非农就业机会多
        # 政策参数
        subsidy_multiplier=1.1,
        policy_priority=0.85,
        infrastructure_level=0.9,  # 基础设施最好
        # 社会网络
        network_density=0.8,
        information_diffusion_speed=0.9,
    ),
    
    Region.SOUTHWEST_MOUNTAIN: RegionalParameters(
        region=Region.SOUTHWEST_MOUNTAIN,
        province="四川/云南/贵州",
        description="西南山区，地形复杂，小规模传统农业",
        # 自然条件
        avg_plot_size=2.0,  # 最小
        fragmentation_index=0.8,  # 最高(山地梯田)
        soil_quality=0.6,
        water_availability=0.7,
        # 生产参数
        grain_yield_multiplier=0.85,  # 较低
        mechanization_potential=0.3,  # 山地不适合机械化
        production_cost_multiplier=1.2,  # 人工成本高
        # 经济参数
        land_price_multiplier=0.6,
        wage_level_multiplier=0.7,
        non_farm_opportunity=0.4,  # 经济欠发达
        # 政策参数
        subsidy_multiplier=1.0,  # 标准补贴
        policy_priority=0.6,
        infrastructure_level=0.5,  # 基础设施差
        # 社会网络
        network_density=0.9,  # 熟人社会
        information_diffusion_speed=0.5,  # 信息闭塞
    ),
    
    Region.NORTHWEST_ARID: RegionalParameters(
        region=Region.NORTHWEST_ARID,
        province="陕西/甘肃/新疆",
        description="西北干旱区，灌溉农业，特色作物",
        # 自然条件
        avg_plot_size=6.0,
        fragmentation_index=0.35,
        soil_quality=0.5,
        water_availability=0.3,  # 严重缺水
        # 生产参数
        grain_yield_multiplier=0.75,  # 最低(干旱限制)
        mechanization_potential=0.8,  # 地形平坦
        production_cost_multiplier=1.3,  # 灌溉成本高
        # 经济参数
        land_price_multiplier=0.5,  # 最低
        wage_level_multiplier=0.65,
        non_farm_opportunity=0.35,
        # 政策参数
        subsidy_multiplier=0.9,  # 略低于平均
        policy_priority=0.5,
        infrastructure_level=0.4,
        # 社会网络
        network_density=0.6,
        information_diffusion_speed=0.6,
    ),
}


class SpatialHeterogeneityManager:
    """空间异质性管理器"""
    
    def __init__(self, region: Region = Region.NORTH_CHINA_PLAIN):
        """
        初始化空间异质性管理器
        
        Args:
            region: 模拟的区域
        """
        self.region = region
        self.params = REGIONAL_PARAMS[region]
    
    def adjust_farmer_params(self, base_params: Dict) -> Dict:
        """
        调整农户参数
        
        Args:
            base_params: 基础参数
        
        Returns:
            调整后的参数
        """
        adjusted = base_params.copy()
        
        # 调整土地规模
        if 'initial_land_area' in adjusted:
            adjusted['initial_land_area'] *= self.params.avg_plot_size / 3.5  # 相对全国平均
        
        # 调整非农工资
        if 'off_farm_wage' in adjusted:
            adjusted['off_farm_wage'] *= self.params.wage_level_multiplier
        
        # 调整社会网络
        if 'network_density' in adjusted:
            adjusted['network_density'] = self.params.network_density
        
        return adjusted
    
    def adjust_production_params(self, base_params: Dict) -> Dict:
        """
        调整生产参数
        
        Args:
            base_params: 基础参数
        
        Returns:
            调整后的参数
        """
        adjusted = base_params.copy()
        
        # 调整单产
        if 'grain_yield' in adjusted:
            adjusted['grain_yield'] *= self.params.grain_yield_multiplier
        
        # 调整生产成本
        if 'production_cost' in adjusted:
            adjusted['production_cost'] *= self.params.production_cost_multiplier
        
        # 调整机械化水平
        if 'mechanization_level' in adjusted:
            adjusted['mechanization_level'] = min(
                adjusted.get('mechanization_level', 0.5),
                self.params.mechanization_potential
            )
        
        return adjusted
    
    def adjust_policy_params(self, base_params: Dict) -> Dict:
        """
        调整政策参数
        
        Args:
            base_params: 基础参数
        
        Returns:
            调整后的参数
        """
        adjusted = base_params.copy()
        
        # 调整补贴水平
        if 'grain_subsidy_rate' in adjusted:
            adjusted['grain_subsidy_rate'] *= self.params.subsidy_multiplier
        
        # 调整基础设施
        if 'infrastructure_quality' in adjusted:
            adjusted['infrastructure_quality'] = self.params.infrastructure_level
        
        return adjusted
    
    def adjust_land_price(self, base_price: float) -> float:
        """
        调整土地价格
        
        Args:
            base_price: 基础价格
        
        Returns:
            调整后的价格
        """
        return base_price * self.params.land_price_multiplier
    
    def get_fragmentation_level(self) -> float:
        """获取区域细碎化程度"""
        return self.params.fragmentation_index
    
    def get_region_summary(self) -> Dict:
        """
        获取区域参数摘要
        
        Returns:
            区域参数字典
        """
        return {
            '区域': self.params.region.value,
            '省份': self.params.province,
            '描述': self.params.description,
            '平均地块大小': f"{self.params.avg_plot_size}亩",
            '细碎化程度': f"{self.params.fragmentation_index:.2f}",
            '单产倍数': f"{self.params.grain_yield_multiplier:.2f}",
            '补贴倍数': f"{self.params.subsidy_multiplier:.2f}",
            '工资倍数': f"{self.params.wage_level_multiplier:.2f}",
            '非农机会': f"{self.params.non_farm_opportunity:.2f}",
        }


def create_regional_comparison_scenarios() -> Dict[str, SpatialHeterogeneityManager]:
    """
    创建区域对比场景
    
    Returns:
        区域管理器字典
    """
    scenarios = {}
    for region in Region:
        scenarios[region.value] = SpatialHeterogeneityManager(region)
    return scenarios


def get_national_average_params() -> Dict:
    """
    计算全国平均参数(加权平均)
    
    基于各区域粮食产量占比:
    - 东北平原: 30%
    - 华北平原: 25%
    - 长江中下游: 25%
    - 西南山区: 12%
    - 西北干旱区: 8%
    """
    weights = {
        Region.NORTHEAST_PLAIN: 0.30,
        Region.NORTH_CHINA_PLAIN: 0.25,
        Region.YANGTZE_RIVER: 0.25,
        Region.SOUTHWEST_MOUNTAIN: 0.12,
        Region.NORTHWEST_ARID: 0.08,
    }
    
    # 计算加权平均
    avg_params = {
        'grain_yield_multiplier': 0.0,
        'subsidy_multiplier': 0.0,
        'land_price_multiplier': 0.0,
        'wage_level_multiplier': 0.0,
        'mechanization_potential': 0.0,
        'fragmentation_index': 0.0,
    }
    
    for region, weight in weights.items():
        params = REGIONAL_PARAMS[region]
        avg_params['grain_yield_multiplier'] += params.grain_yield_multiplier * weight
        avg_params['subsidy_multiplier'] += params.subsidy_multiplier * weight
        avg_params['land_price_multiplier'] += params.land_price_multiplier * weight
        avg_params['wage_level_multiplier'] += params.wage_level_multiplier * weight
        avg_params['mechanization_potential'] += params.mechanization_potential * weight
        avg_params['fragmentation_index'] += params.fragmentation_index * weight
    
    return avg_params


if __name__ == "__main__":
    # 示例：打印所有区域参数
    print("=" * 60)
    print("中国农业区域参数库")
    print("=" * 60)
    
    for region in Region:
        manager = SpatialHeterogeneityManager(region)
        summary = manager.get_region_summary()
        print(f"\n【{summary['区域']}】")
        print(f"  省份: {summary['省份']}")
        print(f"  描述: {summary['描述']}")
        print(f"  平均地块: {summary['平均地块大小']}")
        print(f"  细碎化: {summary['细碎化程度']}")
        print(f"  单产倍数: {summary['单产倍数']}")
        print(f"  补贴倍数: {summary['补贴倍数']}")
        print(f"  工资倍数: {summary['工资倍数']}")
        print(f"  非农机会: {summary['非农机会']}")
    
    print("\n" + "=" * 60)
    print("全国加权平均参数")
    print("=" * 60)
    avg_params = get_national_average_params()
    for key, value in avg_params.items():
        print(f"  {key}: {value:.3f}")
