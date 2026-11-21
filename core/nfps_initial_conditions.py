# -*- coding: utf-8 -*-
"""
基于全国农村固定观察点(NFPS)数据的初始条件加载器

@update 2025-12-23: 
将初始条件数据源从CLES(2020-2021)转换为NFPS(2009-2015)
确保模型校准、验证和运行都基于同一数据源

数据来源：
- 校准期: 2009-2015年全国固定观察点数据 (144,086个样本)
- 验证期: 2016-2017年全国固定观察点数据 (12,252个样本)
- 覆盖: 31个省份，东中西东北四大区域

核心理念：
全国尺度模型需要全国尺度数据驱动
校准、验证、初始化三者必须明确各自数据口径：NFPS用于土地、收入和流转校准，生计t=0使用CFPS 2010基期。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NFPSInitialConditions:
    """基于NFPS校准结果的初始条件管理器"""
    
    # 区域权重（基于NFPS样本量）
    # 2009-2015年144,086样本分布
    REGION_WEIGHTS = {
        '东部': 0.28,   # 40,532 / 144,086 ≈ 28%
        '中部': 0.27,   # 39,035 / 144,086 ≈ 27%
        '西部': 0.29,   # 42,203 / 144,086 ≈ 29%
        '东北': 0.16,   # 22,316 / 144,086 ≈ 16%
    }
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Args:
            calibration_file: 校准结果JSON文件路径，默认为config/calibration_results_2009_2015.json
        """
        if calibration_file is None:
            calibration_file = PROJECT_ROOT / "config" / "calibration_results_2009_2015.json"
        else:
            calibration_file = Path(calibration_file)
        
        self.calibration_file = calibration_file
        self.calibration_data: Optional[Dict] = None
        self._load_calibration_results()
    
    def _load_calibration_results(self) -> None:
        """加载校准结果"""
        if not self.calibration_file.exists():
            logger.warning(f"校准结果文件不存在: {self.calibration_file}")
            return
        
        with open(self.calibration_file, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        logger.info(f"✓ 加载NFPS校准结果: {self.calibration_file}")
        metadata = self.calibration_data.get('metadata', {})
        logger.info(f"  数据源: {metadata.get('data_source', 'N/A')}")
        logger.info(f"  样本量: {metadata.get('sample_size', 'N/A'):,}")
        logger.info(f"  区域: {', '.join(metadata.get('regions_covered', []))}")
    
    def get_regional_parameters(self, region: str) -> Dict[str, Any]:
        """获取指定区域的校准参数
        
        Args:
            region: 区域名称 (东部/中部/西部/东北)
            
        Returns:
            包含该区域所有校准参数的字典
        """
        if self.calibration_data is None:
            return {}
        
        params = {
            'land_transfer': self.calibration_data.get('land_transfer_rate', {}).get(region, {}),
            'income_structure': self.calibration_data.get('income_structure', {}).get(region, {}),
            'agricultural_dependency': self.calibration_data.get('agricultural_dependency', {}).get(region, {}),
            'land_distribution': self.calibration_data.get('land_distribution', {}).get(region, {}),
        }
        return params
    
    def get_national_livelihood_distribution(self) -> Dict[str, float]:
        """计算全国加权的生计策略分布
        
        基于NFPS各区域收入结构和区域权重计算全国分布
        
        Returns:
            dict: {策略类型: 比例}
        """
        if self.calibration_data is None:
            # 默认保守估计
            return {
                'Pure_Farmer': 0.05,    # 纯农户5%
                'Pluriactive': 0.10,    # 兼业户10%
                'Non_Farmer': 0.85,     # 非农户85%
            }
        
        # 基于各区域生计分布和权重计算全国加权平均
        income_data = self.calibration_data.get('income_structure', {})
        
        target_keys = ('非农型', '兼业型', '纯农型')
        weighted_dist = {key: 0.0 for key in target_keys}
        
        for region, weight in self.REGION_WEIGHTS.items():
            region_data = income_data.get(region, {})
            livelihood = region_data.get('livelihood_distribution', {})
            
            for strategy in target_keys:
                ratio = float(livelihood.get(strategy, 0.0) or 0.0)
                weighted_dist[strategy] += weight * ratio
        
        # 标准化确保总和为1
        total = sum(weighted_dist.values())
        if total > 0:
            weighted_dist = {k: v / total for k, v in weighted_dist.items()}
        
        # 转换为模型使用的英文键名
        result = {
            'Pure_Farmer': round(weighted_dist.get('纯农型', 0.05), 4),
            'Pluriactive': round(weighted_dist.get('兼业型', 0.10), 4),
            'Non_Farmer': round(weighted_dist.get('非农型', 0.85), 4),
        }
        
        logger.info(f"全国生计策略分布(NFPS加权): Pure_Farmer={result['Pure_Farmer']:.1%}, "
                   f"Pluriactive={result['Pluriactive']:.1%}, Non_Farmer={result['Non_Farmer']:.1%}")
        
        return result
    
    def get_national_land_distribution(self) -> Dict[str, Any]:
        """计算全国加权的土地规模分布参数
        
        Returns:
            dict: 包含对数正态分布参数的字典
        """
        if self.calibration_data is None:
            return {
                'lognormal_params': {'mu': 1.5, 'sigma': 1.0},
                'statistics': {'mean': 7.2, 'median': 4.5},
            }
        
        land_data = self.calibration_data.get('land_distribution', {})
        
        # 加权计算对数正态分布参数
        weighted_mu = 0
        weighted_sigma = 0
        weighted_mean = 0
        
        for region, weight in self.REGION_WEIGHTS.items():
            region_land = land_data.get(region, {})
            params = region_land.get('lognormal_params', {})
            stats = region_land.get('statistics', {})
            
            weighted_mu += weight * params.get('mu', 1.5)
            weighted_sigma += weight * params.get('sigma', 1.0)
            weighted_mean += weight * stats.get('mean', 7.2)
        
        result = {
            'lognormal_params': {
                'mu': round(weighted_mu, 4),
                'sigma': round(weighted_sigma, 4),
            },
            'statistics': {
                'mean': round(weighted_mean, 2),
            }
        }
        
        logger.info(f"全国土地分布(NFPS加权): μ={result['lognormal_params']['mu']:.3f}, "
                   f"σ={result['lognormal_params']['sigma']:.3f}, 均值={result['statistics']['mean']:.1f}亩")
        
        return result
    
    def get_national_income_statistics(self) -> Dict[str, Any]:
        """计算全国加权的收入统计参数
        
        Returns:
            dict: 包含收入分布参数的字典
        """
        if self.calibration_data is None:
            return {
                'mean': 40000,
                'median': 32000,
                'gini': 0.42,
            }
        
        income_data = self.calibration_data.get('income_structure', {})
        
        weighted_mean = 0
        weighted_median = 0
        weighted_gini = 0
        
        for region, weight in self.REGION_WEIGHTS.items():
            region_income = income_data.get(region, {})
            stats = region_income.get('income_stats', {})
            
            weighted_mean += weight * stats.get('mean', 40000)
            weighted_median += weight * stats.get('median', 32000)
            weighted_gini += weight * stats.get('gini', 0.42)
        
        result = {
            'mean': round(weighted_mean, 2),
            'median': round(weighted_median, 2),
            'gini': round(weighted_gini, 4),
        }
        
        # 计算对数正态分布参数（用于初始资本分布）
        # ln(mean^2 / sqrt(var + mean^2))
        import numpy as np
        mean_val = result['mean']
        # 假设变异系数CV约0.8（基于GINI）
        cv = result['gini'] * 1.5  # 粗略近似
        std_val = mean_val * cv
        var_val = std_val ** 2
        
        log_mean = np.log(mean_val ** 2 / np.sqrt(var_val + mean_val ** 2))
        log_std = np.sqrt(np.log(1 + var_val / mean_val ** 2))
        
        result['lognormal'] = {
            'log_mean': round(log_mean, 4),
            'log_std': round(log_std, 4),
        }
        
        logger.info(f"全国收入统计(NFPS加权): 均值={result['mean']:.0f}元, "
                   f"中位数={result['median']:.0f}元, Gini={result['gini']:.3f}")
        
        return result
    
    def get_national_transfer_rates(self) -> Dict[str, float]:
        """计算全国加权的土地流转率
        
        Returns:
            dict: 包含转入率、转出率等的字典
        """
        if self.calibration_data is None:
            return {
                'rent_out_rate': 0.18,
                'rent_in_rate': 0.06,
                'participation_rate': 0.23,
            }
        
        transfer_data = self.calibration_data.get('land_transfer_rate', {})
        
        weighted_out = 0
        weighted_in = 0
        weighted_participation = 0
        
        for region, weight in self.REGION_WEIGHTS.items():
            region_transfer = transfer_data.get(region, {})
            
            weighted_out += weight * region_transfer.get('rent_out_rate', 0.18)
            weighted_in += weight * region_transfer.get('rent_in_rate', 0.06)
            weighted_participation += weight * region_transfer.get('participation_rate', 0.23)
        
        result = {
            'rent_out_rate': round(weighted_out, 4),
            'rent_in_rate': round(weighted_in, 4),
            'participation_rate': round(weighted_participation, 4),
        }
        
        logger.info(f"全国流转率(NFPS加权): 转出率={result['rent_out_rate']:.1%}, "
                   f"转入率={result['rent_in_rate']:.1%}, 参与率={result['participation_rate']:.1%}")
        
        return result
    
    def get_time_trends(self) -> Dict[str, Any]:
        """获取时间趋势参数
        
        Returns:
            dict: 各指标的年均增长率和显著性
        """
        if self.calibration_data is None:
            return {}
        
        return self.calibration_data.get('time_trends', {}).get('variable_trends', {})


def apply_nfps_initial_conditions(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """应用NFPS数据驱动的初始条件到配置
    
    这是主入口函数，替代原来的CLES数据驱动
    
    Args:
        config: 原始配置字典
        
    Returns:
        tuple: (更新后的配置, 初始条件统计信息)
    """
    ic_cfg = config.get("initial_conditions", {})
    
    # 检查是否启用NFPS数据源
    data_source = ic_cfg.get("data_source", "nfps")
    if data_source.lower() != "nfps":
        logger.info(f"初始条件数据源为 {data_source}，跳过NFPS加载")
        return config, {}
    
    logger.info("=" * 60)
    logger.info("应用NFPS数据驱动的初始条件")
    logger.info("=" * 60)
    
    # 加载NFPS校准结果
    calibration_file = ic_cfg.get("calibration_file")
    nfps = NFPSInitialConditions(calibration_file)
    
    if nfps.calibration_data is None:
        logger.warning("无法加载NFPS校准结果，使用默认配置")
        return config, {}
    
    stats = {}
    
    # 1. 更新生计策略分布
    livelihood_dist = nfps.get_national_livelihood_distribution()
    farmer_cfg = config.setdefault("agents", {}).setdefault("farmer", {})
    existing_livelihood_dist = dict(farmer_cfg.get("livelihood_strategy_distribution", {}) or {})
    stats['nfps_livelihood_distribution'] = livelihood_dist

    explicit_cfps_base_year_dist = (
        existing_livelihood_dist
        and float(existing_livelihood_dist.get("Pure_Farmer", 0.0) or 0.0) > 0.10
    )
    if explicit_cfps_base_year_dist:
        stats['livelihood_distribution'] = existing_livelihood_dist
        farmer_cfg["reference_livelihood_distribution_nfps"] = livelihood_dist
        logger.info(
            "保留配置中的CFPS基期生计分布；NFPS生计分布仅作为参照统计，不覆盖模型首端口径"
        )
    else:
        stats['livelihood_distribution'] = livelihood_dist
        farmer_cfg["livelihood_strategy_distribution"] = livelihood_dist
    
    # 2. 更新土地分布参数
    land_dist = nfps.get_national_land_distribution()
    stats['land_distribution'] = land_dist
    
    # 更新土地流转相关参数
    land_transfer_cfg = farmer_cfg.setdefault("land_transfer", {})
    land_params = land_dist['lognormal_params']
    land_transfer_cfg["land_area_lognormal_mu"] = land_params['mu']
    land_transfer_cfg["land_area_lognormal_sigma"] = land_params['sigma']
    
    # 3. 更新收入参数
    income_stats = nfps.get_national_income_statistics()
    stats['income_statistics'] = income_stats
    
    # 更新初始资本分布（基于收入的对数正态分布）
    if 'lognormal' in income_stats:
        farmer_cfg["initial_capital_dist"] = {
            "type": "lognormal",
            "mean": income_stats['lognormal']['log_mean'],
            "std": income_stats['lognormal']['log_std'],
        }
    
    # 4. 更新流转率参数
    transfer_rates = nfps.get_national_transfer_rates()
    stats['transfer_rates'] = transfer_rates
    
    # 将校准的流转率写入配置，用于验证模型输出
    config.setdefault("calibration_targets", {})
    config["calibration_targets"]["transfer_rates"] = transfer_rates
    
    # 5. 更新区域参数
    regional_params = {}
    for region in ['东部', '中部', '西部', '东北']:
        regional_params[region] = nfps.get_regional_parameters(region)
    stats['regional_parameters'] = regional_params
    
    # 写入区域差异化配置
    config.setdefault("regional_heterogeneity", {})
    config["regional_heterogeneity"]["parameters"] = regional_params
    
    # 6. 更新时间趋势
    time_trends = nfps.get_time_trends()
    stats['time_trends'] = time_trends
    
    logger.info("=" * 60)
    logger.info("✓ NFPS初始条件应用完成")
    logger.info("=" * 60)
    
    return config, stats


# 便捷函数
def get_nfps_initial_conditions() -> NFPSInitialConditions:
    """获取NFPS初始条件实例"""
    return NFPSInitialConditions()
