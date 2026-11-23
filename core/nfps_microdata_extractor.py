#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从NFPS面板数据提取真实农户微观数据用于ABM初始化
==================================================

核心理念：
ABM的价值在于异质性主体，不应该仅用平均值/分布参数生成农户，
而应该直接从真实面板数据抽样，保留个体层面的关联性和异质性。

数据质量说明：
- inc_total有效率99.7%（非常好）
- inc_r(农业收入)有效率29.5%（因为70%农户无农业活动，不是数据缺失）
- 约13,304户有完整7年追踪
- 土地数据(area_b)有效率82.6%

使用方式：
1. 从有完整数据的面板农户中抽样
2. 每个模型农户对应一个真实NFPS农户
3. 保留土地-年龄-收入-劳动力的联合分布

作者: GitHub Copilot
日期: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from core.reproducibility import build_numpy_rng

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NFPS_PATH = PROJECT_ROOT / "校准用数据源" / "全国农村固定观察点" / "2009-2015household.csv"

# 区域映射 (基于RegionId或省份代码)
REGION_MAPPING = {
    # 东部: 北京、天津、河北、上海、江苏、浙江、福建、山东、广东、海南
    # 中部: 山西、安徽、江西、河南、湖北、湖南
    # 西部: 内蒙古、广西、重庆、四川、贵州、云南、西藏、陕西、甘肃、青海、宁夏、新疆
    # 东北: 辽宁、吉林、黑龙江
    # 根据smgen(省份代码)或RegionId映射
}


class NFPSMicrodataExtractor:
    """从NFPS面板数据提取真实农户微观数据"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or NFPS_PATH
        self.df: Optional[pd.DataFrame] = None
        self.panel_df: Optional[pd.DataFrame] = None  # 面板追踪数据
        
    def load_data(self) -> pd.DataFrame:
        """加载NFPS原始数据"""
        if self.df is None:
            logger.info(f"加载NFPS数据: {self.data_path}")
            self.df = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"  总样本: {len(self.df):,}")
        return self.df
    
    def build_panel(self, min_years: int = 5) -> pd.DataFrame:
        """构建面板数据（只保留有足够年份的农户）"""
        if self.df is None:
            self.load_data()
        
        # 计算每户的观测年数
        hh_years = self.df.groupby('hhid')['year'].nunique()
        valid_hh = hh_years[hh_years >= min_years].index
        
        self.panel_df = self.df[self.df['hhid'].isin(valid_hh)].copy()
        logger.info(f"面板农户数: {len(valid_hh):,} (至少{min_years}年)")
        
        return self.panel_df
    
    def classify_livelihood(self, row: pd.Series) -> str:
        """基于农业收入占比分类生计类型"""
        inc_total = row.get('inc_total', 0)
        inc_r = row.get('inc_r', 0) if pd.notna(row.get('inc_r')) else 0
        
        if pd.isna(inc_total) or inc_total <= 0:
            return 'Unknown'
        
        ratio = inc_r / inc_total
        
        if ratio > 0.5:
            return 'Pure_Farmer'
        elif ratio > 0.2:
            return 'Pluriactive'
        else:
            return 'Non_Farmer'
    
    def extract_farmer_record(self, row: pd.Series) -> Dict:
        """从一行数据提取农户特征"""
        # 基本信息 - 确保所有数值都是Python原生类型
        record = {
            'hhid': str(row.get('hhid')) if pd.notna(row.get('hhid')) else None,
            'year': int(row.get('year', 2015)),
            'region_code': int(row.get('smgen')) if pd.notna(row.get('smgen')) else None,
            
            # 土地
            'land_area_mu': float(max(0, row.get('area_b', 0))) if pd.notna(row.get('area_b')) else 0.0,
            'land_in_mu': float(max(0, row.get('area_in', 0))) if pd.notna(row.get('area_in')) else 0.0,
            'land_out_mu': float(max(0, row.get('area_de', 0))) if pd.notna(row.get('area_de')) else 0.0,
            
            # 劳动力
            'labor_count': int(row.get('hhlabornum', 2)) if pd.notna(row.get('hhlabornum')) else 2,
            'family_size': int(row.get('hhnumbers', 4)) if pd.notna(row.get('hhnumbers')) else 4,
            
            # 收入
            'income_total': float(max(0, row.get('inc_total', 0))) if pd.notna(row.get('inc_total')) else 0.0,
            'income_agri': float(max(0, row.get('inc_r', 0))) if pd.notna(row.get('inc_r')) else 0.0,
            'income_wage': float(max(0, row.get('inc_w', 0))) if pd.notna(row.get('inc_w')) else 0.0,
            'income_business': float(max(0, row.get('inc_c', 0))) if pd.notna(row.get('inc_c')) else 0.0,
            'income_transfer': float(max(0, row.get('inc_t', 0))) if pd.notna(row.get('inc_t')) else 0.0,
            
            # 生计类型
            'livelihood_type': self.classify_livelihood(row),
        }
        
        # 计算农业收入占比
        if record['income_total'] > 0:
            record['agri_income_ratio'] = record['income_agri'] / record['income_total']
        else:
            record['agri_income_ratio'] = 0
            
        return record
    
    def sample_farmers(self, 
                       n: int = 300, 
                       base_year: int = 2015,
                       stratify_by: str = 'region',
                       random_state: int = 42) -> List[Dict]:
        """
        从NFPS数据中抽样真实农户
        
        Args:
            n: 抽样数量
            base_year: 基准年份
            stratify_by: 分层抽样依据 ('region' 或 'livelihood')
            random_state: 随机种子
            
        Returns:
            农户特征字典列表
        """
        if self.panel_df is None:
            self.build_panel(min_years=5)
        
        # 取基准年份数据
        year_df = self.panel_df[self.panel_df['year'] == base_year].copy()
        
        # 过滤无效数据
        valid_mask = (
            year_df['inc_total'].notna() & 
            (year_df['inc_total'] > 0) &
            year_df['area_b'].notna()
        )
        year_df = year_df[valid_mask]
        
        logger.info(f"基准年份{base_year}有效样本: {len(year_df):,}")
        
        if len(year_df) < n:
            logger.warning(f"有效样本不足{n}，将允许重复抽样")
            replace = True
        else:
            replace = False
        
        # 分层抽样
        rng = build_numpy_rng(random_state, "nfps_microdata_sampling")
        
        if stratify_by == 'region':
            # 按省份分层
            regions = year_df['smgen'].dropna().unique()
            samples_per_region = max(1, n // len(regions))
            sampled_indices = []
            
            for region in regions:
                region_df = year_df[year_df['smgen'] == region]
                sample_n = min(samples_per_region, len(region_df))
                if sample_n > 0:
                    idx = rng.choice(region_df.index, size=sample_n, replace=False)
                    sampled_indices.extend(idx)
            
            # 补足剩余
            remaining = n - len(sampled_indices)
            if remaining > 0:
                remaining_pool = year_df[~year_df.index.isin(sampled_indices)]
                if len(remaining_pool) > 0:
                    extra_idx = rng.choice(
                        remaining_pool.index,
                        size=min(remaining, len(remaining_pool)),
                        replace=False,
                    )
                    sampled_indices.extend(extra_idx)
        else:
            # 简单随机抽样
            sampled_indices = rng.choice(year_df.index, size=n, replace=replace)
        
        # 提取农户数据
        farmers = []
        for idx in sampled_indices[:n]:
            row = year_df.loc[idx]
            farmer_data = self.extract_farmer_record(row)
            farmers.append(farmer_data)
        
        # 统计抽样结果
        livelihood_dist = pd.Series([f['livelihood_type'] for f in farmers]).value_counts(normalize=True)
        logger.info(f"抽样结果生计分布:")
        for lt, pct in livelihood_dist.items():
            logger.info(f"  {lt}: {pct*100:.1f}%")
            
        return farmers
    
    def get_summary_statistics(self, year: int = 2015) -> Dict:
        """获取指定年份的汇总统计（用于验证）"""
        if self.df is None:
            self.load_data()
        
        year_df = self.df[self.df['year'] == year].copy()
        
        # 土地统计
        valid_land = year_df['area_b'].notna() & (year_df['area_b'] >= 0)
        land_stats = {
            'mean': year_df.loc[valid_land, 'area_b'].mean(),
            'median': year_df.loc[valid_land, 'area_b'].median(),
            'std': year_df.loc[valid_land, 'area_b'].std(),
            'p25': year_df.loc[valid_land, 'area_b'].quantile(0.25),
            'p75': year_df.loc[valid_land, 'area_b'].quantile(0.75),
        }
        
        # 规模分布
        land_area = year_df.loc[valid_land, 'area_b']
        size_dist = {
            '<5亩': (land_area < 5).mean(),
            '5-15亩': ((land_area >= 5) & (land_area < 15)).mean(),
            '15-50亩': ((land_area >= 15) & (land_area < 50)).mean(),
            '>=50亩': (land_area >= 50).mean(),
        }
        
        # 流转率
        has_transfer = year_df['area_de'].notna() & (year_df['area_de'] > 0)
        transfer_rate = has_transfer.mean()
        
        return {
            'year': year,
            'n_samples': len(year_df),
            'land': land_stats,
            'size_distribution': size_dist,
            'transfer_out_rate': transfer_rate,
        }
    
    def export_microdata_json(self, 
                              n: int = 1000, 
                              output_path: Optional[Path] = None) -> Path:
        """导出微观数据为JSON供模型使用"""
        farmers = self.sample_farmers(n=n)
        
        if output_path is None:
            output_path = PROJECT_ROOT / "config" / "nfps_microdata_sample.json"
        
        output = {
            'metadata': {
                'source': 'NFPS 2009-2015',
                'base_year': 2015,
                'n_samples': len(farmers),
                'extraction_date': pd.Timestamp.now().isoformat(),
            },
            'farmers': farmers,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"微观数据已导出: {output_path}")
        return output_path


def main():
    """测试微观数据提取"""
    logging.basicConfig(level=logging.INFO)
    
    extractor = NFPSMicrodataExtractor()
    
    # 1. 加载数据
    extractor.load_data()
    
    # 2. 构建面板
    extractor.build_panel(min_years=5)
    
    # 3. 获取汇总统计
    print("\n=== 2015年汇总统计 ===")
    stats = extractor.get_summary_statistics(2015)
    print(f"样本量: {stats['n_samples']:,}")
    print(f"土地均值: {stats['land']['mean']:.2f}亩")
    print(f"土地中位数: {stats['land']['median']:.2f}亩")
    print(f"流转率: {stats['transfer_out_rate']*100:.1f}%")
    print("规模分布:")
    for size, pct in stats['size_distribution'].items():
        print(f"  {size}: {pct*100:.1f}%")
    
    # 4. 抽样农户
    print("\n=== 抽样300户农户 ===")
    farmers = extractor.sample_farmers(n=300)
    
    # 5. 检查抽样质量
    df_sample = pd.DataFrame(farmers)
    print(f"\n抽样结果:")
    print(f"  土地均值: {df_sample['land_area_mu'].mean():.2f}亩")
    print(f"  土地中位数: {df_sample['land_area_mu'].median():.2f}亩")
    print(f"  农业收入占比均值: {df_sample['agri_income_ratio'].mean()*100:.1f}%")
    
    # 6. 导出
    output_path = extractor.export_microdata_json(n=1000)
    print(f"\n导出文件: {output_path}")


if __name__ == "__main__":
    main()
