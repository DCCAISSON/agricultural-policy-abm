"""
CFPS特征提取器：从CFPS截面数据中提取"新进入者"特征

设计目标:
- 筛选"返乡精英"、"规模经营户"等特定人群
- 提取特征分布用于PopulationManager的Bootstrap重采样
- 支持多年度截面数据的合并

数据源: 校准用数据源/CFPS/ 目录下的各年份数据

作者: Agricultural Policy ABM Team
日期: 2026-01-06
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from core.reproducibility import build_numpy_rng

logger = logging.getLogger(__name__)


class CFPSFeatureExtractor:
    """
    CFPS特征提取器
    
    从预处理的CFPS截面数据中筛选特定人群，提取用于ABM新进入者生成的特征
    
    数据源: 校准用数据源/CFPS/cfps_micro_data_for_initialization.csv
    - 43,525条记录，覆盖2010-2022年7个调查波次
    - 包含区域、生计类型、收入、土地等关键变量
    """
    
    def __init__(self, cfps_base_path: Optional[str] = None, seed: Optional[int] = None):
        """
        初始化提取器
        
        Args:
            cfps_base_path: CFPS数据根目录
        """
        if cfps_base_path is None:
            cfps_base_path = Path(__file__).parent.parent / '校准用数据源' / 'CFPS'
        self.cfps_path = Path(cfps_base_path)
        self.seed = seed
        self.rng = build_numpy_rng(seed, "cfps_feature_extractor")
        
        # 预处理好的汇总数据文件
        self.preprocessed_file = self.cfps_path / 'cfps_micro_data_for_initialization.csv'
        
        # 年份到文件的映射（备用：原始dta文件）
        self.year_file_mapping = {
            2010: 'cfps2010',
            2012: 'cfps2012', 
            2014: 'cfps2014',
            2016: 'cfps2016',
            2018: 'cfps2018',
            2020: 'cfps2020',
            2022: 'cfps2022'
        }
        
        # 特征提取结果
        self.extracted_features = {
            'return_migrant': [],
            'family_farm': [],
            'cooperative': [],
            'young_household': []
        }
        
        # 预处理文件的列名（cfps_micro_data_for_initialization.csv）
        # fid, region, livelihood, agri_income, net_income, total_income, 
        # wage_income, land_area_mu, land_rent_in_area, land_rent_out_area, 
        # land_rent_income, land_rent_cost, year
        self.preprocessed_columns = {
            'household_id': 'fid',
            'region': 'region',  # 东部/中部/西部/东北
            'livelihood': 'livelihood',  # 纯农型/兼业型/非农型
            'income_agri': 'agri_income',
            'income_net': 'net_income',
            'income_total': 'total_income',
            'income_wage': 'wage_income',
            'land_area': 'land_area_mu',
            'land_rent_in': 'land_rent_in_area',
            'land_rent_out': 'land_rent_out_area',
            'year': 'year'
        }
        
        # 变量名映射（原始dta文件的变量名）
        self.var_mapping = {
            'household_id': ['fid', 'fid10', 'fid12', 'fid14', 'fid16', 'fid18', 'fid20', 'fid22'],
            'hukou_type': ['qa301', 'qa1', 'cfps_hukou', 'hukou'],
            'residence_type': ['urban', 'urban10', 'provcd', 'countyid'],
            'age': ['cfps_age', 'age', 'qa1age'],
            'gender': ['cfps_gender', 'gender', 'qa2'],
            'education': ['cfps_edu', 'edu', 'tr1', 'feduc'],
            'income_total': ['fincome', 'fincome1', 'faminc', 'income'],
            'income_agri': ['fincome_agri', 'agri_income', 'fa1'],
            'land_area': ['land_mu', 'landarea', 'fa2'],
            'migrant_exp': ['migrant', 'qa301a', 'outwork'],
            'household_size': ['familysize', 'fsize', 'qa101']
        }
    
    def load_preprocessed_data(self) -> Optional[pd.DataFrame]:
        """
        加载预处理好的CFPS汇总数据
        
        优先使用: cfps_micro_data_for_initialization.csv
        包含43,525条记录，2010-2022年7个波次
        
        Returns:
            DataFrame或None
        """
        if self.preprocessed_file.exists():
            try:
                df = pd.read_csv(self.preprocessed_file)
                logger.info(f"成功加载预处理数据: {len(df)}条记录")
                return df
            except Exception as e:
                logger.warning(f"加载预处理数据失败: {e}")
        return None

    @staticmethod
    def _safe_positive(value, default=0.0) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(numeric_value) or numeric_value <= 0:
            return float(default)
        return numeric_value

    def _prepare_preprocessed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗预处理截面表，统一 entrant 抽样的收入与规模口径。"""
        cleaned = df.copy()

        numeric_cols = [
            'agri_income',
            'net_income',
            'total_income',
            'wage_income',
            'land_area_mu',
            'land_rent_in_area',
            'land_rent_out_area',
        ]
        for col in numeric_cols:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

        for col in ['land_area_mu', 'land_rent_in_area', 'land_rent_out_area']:
            if col in cleaned.columns:
                cleaned.loc[cleaned[col] < 0, col] = np.nan

        for col in ['agri_income', 'net_income', 'total_income', 'wage_income']:
            if col in cleaned.columns:
                cleaned.loc[cleaned[col] < 0, col] = np.nan

        def _series_or_nan(column_name: str) -> pd.Series:
            if column_name in cleaned.columns:
                return cleaned[column_name]
            return pd.Series(np.nan, index=cleaned.index, dtype=float)

        agri_income_raw = _series_or_nan('agri_income')
        net_income_raw = _series_or_nan('net_income')
        total_income_raw = _series_or_nan('total_income')
        wage_income_raw = _series_or_nan('wage_income')

        # Keep gross, net, and component incomes distinct instead of inflating
        # total income with a row-wise max across incompatible concepts.
        component_income = pd.concat(
            [
                agri_income_raw.fillna(0.0).clip(lower=0.0),
                wage_income_raw.fillna(0.0).clip(lower=0.0),
            ],
            axis=1,
        ).sum(axis=1, min_count=1)
        total_income_candidates = pd.concat(
            [total_income_raw, net_income_raw, component_income],
            axis=1,
        )
        cleaned['income_total_sanitized'] = (
            total_income_candidates.bfill(axis=1).iloc[:, 0].fillna(0.0).clip(lower=0.0)
        )

        cleaned['income_agri_sanitized'] = np.minimum(
            agri_income_raw.fillna(0.0).clip(lower=0.0),
            cleaned['income_total_sanitized'],
        )
        cleaned['income_net_sanitized'] = np.minimum(
            net_income_raw.fillna(cleaned['income_total_sanitized']).clip(lower=0.0),
            cleaned['income_total_sanitized'],
        )
        cleaned['income_wage_sanitized'] = np.minimum(
            wage_income_raw.fillna(0.0).clip(lower=0.0),
            cleaned['income_total_sanitized'],
        )

        inferred_area_from_income = cleaned['income_agri_sanitized'] / 1800.0
        land_components = pd.concat(
            [
                cleaned.get('land_area_mu', pd.Series(np.nan, index=cleaned.index)),
                cleaned.get('land_rent_in_area', pd.Series(np.nan, index=cleaned.index)),
                cleaned.get('land_rent_out_area', pd.Series(np.nan, index=cleaned.index)),
                inferred_area_from_income,
            ],
            axis=1,
        )
        cleaned['operated_area_mu_sanitized'] = land_components.max(axis=1, skipna=True)
        cleaned.loc[cleaned['operated_area_mu_sanitized'] <= 0, 'operated_area_mu_sanitized'] = np.nan

        return cleaned
    
    def extract_from_preprocessed(self, years: Optional[List[int]] = None) -> Dict[str, List[Dict]]:
        """
        从预处理数据中提取各类新进入者特征
        
        Args:
            years: 要提取的年份列表，默认使用最近3年
        
        Returns:
            特征字典
        """
        df = self.load_preprocessed_data()
        if df is None:
            logger.warning("无法加载预处理数据，将生成合成特征库")
            return self.generate_synthetic_library(100)
        
        if years is None:
            years = [2018, 2020, 2022]  # 默认使用最近3年

        df = df[df['year'].isin(years)].copy()
        df = self._prepare_preprocessed_dataframe(df)
        logger.info(f"筛选年份 {years}: {len(df)}条记录")

        # 提取返乡精英（非农转纯农或兼业，高收入）
        return_migrants = self._extract_return_migrants_from_preprocessed(df)
        
        # 提取家庭农场（纯农型，中等以上土地）
        family_farms = self._extract_family_farms_from_preprocessed(df)
        
        # 提取兼业型（作为young_household）
        young_households = self._extract_young_households_from_preprocessed(df)
        
        # 合成合作社数据（因CFPS无企业数据）
        cooperatives = self._generate_cooperative_samples(len(family_farms) // 4)
        
        self.extracted_features = {
            'return_migrant': return_migrants,
            'family_farm': family_farms,
            'cooperative': cooperatives,
            'young_household': young_households
        }
        
        logger.info("=== 预处理数据特征提取汇总 ===")
        for category, samples in self.extracted_features.items():
            logger.info(f"  {category}: {len(samples)}个样本")
        
        return self.extracted_features
    
    def _extract_return_migrants_from_preprocessed(self, df: pd.DataFrame) -> List[Dict]:
        """
        从预处理数据中提取返乡精英样本
        
        筛选条件:
        - 非农型但有农业收入 > 0（表示曾务农或返乡）
        - 或兼业型收入高于中位数
        """
        samples = []
        median_income = df['income_total_sanitized'].median()

        # 条件1: 非农型但有农业收入
        mask1 = (df['livelihood'] == '非农型') & (df['income_agri_sanitized'] > 0)
        # 条件2: 兼业型高收入
        mask2 = (df['livelihood'] == '兼业型') & (df['income_total_sanitized'] > median_income)

        candidates = df[mask1 | mask2].sample(n=min(200, len(df[mask1 | mask2])),
                                               random_state=42).to_dict('records')

        for row in candidates:
            income = max(self._safe_positive(row.get('income_total_sanitized'), 50000), 10000.0)
            samples.append({
                'source': 'cfps_preprocessed',
                'year': int(row.get('year', 2020)),
                'region': row.get('region', '中部'),
                'age': int(self.rng.normal(52, 8)),  # CFPS预处理数据无年龄，模拟
                'gender': 'male' if self.rng.random_sample() > 0.35 else 'female',
                'education_level': int(np.clip(self.rng.normal(2.5, 0.8), 1, 4)),
                'household_size': int(np.clip(self.rng.normal(2.8, 1), 1, 5)),
                'initial_capital': float(income * 1.5),
                'has_migrant_experience': True,
                'tech_adoption': float(np.clip(self.rng.normal(0.35, 0.12), 0.1, 0.6)),
                'livelihood_original': row.get('livelihood', '非农型')
            })
        
        return samples
    
    def _extract_family_farms_from_preprocessed(self, df: pd.DataFrame) -> List[Dict]:
        """
        从预处理数据中提取家庭农场样本
        
        筛选条件:
        - 纯农型
        - 土地面积 > 10亩（或收入较高）
        """
        samples = []

        # 纯农型，优先保留有明确经营规模的样本
        mask = (df['livelihood'] == '纯农型') & (df['operated_area_mu_sanitized'] >= 10)
        candidates_df = df[mask]

        if len(candidates_df) < 80:
            # 规模样本不足时，放宽到中等经营规模。
            fallback_mask = (df['livelihood'] == '纯农型') & (df['operated_area_mu_sanitized'] >= 5)
            candidates_df = pd.concat([candidates_df, df[fallback_mask]], ignore_index=False).drop_duplicates()

        if len(candidates_df) == 0:
            # 如果显式面积不可用，使用纯农型高农业收入，并由收入反推经营规模。
            mask = (df['livelihood'] == '纯农型') & (
                df['income_agri_sanitized'] > df['income_agri_sanitized'].median()
            )
            candidates_df = df[mask]

        candidates = candidates_df.sample(n=min(200, len(candidates_df)),
                                          random_state=42).to_dict('records')

        for row in candidates:
            inferred_land = self._safe_positive(
                row.get('operated_area_mu_sanitized'),
                self._safe_positive(row.get('income_agri_sanitized'), 9000.0) / 1800.0,
            )
            land = max(inferred_land, 2.0)
            income = max(self._safe_positive(row.get('income_agri_sanitized'), 20000), 5000.0)
            samples.append({
                'source': 'cfps_preprocessed',
                'year': int(row.get('year', 2020)),
                'region': row.get('region', '中部'),
                'age': int(self.rng.normal(48, 10)),
                'gender': 'male' if self.rng.random_sample() > 0.4 else 'female',
                'education_level': int(np.clip(self.rng.normal(2.2, 0.7), 1, 4)),
                'household_size': int(np.clip(self.rng.normal(3.2, 1.2), 2, 6)),
                'initial_capital': float(income * 2),
                'land_area_mu': float(land),
                'has_migrant_experience': self.rng.random_sample() > 0.7,
                'tech_adoption': float(np.clip(self.rng.normal(0.4, 0.15), 0.15, 0.7)),
                'livelihood_original': '纯农型'
            })
        
        return samples
    
    def _extract_young_households_from_preprocessed(self, df: pd.DataFrame) -> List[Dict]:
        """
        从预处理数据中提取年轻农户样本（兼业型）
        """
        samples = []
        
        mask = df['livelihood'] == '兼业型'
        candidates = df[mask].sample(n=min(100, len(df[mask])),
                                     random_state=42).to_dict('records')

        for row in candidates:
            income = max(self._safe_positive(row.get('income_total_sanitized'), 40000), 10000.0)
            samples.append({
                'source': 'cfps_preprocessed',
                'year': int(row.get('year', 2020)),
                'region': row.get('region', '中部'),
                'age': int(self.rng.normal(38, 8)),  # 较年轻
                'gender': 'male' if self.rng.random_sample() > 0.45 else 'female',
                'education_level': int(np.clip(self.rng.normal(2.8, 0.8), 1, 4)),
                'household_size': int(np.clip(self.rng.normal(3.5, 1), 2, 5)),
                'initial_capital': float(income),
                'has_migrant_experience': self.rng.random_sample() > 0.4,
                'tech_adoption': float(np.clip(self.rng.normal(0.45, 0.15), 0.2, 0.7)),
                'livelihood_original': '兼业型'
            })
        
        return samples
    
    def _generate_cooperative_samples(self, n: int = 50) -> List[Dict]:
        """
        生成合作社/企业样本（CFPS无企业数据，基于文献合成）
        """
        samples = []
        for _ in range(max(n, 20)):
            samples.append({
                'source': 'synthetic_cooperative',
                'age': None,  # 非自然人
                'gender': None,
                'education_level': 4,  # 高学历管理
                'household_size': None,
                'initial_capital': float(self.rng.lognormal(13.5, 1)),  # 50-500万
                'land_area_mu': float(self.rng.lognormal(5, 0.8)),  # 100-500亩
                'has_migrant_experience': False,
                'tech_adoption': float(np.clip(self.rng.normal(0.7, 0.15), 0.4, 0.95)),
                'is_enterprise': True
            })
        return samples
    
    def _find_variable(self, df: pd.DataFrame, var_type: str) -> Optional[str]:
        """查找DataFrame中对应的变量名"""
        possible_names = self.var_mapping.get(var_type, [])
        for name in possible_names:
            if name in df.columns:
                return name
            # 尝试大小写不敏感匹配
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        return None
    
    def load_cfps_year(self, year: int) -> Optional[pd.DataFrame]:
        """
        加载指定年份的CFPS数据（原始dta文件）
        
        Args:
            year: 调查年份
        
        Returns:
            DataFrame或None
        """
        year_folder = self.year_file_mapping.get(year)
        if not year_folder:
            logger.warning(f"未配置{year}年的数据路径")
            return None
        
        # 尝试多种可能的文件路径
        possible_paths = [
            self.cfps_path / year_folder / f'cfps{year}_famecon_202008.dta',
            self.cfps_path / year_folder / f'cfps{year}_family.dta',
            self.cfps_path / year_folder / f'cfps{year}_household.dta',
            self.cfps_path / f'cfps{year}_famecon.dta',
            self.cfps_path / f'CFPS{year}' / f'cfps{year}_famecon.dta'
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    df = pd.read_stata(path, convert_categoricals=False)
                    logger.info(f"成功加载 {year} 年数据: {path}, 样本量: {len(df)}")
                    return df
                except Exception as e:
                    logger.warning(f"加载失败 {path}: {e}")
        
        logger.warning(f"未找到 {year} 年的CFPS数据文件")
        return None
    
    def extract_return_migrants(self, df: pd.DataFrame, year: int) -> List[Dict]:
        """
        提取返乡精英样本
        
        筛选条件:
        - 当前户口性质 = 农业
        - 当前居住地 = 乡村
        - 有外出务工经历 (>1年)
        - 家庭收入 > 乡村中位数 OR 受教育程度 > 初中
        """
        samples = []
        
        # 查找变量名
        hukou_var = self._find_variable(df, 'hukou_type')
        migrant_var = self._find_variable(df, 'migrant_exp')
        income_var = self._find_variable(df, 'income_total')
        edu_var = self._find_variable(df, 'education')
        age_var = self._find_variable(df, 'age')
        gender_var = self._find_variable(df, 'gender')
        hh_size_var = self._find_variable(df, 'household_size')
        
        if not all([hukou_var, income_var]):
            logger.warning(f"缺少关键变量，无法提取返乡精英")
            return samples
        
        # 计算农村收入中位数
        rural_median_income = df[income_var].median()
        
        for idx, row in df.iterrows():
            try:
                # 只有能明确识别为农村户口的样本才进入返乡精英池，避免静默污染 entrant source。
                hukou_value = row.get(hukou_var, np.nan) if hukou_var else np.nan
                hukou_text = str(hukou_value).strip().lower()
                if pd.isna(hukou_value):
                    is_rural_hukou = False
                elif any(token in hukou_text for token in ('农业', '农村', 'rural', 'agri')):
                    is_rural_hukou = True
                elif any(token in hukou_text for token in ('非农', '城镇', 'urban', 'non-agri')):
                    is_rural_hukou = False
                else:
                    try:
                        is_rural_hukou = float(hukou_value) == 1.0
                    except (TypeError, ValueError):
                        is_rural_hukou = False
                has_migrant_exp = pd.notna(row.get(migrant_var, np.nan)) if migrant_var else False
                
                income = row.get(income_var, 0) or 0
                edu = row.get(edu_var, 0) or 0
                
                is_elite = (income > rural_median_income * 1.2) or (edu >= 9)  # 9年=初中毕业
                
                if is_rural_hukou and (has_migrant_exp or is_elite):
                    sample = {
                        'year': year,
                        'source': 'cfps_return_migrant',
                        'age': int(row.get(age_var, 55)) if age_var and pd.notna(row.get(age_var)) else 55,
                        'gender': 'male' if row.get(gender_var, 1) == 1 else 'female',
                        'education_level': min(int(edu / 3) + 1, 4) if edu else 2,  # 转换为1-4级
                        'household_size': int(row.get(hh_size_var, 2)) if hh_size_var else 2,
                        'initial_capital': float(income) * 2 if income > 0 else 100000,
                        'has_migrant_experience': True,
                        'tech_adoption': 0.3 + (edu / 16) * 0.3 if edu else 0.3
                    }
                    samples.append(sample)
                    
            except Exception as e:
                continue
        
        logger.info(f"{year}年提取返乡精英样本: {len(samples)}个")
        return samples
    
    def extract_family_farms(self, df: pd.DataFrame, year: int) -> List[Dict]:
        """
        提取家庭农场样本
        
        筛选条件:
        - 经营耕地面积 > 区域阈值 (如15亩)
        - 主要依靠家庭劳动力
        - 有一定的技术采纳
        """
        samples = []
        
        land_var = self._find_variable(df, 'land_area')
        income_var = self._find_variable(df, 'income_total')
        income_agri_var = self._find_variable(df, 'income_agri')
        age_var = self._find_variable(df, 'age')
        edu_var = self._find_variable(df, 'education')
        hh_size_var = self._find_variable(df, 'household_size')
        
        # 设定规模阈值
        scale_threshold = 15  # 亩
        
        for idx, row in df.iterrows():
            try:
                land_area = row.get(land_var, 0) if land_var else 0
                if pd.isna(land_area) or land_area < scale_threshold:
                    continue
                
                income = row.get(income_var, 0) or 0
                edu = row.get(edu_var, 0) or 0
                age = row.get(age_var, 45) or 45
                
                # 规模经营户
                sample = {
                    'year': year,
                    'source': 'cfps_family_farm',
                    'age': int(age) if pd.notna(age) else 45,
                    'gender': 'male',  # 简化
                    'education_level': min(int(edu / 3) + 1, 4) if edu else 2,
                    'household_size': int(row.get(hh_size_var, 3)) if hh_size_var else 3,
                    'initial_capital': float(income) * 3 if income > 0 else 200000,
                    'land_area_mu': float(land_area),
                    'has_migrant_experience': False,
                    'tech_adoption': 0.4 + (edu / 16) * 0.4 if edu else 0.4
                }
                samples.append(sample)
                
            except Exception as e:
                continue
        
        logger.info(f"{year}年提取家庭农场样本: {len(samples)}个")
        return samples
    
    def extract_young_households(self, df: pd.DataFrame, year: int) -> List[Dict]:
        """
        提取年轻户/新立户样本
        
        用于一般性的新进入者生成
        """
        samples = []
        
        age_var = self._find_variable(df, 'age')
        income_var = self._find_variable(df, 'income_total')
        edu_var = self._find_variable(df, 'education')
        hh_size_var = self._find_variable(df, 'household_size')
        gender_var = self._find_variable(df, 'gender')
        
        if not age_var:
            return samples
        
        for idx, row in df.iterrows():
            try:
                age = row.get(age_var, 50)
                if pd.isna(age) or age > 40:  # 只取年轻户主
                    continue
                
                income = row.get(income_var, 0) or 0
                edu = row.get(edu_var, 0) or 0
                
                sample = {
                    'year': year,
                    'source': 'cfps_young_household',
                    'age': int(age),
                    'gender': 'male' if row.get(gender_var, 1) == 1 else 'female',
                    'education_level': min(int(edu / 3) + 1, 4) if edu else 2,
                    'household_size': int(row.get(hh_size_var, 2)) if hh_size_var else 2,
                    'initial_capital': float(income) * 1.5 if income > 0 else 80000,
                    'has_migrant_experience': False,
                    'tech_adoption': 0.3 + (edu / 16) * 0.5 if edu else 0.4
                }
                samples.append(sample)
                
            except Exception as e:
                continue
        
        logger.info(f"{year}年提取年轻户样本: {len(samples)}个")
        return samples
    
    def extract_all_years(self, years: Optional[List[int]] = None) -> Dict[str, List[Dict]]:
        """
        从多个年份提取特征
        
        Args:
            years: 要处理的年份列表，默认为所有可用年份
        
        Returns:
            分类的特征样本
        """
        if years is None:
            years = list(self.year_file_mapping.keys())
        
        for year in years:
            df = self.load_cfps_year(year)
            if df is None:
                continue
            
            # 提取各类样本
            return_migrants = self.extract_return_migrants(df, year)
            family_farms = self.extract_family_farms(df, year)
            young_households = self.extract_young_households(df, year)
            
            # 合并
            self.extracted_features['return_migrant'].extend(return_migrants)
            self.extracted_features['family_farm'].extend(family_farms)
            self.extracted_features['young_household'].extend(young_households)
        
        # 统计
        logger.info("=== 特征提取汇总 ===")
        for category, samples in self.extracted_features.items():
            logger.info(f"  {category}: {len(samples)}个样本")
        
        return self.extracted_features
    
    def export_feature_library(self, output_path: str) -> None:
        """
        导出特征库到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        # 处理numpy类型
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                if not np.isfinite(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 转换所有值
        export_data = {
            'metadata': {
                'description': 'CFPS特征库 - 用于ABM新进入者Bootstrap重采样',
                'created': pd.Timestamp.now().isoformat(),
                'source': 'CFPS截面数据',
                'sample_universe': 'CFPS rural cross-sections used for entrant bootstrap pools',
                'years_used': sorted({
                    int(sample.get('year'))
                    for samples in self.extracted_features.values()
                    for sample in samples
                    if sample.get('year') is not None
                }),
                'unknown_handling': 'retain original livelihood labels for entrant-pool matching and normalize at runtime',
            }
        }
        
        for category, samples in self.extracted_features.items():
            export_data[category] = [
                {k: convert_types(v) for k, v in sample.items()}
                for sample in samples
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"特征库已导出: {output_path}")
    
    def generate_synthetic_library(self, n_per_category: int = 100) -> Dict[str, List[Dict]]:
        """
        生成合成特征库（当真实CFPS数据不可用时）
        
        基于文献中的典型特征分布生成
        """
        logger.info("生成合成特征库...")
        
        # 返乡精英特征（基于文献）
        return_migrants = []
        for _ in range(n_per_category):
            return_migrants.append({
                'source': 'synthetic',
                'age': int(self.rng.normal(55, 8)),  # 51-70岁为主
                'gender': 'male' if self.rng.random_sample() > 0.3 else 'female',  # 男性为主
                'education_level': int(np.clip(self.rng.normal(2.5, 0.8), 1, 4)),
                'household_size': int(np.clip(self.rng.normal(2.5, 1), 1, 5)),
                'initial_capital': float(self.rng.lognormal(11.5, 0.8)),  # ~10-30万
                'has_migrant_experience': True,
                'tech_adoption': float(np.clip(self.rng.normal(0.35, 0.15), 0.1, 0.7))
            })
        
        # 家庭农场特征
        family_farms = []
        for _ in range(n_per_category):
            family_farms.append({
                'source': 'synthetic',
                'age': int(self.rng.normal(42, 10)),
                'gender': 'male' if self.rng.random_sample() > 0.4 else 'female',
                'education_level': int(np.clip(self.rng.normal(3, 0.8), 1, 4)),
                'household_size': int(np.clip(self.rng.normal(3.5, 1), 2, 6)),
                'initial_capital': float(self.rng.lognormal(12, 0.7)),  # ~15-50万
                'land_area_mu': float(self.rng.lognormal(3.5, 0.8)),  # 30-100亩
                'has_migrant_experience': self.rng.random_sample() > 0.7,
                'tech_adoption': float(np.clip(self.rng.normal(0.5, 0.15), 0.2, 0.8))
            })
        
        # 合作社/企业特征
        cooperatives = []
        for _ in range(n_per_category // 2):  # 合作社数量较少
            cooperatives.append({
                'source': 'synthetic',
                'age': None,  # 非自然人
                'gender': None,
                'education_level': 4,  # 高学历管理
                'household_size': None,
                'initial_capital': float(self.rng.lognormal(13.5, 1)),  # 50-500万
                'land_area_mu': float(self.rng.lognormal(5, 0.8)),  # 100-500亩
                'has_migrant_experience': False,
                'tech_adoption': float(np.clip(self.rng.normal(0.7, 0.15), 0.4, 0.95)),
                'is_enterprise': True
            })
        
        self.extracted_features = {
            'return_migrant': return_migrants,
            'family_farm': family_farms,
            'cooperative': cooperatives,
            'young_household': return_migrants[:n_per_category//2]  # 复用部分数据
        }
        
        logger.info(f"合成特征库生成完成: {sum(len(v) for v in self.extracted_features.values())}条记录")
        return self.extracted_features


def main():
    """主函数：提取特征并导出"""
    logging.basicConfig(level=logging.INFO)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    cfps_path = project_root / '校准用数据源' / 'CFPS'
    output_path = project_root / 'config' / 'cfps_feature_library.json'
    
    extractor = CFPSFeatureExtractor(str(cfps_path))
    
    # 优先使用预处理好的汇总数据（推荐）
    if extractor.preprocessed_file.exists():
        logger.info(f"发现预处理数据文件: {extractor.preprocessed_file}")
        features = extractor.extract_from_preprocessed(years=[2018, 2020, 2022])
        
        total_samples = sum(len(v) for v in features.values())
        if total_samples >= 100:
            logger.info(f"从预处理数据提取成功: {total_samples}个样本")
        else:
            logger.warning(f"预处理数据样本不足({total_samples})，补充合成数据")
            extractor.generate_synthetic_library(100)
    
    # 备选：尝试从原始dta文件提取
    elif cfps_path.exists():
        logger.info("尝试从原始dta文件提取...")
        features = extractor.extract_all_years([2014, 2016, 2018, 2020])
        
        total_samples = sum(len(v) for v in features.values())
        if total_samples < 100:
            logger.warning(f"真实样本不足({total_samples}个)，补充合成数据")
            extractor.generate_synthetic_library(100)
    else:
        logger.warning(f"CFPS数据路径不存在: {cfps_path}")
        logger.info("生成合成特征库...")
        extractor.generate_synthetic_library(100)
    
    # 导出
    extractor.export_feature_library(str(output_path))
    
    print(f"\n特征库已保存到: {output_path}")
    print("\n各类别样本量:")
    for category, samples in extractor.extracted_features.items():
        print(f"  {category}: {len(samples)}")


if __name__ == '__main__':
    main()
