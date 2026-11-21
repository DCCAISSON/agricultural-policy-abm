"""
微观数据驱动的智能体特征库
==========================================
从CFPS和NFPS户级数据提取农户、村集体特征，用于初始化ABM智能体

@theory 真正的ABM应该是微观数据驱动的
- 每个智能体都有来自真实数据的异质性特征
- 不是从分布采样，而是直接从微观数据Bootstrap采样
- 保留数据中的相关性结构（土地-收入-生计策略的联合分布）

@data_sources
- CFPS 2010: 基准年微观数据（农户特征）
- NFPS 2009-2015: 土地流转、收入结构

@output
- config/agent_feature_library.json: 户级特征库
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
from core.cfps_livelihood_alignment import load_canonical_livelihood_map, normalize_record_id

# ============== 配置 ==============
BASE_PATH = Path(r".\校准用数据源")
CFPS_PATH = BASE_PATH / "CFPS"
NFPS_PATH = BASE_PATH / "全国农村固定观察点"
OUTPUT_PATH = Path(r".\config")

# 区域映射
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "校准用数据源"
CFPS_PATH = BASE_PATH / "CFPS"
NFPS_PATH = BASE_PATH / "全国农村固定观察点"
OUTPUT_PATH = PROJECT_ROOT / "config"

REGION_MAPPING = {
    '东部': [11, 12, 13, 31, 32, 33, 35, 37, 44, 46],
    '中部': [14, 34, 36, 41, 42, 43],
    '西部': [50, 51, 52, 53, 54, 61, 62, 63, 64, 65, 45, 15],
    '东北': [21, 22, 23]
}

def get_region(provcd):
    """省份代码转区域"""
    try:
        prov = int(float(provcd))
    except:
        return '其他'
    for region, codes in REGION_MAPPING.items():
        if prov in codes:
            return region
    return '其他'

def get_numeric(df, var_name, default=np.nan):
    """安全获取数值列，处理CFPS缺失值代码"""
    if var_name is None or var_name not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    val = pd.to_numeric(df[var_name], errors='coerce')
    # CFPS缺失值代码：-8=不适用, -1=不知道, -2=拒绝回答, -9=缺失
    val = val.replace([-8, -1, -2, -9], np.nan)
    val = val.where(val >= 0, np.nan)  # 负值视为缺失
    return val

def classify_livelihood(ratio):
    """生计类型分类"""
    if pd.isna(ratio):
        return None
    elif ratio > 0.5:
        return 'Pure_Farmer'
    elif ratio > 0.2:
        return 'Pluriactive'
    else:
        return 'Non_Farmer'

def classify_scale(land_mu):
    """农户规模分类"""
    if pd.isna(land_mu) or land_mu <= 0:
        return 'Smallholder'
    elif land_mu < 5:
        return 'Smallholder'
    elif land_mu < 15:
        return 'Medium_Farmer'
    elif land_mu < 50:
        return 'Large_Farmer'
    else:
        return 'Family_Farm'

def extract_cfps_2010_features():
    """
    从CFPS 2010提取农户微观特征
    
    提取变量：
    - 土地面积 (亩)
    - 农业收入、总收入
    - 生计类型
    - 区域
    - 家庭规模
    - 劳动力数量
    """
    print("=" * 70)
    print("提取 CFPS 2010 农户微观特征")
    print("=" * 70)
    
    # 读取家庭经济数据
    famecon_path = CFPS_PATH / "2010" / "[CFPS Public Data] CFPS2010 in STATA (Chinese)" / "cfps2010famecon_201906.dta"
    if not famecon_path.exists():
        print(f"  [错误] 文件不存在: {famecon_path}")
        return None
    
    df = pd.read_stata(famecon_path, convert_categoricals=False)
    print(f"  原始样本: {len(df)}")
    
    # 筛选农村样本
    df = df[df['urban'] == 0].copy()
    print(f"  农村样本: {len(df)}")
    
    # 提取特征
    features = pd.DataFrame()
    features['fid'] = df['fid']
    
    # 区域
    if 'provcd' in df.columns:
        features['region'] = df['provcd'].apply(get_region)
    else:
        features['region'] = '其他'
    
    # 收入变量（统一口径）
    features['agri_income'] = get_numeric(df, 'inc_agri')
    features['total_income'] = get_numeric(df, 'faminc')  # 家庭总收入
    wage_income = None
    for wage_col in ('inc_wage', 'wage_income', 'foperate_2', 'foperate'):
        if wage_col in df.columns:
            wage_income = get_numeric(df, wage_col)
            break
    if wage_income is None:
        wage_income = pd.Series(0.0, index=df.index)
    features['wage_income'] = wage_income
    features['net_income'] = get_numeric(df, 'faminc_net')

    component_income = pd.concat(
        [
            features['agri_income'].fillna(0.0),
            features['wage_income'].fillna(0.0),
        ],
        axis=1,
    ).sum(axis=1, min_count=1)
    features['total_income'] = features['total_income'].where(features['total_income'].notna(), features['net_income'])
    features['total_income'] = features['total_income'].where(
        features['total_income'].notna(),
        component_income.where(component_income > 0),
    )
    features['total_income'] = features['total_income'].clip(lower=0)
    features['agri_income'] = features['agri_income'].clip(lower=0)
    features['agri_income'] = np.minimum(features['agri_income'], features['total_income'])
    features['net_income'] = features['net_income'].clip(lower=0)
    features['net_income'] = features['net_income'].where(features['net_income'].notna(), features['total_income'])
    features['net_income'] = np.minimum(features['net_income'], features['total_income'])
    
    # 土地变量
    # 2010年土地面积需要从多个字段求和
    # 注意：CFPS用-8表示缺失
    land_cols = [f'fk201_a_{i}' for i in range(1, 7)]
    existing_land_cols = [c for c in land_cols if c in df.columns]
    if existing_land_cols:
        land_sum = pd.Series(0.0, index=df.index)
        for col in existing_land_cols:
            val = get_numeric(df, col, 0)
            land_sum += val.fillna(0)
        features['land_area_mu'] = land_sum
        # 如果所有字段都是缺失，则标记为NaN
        all_missing = df[existing_land_cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce').isin([-8, -1, -2, -9, np.nan]).all(), axis=1
        )
        features.loc[all_missing, 'land_area_mu'] = np.nan
    else:
        features['land_area_mu'] = np.nan
    
    # 土地流转
    rent_out_cols = [f'fk204_a_{i}' for i in range(1, 7)]
    existing_rent_out = [c for c in rent_out_cols if c in df.columns]
    if existing_rent_out:
        rent_out_sum = pd.Series(0.0, index=df.index)
        for col in existing_rent_out:
            val = get_numeric(df, col, 0)
            rent_out_sum += val.fillna(0)
        features['land_rent_out_area'] = rent_out_sum
    else:
        features['land_rent_out_area'] = 0
    
    rent_in_cols = [f'fk203_a_{i}' for i in range(1, 7)]
    existing_rent_in = [c for c in rent_in_cols if c in df.columns]
    if existing_rent_in:
        rent_in_sum = pd.Series(0.0, index=df.index)
        for col in existing_rent_in:
            val = get_numeric(df, col, 0)
            rent_in_sum += val.fillna(0)
        features['land_rent_in_area'] = rent_in_sum
    else:
        features['land_rent_in_area'] = 0
    
    # 家庭人口
    if 'familysize' in df.columns:
        features['family_size'] = get_numeric(df, 'familysize')
    else:
        features['family_size'] = 4  # 默认值
    
    # 计算生计类型
    features['agri_ratio'] = features['agri_income'] / features['total_income'].replace(0, np.nan)
    features['agri_ratio'] = features['agri_ratio'].clip(0, 1)
    features['livelihood'] = features['agri_ratio'].apply(classify_livelihood)
    
    # 农户规模
    features['farmer_scale'] = features['land_area_mu'].apply(classify_scale)
    
    # 流转状态
    features['is_rent_out'] = (features['land_rent_out_area'] > 0).astype(int)
    features['is_rent_in'] = (features['land_rent_in_area'] > 0).astype(int)
    
    # 有效样本（有生计类型）
    valid = features['livelihood'].notna()
    features = features[valid].copy()
    canonical_map = load_canonical_livelihood_map(year=2010)
    if canonical_map:
        fid_keys = features['fid'].apply(normalize_record_id)
        aligned_labels = fid_keys.map(canonical_map)
        matched = int(aligned_labels.notna().sum())
        changed = int(((aligned_labels.notna()) & (aligned_labels != features['livelihood'])).sum())
        features.loc[aligned_labels.notna(), 'livelihood'] = aligned_labels[aligned_labels.notna()]
        print(f"  Canonical CFPS alignment: matched={matched}, changed={changed}")
    known_livelihoods = {'Pure_Farmer', 'Pluriactive', 'Non_Farmer'}
    excluded_unknown = int((~features['livelihood'].isin(known_livelihoods)).sum())
    if excluded_unknown:
        print(f"  Excluded non-canonical livelihoods after alignment: {excluded_unknown}")
    features = features[features['livelihood'].isin(known_livelihoods)].copy()
    print(f"  有效样本: {len(features)}")
    
    # 统计
    print(f"\n  生计类型分布:")
    for lv, pct in features['livelihood'].value_counts(normalize=True).items():
        print(f"    {lv}: {pct:.1%}")
    
    print(f"\n  区域分布:")
    for region, pct in features['region'].value_counts(normalize=True).items():
        print(f"    {region}: {pct:.1%}")
    
    print(f"\n  土地面积: 均值={features['land_area_mu'].mean():.1f}亩, 中位数={features['land_area_mu'].median():.1f}亩")
    
    return features

def extract_nfps_features():
    """
    从NFPS 2009-2015提取补充特征
    主要用于：
    - 更大样本量的土地分布
    - 更详细的收入结构
    """
    print("\n" + "=" * 70)
    print("提取 NFPS 2009-2015 补充特征")
    print("=" * 70)
    
    nfps_file = NFPS_PATH / "2009-2015household.csv"
    if not nfps_file.exists():
        print(f"  [警告] NFPS文件不存在: {nfps_file}")
        return None
    
    df = pd.read_csv(nfps_file, encoding='utf-8')
    print(f"  原始样本: {len(df)}")
    
    # 筛选2010年作为基准
    if 'year' in df.columns:
        df_2010 = df[df['year'] == 2010].copy()
        print(f"  2010年样本: {len(df_2010)}")
    else:
        df_2010 = df.copy()
    
    # 提取土地分布（NFPS样本更大）
    if '承包总面积' in df_2010.columns:
        land_stats = {
            'mean': df_2010['承包总面积'].mean(),
            'median': df_2010['承包总面积'].median(),
            'std': df_2010['承包总面积'].std(),
            'percentiles': {
                '25': df_2010['承包总面积'].quantile(0.25),
                '50': df_2010['承包总面积'].quantile(0.50),
                '75': df_2010['承包总面积'].quantile(0.75),
                '90': df_2010['承包总面积'].quantile(0.90),
            }
        }
        print(f"\n  土地分布 (NFPS):")
        print(f"    均值: {land_stats['mean']:.1f}亩")
        print(f"    中位数: {land_stats['median']:.1f}亩")
        return land_stats
    
    return None

def build_agent_feature_library(cfps_features):
    """
    构建智能体特征库
    
    特征库结构：
    {
        'metadata': {...},
        'farmers': [
            {
                'id': int,
                'region': str,
                'land_area_mu': float,
                'agri_income': float,
                'total_income': float,
                'livelihood': str,
                'farmer_scale': str,
                ...
            },
            ...
        ],
        'statistics': {...}
    }
    """
    print("\n" + "=" * 70)
    print("构建智能体特征库")
    print("=" * 70)
    
    # 清理数据
    features = cfps_features.copy()
    known_livelihoods = {'Pure_Farmer', 'Pluriactive', 'Non_Farmer'}
    features = features[features['livelihood'].isin(known_livelihoods)].copy()
    
    # 替换NaN为合理默认值
    features['land_area_mu'] = features['land_area_mu'].fillna(features['land_area_mu'].median())
    features['agri_income'] = features['agri_income'].fillna(0).clip(lower=0)
    features['total_income'] = features['total_income'].fillna(features['total_income'].median()).clip(lower=0)
    features['net_income'] = features['net_income'].fillna(features['total_income']).clip(lower=0)
    features['net_income'] = np.minimum(features['net_income'], features['total_income'])
    features['agri_income'] = np.minimum(features['agri_income'], features['total_income'])
    features['agri_ratio'] = (
        features['agri_income'] / features['total_income'].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    features['family_size'] = features['family_size'].fillna(4)
    features['land_rent_out_area'] = features['land_rent_out_area'].fillna(0)
    features['land_rent_in_area'] = features['land_rent_in_area'].fillna(0)
    
    # 转换为字典列表
    farmers = []
    for idx, row in features.iterrows():
        farmer = {
            'id': int(row['fid']) if pd.notna(row['fid']) else idx,
            'region': row['region'],
            'land_area_mu': round(float(row['land_area_mu']), 2),
            'agri_income': round(float(row['agri_income']), 0),
            'total_income': round(float(row['total_income']), 0),
            'net_income': round(float(row['net_income']), 0) if pd.notna(row['net_income']) else None,
            'agri_ratio': round(float(row['agri_ratio']), 4) if pd.notna(row['agri_ratio']) else 0,
            'livelihood': row['livelihood'],
            'farmer_scale': row['farmer_scale'],
            'family_size': int(row['family_size']),
            'is_rent_out': int(row['is_rent_out']),
            'is_rent_in': int(row['is_rent_in']),
            'land_rent_out_area': round(float(row['land_rent_out_area']), 2),
            'land_rent_in_area': round(float(row['land_rent_in_area']), 2),
        }
        farmers.append(farmer)
    
    # 计算统计量
    statistics = {
        'n_total': len(farmers),
        'livelihood_distribution': features['livelihood'].value_counts(normalize=True).to_dict(),
        'scale_distribution': features['farmer_scale'].value_counts(normalize=True).to_dict(),
        'region_distribution': features['region'].value_counts(normalize=True).to_dict(),
        'land_area': {
            'mean': round(features['land_area_mu'].mean(), 2),
            'median': round(features['land_area_mu'].median(), 2),
            'std': round(features['land_area_mu'].std(), 2),
        },
        'income': {
            'agri_mean': round(features['agri_income'].mean(), 0),
            'total_mean': round(features['total_income'].mean(), 0),
        },
        'transfer_rate': {
            'rent_out': round(features['is_rent_out'].mean(), 4),
            'rent_in': round(features['is_rent_in'].mean(), 4),
        }
    }
    
    # 构建完整库
    library = {
        'metadata': {
            'version': '1.0',
            'data_source': 'CFPS 2010 截面数据（仅农村样本）',
            'created': pd.Timestamp.now().isoformat(),
            'description': '微观数据驱动的农户特征库，用于ABM智能体初始化',
            'sample_universe': 'CFPS 2010 rural households',
            'years_used': [2010],
            'unknown_handling': 'excluded from ABM baseline; initialization uses known-only livelihood shares',
            'valid_income_filter': 'retain canonical livelihoods and sanitize income semantics so net_income <= total_income',
            'variables': {
                'land_area_mu': '土地面积（亩）',
                'agri_income': '农业收入（元）',
                'total_income': '家庭总收入（元）',
                'agri_ratio': '农业收入占比',
                'livelihood': '生计类型 (Pure_Farmer/Pluriactive/Non_Farmer)',
                'farmer_scale': '农户规模 (Smallholder/Medium_Farmer/Large_Farmer/Family_Farm)',
                'region': '区域 (东部/中部/西部/东北)',
            },
            'livelihood_thresholds': {
                'Pure_Farmer': '农业收入占比 > 50%',
                'Pluriactive': '农业收入占比 20%-50%',
                'Non_Farmer': '农业收入占比 < 20%',
            }
        },
        'statistics': statistics,
        'farmers': farmers
    }
    
    print(f"  特征库大小: {len(farmers)} 农户")
    print(f"  生计分布: {statistics['livelihood_distribution']}")
    
    return library

def save_feature_library(library, filename='agent_feature_library.json'):
    """保存特征库"""
    output_file = OUTPUT_PATH / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"\n特征库已保存: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024:.1f} KB")

def main():
    print("=" * 70)
    print("微观数据驱动的智能体特征库构建")
    print("=" * 70)
    
    # 1. 提取CFPS 2010微观特征
    cfps_features = extract_cfps_2010_features()
    if cfps_features is None:
        print("[错误] CFPS特征提取失败")
        return
    
    # 2. 提取NFPS补充特征（可选）
    nfps_stats = extract_nfps_features()
    
    # 3. 构建特征库
    library = build_agent_feature_library(cfps_features)
    
    # 4. 如果有NFPS数据，添加补充统计
    if nfps_stats:
        library['nfps_reference'] = {
            'land_distribution': nfps_stats,
            'note': 'NFPS数据提供更大样本量的土地分布参考'
        }
    
    # 5. 保存
    save_feature_library(library)
    
    # 6. 打印使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("""
在ABM中使用微观数据驱动初始化：

from core.microdata_agent_initializer import MicrodataAgentInitializer

# 加载特征库
initializer = MicrodataAgentInitializer('config/agent_feature_library.json')

# Bootstrap采样初始化农户
for i in range(n_farmers):
    features = initializer.sample_farmer_features(
        region='东部',  # 可选：指定区域
        livelihood='Pure_Farmer'  # 可选：指定生计类型
    )
    farmer = FarmerAgent(
        unique_id=i,
        model=model,
        land_area=features['land_area_mu'],
        initial_capital=features['total_income'],
        livelihood_strategy=features['livelihood'],
        ...
    )
""")

if __name__ == "__main__":
    main()
