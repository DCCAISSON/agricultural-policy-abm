# -*- coding: utf-8 -*-
"""
从CLES数据构建综合指标
- 耕作能力综合指标
- 技术水平指标
- 生计策略分类
- 收入分位数

说明：CLES数据中没有这些现成的综合指标，需要从原始变量中构建
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# --------- 收入计算辅助常量与工具 ----------
INCOME_SENTINELS = {
    99999999, 9999999, 999999, 99999, 9999,
    88888888, 8888888, 888888, 777777, 666666,
    -999999, -99999, -9999, -999, -888, -777
}


def _clean_income_series(df: pd.DataFrame, column: str, index) -> pd.Series:
    """
    统一的收入列清洗逻辑：
    - 将列转换为浮点数
    - 处理CLES中常见的缺失/哑值编码（999999、-999等）
    - 回填缺失为0
    """
    if column not in df.columns:
        return pd.Series(0.0, index=index)
    series = pd.to_numeric(df[column], errors='coerce')
    series = series.replace(list(INCOME_SENTINELS), np.nan)
    return series.fillna(0.0)


def _sum_series(df: pd.DataFrame, columns, index) -> pd.Series:
    total = pd.Series(0.0, index=index)
    for col in columns:
        if isinstance(col, str):
            total += _clean_income_series(df, col, index)
        else:
            # 允许传入 (prefix, start, end) 的元组
            prefix, start, end = col
            for i in range(start, end + 1):
                col_name = f"{prefix}{i}"
                total += _clean_income_series(df, col_name, index)
    return total


def compute_income_components(df: pd.DataFrame) -> dict:
    """
    按照国家统计局 / CLES 指南计算家庭各类收入。
    返回:
        {
            'agricultural_income': Series,
            'non_ag_income': Series,
            'transfer_income': Series,
            'property_income': Series,
            'total_income': Series
        }
    """
    index = df.index

    # ---- 农业收入：种植 + 畜牧（水产） ----
    crop_revenue = pd.Series(0.0, index=index)
    crop_cost = pd.Series(0.0, index=index)
    cost_components = ['e108', 'e109', 'e110', 'e111', 'e112', 'e114', 'e117']

    for i in range(1, 7):
        qty = _clean_income_series(df, f'e105a{i}', index)
        price = _clean_income_series(df, f'e106a{i}', index)
        crop_revenue += qty * price

        area = _clean_income_series(df, f'e101a{i}', index)
        unit_cost = pd.Series(0.0, index=index)
        for comp in cost_components:
            unit_cost += _clean_income_series(df, f'{comp}a{i}', index)
        crop_cost += unit_cost * area

    crop_income = (crop_revenue - crop_cost).clip(lower=0.0)

    livestock_income = pd.Series(0.0, index=index)
    for i in range(1, 7):
        gross = _clean_income_series(df, f'e206a{i}', index)
        cost = _clean_income_series(df, f'e205a{i}', index)
        livestock_income += (gross - cost)
    livestock_income = livestock_income.clip(lower=0.0)

    agricultural_income = crop_income + livestock_income

    # ---- 非农收入：工资 + 自营 ----
    wage_income = _sum_series(df, [(f'a215a', 1, 9)], index)
    self_emp_gross = _clean_income_series(df, 'e403', index)
    self_emp_cost = _clean_income_series(df, 'e404', index)
    self_emp_income = (self_emp_gross - self_emp_cost).clip(lower=0.0)
    non_ag_income = wage_income + self_emp_income

    # ---- 转移性收入：B2 + 养老金/汇款/救助 ----
    b2_cols = [f'b20{i}' if i < 10 else 'b210' for i in range(1, 11)]
    # b210 有 a/b 两列，分别加上
    b2_cols.append('b210a')
    b2_cols.append('b210b')
    transfer_income = _sum_series(df, b2_cols, index)
    transfer_income += _sum_series(
        df,
        ['b105', 'b107', 'b109', 'b110'],
        index
    )

    # ---- 财产性收入：资产租赁、股息、征地补偿、宅基地流转 ----
    property_income = _sum_series(
        df,
        ['b101', 'b102', 'b103', 'b104', 'c309'],
        index
    )

    total_income = agricultural_income + non_ag_income + transfer_income + property_income

    # 对于缺失或极端值做一次温和Winsorize，避免分位数偏斜
    positive_mask = total_income > 0
    if positive_mask.any():
        upper = total_income[positive_mask].quantile(0.995)
        total_income = total_income.clip(lower=0.0, upper=upper if upper > 0 else None)

    return {
        'agricultural_income': agricultural_income,
        'non_ag_income': non_ag_income,
        'transfer_income': transfer_income,
        'property_income': property_income,
        'total_income': total_income
    }


def build_farming_ability_index(df):
    """
    从CLES数据构建耕作能力综合指标
    
    使用变量：
    - a206a1: 教育程度（年）
    - a209a1: 农业技术培训（0/1）
    - a208a1: 非农职业培训（0/1）
    """
    # 1. 标准化教育程度（0-20年 -> 0-1）
    education = df['a206a1'].fillna(df['a206a1'].median())
    education_max = education.quantile(0.95)  # 使用95分位数作为最大值
    education_normalized = (education - 0) / max(education_max, 1)
    education_normalized = education_normalized.clip(0, 1)
    
    # 2. 培训加分（农业培训权重更高）
    agri_training = df['a209a1'].fillna(0)
    non_agri_training = df['a208a1'].fillna(0)
    
    training_bonus = (
        agri_training * 0.3 +  # 农业培训：+0.3
        non_agri_training * 0.1    # 非农培训：+0.1
    )
    
    # 3. 综合指标（0-1范围）
    farming_ability = education_normalized * 0.7 + training_bonus
    farming_ability = farming_ability.clip(0, 1)
    
    # 4. 映射到模型范围 [0.8, 1.2]
    farming_ability_scaled = 0.8 + (farming_ability * 0.4)
    
    return farming_ability_scaled


def build_technology_level_index(df):
    """
    从CLES数据构建技术水平指标
    
    基于：
    - 机械化水平（机械费用 e111a1）
    - 技术采用（农业技术服务类型 e301a1-e301a6）
    
    根据问卷：
    - e111a1: 机械作业费用（元/亩）
    - e301a1-e301a6: 您获过的农业技术服务类型（1-18种技术类型）
    
    技术采用变量说明：
    - e301a1-e301a6是数值型变量
    - 每个变量只能记录一种技术类型（1-18）
    - 如果值为0，表示没采取
    - 如果值为1，表示采取了1类技术（良种服务）
    - 如果值为2，表示采取了2类技术（测土施肥等技术）
    - 等等
    """
    tech_level = pd.Series([1.0] * len(df))  # 基础技术水平
    
    # 1. 机械化水平（基于机械费用）
    if 'e111a1' in df.columns:  # 机械作业费用（元/亩）
        machinery_cost = df['e111a1'].fillna(0)
        # 标准化机械费用（使用95分位数作为最大值）
        machinery_max = machinery_cost.quantile(0.95)
        if machinery_max > 0:
            machinery_level = (machinery_cost / machinery_max).clip(0, 1)
            tech_level += machinery_level * 0.10  # 机械化可提升10%
    
    # 2. 技术采用（基于农业技术服务类型 e301a1-e301a6）
    # 根据用户说明：每个e301a变量是数值型，只能记录一种技术类型（1-18）
    # 如果值为0，表示没采取；如果值>0，表示采取了对应类型的技术
    tech_adoption_cols = [f'e301a{i}' for i in range(1, 7) if f'e301a{i}' in df.columns]
    
    if tech_adoption_cols:
        # 计算技术采用数量（统计有多少个e301a变量值>0，即采用了多少种不同的技术）
        tech_count = pd.Series([0] * len(df))
        
        for col in tech_adoption_cols:
            tech_data = df[col].fillna(0)
            # 如果值>0，表示采用了技术（无论是什么类型）
            tech_count += (tech_data > 0).astype(int)
        
        # 标准化技术采用数量（最多6种技术，使用最大值作为基准）
        tech_count_max = tech_count.quantile(0.95) if tech_count.max() > 0 else 1
        if tech_count_max > 0:
            tech_adoption_normalized = (tech_count / tech_count_max).clip(0, 1)
            tech_level += tech_adoption_normalized * 0.10  # 技术采用可提升10%
    
    # 限制在合理范围 [1.0, 1.2]
    tech_level = tech_level.clip(1.0, 1.2)
    
    return tech_level


def build_livelihood_strategy(df):
    """
    基于CLES问卷编码计算农业/非农/转移/财产收入，并据此判定生计策略。
    收入构成遵循国家统计局农户调查指标：
        Y_total = Y_ag + Y_non-ag + Y_transfer + Y_property
    """
    income_components = compute_income_components(df)
    total_income = income_components['total_income']
    agri_income = income_components['agricultural_income']

    denominator = np.where(total_income > 0, total_income, 1e-6)
    agri_income_ratio = (agri_income / denominator).clip(0, 1)

    # @fix: 使用与模型一致的阈值（70%/30%），而不是95%/5%
    # 这样可以获得更合理的分布，符合中国农村实际情况
    livelihood_strategy = pd.Series(['PLURIACTIVE'] * len(df), index=df.index)
    livelihood_strategy[agri_income_ratio > 0.70] = 'PURE_FARMER'  # 从0.95改为0.70
    livelihood_strategy[agri_income_ratio < 0.30] = 'NON_FARMER'   # 从0.05改为0.30

    return livelihood_strategy, agri_income_ratio


def build_income_quartiles(df):
    """
    以 compute_income_components 生成的总收入为基础划分收入分位数。
    """
    income_components = compute_income_components(df)
    total_income = income_components['total_income']

    total_income = total_income.fillna(0).clip(lower=0)
    total_income = pd.Series(total_income.values, index=df.index)

    positive_income = total_income[total_income > 0]
    if len(positive_income) > 0:
        trimmed = positive_income[positive_income < positive_income.quantile(0.99)]
        q1_threshold = float(trimmed.quantile(0.25))
        q2_threshold = float(trimmed.quantile(0.50))
        q3_threshold = float(trimmed.quantile(0.75))
        q4_threshold = float(trimmed.max())
    else:
        q1_threshold, q2_threshold, q3_threshold, q4_threshold = 5000, 10000, 20000, 50000

    income_quartile = pd.Series(['Q2'] * len(df), index=df.index)
    income_quartile[total_income < q1_threshold] = 'Q1'
    income_quartile[(total_income >= q1_threshold) & (total_income < q2_threshold)] = 'Q2'
    income_quartile[(total_income >= q2_threshold) & (total_income < q3_threshold)] = 'Q3'
    income_quartile[total_income >= q3_threshold] = 'Q4'

    quartile_thresholds = {
        'q1_threshold': q1_threshold,
        'q2_threshold': q2_threshold,
        'q3_threshold': q3_threshold,
        'q4_threshold': q4_threshold
    }

    return income_quartile, quartile_thresholds


def explore_cles_variables(df):
    """探索CLES数据中的变量，帮助查找所需变量"""
    print("\n" + "=" * 80)
    print("探索CLES数据变量")
    print("=" * 80)
    
    # 显示所有列名（前50个）
    print(f"\n数据总列数: {len(df.columns)}")
    print(f"前50个列名:")
    for i, col in enumerate(df.columns[:50]):
        print(f"  {i+1}. {col}")
    
    # 1. 查找收入相关变量（更广泛的搜索）
    print("\n1. 收入相关变量:")
    income_keywords = ['收入', 'income', '收益', '利润', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']
    income_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(kw in col_str for kw in income_keywords):
            income_cols.append(col)
    
    print(f"   找到 {len(income_cols)} 个可能的收入相关变量")
    for i, col in enumerate(income_cols[:20]):
        non_null = df[col].notna().sum()
        if non_null > 0:
            try:
                mean_val = df[col].mean()
                median_val = df[col].median()
                print(f"   {i+1}. {col}: {non_null} 个非空值, 均值={mean_val:.2f}, 中位数={median_val:.2f}")
            except:
                print(f"   {i+1}. {col}: {non_null} 个非空值 (非数值型)")
    
    # 2. 查找技术相关变量
    print("\n2. 技术相关变量:")
    tech_keywords = ['技术', '品种', '方法', '创新', '培训', 'e1', 'e2', 'e3']
    tech_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(kw in col_str for kw in tech_keywords):
            tech_cols.append(col)
    
    print(f"   找到 {len(tech_cols)} 个可能的技术相关变量")
    for i, col in enumerate(tech_cols[:20]):
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"   {i+1}. {col}: {non_null} 个非空值")
    
    # 3. 查找能力相关变量
    print("\n3. 能力相关变量:")
    ability_keywords = ['教育', '培训', '经验', '能力', '技能', 'a20', 'a21']
    ability_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(kw in col_str for kw in ability_keywords):
            ability_cols.append(col)
    
    print(f"   找到 {len(ability_cols)} 个可能的能力相关变量")
    for i, col in enumerate(ability_cols[:20]):
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"   {i+1}. {col}: {non_null} 个非空值")
    
    # 4. 查找所有c系列变量（收入通常在c系列）
    print("\n4. C系列变量（可能包含收入）:")
    c_cols = [col for col in df.columns if str(col).startswith('c')]
    print(f"   找到 {len(c_cols)} 个C系列变量")
    for i, col in enumerate(c_cols[:30]):
        non_null = df[col].notna().sum()
        if non_null > 0:
            try:
                mean_val = df[col].mean()
                print(f"   {i+1}. {col}: {non_null} 个非空值, 均值={mean_val:.2f}")
            except:
                print(f"   {i+1}. {col}: {non_null} 个非空值")
    
    return {
        'income_cols': income_cols,
        'tech_cols': tech_cols,
        'ability_cols': ability_cols,
        'c_cols': c_cols
    }


def build_all_composite_indicators(data_dir: Path, output_file: str = "composite_indicators.csv", explore: bool = True):
    """
    从CLES数据构建所有综合指标
    """
    print("=" * 80)
    print("从CLES数据构建综合指标")
    print("=" * 80)
    
    # 读取数据
    household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.xlsx"
    
    if not household_file.exists():
        # 尝试CSV格式
        household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.csv"
    
    if not household_file.exists():
        print(f"错误: 找不到数据文件")
        return None
    
    print(f"\n读取数据文件: {household_file}")
    try:
        if household_file.suffix == '.xlsx':
            df = pd.read_excel(household_file, sheet_name=0)
        else:
            df = pd.read_csv(household_file, encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    # 筛选2020年数据
    if 'year' in df.columns:
        df = df[df['year'] == 2020].copy()
        print(f"筛选2020年数据: {len(df)} 行")
    else:
        print(f"未找到year列，使用全部数据: {len(df)} 行")
    
    # 探索变量（可选）
    if explore:
        explore_cles_variables(df)
    
    # 检查必需变量是否存在
    required_vars = ['a206a1', 'a209a1', 'a208a1', 'e111a1', 'c107a', 'e103a1']
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        print(f"\n[警告] 以下变量缺失: {missing_vars}")
        print("   将使用替代方法或默认值")
    
    # 1. 构建耕作能力综合指标
    print("\n" + "=" * 80)
    print("1. 构建耕作能力综合指标")
    print("=" * 80)
    try:
        df['farming_ability_composite'] = build_farming_ability_index(df)
        print(f"   [OK] 耕作能力范围: [{df['farming_ability_composite'].min():.2f}, {df['farming_ability_composite'].max():.2f}]")
        print(f"   [OK] 耕作能力均值: {df['farming_ability_composite'].mean():.2f}")
        print(f"   [OK] 耕作能力标准差: {df['farming_ability_composite'].std():.2f}")
    except Exception as e:
        print(f"   [ERROR] 构建失败: {e}")
        df['farming_ability_composite'] = 1.0  # 默认值
    
    # 2. 构建技术水平指标
    print("\n" + "=" * 80)
    print("2. 构建技术水平指标")
    print("=" * 80)
    try:
        df['technology_level'] = build_technology_level_index(df)
        print(f"   [OK] 技术水平范围: [{df['technology_level'].min():.2f}, {df['technology_level'].max():.2f}]")
        print(f"   [OK] 技术水平均值: {df['technology_level'].mean():.2f}")
        print(f"   [OK] 技术水平标准差: {df['technology_level'].std():.2f}")
    except Exception as e:
        print(f"   [ERROR] 构建失败: {e}")
        df['technology_level'] = 1.0  # 默认值
    
    # 3. 构建生计策略分类
    print("\n" + "=" * 80)
    print("3. 构建生计策略分类")
    print("=" * 80)
    try:
        df['livelihood_strategy'], df['agri_income_ratio'] = build_livelihood_strategy(df)
        print(f"   [OK] 生计策略分布:")
        strategy_counts = df['livelihood_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"      {strategy}: {count} ({count/len(df)*100:.1f}%)")
        print(f"   [OK] 农业收入占比均值: {df['agri_income_ratio'].mean():.3f}")
        print(f"   [OK] 农业收入占比中位数: {df['agri_income_ratio'].median():.3f}")
    except Exception as e:
        print(f"   [ERROR] 构建失败: {e}")
        df['livelihood_strategy'] = 'PLURIACTIVE'  # 默认值
        df['agri_income_ratio'] = 0.5  # 默认值
    
    # 4. 构建收入分位数
    print("\n" + "=" * 80)
    print("4. 构建收入分位数")
    print("=" * 80)
    try:
        df['income_quartile'], quartile_thresholds = build_income_quartiles(df)
        print(f"   [OK] 收入分位数分布:")
        quartile_counts = df['income_quartile'].value_counts()
        for quartile, count in quartile_counts.items():
            print(f"      {quartile}: {count} ({count/len(df)*100:.1f}%)")
        print(f"   [OK] 分位数阈值:")
        for key, value in quartile_thresholds.items():
            print(f"      {key}: {value:.2f}")
    except Exception as e:
        print(f"   [ERROR] 构建失败: {e}")
        df['income_quartile'] = 'q2'  # 默认值
    
    # 保存结果
    output_path = data_dir / output_file
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n" + "=" * 80)
        print(f"结果已保存到: {output_path}")
        print("=" * 80)
        
        # 保存构建的指标到单独文件（只包含ID和指标）
        indicator_cols = ['farming_ability_composite', 'technology_level', 
                         'livelihood_strategy', 'agri_income_ratio', 'income_quartile']
        if 'id' in df.columns:
            indicator_df = df[['id'] + indicator_cols].copy()
        else:
            indicator_df = df[indicator_cols].copy()
        
        indicator_file = data_dir / "composite_indicators_only.csv"
        indicator_df.to_csv(indicator_file, index=False, encoding='utf-8-sig')
        print(f"指标文件已保存到: {indicator_file}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='从CLES数据构建综合指标')
    parser.add_argument('--no-explore', action='store_true', help='不探索变量')
    parser.add_argument('--output', type=str, default='composite_indicators.csv', help='输出文件名')
    
    args = parser.parse_args()
    
    data_dir = _project_root / "校准用数据源"
    df = build_all_composite_indicators(data_dir, output_file=args.output, explore=not args.no_explore)
    
    if df is not None:
        print("\n构建完成！")

