# -*- coding: utf-8 -*-
"""
从CLES数据中提取非农劳动分配信息
用于校准模型参数
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def extract_non_farm_labor_data(data_dir: Path):
    """
    从CLES数据中提取非农劳动分配信息
    """
    print("=" * 80)
    print("提取CLES 2020年非农劳动分配数据")
    print("=" * 80)
    
    household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.xlsx"
    
    if not household_file.exists():
        print(f"Error: File not found: {household_file}")
        return None
    
    try:
        # 读取数据
        print(f"Reading: {household_file}")
        df = pd.read_excel(household_file, sheet_name=0)
        
        # 筛选2020年数据
        if 'year' in df.columns:
            df_2020 = df[df['year'] == 2020].copy()
            print(f"2020 household data: {len(df_2020)} records")
        else:
            df_2020 = df.copy()
            print(f"Household data (no year filter): {len(df_2020)} records")
        
        # 查找非农劳动相关列
        # 根据CLES数据结构，非农劳动可能在以下列中：
        # - c系列：收入相关
        # - d系列：支出相关
        # - e系列：土地与生产相关
        # - 可能包含"务工"、"非农"、"劳动"等关键词
        
        print("\n查找非农劳动相关列...")
        labor_cols = []
        for col in df_2020.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['务工', '非农', 'off', 'labor', 'work', 'wage', '工资']):
                labor_cols.append(col)
        
        print(f"找到 {len(labor_cols)} 个可能的非农劳动相关列:")
        for col in labor_cols[:20]:  # 只显示前20个
            print(f"  - {col}")
        
        # 尝试提取关键指标
        results = {}
        
        # 1. 查找总收入和非农收入
        income_cols = [col for col in df_2020.columns if '收入' in str(col) or 'income' in str(col).lower()]
        print(f"\n找到 {len(income_cols)} 个收入相关列")
        
        # 查找总收入和工资性收入（非农收入的主要来源）
        total_income_col = None
        wage_income_col = None
        
        for col in income_cols:
            col_str = str(col).lower()
            if '总' in str(col) and '收入' in str(col):
                total_income_col = col
            elif any(keyword in col_str for keyword in ['工资', 'wage', '务工']):
                wage_income_col = col
        
        if total_income_col:
            total_income = df_2020[total_income_col].dropna()
            if len(total_income) > 0:
                results['avg_total_income'] = float(total_income.mean())
                print(f"\n平均总收入: {results['avg_total_income']:.2f} 元")
        
        if wage_income_col:
            wage_income = df_2020[wage_income_col].dropna()
            if len(wage_income) > 0:
                results['avg_wage_income'] = float(wage_income.mean())
                results['wage_income_ratio'] = float(wage_income.mean() / total_income.mean()) if total_income_col and len(total_income) > 0 else 0
                print(f"平均工资性收入: {results['avg_wage_income']:.2f} 元")
                print(f"工资性收入占比: {results['wage_income_ratio']*100:.2f}%")
        
        # 2. 查找劳动时间分配
        # 查找农业劳动和非农劳动时间
        agri_labor_cols = [col for col in df_2020.columns if any(keyword in str(col).lower() for keyword in ['农业', '农', 'agri', 'farm', '种植', '耕作'])]
        non_agri_labor_cols = [col for col in df_2020.columns if any(keyword in str(col).lower() for keyword in ['非农', 'off', '务工', '打工'])]
        
        print(f"\n找到 {len(agri_labor_cols)} 个农业劳动相关列")
        print(f"找到 {len(non_agri_labor_cols)} 个非农劳动相关列")
        
        # 3. 计算非农收入占比
        if 'wage_income_ratio' in results:
            # 根据搜索结果，2020年非农收入占比约为35-50%
            # 如果数据中的占比低于这个范围，可能需要调整
            print(f"\n非农收入占比: {results['wage_income_ratio']*100:.2f}%")
            if results['wage_income_ratio'] < 0.3:
                print("  [注意] 非农收入占比偏低，建议调整模型参数")
            elif results['wage_income_ratio'] > 0.6:
                print("  [注意] 非农收入占比偏高，可能需要进一步验证")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    data_dir = _project_root / "校准用数据源"
    results = extract_non_farm_labor_data(data_dir)
    
    if results:
        print("\n" + "=" * 80)
        print("提取结果总结")
        print("=" * 80)
        for key, value in results.items():
            print(f"{key}: {value}")


