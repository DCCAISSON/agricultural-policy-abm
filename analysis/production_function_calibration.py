# -*- coding: utf-8 -*-
"""
生产函数弹性参数校准脚本

从CLES数据的e1**系列中提取面积、产量、劳动投入、资本投入数据，
使用Cobb-Douglas生产函数回归估计α, β, γ参数。

生产函数形式：
Y = TFP * A^α * L^β * K^γ * (1 + ε_climate)

其中：
- Y: 产量 (e103a*)
- A: 土地面积 (e102a*)
- L: 劳动投入 (e112a* + e113a*)
- K: 资本投入 (e107a* + e108a* + e109a* + e110a* + e111a* + e115a*)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import json

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def load_cles_production_data(data_dir: Path, year: int = 2020) -> pd.DataFrame:
    """
    从CLES数据中加载生产相关数据
    
    参数:
        data_dir: 数据目录路径
        year: 要加载的年份
    
    返回:
        包含生产数据的DataFrame
    """
    household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.xlsx"
    
    if not household_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {household_file}")
    
    print(f"正在读取: {household_file}")
    df = pd.read_excel(household_file, sheet_name=0)
    
    # 如果有year列，筛选指定年份
    if 'year' in df.columns:
        df = df[df['year'] == year].copy()
        print(f"筛选{year}年数据: {len(df)}条记录")
    
    return df


def extract_production_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    从DataFrame中提取生产函数所需的变量
    
    参数:
        df: 原始数据DataFrame
    
    返回:
        包含提取变量的DataFrame
    """
    production_data = []
    
    # 遍历每个作物（最多6个作物：a1到a6）
    for i in range(1, 7):
        suffix = f"a{i}"
        
        # 提取变量
        area_col = f"e102{suffix}"  # 种植面积（亩）
        yield_col = f"e103{suffix}"  # 总产量（斤）
        labor_self_col = f"e112{suffix}"  # 自投工日（天/亩）
        labor_hired_col = f"e113{suffix}"  # 雇工费用（元/亩）
        
        # 资本投入（各种费用，元/亩）
        seed_col = f"e107{suffix}"  # 种子秧苗费
        fertilizer_col = f"e108{suffix}"  # 化肥费用
        pesticide_col = f"e109{suffix}"  # 农药费用
        irrigation_col = f"e110{suffix}"  # 水电及灌溉费用
        machinery_col = f"e111{suffix}"  # 机械作业费用
        depreciation_col = f"e115{suffix}"  # 固定资产折旧
        
        # 检查列是否存在
        if area_col not in df.columns or yield_col not in df.columns:
            continue
        
        # 提取数据
        for idx, row in df.iterrows():
            area = pd.to_numeric(row.get(area_col, 0), errors='coerce')
            yield_val = pd.to_numeric(row.get(yield_col, 0), errors='coerce')
            
            # 跳过缺失或无效数据
            if pd.isna(area) or pd.isna(yield_val) or area <= 0 or yield_val <= 0:
                continue
            
            # 劳动投入：自投工日 + 雇工（将雇工费用转换为工日，假设每天100元）
            labor_self = pd.to_numeric(row.get(labor_self_col, 0), errors='coerce') or 0
            labor_hired_cost = pd.to_numeric(row.get(labor_hired_col, 0), errors='coerce') or 0
            labor_hired_days = labor_hired_cost / 100.0  # 假设每天100元
            labor_total = (labor_self + labor_hired_days) * area  # 总工日
            
            # 资本投入：各种费用总和（元）
            seed_cost = pd.to_numeric(row.get(seed_col, 0), errors='coerce') or 0
            fertilizer_cost = pd.to_numeric(row.get(fertilizer_col, 0), errors='coerce') or 0
            pesticide_cost = pd.to_numeric(row.get(pesticide_col, 0), errors='coerce') or 0
            irrigation_cost = pd.to_numeric(row.get(irrigation_col, 0), errors='coerce') or 0
            machinery_cost = pd.to_numeric(row.get(machinery_col, 0), errors='coerce') or 0
            depreciation_cost = pd.to_numeric(row.get(depreciation_col, 0), errors='coerce') or 0
            
            capital_total = (seed_cost + fertilizer_cost + pesticide_cost + 
                           irrigation_cost + machinery_cost + depreciation_cost) * area
            
            # 只保留有效数据
            if labor_total > 0 and capital_total > 0:
                production_data.append({
                    'area': area,
                    'yield': yield_val,
                    'labor': labor_total,
                    'capital': capital_total,
                    'household_id': row.get('hid', idx)
                })
    
    production_df = pd.DataFrame(production_data)
    print(f"提取到 {len(production_df)} 条有效生产记录")
    
    return production_df


def estimate_production_elasticities(production_df: pd.DataFrame) -> Dict[str, float]:
    """
    使用Cobb-Douglas生产函数估计弹性参数
    
    对数线性化：ln(Y) = ln(TFP) + α*ln(A) + β*ln(L) + γ*ln(K) + ε
    
    参数:
        production_df: 包含生产数据的DataFrame
    
    返回:
        估计的参数字典
    """
    if len(production_df) < 50:
        print(f"警告: 数据点过少 ({len(production_df)}), 可能影响估计精度")
    
    # 准备数据
    Y = production_df['yield'].values
    A = production_df['area'].values
    L = production_df['labor'].values
    K = production_df['capital'].values
    
    # 对数变换（避免0值）
    log_Y = np.log(Y + 1)
    log_A = np.log(A + 1)
    log_L = np.log(L + 1)
    log_K = np.log(K + 1)
    
    # 构建回归模型
    X = np.column_stack([log_A, log_L, log_K])
    X = sm.add_constant(X)  # 添加常数项（代表ln(TFP)）
    
    # 执行OLS回归
    model = sm.OLS(log_Y, X)
    results = model.fit()
    
    # 提取参数
    params = results.params
    alpha = params[1]  # 土地弹性
    beta = params[2]   # 劳动弹性
    gamma = params[3]  # 资本弹性
    tfp_log = params[0]  # ln(TFP)
    
    # 计算统计量
    r_squared = results.rsquared
    pvalues = results.pvalues
    
    print("\n" + "=" * 80)
    print("生产函数弹性参数估计结果")
    print("=" * 80)
    print(f"\n模型形式: ln(Y) = ln(TFP) + α*ln(A) + β*ln(L) + γ*ln(K)")
    print(f"\n估计参数:")
    print(f"  α (土地弹性): {alpha:.4f} (p-value: {pvalues[1]:.4f})")
    print(f"  β (劳动弹性): {beta:.4f} (p-value: {pvalues[2]:.4f})")
    print(f"  γ (资本弹性): {gamma:.4f} (p-value: {pvalues[3]:.4f})")
    print(f"  ln(TFP): {tfp_log:.4f}")
    print(f"\n模型拟合度:")
    print(f"  R²: {r_squared:.4f}")
    print(f"  样本数: {len(production_df)}")
    
    # 验证参数合理性
    total_returns = alpha + beta + gamma
    print(f"\n规模报酬: {total_returns:.4f}")
    if total_returns > 1.1:
        print("  警告: 规模报酬递增，可能存在问题")
    elif total_returns < 0.9:
        print("  警告: 规模报酬递减，可能存在问题")
    else:
        print("  规模报酬接近常数，符合预期")
    
    # 返回结果
    return {
        'production_alpha': float(alpha),
        'production_beta': float(beta),
        'production_gamma': float(gamma),
        'tfp_log_mean': float(tfp_log),
        'r_squared': float(r_squared),
        'n_observations': len(production_df),
        'total_returns_to_scale': float(total_returns),
        'pvalues': {
            'alpha': float(pvalues[1]),
            'beta': float(pvalues[2]),
            'gamma': float(pvalues[3])
        }
    }


def estimate_climate_shock_parameters(production_df: pd.DataFrame, 
                                     data_dir: Path) -> Dict[str, float]:
    """
    估计气候冲击参数
    
    从产量波动中分离出气候冲击的影响
    假设：产量波动 = 系统性因素 + 气候冲击
    
    参数:
        production_df: 生产数据DataFrame
        data_dir: 数据目录（用于查找灾害数据）
    
    返回:
        气候冲击参数
    """
    # 方法1: 从产量波动估计
    # 计算产量的变异系数（CV = std/mean）
    yields = production_df['yield'].values
    mean_yield = np.mean(yields)
    std_yield = np.std(yields)
    cv = std_yield / mean_yield if mean_yield > 0 else 0
    
    # 假设气候冲击的变异系数约为总变异的30-50%
    climate_cv = cv * 0.4
    
    # 转换为标准差（假设均值为0）
    climate_std = climate_cv
    
    # 方法2: 如果有灾害数据，可以从灾害数据估计
    # 这里使用简化方法，基于文献值调整
    # 根据文献，中国农业产量年际波动系数约为0.05-0.15
    climate_mean = 0.0  # 气候冲击均值为0（无趋势）
    climate_std_refined = max(0.05, min(0.15, climate_std))
    
    print("\n" + "=" * 80)
    print("气候冲击参数估计")
    print("=" * 80)
    print(f"\n产量统计:")
    print(f"  平均产量: {mean_yield:.2f}")
    print(f"  标准差: {std_yield:.2f}")
    print(f"  变异系数: {cv:.4f}")
    print(f"\n气候冲击参数:")
    print(f"  μ_climate (均值): {climate_mean:.4f}")
    print(f"  σ_climate (标准差): {climate_std_refined:.4f}")
    
    return {
        'climate_mean': float(climate_mean),
        'climate_std': float(climate_std_refined),
        'yield_cv': float(cv),
        'method': 'yield_volatility_analysis'
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生产函数弹性参数校准")
    parser.add_argument('--data-dir', type=str, default='校准用数据源',
                       help='数据目录路径')
    parser.add_argument('--year', type=int, default=2020,
                       help='要分析的年份')
    parser.add_argument('--output', type=str, default='results/production_function_params.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    data_dir = _project_root / args.data_dir
    output_path = _project_root / args.output
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("生产函数弹性参数校准")
    print("=" * 80)
    
    # 1. 加载数据
    df = load_cles_production_data(data_dir, args.year)
    
    # 2. 提取生产变量
    production_df = extract_production_variables(df)
    
    if len(production_df) == 0:
        print("错误: 未能提取到有效的生产数据")
        return
    
    # 3. 估计生产函数弹性
    elasticity_results = estimate_production_elasticities(production_df)
    
    # 4. 估计气候冲击参数
    climate_results = estimate_climate_shock_parameters(production_df, data_dir)
    
    # 5. 合并结果
    results = {
        'year': args.year,
        'production_elasticities': elasticity_results,
        'climate_shock': climate_results,
        'data_summary': {
            'n_observations': len(production_df),
            'mean_area': float(production_df['area'].mean()),
            'mean_yield': float(production_df['yield'].mean()),
            'mean_labor': float(production_df['labor'].mean()),
            'mean_capital': float(production_df['capital'].mean())
        }
    }
    
    # 6. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    return results


if __name__ == '__main__':
    main()



