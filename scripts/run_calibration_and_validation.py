# -*- coding: utf-8 -*-
"""
校准和验证脚本

功能：
1. 使用2020年CLES数据进行参数校准
2. 使用2021-2022年数据进行验证

使用方法：
    python scripts/run_calibration_and_validation.py --calibrate
    python scripts/run_calibration_and_validation.py --validate
    python scripts/run_calibration_and_validation.py --full  # 完整流程
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.calibration import CalibrationManager
from analysis.empirical_calibration import EmpiricalMomentLoader
from analysis.batch_simulation import run_batch_simulation, aggregate_results
from scripts.build_composite_indicators import (
    build_farming_ability_index,
    build_technology_level_index,
    build_livelihood_strategy,
    build_income_quartiles
)
from analysis.production_function_calibration import (
    load_cles_production_data,
    extract_production_variables,
    estimate_production_elasticities,
    estimate_climate_shock_parameters
)


def update_master_config_with_production_params(
    elasticity_results: Dict[str, float],
    climate_results: Dict[str, float],
    config_path: Path = _project_root / "config" / "master_config.yaml"
) -> None:
    """
    将生产函数与气候冲击参数写入主配置，确保后续模型与校准使用最新实证结果。
    """
    if not config_path.exists():
        print(f"[WARNING] 配置文件未找到，无法写入生产函数参数: {config_path}")
        return

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[WARNING] 读取 {config_path} 失败，跳过自动写入: {exc}")
        return

    production_cfg = config.setdefault('production', {})
    elasticities_cfg = production_cfg.setdefault('production_elasticities', {})
    climate_cfg = production_cfg.setdefault('climate', {})

    updated = False

    # 写入生产弹性参数
    if elasticity_results:
        alpha = elasticity_results.get('production_alpha')
        beta = elasticity_results.get('production_beta')
        gamma = elasticity_results.get('production_gamma')

        if alpha is not None:
            elasticities_cfg['alpha'] = float(alpha)
            updated = True
        if beta is not None:
            elasticities_cfg['beta'] = float(beta)
            updated = True
        if gamma is not None:
            elasticities_cfg['gamma'] = float(gamma)
            updated = True

    # 写入气候冲击参数
    if climate_results:
        climate_mean = climate_results.get('climate_mean')
        climate_std = climate_results.get('climate_std')

        if climate_mean is not None:
            climate_cfg['mean'] = float(climate_mean)
            updated = True
        if climate_std is not None:
            climate_cfg['volatility'] = float(climate_std)
            updated = True

    if updated:
        try:
            with config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
            print(f"[OK] 生产函数和气候参数已写入 {config_path}")
        except Exception as exc:
            print(f"[WARNING] 写入 {config_path} 失败: {exc}")
    else:
        print("[INFO] 生产函数参数无变化，无需写入配置")


def extract_2020_moments(data_dir: Path) -> Dict[str, Any]:
    """
    从2020年CLES数据中提取目标矩（复用_extract_moments_from_dataframe）
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        moments: 目标矩字典
    """
    print("=" * 80)
    print("Step 1: Extract target moments from 2020 CLES data")
    print("=" * 80)
    
    # 加载2020-2021面板数据（筛选2020年）
    household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.xlsx"
    
    if not household_file.exists():
        print(f"Warning: File not found: {household_file}")
        return {}
    
    try:
        # 读取农户数据
        print(f"Reading: {household_file}")
        df_household = pd.read_excel(household_file, sheet_name=0)
        
        # 如果有year列，筛选2020年数据
        if 'year' in df_household.columns:
            df_2020 = df_household[df_household['year'] == 2020].copy()
            print(f"2020 household data: {len(df_2020)} records")
        else:
            df_2020 = df_household.copy()
            print(f"Household data (no year filter): {len(df_2020)} records")
        
        # 构建综合指标（新增）
        print("\n[新增] 构建综合指标...")
        try:
            df_2020['farming_ability'] = build_farming_ability_index(df_2020)
            print("[OK] 耕作能力综合指标构建完成")
        except Exception as e:
            print(f"[ERROR] 耕作能力综合指标构建失败: {e}")
            df_2020['farming_ability'] = 0.5  # 默认值
        
        try:
            df_2020['technology_level'] = build_technology_level_index(df_2020)
            print("[OK] 技术水平指标构建完成")
        except Exception as e:
            print(f"[ERROR] 技术水平指标构建失败: {e}")
            df_2020['technology_level'] = 1.0  # 默认值
        
        try:
            df_2020['livelihood_strategy'], df_2020['agri_income_ratio'] = build_livelihood_strategy(df_2020)
            print("[OK] 生计策略分类构建完成")
        except Exception as e:
            print(f"[ERROR] 生计策略分类构建失败: {e}")
            df_2020['livelihood_strategy'] = 'PLURIACTIVE'  # 默认值
            df_2020['agri_income_ratio'] = 0.5  # 默认值
        
        try:
            df_2020['income_quartile'], quartile_thresholds = build_income_quartiles(df_2020)
            # 转换为大写以匹配目标矩格式
            df_2020['income_quartile'] = df_2020['income_quartile'].str.upper()
            print("[OK] 收入分位数构建完成")
        except Exception as e:
            print(f"[ERROR] 收入分位数构建失败: {e}")
            import traceback
            traceback.print_exc()
            df_2020['income_quartile'] = 'Q2'  # 默认值
        
        # 提取目标矩（包括综合指标）
        moments = _extract_moments_from_dataframe(df_2020)
        
        # 添加综合指标相关的目标矩（新增）
        if 'farming_ability' in df_2020.columns:
            moments['farming_ability_dist'] = {
                'mean': float(df_2020['farming_ability'].mean()),
                'std': float(df_2020['farming_ability'].std()),
                'min': float(df_2020['farming_ability'].min()),
                'max': float(df_2020['farming_ability'].max())
            }
        
        if 'technology_level' in df_2020.columns:
            moments['technology_level_dist'] = {
                'mean': float(df_2020['technology_level'].mean()),
                'std': float(df_2020['technology_level'].std())
            }
        
        if 'livelihood_strategy' in df_2020.columns:
            strategy_counts = df_2020['livelihood_strategy'].value_counts()
            total = len(df_2020)
            moments['livelihood_strategy_dist'] = {
                'PURE_FARMER': float(strategy_counts.get('PURE_FARMER', 0) / total),
                'PLURIACTIVE': float(strategy_counts.get('PLURIACTIVE', 0) / total),
                'NON_FARMER': float(strategy_counts.get('NON_FARMER', 0) / total)
            }
            moments['avg_agri_income_ratio'] = float(df_2020['agri_income_ratio'].mean())
        
        if 'income_quartile' in df_2020.columns:
            quartile_counts = df_2020['income_quartile'].value_counts()
            total = len(df_2020)
            moments['income_quartile_dist'] = {
                'Q1': float(quartile_counts.get('Q1', 0) / total),
                'Q2': float(quartile_counts.get('Q2', 0) / total),
                'Q3': float(quartile_counts.get('Q3', 0) / total),
                'Q4': float(quartile_counts.get('Q4', 0) / total)
            }
        
        print("\nExtracted target moments:")
        for key, value in moments.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        return moments
        
    except Exception as e:
        print(f"Error: Failed to extract target moments: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_calibration(calibration_config: Path, data_dir: Path, 
                   output_dir: Path,
                   max_iterations: int = 50,
                   runs_per_iteration: int = 10,
                   optimization_method: str = 'Nelder-Mead') -> Dict[str, Any]:
    """
    运行参数校准
    
    参数:
        calibration_config: 校准配置文件路径
        data_dir: 数据目录路径
        output_dir: 输出目录路径
    
    返回:
        calibration_result: 校准结果
    """
    print("\n" + "=" * 80)
    print("Step 1.5: Estimate production function parameters from CLES data")
    print("=" * 80)
    
    # 估计生产函数参数（新增）
    production_params = {}
    try:
        print("\n[新增] 从CLES数据估计生产函数弹性参数...")
        df_production = load_cles_production_data(data_dir, year=2020)
        production_df = extract_production_variables(df_production)
        
        if len(production_df) > 0:
            elasticity_results = estimate_production_elasticities(production_df)
            climate_results = estimate_climate_shock_parameters(production_df, data_dir)
            
            production_params = {
                'production_elasticities': elasticity_results,
                'climate_shock': climate_results
            }
            
            print("[OK] 生产函数参数估计完成:")
            print(f"  - 土地弹性 (α): {elasticity_results.get('production_alpha', 'N/A')}")
            print(f"  - 劳动弹性 (β): {elasticity_results.get('production_beta', 'N/A')}")
            print(f"  - 资本弹性 (γ): {elasticity_results.get('production_gamma', 'N/A')}")
            print(f"  - 气候冲击均值: {climate_results.get('climate_mean', 'N/A')}")
            print(f"  - 气候冲击标准差: {climate_results.get('climate_std', 'N/A')}")
            
            # 自动写入master_config，确保校准和模拟使用最新参数
            update_master_config_with_production_params(
                elasticity_results,
                climate_results,
                _project_root / "config" / "master_config.yaml"
            )
        else:
            print("[WARNING] 未能提取到有效的生产数据，使用默认值")
    except Exception as e:
        print(f"[ERROR] 生产函数参数估计失败: {e}")
        import traceback
        traceback.print_exc()
        print("将继续使用默认参数进行校准")
    
    print("\n" + "=" * 80)
    print("Step 2: Run parameter calibration")
    print("=" * 80)
    
    # 提取2020年目标矩（包括综合指标）
    target_moments = extract_2020_moments(data_dir)
    
    if not target_moments:
        print("Warning: Failed to extract target moments from 2020 data, using default target moments")
        # 使用CalibrationManager的默认目标矩
        cal_manager = CalibrationManager()
        target_moments = cal_manager.target_moments
    
    # 初始化校准管理器
    print("\nInitializing calibration manager...")
    cal_manager = CalibrationManager()
    
    # 更新目标矩（如果从数据中提取到了）
    if target_moments:
        # 更新相关目标矩
        if 'land_transfer_rate' in target_moments:
            cal_manager.target_moments['land_transfer']['transfer_rate'] = target_moments['land_transfer_rate']
        if 'avg_farm_income' in target_moments:
            cal_manager.target_moments['income_distribution']['avg_farm_income'] = target_moments['avg_farm_income']
        if 'gini_coefficient' in target_moments:
            cal_manager.target_moments['income_distribution']['gini_coefficient'] = target_moments['gini_coefficient']
        if 'grain_crop_ratio' in target_moments:
            cal_manager.target_moments['crop_structure']['grain_crop_ratio'] = target_moments['grain_crop_ratio']
        if 'cash_crop_ratio' in target_moments:
            cal_manager.target_moments['crop_structure']['cash_crop_ratio'] = target_moments['cash_crop_ratio']
        if 'avg_yield_per_hectare' in target_moments:
            cal_manager.target_moments['productivity_indicators']['avg_yield_per_hectare'] = target_moments['avg_yield_per_hectare']
        
        # 新增：更新综合指标相关的目标矩
        if 'livelihood_strategy_dist' in target_moments:
            # 农户异质性分布作为校准目标
            strategy_dist = target_moments['livelihood_strategy_dist']
            if 'farmer_heterogeneity' not in cal_manager.target_moments:
                cal_manager.target_moments['farmer_heterogeneity'] = {}
            cal_manager.target_moments['farmer_heterogeneity']['livelihood_strategy_dist'] = strategy_dist
            print(f"[新增] 生计策略分布目标: PURE_FARMER={strategy_dist.get('PURE_FARMER', 0):.3f}, "
                  f"PLURIACTIVE={strategy_dist.get('PLURIACTIVE', 0):.3f}, "
                  f"NON_FARMER={strategy_dist.get('NON_FARMER', 0):.3f}")
        
        if 'avg_agri_income_ratio' in target_moments:
            # 农业收入占比目标
            if 'farmer_heterogeneity' not in cal_manager.target_moments:
                cal_manager.target_moments['farmer_heterogeneity'] = {}
            cal_manager.target_moments['farmer_heterogeneity']['avg_agri_income_ratio'] = target_moments['avg_agri_income_ratio']
            print(f"[新增] 平均农业收入占比目标: {target_moments['avg_agri_income_ratio']:.3f}")
        
        if 'farming_ability_dist' in target_moments:
            # 耕作能力分布目标
            if 'farmer_heterogeneity' not in cal_manager.target_moments:
                cal_manager.target_moments['farmer_heterogeneity'] = {}
            cal_manager.target_moments['farmer_heterogeneity']['farming_ability_dist'] = target_moments['farming_ability_dist']
            print(f"[新增] 耕作能力分布目标: mean={target_moments['farming_ability_dist']['mean']:.3f}, "
                  f"std={target_moments['farming_ability_dist']['std']:.3f}")
    
    print("\nTarget moments:")
    for category, values in cal_manager.target_moments.items():
        print(f"  {category}:")
        for key, value in values.items():
            print(f"    {key}: {value}")
    
    # 运行校准
    print("\nStarting calibration (this may take a long time)...")
    print(f"Configuration:")
    print(f"  - Max iterations: {max_iterations}")
    print(f"  - Runs per iteration: {runs_per_iteration}")
    print(f"  - Optimization method: {optimization_method}")
    print(f"  - Estimated total model runs: {max_iterations * runs_per_iteration}")
    
    try:
        # 执行校准
        calibration_result = cal_manager.calibrate_parameters(
            method=optimization_method,
            max_iterations=max_iterations,
            model_runs_per_iteration=runs_per_iteration
        )
        
        if calibration_result.get('success'):
            print("\nCalibration successful!")
            print(f"Objective function value: {calibration_result.get('objective_value', 'N/A')}")
            
            # 保存校准结果
            calibrated_params = calibration_result.get('calibrated_parameters', {})
            output_file = output_dir / "calibrated_parameters.json"
            cal_manager.save_calibrated_parameters(
                calibrated_params,
                str(output_file)
            )
            print(f"Calibrated parameters saved to: {output_file}")
            
            # 保存校准报告
            report_file = output_dir / "calibration_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(calibration_result, f, ensure_ascii=False, indent=2)
            print(f"Calibration report saved to: {report_file}")
            
            return calibration_result
        else:
            print(f"\nCalibration failed: {calibration_result.get('message', 'Unknown error')}")
            return calibration_result
            
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def extract_validation_moments(data_dir: Path, year: int = None) -> Dict[str, Any]:
    """
    从验证数据中提取目标矩（复用extract_2020_moments的逻辑）
    
    参数:
        data_dir: 数据目录路径
        year: 要提取的年份（2021或2022），如果为None则提取所有可用年份
    
    返回:
        moments: 目标矩字典，格式为 {year: {moment_key: value}}
    """
    moments_by_year = {}
    
    # 提取2021年数据
    household_file = data_dir / "CLES_2020_2021_household_panel_mapping_label.xlsx"
    if household_file.exists() and (year is None or year == 2021):
        try:
            df_household = pd.read_excel(household_file, sheet_name=0)
            if 'year' in df_household.columns:
                df_year = df_household[df_household['year'] == 2021].copy()
                if len(df_year) > 0:
                    moments_by_year[2021] = _extract_moments_from_dataframe(df_year)
                    print(f"Extracted 2021 moments: {len(moments_by_year[2021])} indicators")
        except Exception as e:
            print(f"Warning: Failed to extract 2021 moments: {e}")
    
    # 提取2022年数据
    household_2022_file = data_dir / "CLES_2022_household_mapping_label.xlsx"
    if household_2022_file.exists() and (year is None or year == 2022):
        try:
            df_2022 = pd.read_excel(household_2022_file, sheet_name=0)
            if len(df_2022) > 0:
                moments_by_year[2022] = _extract_moments_from_dataframe(df_2022)
                print(f"Extracted 2022 moments: {len(moments_by_year[2022])} indicators")
        except Exception as e:
            print(f"Warning: Failed to extract 2022 moments: {e}")
    
    return moments_by_year


def _extract_moments_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """从DataFrame中提取目标矩（复用逻辑）"""
    moments = {}
    
    # 1. 土地流转率
    if 'land_transfer_area' in df.columns and 'total_land_area' in df.columns:
        transfer_rate = (df['land_transfer_area'].sum() / df['total_land_area'].sum())
        moments['land_transfer_rate'] = float(transfer_rate)
    
    # 2. 平均收入
    income_cols = [col for col in df.columns if 'income' in col.lower()]
    if income_cols:
        total_income_col = None
        for col in income_cols:
            if 'total' in col.lower():
                total_income_col = col
                break
        if total_income_col:
            avg_income = df[total_income_col].mean()
            moments['avg_farm_income'] = float(avg_income)
            
            # 3. 基尼系数
            incomes = df[total_income_col].dropna()
            if len(incomes) > 1:
                incomes_sorted = np.sort(incomes)
                n = len(incomes_sorted)
                gini = (2 * np.sum((np.arange(1, n+1)) * incomes_sorted)) / (n * np.sum(incomes_sorted)) - (n + 1) / n
                moments['gini_coefficient'] = float(max(0, min(1, gini)))
    
    # 4. 作物结构
    grain_cols = [col for col in df.columns if 'grain' in col.lower()]
    cash_cols = [col for col in df.columns if 'cash' in col.lower()]
    if grain_cols and cash_cols:
        grain_area = df[grain_cols[0]].sum()
        cash_area = df[cash_cols[0]].sum()
        total_area = grain_area + cash_area
        if total_area > 0:
            moments['grain_crop_ratio'] = float(grain_area / total_area)
            moments['cash_crop_ratio'] = float(cash_area / total_area)
    
    # 5. 单产
    yield_cols = [col for col in df.columns if 'yield' in col.lower()]
    if yield_cols and grain_cols:
        total_yield = df[yield_cols[0]].sum()
        total_area_hectare = df[grain_cols[0]].sum() / 15
        if total_area_hectare > 0:
            yield_per_hectare = total_yield / total_area_hectare
            moments['avg_yield_per_hectare'] = float(yield_per_hectare)
    
    return moments


def run_validation(data_dir: Path, output_dir: Path, 
                  calibrated_params_path: Path = None,
                  calibration_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    使用2021-2022年数据验证模型
    
    参数:
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        calibrated_params_path: 校准后的参数文件路径
        calibration_result: 校准结果（如果提供，可以复用其中的模拟矩，避免重复计算）
    
    返回:
        validation_result: 验证结果
    """
    print("\n" + "=" * 80)
    print("Step 3: Validate model using 2021-2022 data")
    print("=" * 80)
    
    try:
        # 提取2021-2022年真实数据的目标矩
        print("\nExtracting validation moments from 2021-2022 data...")
        validation_moments = extract_validation_moments(data_dir)
        
        # 获取模拟结果（优先使用校准结果中的，避免重复计算）
        if calibration_result and 'simulated_moments' in calibration_result:
            print("\nUsing simulated moments from calibration result (avoiding redundant simulation)...")
            simulated_moments = calibration_result['simulated_moments']
            # 转换为统一格式
            simulated_moments_flat = {
                'land_transfer_rate': simulated_moments.get('land_transfer', {}).get('transfer_rate', 0),
                'gini_coefficient': simulated_moments.get('income_distribution', {}).get('gini_coefficient', 0),
                'avg_farm_income': simulated_moments.get('income_distribution', {}).get('avg_farm_income', 0),
                'grain_crop_ratio': simulated_moments.get('crop_structure', {}).get('grain_crop_ratio', 0),
            }
        else:
            # 如果没有校准结果，才运行新的模拟
            print("\nRunning new simulation for validation...")
            print("Note: This is redundant if calibration was just completed. Consider passing calibration_result.")
            simulation_results = run_batch_simulation(
                config_path="config/master_config.yaml",
                n_runs=10,
                n_steps=30,
                seed=None
            )
            aggregated = aggregate_results(simulation_results)
            simulated_moments_flat = {
                'land_transfer_rate': aggregated.get('land_transfer', {}).get('transfer_rate', {}).get('mean', 0),
                'gini_coefficient': aggregated.get('income_distribution', {}).get('gini_coefficient', {}).get('mean', 0),
                'avg_farm_income': aggregated.get('income_distribution', {}).get('avg_farm_income', {}).get('mean', 0),
                'grain_crop_ratio': aggregated.get('crop_structure', {}).get('grain_crop_ratio', {}).get('mean', 0),
            }
        
        # 计算拟合度指标
        print("\nCalculating fit metrics...")
        fit_metrics = {}
        for year, empirical_moments in validation_moments.items():
            fit_metrics[year] = {}
            for key in ['land_transfer_rate', 'gini_coefficient', 'avg_farm_income', 'grain_crop_ratio']:
                if key in empirical_moments and key in simulated_moments_flat:
                    empirical_val = empirical_moments[key]
                    simulated_val = simulated_moments_flat[key]
                    if empirical_val != 0:
                        relative_error = abs(simulated_val - empirical_val) / abs(empirical_val)
                        fit_metrics[year][key] = {
                            'empirical': empirical_val,
                            'simulated': simulated_val,
                            'absolute_error': abs(simulated_val - empirical_val),
                            'relative_error': relative_error,
                            'relative_error_pct': relative_error * 100
                        }
        
        validation_result = {
            'simulated_moments': simulated_moments_flat,
            'empirical_moments': validation_moments,
            'fit_metrics': fit_metrics,
            'summary': {
                'avg_relative_error': np.mean([
                    metrics.get('relative_error_pct', 0)
                    for year_metrics in fit_metrics.values()
                    for metrics in year_metrics.values()
                ]) if fit_metrics else 0
            }
        }
        
        # 保存验证结果
        validation_file = output_dir / "validation_report.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, ensure_ascii=False, indent=2)
        print(f"Validation report saved to: {validation_file}")
        
        # 打印对比结果
        print("\n" + "=" * 80)
        print("Validation Results: Simulated vs Empirical")
        print("=" * 80)
        for year, metrics in fit_metrics.items():
            print(f"\nYear {year}:")
            for key, fit_data in metrics.items():
                print(f"  {key}:")
                print(f"    Empirical: {fit_data['empirical']:.4f}")
                print(f"    Simulated: {fit_data['simulated']:.4f}")
                print(f"    Error: {fit_data['relative_error_pct']:.2f}%")
        
        if validation_result['summary']['avg_relative_error'] > 0:
            print(f"\nAverage relative error: {validation_result['summary']['avg_relative_error']:.2f}%")
        
        return validation_result
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="运行模型校准和验证")
    parser.add_argument('--calibrate', action='store_true',
                       help='运行校准（使用2020年数据）')
    parser.add_argument('--validate', action='store_true',
                       help='运行验证（使用2021-2022年数据）')
    parser.add_argument('--full', action='store_true',
                       help='运行完整流程（校准+验证）')
    parser.add_argument('--data-dir', type=str, default='校准用数据源',
                       help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='results/calibration',
                       help='输出目录路径')
    parser.add_argument('--calibration-config', type=str, 
                       default='config/calibration_config.yaml',
                       help='校准配置文件路径')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='最大迭代次数（默认：50）')
    parser.add_argument('--runs-per-iteration', type=int, default=10,
                       help='每次迭代的模型运行次数（默认：10）')
    parser.add_argument('--optimization-method', type=str, default='Nelder-Mead',
                       choices=['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC'],
                       help='优化算法（默认：Nelder-Mead）')
    
    args = parser.parse_args()
    if args.runs_per_iteration < 10:
        print("[INFO] runs-per-iteration 低于10，自动调整为10以降低模拟噪声。")
        args.runs_per_iteration = 10
    
    # 设置路径
    data_dir = _project_root / args.data_dir
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calibration_config = _project_root / args.calibration_config
    
    if not data_dir.exists():
        error_msg = f"Error: Data directory does not exist: {data_dir}"
        print(error_msg)
        return
    
    # 运行流程
    if args.full or args.calibrate:
        calibration_result = run_calibration(
            calibration_config, data_dir, output_dir,
            max_iterations=args.max_iterations,
            runs_per_iteration=args.runs_per_iteration,
            optimization_method=args.optimization_method
        )
        
        if args.full:
            # 如果校准成功，继续验证（传递校准结果，避免重复计算）
            if calibration_result.get('success'):
                calibrated_params_path = output_dir / "calibrated_parameters.json"
                run_validation(data_dir, output_dir, calibrated_params_path, calibration_result)
            else:
                print("\nCalibration failed, skipping validation step")
    
    elif args.validate:
        # 只运行验证
        calibrated_params_path = output_dir / "calibrated_parameters.json"
        if not calibrated_params_path.exists():
            print("Warning: Calibrated parameters file not found, using default parameters for validation")
        run_validation(data_dir, output_dir, calibrated_params_path)
    
    else:
        parser.print_help()
        print("\nPlease specify --calibrate, --validate, or --full")


if __name__ == '__main__':
    main()

