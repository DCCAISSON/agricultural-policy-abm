# -*- coding: utf-8 -*-
"""
完整参数校准脚本

整合所有参数校准：
1. 从CLES数据估计生产函数弹性参数
2. 从文献获取其他参数
3. 合并所有参数并更新校准配置
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.production_function_calibration import (
    load_cles_production_data,
    extract_production_variables,
    estimate_production_elasticities,
    estimate_climate_shock_parameters
)
from analysis.literature_based_parameters import get_all_literature_parameters


def merge_all_parameters(production_params: Dict, 
                        literature_params: Dict,
                        existing_calibrated: Dict = None) -> Dict[str, Any]:
    """
    合并所有校准参数
    
    参数:
        production_params: 从CLES数据估计的生产函数参数
        literature_params: 从文献获取的参数
        existing_calibrated: 已有的校准参数
    
    返回:
        合并后的参数字典
    """
    merged = {}
    
    # 1. 生产函数弹性参数
    if 'production_elasticities' in production_params:
        elasticities = production_params['production_elasticities']
        merged['production_alpha'] = elasticities.get('production_alpha', 0.4)
        merged['production_beta'] = elasticities.get('production_beta', 0.3)
        merged['production_gamma'] = elasticities.get('production_gamma', 0.2)
        merged['tfp_log_mean'] = elasticities.get('tfp_log_mean', 0.0)
    
    # 2. 气候冲击参数
    if 'climate_shock' in production_params:
        climate = production_params['climate_shock']
        merged['climate_mean'] = climate.get('climate_mean', 0.0)
        merged['climate_std'] = climate.get('climate_std', 0.08)
    
    # 3. 从文献参数中提取
    if 'prospect_theory' in literature_params:
        pt = literature_params['prospect_theory']
        merged['loss_aversion_lambda_mean'] = pt.get('loss_aversion_lambda_mean', 3.0)
        merged['value_function_curvature_mean'] = pt.get('value_function_curvature_mean', 0.55)
        merged['probability_weighting_mean'] = pt.get('probability_weighting_mean', 0.7)
    
    if 'social_network' in literature_params:
        sn = literature_params['social_network']
        merged['small_world_k'] = sn.get('small_world_k', 6)
        merged['small_world_p'] = sn.get('small_world_p', 0.1)
    
    if 'market_mechanism' in literature_params:
        mm = literature_params['market_mechanism']
        merged['wage_adjustment_elasticity'] = mm.get('wage_adjustment_elasticity', 0.05)
        merged['price_adjustment_speed'] = mm.get('price_adjustment_speed', 0.2)
        merged['rent_growth_rate'] = mm.get('rent_growth_rate_mean', 0.03)
    
    if 'technology_diffusion' in literature_params:
        td = literature_params['technology_diffusion']
        merged['info_transmission_rate'] = td.get('info_transmission_rate', 0.12)
        merged['adoption_threshold_mean'] = td.get('adoption_threshold_mean', 0.4)
    
    # 4. 保留已有的校准参数（如果有）
    if existing_calibrated:
        for key, value in existing_calibrated.items():
            if key not in merged:
                merged[key] = value
    
    return merged


def update_calibration_config(calibrated_params: Dict[str, Any],
                             config_path: Path):
    """
    更新校准配置文件
    
    参数:
        calibrated_params: 校准后的参数
        config_path: 配置文件路径
    """
    import yaml
    
    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加新的参数说明（不覆盖现有配置，只添加注释）
    if 'calibrated_parameters' not in config:
        config['calibrated_parameters'] = {}
    
    # 更新参数值
    for key, value in calibrated_params.items():
        config['calibrated_parameters'][key] = value
    
    # 保存
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"配置文件已更新: {config_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="完整参数校准")
    parser.add_argument('--data-dir', type=str, default='校准用数据源',
                       help='数据目录路径')
    parser.add_argument('--year', type=int, default=2020,
                       help='要分析的年份')
    parser.add_argument('--output', type=str, 
                       default='results/complete_calibrated_parameters.json',
                       help='输出文件路径')
    parser.add_argument('--update-config', action='store_true',
                       help='更新校准配置文件')
    
    args = parser.parse_args()
    
    data_dir = _project_root / args.data_dir
    output_path = _project_root / args.output
    
    print("=" * 80)
    print("完整参数校准")
    print("=" * 80)
    
    # 1. 从CLES数据估计生产函数参数
    print("\n步骤1: 从CLES数据估计生产函数弹性参数...")
    try:
        df = load_cles_production_data(data_dir, args.year)
        production_df = extract_production_variables(df)
        
        if len(production_df) > 0:
            elasticity_results = estimate_production_elasticities(production_df)
            climate_results = estimate_climate_shock_parameters(production_df, data_dir)
            production_params = {
                'production_elasticities': elasticity_results,
                'climate_shock': climate_results
            }
        else:
            print("警告: 未能提取到有效的生产数据，使用默认值")
            production_params = {
                'production_elasticities': {
                    'production_alpha': 0.4,
                    'production_beta': 0.3,
                    'production_gamma': 0.2
                },
                'climate_shock': {
                    'climate_mean': 0.0,
                    'climate_std': 0.08
                }
            }
    except Exception as e:
        print(f"错误: 生产函数参数估计失败: {e}")
        import traceback
        traceback.print_exc()
        production_params = {}
    
    # 2. 从文献获取其他参数
    print("\n步骤2: 从文献获取其他参数...")
    try:
        literature_params = get_all_literature_parameters()
    except Exception as e:
        print(f"错误: 文献参数获取失败: {e}")
        import traceback
        traceback.print_exc()
        literature_params = {}
    
    # 3. 合并所有参数
    print("\n步骤3: 合并所有参数...")
    try:
        # 尝试加载已有的校准参数
        existing_path = _project_root / "config" / "calibrated_params.json"
        existing_calibrated = {}
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'calibrated_parameters' in existing_data:
                    existing_calibrated = existing_data['calibrated_parameters']
        
        merged_params = merge_all_parameters(
            production_params,
            literature_params,
            existing_calibrated
        )
        
        print(f"\n合并后的参数数量: {len(merged_params)}")
        print("\n主要参数:")
        for key, value in list(merged_params.items())[:10]:
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"错误: 参数合并失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 保存结果
    print("\n步骤4: 保存校准结果...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    complete_results = {
        'calibrated_parameters': merged_params,
        'production_function_estimation': production_params,
        'literature_based_parameters': literature_params,
        'calibration_metadata': {
            'year': args.year,
            'data_source': 'CLES + Literature',
            'method': 'Production Function Regression + Literature Review'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, ensure_ascii=False, indent=2)
    
    print(f"完整校准结果已保存到: {output_path}")
    
    # 5. 更新配置文件（可选）
    if args.update_config:
        config_path = _project_root / "config" / "calibrated_params.json"
        update_calibration_config(merged_params, config_path)
    
    print("\n" + "=" * 80)
    print("校准完成！")
    print("=" * 80)
    print(f"\n下一步:")
    print(f"  1. 检查校准结果: {output_path}")
    print(f"  2. 运行模型校准: python scripts/run_calibration_and_validation.py --calibrate")
    print(f"  3. 验证模型: python scripts/run_calibration_and_validation.py --validate")
    
    return complete_results


if __name__ == '__main__':
    main()



