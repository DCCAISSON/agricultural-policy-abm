#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看政策优化实验结果

自动查找最新的结果文件夹并展示结果

用法:
    python view_results.py                    # 查看最新结果
    python view_results.py --script run_policy_optimization  # 查看指定脚本结果
    python view_results.py --folder run_policy_optimization_20251207_123456  # 查看指定文件夹
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'

def find_latest_result_folder(script_prefix: str = None):
    """找到最新的结果文件夹"""
    if not RESULTS_DIR.exists():
        return None
    
    # 获取所有结果文件夹
    folders = [f for f in RESULTS_DIR.iterdir() if f.is_dir()]
    
    # 如果指定了脚本前缀，则过滤
    if script_prefix:
        folders = [f for f in folders if f.name.startswith(script_prefix)]
    
    if not folders:
        return None
    
    # 按修改时间排序，返回最新的
    return max(folders, key=lambda x: x.stat().st_mtime)

def find_json_file(folder: Path, pattern: str = '*.json'):
    """在文件夹中查找JSON文件"""
    json_files = list(folder.glob(pattern))
    if not json_files:
        return None
    # 优先返回 optimization_results.json 或第一个找到的
    for jf in json_files:
        if 'optimization_results' in jf.name or 'results' in jf.name:
            return jf
    return json_files[0]

def main():
    parser = argparse.ArgumentParser(description='查看实验结果')
    parser.add_argument('--script', type=str, default=None,
                        help='指定脚本名前缀 (如 run_policy_optimization)')
    parser.add_argument('--folder', type=str, default=None,
                        help='直接指定结果文件夹名')
    parser.add_argument('--list', action='store_true',
                        help='列出所有结果文件夹')
    args = parser.parse_args()
    
    # 列出所有文件夹
    if args.list:
        print("=" * 60)
        print("所有结果文件夹")
        print("=" * 60)
        if RESULTS_DIR.exists():
            folders = sorted([f for f in RESULTS_DIR.iterdir() if f.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
            for f in folders[:20]:  # 只显示最近20个
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"  {f.name:<50} {mtime.strftime('%Y-%m-%d %H:%M')}")
        return
    
    # 找到结果文件夹
    if args.folder:
        result_folder = RESULTS_DIR / args.folder
        if not result_folder.exists():
            print(f"文件夹不存在: {result_folder}")
            return
    else:
        result_folder = find_latest_result_folder(args.script)
    
    if not result_folder:
        print("未找到结果文件夹")
        return
    
    print(f"读取结果文件夹: {result_folder}")
    
    # 查找JSON文件
    result_file = find_json_file(result_folder)
    if not result_file:
        print(f"文件夹中未找到JSON结果文件")
        # 列出文件夹内容
        print("文件夹内容:")
        for f in result_folder.iterdir():
            print(f"  {f.name}")
        return
    
    print(f"读取结果文件: {result_file.name}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 显示反事实实验结果
    print("\n" + "="*60)
    print("反事实实验结果对比")
    print("="*60)
    
    counterfactual = data.get('counterfactual', {})
    
    # 表头
    print(f"\n{'场景':<20} {'粮食产量(百万斤)':<18} {'纯农户收入(元)':<15} {'补贴成本(百万元)':<18}")
    print("-"*75)
    
    for scenario, metrics in counterfactual.items():
        grain = metrics['grain_output_mean'] / 1e6
        income = metrics['pure_farmer_income_mean']
        cost = metrics['subsidy_cost_mean'] / 1e6
        print(f"{scenario:<20} {grain:<18.2f} {income:<15.0f} {cost:<18.2f}")
    
    # 计算基准对比
    print("\n" + "="*60)
    print("相对基准变化 (以Baseline为基准)")
    print("="*60)
    
    baseline = counterfactual.get('Baseline_无政策', counterfactual.get(list(counterfactual.keys())[0]))
    baseline_grain = baseline['grain_output_mean']
    baseline_income = baseline['pure_farmer_income_mean']
    baseline_cost = baseline['subsidy_cost_mean']
    
    print(f"\n{'场景':<20} {'粮食产量变化%':<15} {'收入变化%':<12} {'成本变化%':<15}")
    print("-"*65)
    
    for scenario, metrics in counterfactual.items():
        grain_change = (metrics['grain_output_mean'] - baseline_grain) / baseline_grain * 100
        income_change = (metrics['pure_farmer_income_mean'] - baseline_income) / baseline_income * 100
        # 避免除零错误
        if baseline_cost > 0:
            cost_change = (metrics['subsidy_cost_mean'] - baseline_cost) / baseline_cost * 100
            cost_str = f"{cost_change:>+14.1f}%"
        else:
            cost_str = f"{'N/A (基准=0)':>15}"
        print(f"{scenario:<20} {grain_change:>+14.1f}% {income_change:>+11.1f}% {cost_str}")
    
    # 显示Pareto前沿
    print("\n" + "="*60)
    print("Pareto最优解前沿 (前5个)")
    print("="*60)
    
    pareto = data.get('pareto_frontier', [])
    if pareto:
        print(f"\n{'#':<3} {'粮食产量':<12} {'收入':<12} {'补贴率':<10} {'托市价':<10} {'服务率':<10}")
        print("-"*60)
        for i, sol in enumerate(pareto[:5]):
            params = sol.get('params', {})
            print(f"{i+1:<3} {sol['grain_output']/1e6:<12.2f} {sol['pure_farmer_income']:<12.0f} "
                  f"{params.get('subsidy_rate', 'N/A'):<10} {params.get('price_support', 'N/A'):<10} "
                  f"{params.get('service_coverage', 'N/A'):<10}")
    
    # 显示实验配置
    print("\n" + "="*60)
    print("实验配置")
    print("="*60)
    config = data.get('config', {})
    print(f"模拟步数: {config.get('n_steps', 'N/A')}")
    print(f"重复次数: {config.get('n_replicates', 'N/A')}")
    print(f"村庄数量: {config.get('n_villages', 'N/A')}")
    print(f"粮食种植约束: {config.get('grain_requirement', 'N/A')}")
    print(f"快速模式: {config.get('quick_mode', 'N/A')}")

if __name__ == '__main__':
    main()
