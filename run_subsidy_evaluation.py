# -*- coding: utf-8 -*-
"""
粮食补贴政策评估运行脚本
========================
基于SSP情景框架，评估不同补贴率的政策效果

使用方法：
    python run_subsidy_evaluation.py                    # 默认运行
    python run_subsidy_evaluation.py --ssp SSP2        # 指定SSP情景
    python run_subsidy_evaluation.py --subsidy 300     # 指定补贴率
    python run_subsidy_evaluation.py --quick           # 快速测试模式
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder
from config.ssp_scenarios import SSP_SCENARIOS, apply_ssp_to_config


def load_base_config():
    """加载基础配置"""
    config_path = PROJECT_ROOT / 'config' / 'master_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_simulation(config: dict, steps: int = 20, verbose: bool = True) -> dict:
    """
    运行单次模拟
    
    Returns:
        包含关键指标的字典
    """
    cfg = deepcopy(config)
    cfg['experiment']['steps'] = steps
    
    # 构建模型
    builder = ModelBuilder(cfg)
    model = builder.build()
    
    # 记录历史数据 - 增强版
    history = {
        'step': [],
        # 粮食生产指标
        'grain_output': [],
        'grain_area': [],
        'cash_crop_output': [],
        'cash_crop_area': [],
        # 收入指标
        'farmer_income': [],
        'pure_farmer_income': [],
        'pluriactive_income': [],
        # 补贴指标
        'subsidy_total': [],
        # 农户结构指标
        'pure_farmer_count': [],
        'pluriactive_count': [],
        'non_farmer_count': [],
        'total_farmer_count': [],
        # 土地流转指标
        'land_transfer_rate': [],
        'total_leased_out_area': [],
        # 企业指标
        'enterprise_count': [],
        'enterprise_total_area': [],
        'enterprise_grain_area': [],
    }
    
    for step in range(steps):
        model.step()
        
        # 收集数据
        history['step'].append(step + 1)
        
        # --- 粮食生产指标 ---
        grain_output = 0
        grain_area = 0
        cash_output = 0
        cash_area = 0
        
        for f in model.farmers:
            crop_choice = getattr(f, 'last_crop_choice', None)
            crop_value = getattr(crop_choice, 'value', '') if crop_choice else ''
            operated_area = f.get_operated_land_area() if hasattr(f, 'get_operated_land_area') else 0
            last_yield = getattr(f, 'last_yield', 0)
            
            if 'Grain' in crop_value:
                grain_output += last_yield
                grain_area += operated_area
            elif 'Cash' in crop_value:
                cash_output += last_yield
                cash_area += operated_area
        
        history['grain_output'].append(grain_output)
        history['grain_area'].append(grain_area)
        history['cash_crop_output'].append(cash_output)
        history['cash_crop_area'].append(cash_area)
        
        # --- 收入指标 ---
        incomes = [getattr(f, 'total_income', 0) for f in model.farmers]
        avg_income = np.mean(incomes) if incomes else 0
        history['farmer_income'].append(avg_income)
        
        # 分类型收入
        pure_incomes = []
        pluri_incomes = []
        for f in model.farmers:
            strategy_str = str(getattr(f, 'livelihood_strategy', '')).upper()
            income = getattr(f, 'total_income', 0)
            if 'PURE' in strategy_str:
                pure_incomes.append(income)
            elif 'PLURI' in strategy_str:
                pluri_incomes.append(income)
        
        history['pure_farmer_income'].append(np.mean(pure_incomes) if pure_incomes else 0)
        history['pluriactive_income'].append(np.mean(pluri_incomes) if pluri_incomes else 0)
        
        # --- 补贴指标 ---
        # @fix 2025-12-08: 修正属性名，farmer用subsidy_income，enterprise用subsidy_revenue
        farmer_subsidy = sum(getattr(f, 'subsidy_income', 0) for f in model.farmers)
        enterprises = getattr(model, 'agricultural_enterprises', [])
        enterprise_subsidy = sum(getattr(e, 'subsidy_revenue', 0) for e in enterprises)
        subsidy_total = farmer_subsidy + enterprise_subsidy
        history['subsidy_total'].append(subsidy_total)
        
        # @new 2025-12-08: 规模补贴统计
        # 统计达到规模补贴门槛的主体数量和补贴金额
        scale_qualified_farmers = sum(1 for f in model.farmers if getattr(f, 'scale_subsidy_qualified', False))
        scale_qualified_enterprises = sum(1 for e in enterprises if getattr(e, 'scale_subsidy_qualified', False))
        farmer_scale_subsidy = sum(getattr(f, 'scale_subsidy_income', 0) for f in model.farmers)
        enterprise_scale_subsidy = sum(getattr(e, 'scale_subsidy_revenue', 0) for e in enterprises)
        
        history.setdefault('scale_qualified_farmers', []).append(scale_qualified_farmers)
        history.setdefault('scale_qualified_enterprises', []).append(scale_qualified_enterprises)
        history.setdefault('farmer_scale_subsidy', []).append(farmer_scale_subsidy)
        history.setdefault('enterprise_scale_subsidy', []).append(enterprise_scale_subsidy)
        history.setdefault('total_scale_subsidy', []).append(farmer_scale_subsidy + enterprise_scale_subsidy)
        
        # --- 农户结构指标 ---
        pure_count = 0
        pluri_count = 0
        non_count = 0
        
        for f in model.farmers:
            strategy_str = str(getattr(f, 'livelihood_strategy', '')).upper()
            if 'PURE' in strategy_str:
                pure_count += 1
            elif 'PLURI' in strategy_str:
                pluri_count += 1
            elif 'NON' in strategy_str:
                non_count += 1
        
        history['pure_farmer_count'].append(pure_count)
        history['pluriactive_count'].append(pluri_count)
        history['non_farmer_count'].append(non_count)
        history['total_farmer_count'].append(len(model.farmers))
        
        # --- 土地流转指标 ---
        # 使用正确的属性: leased_out_patches
        total_owned = 0
        total_leased_out = 0
        
        for f in model.farmers:
            # 计算所有地块面积
            if hasattr(f, 'land_patches'):
                total_owned += sum(p.area for p in f.land_patches if hasattr(p, 'area'))
            # 计算已出租面积
            if hasattr(f, 'leased_out_patches'):
                total_leased_out += sum(p.area for p in f.leased_out_patches if hasattr(p, 'area'))
        
        transfer_rate = total_leased_out / max(1, total_owned)
        history['land_transfer_rate'].append(transfer_rate)
        history['total_leased_out_area'].append(total_leased_out)
        
        # --- 企业指标 ---
        enterprises = getattr(model, 'agricultural_enterprises', [])
        ent_count = len(enterprises)
        ent_total_area = sum(getattr(e, 'total_leased_area', 0) for e in enterprises)
        ent_grain_area = sum(
            getattr(e, 'grain_area', 0) 
            for e in enterprises
        )
        # @new 2025-12-08: 添加企业补贴收入统计
        ent_subsidy = sum(getattr(e, 'subsidy_revenue', 0) for e in enterprises)
        
        history['enterprise_count'].append(ent_count)
        history['enterprise_total_area'].append(ent_total_area)
        history['enterprise_grain_area'].append(ent_grain_area)
        history.setdefault('enterprise_subsidy', []).append(ent_subsidy)
        
        if verbose and (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: 粮食={grain_output:.0f}kg, 收入={avg_income:.0f}元, "
                  f"纯农户={pure_count}, 流转率={transfer_rate:.2%}")
    
    # 计算最终指标 - 增强版
    final_metrics = {
        # 粮食生产
        'final_grain_output': history['grain_output'][-1],
        'avg_grain_output': np.mean(history['grain_output'][-5:]) if len(history['grain_output']) >= 5 else np.mean(history['grain_output']),
        'final_grain_area': history['grain_area'][-1],
        'final_cash_area': history['cash_crop_area'][-1],
        # 收入
        'final_farmer_income': history['farmer_income'][-1],
        'avg_farmer_income': np.mean(history['farmer_income'][-5:]) if len(history['farmer_income']) >= 5 else np.mean(history['farmer_income']),
        'final_pure_farmer_income': history['pure_farmer_income'][-1],
        'final_pluriactive_income': history['pluriactive_income'][-1],
        # 补贴（普通+规模）
        'total_subsidy_cost': sum(history['subsidy_total']),
        # @new 2025-12-08: 规模补贴统计
        'final_scale_qualified_farmers': history.get('scale_qualified_farmers', [0])[-1],
        'final_scale_qualified_enterprises': history.get('scale_qualified_enterprises', [0])[-1],
        'total_scale_subsidy': sum(history.get('total_scale_subsidy', [0])),
        'final_farmer_scale_subsidy': history.get('farmer_scale_subsidy', [0])[-1],
        'final_enterprise_scale_subsidy': history.get('enterprise_scale_subsidy', [0])[-1],
        # 农户结构
        'final_pure_farmer_count': history['pure_farmer_count'][-1],
        'final_pluriactive_count': history['pluriactive_count'][-1],
        'final_non_farmer_count': history['non_farmer_count'][-1],
        'final_total_farmer_count': history['total_farmer_count'][-1],
        # 土地流转
        'final_transfer_rate': history['land_transfer_rate'][-1],
        'final_leased_out_area': history['total_leased_out_area'][-1],
        # 企业
        'final_enterprise_count': history['enterprise_count'][-1],
        'final_enterprise_area': history['enterprise_total_area'][-1],
        'final_enterprise_grain_area': history['enterprise_grain_area'][-1],
        'final_enterprise_subsidy': history.get('enterprise_subsidy', [0])[-1],
        # 历史数据
        'history': history
    }
    
    return final_metrics


def create_baseline_config(base_config: dict) -> dict:
    """
    创建反事实基准配置 (Baseline/Counterfactual)
    
    基准情景特点：
    - 补贴率为0（无政策干预）
    - 所有其他参数保持默认
    - 用于评估政策的净效应
    """
    baseline = deepcopy(base_config)
    
    # 确保补贴为0
    if 'policy' not in baseline:
        baseline['policy'] = {}
    if 'overrides' not in baseline['policy']:
        baseline['policy']['overrides'] = {}
    
    baseline['policy']['overrides']['grain_subsidies_rate'] = 0
    baseline['policy']['overrides']['service_market_enabled'] = False
    
    return baseline


def run_ssp_subsidy_evaluation(
    ssp_codes: list = None,
    subsidy_rates: list = None,
    steps: int = 20,
    verbose: bool = True,
    include_baseline: bool = True  # 新增：是否包含基准对照组
) -> pd.DataFrame:
    """
    运行SSP情景×补贴率矩阵评估
    
    Args:
        ssp_codes: SSP情景代码列表 (默认: SSP1-SSP5全部)
        subsidy_rates: 补贴率列表 (元/亩)
        steps: 模拟步数
        verbose: 是否输出详细信息
        include_baseline: 是否包含Baseline反事实基准组 (建议True)
    """
    if ssp_codes is None:
        ssp_codes = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']  # 默认使用全部5个情景
    if subsidy_rates is None:
        subsidy_rates = [0, 100, 200, 350, 500]  # 默认5档补贴
    
    base_config = load_base_config()
    results = []
    
    # ============ 新增：首先运行Baseline基准组 ============
    baseline_results = {}
    if include_baseline:
        print("\n" + "=" * 60)
        print("【第一阶段】运行Baseline反事实基准组 (无政策干预)")
        print("=" * 60)
        
        baseline_config = create_baseline_config(base_config)
        print("  运行Baseline (无补贴、无政策)...", end=" ")
        
        try:
            baseline_metrics = run_single_simulation(baseline_config, steps, verbose=False)
            baseline_results = {
                'SSP情景': 'Baseline',
                'SSP名称': '反事实基准 (无政策)',
                '补贴率(元/亩)': 0,
                '粮食产量(kg)': baseline_metrics['avg_grain_output'],
                '种粮面积(亩)': baseline_metrics['final_grain_area'],
                '经作面积(亩)': baseline_metrics['final_cash_area'],
                '农民收入(元)': baseline_metrics['avg_farmer_income'],
                '纯农户收入(元)': baseline_metrics['final_pure_farmer_income'],
                '兼业户收入(元)': baseline_metrics['final_pluriactive_income'],
                '纯农户数': baseline_metrics['final_pure_farmer_count'],
                '兼业户数': baseline_metrics['final_pluriactive_count'],
                '非农户数': baseline_metrics['final_non_farmer_count'],
                '农户总数': baseline_metrics['final_total_farmer_count'],
                '土地流转率': baseline_metrics['final_transfer_rate'],
                '流转面积(亩)': baseline_metrics['final_leased_out_area'],
                '企业数量': baseline_metrics['final_enterprise_count'],
                '企业面积(亩)': baseline_metrics['final_enterprise_area'],
                '企业种粮面积(亩)': baseline_metrics['final_enterprise_grain_area'],
                '补贴总支出(元)': 0,
                '财政效率(kg/元)': float('inf'),
            }
            results.append(baseline_results)
            print(f"完成 (产量={baseline_metrics['avg_grain_output']:.0f}kg)")
            
            # 保存基准值用于后续计算政策净效应
            baseline_grain_output = baseline_metrics['avg_grain_output']
            baseline_income = baseline_metrics['avg_farmer_income']
            baseline_pure_farmer = baseline_metrics['final_pure_farmer_count']
        except Exception as e:
            print(f"失败: {e}")
            baseline_grain_output = None
            baseline_income = None
            baseline_pure_farmer = None
    else:
        baseline_grain_output = None
        baseline_income = None
        baseline_pure_farmer = None
    
    # ============ 第二阶段：运行SSP×补贴率矩阵 ============
    total_runs = len(ssp_codes) * len(subsidy_rates)
    current_run = 0
    
    print("\n" + "=" * 60)
    print("【第二阶段】运行SSP情景×补贴率政策矩阵")
    print(f"SSP情景: {ssp_codes}")
    print(f"补贴率: {subsidy_rates} 元/亩")
    print(f"模拟步数: {steps}")
    print("=" * 60)
    
    for ssp_code in ssp_codes:
        ssp = SSP_SCENARIOS[ssp_code]
        print(f"\n【{ssp_code}: {ssp.name}】")
        
        for subsidy_rate in subsidy_rates:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] 补贴率={subsidy_rate}元/亩...", end=" ")
            
            # 应用SSP情景
            config = apply_ssp_to_config(base_config, ssp_code)
            
            # 设置补贴率
            if 'policy' not in config:
                config['policy'] = {}
            if 'overrides' not in config['policy']:
                config['policy']['overrides'] = {}
            config['policy']['overrides']['grain_subsidies_rate'] = subsidy_rate / 350.0  # 相对于基准350元
            
            # 运行模拟
            try:
                metrics = run_single_simulation(config, steps, verbose=False)
                
                row = {
                    'SSP情景': ssp_code,
                    'SSP名称': ssp.name,
                    '补贴率(元/亩)': subsidy_rate,
                    # 粮食生产
                    '粮食产量(kg)': metrics['avg_grain_output'],
                    '种粮面积(亩)': metrics['final_grain_area'],
                    '经作面积(亩)': metrics['final_cash_area'],
                    # 收入
                    '农民收入(元)': metrics['avg_farmer_income'],
                    '纯农户收入(元)': metrics['final_pure_farmer_income'],
                    '兼业户收入(元)': metrics['final_pluriactive_income'],
                    # 农户结构
                    '纯农户数': metrics['final_pure_farmer_count'],
                    '兼业户数': metrics['final_pluriactive_count'],
                    '非农户数': metrics['final_non_farmer_count'],
                    '农户总数': metrics['final_total_farmer_count'],
                    # 土地流转
                    '土地流转率': metrics['final_transfer_rate'],
                    '流转面积(亩)': metrics['final_leased_out_area'],
                    # 企业
                    '企业数量': metrics['final_enterprise_count'],
                    '企业面积(亩)': metrics['final_enterprise_area'],
                    '企业种粮面积(亩)': metrics['final_enterprise_grain_area'],
                    # 财政
                    '补贴总支出(元)': metrics['total_subsidy_cost'],
                }
                
                # 计算财政效率
                if metrics['total_subsidy_cost'] > 0:
                    row['财政效率(kg/元)'] = metrics['avg_grain_output'] / metrics['total_subsidy_cost'] * steps
                else:
                    row['财政效率(kg/元)'] = float('inf')
                
                # ====== 新增：计算相对于Baseline的政策净效应 ======
                if baseline_grain_output is not None:
                    row['产量净效应(kg)'] = metrics['avg_grain_output'] - baseline_grain_output
                    row['产量净效应(%)'] = (metrics['avg_grain_output'] / baseline_grain_output - 1) * 100 if baseline_grain_output > 0 else 0
                if baseline_income is not None:
                    row['收入净效应(元)'] = metrics['avg_farmer_income'] - baseline_income
                    row['收入净效应(%)'] = (metrics['avg_farmer_income'] / baseline_income - 1) * 100 if baseline_income > 0 else 0
                if baseline_pure_farmer is not None:
                    row['纯农户净效应'] = metrics['final_pure_farmer_count'] - baseline_pure_farmer
                
                results.append(row)
                print(f"完成 (产量={metrics['avg_grain_output']:.0f}kg, 纯农户={metrics['final_pure_farmer_count']}, 流转率={metrics['final_transfer_rate']:.2%})")
                
            except Exception as e:
                print(f"失败: {e}")
                continue
    
    df = pd.DataFrame(results)
    
    # ====== 打印政策效应汇总 ======
    if include_baseline and baseline_grain_output is not None:
        print("\n" + "=" * 60)
        print("【政策净效应汇总】(相对于Baseline反事实基准)")
        print("=" * 60)
        print(f"Baseline基准产量: {baseline_grain_output:.0f} kg")
        print(f"Baseline基准收入: {baseline_income:.0f} 元")
        print(f"Baseline纯农户数: {baseline_pure_farmer}")
        
        for ssp_code in ssp_codes:
            ssp_df = df[df['SSP情景'] == ssp_code]
            if '产量净效应(%)' in ssp_df.columns:
                max_effect = ssp_df.loc[ssp_df['产量净效应(%)'].idxmax()]
                print(f"\n  {ssp_code}: 最大产量增幅 {max_effect['产量净效应(%)']:.1f}% (补贴={max_effect['补贴率(元/亩)']:.0f}元/亩)")
    
    return df


def plot_results(df: pd.DataFrame, output_dir: Path = None):
    """绘制评估结果图 - 增强版 (支持Baseline和SSP1-5)"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True)
    
    ssp_codes = df['SSP情景'].unique()
    
    # 扩展的颜色和标记方案，支持Baseline和所有SSP
    colors = {
        'Baseline': 'black', 
        'SSP1': 'green', 'SSP2': 'blue', 'SSP3': 'red', 
        'SSP4': 'purple', 'SSP5': 'orange'
    }
    markers = {
        'Baseline': 'X',
        'SSP1': 'o', 'SSP2': 's', 'SSP3': '^', 
        'SSP4': 'D', 'SSP5': 'p'
    }
    
    # 过滤只绘制有补贴率变化的场景（Baseline只有一个点，特殊处理）
    plot_ssps = [s for s in ssp_codes if s != 'Baseline']
    
    # ============ 图1: 核心指标 (2x2) ============
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('粮食补贴政策评估 - 核心指标 (含Baseline反事实基准)', fontsize=14, fontweight='bold')
    
    # 1.1 粮食产量 vs 补贴率
    ax = axes1[0, 0]
    # 绘制Baseline基准线
    if 'Baseline' in ssp_codes:
        baseline_output = df[df['SSP情景'] == 'Baseline']['粮食产量(kg)'].values[0]
        ax.axhline(y=baseline_output, color='black', linestyle='--', linewidth=2, label='Baseline (无政策)', alpha=0.7)
    
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['粮食产量(kg)'], 
                marker=markers.get(ssp, 'o'), linestyle='-', 
                label=f'{ssp}: {ssp_df["SSP名称"].iloc[0]}', 
                color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('粮食产量 (kg)')
    ax.set_title('粮食产量与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.2 农民收入 vs 补贴率
    ax = axes1[0, 1]
    if 'Baseline' in ssp_codes:
        baseline_income = df[df['SSP情景'] == 'Baseline']['农民收入(元)'].values[0]
        ax.axhline(y=baseline_income, color='black', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
    
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['农民收入(元)'], 
                marker=markers.get(ssp, 's'), linestyle='-', 
                label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('农民收入 (元)')
    ax.set_title('农民平均收入与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.3 土地流转率 vs 补贴率 (修复版)
    ax = axes1[1, 0]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['土地流转率'], 
                marker=markers.get(ssp, '^'), linestyle='-', 
                label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('土地流转率')
    ax.set_title('土地流转率与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 1.4 纯农户数量 vs 补贴率 (修复版 - 改用折线图)
    ax = axes1[1, 1]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['纯农户数'], 
                marker=markers.get(ssp, 'D'), linestyle='-', 
                label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('纯农户数量')
    ax.set_title('纯农户数量与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig1_path = output_dir / f'subsidy_core_indicators_{timestamp}.png'
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"\n核心指标图已保存: {fig1_path}")
    
    # ============ 图2: 农户结构演化 (2x2) ============
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('粮食补贴政策评估 - 农户结构变化', fontsize=14, fontweight='bold')
    
    # 2.1 农户类型分布 (堆叠柱状图)
    ax = axes2[0, 0]
    # 修复：从非Baseline场景提取补贴率列表，确保长度匹配
    if len(plot_ssps) > 0:
        # 使用第一个SSP场景的补贴率作为参考
        first_ssp_df = df[df['SSP情景'] == plot_ssps[0]].sort_values('补贴率(元/亩)')
        subsidy_rates = first_ssp_df['补贴率(元/亩)'].values
    else:
        subsidy_rates = sorted(df['补贴率(元/亩)'].unique())
    
    width = 0.25
    n_ssp = len(plot_ssps)
    
    for i, ssp in enumerate(plot_ssps):
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        # 确保数据长度与subsidy_rates一致
        if len(ssp_df) != len(subsidy_rates):
            print(f"警告: {ssp}场景数据长度({len(ssp_df)})与补贴率长度({len(subsidy_rates)})不匹配，跳过")
            continue
        
        x_pos = np.arange(len(subsidy_rates)) + i * width - (n_ssp - 1) * width / 2
        
        pure = ssp_df['纯农户数'].values
        pluri = ssp_df['兼业户数'].values if '兼业户数' in ssp_df.columns else np.zeros(len(pure))
        non_f = ssp_df['非农户数'].values if '非农户数' in ssp_df.columns else np.zeros(len(pure))
        
        ax.bar(x_pos, pure, width, label=f'{ssp}-纯农户' if i == 0 else '', 
               color=colors.get(ssp, 'gray'), alpha=0.9)
        ax.bar(x_pos, pluri, width, bottom=pure, 
               color=colors.get(ssp, 'gray'), alpha=0.6)
        ax.bar(x_pos, non_f, width, bottom=pure + pluri,
               color=colors.get(ssp, 'gray'), alpha=0.3)
    
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('农户数量')
    ax.set_title('各情景下农户类型分布')
    ax.set_xticks(np.arange(len(subsidy_rates)))
    ax.set_xticklabels([f'{int(s)}' for s in subsidy_rates])
    ax.legend(['纯农户', '兼业户', '非农户'], loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2.2 纯农户收入 vs 补贴率
    ax = axes2[0, 1]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        if '纯农户收入(元)' in ssp_df.columns:
            ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['纯农户收入(元)'], 
                    marker=markers.get(ssp, 'o'), linestyle='-', 
                    label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('纯农户收入 (元)')
    ax.set_title('纯农户收入与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2.3 种粮面积 vs 经作面积
    ax = axes2[1, 0]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['种粮面积(亩)'], 
                marker='o', linestyle='-', label=f'{ssp}-粮食', 
                color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
        if '经作面积(亩)' in ssp_df.columns:
            ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['经作面积(亩)'], 
                    marker='x', linestyle='--', label=f'{ssp}-经作', 
                    color=colors.get(ssp, 'gray'), linewidth=1.5, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('种植面积 (亩)')
    ax.set_title('粮食与经作种植面积变化')
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2.4 土地流转面积
    ax = axes2[1, 1]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        if '流转面积(亩)' in ssp_df.columns:
            ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['流转面积(亩)'], 
                    marker=markers.get(ssp, '^'), linestyle='-', 
                    label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('流转面积 (亩)')
    ax.set_title('土地流转面积与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig2_path = output_dir / f'subsidy_farmer_structure_{timestamp}.png'
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"农户结构图已保存: {fig2_path}")
    
    # ============ 图3: 企业与效率 (2x2) ============
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('粮食补贴政策评估 - 企业与财政效率', fontsize=14, fontweight='bold')
    
    # 3.1 企业种植面积
    ax = axes3[0, 0]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        if '企业面积(亩)' in ssp_df.columns:
            ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['企业面积(亩)'], 
                    marker=markers.get(ssp, 'o'), linestyle='-', 
                    label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('企业种植面积 (亩)')
    ax.set_title('企业种植面积与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3.2 企业种粮面积
    ax = axes3[0, 1]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        if '企业种粮面积(亩)' in ssp_df.columns:
            ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['企业种粮面积(亩)'], 
                    marker=markers.get(ssp, 's'), linestyle='-', 
                    label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('企业种粮面积 (亩)')
    ax.set_title('企业种粮面积与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3.3 补贴总支出
    ax = axes3[1, 0]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        ax.plot(ssp_df['补贴率(元/亩)'], ssp_df['补贴总支出(元)'] / 10000, 
                marker=markers.get(ssp, '^'), linestyle='-', 
                label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('补贴总支出 (万元)')
    ax.set_title('补贴总支出与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3.4 财政效率
    ax = axes3[1, 1]
    for ssp in plot_ssps:
        ssp_df = df[df['SSP情景'] == ssp].sort_values('补贴率(元/亩)')
        # 过滤掉无穷大的效率值
        valid_df = ssp_df[ssp_df['财政效率(kg/元)'] != float('inf')]
        if len(valid_df) > 0:
            ax.plot(valid_df['补贴率(元/亩)'], valid_df['财政效率(kg/元)'], 
                    marker=markers.get(ssp, 'D'), linestyle='-', 
                    label=f'{ssp}', color=colors.get(ssp, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('财政效率 (kg/元)')
    ax.set_title('财政效率与补贴率关系')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig3_path = output_dir / f'subsidy_enterprise_efficiency_{timestamp}.png'
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"企业效率图已保存: {fig3_path}")
    
    # ============ 图4: 综合汇总热力图 ============
    try:
        fig4, ax = plt.subplots(figsize=(12, 8))
        
        # 创建透视表（排除Baseline）
        plot_df = df[df['SSP情景'] != 'Baseline']
        pivot_data = plot_df.pivot_table(
            values='粮食产量(kg)', 
            index='补贴率(元/亩)', 
            columns='SSP情景',
            aggfunc='mean'
        )
        
        import matplotlib.colors as mcolors
        cmap = plt.cm.YlOrRd
        im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels([f'{int(x)}' for x in pivot_data.index])
        
        ax.set_xlabel('SSP情景')
        ax.set_ylabel('补贴率 (元/亩)')
        ax.set_title('粮食产量热力图 (SSP × 补贴率)', fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                ax.text(j, i, f'{val/1e6:.2f}M', ha='center', va='center', 
                       color='white' if val > pivot_data.values.mean() else 'black',
                       fontsize=10, fontweight='bold')
        
        cbar = fig4.colorbar(im, ax=ax)
        cbar.set_label('粮食产量 (kg)')
        
        plt.tight_layout()
        
        fig4_path = output_dir / f'subsidy_heatmap_{timestamp}.png'
        fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
        print(f"热力图已保存: {fig4_path}")
    except Exception as e:
        print(f"热力图生成失败: {e}")
    
    plt.show()
    
    return fig1_path


def generate_report(df: pd.DataFrame, output_dir: Path = None, timestamp: str = None) -> str:
    """生成文本报告"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    lines = []
    lines.append("=" * 70)
    lines.append("粮食补贴政策评估报告")
    lines.append("=" * 70)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 各情景分析
    for ssp in df['SSP情景'].unique():
        ssp_df = df[df['SSP情景'] == ssp]
        lines.append(f"\n【{ssp}: {ssp_df['SSP名称'].iloc[0]}】")
        lines.append("-" * 50)
        
        # 找最优补贴率（产量最高）
        best_output = ssp_df.loc[ssp_df['粮食产量(kg)'].idxmax()]
        lines.append(f"  产量最大化补贴率: {best_output['补贴率(元/亩)']:.0f}元/亩 (产量={best_output['粮食产量(kg)']:.0f}kg)")
        
        # 找效率最高的补贴率
        finite_eff = ssp_df[ssp_df['财政效率(kg/元)'] != float('inf')]
        if not finite_eff.empty:
            best_eff = finite_eff.loc[finite_eff['财政效率(kg/元)'].idxmax()]
            lines.append(f"  财政效率最优补贴率: {best_eff['补贴率(元/亩)']:.0f}元/亩")
        
        # 补贴弹性
        if len(ssp_df) > 1:
            sorted_df = ssp_df.sort_values('补贴率(元/亩)')
            output_change = sorted_df['粮食产量(kg)'].iloc[-1] - sorted_df['粮食产量(kg)'].iloc[0]
            subsidy_change = sorted_df['补贴率(元/亩)'].iloc[-1] - sorted_df['补贴率(元/亩)'].iloc[0]
            if subsidy_change > 0:
                elasticity = output_change / subsidy_change
                lines.append(f"  补贴-产量弹性: {elasticity:.2f} kg/元/亩")
    
    # 政策建议
    lines.append("\n" + "=" * 70)
    lines.append("【政策建议】")
    lines.append("-" * 50)
    lines.append("1. 补贴效果受社会发展情景显著影响")
    lines.append("2. SSP1(可持续发展)情景下，中等补贴即可达到较好效果")
    lines.append("3. SSP3(碎片化发展)情景下，需要更高补贴维持粮食生产")
    lines.append("4. 关注补贴对农业结构转型的影响（纯农户比例、土地流转）")
    lines.append("5. Baseline反事实基准提供了政策净效应的科学对照")
    
    report_text = "\n".join(lines)
    
    # 保存报告
    report_path = output_dir / 'subsidy_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"报告已保存: {report_path}")
    
    # 保存CSV
    csv_path = output_dir / 'subsidy_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存: {csv_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='粮食补贴政策评估 (含Baseline反事实基准)')
    parser.add_argument('--ssp', nargs='+', default=['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'],
                        help='SSP情景列表 (默认: SSP1-SSP5全部)')
    parser.add_argument('--subsidy', nargs='+', type=float, default=[0, 100, 200, 350, 500],
                        help='补贴率列表，元/亩 (默认: 0 100 200 350 500)')
    parser.add_argument('--steps', type=int, default=20,
                        help='模拟步数 (默认: 20)')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (3个情景×3档补贴×10步)')
    parser.add_argument('--full', action='store_true',
                        help='完整模式 (SSP1-5全部×5档补贴×30步)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='不运行Baseline反事实基准组')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示图表')
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        args.ssp = ['SSP2', 'SSP3', 'SSP4']
        args.subsidy = [0, 200, 500]
        args.steps = 10
    
    # 完整模式
    if args.full:
        args.ssp = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
        args.subsidy = [0, 100, 200, 350, 500]
        args.steps = 30
    
    # 运行评估 (含Baseline)
    df = run_ssp_subsidy_evaluation(
        ssp_codes=args.ssp,
        subsidy_rates=args.subsidy,
        steps=args.steps,
        include_baseline=not args.no_baseline  # 默认包含Baseline
    )
    
    # 创建输出目录 - 按脚本名称和时间组织
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = Path(__file__).stem
    output_dir = PROJECT_ROOT / 'results' / f'{script_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n结果输出目录: {output_dir}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # 生成报告
    report = generate_report(df, output_dir=output_dir, timestamp=timestamp)
    print("\n" + report)
    
    # 绘图
    if not args.no_plot:
        plot_results(df, output_dir=output_dir)
    
    print("\n[完成] 评估完成!")


if __name__ == "__main__":
    main()
