# -*- coding: utf-8 -*-
"""
详细博弈分析模块
专门分析每一期的作物选择博弈和土地流转博弈的详细情况
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.font_config import setup_chinese_fonts

def setup_plotting():
    """设置绘图参数"""
    setup_chinese_fonts()  # 使用统一的字体配置
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

def load_simulation_data():
    """加载仿真数据"""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    results_dir = script_dir.parent / "results"
    
    timestamp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('2025-')]
    if not timestamp_dirs:
        raise FileNotFoundError("未找到仿真结果目录")
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    print(f"使用结果目录: {latest_dir}")
    
    agent_data_file = latest_dir / "agent_data.csv"
    model_data_file = latest_dir / "model_data.csv"
    
    if not agent_data_file.exists():
        raise FileNotFoundError(f"未找到agent_data.csv文件: {agent_data_file}")
    
    agent_data = pd.read_csv(agent_data_file)
    model_data = pd.read_csv(model_data_file) if model_data_file.exists() else None
    
    return agent_data, model_data, latest_dir

def analyze_period_game_results(agent_data, output_dir):
    """分析每一期的博弈结果"""
    print("开始分析每一期的博弈结果...")
    
    periods = sorted(agent_data['Step'].unique())
    
    for period in periods[:10]:  # 只分析前10期
        print(f"分析第{period}期博弈结果...")
        
        period_data = agent_data[agent_data['Step'] == period]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'第{period}期博弈结果详细分析', fontsize=16, fontweight='bold')
        
        # 1. 农户作物选择分布
        ax1 = axes[0, 0]
        farmer_data = period_data[period_data['AgentID'].str.startswith('farmer')]
        if not farmer_data.empty and 'Agricultural_Income_Ratio' in farmer_data.columns:
            farmer_data['Crop_Choice'] = np.where(
                farmer_data['Agricultural_Income_Ratio'] > 0.95, '粮食作物',
                np.where(farmer_data['Agricultural_Income_Ratio'] > 0.85, '混合策略', '经济作物')
            )
            
            choice_counts = farmer_data['Crop_Choice'].value_counts()
            colors = ['lightgreen', 'orange', 'lightcoral']
            ax1.pie(choice_counts.values, labels=choice_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(choice_counts)], startangle=90)
            ax1.set_title(f'农户作物选择分布\n(第{period}期)')
        
        # 2. 农业企业策略分布
        ax2 = axes[0, 1]
        enterprise_data = period_data[period_data['AgentID'].str.startswith('enterprise')]
        if not enterprise_data.empty and 'Target_Return_Rate' in enterprise_data.columns:
            enterprise_data['Strategy'] = np.where(
                enterprise_data['Target_Return_Rate'] > 0.15, '高回报策略',
                np.where(enterprise_data['Target_Return_Rate'] > 0.10, '稳健策略', '保守策略')
            )
            
            strategy_counts = enterprise_data['Strategy'].value_counts()
            colors = ['lightcoral', 'skyblue', 'lightyellow']
            ax2.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(strategy_counts)], startangle=90)
            ax2.set_title(f'农业企业策略分布\n(第{period}期)')
        
        # 3. 土地经营面积分布
        ax3 = axes[0, 2]
        if 'Operated_Land_Area' in period_data.columns:
            farmer_land = farmer_data['Operated_Land_Area']
            enterprise_land = enterprise_data['Operated_Land_Area']
            
            ax3.hist(farmer_land, bins=15, alpha=0.7, label='农户', color='lightgreen', density=True)
            if not enterprise_land.empty:
                ax3.hist(enterprise_land, bins=15, alpha=0.7, label='农业企业', color='lightcoral', density=True)
            ax3.set_xlabel('经营土地面积')
            ax3.set_ylabel('密度')
            ax3.set_title(f'土地经营面积分布\n(第{period}期)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 收入分布对比
        ax4 = axes[1, 0]
        if 'Total_Income' in period_data.columns:
            farmer_income = farmer_data['Total_Income']
            enterprise_income = enterprise_data['Total_Income']
            
            ax4.hist(farmer_income, bins=20, alpha=0.7, label='农户', color='lightgreen', density=True)
            if not enterprise_income.empty:
                ax4.hist(enterprise_income, bins=20, alpha=0.7, label='农业企业', color='lightcoral', density=True)
            ax4.set_xlabel('总收入')
            ax4.set_ylabel('密度')
            ax4.set_title(f'收入分布对比\n(第{period}期)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 生计策略分布
        ax5 = axes[1, 1]
        if 'Livelihood_Strategy' in period_data.columns:
            strategy_counts = period_data['Livelihood_Strategy'].value_counts()
            bars = ax5.bar(strategy_counts.index, strategy_counts.values,
                          color=['lightgreen', 'lightblue', 'lightcoral'], alpha=0.8)
            ax5.set_xlabel('生计策略')
            ax5.set_ylabel('智能体数量')
            ax5.set_title(f'生计策略分布\n(第{period}期)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # 6. 作物选择与土地经营关系
        ax6 = axes[1, 2]
        if all(col in farmer_data.columns for col in ['Agricultural_Income_Ratio', 'Operated_Land_Area']):
            scatter = ax6.scatter(farmer_data['Agricultural_Income_Ratio'], 
                                farmer_data['Operated_Land_Area'],
                                alpha=0.6, c='green', s=30)
            ax6.set_xlabel('农业收入比例（作物选择代理）')
            ax6.set_ylabel('经营土地面积')
            ax6.set_title(f'作物选择与土地经营关系\n(第{period}期)')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'period_{period}_game_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("每期博弈结果分析完成")

def analyze_crop_choice_dynamics(agent_data, output_dir):
    """分析作物选择博弈的动态演化"""
    print("开始分析作物选择博弈动态演化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('作物选择博弈动态演化分析', fontsize=16, fontweight='bold')
    
    # 1. 作物选择策略演化时间序列
    ax1 = axes[0, 0]
    farmer_data = agent_data[agent_data['AgentID'].str.startswith('farmer')]
    if 'Agricultural_Income_Ratio' in farmer_data.columns:
        crop_choice_evolution = []
        for step in sorted(farmer_data['Step'].unique()):
            step_data = farmer_data[farmer_data['Step'] == step]
            step_data['Crop_Choice'] = np.where(
                step_data['Agricultural_Income_Ratio'] > 0.95, '粮食作物',
                np.where(step_data['Agricultural_Income_Ratio'] > 0.85, '混合策略', '经济作物')
            )
            choice_dist = step_data['Crop_Choice'].value_counts(normalize=True)
            crop_choice_evolution.append(choice_dist)
        
        evolution_df = pd.DataFrame(crop_choice_evolution, index=sorted(farmer_data['Step'].unique()))
        evolution_df.plot(kind='line', ax=ax1, marker='o', linewidth=2)
        ax1.set_xlabel('仿真时期')
        ax1.set_ylabel('策略比例')
        ax1.set_title('作物选择策略演化')
        ax1.legend(title='作物策略')
        ax1.grid(True, alpha=0.3)
    
    # 2. 农业收入比例分布演化
    ax2 = axes[0, 1]
    if 'Agricultural_Income_Ratio' in farmer_data.columns:
        income_evolution = farmer_data.groupby('Step')['Agricultural_Income_Ratio'].agg(['mean', 'std', 'min', 'max'])
        ax2.fill_between(income_evolution.index,
                        income_evolution['mean'] - income_evolution['std'],
                        income_evolution['mean'] + income_evolution['std'],
                        alpha=0.3, color='lightgreen')
        ax2.plot(income_evolution.index, income_evolution['mean'], 'o-', 
                color='darkgreen', linewidth=2, label='平均值')
        ax2.plot(income_evolution.index, income_evolution['min'], 's-', 
                color='red', linewidth=1, label='最小值')
        ax2.plot(income_evolution.index, income_evolution['max'], '^-', 
                color='blue', linewidth=1, label='最大值')
        ax2.set_xlabel('仿真时期')
        ax2.set_ylabel('农业收入比例')
        ax2.set_title('农业收入比例分布演化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 作物选择与收入关系演化
    ax3 = axes[1, 0]
    if all(col in farmer_data.columns for col in ['Agricultural_Income_Ratio', 'Total_Income']):
        correlation_evolution = []
        for step in sorted(farmer_data['Step'].unique()):
            step_data = farmer_data[farmer_data['Step'] == step]
            if len(step_data) > 1:
                corr = step_data['Agricultural_Income_Ratio'].corr(step_data['Total_Income'])
                correlation_evolution.append(corr)
            else:
                correlation_evolution.append(np.nan)
        
        steps = sorted(farmer_data['Step'].unique())
        ax3.plot(steps, correlation_evolution, 'o-', color='purple', linewidth=2)
        ax3.set_xlabel('仿真时期')
        ax3.set_ylabel('相关系数')
        ax3.set_title('作物选择与收入关系演化')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 作物选择策略转换矩阵
    ax4 = axes[1, 1]
    if 'Agricultural_Income_Ratio' in farmer_data.columns:
        strategy_transitions = []
        steps = sorted(farmer_data['Step'].unique())
        for i in range(len(steps)-1):
            current_step = farmer_data[farmer_data['Step'] == steps[i]]
            next_step = farmer_data[farmer_data['Step'] == steps[i+1]]
            
            current_strategy = np.where(
                current_step['Agricultural_Income_Ratio'] > 0.95, '粮食作物',
                np.where(current_step['Agricultural_Income_Ratio'] > 0.85, '混合策略', '经济作物')
            )
            next_strategy = np.where(
                next_step['Agricultural_Income_Ratio'] > 0.95, '粮食作物',
                np.where(next_step['Agricultural_Income_Ratio'] > 0.85, '混合策略', '经济作物')
            )
            
            if len(current_strategy) > 0 and len(next_strategy) > 0:
                transition_rate = np.mean(current_strategy != next_strategy)
                strategy_transitions.append(transition_rate)
            else:
                strategy_transitions.append(0)
        
        ax4.plot(steps[:-1], strategy_transitions, 'o-', color='orange', linewidth=2)
        ax4.set_xlabel('仿真时期')
        ax4.set_ylabel('策略转换率')
        ax4.set_title('作物选择策略转换率')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crop_choice_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("作物选择博弈动态演化分析完成")

def analyze_land_transfer_dynamics(agent_data, output_dir):
    """分析土地流转博弈的动态演化"""
    print("开始分析土地流转博弈动态演化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('土地流转博弈动态演化分析', fontsize=16, fontweight='bold')
    
    # 1. 土地经营面积演化
    ax1 = axes[0, 0]
    if 'Operated_Land_Area' in agent_data.columns:
        land_evolution = agent_data.groupby('Step')['Operated_Land_Area'].agg(['mean', 'std', 'min', 'max'])
        ax1.fill_between(land_evolution.index,
                        land_evolution['mean'] - land_evolution['std'],
                        land_evolution['mean'] + land_evolution['std'],
                        alpha=0.3, color='lightblue')
        ax1.plot(land_evolution.index, land_evolution['mean'], 'o-', 
                color='darkblue', linewidth=2, label='平均值')
        ax1.plot(land_evolution.index, land_evolution['min'], 's-', 
                color='red', linewidth=1, label='最小值')
        ax1.plot(land_evolution.index, land_evolution['max'], '^-', 
                color='green', linewidth=1, label='最大值')
        ax1.set_xlabel('仿真时期')
        ax1.set_ylabel('经营土地面积')
        ax1.set_title('土地经营面积演化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 土地集中度演化
    ax2 = axes[0, 1]
    if 'Operated_Land_Area' in agent_data.columns:
        gini_evolution = []
        for step in sorted(agent_data['Step'].unique()):
            step_data = agent_data[agent_data['Step'] == step]
            land_areas = step_data['Operated_Land_Area'].values
            if len(land_areas) > 1:
                land_areas = np.sort(land_areas)
                n = len(land_areas)
                cumsum = np.cumsum(land_areas)
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                gini_evolution.append(gini)
            else:
                gini_evolution.append(0)
        
        steps = sorted(agent_data['Step'].unique())
        ax2.plot(steps, gini_evolution, 'o-', color='darkred', linewidth=2)
        ax2.set_xlabel('仿真时期')
        ax2.set_ylabel('土地集中度（基尼系数）')
        ax2.set_title('土地集中度演化')
        ax2.grid(True, alpha=0.3)
    
    # 3. 农户vs企业土地经营对比
    ax3 = axes[1, 0]
    if 'Operated_Land_Area' in agent_data.columns:
        farmer_land_evolution = agent_data[agent_data['AgentID'].str.startswith('farmer')].groupby('Step')['Operated_Land_Area'].mean()
        enterprise_land_evolution = agent_data[agent_data['AgentID'].str.startswith('enterprise')].groupby('Step')['Operated_Land_Area'].mean()
        
        ax3.plot(farmer_land_evolution.index, farmer_land_evolution.values, 'o-', 
                color='lightgreen', linewidth=2, label='农户')
        if not enterprise_land_evolution.empty:
            ax3.plot(enterprise_land_evolution.index, enterprise_land_evolution.values, 's-', 
                    color='lightcoral', linewidth=2, label='农业企业')
        ax3.set_xlabel('仿真时期')
        ax3.set_ylabel('平均经营土地面积')
        ax3.set_title('农户vs企业土地经营对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 土地流转活跃度演化
    ax4 = axes[1, 1]
    if 'Operated_Land_Area' in agent_data.columns:
        land_std_evolution = agent_data.groupby('Step')['Operated_Land_Area'].std()
        ax4.plot(land_std_evolution.index, land_std_evolution.values, 'o-', 
                color='orange', linewidth=2)
        ax4.set_xlabel('仿真时期')
        ax4.set_ylabel('土地面积标准差（流转活跃度）')
        ax4.set_title('土地流转活跃度演化')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'land_transfer_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("土地流转博弈动态演化分析完成")

def create_detailed_game_report(agent_data, model_data, output_dir):
    """创建详细的博弈分析报告"""
    print("开始生成详细博弈分析报告...")
    
    report_dir = output_dir / "detailed_game_analysis"
    report_dir.mkdir(exist_ok=True)
    
    # 生成详细统计报告
    detailed_stats = {
        "博弈数据概览": {
            "总记录数": len(agent_data),
            "仿真时期数": agent_data['Step'].nunique(),
            "农户数量": len(agent_data[agent_data['AgentID'].str.startswith('farmer')]),
            "企业数量": len(agent_data[agent_data['AgentID'].str.startswith('enterprise')]),
            "政府数量": len(agent_data[agent_data['AgentID'].str.startswith('government')])
        },
        "作物选择博弈详细统计": {
            "平均农业收入比例": agent_data['Agricultural_Income_Ratio'].mean(),
            "农业收入比例标准差": agent_data['Agricultural_Income_Ratio'].std(),
            "高农业收入比例农户比例": (agent_data['Agricultural_Income_Ratio'] > 0.9).mean(),
            "粮食作物偏好农户比例": (agent_data['Agricultural_Income_Ratio'] > 0.95).mean(),
            "经济作物偏好农户比例": (agent_data['Agricultural_Income_Ratio'] < 0.85).mean()
        },
        "土地流转博弈详细统计": {
            "平均经营土地面积": agent_data['Operated_Land_Area'].mean(),
            "土地面积标准差": agent_data['Operated_Land_Area'].std(),
            "大土地经营者比例": (agent_data['Operated_Land_Area'] > 20).mean(),
            "小土地经营者比例": (agent_data['Operated_Land_Area'] < 10).mean()
        },
        "生计策略详细分布": agent_data['Livelihood_Strategy'].value_counts().to_dict(),
        "收入分布统计": {
            "平均总收入": agent_data['Total_Income'].mean(),
            "收入标准差": agent_data['Total_Income'].std()
        }
    }
    
    # 保存详细统计报告
    with open(report_dir / "detailed_game_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
    
    # 生成详细可视化
    analyze_period_game_results(agent_data, report_dir)
    analyze_crop_choice_dynamics(agent_data, report_dir)
    analyze_land_transfer_dynamics(agent_data, report_dir)
    
    print(f"详细博弈分析报告已保存到: {report_dir}")

def main():
    """主函数"""
    print("开始详细博弈结果可视化分析...")
    
    setup_plotting()
    
    try:
        agent_data, model_data, results_dir = load_simulation_data()
        
        output_dir = results_dir / "detailed_game_visualization"
        output_dir.mkdir(exist_ok=True)
        
        create_detailed_game_report(agent_data, model_data, output_dir)
        
        print(f"详细博弈可视化分析完成！结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 