# -*- coding: utf-8 -*-
"""
微观博弈结果可视化模块
专门用于分析作物选择博弈和土地流转的详细情况
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.font_config import setup_chinese_fonts

def setup_plotting():
    """设置绘图参数，解决字体显示问题"""
    setup_chinese_fonts()  # 使用统一的字体配置
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

def load_simulation_data():
    """加载仿真数据"""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    results_dir = script_dir.parent / "results"
    
    # 查找最新的结果目录
    timestamp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('2025-')]
    if not timestamp_dirs:
        raise FileNotFoundError("未找到仿真结果目录")
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    print(f"使用结果目录: {latest_dir}")
    
    # 加载数据
    agent_data_file = latest_dir / "agent_data.csv"
    model_data_file = latest_dir / "model_data.csv"
    
    if not agent_data_file.exists():
        raise FileNotFoundError(f"未找到agent_data.csv文件: {agent_data_file}")
    
    agent_data = pd.read_csv(agent_data_file)
    model_data = pd.read_csv(model_data_file) if model_data_file.exists() else None
    
    return agent_data, model_data, latest_dir

def analyze_crop_choice_game(agent_data, output_dir):
    """分析作物选择博弈的详细情况"""
    print("开始分析作物选择博弈...")
    
    # 创建作物选择博弈分析图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('作物选择博弈微观分析', fontsize=16, fontweight='bold')
    
    # 1. 农户作物选择策略分布（按时期）
    ax1 = axes[0, 0]
    if 'Agricultural_Income_Ratio' in agent_data.columns:
        # 使用农业收入比例作为作物选择的代理指标
        # 高农业收入比例可能表示选择粮食作物（享受补贴）
        crop_choice_data = agent_data[agent_data['AgentID'].str.startswith('farmer')].copy()
        crop_choice_data['Crop_Strategy'] = np.where(
            crop_choice_data['Agricultural_Income_Ratio'] > 0.9, 
            '粮食作物偏好', '经济作物偏好'
        )
        
        strategy_by_step = crop_choice_data.groupby(['Step', 'Crop_Strategy']).size().unstack(fill_value=0)
        strategy_by_step.plot(kind='bar', stacked=True, ax=ax1, 
                            color=['lightgreen', 'orange'], alpha=0.8)
        ax1.set_title('农户作物选择策略分布（按时期）')
        ax1.set_xlabel('仿真时期')
        ax1.set_ylabel('农户数量')
        ax1.legend(title='作物策略')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. 农业企业作物选择分析
    ax2 = axes[0, 1]
    enterprise_data = agent_data[agent_data['AgentID'].str.startswith('enterprise')]
    if not enterprise_data.empty and 'Target_Return_Rate' in enterprise_data.columns:
        # 分析企业的目标回报率与作物选择的关系
        enterprise_data['Return_Strategy'] = np.where(
            enterprise_data['Target_Return_Rate'] > 0.15, 
            '高回报策略', '稳健策略'
        )
        
        strategy_counts = enterprise_data['Return_Strategy'].value_counts()
        ax2.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
                colors=['lightcoral', 'skyblue'], startangle=90)
        ax2.set_title('农业企业策略分布')
    
    # 3. 作物选择与收入关系
    ax3 = axes[1, 0]
    if 'Agricultural_Income_Ratio' in agent_data.columns and 'Total_Income' in agent_data.columns:
        farmer_data = agent_data[agent_data['AgentID'].str.startswith('farmer')]
        ax3.scatter(farmer_data['Agricultural_Income_Ratio'], farmer_data['Total_Income'],
                   alpha=0.6, c='green', s=20)
        ax3.set_xlabel('农业收入比例（作物选择代理指标）')
        ax3.set_ylabel('总收入')
        ax3.set_title('作物选择与收入关系')
        ax3.grid(True, alpha=0.3)
    
    # 4. 作物选择策略演化
    ax4 = axes[1, 1]
    if 'Agricultural_Income_Ratio' in agent_data.columns:
        # 计算每个时期的平均农业收入比例（作为作物选择趋势）
        avg_agr_ratio = agent_data[agent_data['AgentID'].str.startswith('farmer')].groupby('Step')['Agricultural_Income_Ratio'].mean()
        ax4.plot(avg_agr_ratio.index, avg_agr_ratio.values, 'o-', color='darkgreen', linewidth=2)
        ax4.set_xlabel('仿真时期')
        ax4.set_ylabel('平均农业收入比例')
        ax4.set_title('作物选择策略演化趋势')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crop_choice_game_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("作物选择博弈分析完成")

def analyze_land_transfer_game(agent_data, output_dir):
    """分析土地流转博弈的详细情况"""
    print("开始分析土地流转博弈...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('土地流转博弈微观分析', fontsize=16, fontweight='bold')
    
    # 1. 土地经营面积分布
    ax1 = axes[0, 0]
    if 'Operated_Land_Area' in agent_data.columns:
        farmer_land = agent_data[agent_data['AgentID'].str.startswith('farmer')]['Operated_Land_Area']
        enterprise_land = agent_data[agent_data['AgentID'].str.startswith('enterprise')]['Operated_Land_Area']
        
        ax1.hist(farmer_land, bins=20, alpha=0.7, label='农户', color='lightgreen')
        if not enterprise_land.empty:
            ax1.hist(enterprise_land, bins=20, alpha=0.7, label='农业企业', color='lightcoral')
        ax1.set_xlabel('经营土地面积')
        ax1.set_ylabel('频次')
        ax1.set_title('土地经营面积分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 土地流转活跃度（按时期）
    ax2 = axes[0, 1]
    if 'Operated_Land_Area' in agent_data.columns:
        # 计算每个时期的土地经营面积变化（作为流转活跃度的代理指标）
        land_changes = agent_data.groupby('Step')['Operated_Land_Area'].agg(['mean', 'std'])
        ax2.plot(land_changes.index, land_changes['std'], 'o-', color='orange', linewidth=2)
        ax2.set_xlabel('仿真时期')
        ax2.set_ylabel('土地面积标准差（流转活跃度）')
        ax2.set_title('土地流转活跃度变化')
        ax2.grid(True, alpha=0.3)
    
    # 3. 生计策略与土地经营关系
    ax3 = axes[1, 0]
    if 'Livelihood_Strategy' in agent_data.columns and 'Operated_Land_Area' in agent_data.columns:
        strategy_land = agent_data.groupby('Livelihood_Strategy')['Operated_Land_Area'].mean()
        bars = ax3.bar(strategy_land.index, strategy_land.values, 
                      color=['lightgreen', 'lightblue', 'lightcoral'], alpha=0.8)
        ax3.set_xlabel('生计策略')
        ax3.set_ylabel('平均经营土地面积')
        ax3.set_title('生计策略与土地经营关系')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. 土地流转与收入关系
    ax4 = axes[1, 1]
    if 'Operated_Land_Area' in agent_data.columns and 'Total_Income' in agent_data.columns:
        ax4.scatter(agent_data['Operated_Land_Area'], agent_data['Total_Income'],
                   alpha=0.6, c='purple', s=20)
        ax4.set_xlabel('经营土地面积')
        ax4.set_ylabel('总收入')
        ax4.set_title('土地经营与收入关系')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'land_transfer_game_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("土地流转博弈分析完成")

def analyze_agent_behavior_evolution(agent_data, output_dir):
    """分析智能体行为演化"""
    print("开始分析智能体行为演化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('智能体行为演化分析', fontsize=16, fontweight='bold')
    
    # 1. 生计策略演化
    ax1 = axes[0, 0]
    if 'Livelihood_Strategy' in agent_data.columns:
        strategy_evolution = agent_data.groupby(['Step', 'Livelihood_Strategy']).size().unstack(fill_value=0)
        strategy_evolution.plot(kind='line', ax=ax1, marker='o', linewidth=2)
        ax1.set_xlabel('仿真时期')
        ax1.set_ylabel('智能体数量')
        ax1.set_title('生计策略演化')
        ax1.legend(title='生计策略')
        ax1.grid(True, alpha=0.3)
    
    # 2. 收入结构演化
    ax2 = axes[0, 1]
    if 'Agricultural_Income_Ratio' in agent_data.columns:
        income_evolution = agent_data.groupby('Step')['Agricultural_Income_Ratio'].agg(['mean', 'std'])
        ax2.fill_between(income_evolution.index, 
                        income_evolution['mean'] - income_evolution['std'],
                        income_evolution['mean'] + income_evolution['std'],
                        alpha=0.3, color='lightgreen')
        ax2.plot(income_evolution.index, income_evolution['mean'], 'o-', 
                color='darkgreen', linewidth=2)
        ax2.set_xlabel('仿真时期')
        ax2.set_ylabel('农业收入比例')
        ax2.set_title('收入结构演化')
        ax2.grid(True, alpha=0.3)
    
    # 3. 资本积累演化
    ax3 = axes[1, 0]
    if 'Capital' in agent_data.columns:
        capital_evolution = agent_data.groupby('Step')['Capital'].agg(['mean', 'std'])
        ax3.fill_between(capital_evolution.index,
                        capital_evolution['mean'] - capital_evolution['std'],
                        capital_evolution['mean'] + capital_evolution['std'],
                        alpha=0.3, color='lightblue')
        ax3.plot(capital_evolution.index, capital_evolution['mean'], 'o-',
                color='darkblue', linewidth=2)
        ax3.set_xlabel('仿真时期')
        ax3.set_ylabel('平均资本')
        ax3.set_title('资本积累演化')
        ax3.grid(True, alpha=0.3)
    
    # 4. 风险偏好演化
    ax4 = axes[1, 1]
    if 'Transition_Probability' in agent_data.columns:
        risk_evolution = agent_data.groupby('Step')['Transition_Probability'].agg(['mean', 'std'])
        ax4.fill_between(risk_evolution.index,
                        risk_evolution['mean'] - risk_evolution['std'],
                        risk_evolution['mean'] + risk_evolution['std'],
                        alpha=0.3, color='lightcoral')
        ax4.plot(risk_evolution.index, risk_evolution['mean'], 'o-',
                color='darkred', linewidth=2)
        ax4.set_xlabel('仿真时期')
        ax4.set_ylabel('平均转换概率')
        ax4.set_title('风险偏好演化')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_behavior_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("智能体行为演化分析完成")

def create_micro_game_report(agent_data, model_data, output_dir):
    """创建微观博弈分析报告"""
    print("开始生成微观博弈分析报告...")
    
    # 创建报告目录
    report_dir = output_dir / "micro_game_analysis"
    report_dir.mkdir(exist_ok=True)
    
    # 生成统计报告
    report_stats = {
        "数据概览": {
            "总记录数": len(agent_data),
            "仿真时期数": agent_data['Step'].nunique(),
            "农户数量": len(agent_data[agent_data['AgentID'].str.startswith('farmer')]),
            "企业数量": len(agent_data[agent_data['AgentID'].str.startswith('enterprise')]),
            "政府数量": len(agent_data[agent_data['AgentID'].str.startswith('government')])
        },
        "作物选择博弈统计": {
            "平均农业收入比例": agent_data['Agricultural_Income_Ratio'].mean(),
            "农业收入比例标准差": agent_data['Agricultural_Income_Ratio'].std(),
            "高农业收入比例农户比例": (agent_data['Agricultural_Income_Ratio'] > 0.9).mean()
        },
        "土地流转博弈统计": {
            "平均经营土地面积": agent_data['Operated_Land_Area'].mean(),
            "土地面积标准差": agent_data['Operated_Land_Area'].std(),
            "大土地经营者比例": (agent_data['Operated_Land_Area'] > 20).mean()
        },
        "生计策略分布": agent_data['Livelihood_Strategy'].value_counts().to_dict()
    }
    
    # 保存统计报告
    with open(report_dir / "micro_game_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(report_stats, f, ensure_ascii=False, indent=2)
    
    # 生成可视化
    analyze_crop_choice_game(agent_data, report_dir)
    analyze_land_transfer_game(agent_data, report_dir)
    analyze_agent_behavior_evolution(agent_data, report_dir)
    
    # 创建综合报告
    create_comprehensive_report(agent_data, model_data, report_dir)
    
    print(f"微观博弈分析报告已保存到: {report_dir}")

def create_comprehensive_report(agent_data, model_data, report_dir):
    """创建综合报告"""
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('微观博弈综合分析报告', fontsize=18, fontweight='bold')
    
    # 1. 作物选择博弈时间序列
    ax1 = axes[0, 0]
    if 'Agricultural_Income_Ratio' in agent_data.columns:
        crop_choice_ts = agent_data.groupby('Step')['Agricultural_Income_Ratio'].agg(['mean', 'std'])
        ax1.plot(crop_choice_ts.index, crop_choice_ts['mean'], 'o-', color='darkgreen', linewidth=2, label='平均农业收入比例')
        ax1.fill_between(crop_choice_ts.index,
                        crop_choice_ts['mean'] - crop_choice_ts['std'],
                        crop_choice_ts['mean'] + crop_choice_ts['std'],
                        alpha=0.3, color='lightgreen')
        ax1.set_title('作物选择博弈演化')
        ax1.set_xlabel('仿真时期')
        ax1.set_ylabel('农业收入比例')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 土地流转博弈时间序列
    ax2 = axes[0, 1]
    if 'Operated_Land_Area' in agent_data.columns:
        land_transfer_ts = agent_data.groupby('Step')['Operated_Land_Area'].agg(['mean', 'std'])
        ax2.plot(land_transfer_ts.index, land_transfer_ts['mean'], 'o-', color='darkblue', linewidth=2, label='平均经营土地面积')
        ax2.fill_between(land_transfer_ts.index,
                        land_transfer_ts['mean'] - land_transfer_ts['std'],
                        land_transfer_ts['mean'] + land_transfer_ts['std'],
                        alpha=0.3, color='lightblue')
        ax2.set_title('土地流转博弈演化')
        ax2.set_xlabel('仿真时期')
        ax2.set_ylabel('经营土地面积')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 收入分布对比
    ax3 = axes[1, 0]
    if 'Total_Income' in agent_data.columns:
        farmer_income = agent_data[agent_data['AgentID'].str.startswith('farmer')]['Total_Income']
        enterprise_income = agent_data[agent_data['AgentID'].str.startswith('enterprise')]['Total_Income']
        
        ax3.hist(farmer_income, bins=30, alpha=0.7, label='农户', color='lightgreen', density=True)
        if not enterprise_income.empty:
            ax3.hist(enterprise_income, bins=30, alpha=0.7, label='农业企业', color='lightcoral', density=True)
        ax3.set_xlabel('总收入')
        ax3.set_ylabel('密度')
        ax3.set_title('收入分布对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 生计策略演化热图
    ax4 = axes[1, 1]
    if 'Livelihood_Strategy' in agent_data.columns:
        strategy_pivot = agent_data.groupby(['Step', 'Livelihood_Strategy']).size().unstack(fill_value=0)
        im = ax4.imshow(strategy_pivot.T, cmap='YlOrRd', aspect='auto')
        ax4.set_xlabel('仿真时期')
        ax4.set_ylabel('生计策略')
        ax4.set_title('生计策略演化热图')
        ax4.set_yticks(range(len(strategy_pivot.columns)))
        ax4.set_yticklabels(strategy_pivot.columns)
        plt.colorbar(im, ax=ax4)
    
    # 5. 博弈结果相关性分析
    ax5 = axes[2, 0]
    if all(col in agent_data.columns for col in ['Agricultural_Income_Ratio', 'Operated_Land_Area', 'Total_Income']):
        correlation_data = agent_data[['Agricultural_Income_Ratio', 'Operated_Land_Area', 'Total_Income']].corr()
        im = ax5.imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(correlation_data.columns)))
        ax5.set_yticks(range(len(correlation_data.columns)))
        ax5.set_xticklabels(correlation_data.columns, rotation=45)
        ax5.set_yticklabels(correlation_data.columns)
        ax5.set_title('博弈结果相关性')
        
        # 添加相关系数标签
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                text = ax5.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax5)
    
    # 6. 政策效果评估
    ax6 = axes[2, 1]
    if model_data is not None and 'Land_Rental_Transactions' in model_data.columns:
        policy_effects = model_data[['Land_Rental_Transactions', 'Average_Land_Rental_Price']]
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(policy_effects.index, policy_effects['Land_Rental_Transactions'], 
                         'o-', color='darkgreen', linewidth=2, label='土地流转交易量')
        line2 = ax6_twin.plot(policy_effects.index, policy_effects['Average_Land_Rental_Price'], 
                              's-', color='darkred', linewidth=2, label='平均土地租金')
        
        ax6.set_xlabel('仿真时期')
        ax6.set_ylabel('土地流转交易量', color='darkgreen')
        ax6_twin.set_ylabel('平均土地租金', color='darkred')
        ax6.set_title('政策效果评估')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / "comprehensive_micro_game_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始微观博弈结果可视化分析...")
    
    # 设置绘图参数
    setup_plotting()
    
    try:
        # 加载数据
        agent_data, model_data, results_dir = load_simulation_data()
        
        # 创建输出目录
        output_dir = results_dir / "micro_game_visualization"
        output_dir.mkdir(exist_ok=True)
        
        # 生成微观博弈分析报告
        create_micro_game_report(agent_data, model_data, output_dir)
        
        print(f"微观博弈可视化分析完成！结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 