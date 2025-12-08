# -*- coding: utf-8 -*-
"""
博弈过程数据分析模块
专门提取每一期博弈后的各种指标
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
import ast
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

def analyze_crop_choice_by_period(agent_data):
    """分析每一期的作物选择情况"""
    print("开始分析每一期的作物选择情况...")
    
    # 分离农户和企业数据
    farmer_data = agent_data[agent_data['AgentID'].str.startswith('farmer')]
    enterprise_data = agent_data[agent_data['AgentID'].str.startswith('enterprise')]
    
    crop_choice_results = []
    
    for step in sorted(agent_data['Step'].unique()):
        step_farmers = farmer_data[farmer_data['Step'] == step]
        step_enterprises = enterprise_data[enterprise_data['Step'] == step]
        
        # 基于农业收入比例推断作物选择
        # 高农业收入比例（>0.9）可能表示选择粮食作物（享受补贴）
        # 低农业收入比例（<0.8）可能表示选择经济作物
        
        if not step_farmers.empty:
            # 农户作物选择分析
            farmer_grain_ratio = (step_farmers['Agricultural_Income_Ratio'] > 0.9).mean()
            farmer_cash_ratio = (step_farmers['Agricultural_Income_Ratio'] < 0.8).mean()
            farmer_mixed_ratio = 1 - farmer_grain_ratio - farmer_cash_ratio
            
            # 农户生计策略分布
            farmer_strategy_dist = step_farmers['Livelihood_Strategy'].value_counts(normalize=True).to_dict()
        else:
            farmer_grain_ratio = farmer_cash_ratio = farmer_mixed_ratio = 0
            farmer_strategy_dist = {}
        
        if not step_enterprises.empty:
            # 企业策略分析（基于目标回报率）
            enterprise_high_return = (step_enterprises['Target_Return_Rate'] > 0.15).mean()
            enterprise_stable = ((step_enterprises['Target_Return_Rate'] <= 0.15) & 
                               (step_enterprises['Target_Return_Rate'] > 0.10)).mean()
            enterprise_conservative = (step_enterprises['Target_Return_Rate'] <= 0.10).mean()
        else:
            enterprise_high_return = enterprise_stable = enterprise_conservative = 0
        
        crop_choice_results.append({
            'Step': step,
            'Farmer_Grain_Ratio': farmer_grain_ratio,
            'Farmer_Cash_Ratio': farmer_cash_ratio,
            'Farmer_Mixed_Ratio': farmer_mixed_ratio,
            'Enterprise_High_Return_Ratio': enterprise_high_return,
            'Enterprise_Stable_Ratio': enterprise_stable,
            'Enterprise_Conservative_Ratio': enterprise_conservative,
            'Farmer_Strategy_Distribution': farmer_strategy_dist
        })
    
    return pd.DataFrame(crop_choice_results)

def analyze_land_market_by_period(agent_data, model_data):
    """分析每一期的土地市场情况"""
    print("开始分析每一期的土地市场情况...")
    
    land_market_results = []
    
    for step in sorted(agent_data['Step'].unique()):
        step_data = agent_data[agent_data['Step'] == step]
        
        # 土地经营情况
        avg_land_area = step_data['Operated_Land_Area'].mean()
        land_area_std = step_data['Operated_Land_Area'].std()
        
        # 农户vs企业土地经营对比
        farmer_land = step_data[step_data['AgentID'].str.startswith('farmer')]['Operated_Land_Area']
        enterprise_land = step_data[step_data['AgentID'].str.startswith('enterprise')]['Operated_Land_Area']
        
        avg_farmer_land = farmer_land.mean() if not farmer_land.empty else 0
        avg_enterprise_land = enterprise_land.mean() if not enterprise_land.empty else 0
        
        # 土地集中度（基尼系数）
        land_areas = step_data['Operated_Land_Area'].values
        if len(land_areas) > 1:
            land_areas = np.sort(land_areas)
            n = len(land_areas)
            cumsum = np.cumsum(land_areas)
            gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            gini_coefficient = 0
        
        # 从model_data获取土地流转信息
        if model_data is not None and step < len(model_data):
            land_transactions = model_data.iloc[step]['Land_Rental_Transactions']
            avg_rental_price = model_data.iloc[step]['Average_Land_Rental_Price']
        else:
            land_transactions = 0
            avg_rental_price = 0
        
        land_market_results.append({
            'Step': step,
            'Avg_Land_Area': avg_land_area,
            'Land_Area_Std': land_area_std,
            'Avg_Farmer_Land': avg_farmer_land,
            'Avg_Enterprise_Land': avg_enterprise_land,
            'Land_Gini_Coefficient': gini_coefficient,
            'Land_Rental_Transactions': land_transactions,
            'Avg_Land_Rental_Price': avg_rental_price
        })
    
    return pd.DataFrame(land_market_results)

def analyze_income_and_prices_by_period(agent_data, model_data):
    """分析每一期的收入和价格情况"""
    print("开始分析每一期的收入和价格情况...")
    
    income_price_results = []
    
    for step in sorted(agent_data['Step'].unique()):
        step_data = agent_data[agent_data['Step'] == step]
        
        # 收入分析
        avg_total_income = step_data['Total_Income'].mean()
        income_std = step_data['Total_Income'].std()
        
        # 农户vs企业收入对比
        farmer_income = step_data[step_data['AgentID'].str.startswith('farmer')]['Total_Income']
        enterprise_income = step_data[step_data['AgentID'].str.startswith('enterprise')]['Total_Income']
        
        avg_farmer_income = farmer_income.mean() if not farmer_income.empty else 0
        avg_enterprise_income = enterprise_income.mean() if not enterprise_income.empty else 0
        
        # 农业收入比例分析（作为作物选择的代理指标）
        avg_agr_income_ratio = step_data['Agricultural_Income_Ratio'].mean()
        agr_income_ratio_std = step_data['Agricultural_Income_Ratio'].std()
        
        # 从model_data获取其他指标
        if model_data is not None and step < len(model_data):
            income_gini = model_data.iloc[step]['Income_Gini_Coefficient_Rural']
            subsidy_incidence = model_data.iloc[step]['Subsidy_Incidence_Ratio']
        else:
            income_gini = 0
            subsidy_incidence = 0
        
        income_price_results.append({
            'Step': step,
            'Avg_Total_Income': avg_total_income,
            'Income_Std': income_std,
            'Avg_Farmer_Income': avg_farmer_income,
            'Avg_Enterprise_Income': avg_enterprise_income,
            'Avg_Agr_Income_Ratio': avg_agr_income_ratio,
            'Agr_Income_Ratio_Std': agr_income_ratio_std,
            'Income_Gini_Coefficient': income_gini,
            'Subsidy_Incidence_Ratio': subsidy_incidence
        })
    
    return pd.DataFrame(income_price_results)

def create_period_summary_report(agent_data, model_data, output_dir):
    """创建每一期的详细汇总报告"""
    print("开始创建每一期的详细汇总报告...")
    
    # 分析各种指标
    crop_choice_df = analyze_crop_choice_by_period(agent_data)
    land_market_df = analyze_land_market_by_period(agent_data, model_data)
    income_price_df = analyze_income_and_prices_by_period(agent_data, model_data)
    
    # 合并所有指标
    summary_df = pd.merge(crop_choice_df, land_market_df, on='Step')
    summary_df = pd.merge(summary_df, income_price_df, on='Step')
    
    # 保存汇总报告
    summary_df.to_csv(output_dir / 'period_summary_report.csv', index=False, encoding='utf-8')
    
    # 创建可视化报告
    create_period_visualization(summary_df, output_dir)
    
    # 生成统计报告
    create_statistical_report(summary_df, output_dir)
    
    print(f"每一期详细汇总报告已保存到: {output_dir}")
    return summary_df

def create_period_visualization(summary_df, output_dir):
    """创建每一期的可视化报告"""
    print("开始创建可视化报告...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('每一期博弈结果详细分析', fontsize=16, fontweight='bold')
    
    # 1. 农户作物选择比例演化
    ax1 = axes[0, 0]
    ax1.plot(summary_df['Step'], summary_df['Farmer_Grain_Ratio'], 'o-', 
             color='lightgreen', linewidth=2, label='粮食作物')
    ax1.plot(summary_df['Step'], summary_df['Farmer_Cash_Ratio'], 's-', 
             color='orange', linewidth=2, label='经济作物')
    ax1.plot(summary_df['Step'], summary_df['Farmer_Mixed_Ratio'], '^-', 
             color='purple', linewidth=2, label='混合策略')
    ax1.set_xlabel('仿真时期')
    ax1.set_ylabel('比例')
    ax1.set_title('农户作物选择比例演化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 企业策略分布演化
    ax2 = axes[0, 1]
    ax2.plot(summary_df['Step'], summary_df['Enterprise_High_Return_Ratio'], 'o-', 
             color='lightcoral', linewidth=2, label='高回报策略')
    ax2.plot(summary_df['Step'], summary_df['Enterprise_Stable_Ratio'], 's-', 
             color='skyblue', linewidth=2, label='稳健策略')
    ax2.plot(summary_df['Step'], summary_df['Enterprise_Conservative_Ratio'], '^-', 
             color='lightyellow', linewidth=2, label='保守策略')
    ax2.set_xlabel('仿真时期')
    ax2.set_ylabel('比例')
    ax2.set_title('农业企业策略分布演化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 土地经营面积演化
    ax3 = axes[0, 2]
    ax3.plot(summary_df['Step'], summary_df['Avg_Land_Area'], 'o-', 
             color='darkblue', linewidth=2, label='平均土地面积')
    ax3.plot(summary_df['Step'], summary_df['Avg_Farmer_Land'], 's-', 
             color='lightgreen', linewidth=2, label='农户平均土地')
    ax3.plot(summary_df['Step'], summary_df['Avg_Enterprise_Land'], '^-', 
             color='lightcoral', linewidth=2, label='企业平均土地')
    ax3.set_xlabel('仿真时期')
    ax3.set_ylabel('土地面积')
    ax3.set_title('土地经营面积演化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 土地集中度演化
    ax4 = axes[1, 0]
    ax4.plot(summary_df['Step'], summary_df['Land_Gini_Coefficient'], 'o-', 
             color='darkred', linewidth=2)
    ax4.set_xlabel('仿真时期')
    ax4.set_ylabel('基尼系数')
    ax4.set_title('土地集中度演化')
    ax4.grid(True, alpha=0.3)
    
    # 5. 土地流转市场活跃度
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(summary_df['Step'], summary_df['Land_Rental_Transactions'], 
                     'o-', color='darkgreen', linewidth=2, label='土地流转交易量')
    line2 = ax5_twin.plot(summary_df['Step'], summary_df['Avg_Land_Rental_Price'], 
                          's-', color='darkred', linewidth=2, label='平均土地租金')
    ax5.set_xlabel('仿真时期')
    ax5.set_ylabel('土地流转交易量', color='darkgreen')
    ax5_twin.set_ylabel('平均土地租金', color='darkred')
    ax5.set_title('土地流转市场活跃度')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. 收入分布演化
    ax6 = axes[1, 2]
    ax6.plot(summary_df['Step'], summary_df['Avg_Total_Income'], 'o-', 
             color='darkblue', linewidth=2, label='平均总收入')
    ax6.plot(summary_df['Step'], summary_df['Avg_Farmer_Income'], 's-', 
             color='lightgreen', linewidth=2, label='农户平均收入')
    ax6.plot(summary_df['Step'], summary_df['Avg_Enterprise_Income'], '^-', 
             color='lightcoral', linewidth=2, label='企业平均收入')
    ax6.set_xlabel('仿真时期')
    ax6.set_ylabel('收入')
    ax6.set_title('收入分布演化')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 农业收入比例演化
    ax7 = axes[2, 0]
    ax7.plot(summary_df['Step'], summary_df['Avg_Agr_Income_Ratio'], 'o-', 
             color='darkgreen', linewidth=2)
    ax7.fill_between(summary_df['Step'],
                     summary_df['Avg_Agr_Income_Ratio'] - summary_df['Agr_Income_Ratio_Std'],
                     summary_df['Avg_Agr_Income_Ratio'] + summary_df['Agr_Income_Ratio_Std'],
                     alpha=0.3, color='lightgreen')
    ax7.set_xlabel('仿真时期')
    ax7.set_ylabel('农业收入比例')
    ax7.set_title('农业收入比例演化')
    ax7.grid(True, alpha=0.3)
    
    # 8. 收入不平等演化
    ax8 = axes[2, 1]
    ax8.plot(summary_df['Step'], summary_df['Income_Gini_Coefficient'], 'o-', 
             color='darkred', linewidth=2)
    ax8.set_xlabel('仿真时期')
    ax8.set_ylabel('收入基尼系数')
    ax8.set_title('收入不平等演化')
    ax8.grid(True, alpha=0.3)
    
    # 9. 补贴影响演化
    ax9 = axes[2, 2]
    ax9.plot(summary_df['Step'], summary_df['Subsidy_Incidence_Ratio'], 'o-', 
             color='orange', linewidth=2)
    ax9.set_xlabel('仿真时期')
    ax9.set_ylabel('补贴影响率')
    ax9.set_title('补贴政策影响演化')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'period_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_report(summary_df, output_dir):
    """创建统计报告"""
    print("开始创建统计报告...")
    
    # 计算关键统计指标
    stats_report = {
        "数据概览": {
            "总仿真时期数": len(summary_df),
            "分析指标数": len(summary_df.columns) - 1  # 减去Step列
        },
        "作物选择博弈统计": {
            "平均粮食作物种植比例": summary_df['Farmer_Grain_Ratio'].mean(),
            "平均经济作物种植比例": summary_df['Farmer_Cash_Ratio'].mean(),
            "平均混合策略比例": summary_df['Farmer_Mixed_Ratio'].mean(),
            "粮食作物种植比例变化率": summary_df['Farmer_Grain_Ratio'].std(),
            "经济作物种植比例变化率": summary_df['Farmer_Cash_Ratio'].std()
        },
        "农业企业策略统计": {
            "平均高回报策略比例": summary_df['Enterprise_High_Return_Ratio'].mean(),
            "平均稳健策略比例": summary_df['Enterprise_Stable_Ratio'].mean(),
            "平均保守策略比例": summary_df['Enterprise_Conservative_Ratio'].mean()
        },
        "土地市场统计": {
            "平均经营土地面积": summary_df['Avg_Land_Area'].mean(),
            "平均土地集中度": summary_df['Land_Gini_Coefficient'].mean(),
            "平均土地流转交易量": summary_df['Land_Rental_Transactions'].mean(),
            "平均土地租金": summary_df['Avg_Land_Rental_Price'].mean()
        },
        "收入分布统计": {
            "平均总收入": summary_df['Avg_Total_Income'].mean(),
            "平均农户收入": summary_df['Avg_Farmer_Income'].mean(),
            "平均企业收入": summary_df['Avg_Enterprise_Income'].mean(),
            "平均收入不平等": summary_df['Income_Gini_Coefficient'].mean()
        },
        "政策效果统计": {
            "平均农业收入比例": summary_df['Avg_Agr_Income_Ratio'].mean(),
            "平均补贴影响率": summary_df['Subsidy_Incidence_Ratio'].mean()
        }
    }
    
    # 保存统计报告
    with open(output_dir / 'statistical_report.json', 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    
    # 创建关键指标表格
    key_indicators = summary_df[['Step', 'Farmer_Grain_Ratio', 'Farmer_Cash_Ratio', 
                                'Avg_Land_Area', 'Land_Gini_Coefficient', 
                                'Avg_Total_Income', 'Income_Gini_Coefficient']].copy()
    key_indicators.columns = ['时期', '农户粮食作物比例', '农户经济作物比例', 
                             '平均土地面积', '土地集中度', '平均总收入', '收入不平等']
    key_indicators.to_csv(output_dir / 'key_indicators_table.csv', index=False, encoding='utf-8')
    
    print("统计报告已保存")

def main():
    """主函数"""
    print("开始博弈过程数据分析...")
    
    setup_plotting()
    
    try:
        # 加载数据
        agent_data, model_data, results_dir = load_simulation_data()
        
        # 创建输出目录
        output_dir = results_dir / "game_process_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # 创建每一期的详细汇总报告
        summary_df = create_period_summary_report(agent_data, model_data, output_dir)
        
        print(f"博弈过程数据分析完成！结果保存在: {output_dir}")
        print("\n=== 关键发现 ===")
        print(f"1. 农户平均粮食作物种植比例: {summary_df['Farmer_Grain_Ratio'].mean():.2%}")
        print(f"2. 农户平均经济作物种植比例: {summary_df['Farmer_Cash_Ratio'].mean():.2%}")
        print(f"3. 平均土地集中度: {summary_df['Land_Gini_Coefficient'].mean():.3f}")
        print(f"4. 平均收入不平等: {summary_df['Income_Gini_Coefficient'].mean():.3f}")
        print(f"5. 平均农业收入比例: {summary_df['Avg_Agr_Income_Ratio'].mean():.3f}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 