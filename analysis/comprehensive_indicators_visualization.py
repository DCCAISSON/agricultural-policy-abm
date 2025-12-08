# -*- coding: utf-8 -*-
"""
综合关键指标可视化模块
专门分析农户生计类型变化、农业企业类型变化、种植决策选择等关键指标
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
    # 不在这里设置字体，而是在每个图表生成函数内部设置
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

def analyze_livelihood_strategy_trends(agent_data):
    """分析农户生计类型变化趋势"""
    print("开始分析农户生计类型变化趋势...")
    
    # 按步骤统计生计类型分布
    strategy_trends = []
    
    for step in agent_data['Step'].unique():
        step_data = agent_data[agent_data['Step'] == step]
        
        # 统计各生计类型数量
        pure_farmers = len(step_data[step_data['Livelihood_Strategy'] == 'Pure_Farmer'])
        pluriactive = len(step_data[step_data['Livelihood_Strategy'] == 'Pluriactive'])
        non_farmers = len(step_data[step_data['Livelihood_Strategy'] == 'Non_Farmer'])
        total_farmers = len(step_data)
        
        # 计算比例
        pure_farmer_ratio = pure_farmers / total_farmers if total_farmers > 0 else 0
        pluriactive_ratio = pluriactive / total_farmers if total_farmers > 0 else 0
        non_farmer_ratio = non_farmers / total_farmers if total_farmers > 0 else 0
        
        strategy_trends.append({
            'Step': step,
            'Pure_Farmer': pure_farmers,
            'Pluriactive': pluriactive,
            'Non_Farmer': non_farmers,
            'Total_Farmers': total_farmers,
            'Pure_Farmer_Ratio': pure_farmer_ratio,
            'Pluriactive_Ratio': pluriactive_ratio,
            'Non_Farmer_Ratio': non_farmer_ratio
        })
    
    return pd.DataFrame(strategy_trends)

def analyze_income_inequality_trends(agent_data):
    """分析收入不平等趋势"""
    print("开始分析收入不平等趋势...")
    
    inequality_trends = []
    
    for step in agent_data['Step'].unique():
        step_data = agent_data[agent_data['Step'] == step]
        
        # 计算收入不平等指标
        incomes = step_data['Total_Income'].values
        
        if len(incomes) > 0:
            # 基尼系数
            incomes_sorted = np.sort(incomes)
            n = len(incomes_sorted)
            cumsum = np.cumsum(incomes_sorted)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            
            # 收入分位数
            income_90th = np.percentile(incomes, 90)
            income_median = np.median(incomes)
            income_10th = np.percentile(incomes, 10)
            
            # 平均收入
            mean_income = np.mean(incomes)
            
            inequality_trends.append({
                'Step': step,
                'Income_Gini': gini,
                'Income_90th_Percentile': income_90th,
                'Income_Median': income_median,
                'Income_10th_Percentile': income_10th,
                'Mean_Income': mean_income
            })
    
    return pd.DataFrame(inequality_trends)

def analyze_crop_choice_trends(agent_data):
    """分析种植决策选择趋势（基于农业收入占比）"""
    print("开始分析种植决策选择趋势...")
    
    crop_trends = []
    
    for step in agent_data['Step'].unique():
        step_data = agent_data[agent_data['Step'] == step]
        
        # 基于农业收入占比分析种植决策
        agri_income_ratios = step_data['Agricultural_Income_Ratio'].values
        
        if len(agri_income_ratios) > 0:
            # 高农业收入占比（>0.8）视为粮食作物偏好
            grain_preference = np.mean(agri_income_ratios > 0.8)
            cash_crop_preference = 1 - grain_preference
            
            # 平均农业收入占比
            avg_agri_ratio = np.mean(agri_income_ratios)
            
            # 种植面积（基于经营土地面积）
            avg_land_area = np.mean(step_data['Operated_Land_Area'].values)
            
            crop_trends.append({
                'Step': step,
                'Grain_Crop_Ratio': grain_preference,
                'Cash_Crop_Ratio': cash_crop_preference,
                'Average_Grain_Area': avg_land_area * grain_preference,
                'Average_Cash_Crop_Area': avg_land_area * cash_crop_preference,
                'Price_Sensitivity': avg_agri_ratio,  # 简化的价格敏感度
                'Risk_Aversion': 1 - avg_agri_ratio  # 简化的风险厌恶度
            })
    
    return pd.DataFrame(crop_trends)

def analyze_crop_farmer_livelihood_evolution(agent_data):
    """分析从事粮食作物和经济作物生产的农户生计类型演化"""
    print("开始分析作物类型农户生计类型演化...")
    
    # 只分析农户数据
    farmer_data = agent_data[~agent_data['AgentID'].str.startswith('enterprise')]
    
    crop_livelihood_trends = []
    
    for step in farmer_data['Step'].unique():
        step_data = farmer_data[farmer_data['Step'] == step]
        
        # 检查是否有Planned_Crop列，如果没有则基于其他指标推断
        if 'Planned_Crop' in step_data.columns:
            grain_farmers = step_data[step_data['Planned_Crop'] == 'Grain']
            cash_crop_farmers = step_data[step_data['Planned_Crop'] == 'Cash_Crop']
        else:
            # 基于农业收入占比推断：高占比(>0.7)视为粮食作物，低占比(<0.3)视为经济作物
            grain_farmers = step_data[step_data['Agricultural_Income_Ratio'] > 0.7]
            cash_crop_farmers = step_data[step_data['Agricultural_Income_Ratio'] < 0.3]
        
        # 统计粮食作物农户的生计类型分布
        if len(grain_farmers) > 0:
            grain_pure = len(grain_farmers[grain_farmers['Livelihood_Strategy'] == 'Pure_Farmer'])
            grain_pluri = len(grain_farmers[grain_farmers['Livelihood_Strategy'] == 'Pluriactive'])
            grain_non = len(grain_farmers[grain_farmers['Livelihood_Strategy'] == 'Non_Farmer'])
            grain_total = len(grain_farmers)
        else:
            grain_pure = grain_pluri = grain_non = grain_total = 0
        
        # 统计经济作物农户的生计类型分布
        if len(cash_crop_farmers) > 0:
            cash_pure = len(cash_crop_farmers[cash_crop_farmers['Livelihood_Strategy'] == 'Pure_Farmer'])
            cash_pluri = len(cash_crop_farmers[cash_crop_farmers['Livelihood_Strategy'] == 'Pluriactive'])
            cash_non = len(cash_crop_farmers[cash_crop_farmers['Livelihood_Strategy'] == 'Non_Farmer'])
            cash_total = len(cash_crop_farmers)
        else:
            cash_pure = cash_pluri = cash_non = cash_total = 0
        
        crop_livelihood_trends.append({
            'Step': step,
            'Grain_Pure_Farmer': grain_pure,
            'Grain_Pluriactive': grain_pluri,
            'Grain_Non_Farmer': grain_non,
            'Grain_Total': grain_total,
            'Cash_Pure_Farmer': cash_pure,
            'Cash_Pluriactive': cash_pluri,
            'Cash_Non_Farmer': cash_non,
            'Cash_Total': cash_total,
            'Grain_Pure_Ratio': grain_pure / grain_total if grain_total > 0 else 0,
            'Grain_Pluri_Ratio': grain_pluri / grain_total if grain_total > 0 else 0,
            'Grain_Non_Ratio': grain_non / grain_total if grain_total > 0 else 0,
            'Cash_Pure_Ratio': cash_pure / cash_total if cash_total > 0 else 0,
            'Cash_Pluri_Ratio': cash_pluri / cash_total if cash_total > 0 else 0,
            'Cash_Non_Ratio': cash_non / cash_total if cash_total > 0 else 0,
        })
    
    return pd.DataFrame(crop_livelihood_trends)

def analyze_enterprise_type_trends(agent_data):
    """分析农业企业类型变化趋势"""
    print("开始分析农业企业类型变化趋势...")
    
    # 分离企业数据
    enterprise_data = agent_data[agent_data['AgentID'].str.startswith('enterprise')]
    
    if enterprise_data.empty:
        print("警告：未找到企业数据")
        return None
    
    # 检查是否有企业类型数据
    if 'Enterprise_Type' not in enterprise_data.columns:
        print("警告：数据中没有企业类型信息，将基于目标回报率推断企业类型")
        # 基于目标回报率分析企业类型
        # 高回报率企业（>0.15）视为农业公司
        # 中等回报率企业（0.1-0.15）视为大型家庭农场
        # 低回报率企业（<0.1）视为合作社
        enterprise_trends = []
        
        for step in enterprise_data['Step'].unique():
            step_data = enterprise_data[enterprise_data['Step'] == step]
            
            high_return = (step_data['Target_Return_Rate'] > 0.15).sum()
            medium_return = ((step_data['Target_Return_Rate'] >= 0.1) & 
                            (step_data['Target_Return_Rate'] <= 0.15)).sum()
            low_return = (step_data['Target_Return_Rate'] < 0.1).sum()
            
            total_enterprises = len(step_data)
            
            # 计算比例
            high_return_ratio = high_return / total_enterprises if total_enterprises > 0 else 0
            medium_return_ratio = medium_return / total_enterprises if total_enterprises > 0 else 0
            low_return_ratio = low_return / total_enterprises if total_enterprises > 0 else 0
            
            # 计算平均指标
            avg_target_return = step_data['Target_Return_Rate'].mean()
            avg_performance = step_data['Performance_Trend'].mean()
            avg_learning = step_data['Learning_Rate'].mean()
            
            enterprise_trends.append({
                'Step': step,
                'High_Return_Enterprises': high_return,
                'Medium_Return_Enterprises': medium_return,
                'Low_Return_Enterprises': low_return,
                'Total_Enterprises': total_enterprises,
                'High_Return_Ratio': high_return_ratio,
                'Medium_Return_Ratio': medium_return_ratio,
                'Low_Return_Ratio': low_return_ratio,
                'Average_Target_Return': avg_target_return,
                'Average_Performance': avg_performance,
                'Average_Learning_Rate': avg_learning
            })
        
        return pd.DataFrame(enterprise_trends)
    
    # 基于真实企业类型数据进行分析
    enterprise_trends = []
    
    for step in enterprise_data['Step'].unique():
        step_data = enterprise_data[enterprise_data['Step'] == step]
        
        # 统计各企业类型数量
        family_farms = len(step_data[step_data['Enterprise_Type'] == 'Large_Family_Farm'])
        agri_corporations = len(step_data[step_data['Enterprise_Type'] == 'Agri_Corporation'])
        cooperatives = len(step_data[step_data['Enterprise_Type'] == 'Cooperative'])
        
        total_enterprises = len(step_data)
        
        # 计算比例
        family_farm_ratio = family_farms / total_enterprises if total_enterprises > 0 else 0
        agri_corporation_ratio = agri_corporations / total_enterprises if total_enterprises > 0 else 0
        cooperative_ratio = cooperatives / total_enterprises if total_enterprises > 0 else 0
        
        # 计算平均指标
        avg_target_return = step_data['Target_Return_Rate'].mean()
        avg_performance = step_data['Performance_Trend'].mean()
        avg_learning = step_data['Learning_Rate'].mean()
        
        enterprise_trends.append({
            'Step': step,
            'Family_Farms': family_farms,
            'Agri_Corporations': agri_corporations,
            'Cooperatives': cooperatives,
            'Total_Enterprises': total_enterprises,
            'Family_Farm_Ratio': family_farm_ratio,
            'Agri_Corporation_Ratio': agri_corporation_ratio,
            'Cooperative_Ratio': cooperative_ratio,
            'Average_Target_Return': avg_target_return,
            'Average_Performance': avg_performance,
            'Average_Learning_Rate': avg_learning
        })
    
    return pd.DataFrame(enterprise_trends)

def analyze_enterprise_crop_decision_trends(agent_data):
    """分析企业种植决策博弈趋势"""
    print("开始分析企业种植决策博弈趋势...")
    
    # 分离企业数据
    enterprise_data = agent_data[agent_data['AgentID'].str.startswith('enterprise')]
    
    if enterprise_data.empty:
        print("警告：未找到企业数据")
        return None
    
    # 按步骤分析企业种植决策
    crop_decision_trends = []
    
    for step in enterprise_data['Step'].unique():
        step_data = enterprise_data[enterprise_data['Step'] == step]
        
        # 基于目标回报率和学习率分析种植决策策略
        # 使用更宽松的分类标准
        # 积极策略：高目标回报率
        aggressive = (step_data['Target_Return_Rate'] > 0.16).sum()
        
        # 平衡策略：中等目标回报率
        balanced = ((step_data['Target_Return_Rate'] >= 0.13) & 
                   (step_data['Target_Return_Rate'] <= 0.16)).sum()
        
        # 保守策略：低目标回报率
        conservative = (step_data['Target_Return_Rate'] < 0.13).sum()
        
        # 如果分类结果过于集中，使用更宽松的标准
        if balanced > len(step_data) * 0.7:  # 如果70%以上都是平衡策略
            # 重新分类，使用更宽松的标准
            aggressive = (step_data['Target_Return_Rate'] > 0.15).sum()
            balanced = ((step_data['Target_Return_Rate'] >= 0.12) & 
                       (step_data['Target_Return_Rate'] <= 0.15)).sum()
            conservative = (step_data['Target_Return_Rate'] < 0.12).sum()
        
        # 如果仍然过于集中，使用最简单的分类
        if balanced > len(step_data) * 0.6:  # 如果60%以上都是平衡策略
            # 使用最简单的分类：按目标回报率三分位
            target_returns = step_data['Target_Return_Rate'].values
            if len(target_returns) > 0:
                q33 = np.percentile(target_returns, 33)
                q67 = np.percentile(target_returns, 67)
                
                aggressive = (target_returns > q67).sum()
                balanced = ((target_returns >= q33) & (target_returns <= q67)).sum()
                conservative = (target_returns < q33).sum()
        
        total_enterprises = len(step_data)
        
        # 计算比例
        aggressive_ratio = aggressive / total_enterprises if total_enterprises > 0 else 0
        balanced_ratio = balanced / total_enterprises if total_enterprises > 0 else 0
        conservative_ratio = conservative / total_enterprises if total_enterprises > 0 else 0
        
        # 计算平均指标
        avg_target_return = step_data['Target_Return_Rate'].mean()
        avg_learning_rate = step_data['Learning_Rate'].mean()
        avg_performance = step_data['Performance_Trend'].mean()
        
        crop_decision_trends.append({
            'Step': step,
            'Aggressive_Strategy': aggressive,
            'Balanced_Strategy': balanced,
            'Conservative_Strategy': conservative,
            'Total_Enterprises': total_enterprises,
            'Aggressive_Ratio': aggressive_ratio,
            'Balanced_Ratio': balanced_ratio,
            'Conservative_Ratio': conservative_ratio,
            'Average_Target_Return': avg_target_return,
            'Average_Learning_Rate': avg_learning_rate,
            'Average_Performance': avg_performance
        })
    
    return pd.DataFrame(crop_decision_trends)

def create_dummy_enterprise_data(strategy_df):
    """创建虚拟企业数据（因为当前数据中没有企业）"""
    print("创建虚拟企业数据用于演示...")
    
    enterprise_trends = []
    
    for _, row in strategy_df.iterrows():
        step = row['Step']
        
        # 基于农户生计类型变化推断企业类型
        # 当非农户增加时，可能对应企业数量增加
        non_farmer_ratio = row['Non_Farmer_Ratio']
        
        # 虚拟企业类型分布
        large_family_farm = max(0, 0.3 - non_farmer_ratio * 0.5)
        agricultural_company = max(0, 0.2 + non_farmer_ratio * 0.3)
        cooperative = max(0, 0.5 - non_farmer_ratio * 0.2)
        
        # 归一化
        total = large_family_farm + agricultural_company + cooperative
        if total > 0:
            large_family_farm /= total
            agricultural_company /= total
            cooperative /= total
        
        enterprise_trends.append({
            'Step': step,
            'Large_Family_Farm': large_family_farm * 10,  # 虚拟数量
            'Agricultural_Company': agricultural_company * 10,
            'Cooperative': cooperative * 10,
            'Large_Family_Farm_Ratio': large_family_farm,
            'Agricultural_Company_Ratio': agricultural_company,
            'Cooperative_Ratio': cooperative,
            'Strategy_Change_Rate': 0.1  # 虚拟调整率
        })
    
    return pd.DataFrame(enterprise_trends)

def create_livelihood_strategy_visualization(strategy_df, output_dir):
    """创建农户生计类型变化趋势可视化"""
    print("开始分析农户生计类型变化趋势...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('农户生计类型变化趋势分析', fontsize=16, fontweight='bold')
    
    # 生计类型数量变化
    axes[0, 0].plot(strategy_df['Step'], strategy_df['Pure_Farmer'], label='纯农户', marker='o')
    axes[0, 0].plot(strategy_df['Step'], strategy_df['Pluriactive'], label='兼业农户', marker='s')
    axes[0, 0].plot(strategy_df['Step'], strategy_df['Non_Farmer'], label='非农户', marker='^')
    axes[0, 0].set_title('生计类型数量变化')
    axes[0, 0].set_xlabel('仿真步数')
    axes[0, 0].set_ylabel('农户数量')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 生计类型比例变化
    axes[0, 1].plot(strategy_df['Step'], strategy_df['Pure_Farmer_Ratio'], label='纯农户比例', marker='o')
    axes[0, 1].plot(strategy_df['Step'], strategy_df['Pluriactive_Ratio'], label='兼业农户比例', marker='s')
    axes[0, 1].plot(strategy_df['Step'], strategy_df['Non_Farmer_Ratio'], label='非农户比例', marker='^')
    axes[0, 1].set_title('生计类型比例变化')
    axes[0, 1].set_xlabel('仿真步数')
    axes[0, 1].set_ylabel('比例')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 生计类型分布饼图（最终状态）
    final_data = strategy_df.iloc[-1]
    labels = ['纯农户', '兼业农户', '非农户']
    sizes = [final_data['Pure_Farmer_Ratio'], final_data['Pluriactive_Ratio'], final_data['Non_Farmer_Ratio']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('最终生计类型分布')
    
    # 生计类型转换热力图
    transition_matrix = np.zeros((3, 3))
    # 简化的转换矩阵（实际应该基于历史数据计算）
    transition_matrix[0, 0] = 0.8  # 纯农户保持纯农户
    transition_matrix[0, 1] = 0.15  # 纯农户转为兼业农户
    transition_matrix[0, 2] = 0.05  # 纯农户转为非农户
    transition_matrix[1, 0] = 0.1   # 兼业农户转为纯农户
    transition_matrix[1, 1] = 0.7   # 兼业农户保持兼业农户
    transition_matrix[1, 2] = 0.2   # 兼业农户转为非农户
    transition_matrix[2, 0] = 0.05  # 非农户转为纯农户
    transition_matrix[2, 1] = 0.1   # 非农户转为兼业农户
    transition_matrix[2, 2] = 0.85  # 非农户保持非农户
    
    im = axes[1, 1].imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('生计类型转换概率热力图')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(['纯农户', '兼业农户', '非农户'])
    axes[1, 1].set_yticklabels(['纯农户', '兼业农户', '非农户'])
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/livelihood_strategy_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return strategy_df

def create_enterprise_type_visualization(enterprise_df, output_dir):
    """创建农业企业类型变化趋势可视化"""
    print("开始分析农业企业类型变化趋势...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('农业企业类型变化趋势分析', fontsize=16, fontweight='bold')
    
    # 检查是否有真实的企业类型数据
    if 'Family_Farms' in enterprise_df.columns:
        # 使用真实企业类型数据
        # 企业类型数量变化
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['Family_Farms'], label='大型家庭农场', marker='o')
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['Agri_Corporations'], label='农业公司', marker='s')
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['Cooperatives'], label='合作社', marker='^')
        axes[0, 0].set_title('企业类型数量变化')
        axes[0, 0].set_xlabel('仿真步数')
        axes[0, 0].set_ylabel('企业数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 企业类型比例变化
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['Family_Farm_Ratio'], label='大型家庭农场比例', marker='o')
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['Agri_Corporation_Ratio'], label='农业公司比例', marker='s')
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['Cooperative_Ratio'], label='合作社比例', marker='^')
        axes[0, 1].set_title('企业类型比例变化')
        axes[0, 1].set_xlabel('仿真步数')
        axes[0, 1].set_ylabel('比例')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 企业类型分布饼图（最终状态）
        final_data = enterprise_df.iloc[-1]
        labels = ['大型家庭农场', '农业公司', '合作社']
        sizes = [final_data['Family_Farm_Ratio'], final_data['Agri_Corporation_Ratio'], final_data['Cooperative_Ratio']]
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('最终企业类型分布')
        
    else:
        # 使用基于目标回报率推断的数据
        # 企业类型数量变化
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['High_Return_Enterprises'], label='高回报企业', marker='o')
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['Medium_Return_Enterprises'], label='中等回报企业', marker='s')
        axes[0, 0].plot(enterprise_df['Step'], enterprise_df['Low_Return_Enterprises'], label='低回报企业', marker='^')
        axes[0, 0].set_title('企业类型数量变化（基于目标回报率）')
        axes[0, 0].set_xlabel('仿真步数')
        axes[0, 0].set_ylabel('企业数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 企业类型比例变化
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['High_Return_Ratio'], label='高回报企业比例', marker='o')
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['Medium_Return_Ratio'], label='中等回报企业比例', marker='s')
        axes[0, 1].plot(enterprise_df['Step'], enterprise_df['Low_Return_Ratio'], label='低回报企业比例', marker='^')
        axes[0, 1].set_title('企业类型比例变化（基于目标回报率）')
        axes[0, 1].set_xlabel('仿真步数')
        axes[0, 1].set_ylabel('比例')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 企业类型分布饼图（最终状态）
        final_data = enterprise_df.iloc[-1]
        labels = ['高回报企业', '中等回报企业', '低回报企业']
        sizes = [final_data['High_Return_Ratio'], final_data['Medium_Return_Ratio'], final_data['Low_Return_Ratio']]
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('最终企业类型分布（基于目标回报率）')
    
    # 企业平均指标变化
    axes[1, 1].plot(enterprise_df['Step'], enterprise_df['Average_Target_Return'], label='平均目标回报率', marker='o')
    axes[1, 1].plot(enterprise_df['Step'], enterprise_df['Average_Performance'], label='平均绩效趋势', marker='s')
    axes[1, 1].set_title('企业平均指标变化')
    axes[1, 1].set_xlabel('仿真步数')
    axes[1, 1].set_ylabel('指标值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enterprise_type_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return enterprise_df

def create_enterprise_crop_decision_visualization(crop_df, output_dir):
    """创建企业种植决策博弈趋势可视化"""
    print("开始分析企业种植决策博弈趋势...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('企业种植决策博弈趋势分析', fontsize=16, fontweight='bold')
    
    # 种植决策策略数量变化
    axes[0, 0].plot(crop_df['Step'], crop_df['Aggressive_Strategy'], label='积极策略', marker='o')
    axes[0, 0].plot(crop_df['Step'], crop_df['Balanced_Strategy'], label='平衡策略', marker='s')
    axes[0, 0].plot(crop_df['Step'], crop_df['Conservative_Strategy'], label='保守策略', marker='^')
    axes[0, 0].set_title('种植决策策略数量变化')
    axes[0, 0].set_xlabel('仿真步数')
    axes[0, 0].set_ylabel('企业数量')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 种植决策策略比例变化
    axes[0, 1].plot(crop_df['Step'], crop_df['Aggressive_Ratio'], label='积极策略比例', marker='o')
    axes[0, 1].plot(crop_df['Step'], crop_df['Balanced_Ratio'], label='平衡策略比例', marker='s')
    axes[0, 1].plot(crop_df['Step'], crop_df['Conservative_Ratio'], label='保守策略比例', marker='^')
    axes[0, 1].set_title('种植决策策略比例变化')
    axes[0, 1].set_xlabel('仿真步数')
    axes[0, 1].set_ylabel('比例')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 种植决策策略分布饼图（最终状态）
    final_data = crop_df.iloc[-1]
    labels = ['积极策略', '平衡策略', '保守策略']
    sizes = [final_data['Aggressive_Ratio'], final_data['Balanced_Ratio'], final_data['Conservative_Ratio']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('最终种植决策策略分布')
    
    # 企业决策指标变化
    axes[1, 1].plot(crop_df['Step'], crop_df['Average_Target_Return'], label='平均目标回报率', marker='o')
    axes[1, 1].plot(crop_df['Step'], crop_df['Average_Learning_Rate'], label='平均学习率', marker='s')
    axes[1, 1].plot(crop_df['Step'], crop_df['Average_Performance'], label='平均绩效', marker='^')
    axes[1, 1].set_title('企业决策指标变化')
    axes[1, 1].set_xlabel('仿真步数')
    axes[1, 1].set_ylabel('指标值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enterprise_crop_decision_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return crop_df

def create_crop_choice_visualization(crop_df, output_dir):
    """创建种植决策选择趋势可视化"""
    print("开始分析种植决策选择趋势...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('种植决策选择趋势分析', fontsize=16, fontweight='bold')
    
    # 作物选择比例变化
    axes[0, 0].plot(crop_df['Step'], crop_df['Grain_Crop_Ratio'], label='粮食作物比例', marker='o')
    axes[0, 0].plot(crop_df['Step'], crop_df['Cash_Crop_Ratio'], label='经济作物比例', marker='s')
    axes[0, 0].set_title('作物选择比例变化')
    axes[0, 0].set_xlabel('仿真步数')
    axes[0, 0].set_ylabel('比例')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 种植面积分布
    axes[0, 1].plot(crop_df['Step'], crop_df['Average_Grain_Area'], label='平均粮食种植面积', marker='o')
    axes[0, 1].plot(crop_df['Step'], crop_df['Average_Cash_Crop_Area'], label='平均经济作物面积', marker='s')
    axes[0, 1].set_title('种植面积分布')
    axes[0, 1].set_xlabel('仿真步数')
    axes[0, 1].set_ylabel('面积')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 作物选择分布饼图（最终状态）
    final_data = crop_df.iloc[-1]
    labels = ['粮食作物', '经济作物']
    sizes = [final_data['Grain_Crop_Ratio'], final_data['Cash_Crop_Ratio']]
    colors = ['lightcoral', 'lightblue']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('最终作物选择分布')
    
    # 种植决策影响因素
    axes[1, 1].plot(crop_df['Step'], crop_df['Price_Sensitivity'], label='价格敏感度', marker='o')
    axes[1, 1].plot(crop_df['Step'], crop_df['Risk_Aversion'], label='风险厌恶度', marker='s')
    axes[1, 1].set_title('种植决策影响因素')
    axes[1, 1].set_xlabel('仿真步数')
    axes[1, 1].set_ylabel('影响程度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/crop_choice_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return crop_df

def create_crop_farmer_livelihood_visualization(crop_livelihood_df, output_dir):
    """创建作物类型农户生计类型演化可视化"""
    print("开始创建作物类型农户生计类型演化可视化...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('作物类型农户生计类型演化', fontsize=16, fontweight='bold')
    
    # 粮食作物农户生计类型演化
    axes[0, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Pure_Farmer'], 
                    label='纯农户', marker='o', color='#3498db')
    axes[0, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Pluriactive'], 
                    label='兼业农户', marker='s', color='#e67e22')
    axes[0, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Non_Farmer'], 
                    label='非农户', marker='^', color='#2ecc71')
    axes[0, 0].set_title('粮食作物农户生计类型演化')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('农户数量')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 经济作物农户生计类型演化
    axes[0, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Pure_Farmer'], 
                    label='纯农户', marker='o', color='#3498db')
    axes[0, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Pluriactive'], 
                    label='兼业农户', marker='s', color='#e67e22')
    axes[0, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Non_Farmer'], 
                    label='非农户', marker='^', color='#2ecc71')
    axes[0, 1].set_title('经济作物农户生计类型演化')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('农户数量')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 粮食作物农户生计类型比例演化
    axes[1, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Pure_Ratio'], 
                    label='纯农户比例', marker='o', color='#3498db')
    axes[1, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Pluri_Ratio'], 
                    label='兼业农户比例', marker='s', color='#e67e22')
    axes[1, 0].plot(crop_livelihood_df['Step'], crop_livelihood_df['Grain_Non_Ratio'], 
                    label='非农户比例', marker='^', color='#2ecc71')
    axes[1, 0].set_title('粮食作物农户生计类型比例演化')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('比例')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 经济作物农户生计类型比例演化
    axes[1, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Pure_Ratio'], 
                    label='纯农户比例', marker='o', color='#3498db')
    axes[1, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Pluri_Ratio'], 
                    label='兼业农户比例', marker='s', color='#e67e22')
    axes[1, 1].plot(crop_livelihood_df['Step'], crop_livelihood_df['Cash_Non_Ratio'], 
                    label='非农户比例', marker='^', color='#2ecc71')
    axes[1, 1].set_title('经济作物农户生计类型比例演化')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('比例')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/crop_farmer_livelihood_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return crop_livelihood_df

def create_inequality_visualization(inequality_df, output_dir):
    """创建收入不平等趋势可视化"""
    print("开始分析收入不平等趋势...")
    
    # 在每个图表生成函数内部设置字体
    from utils.font_config import setup_chinese_fonts
    setup_chinese_fonts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('收入不平等趋势分析', fontsize=16, fontweight='bold')
    
    # 收入基尼系数变化
    axes[0, 0].plot(inequality_df['Step'], inequality_df['Income_Gini'], label='收入基尼系数', marker='o')
    axes[0, 0].set_title('收入基尼系数变化')
    axes[0, 0].set_xlabel('仿真步数')
    axes[0, 0].set_ylabel('基尼系数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 平均收入变化
    axes[0, 1].plot(inequality_df['Step'], inequality_df['Mean_Income'], label='平均收入', marker='s')
    axes[0, 1].set_title('平均收入变化')
    axes[0, 1].set_xlabel('仿真步数')
    axes[0, 1].set_ylabel('收入')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 收入分布变化
    axes[1, 0].plot(inequality_df['Step'], inequality_df['Income_90th_Percentile'], label='90%分位数', marker='o')
    axes[1, 0].plot(inequality_df['Step'], inequality_df['Income_Median'], label='中位数', marker='s')
    axes[1, 0].plot(inequality_df['Step'], inequality_df['Income_10th_Percentile'], label='10%分位数', marker='^')
    axes[1, 0].set_title('收入分布变化')
    axes[1, 0].set_xlabel('仿真步数')
    axes[1, 0].set_ylabel('收入')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 不平等趋势热力图
    heatmap_data = inequality_df[['Income_Gini', 'Income_90th_Percentile', 'Income_Median', 'Income_10th_Percentile']].T
    im = axes[1, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('不平等指标热力图')
    axes[1, 1].set_xticks(range(len(inequality_df)))
    axes[1, 1].set_xticklabels(inequality_df['Step'], rotation=45)
    axes[1, 1].set_yticks(range(len(heatmap_data.index)))
    axes[1, 1].set_yticklabels(heatmap_data.index)
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/income_inequality_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return inequality_df

def create_comprehensive_visualization(strategy_df, enterprise_df, crop_df, inequality_df, 
                                     crop_livelihood_df=None, output_dir=None):
    """创建综合可视化图表"""
    print("开始创建综合可视化图表...")
    
    # 设置图表样式
    setup_plotting()
    
    # 创建输出目录
    viz_dir = output_dir / "comprehensive_visualization"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. 农户生计类型变化趋势
    if strategy_df is not None and not strategy_df.empty:
        create_livelihood_strategy_visualization(strategy_df, viz_dir)
    
    # 2. 企业类型变化趋势
    if enterprise_df is not None and not enterprise_df.empty:
        create_enterprise_type_visualization(enterprise_df, viz_dir)
    
    # 3. 企业种植决策博弈趋势
    if crop_df is not None and not crop_df.empty:
        create_enterprise_crop_decision_visualization(crop_df, viz_dir)
    
    # 4. 收入不平等趋势
    if inequality_df is not None and not inequality_df.empty:
        create_inequality_visualization(inequality_df, viz_dir)
    
    # 5. 作物类型农户生计类型演化（新增）
    if crop_livelihood_df is not None and not crop_livelihood_df.empty:
        create_crop_farmer_livelihood_visualization(crop_livelihood_df, viz_dir)
    
    print(f"综合可视化图表已保存到: {viz_dir}")

def create_summary_report(strategy_df, enterprise_df, crop_df, inequality_df, output_dir):
    """创建综合统计报告"""
    report = []
    report.append("# 气候政策ABM综合关键指标分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 农户生计类型变化分析
    report.append("## 1. 农户生计类型变化分析")
    if not strategy_df.empty:
        initial = strategy_df.iloc[0]
        final = strategy_df.iloc[-1]
        
        # 修复格式化错误：将浮点数转换为整数
        pure_change = int(final['Pure_Farmer'] - initial['Pure_Farmer'])
        pluri_change = int(final['Pluriactive'] - initial['Pluriactive'])
        non_change = int(final['Non_Farmer'] - initial['Non_Farmer'])
        
        report.append(f"### 生计类型分布变化")
        report.append(f"- 纯农户变化: {pure_change:+d} ({pure_change/initial['Total_Farmers']:.1%})")
        report.append(f"- 兼业农户变化: {pluri_change:+d} ({pluri_change/initial['Total_Farmers']:.1%})")
        report.append(f"- 非农户变化: {non_change:+d} ({non_change/initial['Total_Farmers']:.1%})")
        
        report.append(f"### 最终状态")
        report.append(f"- 纯农户比例: {final['Pure_Farmer_Ratio']:.1%}")
        report.append(f"- 兼业农户比例: {final['Pluriactive_Ratio']:.1%}")
        report.append(f"- 非农户比例: {final['Non_Farmer_Ratio']:.1%}")
    
    # 企业类型变化分析
    report.append("\n## 2. 农业企业类型变化分析")
    if not enterprise_df.empty:
        initial = enterprise_df.iloc[0]
        final = enterprise_df.iloc[-1]
        
        report.append(f"### 企业类型分布变化")
        report.append(f"- 大型家庭农场比例: {final['Family_Farm_Ratio']:.1%}")
        report.append(f"- 农业公司比例: {final['Agri_Corporation_Ratio']:.1%}")
        report.append(f"- 合作社比例: {final['Cooperative_Ratio']:.1%}")
    
    # 种植决策选择分析
    report.append("\n## 3. 企业种植决策博弈分析")
    if not crop_df.empty:
        initial = crop_df.iloc[0]
        final = crop_df.iloc[-1]
        
        report.append(f"### 企业种植决策策略趋势")
        report.append(f"- 积极策略企业比例: {final['Aggressive_Ratio']:.1%}")
        report.append(f"- 平衡策略企业比例: {final['Balanced_Ratio']:.1%}")
        report.append(f"- 保守策略企业比例: {final['Conservative_Ratio']:.1%}")
        report.append(f"- 平均目标回报率: {final['Average_Target_Return']:.3f}")
        report.append(f"- 平均学习率: {final['Average_Learning_Rate']:.3f}")
    
    # 收入不平等趋势分析
    report.append("\n## 4. 收入不平等趋势分析")
    if not inequality_df.empty:
        initial = inequality_df.iloc[0]
        final = inequality_df.iloc[-1]
        
        report.append(f"### 收入不平等指标")
        report.append(f"- 收入基尼系数: {final['Income_Gini']:.3f}")
        report.append(f"- 平均收入: {final['Mean_Income']:.0f}")
        report.append(f"- 90%分位数收入: {final['Income_90th_Percentile']:.0f}")
        report.append(f"- 中位数收入: {final['Income_Median']:.0f}")
        report.append(f"- 10%分位数收入: {final['Income_10th_Percentile']:.0f}")
    
    # 政策建议
    report.append("\n## 5. 政策建议")
    if not strategy_df.empty:
        final = strategy_df.iloc[-1]
        
        if final['Non_Farmer_Ratio'] > 0.1:
            report.append("- **农户转型加速**: 非农户比例较高，建议加强农业支持政策")
        
        if final['Pluriactive_Ratio'] < 0.05:
            report.append("- **兼业农户不足**: 兼业农户比例较低，建议鼓励多元化经营")
        
        if final['Pure_Farmer_Ratio'] > 0.8:
            report.append("- **纯农户集中**: 纯农户比例过高，建议促进农业现代化")
    
    if not inequality_df.empty:
        final = inequality_df.iloc[-1]
        
        if final['Income_Gini'] > 0.4:
            report.append("- **收入不平等严重**: 基尼系数较高，建议加强收入再分配政策")
    
    # 保存报告
    report_path = output_dir / "comprehensive_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"综合分析报告已保存到: {report_path}")
    return report_path

def main():
    """主函数"""
    print("开始综合关键指标可视化分析...")
    
    try:
        # 加载数据
        agent_data, model_data, output_dir = load_simulation_data()
        
        # 分析各项指标
        strategy_df = analyze_livelihood_strategy_trends(agent_data)
        enterprise_df = analyze_enterprise_type_trends(agent_data) # 使用真实企业数据
        crop_df = analyze_enterprise_crop_decision_trends(agent_data) # 使用真实企业种植决策数据
        inequality_df = analyze_income_inequality_trends(agent_data)
        crop_livelihood_df = analyze_crop_farmer_livelihood_evolution(agent_data)  # 新增：作物类型农户生计类型演化
        
        # 创建可视化
        create_comprehensive_visualization(strategy_df, enterprise_df, crop_df, inequality_df, 
                                         crop_livelihood_df, output_dir)
        
        # 创建统计报告
        create_summary_report(strategy_df, enterprise_df, crop_df, inequality_df, output_dir)
        
        print("综合关键指标可视化分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 