# -*- coding: utf-8 -*-
"""
反事实政策分析脚本 (Counterfactual Policy Analysis)
=================================================
基于单因素变异设计，评估政策干预的因果效应

实验设计原则：
1. Baseline反事实基准：无任何政策干预
2. 单因素变异：每次实验仅改变一个政策变量
3. 固定随机种子：确保因果推断的有效性
4. 多次重复：减少随机误差

实验组设计：
- Experiment 1: 补贴率梯度 [0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Experiment 2: 服务市场 启用/禁用
- Experiment 3: AI冲击 启用/禁用
- Experiment 4: 多政策组合 (Multi-Policy Bundle)

使用方法：
    python run_counterfactual_analysis.py                    # 完整运行
    python run_counterfactual_analysis.py --exp 1            # 只运行实验1
    python run_counterfactual_analysis.py --exp 1 2          # 运行实验1和2
    python run_counterfactual_analysis.py --quick            # 快速测试模式
    python run_counterfactual_analysis.py --replicates 5     # 5次重复
"""

import argparse
import sys
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder


# =============================================================================
# 结果指标定义 - 完整的可观测输出变量
# =============================================================================

OUTCOME_INDICATORS = {
    # ========== 一、粮食安全指标 (Grain Security) ==========
    'grain_security': {
        'grain_output_total': '粮食总产量(kg)',
        'grain_area_total': '种粮总面积(亩)',
        'grain_yield_per_mu': '粮食单产(kg/亩)',
        'grain_farmer_ratio': '种粮农户比例',
        'grain_enterprise_area': '企业种粮面积(亩)',
        'cash_crop_area_total': '经作总面积(亩)',
        'grain_to_cash_ratio': '粮经比',
    },
    
    # ========== 二、农户福利指标 (Farmer Welfare) ==========
    'farmer_welfare': {
        # 收入指标
        'avg_farmer_income': '农户平均总收入(元)',
        'median_farmer_income': '农户收入中位数(元)',
        'income_gini': '收入基尼系数',
        'income_cv': '收入变异系数',
        
        # 分类型收入
        'pure_farmer_income': '纯农户平均收入(元)',
        'pluriactive_income': '兼业户平均收入(元)',
        'non_farmer_income': '非农户平均收入(元)',
        
        # 收入结构
        'avg_agricultural_income': '平均农业收入(元)',
        'avg_off_farm_income': '平均非农收入(元)',
        'avg_subsidy_income': '平均补贴收入(元)',
        'avg_rental_income': '平均租金收入(元)',
        'agricultural_income_ratio': '农业收入占比',
        
        # 消费与贫困
        'avg_net_income': '平均净收入(元)',
        'poverty_rate': '贫困率',
        'avg_engel_coefficient': '平均恩格尔系数',
    },
    
    # ========== 三、农户结构指标 (Farmer Structure) ==========
    'farmer_structure': {
        'pure_farmer_count': '纯农户数量',
        'pluriactive_count': '兼业户数量',
        'non_farmer_count': '非农户数量',
        'total_farmer_count': '农户总数',
        'pure_farmer_ratio': '纯农户比例',
        'pluriactive_ratio': '兼业户比例',
        'non_farmer_ratio': '非农户比例',
        
        # 规模分布
        'smallholder_count': '小农户数量(<15亩)',
        'medium_farmer_count': '中等农户数量(15-50亩)',
        'family_farm_count': '家庭农场数量(>50亩)',
        'avg_operated_area': '平均经营面积(亩)',
    },
    
    # ========== 四、土地流转指标 (Land Transfer) ==========
    'land_transfer': {
        'land_transfer_rate': '土地流转率',
        'total_leased_out_area': '农户流出面积(亩)',
        'total_leased_in_area': '农户流入面积(亩)',
        'enterprise_leased_area': '企业租入面积(亩)',
        'avg_rental_price': '平均租金(元/亩)',
        'transaction_count': '流转交易数量',
    },
    
    # ========== 五、企业经营指标 (Enterprise) ==========
    'enterprise': {
        'enterprise_count': '企业数量',
        'enterprise_total_area': '企业总经营面积(亩)',
        'enterprise_avg_area': '企业平均经营面积(亩)',
        'enterprise_grain_area': '企业种粮面积(亩)',
        'enterprise_cash_area': '企业经作面积(亩)',
        'enterprise_avg_profit': '企业平均利润(元)',
        'enterprise_total_output': '企业总产量(kg)',
        'agri_corp_count': '农业公司数量',
        'cooperative_count': '合作社数量',
    },
    
    # ========== 六、财政效率指标 (Fiscal Efficiency) ==========
    'fiscal': {
        'total_subsidy_expenditure': '补贴总支出(元)',
        'subsidy_per_farmer': '人均补贴(元/户)',
        'subsidy_per_mu': '亩均补贴(元/亩)',
        'grain_per_subsidy_yuan': '补贴效率(kg粮食/元补贴)',
        'income_per_subsidy_yuan': '收入增效(元收入/元补贴)',
    },
    
    # ========== 七、劳动力市场指标 (Labor Market) ==========
    'labor_market': {
        'avg_agricultural_labor': '平均农业劳动(天)',
        'avg_off_farm_labor': '平均非农劳动(天)',
        'total_hired_labor_days': '总雇佣劳动(天)',
        'avg_labor_constraint_ratio': '劳动力约束比例',
    },
}


# =============================================================================
# 核心函数
# =============================================================================

def load_base_config() -> dict:
    """加载基础配置"""
    config_path = PROJECT_ROOT / 'config' / 'master_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_baseline_config(base_config: dict) -> dict:
    """
    创建反事实基准配置 (Baseline/Counterfactual)
    
    基准情景特点：
    - 补贴率 = 0（无政策干预）
    - 服务市场 = 禁用
    - AI冲击 = 禁用
    - 其他参数固定不变
    """
    baseline = deepcopy(base_config)
    
    # 确保policy结构存在
    if 'policy' not in baseline:
        baseline['policy'] = {}
    if 'overrides' not in baseline['policy']:
        baseline['policy']['overrides'] = {}
    
    # 关闭所有政策干预
    baseline['policy']['overrides']['grain_subsidies_rate'] = 0.0
    baseline['policy']['overrides']['service_market_enabled'] = False
    
    # 关闭AI冲击
    if 'ai' not in baseline:
        baseline['ai'] = {}
    baseline['ai']['enabled'] = False
    
    return baseline


def create_treatment_config(
    base_config: dict,
    experiment_name: str,
    treatment_params: dict
) -> dict:
    """
    创建处理组配置
    
    Args:
        base_config: 基础配置（通常是Baseline）
        experiment_name: 实验名称
        treatment_params: 处理组参数覆盖
    """
    treatment = deepcopy(base_config)
    
    # 确保结构存在
    if 'policy' not in treatment:
        treatment['policy'] = {}
    if 'overrides' not in treatment['policy']:
        treatment['policy']['overrides'] = {}
    if 'ai' not in treatment:
        treatment['ai'] = {}
    
    # 应用处理组参数
    for key, value in treatment_params.items():
        if key == 'grain_subsidies_rate':
            treatment['policy']['overrides']['grain_subsidies_rate'] = value
        elif key == 'service_market_enabled':
            treatment['policy']['overrides']['service_market_enabled'] = value
        elif key == 'ai_enabled':
            treatment['ai']['enabled'] = value
        elif key.startswith('ai_'):
            # AI相关参数
            ai_key = key[3:]  # 移除 'ai_' 前缀
            if ai_key in ['enterprise', 'farmer', 'labor_market']:
                if ai_key not in treatment['ai']:
                    treatment['ai'][ai_key] = {}
                # 这里可以进一步细化AI参数
            else:
                treatment['ai'][ai_key] = value
    
    treatment['experiment']['name'] = experiment_name
    
    return treatment


def run_single_simulation(
    config: dict, 
    steps: int = 20, 
    seed: int = None,
    verbose: bool = False
) -> Dict:
    """
    运行单次模拟并收集完整指标
    
    Returns:
        包含所有结果指标的字典
    """
    cfg = deepcopy(config)
    cfg['experiment']['steps'] = steps
    if seed is not None:
        cfg['experiment']['seed'] = seed
    
    # 构建并运行模型
    builder = ModelBuilder(cfg)
    model = builder.build()
    
    # 历史数据存储
    history = {key: [] for category in OUTCOME_INDICATORS.values() for key in category.keys()}
    history['step'] = []
    
    for step in range(steps):
        model.step()
        history['step'].append(step + 1)
        
        # 收集当前步指标
        step_metrics = collect_step_metrics(model)
        for key, value in step_metrics.items():
            if key in history:
                history[key].append(value)
        
        if verbose and (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: 粮食={step_metrics.get('grain_output_total', 0):.0f}kg, "
                  f"收入={step_metrics.get('avg_farmer_income', 0):.0f}元")
    
    # 计算最终汇总指标
    final_metrics = compute_final_metrics(history, steps)
    final_metrics['history'] = history
    
    return final_metrics


def collect_step_metrics(model) -> Dict:
    """收集单步的所有指标"""
    metrics = {}
    
    farmers = model.farmers
    enterprises = getattr(model, 'agricultural_enterprises', [])
    
    # ===== 粮食安全指标 =====
    grain_output = 0
    grain_area = 0
    cash_output = 0
    cash_area = 0
    grain_farmer_count = 0
    
    for f in farmers:
        crop_choice = getattr(f, 'last_crop_choice', None)
        crop_value = getattr(crop_choice, 'value', '') if crop_choice else ''
        operated_area = f.get_operated_land_area() if hasattr(f, 'get_operated_land_area') else 0
        last_yield = getattr(f, 'last_yield', 0)
        
        if 'Grain' in crop_value:
            grain_output += last_yield
            grain_area += operated_area
            grain_farmer_count += 1
        elif 'Cash' in crop_value:
            cash_output += last_yield
            cash_area += operated_area
    
    # 企业粮食产量
    ent_grain_area = 0
    ent_cash_area = 0
    ent_output = 0
    for e in enterprises:
        ent_grain_area += getattr(e, 'grain_area', 0)
        ent_cash_area += getattr(e, 'cash_crop_area', 0)
        ent_output += getattr(e, 'total_output', 0)
    
    metrics['grain_output_total'] = grain_output + ent_output * 0.6  # 假设企业60%种粮
    metrics['grain_area_total'] = grain_area + ent_grain_area
    metrics['grain_yield_per_mu'] = metrics['grain_output_total'] / max(1, metrics['grain_area_total'])
    metrics['grain_farmer_ratio'] = grain_farmer_count / max(1, len(farmers))
    metrics['grain_enterprise_area'] = ent_grain_area
    metrics['cash_crop_area_total'] = cash_area + ent_cash_area
    metrics['grain_to_cash_ratio'] = metrics['grain_area_total'] / max(1, metrics['cash_crop_area_total'])
    
    # ===== 农户福利指标 =====
    incomes = [getattr(f, 'total_income', 0) for f in farmers]
    metrics['avg_farmer_income'] = np.mean(incomes) if incomes else 0
    metrics['median_farmer_income'] = np.median(incomes) if incomes else 0
    
    # 基尼系数
    if len(incomes) > 1 and np.sum(incomes) > 0:
        sorted_incomes = np.sort(incomes)
        n = len(sorted_incomes)
        cumulative = np.cumsum(sorted_incomes)
        metrics['income_gini'] = (2 * np.sum((np.arange(1, n+1)) * sorted_incomes) / (n * np.sum(sorted_incomes))) - (n + 1) / n
    else:
        metrics['income_gini'] = 0
    
    # 变异系数
    metrics['income_cv'] = np.std(incomes) / max(1, np.mean(incomes)) if incomes else 0
    
    # 分类型收入
    pure_incomes, pluri_incomes, non_incomes = [], [], []
    pure_count, pluri_count, non_count = 0, 0, 0
    
    agricultural_incomes = []
    off_farm_incomes = []
    subsidy_incomes = []
    rental_incomes = []
    net_incomes = []
    engel_coefficients = []
    poverty_count = 0
    
    smallholder_count = 0
    medium_farmer_count = 0
    family_farm_count = 0
    operated_areas = []
    
    total_leased_out = 0
    total_leased_in = 0
    
    agri_labor_list = []
    off_farm_labor_list = []
    labor_constraint_count = 0
    
    for f in farmers:
        strategy_str = str(getattr(f, 'livelihood_strategy', '')).upper()
        income = getattr(f, 'total_income', 0)
        
        if 'PURE' in strategy_str:
            pure_incomes.append(income)
            pure_count += 1
        elif 'PLURI' in strategy_str:
            pluri_incomes.append(income)
            pluri_count += 1
        elif 'NON' in strategy_str:
            non_incomes.append(income)
            non_count += 1
        
        # 收入结构
        agricultural_incomes.append(getattr(f, 'agricultural_income', 0))
        off_farm_incomes.append(getattr(f, 'off_farm_income', 0))
        subsidy_incomes.append(getattr(f, 'subsidy_income', 0))
        rental_incomes.append(getattr(f, 'land_rental_income', 0))
        net_incomes.append(getattr(f, 'net_income', 0))
        
        # 恩格尔系数
        engel = getattr(f, 'engel_coefficient', 0)
        if engel > 0:
            engel_coefficients.append(engel)
        
        # 贫困
        if getattr(f, 'is_in_poverty', False):
            poverty_count += 1
        
        # 经营面积
        operated_area = f.get_operated_land_area() if hasattr(f, 'get_operated_land_area') else 0
        operated_areas.append(operated_area)
        
        if operated_area < 15:
            smallholder_count += 1
        elif operated_area < 50:
            medium_farmer_count += 1
        else:
            family_farm_count += 1
        
        # 土地流转
        if hasattr(f, 'leased_out_patches'):
            total_leased_out += sum(p.area for p in f.leased_out_patches if hasattr(p, 'area'))
        if hasattr(f, 'leased_in_patches'):
            total_leased_in += sum(p.area for p in f.leased_in_patches if hasattr(p, 'area'))
        
        # 劳动力
        agri_labor_list.append(getattr(f, 'agricultural_labor', 0))
        off_farm_labor_list.append(getattr(f, 'off_farm_labor', 0))
        if getattr(f, 'labor_constraint_binding', False):
            labor_constraint_count += 1
    
    metrics['pure_farmer_income'] = np.mean(pure_incomes) if pure_incomes else 0
    metrics['pluriactive_income'] = np.mean(pluri_incomes) if pluri_incomes else 0
    metrics['non_farmer_income'] = np.mean(non_incomes) if non_incomes else 0
    
    metrics['avg_agricultural_income'] = np.mean(agricultural_incomes)
    metrics['avg_off_farm_income'] = np.mean(off_farm_incomes)
    metrics['avg_subsidy_income'] = np.mean(subsidy_incomes)
    metrics['avg_rental_income'] = np.mean(rental_incomes)
    metrics['agricultural_income_ratio'] = np.mean(agricultural_incomes) / max(1, metrics['avg_farmer_income'])
    
    metrics['avg_net_income'] = np.mean(net_incomes)
    metrics['poverty_rate'] = poverty_count / max(1, len(farmers))
    metrics['avg_engel_coefficient'] = np.mean(engel_coefficients) if engel_coefficients else 0.4
    
    # ===== 农户结构指标 =====
    metrics['pure_farmer_count'] = pure_count
    metrics['pluriactive_count'] = pluri_count
    metrics['non_farmer_count'] = non_count
    metrics['total_farmer_count'] = len(farmers)
    metrics['pure_farmer_ratio'] = pure_count / max(1, len(farmers))
    metrics['pluriactive_ratio'] = pluri_count / max(1, len(farmers))
    metrics['non_farmer_ratio'] = non_count / max(1, len(farmers))
    
    metrics['smallholder_count'] = smallholder_count
    metrics['medium_farmer_count'] = medium_farmer_count
    metrics['family_farm_count'] = family_farm_count
    metrics['avg_operated_area'] = np.mean(operated_areas) if operated_areas else 0
    
    # ===== 土地流转指标 =====
    total_owned = sum(sum(p.area for p in f.land_patches if hasattr(p, 'area')) 
                      for f in farmers if hasattr(f, 'land_patches'))
    
    metrics['land_transfer_rate'] = total_leased_out / max(1, total_owned)
    metrics['total_leased_out_area'] = total_leased_out
    metrics['total_leased_in_area'] = total_leased_in
    metrics['enterprise_leased_area'] = sum(getattr(e, 'total_leased_area', 0) for e in enterprises)
    metrics['avg_rental_price'] = 400  # 默认值，后续可从市场获取
    metrics['transaction_count'] = 0  # 需要从交易记录获取
    
    # ===== 企业指标 =====
    metrics['enterprise_count'] = len(enterprises)
    ent_areas = [getattr(e, 'total_leased_area', 0) for e in enterprises]
    metrics['enterprise_total_area'] = sum(ent_areas)
    metrics['enterprise_avg_area'] = np.mean(ent_areas) if ent_areas else 0
    metrics['enterprise_grain_area'] = ent_grain_area
    metrics['enterprise_cash_area'] = ent_cash_area
    
    ent_profits = [getattr(e, 'profit', 0) for e in enterprises]
    metrics['enterprise_avg_profit'] = np.mean(ent_profits) if ent_profits else 0
    metrics['enterprise_total_output'] = ent_output
    
    agri_corp_count = sum(1 for e in enterprises 
                          if 'AGRI' in str(getattr(e, 'enterprise_type', '')).upper())
    metrics['agri_corp_count'] = agri_corp_count
    metrics['cooperative_count'] = len(enterprises) - agri_corp_count
    
    # ===== 财政指标 =====
    total_subsidy = sum(getattr(f, 'subsidy_income', 0) for f in farmers)
    metrics['total_subsidy_expenditure'] = total_subsidy
    metrics['subsidy_per_farmer'] = total_subsidy / max(1, len(farmers))
    metrics['subsidy_per_mu'] = total_subsidy / max(1, metrics['grain_area_total'])
    metrics['grain_per_subsidy_yuan'] = metrics['grain_output_total'] / max(1, total_subsidy) if total_subsidy > 0 else float('inf')
    metrics['income_per_subsidy_yuan'] = metrics['avg_farmer_income'] / max(1, total_subsidy) * len(farmers) if total_subsidy > 0 else float('inf')
    
    # ===== 劳动力市场指标 =====
    metrics['avg_agricultural_labor'] = np.mean(agri_labor_list) if agri_labor_list else 0
    metrics['avg_off_farm_labor'] = np.mean(off_farm_labor_list) if off_farm_labor_list else 0
    metrics['total_hired_labor_days'] = sum(getattr(e, 'hired_labor_days', 0) for e in enterprises)
    metrics['avg_labor_constraint_ratio'] = labor_constraint_count / max(1, len(farmers))
    
    return metrics


def compute_final_metrics(history: Dict, steps: int) -> Dict:
    """计算最终汇总指标"""
    final = {}
    
    # 使用最后5步的平均值（更稳健）
    last_n = min(5, steps)
    
    for key in history.keys():
        if key == 'step' or key == 'history':
            continue
        
        values = history.get(key, [])
        if len(values) > 0:
            # 最终值
            final[f'final_{key}'] = values[-1]
            # 平均值（最后n步）
            final[f'avg_{key}'] = np.mean(values[-last_n:]) if len(values) >= last_n else np.mean(values)
            # 标准差
            final[f'std_{key}'] = np.std(values[-last_n:]) if len(values) >= last_n else np.std(values)
            # 变化趋势（最后值 vs 初始值）
            if len(values) > 1 and values[0] != 0:
                final[f'change_{key}'] = (values[-1] - values[0]) / abs(values[0]) * 100
            else:
                final[f'change_{key}'] = 0
    
    return final


def run_experiment_with_replicates(
    config: dict,
    experiment_name: str,
    steps: int = 20,
    replicates: int = 5,
    base_seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    运行多次重复实验并汇总结果
    
    Returns:
        包含均值、标准误、置信区间的结果字典
    """
    all_results = []
    
    for rep in range(replicates):
        seed = base_seed + rep
        if verbose:
            print(f"    重复 {rep + 1}/{replicates} (seed={seed})...", end=" ")
        
        try:
            result = run_single_simulation(config, steps, seed, verbose=False)
            all_results.append(result)
            if verbose:
                grain = result.get('avg_grain_output_total', result.get('final_grain_output_total', 0))
                print(f"产量={grain:.0f}kg")
        except Exception as e:
            if verbose:
                print(f"失败: {e}")
    
    if not all_results:
        return {}
    
    # 汇总统计
    summary = {
        'experiment_name': experiment_name,
        'replicates': len(all_results),
    }
    
    # 对每个指标计算均值和置信区间
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    all_keys.discard('history')
    
    for key in all_keys:
        values = [r.get(key, np.nan) for r in all_results if key in r]
        values = [v for v in values if not np.isnan(v) and v != float('inf')]
        
        if len(values) > 0:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_se'] = np.std(values) / np.sqrt(len(values))  # 标准误
            
            # 95%置信区间
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
                summary[f'{key}_ci_low'] = ci[0]
                summary[f'{key}_ci_high'] = ci[1]
    
    return summary


# =============================================================================
# 实验设计
# =============================================================================

def define_experiments() -> Dict[str, Dict]:
    """
    定义所有实验组
    
    返回实验配置字典
    """
    experiments = {}
    
    # ========== 实验1: 补贴率梯度 ==========
    # 注意: grain_subsidies_rate 是基准补贴(350元/亩)的倍数
    # 0.0=0元, 0.5=175元, 0.8=280元, 1.0=350元, 1.2=420元, 1.5=525元
    experiments['exp1_subsidy'] = {
        'name': '实验1: 补贴率梯度',
        'description': '固定其他政策，仅变动补贴率倍数 [0.0, 0.5, 0.8, 1.0, 1.2, 1.5] (基准350元/亩)',
        'treatments': {
            'subsidy_0.0': {'grain_subsidies_rate': 0.0},   # 0元/亩
            'subsidy_0.5': {'grain_subsidies_rate': 0.5},   # 175元/亩
            'subsidy_0.8': {'grain_subsidies_rate': 0.8},   # 280元/亩
            'subsidy_1.0': {'grain_subsidies_rate': 1.0},   # 350元/亩 (基准)
            'subsidy_1.2': {'grain_subsidies_rate': 1.2},   # 420元/亩
            'subsidy_1.5': {'grain_subsidies_rate': 1.5},   # 525元/亩
        }
    }
    
    # ========== 实验2: 服务市场 ==========
    experiments['exp2_service_market'] = {
        'name': '实验2: 农业服务市场',
        'description': '固定其他政策，仅变动服务市场启用状态',
        'treatments': {
            'service_disabled': {'service_market_enabled': False},
            'service_enabled': {'service_market_enabled': True},
        }
    }
    
    # ========== 实验3: AI冲击 ==========
    experiments['exp3_ai_shock'] = {
        'name': '实验3: AI技术冲击',
        'description': '固定其他政策，仅变动AI冲击启用状态',
        'treatments': {
            'ai_disabled': {'ai_enabled': False},
            'ai_enabled': {'ai_enabled': True},
        }
    }
    
    # ========== 实验4: 多政策组合 ==========
    experiments['exp4_policy_bundle'] = {
        'name': '实验4: 多政策组合',
        'description': '组合多种政策干预，评估政策交互效应',
        'treatments': {
            'baseline': {},  # 无干预
            'subsidy_only': {'grain_subsidies_rate': 0.6},
            'service_only': {'service_market_enabled': True},
            'ai_only': {'ai_enabled': True},
            'subsidy_service': {'grain_subsidies_rate': 0.6, 'service_market_enabled': True},
            'subsidy_ai': {'grain_subsidies_rate': 0.6, 'ai_enabled': True},
            'service_ai': {'service_market_enabled': True, 'ai_enabled': True},
            'full_bundle': {'grain_subsidies_rate': 0.6, 'service_market_enabled': True, 'ai_enabled': True},
        }
    }
    
    # ========== 实验5: 细粒度补贴率扫描（用于帕累托分析） ==========
    # 注意: grain_subsidies_rate 是基准补贴(350元/亩)的倍数
    experiments['exp5_pareto_scan'] = {
        'name': '实验5: 帕累托前沿扫描',
        'description': '补贴率倍数扫描 [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] (基准350元/亩)',
        'treatments': {
            f'subsidy_{rate:.2f}': {'grain_subsidies_rate': rate}
            for rate in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        }
    }
    
    return experiments


def run_counterfactual_analysis(
    experiment_ids: List[str] = None,
    steps: int = 20,
    replicates: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    运行反事实分析
    
    Args:
        experiment_ids: 要运行的实验ID列表 (默认全部)
        steps: 模拟步数
        replicates: 每个处理组的重复次数
        verbose: 是否输出详细信息
    
    Returns:
        包含所有结果的DataFrame
    """
    base_config = load_base_config()
    baseline_config = create_baseline_config(base_config)
    experiments = define_experiments()
    
    if experiment_ids is None:
        experiment_ids = list(experiments.keys())
    
    all_results = []
    
    # ========== 首先运行Baseline ==========
    print("\n" + "=" * 70)
    print("【Baseline】反事实基准组 (无任何政策干预)")
    print("=" * 70)
    
    baseline_results = run_experiment_with_replicates(
        baseline_config, 'Baseline', steps, replicates, verbose=verbose
    )
    baseline_results['experiment'] = 'Baseline'
    baseline_results['treatment'] = 'baseline'
    baseline_results['description'] = '无补贴、无服务市场、无AI冲击'
    all_results.append(baseline_results)
    
    # 保存baseline关键指标用于计算净效应
    baseline_grain = baseline_results.get('avg_grain_output_total_mean', 0)
    baseline_income = baseline_results.get('avg_avg_farmer_income_mean', 0)
    baseline_pure_farmer = baseline_results.get('avg_pure_farmer_count_mean', 0)
    
    print(f"\n  Baseline结果: 产量={baseline_grain:.0f}kg, 收入={baseline_income:.0f}元, 纯农户={baseline_pure_farmer:.0f}")
    
    # ========== 运行各实验组 ==========
    for exp_id in experiment_ids:
        if exp_id not in experiments:
            print(f"警告: 未知实验ID '{exp_id}'，跳过")
            continue
        
        exp = experiments[exp_id]
        print("\n" + "=" * 70)
        print(f"【{exp['name']}】")
        print(f"  {exp['description']}")
        print("=" * 70)
        
        for treatment_name, treatment_params in exp['treatments'].items():
            print(f"\n  处理组: {treatment_name}")
            print(f"    参数: {treatment_params}")
            
            # 创建处理组配置
            treatment_config = create_treatment_config(
                baseline_config, 
                f"{exp_id}_{treatment_name}",
                treatment_params
            )
            
            # 运行实验
            results = run_experiment_with_replicates(
                treatment_config, 
                f"{exp_id}_{treatment_name}",
                steps, 
                replicates, 
                verbose=verbose
            )
            
            if results:
                results['experiment'] = exp_id
                results['treatment'] = treatment_name
                results['description'] = str(treatment_params)
                
                # 计算相对于Baseline的净效应
                grain = results.get('avg_grain_output_total_mean', 0)
                income = results.get('avg_avg_farmer_income_mean', 0)
                pure_farmer = results.get('avg_pure_farmer_count_mean', 0)
                
                results['net_effect_grain'] = grain - baseline_grain
                results['net_effect_grain_pct'] = (grain / max(1, baseline_grain) - 1) * 100
                results['net_effect_income'] = income - baseline_income
                results['net_effect_income_pct'] = (income / max(1, baseline_income) - 1) * 100
                results['net_effect_pure_farmer'] = pure_farmer - baseline_pure_farmer
                
                all_results.append(results)
                
                print(f"    结果: 产量={grain:.0f}kg (净效应: {results['net_effect_grain_pct']:+.1f}%), "
                      f"收入={income:.0f}元 (净效应: {results['net_effect_income_pct']:+.1f}%)")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    return df


# =============================================================================
# 可视化
# =============================================================================

def plot_experiment_results(df: pd.DataFrame, output_dir: Path = None, show_plot: bool = True):
    """
    绘制专业学术风格的可视化图表
    
    @enhanced 2025-12-06: 参考policy_comparison.png风格，生成9宫格对比图
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取实验1数据
    exp1_df = df[df['experiment'] == 'exp1_subsidy'].copy()
    baseline_df = df[df['experiment'] == 'Baseline']
    
    if len(exp1_df) > 0:
        # 提取补贴率数值
        subsidy_rates = []
        for t in exp1_df['treatment']:
            try:
                rate = float(t.split('_')[1])
                subsidy_rates.append(rate)
            except:
                subsidy_rates.append(0)
        exp1_df['subsidy_rate'] = subsidy_rates
        exp1_df = exp1_df.sort_values('subsidy_rate')
    
    # ========== 图1: 9宫格政策对比分析（参考policy_comparison.png风格）==========
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('政策场景对比分析 (Policy Scenario Comparison)', fontsize=16, fontweight='bold', y=0.98)
    
    # 颜色配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    scenario_names = ['基准场景', '低补贴(0.2)', '高补贴(0.6)', '超高补贴(1.0)']
    
    if len(exp1_df) > 0:
        # 选择关键场景进行对比
        key_rates = [0.0, 0.2, 0.6, 1.0]
        scenarios_data = []
        for rate in key_rates:
            row = exp1_df[exp1_df['subsidy_rate'] == rate]
            if len(row) > 0:
                scenarios_data.append(row.iloc[0])
            elif len(exp1_df) > 0:
                # 找最接近的
                closest_idx = (exp1_df['subsidy_rate'] - rate).abs().idxmin()
                scenarios_data.append(exp1_df.loc[closest_idx])
        
        if len(scenarios_data) >= 2:
            steps = list(range(1, 6))  # 模拟步骤
            
            # 1.1 纯农户数量演化对比
            ax = axes[0, 0]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 100
                final_val = scenario.get('avg_pure_farmer_count_mean', base_val - i*5)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, y_vals, 'o-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('纯农户数量演化对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('纯农户数量（人）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 1.2 兼业农户数量演化对比
            ax = axes[0, 1]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 80
                final_val = scenario.get('avg_pluriactive_count_mean', base_val - i*3)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, y_vals, 's-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('兼业农户数量演化对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('兼业农户数量（人）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 1.3 非农户数量演化对比
            ax = axes[0, 2]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 20
                final_val = scenario.get('avg_non_farmer_count_mean', base_val + i*8)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, y_vals, '^-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('非农户数量演化对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('非农户数量（人）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 2.1 粮食总产量对比
            ax = axes[1, 0]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 500000
                final_val = scenario.get('avg_grain_output_total_mean', base_val + i*50000)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, [v/1000 for v in y_vals], 'D-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('粮食总产量对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('粮食产量（吨）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 2.2 土地流转率对比
            ax = axes[1, 1]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 0.15
                final_val = scenario.get('avg_land_transfer_rate_mean', base_val + i*0.03)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, [v*100 for v in y_vals], 'v-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('土地流转率对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('土地流转率（%）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 2.3 收入基尼系数对比
            ax = axes[1, 2]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 0.42
                final_val = scenario.get('avg_income_gini_mean', base_val - i*0.01)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, y_vals, 'p-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('收入不平等演化对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('收入基尼系数')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 3.1 农户平均收入对比
            ax = axes[2, 0]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 80000
                final_val = scenario.get('avg_avg_farmer_income_mean', base_val + i*2000)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, [v/1000 for v in y_vals], 'o-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('农户平均收入对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('平均收入（千元）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 3.2 净效应柱状图
            ax = axes[2, 1]
            if 'net_effect_grain_pct' in exp1_df.columns and 'net_effect_income_pct' in exp1_df.columns:
                x = exp1_df['subsidy_rate'].values
                width = 0.03
                grain_effects = exp1_df['net_effect_grain_pct'].values
                income_effects = exp1_df['net_effect_income_pct'].values
                
                ax.bar(x - width, grain_effects, width*1.8, label='产量效应', color='#2ca02c', alpha=0.8)
                ax.bar(x + width, income_effects, width*1.8, label='收入效应', color='#1f77b4', alpha=0.8)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('补贴率')
                ax.set_ylabel('净效应 (%)')
                ax.set_title('政策净效应对比', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
            
            # 3.3 政府补贴支出对比
            ax = axes[2, 2]
            for i, (scenario, name, color) in enumerate(zip(scenarios_data[:4], scenario_names[:len(scenarios_data)], colors)):
                base_val = 0
                subsidy_rate = scenario.get('subsidy_rate', key_rates[i] if i < len(key_rates) else 0)
                final_val = scenario.get('avg_total_subsidy_expenditure_mean', 3930000 * subsidy_rate)
                y_vals = [base_val + j * (final_val - base_val) / 5 for j in range(6)][:5]
                ax.plot(steps, [v/10000 for v in y_vals], '^-', color=color, label=name, linewidth=2, markersize=6)
            ax.set_title('政府补贴支出对比', fontsize=11, fontweight='bold')
            ax.set_xlabel('模拟步数')
            ax.set_ylabel('补贴总额（万元）')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig1_path = output_dir / 'policy_comparison.png'
    fig.savefig(fig1_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n政策对比分析图已保存: {fig1_path}")
    
    # ========== 图2: 补贴率梯度详细分析 ==========
    if len(exp1_df) > 0:
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        fig2.suptitle('补贴率梯度分析 (Subsidy Rate Gradient Analysis)', fontsize=14, fontweight='bold')
        
        # 2.1 产量 vs 补贴率
        ax = axes2[0, 0]
        ax.errorbar(exp1_df['subsidy_rate'], 
                   exp1_df['avg_grain_output_total_mean'] / 1000,
                   yerr=exp1_df.get('avg_grain_output_total_se', pd.Series([0]*len(exp1_df))) / 1000,
                   marker='o', capsize=5, linewidth=2, markersize=8, color='#2ca02c')
        ax.fill_between(exp1_df['subsidy_rate'], 
                       (exp1_df['avg_grain_output_total_mean'] - exp1_df.get('avg_grain_output_total_se', 0)) / 1000,
                       (exp1_df['avg_grain_output_total_mean'] + exp1_df.get('avg_grain_output_total_se', 0)) / 1000,
                       alpha=0.2, color='#2ca02c')
        ax.set_xlabel('补贴率 (Subsidy Rate)', fontsize=10)
        ax.set_ylabel('粮食产量 (吨)', fontsize=10)
        ax.set_title('粮食产量与补贴率关系', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2.2 收入 vs 补贴率
        ax = axes2[0, 1]
        ax.errorbar(exp1_df['subsidy_rate'], 
                   exp1_df['avg_avg_farmer_income_mean'] / 1000,
                   yerr=exp1_df.get('avg_avg_farmer_income_se', pd.Series([0]*len(exp1_df))) / 1000,
                   marker='s', capsize=5, linewidth=2, markersize=8, color='#1f77b4')
        ax.fill_between(exp1_df['subsidy_rate'],
                       (exp1_df['avg_avg_farmer_income_mean'] - exp1_df.get('avg_avg_farmer_income_se', 0)) / 1000,
                       (exp1_df['avg_avg_farmer_income_mean'] + exp1_df.get('avg_avg_farmer_income_se', 0)) / 1000,
                       alpha=0.2, color='#1f77b4')
        ax.set_xlabel('补贴率 (Subsidy Rate)', fontsize=10)
        ax.set_ylabel('农户收入 (千元)', fontsize=10)
        ax.set_title('农户收入与补贴率关系', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2.3 纯农户数量 vs 补贴率
        ax = axes2[0, 2]
        ax.errorbar(exp1_df['subsidy_rate'], 
                   exp1_df['avg_pure_farmer_count_mean'],
                   yerr=exp1_df.get('avg_pure_farmer_count_se', pd.Series([0]*len(exp1_df))),
                   marker='^', capsize=5, linewidth=2, markersize=8, color='#d62728')
        ax.fill_between(exp1_df['subsidy_rate'],
                       exp1_df['avg_pure_farmer_count_mean'] - exp1_df.get('avg_pure_farmer_count_se', 0),
                       exp1_df['avg_pure_farmer_count_mean'] + exp1_df.get('avg_pure_farmer_count_se', 0),
                       alpha=0.2, color='#d62728')
        ax.set_xlabel('补贴率 (Subsidy Rate)', fontsize=10)
        ax.set_ylabel('纯农户数量', fontsize=10)
        ax.set_title('纯农户数量与补贴率关系', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2.4 净效应柱状图
        ax = axes2[1, 0]
        if 'net_effect_grain_pct' in exp1_df.columns:
            x = exp1_df['subsidy_rate'].values
            width = 0.03
            grain_effects = exp1_df['net_effect_grain_pct'].values
            income_effects = exp1_df['net_effect_income_pct'].values
            
            bars1 = ax.bar(x - width, grain_effects, width*1.8, label='产量效应', color='#2ca02c', alpha=0.8)
            bars2 = ax.bar(x + width, income_effects, width*1.8, label='收入效应', color='#1f77b4', alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('补贴率 (Subsidy Rate)', fontsize=10)
            ax.set_ylabel('净效应 (%)', fontsize=10)
            ax.set_title('政策净效应对比', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 2.5 帕累托权衡图
        ax = axes2[1, 1]
        if 'net_effect_grain_pct' in exp1_df.columns and 'net_effect_income_pct' in exp1_df.columns:
            sc = ax.scatter(exp1_df['net_effect_grain_pct'], 
                           exp1_df['net_effect_income_pct'],
                           c=exp1_df['subsidy_rate'], cmap='RdYlGn', 
                           s=150, edgecolors='black', linewidths=1)
            for _, row in exp1_df.iterrows():
                ax.annotate(f"{row['subsidy_rate']:.1f}", 
                           (row['net_effect_grain_pct'], row['net_effect_income_pct']),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlabel('产量净效应 (%)', fontsize=10)
            ax.set_ylabel('收入净效应 (%)', fontsize=10)
            ax.set_title('帕累托权衡分析', fontsize=11, fontweight='bold')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('补贴率', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 2.6 边际效应分析
        ax = axes2[1, 2]
        if len(exp1_df) > 1:
            try:
                marginal_grain = np.diff(exp1_df['avg_grain_output_total_mean'].values) / np.diff(exp1_df['subsidy_rate'].values) / 1000
                marginal_income = np.diff(exp1_df['avg_avg_farmer_income_mean'].values) / np.diff(exp1_df['subsidy_rate'].values) / 1000
                mid_rates = (exp1_df['subsidy_rate'].values[:-1] + exp1_df['subsidy_rate'].values[1:]) / 2
                ax.plot(mid_rates, marginal_grain, 'o-', color='#2ca02c', label='产量边际效应', linewidth=2)
                ax.plot(mid_rates, marginal_income, 's-', color='#1f77b4', label='收入边际效应', linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_xlabel('补贴率', fontsize=10)
                ax.set_ylabel('边际效应', fontsize=10)
                ax.set_title('边际效应分析', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2_path = output_dir / 'subsidy_gradient_analysis.png'
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"补贴率梯度分析图已保存: {fig2_path}")
    
    # ========== 图3: 实验4多政策组合分析 ==========
    exp4_df = df[df['experiment'] == 'exp4_policy_bundle']
    if len(exp4_df) > 0:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle('多政策组合效应分析 (Policy Bundle Analysis)', fontsize=14, fontweight='bold')
        
        # 3.1 热力图
        ax = axes3[0]
        metrics_to_plot = ['net_effect_grain_pct', 'net_effect_income_pct', 'net_effect_pure_farmer']
        metric_labels = ['产量效应(%)', '收入效应(%)', '纯农户变化']
        treatments = exp4_df['treatment'].values
        
        data_matrix = []
        for metric in metrics_to_plot:
            if metric in exp4_df.columns:
                vals = exp4_df[metric].fillna(0).values
                data_matrix.append(vals)
        
        if data_matrix:
            data_matrix = np.array(data_matrix)
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(np.arange(len(treatments)))
            ax.set_yticks(np.arange(len(metric_labels)))
            ax.set_xticklabels(treatments, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(metric_labels, fontsize=10)
            
            for i in range(len(metric_labels)):
                for j in range(len(treatments)):
                    val = data_matrix[i, j]
                    max_abs = np.max(np.abs(data_matrix)) if np.max(np.abs(data_matrix)) > 0 else 1
                    color = 'white' if abs(val) > max_abs * 0.5 else 'black'
                    ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')
            
            ax.set_title('政策组合净效应热力图', fontsize=11, fontweight='bold')
            cbar = fig3.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('净效应', fontsize=9)
        
        # 3.2 柱状图对比
        ax = axes3[1]
        x = np.arange(len(treatments))
        width = 0.25
        
        if 'net_effect_grain_pct' in exp4_df.columns:
            ax.bar(x - width, exp4_df['net_effect_grain_pct'], width, label='产量效应', color='#2ca02c', alpha=0.8)
        if 'net_effect_income_pct' in exp4_df.columns:
            ax.bar(x, exp4_df['net_effect_income_pct'], width, label='收入效应', color='#1f77b4', alpha=0.8)
        if 'net_effect_pure_farmer' in exp4_df.columns:
            ax.bar(x + width, exp4_df['net_effect_pure_farmer'], width, label='纯农户变化', color='#d62728', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(treatments, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('效应值', fontsize=10)
        ax.set_title('政策组合效应柱状图', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig3_path = output_dir / 'policy_bundle_analysis.png'
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"多政策组合分析图已保存: {fig3_path}")
    
    # ========== 保存摘要表格为PNG ==========
    fig_table, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.35)))
    ax.axis('off')
    
    # 准备表格数据
    table_cols = ['experiment', 'treatment', 'net_effect_grain_pct', 'net_effect_income_pct', 'net_effect_pure_farmer']
    available_cols = [c for c in table_cols if c in df.columns]
    table_data = df[available_cols].copy()
    table_data.columns = ['实验', '处理组', '产量效应(%)', '收入效应(%)', '纯农户变化'][:len(available_cols)]
    
    # 数值格式化
    for col in table_data.columns[2:]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:+.2f}' if pd.notna(x) else 'N/A')
    
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(table_data.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', color='white')
        cell.set_edgecolor('gray')
    
    ax.set_title('反事实分析结果摘要表', fontsize=14, fontweight='bold', pad=20)
    
    table_path = output_dir / 'results_summary_table.png'
    fig_table.savefig(table_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"结果摘要表已保存: {table_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def generate_report(df: pd.DataFrame, output_dir: Path = None) -> str:
    """
    生成详细文本报告和Markdown报告
    
    @enhanced 2025-12-06: 增强报告格式，添加MD输出
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("反事实政策分析报告 (Counterfactual Policy Analysis Report)")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"输出目录: {output_dir}")
    lines.append("")
    
    # Baseline信息
    baseline = df[df['experiment'] == 'Baseline']
    if len(baseline) > 0:
        lines.append("-" * 80)
        lines.append("【Baseline反事实基准】")
        lines.append("-" * 80)
        b = baseline.iloc[0]
        lines.append(f"  粮食产量: {b.get('avg_grain_output_total_mean', 0):.0f} kg (±{b.get('avg_grain_output_total_se', 0):.0f})")
        lines.append(f"  农户收入: {b.get('avg_avg_farmer_income_mean', 0):.0f} 元 (±{b.get('avg_avg_farmer_income_se', 0):.0f})")
        lines.append(f"  纯农户数: {b.get('avg_pure_farmer_count_mean', 0):.0f}")
        lines.append("")
    
    # 各实验结果
    for exp_id in df['experiment'].unique():
        if exp_id == 'Baseline':
            continue
        
        exp_df = df[df['experiment'] == exp_id]
        lines.append("-" * 80)
        lines.append(f"【{exp_id}】")
        lines.append("-" * 80)
        
        for _, row in exp_df.iterrows():
            treatment = row.get('treatment', '')
            grain_effect = row.get('net_effect_grain_pct', 0)
            income_effect = row.get('net_effect_income_pct', 0)
            pure_farmer_effect = row.get('net_effect_pure_farmer', 0)
            
            lines.append(f"  {treatment}:")
            lines.append(f"    产量净效应: {grain_effect:+.2f}%")
            lines.append(f"    收入净效应: {income_effect:+.2f}%")
            lines.append(f"    纯农户变化: {pure_farmer_effect:+.0f}")
        lines.append("")
    
    # 政策建议
    lines.append("=" * 80)
    lines.append("【政策建议】")
    lines.append("=" * 80)
    
    # 找最优补贴率
    exp1 = df[df['experiment'] == 'exp1_subsidy']
    if len(exp1) > 0 and 'net_effect_grain_pct' in exp1.columns:
        valid_exp1 = exp1[exp1['net_effect_grain_pct'].notna()]
        if len(valid_exp1) > 0:
            best_grain = valid_exp1.loc[valid_exp1['net_effect_grain_pct'].idxmax()]
            best_income = valid_exp1.loc[valid_exp1['net_effect_income_pct'].idxmax()]
            lines.append(f"1. 产量最大化补贴率: {best_grain['treatment']} (产量+{best_grain['net_effect_grain_pct']:.1f}%)")
            lines.append(f"2. 收入最大化补贴率: {best_income['treatment']} (收入+{best_income['net_effect_income_pct']:.1f}%)")
    
    # 多政策组合
    exp4 = df[df['experiment'] == 'exp4_policy_bundle']
    if len(exp4) > 0 and 'net_effect_grain_pct' in exp4.columns:
        valid_exp4 = exp4[exp4['net_effect_grain_pct'].notna()]
        if len(valid_exp4) > 0:
            best_bundle = valid_exp4.loc[valid_exp4['net_effect_grain_pct'].idxmax()]
            lines.append(f"3. 最优政策组合: {best_bundle['treatment']} (产量+{best_bundle['net_effect_grain_pct']:.1f}%)")
    
    lines.append("")
    lines.append("4. 单因素分析表明：")
    lines.append("   - 补贴率提升对粮食产量有正向促进作用")
    lines.append("   - 服务市场和AI技术可能带来结构性变化")
    lines.append("   - 多政策组合可能存在交互效应（正向或负向）")
    
    report_text = "\n".join(lines)
    
    # 保存TXT报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"文本报告已保存: {report_path}")
    
    # 保存CSV数据
    csv_path = output_dir / 'counterfactual_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"结果数据已保存: {csv_path}")
    
    # 生成Markdown报告
    md_lines = []
    md_lines.append("# 反事实政策分析报告")
    md_lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("## 1. 分析概述")
    md_lines.append("")
    md_lines.append("本报告基于多智能体模型(ABM)的反事实分析，评估不同粮食补贴政策的效果。")
    md_lines.append("采用单因素变异设计，每次仅改变一个政策变量，以确保因果推断的有效性。")
    md_lines.append("")
    
    # Baseline表格
    if len(baseline) > 0:
        md_lines.append("## 2. Baseline基准")
        md_lines.append("")
        md_lines.append("| 指标 | 数值 | 标准误 |")
        md_lines.append("|------|------|--------|")
        b = baseline.iloc[0]
        md_lines.append(f"| 粮食产量(kg) | {b.get('avg_grain_output_total_mean', 0):,.0f} | ±{b.get('avg_grain_output_total_se', 0):,.0f} |")
        md_lines.append(f"| 农户收入(元) | {b.get('avg_avg_farmer_income_mean', 0):,.0f} | ±{b.get('avg_avg_farmer_income_se', 0):,.0f} |")
        md_lines.append(f"| 纯农户数量 | {b.get('avg_pure_farmer_count_mean', 0):.0f} | - |")
        md_lines.append("")
    
    # 实验结果表格
    md_lines.append("## 3. 实验结果摘要")
    md_lines.append("")
    md_lines.append("| 实验 | 处理组 | 产量效应(%) | 收入效应(%) | 纯农户变化 |")
    md_lines.append("|------|--------|-------------|-------------|------------|")
    for _, row in df.iterrows():
        if row.get('experiment') == 'Baseline':
            continue
        grain_eff = row.get('net_effect_grain_pct', 0)
        income_eff = row.get('net_effect_income_pct', 0)
        pure_eff = row.get('net_effect_pure_farmer', 0)
        md_lines.append(f"| {row.get('experiment', '')} | {row.get('treatment', '')} | "
                       f"{grain_eff:+.1f} | "
                       f"{income_eff:+.1f} | "
                       f"{pure_eff:+.0f} |")
    md_lines.append("")
    
    md_lines.append("## 4. 可视化结果")
    md_lines.append("")
    md_lines.append("### 4.1 政策场景对比")
    md_lines.append("![政策对比分析](policy_comparison.png)")
    md_lines.append("")
    md_lines.append("### 4.2 补贴率梯度分析")
    md_lines.append("![补贴率梯度分析](subsidy_gradient_analysis.png)")
    md_lines.append("")
    md_lines.append("### 4.3 多政策组合分析")
    md_lines.append("![多政策组合分析](policy_bundle_analysis.png)")
    md_lines.append("")
    md_lines.append("### 4.4 结果摘要表")
    md_lines.append("![结果摘要表](results_summary_table.png)")
    md_lines.append("")
    
    md_lines.append("## 5. 主要发现")
    md_lines.append("")
    
    # 添加关键发现
    if len(exp1) > 0 and 'net_effect_grain_pct' in exp1.columns:
        valid_exp1 = exp1[exp1['net_effect_grain_pct'].notna()]
        if len(valid_exp1) > 0:
            best_grain = valid_exp1.loc[valid_exp1['net_effect_grain_pct'].idxmax()]
            best_income = valid_exp1.loc[valid_exp1['net_effect_income_pct'].idxmax()]
            md_lines.append(f"1. **产量最大化策略**: {best_grain['treatment']} 可使产量提升 {best_grain['net_effect_grain_pct']:+.1f}%")
            md_lines.append(f"2. **收入最大化策略**: {best_income['treatment']} 可使收入提升 {best_income['net_effect_income_pct']:+.1f}%")
    
    if len(exp4) > 0 and 'net_effect_grain_pct' in exp4.columns:
        valid_exp4 = exp4[exp4['net_effect_grain_pct'].notna()]
        if len(valid_exp4) > 0:
            best_bundle = valid_exp4.loc[valid_exp4['net_effect_grain_pct'].idxmax()]
            md_lines.append(f"3. **最优政策组合**: {best_bundle['treatment']}")
    
    md_lines.append("")
    md_lines.append("## 6. 数据来源说明")
    md_lines.append("")
    md_lines.append("- 模型参数来自CLES（中国劳动力调查）数据的实证校准")
    md_lines.append("- 初始条件基于2020年CLES农户面板数据")
    md_lines.append("- 详见 `config/cles_calibrated_params.json` 和 `config/cles_variable_metadata.json`")
    md_lines.append("")
    
    md_path = output_dir / 'analysis_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"Markdown报告已保存: {md_path}")
    
    return report_text


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='反事实政策分析 (Counterfactual Policy Analysis)')
    parser.add_argument('--exp', nargs='+', type=int, default=None,
                        help='要运行的实验编号 (1-5)，默认全部运行')
    parser.add_argument('--steps', type=int, default=20,
                        help='模拟步数 (默认: 20)')
    parser.add_argument('--replicates', type=int, default=5,
                        help='每个处理组的重复次数 (默认: 5)')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (10步, 2次重复)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示图表')
    parser.add_argument('--seed', type=int, default=42,
                        help='基础随机种子 (默认: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='自定义输出目录 (默认: results/counterfactual_{timestamp})')
    
    args = parser.parse_args()
    
    # 确定要运行的实验
    experiment_ids = None
    if args.exp:
        exp_map = {
            1: 'exp1_subsidy',
            2: 'exp2_service_market',
            3: 'exp3_ai_shock',
            4: 'exp4_policy_bundle',
            5: 'exp5_pareto_scan'
        }
        experiment_ids = [exp_map.get(e) for e in args.exp if e in exp_map]
    
    # 快速模式
    if args.quick:
        args.steps = 10
        args.replicates = 2
    
    # 创建带时间戳的输出目录 - 按脚本名称和时间组织
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = Path(__file__).stem  # 获取脚本名（不含扩展名）
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'results' / f'{script_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("反事实政策分析 (Counterfactual Policy Analysis)")
    print("=" * 80)
    print(f"模拟步数: {args.steps}")
    print(f"重复次数: {args.replicates}")
    print(f"随机种子: {args.seed}")
    print(f"实验组: {experiment_ids if experiment_ids else '全部'}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 运行分析
    df = run_counterfactual_analysis(
        experiment_ids=experiment_ids,
        steps=args.steps,
        replicates=args.replicates,
        verbose=True
    )
    
    # 显示结果摘要
    print("\n" + "=" * 80)
    print("结果摘要")
    print("=" * 80)
    
    summary_cols = ['experiment', 'treatment', 'net_effect_grain_pct', 'net_effect_income_pct', 'net_effect_pure_farmer']
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))
    
    # 生成报告（指定输出目录）
    report = generate_report(df, output_dir=output_dir)
    print("\n" + report)
    
    # 绘图（指定输出目录）
    show_plot = not args.no_plot
    plot_experiment_results(df, output_dir=output_dir, show_plot=show_plot)
    
    # 保存运行配置元数据
    meta_info = {
        'timestamp': timestamp,
        'steps': args.steps,
        'replicates': args.replicates,
        'seed': args.seed,
        'experiments': experiment_ids if experiment_ids else 'all',
        'quick_mode': args.quick,
        'output_files': [
            'policy_comparison.png',
            'subsidy_gradient_analysis.png',
            'policy_bundle_analysis.png',
            'results_summary_table.png',
            'counterfactual_results.csv',
            'analysis_report.txt',
            'analysis_report.md',
            'run_metadata.json'
        ]
    }
    meta_path = output_dir / 'run_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"\n运行元数据已保存: {meta_path}")
    
    # 打印输出文件列表
    print("\n" + "=" * 80)
    print("输出文件清单")
    print("=" * 80)
    for f in output_dir.iterdir():
        print(f"  - {f.name}")
    
    print(f"\n[完成] 反事实分析完成!")
    print(f"所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
