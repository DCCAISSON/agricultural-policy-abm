# -*- coding: utf-8 -*-
"""
政策优化评估脚本
======================================

简化版的政策评估脚本，关闭SSP场景
专注于：
1. 反事实实验（Baseline对照组）
2. 补贴率单目标优化
3. 多目标优化（帕累托前沿）
4. 单目标 vs 多目标结果对比
5. 不确定性量化

核心政策参数：
- 补贴率 (0-500 元/亩)
- 托市收购价 (2.0-3.0 元/kg)
- 农业社会化服务 (开/关)
- 种粮约束比例 (0-0.6)

运行方式：
    python run_policy_optimization.py --quick        # 快速测试
    python run_policy_optimization.py --full         # 完整分析
    python run_policy_optimization.py --pareto       # 帕累托分析
"""

import os
import sys
import argparse
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入核心模块
try:
    import yaml
except ImportError:
    print("请安装PyYAML: pip install pyyaml")
    yaml = None


# ============== 数据类定义 ==============

@dataclass
class PolicyConfig:
    """政策配置"""
    name: str
    subsidy_rate: float  # 元/亩
    floor_price: float  # 托市价 元/kg
    service_enabled: bool  # 服务市场
    grain_requirement: float  # 种粮约束
    is_baseline: bool = False


@dataclass
class SimulationResult:
    """模拟结果"""
    policy_name: str
    seed: int
    # 产出指标
    grain_output: float
    grain_area: float
    cash_area: float
    # 收入指标
    avg_income: float
    pure_farmer_income: float
    pluriactive_income: float
    # 结构指标
    pure_farmer_count: int
    pluriactive_count: int
    non_farmer_count: int
    # 土地指标
    transfer_rate: float
    leased_area: float
    # 财政指标
    subsidy_cost: float
    fiscal_efficiency: float
    # 公平指标
    gini_coefficient: float


# ============== 核心函数 ==============

def load_base_config() -> Dict[str, Any]:
    """加载基础配置"""
    config_path = PROJECT_ROOT / 'config' / 'master_config.yaml'
    if config_path.exists() and yaml:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # 默认配置
    return {
        'simulation': {'steps': 30, 'seed': 42},
        'farmers': {'n_farmers': 200, 'n_enterprises': 15},
        'environment': {'grid_width': 20, 'grid_height': 20},
        'policy': {}
    }


def apply_policy_config(base_config: Dict, policy: PolicyConfig) -> Dict:
    """应用政策配置
    
    @critical 2025-12-07: 修复补贴参数传递问题
    必须将实验参数写入 policy.overrides，才能覆盖 policy_params.json 的默认值
    """
    config = copy.deepcopy(base_config)
    
    # 初始化policy部分
    if 'policy' not in config:
        config['policy'] = {}
    if 'subsidies' not in config['policy']:
        config['policy']['subsidies'] = {}
    if 'grain_reserve_bureau' not in config:
        config['grain_reserve_bureau'] = {}
    if 'agricultural_service' not in config:
        config['agricultural_service'] = {}
    if 'village_collective' not in config:
        config['village_collective'] = {}
    
    # @critical: 通过 overrides 传递实验参数，覆盖 policy_params.json 的默认值
    config['policy']['overrides'] = {
        'grain_subsidies_rate': policy.subsidy_rate / 350.0 if policy.subsidy_rate > 0 else 0,
        'base_subsidy_per_mu': policy.subsidy_rate,
    }
    
    # 补贴设置 (保留兼容性)
    config['policy']['subsidies']['grain_subsidies_rate'] = policy.subsidy_rate / 350.0
    config['policy']['subsidies']['base_subsidy_per_mu'] = policy.subsidy_rate
    
    # 托市价格 - 同时设置到 overrides 和 grain_reserve_bureau
    config['policy']['overrides']['grain_price_floor'] = policy.floor_price
    config['grain_reserve_bureau']['floor_price'] = policy.floor_price
    
    # 服务市场
    config['agricultural_service']['enabled'] = policy.service_enabled
    
    # 种粮约束
    config['village_collective']['grain_requirement'] = {
        'ratio': policy.grain_requirement,
        'enforcement_strength': 0.6 if policy.grain_requirement > 0 else 0
    }
    
    # 如果是基准情景，关闭补贴和托市政策（但保留种粮约束）
    # @critical 2025-01: 种粮约束是粮食安全底线，不随政策情景变化
    if policy.is_baseline:
        config['policy']['overrides']['grain_subsidies_rate'] = 0
        config['policy']['overrides']['base_subsidy_per_mu'] = 0
        config['policy']['overrides']['grain_price_floor'] = 0
        config['policy']['subsidies']['grain_subsidies_rate'] = 0
        config['policy']['subsidies']['base_subsidy_per_mu'] = 0
        config['grain_reserve_bureau']['floor_price'] = 0
        config['agricultural_service']['enabled'] = False
        # 保留种粮约束，不再清零
    
    return config


def run_single_simulation(config: Dict, seed: int = 42, steps: int = 30) -> SimulationResult:
    """
    运行单次模拟
    
    Args:
        config: 模型配置
        seed: 随机种子
        steps: 模拟步数
    
    Returns:
        SimulationResult
    """
    try:
        from copy import deepcopy
        from core.model_builder import ModelBuilder
        from agents.farmer_agent import LivelihoodStrategy
        
        # 准备配置
        cfg = deepcopy(config)
        if 'experiment' not in cfg:
            cfg['experiment'] = {}
        cfg['experiment']['steps'] = steps
        cfg['experiment']['seed'] = seed
        
        # 使用ModelBuilder构建模型
        np.random.seed(seed)
        builder = ModelBuilder(cfg)
        model = builder.build()
        
        for _ in range(steps):
            model.step()
        
        # 提取结果 - 使用model.farmers属性
        farmers = model.farmers if hasattr(model, 'farmers') else [
            a for a in model.schedule.agents 
            if hasattr(a, 'total_income') and not hasattr(a, 'enterprise_area')
        ]
        enterprises = model.agricultural_enterprises if hasattr(model, 'agricultural_enterprises') else [
            a for a in model.schedule.agents if hasattr(a, 'enterprise_area')
        ]
        
        # 分类农户
        pure_farmers = [f for f in farmers if getattr(f, 'livelihood_strategy', None) == LivelihoodStrategy.PURE_FARMER]
        pluriactive = [f for f in farmers if getattr(f, 'livelihood_strategy', None) == LivelihoodStrategy.PLURIACTIVE]
        non_farmers = [f for f in farmers if getattr(f, 'livelihood_strategy', None) == LivelihoodStrategy.NON_FARMER]
        
        # 收入
        all_incomes = [f.total_income for f in farmers if hasattr(f, 'total_income')]
        pure_incomes = [f.total_income for f in pure_farmers if hasattr(f, 'total_income')]
        pluri_incomes = [f.total_income for f in pluriactive if hasattr(f, 'total_income')]
        
        # 基尼系数
        gini = calculate_gini(all_incomes) if all_incomes else 0
        
        # 产出 - 使用 last_yield 和 crop_choice 计算
        grain_output = 0
        grain_area = 0
        cash_area = 0
        
        for f in farmers:
            crop_choice = getattr(f, 'last_crop_choice', None)
            crop_value = getattr(crop_choice, 'value', '') if crop_choice else ''
            operated_area = f.get_operated_land_area() if hasattr(f, 'get_operated_land_area') else 0
            last_yield = getattr(f, 'last_yield', 0)
            
            if 'Grain' in crop_value:
                grain_output += last_yield
                grain_area += operated_area
            elif 'Cash' in crop_value:
                cash_area += operated_area
        
        # 土地流转 - 使用 leased_out_patches 属性
        total_owned = 0
        total_leased_out = 0
        
        for f in farmers:
            if hasattr(f, 'land_patches'):
                total_owned += sum(p.area for p in f.land_patches if hasattr(p, 'area'))
            if hasattr(f, 'leased_out_patches'):
                total_leased_out += sum(p.area for p in f.leased_out_patches if hasattr(p, 'area'))
        
        total_area = grain_area + cash_area
        transfer_rate = total_leased_out / max(1, total_owned)
        
        # 财政 - @fix: 修正属性名称从subsidy_expenditure到total_subsidy_paid
        gov = next((a for a in model.schedule.agents if hasattr(a, 'total_subsidy_paid')), None)
        subsidy_cost = getattr(gov, 'total_subsidy_paid', 0) if gov else 0
        fiscal_efficiency = grain_output / subsidy_cost if subsidy_cost > 0 else float('inf')
        
        return SimulationResult(
            policy_name=config.get('_policy_name', 'unknown'),
            seed=seed,
            grain_output=grain_output,
            grain_area=grain_area,
            cash_area=cash_area,
            avg_income=np.mean(all_incomes) if all_incomes else 0,
            pure_farmer_income=np.mean(pure_incomes) if pure_incomes else 0,
            pluriactive_income=np.mean(pluri_incomes) if pluri_incomes else 0,
            pure_farmer_count=len(pure_farmers),
            pluriactive_count=len(pluriactive),
            non_farmer_count=len(non_farmers),
            transfer_rate=transfer_rate,
            leased_area=total_leased_out,
            subsidy_cost=subsidy_cost,
            fiscal_efficiency=fiscal_efficiency,
            gini_coefficient=gini,
        )
        
    except Exception as e:
        print(f"模拟失败: {e}")
        # 返回零结果
        return SimulationResult(
            policy_name=config.get('_policy_name', 'error'),
            seed=seed,
            grain_output=0, grain_area=0, cash_area=0,
            avg_income=0, pure_farmer_income=0, pluriactive_income=0,
            pure_farmer_count=0, pluriactive_count=0, non_farmer_count=0,
            transfer_rate=0, leased_area=0,
            subsidy_cost=0, fiscal_efficiency=0, gini_coefficient=0
        )


def calculate_gini(values: List[float]) -> float:
    """计算基尼系数"""
    if not values or len(values) < 2:
        return 0.0
    values = np.array(sorted(values))
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


# ============== 实验设计 ==============

def design_subsidy_gradient(n_levels: int = 6) -> List[PolicyConfig]:
    """
    设计补贴率梯度实验
    
    单一变量：补贴率从0到500元/亩
    其他参数固定为现实值
    """
    subsidy_levels = np.linspace(0, 500, n_levels)
    
    policies = []
    for rate in subsidy_levels:
        policy = PolicyConfig(
            name=f'subsidy_{int(rate)}',
            subsidy_rate=rate,
            floor_price=2.36,  # 固定托市价
            service_enabled=True,  # 固定开启服务市场
            grain_requirement=0.3,  # 固定种粮约束
            is_baseline=False
        )
        policies.append(policy)
    
    return policies


def design_counterfactual_experiment() -> List[PolicyConfig]:
    """
    设计反事实实验
    
    5个情景：
    1. Baseline (无政策) - 市场价格为基准，无补贴无托市
    2. 仅补贴 - 350元/亩
    3. 仅托市 - 托市价1.40元/斤（市场价+12%）
    4. 仅服务市场
    5. 完整政策组合
    
    @critical 2025-12: 种粮约束简化为土地类型约束
    - 基本农田（80%耕地）：流转后≥80%种粮（刚性）
    - 一般耕地（20%耕地）：无强制约束，靠激励引导
    - 不再区分流转渠道（村集体/直接流转）
    
    @critical 2025-12: 托市价参数校准
    - 市场价：1.29元/斤（= 2.58元/公斤，与master_config.yaml一致）
    - 托市价：1.40元/斤（市场价+9%，符合现实政策）
    - 之前的2.36元/斤是市场价的188%，严重偏离现实
    
    理论依据：
    1. OECD农业政策评估：托市价通常在市场价95-115%区间
    2. 中国最低收购价实践：小麦1.17元/斤（2020），约为市场价95%
    3. 预期效果：托市价+9%应带来5-15%的产量增加，而非200%
    """
    # 种粮约束比例（简化：仅用于全局配置，实际由土地类型决定）
    # 基本农田：80%刚性约束；一般耕地：0%约束
    GRAIN_REQUIREMENT = 0.80  # 基本农田的种粮约束
    
    # 市场基准价格 - 统一为1.29元/斤（= 2.58元/公斤，与master_config.yaml一致）
    # @fix 2025-12-08: 与master_config.yaml的floor_price: 2.58保持一致
    MARKET_PRICE = 1.29  # 元/斤（2020年粮食市场价基准，= 2.58元/公斤）
    
    # 托市价设置（市场价+9%，从1.29到1.40）
    # 具体政策干预时启用
    FLOOR_PRICE = 1.40  # 元/斤
    
    return [
        PolicyConfig(
            name='Baseline_无政策',
            subsidy_rate=0, floor_price=MARKET_PRICE,  # 保留市场价作为基准，非0！
            service_enabled=False, grain_requirement=GRAIN_REQUIREMENT,
            is_baseline=True  # is_baseline=True会在apply_policy_config中关闭补贴
        ),
        PolicyConfig(
            name='仅补贴',
            subsidy_rate=350, floor_price=MARKET_PRICE,  # 补贴但无托市
            service_enabled=False, grain_requirement=GRAIN_REQUIREMENT
        ),
        PolicyConfig(
            name='仅托市',
            subsidy_rate=0, floor_price=FLOOR_PRICE,  # 托市价1.40元（+12%）
            service_enabled=False, grain_requirement=GRAIN_REQUIREMENT
        ),
        PolicyConfig(
            name='仅服务市场',
            subsidy_rate=0, floor_price=MARKET_PRICE,
            service_enabled=True, grain_requirement=GRAIN_REQUIREMENT
        ),
        PolicyConfig(
            name='完整政策组合',
            subsidy_rate=350, floor_price=FLOOR_PRICE,  # 补贴+托市
            service_enabled=True, grain_requirement=GRAIN_REQUIREMENT
        ),
    ]


def design_factorial_experiment() -> List[PolicyConfig]:
    """
    设计全因子实验（用于帕累托分析）
    
    @critical 2025-12: 参数校准为现实合理范围
    - 补贴：0/200/400 元/亩（现实中100-500元/亩）
    - 托市价：1.25/1.35/1.50 元/斤（市场价/+8%/+20%）
    - 服务市场：开/关
    
    3因子 × 2-3水平 = 18种组合
    """
    from itertools import product
    
    # 市场基准价
    MARKET_PRICE = 1.25
    
    factors = {
        'subsidy': [0, 200, 400],  # 元/亩
        'floor_price': [MARKET_PRICE, 1.35, 1.50],  # 市场价/+8%/+20%
        'service': [False, True]
    }
    
    policies = []
    for i, (sub, price, svc) in enumerate(product(
        factors['subsidy'], factors['floor_price'], factors['service']
    )):
        policy = PolicyConfig(
            name=f'factorial_{i:02d}',
            subsidy_rate=sub,
            floor_price=price,
            service_enabled=svc,
            grain_requirement=0.80,  # 基本农田约束
            is_baseline=(sub == 0 and price == MARKET_PRICE and not svc)
        )
        policies.append(policy)
    
    return policies


# ============== 优化分析 ==============

def run_monte_carlo(policy: PolicyConfig, base_config: Dict, 
                    n_replications: int = 10, steps: int = 30) -> List[SimulationResult]:
    """运行蒙特卡洛实验"""
    config = apply_policy_config(base_config, policy)
    config['_policy_name'] = policy.name
    
    results = []
    for seed in range(n_replications):
        result = run_single_simulation(config, seed=seed, steps=steps)
        results.append(result)
    
    return results


def aggregate_results(results: List[SimulationResult]) -> Dict[str, Any]:
    """聚合多次实验结果"""
    metrics = [
        'grain_output', 'avg_income', 'pure_farmer_income',
        'pure_farmer_count', 'transfer_rate', 'subsidy_cost',
        'gini_coefficient', 'fiscal_efficiency'
    ]
    
    agg = {}
    for metric in metrics:
        values = [getattr(r, metric) for r in results]
        agg[f'{metric}_mean'] = np.mean(values)
        agg[f'{metric}_std'] = np.std(values)
        agg[f'{metric}_ci95_low'] = np.percentile(values, 2.5)
        agg[f'{metric}_ci95_high'] = np.percentile(values, 97.5)
    
    return agg


def find_pareto_frontier(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    识别帕累托前沿
    
    优化目标：
    - 最大化粮食产量
    - 最大化农民收入
    - 最小化财政支出
    - 最小化基尼系数
    """
    is_pareto = []
    
    for i, row_i in results_df.iterrows():
        dominated = False
        
        for j, row_j in results_df.iterrows():
            if i == j:
                continue
            
            # 检查 row_j 是否支配 row_i
            better_or_equal = (
                row_j['grain_output_mean'] >= row_i['grain_output_mean'] and
                row_j['avg_income_mean'] >= row_i['avg_income_mean'] and
                row_j['subsidy_cost_mean'] <= row_i['subsidy_cost_mean'] and
                row_j['gini_coefficient_mean'] <= row_i['gini_coefficient_mean']
            )
            
            strictly_better = (
                row_j['grain_output_mean'] > row_i['grain_output_mean'] or
                row_j['avg_income_mean'] > row_i['avg_income_mean'] or
                row_j['subsidy_cost_mean'] < row_i['subsidy_cost_mean'] or
                row_j['gini_coefficient_mean'] < row_i['gini_coefficient_mean']
            )
            
            if better_or_equal and strictly_better:
                dominated = True
                break
        
        is_pareto.append(not dominated)
    
    results_df['is_pareto'] = is_pareto
    return results_df


def single_objective_optimization(results_df: pd.DataFrame, 
                                  objective: str = 'grain_output_mean') -> Tuple[str, float]:
    """单目标优化：找最大值"""
    best_idx = results_df[objective].idxmax()
    best_row = results_df.loc[best_idx]
    return best_row['policy_name'], best_row[objective]


# ============== 可视化 ==============

def plot_counterfactual_results(results: Dict[str, Dict], output_path: Path):
    """绘制反事实实验结果"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('反事实实验结果 - 政策净效应分析', fontsize=14, fontweight='bold')
    
    policies = list(results.keys())
    x = np.arange(len(policies))
    
    # 获取Baseline值
    baseline = results.get('Baseline_无政策', {})
    baseline_grain = baseline.get('grain_output_mean', 0)
    baseline_income = baseline.get('avg_income_mean', 0)
    
    # 1. 粮食产量
    ax = axes[0, 0]
    values = [results[p].get('grain_output_mean', 0) for p in policies]
    errors = [results[p].get('grain_output_std', 0) for p in policies]
    colors = ['red' if v < baseline_grain else 'green' for v in values]
    ax.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=baseline_grain, color='black', linestyle='--', label='Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('粮食产量 (kg)')
    ax.set_title('粮食产量')
    ax.legend()
    
    # 2. 农民收入
    ax = axes[0, 1]
    values = [results[p].get('avg_income_mean', 0) for p in policies]
    errors = [results[p].get('avg_income_std', 0) for p in policies]
    colors = ['red' if v < baseline_income else 'green' for v in values]
    ax.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=baseline_income, color='black', linestyle='--', label='Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('平均收入 (元)')
    ax.set_title('农民收入')
    ax.legend()
    
    # 3. 财政支出
    ax = axes[0, 2]
    values = [results[p].get('subsidy_cost_mean', 0) for p in policies]
    ax.bar(x, np.array(values)/10000, color='orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('财政支出 (万元)')
    ax.set_title('财政成本')
    
    # 4. 纯农户数量
    ax = axes[1, 0]
    values = [results[p].get('pure_farmer_count_mean', 0) for p in policies]
    ax.bar(x, values, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('纯农户数量')
    ax.set_title('农户结构')
    
    # 5. 土地流转率
    ax = axes[1, 1]
    values = [results[p].get('transfer_rate_mean', 0) * 100 for p in policies]
    ax.bar(x, values, color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('土地流转率 (%)')
    ax.set_title('土地流转')
    
    # 6. 基尼系数
    ax = axes[1, 2]
    values = [results[p].get('gini_coefficient_mean', 0) for p in policies]
    colors = ['red' if v > 0.4 else 'green' for v in values]
    ax.bar(x, values, color=colors, alpha=0.7)
    ax.axhline(y=0.4, color='red', linestyle='--', label='警戒线')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('基尼系数')
    ax.set_title('收入不平等')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"反事实实验图已保存: {output_path}")


def plot_subsidy_gradient(results_df: pd.DataFrame, output_path: Path):
    """绘制补贴率梯度实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('补贴率梯度实验 - 单目标优化分析', fontsize=14, fontweight='bold')
    
    # 提取补贴率
    results_df['subsidy_rate'] = results_df['policy_name'].str.extract(r'(\d+)').astype(float)
    results_df = results_df.sort_values('subsidy_rate')
    
    x = results_df['subsidy_rate'].values
    
    # 1. 粮食产量
    ax = axes[0, 0]
    y = results_df['grain_output_mean'].values
    yerr = results_df['grain_output_std'].values
    ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('粮食产量 (kg)')
    ax.set_title('粮食产量与补贴率关系')
    ax.grid(True, alpha=0.3)
    
    # 标记最优点
    best_idx = results_df['grain_output_mean'].idxmax()
    ax.scatter(results_df.loc[best_idx, 'subsidy_rate'], 
              results_df.loc[best_idx, 'grain_output_mean'],
              s=200, c='red', marker='*', zorder=5, label='产量最优')
    ax.legend()
    
    # 2. 农民收入
    ax = axes[0, 1]
    y = results_df['avg_income_mean'].values
    yerr = results_df['avg_income_std'].values
    ax.errorbar(x, y, yerr=yerr, marker='s', capsize=5, linewidth=2, color='green')
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='green')
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('平均收入 (元)')
    ax.set_title('农民收入与补贴率关系')
    ax.grid(True, alpha=0.3)
    
    # 标记最优点
    best_idx = results_df['avg_income_mean'].idxmax()
    ax.scatter(results_df.loc[best_idx, 'subsidy_rate'], 
              results_df.loc[best_idx, 'avg_income_mean'],
              s=200, c='red', marker='*', zorder=5, label='收入最优')
    ax.legend()
    
    # 3. 财政效率
    ax = axes[1, 0]
    y = results_df['grain_output_mean'].values / (results_df['subsidy_cost_mean'].values + 1)
    ax.plot(x, y, marker='^', linewidth=2, color='orange')
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('财政效率 (kg/元)')
    ax.set_title('财政效率与补贴率关系')
    ax.grid(True, alpha=0.3)
    
    # 4. 多指标归一化对比
    ax = axes[1, 1]
    # 归一化
    grain_norm = (results_df['grain_output_mean'] - results_df['grain_output_mean'].min()) / \
                 (results_df['grain_output_mean'].max() - results_df['grain_output_mean'].min() + 1e-9)
    income_norm = (results_df['avg_income_mean'] - results_df['avg_income_mean'].min()) / \
                  (results_df['avg_income_mean'].max() - results_df['avg_income_mean'].min() + 1e-9)
    cost_norm = 1 - (results_df['subsidy_cost_mean'] - results_df['subsidy_cost_mean'].min()) / \
                    (results_df['subsidy_cost_mean'].max() - results_df['subsidy_cost_mean'].min() + 1e-9)
    
    ax.plot(x, grain_norm, marker='o', label='产量(归一化)', linewidth=2)
    ax.plot(x, income_norm, marker='s', label='收入(归一化)', linewidth=2)
    ax.plot(x, cost_norm, marker='^', label='财政效率(归一化)', linewidth=2)
    ax.set_xlabel('补贴率 (元/亩)')
    ax.set_ylabel('归一化得分')
    ax.set_title('多指标归一化对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"补贴率梯度图已保存: {output_path}")


def plot_pareto_frontier(results_df: pd.DataFrame, output_path: Path):
    """绘制帕累托前沿"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('多目标优化 - 帕累托前沿分析', fontsize=14, fontweight='bold')
    
    pareto = results_df[results_df['is_pareto'] == True]
    non_pareto = results_df[results_df['is_pareto'] == False]
    
    # 1. 产量 vs 收入
    ax = axes[0, 0]
    ax.scatter(non_pareto['grain_output_mean'], non_pareto['avg_income_mean'],
              c='gray', alpha=0.5, s=50, label='非最优解')
    ax.scatter(pareto['grain_output_mean'], pareto['avg_income_mean'],
              c='red', s=150, marker='*', label='帕累托最优')
    ax.set_xlabel('粮食产量 (kg)')
    ax.set_ylabel('农民收入 (元)')
    ax.set_title('粮食安全 vs 农民福利')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 产量 vs 财政成本
    ax = axes[0, 1]
    ax.scatter(non_pareto['grain_output_mean'], non_pareto['subsidy_cost_mean']/10000,
              c='gray', alpha=0.5, s=50, label='非最优解')
    ax.scatter(pareto['grain_output_mean'], pareto['subsidy_cost_mean']/10000,
              c='red', s=150, marker='*', label='帕累托最优')
    ax.set_xlabel('粮食产量 (kg)')
    ax.set_ylabel('财政支出 (万元)')
    ax.set_title('粮食安全 vs 财政约束')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 收入 vs 基尼系数
    ax = axes[1, 0]
    ax.scatter(non_pareto['avg_income_mean'], non_pareto['gini_coefficient_mean'],
              c='gray', alpha=0.5, s=50, label='非最优解')
    ax.scatter(pareto['avg_income_mean'], pareto['gini_coefficient_mean'],
              c='red', s=150, marker='*', label='帕累托最优')
    ax.set_xlabel('农民收入 (元)')
    ax.set_ylabel('基尼系数')
    ax.set_title('收入水平 vs 收入公平')
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='警戒线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 3D表示（用颜色代表第三维度）
    ax = axes[1, 1]
    sc = ax.scatter(results_df['grain_output_mean'], 
                   results_df['avg_income_mean'],
                   c=results_df['subsidy_cost_mean']/10000,
                   s=100, cmap='viridis', alpha=0.7)
    # 标记帕累托点
    ax.scatter(pareto['grain_output_mean'], pareto['avg_income_mean'],
              s=200, marker='*', facecolors='none', edgecolors='red', linewidth=2)
    plt.colorbar(sc, ax=ax, label='财政支出 (万元)')
    ax.set_xlabel('粮食产量 (kg)')
    ax.set_ylabel('农民收入 (元)')
    ax.set_title('三维帕累托空间 (颜色=财政成本)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"帕累托前沿图已保存: {output_path}")


def plot_single_vs_multi_comparison(single_results: Dict, pareto_df: pd.DataFrame, 
                                    output_path: Path):
    """对比单目标优化与多目标优化结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('单目标优化 vs 多目标优化对比', fontsize=14, fontweight='bold')
    
    # 1. 柱状图对比
    ax = axes[0]
    metrics = ['grain_output_mean', 'avg_income_mean', 'subsidy_cost_mean']
    metric_names = ['粮食产量', '农民收入', '财政支出']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # 单目标最优
    single_vals = [
        single_results['grain'].get('grain_output_mean', 0),
        single_results['grain'].get('avg_income_mean', 0),
        single_results['grain'].get('subsidy_cost_mean', 0) / 10000,
    ]
    
    # 帕累托最优（取均值）
    pareto_vals = [
        pareto_df['grain_output_mean'].mean(),
        pareto_df['avg_income_mean'].mean(),
        pareto_df['subsidy_cost_mean'].mean() / 10000,
    ]
    
    ax.bar(x - width/2, single_vals, width, label='单目标最优(产量)', alpha=0.8)
    ax.bar(x + width/2, pareto_vals, width, label='帕累托均值', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('数值')
    ax.set_title('关键指标对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 散点图对比
    ax = axes[1]
    ax.scatter(pareto_df['grain_output_mean'], pareto_df['avg_income_mean'],
              c='blue', s=100, alpha=0.7, label='帕累托最优集')
    ax.scatter(single_results['grain']['grain_output_mean'],
              single_results['grain']['avg_income_mean'],
              c='red', s=200, marker='*', label='产量最优')
    ax.scatter(single_results['income']['grain_output_mean'],
              single_results['income']['avg_income_mean'],
              c='green', s=200, marker='^', label='收入最优')
    ax.set_xlabel('粮食产量 (kg)')
    ax.set_ylabel('农民收入 (元)')
    ax.set_title('单目标最优点在帕累托前沿中的位置')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"单目标vs多目标对比图已保存: {output_path}")


# ============== 主函数 ==============

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='政策优化评估')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--full', action='store_true', help='完整分析模式')
    parser.add_argument('--pareto', action='store_true', help='帕累托分析模式')
    parser.add_argument('--counterfactual', action='store_true', help='反事实分析')
    parser.add_argument('--gradient', action='store_true', help='梯度实验')
    parser.add_argument('--replications', type=int, default=5, help='重复次数')
    parser.add_argument('--steps', type=int, default=20, help='模拟步数')
    args = parser.parse_args()
    
    # 默认运行所有分析
    if not any([args.counterfactual, args.gradient, args.pareto]):
        args.counterfactual = True
        args.gradient = True
        args.pareto = True
    
    # 参数设置
    n_replications = 3 if args.quick else args.replications
    steps = 10 if args.quick else args.steps
    
    print("=" * 70)
    print("【政策优化评估系统】（关闭SSP场景）")
    print("=" * 70)
    print(f"模式: {'快速测试' if args.quick else '完整分析'}")
    print(f"重复次数: {n_replications}")
    print(f"模拟步数: {steps}")
    
    # 输出目录 - 按脚本名称和时间组织
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = Path(__file__).stem  # 获取脚本名（不含扩展名）
    output_dir = PROJECT_ROOT / 'results' / f'{script_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果输出目录: {output_dir}")
    
    # 加载基础配置
    base_config = load_base_config()
    
    all_results = {}
    
    # ============ 反事实实验 ============
    if args.counterfactual:
        print("\n" + "=" * 60)
        print("【阶段1】反事实实验 - 政策净效应分析")
        print("=" * 60)
        
        cf_policies = design_counterfactual_experiment()
        cf_results = {}
        
        for policy in cf_policies:
            print(f"  运行: {policy.name}...", end=" ")
            results = run_monte_carlo(policy, base_config, n_replications, steps)
            agg = aggregate_results(results)
            cf_results[policy.name] = agg
            print(f"完成 (产量={agg['grain_output_mean']:.0f}kg)")
        
        # 可视化
        plot_counterfactual_results(
            cf_results, 
            output_dir / 'counterfactual.png'
        )
        
        all_results['counterfactual'] = cf_results
        
        # 计算政策净效应
        baseline = cf_results.get('Baseline_无政策', {})
        print("\n【政策净效应】")
        for name, res in cf_results.items():
            if name == 'Baseline_无政策':
                continue
            grain_effect = res['grain_output_mean'] - baseline.get('grain_output_mean', 0)
            income_effect = res['avg_income_mean'] - baseline.get('avg_income_mean', 0)
            print(f"  {name}:")
            print(f"    产量效应: {grain_effect:+.0f} kg ({grain_effect/baseline.get('grain_output_mean', 1)*100:+.1f}%)")
            print(f"    收入效应: {income_effect:+.0f} 元 ({income_effect/baseline.get('avg_income_mean', 1)*100:+.1f}%)")
    
    # ============ 补贴率梯度实验 ============
    if args.gradient:
        print("\n" + "=" * 60)
        print("【阶段2】补贴率梯度实验 - 单目标优化")
        print("=" * 60)
        
        gradient_policies = design_subsidy_gradient(n_levels=6)
        gradient_results = []
        
        for policy in gradient_policies:
            print(f"  运行: {policy.name}...", end=" ")
            results = run_monte_carlo(policy, base_config, n_replications, steps)
            agg = aggregate_results(results)
            agg['policy_name'] = policy.name
            gradient_results.append(agg)
            print(f"完成 (产量={agg['grain_output_mean']:.0f}kg)")
        
        gradient_df = pd.DataFrame(gradient_results)
        
        # 单目标最优
        best_grain_policy, best_grain_val = single_objective_optimization(
            gradient_df, 'grain_output_mean')
        best_income_policy, best_income_val = single_objective_optimization(
            gradient_df, 'avg_income_mean')
        
        print(f"\n【单目标最优】")
        print(f"  产量最优: {best_grain_policy} = {best_grain_val:.0f} kg")
        print(f"  收入最优: {best_income_policy} = {best_income_val:.0f} 元")
        
        # 可视化
        plot_subsidy_gradient(
            gradient_df,
            output_dir / 'subsidy_gradient.png'
        )
        
        all_results['gradient'] = gradient_df.to_dict()
        all_results['single_optimal'] = {
            'grain': gradient_df[gradient_df['policy_name'] == best_grain_policy].iloc[0].to_dict(),
            'income': gradient_df[gradient_df['policy_name'] == best_income_policy].iloc[0].to_dict(),
        }
    
    # ============ 帕累托分析 ============
    if args.pareto:
        print("\n" + "=" * 60)
        print("【阶段3】全因子实验 - 多目标帕累托优化")
        print("=" * 60)
        
        factorial_policies = design_factorial_experiment()
        factorial_results = []
        
        total = len(factorial_policies)
        for i, policy in enumerate(factorial_policies):
            print(f"  [{i+1}/{total}] {policy.name}...", end=" ")
            results = run_monte_carlo(policy, base_config, n_replications, steps)
            agg = aggregate_results(results)
            agg['policy_name'] = policy.name
            agg['subsidy_rate'] = policy.subsidy_rate
            agg['floor_price'] = policy.floor_price
            agg['service_enabled'] = policy.service_enabled
            factorial_results.append(agg)
            print(f"完成")
        
        factorial_df = pd.DataFrame(factorial_results)
        
        # 识别帕累托前沿
        factorial_df = find_pareto_frontier(factorial_df)
        pareto_count = factorial_df['is_pareto'].sum()
        print(f"\n【帕累托前沿】共 {pareto_count} 个最优解")
        
        pareto_df = factorial_df[factorial_df['is_pareto'] == True]
        for _, row in pareto_df.iterrows():
            print(f"  {row['policy_name']}: "
                  f"产量={row['grain_output_mean']:.0f}kg, "
                  f"收入={row['avg_income_mean']:.0f}元, "
                  f"成本={row['subsidy_cost_mean']/10000:.1f}万")
        
        # 可视化
        plot_pareto_frontier(
            factorial_df,
            output_dir / 'pareto_frontier.png'
        )
        
        all_results['factorial'] = factorial_df.to_dict()
        
        # 对比单目标与多目标
        if 'single_optimal' in all_results:
            plot_single_vs_multi_comparison(
                all_results['single_optimal'],
                pareto_df,
                output_dir / 'single_vs_multi.png'
            )
    
    # ============ 保存结果 ============
    print("\n" + "=" * 60)
    print("【保存结果】")
    print("=" * 60)
    
    # 转换为可序列化格式
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    # 保存JSON结果
    results_path = output_dir / 'optimization_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2, ensure_ascii=False)
    print(f"JSON结果已保存: {results_path}")
    
    # 保存运行配置信息
    config_info = {
        'script_name': script_name,
        'timestamp': timestamp,
        'mode': 'quick' if args.quick else 'full',
        'n_replications': n_replications,
        'steps': steps,
        'analyses_run': {
            'counterfactual': args.counterfactual,
            'gradient': args.gradient,
            'pareto': args.pareto
        }
    }
    config_path = output_dir / 'run_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"配置信息已保存: {config_path}")
    
    print("\n" + "=" * 70)
    print("政策优化评估完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
