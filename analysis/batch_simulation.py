"""
批量运行模型并收集统计指标，用于校准和敏感性分析。

Usage:
    python analysis/batch_simulation.py --runs 50 --steps 30 --output results/batch_output.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到 Python 路径，确保可以导入 core 模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.model_builder import ModelBuilder

# 尝试导入字体配置
try:
    from utils.font_config import setup_chinese_fonts
    # 捕获编码错误
    try:
        setup_chinese_fonts()
    except (UnicodeEncodeError, Exception):
        # 如果字体配置失败，使用默认设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    # 如果没有字体配置，使用默认设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


logger = logging.getLogger("batch_simulation")


def _summarize_scalar(values: List[float]) -> Dict[str, Any]:
    """统一的统计摘要：均值、样本标准差、95%置信区间、样本量。"""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci95': 0.0, 'n': 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 and std > 0 else 0.0
    return {'mean': mean, 'std': std, 'ci95': ci95, 'n': n}


def collect_model_statistics(model) -> Dict[str, Any]:
    """
    从模型运行结果中收集统计指标
    
    返回:
        statistics: 包含各种统计量的字典
    """
    stats = {}
    
    # 1. 企业类型分布
    enterprise_types = {}
    for ent in model.agricultural_enterprises:
        ent_type = ent.enterprise_type.value if hasattr(ent, 'enterprise_type') else 'UNKNOWN'
        enterprise_types[ent_type] = enterprise_types.get(ent_type, 0) + 1
    
    total_enterprises = len(model.agricultural_enterprises)
    if total_enterprises > 0:
        stats['enterprise_type_distribution'] = {
            k: v / total_enterprises for k, v in enterprise_types.items()
        }
    else:
        stats['enterprise_type_distribution'] = {}
    
    # 2. 作物结构
    grain_area = 0
    cash_crop_area = 0
    total_operated_area = 0
    
    for farmer in model.farmers:
        operated = farmer.get_operated_land_area()
        total_operated_area += operated
        if hasattr(farmer, 'planned_crop') and farmer.planned_crop:
            if farmer.planned_crop.name == 'GRAIN':
                grain_area += operated
            else:
                cash_crop_area += operated
    
    for ent in model.agricultural_enterprises:
        operated = ent.get_operated_land_area()
        total_operated_area += operated
        if hasattr(ent, 'planned_crop') and ent.planned_crop:
            if ent.planned_crop.name == 'GRAIN':
                grain_area += operated
            else:
                cash_crop_area += operated
    
    if total_operated_area > 0:
        stats['crop_structure'] = {
            'grain_crop_ratio': grain_area / total_operated_area,
            'cash_crop_ratio': cash_crop_area / total_operated_area
        }
    else:
        stats['crop_structure'] = {'grain_crop_ratio': 0.0, 'cash_crop_ratio': 0.0}
    
    # 3. 土地流转
    total_land_area = sum(patch.area for patch in model.land_patches)
    leased_area = sum(
        patch.area for patch in model.land_patches 
        if patch.leased_to is not None
    )
    
    if total_land_area > 0:
        stats['land_transfer'] = {
            'transfer_rate': leased_area / total_land_area,
            'total_leased_area': leased_area,
            'total_land_area': total_land_area
        }
        
        # 计算平均流转价格
        rental_prices = [p.rental_price for p in model.land_patches if p.leased_to is not None and p.rental_price > 0]
        if rental_prices:
            stats['land_transfer']['average_transfer_price'] = np.mean(rental_prices)
        else:
            stats['land_transfer']['average_transfer_price'] = 0.0
        stats['land_transfer']['land_concentration_top10'] = model.get_land_concentration_top10()
        stats['land_transfer']['land_fragmentation_index'] = model.get_land_fragmentation_index()
    else:
        stats['land_transfer'] = {
            'transfer_rate': 0.0,
            'average_transfer_price': 0.0
        }
    
    # 4. 收入分布
    farmer_incomes = [f.total_income for f in model.farmers if hasattr(f, 'total_income')]
    if farmer_incomes:
        stats['income_distribution'] = {
            'avg_farm_income': np.mean(farmer_incomes),
            'median_farm_income': np.median(farmer_incomes),
            'std_farm_income': np.std(farmer_incomes),
            'gini_coefficient': calculate_gini(farmer_incomes)
        }
    else:
        stats['income_distribution'] = {
            'avg_farm_income': 0.0,
            'gini_coefficient': 0.0
        }
    
    # 5. 生产率指标
    total_yield = 0
    total_operated = 0
    for farmer in model.farmers:
        if hasattr(farmer, 'last_yield') and farmer.last_yield:
            total_yield += farmer.last_yield
        total_operated += farmer.get_operated_land_area()
    
    for ent in model.agricultural_enterprises:
        if hasattr(ent, 'last_yield') and ent.last_yield:
            total_yield += ent.last_yield
        total_operated += ent.get_operated_land_area()
    
    if total_operated > 0:
        # 转换为公斤/公顷 (1斤=0.5公斤, 1亩=1/15公顷)
        yield_per_hectare = (total_yield * 0.5) / (total_operated / 15)
        stats['productivity_indicators'] = {
            'avg_yield_per_hectare': yield_per_hectare,
            'total_yield': total_yield,
            'total_operated_area': total_operated
        }
    else:
        stats['productivity_indicators'] = {
            'avg_yield_per_hectare': 0.0
        }
    
    # 6. 补贴统计
    if hasattr(model, 'government') and model.government:
        try:
            govt_stats = model.government.get_subsidy_statistics()
            stats['subsidy_statistics'] = {
                'total_subsidy_paid': govt_stats.get('total_subsidy_paid', 0),
                'subsidy_records_count': govt_stats.get('subsidy_records_count', 0)
            }
        except Exception:
            stats['subsidy_statistics'] = {}
    else:
        stats['subsidy_statistics'] = {}
    
    # 7. 市场统计
    if hasattr(model, 'environment') and model.environment:
        stats['market_statistics'] = {
            'grain_price': model.environment.market_prices.get('crop_price', 0),
            'cash_crop_price': model.environment.market_prices.get('cash_crop', 0),
            'wage_rate': model.environment.market_prices.get('wage_rate', 0)
        }
    else:
        stats['market_statistics'] = {}
    
    # 8. 农户异质性指标（新增，用于校准）
    from agents.farmer_agent import LivelihoodStrategy
    strategy_counts = {'PURE_FARMER': 0, 'PLURIACTIVE': 0, 'NON_FARMER': 0}
    agri_income_ratios = []
    farming_abilities = []
    
    for farmer in model.farmers:
        if hasattr(farmer, 'livelihood_strategy'):
            strategy = farmer.livelihood_strategy
            if strategy == LivelihoodStrategy.PURE_FARMER:
                strategy_counts['PURE_FARMER'] += 1
            elif strategy == LivelihoodStrategy.PLURIACTIVE:
                strategy_counts['PLURIACTIVE'] += 1
            elif strategy == LivelihoodStrategy.NON_FARMER:
                strategy_counts['NON_FARMER'] += 1
        
        # 计算农业收入占比
        if hasattr(farmer, 'total_income') and hasattr(farmer, 'agricultural_income'):
            if farmer.total_income > 0:
                agri_ratio = farmer.agricultural_income / farmer.total_income
                agri_income_ratios.append(agri_ratio)
        
        # 收集耕作能力
        if hasattr(farmer, 'farming_ability'):
            farming_abilities.append(farmer.farming_ability)
    
    total_farmers = len(model.farmers)
    if total_farmers > 0:
        stats['farmer_heterogeneity'] = {
            'livelihood_strategy_dist': {
                'PURE_FARMER': strategy_counts['PURE_FARMER'] / total_farmers,
                'PLURIACTIVE': strategy_counts['PLURIACTIVE'] / total_farmers,
                'NON_FARMER': strategy_counts['NON_FARMER'] / total_farmers
            }
        }
        
        if agri_income_ratios:
            stats['farmer_heterogeneity']['avg_agri_income_ratio'] = np.mean(agri_income_ratios)
        
        if farming_abilities:
            stats['farmer_heterogeneity']['farming_ability_dist'] = {
                'mean': float(np.mean(farming_abilities)),
                'std': float(np.std(farming_abilities)),
                'min': float(np.min(farming_abilities)),
                'max': float(np.max(farming_abilities))
            }
    else:
        stats['farmer_heterogeneity'] = {
            'livelihood_strategy_dist': {'PURE_FARMER': 0, 'PLURIACTIVE': 0, 'NON_FARMER': 0},
            'avg_agri_income_ratio': 0.0,
            'farming_ability_dist': {'mean': 0.0, 'std': 0.0}
        }
    
    return stats


def calculate_gini(income_list: List[float]) -> float:
    """计算基尼系数"""
    if not income_list or len(income_list) < 2:
        return 0.0
    
    income_list = sorted(income_list)
    n = len(income_list)
    cumsum = np.cumsum(income_list)
    
    # Gini = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
    numerator = 2 * sum((i + 1) * y for i, y in enumerate(income_list))
    denominator = n * sum(income_list)
    
    if denominator == 0:
        return 0.0
    
    gini = (numerator / denominator) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def _run_single_simulation(config_dict: Dict[str, Any], run_id: int, n_steps: int, run_seed: int) -> Optional[Dict[str, Any]]:
    """
    运行单次模拟（用于并行计算）
    
    @fix 2024-11-20: 提取为独立函数，支持并行计算
    专家审阅意见：SMM校准需要运行上千次模型，单核跑会让校准耗时数天
    
    参数:
        config_dict: 配置字典（已深拷贝）
        run_id: 运行ID
        n_steps: 运行步数
        run_seed: 随机种子
    
    返回:
        stats: 统计结果，如果失败返回None
    """
    try:
        # 设置随机种子
        config_dict['experiment']['seed'] = run_seed
        
        # 构建模型
        builder = ModelBuilder(config_dict)
        model = builder.build()
        
        # 运行模型
        for step in range(n_steps):
            model.step()
        
        # 收集统计量
        stats = collect_model_statistics(model)
        stats['run_id'] = run_id
        stats['seed'] = run_seed
        stats['steps'] = n_steps
        
        return stats
        
    except Exception as e:
        import traceback
        error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        error_trace = traceback.format_exc().encode('utf-8', errors='replace').decode('utf-8')
        logger.error(f"运行 {run_id + 1} 失败: {error_msg}")
        logger.debug(f"错误堆栈:\n{error_trace}")
        return None


def run_batch_simulation(
    config_path: str = "config/master_config.yaml",
    n_runs: int = 10,
    n_steps: int = 30,
    seed: int = None,
    scenario_id: Optional[str] = None,
    n_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    批量运行模型
    
    @fix 2024-11-20: 引入并行计算支持
    专家审阅意见：使用multiprocessing加速批量运行
    
    参数:
        config_path: 配置文件路径（相对于项目根目录）
        n_runs: 运行次数
        n_steps: 每次运行的步数
        seed: 随机种子（如果为None，每次运行使用不同种子）
        scenario_id: 政策情景ID
        n_workers: 并行工作进程数（None=自动检测CPU核心数）
    
    返回:
        results: 每次运行的统计结果列表
    """
    from copy import deepcopy
    from multiprocessing import Pool, cpu_count
    
    # 确保配置文件路径是绝对路径
    if not Path(config_path).is_absolute():
        config_path = _project_root / config_path
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 如果指定了政策情景，更新配置
    if scenario_id:
        if 'policy' not in config:
            config['policy'] = {}
        config['policy']['active_scenario'] = scenario_id
        logger.info(f"使用政策情景: {scenario_id}")
    
    # 确定并行工作进程数
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # 保留一个核心给系统
    n_workers = min(n_workers, n_runs)  # 不超过运行次数
    
    # 准备并行任务参数
    tasks = []
    for run_id in range(n_runs):
        run_seed = seed if seed is not None else config.get('experiment', {}).get('seed', 42) + run_id
        # 深拷贝配置，避免并行时修改冲突
        config_copy = deepcopy(config)
        tasks.append((config_copy, run_id, n_steps, run_seed))
    
    # 并行执行
    if n_workers > 1 and n_runs > 1:
        logger.info(f"使用 {n_workers} 个进程并行运行 {n_runs} 次模拟...")
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(_run_single_simulation, tasks)
        # 过滤掉None（失败的运行）
        results = [r for r in results if r is not None]
    else:
        # 串行执行（单进程或单次运行）
        logger.info(f"串行运行 {n_runs} 次模拟...")
        results = []
        for config_copy, run_id, n_steps, run_seed in tasks:
            logger.info(f"运行 {run_id + 1}/{n_runs}...")
            stats = _run_single_simulation(config_copy, run_id, n_steps, run_seed)
            if stats:
                results.append(stats)
                logger.info(f"运行 {run_id + 1} 完成，土地流转率: {stats['land_transfer'].get('transfer_rate', 0):.3f}")
    
    logger.info(f"批量运行完成：{len(results)}/{n_runs} 次成功")
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    聚合多次运行的结果，计算均值和标准差
    """
    if not results:
        return {}
    
    aggregated = {}
    
    # 聚合企业类型分布
    enterprise_dists = [r.get('enterprise_type_distribution', {}) for r in results]
    if enterprise_dists:
        all_types = set()
        for dist in enterprise_dists:
            all_types.update(dist.keys())
        
        aggregated['enterprise_type_distribution'] = {}
        for ent_type in all_types:
            values = [d.get(ent_type, 0) for d in enterprise_dists]
            aggregated['enterprise_type_distribution'][ent_type] = _summarize_scalar(values)
    
    # 聚合作物结构
    crop_ratios = [r.get('crop_structure', {}) for r in results]
    if crop_ratios:
        aggregated['crop_structure'] = {
            'grain_crop_ratio': _summarize_scalar([c.get('grain_crop_ratio', 0) for c in crop_ratios]),
            'cash_crop_ratio': _summarize_scalar([c.get('cash_crop_ratio', 0) for c in crop_ratios])
        }
    
    # 聚合土地流转
    land_transfers = [r.get('land_transfer', {}) for r in results]
    if land_transfers:
        aggregated['land_transfer'] = {
            'transfer_rate': _summarize_scalar([l.get('transfer_rate', 0) for l in land_transfers]),
            'average_transfer_price': _summarize_scalar([l.get('average_transfer_price', 0) for l in land_transfers])
        }
    
    # 聚合收入分布
    income_dists = [r.get('income_distribution', {}) for r in results]
    if income_dists:
        aggregated['income_distribution'] = {
            'avg_farm_income': _summarize_scalar([i.get('avg_farm_income', 0) for i in income_dists]),
            'gini_coefficient': _summarize_scalar([i.get('gini_coefficient', 0) for i in income_dists])
        }
    
    # 聚合生产率
    productivity = [r.get('productivity_indicators', {}) for r in results]
    if productivity:
        aggregated['productivity_indicators'] = {
            'avg_yield_per_hectare': _summarize_scalar([p.get('avg_yield_per_hectare', 0) for p in productivity])
        }
    
    # 聚合补贴统计
    subsidies = [r.get('subsidy_statistics', {}) for r in results]
    if subsidies:
        aggregated['subsidy_statistics'] = {
            'total_subsidy_paid': _summarize_scalar([s.get('total_subsidy_paid', 0) for s in subsidies])
        }
    
    # 聚合农户异质性指标（新增）
    farmer_heteros = [r.get('farmer_heterogeneity', {}) for r in results]
    if farmer_heteros and any(fh for fh in farmer_heteros):
        aggregated['farmer_heterogeneity'] = {}
        
        # 聚合生计策略分布
        strategy_dists = [fh.get('livelihood_strategy_dist', {}) for fh in farmer_heteros]
        if strategy_dists:
            aggregated['farmer_heterogeneity']['livelihood_strategy_dist'] = {
                'PURE_FARMER': _summarize_scalar([sd.get('PURE_FARMER', 0) for sd in strategy_dists]),
                'PLURIACTIVE': _summarize_scalar([sd.get('PLURIACTIVE', 0) for sd in strategy_dists]),
                'NON_FARMER': _summarize_scalar([sd.get('NON_FARMER', 0) for sd in strategy_dists])
            }
        
        # 聚合平均农业收入占比
        agri_ratios = [fh.get('avg_agri_income_ratio', 0) for fh in farmer_heteros if 'avg_agri_income_ratio' in fh]
        if agri_ratios:
            aggregated['farmer_heterogeneity']['avg_agri_income_ratio'] = _summarize_scalar(agri_ratios)
        
        # 聚合耕作能力分布
        ability_dists = [fh.get('farming_ability_dist', {}) for fh in farmer_heteros]
        if ability_dists and any(ad for ad in ability_dists):
            ability_means = [ad.get('mean', 0) for ad in ability_dists if 'mean' in ad]
            ability_stds = [ad.get('std', 0) for ad in ability_dists if 'std' in ad]
            
            if ability_means:
                aggregated['farmer_heterogeneity']['farming_ability_dist'] = {
                    'mean': _summarize_scalar(ability_means),
                    'std': _summarize_scalar(ability_stds) if ability_stds else _summarize_scalar([])
                }
    
    aggregated['n_runs'] = len(results)
    
    return aggregated


def list_available_scenarios():
    """列出可用的政策情景"""
    policy_path = _project_root / "config" / "policy_params.json"
    if policy_path.exists():
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
            scenarios = policy_data.get('scenarios', {})
            print("\n可用的政策情景:")
            for scenario_id, scenario_info in scenarios.items():
                print(f"  {scenario_id}: {scenario_info.get('display_name', scenario_id)}")
                print(f"    描述: {scenario_info.get('description', '无描述')}")
            return list(scenarios.keys())
    return []

def create_visualizations(results: List[Dict[str, Any]], aggregated: Dict[str, Any], 
                          output_dir: Path, scenario_id: Optional[str] = None):
    """创建可视化图表"""
    logger.info("生成可视化图表...")
    
    # 创建图表目录
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 企业类型分布对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 企业类型分布（条形图）
    ax = axes[0, 0]
    ent_dist = aggregated.get('enterprise_type_distribution', {})
    if ent_dist:
        types = list(ent_dist.keys())
        means = [ent_dist[t].get('mean', 0) for t in types]
        stds = [ent_dist[t].get('std', 0) for t in types]
        x_pos = np.arange(len(types))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.set_ylabel('比例')
        ax.set_title('企业类型分布')
        ax.grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom')
    
    # 1.2 作物结构（饼图）
    ax = axes[0, 1]
    crop_struct = aggregated.get('crop_structure', {})
    if crop_struct:
        grain_ratio = crop_struct.get('grain_crop_ratio', {}).get('mean', 0)
        cash_ratio = crop_struct.get('cash_crop_ratio', {}).get('mean', 0)
        labels = ['粮食作物', '经济作物']
        sizes = [grain_ratio, cash_ratio]
        colors = ['#ff9999', '#66b3ff']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('作物结构分布')
    
    # 1.3 土地流转率时间序列
    ax = axes[1, 0]
    transfer_rates = [r.get('land_transfer', {}).get('transfer_rate', 0) for r in results]
    ax.plot(range(len(transfer_rates)), transfer_rates, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('运行编号')
    ax.set_ylabel('土地流转率')
    ax.set_title('土地流转率变化')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(transfer_rates), color='r', linestyle='--', 
               label=f'平均值: {np.mean(transfer_rates):.4f}')
    ax.legend()
    
    # 1.4 收入分布（基尼系数）
    ax = axes[1, 1]
    gini_coeffs = [r.get('income_distribution', {}).get('gini_coefficient', 0) for r in results]
    avg_incomes = [r.get('income_distribution', {}).get('avg_farm_income', 0) for r in results]
    scatter = ax.scatter(avg_incomes, gini_coeffs, s=100, alpha=0.6, c=range(len(gini_coeffs)), 
                        cmap='viridis')
    ax.set_xlabel('平均农户收入')
    ax.set_ylabel('基尼系数')
    ax.set_title('收入分布与不平等')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='运行编号')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "key_indicators.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 补贴统计
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2.1 补贴总额分布
    ax = axes[0]
    subsidy_totals = [r.get('subsidy_statistics', {}).get('total_subsidy_paid', 0) for r in results]
    ax.hist(subsidy_totals, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=np.mean(subsidy_totals), color='r', linestyle='--', 
               label=f'平均值: {np.mean(subsidy_totals):.2f}')
    ax.set_xlabel('补贴总额')
    ax.set_ylabel('频数')
    ax.set_title('补贴总额分布')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2.2 市场价格变化
    ax = axes[1]
    grain_prices = [r.get('market_statistics', {}).get('grain_price', 0) for r in results]
    cash_prices = [r.get('market_statistics', {}).get('cash_crop_price', 0) for r in results]
    wages = [r.get('market_statistics', {}).get('wage_rate', 0) for r in results]
    x = range(len(results))
    ax.plot(x, grain_prices, 'o-', label='粮食价格', linewidth=2)
    ax.plot(x, cash_prices, 's-', label='经济作物价格', linewidth=2)
    ax.plot(x, wages, '^-', label='工资率', linewidth=2)
    ax.set_xlabel('运行编号')
    ax.set_ylabel('价格/工资')
    ax.set_title('市场价格变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "subsidy_and_market.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可视化图表已保存到: {viz_dir}")

def generate_result_interpretation(results: List[Dict[str, Any]], aggregated: Dict[str, Any],
                                   scenario_id: Optional[str] = None) -> str:
    """生成结果解读报告"""
    report = []
    report.append("=" * 80)
    report.append("批量模拟结果解读报告")
    report.append("=" * 80)
    report.append("")
    
    if scenario_id:
        report.append(f"政策情景: {scenario_id}")
        report.append("")
    
    report.append(f"运行次数: {len(results)}")
    report.append("")
    
    # 1. 企业类型分布
    report.append("1. 企业类型分布")
    report.append("-" * 80)
    ent_dist = aggregated.get('enterprise_type_distribution', {})
    if ent_dist:
        for ent_type, stats in ent_dist.items():
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            report.append(f"  {ent_type}: {mean:.1%} (±{std:.1%})")
    report.append("")
    
    # 2. 作物结构
    report.append("2. 作物结构")
    report.append("-" * 80)
    crop_struct = aggregated.get('crop_structure', {})
    if crop_struct:
        grain_ratio = crop_struct.get('grain_crop_ratio', {}).get('mean', 0)
        cash_ratio = crop_struct.get('cash_crop_ratio', {}).get('mean', 0)
        report.append(f"  粮食作物比例: {grain_ratio:.1%}")
        report.append(f"  经济作物比例: {cash_ratio:.1%}")
    report.append("")
    
    # 3. 土地流转
    report.append("3. 土地流转")
    report.append("-" * 80)
    land_transfer = aggregated.get('land_transfer', {})
    if land_transfer:
        transfer_rate = land_transfer.get('transfer_rate', {}).get('mean', 0)
        avg_price = land_transfer.get('average_transfer_price', {}).get('mean', 0)
        report.append(f"  平均流转率: {transfer_rate:.2%}")
        report.append(f"  平均流转价格: {avg_price:.2f} 元/亩")
    report.append("")
    
    # 4. 收入分布
    report.append("4. 收入分布")
    report.append("-" * 80)
    income_dist = aggregated.get('income_distribution', {})
    if income_dist:
        avg_income = income_dist.get('avg_farm_income', {}).get('mean', 0)
        gini = income_dist.get('gini_coefficient', {}).get('mean', 0)
        report.append(f"  平均农户收入: {avg_income:.2f} 元")
        report.append(f"  基尼系数: {gini:.3f}")
        if gini < 0.3:
            report.append("  解读: 收入分配相对平等")
        elif gini < 0.5:
            report.append("  解读: 收入分配中等不平等")
        else:
            report.append("  解读: 收入分配高度不平等")
    report.append("")
    
    # 5. 生产率
    report.append("5. 生产率指标")
    report.append("-" * 80)
    productivity = aggregated.get('productivity_indicators', {})
    if productivity:
        yield_per_hectare = productivity.get('avg_yield_per_hectare', {}).get('mean', 0)
        report.append(f"  平均单产: {yield_per_hectare:.2f} 公斤/公顷")
    report.append("")
    
    # 6. 补贴统计
    report.append("6. 补贴统计")
    report.append("-" * 80)
    subsidy_stats = aggregated.get('subsidy_statistics', {})
    if subsidy_stats:
        total_subsidy = subsidy_stats.get('total_subsidy_paid', {}).get('mean', 0)
        report.append(f"  平均补贴总额: {total_subsidy:.2f} 元")
    report.append("")
    
    # 7. 模型校准状态
    report.append("7. 模型校准状态")
    report.append("-" * 80)
    report.append("  当前状态: 未完成完整校准流程")
    report.append("  建议:")
    report.append("    - 使用2020年CLES数据进行参数校准")
    report.append("    - 使用2021-2022年数据进行仿真验证")
    report.append("    - 对比模拟结果与真实数据，评估模型拟合度")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="批量运行模型并收集统计指标")
    parser.add_argument('--config', type=str, default='config/master_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--runs', '--run', type=int, default=10,
                       dest='runs',
                       help='运行次数')
    parser.add_argument('--steps', type=int, default=30,
                       help='每次运行的步数')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（如果指定，所有运行使用相同种子）')
    parser.add_argument('--scenario', '--policy', type=str, default=None,
                       dest='scenario',
                       help='政策情景ID (例如: baseline_cash, owner_tiered_cash, mixed_in_kind_support)')
    parser.add_argument('--list-scenarios', action='store_true',
                       help='列出所有可用的政策情景')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（默认: results/batch_output_<scenario>_<timestamp>.json）')
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成可视化图表')
    parser.add_argument('--verbose', action='store_true',
                       help='启用详细日志')
    
    args = parser.parse_args()
    
    # 如果请求列出情景，则列出并退出
    if args.list_scenarios:
        list_available_scenarios()
        return
    
    # 配置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    
    # 确定输出路径
    if args.output:
        if not Path(args.output).is_absolute():
            output_path = _project_root / args.output
        else:
            output_path = Path(args.output)
    else:
        # 生成默认输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_suffix = f"_{args.scenario}" if args.scenario else ""
        output_path = _project_root / "results" / f"batch_output{scenario_suffix}_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent
    
    # 运行批量模拟
    logger.info(f"开始批量运行：{args.runs} 次，每次 {args.steps} 步")
    if args.scenario:
        logger.info(f"政策情景: {args.scenario}")
    
    results = run_batch_simulation(
        config_path=args.config,
        n_runs=args.runs,
        n_steps=args.steps,
        seed=args.seed,
        scenario_id=args.scenario
    )
    
    # 聚合结果
    aggregated = aggregate_results(results)
    
    # 保存结果
    output_data = {
        'individual_runs': results,
        'aggregated_statistics': aggregated,
        'metadata': {
            'n_runs': len(results),
            'n_steps': args.steps,
            'config_path': args.config,
            'seed': args.seed,
            'scenario_id': args.scenario,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {output_path}")
    
    # 生成可视化
    if not args.no_viz:
        create_visualizations(results, aggregated, output_dir, args.scenario)
    
    # 生成结果解读报告
    interpretation = generate_result_interpretation(results, aggregated, args.scenario)
    interpretation_path = output_dir / f"{output_path.stem}_interpretation.txt"
    with open(interpretation_path, 'w', encoding='utf-8') as f:
        f.write(interpretation)
    logger.info(f"结果解读报告已保存到: {interpretation_path}")
    
    # 打印关键指标
    logger.info("=" * 60)
    logger.info("关键指标摘要:")
    logger.info(f"  平均土地流转率: {aggregated.get('land_transfer', {}).get('transfer_rate', {}).get('mean', 0):.3f}")
    logger.info(f"  平均基尼系数: {aggregated.get('income_distribution', {}).get('gini_coefficient', {}).get('mean', 0):.3f}")
    logger.info(f"  平均农户收入: {aggregated.get('income_distribution', {}).get('avg_farm_income', {}).get('mean', 0):.2f} 元")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

