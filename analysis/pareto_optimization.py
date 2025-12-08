# -*- coding: utf-8 -*-
"""
多目标帕累托优化分析模块 (Multi-Objective Pareto Optimization)
=============================================================
基于NSGA-II算法的粮食补贴政策多目标优化

理论基础：
1. 帕累托最优：不存在另一解能在不牺牲任何目标的情况下改善某目标
2. NSGA-II (Non-dominated Sorting Genetic Algorithm II)：
   - Deb et al. (2002) 提出的经典多目标进化算法
   - 使用非支配排序 + 拥挤距离选择
   
政策目标权重文献依据：
1. 粮食安全 (40%):
   - 《中国粮食安全白皮书(2019)》：粮食安全是国家安全战略基石
   - 习近平(2022)："中国人的饭碗任何时候都要牢牢端在自己手中"
   - FAO指标：粮食自给率、人均占有量是核心指标
   
2. 农户福利 (35%):
   - 《乡村振兴战略规划(2018-2022)》：农民增收是核心目标
   - Huang et al. (2012): 补贴对农户收入有显著正向效应
   - 黄宗智(2014)：保护小农利益是中国农业政策的底线
   
3. 财政效率 (25%):
   - 财政部《农业补贴政策绩效评估指南》
   - 曹俊杰(2018)：补贴边际效益递减，需关注财政可持续性
   - OECD(2020)：农业补贴应向"绿箱"政策转型

参考文献：
- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
- 黄季焜等 (2012). 中国粮食直接补贴政策的效果评估
- 曹俊杰 (2018). 农业补贴政策的财政效率分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 政策目标权重配置 - 可灵活调整
# =============================================================================

@dataclass
class ObjectiveWeights:
    """
    政策目标权重配置
    
    文献依据及建议范围：
    - grain_security: 30-50% (粮食安全国家战略，权重较高)
    - farmer_welfare: 25-45% (乡村振兴核心，中高权重)
    - fiscal_efficiency: 15-30% (财政约束，中低权重)
    - structural_transformation: 0-20% (可选：农业结构转型)
    """
    grain_security: float = 0.40        # 粮食安全
    farmer_welfare: float = 0.35        # 农户福利
    fiscal_efficiency: float = 0.25     # 财政效率
    structural_transformation: float = 0.0  # 结构转型（可选）
    
    def __post_init__(self):
        total = (self.grain_security + self.farmer_welfare + 
                 self.fiscal_efficiency + self.structural_transformation)
        if abs(total - 1.0) > 0.01:
            # 自动归一化
            self.grain_security /= total
            self.farmer_welfare /= total
            self.fiscal_efficiency /= total
            self.structural_transformation /= total
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ObjectiveWeights':
        return cls(
            grain_security=d.get('grain_security', 0.40),
            farmer_welfare=d.get('farmer_welfare', 0.35),
            fiscal_efficiency=d.get('fiscal_efficiency', 0.25),
            structural_transformation=d.get('structural_transformation', 0.0)
        )
    
    def to_dict(self) -> dict:
        return {
            'grain_security': self.grain_security,
            'farmer_welfare': self.farmer_welfare,
            'fiscal_efficiency': self.fiscal_efficiency,
            'structural_transformation': self.structural_transformation
        }


# 预定义的权重配置方案
WEIGHT_PRESETS = {
    'balanced': ObjectiveWeights(0.40, 0.35, 0.25, 0.0),  # 平衡型（默认）
    'grain_first': ObjectiveWeights(0.55, 0.25, 0.20, 0.0),  # 粮食优先
    'welfare_first': ObjectiveWeights(0.30, 0.50, 0.20, 0.0),  # 福利优先
    'fiscal_first': ObjectiveWeights(0.30, 0.30, 0.40, 0.0),  # 财政优先
    'transformation': ObjectiveWeights(0.30, 0.30, 0.20, 0.20),  # 结构转型
}


# =============================================================================
# NSGA-II 核心算法实现
# =============================================================================

@dataclass
class Individual:
    """个体（政策组合）"""
    policy_params: Dict[str, float]  # 政策参数
    objectives: Dict[str, float] = None  # 目标函数值
    rank: int = 0  # 非支配排名
    crowding_distance: float = 0.0  # 拥挤距离


def dominates(a: Dict, b: Dict, objectives: List[str], maximize: List[bool]) -> bool:
    """
    判断a是否支配b
    
    Args:
        a, b: 两个个体的目标值字典
        objectives: 目标名称列表
        maximize: 每个目标是否最大化
    """
    better_in_one = False
    for obj, is_max in zip(objectives, maximize):
        val_a, val_b = a.get(obj, 0), b.get(obj, 0)
        if is_max:
            if val_a < val_b:
                return False
            if val_a > val_b:
                better_in_one = True
        else:
            if val_a > val_b:
                return False
            if val_a < val_b:
                better_in_one = True
    return better_in_one


def non_dominated_sort(population: List[Individual], 
                       objectives: List[str],
                       maximize: List[bool]) -> List[List[int]]:
    """
    非支配排序
    
    Returns:
        fronts: 帕累托前沿列表，fronts[0]是第一前沿（最优）
    """
    n = len(population)
    domination_count = [0] * n  # 被支配次数
    dominated_set = [[] for _ in range(n)]  # 支配的个体集合
    fronts = [[]]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i].objectives, population[j].objectives, objectives, maximize):
                dominated_set[i].append(j)
            elif dominates(population[j].objectives, population[i].objectives, objectives, maximize):
                domination_count[i] += 1
        
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts


def calculate_crowding_distance(population: List[Individual],
                                front: List[int],
                                objectives: List[str]):
    """计算拥挤距离"""
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = float('inf')
        return
    
    for idx in front:
        population[idx].crowding_distance = 0
    
    for obj in objectives:
        # 按目标值排序
        sorted_front = sorted(front, key=lambda x: population[x].objectives.get(obj, 0))
        
        # 边界点设为无穷大
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')
        
        # 计算中间点的拥挤距离
        obj_min = population[sorted_front[0]].objectives.get(obj, 0)
        obj_max = population[sorted_front[-1]].objectives.get(obj, 0)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1
        
        for i in range(1, len(sorted_front) - 1):
            prev_val = population[sorted_front[i-1]].objectives.get(obj, 0)
            next_val = population[sorted_front[i+1]].objectives.get(obj, 0)
            population[sorted_front[i]].crowding_distance += (next_val - prev_val) / obj_range


def binary_tournament(pop: List[Individual], idx1: int, idx2: int) -> int:
    """二元锦标赛选择"""
    if pop[idx1].rank < pop[idx2].rank:
        return idx1
    elif pop[idx1].rank > pop[idx2].rank:
        return idx2
    else:
        if pop[idx1].crowding_distance > pop[idx2].crowding_distance:
            return idx1
        else:
            return idx2


def crossover(parent1: Dict, parent2: Dict, crossover_rate: float = 0.9) -> Tuple[Dict, Dict]:
    """模拟二进制交叉 (SBX)"""
    child1, child2 = deepcopy(parent1), deepcopy(parent2)
    
    if np.random.random() < crossover_rate:
        for key in parent1.keys():
            if np.random.random() < 0.5:
                # SBX交叉
                eta = 20  # 分布指数
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                p1, p2 = parent1[key], parent2[key]
                child1[key] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                child2[key] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    
    return child1, child2


def mutate(individual: Dict, bounds: Dict, mutation_rate: float = 0.1) -> Dict:
    """多项式变异"""
    mutated = deepcopy(individual)
    eta = 20  # 分布指数
    
    for key, (low, high) in bounds.items():
        if np.random.random() < mutation_rate:
            x = mutated[key]
            delta = (x - low) / (high - low) if high != low else 0.5
            
            u = np.random.random()
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            mutated[key] = np.clip(x + delta_q * (high - low), low, high)
    
    return mutated


class NSGAII:
    """
    NSGA-II 多目标优化器
    
    用于寻找政策参数的帕累托最优前沿
    """
    
    def __init__(
        self,
        evaluate_func: Callable[[Dict], Dict],
        param_bounds: Dict[str, Tuple[float, float]],
        objectives: List[str],
        maximize: List[bool],
        pop_size: int = 50,
        n_generations: int = 30,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1
    ):
        """
        Args:
            evaluate_func: 评估函数，输入政策参数，返回目标值字典
            param_bounds: 参数边界 {param_name: (min, max)}
            objectives: 目标名称列表
            maximize: 每个目标是否最大化
            pop_size: 种群大小
            n_generations: 迭代代数
        """
        self.evaluate = evaluate_func
        self.param_bounds = param_bounds
        self.objectives = objectives
        self.maximize = maximize
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.population = []
        self.history = []
    
    def initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            params = {}
            for key, (low, high) in self.param_bounds.items():
                params[key] = np.random.uniform(low, high)
            
            ind = Individual(policy_params=params)
            ind.objectives = self.evaluate(params)
            population.append(ind)
        
        return population
    
    def evolve(self, verbose: bool = True) -> List[Individual]:
        """
        执行进化优化
        
        Returns:
            最终帕累托前沿的个体列表
        """
        # 初始化
        self.population = self.initialize_population()
        
        for gen in range(self.n_generations):
            if verbose:
                print(f"  Generation {gen + 1}/{self.n_generations}...", end=" ")
            
            # 非支配排序
            fronts = non_dominated_sort(self.population, self.objectives, self.maximize)
            
            # 计算拥挤距离
            for front in fronts:
                calculate_crowding_distance(self.population, front, self.objectives)
            
            # 选择、交叉、变异
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选择
                idx1 = binary_tournament(self.population, 
                                         np.random.randint(self.pop_size),
                                         np.random.randint(self.pop_size))
                idx2 = binary_tournament(self.population,
                                         np.random.randint(self.pop_size),
                                         np.random.randint(self.pop_size))
                
                # 交叉
                child1_params, child2_params = crossover(
                    self.population[idx1].policy_params,
                    self.population[idx2].policy_params,
                    self.crossover_rate
                )
                
                # 变异
                child1_params = mutate(child1_params, self.param_bounds, self.mutation_rate)
                child2_params = mutate(child2_params, self.param_bounds, self.mutation_rate)
                
                # 评估
                child1 = Individual(policy_params=child1_params)
                child1.objectives = self.evaluate(child1_params)
                offspring.append(child1)
                
                if len(offspring) < self.pop_size:
                    child2 = Individual(policy_params=child2_params)
                    child2.objectives = self.evaluate(child2_params)
                    offspring.append(child2)
            
            # 合并父代和子代
            combined = self.population + offspring
            
            # 非支配排序
            fronts = non_dominated_sort(combined, self.objectives, self.maximize)
            
            # 选择下一代
            new_population = []
            front_idx = 0
            while len(new_population) + len(fronts[front_idx]) <= self.pop_size:
                for idx in fronts[front_idx]:
                    new_population.append(combined[idx])
                front_idx += 1
                if front_idx >= len(fronts):
                    break
            
            # 如果还需要填充，用拥挤距离排序
            if len(new_population) < self.pop_size and front_idx < len(fronts):
                calculate_crowding_distance(combined, fronts[front_idx], self.objectives)
                sorted_front = sorted(fronts[front_idx], 
                                     key=lambda x: combined[x].crowding_distance,
                                     reverse=True)
                remaining = self.pop_size - len(new_population)
                for idx in sorted_front[:remaining]:
                    new_population.append(combined[idx])
            
            self.population = new_population
            
            # 记录历史
            pareto_front = [self.population[i] for i in fronts[0] if i < len(self.population)]
            self.history.append({
                'generation': gen + 1,
                'pareto_size': len(pareto_front),
                'best_objectives': {
                    obj: max(ind.objectives.get(obj, 0) for ind in pareto_front) if is_max 
                         else min(ind.objectives.get(obj, 0) for ind in pareto_front)
                    for obj, is_max in zip(self.objectives, self.maximize)
                }
            })
            
            if verbose:
                print(f"帕累托前沿大小: {len(pareto_front)}")
        
        # 返回最终帕累托前沿
        fronts = non_dominated_sort(self.population, self.objectives, self.maximize)
        return [self.population[i] for i in fronts[0]]
    
    def get_pareto_front_df(self) -> pd.DataFrame:
        """获取帕累托前沿的DataFrame"""
        fronts = non_dominated_sort(self.population, self.objectives, self.maximize)
        pareto_individuals = [self.population[i] for i in fronts[0]]
        
        records = []
        for ind in pareto_individuals:
            record = {**ind.policy_params, **ind.objectives}
            records.append(record)
        
        return pd.DataFrame(records)


# =============================================================================
# 政策评估函数封装
# =============================================================================

def create_policy_evaluator(
    model_runner: Callable[[Dict], Dict],
    baseline_metrics: Dict,
    weights: ObjectiveWeights = None
) -> Callable[[Dict], Dict]:
    """
    创建政策评估函数
    
    Args:
        model_runner: 模型运行函数，输入配置，返回指标
        baseline_metrics: 基准情景指标
        weights: 目标权重
    
    Returns:
        评估函数，输入政策参数，返回目标值
    """
    if weights is None:
        weights = ObjectiveWeights()
    
    def evaluate(policy_params: Dict) -> Dict:
        # 运行模型
        metrics = model_runner(policy_params)
        
        # 计算目标值
        grain_output = metrics.get('grain_output_total', 0)
        farmer_income = metrics.get('avg_farmer_income', 0)
        subsidy_cost = metrics.get('total_subsidy_expenditure', 1)
        pure_farmer_count = metrics.get('pure_farmer_count', 0)
        
        baseline_grain = baseline_metrics.get('grain_output_total', 1)
        baseline_income = baseline_metrics.get('avg_farmer_income', 1)
        
        objectives = {
            # 粮食安全：产量增幅
            'grain_security': (grain_output / baseline_grain - 1) * 100,
            # 农户福利：收入增幅
            'farmer_welfare': (farmer_income / baseline_income - 1) * 100,
            # 财政效率：产量/补贴（越高越好）
            'fiscal_efficiency': grain_output / max(1, subsidy_cost) * 100,
            # 结构转型：纯农户比例
            'structural_transformation': pure_farmer_count,
            # 综合得分（加权和）
            'weighted_score': (
                weights.grain_security * (grain_output / baseline_grain - 1) * 100 +
                weights.farmer_welfare * (farmer_income / baseline_income - 1) * 100 +
                weights.fiscal_efficiency * np.log1p(grain_output / max(1, subsidy_cost)) +
                weights.structural_transformation * pure_farmer_count / 100
            )
        }
        
        return objectives
    
    return evaluate


# =============================================================================
# 可视化
# =============================================================================

def plot_pareto_front(
    pareto_df: pd.DataFrame,
    obj_x: str = 'grain_security',
    obj_y: str = 'farmer_welfare',
    obj_color: str = 'fiscal_efficiency',
    title: str = '帕累托最优前沿',
    output_path: Path = None
):
    """绘制帕累托前沿"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        pareto_df[obj_x],
        pareto_df[obj_y],
        c=pareto_df[obj_color],
        cmap='RdYlGn',
        s=100,
        alpha=0.7,
        edgecolors='black'
    )
    
    ax.set_xlabel(f'{obj_x} (%)', fontsize=12)
    ax.set_ylabel(f'{obj_y} (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(obj_color, fontsize=10)
    
    # 标注极端点
    if len(pareto_df) > 0:
        # 产量最优
        grain_best = pareto_df.loc[pareto_df[obj_x].idxmax()]
        ax.annotate(f'产量最优\n补贴率={grain_best.get("grain_subsidies_rate", 0):.2f}',
                   xy=(grain_best[obj_x], grain_best[obj_y]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='red')
        
        # 福利最优
        welfare_best = pareto_df.loc[pareto_df[obj_y].idxmax()]
        ax.annotate(f'福利最优\n补贴率={welfare_best.get("grain_subsidies_rate", 0):.2f}',
                   xy=(welfare_best[obj_x], welfare_best[obj_y]),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=9, color='blue')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"帕累托前沿图已保存: {output_path}")
    
    return fig


def plot_objective_tradeoffs(
    pareto_df: pd.DataFrame,
    objectives: List[str],
    title: str = '目标权衡分析',
    output_path: Path = None
):
    """绘制目标间权衡关系"""
    n_obj = len(objectives)
    fig, axes = plt.subplots(n_obj, n_obj, figsize=(12, 12))
    
    for i, obj_i in enumerate(objectives):
        for j, obj_j in enumerate(objectives):
            ax = axes[i, j]
            if i == j:
                ax.hist(pareto_df[obj_i], bins=20, color='steelblue', alpha=0.7)
                ax.set_xlabel(obj_i if i == n_obj - 1 else '')
            else:
                ax.scatter(pareto_df[obj_j], pareto_df[obj_i], alpha=0.5, s=30)
                ax.set_xlabel(obj_j if i == n_obj - 1 else '')
                ax.set_ylabel(obj_i if j == 0 else '')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("多目标帕累托优化模块测试")
    print("=" * 70)
    
    # 测试用简化评估函数
    def mock_evaluate(params: Dict) -> Dict:
        subsidy = params.get('grain_subsidies_rate', 0.5)
        # 模拟：补贴越高，收入越高，但产量有上限
        grain = 100 + 50 * subsidy - 20 * subsidy ** 2 + np.random.normal(0, 5)
        income = 50 + 80 * subsidy + np.random.normal(0, 3)
        fiscal = 100 - 30 * subsidy + np.random.normal(0, 2)
        
        return {
            'grain_security': grain,
            'farmer_welfare': income,
            'fiscal_efficiency': fiscal
        }
    
    # 运行NSGA-II
    optimizer = NSGAII(
        evaluate_func=mock_evaluate,
        param_bounds={'grain_subsidies_rate': (0.0, 1.0)},
        objectives=['grain_security', 'farmer_welfare', 'fiscal_efficiency'],
        maximize=[True, True, True],
        pop_size=30,
        n_generations=20
    )
    
    print("\n开始优化...")
    pareto_front = optimizer.evolve(verbose=True)
    
    print(f"\n最终帕累托前沿包含 {len(pareto_front)} 个解")
    
    # 获取DataFrame
    df = optimizer.get_pareto_front_df()
    print("\n帕累托最优解:")
    print(df.to_string())
    
    # 显示不同权重方案
    print("\n\n预定义权重方案:")
    for name, weights in WEIGHT_PRESETS.items():
        print(f"  {name}: 粮食{weights.grain_security:.0%}, 福利{weights.farmer_welfare:.0%}, "
              f"财政{weights.fiscal_efficiency:.0%}, 转型{weights.structural_transformation:.0%}")
