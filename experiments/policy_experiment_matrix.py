# -*- coding: utf-8 -*-
"""
@module policy_experiment_matrix.py
@description 政策实验设计框架 - 反事实分析与政策优化
@created 2025-12-06
@author GitHub Copilot

理论参考：
1. Rubin Causal Model (Rubin, 1974)
   - 潜在结果框架
   - 平均处理效应 (ATE)
   - 反事实推断

2. 多目标政策优化 (Pareto优化)
   - 帕累托前沿
   - 多目标权衡

3. 实验设计方法
   - 全因子设计
   - 正交设计 (Taguchi方法)
   - 响应面设计

核心功能：
1. 反事实政策实验 - 评估政策净效应
2. 政策梯度实验 - 寻找最优政策组合
3. 政策交互实验 - 测试政策协同效应
4. 帕累托前沿分析 - 多目标政策优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from datetime import datetime
import copy
import warnings

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class PolicyScenario:
    """政策情景定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    is_baseline: bool = False
    is_counterfactual: bool = False


@dataclass
class ExperimentResult:
    """实验结果"""
    scenario_name: str
    parameters: Dict[str, Any]
    outputs: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreatmentEffect:
    """处理效应结果"""
    outcome_name: str
    ate: float  # 平均处理效应
    att: float  # 处理组平均效应
    ci_lower: float  # 置信区间下限
    ci_upper: float  # 置信区间上限
    p_value: float  # 假设检验p值
    n_treated: int
    n_control: int


class PolicyExperimentMatrix:
    """
    政策实验设计矩阵
    
    实现Rubin因果模型框架下的政策实验设计
    
    使用方法：
    ```python
    matrix = PolicyExperimentMatrix()
    
    # 全因子实验
    scenarios = matrix.design_full_factorial_experiment()
    
    # 正交实验
    scenarios = matrix.design_orthogonal_array_experiment()
    
    # 运行实验
    results = matrix.run_experiments(run_function, scenarios)
    ```
    """
    
    def __init__(self,
                 policy_factors: Optional[Dict[str, List[float]]] = None,
                 n_replications: int = 10,
                 n_jobs: int = -1):
        """
        初始化实验矩阵
        
        Args:
            policy_factors: 政策因子定义 {param_name: [value1, value2, ...]}
            n_replications: 每个场景的重复次数
            n_jobs: 并行作业数
        """
        self.policy_factors = policy_factors or self._default_policy_factors()
        self.n_replications = n_replications
        self.n_jobs = n_jobs
        
    def _default_policy_factors(self) -> Dict[str, List[float]]:
        """默认政策因子"""
        return {
            'grain_subsidies_rate': [0, 100, 200, 350, 500],  # 元/亩
            'grain_price_floor': [1.0, 1.1, 1.2, 1.3],  # 保护价比例
            'service_market_enabled': [False, True],  # 服务市场
        }
    
    def design_full_factorial_experiment(self) -> List[PolicyScenario]:
        """
        全因子实验设计
        
        生成所有政策因子组合
        
        优点：可完整估计主效应和交互效应
        缺点：组合数随因子增加指数增长
        
        Returns:
            政策情景列表
        """
        scenarios = []
        
        # 生成所有组合
        factor_names = list(self.policy_factors.keys())
        factor_values = list(self.policy_factors.values())
        
        for i, combination in enumerate(product(*factor_values)):
            params = {name: val for name, val in zip(factor_names, combination)}
            
            # 生成情景名称
            name = f"scenario_{i+1}"
            desc = ", ".join([f"{k}={v}" for k, v in params.items()])
            
            scenario = PolicyScenario(
                name=name,
                description=desc,
                parameters=params,
                is_baseline=(i == 0)
            )
            scenarios.append(scenario)
        
        print(f"生成了 {len(scenarios)} 个全因子实验情景")
        return scenarios
    
    def design_orthogonal_array_experiment(self, n_levels: int = 3) -> List[PolicyScenario]:
        """
        正交实验设计（Taguchi方法）
        
        减少实验次数同时保持参数空间覆盖
        
        Returns:
            政策情景列表
        """
        # 简化的正交设计（使用拉丁超立方采样近似）
        from scipy.stats import qmc
        
        n_factors = len(self.policy_factors)
        
        # 使用拉丁超立方采样
        sampler = qmc.LatinHypercube(d=n_factors)
        n_samples = n_levels ** 2  # 简化的样本数
        sample = sampler.random(n=n_samples)
        
        scenarios = []
        factor_names = list(self.policy_factors.keys())
        
        for i, point in enumerate(sample):
            params = {}
            for j, (name, values) in enumerate(self.policy_factors.items()):
                # 将[0,1]采样点映射到实际值
                if isinstance(values[0], bool):
                    params[name] = bool(round(point[j]))
                else:
                    min_val, max_val = min(values), max(values)
                    params[name] = min_val + point[j] * (max_val - min_val)
            
            scenario = PolicyScenario(
                name=f"ortho_{i+1}",
                description=f"正交实验点 {i+1}",
                parameters=params
            )
            scenarios.append(scenario)
        
        print(f"生成了 {len(scenarios)} 个正交实验情景")
        return scenarios
    
    def run_experiments(self,
                       run_function: Callable,
                       scenarios: List[PolicyScenario],
                       base_config: Dict,
                       steps: int = 30,
                       verbose: bool = True) -> List[ExperimentResult]:
        """
        运行实验
        
        Args:
            run_function: 模型运行函数 f(config, seed) -> results
            scenarios: 政策情景列表
            base_config: 基础配置
            steps: 模拟步数
            verbose: 是否打印进度
            
        Returns:
            实验结果列表
        """
        results = []
        total = len(scenarios) * self.n_replications
        current = 0
        
        for scenario in scenarios:
            scenario_results = []
            
            for rep in range(self.n_replications):
                current += 1
                
                if verbose and current % 10 == 0:
                    print(f"  进度: {current}/{total} ({current/total*100:.1f}%)")
                
                # 应用政策参数
                config = self._apply_policy_parameters(base_config, scenario.parameters)
                
                try:
                    result = run_function(config, seed=rep, steps=steps)
                    scenario_results.append(result)
                except Exception as e:
                    warnings.warn(f"场景 {scenario.name} 重复 {rep} 失败: {e}")
            
            # 聚合结果
            if scenario_results:
                aggregated = self._aggregate_scenario_results(scenario_results)
                
                exp_result = ExperimentResult(
                    scenario_name=scenario.name,
                    parameters=scenario.parameters,
                    outputs=aggregated,
                    metadata={
                        'n_replications': len(scenario_results),
                        'description': scenario.description
                    }
                )
                results.append(exp_result)
        
        return results
    
    def _apply_policy_parameters(self, base_config: Dict, parameters: Dict) -> Dict:
        """应用政策参数到配置"""
        config = copy.deepcopy(base_config)
        
        # 确保policy部分存在
        if 'policy' not in config:
            config['policy'] = {}
        if 'overrides' not in config['policy']:
            config['policy']['overrides'] = {}
        
        for param_name, value in parameters.items():
            config['policy']['overrides'][param_name] = value
        
        return config
    
    def _aggregate_scenario_results(self, results: List[Dict]) -> Dict:
        """聚合场景结果"""
        aggregated = {}
        
        # 识别数值字段
        first = results[0]
        for key, value in first.items():
            if key == 'history':
                continue
            if isinstance(value, (int, float)):
                values = [r.get(key, 0) for r in results if key in r]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated


class CounterfactualExperiment:
    """
    反事实实验
    
    实现因果推断框架，估计政策处理效应
    
    使用方法：
    ```python
    exp = CounterfactualExperiment()
    
    # 定义事实与反事实情景
    factual = {'grain_subsidies_rate': 350}
    counterfactual = {'grain_subsidies_rate': 0}
    
    # 运行实验
    effects = exp.estimate_treatment_effects(
        run_function, base_config, 
        factual, counterfactual
    )
    ```
    """
    
    def __init__(self, 
                 n_replications: int = 50,
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95):
        """
        初始化反事实实验
        
        Args:
            n_replications: 每个情景的重复次数
            bootstrap_samples: Bootstrap采样次数
            confidence_level: 置信水平
        """
        self.n_replications = n_replications
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def estimate_treatment_effects(self,
                                  run_function: Callable,
                                  base_config: Dict,
                                  factual_params: Dict,
                                  counterfactual_params: Dict,
                                  steps: int = 30,
                                  outcome_metrics: Optional[List[str]] = None) -> Dict[str, TreatmentEffect]:
        """
        估计平均处理效应 (ATE)
        
        ATE = E[Y(1)] - E[Y(0)]
        其中 Y(1) 是有政策时的结果，Y(0) 是无政策时的结果
        
        Args:
            run_function: 模型运行函数
            base_config: 基础配置
            factual_params: 事实情景参数（有政策）
            counterfactual_params: 反事实情景参数（无政策）
            steps: 模拟步数
            outcome_metrics: 要估计的结果指标列表
            
        Returns:
            各指标的处理效应
        """
        print("\n开始反事实实验...")
        
        # 默认结果指标
        if outcome_metrics is None:
            outcome_metrics = [
                'avg_grain_output',
                'avg_farmer_income',
                'final_transfer_rate',
                'final_pure_farmer_count'
            ]
        
        # 1. 运行事实情景
        print(f"  运行事实情景 ({self.n_replications}次)...")
        factual_results = self._run_scenario(
            run_function, base_config, factual_params, steps
        )
        
        # 2. 运行反事实情景
        print(f"  运行反事实情景 ({self.n_replications}次)...")
        counterfactual_results = self._run_scenario(
            run_function, base_config, counterfactual_params, steps
        )
        
        # 3. 估计处理效应
        print("  估计处理效应...")
        effects = {}
        
        for metric in outcome_metrics:
            try:
                effect = self._estimate_single_effect(
                    factual_results, 
                    counterfactual_results,
                    metric
                )
                effects[metric] = effect
            except Exception as e:
                warnings.warn(f"无法估计 {metric} 的处理效应: {e}")
        
        return effects
    
    def _run_scenario(self,
                     run_function: Callable,
                     base_config: Dict,
                     params: Dict,
                     steps: int) -> List[Dict]:
        """运行单个情景"""
        config = copy.deepcopy(base_config)
        
        if 'policy' not in config:
            config['policy'] = {}
        if 'overrides' not in config['policy']:
            config['policy']['overrides'] = {}
        
        for k, v in params.items():
            config['policy']['overrides'][k] = v
        
        results = []
        for seed in range(self.n_replications):
            try:
                result = run_function(config, seed=seed, steps=steps)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Seed {seed} 失败: {e}")
        
        return results
    
    def _estimate_single_effect(self,
                               factual_results: List[Dict],
                               counterfactual_results: List[Dict],
                               metric: str) -> TreatmentEffect:
        """估计单个指标的处理效应"""
        # 提取结果
        y1 = np.array([r.get(metric, 0) for r in factual_results])
        y0 = np.array([r.get(metric, 0) for r in counterfactual_results])
        
        # 计算ATE
        ate = np.mean(y1) - np.mean(y0)
        att = ate  # 在随机实验中 ATT = ATE
        
        # Bootstrap置信区间
        bootstrap_ates = []
        for _ in range(self.bootstrap_samples):
            # 有放回采样
            idx1 = np.random.choice(len(y1), size=len(y1), replace=True)
            idx0 = np.random.choice(len(y0), size=len(y0), replace=True)
            
            boot_ate = np.mean(y1[idx1]) - np.mean(y0[idx0])
            bootstrap_ates.append(boot_ate)
        
        ci_lower = np.percentile(bootstrap_ates, self.alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_ates, (1 - self.alpha / 2) * 100)
        
        # 假设检验 (H0: ATE = 0)
        # 使用t检验
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(y1, y0)
        
        return TreatmentEffect(
            outcome_name=metric,
            ate=ate,
            att=att,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=len(y1),
            n_control=len(y0)
        )
    
    def print_effects(self, effects: Dict[str, TreatmentEffect]):
        """打印处理效应结果"""
        print("\n" + "=" * 70)
        print("反事实分析结果 - 平均处理效应 (ATE)")
        print("=" * 70)
        
        for metric, effect in effects.items():
            sig = "***" if effect.p_value < 0.01 else "**" if effect.p_value < 0.05 else "*" if effect.p_value < 0.1 else ""
            
            print(f"\n【{metric}】")
            print(f"  ATE: {effect.ate:+.4f} {sig}")
            print(f"  95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")
            print(f"  p-value: {effect.p_value:.4f}")
            print(f"  样本量: 处理组 {effect.n_treated}, 对照组 {effect.n_control}")
            
            # 效应解释
            if effect.p_value < 0.05:
                direction = "增加" if effect.ate > 0 else "减少"
                print(f"  解释: 政策显著{direction}了该指标")
            else:
                print(f"  解释: 政策对该指标无显著影响")
        
        print("\n" + "=" * 70)
        print("显著性标记: *** p<0.01, ** p<0.05, * p<0.1")


class PolicyGradientExperiment:
    """
    政策梯度实验
    
    在多目标框架下寻找最优政策组合
    
    使用方法：
    ```python
    exp = PolicyGradientExperiment()
    
    # 网格搜索
    pareto_frontier = exp.grid_search_optimal_policy(
        run_function, base_config
    )
    
    # 可视化
    exp.plot_pareto_frontier(pareto_frontier)
    ```
    """
    
    def __init__(self,
                 objective_weights: Optional[Dict[str, float]] = None):
        """
        初始化政策梯度实验
        
        Args:
            objective_weights: 目标权重 {objective_name: weight}
        """
        self.objective_weights = objective_weights or {
            'grain_output': 0.35,
            'farmer_welfare': 0.35,
            'fiscal_efficiency': 0.30
        }
    
    def grid_search_optimal_policy(self,
                                   run_function: Callable,
                                   base_config: Dict,
                                   subsidy_grid: Optional[np.ndarray] = None,
                                   price_floor_grid: Optional[np.ndarray] = None,
                                   steps: int = 30,
                                   n_replications: int = 5) -> pd.DataFrame:
        """
        网格搜索最优政策组合
        
        Args:
            run_function: 模型运行函数
            base_config: 基础配置
            subsidy_grid: 补贴率网格
            price_floor_grid: 价格支持网格
            steps: 模拟步数
            n_replications: 重复次数
            
        Returns:
            包含所有组合结果的DataFrame
        """
        if subsidy_grid is None:
            subsidy_grid = np.array([0, 100, 200, 300, 400, 500])
        if price_floor_grid is None:
            price_floor_grid = np.array([1.0, 1.1, 1.2, 1.3])
        
        results = []
        total = len(subsidy_grid) * len(price_floor_grid)
        current = 0
        
        print(f"\n开始政策梯度搜索 ({total}个组合)...")
        
        for subsidy in subsidy_grid:
            for price_floor in price_floor_grid:
                current += 1
                print(f"  [{current}/{total}] 补贴={subsidy}元, 价格系数={price_floor}...", end=" ")
                
                # 运行多次并平均
                rep_results = []
                for rep in range(n_replications):
                    config = copy.deepcopy(base_config)
                    if 'policy' not in config:
                        config['policy'] = {}
                    if 'overrides' not in config['policy']:
                        config['policy']['overrides'] = {}
                    
                    config['policy']['overrides']['grain_subsidies_rate'] = subsidy / 350.0
                    config['policy']['overrides']['grain_price_floor'] = price_floor
                    
                    try:
                        result = run_function(config, seed=rep, steps=steps)
                        rep_results.append(result)
                    except:
                        pass
                
                if rep_results:
                    # 计算均值
                    avg_result = {
                        'subsidy_rate': subsidy,
                        'price_floor': price_floor,
                        'grain_output': np.mean([r.get('avg_grain_output', 0) for r in rep_results]),
                        'farmer_income': np.mean([r.get('avg_farmer_income', 0) for r in rep_results]),
                        'subsidy_cost': np.mean([r.get('total_subsidy_cost', 0) for r in rep_results]),
                    }
                    
                    # 计算综合得分
                    avg_result['composite_score'] = self._calculate_composite_score(avg_result)
                    
                    results.append(avg_result)
                    print(f"完成 (得分={avg_result['composite_score']:.3f})")
                else:
                    print("失败")
        
        df = pd.DataFrame(results)
        
        # 标记帕累托最优解
        df['is_pareto_optimal'] = self._identify_pareto_frontier(df)
        
        return df
    
    def _calculate_composite_score(self, result: Dict) -> float:
        """计算综合得分"""
        # 标准化各指标
        grain_score = result['grain_output'] / 3000000  # 假设目标300万kg
        welfare_score = result['farmer_income'] / 50000  # 假设目标5万元
        
        # 财政效率：产出/成本比
        if result['subsidy_cost'] > 0:
            efficiency_score = min(1.0, result['grain_output'] / result['subsidy_cost'] / 10)
        else:
            efficiency_score = 1.0
        
        # 加权平均
        composite = (
            self.objective_weights.get('grain_output', 0.35) * grain_score +
            self.objective_weights.get('farmer_welfare', 0.35) * welfare_score +
            self.objective_weights.get('fiscal_efficiency', 0.30) * efficiency_score
        )
        
        return composite
    
    def _identify_pareto_frontier(self, df: pd.DataFrame) -> List[bool]:
        """识别帕累托最优解"""
        is_optimal = []
        
        for i, row_i in df.iterrows():
            dominated = False
            
            for j, row_j in df.iterrows():
                if i == j:
                    continue
                
                # 检查row_i是否被row_j支配
                # 支配条件：row_j在所有目标上都不差于row_i，且至少一个目标更好
                grain_better = row_j['grain_output'] >= row_i['grain_output']
                income_better = row_j['farmer_income'] >= row_i['farmer_income']
                cost_better = row_j['subsidy_cost'] <= row_i['subsidy_cost']
                
                strictly_better = (
                    row_j['grain_output'] > row_i['grain_output'] or
                    row_j['farmer_income'] > row_i['farmer_income'] or
                    row_j['subsidy_cost'] < row_i['subsidy_cost']
                )
                
                if grain_better and income_better and cost_better and strictly_better:
                    dominated = True
                    break
            
            is_optimal.append(not dominated)
        
        return is_optimal
    
    def plot_pareto_frontier(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """绘制帕累托前沿图"""
        if not HAS_MATPLOTLIB:
            print("未安装matplotlib，无法绘图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 产量 vs 成本
        ax = axes[0, 0]
        pareto = df[df['is_pareto_optimal']]
        non_pareto = df[~df['is_pareto_optimal']]
        
        ax.scatter(non_pareto['subsidy_cost'] / 10000, non_pareto['grain_output'] / 1e6,
                  c='gray', alpha=0.5, label='非最优解')
        ax.scatter(pareto['subsidy_cost'] / 10000, pareto['grain_output'] / 1e6,
                  c='red', s=100, marker='*', label='帕累托最优')
        ax.set_xlabel('补贴成本 (万元)')
        ax.set_ylabel('粮食产量 (百万kg)')
        ax.set_title('粮食产量 vs 财政成本')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 收入 vs 成本
        ax = axes[0, 1]
        ax.scatter(non_pareto['subsidy_cost'] / 10000, non_pareto['farmer_income'],
                  c='gray', alpha=0.5, label='非最优解')
        ax.scatter(pareto['subsidy_cost'] / 10000, pareto['farmer_income'],
                  c='red', s=100, marker='*', label='帕累托最优')
        ax.set_xlabel('补贴成本 (万元)')
        ax.set_ylabel('农民收入 (元)')
        ax.set_title('农民收入 vs 财政成本')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 综合得分热力图
        ax = axes[1, 0]
        pivot = df.pivot_table(
            values='composite_score',
            index='subsidy_rate',
            columns='price_floor',
            aggfunc='mean'
        )
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
        ax.set_yticklabels([f'{int(x)}' for x in pivot.index])
        ax.set_xlabel('价格支持系数')
        ax.set_ylabel('补贴率 (元/亩)')
        ax.set_title('综合得分热力图')
        fig.colorbar(im, ax=ax)
        
        # 4. 三维帕累托前沿（投影）
        ax = axes[1, 1]
        scatter = ax.scatter(
            df['grain_output'] / 1e6,
            df['farmer_income'],
            c=df['subsidy_cost'] / 10000,
            cmap='viridis',
            alpha=0.7
        )
        # 标记帕累托最优
        ax.scatter(
            pareto['grain_output'] / 1e6,
            pareto['farmer_income'],
            c='red', s=100, marker='*', edgecolors='black'
        )
        ax.set_xlabel('粮食产量 (百万kg)')
        ax.set_ylabel('农民收入 (元)')
        ax.set_title('三目标权衡（颜色=成本）')
        fig.colorbar(scatter, ax=ax, label='补贴成本(万元)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {output_path}")
        
        plt.show()
    
    def generate_policy_recommendation(self, df: pd.DataFrame) -> str:
        """生成政策建议"""
        lines = []
        lines.append("=" * 70)
        lines.append("政策优化分析报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 找到帕累托最优解
        pareto = df[df['is_pareto_optimal']].copy()
        
        lines.append(f"【帕累托最优政策组合】(共{len(pareto)}个)")
        lines.append("-" * 50)
        
        for _, row in pareto.iterrows():
            lines.append(f"  补贴率={row['subsidy_rate']:.0f}元/亩, "
                        f"价格系数={row['price_floor']:.2f}")
            lines.append(f"    产量={row['grain_output']/1e6:.2f}百万kg, "
                        f"收入={row['farmer_income']:.0f}元, "
                        f"成本={row['subsidy_cost']/10000:.1f}万元")
        
        # 推荐最优解
        best = df.loc[df['composite_score'].idxmax()]
        lines.append("\n【推荐政策】(综合得分最高)")
        lines.append("-" * 50)
        lines.append(f"  补贴率: {best['subsidy_rate']:.0f} 元/亩")
        lines.append(f"  价格支持系数: {best['price_floor']:.2f}")
        lines.append(f"  预期产量: {best['grain_output']/1e6:.2f} 百万kg")
        lines.append(f"  预期农民收入: {best['farmer_income']:.0f} 元")
        lines.append(f"  预期财政支出: {best['subsidy_cost']/10000:.1f} 万元")
        lines.append(f"  综合得分: {best['composite_score']:.3f}")
        
        return "\n".join(lines)


# ============ 便捷函数 ============

def run_counterfactual_analysis(run_function: Callable,
                               base_config: Dict,
                               treatment_param: str = 'grain_subsidies_rate',
                               treatment_value: float = 350,
                               control_value: float = 0,
                               n_replications: int = 50,
                               steps: int = 30) -> Dict[str, TreatmentEffect]:
    """
    快速运行反事实分析
    
    使用方法：
    ```python
    from experiments.policy_experiment_matrix import run_counterfactual_analysis
    
    effects = run_counterfactual_analysis(
        run_model_function,
        base_config,
        treatment_param='grain_subsidies_rate',
        treatment_value=350,
        control_value=0
    )
    ```
    """
    exp = CounterfactualExperiment(n_replications=n_replications)
    
    effects = exp.estimate_treatment_effects(
        run_function,
        base_config,
        factual_params={treatment_param: treatment_value},
        counterfactual_params={treatment_param: control_value},
        steps=steps
    )
    
    exp.print_effects(effects)
    return effects
