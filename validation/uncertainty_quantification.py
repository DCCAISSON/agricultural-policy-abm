# -*- coding: utf-8 -*-
"""
@module uncertainty_quantification.py
@description 不确定性量化框架 - 蒙特卡洛分析与置信区间计算
@created 2025-12-06
@author GitHub Copilot

理论参考：
1. Kennedy & O'Hagan (2001) "Bayesian calibration of computer models"
2. Saltelli et al. (2008) "Global Sensitivity Analysis"
3. Grimm et al. (2020) "The ODD protocol second update"

核心功能：
1. 蒙特卡洛不确定性分析
2. 不确定性来源分解
3. 置信区间可视化
4. 参数扰动鲁棒性检验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import warnings

# 可选导入
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
class UncertaintyResults:
    """不确定性分析结果"""
    n_replications: int
    metrics: Dict[str, Dict[str, float]]  # {metric_name: {mean, std, ci_lower, ci_upper, cv}}
    trajectories: Dict[str, np.ndarray] = None  # 时间序列轨迹
    variance_decomposition: Dict[str, float] = None  # 方差分解结果


class UncertaintyQuantification:
    """
    不确定性量化框架
    
    提供蒙特卡洛分析、置信区间计算和不确定性来源分解功能
    
    使用方法：
    ```python
    uq = UncertaintyQuantification(config_path='config/master_config.yaml')
    results = uq.run_monte_carlo_analysis(n_replications=100, steps=50)
    uq.plot_uncertainty_bands(results)
    ```
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 n_jobs: int = -1,
                 confidence_level: float = 0.95):
        """
        初始化不确定性量化器
        
        Args:
            config_path: 配置文件路径
            n_jobs: 并行作业数 (-1表示使用所有核心)
            confidence_level: 置信水平
        """
        self.config_path = config_path
        self.n_jobs = n_jobs
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def run_monte_carlo_analysis(self, 
                                  run_function: Callable,
                                  n_replications: int = 100,
                                  steps: int = 50,
                                  base_config: Optional[Dict] = None,
                                  verbose: bool = True) -> UncertaintyResults:
        """
        蒙特卡洛不确定性分析
        
        Args:
            run_function: 模型运行函数 f(config, seed) -> results
            n_replications: 重复运行次数
            steps: 每次运行的步数
            base_config: 基础配置
            verbose: 是否打印进度
            
        Returns:
            不确定性分析结果
        """
        if verbose:
            print(f"\n开始蒙特卡洛分析: {n_replications}次重复...")
        
        results_list = []
        
        if HAS_JOBLIB and self.n_jobs != 1:
            # 并行运行
            def run_single(seed):
                try:
                    result = run_function(base_config, seed=seed, steps=steps)
                    return result
                except Exception as e:
                    warnings.warn(f"Seed {seed} 运行失败: {e}")
                    return None
            
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(run_single)(seed) 
                for seed in range(n_replications)
            )
            results_list = [r for r in results_list if r is not None]
        else:
            # 串行运行
            for seed in range(n_replications):
                try:
                    result = run_function(base_config, seed=seed, steps=steps)
                    results_list.append(result)
                    
                    if verbose and (seed + 1) % 10 == 0:
                        print(f"  完成 {seed + 1}/{n_replications}")
                except Exception as e:
                    warnings.warn(f"Seed {seed} 运行失败: {e}")
        
        if not results_list:
            raise ValueError("所有运行都失败了")
        
        # 聚合结果
        aggregated = self._aggregate_results(results_list)
        
        if verbose:
            print(f"完成! 成功运行: {len(results_list)}/{n_replications}")
        
        return aggregated
    
    def _aggregate_results(self, results_list: List[Dict]) -> UncertaintyResults:
        """聚合多次运行结果"""
        n = len(results_list)
        
        # 识别所有数值指标
        first_result = results_list[0]
        numeric_keys = []
        trajectory_keys = []
        
        for key, value in first_result.items():
            if key == 'history':
                continue
            if isinstance(value, (int, float)):
                numeric_keys.append(key)
        
        # 聚合数值指标
        metrics = {}
        for key in numeric_keys:
            values = [r.get(key, 0) for r in results_list if key in r]
            if values:
                values = np.array(values)
                ci_lower = np.percentile(values, self.alpha / 2 * 100)
                ci_upper = np.percentile(values, (1 - self.alpha / 2) * 100)
                
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_samples': len(values)
                }
        
        # 聚合时间序列轨迹
        trajectories = {}
        if 'history' in first_result and isinstance(first_result['history'], dict):
            for ts_key in first_result['history'].keys():
                try:
                    all_trajectories = []
                    for r in results_list:
                        if 'history' in r and ts_key in r['history']:
                            ts = r['history'][ts_key]
                            if ts:
                                all_trajectories.append(ts)
                    
                    if all_trajectories:
                        # 确保长度一致
                        min_len = min(len(t) for t in all_trajectories)
                        aligned = [t[:min_len] for t in all_trajectories]
                        trajectories[ts_key] = np.array(aligned)
                except Exception as e:
                    warnings.warn(f"无法聚合时间序列 {ts_key}: {e}")
        
        return UncertaintyResults(
            n_replications=n,
            metrics=metrics,
            trajectories=trajectories if trajectories else None
        )
    
    def decompose_uncertainty_sources(self, 
                                      run_function: Callable,
                                      base_config: Dict,
                                      n_replications: int = 30,
                                      steps: int = 50,
                                      param_perturbations: Optional[Dict] = None) -> Dict:
        """
        分解不确定性来源
        
        将总方差分解为：
        1. 随机性引起的方差（固定参数，改变种子）
        2. 参数不确定性引起的方差（扰动参数，固定种子）
        
        Args:
            run_function: 模型运行函数
            base_config: 基础配置
            n_replications: 重复次数
            steps: 模拟步数
            param_perturbations: 参数扰动定义 {param_name: perturbation_range}
            
        Returns:
            方差分解结果
        """
        print("\n开始不确定性来源分解...")
        
        # 1. 估计随机性方差（固定参数，改变种子）
        print("  阶段1: 估计随机性方差...")
        random_results = []
        for seed in range(n_replications):
            try:
                result = run_function(base_config, seed=seed, steps=steps)
                random_results.append(result.get('avg_grain_output', 0))
            except:
                pass
        
        random_variance = np.var(random_results) if random_results else 0
        
        # 2. 估计参数不确定性方差
        print("  阶段2: 估计参数不确定性方差...")
        if param_perturbations is None:
            param_perturbations = {
                'grain_subsidies_rate': 0.3,  # ±30%扰动
                'grain_price_base': 0.2,
                'urbanization_rate': 0.2,
            }
        
        param_results = []
        fixed_seed = 42
        
        for i in range(n_replications):
            try:
                # 创建扰动配置
                perturbed_config = self._perturb_config(base_config, param_perturbations, i)
                result = run_function(perturbed_config, seed=fixed_seed, steps=steps)
                param_results.append(result.get('avg_grain_output', 0))
            except:
                pass
        
        param_variance = np.var(param_results) if param_results else 0
        
        # 3. 估计总方差
        print("  阶段3: 估计总方差...")
        total_results = []
        for seed in range(n_replications):
            try:
                perturbed_config = self._perturb_config(base_config, param_perturbations, seed)
                result = run_function(perturbed_config, seed=seed, steps=steps)
                total_results.append(result.get('avg_grain_output', 0))
            except:
                pass
        
        total_variance = np.var(total_results) if total_results else 0
        
        # 计算比例
        decomposition = {
            'total_variance': total_variance,
            'random_variance': random_variance,
            'parameter_variance': param_variance,
            'random_fraction': random_variance / total_variance if total_variance > 0 else 0,
            'parameter_fraction': param_variance / total_variance if total_variance > 0 else 0,
            'interaction_residual': max(0, total_variance - random_variance - param_variance),
            'n_samples': {
                'random': len(random_results),
                'parameter': len(param_results),
                'total': len(total_results)
            }
        }
        
        print(f"\n不确定性分解结果:")
        print(f"  总方差: {total_variance:.2e}")
        print(f"  随机性方差: {random_variance:.2e} ({decomposition['random_fraction']:.1%})")
        print(f"  参数方差: {param_variance:.2e} ({decomposition['parameter_fraction']:.1%})")
        
        return decomposition
    
    def _perturb_config(self, config: Dict, perturbations: Dict, index: int) -> Dict:
        """对配置进行随机扰动"""
        import copy
        perturbed = copy.deepcopy(config)
        
        np.random.seed(index)
        
        for param_path, pct in perturbations.items():
            # 解析参数路径（支持嵌套）
            parts = param_path.split('.')
            
            # 找到参数位置
            current = perturbed
            for part in parts[:-1]:
                if part in current:
                    current = current[part]
                else:
                    break
            
            # 扰动参数
            final_key = parts[-1]
            if final_key in current:
                original = current[final_key]
                if isinstance(original, (int, float)):
                    perturbation = np.random.uniform(-pct, pct)
                    current[final_key] = original * (1 + perturbation)
        
        return perturbed
    
    def plot_uncertainty_bands(self, 
                               results: UncertaintyResults,
                               metric_key: str = 'grain_output',
                               output_path: Optional[str] = None):
        """
        绘制不确定性区间图
        
        Args:
            results: 不确定性分析结果
            metric_key: 要绘制的指标
            output_path: 输出路径
        """
        if not HAS_MATPLOTLIB:
            print("未安装matplotlib，无法绘图")
            return
        
        if results.trajectories is None or metric_key not in results.trajectories:
            print(f"无时间序列数据: {metric_key}")
            return
        
        trajectories = results.trajectories[metric_key]
        n_steps = trajectories.shape[1]
        steps = np.arange(1, n_steps + 1)
        
        # 计算统计量
        mean_traj = np.mean(trajectories, axis=0)
        std_traj = np.std(trajectories, axis=0)
        ci_lower = np.percentile(trajectories, self.alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(trajectories, (1 - self.alpha / 2) * 100, axis=0)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制部分轨迹（淡化）
        n_show = min(20, len(trajectories))
        for traj in trajectories[:n_show]:
            ax.plot(steps, traj, alpha=0.1, color='gray', linewidth=0.5)
        
        # 绘制均值
        ax.plot(steps, mean_traj, color='blue', linewidth=2, label='均值')
        
        # 绘制标准差区间
        ax.fill_between(steps, 
                       mean_traj - std_traj, 
                       mean_traj + std_traj,
                       alpha=0.3, color='blue', label='±1标准差')
        
        # 绘制置信区间
        ax.fill_between(steps, ci_lower, ci_upper,
                       alpha=0.15, color='blue', 
                       label=f'{self.confidence_level*100:.0f}%置信区间')
        
        ax.set_xlabel('时间步 (年)')
        ax.set_ylabel(metric_key)
        ax.set_title(f'{metric_key} 不确定性分析\n({results.n_replications}次蒙特卡洛重复)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {output_path}")
        
        plt.show()
    
    def generate_report(self, results: UncertaintyResults) -> str:
        """生成不确定性分析报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("不确定性量化分析报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"蒙特卡洛重复次数: {results.n_replications}")
        lines.append(f"置信水平: {self.confidence_level*100:.0f}%")
        lines.append("")
        
        lines.append("【数值指标统计】")
        lines.append("-" * 50)
        
        for metric_name, stats in results.metrics.items():
            lines.append(f"\n{metric_name}:")
            lines.append(f"  均值: {stats['mean']:.4f}")
            lines.append(f"  标准差: {stats['std']:.4f}")
            lines.append(f"  变异系数(CV): {stats['cv']:.2%}")
            lines.append(f"  {self.confidence_level*100:.0f}%置信区间: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            lines.append(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if results.variance_decomposition:
            lines.append("\n" + "=" * 70)
            lines.append("【不确定性来源分解】")
            lines.append("-" * 50)
            vd = results.variance_decomposition
            lines.append(f"总方差: {vd['total_variance']:.4e}")
            lines.append(f"随机性方差: {vd['random_variance']:.4e} ({vd['random_fraction']:.1%})")
            lines.append(f"参数方差: {vd['parameter_variance']:.4e} ({vd['parameter_fraction']:.1%})")
        
        return "\n".join(lines)


# ============ 便捷函数 ============

def quick_uncertainty_analysis(config_path: str,
                               run_function: Callable,
                               n_replications: int = 50,
                               steps: int = 30) -> UncertaintyResults:
    """
    快速执行不确定性分析
    
    使用方法：
    ```python
    from validation.uncertainty_quantification import quick_uncertainty_analysis
    
    def run_model(config, seed, steps):
        # ... 运行模型并返回结果字典
        return results
    
    results = quick_uncertainty_analysis(
        'config/master_config.yaml',
        run_model,
        n_replications=50
    )
    ```
    """
    uq = UncertaintyQuantification(config_path=config_path)
    results = uq.run_monte_carlo_analysis(
        run_function=run_function,
        n_replications=n_replications,
        steps=steps
    )
    return results
