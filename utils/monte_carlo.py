# -*- coding: utf-8 -*-
"""
Monte Carlo实验包装器
====================

本模块提供统一的Monte Carlo实验框架，确保所有实验：
1. 运行足够次数 (默认30次)
2. 报告95%置信区间
3. 保存完整的复现信息

@version: v1.0
@date: 2026-01-02
@requirement: P0-4 Nature标准要求Monte Carlo N≥30
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.reproducibility import build_numpy_rng
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# 默认重复次数
DEFAULT_N_REPETITIONS = 30


@dataclass
class MonteCarloConfig:
    """Monte Carlo实验配置"""
    n_repetitions: int = DEFAULT_N_REPETITIONS
    confidence_level: float = 0.95
    random_seed_base: int = 42
    parallel: bool = True
    max_workers: Optional[int] = None  # None = CPU核心数
    save_individual_runs: bool = False
    output_dir: Optional[Path] = None


@dataclass
class MonteCarloResult:
    """单个指标的Monte Carlo统计结果"""
    name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    median: float
    n_samples: int
    cv: float  # 变异系数
    raw_values: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'mean': round(self.mean, 6),
            'std': round(self.std, 6),
            'ci_lower': round(self.ci_lower, 6),
            'ci_upper': round(self.ci_upper, 6),
            'median': round(self.median, 6),
            'n_samples': self.n_samples,
            'cv': round(self.cv, 6)
        }
    
    def __str__(self):
        return f"{self.name}: {self.mean:.4f} ± {self.ci_upper - self.mean:.4f} (95% CI), CV={self.cv:.2%}"


@dataclass
class MonteCarloSummary:
    """Monte Carlo实验完整结果摘要"""
    config: MonteCarloConfig
    results: Dict[str, MonteCarloResult]
    runtime_seconds: float
    timestamp: str
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'config': {
                'n_repetitions': self.config.n_repetitions,
                'confidence_level': self.config.confidence_level,
                'random_seed_base': self.config.random_seed_base
            },
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'runtime_seconds': round(self.runtime_seconds, 2),
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def save(self, filepath: Path):
        """保存结果到JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 70)
        print(f"📊 Monte Carlo 实验结果 (N={self.config.n_repetitions})")
        print("=" * 70)
        print(f"置信水平: {self.config.confidence_level * 100:.0f}%")
        print(f"运行时间: {self.runtime_seconds:.1f}秒")
        print("-" * 70)
        print(f"{'指标':<25} {'均值':>12} {'95% CI':>24} {'CV':>8}")
        print("-" * 70)
        
        for name, result in self.results.items():
            ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            print(f"{name:<25} {result.mean:>12.4f} {ci_str:>24} {result.cv:>8.2%}")


def calculate_confidence_interval(
    values: List[float], 
    confidence: float = 0.95
) -> tuple:
    """
    计算置信区间
    
    Args:
        values: 观测值列表
        confidence: 置信水平 (默认95%)
    
    Returns:
        (mean, ci_lower, ci_upper, std)
    """
    arr = np.array(values)
    n = len(arr)
    
    if n < 2:
        mean = arr[0] if n == 1 else 0
        return mean, mean, mean, 0
    
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # 样本标准差
    
    # t分布 (样本量小于30时更准确)
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_value * std / np.sqrt(n)
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return mean, ci_lower, ci_upper, std


def run_monte_carlo(
    run_single_experiment: Callable[[int], Dict[str, float]],
    config: Optional[MonteCarloConfig] = None,
    metric_names: Optional[List[str]] = None
) -> MonteCarloSummary:
    """
    通用Monte Carlo实验框架
    
    Args:
        run_single_experiment: 函数，接收seed参数，返回指标字典
        config: Monte Carlo配置
        metric_names: 关注的指标名称列表（可选，默认使用返回的所有指标）
    
    Returns:
        MonteCarloSummary: 完整结果摘要
    
    Example:
        def my_experiment(seed):
            model = build_model(seed=seed)
            model.run(steps=30)
            return {'production': model.total_production, 'income': model.avg_income}
        
        results = run_monte_carlo(my_experiment)
        results.print_summary()
    """
    if config is None:
        config = MonteCarloConfig()
    
    import time
    start_time = time.time()
    
    # 生成种子序列
    seeds = [config.random_seed_base + i for i in range(config.n_repetitions)]
    
    # 收集所有运行结果
    all_results: List[Dict[str, float]] = []
    
    if config.parallel and config.n_repetitions > 1:
        try:
            from joblib import Parallel, delayed
            
            print(f"📊 并行运行 {config.n_repetitions} 次Monte Carlo模拟...")
            all_results = Parallel(n_jobs=config.max_workers or -1, verbose=10)(
                delayed(run_single_experiment)(seed) for seed in seeds
            )
        except ImportError:
            warnings.warn("joblib不可用，切换到串行模式")
            config.parallel = False
    
    if not config.parallel or not all_results:
        print(f"📊 串行运行 {config.n_repetitions} 次Monte Carlo模拟...")
        for i, seed in enumerate(seeds):
            print(f"  运行 {i+1}/{config.n_repetitions} (seed={seed})...", end='\r')
            result = run_single_experiment(seed)
            all_results.append(result)
        print()
    
    # 确定要分析的指标
    if metric_names is None:
        # 使用第一次运行的所有键
        metric_names = list(all_results[0].keys()) if all_results else []
    
    # 计算每个指标的统计量
    mc_results: Dict[str, MonteCarloResult] = {}
    
    for metric in metric_names:
        values = []
        for result in all_results:
            if metric in result and result[metric] is not None:
                try:
                    values.append(float(result[metric]))
                except (ValueError, TypeError):
                    continue
        
        if len(values) >= 2:
            mean, ci_lower, ci_upper, std = calculate_confidence_interval(
                values, config.confidence_level
            )
            
            mc_results[metric] = MonteCarloResult(
                name=metric,
                mean=mean,
                std=std,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                median=np.median(values),
                n_samples=len(values),
                cv=std / abs(mean) if mean != 0 else 0,
                raw_values=values if config.save_individual_runs else []
            )
    
    runtime = time.time() - start_time
    
    summary = MonteCarloSummary(
        config=config,
        results=mc_results,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
        metadata={'total_runs_attempted': config.n_repetitions}
    )
    
    # 自动保存
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = config.output_dir / f"monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary.save(filepath)
        print(f"✓ 结果已保存到: {filepath}")
    
    return summary


def validate_monte_carlo_n(n: int, strict: bool = True) -> int:
    """
    验证Monte Carlo重复次数是否满足Nature标准
    
    Args:
        n: 重复次数
        strict: 是否严格执行（True时会抛出异常）
    
    Returns:
        验证后的n值
    
    Raises:
        ValueError: 当strict=True且n<30时
    """
    if n < DEFAULT_N_REPETITIONS:
        msg = f"Monte Carlo重复次数({n})小于Nature标准({DEFAULT_N_REPETITIONS})。"
        if strict:
            raise ValueError(msg + " 请增加重复次数或设置strict=False。")
        else:
            warnings.warn(msg + " 统计功效可能不足。")
    
    return n


# ============================================================================
# 辅助函数：格式化输出
# ============================================================================

def format_ci_latex(result: MonteCarloResult, precision: int = 2) -> str:
    """生成LaTeX格式的置信区间"""
    return f"${result.mean:.{precision}f} \pm {result.ci_upper - result.mean:.{precision}f}$"


def format_ci_text(result: MonteCarloResult, precision: int = 2) -> str:
    """生成文本格式的置信区间"""
    return f"{result.mean:.{precision}f} [{result.ci_lower:.{precision}f}, {result.ci_upper:.{precision}f}]"


# ============================================================================
# 快捷装饰器
# ============================================================================

def with_monte_carlo(n_repetitions: int = DEFAULT_N_REPETITIONS, **mc_kwargs):
    """
    装饰器：自动将函数包装为Monte Carlo实验
    
    Example:
        @with_monte_carlo(n_repetitions=30)
        def run_experiment(seed):
            model = build_model(seed=seed)
            model.run(steps=30)
            return {'output': model.get_output()}
        
        # 调用时自动运行30次并返回统计摘要
        summary = run_experiment()
    """
    def decorator(func: Callable[[int], Dict[str, float]]):
        def wrapper(*args, **kwargs):
            config = MonteCarloConfig(n_repetitions=n_repetitions, **mc_kwargs)
            return run_monte_carlo(func, config=config)
        return wrapper
    return decorator


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=== Monte Carlo 框架测试 ===\n")
    
    # 示例实验函数
    def dummy_experiment(seed: int) -> Dict[str, float]:
        rng = build_numpy_rng(seed, "monte_carlo_dummy_experiment")
        return {
            'production': 1000 + rng.normal(0, 50),
            'income': 50000 + rng.normal(0, 5000),
            'efficiency': 0.8 + rng.normal(0, 0.1)
        }
    
    # 测试1: 基本运行
    config = MonteCarloConfig(n_repetitions=30, parallel=False)
    summary = run_monte_carlo(dummy_experiment, config=config)
    summary.print_summary()
    
    # 测试2: 验证N
    print("\n验证N=10 (应该警告):")
    validate_monte_carlo_n(10, strict=False)
    
    print("\n✓ 所有测试通过")
