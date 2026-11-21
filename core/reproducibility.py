# -*- coding: utf-8 -*-
"""
可复现性与随机性控制模块
@module: core/reproducibility.py
@author: Agricultural ABM Team
@date: 2026-01-04

@purpose:
1. 统一管理所有随机源（random/numpy/气候序列）
2. 生成可复现的"世界线"（气候冲击序列）
3. 支持配对实验设计（paired design）
4. 提供随机状态检查点保存/恢复

@theory:
仿真随机试验（simulation-based randomized experiment）需要：
- 固定初始条件/参数抽样
- 对同一组"世界线"在不同政策下配对运行
- ATE用配对差分，SE用paired t / bootstrap

@reference:
- Holland (1986) "Statistics and Causal Inference"
- Imbens & Rubin (2015) "Causal Inference for Statistics"
"""

import random
import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ==================== 全局随机控制器 ====================

class RandomController:
    """
    统一随机源控制器
    
    确保所有随机源被同一个主种子控制
    支持子种子派生（防止并行破坏确定性）
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, master_seed: int = 42):
        if self._initialized:
            return
        self._initialized = True
        
        self.master_seed = master_seed
        self.sub_seed_counter = 0
        self._rng_states: Dict[str, Any] = {}
        # @fix 2026-06-01: 记录"外部调用方是否显式指定了种子"。
        # 这是一个一次性(one-shot)凭证：set_seed() 武装它，
        # 紧随其后的 apply_config_seed() 消费它。用于阻止
        # ModelBuilder 用配置文件默认种子(experiment.seed=42)
        # 静默覆盖调用方刚通过 set_seed() 指定的种子。
        self._pending_external_seed: Optional[int] = None
        
        # 初始化所有随机源
        self.reset(master_seed)
    
    def reset(self, seed: Optional[int] = None):
        """
        重置所有随机源到指定种子
        """
        if seed is not None:
            self.master_seed = seed
        
        self.sub_seed_counter = 0
        
        # Python标准库random
        random.seed(self.master_seed)
        
        # NumPy随机
        np.random.seed(self.master_seed)
        
        # 保存初始状态
        self._save_state('initial')
        
        print(f"[RandomController] 所有随机源已重置为种子: {self.master_seed}")
    
    def get_subseed(self, name: str) -> int:
        """
        获取命名子种子（用于并行或模块隔离）
        确保同名子种子在同一主种子下总是相同
        """
        # 基于名称和主种子生成确定性子种子
        hasher = hashlib.md5(f"{self.master_seed}_{name}".encode())
        subseed = int(hasher.hexdigest()[:8], 16) % (2**31)
        return subseed
    
    def _save_state(self, checkpoint_name: str):
        """保存当前随机状态"""
        self._rng_states[checkpoint_name] = {
            'random': random.getstate(),
            'numpy': np.random.get_state()
        }
    
    def _restore_state(self, checkpoint_name: str):
        """恢复到指定检查点"""
        if checkpoint_name not in self._rng_states:
            raise ValueError(f"检查点 {checkpoint_name} 不存在")
        
        state = self._rng_states[checkpoint_name]
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
    
    @classmethod
    def instance(cls) -> 'RandomController':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_seed(cls, seed: int):
        """
        类方法：设置全局种子
        
        这是一个便捷方法，允许直接调用 RandomController.set_seed(42)

        @fix 2026-06-01: 调用此方法被视为"外部调用方显式指定种子"。
        它会武装一个一次性凭证 `_pending_external_seed`，使得紧随其后的
        ModelBuilder 构造（其内部会调用 apply_config_seed）尊重此处的种子，
        而不是用配置文件默认值(experiment.seed=42)将其覆盖。
        """
        inst = cls.instance()
        inst.reset(seed)
        inst._pending_external_seed = int(seed)

    @classmethod
    def apply_config_seed(cls, config_seed: Optional[int]) -> int:
        """
        类方法：在模型构建阶段协调"配置种子"与"外部显式种子"。

        @fix 2026-06-01: 解决"最后写入者赢"反模式。此前 ModelBuilder.__init__
        无条件用 config['experiment']['seed'] 调 init_reproducibility，会把调用方
        刚通过 set_seed() 指定的种子静默回退掉（典型表现：多种子循环全部跑成 seed=42）。

        协调规则：
        - 若存在待消费的外部显式种子(_pending_external_seed)，则以它为准，
          忽略配置默认种子，并消费(清空)该凭证。
        - 否则按配置种子重置（保持旧行为，向后兼容未调用 set_seed 的调用方）。

        返回最终生效的种子。
        """
        inst = cls.instance()
        pending = inst._pending_external_seed
        if pending is not None:
            # 外部显式种子优先；reset 已在 set_seed 时执行，这里仅确保 master_seed 一致
            if inst.master_seed != pending:
                inst.reset(pending)
            inst._pending_external_seed = None  # 一次性消费
            return int(pending)
        # 无外部种子：沿用配置种子（旧行为）
        if config_seed is not None:
            inst.reset(int(config_seed))
            return int(config_seed)
        return int(inst.master_seed)


def _resolve_model_context(context: Any = None) -> Any:
    """Resolve an arbitrary object to the model that owns its RNGs when possible."""
    if context is None:
        return None
    if hasattr(context, 'random') or hasattr(context, 'np_random'):
        return context
    return getattr(context, 'model', None)


def get_python_rng(context: Any = None):
    """Return the model-scoped Python RNG when available, otherwise fall back to global random."""
    model = _resolve_model_context(context)
    rng = getattr(model, 'random', None) if model is not None else None
    if rng is not None and hasattr(rng, 'random'):
        return rng
    return random


def get_numpy_rng(context: Any = None):
    """Return the model-scoped NumPy RNG when available, otherwise fall back to global NumPy RNG."""
    model = _resolve_model_context(context)
    rng = getattr(model, 'np_random', None) if model is not None else None
    if rng is not None:
        return rng
    return np.random


def get_effective_seed(seed: Optional[int] = None) -> int:
    """Resolve an explicit seed or fall back to the controller's master seed."""
    if seed is not None:
        return int(seed)
    return int(RandomController.instance().master_seed)


def deterministic_subseed(seed: Optional[int] = None, stream: Optional[str] = None) -> int:
    """Build a deterministic sub-seed for a named stream without mutating global RNG state."""
    master_seed = get_effective_seed(seed)
    if not stream:
        return master_seed
    hasher = hashlib.md5(f"{master_seed}_{stream}".encode("utf-8"))
    return int(hasher.hexdigest()[:8], 16) % (2**31)


def build_python_rng(seed: Optional[int] = None, stream: Optional[str] = None):
    """Create a standalone Python RNG tied to RandomController semantics."""
    return random.Random(deterministic_subseed(seed, stream))


def build_numpy_rng(seed: Optional[int] = None, stream: Optional[str] = None):
    """Create a standalone NumPy RNG tied to RandomController semantics."""
    return np.random.RandomState(deterministic_subseed(seed, stream))


def rng_random(context: Any = None) -> float:
    return float(get_python_rng(context).random())


def rng_uniform(context: Any, a: float, b: float) -> float:
    return float(get_python_rng(context).uniform(a, b))


def rng_randint(context: Any, a: int, b: int) -> int:
    return int(get_python_rng(context).randint(a, b))


def rng_shuffle(context: Any, items: List[Any]) -> List[Any]:
    get_python_rng(context).shuffle(items)
    return items


def rng_sample(context: Any, population: List[Any], k: int) -> List[Any]:
    return list(get_python_rng(context).sample(list(population), k))


def rng_choice(context: Any, population: List[Any], weights: Optional[List[float]] = None):
    population = list(population)
    if not population:
        raise IndexError("Cannot choose from an empty population")
    rng = get_python_rng(context)
    if weights is None:
        return rng.choice(population)
    return rng.choices(population, weights=weights, k=1)[0]


def np_random_value(context: Any = None) -> float:
    rng = get_numpy_rng(context)
    if hasattr(rng, 'random_sample'):
        return float(rng.random_sample())
    return float(rng.random())


def np_uniform(context: Any, low: float, high: float = None, size=None):
    rng = get_numpy_rng(context)
    return rng.uniform(low, high, size)


def np_normal(context: Any, loc: float = 0.0, scale: float = 1.0, size=None):
    return get_numpy_rng(context).normal(loc, scale, size)


def np_lognormal(context: Any, mean: float = 0.0, sigma: float = 1.0, size=None):
    return get_numpy_rng(context).lognormal(mean, sigma, size)


def np_dirichlet(context: Any, alpha, size=None):
    return get_numpy_rng(context).dirichlet(alpha, size)


def np_randint(context: Any, low: int, high: Optional[int] = None, size=None):
    return get_numpy_rng(context).randint(low, high, size)


def np_choice(context: Any, a, size=None, replace: bool = True, p=None):
    return get_numpy_rng(context).choice(a, size=size, replace=replace, p=p)


def np_beta(context: Any, a: float, b: float, size=None):
    return get_numpy_rng(context).beta(a, b, size)


def np_permutation(context: Any, x):
    return get_numpy_rng(context).permutation(x)


def np_rand(context: Any, size=None):
    rng = get_numpy_rng(context)
    if hasattr(rng, 'random_sample'):
        return rng.random_sample(size)
    return rng.random(size)


# ==================== 世界线生成器 ====================

@dataclass
class WorldLine:
    """
    世界线：固定的外生冲击序列
    
    包含：
    - 气候冲击序列（每年是否发生、强度）
    - 价格冲击序列（市场外生扰动）
    - 政策外生变化（如中央文件）
    
    同一世界线可在不同政策下配对运行
    """
    seed: int
    n_years: int
    
    # 气候序列
    climate_shocks: List[bool] = field(default_factory=list)  # 是否发生冲击
    climate_severities: List[float] = field(default_factory=list)  # 冲击强度
    
    # 价格冲击
    price_shocks: List[float] = field(default_factory=list)  # 价格扰动因子
    
    # 元数据
    created_at: str = ""
    hash_id: str = ""
    
    def __post_init__(self):
        if not self.climate_shocks:
            self._generate()
    
    def _generate(self):
        """生成世界线序列"""
        # 使用专用子种子，确保可复现
        rng = build_numpy_rng(self.seed)
        
        # 气候冲击参数（可从config读取）
        extreme_weather_prob = 0.10  # 10%概率
        max_yield_loss = 0.20  # 最大产量损失20%
        
        self.climate_shocks = []
        self.climate_severities = []
        self.price_shocks = []
        
        for _ in range(self.n_years):
            # 气候冲击
            shock_occurs = rng.random() < extreme_weather_prob
            self.climate_shocks.append(shock_occurs)
            
            if shock_occurs:
                severity = max_yield_loss * (0.5 + rng.random() * 0.5)
            else:
                severity = 0.0
            self.climate_severities.append(severity)
            
            # 价格冲击（正态扰动）
            price_shock = rng.normal(1.0, 0.05)  # 均值1，标准差5%
            self.price_shocks.append(price_shock)
        
        # 生成哈希ID
        self.created_at = f"deterministic_seed_{self.seed}_years_{self.n_years}"
        content = json.dumps({
            'seed': self.seed,
            'n_years': self.n_years,
            'climate': self.climate_shocks,
            'severity': self.climate_severities,
            'price': self.price_shocks
        }, sort_keys=True)
        self.hash_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_year(self, year: int) -> Dict[str, Any]:
        """获取指定年份的外生条件"""
        if year < 0 or year >= self.n_years:
            raise IndexError(f"年份 {year} 超出世界线范围 [0, {self.n_years})")
        
        return {
            'climate_shock': self.climate_shocks[year],
            'climate_severity': self.climate_severities[year],
            'price_shock': self.price_shocks[year]
        }
    
    @property
    def shocks(self) -> Dict[str, List]:
        """返回所有冲击序列（兼容测试和旧接口）"""
        return {
            'climate_shocks': self.climate_shocks,
            'climate_severities': self.climate_severities,
            'price_shocks': self.price_shocks
        }
    
    @classmethod
    def generate(cls, n_years: int, seed: int) -> 'WorldLine':
        """
        类方法：生成指定参数的世界线
        
        Args:
            n_years: 模拟年数
            seed: 随机种子
        
        Returns:
            WorldLine: 生成的世界线对象
        """
        return cls(seed=seed, n_years=n_years)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldLine':
        return cls(**data)
    
    def save(self, path: Path):
        """保存世界线到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'WorldLine':
        """从文件加载世界线"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class WorldLineManager:
    """
    世界线管理器
    
    管理多条世界线，支持配对实验设计
    """
    
    def __init__(self, base_seed: int = 42, n_worldlines: int = 10, n_years: int = 30):
        self.base_seed = base_seed
        self.worldlines: Dict[int, WorldLine] = {}
        self.n_worldlines = n_worldlines
        self._n_years = n_years
        
        # 自动生成世界线
        if n_worldlines > 0 and n_years > 0:
            self.generate_worldlines(n_years, n_worldlines)
    
    def generate_worldlines(self, n_years: int, n_lines: Optional[int] = None):
        """
        生成多条世界线
        
        每条世界线使用不同的子种子，但所有世界线在给定base_seed下可复现
        """
        if n_lines is None:
            n_lines = self.n_worldlines
        
        self.worldlines.clear()
        
        for i in range(n_lines):
            # 派生子种子
            subseed = self.base_seed * 1000 + i
            wl = WorldLine(seed=subseed, n_years=n_years)
            self.worldlines[i] = wl
        
        print(f"[WorldLineManager] 生成 {n_lines} 条世界线，每条 {n_years} 年")
        return self.worldlines
    
    def get_worldline(self, index: int) -> WorldLine:
        """获取指定索引的世界线"""
        if index not in self.worldlines:
            raise KeyError(f"世界线 {index} 不存在")
        return self.worldlines[index]
    
    def save_all(self, directory: Path):
        """保存所有世界线"""
        directory.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'base_seed': self.base_seed,
            'n_worldlines': len(self.worldlines),
            'worldlines': {}
        }
        
        for idx, wl in self.worldlines.items():
            filename = f"worldline_{idx:03d}.json"
            wl.save(directory / filename)
            manifest['worldlines'][idx] = {
                'filename': filename,
                'hash_id': wl.hash_id,
                'n_years': wl.n_years
            }
        
        with open(directory / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[WorldLineManager] 已保存到 {directory}")
    
    @classmethod
    def load_all(cls, directory: Path) -> 'WorldLineManager':
        """从目录加载所有世界线"""
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到 manifest.json: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        manager = cls(
            base_seed=manifest['base_seed'],
            n_worldlines=manifest['n_worldlines']
        )
        
        for idx_str, info in manifest['worldlines'].items():
            idx = int(idx_str)
            wl = WorldLine.load(directory / info['filename'])
            manager.worldlines[idx] = wl
        
        print(f"[WorldLineManager] 从 {directory} 加载 {len(manager.worldlines)} 条世界线")
        return manager


# ==================== 配对实验设计 ====================

@dataclass
class PairedExperimentResult:
    """
    配对实验结果
    """
    worldline_id: int
    treatment_name: str
    control_name: str
    
    # 处理组结果
    treatment_outcomes: Dict[str, float] = field(default_factory=dict)
    # 对照组结果
    control_outcomes: Dict[str, float] = field(default_factory=dict)
    
    # 配对差分
    paired_differences: Dict[str, float] = field(default_factory=dict)
    
    def compute_differences(self):
        """计算配对差分"""
        for key in self.treatment_outcomes:
            if key in self.control_outcomes:
                self.paired_differences[key] = (
                    self.treatment_outcomes[key] - self.control_outcomes[key]
                )


@dataclass
class ATEResult:
    """
    平均处理效应（ATE）结果
    """
    treatment_name: str
    control_name: str
    n_pairs: int
    
    # 各指标的ATE
    ate_estimates: Dict[str, float] = field(default_factory=dict)
    # 标准误
    se_estimates: Dict[str, float] = field(default_factory=dict)
    # 95%置信区间
    ci_lower: Dict[str, float] = field(default_factory=dict)
    ci_upper: Dict[str, float] = field(default_factory=dict)
    # p值（配对t检验）
    p_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dataframe(self):
        """转换为DataFrame"""
        import pandas as pd
        
        rows = []
        for metric in self.ate_estimates:
            rows.append({
                'metric': metric,
                'ATE': self.ate_estimates[metric],
                'SE': self.se_estimates.get(metric, np.nan),
                'CI_lower': self.ci_lower.get(metric, np.nan),
                'CI_upper': self.ci_upper.get(metric, np.nan),
                'p_value': self.p_values.get(metric, np.nan)
            })
        
        return pd.DataFrame(rows)


def compute_ate_paired(
    paired_results: List[PairedExperimentResult],
    method: str = 'paired_t'
) -> ATEResult:
    """
    计算配对实验的ATE
    
    Args:
        paired_results: 配对实验结果列表
        method: 'paired_t' 或 'bootstrap'
    
    Returns:
        ATEResult对象
    """
    from scipy import stats
    
    if not paired_results:
        raise ValueError("需要至少一对配对结果")
    
    # 收集所有差分
    all_diffs: Dict[str, List[float]] = {}
    
    for pr in paired_results:
        for metric, diff in pr.paired_differences.items():
            if metric not in all_diffs:
                all_diffs[metric] = []
            all_diffs[metric].append(diff)
    
    result = ATEResult(
        treatment_name=paired_results[0].treatment_name,
        control_name=paired_results[0].control_name,
        n_pairs=len(paired_results)
    )
    
    for metric, diffs in all_diffs.items():
        diffs_arr = np.array(diffs)
        n = len(diffs_arr)
        
        # ATE = 差分的均值
        ate = np.mean(diffs_arr)
        result.ate_estimates[metric] = ate
        
        if method == 'paired_t':
            # 配对t检验
            se = np.std(diffs_arr, ddof=1) / np.sqrt(n)
            result.se_estimates[metric] = se
            
            # t统计量和p值
            if se > 0:
                t_stat = ate / se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
            else:
                p_val = 1.0 if ate == 0 else 0.0
            result.p_values[metric] = p_val
            
            # 95% CI
            t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
            result.ci_lower[metric] = ate - t_crit * se
            result.ci_upper[metric] = ate + t_crit * se
        
        elif method == 'bootstrap':
            # Bootstrap置信区间
            n_bootstrap = 1000
            bootstrap_ates = []
            
            rng = build_numpy_rng(42, "paired_bootstrap")
            for _ in range(n_bootstrap):
                sample = rng.choice(diffs_arr, size=n, replace=True)
                bootstrap_ates.append(np.mean(sample))
            
            bootstrap_ates = np.array(bootstrap_ates)
            result.se_estimates[metric] = np.std(bootstrap_ates)
            result.ci_lower[metric] = np.percentile(bootstrap_ates, 2.5)
            result.ci_upper[metric] = np.percentile(bootstrap_ates, 97.5)
            
            # Bootstrap p值（双侧）
            p_val = 2 * min(
                np.mean(bootstrap_ates <= 0),
                np.mean(bootstrap_ates >= 0)
            )
            result.p_values[metric] = p_val
    
    return result


# ==================== 复现性验证工具 ====================

def verify_reproducibility(
    run_func,
    config: Dict,
    seed: int = 42,
    n_runs: int = 2,
    tolerance: float = 1e-10
) -> Tuple[bool, Dict]:
    """
    验证模型可复现性
    
    执行同一配置多次，检查输出是否完全一致
    
    Args:
        run_func: 运行函数，签名为 run_func(config, seed) -> Dict[str, float]
        config: 配置字典
        seed: 随机种子
        n_runs: 运行次数
        tolerance: 数值容差
    
    Returns:
        (是否可复现, 详细结果)
    """
    results = []
    
    for i in range(n_runs):
        # 重置随机状态
        RandomController.instance().reset(seed)
        
        # 运行
        output = run_func(config, seed)
        results.append(output)
    
    # 比较结果
    is_reproducible = True
    differences = {}
    
    baseline = results[0]
    for i, result in enumerate(results[1:], start=1):
        for key in baseline:
            if key not in result:
                is_reproducible = False
                differences[f"run_{i}_missing_{key}"] = True
                continue
            
            diff = abs(baseline[key] - result[key])
            if diff > tolerance:
                is_reproducible = False
                differences[f"run_{i}_{key}_diff"] = diff
    
    return is_reproducible, {
        'reproducible': is_reproducible,
        'tolerance': tolerance,
        'n_runs': n_runs,
        'seed': seed,
        'differences': differences,
        'baseline_output': baseline
    }


def generate_output_hash(output: Dict[str, Any]) -> str:
    """
    生成输出的哈希值，用于快速比较
    """
    metadata_keys = {'timestamp', 'created_at', 'generated_at', 'wall_clock_time'}

    # 将浮点数四舍五入到合理精度
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 10)
        elif isinstance(obj, dict):
            return {
                k: round_floats(v)
                for k, v in obj.items()
                if k not in metadata_keys
            }
        elif isinstance(obj, list):
            return [round_floats(item) for item in obj]
        return obj
    
    rounded = round_floats(output)
    content = json.dumps(rounded, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


# ==================== 数据血缘追踪 ====================

@dataclass
class DataLineage:
    """
    数据血缘信息
    
    每份输出携带：输入数据版本、参数版本、代码版本、seed、情景ID、时间戳
    """
    # 基本信息
    scenario_id: str
    seed: int
    timestamp: str
    
    # 版本信息
    input_data_version: str = ""
    parameter_version: str = ""
    code_version: str = ""  # git commit如果有
    
    # 配置摘要
    config_hash: str = ""
    n_farmers: int = 0
    n_enterprises: int = 0
    n_steps: int = 0
    
    # 输入文件路径
    calibration_file: str = ""
    validation_file: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def create(cls, config: Dict, scenario_id: str, seed: int) -> 'DataLineage':
        """从配置创建血缘信息"""
        import subprocess
        
        # 尝试获取git commit
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()[:8]
        except:
            git_hash = "unknown"
        
        # 配置哈希
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return cls(
            scenario_id=scenario_id,
            seed=seed,
            timestamp=f"deterministic_{scenario_id}_seed_{seed}",
            code_version=git_hash,
            config_hash=config_hash,
            n_farmers=config.get('scale', {}).get('n_farmers', 0),
            n_enterprises=config.get('scale', {}).get('n_enterprises', 0),
            n_steps=config.get('experiment', {}).get('steps', 0),
            calibration_file=config.get('initial_conditions', {}).get('calibration_file', ''),
            parameter_version=config.get('experiment', {}).get('name', 'unknown')
        )


# ==================== 初始化函数 ====================

def init_reproducibility(seed: int = 42) -> RandomController:
    """
    初始化可复现性控制
    
    应在每个脚本开头调用
    """
    controller = RandomController(seed)
    controller.reset(seed)
    return controller


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("可复现性模块测试")
    print("=" * 60)
    
    # 测试随机控制器
    ctrl = init_reproducibility(42)
    
    # 测试世界线生成
    manager = WorldLineManager(base_seed=42, n_worldlines=5)
    worldlines = manager.generate_worldlines(n_years=30)
    
    print(f"\n世界线0的前5年气候冲击:")
    for i in range(5):
        year_data = worldlines[0].get_year(i)
        print(f"  年{i}: 冲击={year_data['climate_shock']}, 强度={year_data['climate_severity']:.3f}")
    
    # 验证可复现性
    print("\n验证世界线可复现性...")
    manager2 = WorldLineManager(base_seed=42, n_worldlines=5)
    worldlines2 = manager2.generate_worldlines(n_years=30)
    
    match = all(
        wl1.hash_id == wl2.hash_id 
        for wl1, wl2 in zip(worldlines.values(), worldlines2.values())
    )
    print(f"  世界线哈希一致: {'✅' if match else '❌'}")
    
    print("\n测试完成！")
