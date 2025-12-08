# -*- coding: utf-8 -*-
"""
@module sensitivity_analysis.py
@description 系统性参数敏感性分析工具
@created 2025-12-05
@author GitHub Copilot

理论依据：
1. 敏感性分析方法 (Saltelli et al., 2008)
   - 局部敏感性分析 (OAT - One-At-a-Time)
   - 全局敏感性分析 (GSA)
   - Sobol敏感性指数

2. ABM模型验证 (Railsback & Grimm, 2019)
   - 参数空间探索
   - 鲁棒性测试
   - 不确定性量化

3. 蒙特卡洛方法
   - 拉丁超立方采样 (LHS)
   - 正交采样设计
   - Bootstrap置信区间

核心功能：
1. 单参数敏感性分析 (OAT)
2. 多参数联合敏感性分析 (Morris方法)
3. 全局敏感性分析 (Sobol指数)
4. 情景组合分析
5. 结果可视化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
import json
from datetime import datetime
import warnings


@dataclass
class ParameterRange:
    """
    参数范围定义
    
    描述一个参数的取值范围和采样方式
    """
    name: str                               # 参数名称
    path: str                               # 参数路径（支持嵌套，如 'market.grain_price'）
    min_value: float                        # 最小值
    max_value: float                        # 最大值
    default_value: Optional[float] = None   # 默认值
    description: str = ""                   # 参数描述
    
    # 采样配置
    num_samples: int = 5                    # 采样点数量
    sampling_type: str = "linear"           # 采样类型：linear, log, custom
    custom_values: Optional[List[float]] = None  # 自定义采样值
    
    # 理论依据
    literature_source: str = ""             # 文献来源
    typical_range: str = ""                 # 实证范围
    
    def get_sample_values(self) -> np.ndarray:
        """获取采样值"""
        if self.custom_values is not None:
            return np.array(self.custom_values)
        
        if self.sampling_type == "log":
            return np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                self.num_samples
            )
        else:  # linear
            return np.linspace(self.min_value, self.max_value, self.num_samples)


@dataclass
class SensitivityResult:
    """
    敏感性分析结果
    """
    parameter_name: str
    output_metric: str
    
    # 基础统计
    elasticity: float = 0.0                 # 弹性系数
    normalized_sensitivity: float = 0.0     # 归一化敏感性
    
    # 详细数据
    parameter_values: List[float] = field(default_factory=list)
    output_values: List[float] = field(default_factory=list)
    
    # 拟合结果
    regression_coefficients: Dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    
    # 统计显著性
    p_value: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class MorrisResult:
    """
    Morris方法敏感性结果
    """
    parameter_name: str
    
    # Morris指标
    mu: float = 0.0                         # 均值效应
    mu_star: float = 0.0                    # 绝对均值效应
    sigma: float = 0.0                      # 标准差（交互效应）
    
    # 分类
    importance: str = "medium"              # low, medium, high
    interaction: str = "linear"             # linear, nonlinear


@dataclass
class SobolResult:
    """
    Sobol全局敏感性结果
    """
    parameter_name: str
    
    # Sobol指数
    S1: float = 0.0                         # 一阶敏感性指数
    ST: float = 0.0                         # 总敏感性指数
    S2: Dict[str, float] = field(default_factory=dict)  # 二阶交互指数
    
    # 置信区间
    S1_conf: Tuple[float, float] = (0.0, 0.0)
    ST_conf: Tuple[float, float] = (0.0, 0.0)


class SensitivityAnalyzer:
    """
    敏感性分析器
    
    提供多种敏感性分析方法的统一接口
    """
    
    def __init__(
        self,
        model_class,
        base_config: Dict[str, Any],
        output_metrics: List[str]
    ):
        """
        初始化分析器
        
        Args:
            model_class: 模型类
            base_config: 基准配置
            output_metrics: 关注的输出指标列表
        """
        self.model_class = model_class
        self.base_config = base_config.copy()
        self.output_metrics = output_metrics
        
        # 预定义参数范围
        self.parameter_ranges: Dict[str, ParameterRange] = {}
        self._define_default_parameters()
        
        # 存储结果
        self.oat_results: Dict[str, List[SensitivityResult]] = {}
        self.morris_results: List[MorrisResult] = []
        self.sobol_results: List[SobolResult] = []
    
    def _define_default_parameters(self):
        """定义默认参数范围（基于中国农业模型）"""
        
        default_params = [
            # 市场参数
            ParameterRange(
                name="粮食基准价格",
                path="market.grain_price",
                min_value=2.0,
                max_value=4.0,
                default_value=2.58,
                description="粮食市场基准价格（元/公斤）",
                literature_source="国家统计局，农产品成本收益调查",
                typical_range="2.2-3.5元/kg"
            ),
            ParameterRange(
                name="经作价格",
                path="market.cash_crop_price",
                min_value=3.0,
                max_value=8.0,
                default_value=5.0,
                description="经济作物基准价格（元/公斤）"
            ),
            ParameterRange(
                name="劳动力工资率",
                path="market.wage_rate",
                min_value=80,
                max_value=200,
                default_value=120,
                description="农业劳动力日工资（元/日）",
                typical_range="100-180元/日"
            ),
            
            # 补贴参数
            ParameterRange(
                name="粮食补贴率",
                path="policy.grain_subsidy_rate",
                min_value=0.0,
                max_value=0.3,
                default_value=0.15,
                description="粮食种植补贴比例"
            ),
            ParameterRange(
                name="每亩补贴",
                path="policy.subsidy_per_mu",
                min_value=50,
                max_value=200,
                default_value=100,
                description="每亩直接补贴金额（元）"
            ),
            
            # 生产参数
            ParameterRange(
                name="基础产量",
                path="production.base_yield",
                min_value=300,
                max_value=600,
                default_value=450,
                description="粮食基础单产（公斤/亩）",
                typical_range="350-550 kg/亩"
            ),
            ParameterRange(
                name="技术进步率",
                path="production.tech_progress_rate",
                min_value=0.01,
                max_value=0.05,
                default_value=0.02,
                description="年技术进步率"
            ),
            
            # 土地流转参数
            ParameterRange(
                name="流转交易成本",
                path="land.transfer_cost",
                min_value=0.05,
                max_value=0.20,
                default_value=0.10,
                description="土地流转交易成本比例"
            ),
            ParameterRange(
                name="熟人折扣",
                path="land.acquaintance_discount",
                min_value=0.70,
                max_value=0.95,
                default_value=0.80,
                description="熟人流转价格折扣"
            ),
            
            # 行为参数
            ParameterRange(
                name="风险厌恶系数",
                path="behavior.risk_aversion",
                min_value=0.3,
                max_value=0.8,
                default_value=0.5,
                description="农户平均风险厌恶程度"
            ),
            ParameterRange(
                name="损失厌恶系数",
                path="behavior.loss_aversion",
                min_value=1.5,
                max_value=3.0,
                default_value=2.25,
                description="前景理论损失厌恶系数",
                literature_source="Kahneman & Tversky (1979)"
            ),
            
            # 代际参数
            ParameterRange(
                name="代际传承概率",
                path="succession.inheritance_prob",
                min_value=0.1,
                max_value=0.5,
                default_value=0.3,
                description="子女继承农业经营概率"
            ),
            ParameterRange(
                name="青年务农意愿",
                path="succession.youth_farming_willingness",
                min_value=0.05,
                max_value=0.25,
                default_value=0.10,
                description="青年主动选择务农概率"
            )
        ]
        
        for param in default_params:
            self.parameter_ranges[param.name] = param
    
    def add_parameter_range(self, param_range: ParameterRange):
        """添加自定义参数范围"""
        self.parameter_ranges[param_range.name] = param_range
    
    def _modify_config(
        self,
        config: Dict[str, Any],
        param_path: str,
        value: float
    ) -> Dict[str, Any]:
        """修改配置中的参数值"""
        import copy
        modified = copy.deepcopy(config)
        
        keys = param_path.split('.')
        target = modified
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        
        return modified
    
    def _run_single_simulation(
        self,
        config: Dict[str, Any],
        years: int = 10,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """运行单次模拟并返回指标值"""
        if seed is not None:
            np.random.seed(seed)
        
        model = self.model_class(config)
        
        for _ in range(years):
            model.step()
        
        # 提取输出指标
        results = {}
        for metric in self.output_metrics:
            results[metric] = self._extract_metric(model, metric)
        
        return results
    
    def _extract_metric(self, model, metric_name: str) -> float:
        """从模型提取指标值"""
        # 尝试从datacollector获取
        if hasattr(model, 'datacollector'):
            try:
                df = model.datacollector.get_model_vars_dataframe()
                if metric_name in df.columns:
                    return df[metric_name].iloc[-1]
            except:
                pass
        
        # 内置指标计算
        if metric_name == 'total_grain_output':
            return sum(
                getattr(a, 'grain_output', 0) 
                for a in model.schedule.agents
            )
        elif metric_name == 'average_income':
            incomes = [
                getattr(a, 'total_income', 0) 
                for a in model.schedule.agents
            ]
            return np.mean(incomes) if incomes else 0
        elif metric_name == 'gini_coefficient':
            incomes = [
                max(0, getattr(a, 'total_income', 0)) 
                for a in model.schedule.agents
            ]
            if not incomes or sum(incomes) == 0:
                return 0
            incomes = np.sort(incomes)
            n = len(incomes)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * incomes) - (n + 1) * np.sum(incomes)) / (n * np.sum(incomes))
        elif metric_name == 'land_transfer_rate':
            total = sum(
                getattr(a, 'owned_land_area', 0) 
                for a in model.schedule.agents
            )
            transferred = sum(
                getattr(a, 'rented_out_area', 0) 
                for a in model.schedule.agents
            )
            return transferred / total if total > 0 else 0
        elif metric_name == 'farmer_count':
            return sum(
                1 for a in model.schedule.agents 
                if hasattr(a, 'is_farming') and a.is_farming
            )
        
        return 0.0
    
    # ============ OAT敏感性分析 ============
    
    def run_oat_analysis(
        self,
        parameter_names: Optional[List[str]] = None,
        years: int = 10,
        replications: int = 3
    ) -> Dict[str, List[SensitivityResult]]:
        """
        单参数敏感性分析 (One-At-a-Time)
        
        每次只改变一个参数，观察输出变化
        
        Args:
            parameter_names: 分析的参数列表（None则分析所有）
            years: 模拟年数
            replications: 每个参数值的重复次数
            
        Returns:
            各参数对各指标的敏感性结果
        """
        if parameter_names is None:
            parameter_names = list(self.parameter_ranges.keys())
        
        results = {}
        
        for param_name in parameter_names:
            if param_name not in self.parameter_ranges:
                warnings.warn(f"未知参数: {param_name}")
                continue
            
            param_range = self.parameter_ranges[param_name]
            sample_values = param_range.get_sample_values()
            
            print(f"分析参数: {param_name}")
            
            metric_results = {metric: [] for metric in self.output_metrics}
            
            for value in sample_values:
                config = self._modify_config(
                    self.base_config, 
                    param_range.path, 
                    value
                )
                
                # 多次重复
                rep_results = {metric: [] for metric in self.output_metrics}
                for rep in range(replications):
                    sim_result = self._run_single_simulation(
                        config, years, seed=42 + rep
                    )
                    for metric in self.output_metrics:
                        rep_results[metric].append(sim_result[metric])
                
                # 计算均值
                for metric in self.output_metrics:
                    metric_results[metric].append(np.mean(rep_results[metric]))
            
            # 计算敏感性指标
            results[param_name] = []
            for metric in self.output_metrics:
                sens_result = self._calculate_oat_sensitivity(
                    param_name, metric,
                    sample_values.tolist(),
                    metric_results[metric],
                    param_range.default_value
                )
                results[param_name].append(sens_result)
        
        self.oat_results = results
        return results
    
    def _calculate_oat_sensitivity(
        self,
        param_name: str,
        metric_name: str,
        param_values: List[float],
        output_values: List[float],
        default_value: Optional[float]
    ) -> SensitivityResult:
        """计算OAT敏感性指标"""
        result = SensitivityResult(
            parameter_name=param_name,
            output_metric=metric_name,
            parameter_values=param_values,
            output_values=output_values
        )
        
        # 计算弹性（在默认值附近）
        if default_value is not None and len(param_values) >= 2:
            # 找到最接近默认值的两个点
            diffs = [abs(v - default_value) for v in param_values]
            idx = np.argmin(diffs)
            
            if idx > 0 and idx < len(param_values) - 1:
                # 中间点：使用前后两点
                delta_p = param_values[idx + 1] - param_values[idx - 1]
                delta_o = output_values[idx + 1] - output_values[idx - 1]
            elif idx == 0:
                delta_p = param_values[1] - param_values[0]
                delta_o = output_values[1] - output_values[0]
            else:
                delta_p = param_values[-1] - param_values[-2]
                delta_o = output_values[-1] - output_values[-2]
            
            if delta_p != 0 and output_values[idx] != 0:
                result.elasticity = (delta_o / output_values[idx]) / (delta_p / param_values[idx])
        
        # 归一化敏感性
        if np.std(param_values) > 0 and np.std(output_values) > 0:
            result.normalized_sensitivity = np.std(output_values) / np.std(param_values)
        
        # 线性回归
        if len(param_values) >= 2:
            coeffs = np.polyfit(param_values, output_values, 1)
            result.regression_coefficients = {'slope': coeffs[0], 'intercept': coeffs[1]}
            
            # R²
            y_pred = np.polyval(coeffs, param_values)
            ss_res = np.sum((np.array(output_values) - y_pred) ** 2)
            ss_tot = np.sum((np.array(output_values) - np.mean(output_values)) ** 2)
            result.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return result
    
    # ============ Morris方法 ============
    
    def run_morris_analysis(
        self,
        parameter_names: Optional[List[str]] = None,
        trajectories: int = 10,
        levels: int = 4,
        years: int = 10
    ) -> List[MorrisResult]:
        """
        Morris方法敏感性分析
        
        高效的全局敏感性筛选方法
        
        Args:
            parameter_names: 分析的参数列表
            trajectories: 轨迹数量
            levels: 参数离散化级别
            years: 模拟年数
            
        Returns:
            Morris分析结果列表
        """
        if parameter_names is None:
            parameter_names = list(self.parameter_ranges.keys())[:6]  # 限制参数数量
        
        k = len(parameter_names)  # 参数数量
        
        # 生成Morris采样矩阵
        samples = self._generate_morris_samples(parameter_names, trajectories, levels)
        
        # 运行模拟
        all_outputs = []
        for sample in samples:
            config = self.base_config.copy()
            for i, param_name in enumerate(parameter_names):
                param_range = self.parameter_ranges[param_name]
                config = self._modify_config(config, param_range.path, sample[i])
            
            result = self._run_single_simulation(config, years)
            all_outputs.append(result[self.output_metrics[0]])  # 使用第一个指标
        
        # 计算Morris指标
        results = self._calculate_morris_indices(
            parameter_names, samples, all_outputs, trajectories, k
        )
        
        self.morris_results = results
        return results
    
    def _generate_morris_samples(
        self,
        parameter_names: List[str],
        trajectories: int,
        levels: int
    ) -> np.ndarray:
        """生成Morris采样矩阵"""
        k = len(parameter_names)
        delta = levels / (2 * (levels - 1))
        
        samples = []
        for _ in range(trajectories):
            # 基础点
            base = np.random.randint(0, levels - 1, k) / (levels - 1)
            trajectory = [base.copy()]
            
            # 随机顺序改变参数
            order = np.random.permutation(k)
            current = base.copy()
            for i in order:
                current[i] += delta if np.random.random() > 0.5 else -delta
                current[i] = np.clip(current[i], 0, 1)
                trajectory.append(current.copy())
            
            samples.extend(trajectory)
        
        # 转换到实际参数范围
        actual_samples = []
        for sample in samples:
            actual = []
            for i, param_name in enumerate(parameter_names):
                param_range = self.parameter_ranges[param_name]
                value = param_range.min_value + sample[i] * (
                    param_range.max_value - param_range.min_value
                )
                actual.append(value)
            actual_samples.append(actual)
        
        return np.array(actual_samples)
    
    def _calculate_morris_indices(
        self,
        parameter_names: List[str],
        samples: np.ndarray,
        outputs: List[float],
        trajectories: int,
        k: int
    ) -> List[MorrisResult]:
        """计算Morris指标"""
        results = []
        
        # 每个参数的elementary effects
        for i, param_name in enumerate(parameter_names):
            elementary_effects = []
            
            for t in range(trajectories):
                base_idx = t * (k + 1)
                for j in range(k):
                    idx1 = base_idx + j
                    idx2 = base_idx + j + 1
                    
                    if idx2 < len(outputs):
                        delta_param = samples[idx2, i] - samples[idx1, i]
                        if abs(delta_param) > 1e-10:  # 参数有变化
                            delta_output = outputs[idx2] - outputs[idx1]
                            ee = delta_output / delta_param
                            elementary_effects.append(ee)
            
            if elementary_effects:
                mu = np.mean(elementary_effects)
                mu_star = np.mean(np.abs(elementary_effects))
                sigma = np.std(elementary_effects)
            else:
                mu = mu_star = sigma = 0
            
            # 判断重要性和交互性
            importance = "high" if mu_star > np.median([r.mu_star for r in results] + [mu_star]) else "medium"
            interaction = "nonlinear" if sigma > 0.5 * mu_star else "linear"
            
            results.append(MorrisResult(
                parameter_name=param_name,
                mu=mu,
                mu_star=mu_star,
                sigma=sigma,
                importance=importance,
                interaction=interaction
            ))
        
        return results
    
    # ============ 情景组合分析 ============
    
    def run_scenario_analysis(
        self,
        scenario_definitions: List[Dict[str, Any]],
        years: int = 10,
        replications: int = 3
    ) -> pd.DataFrame:
        """
        多情景组合分析
        
        Args:
            scenario_definitions: 情景定义列表
            years: 模拟年数
            replications: 重复次数
            
        Returns:
            情景结果DataFrame
        """
        results = []
        
        for scenario in scenario_definitions:
            scenario_name = scenario.get('name', 'unnamed')
            param_changes = scenario.get('parameters', {})
            
            # 应用参数修改
            config = self.base_config.copy()
            for param_path, value in param_changes.items():
                config = self._modify_config(config, param_path, value)
            
            # 运行多次
            metric_values = {metric: [] for metric in self.output_metrics}
            for rep in range(replications):
                sim_result = self._run_single_simulation(config, years, seed=42 + rep)
                for metric in self.output_metrics:
                    metric_values[metric].append(sim_result[metric])
            
            # 汇总结果
            row = {'scenario': scenario_name}
            for metric in self.output_metrics:
                row[f'{metric}_mean'] = np.mean(metric_values[metric])
                row[f'{metric}_std'] = np.std(metric_values[metric])
            results.append(row)
        
        return pd.DataFrame(results)
    
    def generate_factorial_scenarios(
        self,
        parameter_names: List[str],
        levels: int = 3
    ) -> List[Dict[str, Any]]:
        """
        生成全因子设计情景
        
        Args:
            parameter_names: 参数名称列表
            levels: 每个参数的水平数
            
        Returns:
            情景定义列表
        """
        # 获取各参数的水平值
        param_levels = {}
        for name in parameter_names:
            if name in self.parameter_ranges:
                param_range = self.parameter_ranges[name]
                values = np.linspace(
                    param_range.min_value,
                    param_range.max_value,
                    levels
                )
                param_levels[name] = values.tolist()
        
        # 生成所有组合
        scenarios = []
        param_names = list(param_levels.keys())
        for combo in product(*[param_levels[name] for name in param_names]):
            scenario = {
                'name': '_'.join([f'{name}={combo[i]:.2f}' for i, name in enumerate(param_names)]),
                'parameters': {
                    self.parameter_ranges[param_names[i]].path: combo[i]
                    for i in range(len(param_names))
                }
            }
            scenarios.append(scenario)
        
        return scenarios
    
    # ============ 结果可视化与报告 ============
    
    def generate_sensitivity_report(
        self,
        output_path: str
    ) -> str:
        """
        生成敏感性分析报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 70)
        report.append("参数敏感性分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        # OAT结果
        if self.oat_results:
            report.append("【单参数敏感性分析 (OAT)】")
            report.append("-" * 50)
            
            for param_name, sens_results in self.oat_results.items():
                report.append(f"\n参数: {param_name}")
                for sens in sens_results:
                    report.append(f"  对 {sens.output_metric}:")
                    report.append(f"    弹性系数: {sens.elasticity:.4f}")
                    report.append(f"    R²: {sens.r_squared:.4f}")
                    if sens.regression_coefficients:
                        report.append(f"    斜率: {sens.regression_coefficients.get('slope', 0):.4f}")
            report.append("")
        
        # Morris结果
        if self.morris_results:
            report.append("【Morris方法敏感性分析】")
            report.append("-" * 50)
            report.append(f"{'参数':<20} {'μ*':<12} {'σ':<12} {'重要性':<10} {'交互性':<10}")
            
            for result in sorted(self.morris_results, key=lambda x: -x.mu_star):
                report.append(
                    f"{result.parameter_name:<20} "
                    f"{result.mu_star:<12.4f} "
                    f"{result.sigma:<12.4f} "
                    f"{result.importance:<10} "
                    f"{result.interaction:<10}"
                )
            report.append("")
        
        # 关键发现
        report.append("【关键发现】")
        report.append("-" * 50)
        
        if self.oat_results:
            # 找出最敏感的参数
            all_elasticities = []
            for param_name, sens_results in self.oat_results.items():
                for sens in sens_results:
                    all_elasticities.append((param_name, sens.output_metric, abs(sens.elasticity)))
            
            all_elasticities.sort(key=lambda x: -x[2])
            report.append("最敏感参数（按弹性绝对值排序）:")
            for param, metric, elast in all_elasticities[:5]:
                report.append(f"  - {param} 对 {metric}: 弹性 = {elast:.4f}")
        
        report.append("")
        
        # 政策建议
        report.append("【政策建议】")
        report.append("-" * 50)
        report.append("基于敏感性分析结果，以下参数需要重点关注:")
        
        if self.morris_results:
            high_importance = [r for r in self.morris_results if r.importance == "high"]
            for r in high_importance[:3]:
                report.append(f"  - {r.parameter_name} (高重要性)")
        
        report_text = "\n".join(report)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def export_results_to_csv(self, output_dir: str):
        """
        导出结果到CSV
        
        Args:
            output_dir: 输出目录
        """
        import os
        
        # OAT结果
        if self.oat_results:
            oat_data = []
            for param_name, sens_results in self.oat_results.items():
                for sens in sens_results:
                    oat_data.append({
                        'parameter': param_name,
                        'metric': sens.output_metric,
                        'elasticity': sens.elasticity,
                        'normalized_sensitivity': sens.normalized_sensitivity,
                        'r_squared': sens.r_squared
                    })
            pd.DataFrame(oat_data).to_csv(
                os.path.join(output_dir, 'oat_sensitivity.csv'),
                index=False, encoding='utf-8-sig'
            )
        
        # Morris结果
        if self.morris_results:
            morris_data = [{
                'parameter': r.parameter_name,
                'mu': r.mu,
                'mu_star': r.mu_star,
                'sigma': r.sigma,
                'importance': r.importance,
                'interaction': r.interaction
            } for r in self.morris_results]
            pd.DataFrame(morris_data).to_csv(
                os.path.join(output_dir, 'morris_sensitivity.csv'),
                index=False, encoding='utf-8-sig'
            )


def run_sensitivity_analysis_example():
    """
    敏感性分析示例代码
    """
    # 注意：需要导入实际的模型类
    # from core.model import GrainProductionModel
    
    # 示例配置
    base_config = {
        'num_farmers': 100,
        'market': {'grain_price': 2.58},
        'policy': {'grain_subsidy_rate': 0.15}
    }
    
    # 创建分析器
    # analyzer = SensitivityAnalyzer(
    #     GrainProductionModel,
    #     base_config,
    #     output_metrics=['total_grain_output', 'gini_coefficient']
    # )
    
    # 运行OAT分析
    # oat_results = analyzer.run_oat_analysis(
    #     parameter_names=['粮食基准价格', '粮食补贴率'],
    #     years=10,
    #     replications=3
    # )
    
    # 运行Morris分析
    # morris_results = analyzer.run_morris_analysis(
    #     parameter_names=['粮食基准价格', '粮食补贴率', '劳动力工资率'],
    #     trajectories=10
    # )
    
    # 生成报告
    # analyzer.generate_sensitivity_report('sensitivity_report.txt')
    
    print("敏感性分析模块已创建。使用示例见函数注释。")


if __name__ == "__main__":
    run_sensitivity_analysis_example()
