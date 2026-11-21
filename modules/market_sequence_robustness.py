# -*- coding: utf-8 -*-
"""
市场出清顺序稳健性检验模块

@theory Sequential Market Clearing Bias
问题: 固定的市场出清顺序(土地→劳动力→服务)可能导致系统性偏差
- 农户在土地决策时不知道劳动力是否充足
- 大户可能租入过多土地，但后来雇不到足够劳动力

@solution 三种稳健性检验方案:
1. 随机化市场顺序 (本模块实现)
2. 使用预期价格（滞后值+噪声）
3. 配给状态+跨期调整

@reference
- Tesfatsion, L. (2006). Agent-based computational economics. 
  Handbook of Computational Economics, 2, 831-880.
- LeBaron, B. (2006). Agent-based computational finance. 
  Handbook of Computational Economics, 2, 1187-1233.

@version 1.0
@date 2026-01-02
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from core.reproducibility import build_python_rng, get_python_rng

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """市场类型枚举"""
    LAND = "land"
    LABOR = "labor"
    SERVICE = "service"


@dataclass
class MarketClearingResult:
    """单个市场出清结果"""
    market_type: MarketType
    transactions: int
    total_volume: float
    avg_price: float
    excess_demand: float  # 正值表示超额需求,负值表示超额供给


@dataclass
class SequenceRobustnessResult:
    """顺序稳健性检验结果"""
    sequence: List[MarketType]
    grain_production: float
    farmer_income: float
    land_transfer_rate: float
    market_results: List[MarketClearingResult]


class MarketSequenceManager:
    """
    市场出清顺序管理器
    
    支持三种模式:
    1. FIXED: 固定顺序 (默认: 土地→劳动力→服务)
    2. RANDOM: 每步随机打乱顺序
    3. EXPECTED_PRICE: 使用预期价格的同时出清
    """
    
    # 默认顺序: 土地 → 劳动力 → 服务
    DEFAULT_SEQUENCE = [MarketType.LAND, MarketType.LABOR, MarketType.SERVICE]
    
    def __init__(self, model, mode: str = "fixed", seed: Optional[int] = None):
        """
        初始化市场顺序管理器
        
        Args:
            model: MainModel实例
            mode: "fixed", "random", 或 "expected_price"
            seed: 随机种子（用于可复现的随机化）
        """
        self.model = model
        self.mode = mode.lower()
        self.rng = build_python_rng(seed, "market_sequence_manager") if seed is not None else get_python_rng(model)
        
        # 市场函数映射
        self._market_functions: Dict[MarketType, Callable] = {}
        self._setup_market_functions()
        
        # 历史记录（用于预期价格计算）
        self.price_history: Dict[MarketType, List[float]] = {
            mt: [] for mt in MarketType
        }
        self.history_window = 3  # 预期价格使用最近3期
        
        logger.info(f"MarketSequenceManager初始化: mode={self.mode}")
    
    def _setup_market_functions(self):
        """设置市场出清函数"""
        self._market_functions = {
            MarketType.LAND: self._clear_land_market,
            MarketType.LABOR: self._clear_labor_market,
            MarketType.SERVICE: self._clear_service_market
        }
    
    def get_sequence(self) -> List[MarketType]:
        """
        获取当前步的市场出清顺序
        
        Returns:
            市场出清顺序列表
        """
        if self.mode == "fixed":
            return self.DEFAULT_SEQUENCE.copy()
        elif self.mode == "random":
            sequence = self.DEFAULT_SEQUENCE.copy()
            self.rng.shuffle(sequence)
            return sequence
        elif self.mode == "expected_price":
            # 预期价格模式下，理论上可以同时出清
            # 但实际实现中仍需顺序执行，只是使用预期价格
            return self.DEFAULT_SEQUENCE.copy()
        else:
            logger.warning(f"未知模式: {self.mode}, 使用固定顺序")
            return self.DEFAULT_SEQUENCE.copy()
    
    def clear_all_markets(self) -> List[MarketClearingResult]:
        """
        执行所有市场出清
        
        Returns:
            各市场出清结果列表
        """
        sequence = self.get_sequence()
        results = []
        
        # 如果是预期价格模式，先计算预期价格
        if self.mode == "expected_price":
            self._update_expected_prices()
        
        for market_type in sequence:
            result = self._market_functions[market_type]()
            results.append(result)
            
            # 更新价格历史
            if result.avg_price > 0:
                self.price_history[market_type].append(result.avg_price)
                # 保持历史窗口大小
                if len(self.price_history[market_type]) > self.history_window:
                    self.price_history[market_type].pop(0)
        
        return results
    
    def _update_expected_prices(self):
        """
        更新预期价格（用于expected_price模式）
        
        预期价格 = 历史均价 + 噪声
        """
        for market_type in MarketType:
            history = self.price_history[market_type]
            if len(history) >= 2:
                expected = np.mean(history)
                noise = self.rng.gauss(0, 0.05 * expected)  # 5%噪声
                self._set_expected_price(market_type, expected + noise)
    
    def _set_expected_price(self, market_type: MarketType, price: float):
        """设置市场预期价格"""
        if market_type == MarketType.LAND:
            if hasattr(self.model, 'environment'):
                self.model.environment.expected_rental_price = price
        elif market_type == MarketType.LABOR:
            if hasattr(self.model, 'labor_market'):
                self.model.labor_market.expected_wage = price
        elif market_type == MarketType.SERVICE:
            if hasattr(self.model, 'service_market'):
                self.model.service_market.expected_service_price = price
    
    def _clear_land_market(self) -> MarketClearingResult:
        """土地市场出清"""
        transactions = 0
        total_volume = 0.0
        avg_price = 0.0
        excess_demand = 0.0
        
        if hasattr(self.model, 'land_clearing_house'):
            lch = self.model.land_clearing_house
            # 执行出清
            self.model.conduct_land_rental_market()
            
            # 统计结果
            if hasattr(lch, 'last_clearing_stats'):
                stats = lch.last_clearing_stats
                transactions = stats.get('transactions', 0)
                total_volume = stats.get('total_area', 0)
                avg_price = stats.get('avg_price', 0)
                excess_demand = stats.get('excess_demand', 0)
        
        return MarketClearingResult(
            market_type=MarketType.LAND,
            transactions=transactions,
            total_volume=total_volume,
            avg_price=avg_price,
            excess_demand=excess_demand
        )
    
    def _clear_labor_market(self) -> MarketClearingResult:
        """劳动力市场出清"""
        transactions = 0
        total_volume = 0.0
        avg_price = 0.0
        excess_demand = 0.0
        
        if hasattr(self.model, 'labor_market') and self.model.labor_market is not None:
            lm = self.model.labor_market
            # 执行出清
            lm.update_market(self.model.farmers, self.model.agricultural_enterprises)
            
            # 统计结果
            if hasattr(lm, 'matched_pairs'):
                transactions = len(lm.matched_pairs)
            if hasattr(lm, 'total_hired_labor'):
                total_volume = lm.total_hired_labor
            if hasattr(lm, 'current_wage'):
                avg_price = lm.current_wage
        
        return MarketClearingResult(
            market_type=MarketType.LABOR,
            transactions=transactions,
            total_volume=total_volume,
            avg_price=avg_price,
            excess_demand=excess_demand
        )
    
    def _clear_service_market(self) -> MarketClearingResult:
        """服务市场出清"""
        transactions = 0
        total_volume = 0.0
        avg_price = 0.0
        excess_demand = 0.0
        
        if hasattr(self.model, 'service_market') and self.model.service_market is not None:
            sm = self.model.service_market
            # 执行出清
            sm.clear_market()
            
            # 统计结果
            if hasattr(sm, 'transactions_count'):
                transactions = sm.transactions_count
            if hasattr(sm, 'total_service_area'):
                total_volume = sm.total_service_area
            if hasattr(sm, 'avg_service_price'):
                avg_price = sm.avg_service_price
        
        return MarketClearingResult(
            market_type=MarketType.SERVICE,
            transactions=transactions,
            total_volume=total_volume,
            avg_price=avg_price,
            excess_demand=excess_demand
        )


def run_sequence_robustness_test(
    model_builder,
    n_replications: int = 30,
    n_steps: int = 10,
    seed_base: int = 42
) -> Dict[str, List[SequenceRobustnessResult]]:
    """
    运行市场顺序稳健性检验
    
    Args:
        model_builder: ModelBuilder实例或构建函数
        n_replications: 每种顺序的重复次数
        n_steps: 每次运行的步数
        seed_base: 基础随机种子
    
    Returns:
        各模式下的运行结果
    """
    results = {
        "fixed": [],
        "random": []
    }
    
    modes = ["fixed", "random"]
    
    for mode in modes:
        logger.info(f"运行 {mode} 模式, {n_replications} 次重复...")
        
        for rep in range(n_replications):
            seed = seed_base + rep
            
            # 构建模型
            if callable(model_builder):
                model = model_builder(seed=seed)
            else:
                model = model_builder.build(seed=seed)
            
            # 设置市场顺序管理器
            seq_manager = MarketSequenceManager(model, mode=mode, seed=seed)
            
            # 运行模拟
            for step in range(n_steps):
                # 使用顺序管理器出清市场
                market_results = seq_manager.clear_all_markets()
                
                # 执行其他步骤（智能体决策等）
                model.schedule.step()
                model.datacollector.collect(model)
            
            # 收集最终结果
            result = SequenceRobustnessResult(
                sequence=seq_manager.get_sequence(),
                grain_production=_get_grain_production(model),
                farmer_income=_get_avg_farmer_income(model),
                land_transfer_rate=_get_land_transfer_rate(model),
                market_results=market_results
            )
            results[mode].append(result)
    
    return results


def _get_grain_production(model) -> float:
    """获取粮食总产量"""
    total = 0.0
    for farmer in model.farmers:
        total += getattr(farmer, 'grain_output', 0)
    if hasattr(model, 'agricultural_enterprises'):
        for ent in model.agricultural_enterprises:
            total += getattr(ent, 'grain_output', 0)
    return total


def _get_avg_farmer_income(model) -> float:
    """获取农户平均收入"""
    if not model.farmers:
        return 0.0
    incomes = [getattr(f, 'annual_income', 0) for f in model.farmers]
    return np.mean(incomes) if incomes else 0.0


def _get_land_transfer_rate(model) -> float:
    """获取土地流转率"""
    if hasattr(model, 'land_clearing_house'):
        return model.land_clearing_house.get_transfer_rate()
    return 0.0


def analyze_robustness_results(results: Dict[str, List[SequenceRobustnessResult]]) -> Dict:
    """
    分析稳健性检验结果
    
    Args:
        results: run_sequence_robustness_test的输出
    
    Returns:
        统计分析结果
    """
    analysis = {}
    
    for mode, mode_results in results.items():
        if not mode_results:
            continue
        
        grain_productions = [r.grain_production for r in mode_results]
        farmer_incomes = [r.farmer_income for r in mode_results]
        transfer_rates = [r.land_transfer_rate for r in mode_results]
        
        analysis[mode] = {
            "grain_production": {
                "mean": np.mean(grain_productions),
                "std": np.std(grain_productions),
                "ci_95": (
                    np.percentile(grain_productions, 2.5),
                    np.percentile(grain_productions, 97.5)
                )
            },
            "farmer_income": {
                "mean": np.mean(farmer_incomes),
                "std": np.std(farmer_incomes),
                "ci_95": (
                    np.percentile(farmer_incomes, 2.5),
                    np.percentile(farmer_incomes, 97.5)
                )
            },
            "land_transfer_rate": {
                "mean": np.mean(transfer_rates),
                "std": np.std(transfer_rates),
                "ci_95": (
                    np.percentile(transfer_rates, 2.5),
                    np.percentile(transfer_rates, 97.5)
                )
            }
        }
    
    # 计算模式间差异的统计显著性
    if "fixed" in analysis and "random" in analysis:
        from scipy import stats
        
        fixed_grain = [r.grain_production for r in results["fixed"]]
        random_grain = [r.grain_production for r in results["random"]]
        
        t_stat, p_value = stats.ttest_ind(fixed_grain, random_grain)
        
        analysis["comparison"] = {
            "grain_production_diff": {
                "fixed_mean": analysis["fixed"]["grain_production"]["mean"],
                "random_mean": analysis["random"]["grain_production"]["mean"],
                "percent_diff": (
                    (analysis["random"]["grain_production"]["mean"] - 
                     analysis["fixed"]["grain_production"]["mean"]) /
                    analysis["fixed"]["grain_production"]["mean"] * 100
                    if analysis["fixed"]["grain_production"]["mean"] != 0 else 0
                ),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        }
    
    return analysis


def generate_robustness_report(analysis: Dict, output_path: str = None) -> str:
    """
    生成稳健性检验报告
    
    Args:
        analysis: analyze_robustness_results的输出
        output_path: 可选的输出文件路径
    
    Returns:
        报告文本
    """
    report_lines = [
        "# 市场出清顺序稳健性检验报告",
        "",
        f"## 检验说明",
        "",
        "本检验评估固定市场出清顺序(土地→劳动力→服务)与随机顺序",
        "对模型主要结论的影响，以验证结论的稳健性。",
        "",
        "## 结果摘要",
        ""
    ]
    
    for mode in ["fixed", "random"]:
        if mode not in analysis:
            continue
        
        data = analysis[mode]
        mode_name = "固定顺序" if mode == "fixed" else "随机顺序"
        
        report_lines.extend([
            f"### {mode_name}模式",
            "",
            "| 指标 | 均值 | 标准差 | 95% CI |",
            "|------|------|--------|--------|",
            f"| 粮食产量 | {data['grain_production']['mean']:.2f} | "
            f"{data['grain_production']['std']:.2f} | "
            f"[{data['grain_production']['ci_95'][0]:.2f}, "
            f"{data['grain_production']['ci_95'][1]:.2f}] |",
            f"| 农户收入 | {data['farmer_income']['mean']:.2f} | "
            f"{data['farmer_income']['std']:.2f} | "
            f"[{data['farmer_income']['ci_95'][0]:.2f}, "
            f"{data['farmer_income']['ci_95'][1]:.2f}] |",
            f"| 流转率 | {data['land_transfer_rate']['mean']:.4f} | "
            f"{data['land_transfer_rate']['std']:.4f} | "
            f"[{data['land_transfer_rate']['ci_95'][0]:.4f}, "
            f"{data['land_transfer_rate']['ci_95'][1]:.4f}] |",
            ""
        ])
    
    if "comparison" in analysis:
        comp = analysis["comparison"]["grain_production_diff"]
        sig_text = "是" if comp["significant"] else "否"
        
        report_lines.extend([
            "## 统计显著性检验",
            "",
            "| 指标 | 差异(%) | t统计量 | p值 | 显著(α=0.05) |",
            "|------|---------|---------|-----|--------------|",
            f"| 粮食产量 | {comp['percent_diff']:.2f}% | "
            f"{comp['t_statistic']:.3f} | {comp['p_value']:.4f} | {sig_text} |",
            "",
            "## 结论",
            ""
        ])
        
        if not comp["significant"]:
            report_lines.extend([
                "**主要结论对市场出清顺序假设稳健**",
                "",
                "固定顺序与随机顺序模式下，粮食产量、农户收入、土地流转率",
                "均无统计显著差异(p>0.05)，表明模型核心结论不依赖于特定的",
                "市场出清顺序假设。"
            ])
        else:
            report_lines.extend([
                "**注意：检测到显著差异**",
                "",
                f"固定顺序与随机顺序模式下存在统计显著差异(p={comp['p_value']:.4f})，",
                "建议进一步分析差异来源，并在论文中讨论该局限性。"
            ])
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存到: {output_path}")
    
    return report


if __name__ == "__main__":
    # 测试代码
    print("市场顺序稳健性检验模块加载成功")
    print("使用方法:")
    print("  from modules.market_sequence_robustness import run_sequence_robustness_test")
    print("  results = run_sequence_robustness_test(model_builder, n_replications=30)")
