# -*- coding: utf-8 -*-
"""
@module agents.village_reputation
@new 2026-05-02

村集体声誉学习系统（Reputation Belief System）。

这个模块解决的核心问题：
    项目原 VillageCollective 仅靠 integration_threshold / acceptance_threshold
    等阈值规则做撮合决策，没有任何"声誉信念"或学习机制。这违反中国村集体
    作为"制度执行学习节点"的现实角色——真实的村干部在长年撮合中，会逐步
    形成对每户/每个企业的可信度判断，影响后续撮合优先级。

设计理念：
    - 对每个交易对手（农户ID或企业ID）维护一个 Beta(α, β) 信念
    - 履约成功 → α += 1；履约失败 → β += 1
    - 由 expected_reliability = α/(α+β) 给出"履约可信度"
    - 每年末 belief 向先验回归（decay），防止历史锁死
    - matching_priority_boost 对撮合分数做修正，可信者优先匹配

通过 mechanism_switches.use_reputation_village = True 启用。

@theory:
    - Bayesian belief update over discrete outcomes (Gelman et al., 2013)
    - Trust dynamics in repeated games (Kreps et al., 1982)
    - Reputation systems on platforms (Resnick et al., 2000)
    - 中国农村社会资本与"熟人圈"效应（Fei Xiaotong, 1948; Yan Yunxiang, 2003）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReputationBelief:
    """对单一交易对手（农户或企业）的 Beta 信念。

    α = 履约成功观察数 + 1（先验）
    β = 履约失败观察数 + 1（先验）
    使得 Beta(1, 1) = Uniform[0, 1] 作为中性先验。
    """
    alpha: float = 1.0
    beta: float = 1.0
    n_observations: int = 0

    def expected_reliability(self) -> float:
        """E[X] = α / (α + β)。"""
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.5
        return self.alpha / denom

    def variance(self) -> float:
        """Var[X] = αβ / [(α+β)² (α+β+1)]，用于不确定性量化。"""
        a, b = self.alpha, self.beta
        s = a + b
        if s <= 0:
            return 0.0
        return (a * b) / ((s ** 2) * (s + 1.0))

    def credibility_interval_95(self) -> tuple[float, float]:
        """简化的 95% credibility interval（基于正态近似），仅用于审计。"""
        mu = self.expected_reliability()
        var = self.variance()
        sd = var ** 0.5
        lo = max(0.0, mu - 1.96 * sd)
        hi = min(1.0, mu + 1.96 * sd)
        return (lo, hi)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """履约更新。

        Args:
            success: True 表示对手履约（合同执行良好），False 表示违约
            weight: 更新强度（默认 1.0）。可用于区分"重要合同"vs"小合同"
        """
        if weight <= 0:
            return
        if success:
            self.alpha += float(weight)
        else:
            self.beta += float(weight)
        self.n_observations += 1


class ReputationBeliefSystem:
    """
    村集体维护的对所有交易对手的信念字典。

    使用方式：
        sys = ReputationBeliefSystem()
        sys.record_outcome(agent_id=42, success=True)
        score = sys.reputation_score(42)
        boost = sys.matching_priority_boost(42)  # 用于修正撮合优先级
        sys.annual_decay()  # 每年末调用一次
    """

    def __init__(self, decay_rate: float = 0.95, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Args:
            decay_rate: 每年末对所有 belief 的衰减强度。
                        decay_rate=0.95 意味着 95% 历史信息保留，5% 回归先验。
                        decay_rate=1.0 表示历史信息永久保留（不推荐）。
                        decay_rate=0.0 表示完全遗忘（每年重置）。
            prior_alpha, prior_beta: Beta 分布的先验参数，默认 (1, 1) = Uniform。
                                     若想偏向"高信任先验"可设为 (3, 1)。
        """
        self.beliefs: Dict[Any, ReputationBelief] = {}
        self.decay_rate = float(decay_rate)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        # 审计：跨年累计观察事件
        self._total_outcomes_recorded = 0
        self._total_successes = 0

    def get_or_create(self, agent_id: Any) -> ReputationBelief:
        """根据 agent_id 取或创建 belief。"""
        if agent_id not in self.beliefs:
            self.beliefs[agent_id] = ReputationBelief(
                alpha=self.prior_alpha,
                beta=self.prior_beta,
            )
        return self.beliefs[agent_id]

    def record_outcome(self, agent_id: Any, success: bool, weight: float = 1.0) -> None:
        """记录一次对手的履约结果。"""
        belief = self.get_or_create(agent_id)
        belief.update(success, weight=weight)
        self._total_outcomes_recorded += 1
        if success:
            self._total_successes += 1

    def annual_decay(self) -> None:
        """
        每年末调用：所有 belief 向先验回归，防止历史信息永久锁死。

        decay 公式：
            new_alpha = prior_alpha + (alpha - prior_alpha) × decay_rate
            new_beta  = prior_beta  + (beta  - prior_beta)  × decay_rate

        这意味着即使一个农户连续 10 年履约良好，30 年后他的 belief
        也会有所"重置"（除非持续履约），避免"过度依赖历史"。
        """
        if self.decay_rate >= 1.0:
            return  # 不衰减
        for belief in self.beliefs.values():
            belief.alpha = self.prior_alpha + (belief.alpha - self.prior_alpha) * self.decay_rate
            belief.beta = self.prior_beta + (belief.beta - self.prior_beta) * self.decay_rate

    def reputation_score(self, agent_id: Any) -> float:
        """[0, 1]，0=完全不可信，1=完全可信。未观察过的对手返回 0.5。"""
        if agent_id not in self.beliefs:
            return 0.5
        return self.beliefs[agent_id].expected_reliability()

    def matching_priority_boost(self, agent_id: Any, max_boost: float = 0.3) -> float:
        """
        返回撮合优先级修正量，[-max_boost, +max_boost]。

        score = 0.5 → boost = 0
        score = 1.0 → boost = +max_boost
        score = 0.0 → boost = -max_boost (远离均值)
        """
        score = self.reputation_score(agent_id)
        deviation = (score - 0.5) * 2.0  # 平移到 [-1, 1]
        return float(deviation * max_boost)

    def confidence_weight(self, agent_id: Any) -> float:
        """
        返回信念的"确信度"，[0, 1]。
        基于 (α + β) 的对数标度，观察次数越多，confidence 越高。

        新对手（n=0）→ 0.0
        中等观察（n=10）→ ~0.5
        长期观察（n=100）→ ~1.0
        """
        if agent_id not in self.beliefs:
            return 0.0
        belief = self.beliefs[agent_id]
        n = belief.n_observations
        if n <= 0:
            return 0.0
        # 对数饱和：log(1+n) / log(1+100)
        import math
        return min(1.0, math.log(1 + n) / math.log(101))

    def summary_statistics(self) -> Dict[str, Any]:
        """用于审计：返回当前系统的全局统计。"""
        if not self.beliefs:
            return {
                'n_partners': 0,
                'total_outcomes_recorded': 0,
                'total_successes': 0,
                'global_success_rate': float('nan'),
                'mean_reputation_score': float('nan'),
            }
        scores = [b.expected_reliability() for b in self.beliefs.values()]
        return {
            'n_partners': len(self.beliefs),
            'total_outcomes_recorded': self._total_outcomes_recorded,
            'total_successes': self._total_successes,
            'global_success_rate': (
                self._total_successes / self._total_outcomes_recorded
                if self._total_outcomes_recorded > 0 else float('nan')
            ),
            'mean_reputation_score': float(sum(scores) / len(scores)),
            'min_reputation_score': float(min(scores)),
            'max_reputation_score': float(max(scores)),
        }

    def export_top_partners(self, n_top: int = 10, ascending: bool = False) -> List[Dict[str, Any]]:
        """
        导出 top-N 对手列表，用于诊断分析。

        Args:
            n_top: 返回的对手数
            ascending: True 返回最差 N 个，False 返回最好 N 个
        """
        items = []
        for agent_id, belief in self.beliefs.items():
            items.append({
                'agent_id': agent_id,
                'reputation_score': belief.expected_reliability(),
                'n_observations': belief.n_observations,
                'alpha': belief.alpha,
                'beta': belief.beta,
                'ci_95': belief.credibility_interval_95(),
            })
        items.sort(key=lambda x: x['reputation_score'], reverse=not ascending)
        return items[:n_top]


__all__ = [
    'ReputationBelief',
    'ReputationBeliefSystem',
]
