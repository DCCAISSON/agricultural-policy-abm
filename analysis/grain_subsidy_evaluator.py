# -*- coding: utf-8 -*-
"""
@module grain_subsidy_evaluator.py
@description 粮食补贴政策评估器 - 聚焦单一政策在不同SSP情景下的效果
@created 2025-12-06
@author GitHub Copilot

研究聚焦：
1. 核心政策：粮食生产补贴（不同补贴率）
2. 社会情景：SSP1-5（共享社会经济路径）
3. 评估目标：粮食安全、农民福利、财政效率

理论依据：
1. SSP框架 (O'Neill et al., 2014; 2017)
   - SSP1: 可持续发展（绿色发展）
   - SSP2: 中间道路（现状延续）
   - SSP3: 区域竞争（碎片化发展）
   - SSP4: 不平等（贫富分化）
   - SSP5: 化石燃料发展（高经济增长）

2. 粮食补贴政策文献
   - Huang et al. (2011): 中国粮食补贴政策效果
   - Gale (2013): 中国农业补贴的演变
   - Yi et al. (2015): 补贴对粮食生产的影响

3. 政策评估方法
   - 反事实分析
   - 情景比较
   - 敏感性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import copy


@dataclass
class SSPScenario:
    """
    SSP社会发展情景定义
    
    基于IPCC SSP框架，结合中国农业特点设定参数
    
    参考：
    - O'Neill et al. (2017). "The roads ahead: Narratives for SSPs"
    - Jiang & O'Neill (2017). "Global urbanization projections for SSPs"
    - Chen et al. (2020). "SSPs中国化应用"
    """
    name: str                               # 情景名称
    code: str                               # 情景代码 (SSP1-5)
    description: str                        # 情景描述
    
    # 人口与城镇化参数
    urbanization_rate: float = 0.65         # 城镇化率
    rural_population_decline_rate: float = 0.02  # 农村人口年下降率
    youth_farming_willingness: float = 0.10  # 青年务农意愿
    
    # 经济发展参数
    gdp_growth_rate: float = 0.05           # GDP增长率
    non_farm_wage_growth: float = 0.05      # 非农工资增长率
    agricultural_tech_progress: float = 0.02  # 农业技术进步率
    
    # 土地市场参数
    land_transfer_friction: float = 0.10    # 土地流转摩擦系数
    scale_operation_trend: float = 0.02     # 规模化趋势
    
    # 政策环境参数
    fiscal_capacity: float = 1.0            # 财政能力系数
    policy_effectiveness: float = 1.0       # 政策执行效率


@dataclass
class SubsidyPolicy:
    """
    粮食补贴政策定义
    
    聚焦粮食生产补贴这一核心政策变量
    """
    name: str                               # 政策名称
    subsidy_rate: float                     # 补贴率（元/亩）
    description: str = ""                   # 政策描述
    
    # 补贴发放方式
    payment_timing: str = "post_harvest"    # 发放时机：pre_planting/post_harvest
    payment_lag: int = 0                    # 支付滞后（期）
    
    # 补贴条件
    min_grain_ratio: float = 0.0            # 最低种粮比例要求
    scale_bonus: float = 0.0                # 规模经营加成
    
    # 财政预算约束
    budget_cap: Optional[float] = None      # 预算上限


@dataclass
class EvaluationMetrics:
    """
    评估指标结果
    """
    # 粮食安全指标
    total_grain_output: float = 0.0         # 粮食总产量（吨）
    grain_planting_area: float = 0.0        # 种粮面积（亩）
    grain_yield_per_mu: float = 0.0         # 单产（公斤/亩）
    grain_share_in_cropping: float = 0.0    # 种粮面积占比
    
    # 农民福利指标
    average_farmer_income: float = 0.0      # 平均农户收入（元）
    income_gini: float = 0.0                # 收入基尼系数
    subsidy_share_in_income: float = 0.0    # 补贴占收入比例
    pure_farmer_income: float = 0.0         # 纯农户平均收入
    
    # 结构变化指标
    land_transfer_rate: float = 0.0         # 土地流转率
    scale_operation_share: float = 0.0      # 规模经营面积占比
    farmer_count: int = 0                   # 务农人数
    youth_farmer_share: float = 0.0         # 青年农民占比
    
    # 财政效率指标
    total_subsidy_cost: float = 0.0         # 补贴总支出
    subsidy_per_ton_grain: float = 0.0      # 每吨粮食补贴成本
    fiscal_efficiency: float = 0.0          # 财政效率（产出/投入）


# ============ SSP情景预设 ============

SSP_SCENARIOS = {
    'SSP1': SSPScenario(
        name="可持续发展路径",
        code="SSP1",
        description="绿色发展、城乡融合、技术进步快、政策效率高",
        urbanization_rate=0.75,
        rural_population_decline_rate=0.025,
        youth_farming_willingness=0.15,  # 返乡创业增加
        gdp_growth_rate=0.045,
        non_farm_wage_growth=0.04,
        agricultural_tech_progress=0.03,  # 技术进步快
        land_transfer_friction=0.05,      # 流转顺畅
        scale_operation_trend=0.03,
        fiscal_capacity=1.2,
        policy_effectiveness=1.2
    ),
    
    'SSP2': SSPScenario(
        name="中间道路",
        code="SSP2",
        description="现状延续、渐进式发展、政策维持现状",
        urbanization_rate=0.70,
        rural_population_decline_rate=0.02,
        youth_farming_willingness=0.10,
        gdp_growth_rate=0.05,
        non_farm_wage_growth=0.05,
        agricultural_tech_progress=0.02,
        land_transfer_friction=0.10,
        scale_operation_trend=0.02,
        fiscal_capacity=1.0,
        policy_effectiveness=1.0
    ),
    
    'SSP3': SSPScenario(
        name="区域竞争路径",
        code="SSP3",
        description="发展碎片化、城乡分割、政策效率低",
        urbanization_rate=0.60,
        rural_population_decline_rate=0.015,
        youth_farming_willingness=0.05,  # 青年流失严重
        gdp_growth_rate=0.03,
        non_farm_wage_growth=0.03,
        agricultural_tech_progress=0.01,  # 技术进步慢
        land_transfer_friction=0.20,      # 流转困难
        scale_operation_trend=0.01,
        fiscal_capacity=0.8,
        policy_effectiveness=0.7
    ),
    
    'SSP4': SSPScenario(
        name="不平等路径",
        code="SSP4",
        description="贫富分化、大资本进入、小农边缘化",
        urbanization_rate=0.72,
        rural_population_decline_rate=0.03,  # 农村人口快速流失
        youth_farming_willingness=0.08,
        gdp_growth_rate=0.04,
        non_farm_wage_growth=0.06,  # 非农工资涨得快
        agricultural_tech_progress=0.025,
        land_transfer_friction=0.08,  # 流转市场发达但不平等
        scale_operation_trend=0.04,   # 规模化快但集中
        fiscal_capacity=1.1,
        policy_effectiveness=0.85  # 政策偏向大户
    ),
    
    'SSP5': SSPScenario(
        name="高增长路径",
        code="SSP5",
        description="高经济增长、高城镇化、农业快速转型",
        urbanization_rate=0.80,
        rural_population_decline_rate=0.035,
        youth_farming_willingness=0.05,
        gdp_growth_rate=0.065,
        non_farm_wage_growth=0.07,  # 非农工资涨得最快
        agricultural_tech_progress=0.03,
        land_transfer_friction=0.05,
        scale_operation_trend=0.05,  # 规模化最快
        fiscal_capacity=1.5,
        policy_effectiveness=1.1
    )
}


# ============ 补贴政策预设 ============

SUBSIDY_POLICIES = {
    'no_subsidy': SubsidyPolicy(
        name="零补贴",
        subsidy_rate=0.0,
        description="取消所有粮食补贴"
    ),
    
    'low_subsidy': SubsidyPolicy(
        name="低补贴",
        subsidy_rate=100.0,
        description="每亩100元基础补贴"
    ),
    
    'baseline_subsidy': SubsidyPolicy(
        name="基准补贴",
        subsidy_rate=200.0,
        description="每亩200元基准补贴（接近当前水平）"
    ),
    
    'medium_subsidy': SubsidyPolicy(
        name="中等补贴",
        subsidy_rate=350.0,
        description="每亩350元中等补贴"
    ),
    
    'high_subsidy': SubsidyPolicy(
        name="高补贴",
        subsidy_rate=500.0,
        description="每亩500元高补贴"
    ),
    
    'very_high_subsidy': SubsidyPolicy(
        name="极高补贴",
        subsidy_rate=800.0,
        description="每亩800元极高补贴（接近种粮成本）"
    ),
    
    'scale_bonus': SubsidyPolicy(
        name="规模加成补贴",
        subsidy_rate=200.0,
        scale_bonus=0.5,  # 规模经营加成50%
        description="基础200元+规模经营加成50%"
    )
}


class GrainSubsidyEvaluator:
    """
    粮食补贴政策评估器
    
    聚焦评估不同补贴率在不同SSP情景下的政策效果
    
    使用方法：
    1. 初始化评估器
    2. 选择SSP情景和补贴政策组合
    3. 运行评估
    4. 分析结果
    """
    
    def __init__(self, model_class, base_config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            model_class: 模型类
            base_config: 基准配置
        """
        self.model_class = model_class
        self.base_config = copy.deepcopy(base_config)
        
        # 可用情景和政策
        self.ssp_scenarios = SSP_SCENARIOS
        self.subsidy_policies = SUBSIDY_POLICIES
        
        # 存储结果
        self.results: Dict[str, Dict[str, EvaluationMetrics]] = {}
    
    def apply_ssp_scenario(
        self,
        config: Dict[str, Any],
        ssp: SSPScenario
    ) -> Dict[str, Any]:
        """
        将SSP情景参数应用到配置
        
        Args:
            config: 基础配置
            ssp: SSP情景
            
        Returns:
            修改后的配置
        """
        modified = copy.deepcopy(config)
        
        # 更新人口相关参数
        if 'agents' in modified and 'farmer' in modified['agents']:
            farmer_cfg = modified['agents']['farmer']
            
            # 青年务农意愿
            if 'succession' not in modified:
                modified['succession'] = {}
            modified['succession']['youth_farming_willingness'] = ssp.youth_farming_willingness
        
        # 更新经济参数
        if 'environment' not in modified:
            modified['environment'] = {}
        if 'market' not in modified['environment']:
            modified['environment']['market'] = {}
        
        # 非农工资增长
        modified['environment']['market']['wage_growth_rate'] = ssp.non_farm_wage_growth
        
        # 更新生产参数
        if 'production' not in modified:
            modified['production'] = {}
        modified['production']['tech_progress_rate'] = ssp.agricultural_tech_progress
        
        # 更新土地市场参数
        if 'land' not in modified['environment']:
            modified['environment']['land'] = {}
        modified['environment']['land']['transfer_friction'] = ssp.land_transfer_friction
        
        # 更新政策效率
        if 'policy' not in modified:
            modified['policy'] = {}
        modified['policy']['fiscal_capacity'] = ssp.fiscal_capacity
        modified['policy']['effectiveness'] = ssp.policy_effectiveness
        
        return modified
    
    def apply_subsidy_policy(
        self,
        config: Dict[str, Any],
        policy: SubsidyPolicy,
        ssp: Optional[SSPScenario] = None
    ) -> Dict[str, Any]:
        """
        将补贴政策应用到配置
        
        Args:
            config: 配置
            policy: 补贴政策
            ssp: SSP情景（用于调整政策效率）
            
        Returns:
            修改后的配置
        """
        modified = copy.deepcopy(config)
        
        # 设置补贴率
        if 'policy' not in modified:
            modified['policy'] = {}
        
        # 基础补贴率
        base_rate = policy.subsidy_rate
        
        # 如果有SSP情景，根据财政能力和政策效率调整
        if ssp is not None:
            effective_rate = base_rate * ssp.policy_effectiveness
        else:
            effective_rate = base_rate
        
        modified['policy']['grain_subsidy_per_mu'] = effective_rate
        modified['policy']['subsidy_payment_timing'] = policy.payment_timing
        modified['policy']['subsidy_payment_lag'] = policy.payment_lag
        modified['policy']['min_grain_ratio'] = policy.min_grain_ratio
        modified['policy']['scale_bonus'] = policy.scale_bonus
        
        if policy.budget_cap is not None:
            if ssp is not None:
                # 预算上限根据财政能力调整
                modified['policy']['subsidy_budget_cap'] = policy.budget_cap * ssp.fiscal_capacity
            else:
                modified['policy']['subsidy_budget_cap'] = policy.budget_cap
        
        return modified
    
    def run_single_evaluation(
        self,
        ssp_code: str,
        policy_name: str,
        years: int = 10,
        replications: int = 3
    ) -> EvaluationMetrics:
        """
        运行单个SSP+政策组合的评估
        
        Args:
            ssp_code: SSP代码 (SSP1-5)
            policy_name: 政策名称
            years: 模拟年数
            replications: 重复次数
            
        Returns:
            评估指标
        """
        ssp = self.ssp_scenarios.get(ssp_code)
        policy = self.subsidy_policies.get(policy_name)
        
        if ssp is None:
            raise ValueError(f"未知SSP情景: {ssp_code}")
        if policy is None:
            raise ValueError(f"未知补贴政策: {policy_name}")
        
        # 应用配置
        config = self.apply_ssp_scenario(self.base_config, ssp)
        config = self.apply_subsidy_policy(config, policy, ssp)
        
        # 运行多次模拟
        all_metrics = []
        for rep in range(replications):
            metrics = self._run_simulation(config, years, seed=42 + rep)
            all_metrics.append(metrics)
        
        # 平均结果
        return self._aggregate_metrics(all_metrics)
    
    def run_full_evaluation(
        self,
        years: int = 10,
        replications: int = 3,
        ssp_codes: Optional[List[str]] = None,
        policy_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        运行完整的情景-政策矩阵评估
        
        Args:
            years: 模拟年数
            replications: 重复次数
            ssp_codes: 要评估的SSP代码列表（None则全部）
            policy_names: 要评估的政策列表（None则全部）
            
        Returns:
            结果DataFrame
        """
        if ssp_codes is None:
            ssp_codes = list(self.ssp_scenarios.keys())
        if policy_names is None:
            policy_names = list(self.subsidy_policies.keys())
        
        results = []
        total = len(ssp_codes) * len(policy_names)
        count = 0
        
        for ssp_code in ssp_codes:
            for policy_name in policy_names:
                count += 1
                print(f"[{count}/{total}] 评估 {ssp_code} + {policy_name}")
                
                try:
                    metrics = self.run_single_evaluation(
                        ssp_code, policy_name, years, replications
                    )
                    
                    row = {
                        'SSP情景': ssp_code,
                        'SSP描述': self.ssp_scenarios[ssp_code].description,
                        '补贴政策': policy_name,
                        '补贴率(元/亩)': self.subsidy_policies[policy_name].subsidy_rate,
                        '粮食产量(吨)': metrics.total_grain_output,
                        '种粮面积(亩)': metrics.grain_planting_area,
                        '单产(kg/亩)': metrics.grain_yield_per_mu,
                        '种粮占比': metrics.grain_share_in_cropping,
                        '农户平均收入': metrics.average_farmer_income,
                        '收入基尼系数': metrics.income_gini,
                        '补贴占收入比': metrics.subsidy_share_in_income,
                        '土地流转率': metrics.land_transfer_rate,
                        '规模经营占比': metrics.scale_operation_share,
                        '务农人数': metrics.farmer_count,
                        '青年农民占比': metrics.youth_farmer_share,
                        '补贴总支出': metrics.total_subsidy_cost,
                        '每吨粮食补贴': metrics.subsidy_per_ton_grain,
                        '财政效率': metrics.fiscal_efficiency
                    }
                    results.append(row)
                    
                    # 存储到results字典
                    if ssp_code not in self.results:
                        self.results[ssp_code] = {}
                    self.results[ssp_code][policy_name] = metrics
                    
                except Exception as e:
                    print(f"  评估失败: {e}")
        
        return pd.DataFrame(results)
    
    def _run_simulation(
        self,
        config: Dict[str, Any],
        years: int,
        seed: int
    ) -> EvaluationMetrics:
        """运行单次模拟"""
        np.random.seed(seed)
        
        model = self.model_class(config)
        
        for _ in range(years):
            model.step()
        
        return self._extract_metrics(model)
    
    def _extract_metrics(self, model) -> EvaluationMetrics:
        """从模型提取评估指标"""
        metrics = EvaluationMetrics()
        
        # 粮食安全指标
        total_grain = 0
        total_grain_area = 0
        total_crop_area = 0
        
        for agent in model.schedule.agents:
            if hasattr(agent, 'grain_output'):
                total_grain += agent.grain_output
            if hasattr(agent, 'grain_area'):
                total_grain_area += agent.grain_area
            if hasattr(agent, 'get_operated_land_area'):
                total_crop_area += agent.get_operated_land_area()
        
        metrics.total_grain_output = total_grain / 1000  # 转为吨
        metrics.grain_planting_area = total_grain_area
        metrics.grain_yield_per_mu = total_grain / max(1, total_grain_area)
        metrics.grain_share_in_cropping = total_grain_area / max(1, total_crop_area)
        
        # 农民福利指标
        incomes = []
        pure_farmer_incomes = []
        subsidies = []
        
        for agent in model.schedule.agents:
            if hasattr(agent, 'total_income'):
                income = agent.total_income
                incomes.append(income)
                
                if hasattr(agent, 'livelihood_strategy'):
                    if 'PURE' in str(agent.livelihood_strategy):
                        pure_farmer_incomes.append(income)
                
                if hasattr(agent, 'subsidy_received'):
                    subsidies.append(agent.subsidy_received)
        
        if incomes:
            metrics.average_farmer_income = np.mean(incomes)
            metrics.income_gini = self._calculate_gini(incomes)
        
        if pure_farmer_incomes:
            metrics.pure_farmer_income = np.mean(pure_farmer_incomes)
        
        total_subsidy = sum(subsidies)
        metrics.total_subsidy_cost = total_subsidy
        if incomes:
            metrics.subsidy_share_in_income = total_subsidy / (sum(incomes) + 1)
        
        # 结构变化指标
        total_owned = 0
        total_transferred = 0
        scale_area = 0
        farmer_count = 0
        youth_count = 0
        
        for agent in model.schedule.agents:
            if hasattr(agent, 'owned_land_area'):
                total_owned += agent.owned_land_area
            if hasattr(agent, 'rented_out_area'):
                total_transferred += agent.rented_out_area
            if hasattr(agent, 'get_operated_land_area'):
                area = agent.get_operated_land_area()
                if area > 30:  # 规模经营阈值
                    scale_area += area
            if hasattr(agent, 'is_farming') and agent.is_farming:
                farmer_count += 1
                if hasattr(agent, 'age') and agent.age < 45:
                    youth_count += 1
        
        metrics.land_transfer_rate = total_transferred / max(1, total_owned)
        metrics.scale_operation_share = scale_area / max(1, total_crop_area)
        metrics.farmer_count = farmer_count
        metrics.youth_farmer_share = youth_count / max(1, farmer_count)
        
        # 财政效率指标
        if metrics.total_grain_output > 0:
            metrics.subsidy_per_ton_grain = total_subsidy / metrics.total_grain_output
        if total_subsidy > 0:
            metrics.fiscal_efficiency = metrics.total_grain_output * 1000 / total_subsidy  # kg/元
        
        return metrics
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or len(values) < 2:
            return 0
        values = np.sort([max(0, v) for v in values])
        if sum(values) == 0:
            return 0
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))
    
    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """平均多次运行的结果"""
        result = EvaluationMetrics()
        
        for attr in vars(result):
            values = [getattr(m, attr) for m in metrics_list]
            if all(isinstance(v, (int, float)) for v in values):
                setattr(result, attr, np.mean(values))
        
        return result
    
    def generate_policy_recommendation(
        self,
        target_grain_output: float,
        max_subsidy_budget: float
    ) -> Dict[str, Any]:
        """
        基于评估结果生成政策建议
        
        Args:
            target_grain_output: 目标粮食产量（吨）
            max_subsidy_budget: 最大补贴预算（元）
            
        Returns:
            政策建议字典
        """
        if not self.results:
            return {"error": "请先运行评估 (run_full_evaluation)"}
        
        recommendations = {}
        
        for ssp_code, policies in self.results.items():
            best_policy = None
            best_score = -float('inf')
            
            for policy_name, metrics in policies.items():
                # 检查约束
                if metrics.total_grain_output < target_grain_output:
                    continue
                if metrics.total_subsidy_cost > max_subsidy_budget:
                    continue
                
                # 计算综合得分
                # 目标：最大化产量、最小化成本、保护农民收入
                score = (
                    metrics.total_grain_output / target_grain_output * 0.4 +
                    (1 - metrics.total_subsidy_cost / max_subsidy_budget) * 0.3 +
                    (1 - metrics.income_gini) * 0.2 +
                    metrics.youth_farmer_share * 0.1
                )
                
                if score > best_score:
                    best_score = score
                    best_policy = policy_name
            
            recommendations[ssp_code] = {
                'recommended_policy': best_policy,
                'score': best_score,
                'metrics': policies.get(best_policy) if best_policy else None
            }
        
        return recommendations
    
    def generate_report(self, df: pd.DataFrame, output_path: str) -> str:
        """
        生成评估报告
        
        Args:
            df: 评估结果DataFrame
            output_path: 输出路径
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 70)
        report.append("粮食补贴政策评估报告")
        report.append("=" * 70)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 概述
        report.append("【研究聚焦】")
        report.append("  核心政策变量：粮食生产补贴率（元/亩）")
        report.append("  社会情景框架：SSP1-5（共享社会经济路径）")
        report.append("")
        
        # SSP情景说明
        report.append("【SSP情景说明】")
        for code, ssp in self.ssp_scenarios.items():
            report.append(f"  {code}: {ssp.name} - {ssp.description}")
        report.append("")
        
        # 补贴政策说明
        report.append("【补贴政策设定】")
        for name, policy in self.subsidy_policies.items():
            report.append(f"  {name}: {policy.subsidy_rate}元/亩 - {policy.description}")
        report.append("")
        
        # 关键发现
        report.append("【关键发现】")
        report.append("-" * 50)
        
        # 按SSP分组分析
        for ssp_code in df['SSP情景'].unique():
            ssp_df = df[df['SSP情景'] == ssp_code]
            report.append(f"\n{ssp_code}情景:")
            
            # 找出该情景下产量最高的政策
            best_output = ssp_df.loc[ssp_df['粮食产量(吨)'].idxmax()]
            report.append(f"  产量最高: {best_output['补贴政策']} ({best_output['粮食产量(吨)']:.1f}吨)")
            
            # 找出该情景下效率最高的政策
            if (ssp_df['每吨粮食补贴'] > 0).any():
                efficient_df = ssp_df[ssp_df['每吨粮食补贴'] > 0]
                best_efficiency = efficient_df.loc[efficient_df['每吨粮食补贴'].idxmin()]
                report.append(f"  效率最高: {best_efficiency['补贴政策']} ({best_efficiency['每吨粮食补贴']:.1f}元/吨)")
        
        report.append("")
        
        # 政策建议
        report.append("【政策建议】")
        report.append("-" * 50)
        report.append("1. 补贴率选择应考虑社会发展情景")
        report.append("2. SSP1(可持续)情景下中等补贴即可实现目标")
        report.append("3. SSP3(碎片化)情景需要更高补贴以维持产量")
        report.append("4. 关注青年农民占比变化，防止农业后继无人")
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def run_example():
    """使用示例"""
    print("粮食补贴政策评估器已创建。")
    print("\n使用示例:")
    print("""
    from analysis.grain_subsidy_evaluator import GrainSubsidyEvaluator
    from core.model import MainModel
    
    # 加载配置
    with open('config/master_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 创建评估器
    evaluator = GrainSubsidyEvaluator(MainModel, config)
    
    # 运行完整评估
    results_df = evaluator.run_full_evaluation(
        years=10,
        replications=3,
        ssp_codes=['SSP1', 'SSP2', 'SSP3'],
        policy_names=['low_subsidy', 'baseline_subsidy', 'high_subsidy']
    )
    
    # 生成报告
    evaluator.generate_report(results_df, 'subsidy_evaluation_report.txt')
    
    # 获取政策建议
    recommendations = evaluator.generate_policy_recommendation(
        target_grain_output=5000,  # 目标5000吨
        max_subsidy_budget=10000000  # 预算1000万
    )
    """)


if __name__ == "__main__":
    run_example()
