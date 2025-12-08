# -*- coding: utf-8 -*-
"""
@module policy_evaluator.py
@description 政策效果评估器 - 系统性评估不同政策干预的效果
@created 2025-12-05
@author GitHub Copilot

理论依据：
1. 政策评估框架 (Policy Evaluation Framework)
   - Rubin因果模型：比较处理组与对照组
   - 差分中的差分 (DID)：政策前后对比
   - 反事实分析：模拟无政策情景

2. ABM政策实验方法 (Tesfatsion, 2006)
   - ABM作为政策实验室
   - 多情景比较分析
   - 参数敏感性评估

3. 农业政策评估标准
   - 效率：资源配置效率、生产效率
   - 公平：收入分配、代际公平
   - 可持续性：环境、社会、经济可持续
   - 粮食安全：产量稳定性、供应充足性

核心功能：
1. 定义政策干预场景
2. 运行基准与干预对比实验
3. 计算政策效果指标
4. 生成评估报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import copy


class PolicyType(Enum):
    """政策类型枚举"""
    SUBSIDY = "subsidy"                     # 补贴政策
    PRICE_SUPPORT = "price_support"         # 价格支持
    LAND_TRANSFER = "land_transfer"         # 土地流转政策
    CREDIT_ACCESS = "credit_access"         # 信贷政策
    TECHNOLOGY = "technology"               # 技术推广
    MARKET_REFORM = "market_reform"         # 市场改革
    ENVIRONMENTAL = "environmental"         # 环境政策
    SOCIAL_SECURITY = "social_security"     # 社会保障
    COMPOSITE = "composite"                 # 组合政策


@dataclass
class PolicyIntervention:
    """
    政策干预定义
    
    描述一个具体的政策干预措施及其参数
    """
    name: str                               # 政策名称
    policy_type: PolicyType                 # 政策类型
    description: str                        # 政策描述
    
    # 政策参数修改
    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    
    # 政策实施时机
    start_year: int = 0                     # 开始年份（模拟内）
    end_year: Optional[int] = None          # 结束年份（None表示持续）
    
    # 政策强度
    intensity: float = 1.0                  # 政策强度系数 [0, 2]
    
    # 目标群体
    target_agents: Optional[List[str]] = None  # 目标主体类型
    
    # 政策实施函数（可自定义）
    implementation_func: Optional[Callable] = None


@dataclass 
class PolicyScenario:
    """
    政策情景定义
    
    一个情景可以包含多个政策干预的组合
    """
    name: str                               # 情景名称
    description: str                        # 情景描述
    interventions: List[PolicyIntervention] = field(default_factory=list)
    
    # 情景运行参数
    simulation_years: int = 10              # 模拟年限
    replications: int = 5                   # 重复次数（考虑随机性）
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationResult:
    """
    政策评估结果
    
    包含单个情景的评估结果
    """
    scenario_name: str
    
    # 效率指标
    total_grain_output: float = 0.0         # 粮食总产量
    output_per_hectare: float = 0.0         # 单位面积产出
    land_utilization_rate: float = 0.0      # 土地利用率
    land_transfer_rate: float = 0.0         # 土地流转率
    scale_operation_ratio: float = 0.0      # 规模经营比例
    
    # 公平指标
    gini_coefficient: float = 0.0           # 基尼系数
    income_by_quantile: Dict[str, float] = field(default_factory=dict)  # 分位数收入
    urban_rural_gap: float = 0.0            # 城乡差距
    
    # 粮食安全指标
    grain_self_sufficiency: float = 0.0     # 粮食自给率
    output_stability_cv: float = 0.0        # 产量稳定性（变异系数）
    grain_price_volatility: float = 0.0     # 价格波动性
    
    # 可持续性指标
    farmer_retention_rate: float = 0.0      # 农民留存率
    youth_farming_rate: float = 0.0         # 青年务农率
    agricultural_income_share: float = 0.0  # 农业收入占比
    
    # 财政成本
    total_subsidy_cost: float = 0.0         # 补贴总成本
    cost_per_ton_grain: float = 0.0         # 单位粮食补贴成本
    
    # 时间序列数据
    annual_data: List[Dict] = field(default_factory=list)
    
    # 统计显著性
    confidence_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class PolicyEvaluator:
    """
    政策效果评估器
    
    提供系统性的政策评估功能
    
    使用方法：
    1. 定义基准情景
    2. 定义干预情景
    3. 运行对比实验
    4. 分析政策效果
    5. 生成评估报告
    """
    
    def __init__(self, model_class, base_config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            model_class: 模型类（如GrainProductionModel）
            base_config: 基准配置字典
        """
        self.model_class = model_class
        self.base_config = copy.deepcopy(base_config)
        
        # 存储评估结果
        self.results: Dict[str, List[EvaluationResult]] = {}
        
        # 预定义政策模板
        self.policy_templates = self._define_policy_templates()
        
        # 评估指标权重（可调整）
        self.indicator_weights = {
            'efficiency': 0.30,
            'equity': 0.25,
            'food_security': 0.25,
            'sustainability': 0.20
        }
    
    def _define_policy_templates(self) -> Dict[str, PolicyIntervention]:
        """
        定义常用政策模板
        
        基于中国现实农业政策体系
        """
        templates = {}
        
        # 1. 粮食直补政策
        templates['grain_direct_subsidy'] = PolicyIntervention(
            name="粮食直补政策",
            policy_type=PolicyType.SUBSIDY,
            description="按种粮面积发放直接补贴",
            parameter_changes={
                'grain_subsidy_per_mu': 100.0,  # 每亩补贴（元）
                'subsidy_type': 'area_based'
            }
        )
        
        # 2. 最低收购价政策
        templates['minimum_purchase_price'] = PolicyIntervention(
            name="最低收购价政策",
            policy_type=PolicyType.PRICE_SUPPORT,
            description="政府设定粮食最低收购价",
            parameter_changes={
                'floor_price_grain': 2.58,  # 最低收购价
                'price_support_enabled': True
            }
        )
        
        # 3. 土地流转促进政策
        templates['land_transfer_promotion'] = PolicyIntervention(
            name="土地流转促进政策",
            policy_type=PolicyType.LAND_TRANSFER,
            description="降低流转交易成本，鼓励规模经营",
            parameter_changes={
                'transfer_transaction_cost': 0.05,  # 降低交易成本
                'scale_operation_bonus': 0.10,      # 规模经营奖励
                'land_bank_enabled': True           # 启用土地银行
            }
        )
        
        # 4. 新型经营主体培育
        templates['new_business_entity'] = PolicyIntervention(
            name="新型经营主体培育",
            policy_type=PolicyType.SUBSIDY,
            description="支持家庭农场、合作社发展",
            parameter_changes={
                'family_farm_subsidy': 50000,       # 家庭农场补贴
                'cooperative_subsidy': 100000,      # 合作社补贴
                'training_subsidy': 5000            # 培训补贴
            },
            target_agents=['enterprise', 'large_farmer']
        )
        
        # 5. 农业社会化服务
        templates['socialized_service'] = PolicyIntervention(
            name="农业社会化服务政策",
            policy_type=PolicyType.TECHNOLOGY,
            description="推广托管、代耕等社会化服务",
            parameter_changes={
                'service_coverage': 0.8,            # 服务覆盖率
                'service_subsidy_rate': 0.3,        # 服务补贴比例
                'mechanization_promotion': True
            }
        )
        
        # 6. 农业保险政策
        templates['agricultural_insurance'] = PolicyIntervention(
            name="农业保险政策",
            policy_type=PolicyType.CREDIT_ACCESS,
            description="补贴农业保险保费",
            parameter_changes={
                'insurance_premium_subsidy': 0.8,   # 保费补贴比例
                'coverage_rate': 0.7,               # 覆盖率目标
                'compensation_ratio': 0.6           # 赔付比例
            }
        )
        
        # 7. 退休农民基本保障
        templates['farmer_pension'] = PolicyIntervention(
            name="农民养老保障",
            policy_type=PolicyType.SOCIAL_SECURITY,
            description="为退出农业的老年农民提供养老保障",
            parameter_changes={
                'pension_monthly': 500,             # 月养老金
                'retirement_age': 60,               # 退休年龄
                'pension_subsidy_ratio': 0.5        # 财政补贴比例
            },
            target_agents=['farmer']
        )
        
        # 8. 耕地保护政策
        templates['farmland_protection'] = PolicyIntervention(
            name="耕地保护政策",
            policy_type=PolicyType.ENVIRONMENTAL,
            description="严格保护耕地，限制非农用途",
            parameter_changes={
                'conversion_restriction': 0.95,     # 限制转非率
                'ecological_compensation': 200,     # 生态补偿/亩
                'quality_protection_bonus': 100     # 质量保护奖励
            }
        )
        
        return templates
    
    def create_baseline_scenario(self, name: str = "基准情景") -> PolicyScenario:
        """
        创建基准情景（无政策干预）
        """
        return PolicyScenario(
            name=name,
            description="无政策干预的基准情景",
            interventions=[],
            simulation_years=10,
            replications=5
        )
    
    def create_scenario_from_template(
        self,
        template_name: str,
        scenario_name: str,
        intensity: float = 1.0,
        **kwargs
    ) -> PolicyScenario:
        """
        从模板创建政策情景
        
        Args:
            template_name: 模板名称
            scenario_name: 情景名称
            intensity: 政策强度
            **kwargs: 覆盖模板参数
        """
        if template_name not in self.policy_templates:
            raise ValueError(f"未知政策模板: {template_name}")
        
        template = copy.deepcopy(self.policy_templates[template_name])
        template.intensity = intensity
        
        # 更新参数
        for key, value in kwargs.items():
            if key in template.parameter_changes:
                template.parameter_changes[key] = value
        
        return PolicyScenario(
            name=scenario_name,
            description=f"基于{template.name}模板的政策情景",
            interventions=[template],
            simulation_years=10,
            replications=5
        )
    
    def create_composite_scenario(
        self,
        name: str,
        template_names: List[str],
        description: str = ""
    ) -> PolicyScenario:
        """
        创建组合政策情景
        
        Args:
            name: 情景名称
            template_names: 模板名称列表
            description: 情景描述
        """
        interventions = []
        for tname in template_names:
            if tname in self.policy_templates:
                interventions.append(copy.deepcopy(self.policy_templates[tname]))
        
        return PolicyScenario(
            name=name,
            description=description or f"组合政策情景: {', '.join(template_names)}",
            interventions=interventions,
            simulation_years=10,
            replications=5
        )
    
    def apply_policy_to_config(
        self,
        config: Dict[str, Any],
        scenario: PolicyScenario
    ) -> Dict[str, Any]:
        """
        将政策干预应用到配置
        
        Args:
            config: 基础配置
            scenario: 政策情景
            
        Returns:
            修改后的配置
        """
        modified_config = copy.deepcopy(config)
        
        for intervention in scenario.interventions:
            for param_path, value in intervention.parameter_changes.items():
                # 支持嵌套路径，如 'market.grain_price'
                keys = param_path.split('.')
                target = modified_config
                
                for key in keys[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                
                # 应用政策强度系数
                if isinstance(value, (int, float)):
                    value = value * intervention.intensity
                
                target[keys[-1]] = value
        
        return modified_config
    
    def run_simulation(
        self,
        config: Dict[str, Any],
        years: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行单次模拟并收集数据
        
        Args:
            config: 模型配置
            years: 模拟年数
            seed: 随机种子
            
        Returns:
            模拟结果数据
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 创建模型实例
        model = self.model_class(config)
        
        # 运行模拟并收集年度数据
        annual_data = []
        for year in range(years):
            model.step()
            
            # 收集关键指标
            year_data = {
                'year': year,
                'total_grain_output': self._get_total_grain_output(model),
                'average_income': self._get_average_income(model),
                'land_transfer_rate': self._get_land_transfer_rate(model),
                'gini_coefficient': self._calculate_gini(model),
                'farmer_count': self._get_farmer_count(model),
                'youth_farmer_ratio': self._get_youth_farmer_ratio(model),
                'grain_price': self._get_grain_price(model),
                'total_subsidy': self._get_total_subsidy(model),
                'scale_operation_area': self._get_scale_operation_area(model)
            }
            annual_data.append(year_data)
        
        return {
            'annual_data': annual_data,
            'final_state': self._extract_final_state(model)
        }
    
    def evaluate_scenario(
        self,
        scenario: PolicyScenario,
        baseline_results: Optional[List[Dict]] = None
    ) -> EvaluationResult:
        """
        评估单个政策情景
        
        Args:
            scenario: 政策情景
            baseline_results: 基准情景结果（用于对比）
            
        Returns:
            评估结果
        """
        # 应用政策到配置
        policy_config = self.apply_policy_to_config(self.base_config, scenario)
        
        # 多次重复运行
        all_results = []
        for rep in range(scenario.replications):
            seed = 42 + rep  # 可重复的随机种子
            result = self.run_simulation(
                policy_config,
                scenario.simulation_years,
                seed
            )
            all_results.append(result)
        
        # 汇总结果
        evaluation = self._aggregate_results(scenario.name, all_results)
        
        # 如果有基准结果，计算政策效应
        if baseline_results:
            evaluation = self._calculate_policy_effects(evaluation, baseline_results)
        
        return evaluation
    
    def _aggregate_results(
        self,
        scenario_name: str,
        results: List[Dict]
    ) -> EvaluationResult:
        """汇总多次运行结果"""
        evaluation = EvaluationResult(scenario_name=scenario_name)
        
        # 提取最后一年的数据
        final_year_data = [r['annual_data'][-1] for r in results]
        
        # 计算均值
        evaluation.total_grain_output = np.mean([d['total_grain_output'] for d in final_year_data])
        evaluation.gini_coefficient = np.mean([d['gini_coefficient'] for d in final_year_data])
        evaluation.land_transfer_rate = np.mean([d['land_transfer_rate'] for d in final_year_data])
        evaluation.farmer_retention_rate = np.mean([d['farmer_count'] for d in final_year_data]) / np.mean([results[0]['annual_data'][0]['farmer_count'] for r in results])
        evaluation.youth_farming_rate = np.mean([d['youth_farmer_ratio'] for d in final_year_data])
        evaluation.total_subsidy_cost = np.mean([d['total_subsidy'] for d in final_year_data])
        
        # 计算产量稳定性（变异系数）
        all_outputs = []
        for r in results:
            outputs = [d['total_grain_output'] for d in r['annual_data']]
            all_outputs.extend(outputs)
        evaluation.output_stability_cv = np.std(all_outputs) / np.mean(all_outputs) if np.mean(all_outputs) > 0 else 0
        
        # 计算置信区间
        for metric in ['total_grain_output', 'gini_coefficient', 'land_transfer_rate']:
            values = [d[metric] for d in final_year_data]
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            ci = 1.96 * std / np.sqrt(n)
            evaluation.confidence_interval[metric] = (mean - ci, mean + ci)
        
        # 保存年度数据
        evaluation.annual_data = [r['annual_data'] for r in results]
        
        return evaluation
    
    def _calculate_policy_effects(
        self,
        policy_result: EvaluationResult,
        baseline_results: List[Dict]
    ) -> EvaluationResult:
        """计算相对于基准的政策效应"""
        baseline_eval = self._aggregate_results("baseline", baseline_results)
        
        # 计算变化率
        if baseline_eval.total_grain_output > 0:
            policy_result.output_per_hectare = (
                policy_result.total_grain_output / baseline_eval.total_grain_output - 1
            ) * 100  # 百分比变化
        
        # 计算补贴效率
        output_increase = policy_result.total_grain_output - baseline_eval.total_grain_output
        if policy_result.total_subsidy_cost > 0:
            policy_result.cost_per_ton_grain = policy_result.total_subsidy_cost / max(1, output_increase)
        
        return policy_result
    
    def compare_scenarios(
        self,
        scenarios: List[PolicyScenario],
        include_baseline: bool = True
    ) -> pd.DataFrame:
        """
        比较多个政策情景
        
        Args:
            scenarios: 情景列表
            include_baseline: 是否包含基准情景
            
        Returns:
            对比结果DataFrame
        """
        results = []
        baseline_results = None
        
        # 运行基准
        if include_baseline:
            baseline_scenario = self.create_baseline_scenario()
            baseline_config = self.base_config
            baseline_raw = []
            for rep in range(baseline_scenario.replications):
                result = self.run_simulation(
                    baseline_config,
                    baseline_scenario.simulation_years,
                    42 + rep
                )
                baseline_raw.append(result)
            baseline_results = baseline_raw
            
            baseline_eval = self._aggregate_results("基准情景", baseline_raw)
            results.append(self._evaluation_to_dict(baseline_eval))
        
        # 运行各政策情景
        for scenario in scenarios:
            evaluation = self.evaluate_scenario(scenario, baseline_results)
            results.append(self._evaluation_to_dict(evaluation))
        
        df = pd.DataFrame(results)
        return df
    
    def _evaluation_to_dict(self, evaluation: EvaluationResult) -> Dict:
        """将评估结果转换为字典"""
        return {
            '情景名称': evaluation.scenario_name,
            '粮食总产量': evaluation.total_grain_output,
            '基尼系数': evaluation.gini_coefficient,
            '土地流转率': evaluation.land_transfer_rate,
            '农民留存率': evaluation.farmer_retention_rate,
            '青年务农率': evaluation.youth_farming_rate,
            '产量稳定性(CV)': evaluation.output_stability_cv,
            '补贴总成本': evaluation.total_subsidy_cost,
            '单位粮食补贴成本': evaluation.cost_per_ton_grain
        }
    
    def generate_report(
        self,
        comparison_df: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        生成政策评估报告
        
        Args:
            comparison_df: 对比结果DataFrame
            output_path: 输出路径
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("政策效果评估报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 情景概览
        report.append("【情景概览】")
        report.append(comparison_df.to_string())
        report.append("")
        
        # 效率评估
        report.append("【效率评估】")
        baseline_output = comparison_df.loc[comparison_df['情景名称'] == '基准情景', '粮食总产量'].values
        if len(baseline_output) > 0:
            baseline_output = baseline_output[0]
            for _, row in comparison_df.iterrows():
                if row['情景名称'] != '基准情景':
                    change = (row['粮食总产量'] / baseline_output - 1) * 100
                    report.append(f"  {row['情景名称']}: 产量变化 {change:+.2f}%")
        report.append("")
        
        # 公平评估
        report.append("【公平评估】")
        for _, row in comparison_df.iterrows():
            gini = row['基尼系数']
            if gini < 0.3:
                equity_level = "高度公平"
            elif gini < 0.4:
                equity_level = "相对公平"
            else:
                equity_level = "存在不公平"
            report.append(f"  {row['情景名称']}: 基尼系数 {gini:.3f} ({equity_level})")
        report.append("")
        
        # 可持续性评估
        report.append("【可持续性评估】")
        for _, row in comparison_df.iterrows():
            retention = row['农民留存率'] * 100
            youth = row['青年务农率'] * 100
            report.append(f"  {row['情景名称']}: 农民留存率 {retention:.1f}%, 青年务农率 {youth:.1f}%")
        report.append("")
        
        # 政策建议
        report.append("【政策建议】")
        # 找出最优情景
        best_output_idx = comparison_df['粮食总产量'].idxmax()
        best_equity_idx = comparison_df['基尼系数'].idxmin()
        best_sustainability_idx = comparison_df['农民留存率'].idxmax()
        
        report.append(f"  - 产量最优: {comparison_df.loc[best_output_idx, '情景名称']}")
        report.append(f"  - 公平最优: {comparison_df.loc[best_equity_idx, '情景名称']}")
        report.append(f"  - 可持续最优: {comparison_df.loc[best_sustainability_idx, '情景名称']}")
        
        report_text = "\n".join(report)
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    # ============ 数据提取辅助方法 ============
    
    def _get_total_grain_output(self, model) -> float:
        """获取粮食总产量"""
        if hasattr(model, 'datacollector'):
            try:
                model_data = model.datacollector.get_model_vars_dataframe()
                if 'Total_Grain_Output' in model_data.columns:
                    return model_data['Total_Grain_Output'].iloc[-1]
            except:
                pass
        
        # 备选方案：从agents汇总
        total = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'grain_output'):
                total += agent.grain_output
        return total
    
    def _get_average_income(self, model) -> float:
        """获取平均收入"""
        incomes = []
        for agent in model.schedule.agents:
            if hasattr(agent, 'total_income'):
                incomes.append(agent.total_income)
            elif hasattr(agent, 'agricultural_income'):
                incomes.append(agent.agricultural_income)
        return np.mean(incomes) if incomes else 0
    
    def _get_land_transfer_rate(self, model) -> float:
        """获取土地流转率"""
        if hasattr(model, 'environment') and hasattr(model.environment, 'land_transfer_rate'):
            return model.environment.land_transfer_rate
        
        # 备选计算
        total_area = 0
        transferred_area = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'owned_land_area'):
                total_area += agent.owned_land_area
            if hasattr(agent, 'rented_out_area'):
                transferred_area += agent.rented_out_area
        return transferred_area / total_area if total_area > 0 else 0
    
    def _calculate_gini(self, model) -> float:
        """计算基尼系数"""
        incomes = []
        for agent in model.schedule.agents:
            if hasattr(agent, 'total_income'):
                incomes.append(max(0, agent.total_income))
        
        if not incomes or len(incomes) < 2:
            return 0
        
        incomes = np.sort(incomes)
        n = len(incomes)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * incomes) - (n + 1) * np.sum(incomes)) / (n * np.sum(incomes))
    
    def _get_farmer_count(self, model) -> int:
        """获取农民数量"""
        count = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'is_farming') and agent.is_farming:
                count += 1
            elif hasattr(agent, 'livelihood_strategy'):
                count += 1
        return count
    
    def _get_youth_farmer_ratio(self, model) -> float:
        """获取青年农民比例"""
        total = 0
        youth = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'age'):
                total += 1
                if agent.age < 45:
                    youth += 1
        return youth / total if total > 0 else 0
    
    def _get_grain_price(self, model) -> float:
        """获取粮食价格"""
        if hasattr(model, 'environment'):
            return model.environment.get_market_price('grain')
        return 2.58
    
    def _get_total_subsidy(self, model) -> float:
        """获取补贴总额"""
        total = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'subsidy_received'):
                total += agent.subsidy_received
        return total
    
    def _get_scale_operation_area(self, model) -> float:
        """获取规模经营面积"""
        total = 0
        for agent in model.schedule.agents:
            if hasattr(agent, 'get_operated_land_area'):
                area = agent.get_operated_land_area()
                if area > 30:  # 规模经营阈值
                    total += area
        return total
    
    def _extract_final_state(self, model) -> Dict:
        """提取模型最终状态"""
        return {
            'total_agents': len(model.schedule.agents),
            'current_step': model.schedule.steps
        }


def run_policy_evaluation_example():
    """
    政策评估示例代码
    
    演示如何使用PolicyEvaluator进行政策分析
    """
    # 注意：需要导入实际的模型类
    # from core.model import GrainProductionModel
    
    # 示例配置
    base_config = {
        'num_farmers': 100,
        'initial_grain_price': 2.58,
        'subsidy_rate': 0.1,
        # ... 更多配置
    }
    
    # 创建评估器
    # evaluator = PolicyEvaluator(GrainProductionModel, base_config)
    
    # 创建政策情景
    # scenario1 = evaluator.create_scenario_from_template(
    #     'grain_direct_subsidy',
    #     '高补贴情景',
    #     intensity=1.5
    # )
    
    # scenario2 = evaluator.create_composite_scenario(
    #     '综合改革情景',
    #     ['grain_direct_subsidy', 'land_transfer_promotion', 'new_business_entity']
    # )
    
    # 运行对比
    # comparison_df = evaluator.compare_scenarios([scenario1, scenario2])
    
    # 生成报告
    # evaluator.generate_report(comparison_df, 'policy_report.txt')
    
    print("政策评估模块已创建。使用示例见函数注释。")


if __name__ == "__main__":
    run_policy_evaluation_example()
