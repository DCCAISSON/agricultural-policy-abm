# 模型改进实施报告
## 2025-12-05 更新

---

## 概述

本次更新基于之前的深度审阅和参考项目（spatial-climate-ABM）的研究，实施了四项重要改进：

1. **细化预期参数差异** - 实现基于农户/企业特征的异质性预期参数
2. **改进空间邻近性计算** - 引入经营半径和距离成本概念
3. **创建政策效果评估器** - 系统性政策评估工具
4. **创建敏感性分析工具** - 参数敏感性分析框架

---

## 改进详情

### 1. 细化预期参数差异 (`modules/expectation_mechanism.py`)

#### 新增数据结构
```python
@dataclass
class AgentExpectationProfile:
    """主体预期参数配置"""
    # 基础经济参数
    supply_elasticity: float = 0.5
    demand_elasticity: float = -0.4
    
    # 政策相关参数
    policy_credibility: float = 0.85
    policy_attention: float = 0.8
    
    # 波动性与风险参数
    volatility_persistence: float = 0.7
    risk_aversion: float = 0.5
    
    # 信息处理能力
    information_efficiency: float = 0.8
    learning_rate: float = 0.3
    memory_span: int = 5
    
    # 社会网络影响
    social_influence: float = 0.3
    herding_tendency: float = 0.2
    
    # 认知偏差参数（前景理论）
    loss_aversion: float = 2.25
    probability_weighting: float = 0.65
    reference_dependence: float = 0.5
```

#### 农户参数计算函数
`_calculate_farmer_expectation_profile()` 考虑以下因素：
- **年龄效应**: 信息处理能力随年龄变化（倒U型），老年更保守
- **教育效应**: 教育提高信息效率，降低羊群效应
- **规模效应**: 大规模农户供给弹性更高
- **风险偏好效应**: 转换到前景理论参数
- **农户类型效应**: 纯农户vs兼业户vs非农户
- **社会资本效应**: 提高信息效率但增加羊群效应
- **经验效应**: 经验改进供给弹性估计

#### 企业参数计算函数
`_calculate_enterprise_expectation_profile()` 考虑：
- **规模效应**: 大型企业信息效率更高
- **企业类型**: 农业公司vs合作社
- **运营年限**: 经验积累
- **资本密集度**: 影响损失厌恶
- **市场导向**: 内销vs出口

#### 理论依据
- Lusardi & Mitchell (2014): 年龄与金融素养
- Huffman (2001): 教育与农业决策
- Griliches (1957): 规模与技术采纳
- Binswanger (1980): 风险偏好与农业生产
- Coleman (1988): 社会资本理论

---

### 2. 改进空间邻近性计算 (`modules/land_clearing_house.py`)

#### 新增方法

**`_calculate_operation_radius(demander)`**
借鉴spatial-climate-ABM的work_radius概念：
```python
def _calculate_operation_radius(self, demander) -> float:
    """
    计算主体的经营半径
    
    考虑因素：
    - 经营规模（对数效应）
    - 主体类型（企业>大户>普通农户）
    - 社会资本（扩展经营范围）
    """
```

**`_calculate_distance_cost(demander)`**
借鉴spatial-climate-ABM的distance_cost概念：
```python
def _calculate_distance_cost(self, demander) -> float:
    """
    计算主体的距离成本系数
    
    考虑因素：
    - 年龄（老年人距离成本高）
    - 机械化水平
    - 经济能力
    """
```

**增强的`_calculate_spatial_proximity_score()`**
- 引入个体化的经营半径(operation_radius)替代固定的max_spatial_distance
- 引入距离成本系数(distance_cost)实现主体异质性
- 使用效用函数而非简单线性衰减
- 增加连片程度加成（多个相邻地块比单个相邻价值更高）

#### 理论依据
- Deininger et al. (2012): 土地细碎化研究
- spatial-climate-ABM: utility = wage_offer - distance_cost * dist

---

### 3. 政策效果评估器 (`analysis/policy_evaluator.py`)

#### 核心类

**`PolicyIntervention`** - 单个政策干预定义
**`PolicyScenario`** - 政策情景（可包含多个干预）
**`EvaluationResult`** - 评估结果

**`PolicyEvaluator`** - 主评估器类

#### 预定义政策模板
```python
templates = {
    'grain_direct_subsidy': 粮食直补政策,
    'minimum_purchase_price': 最低收购价政策,
    'land_transfer_promotion': 土地流转促进政策,
    'new_business_entity': 新型经营主体培育,
    'socialized_service': 农业社会化服务政策,
    'agricultural_insurance': 农业保险政策,
    'farmer_pension': 农民养老保障,
    'farmland_protection': 耕地保护政策
}
```

#### 评估指标体系
- **效率指标**: 产量、单产、土地利用率、流转率、规模经营比例
- **公平指标**: 基尼系数、分位数收入、城乡差距
- **粮食安全指标**: 自给率、产量稳定性、价格波动性
- **可持续性指标**: 农民留存率、青年务农率、农业收入占比
- **财政成本**: 补贴总成本、单位粮食补贴成本

#### 核心功能
- `compare_scenarios()`: 多情景对比
- `generate_report()`: 生成评估报告

---

### 4. 敏感性分析工具 (`analysis/sensitivity_analysis.py`)

#### 核心类

**`ParameterRange`** - 参数范围定义
**`SensitivityResult`** - OAT分析结果
**`MorrisResult`** - Morris方法结果
**`SobolResult`** - Sobol全局敏感性结果

**`SensitivityAnalyzer`** - 主分析器类

#### 预定义参数
```python
default_params = [
    # 市场参数
    "粮食基准价格", "经作价格", "劳动力工资率",
    # 补贴参数
    "粮食补贴率", "每亩补贴",
    # 生产参数
    "基础产量", "技术进步率",
    # 土地流转参数
    "流转交易成本", "熟人折扣",
    # 行为参数
    "风险厌恶系数", "损失厌恶系数",
    # 代际参数
    "代际传承概率", "青年务农意愿"
]
```

#### 分析方法
1. **OAT分析** (`run_oat_analysis`): 单参数敏感性
   - 弹性系数计算
   - 线性回归拟合
   - R²评估

2. **Morris方法** (`run_morris_analysis`): 全局筛选
   - μ*: 绝对均值效应
   - σ: 交互效应标准差
   - 重要性分类

3. **情景组合分析** (`run_scenario_analysis`)
   - 全因子设计
   - 多指标评估

#### 输出功能
- `generate_sensitivity_report()`: 文本报告
- `export_results_to_csv()`: CSV导出

---

## 使用示例

### 预期机制使用
```python
from modules.expectation_mechanism import ExpectationFactory

# 创建农户预期机制（会自动根据特征计算参数）
mechanism = ExpectationFactory.create_for_farmer(
    farmer_type='Pure_Farmer',
    age=45,
    experience_years=20,
    education_years=9,
    land_scale=15.0,
    risk_preference=0.5,
    social_capital=6.0
)

# 查看计算出的参数配置
print(mechanism.profile.supply_elasticity)
print(mechanism.profile.information_efficiency)
```

### 政策评估使用
```python
from analysis.policy_evaluator import PolicyEvaluator, PolicyScenario
from core.model import GrainProductionModel

evaluator = PolicyEvaluator(GrainProductionModel, base_config)

# 创建政策情景
scenario = evaluator.create_scenario_from_template(
    'grain_direct_subsidy',
    '高补贴情景',
    intensity=1.5
)

# 运行对比
comparison_df = evaluator.compare_scenarios([scenario])
evaluator.generate_report(comparison_df, 'report.txt')
```

### 敏感性分析使用
```python
from analysis.sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(
    GrainProductionModel,
    base_config,
    output_metrics=['total_grain_output', 'gini_coefficient']
)

# 运行OAT分析
oat_results = analyzer.run_oat_analysis(
    parameter_names=['粮食基准价格', '粮食补贴率']
)

# 生成报告
analyzer.generate_sensitivity_report('sensitivity_report.txt')
```

---

## 修改文件清单

| 文件 | 修改类型 | 主要内容 |
|------|---------|---------|
| `modules/expectation_mechanism.py` | 增强 | 新增AgentExpectationProfile、参数计算函数、村集体预期 |
| `modules/land_clearing_house.py` | 增强 | 新增operation_radius、distance_cost方法，重写proximity评分 |
| `agents/farmer_agent.py` | 更新 | 传入更多特征参数到预期机制 |
| `agents/enterprise_agent.py` | 更新 | 传入capital_intensity、market_orientation参数 |
| `analysis/policy_evaluator.py` | 新建 | 完整的政策评估框架 |
| `analysis/sensitivity_analysis.py` | 新建 | 完整的敏感性分析框架 |

---

## 理论支撑总结

本次改进的理论基础：

1. **异质性主体建模**
   - spatial-climate-ABM的distance_cost模式
   - Melitz (2003) 异质性企业模型

2. **空间经济学**
   - Deininger et al. (2012) 土地细碎化
   - 空间效用函数: U = value - distance_cost * distance

3. **政策评估**
   - Rubin因果模型
   - 反事实分析框架

4. **敏感性分析**
   - Saltelli et al. (2008) 全局敏感性分析
   - Morris筛选方法
   - Sobol指数

---

## 后续建议

1. **参数校准**: 使用CLES数据校准异质性参数分布
2. **验证测试**: 运行完整模拟验证新功能
3. **可视化**: 添加敏感性分析和政策评估的可视化图表
4. **文档更新**: 更新ODD+D框架文档

---

*报告生成时间: 2025-12-05*
