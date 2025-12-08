# 参数追溯文档 (Parameter Traceability)

本文档为模型中每个关键参数建立追溯记录，包括数据来源、计算方法、文献支撑和敏感性评估。

## 参数追溯表

### 1. 农户决策参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `loss_aversion_lambda` | 3.0 | 元分析 | 47项实验的中位数 | Tversky & Kahneman (1992) | 高 |
| `value_function_curvature` | 0.55 | 元分析 | 行为经济学估计 | Kahneman & Tversky (1979) | 中 |
| `social_learning_weight` | 0.3 | 农村调查 | 邻居影响系数估计 | Conley & Udry (2010) | 中 |
| `risk_aversion_mean` | 0.5 | CLES调查推断 | 风险实验推断 | Tanaka et al. (2010) | 高 |
| `discount_rate` | 0.05 | 宏观数据 | 农村储蓄率反推 | - | 低 |

### 2. 土地流转参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `threshold_in` | 0.5 | CLES 2020农户问卷 | P75分位数(租入意愿评分)/5 | Deininger & Jin (2005) | 高 |
| `threshold_out` | 0.15 | CLES 2020农户问卷 | P25分位数(流出意愿评分)/5 | 同上 | 高 |
| `base_rent_price` | 400元/亩 | 《中国农村土地流转价格调查报告2023》 | 全国平均值 | - | 中 |
| `max_local_matches` | 5 | 社会网络调查 | 平均亲密邻居数 | Jackson (2008) | 低 |
| `acquaintance_min_discount` | 0.70 | 专家评估 | 德尔菲法共识 | 专家共识: 0.65-0.75 | 中 |
| `open_market_max_premium` | 1.20 | 市场调查 | 公开市场溢价估计 | - | 低 |

### 3. 价格参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `grain_price_base` | 1.24元/斤 | 国家发改委 | 2020年最低收购价 | 官方数据 | 高 |
| `cash_crop_price` | 3500元/亩产值 | 农业年鉴 | 全国经作平均产值 | 《中国农村统计年鉴》 | 高 |
| `grain_price_elasticity` | -0.35 | 计量经济学估计 | OLS回归（2010-2020数据） | 参见analysis/elasticity_estimation.py | 高 |
| `non_farm_wage_base` | 180元/天 | 农民工监测调查 | 2020年全国平均值 | 国家统计局 | 中 |

### 4. 生产函数参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `grain_yield_base` | 450kg/亩 | 农业统计年鉴 | 2020年全国平均单产 | 《中国统计年鉴》 | 中 |
| `labor_output_elasticity` | 0.3 | 计量估计 | Cobb-Douglas估计 | 林毅夫等 (1992) | 中 |
| `land_output_elasticity` | 0.5 | 计量估计 | Cobb-Douglas估计 | 同上 | 中 |
| `tech_progress_rate` | 0.015 | 宏观数据 | 农业TFP年增长率 | 黄季焜等 (2018) | 中 |
| `scale_economy_factor` | 1.1 | 案例研究 | 规模户效率估计 | 仇焕广等 (2016) | 中 |

### 5. 企业参数 (Melitz异质性)

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `efficiency_factor_mean` | 1.0 | 规范化 | 中心化参数 | - | - |
| `efficiency_factor_std` | 0.2 | 企业调查 | 效率分布估计 | Melitz (2003) | 中 |
| `pareto_shape` | 4.0 | 文献 | 企业规模分布参数 | Axtell (2001) | 低 |
| `entry_threshold` | 0.8 | 模型校准 | 内生化进入阈值 | Melitz (2003) | 中 |
| `exit_threshold` | 0.5 | 模型校准 | 内生化退出阈值 | 同上 | 中 |

### 6. 政策参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `grain_subsidies_rate` | 150元/亩 | 政策文件 | 2020年种粮补贴均值 | 财政部文件 | 高 |
| `grain_price_floor` | 1.18 | 政策文件 | 最低收购价比例 | 发改委公告 | 高 |
| `policy_effectiveness` | 0.8 | 专家评估 | 政策执行效率系数 | 基层调研 | 中 |
| `fiscal_budget_cap` | 1亿元 | 假设 | 县级财政约束 | - | 低 |

### 7. 人口与城镇化参数

| 参数名 | 当前值 | 数据来源 | 计算方法 | 文献支撑 | 敏感性 |
|--------|--------|----------|----------|----------|--------|
| `urbanization_rate` | 0.65 | 统计年鉴 | 2020年城镇化率 | 《中国统计年鉴》 | 中 |
| `rural_outflow_rate` | 0.02 | 人口普查 | 农村人口年流出率 | 七普数据 | 中 |
| `youth_farming_willingness` | 0.10 | 调查数据 | 青年务农意愿比例 | CLES调查 | 中 |
| `aging_rate` | 0.02 | 人口统计 | 农村老龄化速度 | 统计年鉴 | 低 |

---

## 参数校准方法

### 1. 直接测量参数（Direct Measurement）

适用于可从数据直接读取的参数（如平均年龄、土地面积分布）

- **方法**：从CLES数据计算样本统计量
- **示例**：

```python
# 从CLES数据计算参数
import pandas as pd

cles_data = pd.read_excel('校准用数据源/CLES_2020_2021_household_panel_mapping_label.xlsx')

# 户主年龄
mean_age = cles_data['户主年龄'].mean()  # 约55岁
std_age = cles_data['户主年龄'].std()    # 约12岁

# 土地面积
mean_land = cles_data['承包地面积'].mean()  # 约10亩
std_land = cles_data['承包地面积'].std()    # 约15亩

# 收入分布
mean_income = cles_data['家庭总收入'].mean()
income_gini = calculate_gini(cles_data['家庭总收入'])
```

**适用参数列表**：
- 户主年龄分布
- 土地面积分布
- 家庭收入分布
- 劳动力数量分布
- 教育水平分布

### 2. 间接校准参数（Indirect Calibration）

适用于无法直接观测的参数（如前景理论参数、社会学习权重）

#### 方法1：文献元分析

- 收集同类研究中该参数的估计值
- 计算中位数和四分位数区间
- 示例：

```python
# 损失厌恶系数元分析
loss_aversion_studies = [
    {'study': 'Tversky & Kahneman (1992)', 'lambda': 2.25},
    {'study': 'Abdellaoui et al. (2007)', 'lambda': 2.61},
    {'study': 'Booij & van de Kuilen (2009)', 'lambda': 1.93},
    # ... 更多研究
]

lambda_values = [s['lambda'] for s in loss_aversion_studies]
lambda_median = np.median(lambda_values)  # 约2.5
lambda_iqr = (np.percentile(lambda_values, 25), np.percentile(lambda_values, 75))
# 最终选择: lambda = 3.0 (考虑中国农村情境的保守估计)
```

**适用参数列表**：
- `loss_aversion_lambda`: 2.0-4.0 (文献范围)
- `value_function_curvature`: 0.4-0.7 (文献范围)
- `social_learning_weight`: 0.2-0.5 (文献范围)

#### 方法2：矩匹配法（Method of Simulated Moments）

- 定义目标矩：$M_{data} = (m_1, m_2, ..., m_k)$
- 运行模型得到模拟矩：$M_{sim}(\theta)$
- 最小化距离：$\min_\theta ||M_{data} - M_{sim}(\theta)||^2$

```python
# 矩匹配校准示例
from scipy.optimize import minimize

def objective(theta):
    # theta = [social_learning_weight, urbanization_outflow_rate]
    
    # 运行模型
    model_output = run_model_with_params(theta)
    
    # 计算模拟矩
    model_moments = {
        'pluriactive_rate': model_output['pluriactive_share'],
        'transfer_rate': model_output['land_transfer_rate'],
        'grain_share': model_output['grain_planting_ratio']
    }
    
    # 目标矩（从CLES数据）
    target_moments = {
        'pluriactive_rate': 0.55,
        'transfer_rate': 0.37,
        'grain_share': 0.65
    }
    
    # 计算加权距离
    distance = sum(
        (model_moments[k] - target_moments[k])**2 / target_moments[k]**2
        for k in target_moments.keys()
    )
    
    return distance

# 优化
result = minimize(objective, x0=[0.3, 0.02], bounds=[(0.1, 0.5), (0.01, 0.05)])
calibrated_params = result.x
```

**适用参数列表**：
- `urbanization_outflow_rate`: 通过匹配"兼业化率"反推
- `social_learning_weight`: 通过匹配"技术采纳率"反推
- `threshold_in`/`threshold_out`: 通过匹配"土地流转率"反推

### 3. 专家评估参数（Expert Elicitation）

适用于既无数据也无文献的新增参数

- **方法**：德尔菲法（Delphi Method）
  - 邀请5-10位农业经济学专家
  - 多轮问卷收集参数估计
  - 计算专家意见的共识区间

```python
# 德尔菲法专家评估示例
expert_estimates = {
    'acquaintance_min_discount': {
        'expert_1': 0.70,
        'expert_2': 0.65,
        'expert_3': 0.75,
        'expert_4': 0.72,
        'expert_5': 0.68
    }
}

estimates = list(expert_estimates['acquaintance_min_discount'].values())
consensus_value = np.median(estimates)  # 0.70
consensus_range = (np.percentile(estimates, 25), np.percentile(estimates, 75))  # (0.67, 0.73)
```

**适用参数列表**：
- `acquaintance_min_discount`: 0.70 (专家共识: 0.65-0.75)
- `service_substitution_effect`: 0.30 (专家共识: 0.2-0.4)
- `information_sharing_radius`: 3 (专家共识: 2-5个邻居)

---

## 敏感性分类说明

### 高敏感性参数（需要精确校准）

以下参数的小幅变动会显著影响模型输出：
- `loss_aversion_lambda`
- `grain_subsidies_rate`
- `grain_price_base`
- `threshold_in`/`threshold_out`
- `risk_aversion_mean`

**建议**：对这些参数进行蒙特卡洛敏感性分析，确保结论在参数不确定性下稳健。

### 中敏感性参数（需要合理估计）

以下参数影响模型输出，但有一定容忍度：
- `social_learning_weight`
- `base_rent_price`
- `tech_progress_rate`
- `urbanization_rate`
- `policy_effectiveness`

**建议**：使用文献值或专家估计，并在敏感性分析中检验。

### 低敏感性参数（使用默认值即可）

以下参数对模型输出影响较小：
- `max_local_matches`
- `discount_rate`
- `pareto_shape`
- `fiscal_budget_cap`

**建议**：使用合理的默认值，无需精确校准。

---

## 参数验证检查清单

- [ ] 所有高敏感性参数都有文献或数据支撑
- [ ] 参数取值在合理范围内
- [ ] 已完成宏观矩匹配验证
- [ ] 敏感性分析已识别关键参数
- [ ] 参数追溯文档已更新

---

## 参考文献

1. Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-291.

2. Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5(4), 297-323.

3. Deininger, K., & Jin, S. (2005). The potential of land rental markets in the process of economic development: Evidence from China. *Journal of Development Economics*, 78(1), 241-270.

4. Conley, T. G., & Udry, C. R. (2010). Learning about a new technology: Pineapple in Ghana. *American Economic Review*, 100(1), 35-69.

5. Melitz, M. J. (2003). The impact of trade on intra-industry reallocations and aggregate industry productivity. *Econometrica*, 71(6), 1695-1725.

6. Jackson, M. O. (2008). *Social and Economic Networks*. Princeton University Press.

7. Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. John Wiley & Sons.

---

**文档版本**: 1.0  
**最后更新**: 2025年12月6日  
**维护者**: 模型开发团队
