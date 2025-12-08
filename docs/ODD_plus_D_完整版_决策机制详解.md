# 气候政策下农业土地再配置多智能体模型：ODD+D 完整文档

**版本**: v2.0  
**日期**: 2025-11-26  
**作者**: 研究团队  
**基于**: ODD+D (Overview, Design concepts, Details + Decision) 协议

---

## 目录

1. [目的（Purpose）](#1-目的purpose)
2. [实体、状态变量与尺度（Entities, State Variables, Scales）](#2-实体状态变量与尺度entities-state-variables-scales)
3. [过程概览与日程（Process Overview and Scheduling）](#3-过程概览与日程process-overview-and-scheduling)
4. [设计概念（Design Concepts）](#4-设计概念design-concepts)
5. [细节（Details）](#5-细节details)
6. **【核心】决策机制详解（Decision Mechanisms）**](#6-核心决策机制详解decision-mechanisms)
7. [经验支撑（Empirical Grounding）](#7-经验支撑empirical-grounding)
8. [验证与校准（Validation and Calibration）](#8-验证与校准validation-and-calibration)

---

## 1. 目的（Purpose）

### 1.1 研究问题

本模型旨在回答以下核心政策问题：

1. **土地流转效率问题**：在粮价托底与种粮补贴存在时，土地是否仍能有效从低效率经营者转移到高效率主体？
2. **区域政策差异化**：差异化区域政策能否防止土地高度集中或碎片化？
3. **气候韧性**：气候冲击下的粮食安全与农户福利能否兼顾？
4. **政策传导机制**：不同补贴形式（脱钩/挂钩、现金/实物、按面积/按产量）如何影响农户生计策略演化与企业扩张决策？

### 1.2 模型定位

- **理论贡献**：整合前景理论、Markowitz 投资组合理论、社会学习理论与土地市场理论，构建农户-企业-政府三方博弈框架
- **实证价值**：通过 SMM 校准，复现中国农村土地流转率、收入基尼系数等典型事实
- **政策工具**：支持反事实模拟，评估不同政策场景下的系统涌现行为

---

## 2. 实体、状态变量与尺度（Entities, State Variables, Scales）

### 2.1 智能体类型

| 实体 | 数量 | 关键状态变量 | 数据来源 |
|------|------|-------------|---------|
| **农户** `FarmerAgent` | 300 | `land_patches`, `livelihood_strategy`, `planned_crop`, `capital`, `social_network`, `transaction_neighbors`, `reference_point`, `risk_aversion`, `off_farm_income` | CLES 2020-2021、国家统计局 |
| **企业** `AgriculturalEnterpriseAgent` | 10-30 | `enterprise_type`, `crop_portfolio`, `capital`, `efficiency_factor`, `target_return_rate`, `risk_aversion`, `leased_patches` | Melitz (2003) 假设、企业调查 |
| **政府** `GovernmentAgent` | 1 | `grain_price_floor`, `grain_subsidies_rate`, `subsidy_type`, `subsidy_target`, `regional_policy_assignments` | 政策文件、`policy_params.json` |
| **地块** `LandPatch` | 300-500 | `area`, `soil_quality`, `rental_price`, `leased_to`, `owner`, `pos` | 土地二调、县级耕地质量分级 |
| **环境** `Environment` | 1 | `market_prices`, `mechanism_coordinator`, `policy_params` | `config/master_config.yaml` |

### 2.2 空间与时间尺度

- **空间尺度**：50×50 栅格，代表典型平原县域（约 2500 平方公里）
- **时间尺度**：年度时间步，每个时间步代表一个完整的农业生产周期
- **网络结构**：
  - **社会网络**：平均度 8，基于地理邻近性与社会相似性构建
  - **交易网络**：平均度 4，从社会网络中派生，用于局部土地匹配

---

## 3. 过程概览与日程（Process Overview and Scheduling）

### 3.1 年度决策周期

每个时间步（年度）的执行顺序：

```
1. 环境阶段（Environment Phase）
   ├─ 政府发布政策（announce_policy）
   ├─ 环境更新政策参数（update_policy）
   └─ 市场更新价格（update_market_prices）

2. 土地流转阶段（Land Transfer Phase）
   ├─ 局部土地流转（Local Transfer）
   │  ├─ 农户在 transaction_neighbors 中匹配
   │  ├─ 基于 Deininger & Jin (2005) 意愿函数
   │  └─ MechanismCoordinator.land_transfer 计算成功概率
   └─ 中央土地市场（Central Clearing House）
      ├─ 收集剩余供需
      ├─ 企业/农户出价
      └─ 市场出清（clear_market）

3. 主体决策阶段（Agent Decision Phase）
   ├─ 农户决策（FarmerAgent.step）
   │  ├─ 作物选择博弈（crop_choice_game）
   │  ├─ 劳动力分配（allocate_labor）
   │  ├─ 生产决策（make_production_decisions）
   │  ├─ 收获与结算（harvest_and_settle）
   │  ├─ 消费决策（make_consumption_decision）
   │  └─ 生计策略演化（livelihood_strategy_evolution）
   └─ 企业决策（AgriculturalEnterpriseAgent.step）
      ├─ 作物选择博弈（crop_choice_game）
      ├─ 生产决策（make_production_decisions）
      ├─ 收获与结算（harvest_and_settle）
      └─ 状态更新与策略调整（update_state_and_adapt_strategy）

4. 政府后续阶段（Government Follow-up）
   ├─ 补贴发放（distribute_subsidies）
   └─ 政策评估与调整（evaluate_and_adjust_policy）

5. 数据收集阶段（Data Collection）
   └─ DataCollector 收集所有指标
```

### 3.2 调度器

- **类型**：`RandomActivation`（随机激活）
- **目的**：消除先发优势，确保所有智能体在统计上平等
- **替代方案**：`StagedActivation`（分阶段激活，已弃用）

### 3.3 顶层架构与数据流

```mermaid
flowchart LR
   subgraph Policy
      GOV[GovernmentAgent\n政策设定/评估]
   end

   subgraph Environment
      ENV[Environment\n政策&市场容器]
      LCH[LandClearingHouse]
      LAB[LaborMarket]
      SERV[ServiceMarket]
   end

   subgraph Agents
      FARM[FarmerAgents\n(家庭农户)]
      ENT[AgriculturalEnterpriseAgents\n(新型经营主体)]
      COL[VillageCollective &\nServiceProviders]
   end

   LAND[LandPatches]
   DATA[DataCollector\n指标采集]

   GOV -->|政策参数| ENV
   ENV -->|价格/补贴| FARM
   ENV -->|价格/补贴| ENT
   ENV -->|价格/补贴| COL

   FARM <--> LAB
   ENT <--> LAB

   FARM <--> SERV
   COL --> SERV

   FARM <--> LCH
   ENT <--> LCH

   FARM --> LAND
   ENT --> LAND
   LAND --> ENV

   GOV -->|补贴发放/评估反馈| FARM
   GOV -->|政策再校准| ENT
   ENV --> DATA
   FARM --> DATA
   ENT --> DATA
   LAB --> DATA
   LCH --> DATA
   SERV --> DATA
```

**解读**：

- **政策层**：政府以年度为周期设置最低收购价、补贴结构及区域差异化策略，通过 `Environment` 广播至全部主体。
- **环境层**：`Environment` 协调价格形成、政策参数与三大市场（土地、劳动力、服务），并保证流转、雇佣、托管三条通路并行。
- **主体层**：农户、企业、村集体/服务商在三个市场中博弈；农户与企业双向影响土地与劳动力市场，服务商提供替代性的规模化服务。
- **资源层**：`LandPatch` 记录权属/租赁状态，随交易更新并影响下一周期产出。
- **观测层**：`DataCollector` 汇总宏观/微观指标，反馈给政府与校准脚本，确保仿真闭环。

### 3.4 主体输入输出与决策要点

| 主体 | 输入信息 | 决策/行为 | 输出与反馈 |
|------|----------|-----------|------------|
| 农户 `FarmerAgent` | 政策参数、市场价格、社会网络信号、土地状态、服务供给 | 前景理论下的作物选择、劳动力分配、生计策略演化、土地出租/承租、服务采购 | 土地经营面积、产量/收入、劳动力供给、服务需求、土地流转意愿，写回 `LandPatch`、`LaborMarket`、`ServiceMarket` |
| 企业 `AgriculturalEnterpriseAgent` | 市场价格统计、政策激励、资本约束、效率分布 | Markowitz 组合优化、土地扩张、劳动力雇佣、托管外包 | 大宗土地需求、劳动需求、产量与资本回流，写回 `LandClearingHouse`、`LaborMarket` |
| 政府 `GovernmentAgent` | `DataCollector` 指标、财政约束、网络密度 | 场景选择、补贴发放、区域策略、政策评估与再校准 | 政策参数推送给 `Environment`，补贴直接写入主体资产，新的政策在下一周期生效 |
| 服务商/村集体 | 农户需求、设备容量、托管收益 | 服务定价、合同匹配、产能调度 | 托管合同、服务覆盖率反馈至 `ServiceMarket` 和 `DataCollector` |
| 土地清算所 & 劳动力/服务市场 | 主体提交的供需报价、约束条件 | 匹配算法（局部+中心化）、工资/租金/服务费结算 | 成交结果写回主体状态并更新市场均衡，供政府和分析脚本使用 |

### 3.5 自洽性与闭环检查

1. **政策闭环**：政府依据 `DataCollector`（粮食产量、收入、财政支出）调整补贴与价格地板，新的政策在下一年度通过 `Environment` 扩散，形成"观测 → 决策 → 实施 → 再观测"闭环。
2. **资源闭环**：地块状态 (`LandPatch`) 记录承租者与租期，土地交易、托管与退出都会实时更新，避免"幽灵土地"；同一块地的属性在生产、市场和统计模块中共享。
3. **市场闭环**：土地、劳动力、服务三大市场在每个时间步按固定顺序出清，交易结果直接影响主体下一阶段的资本、劳动力与生产能力，随后反馈到价格和政策评估。
4. **信息闭环**：农户与企业通过社会/交易网络和贝叶斯更新共享局部信息，政府拥有全局信息；不同层级的信息流互补，避免信息断裂。
5. **校准闭环**：`DataCollector` → 校准/诊断脚本（`tools/diagnostics.py`）→ 参数/结构调整 → 回写到 `config` 文件与政策场景，实现模型-数据一致性维护。

依托上述闭环，模型在主体决策、市场机制与政策反馈之间不存在断点，可支撑专家审阅下的自洽性要求。

---

## 4. 设计概念（Design Concepts）

### 4.1 基本理论

| 理论 | 应用场景 | 文献支撑 |
|------|---------|---------|
| **前景理论（Prospect Theory）** | 农户作物选择、风险感知 | Kahneman & Tversky (1979) |
| **Markowitz 投资组合理论** | 企业作物配置 | Markowitz (1952) |
| **Melitz 异质性模型** | 企业效率分布 | Melitz (2003) |
| **社会学习理论** | 信息扩散、技术采纳 | Bandura (1977) |
| **贝叶斯更新** | 信念形成、风险感知 | DeGroot (1970) |
| **土地市场理论** | 流转概率计算 | Deininger & Jin (2005) |

### 4.2 适应性与目标

- **农户**：维持最低收入阈值，通过生计策略演化适应环境变化
- **企业**：追求目标收益率（15-18%），通过投资组合优化与扩张策略调整适应市场
- **政府**：兼顾粮食安全、农户福利与财政约束，通过政策评估与调整优化目标函数

### 4.3 交互机制

- **信息网络**：用于观察邻居收益、作物选择，触发社会学习
- **交易网络**：用于土地匹配，基于信任强度决定交易可行性
- **市场交互**：价格形成基于供需关系，支持内生价格机制
- **政策交互**：政府根据网络指标（区域交易密度）切换政策场景，实现差异化投放

### 4.4 随机性

- 初始年龄、资本、土地面积、社交距离、政策冲击均含随机项
- 生产过程中的产量波动（正态分布，标准差 10%）
- 决策过程中的探索概率（5-15%）

### 4.5 观测指标

| 指标类别 | 具体指标 | 用途 |
|---------|---------|------|
| **土地结构** | `Land_Transfer_Rate`, `Land_Concentration_Top10`, `Land_Fragmentation_Index` | 验证土地市场效率 |
| **价格动态** | `Grain_Price`, `Cash_Crop_Price`, `Average_Land_Rental_Price` | 验证价格形成机制 |
| **收入分配** | `Income_Gini_Coefficient`, `Average_Income`, `Subsidy_Incidence_Ratio` | 验证政策效果 |
| **生计演化** | `Livelihood_Strategy_Distribution`, `Livelihood_Transition_Rate` | 验证策略演化机制 |

---

## 5. 细节（Details）

### 5.1 初始化

1. **农户初始化**：
   - 根据 `initial_conditions` 或分布为 300 户农户分配 1-3 块地
   - 每块地 3-15 亩，赋予土壤质量与坐标
   - 初始生计策略：70% 纯农户，25% 兼业农户，5% 非农户

2. **企业初始化**：
   - 10-30 家企业，基于企业类型分布（大型家庭农场 60%，农业公司 25%，合作社 15%）
   - 资本规模：对数正态分布（100-2000 万元）
   - 效率因子：正态分布（1.0-2.0）

3. **网络构建**：
   - 构建社会网络（综合地理距离和社会相似度）
   - 派生交易可行性网络（平均度 4）
   - 计算区域网络密度，存入 `model.network_metrics`

### 5.2 输入数据

- `config/master_config.yaml`：实验设定、价格弹性、农户/企业参数
- `config/policy_params.json`：政府政策库（多种场景）
- `analysis/batch_simulation.py`/`core/validation.py`：历史矩（CLES、年鉴）用于校准与验证

### 5.3 子过程

#### 5.3.1 局部土地流转

- **触发条件**：农户在 `transaction_neighbors` 中寻找匹配对象
- **意愿计算**：基于 Deininger & Jin (2005) 的转入/转出意愿函数
- **成功概率**：`MechanismCoordinator.land_transfer` 计算，考虑技术差、土地质量、补贴、网络强度
- **结果**：成功匹配的地块直接修改 `LandPatch.leased_to` 与租金，记录在 `model.local_transfer_records`

#### 5.3.2 中央土地市场

- **触发条件**：未被局部匹配的地块进入清算所
- **出价机制**：企业/农户根据盈利预期出价
- **市场出清**：`LandClearingHouse.clear_market` 按报价出清

#### 5.3.3 验证与分析

- `core/validation.py` 收集指标，与实证序列做 out-of-sample 检验
- `analysis/batch_simulation.py` 输出结构矩以供 SMM 使用

---

## 6. 【核心】决策机制详解（Decision Mechanisms）

### 6.1 农户决策机制（FarmerAgent）

#### 6.1.1 信息集（Information Set）

农户在决策时可获得的信息：

| 信息类别 | 具体内容 | 来源 |
|---------|---------|------|
| **个人状态** | `capital`, `land_patches`, `livelihood_strategy`, `total_income`, `reference_point`, `risk_aversion` | 自身状态变量 |
| **价格信息** | `market_price_grain`, `market_price_cash_crop`, `price_history`（过去 5 年） | `Environment.market_prices` |
| **政策信息** | `grain_price_floor`, `grain_subsidies_rate`, `subsidy_type`, `subsidy_target` | `Environment.policy_params` |
| **社会信息** | 邻居的 `planned_crop`, `total_income`, `crop_choice_history` | `social_network` 中的邻居 |
| **市场预期** | `grain_belief`, `cash_crop_belief`（贝叶斯更新后的信念） | `bayesian_learning_update` |

**信息限制**：
- 农户无法直接观察邻居的生产成本（信息不对称）
- 价格历史记忆有限（5 年）
- 社会网络信息可能不完整（网络稀疏性）

#### 6.1.2 目标函数（Objective Function）

农户的目标是**最大化前景理论效用**，而非传统期望效用：

$$V(\Delta x) = \begin{cases}
(\Delta x)^\alpha & \text{if } \Delta x \ge 0 \\
-\lambda(-\Delta x)^\beta & \text{if } \Delta x < 0
\end{cases}$$

其中：
- $\Delta x = \text{收入} - \text{参考点}$（相对于参考点的损益）
- $\lambda > 1$（损失厌恶系数，默认 2.25）
- $\alpha, \beta \in (0, 1)$（价值函数曲率，默认 0.88）

**目标的多层次性**：
1. **短期目标**：维持收入高于参考点（避免损失）
2. **中期目标**：通过生计策略演化适应环境变化
3. **长期目标**：积累资本，提升社会地位

#### 6.1.3 决策规则（Decision Rules）

##### 规则 1：作物选择博弈（`crop_choice_game`）

**决策流程**：

```
1. 更新动态机制
   ├─ update_reference_point()      # 更新参考点（个人历史 + 社会比较）
   ├─ update_risk_preference()       # 更新风险偏好（基于成功/失败连续次数）
   └─ update_prospect_theory_parameters()  # 更新前景理论参数（基于收入分位数）

2. 获取信息
   ├─ 市场价格（grain_price, cash_crop_price）
   ├─ 政策参数（grain_price_floor, grain_subsidies_rate）
   └─ 价格历史（price_history，过去 5 年）

3. 贝叶斯学习更新
   ├─ 收集证据（自身和邻居的收入数据）
   ├─ 更新信念（grain_belief, cash_crop_belief）
   └─ 计算期望收益

4. 社会学习
   ├─ observe_neighbors()           # 观察邻居的作物选择和收益
   ├─ calculate_social_influence() # 计算社会影响（同质性加权）
   └─ 计算社会影响调整项

5. 前景理论分析
   ├─ generate_crop_scenarios()     # 生成作物情景（收益分布）
   ├─ decision_utility()            # 计算前景理论效用
   └─ 计算基础效用（grain_utility_base, cash_crop_utility_base）

6. 综合决策
   ├─ 权重分配：
   │  ├─ rational_weight = 0.4       # 前景理论效用权重
   │  ├─ belief_weight = 0.3        # 贝叶斯信念权重
   │  └─ social_weight = 0.3        # 社会学习权重
   ├─ 计算最终效用：
   │  ├─ grain_utility_final = rational_weight * grain_utility_base + 
   │  │                        belief_weight * grain_belief_bonus + 
   │  │                        social_weight * grain_social_bonus
   │  └─ cash_crop_utility_final = ...
   └─ 选择效用更高的作物

7. 探索与调整
   ├─ 动态探索概率（5-15%，基于价格差异）
   ├─ 羊群效应（邻居经济作物占比 > 10% 时自动切换）
   └─ 价格差异强制切换（价格差 > 0.5 元/斤）
```

**关键参数**：
- `belief_weight = 0.3`：贝叶斯信念权重
- `social_weight = 0.3`：社会学习权重
- `rational_weight = 0.4`：前景理论效用权重
- `explore_prob = 0.05 + 0.2 * price_diff / grain_price`：动态探索概率

##### 规则 2：劳动力分配（`allocate_labor`）

**决策规则**：
- 根据生计策略分配劳动力：
  - **纯农户**：75-90% 农业劳动
  - **兼业农户**：30-60% 农业劳动
  - **非农户**：5-20% 农业劳动
- 非农工资水平根据生计策略调整：
  - **纯农户**：0.6-0.9（较低）
  - **兼业农户**：0.8-1.2（中等）
  - **非农户**：1.0-1.5（较高）

##### 规则 3：生计策略演化（`livelihood_strategy_evolution`）

**决策规则**：

```
1. 计算农业收入占比
   agricultural_income_ratio = agricultural_income / total_income

2. 判断转换条件
   ├─ Pure_Farmer → Pluriactive：
   │  └─ 条件：agricultural_income_ratio < 0.6 且连续 2 年
   ├─ Pluriactive → Non_Farmer：
   │  └─ 条件：agricultural_income_ratio < 0.3 且连续 2 年
   └─ 反向转换（基于收入恢复）

3. 社会网络影响
   ├─ 观察邻居的生计策略分布
   ├─ 计算社会压力（conformity_tendency）
   └─ 调整转换概率

4. 转换冷却期
   └─ strategy_switch_cooldown（防止频繁切换）
```

**关键参数**：
- `agricultural_income_ratio_threshold = 0.6`（纯农户→兼业）
- `agricultural_income_ratio_threshold = 0.3`（兼业→非农户）
- `consecutive_years_required = 2`（连续年数要求）
- `strategy_switch_cooldown = 3`（转换冷却期，年）

#### 6.1.4 适应性（Adaptation）

农户的适应性体现在：

1. **动态参考点更新**：
   - 基于个人历史收入（线性趋势预测）
   - 基于社会网络收入（社会比较）
   - 融合权重：`internal_orientation`（0-1，决定"向内看"还是"向外看"）

2. **风险偏好动态调整**：
   - 连续成功次数增加 → 风险厌恶降低
   - 连续失败次数增加 → 风险厌恶提高
   - 调整速率：`risk_learning_rate = 0.1`

3. **贝叶斯信念更新**：
   - 根据自身和邻居的收入数据更新对作物收益的信念
   - 网络强度作为权重（关系越强，影响越大）

4. **社会学习**：
   - 观察邻居的作物选择和收益
   - 同质性加权（倾向于模仿资源禀赋相似但稍稍成功的同类）
   - 可见性偏差（产量可见，成本不可见）

---

### 6.2 企业决策机制（AgriculturalEnterpriseAgent）

#### 6.2.1 信息集（Information Set）

企业在决策时可获得的信息：

| 信息类别 | 具体内容 | 来源 |
|---------|---------|------|
| **企业状态** | `capital`, `efficiency_factor`, `target_return_rate`, `risk_aversion`, `leased_patches` | 自身状态变量 |
| **价格信息** | `market_price_grain`, `market_price_cash_crop`, 价格历史统计（均值、方差、协方差） | `Environment.market_prices` |
| **政策信息** | `grain_price_floor`, `grain_subsidies_rate` | `Environment.policy_params` |
| **市场信号** | 价格差异（`cash_crop_price - grain_price`） | 计算得出 |
| **历史业绩** | `profit_history`, `roi_history`, `performance_trend` | 自身历史记录 |

**信息优势**：
- 企业具有更强的市场信息获取能力（`market_info_quality = 0.6-1.0`）
- 能够进行多期规划（不同于农户的单期决策）

#### 6.2.2 目标函数（Objective Function）

企业的目标是**最大化投资组合的期望收益，同时最小化风险**（Markowitz 投资组合理论）：

$$\max \quad \mu'w - \frac{\lambda}{2} w'\Sigma w$$

其中：
- $\mu$：期望收益向量（各作物的期望收益率）
- $w$：投资组合权重向量（各作物的分配比例）
- $\Sigma$：收益协方差矩阵（作物收益的风险）
- $\lambda$：风险厌恶系数（`risk_aversion`，企业类型相关）

**约束条件**：
- $\sum w_i = 1$（权重和为 1）
- $w_i \in [0, 1]$（每个作物权重在 0-1 之间）

#### 6.2.3 决策规则（Decision Rules）

##### 规则 1：作物选择博弈（`crop_choice_game`）

**决策流程**：

```
1. 获取市场价格数据
   ├─ grain_price_floor（政策价格地板）
   ├─ grain_subsidies_rate（补贴率）
   └─ market_price_cash_crop（经作市场价格）

2. 计算作物统计量
   ├─ calculate_crop_statistics()
   │  ├─ 期望收益（基于历史价格均值）
   │  ├─ 收益方差（基于历史价格波动）
   │  └─ 收益协方差（基于历史价格相关性）
   └─ 构建期望收益向量和协方差矩阵

3. Markowitz 投资组合优化
   ├─ optimize_crop_portfolio()
   │  ├─ 目标函数：max(μ'w - (λ/2) * w'Σw)
   │  ├─ 约束条件：Σw = 1, w ∈ [0, 1]
   │  └─ 求解方法：scipy.optimize.minimize (SLSQP)
   └─ 得到最优权重（grain_weight, cash_crop_weight）

4. 决定主要作物类型
   └─ planned_crop = GRAIN if grain_weight >= cash_crop_weight else CASH_CROP

5. 探索与调整
   ├─ 动态探索概率（基于价格差异）
   ├─ 价格差异强制切换（价格差 > 0.5 元/斤）
   └─ 随机探索（15% 概率）
```

**关键参数**：
- `risk_aversion`：风险厌恶系数（企业类型相关）
  - 大型家庭农场：1.5 ± 0.3
  - 农业公司：1.0 ± 0.3
  - 合作社：1.3 ± 0.3
- `default_correlation = 0.3`：默认作物收益相关性
- `explore_prob = 0.05 + 0.2 * price_diff / grain_price`：动态探索概率

##### 规则 2：土地扩张决策（`participate_land_rental_market`）

**决策流程**：

```
1. 基础参与概率（基于企业类型）
   ├─ 大型家庭农场：0.6
   ├─ 农业公司：0.8
   └─ 合作社：0.5

2. 市场信号影响
   ├─ 价格差异因子（price_factor）
   │  └─ price_diff / grain_price * price_factor_weight (0.2)
   └─ 价格差异越大，扩张意愿越强

3. 效率优势影响
   ├─ efficiency_advantage = (efficiency_factor - 1.0) * efficiency_factor_weight (0.3)
   └─ 效率越高，扩张意愿越强

4. 资本充足度影响
   ├─ capital_availability = capital / required_capital_per_100mu
   └─ 资本越多，扩张意愿越强

5. 计算最终参与概率
   └─ participation_prob = base_prob + price_factor + efficiency_factor + capital_factor

6. 决定是否参与
   └─ if random() < participation_prob: 参与土地市场
```

**关键参数**：
- `base_participation_probability`：基础参与概率（企业类型相关）
- `factor_weights`：各因子权重
  - `price_factor = 0.2`
  - `efficiency_factor = 0.3`
  - `capital_factor = 0.35`
- `required_capital_per_100mu = 100000`：每 100 亩所需资本（10 万元）

#### 6.2.4 适应性（Adaptation）

企业的适应性体现在：

1. **投资组合动态调整**：
   - 根据市场价格变化调整作物组合
   - 基于历史价格统计更新期望收益和风险

2. **目标回报率调整**：
   - 根据历史业绩（`performance_trend`）调整目标回报率
   - 业绩好时提高目标，业绩差时降低目标

3. **扩张策略调整**：
   - 根据市场信号（价格差异）调整扩张意愿
   - 根据资本充足度调整扩张规模

4. **学习机制**：
   - `learning_rate = 0.1`：学习速率
   - `aggressive_threshold = 1.2`：激进阈值（ROI > 1.2 时更激进）
   - `conservative_threshold = 0.8`：保守阈值（ROI < 0.8 时更保守）

---

### 6.3 政府决策机制（GovernmentAgent）

#### 6.3.1 信息集（Information Set）

政府在决策时可获得的信息：

| 信息类别 | 具体内容 | 来源 |
|---------|---------|------|
| **宏观指标** | `total_grain_production`, `average_farmer_income`, `land_transfer_rate`, `income_gini_coefficient` | `DataCollector` |
| **网络指标** | `network_metrics`（区域交易密度） | `model.network_metrics` |
| **政策历史** | `policy_history`（过去政策效果） | 自身历史记录 |
| **财政约束** | `annual_budget`（年度预算） | 配置参数 |

**信息优势**：
- 政府具有全局信息（所有智能体的状态）
- 能够进行多目标优化（粮食安全、农户福利、财政约束）

#### 6.3.2 目标函数（Objective Function）

政府的目标是**最大化社会福利函数**：

$$\max_{S} \quad U_G = w_s \cdot Q_G(S) + w_f \cdot W_F(S) - C(S)$$

其中：
- $w_s$：粮食安全权重（默认 0.4）
- $w_f$：农户福利权重（默认 0.4）
- $w_b$：财政约束权重（默认 0.2）
- $Q_G(S)$：粮食总产量（政策 $S$ 的函数）
- $W_F(S)$：农户总福利（政策 $S$ 的函数）
- $C(S)$：财政总支出（政策 $S$ 的函数）

#### 6.3.3 决策规则（Decision Rules）

##### 规则 1：政策发布（`announce_policy`）

**决策流程**：

```
1. 读取政策场景
   ├─ 从 policy_params.json 读取场景配置
   ├─ 应用 policy.overrides（动态覆盖）
   └─ 确定活动场景（active_scenario_id）

2. 设置政策参数
   ├─ grain_price_floor（粮食最低收购价）
   ├─ grain_subsidies_rate（粮食补贴率）
   ├─ subsidy_type（补贴类型：area_based / output_based）
   ├─ subsidy_target（补贴对象：operator / owner）
   └─ 其他政策参数（delivery_mode, payment_lag, cash_share, ...）

3. 区域差异化策略
   ├─ _assign_regional_policies()
   │  ├─ 基于 network_metrics（区域交易密度）
   │  ├─ 低密度区域（< 0.25）→ high_support 场景
   │  ├─ 中密度区域（0.25-0.5）→ baseline 场景
   │  └─ 高密度区域（> 0.5）→ low_support 场景
   └─ 记录区域政策分配（regional_policy_assignments）

4. 传递政策信息
   ├─ update_policy()（更新环境政策参数）
   └─ set_active_policy()（设置活动政策）
```

**关键参数**：
- `scenario_id`：政策场景 ID（从 `policy_params.json` 读取）
- `network_density_threshold = 0.25`：网络密度阈值（用于区域差异化）

##### 规则 2：补贴发放（`distribute_subsidies`）

**决策规则**：

```
1. 确定补贴对象
   ├─ 如果 subsidy_target == "operator"：补贴给耕作者（租地者）
   └─ 如果 subsidy_target == "owner"：补贴给土地所有者

2. 计算补贴金额
   ├─ 如果 subsidy_type == "area_based"：
   │  └─ subsidy = base_subsidy_per_mu * operated_area * grain_subsidies_rate
   └─ 如果 subsidy_type == "output_based"：
      └─ subsidy = base_subsidy_per_mu * grain_yield * grain_subsidies_rate

3. 发放补贴
   ├─ 现金补贴（cash_share = 1.0）：直接增加 capital
   ├─ 实物补贴（cash_share < 1.0）：增加 in_kind_support_balance
   └─ 记录补贴历史（subsidy_history）
```

**关键参数**：
- `subsidy_type`：补贴类型（area_based / output_based）
- `subsidy_target`：补贴对象（operator / owner）
- `cash_share`：现金比例（0-1）
- `payment_lag`：支付滞后（年）

##### 规则 3：政策评估与调整（`evaluate_and_adjust_policy`）

**决策规则**：

```
1. 收集宏观指标
   ├─ total_grain_production（粮食总产量）
   ├─ average_farmer_income（农户平均收入）
   ├─ land_transfer_rate（土地流转率）
   ├─ income_gini_coefficient（收入基尼系数）
   └─ fiscal_cost（财政成本）

2. 计算目标函数值
   └─ U_G = w_s * Q_G + w_f * W_F - w_b * C

3. 政策调整（简化版）
   ├─ 如果粮食产量 < 目标：提高 grain_subsidies_rate
   ├─ 如果农户收入 < 目标：提高 grain_subsidies_rate
   ├─ 如果财政成本 > 预算：降低 grain_subsidies_rate
   └─ 记录政策调整历史
```

**关键参数**：
- `policy_goal_weights`：目标权重（w_s, w_f, w_b）
- `adjustment_rate = 0.1`：调整速率（每次调整 10%）

#### 6.3.4 适应性（Adaptation）

政府的适应性体现在：

1. **区域差异化策略**：
   - 根据网络指标（区域交易密度）动态调整政策场景
   - 低密度区域自动切换到 `high_support` 场景

2. **政策评估与调整**：
   - 根据宏观指标（粮食产量、农户收入、财政成本）调整政策参数
   - 调整速率：`adjustment_rate = 0.1`（每次调整 10%）

3. **多场景支持**：
   - 支持从 `policy_params.json` 读取多种政策场景
   - 支持 `policy.overrides` 动态覆盖（用于敏感性分析）

---

## 7. 经验支撑（Empirical Grounding）

### 7.1 理论依据

| 理论 | 应用 | 文献 |
|------|------|------|
| **前景理论** | 农户作物选择、风险感知 | Kahneman & Tversky (1979) |
| **Markowitz 投资组合理论** | 企业作物配置 | Markowitz (1952) |
| **Melitz 异质性模型** | 企业效率分布 | Melitz (2003) |
| **土地市场理论** | 流转概率计算 | Deininger & Jin (2005) |
| **社会学习理论** | 信息扩散、技术采纳 | Bandura (1977) |
| **贝叶斯更新** | 信念形成、风险感知 | DeGroot (1970) |

### 7.2 实证数据

| 参数类别 | 数值 | 数据来源 |
|---------|------|---------|
| **价格参数** | 粮价 1.25 元/斤、经作价 1.8 元/斤 | 2020 年农业农村部统计 |
| **需求弹性** | 粮食 -0.3，经作 -0.5 | 《农产品供求年鉴》 |
| **网络密度阈值** | 0.25（交易网络低密度警戒线） | 江浙典型县的土地互助组调查 |
| **验证矩** | 土地流转率≈36%、平均租金≈850 元/亩 | CLES 与《中国农村统计年鉴》 |
| **土地集中度** | 前 10% 经营主体占比≈0.4 | CLES 与《中国农村统计年鉴》 |
| **碎片化指数** | 拥有土地的农户比例≈0.75 | CLES 与《中国农村统计年鉴》 |

### 7.3 校准方法

- **方法**：Simulated Method of Moments (SMM)
- **优化算法**：Differential Evolution（差分进化算法）
- **目标矩**：
  - 土地流转率
  - 平均土地租金
  - 土地集中度（前 10%）
  - 碎片化指数
  - 收入基尼系数
  - 平均收入
- **数据分割**：训练集（校准）与验证集（测试）严格分离

---

## 8. 验证与校准（Validation and Calibration）

### 8.1 验证标准

#### 8.1.1 微观验证（Micro-validation）

- **农户行为**：风险厌恶行为是否符合前景理论预测？
- **企业行为**：投资组合优化是否符合 Markowitz 理论？
- **社会学习**：信息扩散是否符合 S 型曲线？

#### 8.1.2 宏观验证（Macro-validation）

模型必须能够复现以下典型事实：

1. **流转率的非线性动态**：随非农工资上涨而上升，但存在上限
2. **经营规模的双峰分布**：小规模自给农户与大规模商业大户并存，"中等规模陷阱"显现
3. **租金粘性**：农产品价格下跌时，土地租金往往滞后下跌
4. **价格动态**：粮食价格稳定（价格地板保护），经作价格波动较大

### 8.2 校准流程

1. **参数识别**：识别需要校准的关键参数
2. **目标矩选择**：选择与实证数据匹配的目标矩
3. **优化算法**：使用 Differential Evolution 搜索最优参数组合
4. **验证**：在验证集上测试模型性能
5. **敏感性分析**：进行全局敏感性分析（Sobol indices）

### 8.3 敏感性分析

模型必须附带详尽的全局敏感性分析，指明哪些参数对结果影响最大：

- **农户参数**：`risk_aversion`, `social_learning_weight`, `belief_weight`
- **企业参数**：`risk_aversion`, `target_return_rate`, `efficiency_factor`
- **政策参数**：`grain_subsidies_rate`, `grain_price_floor`
- **市场参数**：`price_elasticity`, `price_stickiness`

---

## 附录 A：关键代码位置

| 功能 | 文件路径 | 关键方法 |
|------|---------|---------|
| **农户作物选择** | `agents/farmer_agent.py` | `crop_choice_game()` |
| **农户生计演化** | `agents/farmer_agent.py` | `livelihood_strategy_evolution()` |
| **企业投资组合优化** | `agents/enterprise_agent.py` | `optimize_crop_portfolio()` |
| **企业土地扩张** | `agents/enterprise_agent.py` | `participate_land_rental_market()` |
| **政府政策发布** | `agents/government_agent.py` | `announce_policy()` |
| **价格形成机制** | `core/environment.py` | `_update_grain_market()`, `_update_cash_crop_market()` |
| **土地流转机制** | `core/mechanisms.py` | `land_transfer()` |
| **模型调度** | `core/model.py` | `step()` |

---

## 附录 B：配置参数说明

详见 `config/master_config.yaml` 和 `config/policy_params.json`。

---

**文档维护**：本 ODD+D 文档反映了当前迭代（引入局部土地流转与网络靶向政策）后的模型状态。后续若添加新机制，需同步更新此文件以保持可复现性。


