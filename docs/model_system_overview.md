# 多智能体粮食补贴政策模型：框架与校准说明

## 1. 外部项目与文献借鉴
- **AgriPoliS**（Happe et al., 2006）[^agripolis]: 经典的区域农业结构演化 ABM，强调异质农户、基于利润的土地流转与政策情景实验。启发我们将土地租赁、资本约束与政府干预放入统一的市场撮合框架。
- **SWISSland**（Schmid & Sinabell, 2021）[^swissland]: 瑞士农业部门政策评估模型，突出精细的农场簿记数据和政策模块化；可借鉴其“数据驱动初始化 + 政策脚本化”的流程。
- **MOSAICA**（Berger et al., 2017）[^mosaica]: 聚焦于农业技术扩散的多智能体平台，强调社会网络结构、学习机制与空间异质性；与我们模型中的社会学习、参考点更新直接对应。
- **FarmDyn / AgriMod 套件**（Zimmermann et al., 2020）[^farmdyn]: 混合 MILP 与模拟模型，强调补贴、保险等政策对农户投资/生产的影响。可借鉴其对补贴分类、现金流与资产负债表校准的做法。
- **开源 GitHub 项目 “open-agriculture-abm”**[^github]: 展示了使用 Mesa 实现的作物-市场联动 ABM，提供 CI/CD、自动化参数扫描、可视化管线等工程实践。

> 小结：上述项目共同强调（1）异质主体与行为数据的严密校准、（2）政策模块与市场模块的解耦、（3）社会网络和空间结构的重要性。这与我们现有代码架构（`core/`、`agents/`、`modules/`）完全契合，也为后续工程化提供了模板。

## 2. 现有模型框架与逻辑结构
### 2.1 主框架
- `core/model.py`：`MainModel` 定义年度循环的宏观流程，包括环境更新 → 土地成交 → 调度所有主体 → 数据采集。
- `core/model_builder.py`：按照 `config/master_config.yaml` 初始化政府、300 农户、30 企业以及土地地块，抽样异质参数并构建社会网络。
- `core/environment.py`：集中管理市场价格、工资率、补贴政策；实现基于供需的价格内生化与历史记录缓存。
- `modules/land_clearing_house.py`：双向拍卖的土地清算所，处理租赁供需、成交价格和“村精英”优先权。

### 2.2 智能体层
- `agents/farmer_agent.py`：
  - **首要博弈**：基于前景理论 + 贝叶斯学习 + 社会学习的作物选择。
  - **土地市场**：按照自耕 vs. 出租的收益比较提交报价（已修正为按地块面积计算利润与租金）。
  - **生产与结算**：随机产量、价格乘以产量、成本率、补贴、非农收入，更新资本与生计策略。
- `agents/enterprise_agent.py`：
  - 基于 Markowitz 组合优化作物投资（粮食/经济作物）。
  - 扩张概率与土地竞价与市场价格差关联，并考虑异质风险偏好等参数。
- `agents/government_agent.py`：多政策参数（补贴率、最低收购价、补贴对象）。`Environment.update_policy()` 每期读取政府当前政策。

### 2.3 关键模块
- `core/mechanisms.py`：前景理论、Melitz 异质性、创新扩散、土地流转等机制类，供主体调用。
- `modules/market.py`：目前未接入主循环，但提供工资、土地价格、竞价等可拓展的市场机制。
- `analysis/`：新建 `empirical_calibration.py`，用于读取实证数据构建矩估计（详见第 4 节）。

### 2.4 数据采集
`ModelBuilder._setup_data_collector()` 注册土地经营面积、补贴资本化、收入基尼、企业学习率等关键指标，实现宏观趋势与微观结构的联动。

## 3. 补贴政策背景（《粮食补贴内涵.md》摘要）
- **补贴类型**：耕地地力保护补贴（合并粮食直补、良种补贴、农资综合补贴）、稻谷补贴、种粮大户一次性补贴等，覆盖自有地、流转地、家庭农场。
- **发放标准（案例）**：成都 2023 稻谷补贴 57.45 元/亩 + 16.6 元/亩一次性补贴；北京 2025 耕地地力保护补贴 300 元/亩；四川 2025 种粮大户补贴 100–200 元/亩。
- **发放流程**：农户申报→村级公示→乡镇核查→县级系统录入→财政发放。
- **政策演进**：2009 现金补贴试点 → 2016 三项补贴合并 → 2018 聚焦耕地地力保护，并逐步探索“补农资”“技术服务”形式。
- **实际问题**：基层财政人手/设备不足、成本高、补贴效果易被贩粮商吸收、补贴金额有限等。

> 对模型启示：补贴不仅包含面积系数，还涉及流程滞后、对象区分（种植 vs. 所有权 vs. 经营权）与“补物资”形式。未来可扩展补贴类型枚举、发放时滞、银行通道等机制。

## 4. CLES 数据集结构与校准建议
数据位于 `校准用数据源/`，对应 CLES（China Land Economy Survey）2020-2022 的村/户面板映射。通过脚本写入 `analysis/cles_columns.json` 可得到列名。

### 4.1 文件结构
| 文件 | 特征 | 备注 |
| --- | --- | --- |
| `CLES_2020_2021_household_panel_mapping_label.csv` | 900+ 列 | 第 1 行为中文指标描述，从第 2 行开始为数据；包含 `hid`, `year` 与大量 `a***` ~ `k***` 指标。 |
| `CLES_2020_2021_village_panel_label_mapping.csv` | 1300+ 列 | 覆盖村级资产、公共服务、组织、技术推广、社会网络等。 |
| `CLES_2022_household_mapping_label.csv` | 918 列 | 结构类似 2020–2021，字段编码一致，可用于跨期面板。 |
| `CLES_2022_village_label_mapping.csv` | 1300+ 列 | 与前期村级文件匹配，补充 2022 年数据。 |

编码说明：
- 列头包含中文解释（如 `hid` 一行说明“农户编码，唯一识别码”）→ 需要剥离首行作为元数据表。
- 字段命名遵循主题 + 顺序（`a***`：人口与户属性、`c***`：收入、`d***`：支出、`e***`：土地与生产、`f***`：补贴/保险、`g***`：社会网络、`h***`：公共设施、`k***`：法律与社会资本等）。
- 需要配套代码表才能精确匹配含义，可根据列名安排人工字典或解析问卷文档。

### 4.2 校准思路
1. **数据预处理**  
   - 读取 CSV → 将第 1 行保存为 “变量中文描述” 字典 → 其余行转为数值/分类型数据。  
   - 统一编码（`utf-8`/`gb18030`），处理缺失值与 “99/9999/−1” 等问卷缺省码。
2. **农户主体参数**  
   - `e1**`~`e3**`: 作物面积、单产、化肥/农药使用，可用于初始化地块面积、产量分布、成本参数。  
   - `c***`：收入结构（农业/非农/转移），用于校准生计策略阈值、参考点与风险学习。  
   - `f***`：补贴、保险类型与金额，用于政府补贴率、目标对象、领取概率。  
   - `g101*`：社会网络连接度（向谁咨询、合作社关系），映射至 `social_network_strength`、`information_receptivity`。
3. **村级/环境参数**  
   - `b***`、`c***`（村级）：基础设施、市场距离、信息服务，用于 `Environment` 中的价格粘性、运输成本、外部机会。  
   - `h***`：公共服务、政策推广频率 → 可映射为补贴覆盖率、技术推广成功率。
4. **校准流程**  
   - `analysis/empirical_calibration.py` 可扩展为：读取清洗后的 DataFrame → 计算均值/方差/分位数 → 写入 `config/calibrated_params_enhanced.json`。  
   - 使用 `CalibrationManager.calibrate_from_empirical_data()` 将分布参数注入 `config/master_config.yaml`（例如 lognormal 均值、std）。  
   - 后续可以基于目标矩（如平均补贴金额、土地流转率、社会网络度数）进行 SMM/ABC 优化。

## 5. 校准与补贴联动建议
1. **补贴层**  
   - 结合 CLES `f10**` 字段统计不同类型补贴的覆盖率与金额，映射到 `policy_params.json` 中的 `grain_subsidies_rate`、`base_subsidy_per_mu`。  
   - 引入“经营者与所有者差异”的判别逻辑：根据流转面积 vs. 所有面积（`e101*` vs. `e201*`）决定补贴对象。  
   - 设计按省份/地区的补贴参数包，模拟成都市、北京市、四川丘陵 vs. 平原的差异。
2. **行为层**  
   - 利用收入结构 (`c***`) 校准前景理论参考点（上一年度收入）与贝叶斯学习的成功判别阈值。  
   - 利用社会网络（`g***`）校准 `social_learning_weight`, `information_receptivity`。  
   - 根据风险感知题 (`k***`) 调整 `base_risk_aversion` 与 `risk_learning_rate`。
3. **市场层**  
   - 村级数据中的市场距离、交通状况可用于设置 `transport_cost_rate`、`price_stickiness` 与劳动力市场基准需求。

## 6. 待办与下一步
1. **数据管线**：实现一个 `notebooks/` 或 `analysis/cles_preprocessing.py`，完成 CSV → 清洗 → 变量映射 → JSON 配置。  
2. **政策模块**：在 `policy_params.json` 中添加多种补贴情景（台阶补贴、实物补贴、按户均/面积上限），并在政府代理中实现切换。  
3. **市场机制整合**：评估 `modules/market.py` 与现有 `Environment` 的对接，决定是否采用统一市场机制或分市场处理。  
4. **敏感性分析**：借助 `runner.py` 中的 batch 工具或新增 CLI，实现对补贴率、风险参数、网络密度的多情景实验。  
5. **文档与验证**：结合 README、该文档以及补贴背景资料，形成可提交的技术报告或论文附录。

---

[^agripolis]: Happe, K., Kellermann, K., & Balmann, A. (2006). Agent-based analysis of agricultural policies: an illustration of the AgriPoliS model. *Agricultural Economics*, 35(2).  
[^swissland]: Schmid, E., & Sinabell, F. (2021). SWISSland: An agent-based agricultural sector model for Switzerland. Swiss Federal Office for Agriculture.  
[^mosaica]: Berger, T., Troost, C., & Magliocca, N. R. (2017). Agent-based modelling of agri-food systems. *Annual Review of Resource Economics*, 9.  
[^farmdyn]: Zimmermann, A. et al. (2020). FarmDyn: A flexible and modular integrated model for farm-level economic and policy analysis. *European Review of Agricultural Economics*, 47(2).  
[^github]: 例如 <https://github.com/Open-ABM-agriculture/open-agriculture-abm> 提供的 Mesa 农业 ABM 样板代码与实验脚本。***




