# 气候政策ABM模型运行流程

## 模型架构概览

本项目是一个基于多智能体建模（Agent-Based Modeling, ABM）的气候政策农业经济仿真系统，重点关注粮食补贴政策对农户和农业企业行为的影响。

### 核心设计理念
- **理论驱动开发**：基于成熟的经济学理论（如Melitz企业异质性模型、前景理论等）
- **化繁为简原则**：模型复杂性必须被审慎管理，首要目标是解答核心研究问题
- **模块化设计**：实现模块间的低耦合与模块内的高内聚

## 文件结构

```
从0到1/
├── core/                          # 核心模型逻辑
│   ├── model.py                   # 主模型类（MainModel）
│   └── mechanisms.py              # 博弈机制实现
├── agents/                        # 智能体定义
│   ├── farmer_agent.py           # 农户智能体
│   ├── enterprise_agent.py        # 农业企业智能体
│   └── government_agent.py        # 政府智能体
├── modules/                       # 功能模块
│   ├── land_clearing_house.py    # 土地租赁中央清算所
│   ├── market.py                 # 市场机制
│   └── land_patch.py             # 地块类
├── config/                        # 配置文件
│   ├── scale_configs.py          # 规模配置管理
│   ├── model_params.json         # 模型参数
│   ├── policy_params.json        # 政策参数
│   └── calibrated_params.json    # 校准参数
├── analysis/                      # 分析脚本
│   ├── game_process_analysis.py  # 博弈过程分析（推荐）
│   ├── micro_game_visualization.py # 微观博弈可视化
│   └── detailed_game_analysis.py # 详细博弈分析
├── results/                       # 仿真结果
├── docs/                         # 文档
├── utils/                        # 工具函数
├── run_experiment.py             # 主运行脚本
└── README.md                     # 项目说明
```

## 模型运行流程

### 1. 初始化阶段

#### 1.1 配置加载
```python
# 从config/scale_configs.py加载规模配置
scale_config = get_scale_config("medium")  # 推荐使用medium配置
```

**规模配置选项：**
- `test`: 20农户, 3企业, 15×15网格（快速测试）
- `small`: 100农户, 10企业, 30×30网格（初步验证）
- `medium`: 300农户, 30企业, 50×50网格（推荐配置）
- `large`: 500农户, 50企业, 70×70网格（深入分析）
- `xlarge`: 1000农户, 100企业, 100×100网格（复杂场景）

#### 1.2 模型创建
```python
model = MainModel(
    scale_config=scale_config,
    policy_params={
        "subsidy_policy": {
            "subsidy_type": "area_based",
            "subsidy_target": "owner",
            "subsidy_rate": 0.15
        }
    },
    seed=42
)
```

#### 1.3 智能体初始化
- **农户智能体**：创建N_farmers个农户，分配初始地块
- **农业企业智能体**：创建N_enterprises个企业
- **政府智能体**：创建1个政府智能体
- **地块系统**：创建土地地块，分配给农户

### 2. 年度决策周期

模型严格按照以下五个阶段执行：

#### 2.1 阶段一：政策发布（Government Sets the Rules）
- GovernmentAgent宣布本年度的各项补贴率和价格地板
- 为所有参与者的决策设定外部支付结构

#### 2.2 阶段二：核心博弈 - 作物选择（The Crop Choice Game）
- **核心博弈**：所有潜在的土地经营者必须先决定他们计划种植什么
- 博弈方：所有AgriculturalEnterpriseAgent和FarmerAgent
- 策略选项：种植粮食作物 vs 种植经济作物
- 支付矩阵计算：风险调整后的预期效用
- 决策机制：主体根据其risk_aversion属性选择策略

#### 2.3 阶段三：次要博弈 - 土地租赁市场（The Land Market Game）
- 土地供需双方基于保留价格进行博弈
- 中央清算所机制进行市场出清
- 确定土地租赁价格和交易量

#### 2.4 阶段四：收获与结算（Harvest and Settlement）
- 计算各主体的收入和支出
- 更新资本和收入状态
- 计算补贴分配

#### 2.5 阶段五：生计策略演化（Livelihood Strategy Evolution）
- 农户根据博弈结果动态调整自身类型
- 更新生计策略（Pure_Farmer → Pluriactive → Non_Farmer）
- 调整未来策略和风险偏好

### 3. 数据收集

#### 3.1 模型级数据（model_data.csv）
- Land_Operated_by_Enterprises：企业经营土地面积
- Land_Distribution_by_Livelihood_Type：按生计类型的土地分布
- Gini_Coefficient_of_Land_Operation：土地经营基尼系数
- Subsidy_Incidence_Ratio：补贴影响率
- Income_Gini_Coefficient_Rural：农村收入基尼系数
- Number_of_Pure_Farmers：纯农户数量
- Smallholder_Exit_Rate：小农退出率
- Land_Rental_Transactions：土地流转交易量
- Average_Land_Rental_Price：平均土地租金
- Livelihood_Strategy_Distribution：生计策略分布

#### 3.2 智能体级数据（agent_data.csv）
- Step：仿真时期
- AgentID：智能体ID
- Capital：资本
- Total_Income：总收入
- Livelihood_Strategy：生计策略
- Operated_Land_Area：经营土地面积
- Agricultural_Income_Ratio：农业收入比例
- Transition_Probability：转换概率
- 其他智能体特定属性

### 4. 运行脚本

#### 4.1 主运行脚本
```bash
# 运行基准实验
python run_experiment.py

# 运行特定规模测试
python -c "from run_experiment import test_different_scales; test_different_scales()"
```

#### 4.2 分析脚本
```bash
# 博弈过程分析（推荐）
python analysis/game_process_analysis.py

# 微观博弈可视化
python analysis/micro_game_visualization.py

# 详细博弈分析
python analysis/detailed_game_analysis.py
```

## 核心博弈机制

### 作物选择博弈（核心博弈）

**博弈参与者：**
- 农户：根据风险规避系数和预期效用选择作物
- 农业企业：根据目标回报率和生产效率选择作物

**策略空间：**
- 策略A：种植粮食作物（Grain）- 享受粮食补贴政策支持
- 策略B：种植经济作物（Cash Crop）- 完全市场化，无政策支持

**支付矩阵计算：**
- 粮食作物的预期效用：享受价格托底和补贴支持
- 经济作物的预期效用：完全市场化，价格波动性高

**决策机制：**
- 基于风险调整后的预期效用进行决策
- 粮食补贴政策直接影响作物选择博弈

### 土地租赁市场博弈（次要博弈）

**市场机制：**
- 中央清算所机制
- 供需匹配：农户保留价格 vs 企业出价
- 动态定价：基于市场供需的租金形成

**交易流程：**
1. 农户计算自耕净收益（保留价格）
2. 企业计算最大出价
3. 中央清算所进行匹配
4. 确定交易价格和数量

## 政策影响机制

### 粮食补贴政策传导效应

1. **价格托底效应**：粮食最低收购价降低价格风险
2. **补贴激励效应**：粮食补贴率提高粮食作物预期收益
3. **风险调整效应**：不同作物的价格波动性差异影响风险规避型主体的选择
4. **政策传导效应**：补贴政策通过作物选择博弈传导到土地市场和生计策略

### 数学表达

**粮食作物的预期利润受补贴政策影响：**
$$E[\pi_G] = E[Y_G \cdot P_G] + S_G - C_G$$

其中：
- $E[\pi_G]$ 是粮食作物的预期利润
- $E[Y_G \cdot P_G]$ 是预期产量与价格的乘积
- $S_G$ 是粮食补贴，直接体现政策影响
- $C_G$ 是生产成本

## 输出指标分类

### 核心指标：作物选择指标
- 粮食作物种植比例
- 作物选择变化率
- 风险规避效应
- 补贴政策响应度

### 次要指标：土地结构指标
- 企业经营土地面积
- 土地分布基尼系数
- 土地流转率

### 政策效果指标：补贴分配指标
- 补贴资本化程度
- 收入基尼系数
- 补贴效率

### 长期演化指标：小农生存指标
- 纯农户数量
- 小农退出率
- 生计多样性

## 冗余文件识别

### 需要保留的核心文件
- `run_experiment.py`：主运行脚本
- `core/model.py`：核心模型
- `agents/`：智能体定义
- `modules/`：功能模块
- `config/`：配置文件
- `analysis/game_process_analysis.py`：博弈过程分析（推荐）

### 可以删除的冗余文件
- `README_backup.md`：README备份文件
- `debug_land_market.py`：调试文件
- `analysis/micro_agent_analysis.py`：旧版微观分析
- `analysis/improved_heterogeneity_analysis.py`：旧版异质性分析
- `analysis/improved_heterogeneity_analysis_fixed.py`：修复版异质性分析
- `analysis/micro_agent_analysis_fixed.py`：修复版微观分析
- `python`：空文件

### 建议保留的分析文件
- `analysis/game_process_analysis.py`：博弈过程分析（最新）
- `analysis/micro_game_visualization.py`：微观博弈可视化
- `analysis/detailed_game_analysis.py`：详细博弈分析

## 运行建议

1. **首次运行**：使用`test`配置快速验证模型功能
2. **正式分析**：使用`medium`配置进行平衡性能和详细度的分析
3. **深度研究**：使用`large`或`xlarge`配置进行深入分析
4. **结果分析**：优先使用`game_process_analysis.py`进行博弈过程分析

## 注意事项

1. **数据依赖**：分析脚本需要先运行仿真生成数据
2. **字体问题**：中文显示可能需要配置合适的字体
3. **内存使用**：大规模配置需要足够的内存
4. **结果保存**：仿真结果会自动保存到`results/`目录下 