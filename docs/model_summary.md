# 气候政策ABM模型运行总结

## 模型概述

本项目是一个基于多智能体建模（Agent-Based Modeling, ABM）的气候政策农业经济仿真系统，**重点关注粮食补贴政策对农户和农业企业行为的影响**。

### 核心特点
- **理论驱动**：基于成熟的经济学理论（Melitz企业异质性模型、前景理论等）
- **化繁为简**：模型复杂性被审慎管理，首要目标是解答核心研究问题
- **模块化设计**：实现模块间的低耦合与模块内的高内聚

## 模型运行流程

### 1. 初始化阶段
```python
# 加载规模配置
scale_config = get_scale_config("medium")  # 推荐配置

# 创建模型
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

### 2. 年度决策周期（5个阶段）

#### 阶段一：政策发布
- GovernmentAgent宣布年度补贴政策和价格地板

#### 阶段二：核心博弈 - 作物选择
- **核心博弈**：所有土地经营者决定种植粮食作物 vs 经济作物
- 基于风险调整后的预期效用进行决策
- 粮食补贴政策直接影响决策结果

#### 阶段三：次要博弈 - 土地租赁市场
- 土地供需双方基于保留价格进行博弈
- 中央清算所机制进行市场出清

#### 阶段四：收获与结算
- 计算各主体的收入和支出
- 更新资本和收入状态

#### 阶段五：生计策略演化
- 农户动态调整生计策略（Pure_Farmer → Pluriactive → Non_Farmer）

## 核心博弈机制

### 作物选择博弈（核心博弈）
- **博弈参与者**：农户和农业企业
- **策略空间**：粮食作物（享受补贴）vs 经济作物（市场化）
- **决策机制**：基于风险调整后的预期效用
- **政策影响**：粮食补贴政策直接影响作物选择

### 土地租赁市场博弈（次要博弈）
- **市场机制**：中央清算所机制
- **交易流程**：农户保留价格 vs 企业出价
- **动态定价**：基于市场供需的租金形成

## 数据输出

### 模型级数据（model_data.csv）
- 土地结构指标：企业经营土地面积、土地分布基尼系数
- 政策效果指标：补贴影响率、收入基尼系数
- 演化指标：纯农户数量、小农退出率、土地流转交易量

### 智能体级数据（agent_data.csv）
- 个体特征：资本、收入、生计策略
- 经营状况：经营土地面积、农业收入比例
- 行为演化：转换概率、连续年份统计

## 运行脚本

### 主运行脚本
```bash
# 运行基准实验
python run_experiment.py

# 测试不同规模配置
python -c "from run_experiment import test_different_scales; test_different_scales()"
```

### 分析脚本
```bash
# 博弈过程分析（推荐）
python analysis/game_process_analysis.py

# 微观博弈可视化
python analysis/micro_game_visualization.py

# 详细博弈分析
python analysis/detailed_game_analysis.py
```

## 规模配置选项

| 配置名称 | 农户数量 | 企业数量 | 空间网格 | 适用场景 |
|---------|---------|---------|---------|---------|
| test    | 20      | 3       | 15×15   | 快速测试和调试 |
| small   | 100     | 10      | 30×30   | 初步验证和概念测试 |
| medium  | 300     | 30      | 50×50   | 平衡性能和详细度（推荐） |
| large   | 500     | 50      | 70×70   | 深入分析和政策评估 |
| xlarge  | 1000    | 100     | 100×100 | 复杂场景和长期仿真 |

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

## 文件结构（清理后）

```
从0到1/
├── core/                          # 核心模型逻辑
│   ├── model.py                   # 主模型类
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

## 已删除的冗余文件

- `README_backup.md`：README备份文件
- `debug_land_market.py`：调试文件
- `analysis/micro_agent_analysis.py`：旧版微观分析
- `analysis/improved_heterogeneity_analysis.py`：旧版异质性分析
- `analysis/improved_heterogeneity_analysis_fixed.py`：修复版异质性分析
- `analysis/micro_agent_analysis_fixed.py`：修复版微观分析
- `python`：空文件

## 验证结果

博弈过程分析脚本已成功运行，生成了以下文件：
- `period_summary_report.csv`：每一期的详细汇总报告
- `period_analysis_visualization.png`：可视化分析图表
- `statistical_report.json`：统计报告
- `key_indicators_table.csv`：关键指标表格

模型运行流程完整，数据输出正常，分析功能完善。 