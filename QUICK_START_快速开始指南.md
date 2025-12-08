# 粮食补贴政策多智能体模型 - 快速开始指南# 模型仿真模拟 - 快速开始指南



**更新时间**: 2024-12-01  **更新时间**: 2024-11-20  

**状态**: ✅ 模型就绪，支持完整仿真**状态**: ✅ 模型已完全准备好进行仿真模拟



------



## 🚀 快速开始## ✅ 模型就绪状态



### 统一入口脚本：`run_integrated_workflow.py`经过专家审阅和全面修复，模型现在**完全支持**进行仿真模拟：



所有模拟功能已整合到单一脚本，支持三种运行模式：- ✅ 所有P0致命问题已修复

- ✅ 所有P1架构建议已实现

```bash- ✅ 所有P2代码优化已完成

# 模式1️⃣ 单次模拟 + 8面板可视化仪表盘- ✅ 企业风险项已恢复

python run_integrated_workflow.py --mode single --steps 20 --tag baseline- ✅ 并行计算已支持

- ✅ 机制解耦已完成

# 模式2️⃣ 补贴率政策对比实验（倍率相对基准350元/亩）

python run_integrated_workflow.py --mode policy-sweep --steps 20 --subsidy-list 0.8,1.0,1.2,1.5---



# 模式3️⃣ Mesa 浏览器交互式可视化## 🚀 快速开始

python run_integrated_workflow.py --mode mesa --port 8521

```### 方法0：全集成工作流脚本（新）



---```bash

# 单次运行并生成八宫格仪表盘

## 📊 输出说明python run_integrated_workflow.py --mode single --steps 20 --tag baseline



### 单次模拟输出 (`--mode single`)# 补贴率政策情景（倍率相对基准350元/亩）

python run_integrated_workflow.py --mode policy-sweep --steps 20 --subsidy-list 0.8,1.0,1.2

```

results/<tag>_<timestamp>/# 一键启动 Mesa 浏览器可视化

├── dashboard.png          # 8面板可视化仪表盘python run_integrated_workflow.py --mode mesa --port 8521

├── simulation_data.csv    # 完整时序数据```

├── summary_report.txt     # 文本摘要

└── summary.json           # 结构化摘要**输出**：

```- `results/<tag>_<timestamp>/dashboard.png`：与之前 `simulation_results.png` 相同风格的八面板图

- `results/policy_sweep_<timestamp>/policy_sweep_summary.csv|png`：补贴倍率对粮食供给和农户收入的对比

**仪表盘面板包含**：- `summary_report.txt / summary.json`：运行摘要

1. 市场价格动态（粮食 vs 经济作物）

2. 生计策略演化堆叠图### 方法1：使用统一运行脚本（推荐）

3. 粮食供给趋势

4. 经济作物供给趋势```bash

5. 农户平均收入# 基本运行（20步，自动可视化）

6. 企业盈利率 & 土地基尼系数python run_model_with_visualization.py

7. 年度补贴发放

8. 生计策略变化对比（初始 vs 最终）# 指定步数

python run_model_with_visualization.py --steps 50

### 政策对比输出 (`--mode policy-sweep`)

# 指定标签（用于结果文件夹命名）

```python run_model_with_visualization.py --steps 30 --tag my_experiment

results/policy_sweep_<timestamp>/```

├── policy_sweep.png           # 补贴倍率对粮食供给/收入的影响图

└── policy_sweep_summary.csv   # 各补贴率下的核心指标对比**输出**:

```- 数据CSV文件：`results/run_YYYYMMDD_HHMMSS/simulation_data.csv`

- 可视化图表：`results/run_YYYYMMDD_HHMMSS/visualization.png`

### Mesa浏览器可视化 (`--mode mesa`)- 摘要报告：`results/run_YYYYMMDD_HHMMSS/summary_report.txt`



访问 `http://127.0.0.1:8521`，可交互调整参数观察模型动态。---



---### 方法2：使用runner.py（功能更全面）



## 📈 示例运行```bash

# 单次仿真

```bashpython runner.py --mode single --steps 50

# 运行20步基准模拟

python run_integrated_workflow.py --mode single --steps 20 --tag demo# 敏感性分析

python runner.py --mode sensitivity --param-name "policy.government.grain_subsidies_rate" --param-values "0.05,0.10,0.15,0.20"

# 预期输出

======================================================================# 政策对比

模拟结果摘要python runner.py --mode policy

======================================================================

粮价: 1.23 → 3.00 元/斤# 参数校准

经作价: 2.76 → 3.05 元/斤python runner.py --mode calibration

纯农户: 103 → 102```

兼业户: 94 → 89

非农户: 103 → 109---

平均农户收入: 98899 → 121909 元

企业盈利率(最终): 100.00%### 方法3：批量运行（并行计算）

土地基尼系数(最终): 0.417

累计补贴: 8072.6 万元```python

from analysis.batch_simulation import run_batch_simulation

[OK] 单次运行完成，结果位于: results/demo_20241201_100000

```# 并行运行100次（自动使用多核CPU）

results = run_batch_simulation(

---    config_path="config/master_config.yaml",

    n_runs=100,

## 📖 Python代码中运行    n_steps=30,

    n_workers=None  # 自动检测CPU核心数

```python)

from core.model_builder import ModelBuilder

import yaml# 查看结果

for result in results:

# 加载配置    print(f"运行 {result['run_id']}: 土地流转率={result['land_transfer']['transfer_rate']:.3f}")

with open("config/master_config.yaml", 'r', encoding='utf-8') as f:```

    config = yaml.safe_load(f)

---

# 构建模型

builder = ModelBuilder(config)## 📊 运行示例

model = builder.build()

### 示例1：基本仿真（20步）

# 运行模拟

for step in range(20):```bash

    model.step()python run_model_with_visualization.py --steps 20

    ```

    # 获取关键指标

    grain_price = model.environment.get_market_price('crop_price')**预期输出**:

    print(f"Step {step+1}: 粮价={grain_price:.2f}元")```

======================================================================

# 获取数据运行模拟并收集数据...

model_data = model.datacollector.get_model_vars_dataframe()======================================================================

```

正在创建 300 个农户...

---正在创建 30 个农业企业...

[Step 1] 纯农户: 92人 | 兼业: 162人 | 非农户: 46人

## ⚙️ 命令行参数[Step 1] 企业土地: 211亩

[Step 1] 企业粮食配置: 45.3%

| 参数 | 默认值 | 说明 |[Step 1] 粮价: 1.25元/斤 | 经作价: 1.80元/斤

|------|--------|------|...

| `--mode` | `single` | 运行模式：`single`/`policy-sweep`/`mesa` |[Step 20] 纯农户: 82人 | 兼业: 145人 | 非农户: 73人

| `--config` | `config/master_config.yaml` | 配置文件路径 |[Step 20] 企业土地: 631亩

| `--steps` | `20` | 模拟步数 |[Step 20] 企业粮食配置: 100.0%

| `--output-dir` | `results` | 输出目录 |[Step 20] 粮食供需比: 8.14倍

| `--tag` | `run` | 结果子目录标签 |[Step 20] 粮价: 1.26元/斤 | 经作价: 0.63元/斤

| `--subsidy-list` | - | 补贴倍率列表（policy-sweep模式必需） |

| `--port` | `8521` | Mesa服务器端口 |✓ 数据已保存至: results/run_20241120_103124/simulation_data.csv

| `--quiet` | - | 静默模式（不输出逐步日志） |✓ 图表已保存至: results/run_20241120_103124/visualization.png

✓ 报告已保存至: results/run_20241120_103124/summary_report.txt

---```



## 🔧 核心配置---



主配置文件：`config/master_config.yaml`### 示例2：Python代码中运行



```yaml```python

scale:from core.model_builder import ModelBuilder

  n_farmers: 300          # 农户数量import yaml

  n_enterprises: 80       # 企业数量

# 加载配置

policy:with open("config/master_config.yaml", 'r', encoding='utf-8') as f:

  active_scenario: baseline_cash  # 使用350元/亩基准补贴    config = yaml.safe_load(f)



environment:# 构建模型

  market:builder = ModelBuilder(config)

    use_endogenous_pricing: true  # 内生价格model = builder.build()

    base_grain_demand: 1500000    # 粮食基础需求（斤）

```# 运行模拟

for step in range(20):

政策场景库：`config/policy_params.json`    model.step()

    

---    # 获取关键指标

    pure_farmers = model.get_number_of_pure_farmers()

## 📁 项目结构    grain_price = model.environment.get_market_price('crop_price')

    cash_crop_price = model.environment.get_market_price('cash_crop')

```    

从0到1/    print(f"Step {step+1}: 纯农户={pure_farmers}人, 粮价={grain_price:.2f}元, 经作价={cash_crop_price:.2f}元")

├── run_integrated_workflow.py   # ← 唯一入口脚本

├── config/                      # 配置文件# 获取数据

│   ├── master_config.yaml       # 主配置model_data = model.datacollector.get_model_vars_dataframe()

│   └── policy_params.json       # 政策场景库agent_data = model.datacollector.get_agent_vars_dataframe()

├── core/                        # 核心模型```

├── agents/                      # 智能体定义

├── modules/                     # 功能模块---

├── visualization/               # Mesa可视化服务

├── docs/                        # 详细文档## 📈 监控的关键指标

│   └── ODD_plus_D_完整版_决策机制详解.md  # 模型完整描述

├── results/                     # 模拟结果输出运行仿真时，可以监控以下关键指标：

└── archive/                     # 历史文档和脚本

```### 农户指标

- **纯农户数量**: 目标30-50人

---- **兼业农户数量**: 动态变化

- **非农户数量**: 动态变化

## 📚 关键文档- **生计策略演化**: 观察策略切换



| 文档 | 位置 | 内容 |### 企业指标

|------|------|------|- **企业粮食配置**: 目标30-50%

| **ODD+D完整版** | `docs/ODD_plus_D_完整版_决策机制详解.md` | 模型完整描述 |- **企业土地面积**: 观察扩张趋势

| **模型描述** | `ODD_D_MODEL_DESCRIPTION_模型描述文档.md` | 概述版本 |- **企业作物组合**: 粮食vs经济作物

| **理论框架** | `最终理论框架.md` / `理论框架Final.md` | 理论基础 |

| **参数数据源** | `docs/权威参数数据源.md` | 参数来源 |### 市场指标

- **粮食供需比**: 目标<5倍

---- **粮食价格**: 观察价格动态

- **经济作物价格**: 观察价格动态（应自由波动）

## ❓ 常见问题

---

### Q1: 模块导入失败

确保在项目根目录运行，或将根目录添加到 `PYTHONPATH`## ⚙️ 配置参数



### Q2: 运行缓慢主要配置在 `config/master_config.yaml` 中：

- 减少 `n_farmers`/`n_enterprises`

- 减少 `--steps````yaml

- 使用 `--quiet` 减少输出experiment:

  steps: 50        # 模拟步数

### Q3: 中文显示乱码  seed: 42         # 随机种子

确保安装了 SimHei 或 Microsoft YaHei 字体

scale:

---  n_farmers: 300   # 农户数量

  n_enterprises: 30 # 企业数量

**模型已完全就绪！** 🎉

environment:

运行 `python run_integrated_workflow.py --mode single --steps 20` 开始您的第一次模拟。  market:

    use_endogenous_pricing: true  # 内生价格机制
    base_grain_demand: 400000     # 粮食基础需求
    base_cash_crop_demand: 200000 # 经作基础需求
```

---

## 🔧 常见问题

### Q1: 运行时报错 "ModuleNotFoundError"
**A**: 确保在项目根目录运行，或添加项目路径到PYTHONPATH

### Q2: 运行速度慢
**A**: 
- 使用并行计算：`n_workers=4`（4核CPU）
- 减少步数：`--steps 20`（快速测试）
- 减少智能体数量：修改`config/master_config.yaml`中的`n_farmers`和`n_enterprises`

### Q3: 结果不稳定
**A**: 
- 使用固定随机种子：`seed: 42`
- 增加运行次数：批量运行100次取平均
- 检查初始条件：确保配置合理

### Q4: 企业100%种粮
**A**: 这是已知问题，已修复风险项。如果仍出现，检查：
- 风险厌恶系数是否合理
- 价格历史窗口是否足够（10期）
- 经作价格是否被锁定

---

## 📝 输出文件说明

### simulation_data.csv
包含每步的关键指标：
- `step`: 步数
- `pure_farmer_count`: 纯农户数量
- `enterprise_grain_pct`: 企业粮食配置百分比
- `supply_demand_ratio`: 粮食供需比
- `grain_price`: 粮食价格
- `cash_crop_price`: 经济作物价格

### visualization.png
多面板可视化图表：
- 农户生计策略分布
- 企业土地和粮食配置
- 市场供需比
- 价格动态

### summary_report.txt
文本摘要报告，包含：
- 最终状态
- 趋势分析
- 目标完成情况

---

## 🎯 下一步

1. **运行基本仿真**: `python run_model_with_visualization.py --steps 20`
2. **查看结果**: 检查`results/`目录下的输出文件
3. **调整参数**: 修改`config/master_config.yaml`中的参数
4. **批量运行**: 使用`run_batch_simulation`进行多次运行
5. **参数校准**: 使用`runner.py --mode calibration`进行校准

---

## ✅ 验证清单

运行前确认：
- [x] Python环境已配置（Python 3.8+）
- [x] 依赖包已安装（numpy, pandas, matplotlib, mesa, yaml等）
- [x] 配置文件存在（`config/master_config.yaml`）
- [x] 项目根目录正确

运行后检查：
- [ ] 模型成功运行（无错误）
- [ ] 数据文件已生成
- [ ] 可视化图表已生成
- [ ] 指标在合理范围内

---

**模型已完全准备好进行仿真模拟！** 🎉

如有问题，请查看：
- `EXPERT_REVIEW_FIXES_专家审阅修复报告.md` - 修复详情
- `ARCHITECTURE_IMPROVEMENTS_架构改进完成报告.md` - 架构改进
- `MODEL_STATUS_模型当前状态说明.md` - 模型状态







