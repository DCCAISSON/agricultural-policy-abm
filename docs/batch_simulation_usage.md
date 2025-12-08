# 批量模拟脚本使用指南

## 功能概述

增强后的 `analysis/batch_simulation.py` 提供了以下功能：

1. ✅ **政策情景选择** - 支持选择不同的补贴政策情景
2. ✅ **批量运行** - 支持多次运行并聚合结果
3. ✅ **可视化生成** - 自动生成关键指标图表
4. ✅ **结果解读** - 自动生成结果解读报告
5. ✅ **输出管理** - 结果自动保存到 `results/` 文件夹

## 使用方法

### 1. 列出可用的政策情景

```bash
python analysis/batch_simulation.py --list-scenarios
```

输出示例：
```
可用的政策情景:
  baseline_cash: 基准面积现金补贴
    描述: 按经营者面积发放现金补贴，基准补贴 100 元/亩，补贴率 15%。
  owner_tiered_cash: 所有者分段现金补贴
    描述: 依据自有土地面积分段给予补贴，模拟部分地区对土地承包权人的支持。
  mixed_in_kind_support: 现金+实物综合补贴
    描述: 部分补贴以现金形式到账，部分转化为农资抵扣和技术培训。
```

### 2. 运行批量模拟（默认情景）

```bash
python analysis/batch_simulation.py --runs 10 --steps 30
```

### 3. 运行批量模拟（指定政策情景）

```bash
# 使用基准现金补贴情景
python analysis/batch_simulation.py --runs 10 --steps 30 --scenario baseline_cash

# 使用所有者分段现金补贴情景
python analysis/batch_simulation.py --runs 10 --steps 30 --scenario owner_tiered_cash

# 使用现金+实物综合补贴情景
python analysis/batch_simulation.py --runs 10 --steps 30 --scenario mixed_in_kind_support
```

### 4. 指定输出路径

```bash
python analysis/batch_simulation.py --runs 10 --steps 30 --scenario baseline_cash --output results/my_output.json
```

如果不指定 `--output`，会自动生成带时间戳的文件名：
- `results/batch_output_baseline_cash_20250101_120000.json`

### 5. 禁用可视化（加快运行速度）

```bash
python analysis/batch_simulation.py --runs 50 --steps 30 --no-viz
```

### 6. 启用详细日志

```bash
python analysis/batch_simulation.py --runs 10 --steps 30 --verbose
```

## 输出文件说明

运行完成后，会在 `results/` 文件夹生成以下文件：

### 1. JSON 结果文件

文件名格式：`batch_output_<scenario>_<timestamp>.json`

包含：
- `individual_runs`: 每次运行的详细统计结果
- `aggregated_statistics`: 聚合统计量（均值、标准差）
- `metadata`: 运行元数据

### 2. 可视化图表

保存在 `results/visualizations/` 文件夹：

- `key_indicators.png`: 关键指标综合图表
  - 企业类型分布（条形图）
  - 作物结构分布（饼图）
  - 土地流转率变化（折线图）
  - 收入分布与不平等（散点图）

- `subsidy_and_market.png`: 补贴与市场图表
  - 补贴总额分布（直方图）
  - 市场价格变化（折线图）

### 3. 结果解读报告

文件名格式：`batch_output_<scenario>_<timestamp>_interpretation.txt`

包含：
- 关键指标摘要
- 结果解读
- 模型校准状态
- 建议

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--config` | - | 配置文件路径 | `config/master_config.yaml` |
| `--runs` | `--run` | 运行次数 | 10 |
| `--steps` | - | 每次运行的步数 | 30 |
| `--seed` | - | 随机种子（可选） | None |
| `--scenario` | `--policy` | 政策情景ID | None |
| `--list-scenarios` | - | 列出所有可用情景 | - |
| `--output` | - | 输出文件路径 | 自动生成 |
| `--no-viz` | - | 不生成可视化图表 | False |
| `--verbose` | - | 启用详细日志 | False |

## 示例场景

### 场景1：快速测试（3次运行，5步）

```bash
python analysis/batch_simulation.py --runs 3 --steps 5 --scenario baseline_cash
```

### 场景2：完整批量模拟（50次运行，30步）

```bash
python analysis/batch_simulation.py --runs 50 --steps 30 --scenario baseline_cash --output results/full_simulation.json
```

### 场景3：政策对比（分别运行不同情景）

```bash
# 运行基准情景
python analysis/batch_simulation.py --runs 20 --steps 30 --scenario baseline_cash --output results/baseline.json

# 运行分段补贴情景
python analysis/batch_simulation.py --runs 20 --steps 30 --scenario owner_tiered_cash --output results/tiered.json

# 运行混合补贴情景
python analysis/batch_simulation.py --runs 20 --steps 30 --scenario mixed_in_kind_support --output results/mixed.json
```

## 结果解读

详细的结果解读方法请参考：
- `docs/batch_output_interpretation.md` - 完整的结果解读指南

## 模型校准状态

⚠️ **重要提示**: 当前模型尚未完成基于2020年数据的参数校准。

建议的校准流程：
1. 使用2020年CLES数据进行参数校准
2. 使用2021-2022年数据进行仿真验证
3. 对比模拟结果与真实数据

详细说明请参考 `docs/batch_output_interpretation.md` 中的"模型校准状态"部分。

## 常见问题

### Q: 如何查看运行进度？

A: 使用 `--verbose` 参数启用详细日志。

### Q: 结果文件保存在哪里？

A: 默认保存在 `results/` 文件夹，文件名包含时间戳和情景ID。

### Q: 如何比较不同政策情景的结果？

A: 分别运行不同情景，保存到不同文件，然后对比JSON文件中的 `aggregated_statistics` 部分。

### Q: 可视化图表在哪里？

A: 保存在 `results/visualizations/` 文件夹中。

### Q: 如何加快运行速度？

A: 使用 `--no-viz` 禁用可视化，或减少 `--runs` 和 `--steps` 参数。

## 技术细节

- 脚本会自动处理路径问题，可以从任何目录运行
- 支持相对路径和绝对路径
- 自动创建输出目录
- 使用UTF-8编码保存所有文件
- 可视化图表使用300 DPI高分辨率

