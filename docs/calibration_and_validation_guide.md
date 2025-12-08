# 模型校准与验证指南

## 概述

本指南说明如何使用2020年CLES数据进行模型参数校准，然后使用2021-2022年数据进行验证。

## 快速开始

### 1. 运行完整流程（校准+验证）

```bash
python scripts/run_calibration_and_validation.py --full
```

### 2. 只运行校准

```bash
python scripts/run_calibration_and_validation.py --calibrate
```

### 3. 只运行验证

```bash
python scripts/run_calibration_and_validation.py --validate
```

## 详细说明

### 校准流程

1. **提取2020年目标矩**
   - 从 `CLES_2020_2021_household_panel_mapping_label.xlsx` 中提取2020年数据
   - 计算关键指标：
     - 土地流转率
     - 平均农户收入
     - 基尼系数
     - 作物结构（粮食/经济作物比例）
     - 平均单产

2. **运行参数校准**
   - 使用 Simulated Method of Moments (SMM) 方法
   - 最小化模拟统计量与目标统计量之间的距离
   - 优化模型参数

3. **保存校准结果**
   - `results/calibration/calibrated_parameters.json` - 校准后的参数
   - `results/calibration/calibration_report.json` - 校准报告

### 验证流程

1. **加载2021-2022年数据**
   - 从 `CLES_2020_2021_household_panel_mapping_label.xlsx` 提取2021年数据
   - 从 `CLES_2022_household_mapping_label.xlsx` 提取2022年数据

2. **运行模型仿真**
   - 使用校准后的参数运行模型
   - 进行多次运行并聚合结果

3. **对比结果**
   - 对比模拟结果与2021-2022年真实数据
   - 计算拟合度指标
   - 生成验证报告

## 输出文件

### 校准输出

- `results/calibration/calibrated_parameters.json` - 校准后的参数
- `results/calibration/calibration_report.json` - 校准报告（包含目标函数值、迭代次数等）

### 验证输出

- `results/calibration/validation_report.json` - 验证报告（包含模拟结果与真实数据对比）

## 注意事项

1. **数据文件位置**: 确保数据文件在 `校准用数据源/` 目录下
2. **运行时间**: 完整校准可能需要数小时，建议使用 `--calibrate` 先测试
3. **参数映射**: 校准后的参数需要映射到 `config/master_config.yaml`，这部分需要根据具体参数结构实现

## 故障排除

### 问题1: 找不到数据文件

**错误**: `File not found: ...`

**解决**: 检查数据文件是否在 `校准用数据源/` 目录下，文件名是否正确

### 问题2: 列名不匹配

**错误**: `Column 'xxx' not found in dataset`

**解决**: 检查 `config/calibration_config.yaml` 中的列映射配置，确保列名与数据文件一致

### 问题3: 校准失败

**错误**: `Calibration failed`

**解决**: 
- 检查目标矩是否合理
- 调整参数边界
- 增加迭代次数
- 检查模型是否能正常运行

## 下一步

1. **完善参数映射**: 实现校准参数到模型配置的自动映射
2. **增强验证**: 添加更多拟合度指标（R²、MAE等）
3. **可视化**: 生成校准和验证的可视化图表
4. **敏感性分析**: 分析参数对模型输出的影响

