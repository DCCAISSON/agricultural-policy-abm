# 全国农业 ABM：V4.1之后的最终经济行为闭合候选 F1

F1在冻结提交 `7c5460d6b85f8c5f7243dbe515d16abe11d8c9aa` 上统一家庭经营参与与可行作物方案的比较。**代码与账户通过核验；四项行为联合验收未通过。** 本轮不替代M0、R0或R3，不运行政策排序及政府学习。

| 对照 | 定义 | 输入与路径 |
|---|---|---|
| M0 | 原始基准 | 复核V4.1冻结逐户账，不重新运行 |
| R0 | V4家庭行为修复版 | 同上 |
| R3 | V4.1开放外部土地池并取消二次参与抽签 | 同上 |
| F1 | 保留R3外部池，在参与和作物方向之间不再允许不可行维持方案挤占正剩余方案 | 本轮实际运行 |

诊断种子103、109，正式复验配对种子301–320，均运行2011–2015年，另重复F1的103。600户队列、组织配置、外生价格和天气抽取机制未变。结果是历史回放，不是独立样本外检验。

## 交付索引

| 工作包 | 公开文件 | 授权本地明细 |
|---|---|---|
| 土地、初态与规模 | [同户行为](annual_behavior.csv)、[土地来源](land_stock_bridge.csv)、[调查覆盖](survey_land_coverage.csv)、[面积分解](area_shapley.csv)、[排除原因](land_exclusion_reasons.csv) | 按种子R3、F1逐户土地明细；69条记录详表 |
| 劳动和资金 | [家庭收支与劳动](labor_cash_and_service.csv)、[成员实际报告覆盖](labor_member_coverage.csv)、[作物投工与机械费](labor_survey_accounts.csv)、[CFPS参照](labor_parameters.csv)、[CHFS流动性参照](liquidity_parameters.csv)、[2016–2017年预算区间](cash_budget_ranges.csv) | 既有24户审计和本轮家庭账 |
| 参与与粮经 | [年度状态转移](crop_transitions.csv)、[兼种偏差来源](mixed_sources.csv) | R3、F1实种面积与上年状态 |
| 生产能力 | [19户核验汇总](rice_history_19_summary.csv)、[分作物生产参数](production_parameters.csv)、[CHFS独立亩产](chfs_crop_yield_reference.csv) | 逐例历史核验，不含入发布包 |
| 组织、外部和财政 | [组织账](organization_accounts.csv)、[模型账户](accounts.csv)、[外部池参数](pool_profiles.csv)、[外部容量证据](pool_capacity_sources.csv) | 组织、地块、订单、成交、政府及外部池账 |
| 对照、出处与核验 | [四项验收](acceptance.csv)、[20种子配对](paired_20.csv)、[固定队列和NFPS指纹](source_receipt.csv)、[其他输入和配置哈希](additional_input_receipt.csv)、[同种子复算](reproduction_checks.csv)、[测试](tests.xml) | 控制组冻结账与F1新账 |

详见[实算与定稿判断](研究说明.md)、[ODD+D和三张图](ODD+D.md)、[参数追溯表](parameters.csv)。Git补丁、发布压缩包和远端回读另见本地同级 `release`。

## 复算

复用 [V4.1 授权输入前置条件](../../national_abm_v41_scale_closure_20261124/run_01/README.md)，保证执行的源码对应本轮提交。通过同一目录结构或本地授权目录链接，使NFPS/CFPS原表、冻结基准清单、权重证据及2016–2017年预算输入可读。设置 `AP_ABM_CHFS2011` 为授权CHFS 2011家庭表路径；另准备 `$chfs2015` 为CHFS 2015家庭表路径。运行 `$frozen` 指向原V4.1实验目录（含未公开的冻结对照账）。以下命令不修改旧结果，若目标已存在，请另用新目标：

```powershell
$out = 'results/local_f1_replay'
New-Item -ItemType Directory "$out/run_01" -ErrorAction Stop | Out-Null
python -B -X utf8 -m scripts.run_national_household_portfolio v4 V4 "$out/PRIVATE_DO_NOT_SHARE/F1_diag" --repair --scale F1 --seeds 103 109
if ($LASTEXITCODE -ne 0) { throw "F1 diagnostic run failed" }
foreach ($lo in 301,306,311,316) {
    $seeds = $lo..($lo+4)
    python -B -X utf8 -m scripts.run_national_household_portfolio v4 V4 "$out/PRIVATE_DO_NOT_SHARE/F1_$lo" --repair --scale F1 --no-repeat --seeds @seeds
    if ($LASTEXITCODE -ne 0) { throw "F1 run failed for $lo" }
}
python -B -X utf8 -m scripts.audit_v41_final_closure $out $frozen $chfs2015
python -B -X utf8 -m pytest -q tests/test_v4_area_economy.py
```

有授权测试输入时，可以使用V4.1相同的十个回归测试文件，以及 `tests/test_continuous_crop_history.py`、`tests/test_v3_crop_response.py` 和 `tests/test_v4_area_economy.py`，生成本轮 JUnit 回执。两个历史回归用例需要本地已封存的N1初始账和P3边界用例输入，未随公开分支发布。复算不需要训练或模拟政策排名。
