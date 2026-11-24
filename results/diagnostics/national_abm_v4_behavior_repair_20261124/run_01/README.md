# V4家庭行为失配：一次性修复交付

父提交 `c0e36a5e52adfc8eaa83ab0d543a70b3f1dba66c`。保持原600户、外生价格、工资和财政规则。原V4的M0–M4及V4额外种子结果从冻结运行账读取；本轮重跑M0的20个追加种子并逐文件核对历史账。

| 材料 | 用途 |
|---|---|
| [论文用模型修订与复验](模型修订与行为复验.md) | 根因、一次性修复、五年结果与结论边界 |
| [数据覆盖](survey_coverage.csv)、[字段字典与有效值](variable_coverage.csv)、[队列及源表指纹](source_receipt.csv) | 问卷、名单、重复和缺失 |
| [2011参与消融](participation_2011.csv)、[资金分组](liquidity_2011.csv)、[风险集](risk_set_counts.csv) | A到G（G包括劳动和现金两个扰动） |
| [年度同户转移](transitions.csv)、[2015兼种来源](mix_sources_2015.csv)、[精度比较](precision_2015.csv) | 调查、模型和追访敏感性 |
| [播种强度](sowing_intensity.csv)、[粮食缺口分解](grain_gap.csv) | 实物亩、播种亩次和结果来源 |
| [年度行为](behavior_comparison.csv)、[20种子配对](paired_20.csv)、[账目](accounts.csv)、[一致性](receipts.csv)、[测试](tests.xml) | 复验和守恒 |

在授权环境下的复算（需要原NFPS、CFPS、CHFS文件、原V3及V4冻结账、现有作物预算与项目依赖）：

```powershell
$env:AP_ABM_CHFS2011 = "<已授权的2011年CHFS家庭表绝对路径>"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe -B -X utf8 -m scripts.run_national_household_portfolio v4-repair-suite results/diagnostics/national_abm_v4_repair_reproduction --workers 5
```

只重跑一组：`python -m scripts.run_national_household_portfolio v4 M2 NEW_DIR --ablation F --no-repeat`，或修复版 `python -m scripts.run_national_household_portfolio v4 V4 NEW_DIR --repair`。已运行目录可以用 `python -m scripts.audit_household_portfolio_behavior audit-v4-repair RUN_ROOT final` 重新聚合。年度诊断种子为103和109，修复版包含独立重置的103重复；追加种子201至220。四项标准依次为参与率和兼种率绝对差各不超过10个百分点、份额分位误差不超过0.15、粮食面积相对误差不超过10%。

全逐户源表核对、计划、选择概率、随机抽样数、约束、交易和实际生产账留在本地 `PRIVATE_DO_NOT_SHARE`。公开仓库只包含源码、测试和汇总；无政策排序、无政府训练。
