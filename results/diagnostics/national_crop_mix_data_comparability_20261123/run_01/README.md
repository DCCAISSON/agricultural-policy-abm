# 家庭粮经选择：数据比较、条件份额与递推检验

公开包只发布联合分布的作物状态合计和总体转移矩阵，不披露细小联合单元；分组明细保存在授权工作区。测试回执仅去除运行电脑名称。

先读 [数据与研究判断](data_source_audit.md)，再看 [两版同户复验](model_behavior.csv)、[历史输入替换](observed_lag_replay.csv)、[土地约束分解](shapley_land_exit.csv) 和 [复验图](paired_behavior.png)。这是第二候选，不是政策效果版本。两个种子按103、109顺序报告。

年度分布按全部可用行、五年能匹配的450户、冻结600户及基期估计家庭分别计算。份额分母仅为粮食加经济作物播种面积，两者都为零不计算份额。其他作物、完全未播、未确认身份分开。转移矩阵的区间按单期每户一对观测计算；跨期频率差按户聚类，结果为样本描述。比较矩阵保留所有年份，不以是否过线选年。

原始基线 a8ef70620028b6031fedfe874cfdc7d5acd22dd6；对照版为本地 b5a7df5bfd384c73f156cfe8f01d5475fedea6ce、公开 b76149195a3d5a597ccdd54ada21b9e7331302e6。唯一模型改动在 core/fixed_rule_runtime.py 的 continuous_crop_target，由 core/model.py 的年度经营决策调用，入口 scripts/run_national_household_portfolio.py 的 continuous 选项；candidate 仍走原规则。农户之前是否种粮、种多少来自实种账，不用调查验证年更新正式运行。

授权工作区在一个**未用的新目录**复算，保留原冻结输入、旧版归档和调查源路径，项目解释器如下：

```powershell
$d='results/diagnostics/national_crop_mix_data_comparability_reproduction'
.venv/Scripts/python.exe -B -X utf8 -m scripts.audit_crop_time_comparability "$d/run_01"
.venv/Scripts/python.exe -B -X utf8 -m scripts.run_national_household_portfolio crop-mix candidate "$d/PRIVATE_DO_NOT_SHARE/v1"
.venv/Scripts/python.exe -B -X utf8 -m scripts.run_national_household_portfolio crop-mix continuous "$d/PRIVATE_DO_NOT_SHARE/v2"
.venv/Scripts/python.exe -B -X utf8 -c "from pathlib import Path;from scripts.verify_crop_time_candidate import verify;verify(Path('$d/run_01'))"
$m=Get-Content results/diagnostics/national_crop_mix_repair_20261123/run_01/run_manifest.json -Raw | ConvertFrom-Json
$tests=($m.tests_command -split ' ') | Where-Object {$_ -like 'tests/*'}
.venv/Scripts/python.exe -B -X utf8 -m pytest -q @tests tests/test_continuous_crop_history.py "--junitxml=$d/run_01/tests.xml"
```

replication.csv 登记冻结版原账哈希和同种子复算；检验包括逐笔租金成交、收支结转、作物面积、产量、雇工和财政。model_grain_decomposition.csv 为全600户两版面积差，model_behavior.csv 和 model_accounts.csv 为逐年能匹配的家庭。模型金额均为2024年元，现金收入不是福利指标。调查账与模型账只在本地连接。公开版保留程序、分母、汇总表及测试，不提供个体账、细小联合单元或原始调查；公开源码不替代受限数据使用权。
