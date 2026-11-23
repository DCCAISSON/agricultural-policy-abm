# NFPS家庭动态经营与粮经配置：V3候选复验

研究问题是有限支持在什么条件下转化为粮食生产能力。本轮在相同的基础支持下比较家庭行为，不识别补贴效应。正文见[模型设计与实算结果](模型设计与实算结果.md)，同户年度表见 household_behavior.csv，四种模型完整经营账汇总见 production_and_land.csv。

## 版本与口径

A为原600户、V2硬份额；B只改决策规则；C只改600户；D同时改两项。规则对照指整个份额决策机制，包括历史估计窗口、地区分组与偏离参考的代价，并非只改变一个线性规划系数。Dmid、Dlow、Dhigh是同一新队列下的统一调整成本敏感性，不是三个政策。原基准提交 d2c4de0c599379360607b57f0b092805e085b735。本文五年是历史检验，不能称为未使用的独立验证期。

每次模型运行包括103、109，以及独立重置的103重复。模型份额为粮食除以粮经总面积；正经营却仅生产其他作物的家庭，在调查中仍然是生产者。其他作物另报面积、户数和全作物播种强度，不伪造其他作物在模拟地块、季次上的产量。household_behavior 按当年能匹配的户数汇总；production_and_land 和 factorial 的家庭口径是所有600户。两个口径不直接相减。

## 授权数据环境复算

使用项目既有Python环境和冻结数据目录；不能仅靠公开仓库恢复私人调查。输入需要旧队列、地区收益、2010年NFPS合格来源索引、2009至2015年户表及成员表、作物预算和冻结V2逐户对照账。先取新目录，不能指向已完成的路径。以下命令保持历史表、源码及旧分支不变。

```powershell
$d='results/diagnostics/national_crop_choice_soft_response_reproduction'
$p="$d/PRIVATE_DO_NOT_SHARE"
New-Item -ItemType Directory -Path "$d/run_01","$p/equal_consumption","$p/mobility" -ErrorAction Stop
$py='.venv/Scripts/python.exe'
& $py -B -X utf8 -c "from pathlib import Path;from scripts.prepare_nfps_v3 import candidate_cohort;candidate_cohort(Path('$d'))"
$runs=@{'A'='continuous';'B'='Bmid';'Bhigh'='Bhigh';'equal_consumption/C'='C';'equal_consumption/D'='Dmid';'equal_consumption/Dlow'='Dlow';'equal_consumption/Dhigh'='Dhigh';'mobility/B'='B';'mobility/D'='D'}
foreach($run in $runs.GetEnumerator()) {
  & $py -B -X utf8 -m scripts.run_national_household_portfolio crop-mix $run.Value "$p/$($run.Key)"
  if($LASTEXITCODE -ne 0){throw $run.Key}
}
& $py -B -X utf8 -m scripts.audit_household_portfolio_behavior soft-response $d
$m=Get-Content results/diagnostics/national_crop_mix_repair_20261123/run_01/run_manifest.json -Raw | ConvertFrom-Json
$t=($m.tests_command -split ' ')|Where-Object {$_ -like 'tests/*'}
& $py -B -X utf8 -m pytest -q @t tests/test_continuous_crop_history.py tests/test_v3_crop_response.py "--junitxml=$d/run_01/tests.xml"
```

复核依据：survey_cleaning 为两字段异常清理前后各年合计与有效数；selection、cohort_margins、cohort_distance 为入选和分布；fitted_reference 的 n 包含次年未播种者，positive 是计算条件份额的分母；其中 penalty 是统一规格成本，在异质规格是上界，实际成本见 adjustment_costs。calibration 仅比较三档统一成本的2011—2012年误差和，不用后期年份。target_response 为当年所有实际播种且有参考值的家庭；conditional_price_response 固定当年最终土地和资金条件，没有重新进行市场或政策均衡。one_step_history 把实际与模拟滞后输入置于同一家庭、同一抽取数之下。repeat_receipts 和 tests.xml 分别核对重复运行与程序断言。逐户核对表、身份映射和含家庭号的原始异常值记录不向外发布。
