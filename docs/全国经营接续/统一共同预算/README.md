# 统一共同预算模型复算

## 本轮四区接续

执行版本以本地 `358bdf9` 为父。使用[登记协议](sprint_protocol.json)和[农业冻结选择](agricultural_freeze.json)，不把下面的旧540xxx、550xxx、560xxx种子用于本轮评价。原四区版本命令从原 `358bdf9` 树执行。新结果的唯一正式入口仍为 `scripts.run_succession_empirical_budget` 的 `train/evaluate`。先看[家庭修订与口径](家庭修订与口径.md)及[四项裁决](冲刺结果与裁决.md)。

从复现包的 `source` 目录执行。目录名须新建，`$Project` 指向授权原表和冻结档案所在的项目根：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -B -X utf8 -m scripts.diagnose_succession_sprint $Project hist_m1
python -B -X utf8 -m scripts.diagnose_succession_sprint $Project hist_trial --refine
python -B -X utf8 -m scripts.run_succession_empirical_budget train $Project new_train --workers 14
python -B -X utf8 -m scripts.run_succession_empirical_budget evaluate $Project new_eval --trained new_train --workers 14
python -B -X utf8 -m scripts.audit_succession_empirical_budget new_train new_eval new_audit
python -B -X utf8 -m scripts.check_succession_empirical_households $Project new_train protected_trace --evaluation new_eval --seed 574001
python -B -m pytest -q -p no:cacheprovider tests --junitxml=tests-recomputed.xml
```

未取得微观资料，仍可用包内源表重新计算训练筛选、主策略和稳定性复拟、执行时的滞后状态及Q值、财政账户、结果区间：

```powershell
python -B -X utf8 -m scripts.audit_succession_empirical_budget ../evidence/training ../evidence/evaluation checked
```

`screen.csv.gz` 记录哪些不同父路径年初被选入；`openings.csv.gz` 每次入选均分叉21动作；`frozen.json` 保存每个受支持状态的当年期望粮食、Q值和动作，以及低支持状态的估计记录。评价地区年度账同时记录实际动作、期初公开状态、是否回退及所选动作估值；固定政策和制度强制P0不填Q值。保护目录中的逐户与逐合成块日志只在授权环境生成。重复训练依登记去掉父种子可被4整除的子集，保持算法与奖励不变；不是一批独立的现实数据或额外可供挑选的主结果。

## 继承机制（原四区版）

本版用 F1 冻结队列的600户初始化同一个经济环境。四个政策区是东部、中部、西部和东北；省级样本内有独立的土地供需撮合，只有年度预算跨政策区竞争。既有全国F1复现、单区7a、两区首年拨款和前六年分配试验保持冻结，不视作本版不同配置。

## 输入与源码

| 对象 | 口径与入口 |
| --- | --- |
| 调查家庭 | `scripts/prepare_succession_budget_inputs.py::prepare` 核对三个输入指纹、600户唯一来源行及2010年户表、地区和主要字段；只使用授权环境的 F1 基期表、本地队列联结件及NFPS原表。 |
| 地块面积与历史 | `split_lots` 按户保留实际经营物理亩及可核承包亩。全年粮经播种仅给出作物比例，按比例映射到不超过4亩的合成单元，其他作物或剩余部分不置作粮食。未核可出租部分不进市场。不直接用播种亩次生成经营权，也不推定旧租约双方。 |
| 资金、劳动、非农机会 | F1 基期现金扣单列保留额；资金和非农收入有F1组别代理。用NFPS劳动力人数及F1的120/180/176日组别口径，四分之一为农时决策窗口；对应非农日工资从归档收入及潜在日数派生，窗口外工资年末结算。没有配对CFPS成员、真实时长或真实日工资。 |
| 市场与生产 | `core/land_succession.py::Succession.step` 是唯一年度入口。经证实可出租土地才允许释放和搜索，省级竞价按亩比较、最低接受价与边际中标价取中点。按授标价上界、交易费、启动成本和生产资源逐宗扣减报价容量。合同设为两年，实种仍受租费支付后的流动性与到岗劳动约束。 |
| 作物技术 | 保持7a的产出、售价、成本及农时每亩参数，不用产量残差给各户赋能力，也不重估F1。每个单元一年一作，质量统一为1。变量面积的小组合精确比较、大组合保留三种排序的最佳可行结果，并非全局优化保证。现金有正现金余额，不等于足够支付租地和播前投入。 |
| 财政与策略 | `core/regional_succession_budget.py` 在年初共享唯一上限，21种动作包含P0、各区集中包和相同工具的均分包；工具区分普遍/承接、播前/产后。名义85元/亩受分区预留与物理亩限制，不承诺观察产出后再拨款。公开状态仅为期剩、上年地区粮食二档和上限。 |

上限为联合样本物理亩乘18.75并向下取整，不是实际行政预算。样本初始粮食用调查合计作为公共滞后状态；以后使用模型实际产量。主实验第1至第8年同样可选全部动作。停付试验在相同第六年结束状态，从第七年起统一选P0；对比原策略连续执行和始终P0。面积累加列为亩年，现金库存优先看逐年地区账，不将跨年累计库存解释为期末余额。

## 原四区复算（仅在冻结父树）

从包的 `source` 目录执行，`$Project` 指向具备授权微观原表和原冻结归档的项目根。程序核验指纹且输出目录必须不存在。原始户表、户号、逐户现金和逐地账均不在复现包。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -B -X utf8 -m scripts.run_succession_empirical_budget train $Project run_training --workers 12
python -B -X utf8 -m scripts.run_succession_empirical_budget evaluate $Project run_evaluation --trained run_training --workers 12
python -B -X utf8 -m scripts.audit_succession_empirical_budget run_training run_evaluation run_audit
python -B -X utf8 -m scripts.check_succession_empirical_households $Project run_training run_history
python -B -m pytest -q -p no:cacheprovider tests --junitxml=run_tests.xml
```

无授权材料仍可从已打包的源表重建两张策略表并审核执行动作、共同预算、按区交接、现金结转和停付分叉：

```powershell
python -B -X utf8 -m scripts.audit_succession_empirical_budget ../evidence/training ../evidence/evaluation replay_audit
```

前16个开发种子为 `540001–540016`，探索训练为 `550001–550096`，评价为 `560001–560048`，不读取旧评价种子。固定用开发样本最优八年粮食动作；短视和Q从相同分叉拟合，只有后者计入后续收益。评价不训练也不重新选择固定对照。各区使用同一气候冲击分布，地区随机流分开，区域种子为 `4*seed+region_index`；同一地区各策略共用同一组冲击。焦点是同样的家庭样本中改变财政规则，不做全国总体加权或估计现实政策因果效应。

技术环境：Python 3.11.2、NumPy 1.26.4、pandas 2.2.3、SciPy 1.15.2、pytest 9.0.2；授权Excel字典读取另需 openpyxl 3.1.2。参见[实验登记](design.json)、[ODD+D](../ODD+D.md)及[结果和验收](结果与验收.md)。旧实验的程序指纹在各自冻结工作树检查。
