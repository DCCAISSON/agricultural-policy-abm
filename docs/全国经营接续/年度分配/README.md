# 两地区逐年分配共同财政预算

## 版本与运行

新实验 `7a-AB-annual-6plus2-v1` 继承 `81279282e0a9d9d5d72855a07f4614422374c7c6`（树 `515be6887ed3f41297e750a7564da007a07a20b3`）；[设计](design.json)于 `819b48a02832b29fced71799637797d2cffac116` 先提交，后训练和评价。程序版本 `801ca7417fecb066baaf2ea40c4ca825951ce2da`。F1、I/C 开发裁决、7a 农业代码和首年拨款均未变。

一个政府只在前六年配置两区的7,200元共同年度额度，第七、八年强制P0。P1为普遍播前支持，P3为合格新承接者播前支持。区内低、高预留3,600、7,200元，对应18.75、37.50元/亩。每个符合资格且实际种粮的4亩单元可以支付75或150元；承诺或预留不是实付。未花额度不补拨、不跨年。A、B每区192亩，只有财政相互约束。

96个开发种子 `410001–410096` 枚举13种地区固定配额，选 `fixed_02`（仅B高档P1）；`fixed_00` 为始终P0，`fixed_06`、`fixed_11` 为两区分别各半P1、P3。两种动态规则共用均匀菜单探索下的同一开局、两区上期粮食档及剩余年数和财政上限。粮食档以地块满种时的未调整参考产量标准化，档界0.55/0.70/0.85；第一年使用同一模拟公开初始汇总。每个开发开局生成13种动作的共同冲击分叉，只算一个开局。分叉按地区局部分解，自身转移与财政竞争以外的地区状态无关；测试对照直接逐动作运行的现金、合同、经营者、作物和停补粮食。Q按剩余期限反向估计累计产量，短视只看本年；最后一个支持年的Q目标包括随后两年P0。稀疏和未见状态都回退P0；本实验不把公开状态视为家庭经济完整状态。

三次训练分别用 `420001–420192`、`421001–421192`、`422001–422192`，每次1,152个开局，合计44,928行开局动作转移。每年达到该次回合数1%（向上取整，至少2次）的状态纳入常见状态分母，每对至少5个开局才支持。三次分别为884/1,053、858/1,066、884/1,079，均不足95%。第一次预定为主结果；不重选训练轮次。评价 `430001–430128`：每个种子比较13个固定策略、3个Q、3个短视及主Q/短视各1个关闭现金结转对照，共2,688条两区八年路径、43,008个地区年度。每区冲击索引为 `2*seed + region_index`。新的是随机路径，不是新地区或现实数据。

## 取数与裁决

- [方案均值](结果/strategy_means.csv)：按策略及A、B、合计；累计面积为亩年，现金跨年和是年度余额加总，某年余额查地区年度源账。`retained_grain` 是同一经营者在同地连续两年种粮，`continuity_area` 是同地换人且两年均实际种粮，`leased_idle` 是在履租约对应未播面积。成交或面积指标不代替实际收获。
- [配对估计](结果/paired_effects.csv)及[研究裁决](结果/research_decision.json)：主Q对两项主对照的三类终点共六项同时区间，其余为说明性95%区间。区域结果不按地区年度当独立样本。相近实付容差±720元以同时区间整体落入为准。主Q对同信息短视少产2,220.56公斤，实付多116.60元；同时区间未落入等实付范围。对预选固定少花3,566.02元，未建立累计和停补产量无损。
- [实际强度](结果/effective_tiers.csv)、[动作频率](结果/action_counts.csv)、[回退](结果/fallback.csv)和[已测组合前沿](结果/tested_frontier.csv)。固定各半P1的评价均值为943,231.60公斤、31,146.09元，优于主Q产量点估计但实付更多；它没有在评价后替换开发期选出的固定对照。主Q的产量—实付均值被同信息短视支配，不表示对实付差已经作出统计判断。
- [现金敏感性](结果/cash_sensitivity.csv)及[核验回执](结果/audit_receipt.json)。关闭现金只关跨年余额仍保留经营和合同规则；策略同样按各自新观察选动作，不重新训练。Q相对短视的产量差中差跨零。不可凭年度租约与现金的伴随变化归因于唯一机制。
- [旧拨款补充对照](结果/pulse_supplement.csv)和[带容差回执](结果/pulse_readback.json)。旧源文件只读，锁定绝对容差 `1e-6`、相对 `1e-12`，复核严格表键、数值及原裁决；本机原严格字符串审计也通过。补充六组对比跨工具、时间、粮食和实付的24个合计对比用Bonferroni校正；分区只作说明。不是旧第二阶段。
- 全国证据与政策边界见[验收记录](验收与经验准入.md)，论文见[主稿](../论文初稿.md)。

## 从复现包执行

从解压后的 `source` 目录运行。用 Python 3.11.2、NumPy 1.26.4、SciPy 1.15.2，可用 pytest 9.0.2 执行测试。目录名须未存在；复算不覆盖 `evidence`。运行时不读取任何真实家庭资料。

```powershell
python -B -m scripts.run_succession_annual_allocation train rerun_training --workers 8
python -B -m scripts.run_succession_annual_allocation evaluate rerun_evaluation --frozen rerun_training --workers 8
python -B -m scripts.audit_succession_annual_allocation rerun_training rerun_evaluation rerun_audited
python -B -m scripts.audit_succession_shared_budget ../evidence/old_pulse
python -B -m scripts.audit_succession_handoff ../evidence/old_pulse rerun_old_pulse
python -B -m pytest -q -p no:cacheprovider tests --junitxml=rerun_tests.xml
```

冻结策略和开发输入见 `evidence/training`；完整地区年度及评价回执见 `evidence/evaluation`；逐路径窗与统计结果见 `evidence/audited`。原单区 ZIP 单独解压后从其包根运行 `python -B -m scripts.audit_succession_government_fair results/synthetic_fair_comparison`，不用重新训练。两份历史 ZIP 均原样收录，不承载本地 Git 历史。新补丁从指定父提交用 `git am` 回放，压缩包不含受限户表、户号或逐户调查账。主稿对更早的外部调查归档仅提供定位，不打包受限数据。
