# 四区农业经济内核：纠错、证据和复算

## 交付范围

以 476b492 为父版本。保留 F1 的 600 户、29 个省级样本边界、四区财政壳和历史结果，在原年度入口替换自营、竞租与实际生产所用的经济规划。没有用 2014–2015 年改价格、用工或转换概率；没有启动新的政府训练、重放旧策略进新环境或排名。新规则用显式技术参数启用，旧入口不自动切换。

主检验均无新增专项，沿用 571001–571004 的共同冲击并覆盖 2011–2015：M1 冻结规则、旧技术参数配同一新决策账、以及具名活动预算候选。后两项的差不是单一价格效应；同时改变了作物菜单和投入、产量及劳动定义。同期实际经营亩、年度播种亩次、三种谷物公斤与调查全部粮食公斤不能混用。三组不是正式政策比较。

旧四区实际调用的是 `core/succession_allocation.py` 的合成两作物预算：粮、经价格均 2.50 元/公斤，扣播前投入后分别剩 505、635 元/亩，尚未扣劳动、租金或费用。两个价格相等不是数学错误，但原现金作物没有可核实的产品身份。更早环境的默认值 1.25 和 1.80 元/斤没有被四区调用，也不是经过本轮核实的实际销售价格，未直接移植。

## 参数与来源

| 规则或参数 | 证据等级与实际来源 | 使用限制 |
| --- | --- | --- |
| 家庭、土地、基期工资和现金 | 2010 队列、NFPS 辞典、F1 冻结基期账和缺项代理；prepare() 按原指纹及同户原值核对 | 可用现金为存款口径扣原保留额；非农工资按原劳动窗、工作机会和收入代理构造，不是逐季观测 |
| 三种谷物 | 2010 非队列原表的播种亩、产量、报告总收入、投工及八项可比投入；按同记录面积总量归一，见 activity_evidence.csv | 缺费用不填零。产值/产量是报告单位产值而非成交价；早期销售份额缺失，现金端只取全售结构边界 |
| 蔬菜 | 2016–2017 原表 d50/d86/d106/d472 与分项费用，NH5 销售 e33/e34；CPI 换算至 2024 元 | 是具名的一类蔬菜，不替代全部经济作物，不代表 2010–2015 的相对价格；销售/产量同记录核验，超产销量不纳入份额 |
| 生产投入和劳动 | 种苗、化肥、农膜、农药、水电灌溉、畜力、机械、小农具八项；家庭工日、雇工和非农分别核账 | 报告投入是年度费用；全部作为播前占款是保守结构设定。农家肥折价、折旧修理混合项、土地租金、雇工不在八项中重复计入；不是总生产成本 |
| 地租、费用和期限 | 从旧四区继承数值：130 元/亩年度保留价、280 元/新地块交易费、1,000 元/承租主体年管理费、两年合同 | 都是研究情景设定，期限折现系数 0.95 亦非估计；成交按保留价，不用于解释市场均衡租金或补贴资本化 |
| 遇险、雇工和经营现金 | 保留冻结资源冲击及可动用比例；另在固定期初土地下放松现金、家庭工日或提供有界雇工 | 主路径没有雇工；小范围稻谷雇工中位工资仅是对照用价格。机械成本保留，但无证据不生成额外机械供给或省工率 |
| 财政 | 继承事前额度及合格生产抵减，核账区分承诺、实付和支付时点 | 全区面积是担保分母，不是当期合格面积估计；播前抵减不可用于付租或雇工押金。正式结果只跑 P0 |

借鉴的是 [AgriPoliS 的有约束农场决策与租地安排](https://www.econstor.eu/bitstream/10419/216735/1/Happe_2006_Agent_based_analysis.pdf)，不是其技术系数或制度。线性规划首先作为共享可行性与收益账；没有使用 [Howitt（1995）](https://doi.org/10.2307/1243543) 的正向数学规划从单期面积反解未经验证的偏好或政策弹性。

## 结果索引

- activity_evidence.csv、enabled_parameters.csv、dictionary_trace.csv：实读字段、有效分母和实际生效单位。
- household_history.csv：逐年、规格、种子的同户汇总，不含户号。region_year.csv、crop_year.csv：地区年度账及具名作物实物量。
- opening_resource_sensitivity.csv：固定已知期初土地组合上的资源对照。nfps_reported_land_scope.csv：两项均报告的地租流转记录及单独的期初期末面积分母。
- member_labor_coverage.csv：核对成员辞典及原表，分别列有效记录及真实零。2010 年的 80,061 条成员记录里，本村农业、非农用工和外出收入均无报告，外出工日有 19,065 条；全年外出劳动不能换算成家户农忙可用工日。
- grain_labor_marginal_coverage.csv：预算完整样本与只要求播种和投工均为正的较宽分母。玉米分别为 21.26 和 9.32 日/亩，这一差异说明完整记录选择影响工日口径；本轮保留原候选不反调工日。
- ISSUE_REGISTER.csv 逐项对照独立纠错报告的 29 项；ODD+D.md 说明行为和年度顺序。

## 复算

在本候选工作树运行。project 指有授权原表和冻结基期输入的原工程目录，两个输出目录须尚不存在。脚本不写入原数据或已存在的结果。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m scripts.run_succession_economic_core --project <原工程> --out <新目录> --workers 8
python -m scripts.check_succession_core_resources --project <原工程> --out <另一个新目录>
python -m scripts.audit_succession_economic_core --main <结果> --replay <干净树结果> --resources <资源对照> --replayed-resources <干净树资源对照> --out <核验目录>
python -m pytest -q tests/test_succession_economic_core.py tests/test_succession_sprint.py tests/test_succession_empirical_budget.py tests/test_land_succession.py tests/test_land_succession_transmission.py tests/test_succession_shared_budget.py
```

旧两区脚本保留字节指纹校验，须在自己的封存树 1ba724d 运行其 6 项测试。不能取消锁定后拿新农业源码重放旧实验。
