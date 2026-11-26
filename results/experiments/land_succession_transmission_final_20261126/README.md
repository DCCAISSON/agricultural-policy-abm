# 财政支持与土地经营接续：正式结果索引

[研究结果与复算](../../../docs/土地接续理论模型/复现与结果.md) · [ODD+D](../../../docs/土地接续理论模型/ODD+D.md) · [参数和文献](../../../docs/土地接续理论模型/文献与参数证据.md)

## 结果和比较对象

| 文件 | 内容 |
|---|---|
| `annual.csv.gz` · `annual_means.csv` · `world_means.csv.gz` | 186,800个年度记录及年度、种子路径均值；同一条八年路径不是八个独立样本。 |
| `paired_effects.csv.gz` · `distribution.csv.gz` | 五项政策、同合同支付时点、2×2交互、机制敏感性和独立策略评价的配对差、分位数和区间；路径汇总使用同一组独立种子。 |
| `transmission_accounting_summary.csv.gz` | 新增产量、地租的价格和面积分解、同一地块租金、相对P0未增产的领款、租金与收入分别核算。 |
| `transactions.csv.gz` · `fiscal.csv.gz` | 合同、挂牌和未成交明细，以及地块资格、承诺和实际支付。实验、参数情景、种子、年、策略和地块ID共同组成联结键。 |
| `withdrawal_cohort.csv.gz` · `cessation_paired_effects.csv.gz` | 停止支付后固定地块的经营者、产量、利润和现金；同地块的停付、未支持和持续支持比较。 |
| `training_sar.csv.gz` · `q_table.json` · `q_visits.csv` · `training_checkpoints.csv` · `policy_choices.csv` | 政府状态、动作、奖励、策略和评价。训练的0.95覆盖门槛没有通过；未以训练奖励替代正式比较。 |
| `execution.json` · `manifest.json` · `audit_provenance.json` | 分组账表行数、种子、情景、程序输入指纹，和独立核账程序及输入指纹。 |
| `gate.xml` · `regression.xml` · `ledger_verification.json` · `reproduction.json` · `baseline_replay.json` | 新模型测试、全量账目核验、预先指定路径独立复演、上一版复算。公开XML省略本地计算机名称，测试结果不变。 |

固定合同组每年以P0合同及其期初资源为条件，预先承诺相同金额后只改支付时点。内生市场组只保证政策使用相同年度预算上限，实际支付可不同。货币、产量和面积按不同单位分别报告，不当作同一笔财政金额的三个份额。

## 完整账与发布范围

`plots.csv.gz`、`operators.csv.gz` 为完整的合成地块账和经营主体收支账，分别约245MB和559MB，超过GitHub单文件限制；保存在本地可交付复现包中。公开表可按上方复现命令生成这两项明细，其输入哈希可用 `audit_provenance.json` 校验。公开与本地包均不含NFPS、CFPS或CHFS原表、真实户号和真实租赁合同。
