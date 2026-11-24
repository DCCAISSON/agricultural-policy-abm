# 全国农业 ABM V4.1：土地市场边界与生产规模闭合

固定600户、60家合成农业企业和合作社，以及2011至2015年固定基础支持。R0是父提交 `5c24a91db015aaf9a76cf2656db92815c34abecb` 的V4行为修复版。V4.1在原年度运行入口做两项比较，不改粮经选择分布、工资、价格、投入、生产率、政府和八邻接设置。

| 规格 | 区域外部土地交易 | 经营方案后的第二次抽签 | 种子 |
|---|---|---|---|
| M0 | 关闭 | 原基准规则 | 103、109；301–320 |
| R0 | 关闭 | 保留 | 同 M0，另重复103 |
| R1 | 开启 | 保留 | 103、109 |
| R2 | 关闭 | 取消 | 103、109 |
| R3 | 开启 | 取消 | 同 R0 |

没有运行R4（见[研究说明](研究说明.md)）。本轮都属于历史回放，不是独立样本外验证。

- [研究说明与论文用结论](研究说明.md)；[ODD+D与三张核心图](ODD+D.md)；[参数与来源](parameters.csv)。
- [四项验收计数](acceptance.csv)；[20种子配对均值、分位数和区间](paired_20.csv)；[各规格逐种子逐年同户比较](annual_behavior.csv)；[四因素面积分解](area_shapley.csv)。
- [调查土地覆盖](survey_land_coverage.csv)；[外部池参数](pool_profiles.csv)；[估计分母与时间界限](pool_capacity_sources.csv)；[内部、外部与财政账](accounts.csv)。
- [输入指纹](source_receipt.csv)；[相同种子复算](reproduction_checks.csv)；[测试记录](tests.xml)。

## 复算

从冻结提交启动。使用项目既有Python环境、已授权的NFPS原表及配套原问卷/变量字典、冻结队列与作物预算，按[数据配置](../../national_abm_v4_data_driven_20261124/run_01/README.md)设置。原基准对照读既有本地输入，`AP_ABM_CHFS2011` 继续指向授权的 CHFS 2011 家庭表。无权访问输入时，可查看脱敏结果和运行焦点单元测试，不声称复现真实队列。

```powershell
python -B -X utf8 -m scripts.run_national_household_portfolio v41-suite results/local_v41_replay --workers 4
python -B -X utf8 -m scripts.audit_household_portfolio_behavior audit-v41 results/local_v41_replay final
python -B -X utf8 -m pytest -q tests/test_v4_area_economy.py
```

指定尚不存在的输出目录。逐户数据、合同和家庭身份只写入本地 `PRIVATE_DO_NOT_SHARE`。地区和年份汇总、代码及复算命令可公开；不能由情景参数推断全国真实土地、产量或政策效力。
