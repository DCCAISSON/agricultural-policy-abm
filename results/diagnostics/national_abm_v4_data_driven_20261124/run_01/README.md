# 全国农业ABM V4：源码、实算与复现

唯一实施规格为本地《执行v4任务书.docx》（SHA256 61cb798699dc188868fe08dd0876c48c15df41b0f5e18df9b439a0d8794a702c），父提交 `2ca69fd7e389f04af6e65db205a87c0bf2625bec`。本候选保留V3的真实600户队列，不对发布树上传原调查和逐户账户。

- [论文用研究设计与实算结果](研究设计与实算结果.md)
- [ODD+D、主体图、年度流程图和空间图](ODD+D.md)
- [参数、问卷、代码与政策执行对照](参数追溯与执行链.md)
- [文献与模型规则对照](参考研究.md)
- [逐年行为](behavior.csv)、[家庭和组织年度账户](accounts_summary.csv)、[20配对种子不确定性](paired_seeds.csv)、[收支和市场敏感性](sensitivity.csv)
- [完整的字段口径及缺失覆盖](field_mapping.csv)、[校准系数](behavior_parameters.csv)、[政府门槛](government_gate.csv)、[测试回执](tests.xml)

## 授权环境的一键复算

在具有原项目授权数据、原模型依赖和旧版回放账的根目录，使用本冻结提交的程序。公开仓库不包含受限 NFPS、CFPS、CHFS、成套作物预算逐行表、旧队列及逐户账，因而不能用公开下载替代授权环境。源文件指纹见 source_manifest。CHFS路径通过环境变量提供；本发布的主实验使用CHFS2011家庭表，不设该变量时改用同一数据源NFPS的既有地区和规模回退，不能声称得到完全相同的M2–M4。

```powershell
$env:AP_ABM_CHFS2011 = "<已授权的 chfs2011_hh_20191120_version14.dta 绝对路径>"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe -B -X utf8 -m scripts.run_national_household_portfolio v4-suite results/diagnostics/national_abm_v4_reproduction --workers 5
```

既有年度入口：`python -m scripts.run_national_household_portfolio v4 M4 NEW_DIRECTORY --seeds 103 109`。指定新文件夹，程序不覆盖旧运行。M0也可用前版 `crop-mix D` 启动。M0–M4使用103、109与103重复；新增201至220号种子只跑M0与V4。敏感性用103、109；逐户标量参考用103。相同种子共享地区天气流和固定初始条件，决策抽样独立按主体与年份稳定派生。既有官方补贴口径未通过训练准入门槛，所有发布实验均采用P0，不排名政策。

读取已有完整运行时用 `python -m scripts.audit_household_portfolio_behavior audit-v4 RUN_ROOT`。同一目录的 `PRIVATE_DO_NOT_SHARE` 是账户、合同、订单、地块布局和身份衔接的本地入口，不公开。公开焦点测试不需要受限样本：`python -m pytest -q tests/test_v4_area_economy.py tests/test_v3_crop_response.py tests/test_continuous_crop_history.py`。

先查看行为与账户两个验收结论。程序守恒和复算通过不代表五年行为通过。表中的配对区间只反映模拟种子不确定性，不是调查推断或现实政策置信区间。
