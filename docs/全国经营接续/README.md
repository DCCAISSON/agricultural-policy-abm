# 农业经营接续模型、经验核验与论文

**本轮材料：** [家庭行为开发比较与长期验证条件](开发比较/README.md)。下方为此前封存的历史回放，不是完成的统一经验模型。

研究问题：**有限财政支持在什么条件下转化为持续的粮食生产能力？** 以全国固定观察点家庭约束经营建模，合成土地单元只用于核对模型内的承接。不存在合格独立递归留出时，不在全国候选上做正式政策排名或政府训练。当前交付按方案 C 保留候选及验证，合成市场机制实验单列。

1. **源码与复现。** [模型说明与三个机制图](ODD+D.md)、[字段和参数证据](逐字段证据.csv)、[冻结验收协议](validation_protocol.json)。F1 与 7a 的原提交、参数和结果不变。
2. **验证与裁决。** [报告](验证与裁决.md)、[历史回放表](回放结果/)、[两种冻结模型及数据审计](双模型与数据证据.md)、[7a 合成机制及政府学习回读](7a回读/复算说明.md)。
3. **论文。** [七部分初稿](论文初稿.md)，附中英文摘要、图表、来源与参数说明。

## 合成的最小核账输入

不需要调查文件即可从候选工作树运行下列命令。已编入测试的极小输入含两个模拟经营者及土地账户；覆盖同一经营者、换人且两期种粮、未知外部出处和签后未播的区别。随后执行继承的面积经济与独立 7a 回归。本部分不等同于 NFPS 原数据复算。

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_empirical_succession.py tests/test_v4_area_economy.py tests/test_land_succession_transmission.py --junitxml=tests-recomputed.xml
```

Python 和直接依赖版本见 [环境快照](environment.txt)，基线包依赖见项目根目录 `requirements.txt`。不要覆盖冻结结果。

## 授权环境内历史回放

下列命令在候选树内执行。`$Project` 是保留原调查、原队列、配置及 F1 本地证据的项目根；`$Chfs2011` 是本地依法可读的原有参照户表路径。后者用于原 F1 来源检查，不跨表拼户。案头复算使用相同环境，运行结果目录须不存在。两项线程配置应在启动 Python 前设置。由于重新拟合未能逐位复现旧系数，正式历史回放只使用 F1 原运行实际留存、与样本分组逐项核对并经 SHA256 校验的系数；新拟合仅作数值诊断。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:AP_ABM_CHFS2011 = $Chfs2011
python -B -X utf8 -m scripts.run_empirical_succession $Project ../replay_301 --seeds (301..305)
python -B -X utf8 -m scripts.run_empirical_succession $Project ../replay_306 --seeds (306..310)
python -B -X utf8 -m scripts.run_empirical_succession $Project ../replay_311 --seeds (311..315)
python -B -X utf8 -m scripts.run_empirical_succession $Project ../replay_316 --seeds (316..320)
python -B -X utf8 -m scripts.verify_empirical_succession $Project ../verified ../replay_301 ../replay_306 ../replay_311 ../replay_316
python -B -X utf8 -m scripts.plot_empirical_succession ../verified ../replay.png
```

其中逐户、原始身份、合同明细和合成区间的经营者联结都保存在输出的 `PRIVATE_DO_NOT_SHARE`，不随包分发。公开的家庭比较使用未加权同户样本；算法初始抽样权重不能当作验证权重。若无原始授权，只使用包内合成测试与汇总。

## 独立合成模型回读

```powershell
python -B -X utf8 scripts/readback_succession_7a.py --frozen $Frozen7a --output ../verified_7a
```

`$Frozen7a` 须含冻结 7a 的 `formal/` 及 `worktree/results/experiments/land_succession_transmission_final_20261126/`。已有数据的回读会核对十个源文件 SHA256，不重训 Q，不重新运行正式政策。详细复算和表义见 [7a说明](7a回读/复算说明.md)；与候选源码、数据及制图全部对应到本地同一诊断冻结。
