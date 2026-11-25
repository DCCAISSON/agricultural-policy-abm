# 全国农业经济ABM：V4家庭行为一次修复

**土地经营接续理论模型。** 本分支从F1冻结提交继承土地竞价和政府Q更新，在独立合成区域执行土地接续、支付时点、资源约束和政府独立评价。[当前进度、交付文件和已知问题](docs/土地接续理论模型/交付与当前状态.md) · [模型、结果和复算](docs/土地接续理论模型/复现与结果.md) · [参考资料与可公开下载的PDF](docs/reference_project/土地接续参考资料索引.md)。F1经验候选未修改，此处结果不作现实政策排序。

**F1最终综合候选。** 从V4.1固定版本统一家庭参与与可行粮经方案，复核经营面积、劳动、生产和跨年度账户。实现与审计已完成，联合行为检验未通过；[查看完整证据、模型说明和复算](results/diagnostics/national_abm_v41_final_economic_closure_20261124/run_01/README.md)。

**V4.1 规模闭合候选。** 在固定600户上分别比较区域外部土地池和取消经营方案后的重复抽签；保留原粮经选择及历史版本。查看[本轮源码配套结果、ODD+D和复算](./results/diagnostics/national_abm_v41_scale_closure_20261124/run_01/README.md)。本轮不排序政策，不训练政府。

[本轮诊断、修复结果、测试和一键复算](results/diagnostics/national_abm_v4_behavior_repair_20261124/run_01/README.md)。2011年参与和2015年兼种偏差缩小；联合行为门槛仍未通过，不启动政策排序或政府学习。

## 数据驱动V4候选

固定V3的600户，以明确问卷和实际原表估计粮经调整，不把同户未来选择、历史参考份额或固定阈值作为配种约束。主体的预期收益、资源约束、市场成交、实种与政府支付由同一年度路径核算。行为门槛未通过，政策比较和政府RL训练保持关闭。[V4结果、程序及复算入口](results/diagnostics/national_abm_v4_data_driven_20261124/run_01/README.md)。

## 家庭动态经营：粮经收益响应第三候选

V3保留V2的源码路径与结果，新增NFPS联合分层队列，以及允许收益改变作物面积的家庭选择。见[五年对照、模型正文及复算入口](results/diagnostics/national_crop_choice_soft_response_20261123/run_01/README.md)。新队列软规则通过6/10项年度与种子行为检验；不作为补贴政策排序依据。

## 连续份额第二候选

见[同户实算、数据口径与复算入口](results/diagnostics/national_crop_mix_data_comparability_20261123/run_01/README.md)。相比下列第一候选，十个年份与种子组合中，原定行为检验通过数由六项增至七项。2015年两条路径仍未通过。本版用来复核份额的形成与跨年传播，不作为政策评价基准。

## 第一候选的版本记录

本次公开源码对应本地冻结提交 `b5a7df5bfd384c73f156cfe8f01d5475fedea6ce`；其直接父提交是 `a8ef70620028b6031fedfe874cfdc7d5acd22dd6`。本地完整补丁 SHA256 为 `47a0f4b5fa7b13127920fc3e2a24aaf7626a5324271820a56ca14c7379d4d2a1`。公开版本为脱敏独立历史，保留研究源码和汇总表，不发布本地父版本、原始个体资料或私人提交邮箱。授权研究环境用于复算五条路径，公开环境可核对决策代码、运行汇总和不需要受限输入的测试。

本轮结果：工程和账户通过，家庭行为未整体通过。保留候选，不替换旧模型，不启动政策排名或政府训练。

**怎么读。** 先读 diagnosis.md。behavior_before_after.csv 的 all_gates 是最终逐年判断，passed 仅含同年两项。crop_mix_distribution.csv 按匹配家庭计算四类数量、面积和分位数；零播种户计入参与率，不填入份额分布。household_behavior_summary.csv 是同户总账，production_summary.csv 是收获；市场表覆盖所有参试主体而非仅调查匹配户。repeat 是复算，不能并入平均。grain_area_decomposition.csv 使用全部600户，勿与527户表直接相减。图中为种子103，两种子的数值都在表内。

**如何运行。** 在有授权输入的项目根目录，使用原环境 Python。将下方 OUT 换成一个新目录的相对路径；运行器拒绝覆盖旧路径。原户表、成员表、字典、作物预算、队列和配置指纹见 run_manifest.json。公开源码不是受限数据的替代品。

```text
python -B -X utf8 -m scripts.run_national_household_portfolio crop-mix baseline OUT/PRIVATE_DO_NOT_SHARE/baseline
python -B -X utf8 -m scripts.run_national_household_portfolio crop-mix candidate OUT/PRIVATE_DO_NOT_SHARE/candidate
python -B -X utf8 -m scripts.audit_household_portfolio_behavior crop-mix OUT
```

两次模拟、账户复核、相关回归测试均以状态0退出。10个原测试文件全部保留，新增3个参数化方向及资源测试，合计192项通过，0项跳过。精确命令见 run_manifest.json。新行为抽数按命名空间、种子、年份和家庭标识生成，与地区天气随机数独立。重复路径对名册、订单、合同、收支和政府记录作文件比较。生产、劳动、资金和租金按人按年复算，政府仅执行P0。

**版本。** 本地精确父版本 a8ef70620028b6031fedfe874cfdc7d5acd22dd6。公开版本按研究负责人授权使用独立历史，不传原父提交。2026年11月23日冻结，核对具体发布提交请使用指定同名分支和版本标签。
