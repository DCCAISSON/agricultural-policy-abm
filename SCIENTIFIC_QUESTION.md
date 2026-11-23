# The One Scientific Question

**本轮位置：家庭经营的跨年传导。** 政策改变收益以前，模型须分清谁愿意种、谁取得经营土地，以及实际经营者怎样配置粮经面积。一步条件份额预测和连续五年经营历史分别检验；本轮只报告两种经营规则的比较，不把差额作为财政支持的效果。

> 项目唯一锚点。所有 spec、论文、PPT、图表、实验设计的故事必须从这一句出发，而不是从政策叙事、三轴框架、SCRI/IDI/AESI 任何一个分支出发。
>
> Single anchor. Every downstream spec, paper draft, slide deck, figure, and experiment must trace its narrative back to the question on this page — not to policy framings, not to the three-axis frame, not to any single diagnostic.

---

## 1. 那一句话 / The Question

**中文**

> **公共财政支持进入乡村食物系统后，其边际效力如何被系统内部的生产组织、资源配置与制度执行内生塑造；在什么条件下，支持能够保留为粮食生产能力，又在什么条件下被资本化、再分配或削弱？**

**English**

> **How is the marginal effectiveness of public support endogenously shaped by production organization, resource allocation and institutional implementation within rural food systems, and under what conditions is support retained as productive capacity, capitalized, redistributed or attenuated?**

---

## 2. 为什么这才是科学问题（不是政策问题）

**第一篇论文的研究边界。** 有限财政支持在什么条件下转化为持续的粮食生产能力？在不完全要素市场中，补贴改变农户、合作社和农业企业的预期收益，进而影响粮经配置和土地报价；能否成交以及承接者实际种什么，决定粮食面积、单产、收入和财政支出。维持粮食生产不一定要求原经营者留下。初始禀赋、天气、预算和制度菜单给定；价格与服务能力哪些给定、哪些反馈，以实验配置为准。土地成交与生产由主体互动形成，不另设区域产量目标方程生成结果。

生产能力指给定资源与技术下可实现的生产水平；经营约束指播前资金、劳动、可得土地或交易条件对可行方案的限制；政策附加性指相同外部条件下相对于无该项支持的新增生产。高产不等于受约束，受补也不等于有新增产量。个体条件生产率只有在播前预测中优于地区或经营组基准且适用于该作物时才考虑接入；否则保留组基准，不妨碍研究土地和财政配置。三项待检验的认识分别依赖同地经营与作物账、跨地区配对结果，以及只看既往区域生产和财政汇总的政府与固定或透明规则的同预算留出比较；三者不是预设结论。

科学问题与政策问题的分界线只有一条：

| 政策问题（不接受） | 科学问题（接受） |
|---|---|
| 政策好不好 / 应该不应该 / 怎么调整 | 现象为什么发生 / 在什么条件下发生 |
| 答案 = 政策建议 | 答案 = 一个机制理论 / 一组可证伪命题 |

本问题研究的是政策有效性的**条件形成机制**，而不是预先认定公共支持必然被吸收、制度安排必然能够修复：

- *how is effectiveness shaped*：比较生产组织、资源配置和制度执行如何改变支持的边际传导；
- *under what conditions*：允许生产性保留、资本化、主体间再分配和传导削弱分别为零、为正、为负或跨尺度不一致；
- *candidate institutional moderators*：村级协调、社会化服务、订单安排和信息治理首先是待检验的调节条件，不是已经成立的修复机制；
- *falsifiability*：如果支持能够稳定转化为粮食生产能力，或候选制度条件不改变传导结果，模型必须保留该结果。

设粮食生产能力的边际传导为：

`ME_Q = ΔGrainCapacity / ΔPublicSupport`

本项目将**吸收**定义为：公共支持进入系统后，由于经营权调整、经营主体重组、价格变化或制度约束，边际资源没有同比例转化为目标生产能力，而表现为其他价格、收入、组织结果或未实现响应。吸收不是执行失败的同义词，也允许为零。

只有当某项制度安排稳定提高 `ME_Q`，或稳定提高 `ΔSuccessionCapacity / ΔPublicSupport`，并通过机制消融与敏感性检验，才称为**修复机制**；否则称为 `non_binding institutional condition`。因此，本问题对应的是一组待检验命题，而不是预先写定的答案。

---

## 3. 与四条经典理论谱系的对话位置

| 谱系 | 经典命题 | 我们的位置 |
|---|---|---|
| Acemoglu / Robinson 制度经济学 | institutions → long-run outcomes | 把 "institutions" 展开为可建模的反馈结构 |
| Ostrom 集体行动 | collective action → resource governance | 把村集体作为集体经济组织主体显式建模 |
| Banerjee & Duflo 发展经济学 | local governance → public good provision | 比较地方组织条件如何改变政策传导 |
| Rodrik / Acemoglu 政策有效性 | policy as endogenous variable | 将政策边际效力写成系统结构的条件函数 |

这一定位决定了：**我们不是在 CGE + microsim 那一档对话；我们在和 *Why Nations Fail* / *Governing the Commons* / *Poor Economics* 这一档对话。**

---

## 4. ABM 在本问题中的角色

| 方法 | 给得出 | 给不出 |
|---|---|---|
| 计量识别（DID/IV/RDD） | 平均效应 | 主体重组和多渠道接续的完整年度路径 |
| CGE + 微观模拟 | 总量均衡与分配 | 经营权逐步重配和主体接续过程 |
| 机理性作物模型 | 生物物理响应 | 政策反馈层 |
| 系统动力学 | 反馈环路 | 异质主体 |
| **ABM** | **异质主体 + 内生市场 + 制度执行 + 政策反馈 + 长期路径** | — |

ABM 用于同时表示异质主体、经营权重新配置、制度约束和年度反馈。它承担机制实验，不替代外部因果识别，也不因结构更复杂而自动获得经验有效性。

---

## 5. 传导结果与制度调节条件

### 5.1 传导结果

公共支持进入系统后，模型分别观察以下结果。它们不是预先闭合的财政去向，只有统一货币单位并通过会计闭合时才能称为货币归宿分解。

| 传导结果 | 模型层观测 | 解释边界 |
|---|---|---|
| **保留为产能** Productive retention | 粮食产出、稳定种粮面积、粮食生产保留率、经营主体持续经营面积 | 支持转化为目标生产能力 |
| **资本化** Capitalization | 地租变化、经营权价格变化 | 需要价格和权利账户支持，不预设占主导 |
| **经营主体间再分配** Operator redistribution | 不同主体承接面积、经营面积分布及集中度 | 描述组织结构变化，不等同福利改善或损失 |
| **削弱或残差** Attenuation / residual | 未被生产、价格或主体账户解释的边际响应 | 必须保留，不得静默并入其他结果 |

### 5.2 候选制度调节条件

| 候选条件 | 待检验作用 | 当前纪律 |
|---|---|---|
| 村级协调 | 是否改变释放土地的匹配和生产接续 | 有稳定行为差异后才能称为修复 |
| 社会化服务 | 是否缓解劳动力或设备约束 | 零响应应标记为 `non_binding` |
| 订单安排 | 是否改变继续种粮和承接土地的收益比较 | 不预设必然提高粮食生产 |
| 信息治理 | 是否改善资格核验和资源匹配 | 缺少活动代码或数据时只作为后续机制 |

制度调节条件回答“什么改变传导结果”，不属于支持本身的结果去向。

---

## 6. 三个机制层诊断指标

三个指标**仅作为机制层诊断**使用，不作为绩效指标：

| 指标 | 全称 | 数学定义 | 文献对接 |
|---|---|---|---|
| **SCRI** | State-Capacity Retention Index | `ME_t / ME_baseline`，其中 `ME_t = Δ verified-food-security-outcome / Δ public-support` | fiscal pass-through to food security |
| **IDI** | Inclusion Diagnostic Index | 小农持续经营份额、真实净收入、生计内 Gini、兼业户韧性的复合诊断 | smallholder persistence + within-livelihood inequality |
| **AESI** | Agro-Environmental Sustainability Index | 农业碳强度、氮流失、耕地保护、生态产品价值实现的复合诊断（部分实现） | IPCC AR6 + EAT-Lancet 2.0 sustainability pillar |

三轴整体不再叫 trilemma；用 *multi-objective policy frontier* / 多目标政策协同前沿。

---

## 7. 中国的位置：高密度制度试验场

中国不是普通 country case，而是把粮食安全、小农转型、村集体、土地流转、财政工具、服务体系压缩在一起的**高密度制度试验场（high-density institutional case）**。

中国制度场景的不可替代性来自三件事：
1. 双层经营 + 三权分置 + 集体经济组织——世界其他国家没有的制度变量；
2. 财政化大规模托底转型——同时启动 SDG 2 / SDG 10 / SDG 13；
3. NFPS（2009–2015，144,086 户）+ CFPS（2010–2022 面板）+ NBS 锚定——少见的三层经验数据底座。

但本研究的科学问题对所有 smallholder-dominated food system 适用，不限于中国：印度、撒哈拉以南非洲、拉丁美洲都是潜在的可移植对象。中国是问题的**显化场地**，不是问题本身。

---

## 8. 与论文骨架的逐节对应

| 论文章节 | 该章节如何回答这个核心问题 |
|---|---|
| **标题** | 围绕政策传导的条件形成及生产接续机制，不预告吸收或修复必然成立 |
| **摘要** | 交代结构转型压力、生产接续机制、两尺度证据和实际结果；不得预先规定资本化或制度修复方向 |
| **引言段 1** | 投入-产出落差是一个反复被观察、未被统一解释的现象 |
| **引言段 2** | 既有研究分别处理支持效应、土地流转和经营主体变化，但较少追踪退出后的组织传导过程 |
| **引言段 3** | 缺口收紧到：政策边际效力如何随生产组织、资源配置和制度条件变化 |
| **引言段 4** | CFPS 锚定退出压力，全国 ABM 研究组织响应，衢州提供地方过程证据 |
| **Results F1｜结构转型压力** | CFPS 锚定农业生计退出压力；不要求全国 ABM 预测全部生计转换 |
| **Results F2｜生产接续机制** | 描述 `ReleasedArea → AbsorbedArea → GrainRetention` 及市场、直接流转、村级协调等渠道 |
| **Results F3｜接续能力与支持传导** | 检验 `∂GrainCapacity / ∂PublicSupport = f(SuccessionCapacity)`，不预设方向 |
| **Results F4｜支持传导结果** | 根据结果比较生产性保留、资本化、经营主体间再分配和削弱，不预设何者主导 |
| **Results F5｜制度条件** | 分别检验村级协调、社会化服务和订单安排是否改变 `Support → Succession → Grain`；零响应报告为 `non_binding` |
| **Discussion ①** | 与 Acemoglu / Ostrom / Banerjee / Rodrik 四谱系对接——我们补的是 "institutions → outcomes" 中间那条可观测的传导通道 |
| **Discussion ②** | 讨论观察到的传导转向是否构成吸收，以及哪些候选制度条件达到修复机制的证据门 |
| **Discussion ③** | 局限与展望：全国模型是经验锚定的机制型 ABM；ABM 内部世界线不确定性不等于现实预测误差 |

---

## 9. 与既有项目工件的同步

- **AGENTS.md** 的 *Canonical Storyline* 已升级为本文件的 §1。AGENTS.md 中关于写作纪律的条款（*Do Not Use* / *Evidence Boundaries* / *Contribution Logic*）继续完整保留。
- **`.kiro/specs/nature-story-state-capacity-trap/`** 旧 supersession notice 中的"国家能力陷阱"叙事不再是主线，但保留作为机制环节的解释语言（资本化、租金、土地控制重组），以本文件为锚点重新组织。
- 所有新建 spec（`scri-evaluator-and-experiments`、`inclusion-diagnostic-evaluator`、`agro-env-sustainability-evaluator`、`shock-recovery-rigor-fix`）必须在其引言部分引用本文件 §1 作为研究问题来源，且不得引入新的"那一个"科学问题。

---

## 10. 写作纪律（不变）

继承自 AGENTS.md，本文件再次明示，作为科学问题对写作的硬约束：

- **不**说"第一个建模框架"、"国际同行至今没有做到过"、"全球唯一活体案例"；
- **不**预言或声明"AI + 服务把前沿外推"；只检验不同结构条件下的传导结果；
- **不**预设吸收一定存在；生产性传导充分时，吸收可以为零；
- **不**把协调、服务、订单或信息治理预先称为修复；只有通过行为差异、机制消融和敏感性检验后才使用该名称；
- **不**把 SCRI 说成现实国家能力的直接估计；
- **不**把 paired-worldlines CI 说成现实预测误差，只说 ABM 内部世界线不确定性；
- **不**把 paired-worldlines 说成现实政策因果识别或外部实证识别（DID/IV/RCT）的替代；它只是 **ABM 内部配对世界线的机制反事实**，用于诊断模型内机制通道；
- **不**把全国合成基准结果直接转为衢州政策处方；
- 三轴整体**不**叫 trilemma；用 *multi-objective policy frontier*。

---

## 11. 修订记录

- **v1.0 / 2026-05-31**：首版固化。把核心科学问题从政策叙事升级为系统反身性的机制问题。这一升级是项目对外定位从"中国农业政策评估"到"复杂农业系统中政策有效性的内生理论"的根本性转换，所有下游叙事以此为锚。
- **v1.1 / 2026-08-14**：依据 V4.8 Task 5.5 证据，将总问题改为条件式、可证伪表述；区分传导结果与制度调节条件；生产接续成为当前论文的组织机制入口；删除资本化主导、长期必然衰减和制度必然修复等预设结果。
