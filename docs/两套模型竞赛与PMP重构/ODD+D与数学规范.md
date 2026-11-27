# 不可分农户 AHM-PMP 与双层共同预算模型：ODD+D 说明与复算指南

## 1. 概述（Overview）

### 1.1 研究目的
本模型旨在以机制可解释、账目闭合且与既有线性规划基准严格隔离的方式，检验有限公共财政支持进入多区域农户—土地流转系统后，在何种要素与制度条件下保留为粮食生产能力，在何种条件下表现为地租资本化、经营主体间再分配或预算未支削弱，并检验基于宏观结构指标的双层政府强化学习策略相对固定规则与同信息短视规则的真实表现。

### 1.2 主体、空间与时间尺度
- **微观权利与生产主体（600 户真实调查家庭，`HouseholdAgent`）**：初始化自 2010 年全国农村固定观察点冻结队列（覆盖东、中、西、东北四大政策区与 29 个省级编码，物理经营面积 `5,277.5` 亩，其中已核实可流转承包地 `4,009.2` 亩）。样本户数不足 12 户的极薄省级切片并入同政策大区内部子市场，消除“单户省份零交易对手”的人为结构死锁。
- **区域规模承接主体（`RegionalEntrant`，`is_entrant=True`）**：每个子市场配置受同等营运资金、劳动/雇工成本及固定管理费约束的家庭农场型承接者，期初自有地为 `0` 亩，仅能通过流转竞租取得耕地经营权，收支与产量同 600 户原权利家庭分账统计。
- **中央/跨区共同预算规划者（`SharedRegionalBudget` + `FQI Planner`）**：每年初支配固定总额 `98,953` 元（`floor(18.75 * 5277.5)`），在 21 个合法区域—工具组合动作中选择当期财政配置方案，主实验连续决策 8 年，另设第 6 年末停补分叉检验第 7–8 年产能留存。

---

## 2. 核心数学规范（Design Concepts & Mathematical Formulation）

### 2.1 不可分农户三段劳动供给与凹二次 PMP 生产规划
设主体 $i$ 在区域 $r$ 第 $t$ 年支配可耕物理面积 $A_{i,t}$，复种强度上限为 $\kappa_i$（取自 2010 基期观测复种指数并受区域农艺上限约束），最大可播亩次为 $\bar{S}_{i,t} = \kappa_i A_{i,t}$。主体选择总播种亩次 $S \in [0, \bar{S}_{i,t}]$ 与粮食播种份额 $\theta \in [0, 1]$（其中 $x_g = \theta S$，$x_c = (1-\theta)S$，实种物理面积 $A^{\text{plant}} = S / \kappa_i$，闲置物理面积 $A^{\text{idle}} = A_{i,t} - A^{\text{plant}}$）：

$$\max_{S \in [0, \bar{S}_{i,t}],\, \theta \in [0, 1]} \; \Pi_{i,t}(S, \theta) = R^{\text{gross}}_{i,t}(S, \theta) + S^{\text{pre}}_{i,t} + S^{\text{post}}_{i,t} - C^{\text{mat}}_{r}(S, \theta) - C^{\text{hire}}_{r}(h_{i,t}) - C^{\text{opp}}_{i,t}(f_{i,t}) - C^{\text{PMP}}_{i,r}(S, \theta)$$

1. **三段非分离劳动成本（de Janvry et al., 1991）**：总农时需求 $L^{\text{req}} = (l_{r,g}\theta + l_{r,c}(1-\theta))S$。家庭农时上限 $L^{\text{fam}}_{i,t}$ 分为村内非竞争闲暇农时 $L^{\text{slack}}_{i,t}$（按保留负效用 $w^{\text{res}}_r = 0.12 w^{\text{hire}}_r$ 计价）与挤占非农务工的竞争窗口农时 $L^{\text{comp}}_{i,t}$（按 $\min(w^{\text{opp}}_{i,t}, w^{\text{hire}}_r)$ 计价）；超出 $L^{\text{fam}}_{i,t}$ 的缺口触发市场雇工或社会化农机作业购买 $h_{i,t} = \max(0, L^{\text{req}} - L^{\text{fam}}_{i,t})$，单价 $w^{\text{hire}}_r$。
2. **Howitt（1995）一阶齐次凹二次 PMP 隐性成本**：
   $$C^{\text{PMP}}_{i,r}(S, \theta) = \Delta \bar{\pi}^0_r (1 - \theta) S + \frac{1}{2} \gamma_r S (\theta - \theta^{\text{ref}}_{i,t})^2$$
   其中 $\Delta \bar{\pi}^0_r$ 由 2009–2010 年基期粮经单位净收益差唯一标定，$\gamma_r$ 控制边际转换曲率，保证在基期价格下精确重现内点兼种解且具备规模一阶齐次性。
3. **春耕播前营运资金硬约束**：农资费用与春耕预付雇工份额（`45%`）扣减播前到账补贴后不得超过当期可用农业营运周转金 $W^{\text{work}}_{i,t}$：
   $$C^{\text{mat}}_r(S, \theta) + 0.45 w^{\text{hire}}_r h_{i,t} - S^{\text{pre}}_{i,t} \le W^{\text{work}}_{i,t}$$
   秋收结算部分（`55%` 雇工费）与产后补贴 $S^{\text{post}}_{i,t}$ 在收获后进入现金流，先回收周转金本金，再按留存率补充下期营运资金。

### 2.2 合同期增量影子价格序贯竞租
1. **挂牌释放与统一可行账**：年初扣除在履两年期租约租金后，原权利农户在同一 AHM-PMP 目标下比较“全留自种”“部分转出可流转地”与“全部转出可流转地退出自耕”的期望总剩余（含预期租金），内生决定挂牌供给。
2. **增量影子价格报价**：潜在承租者 $i$ 对面积 $a_p$ 地块的最高愿付单价由并入前后的最优经营剩余增量减去按合同期摊销的登记搜寻费给出，并受春耕周转金上限硬约束：
   $$b_i(p) = \min\left( 0.85 \cdot \frac{\Pi_i^*(A_i + a_p) - \Pi_i^*(A_i) - \tilde{F}_{\text{fee}}}{a_p},\; \frac{W^{\text{work}}_i - F_{\text{fee}} - U^{\text{net}}_i(A_i + a_p)}{a_p} \right)$$
3. **序贯出清与未成交回撤**：子市场按最高有效报价序贯成交，成交单价取保留价/次高价锚点与最高报价的中点；每成交一笔立即扣减承租方营运资金并更新其已支配面积。未获买方匹配的挂牌地块退回原主，在零租金条件下重新按可行性决定实种或休耕。

### 2.3 共同预算分配与双层正则化拟合 Q 迭代（Ridge-FQI）
- **预算与承诺封顶**：四区共享 `98,953` 元/年；动作菜单包含 `P0`、16 项区域重点投放（超出 `85` 元/物理亩法定上限部分按其余产区耕地面积比例自动溢出回流）及 4 项全国按耕地面积比例均等投放。任何工具若合格申领总额超过分区预留额，按比例等额折算并触发紧流动性农户定点重解，严格保证 `0 <= spending <= committed <= reservation`。
- **去噪结构状态与逆向 FQI**：政府年初观测去除单年随机气象冲击后的四区结构指标（粮食实种率、未吸纳流转与闲置率、在履租约率、营运资金紧约束户占比），在 96 个独立训练种子的精确 21 分叉反事实轨迹上按有限期逆向递推（$t = 8 \dots 1$）拟合岭回归价值矩阵（`q` 取跨期累计产量 $\gamma=1$，`myopic` 取当期产量 $\gamma=0$）；安全回退动作固定为全国按面积比例播前普惠（`action 17`），彻底废除惩罚性归零（`P0`）。

---

## 3. 一键复算命令

在干净工作树根目录运行（`<项目根目录>` 指向含授权 2009–2015 固定观察点源表与冻结队列文件的本地工程根目录）：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

# 1. 运行新旧全部单元测试（29项）
python -B -m pytest -q tests/test_competing_pmp.py tests/test_succession_sprint.py tests/test_succession_empirical_budget.py tests/test_land_succession.py

# 2. 运行第一阶段：基期校准、一步条件预测、五年递推与机制消融
python -B -m scripts.run_competing_pmp_validation --project <项目根目录> --out <新输出目录_stage1>

# 3. 运行第二阶段：16开发种子筛选 + 96训练种子FQI拟合 + 48独立新种子评价
python -B -m scripts.run_competing_pmp_budget_rl --project <项目根目录> --out <新输出目录_stage2> --workers 16

# 4. 运行独立双式记账、重放确定性、隐私脱敏与两套模型竞赛审计
python -B -m scripts.audit_competing_pmp --project <项目根目录> --worktree . --stage1 <新输出目录_stage1> --stage2 <新输出目录_stage2> --out <新输出目录_audit>
```
