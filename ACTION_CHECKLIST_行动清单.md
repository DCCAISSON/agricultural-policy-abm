# 3个月冲刺行动清单 ✅

**目标**: 完成模型验证，投稿国际期刊  
**截止日期**: 2025年3月6日

---

## 第1个月：模型验证 (最关键！)

### Week 1-2: 微观验证
- [ ] **Day 1**: 创建 `validation/micro_validation.py`
- [ ] **Day 2-3**: 实现前景理论损失厌恶检验
  - 代码: `test_loss_aversion(farmers)`
  - 期望: |损失效用| > |收益效用|
- [ ] **Day 4-5**: 实现社会学习邻居效应检验
  - 代码: `test_neighbor_effect(farmers)`
  - 期望: 邻居种经作 ↑ → 自己种概率 ↑
- [ ] **Day 6-7**: 实现Melitz企业排序检验
  - 代码: `test_melitz_sorting(enterprises)`
  - 期望: 规模 ∝ 效率
- [ ] **Week 2末**: 输出 `validation/micro_validation_report.pdf`

**检查点**: 至少3个微观机制通过检验

---

### Week 3-4: 宏观验证（最重要！）⭐⭐⭐⭐⭐
- [ ] **Day 1**: 提取CLES 2020的15个目标矩
  ```python
  cles_moments = {
      'mean_income': cles['S_A1_8'].mean(),
      'income_gini': calculate_gini(cles['S_A1_8']),
      'agri_share_median': cles['agri_share'].median(),
      'transfer_rate': cles['land_transfer'].mean(),
      # ... 共15个矩
  }
  ```
- [ ] **Day 2-3**: 运行100次蒙特卡洛模拟
  - 代码: `results = run_monte_carlo(n=100)`
  - 并行运行（8核约1小时）
- [ ] **Day 4-5**: 计算模拟矩与数据矩的距离
  - 公式: `Distance = √Σ(M_sim - M_data)²`
  - 目标: Distance < 0.2
- [ ] **Day 6-7**: Bootstrap拟合优度检验
  - 代码: `p_value = bootstrap_test(model, cles, n=1000)`
  - 目标: p > 0.05
- [ ] **Week 4末**: 输出 `validation/macro_validation_report.pdf`

**检查点**: 至少10个矩的相对误差 < 15%

---

### Week 5-6: 时间序列回测
- [ ] **Day 1-2**: 实现回测脚本
  ```python
  # 用2020初始化 → 运行2步 → 对比2022
  model = init_from_cles_2020()
  model.run(steps=2)
  compare_with_cles_2022(model)
  ```
- [ ] **Day 3-4**: 验证4个趋势
  - [ ] 粮食产量 ↑ (技术进步)
  - [ ] 兼业化率 ↑ (城镇化)
  - [ ] 土地流转率 ↑ (规模化)
  - [ ] 经作价格 > 粮食价格
- [ ] **Day 5**: 绘制对比图
- [ ] **Week 6末**: 输出 `validation/backtest_comparison.png`

**检查点**: 4个趋势与历史数据吻合

---

### Week 7-8: 敏感性分析
- [ ] **Day 1**: 定义15个参数的变动范围
  ```yaml
  parameters:
    - loss_aversion_lambda: [2.1, 3.9]
    - grain_subsidies_rate: [0.0, 0.30]
    - grain_price_elasticity: [-0.5, -0.2]
    # ... 共15个参数
  ```
- [ ] **Day 2-3**: Sobol采样（1024×34 = 34,816次）
  - 代码: `param_values = saltelli.sample(problem, 1024)`
- [ ] **Day 4-6**: 并行运行实验
  - 代码: `results = Parallel(n_jobs=-1)(run(p) for p in param_values)`
  - 预计时间: 48小时（建议服务器运行）
- [ ] **Day 7**: 计算Sobol指数
  - 代码: `Si = sobol.analyze(problem, Y)`
- [ ] **Week 8末**: 输出 `sensitivity/sobol_report.pdf`

**检查点**: 识别5-8个高敏感性参数（Si > 0.1）

---

**第1个月结束检查** ✅
- [ ] `validation/micro_validation_report.pdf` 已完成
- [ ] `validation/macro_validation_report.pdf` 已完成
- [ ] `validation/backtest_comparison.png` 已完成
- [ ] `sensitivity/sobol_report.pdf` 已完成
- [ ] 宏观验证通过率 ≥ 67% (10/15个矩)

---

## 第2个月：政策实验

### Week 9-10: 反事实实验（核心！）⭐⭐⭐⭐⭐
- [ ] **Day 1**: 设计3个场景
  - Scenario A: 现实（有补贴）
  - Scenario B: 反事实（无补贴）
  - Scenario C: 反事实（无服务市场）
- [ ] **Day 2-5**: 运行300次模拟（每场景100次）
  ```python
  results_A = run_simulation(subsidy=0.15, n=100)
  results_B = run_simulation(subsidy=0.0, n=100)
  results_C = run_simulation(service=False, n=100)
  ```
- [ ] **Day 6-7**: 计算政策效应
  - ATE_subsidy = E[产量|A] - E[产量|B]
  - ATE_service = E[产量|A] - E[产量|C]
- [ ] **Day 8-9**: Bootstrap假设检验
  - H0: ATE = 0 vs H1: ATE ≠ 0
  - p值 < 0.05 → 拒绝H0 → 政策有显著效应
- [ ] **Week 10末**: 输出 `policy_experiments/counterfactual_analysis.pdf`

**检查点**: 
- ATE_subsidy 显著 (p < 0.05)
- 补贴对粮食产量的效应 > 10%

---

### Week 11-12: 政策梯度实验
- [ ] **Day 1**: 定义政策网格
  ```python
  subsidy_grid = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 7个水平
  price_grid = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # 6个水平
  # 总计 7×6 = 42种组合
  ```
- [ ] **Day 2-7**: 运行4200次模拟（42×100）
  - 并行运行（服务器24小时）
- [ ] **Day 8-9**: 计算多目标得分
  ```python
  Score = 0.4×粮食安全 + 0.35×农民福利 + 0.25×财政平衡
  ```
- [ ] **Day 10**: 找到帕累托最优解集
- [ ] **Week 12末**: 输出 `policy_experiments/pareto_frontier.png`

**检查点**: 
- 识别出5-10个帕累托最优政策组合
- 绘制精美的帕累托前沿图

---

**第2个月结束检查** ✅
- [ ] `policy_experiments/counterfactual_analysis.pdf` 已完成
- [ ] `policy_experiments/pareto_frontier.png` 已完成
- [ ] 至少2个政策效应显著（p < 0.05）
- [ ] 可以回答"最优补贴率是多少？"

---

## 第3个月：论文撰写与投稿

### Week 13-14: 撰写验证章节
- [ ] **Day 1-2**: 撰写3.1 Data Sources
  - 描述CLES 2020-2022数据
  - 展示关键变量的描述性统计
- [ ] **Day 3-4**: 撰写3.2 Parameter Calibration
  - 表格：15个参数的来源和数值
  - 方法：矩匹配法的数学表达
- [ ] **Day 5-6**: 撰写3.3-3.5 Validation
  - 3.3 Micro-validation
  - 3.4 Macro-validation（核心！）
  - 3.5 Historical validation
- [ ] **Day 7-8**: 撰写3.6 Sensitivity Analysis
  - Sobol指数图表
  - 高敏感性参数讨论
- [ ] **Week 14末**: Validation章节初稿（15页）

**检查点**: 验证章节包含至少15张图表

---

### Week 15: 撰写政策实验章节
- [ ] **Day 1-2**: 撰写4.1 Baseline Scenario
  - 展示基准情景的动态演化
  - 关键指标的时间序列图
- [ ] **Day 3-4**: 撰写4.2 Counterfactual Analysis
  - 有补贴 vs 无补贴对比图
  - ATE估计值和置信区间
- [ ] **Day 5**: 撰写4.3 Policy Gradient Analysis
  - 帕累托前沿图
  - 最优政策组合推荐
- [ ] **Day 6-7**: 撰写4.4 Results Discussion
  - 理论解释
  - 政策启示
- [ ] **Week 15末**: Policy Experiments章节初稿（12页）

**检查点**: 政策章节包含至少10张图表

---

### Week 16: 论文润色与投稿
- [ ] **Day 1-2**: 整合所有章节
  - Introduction (已有框架)
  - Model Description (使用ODD文档)
  - Validation (Week 13-14完成)
  - Policy Experiments (Week 15完成)
  - Conclusion (新写5页)
  - 预计总页数: 50-60页
- [ ] **Day 3**: 英文润色
  - 使用Grammarly Premium
  - 或专业润色服务（如Editage）
- [ ] **Day 4**: 补充图表
  - 至少15张高质量图（300 DPI）
  - 统一配色方案
  - 所有图表有清晰标题和注释
- [ ] **Day 5**: 撰写Cover Letter
  - 说明研究贡献
  - 强调原创性（熟人网络、服务规模经济）
  - 推荐3-5位审稿人
- [ ] **Day 6**: 选择目标期刊并格式调整
  - **推荐1**: *Land Use Policy* (IF=6.0, SSCI Q1)
    - 周期: 6-8个月
    - 录用率: ~25%
  - **推荐2**: *JASSS* (SSCI Q1)
    - 周期: 4-6个月
    - 录用率: ~30%
  - **备选**: *Agricultural Systems* (IF=6.1)
- [ ] **Day 7**: **提交！** 🎉

**检查点**: 
- [ ] 论文50-60页（双栏格式）
- [ ] 至少40个引用文献
- [ ] 至少25张图表
- [ ] 通过Grammarly检查（零错误）
- [ ] Cover Letter完整

---

**第3个月结束检查** ✅
- [ ] 完整论文已提交到期刊
- [ ] 工作论文已上传SSRN/arXiv
- [ ] 代码已开源到GitHub
- [ ] 补充材料已准备（包括所有验证脚本）

---

## 每日工作流程建议

### 上午（4小时）
- 08:00-09:00: 阅读文献、更新理论框架
- 09:00-12:00: 编写代码或运行实验
- 12:00-12:30: 记录进展、更新清单

### 下午（4小时）
- 14:00-16:00: 数据分析、制作图表
- 16:00-18:00: 撰写报告或论文
- 18:00-18:30: 检查邮件、备份数据

### 每周总结（周五下午）
- 回顾本周完成的任务
- 对比计划进度
- 调整下周计划
- 备份所有代码和数据

---

## 紧急求助清单

如果遇到以下问题，请立即联系我：

### 技术问题
- [ ] 蒙特卡洛模拟运行超过24小时
- [ ] 宏观验证通过率 < 50%
- [ ] Sobol采样内存溢出
- [ ] 并行计算报错

### 方法论问题
- [ ] 不确定某个参数该如何校准
- [ ] Bootstrap检验p值一直很低
- [ ] 反事实实验结果不显著
- [ ] 不知道如何解释某个现象

### 写作问题
- [ ] 不知道如何组织验证章节
- [ ] 图表不够美观
- [ ] Cover Letter不知道如何写
- [ ] 期刊选择犹豫不决

**联系方式**: 在Cursor中@我即可

---

## 里程碑奖励 🎁

为了保持动力，设置小奖励：

- [ ] **Week 2完成**: 微观验证通过 → 奖励自己一顿大餐
- [ ] **Week 4完成**: 宏观验证通过 → 买一本心仪的书
- [ ] **Week 8完成**: 第1个月全部完成 → 放假1天休息
- [ ] **Week 12完成**: 政策实验完成 → 去看一场电影
- [ ] **Week 16完成**: 论文提交！ → 庆祝晚宴 🎉

---

## 最后的提醒

### ⚠️ 常见陷阱（避免！）
1. **过度优化代码**：代码够用就行，专注于验证
2. **追求完美**：80分可以投稿，不需要100分
3. **拖延症**：每天4小时专注工作，胜过8小时分心
4. **孤军奋战**：遇到问题立即求助，不要浪费时间

### ✅ 成功秘诀
1. **每日进展可视化**：在墙上贴进度表，每完成一项打勾
2. **番茄工作法**：25分钟专注+5分钟休息
3. **寻找同伴**：找一位同学互相监督进度
4. **庆祝小胜利**：每完成一个里程碑就奖励自己

---

## 紧急情况应对

### 如果进度落后
- Week 4时仍未完成宏观验证 → **立即简化目标矩数量**（15个→10个）
- Week 8时敏感性分析未完成 → **跳过，进入政策实验**（敏感性可作为补充材料）
- Week 12时政策实验未完成 → **仅完成反事实实验**（政策梯度可作为未来研究）

### 最低可投稿标准（必须完成）
- [x] 宏观验证通过（至少8/15个矩）
- [x] 反事实实验（有补贴 vs 无补贴）
- [x] Bootstrap检验显著（p < 0.05）
- [x] 论文40页以上
- [x] 至少15张图表

**只要完成以上5项，就可以投稿！**

---

## 你不是一个人在战斗！

记住：
- 您的项目**已经有80分**
- 只需要再完成**20分的验证工作**
- 这20分是**技术性任务，有清晰路径**
- **3个月后，您将拥有一篇国际发表级别的论文**

**现在就开始吧！** 🚀

**从最简单的开始**：
```bash
# 今天就创建第一个验证脚本
mkdir validation
cd validation
touch cles_moment_matching.py
# 然后打开文件，开始编写...
```

**您可以做到！加油！** 💪

---

*行动清单创建日期: 2024年12月6日*  
*更新日期: __________*  
*完成日期: __________*

**签名**: ______________ （完成后请签名并庆祝！）


