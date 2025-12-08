# 政策参数实验设计指南
# Policy Parameters Guide for Experimental Design

> **更新日期**: 2025-12-06  
> **项目**: 中国粮食补贴政策多智能体模型  
> **用途**: 为政策模拟实验提供可调整参数的完整清单

---

## 📋 目录

1. [补贴政策参数](#1-补贴政策参数)
2. [粮食储备局政策参数](#2-粮食储备局政策参数)
3. [财政系统参数](#3-财政系统参数)
4. [政府多目标优化参数](#4-政府多目标优化参数)
5. [AI冲击政策参数](#5-ai冲击政策参数)
6. [劳动力市场政策参数](#6-劳动力市场政策参数)
7. [农业社会化服务政策参数](#7-农业社会化服务政策参数)
8. [土地流转政策参数](#8-土地流转政策参数)
9. [区域差异化政策参数](#9-区域差异化政策参数)
10. [技术进步与规模化政策参数](#10-技术进步与规模化政策参数)
11. [气候与环境政策参数](#11-气候与环境政策参数)
12. [村集体政策参数](#12-村集体政策参数)
13. [政策实验设计建议](#13-政策实验设计建议)

---

## 1. 补贴政策参数

### 1.1 核心补贴参数
**配置位置**: `config/policy_params.json`

| 参数名 | 默认值 | 取值范围 | 实验方向 |
|--------|--------|----------|----------|
| `base_subsidy_per_mu` | 350元/亩 | 50-500元 | **补贴强度实验**：测试不同补贴水平对粮食产量的边际效应 |
| `grain_subsidies_rate` | 1.0 | 0.05-1.0 | **补贴率实验**：评估补贴率对种粮积极性的影响 |
| `grain_price_floor` | 1.2元/斤 | 0.8-3.0元 | **价格支持实验**：比较价格支持vs面积补贴的效率 |

### 1.2 补贴类型参数

| 参数名 | 可选值 | 实验设计 |
|--------|--------|----------|
| `subsidy_type` | `area_based`（面积补贴）<br>`tiered_area`（分段补贴）<br>`output_based`（产量补贴） | **补贴设计实验**：对比不同补贴机制的粮食安全效果 |
| `subsidy_target` | `owner`（所有者）<br>`operator`（经营者） | **目标对象实验**：评估补贴归属对土地流转和规模化的影响 |
| `delivery_mode` | `cash`（现金）<br>`mixed`（现金+实物） | **发放方式实验**：测试实物补贴vs现金补贴的福利效率 |

### 1.3 分段补贴参数（仅`tiered_area`类型）

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `tier_breakpoints` | [50, 150]亩 | **规模门槛实验**：调整门槛测试对不同规模农户的激励效应 |
| `tier_rates` | [0.12, 0.18, 0.22] | **累进/累退实验**：设计累进（鼓励小农）或累退（鼓励规模化）补贴结构 |

### 1.4 实物补贴参数（仅`mixed`模式）

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `cash_share` | 0.5-0.7 | **现金比例实验**：测试现金与实物的最优配比 |
| `fertilizer_offset_per_mu` | 35-45元/亩 | **农资补贴实验**：评估肥料补贴的成本节约效应 |
| `training_service_per_mu` | 12-25元/亩 | **人力资本投资实验**：测试培训补贴对效率的长期影响 |

### 1.5 补贴发放时滞参数

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `payment_lag` | 0-1步 | 0-2步 | **时滞效应实验**：模拟补贴延迟发放对农户决策的影响 |

### 1.6 主体差异化系数（现实校准）

**配置位置**: `config/policy_params.json` → `subsidy_standards` → `subject_adjustment`

| 主体类型 | 小麦系数 | 玉米系数 | 实验设计 |
|----------|----------|----------|----------|
| 农户 (farmer) | 1.0 | 1.0 | **公平性基准** |
| 企业 (enterprise) | 1.15 | 1.15 | **规模化激励实验**：调整系数测试对企业参与的影响 |
| 合作社 (cooperative) | 1.1 | 1.1 | **组织化激励实验** |

---

## 2. 粮食储备局政策参数

### 2.1 价格干预参数
**配置位置**: `config/master_config.yaml` → `agents.grain_reserve_bureau`

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `floor_price` | 2.58元/kg | 2.0-3.0元 | **托市收购实验**：测试不同托市价对粮食供给的刺激作用 |
| `ceiling_price` | 2.97元/kg | 2.5-3.5元 | **抛储稳价实验**：评估抛储价格对市场稳定的作用 |
| `annual_budget` | 100万元 | 50-500万 | **收购能力实验**：测试预算约束对价格干预效果的限制 |

### 2.2 库存管理参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `initial_storage` | 100吨 | **初始库存实验**：测试不同库存水平对价格稳定能力的影响 |
| `max_storage_capacity` | 500吨 | **库容约束实验**：模拟库容限制对收购政策的约束 |
| `min_safety_stock` | 20吨 | **安全库存实验**：评估最低库存要求对粮食安全的保障作用 |

### 2.3 财政可持续性参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `storage_cost` | 90元/吨/年 | **储存成本实验**：模拟高储存成本对收储政策的负担 |
| `fiscal_cliff_threshold` | 连续3年触发 | **财政悬崖实验**：测试不同触发条件对政策可持续性的影响 |
| `fiscal_cliff_reduction_rate` | 每年削减15% | **削减速度实验**：评估财政悬崖后不同削减速度的福利损失 |

---

## 3. 财政系统参数

### 3.1 预算约束参数
**配置位置**: `config/master_config.yaml` → `agents.fiscal_system`

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `annual_subsidy_budget` | 800万元 | 200-2000万 | **预算规模实验**：测试不同预算水平的政策空间 |
| `hard_cap` | true | true/false | **硬约束vs软约束实验**：评估预算硬约束对政策效果的影响 |
| `overspending_threshold` | 1.2（超支20%） | 1.1-1.5 | **超支容忍度实验**：测试不同容忍度对政策稳定性的影响 |

### 3.2 财政悬崖参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `consecutive_overspending_years` | 3年 | **触发时长实验**：调整触发年限测试政策韧性 |
| `reduction_rate` | 每年削减15% | **削减强度实验**：评估不同削减速度的社会冲击 |
| `minimum_budget_ratio` | 最低保留50% | **最低保障实验**：测试托底预算的福利兜底作用 |

### 3.3 补贴拖欠机制参数（新增）

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `arrears_mechanism.enable` | true | **拖欠政策实验**：对比有无拖欠机制的政策灵活性 |
| `max_arrears_ratio` | 30% | **拖欠上限实验**：测试不同拖欠比例对农户信心的影响 |
| `clearance_rate` | 年度清偿50% | **清偿速度实验**：评估清偿速度对补贴预期的影响 |

### 3.4 分项预算配置（预留）

**配置位置**: `config/master_config.yaml` → `agents.fiscal_system.budget_allocation`

| 补贴类型 | 费率/阈值 | 实验设计 |
|----------|----------|----------|
| `area_subsidy.rate` | 83元/亩 | **地力保护补贴实验**：调整基础补贴水平 |
| `scale_subsidy.threshold_area` | 50亩 | **规模门槛实验**：测试不同门槛对规模化经营的激励 |
| `scale_subsidy.rate` | 100元/亩 | **规模加码实验**：评估规模补贴的边际效应 |
| `grain_subsidy.rate` | 150元/亩 | **种粮补贴实验**：测试种粮补贴对作物选择的引导作用 |

---

## 4. 政府多目标优化参数

### 4.1 目标权重参数
**配置位置**: `config/master_config.yaml` → `policy.government_objectives.weights`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `grain_security` | 0.4 | **目标优先级实验**：调整权重测试不同政策导向<br>（例如：粮食安全优先 vs 农民福利优先） |
| `farmer_welfare` | 0.35 | |
| `fiscal_balance` | 0.25 | |

**实验建议**：设计3组权重组合：
- **粮食安全型**：[0.6, 0.25, 0.15]
- **福利优先型**：[0.25, 0.6, 0.15]
- **财政保守型**：[0.3, 0.3, 0.4]

### 4.2 目标阈值参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `targets.grain_output` | 2.5亿kg | **目标雄心度实验**：设置不同粮食产量目标测试政策强度 |
| `targets.avg_farmer_income` | 2万元 | **收入目标实验**：调整收入目标测试补贴需求 |
| `targets.subsidy_budget` | 1亿元 | **预算目标实验**：测试不同预算约束下的政策空间 |

### 4.3 自适应调整参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `adjustments.subsidy_step` | 0.01 | **调整速度实验**：测试不同调整速度对政策稳定性的影响 |
| `adjustments.price_floor_step` | 0.05 | |
| `adjustments.rent_incentive_step` | 0.01 | |

---

## 5. AI冲击政策参数

### 5.1 AI政策开关
**配置位置**: `config/master_config.yaml` → `ai`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `enabled` | true | **AI冲击vs无AI对照实验**：评估AI对农业劳动力市场的影响 |

### 5.2 企业AI应用参数

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `enterprise.tfp_boost` | 0.1（10%提升） | 0-0.3 | **AI生产率实验**：测试不同AI技术水平对企业效率的影响 |
| `enterprise.labor_substitution` | 0.2（替代20%） | 0-0.5 | **劳动替代实验**：评估AI对农业就业的挤出效应 |
| `enterprise.input_saving` | 0.15（节约15%） | 0-0.3 | **投入节约实验**：测试AI对成本控制的作用 |

### 5.3 农户AI影响参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `farmer.non_farm_job_shock` | 0.3（非农岗位减少30%） | **就业冲击实验**：模拟AI对农民非农就业的挤出效应 |
| `farmer.wage_penalty` | 0.2（工资下降20%） | **工资挤压实验**：评估AI对非农工资的负面影响 |
| `farmer.adoption_lag` | 0.5（滞后系数） | **技术扩散实验**：测试农户AI采纳速度的差异化影响 |

### 5.4 劳动力市场AI参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `labor_market.non_farm_demand_penalty` | 0.25（需求减少25%） | **市场冲击实验**：测试AI对非农劳动力需求的全面冲击 |
| `labor_market.wage_floor` | 0.85（最低工资率85%） | **最低工资保护实验**：评估工资下限政策对AI冲击的缓冲作用 |

**政策实验建议**：
- **AI强冲击情景**：tfp_boost=0.2, labor_substitution=0.4, non_farm_job_shock=0.5
- **AI弱冲击情景**：tfp_boost=0.05, labor_substitution=0.1, non_farm_job_shock=0.15
- **AI政策干预情景**：启用wage_floor=0.9 + training_subsidy补贴（需自行添加）

---

## 6. 劳动力市场政策参数

### 6.1 工资政策参数
**配置位置**: `config/master_config.yaml` → `labor_market`

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `base_agricultural_wage` | 100元/天 | 80-150元 | **农业工资实验**：测试提高农业工资对劳动力回流的吸引力 |
| `base_non_farm_wage` | 150元/天 | 120-200元 | **非农工资实验**：评估工资差距对劳动力流失的影响 |
| `wage_floor` | 80元/天 | 60-100元 | **最低工资实验**：测试最低工资对劳动力市场的保护作用 |
| `wage_ceiling` | 300元/天 | 250-400元 | **工资上限实验**：评估工资上限对市场失衡的调节 |

### 6.2 季节性工资调节参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `seasonal_factors.spring` | 1.3 | **季节性工资实验**：调整季节因子测试对农忙期劳动力供给的影响 |
| `seasonal_factors.autumn` | 1.5 | |
| `seasonal_factors.winter` | 0.6 | |

### 6.3 机械替代政策参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `machinery_substitution.grain` | 0.6 | **机械化政策实验**：提高机械替代率模拟"机器换人"政策 |
| `machinery_substitution.cash_crop` | 0.3 | |
| `mechanization_efficiency_bonus.grain` | 0.20（+20%产量） | **机械化效率实验**：测试机械化对粮食产量的提升作用 |
| `machinery_cost_per_day` | 80元/天 | **机械成本实验**：调整机械成本测试对机械化意愿的影响 |

---

## 7. 农业社会化服务政策参数

### 7.1 服务市场激活开关
**配置位置**: `config/master_config.yaml` → `production.agricultural_service`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `enabled` | true | **服务vs土地流转对照实验**：关闭服务市场测试土地流转的替代效应 |
| `num_providers` | 10个 | **供给规模实验**：调整服务商数量测试市场覆盖度 |

### 7.2 服务类型与定价参数

| 服务类型 | 费率 | 劳动节约 | 产量提升 | 实验设计 |
|----------|------|----------|----------|----------|
| `full_trusteeship` | 500元/亩 | 85% | 10% | **全托管补贴实验**：政府补贴托管费用降低小农成本 |
| `partial_trusteeship` | 280元/亩 | 50% | 5% | **关键环节服务实验**：重点补贴播种/收割环节 |
| `single_service` | 100元/亩 | 20% | 2% | **单项服务补贴实验**：补贴机收服务降低收获损失 |

**政策实验建议**：
- **服务补贴政策**：政府承担`full_trusteeship`费用的50%（250元/亩），测试对小农参与的激励
- **服务券政策**：发放服务券抵扣服务费用，测试市场化运作效率

### 7.3 服务商规模经济参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `provider_economics.scale_economy_factor` | 0.15 | **规模经济实验**：调整系数测试服务商扩张动力 |
| `provider_economics.service_radius` | 15格子 | **服务半径实验**：测试地理约束对服务覆盖的限制 |
| `provider_economics.equipment_investment` | 5万元 | **准入门槛实验**：降低门槛鼓励更多服务商进入 |

### 7.4 小农户购买服务决策参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `farmer_decision.labor_scarcity_threshold` | 0.5 | **劳动力约束实验**：降低阈值使更多农户购买服务 |
| `farmer_decision.age_threshold` | 55岁 | **老龄化政策实验**：降低阈值使中年农户也购买服务 |
| `farmer_decision.cost_benefit_threshold` | 1.2 | **成本效益实验**：提高阈值使农户对服务更敏感 |

### 7.5 与土地流转的替代关系参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `land_transfer_substitution.substitution_effect` | 0.3 | **替代效应实验**：调整系数测试服务对土地流转的抑制作用 |

**理论贡献**：验证陈义媛/高原"隐性农业革命"假说——服务规模化vs土地规模化

---

## 8. 土地流转政策参数

### 8.1 流转门槛参数
**配置位置**: `config/master_config.yaml` → `agents.farmer.land_transfer`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `threshold_in` | 0.5 | **租入门槛实验**：降低门槛促进规模化经营 |
| `threshold_out` | 0.15 | **流出门槛实验**：降低门槛加速土地流转 |
| `elderly_age_threshold` | 60岁 | **老龄化流转实验**：调整阈值测试老年农户流转意愿 |
| `min_retained_area_per_capita` | 0.2亩/人 | **保留地政策实验**：提高保留面积保障农户土地权益 |

### 8.2 流转定价参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `base_rent_price` | 400元/亩 | **租金政策实验**：政府指导价vs市场化定价对比 |
| `max_local_matches` | 5对 | **熟人网络实验**：扩大熟人网络交易机会降低交易成本 |

### 8.3 生计策略差异化参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `non_farm_out_multiplier` | 0.3 | **非农户流出激励**：调整系数鼓励非农户退出农业 |
| `pure_farmer_in_multiplier` | 0.6 | **纯农户扩张激励**：降低系数鼓励纯农户租入土地 |
| `base_target_operated_area` | 100亩 | **目标规模实验**：提高目标规模促进规模化经营 |

---

## 9. 区域差异化政策参数

### 9.1 区域差异化开关
**配置位置**: `config/master_config.yaml` → `policy.government`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `enable_regional_heterogeneity` | false | **区域政策实验**：启用后根据区域特征自动匹配政策场景 |

### 9.2 区域分类阈值参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `regional_thresholds.income_low` | 22000元 | **贫困线实验**：调整阈值测试对扶持政策触发的影响 |
| `regional_thresholds.agri_ratio_high` | 0.75 | **农业依赖度实验**：调整阈值识别重点产粮区 |
| `regional_thresholds.network_density_low` | 0.25 | **社会资本实验**：低社会资本地区给予更多支持 |

### 9.3 区域场景映射参数

| 区域类型 | 匹配场景 | 实验设计 |
|----------|----------|----------|
| `vulnerable`（脆弱地区） | `high_support`（高支持） | **精准扶持实验**：对比统一政策vs差异化政策的效率 |
| `transforming`（转型地区） | `targeted_ai_shock`（AI技术券） | **转型政策实验**：测试数字农业政策对转型地区的适配性 |

---

## 10. 技术进步与规模化政策参数

### 10.1 技术进步参数
**配置位置**: `config/master_config.yaml` → `production.technology_progress`

| 参数名 | 默认值 | 取值范围 | 实验设计 |
|--------|--------|----------|----------|
| `annual_yield_growth_rate` | 0.015（1.5%/年） | 0-0.03 | **技术进步实验**：模拟不同R&D投入对产量增长的影响 |

### 10.2 规模效率参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `scale_efficiency_bonus.threshold_area` | 30亩 | **规模门槛实验**：调整门槛测试对规模化的定义 |
| `scale_efficiency_bonus.medium_scale` | 1.10（+10%） | **规模效率实验**：提高系数鼓励中等规模经营 |
| `scale_efficiency_bonus.large_scale` | 1.20（+20%） | |

---

## 11. 气候与环境政策参数

### 11.1 气候冲击参数
**配置位置**: `config/master_config.yaml` → `production.climate`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `mean` | 0.0 | **气候趋势实验**：负值模拟气候恶化趋势 |
| `volatility` | 0.15 | **气候风险实验**：提高波动率测试极端天气影响 |

### 11.2 土地撂荒政策参数
**配置位置**: `config/master_config.yaml` → `agents.farmer.land_abandonment`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `base_rate` | 0.02 | **撂荒防控实验**：提高补贴降低撂荒率 |
| `subsidy_suppression_factor` | 0.5 | **补贴抑制实验**：测试补贴对撂荒的抑制效果 |

---

## 12. 村集体政策参数

### 12.1 村集体角色参数
**配置位置**: `config/master_config.yaml` → `agents.village_collective`

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `enabled` | true | **村集体vs市场化对照实验** |
| `service_fee_rate` | 3% | **服务费实验**：调整费率测试对流转成本的影响 |
| `participation_rate` | 30% | **参与率实验**：提高村集体参与度测试流转效率 |

### 12.2 强制种粮政策参数

| 参数名 | 默认值 | 实验设计 |
|--------|--------|----------|
| `grain_requirement.ratio` | 0.40（40%） | **非粮化防控实验**：提高比例测试对粮食安全的保障作用 |
| `grain_requirement.enforcement_strength` | 0.8 | **执行力度实验**：测试不同执行强度的合规率 |

---

## 13. 政策实验设计建议

### 13.1 单因素实验设计

**目的**：测试单个政策参数的边际效应

**示例**：补贴强度实验
```yaml
experiment:
  name: Subsidy_Intensity_Sweep
  parameters:
    - base_subsidy_per_mu: [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
  fixed_parameters:
    - grain_price_floor: 1.2
    - subsidy_target: operator
  replicates: 30
  steps: 50
```

### 13.2 多因素交互实验设计

**目的**：测试政策组合效应

**示例**：补贴类型 × 补贴水平 × AI冲击
```python
import itertools

subsidy_types = ['area_based', 'tiered_area', 'output_based']
subsidy_levels = [100, 200, 300, 400]
ai_enabled = [True, False]

experiments = list(itertools.product(subsidy_types, subsidy_levels, ai_enabled))
# 总计 3×4×2 = 24 组实验
```

### 13.3 情景对比实验设计

**目的**：评估不同政策组合的宏观效果

| 情景名 | 补贴设计 | 价格干预 | AI冲击 | 服务市场 |
|--------|----------|----------|--------|----------|
| **基准情景** | 350元/亩面积补贴 | 托市价2.58元 | 启用 | 启用 |
| **价格支持情景** | 150元/亩 | 托市价3.0元 | 启用 | 启用 |
| **市场化情景** | 100元/亩 | 无托市 | 启用 | 启用 |
| **强AI冲击情景** | 400元/亩 | 托市价2.8元 | 强冲击 | 启用 |
| **服务替代情景** | 200元/亩 | 托市价2.58元 | 启用 | 关闭 |

### 13.4 敏感性分析设计

**目的**：识别关键政策参数

**方法**：Sobol全局敏感性分析

**示例**：
```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 10,
    'names': [
        'base_subsidy_per_mu',
        'grain_price_floor',
        'annual_subsidy_budget',
        'ai_tfp_boost',
        'floor_price',
        'ceiling_price',
        'full_trusteeship_fee',
        'wage_floor',
        'grain_requirement_ratio',
        'annual_yield_growth_rate'
    ],
    'bounds': [
        [50, 500],      # base_subsidy_per_mu
        [0.8, 3.0],     # grain_price_floor
        [200, 2000],    # annual_subsidy_budget (万元)
        [0, 0.3],       # ai_tfp_boost
        [2.0, 3.0],     # floor_price
        [2.5, 3.5],     # ceiling_price
        [300, 600],     # full_trusteeship_fee
        [60, 100],      # wage_floor
        [0.2, 0.6],     # grain_requirement_ratio
        [0, 0.03]       # annual_yield_growth_rate
    ]
}

# 生成样本（N=1024, 共10240组实验）
param_values = saltelli.sample(problem, 1024)

# 运行模型并分析敏感性
# ... (模型运行代码)
# Si = sobol.analyze(problem, Y, print_to_console=True)
```

### 13.5 政策优化实验设计

**目的**：寻找多目标最优政策组合

**方法**：NSGA-II多目标遗传算法

**目标函数**：
- 最大化：粮食产量
- 最大化：农民平均收入
- 最小化：财政支出
- 最小化：土地基尼系数

**约束条件**：
- 补贴预算 ≤ 800万元
- 托市价格 ≥ 1.2元/斤
- 粮食产量 ≥ 目标产量的95%

---

## 14. 实验输出指标建议

### 14.1 政策效果指标

| 指标类别 | 指标名称 | 计算方法 |
|----------|----------|----------|
| **粮食安全** | 粮食总产量 | sum(grain_output) |
| | 粮食自给率 | grain_supply / grain_demand |
| | 种粮面积占比 | grain_area / total_area |
| **农民福利** | 农户平均收入 | mean(farmer_income) |
| | 收入基尼系数 | gini(farmer_income) |
| | 贫困率 | count(income < poverty_line) / N |
| **财政可持续** | 补贴总支出 | sum(subsidies) |
| | 预算执行率 | subsidies / budget |
| | 财政悬崖触发次数 | count(fiscal_cliff_triggered) |
| **结构转型** | 土地流转率 | transferred_area / total_area |
| | 土地基尼系数 | gini(operated_area) |
| | 纯农户占比 | count(Pure_Farmer) / N |
| | 规模经营占比 | count(area > 30亩) / N |

### 14.2 机制验证指标

| 理论机制 | 验证指标 |
|----------|----------|
| **前景理论** | 损失域vs收益域决策差异 |
| **社会学习** | 社会网络密度与作物选择相关性 |
| **规模效率** | 经营规模与单产的相关性 |
| **补贴资本化** | 补贴率与土地租金的回归系数 |

---

## 15. 快速参考表

### 15.1 高影响力参数（优先实验）

| 参数名 | 影响对象 | 预期效应强度 |
|--------|----------|-------------|
| `base_subsidy_per_mu` | 粮食产量、农民收入 | ★★★★★ |
| `grain_price_floor` | 粮食产量、财政支出 | ★★★★★ |
| `annual_subsidy_budget` | 补贴覆盖率、财政压力 | ★★★★★ |
| `ai_enabled` | 劳动力市场、收入分配 | ★★★★☆ |
| `subsidy_target` (owner/operator) | 土地流转、规模化 | ★★★★☆ |
| `grain_requirement_ratio` | 粮食产量、企业利润 | ★★★★☆ |
| `full_trusteeship_fee` | 服务市场、土地流转 | ★★★☆☆ |

### 15.2 建议实验优先级

**第一阶段**（核心政策）：
1. 补贴强度实验（`base_subsidy_per_mu`）
2. 补贴目标实验（`subsidy_target`: owner vs operator）
3. 价格支持实验（`grain_price_floor`）

**第二阶段**（政策组合）：
4. 补贴类型实验（`subsidy_type`: area vs tiered vs output）
5. 财政约束实验（`annual_subsidy_budget`）
6. AI冲击实验（`ai_enabled` + `ai_tfp_boost`）

**第三阶段**（结构改革）：
7. 农业服务市场实验（`agricultural_service.enabled`）
8. 村集体强制种粮实验（`grain_requirement_ratio`）
9. 区域差异化政策实验（`enable_regional_heterogeneity`）

**第四阶段**（敏感性分析）：
10. 全参数Sobol敏感性分析
11. 多目标政策优化（NSGA-II）

---

## 16. 技术实现

### 16.1 参数扫描脚本示例

**文件位置**: `scripts/run_policy_sweep.py`

```python
import itertools
from scripts.run_integrated_workflow import run_single_experiment

# 定义参数网格
subsidy_levels = [100, 200, 300, 400, 500]
subsidy_targets = ['owner', 'operator']
ai_enabled_options = [True, False]

# 生成所有组合
experiments = list(itertools.product(
    subsidy_levels, 
    subsidy_targets, 
    ai_enabled_options
))

# 运行实验
for subsidy, target, ai_on in experiments:
    overrides = {
        'policy.scenarios.baseline_cash.base_subsidy_per_mu': subsidy,
        'policy.scenarios.baseline_cash.subsidy_target': target,
        'ai.enabled': ai_on
    }
    
    run_single_experiment(
        config_overrides=overrides,
        steps=50,
        replicates=10,
        output_dir=f'results/sweep_subsidy{subsidy}_{target}_ai{ai_on}'
    )
```

### 16.2 调用方式

```bash
# 单参数扫描
python scripts/run_policy_sweep.py --parameter base_subsidy_per_mu --range 50:500:50 --replicates 30

# 多参数交互实验
python scripts/run_policy_sweep.py --parameters base_subsidy_per_mu,grain_price_floor --values 100,200,300:1.2,1.5,1.8

# 情景对比实验
python scripts/run_integrated_workflow.py --mode policy_comparison --scenarios baseline,high_support,targeted_ai_shock
```

---

## 17. 参考文献

1. **补贴政策**: 国家发改委发改价格〔2023〕1314号、财政部财农〔2024〕83号
2. **粮食收储**: 国家粮食和物资储备局《粮食流通管理条例》（2021年修订）
3. **AI冲击**: 农业农村部《数字乡村发展行动计划（2024-2027年）》
4. **农业服务**: 陈义媛、高原（2019）《服务规模经营：农业转型的中国道路》
5. **土地流转**: Deininger & Jin (2005), Journal of Development Economics
6. **行为经济学**: Kahneman & Tversky (1979), Prospect Theory

---

**版本历史**：
- v1.0 (2025-12-06): 初始版本，整合所有政策参数
- 持续更新中...

**联系方式**：
- 项目负责人：[您的姓名]
- 邮箱：[您的邮箱]
- GitHub: [项目链接]


