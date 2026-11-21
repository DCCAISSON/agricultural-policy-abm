# -*- coding: utf-8 -*-
"""
@theory
GovernmentAgent - 基于多层次博弈设计规范的政府主体
核心功能：政策发布和补贴发放
新增：多层次博弈政策设定、价格地板机制
"""

import json
from pathlib import Path
import numpy as np

from mesa import Agent
from enum import Enum
from canonical.constants.policy_constants import (
    BASELINE_GRAIN_PRICE_FLOOR,
    DEFAULT_ANNUAL_SUBSIDY_BUDGET,
    LEGACY_GRAIN_SUBSIDIES_RATE,
    LEGACY_SUBSIDY_RATE,
)
from core.config_utils import (
    DEFAULT_BASE_NON_FARM_WAGE,
    resolve_base_non_farm_wage,
    sync_base_non_farm_wage,
)
from core.income_utils import (
    get_farmer_gross_income,
    get_farmer_livelihood_agriculture_ratio,
    get_farmer_net_income,
    get_farmer_operated_land_area,
    is_large_operator,
    is_pure_farmer,
    is_professional_large_operator,
)
from core.reproducibility import np_random_value, np_randint
from utils.policy_runtime import resolve_fiscal_system_config
from core.p3_project_payments import PaymentBatch, PaymentRow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _float_config_value(config, key, default):
    """Read numeric config values while preserving explicit zero."""
    if key not in config or config.get(key) is None:
        return float(default)
    try:
        return float(config.get(key))
    except (TypeError, ValueError):
        return float(default)

class SubsidyType(Enum):
    """补贴类型枚举"""
    AREA_BASED = "area_based"      # 按面积补贴
    OUTPUT_BASED = "output_based"  # 按产量补贴

class SubsidyTarget(Enum):
    """补贴目标枚举"""
    OWNER = "owner"        # 补贴给所有者
    OPERATOR = "operator"  # 补贴给耕作者

class GovernmentAgent(Agent):
    p3_batch: PaymentBatch
    p3_project_payments: list[PaymentRow]
    p3_implementation_cost_yuan: float

    def __init__(self, unique_id, model):
        # Mesa 3.x compatibility: Agent.__init__ may pass args to object incorrectly
        # We manually set required attributes first, then try super() but ignore failures
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # Required by Mesa grid system
        try:
            super().__init__(unique_id, model)
        except TypeError:
            # Mesa 3.x issue: don't call super().__init__() at all
            # Required attributes (unique_id, model, pos) already set above
            pass
        
        # 政策参数（新增）
        self.grain_price_floor = BASELINE_GRAIN_PRICE_FLOOR  # 粮食最低收购价
        self.grain_subsidies_rate = LEGACY_GRAIN_SUBSIDIES_RATE  # 粮食补贴率
        self.market_price_cash_crop = 2.0               # 经济作物市场价格
        self.subsidy_rate = LEGACY_SUBSIDY_RATE         # 一般补贴率
        
        # 补贴政策
        self.subsidy_type = SubsidyType.AREA_BASED      # 补贴类型
        self.subsidy_target = SubsidyTarget.OWNER       # 补贴目标
        
        # 补贴发放记录
        self.total_subsidy_paid = 0                     # 总补贴支出
        self.subsidy_records = []                       # 补贴记录
        self.subsidy_announcements = []
        self.subsidy_claims_submitted = []
        self.verified_subsidy_queue = []
        self.current_subsidy_runtime = {}
        self._subsidy_claim_counter = 0
        
        # ========== @new 2025-11-30: 财政约束系统 ==========
        # 核心理念：补贴预算是外生给定的，来自中央财政转移支付
        # 模型内部只能在预算约束下分配，不能自行创造
        fiscal_cfg = resolve_fiscal_system_config(self.model.config, create=False)
        
        # 年度补贴预算（元）- 外生给定
        self.annual_subsidy_budget = _float_config_value(
            fiscal_cfg,
            'annual_subsidy_budget',
            DEFAULT_ANNUAL_SUBSIDY_BUDGET,
        )
        self.base_annual_subsidy_budget = self.annual_subsidy_budget
        self.current_annual_subsidy_budget = self.annual_subsidy_budget
        self.remaining_budget = self.annual_subsidy_budget
        
        # 预算约束参数
        budget_constraints = fiscal_cfg.get('budget_constraints', {})
        self.hard_cap = budget_constraints.get('hard_cap', True)  # 硬性约束
        self.overspending_threshold = budget_constraints.get('overspending_threshold', 1.2)
        self.consecutive_overspending_years = budget_constraints.get('consecutive_overspending_years', 3)
        self.fiscal_reduction_rate = budget_constraints.get('reduction_rate', 0.15)
        self.minimum_budget_ratio = budget_constraints.get('minimum_budget_ratio', 0.5)
        
        # 财政状态跟踪
        self.consecutive_overspending_count = 0
        self.fiscal_cliff_active = False
        self.budget_utilization_history = []  # 历年预算使用率
        
        # ========== @new 2025-12-22: 宏观经济闭环机制 ==========
        # Nature级模型要求系统封闭性：补贴→税收→非农工资→农户收入 反馈循环
        sync_base_non_farm_wage(self.model.config, default=DEFAULT_BASE_NON_FARM_WAGE)
        macro_cfg = fiscal_cfg.get('macro_economy', {})
        self.enable_macro_feedback = macro_cfg.get('enable', True)
        self.base_non_farm_gdp = macro_cfg.get('base_non_farm_gdp', 50000000)  # 5000万非农GDP
        self.tax_rate = macro_cfg.get('tax_rate', 0.15)  # 税率15%
        self.subsidy_crowdout_elasticity = macro_cfg.get('crowdout_elasticity', 0.3)  # 挤出弹性
        canonical_daily_wage = resolve_base_non_farm_wage(
            self.model.config,
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        self.non_farm_wage_base = self._normalize_non_farm_daily_wage(
            macro_cfg.get('non_farm_wage_base', canonical_daily_wage),
            fallback=canonical_daily_wage,
        )
        
        # 宏观状态跟踪
        self.current_non_farm_gdp = self.base_non_farm_gdp
        self.total_tax_revenue = self.base_non_farm_gdp * self.tax_rate
        self.crowdout_effect = 0  # 补贴挤出效应
        self.adjusted_non_farm_wage = self.non_farm_wage_base  # 调整后非农工资
        
        # ========== @new 2025-12-04: 补贴拖欠机制 ==========
        # 核心理念：财政压力大时，政府不是立即削减补贴标准，而是产生拖欠
        # 拖欠会在后续年度逐步清偿，影响农户当期可支配收入
        arrears_cfg = fiscal_cfg.get('arrears_mechanism', {})
        self.enable_arrears = arrears_cfg.get('enable', True)  # 启用拖欠机制
        self.max_arrears_ratio = arrears_cfg.get('max_arrears_ratio', 0.3)  # 最大拖欠比例
        self.arrears_clearance_rate = arrears_cfg.get('clearance_rate', 0.5)  # 年度清偿比例
        self.arrears_interest_rate = arrears_cfg.get('interest_rate', 0.0)  # 拖欠利息（通常为0）
        
        # 拖欠状态跟踪（agent_id -> 拖欠金额）
        self.subsidy_arrears = {}  # 累计拖欠
        self.current_year_arrears = {}  # 本年新增拖欠
        self.arrears_paid_this_year = {}  # 本年清偿的历史拖欠
        self.total_arrears_outstanding = 0  # 总未清偿拖欠
        
        # 政策历史（新增）
        self.policy_history = []                        # 政策历史
        self.price_history = []                         # 价格历史
        self.regional_policy_assignments = {}
        self.latest_regional_summary = {}
        self.active_scenario_id = None
        results_cfg = self.model.config.get('results', {})
        dashboard_rel = results_cfg.get('policy_dashboard_path', 'results/policy_dashboard.json')
        dashboard_path = Path(dashboard_rel)
        if not dashboard_path.is_absolute():
            dashboard_path = PROJECT_ROOT / dashboard_rel
        self.policy_dashboard_path = dashboard_path
        policy_section = self.model.config.get('policy', {})
        self.gov_policy_config = policy_section.get('government', {})
        self.adaptive_adjustment_enabled = bool(
            self.gov_policy_config.get(
                'adaptive_adjustment_enabled',
                policy_section.get('adaptive_adjustment_enabled', True),
            )
        )

        # 多目标政府优化参数
        policy_cfg = self.model.config.get('policy', {})
        objectives = policy_cfg.get('government_objectives', {})
        weights = objectives.get('weights', {})
        targets = objectives.get('targets', {})
        adjustments = objectives.get('adjustments', {})
        self.objective_weights = {
            'grain_security': weights.get('grain_security', 0.4),
            'farmer_welfare': weights.get('farmer_welfare', 0.35),
            'fiscal_balance': weights.get('fiscal_balance', 0.25)
        }
        self.target_grain_output = float(targets.get('grain_output', 200000000) or 200000000)
        self.target_avg_income = float(
            targets.get('avg_farmer_net_income', targets.get('avg_farmer_income', 20000)) or 20000
        )
        self.subsidy_budget = float(targets.get('subsidy_budget', 100000000) or 100000000)
        self.subsidy_step = adjustments.get('subsidy_step', 0.01)
        self.price_floor_step = adjustments.get('price_floor_step', 0.05)
        self.rent_incentive_step = adjustments.get('rent_incentive_step', 0.01)

        # @new 2025-12-28: 强化学习策略优化 (P4)
        # ========== RL定位声明 (Nature审稿需求) ==========
        # @CRITICAL FRAMING: RL在本模型中的定位
        #
        # 【RL不是什么】
        # - RL不代表中国政府的实际决策能力
        # - RL不意味着政府真的会用机器学习制定政策
        #
        # 【RL是什么】
        # - RL是一个"理论最优路径探索工具"
        # - 作用: 帮助研究者发现"如果决策者能够完美优化，最优策略是什么？"
        # - 类比: 类似经济学中的"社会计划者"(Social Planner)基准
        #
        # 【实验设计逻辑】
        # 1. Baseline: 启发式规则 → 代表"现实中可能的决策"
        # 2. RL优化: Q-Learning → 代表"理论最优基准"
        # 3. 对比: Gap = RL - Baseline → 量化"政策改进空间"
        #
        # 【预训练机制目的】
        # - pretrain_steps=200: 让RL先探索学习
        # - freeze_q_table(): 固定策略后纯执行
        # - 这样可以分离"学习过程"和"执行效果"
        # =====================================================
        #
        # @theory: Q-Learning for Policy Optimization
        # State: [Grain(3), Income(3), Fiscal(2)] = 18 states
        # Action: [Maintain, Sub++, Sub--, Price++, Price--] = 5 actions
        self.rl_enabled = policy_cfg.get('rl_enabled', False)
        self.q_table = {}  # state -> [q_values]
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2  # Exploration rate
        self.last_state = None
        self.last_action = None
        
        # @new 2026-01-03: 预训练机制
        # @theory: 先探索学习，再固定策略执行
        # 解决"自适应学习"的可解释性问题
        self.pretrain_steps = policy_cfg.get('pretrain_steps', 200)  # 默认200步预训练
        self.q_table_frozen = False  # 是否冻结Q-table
        self.pretrain_completed = False
        self.training_step_count = 0
        self._frozen = False  # 外部加载冻结Q-table标志（纯exploitation模式）
        self.government_q_sar = []
        self._government_q_pending = None


    def step(self):
        """
        @theory
        政府年度决策周期 - 严格按照多层次博弈设计规范的阶段顺序执行
        """
        self.begin_annual_cycle()
        self.end_annual_cycle()

    def _get_current_step(self):
        """Return the model year index even before RandomActivation increments."""
        return getattr(self.model, 'step_count', getattr(self.model.schedule, 'steps', 0))

    def begin_annual_cycle(self):
        """Run ex-ante government actions before markets and production."""
        from core.p3_project_contracts import configured_program
        from core.p3_project_payments import begin
        program = configured_program(self.model)
        if self.government_q_project_enabled():
            program = self._begin_government_q_project(program)
        if program is not None:
            begin(self, program)
            return
        self._reset_annual_budget()
        self.announce_policy()
        self._reset_annual_subsidy_runtime()
        self.announce_subsidy_program()

    def end_annual_cycle(self):
        """Run ex-post evaluation after the year's outcomes are realized."""
        self.verify_subsidy_claims()
        self.distribute_subsidies()
        if self.government_q_project_enabled():
            self._government_q_outcome = self._government_q_performance()
            self._evaluate_fiscal_status()
            return
        self.evaluate_and_adjust_policy()
        self._evaluate_fiscal_status()

    def government_q_project_enabled(self):
        return self.model.config.get('policy', {}).get('government_q_project', {}).get('enabled', False) is True

    def _government_q_performance(self, initial=False):
        # Capture after payments and before demographic removal of this year's producers.
        cohort = self._government_q_h0
        grain = 0.0
        if not initial:
            records = getattr(self.model, 'revision_harvest_records', None)
            if not isinstance(records, dict) or not records:
                raise RuntimeError('Government Q requires current revision_harvest_records from the harvest observer')
            if any(row['step'] != self._get_current_step() for row in records.values()):
                raise RuntimeError('Government Q rejects stale harvest observations')
            if any(str(f.unique_id) not in records or
                   getattr(f, 'last_income_settlement', {}).get('step') != self._get_current_step()
                   for f in cohort):
                raise RuntimeError('Government Q requires every H0 household to settle this year')
            grain = sum(float(row['grain_physical_kg']) for row in records.values())
        income = sum(get_farmer_net_income(f) for f in cohort) / len(cohort)
        runtime = self.current_subsidy_runtime
        cost = 0.0 if initial else float(runtime['total_fiscal_cost_yuan'])
        envelope = 0.0 if initial else float(runtime['fiscal_envelope_yuan'])
        if any(not np.isfinite(v) for v in (grain, income, cost, envelope)):
            raise ValueError('Government Q observations must be finite')
        if grain < 0 or cost < 0 or envelope < 0 or cost > envelope + 1e-6:
            raise ValueError('Government Q requires nonnegative output and funded fiscal payments')
        return dict(grain_output_kg=grain, h0_mean_net_income_yuan=income, h0_count=len(cohort),
                    fiscal_cost_yuan=cost, fiscal_envelope_yuan=envelope,
                    subsidy_paid_yuan=float(self.total_subsidy_paid),
                    implementation_cost_yuan=float(runtime.get('implementation_cost_yuan', 0.0)))

    def _government_q_state(self, observed, support_index, remaining):
        grain_ref, income_ref = self._government_q_references
        grain = observed['grain_output_kg'] / grain_ref
        income = observed['h0_mean_net_income_yuan'] / income_ref
        envelope = observed['fiscal_envelope_yuan']
        fiscal = observed['fiscal_cost_yuan'] / envelope if envelope > 0 else 0.0
        return (0 if grain < .9 else 1 if grain < 1.1 else 2,
                0 if income < .9 else 1 if income < 1.1 else 2,
                int(fiscal > .5), support_index, remaining)

    def _begin_government_q_project(self, program):
        from core.p3_project_contracts import P3Program
        cfg = self.model.config['policy']['government_q_project']
        menu = cfg.get('budget_area_menu')
        allowed_rules = ('area_universal', 'scale_eligibility') if menu is not None else ('area_universal',)
        if program is None or program.rule not in allowed_rules:
            raise ValueError('Government Q project requires a funded P3 area_universal program')
        if self._government_q_pending is not None:
            raise RuntimeError('Previous government Q transition has not finished')
        own_rng = self.__dict__.get('np_random')
        if own_rng is None or own_rng is getattr(self.model, 'np_random', None) or not hasattr(own_rng, 'randint'):
            raise ValueError('Government Q requires an independent government.np_random RandomState')
        step = int(self._get_current_step())
        terminal = cfg['episode_end_step']
        if not isinstance(terminal, int) or isinstance(terminal, bool) or not 1 <= step <= terminal:
            raise ValueError('Government Q requires an integer episode_end_step at or after this year')
        if not self.government_q_sar:
            refs = (float(cfg['grain_reference_kg']), float(cfg['h0_income_reference_yuan']))
            if any(not np.isfinite(v) or v <= 0 for v in refs):
                raise ValueError('Government Q references must be finite positive ex-ante constants')
            cohort = tuple(getattr(self.model, 'initial_household_cohort', ()))
            if not cohort or len({str(f.unique_id) for f in cohort}) != len(cohort):
                raise ValueError('Government Q requires a nonempty fixed initial household cohort')
            self._government_q_h0 = cohort
            self._government_q_references = refs
            self._government_q_terminal = terminal
            observed = self._government_q_performance(initial=True)
        else:
            if (terminal != self._government_q_terminal or
                    (float(cfg['grain_reference_kg']), float(cfg['h0_income_reference_yuan'])) != self._government_q_references):
                raise ValueError('Government Q references and terminal year cannot change within an episode')
            observed = self._government_q_outcome
            if self.government_q_sar[-1]['done'] or self.government_q_sar[-1]['step'] != step - 1:
                raise RuntimeError('Government Q years must be consecutive within one episode')
        if menu is None:
            rates = (0.0, 350.0, 525.0)
            prior_rate = self.government_q_sar[-1]['nominal_rate_yuan_per_mu'] if self.government_q_sar else program.nominal_rate
            if prior_rate not in rates:
                raise ValueError('Initial government Q support must be 0, 350 or 525')
            prior_action = rates.index(prior_rate)
        else:
            if not np.isfinite(menu) or menu <= 0:
                raise ValueError('Budget menu requires positive fixed reference land area')
            program.require_ready()
            rate = program.trial_budget_yuan / (menu * program.trial_to_model)
            rates = (0.0, rate, rate)
            prior_action = self.government_q_sar[-1]['action'] if self.government_q_sar else 0
        state = self._government_q_state(observed, prior_action, terminal - step + 1)
        action = self._choose_action_rl(state, action_count=3)
        if action not in (0, 1, 2):
            raise ValueError('Government Q project action must select 0, 350 or 525')
        payload = program.model_dump()
        payload.update(nominal_rate=rates[action], rate_origin='b19_scenario',
                       scenario_id='government_q_p3_rate',
                       scenario_rate_source='author_government_q_actions_0_350_525_not_estimated')
        if menu is not None:
            # Total land bounds eligible grain area, so announced rates are fully funded.
            payload.update(rule='scale_eligibility' if action == 2 else 'area_universal',
                           scenario_id='government_q_budget_area_menu',
                           scenario_rate_source='trial_budget_divided_by_fixed_initial_total_land')
        program = P3Program.model_validate(payload)
        self.model.config['fiscal_system']['p3_support'] = program.model_dump()
        self.last_state, self.last_action = state, action
        self._government_q_pending = dict(step=step, state=state, action=action,
            nominal_rate_yuan_per_mu=rates[action], trial_price_year=program.trial_price_year,
            trial_to_model=program.trial_to_model, state_observations=dict(observed),
            q_before=float(self.q_table.get(state, [0.0] * 3)[action]),
            base_budget_yuan=program.base_budget_yuan, trial_budget_yuan=program.trial_budget_yuan,
            implementation_reserved_yuan=program.implementation_cost_yuan,
            grain_price_floor=float(self.grain_price_floor))
        if menu is not None:
            self._government_q_pending['program_rule'] = program.rule
        return program

    def finish_government_q_year(self):
        """Called once at the end of MainModel.step; never select next year's action here."""
        if not self.government_q_project_enabled() or self._government_q_pending is None:
            return
        row = self._government_q_pending
        if not self.p3_batch.paid or self.p3_batch.step != row['step']:
            raise RuntimeError('Government Q update requires this year\'s actual project payments')
        observed = self._government_q_outcome
        grain_ref, income_ref = self._government_q_references
        envelope = observed['fiscal_envelope_yuan']
        fiscal_ratio = observed['fiscal_cost_yuan'] / envelope if envelope > 0 else 0.0
        income_ratio = observed['h0_mean_net_income_yuan'] / income_ref
        uncapped = self.model.config['policy']['government_q_project'].get('uncapped_income_reward', False)
        income_component = float(np.arcsinh(income_ratio - 1.0)) if uncapped else 1.5 * min(income_ratio, 1.5)
        components = dict(grain_security=2.0 * (observed['grain_output_kg'] / grain_ref - 1.0),
            farmer_welfare=income_component,
            fiscal_balance=-3.0 * fiscal_ratio)
        configured_weights = self.model.config.get('policy', {}).get('rl_reward_weights', {})
        weights = {key: float(configured_weights.get(key, self.objective_weights[key])) for key in components}
        if any(not np.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError('Government Q reward weights must be finite nonnegative values')
        reward = sum(weights[key] * value for key, value in components.items())
        done = row['step'] == self._government_q_terminal
        next_state = self._government_q_state(observed, row['action'], self._government_q_terminal - row['step'])
        self._update_q_table(reward, next_state, done=done)
        row.update(next_state=next_state, reward=reward, done=done,
                   reward_components=components, reward_weights=weights,
                   outcome_observations=dict(observed),
                   grain_reference_kg=grain_ref, h0_income_reference_yuan=income_ref,
                   q_after=float(self.q_table.get(row['state'], [0.0] * 3)[row['action']]),
                   q_updated=not self.q_table_frozen,
                   governance_mode=('government_q_budget_area_menu' if 'program_rule' in row else 'government_q_p3_rate'))
        if uncapped:
            row['income_reward_rule'] = 'asinh_centered_on_fixed_initial_income'
        self.government_q_sar.append(row)
        self._government_q_pending = None

    def _reset_annual_budget(self):
        """
        @new 2025-11-30: 年初重置年度预算
        @fix 2025-12-22: 添加GDP联动预算增长，避免30年模拟财政爆炸
        
        财政悬崖机制：
        - 如果连续多年超支，预算会被削减
        - 反映中央财政对地方的约束
        
        GDP联动增长机制：
        - 年度预算随GDP增长同步调整
        - 反映财政收入与经济增长的关联
        """
        # @fix 2025-12-22: GDP联动预算调整
        fiscal_cfg = resolve_fiscal_system_config(self.model.config, create=False)
        gdp_growth_rate = fiscal_cfg.get('gdp_growth_rate', 0.05)  # 默认5%年增长
        
        # 根据模拟年份调整基础预算
        year = getattr(self.model, 'step_count', 0)
        configured_budget = None
        if 'annual_subsidy_budget' in fiscal_cfg:
            configured_budget = _float_config_value(fiscal_cfg, 'annual_subsidy_budget', self.base_annual_subsidy_budget)
            self.base_annual_subsidy_budget = configured_budget

        if configured_budget is not None and configured_budget <= 0:
            gdp_adjusted_budget = 0.0
        else:
            gdp_adjusted_budget = self.base_annual_subsidy_budget * ((1 + gdp_growth_rate) ** year)
        
        base_budget = gdp_adjusted_budget
        
        if self.fiscal_cliff_active:
            # 计算削减后的预算
            reduction = 1 - (self.fiscal_reduction_rate * 
                           (self.consecutive_overspending_count - self.consecutive_overspending_years + 1))
            reduction = max(self.minimum_budget_ratio, reduction)
            base_budget = gdp_adjusted_budget * reduction
            print(f"  [⚠ 财政悬崖] 预算削减至 {reduction*100:.0f}%，本年预算 {base_budget/10000:.1f}万元")
        
        self.current_annual_subsidy_budget = float(base_budget)
        self.annual_subsidy_budget = self.current_annual_subsidy_budget
        self.remaining_budget = self.current_annual_subsidy_budget
        
        # @new 2025-12-22: 更新宏观经济状态
        self._update_macro_economy()

    def _get_non_farm_wage_bounds(self):
        labor_cfg = self.model.config.get('labor_market', {}) or {}
        floor = float(labor_cfg.get('wage_floor', 80.0) or 80.0)
        ceiling = float(labor_cfg.get('wage_ceiling', max(300.0, floor)) or max(300.0, floor))
        if ceiling < floor:
            ceiling = floor
        return floor, ceiling

    def _normalize_non_farm_daily_wage(self, wage_value, fallback=None):
        canonical_fallback = resolve_base_non_farm_wage(
            self.model.config,
            default=DEFAULT_BASE_NON_FARM_WAGE,
        )
        if fallback is None:
            fallback = canonical_fallback
        try:
            candidate = float(wage_value)
        except (TypeError, ValueError):
            candidate = float(fallback)

        if candidate <= 0:
            candidate = float(fallback)

        # Legacy macro configs sometimes stored an annual wage (~50000 yuan/year)
        # while the ABM downstream expects a daily wage signal.
        if candidate > 1000:
            candidate = candidate / 260.0

        floor, ceiling = self._get_non_farm_wage_bounds()
        return float(min(ceiling, max(floor, candidate)))

    def _get_grain_price_floor_bounds(self):
        market_cfg = self.model.config.get('environment', {}).get('market', {})
        grain_limits = market_cfg.get('price_limits', {}).get('grain', {})
        min_floor = float(grain_limits.get('min', 0.8) or 0.8)
        max_floor = float(grain_limits.get('max', max(min_floor, 3.0)) or max(min_floor, 3.0))
        if max_floor < min_floor:
            max_floor = min_floor
        return min_floor, max_floor

    def _policy_disables_grain_intervention(self, policy_state=None, scenario_id=None):
        scenario_label = str(scenario_id or '').strip().lower()
        if scenario_label in {'no_policy', 'none', 'market_only'}:
            return True

        policy = policy_state or {}
        if not isinstance(policy, dict):
            return False

        explicit_disable = policy.get('disable_grain_intervention')
        if explicit_disable is not None:
            return bool(explicit_disable)

        try:
            floor_value = float(policy.get('grain_price_floor', 0.0) or 0.0)
        except (TypeError, ValueError):
            floor_value = 0.0
        try:
            subsidy_rate = float(policy.get('grain_subsidies_rate', 0.0) or 0.0)
        except (TypeError, ValueError):
            subsidy_rate = 0.0

        return floor_value <= 0.0 and subsidy_rate <= 0.0

    def _clamp_grain_price_floor(self, floor_value, allow_zero=False):
        min_floor, max_floor = self._get_grain_price_floor_bounds()
        try:
            candidate = float(floor_value)
        except (TypeError, ValueError):
            candidate = min_floor
        if allow_zero and candidate <= 0:
            return 0.0
        return float(max(min_floor, min(max_floor, candidate)))

    def _get_current_subsidy_budget_target(self):
        current_budget = float(getattr(self, 'current_annual_subsidy_budget', 0.0) or 0.0)
        if current_budget <= 0:
            current_budget = float(getattr(self, 'annual_subsidy_budget', 0.0) or 0.0)
        if current_budget <= 0:
            current_budget = float(getattr(self, 'remaining_budget', 0.0) or 0.0)
        return max(1.0, current_budget)

    def _get_model_scale_grain_reference(self, market_performance=None):
        stored_reference = float(getattr(self, '_grain_target_reference', 0.0) or 0.0)
        structural_candidates = []

        market_cfg = self.model.config.get('environment', {}).get('market', {})
        structural_candidates.append(market_cfg.get('base_grain_demand', 0.0))

        env = getattr(self.model, 'environment', None)
        grain_market_history = getattr(env, 'grain_market_history', []) if env is not None else []
        if grain_market_history:
            early_snapshots = grain_market_history[: max(1, min(3, len(grain_market_history)))]
            for snapshot in early_snapshots:
                structural_candidates.append((snapshot or {}).get('grain_supply', 0.0))

        if stored_reference > 0:
            structural_candidates.append(stored_reference)

        numeric_structural = []
        for value in structural_candidates:
            try:
                numeric_structural.append(float(value or 0.0))
            except (TypeError, ValueError):
                continue

        reference = max([1.0] + numeric_structural)

        if reference <= 1.0:
            fallback_candidates = []
            if isinstance(market_performance, dict):
                fallback_candidates.append(market_performance.get('grain_output', 0.0))
            if grain_market_history:
                fallback_candidates.append((grain_market_history[-1] or {}).get('grain_supply', 0.0))

            numeric_fallback = []
            for value in fallback_candidates:
                try:
                    numeric_fallback.append(float(value or 0.0))
                except (TypeError, ValueError):
                    continue
            reference = max([1.0] + numeric_fallback)

        if reference > 1.0:
            self._grain_target_reference = max(stored_reference, reference)

        return max(1.0, float(getattr(self, '_grain_target_reference', reference) or reference))

    def _get_effective_grain_target(self, market_performance=None):
        raw_target = float(self.target_grain_output or 0.0)
        reference_scale = self._get_model_scale_grain_reference(market_performance)
        market_cfg = self.model.config.get('environment', {}).get('market', {})
        target_cfg = (
            self.model.config.get('policy', {})
            .get('government_objectives', {})
            .get('targets', {})
        )

        demand_reference = float(market_cfg.get('base_grain_demand', reference_scale) or reference_scale)
        demand_reference = max(1.0, demand_reference)

        security_buffer = float(target_cfg.get('grain_security_buffer', 1.15) or 1.15)
        security_buffer = max(1.0, security_buffer)
        max_target_demand_ratio = float(target_cfg.get('max_target_demand_ratio', 2.6) or 2.6)
        max_target_demand_ratio = max(security_buffer, max_target_demand_ratio)
        rolling_supply_weight = float(target_cfg.get('rolling_supply_weight', 0.35) or 0.35)
        rolling_supply_weight = max(0.0, min(1.0, rolling_supply_weight))

        demand_buffer_target = demand_reference * security_buffer
        blended_scale_target = (
            rolling_supply_weight * reference_scale
            + (1.0 - rolling_supply_weight) * demand_reference
        )
        demand_aligned_target = min(
            max(demand_buffer_target, blended_scale_target),
            demand_reference * max_target_demand_ratio,
        )

        if raw_target <= 0:
            return demand_aligned_target

        if raw_target > reference_scale * 5.0 or raw_target < demand_reference * 0.25:
            return demand_aligned_target

        return max(
            demand_buffer_target,
            min(raw_target, demand_reference * max_target_demand_ratio),
        )

    def _get_effective_policy_targets(self, market_performance=None):
        return {
            'grain_output': self._get_effective_grain_target(market_performance),
            'avg_income': max(1.0, float(self.target_avg_income or 1.0)),
            'subsidy_budget': self._get_current_subsidy_budget_target(),
        }
    
    def _update_macro_economy(self):
        """
        @new 2025-12-22: 宏观经济闭环更新
        @enhanced 2025-12-28: 添加劳动力反馈回路（一般均衡视角）
        
        核心机制：
        1. 补贴支出 → 挤出非农投资 → 降低非农GDP增速 → 减少税收 → 压缩非农工资
        2. @new: 农村劳动力转移 → 城市劳动力供给增加 → 工资增长放缓 → 抑制进一步转移
        
        理论依据：
        - Lewis (1954): 二元经济模型
        - Harris-Todaro (1970): 城乡迁移模型
        这实现了Nature级模型要求的系统封闭性与自调节性
        """
        if not self.enable_macro_feedback:
            return
        
        fiscal_cfg = resolve_fiscal_system_config(self.model.config, create=False)
        gdp_growth_rate = fiscal_cfg.get('gdp_growth_rate', 0.05)
        year = getattr(self.model, 'step_count', 0)
        
        # ========== 1. 计算农村劳动力转移率 ==========
        # @new 2025-12-28: 劳动力反馈回路核心
        rural_labor_transfer_rate = self._calculate_rural_labor_transfer()
        
        # 城市劳动力供给冲击：农村转移劳动力占城镇劳动力增量的比例
        # 假设城镇劳动力每年自然增长2%，农村转移贡献额外增量
        urban_labor_natural_growth = 0.02
        urban_labor_supply_shock = rural_labor_transfer_rate * 0.3  # 农村转移占城镇新增的30%
        
        # 工资增长阻尼：劳动力供给增加会抑制工资增长
        # Phillips曲线逻辑：供给冲击大时，工资增长放缓
        wage_growth_damping = max(0.5, 1.0 - urban_labor_supply_shock * 0.8)
        
        # ========== 2. 计算补贴挤出效应 ==========
        subsidy_to_tax_ratio = self.total_subsidy_paid / max(self.total_tax_revenue, 1)
        
        threshold = 0.2  # 20%以下不产生挤出
        if subsidy_to_tax_ratio > threshold:
            self.crowdout_effect = self.subsidy_crowdout_elasticity * (subsidy_to_tax_ratio - threshold)
        else:
            self.crowdout_effect = 0
        
        # ========== 3. 更新非农GDP ==========
        effective_growth = max(0, gdp_growth_rate - self.crowdout_effect * 0.5)
        self.current_non_farm_gdp = self.base_non_farm_gdp * ((1 + effective_growth) ** year)
        
        # ========== 4. 更新税收 ==========
        self.total_tax_revenue = self.current_non_farm_gdp * self.tax_rate
        
        # ========== 5. 更新非农工资（综合考虑挤出效应+劳动力供给冲击） ==========
        base_wage_growth = effective_growth * 0.8  # 工资增长略低于GDP增长
        # 劳动力供给增加会抑制工资增长
        adjusted_wage_growth = base_wage_growth * wage_growth_damping
        
        self.adjusted_non_farm_wage = (
            self.non_farm_wage_base * 
            ((1 + adjusted_wage_growth) ** year) * 
            (1 - self.crowdout_effect * 0.3)
        )
        self.adjusted_non_farm_wage = self._normalize_non_farm_daily_wage(
            self.adjusted_non_farm_wage,
            fallback=self.non_farm_wage_base,
        )
        
        # ========== 6. 计算迁移吸引力指标（反馈给农户决策） ==========
        # 当工资增长放缓时，迁移吸引力下降 → 抑制进一步转移
        migration_attractiveness = self.adjusted_non_farm_wage / max(self.non_farm_wage_base, 1)
        # 加入城市生活成本因素（工资越高的城市生活成本也越高）
        migration_attractiveness *= max(0.6, 1.0 - urban_labor_supply_shock * 0.5)
        
        # ========== 7. 传递给环境 ==========
        if hasattr(self.model, 'environment'):
            env = self.model.environment
            # Keep the canonical market wage and the legacy alias synchronized so
            # farmer/enterprise decision modules observe the same wage signal.
            env.off_farm_wage = self.adjusted_non_farm_wage
            if hasattr(env, 'market_prices') and isinstance(env.market_prices, dict):
                env.market_prices['wage_rate'] = self.adjusted_non_farm_wage
            env.base_wage = self.adjusted_non_farm_wage
            env.migration_attractiveness = migration_attractiveness
            env.macro_state = {
                'non_farm_gdp': self.current_non_farm_gdp,
                'tax_revenue': self.total_tax_revenue,
                'crowdout_effect': self.crowdout_effect,
                'adjusted_wage': self.adjusted_non_farm_wage,
                'rural_labor_transfer_rate': rural_labor_transfer_rate,
                'wage_growth_damping': wage_growth_damping,
                'migration_attractiveness': migration_attractiveness
            }
    
    def _calculate_rural_labor_transfer(self):
        """
        @new 2025-12-28: 计算农村劳动力转移率
        
        基于当期农户的生计策略分布计算转移率：
        - Non_Farmer比例增加 = 劳动力转移
        - 与上年比较得到转移率
        
        理论依据：Todaro (1969) 预期收入迁移模型
        """
        if not hasattr(self.model, 'farmers') or len(self.model.farmers) == 0:
            return 0.0
        
        # 计算当前Non_Farmer比例
        non_farmer_count = sum(
            1 for f in self.model.farmers 
            if hasattr(f, 'livelihood_strategy') and 
            getattr(f.livelihood_strategy, 'name', '') == 'Non_Farmer'
        )
        current_non_farmer_ratio = non_farmer_count / len(self.model.farmers)
        
        # 获取历史数据计算转移率
        if not hasattr(self, 'non_farmer_ratio_history'):
            self.non_farmer_ratio_history = []
        
        transfer_rate = 0.0
        if len(self.non_farmer_ratio_history) > 0:
            prev_ratio = self.non_farmer_ratio_history[-1]
            transfer_rate = max(0, current_non_farmer_ratio - prev_ratio)
        
        # 记录当期比例
        self.non_farmer_ratio_history.append(current_non_farmer_ratio)
        
        return transfer_rate
    
    def _evaluate_fiscal_status(self):
        """
        @new 2025-11-30: 年末财政状况评估
        
        检查是否超支，更新财政悬崖状态
        """
        utilization = self.total_subsidy_paid / self.annual_subsidy_budget if self.annual_subsidy_budget > 0 else 0
        self.budget_utilization_history.append({
            'step': self._get_current_step(),
            'budget': self.annual_subsidy_budget,
            'spent': self.total_subsidy_paid,
            'utilization': utilization,
            'crowdout_effect': self.crowdout_effect,
            'adjusted_wage': self.adjusted_non_farm_wage
        })
        
        if utilization > self.overspending_threshold:
            self.consecutive_overspending_count += 1
            print(f"  [⚠ 财政预警] 预算超支 {(utilization-1)*100:.1f}%，连续{self.consecutive_overspending_count}年")
            
            if self.consecutive_overspending_count >= self.consecutive_overspending_years:
                if not self.fiscal_cliff_active:
                    self.fiscal_cliff_active = True
                    print(f"  [! 财政悬崖触发] 连续{self.consecutive_overspending_count}年超支，明年预算将削减")
        else:
            # 如果今年没超支，重置计数器
            self.consecutive_overspending_count = 0
            self.fiscal_cliff_active = False

    def announce_policy(self):
        """
        @theory
        阶段一：政策发布
        公布本年度的各项补贴率和价格地板，为所有参与者的决策设定外部支付结构
        支持从 policy_params.json 读取多种补贴场景，并支持 policy.overrides 动态覆盖
        """
        import json
        from pathlib import Path
        
        # 读取配置
        policy_section = self.model.config.get('policy', {})
        policy_config = self.gov_policy_config or policy_section.get('government', {})
        scenario_id = policy_config.get('scenario_id', policy_section.get('active_scenario', 'baseline_cash'))
        policy_overrides = policy_section.get('overrides', {})
        self.active_scenario_id = scenario_id
        
        # 读取 policy_params.json 中的场景配置
        policy_params_path = PROJECT_ROOT / 'config' / 'policy_params.json'
        policy_scenarios = {}
        if policy_params_path.exists():
            try:
                with open(policy_params_path, 'r', encoding='utf-8') as f:
                    policy_data = json.load(f)
                    policy_scenarios = policy_data.get('scenarios', {})
            except Exception as e:
                print(f"[警告] 无法读取政策配置文件: {e}")
        
        # 构建政策参数（优先级：overrides > runtime policy > policy_params.json > 默认值）
        scenario = {}
        if scenario_id in policy_scenarios:
            scenario = dict(policy_scenarios[scenario_id])

        runtime_policy = dict(getattr(self.model.environment, 'active_policy', {}) or {})
        if runtime_policy:
            scenario.update(runtime_policy)
        
        # 应用 overrides（关键修复）
        if policy_overrides:
            scenario.update(policy_overrides)
            print(f"[政府] 检测到 policy.overrides，已应用: {list(policy_overrides.keys())}")
        
        # 设置政府参数
        disable_grain_intervention = self._policy_disables_grain_intervention(
            policy_state=scenario,
            scenario_id=scenario_id,
        )
        self.grain_price_floor = self._clamp_grain_price_floor(
            scenario.get('grain_price_floor', policy_config.get('grain_price_floor', BASELINE_GRAIN_PRICE_FLOOR)),
            allow_zero=disable_grain_intervention,
        )
        self.grain_subsidies_rate = scenario.get(
            'grain_subsidies_rate',
            policy_config.get('grain_subsidies_rate', LEGACY_GRAIN_SUBSIDIES_RATE),
        )
        self.subsidy_rate = scenario.get(
            'subsidy_rate',
            scenario.get('grain_subsidies_rate', self.grain_subsidies_rate)
        )
        
        subsidy_type_str = scenario.get('subsidy_type', policy_config.get('subsidy_type', 'area_based'))
        self.subsidy_type = SubsidyType.AREA_BASED if subsidy_type_str == 'area_based' else SubsidyType.OUTPUT_BASED
        
        subsidy_target_str = scenario.get('subsidy_target', policy_config.get('subsidy_target', 'operator'))
        self.subsidy_target = SubsidyTarget.OPERATOR if subsidy_target_str == 'operator' else SubsidyTarget.OWNER
        
        # 构建完整政策字典传递给环境
        policy_payload = {
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate,
            'base_subsidy_per_mu': scenario.get(
                'base_subsidy_per_mu',
                policy_config.get('base_subsidy_per_mu', 0)
            ),
            'subsidy_type': subsidy_type_str,
            'subsidy_target': subsidy_target_str,
            # In Environment.active_policy, subsidy_rate is retained only as a
            # legacy alias of the canonical grain_subsidies_rate. The
            # government's adaptive self.subsidy_rate remains an internal
            # learning variable and must not create a runtime alias conflict.
            'subsidy_rate': self.grain_subsidies_rate,
            'delivery_mode': scenario.get('delivery_mode', 'cash'),
            'payment_lag': scenario.get('payment_lag', 0),
            'cash_share': scenario.get('cash_share', 1.0),
            'in_kind_package': scenario.get('in_kind_package', {}),
            'tier_breakpoints': scenario.get('tier_breakpoints', []),
            'tier_rates': scenario.get('tier_rates', [])
            ,
            'disable_grain_intervention': disable_grain_intervention,
        }
        
        # 将政策信息传递给环境
        self.model.environment.update_policy(self)
        self.model.environment.set_active_policy(policy_payload, policy_id=scenario_id)
        self.model.environment.manual_subsidy_dispatch = True
        region_summary = self._assign_regional_policies(policy_scenarios)
        
        # 记录政策历史
        self.policy_history.append({
            'step': self._get_current_step(),
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate,
            'subsidy_rate': self.subsidy_rate,
            'scenario_id': scenario_id
        })
        
        if region_summary:
            region_display = {region: info.get('scenario_id') for region, info in region_summary.items()}
            print(f"[政府] 区域差异化策略: {region_display}")
        
        print(f"政府发布政策（场景：{scenario_id}）：粮食最低收购价={self.grain_price_floor}，粮食补贴率={self.grain_subsidies_rate}")

    def get_current_policy(self):
        """
        @fix: 新增方法，返回当前政策字典以修复AttributeError
        """
        return {
            'subsidy_type': self.subsidy_type.value,
            'subsidy_target': self.subsidy_target.value,
            'subsidy_rate': self.subsidy_rate,
            'grain_price_floor': self.grain_price_floor,
            'grain_subsidies_rate': self.grain_subsidies_rate
        }
    
    def set_active_policy(self, policy_id, policy_state):
        """
        设置当前活动政策（由ModelBuilder调用）
        
        参数:
            policy_id: 政策场景ID
            policy_state: 政策状态字典
        """
        self.active_scenario_id = policy_id
        if policy_state:
            disable_grain_intervention = self._policy_disables_grain_intervention(
                policy_state=policy_state,
                scenario_id=policy_id,
            )
            self.grain_price_floor = self._clamp_grain_price_floor(
                policy_state.get('grain_price_floor', BASELINE_GRAIN_PRICE_FLOOR),
                allow_zero=disable_grain_intervention,
            )
            self.grain_subsidies_rate = policy_state.get('grain_subsidies_rate', LEGACY_GRAIN_SUBSIDIES_RATE)
            self.subsidy_rate = policy_state.get(
                'subsidy_rate',
                policy_state.get('grain_subsidies_rate', LEGACY_GRAIN_SUBSIDIES_RATE)
            )
            if disable_grain_intervention:
                self.grain_price_floor = 0.0
                self.grain_subsidies_rate = 0.0
                self.subsidy_rate = 0.0
            
            subsidy_type_str = policy_state.get('subsidy_type', 'area_based')
            self.subsidy_type = SubsidyType.AREA_BASED if subsidy_type_str == 'area_based' else SubsidyType.OUTPUT_BASED
            
            subsidy_target_str = policy_state.get('subsidy_target', 'operator')
            self.subsidy_target = SubsidyTarget.OPERATOR if subsidy_target_str == 'operator' else SubsidyTarget.OWNER

    def _get_region_label(self, actor):
        if hasattr(self.model, 'get_region_label'):
            return self.model.get_region_label(actor)
        if not hasattr(actor, 'pos') or actor.pos is None:
            return 'global'
        _, y = actor.pos
        height = max(1, getattr(self.model.grid, 'height', 1))
        if y < height / 3:
            return 'north'
        if y < (2 * height) / 3:
            return 'central'
        return 'south'

    def _compute_region_statistics(self):
        stats = {}
        regions = {}
        for farmer in getattr(self.model, 'farmers', []):
            region = self._get_region_label(farmer)
            regions.setdefault(region, []).append(farmer)
        for region, farmers in regions.items():
            if not farmers:
                continue
            net_incomes = [get_farmer_net_income(f) for f in farmers]
            gross_incomes = [get_farmer_gross_income(f) for f in farmers]
            agri_ratios = [get_farmer_livelihood_agriculture_ratio(f) for f in farmers]
            avg_income = sum(net_incomes) / len(net_incomes) if net_incomes else 0.0
            avg_gross_income = sum(gross_incomes) / len(gross_incomes) if gross_incomes else 0.0
            avg_agri_ratio = sum(agri_ratios) / len(agri_ratios) if agri_ratios else 0.0
            stats[region] = {
                'avg_income': avg_income,
                'avg_net_income': avg_income,
                'avg_gross_income': avg_gross_income,
                'avg_agri_ratio': avg_agri_ratio,
                'population': len(farmers)
            }
            network_meta = getattr(self.model, 'network_metrics', {}).get(region, {})
            stats[region]['avg_transaction_degree'] = network_meta.get('avg_transaction_degree', 0.0)
            stats[region]['transaction_density'] = network_meta.get('transaction_density', 0.0)
        self.region_stats = stats
        return stats

    def _assign_regional_policies(self, policy_scenarios):
        cfg = self.gov_policy_config or {}
        if not cfg.get('enable_regional_heterogeneity'):
            self.regional_policy_assignments = {}
            self.latest_regional_summary = {}
            return None
        if str(self.active_scenario_id or '').lower() == 'no_policy':
            self.regional_policy_assignments = {}
            self.latest_regional_summary = {}
            return None
        stats = self._compute_region_statistics()
        if not stats:
            self.regional_policy_assignments = {}
            self.latest_regional_summary = {}
            return None
        thresholds = cfg.get('regional_thresholds', {})
        income_low = thresholds.get('income_low', 20000)
        agri_ratio_high = thresholds.get('agri_ratio_high', 0.8)
        density_low = thresholds.get('network_density_low', 0.2)
        scenario_map = cfg.get('regional_scenarios', {})
        vulnerable_scenario = scenario_map.get('vulnerable', self.active_scenario_id)
        transforming_scenario = scenario_map.get('transforming', self.active_scenario_id)
        assignments = {}
        summary = {}
        for region, region_stats in stats.items():
            scenario_choice = transforming_scenario
            density = region_stats.get('transaction_density', 0)
            if (region_stats.get('avg_income', 0) < income_low or
                region_stats.get('avg_agri_ratio', 0) > agri_ratio_high or
                density < density_low):
                scenario_choice = vulnerable_scenario
            scenario_payload = policy_scenarios.get(scenario_choice)
            if not scenario_payload:
                scenario_payload = policy_scenarios.get(self.active_scenario_id, self.model.environment.active_policy)
            assignments[region] = {
                'scenario_id': scenario_choice,
                'policy': scenario_payload
            }
            summary[region] = {
                'scenario_id': scenario_choice,
                'avg_income': region_stats.get('avg_income', 0),
                'avg_net_income': region_stats.get('avg_net_income', region_stats.get('avg_income', 0)),
                'avg_gross_income': region_stats.get('avg_gross_income', 0),
                'avg_agri_ratio': region_stats.get('avg_agri_ratio', 0),
                'population': region_stats.get('population', 0)
            }
        self.regional_policy_assignments = assignments
        self.latest_regional_summary = summary
        return summary

    def _get_region_scenario_id(self, region):
        entry = self.regional_policy_assignments.get(region)
        return entry.get('scenario_id') if entry else self.active_scenario_id

    def _get_policy_for_region(self, region):
        entry = self.regional_policy_assignments.get(region)
        return entry.get('policy') if entry else None
    
    def _find_agent_by_id(self, agent_id):
        """
        @new 2025-12-04: 根据ID查找主体
        用于拖欠清偿时找到对应的农户或企业
        """
        model_lookup = getattr(self.model, 'get_agent_by_id', None)
        if callable(model_lookup):
            found = model_lookup(agent_id)
            if found is not None:
                return found
        # 先在农户中查找
        for farmer in self.model.farmers:
            if farmer.unique_id == agent_id:
                return farmer
        # 再在企业中查找
        for enterprise in self.model.agricultural_enterprises:
            if enterprise.unique_id == agent_id:
                return enterprise
        return None

    def _sanitize_runtime_payload(self, payload):
        """Convert runtime ledger payloads into deterministic, JSON-safe records."""
        if isinstance(payload, dict):
            sanitized = {}
            for key, value in payload.items():
                if key == 'agent':
                    agent_id = getattr(value, 'unique_id', None)
                    if agent_id is not None:
                        sanitized['agent_id'] = agent_id
                    continue
                sanitized[key] = self._sanitize_runtime_payload(value)
            return sanitized
        if isinstance(payload, list):
            return [self._sanitize_runtime_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._sanitize_runtime_payload(item) for item in payload]
        if hasattr(payload, 'unique_id'):
            return getattr(payload, 'unique_id')
        return payload

    def _reset_annual_subsidy_runtime(self):
        """Initialize an auditable annual subsidy runtime ledger."""
        self.subsidy_claims_submitted = []
        self.current_subsidy_runtime = {
            'step': self._get_current_step(),
            'scenario_id': self.active_scenario_id,
            'policy': dict(getattr(self.model.environment, 'active_policy', {}) or {}),
            'claims_submitted': [],
            'claims_pre_reviewed': [],
            'claims_verified': [],
            'claims_rejected': [],
            'payments_made': [],
            'queued_payments': [],
        }

    def announce_subsidy_program(self):
        """Record the announced program before agents react to it."""
        announcement = {
            'step': self._get_current_step(),
            'scenario_id': self.active_scenario_id,
            'policy': dict(getattr(self.model.environment, 'active_policy', {}) or {}),
        }
        self.subsidy_announcements.append(announcement)
        if self.current_subsidy_runtime:
            self.current_subsidy_runtime['announcement'] = announcement

    def register_subsidy_claim(self, claim):
        """Collect a realized subsidy claim generated after production is observed."""
        if getattr(self, 'p3_batch', None) is not None:
            from core.p3_project_payments import submit
            submit(self, claim['agent'])
            return claim
        if not isinstance(claim, dict):
            return None

        actor = claim.get('agent')
        agent_id = claim.get('agent_id') or getattr(actor, 'unique_id', None)
        if agent_id is None:
            return None

        normalized = dict(claim)
        normalized['agent'] = actor or self._find_agent_by_id(agent_id)
        normalized['agent_id'] = agent_id
        normalized.setdefault('claim_step', self._get_current_step())
        default_claim_id = (
            f"subsidy_claim_{normalized['claim_step']}_"
            f"{normalized.get('actor_type', 'farmer')}_{agent_id}"
        )
        normalized.setdefault('claim_id', default_claim_id)
        normalized.setdefault('region', self._get_region_label(normalized['agent']) if normalized.get('agent') is not None else 'global')
        normalized.setdefault('actor_type', 'farmer')
        normalized.setdefault('eligible_area', 0.0)
        normalized.setdefault('grain_area', 0.0)
        normalized.setdefault('scale_qualified', False)
        normalized.setdefault('priority_score', normalized.get('grain_area', 0.0))
        normalized.setdefault('primary_village_id', None)
        normalized.setdefault('village_area_breakdown', {})
        normalized.setdefault('village_grain_area_breakdown', {})
        normalized.setdefault('village_reviews', [])

        self.subsidy_claims_submitted.append(normalized)
        if self.current_subsidy_runtime:
            self.current_subsidy_runtime['claims_submitted'].append(
                self._sanitize_runtime_payload(normalized)
            )
            self.current_subsidy_runtime['claims_submitted'].sort(
                key=lambda item: (
                    int(item.get('claim_step', 0) or 0),
                    str(item.get('actor_type', 'farmer')),
                    str(item.get('agent_id', '')),
                    str(item.get('claim_id', '')),
                )
            )
        return normalized

    def verify_subsidy_claims(self):
        """Verify realized subsidy claims against the announced policy rules."""
        if getattr(self, 'p3_batch', None) is not None:
            from core.p3_project_payments import verify
            return verify(self)
        if not self.subsidy_claims_submitted:
            return []

        current_step = self._get_current_step()
        payment_lag = int((getattr(self.model.environment, 'active_policy', {}) or {}).get('payment_lag', 0))
        verified_claims = []
        rejected_claims = []
        village_review_cfg = self.model.config.get('agents', {}).get('village_collective', {}).get('subsidy_review', {})
        direct_review_actor_types = set(village_review_cfg.get('direct_review_actor_types', ['enterprise']))
        area_tolerance = float(village_review_cfg.get('area_tolerance', 1e-6))

        claims_to_verify = sorted(
            self.subsidy_claims_submitted,
            key=lambda item: (
                int(item.get('claim_step', 0) or 0),
                str(item.get('actor_type', 'farmer')),
                str(item.get('agent_id', '')),
                str(item.get('claim_id', '')),
            )
        )

        for claim in claims_to_verify:
            agent = claim.get('agent') or self._find_agent_by_id(claim.get('agent_id'))
            if agent is None:
                continue

            eligible_area = max(0.0, float(claim.get('eligible_area', 0.0) or 0.0))
            grain_area = max(0.0, float(claim.get('grain_area', 0.0) or 0.0))
            if eligible_area <= 0 and grain_area <= 0:
                continue

            actor_type = claim.get('actor_type', 'farmer')
            village_reviews = list(claim.get('village_reviews', []) or [])
            approved_reviews = sorted(
                [review for review in village_reviews if review.get('status') == 'approved'],
                key=lambda review: str(review.get('village_id', '')),
            )
            rejected_reviews = sorted(
                [review for review in village_reviews if review.get('status') == 'rejected'],
                key=lambda review: str(review.get('village_id', '')),
            )
            village_area_breakdown = dict(claim.get('village_area_breakdown', {}) or {})
            village_grain_breakdown = dict(claim.get('village_grain_area_breakdown', {}) or {})
            has_village_scope = (
                any(float(value or 0.0) > area_tolerance for value in village_area_breakdown.values())
                or any(float(value or 0.0) > area_tolerance for value in village_grain_breakdown.values())
                or claim.get('primary_village_id') is not None
            )

            verification_path = 'direct_government_review'
            village_review_status = 'not_required'
            if approved_reviews:
                eligible_area = max(
                    0.0,
                    sum(float(review.get('reviewed_eligible_area', 0.0) or 0.0) for review in approved_reviews),
                )
                grain_area = max(
                    0.0,
                    sum(float(review.get('reviewed_grain_area', 0.0) or 0.0) for review in approved_reviews),
                )
                verification_path = 'village_pre_review_then_government_re_review'
                village_review_status = 'approved'
            elif has_village_scope and actor_type not in direct_review_actor_types:
                rejected = dict(claim)
                rejected.update({
                    'agent': agent,
                    'verification_path': 'rejected_after_village_review_gap',
                    'government_re_review_status': 'rejected',
                    'rejection_reason': 'missing_village_pre_review_approval',
                    'rejected_step': current_step,
                })
                rejected_claims.append(rejected)
                continue
            elif rejected_reviews:
                village_review_status = 'rejected_but_direct_government_review'

            if eligible_area <= 0 and grain_area <= 0:
                rejected = dict(claim)
                rejected.update({
                    'agent': agent,
                    'verification_path': verification_path,
                    'government_re_review_status': 'rejected',
                    'rejection_reason': 'no_verified_area_after_review',
                    'rejected_step': current_step,
                })
                rejected_claims.append(rejected)
                continue

            region = claim.get('region') or self._get_region_label(agent)
            policy_override = self._get_policy_for_region(region)
            active_policy = policy_override or getattr(self.model.environment, 'active_policy', {}) or {}
            subsidy_target_str = str(active_policy.get('subsidy_target', 'operator') or 'operator').lower()
            subsidized_area = grain_area if subsidy_target_str == 'operator' else eligible_area
            subsidy_info = self.model.environment.calculate_subsidy_for_actor(
                subsidized_area,
                actor_type=actor_type,
                policy_override=policy_override,
            )
            scale_info = self.model.environment.calculate_scale_subsidy(grain_area, actor_type=actor_type)
            basic_cash = float(subsidy_info.get('cash', 0.0) or 0.0)
            scale_cash = float(scale_info.get('scale_subsidy', 0.0) or 0.0)
            in_kind_value = float(subsidy_info.get('in_kind_value', 0.0) or 0.0)
            verified = dict(claim)
            verified.update({
                'agent': agent,
                'region': region,
                'subsidy_info': subsidy_info,
                'basic_cash': basic_cash,
                'scale_cash': scale_cash,
                'in_kind_value': in_kind_value,
                'in_kind_breakdown': dict(subsidy_info.get('in_kind_breakdown', {}) or {}),
                'planned_amount': basic_cash + scale_cash + in_kind_value,
                'scale_qualified': bool(scale_info.get('qualified', False) or claim.get('scale_qualified', False)),
                'government_verified_eligible_area': eligible_area,
                'government_verified_grain_area': grain_area,
                'government_verified_subsidized_area': subsidized_area,
                'village_reviews': village_reviews,
                'village_review_status': village_review_status,
                'village_reviewed_area': sum(
                    float(review.get('reviewed_eligible_area', 0.0) or 0.0) for review in approved_reviews
                ),
                'village_reviewed_grain_area': sum(
                    float(review.get('reviewed_grain_area', 0.0) or 0.0) for review in approved_reviews
                ),
                'village_reviewer_ids': [review.get('village_id') for review in approved_reviews],
                'verification_path': verification_path,
                'government_re_review_status': 'approved',
                'verified_step': current_step,
                'payable_step': current_step + payment_lag,
                'regional_policy': self._get_region_scenario_id(region),
            })
            verified['priority_score'] = max(
                float(verified.get('priority_score', 0.0) or 0.0),
                (1000.0 if verified['scale_qualified'] else 0.0) + grain_area,
            )
            verified_claims.append(verified)

        verified_claims = sorted(
            verified_claims,
            key=lambda item: (
                -float(item.get('priority_score', 0.0) or 0.0),
                str(item.get('claim_id', '')),
                str(item.get('agent_id', '')),
            )
        )
        rejected_claims = sorted(
            rejected_claims,
            key=lambda item: (
                str(item.get('claim_id', '')),
                str(item.get('agent_id', '')),
            )
        )
        self.subsidy_claims_submitted = []
        self.verified_subsidy_queue.extend(verified_claims)
        if self.current_subsidy_runtime:
            self.current_subsidy_runtime['claims_verified'].extend(
                self._sanitize_runtime_payload(item) for item in verified_claims
            )
            self.current_subsidy_runtime['claims_rejected'].extend(
                self._sanitize_runtime_payload(item) for item in rejected_claims
            )
            self.current_subsidy_runtime['queued_payments'].extend(
                self._sanitize_runtime_payload(item) for item in verified_claims
            )
        return verified_claims

    def _collect_payable_verified_claims(self):
        """Split the verified queue into payable-now and deferred claims."""
        current_step = self._get_current_step()
        payable_claims = []
        deferred_claims = []
        for claim in self.verified_subsidy_queue:
            if int(claim.get('payable_step', current_step)) <= current_step:
                payable_claims.append(claim)
            else:
                deferred_claims.append(claim)
        self.verified_subsidy_queue = deferred_claims
        return payable_claims

    def _apply_verified_payment_to_agent(self, item, cash_amount, scale_cash, in_kind_value, in_kind_breakdown):
        """Apply a verified payment to the recipient's accounting state."""
        agent = item.get('agent')
        if agent is None:
            return

        if hasattr(agent, 'apply_verified_subsidy_payment'):
            agent.apply_verified_subsidy_payment(
                cash_amount=cash_amount,
                basic_cash=max(0.0, cash_amount - scale_cash),
                scale_cash=scale_cash,
                in_kind_value=in_kind_value,
                in_kind_breakdown=in_kind_breakdown,
                payment_record=item,
            )
            return

        if hasattr(agent, 'subsidy_income'):
            agent.subsidy_income = cash_amount
        elif hasattr(agent, 'subsidy_revenue'):
            agent.subsidy_revenue = cash_amount
        if in_kind_value > 0 and hasattr(agent, 'receive_in_kind_support'):
            agent.receive_in_kind_support(in_kind_value, in_kind_breakdown)

    def distribute_subsidies(self):
        """
        @theory
        阶段五：补贴发放
        根据政府设定的subsidy_target精确发放补贴，支持时滞和实物补贴
        
        @new 2025-11-30: 添加财政预算约束
        - 如果启用hard_cap，补贴总额不能超过年度预算
        - 超出预算时按比例缩减每个主体的补贴
        """
        if getattr(self, 'p3_batch', None) is not None:
            from core.p3_project_payments import pay
            pay(self)
            return
        self.total_subsidy_paid = 0
        self.subsidy_records = []

        active_policy = self.model.environment.active_policy or {}
        subsidy_target_str = active_policy.get('subsidy_target', 'operator')
        enable_priority = active_policy.get('enable_subsidy_priority', True)
        pending_subsidies = self._collect_payable_verified_claims()
        total_planned = sum(float(item.get('planned_amount', 0.0) or 0.0) for item in pending_subsidies)

        if enable_priority and pending_subsidies:
            pending_subsidies = sorted(
                pending_subsidies,
                key=lambda item: (
                    -float(item.get('priority_score', 0.0) or 0.0),
                    str(item.get('claim_id', '')),
                    str(item.get('agent_id', '')),
                ),
            )
        
        # ========== 第二阶段：先清偿历史拖欠 ==========
        self.arrears_paid_this_year = {}
        arrears_payment_budget = 0
        
        if self.enable_arrears and self.total_arrears_outstanding > 0:
            # 用当期预算的一部分清偿历史拖欠
            arrears_payment_budget = min(
                self.remaining_budget * self.arrears_clearance_rate,
                self.total_arrears_outstanding
            )
            
            if arrears_payment_budget > 0:
                # 按比例清偿各主体的拖欠
                for agent_id, arrears_amount in list(self.subsidy_arrears.items()):
                    if arrears_amount > 0:
                        payment_ratio = arrears_payment_budget / self.total_arrears_outstanding
                        payment = arrears_amount * payment_ratio
                        self.arrears_paid_this_year[agent_id] = payment
                        self.subsidy_arrears[agent_id] -= payment
                        
                        agent = self._find_agent_by_id(agent_id)
                        if agent:
                            agent.arrears_received = payment
                
                self.total_arrears_outstanding -= arrears_payment_budget
                self.remaining_budget -= arrears_payment_budget
                print(f"  [拖欠清偿] 清偿历史拖欠 {arrears_payment_budget/10000:.1f}万元，"
                      f"剩余未清偿 {self.total_arrears_outstanding/10000:.1f}万元")
        
        # ========== 第三阶段：计算当期补贴发放方案 ==========
        scaling_factor = 1.0
        arrears_factor = 0.0  # 拖欠比例
        budget_exceeded = False
        
        if total_planned > self.remaining_budget and self.remaining_budget > 0:
            if self.enable_arrears:
                # 启用拖欠机制：先按预算发放，超出部分记为拖欠
                if total_planned > 0:
                    # 计算可发放比例
                    scaling_factor = min(1.0, self.remaining_budget / total_planned)
                    # 超出部分记为拖欠，但不超过最大拖欠比例
                    arrears_factor = min(1.0 - scaling_factor, self.max_arrears_ratio)
                    scaling_factor = max(scaling_factor, 1.0 - self.max_arrears_ratio)
                budget_exceeded = True
                print(f"  [财政约束-拖欠机制] 计划 {total_planned/10000:.1f}万元，"
                      f"预算 {self.remaining_budget/10000:.1f}万元，"
                      f"发放 {scaling_factor*100:.0f}%，拖欠 {arrears_factor*100:.0f}%")
            elif self.hard_cap:
                # 硬性约束：直接按比例缩减
                scaling_factor = self.remaining_budget / total_planned
                budget_exceeded = True
                print(f"  [财政约束] 计划补贴 {total_planned/10000:.1f}万元 > 预算 {self.remaining_budget/10000:.1f}万元，"
                      f"按 {scaling_factor*100:.1f}% 比例发放")
        
        # ========== 第四阶段：发放补贴并记录拖欠 ==========
        self.current_year_arrears = {}
        
        for item in pending_subsidies:
            agent = item.get('agent')
            if agent is None:
                continue

            region = item.get('region', 'global')
            basic_cash = float(item.get('basic_cash', 0.0) or 0.0)
            scale_cash = float(item.get('scale_cash', 0.0) or 0.0)
            planned_cash = basic_cash + scale_cash
            planned_in_kind = float(item.get('in_kind_value', 0.0) or 0.0)

            actual_cash = planned_cash * scaling_factor
            actual_in_kind_value = planned_in_kind * scaling_factor
            actual_scale_cash = scale_cash * scaling_factor
            actual_basic_cash = max(0.0, actual_cash - actual_scale_cash)

            arrears_amount = (planned_cash + planned_in_kind) * arrears_factor
            if arrears_amount > 0:
                agent_id = agent.unique_id
                self.current_year_arrears[agent_id] = arrears_amount
                self.subsidy_arrears[agent_id] = self.subsidy_arrears.get(agent_id, 0) + arrears_amount
                self.total_arrears_outstanding += arrears_amount

            scaled_in_kind = {
                key: value * scaling_factor
                for key, value in dict(item.get('in_kind_breakdown', {}) or {}).items()
            }
            self._apply_verified_payment_to_agent(
                item,
                cash_amount=actual_cash,
                scale_cash=actual_scale_cash,
                in_kind_value=actual_in_kind_value,
                in_kind_breakdown=scaled_in_kind,
            )
            agent.subsidy_arrears_owed = self.subsidy_arrears.get(agent.unique_id, 0)

            record = {
                'recipient': agent.unique_id,
                'claim_id': item.get('claim_id'),
                'claim_step': item.get('claim_step'),
                'verified_step': item.get('verified_step'),
                'payable_step': item.get('payable_step'),
                'amount': actual_cash,
                'basic_cash': actual_basic_cash,
                'scale_cash': actual_scale_cash,
                'in_kind_value': actual_in_kind_value,
                'planned_amount': item.get('planned_amount', 0.0),
                'scaling_factor': scaling_factor,
                'arrears_amount': arrears_amount,
                'type': f"{item.get('actor_type', 'farmer')}_{'owner' if subsidy_target_str == 'owner' else 'operator'}",
                'region': region,
                'regional_policy': item.get('regional_policy', self._get_region_scenario_id(region)),
                'verification_path': item.get('verification_path'),
                'village_review_status': item.get('village_review_status'),
                'village_reviewed_area': item.get('village_reviewed_area', 0.0),
                'village_reviewer_ids': list(item.get('village_reviewer_ids', []) or []),
            }
            self.total_subsidy_paid += actual_cash + actual_in_kind_value
            self.subsidy_records.append(record)
            if self.current_subsidy_runtime:
                self.current_subsidy_runtime['payments_made'].append(
                    self._sanitize_runtime_payload(record)
                )
        
        # 更新剩余预算
        self.remaining_budget -= self.total_subsidy_paid
        
        if budget_exceeded:
            print(f"  [补贴发放] 实际发放 {self.total_subsidy_paid/10000:.1f}万元，"
                  f"新增拖欠 {sum(self.current_year_arrears.values())/10000:.1f}万元，"
                  f"剩余预算 {self.remaining_budget/10000:.1f}万元")

    def calculate_subsidy_for_farmer(self, farmer):
        """
        @theory
        计算农户补贴金额
        """
        # 根据农户拥有的土地面积计算补贴
        owned_area = sum(patch.area for patch in farmer.land_patches if patch.owner == farmer)
        
        # 基础补贴
        base_subsidy = owned_area * self.subsidy_rate
        
        # 粮食补贴（如果种植粮食作物）
        grain_subsidy = 0
        if hasattr(farmer, 'planned_crop') and farmer.planned_crop:
            if farmer.planned_crop.name == 'GRAIN':
                grain_subsidy = owned_area * self.grain_subsidies_rate
        
        return base_subsidy + grain_subsidy

    def calculate_subsidy_by_area(self, area):
        """
        @theory
        根据面积计算补贴金额
        """
        return area * self.subsidy_rate

    def evaluate_and_adjust_policy(self):
        """
        @theory
        阶段六：政策评估与调整
        根据市场表现调整政策参数
        """
        # 分析市场表现
        market_performance = self.analyze_market_performance()

        if not self._is_adaptive_adjustment_enabled():
            self.price_history.append({
                'step': self._get_current_step(),
                'grain_price_floor': self.grain_price_floor,
                'total_subsidy_paid': self.total_subsidy_paid,
                'adjustment_mode': 'fixed_policy_no_adaptive_adjustment',
            })
            self.policy_history.append({
                'step': self._get_current_step(),
                'grain_price_floor': self.grain_price_floor,
                'subsidy_rate': self.subsidy_rate,
                'grain_subsidies_rate': self.grain_subsidies_rate,
                'adjustment_mode': 'fixed_policy_no_adaptive_adjustment',
            })
            self._sync_policy_with_environment()
            self._record_policy_dashboard(market_performance)
            return
        
        # @new 2025-12-28: P4 - 强化学习介入
        if getattr(self, 'rl_enabled', False):
            # @new 2026-05-02: 检查是否启用扩展 RL（8 维状态、12 个动作）
            try:
                use_extended = (
                    self.model.config.get('mechanism_switches', {}) or {}
                ).get('use_extended_government_rl', False)
            except Exception:
                use_extended = False

            if use_extended:
                # 扩展 RL 路径：把资本化、流转、老龄化、生态压力纳入状态空间
                current_state = self._get_rl_state_extended(market_performance)
                reward = self._calculate_rl_reward(market_performance)
                self._update_q_table_extended(reward, current_state)
                action = self._choose_action_rl_extended(current_state)
                self._apply_rl_action_extended(action)
                rl_mode = 'RL_Q_Learning_Extended'
            else:
                # 原始 RL 路径（3 维状态、5 个动作）
                current_state = self._get_rl_state(market_performance)
                reward = self._calculate_rl_reward(market_performance)
                self._update_q_table(reward, current_state)
                action = self._choose_action_rl(current_state)
                self._apply_rl_action(action)
                rl_mode = 'RL_Q_Learning'

            # 5. 更新记录
            self.last_state = current_state
            self.last_action = action

            # 记录调整（标记为RL模式）
            self._record_policy_adjustment(market_performance, {'mode': rl_mode, 'action': action, 'reward': reward})
            self._sync_policy_with_environment()

            # 记录数据
            self.price_history.append({
                'step': self._get_current_step(),
                'grain_price_floor': self.grain_price_floor,
                'total_subsidy_paid': self.total_subsidy_paid,
                'rl_reward': reward,
                'rl_mode': rl_mode,
            })
            self._record_policy_dashboard(market_performance)
            return

        # 原有启发式规则（当RL未启用时执行）
        # 根据市场表现调整政策
        self.adjust_policy_parameters(market_performance)
        
        # 记录价格历史
        self.price_history.append({
            'step': self._get_current_step(),
            'grain_price_floor': self.grain_price_floor,
            # @fix: market_price_cash_crop 由环境决定，此处不再重复记录
            # 'market_price_cash_crop': self.market_price_cash_crop,
            'total_subsidy_paid': self.total_subsidy_paid
        })
        self._record_policy_dashboard(market_performance)


    def _is_adaptive_adjustment_enabled(self):
        """Return whether heuristic/RL policy adjustment is active for this run."""
        policy_section = self.model.config.get('policy', {}) if hasattr(self.model, 'config') else {}
        gov_cfg = policy_section.get('government', {}) if isinstance(policy_section, dict) else {}
        if isinstance(gov_cfg, dict) and 'adaptive_adjustment_enabled' in gov_cfg:
            return bool(gov_cfg.get('adaptive_adjustment_enabled'))
        if isinstance(policy_section, dict) and 'adaptive_adjustment_enabled' in policy_section:
            return bool(policy_section.get('adaptive_adjustment_enabled'))
        return bool(getattr(self, 'adaptive_adjustment_enabled', True))


    def analyze_market_performance(self):
        """
        @theory
        分析市场表现
        """
        farmers = list(getattr(self.model, 'farmers', []))
        net_incomes = [get_farmer_net_income(farmer) for farmer in farmers]
        gross_incomes = [get_farmer_gross_income(farmer) for farmer in farmers]
        avg_farmer_income = sum(net_incomes) / len(net_incomes) if net_incomes else 0
        avg_farmer_gross_income = sum(gross_incomes) / len(gross_incomes) if gross_incomes else 0

        pure_farmer_net_incomes = [
            get_farmer_net_income(farmer)
            for farmer in farmers
            if is_pure_farmer(farmer)
        ]
        retention_cfg = (
            self.model.config.get('agents', {})
            .get('farmer', {})
            .get('livelihood_evolution', {})
            .get('professional_farmer_retention', {})
        )
        large_operator_threshold = float(retention_cfg.get('min_operated_area', 20) or 20)
        min_ag_ratio = float(retention_cfg.get('min_agricultural_income_ratio', 0.78) or 0.78)
        max_off_farm_income = float(retention_cfg.get('max_off_farm_income', 12000) or 12000)

        professional_large_operator_net_incomes = [
            get_farmer_net_income(farmer)
            for farmer in farmers
            if is_professional_large_operator(
                farmer,
                min_operated_area=large_operator_threshold,
                min_agricultural_income_ratio=min_ag_ratio,
                max_off_farm_income=max_off_farm_income,
            )
        ]
        all_large_operator_net_incomes = [
            get_farmer_net_income(farmer)
            for farmer in farmers
            if is_large_operator(farmer, min_operated_area=large_operator_threshold)
        ]
        
        # 计算企业平均利润
        enterprise_profits = [enterprise.profit for enterprise in self.model.agricultural_enterprises]
        avg_enterprise_profit = sum(enterprise_profits) / len(enterprise_profits) if enterprise_profits else 0
        
        # 计算土地流转率（按面积）
        total_land_area = sum(patch.area for patch in self.model.land_patches)
        leased_land_area = sum(patch.area for patch in self.model.land_patches if patch.leased_to is not None)
        land_transfer_rate = leased_land_area / total_land_area if total_land_area > 0 else 0

        # 计算粮食总产量
        # 统一使用主体的“实际粮食产量”口径，避免政府用主导作物标签或缺失字段
        # 低估企业粮食供给，进而错误上调托市和补贴。
        grain_output = 0.0
        for farmer in self.model.farmers:
            farmer_grain_output = getattr(farmer, 'grain_production', None)
            if farmer_grain_output is None:
                if getattr(farmer, 'last_crop_choice', None) and getattr(farmer.last_crop_choice, 'value', '') == "Grain":
                    farmer_grain_output = getattr(farmer, 'last_yield', 0.0)
                else:
                    farmer_grain_output = 0.0
            grain_output += float(farmer_grain_output or 0.0)
        for enterprise in self.model.agricultural_enterprises:
            enterprise_grain_output = getattr(enterprise, 'grain_production', None)
            if enterprise_grain_output is None:
                last_yields = getattr(enterprise, 'last_yields', None)
                if isinstance(last_yields, dict):
                    enterprise_grain_output = last_yields.get('grain', 0.0)
                else:
                    enterprise_grain_output = getattr(enterprise, 'last_yield', 0.0)
            grain_output += float(enterprise_grain_output or 0.0)

        grain_area = 0.0
        for farmer in farmers:
            area_value = getattr(farmer, 'grain_area', None)
            if area_value is None:
                if getattr(farmer, 'last_crop_choice', None) and getattr(farmer.last_crop_choice, 'value', '') == "Grain":
                    area_value = getattr(farmer, 'last_operated_area', getattr(farmer, 'get_operated_land_area', lambda: 0.0)())
                else:
                    area_value = 0.0
            grain_area += float(area_value or 0.0)
        for enterprise in self.model.agricultural_enterprises:
            grain_area += float(getattr(enterprise, 'grain_area', 0.0) or 0.0)
        grain_yield_per_mu = grain_output / max(grain_area, 1.0) if grain_area > 0 else 0.0

        reserve_storage_ratio = 0.0
        reserve_intervention_intensity = 0.0
        if hasattr(self.model, 'grain_reserve_bureau') and self.model.grain_reserve_bureau is not None:
            try:
                reserve_stats = self.model.grain_reserve_bureau.get_statistics()
                reserve_storage_ratio = float(reserve_stats.get('storage_ratio', 0.0) or 0.0)
            except Exception:
                reserve_storage_ratio = 0.0
            try:
                reserve_intervention_intensity = float(
                    self.model.grain_reserve_bureau.get_market_intervention_intensity() or 0.0
                )
            except Exception:
                reserve_intervention_intensity = 0.0
        
        return {
            'avg_farmer_income': avg_farmer_income,
            'avg_farmer_net_income': avg_farmer_income,
            'avg_farmer_gross_income': avg_farmer_gross_income,
            'all_household_net_income': avg_farmer_income,
            'all_household_gross_income': avg_farmer_gross_income,
            'pure_farmer_net_income': (
                sum(pure_farmer_net_incomes) / len(pure_farmer_net_incomes)
                if pure_farmer_net_incomes else 0
            ),
            'professional_large_operator_net_income': (
                sum(professional_large_operator_net_incomes) / len(professional_large_operator_net_incomes)
                if professional_large_operator_net_incomes else 0
            ),
            'all_large_operator_net_income': (
                sum(all_large_operator_net_incomes) / len(all_large_operator_net_incomes)
                if all_large_operator_net_incomes else 0
            ),
            'large_operator_net_income': (
                sum(professional_large_operator_net_incomes) / len(professional_large_operator_net_incomes)
                if professional_large_operator_net_incomes else 0
            ),
            'professional_large_operator_count': len(professional_large_operator_net_incomes),
            'all_large_operator_count': len(all_large_operator_net_incomes),
            'avg_enterprise_profit': avg_enterprise_profit,
            'land_transfer_rate': land_transfer_rate,
            'total_subsidy_paid': self.total_subsidy_paid,
            'grain_output': grain_output,
            'grain_area': grain_area,
            'grain_yield_per_mu': grain_yield_per_mu,
            'grain_reserve_storage_ratio': reserve_storage_ratio,
            'grain_intervention_intensity': reserve_intervention_intensity,
            # @new 2026-05-02: 扩展 RL 所需的结构性状态变量
            'capitalization_rate': self._compute_capitalization_rate_for_rl(),
            'aging_index': self._compute_aging_index_for_rl(),
            'ecological_pressure': self._compute_ecological_pressure_for_rl(),
        }

    def adjust_policy_parameters(self, market_performance):
        """
        @theory
        根据市场表现调整政策参数
        
        @enhanced 2025-12-22: 添加政策路径依赖性
        
        政策调整遵循路径依赖原则：
        1. 政策惯性：政策变化幅度受历史约束，不能突然大幅调整
        2. 学习效应：根据历史政策效果调整策略
        3. 制度刚性：某些政策一旦实施难以取消（如补贴）
        
        理论依据：
        - North (1990): 制度变迁的路径依赖
        - Pierson (2000): 正反馈与政策锁定
        - 中国农业政策实践：托市价连续上调难下调
        """
        active_policy = dict(getattr(self.model.environment, 'active_policy', {}) or {})
        scenario_id = getattr(self, 'active_scenario_id', None) or getattr(self.model, 'active_policy_id', None)
        disable_grain_intervention = self._policy_disables_grain_intervention(
            policy_state=active_policy,
            scenario_id=scenario_id,
        )
        if disable_grain_intervention:
            self.grain_price_floor = 0.0
            self.grain_subsidies_rate = 0.0
            self.subsidy_rate = 0.0
            self._sync_policy_with_environment()
            return

        effective_targets = self._get_effective_policy_targets(market_performance)
        grain_target = effective_targets['grain_output']
        income_target = effective_targets['avg_income']
        subsidy_budget = effective_targets['subsidy_budget']

        grain_score = market_performance['grain_output'] / grain_target
        income_score = market_performance['avg_farmer_income'] / income_target
        fiscal_ratio = market_performance['total_subsidy_paid'] / subsidy_budget
        reserve_pressure = max(
            float(market_performance.get('grain_reserve_storage_ratio', 0.0) or 0.0),
            float(market_performance.get('grain_intervention_intensity', 0.0) or 0.0),
        )
        
        # @new 2025-12-22: 计算政策变化的路径依赖约束
        policy_inertia = self._calculate_policy_inertia()
        adjustment_constraint = self._get_adjustment_constraint()

        # 粮食安全优先：低于目标则提高托市价
        # @enhanced 2025-12-22: 考虑政策惯性和历史效果
        if grain_score < 0.95:
            # 当库存/干预强度已经很高时，低产量更可能是配置效率或单产问题，
            # 继续抬托市价只会进一步抬租金、抬预期，而不是修复真实供给能力。
            if reserve_pressure > 0.88 and fiscal_ratio > 0.85:
                effective_step = max(
                    self.price_floor_step * 0.35,
                    self.price_floor_step * reserve_pressure * 0.45,
                )
                self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor - effective_step)
            else:
                # 提价容易：政策惯性小
                effective_step = self.price_floor_step * (1.0 + policy_inertia['upward_bias'])
                effective_step *= max(0.20, 1.0 - 0.85 * reserve_pressure)
                self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor + effective_step)
        elif grain_score > 1.05 and (fiscal_ratio > 1.0 or reserve_pressure > 0.65):
            # 降价难：政策惯性大（利益固化）
            effective_step = self.price_floor_step * (1.0 - policy_inertia['downward_resistance'])
            effective_step = max(effective_step, self.price_floor_step * 0.3)  # 最低30%调整幅度
            self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor - effective_step)
        elif reserve_pressure > 0.75 and grain_score >= 0.95:
            effective_step = max(self.price_floor_step * 0.35, self.price_floor_step * reserve_pressure * 0.55)
            self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor - effective_step)

        # 农民福利：低于目标则提高补贴
        # @enhanced 2025-12-22: 补贴增加容易，削减难（政治经济学约束）
        if income_score < 0.95:
            effective_step = self.subsidy_step * (1.0 + policy_inertia['upward_bias'])
            self.subsidy_rate = min(self.subsidy_rate + effective_step, 0.3)
        elif income_score > 1.1 and fiscal_ratio > 1.0:
            # 削减补贴面临强烈抵制
            effective_step = self.subsidy_step * (1.0 - policy_inertia['subsidy_stickiness'])
            effective_step = max(effective_step, self.subsidy_step * 0.2)  # 最低20%调整幅度
            self.subsidy_rate = max(self.subsidy_rate - effective_step, 0.05)

        # 土地流转效率仍作为辅助调整
        land_rate = market_performance['land_transfer_rate']
        pure_farmer_income = market_performance.get('pure_farmer_net_income', 0.0)
        large_operator_income = market_performance.get(
            'professional_large_operator_net_income',
            market_performance.get('large_operator_net_income', 0.0),
        )
        professional_core_income = min(pure_farmer_income, large_operator_income)
        professional_core_stable = professional_core_income >= income_target * 0.9
        if land_rate < 0.3:
            self.subsidy_rate = min(self.subsidy_rate + self.rent_incentive_step, 0.35)
        elif land_rate > 0.7:
            # 高流转不等于可以机械退补。只有职业化核心群体也稳住时，才允许温和减补。
            if grain_score > 1.02 and income_score > 1.05 and professional_core_stable:
                self.subsidy_rate = max(self.subsidy_rate - self.rent_incentive_step * 0.5, 0.08)
            else:
                self.subsidy_rate = max(self.subsidy_rate, 0.12)

        # @fix 2024-11-19: 不要覆盖grain_subsidies_rate，它应该保持政策配置值（如1.0）
        # 原因：自适应调整的subsidy_rate（约0.15-0.35）会导致企业补贴从403元降到125元
        # 解决：grain_subsidies_rate由政策场景控制，不参与自适应调整
        # self.grain_subsidies_rate = self.subsidy_rate  # ❌ 已禁用
        
        # @new 2025-12-22: 记录政策调整历史（用于路径依赖计算）
        self._record_policy_adjustment(market_performance, policy_inertia)
        
        self._sync_policy_with_environment()
    
    def _calculate_policy_inertia(self):
        """
        @new 2025-12-22: 计算政策惯性系数
        
        政策惯性来源：
        1. 历史政策持续时间：越久惯性越大
        2. 利益固化程度：受益群体越多惯性越大
        3. 政策效果评价：效果好的政策惯性大
        
        Returns:
            dict: 包含各类惯性系数
        """
        # 默认惯性系数
        inertia = {
            'upward_bias': 0.2,        # 向上调整偏好（提价/增补贴容易）
            'downward_resistance': 0.5, # 向下调整阻力（降价/减补贴难）
            'subsidy_stickiness': 0.6,  # 补贴粘性（补贴一旦给予难以取消）
            'policy_lock_in': 0.3      # 政策锁定程度
        }
        
        # 根据历史政策调整惯性
        if len(self.policy_history) >= 3:
            recent_policies = self.policy_history[-3:]
            
            # 计算政策变化趋势
            price_changes = []
            subsidy_changes = []
            for i in range(1, len(recent_policies)):
                price_change = recent_policies[i].get('grain_price_floor', BASELINE_GRAIN_PRICE_FLOOR) - \
                              recent_policies[i-1].get('grain_price_floor', BASELINE_GRAIN_PRICE_FLOOR)
                subsidy_change = recent_policies[i].get('subsidy_rate', LEGACY_SUBSIDY_RATE) - \
                                recent_policies[i-1].get('subsidy_rate', LEGACY_SUBSIDY_RATE)
                price_changes.append(price_change)
                subsidy_changes.append(subsidy_change)
            
            # 如果历史上持续提价/加补贴，则进一步增加向上偏好
            if all(c >= 0 for c in price_changes):
                inertia['upward_bias'] += 0.1
            
            # 如果历史上从未削减补贴，增加补贴粘性
            if all(c >= 0 for c in subsidy_changes):
                inertia['subsidy_stickiness'] += 0.15
        
        return inertia
    
    def _get_adjustment_constraint(self):
        """
        @new 2025-12-22: 获取政策调整约束
        
        约束来源：
        1. 年度调整幅度上限（政策稳定性要求）
        2. 累计变化幅度限制（制度刚性）
        3. 财政约束传导
        """
        return {
            'max_annual_price_change': 0.2,  # 年度价格最大变化20%
            'max_annual_subsidy_change': 0.05,  # 年度补贴率最大变化5个百分点
            'fiscal_constrained': self.fiscal_cliff_active  # 财政约束激活
        }
    
    def _record_policy_adjustment(self, market_performance, policy_inertia):
        """
        @new 2025-12-22: 记录政策调整用于路径依赖分析
        """
        effective_targets = self._get_effective_policy_targets(market_performance)
        self.policy_history.append({
            'step': getattr(self.model, 'step_count', 0),
            'grain_price_floor': self.grain_price_floor,
            'subsidy_rate': self.subsidy_rate,
            'grain_subsidies_rate': self.grain_subsidies_rate,
            'effective_grain_target': effective_targets['grain_output'],
            'effective_subsidy_budget': effective_targets['subsidy_budget'],
            'market_grain_score': market_performance.get('grain_output', 0) / effective_targets['grain_output'],
            'market_income_score': market_performance.get('avg_farmer_income', 0) / effective_targets['avg_income'],
            'policy_inertia': policy_inertia.copy(),
            'fiscal_ratio': market_performance.get('total_subsidy_paid', 0) / effective_targets['subsidy_budget'],
            'reserve_pressure': max(
                float(market_performance.get('grain_reserve_storage_ratio', 0.0) or 0.0),
                float(market_performance.get('grain_intervention_intensity', 0.0) or 0.0),
            ),
            'grain_yield_per_mu': float(market_performance.get('grain_yield_per_mu', 0.0) or 0.0),
        })
        
        # @new 2025-12-28: 政策学习周期
        self._policy_learning_cycle()
    
    def _policy_learning_cycle(self):
        """
        @new 2025-12-28: 政府政策学习周期机制
        
        @literature:
        - Lindblom (1959): 渐进主义，政策小步调整
        - Hall (1993): 政策范式转换，三阶学习
        - 中国农业政策实践：五年规划周期
        
        机制：
        1. 短期学习（年度）：微调参数，响应市场波动
        2. 中期学习（3-5年）：评估政策效果，调整目标
        3. 长期学习（10年+）：反思政策范式，可能发生根本性变革
        
        特点：
        - 滞后性：政策效果需要时间观察
        - 保守性：倾向于渐进调整而非激进变革
        - 路径依赖：历史政策影响未来选择
        """
        current_step = getattr(self.model, 'step_count', 0)
        
        # 初始化政策学习状态
        if not hasattr(self, 'last_major_review_step'):
            self.last_major_review_step = 0
            self.policy_learning_memory = []
            self.policy_effectiveness_score = 0.5  # 初始中等
        
        # === 短期学习：每年进行（已在adjust_policy_parameters中实现） ===
        
        # === 中期学习：每5年进行重大评估 ===
        review_cycle = 5  # 五年规划周期
        if current_step - self.last_major_review_step >= review_cycle:
            self._conduct_major_policy_review()
            self.last_major_review_step = current_step
    
    def _conduct_major_policy_review(self):
        """
        @new 2025-12-28: 重大政策评估与学习
        
        基于过去5年的政策效果进行综合评估：
        1. 计算政策有效性得分
        2. 根据得分调整政策目标和策略
        3. 可能触发政策范式微调
        """
        import numpy as np
        
        if len(self.policy_history) < 3:
            return  # 历史数据不足
        
        # 取最近5期数据计算政策效果
        recent_history = self.policy_history[-5:]
        
        # === 1. 计算各维度效果 ===
        avg_grain_score = np.mean([h.get('market_grain_score', 0.5) for h in recent_history])
        avg_income_score = np.mean([h.get('market_income_score', 0.5) for h in recent_history])
        avg_fiscal_ratio = np.mean([h.get('fiscal_ratio', 1.0) for h in recent_history])
        avg_reserve_pressure = np.mean([h.get('reserve_pressure', 0.0) for h in recent_history])
        
        # === 2. 综合评分（加权） ===
        # 权重基于政策目标优先级
        self.policy_effectiveness_score = (
            avg_grain_score * 0.4 +      # 粮食安全权重40%
            avg_income_score * 0.35 +    # 农民收入权重35%
            (1 - min(avg_fiscal_ratio, 1.5) / 1.5) * 0.18 +  # 财政可持续权重18%
            (1 - min(avg_reserve_pressure, 1.0)) * 0.07       # 收储压力权重7%
        )
        
        # === 3. 基于学习的策略调整 ===
        # Hall (1993): 三阶学习
        # 一阶：调整政策工具参数（已在年度调整中实现）
        # 二阶：调整政策目标
        # 三阶：改变政策范式（罕见）
        
        # 二阶学习：调整目标
        if avg_grain_score < 0.8:
            # 粮食安全目标未达成→提高投入
            self.price_floor_step *= 1.2  # 放大调整幅度
        elif avg_grain_score > 1.2:
            # 粮食过剩→可适度下调目标
            self.price_floor_step *= 0.9
        
        if avg_fiscal_ratio > 1.3:
            # 财政压力大→提高财政约束意识
            self.subsidy_step *= 0.8  # 减小调整幅度
        
        # === 4. 记录学习结果 ===
        if not hasattr(self, 'policy_learning_log'):
            self.policy_learning_log = []
        self.policy_learning_log.append({
            'step': getattr(self.model, 'step_count', 0),
            'effectiveness_score': self.policy_effectiveness_score,
            'grain_score': avg_grain_score,
            'income_score': avg_income_score,
            'fiscal_ratio': avg_fiscal_ratio,
            'learning_action': 'major_review'
        })
        if len(self.policy_learning_log) > 10:
            self.policy_learning_log = self.policy_learning_log[-10:]

    def _sync_policy_with_environment(self):
        """将调整后的政策参数写回环境，确保后续主体读取的是最新政策。"""
        active_policy = dict(self.model.environment.active_policy or {})
        scenario_id = getattr(self, 'active_scenario_id', None) or getattr(self.model, 'active_policy_id', None)
        disable_grain_intervention = self._policy_disables_grain_intervention(
            policy_state=active_policy,
            scenario_id=scenario_id,
        )
        self.grain_price_floor = self._clamp_grain_price_floor(
            self.grain_price_floor,
            allow_zero=disable_grain_intervention,
        )
        if disable_grain_intervention:
            self.grain_price_floor = 0.0
            self.grain_subsidies_rate = 0.0
            self.subsidy_rate = 0.0
        active_policy['grain_price_floor'] = self.grain_price_floor
        active_policy['grain_subsidies_rate'] = self.grain_subsidies_rate
        active_policy['base_subsidy_per_mu'] = 0.0 if disable_grain_intervention else active_policy.get('base_subsidy_per_mu', 0)
        active_policy['subsidy_type'] = self.subsidy_type.value
        active_policy['subsidy_target'] = self.subsidy_target.value
        active_policy['subsidy_rate'] = self.grain_subsidies_rate
        active_policy['disable_grain_intervention'] = disable_grain_intervention
        self.model.environment.set_active_policy(active_policy, getattr(self.model, 'active_policy_id', None))

    def get_subsidy_statistics(self):
        """
        @theory
        获取补贴统计信息
        """
        return {
            'total_subsidy_paid': self.total_subsidy_paid,
            'subsidy_records_count': len(self.subsidy_records),
            'current_policy': {
                'grain_price_floor': self.grain_price_floor,
                'grain_subsidies_rate': self.grain_subsidies_rate,
                # @fix: 移除 market_price_cash_crop
                # 'market_price_cash_crop': self.market_price_cash_crop,
                'subsidy_rate': self.subsidy_rate,
                'subsidy_type': self.subsidy_type.value,
                'subsidy_target': self.subsidy_target.value
            }
        } 

    def _record_policy_dashboard(self, market_performance):
        path = self.policy_dashboard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dashboard = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        except json.JSONDecodeError:
            dashboard = []
        effective_targets = self._get_effective_policy_targets(market_performance)
        active_policy = dict(self.model.environment.active_policy or {})
        scenario_id = getattr(self, 'active_scenario_id', None) or getattr(self.model, 'active_policy_id', None)
        disable_grain_intervention = self._policy_disables_grain_intervention(
            policy_state=active_policy,
            scenario_id=scenario_id,
        )
        market_cfg = (
            getattr(self.model, 'config', {})
            .get('environment', {})
            .get('market', {})
        )
        numeric_floor = float(market_cfg.get('market_only_numeric_floor', 0.0) or 0.0)
        entry = {
            'step': self._get_current_step(),
            'scenario_id': self.active_scenario_id,
            'regional_assignments': {region: info.get('scenario_id') for region, info in self.regional_policy_assignments.items()},
            'regional_summary': self.latest_regional_summary,
            'policy_state': {
                'grain_price_floor': self.grain_price_floor,
                'administrative_grain_price_floor': self.grain_price_floor,
                'numeric_stability_floor': numeric_floor,
                'effective_runtime_floor': numeric_floor if disable_grain_intervention else self.grain_price_floor,
                'disable_grain_intervention': disable_grain_intervention,
                'grain_subsidies_rate': self.grain_subsidies_rate,
                'subsidy_rate': self.subsidy_rate,
                'subsidy_target': self.subsidy_target.value,
                'effective_grain_target': effective_targets.get('grain_output'),
                'effective_income_target': effective_targets.get('avg_income'),
                'effective_subsidy_budget': effective_targets.get('subsidy_budget'),
            },
            'kpis': market_performance
        }
        dashboard.append(entry)
        path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding='utf-8')

    # ========== P4: Reinforcement Learning Methods ==========
    
    def _get_rl_state(self, performance):
        """
        @new 2025-12-28: 获取RL状态
        Discretize continuous variables into discrete states.
        """
        effective_targets = self._get_effective_policy_targets(performance)
        # 1. 粮食产量状态 (0:低, 1:适中, 2:高)
        grain_ratio = performance.get('grain_output', 0) / effective_targets['grain_output']
        if grain_ratio < 0.9: grain_state = 0
        elif grain_ratio < 1.1: grain_state = 1
        else: grain_state = 2
        
        # 2. 收入水平状态 (0:低, 1:适中, 2:高)
        income_ratio = performance.get('avg_farmer_income', 0) / effective_targets['avg_income']
        if income_ratio < 0.9: income_state = 0
        elif income_ratio < 1.1: income_state = 1
        else: income_state = 2
        
        # 3. 财政压力状态 (0:正常, 1:紧张)
        fiscal_ratio = performance.get('total_subsidy_paid', 0) / effective_targets['subsidy_budget']
        fiscal_state = 1 if fiscal_ratio > 1.0 else 0
        
        return (grain_state, income_state, fiscal_state)

    def _choose_action_rl(self, state, action_count=5):
        """
        @new 2025-12-28: 选择动作 (Epsilon-Greedy)
        @update 2026-01-03: 支持预训练后冻结Q-table
        
        Actions:
        0: Maintain (No Change)
        1: Increase Subsidy (+Step)
        2: Decrease Subsidy (-Step)
        3: Increase Price Floor (+Step)
        4: Decrease Price Floor (-Step)
        """
        if self.government_q_project_enabled() and self.q_table_frozen:
            if action_count != 3 or (state in self.q_table and len(self.q_table[state]) != 3):
                raise ValueError('Frozen government Q project requires exactly three actions')
            return self.select_action_frozen(state)

        # @new: 如果外部加载了冻结Q-table，使用纯greedy选择
        if getattr(self, '_frozen', False):
            return self.select_action_frozen(state)

        # Init Q-table for state if not exists
        if state not in self.q_table:
            self.q_table[state] = [0.0] * action_count
        elif len(self.q_table[state]) != action_count:
            raise ValueError('Q-table action count differs from the active government mode')
        
        # @new 2026-01-03: 预训练阶段 vs 执行阶段
        self.training_step_count += 1
        
        if self.q_table_frozen:
            # 冻结模式：纯利用（exploitation），不再探索
            return int(np.argmax(self.q_table[state]))
        
        # 预训练阶段：使用衰减的epsilon
        if self.training_step_count <= self.pretrain_steps:
            # 探索率随预训练进度衰减
            decay_factor = 1.0 - (self.training_step_count / self.pretrain_steps) * 0.8
            current_epsilon = self.epsilon * decay_factor
        else:
            # 超过预训练步数，使用较低探索率
            current_epsilon = 0.05
            
        # Exploration
        if np_random_value(self) < current_epsilon:
            return int(np_randint(self, 0, action_count))
            
        # Exploitation
        return int(np.argmax(self.q_table[state]))

    def _apply_rl_action(self, action):
        """执行RL动作"""
        if action == 0:
            pass # Maintain
        elif action == 1:
            self.subsidy_rate = min(self.subsidy_rate + self.subsidy_step, 0.3)
        elif action == 2:
            self.subsidy_rate = max(self.subsidy_rate - self.subsidy_step, 0.0)
        elif action == 3:
            self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor + self.price_floor_step)
        elif action == 4:
            self.grain_price_floor = self._clamp_grain_price_floor(self.grain_price_floor - self.price_floor_step)

    def _update_q_table(self, reward, next_state, done=False):
        """
        Q-Learning Update Rule
        @update 2026-01-03: 支持冻结模式
        """
        # @new 2026-01-03: 冻结后不再更新
        if self.q_table_frozen:
            return
            
        if self.last_state is not None and self.last_action is not None:
            # Init next state if needed
            if not done and next_state not in self.q_table:
                self.q_table[next_state] = [0.0] * len(self.q_table[self.last_state])
                
            old_q = self.q_table[self.last_state][self.last_action]
            max_future_q = 0.0 if done else np.max(self.q_table[next_state])
            
            # Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
            self.q_table[self.last_state][self.last_action] = new_q
    
    # ========== @new 2026-01-03: 预训练机制方法 ==========
    
    def freeze_q_table(self):
        """
        冻结Q-table，停止学习
        
        用途: 预训练完成后，固定策略进行长期评估
        """
        self.q_table_frozen = True
        self.pretrain_completed = True
        print(f"[RL] Q-table已冻结: {len(self.q_table)}个状态")
        return self.q_table.copy()
    
    def unfreeze_q_table(self):
        """解冻Q-table，恢复学习"""
        self.q_table_frozen = False
        print("[RL] Q-table已解冻，恢复学习")
    
    def load_pretrained_q_table(self, q_table_data: dict, freeze: bool = True):
        """
        Load a pre-trained Q-table for evaluation (pure exploitation mode).
        
        Args:
            q_table_data: Dict mapping state strings to action value lists
                          e.g. {"(0, 1, 2)": [0.12, -0.05, 0.08, 0.15, -0.02]}
            freeze: If True, set epsilon=0 and disable Q-table updates
        """
        self.q_table = {}
        for key_str, values in q_table_data.items():
            # Convert string key back to tuple
            try:
                key = eval(key_str)
            except:
                key = tuple(int(x.strip()) for x in key_str.strip('()').split(',') if x.strip())
            self.q_table[key] = values if isinstance(values, list) else list(values)
        
        if freeze:
            self.epsilon = 0.0
            self._frozen = True
            self.q_table_frozen = True
            self.pretrain_completed = True

        print(f"[GovernmentAgent] Loaded Q-table: {len(self.q_table)} states, frozen={freeze}")

    def select_action_frozen(self, state: tuple) -> int:
        """
        Select action in frozen mode (pure greedy, no exploration).
        Falls back to action 0 (maintain) for unseen states.
        
        Args:
            state: Current discretized state tuple

        Returns:
            Action index (0-4)
        """
        if state in self.q_table:
            return int(np.argmax(self.q_table[state]))
        return 0  # Default: maintain current policy
    
    def save_q_table(self, filepath):
        """保存Q-table到文件"""
        import json
        q_table_export = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'q_table': q_table_export,
                'training_steps': self.training_step_count,
                'pretrain_completed': self.pretrain_completed
            }, f, indent=2)
        print(f"[RL] Q-table保存至: {filepath}")
    
    def get_pretrain_status(self):
        """获取预训练状态"""
        return {
            'training_steps': self.training_step_count,
            'pretrain_steps': self.pretrain_steps,
            'pretrain_progress': min(1.0, self.training_step_count / self.pretrain_steps),
            'q_table_frozen': self.q_table_frozen,
            'n_states': len(self.q_table),
            'pretrain_completed': self.pretrain_completed
        }
            
    def _calculate_rl_reward(self, performance):
        """
        计算RL奖励函数
        基于多目标加权

        @fix 2026-06-24: grain_penalty 改为 grain_reward — 原版 -abs(1-grain)*2
        对标对称惩罚，RL学到"别偏离目标"而非"多产粮"。改为线性reward：
        grain>target → 正奖励，grain<target → 负惩罚。激励增产。
        """
        effective_targets = self._get_effective_policy_targets(performance)
        grain_score = performance.get('grain_output', 0) / max(effective_targets['grain_output'], 1.0)
        income_score = performance.get('avg_farmer_income', 0) / max(effective_targets['avg_income'], 1.0)
        fiscal_ratio = performance.get('total_subsidy_paid', 0) / max(effective_targets['subsidy_budget'], 1.0)

        # @fix: grain reward is now directional — positive for exceeding target
        grain_reward = (grain_score - 1.0) * 2.0  # >0 if above, <0 if below
        income_reward = min(income_score, 1.5) * 1.5
        fiscal_penalty = -max(0, fiscal_ratio - 1.0) * 3.0

        # @fix 2026-06-24: 优先使用 config.policy.rl_reward_weights
        try:
            rl_weights = (
                self.model.config.get("policy", {}).get("rl_reward_weights", {})
            )
        except Exception:
            rl_weights = {}
        w_grain = float(rl_weights.get("grain_security", self.objective_weights.get("grain_security", 0.40)))
        w_income = float(rl_weights.get("farmer_welfare", self.objective_weights.get("farmer_welfare", 0.35)))
        w_fiscal = float(rl_weights.get("fiscal_balance", self.objective_weights.get("fiscal_balance", 0.25)))

        total_reward = (
            grain_reward * w_grain +
            income_reward * w_income +
            fiscal_penalty * w_fiscal
        )
        return total_reward

    # ========================================================================
    # @new 2026-05-02: 扩展 RL（8 维状态空间 × 12 动作）
    # ========================================================================
    # 解决问题：原 RL 状态空间 (grain×3, income×3, fiscal×2) = 18 个状态
    # 不包含资本化率、流转率、老龄化、生态压力等关键结构性状态变量。
    # 扩展后 8 维状态空间 = 3×3×2×3×3×3×2 = 972 状态（仍可 Q-table 学习）。
    # 动作空间从 5 扩展到 12，新增结构性政策工具（流转摩擦税、补贴定向、规模门槛）。
    # 通过 mechanism_switches.use_extended_government_rl = True 启用。
    # ========================================================================

    # 扩展动作集
    EXTENDED_ACTIONS = [
        'maintain',                    # 0
        'subsidy_increase',            # 1
        'subsidy_decrease',            # 2
        'price_floor_increase',        # 3
        'price_floor_decrease',        # 4
        'tighten_scale_threshold',     # 5: 收紧规模补贴门槛（抑制资本化）
        'loosen_scale_threshold',      # 6: 放宽规模补贴门槛
        'increase_transfer_friction',  # 7: 增加流转摩擦税
        'decrease_transfer_friction',  # 8: 减少流转摩擦税
        'subsidy_target_grain_only',   # 9: 提高补贴目标精度
        'subsidy_target_operator',     # 10: 通用补贴
        'emergency_intervention',      # 11: 紧急维稳介入
    ]

    def _get_rl_state_extended(self, performance):
        """
        扩展 8 维状态空间：粮食×收入×财政×资本化×流转×老龄化×生态。

        Returns:
            7-tuple of int 状态。
        """
        # 复用原 3 维状态
        base = self._get_rl_state(performance)  # (grain, income, fiscal)

        # 资本化率（3 档）
        cap_rate = float(performance.get('capitalization_rate', 0.0) or 0.0)
        if cap_rate < 0.2:
            cap_state = 0
        elif cap_rate < 0.4:
            cap_state = 1
        else:
            cap_state = 2

        # 流转率（3 档）
        transfer_rate = float(performance.get('land_transfer_rate', 0.0) or 0.0)
        if transfer_rate < 0.15:
            transfer_state = 0
        elif transfer_rate < 0.30:
            transfer_state = 1
        else:
            transfer_state = 2

        # 老龄化指数（3 档）
        aging = float(performance.get('aging_index', 0.3) or 0.3)
        if aging < 0.30:
            aging_state = 0
        elif aging < 0.50:
            aging_state = 1
        else:
            aging_state = 2

        # 生态压力（2 档）
        eco = float(performance.get('ecological_pressure', 0.0) or 0.0)
        eco_state = 1 if eco > 0.5 else 0

        return base + (cap_state, transfer_state, aging_state, eco_state)

    def _choose_action_rl_extended(self, state):
        """ε-greedy 动作选择（12 动作）。"""
        # @new: 如果外部加载了冻结Q-table，使用纯greedy选择
        if getattr(self, '_frozen', False):
            return self.select_action_frozen(state)

        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.EXTENDED_ACTIONS)
        elif len(self.q_table[state]) < len(self.EXTENDED_ACTIONS):
            # 兼容旧 5-action 状态：扩展为 12-action
            old_q = self.q_table[state]
            new_q = [0.0] * len(self.EXTENDED_ACTIONS)
            for i in range(min(len(old_q), len(new_q))):
                new_q[i] = old_q[i]
            self.q_table[state] = new_q

        if self.q_table_frozen:
            return int(np.argmax(self.q_table[state]))

        self.training_step_count += 1
        if self.training_step_count <= self.pretrain_steps:
            decay_factor = 1.0 - (self.training_step_count / self.pretrain_steps) * 0.8
            current_epsilon = self.epsilon * decay_factor
        else:
            current_epsilon = 0.05

        if np_random_value(self) < current_epsilon:
            return int(np_randint(self, 0, len(self.EXTENDED_ACTIONS)))
        return int(np.argmax(self.q_table[state]))

    def _apply_rl_action_extended(self, action):
        """执行扩展 RL 动作（12 动作）。"""
        if action <= 4:
            # 0-4 复用原动作
            return self._apply_rl_action(action)

        if action == 5:  # tighten_scale_threshold
            current = float(getattr(self, 'scale_subsidy_threshold', 50.0))
            self.scale_subsidy_threshold = min(current * 1.10, 200.0)
        elif action == 6:  # loosen_scale_threshold
            current = float(getattr(self, 'scale_subsidy_threshold', 50.0))
            self.scale_subsidy_threshold = max(current * 0.90, 5.0)
        elif action == 7:  # increase_transfer_friction
            current = float(getattr(self, 'transfer_friction_tax', 0.0))
            self.transfer_friction_tax = min(current + 0.02, 0.10)
        elif action == 8:  # decrease_transfer_friction
            current = float(getattr(self, 'transfer_friction_tax', 0.0))
            self.transfer_friction_tax = max(current - 0.02, 0.0)
        elif action == 9:  # subsidy_target_grain_only
            self.subsidy_target_extended = 'grain_only'
        elif action == 10:  # subsidy_target_operator
            self.subsidy_target_extended = 'operator'
        elif action == 11:  # emergency_intervention
            self.emergency_intervention_active = True
            current_floor = float(
                getattr(self, 'grain_price_floor', BASELINE_GRAIN_PRICE_FLOOR) or BASELINE_GRAIN_PRICE_FLOOR
            )
            current_subsidy_rate = float(getattr(self, 'subsidy_rate', 0.0) or 0.0)
            self.grain_price_floor = self._clamp_grain_price_floor(
                current_floor * 1.03
            )
            self.subsidy_rate = min(current_subsidy_rate + 0.03, 0.35)

    def _update_q_table_extended(self, reward, next_state):
        """
        Q-Learning 更新规则（扩展版本，支持 12-action）。

        与 _update_q_table 同结构，但确保 q_table 容量为 12。
        """
        if self.q_table_frozen:
            return
        if self.last_state is not None and self.last_action is not None:
            if next_state not in self.q_table:
                self.q_table[next_state] = [0.0] * len(self.EXTENDED_ACTIONS)
            elif len(self.q_table[next_state]) < len(self.EXTENDED_ACTIONS):
                old_q = self.q_table[next_state]
                new_q = [0.0] * len(self.EXTENDED_ACTIONS)
                for i in range(min(len(old_q), len(new_q))):
                    new_q[i] = old_q[i]
                self.q_table[next_state] = new_q

            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = [0.0] * len(self.EXTENDED_ACTIONS)
            elif len(self.q_table[self.last_state]) < len(self.EXTENDED_ACTIONS):
                old_q = self.q_table[self.last_state]
                new_q = [0.0] * len(self.EXTENDED_ACTIONS)
                for i in range(min(len(old_q), len(new_q))):
                    new_q[i] = old_q[i]
                self.q_table[self.last_state] = new_q

            old_q_val = self.q_table[self.last_state][self.last_action]
            max_future_q = float(np.max(self.q_table[next_state]))
            new_q_val = old_q_val + self.learning_rate * (
                reward + self.discount_factor * max_future_q - old_q_val
            )
            self.q_table[self.last_state][self.last_action] = new_q_val

    def _compute_capitalization_rate_for_rl(self):
        """从 environment 取年度资本化率。"""
        env = getattr(self.model, 'environment', None)
        if env is None:
            return 0.0
        return float(getattr(env, 'annual_rent_capitalization_rate', 0.0) or 0.0)

    def _compute_aging_index_for_rl(self):
        """农户老龄化指数（>=60 岁占比）。"""
        farmers = list(getattr(self.model, 'farmers', []))
        if not farmers:
            return 0.3
        elderly = sum(1 for f in farmers if getattr(f, 'age', 0) >= 60)
        return elderly / len(farmers)

    def _compute_ecological_pressure_for_rl(self):
        """从 environment 取生态压力（土壤退化累计）。"""
        env = getattr(self.model, 'environment', None)
        if env is None:
            return 0.0
        return float(getattr(env, 'soil_degradation_cumulative', 0.0) or 0.0)

    # ========== @new 2026-01-03: RL透明度可视化方法 ==========
    # @theory: 解决Nature审稿人关于RL"黑箱"问题的透明度要求
    # 参考: Molnar (2020) "Interpretable Machine Learning"
    
    def get_q_table_as_matrix(self):
        """
        将Q-table转换为矩阵格式，便于热力图可视化
        
        Returns:
            states: 状态列表 [(grain, income, fiscal), ...]
            actions: 动作名称列表
            matrix: Q值矩阵 (n_states x n_actions)
        """
        if not self.q_table:
            return [], [], np.array([])
        
        action_names = [
            'Maintain',
            'Subsidy+',
            'Subsidy-',
            'Price+',
            'Price-'
        ]
        
        # 排序状态以获得一致的顺序
        states = sorted(self.q_table.keys())
        n_states = len(states)
        n_actions = 5
        
        matrix = np.zeros((n_states, n_actions))
        for i, state in enumerate(states):
            matrix[i, :] = self.q_table[state]
        
        return states, action_names, matrix
    
    def get_optimal_policy_map(self):
        """
        生成最优策略映射表：state → optimal_action
        
        Returns:
            policy_map: dict {state_str: (action_name, q_value)}
        """
        action_names = ['Maintain', 'Subsidy+', 'Subsidy-', 'Price+', 'Price-']
        policy_map = {}
        
        for state, q_values in self.q_table.items():
            best_action_idx = int(np.argmax(q_values))
            best_q = q_values[best_action_idx]
            
            # 状态解释
            grain_str = ['Low', 'Med', 'High'][state[0]]
            income_str = ['Low', 'Med', 'High'][state[1]]
            fiscal_str = ['Normal', 'Tight'][state[2]]
            state_str = f"G:{grain_str},I:{income_str},F:{fiscal_str}"
            
            policy_map[state_str] = {
                'state_tuple': state,
                'best_action': action_names[best_action_idx],
                'best_q': best_q,
                'all_q_values': q_values.copy() if isinstance(q_values, list) else list(q_values)
            }
        
        return policy_map
    
    def export_q_table_for_visualization(self, filepath=None):
        """
        导出Q-table和策略映射到JSON文件，供可视化使用
        
        Args:
            filepath: 保存路径，None则返回dict不保存
            
        Returns:
            export_data: 包含Q-table、策略映射、元数据的字典
        """
        states, action_names, matrix = self.get_q_table_as_matrix()
        policy_map = self.get_optimal_policy_map()
        
        export_data = {
            'metadata': {
                'n_states': len(states),
                'n_actions': 5,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'state_space': {
                    'grain': ['Low (<0.9x target)', 'Med (0.9-1.1x)', 'High (>1.1x)'],
                    'income': ['Low (<0.9x target)', 'Med (0.9-1.1x)', 'High (>1.1x)'],
                    'fiscal': ['Normal (ratio<1)', 'Tight (ratio>1)']
                },
                'action_space': action_names
            },
            'q_table': {str(s): list(self.q_table[s]) for s in states},
            'policy_map': policy_map,
            'q_matrix': matrix.tolist() if matrix.size > 0 else [],
            'states': [str(s) for s in states]
        }
        
        if filepath:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return export_data
    
    def visualize_q_table_heatmap(self, save_path=None):
        """
        生成Q-table热力图
        
        Args:
            save_path: 保存路径，None则显示
            
        Returns:
            fig: matplotlib figure对象
        """
        import matplotlib.pyplot as plt
        
        states, action_names, matrix = self.get_q_table_as_matrix()
        
        if matrix.size == 0:
            print("[RL] Q-table为空，无法生成热力图")
            return None
        
        # 创建状态标签
        state_labels = []
        for state in states:
            grain_str = ['L', 'M', 'H'][state[0]]
            income_str = ['L', 'M', 'H'][state[1]]
            fiscal_str = ['N', 'T'][state[2]]
            state_labels.append(f"({grain_str},{income_str},{fiscal_str})")
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(states) * 0.4)))
        
        # 热力图
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # 标签
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(action_names, rotation=45, ha='right')
        ax.set_yticks(range(len(state_labels)))
        ax.set_yticklabels(state_labels)
        
        # 标注数值
        for i in range(len(state_labels)):
            for j in range(len(action_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(matrix[i, j]) > 0.5 else 'black')
        
        ax.set_xlabel('Actions', fontsize=12)
        ax.set_ylabel('States (Grain, Income, Fiscal)', fontsize=12)
        ax.set_title('Q-Table Heatmap: State-Action Values\n'
                     '(L=Low, M=Med, H=High; N=Normal, T=Tight)', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Q-Value')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[RL] Q-table热力图已保存: {save_path}")
        
        return fig
    
    def visualize_policy_phase_portrait(self, save_path=None):
        """
        生成政策相图：展示不同状态下的最优动作分布
        
        Args:
            save_path: 保存路径
            
        Returns:
            fig: matplotlib figure对象
        """
        import matplotlib.pyplot as plt
        
        policy_map = self.get_optimal_policy_map()
        
        if not policy_map:
            print("[RL] Q-table为空，无法生成政策相图")
            return None
        
        # 动作颜色映射
        action_colors = {
            'Maintain': '#808080',
            'Subsidy+': '#2E7D32',  # 绿色 - 增加补贴
            'Subsidy-': '#C62828',  # 红色 - 减少补贴
            'Price+': '#1565C0',    # 蓝色 - 提高价格
            'Price-': '#F57F17'     # 黄色 - 降低价格
        }
        
        # 创建3D状态空间可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 图1: Fiscal=Normal 时的政策选择
        ax1 = axes[0]
        ax1.set_title('Policy Map (Fiscal=Normal)', fontsize=12)
        ax1.set_xlabel('Grain State')
        ax1.set_ylabel('Income State')
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['Low', 'Med', 'High'])
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Low', 'Med', 'High'])
        
        for state_str, info in policy_map.items():
            state = info['state_tuple']
            if state[2] == 0:  # Fiscal=Normal
                color = action_colors[info['best_action']]
                ax1.scatter(state[0], state[1], c=color, s=300, edgecolors='black')
                ax1.annotate(info['best_action'][:3], (state[0], state[1]), 
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
        
        # 图2: Fiscal=Tight 时的政策选择
        ax2 = axes[1]
        ax2.set_title('Policy Map (Fiscal=Tight)', fontsize=12)
        ax2.set_xlabel('Grain State')
        ax2.set_ylabel('Income State')
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['Low', 'Med', 'High'])
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Med', 'High'])
        
        for state_str, info in policy_map.items():
            state = info['state_tuple']
            if state[2] == 1:  # Fiscal=Tight
                color = action_colors[info['best_action']]
                ax2.scatter(state[0], state[1], c=color, s=300, edgecolors='black')
                ax2.annotate(info['best_action'][:3], (state[0], state[1]), 
                           ha='center', va='center', fontsize=8, color='white',
                           fontweight='bold')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, edgecolor='black', label=a) 
                         for a, c in action_colors.items()]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                  bbox_to_anchor=(0.5, -0.05), fontsize=10)
        
        fig.suptitle('RL Policy Phase Portrait: State → Optimal Action Mapping', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[RL] 政策相图已保存: {save_path}")
        
        return fig
