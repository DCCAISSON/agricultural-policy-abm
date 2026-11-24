"""Run one behavior-only job, without overwriting any experiment or archive."""
import argparse
from datetime import datetime
import math
from pathlib import Path

from agents.government_agent import get_farmer_net_income
from scripts.national_actor_trace import write_actor_trace
from scripts.run_national_inertia_pairs import OLD, legacy, old, parcels, snapshot_plot_use


ROOT = old.ROOT / "results/diagnostics/national_household_portfolio"
ARCHIVE = old.ROOT / "results/diagnostics/national_inertia_pairs_20261121/run_02"
POLICY = {"base": 0, "ordinary_post": 1, "scale80_post": 2}
FILES = ("decision_trace.json", "actors.jsonl", "cash_flow.json", "fiscal_account.json",
         "actual_harvests.json", "actual_plot_use.json", "plot_production_status.json",
         "land.json", "lease_transactions.json", "government_sar.json")


def v1_household_inputs(cohort=None):
    """Use only pre-simulation NFPS observations; never join a CFPS person to a farm."""
    import pandas as pd
    from core.extract_microdata_features import REGION_MAPPING
    from scripts.prepare_household_portfolio_reference import MANIFEST, RAW
    job = old.read(MANIFEST)["jobs"][8]
    file = Path(job["cohort"])
    assert old.digest(file) == job["cohort_sha256"]
    if cohort is None: cohort = pd.DataFrame(old.read(file)["farmers"])
    raw = pd.read_csv(RAW, usecols=["year","smgen","hhnumbers","hhlabornum","sow_food","prod_food",
                                    "inc_wage_l","inc_wage_o"])
    origin = raw.iloc[cohort.source_row.to_numpy()-2].reset_index(drop=True).copy()
    assert origin.year.eq(2010).all() and origin.hhnumbers.eq(cohort.family_size).all()
    origin["region"] = cohort.region
    geo = {p:r for r,provinces in REGION_MAPPING.items() for p in provinces}
    assert origin.smgen.map(geo).eq(origin.region).all()
    pre = raw.loc[raw.year.eq(2010)].copy()
    pre["region"] = pre.smgen.map(geo)
    unmapped = pre.loc[pre.region.isna(),"smgen"].value_counts().to_dict()
    crops = pre[pre.region.notna() & pre.sow_food.gt(0) & pre.prod_food.ge(0)].copy()
    crops["kg_mu"] = crops.prod_food / crops.sow_food
    base = crops.groupby("region").kg_mu.agg(["size","median"])
    valid = pre.hhlabornum.between(0,pre.hhnumbers) & pre.hhlabornum.mod(1).eq(0)
    group_workers = pre[valid].groupby(["region","hhnumbers"]).hhlabornum.quantile(.5,interpolation="lower")
    workers, missing = {}, 0
    for r, key in zip(origin.itertuples(),cohort.cohort_agent_id):
        n = r.hhlabornum
        proxy = not (0 <= n <= r.hhnumbers and n % 1 == 0)
        if proxy:
            n = group_workers.loc[r.region,r.hhnumbers]
            missing += 1
        wage_receipt = True if r.inc_wage_l > 0 or r.inc_wage_o > 0 else (
            False if r.inc_wage_l == r.inc_wage_o == 0 else None)
        workers[str(key)] = (int(n),wage_receipt,proxy)
    assert len(workers)==600 and len(base)==4
    budget_path = old.ROOT/"results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv"
    budget = pd.read_csv(budget_path,usecols=["eligible","crop","output","sales_quantity"])
    budget = budget[budget.eligible]
    comparable = budget[budget.output.gt(0) & budget.sales_quantity.between(0,budget.output)]
    volumes = comparable.groupby("crop")[["output","sales_quantity"]].sum()
    cash = volumes.loc[["蔬菜","水果"]].sum()
    ratios = {"grain":float(volumes.at["稻谷","sales_quantity"]/volumes.at["稻谷","output"]),
              "cash_crop":float(cash.sales_quantity/cash.output)}
    return base["median"].to_dict(), workers, ratios, dict(
        nfps_sha256=old.digest(RAW), nfps_year=2010, labor_proxy_households=missing,
        budget_sha256=old.digest(budget_path), crop_budget_eligible=budget.groupby("crop").size().to_dict(),
        sales_comparable=comparable.groupby("crop").size().to_dict(),
        sold_fraction_by_crop=(volumes.sales_quantity/volumes.output).to_dict(),
        unmapped_province_rows=unmapped, grain_positive_sown_valid_output=base["size"].to_dict())


def run(seed, rule, policy="base", paired_old=None, organizations=60):
    v1 = rule == "v1"
    if seed not in (103, 109) or rule not in ("old", "new", "v1") or policy != "base":
        raise ValueError("behavior phase: seeds 103/109 and base only; policy phase locked")
    if organizations not in ((30, 60, 90) if v1 else (60,)):
        raise ValueError("organization supply sensitivity is restricted to V1")
    folder = ROOT.parent / "national_v1_20261122" if v1 else ROOT
    output = folder / "runs" / (f"s{seed}_{rule}_{policy}_" + (f"n{organizations}_" if v1 else "") + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    archive = ARCHIVE / f"s{seed}_m50_base_open"
    cfg, economics, budget, inputs = legacy.load_base()
    sources = legacy.source_hashes()
    assert inputs == old.read(ARCHIVE / "design.json")["frozen_inputs"]
    portfolio = rule in ("new", "v1")
    cfg["runtime_fixed_rule"]["household_crop_portfolio"] = portfolio
    if v1:
        cfg["runtime_fixed_rule"]["national_v1"] = True
        from core.fixed_rule_runtime import NFPS_HOUSEHOLD_ADJUSTMENT_PROXY
        rates, workers, ratios, reference_details = v1_household_inputs()
        cfg["runtime_fixed_rule"]["grain_kg_mu_by_region"] = rates
        cfg["runtime_fixed_rule"].update(household_sales_ratio=ratios,
            household_crop_adjustment_rate=NFPS_HOUSEHOLD_ADJUSTMENT_PROXY,organization_inertia_margin_yuan_per_mu=50.)
        cfg["scale"]["n_enterprises"] = organizations
        cfg["scale"]["enterprise_ratio"] = organizations / cfg["scale"]["n_farmers"]
    share_path = ROOT / "reference/initial_grain_shares.json"
    if portfolio:
        if paired_old is None:
            raise ValueError("supply an audited old run of the same seed")
        manifest = old.read(paired_old / "audit.json")
        assert manifest["ok"] and (manifest["seed"], manifest["rule"]) == (seed, "old")
        shares = old.read(share_path)
    model = legacy.build_model(cfg, economics, seed)
    if v1:
        for f in model.initial_household_cohort:
            n, has_wage, imputed = workers[str(f.unique_id)]
            f.set_v1_labor(n, has_wage)
            f.worker_count_source = "region_size_proxy" if imputed else "NFPS_2010_workers"
    if portfolio:
        ids = {str(a.unique_id) for a in model.initial_household_cohort}
        assert shares and shares.keys() <= ids
        assert all(type(x) in (float, int) and 0 <= x <= 1 for x in shares.values())
        model.household_initial_grain_shares = shares
    model.config["policy"]["government_q_project"].update(
        episode_end_step=3, uncapped_income_reward=True,
        grain_reference_kg=old.read(OLD / "training.json")["grain_reference_kg"],
        budget_area_menu=math.fsum(p.area for p in model.land_patches),
        h0_income_reference_yuan=math.fsum(get_farmer_net_income(f) for f in model.initial_household_cohort)/600)
    expected_cfg = old.read(archive / "effective_config.json")
    expected_cfg["runtime_fixed_rule"]["household_crop_portfolio"] = portfolio
    if v1:
        expected_cfg["runtime_fixed_rule"].update(cfg["runtime_fixed_rule"])
        from agents.government_agent import NATIONAL_V1_REWARD_WEIGHTS
        expected_cfg["policy"]["rl_reward_weights"] = dict(NATIONAL_V1_REWARD_WEIGHTS)
        expected_cfg["scale"]["n_enterprises"] = organizations
        expected_cfg["scale"]["enterprise_ratio"] = cfg["scale"]["enterprise_ratio"]
        # Both references are defined from the actual opening landscape/cohort.
        for key in ("budget_area_menu", "h0_income_reference_yuan"):
            expected_cfg["policy"]["government_q_project"][key] = model.config["policy"]["government_q_project"][key]
    assert model.config == expected_cfg
    assert model.config["runtime_fixed_rule"]["minimum_inertia_margin_yuan_per_mu"] == 50
    legacy.configure_government(model.government, {}, seed, False)
    # The inertia experiment fixed the choice directly, without counting unknown Q states.
    model.government._choose_action_rl = lambda state, action_count=3: POLICY[policy]
    output.mkdir(parents=True)
    if v1:
        old.write(output/"reference_counts.json", reference_details)
    path = output / "episode"
    old.write(output / "design.json", dict(seed=seed, rule=rule, policy=policy, policy_locked=True,
              source_hashes=sources, runner_sha256=old.digest(Path(__file__)), inputs=inputs,
              initial_parcels_sha256=legacy.fingerprint(parcels(model)), archive=str(archive),
              share_sha256=old.digest(share_path) if portfolio else None))

    def observe(current, year):
        old.write(year / "actual_plot_use.json", snapshot_plot_use(current))
        write_actor_trace(current, year)
        env = current.environment
        cfg = current.config
        last = env.labor_market_history[-1]
        old.write(year / "price_inputs.json", {
            "year": current.simulation_base_year + current.step_count,
            "base_crop_prices": cfg["environment"]["market"]["base_prices"],
            "base_land_rent": cfg["environment"]["market"]["base_land_rent"],
            "other_cost_per_mu": cfg["production"]["cost_structure"]["input_requirements"]["other_cost_per_mu"],
            "labor_shock_factor": last["shock_factor"],
        })
        old.write(year / "market_outcomes.json", dict(env.market_prices))
        if v1:
            p = cfg["environment"]["market"]["base_prices"]
            assert env.market_prices["crop_price"] == p["grain"] and env.market_prices["cash_crop"] == p["cash_crop"]

    print(path, flush=True)
    result = legacy.run_episode(model, budget, path, seed, sources, False, full_audit=True, observe_year=observe)
    assert not result["training"] and result["frozen_q_unchanged"]
    baseline = paired_old / "episode" if portfolio else archive
    checks = {}
    def compare(rel):
        lhs, rhs = old.digest(path / rel), old.digest(baseline / rel)
        checks[rel] = dict(match=lhs == rhs, actual=lhs, reference=rhs)
    for name in (("initial_exogenous.json",) if v1 else ("initial_households.json", "initial_operators.json", "initial_exogenous.json")):
        compare(name)
    if v1:
        lhs, rhs = (old.read(p/"initial_households.json") for p in (path, baseline))
        keys = ("id","microdata_id","weight","frozen_initial_liquidity","frozen_household_reserve")
        checks["fixed_cohort"] = dict(match=len(lhs)==len(rhs)==600 and
            [[r[k] for k in keys] for r in lhs] == [[r[k] for k in keys] for r in rhs])
        if organizations == 60:
            compare("initial_households.json")
    for i in (1, 2, 3):
        year = f"year_{i:02d}/"
        prices = old.read(path / (year+"price_inputs.json"))
        scenario = expected_cfg["paper_b_flagship_scenario"]
        assert prices["year"] == 2010+i
        assert prices["base_crop_prices"] == expected_cfg["environment"]["market"]["base_prices"]
        assert prices["base_land_rent"] == scenario["initial_land_rent"]*(1+scenario["rental_growth_rate"])**(i-1)
        for crop, key in (("grain", "initial_grain_other_cost"), ("cash_crop", "initial_cash_other_cost")):
            assert prices["other_cost_per_mu"][crop] == scenario[key]*(1+scenario["input_cost_growth_rate"])**(i-1)
        if portfolio and i == 1:
            decisions = old.read(path / (year+"decision_trace.json"))
            observed = {r["operator_id"]: r["previous_grain_share"] for r in decisions
                        if r["operator_type"] == "household" and r["previous_grain_share"] is not None}
            assert observed == shares, "initial shares did not enter actual household decisions"
        assert old.read(path / (year+"government_sar.json"))["action"] == 0
        weather = old.read(path / (year+"exogenous_shocks.json"))
        assert set(weather["stream_state_sha256"]) == {"b19_environment","b19_spatial_climate"}
        assert set(weather["spatial_maps"]) == {"current_hazard_map","current_favorable_map"}
        for name in (("price_inputs.json",) if portfolio else FILES) + ("exogenous_shocks.json",):
            compare(year + name)
    ok = all(r["match"] for r in checks.values())
    if portfolio and organizations == 60:
        ok = ok and old.read(output/"design.json")["initial_parcels_sha256"] == old.read(
            paired_old/"design.json")["initial_parcels_sha256"]
    old.write(output/"audit.json", dict(seed=seed, rule=rule, ok=ok, baseline=str(baseline), checks=checks,
              scope="Bound weather streams/maps; no separate crop-price innovation in the crop market. "
                    "Realized market prices and individual yields are not common exogenous draws. "
                    "Archived paths do not store the separate price-input observer."))
    if not ok:
        raise RuntimeError(f"pairing/reproduction incomplete: {output}/audit.json")
    print("verified", output, flush=True)


def estimate_crop_transitions(raw_file, excluded, continuous=False, end_year=2010, regional_pairs=False):
    """One pre-simulation transition per independently linked, non-cohort household."""
    import pandas as pd
    import numpy as np
    from core.fixed_rule_runtime import crop_history_group
    d=pd.read_csv(raw_file,usecols='year hhid id samplast smgen sow_food sow_cash'.split(),
                  dtype={'hhid':'string','id':'string'})
    years=[end_year-1,end_year]
    d=d[d.year.isin(years) & ~d.duplicated(['year','hhid'],keep=False) &
        d.hhid.eq(d.id) & ~d.hhid.isin(excluded) & d[['sow_food','sow_cash']].ge(0).all(axis=1)]
    p=d[d.year.eq(years[0])].merge(d[d.year.eq(end_year)&d.samplast.eq(1)],on=['hhid','smgen'],validate='1:1')
    m=pd.read_csv(raw_file.parent/'2009-2015individual.csv',usecols=['year','hhid','relation','sex','age'],dtype={'hhid':'string'})
    m=m[m.year.isin(years) & m.hhid.isin(p.hhid) & m.sex.isin([1,2]) & m.age.between(0,110)]
    roster={(k,y):list(g[['relation','sex','age']].itertuples(index=False,name=None)) for (k,y),g in m.groupby(['hhid','year'])}
    def linked(k):
        a,b=roster.get((k,years[0]),[]),roster.get((k,end_year),[])
        pairs=[(i,j) for i,r in enumerate(a) for j,s in enumerate(b) if r[:2]==s[:2] and abs(s[2]-r[2]-1)<=1]
        rel=[a[i][0] for i,j in pairs if sum(x==i for x,y in pairs)==sum(y==j for x,y in pairs)==1]
        return 1 in rel and len(rel)>=2
    p=p[p.hhid.map(linked)].copy()
    p['prior']=[crop_history_group(g,c) for g,c in zip(p.sow_food_x,p.sow_cash_x)]
    p['share']=p.sow_food_y/(p.sow_food_y+p.sow_cash_y)
    if continuous:
        p['prior']=p.prior.replace({'mixed_lower':'mixed','mixed_upper':'mixed'})
        p['prior_share']=p.sow_food_x/(p.sow_food_x+p.sow_cash_x)
        if regional_pairs: return p[['hhid','smgen','prior','prior_share','share']]
        return {s:(g.prior_share.to_numpy(),g.share.to_numpy()) for s,g in p.groupby('prior')}
    p=p[p.share.notna()].copy()
    # Five interior intervals plus the two exact endpoints. No evaluation-year observations.
    p['category']=np.where(p.share==0,0,np.where(p.share==1,6,1+np.minimum(4,(p.share*5).astype(int))))
    rows=[]
    for prior,g in p.groupby('prior'):
        for category in range(7):
            x=g.loc[g.category.eq(category),'share']; n=len(g); prob=len(x)/n
            rows.append(dict(prior=prior,category=category,n=n,count=len(x),probability=prob,
                             cluster_se=np.sqrt(prob*(1-prob)/(n-1)),target=x.mean()))
    return pd.DataFrame(rows)


def run_area_structures(destination, mix_mode=None, *, settings=None):
    """One entry for paired, fixed-policy structural comparisons. Private rows stay local."""
    import json
    import hashlib
    import numpy as np
    import pandas as pd
    from scripts.prepare_household_portfolio_reference import RAW, MANIFEST
    from core.extract_microdata_features import REGION_MAPPING
    dest=Path(destination)
    settings=settings or {}
    stage=settings.get('stage',0)
    dest.mkdir(parents=True,exist_ok=False)
    private=dest/'PRIVATE_DO_NOT_SHARE'
    private.mkdir()
    geo={p:r for r,ps in REGION_MAPPING.items() for p in ps}
    cfg,econ,budget,_=legacy.load_base()
    cfg['runtime_fixed_rule'].update(national_v1=True,initial_tenure_mode='A')
    cfg['scale'].update(n_enterprises=90,enterprise_ratio=90/600)
    job=old.read(MANIFEST)['jobs'][8]
    cohort=pd.DataFrame(old.read(Path(job['cohort']))['farmers'])
    frozen=cohort
    soft=mix_mode in ('B','Bmid','Blow','Bhigh','D','Dmid','Dlow','Dhigh')
    calibrated=mix_mode in ('C','D','Dmid','Dlow','Dhigh')
    factorial=soft or calibrated
    if factorial:
        from scripts.prepare_nfps_v3 import candidate_cohort
        candidate=candidate_cohort()
        if calibrated: cohort=candidate
    rates,workers,ratios,refs=v1_household_inputs(cohort)
    cols='year hhid id smgen hhnumbers area_e area_e1 area_in1 area_de2 sow_food sow_cash sow_total prod_food saving_1 saving_2 inc_wage_l inc_wage_o'.split()
    if stage: cols+=['sow_oother','area_b','area_b1']
    dictionary=pd.read_excel(RAW.parent/'household variables2009-2015.xlsx',header=None)
    dictionary[dictionary[0].isin(cols+['hhlabornum','prod_food'])].to_csv(dest/'survey_dictionary.csv',index=False)
    raw=pd.read_csv(RAW,usecols=cols,dtype={'hhid':'string','id':'string'})
    original=raw.iloc[frozen.source_row.to_numpy()-2]
    if factorial:
        from scripts.prepare_nfps_v3 import clean_area
        clean_area(raw)
    b=raw.iloc[cohort.source_row.to_numpy()-2].reset_index(drop=True).copy()
    assert b.year.eq(2010).all() and b.hhnumbers.eq(cohort.family_size).all() and len(b)==600
    b['actor']=cohort.cohort_agent_id
    b['region']=b.smgen.map(geo)
    b['region_land']=b.groupby('smgen').area_e.transform('sum')
    assert (b.region==cohort.region).all() and b.area_e.gt(0).all()
    assert not (b.area_e1.dropna()<0).any() and not (b.area_e1>b.area_e).any()
    b['verified_right']=b.area_e1.fillna(0.) # no verified right to offer, not an observed zero
    b['intensity']=(b.sow_food+b.sow_cash)/b.area_e
    valid=b.intensity.where(b.intensity.gt(0))
    b['intensity_proxy']=valid.isna()
    b['intensity']=valid.fillna(valid.groupby(b.region).transform('median'))
    cpi=pd.read_csv(old.ROOT/'config/experiments/national_real2024_cpi_20260909.csv')
    deflator=(1+cpi.loc[cpi.year.between(2011,2024),'annual_change_percent']/100).prod()
    b['opening_cash']=0.
    b['wage_income']=0.
    donor=raw[raw.year.eq(2010)].copy()
    def scale(x): return np.digitize(x,[3,10,30])
    for col,target in [('saving_1','opening_cash'),('saving_2','opening_cash'),
                       ('inc_wage_l','wage_income'),('inc_wage_o','wage_income')]:
        valid=donor[col].where(donor[col]>=0)
        med=valid.groupby([donor.smgen,scale(donor.area_e)]).median()
        source=b[col].where(b[col]>=0)
        proxy=pd.Series([med.get((z,s),np.nan) for z,s in zip(b.smgen,scale(b.area_e))])
        proxy=proxy.fillna(b.smgen.map(valid.groupby(donor.smgen).median())).fillna(valid.median())
        b[col+'_proxy']=source.isna()
        b[target]+=source.fillna(proxy)*deflator
    b['held_reserve']=[v['held_reserve_yuan2024'] for v in cohort.economic_basis_20261021]
    b['consumption']=[v['annual_reported_cash_candidate_yuan2024'] for v in cohort.economic_basis_20261021]
    assert b[['intensity','opening_cash','wage_income','consumption']].notna().all().all()
    bb=pd.read_csv(old.ROOT/'results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv')
    sell=bb[bb.eligible & bb.crop.isin(['蔬菜','水果']) & bb.sales_amount.ge(0) & bb.area.gt(0)]
    cash_sale=(sell.sales_amount*sell.to_2024).sum()/sell.area.sum()
    req=cfg['production']['cost_structure']['input_requirements']
    pp=cfg['fiscal_system']['p3_support']
    zones=sorted(set(b.smgen)|set(original.smgen))
    yield_map={int(z):rates[geo[z]] for z in zones}
    target={int(z):float(b.loc[b.smgen==z,'sow_food'].sum()*yield_map[z]) for z in zones}
    inputs={z:(y*cfg['environment']['market']['base_prices']['grain']*2*ratios['grain'],
               cash_sale,*[req['other_cost_per_mu'][t] for t in ('grain','cash_crop')],
               *[req['labor_days'][t] for t in ('grain','cash_crop')]) for z,y in yield_map.items()}
    base=b.set_index('actor')
    if stage:
        from scripts.prepare_nfps_v3 import prepare_v4_inputs
        import os
        fitted,profiles=prepare_v4_inputs(cohort,dest,deflator,os.environ.get('AP_ABM_CHFS2011'))
        paired=sell[sell.output.gt(0)&sell.sales_quantity.between(0,sell.output)&sell.labor.gt(0)&sell.nonlabor_low.ge(0)].copy()
        paired['sale2024']=paired.sales_amount*paired.to_2024
        paired['mech2024']=paired.machinery*paired.to_2024
        paired['unit_margin']=(paired.sale2024-paired.nonlabor_low)/paired.area
        paired['band']=np.minimum(2,(paired.unit_margin.rank(method='first',pct=True)*3-1e-8).astype(int))
        budgets={}
        for band,tab in paired.groupby('band'):
            aa=tab.area.sum()
            budgets[['low','mid','high'][band]]=dict(n=len(tab),unit_sale=tab.sale2024.sum()/aa,
                unit_input=tab.nonlabor_low.sum()/aa,unit_labor=tab.labor.sum()/aa,
                unit_output=tab.output.sum()/aa,unit_mechanical=tab.mech2024.sum(min_count=1)/tab.loc[tab.mech2024.notna(),'area'].sum(),
                sold_fraction=tab.sales_quantity.sum()/tab.output.sum(),cpi_area_mean=(tab.area*tab.to_2024).sum()/aa)
        pd.DataFrame(budgets).T.to_csv(dest/'cash_budget_ranges.csv')
        selected_budget=budgets.get(settings.get('budget','frozen'))
        if selected_budget:
            inputs={z:(r[0],selected_budget['unit_sale'],r[2],selected_budget['unit_input'],r[4],selected_budget['unit_labor']) for z,r in inputs.items()}
        cash_yield=(selected_budget or dict(unit_output=paired.output.sum()/paired.area.sum()))['unit_output']
        inputs={z:tuple(c*(1.+settings.get(k,0.)) for k,c in zip(('grain','cash','fixed','fixed','fixed','fixed'),s)) for z,s in inputs.items()}
        if settings.get('rights',False):
            # The opening contracted-area report is only an alternative source in this scenario.
            base['verified_right']=base.area_e1.fillna(base.area_b1.where(base.area_b1.between(0,base.area_e))).fillna(0.)
    transition=None
    history=None
    if mix_mode in ('continuous','C'):
        history=estimate_crop_transitions(RAW,set(original.hhid.dropna()),continuous=True)
        transition={}
    if soft and not stage:
        excluded=set(original.hhid)|set(raw.iloc[candidate.source_row.to_numpy()-2].hhid)
        pairs=[];annual={}
        for yr in (2010,2011,2012):
            p=estimate_crop_transitions(RAW,excluded,True,yr,True)
            p['region']=p.smgen.map(geo);pairs.append(p)
            h=pd.concat(pairs)
            annual[yr]={z:{s:(g.prior_share.to_numpy(),g.share.to_numpy()) for s,g in rx.groupby('prior')}
                        for z,rx in h.groupby('region')}
        premium=np.median([cash_sale-req['other_cost_per_mu']['cash_crop']-(rates[z]*cfg['environment']['market']['base_prices']['grain']*2*ratios['grain']-req['other_cost_per_mu']['grain']) for z in rates])
        penalty=premium*{'Blow':.5,'Dlow':.5,'Bmid':1.,'Dmid':1.}.get(mix_mode,1.5)
        penalties=None
        if mix_mode in ('B','D'):
            past=estimate_crop_transitions(RAW,set(),True,2010,True)
            past=past.dropna(subset=['prior_share','share']).copy()
            final=np.select([past.share.eq(0),past.share.eq(1)],['cash','grain'],default='mixed')
            past['switched']=past.prior.ne(final)
            regional=past[~past.hhid.isin(excluded)].groupby(past.smgen.map(geo)).switched.mean()
            measured=b.hhid.map(past.set_index('hhid').switched)
            tendency=measured.where(measured.notna(),b.region.map(regional)).astype(float)
            assert tendency.between(0,1).all()
            penalties=dict(zip(b.actor,premium*(1.5-.5*tendency)))
            pd.DataFrame(dict(actor=b.actor,observed=measured.notna(),switch_proxy=tendency,
                              penalty=b.actor.map(penalties))).to_csv(private/'historical_mobility.csv',index=False)
        transition={}
    if mix_mode=='candidate':
        estimates=estimate_crop_transitions(RAW,set(b.hhid.dropna()))
        estimates.to_csv(dest/'transition_estimates.csv',index=False)
        transition={state:(g.probability.cumsum().to_numpy(),g.target.to_numpy())
                    for state,g in estimates.groupby('prior')}
    benchmark=raw.loc[raw.year.eq(2010),'hhid']
    keys=b.loc[~b.hhid.isin(benchmark[benchmark.duplicated(False)]),['actor','hhid','id']].dropna()
    survey=[]
    for yr in range(2011,2016):
        v=raw.loc[raw.year.eq(yr)].dropna(subset=['hhid','id'])
        survey.append(keys.merge(v.loc[~v.hhid.duplicated(False)],on=['hhid','id'],validate='1:1'))
    survey=pd.concat(survey)
    b.drop(columns=['hhid','id']).to_csv(private/'baseline.csv',index=False)
    results=[]; checks=[]; fingerprints=[]
    for seed in settings.get('seeds',(103,109)):
        model=legacy.build_model(cfg,econ,seed)
        families=list(model.initial_household_cohort)
        orgs=list(model.agricultural_enterprises)
        assert len(orgs)==90
        cases=[('family',60,1.),('company',60,1.),('coop',60,1.),('both',60,1.),
               ('both',30,1.),('both',90,1.),('both',60,.5),('both',60,1.5)]
        if seed==103: cases.append(('repeat',60,1.))
        if mix_mode:
            cases=[('both',settings.get('organizations',60),1.)]
            if (mix_mode in ('candidate','continuous') or factorial) and seed==103 and settings.get('repeat',True):
                cases.append(('repeat',settings.get('organizations',60),1.))
        for structure,n,supply in cases:
            name=f'{seed}_{structure}_{n}_{supply}'
            actors=[]
            for a in families:
                count,paid,imputed=workers[str(a.unique_id)]
                if calibrated: a.household_size=int(base.loc[a.unique_id,'hhnumbers'])
                a.set_v1_labor(count,paid)
                a.initialize_area_household(base.loc[a.unique_id])
                if transition is not None or stage:
                    row=base.loc[a.unique_id]
                    a.prior_crops=(row.sow_food,row.sow_cash)
                if stage:
                    profile=profiles.loc[a.unique_id]
                    a.area_prior_transfer=int(profile.flow)
                    yy=float(profile.grain_yield)
                    a.area_grain_yield=yy if np.isfinite(yy) and yy>0 else yield_map[a.zone]
                    if stage>=2 and np.isfinite(profile.labor):
                        a.family_days=float(profile.labor)
                    if stage>=2 and np.isfinite(profile.initial_liquidity):
                        a.balance=float(profile.initial_liquidity)
                    a.balance*=1.+settings.get('cash_funds',0.)
                actors.append(a)
            roster=[]
            for kind,prob in [('AGRI_CORPORATION',.6),('COOPERATIVE',.4)]:
                count=round(n*prob)
                # Split each reference competitor proportionally; keep region/type totals exact.
                rng=np.random.default_rng([seed,0 if kind=='COOPERATIVE' else 1])
                base_count=round(30*prob)
                split=n//30
                weights=rng.lognormal(0,.6,base_count);weights/=weights.sum()
                weights=np.repeat(weights/split,split)
                rz=np.repeat(rng.choice(original.smgen,base_count),split)
                for j,(w,z) in enumerate(zip(weights,rz)):
                    index=j+(0 if kind=='AGRI_CORPORATION' else round(n*.6))
                    a=orgs[index]
                    cap=600000.*prob*w
                    if stage: cap*=settings.get('capital',1.)
                    limit=600.*prob*w*supply
                    a.initialize_area_business(kind,z,cap,limit)
                    active=structure in ('both','repeat') or (structure=='company' and kind=='AGRI_CORPORATION') or (structure=='coop' and kind=='COOPERATIVE')
                    roster.append(dict(id=a.unique_id,kind=kind,region=int(z),capital=cap,capacity=limit,
                        fixed=a.fixed_area_cost,management=a.management_area_cost,initial_area=0,activated=active))
                    if active: actors.append(a)
            pd.DataFrame(roster).to_csv(private/(name+'_organizations.csv'),index=False)
            model.step_count=0
            model.area_economy=dict(seed=seed,actors=actors,inputs=inputs,yields=yield_map,target=target,
                rate=pp['preserved_projects'][0]['rate'],budget=pp['base_budget_yuan'],
                wage=cfg['labor_market']['base_agricultural_wage'],term=1,contracts=[],
                rents={z:cfg['environment']['market']['base_land_rent'] for z in zones},
                state={z:dict(grain_kg=q,grain_gap=0.,grain_mu=float(b.loc[b.smgen==z,'sow_food'].sum()),
                              leased_mu=None,paid=None,budget_fraction_national=None) for z,q in target.items()})
            if stage:
                model.area_economy.update(v4=stage,v4_fit=fitted,macro_region=geo,
                    wage=model.area_economy['wage']*(1.+settings.get('wage',0.)),
                    rent_shift=settings.get('rent',0.),incentive=settings.get('incentive',1.),
                    scalar_reference=settings.get('scalar',False),cash_yield=cash_yield)
                if stage>=4:
                    from modules.spatial_index import area_moore_neighbors
                    space=area_moore_neighbors(actors)
                    space.update(search=settings.get('space',True),fee=.05*cfg['environment']['market']['base_land_rent'])
                    model.area_economy['v4_space']=space
                    old.write(private/(name+'_space.json'),dict(synthetic=True,coordinates=space['coordinates'],areas=space['area'],
                         rights=space['rights'],anchors=space['anchors'],zone=space['zone'],
                         edges=int(space['neighbors'].nnz)))
                    from scipy.sparse import save_npz
                    save_npz(private/(name+'_neighbors.npz'),space['neighbors'])
            logs=[];contracts=[];offers=[];sars=[]
            if transition is not None: model.area_economy['transition']=transition
            if history is not None: model.area_economy['continuous_crop_history']=history
            for yr in range(2011,2016):
                if soft and not stage:
                    model.area_economy.update(regional_history=annual[min(2012,yr-1)],macro_region=geo,crop_penalty=penalty)
                    if penalties is not None:model.area_economy['actor_crop_penalty']=penalties
                model.step()
                rows,orders,deals,sar=model.area_last
                ar=pd.DataFrame(rows); logs.append(ar);offers.extend(dict(year=yr,**o) for o in orders)
                contracts.extend(deals);sars.append(sar)
                fam=ar[ar.kind=='household']; org=ar[ar.kind!='household']
                assert abs(ar.operated.sum()-b.area_e.sum())<1e-6
                assert abs(ar.rent_received.sum()-ar.rent_paid.sum())<1e-6
                assert (ar.closing-ar.opening-ar.net+ar.consumption).abs().max()<1e-6
                for kind,group in [('all',ar),('household',fam),('organization',org),
                                 ('company',ar[ar.kind=='AGRI_CORPORATION']),('coop',ar[ar.kind=='COOPERATIVE'])]:
                    results.append(dict(case=name,seed=seed,structure=structure,n=n,supply=supply,year=yr,kind=kind,
                        **group[['operated','cultivated','rented_in','rented_out','grain','crop','grain_kg','net','surplus','paid','idle','hired_days','exited']].sum().to_dict(),
                        avg_rent=group.rent_paid.sum()/group.rented_in.sum() if group.rented_in.sum()>0 else None))
                linked=survey[survey.year==yr].merge(fam,left_on='actor',right_on='id',validate='1:1').dropna(subset=['sow_food','sow_cash'])
                G,C=linked.sow_food,linked.sow_cash
                g,c=linked.grain,linked.crop
                share=(G/(G+C).where(G+C>0)).quantile([.25,.5,.75])
                sim=(g/(g+c).where(g+c>1e-7)).quantile([.25,.5,.75])
                error=abs(((G>0)&(C>0)).mean()-((g>1e-7)&(c>1e-7)).mean())
                mae=(share-sim).abs().mean()
                checks.append(dict(case=name,year=yr,matched=len(linked),survey_planted=(G+C>0).sum(),sim_planted=(g+c>1e-7).sum(),
                    survey_mixed=((G>0)&(C>0)).sum(),sim_mixed=((g>1e-7)&(c>1e-7)).sum(),survey_grain=G.sum(),sim_grain=g.sum(),
                    survey_operated_median=linked.area_e.median(),sim_operated_median=linked.operated.median(),
                    survey_sown_intensity=((G+C)/linked.area_e.where(linked.area_e>0)).median(),
                    sim_sown_intensity=((g+c)/linked.operated.where(linked.operated>0)).median(),
                    survey_in_n=linked.area_in1.count(),survey_in_reported_mu=linked.area_in1.sum(min_count=1),
                    sim_in_on_reporting_mu=linked.loc[linked.area_in1.notna(),'rented_in'].sum(),
                    survey_out_n=linked.area_de2.count(),survey_out_reported_mu=linked.area_de2.sum(min_count=1),
                    sim_out_on_reporting_mu=linked.loc[linked.area_de2.notna(),'rented_out'].sum(),
                    survey_grain_kg=linked.prod_food.sum(min_count=1),survey_output_n=linked.prod_food.count(),
                    sim_grain_kg_on_reporting=linked.loc[linked.prod_food.notna(),'grain_kg'].sum(),
                    survey_kg_per_mu=linked.prod_food.sum()/G[linked.prod_food.notna()].sum(),
                    sim_kg_per_mu=linked.grain_kg.sum()/g.sum() if g.sum()>0 else None,
                    sim_operating_surplus_mean=linked.surplus.mean(),sim_wage_mean=linked.nonfarm.mean(),
                    survey_wage_joint_n=(linked.inc_wage_l.notna()&linked.inc_wage_o.notna()).sum(),
                    survey_wage_joint_mean_real=((linked.inc_wage_l+linked.inc_wage_o)*(1+cpi.loc[cpi.year.between(yr+1,2024),'annual_change_percent']/100).prod()).mean(),
                    survey_share_p25=share.iloc[0],survey_share_median=share.iloc[1],survey_share_p75=share.iloc[2],
                    sim_share_p25=sim.iloc[0],sim_share_median=sim.iloc[1],sim_share_p75=sim.iloc[2],
                    cash_binding=(linked.cash_shadow>1e-7).mean(),mixed_error=error,share_mae=mae,
                    passed=bool(error<=.10 and mae<=.15)))
                if stage:
                    checks[-1].update(survey_grain_only=int(((G>0)&(C==0)).sum()),survey_cash_only=int(((G==0)&(C>0)).sum()),
                        survey_other_only=int(((G+C==0)&linked.sow_oother.gt(0)).sum()),
                        sim_grain_only=int(((g>1e-7)&(c<=1e-7)).sum()),sim_cash_only=int(((g<=1e-7)&(c>1e-7)).sum()),
                        grain_area_error=float(abs(g.sum()-G.sum())/G.sum()),survey_cash=float(C.sum()),sim_cash=float(c.sum()),
                        survey_all_sown=float(linked.sow_total.sum()),survey_other_sown=float(linked.sow_oother.sum()),
                        full_gate=bool(error<=.10 and mae<=.15 and abs(g.sum()-G.sum())/G.sum()<=.10))
            df=pd.concat(logs)
            df.to_csv(private/(name+'_accounts.csv'),index=False)
            pd.DataFrame(contracts).to_csv(private/(name+'_contracts.csv'),index=False)
            pd.DataFrame(offers).to_csv(private/(name+'_orders.csv'),index=False)
            def clean(obj):
                if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
                if isinstance(obj,np.generic): return obj.item()
                return obj
            old.write(private/(name+'_government.json'),[clean(s) for s in sars])
            fingerprints.append(dict(case=name,accounts=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()))
            print(name,len(contracts),'contracts',flush=True)
            pd.DataFrame(results).to_csv(dest/'structures.csv',index=False)
            pd.DataFrame(checks).to_csv(dest/'behavior.csv',index=False)
    fp=pd.DataFrame(fingerprints).set_index('case')
    if mix_mode!='baseline':
        for name in fp.index[fp.index.str.contains('_repeat_')]:
            assert fp.loc[name.replace('_repeat_','_both_'),'accounts']==fp.loc[name,'accounts']
    fp.to_csv(dest/'fingerprints.csv')
    if soft and not stage: pd.DataFrame([dict(last_year=y,prior=s,region=z,n=len(v[0]),positive=int(np.isfinite(v[1]).sum()),penalty=penalty)
            for y,h in annual.items() for z,r in h.items() for s,v in r.items()]).to_csv(dest/'fit.csv',index=False)
    old.write(dest/'inputs.json',dict(references=refs,cash_sale=cash_sale,cash_sale_n=len(sell),deflator=deflator,
            source_sha256=old.digest(RAW),cohort_sha256=(hashlib.sha256(cohort.to_json(orient='records',force_ascii=False).encode()).hexdigest() if calibrated else job['cohort_sha256']),unverified_rights=int(b.area_e1.isna().sum()),
            positive_intensity_proxy=int(b.intensity_proxy.sum()),other_sown_gap_mu=float((b.sow_total-b.sow_food-b.sow_cash).sum())))
    if settings: old.write(dest/'settings.json',settings)


def run_v4_suite(destination, workers):
    """Reproduce the published matrix through this module, then audit saved accounts."""
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    import sys
    dst=Path(destination)
    root=dst/'PRIVATE_DO_NOT_SHARE'
    root.mkdir(parents=True,exist_ok=False)
    (dst/'run_01').mkdir()
    tests=old.read(old.ROOT/'results/diagnostics/national_crop_mix_repair_20261123/run_01/run_manifest.json')
    t=[arg for arg in tests['tests_command'].split() if arg.startswith('tests/')]
    subprocess.run([sys.executable,'-B','-X','utf8','-m','pytest','-q',*t,
        'tests/test_continuous_crop_history.py','tests/test_v3_crop_response.py','tests/test_v4_area_economy.py',
        f'--junitxml={dst/"run_01"/"tests.xml"}'],check=True)
    suite=[(f'M{k}',f'M{k}',[]) for k in range(5)]
    for lo in range(201,221,5):
        for mode in ('M0','V4'):
            suite.append((mode,f'{mode}_{lo}',['--no-repeat','--seeds',*map(str,range(lo,lo+5))]))
    sens=[('scalar',['--scalar']),('nospatial',['--no-space']),('rights',['--rights']),
          ('org30',['--organizations','30']),('org90',['--organizations','90']),
          ('caplow',['--capital','0.5']),('caphigh',['--capital','1.5']),
          ('incentivelow',['--incentive','0.5']),('incentivehigh',['--incentive','2.0'])]
    for field in ('grain','cash','wage','rent','cash-funds'):
        for tag,delta in [('low','-0.1'),('high','0.1')]: sens.append((field+tag,['--'+field,delta]))
    for tier in ('low','mid','high'):
        sens.append(('budget_'+tier,['--budget',tier]))
    for name,args in sens:
        suite.append(('V4',name,['--no-repeat','--seeds',*(['103'] if name=='scalar' else ['103','109']),*args]))

    def run(row):
        mode,name,args=row
        with (root/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B','-X','utf8','-m','scripts.run_national_household_portfolio',
                            'v4',mode,str(root/name),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,'complete',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run,suite))
    from scripts.audit_household_portfolio_behavior import audit_v4
    audit_v4(dst)


if __name__ == "__main__" and __import__('sys').argv[1:2]==['v4-suite']:
    import sys
    p=argparse.ArgumentParser()
    p.add_argument('destination')
    p.add_argument('--workers',type=int,default=5)
    ns=p.parse_args(sys.argv[2:])
    run_v4_suite(ns.destination,ns.workers)
elif __name__ == "__main__" and __import__('sys').argv[1:2]==['crop-mix']:
    import sys
    if len(sys.argv)!=4 or sys.argv[2] not in ('baseline','candidate','continuous','B','Bmid','Blow','Bhigh','C','D','Dmid','Dlow','Dhigh'):
        raise SystemExit('crop-mix baseline|candidate|continuous|B|Bmid|Blow|Bhigh|C|D|Dmid|Dlow|Dhigh NEW_DIRECTORY')
    run_area_structures(sys.argv[3],sys.argv[2])
elif __name__ == "__main__" and __import__('sys').argv[1:2]==['area']:
    import sys
    run_area_structures(sys.argv[2])
elif __name__ == "__main__" and __import__('sys').argv[1:2]==['v4']:
    import sys
    p=argparse.ArgumentParser(description='One annual model for all frozen V4 comparisons')
    p.add_argument('mode',choices=['M0','M1','M2','M3','M4','V4'])
    p.add_argument('destination')
    p.add_argument('--seeds',type=int,nargs='+',default=[103,109])
    p.add_argument('--no-repeat',dest='repeat',action='store_false')
    p.add_argument('--scalar',action='store_true')
    p.add_argument('--no-space',dest='space',action='store_false')
    p.add_argument('--rights',action='store_true')
    p.add_argument('--organizations',type=int,choices=(30,60,90),default=60)
    p.add_argument('--capital',type=float,default=1.)
    p.add_argument('--incentive',type=float,default=1.)
    p.add_argument('--budget',choices=['frozen','low','mid','high'],default='frozen')
    for x in ('grain','cash','wage','rent','cash-funds'): p.add_argument('--'+x,type=float,default=0.)
    v=vars(p.parse_args(sys.argv[2:]))
    v['stage']=4 if v['mode']=='V4' else int(v['mode'][1])
    dest=v.pop('destination')
    run_area_structures(dest,'D',settings=v)
elif __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int, choices=(103,109))
    p.add_argument("rule", choices=("old","new","v1"))
    p.add_argument("--paired-old", type=Path)
    p.add_argument("--organizations", type=int, choices=(30,60,90), default=60)
    args = p.parse_args()
    run(args.seed, args.rule, paired_old=args.paired_old, organizations=args.organizations)
