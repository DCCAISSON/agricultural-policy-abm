"""Audit the locked F1 household bridges; emit aggregates only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "0a088711c6b828a7e36ced5fbf2935acbdb85a1e"
YEARS = range(2011, 2016)
SEEDS = range(301, 321)
EPS = 1e-7
MIN_N = 30
SURVEY = "area_b area_e area_b1 area_e1 area_in1 area_de2 sow_total sow_food sow_cash sow_oother".split()
KEY = ["year", "id"]


def raw_panel(project: Path) -> pd.DataFrame:
    from scripts import prepare_household_portfolio_reference as ref
    from scripts import prepare_nfps_v3 as prep

    raw = project / ref.RAW.relative_to(ROOT)
    protocol = json.loads((ROOT / "docs/全国经营接续/validation_protocol.json").read_bytes())
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == protocol["authorized_household_sha256"]
    with patch.multiple(prep, RAW=raw, MANIFEST=project / ref.MANIFEST.relative_to(ROOT),
                        WEIGHTS=project / prep.WEIGHTS.relative_to(ROOT)):
        cohort = prep.candidate_cohort()
    source = pd.read_csv(raw, usecols=["year", "hhid", "id", "area_in", "area_de", *SURVEY],
                         dtype={"hhid": "string", "id": "string"})
    prep.clean_area(source)
    base = source.iloc[cohort.source_row.to_numpy() - 2]
    assert base.year.eq(2010).all() and base.hhid.is_unique
    keys = base[["hhid", "id", "area_e"]].rename(columns={"area_e": "survey_base_2010"})
    keys = keys.assign(actor=cohort.cohort_agent_id.to_numpy())
    valid = source.loc[source.hhid.notna() & source.id.notna()
                       & ~source.duplicated(["year", "hhid"], keep=False)]
    return (valid.merge(keys, on=["hhid", "id"], validate="m:1")
            .drop(columns=["hhid", "id"]).rename(columns={"actor": "id"})
            .set_index(KEY).sort_index())


def lease_position(book: pd.DataFrame, ids: pd.Index) -> tuple[pd.Series, pd.Series]:
    ins = book.groupby("buyer").area.sum().reindex(ids, fill_value=0.)
    outs = book.groupby("seller").area.sum().reindex(ids, fill_value=0.)
    return ins, outs


def archive_prefix(project: Path, seed: int) -> Path:
    diagnostics = project / "results/diagnostics"
    archive = diagnostics / "national_empirical_succession_20261126"
    locked = archive / f"locked_{301 + ((seed - 301) // 5) * 5}" / "PRIVATE_DO_NOT_SHARE"
    return locked / f"{seed}_both_60_1.0"


def ledger(project: Path, seed: int, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = archive_prefix(project, seed)
    survey_file = (project / "results/diagnostics/national_abm_v41_final_economic_closure_20261124"
                   / "PRIVATE_DO_NOT_SHARE" / f"F1_land_bridge_{seed}.csv")
    saved = pd.read_csv(survey_file).set_index(KEY).sort_index()
    data = pd.read_csv(f"{prefix}_accounts.csv").query("kind == 'household'").set_index(KEY).sort_index()
    assert data.index.is_unique and data.index.equals(saved.index) and len(data) == 3000
    identity = ["base_area", "operated", "cultivated", "grain", "crop", "rented_in", "rented_out"]
    np.testing.assert_allclose(data[identity], saved[identity], atol=1e-6, rtol=1e-10)
    matched = panel.reindex(data.index)
    np.testing.assert_allclose(matched[SURVEY], saved[SURVEY], atol=1e-6, rtol=0, equal_nan=True)
    data = data.join(matched)
    data["model_base_2010"] = saved.initial_operated
    np.testing.assert_allclose(data.loc[2011].model_base_2010, panel.loc[2010].loc[data.loc[2011].index].area_e,
                               atol=1e-7, rtol=0)
    data["survey_base_2010"] = data.index.get_level_values("id").map(panel.loc[2010].area_e)
    ids = data.loc[2011].index
    data["survey_open_2011"] = data.index.get_level_values("id").map(data.loc[2011].area_b)
    data["sim_prev_close"] = data.groupby(level="id").operated.shift().fillna(data.model_base_2010)
    data["obs_prev_close"] = data.groupby(level="id").area_e.shift()
    data.loc[pd.IndexSlice[2011, :], "obs_prev_close"] = data.loc[2011].survey_base_2010.to_numpy()
    contracts = pd.read_csv(f"{prefix}_contracts.csv")
    assert contracts.end.eq(contracts.start).all()
    assert (contracts.area > 0).all()
    orders = pd.read_csv(f"{prefix}_orders.csv")
    assert not orders.duplicated(["year", "id"]).any()
    external = pd.read_csv(f"{prefix}_external.csv")
    synthetic = (pd.read_csv(prefix.parents[1]/"succession_summary.csv")
                 .query("seed == @seed").set_index("year"))
    summary = []
    for y in YEARS:
        h = data.loc[y].copy()
        new = contracts[contracts.start.eq(y)]
        old = contracts[contracts.start.lt(y) & contracts.end.ge(y - 1)]
        expired = old[old.end.lt(y)]
        carry = old[old.end.ge(y)]
        prev_in, prev_out = lease_position(old, ids)
        exp_in, exp_out = lease_position(expired, ids)
        carry_in, carry_out = lease_position(carry, ids)
        new_in, new_out = lease_position(new, ids)
        mask = pd.IndexSlice[y, :]
        fields = dict(open_pre_expiry=h.base_area + prev_in - prev_out,
                      expired_in=exp_in, expired_out=exp_out, carry_in=carry_in, carry_out=carry_out,
                      pre_match=h.base_area + carry_in - carry_out, new_in=new_in, new_out=new_out)
        for col, values in fields.items():
            data.loc[mask, col] = values.to_numpy()
        np.testing.assert_allclose(h.sim_prev_close, fields["open_pre_expiry"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(h.operated, fields["pre_match"] + new_in - new_out, atol=1e-6, rtol=0)
        np.testing.assert_allclose(h.rented_in, carry_in + new_in, atol=1e-6, rtol=0)
        np.testing.assert_allclose(h.rented_out, carry_out + new_out, atol=1e-6, rtol=0)
        assert h.cultivated.le(h.operated + EPS).all()
        outside = external[external.year.eq(y)]
        assert abs(new.loc[new.channel.eq("external_supply"), "area"].sum() - outside.rented_out.sum()) < 1e-6
        assert abs(new.loc[new.channel.eq("external_demand"), "area"].sum() - outside.rented_in.sum()) < 1e-6
        sides = ["seller", "buyer"]
        pair = pd.concat([expired.groupby(sides).area.sum(), new.groupby(sides).area.sum()],
                         axis=1, join="inner")
        # The external supply side has no plot/slice; the same pair alone does not identify a plot.
        paired = expired.dropna(subset=["plot"]).merge(new.dropna(subset=["plot"]), on="plot")
        overlap = (paired[["slice_end_x", "slice_end_y"]].min(axis=1)
                   - paired[["slice_start_x", "slice_start_y"]].max(axis=1)).clip(lower=0)
        np.testing.assert_allclose(h.base_area, h.model_base_2010, atol=1e-6, rtol=0)
        buyers = orders.loc[orders.year.eq(y) & orders.side.eq("buy")].set_index("id")
        hh_buys = buyers.reindex(ids).quantity.fillna(0)
        np.testing.assert_allclose(h.demand.where(h.demand > 1e-8, 0), hh_buys, atol=1e-6, rtol=0)
        summary.append(dict(
            seed=seed, year=y, current_buy_households=int(hh_buys.gt(EPS).sum()),
            current_buy_order_mu=hh_buys.sum(), delivered_to_household_mu=new_in.sum(),
            unfilled_buy_mu=(hh_buys - new_in).clip(lower=0).sum(),
            buyers_with_delivery=int(new_in.gt(EPS).sum()),
            delivery_buyer_zero_sown_households=int((new_in.gt(EPS) & h.cultivated.le(EPS)).sum()),
            delivery_to_zero_sown_buyer_mu=new_in[h.cultivated.le(EPS)].sum(),
            signed_not_delivered_mu=0., old_contracts=len(expired), expired_mu=expired.area.sum(),
            signed_not_sown_synthetic_assignment_mu=synthetic.loc[y,"contracted_not_sown_mu"],
            leased_not_sown_synthetic_assignment_mu=synthetic.loc[y,"leased_not_sown_mu"],
            carry_contracts=len(carry), explicit_renewal_mu=0., new_contracts=len(new), new_mu=new.area.sum(),
            same_pair_new_contract_overlap_upper_mu=pair.min(axis=1).sum(),
            synthetic_slice_relet_same_buyer_mu=overlap[paired.buyer_x.eq(paired.buyer_y)].sum(),
            synthetic_slice_relet_other_buyer_mu=overlap[paired.buyer_x.ne(paired.buyer_y)].sum(),
            external_supply_mu=outside.rented_out.sum(), external_demand_mu=outside.rented_in.sum(),
            household_net_mu=(new_in - new_out).sum(),
            sim_previous_to_open_max_error=(h.sim_prev_close - fields["open_pre_expiry"]).abs().max(),
            sim_prematch_to_close_max_error=(h.operated-fields["pre_match"]-new_in+new_out).abs().max(),
        ))
    return data.reset_index(), pd.DataFrame(summary)


def aggregates(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stocks, flows, quality, groups, transitions = [], [], [], [], []
    sides = ["area_in1", "area_de2"]
    components = data[["sow_food", "sow_cash", "sow_oother"]]
    active = data.sow_total.gt(0) | components.gt(0).any(axis=1)
    zero = data.area_e.eq(0) & data.sow_total.eq(0) & ~active
    for (s, y), annual in data.groupby(["seed", "year"], sort=True):
        q = annual.index
        base = dict(seed=s, year=y)
        quality.append(dict(**base, cohort=len(annual), open_missing=int(annual.area_b.isna().sum()),
                            open_negative=int(annual.area_b.lt(0).sum()), open_zero=int(annual.area_b.eq(0).sum()),
                            close_missing=int(annual.area_e.isna().sum()), close_negative=int(annual.area_e.lt(0).sum()),
                            close_reported_zero=int(annual.area_e.eq(0).sum()), close_verified_zero=int(zero[q].sum()),
                            zero_positive_sown=int((annual.area_e.eq(0) & active[q]).sum()),
                            missing_positive_sown=int((annual.area_e.isna() & active[q]).sum()),
                            zero_unresolved=int((annual.area_e.eq(0) & ~active[q] & ~zero[q]).sum())))
        for basis, condition in (("historical_positive", annual.area_e.gt(0)),
                                 ("verified_zero_inclusive", annual.area_e.gt(0) | zero[q])):
            h = annual[annual.area_b.ge(0) & condition]
            tags = dict(**base, basis=basis)
            stock_cols = ("area_b area_e base_area rights sim_prev_close open_pre_expiry expired_in expired_out "
                          "carry_in carry_out pre_match offer demand new_in new_out operated cultivated grain crop").split()
            stocks.append(dict(**tags, n=len(h), **h[stock_cols].sum().to_dict(),
                               close_signed_gap=(h.operated-h.area_e).sum(),
                               close_abs_rel_error=abs(h.operated.sum()/h.area_e.sum()-1),
                               **{f"{c}_n": int(h[c].notna().sum()) for c in
                                  ["obs_prev_close", "area_b1", "area_e1", "sow_total"]},
                               **h[["obs_prev_close", "area_b1", "area_e1", "sow_total"]].sum(min_count=1).to_dict(),
                               obs_open_previous_paired_n=int(h.obs_prev_close.ge(0).sum()),
                               obs_open_previous_paired_gap=(h.area_b-h.obs_prev_close).loc[h.obs_prev_close.ge(0)].sum(),
                               **distribution(h)))
            for key in ["model_base_2010", "survey_base_2010", "survey_open_2011"]:
                small_missing = 0 < h[key].isna().sum() < MIN_N
                small_tail = 0 < h[key].ge(30).sum() < MIN_N
                for bins, labels in [
                    ([0,5,15,np.inf], ["lt5","5_to_lt15","ge15"]),
                    ([0,5,15,30,np.inf], ["lt5","5_to_lt15","15_to_lt30","ge30"]),
                ]:
                    bucket = pd.cut(h[key].where(h[key].ge(0)), bins, right=False, labels=labels)
                    categories = bucket.cat.add_categories("unknown").fillna("unknown")
                    for label, group in h.groupby(categories, observed=True):
                        row = dict(**tags, grouping=key, bins=len(labels), group=label, n=len(group))
                        complementary = ((small_tail and label in ("15_to_lt30","ge30"))
                                         or (small_missing and label in ("ge15","15_to_lt30","ge30")))
                        if len(group) >= MIN_N and not complementary:
                            row.update(distribution(group))
                        groups.append(row)
            for name, frame, opening, closing in [
                ("survey_within_year", h, "area_b", "area_e"),
                ("survey_across_year", h, "obs_prev_close", "area_e"),
                ("model_across_year", h, "sim_prev_close", "operated"),
                ("model_after_expiry", h, "pre_match", "operated"),
            ]:
                def band(values: pd.Series) -> pd.Series:
                    return pd.cut(values.where(values.ge(0)), [-EPS,0,5,15,30,np.inf], right=False,
                                  labels=["zero","lt5","5_to_lt15","15_to_lt30","ge30"]).mask(values.eq(0), "zero")
                for (f, t), value in frame.groupby([band(frame[opening]), band(frame[closing])], observed=True).size().items():
                    transitions.append(dict(**tags, account=name, from_size=f, to_size=t,
                                            count=int(value) if value >= MIN_N else f"<{MIN_N}"))
            for scope, g in [(basis, h)]:
                flows.extend(flow_rows(g, dict(**base, scope=scope)))
        flows.extend(flow_rows(annual, dict(**base, scope="all_reporters")))
    return {k:pd.DataFrame(v) for k,v in dict(annual_stages=stocks, annual_flows=flows,
                                             missing_and_zero=quality, groups=groups, transitions=transitions).items()}


def distribution(group: pd.DataFrame) -> dict[str, float]:
    observed, modeled = group.area_e, group.operated
    gap = modeled - observed
    error = gap.sum()
    total = observed.sum()
    leave = ((error-gap)/(total-observed).replace(0,np.nan)).abs()
    return dict(survey_mu=total, model_mu=modeled.sum(), gap_mu=error,
                **{f"{tag}_p{q}": v.quantile(q/100) for tag,v in [("survey",observed),("model",modeled)]
                   for q in (25,50,75,90)}, tail_n_ge50=int(observed.ge(50).sum()),
                leave_one_signed_min_mu=round((error-gap).min(), -1),
                leave_one_signed_max_mu=round((error-gap).max(), -1),
                leave_one_min_abs_error_pct=round(100*leave.min(), 1),
                leave_one_max_abs_error_pct=round(100*leave.max(), 1))


def flow_rows(group: pd.DataFrame, tags: dict[str, int | str]) -> list[dict[str, int | float | str]]:
    fi, fo = group.area_in1, group.area_de2
    inc, dec = group.area_in, group.area_de
    # A blank child is logically zero only when its reported parent is explicitly zero.
    known_i = fi.where(fi.ge(0)).mask(fi.isna() & inc.eq(0), 0.)
    known_o = fo.where(fo.ge(0)).mask(fo.isna() & dec.eq(0), 0.)
    stock = group.area_b.ge(0)&group.area_e.ge(0)
    rows = []
    for name, inflow, outflow, same, both_parents in [
        ("both_components_reported",fi,fo,fi.ge(0)&fo.ge(0),False),
        ("both_with_parent_zero",known_i,known_o,known_i.notna()&known_o.notna(),False),
        ("both_with_parent_zero_and_stocks",known_i,known_o,known_i.notna()&known_o.notna()&stock,False),
        ("only_total_change_known",inc,dec,inc.ge(0)&dec.ge(0)&stock
         &~(known_i.notna()&known_o.notna()),True),
    ]:
        m = same
        g = group.loc[m]
        other = m & inc.ge(inflow) & dec.ge(outflow) & stock & (not both_parents)
        residual = group.area_e-group.area_b-(inflow-outflow)
        change = residual-(inc-inflow)+(dec-outflow)
        values = dict(survey_net_mu=(inflow-outflow)[m].sum(min_count=1),
                      survey_in_mu=inflow[m].sum(min_count=1), survey_out_mu=outflow[m].sum(min_count=1),
                      model_net_mu=(g.rented_in-g.rented_out).sum(min_count=1),
                      survey_stock_change_mu=(group.area_e-group.area_b)[m&stock].sum(min_count=1),
                      stock_change_after_net_mu=residual[m&stock].sum(min_count=1))
        if both_parents:
            values = {f"{k}_total_not_lease":v for k,v in values.items() if k!="model_net_mu"}
        rows.append(dict(**tags, convention=name, n=len(g), stock_n=int((m&stock).sum()),
                         survey_in_n=int(fi.ge(0).sum()), survey_out_n=int(fo.ge(0).sum()),
                         parent_zero_in_n=int((m&fi.isna()&inc.eq(0)).sum()),
                         parent_zero_out_n=int((m&fo.isna()&dec.eq(0)).sum()),
                         other_joint_n=int(other.sum()), **(values if len(g)>=MIN_N else {}),
                         **(dict(reported_other_in_mu=(inc-inflow)[other].sum(min_count=1),
                                 reported_other_out_mu=(dec-outflow)[other].sum(min_count=1),
                                 unknown_balance_after_other_mu=change[other].sum(min_count=1))
                            if other.sum()>=MIN_N else {})))
    return rows


def pool_cap_replay(project: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """Re-clear archived orders; round-trip parsing preserves byte-level price ordering."""
    from scipy.sparse import load_npz
    from modules.land_clearing_house import LandClearingHouse
    from scripts.run_national_household_portfolio import legacy

    with patch.object(legacy, "BASE", project/"results/diagnostics/national_formal_revision_20261118/run_01"):
        fee = .05 * legacy.load_base()[0]["environment"]["market"]["base_land_rent"]
    results = []
    for seed in SEEDS:
        prefix = archive_prefix(project, seed)
        space = json.loads(Path(f"{prefix}_space.json").read_bytes())
        households = set(space["coordinates"])
        space.update(index={k:i for i,k in enumerate(space["coordinates"])},
                     neighbors=load_npz(f"{prefix}_neighbors.npz"),search=True,fee=fee)
        orders, ext, locked = [pd.read_csv(f"{prefix}_{k}.csv",float_precision="round_trip")
                               for k in ("orders","external","contracts")]
        for year, group in orders.groupby("year",sort=True):
            live = group.drop(columns="year").to_dict("records")
            pool = (ext.loc[ext.year.eq(year)].rename(columns={"opening_area":"supply","demand_capacity":"demand"})
                    .set_index("region")[["supply","demand","reference_rent"]].to_dict("index"))
            clear = LandClearingHouse.clear_area_orders
            flags = dict(space=space,household_ids=households)
            base = pd.DataFrame(clear(live,year,1,external=pool,**flags))
            pd.testing.assert_frame_equal(base[locked.columns].fillna("").reset_index(drop=True),
                                          locked.loc[locked.start.eq(year)].fillna("").reset_index(drop=True),
                                          check_dtype=False,atol=1e-7,rtol=1e-10)
            expanded = {z:dict(v,supply=v["supply"]+sum(
                o["quantity"] for o in live if o["side"]=="buy" and o["id"] in households and o["region"]==z))
                for z,v in pool.items()}
            candidate = pd.DataFrame(clear(live,year,1,external=expanded,**flags))
            take = base.channel.eq("external_supply")
            new_take = candidate.channel.eq("external_supply")
            pd.testing.assert_frame_equal(base.loc[~take].reset_index(drop=True),
                                          candidate.loc[~new_take].reset_index(drop=True))
            before, after = base.loc[take,"area"].sum(), candidate.loc[new_take,"area"].sum()
            s_b = base.loc[take].groupby("buyer").area.sum()
            s_a = candidate.loc[new_take].groupby("buyer").area.sum()
            extra = s_a.sub(s_b,fill_value=0)
            survey = panel.loc[year]
            positive = survey.index[survey.area_b.ge(0)&survey.area_e.gt(0)]
            results.append(dict(seed=seed,year=year,archived_contracts_reproduced=len(base),
                                B0_external_delivered_mu=before, relaxed_external_delivered_mu=after,
                                extra_mu=after-before, extra_on_positive_reporters_mu=extra.reindex(positive).sum(),
                                newly_served_households=int((~s_a.index.isin(s_b.index)).sum()),
                                households_more_area=int(extra.gt(EPS).sum())))
    assert len(results)==len(YEARS)*len(SEEDS)
    return pd.DataFrame(results)


def observed_replay(project: Path, output: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace actual calls from the existing annual entry, without changing the baseline."""
    from core import v4_household
    from core.model import MainModel
    from modules.land_clearing_house import LandClearingHouse
    from scripts import run_empirical_succession

    def source(path: str) -> str:
        return subprocess.check_output(["git", "show", f"{FROZEN}:{path}"], cwd=ROOT).decode("utf-8")

    farm_file, market_file = "core/v4_household.py", "modules/land_clearing_house.py"
    farm_code, market_code = source(farm_file), source(market_file)
    farm_globals = v4_household.__dict__.copy()
    market_globals = {"__name__": "modules.land_clearing_house"}
    exec(compile(farm_code, farm_file, "exec"), farm_globals)
    exec(compile(market_code, market_file, "exec"), market_globals)
    farm = farm_globals["farm_batch"]
    market = market_globals["LandClearingHouse"].clear_area_orders
    markers = {
        tag: next(i for i,line in enumerate(market_code.splitlines(),1) if line.strip().startswith(fragment))
        for tag,fragment in {
            "edge":"ask,bid=seller['price'],buyer['price']-fee",
            "price":"if sr[i]<=1e-8 or seller['id']==buyer['id']",
            "external":"for order in sorted(eligible,",
            "pool_gate":"if room<=1e-8:break",
        }.items()
    }
    planner: list[dict[str, int | float | str]] = []
    clearing: list[dict[str, int | float | str]] = []
    features: dict[str, set[str]] = {}
    external_gate: dict[str, str] = {}
    replay_deals: list[dict] = []
    replay_orders: list[dict] = []
    replay_pre: list[tuple[int, str, float]] = []

    def farm_trace(frame, event: str, result):
        if event != "return" or result is None:
            return farm_trace
        d = frame.f_locals
        if not d["trade"]:
            return farm_trace
        parent = frame.f_back.f_back.f_locals
        y, ids = parent["year"], parent["ids"]
        replay_pre.extend(zip([y]*len(ids), ids, d["land"]))
        relaxed = farm(d["state"], d["fit"], d["u"], stage=d["stage"], trade=True,
                       incentive=d["incentive"], grain_aid=d["grain_aid"],
                       diagnostic={**d["flags"], "rent_in": False})["demand"]
        unfilled = result["demand"] <= 1e-8
        buckets = [
            ("posted_buy", ~unfilled),
            ("zero_selected_plan", d["chosen"] <= 1e-8),
            ("reference_rent_nonpositive", d["rent"] <= 0.),
            ("rent_in_logit", ~d["can_buy"]),
            ("no_physical_headroom", d["unbounded"] <= d["full"]+EPS),
            ("cannot_fund_next_tranche", (d["with_rent"] <= d["full"][:,None]+EPS).all(axis=1)),
            ("cash_return_and_labor_at_reference", (d["grow"] <= d["full"][:,None]+EPS).all(axis=1)),
            ("expanded_plan_not_selected", np.ones(len(ids),dtype=bool)),
        ]
        left = np.ones(len(ids),dtype=bool)
        for reason, eligible in buckets:
            m = left & eligible
            left &= ~m
            planner.append(dict(seed=seed, year=y, stage=reason, households=int(m.sum()),
                                base_demand_mu=result["demand"][m].sum(),
                                no_logit_demand_mu=relaxed[m].sum(),
                                new_orders_in_isolated_no_logit=int((m&unfilled&(relaxed>1e-8)).sum())))
        return farm_trace

    def market_trace(frame, event: str, result):
        d = frame.f_locals
        if event == "line":
            line = frame.f_lineno
            if line == markers["price"]:
                buyer, seller = d["buyer"]["id"], d["seller"]["id"]
                if buyer != seller:
                    seen = features.setdefault(buyer, set())
                    seen.add("edge")
                    if d["sr"][d["i"]] > 1e-8:
                        seen.add("remaining")
                        if d["ask"] <= d["bid"]:
                            seen.add("quotes")
                            if max(d["lower"],d["ask"]) <= min(d["upper"],d["bid"]):
                                seen.add("region_price")
            elif line == markers["external"] and d["side"] == "buy":
                external_gate.update({o["id"]: "pool_capacity_not_reached" for o in d["eligible"]})
            elif line == markers["pool_gate"] and d["side"] == "buy":
                external_gate[d["order"]["id"]] = (
                    "pool_capacity_exhausted" if d["room"] <= 1e-8
                    else "pool_price" if d["order"]["price"] < d["rent"] else "pool_price_passed"
                )
        if event == "return" and result is not None:
            y = d["year"]
            orders = d["orders"]
            replay_deals.extend(result)
            replay_orders.extend(dict(year=y, **o) for o in orders)
            buyer_orders = [o for o in orders if o["side"]=="buy" and o["id"] in d["household_ids"]]
            # Only lift the external supply ceiling. Orders, local matching and pool price stay fixed.
            pool = {z:dict(v, supply=v["supply"]+sum(o["quantity"] for o in buyer_orders if o["region"]==z))
                    for z,v in d["external"].items()}
            expanded = market(orders,y,d["term"],space=d["space"],external=pool,
                              household_ids=d["household_ids"])
            in0, _ = lease_position(pd.DataFrame(result, columns=["buyer","seller","area"]),
                                    pd.Index([o["id"] for o in buyer_orders]))
            in1, _ = lease_position(pd.DataFrame(expanded, columns=["buyer","seller","area"]), in0.index)
            supply_regions = {o["region"] for o in orders if o["side"] == "sell" and o["quantity"]>1e-8}
            grouped: dict[tuple[str,str], list[float]] = {}
            for o in buyer_orders:
                who = o["id"]
                remaining = max(0.,o["quantity"]-in0[who])
                seen = features.get(who,set())
                if remaining <= EPS:
                    reasons = ("fully_delivered","not_needed")
                else:
                    local = ("no_release_in_region" if o["region"] not in supply_regions else
                             "no_synthetic_edge" if "edge" not in seen else
                             "local_stock_claimed_first" if "remaining" not in seen else
                             "noncrossing_quotes_after_fee" if "quotes" not in seen else
                             "common_regional_price" if "region_price" not in seen else
                             "partial_local_or_external_delivery")
                    reasons = (local,external_gate.get(who,"not_eligible_or_no_region_pool"))
                grouped.setdefault(reasons,[0.,0.,0.,0.,0.])
                grouped[reasons] = [a+b for a,b in zip(grouped[reasons],
                                    [1.,o["quantity"],in0[who],remaining,in1[who]-in0[who]])]
            for (local,pool_stage), vals in grouped.items():
                clearing.append(dict(seed=seed,year=y,local_stage=local,pool_stage=pool_stage,
                                     **dict(zip(["households","order_mu","delivered_mu","residual_mu",
                                                 "extra_if_supply_ceiling_removed_mu"],vals))))
            features.clear()
            external_gate.clear()
        return market_trace

    def trace(frame, event: str, arg):
        if frame.f_code is farm.__code__:
            frame.f_trace_lines = False
            return farm_trace
        if frame.f_code is market.__code__:
            return market_trace
        return None

    annual_step = MainModel._step_area_economy

    def trace_annual(model):
        try:
            sys.settrace(trace)
            return annual_step(model)
        finally:
            sys.settrace(None)

    with (
        TemporaryDirectory(prefix=".private-replay-", dir=output) as tmp,
        patch.object(v4_household, "farm_batch", farm),
        patch.object(LandClearingHouse, "clear_area_orders", staticmethod(market)),
        patch.object(MainModel, "_step_area_economy", trace_annual),
    ):
        run_empirical_succession.run(project, Path(tmp)/"run", [seed])

    prefix = archive_prefix(project, seed)
    saved_orders = pd.read_csv(f"{prefix}_orders.csv")
    pd.testing.assert_frame_equal(pd.DataFrame(replay_orders),saved_orders,check_dtype=False,atol=1e-7,rtol=1e-10)
    saved_deals = pd.read_csv(f"{prefix}_contracts.csv")
    pd.testing.assert_frame_equal(pd.DataFrame(replay_deals)[saved_deals.columns].fillna(""),saved_deals.fillna(""),
                                  check_dtype=False,atol=1e-7,rtol=1e-10)
    accounts = pd.read_csv(f"{prefix}_accounts.csv").query("kind == 'household'").set_index(KEY)
    pref = pd.DataFrame(replay_pre,columns=[*KEY,"pre_match"]).set_index(KEY)
    np.testing.assert_allclose(pref.pre_match,accounts.loc[pref.index].base_area,atol=1e-7,rtol=0)
    return pd.DataFrame(planner), pd.DataFrame(clearing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--replay", nargs="+", metavar="SEED", type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path, metavar="NEW_DIR",
                        default=ROOT / "docs/全国经营接续/开发比较/土地诊断/复算",
                        help="write to a new directory; never overwrite prior results")
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.absolute()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}. Choose a new directory with --output.")
    panel = raw_panel(project)
    data, market = [], []
    for seed in SEEDS:
        h, book = ledger(project,seed,panel)
        data.append(h.assign(seed=seed))
        market.append(book)
    combined = pd.concat(data,ignore_index=True)
    survey_static = combined.loc[combined.seed.eq(301)].set_index(KEY)[SURVEY]
    for seed in SEEDS:
        pd.testing.assert_frame_equal(survey_static,combined.loc[combined.seed.eq(seed)].set_index(KEY)[SURVEY])
    tables = aggregates(combined)
    tables["settlement"] = pd.concat(market,ignore_index=True)
    tables["pool_cap_replay"] = pool_cap_replay(project,panel)
    for name,table in tables.items():
        table.to_csv(output/f"{name}.csv",index=False,float_format="%.8f")
    print(tables["annual_stages"].groupby(["basis","year"])[["n","area_e","operated","close_signed_gap"]].mean())
    if args.replay:
        replay = [observed_replay(project,output,seed) for seed in args.replay]
        for name,index in [("planner_replay",0),("clearing_replay",1)]:
            pd.concat([r[index] for r in replay],ignore_index=True).to_csv(
                output/f"{name}.csv",index=False,float_format="%.8f")


if __name__ == "__main__":
    main()
