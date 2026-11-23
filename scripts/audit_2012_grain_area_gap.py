"""Reconcile the 2012 matched households against two archived P0 production ledgers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_household_portfolio_behavior import OUT, V1_OUT, areas, v1_run
from scripts.prepare_household_portfolio_reference import prepare


def main():
    # Reread the authorized source and assert byte equality to the frozen reference.
    prepare()
    ref = pd.read_csv(OUT / "reference/reference_households.csv")
    obs = (ref.loc[ref.year.eq(2012)].set_index("actor_id", verify_integrity=True)
           [["grain_area", "cash_area"]].dropna().rename(columns={"grain_area": "og", "cash_area": "oc"}))
    assert len(obs) == 527 and (obs >= 0).all().all()
    receipt = json.loads((V1_OUT / "behavior_gate.json").read_bytes())
    benchmarks = pd.read_csv(V1_OUT / "behavior_2012.csv").query("rule == 'v1'").set_index("seed", verify_integrity=True)
    parts = ["obs_only_mu", "sim_only_mu", "area_effect_mu", "share_effect_mu"]
    ledgers, totals = [], []
    for seed in (103, 109):
        paths = {r["rule"]: Path(r["episode"]) for r in receipt["episodes"] if r["seed"] == seed}
        path = paths["v1"]
        b = benchmarks.loc[seed]
        assert path.parent.name == b.run and v1_run(path, seed, "v1", paths["old"]) == receipt["version"]
        sim = areas(path, 2012).rename(columns={"grain": "mg", "cash": "mc"})
        x = obs.join(sim, how="left", validate="one_to_one")
        assert np.isfinite(x.to_numpy()).all() and (x >= 0).all().all()
        # No-area households are zero only when the actual ledger also reports no output.
        harvest = pd.DataFrame.from_dict(json.loads((path / "year_02/actual_harvests.json").read_bytes()), orient="index")
        x["O"], x["M"] = x.og + x.oc, x.mg + x.mc
        no_area = x.index[x.M.eq(0)]
        assert (harvest.loc[no_area, ["grain_physical_kg", "cash_output_model_units"]] == 0).all().all()
        planted_o, planted_m = x.O.gt(0), x.M.gt(0)
        both = planted_o & planted_m
        assert (len(x), planted_o.sum(), planted_m.sum(), both.sum()) == (
            b.obs_n, b.obs_positive_n, b.sim_positive_n, b.common_positive_n)
        x["obs_share"], x["sim_share"] = x.og / x.O.where(planted_o), x.mg / x.M.where(planted_m)
        x["participation"] = np.select([both, planted_o, planted_m], ["both", "obs_only", "sim_only"], default="neither")
        x["obs_only_mu"] = x.og.where(planted_o & ~planted_m, 0.)
        x["sim_only_mu"] = -x.mg.where(planted_m & ~planted_o, 0.)
        x["area_effect_mu"] = ((x.O - x.M) * (x.obs_share + x.sim_share) / 2).where(both, 0.)
        x["share_effect_mu"] = ((x.obs_share - x.sim_share) * (x.O + x.M) / 2).where(both, 0.)
        x["grain_gap_mu"] = x.og - x.mg
        x["residual_mu"] = x.grain_gap_mu - x[parts].sum(axis=1, skipna=False)
        assert x.residual_mu.abs().max() < 1e-9
        s = x[parts].sum()
        gap = x.og.sum() - x.mg.sum()
        assert abs(s.sum() - gap) < 1e-8
        decision = pd.DataFrame(json.loads((path / "year_02/decision_trace.json").read_bytes()))
        x = x.join(decision.set_index("operator_id", verify_integrity=True)[["action", "maintain_grain_area_mu"]])
        missed = x.participation.eq("obs_only")
        zero_plan = missed & x.action.eq("maintain_grain") & x.maintain_grain_area_mu.eq(0)
        totals.append(dict(seed=seed, run=path.parent.name, households=len(x),
                           **{f"{v}_n": int(x.participation.eq(v).sum()) for v in ("obs_only","sim_only","both","neither")},
                           observed_grain_mu=x.og.sum(), simulated_grain_mu=x.mg.sum(), **s.to_dict(),
                           difference_mu=gap, sum_components_mu=s.sum(), max_household_residual_mu=x.residual_mu.abs().max(),
                           obs_only_exit_n=int((missed & x.action.eq("exit")).sum()),
                           obs_only_no_decision_n=int((missed & x.action.isna()).sum()),
                           obs_only_zero_grain_plan_n=int(zero_plan.sum()), obs_grain_on_zero_grain_plan_mu=x.loc[zero_plan,"og"].sum()))
        ledgers.append(x.assign(seed=seed))
    dest = V1_OUT.parent / "national_v1_area_gap_2012"
    private = dest / "PRIVATE_DO_NOT_SHARE"
    private.mkdir(parents=True, exist_ok=True)
    pd.concat(ledgers).to_csv(private / "same_household.csv", encoding="utf-8-sig")
    result = pd.DataFrame(totals)
    result.to_csv(dest / "grain_area_gap.csv", index=False, encoding="utf-8-sig")
    print(result.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
