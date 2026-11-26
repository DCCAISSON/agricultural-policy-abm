"""Recalculate succession bounds on frozen B0 accounts, including a rename risk case."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.empirical_succession import SyntheticLandLedger


FIELDS=["strict_succession_lower_mu","strict_succession_upper_mu","prior_crop_unknown_mu",
        "prior_external_pool_unknown_mu","grain_on_prior_unknown_lower_mu","grain_on_prior_unknown_upper_mu"]
TOTALS=["contracted_mu","contract_rent_yuan","annual_grain_mu_times","grain_kg","paid_yuan","physical_operated_mu"]


def rebuild(destination: Path, runs: list[Path]) -> None:
    audits=[]
    for run in runs:
        private=run/"PRIVATE_DO_NOT_SHARE"
        base=pd.read_csv(private/"baseline.csv")
        reference=pd.read_csv(run/"succession_summary.csv").set_index(["seed","year"])
        for file in sorted(private.glob("*_both_*_accounts.csv")):
            stem=file.name.removesuffix("_accounts.csv")
            seed=int(stem.split("_")[0])
            actors=pd.read_csv(file)
            contracts=pd.read_csv(private/f"{stem}_contracts.csv")
            external=pd.read_csv(private/f"{stem}_external.csv")
            keys=sorted(set(actors.id)|set(base.actor))
            mapping=dict(zip(keys,reversed(keys)))
            original=SyntheticLandLedger(base.to_dict("records"))
            renamed=SyntheticLandLedger(base.assign(actor=base.actor.replace(mapping)).to_dict("records"))
            for year,world in actors.groupby("year"):
                fiscal=reference.loc[(seed,year),"budget_yuan"]
                lease=contracts.loc[contracts.start.le(year)&contracts.end.ge(year)]
                outside=external.loc[external.year.eq(year)]
                ext=outside.to_dict("records")
                _,a=original.advance(year,world.to_dict("records"),lease.to_dict("records"),ext,fiscal)
                altered=lease.assign(**{col:lease[col].replace(mapping) for col in ("seller","buyer","plot")})
                _,b=renamed.advance(year,world.assign(id=world.id.replace(mapping)).to_dict("records"),
                                    altered.to_dict("records"),ext,fiscal)
                np.testing.assert_allclose([a[k] for k in TOTALS],reference.loc[(seed,year),TOTALS].astype(float),
                                           rtol=1e-10,atol=1e-7)
                np.testing.assert_allclose([a[k] for k in FIELDS+TOTALS],[b[k] for k in FIELDS+TOTALS],
                                           rtol=1e-10,atol=1e-7)
                if not 0<=a[FIELDS[0]]<=a[FIELDS[1]]+1e-7:
                    raise ValueError(f"Invalid overlap bounds for {seed}/{year}")
                audits.append(dict(seed=seed,year=year,**{k:a[k] for k in FIELDS+TOTALS},
                    old_sorted_point_mu=a["strict_synthetic_succession_mu"],
                    renamed_sorted_point_mu=b["strict_synthetic_succession_mu"]))
    frame=pd.DataFrame(audits).sort_values(["seed","year"])
    expected={(y,s) for y in range(2011,2016) for s in range(301,321)}
    if set(zip(frame.year,frame.seed))!=expected or len(frame)!=len(expected):
        raise ValueError("Not all predeclared B0 paths were present")
    if destination.exists():
        raise FileExistsError(destination)
    frame.to_csv(destination,index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination",type=Path)
    p.add_argument("runs",type=Path,nargs="+")
    a=p.parse_args()
    rebuild(a.destination,a.runs)
