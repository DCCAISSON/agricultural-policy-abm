"""Render the locked national replay and the separately frozen 7a comparison."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(replay: Path, out: Path):
    gates = pd.read_csv(replay/"candidate_historical_gates.csv")
    years = gates.groupby("year")
    fig, (left, right) = plt.subplots(1,2,figsize=(8.4,3.3),layout="constrained")
    for key, label, color in [
        ("historical_four","Original 4 gates","#4076aa"),
        ("historical_five","All 5 gates","#173e65"),
    ]:
        n=years[key].sum()
        left.plot(n.index,n,marker="o",label=label,color=color)
    left.set(ylim=(0,21),ylabel="Passing histories (out of 20)",title="A. Joint behavior tests")
    for key,label,color in [
        ("grain_mu_times_relative_error","Grain sown area","#4076aa"),
        ("closing_physical_mu_relative_error","Physical operated area","#173e65"),
    ]:
        y=years[key].mean()*100
        right.plot(y.index,y,marker="o",label=label,color=color)
        right.fill_between(y.index,years[key].quantile(.1)*100,years[key].quantile(.9)*100,color=color,alpha=.12)
    right.axhline(10,linestyle="--",color="#6a7179",linewidth=1,label="10% criterion")
    right.set(ylabel="Absolute relative error (%)",title="B. Area errors by year")
    for ax in (left,right):
        ax.set_xticks(list(years.groups))
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(out,dpi=220,metadata={"Software":"Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    plot(args.replay,args.output)
