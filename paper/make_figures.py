"""Regenerate the paper figures from the released, non-identifying tables."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

P = Path(__file__).resolve().parent
E = P.parent / "evidence"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"],
                     "axes.unicode_minus": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(8.6, 2.9))
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
for x,y,label in [(0.15,.71,"基础支持公告"),(.50,.71,"选择与交易"),(.85,.71,"实种与收获"),
                  (.85,.24,"收入与财政结算"),(.50,.24,"下年经营条件"),(.15,.24,"可选政府学习")]:
    ax.text(x,y,label,ha="center",va="center",color="#19374C",fontsize=13,
            bbox=dict(boxstyle="square,pad=0.48",fc="white",ec="#386581",lw=1.3))
for a,b,dash in [((.25,.71),(.39,.71),False),((.60,.71),(.75,.71),False),
                 ((.85,.59),(.85,.37),False),((.72,.24),(.62,.24),False),
                 ((.50,.36),(.50,.58),False),((.38,.24),(.27,.24),True),
                 ((.15,.36),(.15,.58),True)]:
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.5,color="#386581",
        linestyle="--" if dash else "-"))
fig.savefig(P/"fig1.png",dpi=190,bbox_inches="tight"); plt.close(fig)

cells=pd.read_csv(E/"synthetic_cells.csv")
fig,axes=plt.subplots(1,2,figsize=(8.1,3.7),layout="constrained")
norm=Normalize(0,cells[["identified_mu","unknown_mu"]].max().max())
for ax,col,title in zip(axes,["identified_mu","unknown_mu"],["作物可定位","作物不可定位"]):
    grid=np.full((int(cells.y.max())+1,int(cells.x.max())+1),np.nan)
    for x,y,m in cells[["x","y",col]].itertuples(index=False,name=None):
        grid[int(y),int(x)]=m
    im=ax.imshow(grid,origin="lower",cmap="Blues",norm=norm,interpolation="nearest")
    ax.set(title=title,xlabel="合成网格横坐标",ylabel="合成网格纵坐标")
fig.colorbar(im,ax=axes,label="实耕面积（亩／单元）",shrink=.8)
fig.savefig(P/"fig2.png",dpi=195); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,3.1),layout="constrained")
for i,y in enumerate((2011,2012)):
    d=pd.read_csv(E/f"behavior_{y}.csv")
    rates=[d.obs_mixed_rate.iloc[0],d[d.rule=="old"].sim_mixed_rate.mean(),
           *d[d.rule=="portfolio"].sort_values("seed").sim_mixed_rate]
    for j,(rate,label,color) in enumerate(zip(rates,["调查","原规则（两次均为零）","新规则103","新规则109"],
                                                ["#20394d","#a7b9c4","#3c769a","#59978e"])):
        x=i+(j-1.5)*.19
        ax.bar(x,rate*100,width=.17,color=color,label=label if i==0 else None)
        ax.text(x,rate*100+1,f"{rate*100:.1f}",ha="center",fontsize=9)
ax.set(xticks=[0,1],xticklabels=["2011年","2012年"],ylabel="年度兼种率（%）",ylim=(0,75))
ax.spines[["top","right"]].set_visible(False)
ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.5,-.14),frameon=False)
fig.savefig(P/"fig3.png",dpi=195); plt.close(fig)
