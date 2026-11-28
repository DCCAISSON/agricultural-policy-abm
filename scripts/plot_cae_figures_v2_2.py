"""Generate 5 publication-grade 300-dpi Chinese academic figures for《中国工程科学》.

Rules strictly enforced:
- SimSun (Chinese) + Times New Roman (digits/symbols) with `axes.unicode_minus = False`.
- ZERO English abbreviations or internal code/script identifiers in any text, label, legend, or title.
- All empirical/simulation data points read directly from `docs/Mesa农业经营接续V2重构/结果_V2_2净版/*.csv`.
"""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

WT_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = WT_ROOT / "docs/Mesa农业经营接续V2重构/结果_V2_2净版"
FIG_DIR = WT_ROOT / "docs/Mesa农业经营接续V2重构/图表_V2_2"

C_NAVY = "#1F4E8C"
C_BLUE = "#3B75AF"
C_TEAL = "#1F8A70"
C_AMBER = "#D9822B"
C_RED = "#C04851"
C_SLATE = "#4A5A6A"
C_LIGHT_BG = "#F5F7FA"
C_CARD_BLUE = "#EBF2FA"
C_CARD_TEAL = "#EAF6F3"


def _r1(val: float) -> str:
    return str(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _parse_ci(ci_str: str) -> tuple[float, float]:
    s = str(ci_str).strip().strip("[]")
    parts = [float(x.strip()) for x in s.split(",")]
    return parts[0], parts[1]


def _draw_box(ax, x, y, w, h, *, fc="#FFFFFF", ec=C_SLATE, lw=1.3, r=0.015, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _draw_arrow(ax, x1, y1, x2, y2, *, color=C_SLATE, lw=1.7, rad=0.0, both=False, zorder=5):
    style = "<|-|>" if both else "-|>"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(arr)


def plot_fig1_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Main title banner
    ax.text(
        0.50, 0.962,
        "农业经营接续、财政支持传导与动态调控系统总体机理框架",
        ha="center", va="center", fontsize=16.0, fontweight="bold", color="#14233C",
    )
    ax.text(
        0.50, 0.927,
        "严格遵循 2010 年及以前微观数据校准边界与双边实物守恒约束",
        ha="center", va="center", fontsize=10.5, color=C_SLATE,
    )

    # Three column outer containers
    _draw_box(ax, 0.015, 0.04, 0.275, 0.85, fc=C_LIGHT_BG, ec="#CBD5E1", lw=1.2, r=0.016, zorder=1)
    _draw_box(ax, 0.308, 0.04, 0.398, 0.85, fc="#F2F6FB", ec="#93B4D8", lw=1.4, r=0.016, zorder=1)
    _draw_box(ax, 0.724, 0.04, 0.261, 0.85, fc=C_LIGHT_BG, ec="#CBD5E1", lw=1.2, r=0.016, zorder=1)

    ax.text(0.1525, 0.865, "一、微观调查与制度约束底座", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_NAVY)
    ax.text(0.507, 0.865, "二、双层嵌套多主体动态仿真核心", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_NAVY)
    ax.text(0.8545, 0.865, "三、核验审计与多维评估输出", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_NAVY)

    # Left Column - Top Card: Empirical Data
    _draw_box(ax, 0.028, 0.475, 0.249, 0.360, fc="#FFFFFF", ec=C_BLUE, lw=1.3, r=0.013, zorder=2)
    _draw_box(ax, 0.036, 0.779, 0.233, 0.046, fc=C_NAVY, ec=C_NAVY, lw=0.8, r=0.008, zorder=3)
    ax.text(0.1525, 0.802, "全国农村固定观察点实证底座", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#FFFFFF", zorder=4)
    left_top_items = [
        "1. 600户固定队列（4区 5,277.5 亩；\n   可流转确权地 4,009.2 亩）",
        "2. 2010年个体调查匹配（600户全覆盖）：\n   户主均龄53.19岁，外出217.6日/年",
        "3. 四区具名主粮与实测农机作业费\n   （小麦、稻谷、玉米、大豆）",
        "4. 主食自留消费量（596户实测）\n   与各省基期农产品产地价格",
    ]
    y_pos = 0.742
    for txt in left_top_items:
        ax.text(0.038, y_pos, txt, ha="left", va="top", fontsize=8.4, color="#1E293B", linespacing=1.32, zorder=4)
        y_pos -= 0.066

    # Left Column - Bottom Card: Institutional Rules
    _draw_box(ax, 0.028, 0.065, 0.249, 0.385, fc="#FFFFFF", ec=C_TEAL, lw=1.3, r=0.013, zorder=2)
    _draw_box(ax, 0.036, 0.394, 0.233, 0.046, fc=C_TEAL, ec=C_TEAL, lw=0.8, r=0.008, zorder=3)
    ax.text(0.1525, 0.417, "现实政策与资源边界约束", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#FFFFFF", zorder=4)
    left_bot_items = [
        "1. 存量托底：耕地地力保护补贴\n   （连续抛荒 1 年以上次年停发）",
        "2. 共同硬预算：年 15 万元当年清零\n   （规模门槛东中50 / 西30 / 东北100亩）",
        "3. 双轨流转契约期限结构：\n   小农1年 / 农场合作社3年 / 企业4年",
        "4. 货币可替代性与口粮购销价差：\n   春耕资金统一约束、自留优先于售粮",
    ]
    y_pos = 0.362
    for txt in left_bot_items:
        ax.text(0.038, y_pos, txt, ha="left", va="top", fontsize=8.3, color="#1E293B", linespacing=1.32, zorder=4)
        y_pos -= 0.072

    # Center Column - Top Card: Government Upper Layer
    _draw_box(ax, 0.324, 0.625, 0.366, 0.21, fc=C_NAVY, ec="#153560", lw=1.5, r=0.014, zorder=2)
    ax.text(0.507, 0.804, "上层：中央财政动态调控主体（跨期寻优与基准对照）", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#FFFFFF", zorder=3)
    _draw_box(ax, 0.338, 0.720, 0.338, 0.058, fc="#2B62A8", ec="#6FA3E0", lw=1.0, r=0.009, zorder=3)
    ax.text(0.507, 0.749, "仅观测上年度 4 区滞后公共指标\n（粮食耕种率、闲置率、流转率、资金紧缺户比例）", ha="center", va="center", fontsize=8.3, color="#FFFFFF", linespacing=1.28, zorder=4)
    _draw_box(ax, 0.338, 0.640, 0.338, 0.068, fc="#FFFFFF", ec="#8BB4E3", lw=1.0, r=0.009, zorder=3)
    ax.text(0.507, 0.674, "6类策略同场公平对比（30项共享预算方案）：\n基准无增补 | 统一固定 | 区域固定 | 透明规则 | 单期短视 | 跨期寻优", ha="center", va="center", fontsize=8.1, color="#14233C", linespacing=1.32, zorder=4)

    # Center Column - Bottom Card: Inner ABM Environment
    _draw_box(ax, 0.324, 0.065, 0.366, 0.475, fc=C_CARD_TEAL, ec=C_TEAL, lw=1.5, r=0.014, zorder=2)
    ax.text(0.507, 0.514, "下层：四区异质主体交互与双市场均衡仿真环境", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0F5243", zorder=3)

    # Sub-block A: Heterogeneous Actors
    _draw_box(ax, 0.338, 0.398, 0.338, 0.092, fc="#FFFFFF", ec=C_TEAL, lw=1.1, r=0.010, zorder=3)
    ax.text(0.507, 0.467, "异质农业经营主体同口径决策规划", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0F5243", zorder=4)
    ax.text(0.507, 0.427, "600户调查小农（老龄与务工退出惯性） | 23户家庭农场\n15个农民合作社（农机连片降本） | 3家农业企业（资本底线退出）", ha="center", va="center", fontsize=8.0, color="#1E293B", linespacing=1.32, zorder=4)

    # Sub-block B: Two Markets Side-by-Side
    _draw_box(ax, 0.338, 0.215, 0.163, 0.162, fc="#FFFFFF", ec=C_BLUE, lw=1.1, r=0.010, zorder=3)
    ax.text(0.4195, 0.354, "省内耕地流转市场", ha="center", va="center", fontsize=9.0, fontweight="bold", color=C_NAVY, zorder=4)
    ax.text(0.4195, 0.280, "· 自耕保留与挂牌转出\n· 序贯增量影子价竞租\n· 村社连片归集（可选）\n· 履约违约与流拍复耕", ha="center", va="center", fontsize=8.0, color="#1E293B", linespacing=1.36, zorder=4)

    _draw_box(ax, 0.513, 0.215, 0.163, 0.162, fc="#FFFFFF", ec=C_AMBER, lw=1.1, r=0.010, zorder=3)
    ax.text(0.5945, 0.354, "生产与省际购销模块", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#9C5610", zorder=4)
    ax.text(0.5945, 0.280, "· 凹二次规划粮经配置\n· 托管扣减农机作业费\n· 订单价外奖与省际套利\n  （浙闽苏粤与赣皖湘豫）", ha="center", va="center", fontsize=8.0, color="#1E293B", linespacing=1.36, zorder=4)

    # Sub-block C: Weather & Settlement
    _draw_box(ax, 0.338, 0.085, 0.338, 0.108, fc="#FFFFFF", ec=C_SLATE, lw=1.1, r=0.010, zorder=3)
    ax.text(0.507, 0.166, "年度收获结算与跨期状态更新", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1E293B", zorder=4)
    ax.text(0.507, 0.119, "区域气象冲击实现 → 先留口粮后售余粮 → 实支核定与\n双边商品余粮扣减 → 更新下期资金、储蓄与剩余租期", ha="center", va="center", fontsize=8.0, color="#334155", linespacing=1.32, zorder=4)

    # Arrows inside Center Column (Closed Loop)
    _draw_arrow(ax, 0.440, 0.625, 0.440, 0.542, color=C_NAVY, lw=2.0)
    ax.text(0.431, 0.584, "下达分区分工具\n补贴限额", ha="right", va="center", fontsize=8.3, fontweight="bold", color=C_NAVY)
    _draw_arrow(ax, 0.574, 0.542, 0.574, 0.625, color=C_TEAL, lw=2.0)
    ax.text(0.583, 0.584, "回传滞后公共指标\n与奖励回报", ha="left", va="center", fontsize=8.3, fontweight="bold", color=C_TEAL)

    # Arrows Left -> Center
    _draw_arrow(ax, 0.278, 0.66, 0.323, 0.72, color=C_BLUE, lw=1.8)
    _draw_arrow(ax, 0.278, 0.28, 0.323, 0.28, color=C_TEAL, lw=1.8)

    # Right Column - Top Card: Verification & Conservation Gate
    _draw_box(ax, 0.736, 0.465, 0.237, 0.370, fc="#FFFFFF", ec=C_TEAL, lw=1.3, r=0.013, zorder=2)
    _draw_box(ax, 0.744, 0.779, 0.221, 0.046, fc=C_TEAL, ec=C_TEAL, lw=0.8, r=0.008, zorder=3)
    ax.text(0.8545, 0.802, "经验核验与守恒审计门禁", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#FFFFFF", zorder=4)
    right_top_items = [
        "1. 历史同户独立核验：\n   2011—2015年一步与五年递推\n   （严格不用2011+数据反调参）",
        "2. 双边复式与实物守恒断言：\n   6类地块、租收支与财政闭合；\n   跨省买粮 ≡ 产区余粮扣减量",
        "3. 样本外留出评价隔离：\n   12开发 / 32训练 / 48独立评价",
    ]
    y_pos = 0.744
    for txt in right_top_items:
        ax.text(0.747, y_pos, txt, ha="left", va="top", fontsize=8.2, color="#1E293B", linespacing=1.32, zorder=4)
        y_pos -= 0.092

    # Right Column - Bottom Card: Mechanism & Policy Outputs
    _draw_box(ax, 0.736, 0.065, 0.237, 0.375, fc="#FFFFFF", ec=C_NAVY, lw=1.3, r=0.013, zorder=2)
    _draw_box(ax, 0.744, 0.384, 0.221, 0.046, fc=C_NAVY, ec=C_NAVY, lw=0.8, r=0.008, zorder=3)
    ax.text(0.8545, 0.407, "机制分解与政策效应输出", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#FFFFFF", zorder=4)
    right_bot_items = [
        "1. 流动性效应与土地重配剥离\n   （固定基准租约快照对照）",
        "2. 地租“价格—面积”对称分解\n   与同地块交集纯单价差",
        "3. 跨省订单套利规模与封顶效应\n   （宽松额度与实种上限对照）",
        "4. 六类策略四维并列公平评估\n   （产粮、实支、效率与停补留存）",
    ]
    y_pos = 0.354
    for txt in right_bot_items:
        ax.text(0.747, y_pos, txt, ha="left", va="top", fontsize=8.2, color="#1E293B", linespacing=1.32, zorder=4)
        y_pos -= 0.072

    # Arrows Center -> Right
    _draw_arrow(ax, 0.691, 0.66, 0.735, 0.66, color=C_TEAL, lw=1.8)
    _draw_arrow(ax, 0.691, 0.25, 0.735, 0.25, color=C_NAVY, lw=1.8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_fig2_stage1_validation(out_path: Path) -> None:
    df = pd.read_csv(RES_DIR / "stage1_cohort_validation_2011_2015.csv")
    v20 = df[df["arm"] == "v2_0_baseline"].sort_values("year")
    v22 = df[df["arm"] == "v2_2_clean_full"].sort_values("year")
    yrs = v22["year"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    fig.patch.set_facecolor("#FFFFFF")

    # Panel (a): Cultivating household count
    ax = axes[0]
    ax.plot(yrs, v22["survey_sown"], "o-", color="#111827", lw=2.2, ms=6, label="调查实测")
    ax.plot(yrs, v22["sim_sown_mean"], "s-", color=C_NAVY, lw=2.1, ms=5.5, label="改进方案（含个体表退耕）")
    ax.plot(yrs, v20["sim_sown_mean"], "^--", color=C_AMBER, lw=1.8, ms=5.5, label="基础方案（未含个体表退耕）")
    for x, y_obs, y_sim in zip(yrs, v22["survey_sown"], v22["sim_sown_mean"]):
        if x == 2012:
            ax.annotate(f"{int(y_obs)}", (x, y_obs), textcoords="offset points", xytext=(-6, -15), ha="center", fontsize=8.8, color="#111827")
            ax.annotate(_r1(float(y_sim)), (x, y_sim), textcoords="offset points", xytext=(10, 10), ha="center", fontsize=8.8, color=C_NAVY)
        elif x == 2015:
            ax.annotate(f"{int(y_obs)}", (x, y_obs), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.8, color="#111827")
            ax.annotate(_r1(float(y_sim)), (x, y_sim), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.8, color=C_NAVY)
    ax.set_title("(a) 参与农业播种户数轨迹 (2011—2015年)", fontsize=10.8, fontweight="bold", pad=8)
    ax.set_xlabel("年份 / 年", fontsize=9.8)
    ax.set_ylabel("播种户数 / 户", fontsize=9.8)
    ax.set_xticks(yrs)
    ax.set_ylim(410, 585)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    # Panel (b): Mixed grain+cash households
    ax = axes[1]
    ax.plot(yrs, v22["survey_mixed"], "o-", color="#111827", lw=2.2, ms=6, label="调查实测")
    ax.plot(yrs, v22["sim_mixed_mean"], "s-", color=C_TEAL, lw=2.1, ms=5.5, label="改进方案")
    ax.plot(yrs, v20["sim_mixed_mean"], "^--", color=C_AMBER, lw=1.8, ms=5.5, label="基础方案")
    near_lp = df[df["arm"] == "ablation_near_lp_gamma5"].sort_values("year")
    ax.plot(yrs, near_lp["sim_mixed_mean"], "d:", color=C_RED, lw=1.7, ms=5, label="对照：取消凹二次曲率")
    ax.set_title("(b) 粮经兼种户数与曲率消融对照", fontsize=10.8, fontweight="bold", pad=8)
    ax.set_xlabel("年份 / 年", fontsize=9.8)
    ax.set_ylabel("粮经兼种户数 / 户", fontsize=9.8)
    ax.set_xticks(yrs)
    ax.set_ylim(60, 345)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")

    # Panel (c): 2015 grain output & validation errors across specs
    ax = axes[2]
    d15 = df[df["year"] == 2015]
    order_arms = [
        ("v2_2_clean_full", "改进完整方案"),
        ("v2_0_baseline", "基础方案"),
        ("ablation_no_restart_cost", "取消复耕门槛"),
        ("ablation_no_rent_foodc_offset", "取消租金抵口粮"),
        ("ablation_no_machinery_service", "取消农机与雇工"),
    ]
    labels = [lbl for _, lbl in order_arms]
    p_errs = [float(d15.loc[d15["arm"] == a, "participation_err_pp"].iloc[0]) for a, _ in order_arms]
    g_errs = [float(d15.loc[d15["arm"] == a, "grain_tons_rel_err_pct"].iloc[0]) for a, _ in order_arms]
    y_idx = np.arange(len(labels))
    h_b = 0.34
    b1 = ax.barh(y_idx - h_b / 2, p_errs, height=h_b, color=C_NAVY, label="2015年参与误差（百分点）")
    b2 = ax.barh(y_idx + h_b / 2, g_errs, height=h_b, color=C_TEAL, label="2015年粮产相对误差（%）")
    for bar in list(b1) + list(b2):
        val = bar.get_width()
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha="left", fontsize=8.4)
    ax.set_yticks(y_idx)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 19.5)
    ax.set_title("(c) 2015年末端参与率与粮产误差对照", fontsize=10.8, fontweight="bold", pad=8)
    ax.set_xlabel("偏差幅度（百分点 或 %）", fontsize=9.8)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    for ax_i in axes:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig3_rent_and_efficiency(out_path: Path) -> None:
    df2 = pd.read_csv(RES_DIR / "exp2_rent_capitalization_v2_2.csv")
    df_iso = pd.read_csv(RES_DIR / "exp2_equal_spending_v2_2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.patch.set_facecolor("#FFFFFF")

    name_map = {
        "producer_pre": "播前种粮直补",
        "producer_post": "产后种粮直补",
        "scale_takeover_pre": "播前规模承接补贴",
        "scale_takeover_post": "产后规模承接补贴",
        "service_trusteeship": "农机生产托管补助",
        "order_farming_capped": "封顶订单粮食奖励",
    }

    # Panel (a): Symmetric Rent Decomposition (Price Effect vs Area Effect) & Common-Plot Unit Diff
    ax = axes[0]
    keys = ["producer_pre", "producer_post", "scale_takeover_pre", "scale_takeover_post", "service_trusteeship", "order_farming_capped"]
    sub = df2.set_index("policy").loc[keys]
    x = np.arange(len(keys))
    w = 0.35
    b_p = ax.bar(x - w / 2, sub["rent_price_effect_wan"], width=w, color=C_RED, label="纯单价效应（万元）")
    b_a = ax.bar(x + w / 2, sub["rent_area_effect_wan"], width=w, color=C_BLUE, label="租地面积效应（万元）")
    ax.axhline(0, color="#64748B", lw=0.9)
    for idx_k, (bar_p, cap_pct) in enumerate(zip(b_p, sub["pure_price_capitalization_pct"])):
        yv = bar_p.get_height()
        offset = 2.0 if yv >= 0 else -5.5
        ax.text(bar_p.get_x() + bar_p.get_width() / 2, yv + offset, f"{cap_pct:.1f}%", ha="center", va="bottom" if yv >= 0 else "top", fontsize=8.4, fontweight="bold", color=C_RED)
    ax.set_xticks(x)
    ax.set_xticklabels([name_map[k] for k in keys], rotation=18, ha="right", fontsize=9.0)
    ax.set_ylabel("八年租金变化分解量 / 万元", fontsize=9.8)
    ax.set_title("(a) 不同工具总租金变化的“价格—面积”对称分解\n（红字标注为纯单价地租资本化率）", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_ylim(-50, 68)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")

    # Panel (b): Equal Actual Spending Grain Efficiency (kg per 100 yuan) with 95% CI
    ax = axes[1]
    iso_keys = ["producer_pre", "scale_takeover_pre", "service_trusteeship", "order_farming_capped"]
    sub_iso = df_iso.set_index("policy").loc[iso_keys]
    means = sub_iso["kg_per_100yuan_spent"].to_numpy()
    los, his = zip(*[_parse_ci(s) for s in sub_iso["kg_per_100yuan_ci95"]])
    yerr = np.vstack([means - np.asarray(los), np.asarray(his) - means])
    colors_b = [C_BLUE, C_RED, C_TEAL, C_AMBER]
    xi = np.arange(len(iso_keys))
    bars = ax.bar(xi, means, width=0.52, color=colors_b, yerr=yerr, capsize=4, error_kw={"lw": 1.3, "ecolor": "#1E293B"})
    for idx_b, (bar, m_val, g_val) in enumerate(zip(bars, means, sub_iso["grain_gain_vs_p0_tons"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m_val + yerr[1][idx_b] + 1.8,
            f"{m_val:.1f} kg\n(增产+{g_val:.0f}t)",
            ha="center", va="bottom", fontsize=8.6, color="#1E293B",
        )
    ax.set_xticks(xi)
    ax.set_xticklabels([name_map[k] for k in iso_keys], rotation=12, ha="right", fontsize=9.2)
    ax.set_ylabel("每百元财政实支增粮效率 / (kg/百元)", fontsize=9.8)
    ax.set_title("(b) 统一实际支出口径下各工具增粮效率对比\n（含 95% 置信区间）", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_ylim(0, max(means) * 1.28)
    ax.grid(axis="y", ls=":", alpha=0.5)

    for ax_i in axes:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.4)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig4_village_and_arbitrage(out_path: Path) -> None:
    df3 = pd.read_csv(RES_DIR / "exp3_village_leases_v2_2.csv")
    df4 = pd.read_csv(RES_DIR / "exp4_interprovincial_order_arbitrage_v2_2.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2))
    fig.patch.set_facecolor("#FFFFFF")

    # Panel (a): Village Coordination & Dual-Track Lease Terms
    ax = axes[0]
    arms3 = [
        ("scale_alone_dual", "单用规模\n补贴"),
        ("scale_plus_village", "规模补贴+\n村社协调"),
        ("service_alone_dual", "单用生产\n托管"),
        ("service_plus_village", "生产托管+\n村社协调"),
        ("portfolio29_1yr", "区域组合\n(1年短租)"),
        ("portfolio29_dual", "区域组合\n(双轨租期)"),
        ("portfolio29_3yr", "区域组合\n(3年长租)"),
    ]
    sub3 = df3.set_index("arm").loc[[a for a, _ in arms3]]
    x3 = np.arange(len(arms3))
    gains3 = sub3["grain_gain_vs_p0_tons"].to_numpy()
    unm3 = sub3["unmatched_idle_mu_yr"].to_numpy()
    w3 = 0.48
    bars3 = ax.bar(x3, gains3, width=w3, color=[C_AMBER, C_TEAL, C_BLUE, C_NAVY, "#64748B", C_TEAL, "#1F4E8C"], label="八年累计增粮（t）")
    ax2 = ax.twinx()
    ax2.plot(x3, unm3, "o--", color=C_RED, lw=1.9, ms=5.5, label="年均未成交释放闲置（亩/年）")
    for b, gv in zip(bars3, gains3):
        ax.text(b.get_x() + b.get_width() / 2, gv + 12, f"+{gv:.0f}t", ha="center", va="bottom", fontsize=8.3)
    ax.set_xticks(x3)
    ax.set_xticklabels([l for _, l in arms3], fontsize=8.5)
    ax.set_ylabel("八年累计粮食增产 / t", fontsize=9.8, color=C_NAVY)
    ax2.set_ylabel("释放未成交耕地 / (亩/年)", fontsize=9.5, color=C_RED)
    ax.set_ylim(0, max(gains3) * 1.24)
    ax2.set_ylim(0, max(unm3) * 1.35)
    ax.set_title("(a) 村集体连片协调与流转租期对比（48 个独立评价种子）", fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")

    # Panel (b): Interprovincial Order-Grain Arbitrage vs Verified Grain-Area Cap
    ax = axes[1]
    arms4 = [
        ("east_order_zero_bonus_control", "零奖励对照\n(无价外补贴)"),
        ("east_order_loose_quota", "宽松面积额度\n(未核实粮面积)"),
        ("east_order_verified_cap_400", "实种面积挂钩\n(上限 400 kg/亩)"),
        ("east_order_verified_cap_350", "从严实种上限\n(上限 350 kg/亩)"),
    ]
    sub4 = df4.set_index("arm").loc[[a for a, _ in arms4]]
    x4 = np.arange(len(arms4))
    w4 = 0.34
    eg = sub4["east_grain_gain_vs_p0_tons"].to_numpy()
    ak = sub4["interprov_arbitrage_tons"].to_numpy()
    lk_sh = sub4["arb_leakage_share_pct"].to_numpy()
    b_eg = ax.bar(x4 - w4 * 0.56, eg, width=w4, color=C_TEAL, label="东部本地八年粮产增量（t）")
    b_ak = ax.bar(x4 + w4 * 0.56, ak, width=w4, color=C_RED, label="跨省买粮顶抵套利量（t）")
    ax.axhline(0, color="#64748B", lw=0.9)
    for bar_g, gv in zip(b_eg, eg):
        ax.text(bar_g.get_x() + bar_g.get_width() / 2, gv + (8 if gv >= 0 else -18), f"{gv:+.0f}t", ha="center", va="bottom" if gv >= 0 else "top", fontsize=8.3, color=C_TEAL, fontweight="bold")
    for bar_a, av, lks in zip(b_ak, ak, lk_sh):
        ax.text(bar_a.get_x() + bar_a.get_width() / 2, av + 10, f"{av:.0f}t\n(漏损{lks:.1f}%)", ha="center", va="bottom", fontsize=7.9, color=C_RED)
    ax.set_xticks(x4)
    ax.set_xticklabels([l for _, l in arms4], fontsize=8.8)
    ax.set_ylabel("八年粮食数量变化 / t", fontsize=9.8)
    ax.set_title("(b) 毗邻省际订单粮跨区买粮套利与亩均上限封顶效应", fontsize=10.5, fontweight="bold", pad=8)
    ymin = min(-35.0, float(np.min(eg)) - 40.0)
    ymax = max(float(np.max(eg)), float(np.max(ak))) * 1.30
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    plt.tight_layout(w_pad=2.4)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig5_government_6way(out_path: Path) -> None:
    df5 = pd.read_csv(RES_DIR / "exp5_government_6way_v2_2_summary.csv")
    df_yr = pd.read_csv(RES_DIR / "exp5_government_6way_v2_2_yearly.csv")

    pol_order = ["uniform_fixed", "regional_fixed", "transparent_rule", "myopic", "gov_rl"]
    pol_labels = {
        "uniform_fixed": "全国统一固定",
        "regional_fixed": "区域差异固定",
        "transparent_rule": "透明动态规则",
        "myopic": "单期短视寻优",
        "gov_rl": "跨期强化学习",
    }
    colors = {
        "uniform_fixed": "#64748B",
        "regional_fixed": C_BLUE,
        "transparent_rule": C_AMBER,
        "myopic": C_TEAL,
        "gov_rl": C_NAVY,
    }

    sub = df5.set_index("policy_label").loc[pol_order]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))
    fig.patch.set_facecolor("#FFFFFF")
    x = np.arange(len(pol_order))

    # (a) 8-year cumulative grain gain vs baseline + yearly gain path (clean inset or 2x2 metrics)
    ax = axes[0, 0]
    g_m = sub["grain_gain_vs_p0_tons"].to_numpy()
    glo, ghi = zip(*[_parse_ci(s) for s in sub["grain_gain_ci95"]])
    gyerr = np.vstack([g_m - np.asarray(glo), np.asarray(ghi) - g_m])
    bars_a = ax.bar(x, g_m, width=0.52, color=[colors[k] for k in pol_order], yerr=gyerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for idx_b, (b, mv) in enumerate(zip(bars_a, g_m)):
        ax.text(b.get_x() + b.get_width() / 2, mv + gyerr[1][idx_b] + 8, f"+{_r1(float(mv))} t", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([pol_labels[k] for k in pol_order], fontsize=9.2)
    ax.set_ylabel("较存量基准增产 / t", fontsize=9.8)
    ax.set_title("(a) 八年累计粮食增产量（48 个独立留出种子均值与 95% 置信区间）", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_ylim(0, max(g_m) * 1.22)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (b) 8-year actual fiscal spending (wan yuan)
    ax = axes[0, 1]
    sp_m = sub["spending_8yr_wan"].to_numpy()
    slo, shi = zip(*[_parse_ci(s) for s in sub["spending_ci95"]])
    syerr = np.vstack([sp_m - np.asarray(slo), np.asarray(shi) - sp_m])
    bars_b = ax.bar(x, sp_m, width=0.52, color=[colors[k] for k in pol_order], yerr=syerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for b, mv in zip(bars_b, sp_m):
        ax.text(b.get_x() + b.get_width() / 2, mv - 12.0, f"{mv:.2f}\n万元", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")
    ax.axhline(120.0, color=C_RED, ls="--", lw=1.4, label="八年累计共同预算上限（120.0 万元）")
    ax.set_xticks(x)
    ax.set_xticklabels([pol_labels[k] for k in pol_order], fontsize=9.2)
    ax.set_ylabel("八年实际财政支出 / 万元", fontsize=9.8)
    ax.set_title("(b) 八年累计实际财政支出（严格不突破 120 万元上限）", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_ylim(0, 142)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (c) Fiscal efficiency (kg grain per 100 yuan spent)
    ax = axes[1, 0]
    ef_m = sub["kg_per_100yuan_spent"].to_numpy()
    elo, ehi = zip(*[_parse_ci(s) for s in sub["kg_per_100yuan_ci95"]])
    eyerr = np.vstack([ef_m - np.asarray(elo), np.asarray(ehi) - ef_m])
    bars_c = ax.bar(x, ef_m, width=0.52, color=[colors[k] for k in pol_order], yerr=eyerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for idx_b, (b, mv) in enumerate(zip(bars_c, ef_m)):
        ax.text(b.get_x() + b.get_width() / 2, mv + eyerr[1][idx_b] + 0.9, f"{mv:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([pol_labels[k] for k in pol_order], fontsize=9.2)
    ax.set_ylabel("单位实支增粮效率 / (kg/百元)", fontsize=9.8)
    ax.set_title("(c) 单位财政实支增粮效率（均以实际支出为分母）", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_ylim(0, max(ef_m) * 1.24)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (d) Year 7-8 withdrawal retention + yearly gain trajectories
    ax = axes[1, 1]
    wd_m = sub["withdrawal_yr7_8_gain_tons"].to_numpy()
    wlo, whi = zip(*[_parse_ci(s) for s in sub["withdrawal_gain_ci95"]])
    wyerr = np.vstack([wd_m - np.asarray(wlo), np.asarray(whi) - wd_m])
    bars_d = ax.bar(x, wd_m, width=0.52, color=[colors[k] for k in pol_order], yerr=wyerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for idx_b, (b, mv) in enumerate(zip(bars_d, wd_m)):
        ax.text(b.get_x() + b.get_width() / 2, mv + wyerr[1][idx_b] + 1.8, f"+{mv:.1f} t", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([pol_labels[k] for k in pol_order], fontsize=9.2)
    ax.set_ylabel("第 7—8 年停补留存增量 / t", fontsize=9.8)
    ax.set_title("(d) 第 6 年末全面停补后第 7—8 年粮食产能留存", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_ylim(0, max(max(wd_m) * 1.28, 30.0))
    ax.grid(axis="y", ls=":", alpha=0.5)

    for ax_i in axes.flat:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=2.6, w_pad=2.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fig1", action="store_true")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_fig1_architecture(FIG_DIR / "图1_农业经营接续与动态调控系统总体机理框架图.png")
    print("Generated Figure 1.")
    if args.only_fig1:
        return

    plot_fig2_stage1_validation(FIG_DIR / "图2_2011至2015年固定观察点同户面板验证与退耕校准对比图.png")
    plot_fig3_rent_and_efficiency(FIG_DIR / "图3_不同财政支持工具的地租资本化分解与等实支增粮效率对比图.png")
    plot_fig4_village_and_arbitrage(FIG_DIR / "图4_村社连片协调双轨租期与省际订单粮套利封顶效应图.png")
    plot_fig5_government_6way(FIG_DIR / "图5_共同预算下六类政府调控方案八年绩效与停补期留存对比图.png")
    print("Generated all 5 CAE 300-dpi figures ->", FIG_DIR)


if __name__ == "__main__":
    main()
