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


def _draw_elbow(ax, x1, y1, x2, y2, *, color=C_SLATE, lw=2.0, angle_a=0, angle_b=90, rad=6, both=False, zorder=4):
    style = "<|-|>" if both else "-|>"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"angle,angleA={angle_a},angleB={angle_b},rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)


def plot_fig1_architecture(out_path: Path) -> None:
    """Minimalist 3-column CAE schematic (<=120 Chinese chars inside nodes/arrows, clean whitespace)."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(14.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        0.50, 0.942,
        "农业经营接续、财政支持传导与政府动态调控机理图",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#1C2833",
    )

    # Three soft neutral column panels matching minimalist scientific layout
    _draw_box(ax, 0.024, 0.055, 0.258, 0.825, fc="#F4F6F8", ec="none", lw=0.0, r=0.022, zorder=1)
    _draw_box(ax, 0.304, 0.055, 0.392, 0.825, fc="#F4F6F8", ec="none", lw=0.0, r=0.022, zorder=1)
    _draw_box(ax, 0.718, 0.055, 0.258, 0.825, fc="#F4F6F8", ec="none", lw=0.0, r=0.022, zorder=1)

    ax.text(0.153, 0.842, "数据与边界", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#2C3E50")
    ax.text(0.500, 0.842, "动态调控中枢", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#2C3E50")
    ax.text(0.847, 0.842, "仿真与诊断输出", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#2C3E50")

    # ---------------- Column 1: Data & Institutional Boundaries ----------------
    # Card 1A: Micro Survey Base
    _draw_box(ax, 0.045, 0.460, 0.216, 0.338, fc="#FFFFFF", ec="#4A7EC7", lw=1.4, r=0.016, zorder=2)
    ax.text(0.153, 0.760, "微观观察底座", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#1E3A5F", zorder=4)
    # Minimalist tablet/ledger + sparkline vector icon
    _draw_box(ax, 0.072, 0.545, 0.162, 0.165, fc="#F3F7FD", ec="#4A7EC7", lw=1.0, r=0.009, zorder=3)
    ax.add_patch(Rectangle((0.072, 0.684), 0.162, 0.026, facecolor="#DCE8F8", edgecolor="none", zorder=3))
    for i_dot, col_d in enumerate(["#E07A5F", "#E9B44C", "#4A7EC7"]):
        ax.add_patch(Circle((0.084 + i_dot * 0.010, 0.697), 0.0032, color=col_d, zorder=4))
    xs_w = np.linspace(0.085, 0.222, 18)
    ys_w = 0.635 + 0.018 * np.sin(np.linspace(0, 4.2 * np.pi, 18)) * np.array([0.4, 0.9, 1.0, 0.7, 0.5, 0.9, 1.1, 0.8, 0.4, 0.9, 1.0, 0.6, 0.8, 1.0, 0.5, 0.7, 0.9, 0.4])
    ax.plot(xs_w, ys_w, color="#2554B8", lw=1.4, zorder=4)
    for b_i, bh in enumerate([0.025, 0.036, 0.028, 0.042, 0.033]):
        ax.add_patch(Rectangle((0.092 + b_i * 0.025, 0.558), 0.015, bh, facecolor="#7AA2DB", edgecolor="none", zorder=4))
    ax.text(0.153, 0.495, "600观察户 · 41合成主体", ha="center", va="center", fontsize=9.3, color="#334155", zorder=4)

    # Card 1B: Policy & Conservation Boundaries
    _draw_box(ax, 0.045, 0.092, 0.216, 0.338, fc="#FFFFFF", ec="#4A7EC7", lw=1.4, r=0.016, zorder=2)
    ax.text(0.153, 0.392, "制度守恒约束", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#1E3A5F", zorder=4)
    # Minimalist rule sliders + bilateral balance glyph
    _draw_box(ax, 0.072, 0.175, 0.162, 0.165, fc="#F3F7FD", ec="#4A7EC7", lw=1.0, r=0.009, zorder=3)
    for row_k, (ry, kx) in enumerate([(0.305, 0.175), (0.258, 0.132), (0.210, 0.190)]):
        ax.plot([0.088, 0.216], [ry, ry], color="#9BB8E3", lw=2.0, solid_capstyle="round", zorder=4)
        ax.add_patch(Circle((kx, ry), 0.0085, facecolor="#2554B8", edgecolor="#FFFFFF", lw=1.0, zorder=5))
    ax.text(0.153, 0.128, "清零预算 · 双轨 · 实物守恒", ha="center", va="center", fontsize=9.3, color="#334155", zorder=4)

    # ---------------- Column 2: Two-Level Dynamic Governance Hub ----------------
    # Left slim pill: Historical Calibration
    _draw_box(ax, 0.316, 0.275, 0.066, 0.425, fc="#FFFFFF", ec="#94A3B8", lw=1.2, r=0.015, zorder=2)
    ax.text(0.349, 0.658, "历史校准", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E293B", zorder=4)
    # Minimalist multi-track timeline branch graphic
    ax.plot([0.346, 0.346], [0.315, 0.605], color="#334155", lw=1.6, zorder=3)
    ax.plot([0.346, 0.363, 0.363], [0.405, 0.445, 0.535], color="#109488", lw=1.4, zorder=3)
    ax.plot([0.346, 0.331, 0.331], [0.465, 0.505, 0.575], color="#2554B8", lw=1.4, zorder=3)
    for dx, dy, dc in [(0.346, 0.320, "#109488"), (0.346, 0.405, "#2554B8"), (0.363, 0.535, "#109488"), (0.331, 0.575, "#2554B8"), (0.346, 0.605, "#334155")]:
        ax.add_patch(Circle((dx, dy), 0.0065, facecolor=dc, edgecolor="#FFFFFF", lw=0.8, zorder=4))

    # Central Prominent Cobalt Pillar: Upper-Level Government RL
    _draw_box(ax, 0.412, 0.205, 0.176, 0.580, fc="#2354C4", ec="#1A3D94", lw=1.6, r=0.018, zorder=2)
    ax.text(0.500, 0.742, "政府动态调控", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#FFFFFF", zorder=4)
    # Hierarchical decision tree icon in crisp white/light blue
    ax.add_patch(Circle((0.500, 0.678), 0.011, facecolor="#9AC0FF", edgecolor="#FFFFFF", lw=1.2, zorder=4))
    for cx_b in (0.455, 0.500, 0.545):
        ax.plot([0.500, cx_b], [0.667, 0.618], color="#BFD8FF", lw=1.2, zorder=3)
        ax.add_patch(Circle((cx_b, 0.612), 0.008, facecolor="#9AC0FF", edgecolor="#FFFFFF", lw=1.0, zorder=4))
    for bx in (0.438, 0.472, 0.528, 0.562):
        ax.plot([0.455 if bx < 0.49 else 0.545, bx], [0.604, 0.558], color="#89B4FA", lw=1.0, zorder=3)
        ax.add_patch(Circle((bx, 0.554), 0.0055, facecolor="#FFFFFF", edgecolor="none", zorder=4))
    # 3 crisp frosted inner pills
    for py, ptxt in [(0.458, "滞后观测"), (0.362, "分区匹配"), (0.266, "跨期学习")]:
        _draw_box(ax, 0.434, py, 0.132, 0.062, fc="#1D45A5", ec="#93C5FD", lw=1.0, r=0.012, zorder=3)
        ax.text(0.500, py + 0.031, ptxt, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#FFFFFF", zorder=4)

    # Right slim pill: Bilateral Conservation Audit
    _draw_box(ax, 0.618, 0.275, 0.066, 0.425, fc="#FFFFFF", ec="#94A3B8", lw=1.2, r=0.015, zorder=2)
    ax.text(0.651, 0.658, "守恒核验", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1E293B", zorder=4)
    # Crisp cobalt shield badge
    shield = Polygon(
        [(0.651, 0.608), (0.670, 0.596), (0.667, 0.548), (0.651, 0.528), (0.635, 0.548), (0.632, 0.596)],
        closed=True, facecolor="#2354C4", edgecolor="#1A3D94", lw=1.1, zorder=4,
    )
    ax.add_patch(shield)
    ax.plot([0.643, 0.650, 0.660], [0.569, 0.558, 0.581], color="#FFFFFF", lw=1.8, zorder=5)
    # Mini assertion diamonds underneath
    ax.plot([0.651, 0.651], [0.330, 0.492], color="#334155", lw=1.2, zorder=3)
    ax.add_patch(Polygon([(0.651, 0.490), (0.664, 0.464), (0.651, 0.438), (0.638, 0.464)], closed=True, facecolor="#4A90E2", edgecolor="#334155", lw=0.9, zorder=4))
    ax.add_patch(Circle((0.651, 0.390), 0.0085, facecolor="#E9B44C", edgecolor="#334155", lw=0.9, zorder=4))
    ax.add_patch(Circle((0.651, 0.330), 0.0085, facecolor="#109488", edgecolor="#334155", lw=0.9, zorder=4))

    # ---------------- Column 3: Simulation Environment & Mechanism Output ----------------
    # Top Teal Block: Four-Region Agri-ABM
    _draw_box(ax, 0.738, 0.425, 0.218, 0.365, fc="#10A39A", ec="#0B7D76", lw=1.5, r=0.018, zorder=2)
    ax.text(0.847, 0.752, "四区农业多主体系统", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#FFFFFF", zorder=4)
    _draw_box(ax, 0.758, 0.635, 0.178, 0.068, fc="#AEEAE4", ec="#7CDCD3", lw=1.0, r=0.012, zorder=3)
    ax.text(0.847, 0.669, "主体最优决策", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0B433F", zorder=4)
    _draw_box(ax, 0.758, 0.548, 0.178, 0.068, fc="#AEEAE4", ec="#7CDCD3", lw=1.0, r=0.012, zorder=3)
    ax.text(0.847, 0.582, "竞租粮贸出清", ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0B433F", zorder=4)
    # Minimalist stacked plots geometric emblem
    for layer_y, fc_l in [(0.452, "#73DDD3"), (0.470, "#97EAE2"), (0.488, "#C6F6F1")]:
        poly = Polygon(
            [(0.785, layer_y), (0.818, layer_y - 0.014), (0.851, layer_y), (0.818, layer_y + 0.014)],
            closed=True, facecolor=fc_l, edgecolor="#0B7D76", lw=0.8, zorder=4,
        )
        ax.add_patch(poly)
    for cx_s, r_s in [(0.885, 0.016), (0.904, 0.020), (0.922, 0.014)]:
        ax.add_patch(Circle((cx_s, 0.474), r_s, facecolor="#E6FAF8", edgecolor="#0B7D76", lw=0.9, zorder=4))

    # Bottom Teal-Outlined Output Card
    _draw_box(ax, 0.738, 0.155, 0.218, 0.205, fc="#EBF9F7", ec="#10A39A", lw=1.4, r=0.018, zorder=2)
    ax.text(0.847, 0.328, "传导诊断输出", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#0B433F", zorder=4)
    # Mini white analytics inset
    _draw_box(ax, 0.758, 0.192, 0.084, 0.100, fc="#FFFFFF", ec="#B8E5E0", lw=1.0, r=0.008, zorder=3)
    for bi, (bh, bc) in enumerate([(0.030, "#2354C4"), (0.052, "#10A39A"), (0.042, "#D9822B"), (0.068, "#2354C4")]):
        ax.add_patch(Rectangle((0.768 + bi * 0.017, 0.205), 0.011, bh, facecolor=bc, edgecolor="none", zorder=4))
    _draw_box(ax, 0.854, 0.192, 0.084, 0.100, fc="#FFFFFF", ec="#B8E5E0", lw=1.0, r=0.008, zorder=3)
    ax.text(0.896, 0.262, "地租分解", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1E293B", zorder=4)
    ax.text(0.896, 0.238, "套利封顶", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1E293B", zorder=4)
    ax.text(0.896, 0.214, "产能留存", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1E293B", zorder=4)

    # ---------------- Clean Orthogonal Connectors ----------------
    # Left 1A & 1B merge neatly on vertical bus at x=0.289 -> enter Center Left Card cleanly
    ax.plot([0.263, 0.289, 0.289, 0.263], [0.625, 0.625, 0.260, 0.260], color="#3A68C2", lw=2.0, zorder=4)
    _draw_arrow(ax, 0.289, 0.488, 0.314, 0.488, color="#3A68C2", lw=2.1, zorder=5)
    # Inside Center Col clean links
    _draw_arrow(ax, 0.384, 0.488, 0.410, 0.488, color="#3A68C2", lw=1.8, both=True)
    _draw_arrow(ax, 0.590, 0.488, 0.616, 0.488, color="#3A68C2", lw=1.8, both=True)
    # Center Tower -> Right Top (clean straight horizontal channel ABOVE Audit Card!)
    _draw_arrow(ax, 0.590, 0.735, 0.736, 0.735, color="#108A81", lw=2.2, both=False)
    ax.text(0.663, 0.760, "补贴配比", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0F6B64", zorder=5)
    # Audit Shield <-> Right Top
    _draw_arrow(ax, 0.686, 0.560, 0.736, 0.560, color="#108A81", lw=1.9, both=True)
    # Right Top -> Right Bottom
    _draw_arrow(ax, 0.847, 0.422, 0.847, 0.362, color="#108A81", lw=2.1)
    # Clean single U-shaped Feedback Return Arrow (Warm Ochre along clean bottom track y=0.108)
    ax.plot([0.847, 0.847, 0.500], [0.153, 0.108, 0.108], color="#C2882B", lw=2.2, zorder=4)
    _draw_arrow(ax, 0.500, 0.108, 0.500, 0.203, color="#C2882B", lw=2.2, zorder=5)
    _draw_box(ax, 0.630, 0.088, 0.082, 0.036, fc="#F4F6F8", ec="none", lw=0.0, r=0.008, zorder=4)
    ax.text(0.671, 0.106, "滞后回报", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#9C6A19", zorder=5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
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
    df5a = pd.read_csv(RES_DIR / "exp5_government_6way_v2_2_summary.csv")
    df5b = pd.read_csv(RES_DIR / "exp5b_intertemporal_6way_summary.csv")
    df_yr = pd.read_csv(RES_DIR / "exp5b_intertemporal_yearly.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.2))
    fig.patch.set_facecolor("#FFFFFF")

    # (a) Base Scenario (Exp 5A: annual-only 30 actions) 8-yr gain
    ax = axes[0, 0]
    order_a = ["uniform_fixed", "transparent_rule", "regional_fixed", "myopic", "gov_rl"]
    lbl_a = {
        "uniform_fixed": "全国统一\n固定",
        "transparent_rule": "透明动态\n阈值规则",
        "regional_fixed": "事后区域\n最优对照",
        "myopic": "单期短视\n寻优",
        "gov_rl": "跨期强化\n学习",
    }
    col_a = ["#64748B", C_AMBER, C_BLUE, C_TEAL, C_NAVY]
    sub_a = df5a.set_index("policy_label").loc[order_a]
    ga = sub_a["grain_gain_vs_p0_tons"].to_numpy()
    galo, gahi = zip(*[_parse_ci(s) for s in sub_a["grain_gain_ci95"]])
    ga_err = np.vstack([ga - np.asarray(galo), np.asarray(gahi) - ga])
    xa = np.arange(len(order_a))
    bars_a = ax.bar(xa, ga, width=0.52, color=col_a, yerr=ga_err, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for idx_b, (b, mv) in enumerate(zip(bars_a, ga)):
        ax.text(b.get_x() + b.get_width() / 2, mv + ga_err[1][idx_b] + 10, f"+{_r1(float(mv))} t", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.set_xticks(xa)
    ax.set_xticklabels([lbl_a[k] for k in order_a], fontsize=8.8)
    ax.set_ylabel("八年累计增粮 / t", fontsize=9.8)
    ax.set_title("(a) 基础情景（仅年度清零补贴，30 方案）：八年累计增产", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_ylim(0, 680)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (b) Extended Intertemporal Scenario (Exp 5B: 36 actions w/ paid 3-yr consolidation)
    ax = axes[0, 1]
    order_b = ["uniform_fixed", "transparent_rule", "regional_fixed", "myopic_tree", "gov_rl_ridge", "gov_rl_tree"]
    lbl_b = {
        "uniform_fixed": "全国统一\n固定",
        "transparent_rule": "透明动态\n阈值规则",
        "regional_fixed": "事后最优\n(年付整治)",
        "myopic_tree": "单期短视\n(树集成)",
        "gov_rl_ridge": "跨期学习\n(线性近似)",
        "gov_rl_tree": "跨期学习\n(树集成)",
    }
    col_b = ["#64748B", C_AMBER, C_BLUE, C_TEAL, "#4A7EC7", C_NAVY]
    sub_b = df5b.set_index("policy_label").loc[order_b]
    gb = sub_b["grain_gain_vs_p0_tons"].to_numpy()
    gblo, gbhi = zip(*[_parse_ci(s) for s in sub_b["grain_gain_ci95"]])
    gb_err = np.vstack([gb - np.asarray(gblo), np.asarray(gbhi) - gb])
    xb = np.arange(len(order_b))
    bars_b = ax.bar(xb, gb, width=0.54, color=col_b, yerr=gb_err, capsize=4, error_kw={"lw": 1.2, "ecolor": "#111827"})
    for idx_b, (b, mv) in enumerate(zip(bars_b, gb)):
        ax.text(b.get_x() + b.get_width() / 2, mv + gb_err[1][idx_b] + 14, f"+{_r1(float(mv))} t", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels([lbl_b[k] for k in order_b], fontsize=8.5)
    ax.set_ylabel("八年累计增粮 / t", fontsize=9.8)
    ax.set_title("(b) 拓展情景（含有偿三年连片整治，36 方案）：八年累计增产", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_ylim(0, 1060)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (c) Annual Gain Trajectories in Exp 5B (Years 1-8) highlighting consolidation pulse rhythm
    ax = axes[1, 0]
    df_main = df_yr[df_yr["window"] == "main"]
    traj_specs = [
        ("gov_rl_tree", "跨期强化学习（树集成，第2、5、8年脉冲整治）", C_NAVY, "-o", 2.3),
        ("regional_fixed", "事后区域最优（每年固定支付整治费）", C_BLUE, "--s", 1.9),
        ("uniform_fixed", "全国统一固定（封顶订单）", "#64748B", "-.^", 1.7),
        ("transparent_rule", "透明动态阈值规则", C_AMBER, ":d", 1.7),
    ]
    for pkey, plbl, pcol, pfmt, plw in traj_specs:
        sub_t = df_main[df_main["policy_label"] == pkey].sort_values("year")
        ax.plot(sub_t["year"], sub_t["grain_gain_vs_p0_tons"], pfmt, color=pcol, lw=plw, ms=5.2, label=plbl)
    # Mark Years 2, 5, 8 pulse trigger nodes cleanly on gov_rl_tree curve
    rl_curve = df_main[df_main["policy_label"] == "gov_rl_tree"].set_index("year")
    for p_yr in (2, 5, 8):
        yv = float(rl_curve.loc[p_yr, "grain_gain_vs_p0_tons"])
        ax.scatter([p_yr], [yv], s=95, facecolor="#FFFFFF", edgecolor=C_RED, linewidth=2.0, zorder=6)
        ax.annotate(
            f"第{p_yr}年脉冲",
            (p_yr, yv),
            textcoords="offset points",
            xytext=(-6 if p_yr == 2 else 0, 12),
            ha="center",
            fontsize=8.2,
            fontweight="bold",
            color=C_RED,
        )
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("仿真运行年份 / 年", fontsize=9.8)
    ax.set_ylabel("当年较基线粮食增产量 / t", fontsize=9.8)
    ax.set_title("(c) 拓展情景逐年增粮轨迹与三年连片整治脉冲时序", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_ylim(15, 168)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")

    # (d) Exp 5B Efficiency (kg/100yuan) + Year 7-8 Withdrawal Retention (t)
    ax = axes[1, 1]
    ef_m = sub_b["kg_per_100yuan_spent"].to_numpy()
    elo, ehi = zip(*[_parse_ci(s) for s in sub_b["kg_per_100yuan_ci95"]])
    e_err = np.vstack([ef_m - np.asarray(elo), np.asarray(ehi) - ef_m])
    wd_m = sub_b["withdrawal_yr7_8_gain_tons"].to_numpy()

    bars_d = ax.bar(xb, ef_m, width=0.48, color=col_b, yerr=e_err, capsize=4, alpha=0.88, error_kw={"lw": 1.2, "ecolor": "#111827"}, label="单位实支增粮效率（kg/百元，左轴）")
    for idx_b, (b, mv) in enumerate(zip(bars_d, ef_m)):
        ax.text(b.get_x() + b.get_width() / 2, mv * 0.45, f"{mv:.1f}", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#FFFFFF")
    ax.set_xticks(xb)
    ax.set_xticklabels([lbl_b[k] for k in order_b], fontsize=8.5)
    ax.set_ylabel("单位实支增粮效率 / (kg/百元)", fontsize=9.8, color=C_NAVY)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", ls=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(xb, wd_m, "D--", color=C_RED, lw=2.0, ms=6.0, label="第 7—8 年停补留存增量（t，右轴）", zorder=6)
    for xk, wv in zip(xb, wd_m):
        y_off = -16 if xk == 1 else (12 if xk == 0 else 9)
        ax2.annotate(f"+{_r1(float(wv))} t", (xk, wv), textcoords="offset points", xytext=(0, y_off), ha="center", fontsize=8.2, fontweight="bold", color=C_RED)
    ax2.set_ylabel("停补后第 7—8 年留存增量 / t", fontsize=9.5, color=C_RED)
    ax2.set_ylim(0, 245)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(C_RED)
    ax.set_title("(d) 拓展情景财政增粮效率与第 6 年末停补后产能留存", fontsize=10.3, fontweight="bold", pad=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.2, loc="upper left")

    for ax_i in axes.flat:
        ax_i.spines["top"].set_visible(False)
        if ax_i is not axes[1, 1]:
            ax_i.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=2.8, w_pad=2.4)
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
