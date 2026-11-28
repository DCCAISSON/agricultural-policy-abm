"""Interactive Research Inspector bound directly to live `AgriSuccessionMesaModel` states.

Provides TWO entry modes sharing 100% of the same `AgriSuccessionMesaModel` execution engine:
1. `export_inspector_html(...)`: Executes `AgriSuccessionMesaModel.step(collect_micro_ledger=True)`
   across identical initial states & shocks under 7 policies (Years 1-8 + Withdrawal Years 7-8) and
   packages the authentic Mesa parcel, operator, regional, and government budget ledgers into a
   compact (~5 MB), self-contained interactive HTML workbench.
2. `serve_live_inspector(...)`: Starts a local HTTP server where any seed and policy pair can be
   stepped live in real time on `AgriSuccessionMesaModel`.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mesa_agri_abm.calibration import CalibratedRegionV2, load_v2_calibrated_regions, resolve_project_data_root
from mesa_agri_abm.config import V2Design, clean_v2_2_design, default_v2_design, intertemporal_v2_2_design
from mesa_agri_abm.model import AgriSuccessionMesaModel, JointActionV2

PROV_NAMES: dict[int, str] = {
    11: "北京", 12: "天津", 13: "河北", 14: "山西", 15: "内蒙古",
    21: "辽宁", 22: "吉林", 23: "黑龙江",
    31: "上海", 32: "江苏", 33: "浙江", 34: "安徽", 35: "福建", 36: "江西", 37: "山东",
    41: "河南", 42: "湖北", 43: "湖南", 44: "广东", 45: "广西", 46: "海南",
    50: "重庆", 51: "四川", 52: "贵州", 53: "云南",
    61: "陕西", 62: "甘肃", 63: "青海", 64: "宁夏", 65: "新疆",
    901: "东部薄样本合并池", 902: "中部薄样本合并池", 903: "西部薄样本合并池", 904: "东北薄样本合并池",
}


def capture_policy_mesa_trace(
    seed: int,
    regions: tuple[CalibratedRegionV2, ...],
    design: V2Design,
    *,
    label: str,
    policy_mode: str,
    fixed_action_idx: int = 0,
    learned_weights: dict[int, object] | None = None,
    explicit_joint_action: JointActionV2 | None = None,
    include_v22_spatial: bool = False,
) -> dict:
    """Step a real `AgriSuccessionMesaModel` with `collect_micro_ledger=True` and extract compact full UI trace."""
    m = AgriSuccessionMesaModel(
        seed,
        regions,
        design,
        policy_mode=policy_mode,
        fixed_action_idx=fixed_action_idx,
        learned_weights=learned_weights,
    )
    years_data: list[dict] = []
    wd_model: AgriSuccessionMesaModel | None = None

    for yr in range(1, design.main_years + 1):
        j_sum, r_sums, _ = m.step(
            explicit_joint_action=explicit_joint_action,
            collect_micro_ledger=True,
        )
        # Compact tuple schema for fast browser parsing & GitHub-friendly file size (<8 MB):
        # parcel: [0:id, 1:reg, 2:prov, 3:own, 4:op, 5:op_cat, 6:fate, 7:area, 8:kg, 9:rent, 10:sup]
        # actor:  [0:cat, 1:op_mu, 2:sow_g, 3:sow_c, 4:kg, 5:sub_kg, 6:mkt_kg, 7:sales, 8:rent_in, 9:rent_out, 10:sup, 11:fert, 12:inc, 13:wc(, 14:prov, 15:reg, 16:own_mu)]
        parcels_yr = []
        actors_yr = {}
        regions_out = []
        for reg_obj, r_sum_raw in zip(m.regions, r_sums):
            if include_v22_spatial:
                r_copy = dict(r_sum_raw)
                r_copy["village_pool_remaining_years"] = int(reg_obj.village_pool_remaining_years)
                regions_out.append(r_copy)
            else:
                regions_out.append(r_sum_raw)
            for pr in reg_obj.last_parcel_rows:
                parcels_yr.append([
                    pr["parcel_id"],
                    pr["region"],
                    PROV_NAMES.get(int(pr["province"]), str(pr["province"])),
                    pr["owner"],
                    pr["operator"],
                    pr["operator_category"],
                    pr["fate"],
                    round(pr["physical_area"], 2),
                    round(pr["harvest_grain_kg"], 0),
                    round(pr["rent_yuan"], 0),
                    round(pr["incremental_support_yuan"], 0),
                ])
            for ar in reg_obj.last_actor_rows:
                row_a = [
                    ar["actor_category"],
                    round(ar["operated_physical_mu"], 2),
                    round(ar["sown_grain_mu"], 2),
                    round(ar["sown_cash_mu"], 2),
                    round(ar["grain_kg"], 0),
                    round(ar["subsistence_grain_kg"], 0),
                    round(ar["marketed_grain_kg"], 0),
                    round(ar["grain_cash_sales_yuan"], 0),
                    round(ar["rent_income"], 0),
                    round(ar["rent_paid"], 0),
                    round(ar["incremental_support_yuan"], 0),
                    round(ar["fertility_support_yuan"], 0),
                    round(ar["full_net_income_yuan"], 0),
                    round(ar["working_cap_end"], 0),
                ]
                if include_v22_spatial:
                    op_obj = reg_obj.operators[ar["actor"]]
                    row_a.extend([
                        PROV_NAMES.get(int(op_obj.province), str(op_obj.province)),
                        reg_obj.name,
                        round(op_obj.bp.own_physical_mu, 2),
                    ])
                actors_yr[ar["actor"]] = row_a
        yr_entry: dict = {
            "year": yr,
            "joint": j_sum,
            "regions": regions_out,
            "parcels": parcels_yr,
            "actors": actors_yr,
        }
        if include_v22_spatial:
            yr_entry["pair_flows"] = {
                k: round(float(v), 1)
                for k, v in m.last_interprov_pair_flows.items()
                if float(v) > 0.5
            }
        years_data.append(yr_entry)
        if yr == design.withdrawal_after:
            wd_model = m.fork()
            wd_model.government.policy_mode = "p0"
            wd_model.government.fixed_action_idx = 0

    withdrawal_joint = []
    if wd_model is not None:
        for _ in range(design.withdrawal_after + 1, design.main_years + 1):
            wj, _, _ = wd_model.step(explicit_action_idx=0)
            withdrawal_joint.append(wj)

    return {
        "label": label,
        "policy_mode": policy_mode,
        "years": years_data,
        "withdrawal_joint": withdrawal_joint,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>农业经营接续、财政支持与政府RL —— Mesa 实时状态交互检查台 (V2)</title>
<style>
  :root {
    --bg: #f5f8fc; --card: #ffffff; --navy: #14365d; --blue: #1f6feb;
    --border: #d8e1ec; --text: #1c2733; --muted: #5c6b7a;
  }
  * { box-sizing: border-box; font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); }
  header { background: var(--navy); color: #fff; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .badge { background: rgba(255,255,255,0.16); padding: 5px 10px; border-radius: 4px; font-size: 12px; }
  .notice { background: #eef4ff; border-bottom: 1px solid #c9dcff; padding: 8px 24px; font-size: 12.5px; color: #194173; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; padding: 14px 24px; background: var(--card); border-bottom: 1px solid var(--border); align-items: center; }
  .controls label { font-size: 13px; font-weight: 600; color: var(--navy); display: flex; align-items: center; gap: 6px; }
  select, input[type=range] { padding: 5px 8px; border: 1px solid #b9c7d8; border-radius: 4px; font-size: 13px; }
  .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(15,35,60,0.04); }
  .panel h2 { margin: 0 0 10px 0; font-size: 15px; color: var(--navy); border-bottom: 1px solid #edf2f7; padding-bottom: 7px; }
  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }
  .kpi { background: #f8fafd; border: 1px solid #e3ebf5; border-radius: 6px; padding: 8px 10px; }
  .kpi .k-label { font-size: 11.5px; color: var(--muted); }
  .kpi .k-val { font-size: 16px; font-weight: 700; color: var(--navy); margin-top: 3px; }
  .parcel-wrap { display: flex; flex-wrap: wrap; gap: 3px; max-height: 270px; overflow-y: auto; padding: 6px; background: #f9fbfd; border: 1px solid #e5edf6; border-radius: 6px; }
  .p-cell { width: 15px; height: 15px; border-radius: 2px; cursor: pointer; transition: transform 0.1s; border: 1px solid rgba(0,0,0,0.08); }
  .p-cell:hover { transform: scale(1.45); z-index: 5; border-color: #000; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; margin: 8px 0; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 4px; }
  .dot { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  th, td { border-bottom: 1px solid #edf2f7; padding: 5px 6px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f4f7fb; color: var(--navy); font-weight: 600; }
  .inspector-box { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .detail-card { background: #f8fbff; border: 1px solid #d4e3f7; border-radius: 6px; padding: 10px 12px; font-size: 12.5px; line-height: 1.65; }
</style>
</head>
<body>
<header>
  <h1>农业经营接续、财政支持与政府动态决策 —— Mesa 2.3.2 实时状态交互检查台</h1>
  <div class="badge">同初始状态 · 同随机冲击 · 双策略逐年逐地块对照</div>
</header>
<div class="notice">
  <strong>模型与空间边界说明：</strong>本界面全部数据直接由 <code>AgriSuccessionMesaModel</code> 真实年度状态账导出，无独立展示模拟逻辑。省份标签为 NFPS 调查真实省级归属，地块色块矩阵为<strong>合成经营单元布局（非真实测绘 GIS 宗地位置）</strong>；600 户调查家庭与三类合成规模经营主体严格分账。
</div>
<div class="controls">
  <label>模拟年份：<input type="range" id="yrSlider" min="1" max="8" value="1"> <span id="yrLabel">第 1 年</span></label>
  <label>政策区域：<select id="regSel"><option value="东部">东部</option><option value="中部">中部</option><option value="西部">西部</option><option value="东北">东北</option></select></label>
  <label>左侧策略 A：<select id="polASel"></select></label>
  <label>右侧策略 B：<select id="polBSel"></select></label>
</div>
<div class="main-grid">
  <div class="panel" id="panelA"></div>
  <div class="panel" id="panelB"></div>
  <div class="panel inspector-box">
    <div>
      <h2>点击任意合成地块单元查看双策略同地块微观经营与补贴对比</h2>
      <div id="parcelDetail" class="detail-card">请点击上方任意色块查看该土地经营单元在策略 A 与策略 B 下的权利人、实际经营者、流转租金、作物产出与财政补贴账目。</div>
    </div>
    <div>
      <h2>该地块对应经营主体（农户 / 家庭农场 / 合作社 / 农业公司）收支与口粮账</h2>
      <div id="actorDetail" class="detail-card">点击地块后此处同步显示实际经营者当年的物理经营面积、口粮自留量、商品售粮现金、地力底盘、新增专项与期末营运资金。</div>
    </div>
  </div>
</div>
<script>
const DATA = __MESA_PAYLOAD__;
const FATE_COLORS = {
  "incumbent_grain": ["#1f6feb", "原户连续种粮"],
  "strict_succession_grain": ["#0e9f6e", "换人严格接续种粮"],
  "idle_recovered_grain": ["#38bdf8", "上期闲置复耕种粮"],
  "converted_to_grain": ["#6366f1", "非粮改种粮食"],
  "own_cash": ["#f59e0b", "原户种经济作物"],
  "leased_cash": ["#d97706", "流转后种经济作物"],
  "leased_unplanted": ["#dc2626", "签约在租未播"],
  "own_idle": ["#cbd5e1", "未流转自留闲置"]
};
const polKeys = Object.keys(DATA.traces);
const polASel = document.getElementById("polASel");
const polBSel = document.getElementById("polBSel");
polKeys.forEach((k, idx) => {
  polASel.add(new Option(DATA.traces[k].label, k, idx === 0, idx === 0));
  polBSel.add(new Option(DATA.traces[k].label, k, idx === polKeys.length - 1, idx === polKeys.length - 1));
});
let selectedPid = null;

function renderSide(containerId, polKey, yr, regName) {
  const tr = DATA.traces[polKey];
  const yd = tr.years[yr - 1];
  const j = yd.joint;
  const r = yd.regions.find(x => x.region === regName);
  const plots = yd.parcels.filter(p => p[1] === regName);

  let legHtml = '<div class="legend">';
  for (const [fk, [col, txt]] of Object.entries(FATE_COLORS)) {
    legHtml += `<span><i class="dot" style="background:${col}"></i>${txt}</span>`;
  }
  legHtml += '</div>';

  let cells = '';
  plots.forEach(p => {
    const col = (FATE_COLORS[p[6]] || ["#94a3b8", "其他"])[0];
    const ring = (p[0] === selectedPid) ? "outline:2px solid #000;" : "";
    cells += `<div class="p-cell" style="background:${col};${ring}" title="${p[0]} (${p[2]}) ${p[7]}亩 | 经营:${p[4]}" onclick="selectParcel('${p[0]}')"></div>`;
  });

  let regRows = '';
  yd.regions.forEach(rr => {
    regRows += `<tr><td>${rr.region}</td><td>${rr.policy}</td><td>${(rr.reservation/10000).toFixed(2)}</td><td>${(rr.spending/10000).toFixed(2)}</td><td>${(rr.grain_kg/1000).toFixed(1)}</td><td>${rr.leased_area_mu.toFixed(1)}</td><td>${rr.fate_2_strict_succession_grain_mu.toFixed(1)}</td><td>${rr.unit_rent_yuan_mu.toFixed(0)}</td></tr>`;
  });

  document.getElementById(containerId).innerHTML = `
    <h2>${tr.label}（第 ${yr} 年 · 全国合计与 ${regName} 合成地块矩阵）</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="k-label">全国当年粮食总产</div><div class="k-val">${(j.grain_kg/1000).toFixed(1)} 吨</div></div>
      <div class="kpi"><div class="k-label">新增专项实付 / 预留</div><div class="k-val">${(j.spending/10000).toFixed(2)} / ${(j.reservation/10000).toFixed(2)} 万</div></div>
      <div class="kpi"><div class="k-label">流转面积 / 严格续粮</div><div class="k-val">${j.leased_area_mu.toFixed(0)} / ${j.fate_2_strict_succession_grain_mu.toFixed(0)} 亩</div></div>
      <div class="kpi"><div class="k-label">平均亩租 / 小农包容 IDI</div><div class="k-val">${j.unit_rent_yuan_mu.toFixed(0)} 元 · ${j.idi_score.toFixed(3)}</div></div>
    </div>
    <div style="font-size:12.5px;color:#334e68;margin-bottom:4px;"><strong>${regName}</strong>（${plots.length} 个合成经营单元，物理面积 ${r.physical_area_mu.toFixed(1)} 亩；600调查户产粮 ${(r.survey_600_grain_kg/1000).toFixed(1)} 吨，规模组织产粮 ${((r.grain_kg - r.survey_600_grain_kg)/1000).toFixed(1)} 吨）</div>
    ${legHtml}
    <div class="parcel-wrap">${cells}</div>
    <table>
      <thead><tr><th>区域</th><th>当年工具</th><th>预算(万)</th><th>实付(万)</th><th>粮产(吨)</th><th>在租(亩)</th><th>换人续粮(亩)</th><th>亩租(元)</th></tr></thead>
      <tbody>${regRows}</tbody>
    </table>
  `;
}

function renderInspector() {
  const yr = parseInt(document.getElementById("yrSlider").value, 10);
  const regName = document.getElementById("regSel").value;
  document.getElementById("yrLabel").textContent = `第 ${yr} 年`;
  const kA = polASel.value, kB = polBSel.value;
  renderSide("panelA", kA, yr, regName);
  renderSide("panelB", kB, yr, regName);

  if (selectedPid) {
    const ydA = DATA.traces[kA].years[yr - 1];
    const ydB = DATA.traces[kB].years[yr - 1];
    const pA = ydA.parcels.find(x => x[0] === selectedPid);
    const pB = ydB.parcels.find(x => x[0] === selectedPid);
    if (pA && pB) {
      document.getElementById("parcelDetail").innerHTML = `
        <strong>合成单元编号：</strong><code>${pA[0]}</code>（真实省份归属：<strong>${pA[2]}</strong>，物理面积：<strong>${pA[7]} 亩</strong>，原承包权利人：<code>${pA[3]}</code>）<br>
        • <strong>[${DATA.traces[kA].label}]</strong> 实际经营者：<code>${pA[4]}</code> (${pA[5]}) | 状态：<strong>${(FATE_COLORS[pA[6]]||['',pA[6]])[1]}</strong> | 产粮：${pA[8]} kg | 租金：${pA[9]} 元 | 专项补贴：${pA[10]} 元<br>
        • <strong>[${DATA.traces[kB].label}]</strong> 实际经营者：<code>${pB[4]}</code> (${pB[5]}) | 状态：<strong>${(FATE_COLORS[pB[6]]||['',pB[6]])[1]}</strong> | 产粮：${pB[8]} kg | 租金：${pB[9]} 元 | 专项补贴：${pB[10]} 元
      `;
      const aA = ydA.actors[pA[4]];
      const aB = ydB.actors[pB[4]];
      if (aA && aB) {
        document.getElementById("actorDetail").innerHTML = `
          • <strong>[${DATA.traces[kA].label}] 经营者 <code>${pA[4]}</code> (${aA[0]})</strong>：经营 ${aA[1]} 亩（播粮 ${aA[2]} 亩次 / 经作 ${aA[3]} 亩次），产粮 ${aA[4]} kg（自留口粮 ${aA[5]} kg / 出售余粮 ${aA[6]} kg，售粮现金 ${aA[7]} 元），租金收/支 ${aA[8]}/${aA[9]} 元，新增专项 ${aA[10]} 元，地力底盘 ${aA[11]} 元，全年综合净收入 <strong>${aA[12]} 元</strong>，年末营运资金 ${aA[13]} 元。<br>
          • <strong>[${DATA.traces[kB].label}] 经营者 <code>${pB[4]}</code> (${aB[0]})</strong>：经营 ${aB[1]} 亩（播粮 ${aB[2]} 亩次 / 经作 ${aB[3]} 亩次），产粮 ${aB[4]} kg（自留口粮 ${aB[5]} kg / 出售余粮 ${aB[6]} kg，售粮现金 ${aB[7]} 元），租金收/支 ${aB[8]}/${aB[9]} 元，新增专项 ${aB[10]} 元，地力底盘 ${aB[11]} 元，全年综合净收入 <strong>${aB[12]} 元</strong>，年末营运资金 ${aB[13]} 元。
        `;
      }
    }
  }
}

window.selectParcel = function(pid) { selectedPid = pid; renderInspector(); };
document.getElementById("yrSlider").addEventListener("input", renderInspector);
document.getElementById("regSel").addEventListener("change", () => { selectedPid = null; renderInspector(); });
polASel.addEventListener("change", renderInspector);
polBSel.addEventListener("change", renderInspector);
renderInspector();
</script>
</body>
</html>
"""


def export_inspector_html(
    output_html: Path,
    project_root: Path,
    *,
    seed: int = 574001,
    design: V2Design | None = None,
    learned_q_weights: dict[int, list[list[float]]] | None = None,
    uniform_fixed_idx: int = 1,
    regional_fixed_idx: int = 6,
) -> Path:
    d = design or default_v2_design()
    regions, _ = load_v2_calibrated_regions(project_root, d)

    specs = [
        ("p0", "P0 基准（仅耕地地力底盘）", "p0", 0, None),
        ("producer_pre", "全国普惠：实际种粮播前直补", "uniform_fixed", 1, None),
        ("scale_pre", "全国规模：适度规模/承接播前补贴", "uniform_fixed", 3, None),
        ("service", "全国托管：农业社会化服务补助", "uniform_fixed", 5, None),
        ("regional_mix", "因地制宜区域倾斜固定策略", "regional_fixed", regional_fixed_idx, None),
        ("transparent", "简单透明动态规则", "transparent_rule", 0, None),
    ]
    if learned_q_weights is not None:
        specs.append(("gov_rl", "政府跨期强化学习 (Gov-RL)", "gov_rl", 0, learned_q_weights))

    traces = {}
    for key, label, mode, f_idx, w_mat in specs:
        traces[key] = capture_policy_mesa_trace(
            seed, regions, d, label=label, policy_mode=mode, fixed_action_idx=f_idx, learned_weights=w_mat
        )

    payload = {"seed": seed, "budget_yuan": d.annual_common_budget_yuan, "traces": traces}
    html_str = HTML_TEMPLATE.replace("__MESA_PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_str, encoding="utf-8")
    return output_html


WORKBENCH_V2_2_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>农业经营接续与动态调控 —— Mesa V2.2 空间交互白箱工作台</title>
<style>
  :root {
    --bg: #fafbfa; --stage: #f1f5f2; --card: #ffffff;
    --ink: #24352e; --muted: #596660; --accent: #1e5549; --accent-bright: #137953;
    --line: #c9d2cb; --line-soft: #d8e1dc; --warn: #b66c26; --danger: #c04851;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14.5px/1.55 "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  }
  header, main, footer { max-width: 1440px; padding: 0 3%; margin: auto; }
  header {
    padding-top: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  }
  .kicker { font-size: 12.5px; color: var(--muted); letter-spacing: 0.02em; }
  h1 { margin: 2px 0 0; font-size: 23px; color: var(--ink); font-weight: 700; }
  h2 { margin: 0 0 8px; font-size: 16px; color: var(--ink); font-weight: 700; }
  h3 { margin: 0 0 6px; font-size: 14.5px; color: var(--accent); font-weight: 700; }
  .pill-ok {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
    border-radius: 999px; background: #e8f5ef; color: #146649; border: 1px solid #b8dfcd;
    font-size: 12.5px; font-weight: 600; white-space: nowrap;
  }
  .notice {
    margin-top: 12px; padding: 10px 14px; background: #f3f7f4; border: 1px solid var(--line-soft);
    border-left: 3px solid var(--accent); border-radius: 4px; font-size: 12.5px; color: #3a4e45;
  }
  .controls {
    position: sticky; top: 0; z-index: 30; background: rgba(250, 251, 250, 0.96);
    backdrop-filter: blur(4px); border-bottom: 1px solid var(--line);
    padding: 10px 3%; margin: 10px -3% 0;
    display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: space-between;
  }
  .ctrl-group { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
  .ctrl-label { font-size: 12.5px; color: var(--muted); font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
  button, select, input { font: inherit; color: inherit; }
  button, select {
    border: 1px solid #b6c5bd; background: #fff; border-radius: 4px; padding: 4px 10px; font-size: 13px;
  }
  button { cursor: pointer; }
  button:hover, select:hover { border-color: var(--accent-bright); }
  button:focus-visible, select:focus-visible, input:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid var(--warn); outline-offset: 2px;
  }
  button:disabled { cursor: default; opacity: 0.4; }
  .preset-btn { background: #f3f7f4; border-color: #b8c9c0; color: var(--accent); font-weight: 600; font-size: 12.5px; }
  .preset-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .transport { display: flex; align-items: center; gap: 6px; }
  .transport button { width: 32px; height: 30px; padding: 0; font-weight: 700; }
  .transport input[type=range] { width: 120px; accent-color: var(--accent-bright); }
  .transport strong { min-width: 5.5ch; font-variant-numeric: tabular-nums; color: var(--accent); text-align: center; }
  .seg { display: inline-flex; border: 1px solid #b6c5bd; border-radius: 4px; overflow: hidden; }
  .seg button { border: 0; border-radius: 0; padding: 4px 10px; font-size: 12.5px; }
  .seg button[aria-pressed="true"] { background: var(--accent); color: #fff; font-weight: 600; }
  .metrics {
    display: grid; grid-template-columns: repeat(4, 1fr); margin-top: 12px;
    border-block: 1px solid var(--line); background: #fff;
  }
  .metrics > div { padding: 10px 14px; border-right: 1px solid #e6ece8; }
  .metrics > div:last-child { border-right: 0; }
  .metrics .m-lbl { font-size: 12px; color: var(--muted); }
  .metrics b { font-size: 20px; display: block; line-height: 1.3; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .metrics .m-sub { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .stage-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
  .stage-card { background: var(--card); border: 1px solid var(--line-soft); border-radius: 6px; padding: 12px 14px; min-width: 0; }
  .stage-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
  .stage-badge { font-size: 12px; color: var(--accent); font-weight: 600; background: #eef5f1; padding: 2px 8px; border-radius: 4px; }
  .viz {
    display: block; width: 100%; height: auto; max-height: 58vh; min-height: 290px;
    background: var(--stage); border-block: 1px solid var(--line-soft); border-radius: 3px;
  }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--muted); margin-top: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
  .prov-tile { cursor: pointer; }
  .prov-tile:hover rect.tile-bg { stroke: #15755c; stroke-width: 2.2; }
  .actor-node { cursor: pointer; transition: opacity 0.12s; }
  .link-edge { fill: none; transition: opacity 0.12s; }
  .link-edge.focus { opacity: 1 !important; stroke-width: 2.6 !important; }
  .dimmed { opacity: 0.15; }
  .ledger-grid { display: grid; grid-template-columns: 1.05fr 1fr; gap: 16px; margin-top: 14px; margin-bottom: 24px; }
  .ledger-card { background: var(--card); border: 1px solid var(--line-soft); border-radius: 6px; padding: 14px 16px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-top: 6px; }
  th, td { border-bottom: 1px solid #e3e9e5; padding: 5px 6px; text-align: right; font-size: 12.5px; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #4b6156; font-weight: 600; background: #f8faf8; }
  .delta-pos { color: #146649; font-weight: 600; }
  .delta-neg { color: #b65326; font-weight: 600; }
  .peer-btn {
    display: inline-block; margin: 3px 4px 0 0; padding: 2px 8px; border-radius: 999px;
    background: #f1f5f2; border: 1px solid #b8c9c0; font-size: 12px; color: var(--accent);
  }
  .peer-btn:hover { background: var(--accent); color: #fff; }
  footer { border-top: 1px solid var(--line); padding-block: 14px 26px; font-size: 12.5px; color: var(--muted); }
  @media (max-width: 900px) {
    .stage-grid, .ledger-grid { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<header>
  <div>
    <div class="kicker">Mesa 2.3.2 原生多主体农业经济仿真系统（V2.2 脱敏空间白箱检查台 · 留出随机种子 __SEED__）</div>
    <h1>农业经营接续、跨省买粮套利与动态调控空间白箱工作台</h1>
  </div>
  <div class="pill-ok" id="consBadge">双边实物守恒误差：0.00 kg · 零原生身份暴露</div>
</header>

<main>
  <div class="notice">
    <strong>白箱数据与空间边界说明：</strong>本页面全部数值与流转网络均由 <code>AgriSuccessionMesaModel</code> 在同一初始状态与外生天气冲击下逐年步进导出（不含前端独立模拟）。全国 29 省方格地图基于真实省级调查编码示意排布，省内网络与合成规模主体位置均为非测绘交互布局，600 户调查家庭与 41 个合成规模主体严格分账。
  </div>

  <div class="controls">
    <div class="ctrl-group">
      <label class="ctrl-label">空间层级：
        <select id="scopeSel"></select>
      </label>
      <button id="btnNational" type="button">返回全国 29 省</button>
      <label class="ctrl-label">策略 A：
        <select id="polASel"></select>
      </label>
      <label class="ctrl-label">策略 B：
        <select id="polBSel"></select>
      </label>
    </div>

    <div class="ctrl-group">
      <button class="preset-btn active" id="presetArb" type="button">预设 1：跨省套利 vs 400kg封顶</button>
      <button class="preset-btn" id="presetPulse" type="button">预设 2：跨期三年脉冲 RL vs 短视</button>
      <button class="preset-btn" id="presetProv" type="button">钻取产销省（黑龙江/浙江）</button>
      <div class="seg" role="group" aria-label="着色模式">
        <button id="modeFate" type="button" aria-pressed="true">经营命运着色</button>
        <button id="modeDiff" type="button" aria-pressed="false">策略差异高亮</button>
      </div>
      <div class="transport">
        <button id="btnPrev" type="button" title="上一年" aria-label="上一年">◀</button>
        <button id="btnPlay" type="button" title="播放" aria-label="播放">▶</button>
        <button id="btnNext" type="button" title="下一年" aria-label="下一年">▶|</button>
        <input type="range" id="yrSlider" min="1" max="8" value="3" aria-label="仿真年份">
        <strong id="yrLabel">第 3 年</strong>
      </div>
    </div>
  </div>

  <div class="metrics" id="kpiStrip"></div>

  <div class="stage-grid">
    <div class="stage-card">
      <div class="stage-head">
        <h2 id="titleA">策略 A</h2>
        <span class="stage-badge" id="badgeA"></span>
      </div>
      <svg class="viz" id="svgA" viewBox="0 0 820 495" role="img" aria-label="策略A空间态势图"></svg>
      <div class="legend" id="legendBox"></div>
    </div>
    <div class="stage-card">
      <div class="stage-head">
        <h2 id="titleB">策略 B</h2>
        <span class="stage-badge" id="badgeB"></span>
      </div>
      <svg class="viz" id="svgB" viewBox="0 0 820 495" role="img" aria-label="策略B空间态势图"></svg>
      <div class="legend">
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="#4b6b8a"/></svg> 调查农户（圆）</span>
        <span><svg width="12" height="12"><polygon points="6,1 11,6 6,11 1,6" fill="#1e7f5c"/></svg> 家庭农场（菱形）</span>
        <span><svg width="12" height="12"><polygon points="6,1 10.5,3.5 10.5,8.5 6,11 1.5,8.5 1.5,3.5" fill="#2b62a8"/></svg> 农业合作社（六边形）</span>
        <span><svg width="12" height="12"><rect x="2" y="2" width="8" height="8" rx="1.5" fill="#77649a"/></svg> 农业企业（方框）</span>
        <span>━━ 实线箭头：严格接续种粮　╌╌ 虚线箭头：非粮/未播流转　<strong style="color:#c04851">━━ 红弧线：跨省订单买粮通道</strong></span>
      </div>
    </div>
  </div>

  <div class="ledger-grid">
    <div class="ledger-card">
      <h3>宏观绩效、政府行动与跨省守恒白箱总账（当年对照：策略 A vs 策略 B）</h3>
      <div id="macroLedger"></div>
    </div>
    <div class="ledger-card">
      <h3 id="microTitle">微观主体与流转网络白箱对照（点击上方任意省份方格或网络节点聚焦）</h3>
      <div id="microLedger"></div>
    </div>
  </div>
</main>

<footer>
  数据来源与脱敏说明：微观基期底账基于农业农村部全国农村固定观察点 2009—2010 年 600 户同户样本（内部编号匿名重编码为 H0000—H0599）与 41 个合成规模经营主体；界面仅保留脱敏仿真汇总账目。
</footer>

<script>
const DATA = __MESA_V22_PAYLOAD__;
const PROV_GRID = {
  "黑龙江":[8,0], "新疆":[0,1], "内蒙古":[3,1], "北京":[5,1], "天津":[6,1], "吉林":[8,1],
  "甘肃":[1,2], "青海":[2,2], "宁夏":[3,2], "山西":[4,2], "河北":[5,2], "山东":[6,2], "辽宁":[8,2],
  "陕西":[3,3], "河南":[4,3], "安徽":[5,3], "江苏":[6,3], "上海":[7,3],
  "四川":[2,4], "湖北":[3,4], "湖南":[4,4], "江西":[5,4], "浙江":[6,4],
  "云南":[1,5], "贵州":[2,5], "广西":[3,5], "广东":[4,5], "福建":[5,5],
  "海南":[4,6]
};
const PROV_REG = {
  "北京":"东部","天津":"东部","河北":"东部","上海":"东部","江苏":"东部","浙江":"东部","福建":"东部","山东":"东部","广东":"东部","海南":"东部",
  "山西":"中部","安徽":"中部","江西":"中部","河南":"中部","湖北":"中部","湖南":"中部",
  "内蒙古":"西部","广西":"西部","四川":"西部","贵州":"西部","云南":"西部","陕西":"西部","甘肃":"西部","青海":"西部","宁夏":"西部","新疆":"西部",
  "辽宁":"东北","吉林":"东北","黑龙江":"东北"
};
const REG_TINTS = {
  "东部": "#eef4fa", "中部": "#eef7f3", "西部": "#faf5ec", "东北": "#f3f1f9"
};
const INST_CN = {
  "none": "无增量(仅地力)",
  "producer_pre": "播前直补",
  "producer_post": "产后直补",
  "scale_takeover_pre": "播前规模",
  "scale_takeover_post": "产后规模",
  "service_trusteeship": "农机托管",
  "order_farming": "订单奖励"
};
function fmtPolicyCn(raw) {
  if (!raw) return "无";
  if (raw.startsWith("vc+")) return "整治+" + (INST_CN[raw.slice(3)] || raw.slice(3));
  return INST_CN[raw] || raw;
}
const FATE_STYLES = {
  "incumbent_grain": ["#4b749f", "原户连续种粮"],
  "strict_succession_grain": ["#23855d", "换人接续种粮"],
  "idle_recovered_grain": ["#3794a8", "闲置复耕种粮"],
  "converted_to_grain": ["#666fa8", "非粮改种粮食"],
  "own_cash": ["#db952e", "原户种经济作物"],
  "leased_cash": ["#c26e1f", "流转种经济作物"],
  "leased_unplanted": ["#c04851", "在租未播"],
  "own_idle": ["#adb6b1", "自留闲置"]
};
const CAT_CN = {
  "survey_household": "调查农户",
  "family_farm": "合成家庭农场",
  "cooperative": "合成农业合作社",
  "agri_company": "合成农业企业"
};

let state = {
  yr: 3,
  scope: "全国",
  polA: "east_loose_arb",
  polB: "east_capped_arb",
  colorMode: "fate",
  selectedActor: null,
  playing: false,
  timer: null,
  provToggle: 0
};

const polKeys = Object.keys(DATA.traces);
const polASel = document.getElementById("polASel");
const polBSel = document.getElementById("polBSel");
polKeys.forEach(k => {
  polASel.add(new Option(DATA.traces[k].label, k));
  polBSel.add(new Option(DATA.traces[k].label, k));
});
if (!DATA.traces[state.polA]) state.polA = polKeys[0];
if (!DATA.traces[state.polB]) state.polB = polKeys[polKeys.length - 1];
polASel.value = state.polA;
polBSel.value = state.polB;

const scopeSel = document.getElementById("scopeSel");
scopeSel.add(new Option("全国 · 四区 29 省", "全国"));
["东部", "中部", "西部", "东北"].forEach(r => scopeSel.add(new Option(`政策区 · ${r}`, `reg:${r}`)));
Object.keys(PROV_GRID).sort((a,b) => a.localeCompare(b, "zh")).forEach(p => {
  scopeSel.add(new Option(`省份网络 · ${p} (${PROV_REG[p]})`, `prov:${p}`));
});

function renderLegend() {
  const box = document.getElementById("legendBox");
  let html = "";
  for (const [, [col, txt]] of Object.entries(FATE_STYLES)) {
    html += `<span><i style="background:${col}"></i>${txt}</span>`;
  }
  box.innerHTML = html;
}
renderLegend();

function fmtDelta(val, digits=1, unit="") {
  const cls = val > 1e-6 ? "delta-pos" : (val < -1e-6 ? "delta-neg" : "");
  const sign = val > 1e-6 ? "+" : "";
  return `<span class="${cls}">${sign}${val.toFixed(digits)}${unit}</span>`;
}

function getProvinceStats(yd, provName) {
  const actIds = [];
  let grainKg = 0, leasedMu = 0, strictMu = 0, supYuan = 0;
  const provParcels = yd.parcels.filter(p => p[2] === provName);
  const actorFates = {};
  const actorPri = {"strict_succession_grain":8,"leased_unplanted":7,"leased_cash":6,"idle_recovered_grain":5,"converted_to_grain":4,"own_cash":3,"incumbent_grain":2,"own_idle":1};
  provParcels.forEach(p => {
    grainKg += p[8];
    if (p[3] !== p[4]) leasedMu += p[7];
    if (p[6] === "strict_succession_grain") strictMu += p[7];
    supYuan += p[10];
    [p[3], p[4]].forEach(aid => {
      const cur = actorFates[aid];
      if (!cur || (actorPri[p[6]] || 0) > (actorPri[cur] || 0)) actorFates[aid] = p[6];
    });
  });
  for (const [aid, row] of Object.entries(yd.actors)) {
    if (row[14] === provName) actIds.push(aid);
  }
  return { provParcels, actIds, grainKg, leasedMu, strictMu, supYuan, actorFates };
}

function tileCenter(prov) {
  const pos = PROV_GRID[prov];
  if (!pos) return [410, 240];
  return [10 + pos[0] * 89 + 43, 6 + pos[1] * 69 + 32];
}

function renderCartogramSvg(svgId, markerPrefix, yd, ydOther, filterReg) {
  const svg = document.getElementById(svgId);
  const provs = Object.keys(PROV_GRID).filter(p => !filterReg || PROV_REG[p] === filterReg);
  let minX = 9999, minY = 9999, maxX = -9999, maxY = -9999;
  provs.forEach(p => {
    const [c, r] = PROV_GRID[p];
    const x = 10 + c * 89, y = 6 + r * 69;
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + 86); maxY = Math.max(maxY, y + 64);
  });
  if (!filterReg) { svg.setAttribute("viewBox", "0 0 820 495"); }
  else { svg.setAttribute("viewBox", `${minX - 18} ${minY - 14} ${(maxX - minX) + 36} ${(maxY - minY) + 28}`); }

  const regMap = {};
  yd.regions.forEach(rr => { regMap[rr.region] = rr; });
  const otherRegMap = {};
  ydOther.regions.forEach(rr => { otherRegMap[rr.region] = rr; });

  let defs = `<defs>
    <marker id="${markerPrefix}Arb" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.5" markerHeight="5.5" orient="auto">
      <path d="M0,1 L7,4 L0,7 Z" fill="#c04851"/>
    </marker>
  </defs>`;

  let tilesSvg = "";
  provs.forEach(p => {
    const [c, r] = PROV_GRID[p];
    const x = 10 + c * 89, y = 6 + r * 69;
    const regName = PROV_REG[p];
    const st = getProvinceStats(yd, p);
    const stOther = getProvinceStats(ydOther, p);
    const dTons = (st.grainKg - stOther.grainKg) / 1000.0;
    const dLease = Math.abs(st.leasedMu - stOther.leasedMu);
    const hasDiff = Math.abs(dTons) > 0.05 || dLease > 0.1 || (regMap[regName]?.policy !== otherRegMap[regName]?.policy);

    let bgFill = REG_TINTS[regName] || "#fff";
    let strokeCol = "#c8d4cb";
    let strokeW = "1.1";
    let tileOpacity = "1.0";
    if (state.colorMode === "diff") {
      if (Math.abs(dTons) > 0.05 || dLease > 0.1) {
        strokeCol = dTons >= 0 ? "#137953" : "#c04851";
        strokeW = "2.2";
        bgFill = dTons >= 0 ? "#eaf7f0" : "#fdf1f2";
      } else {
        tileOpacity = "0.55";
      }
    }

    const rr = regMap[regName] || {};
    const polShort = fmtPolicyCn(rr.policy);
    // Micro-grid packing of household nodes inside tile
    const hhIds = st.actIds.filter(a => yd.actors[a][0] === "survey_household");
    const n = hhIds.length;
    const cols = Math.max(2, Math.ceil(Math.sqrt(Math.max(1, n) * 1.95)));
    const nr = Math.ceil(n / cols);
    const rad = Math.max(1.8, Math.min(3.3, 28 / cols, 12 / Math.max(1, nr)));
    const gx0 = x + 8, gy0 = y + 21;
    const gdx = (86 - 16) / cols, gdy = 30 / Math.max(1, nr);
    let dots = "";
    hhIds.forEach((aid, idx) => {
      const cx = gx0 + (idx % cols + 0.5) * gdx;
      const cy = gy0 + (Math.floor(idx / cols) + 0.5) * gdy;
      const fk = st.actorFates[aid] || "own_idle";
      const dcol = (FATE_STYLES[fk] || ["#adb6b1"])[0];
      dots += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${rad.toFixed(1)}" fill="${dcol}"/>`;
    });

    const orgCount = st.actIds.filter(a => yd.actors[a][0] !== "survey_household").length;
    const bottomTxt = state.colorMode === "diff"
      ? `${dTons >= 0 ? "+" : ""}${dTons.toFixed(1)}t`
      : `${(st.grainKg/1000).toFixed(1)}t${orgCount ? " · " + orgCount + "体" : ""}`;

    tilesSvg += `<g class="prov-tile" data-prov="${p}" tabindex="0" role="button" aria-label="查看${p}流转网络" opacity="${tileOpacity}">
      <rect class="tile-bg" x="${x}" y="${y}" width="86" height="64" rx="4" fill="${bgFill}" stroke="${strokeCol}" stroke-width="${strokeW}"/>
      <text x="${x+5}" y="${y+13}" font-size="11.5" font-weight="700" fill="#273e35">${p}</text>
      <text x="${x+81}" y="${y+13}" font-size="9.5" text-anchor="end" fill="#5b6d64">${st.actIds.length}</text>
      ${dots}
      <text x="${x+5}" y="${y+59}" font-size="8.6" fill="#4f6158">${polShort.slice(0,5)}</text>
      <text x="${x+81}" y="${y+59}" font-size="8.8" font-weight="600" text-anchor="end" fill="${state.colorMode==='diff' && dTons>0.05 ? '#137953' : '#33463d'}">${bottomTxt}</text>
    </g>`;
  });

  // Curved Interprovincial Arbitrage Flow Arrows
  let flowsSvg = "";
  const flows = Object.entries(yd.pair_flows || {}).filter(([, kg]) => kg > 10.0);
  flows.forEach(([key, kg], idx) => {
    const parts = key.split("<-");
    if (parts.length !== 2) return;
    const [tgtProv, srcProv] = parts;
    if (filterReg && PROV_REG[tgtProv] !== filterReg && PROV_REG[srcProv] !== filterReg) return;
    if (!PROV_GRID[srcProv] || !PROV_GRID[tgtProv]) return;
    const [x1, y1] = tileCenter(srcProv);
    const [x2, y2] = tileCenter(tgtProv);
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const sx = x1 + (dx / len) * 20, sy = y1 + (dy / len) * 20;
    const tx = x2 - (dx / len) * 22, ty = y2 - (dy / len) * 22;
    const nx = -dy / len, ny = dx / len;
    const curve = (idx % 2 === 0 ? 1 : -1) * 26;
    const cx = 0.5 * (sx + tx) + curve * nx;
    const cy = 0.5 * (sy + ty) + curve * ny;
    const tons = kg / 1000.0;
    const sw = Math.min(6.2, 2.0 + tons * 0.08);
    const lx = 0.25 * sx + 0.5 * cx + 0.25 * tx;
    const ly = 0.25 * sy + 0.5 * cy + 0.25 * ty;
    flowsSvg += `<g>
      <path d="M${sx.toFixed(1)},${sy.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${tx.toFixed(1)},${ty.toFixed(1)}"
            fill="none" stroke="#c04851" stroke-width="${sw.toFixed(2)}" stroke-opacity="0.86"
            marker-end="url(#${markerPrefix}Arb)"/>
      <rect x="${(lx-19).toFixed(1)}" y="${(ly-8).toFixed(1)}" width="38" height="15" rx="3" fill="rgba(255,255,255,0.92)" stroke="#c04851" stroke-width="0.7"/>
      <text x="${lx.toFixed(1)}" y="${(ly+3).toFixed(1)}" font-size="9.5" font-weight="700" text-anchor="middle" fill="#9c2b35">${tons.toFixed(1)}t</text>
    </g>`;
  });

  svg.innerHTML = defs + tilesSvg + flowsSvg;
  svg.querySelectorAll(".prov-tile").forEach(el => {
    const go = () => {
      state.scope = "prov:" + el.dataset.prov;
      state.selectedActor = null;
      scopeSel.value = state.scope;
      renderAll();
    };
    el.addEventListener("click", go);
    el.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); } });
  });
}

function actorLabelCn(aid) {
  if (aid.startsWith("FF_")) return "家庭农场" + aid.split("_").slice(-1)[0];
  if (aid.startsWith("COOP_")) return "合作社" + aid.split("_").slice(-1)[0];
  if (aid.startsWith("CORP_")) return "农业企业" + aid.split("_").slice(-1)[0];
  return aid;
}

function hexPoints(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return pts.join(" ");
}

function renderProvinceNetworkSvg(svgId, markerPrefix, provName, yd, ydOther) {
  const svg = document.getElementById(svgId);
  svg.setAttribute("viewBox", "0 0 760 460");

  const provParcels = yd.parcels.filter(p => p[2] === provName);
  const hhSet = new Set();
  const orgSet = new Set();
  for (const [aid, row] of Object.entries(yd.actors)) {
    if (row[14] === provName) {
      if (row[0] === "survey_household") hhSet.add(aid);
      else orgSet.add(aid);
    }
  }
  provParcels.forEach(p => {
    [p[3], p[4]].forEach(aid => {
      const cat = yd.actors[aid]?.[0];
      if (cat === "survey_household") hhSet.add(aid);
      else if (cat) orgSet.add(aid);
    });
  });

  const hhList = Array.from(hhSet).sort();
  const orgList = Array.from(orgSet).sort();
  const posMap = {};

  // Pack households in upper area (34..726, 28..330)
  const nh = hhList.length;
  const cols = Math.max(2, Math.ceil(Math.sqrt(Math.max(1, nh) * 1.75)));
  const rows = Math.max(1, Math.ceil(nh / cols));
  const dx = 656 / cols, dy = 290 / rows;
  const baseR = Math.max(7.5, Math.min(18, 0.28 * Math.min(dx, dy)));
  hhList.forEach((aid, idx) => {
    const cx = 52 + (idx % cols + 0.5) * dx;
    const cy = 28 + (Math.floor(idx / cols) + 0.5) * dy;
    posMap[aid] = { x: cx, y: cy, r: baseR };
  });

  // Dock scale organizations along clean bottom strip (y=396)
  if (orgList.length > 0) {
    const stepX = 600 / (orgList.length + 1);
    orgList.forEach((aid, idx) => {
      posMap[aid] = { x: 80 + (idx + 1) * stepX, y: 398, r: 15.5 };
    });
  }

  // Primary fate per actor
  const actorFates = {};
  const pri = {"strict_succession_grain":8,"leased_unplanted":7,"leased_cash":6,"idle_recovered_grain":5,"converted_to_grain":4,"own_cash":3,"incumbent_grain":2,"own_idle":1};
  provParcels.forEach(p => {
    [p[3], p[4]].forEach(aid => {
      const cur = actorFates[aid];
      if (!cur || (pri[p[6]] || 0) > (pri[cur] || 0)) actorFates[aid] = p[6];
    });
  });

  // Aggregate directed lease edges owner -> operator
  const edgeMap = {};
  provParcels.forEach(p => {
    if (p[3] === p[4]) return;
    const ek = `${p[3]}->${p[4]}`;
    if (!edgeMap[ek]) edgeMap[ek] = { src: p[3], dst: p[4], area: 0, rent: 0, n: 0, fate: p[6] };
    edgeMap[ek].area += p[7];
    edgeMap[ek].rent += p[9];
    edgeMap[ek].n += 1;
    if ((pri[p[6]] || 0) > (pri[edgeMap[ek].fate] || 0)) edgeMap[ek].fate = p[6];
  });
  const edges = Object.values(edgeMap);
  const sel = state.selectedActor;

  const incidentActors = new Set();
  if (sel) {
    incidentActors.add(sel);
    edges.forEach(e => { if (e.src === sel || e.dst === sel) { incidentActors.add(e.src); incidentActors.add(e.dst); } });
  }

  let defs = `<defs>
    <marker id="${markerPrefix}G" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,1 L7,4 L0,7 Z" fill="#1e7f5c"/></marker>
    <marker id="${markerPrefix}W" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,1 L7,4 L0,7 Z" fill="#c26e1f"/></marker>
    <marker id="${markerPrefix}R" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,1 L7,4 L0,7 Z" fill="#c04851"/></marker>
    <marker id="${markerPrefix}F" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,1 L7,4 L0,7 Z" fill="#1b4353"/></marker>
  </defs>`;

  let bg = "";
  if (orgList.length > 0) {
    bg = `<rect x="28" y="362" width="704" height="76" rx="6" fill="#ffffff" stroke="#d0dcd5" stroke-width="1"/>
          <text x="40" y="378" font-size="11" font-weight="600" fill="#596660">省内合成规模经营主体停靠区（严格与调查小农分账）</text>`;
  }

  // Render edges (sort focused edges to end so they draw on top)
  edges.sort((a, b) => ((a.src === sel || a.dst === sel) ? 1 : 0) - ((b.src === sel || b.dst === sel) ? 1 : 0));
  let edgesSvg = "";
  edges.forEach(e => {
    const p1 = posMap[e.src], p2 = posMap[e.dst];
    if (!p1 || !p2) return;
    const dx = p2.x - p1.x, dy = p2.y - p1.y, len = Math.hypot(dx, dy) || 1;
    const x1 = p1.x + (dx / len) * (p1.r + 2), y1 = p1.y + (dy / len) * (p1.r + 2);
    const x2 = p2.x - (dx / len) * (p2.r + 4), y2 = p2.y - (dy / len) * (p2.r + 4);
    const isFocus = sel && (e.src === sel || e.dst === sel);
    const dimCls = sel && !isFocus ? "dimmed" : (isFocus ? "focus" : "");
    let col = "#1e7f5c", dash = "", mk = `${markerPrefix}G`;
    if (e.fate === "leased_cash") { col = "#c26e1f"; dash = 'stroke-dasharray="4 3"'; mk = `${markerPrefix}W`; }
    else if (e.fate === "leased_unplanted") { col = "#c04851"; dash = 'stroke-dasharray="4 3"'; mk = `${markerPrefix}R`; }
    if (isFocus) mk = `${markerPrefix}F`;
    const sw = Math.min(3.5, 1.3 + e.area * 0.12);
    edgesSvg += `<line class="link-edge ${dimCls}" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
      stroke="${isFocus ? '#1b4353' : col}" stroke-width="${sw.toFixed(2)}" ${dash} marker-end="url(#${mk})">
      <title>${actorLabelCn(e.src)} → ${actorLabelCn(e.dst)} | ${e.n}块 ${e.area.toFixed(2)}亩 | 租金 ${e.rent.toFixed(0)}元</title></line>`;
  });

  // Render actors
  let nodesSvg = "";
  [...hhList, ...orgList].forEach(aid => {
    const pt = posMap[aid];
    const row = yd.actors[aid] || [];
    const rowO = ydOther.actors[aid] || [];
    const cat = row[0] || "survey_household";
    const isSel = sel === aid;
    let dim = sel && !incidentActors.has(aid) ? 0.18 : 1.0;

    let fill = (FATE_STYLES[actorFates[aid] || "own_idle"] || ["#adb6b1"])[0];
    let stroke = "#ffffff", sw = isSel ? 3.0 : 1.3;
    if (isSel) stroke = "#182e47";
    if (state.colorMode === "diff") {
      const diffG = Math.abs((row[4]||0) - (rowO[4]||0)) > 5 || Math.abs((row[1]||0) - (rowO[1]||0)) > 0.05;
      if (diffG) { stroke = "#b66c26"; sw = 2.6; }
      else if (!isSel) { dim = Math.min(dim, 0.35); }
    }

    let shape = "";
    if (cat === "survey_household") {
      const rr = pt.r + Math.min(4.5, (row[1] || 0) * 0.14);
      shape = `<circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="${rr.toFixed(1)}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
    } else if (cat === "family_farm") {
      const r = 13.5;
      shape = `<polygon points="${pt.x},${pt.y-r} ${pt.x+r},${pt.y} ${pt.x},${pt.y+r} ${pt.x-r},${pt.y}" fill="#1e7f5c" stroke="${stroke}" stroke-width="${sw}"/>`;
    } else if (cat === "cooperative") {
      shape = `<polygon points="${hexPoints(pt.x, pt.y, 14)}" fill="#2b62a8" stroke="${stroke}" stroke-width="${sw}"/>`;
    } else {
      shape = `<rect x="${pt.x-12}" y="${pt.y-12}" width="24" height="24" rx="4" fill="#77649a" stroke="${stroke}" stroke-width="${sw}"/>`;
    }
    const lbl = cat === "survey_household" ? aid.slice(-3) : actorLabelCn(aid);
    const lblY = cat === "survey_household" ? pt.y + pt.r + 11 : pt.y + 25;
    nodesSvg += `<g class="actor-node" data-actor="${aid}" opacity="${dim}" tabindex="0" role="button" aria-label="${aid} (${CAT_CN[cat]||cat})">
      ${shape}
      <text x="${pt.x.toFixed(1)}" y="${lblY.toFixed(1)}" font-size="${cat==='survey_household' ? 8.8 : 10.2}" font-weight="${cat==='survey_household' ? 400 : 700}" text-anchor="middle" fill="#24352e">${lbl}</text>
      <title>${actorLabelCn(aid)} (${CAT_CN[cat]||cat}) | 经营:${row[1]||0}亩 | 产粮:${row[4]||0}kg</title>
    </g>`;
  });

  svg.innerHTML = defs + bg + edgesSvg + nodesSvg;
  svg.querySelectorAll(".actor-node").forEach(el => {
    const pick = () => { state.selectedActor = el.dataset.actor; renderAll(); };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
  });
}

function renderLedgers(trA, trB, ydA, ydB) {
  const jA = ydA.joint, jB = ydB.joint;
  const vcA = ydA.regions.reduce((s, r) => s + (r.village_coord_fee_yuan || 0), 0);
  const vcB = ydB.regions.reduce((s, r) => s + (r.village_coord_fee_yuan || 0), 0);
  const poolA = Math.max(...ydA.regions.map(r => r.village_pool_remaining_years || 0));
  const poolB = Math.max(...ydB.regions.map(r => r.village_pool_remaining_years || 0));

  document.getElementById("kpiStrip").innerHTML = `
    <div><span class="m-lbl">当年全国总产粮（B vs A）</span><b>${(jB.grain_kg/1000).toFixed(1)} 吨</b><span class="m-sub">较 A ${fmtDelta((jB.grain_kg - jA.grain_kg)/1000, 1, " 吨")}</span></div>
    <div><span class="m-lbl">当年实际支出 / 硬上限</span><b>${(jB.spending/10000).toFixed(2)} 万元</b><span class="m-sub">上限 15.00 万（整治费 ${(vcB/10000).toFixed(2)} 万）</span></div>
    <div><span class="m-lbl">严格换人续粮 / 单位地租</span><b>${jB.fate_2_strict_succession_grain_mu.toFixed(1)} 亩</b><span class="m-sub">${jB.unit_rent_yuan_mu.toFixed(0)} 元/亩（较 A ${fmtDelta(jB.unit_rent_yuan_mu - jA.unit_rent_yuan_mu, 0, " 元")}）</span></div>
    <div><span class="m-lbl">跨省购粮套利 / 守恒误差</span><b>${(jB.interprov_arbitrage_grain_kg/1000).toFixed(1)} 吨</b><span class="m-sub">较 A ${fmtDelta((jB.interprov_arbitrage_grain_kg - jA.interprov_arbitrage_grain_kg)/1000, 1, " t")} · 误差 ${(jB.bilateral_conservation_gap_kg||0).toFixed(2)} kg</span></div>
  `;

  document.getElementById("titleA").textContent = `策略 A：${trA.label}`;
  document.getElementById("titleB").textContent = `策略 B：${trB.label}`;
  document.getElementById("badgeA").textContent = `第 ${state.yr} 年 · 实支 ${(jA.spending/10000).toFixed(2)} 万`;
  document.getElementById("badgeB").textContent = `第 ${state.yr} 年 · 实支 ${(jB.spending/10000).toFixed(2)} 万`;

  const consOk = Math.abs(jA.bilateral_conservation_gap_kg || 0) < 1e-4 && Math.abs(jB.bilateral_conservation_gap_kg || 0) < 1e-4;
  document.getElementById("consBadge").textContent = consOk
    ? "双边实物守恒误差：0.00 kg · 零原生身份暴露"
    : "检测到守恒偏差，请核对";

  const macroRows = [
    ["全国当年粮食总产量 (吨)", jA.grain_kg/1000, jB.grain_kg/1000, 2],
    ["其中：600 调查户产粮 (吨)", jA.survey_600_grain_kg/1000, jB.survey_600_grain_kg/1000, 2],
    ["新增财政实际总支出 (万元)", jA.spending/10000, jB.spending/10000, 3],
    ["连片整治核定服务费 (万元)", vcA/10000, vcB/10000, 2],
    ["在租流转总面积 (亩)", jA.leased_area_mu, jB.leased_area_mu, 1],
    ["换人严格接续种粮 (亩)", jA.fate_2_strict_succession_grain_mu, jB.fate_2_strict_succession_grain_mu, 1],
    ["流转平均单位地租 (元/亩)", jA.unit_rent_yuan_mu, jB.unit_rent_yuan_mu, 1],
    ["跨省低价买粮顶抵 (吨)", jA.interprov_arbitrage_grain_kg/1000, jB.interprov_arbitrage_grain_kg/1000, 2],
    ["跨省套补财政漏损 (万元)", jA.interprov_arbitrage_subsidy_leakage_yuan/10000, jB.interprov_arbitrage_subsidy_leakage_yuan/10000, 3],
    ["连片组织池年末剩余 (年)", poolA, poolB, 0],
    ["省际双边余粮守恒差 (kg)", jA.bilateral_conservation_gap_kg||0, jB.bilateral_conservation_gap_kg||0, 2]
  ];
  let mHtml = `<table><thead><tr><th>宏观账目指标</th><th>策略 A</th><th>策略 B</th><th>差值 (B−A)</th></tr></thead><tbody>`;
  macroRows.forEach(([nm, vA, vB, dg]) => {
    mHtml += `<tr><td>${nm}</td><td>${vA.toFixed(dg)}</td><td>${vB.toFixed(dg)}</td><td>${fmtDelta(vB - vA, dg)}</td></tr>`;
  });
  mHtml += `</tbody></table><h3 style="margin-top:12px">四大政策区工具配比与分区分账（第 ${state.yr} 年）</h3>
    <table><thead><tr><th>区域</th><th>A 工具</th><th>B 工具</th><th>A 产粮(吨)</th><th>B 产粮(吨)</th><th>B−A 产粮差</th></tr></thead><tbody>`;
  ydA.regions.forEach((rA, i) => {
    const rB = ydB.regions[i];
    mHtml += `<tr><td>${rA.region}</td><td>${fmtPolicyCn(rA.policy)}</td><td>${fmtPolicyCn(rB.policy)}</td>
      <td>${(rA.grain_kg/1000).toFixed(1)}</td><td>${(rB.grain_kg/1000).toFixed(1)}</td>
      <td>${fmtDelta((rB.grain_kg - rA.grain_kg)/1000, 1, " t")}</td></tr>`;
  });
  mHtml += `</tbody></table>`;
  document.getElementById("macroLedger").innerHTML = mHtml;

  // Right White-Box Inspector: Actor if selected, else Province summary if in prov:, else Cross-border flows summary
  const microTitle = document.getElementById("microTitle");
  const microBox = document.getElementById("microLedger");
  if (state.selectedActor && (ydA.actors[state.selectedActor] || ydB.actors[state.selectedActor])) {
    const aid = state.selectedActor;
    const aA = ydA.actors[aid] || new Array(17).fill(0);
    const aB = ydB.actors[aid] || new Array(17).fill(0);
    const cat = CAT_CN[aA[0] || aB[0]] || "经营主体";
    const pName = aA[14] || aB[14] || "";
    microTitle.textContent = `微观主体白箱账本：${actorLabelCn(aid)}（${pName} · ${cat}）`;
    const rows = [
      ["物理经营面积 (亩)", aA[1], aB[1], 2],
      ["粮食播种面积 (亩次)", aA[2], aB[2], 2],
      ["经济作物面积 (亩次)", aA[3], aB[3], 2],
      ["当年收获总产粮 (kg)", aA[4], aB[4], 0],
      ["其中自留口粮 (kg)", aA[5], aB[5], 0],
      ["出售商品余粮 (kg)", aA[6], aB[6], 0],
      ["商品售粮现金 (元)", aA[7], aB[7], 0],
      ["转出净租金收入 (元)", aA[8], aB[8], 0],
      ["转入流转地租支出 (元)", aA[9], aB[9], 0],
      ["当年获得新增专项 (元)", aA[10], aB[10], 0],
      ["耕地地力底盘补贴 (元)", aA[11], aB[11], 0],
      ["全年综合净收益 (元)", aA[12], aB[12], 0],
      ["年末春耕备用金 (元)", aA[13], aB[13], 0]
    ];
    let h = `<table><thead><tr><th>主体收支与口粮账目</th><th>策略 A</th><th>策略 B</th><th>差值 (B−A)</th></tr></thead><tbody>`;
    rows.forEach(([nm, vA, vB, dg]) => {
      h += `<tr><td>${nm}</td><td>${Number(vA||0).toFixed(dg)}</td><td>${Number(vB||0).toFixed(dg)}</td><td>${fmtDelta((vB||0)-(vA||0), dg)}</td></tr>`;
    });
    h += `</tbody></table>`;
    // Counterparty links under Strategy B
    const peers = [];
    const seenP = new Set();
    ydB.parcels.forEach(p => {
      if (p[3] !== p[4]) {
        if (p[3] === aid && !seenP.has("to:" + p[4])) { seenP.add("to:" + p[4]); peers.push(["租给", p[4]]); }
        if (p[4] === aid && !seenP.has("fr:" + p[3])) { seenP.add("fr:" + p[3]); peers.push(["租自", p[3]]); }
      }
    });
    if (peers.length) {
      h += `<div style="margin-top:8px;font-size:12px;color:#4b6156"><strong>策略 B 下合同关联方（点击切换）：</strong><br>`;
      peers.slice(0, 12).forEach(([rel, peerId]) => {
        h += `<button class="peer-btn" data-peer="${peerId}" type="button">${rel} ${actorLabelCn(peerId)}</button>`;
      });
      h += `</div>`;
    }
    microBox.innerHTML = h;
    microBox.querySelectorAll(".peer-btn").forEach(btn => {
      btn.addEventListener("click", () => { state.selectedActor = btn.dataset.peer; renderAll(); });
    });
  } else if (state.scope.startsWith("prov:")) {
    const prov = state.scope.slice(5);
    const sA = getProvinceStats(ydA, prov);
    const sB = getProvinceStats(ydB, prov);
    microTitle.textContent = `省级白箱汇总：${prov}（${PROV_REG[prov]} · 请点击上方图中节点看个体账目）`;
    const rows = [
      ["省内样本经营单元数 (个)", sA.provParcels.length, sB.provParcels.length, 0],
      ["当年粮食总产量 (吨)", sA.grainKg/1000, sB.grainKg/1000, 2],
      ["省内在租流转面积 (亩)", sA.leasedMu, sB.leasedMu, 2],
      ["换人严格接续种粮 (亩)", sA.strictMu, sB.strictMu, 2],
      ["省内新增专项补贴 (万元)", sA.supYuan/10000, sB.supYuan/10000, 3]
    ];
    let h = `<table><thead><tr><th>${prov}核算指标</th><th>策略 A</th><th>策略 B</th><th>差值 (B−A)</th></tr></thead><tbody>`;
    rows.forEach(([nm, vA, vB, dg]) => {
      h += `<tr><td>${nm}</td><td>${vA.toFixed(dg)}</td><td>${vB.toFixed(dg)}</td><td>${fmtDelta(vB - vA, dg)}</td></tr>`;
    });
    h += `</tbody></table>`;
    microBox.innerHTML = h;
  } else {
    microTitle.textContent = `毗邻省际订单买粮通道流量与全国对比（点击地图省格深入查看流转网络）`;
    const allKeys = Array.from(new Set([...Object.keys(ydA.pair_flows||{}), ...Object.keys(ydB.pair_flows||{})]));
    if (allKeys.length === 0) {
      microBox.innerHTML = `<p style="font-size:13px;color:#596660;margin-top:8px">第 ${state.yr} 年当前两策略均未触发跨省买粮顶抵套利流量（0.00 t）。点击地图中任一省份方格（例如 <strong>黑龙江</strong> 或 <strong>安徽</strong>），即可下钻查看该省 600 户调查家庭与合成规模主体的有向流转契约网络。</p>`;
    } else {
      let h = `<table><thead><tr><th>跨省毗邻粮道 (销区←产区)</th><th>策略 A (吨)</th><th>策略 B (吨)</th><th>差值 (B−A)</th></tr></thead><tbody>`;
      allKeys.sort((k1, k2) => ((ydA.pair_flows[k2]||0) - (ydA.pair_flows[k1]||0))).forEach(k => {
        const tA = (ydA.pair_flows[k] || 0) / 1000.0;
        const tB = (ydB.pair_flows[k] || 0) / 1000.0;
        h += `<tr><td>${k.replace("<-", " ← ")}</td><td>${tA.toFixed(2)}</td><td>${tB.toFixed(2)}</td><td>${fmtDelta(tB - tA, 2, " t")}</td></tr>`;
      });
      h += `</tbody></table>`;
      microBox.innerHTML = h;
    }
  }
}

function renderAll() {
  const trA = DATA.traces[state.polA];
  const trB = DATA.traces[state.polB];
  const ydA = trA.years[state.yr - 1];
  const ydB = trB.years[state.yr - 1];

  document.getElementById("yrSlider").value = String(state.yr);
  document.getElementById("yrLabel").textContent = `第 ${state.yr} 年`;
  document.getElementById("btnPrev").disabled = state.yr <= 1;
  document.getElementById("btnNext").disabled = state.yr >= 8;
  document.getElementById("btnNational").disabled = state.scope === "全国";
  document.getElementById("modeFate").setAttribute("aria-pressed", state.colorMode === "fate" ? "true" : "false");
  document.getElementById("modeDiff").setAttribute("aria-pressed", state.colorMode === "diff" ? "true" : "false");

  if (state.scope.startsWith("prov:")) {
    const prov = state.scope.slice(5);
    renderProvinceNetworkSvg("svgA", "pA", prov, ydA, ydB);
    renderProvinceNetworkSvg("svgB", "pB", prov, ydB, ydA);
  } else {
    const reg = state.scope.startsWith("reg:") ? state.scope.slice(4) : null;
    renderCartogramSvg("svgA", "cA", ydA, ydB, reg);
    renderCartogramSvg("svgB", "cB", ydB, ydA, reg);
  }
  renderLedgers(trA, trB, ydA, ydB);
}

function stopPlay() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  state.playing = false;
  const bp = document.getElementById("btnPlay");
  bp.textContent = "▶"; bp.title = "播放"; bp.setAttribute("aria-label", "播放");
}

document.getElementById("btnPlay").addEventListener("click", () => {
  if (state.playing) { stopPlay(); return; }
  if (state.yr >= 8) state.yr = 1;
  state.playing = true;
  const bp = document.getElementById("btnPlay");
  bp.textContent = "Ⅱ"; bp.title = "暂停"; bp.setAttribute("aria-label", "暂停");
  renderAll();
  state.timer = setInterval(() => {
    if (state.yr >= 8) { stopPlay(); return; }
    state.yr += 1;
    renderAll();
  }, 1250);
});
document.getElementById("btnPrev").addEventListener("click", () => { stopPlay(); if (state.yr > 1) { state.yr -= 1; renderAll(); } });
document.getElementById("btnNext").addEventListener("click", () => { stopPlay(); if (state.yr < 8) { state.yr += 1; renderAll(); } });
document.getElementById("yrSlider").addEventListener("input", ev => { stopPlay(); state.yr = parseInt(ev.target.value, 10); renderAll(); });
scopeSel.addEventListener("change", ev => { stopPlay(); state.scope = ev.target.value; state.selectedActor = null; renderAll(); });
document.getElementById("btnNational").addEventListener("click", () => { stopPlay(); state.scope = "全国"; state.selectedActor = null; scopeSel.value = "全国"; renderAll(); });
polASel.addEventListener("change", ev => { state.polA = ev.target.value; renderAll(); });
polBSel.addEventListener("change", ev => { state.polB = ev.target.value; renderAll(); });
document.getElementById("modeFate").addEventListener("click", () => { state.colorMode = "fate"; renderAll(); });
document.getElementById("modeDiff").addEventListener("click", () => { state.colorMode = "diff"; renderAll(); });

document.getElementById("presetArb").addEventListener("click", () => {
  stopPlay();
  state.polA = "east_loose_arb"; state.polB = "east_capped_arb";
  polASel.value = state.polA; polBSel.value = state.polB;
  state.yr = 3; state.scope = "全国"; scopeSel.value = "全国"; state.selectedActor = null;
  document.getElementById("presetArb").classList.add("active");
  document.getElementById("presetPulse").classList.remove("active");
  renderAll();
});
document.getElementById("presetPulse").addEventListener("click", () => {
  stopPlay();
  state.polA = "exp5b_regional"; state.polB = "exp5b_rl";
  polASel.value = state.polA; polBSel.value = state.polB;
  state.yr = 5; state.scope = "全国"; scopeSel.value = "全国"; state.selectedActor = null;
  document.getElementById("presetPulse").classList.add("active");
  document.getElementById("presetArb").classList.remove("active");
  renderAll();
});
document.getElementById("presetProv").addEventListener("click", () => {
  stopPlay();
  const choices = ["黑龙江", "安徽", "浙江", "江西"];
  const prov = choices[state.provToggle % choices.length];
  state.provToggle += 1;
  state.scope = "prov:" + prov; scopeSel.value = state.scope; state.selectedActor = null;
  renderAll();
});

renderAll();
</script>
</body>
</html>
"""


def _int_keys(raw_map: dict | None) -> dict[int, object] | None:
    if raw_map is None:
        return None
    return {int(k): v for k, v in raw_map.items()}


def export_v2_2_spatial_workbench(
    output_html: Path,
    project_root: Path,
    *,
    seed: int = 574001,
) -> Path:
    """Export desensitized V2.2 Spatial White-Box Workbench fusing China Tile Cartogram + Lease Network."""
    wt_root = Path(__file__).resolve().parents[1]
    fqi_json = wt_root / "docs/Mesa农业经营接续V2重构/结果_V2_2净版/exp5b_mlp_fqi_training_summary.json"
    summary_5b = json.loads(fqi_json.read_text(encoding="utf-8")) if fqi_json.exists() else {}
    r_idx_5b = int(summary_5b.get("best_regional_fixed_idx", 33))
    tree_q_w = _int_keys(summary_5b.get("tree_q_weights") or summary_5b.get("mlp_q_weights"))
    tree_myo_w = _int_keys(summary_5b.get("tree_myopic_weights") or summary_5b.get("mlp_myopic_weights"))

    d_clean = clean_v2_2_design()
    d_loose = d_clean.with_overrides(enable_interprovincial_arbitrage=True, order_quota_mode="loose_area_quota")
    d_capped = d_clean.with_overrides(enable_interprovincial_arbitrage=True, order_quota_mode="verified_grain_area_cap")
    d_5b = intertemporal_v2_2_design()

    regs_clean, _ = load_v2_calibrated_regions(project_root, d_clean)
    regs_5b, _ = load_v2_calibrated_regions(project_root, d_5b)

    east_75k_order: JointActionV2 = (("order_farming", 75_000), ("none", 0), ("none", 0), ("none", 0))

    traces = {
        "p0": capture_policy_mesa_trace(
            seed, regs_clean, d_clean, label="无财政支持基线（仅耕地地力底盘）", policy_mode="p0", include_v22_spatial=True
        ),
        "uniform_order": capture_policy_mesa_trace(
            seed, regs_clean, d_clean, label="全国统一固定（封顶订单粮食奖励）", policy_mode="uniform_fixed", fixed_action_idx=28, include_v22_spatial=True
        ),
        "east_loose_arb": capture_policy_mesa_trace(
            seed, regs_clean, d_loose, label="东部宽松订单（跨省买粮套利对照）", policy_mode="fixed",
            explicit_joint_action=east_75k_order, include_v22_spatial=True
        ),
        "east_capped_arb": capture_policy_mesa_trace(
            seed, regs_clean, d_capped, label="东部实种400 kg封顶订单（阻断套利）", policy_mode="fixed",
            explicit_joint_action=east_75k_order, include_v22_spatial=True
        ),
        "exp5b_regional": capture_policy_mesa_trace(
            seed, regs_5b, d_5b, label="拓展情景：事后区域最优（年付整治费）", policy_mode="regional_fixed",
            fixed_action_idx=r_idx_5b, include_v22_spatial=True
        ),
        "exp5b_myopic": capture_policy_mesa_trace(
            seed, regs_5b, d_5b, label="拓展情景：单期短视寻优（树集成，贴现0.00）", policy_mode="myopic",
            learned_weights=tree_myo_w, include_v22_spatial=True
        ),
        "exp5b_rl": capture_policy_mesa_trace(
            seed, regs_5b, d_5b, label="拓展情景：跨期强化学习（树集成，贴现0.96）", policy_mode="gov_rl",
            learned_weights=tree_q_w, include_v22_spatial=True
        ),
    }

    payload = {"seed": seed, "budget_yuan": d_clean.annual_common_budget_yuan, "traces": traces}
    html_str = (
        WORKBENCH_V2_2_HTML
        .replace("__SEED__", str(seed))
        .replace("__MESA_V22_PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_str, encoding="utf-8")
    return output_html


def serve_live_inspector(project_root: Path, port: int = 8765) -> None:
    d = default_v2_design()
    regions, _ = load_v2_calibrated_regions(project_root, d)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            seed = int(qs.get("seed", ["574001"])[0])
            traces = {
                "p0": capture_policy_mesa_trace(seed, regions, d, label=f"P0 基准 (Seed {seed})", policy_mode="p0"),
                "prod": capture_policy_mesa_trace(seed, regions, d, label=f"种粮者播前直补 (Seed {seed})", policy_mode="uniform_fixed", fixed_action_idx=1),
                "scale": capture_policy_mesa_trace(seed, regions, d, label=f"适度规模补贴 (Seed {seed})", policy_mode="uniform_fixed", fixed_action_idx=3),
                "svc": capture_policy_mesa_trace(seed, regions, d, label=f"生产托管补助 (Seed {seed})", policy_mode="uniform_fixed", fixed_action_idx=5),
                "trans": capture_policy_mesa_trace(seed, regions, d, label=f"透明动态规则 (Seed {seed})", policy_mode="transparent_rule"),
            }
            html = HTML_TEMPLATE.replace(
                "__MESA_PAYLOAD__",
                json.dumps({"seed": seed, "budget_yuan": d.annual_common_budget_yuan, "traces": traces}, ensure_ascii=False, separators=(",", ":")),
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Live Mesa V2 Inspector serving at http://127.0.0.1:{port}/?seed=574001")
    srv.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=resolve_project_data_root(Path(__file__)))
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--export-v2-2", type=Path, default=None)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.export:
        export_inspector_html(args.export, args.project_root)
    if args.export_v2_2:
        out_p = export_v2_2_spatial_workbench(args.export_v2_2, args.project_root)
        print(f"Exported V2.2 Spatial White-Box Workbench -> {out_p} ({out_p.stat().st_size} bytes)")
    elif args.serve and not args.export:
        serve_live_inspector(args.project_root, args.port)
