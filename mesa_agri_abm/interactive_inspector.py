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
from mesa_agri_abm.config import V2Design, default_v2_design
from mesa_agri_abm.model import AgriSuccessionMesaModel

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
    learned_weights: dict[int, list[list[float]]] | None = None,
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
        j_sum, r_sums, _ = m.step(collect_micro_ledger=True)
        # Compact tuple schema for fast browser parsing & GitHub-friendly file size (<8 MB):
        # parcel: [0:id, 1:reg, 2:prov, 3:own, 4:op, 5:op_cat, 6:fate, 7:area, 8:kg, 9:rent, 10:sup]
        # actor:  [0:id, 1:cat, 2:op_mu, 3:sow_g, 4:sow_c, 5:kg, 6:sub_kg, 7:mkt_kg, 8:sales, 9:rent_in, 10:rent_out, 11:sup, 12:fert, 13:inc, 14:wc]
        parcels_yr = []
        actors_yr = {}
        for reg_obj in m.regions:
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
                actors_yr[ar["actor"]] = [
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
        years_data.append({
            "year": yr,
            "joint": j_sum,
            "regions": list(r_sums),
            "parcels": parcels_yr,
            "actors": actors_yr,
        })
        if yr == design.withdrawal_after:
            wd_model = m.fork()
            wd_model.government.policy_mode = "p0"

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
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.export:
        export_inspector_html(args.export, args.project_root)
    elif args.serve:
        serve_live_inspector(args.project_root, args.port)
