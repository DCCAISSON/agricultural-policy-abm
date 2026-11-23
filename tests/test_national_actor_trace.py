import json
from pathlib import Path
from types import SimpleNamespace

from scripts import national_actor_trace as trace
from scripts import run_national_government_q_20261119 as runner


def test_frozen_first_year(tmp_path):
    hashes = runner.source_hashes()
    observer = Path(trace.__file__).resolve()
    frozen = {**hashes, str(observer): runner.old.digest(observer)}
    runner.old.write(tmp_path / "trace_source_hashes.json", frozen)
    base, economics, budget, _ = runner.load_base()
    model = runner.build_model(base, economics, 31)
    model.config["policy"]["government_q_project"]["episode_end_step"] = 1
    runner.configure_government(model.government, {}, 31, training=False, fixed_action=0)
    output = tmp_path / "episode"
    runner.run_episode(model, budget, output, 31, hashes, False, full_audit=True,
                       observe_year=trace.write_actor_trace)
    assert {**runner.source_hashes(), str(observer): runner.old.digest(observer)} == frozen
    directory = output / "year_01"
    def lines(name):
        return [json.loads(line) for line in (directory / name).read_text(encoding="utf8").splitlines()]
    rows, actors = lines("actor_trace.jsonl"), lines("actors.jsonl")
    assert len(rows) == len(actors) == len({r["actor_id"] for r in rows})
    decisions = runner.old.read(directory / "decision_trace.json")
    tx = runner.old.read(directory / "lease_transactions.json")
    service = runner.old.read(directory / "service_tasks.json")
    payments = runner.old.read(directory / "project_payments.json")
    land = {r["id"] for r in runner.old.read(directory / "land.json")}
    modes = [(action, mode) for action in ("maintain_grain", "expand_cash")
             for mode in ("self", "outsource")] + [("exit", "transfer")]
    counts = {"household": 0, "Cooperative": 0, "Agri_Corporation": 0}
    for row in rows:
        key, links = row["actor_id"], row["links"]
        assert actors[links["actors.jsonl:account"][0]]["id"] == key
        if "decision_trace.json:operator" in links:
            dec = decisions[links["decision_trace.json:operator"][0]]
            assert (dec["operator_id"], dec["action"]) == (key, row["selected_action"])
            options = row["alternatives"]
            selected = options[row["selected_index"]]
            assert (selected["action"], selected["mode"]) == (dec["action"], row["selected_mode"])
            kind = row["enterprise_type"] or "household"
            counts[kind] += 1
            if kind == "household":
                assert [(a["action"], a["mode"]) for a in options] == modes
                assert selected["service_available"]
            else:
                assert [a["action"] for a in options] == ["maintain_grain", "expand_cash", "contract"]
        for role in ("supplier", "demander"):
            assert all(tx[i][role + "_id"] == key for i in links.get("lease_transactions.json:" + role, []))
        for i in links.get("service_tasks.json:paid", []):
            assert service[i]["operator_id"] == key and service[i]["p3_payment_step"] == 1
        for i in links.get("project_payments.json:recipient", []):
            assert payments[i]["recipient_id"] == key
        assert land.issuperset(links.get("actual_plot_use.json:patch_id:plot_use_operator", []))
    assert counts["household"] == 600
    assert counts["Cooperative"] + counts["Agri_Corporation"] == 60 and min(counts.values()) > 0


def test_previous_decision_not_reused_for_inactive_operator():
    stale = SimpleNamespace(p3_organization_accounts=[object()], fixed_rule_selected_account=object())
    assert trace._alternatives(stale, None) == (None, None, ["no_current_decision"])
