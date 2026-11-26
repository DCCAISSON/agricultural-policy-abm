"""Reconcile succession ledgers and estimate paired, model-internal contrasts.

Year 0 in summaries is a path mean, not an observed year. Undefined market
prices/rates are missing; their n counts only usable paired paths. Rent and
income accounts are monetary identities, not a partition of physical effects.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cache
from hashlib import sha256
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.stats import t

from core.succession_allocation import CROPS

KEY = ("group", "condition", "seed", "year", "strategy", "policy")
CELL = ("group", "condition", "seed")
PATH = ("group", "condition", "strategy")
POLICIES = ("none", "grain_pre", "grain_post", "takeover_pre", "takeover_post")
YEARS = tuple(range(1, 9))
TOL = 1e-7
Row = dict[str, str]
Number = float | int
Output = dict[str, str | Number]


class LedgerError(ValueError):
    """A required ledger key or accounting identity is invalid."""


@dataclass(slots=True)
class Audit:
    source_sha256: str = field(default_factory=lambda: sha256(Path(__file__).read_bytes()).hexdigest())
    rows: Counter[str] = field(default_factory=Counter)
    failures: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, str | tuple[str, ...]]] = field(default_factory=list)
    key: tuple[str, ...] = ()

    def check(self, test: bool, name: str, detail: str = "") -> None:
        if not test:
            self.failures[name] += 1
            if len(self.examples) < 30:
                self.examples.append(dict(key=self.key, check=name, detail=detail))

    def equal(self, a: float, b: float, name: str, detail: str = "") -> None:
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=TOL):
            self.check(False, name, f"{detail}: {a:.12g} != {b:.12g}")

    def write(self, out: Path) -> None:
        receipt = dict(
            years=self.rows["annual"], plots=self.rows["plots"],
            operators=self.rows["operators"], transactions=self.rows["transactions"],
            fiscal=self.rows["fiscal"] if (out / "fiscal.csv.gz").is_file() else None,
            analysis_sha256=self.source_sha256,
            checks="failed" if self.failures else "passed",
            failure_counts=dict(self.failures), examples=self.examples,
        )
        (out / "ledger_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def _key(row: Row, fields: tuple[str, ...] = KEY) -> tuple[str, ...]:
    return tuple(row[k] for k in fields)


def _num(row: Row, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise LedgerError(f"Non-finite {key}: {row[key]}")
    return value


def _flag(row: Row, key: str) -> bool:
    value = row[key].lower()
    if value not in {"true", "false", "1", "0"}:
        raise LedgerError(f"Invalid flag {key}: {row[key]}")
    return value in {"true", "1"}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > TOL else math.nan


def _rows(stack: ExitStack, path: Path) -> Iterator[Row]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    source = stack.enter_context(opener(path, "rt", encoding="utf-8-sig", newline=""))
    reader = csv.DictReader(source)
    if not set(KEY).issubset(reader.fieldnames or ()):
        raise LedgerError(f"{path.name} lacks the annual key {KEY}")
    return reader


def _write(path: Path, rows: Iterable[Output], fields: Iterable[str] | None = None) -> None:
    records = iter(rows)
    first = next(records, None)
    if fields is None and first is None:
        return
    if fields is None:
        assert first is not None
        fields = first.keys()
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields))
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(records)


def _by_id(rows: Iterable[Row], id_field: str, name: str, audit: Audit) -> dict[str, Row]:
    result = {}
    for row in rows:
        audit.rows[name] += 1
        identity = row[id_field]
        audit.check(bool(identity) and identity not in result, f"{name}_unique_id", identity)
        result[identity] = row
    return result


@dataclass(slots=True)
class Year:
    annual: Row
    plots: dict[str, Row]
    actors: dict[str, Row]
    trades: dict[str, Row]
    fiscal: dict[str, Row] | None
    metrics: dict[str, float] = field(default_factory=dict)


def _stream(out: Path, audit: Audit) -> Iterator[Year]:
    """Join one annual block at a time; transaction blocks may be absent."""
    with ExitStack() as stack:
        annual = _rows(stack, out / "annual.csv")
        has_fiscal = (out / "fiscal.csv.gz").is_file()
        names = ("plots", "operators", "transactions") + (("fiscal",) if has_fiscal else ())
        blocks = {
            name: groupby(_rows(stack, out / f"{name}.csv.gz"), key=_key) for name in names
        }
        pending = {name: next(block, None) for name, block in blocks.items()}
        seen: set[tuple[str, ...]] = set()
        for row in annual:
            key = _key(row)
            audit.key = key
            if key in seen:
                raise LedgerError(f"Repeated annual key {key}")
            seen.add(key)
            audit.rows["annual"] += 1
            parts = []
            for name in names:
                id_field = "id" if name == "operators" else "plot"
                item = pending[name]
                if item is not None and item[0] == key:
                    parts.append(_by_id(item[1], id_field, name, audit))
                    pending[name] = next(blocks[name], None)
                elif name in {"transactions", "fiscal"}:
                    parts.append({})
                else:
                    observed = None if item is None else item[0]
                    raise LedgerError(f"{name}: expected {key}, found {observed}")
            yield Year(row, *parts, *(() if has_fiscal else (None,)))
        if not seen or any(pending.values()):
            remaining = {name: item[0] for name, item in pending.items() if item is not None}
            raise LedgerError(f"Missing annual rows or extra/out-of-order ledger blocks: {remaining}")


def _actor_accounts(y: Year, audit: Audit) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    fields = ("opening", "closing", "profit", "income", "sales", "inputs", "support",
              "rent_paid", "rent_income", "job_income", "farm_days", "job_days",
              "fees", "setup", "consumption", "restart_capital", "recovery_cost", "recovery_days",
              "quoted_plots")
    for identity, row in y.actors.items():
        a = {name: _num(row, name) for name in fields}
        kind = row["kind"]
        audit.check(kind in {"owner", "tenant"}, "actor_kind", identity)
        audit.check(all(a[m] >= -TOL for m in ("opening", "closing", "farm_days", "job_days")),
                    "actor_cash_or_days_negative", identity)
        liquidity = _num(row, "liquidity")
        advance, post = _num(row, "pre_support"), _num(row, "post_support")
        audit.check(a["farm_days"] + a["recovery_days"] + a["job_days"] <= _num(row, "labor_capacity") + TOL,
                    "labor_capacity", identity)
        audit.equal(advance + post, a["support"], "actor_support_timing", identity)
        audit.equal(post if y.annual["policy"].endswith("_pre") else advance, 0.,
                    "wrong_support_timing", identity)
        after_rent = a["opening"] + a["restart_capital"] + a["rent_income"] - a["rent_paid"] - a["fees"] - a["setup"]
        audit.check(-TOL <= liquidity <= after_rent + TOL, "preplant_liquidity", identity)
        audit.check(a["inputs"] + a["recovery_cost"] - advance <= liquidity + TOL,
                    "production_liquidity", identity)
        farm_margin = a["sales"] - a["inputs"] - a["fees"] - a["setup"] - a["recovery_cost"]
        audit.equal(farm_margin - a["rent_paid"], a["profit"], "actor_profit", identity)
        income_parts = farm_margin + a["rent_income"] - a["rent_paid"] + a["job_income"] + a["support"]
        residual = a["income"] - income_parts
        audit.equal(residual, 0., "actor_income", identity)
        audit.equal(a["opening"] + a["restart_capital"] + a["income"] - a["consumption"],
                    a["closing"], "actor_cash_identity", identity)
        for name, value in a.items():
            totals[f"{kind}_{name}"] += value
        totals[f"{kind}_farm_margin_before_rent"] += farm_margin
        totals[f"{kind}_income_residual"] += residual
        totals[f"{kind}_n"] += 1
        totals[f"{kind}_liquidity"] += liquidity
        totals[f"{kind}_weighted_bid_mu"] += a["quoted_plots"] * _num(row, "highest_bid_per_mu")
        totals["exits"] += _flag(row, "exit")
        for name in ("expanded", "shrunk"):
            totals[name] += kind == "tenant" and _flag(row, name)

    for kind in ("owner", "tenant"):
        totals[f"{kind}_quoted_bid_mu"] = _ratio(totals.pop(f"{kind}_weighted_bid_mu", 0.),
                                                totals[f"{kind}_quoted_plots"])
    return dict.fromkeys((f"{k}_{m}" for k in ("owner", "tenant") for m in (
        *fields, "farm_margin_before_rent", "income_residual", "n", "liquidity"
    )), 0.) | dict(totals)


def _plot_accounts(y: Year, audit: Audit) -> dict[str, float]:
    names = (
        "land_area", "grain_area", "grain_kg", "idle_area", "cash_crop_area",
        "retained_grain_area", "retained_grain", "same_operator_area",
        "same_operator_switch_to_cash_area", "operator_change", "continuity_area",
        "direct_new_operator_grain_area", "reactivated_area", "reactivated_grain_area",
        "same_operator_released_rematch_area", "successor_grain", "offered_area",
        "posted_area", "matched_area", "unfilled_area", "leased_area", "leased_idle",
        "leased_cash", "leased_grain", "rent_total", "new_contract_rent_total", "spending",
        "expected_support", "commitment", "cash_unplanted", "labor_unplanted",
        "renewed_area", "held_area", "default_area", "recovered_area",
        "new_lease_grain_area", "new_lease_direct_grain_area", "new_lease_reactivated_grain_area",
        "owner_operated_area", "tenant_operated_area", "owner_planted_area", "tenant_planted_area",
    )
    totals = dict.fromkeys(names, 0.)
    area_by_actor: defaultdict[str, float] = defaultdict(float)
    cross: defaultdict[tuple[str, str], float] = defaultdict(float)
    harvest_factor = None
    contract_year = int(y.annual["year"])
    budget_per_plot = _num(y.annual, "budget") / len(y.plots)
    rate = _num(y.annual, "announced_rate")
    for pid, p in y.plots.items():
        area, planted = _num(p, "physical_area"), _num(p, "planted_area")
        harvest, rent, payment = (_num(p, k) for k in ("harvest", "rent", "support"))
        expected, commitment = (_num(p, k) for k in ("expected_support", "commitment"))
        owner, operator, previous = (p[k] for k in ("owner", "operator", "previous_operator"))
        crop, old_crop = p["crop"], p["previous_crop"]
        grain, prior_grain = crop == "grain", old_crop == "grain"
        idle = crop == "idle"
        released, posted, matched, leased = (_flag(p, k) for k in ("released", "posted", "matched", "lease"))
        changed = bool(previous and operator and previous != operator)
        same = bool(operator and previous == operator)
        reactivated = bool(not previous and operator)
        strict = changed and grain and prior_grain
        held, defaulted, recovered = (_flag(p, k) for k in ("continued_lease", "defaulted", "recovered"))
        eligible = bool(p["assigned_operator"]) and (
            y.annual["policy"].startswith("grain_")
            or (y.annual["policy"].startswith("takeover_") and matched and previous != operator)
        )
        audit.check(owner in y.actors and y.actors[owner]["kind"] == "owner", "plot_rights_holder", pid)
        audit.check(not operator or operator in y.actors, "plot_operator", pid)
        audit.check(not previous or previous in y.actors, "plot_previous_operator", pid)
        audit.check(crop in {"grain", "cash", "idle"} and area > 0. and harvest >= 0.,
                    "plot_area_crop_harvest", pid)
        audit.equal(planted, 0. if idle else area, "actual_planted_area", pid)
        audit.check(not idle or abs(harvest) <= TOL, "unplanted_harvest", pid)
        audit.check(idle or bool(operator), "planted_operator", pid)
        audit.check(not posted or released, "posted_without_release", pid)
        audit.check(not matched or (posted and leased), "match_without_post_or_lease", pid)
        audit.check(leased == bool(operator and operator != owner), "lease_operator_identity", pid)
        audit.check(leased == (matched or held) and not (matched and held), "lease_event", pid)
        audit.check(rent >= -TOL and (leased or abs(rent) <= TOL), "rent_without_lease", pid)
        audit.equal(rent, _num(p, "rent_per_mu") * area, "plot_rent_unit", pid)
        if leased:
            audit.check(1 <= int(p["contract_start"]) <= contract_year <= int(p["contract_end"]),
                        "active_contract_window", pid)
            audit.check((int(p["contract_start"]) == contract_year) == matched, "contract_start_event", pid)
        audit.check(_flag(p, "continuation") == strict, "strict_continuity", pid)
        audit.check(_flag(p, "eligible") == eligible, "plot_policy_eligibility", pid)
        audit.check(min(payment, expected, commitment) >= -TOL, "nonnegative_support", pid)
        reserve = min(rate * area, budget_per_plot, area * CROPS["grain"][2]) if eligible else 0.
        audit.equal(expected, reserve, "announced_expected_support", pid)
        audit.equal(expected, commitment, "expected_commitment", pid)
        audit.equal(payment, commitment if grain and eligible else 0., "verified_plot_payment", pid)
        audit.check(eligible or abs(commitment) <= TOL, "ineligible_commitment", pid)
        audit.check(p["next_operator"] == operator and p["next_crop"] == crop, "plot_next_state", pid)
        if y.fiscal is not None:
            fiscal = y.fiscal.get(pid)
            audit.check((fiscal is not None) == (eligible or bool(commitment) or bool(payment)),
                        "fiscal_plot_coverage", pid)
            if fiscal is not None:
                audit.check(all(fiscal[k] == p[k] for k in ("operator", "crop", "eligible")),
                            "fiscal_plot_identity", pid)
                for name in ("commitment", "expected_support", "support"):
                    audit.equal(_num(fiscal, name), _num(p, name), f"fiscal_plot_{name}", pid)

        trade = y.trades.get(pid)
        audit.check((trade is not None) == (released or leased or defaulted), "plot_transaction_coverage", pid)
        if trade is not None:
            audit.equal(_num(trade, "area"), area, "transaction_area", pid)
            audit.check(trade["owner"] == owner, "transaction_owner", pid)
            audit.check(bool(trade["buyer"]) == leased and _flag(trade, "matched") == matched
                        and _flag(trade, "continued") == held and _flag(trade, "posted") == posted
                        and _flag(trade, "defaulted") == defaulted, "transaction_status", pid)
            audit.equal(_num(trade, "rent"), rent, "transaction_rent", pid)
            audit.equal(_num(trade, "rent_per_mu") * area, rent, "transaction_rent_unit", pid)
            if leased:
                audit.check(trade["buyer"] == operator, "transaction_buyer", pid)
                audit.check(trade["start"] == p["contract_start"] and trade["end"] == p["contract_end"],
                            "transaction_contract", pid)
                cross[operator, "fees"] += _num(trade, "fee")
        cross[owner, "rent_income"] += rent
        if operator:
            cross[operator, "rent_paid"] += rent
            cross[operator, "support"] += payment
            if not idle:
                yield_mu, price, inputs_mu, days_mu = CROPS[crop]
                cross[operator, "inputs"] += planted * inputs_mu
                cross[operator, "sales"] += harvest * price
                cross[operator, "farm_days"] += planted * days_mu
                factor = harvest / (planted * yield_mu * _num(p, "quality"))
                audit.equal(factor, factor if harvest_factor is None else harvest_factor,
                            "common_harvest_factor", pid)
                harvest_factor = factor
            area_by_actor[operator] += planted
            actor = y.actors.get(operator)
            if actor:
                kind = actor["kind"]
                totals[f"{kind}_operated_area"] += area
                totals[f"{kind}_planted_area"] += planted
        for name, value in (
            ("land_area", area), ("grain_area", planted * grain), ("grain_kg", harvest * grain),
            ("idle_area", area * idle), ("cash_crop_area", planted * (crop == "cash")),
            ("retained_grain_area", area * (prior_grain and grain)),
            ("retained_grain", area * (same and prior_grain and grain)),
            ("same_operator_area", planted * same),
            ("same_operator_switch_to_cash_area", area * (same and prior_grain and crop == "cash")),
            ("operator_change", area * changed), ("continuity_area", area * strict),
            ("direct_new_operator_grain_area", area * (changed and grain)),
            ("reactivated_area", planted * reactivated),
            ("reactivated_grain_area", area * (reactivated and grain)),
            ("same_operator_released_rematch_area", planted * (same and matched)),
            ("successor_grain", area * (grain and changed)),
            ("new_lease_grain_area", area * (grain and matched and not same)),
            ("new_lease_direct_grain_area", area * (grain and matched and changed)),
            ("new_lease_reactivated_grain_area", area * (grain and matched and reactivated)),
            ("offered_area", area * released), ("posted_area", area * posted),
            ("matched_area", area * matched), ("unfilled_area", area * (released and not matched)),
            ("leased_area", area * leased), ("leased_idle", area * (leased and idle)),
            ("leased_cash", area * (leased and crop == "cash")), ("leased_grain", area * (leased and grain)),
            ("rent_total", rent), ("new_contract_rent_total", rent * matched), ("spending", payment),
            ("expected_support", expected), ("commitment", commitment),
            ("cash_unplanted", area * (p["failure"] in {"cash", "cash_and_labor"})),
            ("labor_unplanted", area * (p["failure"] in {"labor", "cash_and_labor"})),
            ("renewed_area", area * (matched and same)), ("held_area", area * held),
            ("default_area", area * defaulted), ("recovered_area", area * recovered),
        ):
            totals[name] += value
    audit.check(set(y.trades).issubset(y.plots), "orphan_transaction")
    if y.fiscal is not None:
        audit.check(set(y.fiscal).issubset(y.plots), "orphan_fiscal_entry")
    for identity, actor in y.actors.items():
        for item in ("support", "rent_paid", "rent_income", "fees", "inputs", "sales", "farm_days"):
            audit.equal(cross[identity, item], _num(actor, item), f"plot_actor_{item}", identity)
        audit.equal(area_by_actor[identity], _num(actor, "operation_area"), "plot_actor_operation_area", identity)
    audit.equal(totals["grain_area"] + totals["cash_crop_area"] + totals["idle_area"],
                totals["land_area"], "land_conservation")
    totals["rent_mu"] = _ratio(totals["rent_total"], totals["leased_area"])
    totals["new_contract_rent_mu"] = _ratio(totals["new_contract_rent_total"], totals["matched_area"])
    totals["match_rate"] = _ratio(totals["matched_area"], totals["offered_area"])
    totals["planted_area_hhi"] = _ratio(sum(a * a for a in area_by_actor.values()), sum(area_by_actor.values())**2)
    return totals


def _reconcile(y: Year, audit: Audit) -> None:
    reported = y.annual
    plots, actors = _plot_accounts(y, audit), _actor_accounts(y, audit)
    audit.equal(plots["rent_total"], actors["owner_rent_income"] + actors["tenant_rent_income"],
                "aggregate_rent_in")
    audit.equal(plots["rent_total"], actors["owner_rent_paid"] + actors["tenant_rent_paid"],
                "aggregate_rent_out")
    audit.equal(plots["spending"], actors["owner_support"] + actors["tenant_support"], "aggregate_support")
    budget = _num(reported, "budget")
    audit.check(plots["spending"] <= budget + TOL, "fiscal_budget")
    if reported["policy"] == "none":
        audit.equal(plots["spending"], 0., "no_support_baseline")
    calculated = plots | actors | {"unused_budget": budget - plots["spending"], "budget": budget}
    for name in ("grain_area", "grain_kg", "cash_crop_area", "idle_area", "continuity_area", "successor_grain",
                 "offered_area", "unfilled_area", "leased_area", "leased_idle", "leased_cash",
                 "operator_change", "spending", "tenant_profit", "owner_income", "expanded", "shrunk", "exits",
                 "retained_grain", "reactivated_area", "renewed_area", "held_area", "default_area", "recovered_area",
                 "leased_grain", "posted_area", "matched_area", "rent_total", "unused_budget",
                 "cash_unplanted", "labor_unplanted", "owner_rent_income", "owner_profit",
                 "owner_support", "tenant_support", "owner_rent_paid", "tenant_rent_paid", "tenant_income"):
        audit.equal(calculated[name], _num(reported, name), f"annual_{name}")
    aliases = {"announced_rate": "announced_rate", "committed": "commitment", "owner_nonfarm": "owner_job_income",
               "tenant_area": "tenant_planted_area", "owner_area": "owner_planted_area",
               "owner_opening_cash": "owner_opening", "tenant_opening_cash": "tenant_opening",
               "owner_closing_cash": "owner_closing", "tenant_closing_cash": "tenant_closing",
               "rent_mu": "rent_mu", "concentration": "planted_area_hhi",
               "total_area": "land_area", "land_area": "land_area", "retained_grain_area": "retained_grain_area"}
    calculated["announced_rate"] = _num(reported, "announced_rate")
    for name, source in aliases.items():
        if name in reported:
            value = calculated[source]
            audit.equal(0. if math.isnan(value) else value, _num(reported, name), f"annual_{name}")
    calculated["concentration"] = calculated["planted_area_hhi"]
    y.metrics = calculated | {"reward": _num(reported, "reward")}


def _rent_parts(value: float, area: float, base_value: float, base_area: float) -> tuple[float, float]:
    """Symmetric price/area decomposition; with one empty market use the observed price."""
    price, base_price = _ratio(value, area), _ratio(base_value, base_area)
    if not area or not base_area:
        return 0., value - base_value
    return (price - base_price) * (area + base_area) / 2, (area - base_area) * (price + base_price) / 2


def _grain(p: Row) -> float:
    return _num(p, "harvest") if p["crop"] == "grain" else 0.


def _pair_accounts(y: Year, b: Year, audit: Audit) -> dict[str, float]:
    m, r = y.metrics, b.metrics
    audit.equal(m["budget"], r["budget"], "paired_budget")
    result = {}
    for prefix, value, area in (("active", "rent_total", "leased_area"),
                                ("new", "new_contract_rent_total", "matched_area")):
        price, quantity = _rent_parts(m[value], m[area], r[value], r[area])
        audit.equal(price + quantity, m[value] - r[value], f"{prefix}_rent_decomposition")
        result[f"{prefix}_rent_price_component"] = price
        result[f"{prefix}_rent_area_component"] = quantity
        result[f"{prefix}_rent_both_markets_observed"] = float(m[area] > 0. and r[area] > 0.)
    result.update(
        support_no_positive_grain_kg=0., supported_area_no_positive_grain_kg=0.,
        support_no_positive_grain_area=0., support_with_positive_grain_kg=0.,
        supported_plot_grain_kg_delta=0., shared_active_rent_area=0., shared_new_rent_area=0.,
        shared_active_rent_delta=0., shared_new_rent_delta=0.,
        shared_new_bid_area=0., shared_new_bid_delta=0.,
    )
    for pid, p in y.plots.items():
        q = b.plots[pid]
        area, payment = _num(p, "physical_area"), _num(p, "support")
        if payment > TOL:
            delta = _grain(p) - _grain(q)
            result["supported_plot_grain_kg_delta"] += delta
            if delta <= TOL:
                result["support_no_positive_grain_kg"] += payment
                result["supported_area_no_positive_grain_kg"] += area
            else:
                result["support_with_positive_grain_kg"] += payment
            if q["crop"] == "grain" and _num(q, "planted_area") >= _num(p, "planted_area") - TOL:
                result["support_no_positive_grain_area"] += payment
        for prefix, flag in (("active", "lease"), ("new", "matched")):
            if _flag(p, flag) and _flag(q, flag):
                result[f"shared_{prefix}_rent_area"] += area
                result[f"shared_{prefix}_rent_delta"] += (_num(p, "rent_per_mu") - _num(q, "rent_per_mu")) * area
        if _flag(p, "matched") and _flag(q, "matched"):
            deal, base_deal = y.trades[pid], b.trades[pid]
            if deal["bid"] and base_deal["bid"]:
                result["shared_new_bid_area"] += area
                result["shared_new_bid_delta"] += _num(deal, "bid") - _num(base_deal, "bid")
    for prefix in ("active", "new"):
        result[f"shared_{prefix}_rent_mu_delta"] = _ratio(result[f"shared_{prefix}_rent_delta"],
                                                       result[f"shared_{prefix}_rent_area"])
    result["shared_new_bid_mu_delta"] = _ratio(result["shared_new_bid_delta"], result["shared_new_bid_area"])
    payment_delta = m["spending"] - r["spending"]
    grain_delta = m["grain_kg"] - r["grain_kg"]
    continuity_delta = m["continuity_area"] - r["continuity_area"]
    result.update(spending_delta=payment_delta, grain_kg_delta=grain_delta,
                  continuity_area_delta=continuity_delta,
                  grain_kg_per_fiscal_yuan=_ratio(grain_delta, payment_delta),
                  continuity_mu_per_fiscal_yuan=_ratio(continuity_delta, payment_delta))
    for kind in ("owner", "tenant"):
        for item in ("opening", "closing", "farm_margin_before_rent", "profit", "rent_income",
                     "rent_paid", "job_income", "support", "income", "restart_capital", "consumption"):
            name = f"{kind}_{item}"
            result[f"{name}_delta"] = m[name] - r[name]
        net_rent = result[f"{kind}_rent_income_delta"] - result[f"{kind}_rent_paid_delta"]
        result[f"{kind}_net_rent_delta"] = net_rent
        result[f"{kind}_income_residual_delta"] = (
            result[f"{kind}_income_delta"] - result[f"{kind}_farm_margin_before_rent_delta"]
            - net_rent - result[f"{kind}_job_income_delta"] - result[f"{kind}_support_delta"]
        )
    result["unreconciled_fiscal_receipt_delta"] = (
        payment_delta - result["owner_support_delta"] - result["tenant_support_delta"]
    )
    result["monetary_income_residual_delta"] = sum(
        result[f"{kind}_income_residual_delta"] for kind in ("owner", "tenant")
    )
    return result


def _check_paths(paths: dict[str, dict[int, Year]], audit: Audit) -> None:
    sample = next(iter(paths.values()))
    template = next(iter(sample.values()))
    group = template.annual["group"]
    required = set()
    if group in {"factorial", "fixed_contract", "diagnostic", "evaluation"}:
        required.update(POLICIES)
    if group == "evaluation":
        required.add("rule")
    if group == "duration":
        required.update(("none", "takeover_pre_y1", "takeover_pre_y3", "takeover_pre_y8"))
    if group == "plot_order":
        required.update(f"takeover_pre_{order}" for order in ("original", "reverse", "shuffle1", "shuffle2", "shuffle3"))
    if required - paths.keys():
        raise LedgerError(f"Missing strategies at {audit.key}: {sorted(required - paths.keys())}")
    for strategy, years in paths.items():
        if set(years) != set(YEARS):
            raise LedgerError(f"Incomplete eight-year path {audit.key}: {strategy}, {sorted(years)}")
        for year, current in years.items():
            audit.key = _key(current.annual)
            if current.plots.keys() != template.plots.keys() or current.actors.keys() != template.actors.keys():
                raise LedgerError(f"Unstable plot or actor IDs at {audit.key}")
            for pid, p in current.plots.items():
                start = template.plots[pid]
                audit.check(p["owner"] == start["owner"], "stable_plot_owner", pid)
                audit.equal(_num(p, "physical_area"), _num(start, "physical_area"), "stable_plot_area", pid)
                if year > 1 and (group != "fixed_contract" or strategy == "none"):
                    previous = years[year - 1].plots[pid]
                    audit.check(p["previous_crop"] == previous["crop"], "plot_crop_history", pid)
                    audit.check(p["previous_operator"] == previous["operator"], "plot_operator_history", pid)
                    if _flag(p, "continued_lease"):
                        audit.check(_flag(previous, "lease") and all(
                            p[k] == previous[k] for k in ("operator", "rent", "contract_start", "contract_end")
                        ), "held_contract_history", pid)


def _fixed_pairs(paths: dict[str, dict[int, Year]]) -> Iterator[tuple[str, str]]:
    strategies = {path[1].annual["policy"]: name for name, path in paths.items()}
    for pre, post in (("grain_pre", "grain_post"), ("takeover_pre", "takeover_post")):
        if pre in strategies and post in strategies:
            yield strategies[pre], strategies[post]


def _check_fixed(y: Year, b: Year, audit: Audit, *, same_commitment: bool) -> None:
    fields = ("assigned_operator", "owner", "lease", "rent", "contract_start", "contract_end",
              "released", "posted", "matched", "previous_operator", "previous_crop")
    if same_commitment:
        fields += ("eligible", "commitment", "expected_support")
    for pid, p in y.plots.items():
        q = b.plots[pid]
        for name in fields:
            if name in {"rent", "commitment", "expected_support"}:
                audit.equal(_num(p, name), _num(q, name), f"fixed_{name}", pid)
            else:
                audit.check(p[name] == q[name], f"fixed_{name}", pid)
    for identity, operator in y.actors.items():
        for name in ("opening", "liquidity", "labor_capacity", "restart_capital"):
            audit.equal(_num(operator, name), _num(b.actors[identity], name), f"fixed_{name}", identity)
    audit.check(y.trades == b.trades or {
        pid: {name: val for name, val in trade.items() if name not in KEY}
        for pid, trade in y.trades.items()
    } == {
        pid: {name: val for name, val in trade.items() if name not in KEY}
        for pid, trade in b.trades.items()
    }, "fixed_transactions")


def _cohort_rows(paths: dict[str, dict[int, Year]], audit: Audit,
                 members: list[Output]) -> Iterator[Output]:
    """Select recipients once in the stopped arm and reuse their plot IDs in all arms."""
    sequences = {tuple(path[i].annual["policy"] for i in YEARS): name for name, path in paths.items()}
    never, always = sequences.get(("none",) * len(YEARS)), sequences.get(("takeover_pre",) * len(YEARS))
    for stop in (1, 3):
        stopped = sequences.get(("takeover_pre",) * stop + ("none",) * (len(YEARS) - stop))
        if stopped is None:
            continue
        if never is None or always is None:
            raise LedgerError(f"Cessation path needs P0 and continuous P3 at {audit.key}")
        path = paths[stopped]
        receipts = {}
        for year in range(1, stop + 1):
            for pid, row in path[year].plots.items():
                payment = _num(row, "support")
                if payment > TOL:
                    receipts[pid] = (row["operator"], year, payment)
        recipients = {pid: receipt[0] for pid, receipt in receipts.items()}
        stop_operators = {pid: path[stop].plots[pid]["operator"] for pid in recipients}
        group, condition, seed = _key(path[1].annual, CELL)
        members.extend(dict(group=group, condition=condition, seed=seed, stop_after=stop,
                            plot=pid, recipient=recipient, cutoff_operator=stop_operators[pid],
                            last_receipt_year=receipt_year, last_payment=last_payment)
                       for pid, (recipient, receipt_year, last_payment) in sorted(receipts.items()))
        for strategy, arm in ((stopped, "stopped"), (never, "never"), (always, "continuous")):
            for year, y in sorted(paths[strategy].items()):
                if arm == "stopped" and year > stop:
                    audit.key = _key(y.annual)
                    audit.equal(y.metrics["spending"], 0., "cessation_support_after_stop")
                if arm == "continuous" and year <= stop:
                    audit.key = _key(y.annual)
                    for pid, p in y.plots.items():
                        audit.check(p["operator"] == path[year].plots[pid]["operator"]
                                    and p["crop"] == path[year].plots[pid]["crop"],
                                    "cessation_pre_stop_path", pid)
                        audit.equal(_num(p, "support"), _num(path[year].plots[pid], "support"),
                                    "cessation_pre_stop_support", pid)
                land = [y.plots[pid] for pid in recipients]
                current_ids = {p["operator"] for p in land} - {""}
                recipient_ids = set(recipients.values())
                yield dict(
                    group=group, condition=condition, seed=seed, stop_after=stop,
                    year=year, event_year=year-stop, strategy=strategy, arm=arm,
                    cohort_plots=len(land), cohort_mu=sum(_num(p, "physical_area") for p in land),
                    original_recipient_mu=sum(_num(p, "physical_area") for p in land
                                              if p["operator"] == recipients[p["plot"]]),
                    cutoff_operator_mu=sum(_num(p, "physical_area") for p in land
                                           if p["operator"] and p["operator"] == stop_operators[p["plot"]]),
                    grain_area=sum(_num(p, "planted_area") for p in land if p["crop"] == "grain"),
                    grain_kg=sum(_grain(p) for p in land),
                    cash_crop_area=sum(_num(p, "planted_area") for p in land if p["crop"] == "cash"),
                    idle_area=sum(_num(p, "physical_area") for p in land if p["crop"] == "idle"),
                    support=sum(_num(p, "support") for p in land),
                    current_operators=len(current_ids),
                    current_operators_profit=sum(_num(y.actors[k], "profit") for k in current_ids),
                    current_operators_cash=sum(_num(y.actors[k], "closing") for k in current_ids),
                    original_recipients=len(recipient_ids),
                    original_recipients_profit=sum(_num(y.actors[k], "profit") for k in recipient_ids),
                    original_recipients_cash=sum(_num(y.actors[k], "closing") for k in recipient_ids),
                )


@cache
def _paired_critical_value(n: int) -> float:
    return float(t.ppf(.975, n - 1)) if n > 1 else math.nan


def _statistics(values: np.ndarray, *, paired: bool = False) -> dict[str, Number]:
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else math.nan
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    p05, p50, p95 = np.quantile(values, (.05, .5, .95)) if n else (math.nan,) * 3
    result = dict(n=n, mean=mean, std=sd, p05=float(p05), p50=float(p50), p95=float(p95))
    if paired:
        half = _paired_critical_value(n) * sd / math.sqrt(max(n, 1))
        result.update(ci95_low=mean-half, ci95_high=mean+half,
                      positive=int((values > TOL).sum()), negative=int((values < -TOL).sum()))
    return result


def _yearly_and_path(block: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    n = np.isfinite(block).sum(axis=-2)
    horizon = np.divide(np.nansum(block, axis=-2), n,
                        out=np.full(n.shape, np.nan), where=n > 0)
    if "grain_kg_per_fiscal_yuan" in names:
        fiscal = horizon[..., names.index("spending_delta")]
        for ratio, outcome in (("grain_kg_per_fiscal_yuan", "grain_kg_delta"),
                               ("continuity_mu_per_fiscal_yuan", "continuity_area_delta")):
            horizon[..., names.index(ratio)] = np.divide(
                horizon[..., names.index(outcome)], fiscal,
                out=np.full(fiscal.shape, np.nan), where=fiscal > TOL,
            )
    return np.concatenate((horizon[..., None, :], block), axis=-2)


def _summaries(block: np.ndarray, names: tuple[str, ...], labels: Output, *,
               paired: bool) -> Iterator[Output]:
    for year, panel in enumerate(block.transpose(1, 0, 2)):
        count = np.isfinite(panel).sum(axis=0)
        avg = np.full(len(names), np.nan)
        sd = avg.copy()
        quantiles = np.full((3, len(names)), np.nan)
        seen = count > 0
        repeated = count > 1
        complete = count == len(panel)
        partial = seen & ~complete
        avg[seen] = np.nanmean(panel[:, seen], axis=0)
        sd[repeated] = np.nanstd(panel[:, repeated], axis=0, ddof=1)
        quantiles[:, complete] = np.quantile(panel[:, complete], (.05, .5, .95), axis=0)
        if partial.any():
            quantiles[:, partial] = np.nanquantile(panel[:, partial], (.05, .5, .95), axis=0)
        if paired:
            critical = np.fromiter((_paired_critical_value(int(n)) for n in count), dtype=float)
            half = sd * critical / np.sqrt(np.maximum(count, 1))
            positive = (panel > TOL).sum(axis=0)
            negative = (panel < -TOL).sum(axis=0)
        for j, name in enumerate(names):
            p05, p50, p95 = quantiles[:, j]
            row = labels | dict(year=year, metric=name, n=count[j], mean=avg[j],
                                std=sd[j], p05=p05, p50=p50, p95=p95)
            if paired:
                row.update(ci95_low=avg[j]-half[j], ci95_high=avg[j]+half[j],
                           positive=positive[j], negative=negative[j])
            yield row


def _effects(group: str, strategies: Iterable[str]) -> Iterator[tuple[str, str, str]]:
    names = set(strategies)
    if "fixed" in group:
        return
    if "none" in names:
        for name in sorted(names - {"none"}):
            yield name, "none", "policy_vs_p0"
    if group == "duration":
        for policy in POLICIES[1:]:
            for stop in (1, 3):
                left, right = f"{policy}_y{stop}", f"{policy}_y8"
                if {left, right} <= names:
                    yield left, right, "cessation_vs_continuous"
    if group == "plot_order":
        reference = "takeover_pre_original"
        for name in sorted(names - {reference}):
            yield name, reference, "plot_order_invariance"
    for left, right, label in (
        ("grain_pre", "grain_post", "timing_all_grain"),
        ("takeover_pre", "takeover_post", "timing_new_acquisition"),
        ("takeover_pre", "grain_pre", "eligibility_pre"),
        ("takeover_post", "grain_post", "eligibility_post"),
    ):
        if {left, right} <= names:
            yield left, right, label
    if "evaluation" in group:
        for name in sorted(names - set(POLICIES)):
            for ref in sorted((set(POLICIES) | {"rule"}) & names - {"none", name}):
                yield name, ref, "evaluation_comparator"


def _paired_cohort_summary(rows: list[Output]) -> Iterator[Output]:
    order = ("group", "condition", "stop_after", "year")
    rows.sort(key=lambda row: tuple(row[k] for k in order))
    for key, records in groupby(rows, key=lambda row: tuple(row[k] for k in order)):
        by_arm: defaultdict[str, dict[str, Output]] = defaultdict(dict)
        for row in records:
            by_arm[str(row["arm"])][str(row["seed"])] = row
        ids = sorted(by_arm["stopped"])
        names = tuple(k for k in by_arm["stopped"][ids[0]] if k not in (
            *order, "seed", "event_year", "arm", "strategy",
        ))
        for arm, reference in (("stopped", "never"), ("stopped", "continuous"), ("continuous", "never")):
            if set(by_arm[arm]) != set(by_arm[reference]):
                raise LedgerError(f"Unpaired cessation cohort: {key}, {arm}, {reference}")
            for name in names:
                x = np.array([float(by_arm[arm][i][name]) - float(by_arm[reference][i][name]) for i in ids])
                yield dict(zip(order, key, strict=True)) | dict(
                    event_year=int(key[-1])-int(key[-2]), arm=arm, reference=reference,
                    metric=name,
                ) | _statistics(x, paired=True)


def _matrix(names: tuple[str, ...], rows: Iterable[dict[str, float]]) -> np.ndarray:
    return np.array([[row[m] for m in names] for row in rows])


def _across_conditions(worlds: dict[tuple[str, ...], dict[str, np.ndarray]],
                       comparisons: dict[tuple[str, ...], dict[str, np.ndarray]]) -> None:
    """Compare levels and P0-referenced effects with the same seed across conditions."""
    for (group, condition, strategy), values in worlds.items():
        if group in {"diagnostic", "duration"} and condition != "middle":
            references = (("middle", "condition_vs_middle"),)
        elif group == "factorial":
            references = tuple((condition.replace(high, low), label) for high, low, label in (
                ("high_search", "low_search", "search_moderation"),
                ("tight_cash", "ample_cash", "cash_moderation"),
                ("tight_labor", "ample_labor", "labor_moderation"),
            ) if high in condition)
        else:
            references = ()
        for reference, label in references:
            base = worlds.get((group, reference, strategy))
            if base is None:
                continue
            if set(base) != set(values):
                raise LedgerError(f"Cross-condition seeds differ: {group}/{condition}/{reference}/{strategy}")
            comparisons[group, condition, strategy, strategy, label, reference] = {
                seed: value - base[seed] for seed, value in values.items()
            }
            if strategy != "none":
                control = worlds[group, condition, "none"]
                baseline = worlds[group, reference, "none"]
                comparisons[group, condition, strategy, "none", label + "_p0_did", reference] = {
                    seed: (value - control[seed]) - (base[seed] - baseline[seed])
                    for seed, value in values.items()
                }
    feedback = ("middle", "cash_reset", "exit_reset", "no_feedback")
    for strategy in POLICIES:
        if not all(("diagnostic", condition, strategy) in worlds for condition in feedback):
            continue
        a, b, c, d = (worlds["diagnostic", condition, strategy] for condition in feedback)
        comparisons["diagnostic", "middle", strategy, strategy, "cash_exit_interaction", "no_feedback"] = {
            seed: a[seed] - b[seed] - c[seed] + d[seed] for seed in a
        }
        if strategy != "none":
            a0, b0, c0, d0 = (worlds["diagnostic", condition, "none"] for condition in feedback)
            comparisons["diagnostic", "middle", strategy, "none", "cash_exit_interaction_p0_did", "no_feedback"] = {
                seed: (a[seed] - a0[seed]) - (b[seed] - b0[seed]) - (c[seed] - c0[seed]) + (d[seed] - d0[seed])
                for seed in a
            }


def _write_statistics(out: Path, names: tuple[str, ...],
                      worlds: dict[tuple[str, ...], dict[str, np.ndarray]],
                      comparisons: dict[tuple[str, ...], dict[str, np.ndarray]],
                      accounting_names: tuple[str, ...],
                      accounts: dict[tuple[str, ...], dict[str, np.ndarray]]) -> None:
    blocks = {key: _yearly_and_path(np.stack(list(by_seed.values())), names)
              for key, by_seed in worlds.items()}
    cumulative = ("spending", "budget", "commitment", "unused_budget", "grain_kg", "continuity_area")
    _write(out / "world_means.csv", (
        dict(zip(PATH, key, strict=True)) | {"seed": seed} | dict(zip(names, metrics, strict=True)) | {
            f"cumulative_{name}": metrics[names.index(name)] * len(YEARS) for name in cumulative
        }
        for key, by_seed in worlds.items()
        for seed, metrics in zip(by_seed, blocks[key][:, 0, :], strict=True)
    ))
    def annual_rows() -> Iterator[Output]:
        for key, block in blocks.items():
            total = np.nansum(block, axis=0)
            count = np.isfinite(block).sum(axis=0)
            means = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
            for year in YEARS:
                yield dict(zip(PATH, key, strict=True)) | {"year": year} | dict(zip(names, means[year], strict=True))

    _write(out / "annual_means.csv", annual_rows())
    _write(out / "distribution.csv", (
        (dict(zip(PATH, key, strict=True)) | {
            f"{name}_{stat}": value
            for j, name in enumerate(names)
            for stat, value in _statistics(block[:, 0, j]).items()
        }) for key, block in blocks.items()
    ))
    pair_key = (*PATH, "reference", "contrast", "reference_condition")
    fields = (*pair_key, "year", "metric", "n", "mean", "std", "p05", "p50", "p95",
              "ci95_low", "ci95_high", "positive", "negative")
    with ((out / "paired_effects.csv").open("w", encoding="utf-8", newline="") as paired_csv,
          (out / "mechanism_contrasts.csv").open("w", encoding="utf-8", newline="") as mechanisms_csv):
        paired_writer = csv.DictWriter(paired_csv, fieldnames=fields)
        mechanisms_writer = csv.DictWriter(mechanisms_csv, fieldnames=fields)
        paired_writer.writeheader()
        mechanisms_writer.writeheader()
        for key, by_seed in comparisons.items():
            mechanism = key[-2] not in {"policy_vs_p0", "evaluation_comparator"}
            block = _yearly_and_path(np.stack(list(by_seed.values())), names)
            for row in _summaries(block, names, dict(zip(pair_key, key, strict=True)), paired=True):
                paired_writer.writerow(row)
                if mechanism:
                    mechanisms_writer.writerow(row)
    _write(out / "transmission_accounting_summary.csv", (
        row for key, by_seed in accounts.items()
        for row in _summaries(_yearly_and_path(np.stack(list(by_seed.values())), accounting_names),
                              accounting_names, dict(zip(pair_key, key, strict=True)), paired=True)
    ), fields)
    _write(out / "transmission_accounting.csv", (
        dict(zip(pair_key, key, strict=True)) | {"seed": seed, "year": year} |
        dict(zip(accounting_names, row, strict=True))
        for key, by_seed in accounts.items() for seed, block in by_seed.items()
        for year, row in enumerate(_yearly_and_path(block, accounting_names))
    ), (*pair_key, "seed", "year", *accounting_names))


def analyze(out: Path) -> None:
    """Analyze the four completed ledgers and refuse publication on failed reconciliation."""
    audit = Audit()
    worlds: defaultdict[tuple[str, ...], dict[str, np.ndarray]] = defaultdict(dict)
    comparisons: defaultdict[tuple[str, ...], dict[str, np.ndarray]] = defaultdict(dict)
    accounts: defaultdict[tuple[str, ...], dict[str, np.ndarray]] = defaultdict(dict)
    choices: Counter[tuple[str, ...]] = Counter()
    cohorts: list[Output] = []
    members: list[Output] = []
    seen: set[tuple[str, ...]] = set()
    names: tuple[str, ...] = ()
    accounting_names: tuple[str, ...] = ()
    try:
        for cell, block in groupby(_stream(out, audit), key=lambda y: _key(y.annual, CELL)):
            if cell in seen:
                raise LedgerError(f"Non-contiguous group/condition/seed {cell}")
            seen.add(cell)
            group, condition, seed = cell
            paths: defaultdict[str, dict[int, Year]] = defaultdict(dict)
            for y in block:
                audit.key = _key(y.annual)
                _reconcile(y, audit)
                year, strategy, policy = int(y.annual["year"]), y.annual["strategy"], y.annual["policy"]
                if year in paths[strategy]:
                    raise LedgerError(f"Multiple policies for strategy/year {audit.key}")
                paths[strategy][year] = y
                choices[group, condition, strategy, policy] += 1
            _check_paths(paths, audit)
            if not names:
                names = tuple(next(iter(paths.values()))[1].metrics)
            blocks = {}
            for strategy, path in paths.items():
                blocks[strategy] = _matrix(names, (path[i].metrics for i in YEARS))
                worlds[group, condition, strategy][seed] = blocks[strategy]
            fixed_pairs = tuple(_fixed_pairs(paths)) if "fixed" in group else ()
            if "fixed" in group and len(fixed_pairs) != 2:
                raise LedgerError(f"Missing fixed-contract policy pair at {cell}")
            for left, right, label in (
                *_effects(group, paths),
                *((a, b, "fixed_contract_timing") for a, b in fixed_pairs),
            ):
                key = (group, condition, left, right, label, condition)
                comparisons[key][seed] = blocks[left] - blocks[right]
                if label in {"policy_vs_p0", "fixed_contract_timing"}:
                    records = []
                    for i in YEARS:
                        y, b = paths[left][i], paths[right][i]
                        audit.key = _key(y.annual)
                        if label == "fixed_contract_timing":
                            _check_fixed(y, b, audit, same_commitment=True)
                            if "none" not in paths:
                                raise LedgerError(f"Fixed contracts lack the P0 shadow: {audit.key}")
                            for arm in (y, b):
                                _check_fixed(arm, paths["none"][i], audit, same_commitment=False)
                        records.append(_pair_accounts(y, b, audit))
                    if not accounting_names:
                        accounting_names = tuple(records[0])
                    accounts[key][seed] = _matrix(accounting_names, records)
                if label == "plot_order_invariance":
                    for i in YEARS:
                        y, b = paths[left][i], paths[right][i]
                        audit.key = _key(y.annual)
                        for attr in ("annual", "plots", "actors", "trades"):
                            first, second = getattr(y, attr), getattr(b, attr)
                            if attr == "annual":
                                first, second = {"annual": first}, {"annual": second}
                            audit.check(first.keys() == second.keys() and all(
                                all(left_row[k] == second[pid][k] for k in left_row if k not in KEY)
                                for pid, left_row in first.items()
                            ), f"plot_order_{attr}")
            if "fixed" not in group and set(POLICIES[1:]) <= paths.keys():
                gp, gl, tp, tl = (blocks[s] for s in POLICIES[1:])
                for label, delta in (
                    ("timing_factorial_mean", ((gp - gl) + (tp - tl)) / 2),
                    ("eligibility_factorial_mean", ((tp - gp) + (tl - gl)) / 2),
                    ("eligibility_timing_interaction", (tp - tl) - (gp - gl)),
                ):
                    comparisons[group, condition, "2x2", "", label, condition][seed] = delta
            cohorts.extend(_cohort_rows(paths, audit, members))
        manifest = out / "manifest.json"
        if manifest.is_file():
            run = json.loads(manifest.read_text(encoding="utf-8"))
            expected = {
                (g, condition, str(seed)) for g, conditions in run["scenario"].items()
                for condition in conditions for seed in run["seeds"][g]
            }
            if expected != seen:
                raise LedgerError(
                    f"Ledger cells disagree with manifest: missing={len(expected-seen)}, "
                    f"extra={len(seen-expected)}"
                )
        for name, by_seed in comparisons.items():
            group, condition, left, right, _, _ = name
            if (group, condition, left) in worlds and set(by_seed) != set(worlds[group, condition, left]):
                raise LedgerError(f"Missing strategy pairs: {name}")
            if (group, condition, right) in worlds and set(by_seed) != set(worlds[group, condition, right]):
                raise LedgerError(f"Missing reference pairs: {name}")
        _across_conditions(worlds, comparisons)
    except (KeyError, ValueError, OSError) as exc:
        audit.check(False, "ledger_input_or_structure", str(exc))
        raise
    finally:
        audit.write(out)
    if audit.failures:
        raise LedgerError(f"Ledger checks failed; see {out / 'ledger_verification.json'}")
    _write_statistics(out, names, worlds, comparisons, accounting_names, accounts)
    _write(out / "policy_choices.csv", (
        dict(zip((*PATH, "policy"), key, strict=True)) | {"years": count} for key, count in choices.items()
    ))
    _write(out / "cessation_cohorts.csv", cohorts)
    _write(out / "cessation_cohort_members.csv", members,
           ("group", "condition", "seed", "stop_after", "plot", "recipient", "cutoff_operator",
            "last_receipt_year", "last_payment"))
    _write(out / "cessation_paired_effects.csv", _paired_cohort_summary(cohorts))
