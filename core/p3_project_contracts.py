"""Explicit project inputs; missing formal inputs are not zero-valued policies."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, assert_never

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator

if TYPE_CHECKING:
    from core.model import MainModel


class P3InputsMissing(RuntimeError):
    """A required policy, price or accounting observation is absent."""


class P3AccountingError(RuntimeError):
    """The proposed batch violates project or cash-accounting constraints."""


InputValue = TypeVar("InputValue")
NATIONAL_V1_REGIONS = ('东部', '中部', '西部', '东北')
V1_PRIORITY_SCENARIOS = tuple(f'national_v1_priority_{i}' for i in range(4))


def required_value(value: InputValue | None) -> InputValue:
    if value is None:
        raise P3InputsMissing("required project input is absent")
    return value


class PreservedProject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    project_id: str
    basis: Literal["rights_area", "grain_area", "order_delivery"]
    rate: float | None = Field(ge=0)
    price_year: int | None
    to_model: float | None = Field(gt=0)
    source: str | None
    threshold_mu: float = Field(default=0, ge=0)
    qualification_period: Literal["annual", "season"] = "annual"
    eligible_seasons: tuple[str, ...] = ()
    requires_cultivation: bool = True


class P3Program(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    rule: Literal["area_universal", "scale_eligibility", "family_service", "area_service_mix"]
    model_price_year: int | None
    trial_price_year: int | None
    trial_to_model: float | None = Field(gt=0)
    service_2019_to_model: float | None = Field(gt=0)
    price_conversion_source: str | None
    base_budget_yuan: float | None = Field(ge=0)
    trial_budget_yuan: float | None = Field(ge=0)
    implementation_cost_yuan: float | None = Field(ge=0)
    implementation_cost_rate: float | None = Field(default=None, ge=0)
    expected_area_claims_yuan: float | None = Field(ge=0)
    expected_service_claims_yuan: float | None = Field(ge=0)
    previous_actor_pool_nominal: dict[str, dict[Literal["area", "service"], NonNegativeFloat]] | None = None
    expectation_source_step: int = Field(default=0, ge=0)
    preserved_projects: tuple[PreservedProject, ...] | None
    rate_origin: Literal["p3_fixed350", "b19_scenario"] = "p3_fixed350"
    nominal_rate: float = Field(default=350, ge=0)
    scenario_id: str = "P3"
    scenario_rate_source: str | None = None

    @model_validator(mode="after")
    def check_rate_origin(self) -> P3Program:
        match self.rate_origin:
            case "p3_fixed350":
                if self.nominal_rate != 350:
                    raise P3AccountingError("P3 nationwide nominal_rate must equal 350")
            case "b19_scenario":
                if not self.scenario_rate_source or self.scenario_id == "P3":
                    raise P3InputsMissing("B19 needs its scenario id and rate source")
            case unreachable:
                assert_never(unreachable)
        return self

    def require_ready(self) -> None:
        required = (
            "model_price_year", "trial_price_year", "trial_to_model",
            "service_2019_to_model", "price_conversion_source", "base_budget_yuan",
            "trial_budget_yuan", "implementation_cost_yuan", "expected_area_claims_yuan",
            "expected_service_claims_yuan", "preserved_projects",
        )
        missing = [name for name in required if getattr(self, name) is None]
        for project in self.preserved_projects or ():
            missing.extend(f"{project.project_id}.{name}" for name in
                           ("rate", "price_year", "to_model", "source")
                           if getattr(project, name) is None)
        if missing:
            raise P3InputsMissing(", ".join(missing))
        ids = [p.project_id for p in self.preserved_projects or ()]
        if len(ids) != len(set(ids)):
            raise P3AccountingError("duplicate preserved project id")

    def pool_budget(self, pool: Literal["base", "area", "service"]) -> float:
        self.require_ready()
        if pool == "base":
            return required_value(self.base_budget_yuan)
        budget = required_value(self.trial_budget_yuan)
        match self.rule:
            case "area_universal" | "scale_eligibility":
                return budget if pool == "area" else 0.0
            case "family_service":
                return budget if pool == "service" else 0.0
            case "area_service_mix":
                return budget / 2
            case unreachable:
                assert_never(unreachable)

    def expected_payment_ratio(self, actor_id: str, pool: Literal["area", "service"], nominal: float) -> float:
        total = required_value(self.expected_area_claims_yuan if pool == "area" else self.expected_service_claims_yuan)
        if self.previous_actor_pool_nominal is not None:
            own_previous = self.previous_actor_pool_nominal.get(actor_id, {}).get(pool, 0.0)
            total = total - own_previous + nominal
        return ration(self.pool_budget(pool), total)

    def regional_shares(self) -> dict[str, float]:
        if self.scenario_id not in V1_PRIORITY_SCENARIOS:
            return {}
        priority = V1_PRIORITY_SCENARIOS.index(self.scenario_id)
        return {r: .5 if i == priority else 1 / 6 for i, r in enumerate(NATIONAL_V1_REGIONS)}


class ExpectedProjectClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    project_id: str
    expected_model_yuan: float = Field(ge=0)
    source: str = Field(min_length=1)


class ProductionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    plot_id: str | None
    season: str
    area_mu: float = Field(gt=0)
    grain: bool = True
    harvest_kg: float = Field(ge=0)
    qualified_delivery_kg: float | None = Field(default=None, ge=0)
    source: str

    @model_validator(mode="after")
    def check_delivery(self) -> ProductionRecord:
        if self.qualified_delivery_kg is not None and self.qualified_delivery_kg > self.harvest_kg:
            raise P3AccountingError("qualified delivery exceeds harvest")
        return self


def configured_program(model: MainModel) -> P3Program | None:
    fiscal = model.config.get("fiscal_system", {})
    if "p3_support" not in fiscal:
        return None
    raw = fiscal["p3_support"]
    if raw is None:
        raise P3InputsMissing("p3_support program is missing")
    program = P3Program.model_validate(raw)
    program.require_ready()
    return program


def ration(budget: float, claims: float) -> float:
    return min(1.0, budget / claims) if budget > 0 and claims > 0 else 0.0
