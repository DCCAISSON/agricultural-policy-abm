# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mesa", "pyyaml"]
# ///
# How to run: python -B -m pytest -q tests/test_b19_pairing.py
"""B19-only random-stream binding, after unchanged model initialization."""

from typing import Protocol

from core.environment import Environment
from core.reproducibility import build_numpy_rng
from modules.climate_module import ClimateManager


class PairingModel(Protocol):
    @property
    def environment(self) -> Environment | None: ...

    @property
    def climate_manager(self) -> ClimateManager | None: ...


def bind_exogenous_streams(model: PairingModel, seed: int) -> None:
    assert model.environment is not None
    setattr(model.environment, "np_random", build_numpy_rng(seed, "b19_environment"))
    if model.climate_manager is not None:
        setattr(model.climate_manager, "np_random", build_numpy_rng(seed, "b19_spatial_climate"))
