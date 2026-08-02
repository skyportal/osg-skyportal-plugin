"""Tests for the redback bridge's trigger-time (T0) handling."""

import sys
import types

import pytest

# redback_bridge._model_registry lazily imports redback_jax.sources; stub it so
# _prior_bounds is testable without the real (heavy) redback-jax install.
_rj = types.ModuleType("redback_jax")
_src = types.ModuleType("redback_jax.sources")
_src.PrecomputedSpectraSource = type(
    "PrecomputedSpectraSource",
    (),
    {"from_arnett_model": staticmethod(lambda **k: None)},
)
sys.modules.setdefault("redback_jax", _rj)
sys.modules["redback_jax.sources"] = _src

import redback_bridge  # noqa: E402


def test_trigger_time_pins_t0():
    """A supplied trigger_time (MJD) pins the epoch param to a narrow window."""
    b = redback_bridge._prior_bounds("arnett", {}, 60000.0, trigger_time=60789.4213, t0_window=1e-3)
    lo, hi = b["t0"]
    assert lo == pytest.approx(60789.4213 - 1e-3)
    assert hi == pytest.approx(60789.4213 + 1e-3)


def test_no_trigger_time_is_data_anchored():
    """Without trigger_time, t0 keeps its data-anchored (min_mjd +/- window) prior."""
    b = redback_bridge._prior_bounds("arnett", {}, 60000.0)
    assert b["t0"] == (60000.0 - 15.0, 60000.0 + 10.0)


def test_prior_ranges_override_wins_over_trigger_time():
    """An explicit prior_ranges entry for t0 takes precedence over trigger_time."""
    b = redback_bridge._prior_bounds(
        "arnett", {"t0": [60000.0, 60100.0]}, 60000.0, trigger_time=60789.4213
    )
    assert b["t0"] == (60000.0, 60100.0)
