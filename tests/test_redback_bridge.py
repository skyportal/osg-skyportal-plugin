"""Tests for the redback bridge's trigger-time (T0) handling."""

import sys
import types

import pytest

# redback_bridge imports redback_jax lazily; a bare stub lets the module import
# so _prior_bounds/_model_registry are testable without the heavy redback-jax
# install (neither touches redback_jax -- only the fit/overlay paths do).
sys.modules.setdefault("redback_jax", types.ModuleType("redback_jax"))

import redback_bridge  # noqa: E402

MODELS = ["arnett", "magnetar", "magnetar_nickel", "shock_cooling"]


def test_trigger_time_pins_t0():
    """A supplied trigger_time (MJD) pins the epoch param to a narrow window."""
    b = redback_bridge._prior_bounds("arnett", {}, 60000.0, trigger_time=60789.4213, t0_window=1e-3)
    lo, hi = b["t0"]
    assert lo == pytest.approx(60789.4213 - 1e-3)
    assert hi == pytest.approx(60789.4213 + 1e-3)


def test_prior_ranges_override_wins_over_trigger_time():
    """An explicit prior_ranges entry for t0 takes precedence over trigger_time."""
    b = redback_bridge._prior_bounds(
        "arnett", {"t0": [60000.0, 60100.0]}, 60000.0, trigger_time=60789.4213
    )
    assert b["t0"] == (60000.0, 60100.0)


@pytest.mark.parametrize("model", MODELS)
def test_no_trigger_time_is_data_anchored(model):
    """Without trigger_time, t0 keeps its data-anchored (min_mjd +/- window) prior."""
    b = redback_bridge._prior_bounds(model, {}, 60000.0)
    assert b["t0"] == (60000.0 - 15.0, 60000.0 + 10.0)


@pytest.mark.parametrize("model", MODELS)
def test_registry_entry_is_well_formed(model):
    """Every registered model has ordered priors for its fit params, a
    data-anchored t0, and no fit/fixed param overlap."""
    reg = redback_bridge._model_registry()[model]
    assert reg["priors"][reg["t0_key"]] is None
    assert not set(reg["fit_params"]) & set(reg["fixed"])
    for p in reg["fit_params"]:
        lo, hi = reg["priors"][p]
        assert lo < hi


def test_subthreshold_forced_photometry_is_not_a_detection(tmp_path):
    """redback fits detections only, so a sub-5-sigma fp point (still has a mag)
    must be skipped; the high-SNR point is kept."""
    import pytest

    pytest.importorskip("astropy")
    csv = tmp_path / "phot.csv"
    csv.write_text(
        "mjd,filter,mag,magerr,flux,fluxerr,limiting_mag\n"
        "100.0,ztfg,19.0,0.05,1000,50,20.5\n"  # SNR 20 -> detection
        "101.0,ztfg,21.0,0.5,100,50,21.5\n"  # SNR 2  -> skipped
    )
    times, mags, errs, bands, n_det, _min = redback_bridge._parse_photometry(
        {"photometry": str(csv)}
    )
    assert n_det == 1 and len(times) == 1
