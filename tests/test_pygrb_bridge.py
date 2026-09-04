"""Tests for the pygrb bridge's input handling. The real search (pycbc/gwpy) is
gated behind dry_run, so these run without those heavy deps."""

import pytest

import pygrb_bridge

# GW170817: the deck's validation trigger.
_BASE = {
    "trigger_time": 1187008884,
    "ra": 197.45,
    "dec": -23.38,
    "detectors": ["H1", "L1", "V1"],
}


def _payload(**extra):
    return {"analysis_parameters": {**_BASE, **extra}}


def test_gps_passthrough():
    assert pygrb_bridge._to_gps(1187008884.0, "gps") == 1187008884.0


def test_dry_run_returns_stub():
    out = pygrb_bridge.run_from_skyportal_inputs(_payload(dry_run=True))
    assert out["_stub"] is True
    assert out["status"] == "success"
    assert out["coherent_snr"] is None
    assert out["trigger_gps"] == 1187008884.0
    assert out["detectors"] == ["H1", "L1", "V1"]


@pytest.mark.parametrize("val", ["True", "true", "1", "yes", True])
def test_dry_run_truthy_variants(val):
    assert pygrb_bridge.run_from_skyportal_inputs(_payload(dry_run=val)).get("_stub") is True


def test_validate_requires_trigger_time():
    p = {"analysis_parameters": {"ra": 1.0, "dec": 2.0}}
    with pytest.raises(ValueError, match="trigger_time"):
        pygrb_bridge.validate_inputs(p)


def test_validate_filters_unknown_detectors():
    info = pygrb_bridge.validate_inputs(_payload(detectors=["H1", "L1", "XX", "foo"]))
    assert info["detectors"] == ["H1", "L1"]


def test_validate_rejects_no_valid_detectors():
    with pytest.raises(ValueError, match="no valid detectors"):
        pygrb_bridge.validate_inputs(_payload(detectors=["XX"]))


def test_t0_samples_center_and_widen_window(monkeypatch):
    """A KN T0 posterior (MJD samples) sets the trigger and widens the on-source
    window to +/- n_sigma * sigma (deck Test 3). MJD->GPS is astropy's job; stub it
    here to test the coupling logic without that dep."""
    monkeypatch.setattr(pygrb_bridge, "_to_gps", lambda v, fmt: float(v))
    samples = [57982.52, 57982.53, 57982.54]  # ~0.0082 d scatter
    info = pygrb_bridge.validate_inputs(_payload(t0_samples=samples, onsource_window=8.0))
    assert info["t0_info"]["t0_source"] == "kn_posterior"
    assert info["t0_info"]["t0_mjd"] == pytest.approx(57982.53)
    # 5 sigma * 0.008165 d * 86400 s/d ~ 3.5e3 s, far above the 8 s floor
    assert info["onsource_window"] == pytest.approx(5 * 0.0081650 * 86400, rel=1e-3)
    assert info["onsource_window"] > 8.0


def test_t0_sigma_widens_window_over_floor():
    info = pygrb_bridge.validate_inputs(_payload(t0_sigma_days=0.01, onsource_window=8.0))
    assert info["t0_info"]["t0_source"] == "kn_sigma"
    assert info["onsource_window"] == pytest.approx(5 * 0.01 * 86400)


def test_fixed_trigger_uses_floor_window():
    info = pygrb_bridge.validate_inputs(_payload(onsource_window=6.0))
    assert info["t0_info"]["t0_source"] == "fixed"
    assert info["onsource_window"] == 6.0


def test_fixed_mode_keeps_exact_window_despite_sigma():
    # onsource_mode=fixed must ignore the T0 spread and use onsource_window exactly.
    info = pygrb_bridge.validate_inputs(
        _payload(onsource_mode="fixed", onsource_window=6.0, t0_sigma_days=0.01)
    )
    assert info["t0_info"]["onsource_mode"] == "fixed"
    assert info["onsource_window"] == 6.0


def test_fixed_mode_centers_on_t0_samples(monkeypatch):
    monkeypatch.setattr(pygrb_bridge, "_to_gps", lambda v, fmt: float(v))
    info = pygrb_bridge.validate_inputs(
        _payload(
            onsource_mode="fixed",
            onsource_window=1000.0,
            t0_samples=[57982.52, 57982.53, 57982.54],
        )
    )
    assert info["onsource_window"] == 1000.0  # not widened by the sample spread
    assert info["t0_info"]["t0_mjd"] == pytest.approx(57982.53)


def test_subsegment_centers_tile_window():
    t0, window, subseg = 1187008884.0, 1500.0, 1000.0
    segs = pygrb_bridge._subsegment_centers(t0, window, subseg)
    assert len(segs) == 3  # ceil(2*1500 / 1000)
    assert all(h == pytest.approx(500.0) for _, h in segs)  # window / n_sub
    # Contiguous and covering exactly [t0-window, t0+window].
    assert segs[0][0] - segs[0][1] == pytest.approx(t0 - window)
    assert segs[-1][0] + segs[-1][1] == pytest.approx(t0 + window)
    for (c0, h0), (c1, h1) in zip(segs, segs[1:]):
        assert c0 + h0 == pytest.approx(c1 - h1)


def test_subsegment_centers_single_when_window_small():
    t0, window, subseg = 1187008884.0, 400.0, 1000.0
    segs = pygrb_bridge._subsegment_centers(t0, window, subseg)
    assert len(segs) == 1
    assert segs[0] == (pytest.approx(t0), pytest.approx(window))
