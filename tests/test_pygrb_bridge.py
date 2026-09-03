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
