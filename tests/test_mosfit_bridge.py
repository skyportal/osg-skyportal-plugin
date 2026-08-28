"""Pure-logic tests for mosfit_bridge (no mosfit/numpy/astropy needed).

The fit itself and the numpy/astropy parsing paths run only in the mosfit image;
here we cover the plugin-specific pure helpers: ZTF band mapping, upper-limit
selection, and parameter merging.
"""

import mosfit_bridge


def test_band_tags_maps_ztf_to_instrument():
    # ZTF filters resolve to the baked Palomar/ZTF curves via instrument "ZTF".
    assert mosfit_bridge._band_tags("ztfg") == ("g", "ZTF", "AB")
    assert mosfit_bridge._band_tags("ztfr") == ("r", "ZTF", "AB")
    assert mosfit_bridge._band_tags("ztfi") == ("i", "ZTF", "AB")


def test_band_tags_fallback_for_non_ztf():
    # Unknown filters fall back to the last character, no instrument.
    assert mosfit_bridge._band_tags("sdssr") == ("r", None, "AB")
    assert mosfit_bridge._band_tags("") == ("r", None, "AB")


def test_params_merges_defaults_with_overrides():
    p = mosfit_bridge._params({"analysis_parameters": {"source": "slsn", "iterations": 42}})
    assert p["source"] == "slsn"
    assert p["iterations"] == 42
    assert p["method"] == mosfit_bridge.DEFAULTS["method"]  # untouched default


def test_is_nuisance_param_drops_error_and_variance_terms():
    # MOSFiT's per-source nuisance terms clutter the corner plot.
    assert mosfit_bridge._is_nuisance_param("default_no_error_bar_error")
    assert mosfit_bridge._is_nuisance_param("default_upper_limit_error")
    assert mosfit_bridge._is_nuisance_param("variance")
    # physical parameters are kept
    for p in ("fnickel", "mejecta", "kappa", "kappagamma", "temperature", "lumdist"):
        assert not mosfit_bridge._is_nuisance_param(p)


def test_select_upper_limits_empty_without_detections():
    nondets = [(100.0, "ztfr", "r", "ZTF", "AB", 20.5)]
    assert mosfit_bridge._select_upper_limits([], nondets) == []


def test_select_upper_limits_keeps_last_pre_and_window_only():
    # detections span mjd 105-115 in band r
    dets = [
        (105.0, "ztfr", "r", "ZTF", "AB", 19.0, 0.1),
        (115.0, "ztfr", "r", "ZTF", "AB", 19.5, 0.1),
    ]
    nondets = [
        (90.0, "ztfr", "r", "ZTF", "AB", 20.0),  # old pre-det -> dropped
        (104.0, "ztfr", "r", "ZTF", "AB", 20.2),  # last pre-det -> kept
        (110.0, "ztfr", "r", "ZTF", "AB", 20.4),  # in window -> kept
        (120.0, "ztfr", "r", "ZTF", "AB", 20.6),  # post-det -> dropped
        (108.0, "ztfg", "g", "ZTF", "AB", 20.8),  # non-detected band -> dropped
    ]
    kept = mosfit_bridge._select_upper_limits(dets, nondets)
    kept_times = sorted(k[0] for k in kept)
    assert kept_times == [104.0, 110.0]
    assert all(k[2] == "r" for k in kept)  # only the detected band survives
