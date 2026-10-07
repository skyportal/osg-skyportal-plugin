"""Pure-logic tests for flare_staging (no DB): the opt-in gate and best-effort
guards. The in-process BOOM fetch needs a live SkyPortal and is exercised there."""

import flare_staging


def test_disabled_by_default(tmp_path):
    # No flare.use_boom_context -> nothing staged, FLARE keeps its live fetch.
    out = flare_staging.stage_context({}, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
    assert out == []
    assert not (tmp_path / flare_staging.CONTEXT_FILE).exists()


def test_enabled_without_skyportal_is_non_fatal(tmp_path):
    # Enabled but standalone (no DB) degrades to live fetch, never raises.
    cfg = {"flare": {"use_boom_context": True}}
    out = flare_staging.stage_context(cfg, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
    assert out == []


def test_photoz_prefers_lsdr10_then_ned():
    assert flare_staging._photoz({"LSDR10": [{"z_phot_median": 0.14, "z_phot_std": 0.02}]}) == {
        "z_phot": 0.14,
        "z_phot_std": 0.02,
    }
    assert flare_staging._photoz({"NED": [{"z": 0.03}]}) == {"z_phot": 0.03}
    assert flare_staging._photoz({}) == {}


def test_nearest_picks_first_present_catalog():
    cm = {"PS1_DR2": [{"gMeanPSFMag": 20.0}], "PS1_DR1": [{"gMeanPSFMag": 99.0}]}
    assert flare_staging._nearest(cm, "PS1_DR2", "PS1_DR1")["gMeanPSFMag"] == 20.0
    assert flare_staging._nearest({}, "AllWISE") is None


def test_context_file_matches_bridge():
    import flare_bridge

    assert flare_staging.CONTEXT_FILE == flare_bridge.CONTEXT_FILE
