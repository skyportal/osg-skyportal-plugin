"""Pure-logic tests for fleet_staging (no DB): the best-effort guards that skip
the fetch. The in-process BOOM query needs a live SkyPortal and is exercised
there, not here."""

import fleet_staging


def test_stage_host_skips_without_obj_id(tmp_path):
    assert fleet_staging.stage_host({}, {}, tmp_path, log=lambda *_: None) == []


def test_stage_host_non_fatal_without_skyportal(tmp_path):
    # Standalone (no SkyPortal/DB importable) degrades to hostless, never raises.
    out = fleet_staging.stage_host({}, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
    assert out == []
    assert not (tmp_path / fleet_staging.HOST_FILE).exists()


def test_host_file_matches_bridge():
    import fleet_bridge

    assert fleet_staging.HOST_FILE == fleet_bridge.HOST_FILE
