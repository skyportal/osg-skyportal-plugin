"""Pure-logic tests for oracle_staging (no DB): latest-alert selection, the BOOM
cutout field decode, and the best-effort guards that skip the fetch. The in-process
DB fetch itself needs a live SkyPortal and is exercised there, not here."""

import base64
import gzip

import oracle_staging


def test_latest_alert_picks_highest_jd():
    alerts = [
        {"_id": "a", "candidate": {"jd": 2460000.5}},
        {"_id": "b", "candidate": {"jd": 2460010.5}},
        {"_id": "c", "candidate": {"jd": 2460005.5}},
    ]
    assert oracle_staging._latest_alert(alerts)["_id"] == "b"


def test_latest_alert_empty_is_none():
    assert oracle_staging._latest_alert([]) is None
    assert oracle_staging._latest_alert(None) is None


def test_gzip_fits_bytes_from_base64_and_stampdata():
    raw = gzip.compress(b"SIMPLE  = T / fake fits")
    b64 = base64.b64encode(raw).decode()
    # A base64 string, Avro stampData, and Mongo $binary all yield the gzipped bytes.
    assert oracle_staging._gzip_fits_bytes(b64) == raw
    assert oracle_staging._gzip_fits_bytes({"stampData": b64}) == raw
    assert oracle_staging._gzip_fits_bytes({"$binary": {"base64": b64}}) == raw
    assert oracle_staging._gzip_fits_bytes(None) is None
    assert oracle_staging._gzip_fits_bytes({}) is None


def test_stage_cutout_skips_without_obj_id(tmp_path):
    assert oracle_staging.stage_cutout({}, {}, tmp_path, log=lambda *_: None) == []


def test_stage_cutout_non_fatal_without_skyportal(tmp_path):
    # Standalone (no SkyPortal/DB importable) degrades to no cutout, never raises.
    out = oracle_staging.stage_cutout({}, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
    assert out == []
    assert not (tmp_path / oracle_staging.CUTOUT_FILE).exists()
