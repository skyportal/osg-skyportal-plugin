"""Pure-logic tests for oracle_staging (no network): latest-candid selection, the
BOOM cutout field decode, and the config/inputs guards that skip the fetch."""

import base64
import gzip

import oracle_staging


def test_latest_candid_picks_highest_jd():
    alerts = [
        {"candid": "a", "candidate": {"jd": 2460000.5}},
        {"candid": "b", "candidate": {"jd": 2460010.5}},
        {"candid": "c", "candidate": {"jd": 2460005.5}},
    ]
    assert oracle_staging._latest_candid(alerts) == "b"


def test_latest_candid_empty_is_none():
    assert oracle_staging._latest_candid([]) is None
    assert oracle_staging._latest_candid(None) is None


def test_gzip_fits_bytes_from_base64_and_stampdata():
    raw = gzip.compress(b"SIMPLE  = T / fake fits")
    b64 = base64.b64encode(raw).decode()
    # A base64 string, Avro stampData, and Mongo $binary all yield the gzipped bytes.
    assert oracle_staging._gzip_fits_bytes(b64) == raw
    assert oracle_staging._gzip_fits_bytes({"stampData": b64}) == raw
    assert oracle_staging._gzip_fits_bytes({"$binary": {"base64": b64}}) == raw
    assert oracle_staging._gzip_fits_bytes(None) is None
    assert oracle_staging._gzip_fits_bytes({}) is None


def test_stage_cutout_skips_without_broker_or_token(tmp_path):
    # No broker id / real token configured -> no fetch, no file, no raise.
    cfg = {"skyportal": {"base_url": "http://x", "api_token": "replace_with_token"}}
    assert (
        oracle_staging.stage_cutout(cfg, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
        == []
    )
    cfg = {"skyportal": {"base_url": "http://x", "api_token": "tok"}, "oracle": {"broker_id": None}}
    assert (
        oracle_staging.stage_cutout(cfg, {"obj": {"id": "ZTF1"}}, tmp_path, log=lambda *_: None)
        == []
    )


def test_stage_cutout_skips_without_obj_id(tmp_path):
    cfg = {"skyportal": {"base_url": "http://x", "api_token": "tok"}, "oracle": {"broker_id": 3}}
    assert oracle_staging.stage_cutout(cfg, {}, tmp_path, log=lambda *_: None) == []
