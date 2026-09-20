"""Tests for the IGWN strain fetch helper and the per-detector resolvers. The
gwdatafind/pelican calls shell out to the CVMFS igwn env on a worker, so they are
mocked here; the real fetch is exercised on a live OSG submit."""

import json
import subprocess

import aframe_bridge
import igwn_strain
import pygrb_bridge


def test_find_osdf_urls_parses_subprocess_json(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        urls = ["osdf:///igwn/ligo/frames/O4/hoft_C00/H1/x-1-4096.gwf"]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(urls), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    urls = igwn_strain.find_osdf_urls("H", "H1_HOFT_C00", 1000, 1064, host="datafind.igwn.org")
    assert urls == ["osdf:///igwn/ligo/frames/O4/hoft_C00/H1/x-1-4096.gwf"]
    # discovery runs in the CVMFS igwn env and asks for osdf URLs
    assert seen["cmd"][0] == f"{igwn_strain.IGWN_ENV}/bin/python" or seen["cmd"][0] == "python3"
    assert "urltype='osdf'" in seen["cmd"][-1]


def test_find_osdf_urls_raises_on_failure(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        igwn_strain.find_osdf_urls("H", "H1_HOFT_C00", 1000, 1064)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "gwdatafind" in str(e)


def test_fetch_frames_empty_when_no_urls(monkeypatch, tmp_path):
    monkeypatch.setattr(igwn_strain, "find_osdf_urls", lambda *a, **k: [])
    assert igwn_strain.fetch_frames("H", "H1_HOFT_C00", 1000, 1064, tmp_path) == []


def test_fetch_frames_pelican_gets_each_url(monkeypatch, tmp_path):
    urls = [
        "osdf:///igwn/ligo/frames/O4/hoft_C00/H1/H-a-4096.gwf",
        "osdf:///igwn/ligo/frames/O4/hoft_C00/H1/H-b-4096.gwf",
    ]
    monkeypatch.setattr(igwn_strain, "find_osdf_urls", lambda *a, **k: urls)
    got = []

    def fake_get(url, dest, env):
        from pathlib import Path

        Path(dest).write_bytes(b"gwf")
        got.append((url, Path(dest).name))
        return Path(dest)

    monkeypatch.setattr(igwn_strain, "_pelican_get", fake_get)
    frames = igwn_strain.fetch_frames("H", "H1_HOFT_C00", 1000, 1064, tmp_path)
    assert [p.name for p in frames] == ["H-a-4096.gwf", "H-b-4096.gwf"]
    assert got[0] == (urls[0], "H-a-4096.gwf")


def test_token_env_prefers_condor_creds(monkeypatch, tmp_path):
    tok = tmp_path / "scitokens.use"
    tok.write_text("t")
    monkeypatch.delenv("BEARER_TOKEN_FILE", raising=False)
    monkeypatch.setenv("_CONDOR_CREDS", str(tmp_path))
    assert igwn_strain._token_env()["BEARER_TOKEN_FILE"] == str(tok)


def test_per_ifo_resolvers():
    for mod in (pygrb_bridge, aframe_bridge):
        assert mod._per_ifo({}, "frametype", "{ifo}_HOFT_C00", "L1") == "L1_HOFT_C00"
        assert (
            mod._per_ifo({"channel": "{ifo}:GDS-CALIB_STRAIN"}, "channel", "x", "V1")
            == "V1:GDS-CALIB_STRAIN"
        )
        assert (
            mod._per_ifo({"frametype": {"H1": "H1_HOFT_C01"}}, "frametype", "{ifo}_HOFT_C00", "H1")
            == "H1_HOFT_C01"
        )
