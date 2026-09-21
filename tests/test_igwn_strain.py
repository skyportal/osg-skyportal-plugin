"""Tests for the IGWN strain fetch helper and the per-detector resolvers. The
gwdatafind/pelican calls shell out to the CVMFS igwn env on a worker, so they are
mocked here; the real fetch is exercised on a live OSG submit."""

import json
import subprocess

import igwn_strain


def test_find_osdf_urls_in_process_when_gwdatafind_importable(monkeypatch):
    # The pycbc image ships gwdatafind: discovery runs in-process, never touching
    # the CVMFS igwn env (the subprocess path) that makes workers crawl.
    import sys
    import types

    seen = {}
    fake = types.ModuleType("gwdatafind")

    def find_urls(obs, ft, s, e, **kw):
        seen.update(obs=obs, ft=ft, s=s, e=e, kw=kw)
        return ["osdf:///igwn/ligo/frames/O4/hoft_C00/H1/x-1-4096.gwf"]

    fake.find_urls = find_urls
    monkeypatch.setitem(sys.modules, "gwdatafind", fake)

    def no_subprocess(*a, **k):
        raise AssertionError("must not shell to the CVMFS env when gwdatafind imports")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    urls = igwn_strain.find_osdf_urls("H", "H1_HOFT_C00", 1000, 1064)
    assert urls == ["osdf:///igwn/ligo/frames/O4/hoft_C00/H1/x-1-4096.gwf"]
    assert seen["kw"]["urltype"] == "osdf" and seen["obs"] == "H"


def test_find_osdf_urls_parses_subprocess_json(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "gwdatafind", None)  # force the CVMFS fallback
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
    import sys

    monkeypatch.setitem(sys.modules, "gwdatafind", None)  # force the CVMFS fallback

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        igwn_strain.find_osdf_urls("H", "H1_HOFT_C00", 1000, 1064)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "gwdatafind" in str(e)


def test_pelican_exe_prefers_shipped_binary(monkeypatch, tmp_path):
    # No in-image pelican -> use a ./pelican shipped with the job (buoy has none).
    monkeypatch.setattr(igwn_strain.shutil, "which", lambda _: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pelican").write_bytes(b"#!/bin/true\n")
    assert igwn_strain._pelican_exe() == str(tmp_path / "pelican")


def test_pelican_get_falls_back_to_pelicanfs(monkeypatch, tmp_path):
    # No CLI anywhere -> pelicanfs (Python) is used.
    monkeypatch.setattr(igwn_strain, "_pelican_exe", lambda: None)
    dest = tmp_path / "H-a-4096.gwf"
    called = {}

    def fake_pfs(url, d, env):
        from pathlib import Path

        Path(d).write_bytes(b"gwf")
        called["url"] = url
        return Path(d)

    monkeypatch.setattr(igwn_strain, "_pelicanfs_get", fake_pfs)
    out = igwn_strain._pelican_get("osdf:///x/H-a-4096.gwf", dest, {})
    assert out == dest and called["url"] == "osdf:///x/H-a-4096.gwf"


def test_pelican_get_raises_when_no_client(monkeypatch, tmp_path):
    monkeypatch.setattr(igwn_strain, "_pelican_exe", lambda: None)

    def no_pfs(*a, **k):
        raise ImportError("no pelicanfs")

    monkeypatch.setattr(igwn_strain, "_pelicanfs_get", no_pfs)
    try:
        igwn_strain._pelican_get("osdf:///x/y.gwf", tmp_path / "y.gwf", {})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "pelican get failed" in str(e)


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
    # Shared by both bridges; the default template, a "{ifo}" string, and a dict.
    assert igwn_strain.per_ifo({}, "frametype", "{ifo}_HOFT_C00", "L1") == "L1_HOFT_C00"
    assert (
        igwn_strain.per_ifo({"channel": "{ifo}:GDS-CALIB_STRAIN"}, "channel", "x", "V1")
        == "V1:GDS-CALIB_STRAIN"
    )
    assert (
        igwn_strain.per_ifo(
            {"frametype": {"H1": "H1_HOFT_C01"}}, "frametype", "{ifo}_HOFT_C00", "H1"
        )
        == "H1_HOFT_C01"
    )
