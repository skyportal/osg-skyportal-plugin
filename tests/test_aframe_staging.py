"""Tests for aframe model staging: local paths are copied to canonical basenames,
OSDF/HTTPS URLs are fetched, background is optional, missing required files warn."""

from pathlib import Path

import aframe_staging
import osdf


def test_local_paths_copied_to_canonical_basenames(tmp_path):
    weights = tmp_path / "someweights.pt"
    weights.write_bytes(b"w")
    config = tmp_path / "cfg.yaml"
    config.write_text("k: v")
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models({"weights": str(weights), "config": str(config)}, job)

    assert {p.name for p in staged} == {"aframe.pt", "aframe_config_bbh.yaml"}
    assert (job / "aframe.pt").read_bytes() == b"w"
    assert (job / "aframe_config_bbh.yaml").exists()


def test_osdf_urls_are_fetched(tmp_path, monkeypatch):
    seen = {}

    def fake_download(url, local_path, token_path=None):
        Path(local_path).write_bytes(b"dl")
        seen[url] = token_path
        return local_path

    monkeypatch.setattr(osdf, "download", fake_download)
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models(
        {
            "weights": "https://origin.example/aframe/w.pt",
            "background": "osdf:///ospool/ap41/data/u/aframe/bg.hdf5",
        },
        job,
        read_token_path="/tok",
    )

    assert {p.name for p in staged} == {"aframe.pt", "background.hdf5"}
    assert (job / "aframe.pt").read_bytes() == b"dl"
    assert (job / "background.hdf5").exists()
    assert seen == {
        "https://origin.example/aframe/w.pt": "/tok",
        "osdf:///ospool/ap41/data/u/aframe/bg.hdf5": "/tok",
    }


def test_background_optional_weights_warns(tmp_path):
    warnings = []
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models({}, job, log=warnings.append)

    assert staged == []
    # weights and config warn; background does not
    assert any("weights" in w for w in warnings)
    assert any("config" in w for w in warnings)
    assert not any("background" in w for w in warnings)


def test_transfer_urls_passthrough_returns_urls(tmp_path, monkeypatch):
    # With transfer_urls set, URL sources are handed to transfer_input_files as-is
    # (the worker pulls them); osdf.download must not be called.
    def boom(*a, **k):
        raise AssertionError("download should not run in passthrough mode")

    monkeypatch.setattr(osdf, "download", boom)
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models(
        {
            "transfer_urls": True,
            "weights": "igwn+osdf:///igwn/cit/staging/michael.coughlin/aframe/v1/aframe.pt",
            "config": "osdf:///igwn/cit/staging/michael.coughlin/aframe/v1/aframe_config_bbh.yaml",
        },
        job,
    )

    assert staged == [
        "igwn+osdf:///igwn/cit/staging/michael.coughlin/aframe/v1/aframe.pt",
        "osdf:///igwn/cit/staging/michael.coughlin/aframe/v1/aframe_config_bbh.yaml",
    ]
    # nothing written to the pod's job dir
    assert list(job.iterdir()) == []


def test_transfer_urls_warns_on_basename_mismatch(tmp_path):
    warnings = []
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models(
        {"transfer_urls": True, "weights": "igwn+osdf:///igwn/cit/staging/u/aframe/aframe.v1.pt"},
        job,
        log=warnings.append,
    )

    assert staged == ["igwn+osdf:///igwn/cit/staging/u/aframe/aframe.v1.pt"]
    assert any("aframe.pt" in w and "basename" in w for w in warnings)


def test_transfer_urls_still_copies_local_files(tmp_path):
    # A local path is copied even when transfer_urls is set (only URLs pass through).
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    job = tmp_path / "job"
    job.mkdir()

    staged = aframe_staging.stage_models({"transfer_urls": True, "weights": str(weights)}, job)

    assert [Path(p).name for p in staged] == ["aframe.pt"]
    assert (job / "aframe.pt").read_bytes() == b"w"
