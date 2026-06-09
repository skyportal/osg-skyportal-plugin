"""Tests for the OSG-side wrapper. The real NMMA path is gated behind dry_run."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import nmma_wrapper


def test_run_nmma_dry_run_returns_stub():
    out = nmma_wrapper.run_nmma({"analysis_parameters": {"dry_run": True, "source": "Me2017"}})
    assert out["_stub"] is True
    assert out["_source"] == "Me2017"
    assert "log_bayes_factor" in out


def test_run_nmma_delegates_to_bridge(monkeypatch):
    """Non-dry-run path calls nmma.skyportal_osg.run_from_skyportal_inputs."""
    captured: dict = {}

    def fake_bridge(payload, *, resource_id="obj"):
        captured["payload"] = payload
        captured["resource_id"] = resource_id
        return {
            "status": "success",
            "message": "fake bridge",
            "log_bayes_factor": 2.5,
            "posterior_file": "/tmp/posterior.dat",
            "json_result_file": "/tmp/result.json",
            "plot_file": None,
            "outdir": "/tmp/out",
        }

    import nmma.skyportal_osg

    monkeypatch.setattr(nmma.skyportal_osg, "run_from_skyportal_inputs", fake_bridge)

    out = nmma_wrapper.run_nmma(
        {
            "photometry": "p.csv",
            "redshift": "z.csv",
            "resource_id": "ZTF1",
            "analysis_parameters": {"source": "Piro2021"},
        }
    )
    assert out["log_bayes_factor"] == 2.5
    assert captured["resource_id"] == "ZTF1"
    assert captured["payload"]["analysis_parameters"]["source"] == "Piro2021"


def test_bundle_for_skyportal_stub_includes_results():
    stub = {"log_bayes_factor": 0.5, "_stub": True, "_source": "Me2017"}
    bundle = nmma_wrapper.bundle_for_skyportal(stub)
    assert bundle["status"] == "success"
    assert "log Bayes factor=0.5" in bundle["message"]
    decoded = json.loads(base64.b64decode(bundle["analysis"]["results"]["data"]))
    assert decoded == stub


def test_bundle_for_skyportal_includes_plots(tmp_path):
    plot1 = tmp_path / "corner.png"
    plot1.write_bytes(b"\x89PNG")
    plot2 = tmp_path / "missing.png"  # does not exist; should be skipped
    bundle = nmma_wrapper.bundle_for_skyportal(
        {"log_bayes_factor": 1.0, "_stub": True}, plot_files=[plot1, plot2]
    )
    assert len(bundle["analysis"]["plots"]) == 1
    assert bundle["analysis"]["plots"][0]["format"] == "png"


def test_upload_uses_bearer_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSDF_OUTPUT_URL", "https://origin/jobs/x.json")
    monkeypatch.setenv("BEARER_TOKEN", "abc.def.ghi")
    with patch("nmma_wrapper.requests.put") as mput:
        mput.return_value = MagicMock(raise_for_status=lambda: None)
        ok = nmma_wrapper.upload({"status": "success", "message": "", "analysis": {}})
        assert ok is True
        _, kwargs = mput.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer abc.def.ghi"
        assert kwargs["headers"]["Content-Type"] == "application/json"


def test_upload_returns_false_without_url(monkeypatch):
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    assert nmma_wrapper.upload({"status": "success", "analysis": {}}) is False


def test_main_dry_run_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("inputs.json").write_text(
        json.dumps({"analysis_parameters": {"dry_run": True, "source": "Piro2021"}})
    )
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    rc = nmma_wrapper.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    bundle = json.loads(out)
    assert bundle["status"] == "success"
    assert "Piro2021" in base64.b64decode(bundle["analysis"]["results"]["data"]).decode()


def test_main_failure_path_writes_failure_bundle(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # No inputs.json → load_inputs raises → bundle status is "failure".
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    rc = nmma_wrapper.main()
    assert rc == 1
    bundle = json.loads(capsys.readouterr().out.strip())
    assert bundle["status"] == "failure"
    assert "_traceback" in bundle
