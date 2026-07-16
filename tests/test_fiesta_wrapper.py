"""Tests for the OSG-side fiesta wrapper. The real fiesta path is gated behind dry_run."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fiesta_bridge
import fiesta_wrapper


def test_run_fit_dry_run_returns_stub():
    out = fiesta_wrapper.run_fit({"analysis_parameters": {"dry_run": True, "source": "Bu2025_MLP"}})
    assert out["_stub"] is True
    assert out["_source"] == "Bu2025_MLP"
    assert "log_bayes_factor" in out


def test_run_fit_delegates_to_bridge(monkeypatch):
    """Non-dry-run path calls fiesta_bridge.run_from_skyportal_inputs."""
    captured: dict = {}

    def fake_bridge(payload, *, resource_id="obj"):
        captured["payload"] = payload
        captured["resource_id"] = resource_id
        return {
            "status": "success",
            "message": "fake bridge",
            "json_result_file": "/tmp/result.json",
            "plot_file": None,
            "outdir": "/tmp/out",
        }

    monkeypatch.setattr(fiesta_bridge, "run_from_skyportal_inputs", fake_bridge)

    out = fiesta_wrapper.run_fit(
        {
            "photometry": "p.csv",
            "redshift": "z.csv",
            "resource_id": "ZTF1",
            "analysis_parameters": {"source": "AfterglowModel"},
        }
    )
    assert out["message"] == "fake bridge"
    assert captured["resource_id"] == "ZTF1"
    assert captured["payload"]["analysis_parameters"]["source"] == "AfterglowModel"


def test_bundle_for_skyportal_stub_includes_results():
    stub = {"log_bayes_factor": 0.5, "_stub": True, "_source": "Bu2025_MLP"}
    bundle = fiesta_wrapper.bundle_for_skyportal(stub)
    assert bundle["status"] == "success"
    assert "log Bayes factor=0.5" in bundle["message"]
    assert bundle["analysis"]["results"]["format"] == "json"
    assert bundle["analysis"]["results"]["data"] == stub


def test_bundle_for_skyportal_uses_fit_message_without_evidence():
    """Fiesta (MCMC) has no log Bayes factor; the bundle uses the fit's message."""
    bundle = fiesta_wrapper.bundle_for_skyportal(
        {"status": "success", "message": "fiesta fit complete (model=AfterglowModel)"}
    )
    assert bundle["message"] == "fiesta fit complete (model=AfterglowModel)"


def test_bundle_for_skyportal_includes_plots(tmp_path):
    plot1 = tmp_path / "corner.png"
    plot1.write_bytes(b"\x89PNG")
    plot2 = tmp_path / "missing.png"  # does not exist; should be skipped
    bundle = fiesta_wrapper.bundle_for_skyportal(
        {"log_bayes_factor": 1.0, "_stub": True}, plot_files=[plot1, plot2]
    )
    assert len(bundle["analysis"]["plots"]) == 1
    assert bundle["analysis"]["plots"][0]["format"] == "png"


def test_upload_uses_bearer_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSDF_OUTPUT_URL", "https://origin/jobs/x.json")
    monkeypatch.setenv("BEARER_TOKEN", "abc.def.ghi")
    with patch("fiesta_wrapper.requests.put") as mput:
        mput.return_value = MagicMock(raise_for_status=lambda: None)
        ok = fiesta_wrapper.upload({"status": "success", "message": "", "analysis": {}})
        assert ok is True
        _, kwargs = mput.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer abc.def.ghi"
        assert kwargs["headers"]["Content-Type"] == "application/json"


def test_upload_returns_false_without_url(monkeypatch):
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    assert fiesta_wrapper.upload({"status": "success", "analysis": {}}) is False


def test_main_dry_run_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("inputs.json").write_text(
        json.dumps({"analysis_parameters": {"dry_run": True, "source": "AfterglowModel"}})
    )
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    rc = fiesta_wrapper.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    bundle = json.loads(out)
    assert bundle["status"] == "success"
    assert "AfterglowModel" in json.dumps(bundle["analysis"]["results"]["data"])


def test_main_failure_path_writes_failure_bundle(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # No inputs.json → load_inputs raises → bundle status is "failure".
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    rc = fiesta_wrapper.main()
    assert rc == 1
    bundle = json.loads(capsys.readouterr().out.strip())
    assert bundle["status"] == "failure"
    assert "_traceback" in bundle


# ---- SkyPortal-format input handling --------------------------------------------------------


def test_materialize_inputs_writes_csv_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = "mjd,filter,mag,magerr\n60000.0,ztfg,18.5,0.1\n60000.5,ztfr,18.7,0.1\n"
    inputs = {"photometry": csv, "redshift": "redshift\n0.1\n"}
    out = fiesta_wrapper._materialize_inputs(inputs)
    assert out["photometry"] == "photometry.csv"
    assert out["redshift"] == "redshift.csv"
    assert Path("photometry.csv").read_text() == csv


def test_materialize_inputs_leaves_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = {"photometry": "already_a_file.csv"}
    assert fiesta_wrapper._materialize_inputs(inputs)["photometry"] == "already_a_file.csv"


@pytest.mark.parametrize("val", ["True", "true", "t", "1", "yes", True])
def test_dry_run_truthy_variants_return_stub(val):
    out = fiesta_wrapper.run_fit({"analysis_parameters": {"dry_run": val}})
    assert out.get("_stub") is True


@pytest.mark.parametrize("val", ["False", "false", "", "0", "no"])
def test_dry_run_falsy_variants_hit_bridge(val, monkeypatch):
    monkeypatch.setattr(
        fiesta_bridge,
        "run_from_skyportal_inputs",
        lambda payload, *, resource_id="obj": {"status": "success", "message": "hit bridge"},
    )
    out = fiesta_wrapper.run_fit({"photometry": "p.csv", "analysis_parameters": {"dry_run": val}})
    assert "_stub" not in out
    assert out["message"] == "hit bridge"


def test_main_passes_model_lightcurve_into_bundle(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("inputs.json").write_text(json.dumps({"analysis_parameters": {}}))
    monkeypatch.delenv("OSDF_OUTPUT_URL", raising=False)
    mlc = {"ztfg": [[60000.0, 18.5, 18.0, 19.0]]}
    monkeypatch.setattr(
        fiesta_wrapper,
        "run_fit",
        lambda inputs: {"status": "success", "message": "ok", "model_lightcurve": mlc},
    )
    rc = fiesta_wrapper.main()
    assert rc == 0
    bundle = json.loads(capsys.readouterr().out.strip())
    assert bundle["analysis"]["model_lightcurve"] == mlc
