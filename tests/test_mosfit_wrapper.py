"""Tests for the OSG-side mosfit wrapper. The real mosfit fit is never run here;
the bridge is monkeypatched (mirrors test_fiesta_wrapper)."""

import json
from pathlib import Path

import mosfit_bridge
import mosfit_wrapper


def test_run_delegates_to_bridge(monkeypatch):
    captured: dict = {}

    def fake_bridge(payload, *, resource_id="obj"):
        captured["payload"] = payload
        captured["resource_id"] = resource_id
        return {"status": "success", "message": "fake bridge", "source": "slsn"}

    monkeypatch.setattr(mosfit_bridge, "run_from_skyportal_inputs", fake_bridge)

    out = mosfit_wrapper.run(
        {
            "photometry": "p.csv",
            "resource_id": "ZTF1",
            "analysis_parameters": {"source": "slsn"},
        }
    )
    assert out["message"] == "fake bridge"
    assert captured["resource_id"] == "ZTF1"
    assert captured["payload"]["analysis_parameters"]["source"] == "slsn"


def test_bundle_carries_overlay_and_posteriors(tmp_path):
    rf = tmp_path / "r.json"
    rf.write_text(json.dumps({"model": "slsn", "posterior_medians": {"nhhost": 1.0}}))
    bundle = mosfit_wrapper.bundle_for_skyportal(
        {
            "status": "success",
            "message": "mosfit fit complete (model=slsn, sampler=ensembler)",
            "source": "slsn",
            "model_lightcurve": {"ztfr": [[59000.0, 19.1, 18.9, 19.3]]},
            "posterior_samples": {"mejecta": [1.0, 2.0]},
            "n_detections": 7,
            "json_result_file": str(rf),
        }
    )
    a = bundle["analysis"]
    assert bundle["status"] == "success"
    assert a["model_name"] == "slsn"
    assert a["model_lightcurve"]["ztfr"][0][0] == 59000.0
    assert a["posterior_samples"]["mejecta"] == [1.0, 2.0]
    assert a["n_detections"] == 7
    assert a["results"]["format"] == "json"
    assert a["results"]["data"]["model"] == "slsn"


def test_bundle_failure_is_empty_analysis():
    bundle = mosfit_wrapper.bundle_for_skyportal(
        {"status": "failure", "message": "Not enough detections to fit (need at least 2, have 1)."}
    )
    assert bundle["status"] == "failure"
    assert bundle["analysis"] == {}
    assert "at least 2" in bundle["message"]


def test_materialize_inputs_writes_csv_content(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = mosfit_wrapper._materialize_inputs(
        {
            "photometry": "mjd,filter,mag,magerr\n59000,ztfr,19.1,0.1\n",
            "redshift": "redshift\n0.1\n",
        }
    )
    assert inputs["photometry"] == "photometry.csv"
    assert inputs["redshift"] == "redshift.csv"
    assert Path("photometry.csv").read_text().startswith("mjd,filter")
    assert Path("redshift.csv").read_text().startswith("redshift")


def test_materialize_inputs_leaves_bare_filename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = mosfit_wrapper._materialize_inputs({"photometry": "already.csv"})
    assert inputs["photometry"] == "already.csv"  # no newline -> treated as a path
    assert not Path("photometry.csv").exists()


def test_main_success_prints_bundle(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs.json").write_text(json.dumps({"resource_id": "obj", "photometry": "p.csv"}))
    monkeypatch.setattr(
        mosfit_bridge,
        "run_from_skyportal_inputs",
        lambda payload, *, resource_id="obj": {
            "status": "success",
            "message": "ok",
            "source": "default",
            "model_lightcurve": {"ztfr": [[59000.0, 19.0, 18.8, 19.2]]},
        },
    )
    rc = mosfit_wrapper.main()
    assert rc == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "success"
    assert printed["analysis"]["model_lightcurve"]["ztfr"][0][1] == 19.0


def test_main_missing_inputs_is_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no inputs.json
    rc = mosfit_wrapper.main()
    assert rc == 1
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "failure"
    assert "_traceback" in printed
