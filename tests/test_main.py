"""Smoke + contract tests against `main` using fake htcondor + baselayer shims."""

import json
from unittest.mock import patch

import pytest
import tornado.testing

import main


def test_submit_job_round_trips_into_jobs(plugin_cfg, fake_queue):
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="nmma_osg",
        resource_id="ZTF20abc",
        callback_url="http://callback/x",
        callback_method="POST",
        inputs={"analysis_parameters": {"arguments": "5"}},
    )
    assert cid in main.JOBS
    rec = main.JOBS[cid]
    assert rec.analysis_name == "nmma_osg"
    assert rec.resource_id == "ZTF20abc"
    assert rec.callback_url == "http://callback/x"
    assert any(ad["ClusterId"] == cid for ad in fake_queue)


def test_poll_marks_completed_via_history(plugin_cfg, fake_queue, fake_history):
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id=None,
        callback_url=None,
        callback_method="POST",
        inputs={},
    )
    # Pretend the job left the live queue and landed in history.
    fake_queue.clear()
    fake_history.append({"ClusterId": cid, "JobStatus": 4, "CompletionDate": 1700000000})

    main.poll_once(plugin_cfg)
    rec = main.JOBS[cid]
    assert rec.status == "completed"
    assert rec.completed_at == 1700000000


def test_poll_posts_callback_with_skyportal_shape(
    plugin_cfg, fake_queue, fake_history, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # logs/ writes here
    (tmp_path / "logs").mkdir()
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id="ZTF1",
        callback_url="http://sp/api/obj/x/analysis/1",
        callback_method="POST",
        inputs={},
    )
    (tmp_path / "logs" / f"job.{cid}.0.out").write_bytes(b"results-from-job\n")
    fake_queue.clear()
    fake_history.append({"ClusterId": cid, "JobStatus": 4, "CompletionDate": 1700000000})

    with patch("main.requests.post") as mock_post:
        main.poll_once(plugin_cfg)
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "http://sp/api/obj/x/analysis/1"
        body = kwargs["json"]
        assert body["status"] == "success"
        # The analysis result is base64'd stdout per the SkyPortal contract.
        assert body["analysis"]["results"]["format"] == "text"
    assert main.JOBS[cid].callback_posted is True


def test_poll_marks_held_job_failure_in_callback(plugin_cfg, fake_queue, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id="ZTF1",
        callback_url="http://sp/api/obj/x/analysis/1",
        callback_method="POST",
        inputs={},
    )
    # Live queue says held.
    fake_queue.clear()
    fake_queue.append(
        {"ClusterId": cid, "JobStatus": 5, "HoldReason": "scitoken expired", "HoldReasonCode": 26}
    )
    main.poll_once(plugin_cfg)
    rec = main.JOBS[cid]
    assert rec.status == "held"
    assert rec.hold_reason == "scitoken expired"
    # Held isn't terminal — callback shouldn't fire yet.
    assert rec.callback_posted is False


# ---- HTTP layer -----------------------------------------------------------------------------


class _AppTestCase(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        cfg = _CONFIG_FOR_HTTP
        return main.build_app(cfg)


# Built once at module import so tornado.testing can pick it up.
_CONFIG_FOR_HTTP = {
    "listener": {"host": "127.0.0.1", "port": 0},
    "htcondor": {
        "collector": None,
        "schedd": None,
        "scitoken_path": "/nonexistent",
        "project_name": "T",
    },
    "defaults": {
        "request_cpus": 1,
        "request_memory": 256,
        "request_disk": 256,
        "max_runtime_seconds": 60,
        "singularity_image": None,
    },
    "poller": {"interval_seconds": 1},
    "caps": {
        "max_concurrent_total": None,
        "max_concurrent_per_analysis": None,
        "max_concurrent_per_resource_per_analysis": 1,
    },
    "osdf": {"output_prefix": None, "read_token_path": None, "write_token_path": None},
    "staging_dir": "staging-test",
    "auth": {"incoming_bearer_token": "secret-test-token"},
    "skyportal": {"base_url": "http://localhost", "api_token": "x"},
}


class TestAnalysisEndpoint(_AppTestCase):
    def test_get_returns_status_active(self):
        r = self.fetch("/analysis/nmma_osg")
        assert r.code == 200
        body = json.loads(r.body)
        assert body == {"status": "active", "analysis": "nmma_osg"}

    def test_post_requires_bearer(self):
        r = self.fetch(
            "/analysis/nmma_osg",
            method="POST",
            body=json.dumps({"inputs": {}, "callback_url": "x", "callback_method": "POST"}),
            headers={},
        )
        assert r.code == 401

    def test_post_rejects_invalid_json(self):
        r = self.fetch(
            "/analysis/nmma_osg",
            method="POST",
            body=b"not-json",
            headers={"Authorization": "Bearer secret-test-token"},
        )
        assert r.code == 400

    def test_post_requires_keys(self):
        r = self.fetch(
            "/analysis/nmma_osg",
            method="POST",
            body=json.dumps({"inputs": {}}),
            headers={"Authorization": "Bearer secret-test-token"},
        )
        assert r.code == 400
        assert b"missing required keys" in r.body

    def test_post_happy_path_returns_cluster_id(self):
        r = self.fetch(
            "/analysis/nmma_osg",
            method="POST",
            body=json.dumps(
                {
                    "inputs": {"analysis_parameters": {"arguments": "5"}},
                    "callback_url": "http://callback/x",
                    "callback_method": "POST",
                    "resource_id": "ZTF1",
                }
            ),
            headers={"Authorization": "Bearer secret-test-token"},
        )
        assert r.code == 200, r.body
        body = json.loads(r.body)
        assert body["status"] == "pending"
        assert isinstance(body["cluster_id"], int)
        assert body["cluster_id"] in main.JOBS


class TestJobsEndpoint(_AppTestCase):
    def test_status_404_for_unknown(self):
        r = self.fetch("/jobs/999999")
        assert r.code == 404

    def test_list_jobs_empty(self):
        r = self.fetch("/jobs")
        assert r.code == 200
        assert json.loads(r.body) == {"jobs": []}


# ---- schedd-as-DB rehydrate -----------------------------------------------------------------


def test_submit_stamps_skyportal_classads(plugin_cfg, last_submit_desc):
    main.submit_job(
        plugin_cfg,
        analysis_name="nmma_osg",
        resource_id="ZTF20abc",
        callback_url="http://sp/api/obj/ZTF20abc/analysis/42",
        callback_method="POST",
        inputs={},
    )
    assert last_submit_desc["+SkyPortalAnalysisName"] == '"nmma_osg"'
    assert last_submit_desc["+SkyPortalCallback"] == '"http://sp/api/obj/ZTF20abc/analysis/42"'
    assert last_submit_desc["+SkyPortalCallbackMethod"] == '"POST"'
    assert last_submit_desc["+SkyPortalResourceId"] == '"ZTF20abc"'


def test_rehydrate_picks_up_jobs_from_schedd(plugin_cfg, fake_queue):
    # Simulate a restart: pre-populate the schedd as if jobs are still running.
    fake_queue.append(
        {
            "ClusterId": 42,
            "JobStatus": 2,  # running
            "QDate": 1700000000,
            "ProjectName": "Test",
            "SkyPortalAnalysisName": "nmma_osg",
            "SkyPortalCallback": "http://sp/api/obj/X/analysis/7",
            "SkyPortalCallbackMethod": "POST",
            "SkyPortalResourceId": "ZTF99",
        }
    )
    assert main.JOBS == {}
    n = main.rehydrate_jobs(plugin_cfg)
    assert n == 1
    rec = main.JOBS[42]
    assert rec.status == "running"
    assert rec.analysis_name == "nmma_osg"
    assert rec.callback_url == "http://sp/api/obj/X/analysis/7"
    assert rec.resource_id == "ZTF99"


# ---- caps ----------------------------------------------------------------------------------


def test_check_caps_blocks_global(plugin_cfg, monkeypatch):
    plugin_cfg["caps"] = {"max_concurrent_total": 1}
    main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id="A",
        callback_url=None,
        callback_method="POST",
        inputs={},
    )
    reason = main.check_caps(plugin_cfg, "x", "B")
    assert reason and "global cap" in reason


def test_check_caps_blocks_per_analysis(plugin_cfg):
    plugin_cfg["caps"] = {"max_concurrent_per_analysis": 1}
    main.submit_job(
        plugin_cfg,
        analysis_name="nmma_osg",
        resource_id="A",
        callback_url=None,
        callback_method="POST",
        inputs={},
    )
    assert main.check_caps(plugin_cfg, "nmma_osg", "B") is not None
    assert main.check_caps(plugin_cfg, "different", "B") is None


def test_check_caps_blocks_per_resource(plugin_cfg):
    plugin_cfg["caps"] = {"max_concurrent_per_resource_per_analysis": 1}
    main.submit_job(
        plugin_cfg,
        analysis_name="nmma_osg",
        resource_id="ZTF1",
        callback_url=None,
        callback_method="POST",
        inputs={},
    )
    assert main.check_caps(plugin_cfg, "nmma_osg", "ZTF1") is not None
    assert main.check_caps(plugin_cfg, "nmma_osg", "ZTF2") is None


def test_check_caps_ignores_completed_jobs(plugin_cfg):
    plugin_cfg["caps"] = {"max_concurrent_total": 1}
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id="A",
        callback_url=None,
        callback_method="POST",
        inputs={},
    )
    main.JOBS[cid].completed_at = 1.0  # mark terminal
    assert main.check_caps(plugin_cfg, "x", "B") is None


class TestCapHttp(_AppTestCase):
    def get_app(self):
        cfg = dict(_CONFIG_FOR_HTTP)
        cfg["caps"] = {"max_concurrent_per_resource_per_analysis": 1}
        return main.build_app(cfg)

    def test_429_on_per_resource_cap(self):
        body = json.dumps(
            {
                "inputs": {},
                "callback_url": "http://cb",
                "callback_method": "POST",
                "resource_id": "ZTF1",
            }
        )
        headers = {"Authorization": "Bearer secret-test-token"}
        r1 = self.fetch("/analysis/nmma_osg", method="POST", body=body, headers=headers)
        assert r1.code == 200
        r2 = self.fetch("/analysis/nmma_osg", method="POST", body=body, headers=headers)
        assert r2.code == 429
        assert r2.headers.get("Retry-After") == "60"


# ---- NMMA wrapper bridging -----------------------------------------------------------------


def test_submit_with_wrapper_stages_inputs_and_records_osdf_url(plugin_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin_cfg["osdf"] = {"output_prefix": "https://origin/jobs"}
    plugin_cfg["staging_dir"] = str(tmp_path / "stg")
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="nmma_osg",
        resource_id="ZTF1",
        callback_url="http://cb",
        callback_method="POST",
        inputs={"analysis_parameters": {"use_wrapper": True, "dry_run": True}},
    )
    rec = main.JOBS[cid]
    assert rec.osdf_output_url
    assert rec.osdf_output_url.startswith("https://origin/jobs/")
    # inputs.json was staged
    staged = list((tmp_path / "stg").glob("*/inputs.json"))
    assert staged, "expected one staged inputs.json"
    assert "use_wrapper" in staged[0].read_text()


def test_build_callback_body_prefers_osdf_bundle(plugin_cfg, monkeypatch):
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id="ZTF1",
        callback_url="http://cb",
        callback_method="POST",
        inputs={},
    )
    rec = main.JOBS[cid]
    rec.osdf_output_url = "https://origin/jobs/abc.json"
    rec.status = "completed"

    fake_bundle = {"status": "success", "message": "from wrapper", "analysis": {"x": 1}}

    def fake_fetch(r, cfg):
        return fake_bundle

    monkeypatch.setattr(main, "fetch_osdf_bundle", fake_fetch)
    body = main.build_callback_body(rec, plugin_cfg)
    assert body == fake_bundle


def test_build_callback_body_falls_back_when_no_osdf(plugin_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    cid = main.submit_job(
        plugin_cfg,
        analysis_name="x",
        resource_id=None,
        callback_url="http://cb",
        callback_method="POST",
        inputs={},
    )
    (tmp_path / "logs" / f"job.{cid}.0.out").write_bytes(b"hi")
    rec = main.JOBS[cid]
    rec.status = "completed"
    body = main.build_callback_body(rec, plugin_cfg)
    assert body["status"] == "success"
    assert body["analysis"]["results"]["format"] == "text"


def test_rehydrate_is_idempotent(plugin_cfg, fake_queue):
    fake_queue.append(
        {
            "ClusterId": 100,
            "JobStatus": 1,
            "QDate": 1700000000,
            "ProjectName": "Test",
            "SkyPortalAnalysisName": "x",
            "SkyPortalCallback": "http://cb",
            "SkyPortalCallbackMethod": "POST",
            "SkyPortalResourceId": "",
        }
    )
    main.rehydrate_jobs(plugin_cfg)
    main.rehydrate_jobs(plugin_cfg)  # should not double-adopt
    assert len(main.JOBS) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
