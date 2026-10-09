"""Staged OSDF products are removed once the job that needed them is over."""


import pytest

import main


def _rec(staged, cluster_id=1):
    return main.JobRecord(
        cluster_id=cluster_id,
        proc_id=0,
        analysis_name="alma",
        resource_id=None,
        callback_url=None,
        osdf_staged=list(staged),
    )


CFG = {"alma": {"osdf": {"keypair_path": "/x/s.pem", "pelican_path": "pel"}}}


def test_every_staged_object_is_deleted(monkeypatch):
    gone = []
    monkeypatch.setattr("osdf.delete_object", lambda url, **kw: gone.append(url))
    rec = _rec(["osdf:///ns/a.tar", "osdf:///ns/b.tar"])
    assert main._cleanup_osdf_inputs(rec, CFG) is True
    assert gone == ["osdf:///ns/a.tar", "osdf:///ns/b.tar"]
    assert rec.osdf_staged == []


def test_one_failure_does_not_strand_the_others(monkeypatch):
    def flaky(url, **kw):
        if url.endswith("a.tar"):
            raise RuntimeError("boom")

    monkeypatch.setattr("osdf.delete_object", flaky)
    rec = _rec(["osdf:///ns/a.tar", "osdf:///ns/b.tar"])
    assert main._cleanup_osdf_inputs(rec, CFG) is False
    # the one that failed is kept for the next poll, the other is not retried
    assert rec.osdf_staged == ["osdf:///ns/a.tar"]


def test_a_delete_that_keeps_failing_is_given_up_on(monkeypatch):
    monkeypatch.setattr(
        "osdf.delete_object", lambda url, **kw: (_ for _ in ()).throw(RuntimeError("no"))
    )
    rec = _rec(["osdf:///ns/a.tar"])
    for _ in range(main.OSDF_CLEANUP_MAX_ATTEMPTS - 1):
        assert main._cleanup_osdf_inputs(rec, CFG) is False
    # the last attempt reports done so the poller stops retrying it forever
    assert main._cleanup_osdf_inputs(rec, CFG) is True


def test_nothing_staged_needs_no_client(monkeypatch):
    monkeypatch.setattr(
        "osdf.delete_object", lambda url, **kw: pytest.fail("called with nothing staged")
    )
    assert main._cleanup_osdf_inputs(_rec([]), CFG) is True


def test_deletion_can_be_switched_off(monkeypatch):
    monkeypatch.setattr("osdf.delete_object", lambda url, **kw: pytest.fail("deleted anyway"))
    cfg = {"alma": {"osdf": {"keypair_path": "/x/s.pem", "delete_after_job": False}}}
    rec = _rec(["osdf:///ns/a.tar"])
    assert main._cleanup_osdf_inputs(rec, cfg) is True
    assert rec.osdf_staged == ["osdf:///ns/a.tar"]


def test_a_restart_still_knows_what_to_clean():
    # The list round-trips through the schedd, so a job submitted before a
    # restart is not left holding gigabytes nobody remembers.
    main.JOBS.clear()
    main._adopt_ad(
        {
            "ClusterId": 7,
            "ProcId": 0,
            "JobStatus": 2,
            "SkyPortalAnalysisName": "alma",
            "SkyPortalCallback": "http://x/cb",
            "SkyPortalOsdfStaged": "osdf:///ns/a.tar osdf:///ns/b.tar",
        }
    )
    assert main.JOBS[(7, 0)].osdf_staged == ["osdf:///ns/a.tar", "osdf:///ns/b.tar"]
    main.JOBS.clear()
