"""
osg-skyportal-plugin: submit and track HTCondor/OSG jobs on behalf of SkyPortal.

This service speaks the SkyPortal AnalysisService webhook contract: SkyPortal
POSTs {inputs, callback_url, callback_method, resource_id} to
/analysis/<name>; we submit a Condor job, poll until done, then POST results
back to callback_url. Trivial sleep job by default; pluggable per-analysis spec
configurable later.

Contract reference: skyportal PR #3199 (closed) services/nmma_analysis_service/app.py.
"""

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import tornado.web
from baselayer.app.env import load_env
from baselayer.log import make_log

import osdf

log = make_log("osg")


# JobStatus integer encoding per the HTCondor classads documentation.
CONDOR_STATUS = {
    0: "unexpanded",
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring_output",
    7: "suspended",
}

TERMINAL = {"completed", "removed"}


@dataclass
class JobRecord:
    cluster_id: int
    analysis_name: str
    resource_id: str | None
    callback_url: str | None
    callback_method: str = "POST"
    submitted_at: float = field(default_factory=time.time)
    status: str = "idle"
    last_polled_at: float | None = None
    completed_at: float | None = None
    hold_reason: str | None = None
    callback_posted: bool = False
    osdf_output_url: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)


# In-memory job table for the spike. Rebuilt from the schedd at startup; see rehydrate_jobs.
JOBS: dict[int, JobRecord] = {}


def check_caps(cfg: dict, analysis_name: str, resource_id: str | None) -> str | None:
    """Return None if the request fits within configured caps, else a reason string."""
    caps = cfg.get("caps") or {}
    open_jobs = [r for r in JOBS.values() if r.completed_at is None]
    total_cap = caps.get("max_concurrent_total")
    if total_cap and len(open_jobs) >= total_cap:
        return f"global cap: already {len(open_jobs)} of {total_cap} concurrent jobs"
    per_analysis_cap = caps.get("max_concurrent_per_analysis")
    if per_analysis_cap:
        same = [r for r in open_jobs if r.analysis_name == analysis_name]
        if len(same) >= per_analysis_cap:
            return (
                f"per-analysis cap for {analysis_name}: "
                f"already {len(same)} of {per_analysis_cap} concurrent"
            )
    per_resource_cap = caps.get("max_concurrent_per_resource_per_analysis")
    if per_resource_cap and resource_id:
        same = [
            r
            for r in open_jobs
            if r.analysis_name == analysis_name and r.resource_id == resource_id
        ]
        if len(same) >= per_resource_cap:
            return (
                f"per-resource cap for {analysis_name} on {resource_id}: "
                f"already {len(same)} of {per_resource_cap} concurrent"
            )
    return None


def get_schedd(cfg: dict):
    """Connect to the configured Condor schedd. SciToken via BEARER_TOKEN_FILE."""
    import htcondor

    token_path = os.path.expanduser(cfg["htcondor"]["scitoken_path"])
    if os.path.exists(token_path):
        os.environ["BEARER_TOKEN_FILE"] = token_path
    else:
        log(f"warning: SciToken not found at {token_path}; using whatever creds htcondor finds")

    collector_host = cfg["htcondor"].get("collector")
    schedd_name = cfg["htcondor"].get("schedd")
    if collector_host is None:
        return htcondor.Schedd()
    collector = htcondor.Collector(collector_host)
    ad = (
        collector.locate(htcondor.DaemonTypes.Schedd, schedd_name)
        if schedd_name
        else collector.locate(htcondor.DaemonTypes.Schedd)
    )
    return htcondor.Schedd(ad)


def _stage_wrapper_job(
    cfg: dict, params: dict, inputs: dict, cluster_uuid: str
) -> tuple[dict, str | None]:
    """Build submit overrides + an OSDF output URL when wrapper-mode is requested."""
    if not params.get("use_wrapper"):
        return {}, None

    staging_root = Path(cfg.get("staging_dir", "staging"))
    job_dir = staging_root / cluster_uuid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "inputs.json").write_text(json.dumps(inputs))
    wrapper_src = Path(__file__).parent / "nmma_wrapper.py"

    output_url = None
    osdf_cfg = cfg.get("osdf") or {}
    prefix = osdf_cfg.get("output_prefix")
    if prefix:
        output_url = prefix.rstrip("/") + f"/{cluster_uuid}.json"

    env_parts = []
    if output_url:
        env_parts.append(f"OSDF_OUTPUT_URL={output_url}")
    if osdf_cfg.get("write_token_path"):
        env_parts.append(f"BEARER_TOKEN_FILE={osdf_cfg['write_token_path']}")

    overrides = {
        "executable": "/usr/bin/python3",
        "arguments": "nmma_wrapper.py",
        "transfer_input_files": f"{wrapper_src},{job_dir}/inputs.json",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "initialdir": str(job_dir),
    }
    if env_parts:
        overrides["environment"] = '"' + " ".join(env_parts) + '"'
    return overrides, output_url


def submit_job(
    cfg: dict,
    analysis_name: str,
    resource_id: str | None,
    callback_url: str | None,
    callback_method: str,
    inputs: dict[str, Any],
) -> int:
    """Submit one job. `inputs` may carry `analysis_parameters` and the SkyPortal-staged CSVs."""
    import htcondor

    defaults = cfg["defaults"]
    schedd = get_schedd(cfg)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    params = inputs.get("analysis_parameters", {}) or {}
    cluster_uuid = uuid.uuid4().hex
    wrapper_overrides, osdf_output_url = _stage_wrapper_job(cfg, params, inputs, cluster_uuid)

    submit_desc: dict[str, str] = {
        "executable": params.get("executable", "/bin/sleep"),
        "arguments": params.get("arguments", "60"),
        "output": "logs/job.$(ClusterId).$(ProcId).out",
        "error": "logs/job.$(ClusterId).$(ProcId).err",
        "log": "logs/job.$(ClusterId).log",
        "request_cpus": str(params.get("request_cpus", defaults["request_cpus"])),
        "request_memory": str(params.get("request_memory", defaults["request_memory"])),
        "request_disk": str(params.get("request_disk", defaults["request_disk"])),
        "+ProjectName": f'"{cfg["htcondor"]["project_name"]}"',
        "requirements": '(OSG == True) && (Arch == "X86_64")',
    }
    submit_desc["+MaxRuntime"] = str(
        int(params.get("max_runtime_seconds", defaults["max_runtime_seconds"]))
    )
    if defaults.get("singularity_image"):
        submit_desc["+SingularityImage"] = f'"{defaults["singularity_image"]}"'
    submit_desc.update(wrapper_overrides)
    # Round-trip the SkyPortal binding through the schedd so we can rehydrate after restart.
    submit_desc["+SkyPortalAnalysisName"] = f'"{analysis_name}"'
    submit_desc["+SkyPortalCallback"] = f'"{callback_url}"' if callback_url else '""'
    submit_desc["+SkyPortalCallbackMethod"] = f'"{callback_method}"'
    submit_desc["+SkyPortalResourceId"] = f'"{resource_id}"' if resource_id else '""'
    if osdf_output_url:
        submit_desc["+SkyPortalOsdfOutput"] = f'"{osdf_output_url}"'

    sub = htcondor.Submit(submit_desc)
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn, count=1)

    JOBS[cluster_id] = JobRecord(
        cluster_id=cluster_id,
        analysis_name=analysis_name,
        resource_id=resource_id,
        callback_url=callback_url,
        callback_method=callback_method,
        osdf_output_url=osdf_output_url,
        inputs=inputs,
    )
    log(f"submitted cluster_id={cluster_id} analysis={analysis_name} resource_id={resource_id}")
    return cluster_id


# Custom ClassAds we stamp onto every submitted job so the schedd is our truth.
_SP_AD_PROJECTION = [
    "ClusterId",
    "JobStatus",
    "HoldReason",
    "QDate",
    "CompletionDate",
    "SkyPortalAnalysisName",
    "SkyPortalCallback",
    "SkyPortalCallbackMethod",
    "SkyPortalResourceId",
    "SkyPortalOsdfOutput",
]


def _adopt_ad(ad: dict, from_history: bool = False) -> None:
    """Reconstruct a JobRecord from one schedd/history ClassAd, if we don't already have it."""
    cid = int(ad["ClusterId"])
    if cid in JOBS:
        return
    rec = JobRecord(
        cluster_id=cid,
        analysis_name=ad.get("SkyPortalAnalysisName") or "unknown",
        resource_id=ad.get("SkyPortalResourceId") or None,
        callback_url=ad.get("SkyPortalCallback") or None,
        callback_method=ad.get("SkyPortalCallbackMethod") or "POST",
        submitted_at=float(ad.get("QDate", time.time())),
        osdf_output_url=ad.get("SkyPortalOsdfOutput") or None,
    )
    if from_history:
        rec.status = CONDOR_STATUS.get(int(ad["JobStatus"]), "completed")
        rec.completed_at = float(ad.get("CompletionDate") or time.time())
    else:
        rec.status = CONDOR_STATUS.get(int(ad["JobStatus"]), "idle")
        rec.hold_reason = ad.get("HoldReason")
    JOBS[cid] = rec


def rehydrate_jobs(cfg: dict, history_hours: float = 24.0) -> int:
    """Repopulate `JOBS` from the schedd on startup. Returns count adopted."""
    schedd = get_schedd(cfg)
    project = cfg["htcondor"]["project_name"]
    constraint = f'ProjectName == "{project}" && SkyPortalCallback isnt ""'
    before = len(JOBS)
    for ad in schedd.query(constraint=constraint, projection=_SP_AD_PROJECTION):
        _adopt_ad(ad)
    cutoff = time.time() - history_hours * 3600
    history_constraint = f"{constraint} && CompletionDate > {cutoff:.0f}"
    try:
        for ad in schedd.history(
            constraint=history_constraint,
            projection=_SP_AD_PROJECTION,
            match=1000,
        ):
            _adopt_ad(ad, from_history=True)
    except Exception as e:  # noqa: BLE001 — history is best-effort
        log(f"rehydrate: history sweep failed: {e!r}")
    adopted = len(JOBS) - before
    log(f"rehydrated {adopted} job(s) from schedd")
    return adopted


def collect_outputs(rec: JobRecord) -> dict[str, Any]:
    """Stdout/stderr fallback when no OSDF bundle exists."""
    stdout_path = Path(f"logs/job.{rec.cluster_id}.0.out")
    stderr_path = Path(f"logs/job.{rec.cluster_id}.0.err")
    payload: dict[str, Any] = {}
    if stdout_path.exists():
        payload["results"] = {
            "format": "text",
            "data": base64.b64encode(stdout_path.read_bytes()).decode(),
        }
    if stderr_path.exists():
        payload["log"] = stderr_path.read_text(errors="replace")[-4096:]
    return payload


def fetch_osdf_bundle(rec: JobRecord, cfg: dict) -> dict | None:
    """Pull the wrapper's pre-built SkyPortal bundle from OSDF if it exists."""
    if not rec.osdf_output_url:
        return None
    token_path = (cfg.get("osdf") or {}).get("read_token_path") or cfg["htcondor"].get(
        "scitoken_path"
    )
    try:
        local = Path(f"logs/bundle.{rec.cluster_id}.json")
        osdf.download(rec.osdf_output_url, local, token_path=token_path)
        return json.loads(local.read_text())
    except Exception as e:  # noqa: BLE001 — fall back to stdout if OSDF unreachable
        log(f"OSDF fetch failed for cluster_id={rec.cluster_id}: {e!r}")
        return None


def build_callback_body(rec: JobRecord, cfg: dict | None = None) -> dict:
    """Construct the SkyPortal-shaped body to POST back to callback_url."""
    bundle = fetch_osdf_bundle(rec, cfg) if cfg is not None else None
    if bundle and isinstance(bundle, dict) and bundle.get("status"):
        return bundle
    is_success = rec.status == "completed" and rec.hold_reason is None
    return {
        "status": "success" if is_success else "failure",
        "message": (
            f"OSG cluster_id={rec.cluster_id} completed"
            if is_success
            else (rec.hold_reason or f"OSG job ended in state {rec.status}")
        ),
        "analysis": collect_outputs(rec) if is_success else {},
    }


def post_callback(rec: JobRecord, cfg: dict | None = None) -> bool:
    """POST the SkyPortal-shaped result to rec.callback_url. Return True on send."""
    if not rec.callback_url or rec.callback_method.upper() != "POST":
        return False
    body = build_callback_body(rec, cfg)
    try:
        requests.post(rec.callback_url, json=body, timeout=30)
        return True
    except requests.RequestException as e:
        log(f"callback post to {rec.callback_url} failed: {e!r}")
        return False


def poll_once(cfg: dict) -> None:
    """One sweep of the schedd + history for jobs we own; post callbacks for newly-terminal jobs."""
    if not JOBS:
        return
    schedd = get_schedd(cfg)
    cluster_ids = sorted(c for c, r in JOBS.items() if r.completed_at is None)
    if not cluster_ids:
        return
    constraint = " || ".join(f"ClusterId == {c}" for c in cluster_ids)

    live_ads = schedd.query(
        constraint=constraint,
        projection=["ClusterId", "JobStatus", "HoldReason", "HoldReasonCode"],
    )
    seen_live = set()
    for ad in live_ads:
        cid = int(ad["ClusterId"])
        seen_live.add(cid)
        rec = JOBS[cid]
        rec.status = CONDOR_STATUS.get(int(ad["JobStatus"]), "unknown")
        rec.last_polled_at = time.time()
        rec.hold_reason = ad.get("HoldReason")

    # Jobs not in the live queue are in history (completed) or vanished (removed).
    for cid in (c for c in cluster_ids if c not in seen_live):
        rec = JOBS[cid]
        rec.last_polled_at = time.time()
        history_ads = list(
            schedd.history(
                constraint=f"ClusterId == {cid}",
                projection=["ClusterId", "JobStatus", "ExitCode", "CompletionDate"],
                match=1,
            )
        )
        if history_ads:
            ad = history_ads[0]
            rec.status = CONDOR_STATUS.get(int(ad["JobStatus"]), "completed")
            rec.completed_at = float(ad.get("CompletionDate") or time.time())
        else:
            rec.status = "removed"
            rec.completed_at = time.time()

    for cid in cluster_ids:
        rec = JOBS[cid]
        if rec.status in TERMINAL and not rec.callback_posted:
            rec.callback_posted = post_callback(rec, cfg)


async def poller_loop(cfg: dict):
    interval = float(cfg["poller"]["interval_seconds"])
    while True:
        try:
            poll_once(cfg)
        except Exception as e:  # noqa: BLE001 — poller must outlive any single error
            log(f"poller error: {e!r}")
        await asyncio.sleep(interval)


def _check_bearer(handler: tornado.web.RequestHandler, expected_token: str | None) -> bool:
    if not expected_token:
        return True
    auth = handler.request.headers.get("Authorization", "")
    return auth == f"Bearer {expected_token}"


class AnalysisHandler(tornado.web.RequestHandler):
    """SkyPortal AnalysisService entrypoint at /analysis/<analysis_name>."""

    def initialize(self, cfg: dict):
        self.cfg = cfg

    def get(self, analysis_name: str):
        self.write({"status": "active", "analysis": analysis_name})

    def post(self, analysis_name: str):
        expected = self.cfg.get("auth", {}).get("incoming_bearer_token")
        if not _check_bearer(self, expected):
            self.set_status(401)
            self.write({"error": "missing or wrong Authorization bearer token"})
            return

        try:
            data = json.loads(self.request.body or b"{}")
        except json.JSONDecodeError as e:
            self.set_status(400)
            self.write({"error": f"invalid json: {e}"})
            return

        missing = [k for k in ("inputs", "callback_url", "callback_method") if k not in data]
        if missing:
            self.set_status(400)
            self.write({"error": f"missing required keys: {missing}"})
            return

        cap_reason = check_caps(self.cfg, analysis_name, data.get("resource_id"))
        if cap_reason:
            self.set_status(429)
            self.set_header("Retry-After", "60")
            self.write({"error": "rate-limited", "reason": cap_reason})
            return

        try:
            cluster_id = submit_job(
                self.cfg,
                analysis_name=analysis_name,
                resource_id=data.get("resource_id"),
                callback_url=data["callback_url"],
                callback_method=data["callback_method"],
                inputs=data["inputs"],
            )
        except Exception as e:  # noqa: BLE001 — surface schedd errors to caller
            log(f"submit error: {e!r}")
            self.set_status(500)
            self.write({"error": str(e)})
            return

        self.write({"status": "pending", "cluster_id": cluster_id})


class StatusHandler(tornado.web.RequestHandler):
    def get(self, cluster_id: str):
        rec = JOBS.get(int(cluster_id))
        if rec is None:
            self.set_status(404)
            self.write({"error": f"unknown cluster_id {cluster_id}"})
            return
        self.write(asdict(rec))


class ListHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"jobs": [asdict(r) for r in JOBS.values()]})


def build_app(cfg: dict) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/analysis/([A-Za-z0-9_\-]+)", AnalysisHandler, {"cfg": cfg}),
            (r"/jobs/(\d+)", StatusHandler),
            (r"/jobs", ListHandler),
        ]
    )


def load_plugin_config() -> dict:
    _, app_cfg = load_env()
    return app_cfg["services.external.osg.params"]


async def amain():
    cfg = load_plugin_config()
    try:
        rehydrate_jobs(cfg)
    except Exception as e:  # noqa: BLE001 — startup must continue even if schedd is briefly down
        log(f"rehydrate failed (continuing with empty JOBS): {e!r}")
    app = build_app(cfg)
    app.listen(int(cfg["listener"]["port"]), address=cfg["listener"]["host"])
    log(f"listening on {cfg['listener']['host']}:{cfg['listener']['port']}")
    asyncio.create_task(poller_loop(cfg))
    await asyncio.Event().wait()


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
