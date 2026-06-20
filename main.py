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
import functools
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import tornado.web
from baselayer.app.env import load_env
from baselayer.log import make_log

import osdf

log = make_log("osg")

# Condor submits block ~1-2s each; run them off the event loop in a small pool
# so a single replica can accept and dispatch many concurrent submits instead of
# serializing them (which made the app's start request time out under bursts).
_SUBMIT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="osg-submit")


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
    # proc_id is 0 for single submits; batched (itemdata) submits put several
    # jobs under one cluster as proc 0..N-1, so every job is keyed (cluster, proc).
    proc_id: int = 0
    submitted_at: float = field(default_factory=time.time)
    status: str = "idle"
    last_polled_at: float | None = None
    completed_at: float | None = None
    hold_reason: str | None = None
    callback_posted: bool = False
    osdf_output_url: str | None = None
    spooled: bool = False
    inputs: dict[str, Any] = field(default_factory=dict)


# In-memory job table, keyed (cluster_id, proc_id). Rebuilt from the schedd at
# startup; see rehydrate_jobs.
JOBS: dict[tuple[int, int], JobRecord] = {}


def _key(rec: "JobRecord") -> tuple[int, int]:
    return (rec.cluster_id, rec.proc_id)


# Pending-submit queue for batch mode: (item, future) tuples drained by the
# flusher, which coalesces them into one itemdata RPC. Created in amain().
_BATCH_QUEUE: "asyncio.Queue | None" = None


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


def _htcondor():
    """Lazy, version-tolerant import: v1 on Linux APs, v2 (htcondor2) on macOS/conda."""
    try:
        import htcondor
    except ModuleNotFoundError:
        import htcondor2 as htcondor
    return htcondor


def _commit_submit(schedd, sub, spool: bool) -> int:
    """Queue one job, returning its cluster id. htcondor2 has no transaction()."""
    if hasattr(schedd, "transaction"):  # htcondor v1 / test fake
        with schedd.transaction() as txn:
            return sub.queue(txn, count=1)
    result = schedd.submit(sub, count=1, spool=spool)  # htcondor2
    if spool:
        schedd.spool(result)  # push the input sandbox to the AP
    return result.cluster()


def get_schedd(cfg: dict):
    """Connect to the configured Condor schedd. SciToken via BEARER_TOKEN_FILE."""
    htcondor = _htcondor()

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


def ensure_keepalive(cfg: dict) -> None:
    """Park a placeholder job on the AP, held indefinitely, so the credmon keeps
    the SciToken fresh for remote submission (OSG workaround, support ticket).

    Idempotent — no-op if our keepalive job is already queued. Only matters for
    remote submission; harmless when the plugin runs on the AP. Gated on
    ``htcondor.keepalive`` so on-AP deployments needn't park a job.
    """
    if not cfg["htcondor"].get("keepalive", False):
        return
    schedd = get_schedd(cfg)
    existing = list(schedd.query(constraint="SkyPortalKeepalive == true", projection=["ClusterId"]))
    if existing:
        log(f"keepalive job already queued (cluster {existing[0]['ClusterId']})")
        return
    htcondor = _htcondor()
    # /bin/true, held on submit: it never runs, it just keeps a job in the queue
    # referencing the scitokens credential so the credmon keeps refreshing it.
    sub = htcondor.Submit(
        {
            "executable": "/bin/true",
            "transfer_executable": "False",
            "hold": "true",
            "+SkyPortalKeepalive": "true",
            "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
            "request_cpus": "1",
            "request_memory": "16MB",
            "request_disk": "16MB",
            "+ProjectName": f'"{cfg["htcondor"]["project_name"]}"',
        }
    )
    cluster = _commit_submit(schedd, sub, spool=False)
    log(f"submitted held keepalive job (cluster {cluster})")


def _stage_wrapper_job(
    cfg: dict, params: dict, inputs: dict, cluster_uuid: str
) -> tuple[dict, str | None]:
    """Build submit overrides + an OSDF output URL when wrapper-mode is requested."""
    # Wrapper mode can be forced per-service via config (an NMMA service always
    # wraps) or requested per-job via analysis_parameters.
    use_wrapper = params.get("use_wrapper", cfg.get("defaults", {}).get("use_wrapper", False))
    if not use_wrapper:
        return {}, None

    # Absolute paths: with spool, HTCondor resolves relative transfer paths
    # against Iwd, which doubles them (staging/uuid/staging/uuid/...).
    staging_root = Path(cfg.get("staging_dir", "staging")).resolve()
    job_dir = staging_root / cluster_uuid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "inputs.json").write_text(json.dumps(inputs))
    plugin_dir = Path(__file__).parent
    wrapper_src = (plugin_dir / "fiesta_wrapper.py").resolve()
    bridge_src = (plugin_dir / "fiesta_bridge.py").resolve()

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

    # No initialdir: let Iwd default to the plugin cwd so spooled output files
    # (basenames) are retrieved next to where the poller reads them.
    overrides = {
        "executable": "/usr/bin/python3",
        "arguments": "fiesta_wrapper.py",
        "transfer_input_files": f"{wrapper_src},{bridge_src},{job_dir / 'inputs.json'}",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
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
    htcondor = _htcondor()

    defaults = cfg["defaults"]
    schedd = get_schedd(cfg)

    params = inputs.get("analysis_parameters", {}) or {}
    cluster_uuid = uuid.uuid4().hex
    wrapper_overrides, osdf_output_url = _stage_wrapper_job(cfg, params, inputs, cluster_uuid)

    submit_desc: dict[str, str] = {
        "executable": params.get("executable", "/bin/sleep"),
        # The executable lives on the worker / in the container; don't ship this
        # host's copy (wrong arch/OS → ExitCode 126).
        "transfer_executable": "False",
        "arguments": params.get("arguments", "60"),
        # Basenames (no subdir): spool retrieves them to Iwd (cwd), where the
        # poller reads them. No `log =`: the schedd event log must live in the
        # AP home dir for remote OSPool submission, and the plugin doesn't use it.
        "output": "job.$(ClusterId).$(ProcId).out",
        "error": "job.$(ClusterId).$(ProcId).err",
        "request_cpus": str(params.get("request_cpus", defaults["request_cpus"])),
        # Config values are MB; give explicit units since a bare RequestDisk is
        # KiB in HTCondor (RequestMemory is MiB) — easy to get wrong.
        "request_memory": f"{params.get('request_memory', defaults['request_memory'])}MB",
        "request_disk": f"{params.get('request_disk', defaults['request_disk'])}MB",
        "+ProjectName": f'"{cfg["htcondor"]["project_name"]}"',
        # Target OSPool's Linux/x86_64 glideins. Bindings submission from a
        # non-Linux host otherwise defaults requirements to the local platform.
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
    }
    # jaxlib (used by the fiesta surrogate models) is built with AVX, so it
    # SIGILLs on pre-AVX glideins. Require the CPU advertise AVX: ~99% of OSPool
    # slots set has_avx (vs 87% at Microarch>=x86_64-v3, which needlessly drops
    # older AVX-only CPUs jaxlib runs on). Slots not advertising it evaluate
    # UNDEFINED and are skipped — safe by design. Tunable via
    # defaults.cpu_requirements (set "" to disable) or per-request.
    cpu_requirements = params.get(
        "cpu_requirements", defaults.get("cpu_requirements", "(has_avx == True)")
    )
    if cpu_requirements:
        submit_desc["requirements"] += f" && {cpu_requirements}"
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

    # Remote AP submission must spool: otherwise Iwd defaults to this host's cwd
    # (nonexistent on the AP → held) and input files never reach the sandbox.
    # Set htcondor.spool=false only when the plugin runs co-located on the AP.
    needs_spool = bool(cfg["htcondor"].get("spool", True))
    sub = htcondor.Submit(submit_desc)
    cluster_id = _commit_submit(schedd, sub, spool=needs_spool)

    JOBS[(cluster_id, 0)] = JobRecord(
        cluster_id=cluster_id,
        proc_id=0,
        analysis_name=analysis_name,
        resource_id=resource_id,
        callback_url=callback_url,
        callback_method=callback_method,
        osdf_output_url=osdf_output_url,
        spooled=needs_spool,
        inputs=inputs,
    )
    log(f"submitted cluster_id={cluster_id} analysis={analysis_name} resource_id={resource_id}")
    return cluster_id


def _submit_signature(cfg: dict, params: dict) -> tuple:
    """Submit-level knobs that must match for jobs to share one itemdata cluster
    (resources/requirements live on the cluster ad, not per-proc)."""
    d = cfg["defaults"]
    return (
        str(params.get("request_cpus", d["request_cpus"])),
        str(params.get("request_memory", d["request_memory"])),
        str(params.get("request_disk", d["request_disk"])),
        str(params.get("max_runtime_seconds", d["max_runtime_seconds"])),
        str(params.get("cpu_requirements", d.get("cpu_requirements", ""))),
        str(d.get("singularity_image", "")),
    )


def submit_jobs_batch(cfg: dict, items: list[dict]) -> list[tuple[int, int]]:
    """Submit a group of wrapper jobs in ONE schedd RPC via itemdata: one cluster,
    one proc per item, per-proc values supplied as macros. `items` share submit
    resources (the flusher groups by _submit_signature). Returns [(cluster, proc)].
    """
    htcondor = _htcondor()
    defaults = cfg["defaults"]
    schedd = get_schedd(cfg)
    plugin_dir = Path(__file__).parent
    wrapper_src = (plugin_dir / "fiesta_wrapper.py").resolve()
    bridge_src = (plugin_dir / "fiesta_bridge.py").resolve()
    staging_root = Path(cfg.get("staging_dir", "staging")).resolve()
    osdf_cfg = cfg.get("osdf") or {}
    out_prefix = osdf_cfg.get("output_prefix")

    # Submit-level knobs from the first item (the group shares them by signature).
    p0 = (items[0].get("inputs") or {}).get("analysis_parameters", {}) or {}
    submit_desc: dict[str, str] = {
        "executable": "/usr/bin/python3",
        "arguments": "fiesta_wrapper.py",
        "transfer_executable": "False",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "transfer_input_files": f"{wrapper_src},{bridge_src},$(inputs_json)",
        "output": "job.$(ClusterId).$(ProcId).out",
        "error": "job.$(ClusterId).$(ProcId).err",
        "request_cpus": str(p0.get("request_cpus", defaults["request_cpus"])),
        "request_memory": f"{p0.get('request_memory', defaults['request_memory'])}MB",
        "request_disk": f"{p0.get('request_disk', defaults['request_disk'])}MB",
        "+ProjectName": f'"{cfg["htcondor"]["project_name"]}"',
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+MaxRuntime": str(int(p0.get("max_runtime_seconds", defaults["max_runtime_seconds"]))),
        "+SkyPortalAnalysisName": '"$(sp_name)"',
        "+SkyPortalCallback": '"$(sp_cb)"',
        "+SkyPortalCallbackMethod": '"$(sp_cbm)"',
        "+SkyPortalResourceId": '"$(sp_rid)"',
        "+SkyPortalOsdfOutput": '"$(sp_osdf)"',
    }
    cpu_req = p0.get("cpu_requirements", defaults.get("cpu_requirements", "(has_avx == True)"))
    if cpu_req:
        submit_desc["requirements"] += f" && {cpu_req}"
    if defaults.get("singularity_image"):
        submit_desc["+SingularityImage"] = f'"{defaults["singularity_image"]}"'
    env_parts = []
    if out_prefix:
        env_parts.append("OSDF_OUTPUT_URL=$(sp_osdf)")
    if osdf_cfg.get("write_token_path"):
        env_parts.append(f"BEARER_TOKEN_FILE={osdf_cfg['write_token_path']}")
    if env_parts:
        submit_desc["environment"] = '"' + " ".join(env_parts) + '"'

    itemdata: list[dict] = []
    meta: list[tuple] = []
    for it in items:
        cluster_uuid = uuid.uuid4().hex
        job_dir = staging_root / cluster_uuid
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "inputs.json").write_text(json.dumps(it.get("inputs") or {}))
        osdf_url = (out_prefix.rstrip("/") + f"/{cluster_uuid}.json") if out_prefix else ""
        itemdata.append(
            {
                "inputs_json": str(job_dir / "inputs.json"),
                "sp_name": it["analysis_name"],
                "sp_cb": it.get("callback_url") or "",
                "sp_cbm": it.get("callback_method") or "POST",
                "sp_rid": it.get("resource_id") or "",
                "sp_osdf": osdf_url,
            }
        )
        meta.append(
            (
                it["analysis_name"],
                it.get("resource_id"),
                it.get("callback_url"),
                it.get("callback_method") or "POST",
                osdf_url or None,
                it.get("inputs") or {},
            )
        )

    needs_spool = bool(cfg["htcondor"].get("spool", True))
    sub = htcondor.Submit(submit_desc)
    # Default count=1 -> one job per itemdata row.
    result = schedd.submit(sub, itemdata=iter(itemdata), spool=needs_spool)
    cluster = int(result.cluster())
    if needs_spool:
        schedd.spool(result)

    keys: list[tuple[int, int]] = []
    for proc, (aname, rid, cb, cbm, osdf_url, inp) in enumerate(meta):
        JOBS[(cluster, proc)] = JobRecord(
            cluster_id=cluster,
            proc_id=proc,
            analysis_name=aname,
            resource_id=rid,
            callback_url=cb,
            callback_method=cbm,
            osdf_output_url=osdf_url,
            spooled=needs_spool,
            inputs=inp,
        )
        keys.append((cluster, proc))
    log(f"batch-submitted cluster_id={cluster} jobs={len(items)}")
    return keys


async def batch_flusher(cfg: dict) -> None:
    """Drain the submit queue, coalescing requests into itemdata batches. Waits up
    to window_seconds (or max_size) to gather a batch, groups by submit signature,
    and resolves each request's future with its (cluster, proc)."""
    bcfg = cfg.get("batch") or {}
    max_size = int(bcfg.get("max_size", 50))
    window = float(bcfg.get("window_seconds", 1.0))
    loop = asyncio.get_running_loop()
    assert _BATCH_QUEUE is not None
    while True:
        batch = [await _BATCH_QUEUE.get()]  # block for the first
        deadline = loop.time() + window
        while len(batch) < max_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(_BATCH_QUEUE.get(), remaining))
            except asyncio.TimeoutError:
                break
        groups: dict[tuple, list] = {}
        for item, fut in batch:
            sig = _submit_signature(
                cfg, (item.get("inputs") or {}).get("analysis_parameters", {}) or {}
            )
            groups.setdefault(sig, []).append((item, fut))
        for group in groups.values():
            items = [it for it, _f in group]
            futs = [f for _it, f in group]
            try:
                keys = await loop.run_in_executor(
                    _SUBMIT_POOL, functools.partial(submit_jobs_batch, cfg, items)
                )
                for key, fut in zip(keys, futs):
                    if not fut.done():
                        fut.set_result(key)
            except Exception as e:  # noqa: BLE001 — fail this group's requests, keep the loop alive
                log(f"batch submit failed ({len(items)} jobs): {e!r}")
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(e)


# Custom ClassAds we stamp onto every submitted job so the schedd is our truth.
_SP_AD_PROJECTION = [
    "ClusterId",
    "ProcId",
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
    pid = int(ad.get("ProcId", 0))
    if (cid, pid) in JOBS:
        return
    rec = JobRecord(
        cluster_id=cid,
        proc_id=pid,
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
    JOBS[_key(rec)] = rec


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


def _retrieve_outputs(schedd, rec: JobRecord) -> None:
    """Pull a spooled job's output sandbox back from the AP into logs/. No-op for the test fake."""
    if not rec.spooled or not hasattr(schedd, "retrieve"):
        return
    try:
        schedd.retrieve(f"ClusterId == {rec.cluster_id} && ProcId == {rec.proc_id}")
    except Exception as e:  # noqa: BLE001 — fall back to whatever's on disk
        log(f"retrieve failed for {rec.cluster_id}.{rec.proc_id}: {e!r}")


def collect_outputs(rec: JobRecord) -> dict[str, Any]:
    """Stdout/stderr fallback when no OSDF bundle exists."""
    stdout_path = Path(f"job.{rec.cluster_id}.{rec.proc_id}.out")
    stderr_path = Path(f"job.{rec.cluster_id}.{rec.proc_id}.err")
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
        local = Path(f"logs/bundle.{rec.cluster_id}.{rec.proc_id}.json")
        osdf.download(rec.osdf_output_url, local, token_path=token_path)
        return json.loads(local.read_text())
    except Exception as e:  # noqa: BLE001 — fall back to stdout if OSDF unreachable
        log(f"OSDF fetch failed for cluster_id={rec.cluster_id}: {e!r}")
        return None


def _read_stdout_bundle(rec: JobRecord) -> dict | None:
    """Parse the wrapper's SkyPortal bundle from its stdout (printed as the last JSON line)."""
    stdout_path = Path(f"job.{rec.cluster_id}.{rec.proc_id}.out")
    if not stdout_path.exists():
        return None
    for line in reversed(stdout_path.read_text(errors="replace").splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None  # last line isn't the bundle; let collect_outputs scrape it
        return obj if isinstance(obj, dict) and obj.get("status") else None
    return None


def build_callback_body(rec: JobRecord, cfg: dict | None = None) -> dict:
    """Construct the SkyPortal-shaped body to POST back to callback_url."""
    bundle = fetch_osdf_bundle(rec, cfg) if cfg is not None else None
    if not (bundle and isinstance(bundle, dict) and bundle.get("status")):
        bundle = _read_stdout_bundle(rec)  # spool/no-OSDF path: wrapper bundle is in stdout
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
    open_keys = [k for k, r in JOBS.items() if r.completed_at is None]
    if not open_keys:
        return
    # Query distinct clusters (a batched submit shares one ClusterId across many
    # procs), then match each ad back to its (cluster, proc) record.
    clusters = sorted({c for (c, _p) in open_keys})
    constraint = " || ".join(f"ClusterId == {c}" for c in clusters)

    live_ads = schedd.query(
        constraint=constraint,
        projection=["ClusterId", "ProcId", "JobStatus", "HoldReason", "HoldReasonCode"],
    )
    seen_live = set()
    for ad in live_ads:
        k = (int(ad["ClusterId"]), int(ad.get("ProcId", 0)))
        rec = JOBS.get(k)
        if rec is None:
            continue
        seen_live.add(k)
        rec.status = CONDOR_STATUS.get(int(ad["JobStatus"]), "unknown")
        rec.last_polled_at = time.time()
        rec.hold_reason = ad.get("HoldReason")

    # Jobs not in the live queue are in history (completed) or vanished (removed).
    for k in (k for k in open_keys if k not in seen_live):
        c, p = k
        rec = JOBS[k]
        rec.last_polled_at = time.time()
        history_ads = list(
            schedd.history(
                constraint=f"ClusterId == {c} && ProcId == {p}",
                projection=["ClusterId", "ProcId", "JobStatus", "ExitCode", "CompletionDate"],
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

    for k in open_keys:
        rec = JOBS[k]
        if rec.status in TERMINAL and not rec.callback_posted:
            if rec.status == "completed":
                _retrieve_outputs(schedd, rec)
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

    async def post(self, analysis_name: str):
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

        # Batch mode: hand the request to the flusher, which coalesces many into
        # one itemdata RPC. We await our own future so the response still carries
        # the assigned (cluster, proc) — it resolves within ~window_seconds.
        if (self.cfg.get("batch") or {}).get("enabled", False) and _BATCH_QUEUE is not None:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await _BATCH_QUEUE.put(
                (
                    {
                        "analysis_name": analysis_name,
                        "resource_id": data.get("resource_id"),
                        "callback_url": data["callback_url"],
                        "callback_method": data["callback_method"],
                        "inputs": data["inputs"],
                    },
                    fut,
                )
            )
            try:
                cluster_id, proc_id = await asyncio.wait_for(fut, timeout=25)
            except Exception as e:  # noqa: BLE001 — batch submit failed/slow
                log(f"batch submit error: {e!r}")
                self.set_status(500)
                self.write({"error": str(e)})
                return
            self.write({"status": "pending", "cluster_id": cluster_id, "proc_id": proc_id})
            return

        try:
            # Offload the blocking condor submit to a worker thread so the event
            # loop stays free to accept other requests (and keep the poller running).
            cluster_id = await asyncio.get_running_loop().run_in_executor(
                _SUBMIT_POOL,
                functools.partial(
                    submit_job,
                    self.cfg,
                    analysis_name=analysis_name,
                    resource_id=data.get("resource_id"),
                    callback_url=data["callback_url"],
                    callback_method=data["callback_method"],
                    inputs=data["inputs"],
                ),
            )
        except Exception as e:  # noqa: BLE001 — surface schedd errors to caller
            log(f"submit error: {e!r}")
            self.set_status(500)
            self.write({"error": str(e)})
            return

        self.write({"status": "pending", "cluster_id": cluster_id})


class StatusHandler(tornado.web.RequestHandler):
    def get(self, cluster_id: str):
        # A cluster may hold several procs (batched submit); return them all.
        recs = [r for (c, _p), r in JOBS.items() if c == int(cluster_id)]
        if not recs:
            self.set_status(404)
            self.write({"error": f"unknown cluster_id {cluster_id}"})
            return
        self.write({"jobs": [asdict(r) for r in recs]})


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
    try:
        ensure_keepalive(cfg)
    except Exception as e:  # noqa: BLE001 — keepalive is best-effort, never block startup
        log(f"keepalive setup failed (continuing): {e!r}")
    app = build_app(cfg)
    app.listen(int(cfg["listener"]["port"]), address=cfg["listener"]["host"])
    log(f"listening on {cfg['listener']['host']}:{cfg['listener']['port']}")
    asyncio.create_task(poller_loop(cfg))
    if (cfg.get("batch") or {}).get("enabled", False):
        global _BATCH_QUEUE
        _BATCH_QUEUE = asyncio.Queue()
        asyncio.create_task(batch_flusher(cfg))
        bcfg = cfg.get("batch") or {}
        log(
            f"batch submit enabled (max_size={bcfg.get('max_size', 50)}, "
            f"window={bcfg.get('window_seconds', 1.0)}s)"
        )
    await asyncio.Event().wait()


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
