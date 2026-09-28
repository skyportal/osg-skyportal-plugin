"""
Fetch an object's reference cutout from BOOM (via SkyPortal's broker route) and
stage it for an ORACLE job, so the worker — which can't reach BOOM — runs the full
omni model. Runs on the listener at submit time, like aframe_staging.

The route (``/api/brokers/{id}/alerts/{candid}/cutouts``) holds the BOOM token and
scopes by program, so we call it with the plugin's SkyPortal token. We fetch the
alert list for the object, take the latest candid, pull its cutouts, and write the
reference (template) as the gzipped FITS bytes oracle_bridge expects. Anything
missing (no broker configured, no alert, no cutout) is non-fatal: the job then runs
without the image, same as the BOOM path with an absent template.
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests

# Must match oracle_bridge.CUTOUT_FILE (kept local to avoid importing the bridge,
# and its CUDA env side effect, into the listener).
CUTOUT_FILE = "oracle_cutout.fits.gz"

# BOOM cutout keys for the reference image, most-specific first.
TEMPLATE_KEYS = ("cutoutTemplate", "template", "cutoutReference", "reference")


def _gzip_fits_bytes(field) -> bytes | None:
    """The raw gzipped-FITS bytes from a BOOM cutout field (base64 string, Avro
    ``stampData``, or Mongo ``$binary``)."""
    if field is None:
        return None
    if isinstance(field, dict):
        if "$binary" in field:
            return base64.b64decode(field["$binary"]["base64"])
        sd = field.get("stampData")
        if sd is None:
            return None
        return base64.b64decode(sd) if isinstance(sd, str) else bytes(sd)
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    if isinstance(field, str):
        return base64.b64decode(field)
    return None


def _latest_candid(alerts) -> str | None:
    """The candid of the most recent alert (by candidate jd, else the candid)."""
    if not isinstance(alerts, list) or not alerts:
        return None

    def jd(a):
        cand = a.get("candidate") or {}
        return cand.get("jd") or a.get("jd") or 0

    best = max(alerts, key=jd)
    return best.get("candid") or best.get("_id")


def stage_cutout(cfg: dict, inputs: dict, job_dir: Path, log=print) -> list[Path]:
    """Write the reference cutout into job_dir; return [path] or [] if unavailable."""
    sky = cfg.get("skyportal") or {}
    base = str(sky.get("base_url") or "").rstrip("/")
    token = sky.get("api_token")
    broker_id = (cfg.get("oracle") or {}).get("broker_id")
    if not base or not token or token.startswith("replace_with") or not broker_id:
        log("oracle staging: skyportal.base_url/api_token or oracle.broker_id unset; no cutout")
        return []

    obj = inputs.get("obj") or {}
    obj_id = obj.get("id") or inputs.get("resource_id")
    if not obj_id:
        log("oracle staging: no obj id in inputs; no cutout")
        return []

    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(
            f"{base}/api/brokers/{broker_id}/alerts",
            params={"objectId": obj_id},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        candid = _latest_candid((r.json() or {}).get("data"))
        if candid is None:
            log(f"oracle staging: no BOOM alert for {obj_id}; no cutout")
            return []

        r = requests.get(
            f"{base}/api/brokers/{broker_id}/alerts/{candid}/cutouts",
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        data = (r.json() or {}).get("data") or {}
        field = next((data[k] for k in TEMPLATE_KEYS if data.get(k) is not None), None)
        raw = _gzip_fits_bytes(field)
        if not raw:
            log(f"oracle staging: no reference cutout for {obj_id} (candid {candid})")
            return []
    except Exception as e:  # noqa: BLE001 — cutout is optional; the job runs without it
        log(f"oracle staging: cutout fetch failed for {obj_id}: {e!r}")
        return []

    out = Path(job_dir) / CUTOUT_FILE
    out.write_bytes(raw)
    log(f"oracle staging: staged reference cutout for {obj_id} (candid {candid})")
    return [out]
