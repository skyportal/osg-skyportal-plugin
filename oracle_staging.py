"""
Fetch an object's reference cutout from BOOM and stage it for an ORACLE job, so
the worker — which can't reach BOOM — runs the full omni model. Runs on the
listener at submit time, like aframe_staging.

The listener is an in-pod SkyPortal service, so it goes straight through the ORM:
open a DB session, find the active BOOM broker, resolve the object's latest candid
and pull its cutouts via the broker class. The broker carries its own BOOM
credentials (Broker.altdata), so no token of ours is involved; ``permissions=None``
marks this a trusted in-app call. The reference (template) cutout is written as the
gzipped FITS bytes oracle_bridge expects.

Everything here is best-effort: run standalone (no SkyPortal/DB) or with no alert,
broker or cutout and it returns [] — the job then runs without the image, same as
the BOOM path with an absent template.
"""

from __future__ import annotations

import base64
from pathlib import Path

# Must match oracle_bridge.CUTOUT_FILE (kept local to avoid importing the bridge,
# and its CUDA env side effect, into the listener).
CUTOUT_FILE = "oracle_cutout.fits.gz"

# BOOM cutout keys for the reference image, most-specific first.
TEMPLATE_KEYS = ("cutoutTemplate", "template", "cutoutReference", "reference")

_DB_INITED = False


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


def _ensure_db(cfg: dict) -> bool:
    """Initialise the DB connection once (this service's own, like any SkyPortal
    microservice). False if SkyPortal isn't importable — i.e. running standalone."""
    global _DB_INITED
    if _DB_INITED:
        return True
    from baselayer.app.models import init_db

    init_db(**cfg["database"])
    _DB_INITED = True
    return True


def stage_cutout(cfg: dict, inputs: dict, job_dir: Path, log=print) -> list[Path]:
    """Write the reference cutout into job_dir; return [path] or [] if unavailable."""
    obj = inputs.get("obj") or {}
    obj_id = obj.get("id") or inputs.get("resource_id")
    if not obj_id:
        log("oracle staging: no obj id in inputs; no cutout")
        return []

    try:
        import sqlalchemy as sa

        _ensure_db(cfg)
        from baselayer.app.models import DBSession
        from skyportal.models import Broker

        with DBSession() as session:
            broker = session.scalars(
                sa.select(Broker)
                .where(Broker.broker_classname == "BOOMBROKER", Broker.active.is_(True))
                .order_by(Broker.default_alert_search.desc(), Broker.id)
            ).first()
            if broker is None:
                log("oracle staging: no active BOOM broker; no cutout")
                return []

            alerts = broker.broker_class.query_alerts(
                broker, session, objectId=obj_id, permissions=None
            )
            candid = _latest_candid(alerts)
            if candid is None:
                log(f"oracle staging: no BOOM alert for {obj_id}; no cutout")
                return []

            cutouts = broker.broker_class.get_cutouts(broker, candid, session, permissions=None)
            field = next((cutouts[k] for k in TEMPLATE_KEYS if cutouts.get(k) is not None), None)
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
