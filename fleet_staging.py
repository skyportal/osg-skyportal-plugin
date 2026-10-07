"""
Fetch an object's host-galaxy data from BOOM and stage it for a FLEET job, so the
worker — which can't reach the archives FLEET would query — classifies offline.
Runs on the listener at submit time, like oracle_staging.

The listener is an in-pod SkyPortal service, so it goes straight through the ORM:
open a DB session, find the active BOOM broker, and read the object's alert aux
document (``cross_matches`` + ``host_galaxy``). BOOM exposes no API for
``host_galaxy``, so we query the ``*_alerts_aux`` collection directly via the
broker's own request helper (its credentials live on Broker.altdata, so no token
of ours is involved). The LSDR10 crossmatch carries the host photometry FLEET
needs (fluxes + half-light radius); ``host_galaxy`` carries the DLR association.

The raw BOOM data is written as ``fleet_host.json`` for fleet_bridge to turn into
FLEET's catalogue schema. Everything is best-effort: run standalone (no
SkyPortal/DB) or with no alert/host and it returns [] — the job then runs FLEET
hostless, same as when no host is found.
"""

from __future__ import annotations

import json
from pathlib import Path

# Must match fleet_bridge.HOST_FILE (kept local to avoid importing the bridge,
# and its FLEET dependency, into the listener).
HOST_FILE = "fleet_host.json"

# Catalogs whose rows FLEET uses for host features (LSDR10 = photometry + size;
# NED = angular size / distance as a fallback).
HOST_CATALOGS = ("LSDR10", "NED")

_DB_INITED = False


def _ensure_db() -> None:
    """Initialise the DB connection once (this service's own, like any SkyPortal
    microservice). Reads the full app config directly: the listener is handed only
    its own params block, which has no ``database`` key. Raises if SkyPortal isn't
    importable — i.e. running standalone — and the caller degrades to no host."""
    global _DB_INITED
    if _DB_INITED:
        return
    from baselayer.app.env import load_env
    from baselayer.app.models import init_db

    _, app_cfg = load_env()
    init_db(**app_cfg["database"])
    _DB_INITED = True


def _survey_suffix(obj_id: str) -> str:
    """BOOM aux collection prefix for the survey an objectId belongs to. ZTF is
    the only FLEET-relevant survey (LSDR10 host crossmatch), so default to it."""
    return "ZTF"


def stage_host(cfg: dict, inputs: dict, job_dir: Path, log=print) -> list[Path]:
    """Stage the BOOM host-galaxy data for a FLEET job; return [path] ([] if
    nothing available). Best effort: absence degrades the job to a hostless run."""
    obj = inputs.get("obj") or {}
    obj_id = obj.get("id") or inputs.get("resource_id")
    if not obj_id:
        log("fleet staging: no obj id in inputs; nothing staged")
        return []

    job_dir = Path(job_dir)
    try:
        import sqlalchemy as sa

        _ensure_db()
        from baselayer.app.models import DBSession
        from skyportal.broker_apis import boom as boom_api
        from skyportal.models import Broker

        with DBSession() as session:
            broker = session.scalars(
                sa.select(Broker)
                .where(Broker.broker_classname == "BOOMBROKER", Broker.active.is_(True))
                .order_by(Broker.default_alert_search.desc(), Broker.id)
            ).first()
            if broker is None:
                log("fleet staging: no active BOOM broker; nothing staged")
                return []

            collection = f"{_survey_suffix(obj_id)}_alerts_aux"
            res = boom_api._request(
                broker,
                "POST",
                "queries/find",
                json={
                    "catalog_name": collection,
                    "filter": {"_id": {"$in": [obj_id]}},
                    "projection": {"cross_matches": 1, "host_galaxy": 1},
                },
            )
            docs = res.get("data") if isinstance(res, dict) else res
            if isinstance(docs, dict):
                docs = docs.get("data", docs)
            doc = (docs or [{}])[0] if isinstance(docs, list) and docs else {}
            cross = doc.get("cross_matches") or {}
            host = {cat: cross.get(cat) for cat in HOST_CATALOGS if cross.get(cat)}
            if not host and not doc.get("host_galaxy"):
                log(f"fleet staging: no BOOM host data for {obj_id}; hostless run")
                return []

            payload = {**host, "host_galaxy": doc.get("host_galaxy")}
            out = job_dir / HOST_FILE
            out.write_text(json.dumps(payload))
            n_ls = len(host.get("LSDR10") or [])
            log(f"fleet staging: staged host data for {obj_id} (LSDR10 rows={n_ls})")
            return [out]
    except Exception as e:  # noqa: BLE001 — staging is optional; the job runs hostless
        log(f"fleet staging: fetch failed for {obj_id}: {e!r}")
        return []
