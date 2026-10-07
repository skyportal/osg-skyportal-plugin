"""
Stage FLARE's external context from BOOM so a FLARE job can run its context block
offline, instead of fetching Gaia/AllWISE/PS1/photo-z live from MAST + NOIRLab Data
Lab on the worker (which silently NaNs every context feature if a node lacks egress
or an archive is down). Runs on the listener at submit time, like oracle_staging.

Opt-in: only runs when ``flare.use_boom_context`` is set in the plugin config.
Without it, nothing is staged and the FLARE job keeps its live-fetch behaviour.

In-pod via the ORM: find the active BOOM broker, read the object's alert aux
(``cross_matches`` + ``host_galaxy``) and map the nearest-source rows into the
contract flare.context.context_features(provided=...) expects:
``{gaia, wise, pos, photoz}``. The point-source rows are passed through with their
native catalogue columns (FLARE computes the features). The host 30" aggregation is
NOT reproducible from BOOM and is left out (FLARE then NaNs that block) pending a
host contract with the FLARE maintainer.

Best-effort: standalone (no SkyPortal/DB), no broker, or no crossmatches -> [], and
the job falls back to live fetch.
"""

from __future__ import annotations

import json
from pathlib import Path

# Must match flare_bridge.CONTEXT_FILE (kept local to avoid importing the bridge).
CONTEXT_FILE = "flare_context.json"
SURVEY = "ZTF"

_DB_INITED = False


def _ensure_db() -> None:
    global _DB_INITED
    if _DB_INITED:
        return
    from baselayer.app.env import load_env
    from baselayer.app.models import init_db

    _, app_cfg = load_env()
    init_db(**app_cfg["database"])
    _DB_INITED = True


def _nearest(cross_matches: dict, *keys):
    """The first (nearest) matched row under the first present catalogue key."""
    for k in keys:
        rows = cross_matches.get(k)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows[0]
    return None


def _photoz(cross_matches: dict) -> dict:
    """z_phot block from the Legacy Survey (LSDR10) or NED, in FLARE's field names."""
    ls = _nearest(cross_matches, "LSDR10", "LS_DR10_PHOTOZ")
    if ls and ls.get("z_phot_median") is not None:
        return {"z_phot": ls.get("z_phot_median"), "z_phot_std": ls.get("z_phot_std")}
    ned = _nearest(cross_matches, "NED")
    if ned and ned.get("z") is not None:
        return {"z_phot": ned.get("z")}
    return {}


def stage_context(cfg: dict, inputs: dict, job_dir: Path, log=print) -> list[Path]:
    """Stage FLARE context from BOOM; return [path] ([] if disabled/unavailable)."""
    if not (cfg.get("flare") or {}).get("use_boom_context"):
        return []  # opt-in; default keeps FLARE's live fetch

    obj = inputs.get("obj") or {}
    obj_id = obj.get("id") or inputs.get("resource_id")
    if not obj_id:
        log("flare staging: no obj id in inputs; nothing staged")
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
                log("flare staging: no active BOOM broker; nothing staged")
                return []

            res = boom_api._request(
                broker,
                "POST",
                "queries/find",
                json={
                    "catalog_name": f"{SURVEY}_alerts_aux",
                    "filter": {"_id": {"$in": [obj_id]}},
                    "projection": {"cross_matches": 1},
                },
            )
            docs = res.get("data") if isinstance(res, dict) else res
            if isinstance(docs, dict):
                docs = docs.get("data", docs)
            doc = (docs or [{}])[0] if isinstance(docs, list) and docs else {}
            cm = doc.get("cross_matches") or {}

            # Point-source blocks: pass the nearest row through with its native
            # catalogue columns; flare.context computes the features (and the
            # separation from the obj ra/dec it already has).
            context = {}
            gaia = _nearest(cm, "Gaia_DR3", "Gaia_EDR3")
            if gaia:
                context["gaia"] = gaia
            wise = _nearest(cm, "AllWISE")  # absent until Caltech ingests it -> live/NaN parity
            if wise:
                context["wise"] = wise
            pos = _nearest(cm, "PS1_DR2", "PS1_DR1")
            if pos:
                context["pos"] = pos
            pz = _photoz(cm)
            if pz:
                context["photoz"] = pz

            if not context:
                log(f"flare staging: no usable BOOM context for {obj_id}; live fetch")
                return []

            out = job_dir / CONTEXT_FILE
            out.write_text(json.dumps(context))
            log(f"flare staging: staged context for {obj_id} ({sorted(context)})")
            return [out]
    except Exception as e:  # noqa: BLE001 — staging is optional; the job runs live
        log(f"flare staging: fetch failed for {obj_id}: {e!r}")
        return []
