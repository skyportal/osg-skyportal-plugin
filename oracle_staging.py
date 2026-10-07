"""
Fetch an object's reference cutout from BOOM and stage it for an ORACLE job, so
the worker — which can't reach BOOM — runs the full omni model. Runs on the
listener at submit time, like aframe_staging.

The listener is an in-pod SkyPortal service, so it goes straight through the ORM:
open a DB session, find the active BOOM broker, resolve the object's latest candid
and pull its cutouts via the broker class. The broker carries its own BOOM
credentials (Broker.altdata), so no token of ours is involved. The alert query is
scoped to the same streams the source's groups grant (``survey_permissions``), so
the classifier only ever pulls a cutout it would be allowed to see. The reference
(template) cutout is written as the gzipped FITS bytes oracle_bridge expects.

Everything here is best-effort: run standalone (no SkyPortal/DB) or with no alert,
broker or cutout and it returns [] — the job then runs without the image, same as
the BOOM path with an absent template.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

# Must match oracle_bridge.CUTOUT_FILE / ALERT_FILE (kept local to avoid importing
# the bridge, and its CUDA env side effect, into the listener).
CUTOUT_FILE = "oracle_cutout.fits.gz"
# The full BOOM alert candidate (sky, fwhm, diffmaglim, chinr, sharpnr, PS1 mags,
# ...): the metadata the model was trained on, which SkyPortal annotations lack.
ALERT_FILE = "oracle_alert.json"

# BOOM cutout keys for the reference image, most-specific first.
TEMPLATE_KEYS = ("cutoutTemplate", "template", "cutoutReference", "reference")

# Only ZTF alerts carry an ORACLE-trained BTS light curve; aux lives under this prefix.
SURVEY = "ZTF"

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


def _latest_alert(alerts) -> dict | None:
    """The most recent alert (by candidate jd, else jd)."""
    if not isinstance(alerts, list) or not alerts:
        return None

    def jd(a):
        cand = a.get("candidate") or {}
        return cand.get("jd") or a.get("jd") or 0

    return max(alerts, key=jd)


def _ensure_db() -> None:
    """Initialise the DB connection once (this service's own, like any SkyPortal
    microservice). Reads the full app config directly: the listener is handed only
    its own params block, which has no ``database`` key. Raises if SkyPortal isn't
    importable — i.e. running standalone — and the caller degrades to no cutout."""
    global _DB_INITED
    if _DB_INITED:
        return
    from baselayer.app.env import load_env
    from baselayer.app.models import init_db

    _, app_cfg = load_env()
    init_db(**app_cfg["database"])
    _DB_INITED = True


def stage_cutout(cfg: dict, inputs: dict, job_dir: Path, log=print) -> list[Path]:
    """Stage the BOOM alert metadata (candidate + cross-matches) and the reference
    cutout for an ORACLE job; return the staged paths ([] if nothing available).

    The alert candidate carries the full static/metadata the model was trained on
    (sky, fwhm, diffmaglim, chinr, sharpnr, PS1 mags, ...), which SkyPortal
    annotations are missing -- so the bridge prefers it over annotations. Best
    effort throughout: each piece is independent and absence just degrades the job.
    """
    obj = inputs.get("obj") or {}
    obj_id = obj.get("id") or inputs.get("resource_id")
    if not obj_id:
        log("oracle staging: no obj id in inputs; nothing staged")
        return []

    job_dir = Path(job_dir)
    staged: list[Path] = []
    try:
        import sqlalchemy as sa

        _ensure_db()
        from baselayer.app.models import DBSession
        from skyportal.broker_apis.interface import survey_permissions
        from skyportal.models import (  # noqa: F401 — Stream for the join
            Broker,
            Group,
            Source,
            Stream,
        )

        with DBSession() as session:
            broker = session.scalars(
                sa.select(Broker)
                .where(Broker.broker_classname == "BOOMBROKER", Broker.active.is_(True))
                .order_by(Broker.default_alert_search.desc(), Broker.id)
            ).first()
            if broker is None:
                log("oracle staging: no active BOOM broker; nothing staged")
                return []

            # Scope to the streams the source's own groups grant: the classifier
            # only sees alerts those groups would.
            groups = (
                session.scalars(
                    sa.select(Group)
                    .join(Source, Source.group_id == Group.id)
                    .where(Source.obj_id == obj_id)
                )
                .unique()
                .all()
            )
            permissions = survey_permissions([s for g in groups for s in g.streams])

            alerts = broker.broker_class.query_alerts(
                broker, session, objectId=obj_id, permissions=permissions
            )
            latest = _latest_alert(alerts)
            if latest is None:
                log(f"oracle staging: no accessible BOOM alert for {obj_id}; nothing staged")
                return []
            candid = latest.get("candid") or latest.get("_id")

            # The alert doc (ZTF_alerts) carries the candidate; the detection
            # history (prv_candidates) and cross-matches live in the aux doc. Pull
            # the aux once so the bridge classifies the BOOM light curve ORACLE was
            # trained on (magpsf/sigmapsf/fid), not SkyPortal photometry.
            candidate = latest.get("candidate") or {}
            cross_matches = (
                latest.get("cross_matches") or latest.get("xmatch") or latest.get("aux") or {}
            )
            prv_candidates = []
            try:
                from skyportal.broker_apis import boom as boom_api

                res = boom_api._request(
                    broker,
                    "POST",
                    "queries/find",
                    json={
                        "catalog_name": f"{SURVEY}_alerts_aux",
                        "filter": {"_id": {"$in": [obj_id]}},
                        "projection": {"prv_candidates": 1, "cross_matches": 1},
                    },
                )
                docs = res.get("data") if isinstance(res, dict) else res
                if isinstance(docs, dict):
                    docs = docs.get("data", docs)
                aux = (docs or [{}])[0] if isinstance(docs, list) and docs else {}
                prv_candidates = aux.get("prv_candidates") or []
                cross_matches = aux.get("cross_matches") or cross_matches
            except Exception as e:  # noqa: BLE001 — bridge falls back to SkyPortal LC
                log(f"oracle staging: aux fetch failed for {obj_id}: {e!r}")

            if candidate or prv_candidates:
                alert_path = job_dir / ALERT_FILE
                alert_path.write_text(
                    json.dumps(
                        {
                            "candidate": candidate,
                            "cross_matches": cross_matches,
                            "prv_candidates": prv_candidates,
                        }
                    )
                )
                staged.append(alert_path)
                log(
                    f"oracle staging: staged alert metadata for {obj_id} "
                    f"(candid {candid}, {len(prv_candidates)} prv_candidates)"
                )

            cutouts = broker.broker_class.get_cutouts(
                broker, candid, session, permissions=permissions
            )
            field = next((cutouts[k] for k in TEMPLATE_KEYS if cutouts.get(k) is not None), None)
            raw = _gzip_fits_bytes(field)
            if raw:
                cutout_path = job_dir / CUTOUT_FILE
                cutout_path.write_bytes(raw)
                staged.append(cutout_path)
                log(f"oracle staging: staged reference cutout for {obj_id} (candid {candid})")
            else:
                log(f"oracle staging: no reference cutout for {obj_id} (candid {candid})")
    except Exception as e:  # noqa: BLE001 — staging is optional; the job runs without it
        log(f"oracle staging: fetch failed for {obj_id}: {e!r}")

    return staged
