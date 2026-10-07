"""
FLARE bridge — turns a SkyPortal photometry payload into a FLARE classification.

Parses the payload here (stdlib only, like the other bridges, so tests import it
in a lightweight env) and hands the detections, the redshift, the source's
SkyPortal annotations and the parameters to ``flare.skyportal.analyze`` in the
FLARE image, which owns the science:
features, catalogue context, the hierarchical classifier, conformal sets, the
anomaly energy and the triage verdict. Returns results / annotations / plots in
the shape the wrapper packs for SkyPortal's callback.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
from pathlib import Path

csv.field_size_limit(10**9)

# BOOM context staged by the listener (flare_staging); turns FLARE's context block
# offline when present. Absent -> FLARE fetches it live, as before.
CONTEXT_FILE = "flare_context.json"

# SkyPortal filter names -> ZTF fid (g=1, r=2, i=3), the order flare.fetch.to_events expects.
FILTERS = {
    "ztfg": 1,
    "ztfr": 2,
    "ztfi": 3,
    "g": 1,
    "r": 2,
    "i": 3,
    "sdssg": 1,
    "sdssr": 2,
    "sdssi": 3,
}


def _params(payload: dict) -> dict:
    return dict(payload.get("analysis_parameters") or {})


def merged_annotations(payload: dict) -> dict:
    """Flatten every annotation's ``data`` dict into one lookup of the source's
    catalogue context (sgscore1, distpsnr1, host separations, PS1 mags, ...).
    Later rows win, so the most recently exported value for a repeated field is
    the one kept."""
    merged: dict = {}
    for r in _read_csv(payload.get("annotations")):
        data = r.get("data")
        if isinstance(data, str):
            try:
                data = ast.literal_eval(data)
            except (ValueError, SyntaxError):
                continue
        if isinstance(data, dict):
            for k, v in data.items():
                if v is not None:
                    merged[k] = v
    return merged


def _read_csv(value) -> list[dict]:
    """SkyPortal ships each input type as a CSV string (or a path to one)."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value if "\n" in value else Path(value).read_text()
        return list(csv.DictReader(io.StringIO(text)))
    if isinstance(value, list):
        return value
    return []


def _to_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


# A detection is a >=5-sigma point. ZTF magerr relates to SNR as SNR ~ 1.0857/magerr,
# so 5 sigma is magerr <= 1.0857/5. This drops sub-threshold forced-photometry points
# (SNR<5) that would otherwise be read as a light curve.
DETECTION_SNR = 5.0
_MAX_MAGERR = 1.0857 / DETECTION_SNR


def photometry_rows(payload: dict) -> list[tuple]:
    """5-sigma detections only, as (mjd, fid, mag, magerr); upper limits, non-ZTF
    filters, and sub-5-sigma forced-photometry points are dropped. FLARE was
    trained on detections."""
    rows = []
    for r in _read_csv(payload.get("photometry")):
        fid = FILTERS.get(str(r.get("filter", "")).strip().lower())
        mjd, mag, err = _to_float(r.get("mjd")), _to_float(r.get("mag")), _to_float(r.get("magerr"))
        if fid is None or mjd is None or mag is None or err is None or err <= 0:
            continue
        if err > _MAX_MAGERR:  # SNR < 5
            continue
        rows.append((mjd, fid, mag, err))
    if not rows:
        raise ValueError(
            "no ZTF detections in payload; the analysis service needs input_data_type 'photometry'"
        )
    return sorted(rows)


def resolve_redshift(payload: dict):
    """analysis_parameters.redshift wins; otherwise the source's SkyPortal value."""
    z = _to_float(_params(payload).get("redshift"))
    if z is not None:
        return z
    rows = _read_csv(payload.get("redshift"))
    return _to_float(rows[0].get("redshift")) if rows else None


# FLARE's 6 classes -> the nearest label in SkyPortal's Sitewide Taxonomy. CC has
# no umbrella node (Type II is the modal subtype) and SLSN only exists as subtypes.
FLARE_TAXONOMY = "Sitewide Taxonomy"
FLARE_TO_TAXONOMY = {
    "SN_Ia": "Ia",
    "SN_CC": "Type II",
    "SLSN": "Ic-SLSN",
    "AGN": "AGN",
    "TDE": "Tidal Disruption Event",
    "CV": "Cataclysmic",
}
# Distinct origin so FLARE's rows render as their own set, apart from other
# classifiers and from human labels (ml=True).
FLARE_ORIGIN = "FLARE"


def _load_context(work_dir: str) -> dict:
    """The BOOM context staged by the listener (flare_staging), or {} if absent."""
    path = os.path.join(work_dir, CONTEXT_FILE)
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 — context is optional; fall back to live fetch
        return {}


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    from flare.skyportal import analyze  # in the FLARE image

    params = _params(payload)
    # Skip FLARE's matplotlib PNG; SkyPortal renders the returned results natively.
    params.setdefault("plot", False)
    # BOOM context staged by the listener -> FLARE runs its context block offline
    # (no MAST/Data Lab fetch on the worker). Absent -> FLARE fetches it live.
    ctx = _load_context(work_dir)
    if ctx:
        params["context_data"] = ctx
    # Keep the job deterministic: the rule verdict only. The LLM triage layer is
    # the flare_triage MCP tool, so every assistant call stays server-side.
    params["agent"] = False
    result = analyze(
        photometry_rows(payload),
        resolve_redshift(payload),
        params,
        resource_id=resource_id,
        work_dir=work_dir,
    )
    _shape_for_skyportal(result)
    return result


def _shape_for_skyportal(result: dict) -> None:
    """Turn FLARE's output into the shapes SkyPortal's callback consumes: the flat
    annotation dict into the webhook's ``[{origin, data}]`` list (carrying the full
    probability vector), and an ml classification of the mapped predicted class."""
    cls = (result.get("results") or {}).get("classification") or {}
    probs = cls.get("probabilities") or {}
    predicted = cls.get("predicted")

    ann = result.get("annotations")
    if isinstance(ann, dict):
        data = {**ann, **{f"flare_p_{k}": v for k, v in probs.items()}}
        result["annotations"] = [{"origin": FLARE_ORIGIN, "data": data}]

    label = FLARE_TO_TAXONOMY.get(predicted)
    if label and predicted in probs:
        result["classifications"] = [
            {
                "taxonomy": FLARE_TAXONOMY,
                "classification": label,
                "probability": probs[predicted],
                "ml": True,
                "origin": FLARE_ORIGIN,
            }
        ]
