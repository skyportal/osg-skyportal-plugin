"""
Oracle bridge — turns a SkyPortal photometry payload into an ORACLE-2 classification.

Runs ORACLE-2-lite (light-curve only, a GRU over the light curve) from the
``dev-ved30/oracle`` image. The batch construction mirrors the working
BOOM->SkyPortal path (the ``oracle_support`` repo): 5 time-series features per
detection -- days since first detection, magpsf, sigmapsf, filter mean-wavelength,
and a photflag of 1 -- with no static/metadata (the lite model reads only ``ts``
and ``length``). ``torch`` and ``oracle`` are imported lazily so this module
imports in a bare test env, matching the other bridges.
"""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path

# The lite model is a small GRU; keep it on CPU (OSPool oracle jobs request no GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

csv.field_size_limit(10**9)

# SkyPortal filter name -> ZTF band; only ZTF g/r/i have an Oracle bandpass.
FILTER_TO_BAND = {
    "ztfg": "g",
    "ztfr": "r",
    "ztfi": "i",
    "g": "g",
    "r": "r",
    "i": "i",
}

ORACLE_TAXONOMY = "Sitewide Taxonomy"
# ORACLE-2 BTS leaf classes -> nearest Sitewide Taxonomy label (id 1019).
ORACLE_TO_TAXONOMY = {
    "SN-Ia": "Ia",
    "SN-II": "Type II",
    "SN-Ib/c": "Ib/c",
    "SLSN": "Ic-SLSN",
    "AGN": "AGN",
    "CV": "Cataclysmic",
    "Varstar": "Stellar variable",
}
# Distinct origin so ORACLE's rows render as their own set (ml=True), apart from
# FLARE and human labels.
ORACLE_ORIGIN = "ORACLE"

# BTSv2-lite weights baked into the image; overridable via analysis_parameters.weights.
DEFAULT_WEIGHTS = "/opt/oracle/models/BTSv2-lite/fancy-elevator-567/best_model_f1.pth"

_MODEL = None


def _params(payload: dict) -> dict:
    return dict(payload.get("analysis_parameters") or {})


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


def photometry_rows(payload: dict) -> list[tuple]:
    """Detections only, as (mjd, band, mag, magerr); upper limits and non-ZTF
    filters are dropped. Sorted by time."""
    rows = []
    for r in _read_csv(payload.get("photometry")):
        band = FILTER_TO_BAND.get(str(r.get("filter", "")).strip().lower())
        mjd, mag, err = _to_float(r.get("mjd")), _to_float(r.get("mag")), _to_float(r.get("magerr"))
        if band is None or mjd is None or mag is None or err is None:
            continue
        rows.append((mjd, band, mag, err))
    return sorted(rows)


def _load_model(weights: str):
    global _MODEL
    if _MODEL is None:
        import torch
        from oracle.presets import get_model

        m = get_model("BTSv2-lite")
        m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        m.eval()
        _MODEL = m
    return _MODEL


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    rows = photometry_rows(payload)
    if not rows:
        return {
            "status": "failure",
            "message": "no ZTF g/r/i detections in payload for ORACLE",
        }

    import torch
    from oracle.custom_datasets.BTS import ZTF_passband_to_wavelengths

    weights = _params(payload).get("weights", DEFAULT_WEIGHTS)
    model = _load_model(weights)

    # 5 ts features per detection, matching the trained model / working BOOM path:
    # [days_since_first, magpsf, sigmapsf, mean_wavelength] + photflag. torch.ones
    # leaves the photflag column (index 4) at 1.0 (all points are detections).
    n = len(rows)
    t0 = rows[0][0]
    ts = torch.ones((1, n, 5))
    for i, (mjd, band, mag, err) in enumerate(rows):
        ts[0, i, 0] = mjd - t0
        ts[0, i, 1] = mag
        ts[0, i, 2] = err
        ts[0, i, 3] = ZTF_passband_to_wavelengths[band]
    batch = {"ts": ts, "length": torch.tensor([n])}

    with torch.no_grad():
        df = model.predict_class_probabilities_df(batch)

    leaves = model.taxonomy.get_leaf_nodes()
    probs = {c: float(df[c].iloc[0]) for c in leaves if c in df.columns}
    if not probs:
        return {"status": "failure", "message": "ORACLE produced no leaf probabilities"}
    predicted = max(probs, key=probs.get)
    prob = probs[predicted]

    result = {
        "status": "success",
        "message": f"ORACLE-2-lite: {predicted} (p={prob:.3f}), {n} detections",
        "results": {"predicted": predicted, "probabilities": probs},
        "annotations": [
            {"origin": ORACLE_ORIGIN, "data": {f"oracle_p_{k}": v for k, v in probs.items()}}
        ],
    }
    label = ORACLE_TO_TAXONOMY.get(predicted)
    if label:
        result["classifications"] = [
            {
                "taxonomy": ORACLE_TAXONOMY,
                "classification": label,
                "probability": prob,
                "ml": True,
                "origin": ORACLE_ORIGIN,
            }
        ]
    return result
