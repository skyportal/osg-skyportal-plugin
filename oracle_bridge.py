"""
Oracle bridge — turns a SkyPortal photometry payload into an ORACLE-2 classification.

Runs ORACLE-2 in its full "omni" mode (``BTSv2-pro``, the GRU + metadata + image
model) from the ``dev-ved30/oracle`` image. The lighter models are far weaker:
without the host/context features an ordinary supernova gets mislabelled as AGN,
and with only partial metadata the model is confidently wrong, so we feed all
three inputs it was trained on:

- the light curve (ZTF g/r/i detections);
- a 30-d static vector — the ZTF/BOOM alert fields (sgscore1, distpsnr1, ndethist,
  drb, PS1 mags, ...) pulled from the source's SkyPortal annotations, plus the
  galactic coordinates from the obj ra/dec, flag value where a field is absent;
- the reference (template) cutout, fetched from BOOM by the listener and staged in
  the sandbox (see oracle_staging), placed in the channel of the last detection's
  band. Missing cutout -> a zero postage stamp, matching the BOOM path when the
  template is absent.

The batch mirrors the real-time BOOM path (oracle_support): 5 ts features per
detection (days since first detection, magpsf, sigmapsf, filter mean-wavelength,
photflag=1), the 30-d static vector, and a (1, 3, 63, 63) L2-normalised postage
stamp. ``torch`` and ``oracle`` are imported lazily so this module imports in a
bare test env.
"""

from __future__ import annotations

import ast
import csv
import io
import math
import os
from pathlib import Path

# BTSv2 is a small model; keep it on CPU (OSPool oracle jobs request no GPU).
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

# A few Oracle metadata feature names differ from the annotation field names.
META_ALIASES = {"scorr": ("scorr", "zogy_scorr")}

# ZTF band -> postage-stamp channel; the reference cutout goes in the last
# detection's band, matching the BOOM path (oracle_support).
BAND_TO_CHANNEL = {"g": 0, "r": 1, "i": 2}

# Reference cutout staged into the sandbox by the listener (gzipped FITS bytes).
CUTOUT_FILE = "oracle_cutout.fits.gz"

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

DEFAULT_MODEL = "BTSv2-pro"

_MODEL = None
_MODEL_KEY = None


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


# A detection is a >=5-sigma point. ZTF magerr relates to SNR as SNR ~ 1.0857/magerr,
# so 5 sigma is magerr <= 1.0857/5. This drops sub-threshold forced-photometry points
# (SNR<5) that would otherwise be read as a light curve.
DETECTION_SNR = 5.0
_MAX_MAGERR = 1.0857 / DETECTION_SNR


def photometry_rows(payload: dict) -> list[tuple]:
    """5-sigma detections only, as (mjd, band, mag, magerr); upper limits, non-ZTF
    filters, and sub-5-sigma forced-photometry points are dropped. Sorted by time."""
    rows = []
    for r in _read_csv(payload.get("photometry")):
        band = FILTER_TO_BAND.get(str(r.get("filter", "")).strip().lower())
        mjd, mag, err = _to_float(r.get("mjd")), _to_float(r.get("mag")), _to_float(r.get("magerr"))
        if band is None or mjd is None or mag is None or err is None or err <= 0:
            continue
        if err > _MAX_MAGERR:  # SNR < 5
            continue
        rows.append((mjd, band, mag, err))
    return sorted(rows)


def merged_annotations(payload: dict) -> dict:
    """Flatten every annotation's ``data`` dict into one lookup of alert fields
    (sgscore1, distpsnr1, ndethist, drb, PS1 mags, ...). Later rows win, so the
    most recently exported value for a repeated field is the one kept."""
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


def _feat(merged: dict, name: str, flag: float) -> float:
    """One Oracle static/meta feature from the merged annotations, else the flag
    value. -999 (ZTF's own missing marker) also maps to the flag value."""
    for key in META_ALIASES.get(name, (name,)):
        v = _to_float(merged.get(key))
        if v is not None and v != -999:
            return v
    return flag


def _galactic(payload: dict) -> tuple:
    """Galactic (l, b) from the obj ra/dec SkyPortal sends with the request."""
    obj = payload.get("obj") or {}
    ra, dec = _to_float(obj.get("ra")), _to_float(obj.get("dec"))
    if ra is None or dec is None:
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        return float(c.l.deg), float(c.b.deg)
    except Exception:  # noqa: BLE001 — coords are optional; degrade to the flag value
        return None


def _load_cutout(path):
    """A staged reference cutout (gzipped FITS) -> a 63x63 L2-normalised array,
    or None if it is missing or unreadable. Mirrors oracle_support.load_cutout
    (no flip; the BOOM path feeds the unflipped array the model trained on)."""
    import gzip
    import io

    if not os.path.exists(path):
        return None
    try:
        import numpy as np
        from astropy.io import fits

        raw = gzip.decompress(Path(path).read_bytes())
        with fits.open(io.BytesIO(raw)) as hdul:
            image = hdul[0].data.astype(float)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(image)
        if norm != 0:
            image = image / norm
        return image
    except Exception:  # noqa: BLE001 — cutout is optional; degrade to a zero stamp
        return None


# The proven BTSv2-pro run (matches the real-time BOOM weights); preferred over
# any other run baked into the image.
PREFERRED_RUN = {"BTSv2-pro": "morning-feather-572"}


def _default_weights(model_choice: str) -> str:
    import glob

    hits = sorted(glob.glob(f"/opt/oracle/models/{model_choice}/*/best_model_f1.pth"))
    if not hits:
        raise FileNotFoundError(f"no {model_choice} weights under /opt/oracle/models")
    run = PREFERRED_RUN.get(model_choice)
    if run:
        preferred = [h for h in hits if f"/{run}/" in h]
        if preferred:
            return preferred[0]
    return hits[-1]


def _load_model(model_choice: str, weights: str):
    global _MODEL, _MODEL_KEY
    key = (model_choice, weights)
    if _MODEL is None or _MODEL_KEY != key:
        import torch

        if model_choice == "BTSv2-pro":
            # get_model("BTSv2-pro") builds the spines from relative default dirs
            # that don't exist in the sandbox; build with None spines and load the
            # combined weights, as the BOOM path does.
            from oracle.architectures import GRU_MD_MM_Improved
            from oracle.taxonomies import BTS_Taxonomy

            m = GRU_MD_MM_Improved(BTS_Taxonomy(), lc_md_model_dir=None, image_model_dir=None)
        else:
            from oracle.presets import get_model

            m = get_model(model_choice)
        m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        m.eval()
        _MODEL, _MODEL_KEY = m, key
    return _MODEL


def _build_batch(rows, payload, torch, work_dir="."):
    """(mjd, band, mag, magerr) rows + the source annotations -> the model batch.
    ts: 5 features per point [days_since_first, magpsf, sigmapsf, mean_wavelength]
    with photflag left at 1. static: the 30-d [time-independent + metadata] vector
    the GRU+MD model expects, from annotations (flag value where absent).
    postage_stamp: (1, 3, 63, 63), the reference cutout in the last detection's
    band channel, zeros where it's missing."""
    from oracle.custom_datasets.BTS import (
        ZTF_passband_to_wavelengths,
        flag_value,
        meta_data_feature_list,
        time_independent_feature_list,
    )

    n = len(rows)
    t0 = rows[0][0]
    ts = torch.ones((1, n, 5))
    for i, (mjd, band, mag, err) in enumerate(rows):
        ts[0, i, 0] = mjd - t0
        ts[0, i, 1] = mag
        ts[0, i, 2] = err
        ts[0, i, 3] = ZTF_passband_to_wavelengths[band]

    merged = merged_annotations(payload)
    lb = _galactic(payload)
    static_vals = []
    for col in time_independent_feature_list:
        if col == "l":
            static_vals.append(lb[0] if lb else flag_value)
        elif col == "b":
            static_vals.append(lb[1] if lb else flag_value)
        else:
            static_vals.append(_feat(merged, col, flag_value))
    static_vals += [_feat(merged, col, flag_value) for col in meta_data_feature_list]

    postage_stamp = torch.zeros((1, 3, 63, 63))
    image = _load_cutout(os.path.join(work_dir, CUTOUT_FILE))
    ch = BAND_TO_CHANNEL.get(rows[-1][1])  # last (most recent) detection's band
    if image is not None and getattr(image, "shape", None) == (63, 63) and ch is not None:
        postage_stamp[0, ch] = torch.from_numpy(image).float()

    return {
        "ts": ts,
        "static": torch.tensor([static_vals], dtype=torch.float32),
        "length": torch.tensor([n]),
        "postage_stamp": postage_stamp,
    }


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    rows = photometry_rows(payload)
    if not rows:
        return {"status": "failure", "message": "no ZTF g/r/i detections in payload for ORACLE"}

    import torch

    params = _params(payload)
    model_choice = str(params.get("model", DEFAULT_MODEL))
    weights = params.get("weights") or _default_weights(model_choice)
    model = _load_model(model_choice, weights)

    with torch.no_grad():
        df = model.predict_class_probabilities_df(_build_batch(rows, payload, torch, work_dir))

    leaves = model.taxonomy.get_leaf_nodes()
    probs = {c: round(float(df[c].iloc[0]), 4) for c in leaves if c in df.columns}
    if not probs:
        return {"status": "failure", "message": "ORACLE produced no leaf probabilities"}
    predicted = max(probs, key=probs.get)
    prob = probs[predicted]

    cutout = "with cutout" if os.path.exists(os.path.join(work_dir, CUTOUT_FILE)) else "no cutout"
    msg = f"ORACLE-2 ({model_choice}): {predicted} (p={prob:.3f}), {len(rows)} detections, {cutout}"
    result = {
        "status": "success",
        "message": msg,
        "results": {"model": model_choice, "predicted": predicted, "probabilities": probs},
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
