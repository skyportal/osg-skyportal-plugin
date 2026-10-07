"""
FLEET bridge — turns a SkyPortal payload + BOOM host data into a FLEET
classification, with NO external/network calls (the OSG worker can't reach the
archives FLEET would otherwise query).

FLEET (Gomez et al.) is a scikit-learn random forest over light-curve + host-
galaxy features, aimed at SLSNe and TDEs. Normally it fetches its own data
(ALeRCE/ZTF light curve, TNS, SDSS/PS1/Gaia host crossmatch). We feed all of it
from disk instead and run ``predict`` fully offline:

- the light curve (SkyPortal photometry) -> ``photometry/{name}.txt``;
- the host galaxy (BOOM ``cross_matches.LSDR10`` + ``host_galaxy``, staged by the
  listener as ``fleet_host.json``) -> a cached ``catalogs/{name}.cat`` so FLEET
  reads it with ``reimport_catalog=False`` instead of querying;
- coords / redshift passed as args; SFD dust map baked into the image.

We always write a catalog file (empty if no host is available) so FLEET never
falls through to a network query; absent a host it uses its own hostless path.

``fleet`` is imported lazily so this module imports in a bare test env; the
pure-logic helpers (light-curve + catalog synthesis, taxonomy map) are tested
without FLEET installed.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

csv.field_size_limit(10**9)

# SkyPortal filter name -> FLEET filter token. FLEET's colour features select rows
# by exact Filter == 'g' / 'r', so emit single-char lowercase bands.
FILTERS = {
    "ztfg": "g",
    "ztfr": "r",
    "ztfi": "i",
    "g": "g",
    "r": "r",
    "i": "i",
    "sdssg": "g",
    "sdssr": "r",
    "sdssi": "i",
}

# Host data staged by the listener (raw BOOM crossmatch + association), decoded here.
HOST_FILE = "fleet_host.json"

# FLEET 10-class RF vector -> nearest Sitewide Taxonomy label (mirrors
# flare_bridge's mapping where classes overlap). The SLSN-II / Star rows have no
# clean Sitewide node; confirm these against taxonomy 1020 before trusting them.
FLEET_TAXONOMY = "Sitewide Taxonomy"
FLEET_TO_TAXONOMY = {
    "SNIa": "Ia",
    "SNII": "Type II",
    "SNIIn": "IIn",
    "SNIIb": "IIb",
    "SNIbc": "Ib/c",
    "SLSNI": "Ic-SLSN",
    "SLSNII": "IIn-SLSN",
    "AGN": "AGN",
    "TDE": "Tidal Disruption Event",
    "Star": "Stellar variable",
}
FLEET_CLASSES = tuple(FLEET_TO_TAXONOMY)
FLEET_ORIGIN = "FLEET"

# LSDR10 star/galaxy type codes to exclude when picking host candidates (BOOM's
# host association uses the same exclusions).
_STAR_TYPES = {"PSF", "DUP"}
# SDSS-style bands FLEET parses; LSDR10 lacks u, so u is filled from g (see below).
_CAT_BANDS = ("u", "g", "r", "i", "z")


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


def resolve_redshift(payload: dict):
    """analysis_parameters.redshift wins; otherwise the source's SkyPortal value."""
    z = _to_float(_params(payload).get("redshift"))
    if z is not None:
        return z
    rows = _read_csv(payload.get("redshift"))
    return _to_float(rows[0].get("redshift")) if rows else None


def resolve_coords(payload: dict) -> tuple:
    """(ra, dec) in degrees from the obj SkyPortal sends with the request."""
    obj = payload.get("obj") or {}
    return _to_float(obj.get("ra")), _to_float(obj.get("dec"))


def lightcurve_rows(payload: dict) -> list[dict]:
    """SkyPortal photometry -> FLEET light-curve rows (detections AND upper
    limits; FLEET uses both). Only g/r/i map to a FLEET band."""
    rows = []
    for r in _read_csv(payload.get("photometry")):
        band = FILTERS.get(str(r.get("filter", "")).strip().lower())
        mjd = _to_float(r.get("mjd"))
        if band is None or mjd is None:
            continue
        mag = _to_float(r.get("mag"))
        magerr = _to_float(r.get("magerr"))
        lim = _to_float(r.get("limiting_mag"))
        if mag is not None and magerr is not None:
            rows.append({"MJD": mjd, "Raw": mag, "MagErr": magerr, "Filter": band, "UL": "False"})
        elif lim is not None:  # non-detection -> upper limit at the limiting mag
            rows.append({"MJD": mjd, "Raw": lim, "MagErr": 0.1, "Filter": band, "UL": "True"})
    return sorted(rows, key=lambda x: x["MJD"])


def write_lightcurve(payload: dict, name: str, work_dir: str) -> Path:
    """Write ``photometry/{name}.txt`` in FLEET's local-light-curve schema. FLEET
    applies Galactic extinction itself (baked SFD), so magnitudes stay raw."""
    rows = lightcurve_rows(payload)
    if not rows:
        raise ValueError("no g/r/i photometry in payload for FLEET")
    ra, dec = resolve_coords(payload)
    out_dir = Path(work_dir) / "photometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    cols = ["MJD", "Raw", "MagErr", "Telescope", "Filter", "Source", "UL", "RA", "DEC"]
    with path.open("w") as f:
        f.write(" ".join(cols) + "\n")
        for r in rows:
            f.write(
                f"{r['MJD']} {r['Raw']} {r['MagErr']} ZTF {r['Filter']} ZTF "
                f"{r['UL']} {ra if ra is not None else 'nan'} {dec if dec is not None else 'nan'}\n"
            )
    return path


def _nanomaggy_to_ab(flux):
    """Legacy Survey nanomaggy flux -> AB magnitude (total/model mag)."""
    f = _to_float(flux)
    if f is None or f <= 0:
        return None
    return 22.5 - 2.5 * math.log10(f)


def _host_rows(host: dict) -> list[dict]:
    """Galaxy candidates from the staged BOOM data, each as a FLEET SDSS-style
    catalogue row. Prefers the LSDR10 crossmatch rows (they carry the fluxes +
    half-light radius); the host_galaxy association only tells us which is best,
    not photometry, so we classify all non-stellar LSDR10 rows and let FLEET's
    own get_best_host pick via Pcc."""
    lsdr10 = host.get("LSDR10") or host.get("cross_matches", {}).get("LSDR10") or []
    out = []
    for row in lsdr10:
        objtype = str(row.get("objtype") or "").strip().upper()
        if objtype in _STAR_TYPES:
            continue
        ra = _to_float(row.get("ra"))
        dec = _to_float(row.get("dec"))
        shape_r = _to_float(row.get("shape_r"))
        if ra is None or dec is None:
            continue
        mags = {b: _nanomaggy_to_ab(row.get(f"flux_{b}")) for b in ("g", "r", "i", "z")}
        if mags["g"] is None and mags["r"] is None:
            continue  # no usable host photometry
        mags["u"] = mags["g"]  # LSDR10 has no u; stand in with g for the SDSS schema
        rad = shape_r if shape_r and shape_r > 0 else 0.7  # FLEET's default_radius floor
        rec = {"ra_matched": ra, "dec_matched": dec, "type_sdss": 3, "object_nature": 1.0}
        for b in _CAT_BANDS:
            m = mags.get(b)
            val = m if m is not None else 99.0
            rec[f"psfMag_{b}_sdss"] = val  # object_nature supplied, so PSF is nominal
            rec[f"modelMag_{b}_sdss"] = val
            rec[f"petroR50_{b}_sdss"] = rad
        out.append(rec)
    return out


def write_catalog(name: str, work_dir: str) -> Path:
    """Write ``catalogs/{name}.cat`` from the staged BOOM host data. Always writes
    a file (header-only when no host) so FLEET's reimport_catalog=False reads it
    and never falls through to a network query."""
    host = {}
    hp = Path(work_dir) / HOST_FILE
    if hp.exists():
        try:
            host = json.loads(hp.read_text())
        except Exception:  # noqa: BLE001 — no/invalid host -> hostless run
            host = {}
    rows = _host_rows(host)
    cols = ["ra_matched", "dec_matched", "type_sdss", "object_nature"]
    for b in _CAT_BANDS:
        cols += [f"psfMag_{b}_sdss", f"modelMag_{b}_sdss", f"petroR50_{b}_sdss"]
    out_dir = Path(work_dir) / "catalogs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.cat"
    with path.open("w") as f:
        f.write(" ".join(cols) + "\n")
        for r in rows:
            f.write(" ".join(str(r[c]) for c in cols) + "\n")
    return path


def _extract_probabilities(info_table) -> dict:
    """Pull the per-class probabilities from FLEET's returned info table. FLEET
    names them variously (e.g. P_<class>, P_late_<class>); match the known class
    set against the columns, suffix-first."""
    try:
        columns = list(info_table.colnames)
    except AttributeError:
        columns = list(getattr(info_table, "columns", []))
    probs = {}
    for cls in FLEET_CLASSES:
        for col in columns:
            if col == cls or col.endswith(f"_{cls}") or col.endswith(cls):
                try:
                    val = info_table[col][0]
                except Exception:  # noqa: BLE001
                    continue
                v = _to_float(val)
                if v is not None:
                    probs[cls] = round(v, 4)
                    break
    return probs


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    from fleet.classify import predict  # in the FLEET image

    name = resource_id or "obj"
    write_lightcurve(payload, name, work_dir)
    write_catalog(name, work_dir)  # always present -> no network query
    ra, dec = resolve_coords(payload)
    params = _params(payload)

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        info = predict(
            object_name_in=name,
            ra_in=ra,
            dec_in=dec,
            redshift_in=resolve_redshift(payload),
            object_class_in=params.get("object_class"),
            # Fully offline: read staged light curve + cached catalogue, no queries.
            download_ztf=False,
            download_osc=False,
            download_rubin=False,
            read_local=True,
            query_tns=False,
            reimport_catalog=False,
            dust_map="SFD",
            classify=True,
            plot_output=False,
            do_observability=False,
            save_lc=False,
            save_catalog=False,
            save_params=False,
            emcee_progress=False,
            n_cores=1,
        )
    finally:
        os.chdir(cwd)

    probs = _extract_probabilities(info)
    if not probs:
        return {"status": "failure", "message": "FLEET produced no class probabilities"}
    predicted = max(probs, key=probs.get)
    prob = probs[predicted]
    host_staged = (Path(work_dir) / HOST_FILE).exists()
    msg = (
        f"FLEET: {predicted} (p={prob:.3f}), {len(lightcurve_rows(payload))} points, "
        f"{'with host' if host_staged else 'hostless'}"
    )
    result = {
        "status": "success",
        "message": msg,
        "results": {
            "predicted": predicted,
            "probabilities": probs,
            "p_slsn_i": probs.get("SLSNI"),
            "p_tde": probs.get("TDE"),
        },
    }
    _shape_for_skyportal(result)
    return result


def _shape_for_skyportal(result: dict) -> None:
    """Turn the bridge output into SkyPortal's callback shapes: the probability
    vector as an ``[{origin, data}]`` annotation, and an ml classification of the
    mapped predicted class."""
    res = result.get("results") or {}
    probs = res.get("probabilities") or {}
    predicted = res.get("predicted")

    data = {f"fleet_p_{k}": v for k, v in probs.items()}
    data["fleet_class"] = predicted
    result["annotations"] = [{"origin": FLEET_ORIGIN, "data": data}]

    label = FLEET_TO_TAXONOMY.get(predicted)
    if label and predicted in probs:
        result["classifications"] = [
            {
                "taxonomy": FLEET_TAXONOMY,
                "classification": label,
                "probability": probs[predicted],
                "ml": True,
                "origin": FLEET_ORIGIN,
            }
        ]
