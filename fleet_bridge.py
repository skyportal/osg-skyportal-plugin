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
    "SLSNII": "II-SLSN",
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
# Max LSDR10 offset (arcsec) to adopt as the host when BOOM's DLR best_host is null
# (e.g. a PSF-typed nuclear TDE host it dropped) -- a nucleus sits well inside this.
_HOST_MAX_SEP_ARCSEC = 2.0


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


# FLEET's predict() fits only the detections within 75 days of the first one.
FLEET_PHASE_MAX_DAYS = 75
FLEET_MIN_DETECTIONS = 5
FLEET_MIN_DEC = -32


def unclassifiable_reason(payload: dict) -> str | None:
    """Why FLEET would return no class probabilities for this payload, or None."""
    _, dec = resolve_coords(payload)
    if dec is not None and dec <= FLEET_MIN_DEC:
        return f"FLEET does not classify sources at dec <= {FLEET_MIN_DEC}"
    dets = [r for r in lightcurve_rows(payload) if r["UL"] == "False"]
    early = [r for r in dets if r["MJD"] - dets[0]["MJD"] < FLEET_PHASE_MAX_DAYS]
    n_g = sum(r["Filter"] == "g" for r in early)
    n_r = sum(r["Filter"] == "r" for r in early)
    if n_g and n_r and len(early) >= FLEET_MIN_DETECTIONS:
        return None
    return (
        f"FLEET needs g and r detections and {FLEET_MIN_DETECTIONS} in all within "
        f"{FLEET_PHASE_MAX_DAYS} days of the first detection; this light curve has "
        f"{n_g} g, {n_r} r, {len(early)} in all"
    )


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


def _row_id(row: dict) -> str:
    """Stringified LSDR10 ``_id``, which host_galaxy.best_host.objname carries. The
    raw BSON int64 can arrive as ``{high, low}`` or ``{$numberLong}``; decode those,
    else str() the value."""
    v = row.get("_id")
    if isinstance(v, dict):
        if "high" in v and "low" in v:
            try:
                return str(int(v["high"]) * 2**32 + (int(v["low"]) & 0xFFFFFFFF))
            except (TypeError, ValueError):
                return ""
        if "$numberLong" in v:
            return str(v["$numberLong"])
    return str(v) if v is not None else ""


def _select_host_row(host: dict, lsdr10: list) -> dict | None:
    """The single LSDR10 row FLEET should use as the host, via BOOM's DLR work:
    ``host_galaxy.best_host`` matched to its LSDR10 row by id, else — when best_host
    is null, which happens for a TDE whose galaxy nucleus Legacy types PSF and BOOM's
    star-like cut discards — the nearest non-DUP row by ``distance_arcsec``. Returns
    None to leave the caller on the all-crossmatch/Pcc path."""
    hg = host.get("host_galaxy") or {}
    bh = hg.get("best_host") if isinstance(hg, dict) else None
    phot = (bh or {}).get("photometry") or {}
    if bh and str(bh.get("catalog") or "").upper() == "LSDR10":
        name = str(bh.get("objname") or "")
    elif str(phot.get("catalog") or "").upper() == "LSDR10":
        name = str(phot.get("source_id") or "")
    else:
        name = ""
    for r in lsdr10:
        if name and _row_id(r) == name:
            return r
    cands = []
    for r in lsdr10:
        if str(r.get("objtype") or "").strip().upper() == "DUP":
            continue
        d = _to_float(r.get("distance_arcsec"))
        if d is not None:
            cands.append((d, r))
    if cands:
        d, r = min(cands, key=lambda x: x[0])
        if d <= _HOST_MAX_SEP_ARCSEC:
            return r
    return None


def _row_to_catalog(row: dict) -> dict | None:
    """One LSDR10 row -> a FLEET SDSS-style catalogue record (None if no usable
    photometry). object_nature=1.0 / type_sdss=3 declare it a galaxy, overriding a
    PSF mistyping of a nuclear host."""
    mags = {b: _nanomaggy_to_ab(row.get(f"flux_{b}")) for b in ("g", "r", "i", "z")}
    return _catalog_record(row.get("ra"), row.get("dec"), mags, _to_float(row.get("shape_r")))


def _photometry_to_catalog(best_host: dict) -> dict | None:
    """BOOM's ``best_host.photometry`` (Pan-STARRS when the host is outside Legacy)
    -> a FLEET catalogue record at the host's position."""
    phot = best_host.get("photometry") or {}
    mags = {b: _to_float(phot.get(f"mag_{b}")) for b in ("g", "r", "i", "z")}
    a = _to_float(best_host.get("a_arcsec"))
    # an isophotal semi-major axis is about twice the half-light radius FLEET expects
    radius = a / 2 if a and best_host.get("size_is_isophotal") else a
    return _catalog_record(best_host.get("ra"), best_host.get("dec"), mags, radius)


def _catalog_record(ra, dec, mags: dict, radius: float | None) -> dict | None:
    ra = _to_float(ra)
    dec = _to_float(dec)
    if ra is None or dec is None:
        return None
    if mags["g"] is None and mags["r"] is None:
        return None  # no usable host photometry
    # Fill every missing band (including u, which LSDR10 lacks) from the nearest
    # present band. A missing band must not become a 99-mag sentinel: FLEET's Pcc
    # filter is the i-band, so a host with no i flux would read as a chance
    # coincidence (Pcc=1) and be dropped, even a bright nucleus at <1 arcsec.
    order = ["u", "g", "r", "i", "z"]
    present = {b: m for b, m in mags.items() if m is not None}
    for b in order:
        if mags.get(b) is None:
            nearest = min(present, key=lambda p: abs(order.index(p) - order.index(b)))
            mags[b] = present[nearest]
    rad = radius if radius and radius > 0 else 0.7  # FLEET's default_radius floor
    rec = {"ra_matched": ra, "dec_matched": dec, "type_sdss": 3, "object_nature": 1.0}
    for b in _CAT_BANDS:
        rec[f"psfMag_{b}_sdss"] = mags[b]
        rec[f"modelMag_{b}_sdss"] = mags[b]
        rec[f"petroR50_{b}_sdss"] = rad
    return rec


def _host_rows(host: dict) -> list[dict]:
    """FLEET SDSS-style catalogue rows from the staged BOOM data. Uses BOOM's DLR
    host when it resolves to an LSDR10 row (see ``_select_host_row``) and hands FLEET
    just that one, so it classifies with the DLR host instead of re-picking via Pcc
    (which drops nuclear TDE hosts). Otherwise falls back to every non-stellar
    crossmatch and lets FLEET choose."""
    lsdr10 = host.get("LSDR10") or host.get("cross_matches", {}).get("LSDR10") or []
    hg = host.get("host_galaxy") or {}
    bh = (hg.get("best_host") if isinstance(hg, dict) else None) or {}
    phot = bh.get("photometry") or {}
    if phot and str(phot.get("catalog") or "").upper() != "LSDR10":
        rec = _photometry_to_catalog(bh)
        return [rec] if rec is not None else []
    selected = _select_host_row(host, lsdr10)
    if selected is not None:
        rec = _row_to_catalog(selected)
        return [rec] if rec is not None else []
    out = []
    for row in lsdr10:
        if str(row.get("objtype") or "").strip().upper() in _STAR_TYPES:
            continue
        rec = _row_to_catalog(row)
        if rec is not None:
            out.append(rec)
    return out


def _no_host_sentinel(ra: float | None, dec: float | None) -> dict:
    """A far, faint, float-valued catalogue row for the no-host case. FLEET loads
    this file (reimport_catalog=False) and runs catalog_operations, which does an
    in-place ``psfMag -= extinction``; on a header-only file astropy types those
    columns int64 and numpy 2.x refuses the int-=float cast. One float row keeps the
    columns float64, and being ~2 deg away and 99th-mag it is discarded by FLEET's
    Pcc, so the object classifies hostless instead of crashing."""
    rec = {
        "ra_matched": (ra + 2.0) if ra is not None else 2.0,
        "dec_matched": dec if dec is not None else 0.0,
        "type_sdss": 3,
        "object_nature": 1.0,
    }
    for b in _CAT_BANDS:
        rec[f"psfMag_{b}_sdss"] = 99.0
        rec[f"modelMag_{b}_sdss"] = 99.0
        rec[f"petroR50_{b}_sdss"] = 0.7
    return rec


def write_catalog(
    name: str, work_dir: str, ra: float | None = None, dec: float | None = None
) -> Path:
    """Write ``catalogs/{name}.cat`` from the staged BOOM host data. Always writes a
    file so FLEET's reimport_catalog=False reads it and never falls through to a
    network query; with no host it writes a single discarded sentinel row (see
    ``_no_host_sentinel``) rather than a header-only file."""
    host = {}
    hp = Path(work_dir) / HOST_FILE
    if hp.exists():
        try:
            host = json.loads(hp.read_text())
        except Exception:  # noqa: BLE001 — no/invalid host -> hostless run
            host = {}
    rows = _host_rows(host) or [_no_host_sentinel(ra, dec)]
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


def _info_hostless(info_table) -> bool:
    """FLEET's hostless verdict from its info table (True when it found no host
    above the Pcc threshold). Defaults to True if the column is absent."""
    try:
        return bool(info_table["hostless"][0])
    except Exception:  # noqa: BLE001 — missing column / unexpected shape
        return True


def _extract_probabilities(info_table) -> dict:
    """Per-class probabilities from FLEET's main 10-class classifier. FLEET names
    these ``P_late_<class>`` and reports them as percentages (0-100), so rescale to
    fractions. The ``P_rapid_slsn_*`` / ``P_rapid_tde_*`` columns are the
    specialised early-time detectors and are not the general verdict."""
    try:
        columns = set(info_table.colnames)
    except AttributeError:
        columns = set(getattr(info_table, "columns", []))
    probs = {}
    for cls in FLEET_CLASSES:
        col = f"P_late_{cls}"
        if col not in columns:
            continue
        try:
            v = _to_float(info_table[col][0])
        except Exception:  # noqa: BLE001
            continue
        if v is not None:
            probs[cls] = round(v / 100.0, 4)
    return probs


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    from fleet.classify import predict  # in the FLEET image

    name = resource_id or "obj"
    ra, dec = resolve_coords(payload)
    write_lightcurve(payload, name, work_dir)
    write_catalog(name, work_dir, ra, dec)  # always present -> no network query
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
    # FLEET's own hostless verdict (it rejects a too-far/faint candidate via Pcc),
    # not merely whether a host file was staged.
    hostless = _info_hostless(info)
    msg = (
        f"FLEET: {predicted} (p={prob:.3f}), {len(lightcurve_rows(payload))} points, "
        f"{'hostless' if hostless else 'with host'}"
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
