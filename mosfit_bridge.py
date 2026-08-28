"""
SkyPortal-AnalysisService → MOSFiT bridge.

Takes a SkyPortal analysis payload (photometry + redshift + free-form
``analysis_parameters``), builds a local Open Astronomy Catalog (OAC) event JSON,
fits a MOSFiT model — Nickel-cobalt (``default``), ``slsn``, ``magnetar``,
``tde``, ``csm``, ... — offline with MOSFiT's own sampler (emcee ``ensembler`` or
``dynesty``), and returns a result dict including a per-filter
``model_lightcurve`` for overlaying the fit on SkyPortal's photometry plot. Runs
inside the mosfit runtime image on an OSG worker.

MOSFiT runs fully offline here: the event is a local file (no Open-Catalog
download), filter transmissions ship in the image (ZTF curves are baked in), and
extinction is computed from the event ``ebv``. See containers/mosfit.def.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# numpy/astropy/mosfit are imported lazily inside the functions that use them:
# this module is shipped to (and only runs in) the mosfit image, but the plugin's
# tests import it in a lightweight env without those deps.

# Defaults, overridable per-call via ``analysis_parameters`` in the payload.
DEFAULTS = {
    "source": "default",  # MOSFiT Nickel-cobalt (Arnett) SN model
    "method": "ensembler",  # emcee; "dynesty" for nested sampling
    "iterations": 500,
    "num_walkers": None,  # None -> sampler picks 2*ndim
    "smooth_times": 100,  # dense observer-frame grid for a smooth overlay
    "band_system": "AB",
}

# SkyPortal ZTF filter -> MOSFiT band letter. Emitting these with instrument
# "ZTF" makes MOSFiT resolve Palomar/ZTF.{g,r,i} (baked into the image), which
# avoids both the competing P48/CFH12k filter rule and any network SVO fetch.
_ZTF_BANDS = {"ztfg": "g", "ztfr": "r", "ztfi": "i"}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _band_tags(filt: str) -> tuple[str, str | None, str]:
    """Map a SkyPortal filter to (band, instrument, system) for the OAC entry.
    ZTF filters get instrument "ZTF" so the baked ZTF transmission curves are
    used; anything else falls back to its last character as a generic band."""
    key = str(filt).strip().lower()
    if key in _ZTF_BANDS:
        return _ZTF_BANDS[key], "ZTF", "AB"
    return (key[-1] if key else "r"), None, "AB"


def _resolve_redshift(payload: dict) -> float | None:
    src = payload.get("redshift")
    if src is None:
        return None
    from astropy.table import Table

    table = Table.read(src, format="ascii.csv")
    if len(table) == 0 or "redshift" not in table.colnames:
        return None
    # A missing/masked redshift is NaN through float(); return None so we fall
    # back to a distance-based fit rather than propagating NaN into the model.
    try:
        value = float(table["redshift"][0])
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # value != value is True only for NaN


def _photometry_rows(payload: dict) -> tuple[list, list, dict]:
    """Parse SkyPortal photometry into (detections, non-detections, band_map).

    detections: list of (mjd, sp_filter, band, instrument, system, mag, magerr).
    non-detections: list of (mjd, sp_filter, band, instrument, system, limit_mag).
    band_map: MOSFiT band letter -> the SkyPortal filter it came from (so the
    model light curve can be keyed back to SkyPortal's own filter names)."""
    import numpy as np
    from astropy.table import Table

    # SkyPortal flux zeropoint (µJy AB); NSIGMA is the fallback detection
    # threshold used only when limiting_mag isn't exported.
    PHOT_ZP, NSIGMA = 23.9, 5.0

    table = Table.read(payload["photometry"], format="ascii.csv")
    cols = table.colnames

    def _limit_mag(row) -> float | None:
        if "limiting_mag" in cols and not np.ma.is_masked(row["limiting_mag"]):
            v = float(row["limiting_mag"])
            if np.isfinite(v):
                return v
        if "fluxerr" in cols and not np.ma.is_masked(row["fluxerr"]):
            fe = float(row["fluxerr"])
            if np.isfinite(fe) and fe > 0:
                return -2.5 * np.log10(NSIGMA * fe) + PHOT_ZP
        return None

    dets: list = []
    nondets: list = []
    band_map: dict = {}
    for row in table:
        filt, mjd = str(row["filter"]), float(row["mjd"])
        band, inst, syst = _band_tags(filt)
        band_map.setdefault(band, filt)
        mag, magerr = row["mag"], row["magerr"]
        if not (np.ma.is_masked(mag) or np.ma.is_masked(magerr)):
            try:
                magf, errf = float(mag), float(magerr)
            except (TypeError, ValueError):
                magf = errf = float("nan")
            if np.isfinite(magf) and np.isfinite(errf):
                dets.append((mjd, filt, band, inst, syst, magf, errf))
                continue
        lim = _limit_mag(row)
        if lim is not None:
            nondets.append((mjd, filt, band, inst, syst, lim))
    return dets, nondets, band_map


def _select_upper_limits(dets: list, nondets: list) -> list:
    """Keep the most-recent upper limit before the first detection (pins the
    explosion epoch) plus any within the detection window, restricted to bands
    that actually have detections. Mirrors the fiesta bridge — stray pre/post
    limits add little and can bias the fit."""
    if not dets:
        return []
    first_det = min(d[0] for d in dets)
    last_det = max(d[0] for d in dets)
    det_bands = {d[2] for d in dets}
    usable = [n for n in nondets if n[2] in det_bands]
    kept = [n for n in usable if first_det <= n[0] <= last_det]
    pre = [n for n in usable if n[0] < first_det]
    if pre:
        kept.append(max(pre, key=lambda n: n[0]))
    return kept


def _build_event(payload: dict, name: str) -> tuple[dict, int, dict]:
    """Build a MOSFiT/OAC event dict from the payload. Returns (event, n_det,
    band_map). Redshift and ebv (if given) go in as event-level quantities so
    MOSFiT consumes them directly rather than as fixed sampler parameters."""
    params = _params(payload)
    dets, nondets, band_map = _photometry_rows(payload)
    kept = _select_upper_limits(dets, nondets)

    phot: list = []
    for mjd, _filt, band, inst, syst, mag, err in sorted(dets):
        e = {
            "time": f"{mjd:.6f}",
            "u_time": "MJD",
            "band": band,
            "system": syst,
            "magnitude": f"{mag:.4f}",
            "e_magnitude": f"{err:.4f}",
            "source": "1",
        }
        if inst:
            e["instrument"] = inst
        phot.append(e)
    for mjd, _filt, band, inst, syst, lim in sorted(kept):
        e = {
            "time": f"{mjd:.6f}",
            "u_time": "MJD",
            "band": band,
            "system": syst,
            "magnitude": f"{lim:.4f}",
            "upperlimit": True,
            "source": "1",
        }
        if inst:
            e["instrument"] = inst
        phot.append(e)

    event: dict = {
        "name": name,
        "sources": [{"name": "SkyPortal", "alias": "1"}],
        "photometry": phot,
    }
    z = _resolve_redshift(payload)
    if z is not None and z > 0:
        event["redshift"] = [{"value": f"{z:.6f}", "source": "1"}]
    ebv = params.get("ebv")
    if ebv is not None:
        event["ebv"] = [{"value": str(ebv), "source": "1"}]
    return {name: event}, len(dets), band_map


def count_detections(payload: dict) -> int:
    """Number of finite detections MOSFiT would fit — lets callers fail a request
    fast (and once) when there are fewer than two."""
    dets, _, _ = _photometry_rows(payload)
    return len(dets)


def _model_lightcurve(entry: dict, band_map: dict) -> dict:
    """Per-filter median + 16/84 apparent-mag curves on an MJD grid, from MOSFiT's
    per-realization model photometry (rows carrying a ``realization`` key), for the
    SkyPortal photometry overlay. Keyed by the original SkyPortal filter name."""
    import numpy as np

    by_band: dict = {}  # mosfit band -> {rounded mjd: [mags across realizations]}
    for p in entry.get("photometry", []) or []:
        if "realization" not in p or p.get("upperlimit"):
            continue
        band, mag, t = p.get("band"), p.get("magnitude"), p.get("time")
        if band is None or mag is None or t is None:
            continue
        try:
            tf, magf = float(t), float(mag)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(tf) and np.isfinite(magf)):
            continue
        by_band.setdefault(band, {}).setdefault(round(tf, 4), []).append(magf)

    out: dict = {}
    for band, tm in by_band.items():
        rows = []
        for t in sorted(tm):
            a = np.asarray(tm[t], dtype=float)
            rows.append(
                [
                    float(t),
                    float(np.nanmedian(a)),
                    float(np.nanpercentile(a, 16)),
                    float(np.nanpercentile(a, 84)),
                ]
            )
        out[band_map.get(band, band)] = rows
    return out


def _is_nuisance_param(name: str) -> bool:
    """MOSFiT adds per-source white-noise / error nuisance terms — a ``variance``
    parameter and ``*_error`` terms for no-error-bar and upper-limit handling —
    that clutter the corner plot without carrying physical meaning."""
    n = name.lower()
    return n.endswith("_error") or "variance" in n


def _posterior_samples(entry: dict, cap: int = 400) -> dict:
    """{param: [values]} from the model realizations, for SkyPortal's client-side
    corner plot. Physical values (not the 0-1 fractions); MOSFiT's nuisance
    error/variance terms are dropped."""
    import numpy as np

    models = entry.get("models") or []
    reals = (models[0].get("realizations") or []) if models else []
    samples: dict = {}
    for r in reals:
        for name, pdict in (r.get("parameters") or {}).items():
            if not isinstance(pdict, dict) or "value" not in pdict:
                continue
            if _is_nuisance_param(name):
                continue
            v = pdict.get("value")
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue  # skip vector-valued / non-scalar parameters
            if np.isfinite(vf):
                samples.setdefault(name, []).append(round(vf, 5))
    if len(next(iter(samples.values()), [])) > cap:
        idx = np.linspace(0, len(next(iter(samples.values()))) - 1, cap).astype(int)
        samples = {k: [v[i] for i in idx] for k, v in samples.items()}
    return samples


def run_from_skyportal_inputs(
    payload: dict[str, Any], *, outdir: Path | None = None, resource_id: str = "obj", seed: int = 42
) -> dict[str, Any]:
    """Fit a MOSFiT model offline with its own sampler. Returns {status, message,
    source, model_lightcurve, posterior_samples, posterior_medians,
    n_detections, json_result_file}."""
    import numpy as np
    from mosfit.fitter import Fitter

    params = _params(payload)
    source = str(params["source"])
    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="mosfit_osg_"))
    # Absolute so events/output paths survive the chdir into outdir below.
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    event, n_det, band_map = _build_event(payload, resource_id)
    # A fit needs at least 2 finite detections (upper limits / NaN don't count):
    # fail immediately with a clear message rather than letting MOSFiT error out.
    if n_det < 2:
        return {
            "status": "failure",
            "message": f"Not enough detections to fit (need at least 2, have {n_det}).",
            "source": source,
            "n_detections": n_det,
        }
    # The event file's stem is the event name MOSFiT's local fetcher derives.
    event_path = outdir / f"{resource_id}.json"
    event_path.write_text(json.dumps(event))

    iterations = int(params["iterations"])
    burn = int(params.get("burn", iterations // 2))
    # One arg dict splatted into both the constructor and fit_events (mirrors
    # mosfit.main); each takes **kwargs and ignores the other's keys.
    args = dict(
        quiet=True,
        test=False,
        exit_on_prompt=True,  # never block on an interactive prompt in a batch job
        cuda=False,
        events=[str(event_path)],
        models=[source],
        parameter_paths=["parameters.json"],
        iterations=iterations,
        burn=burn,
        num_walkers=params.get("num_walkers"),
        num_temps=1,
        method=str(params["method"]),
        smooth_times=int(params["smooth_times"]),
        extrapolate_time=float(params.get("extrapolate_time", 0.0)),
        output_path=str(outdir),
        write=True,
        quick_save=False,
        save_full_chain=False,
        return_fits=True,
        seed=int(params.get("seed", seed)),
        suffix=resource_id,
    )
    # MOSFiT scaffolds a CWD-relative modules/observables/filters/ dir for its
    # processed-filter cache (normally done by mosfit.main, which we bypass). Run
    # the fit from outdir so that scaffolding lands there — not in the OSG job
    # scratch, which would get transferred back — and pre-create the filters dir.
    prev_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        Path("modules/observables/filters").mkdir(parents=True, exist_ok=True)
        fitter = Fitter(**args)
        entries, _ps, _lnprobs = fitter.fit_events(**args)
    finally:
        os.chdir(prev_cwd)
    entry = entries[0][0]

    result: dict[str, Any] = {
        "status": "success",
        "message": f"mosfit fit complete (model={source}, sampler={params['method']})",
        "source": source,  # the fitted model name (SkyPortal's per-model overlay label)
        "sampler": str(params["method"]),
        "n_detections": n_det,
        "outdir": str(outdir),
    }
    try:
        result["model_lightcurve"] = _model_lightcurve(entry, band_map)
    except Exception as e:  # noqa: BLE001 — overlay data is best-effort
        result["model_lightcurve_error"] = repr(e)
    try:
        result["posterior_samples"] = _posterior_samples(entry)
    except Exception as e:  # noqa: BLE001 — corner data is best-effort
        result["posterior_samples_error"] = repr(e)
    try:
        samples = result.get("posterior_samples") or {}
        result["posterior_medians"] = {
            k: float(np.median(np.asarray(v))) for k, v in samples.items() if v
        }
        rf = outdir / f"{resource_id}_{source}_mosfit_result.json"
        rf.write_text(
            json.dumps(
                {
                    "model": source,
                    "sampler": str(params["method"]),
                    "posterior_medians": result["posterior_medians"],
                    "n_detections": n_det,
                }
            )
        )
        result["json_result_file"] = str(rf)
    except Exception:  # noqa: BLE001
        pass
    return result
