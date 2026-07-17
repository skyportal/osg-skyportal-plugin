"""redback-jax fit backend for the OSG SkyPortal plugin.

Parallel to ``fiesta_bridge.py``: same SkyPortal payload (photometry + redshift +
free-form ``analysis_parameters``), same return contract — ``{status, message,
source, model_lightcurve, posterior_medians, posterior_samples, n_detections}`` —
so SkyPortal's overlay and the plugin callback handle both backends identically.

Selected per request via ``analysis_parameters.backend = "redback"`` (the wrapper
dispatches). redback-jax and JAX are imported lazily so this module stays
importable in a bare test env (matches fiesta_bridge).

Fit: redback-jax's class-based ``Likelihood`` builds a JIT-compiled multi-band
log-likelihood; we run it through ``run_nested_sampling`` (SMC tempering with NUTS
mutation, on *mainline* blackjax — same SMC approach fiesta uses, gives a
log-evidence, avoids the handley-lab NS fork whose old JAX API is incompatible).
The overlay light curve is built at the posterior medians (concrete values, so a
``PrecomputedSpectraSource`` runs eagerly there).

First increment: Arnett wired end-to-end; other models add a MODELS entry.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

DEFAULTS = {"source": "arnett", "n_particles": 500}
_CURVE_NPTS = 100
# redback/jax_supernovae bandpass registry (register_all_bandpasses). SkyPortal
# filters are mapped onto these; observations in unmappable bands are dropped
# (real sources carry multi-survey photometry with many filter names).
_SUPPORTED = {
    "ztfg",
    "ztfr",
    "c",
    "o",
    "g",
    "r",
    "i",
    "z",
    "H",
    "bessellb",
    "bessellv",
    "bessellr",
    "besselli",
    "bessellux",
}
_BAND_MAP: dict[str, str] = {
    "ztfi": "i",
    "sdssg": "g",
    "sdssr": "r",
    "sdssi": "i",
    "sdssz": "z",
    "sdssu": "bessellux",
    "ps1__g": "g",
    "ps1__r": "r",
    "ps1__i": "i",
    "ps1__z": "z",
    "atlasc": "c",
    "atlaso": "o",
}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _band(filt: str) -> str:
    return _BAND_MAP.get(filt, filt)


# ---------------------------------------------------------------------------
# Model registry. ``model`` is the redback-jax model name for the class-based
# ``Likelihood`` (the JIT-able fit); ``overlay_builder`` reconstructs the light
# curve at the medians. ``fit_params`` are sampled; ``fixed`` are held constant
# (redshift/lum_dist filled from the payload); ``t0_key`` names the epoch param.
# ---------------------------------------------------------------------------
def _model_registry() -> dict[str, dict]:
    from redback_jax.sources import PrecomputedSpectraSource  # lazy

    return {
        "arnett": {
            "model": "arnett_spectra",
            "t0_key": "t0",
            "fit_params": ["f_nickel", "mej", "vej"],
            "fixed": {"temperature_floor": 5000.0, "kappa": 0.07, "kappa_gamma": 0.1},
            "priors": {
                "t0": None,  # data-anchored (min detection mjd +/- window)
                "f_nickel": (0.02, 0.4),
                "mej": (0.3, 5.0),
                "vej": (3000.0, 12000.0),
            },
            # from_arnett_model derives lum_dist from redshift, so no lum_dist here.
            "overlay_builder": PrecomputedSpectraSource.from_arnett_model,
            "overlay_fixed": ["temperature_floor", "kappa", "kappa_gamma"],
        },
    }


def _parse_photometry(payload: dict) -> tuple[list, list, list, list, int, float]:
    """SkyPortal photometry CSV -> flat (times_mjd, mags, magerrs, bands), plus
    (n_detections, min detection mjd). Detections only for now."""
    import numpy as np
    from astropy.table import Table

    table = Table.read(payload["photometry"], format="ascii.csv")
    times, mags, errs, bands = [], [], [], []
    for row in table:
        mag, magerr = row["mag"], row["magerr"]
        if np.ma.is_masked(mag) or np.ma.is_masked(magerr):
            continue
        try:
            magf, errf = float(mag), float(magerr)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(magf) and np.isfinite(errf)):
            continue
        b = _band(str(row["filter"]))
        if b not in _SUPPORTED:  # no bandpass for this filter -> skip
            continue
        times.append(float(row["mjd"]))
        mags.append(magf)
        errs.append(errf)
        bands.append(b)
    return times, mags, errs, bands, len(times), (min(times) if times else 0.0)


def _resolve_redshift(payload: dict) -> float | None:
    from astropy.table import Table

    src = payload.get("redshift")
    if src is None:
        return None
    try:
        table = Table.read(src, format="ascii.csv")
    except Exception:  # noqa: BLE001 — a bare value rather than a CSV
        try:
            return float(src)
        except (TypeError, ValueError):
            return None
    if len(table) == 0 or "redshift" not in table.colnames:
        return None
    import numpy as np

    v = table["redshift"][0]
    if np.ma.is_masked(v):
        return None
    try:
        z = float(v)
    except (TypeError, ValueError):
        return None
    return z if z > 0 else None


def count_detections(payload: dict) -> int:
    """Finite detections a redback fit would use — lets the wrapper fail fast."""
    return _parse_photometry(payload)[4]


def _prior_bounds(source: str, overrides: dict, min_mjd: float) -> dict:
    reg = _model_registry()[source]
    bounds: dict[str, tuple[float, float]] = {}
    for name, b in reg["priors"].items():
        if name in overrides:
            lo, hi = overrides[name]
        elif name == reg["t0_key"] and b is None:
            lo, hi = min_mjd - 15.0, min_mjd + 10.0
        else:
            lo, hi = b
        bounds[name] = (float(lo), float(hi))
    return bounds


def _model_lightcurve(source: str, fixed: dict, medians: dict, unique_bands, t_grid_days) -> dict:
    """Per-filter model mags at the posterior medians over ``t_grid_days`` (source
    days from t0), in the overlay shape {filter: [[mjd, med, lo, hi]]}. Medians are
    concrete, so the source builder runs eagerly here."""
    import numpy as np

    reg = _model_registry()[source]
    builder = reg["overlay_builder"]
    t0 = float(medians.get(reg["t0_key"], 0.0))
    src = builder(
        redshift=fixed["redshift"],
        **{k: fixed[k] for k in reg["overlay_fixed"]},
        **{p: float(medians[p]) for p in reg["fit_params"]},
    )
    mjds = np.asarray(t_grid_days) + t0
    curve: dict[str, list] = {}
    for filt in unique_bands:
        mags = np.asarray(src.bandmag({"amplitude": 1.0}, filt, t_grid_days))
        curve[filt] = [[float(m), float(mag), float(mag), float(mag)] for m, mag in zip(mjds, mags)]
    return curve


def run_from_skyportal_inputs(
    payload: dict[str, Any],
    *,
    outdir: Path | None = None,
    resource_id: str = "obj",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a redback-jax model with SMC nested sampling. Same return contract as
    fiesta_bridge.run_from_skyportal_inputs."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from redback_jax.inference import Likelihood, Prior, Uniform
    from redback_jax.inference.sampler import run_nested_sampling
    from redback_jax.transient import Transient
    from redback_jax.utils import luminosity_distance_cm

    params = _params(payload)
    source = str(params["source"])
    if source not in _model_registry():
        return {
            "status": "failure",
            "source": source,
            "message": f"redback backend: unknown model {source!r}",
        }

    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="redback_osg_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    times, mags, errs, bands, n_det, min_mjd = _parse_photometry(payload)
    if n_det < 2:
        return {
            "status": "failure",
            "source": source,
            "n_detections": n_det,
            "message": f"Not enough detections to fit (need at least 2, have {n_det}).",
        }

    redshift = _resolve_redshift(payload)
    if redshift is None:
        redshift = float(params.get("redshift", 0.0))
    reg = _model_registry()[source]
    fixed = {**reg["fixed"], **(params.get("fixed_params") or {})}
    fixed["redshift"] = max(redshift, 1e-4)
    fixed["lum_dist"] = float(luminosity_distance_cm(fixed["redshift"]))

    transient = Transient(
        time=np.array(times),
        y=np.array(mags),
        y_err=np.array(errs),
        bands=bands,
        data_mode="magnitude",
        name=str(resource_id),
        redshift=redshift,
    )
    bounds = _prior_bounds(source, params.get("prior_ranges") or {}, min_mjd)
    prior = Prior([Uniform(lo, hi, name=n) for n, (lo, hi) in bounds.items()])
    likelihood = Likelihood(
        model=reg["model"],
        transient=transient,
        fixed_params=fixed,
        t0_key=reg["t0_key"],
    )
    # Class likelihood is a JIT'd (ndarray in prior.names order) -> logL; wrap it
    # for run_nested_sampling's dict convention.
    log_like_arr = likelihood._make_log_likelihood(prior)
    names = list(prior.names)

    def loglike(p: dict):
        return log_like_arr(jnp.array([p[n] for n in names]))

    result = run_nested_sampling(
        loglike,
        bounds,
        n_particles=int(params.get("n_particles", 500)),
        rng_key=jax.random.PRNGKey(int(seed)),
        verbose=False,
    )

    medians = {name: float(np.median(np.asarray(samp))) for name, samp in result.samples.items()}
    t_grid = jnp.linspace(0.1, 60.0, _CURVE_NPTS)
    unique_bands = list(dict.fromkeys(bands))

    out: dict[str, Any] = {
        "status": "success",
        "message": f"redback fit complete (model={source}, sampler=nested-smc)",
        "source": source,
        "n_detections": n_det,
        "posterior_medians": medians,
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
    }
    try:
        out["model_lightcurve"] = _model_lightcurve(source, fixed, medians, unique_bands, t_grid)
    except Exception as e:  # noqa: BLE001 — a fit can succeed even if the overlay fails
        out["model_lightcurve"] = None
        out["message"] += f" (overlay unavailable: {e})"
    out["posterior_samples"] = {
        name: [float(x) for x in np.asarray(samp)] for name, samp in result.samples.items()
    }

    # The wrapper attaches this file as the SkyPortal "results" blob (same
    # contract as fiesta_bridge); without json_result_file the fit's numbers
    # never reach SkyPortal.
    result_file = outdir / "redback_result.json"
    result_file.write_text(
        json.dumps(
            {
                "source": source,
                "medians": medians,
                "log_evidence": out["log_evidence"],
                "log_evidence_error": out["log_evidence_error"],
                "n_detections": n_det,
            }
        )
    )
    out["json_result_file"] = str(result_file)
    return out
