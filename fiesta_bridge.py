"""
SkyPortal-AnalysisService → Fiesta bridge (pure Fiesta, no NMMA).

Takes a SkyPortal analysis payload (photometry + redshift + free-form
``analysis_parameters``), fits a Fiesta model — a surrogate (Bu2025_MLP, ...) or
an analytic model (AfterglowModel, ArnettModel, ...) — with Fiesta's native JAX
sampler (default ``blackjax-smc``), and returns a result dict including a
per-filter ``model_lightcurve`` for overlaying the fit on SkyPortal's photometry
plot. Runs inside the fiesta runtime image on an OSG worker.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

# Defaults, overridable per-call via ``analysis_parameters`` in the payload.
DEFAULTS = {
    "source": "Bu2025_MLP",
    "sampler": "blackjax-smc",
    "tmin": 0.1,
    "tmax": 30.0,
    "t0_offset": 2.0,
}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _resolve_redshift(payload: dict) -> float | None:
    src = payload.get("redshift")
    if src is None:
        return None
    from astropy.table import Table

    table = Table.read(src, format="ascii.csv")
    if len(table) == 0 or "redshift" not in table.colnames:
        return None
    return float(table["redshift"][0])


def _write_data_file(payload: dict, outdir: Path) -> tuple[Path, float, list[str]]:
    """Convert SkyPortal photometry to the `time filter mag magerr` data file
    fiesta's ``load_event_data`` reads. Returns (path, min mjd, distinct filters)."""
    from astropy.table import Table

    table = Table.read(payload["photometry"], format="ascii.csv")
    data_path = outdir / "data.dat"
    filters: list[str] = []
    mjds: list[float] = []
    with data_path.open("w") as fh:
        for row in table:
            mag, magerr = row["mag"], row["magerr"]
            # SkyPortal sends non-detections (null mag/magerr); fiesta wants
            # finite detections, so skip them.
            if np.ma.is_masked(mag) or np.ma.is_masked(magerr):
                continue
            try:
                magf, errf = float(mag), float(magerr)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(magf) and np.isfinite(errf)):
                continue
            mjd = float(row["mjd"])
            mjds.append(mjd)
            iso = Time(mjd, format="mjd").isot
            filt = str(row["filter"])  # full survey name (ztfg, sdssg, ...)
            if filt not in filters:
                filters.append(filt)
            fh.write(f"{iso} {filt} {magf} {errf}\n")
    return data_path, (min(mjds) if mjds else 0.0), filters


def _default_param_range(name: str) -> tuple[float, float]:
    """Analytic-model parameter ranges by name pattern; the prior fallback when
    none is supplied. Overridable via analysis_parameters['prior_ranges']."""
    nl = name.lower()
    if "amp_mag" in nl or "base_mag" in nl:
        return (14.0, 26.0)
    if "log10" in nl:
        return (-3.0, 2.0)
    if "t0" in nl or "time" in nl or nl.startswith("t_") or "tau" in nl:
        return (-1.0, 5.0)
    if "alpha" in nl or "beta" in nl or "index" in nl or "gamma" in nl:
        return (0.0, 3.0)
    if "temp" in nl:
        return (3000.0, 15000.0)
    return (0.01, 1.0)


def _build_fiesta_model(source, em_transient_class, filters):
    """Resolve `source` to a fiesta model: an analytic class by name, else a
    surrogate (BullaFlux for KN, AfterglowFlux for GRB). Returns (model, kind)."""
    import importlib

    from fiesta.inference.analytical_models.base import AnalyticalModel

    for mod in (
        "phenomenological_models",
        "kilonova_models",
        "supernova_models",
        "tde_models",
        "shock_powered_models",
    ):
        m = importlib.import_module(f"fiesta.inference.analytical_models.{mod}")
        cls = getattr(m, source, None)
        if (
            isinstance(cls, type)
            and issubclass(cls, AnalyticalModel)
            and cls is not AnalyticalModel
        ):
            return cls(filters=filters), "analytic"

    from fiesta.inference.lightcurve_model import AfterglowFlux, BullaFlux

    if (em_transient_class or "") == "fiesta_grb":
        return AfterglowFlux(name=source, filters=filters), "surrogate"
    return BullaFlux(name=source, filters=filters), "surrogate"


def _build_fiesta_prior(model, kind, overrides, sample_distance):
    from fiesta.inference.prior import ConstrainedPrior, Sine, Uniform

    def make(name, lo, hi):
        lo, hi = overrides.get(name, (lo, hi))
        cls = Sine if "inclination" in name.lower() else Uniform
        return cls(xmin=float(lo), xmax=float(hi), naming=[name])

    pl = []
    if kind == "surrogate":
        fm = getattr(model, "fiesta_model", model)
        for name, dist in getattr(fm, "parameter_distributions", {}).items():
            pl.append(make(name, float(dist[0]), float(dist[1])))
    else:
        for name in model.parameter_names:
            if name in ("redshift", "luminosity_distance"):
                continue
            pl.append(make(name, *_default_param_range(name)))
    if sample_distance:
        pl.append(make("luminosity_distance", 1.0, 1000.0))
    return ConstrainedPrior(pl)


def _fiesta_model_lightcurve(model, posterior, fixed, filters, trigger_time, n_samples=50):
    """Per-filter median + 16/84 apparent-mag curves on an MJD grid, via the
    model's own predict() over posterior samples (for the photometry overlay)."""
    keys = [k for k in posterior if k not in ("log_prob", "log_likelihood")]
    n = len(posterior[keys[0]])
    idx = np.linspace(0, n - 1, min(n_samples, n)).astype(int)
    z = float(fixed.get("redshift", 0.0))
    acc: dict[str, list] = {f: [] for f in filters}
    times = None
    for i in idx:
        x = {k: float(np.asarray(posterior[k])[i]) for k in keys}
        x.update(fixed)
        t, mags = model.predict(x)  # (source-frame times, {filter: apparent mag})
        times = np.asarray(t)
        for f in filters:
            if f in mags:
                acc[f].append(np.asarray(mags[f], dtype=float))
    out: dict[str, list] = {}
    if times is None:
        return out
    mjd = trigger_time + times * (1.0 + z)  # source -> observer frame
    for f, samples in acc.items():
        if not samples:
            continue
        A = np.vstack(samples)
        med = np.nanmedian(A, axis=0)
        lo = np.nanpercentile(A, 16, axis=0)
        hi = np.nanpercentile(A, 84, axis=0)
        out[f] = [
            [float(mjd[i]), float(med[i]), float(lo[i]), float(hi[i])]
            for i in range(len(times))
            if np.isfinite(med[i])
        ]
    return out


def run_from_skyportal_inputs(
    payload: dict[str, Any], *, outdir: Path | None = None, resource_id: str = "obj", seed: int = 42
) -> dict[str, Any]:
    """Fit a fiesta model with its native JAX sampler (default blackjax-smc).
    Returns {status, message, model_lightcurve, posterior_medians, json_result_file}."""
    import jax
    from fiesta.inference.fiesta import Fiesta
    from fiesta.inference.likelihood import EMLikelihood
    from fiesta.utils import load_event_data

    params = _params(payload)
    source = str(params["source"])
    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="fiesta_osg_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path, t0, _ = _write_data_file(payload, outdir)
    data = load_event_data(str(data_path))
    filters = list(data.keys())
    model, kind = _build_fiesta_model(source, params.get("em_transient_class"), filters)

    redshift = _resolve_redshift(payload)
    fixed: dict[str, float] = {
        "redshift": float(redshift if redshift is not None else params.get("redshift", 0.0))
    }
    # Physics models need a distance (phenomenological use amp_mag directly).
    sample_distance = bool(params.get("sample_distance", False))
    if not sample_distance:
        if redshift and redshift > 0:
            from astropy.cosmology import Planck18 as cosmo

            fixed["luminosity_distance"] = float(cosmo.luminosity_distance(redshift).value)
        else:
            fixed["luminosity_distance"] = float(params.get("luminosity_distance", 100.0))

    prior = _build_fiesta_prior(model, kind, params.get("prior_ranges") or {}, sample_distance)
    trigger_time = float(params.get("trigger_time", t0 - float(params.get("t0_offset", 2.0))))
    sampler = str(params.get("sampler", "blackjax-smc"))

    # Clamp the fit window to where the model is valid in the observer frame
    # (e.g. ShockCooling is only defined to ~3.5 d); otherwise EMLikelihood
    # errors on data points past the model's time array.
    data_tmin = float(params.get("tmin", 0.1))
    data_tmax = float(params.get("tmax", 30.0))
    try:
        mt = np.asarray(model.times) * (1.0 + float(fixed.get("redshift", 0.0)))
        data_tmin = max(data_tmin, float(mt.min()))
        data_tmax = min(data_tmax, float(mt.max()))
    except Exception:  # noqa: BLE001 — fall back to the requested window
        pass

    likelihood = EMLikelihood(
        model,
        data,
        trigger_time=trigger_time,
        data_tmin=data_tmin,
        data_tmax=data_tmax,
        fixed_params=fixed,
    )
    # Memory-heavy models (e.g. CSMInteraction, with a 500-pt internal grid)
    # can OOM the GPU at the 8000-particle default; expose n_particles to dial down.
    sampler_kwargs = {}
    if sampler == "blackjax-smc" and params.get("n_particles"):
        sampler_kwargs["n_particles"] = int(params["n_particles"])
    fiesta = Fiesta(
        likelihood,
        prior,
        outdir=str(outdir),
        sampler=sampler,
        seed=int(params.get("seed", seed)),
        **sampler_kwargs,
    )
    fiesta.sample(jax.random.PRNGKey(int(params.get("seed", seed))))
    posterior = fiesta.posterior_samples

    result: dict[str, Any] = {
        "status": "success",
        "message": f"fiesta fit complete (model={source}, sampler={sampler})",
        "sampler": sampler,
        "outdir": str(outdir),
    }
    try:
        result["model_lightcurve"] = _fiesta_model_lightcurve(
            model, posterior, fixed, filters, trigger_time
        )
    except Exception as e:  # noqa: BLE001 — overlay data is best-effort
        result["model_lightcurve_error"] = repr(e)
    try:
        result["posterior_medians"] = {
            k: float(np.median(np.asarray(v)))
            for k, v in posterior.items()
            if k not in ("log_prob", "log_likelihood")
        }
        rf = outdir / f"{resource_id}_{source}_fiesta_result.json"
        rf.write_text(
            json.dumps(
                {
                    "model": source,
                    "sampler": sampler,
                    "posterior_medians": result["posterior_medians"],
                    "n_posterior": int(len(posterior[next(iter(posterior))])),
                }
            )
        )
        result["json_result_file"] = str(rf)
    except Exception:  # noqa: BLE001
        pass
    return result
