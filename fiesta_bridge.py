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

# numpy + astropy are imported lazily inside the functions that use them: this
# module is shipped to (and only runs in) the fiesta image, but the plugin's
# tests import it in a lightweight env without those deps.

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
    # A missing/masked redshift comes through as a masked element, and float() of
    # it is NaN. Return None (not NaN) so callers fall back to a z=0 / fixed-distance
    # fit — otherwise NaN propagates into the observer-frame mjd and the
    # model_lightcurve comes back empty.
    try:
        value = float(table["redshift"][0])
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # value != value is True only for NaN


def _write_data_file(payload: dict, outdir: Path) -> tuple[Path, float, list[str], int]:
    """Convert SkyPortal photometry to the `time filter mag magerr` data file
    fiesta's ``load_event_data`` reads. Detections are written with their error;
    selected upper limits are written as `mag=limiting_mag err=inf`, which fiesta
    treats as a censored (truncated-Gaussian) constraint. Returns
    (path, min detection mjd, distinct filters, n_detections)."""
    import numpy as np
    from astropy.table import Table
    from astropy.time import Time

    # SkyPortal flux zeropoint (µJy AB): mag = -2.5 log10(flux) + ZP. NSIGMA is
    # the fallback detection threshold used only when limiting_mag isn't exported.
    PHOT_ZP, NSIGMA = 23.9, 5.0

    table = Table.read(payload["photometry"], format="ascii.csv")
    cols = table.colnames

    def _limit_mag(row) -> float | None:
        """Upper-limit magnitude for a non-detection: prefer SkyPortal's
        limiting_mag, else derive it from fluxerr. None if neither is usable."""
        if "limiting_mag" in cols and not np.ma.is_masked(row["limiting_mag"]):
            v = float(row["limiting_mag"])
            if np.isfinite(v):
                return v
        if "fluxerr" in cols and not np.ma.is_masked(row["fluxerr"]):
            fe = float(row["fluxerr"])
            if np.isfinite(fe) and fe > 0:
                return -2.5 * np.log10(NSIGMA * fe) + PHOT_ZP
        return None

    # SkyPortal sends non-detections with a masked mag; split them out so we can
    # keep only the informative upper limits below.
    dets: list[tuple[float, str, float, float]] = []
    nondets: list[tuple[float, str, float]] = []
    for row in table:
        filt, mjd = str(row["filter"]), float(row["mjd"])
        mag, magerr = row["mag"], row["magerr"]
        if not (np.ma.is_masked(mag) or np.ma.is_masked(magerr)):
            try:
                magf, errf = float(mag), float(magerr)
            except (TypeError, ValueError):
                magf = errf = float("nan")
            if np.isfinite(magf) and np.isfinite(errf):
                dets.append((mjd, filt, magf, errf))
                continue
        lim = _limit_mag(row)
        if lim is not None:
            nondets.append((mjd, filt, lim))

    # Keep the single most-recent upper limit before the first detection (pins
    # the explosion epoch — "not there yet") plus every upper limit within the
    # detection window; drop the rest (older pre-detection / post-peak limits add
    # little and would just inflate the fit).
    kept: list[tuple[float, str, float]] = []
    if dets:
        first_det = min(d[0] for d in dets)
        last_det = max(d[0] for d in dets)
        # Only in bands we actually fit: a lone UL in an otherwise-undetected
        # band can't constrain its amp_mag/base_mag and just yields a garbage
        # model curve, so restrict upper limits to filters that have detections.
        det_filters = {d[1] for d in dets}
        usable = [n for n in nondets if n[1] in det_filters]
        pre = [n for n in usable if n[0] < first_det]
        if pre:
            kept.append(max(pre, key=lambda n: n[0]))
        kept += [n for n in usable if first_det <= n[0] <= last_det]

    data_path = outdir / "data.dat"
    filters: list[str] = []
    with data_path.open("w") as fh:
        for mjd, filt, mag, err in sorted(dets):
            if filt not in filters:
                filters.append(filt)
            fh.write(f"{Time(mjd, format='mjd').isot} {filt} {mag} {err}\n")
        for mjd, filt, lim in sorted(kept):
            if filt not in filters:
                filters.append(filt)
            fh.write(f"{Time(mjd, format='mjd').isot} {filt} {lim} inf\n")

    min_det_mjd = min((d[0] for d in dets), default=0.0)
    return data_path, min_det_mjd, filters, len(dets)


def count_detections(payload: dict) -> int:
    """Number of finite detections fiesta would actually fit from a SkyPortal
    payload — lets callers fail a request fast (and once) when there are none."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fiesta_det_") as d:
        _, _, _, n_det = _write_data_file(payload, Path(d))
    return n_det


def _default_param_range(
    name: str, mag_bright: float | None = None, mag_faint: float | None = None
) -> tuple[float, float]:
    """Analytic-model parameter ranges by name pattern. Kept physical (wide flat
    priors let the sampler wander and never constrain); the apparent-magnitude
    amplitudes are anchored to the observed data when available. Overridable via
    analysis_parameters['prior_ranges']."""
    nl = name.lower()
    if "amp_mag" in nl:  # peak apparent magnitude — near the brightest detection
        if mag_bright is not None:
            return (mag_bright - 1.5, mag_faint + 1.5)
        return (14.0, 26.0)
    if "base_mag" in nl:  # host-subtracted photometry -> faint/negligible baseline
        if mag_faint is not None:
            return (mag_faint + 1.5, mag_faint + 6.0)
        return (22.0, 30.0)
    # EvolvingBlackbody physical scales (log10 K / log10 cm / days).
    if "log10_temperature" in nl:
        return (3.3, 4.6)  # ~2000-40000 K
    if "log10_radius" in nl:
        return (13.0, 16.0)  # ~1e13-1e16 cm
    if "peak_time" in nl:
        return (0.0, 30.0)  # days
    # Physical timescales in log10 days (after Villar et al.): rise ~0.3-16 d,
    # fall ~2-160 d — far tighter than a blanket [-3, 2].
    if "log10" in nl and "rise" in nl:
        return (-0.5, 1.2)
    if "log10" in nl and "fall" in nl:
        return (0.3, 2.2)
    if "log10" in nl:
        return (-1.5, 2.0)
    # Peak time relative to the trigger; the old [-1, 5] railed against the prior
    # for anything but the fastest transients.
    if "t0" in nl:
        return (-5.0, 30.0)
    if "time" in nl or nl.startswith("t_") or "tau" in nl:
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
    from fiesta.inference.analytical_models.phenomenological_models import (
        PhenomenologicalModel,
    )

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
            # Only amp_mag-based PhenomenologicalModel subclasses (Bazin/Villar/
            # ...) skip distance; physics models — including EvolvingBlackbody,
            # which lives in the phenomenological module but computes L+R through
            # a blackbody SED — need a distance.
            kind = "phenomenological" if issubclass(cls, PhenomenologicalModel) else "analytic"
            return cls(filters=filters), kind

    from fiesta.inference.lightcurve_model import AfterglowFlux, BullaFlux

    if (em_transient_class or "") == "fiesta_grb":
        return AfterglowFlux(name=source, filters=filters), "surrogate"
    return BullaFlux(name=source, filters=filters), "surrogate"


def _build_fiesta_prior(
    model, kind, overrides, sample_distance, band_mags=None, global_mags=(None, None)
):
    from fiesta.inference.prior import ConstrainedPrior, Sine, Uniform

    def make(name, lo, hi):
        lo, hi = overrides.get(name, (lo, hi))
        cls = Sine if "inclination" in name.lower() else Uniform
        return cls(xmin=float(lo), xmax=float(hi), naming=[name])

    def mags_for(name):
        # amp_mag_{band}/base_mag_{band} anchor to that band's own observed span
        # (a band that doesn't sample the faint tail otherwise has a degenerate
        # amplitude/baseline); fall back to the global span.
        for prefix in ("amp_mag_", "base_mag_"):
            if name.startswith(prefix):
                return (band_mags or {}).get(name[len(prefix) :], global_mags)
        return (None, None)

    pl = []
    if kind == "surrogate":
        fm = getattr(model, "fiesta_model", model)
        for name, dist in getattr(fm, "parameter_distributions", {}).items():
            pl.append(make(name, float(dist[0]), float(dist[1])))
    else:
        for name in model.parameter_names:
            if name in ("redshift", "luminosity_distance"):
                continue
            pl.append(make(name, *_default_param_range(name, *mags_for(name))))
    if sample_distance:
        pl.append(make("luminosity_distance", 1.0, 1000.0))
    return ConstrainedPrior(pl)


def _fiesta_model_lightcurve(model, posterior, fixed, filters, trigger_time, n_samples=50):
    """Per-filter median + 16/84 apparent-mag curves on an MJD grid, via the
    model's own predict() over posterior samples (for the photometry overlay)."""
    import numpy as np

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
    import numpy as np
    from fiesta.inference.fiesta import Fiesta
    from fiesta.inference.likelihood import EMLikelihood
    from fiesta.utils import load_event_data

    params = _params(payload)
    source = str(params["source"])
    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="fiesta_osg_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path, t0, _, n_det = _write_data_file(payload, outdir)
    # A fit needs at least 2 finite detections (all upper limits / NaN /
    # negative flux don't count): fail immediately with a clear message rather
    # than letting each model error out on too-little data.
    if n_det < 2:
        return {
            "status": "failure",
            "message": f"Not enough detections to fit (need at least 2, have {n_det}).",
            "source": source,
            "n_detections": n_det,
        }
    data = load_event_data(str(data_path))
    filters = list(data.keys())
    model, kind = _build_fiesta_model(source, params.get("em_transient_class"), filters)

    # Observed apparent-magnitude span per band (detections have finite error;
    # upper limits are inf) — used to anchor each band's amp/base priors.
    band_mags: dict = {}
    for _f, _arr in data.items():
        _ms = [float(r[1]) for r in np.asarray(_arr) if np.isfinite(r[2])]
        if _ms:
            band_mags[_f] = (min(_ms), max(_ms))
    _all = [m for bm in band_mags.values() for m in bm]
    global_mags = (min(_all), max(_all)) if _all else (None, None)

    redshift = _resolve_redshift(payload)
    have_z = redshift is not None and redshift > 0
    # With no measured redshift we can't fix the distance, so SAMPLE over it: the
    # prior then includes luminosity_distance (1-1000 Mpc) as a free parameter and
    # model.predict() scales each posterior draw by the sampled distance. An explicit
    # analysis_parameters['sample_distance'] still wins. Redshift is fixed to 0 in
    # this case (z=0 observer frame for the mjd grid); the distance carries the fit.
    # Phenomenological models use apparent-mag amplitudes directly and have no
    # distance parameter; sampling one just adds an unconstrained nuisance
    # dimension that wrecks convergence. Only non-phenomenological models sample.
    sample_distance = kind != "phenomenological" and bool(params.get("sample_distance", not have_z))
    fixed: dict[str, float] = {
        "redshift": float(redshift) if have_z else float(params.get("redshift", 0.0))
    }
    # Physics models need a distance (phenomenological use amp_mag directly). When
    # sampling distance we leave luminosity_distance out of `fixed` so it stays free.
    if not sample_distance:
        if have_z:
            from astropy.cosmology import Planck18 as cosmo

            fixed["luminosity_distance"] = float(cosmo.luminosity_distance(redshift).value)
        else:
            fixed["luminosity_distance"] = float(params.get("luminosity_distance", 100.0))

    prior = _build_fiesta_prior(
        model,
        kind,
        params.get("prior_ranges") or {},
        sample_distance,
        band_mags,
        global_mags,
    )
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

    # Systematic error added in quadrature to each point. Surrogate (kilonova)
    # models want ~0.3 mag for model inadequacy; the analytic/phenomenological
    # SN fits trust ZTF's ~0.1 mag data far more.
    default_error_budget = 0.3 if kind == "surrogate" else 0.05
    likelihood = EMLikelihood(
        model,
        data,
        trigger_time=trigger_time,
        data_tmin=data_tmin,
        data_tmax=data_tmax,
        fixed_params=fixed,
        error_budget=float(params.get("error_budget", default_error_budget)),
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

    try:
        n_detections = int(sum(len(v) for v in data.values()))
    except Exception:  # noqa: BLE001
        n_detections = None
    result: dict[str, Any] = {
        "status": "success",
        "message": f"fiesta fit complete (model={source}, sampler={sampler})",
        "source": source,  # the fitted model name (for SkyPortal's per-model overlay label)
        "sampler": sampler,
        "n_detections": n_detections,  # detections the fit used (for run versioning)
        "outdir": str(outdir),
    }
    try:
        result["model_lightcurve"] = _fiesta_model_lightcurve(
            model, posterior, fixed, filters, trigger_time
        )
    except Exception as e:  # noqa: BLE001 — overlay data is best-effort
        result["model_lightcurve_error"] = repr(e)
    # Thinned posterior samples for SkyPortal's client-side Plotly corner plot
    # ({param: [values]}). No arviz/netCDF — the frontend renders the scatter
    # matrix. Capped at 400 draws + rounded to keep the stored blob small (one
    # joblib file per analysis accumulates on the shared volume).
    try:
        pkeys = [k for k in posterior if k not in ("log_prob", "log_likelihood")]
        n = len(np.asarray(posterior[pkeys[0]]))
        idx = np.linspace(0, n - 1, min(400, n)).astype(int)
        result["posterior_samples"] = {
            k: [round(float(x), 5) for x in np.asarray(posterior[k])[idx]] for k in pkeys
        }
    except Exception as e:  # noqa: BLE001 — corner data is best-effort
        result["posterior_samples_error"] = repr(e)
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
