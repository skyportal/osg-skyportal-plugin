"""
periodfind bridge — rotation-period finding for (solar-system) sources.

Shares SkyPortal's analysis payload with the fiesta/redback bridges: a
photometry CSV (``mjd, mag, magerr, filter``) plus ``analysis_parameters``. Runs
a period search (Conditional Entropy by default; AOV / Lomb-Scargle optional)
over the detrended light curve and returns the best period, a phase-folded plot,
and a ``period`` annotation so the source page can fold on it.

Asteroid notes:
- Apparent magnitude carries a slow trend from changing observing geometry
  (helio/geo distance, phase angle). When a predicted/ephemeris magnitude is in
  the CSV we search the residual (obs - predicted); otherwise we subtract each
  band's median, which removes the band offset but not the geometry — so pass a
  predicted mag for clean results.
- Asteroid light curves are usually double-peaked (two extrema per rotation), so
  the rotation period is the folding period or twice it; we report the folding
  period and note the 2x alternative rather than blindly doubling.
- Bad difference-image points show up as multi-mag residuals and seed spurious
  peaks, so we sigma-clip the detrended residual before searching.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# Default grid: periods from ~1 h to ~1 day (asteroid rotation). Overridable via
# analysis_parameters (min_period / max_period in days, samples_per_peak).
_DEFAULT_MIN_PERIOD = 0.04
_DEFAULT_MAX_PERIOD = 1.0
_DEFAULT_SAMPLES_PER_PEAK = 8.0

# Predicted/ephemeris magnitude column, under a few possible names.
_PRED_COLS = ("predicted_mag", "ssmagnr", "ephem_mag")


# Frequency bands (cycles/day) removed in large-scale ZTF period analysis:
# yearly and seasonal harmonics, the ~monthly (lunar) window, and daily
# harmonics. A candidate whose frequency falls in any band is a sampling alias.
_ZTF_ALIAS_FREQS = [
    [0.0025, 0.003],  # 1 yr
    [0.00125, 0.0015],  # 2 yr
    [0.000833, 0.001],  # 3 yr
    [0.000625, 0.00075],  # 4 yr
    [0.0005, 0.0006],  # 5 yr
    [0.005, 0.006],  # 0.5 yr
    [3e-2, 4e-2],  # 30 d
    [3.95, 4.05],  # 0.25 d
    [2.95, 3.05],  # 0.33 d
    [1.95, 2.05],  # 0.5 d
    [0.95, 1.05],  # 1 d
    [0.48, 0.52],  # 2 d
    [0.32, 0.34],  # 3 d
]


def _is_daily_alias(period: float, df_tol: float = 0.0, n_harmonics: int = 0) -> bool:
    """True if the period's frequency (cycles/day) lands in a ZTF alias band."""
    if period <= 0:
        return False
    freq = 1.0 / period
    return any(lo <= freq <= hi for lo, hi in _ZTF_ALIAS_FREQS)


def _num(params: dict, key: str, default, cast=float):
    val = params.get(key)
    if val in (None, ""):
        return default
    try:
        return cast(val)
    except (TypeError, ValueError):
        return default


def _parse_photometry(payload: dict):
    """SkyPortal photometry CSV -> (mjd, mag, magerr, band, predicted) arrays,
    detections only. ``predicted`` is NaN where absent."""
    from astropy.table import Table

    table = Table.read(payload["photometry"], format="ascii.csv")
    pred_col = next((c for c in _PRED_COLS if c in table.colnames), None)
    mjd, mag, err, band, pred = [], [], [], [], []
    for row in table:
        m, e = row["mag"], row["magerr"]
        if np.ma.is_masked(m) or np.ma.is_masked(e):
            continue
        try:
            mf, ef = float(m), float(e)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(mf) and np.isfinite(ef)):
            continue
        mjd.append(float(row["mjd"]))
        mag.append(mf)
        err.append(ef)
        band.append(str(row["filter"]))
        if pred_col is not None and not np.ma.is_masked(row[pred_col]):
            pred.append(float(row[pred_col]))
        else:
            pred.append(np.nan)
    return (
        np.array(mjd),
        np.array(mag),
        np.array(err),
        np.array(band, dtype=object),
        np.array(pred),
    )


def _detrend(mag, band, pred, color_correct=True):
    """Remove the slow geometric trend so only rotation modulation is left.

    Prefers the ephemeris residual (obs - predicted) when enough predicted mags
    are present; otherwise subtracts each band's median.
    """
    resid = np.asarray(mag, dtype=float).copy()
    have = np.isfinite(pred)
    if have.sum() >= max(5, int(0.5 * len(mag))):
        resid[have] = mag[have] - pred[have]
        for b in np.unique(band[~have]):
            sel = (~have) & (band == b)
            if sel.any():
                resid[sel] = mag[sel] - np.median(mag[sel])
        # Zero each band's median: the predicted mag is V but obs are g/r, so a
        # color step survives the residual and beats against the g/r cadence.
        if color_correct:
            for b in np.unique(band):
                sel = band == b
                resid[sel] -= np.median(resid[sel])
        return resid, "ephemeris-residual"
    for b in np.unique(band):
        sel = band == b
        resid[sel] = mag[sel] - np.median(mag[sel])
    return resid, "band-median"


def _finder(algorithm: str):
    """(finder, canonical name). Conditional Entropy is the shape-agnostic
    default; AOV and Lomb-Scargle are available for sinusoidal-ish curves."""
    import periodfind

    algo = (algorithm or "conditional_entropy").lower()
    if algo == "aov":
        return periodfind.AOV(n_phase=10), "aov"
    if algo in ("lombscargle", "ls"):
        return periodfind.LombScargle(), "lomb_scargle"
    return periodfind.ConditionalEntropy(n_phase=10, n_mag=10), "conditional_entropy"


def run_from_skyportal_inputs(inputs: dict, resource_id: str = "obj") -> dict:
    params = inputs.get("analysis_parameters") or {}
    mjd, mag, err, band, pred = _parse_photometry(inputs)
    if len(mjd) < 10:
        return {
            "status": "failure",
            "message": f"Need >=10 detections for a period; have {len(mjd)}.",
        }

    cc = str(params.get("color_correct", "true")).strip().lower() not in ("false", "0", "no")
    resid, detrend_method = _detrend(mag, band, pred, color_correct=cc)

    # Drop gross photometric outliers (bad subtractions show up as multi-mag
    # residuals) that otherwise seed spurious CE minima.
    med = np.median(resid)
    mad = float(np.median(np.abs(resid - med))) * 1.4826
    if mad > 0:
        keep = np.abs(resid - med) <= _num(params, "clip_sigma", 5.0) * mad
        mjd, resid, band = mjd[keep], resid[keep], band[keep]

    import periodfind

    periodfind.set_device(str(params.get("device", "cpu")))
    finder, algo = _finder(str(params.get("algorithm", "conditional_entropy")))

    min_p = _num(params, "min_period", _DEFAULT_MIN_PERIOD)
    max_p = _num(params, "max_period", _DEFAULT_MAX_PERIOD)
    # Uniform in frequency with resolution tied to the baseline (SCoPe convention):
    # df = 1/(samples_per_peak * baseline). Even resolution, no long-period pileup.
    baseline = float(np.max(mjd) - np.min(mjd)) or 1.0
    spp = _num(params, "samples_per_peak", _DEFAULT_SAMPLES_PER_PEAK)
    f_lo, f_hi = 1.0 / max_p, 1.0 / min_p
    df = 1.0 / (spp * baseline)
    freqs = f_lo + df * np.arange(int(np.ceil((f_hi - f_lo) / df)))
    periods = (1.0 / freqs).astype(np.float32)
    period_dts = np.array([0.0], dtype=np.float32)

    times = [np.asarray(mjd, dtype=np.float32)]
    mags = [np.asarray(resid, dtype=np.float32)]

    # Peaks come back best-first, so peaks[0] is the top period regardless of
    # whether the statistic is maximised (AOV/LS) or minimised (CE). Pull many so
    # a real peak survives after alias rejection.
    peaks = finder.calc(times, mags, periods, period_dts, output="peaks", n_peaks=64, center=True)
    curve_peaks = peaks[0] if peaks else []
    if not curve_peaks:
        return {"status": "failure", "message": "No period peaks found."}

    # Drop sidereal/solar-day aliases unless masking is disabled; keep the best
    # surviving peak. Fall back to the raw best if everything is aliased.
    if str(params.get("mask_aliases", "true")).strip().lower() not in ("false", "0", "no"):
        tol = _num(params, "alias_tol", 0.02)
        kept = [p for p in curve_peaks if not _is_daily_alias(float(p.params[0]), tol)]
        curve_peaks = kept or curve_peaks

    best = curve_peaks[0]
    # The best phase-folding period. An asteroid light curve is usually
    # double-peaked, so the rotation period is this one (folds both maxima) or,
    # if the fit locked onto a single peak, twice it. Report the folding period;
    # note the 2x alternative rather than blindly doubling.
    period = float(best.params[0])

    plot_file = _phase_fold_plot(mjd, resid, band, period, resource_id, algo)

    results = {
        "period_days": period,
        "period_hours": period * 24.0,
        "period_double_peaked_hours": 2.0 * period * 24.0,
        "statistic_value": float(best.value),
        "algorithm": algo,
        "detrend": detrend_method,
        "n_detections": int(len(mjd)),
        "period_range_days": [min_p, max_p],
        "top_periods_days": [float(p.params[0]) for p in curve_peaks[:5]],
    }
    # The source page folds on an annotation whose data has a `period` key (days).
    annotations = [
        {
            "origin": "periodfind",
            "data": {
                "period": period,
                "period_algorithm": algo,
                "period_statistic": float(best.value),
            },
        }
    ]
    return {
        "status": "success",
        "message": f"P={period * 24:.3f} h ({algo})",
        "results": results,
        "annotations": annotations,
        "plot_file": str(plot_file) if plot_file else None,
    }


def _phase_fold_plot(mjd, resid, band, period, resource_id, algo):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    phase = (np.asarray(mjd) % period) / period
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in np.unique(band):
        sel = band == b
        # Two cycles, for readability of the folded shape.
        ax.plot(
            np.concatenate([phase[sel], phase[sel] + 1.0]),
            np.concatenate([resid[sel], resid[sel]]),
            "o",
            ms=3,
            alpha=0.7,
            label=str(b),
        )
    ax.invert_yaxis()  # brighter up
    ax.set_xlabel("phase")
    ax.set_ylabel("detrended mag")
    ax.set_title(f"{resource_id}  P_rot={period * 24:.3f} h  ({algo})")
    ax.legend(fontsize=8)
    out = Path(tempfile.gettempdir()) / f"periodfind_{resource_id}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def count_detections(payload: dict) -> int:
    """Finite-magnitude detection count (mirrors the fiesta bridge's guard)."""
    return int(len(_parse_photometry(payload)[0]))
