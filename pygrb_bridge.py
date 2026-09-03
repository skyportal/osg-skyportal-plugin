"""SkyPortal-AnalysisService -> coherent GW targeted-search bridge (PyCBC).

An optical fast-transient trigger (sky position + a T0 estimate, ideally the KN
light-curve fit's T0 posterior) drives a PyGRB-style coherent matched-filter
search of GW strain around that time and sky location -- the analysis Marion
Pillas & Noah Jamsin validated on GW170817 (coherent SNR ~28).

Two phases share this bridge:
  * Phase 1 (here): a self-contained network coherent-SNR search over *public*
    GWOSC strain, using PyCBC's Python API and a single template (from the KN-
    inferred masses, or BNS defaults). Recovers the on-source peak and returns a
    coherent SNR + a trigger-distribution plot -- enough to prove the
    trigger -> search -> result flow end to end.
  * Phase 2 (later): swap ``_search_network_snr`` for ``pycbc_multi_inspiral``
    (the production PyGRB coherent statistic) with a KN-constrained bank, and
    point the data fetch at gwdatafind/OSDF for non-public strain (needs the
    IGWN-credentialed AP / scitoken).

Return contract matches the other bridges: ``{status, message, source,
json_result_file, plot_file, ...}`` so ``pygrb_wrapper`` bundles it for SkyPortal
identically to the light-curve fitters. Runs inside the pycbc image on an OSG
worker (containers: /cvmfs/.../pycbc/pycbc-el8:latest); pycbc/gwpy are imported
lazily so this module stays importable in a bare test env.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

# Defaults, overridable per-call via ``analysis_parameters``. BNS-tuned (the KN
# targeted-search use case): low f_lower, TaylorF2, 1.4-1.4 Msun template.
DEFAULTS: dict[str, Any] = {
    "detectors": ["H1", "L1", "V1"],
    "data_source": "gwosc",  # Phase 1: public open data
    "event_name": None,  # GWOSC event (e.g. "GW170817") for the Phase 1 fetch
    "time_format": "gps",  # "gps" or "mjd" (the KN fit reports T0 in MJD)
    "approximant": "TaylorF2",
    "mass1": 1.4,
    "mass2": 1.4,
    "f_lower": 25.0,
    "sample_rate": 2048,
    "psd_seconds": 512.0,  # strain used to estimate each detector's PSD
    "onsource_window": 8.0,  # +/- s around the trigger (floor; widened by T0 below)
    "pad_seconds": 8.0,  # edge padding discarded after filtering
    # KN T0 coupling (deck Test 3): center + width the on-source window on the KN
    # light-curve fit's T0 posterior. Give t0_samples (MJD) or t0_sigma_days.
    "t0_samples": None,
    "t0_sigma_days": None,
    "t0_window_nsigma": 5.0,
}

_VALID_DETECTORS = {"H1", "L1", "V1", "K1", "G1"}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _truthy(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "t", "1", "yes")


def _to_gps(value: float, time_format: str) -> float:
    """Trigger time as GPS seconds. ``mjd`` is converted with astropy (the KN
    fit's T0 posterior is in MJD); ``gps`` is passed through."""
    if str(time_format).lower() == "mjd":
        from astropy.time import Time  # lazy

        return float(Time(float(value), format="mjd").gps)
    return float(value)


def _resolve_sky(payload: dict, params: dict) -> tuple[float, float]:
    """RA/Dec in radians from analysis_parameters or the top-level payload
    (SkyPortal passes the source position); degrees in, radians out."""
    import numpy as np

    ra = params.get("ra", payload.get("ra"))
    dec = params.get("dec", payload.get("dec"))
    if ra is None or dec is None:
        raise ValueError("pygrb: ra and dec (degrees) are required for a targeted search")
    return float(np.deg2rad(float(ra))), float(np.deg2rad(float(dec)))


def _resolve_trigger(params: dict) -> tuple[float, float, dict]:
    """Resolve (trigger_gps, onsource_window_s, t0_info) from the request.

    KN coupling (deck Test 3): if the KN light-curve fit's T0 posterior is passed
    -- as ``t0_samples`` (MJD) or ``t0_sigma_days`` -- center the trigger on it and
    widen the on-source window to +/- ``t0_window_nsigma`` * sigma, so the GW
    search window is set by the KN inference rather than a fixed guess. Otherwise
    use ``trigger_time`` and the fixed ``onsource_window``."""
    floor = float(params["onsource_window"])
    nsigma = float(params["t0_window_nsigma"])
    samples = params.get("t0_samples")
    if samples:
        median_mjd = statistics.median(float(x) for x in samples)
        sigma_days = statistics.pstdev(float(x) for x in samples)
        window = max(floor, nsigma * sigma_days * 86400.0)
        return (
            _to_gps(median_mjd, "mjd"),
            window,
            {
                "t0_source": "kn_posterior",
                "t0_mjd": median_mjd,
                "t0_sigma_days": sigma_days,
            },
        )
    if params.get("trigger_time") is None:
        raise ValueError("pygrb: trigger_time (or t0_samples) is required")
    gps = _to_gps(params["trigger_time"], params["time_format"])
    if params.get("t0_sigma_days") is not None:
        sigma_days = float(params["t0_sigma_days"])
        window = max(floor, nsigma * sigma_days * 86400.0)
        return gps, window, {"t0_source": "kn_sigma", "t0_sigma_days": sigma_days}
    return gps, floor, {"t0_source": "fixed"}


def validate_inputs(payload: dict) -> dict:
    """Cheap pre-flight so the wrapper can fail fast (no data pulled): resolves
    the trigger GPS (from a fixed time or the KN T0 posterior), on-source window,
    sky position, and detector list. Raises on bad input."""
    params = _params(payload)
    gps, window, t0_info = _resolve_trigger(params)
    dets = [d for d in params["detectors"] if d in _VALID_DETECTORS]
    if len(dets) < 1:
        raise ValueError(f"pygrb: no valid detectors in {params['detectors']}")
    return {"trigger_gps": gps, "onsource_window": window, "detectors": dets, "t0_info": t0_info}


# ---------------------------------------------------------------------------
# Phase 1 search engine: network coherent SNR from the PyCBC Python API.
# ---------------------------------------------------------------------------
def _fetch_strain(ifo: str, start: float, end: float, sample_rate: int, event_name):
    """Public GWOSC strain for one detector, sliced to [start, end] and resampled.

    Phase 1 uses ``pycbc.catalog.Merger`` (named GWOSC events), which is what ships
    in the image. GWOSC serves 32 s or 4096 s files; pick the smallest that covers
    the requested span. Arbitrary-GPS/non-public strain is Phase 2 via
    gwdatafind/OSDF (needs the IGWN-credentialed AP)."""
    from pycbc.catalog import Merger  # lazy; ships in the pycbc image
    from pycbc.filter import resample_to_delta_t

    if not event_name:
        raise ValueError(
            "pygrb Phase 1 needs analysis_parameters.event_name (a GWOSC event); "
            "arbitrary-GPS fetch is Phase 2 (gwdatafind/OSDF, IGWN credentials)"
        )
    gwosc_duration = 32 if (end - start) <= 28 else 4096
    # GWOSC native rate is 4096 Hz; fetch that then resample to the analysis rate.
    ts = Merger(event_name).strain(ifo, duration=gwosc_duration, sample_rate=4096)
    ts = ts.time_slice(start, end)
    target_dt = 1.0 / sample_rate
    if abs(ts.delta_t - target_dt) > 1e-12:
        ts = resample_to_delta_t(ts, target_dt)
    return ts


def _search_network_snr(
    payload: dict,
    params: dict,
    trigger_gps: float,
    dets: list,
    onsource_window: float,
    outdir: Path,
) -> dict:
    """Per-detector matched filter + sky-consistent (geocenter-aligned) network
    SNR around the trigger. A prototype of the PyGRB coherent statistic; the
    production statistic (dominant-polarization projection) comes from
    pycbc_multi_inspiral in Phase 2."""
    import numpy as np
    from pycbc.detector import Detector
    from pycbc.filter import matched_filter
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.waveform import get_fd_waveform

    ra, dec = _resolve_sky(payload, params)
    srate = int(params["sample_rate"])
    f_low = float(params["f_lower"])
    win = float(onsource_window)
    pad = float(params["pad_seconds"])
    seg = float(params["psd_seconds"])
    # Margin so the geocenter time-shift (<= Earth light-crossing ~0.043 s) stays
    # inside the cropped SNR series at the window edges.
    edge = 1.0
    start = trigger_gps - seg - win - pad
    end = trigger_gps + win + pad + edge

    per_det = {}
    snr_series = {}
    event_name = params.get("event_name")
    for ifo in dets:
        data = _fetch_strain(ifo, start, end, srate, event_name)
        data = data.highpass_fir(f_low, 512)
        psd = interpolate(data.psd(4), data.delta_f)
        psd = inverse_spectrum_truncation(
            psd, int(4 * data.sample_rate), low_frequency_cutoff=f_low
        )
        hp, _ = get_fd_waveform(
            approximant=params["approximant"],
            mass1=float(params["mass1"]),
            mass2=float(params["mass2"]),
            f_lower=f_low,
            delta_f=data.delta_f,
        )
        hp.resize(len(psd))
        snr = matched_filter(hp, data, psd=psd, low_frequency_cutoff=f_low)
        snr = snr.crop(seg // 2, pad)  # drop PSD-corrupted edges
        snr_series[ifo] = snr
        # Shift each detector's SNR to the geocenter for this sky position, so the
        # network sum is coherent in time-of-arrival.
        dt = Detector(ifo).time_delay_from_earth_center(ra, dec, trigger_gps)
        per_det[ifo] = {"dt": dt, "peak": float(abs(snr).numpy().max())}

    # Common geocenter grid over the on-source window; sum |rho|^2 across dets.
    # Vectorized (index-shift each series by its geocenter delay) so wide,
    # T0-posterior-driven windows stay fast; edge-guarded against short slices.
    n = int(round(2.0 * win * srate))
    times = (trigger_gps - win) + np.arange(n) / srate
    net_sq = np.zeros(n)
    for ifo in dets:
        snr = snr_series[ifo]
        arr = abs(snr.numpy())
        i0 = int(
            round((trigger_gps - win + per_det[ifo]["dt"] - float(snr.start_time)) / snr.delta_t)
        )
        a = max(0, -i0)
        b = min(n, len(arr) - i0)
        if b > a:
            net_sq[a:b] += arr[i0 + a : i0 + b] ** 2
    net = np.sqrt(net_sq)
    ipk = int(np.argmax(net))
    coherent_snr = float(net[ipk])
    best_gps = float(times[ipk])

    plot_file = _plot(outdir, times, net, trigger_gps, coherent_snr, best_gps)
    return {
        "coherent_snr": coherent_snr,
        "best_gps": best_gps,
        "trigger_gps": trigger_gps,
        "detectors": dets,
        "single_detector_peaks": {d: per_det[d]["peak"] for d in dets},
        "plot_file": plot_file,
    }


def _plot(outdir, times, net, trigger_gps, coherent_snr, best_gps) -> str:
    """Coherent-SNR-vs-time around the trigger, in the deck's style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times - trigger_gps, net, ".", ms=3, color="tab:blue", label="Network SNR")
    ax.axvline(0.0, ls="--", color="green", label="Trigger time (T0)")
    ax.plot(
        best_gps - trigger_gps,
        coherent_snr,
        "*",
        ms=16,
        color="red",
        label=f"Best candidate (SNR={coherent_snr:.2f})",
    )
    ax.set_xlabel("t - T0 [s]")
    ax.set_ylabel("Coherent (network) SNR")
    ax.set_title("Targeted-search trigger distribution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = Path(outdir) / "pygrb_trigger_distribution.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def run_from_skyportal_inputs(
    payload: dict[str, Any],
    *,
    outdir: Path | None = None,
    resource_id: str = "obj",
    seed: int = 42,
) -> dict[str, Any]:
    """Run the targeted coherent search. Same return contract as the other
    bridges. ``analysis_parameters.dry_run`` returns a stub without pulling any
    data (used by the test suite and for plumbing checks)."""
    params = _params(payload)

    if _truthy(params.get("dry_run")):
        info = validate_inputs(payload)
        return {
            "_stub": True,
            "status": "success",
            "source": "pygrb",
            "message": "pygrb dry-run",
            "coherent_snr": None,
            **info,
        }

    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="pygrb_osg_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info = validate_inputs(payload)
    trigger_gps, dets = info["trigger_gps"], info["detectors"]
    window, t0_info = info["onsource_window"], info["t0_info"]

    result = _search_network_snr(payload, params, trigger_gps, dets, window, outdir)

    result_json = {
        "source": "pygrb",
        "search": "network_coherent_snr",
        "coherent_snr": result["coherent_snr"],
        "best_gps": result["best_gps"],
        "trigger_gps": trigger_gps,
        "onsource_window": window,
        "t0_info": t0_info,
        "detectors": dets,
        "single_detector_peaks": result["single_detector_peaks"],
        "mass1": float(params["mass1"]),
        "mass2": float(params["mass2"]),
        "f_lower": float(params["f_lower"]),
    }
    result_file = outdir / "pygrb_result.json"
    result_file.write_text(json.dumps(result_json))

    return {
        "status": "success",
        "source": "pygrb",
        "message": (
            f"pygrb coherent search complete: SNR={result['coherent_snr']:.2f} "
            f"at GPS {result['best_gps']:.3f} (T0={trigger_gps:.3f}, "
            f"dets={'+'.join(dets)})"
        ),
        "coherent_snr": result["coherent_snr"],
        "best_gps": result["best_gps"],
        "json_result_file": str(result_file),
        "plot_file": result["plot_file"],
    }
