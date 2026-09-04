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
    # "t0": center + widen the window on the KN T0 posterior (below); "fixed": use
    # onsource_window as the exact +/- window (e.g. 6s PyGRB, 1000s X-pipeline/GRB).
    "onsource_mode": "t0",
    "pad_seconds": 8.0,  # edge padding discarded after filtering
    # KN T0 coupling (deck Test 3): center + width the on-source window on the KN
    # light-curve fit's T0 posterior. Give t0_samples (MJD) or t0_sigma_days.
    "t0_samples": None,
    "t0_sigma_days": None,
    "t0_window_nsigma": 5.0,
    # Glitch gating: list of {ifo?, gps, window, taper} to zero out (tapered) loud
    # transients before PSD/matched-filter -- e.g. the GW170817 L1 glitch.
    "gates": None,
    # Coherent-SNR threshold for the trigger-distribution plot (clustered local
    # maxima above this are shown as the background scatter).
    "trigger_threshold": 4.5,
}

_VALID_DETECTORS = {"H1", "L1", "V1", "K1", "G1"}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


# Chirp-mass priors by inferred merger type (a KN implies a compact-object merger).
# A coarse but honest KN->bank coupling; a full ejecta -> component-mass relation is
# EOS-dependent and left as a modeling choice.
_MERGER_CHIRP_MASS = {"BNS": (1.0, 1.6), "NSBH": (1.5, 4.5)}


def pygrb_params_from_kn_fit(
    kn_result: dict,
    *,
    ra: float,
    dec: float,
    detectors=None,
    event_name=None,
    merger_type: str = "BNS",
    t0_key: str = "t0",
) -> dict:
    """Build pygrb ``analysis_parameters`` from a KN light-curve fit result, closing
    the KN->GW loop: the fit's epoch drives the search window (deck Test 3) and the
    inferred merger type drives a KN-constrained chirp-mass bank (deck Test 2).

    Fitter-agnostic: a fitted merger-time posterior (e.g. redback's ``t0`` samples,
    MJD) becomes ``t0_samples`` (window from its spread); otherwise a point epoch
    (e.g. fiesta's ``trigger_time``, MJD) becomes a fixed ``trigger_time``."""
    params: dict[str, Any] = {"backend": "pygrb", "ra": float(ra), "dec": float(dec)}
    if detectors:
        params["detectors"] = list(detectors)
    if event_name:
        params["event_name"] = event_name

    post = kn_result.get("posterior_samples") or {}
    if t0_key in post and post[t0_key]:
        params["t0_samples"] = [float(x) for x in post[t0_key]]  # MJD
        params["time_format"] = "mjd"
    elif kn_result.get("trigger_time") is not None:
        params["trigger_time"] = float(kn_result["trigger_time"])
        params["time_format"] = "mjd"

    cmr = _MERGER_CHIRP_MASS.get(str(merger_type).upper())
    if cmr:
        params["chirp_mass_range"] = list(cmr)
    return params


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

    Two window modes (``onsource_mode``):
    - ``t0`` (default): center on the KN light-curve fit's T0 posterior and widen
      the on-source window to +/- ``t0_window_nsigma`` * sigma (floored by
      ``onsource_window``), so the search window is set by the KN inference. T0 is
      passed as ``t0_samples`` (MJD) or ``t0_sigma_days``; absent both, falls back
      to ``trigger_time`` with the fixed ``onsource_window``.
    - ``fixed``: use ``onsource_window`` as the exact +/- window (e.g. 6s PyGRB,
      1000s X-pipeline/GRB), still centered on the T0 median (or ``trigger_time``).
    """
    floor = float(params["onsource_window"])
    nsigma = float(params["t0_window_nsigma"])
    mode = str(params.get("onsource_mode", "t0")).strip().lower()
    samples = params.get("t0_samples")

    # Center: KN T0 posterior median if given, else the fixed trigger_time.
    if samples:
        center_mjd = statistics.median(float(x) for x in samples)
        center_gps = _to_gps(center_mjd, "mjd")
        sigma_days = statistics.pstdev(float(x) for x in samples)
    else:
        if params.get("trigger_time") is None:
            raise ValueError("pygrb: trigger_time (or t0_samples) is required")
        center_gps = _to_gps(params["trigger_time"], params["time_format"])
        sig = params.get("t0_sigma_days")
        sigma_days = float(sig) if sig is not None else None

    # Fixed mode: user-set on-source window, ignore the T0 spread for sizing.
    if mode == "fixed":
        info = {"t0_source": "fixed_window", "onsource_mode": "fixed"}
        if samples:
            info["t0_mjd"] = center_mjd
        return center_gps, floor, info

    # T0 mode (default): widen to +/- nsigma * sigma from the KN posterior.
    if samples:
        window = max(floor, nsigma * sigma_days * 86400.0)
        return (
            center_gps,
            window,
            {
                "t0_source": "kn_posterior",
                "t0_mjd": center_mjd,
                "t0_sigma_days": sigma_days,
                "onsource_mode": "t0",
            },
        )
    if sigma_days is not None:
        window = max(floor, nsigma * sigma_days * 86400.0)
        return (
            center_gps,
            window,
            {
                "t0_source": "kn_sigma",
                "t0_sigma_days": sigma_days,
                "onsource_mode": "t0",
            },
        )
    return center_gps, floor, {"t0_source": "fixed", "onsource_mode": "t0"}


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
def _apply_gates(data, ifo: str, gates):
    """Zero out (tapered) loud transients before PSD/filtering. Each gate is
    {gps, window, taper, ifo?}; an ``ifo`` key restricts it to that detector
    (omit to gate all). Used to remove e.g. the GW170817 L1 glitch."""
    for g in gates or []:
        if g.get("ifo") and g["ifo"] != ifo:
            continue
        data = data.gate(
            float(g.get("gps", g.get("time"))),
            window=float(g.get("window", 0.5)),
            taper_width=float(g.get("taper", 0.25)),
        )
    return data


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


def _build_bank(params: dict) -> list:
    """Templates to search (a KN-constrained bank, deck Test 2). Options, in order:
    ``bank`` (explicit [[m1,m2],...]); ``chirp_mass_range`` (equal-mass templates
    finely gridded in chirp mass -- the right variable for BNS, since SNR is very
    sensitive to it); ``mass1_range``/``mass2_range`` (component-mass grid). Gridded
    to ``bank_n`` points. Default is the single (mass1, mass2)."""
    import numpy as np

    bank = params.get("bank")
    if bank:
        return [(float(a), float(b)) for a, b in bank]
    cmr = params.get("chirp_mass_range")
    if cmr:
        nb = int(params.get("bank_n", 20))
        # equal-mass template for chirp mass Mc has component mass Mc * 2**(1/5).
        return [
            (float(mc * 2**0.2), float(mc * 2**0.2))
            for mc in np.linspace(float(cmr[0]), float(cmr[1]), nb)
        ]
    m1r, m2r = params.get("mass1_range"), params.get("mass2_range")
    if m1r and m2r:
        nb = int(params.get("bank_n", 5))
        return [
            (float(a), float(b))
            for a in np.linspace(float(m1r[0]), float(m1r[1]), nb)
            for b in np.linspace(float(m2r[0]), float(m2r[1]), nb)
        ]
    return [(float(params["mass1"]), float(params["mass2"]))]


def _coherent_series(snr_series, sig, fp, fc, dt, dets, trigger_gps, win, srate, n):
    """Dominant-polarization coherent SNR + network SNR over the on-source grid,
    for one template's per-detector matched-filter outputs. Rotates the sigma-
    weighted antenna responses so the two GW polarizations decouple, then projects
    the time-shifted (geocenter-aligned) outputs onto each and combines."""
    import numpy as np

    wp = np.array([sig[i] * fp[i] for i in dets])
    wc = np.array([sig[i] * fc[i] for i in dets])
    two_psi = 0.5 * np.arctan2(2.0 * np.sum(wp * wc), np.sum(wp * wp - wc * wc))
    cos2, sin2 = np.cos(two_psi), np.sin(two_psi)
    ap = wp * cos2 + wc * sin2
    ac = -wp * sin2 + wc * cos2
    a_p, a_c = float(np.sum(ap * ap)), float(np.sum(ac * ac))

    sum_p = np.zeros(n, dtype=complex)
    sum_c = np.zeros(n, dtype=complex)
    net_sq = np.zeros(n)
    for k, ifo in enumerate(dets):
        s = snr_series[ifo]
        arr = s.numpy()  # complex SNR
        i0 = int(round((trigger_gps - win + dt[ifo] - float(s.start_time)) / s.delta_t))
        z = np.zeros(n, dtype=complex)
        a = max(0, -i0)
        b = min(n, len(arr) - i0)
        if b > a:
            z[a:b] = arr[i0 + a : i0 + b]
        sum_p += ap[k] * z
        sum_c += ac[k] * z
        net_sq += np.abs(z) ** 2
    coh_sq = np.zeros(n)
    if a_p > 1e-6:
        coh_sq += np.abs(sum_p) ** 2 / a_p
    if a_c > 1e-6:
        coh_sq += np.abs(sum_c) ** 2 / a_c
    return np.sqrt(coh_sq), np.sqrt(net_sq)


def _coherent_search(
    payload: dict,
    params: dict,
    trigger_gps: float,
    dets: list,
    onsource_window: float,
    outdir: Path,
) -> dict:
    """Dominant-polarization coherent search over the on-source window, optionally
    over a (KN-constrained) template bank. Conditions each detector's data once,
    then for each template matched-filters and coherently combines; the per-time
    max over the bank is the search statistic. This is the coherent search PyGRB
    runs -- the full stat/bank live in Phase 2's pycbc_multi_inspiral."""
    import numpy as np
    from pycbc.detector import Detector
    from pycbc.filter import matched_filter, sigma
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

    # Condition each detector's data once; every template reuses it.
    cond = {}
    event_name = params.get("event_name")
    for ifo in dets:
        data = _fetch_strain(ifo, start, end, srate, event_name)
        data = data.highpass_fir(f_low, 512)
        data = _apply_gates(data, ifo, params.get("gates"))
        psd = interpolate(data.psd(4), data.delta_f)
        psd = inverse_spectrum_truncation(
            psd, int(4 * data.sample_rate), low_frequency_cutoff=f_low
        )
        d = Detector(ifo)
        cond[ifo] = {
            "data": data,
            "psd": psd,
            "fp": d.antenna_pattern(ra, dec, 0.0, trigger_gps)[0],
            "fc": d.antenna_pattern(ra, dec, 0.0, trigger_gps)[1],
            "dt": d.time_delay_from_earth_center(ra, dec, trigger_gps),
        }

    def _template_series(m1, m2):
        """Per-detector matched-filter SNR + sigma for one (m1, m2) template."""
        snr_series, sig = {}, {}
        for ifo in dets:
            c = cond[ifo]
            hp, _ = get_fd_waveform(
                approximant=params["approximant"],
                mass1=m1,
                mass2=m2,
                f_lower=f_low,
                delta_f=c["data"].delta_f,
            )
            hp.resize(len(c["psd"]))
            snr = matched_filter(hp, c["data"], psd=c["psd"], low_frequency_cutoff=f_low)
            snr_series[ifo] = snr.crop(seg // 2, pad)
            sig[ifo] = float(sigma(hp, psd=c["psd"], low_frequency_cutoff=f_low))
        return snr_series, sig

    bank = _build_bank(params)
    fp = {i: cond[i]["fp"] for i in dets}
    fc = {i: cond[i]["fc"] for i in dets}
    dt = {i: cond[i]["dt"] for i in dets}
    n = int(round(2.0 * win * srate))
    times = (trigger_gps - win) + np.arange(n) / srate
    best_coh, best_net = np.zeros(n), np.zeros(n)
    best_idx = np.full(n, 0)
    for ti, (m1, m2) in enumerate(bank):
        snr_series, sig = _template_series(m1, m2)
        coh, net = _coherent_series(snr_series, sig, fp, fc, dt, dets, trigger_gps, win, srate, n)
        mask = coh > best_coh
        best_coh[mask] = coh[mask]
        best_net[mask] = net[mask]
        best_idx[mask] = ti

    ipk = int(np.argmax(best_coh))
    coherent_snr = float(best_coh[ipk])
    best_gps = float(times[ipk])
    best_m1, best_m2 = bank[int(best_idx[ipk])]

    # Recompute the winning template's per-detector numbers for the report.
    win_snr, win_sig = _template_series(best_m1, best_m2)

    trig_t, trig_s = _extract_triggers(times, best_coh, srate, float(params["trigger_threshold"]))
    plot_file = _plot(outdir, trig_t, trig_s, trigger_gps, coherent_snr, best_gps)
    return {
        "coherent_snr": coherent_snr,
        "network_snr": float(best_net[ipk]),
        "best_gps": best_gps,
        "trigger_gps": trigger_gps,
        "detectors": dets,
        "n_templates": len(bank),
        "n_triggers": int(len(trig_t)),
        "best_template": {"mass1": float(best_m1), "mass2": float(best_m2)},
        "single_detector_peaks": {i: float(abs(win_snr[i]).numpy().max()) for i in dets},
        "sigma": {i: win_sig[i] for i in dets},
        "plot_file": plot_file,
    }


def _extract_triggers(times, coh, srate, threshold, cluster_s=0.1, max_n=4000):
    """Clustered local maxima of the coherent SNR above ``threshold`` -> a trigger
    distribution (one point per ~cluster_s), capped to the loudest ``max_n``. This
    turns the continuous SNR series into the discrete trigger scatter PyGRB plots
    (the background at low SNR + any loud on-source candidate)."""
    import numpy as np

    try:
        from scipy.signal import find_peaks

        pk, _ = find_peaks(coh, height=threshold, distance=max(1, int(cluster_s * srate)))
    except Exception:  # noqa: BLE001 — no scipy: fall back to a plain threshold
        pk = np.where(coh >= threshold)[0]
    tt, ss = times[pk], coh[pk]
    if len(ss) > max_n:  # keep the loudest
        keep = np.argsort(ss)[-max_n:]
        tt, ss = tt[keep], ss[keep]
    order = np.argsort(tt)
    return tt[order], ss[order]


def _plot(outdir, trig_times, trig_snrs, trigger_gps, coherent_snr, best_gps) -> str:
    """Trigger distribution (coherent SNR vs t-T0), in the deck's style: the
    clustered triggers as a background scatter, the loudest starred, T0 marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        trig_times - trigger_gps,
        trig_snrs,
        ".",
        ms=3,
        color="tab:blue",
        alpha=0.5,
        label=f"Triggers ({len(trig_times)})",
    )
    ax.axvline(0.0, ls="--", color="green", label="Trigger time (T0)")
    ax.plot(
        best_gps - trigger_gps,
        coherent_snr,
        "*",
        ms=18,
        color="red",
        label=f"Best candidate (SNR={coherent_snr:.2f})",
    )
    ax.set_xlabel("t - T0 [s]")
    ax.set_ylabel("Coherent SNR")
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

    result = _coherent_search(payload, params, trigger_gps, dets, window, outdir)

    result_json = {
        "source": "pygrb",
        "search": "coherent_dominant_polarization",
        "coherent_snr": result["coherent_snr"],
        "network_snr": result["network_snr"],
        "best_gps": result["best_gps"],
        "trigger_gps": trigger_gps,
        "onsource_window": window,
        "t0_info": t0_info,
        "detectors": dets,
        "n_templates": result["n_templates"],
        "n_triggers": result["n_triggers"],
        "best_template": result["best_template"],
        "single_detector_peaks": result["single_detector_peaks"],
        "sigma": result["sigma"],
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
