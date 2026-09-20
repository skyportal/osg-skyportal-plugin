"""
aframe bridge — targeted (on-source) gravitational-wave search.

Runs the aframe detector NN (via ml4gw-buoy's ``buoy.Aframe``) over strain data
around a target GPS time and returns a detection statistic and, when a
background file is supplied, an empirical false-alarm rate. The ML4GW analog of
a PyCBC targeted/on-source search.

Payload (``analysis_parameters``), unlike the photometric bridges:
- ``t_event``   GPS time to target (required).
- ``ifos``      detectors to use (list or comma string; default H1, L1).
- ``weights``   aframe weights file, staged next to this script (default aframe.pt).
- ``config``    aframe config yaml (default aframe_config_bbh.yaml).
- ``background``timeslide background hdf5 for the FAR; omit for score-only.
- ``device``    torch device (default: cuda if present, else cpu).

Torch/buoy/gwpy imports are lazy so the module loads without them (tests).
"""

from __future__ import annotations

SECONDS_PER_YEAR = 3.156e7


def _pick_device(pref):
    import torch

    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ifos(params):
    val = params.get("ifos") or ["H1", "L1"]
    if isinstance(val, str):
        val = [s.strip() for s in val.split(",") if s.strip()]
    return tuple(val)


def _per_ifo(params, key, default_template, ifo):
    """A per-detector value: a dict keyed by ifo, a `{ifo}` template, or the default."""
    val = params.get(key)
    if isinstance(val, dict):
        return val.get(ifo, default_template.format(ifo=ifo))
    if isinstance(val, str) and val:
        return val.format(ifo=ifo)
    return default_template.format(ifo=ifo)


def _fetch_onsource(t_event, model, ifos, pad, params):
    """Strain spanning the model's minimum window around t_event, resampled to the
    model rate. ``data_source`` selects real IGWN strain over OSDF (``gwdatafind``,
    the OSG path -- needs the job scitoken) or public open data (``gwosc``).
    Returns (data_tensor (1, n_ifos, N), t0)."""
    import numpy as np
    import torch
    from gwpy.timeseries import TimeSeries

    min_duration = model.minimum_data_size / model.sample_rate
    fetch_start = t_event - min_duration - pad
    fetch_end = t_event + pad
    source = str(params.get("data_source", "gwosc")).lower()

    series = []
    for ifo in ifos:
        if source in ("gwdatafind", "osdf", "igwn"):
            import igwn_strain  # shipped per-job

            frametype = _per_ifo(params, "frametype", "{ifo}_HOFT_C00", ifo)
            channel = _per_ifo(params, "channel", "{ifo}:GDS-CALIB_STRAIN", ifo)
            host = params.get("gwdatafind_host") or igwn_strain.DEFAULT_GWDATAFIND_HOST
            frames = igwn_strain.fetch_frames(
                ifo[0],
                frametype,
                int(fetch_start) - 1,
                int(fetch_end) + 1,
                outdir="frames",
                host=host,
            )
            if not frames:
                raise ValueError(
                    f"no {frametype} strain available for {ifo} at GPS "
                    f"[{int(fetch_start)},{int(fetch_end)}] (no data at this epoch)"
                )
            ts = TimeSeries.read(
                [str(f) for f in frames], channel, start=fetch_start, end=fetch_end
            )
        else:
            ts = TimeSeries.fetch_open_data(ifo, fetch_start, fetch_end)
        series.append(ts.resample(model.sample_rate))

    n = min(len(ts.value) for ts in series)  # guard off-by-one across detectors
    stacked = np.stack([ts.value[:n] for ts in series])
    data = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    return data, float(series[0].t0.value)


def _run_inference(model, data, t0, t_event):
    """Score time series with the PSD warm-up region masked. Returns dict of
    times-relative-to-event plus raw/integrated online scores and the offline
    significance-integrated score."""
    import numpy as np

    times, ys, timing_integrated, signif_integrated = model(data, t0)

    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    corrected = to_np(times) + model.time_offset
    online = int(model.inference_sampling_rate * model.psd_length)
    offline = int(model.offline_sampling_rate * model.psd_length)
    signif_times = corrected[:: model.online_offline_stride]

    return {
        "t_online": corrected[online:] - t_event,
        "raw": to_np(ys)[online:],
        "integrated": to_np(timing_integrated)[online:],
        "t_offline": signif_times[offline:] - t_event,
        "signif_integrated": to_np(signif_integrated)[offline:],
    }


def _compute_far(loudest, background):
    """(far_per_yr, bound, n_louder, Tb) from a background hdf5, or all None when
    no background is given. ``bound`` is "exact", "upper" (zero louder events, so
    the FAR is an upper limit), or "lower" (a --top-k tail file whose cutoff sits
    above this candidate, so the true FAR can only be larger)."""
    if not background:
        return None, None, None, None

    import h5py
    import numpy as np

    # Count louder background events in chunks; the array can be billions of
    # rows (GB), so never load it whole on a worker.
    n_louder = 0
    with h5py.File(background, "r") as f:
        dset = f["parameters"]["detection_statistic"]
        Tb = float(f.attrs["Tb"])
        is_tail = bool(f.attrs.get("is_tail", 0))
        tail_min = float(f.attrs["tail_min"]) if is_tail else None
        step = 20_000_000
        for i in range(0, dset.shape[0], step):
            block = dset[i : i + step]
            n_louder += int(np.count_nonzero(block >= loudest))

    # A tail file dropped every event below tail_min; a candidate below that
    # cutoff has an incomplete count, so its FAR is only a lower bound.
    if is_tail and loudest < tail_min:
        return (n_louder / Tb) * SECONDS_PER_YEAR, "lower", n_louder, Tb
    if n_louder == 0:
        return (1.0 / Tb) * SECONDS_PER_YEAR, "upper", 0, Tb
    return (n_louder / Tb) * SECONDS_PER_YEAR, "exact", n_louder, Tb


def _score_plot(series, t_event, resource_id):
    import tempfile
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series["t_online"], series["raw"], label="raw", alpha=0.6)
    ax.plot(series["t_online"], series["integrated"], label="integrated")
    ax.plot(series["t_offline"], series["signif_integrated"], label="signif integrated")
    ax.axvline(0, color="red", ls="--", label="event time")
    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel("aframe score")
    ax.set_title(f"aframe targeted search — {resource_id}")
    ax.legend()
    fig.tight_layout()
    out = Path(tempfile.gettempdir()) / f"aframe_{resource_id}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def run_from_skyportal_inputs(inputs: dict, resource_id: str = "obj") -> dict:
    params = inputs.get("analysis_parameters") or {}
    # A gcn_event trigger carries its GPS time; fall back to it when unset.
    if params.get("t_event") in (None, ""):
        params["t_event"] = (inputs.get("gcn_event") or {}).get("gps")
    if params.get("t_event") in (None, ""):
        return {"status": "failure", "message": "aframe needs a `t_event` GPS time."}

    import numpy as np
    from buoy import Aframe

    t_event = float(params["t_event"])
    ifos = _ifos(params)
    pad = float(params.get("pad", 10.0))

    model = Aframe(
        model_weights=str(params.get("weights", "aframe.pt")),
        config=str(params.get("config", "aframe_config_bbh.yaml")),
        device=_pick_device(params.get("device")),
        load_weights=True,
    )

    data, t0 = _fetch_onsource(t_event, model, ifos, pad, params)
    series = _run_inference(model, data, t0, t_event)

    # Loudest on-source statistic: max significance-integrated score over the
    # valid (post-warm-up) window.
    loudest = float(np.max(series["signif_integrated"]))
    # Use a staged background.hdf5 for the FAR unless one is named explicitly.
    background = params.get("background")
    if not background:
        from pathlib import Path

        background = "background.hdf5" if Path("background.hdf5").exists() else None
    far, bound, n_louder, Tb = _compute_far(loudest, background)

    plot_file = _score_plot(series, t_event, resource_id)

    results = {
        "t_event": t_event,
        "ifos": list(ifos),
        "detection_statistic": loudest,
        "far_per_yr": far,
        "far_bound": bound,
        "far_is_upper_limit": bound == "upper",
        "n_louder": n_louder,
        "background_livetime_s": Tb,
    }
    if far is not None:
        op = {"upper": "<", "lower": ">", "exact": "="}[bound]
        message = f"aframe stat={loudest:.4g}, FAR {op} {far:.3g} yr^-1"
    else:
        message = f"aframe stat={loudest:.4g} (no background; FAR n/a)"
    annotations = [
        {
            "origin": "aframe",
            "data": {
                "aframe_detection_statistic": loudest,
                "aframe_far_per_yr": far,
                "aframe_far_bound": bound,
                "aframe_far_is_upper_limit": bound == "upper",
            },
        }
    ]
    return {
        "status": "success",
        "message": message,
        "results": results,
        "annotations": annotations,
        "plot_file": str(plot_file),
    }
