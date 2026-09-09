"""
NGSF bridge — turns a SkyPortal spectrum-analysis payload into NGSF runs.

Takes a SkyPortal payload (``spectra`` + ``redshift`` CSV plus free-form
``analysis_parameters``), writes the chosen spectrum as an NGSF ascii file, and
shells out to NGSF's ``run.py``. Reproduces what the Fritz bot posts: a
free-redshift scan refit at its own best-fit z, plus a fit pinned to the
SkyPortal redshift when the source has one.

NGSF is not importable as a library — ``NGSF/sf_class.py`` reads ``sys.argv``
and builds its parameters at module import — so each pass is a subprocess.

Stdlib only: the payload is all CSV, and keeping numpy/pandas out means tests
import this in a lightweight env (same reason as mosfit_bridge).
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

NGSF_DIR = Path(os.environ.get("NGSF_DIR", "/opt/NGSF"))
NGSF_BANK_DIR = Path(os.environ.get("NGSF_BANK_DIR", "/opt/ngsf-bank"))

# Matches the Fritz bot's per-instrument ranges (main.py:superfit_robot).
DEFAULT_WAV_RANGE = (4000.0, 9500.0)
INSTRUMENT_WAV_RANGES = {
    "NGPS": (5900.0, 10000.0),
    "GHTS": (4000.0, 7000.0),
}

# NGSF's sentinel for "redshift is a free parameter" (sf_class.py).
FREE_Z = 100.0

DEFAULTS = {
    "resolution": 10,
    "n_results": 3,
    "z_range_begin": 0.0,
    "z_range_end": 0.15,
    "z_int": 0.001,
}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _read_csv(value) -> list[dict]:
    """SkyPortal ships each input type as a CSV string."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value if "\n" in value else Path(value).read_text()
        return list(csv.DictReader(io.StringIO(text)))
    if isinstance(value, list):
        return value
    return []


def _as_floats(cell) -> list[float]:
    """A wavelength/flux cell arrives as a list *repr* (ndarray.tolist + to_csv)."""
    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]
    if isinstance(cell, str):
        return [float(x) for x in ast.literal_eval(cell)]
    raise ValueError(f"cannot read spectrum column of type {type(cell).__name__}")


def _to_float(value) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def select_spectrum(payload: dict) -> tuple[dict, int]:
    """Pick the spectrum to fit; the exported columns carry no spectrum id, so
    selection is by row index or observed_at, defaulting to the most recent."""
    rows = _read_csv(payload.get("spectra"))
    if not rows:
        raise ValueError(
            "no spectra in payload; the analysis service needs input_data_type 'spectra'"
        )

    params = _params(payload)
    index = params.get("spectrum_index")
    if index is not None:
        i = int(index)
        if not -len(rows) <= i < len(rows):
            raise ValueError(f"spectrum_index {i} out of range ({len(rows)} spectra)")
        return rows[i], i % len(rows)

    observed_at = params.get("observed_at")
    if observed_at:
        for i, row in enumerate(rows):
            if str(row.get("observed_at", "")).startswith(str(observed_at)):
                return row, i
        raise ValueError(f"no spectrum with observed_at starting {observed_at!r}")

    # observed_at is ISO-8601, so lexical order is chronological order.
    i = max(range(len(rows)), key=lambda j: str(rows[j].get("observed_at") or ""))
    return rows[i], i


def write_spectrum_ascii(row: dict, path: Path) -> int:
    """NGSF reads a two-column wavelength/flux ascii file."""
    lam = _as_floats(row["wavelengths"])
    flux = _as_floats(row["fluxes"])
    if len(lam) != len(flux):
        raise ValueError(f"spectrum has {len(lam)} wavelengths but {len(flux)} fluxes")

    samples = [(w, f) for w, f in zip(lam, flux) if math.isfinite(w) and math.isfinite(f)]
    if not samples:
        raise ValueError("spectrum has no finite samples")
    samples.sort()

    with path.open("w") as fh:
        for w, f in samples:
            fh.write(f"{w:.6f} {f:.6e}\n")
    return len(samples)


def resolve_redshift(payload: dict) -> float | None:
    """analysis_parameters.redshift wins; otherwise the source's SkyPortal value."""
    params = _params(payload)
    override = _to_float(params.get("redshift"))
    if override is not None:
        return override

    rows = _read_csv(payload.get("redshift"))
    if not rows:
        return None
    # An unset redshift arrives as an empty cell or NaN.
    return _to_float(rows[0].get("redshift"))


def wav_range(payload: dict, row: dict) -> tuple[float, float]:
    params = _params(payload)
    if params.get("wav_range"):
        lo, hi = params["wav_range"]
        return float(lo), float(hi)
    if params.get("wav_min") and params.get("wav_max"):
        return float(params["wav_min"]), float(params["wav_max"])

    instrument = str(params.get("instrument") or row.get("origin") or "").upper()
    for name, rng in INSTRUMENT_WAV_RANGES.items():
        if name in instrument:
            return rng
    return DEFAULT_WAV_RANGE


def _prepare_tree(root: Path, payload: dict, lo: float, hi: float) -> Path:
    """Copy the image's NGSF tree into the writable sandbox: pkg_dir is both the
    code root (get_metadata reads under it) and the output root (sf_class
    writes fit_results*/ under it)."""
    root.mkdir(parents=True, exist_ok=True)
    tree = root / "NGSF"
    if not tree.exists():
        shutil.copytree(NGSF_DIR, tree, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    for sub in ("fit_results", "fit_results_z", "spectra_to_fit"):
        (tree / sub).mkdir(exist_ok=True)

    params = _params(payload)
    config_path = tree / "config" / "parameters.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "pkg_dir": str(tree) + "/",
            "bank_dir": str(NGSF_BANK_DIR) + "/",
            "lower_lam": lo,
            "upper_lam": hi,
            "resolution": params["resolution"],
            "z_range_begin": params["z_range_begin"],
            "z_range_end": params["z_range_end"],
            "z_int": params["z_int"],
            "how_many_plots": params["n_results"],
            "show_plot": 0,
            "show_plot_png": 1,
            "fritz_token": "",
        }
    )
    config_path.write_text(json.dumps(config))
    return tree


def _run_ngsf(tree: Path, spectrum: Path, z: float, lo: float, hi: float, timeout: int) -> str:
    env = {
        **os.environ,
        "NGSFCONFIG": str(tree / "config" / "parameters.json"),
        "PYTHONPATH": str(tree),
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, "run.py", str(spectrum), str(z), str(lo), str(hi)],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(f"NGSF exited {proc.returncode} (z={z}):\n{tail}")
    return proc.stdout


def read_model_spectrum(path: Path, max_points: int = 3000) -> list | None:
    """Best-fit model spectrum (NGSF sf_class.py writes ``<stem>_ngsf0_model.txt``,
    two columns wavelength/model-flux) as ``[[wavelength, flux], ...]`` for the
    SkyPortal spectrum-plot overlay. Downsampled so the payload stays small."""
    if not path.exists():
        return None
    pts = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            w, fl = _to_float(parts[0]), _to_float(parts[1])
            if w is not None and fl is not None:
                pts.append([w, fl])
    if not pts:
        return None
    if len(pts) > max_points:
        pts = pts[:: (len(pts) // max_points + 1)]
    return pts


def _collect(tree: Path, stem: str, free_z: bool, n_results: int) -> dict:
    """Read one pass's results CSV and its ranked fit plots."""
    out_dir = tree / ("fit_results" if free_z else "fit_results_z")
    results_csv = out_dir / f"{stem}.csv"
    if not results_csv.exists():
        raise RuntimeError(f"NGSF produced no results at {results_csv}")

    rows = list(csv.DictReader(results_csv.open()))
    # A row whose fit failed carries a non-numeric chi2, so it sorts last.
    rows.sort(key=lambda r: _to_float(r.get("CHI2/dof")) or math.inf)

    def _typed(row: dict) -> dict:
        # Numeric columns come back as strings; keep the text of the rest.
        return {k: (_to_float(v) if _to_float(v) is not None else v) for k, v in row.items()}

    top = [_typed(r) for r in rows[:n_results]]

    plots = []
    for i in range(n_results):
        png = out_dir / f"{stem}_ngsf{i}.png"
        if png.exists():
            plots.append(str(png))
    model_spectrum = read_model_spectrum(out_dir / f"{stem}_ngsf0_model.txt")
    return {
        "rows": top,
        "best": top[0] if top else None,
        "plot_files": plots,
        "model_spectrum": model_spectrum,
    }


def _duplicates_scan(z_skyportal: float | None, z_fit: float | None, z_int: float) -> bool:
    """True when the scan landed within one grid step of the catalog redshift,
    so a fixed-z pass would just repeat the refit."""
    if z_skyportal is None or z_fit is None:
        return False
    return abs(z_skyportal - z_fit) <= z_int


def _sn_type(best: dict | None) -> str | None:
    """Leading segment of e.g. "Ia-norm/2009Y/WFCCD phase-band : 12.36B"."""
    if not best or not best.get("SN"):
        return None
    return str(best["SN"]).split("/")[0].strip() or None


def _overlap_fraction(row: dict, lo: float, hi: float) -> float | None:
    """Fraction of the fitted range [lo, hi] the spectrum spans. NGSF's
    minimum_overlap gate is on this; below it every chi2 comes back non-finite."""
    lam = [w for w in _as_floats(row["wavelengths"]) if math.isfinite(w)]
    if not lam or hi <= lo:
        return None
    covered = min(hi, max(lam)) - max(lo, min(lam))
    return max(0.0, covered) / (hi - lo)


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    params = _params(payload)
    n_results = int(params["n_results"])
    timeout = int(params.get("fit_timeout", 3600))

    row, index = select_spectrum(payload)
    lo, hi = wav_range(payload, row)

    work = Path(work_dir).resolve()
    spectra_dir = work / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{resource_id}_spec{index}"
    spectrum = spectra_dir / f"{stem}.ascii"
    n_samples = write_spectrum_ascii(row, spectrum)

    z_skyportal = resolve_redshift(payload)
    passes: dict = {}
    plot_files: list[str] = []

    # NGSF writes free and fixed passes to different dirs, so one tree does both.
    tree = _prepare_tree(work / "free", payload, lo, hi)
    _run_ngsf(tree, spectrum, FREE_Z, lo, hi, timeout)
    free = _collect(tree, stem, free_z=True, n_results=n_results)
    passes["free_z"] = free
    plot_files += free["plot_files"]

    z_fit = _to_float(free["best"]["Z"]) if free["best"] else None
    if z_fit is not None:
        _run_ngsf(tree, spectrum, z_fit, lo, hi, timeout)
        refit = _collect(tree, stem, free_z=False, n_results=n_results)
        refit["redshift"] = z_fit
        passes["refit_at_best_z"] = refit
        plot_files += refit["plot_files"]

    # Its own tree, so it can't overwrite the refit's fit_results_z/.
    already_fit = _duplicates_scan(z_skyportal, z_fit, float(params["z_int"]))
    if z_skyportal is not None and not already_fit:
        fixed_tree = _prepare_tree(work / "fixed", payload, lo, hi)
        _run_ngsf(fixed_tree, spectrum, z_skyportal, lo, hi, timeout)
        fixed = _collect(fixed_tree, stem, free_z=False, n_results=n_results)
        fixed["redshift"] = z_skyportal
        passes["fixed_z"] = fixed
        plot_files += fixed["plot_files"]

    # Prefer the catalog-redshift fit for annotations; it is the more trustworthy.
    headline = passes.get("fixed_z") or passes.get("refit_at_best_z") or passes.get("free_z")
    best = headline["best"] if headline else None

    # NGSF still writes a full ranked table when nothing clears its
    # minimum_overlap gate, but every chi2 is then non-finite (inf/-inf) and the
    # ranking is meaningless. Treat a non-finite best chi2 as a failed fit so a
    # spectrum that cannot be fit never becomes a confident annotation.
    overlap = _overlap_fraction(row, lo, hi)
    best_chi2 = _to_float(best.get("CHI2/dof")) if best else None
    fit_ok = best is not None and best_chi2 is not None and math.isfinite(best_chi2)

    results = {
        "spectrum": {
            "index": index,
            "observed_at": row.get("observed_at"),
            "origin": row.get("origin"),
            "n_samples": n_samples,
            "wav_range": [lo, hi],
            "wav_overlap_fraction": overlap,
        },
        "redshift_skyportal": z_skyportal,
        "fixed_z_skipped_as_duplicate": already_fit,
        "passes": passes,
    }

    if not fit_ok:
        overlap_txt = f"{overlap:.0%}" if overlap is not None else "unknown"
        return {
            "status": "failure",
            "message": (
                "NGSF produced no finite chi-squared: the spectrum covers "
                f"{overlap_txt} of the fitted range {lo:.0f}-{hi:.0f} A, below "
                "NGSF's minimum overlap, so no reliable classification is possible"
            ),
            "results": results,
            "annotations": {},
            "model_spectrum": None,
            "model_spectrum_summary": None,
            "plot_files": [],
        }

    model_spectrum = headline.get("model_spectrum")

    # Classification headline for the overlay hover (type/z/chi2/host).
    bits = [t for t in [_sn_type(best)] if t]
    z_best = _to_float(best.get("Z"))
    if z_best is not None:
        bits.append(f"z={z_best:.4f}")
    bits.append(f"chi2/dof {best_chi2:.2f}")
    if best.get("GALAXY"):
        bits.append(f"host {best['GALAXY']}")
    model_spectrum_summary = " · ".join(bits) or None

    annotations = {
        "ngsf_classification": _sn_type(best),
        "ngsf_redshift": z_best,
        "ngsf_chi2_dof": best_chi2,
        "ngsf_host_galaxy": best.get("GALAXY"),
        "ngsf_phase": _to_float(best.get("Phase")),
    }
    annotations = {k: v for k, v in annotations.items() if v is not None}

    z_txt = f"{z_best:.4f}" if z_best is not None else "unknown"
    message = f"NGSF matched {_sn_type(best)} at z={z_txt} (chi2/dof={best_chi2:.3f})"

    return {
        "status": "success",
        "message": message,
        "results": results,
        "annotations": annotations,
        "model_spectrum": model_spectrum,
        "model_spectrum_summary": model_spectrum_summary,
        "plot_files": plot_files,
    }
