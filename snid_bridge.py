"""
SNID-SAGE bridge — turns a SkyPortal spectrum-analysis payload into a SNID-SAGE run.

Takes a SkyPortal payload (``spectra`` + ``redshift`` CSV plus free-form
``analysis_parameters``), writes the chosen spectrum as a two-column ascii file,
and shells out to SNID-SAGE's ``sage identify``. Returns the classification
(type/subtype/redshift/age), the ranked template matches, and the diagnostic
plots.

SNID-SAGE runs via its CLI (``sage``); the templates are baked into the image
(``SNID_SAGE_TEMPLATE_DIR``), so the fit is offline.

Stdlib only: the payload is all CSV and SNID-SAGE is a subprocess, so tests
import this in a lightweight env (same reason as ngsf_bridge/mosfit_bridge).
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import subprocess
from pathlib import Path

# ``sage`` is on PATH in the image; overridable for tests / non-standard installs.
SAGE_BIN = os.environ.get("SNID_SAGE_BIN", "sage")
# The image bakes a read-only template bank here; the job runs python directly
# (bypassing the image entrypoint), so the bridge exposes it via writable scratch.
BAKED_TEMPLATES = Path(os.environ.get("SNID_SAGE_BAKED_TEMPLATES", "/opt/snid_sage/templates"))

DEFAULTS = {
    "n_results": 5,
    "fit_timeout": 3600,
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
    """A wavelength/flux cell arrives as a list *repr* (ndarray.tolist + to_csv).
    Parsed with float() rather than ast so bare ``nan``/``inf`` (masked pixels)
    survive to the finite-filter in write_spectrum_ascii."""
    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]
    if isinstance(cell, str):
        return [float(x) for x in cell.strip().strip("[]").split(",") if x.strip()]
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
    """SNID-SAGE reads a two-column wavelength/flux ascii file."""
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
    return _to_float(rows[0].get("redshift"))


def _ensure_template_dir(work: Path) -> None:
    """SNID-SAGE requires a *writable* template dir (W_OK check); the image's baked
    one is read-only. If SNID_SAGE_TEMPLATE_DIR is unset, mirror the image
    entrypoint: symlink the baked templates into writable scratch and point at it."""
    if os.environ.get("SNID_SAGE_TEMPLATE_DIR") or not BAKED_TEMPLATES.is_dir():
        return
    tdir = work / "snid_templates"
    tdir.mkdir(parents=True, exist_ok=True)
    for src in BAKED_TEMPLATES.iterdir():
        link = tdir / src.name
        if not link.exists():
            link.symlink_to(src)
    os.environ["SNID_SAGE_TEMPLATE_DIR"] = str(tdir)


def _run_sage(spectrum: Path, outdir: Path, z: float | None, timeout: int) -> str:
    """Run ``sage identify`` in --complete mode (writes plots + the .output file)."""
    cmd = [SAGE_BIN, "identify", str(spectrum), "--complete", "--output-dir", str(outdir)]
    if z is not None:
        cmd += ["--forced-redshift", f"{z:.6f}"]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(f"sage identify exited {proc.returncode}:\n{tail}")
    return proc.stdout


def parse_summary(stdout: str) -> dict:
    """Parse SNID-SAGE's one-line classification summary, e.g.
    ``name: II II-flash z=0.0025±0.0001 age=-8.0±0.5 Q_cluster=23.2
    MatchQual=High TypeConf=High SubtypeConf=High``. Match-quality/confidence
    values may contain a space (e.g. "Very Low")."""
    line = next((ln for ln in stdout.splitlines() if "MatchQual=" in ln and "z=" in ln), None)
    if line is None:
        return {}

    head = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    before_z = head.split(" z=")[0].strip()
    tokens = before_z.split()

    def _num(key: str) -> tuple[float | None, float | None]:
        m = re.search(rf"{key}=(-?[\d.]+)(?:±(-?[\d.]+))?", line)
        if not m:
            return None, None
        return _to_float(m.group(1)), _to_float(m.group(2))

    z, z_err = _num("z")
    age, age_err = _num("age")
    q = re.search(r"Q_cluster=(-?[\d.]+)", line)
    conf = re.search(r"MatchQual=(.*?)\s+TypeConf=(.*?)\s+SubtypeConf=(.*?)\s*$", line)
    return {
        "type": tokens[0] if tokens else None,
        "subtype": tokens[1] if len(tokens) > 1 else None,
        "redshift": z,
        "redshift_error": z_err,
        "age": age,
        "age_error": age_err,
        "q_cluster": _to_float(q.group(1)) if q else None,
        "match_quality": conf.group(1).strip() if conf else None,
        "type_confidence": conf.group(2).strip() if conf else None,
        "subtype_confidence": conf.group(3).strip() if conf else None,
    }


def parse_template_matches(output_file: Path, n_results: int) -> list[dict]:
    """The ranked template table from the .output file: rows like
    ``1 sn2024ggiEarly II II-flash 35.28 0.002511 0.000264 -8.0``."""
    if not output_file.exists():
        return []
    matches: list[dict] = []
    for line in output_file.read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0].isdigit():
            matches.append(
                {
                    "rank": int(f[0]),
                    "template": f[1],
                    "type": f[2],
                    "subtype": f[3],
                    "score": _to_float(f[4]),
                    "redshift": _to_float(f[5]),
                    "redshift_error": _to_float(f[6]),
                    "age": _to_float(f[7]),
                }
            )
        if len(matches) >= n_results:
            break
    return matches


def _collect_plots(outdir: Path, stem: str) -> list[str]:
    return sorted(str(p) for p in outdir.glob(f"{stem}*.png"))


def read_model_spectrum(outdir: Path, stem: str, max_points: int = 3000) -> list | None:
    """Best-match fluxed template (SNID-SAGE --complete writes
    ``<stem>_template_01_flux.dat``) as ``[[wavelength, flux], ...]`` for the
    SkyPortal spectrum-plot overlay. Downsampled so the payload stays small."""
    f = outdir / f"{stem}_template_01_flux.dat"
    if not f.exists():
        cand = sorted(outdir.glob(f"{stem}_template_*_flux.dat"))
        if not cand:
            return None
        f = cand[0]
    pts = []
    for line in f.read_text().splitlines():
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


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    params = _params(payload)
    n_results = int(params["n_results"])
    timeout = int(params["fit_timeout"])

    row, index = select_spectrum(payload)
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    _ensure_template_dir(work)
    stem = f"{resource_id}_spec{index}"
    spectrum = work / f"{stem}.dat"
    n_samples = write_spectrum_ascii(row, spectrum)

    z = resolve_redshift(payload) if not _params(payload).get("free_redshift") else None
    outdir = work / "snid_out"
    outdir.mkdir(parents=True, exist_ok=True)

    stdout = _run_sage(spectrum, outdir, z, timeout)
    summary = parse_summary(stdout)
    matches = parse_template_matches(outdir / f"{stem}.output", n_results)
    plots = _collect_plots(outdir, stem)
    model_spectrum = read_model_spectrum(outdir, stem)

    annotations = {
        "snid_classification": summary.get("type"),
        "snid_subtype": summary.get("subtype"),
        "snid_redshift": summary.get("redshift"),
        "snid_age": summary.get("age"),
        "snid_match_quality": summary.get("match_quality"),
    }
    annotations = {k: v for k, v in annotations.items() if v is not None}

    if summary.get("type"):
        sub = f" {summary['subtype']}" if summary.get("subtype") else ""
        zval = summary.get("redshift")
        zstr = f" at z={zval:.4f}" if zval is not None else ""
        message = (
            f"SNID-SAGE classified {summary['type']}{sub}{zstr} "
            f"(match quality {summary.get('match_quality')})"
        )
    else:
        message = "SNID-SAGE produced no confident classification"

    return {
        "status": "success",
        "message": message,
        "results": {
            "spectrum": {
                "index": index,
                "observed_at": row.get("observed_at"),
                "origin": row.get("origin"),
                "n_samples": n_samples,
            },
            "redshift_used": z,
            "classification": summary,
            "template_matches": matches,
        },
        "annotations": annotations,
        "model_spectrum": model_spectrum,
        "plot_files": plots,
    }
