"""
Fiesta wrapper script — runs inside the fiesta runtime image on an OSG worker.

Contract:
- Reads ``inputs.json`` from cwd (staged by HTCondor file transfer).
- Fits a Fiesta model via ``fiesta_bridge`` (shipped alongside this file), OR a
  stub if ``analysis_parameters.dry_run`` is true.
- Bundles the result as SkyPortal's analysis-callback JSON
  ``{status, message, analysis: {results, plots, model_lightcurve}}``.
- Optionally PUTs the bundle to ``$OSDF_OUTPUT_URL``; always writes it to stdout
  so the plugin's poller can scrape it.
"""

from __future__ import annotations  # OSPool workers may run older Python; keep PEP 604 unions lazy

import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import requests  # only needed for OSDF upload; bare workers may lack it
except ImportError:
    requests = None


def load_inputs(path: Path = Path("inputs.json")) -> dict:
    return json.loads(path.read_text())


def _materialize_inputs(inputs: dict) -> dict:
    """SkyPortal sends photometry/redshift as CSV *content*; fiesta wants file
    paths. Write any inline CSV (multi-line string) to a file in cwd and point
    the payload at it. A bare filename (no newline) is left as-is.
    """
    for key in ("photometry", "redshift"):
        val = inputs.get(key)
        if isinstance(val, str) and "\n" in val:
            path = Path(f"{key}.csv")
            path.write_text(val)
            inputs[key] = str(path)
    return inputs


def run_fit(inputs: dict) -> dict:
    """Fit via the selected backend bridge (lazy import so this module loads
    without fiesta/redback). analysis_parameters.backend picks the bridge —
    "fiesta" (default) or "redback" (redback-jax). Both bridges share the same
    payload and return contract. Returns a stub when dry_run is set."""
    params = inputs.get("analysis_parameters") or {}
    # SkyPortal sends optional params as strings, so "False" must be falsy.
    dry_run = str(params.get("dry_run", "")).strip().lower() in ("true", "1", "yes", "t")
    if dry_run:
        return {
            "log_bayes_factor": 1.23,
            "_stub": True,
            "_source": params.get("source", "Bu2025_MLP"),
        }

    backend = str(params.get("backend", "fiesta")).strip().lower()
    if backend == "redback":
        from redback_bridge import run_from_skyportal_inputs  # shipped per-job
    else:
        from fiesta_bridge import run_from_skyportal_inputs  # shipped per-job

    inputs = _materialize_inputs(inputs)
    resource_id = inputs.get("resource_id", "obj")
    return run_from_skyportal_inputs(inputs, resource_id=resource_id)


def bundle_for_skyportal(
    fit_result: dict,
    plot_files: list[Path] | None = None,
    result_file: Path | None = None,
) -> dict:
    """Pack outputs into SkyPortal's analysis-callback schema."""
    analysis: dict[str, Any] = {}
    plots = []
    for pf in plot_files or []:
        if pf.exists():
            plots.append(
                {
                    "format": pf.suffix.lstrip(".") or "bin",
                    "data": base64.b64encode(pf.read_bytes()).decode(),
                }
            )
    if plots:
        analysis["plots"] = plots
    # Model name so SkyPortal labels the overlay by the actual fit (e.g.
    # Bu2025_MLP), not the generic analysis-service name.
    if fit_result.get("source"):
        analysis["model_name"] = fit_result["source"]
    if result_file is not None and result_file.exists():
        suffix = result_file.suffix.lstrip(".") or "bin"
        if suffix == "json":
            # SkyPortal's "json" results format expects the object itself, not a
            # base64-encoded string (which it would otherwise render per-char).
            analysis["results"] = {
                "format": "json",
                "data": json.loads(result_file.read_text()),
            }
        else:
            analysis["results"] = {
                "format": suffix,
                "data": base64.b64encode(result_file.read_bytes()).decode(),
            }
    if fit_result.get("_stub") and "results" not in analysis:
        analysis["results"] = {"format": "json", "data": fit_result}
    # Nested samplers give a log Bayes factor; fiesta-native (MCMC) doesn't, so
    # fall back to the fit's own message.
    if fit_result.get("log_bayes_factor") is not None:
        message = f"log Bayes factor={fit_result['log_bayes_factor']}"
    else:
        message = fit_result.get("message", "fit complete")
    return {"status": "success", "message": message, "analysis": analysis}


def _bearer() -> str | None:
    if "BEARER_TOKEN" in os.environ:
        return os.environ["BEARER_TOKEN"].strip() or None
    token_file = os.environ.get("BEARER_TOKEN_FILE")
    if token_file and os.path.exists(token_file):
        return Path(token_file).read_text().strip()
    return None


def upload(bundle: dict) -> bool:
    output_url = os.environ.get("OSDF_OUTPUT_URL")
    if not output_url:
        return False
    if requests is None:
        print("warning: requests unavailable; skipping OSDF upload", file=sys.stderr)
        return False
    token = _bearer()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.put(output_url, data=json.dumps(bundle), headers=headers, timeout=300)
    r.raise_for_status()
    return True


def _bundle_one(result: dict) -> dict:
    """Single-model SkyPortal bundle (unchanged single-fit behaviour)."""
    plot_files = [Path(result["plot_file"])] if result.get("plot_file") else []
    result_file = Path(result["json_result_file"]) if result.get("json_result_file") else None
    bundle = bundle_for_skyportal(result, plot_files=plot_files, result_file=result_file)
    # Carry the per-filter model light curve through for the SkyPortal
    # photometry-plot overlay (median + band per filter, on MJD).
    if result.get("model_lightcurve"):
        bundle["analysis"]["model_lightcurve"] = result["model_lightcurve"]
    if result.get("posterior_samples"):
        bundle["analysis"]["posterior_samples"] = result["posterior_samples"]
    if result.get("n_detections") is not None:
        bundle["analysis"]["n_detections"] = result["n_detections"]
    return bundle


def main() -> int:
    try:
        inputs = load_inputs()
        params = inputs.get("analysis_parameters") or {}
        dry_run = str(params.get("dry_run", "")).strip().lower() in (
            "true",
            "1",
            "yes",
            "t",
        )
        # Fail the whole request fast (and once) when the photometry has fewer
        # than 2 finite detections — all upper limits / NaN / negative flux — a
        # fit needs at least 2 points, and this avoids every model reporting the
        # same emptiness separately.
        n_det_pre = None
        if not dry_run:
            try:
                from fiesta_bridge import count_detections

                n_det_pre = count_detections(_materialize_inputs({**inputs}))
            except Exception:  # noqa: BLE001 — on any read issue, take the normal path
                n_det_pre = None
        # Grouped fit: analysis_parameters.sources = [model, ...] (or a comma
        # string) fits every model in THIS one job and overlays all their curves.
        # Falls back to the single-model path when only `source` is given.
        sources = params.get("sources")
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]
        if n_det_pre is not None and n_det_pre < 2:
            bundle = {
                "status": "failure",
                "message": f"Not enough detections to fit (need at least 2, have {n_det_pre}).",
                "analysis": {"n_detections": n_det_pre},
            }
        elif not sources:
            result = run_fit(inputs)
            if result.get("status") == "failure":
                bundle = {
                    "status": "failure",
                    "message": result.get("message", "fiesta fit failed"),
                    "analysis": {},
                }
            else:
                bundle = _bundle_one(result)
        else:
            curves: dict = {}
            medians: dict = {}
            posteriors: dict = {}
            errors: dict = {}
            n_det = None
            for src in sources:
                sub = {**inputs, "analysis_parameters": {**params, "source": src}}
                try:
                    r = run_fit(sub)
                    if r.get("status") == "failure":
                        errors[src] = r.get("message", "fit failed")
                        continue
                    if r.get("model_lightcurve"):
                        curves[src] = r["model_lightcurve"]
                    if r.get("posterior_medians") is not None:
                        medians[src] = r["posterior_medians"]
                    if r.get("posterior_samples"):
                        posteriors[src] = r["posterior_samples"]
                    if r.get("n_detections") is not None:
                        n_det = r["n_detections"]
                except Exception as e:  # noqa: BLE001 — one model must not sink the rest
                    errors[src] = str(e)
            if not curves:
                bundle = {
                    "status": "failure",
                    "message": "; ".join(f"{k}: {v}" for k, v in errors.items()) or "no models fit",
                    "analysis": {},
                }
            else:
                msg = f"fiesta grouped fit: {len(curves)}/{len(sources)} models" + (
                    f", {len(errors)} failed" if errors else ""
                )
                bundle = {
                    "status": "success",
                    "message": msg,
                    "analysis": {
                        "model_lightcurves": curves,  # {model: {filter: [[mjd,med,lo,hi]]}}
                        "posteriors": posteriors,  # {model: {param: [samples]}} for corner plots
                        "n_detections": n_det,  # detections used (run versioning)
                        "results": {
                            "format": "json",
                            "data": {"models": medians, "errors": errors},
                        },
                    },
                }
    except Exception as e:  # noqa: BLE001 — every failure becomes a SkyPortal "failure" response
        bundle = {
            "status": "failure",
            "message": str(e),
            "analysis": {},
            "_traceback": traceback.format_exc()[-2048:],
        }

    # stdout is the always-available fallback for the plugin's poller.
    print(json.dumps(bundle))

    try:
        upload(bundle)
    except Exception as e:  # noqa: BLE001 — non-fatal; plugin can scrape stdout
        print(f"warning: OSDF upload failed: {e}", file=sys.stderr)

    return 0 if bundle["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
