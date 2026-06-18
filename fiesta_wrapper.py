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
    """Fit via fiesta_bridge (lazy import so this module loads without fiesta).
    Returns a stub when analysis_parameters.dry_run is set."""
    params = inputs.get("analysis_parameters") or {}
    # SkyPortal sends optional params as strings, so "False" must be falsy.
    dry_run = str(params.get("dry_run", "")).strip().lower() in ("true", "1", "yes", "t")
    if dry_run:
        return {
            "log_bayes_factor": 1.23,
            "_stub": True,
            "_source": params.get("source", "Bu2025_MLP"),
        }

    from fiesta_bridge import run_from_skyportal_inputs  # shipped with this wrapper

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
    if result_file is not None and result_file.exists():
        analysis["results"] = {
            "format": result_file.suffix.lstrip(".") or "bin",
            "data": base64.b64encode(result_file.read_bytes()).decode(),
        }
    if fit_result.get("_stub") and "results" not in analysis:
        analysis["results"] = {
            "format": "json",
            "data": base64.b64encode(json.dumps(fit_result).encode()).decode(),
        }
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


def main() -> int:
    try:
        inputs = load_inputs()
        result = run_fit(inputs)
        if result.get("status") == "failure":
            bundle = {
                "status": "failure",
                "message": result.get("message", "fiesta fit failed"),
                "analysis": {},
            }
        else:
            plot_files = []
            if pf := result.get("plot_file"):
                plot_files.append(Path(pf))
            result_file = (
                Path(result["json_result_file"]) if result.get("json_result_file") else None
            )
            bundle = bundle_for_skyportal(result, plot_files=plot_files, result_file=result_file)
            # Carry the per-filter model light curve through for the SkyPortal
            # photometry-plot overlay (median + band per filter, on MJD).
            if result.get("model_lightcurve"):
                bundle["analysis"]["model_lightcurve"] = result["model_lightcurve"]
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
