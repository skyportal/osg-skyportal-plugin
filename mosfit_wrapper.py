"""
mosfit wrapper — runs inside the mosfit runtime image on an OSG worker.

Contract (same as fiesta_wrapper):
- Reads ``inputs.json`` from cwd (staged by HTCondor file transfer).
- Runs a MOSFiT fit via ``mosfit_bridge`` (shipped alongside this file).
- Bundles the result as SkyPortal's analysis-callback JSON
  ``{status, message, analysis: {model_name, model_lightcurve, posterior_samples,
  n_detections, results, plots}}`` — the same overlay contract as fiesta.
- Optionally PUTs the bundle to ``$OSDF_OUTPUT_URL``; always writes it to stdout
  so the plugin's poller can scrape it.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from pathlib import Path

try:
    import requests  # only needed for OSDF upload; bare workers may lack it
except ImportError:
    requests = None


def load_inputs(path: Path = Path("inputs.json")) -> dict:
    return json.loads(path.read_text())


def _materialize_inputs(inputs: dict) -> dict:
    """SkyPortal sends photometry/redshift as CSV content; write inline CSV to
    files so the bridge reads them via astropy as paths."""
    for key, fname in (("photometry", "photometry.csv"), ("redshift", "redshift.csv")):
        val = inputs.get(key)
        if isinstance(val, str) and "\n" in val:
            path = Path(fname)
            path.write_text(val)
            inputs[key] = str(path)
    return inputs


def run(inputs: dict) -> dict:
    from mosfit_bridge import run_from_skyportal_inputs  # shipped per-job

    inputs = _materialize_inputs(inputs)
    resource_id = inputs.get("resource_id", "obj")
    return run_from_skyportal_inputs(inputs, resource_id=resource_id)


def bundle_for_skyportal(result: dict) -> dict:
    """Pack the bridge output into SkyPortal's analysis-callback schema, carrying
    the per-filter model light curve + posterior samples through for the overlay
    and corner plot (same shape fiesta produces)."""
    if result.get("status") == "failure":
        return {
            "status": "failure",
            "message": result.get("message", "mosfit failed"),
            "analysis": {},
        }

    analysis: dict = {}
    # Model name so SkyPortal labels the overlay by the fitted model, not the
    # generic analysis-service name.
    if result.get("source"):
        analysis["model_name"] = result["source"]
    result_file = result.get("json_result_file")
    if result_file and Path(result_file).exists():
        # "json" format expects the object itself, not a base64 string.
        analysis["results"] = {"format": "json", "data": json.loads(Path(result_file).read_text())}
    plot_file = result.get("plot_file")
    if plot_file and Path(plot_file).exists():
        analysis["plots"] = [
            {"format": "png", "data": base64.b64encode(Path(plot_file).read_bytes()).decode()}
        ]
    if result.get("model_lightcurve"):
        analysis["model_lightcurve"] = result["model_lightcurve"]
    if result.get("posterior_samples"):
        analysis["posterior_samples"] = result["posterior_samples"]
    if result.get("n_detections") is not None:
        analysis["n_detections"] = result["n_detections"]
    return {
        "status": "success",
        "message": result.get("message", "fit complete"),
        "analysis": analysis,
    }


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
        bundle = bundle_for_skyportal(run(load_inputs()))
    except Exception as e:  # noqa: BLE001 — every failure becomes a SkyPortal "failure"
        bundle = {
            "status": "failure",
            "message": str(e),
            "analysis": {},
            "_traceback": traceback.format_exc()[-2048:],
        }

    print(json.dumps(bundle))
    try:
        upload(bundle)
    except Exception as e:  # noqa: BLE001 — non-fatal; plugin can scrape stdout
        print(f"warning: OSDF upload failed: {e}", file=sys.stderr)

    return 0 if bundle["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
