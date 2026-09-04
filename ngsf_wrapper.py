"""
NGSF wrapper — runs inside the NGSF runtime image on an OSG worker.

Contract (same as periodfind_wrapper):
- Reads ``inputs.json`` from cwd (staged by HTCondor file transfer).
- Runs a spectral classification via ``ngsf_bridge`` (shipped alongside this file).
- Bundles the result as SkyPortal's analysis-callback JSON
  ``{status, message, analysis: {results, plots, annotations}}``; the
  annotations carry the best-match type, redshift and chi2/dof.
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


def run(inputs: dict) -> dict:
    from ngsf_bridge import run_from_skyportal_inputs  # shipped per-job

    resource_id = inputs.get("resource_id", "obj")
    return run_from_skyportal_inputs(inputs, resource_id=resource_id, work_dir=".")


def bundle_for_skyportal(result: dict) -> dict:
    """Pack the bridge output into SkyPortal's analysis-callback schema."""
    if result.get("status") == "failure":
        return {
            "status": "failure",
            "message": result.get("message", "NGSF failed"),
            "analysis": {},
        }

    analysis: dict = {}
    if result.get("results") is not None:
        # "json" format: the object itself, not a base64 string.
        analysis["results"] = {"format": "json", "data": result["results"]}

    plots = []
    for path in result.get("plot_files") or []:
        png = Path(path)
        if png.exists():
            plots.append({"format": "png", "data": base64.b64encode(png.read_bytes()).decode()})
    if plots:
        analysis["plots"] = plots

    if result.get("annotations"):
        analysis["annotations"] = result["annotations"]
    return {
        "status": "success",
        "message": result.get("message", "NGSF complete"),
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
