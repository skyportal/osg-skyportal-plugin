"""
NMMA wrapper script — runs inside an HTCondor job on an OSG worker.

Contract:
- Reads ``inputs.json`` from cwd (staged by HTCondor file transfer).
- Runs NMMA's EM analysis (mirroring skyportal/skyportal#3199), OR a stub if
  ``analysis_parameters.dry_run`` is true.
- Bundles the result as a SkyPortal-shaped JSON
  ``{status, message, analysis: {results, plots, log}}``.
- Uploads the bundle to ``$OSDF_OUTPUT_URL`` via plain HTTPS PUT
  (bearer-authenticated with ``$BEARER_TOKEN`` or ``$BEARER_TOKEN_FILE``).
- Also writes the bundle to stdout so the plugin's poller can fall back to
  scraping the schedd log if OSDF is unreachable.

Real NMMA work behind a lazy import so this module is unit-testable on a
machine without NMMA installed.
"""

import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import requests


def load_inputs(path: Path = Path("inputs.json")) -> dict:
    return json.loads(path.read_text())


def run_nmma(inputs: dict) -> dict:
    """Invoke NMMA. Returns a dict with ``log_bayes_factor`` plus output paths on success.

    Lazy-imports NMMA so this module is importable on hosts without it.
    """
    params = inputs.get("analysis_parameters") or {}
    if params.get("dry_run"):
        return {
            "log_bayes_factor": 1.23,
            "_stub": True,
            "_source": params.get("source", "Me2017"),
        }

    # Delegated to nmma.skyportal_osg (lives in the NMMA repo) so plugins don't
    # re-implement the SkyPortal-payload → argv → NMMA-main bridge.
    from nmma.skyportal_osg import run_from_skyportal_inputs

    resource_id = inputs.get("resource_id", "obj")
    return run_from_skyportal_inputs(inputs, resource_id=resource_id)


def bundle_for_skyportal(
    nmma_result: dict,
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
    if nmma_result.get("_stub") and "results" not in analysis:
        analysis["results"] = {
            "format": "json",
            "data": base64.b64encode(json.dumps(nmma_result).encode()).decode(),
        }
    return {
        "status": "success",
        "message": f"log Bayes factor={nmma_result.get('log_bayes_factor')}",
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
        result = run_nmma(inputs)
        if result.get("status") == "failure":
            bundle = {
                "status": "failure",
                "message": result.get("message", "NMMA fit failed"),
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
