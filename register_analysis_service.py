"""
One-shot helper: POST /api/analysis_service to register this plugin's URL with
SkyPortal so users can run it from the source-analysis page.

Reads the same config block as main.py; pulls the SkyPortal base URL + token
from `services.external.osg.params.skyportal`.

Usage:
    uv run python register_analysis_service.py \\
        --name NMMA_OSG --display "NMMA on OSG" \\
        --listener-url http://my-host:7100/analysis/nmma_osg \\
        --group-ids 1 2
"""

import argparse
import json
import sys

import requests
from baselayer.app.env import load_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="AnalysisService name (no spaces)")
    p.add_argument("--display", required=True, help="Human-readable display_name")
    p.add_argument(
        "--listener-url",
        required=True,
        help="Plugin URL SkyPortal should POST to, e.g. http://host:7100/analysis/nmma_osg",
    )
    p.add_argument(
        "--analysis-type",
        default="lightcurve_fitting",
        help="SkyPortal analysis_type (default: lightcurve_fitting)",
    )
    p.add_argument(
        "--input-data-types",
        nargs="+",
        default=["photometry", "redshift"],
        help="SkyPortal input_data_types",
    )
    p.add_argument(
        "--optional-params-json",
        default='{"source": ["Me2017", "Piro2021", "nugent-hyper", "TrPi2018"], '
        '"fix_z": ["True", "False"]}',
        help="optional_analysis_parameters as JSON (string of dict)",
    )
    p.add_argument("--group-ids", nargs="+", type=int, default=[])
    p.add_argument(
        "--bearer",
        default=None,
        help="Bearer token to set as the AnalysisService's auth header. "
        "Falls back to auth.incoming_bearer_token from config.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    _, cfg = load_env()
    params = cfg["services.external.osg.params"]
    base = params["skyportal"]["base_url"].rstrip("/")
    api_token = params["skyportal"]["api_token"]
    if not api_token or api_token.startswith("replace_with"):
        print(
            "error: set services.external.osg.params.skyportal.api_token in config", file=sys.stderr
        )
        sys.exit(2)

    bearer = args.bearer or params.get("auth", {}).get("incoming_bearer_token")
    auth_payload = {"header_token": {"Authorization": f"Bearer {bearer}"}} if bearer else {}

    body = {
        "name": args.name,
        "display_name": args.display,
        "description": f"{args.display} (OSG plugin)",
        "version": "0.1",
        "contact_name": "osg-skyportal-plugin",
        "url": args.listener_url,
        "authentication_type": "header_token" if bearer else "none",
        "_authinfo": json.dumps(auth_payload) if auth_payload else None,
        "analysis_type": args.analysis_type,
        "input_data_types": args.input_data_types,
        "optional_analysis_parameters": args.optional_params_json,
        "group_ids": args.group_ids,
    }

    r = requests.post(
        f"{base}/api/analysis_service",
        json=body,
        headers={"Authorization": f"token {api_token}"},
        timeout=30,
    )
    print(f"HTTP {r.status_code}: {r.text}")
    r.raise_for_status()


if __name__ == "__main__":
    main()
