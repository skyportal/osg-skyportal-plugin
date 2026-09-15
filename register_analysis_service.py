"""
Register this plugin's URL with SkyPortal so users can run it from the
source-analysis page, or update a registration that is already there.

Re-running with the same --name updates the existing service rather than
failing on the duplicate, so a changed parameter set is one command away.

Reads the same config block as main.py; pulls the SkyPortal base URL + token
from `services.external.osg.params.skyportal`.

Usage:
    uv run python register_analysis_service.py \\
        --name NMMA_OSG --display "NMMA on OSG" \\
        --listener-url http://my-host:7100/analysis/fiesta_osg \\
        --group-ids 1 2
"""

import argparse
import json
import sys

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="AnalysisService name (no spaces)")
    p.add_argument("--display", required=True, help="Human-readable display_name")
    p.add_argument(
        "--listener-url",
        required=True,
        help="Plugin URL SkyPortal should POST to, e.g. http://host:7100/analysis/fiesta_osg",
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
    p.add_argument("--token", default=None, help="SkyPortal API token (else from config)")
    p.add_argument("--base-url", default=None, help="SkyPortal base URL (else from config)")
    p.add_argument(
        "--bearer",
        default=None,
        help="Bearer token to set as the AnalysisService's auth header. "
        "Falls back to auth.incoming_bearer_token from config.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # --token/--base-url bypass load_env so this runs without the skyportal config.
    if args.token and args.base_url:
        base = args.base_url.rstrip("/")
        api_token = args.token
        bearer = args.bearer
    else:
        from baselayer.app.env import load_env  # needs PYTHONPATH=<skyportal>

        _, cfg = load_env()
        params = cfg["services.external.osg.params"]
        base = params["skyportal"]["base_url"].rstrip("/")
        api_token = params["skyportal"]["api_token"]
        if not api_token or api_token.startswith("replace_with"):
            print(
                "error: set services.external.osg.params.skyportal.api_token in config",
                file=sys.stderr,
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

    headers = {"Authorization": f"token {api_token}"}

    # A service is identified by name, and the parameter set is the thing most
    # likely to change after the first registration, so update in place rather
    # than making the caller find the id or delete and lose the analyses.
    existing = requests.get(f"{base}/api/analysis_service", headers=headers, timeout=30)
    existing.raise_for_status()
    match = next(
        (s for s in (existing.json().get("data") or []) if s.get("name") == args.name),
        None,
    )

    if match:
        r = requests.patch(
            f"{base}/api/analysis_service/{match['id']}",
            json=body,
            headers=headers,
            timeout=30,
        )
        print(f"updated {args.name} (id {match['id']}) -- HTTP {r.status_code}: {r.text}")
    else:
        r = requests.post(f"{base}/api/analysis_service", json=body, headers=headers, timeout=30)
        print(f"registered {args.name} -- HTTP {r.status_code}: {r.text}")
    r.raise_for_status()


if __name__ == "__main__":
    main()
