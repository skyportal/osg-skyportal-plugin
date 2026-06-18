"""
Submit a dry-run NMMA-wrapper job to an HTCondor schedd via its REST API,
poll until completion, print the resulting bundle.

This is the laptop-side smoke test for the OSG plugin. Requires:
- A SciToken valid for the target schedd (htgettoken puts it at /tmp/bt_u<uid>).
- The schedd's REST endpoint exposed and reachable. Confirm with
  ``bin/probe_rest_api.py`` first.

Usage:
    python bin/trigger_via_rest.py --schedd-url URL [--inputs PATH] [--poll-seconds N]

Defaults: --inputs examples/dry_run_inputs.json; polls every 10s up to 300s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def default_token_path() -> str:
    if env := os.environ.get("BEARER_TOKEN_FILE"):
        return env
    return f"/tmp/bt_u{os.getuid()}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--schedd-url",
        required=True,
        help="Base URL the probe found live, e.g. https://ldas-osg.ligo.caltech.edu/api/v1",
    )
    p.add_argument("--token", default=default_token_path())
    p.add_argument("--inputs", default="examples/dry_run_inputs.json")
    p.add_argument("--wrapper", default="fiesta_wrapper.py")
    p.add_argument("--project", default="IGWN", help="HTCondor +ProjectName tag")
    p.add_argument("--poll-seconds", type=int, default=10)
    p.add_argument("--timeout-seconds", type=int, default=300)
    return p.parse_args()


def _headers(token_path: str) -> dict[str, str]:
    p = Path(os.path.expanduser(token_path))
    if not p.exists():
        sys.exit(f"error: SciToken not found at {p}; run htgettoken first")
    return {
        "Authorization": f"Bearer {p.read_text().strip()}",
        "Content-Type": "application/json",
    }


def submit(args, headers: dict) -> int:
    inputs_path = Path(args.inputs)
    wrapper_path = Path(args.wrapper)
    if not inputs_path.exists():
        sys.exit(f"error: inputs file not found at {inputs_path}")
    if not wrapper_path.exists():
        sys.exit(f"error: wrapper not found at {wrapper_path}")

    # The HTCondor REST submit shape isn't yet rock-stable across deployments;
    # this is the canonical "single-job submit description" body. Tweak per
    # what probe_rest_api.py finds out there.
    body = {
        "submit": {
            "executable": "/usr/bin/python3",
            "arguments": wrapper_path.name,
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT",
            "transfer_input_files": f"{wrapper_path},{inputs_path}",
            "output": "logs/job.$(ClusterId).$(ProcId).out",
            "error": "logs/job.$(ClusterId).$(ProcId).err",
            "log": "logs/job.$(ClusterId).log",
            "request_cpus": "1",
            "request_memory": "256",
            "request_disk": "256",
            "+ProjectName": f'"{args.project}"',
            "+SkyPortalAnalysisName": '"nmma_osg_smoke"',
            "+SkyPortalDryRun": "true",
        },
        "count": 1,
    }
    url = f"{args.schedd_url.rstrip('/')}/submit"
    print(f"POST {url}")
    r = requests.post(url, json=body, headers=headers, timeout=60)
    if r.status_code >= 400:
        sys.exit(f"submit failed (HTTP {r.status_code}): {r.text[:500]}")
    resp = r.json()
    cluster_id = (
        resp.get("cluster_id")
        or resp.get("ClusterId")
        or (resp.get("jobs") or [{}])[0].get("ClusterId")
    )
    if cluster_id is None:
        sys.exit(f"could not extract ClusterId from response: {resp}")
    print(f"submitted: cluster_id={cluster_id}")
    return int(cluster_id)


def poll_until_done(args, headers: dict, cluster_id: int) -> dict:
    """Poll the REST API until the job leaves the live queue, then fetch its history."""
    base = args.schedd_url.rstrip("/")
    deadline = time.time() + args.timeout_seconds
    last_status = None
    while time.time() < deadline:
        r = requests.get(
            f"{base}/jobs?constraint=ClusterId=={cluster_id}",
            headers=headers,
            timeout=30,
        )
        live = (r.json().get("jobs") if r.ok else None) or []
        if live:
            last_status = live[0].get("JobStatus")
            print(f"  cluster {cluster_id}: JobStatus={last_status}")
            time.sleep(args.poll_seconds)
            continue
        # Not in live queue — try history.
        h = requests.get(
            f"{base}/history?constraint=ClusterId=={cluster_id}",
            headers=headers,
            timeout=30,
        )
        if h.ok and h.json().get("jobs"):
            return h.json()["jobs"][0]
        time.sleep(args.poll_seconds)
    sys.exit(f"timed out after {args.timeout_seconds}s; last live JobStatus={last_status}")


def main():
    args = parse_args()
    headers = _headers(args.token)
    cluster_id = submit(args, headers)
    print(f"polling every {args.poll_seconds}s for up to {args.timeout_seconds}s...")
    final = poll_until_done(args, headers, cluster_id)
    print()
    print("=== final ad ===")
    print(json.dumps(final, indent=2, default=str)[:2000])
    out = final.get("Out") or final.get("output")
    if out:
        print()
        print("=== stdout ===")
        print(out)
        try:
            bundle = json.loads(out.splitlines()[-1])
            print()
            print("=== parsed SkyPortal bundle ===")
            print(json.dumps(bundle, indent=2)[:2000])
        except (json.JSONDecodeError, IndexError):
            pass


if __name__ == "__main__":
    main()
