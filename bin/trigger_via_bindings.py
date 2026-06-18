"""
Fallback to bin/trigger_via_rest.py: submit a dry-run NMMA-wrapper job using
the native htcondor Python bindings. Run this ON THE ACCESS POINT (the host
that already has htcondor installed and your SciToken in scope) — not from
a macOS laptop where htcondor has no wheel.

Usage (on ldas-osg or similar):
    python bin/trigger_via_bindings.py [--project IGWN] [--inputs PATH]

Then ``condor_q -af ClusterId JobStatus Out`` watches it; or pass
``--poll`` to block until completion and dump the bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="examples/dry_run_inputs.json")
    p.add_argument("--wrapper", default="fiesta_wrapper.py")
    p.add_argument("--project", default="IGWN", help="HTCondor +ProjectName tag")
    p.add_argument("--poll", action="store_true", help="block until completion + print bundle")
    p.add_argument("--poll-seconds", type=int, default=10)
    p.add_argument("--timeout-seconds", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    inputs_path = Path(args.inputs).resolve()
    wrapper_path = Path(args.wrapper).resolve()
    for p in (inputs_path, wrapper_path):
        if not p.exists():
            sys.exit(f"error: {p} not found (run from the plugin repo root)")

    import htcondor  # noqa: PLC0415 — fail loudly if not installed

    Path("logs").mkdir(exist_ok=True)
    submit_desc = {
        "executable": "/usr/bin/python3",
        "arguments": wrapper_path.name,
        "transfer_input_files": f"{wrapper_path},{inputs_path}",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": "logs/job.$(ClusterId).$(ProcId).out",
        "error": "logs/job.$(ClusterId).$(ProcId).err",
        "log": "logs/job.$(ClusterId).log",
        "request_cpus": "1",
        "request_memory": "256",
        "request_disk": "256",
        "+ProjectName": f'"{args.project}"',
        "+SkyPortalAnalysisName": '"nmma_osg_smoke"',
        "+SkyPortalDryRun": "true",
    }
    sub = htcondor.Submit(submit_desc)
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn, count=1)
    print(f"submitted cluster_id={cluster_id}")
    print("  inputs:", inputs_path)
    print("  wrapper:", wrapper_path)

    if not args.poll:
        constraint = f"ClusterId == {cluster_id}"
        print(f"  watch with: condor_q -af ClusterId JobStatus -constraint '{constraint}'")
        return

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        live = list(
            schedd.query(
                constraint=f"ClusterId == {cluster_id}",
                projection=["ClusterId", "JobStatus", "HoldReason"],
            )
        )
        if live:
            ad = live[0]
            print(f"  JobStatus={ad['JobStatus']} HoldReason={ad.get('HoldReason')}")
            time.sleep(args.poll_seconds)
            continue
        history = list(
            schedd.history(
                constraint=f"ClusterId == {cluster_id}",
                projection=["ClusterId", "JobStatus", "ExitCode", "CompletionDate"],
                match=1,
            )
        )
        if history:
            ad = history[0]
            print(f"completed: JobStatus={ad['JobStatus']} ExitCode={ad.get('ExitCode')}")
            stdout_path = Path(f"logs/job.{cluster_id}.0.out")
            if stdout_path.exists():
                text = stdout_path.read_text()
                print()
                print("=== stdout ===")
                print(text[:4000])
                try:
                    bundle = json.loads(text.splitlines()[-1])
                    print()
                    print("=== parsed SkyPortal bundle ===")
                    print(json.dumps(bundle, indent=2)[:2000])
                except (json.JSONDecodeError, IndexError):
                    pass
            return
        time.sleep(args.poll_seconds)
    sys.exit(f"timed out after {args.timeout_seconds}s")


if __name__ == "__main__":
    main()
