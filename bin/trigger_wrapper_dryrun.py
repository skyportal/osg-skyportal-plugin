"""
Integration checkpoint: submit fiesta_wrapper.py in dry-run mode to OSPool via
spool, retrieve its stdout, and confirm the SkyPortal JSON bundle round-trips.
This exercises the exact path main.py uses (wrapper staging + spool + retrieve +
stdout-bundle parsing) on the real AP.

Dry-run skips the NMMA import, so it runs on a bare worker's /usr/bin/python3.
Pass --image to instead run inside the Apptainer NMMA image (production shape).

Run from the repo root after bin/setup_ospool.py.

Usage:
    python bin/trigger_wrapper_dryrun.py [--project UMN_Coughlin] [--image osdf:///.../nmma.sif]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="UMN_Coughlin", help="+ProjectName tag (empty = omit)")
    p.add_argument("--image", default="", help="Apptainer image; empty = bare worker")
    p.add_argument("--wrapper", default="fiesta_wrapper.py")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--timeout-seconds", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    wrapper = Path(args.wrapper).resolve()
    if not wrapper.exists():
        sys.exit(f"error: {wrapper} not found (run from the plugin repo root)")
    try:
        import htcondor  # noqa: PLC0415
    except ModuleNotFoundError:
        import htcondor2 as htcondor

    workdir = Path(tempfile.mkdtemp(prefix="osg-wrapper-"))
    (workdir / "inputs.json").write_text(
        json.dumps({"analysis_parameters": {"dry_run": True, "source": "Me2017"}})
    )
    out_file = workdir / "wrapper.out"
    err_file = workdir / "wrapper.err"
    print(f"workdir: {workdir}")

    submit_desc = {
        "executable": "/usr/bin/python3",
        "transfer_executable": "False",  # use the worker's/container's python3
        "arguments": wrapper.name,
        "transfer_input_files": f"{wrapper},{workdir}/inputs.json",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": out_file.name,
        "error": err_file.name,
        "request_cpus": "1",
        "request_memory": "512",
        "request_disk": "512",
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+ProjectName": f'"{args.project}"' if args.project else '""',
        "+SkyPortalAnalysisName": '"nmma_dryrun_smoke"',
        "+SkyPortalDryRun": "true",
        "initialdir": str(workdir),
    }
    if args.image:
        submit_desc["+SingularityImage"] = f'"{args.image}"'

    schedd = htcondor.Schedd()
    sub = htcondor.Submit(submit_desc)
    result = schedd.submit(sub, count=1, spool=True)
    cluster_id = result.cluster()
    schedd.spool(result)
    print(f"submitted + spooled cluster_id={cluster_id} (image={args.image or 'bare worker'})")

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        live = list(
            schedd.query(
                constraint=f"ClusterId == {cluster_id}",
                projection=["ClusterId", "JobStatus", "HoldReason"],
            )
        )
        if live:
            st = int(live[0]["JobStatus"])
            print(f"  JobStatus={st} HoldReason={live[0].get('HoldReason')}")
            if st == 4:  # spooled jobs sit in the queue as completed until retrieved
                break
            time.sleep(max(args.poll_seconds, 30))
            continue
        break  # left the queue
    else:
        sys.exit(f"timed out after {args.timeout_seconds}s")

    schedd.retrieve(f"ClusterId == {cluster_id}")
    if err_file.exists() and err_file.read_text().strip():
        print(f"=== retrieved stderr ===\n{err_file.read_text()[:2000]}")
    if not out_file.exists():
        sys.exit(f"error: {out_file} not retrieved")
    text = out_file.read_text()
    print(f"=== retrieved stdout ({len(text)} bytes) ===\n{text[:2000]}")

    # Mirror main.py._read_stdout_bundle: the wrapper bundle is the last JSON line.
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            bundle = json.loads(line)
        except json.JSONDecodeError:
            sys.exit("error: last stdout line is not the JSON bundle")
        if isinstance(bundle, dict) and bundle.get("status"):
            print(f"\nBUNDLE OK: status={bundle['status']} message={bundle.get('message')!r}")
            return
        sys.exit(f"error: parsed JSON has no status: {bundle!r}")
    sys.exit("error: empty stdout")


if __name__ == "__main__":
    main()
