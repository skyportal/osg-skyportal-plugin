"""
Validate spool-based file transfer to the OSPool, end to end: stage a local
input file up via spool, run a job that reads it and writes stdout, then
retrieve the output back. Proves the exact submit/spool/retrieve mechanics used
by main.py:submit_job before we wire NMMA through them.

Run from a laptop after bin/setup_ospool.py (uses ~/.condor auto-discovery).

Usage:
    python bin/trigger_transfer.py [--project UMN_Coughlin]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="UMN_Coughlin", help="+ProjectName tag (empty = omit)")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--timeout-seconds", type=int, default=900)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import htcondor  # noqa: PLC0415
    except ModuleNotFoundError:
        import htcondor2 as htcondor

    workdir = Path(tempfile.mkdtemp(prefix="osg-transfer-"))
    input_file = workdir / "hello.txt"
    input_file.write_text("transfer-roundtrip-ok\n")
    output_file = workdir / "result.out"
    print(f"workdir: {workdir}")

    submit_desc = {
        "executable": "/bin/cat",
        "transfer_executable": "False",  # use the worker's /bin/cat, not the Mac's
        "arguments": input_file.name,  # basename: it lands in the job's cwd
        "transfer_input_files": str(input_file),
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": output_file.name,
        "request_cpus": "1",
        "request_memory": "256",
        "request_disk": "256",
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+SkyPortalAnalysisName": '"ospool_transfer_smoke"',
        "+SkyPortalDryRun": "true",
        "initialdir": str(workdir),
    }
    if args.project:
        submit_desc["+ProjectName"] = f'"{args.project}"'

    schedd = htcondor.Schedd()
    sub = htcondor.Submit(submit_desc)
    result = schedd.submit(sub, count=1, spool=True)
    cluster_id = result.cluster()
    schedd.spool(result)
    print(f"submitted + spooled cluster_id={cluster_id}")

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
            st = int(ad["JobStatus"])
            print(f"  JobStatus={st} HoldReason={ad.get('HoldReason')}")
            if st == 4:  # spooled jobs sit in the queue as completed until retrieved
                print("completed (in queue, awaiting retrieve)")
                break
            time.sleep(max(args.poll_seconds, 30))
            continue
        history = list(
            schedd.history(
                constraint=f"ClusterId == {cluster_id}",
                projection=["ClusterId", "JobStatus", "ExitCode"],
                match=1,
            )
        )
        if history:
            ad = history[0]
            print(f"completed: JobStatus={ad['JobStatus']} ExitCode={ad.get('ExitCode')}")
            break
        time.sleep(max(args.poll_seconds, 30))
    else:
        sys.exit(f"timed out after {args.timeout_seconds}s")

    schedd.retrieve(f"ClusterId == {cluster_id}")
    if output_file.exists():
        print(f"retrieved {output_file}:\n  {output_file.read_text()!r}")
        print("SPOOL ROUNDTRIP OK" if "ok" in output_file.read_text() else "unexpected content")
    else:
        sys.exit(f"error: {output_file} not retrieved")


if __name__ == "__main__":
    main()
