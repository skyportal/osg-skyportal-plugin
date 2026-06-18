"""
Smoke-test remote OSPool submission end to end: submit a self-contained sleep
job (NO transfer_input_files, so nothing has to spool) and confirm auth +
schedd discovery work before we tackle real payloads.

Run from a laptop after ``bin/setup_ospool.py``; relies on ~/.condor/user_config
+ the IDTOKEN in ~/.condor/tokens.d/, both auto-discovered by bare Schedd().

Usage:
    python bin/trigger_sleep.py [--project GW-Astro] [--seconds 30] [--poll]
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="", help="HTCondor +ProjectName tag (empty = omit)")
    p.add_argument("--seconds", type=int, default=30, help="how long the job sleeps")
    p.add_argument("--poll", action="store_true", help="block until completion")
    p.add_argument("--poll-seconds", type=int, default=30, help="schedd query interval (>=30)")
    p.add_argument("--timeout-seconds", type=int, default=900)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import htcondor  # noqa: PLC0415 — Linux AP installs ship the v1 shim
    except ModuleNotFoundError:
        import htcondor2 as htcondor  # conda-forge macOS only ships v2

    schedd = htcondor.Schedd()
    print(f"schedd located: {schedd}")

    # No output/error/log files: OSPool requires the event log inside the AP
    # home dir, and a sleep job has no stdout. JobStatus polling is enough to
    # prove auth + discovery + execution.
    submit_desc = {
        "executable": "/bin/sleep",
        "transfer_executable": "False",  # use the worker's /bin/sleep, not the Mac's
        "arguments": str(args.seconds),
        "request_cpus": "1",
        "request_memory": "256",
        "request_disk": "256",
        # Bindings submission from macOS otherwise defaults requirements to the
        # local platform (arm64/macOS) and matches zero OSPool slots.
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+SkyPortalAnalysisName": '"ospool_smoke"',
        "+SkyPortalDryRun": "true",
    }
    if args.project:
        submit_desc["+ProjectName"] = f'"{args.project}"'
    # Spool even this no-input job: remote submission otherwise inherits the
    # laptop's cwd as Iwd and the AP holds it ("cannot access working dir").
    sub = htcondor.Submit(submit_desc)
    if hasattr(schedd, "transaction"):  # htcondor v1
        with schedd.transaction() as txn:
            cluster_id = sub.queue(txn, count=1)
    else:  # htcondor2: Schedd.submit() returns a SubmitResult
        result = schedd.submit(sub, count=1, spool=True)
        schedd.spool(result)
        cluster_id = result.cluster()
    print(f"submitted cluster_id={cluster_id} (sleep {args.seconds}s)")

    if not args.poll:
        print(f"  watch with: condor_q -name <ap> -constraint 'ClusterId == {cluster_id}'")
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
            time.sleep(max(args.poll_seconds, 30))
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
            return
        time.sleep(max(args.poll_seconds, 30))
    sys.exit(f"timed out after {args.timeout_seconds}s")


if __name__ == "__main__":
    main()
