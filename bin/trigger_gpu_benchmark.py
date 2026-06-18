"""
Submit the fiesta afterglow benchmark to an OSPool GPU slot and report timing.
Pairs with running the same script on CPU (JAX_PLATFORMS=cpu) for a GPU-vs-CPU
comparison of the JAX sampler (the analytical model is cheap, so the sampler
dominates).

Usage:
    python bin/trigger_gpu_benchmark.py --image osdf:///.../nmma-gpu-v1.sif \\
        --script /tmp/benchmark_afterglow_rel.py --data /tmp/grb_data.dat
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--project", default="UMN_Coughlin")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--poll-seconds", type=int, default=45)
    p.add_argument("--timeout-seconds", type=int, default=2400)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import htcondor  # noqa: PLC0415
    except ModuleNotFoundError:
        import htcondor2 as htcondor

    wd = Path(tempfile.mkdtemp(prefix="gpu-bench-"))
    script = wd / "benchmark.py"
    script.write_text(Path(args.script).read_text())
    (wd / "data.dat").write_text(Path(args.data).read_text())
    out = wd / "bench.out"
    err = wd / "bench.err"
    print(f"workdir: {wd}")

    desc = {
        "executable": "/usr/bin/python3",
        "transfer_executable": "False",
        "arguments": "benchmark.py",
        "transfer_input_files": f"{script},{wd}/data.dat",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": out.name,
        "error": err.name,
        "request_cpus": "1",
        "request_gpus": str(args.gpus),
        "request_memory": "8GB",
        # 7 GB CUDA image unpacks into scratch.
        "request_disk": "25GB",
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+ProjectName": f'"{args.project}"',
        "+SingularityImage": f'"{args.image}"',
        "+SkyPortalAnalysisName": '"gpu_benchmark"',
        "initialdir": str(wd),
    }
    schedd = htcondor.Schedd()
    sub = htcondor.Submit(desc)
    result = schedd.submit(sub, count=1, spool=True)
    cid = result.cluster()
    schedd.spool(result)
    print(f"submitted + spooled cluster_id={cid} (GPU slot + 7GB image pull takes a while)")

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        live = list(
            schedd.query(constraint=f"ClusterId == {cid}", projection=["JobStatus", "HoldReason"])
        )
        if live:
            st = int(live[0]["JobStatus"])
            print(f"  {time.strftime('%H:%M:%S')} JobStatus={st} {live[0].get('HoldReason') or ''}")
            if st == 4:
                break
            time.sleep(max(args.poll_seconds, 30))
            continue
        break
    else:
        sys.exit("timed out")

    schedd.retrieve(f"ClusterId == {cid}")
    if err.exists() and err.read_text().strip():
        tail = "\n".join(err.read_text().splitlines()[-15:])
        print(f"=== stderr tail ===\n{tail}")
    if out.exists():
        print(f"=== stdout ===\n{out.read_text()}")
    else:
        print("no stdout retrieved")


if __name__ == "__main__":
    main()
