"""
Run a real NMMA fit through the wrapper on OSPool, inside the NMMA image: spool
up the wrapper + inputs.json + a photometry CSV, run fiesta_wrapper.py in the
container (which calls the skyportal_osg bridge), retrieve the SkyPortal bundle.

Defaults to the Bu2025_MLP fiesta (JAX) kilonova surrogate; priors + surrogate
are baked into the image, so nothing downloads at runtime. Use --source/--class
for other models (e.g. --source Me2017 --class "" for the analytic model).

Usage:
    python bin/trigger_fiesta.py --image osdf:///ospool/ap41/data/<user>/nmma.sif
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# A short synthetic kilonova-ish light curve (ztf g/r), trigger-relative ~2-4 d.
PHOTOMETRY = """mjd,filter,mag,magerr
60000.0,ztfg,18.5,0.1
60000.5,ztfr,18.7,0.1
60001.0,ztfg,19.2,0.1
60001.5,ztfr,19.0,0.15
60002.0,ztfg,20.0,0.2
60002.0,ztfr,19.6,0.2
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="osdf:// URL of the NMMA .sif")
    p.add_argument("--project", default="UMN_Coughlin")
    p.add_argument("--source", default="Bu2025_MLP", help="NMMA --model")
    p.add_argument("--class", dest="klass", default="fiesta_kn", help="--em-transient-class")
    p.add_argument("--nlive", type=int, default=32)
    p.add_argument("--wrapper", default="fiesta_wrapper.py")
    p.add_argument("--poll-seconds", type=int, default=45)
    p.add_argument("--timeout-seconds", type=int, default=1800)
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

    workdir = Path(tempfile.mkdtemp(prefix="osg-fiesta-"))
    (workdir / "photometry.csv").write_text(PHOTOMETRY)
    params = {"source": args.source, "nlive": args.nlive}
    if args.klass:
        params["em_transient_class"] = args.klass
    # photometry is read at payload top level by the bridge, not analysis_parameters.
    (workdir / "inputs.json").write_text(
        json.dumps({"photometry": "photometry.csv", "analysis_parameters": params})
    )
    out_file = workdir / "wrapper.out"
    err_file = workdir / "wrapper.err"
    print(f"workdir: {workdir}\n  source={args.source} class={args.klass or '(infer)'}")

    submit_desc = {
        "executable": "/usr/bin/python3",
        "transfer_executable": "False",
        "arguments": wrapper.name,
        "transfer_input_files": f"{wrapper},{workdir}/inputs.json,{workdir}/photometry.csv",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": out_file.name,
        "error": err_file.name,
        "request_cpus": "1",
        "request_memory": "6GB",
        # Explicit units: bare RequestDisk is KiB in HTCondor. The worker unpacks
        # the ~1.3 GB image into scratch + JAX cache + outputs.
        "request_disk": "20GB",
        "requirements": '(Arch == "X86_64") && (OpSys == "LINUX")',
        "+ProjectName": f'"{args.project}"',
        "+SingularityImage": f'"{args.image}"',
        "+SkyPortalAnalysisName": '"nmma_fiesta_osg"',
        "initialdir": str(workdir),
    }

    schedd = htcondor.Schedd()
    sub = htcondor.Submit(submit_desc)
    result = schedd.submit(sub, count=1, spool=True)
    cluster_id = result.cluster()
    schedd.spool(result)
    print(f"submitted + spooled cluster_id={cluster_id} (image pull + fit takes a few min)")

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
            hr = live[0].get("HoldReason")
            print(f"  {time.strftime('%H:%M:%S')} JobStatus={st} HoldReason={hr}")
            if st == 4:
                print("completed (awaiting retrieve)")
                break
            time.sleep(max(args.poll_seconds, 30))
            continue
        break
    else:
        sys.exit(f"timed out after {args.timeout_seconds}s")

    schedd.retrieve(f"ClusterId == {cluster_id}")
    if err_file.exists() and err_file.read_text().strip():
        print(f"=== stderr (tail) ===\n{err_file.read_text()[-1500:]}")
    if not out_file.exists():
        sys.exit(f"error: {out_file} not retrieved")
    text = out_file.read_text()
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            bundle = json.loads(line)
        except json.JSONDecodeError:
            sys.exit(f"error: last stdout line is not JSON:\n{text[-1500:]}")
        if isinstance(bundle, dict) and bundle.get("status"):
            print(f"\nBUNDLE: status={bundle['status']} message={bundle.get('message')!r}")
            print(f"  analysis keys: {list(bundle.get('analysis', {}).keys())}")
            return
        sys.exit(f"error: parsed JSON has no status: {bundle!r}")
    sys.exit("error: empty stdout")


if __name__ == "__main__":
    main()
