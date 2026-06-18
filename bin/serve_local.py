"""
Run the plugin as a persistent local service (build_app + poller_loop) with an
injected config, so SkyPortal can POST analysis requests to it. Unlike
run_local.py, this does not self-drive — it just serves until killed.

Needs baselayer importable (run with PYTHONPATH=<skyportal checkout>).

Usage:
    PYTHONPATH=/path/to/skyportal python bin/serve_local.py --port 7100
    # with the NMMA image for real fits:
    PYTHONPATH=... python bin/serve_local.py --image osdf:///.../nmma.sif
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `import main`
import main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7100)
    p.add_argument("--image", default="", help="osdf:// image; empty = bare worker")
    return p.parse_args()


async def amain():
    args = parse_args()
    cfg = {
        "listener": {"host": "127.0.0.1", "port": args.port},
        "htcondor": {
            "collector": None,
            "schedd": None,
            "scitoken_path": "/nonexistent",
            "project_name": "UMN_Coughlin",
            "spool": True,
        },
        "defaults": {
            "request_cpus": 1,
            "request_memory": 6144,
            "request_disk": 20480 if args.image else 2048,
            "max_runtime_seconds": 3600,
            "singularity_image": args.image or None,
            "use_wrapper": True,  # NMMA service: every analysis is a wrapper fit
        },
        "poller": {"interval_seconds": 30},
        "caps": {
            "max_concurrent_total": 50,
            "max_concurrent_per_analysis": 20,
            # null so a single object can run many models concurrently.
            "max_concurrent_per_resource_per_analysis": None,
        },
        "osdf": {"output_prefix": None, "read_token_path": None, "write_token_path": None},
        "staging_dir": "staging-local",
        "auth": {"incoming_bearer_token": None},
        "skyportal": {"base_url": "http://localhost:5000", "api_token": "x"},
    }
    app = main.build_app(cfg)
    app.listen(args.port, address="127.0.0.1")
    print(f"plugin listening on 127.0.0.1:{args.port} (image={args.image or 'bare worker'})")
    asyncio.create_task(main.poller_loop(cfg))
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(amain())
