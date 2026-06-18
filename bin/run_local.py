"""
Drive the REAL plugin service against OSPool end to end, locally: start the
tornado app (build_app) + poller_loop with an injected config, POST a
SkyPortal-shaped analysis request to ourselves, and catch the callback in the
same process. Validates submit -> spool -> poll -> retrieve -> stdout-bundle ->
callback without needing a full SkyPortal.

Needs baselayer importable (run with PYTHONPATH=<skyportal checkout>).

Usage (fast dry-run on a bare worker):
    PYTHONPATH=/path/to/skyportal python bin/run_local.py
Real fiesta fit in the image:
    PYTHONPATH=... python bin/run_local.py --image osdf:///.../nmma.sif \\
        --source Bu2025_MLP --class fiesta_kn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `import main`
import main

PHOTOMETRY = (
    "mjd,filter,mag,magerr\n"
    "60000.0,ztfg,18.5,0.1\n60000.5,ztfr,18.7,0.1\n60001.0,ztfg,19.2,0.1\n"
    "60001.5,ztfr,19.0,0.15\n60002.0,ztfg,20.0,0.2\n60002.0,ztfr,19.6,0.2\n"
)

_received: dict = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7199)
    p.add_argument("--image", default="", help="osdf:// image; empty = bare worker")
    p.add_argument("--source", default="Me2017")
    p.add_argument("--class", dest="klass", default="", help="--em-transient-class")
    p.add_argument("--dry-run", action="store_true", help="stub fit (fast, no NMMA)")
    p.add_argument("--nlive", type=int, default=32)
    p.add_argument("--timeout-seconds", type=int, default=2400)
    return p.parse_args()


def _start_receiver(port: int) -> None:
    """Threaded HTTP callback sink, separate from the plugin's event loop so the
    poller's synchronous post_callback doesn't deadlock against it."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _received["body"] = json.loads(self.rfile.read(length) or b"{}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


async def amain():
    args = parse_args()
    cb_port = args.port + 1
    _start_receiver(cb_port)
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
            # Dry-run is bare-worker (no image unpack), so a small disk matches faster.
            "request_disk": 2048 if args.dry_run else 20480,
            "max_runtime_seconds": 3600,
            "singularity_image": args.image or None,
        },
        "poller": {"interval_seconds": 30},
        "caps": {
            "max_concurrent_total": None,
            "max_concurrent_per_analysis": None,
            "max_concurrent_per_resource_per_analysis": None,
        },
        "osdf": {"output_prefix": None, "read_token_path": None, "write_token_path": None},
        "staging_dir": "staging-local",
        "auth": {"incoming_bearer_token": None},
        "skyportal": {"base_url": "http://localhost:5000", "api_token": "x"},
    }

    app = main.build_app(cfg)
    app.listen(args.port, address="127.0.0.1")
    asyncio.create_task(main.poller_loop(cfg))

    params = {"source": args.source, "nlive": args.nlive, "use_wrapper": True}
    if args.dry_run:
        params["dry_run"] = True
    if args.klass:
        params["em_transient_class"] = args.klass
    body = {
        "inputs": {"photometry": PHOTOMETRY, "analysis_parameters": params},
        "callback_url": f"http://127.0.0.1:{cb_port}/callback",
        "callback_method": "POST",
        "resource_id": "TESTKN",
    }
    # Off the event loop: the handler's submit_job is synchronous (spools to the AP).
    r = await asyncio.to_thread(
        requests.post, f"http://127.0.0.1:{args.port}/analysis/nmma", json=body, timeout=120
    )
    print("submit response:", r.status_code, r.json())

    deadline = asyncio.get_event_loop().time() + args.timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        if _received:
            b = _received["body"]
            print("CALLBACK status:", b.get("status"), "| message:", b.get("message"))
            print("  analysis keys:", list((b.get("analysis") or {}).keys()))
            return
        await asyncio.sleep(15)
    print("TIMEOUT waiting for callback")


if __name__ == "__main__":
    asyncio.run(amain())
