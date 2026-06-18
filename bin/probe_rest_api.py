"""
Probe an HTCondor schedd for a usable REST submission API.

Tries the documented HTCondor REST endpoint families against the configured
schedd host with a SciToken bearer, reports what answers. Use BEFORE betting on
bin/trigger_via_rest.py.

Usage:
    python bin/probe_rest_api.py [--host HOST] [--token PATH]

Defaults: host from $OSG_SCHEDD_HOST or `ldas-osg.ligo.caltech.edu`;
token from $BEARER_TOKEN_FILE or `/tmp/bt_u<uid>` (htgettoken default).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

# Candidate URL templates as the HTCondor REST API has gone through several
# iterations. The probe just tries them all and reports the verdict.
URL_CANDIDATES = [
    "https://{host}/api/v1/jobs",
    "https://{host}:9618/api/v1/jobs",
    "https://{host}:8200/api/v1/jobs",
    "https://{host}/condor/api/v1/jobs",
    "https://{host}/htcondor/api/v1/jobs",
]


def default_token_path() -> str:
    if env := os.environ.get("BEARER_TOKEN_FILE"):
        return env
    return f"/tmp/bt_u{os.getuid()}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("OSG_SCHEDD_HOST", "ldas-osg.ligo.caltech.edu"))
    p.add_argument("--token", default=default_token_path())
    p.add_argument("--timeout", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    token_path = Path(os.path.expanduser(args.token))
    if not token_path.exists():
        print(
            f"warning: SciToken not found at {token_path}; probing without bearer auth",
            file=sys.stderr,
        )
        headers = {}
    else:
        token = token_path.read_text().strip()
        headers = {"Authorization": f"Bearer {token}"}
        print(f"using SciToken at {token_path} ({len(token)} chars)")

    print(f"probing {args.host}...")
    print()

    found_any = False
    for tmpl in URL_CANDIDATES:
        url = tmpl.format(host=args.host)
        try:
            r = requests.get(url, headers=headers, timeout=args.timeout, verify=True)
            verdict = f"HTTP {r.status_code}"
            if r.status_code < 400:
                verdict += " ✓ (looks live)"
                found_any = True
            elif r.status_code == 401:
                verdict += " (auth required — token may be missing/expired/wrong-scope)"
                found_any = True  # endpoint exists, just rejecting us
            elif r.status_code == 404:
                verdict += " (endpoint not at this path)"
            else:
                verdict += f" — body: {r.text[:120]}"
        except requests.exceptions.SSLError as e:
            verdict = f"TLS error: {e!s}"
        except requests.exceptions.ConnectionError as e:
            verdict = f"connection refused / DNS: {type(e).__name__}"
        except requests.exceptions.Timeout:
            verdict = "timeout"
        except Exception as e:  # noqa: BLE001
            verdict = f"{type(e).__name__}: {e}"
        print(f"  {url}")
        print(f"    -> {verdict}")
        print()

    if not found_any:
        print(
            "No REST endpoint responded. The schedd likely doesn't expose a public REST API.",
            file=sys.stderr,
        )
        print(
            "Fall back to: ssh to the AP and run bin/trigger_via_bindings.py there.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("At least one candidate URL responded — see verdicts above for the live one.")


if __name__ == "__main__":
    main()
