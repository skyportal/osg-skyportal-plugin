"""
One-time setup for remote submission to the OSPool from a laptop, per the OSG
recipe (gist xamberl/e73e05ebee37e0cf8b0b3db88582e44d): fetch an IDTOKEN from
the Access Point over ssh and write ~/.condor/user_config so bare
``htcondor.Schedd()`` resolves to the OSPool AP.

Usage:
    python bin/setup_ospool.py --user user.name

Defaults target ap41/cm-1; override with --ap / --collector if OSG moves you.
The token is auto-discovered from ~/.condor/tokens.d/ — no BEARER_TOKEN_FILE
needed for the schedd connection.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="your OSG username (the ssh login on the AP)")
    p.add_argument("--ap", default="ap41.uw.osg-htc.org", help="Access Point / SCHEDD_HOST")
    p.add_argument("--collector", default="cm-1.ospool.osg-htc.org", help="COLLECTOR_HOST")
    p.add_argument(
        "--token-name",
        default="ap41",
        help="filename under ~/.condor/tokens.d/ to store the IDTOKEN as",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing token / user_config")
    return p.parse_args()


def fetch_token(user: str, ap: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"token already at {dest} (pass --force to refetch); skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching IDTOKEN: ssh {user}@{ap} condor_token_fetch -> {dest}")
    with dest.open("wb") as f:
        proc = subprocess.run(["ssh", f"{user}@{ap}", "condor_token_fetch"], stdout=f, check=False)
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        sys.exit(f"error: condor_token_fetch failed (ssh exit {proc.returncode})")
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        sys.exit("error: fetched token is empty — check your AP login")
    dest.chmod(0o600)
    print(f"  wrote {dest.stat().st_size} bytes, chmod 600")


def write_user_config(ap: str, collector: str, force: bool) -> None:
    cfg = Path(os.path.expanduser("~/.condor/user_config"))
    body = f"SCHEDD_HOST = {ap}\nCOLLECTOR_HOST = {collector}\n"
    if cfg.exists() and not force:
        existing = cfg.read_text()
        if body in existing:
            print(f"user_config already points at {ap}; leaving as-is")
            return
        sys.exit(f"error: {cfg} exists and differs (pass --force to overwrite)")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(body)
    print(f"wrote {cfg}:\n  SCHEDD_HOST = {ap}\n  COLLECTOR_HOST = {collector}")


def main():
    args = parse_args()
    token_dest = Path(os.path.expanduser(f"~/.condor/tokens.d/{args.token_name}"))
    fetch_token(args.user, args.ap, token_dest, args.force)
    write_user_config(args.ap, args.collector, args.force)
    print("\ndone. verify with:  python bin/trigger_sleep.py --poll")


if __name__ == "__main__":
    main()
