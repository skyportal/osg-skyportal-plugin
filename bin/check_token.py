"""
Print SciToken expiry + scopes from a token file. Best-effort JWT decode (no
signature verify; we just want lifetime + audience for ops sanity).

Usage:
    python bin/check_token.py [token_path]

Defaults to $BEARER_TOKEN_FILE if no path is passed.
"""

import base64
import datetime as dt
import json
import os
import sys
from pathlib import Path


def _b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def decode_jwt_payload(token: str) -> dict:
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("token is not a JWT (expected 3 dot-separated segments)")
    return json.loads(_b64url_decode(parts[1]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BEARER_TOKEN_FILE")
    if not path:
        print("error: pass a token path or set BEARER_TOKEN_FILE", file=sys.stderr)
        sys.exit(2)
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"error: no such file {path}", file=sys.stderr)
        sys.exit(2)

    token = Path(path).read_text().strip()
    try:
        payload = decode_jwt_payload(token)
    except Exception as e:  # noqa: BLE001
        print(f"could not decode JWT payload: {e}", file=sys.stderr)
        sys.exit(1)

    now = dt.datetime.now(dt.timezone.utc)
    exp = payload.get("exp")
    iat = payload.get("iat")
    nbf = payload.get("nbf")

    print(f"file:   {path}")
    print(f"issuer: {payload.get('iss', '?')}")
    print(f"sub:    {payload.get('sub', '?')}")
    print(f"aud:    {payload.get('aud', '?')}")
    print(f"scope:  {payload.get('scope', '?')}")
    if iat:
        print(f"iat:    {dt.datetime.fromtimestamp(iat, dt.timezone.utc).isoformat()}")
    if nbf:
        print(f"nbf:    {dt.datetime.fromtimestamp(nbf, dt.timezone.utc).isoformat()}")
    if exp:
        exp_dt = dt.datetime.fromtimestamp(exp, dt.timezone.utc)
        delta = exp_dt - now
        print(f"exp:    {exp_dt.isoformat()}  ({_format_delta(delta)} from now)")
        if delta.total_seconds() < 0:
            print("warning: token is EXPIRED", file=sys.stderr)
            sys.exit(3)
        if delta.total_seconds() < 600:
            print("warning: token expires in under 10 minutes", file=sys.stderr)


def _format_delta(delta: dt.timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h}h{m:02d}m{s:02d}s"


if __name__ == "__main__":
    main()
