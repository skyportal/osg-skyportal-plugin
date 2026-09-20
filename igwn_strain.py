"""Fetch real IGWN strain on an OSPool worker: discover OSDF frame URLs with
gwdatafind and pull them with Pelican, using the CVMFS igwn env for whichever
tool the job image lacks (the pycbc image has no gwdatafind; the buoy image no
pelican). Authenticated by the job's ``use_oauth_services`` scitoken.

Shipped alongside the stdlib-only bridges, so nothing heavy is imported at load;
gwdatafind/gwpy run in the CVMFS env via subprocess, not in-process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# CVMFS is bind-mounted into the job container, so this env is reachable from any
# image and carries gwdatafind + gwpy + pelican.
IGWN_ENV = "/cvmfs/software.igwn.org/conda/envs/igwn"
DEFAULT_GWDATAFIND_HOST = "datafind.igwn.org"


def _igwn_python() -> str:
    p = f"{IGWN_ENV}/bin/python"
    return p if os.path.exists(p) else "python3"


def _pelican() -> str:
    # An in-image pelican (pycbc) wins; else the CVMFS env's (buoy has none).
    return shutil.which("pelican") or f"{IGWN_ENV}/bin/pelican"


def _token_env() -> dict:
    """Point token consumers at the scitoken the AP delivered via use_oauth_services."""
    env = dict(os.environ)
    tok = env.get("BEARER_TOKEN_FILE")
    if not tok:
        creds = env.get("_CONDOR_CREDS")
        cand = os.path.join(creds, "scitokens.use") if creds else ""
        if cand and os.path.exists(cand):
            tok = cand
    if tok:
        env["BEARER_TOKEN_FILE"] = tok
    return env


def find_osdf_urls(observatory, frametype, start, end, host=DEFAULT_GWDATAFIND_HOST):
    """OSDF frame URLs for ``[start, end]`` from gwdatafind (``urltype='osdf'``),
    run in the CVMFS igwn env since the image may not ship gwdatafind. An empty
    list means no strain is indexed for that span (e.g. no active run)."""
    code = (
        "import json, gwdatafind; "
        f"print(json.dumps(gwdatafind.find_urls({observatory!r}, {frametype!r}, "
        f"{int(start)}, {int(end)}, urltype='osdf', host={host!r}, on_gaps='ignore')))"
    )
    r = subprocess.run(
        [_igwn_python(), "-c", code],
        capture_output=True,
        text=True,
        env=_token_env(),
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"gwdatafind lookup failed: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout.strip() or "[]")


def _pelican_get(url, dest, env):
    r = subprocess.run(
        [_pelican(), "object", "get", url, str(dest)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if r.returncode != 0 or not Path(dest).exists():
        raise RuntimeError(f"pelican get failed for {url}: {r.stderr.strip()[:300]}")
    return Path(dest)


def fetch_frames(observatory, frametype, start, end, outdir, host=DEFAULT_GWDATAFIND_HOST):
    """Discover + fetch the frames covering ``[start, end]`` into ``outdir``.
    Returns the local ``.gwf`` paths; an empty list means no strain is available
    for that span, which the caller should surface as a clean 'no data' failure."""
    urls = find_osdf_urls(observatory, frametype, start, end, host=host)
    if not urls:
        return []
    env = _token_env()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [_pelican_get(u, outdir / u.rsplit("/", 1)[-1], env) for u in urls]
