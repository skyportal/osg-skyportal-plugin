"""Fetch real IGWN strain on an OSPool worker: discover OSDF frame URLs with
gwdatafind and pull them with Pelican. Prefer the job image's own tools (the
pycbc image ships gwdatafind + pelican) so a worker never cold-loads the large
CVMFS igwn conda env — that cold-load, not the ~64 MB/s transfer, is what makes
naive runs crawl. For an OSDF client the order is: in-image ``pelican`` CLI, a
static ``pelican`` shipped with the job (for images like buoy that lack one),
``pelicanfs`` (the Python client, once the image adds it), then the CVMFS env —
which isn't mounted on every OSPool node. Authenticated by the job's
``use_oauth_services`` scitoken.

Shipped alongside the stdlib-only bridges, so nothing heavy is imported at load.
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


def per_ifo(params, key, default_template, ifo):
    """A per-detector value: a dict keyed by ifo, a ``{ifo}`` template string, or
    the default template. Shared by the pygrb + aframe bridges."""
    val = (params or {}).get(key)
    if isinstance(val, dict):
        return val.get(ifo, default_template.format(ifo=ifo))
    if isinstance(val, str) and val:
        return val.format(ifo=ifo)
    return default_template.format(ifo=ifo)


def _igwn_python() -> str:
    p = f"{IGWN_ENV}/bin/python"
    return p if os.path.exists(p) else "python3"


def _pelican_exe():
    """A usable pelican CLI, or None. In-image wins (pycbc ships one); else a
    binary shipped with the job (./pelican, for images like buoy that lack it);
    else the CVMFS env's, which isn't mounted on every OSPool node."""
    exe = shutil.which("pelican")
    if exe:
        return exe
    shipped = Path.cwd() / "pelican"
    if shipped.exists():
        shipped.chmod(0o755)  # the executable bit may not survive file transfer
        return str(shipped)
    cvmfs = f"{IGWN_ENV}/bin/pelican"
    return cvmfs if os.path.exists(cvmfs) else None


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
    """OSDF frame URLs for ``[start, end]`` from gwdatafind (``urltype='osdf'``).
    An empty list means no strain is indexed for that span (e.g. no active run).

    Prefer the in-image gwdatafind (imported in-process — no subprocess, no CVMFS)
    and fall back to the CVMFS igwn env only when the image lacks it."""
    try:
        import gwdatafind  # in-image (the pycbc image ships it)
    except ImportError:
        gwdatafind = None
    if gwdatafind is not None:
        # gwdatafind reads the scitoken via igwn-auth-utils (BEARER_TOKEN_FILE).
        tok = _token_env().get("BEARER_TOKEN_FILE")
        if tok:
            os.environ["BEARER_TOKEN_FILE"] = tok
        urls = gwdatafind.find_urls(
            observatory,
            frametype,
            int(start),
            int(end),
            urltype="osdf",
            host=host,
            on_gaps="ignore",
        )
        return list(urls)

    # Fallback: run gwdatafind in the CVMFS igwn env.
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


def _pelicanfs_get(url, dest, env):
    """Fetch via pelicanfs (fsspec) — the Python OSDF client. No CLI, no CVMFS;
    used when the image ships pelicanfs (e.g. buoy, once added)."""
    import fsspec  # pelicanfs registers the osdf:// protocol on import
    import pelicanfs  # noqa: F401 — import for the side-effect of registration

    tok = env.get("BEARER_TOKEN_FILE")
    if tok:
        os.environ["BEARER_TOKEN_FILE"] = tok  # pelicanfs reads it via igwn-auth-utils
    with fsspec.open(url, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    if not Path(dest).exists():
        raise RuntimeError(f"pelicanfs get produced no file for {url}")
    return Path(dest)


def _pelican_get(url, dest, env):
    """Pull one frame over OSDF. Prefer a pelican CLI; fall back to pelicanfs
    (Python) so an image with neither the CLI nor CVMFS still works."""
    exe = _pelican_exe()
    if exe:
        r = subprocess.run(
            [exe, "object", "get", url, str(dest)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        if r.returncode == 0 and Path(dest).exists():
            return Path(dest)
        cli_err = r.stderr.strip()[:300]
    else:
        cli_err = "no pelican CLI available"
    try:
        return _pelicanfs_get(url, dest, env)
    except ImportError:
        raise RuntimeError(f"pelican get failed for {url}: {cli_err}") from None


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
