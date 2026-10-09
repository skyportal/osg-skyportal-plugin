"""
OSDF / Pelican data plane via plain HTTPS.

Pelican origins expose HTTPS endpoints; uploads are `PUT` and downloads are
`GET`, both bearer-authenticated with a SciToken. This module is intentionally
CLI-free (no `pelican` binary dependency) so the plugin can run in minimal
containers.
"""

import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

# OSDF federation director; it routes a request to the namespace's origin (writes)
# or a cache (reads). Overridable for a different federation.
OSDF_DIRECTOR = os.environ.get("OSDF_DIRECTOR", "https://osdf-director.osg-htc.org")

# Resolved write origins, cached per process (the director mapping is stable but
# can move, so a failed write re-probes with force=True).
_write_origin_cache: dict[str, str] = {}

# Bound on origin re-resolution per upload, so an expired token (a permanent 403,
# or a persistent redirect) cannot become a probe loop around every transfer.
MAX_UPLOAD_ATTEMPTS = 2

# Minted bearer tokens cached per namespace. The cache is keyed on the token's own
# exp, not a wall clock, so a retry hours after the first attempt re-mints rather
# than reusing a token that was valid when the first attempt began.
_token_cache: dict[str, tuple[str, float]] = {}
TOKEN_REFRESH_MARGIN_S = 300


class UploadNeedsRetry(RuntimeError):
    """The origin redirected mid-PUT; the streamed body is spent, so the caller
    must re-open the source and upload again (the origin has been re-resolved)."""


def _bearer(token_path: str | None) -> str | None:
    """Return the SciToken bytes from disk, or None if unavailable."""
    path = os.path.expanduser(token_path) if token_path else None
    if path and os.path.exists(path):
        return Path(path).read_text().strip()
    env = os.environ.get("BEARER_TOKEN_FILE")
    if env and os.path.exists(env):
        return Path(env).read_text().strip()
    return None


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_osdf_url(url: str) -> bool:
    """Identify URLs that should round-trip through OSDF rather than local FS."""
    if not url:
        return False
    scheme = urlparse(url).scheme.lower()
    return scheme in {"http", "https", "osdf", "pelican"}


def upload(local_path: Path, remote_url: str, token_path: str | None = None) -> None:
    """PUT a single file to an OSDF/Pelican origin via HTTPS (resolve-first)."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    with local_path.open("rb") as f:
        upload_stream(
            remote_url, f, token_path=token_path, content_length=local_path.stat().st_size
        )


def _namespace(path: str) -> str:
    """First path segment, e.g. /umn-coughlin/a/b.tar -> umn-coughlin."""
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else ""


def resolve_origin_base(object_url: str, force: bool = False) -> str:
    """The origin base URL (scheme://host:port) serving writes for the object's
    namespace, from the director's 307 for a bodyless PUT to a SENTINEL path.

    The sentinel matters: an empty PUT truncates a resource, so probing the target
    object could zero a staged tarball if a token is ever attached or the namespace
    is loose; the sentinel is never read or written for real. Resolution is
    unauthenticated: it needs only the director's redirect, not the write token.
    Cached per namespace; force=True re-probes after an origin move."""
    p = urlparse(object_url)
    ns = _namespace(p.path)
    if not ns:
        raise ValueError(f"no namespace in {object_url}")
    if not force and ns in _write_origin_cache:
        return _write_origin_cache[ns]
    r = requests.put(f"{OSDF_DIRECTOR}/{ns}/.probe", data=b"", allow_redirects=False, timeout=60)
    loc = r.headers.get("Location")
    if r.status_code in (307, 308) and loc:
        lp = urlparse(loc)
        base = f"{lp.scheme}://{lp.netloc}"
        _write_origin_cache[ns] = base
        return base
    raise RuntimeError(f"OSDF write-origin resolution for {ns} returned HTTP {r.status_code}")


def _jwt_exp(token: str) -> float:
    """The exp claim (epoch seconds) of a JWT, or 0 if unreadable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
    except Exception:  # noqa: BLE001
        return 0.0


def mint_token(
    keypair_path: str, object_url: str, pelican_bin: str = "pelican", scope: str = "write"
) -> str:
    """Mint a short-lived bearer token for the object's namespace from the Pelican
    keypair credential file (secret.pem), via the pelican client.

    The keypair carries offline_access, so this re-mints with no browser. The file
    is not static: pelican rewrites it in place when it refreshes, spending a
    single-use refresh token, so keypair_path must name a writable copy that
    outlives the process and no second copy may be used in parallel. Cached per
    namespace and re-minted within TOKEN_REFRESH_MARGIN_S of the token's own expiry,
    so a retry long after the first attempt gets a fresh token rather than a 403."""
    ns = _namespace(urlparse(object_url).path)
    cached = _token_cache.get(ns)
    if cached and cached[1] - time.time() > TOKEN_REFRESH_MARGIN_S:
        return cached[0]
    env = {**os.environ, "PELICAN_CLIENT_CREDENTIALFILE": os.path.expanduser(keypair_path)}
    proc = subprocess.run(
        [pelican_bin, "credentials", "token", "get", scope, f"osdf:///{ns}", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pelican token mint failed (exit {proc.returncode}): {proc.stderr[-300:]}"
        )
    m = re.search(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", proc.stdout)
    if not m:
        raise RuntimeError("pelican token mint produced no JWT")
    token = m.group(0)
    _token_cache[ns] = (token, _jwt_exp(token) or time.time() + 1800)
    return token


def upload_stream(
    object_url: str,
    source,
    token_path: str | None = None,
    keypair_path: str | None = None,
    pelican_bin: str = "pelican",
    content_length: int | None = None,
    timeout: int = 7200,
    _attempt: int = 0,
) -> str:
    """Stream `source` (a file-like) to the OSDF object at `object_url` via an
    authenticated HTTPS PUT straight to the resolved origin.

    Resolve-first is mandatory: a streaming body cannot replay across a redirect
    (requests raises UnrewindableBodyError) and requests strips the auth header
    cross-host, so PUT-to-director-and-follow fails either way. We send the token
    only to the resolved origin and refuse to follow any further redirect, so a
    misroute fails at byte zero rather than mid-transfer. content_length is declared
    and the bytes read are verified against it, so a truncated upload fails loudly
    rather than the worker later failing to untar. Returns the origin URL written."""
    p = urlparse(object_url)
    base = resolve_origin_base(object_url, force=(_attempt > 0))
    target = f"{base}/{p.path.lstrip('/')}"
    # A keypair mints a fresh token (checked against its own exp, so a retry re-mints);
    # otherwise a static bearer from token_path.
    token = (
        mint_token(keypair_path, object_url, pelican_bin) if keypair_path else _bearer(token_path)
    )
    headers = _headers(token)
    counter = {"n": 0}
    body = source
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
        body = _CountingReader(source, counter)
    r = requests.put(target, data=body, headers=headers, allow_redirects=False, timeout=timeout)
    if r.status_code in (307, 308):
        if _attempt + 1 >= MAX_UPLOAD_ATTEMPTS:
            raise RuntimeError(f"OSDF origin kept redirecting writes for {object_url}")
        resolve_origin_base(object_url, force=True)  # refresh the cached origin
        raise UploadNeedsRetry(object_url)  # body is spent; caller re-opens source
    if r.status_code in (401, 403):
        raise PermissionError(
            f"OSDF PUT to {target} refused (HTTP {r.status_code}); the write token may be "
            f"missing or expired"
        )
    r.raise_for_status()
    if content_length is not None and counter["n"] != content_length:
        raise OSError(
            f"OSDF upload to {object_url} truncated: sent {counter['n']} of {content_length} bytes"
        )
    return target


class _CountingReader:
    """Wrap a file-like so a PUT can verify it sent the declared Content-Length."""

    def __init__(self, source, counter: dict):
        self._source = source
        self._counter = counter

    def read(self, size=-1):
        chunk = self._source.read(size)
        self._counter["n"] += len(chunk)
        return chunk


def _read_url(url: str) -> str:
    """HTTPS form for a GET: an osdf://|pelican:// (or igwn+osdf://) path goes
    through the director, which redirects to a cache; https:// passes through."""
    p = urlparse(url)
    scheme = p.scheme.split("+")[-1].lower()
    if scheme in {"osdf", "pelican"}:
        return f"{OSDF_DIRECTOR}/{p.path.lstrip('/')}"
    return url


def download(remote_url: str, local_path: Path, token_path: str | None = None) -> Path:
    """GET a single file from an OSDF/Pelican origin and stream it to disk. A GET
    carries no body, so following the director's redirect to a cache is safe."""
    token = _bearer(token_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        _read_url(remote_url), headers=_headers(token), stream=True, timeout=300
    ) as r:
        r.raise_for_status()
        with local_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    return local_path
