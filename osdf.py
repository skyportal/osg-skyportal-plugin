"""
OSDF / Pelican data plane via plain HTTPS.

Pelican origins expose HTTPS endpoints; uploads are `PUT` and downloads are
`GET`, both bearer-authenticated with a SciToken. This module is intentionally
CLI-free (no `pelican` binary dependency) so the plugin can run in minimal
containers.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import requests


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
    """PUT a single file to an OSDF/Pelican origin via HTTPS."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    token = _bearer(token_path)
    with local_path.open("rb") as f:
        r = requests.put(remote_url, data=f, headers=_headers(token), timeout=300)
    r.raise_for_status()


def download(remote_url: str, local_path: Path, token_path: str | None = None) -> Path:
    """GET a single file from an OSDF/Pelican origin and stream it to disk."""
    token = _bearer(token_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(remote_url, headers=_headers(token), stream=True, timeout=300) as r:
        r.raise_for_status()
        with local_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    return local_path
