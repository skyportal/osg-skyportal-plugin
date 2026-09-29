"""Stage aframe's model files for a job. Kept out of main.py (no baselayer
dependency) so the logic is unit-testable on its own, like alma_staging."""

from __future__ import annotations

import shutil
from pathlib import Path

import osdf

# Config key -> the canonical basename the aframe bridge looks for on the worker.
_MODELS = (
    ("weights", "aframe.pt"),
    ("config", "aframe_config_bbh.yaml"),
    ("background", "background.hdf5"),
)

# Schemes HTCondor's own file-transfer plugins fetch on the worker. With
# transfer_urls set, a source with one of these schemes is passed straight into
# transfer_input_files (optionally with a <token>+ prefix, e.g. igwn+osdf://) so
# the worker pulls it from the OSDF cache -- no download to the pod, no AP spool.
_URL_SCHEMES = ("osdf://", "https://", "http://", "stash://", "pelican://")


def _transfer_url(src: str) -> bool:
    s = str(src)
    scheme = s.split("://", 1)[0]
    if "+" in scheme:  # strip a credential prefix like igwn+osdf
        s = s.split("+", 1)[1]
    return s.startswith(_URL_SCHEMES)


def stage_models(aframe_cfg: dict, job_dir, read_token_path: str | None = None, log=None):
    """Make aframe's model files available to a job under their canonical basenames.

    With ``transfer_urls`` set, URL sources (osdf/https/...) are returned as-is for
    the worker to pull via transfer_input_files -- the URL must end in the canonical
    basename, since HTCondor names a URL input by its basename. Otherwise each source
    (local path or URL) is fetched into ``job_dir`` here, so the worker needs no
    egress. Returns the staged paths/URLs; ``background`` is optional, a missing
    weights/config is only warned about (the job then fails)."""
    passthrough = bool((aframe_cfg or {}).get("transfer_urls", False))
    staged: list = []
    for key, dest in _MODELS:
        src = (aframe_cfg or {}).get(key)
        if not src:
            if key != "background" and log is not None:
                log(f"aframe: no `{key}` available (aframe.{key}); the job will fail without it")
            continue
        if passthrough and _transfer_url(src):
            if Path(src).name != dest and log is not None:
                log(
                    f"aframe: {key} URL basename {Path(src).name!r} != canonical {dest!r}; "
                    "the worker receives it under its URL basename (upload as .../<ver>/"
                    f"{dest})"
                )
            staged.append(src)
            continue
        target = Path(job_dir) / dest
        if osdf.is_osdf_url(src):
            osdf.download(src, target, token_path=read_token_path)
            staged.append(target)
        elif Path(src).exists():
            shutil.copy(src, target)
            staged.append(target)
        elif key != "background" and log is not None:
            log(f"aframe: no `{key}` available (aframe.{key}); the job will fail without it")
    return staged
