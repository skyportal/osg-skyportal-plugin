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


def stage_models(aframe_cfg: dict, job_dir, read_token_path: str | None = None, log=None):
    """Copy/fetch aframe's model files into ``job_dir`` under their canonical
    basenames. Each source may be a local path or an OSDF/HTTPS URL (fetched here
    so the worker needs no egress). Returns the staged paths; ``background`` is
    optional, a missing weights/config is only warned about (the job then fails)."""
    staged = []
    for key, dest in _MODELS:
        src = (aframe_cfg or {}).get(key)
        target = Path(job_dir) / dest
        if src and osdf.is_osdf_url(src):
            osdf.download(src, target, token_path=read_token_path)
            staged.append(target)
        elif src and Path(src).exists():
            shutil.copy(src, target)
            staged.append(target)
        elif key != "background" and log is not None:
            log(f"aframe: no `{key}` available (aframe.{key}); the job will fail without it")
    return staged
