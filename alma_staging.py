"""Fetch ALMA products on the plugin host, ready to ship with the job.

The execute node is not assumed to reach the archive, so the delivered products
are downloaded here and transferred in with the job. Only the pipeline products
are taken by default: the raw ASDM alongside them is roughly twenty times the
size and is only worth fetching to re-image, which this path does not do.
"""

from __future__ import annotations

from pathlib import Path

ARCHIVE_ANNOTATION_ORIGIN = "alma-archive"

# Datalink marks each file's role.
PRODUCT_SEMANTICS = "#this"
AUXILIARY_SEMANTICS = "#auxiliary"
PROGENITOR_SEMANTICS = "#progenitor"

# A ceiling on what one job is allowed to carry, so a pathological dataset
# cannot wedge the submit host or the transfer.
DEFAULT_MAX_BYTES = 2 * 1024**3


def classify_products(rows) -> dict:
    """Split datalink rows by role, dropping the ones with nothing to fetch.

    Rows are dicts with `access_url`, `semantics` and `content_length`; the
    archive returns placeholder rows with an empty URL, which are skipped.
    """
    out = {"products": [], "auxiliary": [], "progenitor": [], "other": []}
    bucket_for = {
        PRODUCT_SEMANTICS: "products",
        AUXILIARY_SEMANTICS: "auxiliary",
        PROGENITOR_SEMANTICS: "progenitor",
    }
    for row in rows or []:
        url = (row.get("access_url") or "").strip()
        if not url:
            continue
        try:
            size = int(row.get("content_length") or 0)
        except (TypeError, ValueError):
            size = 0
        out[bucket_for.get(str(row.get("semantics")), "other")].append(
            {
                "url": url,
                "filename": url.rsplit("/", 1)[-1],
                "bytes": size,
                "semantics": row.get("semantics"),
            }
        )
    return out


def staging_plan(rows, include_auxiliary=False, include_progenitor=False) -> dict:
    """The files to stage for a reduction, and what they weigh."""
    classified = classify_products(rows)
    files = list(classified["products"])
    if include_auxiliary:
        files += classified["auxiliary"]
    if include_progenitor:
        files += classified["progenitor"]
    return {
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "available": {
            role: sum(f["bytes"] for f in entries)
            for role, entries in classified.items()
            if entries
        },
    }


def dataset_uids(inputs: dict) -> list[str]:
    """Datasets to stage: named outright, else read off the coverage annotation.

    SkyPortal's alma_archive service records the uids it found at the source
    position, so a request usually needs no parameters at all.
    """
    params = (inputs or {}).get("analysis_parameters") or {}
    explicit = params.get("dataset_uids")
    if isinstance(explicit, str):
        explicit = [u.strip() for u in explicit.split(",") if u.strip()]
    if explicit:
        return list(explicit)

    for annotation in _annotation_rows(inputs):
        if annotation.get("origin") != ARCHIVE_ANNOTATION_ORIGIN:
            continue
        data = annotation.get("data") or {}
        if isinstance(data, dict) and data.get("datasets"):
            return list(data["datasets"])
    return []


def _annotation_rows(inputs: dict) -> list[dict]:
    """Annotations as SkyPortal sends them: CSV text, or already-parsed rows."""
    raw = (inputs or {}).get("annotations")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    import csv
    import io
    import json

    rows = []
    for row in csv.DictReader(io.StringIO(raw)):
        data = row.get("data")
        if isinstance(data, str):
            try:
                row["data"] = json.loads(data)
            except (ValueError, TypeError):
                try:
                    import ast

                    row["data"] = ast.literal_eval(data)
                except (ValueError, SyntaxError):
                    row["data"] = {}
        rows.append(row)
    return rows


def datalink_rows(uid: str) -> list[dict]:
    """What the archive offers for one dataset."""
    from astroquery.alma import Alma

    table = Alma.get_data_info(uid)
    rows = []
    for row in table:
        rows.append(
            {
                "access_url": str(row["access_url"]),
                "semantics": str(row["semantics"]),
                "content_length": row["content_length"]
                if "content_length" in table.colnames
                else 0,
            }
        )
    return rows


def stage(
    inputs: dict,
    dest: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_auxiliary: bool = False,
) -> tuple[list[Path], list[str]]:
    """Download the products for this request into `dest`.

    Returns the staged paths and any notes worth surfacing (datasets skipped
    for size, datasets the archive had nothing downloadable for).
    """
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    uids = dataset_uids(inputs)
    if not uids:
        return [], ["No ALMA datasets named in the request or its annotations"]

    staged, notes, budget = [], [], max_bytes
    for uid in uids:
        try:
            plan = staging_plan(datalink_rows(uid), include_auxiliary=include_auxiliary)
        except Exception as e:  # noqa: BLE001 -- one bad uid must not lose the rest
            notes.append(f"{uid}: could not read datalink ({e})")
            continue
        if not plan["files"]:
            notes.append(f"{uid}: no delivered products on offer")
            continue
        if plan["total_bytes"] > budget:
            notes.append(
                f"{uid}: skipped, {plan['total_bytes'] / 1e6:.0f} MB exceeds the "
                f"{budget / 1e6:.0f} MB left for this job"
            )
            continue

        for entry in plan["files"]:
            target = dest / entry["filename"]
            with requests.get(entry["url"], stream=True, timeout=600) as response:
                response.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            staged.append(target)
            budget -= target.stat().st_size

    if not staged and not notes:
        notes.append("Nothing was staged for this request")
    return staged, notes
