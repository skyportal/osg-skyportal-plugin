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

# Ceilings on what one job carries. Delivered-product size varies by more than
# an order of magnitude between datasets (tens of MB to hundreds), so the count
# is the useful bound and the byte budget is the backstop.
DEFAULT_MAX_BYTES = 1024**3
DEFAULT_MAX_DATASETS = 1
# The OSG access point refuses to send a single spooled input file larger than
# this, so a bigger tarball is a job that is held after the transfer, not a slow
# one. ALMA delivers one tarball per dataset, so this bounds the dataset too.
DEFAULT_MAX_FILE_BYTES = 5000 * 1000**2


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


# Keep this much pod disk free when spooling a small product, so a download can
# never fill the shared web pod's root filesystem.
DEFAULT_MIN_FREE_BYTES = 5 * 1024**3
# Re-open the archive stream this many times if the origin redirects mid-PUT (the
# streamed body is spent and cannot replay).
_OSDF_RETRIES = 2


def _stream_to_osdf(url, object_url, token_path, content_length):
    """Stream one archive product straight into the OSDF origin, never landing it
    on pod disk. The body can't replay across a redirect, so on an origin move we
    re-open the archive GET and upload again, bounded."""
    import requests

    import osdf

    for attempt in range(_OSDF_RETRIES):
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            response.raw.decode_content = True  # hand osdf decoded bytes, not gzip
            length = content_length or int(response.headers.get("Content-Length") or 0) or None
            try:
                osdf.upload_stream(
                    object_url, response.raw, token_path=token_path, content_length=length
                )
                return length or content_length or 0
            except osdf.UploadNeedsRetry:
                if attempt + 1 >= _OSDF_RETRIES:
                    raise
    return 0


def stage(
    inputs: dict,
    dest: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_auxiliary: bool = False,
    max_datasets: int = DEFAULT_MAX_DATASETS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    osdf_url_base: str | None = None,
    osdf_token_path: str | None = None,
    cluster_uuid: str | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> tuple[list[str], int, list[str]]:
    """Stage the products for this request, returning (transfer items, bytes, notes).

    Each item is a local path to spool or an ``osdf://`` URL for the worker to pull.
    A product over ``max_file_bytes`` is streamed to ``osdf_url_base/<cluster_uuid>/``
    when OSDF is configured (never landing on pod disk), else skipped as before.
    Smaller products are downloaded to ``dest`` and spooled, refused if the pod lacks
    free space. ``bytes`` is the total staged volume, from the archive content length
    rather than stat (an OSDF item has no local file to stat), and drives job sizing.
    """
    import shutil

    import requests

    dest.mkdir(parents=True, exist_ok=True)
    uids = dataset_uids(inputs)
    if not uids:
        return [], 0, ["No ALMA datasets named in the request or its annotations"]
    notes: list[str] = []

    # A source can carry dozens of datasets; fetching every one that fits the
    # byte budget pulls gigabytes for a reduction that reads a few cubes. The
    # bound is on datasets actually staged, not on candidates considered: the
    # archive lists plenty with no delivered products at all, and stopping at
    # those would strand a request whose usable data sits further down.
    transfer: list[str] = []
    staged_bytes, budget, taken = 0, max_bytes, 0
    for uid in uids:
        if max_datasets and taken >= max_datasets:
            notes.append(f"staged {taken} of {len(uids)} datasets; raise max_datasets to widen")
            break
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

        staged_any = False
        for entry in plan["files"]:
            size = entry["bytes"]
            oversized = bool(max_file_bytes) and size > max_file_bytes
            if oversized and not osdf_url_base:
                notes.append(
                    f"{uid}: skipped {entry['filename']}, {size / 1e6:.0f} MB exceeds the "
                    f"{max_file_bytes / 1e6:.0f} MB the access point will transfer and OSDF is off"
                )
                continue
            if oversized:
                object_url = f"{osdf_url_base.rstrip('/')}/{cluster_uuid}/{entry['filename']}"
                try:
                    sent = _stream_to_osdf(entry["url"], object_url, osdf_token_path, size)
                except Exception as e:  # noqa: BLE001 -- one failed upload keeps the rest
                    notes.append(f"{uid}: OSDF upload of {entry['filename']} failed ({e})")
                    continue
                transfer.append(object_url)
                staged_bytes += sent or size
                budget -= size
                staged_any = True
                continue
            # Spool a small product, but never fill the shared pod's filesystem.
            if shutil.disk_usage(dest).free < size + min_free_bytes:
                notes.append(
                    f"{uid}: skipped {entry['filename']}, not enough free pod disk to spool"
                )
                continue
            target = dest / entry["filename"]
            with requests.get(entry["url"], stream=True, timeout=600) as response:
                response.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            transfer.append(str(target))
            written = target.stat().st_size
            staged_bytes += written
            budget -= written
            staged_any = True
        if staged_any:
            taken += 1

    if not transfer and not notes:
        notes.append("Nothing was staged for this request")
    return transfer, staged_bytes, notes
