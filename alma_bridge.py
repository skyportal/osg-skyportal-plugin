"""ALMA cube reduction — runs on an OSG worker in the existing fiesta image.

The job is handed ALMA's *delivered* pipeline products, already staged by the
plugin host (the worker is not assumed to reach the archive). From those this
extracts a spectrum at the source position and a moment-0 map, which is what
makes the observation legible in SkyPortal; it does not re-image from raw.

Only astropy, numpy and matplotlib are needed, which fiestaem already pulls in,
so this reuses `michaelwcoughlin/fiesta` rather than adding an image to the
CVMFS sync list. Re-imaging from the raw ASDM would need CASA and its own
per-cycle image; that is deliberately out of scope here.

Cubes are read a plane at a time rather than whole, since an ALMA cube can be
far larger than the memory an OSG slot gives you.
"""

from __future__ import annotations

import math
import tarfile
from pathlib import Path

# FITS products ALMA delivers that are not the science cube.
_SKIP_SUFFIXES = (".pb.fits", ".mask.fits", ".psf.fits", ".residual.fits", ".wt.fits")

ARCHIVE_ANNOTATION_ORIGIN = "alma-archive"


def source_position(payload: dict) -> tuple[float | None, float | None]:
    """The source's sky position, as SkyPortal now sends it in `inputs.obj`."""
    obj = (payload.get("inputs") or {}).get("obj") or {}
    ra, dec = obj.get("ra"), obj.get("dec")
    if ra is None or dec is None:
        # Fall back to explicit parameters, so the bridge is usable standalone.
        params = (payload.get("inputs") or {}).get("analysis_parameters") or {}
        ra, dec = params.get("ra"), params.get("dec")
    try:
        return float(ra), float(dec)
    except (TypeError, ValueError):
        return None, None


def extract_tarballs(work: Path) -> list[Path]:
    """Unpack every staged tarball; returns the directories they unpacked into."""
    roots = []
    for archive in sorted(work.glob("*.tar")):
        target = work / f"extracted_{archive.stem}"
        target.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            # Refuse members that would escape the extraction directory.
            safe = [
                m
                for m in tar.getmembers()
                if not (m.name.startswith("/") or ".." in Path(m.name).parts)
            ]
            try:
                # "data" refuses absolute paths, links and device files; the
                # member check above still stands for Pythons without it.
                tar.extractall(target, members=safe, filter="data")
            except TypeError:
                tar.extractall(target, members=safe)
        roots.append(target)
    return roots


def find_cubes(roots: list[Path]) -> list[Path]:
    """Science cubes among the delivered products, largest first.

    ALMA ships calibration and diagnostic images alongside the science ones;
    those are named by role and are not what a spectrum should come from.
    """
    cubes = []
    for root in roots:
        for path in root.rglob("*.fits"):
            name = path.name.lower()
            if any(name.endswith(suffix) for suffix in _SKIP_SUFFIXES):
                continue
            cubes.append(path)
    return sorted(cubes, key=lambda p: p.stat().st_size, reverse=True)


def _spectral_axis(header, n_planes):
    """World values along the spectral axis, in Hz where the header says so."""
    for axis in range(1, int(header.get("NAXIS", 0)) + 1):
        ctype = str(header.get(f"CTYPE{axis}", "")).upper()
        if ctype.startswith("FREQ"):
            crval = float(header.get(f"CRVAL{axis}", 0.0))
            cdelt = float(header.get(f"CDELT{axis}", 1.0))
            crpix = float(header.get(f"CRPIX{axis}", 1.0))
            return [crval + (i + 1 - crpix) * cdelt for i in range(n_planes)], "Hz"
    return list(range(n_planes)), "channel"


def _aperture(shape, centre, radius_px):
    """Pixel offsets within `radius_px` of centre, clipped to the plane."""
    ny, nx = shape
    cx, cy = centre
    lo_x, hi_x = max(0, int(cx - radius_px)), min(nx, int(cx + radius_px) + 1)
    lo_y, hi_y = max(0, int(cy - radius_px)), min(ny, int(cy + radius_px) + 1)
    pixels = []
    for y in range(lo_y, hi_y):
        for x in range(lo_x, hi_x):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius_px**2:
                pixels.append((y, x))
    return pixels


def reduce_cube(path: Path, ra: float, dec: float, radius_arcsec: float = 1.0) -> dict:
    """Spectrum at the source position and a moment-0 map, from one cube."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path, memmap=True) as hdul:
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
        if hdu is None:
            raise ValueError(f"{path.name} holds no image data")
        header = hdu.header
        wcs = WCS(header)

        # ALMA cubes are commonly 4D (stokes, freq, dec, ra); drop degenerate axes.
        data = hdu.data
        while data.ndim > 3:
            data = data[0]
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        n_planes = data.shape[0]

        celestial = wcs.celestial
        x, y = celestial.world_to_pixel_values(ra, dec)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("source position does not land on this cube")

        scale = abs(float(celestial.wcs.cdelt[0])) * 3600.0 or 1.0
        radius_px = max(radius_arcsec / scale, 1.0)
        pixels = _aperture(data.shape[1:], (float(x), float(y)), radius_px)
        if not pixels:
            raise ValueError("source position falls outside this cube")

        # A plane at a time: a cube can be much larger than the slot's memory.
        spectrum, moment0 = [], np.zeros(data.shape[1:], dtype="float64")
        for index in range(n_planes):
            plane = np.asarray(data[index], dtype="float64")
            finite = np.where(np.isfinite(plane), plane, 0.0)
            moment0 += finite
            spectrum.append(float(np.mean([finite[j, i] for j, i in pixels])))

    frequencies, unit = _spectral_axis(header, n_planes)
    return {
        "file": path.name,
        "n_planes": n_planes,
        "spectral_unit": unit,
        "frequencies": frequencies,
        "spectrum": spectrum,
        "aperture_arcsec": radius_arcsec,
        "aperture_pixels": len(pixels),
        "bunit": str(header.get("BUNIT", "")).strip() or None,
        "moment0": moment0,
        "position": {"ra": ra, "dec": dec, "x": float(x), "y": float(y)},
    }


def plot_reduction(reduction: dict, work: Path) -> list[str]:
    """A moment-0 image and the extracted spectrum, as PNGs for SkyPortal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths = []
    stem = Path(reduction["file"]).stem

    moment0 = reduction["moment0"]
    finite = moment0[np.isfinite(moment0)]
    fig, ax = plt.subplots(figsize=(5, 5))
    if finite.size:
        ax.imshow(
            moment0,
            origin="lower",
            cmap="inferno",
            vmin=np.percentile(finite, 1),
            vmax=np.percentile(finite, 99),
        )
    ax.plot(reduction["position"]["x"], reduction["position"]["y"], "c+", markersize=12)
    ax.set_title(f"{stem}\nmoment 0")
    fig.tight_layout()
    moment_path = work / f"{stem}_moment0.png"
    fig.savefig(moment_path, dpi=110)
    plt.close(fig)
    paths.append(str(moment_path))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(reduction["frequencies"], reduction["spectrum"], lw=1)
    ax.set_xlabel("Frequency (Hz)" if reduction["spectral_unit"] == "Hz" else "Channel")
    ax.set_ylabel(reduction["bunit"] or "Intensity")
    ax.set_title(f'{stem}\nspectrum in {reduction["aperture_arcsec"]}" aperture')
    fig.tight_layout()
    spectrum_path = work / f"{stem}_spectrum.png"
    fig.savefig(spectrum_path, dpi=110)
    plt.close(fig)
    paths.append(str(spectrum_path))
    return paths


def run_from_skyportal_inputs(payload: dict, resource_id: str = "obj", work_dir: str = ".") -> dict:
    """Reduce whatever products were staged for this job."""
    work = Path(work_dir)
    params = (payload.get("inputs") or {}).get("analysis_parameters") or {}
    try:
        radius_arcsec = float(params.get("aperture_arcsec", 1.0))
    except (TypeError, ValueError):
        radius_arcsec = 1.0
    try:
        max_cubes = int(params.get("max_cubes", 3))
    except (TypeError, ValueError):
        max_cubes = 3

    ra, dec = source_position(payload)
    if ra is None or dec is None:
        return {"status": "failure", "message": "No source position in the request"}

    cubes = find_cubes(extract_tarballs(work))
    if not cubes:
        return {
            "status": "failure",
            "message": "No ALMA science cubes were staged for this job",
        }

    reductions, plot_files, failures = [], [], []
    for cube in cubes[:max_cubes]:
        try:
            reduction = reduce_cube(cube, ra, dec, radius_arcsec=radius_arcsec)
        except Exception as e:  # noqa: BLE001 -- one bad cube must not lose the rest
            failures.append(f"{cube.name}: {e}")
            continue
        plot_files += plot_reduction(reduction, work)
        # The moment map is an array; it went into the plot and is not JSON.
        reduction.pop("moment0", None)
        reductions.append(reduction)

    if not reductions:
        return {
            "status": "failure",
            "message": "; ".join(failures) or "No cube could be reduced",
        }

    message = f"Reduced {len(reductions)} of {len(cubes)} staged cube(s)"
    if failures:
        message += f" ({len(failures)} failed)"
    return {
        "status": "success",
        "message": message,
        "results": {
            "obj_id": resource_id,
            "position": {"ra": ra, "dec": dec},
            "cubes": reductions,
            "skipped": failures,
        },
        "plot_files": plot_files,
        "annotations": {
            "n_cubes": len(reductions),
            "aperture_arcsec": radius_arcsec,
        },
    }
