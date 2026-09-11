"""Tests for alma_bridge, covering the reduction against a synthetic cube.

The cube is shaped the way ALMA delivers them -- 4D (stokes, freq, dec, ra),
with calibration products alongside the science one -- so the parts that trip
on real data are exercised without needing an archive download.
"""

import tarfile

import numpy as np
import pytest
from astropy.io import fits

import alma_bridge

RA, DEC = 201.365063, -43.019113
LINE_CHANNELS = range(4, 8)


def _cube_header(nx, ny):
    header = fits.Header()
    header["NAXIS"] = 4
    header["CTYPE1"], header["CRVAL1"] = "RA---SIN", RA
    header["CDELT1"], header["CRPIX1"] = -0.0002, nx // 2 + 1
    header["CTYPE2"], header["CRVAL2"] = "DEC--SIN", DEC
    header["CDELT2"], header["CRPIX2"] = 0.0002, ny // 2 + 1
    header["CTYPE3"], header["CRVAL3"] = "FREQ", 2.3e11
    header["CDELT3"], header["CRPIX3"] = 1e7, 1.0
    header["CTYPE4"], header["CRVAL4"] = "STOKES", 1.0
    header["CDELT4"], header["CRPIX4"] = 1.0, 1.0
    header["BUNIT"] = "Jy/beam"
    return header


@pytest.fixture()
def staged(tmp_path):
    """A tarball of delivered products: one science cube and one calibration image."""
    nch, ny, nx = 12, 40, 40
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, (1, nch, ny, nx)).astype("float32")
    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = np.exp(-(((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2) / 8.0))
    for channel in range(nch):
        data[0, channel] += (1.0 if channel in LINE_CHANNELS else 0.05) * blob

    names = ["sci.spw25.cube.I.image.fits", "sci.spw25.cube.I.pb.fits"]
    for name in names:
        fits.writeto(tmp_path / name, data, _cube_header(nx, ny), overwrite=True)
    with tarfile.open(tmp_path / "products.tar", "w") as tar:
        for name in names:
            tar.add(tmp_path / name, arcname=name)
    for name in names:
        (tmp_path / name).unlink()
    return tmp_path


def _payload(**params):
    return {
        "inputs": {
            "obj": {"id": "T1", "ra": RA, "dec": DEC},
            "analysis_parameters": params,
        }
    }


def test_position_comes_from_the_obj_block():
    assert alma_bridge.source_position(_payload()) == (RA, DEC)


def test_position_falls_back_to_explicit_parameters():
    """Usable standalone, without SkyPortal's obj block."""
    payload = {"inputs": {"analysis_parameters": {"ra": 10.0, "dec": -20.0}}}
    assert alma_bridge.source_position(payload) == (10.0, -20.0)


def test_missing_position_is_reported_not_guessed():
    assert alma_bridge.source_position({"inputs": {}}) == (None, None)


def test_calibration_products_are_not_mistaken_for_science(staged):
    cubes = alma_bridge.find_cubes(alma_bridge.extract_tarballs(staged))
    assert [c.name for c in cubes] == ["sci.spw25.cube.I.image.fits"]


def test_reduction_recovers_the_line(staged):
    result = alma_bridge.run_from_skyportal_inputs(
        _payload(aperture_arcsec=1.0), resource_id="T1", work_dir=str(staged)
    )
    assert result["status"] == "success", result["message"]

    cube = result["results"]["cubes"][0]
    assert cube["n_planes"] == 12
    assert cube["bunit"] == "Jy/beam"
    # The frequency axis is read from the header, not invented.
    assert cube["spectral_unit"] == "Hz"
    assert cube["frequencies"][0] == pytest.approx(2.3e11)

    spectrum = cube["spectrum"]
    on_line = np.mean([spectrum[i] for i in LINE_CHANNELS])
    off_line = np.mean([s for i, s in enumerate(spectrum) if i not in LINE_CHANNELS])
    assert on_line > 10 * off_line

    assert len(result["plot_files"]) == 2
    # The moment map is an array and must not be left in the JSON results.
    assert "moment0" not in cube


def test_a_position_off_the_cube_fails_cleanly(staged):
    payload = {"inputs": {"obj": {"id": "T1", "ra": RA + 40.0, "dec": DEC}}}
    result = alma_bridge.run_from_skyportal_inputs(payload, work_dir=str(staged))
    assert result["status"] == "failure"
    assert "cube" in result["message"].lower()


def test_no_staged_products_is_a_clean_failure(tmp_path):
    result = alma_bridge.run_from_skyportal_inputs(_payload(), work_dir=str(tmp_path))
    assert result["status"] == "failure"
    assert "no alma science cubes" in result["message"].lower()


def test_tar_members_escaping_the_directory_are_refused(tmp_path):
    """A staged tarball is remote data; it must not write outside the sandbox."""
    victim = tmp_path / "escape.fits"
    victim.write_bytes(b"original")
    inner = tmp_path / "inner.fits"
    inner.write_bytes(b"replacement")
    with tarfile.open(tmp_path / "evil.tar", "w") as tar:
        tar.add(inner, arcname="../escape.fits")
    inner.unlink()

    alma_bridge.extract_tarballs(tmp_path)
    assert victim.read_bytes() == b"original"
