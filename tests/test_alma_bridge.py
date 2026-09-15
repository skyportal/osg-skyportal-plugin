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
        "obj": {"id": "T1", "ra": RA, "dec": DEC},
        "analysis_parameters": params,
    }


def test_position_comes_from_the_obj_block():
    assert alma_bridge.source_position(_payload()) == (RA, DEC)


def test_position_falls_back_to_explicit_parameters():
    """Usable standalone, without SkyPortal's obj block."""
    payload = {"analysis_parameters": {"ra": 10.0, "dec": -20.0}}
    assert alma_bridge.source_position(payload) == (10.0, -20.0)


def test_missing_position_is_reported_not_guessed():
    assert alma_bridge.source_position({}) == (None, None)


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
    payload = {"obj": {"id": "T1", "ra": RA + 40.0, "dec": DEC}}
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


# --- photometry ------------------------------------------------------------

FLUX_JY = 6.9e-3  # a submm afterglow's flux density, in Jy
NOISE_JY = 2.3e-4
BEAM_DEG = 3.0 / 3600.0  # 3" synthesised beam
PIX_DEG = 0.2 / 3600.0


def _continuum_map(tmp_path, flux_jy, seed=1):
    """A single-plane map: a beam-shaped source on noise, as submm continuum is."""
    import math

    ny = nx = 128
    cy = cx = 64
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma_px = (BEAM_DEG / PIX_DEG) / (2 * math.sqrt(2 * math.log(2)))
    image = flux_jy * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_px**2)))
    image = image + np.random.default_rng(seed).normal(0, NOISE_JY, image.shape)

    header = fits.Header()
    header["NAXIS"] = 2
    header["CTYPE1"], header["CRVAL1"] = "RA---SIN", RA
    header["CDELT1"], header["CRPIX1"] = -PIX_DEG, cx + 1
    header["CTYPE2"], header["CRVAL2"] = "DEC--SIN", DEC
    header["CDELT2"], header["CRPIX2"] = PIX_DEG, cy + 1
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = header["BMIN"] = BEAM_DEG
    header["BPA"] = 0.0
    path = tmp_path / "cont.image.fits"
    fits.writeto(path, image.astype("float32"), header, overwrite=True)
    return path


def test_beam_is_read_from_the_header(tmp_path):
    result = alma_bridge.reduce_cube(_continuum_map(tmp_path, FLUX_JY), RA, DEC, radius_arcsec=3.0)
    assert result["beam_arcsec"][0] == pytest.approx(3.0, abs=0.01)
    # pi/(4 ln2) * BMAJ * BMIN / pixel area
    assert result["beam_pixels"] == pytest.approx(254.9, rel=0.01)


def test_peak_recovers_the_flux_density(tmp_path):
    """For a point source the peak in Jy/beam is the flux density."""
    result = alma_bridge.reduce_cube(_continuum_map(tmp_path, FLUX_JY), RA, DEC, radius_arcsec=3.0)
    photometry = result["photometry"]
    assert photometry["detected"] is True
    # The peak pixel is biased high by noise, so allow a few sigma.
    assert photometry["peak_per_beam"] == pytest.approx(FLUX_JY, abs=5 * NOISE_JY)


def test_rms_is_measured_off_source(tmp_path):
    """A bright source must not inflate the noise estimate."""
    result = alma_bridge.reduce_cube(_continuum_map(tmp_path, FLUX_JY), RA, DEC, radius_arcsec=3.0)
    assert result["photometry"]["rms_per_beam"] == pytest.approx(NOISE_JY, rel=0.1)


def test_integrated_flux_is_the_flux_inside_the_aperture(tmp_path):
    """Not a total flux: a 3" aperture on a 3" beam encloses ~94% of a Gaussian."""
    import math

    result = alma_bridge.reduce_cube(_continuum_map(tmp_path, FLUX_JY), RA, DEC, radius_arcsec=3.0)
    sigma_arcsec = 3.0 / (2 * math.sqrt(2 * math.log(2)))
    enclosed = 1 - math.exp(-(3.0**2) / (2 * sigma_arcsec**2))
    assert result["photometry"]["integrated_flux"] == pytest.approx(FLUX_JY * enclosed, rel=0.05)


def test_a_non_detection_yields_an_upper_limit(tmp_path):
    """The usual submm follow-up result: a limit, not a flux."""
    result = alma_bridge.reduce_cube(_continuum_map(tmp_path, 0.0), RA, DEC, radius_arcsec=3.0)
    photometry = result["photometry"]
    assert photometry["detected"] is False
    assert photometry["peak_per_beam"] is None
    assert photometry["integrated_flux"] is None
    assert photometry["upper_limit_per_beam"] == pytest.approx(
        3 * photometry["rms_per_beam"], rel=1e-6
    )


def test_a_map_without_a_beam_reports_no_flux(tmp_path):
    """Jy/beam cannot become Jy without the beam, so it is not guessed."""
    path = _continuum_map(tmp_path, FLUX_JY)
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["BMAJ"]
        del hdul[0].header["BMIN"]
    result = alma_bridge.reduce_cube(path, RA, DEC, radius_arcsec=3.0)
    assert result["beam_pixels"] is None
    assert result["beam_arcsec"] is None
    assert result["photometry"]["integrated_flux"] is None
    # The peak is still a flux density, so it is still reported.
    assert result["photometry"]["peak_per_beam"] is not None


def test_robust_rms_ignores_outliers():
    """Bright pixels in the noise region must not inflate the estimate."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1.0, 4000)
    assert alma_bridge.robust_rms(noise) == pytest.approx(1.0, rel=0.1)

    # A handful of very bright pixels -- a neighbouring source, or an artefact.
    contaminated = np.concatenate([noise, np.full(40, 500.0)])
    assert alma_bridge.robust_rms(contaminated) == pytest.approx(1.0, rel=0.15)
    # Without clipping the same data reads an order of magnitude noisier.
    assert np.std(contaminated) > 10


def test_robust_rms_handles_empty_and_nan_input():
    assert alma_bridge.robust_rms([]) is None
    assert alma_bridge.robust_rms([float("nan")] * 5) is None
