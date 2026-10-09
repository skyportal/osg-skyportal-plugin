"""The fiesta bridge's photometry -> data-file conversion, focused on the
SNR >= 5 detection cut that keeps sub-threshold forced photometry (which still
carries a mag) out of the detection set."""

from pathlib import Path

import fiesta_bridge

HEADER = "mjd,filter,mag,magerr,flux,fluxerr,limiting_mag"


def _payload(tmp_path, rows):
    p = tmp_path / "phot.csv"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return {"photometry": str(p)}


def test_subthreshold_forced_photometry_is_an_upper_limit_not_a_detection(tmp_path):
    payload = _payload(
        tmp_path,
        [
            "100.0,ztfg,19.0,0.05,1000,50,20.5",  # SNR 20 -> detection
            "100.0,ztfg,21.0,0.5,100,50,21.5",  # SNR 2  -> upper limit, same epoch
        ],
    )
    path, _min_mjd, _filters, n_det = fiesta_bridge._write_data_file(payload, tmp_path)
    assert n_det == 1
    assert fiesta_bridge.count_detections(payload) == 1
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    assert sum(1 for ln in lines if not ln.endswith("inf")) == 1  # one real point
    assert any(ln.endswith("inf") for ln in lines)  # the sub-threshold one, censored


def test_snr_falls_back_to_magerr_when_flux_columns_absent(tmp_path):
    # Mag-only photometry (e.g. external spectrograph phot): SNR ~= 1.0857/magerr.
    payload = _payload(
        tmp_path,
        [
            "100.0,ztfr,19.0,0.05,,,20.5",  # SNR ~22 -> detection
            "101.0,ztfr,21.0,0.5,,,21.5",  # SNR ~2  -> not a detection
        ],
    )
    _path, _min_mjd, _filters, n_det = fiesta_bridge._write_data_file(payload, tmp_path)
    assert n_det == 1


def test_bright_forced_photometry_still_counts(tmp_path):
    # A high-SNR forced-photometry point is a real detection and must be kept.
    payload = _payload(tmp_path, ["100.0,ztfg,19.0,0.05,1000,20,20.5"])  # SNR 50
    assert fiesta_bridge.count_detections(payload) == 1
