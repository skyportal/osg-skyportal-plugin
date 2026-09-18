"""Pure-logic tests for flare_bridge (no FLARE install needed): the SkyPortal
photometry wire format, filter mapping, upper limits, redshift resolution,
the parameter passthrough (the science is tested in flare's own suite)."""

import csv
import io

import pytest

import flare_bridge

PHOT_COLUMNS = ["mjd", "filter", "mag", "magerr", "limiting_mag"]


def _csv(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue()


def _payload(**over):
    rows = [
        {"mjd": 60000.0, "filter": "ztfg", "mag": 18.5, "magerr": 0.05, "limiting_mag": 20.5},
        {"mjd": 60000.1, "filter": "ztfr", "mag": 18.7, "magerr": 0.06, "limiting_mag": 20.4},
        {
            "mjd": 60003.0,
            "filter": "ztfg",
            "mag": "",
            "magerr": "",
            "limiting_mag": 20.6,
        },  # upper limit
        {
            "mjd": 60005.0,
            "filter": "sdssu",
            "mag": 18.0,
            "magerr": 0.1,
            "limiting_mag": 21.0,
        },  # non-ZTF
        {"mjd": 60008.0, "filter": "ztfr", "mag": 18.2, "magerr": 0.04, "limiting_mag": 20.3},
    ]
    p = {"analysis_parameters": over.pop("analysis_parameters", {})}
    p["photometry"] = over.pop("photometry", _csv(rows, PHOT_COLUMNS))
    p.update(over)
    return p


def test_photometry_rows_keeps_ztf_detections_only():
    rows = flare_bridge.photometry_rows(_payload())
    assert [r[1] for r in rows] == [1, 2, 2] and rows[0][0] == 60000.0 and rows[-1][2] == 18.2


def test_photometry_rows_empty_raises():
    with pytest.raises(ValueError):
        flare_bridge.photometry_rows(_payload(photometry=_csv([], PHOT_COLUMNS)))


def test_redshift_param_wins_over_skyportal_value():
    p = _payload(redshift=_csv([{"redshift": 0.05}], ["redshift"]))
    assert flare_bridge.resolve_redshift(p) == 0.05
    p = _payload(
        redshift=_csv([{"redshift": 0.05}], ["redshift"]), analysis_parameters={"redshift": "0.12"}
    )
    assert flare_bridge.resolve_redshift(p) == 0.12
    assert flare_bridge.resolve_redshift(_payload()) is None


def test_params_passthrough():
    assert flare_bridge._params(_payload()) == {}
    assert (
        flare_bridge._params(_payload(analysis_parameters={"horizon_days": 20}))["horizon_days"]
        == 20
    )
