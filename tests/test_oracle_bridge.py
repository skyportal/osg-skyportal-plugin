"""Pure-logic tests for oracle_bridge (no torch/oracle install needed): the
SkyPortal photometry wire format, filter->band mapping, upper limits, and the
leaf->Sitewide taxonomy map (the science runs in the ORACLE image)."""

import csv
import io

import oracle_bridge

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


def test_photometry_rows_keeps_ztf_detections_only_sorted():
    rows = oracle_bridge.photometry_rows(_payload())
    # upper limit + non-ZTF dropped; sorted by mjd; (mjd, band, mag, magerr)
    assert [r[1] for r in rows] == ["g", "r", "r"]
    assert rows[0][0] == 60000.0 and rows[-1][2] == 18.2


def test_photometry_rows_empty_when_no_ztf():
    assert oracle_bridge.photometry_rows(_payload(photometry=_csv([], PHOT_COLUMNS))) == []


def test_no_detections_fails_gracefully():
    out = oracle_bridge.run_from_skyportal_inputs(_payload(photometry=_csv([], PHOT_COLUMNS)))
    assert out["status"] == "failure"


def test_taxonomy_map_covers_bts_leaves():
    # The seven BTS_Taxonomy leaves each map to a Sitewide Taxonomy (id 1019) label.
    expected = {"SN-Ia", "SN-II", "SN-Ib/c", "SLSN", "AGN", "CV", "Varstar"}
    assert set(oracle_bridge.ORACLE_TO_TAXONOMY) == expected
    assert oracle_bridge.ORACLE_ORIGIN == "ORACLE"
    assert oracle_bridge.ORACLE_TAXONOMY == "Sitewide Taxonomy"
