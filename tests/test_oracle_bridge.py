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


def test_photometry_rows_drops_sub_5sigma():
    # magerr 0.5 -> SNR ~2.2 (a forced-photometry non-detection); 0.05 -> ~22 kept.
    rows = [
        {"mjd": 60001.0, "filter": "ztfg", "mag": 22.5, "magerr": 0.5, "limiting_mag": 20.3},
        {"mjd": 60002.0, "filter": "ztfr", "mag": 18.5, "magerr": 0.05, "limiting_mag": 20.5},
    ]
    out = oracle_bridge.photometry_rows(_payload(photometry=_csv(rows, PHOT_COLUMNS)))
    assert [r[1] for r in out] == ["r"] and out[0][2] == 18.5


def test_no_detections_fails_gracefully():
    out = oracle_bridge.run_from_skyportal_inputs(_payload(photometry=_csv([], PHOT_COLUMNS)))
    assert out["status"] == "failure"


def test_merged_annotations_flattens_alert_fields():
    # SkyPortal exports annotations as a CSV whose `data` cell is a dict string.
    ann = _csv(
        [
            {"data": "{'sgscore1': 0.036, 'distpsnr1': 7.09, 'drb': 0.99}", "origin": "BOOM"},
            {"data": "{'ndethist': 7, 'zogy_scorr': 8.7}", "origin": "grb"},
        ],
        ["data", "modified", "origin", "created_at"],
    )
    merged = oracle_bridge.merged_annotations({"annotations": ann})
    assert merged["sgscore1"] == 0.036 and merged["distpsnr1"] == 7.09 and merged["ndethist"] == 7


def test_feat_uses_alias_and_flag_fallback():
    merged = {"zogy_scorr": 8.7}
    # scorr aliases to zogy_scorr; a missing feature and ZTF's -999 both -> flag.
    assert oracle_bridge._feat(merged, "scorr", -9) == 8.7
    assert oracle_bridge._feat({}, "sgscore1", -9) == -9
    assert oracle_bridge._feat({"drb": -999}, "drb", -9) == -9


def test_taxonomy_map_covers_bts_leaves():
    # The seven BTS_Taxonomy leaves each map to a Sitewide Taxonomy (id 1019) label.
    expected = {"SN-Ia", "SN-II", "SN-Ib/c", "SLSN", "AGN", "CV", "Varstar"}
    assert set(oracle_bridge.ORACLE_TO_TAXONOMY) == expected
    assert oracle_bridge.ORACLE_ORIGIN == "ORACLE"
    assert oracle_bridge.ORACLE_TAXONOMY == "Sitewide Taxonomy"


def test_default_is_omni_pro_with_reference_channels():
    # Default runs the full omni model; the cutout lands in the last band's channel.
    assert oracle_bridge.DEFAULT_MODEL == "BTSv2-pro"
    assert oracle_bridge.BAND_TO_CHANNEL == {"g": 0, "r": 1, "i": 2}
    assert oracle_bridge.CUTOUT_FILE == "oracle_cutout.fits.gz"


def test_load_cutout_missing_file_is_none():
    # Absent cutout degrades to None (a zero postage stamp), never raises.
    assert oracle_bridge._load_cutout("/nonexistent/oracle_cutout.fits.gz") is None


def _rows(g, r, i=0):
    return (
        [(60000.0 + n, "g", 18.0, 0.05) for n in range(g)]
        + [(60000.0 + n, "r", 18.0, 0.05) for n in range(r)]
        + [(60000.0 + n, "i", 18.0, 0.05) for n in range(i)]
    )


def test_sufficient_gate_needs_8_total_and_2_each_gr():
    assert oracle_bridge._sufficient(_rows(4, 4), {}) is True
    assert oracle_bridge._sufficient(_rows(2, 2), {}) is False  # only 4 total
    assert oracle_bridge._sufficient(_rows(8, 1), {}) is False  # <2 in r
    assert oracle_bridge._sufficient(_rows(1, 8), {}) is False  # <2 in g
    # i-band counts toward the total but not the per-band g/r requirement
    assert oracle_bridge._sufficient(_rows(2, 2, 4), {}) is True
    # thresholds are overridable per request
    assert oracle_bridge._sufficient(_rows(2, 2), {"min_detections": 4}) is True
