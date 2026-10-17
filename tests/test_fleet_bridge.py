"""Pure-logic tests for fleet_bridge (no FLEET install): the SkyPortal photometry
-> FLEET light-curve format, the BOOM host -> FLEET catalogue synthesis
(nanomaggy->AB, galaxy filtering), the taxonomy map and the result shaping. The
science is FLEET's own; here we pin the offline data bridge."""

import csv
import io
import json

import fleet_bridge

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
        {"mjd": 60003.0, "filter": "ztfg", "mag": "", "magerr": "", "limiting_mag": 20.6},  # UL
        {
            "mjd": 60005.0,
            "filter": "atlasc",
            "mag": 18.0,
            "magerr": 0.1,
            "limiting_mag": 21.0,
        },  # non-grizi
    ]
    p = {"analysis_parameters": over.pop("analysis_parameters", {})}
    p["photometry"] = over.pop("photometry", _csv(rows, PHOT_COLUMNS))
    p["obj"] = over.pop("obj", {"id": "ZTF1", "ra": 150.0, "dec": 2.5})
    p.update(over)
    return p


def test_lightcurve_rows_keeps_grizi_detections_and_upper_limits():
    rows = fleet_bridge.lightcurve_rows(_payload())
    # g + r detections and the g upper limit kept; the non-ZTF filter dropped.
    assert [r["Filter"] for r in rows] == ["g", "r", "g"]
    assert [r["UL"] for r in rows] == ["False", "False", "True"]
    assert rows[-1]["Raw"] == 20.6  # UL uses the limiting mag


def test_write_lightcurve_schema(tmp_path):
    path = fleet_bridge.write_lightcurve(_payload(), "ZTF1", str(tmp_path))
    assert path == tmp_path / "photometry" / "ZTF1.txt"
    header = path.read_text().splitlines()[0].split()
    assert header == ["MJD", "Raw", "MagErr", "Telescope", "Filter", "Source", "UL", "RA", "DEC"]


def test_write_lightcurve_empty_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        fleet_bridge.write_lightcurve(
            _payload(photometry=_csv([], PHOT_COLUMNS)), "ZTF1", str(tmp_path)
        )


def test_nanomaggy_to_ab():
    # 22.5 - 2.5*log10(1) == 22.5; non-positive/missing flux -> None.
    assert fleet_bridge._nanomaggy_to_ab(1.0) == 22.5
    assert round(fleet_bridge._nanomaggy_to_ab(100.0), 1) == 17.5
    assert fleet_bridge._nanomaggy_to_ab(0) is None
    assert fleet_bridge._nanomaggy_to_ab(None) is None


def _host(**over):
    row = {
        "ra": 150.001,
        "dec": 2.5001,
        "objtype": "SER",
        "shape_r": 2.3,
        "flux_g": 100.0,
        "flux_r": 158.0,
        "flux_i": 200.0,
        "flux_z": 251.0,
    }
    row.update(over)
    return {"LSDR10": [row], "host_galaxy": {"best_host": {"objname": "x", "catalog": "LSDR10"}}}


def test_write_catalog_from_lsdr10(tmp_path):
    (tmp_path / fleet_bridge.HOST_FILE).write_text(json.dumps(_host()))
    path = fleet_bridge.write_catalog("ZTF1", str(tmp_path))
    assert path == tmp_path / "catalogs" / "ZTF1.cat"
    table = list(
        csv.DictReader(io.StringIO(path.read_text().replace(" ", ",")), skipinitialspace=True)
    )
    assert len(table) == 1
    row = table[0]
    assert row["object_nature"] == "1.0" and row["type_sdss"] == "3"
    assert row["modelMag_g_sdss"] == "17.5"  # 22.5 - 2.5*log10(100)
    assert row["petroR50_i_sdss"] == "2.3"  # shape_r -> half-light radius
    assert row["modelMag_u_sdss"] == "17.5"  # u filled from g


def test_write_catalog_excludes_stars(tmp_path):
    (tmp_path / fleet_bridge.HOST_FILE).write_text(json.dumps(_host(objtype="PSF")))
    path = fleet_bridge.write_catalog("ZTF1", str(tmp_path))
    # No usable host -> one float sentinel row (not header-only), so FLEET's
    # catalog_operations (psfMag -= extinction) doesn't crash on an int64 column.
    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_write_catalog_without_host_writes_float_sentinel(tmp_path):
    # No host file at all: still a float sentinel row FLEET will discard via Pcc.
    path = fleet_bridge.write_catalog("ZTF1", str(tmp_path), ra=10.0, dec=-5.0)
    rows = list(
        csv.DictReader(io.StringIO(path.read_text().replace(" ", ",")), skipinitialspace=True)
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["psfMag_g_sdss"] == "99.0" and r["modelMag_r_sdss"] == "99.0"  # float, faint
    assert r["ra_matched"] == "12.0"  # offset +2 deg from the transient -> discarded


def test_info_hostless_reads_fleet_flag():
    class _T:
        def __getitem__(self, k):
            return {"hostless": [True]}[k]

    assert fleet_bridge._info_hostless(_T()) is True
    # missing column -> default True (treat as hostless rather than mislabel)
    assert fleet_bridge._info_hostless(object()) is True


def test_extract_probabilities_uses_p_late_and_rescales_percent():
    # FLEET reports P_late_<class> as percentages; the rapid columns must be ignored.
    class _T:
        colnames = ["name", "P_late_SNIa", "P_late_TDE", "P_late_SLSNI", "P_rapid_slsn_SLSNI"]

        def __getitem__(self, k):
            return {
                "P_late_SNIa": [70.0],
                "P_late_TDE": [20.0],
                "P_late_SLSNI": [10.0],
                "P_rapid_slsn_SLSNI": [95.0],
            }[k]

    probs = fleet_bridge._extract_probabilities(_T())
    assert probs == {"SNIa": 0.7, "TDE": 0.2, "SLSNI": 0.1}


def test_taxonomy_map_covers_fleet_classes():
    assert set(fleet_bridge.FLEET_CLASSES) == set(fleet_bridge.FLEET_TO_TAXONOMY)
    assert fleet_bridge.FLEET_TO_TAXONOMY["SLSNI"] == "Ic-SLSN"
    assert fleet_bridge.FLEET_ORIGIN == "FLEET"


def test_shape_for_skyportal_emits_annotation_and_classification():
    result = {"results": {"predicted": "SLSNI", "probabilities": {"SLSNI": 0.8, "TDE": 0.2}}}
    fleet_bridge._shape_for_skyportal(result)
    ann = result["annotations"][0]
    assert ann["origin"] == "FLEET" and ann["data"]["fleet_p_SLSNI"] == 0.8
    c = result["classifications"][0]
    assert c["classification"] == "Ic-SLSN" and c["ml"] is True and c["probability"] == 0.8


def test_resolve_redshift_and_coords():
    p = _payload(analysis_parameters={"redshift": "0.15"})
    assert fleet_bridge.resolve_redshift(p) == 0.15
    assert fleet_bridge.resolve_coords(p) == (150.0, 2.5)


def test_row_id_decodes_bson_int64():
    # BOOM's raw-BSON int64 arrives as {high, low}; must stringify to match objname.
    assert fleet_bridge._row_id({"_id": {"high": 2560088, "low": 217056842}}) == "10995494451938890"
    assert fleet_bridge._row_id({"_id": {"$numberLong": "123"}}) == "123"
    assert fleet_bridge._row_id({"_id": 42}) == "42"


def test_select_host_uses_dlr_best_host_by_id():
    rows = [
        {
            "_id": 111,
            "ra": 10.0,
            "dec": 5.0,
            "objtype": "SER",
            "flux_g": 100.0,
            "distance_arcsec": 3.0,
        },
        {
            "_id": 222,
            "ra": 10.001,
            "dec": 5.0,
            "objtype": "SER",
            "flux_g": 50.0,
            "distance_arcsec": 0.5,
        },
    ]
    host = {"LSDR10": rows, "host_galaxy": {"best_host": {"objname": "111", "catalog": "LSDR10"}}}
    sel = fleet_bridge._select_host_row(host, rows)
    assert sel["_id"] == 111  # the DLR-chosen id, not merely the nearest by distance


def test_host_rows_uses_best_host_ps1_photometry_outside_legacy():
    best_host = {
        "objname": "WISEA J173303.02-034534.0",
        "catalog": "NED",
        "ra": 263.2626,
        "dec": -3.7594,
        "a_arcsec": 29.15,
        "size_is_isophotal": True,
        "photometry": {"catalog": "PS1_DR2", "mag_g": 18.2, "mag_r": 17.6, "mag_i": None},
    }
    hr = fleet_bridge._host_rows({"LSDR10": [], "host_galaxy": {"best_host": best_host}})
    assert len(hr) == 1
    assert hr[0]["ra_matched"] == 263.2626
    assert hr[0]["modelMag_r_sdss"] == 17.6
    assert hr[0]["modelMag_i_sdss"] == 17.6  # filled from the nearest band, not 99
    assert hr[0]["petroR50_r_sdss"] == 29.15 / 2


def test_host_rows_recovers_psf_nucleus_when_best_host_null():
    # TDE: BOOM dropped the PSF-typed nucleus so best_host is null; take the nearest.
    rows = [
        {
            "_id": 1,
            "ra": 10.0,
            "dec": 5.0,
            "objtype": "PSF",
            "flux_g": 200.0,
            "distance_arcsec": 0.3,
        },
        {
            "_id": 2,
            "ra": 10.01,
            "dec": 5.0,
            "objtype": "SER",
            "flux_g": 80.0,
            "distance_arcsec": 8.0,
        },
    ]
    hr = fleet_bridge._host_rows({"LSDR10": rows, "host_galaxy": {"best_host": None}})
    assert len(hr) == 1 and hr[0]["ra_matched"] == 10.0  # the PSF nucleus, kept


def test_select_host_none_when_only_dup_or_far():
    rows = [
        {
            "_id": 1,
            "ra": 10.0,
            "dec": 5.0,
            "objtype": "DUP",
            "flux_g": 200.0,
            "distance_arcsec": 0.1,
        },
        {
            "_id": 2,
            "ra": 11.0,
            "dec": 5.0,
            "objtype": "SER",
            "flux_g": 80.0,
            "distance_arcsec": 50.0,
        },
    ]
    assert fleet_bridge._select_host_row({"LSDR10": rows, "host_galaxy": {}}, rows) is None


def test_row_to_catalog_fills_missing_band_not_sentinel():
    # flux_i<=0 -> i-band missing. It must be filled from the nearest present band,
    # not written as 99.0, or FLEET's i-band Pcc drops the host (chance coincidence).
    row = {
        "ra": 10.0,
        "dec": 5.0,
        "objtype": "SER",
        "flux_g": 100.0,
        "flux_r": 158.0,
        "flux_i": 0.0,
        "flux_z": 251.0,
    }
    rec = fleet_bridge._row_to_catalog(row)
    assert rec["psfMag_i_sdss"] != 99.0
    assert rec["psfMag_i_sdss"] in (rec["psfMag_r_sdss"], rec["psfMag_z_sdss"])
    assert rec["psfMag_u_sdss"] == rec["psfMag_g_sdss"]  # u still stands in from g
