"""Pure-logic tests for snid_bridge (no SNID-SAGE or template bank needed).

The classification itself runs only in the SNID-SAGE image; here we cover the
plugin-specific helpers: the SkyPortal spectra wire format, spectrum selection,
redshift resolution, and parsing of `sage identify` output.
"""

import csv
import io

import pytest

import snid_bridge

# A real `sage identify` summary line + its .output template table (SN 2024ggi).
SUMMARY_LINE = (
    "tns_2024ggi: II II-flash z=0.002511±0.000103 age=-8.0±0.5 "
    "Q_cluster=23.2 MatchQual=High TypeConf=High SubtypeConf=High"
)
OUTPUT_TABLE = """\
================================
SNID-SAGE CLASSIFICATION RESULTS
================================

TEMPLATE MATCHES (from Best Cluster - Auto-Selected):
  # Template         Type   Subtype   HsLAP-CCC    Redshift      +Error    Age
  1 sn2024ggiEarly   II     II-flash      35.28    0.002511    0.000264   -8.0
  2 sn2024ggiEarly   II     II-flash      21.60    0.002428    0.000367   -8.0
  3 sn2023ixfEarly   II     II-flash      17.65    0.002350    0.000458  -14.0
"""

SPECTRA_COLUMNS = ["observed_at", "wavelengths", "fluxes", "origin"]


def _csv(rows, columns):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def _payload(**overrides):
    rows = [
        {
            "observed_at": "2024-05-01T00:00:00",
            "wavelengths": "[4000.0, 5000.0, 6000.0]",
            "fluxes": "[1.0, 2.0, 1.5]",
            "origin": "",
        }
    ]
    p = {"analysis_parameters": overrides.pop("analysis_parameters", {})}
    p["spectra"] = overrides.pop("spectra", _csv(rows, SPECTRA_COLUMNS))
    p.update(overrides)
    return p


def test_parse_summary_full_line():
    s = snid_bridge.parse_summary(f"some log\n{SUMMARY_LINE}\nmore log")
    assert s["type"] == "II"
    assert s["subtype"] == "II-flash"
    assert s["redshift"] == pytest.approx(0.002511)
    assert s["redshift_error"] == pytest.approx(0.000103)
    assert s["age"] == pytest.approx(-8.0)
    assert s["q_cluster"] == pytest.approx(23.2)
    assert s["match_quality"] == "High"
    assert s["type_confidence"] == "High"
    assert s["subtype_confidence"] == "High"


def test_parse_summary_multiword_quality():
    line = (
        "sn: Ia Ia-norm z=0.05±0.01 age=3.0±3.0 Q_cluster=1.9 "
        "MatchQual=Very Low TypeConf=No Comp SubtypeConf=Very Low"
    )
    s = snid_bridge.parse_summary(line)
    assert s["match_quality"] == "Very Low"
    assert s["type_confidence"] == "No Comp"
    assert s["subtype_confidence"] == "Very Low"


def test_parse_summary_no_match():
    assert snid_bridge.parse_summary("no classification line here") == {}


def test_parse_template_matches(tmp_path):
    out = tmp_path / "x.output"
    out.write_text(OUTPUT_TABLE)
    matches = snid_bridge.parse_template_matches(out, n_results=2)
    assert len(matches) == 2
    assert matches[0] == {
        "rank": 1,
        "template": "sn2024ggiEarly",
        "type": "II",
        "subtype": "II-flash",
        "score": pytest.approx(35.28),
        "redshift": pytest.approx(0.002511),
        "redshift_error": pytest.approx(0.000264),
        "age": pytest.approx(-8.0),
    }


def test_select_spectrum_defaults_to_most_recent():
    rows = [
        {"observed_at": "2024-01-01T00:00:00", "wavelengths": "[1]", "fluxes": "[1]"},
        {"observed_at": "2024-06-01T00:00:00", "wavelengths": "[2]", "fluxes": "[2]"},
    ]
    payload = {"spectra": _csv(rows, SPECTRA_COLUMNS)}
    row, index = snid_bridge.select_spectrum(payload)
    assert index == 1


def test_select_spectrum_by_index():
    rows = [
        {"observed_at": "2024-01-01T00:00:00", "wavelengths": "[1]", "fluxes": "[1]"},
        {"observed_at": "2024-06-01T00:00:00", "wavelengths": "[2]", "fluxes": "[2]"},
    ]
    payload = _payload(
        spectra=_csv(rows, SPECTRA_COLUMNS), analysis_parameters={"spectrum_index": 0}
    )
    _, index = snid_bridge.select_spectrum(payload)
    assert index == 0


def test_write_spectrum_ascii_filters_nonfinite(tmp_path):
    row = {"wavelengths": "[6000.0, 5000.0, 4000.0]", "fluxes": "[1.0, nan, 2.0]"}
    path = tmp_path / "s.dat"
    n = snid_bridge.write_spectrum_ascii(row, path)
    assert n == 2  # the NaN-flux sample is dropped
    first = path.read_text().splitlines()[0]
    assert first.startswith("4000")  # sorted ascending in wavelength


def test_resolve_redshift_override_wins():
    payload = _payload(analysis_parameters={"redshift": 0.1})
    payload["redshift"] = _csv([{"redshift": "0.2"}], ["redshift"])
    assert snid_bridge.resolve_redshift(payload) == pytest.approx(0.1)


def test_resolve_redshift_from_source():
    payload = _payload()
    payload["redshift"] = _csv([{"redshift": "0.2"}], ["redshift"])
    assert snid_bridge.resolve_redshift(payload) == pytest.approx(0.2)


def test_resolve_redshift_absent():
    assert snid_bridge.resolve_redshift(_payload()) is None


def test_read_model_spectrum(tmp_path):
    (tmp_path / "x_template_01_flux.dat").write_text("4000 1.0\n5000 2.0\nheader line\n6000 3.0\n")
    ms = snid_bridge.read_model_spectrum(tmp_path, "x")
    assert ms == [[4000.0, 1.0], [5000.0, 2.0], [6000.0, 3.0]]


def test_read_model_spectrum_missing(tmp_path):
    assert snid_bridge.read_model_spectrum(tmp_path, "nope") is None


def test_run_end_to_end_stubbed(tmp_path, monkeypatch):
    # Stub the sage subprocess: emit the summary line + write the .output table
    # and a best-template flux file (for the overlay).
    def fake_run(spectrum, outdir, z, timeout):
        stem = spectrum.stem
        (outdir / f"{stem}.output").write_text(OUTPUT_TABLE)
        (outdir / f"{stem}_template_01_flux.dat").write_text("4000 1.0\n5000 2.0\n")
        return SUMMARY_LINE

    monkeypatch.setattr(snid_bridge, "_run_sage", fake_run)
    result = snid_bridge.run_from_skyportal_inputs(
        _payload(), resource_id="AT2024x", work_dir=str(tmp_path)
    )
    assert result["status"] == "success"
    assert result["annotations"]["snid_classification"] == "II"
    assert result["annotations"]["snid_subtype"] == "II-flash"
    assert result["results"]["classification"]["match_quality"] == "High"
    assert len(result["results"]["template_matches"]) == 3
    assert result["model_spectrum"] == [[4000.0, 1.0], [5000.0, 2.0]]
    summary = result["model_spectrum_summary"]
    assert "II II-flash" in summary and "MatchQual High" in summary
    assert "score 35.3" in summary  # best template HσLAP-CCC
    assert "II" in result["message"]
