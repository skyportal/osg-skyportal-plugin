"""Pure-logic tests for ngsf_bridge (no NGSF or template bank needed).

The fit itself runs only in the NGSF image; here we cover the plugin-specific
helpers: the SkyPortal spectra wire format, spectrum selection, redshift
resolution, and wavelength-range precedence.
"""

import csv
import io

import pytest

import ngsf_bridge


def _csv(rows, columns):
    """SkyPortal serializes array columns with ndarray.tolist() then to_csv, so
    wavelengths/fluxes reach the bridge as list reprs inside CSV cells."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


SPECTRA_COLUMNS = ["observed_at", "wavelengths", "fluxes", "origin"]


def _payload(**overrides):
    rows = [
        {
            "observed_at": "2021-07-01T00:00:00",
            "wavelengths": "[4000.0, 4010.0, 4020.0]",
            "fluxes": "[1.0, 2.0, 3.0]",
            "origin": "LRIS",
        },
        {
            "observed_at": "2021-08-06T00:00:00",
            "wavelengths": "[5000.0, 5010.0]",
            "fluxes": "[4.0, 5.0]",
            "origin": "NGPS",
        },
    ]
    payload = {
        "spectra": _csv(rows, SPECTRA_COLUMNS),
        "redshift": _csv([{"redshift": "0.127"}], ["redshift"]),
    }
    payload.update(overrides)
    return payload


def test_as_floats_parses_list_repr_from_csv_cell():
    assert ngsf_bridge._as_floats("[4000.0, 4010.5]") == [4000.0, 4010.5]
    assert ngsf_bridge._as_floats([1.0, 2.0]) == [1.0, 2.0]


def test_as_floats_rejects_unparseable_cell():
    with pytest.raises(ValueError):
        ngsf_bridge._as_floats(3.7)


def test_select_spectrum_defaults_to_most_recent():
    row, index = ngsf_bridge.select_spectrum(_payload())
    assert index == 1
    assert row["origin"] == "NGPS"


def test_select_spectrum_honors_explicit_index():
    payload = _payload(analysis_parameters={"spectrum_index": 0})
    row, index = ngsf_bridge.select_spectrum(payload)
    assert index == 0
    assert row["origin"] == "LRIS"


def test_select_spectrum_by_observed_at_prefix():
    payload = _payload(analysis_parameters={"observed_at": "2021-07-01"})
    assert ngsf_bridge.select_spectrum(payload)[1] == 0


def test_select_spectrum_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        ngsf_bridge.select_spectrum(_payload(analysis_parameters={"spectrum_index": 9}))


def test_select_spectrum_requires_spectra_input():
    with pytest.raises(ValueError, match="input_data_type"):
        ngsf_bridge.select_spectrum({"spectra": None})


def test_resolve_redshift_prefers_explicit_parameter():
    payload = _payload(analysis_parameters={"redshift": 0.25})
    assert ngsf_bridge.resolve_redshift(payload) == 0.25


def test_resolve_redshift_reads_skyportal_value():
    assert ngsf_bridge.resolve_redshift(_payload()) == pytest.approx(0.127)


def test_resolve_redshift_none_when_source_has_no_redshift():
    # An unset redshift reaches the bridge as an empty cell or the string "nan".
    for value in ("", "nan"):
        payload = _payload(redshift=_csv([{"redshift": value}], ["redshift"]))
        assert ngsf_bridge.resolve_redshift(payload) is None


def test_wav_range_precedence():
    row, _ = ngsf_bridge.select_spectrum(_payload())
    # an explicit range wins over the instrument
    payload = _payload(analysis_parameters={"wav_range": [4500, 8000], "instrument": "GHTS"})
    assert ngsf_bridge.wav_range(payload, row) == (4500.0, 8000.0)
    # then the named instrument
    assert ngsf_bridge.wav_range(_payload(analysis_parameters={"instrument": "GHTS"}), row) == (
        4000.0,
        7000.0,
    )
    # then the spectrum's own origin (row 1 is NGPS)
    assert ngsf_bridge.wav_range(_payload(), row) == (5900.0, 10000.0)


def test_wav_range_falls_back_to_default_for_unknown_instrument():
    row, _ = ngsf_bridge.select_spectrum(_payload(analysis_parameters={"spectrum_index": 0}))
    assert ngsf_bridge.wav_range(_payload(), row) == ngsf_bridge.DEFAULT_WAV_RANGE


def test_write_spectrum_ascii_sorts_and_drops_non_finite(tmp_path):
    row = {
        "wavelengths": "[4020.0, 4000.0, 4010.0, 4030.0]",
        "fluxes": [3.0, 1.0, float("nan"), 4.0],
    }
    out = tmp_path / "s.ascii"
    assert ngsf_bridge.write_spectrum_ascii(row, out) == 3
    lam = [float(line.split()[0]) for line in out.read_text().splitlines()]
    assert lam == sorted(lam)
    assert 4010.0 not in lam  # the NaN-flux sample is dropped


def test_write_spectrum_ascii_rejects_mismatched_lengths(tmp_path):
    row = {"wavelengths": "[1.0, 2.0]", "fluxes": "[1.0]"}
    with pytest.raises(ValueError, match="wavelengths but"):
        ngsf_bridge.write_spectrum_ascii(row, tmp_path / "s.ascii")


def test_write_spectrum_ascii_rejects_all_non_finite(tmp_path):
    row = {"wavelengths": [1.0, 2.0], "fluxes": [float("nan"), float("inf")]}
    with pytest.raises(ValueError, match="no finite samples"):
        ngsf_bridge.write_spectrum_ascii(row, tmp_path / "s.ascii")


def test_collect_ranks_by_chi2(tmp_path):
    tree = tmp_path / "NGSF"
    (tree / "fit_results").mkdir(parents=True)
    (tree / "fit_results" / "obj_spec0.csv").write_text(
        "SPECTRUM,GALAXY,SN,Z,Phase,CHI2/dof\n"
        "a,E,Ia-norm/2009Y/WFCCD phase-band : 12.36B,0.10,12.36,9.5\n"
        "a,S0,Ic/1994I/KAST phase-band : -2.57B,0.12,-2.57,4.6\n"
    )
    out = ngsf_bridge._collect(tree, "obj_spec0", free_z=True, n_results=3)
    assert [r["CHI2/dof"] for r in out["rows"]] == [4.6, 9.5]
    assert ngsf_bridge._sn_type(out["best"]) == "Ic"
    assert out["best"]["Z"] == 0.12


def test_collect_raises_when_ngsf_wrote_nothing(tmp_path):
    tree = tmp_path / "NGSF"
    (tree / "fit_results").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no results"):
        ngsf_bridge._collect(tree, "obj_spec0", free_z=True, n_results=3)


def test_sn_type_reads_leading_template_segment():
    assert ngsf_bridge._sn_type({"SN": "Ia-norm/2009Y/WFCCD phase-band : 12.36B"}) == "Ia-norm"
    assert ngsf_bridge._sn_type(None) is None
    assert ngsf_bridge._sn_type({"SN": None}) is None


def test_duplicates_scan_skips_a_redundant_fixed_z_pass():
    # The scan landing on the catalog redshift means the refit already is the
    # catalog-redshift fit; running it again only duplicates plots.
    assert ngsf_bridge._duplicates_scan(0.127, 0.127, 0.001)
    assert ngsf_bridge._duplicates_scan(0.127, 0.1275, 0.001)
    # A genuinely different redshift still earns its own pass.
    assert not ngsf_bridge._duplicates_scan(0.127, 0.100, 0.001)
    # Nothing to compare against.
    assert not ngsf_bridge._duplicates_scan(None, 0.127, 0.001)
    assert not ngsf_bridge._duplicates_scan(0.127, None, 0.001)


def test_read_model_spectrum(tmp_path):
    p = tmp_path / "x_ngsf0_model.txt"
    p.write_text("4000 1.0\n5000 2.0\n# header\n6000 3.0\n")
    assert ngsf_bridge.read_model_spectrum(p) == [
        [4000.0, 1.0],
        [5000.0, 2.0],
        [6000.0, 3.0],
    ]


def test_read_model_spectrum_missing(tmp_path):
    assert ngsf_bridge.read_model_spectrum(tmp_path / "nope.txt") is None


def test_read_model_spectrum_ngsf_format(tmp_path):
    # NGSF's exact output: a comment header, two columns, possible nan rows.
    p = tmp_path / "s_ngsf0_model.txt"
    p.write_text("# wavelength(A) model_flux\n4000 1.05\n4010 nan\n4020 0.97\n")
    assert ngsf_bridge.read_model_spectrum(p) == [[4000.0, 1.05], [4020.0, 0.97]]


def test_read_csv_handles_oversized_spectrum_cell():
    # A high-resolution spectrum's wavelength cell exceeds csv's default 128 KB
    # field cap; the bridge lifts the limit so it parses instead of raising.
    n = 15000
    wl = "[" + ", ".join(f"{4000 + i * 0.5:.4f}" for i in range(n)) + "]"
    fx = "[" + ", ".join("1.0" for _ in range(n)) + "]"
    assert len(wl) > 131072
    payload = _payload(
        spectra=_csv(
            [
                {
                    "observed_at": "2021-01-01T00:00:00",
                    "wavelengths": wl,
                    "fluxes": fx,
                    "origin": "FIRE",
                }
            ],
            SPECTRA_COLUMNS,
        )
    )
    rows = ngsf_bridge._read_csv(payload["spectra"])
    assert len(rows) == 1
    assert len(ngsf_bridge._as_floats(rows[0]["wavelengths"])) == n


def test_overlap_fraction():
    # An ordinary optical spectrum short of the fit range: 4000-7220 over 4000-9500.
    assert ngsf_bridge._overlap_fraction({"wavelengths": [4000.0, 7220.0]}, 4000.0, 9500.0) == (
        pytest.approx((7220 - 4000) / (9500 - 4000))
    )
    assert ngsf_bridge._overlap_fraction({"wavelengths": [3000.0, 11000.0]}, 4000.0, 9500.0) == 1.0
    # Disjoint from the fit range clamps to zero, not a negative fraction.
    assert ngsf_bridge._overlap_fraction({"wavelengths": [5000.0, 5010.0]}, 5900.0, 10000.0) == 0.0
    assert ngsf_bridge._overlap_fraction({"wavelengths": [1.0]}, 5.0, 5.0) is None


def test_chi2_constraint_width_pinned(tmp_path):
    # Only the two grid steps at the minimum sit within 1% of the best chi2. A
    # tight width measures a sharp minimum, not a correct one: NGSF's minima are
    # sharp whether the redshift is right or wrong, so width is not a quality flag.
    p = tmp_path / "obj_chi2_vs_z.csv"
    p.write_text("Z,CHI2/dof2\n0.050,1.00\n0.051,1.005\n0.052,1.50\n0.100,2.0\n")
    assert ngsf_bridge.chi2_constraint_width(p) == pytest.approx(0.001)


def test_chi2_constraint_width_unconstrained(tmp_path):
    # A flat surface: near-min region spans a wide redshift range.
    p = tmp_path / "obj_chi2_vs_z.csv"
    p.write_text("Z,CHI2/dof2\n0.00,1.00\n0.05,1.008\n0.13,1.009\n0.20,2.0\n")
    assert ngsf_bridge.chi2_constraint_width(p) == pytest.approx(0.13)


def test_chi2_constraint_width_absent_or_shapeless(tmp_path):
    assert ngsf_bridge.chi2_constraint_width(tmp_path / "nope.csv") is None
    # A fixed-z run writes a single row: correct, but no shape to measure.
    single = tmp_path / "single_chi2_vs_z.csv"
    single.write_text("Z,CHI2/dof2\n0.086,1.2\n")
    assert ngsf_bridge.chi2_constraint_width(single) is None
    # Nothing cleared minimum_overlap: every chi2 non-finite.
    allinf = tmp_path / "inf_chi2_vs_z.csv"
    allinf.write_text("Z,CHI2/dof2\n0.0,inf\n0.1,inf\n")
    assert ngsf_bridge.chi2_constraint_width(allinf) is None


def _inf_collect(*_args, **_kwargs):
    # What NGSF writes when nothing clears minimum_overlap: a ranked-looking table
    # whose every chi2 is non-finite, arbitrarily ordered.
    best = {
        "SPECTRUM": "s",
        "GALAXY": "Sb",
        "SN": "IIb-flash/2013cu_early/phase-band : 0.0",
        "Z": 0.0,
        "Phase": 0.0,
        "CHI2/dof": float("-inf"),
    }
    return {"rows": [best], "best": best, "plot_files": [], "model_spectrum": [[5000.0, 1.0]]}


def test_non_finite_chi2_is_reported_as_failure(tmp_path, monkeypatch):
    # The default payload's most-recent spectrum (NGPS, 5000-5010 A) does not
    # overlap the NGPS fit range (5900-10000 A), so NGSF cannot fit it.
    monkeypatch.setattr(ngsf_bridge, "_prepare_tree", lambda *a, **k: tmp_path / "NGSF")
    monkeypatch.setattr(ngsf_bridge, "_run_ngsf", lambda *a, **k: None)
    monkeypatch.setattr(ngsf_bridge, "_collect", _inf_collect)

    result = ngsf_bridge.run_from_skyportal_inputs(
        _payload(), resource_id="obj", work_dir=str(tmp_path)
    )

    assert result["status"] == "failure"
    assert "finite" in result["message"]
    assert result["annotations"] == {}
    assert result["model_spectrum"] is None
    assert result["results"]["spectrum"]["wav_overlap_fraction"] == 0.0
