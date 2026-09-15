"""Tests for alma_staging: which datasets to fetch, and which files within them.

The download itself needs the archive, so these cover the decisions around it —
datalink rows are shaped exactly as ALMA returns them.
"""

import alma_staging

# One dataset's datalink rows, sizes as the archive reports them.
DATALINK = [
    {
        "access_url": "https://almascience.org/dl/member.uid___A001_X35f5_X589.README.txt",
        "semantics": "#documentation",
        "content_length": 1024,
    },
    {
        "access_url": "https://almascience.org/dl/2022.1.01487.S_uid___A001_X35f5_X589_001_of_001.tar",
        "semantics": "#this",
        "content_length": 22_000_000,
    },
    {"access_url": "", "semantics": "#this", "content_length": 0},
    {
        "access_url": "https://almascience.org/dl/2022.1.01487.S_uid___A001_X35f5_X589_auxiliary.tar",
        "semantics": "#auxiliary",
        "content_length": 59_000_000,
    },
    {
        "access_url": "https://almascience.org/dl/2022.1.01487.S_uid___A002_X1064920_X1dce.asdm.sdm.tar",
        "semantics": "#progenitor",
        "content_length": 496_000_000,
    },
]


def test_products_are_split_by_role_and_blanks_dropped():
    classified = alma_staging.classify_products(DATALINK)
    assert [f["bytes"] for f in classified["products"]] == [22_000_000]
    assert [f["bytes"] for f in classified["auxiliary"]] == [59_000_000]
    assert [f["bytes"] for f in classified["progenitor"]] == [496_000_000]
    # The archive returns placeholder rows with no URL; they are not fetchable.
    assert all(f["url"] for role in classified.values() for f in role)


def test_staging_defaults_to_the_delivered_products_alone():
    """The raw progenitor is ~20x the products and only needed to re-image."""
    plan = alma_staging.staging_plan(DATALINK)
    assert [f["bytes"] for f in plan["files"]] == [22_000_000]
    # What was on offer is still reported, so a caller can choose to widen.
    assert plan["available"]["progenitor"] == 496_000_000


def test_staging_can_widen_to_auxiliary_and_raw():
    with_aux = alma_staging.staging_plan(DATALINK, include_auxiliary=True)
    assert with_aux["total_bytes"] == 81_000_000
    everything = alma_staging.staging_plan(
        DATALINK, include_auxiliary=True, include_progenitor=True
    )
    assert everything["total_bytes"] == 577_000_000


def test_unparseable_sizes_do_not_break_the_plan():
    plan = alma_staging.staging_plan(
        [{"access_url": "https://x/a.tar", "semantics": "#this", "content_length": None}]
    )
    assert plan["total_bytes"] == 0 and len(plan["files"]) == 1


def test_explicit_uids_win_over_the_annotation():
    inputs = {
        "analysis_parameters": {"dataset_uids": "uid://A/1, uid://A/2"},
        "annotations": [{"origin": "alma-archive", "data": {"datasets": ["uid://OTHER"]}}],
    }
    assert alma_staging.dataset_uids(inputs) == ["uid://A/1", "uid://A/2"]


def test_uids_come_from_the_coverage_annotation_by_default():
    """The usual case: the archive service already found them, so no parameters."""
    inputs = {
        "annotations": [
            {"origin": "some-other-service", "data": {"datasets": ["uid://WRONG"]}},
            {"origin": "alma-archive", "data": {"datasets": ["uid://A/1", "uid://A/2"]}},
        ]
    }
    assert alma_staging.dataset_uids(inputs) == ["uid://A/1", "uid://A/2"]


def test_annotations_arriving_as_csv_are_parsed():
    """SkyPortal serializes annotations to CSV, with `data` as a JSON string."""
    csv_text = (
        'origin,data,created_at\r\nalma-archive,"{""datasets"": [""uid://A/1""]}",2026-09-11\r\n'
    )
    assert alma_staging.dataset_uids({"annotations": csv_text}) == ["uid://A/1"]


def test_no_datasets_anywhere_is_reported_not_guessed(tmp_path):
    staged, notes = alma_staging.stage({}, tmp_path)
    assert staged == []
    assert "No ALMA datasets" in notes[0]


def test_a_dataset_over_budget_is_skipped_with_a_note(tmp_path, monkeypatch):
    monkeypatch.setattr(alma_staging, "datalink_rows", lambda uid: DATALINK)
    staged, notes = alma_staging.stage(
        {"analysis_parameters": {"dataset_uids": ["uid://A/1"]}},
        tmp_path,
        max_bytes=1_000_000,
    )
    assert staged == []
    assert "exceeds" in notes[0]


def test_a_dataset_the_archive_cannot_serve_does_not_lose_the_others(tmp_path, monkeypatch):
    def rows(uid):
        if uid == "uid://BAD":
            raise RuntimeError("datalink unavailable")
        return []

    monkeypatch.setattr(alma_staging, "datalink_rows", rows)
    # max_datasets is what this test is not about, so let both through.
    staged, notes = alma_staging.stage(
        {"analysis_parameters": {"dataset_uids": ["uid://BAD", "uid://OK"]}},
        tmp_path,
        max_datasets=None,
    )
    assert staged == []
    assert any("datalink unavailable" in n for n in notes)
    assert any("no delivered products" in n for n in notes)


def test_only_max_datasets_are_fetched(tmp_path, monkeypatch):
    """A source can carry dozens of uids; fetching all that fit pulls GBs."""
    fetched = []

    def rows(uid):
        fetched.append(uid)
        return []  # no products, so nothing downloads

    monkeypatch.setattr(alma_staging, "datalink_rows", rows)
    _, notes = alma_staging.stage(
        {"analysis_parameters": {"dataset_uids": [f"uid://A/{i}" for i in range(9)]}},
        tmp_path,
        max_datasets=2,
    )
    assert fetched == ["uid://A/0", "uid://A/1"]
    assert any("2 of 9 datasets" in n for n in notes)


def test_max_datasets_none_means_no_count_limit(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(alma_staging, "datalink_rows", lambda uid: fetched.append(uid) or [])
    alma_staging.stage(
        {"analysis_parameters": {"dataset_uids": ["uid://A/1", "uid://A/2"]}},
        tmp_path,
        max_datasets=None,
    )
    assert len(fetched) == 2


def test_the_byte_budget_still_backs_the_count_limit(tmp_path, monkeypatch):
    """Delivered products vary by an order of magnitude, so both bounds apply."""
    monkeypatch.setattr(alma_staging, "datalink_rows", lambda uid: DATALINK)
    staged, notes = alma_staging.stage(
        {"analysis_parameters": {"dataset_uids": ["uid://A/1"]}},
        tmp_path,
        max_bytes=1_000_000,
        max_datasets=5,
    )
    assert staged == []
    assert any("exceeds" in n for n in notes)
