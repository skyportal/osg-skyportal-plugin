"""Unit tests for the OSDF/Pelican HTTPS helper."""

from unittest.mock import MagicMock, patch

import pytest

import osdf


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://origin.osg/foo/bar", True),
        ("http://origin.osg/foo/bar", True),
        ("osdf:///pelican/path", True),
        ("pelican://origin/path", True),
        ("/local/path", False),
        ("", False),
    ],
)
def test_is_osdf_url(url, expected):
    assert osdf.is_osdf_url(url) is expected


def test_upload_uses_bearer_when_token_env_set(tmp_path, monkeypatch):
    local = tmp_path / "input.txt"
    local.write_bytes(b"hello")
    tok = tmp_path / "tok"
    tok.write_text("abc.def.ghi\n")
    monkeypatch.setenv("BEARER_TOKEN_FILE", str(tok))
    with patch("osdf.requests.put") as mput:
        mput.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        osdf.upload(local, "https://origin/foo")
        assert mput.called
        _, kwargs = mput.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer abc.def.ghi"


def test_upload_no_token_no_header(tmp_path, monkeypatch):
    local = tmp_path / "input.txt"
    local.write_bytes(b"hello")
    monkeypatch.delenv("BEARER_TOKEN_FILE", raising=False)
    with patch("osdf.requests.put") as mput:
        mput.return_value = MagicMock(raise_for_status=lambda: None)
        osdf.upload(local, "https://origin/foo")
        _, kwargs = mput.call_args
        assert "Authorization" not in kwargs["headers"]


def test_upload_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        osdf.upload(tmp_path / "does-not-exist", "https://origin/foo")


def test_download_streams_to_disk(tmp_path):
    target = tmp_path / "subdir" / "out.bin"
    with patch("osdf.requests.get") as mget:
        resp = MagicMock()
        resp.iter_content = lambda chunk_size: [b"abc", b"defg"]
        resp.raise_for_status = lambda: None
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: None
        mget.return_value = resp
        out = osdf.download("https://origin/foo", target)
    assert out == target
    assert target.read_bytes() == b"abcdefg"


def test_download_prefers_explicit_token_path(tmp_path, monkeypatch):
    tok = tmp_path / "tok"
    tok.write_text("from-arg\n")
    monkeypatch.setenv("BEARER_TOKEN_FILE", "/nonexistent")
    with patch("osdf.requests.get") as mget:
        resp = MagicMock()
        resp.iter_content = lambda chunk_size: []
        resp.raise_for_status = lambda: None
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: None
        mget.return_value = resp
        osdf.download("https://origin/foo", tmp_path / "out.bin", token_path=str(tok))
        _, kwargs = mget.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer from-arg"
