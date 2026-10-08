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


@pytest.fixture(autouse=True)
def _clear_caches():
    osdf._write_origin_cache.clear()
    osdf._token_cache.clear()
    yield
    osdf._write_origin_cache.clear()
    osdf._token_cache.clear()


def _fake_jwt(exp: int) -> str:
    import base64
    import json

    def seg(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'RS256'})}.{seg({'exp': exp, 'scope': 'storage.create:/'})}.sig"


ORIGIN = "https://kennesaw-origin.nrp.org:50092"


def _resp(status, headers=None, raise_exc=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.raise_for_status = (lambda: (_ for _ in ()).throw(raise_exc)) if raise_exc else (lambda: None)
    return r


def _put_side(drained=None):
    """requests.put stand-in: a .probe URL resolves (307 -> origin), any other URL
    is the real PUT and its streamed body is drained so the counter advances."""

    def side(url, data=None, headers=None, allow_redirects=True, timeout=None):
        if url.endswith("/.probe"):
            return _resp(307, {"Location": f"{ORIGIN}/umn-coughlin/.probe"})
        if hasattr(data, "read"):
            while data.read(1 << 16):
                pass
        if drained is not None:
            drained.append(url)
        return _resp(200)

    return side


def test_upload_resolves_then_puts_to_origin_with_bearer(tmp_path, monkeypatch):
    local = tmp_path / "input.txt"
    local.write_bytes(b"hello")
    tok = tmp_path / "tok"
    tok.write_text("abc.def.ghi\n")
    monkeypatch.setenv("BEARER_TOKEN_FILE", str(tok))
    with patch("osdf.requests.put") as mput:
        mput.side_effect = _put_side()
        osdf.upload(local, "osdf:///umn-coughlin/f.txt")
    # First call resolves via the sentinel; second is the authenticated PUT to the
    # resolved origin (not the director), with the body size declared and no redirect.
    assert mput.call_count == 2
    probe_url = mput.call_args_list[0].args[0]
    assert probe_url.endswith("/umn-coughlin/.probe")
    put_url, put_kwargs = mput.call_args_list[1].args[0], mput.call_args_list[1].kwargs
    assert put_url == f"{ORIGIN}/umn-coughlin/f.txt"
    assert put_kwargs["headers"]["Authorization"] == "Bearer abc.def.ghi"
    assert put_kwargs["headers"]["Content-Length"] == "5"
    assert put_kwargs["allow_redirects"] is False


def test_resolve_caches_per_namespace(monkeypatch):
    with patch("osdf.requests.put") as mput:
        mput.side_effect = _put_side()
        a = osdf.resolve_origin_base("osdf:///umn-coughlin/a.tar")
        b = osdf.resolve_origin_base("osdf:///umn-coughlin/b.tar")
    assert a == b == ORIGIN
    assert mput.call_count == 1  # second resolve served from cache


def test_upload_stream_auth_failure_raises_permissionerror(monkeypatch):
    monkeypatch.delenv("BEARER_TOKEN_FILE", raising=False)

    def side(url, data=None, headers=None, allow_redirects=True, timeout=None):
        if url.endswith("/.probe"):
            return _resp(307, {"Location": f"{ORIGIN}/umn-coughlin/.probe"})
        return _resp(403)

    import io

    with patch("osdf.requests.put", side_effect=side):
        with pytest.raises(PermissionError):
            osdf.upload_stream("osdf:///umn-coughlin/f.tar", io.BytesIO(b"x"))


def test_upload_stream_truncation_raises(monkeypatch):
    import io

    def side(url, data=None, headers=None, allow_redirects=True, timeout=None):
        if url.endswith("/.probe"):
            return _resp(307, {"Location": f"{ORIGIN}/umn-coughlin/.probe"})
        return _resp(200)  # never reads the body -> counter stays 0

    with patch("osdf.requests.put", side_effect=side):
        with pytest.raises(OSError, match="truncated"):
            osdf.upload_stream("osdf:///umn-coughlin/f.tar", io.BytesIO(b"xxxxx"), content_length=5)


def test_upload_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        osdf.upload(tmp_path / "does-not-exist", "https://origin/foo")


def test_read_url_translates_osdf_but_passes_https():
    assert osdf._read_url("osdf:///umn-coughlin/x.tar").startswith(osdf.OSDF_DIRECTOR)
    assert osdf._read_url("igwn+osdf:///igwn/x.pt").startswith(osdf.OSDF_DIRECTOR)
    assert osdf._read_url("https://origin/x") == "https://origin/x"


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


def test_mint_token_caches_until_near_exp(monkeypatch):
    import time

    tok = _fake_jwt(int(time.time()) + 3600)
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or MagicMock(returncode=0, stdout=tok, stderr=""),
    )
    a = osdf.mint_token("/x/secret.pem", "osdf:///umn-coughlin/a.tar", pelican_bin="pel")
    b = osdf.mint_token("/x/secret.pem", "osdf:///umn-coughlin/b.tar")  # same namespace
    assert a == b == tok and len(calls) == 1  # minted once, served from cache
    assert calls[0][:4] == ["pel", "credentials", "token", "get"]


def test_mint_token_remints_when_cached_is_expired(monkeypatch):
    import time

    seq = [_fake_jwt(int(time.time()) - 10), _fake_jwt(int(time.time()) + 3600)]
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: MagicMock(returncode=0, stdout=seq.pop(0), stderr="")
    )
    first = osdf.mint_token("/x/s.pem", "osdf:///ns1/a.tar")
    second = osdf.mint_token("/x/s.pem", "osdf:///ns1/a.tar")  # cached one is expired -> re-mint
    assert first != second


def test_mint_token_failure_raises(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: MagicMock(returncode=1, stdout="", stderr="boom")
    )
    with pytest.raises(RuntimeError, match="mint failed"):
        osdf.mint_token("/x/s.pem", "osdf:///ns1/a.tar")


def test_upload_stream_mints_from_keypair_and_puts_with_it(monkeypatch):
    import io
    import time

    tok = _fake_jwt(int(time.time()) + 3600)
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: MagicMock(returncode=0, stdout=tok, stderr="")
    )
    with patch("osdf.requests.put", side_effect=_put_side()) as mput:
        osdf.upload_stream("osdf:///umn-coughlin/f.tar", io.BytesIO(b"x"), keypair_path="/x/s.pem")
    # the real PUT (not the .probe resolve) carries the minted token
    put = [c for c in mput.call_args_list if not c.args[0].endswith("/.probe")][0]
    assert put.kwargs["headers"]["Authorization"] == f"Bearer {tok}"
