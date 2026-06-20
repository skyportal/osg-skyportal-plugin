"""Shared test fixtures: a fake `htcondor` module + the standard plugin config."""

import sys
import types
from collections.abc import Iterator

import pytest

# ---- Fake htcondor module -------------------------------------------------------------------

_FAKE_QUEUE: list[dict] = []
_FAKE_HISTORY: list[dict] = []
_NEXT_CLUSTER_ID = [10000]


class _FakeTxn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_LAST_SUBMIT_DESC: dict = {}


def _strip_classad_quotes(value):
    """Unquote ClassAd-style string literals (e.g. `"foo"` → `foo`)."""
    if isinstance(value, str) and len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


class _FakeSubmit:
    def __init__(self, desc):
        self.desc = dict(desc)
        _LAST_SUBMIT_DESC.clear()
        _LAST_SUBMIT_DESC.update(self.desc)

    def queue(self, _txn, count=1):
        cid = _NEXT_CLUSTER_ID[0]
        _NEXT_CLUSTER_ID[0] += 1
        # JobStatus 1 = idle. Carry SkyPortal* ClassAds onto the queued ad so
        # rehydrate sees the same values it would on a real schedd.
        ad = {"ClusterId": cid, "JobStatus": 1, "QDate": 1700000000}
        for k, v in self.desc.items():
            if k.startswith("+SkyPortal"):
                ad[k[1:]] = _strip_classad_quotes(v)
            elif k == "+ProjectName":
                ad["ProjectName"] = _strip_classad_quotes(v)
        _FAKE_QUEUE.append(ad)
        return cid


class _FakeSubmitResult:
    """Stand-in for htcondor2's SubmitResult (cluster + proc count)."""

    def __init__(self, cluster_id, num_procs):
        self._cluster_id = cluster_id
        self._num_procs = num_procs

    def cluster(self):
        return self._cluster_id

    def num_procs(self):
        return self._num_procs


class _FakeSchedd:
    def transaction(self):
        return _FakeTxn()

    def submit(self, sub, count=1, spool=False, itemdata=None):
        # htcondor2 path: one cluster, one proc per itemdata row (or `count`).
        items = list(itemdata) if itemdata is not None else [None] * count
        cid = _NEXT_CLUSTER_ID[0]
        _NEXT_CLUSTER_ID[0] += 1
        for proc, _item in enumerate(items):
            ad = {"ClusterId": cid, "ProcId": proc, "JobStatus": 1, "QDate": 1700000000}
            for k, v in sub.desc.items():
                if k.startswith("+SkyPortal"):
                    ad[k[1:]] = _strip_classad_quotes(v)
                elif k == "+ProjectName":
                    ad["ProjectName"] = _strip_classad_quotes(v)
            _FAKE_QUEUE.append(ad)
        return _FakeSubmitResult(cid, len(items))

    def spool(self, _result):
        pass

    def query(self, constraint=None, projection=None):
        return list(_FAKE_QUEUE)

    def history(self, constraint=None, projection=None, match=None):
        # Yield from a copy so we don't expose the mutable backing list.
        return list(_FAKE_HISTORY)


def _install_fake_htcondor():
    mod = types.ModuleType("htcondor")
    mod.Schedd = lambda *a, **k: _FakeSchedd()
    mod.Collector = lambda *a, **k: None
    mod.DaemonTypes = types.SimpleNamespace(Schedd="Schedd")
    mod.Submit = _FakeSubmit
    sys.modules["htcondor"] = mod


def _install_fake_baselayer():
    """Shim baselayer so `import main` works in tests without skyportal installed."""
    pkg = types.ModuleType("baselayer")
    app = types.ModuleType("baselayer.app")
    env = types.ModuleType("baselayer.app.env")
    log = types.ModuleType("baselayer.log")

    def load_env():
        return None, _TEST_CONFIG

    def make_log(_name):
        # Quiet by default; tests can capture via capsys if they want.
        return lambda msg: None

    env.load_env = load_env
    log.make_log = make_log

    sys.modules.setdefault("baselayer", pkg)
    sys.modules.setdefault("baselayer.app", app)
    sys.modules["baselayer.app.env"] = env
    sys.modules["baselayer.log"] = log


# ---- Plugin config used by tests ------------------------------------------------------------


class _DottedDict(dict):
    """Dict that supports `cfg["a.b.c"]` and falls back to nested-dict for `cfg[key]`."""

    def __getitem__(self, key):
        if "." in key:
            d = self
            for part in key.split("."):
                d = dict.__getitem__(d, part) if isinstance(d, dict) else d[part]
            return d
        return dict.__getitem__(self, key)


_TEST_CONFIG = _DottedDict(
    {
        "services": {
            "external": {
                "osg": {
                    "params": {
                        "listener": {"host": "127.0.0.1", "port": 17100},
                        "htcondor": {
                            "collector": None,
                            "schedd": None,
                            "scitoken_path": "/nonexistent/token",
                            "project_name": "Test",
                        },
                        "defaults": {
                            "request_cpus": 1,
                            "request_memory": 256,
                            "request_disk": 256,
                            "max_runtime_seconds": 60,
                            "singularity_image": None,
                        },
                        "poller": {"interval_seconds": 0.05},
                        "caps": {
                            "max_concurrent_total": None,
                            "max_concurrent_per_analysis": None,
                            "max_concurrent_per_resource_per_analysis": None,
                        },
                        "osdf": {
                            "output_prefix": None,
                            "read_token_path": None,
                            "write_token_path": None,
                        },
                        "staging_dir": "staging-test",
                        "auth": {"incoming_bearer_token": "secret-test-token"},
                        "skyportal": {
                            "base_url": "http://localhost:5000",
                            "api_token": "fake_api_token",
                        },
                    }
                }
            }
        }
    }
)


# Install shims BEFORE the test session imports main.
_install_fake_baselayer()
_install_fake_htcondor()


@pytest.fixture(autouse=True)
def _isolate_jobs() -> Iterator[None]:
    """Reset the in-memory JOBS dict + fake schedd state between tests."""
    import main

    main.JOBS.clear()
    _FAKE_QUEUE.clear()
    _FAKE_HISTORY.clear()
    yield
    main.JOBS.clear()
    _FAKE_QUEUE.clear()
    _FAKE_HISTORY.clear()


@pytest.fixture
def plugin_cfg() -> dict:
    return _TEST_CONFIG["services.external.osg.params"]


@pytest.fixture
def fake_queue() -> list[dict]:
    return _FAKE_QUEUE


@pytest.fixture
def fake_history() -> list[dict]:
    return _FAKE_HISTORY


@pytest.fixture
def last_submit_desc() -> dict:
    return _LAST_SUBMIT_DESC
