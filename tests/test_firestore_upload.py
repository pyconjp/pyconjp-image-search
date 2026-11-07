"""Tests for Firestore upload tracking (scripts/firestore_upload_tracking.py)
and the upload diff logic in scripts/upload_to_firestore.py."""

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from firestore_upload_tracking import (  # noqa: E402
    ensure_tracking_table,
    get_uploaded_ids,
    record_uploaded,
    tracking_table_exists,
)


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    yield c
    c.close()


class TestTracking:
    def test_no_table_returns_empty(self, conn):
        assert not tracking_table_exists(conn)
        assert get_uploaded_ids(conn, "images") == set()

    def test_record_and_get(self, conn):
        ensure_tracking_table(conn)
        assert record_uploaded(conn, "images", ["a", "b"]) == 2
        record_uploaded(conn, "face_detections", ["x"])
        assert get_uploaded_ids(conn, "images") == {"a", "b"}
        assert get_uploaded_ids(conn, "face_detections") == {"x"}

    def test_record_is_idempotent(self, conn):
        ensure_tracking_table(conn)
        record_uploaded(conn, "images", ["a", "b"])
        record_uploaded(conn, "images", ["b", "c"])
        assert get_uploaded_ids(conn, "images") == {"a", "b", "c"}
        row = conn.execute("SELECT COUNT(*) FROM firestore_uploads").fetchone()
        assert row is not None and row[0] == 3

    def test_record_empty_is_noop(self, conn):
        ensure_tracking_table(conn)
        assert record_uploaded(conn, "images", []) == 0
        assert get_uploaded_ids(conn, "images") == set()

    def test_ensure_is_idempotent(self, conn):
        ensure_tracking_table(conn)
        record_uploaded(conn, "images", ["a"])
        ensure_tracking_table(conn)
        assert get_uploaded_ids(conn, "images") == {"a"}


# --- upload_to_firestore.py (needs google-cloud-firestore) ---


class FakeDoc:
    def __init__(self, doc_id):
        self.id = doc_id


class FakeCollection:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def document(self, doc_id):
        return (self.name, doc_id)

    def select(self, _fields):
        return self

    def stream(self):
        self.client.reads += len(self.client.store.get(self.name, {}))
        return [FakeDoc(d) for d in self.client.store.get(self.name, {})]


class FakeBatch:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def set(self, ref, data):
        self.ops.append((ref, data))

    def commit(self):
        for (collection, doc_id), data in self.ops:
            self.client.store.setdefault(collection, {})[doc_id] = data


class FakeClient:
    def __init__(self, store=None):
        self.store = store or {}
        self.reads = 0

    def collection(self, name):
        return FakeCollection(self, name)

    def batch(self):
        return FakeBatch(self)


@pytest.fixture
def upload_mod(monkeypatch):
    pytest.importorskip("google.cloud.firestore_v1")
    import upload_to_firestore

    monkeypatch.setattr(upload_to_firestore, "SLEEP_SECONDS", 0)
    return upload_to_firestore


class TestResolveUploadedIds:
    def test_uses_records_without_reading(self, upload_mod, conn):
        ensure_tracking_table(conn)
        record_uploaded(conn, "images", ["a"])
        fs = FakeClient({"images": {"a": {}, "b": {}}})
        got = upload_mod.resolve_uploaded_ids(
            conn, fs, "images", ["a", "b"], dry_run=False, force=False, assume_uploaded=False
        )
        assert got == {"a"}
        assert fs.reads == 0

    def test_bootstraps_from_firestore_once(self, upload_mod, conn):
        ensure_tracking_table(conn)
        fs = FakeClient({"images": {"a": {}, "b": {}}})
        got = upload_mod.resolve_uploaded_ids(
            conn, fs, "images", ["a", "b", "c"], dry_run=False, force=False, assume_uploaded=False
        )
        assert got == {"a", "b"}
        assert fs.reads == 2
        assert get_uploaded_ids(conn, "images") == {"a", "b"}

    def test_assume_uploaded_records_all_without_reading(self, upload_mod, conn):
        ensure_tracking_table(conn)
        fs = FakeClient()
        got = upload_mod.resolve_uploaded_ids(
            conn, fs, "images", ["a", "b"], dry_run=False, force=False, assume_uploaded=True
        )
        assert got == {"a", "b"}
        assert fs.reads == 0
        assert get_uploaded_ids(conn, "images") == {"a", "b"}

    def test_assume_uploaded_ignored_when_records_exist(self, upload_mod, conn):
        ensure_tracking_table(conn)
        record_uploaded(conn, "images", ["a"])
        got = upload_mod.resolve_uploaded_ids(
            conn,
            FakeClient(),
            "images",
            ["a", "b"],
            dry_run=False,
            force=False,
            assume_uploaded=True,
        )
        assert got == {"a"}

    def test_dry_run_does_not_record(self, upload_mod, conn):
        got = upload_mod.resolve_uploaded_ids(
            conn, None, "images", ["a"], dry_run=True, force=False, assume_uploaded=True
        )
        assert got == {"a"}
        assert not tracking_table_exists(conn)

    def test_force_uploads_everything(self, upload_mod, conn):
        ensure_tracking_table(conn)
        record_uploaded(conn, "images", ["a"])
        got = upload_mod.resolve_uploaded_ids(
            conn, FakeClient(), "images", ["a"], dry_run=False, force=True, assume_uploaded=False
        )
        assert got == set()


class TestCommitInBatches:
    def test_records_each_committed_batch(self, upload_mod, conn, monkeypatch):
        monkeypatch.setattr(upload_mod, "BATCH_LIMIT", 2)
        ensure_tracking_table(conn)
        fs = FakeClient()
        docs = [(f"d{i}", {"n": i}) for i in range(5)]
        assert upload_mod.commit_in_batches(fs, conn, "images", docs, len(docs)) == 5
        assert set(fs.store["images"]) == {f"d{i}" for i in range(5)}
        assert get_uploaded_ids(conn, "images") == {f"d{i}" for i in range(5)}

    def test_failed_batch_is_not_recorded(self, upload_mod, conn, monkeypatch):
        monkeypatch.setattr(upload_mod, "BATCH_LIMIT", 2)
        ensure_tracking_table(conn)
        fs = FakeClient()
        commits = {"n": 0}
        original_commit = FakeBatch.commit

        def flaky_commit(self):
            commits["n"] += 1
            if commits["n"] == 2:
                raise RuntimeError("quota exceeded")
            original_commit(self)

        monkeypatch.setattr(FakeBatch, "commit", flaky_commit)
        docs = [(f"d{i}", {}) for i in range(5)]
        with pytest.raises(RuntimeError):
            upload_mod.commit_in_batches(fs, conn, "images", docs, len(docs))
        assert get_uploaded_ids(conn, "images") == {"d0", "d1"}
