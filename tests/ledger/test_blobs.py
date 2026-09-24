"""FileBlobStore: content addressing, verified reads, atomic writes, and no silent repair."""

from pathlib import Path

import pytest

from sentiment_agent.hashing import sha256_hex
from sentiment_agent.ledger.blobs import BlobCorruptedError, BlobNotFoundError, FileBlobStore
from sentiment_agent.types import BlobRef, BlobStore


@pytest.fixture
def store(tmp_path: Path) -> FileBlobStore:
    return FileBlobStore(tmp_path / "blobs")


def test_put_returns_a_content_addressed_reference(store: FileBlobStore) -> None:
    typed: BlobStore = store  # satisfies the contract protocol
    data = b'{"code":"00000","data":[]}'
    ref = typed.put(data, "application/json")
    assert ref == BlobRef(sha256=sha256_hex(data), media_type="application/json", size=len(data))
    assert store.path_of(ref.sha256).read_bytes() == data
    assert typed.get(ref.sha256) == data
    assert store.has(ref.sha256)


def test_the_same_bytes_are_stored_once(store: FileBlobStore) -> None:
    first = store.put(b"same bytes", "application/json")
    second = store.put(b"same bytes", "application/octet-stream")
    assert first.sha256 == second.sha256
    assert (first.media_type, second.media_type) == ("application/json", "application/octet-stream")
    assert [p.name for p in store.root.iterdir()] == [first.sha256]


def test_an_empty_blob_is_a_blob(store: FileBlobStore) -> None:
    ref = store.put(b"", "text/plain")
    assert ref.size == 0
    assert store.get(ref.sha256) == b""


def test_bytearray_and_memoryview_are_accepted(store: FileBlobStore) -> None:
    # The contract types ``data`` as bytes; a reader's bytearray or memoryview is tolerated at run
    # time and stored as the same blob.
    from_array = store.put(bytearray(b"abc"), "text/plain")  # type: ignore[arg-type]
    from_view = store.put(memoryview(b"abc"), "text/plain")  # type: ignore[arg-type]
    assert from_array == from_view == store.put(b"abc", "text/plain")


def test_nothing_is_created_until_the_first_put(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "not-yet")
    assert store.verify_all() == []
    assert not store.has(sha256_hex(b"x"))
    assert not (tmp_path / "not-yet").exists()


def test_a_missing_blob_is_reported(store: FileBlobStore) -> None:
    with pytest.raises(BlobNotFoundError, match="no blob"):
        store.get(sha256_hex(b"never stored"))
    assert not store.has(sha256_hex(b"never stored"))


@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd",
        "/" + "a" * 63,
        "C:" + "a" * 62,
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "",
    ],
)
def test_only_a_sha256_is_a_name(store: FileBlobStore, name: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        store.get(name)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        store.path_of(name)


def test_a_tampered_blob_is_refused_on_read_and_found_by_verify_all(store: FileBlobStore) -> None:
    kept = store.put(b"untouched", "text/plain")
    ref = store.put(b"the venue said 42", "application/json")
    store.path_of(ref.sha256).write_bytes(b"the venue said 43")
    with pytest.raises(BlobCorruptedError, match="altered"):
        store.get(ref.sha256)
    assert not store.has(ref.sha256)
    assert store.has(kept.sha256)
    assert store.verify_all() == [ref.sha256]


def test_put_never_overwrites_an_altered_blob(store: FileBlobStore) -> None:
    ref = store.put(b"original", "text/plain")
    store.path_of(ref.sha256).write_bytes(b"altered")
    with pytest.raises(BlobCorruptedError, match="evidence"):
        store.put(b"original", "text/plain")
    assert store.path_of(ref.sha256).read_bytes() == b"altered"  # the evidence is left in place


def test_a_directory_named_like_a_blob_is_not_a_blob(store: FileBlobStore) -> None:
    name = sha256_hex(b"shadowed")
    (store.root / name).mkdir(parents=True)
    with pytest.raises(BlobCorruptedError, match="directory"):
        store.get(name)
    assert store.verify_all() == [name]


def test_verify_all_reports_foreign_entries_and_skips_writes_in_flight(
    store: FileBlobStore,
) -> None:
    good = store.put(b"good", "text/plain")
    (store.root / "notes.txt").write_text("not a blob", encoding="utf-8")
    (store.root / f".tmp-{good.sha256}-123-abcd").write_bytes(b"half a wr")
    assert store.verify_all() == ["notes.txt"]


def test_writes_leave_no_temporary_files(store: FileBlobStore) -> None:
    refs = {store.put(f"blob {i}".encode(), "text/plain").sha256 for i in range(5)}
    assert {p.name for p in store.root.iterdir()} == refs


@pytest.mark.parametrize("media_type", ["", "   ", "text/plain\nX-Injected: 1", "a" * 256])
def test_a_media_type_is_required_and_bounded(store: FileBlobStore, media_type: str) -> None:
    with pytest.raises(ValueError, match="media type"):
        store.put(b"x", media_type)


def test_only_bytes_are_blobs(store: FileBlobStore) -> None:
    with pytest.raises(TypeError, match="bytes"):
        store.put("text", "text/plain")  # type: ignore[arg-type]
