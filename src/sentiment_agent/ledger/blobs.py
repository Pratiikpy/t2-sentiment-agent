"""The blob store: raw payloads kept once, named by their SHA-256.

A ledger event commits to the raw material behind it (a full API response, a rendered prompt, a raw
completion, an OpenTimestamps proof) by carrying its :class:`~sentiment_agent.types.BlobRef`. The
bytes themselves live here, in ``<root>/<sha256>`` (DESIGN.md §7), so a snapshot that quotes forty
responses does not copy forty responses into the chain, and a judge can still open every one.

Because a blob's name *is* its hash, the store needs no index and cannot hold two versions of
anything. What it must never do is hand back bytes that are not the bytes that were named:

* **Reads verify.** :meth:`FileBlobStore.get` re-hashes what it read and raises
  :class:`BlobCorruptedError` when the file no longer matches its name. A changed blob is found the
  moment anything touches it, not only when someone runs :meth:`FileBlobStore.verify_all`.
* **Writes are atomic.** A blob is written to a temporary file, flushed to disk and renamed into
  place, so a crash leaves either no blob or the whole blob, never half of one under the real name.
* **Writes never repair.** Storing bytes whose name already holds *different* content raises
  :class:`BlobCorruptedError` instead of overwriting: the existing file is evidence that something
  altered the store, and quietly replacing it would erase that evidence.
* **Names are checked.** Only 64 lowercase hex characters are accepted as a name, so no caller can
  reach outside the root (``../`` and absolute paths are refused before any path is built).

The media type is carried by the :class:`BlobRef` in the ledger, not stored beside the blob: the
same bytes are the same blob whatever a caller called them, and the event that used them says what
they were.
"""

import contextlib
import os
import re
import secrets
import time
from pathlib import Path
from typing import Final

from sentiment_agent.hashing import sha256_hex
from sentiment_agent.types import BlobRef

BLOB_NAME: Final = re.compile(r"^[0-9a-f]{64}$")
"""A blob's file name: its full SHA-256, lowercase hex."""

MEDIA_TYPE_MAX_LENGTH: Final = 255

_TEMP_PREFIX: Final = ".tmp-"
"""In-flight writes are named ``.tmp-<sha256>-<pid>-<nonce>`` and renamed when complete."""

_REPLACE_ATTEMPTS: Final = 20
_REPLACE_BACKOFF_S: Final = 0.01


class BlobStoreError(RuntimeError):
    """The store cannot give a truthful answer."""


class BlobNotFoundError(BlobStoreError):
    """No blob of that name is stored."""


class BlobCorruptedError(BlobStoreError):
    """A stored file no longer hashes to its name."""


def _require_name(sha256: str) -> str:
    if not isinstance(sha256, str) or not BLOB_NAME.fullmatch(sha256):
        raise ValueError(f"a blob name is 64 lowercase hex characters, not {sha256!r}")
    return sha256


def _require_media_type(media_type: str) -> str:
    if not isinstance(media_type, str) or not media_type.strip():
        raise ValueError("a blob needs a media type, e.g. 'application/json'")
    if len(media_type) > MEDIA_TYPE_MAX_LENGTH:
        raise ValueError(f"media type longer than {MEDIA_TYPE_MAX_LENGTH} characters")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in media_type):
        raise ValueError("a media type cannot contain control characters")
    return media_type


def _replace(source: Path, target: Path) -> None:
    """``os.replace`` with a short retry.

    On Windows a rename onto a file that another process is reading at that instant fails with
    ``PermissionError`` rather than waiting; the reader holds it for microseconds.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)


class FileBlobStore:
    """Content-addressed files under one directory. Satisfies :class:`types.BlobStore`.

    Nothing is created until the first :meth:`put`, so opening a store to verify an exported record
    never leaves a directory behind.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_of(self, sha256: str) -> Path:
        """Where the blob named ``sha256`` lives. The name is validated first."""
        return self._root / _require_name(sha256)

    def put(self, data: bytes, media_type: str) -> BlobRef:
        """Store ``data`` (once) and return its reference.

        Storing the same bytes again is a no-op that returns an equal reference. Raises
        :class:`BlobCorruptedError` if a file already stored under this hash holds other bytes.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"a blob is bytes, not {type(data).__name__}")
        raw = bytes(data)
        _require_media_type(media_type)
        sha = sha256_hex(raw)
        ref = BlobRef(sha256=sha, media_type=media_type, size=len(raw))
        target = self._root / sha
        if target.is_file():
            self._require_intact(target, sha)
            return ref
        self._root.mkdir(parents=True, exist_ok=True)
        temp = self._root / f"{_TEMP_PREFIX}{sha}-{os.getpid()}-{secrets.token_hex(4)}"
        try:
            with temp.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                _replace(temp, target)
            except PermissionError:
                # Another writer is putting the same blob and a reader holds it open. If what is
                # there now is these bytes, the store already holds exactly what was asked for.
                if not target.is_file():
                    raise
                self._require_intact(target, sha)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()
        return ref

    def get(self, sha256: str) -> bytes:
        """The bytes named ``sha256``, verified against the name before they are returned."""
        target = self.path_of(sha256)
        if target.is_dir():
            raise BlobCorruptedError(f"{sha256} in the store is a directory, not a blob")
        try:
            raw = target.read_bytes()
        except FileNotFoundError:
            raise BlobNotFoundError(f"no blob {sha256} in the store") from None
        if sha256_hex(raw) != sha256:
            raise BlobCorruptedError(
                f"blob {sha256} no longer hashes to its name; the stored file was altered"
            )
        return raw

    def has(self, sha256: str) -> bool:
        """True when the blob is stored and still hashes to its name."""
        try:
            self.get(sha256)
        except (BlobNotFoundError, BlobCorruptedError):
            return False
        return True

    def verify_all(self) -> list[str]:
        """Every entry under the root that is not an intact blob, by name, sorted.

        An entry fails when its content does not hash to its name, when its name is not a
        SHA-256, or when it is not a regular file. In-flight temporary files are skipped. An empty
        list means every stored blob is exactly the bytes its name commits to.
        """
        if not self._root.is_dir():
            return []
        bad: list[str] = []
        for entry in self._root.iterdir():
            name = entry.name
            if name.startswith(_TEMP_PREFIX):
                continue
            if not BLOB_NAME.fullmatch(name) or not entry.is_file():
                bad.append(name)
                continue
            if sha256_hex(entry.read_bytes()) != name:
                bad.append(name)
        return sorted(bad)

    @staticmethod
    def _require_intact(target: Path, sha: str) -> None:
        if sha256_hex(target.read_bytes()) != sha:
            raise BlobCorruptedError(
                f"blob {sha} is already stored with different content; the store was altered. "
                "Refusing to overwrite the evidence"
            )


__all__ = [
    "BLOB_NAME",
    "BlobCorruptedError",
    "BlobNotFoundError",
    "BlobStoreError",
    "FileBlobStore",
]
