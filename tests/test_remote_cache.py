"""Tests for cache writes that must survive interrupted remote downloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from wrappers.location_model_encoders import CSPEncoder, _download_url_atomic
from wrappers.torchgeo_encoders import GoogleSatelliteEmbeddingEncoder


class _Response:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class RemoteCacheTests(unittest.TestCase):
    def test_google_index_replaces_invalid_cache_only_after_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "aef_index.parquet"
            index_path.write_bytes(b"truncated-cache")
            encoder = GoogleSatelliteEmbeddingEncoder.__new__(
                GoogleSatelliteEmbeddingEncoder
            )
            encoder._index_path = index_path
            encoder.remote_index_download_attempts = 1
            encoder.index_url = "https://example.invalid/aef_index.parquet"
            response = _Response([b"complete", b"-parquet"])

            def validate(path: Path) -> None:
                contents = path.read_bytes()
                if contents == b"truncated-cache":
                    raise ValueError("not parquet")
                self.assertEqual(contents, b"complete-parquet")

            with patch.object(encoder, "_validate_remote_index", side_effect=validate), patch(
                "wrappers.torchgeo_encoders.requests.get", return_value=response
            ) as get:
                encoder._ensure_remote_index()

            self.assertEqual(index_path.read_bytes(), b"complete-parquet")
            self.assertTrue(response.closed)
            get.assert_called_once_with(
                encoder.index_url, stream=True, timeout=120
            )
            self.assertEqual(list(Path(tmpdir).glob("*.part")), [])

    def test_atomic_download_never_replaces_destination_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "checkpoint.zip"
            destination.write_bytes(b"known-good")

            def fake_urlretrieve(url: str, filename: Path):
                self.assertEqual(url, "https://example.invalid/model.zip")
                Path(filename).write_bytes(b"bad-download")
                return str(filename), None

            with patch(
                "wrappers.location_model_encoders.urllib.request.urlretrieve",
                side_effect=fake_urlretrieve,
            ):
                with self.assertRaisesRegex(RuntimeError, "valid file"):
                    _download_url_atomic(
                        "https://example.invalid/model.zip",
                        destination,
                        validator=lambda _path: (_ for _ in ()).throw(
                            ValueError("invalid archive")
                        ),
                        attempts=1,
                    )

            self.assertEqual(destination.read_bytes(), b"known-good")
            self.assertEqual(list(Path(tmpdir).glob("*.part")), [])

    def test_csp_extraction_promotes_only_requested_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            archive_path = cache_dir / "model_dir.zip"
            checkpoint_member = "model_dir/test.pth.tar"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(checkpoint_member, b"checkpoint-weights")
                archive.writestr("../must-not-be-extracted", b"unsafe")

            encoder = CSPEncoder.__new__(CSPEncoder)
            with patch(
                "wrappers.location_model_encoders.CSP_CHECKPOINTS",
                {"fmow": checkpoint_member},
            ):
                checkpoint_path, variant = encoder._ensure_checkpoint(
                    {"cache": str(cache_dir)}
                )

            self.assertEqual(variant, "fmow")
            self.assertEqual(Path(checkpoint_path).read_bytes(), b"checkpoint-weights")
            self.assertFalse((cache_dir.parent / "must-not-be-extracted").exists())


if __name__ == "__main__":
    unittest.main()
