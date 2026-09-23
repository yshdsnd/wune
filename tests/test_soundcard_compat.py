"""Check allocation boundaries without opening any audio devices."""
import types
import unittest
from unittest.mock import Mock, patch

from cffi import FFI
from wune.soundcard_compat import _install_propvariant, prepare_soundcard


class PropVariantTests(unittest.TestCase):
    def setUp(self):
        self.ffi = FFI()
        # Fixed Windows-width scalar types also make these tests portable.
        self.ffi.cdef("""
            typedef struct { unsigned short vt, r1, r2, r3; void *data; } PROPVARIANT;
            typedef struct { unsigned int cbSize; unsigned char *pBlobData; } BLOB;
            typedef struct { unsigned short vt, r1, r2, r3; BLOB blob; } BLOB_PROPVARIANT;
        """)
        self.allocations = []
        def allocate(size):
            block = self.ffi.new("unsigned char[]", size)
            self.allocations.append(block)
            return block
        self.ole = types.SimpleNamespace(
            CoTaskMemAlloc=Mock(side_effect=allocate),
            PropVariantClear=Mock(return_value=0), CoTaskMemFree=Mock())
        self.backend = types.SimpleNamespace(
            _ffi=self.ffi, _ole32=self.ole, _com=types.SimpleNamespace(check_error=Mock()),
            _PropVariant=type("Original", (), {}))
        _install_propvariant(self.backend)

    def test_blob_fits_zeroed_allocation_and_is_freed_once(self):
        variant = self.backend._PropVariant()
        size = self.ole.CoTaskMemAlloc.call_args.args[0]
        self.assertGreaterEqual(size, self.ffi.sizeof("BLOB_PROPVARIANT"))
        self.assertEqual(bytes(self.ffi.buffer(variant.ptr, size)), b"\0" * size)
        blob = self.ffi.cast("BLOB_PROPVARIANT *", variant.ptr)
        blob.blob.cbSize = 40
        blob.blob.pBlobData = self.ffi.NULL
        variant.__del__()
        variant.__del__()
        self.ole.PropVariantClear.assert_called_once()
        self.ole.CoTaskMemFree.assert_called_once()

    def test_allocation_failure_does_not_clear_or_free_null(self):
        self.ole.CoTaskMemAlloc.side_effect = None
        self.ole.CoTaskMemAlloc.return_value = self.ffi.NULL
        with self.assertRaises(MemoryError):
            self.backend._PropVariant()
        self.ole.PropVariantClear.assert_not_called()
        self.ole.CoTaskMemFree.assert_not_called()

    def test_clear_failure_still_frees_outer_allocation(self):
        variant = self.backend._PropVariant()
        self.backend._com.check_error.side_effect = RuntimeError("clear failed")
        with self.assertRaisesRegex(RuntimeError, "clear failed"):
            variant.__del__()
        self.ole.CoTaskMemFree.assert_called_once()

    def test_install_is_idempotent(self):
        original = self.backend._PropVariant
        _install_propvariant(self.backend)
        self.assertIs(self.backend._PropVariant, original)

    def test_unknown_version_is_rejected_before_patching(self):
        with patch("wune.soundcard_compat.sys.platform", "win32"), \
             patch("wune.soundcard_compat.version", return_value="0.4.7"):
            with self.assertRaisesRegex(RuntimeError, "requires SoundCard"):
                prepare_soundcard()
