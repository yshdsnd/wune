"""Check allocation boundaries without opening any audio devices."""
import types
import unittest
from unittest.mock import Mock, patch

from cffi import FFI
from wune.soundcard_compat import (
    _install_propvariant,
    _install_com_error_filter,
    _install_recorder_exit,
    prepare_soundcard,
)


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

    def test_freebsd_prepare_soundcard_idempotent(self):
        with patch("wune.soundcard_compat.sys.platform", "freebsd15"):
            # When soundcard is already in sys.modules, prepare_soundcard does nothing
            with patch.dict("sys.modules", {"soundcard": types.ModuleType("soundcard")}):
                prepare_soundcard()
                import sys
                self.assertEqual(sys.platform, "freebsd15")



class ComErrorFilterTests(unittest.TestCase):
    def setUp(self):
        self.orig_check = Mock()
        self.backend = types.SimpleNamespace(
            _com=types.SimpleNamespace(check_error=self.orig_check)
        )
        _install_com_error_filter(self.backend)

    def test_s_ok_and_s_false_do_not_invoke_original_check(self):
        # 0 = S_OK, 1 = S_FALSE (both are success codes)
        self.backend._com.check_error(0)
        self.backend._com.check_error(1)
        self.orig_check.assert_not_called()

    def test_non_negative_success_codes_do_not_invoke_original_check(self):
        # HRESULTs with MSB = 0 are SUCCEEDED
        self.backend._com.check_error(2)
        self.backend._com.check_error(0x7fffffff)
        self.orig_check.assert_not_called()

    def test_failure_hresults_forward_to_original_check(self):
        # Negative HRESULTs (MSB = 1, e.g. E_FAIL = -2147467259 = 0x80004005)
        self.backend._com.check_error(-2147467259)
        self.orig_check.assert_called_once_with(-2147467259)

    def test_install_is_idempotent(self):
        patched = self.backend._com.check_error
        _install_com_error_filter(self.backend)
        self.assertIs(self.backend._com.check_error, patched)


class RecorderExitTests(unittest.TestCase):
    def setUp(self):
        self.ffi = types.SimpleNamespace(NULL=object())
        self.com = types.SimpleNamespace(
            check_error=Mock(),
            release=Mock(),
        )
        self.stop_mock = Mock(return_value=0)
        self.lpvtbl = types.SimpleNamespace(Stop=self.stop_mock)
        self.ptr_target = types.SimpleNamespace(lpVtbl=self.lpvtbl)
        self.ptr = [[self.ptr_target]]
        self.capture_client = [object()]

        class MockRecorder:
            def __exit__(self, exc_type, exc_value, traceback):
                pass

        self.MockRecorder = MockRecorder
        self.backend = types.SimpleNamespace(
            _ffi=self.ffi,
            _com=self.com,
            _Recorder=MockRecorder,
        )
        _install_recorder_exit(self.backend)

    def _create_recorder_instance(self):
        rec = self.MockRecorder()
        rec._ptr = self.ptr
        rec._ppCaptureClient = self.capture_client
        return rec

    def test_clean_stop_releases_com_objects(self):
        rec = self._create_recorder_instance()
        rec.__exit__(None, None, None)
        self.stop_mock.assert_called_once_with(self.ptr[0])
        self.assertEqual(self.com.release.call_count, 2)
        self.com.check_error.assert_not_called()

    def test_s_false_stop_does_not_raise_and_releases_com_objects(self):
        # IAudioClient::Stop returns S_FALSE (1) when already stopped (sleep/suspend)
        self.stop_mock.return_value = 1
        rec = self._create_recorder_instance()
        rec.__exit__(None, None, None)
        self.stop_mock.assert_called_once_with(self.ptr[0])
        self.assertEqual(self.com.release.call_count, 2)
        self.com.check_error.assert_not_called()

    def test_error_during_stop_is_non_fatal_and_releases_com_objects(self):
        # If Stop returns AUDCLNT_E_DEVICE_INVALIDATED, error is handled and release still happens
        self.stop_mock.return_value = -2004287484  # 0x88890004
        self.com.check_error.side_effect = RuntimeError("device invalidated")
        rec = self._create_recorder_instance()
        rec.__exit__(None, None, None)
        self.stop_mock.assert_called_once_with(self.ptr[0])
        self.com.check_error.assert_called_once_with(-2004287484)
        # Release must still be called in finally block
        self.assertEqual(self.com.release.call_count, 2)

    def test_install_is_idempotent(self):
        patched = self.MockRecorder.__exit__
        _install_recorder_exit(self.backend)
        self.assertIs(self.MockRecorder.__exit__, patched)
