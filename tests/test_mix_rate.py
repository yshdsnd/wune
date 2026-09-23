import types
import unittest
from unittest.mock import Mock

from cffi import FFI
from wune.soundcard_compat import _mix_sample_rate


class MixRateTests(unittest.TestCase):
    def setUp(self):
        self.ffi = FFI()
        self.ffi.cdef("""
            typedef struct { unsigned int nSamplesPerSec; } WAVEFORMATEX;
            typedef struct { WAVEFORMATEX Format; } WAVEFORMATEXTENSIBLE;
        """)
        self.format = self.ffi.new("WAVEFORMATEXTENSIBLE *")
        self.format.Format.nSamplesPerSec = 96000
        def get_format(client, out):
            out[0] = self.format
            return 0
        self.get_format = Mock(side_effect=get_format)
        self.client = [[types.SimpleNamespace(lpVtbl=types.SimpleNamespace(GetMixFormat=self.get_format))]]
        self.speaker = types.SimpleNamespace(_audio_client=Mock(return_value=self.client))
        self.backend = types.SimpleNamespace(
            _ffi=self.ffi, _ole32=types.SimpleNamespace(CoTaskMemFree=Mock()),
            _com=types.SimpleNamespace(check_error=Mock(), release=Mock()))

    def test_reads_and_releases_format_and_client(self):
        self.assertEqual(_mix_sample_rate(self.speaker, self.backend), 96000)
        self.backend._ole32.CoTaskMemFree.assert_called_once_with(self.format)
        self.backend._com.release.assert_called_once_with(self.client)

    def test_failed_query_releases_client_without_freeing_null(self):
        self.get_format.side_effect = lambda client, out: -1
        self.backend._com.check_error.side_effect = RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            _mix_sample_rate(self.speaker, self.backend)
        self.backend._com.release.assert_called_once_with(self.client)
        self.backend._ole32.CoTaskMemFree.assert_not_called()

    def test_invalid_rate_releases_resources(self):
        self.format.Format.nSamplesPerSec = 0
        with self.assertRaisesRegex(RuntimeError, "invalid output sample rate"):
            _mix_sample_rate(self.speaker, self.backend)
        self.backend._ole32.CoTaskMemFree.assert_called_once_with(self.format)
        self.backend._com.release.assert_called_once_with(self.client)

    def test_null_format_releases_client(self):
        self.get_format.side_effect = lambda client, out: 0
        with self.assertRaisesRegex(RuntimeError, "no output mix format"):
            _mix_sample_rate(self.speaker, self.backend)
        self.backend._com.release.assert_called_once_with(self.client)
