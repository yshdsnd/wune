"""Narrow workaround for SoundCard 0.4.6's undersized Windows PROPVARIANT.

Its header models the value union as one pointer (16 bytes total on Win64),
but Windows also writes BLOB values (24 bytes total). Keep this workaround
version-pinned until the upstream declaration/allocation is corrected.
"""

import sys
from importlib.metadata import version


def prepare_soundcard():
    if sys.platform != "win32":
        return
    if version("SoundCard") != "0.4.6":
        raise RuntimeError("This Windows capture workaround requires SoundCard 0.4.6.")
    from soundcard import mediafoundation
    _install_propvariant(mediafoundation)
    _install_com_error_filter(mediafoundation)
    _install_recorder_exit(mediafoundation)


def get_soundcard():
    """Import and return the soundcard module with platform compatibility applied."""
    if sys.platform.startswith("freebsd") and "soundcard" not in sys.modules:
        orig = sys.platform
        try:
            sys.platform = "linux"
            import soundcard as sc
        finally:
            sys.platform = orig
        return sc
    import soundcard as sc
    return sc


def _install_propvariant(backend):
    if getattr(backend._PropVariant, "_wune_sized", False):
        return
    ffi, ole = backend._ffi, backend._ole32
    # BLOB carries a length and pointer and covers the largest union layout;
    # DECIMAL imposes a minimum size of 16 bytes on 32-bit Windows.
    size = max(16, ffi.sizeof("BLOB_PROPVARIANT"), ffi.sizeof("PROPVARIANT"))

    class PropVariant:
        _wune_sized = True

        def __init__(self):
            self.ptr = ffi.NULL
            allocation = ole.CoTaskMemAlloc(size)
            if allocation == ffi.NULL:
                raise MemoryError("Cannot allocate Windows PROPVARIANT")
            ffi.buffer(allocation, size)[:] = b"\0" * size
            self.ptr = ffi.cast("PROPVARIANT *", allocation)

        def __del__(self):
            ptr = self.ptr
            self.ptr = ffi.NULL
            if ptr != ffi.NULL:
                try:
                    backend._com.check_error(ole.PropVariantClear(ptr))
                finally:
                    ole.CoTaskMemFree(ptr)

    backend._PropVariant = PropVariant


def _install_com_error_filter(backend):
    """Patch _COMLibrary.check_error to accept S_FALSE (0x00000001) and success codes."""
    com = getattr(backend, "_com", None)
    if com is None:
        return
    check_fn = getattr(com, "check_error", None)
    if check_fn is None or getattr(check_fn, "_wune_patched", False) is True:
        return
    orig_check = check_fn

    def check_error(hresult):
        # S_OK = 0, S_FALSE = 1
        # In COM, non-negative HRESULTs (SUCCEEDED(hr)) denote success.
        # Specifically, IAudioClient::Stop returns S_FALSE (1) when the stream
        # has already been stopped (e.g. following system sleep or reset).
        if hresult == 0 or hresult == 1:
            return
        if 0 <= hresult < 0x80000000:
            return
        orig_check(hresult)

    check_error._wune_patched = True
    com.check_error = check_error
    if hasattr(backend, "_COMLibrary"):
        backend._COMLibrary.check_error = staticmethod(check_error)


def _install_recorder_exit(backend):
    """Patch _Recorder.__exit__ to safely handle Stop errors and ensure COM object release."""
    recorder_cls = getattr(backend, "_Recorder", None)
    if recorder_cls is None or not hasattr(recorder_cls, "__exit__"):
        return
    if getattr(recorder_cls.__exit__, "_wune_safe_exit", False) is True:
        return

    def safe_exit(self, exc_type, exc_value, traceback):
        try:
            ptr = getattr(self, "_ptr", None)
            if ptr and ptr[0] != backend._ffi.NULL:
                hr = ptr[0][0].lpVtbl.Stop(ptr[0])
                # S_OK (0) and S_FALSE (1) are standard success codes.
                if hr not in (0, 1):
                    try:
                        backend._com.check_error(hr)
                    except Exception as error:
                        import warnings
                        warnings.warn(f"Ignored non-fatal error during WASAPI Stop: {error}", RuntimeWarning)
        finally:
            if hasattr(self, "_ppCaptureClient"):
                backend._com.release(self._ppCaptureClient)
            if hasattr(self, "_ptr"):
                backend._com.release(self._ptr)

    safe_exit._wune_safe_exit = True
    recorder_cls.__exit__ = safe_exit



def output_sample_rate(speaker):
    """Read the selected endpoint's shared-mode mix rate before capture starts."""
    from soundcard import mediafoundation
    return _mix_sample_rate(speaker, mediafoundation)


def _mix_sample_rate(speaker, backend):
    ffi = backend._ffi
    client = speaker._audio_client()
    try:
        fmt = ffi.new("WAVEFORMATEXTENSIBLE **")
        try:
            hr = client[0][0].lpVtbl.GetMixFormat(client[0], fmt)
            backend._com.check_error(hr)
            if fmt[0] == ffi.NULL:
                raise RuntimeError("Windows returned no output mix format.")
            rate = int(fmt[0].Format.nSamplesPerSec)
            if rate <= 0:
                raise RuntimeError("Windows returned an invalid output sample rate.")
            return rate
        finally:
            if fmt[0] != ffi.NULL:
                backend._ole32.CoTaskMemFree(fmt[0])
    finally:
        backend._com.release(client)
