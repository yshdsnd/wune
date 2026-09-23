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
