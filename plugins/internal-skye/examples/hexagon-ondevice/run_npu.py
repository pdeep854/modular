#!/usr/bin/env python3
"""Runs a Mojo kernel on the on-device NPU via FastRPC.

Mirrors the marshalling qaic generated into mojo_npu_stub.c, so no C host
compiler is needed.
"""

import os
import sys
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
    cast,
    sizeof,
)

# The NPU's FastRPC library lives in the NSP/MCDM driver package. The
# identically named DLL under qcadsprpc8480 drives the audio DSP and refuses
# the cdsp domain.
DRIVER_STORE = (
    r"C:\Windows\System32\DriverStore\FileRepository"
    r"\qcnspmcdm8480.inf_arm64_4feba84cfc7ee278"
)
SKEL_DIR = r"C:\Users\Public\mojonpu"

CDSP_DOMAIN_ID = 3
DSPRPC_CONTROL_UNSIGNED_MODULE = 2  # session_control_req_id, remote.h
RPCMEM_HEAP_ID_SYSTEM = 25
RPCMEM_DEFAULT_FLAGS = 1

VADD_METHOD_ID = 2  # _mid in the generated stub
ELEMENT_COUNT = 256

# AEE_EOFFSET (0x80000400) plus the offsets in AEEStdErr.h.
AEE_ERRORS = {
    0x00000000: "SUCCESS",
    0x80000406: "AEE_EUNABLETOLOAD",
    0x8000040E: "AEE_EBADPARM",
    0x80000427: "AEE_ENOSUCH",
    0x80000442: "AEE_ENOSUCHSYMBOL",
    0x80000472: "AEE_ECONNREFUSED",
    0x80000473: "AEE_ENOSESSION",
}


class RemoteBuf(Structure):
    """remote_buf from remote.h — 16 bytes on a 64-bit host."""

    _fields_ = [("pv", c_void_p), ("nLen", c_size_t)]


class UnsignedModule(Structure):
    """remote_rpc_control_unsigned_module from remote.h."""

    _fields_ = [("domain", c_int), ("enable", c_int)]


def scalars(
    method: int,
    in_bufs: int,
    out_bufs: int,
    attr: int = 0,
    in_handles: int = 0,
    out_handles: int = 0,
) -> int:
    """REMOTE_SCALARS_MAKEX from remote.h."""
    return (
        (attr & 0x7) << 29
        | (method & 0x1F) << 24
        | (in_bufs & 0xFF) << 16
        | (out_bufs & 0xFF) << 8
        | (in_handles & 0xF) << 4
        | (out_handles & 0xF)
    )


def describe(rc: int) -> str:
    rc &= 0xFFFFFFFF
    return f"0x{rc:08x} {AEE_ERRORS.get(rc, '?')}"


def load_fastrpc() -> CDLL:
    os.environ["ADSP_LIBRARY_PATH"] = SKEL_DIR
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(DRIVER_STORE)
    fastrpc = CDLL(os.path.join(DRIVER_STORE, "libcdsprpc.dll"))

    fastrpc.remote_session_control.argtypes = [c_uint32, c_void_p, c_uint32]
    fastrpc.remote_session_control.restype = c_int
    fastrpc.remote_handle64_open.argtypes = [c_char_p, POINTER(c_uint64)]
    fastrpc.remote_handle64_open.restype = c_int
    fastrpc.remote_handle64_invoke.argtypes = [
        c_uint64,
        c_uint32,
        POINTER(RemoteBuf),
    ]
    fastrpc.remote_handle64_invoke.restype = c_int
    fastrpc.remote_handle64_close.argtypes = [c_uint64]
    fastrpc.remote_handle64_close.restype = c_int
    fastrpc.rpcmem_init.restype = None
    fastrpc.rpcmem_alloc.argtypes = [c_int, c_uint32, c_int]
    fastrpc.rpcmem_alloc.restype = c_void_p
    fastrpc.rpcmem_free.argtypes = [c_void_p]
    fastrpc.rpcmem_free.restype = None
    return fastrpc


def open_session(fastrpc: CDLL) -> c_uint64 | None:
    """Returns a handle to the loaded skel, or None on failure."""
    fastrpc.rpcmem_init()

    # Must precede remote_handle64_open. A zero return only means the request
    # was recorded — the DSP can still refuse at open time.
    unsigned = UnsignedModule(CDSP_DOMAIN_ID, 1)
    rc = fastrpc.remote_session_control(
        DSPRPC_CONTROL_UNSIGNED_MODULE, byref(unsigned), sizeof(unsigned)
    )
    print(f"unsigned PD  : {describe(rc)}")

    uri = (
        b"file:///libmojo_npu_skel.so?mojo_npu_skel_handle_invoke"
        b"&_modver=1.0&_dom=cdsp"
    )
    handle = c_uint64(0)
    rc = fastrpc.remote_handle64_open(uri, byref(handle))
    print(f"handle_open  : {describe(rc)} handle=0x{handle.value:x}")
    if rc != 0:
        print("\nskel did not load. Likely causes:")
        print("  AEE_EUNABLETOLOAD : catalogue unsigned or stale. It is")
        print("                      regenerated on every build, so re-run")
        print("                      sign_skel.ps1 after rebuilding.")
        print("  AEE_ECONNREFUSED  : wrong libcdsprpc.dll (audio DSP, not NSP)")
        return None
    return handle


def invoke_vadd(
    fastrpc: CDLL,
    handle: c_uint64,
    lhs_ptr: int,
    rhs_ptr: int,
    result_ptr: int,
    count: int,
) -> bool:
    """Marshals like _stub_method: primitive-in lengths, 3 in bufs, 1 rout."""
    lengths = (c_uint32 * 3)(count, count, count)
    nbytes = count * sizeof(c_float)

    args = (RemoteBuf * 4)()
    args[0].pv = cast(lengths, c_void_p)
    args[0].nLen = sizeof(lengths)
    for slot, buffer in ((1, lhs_ptr), (2, rhs_ptr), (3, result_ptr)):
        args[slot].pv = buffer
        args[slot].nLen = nbytes

    sc = scalars(VADD_METHOD_ID, in_bufs=3, out_bufs=1)
    print(
        f"invoke       : sc=0x{sc:08x} (method={VADD_METHOD_ID}, 3 in, 1 out)"
    )
    rc = fastrpc.remote_handle64_invoke(handle, sc, args)
    print(f"invoke rc    : {describe(rc)}")
    return rc == 0


def main() -> int:
    print(f"skel dir : {SKEL_DIR}")
    if not os.path.exists(os.path.join(SKEL_DIR, "libmojo_npu_skel.so")):
        print("ERROR: skel missing — run build.sh first")
        return 1

    fastrpc = load_fastrpc()
    handle = open_session(fastrpc)
    if handle is None:
        return 2

    nbytes = ELEMENT_COUNT * sizeof(c_float)
    buffers = []
    try:
        for _ in range(3):
            buffer = fastrpc.rpcmem_alloc(
                RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, nbytes
            )
            if not buffer:
                print("ERROR: rpcmem_alloc failed")
                return 3
            buffers.append(buffer)
        lhs_ptr, rhs_ptr, result_ptr = buffers

        lhs = cast(lhs_ptr, POINTER(c_float))
        rhs = cast(rhs_ptr, POINTER(c_float))
        result = cast(result_ptr, POINTER(c_float))
        for i in range(ELEMENT_COUNT):
            lhs[i] = float(i)
            rhs[i] = float(2 * i)
            result[i] = 0.0

        if not invoke_vadd(
            fastrpc, handle, lhs_ptr, rhs_ptr, result_ptr, ELEMENT_COUNT
        ):
            return 4

        wrong = [
            i for i in range(ELEMENT_COUNT) if abs(result[i] - 3.0 * i) > 1e-5
        ]
        print(f"\nresult[0:6]  : {[result[i] for i in range(6)]}")
        print(f"expected     : {[3.0 * i for i in range(6)]}")
        if wrong:
            print(f"MISMATCH in {len(wrong)}/{ELEMENT_COUNT}, first={wrong[0]}")
            return 5
        print(f"\nPASS - all {ELEMENT_COUNT} elements correct.")
        print("A Mojo-compiled kernel executed on the NPU.")
        return 0
    finally:
        for buffer in buffers:
            fastrpc.rpcmem_free(buffer)
        fastrpc.remote_handle64_close(handle)


if __name__ == "__main__":
    sys.exit(main())
