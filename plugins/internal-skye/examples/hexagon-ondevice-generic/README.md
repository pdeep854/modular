# Running a Mojo kernel on a Snapdragon Hexagon NPU from native Linux

This example compiles a Mojo HVX kernel, wraps it in a FastRPC skel, and
invokes it through the generic `DeviceContext(api="npu")` dispatch path on a
Snapdragon device running native Linux (Debian/Ubuntu aarch64). It has been
designed for the Snapdragon X Elite (X1E80100, Hexagon v81).

This is the Linux counterpart to `hexagon-ondevice/`, which targets Windows +
WSL. The key difference is the transport: Linux BSP drivers load unsigned skels
directly from `ADSP_LIBRARY_PATH` — no test-signing, no Windows Driver Kit, no
catalogue file.

The host uses `DeviceContext(api="npu")` with the generic dispatch path (method
3, selector 0) instead of the legacy typed-vadd direct-transport example. This
is the same path `QuRTDeviceContext` takes in production.

## What works

A 32-wide `float32` vector add compiled from Mojo to Hexagon v81 HVX, invoked
through `DeviceContext` with the generic wire format, validates exact results at
sizes 17, 256, and 257 (exercising the scalar tail, a full eight HVX registers,
and both together).

## Prerequisites

1. **Native Linux aarch64 host** with the CDSP FastRPC driver loaded. Confirm
   with:

   ```bash
   ls /dev/fastrpc*      # expect /dev/fastrpc-cdsp or /dev/fastrpc3
   dmesg | grep -i cdsp  # expect firmware loaded
   ```

2. **`libcdsprpc.so`** on the device. This ships with BSP packages for
   Snapdragon Linux BSP (Linaro, Thundercomm, etc.). Common locations:

   - `/vendor/lib64/libcdsprpc.so`
   - `/usr/lib/aarch64-linux-gnu/libcdsprpc.so`

   If it is in a non-standard path, set `ADSP_LIBRARY_PATH` (the directory, not
   the file) before running the host.

3. **Linux Hexagon SDK** with `qaic` and `hexagon-clang` for aarch64. Set
   `HEXAGON_SDK` to the SDK root (default `/opt/hexagon-sdk`), or set `QAIC`
   and `HEXAGON_TOOLS` individually.

4. **Plugin build of `mojo`** with Hexagon codegen:

   ```bash
   ./bazelw build --config=plugin @plugin//:skye_plugin \
     @plugin//KGEN:mojo-compiler-plugin //KGEN/tools/mojo
   ```

5. **`readelf`** for ELF verification (part of `binutils`, standard on any
   Linux system).

## Build

```bash
# From the repo root on the Linux device (or a Linux aarch64 cross-build host).
# HEX_VER selects the Hexagon revision: 73, 75, 79 or 81.
HEX_VER=81 ./plugins/internal-skye/examples/hexagon-ondevice-generic/build.sh
```

Output lands in `/tmp/mojonpu-generic/` by default. Override with `OUT=<dir>`.

## Deploy

```bash
# Copy the skel to the ADSP search path (requires root or appropriate udev rule).
sudo cp /tmp/mojonpu-generic/libmojo_npu_skel.so /vendor/lib/rfsa/adsp/

# If your BSP uses a different directory, set this instead:
export ADSP_LIBRARY_PATH=/your/bsp/rfsa/adsp
```

## Run

```bash
# From the repo root.
MODULAR_NPU_RPC_ABI=generic mojo run \
  plugins/internal-skye/examples/hexagon-ondevice-generic/device_context_vadd.mojo
```

Expected output (seven checks, all PASS):

```text
DeviceContext create ... ok
vadd n=17: residual=0 nonfinite=0 distinct=6 wrong=0 -> PASS
vadd n=256: residual=0 nonfinite=0 distinct=12 wrong=0 -> PASS
vadd n=257: residual=0 nonfinite=0 distinct=12 wrong=0 -> PASS
vadd_slice start=64 n=129: wrong=0 guard_broken=0 -> PASS
vadd_negative_count: refused=True -> PASS
unknown_selector: refused=True -> PASS

passed: 6  failed: 0
PASS
```

## How it fits together

### DSP side

`npu_kernel.mojo` exports `mojo_vadd_f32` with C ABI. It is compiled by Mojo
to a Hexagon ELF object using `--target-triple=hexagon-unknown-unknown-elf`.

`mojo_npu.idl` declares the FastRPC interface with two methods:

- Method 2 (`vadd`): legacy typed entry point, preserved for ABI compatibility.
- Method 3 (`dispatch`): generic entry point — `qurtScalars(3, 10, 8)` = 10
  input buffers (primitive block, scalar block, 8 allocation slots) and 8
  output buffers (same allocations).

`qaic` generates `mojo_npu_skel.c` from the IDL. `generic_dispatch.c`
implements both methods and routes method 3 by selector (0 = vadd). All three
objects link into `libmojo_npu_skel.so`, which must come out as `DYN`, machine
`QUALCOMM DSP6`, with exactly one writable `PT_LOAD`.

### Host side

`device_context_vadd.mojo` exercises the full `DeviceContext` path:

1. `DeviceContext(api="npu")` — opens the FastRPC session; confirms
   `libcdsprpc.so` is discoverable.
2. `load_function[npu_sig](function_name="vadd")` — resolves selector 0
   against `NPUKernels.def`.
3. `enqueue_function(f, a, b, c, n)` — packs the primitive block and scalar
   block, calls `handleInvoke` with `qurtScalars(3, 10, 8)`.
4. `synchronize()` — waits for the DSP to complete.

### Generic wire format (method 3)

```
primitive_block (19 uint32 words):
  [0]       selector      — kernel selector (0 = vadd; see NPUKernels.def)
  [1]       launch_word   — thread mask in [7:0]
  [2]       scalar_bytes  — byte length of the scalar block
  [3+2*i]   in_len_i      — input byte length for slot i   (i = 0..7)
  [4+2*i]   out_len_i     — output byte length for slot i  (i = 0..7)

scalar_block (variable bytes):
  bytes [i*8 .. +7]              — scalar argument i, zero-padded to 8 bytes
  bytes [(scalarCount+j)*8 .. +7] — uint64 byte offset for buffer slot j
```

Unused slots carry null / zero length. The DSP reconstructs pointers as
`slot_base + offset`.

## Gotchas

- **Two different FastRPC libraries exist on some BSPs.** The audio-DSP library
  also names itself `libcdsprpc.so` on some builds and refuses the `cdsp`
  domain with `AEE_ECONNREFUSED`. Use `ADSP_LIBRARY_PATH` or
  `MODULAR_NPU_TRANSPORT_LIB` to pin the correct one.
- **`ADSP_LIBRARY_PATH` is a directory**, not a file path. The BSP loader
  appends the skel filename itself.
- **Unsigned skels on Linux.** The Linux BSP FastRPC driver does not validate
  DSP ELF signatures — it trusts any skel in `ADSP_LIBRARY_PATH`. The Windows
  path requires a signed `.cat`; this path does not.
- **`kSkelURI` uses a bare filename**, not an absolute path:
  `"file:///libmojo_npu_skel.so?mojo_npu_skel_handle_invoke&_modver=1.0&_dom=cdsp"`.
  Absolute URIs are refused by the FastRPC loader.
- **v81 needs `+hvx-qfloat`**, not `+hvx-ieee-fp`. Using the wrong flag still
  compiles but may produce incorrect results on HVX fp ops.
- **AEE error codes.** `0x80000406` = `AEE_EUNABLETOLOAD` (skel not found or
  refused), `0x80000472` = `AEE_ECONNREFUSED` (wrong DSP domain or firmware
  not up), `0x80000442` = `AEE_ENOSUCHSYMBOL` (IDL mismatch).

## Differences from the Windows example

| | `hexagon-ondevice/` (Windows) | `hexagon-ondevice-generic/` (Linux) |
|---|---|---|
| Host OS | Windows 11 ARM + WSL | Native Linux aarch64 |
| Build tools | `hexagon-clang.exe`, `qaic.exe` | `hexagon-clang`, `qaic` |
| DSP auth | Signed `.cat` (Inf2Cat + PowerShell) | None (Linux BSP trusts `ADSP_LIBRARY_PATH`) |
| Host entry | Python ctypes (`run_npu.py`) | `DeviceContext(api="npu")` |
| Dispatch path | Legacy typed vadd (method 2) | Generic dispatch (method 3, selector 0) |
| Transport lib | `libcdsprpc.dll` via registry | `libcdsprpc.so` via `ADSP_LIBRARY_PATH` |
