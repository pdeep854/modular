# Running a Mojo kernel on an on-device Snapdragon Hexagon NPU

This example takes a Mojo kernel, compiles it for the Hexagon NPU built into
Snapdragon X-series laptops, wraps it in a FastRPC skel, and invokes it from a
host program. It has been run successfully on a Snapdragon X2 Elite Extreme
(X2E94100, Hexagon v81) under Windows 11 on ARM.

This is distinct from the QAIC/AIC100 path that the rest of this plugin
targets. AIC100 is a discrete PCIe accelerator reached through the QAIC SDK
(`libQAic.so`) with a musl userspace on the card. The on-device NPU runs QuRT
and is reached over FastRPC. The Hexagon *codegen* is shared — AIC200 is also
v81 — but the host transport is entirely different.

## What works

A 32-wide `float32` vector add, compiled from Mojo to Hexagon v81 HVX, executes
on the NPU and returns correct results for all 256 elements:

```text
unsigned PD  : rc=0x00000000
handle_open  : rc=0x00000000 handle=0x80015b6600
invoke rc    : 0x00000000
PASS - all 256 elements correct.
```

The build retargets across Hexagon v73, v75, v79 and v81 by changing one
variable, so the same sources work on 8 Gen 3, 8 Elite, and X2-class parts.

## Prerequisites

1. **Hexagon SDK 6.6.0.0**, the `arm64-wos` build, at
   `C:\Qualcomm\Hexagon_SDK\6.6.0.0`. Available as a plain GitHub release from
   [snapdragon-toolchain/hexagon-sdk][sdk] — no Qualcomm account needed. Note
   there is no Linux-aarch64 build of the SDK, so it must live on the Windows
   side even if you build from WSL.
2. **Windows Driver Kit 10.0.26100**, for `Inf2Cat`
   (`winget install Microsoft.WindowsWDK.10.0.26100`). The Windows SDK is *not*
   required — `signtool` is replaced here by PowerShell's built-in
   `Set-AuthenticodeSignature`.
3. **A `--config=plugin` build of `mojo`**, since Hexagon codegen lives in this
   plugin rather than the main tree:

   ```bash
   ./bazelw build --config=plugin @plugin//:skye_plugin \
     @plugin//KGEN:mojo-compiler-plugin //KGEN/tools/mojo
   ```

4. **Test-signing enabled.** See the next section — this is the hard part.

[sdk]: https://github.com/snapdragon-toolchain/hexagon-sdk/releases

## Enabling test-signing on a Secured-core device

Windows refuses to load an unsigned DSP skel unless test-signing is on, and
`bcdedit /set TESTSIGNING ON` **cannot be set from a booted Windows session** on
these laptops. It returns `Access is denied` regardless of elevation, including
under Windows 11's `sudo`.

Two prerequisites, then the recovery environment. `SETUP.md` is the full
recipe; briefly:

1. The OS volume must be **fully decrypted** — suspending BitLocker is not
   enough.
2. The Secure Boot **platform keys must be cleared** so the firmware enters
   Setup Mode. Setting Secure Boot to *Disabled* in the menu is not sufficient
   on its own, and this is the step that actually unlocks the BCD store.
3. Then, from **WinRE** (Shift + Restart, Troubleshoot > Advanced options >
   Command Prompt), at the `X:\windows\system32>` prompt:

   ```cmd
   bcdedit /set {default} testsigning on
   bcdedit /set {default} nointegritychecks on
   ```

   The `{default}` token matters, since WinRE has its own boot context. No
   `diskpart` and no `/store` are needed — WinRE mounts the primary
   installation's BCD hive itself.

Confirm success with a `Test Mode` desktop watermark, or:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control' `
  -Name SystemStartOptions).SystemStartOptions
```

which should contain `TESTSIGNING`.

Note `Disable driver signature enforcement` from Startup Settings is *not* a
substitute: it governs kernel-mode PE drivers rather than DSP ELF images and has
no effect here.

Note that **Windows Device Encryption requires Secure Boot**, so BitLocker and
test-signing are mutually exclusive on this hardware. A machine set up this way
cannot have automatic disk encryption enabled.

## Build, sign, run

```bash
# From WSL. HEX_VER selects the Hexagon revision: 73, 75, 79 or 81.
HEX_VER=81 ./build.sh
```

Then, once per machine, create a self-signed code-signing certificate, trust it,
and sign the catalogue:

```powershell
cd C:\Users\Public\mojonpu
sudo cmd /c "powershell -ExecutionPolicy Bypass -File sign_skel.ps1"
```

Finally:

```powershell
python.exe C:\Users\Public\mojonpu\run_npu.py
```

## How it fits together

`npu_kernel.mojo` holds the kernel. The `abi("C")` effect goes after the
argument list, and gives it a plain C calling convention so the skel can call
it:

```mojo
def mojo_vadd_f32(a: UnsafePointer[Float32, MutAnyOrigin], ...) abi("C"):
```

`mojo_npu.idl` declares the FastRPC interface; `qaic` generates a host-side
stub and a DSP-side skel from it. `mojo_npu_imp.c` implements the three
functions the skel expects (`open`, `close`, `vadd`) and forwards to the Mojo
symbol. All three objects link into `libmojo_npu_skel.so`, which must come out
as `DYN`, machine `QUALCOMM DSP6`, with the ISA revision in the ELF flags and
exactly one writable `PT_LOAD` — more than one is rejected by the NSP loader.

`run_npu.py` is the host, in pure Python via `ctypes`, so no C compiler is
needed on the Windows side. It mirrors the marshalling that `qaic` generated
into `mojo_npu_stub.c`.

## Gotchas worth knowing

- **There are two `libcdsprpc.dll` on the system.** The one under
  `qcadsprpc8480` drives the *audio* DSP and returns `AEE_ECONNREFUSED` for the
  `cdsp` domain — including for skels Windows already trusts. The NPU's copy is
  under `qcnspmcdm8480`. Resolve it from the `qcnspmcdm` service registry entry
  rather than hardcoding a path.
- **`ADSP_LIBRARY_PATH` has no effect on Windows.** It is a Linux and Android
  mechanism. The skel and its signed `.cat` simply sit in one directory; no
  `pnputil` or driver-store installation is needed or possible.
- **Decode error codes properly.** `AEE_EOFFSET` is `0x80000400`, so
  `0x80000406` is `AEE_EUNABLETOLOAD`, `0x80000472` is `AEE_ECONNREFUSED`, and
  `0x80000442` is `AEE_ENOSUCHSYMBOL`. Guessing wastes hours.
- **Unsigned PD returning 0 is not confirmation.**
  `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` succeeds
  because the library merely records the request; the DSP can still refuse
  later at `remote_handle64_open`.
- **v81 wants `+hvx-qfloat`**, not the `+hvx-ieee-fp` used by earlier
  revisions. Getting this wrong still compiles.
- **The toolchain's shared-object startup references
  `__register_frame_info_bases` and `__cxa_finalize`**, which the DSP protection
  domain does not export. `mojo_npu_imp.c` supplies no-op stubs.
- **Windows `sudo` defaults to opening a new window**, so command output —
  including failures — is lost. Redirect to a file, or run
  `sudo config --enable disableInput` once.

## Next steps

This proves the pipeline with a trivial kernel. The ~50 HMX and HVX Mojo
kernels under `plugins/internal-skye/max/kernels/src/qaic/` are the natural
next port — `matmul_hmx.mojo` in particular exercises the matrix engine rather
than just the vector units.

That port is not merely a recompile, though. Those kernels are written and
tested against AIC100, which is **v68**: `//:qaic_arch` defaults to `aic100`,
and in CI only the `skye100` job runs the test suite on hardware, while
`skye200-compile` builds `@plugin//:compile` and nothing more — on the
`persistent-skye100` queue, since there is no AIC200 hardware. So the v81 path
exists but is compile-only and unexercised. Porting means building for v81 and
reviewing anything that assumes v68 HMX tile shapes or intrinsics.

The upside of that same fact: this laptop's NPU is v81, so it may be the first
hardware able to actually *execute* the v81 code path rather than just compile
it.

Longer term this belongs behind the same `M_driver_*` HAL as the QAIC driver
plugin, as a second FastRPC-backed implementation.
