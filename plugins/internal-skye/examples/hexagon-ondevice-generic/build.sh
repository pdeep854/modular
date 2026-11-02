#!/usr/bin/env bash
# Builds a Mojo kernel into a FastRPC skel loadable on an on-device Snapdragon
# Hexagon NPU from a native Linux (aarch64) host. No WSL, no Windows tools, no
# signing step — Linux BSP FastRPC drivers load unsigned skels from any path
# listed in ADSP_LIBRARY_PATH.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEX_VER="${HEX_VER:-81}"           # 73, 75, 79, 81 — 81 = Snapdragon X Elite
OUT="${OUT:-/tmp/mojonpu-generic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The mojo binary must be the plugin build (Hexagon codegen lives in this
# plugin). On a native Linux aarch64 host, bazel-bin lives alongside the repo.
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MOJO="${MOJO:-$REPO/bazel-bin/KGEN/tools/mojo/mojo}"

# The Hexagon SDK (qaic + hexagon-clang) for Linux aarch64. The default path
# matches the standard BSP layout; override with HEXAGON_SDK if it is elsewhere.
SDK="${HEXAGON_SDK:-/opt/hexagon-sdk}"
QAIC="${QAIC:-$SDK/ipc/fastrpc/qaic/bin/qaic}"
TOOLS="${HEXAGON_TOOLS:-$SDK/tools/HEXAGON_Tools/19.0.07/Tools/bin}"
HEX_CC="$TOOLS/hexagon-clang"

# ---------------------------------------------------------------------------
# Validate tools
# ---------------------------------------------------------------------------

if [[ ! -x "$MOJO" ]]; then
  echo "ERROR: mojo not found at $MOJO" >&2
  echo "  Build it: ./bazelw build --config=plugin //KGEN/tools/mojo" >&2
  exit 1
fi

if [[ ! -x "$QAIC" ]]; then
  echo "ERROR: qaic not found at $QAIC" >&2
  echo "  Set HEXAGON_SDK to your SDK root, or set QAIC to the qaic binary." >&2
  exit 1
fi

if [[ ! -x "$HEX_CC" ]]; then
  echo "ERROR: hexagon-clang not found at $HEX_CC" >&2
  echo "  Set HEXAGON_TOOLS to the SDK Tools/bin directory." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# v81 uses qfloat; earlier revisions use the IEEE fp feature instead.
if [[ "$HEX_VER" == "81" ]]; then
  FEATURES="+hvx,+hvx-length128b,+hvx-qfloat,+hvxv81,+v81,-long-calls"
else
  FEATURES="+hvx,+hvx-ieee-fp,+hvx-length128b,+hvxv${HEX_VER},+v${HEX_VER},-long-calls"
fi

# ---------------------------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------------------------

mkdir -p "$OUT"
cp "$SCRIPT_DIR"/{npu_kernel.mojo,mojo_npu.idl,generic_dispatch.c} "$OUT/"

# ---------------------------------------------------------------------------
# Step 1: Mojo kernel -> Hexagon ELF object
# ---------------------------------------------------------------------------

echo ">> Mojo -> Hexagon v${HEX_VER} object"
"$MOJO" build --emit=object \
  --target-triple=hexagon-unknown-unknown-elf \
  --target-cpu="hexagonv${HEX_VER}" \
  --target-features="$FEATURES" \
  "$OUT/npu_kernel.mojo" -o "$OUT/npu_kernel.o"

# ---------------------------------------------------------------------------
# Step 2: QAIC IDL -> stub + skel C source
# ---------------------------------------------------------------------------

echo ">> qaic: IDL -> stub + skel"
"$QAIC" -mdll -o "$OUT" \
  -I "$SDK/incs" -I "$SDK/incs/stddef" \
  "$OUT/mojo_npu.idl"

# ---------------------------------------------------------------------------
# Step 3: compile skel + generic dispatcher
# ---------------------------------------------------------------------------

echo ">> hexagon-clang: compile skel and generic_dispatch"
for src in mojo_npu_skel generic_dispatch; do
  "$HEX_CC" \
    -mv"$HEX_VER" -mhvx -mhvx-length=128B -O2 -fPIC -c \
    -I "$OUT" -I "$SDK/incs" -I "$SDK/incs/stddef" \
    "$OUT/$src.c" -o "$OUT/$src.o"
done

# ---------------------------------------------------------------------------
# Step 4: link libmojo_npu_skel.so
# ---------------------------------------------------------------------------

echo ">> link libmojo_npu_skel.so"
"$HEX_CC" \
  -mv"$HEX_VER" -mhvx -mhvx-length=128B -shared -fPIC \
  -o "$OUT/libmojo_npu_skel.so" \
  "$OUT/mojo_npu_skel.o" \
  "$OUT/generic_dispatch.o" \
  "$OUT/npu_kernel.o" \
  -Wl,-Bsymbolic

# ---------------------------------------------------------------------------
# Step 5: verify ELF shape
# ---------------------------------------------------------------------------

echo ">> verifying ELF shape (want DYN / DSP6 / flags 0x${HEX_VER} / one writable PT_LOAD / 0 undefined)"
readelf -h "$OUT/libmojo_npu_skel.so" | grep -iE 'type:|machine|flags'
echo "  writable PT_LOAD : $(readelf -lW "$OUT/libmojo_npu_skel.so" | grep -c 'LOAD.*RW ')"
echo "  undefined syms   : $(readelf --dyn-syms -W "$OUT/libmojo_npu_skel.so" | awk '$7=="UND" && $8!=""{c++} END{print c+0}')"
echo "  entry exported   : $(readelf --dyn-syms -W "$OUT/libmojo_npu_skel.so" | grep -c 'mojo_npu_skel_handle_invoke$')"

# ---------------------------------------------------------------------------
# Deployment instructions
# ---------------------------------------------------------------------------

cat <<EOF

Build complete. To run on the device:

  1. Copy the skel to the ADSP library path:
       sudo cp $OUT/libmojo_npu_skel.so /vendor/lib/rfsa/adsp/

  2. Set the library search path if your BSP uses a different location:
       export ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp

  3. Confirm the FastRPC device node is present:
       ls -l /dev/fastrpc*     # expect /dev/fastrpc-cdsp or /dev/fastrpc3

  4. Run the Mojo host test:
       MODULAR_NPU_RPC_ABI=generic mojo run \\
         $SCRIPT_DIR/device_context_vadd.mojo

Linux BSP drivers load unsigned skels directly — no signing step is needed.
If the skel is refused at open time, check that your BSP's CDSP firmware is
loaded (dmesg | grep -i cdsp) and that your user has read-write access to the
FastRPC device node (/dev/fastrpc-cdsp or /dev/fastrpc3).
EOF
