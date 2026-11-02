#!/usr/bin/env bash
# Builds a Mojo kernel into a FastRPC skel loadable on an on-device Snapdragon
# Hexagon NPU. Run from WSL; the Hexagon SDK tools are Windows binaries invoked
# through WSL interop, so their *arguments* must be Windows paths even though
# the executables are addressed by their /mnt/c path.
set -euo pipefail

HEX_VER="${HEX_VER:-81}"          # 73, 75, 79, 81 — 81 = Snapdragon X2 / AIC200
OUT_WIN='C:\Users\Public\mojonpu'
OUT_WSL='/mnt/c/Users/Public/mojonpu'
SDK_WIN='C:\Qualcomm\Hexagon_SDK\6.6.0.0'
SDK_WSL='/mnt/c/Qualcomm/Hexagon_SDK/6.6.0.0'
TOOLS="$SDK_WSL/tools/HEXAGON_Tools/19.0.07/Tools/bin"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MOJO="$REPO/bazel-bin/KGEN/tools/mojo/mojo"

# v81 uses qfloat; earlier revisions use the IEEE fp feature instead.
if [[ "$HEX_VER" == "81" ]]; then
  FEATURES="+hvx,+hvx-length128b,+hvx-qfloat,+hvxv81,+v81,-long-calls"
else
  FEATURES="+hvx,+hvx-ieee-fp,+hvx-length128b,+hvxv${HEX_VER},+v${HEX_VER},-long-calls"
fi

mkdir -p "$OUT_WSL"
cp "$(dirname "${BASH_SOURCE[0]}")"/{npu_kernel.mojo,mojo_npu.idl,mojo_npu_imp.c,libmojo_npu.inf,run_npu.py,sign_skel.ps1} "$OUT_WSL/"

echo ">> Mojo -> Hexagon v${HEX_VER} object"
"$MOJO" build --emit=object \
  --target-triple=hexagon-unknown-unknown-elf \
  --target-cpu="hexagonv${HEX_VER}" \
  --target-features="$FEATURES" \
  "$OUT_WSL/npu_kernel.mojo" -o "$OUT_WSL/npu_kernel.o"

echo ">> qaic: IDL -> stub + skel"
(cd /mnt/c && "$SDK_WSL/ipc/fastrpc/qaic/bin/qaic.exe" -mdll -o "$OUT_WIN" \
  -I "$SDK_WIN\\incs" -I "$SDK_WIN\\incs\\stddef" "$OUT_WIN\\mojo_npu.idl")

echo ">> hexagon-clang: compile skel + impl"
for src in mojo_npu_skel mojo_npu_imp; do
  (cd /mnt/c && "$TOOLS/hexagon-clang.exe" -mv"$HEX_VER" -mhvx -mhvx-length=128B -O2 -fPIC -c \
    -I "$OUT_WIN" -I "$SDK_WIN\\incs" -I "$SDK_WIN\\incs\\stddef" \
    "$OUT_WIN\\$src.c" -o "$OUT_WIN\\$src.o")
done

echo ">> link libmojo_npu_skel.so"
(cd /mnt/c && "$TOOLS/hexagon-clang.exe" -mv"$HEX_VER" -mhvx -mhvx-length=128B -shared -fPIC \
  -o "$OUT_WIN\\libmojo_npu_skel.so" \
  "$OUT_WIN\\mojo_npu_skel.o" "$OUT_WIN\\mojo_npu_imp.o" "$OUT_WIN\\npu_kernel.o" \
  -Wl,-Bsymbolic)

echo ">> verifying ELF shape (want DYN / DSP6 / flags 0x${HEX_VER} / one writable PT_LOAD / 0 undefined)"
readelf -h "$OUT_WSL/libmojo_npu_skel.so" | grep -iE 'type:|machine|flags'
echo "  writable PT_LOAD : $(readelf -lW "$OUT_WSL/libmojo_npu_skel.so" | grep -c 'LOAD.*RW ')"
echo "  undefined syms   : $(readelf --dyn-syms -W "$OUT_WSL/libmojo_npu_skel.so" | awk '$7=="UND" && $8!=""{c++} END{print c+0}')"
echo "  entry exported   : $(readelf --dyn-syms -W "$OUT_WSL/libmojo_npu_skel.so" | grep -c 'mojo_npu_skel_handle_invoke$')"

K="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x86"
echo ">> Inf2Cat: generate catalogue"
(cd /mnt/c && "$K/Inf2Cat.exe" /driver:"$OUT_WIN" /os:10_25H2_ARM64 | tail -3)

cat <<EOF

Build complete. Remaining steps require elevation / a running NPU:

  1. Sign the catalogue (once per machine):
       sudo cmd /c "powershell -ExecutionPolicy Bypass -File $OUT_WIN\\sign_skel.ps1"
  2. Run it:
       python.exe $OUT_WIN\\run_npu.py

See README.md for the test-signing prerequisite — it must be set from a UEFI
shell on Secured-core devices.
EOF
