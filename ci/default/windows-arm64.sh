#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# The Windows ARM64 cross-build gate. Three claims, cheapest first:
#
#   1. the cross toolchain still produces a Windows ARM64 PE
#   2. `mojo build --target-triple=aarch64-w64-windows-gnu` still links one
#   3. the host Linux build is unharmed, //:install included
#
# (3) is the reason this runs on every PR rather than behind an if-changed
# filter. Both regressions this gate exists for came from changes that touched
# no Windows file: a global mypy aspect that made every new platform
# unbuildable during loading, and an ungated platform_data dep that made
# //Mojo/tools/mojo -- and therefore //:install -- cross-build the Windows
# archive on hosts with no toolchain for it.
#
# Needs a linux-aarch64 agent. The vendored llvm-mingw is the
# ubuntu-22.04-aarch64 build, and //Mojo/tools/mojo only carries the archive
# under //:linux_aarch64, so anywhere else this job would pass while building
# nothing Windows-shaped -- which is worse than not running it.
#
# Extra bazel flags are forwarded to every invocation, so the caller supplies
# the CI configs and a developer running it by hand keeps their own.

set -euo pipefail

readonly windows_platform="//:windows-arm64-platform"
readonly smoke_target="//bazel/internal/cc-toolchain/tests:hello-windows"
readonly smoke_artifact="bazel-bin/bazel/internal/cc-toolchain/tests/hello-windows.exe"

# Avoid installing bazelisk into workspaces, since Buildkite cleans it on checkout.
if [[ -f /usr/bin/bazelisk ]]; then
  export MODULAR_OVERRIDE_BAZELISK=/usr/bin/bazelisk
fi

# A per-job output base under Buildkite, matching ci/default/test.sh. Left
# alone otherwise: a local run must not fork a second bazel server and throw
# away the developer's analysis cache.
startup_args=()
if [[ -n "${BUILDKITE_LABEL:-}" ]]; then
  startup_args+=("--output_base=$HOME/.cache/bazel/_bazel_runner/$BUILDKITE_LABEL")
fi

bazel() { ./bazelw "${startup_args[@]}" "$@"; }

# `file` is the whole assertion. CI agents run Linux and cannot execute a
# Windows binary, so this proves the artifact is the right kind of thing and
# stops there; running it needs the Windows box, by hand.
assert_windows_pe() {
  local path="$1" description
  description=$(file --brief "$path")
  echo "    $path: $description"
  case "$description" in
    "PE32+ executable (console) Aarch64"*) ;;
    *)
      echo "error: expected a Windows ARM64 console PE, got: $description" >&2
      exit 1
      ;;
  esac
}

host_arch=$(uname -m)
if [[ "$host_arch" != "aarch64" && "$host_arch" != "arm64" ]]; then
  echo "error: needs a linux-aarch64 agent, got $host_arch -- no other host" >&2
  echo "       has a windows-arm64 cc toolchain, so this job would be a no-op" >&2
  exit 1
fi

echo "--- Cross toolchain: a C binary for $windows_platform"
bazel build "$@" "--platforms=$windows_platform" "$smoke_target"
assert_windows_pe "$smoke_artifact"

echo "--- Host build: the mojo driver, which stages the Windows runtime archive"
bazel build "$@" //Mojo/tools/mojo

echo "--- mojo build: a Mojo binary for aarch64-w64-windows-gnu"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf 'def main():\n    print("hello from windows arm64")\n' > "$work/hello.mojo"

# The cross link still needs the mingw driver by hand.
# Ask bazel where the vendored one is rather than spelling the canonical repo
# directory, which bazel may rename.
driver=$(bazel cquery "$@" --output=files @llvm-mingw//:driver_files |
  grep -E '/bin/clang\+\+$' | head -1)
if [[ -z "$driver" ]]; then
  echo "error: no clang++ in @llvm-mingw//:driver_files" >&2
  exit 1
fi
MODULAR_MOJO_MAX_LINKER_DRIVER="$(bazel info "$@" execution_root)/$driver" \
  ./bazel-bin/Mojo/tools/mojo/mojo build \
  --target-triple=aarch64-w64-windows-gnu \
  "$work/hello.mojo" -o "$work/hello.exe"
assert_windows_pe "$work/hello.exe"

echo "--- Host install set"
bazel run "$@" //:install
