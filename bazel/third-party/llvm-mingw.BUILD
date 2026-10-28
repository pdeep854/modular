load("@bazel_skylib//rules/directory:directory.bzl", "directory")
load("@bazel_skylib//rules/directory:subdirectory.bzl", "subdirectory")

package(default_visibility = ["//visibility:public"])

exports_files([
    "bin/clang",
    "bin/clang++",
    "bin/llvm-ar",
    "bin/llvm-objcopy",
    "bin/llvm-strip",
])

# The clang driver resolves its resource dir, the mingw sysroot and ld.lld
# relative to its own location, so every tool needs the whole closure as
# runfiles rather than just its own binary.
filegroup(
    name = "driver_files",
    srcs = [
        "bin/aarch64-w64-windows-gnu.cfg",
        "bin/clang",
        "bin/clang++",
        "bin/ld.lld",
        "bin/lld",
        "bin/mingw32-common.cfg",
        "lib/libLLVM.so.22.1",
        "lib/libclang-cpp.so.22.1",
    ] + glob([
        "aarch64-w64-mingw32/include/**",
        "aarch64-w64-mingw32/lib/**",
        "lib/clang/22/include/**",
        "lib/clang/22/lib/windows/*aarch64*",
    ]),
)

filegroup(
    name = "archiver_files",
    srcs = [
        "bin/llvm-ar",
        "lib/libLLVM.so.22.1",
    ],
)

filegroup(
    name = "objcopy_files",
    srcs = [
        "bin/llvm-objcopy",
        "bin/llvm-strip",
        "lib/libLLVM.so.22.1",
    ],
)

directory(
    name = "root",
    srcs = glob([
        "aarch64-w64-mingw32/include/**",
        "lib/clang/22/include/**",
    ]),
)

subdirectory(
    name = "mingw_include",
    parent = ":root",
    path = "aarch64-w64-mingw32/include",
)

subdirectory(
    name = "resource_include",
    parent = ":root",
    path = "lib/clang/22/include",
)
