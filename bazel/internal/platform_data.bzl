"""Rebuild a target for an explicit platform and forward its outputs.

Used to stage cross-compiled artifacts in a host binary's runfiles: the mojo
driver carries the Windows ARM64 CompilerRT archive this way, so `mojo build
--target-triple=aarch64-w64-windows-gnu` finds the real runtime without being
told where it is. The archive needs no external setup; the link still needs a
mingw driver via MODULAR_MOJO_MAX_LINKER_DRIVER.
"""

def _platform_transition_impl(_settings, attr):
    return {"//command_line_option:platforms": str(attr.platform)}

_platform_transition = transition(
    implementation = _platform_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _platform_data_impl(ctx):
    target = ctx.attr.target[0]
    files = target[DefaultInfo].files
    return [
        DefaultInfo(
            files = files,
            runfiles = ctx.runfiles(transitive_files = files).merge(
                target[DefaultInfo].default_runfiles,
            ),
        ),
    ]

platform_data = rule(
    implementation = _platform_data_impl,
    attrs = {
        "target": attr.label(
            cfg = _platform_transition,
            mandatory = True,
            doc = "The target to build for `platform`.",
        ),
        "platform": attr.label(
            mandatory = True,
            doc = "The platform to build `target` for.",
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    doc = "Builds `target` for `platform` and forwards its default outputs.",
)
