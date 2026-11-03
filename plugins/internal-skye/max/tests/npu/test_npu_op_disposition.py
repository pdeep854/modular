# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Pins where each op lands on the on-device NPU, including the silent rungs.

"Registered or refused" is the rule this directory was written against, and it
is wrong: a graph can compile clean, return correct numbers, and have computed
nothing on the NPU. Two mechanisms reach that outcome, and a compile failure
sees neither.

**Relocation, at staging time.** `ops.cumsum`, `ops.nonzero`, `ops.tile` and the
five `scatter*` forms read their input's device and, finding it is not the CPU,
rewrite the graph around themselves into
`input -> mo.transfer -> op(CPU) -> mo.transfer -> output`. The op never names
the NPU device, so kernel resolution has nothing to refuse.
`_check_device_placement` reports it and its policy defaults to **Warn** -- a
`UserWarning` in a compile log nobody reads. `DevicePlacementPolicy.Error` makes
it a `ValueError` naming the op, which is why every test here sets it and why
the first test below refuses to let a new one forget.

**A kernel that never asks for device code.** A registered built-in can take a
`DeviceContext` and still run on the host, and the operands keep their NPU type
throughout, so no target check fires and `strict_device_placement` cannot see
it. Two shapes reach that, and the second is the live one:

- the registration declares `ctx: DeviceContext` and never enqueues. Exactly one
  `mo.*` op is written this way, `mo.cumsum`
  (`builtin_kernels/reductions.mojo`, calling the scalar loop in
  `nn/cumsum.mojo`), and it is unreachable because relocation fires first.
- the registration hands `ctx` to a kernel that calls
  `algorithm.functional.elementwise` **without forwarding `target`**. That
  parameter defaults to `"cpu"`, so the kernel is a host loop whatever device
  the graph named. `repeat_interleave` is exactly this, and it is why
  `SILENT_HOST_CPU` below is a measured rung rather than a hypothetical one.
  Its Python builder guards `DeviceRef.GPU()` by name, which an NPU device walks
  straight past.

The rung is correct arithmetic here -- device memory on this transport *is* host
memory -- so the failure mode is not a wrong image. It is a graph that satisfies
"every compute op dispatched through MAX's graph path" on paper while the host
does the work.

Every rung below `SIGNED_SELECTOR` is a claim about a marker being **absent**
from the exported MEF, so each case compiles in a process of its own, with a
compile cache of its own. Both are load-bearing; `_probe` says why.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import (
    NPU,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DevicePlacementPolicy,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)

_HERE = Path(__file__).resolve().parent
_PROBE = _HERE / "npu_op_disposition_probe.py"
_KERNELS = _HERE.parents[1] / "kernels" / "src" / "hexagon_ondevice"

# `sys.path` is passed through because a bazel py_test bootstraps its imports
# from a manifest rather than from the environment, so a bare subprocess would
# not find `max`.
_BOOTSTRAP = """
import json, runpy, sys
sys.path[:] = json.loads(sys.argv[1])
sys.argv = [sys.argv[2], sys.argv[3]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

_SILENT = "SILENT_HOST_CPU"

# The measured ladder, from the exhaustive sweep in
# `test_npu_op_disposition_census` (tagged `manual`). Each entry is
# (case, rung, evidence): a selector name for a dispatch, a substring of the
# diagnostic for a refusal, the kernel source for a silent rung.
#
# The cases are the kinds a FLUX denoise step stages -- its matmuls, its
# broadcast arithmetic, its activations, its RoPE, its norms, its movement --
# plus the two ops that name the silent rungs. An op moving between rungs fails
# here rather than quietly widening or narrowing what the NPU runs.
_LADDER: tuple[tuple[str, str, str | None], ...] = (
    ("matmul_f16", "SIGNED_SELECTOR", "matmul"),
    ("batch_matmul_f16", "SIGNED_SELECTOR", "matmul"),
    ("layer_norm_bf16", "SIGNED_SELECTOR", "layernorm_bf16"),
    ("group_norm_bf16", "SIGNED_SELECTOR", "groupnorm_bf16"),
    ("matmul_bf16", "REFUSED", "fp16 only"),
    ("batch_matmul_bf16", "REFUSED", "fp16 only"),
    # `add`/`mul` climbed off REFUSED when `registrations.mojo` claimed the
    # bare binary arithmetic as elementwise host walks (the encoder/VAE
    # whole-graph closes). `add` then climbed again when `Add` grew its
    # device arm: the artifact now names the signed selector its dtype picks
    # (`gate_residual_bf16` here; the ones-gate route), which is what this
    # rung measures -- engagement stays serve-gated behind
    # `MODULAR_NPU_ADD_DEVICE`, unset means the host walk, and the probe's
    # spread side is materialized upstream so the same-shape route is what
    # the artifact carries. `mul` still computes on the host and dispatches
    # nothing, leaving the same no-marker record the cast leaves. `div`
    # keeps the old REFUSED rung honest -- the claimed routes cover exactly
    # the kinds the render stages, not the lambda class wholesale.
    ("add_bf16", "SIGNED_SELECTOR", "gate_residual_bf16"),
    ("mul_bf16", "NO_DEVICE_KERNEL", None),
    ("div_bf16", "REFUSED", "elementwise.mojo"),
    # Both climbed off REFUSED when `registrations.mojo` took the ops over
    # from the elementwise lambda. A bare silu dispatches the signed row its
    # dtype picks; a cast converts on the host and dispatches nothing, which
    # leaves the same no-marker record the pure-movement ops leave -- the
    # registration's own tests (`test_npu_graph_cast.py`) carry the positive
    # tie to the stub.
    ("silu_bf16", "SIGNED_SELECTOR", "silu_bf16"),
    ("cast_bf16_to_f32", "NO_DEVICE_KERNEL", None),
    ("rope_ragged_bf16", "REFUSED", "nn/rope.mojo"),
    ("concat_bf16", "NO_DEVICE_KERNEL", None),
    ("split_bf16", "NO_DEVICE_KERNEL", None),
    ("stack_bf16", "NO_DEVICE_KERNEL", None),
    ("reshape_bf16", "LEFT_THE_NPU", None),
    ("cumsum_bf16", "RELOCATED_TO_CPU", "ops.cumsum"),
    ("nonzero_bf16", "RELOCATED_TO_CPU", "ops.nonzero"),
    ("tile_bf16", "RELOCATED_TO_CPU", "ops.tile"),
    (
        "repeat_interleave_bf16",
        _SILENT,
        "max/kernels/src/nn/repeat_interleave.mojo",
    ),
)

# The one op measured onto the silent rung, and the reason it is tolerable: a
# FLUX denoise step does not stage it. `flux2/layers/embeddings.py` reaches the
# same result with `ops.stack` plus `ops.reshape`, which the ladder above places
# on rungs that need no device kernel. Adding a name here is a decision to let
# the host compute something the NPU was asked for, so it needs a reason of the
# same kind.
_SILENT_ALLOWED = {"repeat_interleave_bf16"}

# What a FLUX denoise step actually names. None of these may be silent, whatever
# else changes about them.
_FLUX_CASES = (
    "matmul_f16",
    "batch_matmul_f16",
    "matmul_bf16",
    "batch_matmul_bf16",
    "add_bf16",
    "mul_bf16",
    "silu_bf16",
    "layer_norm_bf16",
    "rope_ragged_bf16",
    "cast_bf16_to_f32",
    "concat_bf16",
    "split_bf16",
    "stack_bf16",
    "reshape_bf16",
)


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none.

    The settings are process-wide, so they are cleared on the way out; leaving
    them set would retarget every later test in this process.
    """
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


def _probe(case: str, tmp_path: Path) -> dict:
    """Compiles one case in a process of its own and returns its record.

    Two isolations, both load-bearing. A MEF carries every kernel its exporting
    *process* compiled, not only the ones its own graph reached -- so one graph
    per process. And the compile cache is on *disk* and shared, so without a
    `MODULAR_HOME` of its own a case is served a kernel some earlier case built
    and carries that kernel's name in its MEF. Measured: `layer_norm_bf16` named
    `groupnorm_bf16` when a group-norm case had run before it in the same
    action. Both failures corrupt an absence claim in the dangerous direction --
    an op that named no selector reads as though it named one, so a silent rung
    reads as a dispatch.
    """
    env = os.environ.copy()
    env["MODULAR_HOME"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BOOTSTRAP,
            json.dumps(sys.path),
            str(_PROBE),
            case,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("###JSON###"):
            return json.loads(line[len("###JSON###") :])
    raise AssertionError(
        f"probe for {case} produced no record (rc={proc.returncode})\n"
        f"{proc.stderr[-3000:]}"
    )


# The one builder allowed to construct a `Graph` under a policy other than
# `Error`, because comparing the two policies is the whole point of the pair of
# cumsum tests below. Any other exemption is a new argument, made here.
_POLICY_EXEMPT = {("test_npu_op_disposition.py", "_cumsum_graph")}


def _graph_calls(tree: ast.Module) -> Iterator[tuple[ast.Call, str]]:
    """Yields every `Graph(...)` call with the name of the function holding it."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "Graph"
            ):
                yield inner, node.name


def test_no_npu_test_leaves_an_implicit_cpu_transfer_a_warning() -> None:
    """Every `Graph` built here makes a staging-time relocation an error.

    A structural check rather than a behavioural one, because the failure it
    guards against is a test nobody writes: an op that relocates itself to the
    CPU leaves a *passing* test whose graph ran on the host, so no existing test
    can catch it. Reading the sources is what covers the tests that do not exist
    yet.
    """
    missing: list[str] = []
    for path in sorted(_HERE.glob("test_npu_*.py")):
        tree = ast.parse(path.read_text())
        for call, holder in _graph_calls(tree):
            if (path.name, holder) in _POLICY_EXEMPT:
                continue
            policy = next(
                (
                    k
                    for k in call.keywords
                    if k.arg == "strict_device_placement"
                ),
                None,
            )
            if policy is None or (
                ast.unparse(policy.value) != "DevicePlacementPolicy.Error"
            ):
                missing.append(f"{path.name}:{call.lineno}")
    assert not missing, (
        "these `Graph` constructions accept an implicit transfer to the CPU, so"
        " an op that relocates itself off the NPU would pass silently: "
        + ", ".join(missing)
    )


def _cumsum_graph(policy: DevicePlacementPolicy) -> Graph:
    with Graph(
        "npu_cumsum",
        input_types=[TensorType(DType.bfloat16, [4, 64], DeviceRef.NPU())],
        custom_extensions=[_KERNELS],
        strict_device_placement=policy,
    ) as graph:
        graph.output(ops.cumsum(graph.inputs[0].tensor, axis=-1))
    return graph


def test_cumsum_is_relocated_by_its_own_op(virtual_npu: None) -> None:
    """The op rewrites the graph before the NPU is ever consulted.

    `test_npu_unregistered_op` recorded this as "not understood": an `mo.cumsum`
    on an NPU device compiled with no registration and no refusal. The reason is
    that there is no `mo.cumsum` on an NPU device to refuse -- `ops.cumsum`
    transferred the tensor to the CPU while staging, and the refusal never had
    anything to fire on.
    """
    with pytest.raises(ValueError) as excinfo:
        _cumsum_graph(DevicePlacementPolicy.Error)
    assert "ops.cumsum" in str(excinfo.value)


def test_the_default_policy_relocates_cumsum_with_only_a_warning(
    virtual_npu: None,
) -> None:
    """The same graph under the default policy compiles, on the host.

    The negative half of the test above, and the reason the policy is worth
    setting everywhere: nothing fails, the numbers come back right, and the only
    trace is a `UserWarning`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = _cumsum_graph(DevicePlacementPolicy.Warn)
    assert any("ops.cumsum" in str(w.message) for w in caught)
    InferenceSession(devices=[NPU()]).compile(graph)


@pytest.mark.parametrize(("case", "rung", "evidence"), _LADDER)
def test_op_lands_on_its_measured_rung(
    case: str, rung: str, evidence: str | None, tmp_path: Path
) -> None:
    """An op's fate is what it was measured to be, evidence and all.

    The rung carries its evidence: a dispatch names the signed selector it
    reaches, a refusal names the file that refused it, a relocation names the op
    that moved it, and a silent rung names the host kernel that served it. So an
    op cannot drift onto a rung by having its evidence weaken -- only by having
    it change.
    """
    sys.path.insert(0, str(_HERE))
    from npu_op_disposition_probe import classify

    record = _probe(case, tmp_path)
    actual = classify(record)
    assert actual == rung, (
        f"{case}: expected {rung}, measured {actual}\n{record}"
    )

    if rung == "SIGNED_SELECTOR":
        assert evidence in record["selectors"]
        assert record["names_hexagon_ondevice"]
    elif rung == "REFUSED":
        assert evidence in record["diagnostic"]
    elif rung == "RELOCATED_TO_CPU":
        assert any(evidence in w for w in record["staging_warnings"])
    elif rung == _SILENT:
        assert evidence in record["kernel_sources"]


def test_no_flux_op_computes_on_the_host_in_silence() -> None:
    """The rung that would make the directive satisfiable without the NPU.

    The measurement is the parametrized test above; this is the claim that
    measurement exists to support, stated where a reader will find it. An op
    FLUX stages may be dispatched, refused, or need no kernel, but it may never
    compile clean, keep its NPU-typed operands, and be served by a host loop.
    That outcome is a correct image the NPU did not produce -- exactly what
    "every compute op dispatched through MAX's graph path" excludes.

    Reading the ladder rather than re-probing, so the two cannot disagree: an
    op silently changing rung fails above, and an op dropping out of the ladder
    fails here.
    """
    pinned = {case: rung for case, rung, _ in _LADDER}
    unpinned = [c for c in _FLUX_CASES if c not in pinned]
    assert not unpinned, (
        f"a FLUX op with no measured rung: {unpinned}. Add it to the ladder --"
        " an op nobody measured is how the silent rung goes unnoticed."
    )
    silent = [c for c in _FLUX_CASES if pinned[c] == _SILENT]
    assert not silent, (
        "these ops a FLUX denoise step stages compile clean on the NPU and are"
        f" served by a host kernel: {silent}"
    )


def test_the_silent_rung_holds_only_what_was_argued_for() -> None:
    """The allow-list and the ladder agree, so neither can drift alone.

    A new silent op has to be added to both, which is where the argument for
    tolerating it gets written down.
    """
    from_ladder = {c for c, rung, _ in _LADDER if rung == _SILENT}
    assert from_ladder == _SILENT_ALLOWED
    assert not (_SILENT_ALLOWED & set(_FLUX_CASES))
