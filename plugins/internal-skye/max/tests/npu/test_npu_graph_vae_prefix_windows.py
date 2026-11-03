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
"""Mints the VAE PREFIX LADDER: the whole graph truncated after a named seam.

DRAFT-304's instrument. Order 303 proved one conv dispatch correct on
silicon (rel_rms 2.89e-04, repeats byte-identical), the study's falsifier
proved the flat-gray signature is activation-path class, and 298 cleared the
registry values — so the defect lives in MULTI-OP interaction. Each rung
here is the production VAE graph cut after one more seam; the device walks
the ladder and the FIRST divergent rung names the seam.

Rungs (`FLUX_VAE_PREFIX`), all sharing the packed-contract input
`[1, 1024, 128]` fp16 and the front matter (bn denormalize + 2x2
unpatchify), built from the decoder's OWN modules (`vae_flux2`) with
production weight FQNs so the device binds production registry bytes:

    a  front matter -> post_quant_conv -> conv_in                (6 rows)
    b  a -> mid.resnets[0]: norm1 -> silu -> conv1               (10 rows)
    c  a -> mid.resnets[0] complete (+ residual add)             (14 rows)
    d  a -> mid_block complete (resnet0, ATTENTION, resnet1)     (32 rows)
    e  d -> up_blocks[0] (3 resnets + nearest-2x + conv)         (58 rows)

One deliberate deviation from the production stager, with its reason: the
whole-graph MEF keeps two shape-carrier inputs whose LENGTHS convey h/w;
this ladder pins h = w = 32 and takes ONE input, because the lane's probe
driver (`npu_conv_probe_exec.c`, 303) binds a single input and the carriers
never touch data. If the defect were in symbolic-dim handling the ladder
could stay clean while the full graph fails — that lives in DRAFT-304's
all-rungs-clean branch.

Its own process per rung, as every cross-compile here is:
`set_virtual_cpu_target` is process-global and irreversible.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from max.driver import (
    NPU,
    set_virtual_cpu_target,
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

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

# The locked Klein VAE shape (`test_flux_vae_whole_graph._KLEIN_VAE`).
_KLEIN_VAE = {
    "latent_channels": 32,
    "out_channels": 3,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "mid_block_add_attention": True,
    "use_post_quant_conv": True,
    "batch_norm_eps": 1e-4,
}

_PACKED = 32  # 512x512 render: 32x32 packed tokens
# a0-a3 are DRAFT-306's sub-rung split of the divergent rung a (304: rung a
# 576x out of band): a0 = movement ops only (bn skipped), a1 = full front
# matter, a2 = a1 + post_quant_conv (the ConvToMatmul k-pad arm), a3 = a1 +
# conv_in (the materializing arm; post_quant skipped — both are 32-channel,
# so the wiring is legal and the a2/a3 pair isolates each conv arm).
_EXPECTED_ROWS = {
    "a0": 0,
    "a1": 2,
    "a2": 4,
    "a3": 4,
    "a": 6,
    "b": 10,
    "c": 14,
    "d": 32,
    "e": 58,
}


def _build_fused(dev: DeviceRef) -> object:
    from max.pipelines.architectures.autoencoders.vae_flux2 import Decoder
    from max.pipelines.architectures.flux2.components.vae_decoder import (
        PostprocessAndDecode,
    )

    decoder = Decoder(
        in_channels=_KLEIN_VAE["latent_channels"],
        out_channels=_KLEIN_VAE["out_channels"],
        up_block_types=("UpDecoderBlock2D",)
        * len(_KLEIN_VAE["block_out_channels"]),
        block_out_channels=_KLEIN_VAE["block_out_channels"],
        layers_per_block=_KLEIN_VAE["layers_per_block"],
        norm_num_groups=_KLEIN_VAE["norm_num_groups"],
        act_fn="silu",
        norm_type="group",
        mid_block_add_attention=_KLEIN_VAE["mid_block_add_attention"],
        use_post_quant_conv=_KLEIN_VAE["use_post_quant_conv"],
        device=dev,
        dtype=DType.float16,
    )
    fused = PostprocessAndDecode(
        decoder=decoder,
        batch_norm_eps=_KLEIN_VAE["batch_norm_eps"],
        num_channels=_KLEIN_VAE["latent_channels"] * 4,
        device=dev,
        dtype=DType.float16,
    )
    for name, weight, _layer in fused._iter_named_weights():
        weight.name = name
    return fused


def _front_matter(fused: object, lat, dev: DeviceRef, with_bn: bool = True):
    """The packed latent to the decoder's door: mirrors
    `PostprocessAndDecode.__call__` with h = w = _PACKED pinned.

    `with_bn=False` (sub-rung a0) keeps ONLY the movement ops — the
    multiset-preservation discriminator: a pure movement defect must show
    the exact input values in wrong positions.
    """
    h = w = _PACKED
    c = 128
    x = ops.reshape(lat, (1, h, w, c))
    x = ops.permute(x, [0, 3, 1, 2])
    if with_bn:
        bn_mean = fused.decoder_bn_mean.to(dev)
        bn_var = fused.decoder_bn_var.to(dev)
        bn_mean_r = ops.reshape(bn_mean, (1, c, 1, 1))
        bn_var_r = ops.reshape(bn_var, (1, c, 1, 1))
        bn_std = ops.sqrt(
            bn_var_r
            + ops.constant(
                _KLEIN_VAE["batch_norm_eps"], DType.float16, device=dev
            )
        )
        x = x * bn_std + bn_mean_r
    x = ops.reshape(x, (1, c // 4, 2, 2, h, w))
    x = ops.permute(x, [0, 1, 4, 2, 5, 3])
    return ops.reshape(x, (1, c // 4, h * 2, w * 2))


def test_npu_vae_prefix_cross_compile_for_windows_arm64(
    tmp_path: Path,
) -> None:
    rung = os.environ.get("FLUX_VAE_PREFIX", "a")
    assert rung in _EXPECTED_ROWS, rung

    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    fused = _build_fused(dev)
    d = fused.decoder

    with Graph(
        f"npu_vae_prefix_{rung}_fp16_win",
        input_types=[
            TensorType(DType.float16, [1, _PACKED * _PACKED, 128], dev),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        z = _front_matter(
            fused, graph.inputs[0].tensor, dev, with_bn=(rung != "a0")
        )
        if rung in ("a0", "a1"):
            graph.output(z)
        elif rung == "a2":
            graph.output(d.post_quant_conv(z))
        elif rung == "a3":
            graph.output(d.conv_in(z))
        else:
            _tail(graph, d, z, rung)

    graph_str = str(graph)
    if rung != "a0":
        assert "mo.constant.external" in graph_str
    used = sorted(
        name
        for name, _w, _l in fused._iter_named_weights()
        if name in graph_str
    )
    assert len(used) == _EXPECTED_ROWS[rung], (rung, len(used), used)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"flux_vae_prefix_{rung}_fp16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    if rung in ("a0", "a1"):
        # STRUCTURAL FINDING (DRAFT-306): the front matter contains no NPU
        # op at all — movement and bn compile to CPU kernels the MEF runs on
        # the DEVICE HOST (mgp.module.load.cpu; the movement chain
        # canonicalizes to bulk_slice + two mo.transpose). These two rungs
        # therefore probe the windows-arm64 CPU codegen, which no leg before
        # the device run ever executed (303's numeric leg was linux x86).
        assert b"mgp.module.load.cpu" in blob
    else:
        assert b"hexagon_ondevice" in blob
        assert b'{"label":"npu","id":0}' in blob
    for name in used:
        assert name.encode() in blob, name

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
        (Path(outputs) / f"prefix_{rung}_rows.json").write_text(
            json.dumps({"rung": rung, "rows": len(used), "names": used})
            + "\n"
        )


def _tail(graph: Graph, d: object, z, rung: str) -> None:
    """Rungs a-e: the original ladder tails, unchanged from DRAFT-304."""
    s = d.conv_in(d.post_quant_conv(z))
    if rung == "a":
        out = s
    elif rung == "b":
        r0 = d.mid_block.resnets[0]
        out = r0.conv1(ops.silu(r0.norm1(s)))
    elif rung == "c":
        out = d.mid_block.resnets[0](s)
    elif rung == "d":
        out = d.mid_block(s)
    else:
        out = d.up_blocks[0](d.mid_block(s))
    graph.output(out)
