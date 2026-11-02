"""W8A8 integer policy for the HMX matrix array: the loader, the record, the MMA.

Target: Hexagon v81 (HMX, 32x32). The scope is `.agentwork/flux-npu/
HMX_INT8_SCOPE.md`, and this file implements its integration slices 1-4 for the
MATMUL family only. Attention stays fp16 (scope §5).

THE ONE MECHANISM. The integer datapath is `uint8 activation x int8 weight ->
int32 accumulate`, and the readout requants through a 256-byte record per 32-
column output tile (scope §1.3, from the RE'd programming guide's
`03-prepare-the-data.md:119-127,139`):

```text
bytes 0..127     32 x (fp16 scale, fp16 baseline)      per OUTPUT CHANNEL
bytes 128..255   32 x int32 effective_bias
out_u8 = clamp(acc * scale_fp16 / 512 + baseline)
```

THE `/512` IS THE WHOLE LESSON. The fp16 datapath does not divide, so its
identity record is `0x3c00` (1.0); the integer one does, so its identity is
`0x6000` (512.0). Carrying the fp16 constant across is what made every int8
probe this project and `cdsp_peak` ever ran read all-zero — `acc <= 255` over
`/512` truncates to nothing at every position (scope §2 H1). Nothing here writes
an fp16 constant into an integer record.

WHAT v1 QUANTIZES, and what it deliberately does not:

- Weights: per OUTPUT CHANNEL, symmetric int8, over the WHOLE K. Per-K-block
  scales are not expressible in the record and emulating them costs an
  accumulator drain per 32 K (scope §1.3, §5), so they are refused rather than
  approximated.
- Activations: STATIC PER-TENSOR scale and zero point, calibrated offline and
  baked into the artefact. `effective_bias` folds `input_zp` statically, so
  dynamic per-tile quantization would mean recomputing a per-channel int32
  vector per dispatch (scope §5). NAMED CEILING: a static per-tensor activation
  scale is this configuration's accuracy weak point for a DiT, whose activations
  carry outliers across the timestep schedule; if a family fails Tier B on
  activations rather than weights, dynamic scaling is the follow-up and this is
  the seam it lands on.
- The output zero point rides the int32 `effective_bias`, NOT the fp16
  `baseline`, which stays 0. The guide states both `out = clamp(acc * scale/512
  + baseline)` and `baseline[c] = output_zp << 7`, and under an fp16 baseline
  those two disagree by 128x. Folding the zero point into the accumulator side
  — whose units are unambiguous — makes the v1 arithmetic testable without
  settling that, and rung 6 of the ladder is what settles it.

WHAT THIS FILE IS NOT ALLOWED TO CLAIM. No integer HMX instruction in this tree
has ever executed. The MMA below is behind `HMX_EMIT_MMA` exactly as the fp16
one is, the modelled arm is what the host twin scores, and the model's
composition of the two guide statements (`out = clamp((acc + bias) * scale/512 +
baseline)`) is an inference the device ladder tests, not a fact. llama.cpp is
NOT evidence here: it has no integer HMX path at all (scope §1.1).
"""
from std.math import floor, round
from std.memory import AddressSpace, bitcast
from std.sys import align_of, get_defined_bool, inlined_assembly

from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from .hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_RECORD_BYTES,
    HMX_RECORD_LANES,
    HMX_RECORD_VECTORS,
    HMX_TILE,
    CroutonPanelOf,
    HmxRequantRecord,
    HmxScratch,
    HmxRows2D,
    HmxSplit,
    HmxTileBytes,
    HmxOperands,
    _hmx_addr,
    hmx_k_tiles,
    hmx_panel,
    hmx_subpanel,
)

comptime HMX_INT8 = get_defined_bool["SKYE_HMX_INT8", False]()
"""Whether the matmul stripe has a W8A8 arm at all.

OFF, and off compiles the whole arm away: `matmul_bt_stripe_hmx_step` picks its
operand type from this, so a flag-off object is byte-identical to a build of
this branch's base and the inertness gate is `cmp` rather than a claim. It is a
`-D SKYE_HMX_INT8=true` build define (`build.sh HMX_INT8=1`) for the reason
`SKYE_HMX_EMIT_MMA` is one: both arms of the device A/B come from the same
reviewed source with no edit in between.

The define is NOT sufficient on its own, and that is deliberate. An int8
artefact and an fp16 one are neither the same bytes nor the same meaning, and
the pairing that would be silently wrong — a flag-on kernel reading an fp16
registry — is refused by the loader on the layout tag
(`hmx-crouton-cpm-i8`, `npu_registry_load.h`), by name, before any dispatch.

Read at the ONE call site that picks the arm (`_matmul_stripe_hmx`), as a
`comptime if` over two literal dtypes, rather than through an alias like
`DType.uint8 if HMX_INT8 else DType.float16`. A parameter expression built from
a conditional does NOT fold in a signature position -- it stays unfolded at
parse time and then converts to neither concrete panel type -- so the arm is a
plain `dtype` parameter everywhere below and a literal at the branch."""

comptime HMX_RECORD_LATE = get_defined_bool["SKYE_HMX_RECORD_LATE", False]()
"""Whether to load column tile `j`'s requant record AFTER its stores, not before.

THE MUTATION, not a tuning knob: on it the arm is WRONG BY CONSTRUCTION, and a
device leg that scores it against the twin must FAIL. `matmul_bt_stripe_hmx_step`
has carried the `record_late` parameter since the arm landed and the host twin
fires it (`test_hmx_int8.mojo`), but nothing could fire it on silicon, so the
device gate had no control and a PASS could not be told from a comparison that
cannot fail.

A build define rather than a dispatch argument because `record_late` is a
comptime parameter and the `matmul` selector's ABI has no scalar to spare -- the
same constraint that sends the artefact's own quantization down inside `B`
(`_hmx_arm_pack_act`). Threading it at runtime instead would duplicate the step
loop and put a branch in the array thread's body, which is a real cost paid on
the shipping arm to serve a control. So the control is a second signed tuple, and
`MATMUL_WIT_ARM` reports which one the device actually ran.

`build.sh` refuses it without `HMX_INT8=1`: the parameter is inert on the fp16
arm, so that pairing builds a "mutant" identical to its own control -- the
degenerate control this exists to prevent."""

comptime HMX_INT8_DTYPE = DType.uint8
"""The operand type of every W8A8 crouton panel.

BOTH operands, and the output readout. A uint8 activation and an int8 weight are
the same bytes at the same offsets and differ only in which instruction reads
them (`activation.ub` vs `weight.b`), so signedness belongs to the packer and
the model, never to the layout — which is what lets one `HmxStripeArena` and one
stripe body serve both arms."""


comptime HMX_INT8_CONVERT_DIV = 512
"""The integer convert's divisor: `out = clamp(acc * scale_fp16 / 512 + base)`.

Not a tunable and not an fp16 constant re-read. `cdsp_peak`'s int4 arm reproduces
it exactly on v81 silicon — uniform operands giving `acc = 2048` read back as
uniform `4` (scope §2 H1) — which is the only quantitative confirmation of the
rule that exists outside the vendor guide."""

comptime HMX_INT8_IDENTITY_SCALE = Scalar[DType.float16](HMX_INT8_CONVERT_DIV)
"""The integer requant identity, 512.0 (`0x6000`) -- NOT fp16's 1.0 (`0x3c00`).

Rung 2 of the ladder programs exactly this and expects the RAW accumulator back.
It is spelled here so that no caller ever reaches for `hmx_requant_word`'s fp16
identity by habit."""

comptime HMX_INT8_MAX = 127
"""Symmetric int8 range for weights: `-127..127`, not `-128`.

`-128` has no positive partner, so a symmetric scale that admitted it would be
asymmetric in effect. Every reference quantizer makes the same choice."""

comptime HMX_INT8_LOAD_RANGE: UInt32 = UInt32(
    HmxTileBytes[HMX_INT8_DTYPE] - 1
)
"""The activation and weight load range: an INCLUSIVE BYTE EXTENT of one tile.

RESTORED WITH THE PLAIN FORM, and the two go together. Order 122's stream --
the only integer stream ever measured exact on this silicon -- passes
`tile_bytes - 1` on both loads (`hmx_int8_shim.c:380`, derived there from a
shipping backend's fp16 kernels which pass 2047 for a 2048-byte tile). PR #352
replaced this with the vendor bare-metal test's PACKED CONTROL FIELDS (`0xE3`)
at the same time as it replaced the store form, and the contract doc records
that pairing as ASSUMED: `hmx.c`'s fields describe a **4-deep, 2048-byte**
operand set, while ours is 32-deep and 1024 B, and they were never re-derived
for our geometry (`INT8_DATAFLOW_CONTRACT.md` §1.c).

Under the plain form the question disappears rather than being answered: the
extent is a property of the tile, so it is COMPUTED from the tile size instead
of written down. A zero here asks for a 1-byte extent, nothing streams into the
array, and the store then writes a cleared accumulator -- which reads as "the
MAC ran and produced zeros" rather than as the misconfiguration it is.

If a later lane restores `:cm`, this constant does NOT carry over: `:cm` takes
packed fields, and probe H exists to derive them for a 32-deep operand set."""

comptime HMX_INT8_STORE_SCALE: UInt32 = 0
"""The plain store's second operand.

It is NOT the load range. The reference names the matching store macro's second
operand `scale_reg` and every call site passes 0 (`hmx_int8_shim.c:94`,
`:400-408`), which is what order 122 ran."""

comptime HMX_INT8_HEADER_WORDS = 4
"""Words of the artefact's per-tensor quantization header.

`(act_scale f32, act_zp u32, out_scale f32, out_zp u32)`. Per TENSOR because v1
is static per-tensor activation quantization; a dynamic scheme would make the
first two per dispatch and is the named follow-up above."""


comptime HmxWordRow = TileTensor[
    DType.uint32, type_of(row_major(Coord(0, 0))), origin=MutUntrackedOrigin
]
"""A `[rows, cols]` word view of the artefact's non-crouton regions.

The croutons are read through `CroutonPanelOf[uint8, ...]` like every other
panel; this exists because the 256-byte records and the header are not croutons
and describing them as if they were is how a layout starts lying. Both regions
follow a whole number of croutons, so both inherit word alignment."""


@fieldwise_init
struct HmxInt8Qp(Copyable, ImplicitlyCopyable, Movable):
    """The per-tensor half of a W8A8 configuration: what the record cannot hold.

    The per-CHANNEL half (weight scale, zero-point fold, output zero point) is
    already folded into the 256-byte records at repack. What survives to the
    device is exactly what the feeders need and the record has no room for: the
    scale and zero point that turn a fp16 activation into a uint8 one, and the
    pair that turns the uint8 readout back into fp16.

    Calibration is OUT OF SCOPE for v1 and deliberately so: these arrive from
    the offline select file, and a weight whose row does not carry them is
    refused by the repacker rather than given a guessed range.
    """

    var act_scale: Scalar[DType.float32]
    """Activation quantum. `A ~ (Aq - act_zp) * act_scale`."""

    var act_zp: Int
    """Activation zero point, `0..255`. Nonzero because `.ub` is UNSIGNED and a
    DiT activation is not: the ISA has no signed-byte activation form at all
    (scope §1.2), so sign has to live in a zero point."""

    var out_scale: Scalar[DType.float32]
    """Output quantum. `C ~ (out_u8 - out_zp) * out_scale`."""

    var out_zp: Int
    """Output zero point, `0..255`. Also nonzero for a signed output, and folded
    into the int32 `effective_bias` rather than the fp16 baseline (module
    docstring)."""


@always_inline
def hmx_int8_identity_qp() -> HmxInt8Qp:
    """The unit configuration: no scaling, no zero point.

    What a rung programs when it wants the RAW accumulator, and what a caller
    that has no calibration must pass explicitly rather than get by default.
    """
    return HmxInt8Qp(Scalar[DType.float32](1), 0, Scalar[DType.float32](1), 0)


# ===--------------------------------------------------------------------------===
# The 256-byte requant record.
#
# `HmxRequantRecord` is `[2, 32]` uint32 = 256 bytes, which IS the integer
# record's shape: row 0 carries the 32 `(fp16 scale, fp16 baseline)` words and
# row 1 the 32 int32 effective biases. So the integer arm needs a different
# ENCODING, not a different type -- the fp16 arm simply leaves row 1 zero. That
# is the sibling-function split the scope asks for (§4.3 slice 2) rather than a
# flag on `hmx_requant_word`, which packs an fp16 BIAS in the high halfword and
# would read a valid integer record as nonsense.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_int8_record_word(
    scale: Scalar[DType.float16], baseline: Scalar[DType.float16]
) -> Scalar[DType.uint32]:
    """Packs one output channel's `(scale, baseline)`: LOW scale, HIGH baseline.

    The same halfword order as the fp16 encoding, which is settled
    (`hmx_requant_word`), and a different MEANING for both fields: the scale
    carries the `512 *` factor the convert divides out, and the baseline is an
    output offset rather than an fp16 bias. v1 always passes `baseline = 0`; see
    the module docstring for why the guide's `output_zp << 7` is not used.
    """
    var s = bitcast[DType.uint16, 1](scale).cast[DType.uint32]()
    var b = bitcast[DType.uint16, 1](baseline).cast[DType.uint32]()
    return s | (b << 16)


@always_inline
def hmx_int8_record_scale(
    qp: HmxInt8Qp, weight_scale: Scalar[DType.float32]
) -> Scalar[DType.float16]:
    """The fp16 scale for one output channel: `512 * s_a * s_w[c] / s_out`.

    `03-prepare-the-data.md:176`. The `512` is the convert's divisor arriving
    pre-multiplied, so the identity configuration lands on 512.0 rather than on
    1.0 -- the constant whose absence made every earlier probe unfalsifiable.
    """
    return (
        Scalar[DType.float32](HMX_INT8_CONVERT_DIV)
        * qp.act_scale
        * weight_scale
        / qp.out_scale
    ).cast[DType.float16]()


@always_inline
def hmx_int8_record_bias(
    qp: HmxInt8Qp, weight_scale: Scalar[DType.float32], col_sum: Int
) -> Int32:
    """The int32 effective bias for one output channel.

    Two folds, both static, both host-side, and both in ACCUMULATOR units:

    - `-act_zp * sum_k Wq[k, c]` removes the activation zero point's
      contribution (`03-prepare-the-data.md:178`). Without it every output
      carries `act_zp * sum_k W`, which is not a small term.
    - `+ out_zp * s_out / (s_a * s_w[c])` places the output zero point. It goes
      here rather than in the fp16 baseline because the accumulator's units are
      unambiguous and the baseline's are not (module docstring).

    Args:
        qp: The per-tensor configuration.
        weight_scale: This channel's weight quantum.
        col_sum: `sum_k Wq[k, c]` over the WHOLE reduction, which is why this is
            computed once at repack and never at dispatch.
    """
    var zp_term = (
        Scalar[DType.float32](qp.out_zp)
        * qp.out_scale
        / (qp.act_scale * weight_scale)
    )
    return Int32(-qp.act_zp * col_sum) + Int32(round(zp_term))


@always_inline
def hmx_int8_requant(
    acc: Int32, word: Scalar[DType.uint32], bias: Int32
) -> Scalar[DType.uint8]:
    """The integer convert, in portable arithmetic: what the `:after` store does.

    `out = clamp_u8(acc_eff * scale / 512 + baseline)` with `acc_eff = acc +
    bias`. Both the composition and the int32 preset are now MEASURED, not
    inferred: decoding `qemu-hexagon-testing`'s silicon-captured `hmx_ref.h`
    reproduces its 32 published output channels exactly under this expression
    (`INT8_DATAFLOW_CONTRACT.md` §2.2), including both clamp ends.

    IT TRUNCATES. The same decode settles the rounding mode the scope doc left
    open, and it is not round-to-nearest: only `floor` reproduces the vendor
    reference, while round-to-nearest misses at channels 8, 10, 13 and 16. This
    was `round()` until that decode, which made the twin and the reference
    bytes the device leg is scored against wrong in the same direction -- the
    likely source of order 127's "22 disagreements by exactly one step".

    `floor` rather than truncate-toward-zero: the two differ only for a
    negative `v`, and every negative `v` clamps to 0 below either way, so this
    is the vendor-matching form with no behaviour difference at the boundary.

    NOTE for anyone removing the `roundevenf` shim (`qshim_qurt.c:2203`): this
    was not its only caller and it is still needed. `hmx_int8_quantize_act`
    (`:577`) runs on the DEVICE inside the activation packer and still rounds
    to nearest.
    """
    var scale = bitcast[DType.float16, 1](word.cast[DType.uint16]())
    var baseline = bitcast[DType.float16, 1]((word >> 16).cast[DType.uint16]())
    var v = (
        Scalar[DType.float32](acc) + Scalar[DType.float32](bias)
    ) * scale.cast[DType.float32]() / Scalar[DType.float32](
        HMX_INT8_CONVERT_DIV
    ) + baseline.cast[
        DType.float32
    ]()
    var r = floor(v)
    if r < 0:
        return Scalar[DType.uint8](0)
    if r > 255:
        return Scalar[DType.uint8](255)
    return Scalar[DType.uint8](Int(r))


@always_inline
def hmx_int8_record_bias_of[
    space: AddressSpace, //
](record: HmxRequantRecord[space], lane: Int) -> Int32:
    """Row 1 of the record, read back as the int32 it is."""
    return bitcast[DType.int32, 1](record.load[1](Coord(1, lane)))


@always_inline
def hmx_int8_program_record[
    space: AddressSpace, //
](record: HmxRequantRecord[space], scale: Scalar[DType.float16], bias: Int32):
    """Writes one uniform `(scale, bias)` across every output column.

    The uniform form, which is what a rung and the identity configuration want.
    Production records are per-CHANNEL and are built at repack, then copied in
    by `hmx_int8_load_record`; nothing on the device computes one.
    """
    comptime WORD_ALIGN = align_of[Scalar[DType.uint32]]()
    var word = hmx_int8_record_word(scale, Scalar[DType.float16](0))
    var bias_word = bitcast[DType.uint32, 1](bias)
    for lane in range(HMX_RECORD_LANES):
        record.store[1, alignment=WORD_ALIGN](Coord(0, lane), word)
        record.store[1, alignment=WORD_ALIGN](Coord(1, lane), bias_word)


# ===--------------------------------------------------------------------------===
# THE ARTEFACT: int8 croutons, per-column-tile records, and one header, inside a
# file of exactly the fp16 weight's LENGTH.
#
# That length is load-bearing and is M4.3's property carried forward: the staged
# file's byte count, the manifest, the sha256 gate, the census and the graph's
# own constant binding all still agree, and only the layout tag says what the
# bytes mean. An artefact that halved in size would move every one of them.
#
#   [0, N*K)                       int8 croutons, column-panel-major
#   [N*K, N*K + n_tiles*256)       one 256-byte requant record per column tile
#   [.., .. + 16)                  the per-tensor header
#   remainder                      zero
#
# The tail costs `N/32 * 256 + 16` bytes out of `N*K` free ones, so it fits for
# every geometry the array accepts -- and `hmx_int8_artefact_fits` is the
# refusal that keeps that a checked property rather than a comment.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_int8_records_offset(N: Int, K: Int) -> Int:
    """Byte offset of the record region: exactly the crouton region's size."""
    return N * K


@always_inline
def hmx_int8_header_offset(N: Int, K: Int) -> Int:
    """Byte offset of the per-tensor header, past the per-column-tile records.
    """
    return N * K + (N // HMX_TILE) * HMX_RECORD_BYTES


@always_inline
def hmx_int8_artefact_fits(N: Int, K: Int) -> Bool:
    """Whether the tail fits inside the fp16 weight's byte count.

    The refusal that keeps the length invariant true. A geometry failing this
    would need a longer file, which is the one thing this layout exists to
    avoid.
    """
    return hmx_int8_header_offset(N, K) + HMX_INT8_HEADER_WORDS * 4 <= 2 * N * K


@always_inline
def hmx_int8_words(
    B: HmxRows2D, word0: Int, rows: Int, cols: Int
) -> HmxWordRow:
    """A `[rows, cols]` word view of the artefact starting at word `word0`.

    `B` is the kernel's own `[N, K]` fp16 view, unchanged by the repack, so this
    re-originates its pointer exactly as `hmx_native_weight_panel` does. It is
    ONE seam, and everything downstream of it is a coordinate -- which is why
    neither the repacker nor the loader below adds a byte offset to anything.
    """
    return HmxWordRow(
        ptr=UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin](
            unsafe_from_address=Int(B.ptr_at_offset(Coord(0, 0)))
        )
        + word0,
        layout=row_major(Coord(rows, cols)),
    )


@always_inline
def hmx_int8_artefact_record(
    B: HmxRows2D, col_tile: Int
) -> HmxRequantRecord[AddressSpace.GENERIC]:
    """The baked `[2, 32]` record for output column tile `col_tile`.

    The single place the record region is located, so the repacker that writes
    it and the loader that reads it cannot be given different offsets. Its type
    is the arena's own record type, which is what makes the copy in
    `hmx_int8_load_record` a coordinate-for-coordinate move.
    """
    var N = B.layout.shape[0]()
    var K = B.layout.shape[1]()
    var word0 = (
        hmx_int8_records_offset(N, K) + col_tile * HMX_RECORD_BYTES
    ) // 4
    return HmxRequantRecord[AddressSpace.GENERIC](
        ptr=UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin](
            unsafe_from_address=Int(B.ptr_at_offset(Coord(0, 0)))
        )
        + word0,
        layout=row_major(Coord(HMX_RECORD_VECTORS, HMX_RECORD_LANES)),
    )


@always_inline
def hmx_int8_weight_panel[
    dtype: DType = HMX_INT8_DTYPE
](B: HmxRows2D, col_tile: Int, tile0: Int, tiles: Int) -> CroutonPanelOf[
    dtype, AddressSpace.GENERIC
]:
    """`tiles` int8 croutons of column panel `col_tile`, from crouton `tile0`.

    The int8 read side of the column-panel-major order, and the same crouton
    ARITHMETIC as the fp16 one -- `(ct * k_tiles + kt)` -- over a crouton that
    is 1024 bytes instead of 2048. The extents come off `B`'s own layout so the
    two orders cannot be given different geometries by accident.
    """
    var k_tiles = hmx_k_tiles(B.layout.shape[1]())
    var croutons = (B.layout.shape[0]() // HMX_TILE) * k_tiles
    return hmx_subpanel(
        hmx_panel[AddressSpace.GENERIC, dtype](
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(B.ptr_at_offset(Coord(0, 0)))
            ),
            croutons,
        ),
        col_tile * k_tiles + tile0,
        tiles,
    )


@always_inline
def hmx_int8_read_qp(B: HmxRows2D) -> HmxInt8Qp:
    """Reads the artefact's per-tensor header.

    On the device rather than through a kernel argument, because the header is a
    property of the WEIGHT and travels with it: a registry that mixes int8 and
    fp16 families (which slice 5 makes a supported state) would otherwise need a
    per-dispatch scalar the frozen selector ABI has no room for.
    """
    var w = hmx_int8_words(
        B,
        hmx_int8_header_offset(B.layout.shape[0](), B.layout.shape[1]()) // 4,
        1,
        HMX_INT8_HEADER_WORDS,
    )
    return HmxInt8Qp(
        bitcast[DType.float32, 1](w.load[1](Coord(0, 0))),
        Int(w.load[1](Coord(0, 1))),
        bitcast[DType.float32, 1](w.load[1](Coord(0, 2))),
        Int(w.load[1](Coord(0, 3))),
    )


@always_inline
def hmx_int8_write_qp(B: HmxRows2D, qp: HmxInt8Qp):
    """Writes the artefact's per-tensor header: the repacker's side of the seam.
    """
    var w = hmx_int8_words(
        B,
        hmx_int8_header_offset(B.layout.shape[0](), B.layout.shape[1]()) // 4,
        1,
        HMX_INT8_HEADER_WORDS,
    )
    w.store[1](Coord(0, 0), bitcast[DType.uint32, 1](qp.act_scale))
    w.store[1](Coord(0, 1), Scalar[DType.uint32](qp.act_zp))
    w.store[1](Coord(0, 2), bitcast[DType.uint32, 1](qp.out_scale))
    w.store[1](Coord(0, 3), Scalar[DType.uint32](qp.out_zp))


@always_inline
def hmx_int8_load_record[
    space: AddressSpace, //
](record: HmxRequantRecord[space], B: HmxRows2D, col_tile: Int):
    """Copies column tile `col_tile`'s baked record into the arena's record.

    THE ORDER-118 RULE, and the reason this is a named call rather than a line
    inside the MMA: the requant file must hold this tile's record before ANY
    store reads through it, and a store issued against an unprogrammed file
    reads back zeros from a MAC whose operands were streaming correctly --
    observed on silicon for the fp16 arm, and the shape of the failure the int8
    probe spent three device windows on. The stripe calls this once per step, on
    the array thread, ahead of its first `hmx_int8_mma_panel`.

    Nothing is COMPUTED here. The 32 scales and 32 int32 biases were folded at
    repack, which is what keeps quantization parameters out of the hot loop
    (`03-prepare-the-data.md:111-117`).
    """
    comptime WORD_ALIGN = align_of[Scalar[DType.uint32]]()
    var src = hmx_int8_artefact_record(B, col_tile)
    for v in range(HMX_RECORD_VECTORS):
        for lane in range(HMX_RECORD_LANES):
            record.store[1, alignment=WORD_ALIGN](
                Coord(v, lane), src.load[1](Coord(v, lane))
            )


# ===--------------------------------------------------------------------------===
# The packers. Each is its fp16 sibling's axis rule with a quantization step
# fused in, which is the only reason they are separate functions: the geometry
# is `CroutonPanelOf[uint8, ...]`'s and is therefore the packers' own oracles
# unchanged (scope §2 H4 -- the layout expression already described the int8
# tile before any of this was written).
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_int8_quantize_act(
    x: Scalar[DType.float16], qp: HmxInt8Qp
) -> Scalar[DType.uint8]:
    """One activation element, fp16 -> uint8. Saturating, not wrapping."""
    var v = round(x.cast[DType.float32]() / qp.act_scale) + Scalar[
        DType.float32
    ](qp.act_zp)
    if v < 0:
        return Scalar[DType.uint8](0)
    if v > 255:
        return Scalar[DType.uint8](255)
    return Scalar[DType.uint8](Int(v))


@always_inline
def hmx_int8_quantize_weight(
    x: Scalar[DType.float16], weight_scale: Scalar[DType.float32]
) -> Scalar[DType.uint8]:
    """One weight element, fp16 -> int8, returned as its byte.

    Symmetric over `-127..127`. The byte is what the layout holds; `weight.b`
    is what reads it as signed.
    """
    var v = round(x.cast[DType.float32]() / weight_scale)
    if v < -HMX_INT8_MAX:
        v = Scalar[DType.float32](-HMX_INT8_MAX)
    if v > HMX_INT8_MAX:
        v = Scalar[DType.float32](HMX_INT8_MAX)
    return bitcast[DType.uint8, 1](Scalar[DType.int8](Int(v)))


@always_inline
def hmx_int8_pack_act[
    dtype: DType, space: AddressSpace, //
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
    tiles: Int,
    qp: HmxInt8Qp,
):
    """Stages one activation panel, with the quantization fused in.

    THE ROW IS IN THE SLAB POSITION, and order 146 measured that this is
    correct. A depth-major variant was carried here between orders 138 and 146
    on the reasoning that both operands must present the contraction axis in the
    same byte position; the read-off refuted it. Through the plain stream this
    layout returns exactly the right output row (probe B) and output column
    (probe D) -- 0 of 1024 differing, the first exactly-correct device readouts
    in the arc -- and probe A names the paired weight row as `kp = k0`. The
    layout is not the fault.

    What IS the fault is an operand FEED, not a byte rule: within a crouton
    pair, both weight croutons are fed from the LOW activation crouton and the
    upper one is never read (`lane-evidence/146`). That is a pack-interleave or
    issue-order question and it is specified-but-parked, not fixed here.

    Fused rather than a quantize-then-pack pair because the pack already walks    Fused rather than a quantize-then-pack pair because the pack already walks
    every element once and a separate uint8 staging buffer would be a second
    VTCM region the arena has no room for at production `k_tiles`.
    """
    comptime ALIGN = align_of[Scalar[dtype]]()
    comptime SPLIT = HmxSplit[dtype]
    for t in range(tiles):
        for slab in range(HMX_TILE):
            for lane in range(HMX_TILE):
                dst.store[1, alignment=ALIGN](
                    Coord(t, slab // SPLIT, lane, slab % SPLIT),
                    hmx_int8_quantize_act(
                        src.load[1](
                            Coord(row0 + slab, col0 + t * HMX_TILE + lane)
                        ),
                        qp,
                    ).cast[dtype](),
                )


@always_inline
def hmx_int8_unpack_out[
    dtype: DType, space: AddressSpace, //
](
    dst: HmxRows2D,
    src: CroutonPanelOf[dtype, space],
    row0: Int,
    col0: Int,
    qp: HmxInt8Qp,
):
    """Scatters one requanted readout tile back to `C`, dequantizing to fp16.

    Slab `m`, lane `n`, the inverse of the pack -- and order 146 measured this
    correct through the plain store: probe D lights exactly output column `n0`
    at the offset this rule expects. (The `:cm` store writes one byte per
    128-byte slab at a lane stride of 1 instead of 4, which is a fault of that
    operand mode and not of this rule.)

    `(out_u8 - out_zp) * out_scale` fused in. One tile, because a readout is
    one tile; the fp16 sibling takes a count only because it is also the
    packers' inverse.
    """
    comptime ALIGN = align_of[Scalar[dtype]]()
    comptime SPLIT = HmxSplit[dtype]
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            var q = Int(
                src.load[1, alignment=ALIGN](
                    Coord(0, slab // SPLIT, lane, slab % SPLIT)
                )
            )
            var v = (
                Scalar[DType.float32](q) - Scalar[DType.float32](qp.out_zp)
            ) * qp.out_scale
            dst.store[1](
                Coord(row0 + slab, col0 + lane), v.cast[DType.float16]()
            )


# ===--------------------------------------------------------------------------===
# The MMA. Five instruction words, differing from the fp16 arm's in four of
# them, and taken from the two WORKING integer sequences (scope §1.4): the
# vendor-authored bare-metal test (`qemu-hexagon-testing`
# `standalone_systests/src/hmx.c:44-67`) and the RE'd v75 production kernel
# (`qcom_htp` `v73deep_conv1x1_kernel.inc:105-165`).
# ===--------------------------------------------------------------------------===


@always_inline
def _hmx_int8_mma_open[
    space: AddressSpace, //
](record: HmxRequantRecord[space]):
    """Clears the accumulators, then programs the requant file.

    THE OPPOSITE ORDER TO THE fp16 ARM, and taken rather than derived. Both
    working integer sequences clear first and load the record second
    (`hmx.c:44` then `:49`; the QNN kernel's `bias = mxmem2` at `:105-107` sits
    after its own clear), while the fp16 arm's proven order is record-then-clear
    (`_hmx_mma_open`). The invariant both honour, and the one order 118 named,
    is that the record is programmed before any STORE reads through it -- which
    is what `hmx_int8_load_record` at the top of a step guarantees for the
    per-channel contents.

    `mxclracc` has NO `.hf` suffix here. The fp16 spelling would clear a
    different accumulator mode.
    """
    inlined_assembly[
        "mxclracc",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()

    inlined_assembly[
        "bias = mxmem2($0)",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](_hmx_addr(record.ptr_at_offset(Coord(0, 0))))


@always_inline
def _hmx_int8_mma_stream[
    space: AddressSpace, dtype: DType, //
](operands: HmxOperands[space, dtype]):
    """Streams one `(uint8 activation, int8 weight)` pair into the accumulators.

    `activation.ub` and `weight.b`, co-issued in one packet exactly as the fp16
    pair is -- the pairing IS the multiply. BOTH ARE THE PLAIN FORMS, and both
    take an inclusive byte extent (`HMX_INT8_LOAD_RANGE`), which is order 122's
    configuration -- the only integer stream ever measured exact on this
    silicon (`hmx_int8_shim.c:380, 394-395`).

    `:cm` WAS ON THE ACTIVATION AND IS NOT ANY MORE. PR #352 put it there
    because the MC layer modelled the `:cm` store and not the plain one, and
    the substitution was recorded here as a difference in issue count. It is
    not: `:cm` is an operand mode carrying a different layout, and order 127
    measured the result as 1015/1024 wrong
    (`INT8_DATAFLOW_CONTRACT.md` §2.4). The missing instruction is now modelled
    (PR #373), so the arm no longer has to choose the mode it can assemble over
    the mode it has evidence for.

    ONE CROUTON PAIR PER PACKET, not a `:deep` stream, and this part IS a
    toolchain bound rather than a design choice.

    THIS TABLE IS A PROPERTY OF A BINARY, NOT OF THIS SOURCE, and reading it as
    the latter cost orders 132 and 133 outright. The store's row said `yes` from
    the moment PR #373 landed, and it was true of the TREE while being false of
    every `mojo` the device lane actually ran: that binary had been stale since
    2026-08-07, its MC layer predating the patch, so the store was silently
    absent from every int8 object built there (`mojo` prints an inline-asm error
    and exits 0). Two orders of instability were that missing instruction.
    `build.sh` now probe-assembles the store with the ACTIVE binary before it
    builds anything, so the rows below can be trusted to describe the compiler
    in use rather than the repository. Probing every form the ISA header lists,
    at `hexagonv81` with `+hmx`:

    | form                                      | assembles |
    |-------------------------------------------|-----------|
    | `activation.ub = mxmem(Rs, Rt)`           | yes       |
    | `activation.ub = mxmem(Rs, Rt):cm`        | yes       |
    | `activation.ub = mxmem(Rs, Rt):deep`      | NO        |
    | `activation.ub = mxmem(Rs, Rt):deep:cm`   | NO        |
    | `weight.b = mxmem(Rs, Rt)`                | yes       |
    | `mxmem(Rs, Rt):after:cm:sat.ub = acc`     | yes       |
    | `mxmem(Rs, Rt):after:sat.ub = acc`        | yes (#373, and gated)|
    | `cvt.ub = acc(Rt)`                        | NO        |

    So the RE'd v75 production sequence (`:deep:cm`, then a two-step `cvt`
    readout) still cannot be assembled by this tree, and what CAN be is now the
    full plain stream. Successive load-mpy packets accumulate into the live
    accumulator -- `cdsp_peak`'s int4 arm issues 64 of them and reads back the
    sum (scope §2 H1) -- so the arithmetic is the same one `:deep` would
    produce, and only the issue count differs. At the array's measured 0.4%
    share of a matmul dispatch (order 096 leg B) that count is not what to
    spend a compiler change on; extending the MC layer to the `.ub:deep` forms
    is the named follow-up.

    A HAZARD THIS TABLE ONCE HID: under an unpatched toolchain an unassemblable
    inline-asm form prints an error, exits 0, and writes the object with the
    instruction SILENTLY MISSING (PR #373). That is how a "yes" row could be
    wrong and the build still succeed, and it is why `build.sh` counts raw
    instruction words rather than trusting the compile.
    """
    for t in range(operands.tiles):
        inlined_assembly[
            (
                "{\n"
                "    activation.ub = mxmem($0, $2)\n"
                "    weight.b = mxmem($1, $3)\n"
                "}"
            ),
            NoneType,
            constraints="r,r,r,r,~{memory}",
            has_side_effect=True,
        ](
            _hmx_addr(operands.act.ptr_at_offset(Coord(t, 0, 0, 0))),
            _hmx_addr(operands.wgt.ptr_at_offset(Coord(t, 0, 0, 0))),
            HMX_INT8_LOAD_RANGE,
            HMX_INT8_LOAD_RANGE,
        )


@always_inline
def _hmx_int8_mma_store[
    space: AddressSpace, dtype: DType, //
](dst: CroutonPanelOf[dtype, space]):
    """Reads the accumulators back through the requant file as saturated uint8.

    `:after`, NOT `:before`. The parked probe switched to `:before` to explain
    zeros that were really the record's (scope §2 H2), and the switch added a
    second guaranteed zero: `cdsp_peak`'s int4 arm issues both stores from one
    MAC and its `:before` half decodes to 0 while its `:after` half decodes to
    the predicted 4. Every working reference reads with `:after`.

    `:sat.ub` is the saturating byte convert, which is what makes the record's
    scale the thing that has to be right: a scale too small crushes the tile to
    zero and a scale too large flattens it to 255, and neither faults.

    PLAIN `:after:sat.ub`, NOT `:cm` -- restored. PR #352 issued the `:cm`
    store because the `+hmx` MC layer could not assemble this one, and recorded
    the substitution as a difference in issue count. It is not: `:cm` is an
    operand MODE, and the vendor's own `:cm` reference puts N contiguous in the
    output and K contiguous in the activation where our packers write M
    contiguous in both (`INT8_DATAFLOW_CONTRACT.md` §2.2-2.4). That mismatch is
    what order 127 measured as 1015/1024 wrong. Its `:cm` output region is also
    2048 B against this tile's 1024, so the substituted store was a probable
    buffer overrun on top of a wrong answer.

    This form is what order 122 ran and measured exact on silicon, and THESE
    PACKERS ARE THE ONES IT RAN WITH -- which is why this change restores the
    instruction and deliberately leaves `hmx_int8_pack_act` and
    `hmx_int8_unpack_out` alone. Rewriting them to the `:cm` geometry is the
    other branch of §2.4 and must not be taken before a probe says what that
    geometry is.

    The store's second operand is `scale_reg` and is 0, not a byte extent; the
    loads' is the inclusive tile extent (`HMX_INT8_LOAD_RANGE`). Both come from
    122's stream as one configuration.
    """
    inlined_assembly[
        "mxmem($0, $1):after:sat.ub = acc",
        NoneType,
        constraints="r,r,~{memory}",
        has_side_effect=True,
    ](
        _hmx_addr(dst.ptr_at_offset(Coord(0, 0, 0, 0))),
        HMX_INT8_STORE_SCALE,
    )


@always_inline
def _hmx_int8_mma_model[
    space: AddressSpace, dtype: DType, //
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """What `_hmx_int8_mma_*` is believed to compute, in portable arithmetic.

    A model of the DATA FLOW and of the requant, and that is the whole of what a
    twin built on it can claim. It reads the staged bytes in the ARRAY's own
    slab and lane coordinates, so a packer whose axis role is transposed fails
    here; it accumulates in Int32, which is the integer datapath's published
    width and not an assumption; and it requants through the same record the
    `bias = mxmem2` load consumes, so the `/512` arithmetic is scored rather
    than described.

    What it cannot model: the instruction stream, `:cm`, the chunk boundary, and
    the hardware's rounding mode. Those are rungs 3, 5 and 2.
    """
    comptime SPLIT = HmxSplit[dtype]
    for m in range(HMX_TILE):
        for n in range(HMX_TILE):
            var acc = Int32(0)
            for t in range(k_tiles):
                for d in range(HMX_TILE):
                    # `activation.ub` reads the byte unsigned and `weight.b`
                    # reads it signed. The sign is recovered arithmetically
                    # rather than by a bitcast so the model stays valid for any
                    # panel type the arm parameter can take.
                    var a = Int32(
                        Int(
                            scratch.act.load[1](
                                Coord(t, m // SPLIT, d, m % SPLIT)
                            )
                        )
                    )
                    var w = Int32(
                        Int(
                            scratch.wgt.load[1](
                                Coord(t, d // SPLIT, n, d % SPLIT)
                            )
                        )
                    )
                    if w > HMX_INT8_MAX:
                        w -= 256
                    acc += a * w
            scratch.out.store[1](
                Coord(0, m // SPLIT, n, m % SPLIT),
                hmx_int8_requant(
                    acc,
                    scratch.record.load[1](Coord(0, n)),
                    hmx_int8_record_bias_of(scratch.record, n),
                ).cast[dtype](),
            )


@always_inline
def hmx_int8_mma_panel[
    space: AddressSpace, dtype: DType, //, emit_hmx: Bool = HMX_EMIT_MMA
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """Multiplies the staged byte panels into `scratch.out` on the array.

    Does NOT program the record, which is the one place this differs in
    RESPONSIBILITY from `hmx_mma_panel`: the fp16 arm's record is the identity
    and is cheap to rewrite per tile, while this one is per output channel and
    is copied in once per step by `hmx_int8_load_record`. A panel that
    programmed it would either need the artefact (which it is not given) or
    would overwrite the per-channel contents with a uniform record -- the second
    of which is exactly the mutation the twin's control asserts must fail.

    Parameters:
        emit_hmx: Whether to issue the instructions. Defaults to `HMX_EMIT_MMA`,
            which is OFF, so a host twin gets the model without asking and a
            device build must ask.

    Args:
        scratch: The staged byte panels, readout and requant record.
        k_tiles: Crouton tiles spanning the reduction.
    """
    comptime if emit_hmx:
        _hmx_int8_mma_open(scratch.record)
        _hmx_int8_mma_stream(
            HmxOperands[space, dtype](scratch.act, scratch.wgt, k_tiles)
        )
        _hmx_int8_mma_store(scratch.out)
    else:
        _hmx_int8_mma_model(scratch, k_tiles)


# ===--------------------------------------------------------------------------===
# The offline repack: the same transform the device would have done, run once.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_int8_weight_scale(
    src: HmxRows2D, channel: Int, K: Int
) -> Scalar[DType.float32]:
    """The symmetric quantum for one output channel: `max_k |W[c, k]| / 127`.

    Per OUTPUT CHANNEL over the WHOLE K, which is the only granularity the
    record can express (scope §1.3). A channel that is entirely zero would give
    a zero quantum and a division by it, so it gets 1.0 -- its quantized values
    are all zero either way, and the alternative is a NaN scale baked into an
    artefact.
    """
    var peak = Scalar[DType.float32](0)
    for k in range(K):
        var v = abs(src.load[1](Coord(channel, k)).cast[DType.float32]())
        if v > peak:
            peak = v
    if peak == 0:
        return Scalar[DType.float32](1)
    return peak / Scalar[DType.float32](HMX_INT8_MAX)


def hmx_int8_repack_weight_cpm(
    dst: HmxRows2D, src: HmxRows2D, N: Int, K: Int, qp: HmxInt8Qp
):
    """Builds a whole W8A8 artefact from an `[N, K]` fp16 weight.

    THE OFFLINE HALF, and the sibling of `hmx_repack_weight_cpm`. Writes all
    three regions of the artefact into `dst`, which the caller supplies as the
    fp16 weight's own `[N, K]` view because the file's LENGTH does not change:

    1. the int8 croutons, column-panel-major, quantized per output channel;
    2. one 256-byte record per output column tile, with the channel scale, the
       activation zero-point fold and the output zero point already folded in;
    3. the per-tensor header.

    Everything past the header is left as the caller staged it; the tool zeroes
    the buffer first so the tail is deterministic and the sha256 is stable.

    Args:
        dst: The artefact, viewed as the same `[N, K]` fp16 extent.
        src: The row-major fp16 weight the registry staged.
        N: Output columns of the matmul this weight serves.
        K: The reduction extent.
        qp: The per-tensor activation and output quantization, from calibration.
    """
    comptime SPLIT = HmxSplit[HMX_INT8_DTYPE]
    var k_tiles = hmx_k_tiles(K)
    var n_tiles = N // HMX_TILE
    var art = hmx_int8_weight_panel(dst, 0, 0, n_tiles * k_tiles)

    for ct in range(n_tiles):
        var panel = hmx_subpanel(art, ct * k_tiles, k_tiles)
        var record = hmx_int8_artefact_record(dst, ct)
        for lane in range(HMX_TILE):
            var channel = ct * HMX_TILE + lane
            var wscale = hmx_int8_weight_scale(src, channel, K)
            var col_sum = 0
            for kt in range(k_tiles):
                for slab in range(HMX_TILE):
                    # The weight rule: slab is the depth axis and lane is the
                    # output channel, so this is `hmx_pack_panel[True]`'s
                    # transposing gather with the quantization fused in.
                    var q = hmx_int8_quantize_weight(
                        src.load[1](Coord(channel, kt * HMX_TILE + slab)),
                        wscale,
                    )
                    col_sum += Int(bitcast[DType.int8, 1](q))
                    panel.store[1](
                        Coord(kt, slab // SPLIT, lane, slab % SPLIT), q
                    )
            record.store[1](
                Coord(0, lane),
                hmx_int8_record_word(
                    hmx_int8_record_scale(qp, wscale),
                    Scalar[DType.float16](0),
                ),
            )
            record.store[1](
                Coord(1, lane),
                bitcast[DType.uint32, 1](
                    hmx_int8_record_bias(qp, wscale, col_sum)
                ),
            )

    hmx_int8_write_qp(dst, qp)
