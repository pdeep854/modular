from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor, stack_allocation
from std.ffi import external_call
from std.math import align_up, ceildiv, exp, min
from std.memory import AddressSpace, bitcast
from std.sys import (
    align_of,
    get_defined_bool,
    get_defined_int,
    inlined_assembly,
    size_of,
)
from std.sys.intrinsics import llvm_intrinsic
from std.utils import StaticTuple
from std.utils.numerics import min_or_neg_inf

# The matrix-array matmul, the first kernel this file stopped carrying a folded
# copy of. Every kernel here was once duplicated from its authoring module
# because the device build staged this file alone; the HMX path was imported
# first because a hand-folded copy of a crouton layout is exactly the drift the
# layout type exists to prevent, and because the packers' correctness is proved
# by a host twin that runs the imported source.
#
# That reasoning generalized, and the megakernel extraction (slices 3-4) applied
# it: the layer norm (#331), rms norm, SiLU/SwiGLU, modulation, softmax, the
# group norm, the sincos tables and RoPE are now imported too, each verified by
# `kernel_asm_compare.sh` to compile to the same vector work. One folded copy is
# left, the attention family: its source lives beside its twin in
# `test/hexagon_ondevice/` and exports under the same `mojo_*` names this file
# does, so joining the package needs those renamed first, the way RoPE's were.
# `build.sh` stages the whole package (#333), so an extraction needs no staging
# edit.
from hexagon_ondevice.hmx_annihilation_bench import (
    ANNI_ARM_BASELINE,
    ANNI_ARM_NATIVE,
    ANNI_DIM,
    ANNI_FF_N,
    ANNI_HIDDEN,
    ANNI_M,
    ANNI_MARKER_DONE,
    ANNI_MARKER_REFUSED,
    ANNI_SCRATCH_BYTES,
    ANNI_STAGE_FALSIFY,
    ANNI_STAGE_FF,
    ANNI_STAGE_FFO,
    ANNI_STAGE_SWIGLU,
    ANNI_VERDICT_UNSET,
    ANNI_WIT_ARM,
    ANNI_WIT_MARKER,
    ANNI_WIT_STAGE,
    ANNI_WIT_VERDICT,
    annihilation_a_mod,
    annihilation_c_out,
    annihilation_ff_out,
    annihilation_ff_panel,
    annihilation_matmul_step,
    annihilation_mlp_out,
    annihilation_mlp_panel,
    annihilation_shim_verdict,
    annihilation_stage_prepare,
    annihilation_swiglu_thread,
    annihilation_w_ff,
    annihilation_w_ffo,
)
from hexagon_ondevice.hmx_crouton import (
    HMX_ACT_PREFETCH,
    HMX_DTYPE,
    HMX_PHASE_BARRIER,
    HMX_PHASE_OPEN,
    HMX_PHASE_OTHER,
    HMX_SPLIT,
    HMX_TILE,
    HMX_TILE_BYTES,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_UNPACK_VEC,
    CroutonPanelOf,
    HmxOperands,
    HmxRows2D,
    HmxScratch,
    hmx_mma_chain,
    hmx_pack_panel,
    hmx_panel,
    hmx_phase_mark,
    hmx_phase_open,
    hmx_port_probe_bytes,
    hmx_port_probe_chain,
    hmx_port_probe_fold,
    hmx_port_probe_issues,
    hmx_port_probe_min_tiles,
    hmx_program_identity_record,
    hmx_scratch_bytes,
    hmx_scratch_in,
    hmx_guard_damage,
    hmx_guard_poison,
    hmx_stripe_arena_in,
)

# Attention on the array (P5 slices 2-4), imported for the reason the crouton
# layout is: the residency, the axis rules, the softmax seam and now the
# feeder/array schedule have a host twin, and that twin only proves the shipped
# arm if the shipped arm IS the twin's source.
from hexagon_ondevice.hmx_attention import (
    ATTN_BC,
    ATTN_BR,
    ATTN_STAT_L,
    ATTN_STAT_M,
    HMX_ATTN,
    AttnSoftmax,
    AttnStats,
    HmxAttnArena,
    HmxAttnPlan,
    hmx_attn_accepts,
    hmx_attn_acc_probe,
    hmx_attn_arena_bytes,
    hmx_attn_arena_in,
    hmx_attn_normalise,
    hmx_attn_pack_k,
    hmx_attn_pack_q,
    hmx_attn_pack_v,
    hmx_attn_plan,
    hmx_attn_probe_fits,
    hmx_attn_probs_from_rows,
    hmx_attn_program_records,
    hmx_attn_pv_block,
    hmx_attn_qk_block,
    hmx_attn_reset_diag,
    hmx_attn_sched,
    hmx_attn_scores_to_rows,
    hmx_attn_set_diag,
    hmx_attn_softmax_rows,
    hmx_attn_step,
    hmx_attn_step_masked,
    hmx_attn_store_o,
    hmx_attn_zero_panel,
)
from hexagon_ondevice.hmx_conv_act import HmxConvGeom
from hexagon_ondevice.hmx_int8 import hmx_int8_artefact_record

# The standalone HMX bench (re-sign window item 8): pass bodies, fills,
# accounting and refusal floors, imported for the reason the crouton layout
# is — the host gate runs these exact functions modelled, and a hand-folded
# copy here would be the drift the module exists to prevent. This import is
# what makes the module reachable from the shipped object, which is why the
# selector row it serves lands only inside a re-sign window.
from hexagon_ondevice.hmx_standalone_bench import (
    BENCH_ACT_SALT,
    BENCH_ARM_ISSUED,
    BENCH_FORM_DEEP,
    BENCH_FORM_PLAIN,
    BENCH_MODE_FP16_INSTRIPE,
    BENCH_MODE_FP16_PLAIN,
    BENCH_MODE_INT8_PLAIN,
    BENCH_MODE_STORES_OUT,
    BENCH_MODE_UDMA_OUT,
    BENCH_STATUS_NOT_BUILT,
    BENCH_STATUS_NO_GRANT,
    BENCH_STATUS_NO_VTCM,
    BENCH_STATUS_WRONG_ARM,
    BENCH_WGT_SALT,
    bench_array_pass,
    bench_array_program_record,
    bench_array_stage,
    bench_array_stream,
    bench_case_status,
    bench_expected_arm,
    bench_fill_fp16_croutons,
    bench_instripe_ddr_bytes,
    bench_instripe_slot_bytes,
    bench_move_chunk_bytes,
    bench_readout_fold,
    bench_zero_readout,
)
from hexagon_ondevice.matmul_kernel import (
    HMX_EMIT_MMA,
    HMX_INT8,
    HMX_INT8_DTYPE,
    HMX_RECORD_LATE,
    MM_PATH_HMX,
    MMTile,
    HmxStripePlan,
    hmx_stripe_plan,
    hmx_stripe_roles,
    matmul_bt_hmx_accepts,
    matmul_bt_stripe_hmx_step,
)

# The step arena (megakernel slice 6). The two resolutions below used to align
# the grant and spend the alignment out of the budget separately and
# identically; one of them drifting is the order-084 race class in its quietest
# form, since neither site would look wrong on its own.
from hexagon_ondevice.step_arena import StepArena

# The no-affine layer norm, imported rather than folded (cold critique C2): its
# host twin only proves the shipped kernel if the shipped kernel IS the twin's
# source. `RowMajor2D` is aliased because this file already has its own tile
# names and the seam below wants the module's.
from hexagon_ondevice.layernorm_kernel import (
    LN_DTYPE,
    LN_DTYPE_BF,
    RowMajor2D as LnRowMajor2D,
    _layer_norm_entry,
)

# The rms norm, imported for the same reason. Its module is newer than the
# layer norm's — the kernel was born in this file and had no twin at all — so
# what the import closes here is not drift but the absence of one.
from hexagon_ondevice.rms_norm_kernel import (
    RMS_DTYPE,
    RMS_DTYPE_BF,
    RMSNorm2Pass,
    RowMajor2D as RmsRowMajor2D,
    _rms_norm_entry,
    _rms_norm_rope_entry,
    _rms_norm_witness,
    rms_row,
    staged_norm_bytes,
    staged_norm_slot_bytes,
)

# The FFN activations, imported for the same reason. `RowMajor2D` is aliased as
# the norms' are: this file has its own tile names and the seams below want the
# module's.
from hexagon_ondevice.silu_swiglu_kernel import (
    SILU_DTYPE,
    SILU_DTYPE_BF,
    RowMajor2D as SwRowMajor2D,
    _silu_entry,
    _swiglu_entry,
)

# The adaLN modulation and gated residual, imported for the same reason. The two
# tile builders come with them: which of the two a pointer needs is the whole
# broadcast contract, and a seam that picked wrong would be a silent stride bug.
from hexagon_ondevice.modulation_kernel import (
    MOD_DTYPE,
    MOD_DTYPE_BF,
    _ada_modulate_entry,
    _ada_modulate_ln_entry,
    _gate_residual_entry,
    _param_tile as _mod_param_tile,
    _rows_tile as _mod_rows_tile,
)

# The row-wise softmax, imported for the same reason. Only the body moves: the
# vector `exp` and the max reduction that sit in this file's softmax section are
# the ATTENTION kernel's and stay put (see that section's banner).
from hexagon_ondevice.softmax_kernel import (
    SOFTMAX_DTYPE,
    RowMajor2D as SmRowMajor2D,
    _softmax_entry,
)

# The affine group norm, imported for the same reason, and for the layer norm's
# specifically: its module took typed views through every signature below the
# seam (P4.1) while the copy shipped from here still computed `c * spatial` by
# hand, so `test_groupnorm_kernel.mojo` scored a body no device ever ran. No
# tile alias comes with it -- unlike the norms above, this seam hands the entry
# the caller's pointers and extents, because a group is described two ways
# (flat for the reductions, `(channels, spatial)` for the affine) and the module
# builds both from one set of extents rather than taking a tile it would have
# to re-describe.
from hexagon_ondevice.groupnorm_kernel import (
    GN_DTYPE,
    GN_DTYPE_BF,
    _group_norm_entry,
)

# The VAE decode's layout-round-trip fix (F4 in
# `.agentwork/flux-npu/VAE_CODE_CRITIQUE.md`): transpose `[R, N] -> [N, R]`
# with a per-output-row bias, run where the conv/silu data already lives
# instead of on the host after a round trip through rpcmem. New, not folded
# from anywhere -- there is no prior copy to have drifted from.
from hexagon_ondevice.transpose_bias_kernel import (
    TB_DTYPE,
    RowMajor2D as TbRowMajor2D,
    _transpose_bias_entry,
)

# The rotary embedding, imported for the reason the norms are: the module
# describes a head vector and a freqs row through the tile model, while the
# copy that shipped still reached them by adding `row * head_size` to a pointer.
from hexagon_ondevice.rope_kernel import (
    FREQ_DTYPE as ROPE_FREQ_DTYPE,
    IO_DTYPE as ROPE_IO_DTYPE,
    IO_DTYPE_BF as ROPE_IO_DTYPE_BF,
    rope_entry,
)

# The two scalar fp32 table builders of the denoise preamble. The exactness the
# module argues for is only a property of the source both sides compile, which a
# folded copy is precisely what breaks.
from hexagon_ondevice.sincos_tables import (
    rope_table_entry,
    sinusoid_embed_entry,
)

# Flash attention, the last family that was folded. The module owns both bodies
# and the head split; what stays below are the two exports the manifest binds,
# because the dense one is not only a body -- it also holds the HMX arms' grant
# and the launch's cycle bracket, neither of which a host build can have.
from hexagon_ondevice.attention_kernel import (
    ACCUM as MHA_ACCUM,
    BSHD as MhaBSHD,
    FA_BM as MHA_BM,
    FA_BN as MHA_BN,
    FA_HEAD_DIM as MHA_HEAD_DIM,
    IO_DTYPE as MHA_IO_DTYPE,
    MMA_W as MHA_SIMD_W,
    RowMajor2D as MhaBias2D,
    attention_thread_heads,
    flash_attention_heads,
    mha_masked_entry,
)

# The megastep walk, imported for the same reason: it has a host twin, and a
# hand-folded copy of a wire decoder is the drift a shared module prevents.
from hexagon_ondevice.megastep_walker import (
    MegastepSeam,
    SlotBases,
    SlotLengths,
    TableBases,
    TableBytes,
    WireBytes,
    megastep_walk,
)

comptime DTYPE = DType.float32

# An HVX vector register is 128 B wide (see `+hvx-length128b` in build.sh).
# Deriving the lane count from that is deliberate: `simd_width_of[DTYPE]()`
# without an explicit target reports the *host* width — 4 here — and silently
# emits near-scalar code.
comptime LANES = 128 // size_of[DTYPE]()

# Extent is a call-time argument, so the layout is runtime-shaped.
comptime Buffer = TileTensor[
    DTYPE, type_of(row_major(Coord(0))), origin=MutUnsafeAnyOrigin
]


@export
def mojo_vadd_f32(
    lhs: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    rhs: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    result: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    element_count: Int,
) abi("C"):
    """Adds two vectors elementwise.

    Raw pointers appear only because the skel calls this across the C ABI. They
    are wrapped immediately so all indexing goes through `TileTensor`.
    """
    var extent = row_major(Coord(element_count))
    var lhs_buffer = Buffer(ptr=lhs, layout=extent)
    var rhs_buffer = Buffer(ptr=rhs, layout=extent)
    var result_buffer = Buffer(ptr=result, layout=extent)

    var vectorized_end = element_count - (element_count % LANES)
    for i in range(0, vectorized_end, LANES):
        result_buffer.store[LANES](
            Coord(i),
            lhs_buffer.load[LANES](Coord(i)) + rhs_buffer.load[LANES](Coord(i)),
        )

    for i in range(vectorized_end, element_count):
        result_buffer.store[1](
            Coord(i),
            lhs_buffer.load[1](Coord(i)) + rhs_buffer.load[1](Coord(i)),
        )


@export
def mojo_vscale_offset_f32(
    src: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    result: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    scale: Scalar[DTYPE],
    offset: Scalar[DTYPE],
    element_count: Int,
) abi("C"):
    """Scales a vector and shifts it, elementwise.

    Deliberately shaped unlike `mojo_vadd_f32`: two buffers rather than three,
    and two floating-point scalars beside the extent. The dispatch it arrives
    through carries buffers and scalars separately, so a signature no interface
    definition was generated for still reaches its kernel.
    """
    var extent = row_major(Coord(element_count))
    var src_buffer = Buffer(ptr=src, layout=extent)
    var result_buffer = Buffer(ptr=result, layout=extent)

    var scale_lanes = SIMD[DTYPE, LANES](scale)
    var offset_lanes = SIMD[DTYPE, LANES](offset)

    var vectorized_end = element_count - (element_count % LANES)
    for i in range(0, vectorized_end, LANES):
        result_buffer.store[LANES](
            Coord(i),
            src_buffer.load[LANES](Coord(i)) * scale_lanes + offset_lanes,
        )

    for i in range(vectorized_end, element_count):
        result_buffer.store[1](
            Coord(i), src_buffer.load[1](Coord(i)) * scale + offset
        )


# ===--------------------------------------------------------------------------===
# Device services
#
# The same `external_call`s that `std._plugin.qaic.host.qshim` lowers to, spelled
# out here rather than imported: the overlay's target gate is exact equality
# against the musl triple, and this build uses the bare-metal one, so importing it
# would not reach a QAIC implementation. `qshim_qurt.c` beside this file defines
# the C names below over QuRT. Once the gate admits this triple (M3) these go away
# in favour of the overlay.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes, mirroring the overlay constant."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1
comptime QSHIM_QUERY_CORE_ID: UInt32 = 2
comptime QSHIM_QUERY_NUM_CORES: UInt32 = 3
comptime QSHIM_QUERY_VTCM_SIZE: UInt32 = 4


@always_inline
def _qshim_query(attribute: UInt32) -> Int64:
    """Reads one device attribute, as the overlay's `_qshim_query_u8` does."""
    var value: Int64 = 0
    _ = external_call["qshimQuery", UInt32](attribute, UnsafePointer(to=value))
    return value


@always_inline
def _npu_thread_idx() -> Int:
    """Index of this thread within the core, 0-based.

    Narrowed to 8 bits like the overlay's `qshim_get_thread_id`, whose return
    type is `UInt8`.
    """
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qshim_get_base_vtcm_addr() -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Base of the core's VTCM scratch, or null if it could not be acquired."""
    return external_call[
        "qshimGetBaseVtcmAddr", UnsafePointer[UInt8, MutAnyOrigin]
    ]()


@always_inline
def _qshim_get_vtcm_size() -> UInt32:
    """Bytes of VTCM the shim holds."""
    return UInt32(_qshim_query(QSHIM_QUERY_VTCM_SIZE))


@always_inline
def _qshim_get_hmx_vtcm_addr() -> UnsafePointer[UInt8, MutAnyOrigin]:
    """P5 M2.4 order 065: base of the DEDICATED HMX context's own VTCM window
    (`hmx_ctx_acquire`/`qshimGetHmxVtcmAddr` in `qshim_qurt.c`) -- separate
    from `_qshim_get_base_vtcm_addr`'s shared one, which every non-HMX VTCM
    consumer still uses. Null if the dedicated context could not be acquired.
    """
    return external_call[
        "qshimGetHmxVtcmAddr", UnsafePointer[UInt8, MutAnyOrigin]
    ]()


@always_inline
def _qshim_get_hmx_vtcm_size() -> UInt32:
    """Bytes of VTCM the dedicated HMX context holds. Not routed through the
    generic `_qshim_query` enum (which reads the shared context's size) --
    a dedicated C getter instead, since this is a second, independent grant.
    """
    return UInt32(external_call["qshimGetHmxVtcmSize", UInt32]())


@always_inline
def _qshim_sync_threads(
    thread_idx: UInt32,
    sync: UnsafePointer[Scalar[DType.int32], MutUnsafeAnyOrigin],
) -> UInt32:
    """Waits until every thread named in the sync point's mask has arrived.

    `sync` addresses the overlay's `QShimThreadSyncStruct` — a thread mask then a
    word the barrier keeps its own state in — spelled as the two `Int32`s it is
    rather than as a struct, since only the shim reads its fields.
    """
    return external_call["qshimSyncThreads", UInt32](thread_idx, sync)


@always_inline
def _qshim_linear_udma_submit(
    thread_idx: UInt32,
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    size: UInt32,
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    status: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
) -> UInt32:
    """Starts a linear transfer, returning a handle to wait on.

    Addresses go as 32-bit values to the same trampoline the overlay calls, for
    the reason it gives: a 64-bit argument lands in the wrong half of the
    register pair across the `external_call` boundary on this 32-bit ABI.

    Ordering attributes are absent, which is what a transfer that nothing is
    chained behind wants, and absent as an empty `Optional` because that is how a
    null pointer is spelled — `UnsafePointer` itself is non-nullable.
    """
    var no_attrs = Optional[UnsafePointer[UInt8, MutAnyOrigin]](None)
    return external_call["skye_udma_submit_linear_lo32", UInt32](
        thread_idx,
        UInt32(Int(src)),
        size,
        UInt32(Int(dst)),
        no_attrs,
        Int32(1),
        status,
    )


@always_inline
def _qshim_udma_wait(handle: UInt32) -> UInt32:
    """Waits out the transfers this thread started."""
    return external_call["qshimUDmaWait", UInt32](handle)


@always_inline
def _qshim_udma_await(thread_idx: UInt32) -> UInt32:
    """Pops this thread's OLDEST in-flight transfer (FIFO, per-descriptor).

    The ring shim's await: the head descriptor's own completion record,
    `dmpoll` between reads -- never the whole-chain `dmwait`, which under a
    ring would reinstate depth-1 semantics at the await point
    (`UDMA_RING_DESIGN.md` section 5). Returns 0 popped-or-empty, 1 for a
    wedged engine (bounded, never a hang)."""
    return external_call["qshimUDmaAwait", UInt32](thread_idx)


@always_inline
def _qshim_udma_flush(thread_idx: UInt32) -> UInt32:
    """Drains everything this thread has in flight.

    The drain-before-exit entry point a deep consumer's epilogue owes
    before its final barrier: an abandoned in-flight descriptor outlives
    the memory it writes, and a VTCM destination turns that into a precise
    DMA error no user PD can read back (`UDMA_RING_DESIGN.md` section 6,
    the EBADSTATE class)."""
    return external_call["qshimUDmaFlush", UInt32](thread_idx)


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code is running on, which no kernel should need.

    Read only to tell threads apart from the host: a launch index is what the
    shim reports, so an index per participant proves nothing on its own about
    how many threads there are behind those indices.
    """
    return external_call["qurt_thread_get_id", UInt32]()


@export
def mojo_npu_env_probe(
    words: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    word_capacity: Int,
) abi("C"):
    """Reports what the device services answer, for a host to print.

    Both are needed before a VTCM-staged kernel can be believed on this
    transport: a kernel that guards on the thread index does nothing if the
    index is not what it expects, and one that stages through VTCM writes to a
    null base if the grant failed. Neither is visible in a kernel's output, so
    they are read out separately rather than inferred from a wrong answer.
    """
    if word_capacity < 6:
        return
    var vtcm = _qshim_get_base_vtcm_addr()
    words.store(0, UInt32(_npu_thread_idx()))
    words.store(1, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    words.store(2, UInt32(_qshim_query(QSHIM_QUERY_CORE_ID)))
    words.store(3, UInt32(_qshim_query(QSHIM_QUERY_NUM_CORES)))
    words.store(4, _qshim_get_vtcm_size())
    words.store(5, UInt32(Int(vtcm)))


@export
def mojo_npu_launch_probe(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    word_capacity: Int,
) abi("C"):
    """Does no work, so that a dispatch of it measures only the launch.

    The per-dispatch cost of this transport is dominated by the launch rather
    than by the kernel, but nothing on record separates the two, because every
    kernel that has ever been dispatched also computes. This one is the empty
    denominator that separation needs: a round-trip of it is the fixed cost any
    real kernel pays before its first instruction.

    The one store is not work but evidence. A launch that never reached the
    device is indistinguishable from one that reached it and returned, so each
    participant marks its own slot; a host that reads back an unmarked slot
    knows the timing it also read back is of something else.
    """
    var index = _npu_thread_idx()
    if index < word_capacity:
        witness.store(index, UInt32(index) + 1)


@export
def mojo_npu_thread_stripe_reduce(
    stripes: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    reduced: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    sync: UnsafePointer[Scalar[DType.int32], MutUnsafeAnyOrigin],
    slice_len: Int,
    use_barrier: Int,
    delay: Int,
    use_dma: Int,
) abi("C"):
    """Fills a stripe per thread, then reduces across the stripes on thread 0.

    Written so that it cannot produce the right answer unless the launch really
    was spread over threads: the stripe a thread fills is chosen by its launch
    index and its contents are derived from it, so threads sharing an index leave
    a stripe untouched, and the reduction spans as many stripes as the shim says
    participate.

    `use_barrier` and `delay` exist to make the barrier's absence visible rather
    than merely possible. Every thread but the reducer holds back for `delay`
    rounds before publishing its stripe, so a reduction that is not held behind
    the barrier reads stripes that nothing has written yet.

    `use_dma` selects how a thread publishes its stripe: by store, or by the
    linear uDMA whose descriptor slot is chosen by the thread's own launch index.
    """
    var thread_idx = _npu_thread_idx()
    var num_threads = Int(_qshim_query(QSHIM_QUERY_NUM_THREADS))

    # Three words per participant, in runs of `num_threads`: the RTOS thread it
    # ran on, the sink that keeps its wait below from being optimized away, and a
    # uDMA submission status if one was refused.
    witness.store(thread_idx, _qurt_thread_id())

    # Staged through VTCM rather than written straight to the output, because a
    # thread the trampoline created is not the thread that acquired the VTCM
    # grant, and a per-thread slice of one shared base is how a multi-thread
    # kernel divides the scratch.
    var slice_bytes = align_up(
        slice_len * size_of[DType.uint32](), Int(HVX_VECTOR_SIZE)
    )
    var staged = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
        unsafe_from_address=align_up(
            Int(_qshim_get_base_vtcm_addr()), Int(HVX_VECTOR_SIZE)
        )
        + thread_idx * slice_bytes
    ).bitcast[Scalar[DType.uint32]]()

    for j in range(slice_len):
        staged.store(j, UInt32((thread_idx + 1) * 1000 + j))

    if thread_idx != 0:
        var sink: UInt32 = 0
        for _ in range(delay):
            sink = sink * 1664525 + 1013904223
        # Stored where the host can see it so the wait cannot be optimized away.
        witness.store(num_threads + thread_idx, sink)

    var published = stripes + thread_idx * slice_len
    if use_dma != 0:
        var status: UInt32 = 0
        var handle = _qshim_linear_udma_submit(
            UInt32(thread_idx),
            staged.bitcast[UInt8](),
            UInt32(slice_len * size_of[DType.uint32]()),
            published.bitcast[UInt8](),
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        _ = _qshim_udma_wait(handle)
        # Where a refused submission is visible: the stripe it would have written
        # stays at the host's seed, which reads the same as a thread that never
        # ran, so the status says which of the two happened.
        if status != 0:
            witness.store(2 * num_threads + thread_idx, status)
    else:
        for j in range(slice_len):
            published.store(j, staged.load(j))

    if use_barrier != 0:
        _ = _qshim_sync_threads(UInt32(thread_idx), sync)

    if thread_idx != 0:
        return

    for j in range(slice_len):
        var total: UInt32 = 0
        for k in range(num_threads):
            total += stripes.load(k * slice_len + j)
        reduced.store(j, total)
    reduced.store(slice_len, UInt32(num_threads))


# ===--------------------------------------------------------------------------===
# Linear uDMA, both directions
# ===--------------------------------------------------------------------------===

comptime UDMA_MODE_ROUNDTRIP = 0
"""DDR into VTCM by uDMA, then VTCM back out to DDR by uDMA."""

comptime UDMA_MODE_NO_INBOUND = 1
"""The outbound leg only, over a VTCM slice nothing staged into.

The control that says the inbound transfer is what fills VTCM: what arrives in
DDR is then the sentinel, so a round trip that comes out right cannot have come
out right by the host's source surviving in place.
"""

comptime UDMA_MODE_STORES = 2
"""Both legs by scalar store, touching no descriptor.

The control for everything around the transfers — the pattern, the per-thread
slicing, the VTCM offsets, the host's comparison — so a failure of the round
trip is attributable to the transfers rather than to the arithmetic.
"""

comptime UDMA_SENTINEL: UInt32 = 0xBAADF00D
"""Written over a thread's VTCM slice before anything stages into it.

Distinct from the seed the host puts in the destination, so a word arriving in
DDR says which of the two legs did not run: the seed means nothing was
transferred out, the sentinel means nothing was transferred in.
"""

comptime UDMA_NO_MISMATCH: UInt32 = 0xFFFFFFFF
"""Reported as the first mismatching index when a thread found none."""

comptime UDMA_STATUS_NO_VTCM: UInt32 = 1 << 16
"""Reported when the VTCM grant is absent or too small for the staging slices."""

comptime UDMA_WITNESS_STRIDE = 5
"""Words of the witness buffer a participant reports into."""


@always_inline
def _udma_pattern(index: UInt32) -> UInt32:
    """The word the host fills source position `index` with.

    A multiplicative hash rather than the index itself, so a transfer that
    copied the wrong slice, shifted it, or wrote indices instead of contents
    disagrees everywhere rather than in a few places.
    """
    return (index * 2654435761) ^ 0x5BF03635


@export
def mojo_npu_udma_roundtrip(
    src: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    dst: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    words: Int,
    stage_word_offset: Int,
    mode: Int,
) abi("C"):
    """Round-trips a slice of DDR through VTCM by linear uDMA.

    Each participant takes a contiguous slice of `words`, stages it into a VTCM
    slice of its own through the descriptor its launch index selects, checks
    what landed there against the pattern the host filled, and sends it back to
    `dst`. The inbound check happens on the device because that is the only
    place the staged copy exists.

    `stage_word_offset` shifts the VTCM slice off the vector boundary it would
    otherwise start on, which is the one alignment the DDR side never has to
    take: a slice boundary lands wherever the split puts it.
    """
    var thread_idx = _npu_thread_idx()
    var num_threads = Int(_qshim_query(QSHIM_QUERY_NUM_THREADS))

    var per_thread = ceildiv(words, num_threads)
    var begin = thread_idx * per_thread
    var end = min(begin + per_thread, words)
    var count = end - begin if end > begin else 0

    # Reported before anything can fail, so a participant that took no slice or
    # found no VTCM is distinguishable from one that never ran.
    var report = witness + thread_idx * UDMA_WITNESS_STRIDE
    report.store(0, UInt32(count))
    report.store(1, 0)
    report.store(2, UDMA_NO_MISMATCH)
    report.store(3, 0)
    report.store(4, 0)
    if count == 0:
        return

    var stage_bytes = align_up(
        (per_thread + stage_word_offset) * size_of[DType.uint32](),
        Int(HVX_VECTOR_SIZE),
    )
    var vtcm = Int(_qshim_get_base_vtcm_addr())
    if vtcm == 0 or stage_bytes * num_threads > Int(_qshim_get_vtcm_size()):
        report.store(4, UDMA_STATUS_NO_VTCM)
        return

    var staged = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
        unsafe_from_address=align_up(vtcm, Int(HVX_VECTOR_SIZE))
        + thread_idx * stage_bytes
        + stage_word_offset * size_of[DType.uint32]()
    ).bitcast[Scalar[DType.uint32]]()

    for j in range(count):
        staged.store(j, UDMA_SENTINEL)

    var slice_bytes = UInt32(count * size_of[DType.uint32]())
    var status: UInt32 = 0

    if mode == UDMA_MODE_STORES:
        for j in range(count):
            staged.store(j, src.load(begin + j))
    elif mode != UDMA_MODE_NO_INBOUND:
        var inbound: UInt32 = 0
        var handle = _qshim_linear_udma_submit(
            UInt32(thread_idx),
            (src + begin).bitcast[UInt8](),
            slice_bytes,
            staged.bitcast[UInt8](),
            UnsafePointer(to=inbound).as_unsafe_any_origin(),
        )
        _ = _qshim_udma_wait(handle)
        status |= inbound

    var wrong: UInt32 = 0
    var first = UDMA_NO_MISMATCH
    var found: UInt32 = 0
    for j in range(count):
        var have = staged.load(j)
        if have != _udma_pattern(UInt32(begin + j)):
            wrong += 1
            if first == UDMA_NO_MISMATCH:
                first = UInt32(begin + j)
                found = have
    report.store(1, wrong)
    report.store(2, first)
    report.store(3, found)

    if mode == UDMA_MODE_STORES:
        for j in range(count):
            dst.store(begin + j, staged.load(j))
    else:
        var outbound: UInt32 = 0
        var handle = _qshim_linear_udma_submit(
            UInt32(thread_idx),
            staged.bitcast[UInt8](),
            slice_bytes,
            (dst + begin).bitcast[UInt8](),
            UnsafePointer(to=outbound).as_unsafe_any_origin(),
        )
        _ = _qshim_udma_wait(handle)
        status |= outbound

    report.store(4, status)


# ===--------------------------------------------------------------------------===
# `mo.moe.create.indices`
#
# The kernel body below is `_moe_create_indices_kernel` from
# `max/kernels/src/qaic/moe_create_indices.mojo`, which is the registered
# implementation of `mo.moe.create.indices` and the source of truth for this
# algorithm. It is duplicated rather than imported because that module also
# imports `extensibility` and `std.gpu.host`, neither of which a standalone
# `mojo build --emit=object` for the bare-metal Hexagon triple can resolve. Only
# the three device services above are respelled; the arithmetic, the phase order
# and the committed output ordering are unchanged, so what runs here is what the
# op dispatches to on the discrete part.
#
# Keep the two in step: a change to one is a change to both until the kernel can
# be shared.
# ===--------------------------------------------------------------------------===


@export
def mojo_moe_create_indices(
    topk_ids: UnsafePointer[Scalar[DType.int32], MutUnsafeAnyOrigin],
    token_expert_order: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    expert_start_indices: UnsafePointer[
        Scalar[DType.uint32], MutUnsafeAnyOrigin
    ],
    restore_token_order: UnsafePointer[
        Scalar[DType.uint32], MutUnsafeAnyOrigin
    ],
    expert_ids: UnsafePointer[Scalar[DType.int32], MutUnsafeAnyOrigin],
    expert_usage_stats: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    n: Int,
    num_experts: Int,
) abi("C"):
    """Groups MoE token-slots by expert (deterministic counting sort)."""
    # Single-thread pass: everyone but thread 0 returns (no cross-thread work,
    # so no barrier is required).
    if _npu_thread_idx() != 0:
        return

    assert num_experts * 4 <= Int(
        _qshim_get_vtcm_size()
    ), "QAIC moe_create_indices: num_experts cursor exceeds VTCM"

    # `cursor` scratch in VTCM: one uint32 per expert.
    var cursor = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
        unsafe_from_address=align_up(
            Int(_qshim_get_base_vtcm_addr()), Int(HVX_VECTOR_SIZE)
        )
    ).bitcast[Scalar[DType.uint32]]()

    # Phase 1: count slots per expert.
    for e in range(num_experts):
        cursor.store(e, UInt32(0))
    for i in range(n):
        var e = Int(topk_ids.load(i))
        cursor.store(e, cursor.load(e) + 1)

    # Phase 2: usage stats (max tokens on any expert, count of active experts).
    # Read counts before Phase 3 overwrites the cursor.
    var max_count: UInt32 = 0
    var num_active: UInt32 = 0
    for e in range(num_experts):
        var c = cursor.load(e)
        if c > max_count:
            max_count = c
        if c > 0:
            num_active += 1
    expert_usage_stats.store(0, max_count)
    expert_usage_stats.store(1, num_active)

    # Phase 3: exclusive prefix sum -> expert_start_indices; expert_ids = arange;
    # overwrite cursor[e] in place with expert e's start offset.
    var offset: UInt32 = 0
    for e in range(num_experts):
        expert_start_indices.store(e, offset)
        var c = cursor.load(e)
        cursor.store(e, offset)
        offset += c
        expert_ids.store(e, Int32(e))
    expert_start_indices.store(num_experts, offset)  # == n

    # Phase 4: stable scatter. Slots walked in ascending index order, so within
    # each expert the grouped order is ascending (matches argsort-stable).
    for i in range(n):
        var e = Int(topk_ids.load(i))
        var pos = cursor.load(e)
        cursor.store(e, pos + 1)
        token_expert_order.store(Int(pos), UInt32(i))
        restore_token_order.store(i, pos)


# ===--------------------------------------------------------------------------===
# `mo.reduce.rms_norm`
#
# IMPORTED, not folded (cold critique C2; megakernel slice 3). `RMSNorm2Pass`,
# `_rms_norm` and `_rms_norm_entry` now live in `rms_norm_kernel.mojo`, which is
# `max/kernels/src/qaic/rms_norm.mojo`'s math on the tile model and is what
# `test_rms_norm_kernel.mojo` runs on the host. This file used to be the only
# copy: unlike the layer norm beside it, the kernel was authored here and had no
# authoring module at all, so what the extraction closes is not drift between two
# copies but the absence of any host twin over the shipped body.
#
# What stays here are the three exports, under the names the kernel manifest
# knows, and the VTCM-staged variant. `mojo_rms_norm_f16` runs the compute on the
# caller's DDR pointers, which HVX addresses directly; `mojo_rms_norm_f16_staged`
# stages each row through VTCM by uDMA, which is what the registered kernel's
# multi-NSP driver does. Both drive the same imported `RMSNorm2Pass`, and the row
# split across the launch's threads is `_rms_norm`'s own either way.
#
# `_reduce` and `_reduce_add` below are NOT the rms kernel's, though it is where
# they were first written: the GEMV, the matmul and the attention family fold
# with them. The predicate and unroll-factor helpers that used to sit beside them
# went with the group norm, which was their last caller here; every extracted
# module carries its own copies so it can be compiled standalone.
# ===--------------------------------------------------------------------------===

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""


comptime _bin_op = def(lhs: SIMD, rhs: type_of(lhs)) thin -> type_of(lhs)


@always_inline
def _reduce[op: _bin_op](vec: SIMD) -> type_of(vec):
    """Folds a vector into every lane by rotate-and-combine."""
    var reduced = vec

    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < vec.size:
            reduced = op(reduced, reduced.rotate_right[1 << i]())
    return reduced


comptime _reduce_add = _reduce[SIMD.__add__]


# Fused interleaved RoPE inside the qk-norm entry (build.sh ROPE_FUSED_NORM=1;
# define emitted only off-default so a flag-off command line is byte-identical
# to the base tree's). When on, selector 5 grows a second arm the host opts
# into PER DISPATCH through the `multiply_before_cast` slot (0/1 keep their
# legacy meaning, anything else names the fused contract inside the witness
# allocation — rms_norm_kernel.mojo documents the header). The arming gate is
# bit-identity against the selector 5 -> selector 9 sequence, and the arm is
# echoed in witness word 8 so the host self-configures from a one-row probe
# (the 229 double-apply lesson). fp16-only by design — the dispatched
# production selector is 5; the staged (26) and bf16 (30) rows stay unfused.
comptime ROPE_FUSED_NORM = get_defined_int["SKYE_HMX_ROPE_FUSED_NORM", 0]()


@export
def mojo_rms_norm_f16(
    output: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
    weight_offset: Float32,
    multiply_before_cast: Int,
) abi("C"):
    """Normalizes fp16 rows on the launch's threads, split by launch index.

    This is the seam, and the same three lines the layer norm's is: the skel's C
    ABI hands down pointers and extents, the extents become a layout once, here,
    and everything under it is the imported module — the shipped code and the
    tested code at the same time.
    """
    var extent = row_major(Coord(num_rows, length))
    comptime if ROPE_FUSED_NORM != 0:
        _rms_norm_rope_entry[RMS_DTYPE](
            RmsRowMajor2D[RMS_DTYPE](ptr=output, layout=extent),
            RmsRowMajor2D[RMS_DTYPE](ptr=input, layout=extent),
            rms_row[RMS_DTYPE](gamma, length),
            witness,
            eps,
            weight_offset,
            multiply_before_cast,
        )
    else:
        _rms_norm_entry[RMS_DTYPE](
            RmsRowMajor2D[RMS_DTYPE](ptr=output, layout=extent),
            RmsRowMajor2D[RMS_DTYPE](ptr=input, layout=extent),
            rms_row[RMS_DTYPE](gamma, length),
            witness,
            eps,
            weight_offset,
            multiply_before_cast,
        )


@export
def mojo_rms_norm_bf16(
    output: UnsafePointer[Scalar[RMS_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[RMS_DTYPE_BF], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[RMS_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
    weight_offset: Float32,
    multiply_before_cast: Int,
) abi("C"):
    """Normalizes bf16 rows on the launch's threads, split by launch index.

    The seam, as `mojo_rms_norm_f16` documents it.
    """
    var extent = row_major(Coord(num_rows, length))
    _rms_norm_entry[RMS_DTYPE_BF](
        RmsRowMajor2D[RMS_DTYPE_BF](ptr=output, layout=extent),
        RmsRowMajor2D[RMS_DTYPE_BF](ptr=input, layout=extent),
        rms_row[RMS_DTYPE_BF](gamma, length),
        witness,
        eps,
        weight_offset,
        multiply_before_cast,
    )


@always_inline
def _rms_norm_row_staged[
    multiply_before_cast: Bool
](
    thread_idx: Int,
    output: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    gamma_vtcm: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    in_vtcm: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_vtcm: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    row_bytes: Int,
    start_row: Int,
    end_row: Int,
    length: Int,
    eps: Float32,
    weight_offset: Scalar[RMS_DTYPE],
):
    """Normalizes this thread's rows one at a time out of its VTCM slice.

    Split out of the entry point only because `multiply_before_cast` is a
    parameter and both instantiations have to be compiled, as they are for the
    unstaged kernel.

    The three VTCM slots hold one row each, so each is a `(1, length)` row view
    and the compute below is the imported `RMSNorm2Pass` unchanged. The uDMA
    addresses stay raw: a transfer names a base and a byte count, which is not a
    coordinate into anything.
    """
    var in_row = rms_row[RMS_DTYPE](
        in_vtcm.bitcast[Scalar[RMS_DTYPE]](), length
    )
    var out_row = rms_row[RMS_DTYPE](
        out_vtcm.bitcast[Scalar[RMS_DTYPE]](), length
    )
    var gamma_row = rms_row[RMS_DTYPE](
        gamma_vtcm.bitcast[Scalar[RMS_DTYPE]](), length
    )
    var kernel = RMSNorm2Pass[
        RMS_DTYPE, multiply_before_cast=multiply_before_cast
    ](
        RmsRowMajor2D[RMS_DTYPE](
            ptr=in_vtcm.bitcast[Scalar[RMS_DTYPE]](),
            layout=row_major(Coord(1, length)),
        ),
        eps,
        weight_offset,
    )
    var status: UInt32 = 0

    for row in range(start_row, end_row):
        var inbound = _qshim_linear_udma_submit(
            UInt32(thread_idx),
            (input + row * length).bitcast[UInt8](),
            UInt32(row_bytes),
            in_vtcm,
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        _ = _qshim_udma_wait(inbound)

        kernel(out_row, in_row, gamma_row)

        var outbound = _qshim_linear_udma_submit(
            UInt32(thread_idx),
            out_vtcm,
            UInt32(row_bytes),
            (output + row * length).bitcast[UInt8](),
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        _ = _qshim_udma_wait(outbound)


@export
def mojo_rms_norm_f16_staged(
    output: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[RMS_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
    weight_offset: Float32,
    multiply_before_cast: Int,
) abi("C"):
    """Normalizes rows out of VTCM, staged there and back by linear uDMA.

    Same signature, same row split and same witness as `mojo_rms_norm_f16`, so
    the two are interchangeable to a host and comparable against one reference.
    What differs is where the compute reads and writes: each row is transferred
    into this thread's VTCM slice, normalized out of scratch, and transferred
    back out, rather than computed in place on DDR through the cache.

    Every participant drives its own transfers, through the descriptor its
    launch index selects, which is what lets this need no rendezvous: a thread
    stages only the rows it computes. `rms_norm_impl` on the discrete part
    instead dedicates the fifth thread to the transfers so they overlap the
    compute, and pays a barrier per tile for it. Overlap is not what this
    establishes.

    The slots come from the step arena's far end (`StepArena.scratch_top`), not
    from the grant base the HMX stripe carves from. The values are unchanged --
    a staged row is a copy in and a copy out, and the address it passes through
    does not enter the arithmetic -- but under the composed walk the two
    consumers are live at once, and a shared base would make their disjointness
    an accident of scheduling.
    """
    var thread_idx = _npu_thread_idx()
    _rms_norm_witness[RMS_DTYPE](witness, thread_idx)

    # The same range `_rms_norm` computes, so which rows a launch covers does not
    # depend on where the compute reads from.
    if thread_idx >= Int(NUM_COMPUTE_THREADS):
        return
    var rows_per_thread = ceildiv(num_rows, Int(NUM_COMPUTE_THREADS))
    var start_row = thread_idx * rows_per_thread
    if start_row >= num_rows:
        return
    var end_row = min(start_row + rows_per_thread, num_rows)

    var row_bytes = length * size_of[RMS_DTYPE]()
    var stride = align_up(row_bytes, Int(HVX_VECTOR_SIZE))
    var slot_bytes = staged_norm_slot_bytes(length)

    # Through the arena, at its far end, rather than from the raw grant base.
    # Under one dispatch at a time this and the HMX stripe never coexist, so
    # both carving upward from the same base was survivable; under a composed
    # walk they do coexist, and "survivable" would be the whole disjointness
    # argument. The top anchor makes them meet only where the build refusal
    # (`test_step_arena.mojo` gate 7) already forbids.
    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    var scratch = arena.scratch_top(staged_norm_bytes(length))
    if scratch == 0:
        # Leaves this thread's rows at whatever the host seeded, which is what it
        # counts as missing.
        return

    var gamma_vtcm = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
        unsafe_from_address=scratch + thread_idx * slot_bytes
    )
    var in_vtcm = gamma_vtcm + stride
    var out_vtcm = in_vtcm + stride

    var status: UInt32 = 0
    var handle = _qshim_linear_udma_submit(
        UInt32(thread_idx),
        gamma.bitcast[UInt8](),
        UInt32(row_bytes),
        gamma_vtcm,
        UnsafePointer(to=status).as_unsafe_any_origin(),
    )
    _ = _qshim_udma_wait(handle)

    var offset = weight_offset.cast[RMS_DTYPE]()
    if multiply_before_cast != 0:
        _rms_norm_row_staged[multiply_before_cast=True](
            thread_idx,
            output,
            input,
            gamma_vtcm,
            in_vtcm,
            out_vtcm,
            row_bytes,
            start_row,
            end_row,
            length,
            eps,
            offset,
        )
    else:
        _rms_norm_row_staged[multiply_before_cast=False](
            thread_idx,
            output,
            input,
            gamma_vtcm,
            in_vtcm,
            out_vtcm,
            row_bytes,
            start_row,
            end_row,
            length,
            eps,
            offset,
        )


# ===--------------------------------------------------------------------------===
# `mo.reduce.layer_norm` (no-affine) — HVX layer normalization
#
# IMPORTED, not folded (cold critique C2; megakernel slice 2). This file used to
# carry its own copy of `LayerNorm3Pass` / `_layer_norm` / `_layer_norm_entry`
# beside the tested original in `layernorm_kernel.mojo`, "under the same
# obligation: a change to one is a change to both". The two had already drifted:
# the authoring module took typed row views through every signature below its
# seam (P4.1) and the copy shipped here still passed pointers and extents. So the
# host twin `test_layernorm_kernel.mojo` proved a body no device ever ran.
#
# The exports below stay here, under the names the manifest knows, and are the
# same three-line seam the module's own `hvx_layernorm_*` entries are: the skel's
# C ABI hands down pointers and extents, and the extents become a layout once,
# here. Everything under that seam is now the shipped code and the tested code at
# the same time. `build.sh` stages the module for the device compile.


@export
def mojo_layernorm_f16(
    output: UnsafePointer[Scalar[LN_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[LN_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
) abi("C"):
    """No-affine layer norm over fp16 rows, spread across the launch's threads.
    """
    var extent = row_major(Coord(num_rows, length))
    _layer_norm_entry[LN_DTYPE](
        LnRowMajor2D[LN_DTYPE](ptr=output, layout=extent),
        LnRowMajor2D[LN_DTYPE](ptr=input, layout=extent),
        witness,
        eps,
    )


@export
def mojo_layernorm_bf16(
    output: UnsafePointer[Scalar[LN_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[LN_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
) abi("C"):
    """No-affine layer norm over bf16 rows, spread across the launch's threads.
    """
    var extent = row_major(Coord(num_rows, length))
    _layer_norm_entry[LN_DTYPE_BF](
        LnRowMajor2D[LN_DTYPE_BF](ptr=output, layout=extent),
        LnRowMajor2D[LN_DTYPE_BF](ptr=input, layout=extent),
        witness,
        eps,
    )


# ===--------------------------------------------------------------------------===
# `mo.matmul` decode path — HVX GEMV (M == 1)
#
# `gemv_impl`'s M == 1 body from `max/kernels/src/qaic/gemv.mojo`, the HVX GEMV
# that `matmul/adapter.mojo` routes an `M == 1, K % 64 == 0` decode matmul to
# with no HMX. Duplicated for the reason the two kernels above are, and under the
# same obligation: a change to one is a change to both until the kernel can be
# shared.
#
# Three respellings, all forced by the standalone bare-metal build and none
# touching the arithmetic:
#
# - The SIMD width, `simd_width_of[dtype, target=get_gpu_target()]()` in the
#   source, becomes the documented HVX register width over the element size — the
#   `simd_width_of` trap the rms_norm port records. It is load-bearing twice
#   here: `_fmac_sf_hf` asserts its width is 64, so a host-width resolution does
#   not merely emit scalar code, it fails to compile.
# - `fmac_sf_hf` and `reduce_add` are respelled from `utils.mojo` and
#   `softmax/reduce.mojo`, which reach the overlay through their own modules.
#   `_reduce_add` above is the same fold; the fp16 x fp16 -> fp32 MAC is the
#   `V6_vmpy_sf_hf_acc` HVX instruction, spelled out below.
# - `npu_core_idx` / `npu_thread_idx` become the `qshim` queries above.
#
# What is *not* carried over is the VTCM double-buffering and the shared-A uDMA
# staging: linear uDMA faulted this protection domain when this port was written
# (LAUNCH_GEOMETRY_STATUS.md). The general mechanism has since been established
# -- `mojo_rms_norm_f16_staged` above normalizes out of VTCM, and UDMA_STATUS.md
# records what the transfers accept -- so the fault is no longer a property of
# the domain. Whether this path can stage has not been confirmed: it needs A
# shared across threads rather than a slice per thread, which is the case the
# round trip does not cover. Until it is, HVX addresses DDR directly, so each
# thread reads A and its own B rows from the caller's pointers; the N-row split
# across the launch's threads — which is what `gemv_impl` does and what the
# double buffer would have fed — is unaffected. The archive barrier
# `hmx_sync_hvx_threads`, needed there only to publish the VTCM-staged A, is
# dropped rather than substituted: with A read from DDR per thread, no thread
# reads another's work.
# ===--------------------------------------------------------------------------===

comptime GEMV_DTYPE = DType.float16
"""The dtype the decode matmul is used at; `_fmac_sf_hf` requires fp16."""

comptime GEMV_SIMD_W = Int(HVX_VECTOR_SIZE) // size_of[GEMV_DTYPE]()
"""HVX lanes per vector op: 64 half-precision elements in a 128-byte vector."""


@always_inline
def _fmac_sf_hf[
    width: Int
](
    c: SIMD[DType.float32, width],
    a: SIMD[GEMV_DTYPE, width],
    b: SIMD[GEMV_DTYPE, width],
) -> SIMD[DType.float32, width]:
    """Returns `c + a*b` through the HVX `V6_vmpy_sf_hf_acc` instruction.

    Respelled from `fmac_sf_hf` in `utils.mojo`. A single fused fp16 x fp16 +
    fp32 -> fp32 multiply-accumulate: plain `c + a.cast[f32]() * b.cast[f32]()`
    lowers to ~14 packets, so the intrinsic is issued directly. Result lanes land
    in the instruction's interleaved order, which an order-independent horizontal
    reduction tolerates.
    """
    comptime assert (
        width == 64
    ), "V6_vmpy_sf_hf_acc operates on 64 fp16 lanes (1024-bit HVX, 128B mode)"

    # Reinterpret as the i32 vector / pair the intrinsic is typed against (same
    # bitwidth, no value conversion): c -> v64i32, a/b -> v32i32.
    var acc_i = bitcast[DType.int32, width](c)
    var a_i = bitcast[DType.int32, width // 2](a)
    var b_i = bitcast[DType.int32, width // 2](b)

    var res_i = llvm_intrinsic[
        "llvm.hexagon.V6.vmpy.sf.hf.acc.128B",
        SIMD[DType.int32, width],
        has_side_effect=False,
    ](acc_i, a_i, b_i)

    return bitcast[DType.float32, width](res_i)


@export
def mojo_gemv_f16(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    A: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    K: Int,
    N: Int,
) abi("C"):
    """Computes `C[N] = A[K] @ B[N, K]^T` on the launch's threads.

    B is stored N x K (transposed), so output row `n` is the fp32 dot product of
    A with B's row `n`, accumulated 64 lanes at a time and folded to a scalar.
    The N rows divide across the four compute threads by launch index; the fifth
    (DMA) thread takes none — the split `gemv_impl` makes. `K` must be a multiple
    of 64, the decode-matmul contract `matmul/adapter.mojo` routes on.

    `witness` reports what the output cannot: the SIMD width the kernel compiled
    at (64 for the vector units, or the host width had it resolved there), the
    fp32 accumulator width, the participant count the shim reports, and the RTOS
    thread each launch index ran on. The width is the one worth reading — a build
    that resolved it against the host would still compute the right dot products,
    at a lane count that says the vector units never ran.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(GEMV_SIMD_W))
        witness.store(1, UInt32(SIMD[DType.float32, GEMV_SIMD_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    if thread_idx >= Int(NUM_COMPUTE_THREADS):
        return

    # Each compute thread owns a contiguous slice of the N output rows.
    var rows_per_thread = ceildiv(N, Int(NUM_COMPUTE_THREADS))
    var t_start = thread_idx * rows_per_thread
    if t_start >= N:
        return
    var t_end = t_start + rows_per_thread
    if t_end > N:
        t_end = N

    var k_vecs = K // GEMV_SIMD_W
    for n in range(t_start, t_end):
        var b_row = B + n * K
        var acc = SIMD[DType.float32, GEMV_SIMD_W](0)
        for kv in range(k_vecs):
            var k_off = kv * GEMV_SIMD_W
            var av = A.load[GEMV_SIMD_W](k_off)
            var bv = b_row.load[GEMV_SIMD_W](k_off)
            acc = _fmac_sf_hf(acc, av, bv)
        C[n] = _reduce_add(acc)[0].cast[GEMV_DTYPE]()


# ===--------------------------------------------------------------------------===
# `mo.matmul` — batched/tiled HVX matmul (the many-rows-per-dispatch GEMV),
# and `mo.conv` — the same nest over a fused online im2col
#
# The register-blocked extension of the one-output-row `gemv` above, authored and
# host-validated in `matmul_kernel.mojo` and duplicated here under the same
# obligation as the softmax below: a change to the loop nest in one is a change to
# both until the kernel can be shared. Only the MAC differs — the host core widens
# and multiplies; this issues `_fmac_sf_hf` (`V6_vmpy_sf_hf_acc`), whose 64 fp32
# lanes land interleaved and fold to the same dot product under `_reduce_add`.
# The `l2fetch` hints below are the one thing the twin has no counterpart for:
# they move cache lines, not the nest, so the twin still gates the arithmetic.
#
# The nest reads its A operand through a `MatmulAReader`, so `mojo_matmul_f16`
# and `mojo_conv_im2col_f16` are one kernel that differs in where an A vector
# comes from: a row of the activation, or a gather out of an NHWC input map that
# never builds the `[M, K]` im2col matrix. The dense arm is byte-identical to the
# build that had no seam (`kernel_asm_compare.sh` over `mojo_matmul_f16`:
# 1794 insns / 986 HVX ops / 6 vector spills / 1045 packets, unchanged), which is
# what says the signed matmul owes no re-sign for this.
#
# `gemv` computes one output row per dispatch, so a DiT denoise loop at full-Klein
# M costs O(M) dispatches per projection — hundreds of thousands over the loop,
# swamping the compute roofline on dispatch overhead. This computes the whole
# `[M, N]` output in one dispatch (O(1) per projection). Register blocking
# (`MATMUL_MR` A-rows x `MATMUL_NR` B-rows) reuses each loaded activation/weight
# HVX vector across the tile, so a B row loaded once feeds every A row —
# amortizing the weight read that dominates a decode-shaped matmul. Cache
# blocking (`MATMUL_MC_BYTES` of resident activation, an M-tile outside the N
# sweep) is the other half: it bounds the activation rows one N sweep touches, so
# a row crosses DRAM once per M-tile rather than once per N-pair. Same
# `b_transposed` [N, K] fp16 contract and `K % 64 == 0` as the GEMV; the N output
# columns divide across the four compute threads as the GEMV's N output rows do,
# every thread computing all M rows for its column stripe. Each output is a
# scalar-reduced dot product, so the store to `C` is never a masked vector store —
# the N%64 tail-store fault class cannot arise.
# ===--------------------------------------------------------------------------===

comptime MATMUL_MR = 4
"""Register-block extent over A-rows (M). Each of `MATMUL_MR * MATMUL_NR` fp32
accumulators is a 64-lane vector pair; the k-loop working set stays under the 32
HVX vector registers. `MATMUL_MR > MATMUL_NR` favors weight (B) reuse."""

comptime MATMUL_NR = 2
"""Register-block extent over B-rows (N). See `MATMUL_MR`."""

comptime MATMUL_KU = 2
"""K-steps whose operand loads are issued before any of them is consumed.

The k-loop costs ~90 cycles per k-step against a 33-to-39 packet static body, so
roughly 51-57 cycles of it are stall. An HVX vector is exactly one cache line, so
a k-step has no spatial reuse to exploit -- each of its `MR + NR` loads is a
separate line -- and the only lever on that stall is how many of those lines are
in flight at once. Unrolling puts `MATMUL_KU` steps' worth of misses in flight
and pays one latency for all of them.

The 90 cycles are derived from the matmul's own measured rate, 28.3 GMAC/s
(389.77 GMAC in the 13.79 s the block-stack driver attributes to this kernel),
and two vehicles agree on it within 1.4%. It is NOT the 8.2x-off-issue the
earlier record reports: that figure divided matmul MACs by WHOLE-STEP time, and
attention is 52.5% of the step, so more than half of that denominator belongs to
another kernel. The real headroom here is 2.3-2.7x, which bounds what this
unroll can be worth.

Two register-blocked tiles bound it. At `MATMUL_MR` x `MATMUL_NR` the k-loop
working set is `2*MR*NR + (MR + NR) * MATMUL_KU` vectors -- 28 of the 32 HVX
vector registers at KU=2, and 34 at KU=3, which would spill the accumulators
and cost more than the latency it hides. Two is the largest that fits.

One compiles the loop back to the un-unrolled nest, which is the A/B control.
The unroll cannot change any result: accumulator `(mr, nr)` still takes its
k-steps in order with the same operands."""

comptime MATMUL_MC_BYTES = 512 * 1024
"""Activation bytes held resident across one N sweep — the M-tile budget.

Without it the N-outer/M-inner nest re-sweeps the whole `M x K` activation panel
once per N-pair, so every MAC costs a byte of activation traffic and the DiT step
is bound by that stream rather than by the vector units. An M-tile outside the N
sweep drops it to one pass per tile.

The one empirical knob: too small and the weight stripe is re-read once per tiny
tile, too large and the panel is evicted before the N sweep reuses it. 512 KiB is
the measured knee of an on-device sweep of the full DiT step (534 s at 256 KiB,
499 s here, 704 s at 1 MiB, 1203 s untiled), so it is the effective L2 working
set for four compute threads plus a weight and an output stream — not the L2
capacity, which v81 does not publish. See BATCHED_MATMUL_STATUS.md."""

comptime MATMUL_PF_B_ROWS = MATMUL_NR
"""Weight rows `l2fetch`ed one N step ahead of the sweep that reads them.

The k-loop carried no prefetch at all, so an operand line that was not already
resident cost a fully exposed DRAM latency. This asks the L2 prefetch engine
for the next tile's operand rows while the current tile computes: `B` rows
`[n + MATMUL_PF_B_ROWS, n + 2 * MATMUL_PF_B_ROWS)` at each N step, and `A` rows
`[m + MATMUL_PF_A_ROWS, m + 2 * MATMUL_PF_A_ROWS)` at each M step.

Both default to the register-block extent, which an on-device sweep of the DiT
step picked out: the fetch wants to be exactly the next tile's operands and no
more. Fetching further ahead is neutral (2 rows at a lead of 8 matches a lead of
2), but fetching *more* per issue gives the gain back — 4 weight rows, then 8,
then 32, walk the step monotonically back to the un-prefetched time, which is
what an L2 that the M-tile budget already fills to its knee would do. See
BATCHED_MATMUL_STATUS.md.

Zero compiles the respective prefetch out; both zero leaves the kernel
byte-identical to the un-prefetched build, which is the A/B control."""

comptime MATMUL_PF_A_ROWS = MATMUL_MR
"""Activation rows `l2fetch`ed one M step ahead. See `MATMUL_PF_B_ROWS`."""


@always_inline
def _l2fetch_rows(
    addr: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    row_bytes: Int,
    rows: Int,
):
    """Asks the L2 prefetch engine for `rows` contiguous rows of `row_bytes`.

    `l2fetch(Rs32,Rtt32)` takes its 2D region from the control word: height in
    `Rtt[15:0]`, width in `Rtt[31:16]`, stride in `Rtt[47:32]`, direction in
    `Rtt[63:48]`. Width equal to stride describes a contiguous region. Width
    and stride are 16-bit fields, so `row_bytes` must be at most 0xFFFF — the
    caller checks that rather than truncating into a wrong region.

    A prefetch moves cache lines, never data, so it cannot change any value the
    kernel computes; that is what keeps the tiled result bit-identical.
    """
    var control = (
        (UInt64(row_bytes) << 32) | (UInt64(row_bytes) << 16) | UInt64(rows)
    )
    llvm_intrinsic["llvm.hexagon.Y5.l2fetch", NoneType](
        addr.bitcast[NoneType](), control
    )


@always_inline
def _l2fetch_rows_2d(
    addr: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    row_bytes: Int,
    stride_bytes: Int,
    rows: Int,
):
    """Asks the L2 prefetch engine for a STRIDED region: `rows` rows of
    `row_bytes` every `stride_bytes`.

    `_l2fetch_rows` above is this with `stride_bytes == row_bytes`, and cannot
    express anything else: it writes one value into both fields, so a caller
    whose rows are a slice of a wider matrix has no way to say so. Every operand
    this kernel file reads out of a `[seq, heads, dim]` tensor is exactly that
    shape, which is the class order 096 leg A measured at 26x the contiguous
    one.

    Width, height and stride are 16-bit fields. A region that does not fit them
    is DROPPED rather than truncated: a truncated control word names a different
    region, which cannot change a value but would spend bandwidth on lines
    nothing reads, so the hint would stop being the thing under measurement.

    A prefetch moves cache lines, never data, so it cannot change any value the
    kernel computes.
    """
    if (
        rows <= 0
        or rows > 0xFFFF
        or row_bytes > 0xFFFF
        or stride_bytes > 0xFFFF
    ):
        return
    var control = (
        (UInt64(stride_bytes) << 32) | (UInt64(row_bytes) << 16) | UInt64(rows)
    )
    llvm_intrinsic["llvm.hexagon.Y5.l2fetch", NoneType](
        addr.bitcast[NoneType](), control
    )


comptime MatmulImage[o: Origin[mut=True]] = TileTensor[
    GEMV_DTYPE, type_of(row_major(Coord(0, 0, 0))), origin=o
]
"""An NHWC input map for one batch index: `(H, W, C_in)`, channel fastest.

The operand the fused convolution arm gathers from. Rank 3 rather than rank 4
because the batch loop is the dispatcher's — one launch per batch index.

Parameters:
    o: Origin of the input map, carried from whatever the caller built. A struct
        field may not name `AnyOrigin` directly, so the reader that holds one is
        parameterized on the origin rather than pinning it.
"""

comptime MatmulFlatImage[o: Origin[mut=True]] = TileTensor[
    GEMV_DTYPE, type_of(row_major(Coord(0))), origin=o
]
"""The same input map as one contiguous run of `H * W * C_in` elements.

A rank-1 `reshape` of `MatmulImage`, which is a view and moves nothing. The
gather walks the window by a carried linear offset rather than by `(h, w, c)`,
because a rank-3 access re-derives `h * W * C_in + w * C_in` at every k-vector —
two `mpyi` per A vector, `MATMUL_MR` times per k-step. Twin of `MMFlatImage`.

Parameters:
    o: Origin of the input map, carried from the rank-3 view it reshapes.
"""


struct MatmulRowAnchor(TrivialRegisterPassable):
    """The K-invariant part of one A row's address, prebaked once per microtile.

    Dense: `row` is the activation row and `col` is unused. Fused conv: they are
    the input-map coordinates of the output pixel's TOP-LEFT tap, and `base` is
    that same tap as one linear element offset, which is what lets the gather's
    address be a single add. Twin of `MMRowAnchor` in `matmul_kernel.mojo`; see
    that file for the reasoning.
    """

    var row: Int
    """Dense: the activation row. Conv: `h_out - pad_h`."""

    var col: Int
    """Dense: unused. Conv: `w_out - pad_w`."""

    var base: Int
    """Dense: unused. Conv: element offset of the top-left tap in the map."""

    @always_inline
    def __init__(out self, row: Int, col: Int, base: Int = 0):
        """Records the components of a row's K-invariant address.

        Args:
            row: Dense: the activation row. Conv: `h_out - pad_h`.
            col: Dense: unused. Conv: `w_out - pad_w`.
            base: Dense: unused. Conv: linear element offset of that tap.
        """
        self.row = row
        self.col = col
        self.base = base


trait MatmulAReader(Copyable, ImplicitlyDeletable):
    """Produces the A operand of `_matmul_microtile`, one k-vector at a time.

    The loader-swap seam, mirroring `MMAReader` in `matmul_kernel.mojo` as the
    rest of this nest mirrors that file. `_matmul_stripe` supplies the dense
    reader and `_conv_stripe` the fused im2col gather; the k-loop, the unroll,
    the fp32 accumulators, the fold and the scalar store are the same code for
    both. `anchors_rows` False lets the body skip building the row anchors
    entirely, which is what keeps `mojo_matmul_f16` byte-identical to the build
    that had no seam — a signed kernel whose machine code does not change owes no
    re-sign.
    """

    comptime anchors_rows: Bool
    """Whether `load` reads the anchor; False lets the body skip the prebake."""

    @always_inline
    def anchor(self, m: Int) -> MatmulRowAnchor:
        """Prebakes the K-invariant address state of A row `m`."""
        ...

    @always_inline
    def anchor_next(self, prev: MatmulRowAnchor, m: Int) -> MatmulRowAnchor:
        """The anchor of row `m`, which is the row after `prev`."""
        ...

    @always_inline
    def advance(mut self):
        """Steps the carried k-state on by one `GEMV_SIMD_W`-lane k-vector."""
        ...

    @always_inline
    def load(
        self, anchor: MatmulRowAnchor, m: Int, k_off: Int
    ) -> SIMD[GEMV_DTYPE, GEMV_SIMD_W]:
        """Reads the `GEMV_SIMD_W` lanes of A at this row, k offset `k_off`."""
        ...


struct MatmulDenseReader[o: Origin[mut=True]](MatmulAReader):
    """Reads A straight out of the `[M, K]` activation the skel was handed.

    Still pointer-and-extent, unlike its `matmul_kernel.mojo` twin: this file's
    whole matmul adopts the typed signature at the next re-sign window (see that
    file's module docstring), and changing the dense arm's addressing here would
    forfeit exactly the byte-identity this seam was built to preserve. The fused
    reader below is new code and is tile-model clean.

    Parameters:
        o: Origin of the activation pointer, carried from the caller.
    """

    comptime anchors_rows = False
    """A dense `load` takes the row index, so the body builds no anchors."""

    var A: UnsafePointer[Scalar[GEMV_DTYPE], Self.o]
    """The `[M, K]` activation."""

    var K: Int
    """The activation's row stride."""

    @always_inline
    def __init__(
        out self, A: UnsafePointer[Scalar[GEMV_DTYPE], Self.o], K: Int
    ):
        """Wraps the activation pointer and its row stride.

        Args:
            A: The `[M, K]` activation.
            K: The activation's row stride.
        """
        self.A = A
        self.K = K

    @always_inline
    def anchor(self, m: Int) -> MatmulRowAnchor:
        """Names row `m`.

        Args:
            m: The activation row.

        Returns:
            An anchor naming row `m`.
        """
        return MatmulRowAnchor(row=m, col=0)

    @always_inline
    def anchor_next(self, prev: MatmulRowAnchor, m: Int) -> MatmulRowAnchor:
        """Names row `m` outright; a dense row index needs no stepping.

        Args:
            prev: The previous row's anchor; unused.
            m: The activation row.

        Returns:
            An anchor naming row `m`.
        """
        return MatmulRowAnchor(row=m, col=0)

    @always_inline
    def advance(mut self):
        """Nothing to step: a dense row carries no k-state."""
        pass

    @always_inline
    def load(
        self, anchor: MatmulRowAnchor, m: Int, k_off: Int
    ) -> SIMD[GEMV_DTYPE, GEMV_SIMD_W]:
        """One `GEMV_SIMD_W`-lane contiguous A vector.

        Args:
            anchor: The row's anchor; unused, `m` says the same thing.
            m: The activation row.
            k_off: First k index of the vector.

        Returns:
            `A[m, k_off : k_off + GEMV_SIMD_W]`.
        """
        return self.A.load[GEMV_SIMD_W](m * self.K + k_off)


struct MatmulIm2colReader[o: Origin[mut=True], halo: Bool = True](
    MatmulAReader
):
    """Gathers A from an NHWC input map: the fused online im2col.

    The device twin of `MMIm2colReader`; that file carries the full reasoning.
    The short form: row `m` is output pixel `divmod(m, W_out)`, k index
    `k = (r * S + s) * C_in + c` is that pixel's tap `(r, s)` on channel `c`, and
    with the channel-FASTEST ordering all `GEMV_SIMD_W` lanes of a k-vector share
    one `(r, s)` — so the gather is the same single contiguous `vmemu` the dense
    reader issues, and the halo test is one branch per VECTOR. `C_in % 64 == 0`
    is the contract that guarantees it, and it also makes `K = R * S * C_in` a
    multiple of 64 so this arm needs no K padding.

    The `[M, K]` im2col matrix is never built: no host gather, no `M x K_pad`
    device scratch, no copy across the transport.

    **`halo` is why this arm can be as cheap as the dense one.** Order 061 ran
    the True arm on silicon and it cost 1.55x the materialising path's device
    time on the C=128 H=W=128 resnet conv (109,775.9 us against 70,809.5 us),
    outputs bit-identical — a pure speed loss. `micro_cost.sh` prices the k-loop
    that measured at 1.47x the dense nest's packets doing IDENTICAL vector work:
    12 `vmemu` and 44 `vmpy` on both arms, and 34 branches / 22 compares /
    16 `mpyi` / 18 `addasl` against the dense arm's 1 / 1 / 0 / 0. The deficit
    was entirely the per-vector halo test and the address arithmetic feeding it.
    With `halo` False the caller has proved the whole `R x S` window of every row
    in the microtile lands inside the map, both disappear, and the same probe
    prices the arm at 1.02x dense. `_conv_stripe` runs it wherever
    `_conv_window_inside` holds, which on that layer is 92.3% of the microtiles.

    The address is carried linearly: `tap_off` is this k-vector's element offset
    from the row's top-left tap, stepped by add-and-carry in `advance`, so a
    gather is `anchor.base + tap_off`. A tap ROW's `S * C_in` halves are adjacent
    in an NHWC map, so all but every `S * C_in / 64`-th step is a bare `+= 64`
    and the exception adds `(W - S) * C_in`.
    """

    comptime anchors_rows = True
    """The gather reads the prebaked pixel anchor rather than re-deriving it."""

    var img: MatmulFlatImage[Self.o]
    """The input map as one contiguous run, which is how the gather walks it."""

    var w_out: Int
    """Output width; `m == h_out * W_out + w_out`."""

    var pad_h: Int
    """Rows of zero padding above the input map."""

    var pad_w: Int
    """Columns of zero padding left of the input map."""

    var h_lim: Int
    """Input map height, read once from the operand's own extents."""

    var w_lim: Int
    """Input map width, read once from the operand's own extents."""

    var s_extent: Int
    """Filter width; the modulus the carried tap column wraps on."""

    var c_extent: Int
    """Input channels; the modulus the carried channel base wraps on."""

    var tap_row_jump: Int
    """`(W - S) * C_in`: what `tap_off` adds when the tap row steps."""

    var pixel_wrap: Int
    """`(W - W_out + 1) * C_in`: what `base` adds when a row of pixels wraps."""

    var r: Int
    """Carried k-state: the current tap row. Only the halo test reads it."""

    var s: Int
    """Carried k-state: the current tap column."""

    var c0: Int
    """Carried k-state: the current channel base, a multiple of `GEMV_SIMD_W`."""

    var tap_off: Int
    """Carried k-state: element offset of this k-vector from the row's tap."""

    @always_inline
    def __init__(
        out self,
        img: MatmulImage[Self.o],
        S: Int,
        pad_h: Int,
        pad_w: Int,
        W_out: Int,
    ):
        """Binds the input map and seeds the carried k-state at `k = 0`.

        Args:
            img: The `(H, W, C_in)` input map for one batch index.
            S: Filter width.
            pad_h: Rows of zero padding above the input map.
            pad_w: Columns of zero padding left of the input map.
            W_out: Output width.
        """
        var h_lim = img.layout.shape[0]()
        var w_lim = img.layout.shape[1]()
        var c_extent = img.layout.shape[2]()
        self.img = img.reshape(Coord(h_lim * w_lim * c_extent))
        self.w_out = W_out
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.h_lim = h_lim
        self.w_lim = w_lim
        self.s_extent = S
        self.c_extent = c_extent
        self.tap_row_jump = (w_lim - S) * c_extent
        self.pixel_wrap = (w_lim - W_out + 1) * c_extent
        self.r = 0
        self.s = 0
        self.c0 = 0
        self.tap_off = 0

    @always_inline
    def anchor(self, m: Int) -> MatmulRowAnchor:
        """Decomposes row `m` into its output pixel's top-left input tap.

        The one software divide the gather pays. `_conv_walk` calls it once per
        M-tile per N step and steps from there, so it is not per microtile.

        Args:
            m: The matmul row, a linearized output pixel.

        Returns:
            `(h_out - pad_h, w_out - pad_w)` and its linear form.
        """
        var h_out = m // self.w_out
        var row = h_out - self.pad_h
        var col = m - h_out * self.w_out - self.pad_w
        return MatmulRowAnchor(
            row=row, col=col, base=(row * self.w_lim + col) * self.c_extent
        )

    @always_inline
    def anchor_next(self, prev: MatmulRowAnchor, m: Int) -> MatmulRowAnchor:
        """The next output pixel, by add and carry rather than a second divide.

        Args:
            prev: The previous pixel's anchor.
            m: The matmul row; unused, deriving it would be the divide this
                method exists to avoid.

        Returns:
            The anchor of output pixel `m`.
        """
        var col = prev.col + 1
        if col == self.w_out - self.pad_w:
            return MatmulRowAnchor(
                row=prev.row + 1,
                col=-self.pad_w,
                base=prev.base + self.pixel_wrap,
            )
        return MatmulRowAnchor(
            row=prev.row, col=col, base=prev.base + self.c_extent
        )

    @always_inline
    def advance(mut self):
        """Steps `(r, s, c0)` and the linear `tap_off` on by one k-vector."""
        self.c0 += GEMV_SIMD_W
        self.tap_off += GEMV_SIMD_W
        if self.c0 == self.c_extent:
            self.c0 = 0
            self.s += 1
            if self.s == self.s_extent:
                self.s = 0
                self.tap_off += self.tap_row_jump
                comptime if Self.halo:
                    self.r += 1

    @always_inline
    def load(
        self, anchor: MatmulRowAnchor, m: Int, k_off: Int
    ) -> SIMD[GEMV_DTYPE, GEMV_SIMD_W]:
        """One `GEMV_SIMD_W`-lane A vector, gathered from the map or zero.

        Args:
            anchor: The pixel's top-left input tap, from `anchor()`.
            m: The matmul row; unused, `anchor` is its prebaked form.
            k_off: First k index of the vector; unused, the carried state
                already names the same position.

        Returns:
            The tap's `GEMV_SIMD_W` channels, or zero outside the input map.
        """
        comptime if Self.halo:
            var h = anchor.row + self.r
            var w = anchor.col + self.s
            if not (0 <= h < self.h_lim and 0 <= w < self.w_lim):
                return SIMD[GEMV_DTYPE, GEMV_SIMD_W](0)
        return self.img.load[GEMV_SIMD_W](Coord(anchor.base + self.tap_off))


@always_inline
def _conv_window_inside(
    anchor: MatmulRowAnchor, rows: Int, R: Int, S: Int, H: Int, W: Int
) -> Bool:
    """Whether `rows` consecutive output pixels from `anchor` all read in bounds.

    The predicate that lets a microtile drop the halo test: the first pixel's
    top-left tap is inside the map and the last one's bottom-right tap still is,
    which for a stride-1 window is `0 <= row <= H - R` and `0 <= col` and
    `col + rows - 1 <= W - S`. The column bound also rules out a wrap onto the
    next output row, because `W_out - 1 - pad_w` is `W + pad_w - S`, at least
    `W - S`.

    Sound, not exact: a span that wraps is refused even where its taps are all
    in bounds, which needs a filter with no column halo. `mm_conv_window_inside`
    in `matmul_kernel.mojo` is the same predicate and the host test brute-forces
    it against the full tap set — a False positive there is an out-of-bounds read
    here, so that gate is on soundness and prints the conservatism.

    Args:
        anchor: The first pixel's top-left tap, `(h_out - pad_h, w_out - pad_w)`.
        rows: Consecutive output pixels the microtile covers.
        R: Filter height.
        S: Filter width.
        H: Input map height.
        W: Input map width.

    Returns:
        True when every tap of every one of those pixels is inside the map.
    """
    return (
        0 <= anchor.row <= H - R
        and 0 <= anchor.col
        and anchor.col + rows - 1 <= W - S
    )


@always_inline
def _matmul_microtile[
    MR_c: Int, NR_c: Int, R: MatmulAReader
](
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    var reader: R,
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    m0: Int,
    n0: Int,
    N: Int,
    K: Int,
    k_vecs: Int,
):
    """An `MR_c` x `NR_c` block of independent K-contiguous fp32 dot products.

    Output `(m0+mr, n0+nr)` is `C[(m0+mr)*N + n0+nr]`, the dot product of A row
    `m0+mr` with B row `n0+nr`. The `MR_c` A-vectors and `NR_c` B-vectors are
    loaded once per k-vector and reused across the tile. Callers pass only fully
    in-bounds tiles; the store is scalar, so no masked vector store reaches `C`.

    `MATMUL_KU` k-steps' operands are loaded before any of them is consumed. An
    HVX vector is exactly one cache line, so the k direction has no spatial
    reuse: every load is its own line and its own potential miss. Issuing
    `MATMUL_KU` steps' worth together puts that many misses in flight at once,
    which is the only lever this loop has on the ~51-57 stall cycles it carries
    per k-step.

    The accumulation order is untouched: accumulator `(mr, nr)` still takes
    k-step `kv` before `kv + 1`, with the same operands, so the result is
    bit-identical to the one-step-at-a-time nest. `test_matmul_kernel.mojo`
    gates that as exact equality.

    A is a `MatmulAReader` rather than a pointer: the plain matmul reads it out
    of the `[M, K]` activation, the fused convolution gathers it from an NHWC
    input map, and nothing else about this body changes between them. The reader
    is taken BY VALUE so each microtile starts from the caller's `k = 0` seed.
    """
    var anchor: StaticTuple[MatmulRowAnchor, MR_c] = {
        fill = MatmulRowAnchor(row=0, col=0)
    }
    comptime if R.anchors_rows:
        anchor[0] = reader.anchor(m0)
        comptime for mr in range(1, MR_c):
            anchor[mr] = reader.anchor_next(anchor[mr - 1], m0 + mr)

    var acc: StaticTuple[SIMD[DType.float32, GEMV_SIMD_W], MR_c * NR_c] = {
        fill = SIMD[DType.float32, GEMV_SIMD_W](0)
    }
    var kv = 0
    while kv + MATMUL_KU <= k_vecs:
        var av: StaticTuple[SIMD[GEMV_DTYPE, GEMV_SIMD_W], MR_c * MATMUL_KU] = {
            fill = SIMD[GEMV_DTYPE, GEMV_SIMD_W](0)
        }
        var bv: StaticTuple[SIMD[GEMV_DTYPE, GEMV_SIMD_W], NR_c * MATMUL_KU] = {
            fill = SIMD[GEMV_DTYPE, GEMV_SIMD_W](0)
        }
        comptime for ku in range(MATMUL_KU):
            var k_off = (kv + ku) * GEMV_SIMD_W
            comptime for mr in range(MR_c):
                av[ku * MR_c + mr] = reader.load(anchor[mr], m0 + mr, k_off)
            comptime for nr in range(NR_c):
                bv[ku * NR_c + nr] = B.load[GEMV_SIMD_W]((n0 + nr) * K + k_off)
            reader.advance()
        comptime for ku in range(MATMUL_KU):
            comptime for mr in range(MR_c):
                comptime for nr in range(NR_c):
                    acc[mr * NR_c + nr] = _fmac_sf_hf(
                        acc[mr * NR_c + nr],
                        av[ku * MR_c + mr],
                        bv[ku * NR_c + nr],
                    )
        kv += MATMUL_KU

    # The k-steps an unrolled group does not cover, one at a time. A K that is
    # not a multiple of `MATMUL_KU * 64` reaches here; every Klein K is, so this
    # is the small-shape path rather than the hot one.
    while kv < k_vecs:
        var k_off = kv * GEMV_SIMD_W
        var av1: StaticTuple[SIMD[GEMV_DTYPE, GEMV_SIMD_W], MR_c] = {
            fill = SIMD[GEMV_DTYPE, GEMV_SIMD_W](0)
        }
        var bv1: StaticTuple[SIMD[GEMV_DTYPE, GEMV_SIMD_W], NR_c] = {
            fill = SIMD[GEMV_DTYPE, GEMV_SIMD_W](0)
        }
        comptime for mr in range(MR_c):
            av1[mr] = reader.load(anchor[mr], m0 + mr, k_off)
        comptime for nr in range(NR_c):
            bv1[nr] = B.load[GEMV_SIMD_W]((n0 + nr) * K + k_off)
        reader.advance()
        comptime for mr in range(MR_c):
            comptime for nr in range(NR_c):
                acc[mr * NR_c + nr] = _fmac_sf_hf(
                    acc[mr * NR_c + nr], av1[mr], bv1[nr]
                )
        kv += 1

    comptime for mr in range(MR_c):
        comptime for nr in range(NR_c):
            C[(m0 + mr) * N + (n0 + nr)] = _reduce_add(acc[mr * NR_c + nr])[
                0
            ].cast[GEMV_DTYPE]()


@always_inline
def _matmul_tile_rows(K: Int, mc_bytes: Int) -> Int:
    """A-rows a `mc_bytes` activation panel holds: a positive multiple of `MR`.

    Rounding down to `MATMUL_MR` is load-bearing, not cosmetic. Every tile then
    starts on a multiple of `MATMUL_MR`, so the last tile's full-tile bound
    `m0 + ((M - m0) // MR) * MR` equals the untiled `(M // MR) * MR`, and the
    rows that fall to the single-row remainder path are the same rows as before
    the tiling — which is what makes the retile bit-exact rather than merely
    correct. A budget too small for even one `MATMUL_MR`-row tile rounds up to
    `MATMUL_MR` rather than stalling.
    """
    var rows = (
        mc_bytes // (K * size_of[GEMV_DTYPE]()) // MATMUL_MR
    ) * MATMUL_MR
    if rows < MATMUL_MR:
        rows = MATMUL_MR
    return rows


@always_inline
def _conv_tile_rows(
    W: Int, C_in: Int, R: Int, W_out: Int, mc_bytes: Int
) -> Int:
    """Output rows a `mc_bytes` INPUT-MAP panel holds: a multiple of `MR`.

    `_matmul_tile_rows` sizes the M-tile in im2col bytes; a fused loader never
    materialises those, so what stays resident across one N sweep is the slice of
    the input map the tile's pixels read — `(rows_out + R - 1)` rows of
    `W * C_in` halves, the halo counted once rather than `R * S` times. The same
    budget therefore buys several times more output rows, which cuts the weight
    stripe's `ceil(M / mc)` re-read by the same factor. Twin of
    `mm_conv_tile_rows` in `matmul_kernel.mojo`, where the host test gates the
    multiple-of-`MR` property the retile argument rests on.

    `mc_bytes // in_row_bytes - (R - 1)` UNDERFLOWS the instant one input row
    is at or below `mc_bytes / R` — four Klein convolutions sit exactly at
    `mc_bytes / 2` with `R = 3`, where the subtraction goes negative rather
    than merely to zero. Left to the final `< MATMUL_MR` clamp alone, that
    negative row count survives the `* W_out` and the floor-to-`MATMUL_MR`
    rounding as a negative pixel count and is caught only by accident, at the
    smallest possible tile (4 pixels), on precisely the shapes where the
    weight re-read this sizing exists to amortize is most expensive. Clamping
    `in_rows` to at least one output row here — before it is scaled by
    `W_out` — makes that state impossible rather than relying on the pixel
    floor to paper over it: a tile is never sized finer than one full output
    row, no matter how far short the budget falls of holding even the
    filter's `R` input rows resident.
    """
    var in_row_bytes = W * C_in * size_of[GEMV_DTYPE]()
    var in_rows = mc_bytes // in_row_bytes - (R - 1)
    if in_rows < 1:
        in_rows = 1
    var rows = (in_rows * W_out // MATMUL_MR) * MATMUL_MR
    if rows < MATMUL_MR:
        rows = MATMUL_MR
    return rows


@always_inline
def _matmul_stripe(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    A: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    n_start: Int,
    n_end: Int,
    M: Int,
    N: Int,
    K: Int,
    mc_bytes: Int,
):
    """`C[:, n_start:n_end] = A @ B[n_start:n_end]^T`, register-blocked, fp16.

    `[n_start, n_end)` is the N stripe this compute thread owns. Full
    `MATMUL_MR` x `MATMUL_NR` tiles take the register-blocked path; the `M % MR`
    row edge and the `stripe % NR` column edge fall to single-row / single-column
    microtiles — the scalar-remainder idiom — so a non-64-aligned N tail needs no
    masked vector store. `K` must be a multiple of 64.

    An M-tile of `_matmul_tile_rows(K, mc_bytes)` rows sits OUTSIDE the N sweep,
    so the activation rows a sweep touches stay within a bounded budget and each
    row crosses DRAM once per M-tile instead of once per N-pair. Reordering
    which independent output is computed when cannot change any output: each
    `C[m, n]` is still produced by exactly one microtile invocation accumulating
    the same k-vectors in the same order into the same fp32 register pair, and
    the `MATMUL_MR`-aligned tile start keeps the remainder-path row set
    identical. `test_matmul_kernel.mojo` gates that as exact equality, not a
    tolerance.

    The next tile's operand rows are `l2fetch`ed while the current tile
    computes. A cache hint is not part of the nest: it leaves the visiting order
    and the arithmetic alone, so the result stays bit-identical.
    """
    var k_vecs = K // GEMV_SIMD_W
    var mc = _matmul_tile_rows(K, mc_bytes)
    var n_full = n_start + ((n_end - n_start) // MATMUL_NR) * MATMUL_NR
    # The 16-bit width field of the fetch descriptor cannot describe a row this
    # wide, and truncating it would fetch the wrong region.
    var row_bytes = K * size_of[GEMV_DTYPE]()
    var can_prefetch = row_bytes <= 0xFFFF
    var reader = MatmulDenseReader(A, K)

    var m0 = 0
    while m0 < M:
        var m_stop = m0 + mc
        if m_stop > M:
            m_stop = M
        var m_full = m0 + ((m_stop - m0) // MATMUL_MR) * MATMUL_MR

        var n = n_start
        while n < n_full:
            comptime if MATMUL_PF_B_ROWS > 0:
                var pf_n = n + MATMUL_PF_B_ROWS
                # A fetch running past the stripe is dropped rather than
                # clamped: the descriptor then stays loop-invariant, and the
                # rows it would have covered are the stripe's last few.
                if can_prefetch and pf_n + MATMUL_PF_B_ROWS <= n_end:
                    _l2fetch_rows(B + pf_n * K, row_bytes, MATMUL_PF_B_ROWS)
            var m = m0
            while m < m_full:
                comptime if MATMUL_PF_A_ROWS > 0:
                    var pf_m = m + MATMUL_PF_A_ROWS
                    if can_prefetch and pf_m + MATMUL_PF_A_ROWS <= m_stop:
                        _l2fetch_rows(A + pf_m * K, row_bytes, MATMUL_PF_A_ROWS)
                _matmul_microtile[MATMUL_MR, MATMUL_NR](
                    C, reader.copy(), B, m, n, N, K, k_vecs
                )
                m += MATMUL_MR
            while m < m_stop:
                _matmul_microtile[1, MATMUL_NR](
                    C, reader.copy(), B, m, n, N, K, k_vecs
                )
                m += 1
            n += MATMUL_NR
        while n < n_end:
            var m = m0
            while m < m_full:
                _matmul_microtile[MATMUL_MR, 1](
                    C, reader.copy(), B, m, n, N, K, k_vecs
                )
                m += MATMUL_MR
            while m < m_stop:
                _matmul_microtile[1, 1](C, reader.copy(), B, m, n, N, K, k_vecs)
                m += 1
            n += 1
        m0 += mc


@always_inline
def _conv_walk[
    NR_c: Int, RB: MatmulAReader, RI: MatmulAReader
](
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    border: RB,
    interior: RI,
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    R: Int,
    S: Int,
    H: Int,
    W: Int,
    m0: Int,
    m_full: Int,
    m_stop: Int,
    n: Int,
    N: Int,
    K: Int,
    k_vecs: Int,
):
    """One M-tile's rows for output channels `n ..< n + NR_c`, halo-split.

    The two readers compute the same A vector wherever both are valid — the
    interior one IS the border one with a provably-true test deleted — so which
    microtile gets which cannot move a result. That is what keeps the fused arm
    BIT-equal to a materialised im2col of the same k order, which
    `test_conv_im2col_kernel.mojo` gates on the host twin.

    The pixel is carried across the walk by `anchor_next` rather than recovered
    from `m`, so the predicate costs compares and no division, and the walk pays
    ONE `__hexagon_divsi3` per M-tile per N step instead of one per microtile.

    Only the full `MATMUL_MR`-row tiles are split. The `M % MR` remainder rows
    keep the halo reader unconditionally: at most `MR - 1` rows of a tile, and
    giving them their own instantiations would double this kernel's code for no
    measurable work.

    Parameters:
        NR_c: B-rows in these microtiles.
        RB: The halo-testing reader.
        RI: The reader with the halo test compiled out.

    Args:
        C: The `[M, N]` output.
        border: Reader for microtiles whose window can leave the map.
        interior: Reader for microtiles whose window provably cannot.
        B: The `[N, K]` filter.
        R: Filter height.
        S: Filter width.
        H: Input map height.
        W: Input map width.
        m0: First row of this M-tile.
        m_full: One past the last row covered by a full `MATMUL_MR`-row tile.
        m_stop: One past the last row of this M-tile.
        n: First output channel these microtiles own.
        N: Output channel count, the row stride of `C`.
        K: Reduction length.
        k_vecs: `K / GEMV_SIMD_W`.
    """
    var cur = border.anchor(m0)
    var m = m0
    while m < m_full:
        if _conv_window_inside(cur, MATMUL_MR, R, S, H, W):
            _matmul_microtile[MATMUL_MR, NR_c](
                C, interior.copy(), B, m, n, N, K, k_vecs
            )
        else:
            _matmul_microtile[MATMUL_MR, NR_c](
                C, border.copy(), B, m, n, N, K, k_vecs
            )
        comptime for _ in range(MATMUL_MR):
            cur = border.anchor_next(cur, 0)
        m += MATMUL_MR
    while m < m_stop:
        _matmul_microtile[1, NR_c](C, border.copy(), B, m, n, N, K, k_vecs)
        m += 1


@always_inline
def _conv_stripe(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    img: MatmulImage[MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    n_start: Int,
    n_end: Int,
    M: Int,
    N: Int,
    K: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    W_out: Int,
    mc_bytes: Int,
):
    """One N stripe of a stride-1 convolution over a FUSED online im2col.

    The same nest `_matmul_stripe` runs, given `MatmulIm2colReader` instead of
    `MatmulDenseReader`, and differing in exactly the three places the operand
    swap makes different: the M-tile is sized in input-map bytes; there is no
    A-row `l2fetch` because A is not a row of memory — the input map is read
    repeatedly by the tile's own pixels and is what the budget keeps resident;
    and the row walk chooses a reader per microtile (`_conv_walk`), which the
    dense arm never does. The B (weight) prefetch is unchanged.

    The walk is spelled here rather than taken from `_matmul_stripe` so that
    nest can stay byte-identical to the build that had no seam — the M4.6-L1
    record is that even provably dead code added to it moves the schedule, and a
    signed kernel whose machine code does not change owes no re-sign.
    """
    var k_vecs = K // GEMV_SIMD_W
    var W = img.layout.shape[1]()
    var C_in = img.layout.shape[2]()
    var mc = _conv_tile_rows(W, C_in, R, W_out, mc_bytes)
    var n_full = n_start + ((n_end - n_start) // MATMUL_NR) * MATMUL_NR
    var row_bytes = K * size_of[GEMV_DTYPE]()
    var can_prefetch = row_bytes <= 0xFFFF
    var border = MatmulIm2colReader(img, S, pad_h, pad_w, W_out)
    var interior = MatmulIm2colReader[halo=False](img, S, pad_h, pad_w, W_out)
    var H = img.layout.shape[0]()

    var m0 = 0
    while m0 < M:
        var m_stop = m0 + mc
        if m_stop > M:
            m_stop = M
        var m_full = m0 + ((m_stop - m0) // MATMUL_MR) * MATMUL_MR

        var n = n_start
        while n < n_full:
            comptime if MATMUL_PF_B_ROWS > 0:
                var pf_n = n + MATMUL_PF_B_ROWS
                if can_prefetch and pf_n + MATMUL_PF_B_ROWS <= n_end:
                    _l2fetch_rows(B + pf_n * K, row_bytes, MATMUL_PF_B_ROWS)
            _conv_walk[MATMUL_NR](
                C,
                border,
                interior,
                B,
                R,
                S,
                H,
                W,
                m0,
                m_full,
                m_stop,
                n,
                N,
                K,
                k_vecs,
            )
            n += MATMUL_NR
        while n < n_end:
            _conv_walk[1](
                C,
                border,
                interior,
                B,
                R,
                S,
                H,
                W,
                m0,
                m_full,
                m_stop,
                n,
                N,
                K,
                k_vecs,
            )
            n += 1
        m0 += mc


# ===--------------------------------------------------------------------------===
# `mo.matmul` on the MATRIX array (P5 M2.3)
#
# Hexagon v81's HMX unit is where this chip's advertised AI compute lives; the
# HVX path above is the vector unit. This section is what makes the array
# reachable from the kernel the device actually dispatches, behind selector 24
# and behind `MM_PATH_HMX`, which is OFF unless `build.sh HMX_MATMUL=1` defines
# it. With it off nothing below is called and the object is unchanged.
#
# What this owns is exactly the two things `matmul_bt_stripe_hmx` could not be
# written to do from the authoring seat: the VTCM residency its panels live in,
# and the HMX grant its accumulators live under. Everything else — the crouton
# layout, the packers, the five instruction words — is `hmx_crouton.mojo`.
#
# The grant protocol is the one that has actually returned a granted unit on
# this device, and it is copied from the fp16 probe tile that read EXACT off the
# array (`hmx_int8_shim.c:152-187`): `qshimHmxLock` (which acquires the shared
# VTCM context with `set_hmx_param` and `acquire_cached`, then takes the unit),
# `qshimHmxEnter` to raise the busy flag autonomous preemption watches, the
# work, `qshimHmxExit`, `qshimHmxUnlock` through whichever entry point granted
# it. Releasing through the wrong one leaves the unit held and poisons the
# device for every later run.
# ===--------------------------------------------------------------------------===

comptime MATMUL_HMX_SPACE = AddressSpace.SHARED
"""How VTCM residency is spelled here; there is no `AddressSpace.VTCM`."""

comptime MATMUL_PATH_HMX_RAN: UInt32 = 0x484D5831
"""'HMX1' in witness slot 7: the array computed this dispatch's output."""

comptime MATMUL_PATH_HMX_NO_VTCM: UInt32 = 2
"""The VTCM grant is absent or too small for the stripe's panels."""

comptime MATMUL_PATH_HMX_NO_GRANT: UInt32 = 3
"""`qshimHmxLock` refused on a context `qshimVtcmCtxHmxArmed` reports armed —
every fast-path and CRM lock entry point refused despite the unit being set up
for this thread. Distinct from `MATMUL_PATH_HMX_NO_GRANT_CTX`: this is the
"refused anyway" case, not "never armed to begin with"."""

comptime MATMUL_PATH_HMX_NO_GRANT_CTX: UInt32 = 4
"""`qshimHmxLock` refused, AND `qshimVtcmCtxHmxArmed() == 0` — the shared
`g_vtcm_ctx` was never armed in the first place. Structurally
near-unreachable: reaching this function's lock attempt at all already
required `_qshim_get_base_vtcm_addr`'s own acquire to have succeeded (the
earlier `MATMUL_PATH_HMX_NO_VTCM` check would have caught a failure there
first) — kept as a defensive check, not because a live path to it is known.
Word 13 carries `qshimHmxLock`'s lock3 attempt rc (P5 M2.4 order 066); word
14 carries the launch-wide HVX-release rc `npu_run_launch` records before
any participant exists (order 067); words 11/12 are stale from the
order-065 dedicated-context experiment, kept for dump continuity rather
than repurposed."""

comptime MATMUL_PATH_HMX_PREDICATE_MISMATCH: UInt32 = 5
"""The lock GRANTED — entered, ran, exited, unlocked cleanly — but
`matmul_bt_stripe_hmx`'s own shape recheck disagreed with this function's
caller's earlier `matmul_bt_hmx_accepts` and returned `False` before issuing
any MMA instruction. Both predicates are the same pure function of `(M, N, K,
n_start, n_end)`; if this code appears, the two calls saw different values for
one of them and that value-propagation path is where to look, not the lock."""

comptime MATMUL_PATH_HMX_NO_THREADS: UInt32 = 6
"""The launch spawned too few participants to host the pipeline's roles (P5
M4.2). The roles are derived from the participant COUNT rather than assumed to
be five, so this now means one participant only: a lone thread would be the
array thread with nobody to feed it."""

comptime MATMUL_PATH_HMX_SYNC_TIMEOUT: UInt32 = 7
"""A rendezvous hit its deadline and the stripe unwound (P5 M4.2 order 084).

Order 084 wedged a device for 24 minutes on the render's FIRST HMX dispatch. Two
causes, both fixed here, and this code is what a third would look like rather
than a hang: `qshimLaunchSyncWait` bounds every wait, so a participant that never
arrives costs 2 seconds and a witness code instead of a reboot. Reaching this
means the launch machinery did not deliver the participants it promised — read
`g_launch_count` against the threads that actually ran, not the kernel."""


comptime MATMUL_HMX_TRACE = get_defined_bool["SKYE_HMX_TRACE", False]()
"""Whether the array thread records its rendezvous index every step.

The per-dispatch phase words below are unconditional — three stores, and they
are what tells a post-mortem whether a stalled dispatch was inside the HMX lock
or inside a rendezvous. The per-STEP counter is not: it is one store per
rendezvous, up to ~1157 of them on the render's widest shape, so it is a build
define (`build.sh HMX_TRACE=1`) rather than something the hot path pays for."""

comptime MATMUL_WIT_PHASE = 15
"""Where the array thread is, so a dispatch that never returned can say so."""

comptime MATMUL_WIT_STEP = 16
"""Rendezvous the array thread last completed (`MATMUL_HMX_TRACE` only)."""

comptime MATMUL_WIT_ROLES = 17
"""`participants * 16 + feeders`: the role plan this dispatch actually used."""

comptime MATMUL_WIT_ARM = 18
"""Which arm of the stripe this skel actually ran, read back at the dispatch site.

The arm is a build-time `comptime`, so a host that scored a run against an int8
reference had nothing but the directory name to say an int8 skel was loaded --
and this port has three print-vs-dispatch incidents on the books. Word 18 is the
device's own answer: `MATMUL_ARM_W8A8` or `MATMUL_ARM_FP16`, plus the two bits
that change what the numbers mean.

Free by construction. Words 8-14 are the refusal report and this branch always
returns before the HVX nest that also starts at 8 (`mojo_matmul_f16`), 15-17 are
the array thread's own, and `NPU_WITNESS_WORDS` is 32."""

comptime MATMUL_ARM_FP16: UInt32 = 16
"""The shipping fp16 arm."""

comptime MATMUL_ARM_W8A8: UInt32 = 8
"""The integer arm: `B` is a quantized artefact and `C` is dequantized."""

comptime MATMUL_ARM_RECORD_LATE: UInt32 = 0x100
"""Set when `HMX_RECORD_LATE` built the mutant. A scored run with this bit set
must FAIL, and one with it clear must not."""

comptime MATMUL_WIT_GUARD = 19
"""Bytes of the arena's guard crouton an accumulator store overwrote.

0 is healthy. Nonzero says the store wrote past its readout tile, which is
order 132's corruption mechanism -- and because the guard is poisoned rather
than merely reserved, the COUNT is the overrun in bytes, i.e. a measurement of
the store form's true extent. Read it before reading any output: a run with
damage here has a requant record that may have been clobbered mid-sweep, and
its numbers mean nothing."""

comptime MATMUL_WIT_REC_SRC = 20
"""Record word 0 as the ARTEFACT holds it, read by the array thread.

Order 133 established that the readout is a uniform `q = 0`, which needs a zero
requant scale, and the guard then ruled out a store overrun as the cause. Two
possibilities survive and no aggregate can separate them: the record ARRIVES
zero (the artefact's record region is not where the kernel looks, or did not
upload), or it arrives correct and is zero by the time the store reads it.

This word is the SOURCE and `MATMUL_WIT_REC_DST` is the DESTINATION, sampled
after the copy on the same thread that made it. One dispatch tells them apart:
source zero is an addressing or staging fault and no barrier will fix it;
source good with destination zero is the requant file not holding what was
written to it."""

comptime MATMUL_WIT_REC_DST = 21
"""Record word 0 as the ARENA holds it, sampled after the copy. See above."""

comptime MATMUL_WIT_UNPACK_ARM = 22
"""Which arm of the crouton unpack seam ran, echoed by the seam itself.

`HMX_UNPACK_VEC`'s whole failure mode is a SILENT admission early-out:
`hmx_unpack_vec_admits` returning False leaves the scalar arm running, produces
bit-identical output, and looks exactly like a clean null. A probe-off A/B has
nothing but the wall to say the arm it is measuring ever executed, so this word
carries the seam's own three-way answer (`HMX_UNPACK_ARM_*`) rather than a
re-derivation from the build flags -- which would be
`patterns/the-host-prints-the-arm-it-planned-not-the-arm-that-ran` again.

FREE BY CONSTRUCTION, and read this before believing the note at the bottom of
this file that the witness is FULL at 32 words. That note is true of the HVX
NEST, which writes words 8..31 as its per-thread cycle bracket. This word is
written only on the HMX arm, and the HMX arm `return`s at the
`matmul_bt_hmx_accepts` branch in `mojo_matmul_f16` BEFORE reaching that nest --
the same mutual exclusion `MATMUL_WIT_ARM` documents a few lines up. On this arm
the live words are 0-7, the 8-14 refusal report and 15-21, so 22-31 are
unwritten and 22 is the next one. The exclusion is by construction, not by
convention: do not make the array arm fall through past that `return`."""

comptime MATMUL_ARM_EMIT_MMA: UInt32 = 0x200
"""Set when the MMA was ISSUED rather than modelled. `matmul_bt_stripe_hmx_step`
already asserts this at compile time; reporting it costs one bit and makes "the
array computed this" a device fact instead of a build inference."""

comptime MATMUL_ARM_CONV: UInt32 = 0x400
"""Set when the activation panels were gathered from an NHWC input map.

The conv arm's engagement echo, in the same word and for the same reason the
others are there: `conv_im2col_hmx` and `conv_im2col` take the same buffers and
the same ten scalars, so a host that dispatched one and read the other's timing
would have nothing but the selector number it MEANT to send to tell them apart.
This bit is the device's own answer, and a run whose per-conv table claims the
array while word 18 has it clear is reporting the vector unit's numbers."""

comptime MATMUL_PHASE_LOCK_ENTER: UInt32 = 1
"""Inside `qshimHmxLock`. THE UNBOUNDED ONE: `HAP_compute_res_hmx_lock` takes no
timeout, so a dispatch stuck here is stuck before it has reached a rendezvous the
bounded barrier could unwind, and `npu_run_launch` joins it forever. A witness
frozen at this value is order 088's stall, named."""

comptime MATMUL_PHASE_LOCK_DONE: UInt32 = 2
"""The lock returned, one way or the other."""

comptime MATMUL_PHASE_STEPPING: UInt32 = 3
"""In the step loop, so any stall from here IS rendezvous-bounded."""

comptime MATMUL_PHASE_DONE: UInt32 = 4
"""Released cleanly."""

comptime MATMUL_PHASE_UNWOUND: UInt32 = 5
"""A rendezvous hit its deadline and the grant was given back."""


@always_inline
def _matmul_hmx_sync(thread_idx: Int) -> Bool:
    """One rendezvous. True when every participant arrived, False to unwind.

    Bounded by construction (`qshimLaunchSyncWait`), and the return is checked at
    EVERY call site. An unchecked rendezvous is the wedge order 084 recorded.

    The two marks are what make the array's rendezvous share a MEASUREMENT
    rather than `1 - (its MMA share)`, which is all any instrument has reported
    so far. The barrier's own counters — which loop each wait ended in, and how
    many naps it made — come back beside them from `qshimLaunchSyncWait` itself.
    """
    hmx_phase_mark(thread_idx, HMX_PHASE_BARRIER)
    var arrived = (
        Int(external_call["qshimLaunchSyncWait", UInt32](UInt32(thread_idx)))
        == 0
    )
    hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
    return arrived


@always_inline
@always_inline
def _matmul_hmx_sweep[
    dtype: DType, act_conv: Bool = False
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    base: Int,
    plan: HmxStripePlan,
    thread_idx: Int,
    feeders: Int,
    conv: HmxConvGeom = HmxConvGeom(),
) -> Bool:
    """Carves the arena and runs every step of the stripe. False = timed out.

    Split out of `_matmul_stripe_hmx` for ONE reason: the arena's type carries
    the arm's operand type, and a `var` assigned in both branches of a
    `comptime if` must have ONE type. So the branch is on the CALL, the arena is
    created inside, and the schedule below is written once for both arms.

    The rendezvous discipline is unchanged and is the caller's: this returns
    False on a missed deadline and the caller unwinds, so the release still
    happens on every exit including this one.

    Parameters:
        dtype: The arm's crouton operand type.
        act_conv: Whether `A` is a convolution's NHWC input map. The conv arm
            is the same sweep with a different activation loader, so it rides
            this function rather than a copy of it -- the two-bodies trap the
            HVX conv already has one instance of.

    Args:
        C: The `(M, N)` output.
        A: The `(M, K)` activations, or the `(H, W * C_in)` input map.
        B: The `(N, K)` weights, or the W8A8 artefact under the integer arm.
        witness: Slot `MATMUL_WIT_STEP` under `MATMUL_HMX_TRACE`.
        base: The aligned VTCM base the arena is carved at.
        plan: The geometry every participant agreed on, for THIS arm.
        thread_idx: This thread's launch index.
        feeders: Threads that pack and unpack.
        conv: The convolution's geometry; inert unless `act_conv`.

    Returns:
        Whether every rendezvous completed inside its deadline.
    """
    var arena = hmx_stripe_arena_in[MATMUL_HMX_SPACE, dtype](
        UnsafePointer[
            UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
        ](unsafe_from_address=base),
        plan.k_tiles,
        plan.act_slots,
    )
    # Poisoned before any store, checked after the last one, on the thread that
    # owns the array -- so the check cannot race the stores it is measuring.
    if thread_idx >= feeders:
        hmx_guard_poison(arena.guard)
    # Seeded to the flag-off code so a dispatch whose every step declined to
    # unpack reports the arm the build selected rather than a zero the reader
    # would have to guess at.
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        matmul_bt_stripe_hmx_step[
            dtype=dtype, record_late=HMX_RECORD_LATE, act_conv=act_conv
        ](C, A, B, arena, plan, step, thread_idx, feeders, unpack_arm, conv)
        comptime if dtype == HMX_INT8_DTYPE:
            # Step 1 is block 0 phase 1, the first step whose array thread
            # programs a record (column tile 0). Sampling both ends of that
            # copy here is what separates "the record arrives zero" from "the
            # record is zero when the store reads it" -- order 133 left those
            # indistinguishable, and no aggregate over the output can split
            # them.
            if step == 1 and thread_idx >= feeders:
                witness.store(
                    MATMUL_WIT_REC_SRC,
                    hmx_int8_artefact_record(B, 0).load[1](Coord(0, 0)),
                )
                witness.store(
                    MATMUL_WIT_REC_DST, arena.record.load[1](Coord(0, 0))
                )
        if not _matmul_hmx_sync(thread_idx):
            return False
        comptime if MATMUL_HMX_TRACE:
            if thread_idx == feeders:
                witness.store(MATMUL_WIT_STEP, UInt32(step + 1))
    if thread_idx >= feeders:
        witness.store(MATMUL_WIT_GUARD, UInt32(hmx_guard_damage(arena.guard)))
    # Comptime-gated WITH the arm, so a flag-off object is byte-identical to its
    # base and the inertness gate is `cmp` rather than a claim -- the rule
    # `HMX_PHASE_PROBE` already holds itself to. The cost is that flag-off writes
    # nothing here, which is why the reader treats an untouched word as "this
    # skel predates the instrument" and not as a refusal.
    #
    # From ONE feeder rather than every one: the feeders all reach the same
    # verdict (admission is a property of `C`'s base and stride, not of a tile),
    # so four writers would agree while making the word look raced to anyone
    # reading it later. The array thread never unpacks, so it is not the writer.
    comptime if HMX_UNPACK_VEC:
        if thread_idx == 0:
            witness.store(MATMUL_WIT_UNPACK_ARM, unpack_arm)
    return True


def _matmul_stripe_hmx[
    act_conv: Bool = False
](
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    A: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
    thread_idx: Int,
    conv: HmxConvGeom = HmxConvGeom(),
) -> None:
    """Computes the whole `C[M, N]` on the matrix array, across the launch.

    Every participant calls this. The LAST of them holds the HMX grant for the
    whole dispatch and does nothing but multiply; the rest pack the crouton
    panels it reads and unpack the accumulator tiles it writes. The array is one
    resource and a release clears its accumulators, so the parallelism is in the
    staging around the grant, never in contending for it (P5 M4.2; P5 M2.3's
    single-threaded arrangement is `matmul_bt_stripe_hmx`).

    THE CONVOLUTION ARM IS THIS FUNCTION, not a sibling of it. Everything that
    is hard here -- the two thread-uniform refusals ahead of the first
    rendezvous, the array thread's lazy VTCM resolution published through slot
    7, the checked returns, the release on every exit -- is the grant discipline
    order 084 bought with a 24-minute device wedge, and it does not depend on
    where an activation panel came from. `act_conv` reaches only the loader.

    THE ROLES COME FROM THE PARTICIPANT COUNT, not from a constant. At the five
    every `enqueue_function` in `registrations.mojo` asks for, that is threads
    0-3 feeding and thread 4 on the array — the split order 082 measured, so its
    numbers carry over unchanged. At any smaller count the stripe degrades to
    fewer feeders rather than leaving a role unfilled, and at one participant it
    refuses. Hard-coding five is what turns a launch-configuration difference
    into a rendezvous nobody completes.

    THE RENDEZVOUS DISCIPLINE, rewritten by order 084, which wedged a device for
    24 minutes on the render's FIRST HMX dispatch:

    - Only two things are read before the first rendezvous, and both are
      identical on every thread by construction: the participant count (one
      release-stored word, fixed before fan-out) and the shape predicate (pure
      arithmetic on this kernel's own scalar arguments). Every other refusal
      moved BEHIND the rendezvous.
    - In particular the VTCM base did. `qshimGetBaseVtcmAddr` acquires the grant
      LAZILY and without a once-guard, so on the first dispatch of a process
      several threads race into `HAP_compute_res_acquire` and the losers get
      NULL. That made a "pure" refusal thread-divergent: some threads returned
      while others waited for them. It is now resolved on the array thread only,
      published through slot 7, and re-read by the rest after the rendezvous —
      when the shim's `g_vtcm_base != NULL` short-circuit makes it a stable read.
    - Every rendezvous return is checked. `_matmul_hmx_sync` is bounded, so a
      participant that never arrives costs a deadline and
      `MATMUL_PATH_HMX_SYNC_TIMEOUT`, not a reboot.
    - The grant is released on EVERY exit, including the unwind. A held unit
      poisons the device for every later run.

    Writes slot 7 itself rather than returning it: the feeders have to read the
    outcome, and slot 7 is already the word that says which unit computed the
    dispatch.

    Parameters:
        act_conv: Whether `A` is a convolution's NHWC input map.

    Args:
        C: The `(M, N)` output.
        A: The `(M, K)` activations, or under `act_conv` the `(H, W * C_in)`
            input map.
        B: The `(N, K)` transposed weights.
        witness: Slot 7 carries the path code; words 8-14 the lock-refusal
            detail P5 M2.4 added.
        M: Output rows.
        N: Output columns.
        K: The reduction extent.
        thread_idx: This thread's launch index.
        conv: The convolution's geometry; inert unless `act_conv`.
    """
    # The launch's own participant count, and the roles derived from it. One
    # release-stored word set before any participant exists, so every thread
    # reads the same number.
    var participants = Int(_qshim_query(QSHIM_QUERY_NUM_THREADS))
    var roles = hmx_stripe_roles(participants)
    var array_thread = roles.array_thread

    # --- The only two refusals ahead of the first rendezvous. Both are
    # --- identical on every thread, so they all take them and none arrives.
    if not roles.usable():
        if thread_idx == 0:
            witness.store(7, MATMUL_PATH_HMX_NO_THREADS)
        return
    if not matmul_bt_hmx_accepts(M, N, K, 0, N):
        if thread_idx == 0:
            witness.store(7, MATMUL_PATH_HMX_PREDICATE_MISMATCH)
        return

    # After the two thread-uniform refusals, so a dispatch this kernel declines
    # opens no span and reports all-zero buckets — which reads as "never
    # instrumented" rather than as a plausible zero beside real numbers.
    hmx_phase_open(thread_idx)
    hmx_phase_mark(thread_idx, HMX_PHASE_OPEN)

    if thread_idx == array_thread:
        witness.store(
            MATMUL_WIT_ROLES, UInt32(participants * 16 + roles.feeders)
        )
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_LOCK_ENTER)
        witness.store(7, _matmul_hmx_open(witness, M, N, K))
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_LOCK_DONE)
    if not _matmul_hmx_sync(thread_idx):
        _matmul_hmx_unwind(witness, thread_idx == array_thread)
        return
    if witness.load[1](7) != MATMUL_PATH_HMX_RAN:
        return

    # Safe to read on every thread now: the array thread's own resolution
    # published `g_vtcm_base`, and the rendezvous ordered it ahead of this.
    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    var base = arena.base
    var plan = hmx_stripe_plan(M, N, K, arena.budget)
    comptime if HMX_INT8 and not act_conv:
        plan = hmx_stripe_plan[HMX_INT8_DTYPE](M, N, K, arena.budget)

    # Built once for the launch, not per stripe: the resolve-once rule. Under
    # `act_conv`, `A` is the input map rather than an `[M, K]` -- the ONE line
    # of this function that reads the parameter, because the extents differ and
    # nothing else does.
    var Ct = MMTile(ptr=C, layout=row_major(Coord(M, N)))
    var a_rows = M
    var a_cols = K
    comptime if act_conv:
        a_rows = conv.h_lim
        a_cols = conv.w_lim * conv.c_extent
    var At = MMTile(ptr=A, layout=row_major(Coord(a_rows, a_cols)))
    var Bt = MMTile(ptr=B, layout=row_major(Coord(N, K)))

    if thread_idx == array_thread:
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_STEPPING)
        # Beside the arm branch below rather than anywhere else, so the word and
        # the `comptime if` that picks the arm cannot drift apart.
        var arm = MATMUL_ARM_FP16
        comptime if HMX_INT8 and not act_conv:
            arm = MATMUL_ARM_W8A8
        comptime if HMX_RECORD_LATE:
            arm |= MATMUL_ARM_RECORD_LATE
        comptime if HMX_EMIT_MMA:
            arm |= MATMUL_ARM_EMIT_MMA
        comptime if act_conv:
            arm |= MATMUL_ARM_CONV
        witness.store(MATMUL_WIT_ARM, arm)
    hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)

    # THE ONE PLACE THE ARM IS CHOSEN. Two literal dtypes under a comptime
    # condition rather than one conditional alias: a parameter expression built
    # from a conditional never folds in a signature position, so an
    # `HMX_ARM_DTYPE`-style alias would leave every panel type unconvertible to
    # either arm's. `HMX_INT8` is concrete here, so the arm not built is not
    # compiled and the flag-off object is byte-identical to the base.
    #
    # The conv arm is fp16 on EVERY build, int8 included: an integer
    # convolution would have to quantize inside the gather, which is a pack
    # policy this loader does not carry, and a W8A8 skel silently running a
    # crouton-fp16 conv is the mismatch class #366 recorded. `and not act_conv`
    # is what makes that a compile-time fact rather than a runtime hope.
    var ran = False
    comptime if HMX_INT8 and not act_conv:
        ran = _matmul_hmx_sweep[HMX_INT8_DTYPE](
            Ct, At, Bt, witness, base, plan, thread_idx, roles.feeders
        )
    else:
        ran = _matmul_hmx_sweep[GEMV_DTYPE, act_conv=act_conv](
            Ct, At, Bt, witness, base, plan, thread_idx, roles.feeders, conv
        )
    if not ran:
        _matmul_hmx_unwind(witness, thread_idx == array_thread)
        return

    if thread_idx == array_thread:
        external_call["qshimHmxExit", NoneType]()
        _ = external_call["qshimHmxUnlock", Int32]()
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_DONE)


@always_inline
def _matmul_hmx_unwind(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    is_array_thread: Bool,
):
    """Gives the unit back after a rendezvous hit its deadline.

    The release is the load-bearing half: a grant left held survives this
    dispatch and refuses every later one, which is a poisoned device rather than
    a failed launch. Slot 7 is overwritten LAST so a reader can tell a stripe
    that unwound from one that never got the grant.

    Marks the LAUNCH failed before either, on every participant rather than on
    the array thread alone: this stripe leaves part of `C` unwritten, and the
    witness is the only record of that until the trampoline turns this mark into
    a nonzero RPC status (`mojo_npu_launch.h`). A host that reads no witnesses
    would otherwise be told the dispatch succeeded.
    """
    external_call["mojo_npu_launch_fail", NoneType]()
    if not is_array_thread:
        return
    if witness.load[1](7) == MATMUL_PATH_HMX_RAN:
        external_call["qshimHmxExit", NoneType]()
        _ = external_call["qshimHmxUnlock", Int32]()
    witness.store(7, MATMUL_PATH_HMX_SYNC_TIMEOUT)
    witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_UNWOUND)


@always_inline
def _matmul_hmx_open(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
) -> UInt32:
    """Resolves residency and takes the grant, on the array thread ALONE.

    Every step here touches process-wide lazy state — `qshimGetBaseVtcmAddr`
    acquires the VTCM grant on first use with no once-guard, and the HMX lock is
    a single unit — so running it on one thread is not an optimisation, it is
    what keeps the outcome a single fact the rendezvous can broadcast (P5 M4.2
    order 084).

    Returns:
        `MATMUL_PATH_HMX_RAN`, or the refusal that stopped it.
    """
    # P5 M2.4 order 066: the shared VTCM window (`_qshim_get_base_vtcm_addr`/
    # `_qshim_get_vtcm_size`), matching `qshimHmxLock`'s revert to `g_vtcm_ctx`
    # -- 065's dedicated-context experiment (`hmx_ctx_acquire`,
    # `_qshim_get_hmx_vtcm_addr`) refused too, so this stripe's scratch
    # addresses the same window every other kernel in this file uses, under the
    # same lock the granting probe uses. The dedicated-window wrappers stay
    # defined (below `_qshim_get_base_vtcm_addr`) for the record, unused here.
    # Only the BASE is aligned, and the alignment comes out of the budget so the
    # plan never sizes itself into bytes the alignment already spent -- both
    # rules now live in `StepArena.at` rather than being repeated here and after
    # the rendezvous.
    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    if not arena.usable():
        return MATMUL_PATH_HMX_NO_VTCM

    var plan = hmx_stripe_plan(M, N, K, arena.budget)
    if plan.act_slots == 0:
        return MATMUL_PATH_HMX_NO_VTCM
    return _matmul_hmx_take_grant(witness)


@always_inline
def _matmul_hmx_take_grant(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
) -> UInt32:
    """Takes the HMX unit for the whole dispatch. Returns the path code.

    Runs on the array thread only, before the first rendezvous. On success the
    busy flag autonomous preemption watches stays raised until
    `_matmul_stripe_hmx` exits — held across every barrier, which is the
    hold-don't-cycle rule a release clearing the accumulators forces (KB
    `known-limitations/ondevice-hmx-lock-protocol`) and what ggml-hexagon's
    persistent array thread does for the same reason.
    """
    if Int(external_call["qshimHmxLock", Int32]()) != 0:
        # order 060 found code 3 stable across 35 dispatches with the OLD
        # words 8/9 (acquire_cached rc, execute-enabled) both reading 0 --
        # tautological on this branch (armed-context refusal implies
        # acquire_cached already read 0) and a build-time label that gates
        # nothing on this call path, respectively. Neither could name WHICH
        # of qshimHmxLock's own entry points refused. Word 8 the non-blocking
        # try-lock's own rc, word 9 the blocking qurt lock2's rc (-99 if
        # try-lock never reported obtainable), word 10 the final plain
        # HAP_compute_res_hmx_lock's rc.
        #
        # order 062 measured word 10 == 4 (QURT_EINVALID) on the shared
        # context, qshimVtcmCtxHmxArmed reporting it armed. #278 (063) stacked
        # a second acquire_cached on the still-live first one (refused with
        # an undocumented code, 10; word 10 stayed 4 -- acquire_cached is not
        # idempotent). #280 (064) cycled instead (release, fresh acquire,
        # same locking thread) -- both halves read EOK, word 10 STILL 4:
        # freshness and thread identity are both ruled out as the
        # differentiator by that result. order 065 tried a DEDICATED context
        # (`g_hmx_ctx`, `hmx_ctx_acquire`) instead of the shared `g_vtcm_ctx`
        # -- it refused too, ruling out the context object. Word 11/12
        # (context acquire / acquire_cached rc) are stale from that attempt;
        # `qshimHmxLock` reverted to `g_vtcm_ctx` at order 066 (below), so
        # both now read whatever `g_vtcm_ctx`'s own acquire/acquire_cached
        # last reported -- not a per-attempt reading, kept for continuity
        # with #278/#280's dumps rather than repurposed to something new.
        #
        # order 066: `qshimHmxLock` now runs `qshimHmxPr42Probe`'s exact
        # choreography on that same shared context (lock4 -> lock3 -> plain
        # lock, acquire timeout 1000000 not 0) -- the two never-varied deltas
        # 065's side-by-side turned up. Word 13 is the new lock3 attempt's
        # own rc; -99 means the plain-lock branch (which runs unconditionally
        # after it, success or refusal) was never reached, which cannot
        # happen once lock4 has already refused, so -99 here would itself be
        # a bug in this dump rather than in the lock.
        witness.store(
            8, UInt32(Int(external_call["qshimHmxTryRc", Int32]()) & 0xFFFF)
        )
        witness.store(
            9, UInt32(Int(external_call["qshimHmxQurtRc", Int32]()) & 0xFFFF)
        )
        witness.store(
            10,
            UInt32(Int(external_call["qshimHmxFullLockRc", Int32]()) & 0xFFFF),
        )
        witness.store(
            11,
            UInt32(Int(external_call["qshimHmxReleaseRc", Int32]()) & 0xFFFF),
        )
        witness.store(
            12,
            UInt32(Int(external_call["qshimHmxBracketRc", Int32]()) & 0xFFFF),
        )
        witness.store(
            13,
            UInt32(Int(external_call["qshimHmxLock3Rc", Int32]()) & 0xFFFF),
        )
        # order 067: `npu_run_launch` now releases this launch's calling
        # thread's HVX context, unconditionally, before any participant
        # (including this stripe's own locking thread) exists
        # (`mojo_npu_imp.c`, `qshimHvxReleaseAtLaunch` in `qshim_qurt.c`) --
        # the mechanism a live KB gap (`known-limitations/ondevice-hmx-lock-
        # protocol.md`) flagged before order 060: the FastRPC worker thread
        # this skel reuses across every dispatch can carry an implicitly
        # acquired HVX context from an earlier, different kernel into this
        # one's HMX attempt. Word 14 is that release's own rc, masked to the
        # low byte per `qurt_hvx_set`'s documented return shape; -99 means
        # `qshimHvxReleaseAtLaunch` was never reached this dispatch, which
        # would itself be a bug (it runs before any participant, so every
        # dispatch reaching this witness store already ran it).
        witness.store(
            14,
            UInt32(
                Int(external_call["qshimHvxLaunchReleaseRc", Int32]()) & 0xFF
            ),
        )
        if Int(external_call["qshimVtcmCtxHmxArmed", Int32]()) == 0:
            return MATMUL_PATH_HMX_NO_GRANT_CTX
        return MATMUL_PATH_HMX_NO_GRANT
    external_call["qshimHmxEnter", NoneType]()
    return MATMUL_PATH_HMX_RAN


@export
def mojo_matmul_f16(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    A: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
) abi("C"):
    """Computes `C[M, N] = A[M, K] @ B[N, K]^T` on the launch's threads.

    The batched/tiled `gemv`: many output rows per dispatch. `B` is stored N x K
    (transposed), so output `(m, n)` is the fp32 dot product of A row `m` with B
    row `n`, accumulated 64 lanes at a time and folded to a scalar. The N output
    columns divide across the four compute threads by launch index; the fifth
    (DMA) thread takes none of THIS split — the split the GEMV makes over its N
    output rows, here every thread computing all M rows for its column stripe.
    `K` must be a multiple of 64, the decode-matmul contract.

    On an HMX build the fifth thread is not idle: it is where the whole
    stripe's HMX grant runs (P5 M2.4), specifically because it is the one
    thread this kernel's own body never gives HVX work to — "a thread with
    HVX enabled cannot have HMX enabled" (`qurt_hmx.h:184`), which every
    compute thread below risks (the vector-path fallback needs HVX) and this
    one structurally cannot. P5 M4.2 gives the other four threads work on that
    build too: they stage the crouton panels the array reads and write back the
    tiles it produces, so all five participate and only one multiplies.

    `witness` reports what the output cannot, as the GEMV's does: the SIMD width
    the kernel compiled at, the fp32 accumulator width, the participant count the
    shim reports, and the RTOS thread each launch index ran on. Slot 7 carries
    which UNIT computed the dispatch — untouched for the vector path,
    `MATMUL_PATH_HMX_RAN` when the matrix array did, and a refusal code when an
    HMX build could not run and wrote nothing. Reading it is not optional on an
    HMX build: it is the only thing that distinguishes the two units' numbers.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(GEMV_SIMD_W))
        witness.store(1, UInt32(SIMD[DType.float32, GEMV_SIMD_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    # P5 M2.4: bounded to compute threads only. The launch reports 5
    # participants (slot 2) against `NUM_COMPUTE_THREADS == 4` -- a fifth,
    # DMA thread that "takes none" per this docstring -- and an unbounded
    # store here put that fifth thread's `_qurt_thread_id()` at
    # `3 + 4 == 7`, racing the HMX arm's path code for the SAME slot. order
    # 060 did not observe the race (slot 7 read a clean 3 all three legs,
    # both arms), but arm A's slot 7 WAS a QuRT thread id that run, so it is
    # live, just not guaranteed to lose. Bounding this store is the fix.
    if thread_idx < Int(NUM_COMPUTE_THREADS):
        witness.store(3 + thread_idx, _qurt_thread_id())

    comptime if MM_PATH_HMX:
        # P5 M2.4 order 062: runs on `thread_idx == NUM_COMPUTE_THREADS` now,
        # not thread 0. That is the launch's "5th" thread -- the one the HVX
        # nest below never gives work to (it "takes none" of the N output
        # columns) -- so it is the one thread in this launch structurally
        # guaranteed to never execute an HVX instruction in this kernel's own
        # body. ggml-hexagon's (llama.cpp's) equivalent is a dedicated,
        # HVX-free worker thread (`hmx-queue.c`); this is that same shape
        # using the launch's own idle thread rather than spawning a new one --
        # no thread-creation primitive has any precedent in this codebase to
        # copy faithfully.
        #
        # One HMX unit per core, and a release clears its accumulators, so
        # ONE thread takes the whole output under a single grant. P5 M4.2
        # completes the copy of their topology: the other four threads now
        # feed that thread rather than idling, so ALL FIVE enter
        # `_matmul_stripe_hmx` and it owns the roles.
        #
        # The shape test is pure arithmetic on `M`, `N` and `K`, so every
        # thread reaches the same verdict with no barrier — which is what
        # lets a non-conforming shape drop to the HVX nest below with its
        # full four-thread split intact. That drop is the ONLY fallback: a
        # shape the array accepted and then could not run is refused
        # outright, because reporting the vector unit's numbers as the
        # matrix unit's is the one failure this path must not have. Witness
        # slot 7 says which happened, and stays at the host's seed when the
        # stripe went to HVX.
        #
        # `NPU_WITNESS_WORDS` is 32 (#270); per-launch timing comes from
        # `NPUDispatchRecorder`, not this witness, so this arm still reports no
        # cycle counters of its own. It DOES now use words 8-14 on a refusal
        # (P5 M2.4, `_matmul_stripe_hmx`) — safe precisely because this whole
        # branch always `return`s right after storing slot 7, so it can never
        # fall through to the HVX nest below, whose own per-thread cycle report
        # also starts at word 8. The two writers of word 8 are mutually
        # exclusive by construction, not by convention; do not make this arm
        # fall through past the `return` without moving one of them.
        if matmul_bt_hmx_accepts(M, N, K, 0, N):
            _matmul_stripe_hmx(C, A, B, witness, M, N, K, thread_idx)
            return

    if thread_idx >= Int(NUM_COMPUTE_THREADS):
        return

    # Each compute thread owns a contiguous slice of the N output columns, the
    # split the GEMV makes over its N output rows.
    var cols_per_thread = ceildiv(N, Int(NUM_COMPUTE_THREADS))
    var t_start = thread_idx * cols_per_thread
    if t_start >= N:
        return
    var t_end = t_start + cols_per_thread
    if t_end > N:
        t_end = N

    # Bracketed for the same reason the attention loop is: the k-loop's cost in
    # CYCLES has only ever been derived from a clock measured on microsecond
    # brackets, and the NSP boosts to 2112 MHz under sustained load. A matmul
    # dispatch runs hundreds of milliseconds, so its own clock has to be read
    # inside it rather than inherited.
    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()

    _matmul_stripe(C, A, B, t_start, t_end, M, N, K, MATMUL_MC_BYTES)

    var cycles = _read_upcycle() - cycles0
    var ticks = _read_utimer() - ticks0
    var report = witness + 8 + thread_idx * MHA_PROBE_STRIDE
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    # k-iterations this thread ran: its N stripe x M rows x K vectors, over the
    # register block, which is what a cycles-per-iteration figure divides by.
    report.store(
        4,
        UInt32(
            ((t_end - t_start) // MATMUL_NR)
            * (M // MATMUL_MR)
            * (K // GEMV_SIMD_W)
        ),
    )
    report.store(5, UInt32(K // GEMV_SIMD_W))


@export
def mojo_conv_im2col_f16(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    img: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    H_out: Int,
    W_out: Int,
) abi("C"):
    """Computes one stride-1 convolution as a matmul over a FUSED online im2col.

    `C[H_out * W_out, C_out]` from the NHWC input map `img[H, W, C_in]` and the
    b-transposed filter `B[C_out, R * S * C_in]`. It is `mojo_matmul_f16` with
    one thing changed — where the A operand comes from. The `[M, K]` im2col
    matrix is never built: no host gather, no `M x K_pad` device scratch, no copy
    across the transport. Both existing conv paths on this port materialise it,
    which at 1024x1024 puts one convolution's scratch over the transport's 2 GiB
    single-allocation cap, so this is a correctness requirement there and not
    only a speed one.

    Contract, checked by the dispatcher rather than here (a signed kernel has no
    way to refuse): stride 1, dilation 1, `C_in % 64 == 0`, and the filter in
    `k = (r * S + s) * C_in + c` order — channel FASTEST within a tap, which is
    what makes each A vector one contiguous 128-byte load. `C_in % 64 == 0` also
    makes `K = R * S * C_in` a multiple of 64, so unlike the matmul selector this
    one needs no K padding.

    The N output channels divide across the four compute threads exactly as
    `mojo_matmul_f16`'s do, and `witness` reports the same four words.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(GEMV_SIMD_W))
        witness.store(1, UInt32(SIMD[DType.float32, GEMV_SIMD_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    if thread_idx >= Int(NUM_COMPUTE_THREADS):
        return

    var cols_per_thread = ceildiv(C_out, Int(NUM_COMPUTE_THREADS))
    var t_start = thread_idx * cols_per_thread
    if t_start >= C_out:
        return
    var t_end = t_start + cols_per_thread
    if t_end > C_out:
        t_end = C_out

    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()

    _conv_stripe(
        C,
        MatmulImage[MutUnsafeAnyOrigin](
            ptr=img, layout=row_major(Coord(H, W, C_in))
        ),
        B,
        t_start,
        t_end,
        H_out * W_out,
        C_out,
        R * S * C_in,
        R,
        S,
        pad_h,
        pad_w,
        W_out,
        MATMUL_MC_BYTES,
    )

    var cycles = _read_upcycle() - cycles0
    var ticks = _read_utimer() - ticks0
    var report = witness + 8 + thread_idx * MHA_PROBE_STRIDE
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(
        4,
        UInt32(
            ((t_end - t_start) // MATMUL_NR)
            * ((H_out * W_out) // MATMUL_MR)
            * (R * S * C_in // GEMV_SIMD_W)
        ),
    )
    report.store(5, UInt32(R * S * C_in // GEMV_SIMD_W))


@export
def mojo_conv_im2col_hmx_f16(
    C: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    img: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    B: UnsafePointer[Scalar[GEMV_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    H_out: Int,
    W_out: Int,
) abi("C"):
    """`mojo_conv_im2col_f16` routed to the matrix array instead of HVX.

    Same buffers, same ten scalars, same answer to within the array's fold
    order -- and a SEPARATE SELECTOR, which is the point. Selector 36 is
    dispatched unconditionally from the graph path
    (`registrations.mojo:2872`), so an array arm reached through it would
    change the graph VAE's numerics with no gate in between; `NPUKernels.def`
    states that rows may be appended but never reordered, so an older signed
    skel refuses this one by name, which is this port's loud failure.

    THE ARM IS ECHOED, NOT ASSUMED. Witness word 18 carries `MATMUL_ARM_CONV`
    beside the fp16/int8 and issued/modelled bits, because this kernel and the
    HVX one take identical arguments: a host that meant to dispatch the array
    and got the vector unit has nothing else to tell them apart, and this port
    has five print-vs-dispatch incidents on the books.

    THE WEIGHT ARM IS NOT A CHOICE HERE. `_matmul_stripe_hmx` reads `B` per
    `HMX_WEIGHTS_NATIVE`, so on an array-native skel this selector's `B` MUST
    be a crouton artefact and on any other skel it must be the row-major
    `[C_out, R*S*C_in]` -- the same two-direction rule the registry enforces
    per weight (`npu_registry_load.h:154`) and the same mismatch class #366
    re-attributed the VAE failure to. The hand driver has no layout argument to
    carry, so the gate is the driver's own refusal to enable this arm on a skel
    whose weight layout it did not stage; there is nothing in a signed kernel
    that can check it.

    ORDERS 139 AND 142 MEASURED THIS ARM. It works and its value is bimodal, so
    the envelope belongs here rather than in a document nobody opens beside the
    code:

    - 34 of 34 eligible convs reach the array, image-level numerics pass, and
      the conv invocation is 1.166x overall against the HVX arm -- about 4.5 s
      of a ~96 s stage, not the ~40 s the vector unit owes.
    - The first 17 dispatches hold a flat 1.553x. The last 17 -- `M` climbing to
      262144 while `N` falls 512 -> 256 -> 128 -- give 1.073x, with one conv at
      0.828x.
    - THE TAIL IS THE ACTIVATION PACK, not the weights. Order 142's phase shares
      settle it: the array thread spends 99.4% of its time at the barrier and
      0.59% multiplying, and staging weights is 0.60%. Every activation panel is
      packed in phase 0 while the array idles, and per output element that pack
      costs `K / N` -- constant at 9 across this 3x3 family -- while the array's
      work per output element falls with `N`. That is why the tail tracks `N`
      falling and why no weight-side change moves it: order 142 tried holding
      the whole filter resident and it was a pure regression (0.8456x overall,
      first-17 down to 0.622x), because the resident panels consumed the
      activation slots that were buying the interior overlap.
    - The 0.828x conv is by construction, not a defect: at `N = 128` there are 4
      column tiles, so the array works in 4 of every 6 steps
      (`n_tiles / (n_tiles + 2)`) before any pack cost is counted.

    THE FOLLOW-UP LEVER IS A BLOCK-LEVEL ACTIVATION DOUBLE BUFFER -- pack block
    `b + 1` during block `b`'s MMAs -- and order 142's 99.4% barrier share is its
    sizing evidence. It is a schedule change rather than a loader policy, which
    is why it is not in this arm.

    Two refusals, and both leave the answer correct rather than absent:
    a geometry the loader cannot serve (`C_in % 32`, a degenerate extent) and a
    shape the array declines (`matmul_bt_hmx_accepts`) both fall through to the
    HVX kernel below, which is one call rather than a second body -- the conv
    already has two bodies and does not need a third. A shape the array
    ACCEPTED and then could not run is not retried on HVX: reporting the vector
    unit's numbers as the matrix unit's is the one failure this path must not
    have, and witness slot 7 says which happened.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(GEMV_SIMD_W))
        witness.store(1, UInt32(SIMD[DType.float32, GEMV_SIMD_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    comptime if MM_PATH_HMX:
        # Every thread reaches the same verdict from the same scalars with no
        # barrier, which is what lets a declined shape drop to the HVX nest
        # with its full four-thread split intact (order 084's rule).
        var geom = HmxConvGeom(R, S, pad_h, pad_w, W_out, H, W, C_in)
        var m = H_out * W_out
        var k = R * S * C_in
        if geom.usable() and matmul_bt_hmx_accepts(m, C_out, k, 0, C_out):
            _matmul_stripe_hmx[act_conv=True](
                C, img, B, witness, m, C_out, k, thread_idx, geom
            )
            return

    mojo_conv_im2col_f16(
        C,
        img,
        B,
        witness,
        H,
        W,
        C_in,
        C_out,
        R,
        S,
        pad_h,
        pad_w,
        H_out,
        W_out,
    )


# ===--------------------------------------------------------------------------===
# Transpose + per-output-row bias — the VAE decode's layout round trip
# (`transpose_bias_kernel.mojo`; F4 in
# `.agentwork/flux-npu/VAE_CODE_CRITIQUE.md`). Follows the groupnorm/rope
# seam above: the algorithm and the launch-striped entry live in their own
# module, and this is the `abi("C")` symbol `NPUKernels.def` selector 41
# binds.
# ===--------------------------------------------------------------------------===


@export
def mojo_transpose_bias_f16(
    output: UnsafePointer[Scalar[TB_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[TB_DTYPE], MutUnsafeAnyOrigin],
    bias: UnsafePointer[Scalar[TB_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    R: Int,
    N: Int,
) abi("C"):
    """Writes `output[N, R]` as the transpose of `input[R, N]`, adding
    `bias[n]` to every element of output row `n`.

    Two calls in the VAE decode's fused-conv path, one direction each:
    `output[c_out][pixel] = bias[c_out] + conv_result[pixel][c_out]` replaces
    the host's `scatterConvRows` (the fused conv kernel's raw `[M, C_out]`
    result reshaped to `[C_out, spatial]`, real per-channel bias); and
    `output[pixel][c_in] = 0 + silu_result[c_in][pixel]` replaces the host's
    `nhwc_from_chw` (CHW activation packed to the `[H, W, C_in]` map the
    fused im2col gather reads, zero bias since there is none for a pure
    transpose). Both legs pass through the SAME kernel and the SAME
    unconditional bias add; the caller supplies a zero-filled bias row for
    the second.
    """
    _transpose_bias_entry[TB_DTYPE](
        TbRowMajor2D[TB_DTYPE](ptr=output, layout=row_major(Coord(N, R))),
        TbRowMajor2D[TB_DTYPE](ptr=input, layout=row_major(Coord(R, N))),
        TbRowMajor2D[TB_DTYPE](ptr=bias, layout=row_major(Coord(1, N))),
        witness,
    )


# ===--------------------------------------------------------------------------===
# `mo.softmax` — HVX row-wise softmax
#
# IMPORTED, not folded (cold critique C2; megakernel slice 4). The row-wise
# softmax is authored in `softmax_kernel.mojo`; this file used to carry its own
# copy of the whole three-pass body inside the export "under the same obligation:
# a change to one is a change to both". It had drifted the way the others had --
# the module addresses through views and states the element alignment on its
# scalar tails, this copy did neither -- so `test_softmax.mojo` proved a body no
# device ever ran.
#
# What stays here is the export, under the name the manifest knows, as a seam.
#
# What ALSO stays, and is deliberately not moved, is everything below the next
# banner: `_reduce_max`, `_exp_hvx` and its constants. They live in this section
# because softmax is where they were first written, but `_exp_hvx` is ATTENTION's
# vector exp (M4.0b) and softmax has never called it -- softmax and the module
# both use `std.math.exp`, which scalarizes on this triple and is silicon-verified
# as it stands. Moving them with the softmax body would hand the attention family
# a dependency on a module it does not otherwise use, and switching softmax onto
# `_exp_hvx` is a numerics change that belongs with its own device confirm, not
# with an extraction. `SOFTMAX_ACCUM` stays for the same reason: it is the dtype
# `_exp_hvx`'s constants are written in.
# ===--------------------------------------------------------------------------===

comptime SOFTMAX_ACCUM = DType.float32
"""The reduction and exponential dtype; an fp16 sum saturates past 2048.

Kept here rather than moved with the softmax body: `_exp_hvx` below, which is
the attention kernel's, states its constants in it.
"""


@always_inline
def _reduce_max[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Horizontal max across all lanes; every lane holds the result.

    The rotate-and-combine fold `_reduce` uses for the sum, spelled for `max`:
    the hand-rolled form lowers to `V6_*` HVX ops where `SIMD.reduce_max()`
    scalarizes.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced = max(reduced, reduced.rotate_right[1 << i]())
    return reduced


# ===--------------------------------------------------------------------------===
# Vector exponential — the shared softmax primitive `std.math.exp` cannot be.
#
# On `hexagon-unknown-unknown-elf`, `std.math.exp` SCALARIZES at every width, and
# only the disassembly says so: `mojo_softmax_f16` above emits 130 out-of-line
# `floorf` calls and 256 `vinsert` for two passes that read as vector code. In
# the attention kernel below that put three library calls in the innermost loop
# and forced its accumulator to spill around them; `_exp_hvx` is what M4.0b
# replaced them with.
#
# The construction is `sincos_tables._exp_f32`'s at fp32 vector width:
# round-to-nearest by the magic add ("two exact-width IEEE adds and no
# `round` for a bare-metal target to miss"), a two-part ln2 so the reduced
# argument keeps its low bits, a Horner polynomial, and the power of two built by
# placing the exponent field. Every operation is a lane-wise fp32 multiply/add, a
# lane-wise int32 add/shift, or a reinterpretation — no cross-lane traffic, so
# the interleaved-lane class KB `kernels/hvx-fold-lane-order-audit` gates cannot
# arise here.
#
# Only attention calls it today. Softmax and SiLU carry the same scalarization
# and are silicon-verified as they stand, so switching them is a numerics change
# that belongs with its own device confirm, not with this one.
# ===--------------------------------------------------------------------------===

comptime EXP_LOG2_E = Scalar[SOFTMAX_ACCUM](1.44269504088896340736)
"""Base change log2(e), which turns the reduction into an exponent field."""

comptime EXP_LN2_HI = Scalar[SOFTMAX_ACCUM](0.693359375)
"""Leading part of ln2. Exactly representable in fp32 (0x3F318000), so
`n * EXP_LN2_HI` is exact and the subtraction cancels without error."""

comptime EXP_LN2_LO = Scalar[SOFTMAX_ACCUM](-2.12194440e-4)
"""Tail of ln2, i.e. `ln2 - EXP_LN2_HI`. Carrying it separately is what keeps
the reduced argument accurate at the top of the range, where `n` is ~128."""

comptime EXP_RNE_MAGIC = Scalar[SOFTMAX_ACCUM](12582912.0)
"""1.5 * 2^23. Adding it forces round-to-nearest-even into the fp32 mantissa
and subtracting it back leaves the rounded integer, for |x| < 2^22."""

comptime EXP_RNE_MAGIC_BITS = Int32(0x4B400000)
"""Bit pattern of `EXP_RNE_MAGIC`. With the magic add the float's low bits ARE
the integer, so `n` is one int subtract away and no fp-to-int convert (which
this target lowers out of line) is needed."""

comptime EXP_MIN_ARG = Scalar[SOFTMAX_ACCUM](-87.3)
"""Clamp below. `exp(-87.3)` is 1.4e-38, the smallest normal fp32: one ulp lower
and the exponent field would underflow to a denormal or wrap. The online softmax
reaches it through `-inf` — the initial running max, and the padding the
attention kernel writes into a short key tile."""


@always_inline
def _exp_hvx[
    width: Int
](x: SIMD[SOFTMAX_ACCUM, width]) -> SIMD[SOFTMAX_ACCUM, width]:
    """`exp(x)` for `x <= 0`, in lane-wise fp32 and int32 ops only.

    Domain: the online softmax only ever exponentiates `s - m_new` and
    `m - m_new`, both non-positive by construction because `m_new` is the
    running max. Inputs above ~88 would overflow the exponent field, and this
    deliberately does not clamp them — a positive argument here would mean the
    recurrence had broken, and hiding that behind a saturate would turn a wrong
    answer into a plausible one.

    Accuracy: the degree-6 Horner is the Taylor series of `e^r` on
    `|r| <= ln2/2`, whose truncation is ~1.2e-7 relative — three decimal orders
    below fp16's 4.9e-4 resolution.
    """
    comptime A = SOFTMAX_ACCUM
    var xc = max(x, SIMD[A, width](EXP_MIN_ARG))
    var t = xc * SIMD[A, width](EXP_LOG2_E) + SIMD[A, width](EXP_RNE_MAGIC)
    var n = t - SIMD[A, width](EXP_RNE_MAGIC)
    var n_int = bitcast[DType.int32, width](t) - SIMD[DType.int32, width](
        EXP_RNE_MAGIC_BITS
    )
    var r = (xc - n * SIMD[A, width](EXP_LN2_HI)) - n * SIMD[A, width](
        EXP_LN2_LO
    )

    var p = SIMD[A, width](1.0 / 720.0)
    p = p * r + SIMD[A, width](1.0 / 120.0)
    p = p * r + SIMD[A, width](1.0 / 24.0)
    p = p * r + SIMD[A, width](1.0 / 6.0)
    p = p * r + SIMD[A, width](0.5)
    p = p * r + SIMD[A, width](1.0)
    p = p * r + SIMD[A, width](1.0)

    var pow2 = bitcast[A, width]((n_int + SIMD[DType.int32, width](127)) << 23)
    return p * pow2


@export
def mojo_softmax_f16(
    output: UnsafePointer[Scalar[SOFTMAX_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[SOFTMAX_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """Row-wise softmax `y = exp(x - max) / sum(exp(x - max))` on the launch's threads.

    This is the seam. The skel's C ABI hands down pointers and extents, the
    extents become a layout once, here, and everything under it is the imported
    module -- the shipped code and the tested code at the same time.
    """
    var extent = row_major(Coord(num_rows, row_length))
    _softmax_entry[SOFTMAX_DTYPE](
        SmRowMajor2D[SOFTMAX_DTYPE](ptr=input, layout=extent),
        SmRowMajor2D[SOFTMAX_DTYPE](ptr=output, layout=extent),
        witness,
    )


# ===--------------------------------------------------------------------------===
# RoPE — HVX rotary position embedding
#
# The rotary embedding authored and host-validated in
# `hexagon_ondevice.rope_kernel`, imported rather than folded. It reproduces the
# Skye production RoPE's convention
# (`max/kernels/src/qaic/rope_split_store.mojo`) with no HMX and no vendor
# archive.
#
# Each of `num_rows` Q/K head vectors of `head_size` elements is roped
# independently against one precomputed cos/sin row (decode is `num_rows == 1`
# per head); the rows divide across the launch's compute threads by launch
# index, exactly as the GEMV and softmax divide theirs. The four seams below
# differ only in the dtype and the pairing convention the module is
# parameterized on.
# ===--------------------------------------------------------------------------===


@export
def mojo_rope_half_f16(
    dst: UnsafePointer[Scalar[ROPE_IO_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[ROPE_IO_DTYPE], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[ROPE_FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (HF `rotate_half`) over `[num_rows, head_size]` head vectors.

    `x`/`dst` are row-major fp16; `dst` must not alias `x`. `freqs` is
    `[max_position, head_size]` fp32 cos/sin interleaved per pair; `positions[row]`
    picks the freqs row (all heads of one decode token share a position). The
    vectorized `rotate_half` path is the one that computes on the vector units;
    `witness` reports the width it compiled at, as the GEMV's and softmax's do.
    """
    rope_entry[ROPE_IO_DTYPE, interleaved=False](
        dst, x, freqs, positions, witness, num_rows, head_size
    )


@export
def mojo_rope_half_bf16(
    dst: UnsafePointer[Scalar[ROPE_IO_DTYPE_BF], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[ROPE_IO_DTYPE_BF], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[ROPE_FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (HF `rotate_half`) over bf16 `[num_rows, head_size]` head vectors.

    `freqs` stays fp32 as the fp16 entry's does, the rotation being fp32 either
    way: the two entries differ only in how the head vectors are loaded and
    stored, which is why bf16 costs nothing at the angles.
    """
    rope_entry[ROPE_IO_DTYPE_BF, interleaved=False](
        dst, x, freqs, positions, witness, num_rows, head_size
    )


@export
def mojo_rope_interleaved_f16(
    dst: UnsafePointer[Scalar[ROPE_IO_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[ROPE_IO_DTYPE], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[ROPE_FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (interleaved neighbour pairs), the reference's second convention.

    Pairs `(2p, 2p+1)` and runs scalar; `witness[0]` still reports the compiled
    vector-width constant, but the vectorization evidence is the `rotate_half`
    entry above and the object's HVX-mnemonic count.
    """
    rope_entry[ROPE_IO_DTYPE, interleaved=True](
        dst, x, freqs, positions, witness, num_rows, head_size
    )


@export
def mojo_rope_interleaved_bf16(
    dst: UnsafePointer[Scalar[ROPE_IO_DTYPE_BF], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[ROPE_IO_DTYPE_BF], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[ROPE_FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (interleaved neighbour pairs) over bf16 head vectors.

    The bf16 half of the convention pair, so that a dtype reaches either
    convention rather than only the one the fp16 set happened to be signed with.
    """
    rope_entry[ROPE_IO_DTYPE_BF, interleaved=True](
        dst, x, freqs, positions, witness, num_rows, head_size
    )


# ===--------------------------------------------------------------------------===
# SiLU and SwiGLU — HVX feed-forward activations
#
# IMPORTED, not folded (cold critique C2; megakernel slice 4). The FLUX.2-Klein
# FFN activations are authored in `silu_swiglu_kernel.mojo`; this file used to
# carry its own copy of `_silu`, the witness and both striped entries "under the
# same obligation: a change to one is a change to both". The two had already
# drifted, the same way the layer norm's had: the module took typed tiles through
# every signature below its seam and stated the element alignment on its scalar
# tails, and the copy shipped here passed pointers and extents and did not. So
# `test_silu_swiglu.mojo` proved a body no device ever ran.
#
# What stays here are the four exports, under the names the manifest knows, as
# three-line seams. The module's `silu_2d` / `swiglu_2d` and the launch-striped
# `_silu_entry` / `_swiglu_entry` now run one shared row body, so the kernel the
# host twin scores and the kernel a launch runs are the same instructions.
# ===--------------------------------------------------------------------------===


@export
def mojo_silu_f16(
    output: UnsafePointer[Scalar[SILU_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[SILU_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """Plain SiLU over fp16 rows, spread across the launch's threads.

    This is the seam. The skel's C ABI hands down pointers and extents, the
    extents become a layout once, here, and everything under it is the imported
    module.
    """
    var extent = row_major(Coord(num_rows, row_length))
    _silu_entry[SILU_DTYPE](
        SwRowMajor2D[SILU_DTYPE](ptr=input, layout=extent),
        SwRowMajor2D[SILU_DTYPE](ptr=output, layout=extent),
        witness,
    )


@export
def mojo_silu_bf16(
    output: UnsafePointer[Scalar[SILU_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[SILU_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """Plain SiLU over bf16 rows, spread across the launch's threads.

    The seam, as `mojo_silu_f16` documents it.
    """
    var extent = row_major(Coord(num_rows, row_length))
    _silu_entry[SILU_DTYPE_BF](
        SwRowMajor2D[SILU_DTYPE_BF](ptr=input, layout=extent),
        SwRowMajor2D[SILU_DTYPE_BF](ptr=output, layout=extent),
        witness,
    )


@export
def mojo_swiglu_f16(
    output: UnsafePointer[Scalar[SILU_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[SILU_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    inner_dim: Int,
    row_pitch: Int,
) abi("C"):
    """Fused SwiGLU over fp16 rows, spread across the launch's threads.

    The seam, as `mojo_silu_f16` documents it. The two tiles differ in their row
    extent on purpose: the input's is the pitch, which is how a packed gate/up
    pair can be a column slice of a wider fused projection, and the output's is
    the dense `inner_dim`.
    """
    _swiglu_entry[SILU_DTYPE](
        SwRowMajor2D[SILU_DTYPE](
            ptr=input, layout=row_major(Coord(num_rows, row_pitch))
        ),
        SwRowMajor2D[SILU_DTYPE](
            ptr=output, layout=row_major(Coord(num_rows, inner_dim))
        ),
        witness,
    )


@export
def mojo_swiglu_bf16(
    output: UnsafePointer[Scalar[SILU_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[SILU_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    inner_dim: Int,
    row_pitch: Int,
) abi("C"):
    """Fused SwiGLU over bf16 rows, spread across the launch's threads.

    The seam, as `mojo_swiglu_f16` documents it.
    """
    _swiglu_entry[SILU_DTYPE_BF](
        SwRowMajor2D[SILU_DTYPE_BF](
            ptr=input, layout=row_major(Coord(num_rows, row_pitch))
        ),
        SwRowMajor2D[SILU_DTYPE_BF](
            ptr=output, layout=row_major(Coord(num_rows, inner_dim))
        ),
        witness,
    )


# ===--------------------------------------------------------------------------===
# `mo.mha.no_cache` — HVX-only flash attention (Track 1)
#
# IMPORTED, not folded (megakernel slice 4's last family). The online-softmax
# FlashAttention and its masked twin are `attention_kernel.mojo`'s, host-validated
# there by `test_attention_kernel.mojo` and `test_attention_masked_kernel.mojo`.
# This file used to carry a second copy of both "under the same obligation: a
# change to one is a change to both", and they had drifted the way every other
# family's pair had: the copy here reached its operands by adding
# `row * num_heads * head_dim` to a pointer, and it alone carried the order-111
# L2 hint. Dense non-causal (`NULL_MASK`), fp16 I/O, fp32 accumulators.
#
# What stays here are the two exports the manifest binds, as seams. The masked
# one is a call to the module's entry and nothing else. The dense one is not, and
# that is the whole reason it is still written out: it also holds the HMX arms
# (the VTCM grant, the array lock and the `rt` selection) and the per-launch
# cycle bracket, neither of which a host-linkable module can carry. The vector
# arm it selects is the module's body, so the two arms cannot drift on numerics.
#
# The heads divide across the core's compute threads by launch index, one head
# slice per thread with no cross-thread state, so no barrier — the split the GEMV
# and softmax make over their output rows, and the one `attention_thread_heads`
# now states once for both entries. `witness` reports the compiled vector width,
# the fp32 accumulator lane count, the participant count, and the RTOS thread
# each launch index ran on, laid out as the kernels above lay it out.
# ===--------------------------------------------------------------------------===

comptime MHA_PROBE_STRIDE = 6
"""Witness words each participant reports its cycle bracket through.

Laid out AFTER the existing 8 words (widths, participant count, and the five
per-thread RTOS ids), so every current reader of this witness is unaffected and
a driver that does not know about the bracket simply ignores the tail."""


# ===--------------------------------------------------------------------------===
# Attention on the HMX array (P5 slices 2 and 3) — the half that needs the device.
#
# `hmx_attention.mojo` owns everything that is layout: the arena, the packers'
# axis rules, the two matmuls, and the vdeal/vshuff seam. It is host-linkable and
# its whole flash loop is scored against dense attention by `test_hmx_crouton.mojo`.
# What is here is what a host cannot have — the VTCM grant, the HMX lock, and the
# softmax arithmetic the HVX arm above already runs on silicon, which is shared
# rather than rewritten so the two arms cannot drift on numerics.
#
# The arm is selected by the kernel's `rt` scalar, which the graph passes as 0
# (`npu_attention_dispatch.cpp:247`) and which `mojo_mha_no_cache_f16`'s docstring
# has reserved for the Track-2 crouton row-tile since the kernel was written. So
# `SKYE_HMX_ATTN=1` is a build that can still render: the array arm needs the flag
# AND a caller that asks for it, and the flag-off object is byte-identical to its
# base because every call below is behind a `comptime if`.
#
# The grant is the matmul stripe's, entry point for entry point — `qshimHmxLock`,
# `qshimHmxEnter`, the work, `qshimHmxExit`, `qshimHmxUnlock`. Releasing through
# the wrong one leaves the unit held and poisons the device for every later run,
# and the lock's refusal words (witness 8-14) belong to the matmul: this kernel's
# witness is FULL at 32 words (`dispatch.mojo`'s `NPU_WITNESS_WORDS`, all of them
# written by the existing probe bracket), so nothing here reports through it.
# ===--------------------------------------------------------------------------===

comptime MHA_RT_HVX = 0
"""`rt = 0`: the shipped vector path. What the graph sends, always."""

comptime MHA_RT_DARK = 1
"""`rt = 1`: slice 2. The HVX arm computes every head as usual, and the array
computes ONE QK^T tile beside it whose result no attention math reads."""

comptime MHA_RT_FLASH = 2
"""`rt = 2`: slice 3. Head 0 is computed by the array's flash loop and every
other head by the HVX arm, single-threaded, no feeders, no rendezvous."""

comptime MHA_RT_TEAM = 3
"""`rt = 3`: slice 4. EVERY head on the array, across the whole launch.

The last participant holds the grant and does nothing but issue MMA chains; the
rest pack, run the softmax and read `O` back out, one bounded rendezvous per
step. This is the arm that turns slice 3's per-head win into step time — slice 3
left 23 of 24 heads on the vector path, which is why order 118 measured the
dispatch wall flat at 1.001-1.004x while the head itself ran 5.72-6.02x."""

comptime MHA_HMX_SPACE = AddressSpace.SHARED
"""How VTCM residency is spelled here; there is no `AddressSpace.VTCM`."""

comptime MHA_HMX_PATH_RAN: Scalar[MHA_IO_DTYPE] = 1
"""The array computed what it was asked to."""

comptime MHA_HMX_PATH_NO_VTCM: Scalar[MHA_IO_DTYPE] = 2
"""The VTCM grant is absent, or too small for the arena the plan needs."""

comptime MHA_HMX_PATH_NO_GRANT: Scalar[MHA_IO_DTYPE] = 3
"""`qshimHmxLock` refused. The matmul stripe's witness words 8-14 carry the
detail when IT asks; this kernel has no witness room and does not duplicate
them."""

comptime MHA_HMX_PATH_REFUSED: Scalar[MHA_IO_DTYPE] = 4
"""The shape does not divide into the plan's blocks, so nothing was attempted."""

comptime MHA_DARK_MAGIC: Scalar[MHA_IO_DTYPE] = 1234
"""Marks the dark report as written. An integer fp16 holds exactly, so a host
that reads anything else is reading attention output, not a report."""

comptime MHA_DARK_WORDS = 16
"""Elements of `O`'s first query row, head 0, the dark leg overwrites.

The witness cannot carry this — it is full — and no new buffer may be added
without moving the frozen selector ABI, so the report rides the one output path
that already exists. It costs 16 of the 4.7 M elements `O` holds, in DARK mode
only, and `MHA_RT_FLASH`'s output is untouched so the slice-3 A/B stays clean."""


@always_inline
def _mha_hmx_open() -> Scalar[MHA_IO_DTYPE]:
    """Resolves VTCM and takes the HMX unit. Returns a `MHA_HMX_PATH_*` code.

    ONE thread calls this. `qshimGetBaseVtcmAddr` acquires the grant lazily and
    without a once-guard, which is what made the matmul stripe's refusal
    thread-divergent (P5 M4.2 order 084); here the array arm is single-threaded
    by construction, so the race has no second party rather than being resolved.
    """
    if Int(_qshim_get_base_vtcm_addr()) == 0:
        return MHA_HMX_PATH_NO_VTCM
    if Int(external_call["qshimHmxLock", Int32]()) != 0:
        return MHA_HMX_PATH_NO_GRANT
    external_call["qshimHmxEnter", NoneType]()
    return MHA_HMX_PATH_RAN


@always_inline
def _mha_hmx_close():
    """Gives the unit back, through the entry points that granted it."""
    external_call["qshimHmxExit", NoneType]()
    _ = external_call["qshimHmxUnlock", Int32]()


@always_inline
def _mha_hmx_arena(
    plan: HmxAttnPlan,
) -> HmxAttnArena[MHA_HMX_SPACE]:
    """Carves the head's arena at the base of the standing VTCM grant.

    Only the BASE is aligned: `hmx_attn_arena_in` lays every region out on a
    whole number of tiles from it, which is the `:deep` load's requirement.
    """
    var base = align_up(Int(_qshim_get_base_vtcm_addr()), HMX_TILE_BYTES)
    return hmx_attn_arena_in[MHA_HMX_SPACE](
        UnsafePointer[UInt8, MutUntrackedOrigin, address_space=MHA_HMX_SPACE](
            unsafe_from_address=base
        ),
        plan,
    )


@always_inline
def _mha_dot[
    W: Int
](
    a: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    b: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    length: Int,
) -> Scalar[MHA_ACCUM]:
    """fp32 dot of two contiguous length-`length` fp16 vectors.

    The vector arm's own dot moved to `attention_kernel.mojo` with the rest of
    the body and takes row VIEWS. This one stays because the dark probe below is
    the only caller left and it is walking an unpacked `[seq, heads, dim]`
    operand tile by hand to score the array's scores against the vector unit's;
    a view of the row it wants is exactly what it does not have.

    `W` lanes at a time through the shared `_fmac_sf_hf` HVX MAC, folded to a
    scalar by the shared `_reduce_add`, with a scalar tail for a `length` that
    is not a multiple of `W`.
    """
    var acc = SIMD[MHA_ACCUM, W](0)
    var kv = 0
    while kv + W <= length:
        acc = _fmac_sf_hf(acc, a.load[W](kv), b.load[W](kv))
        kv += W
    var total = _reduce_add(acc)[0]
    while kv < length:
        total += (
            a.load[1](kv).cast[MHA_ACCUM]()[0]
            * b.load[1](kv).cast[MHA_ACCUM]()[0]
        )
        kv += 1
    return total


@always_inline
def _mha_hmx_fits(plan: HmxAttnPlan) -> Bool:
    """Whether the standing VTCM grant has room for this plan's arena."""
    var vtcm = Int(_qshim_get_base_vtcm_addr())
    var base = align_up(vtcm, HMX_TILE_BYTES)
    return hmx_attn_arena_bytes(plan) <= Int(_qshim_get_vtcm_size()) - (
        base - vtcm
    )


comptime HMX_EXP_IDENT = get_defined_bool["SKYE_HMX_EXP_IDENT", False]()
"""INSTRUMENT, NEVER A CANDIDATE: replaces attention's exp with a two-vop
clamped copy so `SOFTMAX(control) − SOFTMAX(exp-ident)` prices the exp
polynomial exactly (SOFTMAX_PROBE leg B; the polynomial is ~20-24 vops per
32 lanes against ~2 here, same loads and stores).

The output is numerically WRONG by construction — the arm stages throwaway
only, and its renders MUST diverge from the anchor: a matching sha means the
flag did not take, which is the leg's delivery check. Correctness properties
that keep the run representative rather than degenerate: for `x <= 0` the
result is in `[0, 1]`, equals 1 at `x = 0` (a fully-rescaled first block
behaves), is monotone in `x`, and keeps every softmax sum strictly positive
so `l` never divides by zero. OFF is byte-identical by comptime fold."""


struct MhaExpHvx(AttnSoftmax):
    """Binds the kernel's own `_exp_hvx` into the shared schedule.

    The exponential cannot be imported the other way — `hmx_attention.mojo` is
    below this file in the dependency order, and moving `_exp_hvx` down would put
    the shipped vector arm's emission at risk to no purpose. Injecting it costs
    this struct and keeps ONE spelling of the Horner in the tree.
    """

    @staticmethod
    @always_inline
    def exp_nonpos[
        width: Int
    ](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
        """`exp(x)` for `x <= 0`, from the arm silicon has already ruled on."""
        comptime if HMX_EXP_IDENT:
            return 1.0 + x.clamp(-1.0, 0.0)
        else:
            return _exp_hvx[width](x)


@always_inline
def _mha_hmx_softmax_rows[
    space: AddressSpace, //, W: Int
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    r: Int,
    slab_pair: Int,
    slot: Int,
):
    """Slice 3's softmax: the shared recurrence over staging half 0.

    Slice 3 is single-threaded and never leaves half 0, so it is slice 4's step
    with the halves pinned. Delegating rather than keeping a second copy is what
    makes the twin's coverage of `hmx_attn_softmax_rows` coverage of BOTH arms —
    the two were the same arithmetic when slice 3 shipped, and this is what stops
    them drifting apart now that a second caller exists.
    """
    hmx_attn_softmax_rows[MhaExpHvx, W](arena, plan, 0, r, slab_pair, slot)


@always_inline
def _mha_flash_head_hmx[
    space: AddressSpace, //, D: Int, W: Int
](
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    seq_len: Int,
    num_heads: Int,
    head_idx: Int,
    scale: Scalar[MHA_ACCUM],
):
    """One head's flash attention with both matmuls on the array.

    Structurally `_mha_flash_head` with three substitutions: QK^T is an
    accumulator chain instead of a `_fmac_sf_hf` nest, the P@V axpy is a second
    chain whose HEAD pair applies the online-softmax correction to `O` (so the
    vector units never read or write it), and the final normalise is a third. The
    softmax between them is unchanged.

    `Br` and `Bc` come from the plan rather than from `MHA_BM`/`MHA_BN`: the
    array wants blocks two orders larger than the vector nest does, because its
    cost is per crouton pair and the vector nest's is per key row.

    The caller holds the HMX grant for the WHOLE call. A release clears the
    accumulators, so acquiring per tile would lose the chain silently, as wrong
    numbers rather than as an error.
    """
    var rows = row_major(Coord(seq_len, num_heads * D))
    var Qt = HmxRows2D(ptr=Q, layout=rows)
    var Kt = HmxRows2D(ptr=K, layout=rows)
    var Vt = HmxRows2D(ptr=V, layout=rows)
    var Ot = HmxRows2D(ptr=O, layout=rows)

    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    var br = plan.br_tiles * HMX_TILE
    var bc = plan.bc_tiles * HMX_TILE
    var head_off = head_idx * D

    hmx_attn_program_records(arena, scale.cast[MHA_IO_DTYPE]())
    hmx_attn_reset_diag(arena, plan)

    var q0 = 0
    while q0 < seq_len:
        hmx_attn_pack_q(arena, plan, Qt, q0, head_off)
        # Both halves, because the first key block's rescale reads the one it is
        # not writing and an uninitialised operand would be a NaN times zero.
        hmx_attn_zero_panel(arena.o, 2 * plan.br_tiles * plan.d_tiles)
        for i in range(br):
            arena.stats.store[1](
                Coord(ATTN_STAT_M, i),
                SIMD[MHA_ACCUM, 1](min_or_neg_inf[MHA_ACCUM]()),
            )
            arena.stats.store[1](Coord(ATTN_STAT_L, i), SIMD[MHA_ACCUM, 1](0))

        var cur = 0
        var k0 = 0
        while k0 < seq_len:
            hmx_attn_pack_k(arena, plan, Kt, k0, head_off)
            hmx_attn_pack_v(arena, plan, Vt, k0, head_off)
            hmx_attn_qk_block(arena, plan)

            for r in range(plan.br_tiles):
                for p in range(SLAB_PAIRS):
                    _mha_hmx_softmax_rows[W](arena, plan, r, p, 0)

            hmx_attn_pv_block(arena, plan, cur, 1 - cur)
            cur = 1 - cur
            k0 += bc

        for r in range(plan.br_tiles):
            for local in range(HMX_TILE):
                var l = arena.stats.load[1](
                    Coord(ATTN_STAT_L, r * HMX_TILE + local)
                )[0]
                hmx_attn_set_diag(
                    arena,
                    r,
                    local,
                    (Scalar[MHA_ACCUM](1) / l).cast[MHA_IO_DTYPE](),
                )
        hmx_attn_normalise(arena, plan, cur, 1 - cur)
        cur = 1 - cur
        hmx_attn_store_o(arena, plan, Ot, q0, head_off, cur)
        q0 += br


@always_inline
def _mha_hmx_dark[
    space: AddressSpace, //, D: Int, W: Int
](
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    seq_len: Int,
    num_heads: Int,
    scale: Scalar[MHA_ACCUM],
):
    """Computes ONE QK^T tile on the array beside the HVX arm, and compares them.

    Dark in the sense that matters: no attention value anywhere reads the array's
    `S`. What the array produces is scored against the same 32x32 block of scores
    `_mha_dot` computes, and only the SCORE leaves — a count, two magnitudes and
    one spot pair — beside the accumulator probe's readouts.

    The comparison runs on device rather than on the host because the host has no
    channel to receive a 1024-element tile through: the witness is full and the
    ABI is frozen. Reducing it here to numbers the host can read out of `O` is
    the alternative to inventing a buffer.
    """
    var rows = row_major(Coord(seq_len, num_heads * D))
    var Qt = HmxRows2D(ptr=Q, layout=rows)
    var Kt = HmxRows2D(ptr=K, layout=rows)

    # ORDER 118 (lane instrument on top of 374d8cc7c63, NOT part of #341 as it
    # stands): `hmx_attn_acc_probe` reads its result back through
    # `arena.rec_id`, and `hmx_attn_program_records` is the ONLY thing that
    # fills that record. Called in the PR's order the two probes requant
    # through uninitialised VTCM, and on silicon they came back -298.75 and
    # -281.75 -- the right SUMS through a garbage scale of about -0.1373,
    # matching none of the three predictions the report carries beside them.
    # Programming the records first is the whole fix.
    hmx_attn_program_records(arena, scale.cast[MHA_IO_DTYPE]())

    var probe0 = hmx_attn_acc_probe(arena, plan, 0)
    var probe1 = hmx_attn_acc_probe(arena, plan, 1)
    hmx_pack_panel[False](arena.q_run(plan, 0), Qt, 0, 0, plan.d_tiles)
    hmx_pack_panel[True](arena.k_run(plan, 0), Kt, 0, 0, plan.d_tiles)
    hmx_mma_chain(
        arena.s_tile(plan, 0, 0),
        arena.rec_qk,
        HmxOperands[space](
            arena.q_run(plan, 0), arena.k_run(plan, 0), plan.d_tiles
        ),
        HmxOperands[space](arena.q_run(plan, 0), arena.k_run(plan, 0), 0),
    )

    # fp16 carries 11 mantissa bits, so two ULP of the block's largest score is
    # the resolution the readout itself has. A count above zero at that
    # threshold is a finding; the two magnitudes beside it are what lets the
    # host recompute the threshold it would rather have used.
    var worst = Scalar[MHA_ACCUM](0)
    var peak = Scalar[MHA_ACCUM](0)
    var spot_hmx = Scalar[MHA_IO_DTYPE](0)
    var spot_hvx = Scalar[MHA_IO_DTYPE](0)
    for m in range(HMX_TILE):
        for n in range(HMX_TILE):
            var want = scale * _mha_dot[W](
                Q + m * num_heads * D, K + n * num_heads * D, D
            )
            var got = arena.s.load[1](
                Coord(0, m // HMX_SPLIT, n, m % HMX_SPLIT)
            )[0]
            var diff = abs(got.cast[MHA_ACCUM]() - want)
            if diff > worst:
                worst = diff
            if abs(want) > peak:
                peak = abs(want)
            if m == 0 and n == 0:
                spot_hmx = got
                spot_hvx = want.cast[MHA_IO_DTYPE]()

    # A second sweep rather than a staged copy of the reference: the threshold
    # is block-relative and therefore not known until the first sweep ends, and
    # 1024 more dot products on a leg that runs once per launch is cheaper than
    # borrowing a buffer whose liveness a later slice would have to re-argue.
    var tol = peak * Scalar[MHA_ACCUM](0.0009765625)
    var wrong = 0
    for m in range(HMX_TILE):
        for n in range(HMX_TILE):
            var want = scale * _mha_dot[W](
                Q + m * num_heads * D, K + n * num_heads * D, D
            )
            var got = arena.s.load[1](
                Coord(0, m // HMX_SPLIT, n, m % HMX_SPLIT)
            )[0]
            if abs(got.cast[MHA_ACCUM]() - want) > tol:
                wrong += 1

    O.store(1, Scalar[MHA_IO_DTYPE](wrong))
    O.store(2, worst.cast[MHA_IO_DTYPE]())
    O.store(3, peak.cast[MHA_IO_DTYPE]())
    O.store(4, probe0.readout)
    O.store(5, probe0.wide)
    O.store(6, probe0.per_product)
    O.store(7, probe0.per_pass)
    O.store(8, probe1.readout)
    O.store(9, probe1.wide)
    O.store(10, probe1.per_product)
    O.store(11, probe1.per_pass)
    O.store(12, spot_hmx)
    O.store(13, spot_hvx)
    O.store(14, Scalar[MHA_IO_DTYPE](HMX_TILE * HMX_TILE))
    O.store(15, MHA_DARK_MAGIC)


@always_inline
def _mha_hmx_team[
    D: Int, W: Int, masked: Bool = False
](
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    mask: AttnStats[AddressSpace.GENERIC],
    seq_len: Int,
    num_heads: Int,
    scale: Scalar[MHA_ACCUM],
    thread_idx: Int,
) -> Bool:
    """Every head on the array, across the whole launch. True when it ran.

    ONE DRIVER FOR BOTH ENTRIES. `masked` picks which step the loop below
    issues, and nothing else here differs: the roles, the two thread-uniform
    refusals, the grant, the rendezvous discipline and the unwind are the same
    code for the dense render kernel and the masked encoder one. That is the
    point — this function's failure mode is a wedged domain rather than a wrong
    number, so a second copy of it is the last thing that should exist.

    It is a comptime BRANCH inside the body rather than a shared callee for the
    reason `_attn_softmax_rows_impl` records: lifting a piece of a shipped
    `@always_inline` body into its own function reschedules the dense arm's
    object even when the arithmetic is untouched, and the render ships this arm
    with `SKYE_HMX_ATTN=1`.

    `mask` is read only when `masked`. The dense entry passes a live view over a
    local scalar rather than a null one, so no arm carries a pointer that would
    fault if the comptime branch were ever wrong.

    EVERY participant calls this, including the one the vector arm gives no
    heads to. The last of them holds the HMX grant for the whole dispatch and
    does nothing but multiply; the rest pack, run the softmax and unpack. A
    False return means the caller owes the heads to the vector arm.

    The deadlock discipline is the matmul stripe's, and it has to be because the
    failure mode is a wedged domain rather than a wrong number:

    - The two refusals ahead of the first rendezvous are pure functions of values
      every participant reads identically — the release-stored participant count
      and arithmetic on the kernel's own scalars. All threads take them or none
      does, so nobody is left waiting on an arrival that is not coming.
    - The grant is the ONE asymmetric fact, and this kernel's witness is FULL
      (32 words, all written by the probe bracket), so it cannot be broadcast
      the way the stripe broadcasts it in slot 7. It rides the launch status
      instead: a failed lock marks the launch, and `qshimLaunchSyncWait` returns
      nonzero to EVERY participant from that moment on. That makes the first
      rendezvous the broadcast, which is what slot 7 was doing anyway.
    - Every rendezvous return is checked, and the grant is released on every
      exit including the unwind. A held unit poisons the device for every later
      run.

    Returns:
        True when the array computed every head, False when the caller must.
    """
    var participants = Int(_qshim_query(QSHIM_QUERY_NUM_THREADS))
    var roles = hmx_stripe_roles(participants)
    if not roles.usable():
        return False

    var plan = hmx_attn_plan(ATTN_BR, ATTN_BC, D, roles.feeders)
    var sched = hmx_attn_sched(plan, seq_len, num_heads)
    var bc = plan.bc_tiles * HMX_TILE
    if not sched.usable() or (bc % W) != 0:
        return False

    # After the thread-uniform refusals, so a dispatch this arm declines opens no
    # span and reports all-zero buckets -- which reads as "never instrumented"
    # rather than as a plausible zero beside real numbers.
    hmx_phase_open(thread_idx)
    hmx_phase_mark(thread_idx, HMX_PHASE_OPEN)

    var is_array = thread_idx == roles.array_thread
    var granted = False
    if is_array:
        var path = _mha_hmx_open()
        if path == MHA_HMX_PATH_RAN and not _mha_hmx_fits(plan):
            _mha_hmx_close()
            path = MHA_HMX_PATH_NO_VTCM
        granted = path == MHA_HMX_PATH_RAN
        if not granted:
            # Ahead of any arrival, so this is the sticky fact every peer's
            # first rendezvous reads rather than a race with one.
            external_call["mojo_npu_launch_fail", NoneType]()

    if not _matmul_hmx_sync(thread_idx):
        # The launch is already marked; the caller falls back so `O` is written.
        if granted:
            _mha_hmx_close()
        return False

    var rows = row_major(Coord(seq_len, num_heads * D))
    var arena = _mha_hmx_arena(plan)
    hmx_attn_program_records(arena, scale.cast[MHA_IO_DTYPE]())
    hmx_attn_reset_diag(arena, plan)

    hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
    for step in range(sched.steps()):
        comptime if masked:
            hmx_attn_step_masked[MhaExpHvx, W](
                HmxRows2D(ptr=Q, layout=rows),
                HmxRows2D(ptr=K, layout=rows),
                HmxRows2D(ptr=V, layout=rows),
                HmxRows2D(ptr=O, layout=rows),
                mask,
                arena,
                plan,
                sched,
                step,
                thread_idx,
                roles.feeders,
            )
        else:
            hmx_attn_step[MhaExpHvx, W](
                HmxRows2D(ptr=Q, layout=rows),
                HmxRows2D(ptr=K, layout=rows),
                HmxRows2D(ptr=V, layout=rows),
                HmxRows2D(ptr=O, layout=rows),
                arena,
                plan,
                sched,
                step,
                thread_idx,
                roles.feeders,
            )
        if not _matmul_hmx_sync(thread_idx):
            external_call["mojo_npu_launch_fail", NoneType]()
            if granted:
                _mha_hmx_close()
            # NOT a fallback: part of `O` is already the array's, so recomputing
            # the rest on the vector arm would splice two answers together. The
            # launch is failed and the host is told.
            return True

    if granted:
        _mha_hmx_close()
    return True


@always_inline
def _mha_hmx_arm[
    D: Int, W: Int
](
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    seq_len: Int,
    num_heads: Int,
    scale: Scalar[MHA_ACCUM],
    mode: Int,
) -> Scalar[MHA_IO_DTYPE]:
    """Runs the array arm on ONE thread. Returns its `MHA_HMX_PATH_*` code.

    A `MHA_RT_FLASH` caller must run head 0 on the vector path itself for every
    code but `RAN`: a slice that leaves part of `O` unwritten and returns success
    is the failure mode the matmul stripe's cold critique named, and here it
    would read as a silent numerics regression.

    The caller must also tell the two kinds of refusal apart. `REFUSED` is a
    shape this arm does not serve and is not a fault — the render dispatches
    attention at more than one sequence length, and failing a launch for the ones
    the blocks do not divide would turn a diagnostic flag into an outage. Every
    other code means the arm was owed the work and could not take it, which is a
    fact the host has to receive.
    """
    var plan = hmx_attn_plan(ATTN_BR, ATTN_BC, D, 1)
    var bc = plan.bc_tiles * HMX_TILE
    var ok = (
        hmx_attn_accepts(plan, seq_len)
        and hmx_attn_probe_fits(plan)
        and (bc % W) == 0
    )
    if not ok:
        if mode == MHA_RT_DARK:
            O.store(0, MHA_HMX_PATH_REFUSED)
            O.store(15, MHA_DARK_MAGIC)
        return MHA_HMX_PATH_REFUSED

    var path = _mha_hmx_open()
    if path == MHA_HMX_PATH_RAN and not _mha_hmx_fits(plan):
        _mha_hmx_close()
        path = MHA_HMX_PATH_NO_VTCM
    if path != MHA_HMX_PATH_RAN:
        if mode == MHA_RT_DARK:
            O.store(0, path)
            O.store(15, MHA_DARK_MAGIC)
        return path

    var arena = _mha_hmx_arena(plan)
    if mode == MHA_RT_DARK:
        _mha_hmx_dark[D, W](Q, K, O, arena, plan, seq_len, num_heads, scale)
        O.store(0, MHA_HMX_PATH_RAN)
    else:
        _mha_flash_head_hmx[D, W](
            Q, K, V, O, arena, plan, seq_len, num_heads, 0, scale
        )
    _mha_hmx_close()
    return MHA_HMX_PATH_RAN


@export
def mojo_mha_no_cache_f16(
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    seq_len: Int,
    num_heads: Int,
    scale: Float32,
    rt: Int,
) abi("C"):
    """Dense fp16 flash attention over `[seq_len, num_heads, head_dim=128]`.

    The heads divide across the core's compute threads by launch index (Track-1
    parallelism: one head slice per thread, no cross-thread state, so no
    barrier); finer within-head q-tile parallelism is a device-lane perf
    refinement. `witness` reports what the output cannot — the compiled vector
    width, the fp32 accumulator lane count, the participant count, and the RTOS
    thread each launch index ran on — laid out as the GEMV / softmax / RoPE
    kernels lay it out, so the on-device dispatch check proves both vectorization
    and the by-index thread split the same way.

    `rt` selects the arm, and 0 -- what the graph sends, always -- is the vector
    path this docstring describes. `MHA_RT_DARK` adds one QK^T tile on the matrix
    array whose result nothing reads, and `MHA_RT_FLASH` moves head 0's whole
    flash loop onto it; both exist only in a `SKYE_HMX_ATTN` build and neither is
    reachable from the graph. It was reserved for the Track-2 crouton row-tile
    since this kernel was written, which is what it now carries."""
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(MHA_SIMD_W))
        witness.store(1, UInt32(SIMD[MHA_ACCUM, MHA_SIMD_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    # Ahead of the compute-thread gate because the array thread IS the
    # participant that gate turns away: the vector arm gives the fifth thread no
    # heads, which is exactly what leaves it free to hold the grant.
    comptime if HMX_ATTN:
        if rt == MHA_RT_TEAM:
            # The dense arm has no mask, and `masked` is False, so this view is
            # never read. It is still a live scalar rather than a null pointer:
            # a dead argument that would fault if the comptime branch were ever
            # wrong is a worse dead argument.
            var no_mask = Scalar[MHA_ACCUM](0)
            if _mha_hmx_team[MHA_HEAD_DIM, MHA_SIMD_W](
                Q,
                K,
                V,
                O,
                AttnStats[AddressSpace.GENERIC](
                    ptr=UnsafePointer(to=no_mask).unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                    layout=row_major(Coord(1, 1)),
                ),
                seq_len,
                num_heads,
                scale,
                thread_idx,
            ):
                return

    # The empty run is what the fifth thread and any launch index past the head
    # count get, and returning on it rather than looping zero times is what keeps
    # the cycle bracket below a bracket around work that happened.
    var heads = attention_thread_heads(num_heads, thread_idx)
    var head_begin = heads[0]
    var head_end = heads[1]
    if head_begin >= head_end:
        return

    # Bracketed so the loop's cost can be read against its static packet count.
    # The share of the step attention owns was measured before this existed;
    # what it does not say is whether the remaining gap is stall or issue, and
    # only a per-thread cycle count against the disassembly can. Both counters
    # are the ones the DDR read-rate probe used, and the QTimer is read
    # alongside UPCYCLE so the rate is against wall time rather than a core
    # clock DVFS may have moved.
    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()

    # The array arm, on the thread that owns head 0 and nowhere else. Every
    # other thread's heads run the vector path unchanged, and so do this one's
    # once head 0 is the array's -- which is what makes slice 3 a one-head
    # differential rather than a rewrite of the kernel.
    var hmx_took_head = False
    comptime if HMX_ATTN:
        if thread_idx == 0 and rt == MHA_RT_FLASH:
            var path = _mha_hmx_arm[MHA_HEAD_DIM, MHA_SIMD_W](
                Q, K, V, O, seq_len, num_heads, scale, rt
            )
            hmx_took_head = path == MHA_HMX_PATH_RAN
            if path != MHA_HMX_PATH_RAN and path != MHA_HMX_PATH_REFUSED:
                # The vector path below still writes head 0, so `O` stays
                # correct; the launch is marked failed because an arm that was
                # owed the work and could not take it would otherwise read on
                # the A/B as a numerics result rather than as a missing arm.
                # A shape this arm does not serve is NOT that, and does not
                # fail a launch.
                external_call["mojo_npu_launch_fail", NoneType]()

    var vector_begin = head_begin
    if hmx_took_head:
        vector_begin = head_begin + 1

    # The extents become one layout here and nothing below this line takes them
    # again; the head range stays an argument because it is a work split across
    # the core's threads, not a property of the tensor, and this arm's begin is
    # not the partition's when the array took head 0.
    var extent = row_major(Coord(seq_len, num_heads, MHA_HEAD_DIM))
    flash_attention_heads[
        use_hvx=True, BM=MHA_BM, BN=MHA_BN, D=MHA_HEAD_DIM, W=MHA_SIMD_W
    ](
        MhaBSHD[MHA_IO_DTYPE](ptr=Q, layout=extent),
        MhaBSHD[MHA_IO_DTYPE](ptr=K, layout=extent),
        MhaBSHD[MHA_IO_DTYPE](ptr=V, layout=extent),
        MhaBSHD[MHA_IO_DTYPE](ptr=O, layout=extent),
        vector_begin,
        head_end,
        scale,
    )

    # DARK runs AFTER the vector arm, not before it: its report overwrites 16
    # elements of head 0's first output row, and the vector arm writes that row.
    comptime if HMX_ATTN:
        if thread_idx == 0 and rt == MHA_RT_DARK:
            _ = _mha_hmx_arm[MHA_HEAD_DIM, MHA_SIMD_W](
                Q, K, V, O, seq_len, num_heads, scale, rt
            )

    var cycles = _read_upcycle() - cycles0
    var ticks = _read_utimer() - ticks0
    # Past the per-thread RTOS ids at 3..7, so the existing witness readers are
    # untouched. Pairs are what the cost is per: heads x seq_len x seq_len.
    var report = witness + 8 + thread_idx * MHA_PROBE_STRIDE
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    # The unit the cost is per, computed here rather than reassembled on the
    # host: every query row sweeps every key, so pairs = heads x queries x keys.
    report.store(4, UInt32((head_end - head_begin) * seq_len * seq_len))
    report.store(5, UInt32(seq_len))

    _ = rt


# ===--------------------------------------------------------------------------===
# ===--------------------------------------------------------------------------===
# `mo.mha` causal + padding masked — the Qwen3 text-encoder flash (P4)
#
# The dense kernel above with one delta: an additive score mask. The Qwen3
# encoder attention is causal + padding, delivered as the host-precomputed
# additive bias MAX's `masked_flash_attention_gpu` consumes (`[seq_len, seq_len]`
# fp32, 0 where visible and a large negative where masked, the causal triangle
# and the padding folded together, broadcast across heads). Adding `bias[query,
# key]` to each score before the online-softmax max makes a masked key contribute
# `exp(-inf) = 0`, so the same body serves causal and padding without a per-tile
# skip. GQA (Qwen3 32:8) is resolved on the host by materializing K/V to 32
# heads, so this stays dense MHA.
#
# The body, the head split and the witness are `attention_kernel.mojo`'s, so this
# export is the seam and nothing else -- no HMX arm reaches the masked kernel.
# ===--------------------------------------------------------------------------===


@export
def mojo_mha_masked_f16(
    O: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    Q: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[MHA_IO_DTYPE], MutUnsafeAnyOrigin],
    bias: UnsafePointer[Scalar[MHA_ACCUM], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    seq_len: Int,
    num_heads: Int,
    scale: Float32,
    rt: Int,
) abi("C"):
    """Causal + padding masked fp16 flash attention over `[seq_len, num_heads,
    head_dim=128]`.

    The Qwen3 text-encoder (P4) attention: `mojo_mha_no_cache_f16` with an
    additive `bias` (fp32 `[seq_len, seq_len]`, broadcast across heads) added to
    each score before the softmax max, encoding the causal triangle and the
    padding mask the host folds together. The head split, the witness layout, and
    the fp32 online-softmax numerics are the dense kernel's; only the extra
    `bias` buffer and the one score add differ.

    `rt` selects the arm exactly as the dense entry's does, and 0 -- what an
    unset `MODULAR_NPU_MHA_RT` sends -- is the vector path this docstring
    describes. `MHA_RT_TEAM` puts every head on the matrix array with the bias
    added at the softmax seam; it exists only in a `SKYE_HMX_ATTN` build, and a
    shape or a role the array declines returns False before any rendezvous, so
    the vector arm still writes `O`. The dense entry's `MHA_RT_DARK` and
    `MHA_RT_FLASH` are slice-2/3 diagnostics and have no masked counterpart:
    they exist to probe the array, and the array is probed by the entry that
    already has them."""
    # EVERYTHING THE ARRAY ARM NEEDS LIVES UNDER THE FLAG, including the
    # witness write and the thread index it needs. `mha_masked_entry` already
    # writes those four words, so hoisting them above this block would leave the
    # shipped `SKYE_HMX_ATTN=0` object carrying a duplicate write it can never
    # use -- measured at +36 bytes, which is 36 more than an inert change may
    # cost. Inside the flag, the write is what makes the array arm's early
    # return safe: it never reaches `mha_masked_entry`, and a driver reading the
    # witness of a dispatch that ran would otherwise see the host fill and call
    # the launch dead. On the fallback path the entry writes the same words
    # again, which is idempotent -- each is a property of the build or the
    # thread, not of the work.
    comptime if HMX_ATTN:
        if rt == MHA_RT_TEAM:
            var thread_idx = _npu_thread_idx()
            if thread_idx == 0:
                witness.store(0, UInt32(MHA_SIMD_W))
                witness.store(1, UInt32(SIMD[MHA_ACCUM, MHA_SIMD_W]().size))
                witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
            witness.store(3 + thread_idx, _qurt_thread_id())

            var rows = row_major(Coord(seq_len, seq_len))
            if _mha_hmx_team[MHA_HEAD_DIM, MHA_SIMD_W, masked=True](
                Q,
                K,
                V,
                O,
                AttnStats[AddressSpace.GENERIC](
                    ptr=bias.unsafe_origin_cast[MutUntrackedOrigin](),
                    layout=rows,
                ),
                seq_len,
                num_heads,
                scale,
                thread_idx,
            ):
                return

    mha_masked_entry(O, Q, K, V, bias, witness, seq_len, num_heads, scale)


# ===--------------------------------------------------------------------------===
# adaLN modulation and gated residual — HVX MMDiT elementwise affines
#
# The two elementwise ops every FLUX.2-Klein MMDiT block runs around its
# attention and feed-forward sub-blocks (`flux2_modulev3/transformer_blocks.py`),
# authored and host-validated in `modulation_kernel.mojo` and duplicated here the
# way the kernels above are, under the same obligation: a change to one is a
# change to both until the kernel can be shared.
#
# - adaLN modulate is `out = (1 + scale) * x_normed + shift`, the affine applied
#   to the normalized activation. `scale` and `shift` are per-channel vectors.
# - gated residual is `out = x + gate * y`, closing each sub-block; `gate` is a
#   per-channel vector and `y` the sub-block output.
#
# The parameters broadcast across the token rows: a `param_row_stride` of 0
# shares one length-`D` vector across every row (the FLUX case -- modulation from
# a `[B, D]` time embedding), while `== row_length` selects a materialized
# `[rows, D]`. A stride-0 row axis IS "every row reads the same vector", which is
# what the module's `_param_tile` states as a layout instead of as a scalar the
# loop has to apply by hand.
#
# IMPORTED, not folded (cold critique C2; megakernel slice 4). This file used to
# carry its own copy of the witness and both striped entries "under the same
# obligation: a change to one is a change to both". They had drifted the way the
# other three families had: the module addresses through views and states the
# element alignment at every access, and the copy here passed pointers, extents
# and a stride and took the off-GPU 1-byte default on its scalar tails. So
# `test_modulation.mojo` proved a body no device ever ran.
#
# What stays here are the five exports, under the names the manifest knows, as
# seams. `mojo_gate_residual_clamp_f16` remains its own export because the
# manifest has a row for it, but it is `clamp_fp16=True` on one shared body --
# the comptime parameter the plan expected this extraction to collapse the two
# rows onto.
# ===--------------------------------------------------------------------------===


# Fused no-affine LayerNorm inside the adaLN entry (build.sh ADA_FUSED_LN=1;
# define emitted only off-default so a flag-off command line is byte-identical
# to the base tree's). When on, `x` is the PRE-norm activation and the host
# skips its layernorm dispatch. eps is comptime: every block-stack site passes
# cfg.eps = 1e-6f, and the arming gate is bit-identity against the two-kernel
# sequence, which enforces the match on silicon. fp16-only by design — the
# dispatched production selector is 17 (f16); the bf16 arm stays unfused.
comptime ADA_FUSED_LN = get_defined_int["SKYE_ADA_FUSED_LN", 0]()


@export
def mojo_ada_modulate_f16(
    output: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    scale: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    shift: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
) abi("C"):
    """Applies adaLN modulation over fp16 rows, spread across the launch's threads.
    """
    comptime if ADA_FUSED_LN != 0:
        _ada_modulate_ln_entry[MOD_DTYPE](
            _mod_rows_tile[MOD_DTYPE](x, num_rows, row_length),
            _mod_param_tile[MOD_DTYPE](
                scale, num_rows, row_length, param_row_stride
            ),
            _mod_param_tile[MOD_DTYPE](
                shift, num_rows, row_length, param_row_stride
            ),
            _mod_rows_tile[MOD_DTYPE](output, num_rows, row_length),
            witness,
            Float32(1e-6),
        )
    else:
        _ada_modulate_entry[MOD_DTYPE](
            _mod_rows_tile[MOD_DTYPE](x, num_rows, row_length),
            _mod_param_tile[MOD_DTYPE](
                scale, num_rows, row_length, param_row_stride
            ),
            _mod_param_tile[MOD_DTYPE](
                shift, num_rows, row_length, param_row_stride
            ),
            _mod_rows_tile[MOD_DTYPE](output, num_rows, row_length),
            witness,
        )


@export
def mojo_ada_modulate_bf16(
    output: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    scale: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    shift: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
) abi("C"):
    """Applies adaLN modulation over bf16 rows, spread across the launch's threads.
    """
    _ada_modulate_entry[MOD_DTYPE_BF](
        _mod_rows_tile[MOD_DTYPE_BF](x, num_rows, row_length),
        _mod_param_tile[MOD_DTYPE_BF](
            scale, num_rows, row_length, param_row_stride
        ),
        _mod_param_tile[MOD_DTYPE_BF](
            shift, num_rows, row_length, param_row_stride
        ),
        _mod_rows_tile[MOD_DTYPE_BF](output, num_rows, row_length),
        witness,
    )


@export
def mojo_gate_residual_f16(
    output: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """Gated residual over fp16 rows, spread across the launch's threads."""
    _gate_residual_entry[MOD_DTYPE](
        _mod_rows_tile[MOD_DTYPE](x, num_rows, row_length),
        _mod_rows_tile[MOD_DTYPE](y, num_rows, row_length),
        _mod_param_tile[MOD_DTYPE](gate, num_rows, row_length, gate_row_stride),
        _mod_rows_tile[MOD_DTYPE](output, num_rows, row_length),
        witness,
    )


@export
def mojo_gate_residual_bf16(
    output: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[MOD_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """Gated residual over bf16 rows, spread across the launch's threads."""
    _gate_residual_entry[MOD_DTYPE_BF](
        _mod_rows_tile[MOD_DTYPE_BF](x, num_rows, row_length),
        _mod_rows_tile[MOD_DTYPE_BF](y, num_rows, row_length),
        _mod_param_tile[MOD_DTYPE_BF](
            gate, num_rows, row_length, gate_row_stride
        ),
        _mod_rows_tile[MOD_DTYPE_BF](output, num_rows, row_length),
        witness,
    )


@export
def mojo_gate_residual_clamp_f16(
    output: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[MOD_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """Gated residual over fp16 rows, clamped to fp16's finite range.

    There is no bf16 counterpart: bf16 carries fp32's exponent range, so the
    bound cannot be reached, and the reference guards on fp16 alone.
    """
    _gate_residual_entry[MOD_DTYPE, clamp_fp16=True](
        _mod_rows_tile[MOD_DTYPE](x, num_rows, row_length),
        _mod_rows_tile[MOD_DTYPE](y, num_rows, row_length),
        _mod_param_tile[MOD_DTYPE](gate, num_rows, row_length, gate_row_stride),
        _mod_rows_tile[MOD_DTYPE](output, num_rows, row_length),
        witness,
    )


# ===--------------------------------------------------------------------------===
# `mo.reduce.group_norm` (affine) — HVX GroupNorm for the FLUX.2 VAE decoder
#
# IMPORTED, not folded (megakernel slice 4; the layer norm's cold critique C2).
# This file used to carry its own `GroupNorm3Pass` / `_group_norm` /
# `_group_norm_entry` beside the tested original in `groupnorm_kernel.mojo`,
# "under the same obligation: a change to one is a change to both". The two had
# already drifted: the authoring module describes a group twice through the tile
# model -- flat for the two reductions, `(channels_per_group, spatial)` for the
# affine -- and the copy here still reached a channel by adding `c * spatial` to
# a pointer. So the host twin `test_groupnorm_kernel.mojo` scored a body no
# device ever ran.
#
# The math is `group_norm_gpu_block` from `max/kernels/src/nn/normalization.mojo`,
# the registered `mo.reduce.group_norm`, restricted to the affine case the VAE
# uses (per-channel gamma / beta). The exports below stay here, under the names
# the kernel manifest knows, and are the same seam the module's own
# `hvx_groupnorm_*` entries are: the skel's C ABI hands down pointers and
# extents, and the extents become layouts once, below this line.
# ===--------------------------------------------------------------------------===


@export
def mojo_groupnorm_f16(
    output: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
) abi("C"):
    """Affine group norm over fp16 `[N, C, spatial]`, spread across the threads.
    """
    _group_norm_entry[GN_DTYPE](
        output, input, gamma, beta, witness, N, C, spatial, num_groups, eps
    )


@export
def mojo_groupnorm_bf16(
    output: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
) abi("C"):
    """Affine group norm over bf16 `[N, C, spatial]`, spread across the threads.
    """
    _group_norm_entry[GN_DTYPE_BF](
        output, input, gamma, beta, witness, N, C, spatial, num_groups, eps
    )


# ===--------------------------------------------------------------------------===
# INT8 W8A8 HMX matmul — on-device crouton byte-order confirmation
#
# The inner matrix multiply-accumulate runs on the HMX unit, which Mojo's LLVM
# cannot emit, so it is issued by the public-intrinsic C shim (`hmx_int8_shim.c`,
# linked by `build.sh` with `-mhmx`). This entry is the single-thread device
# orchestration: it hands the host-packed croutons and the qshim VTCM grant to
# the shim, which stages them into VTCM, holds the HMX unit, MACs each 32x32
# output tile over the K-tiles, and dumps the accumulator. The host
# (`npu_hmx_matmul_dispatch.cpp`) reads that accumulator back and identifies the
# in-tile byte order the HMX instructions actually consume.
#
# Whether this device grants HMX at all is still an open question, so the entry
# leads with a non-blocking `qurt_hmx_try_lock` probe that takes no VTCM, runs
# no HMX instruction, and releases the unit immediately. Everything after it is
# gated on that probe succeeding AND on `SKYE_HMX_EXECUTE` being compiled in:
# the blocking `qurt_hmx_lock2` suspends the caller indefinitely when another
# thread holds the unit (`qurt_hmx.h:47-50`), and a wedged invocation is a
# failure mode that returning-before-the-HMX-instruction does not cover.
#
# Runs on one thread (launch index 0) and does no HVX itself — the shim it calls
# is compiled without `-mhvx`. The witness carries every status code, so a run
# that stopped at the probe is distinguishable from one that reached the MAC
# rather than passing on a stale buffer.
# ===--------------------------------------------------------------------------===

comptime HMX_PROBE_MARKER: UInt32 = 0x484D5831
"""'HMX1' — written to the witness once the HMX unit is held, so the host can
tell a real HMX run from a readback the kernel never reached."""


comptime HMX_G_RT_BASE = -10
"""`rt` values at or below this select order 131's probe G instead of a tile MAC.

Cell `c` is `HMX_G_RT_BASE - rt`: bit 0 is the operand MODE (0 plain, 1 `:cm`)
and bit 1 the CONTROL WORDS (0 the fp16 inclusive byte extent, 1 the integer
packed fields). So -10 is the plain stream the shim has always issued, -13 is
the stripe's own `:cm` stream with the stripe's own words, and the two
off-diagonal cells are what separate the mode from the words."""

comptime HMX_G_ACT_EXTENT: Int32 = 1023
"""One 1024-byte crouton as an INCLUSIVE byte extent, the fp16 loads' encoding.
"""

comptime HMX_G_ACT_PACKED: Int32 = 0xE3
"""`HMX_INT8_ACT_CONTROL`, restated here rather than imported.

The probe exists to ask whether the arm's constant is right; taking it from the
arm would make one of the two cells unable to disagree with the other."""

comptime HMX_G_WGT_PACKED: Int32 = 0
"""`HMX_INT8_WGT_CONTROL` (`dW`, zero for a dense 32x32 tile)."""

comptime HMX_G_STORE_PACKED: Int32 = 0xE0
"""`HMX_INT8_STORE_MASK`, the store's operand on the packed-field side."""


@export
def mojo_hmx_int8_probe(
    acc_out: UnsafePointer[Scalar[DType.uint8], MutUnsafeAnyOrigin],
    a_cr: UnsafePointer[Scalar[DType.uint8], MutUnsafeAnyOrigin],
    b_cr: UnsafePointer[Scalar[DType.int8], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    m_pad: Int,
    k_pad: Int,
    n_pad: Int,
    rt: Int,
) abi("C"):
    """Runs one INT8 W8A8 HMX matmul and reads the 32x32 accumulator back.

    `a_cr` / `b_cr` are the host-packed activation (uint8) and weight (int8)
    croutons; `acc_out` receives the uint8 accumulator tiles (tile-major, via the
    shim's saturating-byte accumulator store). The MAC and the readback are the
    shim's; this owns the thread guard, the VTCM grant, and the witness. Only
    launch index 0 participates — the HMX unit is held by one thread.
    """
    var thread_idx = _npu_thread_idx()
    if thread_idx != 0:
        return

    # The non-blocking probe runs FIRST, before any query that could touch the
    # resource manager, so its answer describes a fresh PD rather than one this
    # kernel has already put into some state. It takes no VTCM, executes no HMX
    # instruction, and releases the unit at once if granted.
    var try_rc = external_call["qshimHmxTryLock", Int32]()

    witness.store(2, UInt32(thread_idx))
    witness.store(3, _qurt_thread_id())
    witness.store(4, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    # The device's preemption-capability mask, so the host can see which HMX
    # lock framework this chip implements.
    witness.store(6, external_call["qshimHmxCaps", UInt32]())
    # Low 16 bits keep the (small) QuRT status without a negative-to-unsigned
    # conversion trap; the host sign-extends to recover the exact signed code.
    witness.store(8, UInt32(Int(try_rc) & 0xFFFF))
    witness.store(
        9, UInt32(Int(external_call["qshimHvxUnlockRc", Int32]()) & 0xFFFF)
    )
    witness.store(
        10, UInt32(Int(external_call["qshimHvxMode", Int32]()) & 0xFFFF)
    )

    # A refused QuRT probe ends the run here: no blocking lock (which would
    # suspend indefinitely if another thread held the unit) and no HMX
    # instruction. Before returning, ask the resource manager the same question
    # in the framework it advertises -- the two paths reach the unit
    # independently, so a refusal on the raw syscall does not settle it.
    if Int(try_rc) != 0:
        _ = external_call["qshimHmxCrmProbe", Int32]()
        # Full width, unlike the QuRT codes above: a HAP_compute_res status is
        # 32 bits (HAP_COMPUTE_RES_NOT_SUPPORTED is 0x80000404), so truncating
        # to 16 would turn it into an unrelated small number.
        witness.store(
            12,
            UInt32(
                Int(external_call["qshimHmxCrmAcquireRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            13,
            UInt32(
                Int(external_call["qshimHmxCrmCachedRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            14,
            UInt32(
                Int(external_call["qshimHmxCrmLockRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            15,
            UInt32(
                Int(external_call["qshimHmxCrmLock3Rc", Int32]()) & 0xFFFFFFFF
            ),
        )
        _ = external_call["qshimHmxShippingProbe", Int32]()
        witness.store(
            16,
            UInt32(
                Int(external_call["qshimHmxShippingAcquireRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            17,
            UInt32(
                Int(external_call["qshimHmxShippingLockRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        # The one shape none of the above tries: power the HMX block on and
        # vote its clock through the power manager before asking for it. Runs
        # last so a failure inside it cannot disturb the probes above.
        _ = external_call["qshimHmxLlamacppProbe", Int32]()
        witness.store(
            18,
            UInt32(
                Int(external_call["qshimHmxPowerResident", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            19,
            UInt32(
                Int(external_call["qshimHmxPowerApptypeRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            20,
            UInt32(
                Int(external_call["qshimHmxPowerDcvsRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            21,
            UInt32(
                Int(external_call["qshimHmxPowerHvxRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            22,
            UInt32(
                Int(external_call["qshimHmxPowerHmxRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            23,
            UInt32(
                Int(external_call["qshimHmxPowerHmxV2Rc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            24,
            UInt32(
                Int(external_call["qshimHmxFullAcquireRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            25,
            UInt32(
                Int(external_call["qshimHmxFullHvxUnlockRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            26,
            UInt32(
                Int(external_call["qshimHmxFullLockRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        # Whether the power manager's acceptance of the HMX clock vote means
        # the rail exists, or is what it returns for any rail at all.
        _ = external_call["qshimHmxClockProbe", Int32]()
        witness.store(
            27,
            UInt32(
                Int(external_call["qshimHmxClockGetRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(28, external_call["qshimHmxClockCoreHz", UInt32]())
        witness.store(29, external_call["qshimHmxClockHmxHz", UInt32]())
        witness.store(30, external_call["qshimHmxClockStreamerHz", UInt32]())
        witness.store(
            31,
            UInt32(
                Int(external_call["qshimHmxClockStreamerRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        witness.store(
            32,
            UInt32(
                Int(external_call["qshimHmxPowerStreamerRc", Int32]())
                & 0xFFFFFFFF
            ),
        )
        # The shared VTCM grant's own shape: HMX armed on the acquire AND
        # enabled by the cached acquire, which no probe above combines.
        var pr42_rc = external_call["qshimHmxPr42Probe", Int32]()
        witness.store(33, external_call["qshimHmxPr42Ctx", UInt32]())
        witness.store(
            34,
            UInt32(
                Int(external_call["qshimHmxPr42CachedRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            35,
            UInt32(
                Int(external_call["qshimHmxPr42Lock4Rc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            36,
            UInt32(
                Int(external_call["qshimHmxPr42Lock3Rc", Int32]()) & 0xFFFFFFFF
            ),
        )
        witness.store(
            37,
            UInt32(
                Int(external_call["qshimHmxPr42LockRc", Int32]()) & 0xFFFFFFFF
            ),
        )
        # A grant here is the one case where an HMX instruction may run: the
        # unit was obtained on this very context, so the MAC below executes on
        # a granted unit rather than a hoped-for one.
        if Int(pr42_rc) != 0:
            return
        if Int(external_call["qshimHmxExecuteEnabled", Int32]()) == 0:
            witness.store(38, 1)
            external_call["qshimHmxPr42Release", NoneType]()
            return
    if Int(external_call["qshimHmxExecuteEnabled", Int32]()) == 0:
        return

    var vtcm = _qshim_get_base_vtcm_addr()
    var unlock_rc: Int32 = -2
    var lock_rc: Int32
    if rt == -1 or rt == -2:
        # The fp16 MMA sequence issued as the reference's own instruction
        # words (inline asm, not builtins), so a zero readback is attributable
        # to the hardware rather than to intrinsic lowering. Writes 2048
        # bytes of fp16 output into acc_out; operands are device-generated.
        # rt == -2 additionally runs it on a freshly created QuRT thread that
        # has never executed HVX -- the reference's thread model -- so a dead
        # result on this dispatch worker is distinguishable from a dead array.
        var fresh_thread: Int32 = 1 if rt == -2 else 0
        lock_rc = external_call["skye_hmx_fp16_probe_tile", Int32](
            acc_out, vtcm, fresh_thread, UnsafePointer(to=unlock_rc)
        )
    elif rt <= HMX_G_RT_BASE:
        # ORDER 131, probe G. `rt` selects one cell of a 2x2: the operand MODE
        # (plain vs `:cm`) crossed with the CONTROL WORDS (the fp16 inclusive
        # byte extent vs the integer packed fields). The cells are enumerated
        # here rather than bit-packed into `rt` so the configuration a leg ran
        # is named in the kernel and cannot be mis-decoded from a number.
        #
        # The requant record rides the ACTIVATION buffer, 256 bytes past its
        # crouton region: the probe selector's argument list is frozen by
        # `mojo_npu_imp.c` and a fifth buffer would be an ABI change on a path
        # three other legs depend on.
        var cell = -(rt - HMX_G_RT_BASE)
        var mode: Int32 = 1 if (cell & 1) != 0 else 0
        var packed = (cell & 2) != 0
        var act_ctl: Int32 = HMX_G_ACT_PACKED if packed else HMX_G_ACT_EXTENT
        var wgt_ctl: Int32 = HMX_G_WGT_PACKED if packed else HMX_G_ACT_EXTENT
        var store_ctl: Int32 = HMX_G_STORE_PACKED if packed else 0
        lock_rc = external_call["skye_hmx_int8_probe_g", Int32](
            acc_out,
            a_cr,
            b_cr,
            a_cr + (m_pad * k_pad),
            vtcm,
            m_pad,
            k_pad,
            n_pad,
            mode,
            act_ctl,
            wgt_ctl,
            store_ctl,
            UnsafePointer(to=unlock_rc),
        )
    else:
        lock_rc = external_call["skye_hmx_int8_probe_tile", Int32](
            acc_out,
            a_cr,
            b_cr,
            vtcm,
            m_pad,
            k_pad,
            n_pad,
            rt,
            UnsafePointer(to=unlock_rc),
        )

    witness.store(0, UInt32(Int(lock_rc) & 0xFFFF))
    witness.store(1, UInt32(Int(unlock_rc) & 0xFFFF))
    # Read after the acquire has run (inside the probe), so this reflects the
    # real acquire_cached status -- an acquire problem vs a lock problem.
    witness.store(
        7, UInt32(Int(external_call["qshimHmxCachedRc", Int32]()) & 0xFFFF)
    )
    # The blocking `qurt_hmx_lock2` status on its own. Slot 0 collapses both
    # acquisition paths into one code, so it cannot distinguish "the QuRT
    # syscall took the unit" from "the QuRT syscall refused and the resource
    # manager took it"; this slot can.
    witness.store(
        11, UInt32(Int(external_call["qshimHmxQurtRc", Int32]()) & 0xFFFF)
    )
    if Int(lock_rc) == 0:
        witness.store(5, HMX_PROBE_MARKER)


# ===--------------------------------------------------------------------------===
# Annihilation rung-1 bench shim (selector 43; re-sign window item 7)
#
# One dispatch per stage-arm of the rung-1 microbench
# (`hmx_annihilation_bench.mojo`): baseline FF -> swiglu -> FFO, then native
# FF' -> swiglu' -> FFO', all six over one host-allocated scratch buffer, so
# the `HMX_PHASE_PROBE` table attributes per dispatch — the design's
# pre-registered reading is exactly two of those attributions (native-FF
# UNPACK and native-FFO PACK_ACT drop to the linear copy) plus the bit
# identity the native-FFO dispatch writes into its witness.
#
# The carve, the fills, the arm configuration and the verdict live in the
# bench module, shared with `test_annihilation_bench.mojo`; this entry adds
# only what the host twin cannot run — the launch roles, the HMX grant, the
# bounded rendezvous and the VTCM arena — and it reuses `mojo_matmul_f16`'s
# own grant machinery (`_matmul_hmx_open` / `_matmul_hmx_sync` /
# `_matmul_hmx_unwind`) so the order-084 discipline is inherited, not
# transcribed. fp16 on every build: the bench instantiates the fp16 arm
# explicitly, so an int8 skel runs the same experiment.
# ===--------------------------------------------------------------------------===


@always_inline
def _anni_matmul_case[
    native_a: Bool, native_c: Bool
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    base: Int,
    plan: HmxStripePlan,
    thread_idx: Int,
    feeders: Int,
) -> Bool:
    """`_matmul_hmx_sweep`'s schedule over the bench module's own step seam.

    A sibling rather than a flag on the production sweep so the shipped
    matmul's machine code is untouched by this probe's existence. Two
    deliberate deltas from the production sweep, both part of the protocol:
    the step is `annihilation_matmul_step` (the ONE arm statement the host
    gate also instantiates), and the unpack-arm echo is written
    unconditionally — it is this probe's treatment-delivery fingerprint, and
    a fingerprint behind a build flag is the silent-null class the bench
    exists to refuse.

    Returns:
        Whether every rendezvous completed inside its deadline.
    """
    var arena = hmx_stripe_arena_in[MATMUL_HMX_SPACE](
        UnsafePointer[
            UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
        ](unsafe_from_address=base),
        plan.k_tiles,
        plan.act_slots,
    )
    if thread_idx >= feeders:
        hmx_guard_poison(arena.guard)
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        annihilation_matmul_step[native_a, native_c](
            C, A, B, arena, plan, step, thread_idx, feeders, unpack_arm
        )
        if not _matmul_hmx_sync(thread_idx):
            return False
    if thread_idx >= feeders:
        witness.store(MATMUL_WIT_GUARD, UInt32(hmx_guard_damage(arena.guard)))
    if thread_idx == 0:
        witness.store(MATMUL_WIT_UNPACK_ARM, unpack_arm)
    return True


@always_inline
def _anni_matmul_stage[
    native_a: Bool, native_c: Bool
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
    thread_idx: Int,
    array_thread: Int,
    feeders: Int,
) -> Bool:
    """One bench matmul dispatch: `_matmul_stripe_hmx`'s grant discipline
    around `_anni_matmul_case`.

    The first rendezvous both broadcasts the grant outcome and orders thread
    0's fills (done by the caller before this) ahead of every feeder's pack.

    Returns:
        Whether this thread saw the grant taken and every rendezvous
        complete — on a True from thread 0, every participant reached the
        sweep's last barrier, so `C` is whole.
    """
    if thread_idx == array_thread:
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_LOCK_ENTER)
        witness.store(7, _matmul_hmx_open(witness, M, N, K))
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_LOCK_DONE)
    if not _matmul_hmx_sync(thread_idx):
        _matmul_hmx_unwind(witness, thread_idx == array_thread)
        return False
    if witness.load[1](7) != MATMUL_PATH_HMX_RAN:
        return False

    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    var plan = hmx_stripe_plan(M, N, K, arena.budget)
    if thread_idx == array_thread:
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_STEPPING)
        var arm = MATMUL_ARM_FP16
        comptime if HMX_EMIT_MMA:
            arm |= MATMUL_ARM_EMIT_MMA
        witness.store(MATMUL_WIT_ARM, arm)
    hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)

    if not _anni_matmul_case[native_a, native_c](
        C, A, B, witness, arena.base, plan, thread_idx, feeders
    ):
        _matmul_hmx_unwind(witness, thread_idx == array_thread)
        return False

    if thread_idx == array_thread:
        external_call["qshimHmxExit", NoneType]()
        _ = external_call["qshimHmxUnlock", Int32]()
        witness.store(MATMUL_WIT_PHASE, MATMUL_PHASE_DONE)
    return True


@export
def mojo_hmx_annihilation_bench(
    scratch: UnsafePointer[Scalar[DType.uint8], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    arm: Int,
    stage: Int,
    scratch_bytes: Int,
) abi("C"):
    """Runs ONE stage-arm of the rung-1 annihilation bench.

    The host drives the six protocol dispatches in order — (baseline, FF),
    (baseline, swiglu), (baseline, FFO), then the native three — against ONE
    scratch buffer of at least `ANNI_SCRATCH_BYTES`, then the falsify stage
    as the verdict's must-fire control. Thread 0 prepares the stage's own
    buffers (idempotent, so host-side repeats are safe); the intermediates a
    stage consumes arrive from the previous dispatch through the shared
    scratch, which is the handoff under test.

    The witness: words 7-22 are the matmul map (`MATMUL_WIT_*`, the 8-14
    lock-refusal report, the unconditional unpack-arm echo at 22 — the
    native-FF dispatch must echo `HMX_UNPACK_ARM_NATIVE`, the
    treatment-delivery fingerprint); words 25-28 are the bench's marker, arm
    and stage echoes and the verdict count (`hmx_annihilation_bench.mojo`
    names them). The phase table is not copied here — it rides the launch
    timing slot per dispatch when the probe skel is armed, which is the
    whole reason each stage-arm is its own dispatch.
    """
    var thread_idx = _npu_thread_idx()
    var participants = Int(_qshim_query(QSHIM_QUERY_NUM_THREADS))
    var roles = hmx_stripe_roles(participants)

    # Thread-uniform refusals only, ahead of any rendezvous (order 084's
    # rule): pure arithmetic on this kernel's own scalars, identical on every
    # thread, written rather than scribbled past.
    var bad_args = (
        (arm != ANNI_ARM_BASELINE and arm != ANNI_ARM_NATIVE)
        or stage < ANNI_STAGE_FF
        or stage > ANNI_STAGE_FALSIFY
    )
    if (
        bad_args
        or scratch_bytes < ANNI_SCRATCH_BYTES
        or not roles.usable()
        or not matmul_bt_hmx_accepts(ANNI_M, ANNI_FF_N, ANNI_DIM, 0, ANNI_FF_N)
        or not matmul_bt_hmx_accepts(ANNI_M, ANNI_DIM, ANNI_HIDDEN, 0, ANNI_DIM)
    ):
        if thread_idx == 0:
            witness.store(ANNI_WIT_MARKER, ANNI_MARKER_REFUSED)
            witness.store(ANNI_WIT_ARM, UInt32(arm))
            witness.store(ANNI_WIT_STAGE, UInt32(stage))
        return

    var base = align_up(Int(scratch), HMX_TILE_BYTES)

    hmx_phase_open(thread_idx)
    hmx_phase_mark(thread_idx, HMX_PHASE_OPEN)
    if thread_idx == 0:
        witness.store(
            MATMUL_WIT_ROLES, UInt32(participants * 16 + roles.feeders)
        )
        witness.store(ANNI_WIT_ARM, UInt32(arm))
        witness.store(ANNI_WIT_STAGE, UInt32(stage))
        witness.store(ANNI_WIT_VERDICT, ANNI_VERDICT_UNSET)
        annihilation_stage_prepare(base, arm, stage)

    if stage == ANNI_STAGE_FALSIFY:
        # Thread 0 alone, no rendezvous and no grant: the planted mismatch
        # must reach the verdict word through the same path the real verdict
        # takes, with nothing else running.
        if thread_idx == 0:
            annihilation_shim_verdict(base, witness, True)
            witness.store(ANNI_WIT_MARKER, ANNI_MARKER_DONE)
        return

    if stage == ANNI_STAGE_SWIGLU:
        # One rendezvous orders thread 0's seeds ahead of the compute, one
        # orders the compute ahead of the completion marker.
        if not _matmul_hmx_sync(thread_idx):
            _matmul_hmx_unwind(witness, thread_idx == roles.array_thread)
            return
        var ff_rows = annihilation_ff_out(base, ANNI_ARM_BASELINE)
        var mlp_rows = annihilation_mlp_out(base, ANNI_ARM_BASELINE)
        if arm == ANNI_ARM_NATIVE:
            annihilation_swiglu_thread[True](
                ff_rows,
                mlp_rows,
                annihilation_ff_panel(base),
                annihilation_mlp_panel(base),
                ANNI_M,
                ANNI_HIDDEN,
                thread_idx,
                roles.feeders,
            )
        else:
            annihilation_swiglu_thread[False](
                ff_rows,
                mlp_rows,
                annihilation_ff_panel(base),
                annihilation_mlp_panel(base),
                ANNI_M,
                ANNI_HIDDEN,
                thread_idx,
                roles.feeders,
            )
        if not _matmul_hmx_sync(thread_idx):
            _matmul_hmx_unwind(witness, thread_idx == roles.array_thread)
            return
        if thread_idx == 0:
            witness.store(ANNI_WIT_MARKER, ANNI_MARKER_DONE)
        return

    # The four matmul stage-arms: THE arm table, mirrored by the host gate's
    # `_chain_case`, guarded by the fingerprint echo and the bit identity.
    var ok = False
    if stage == ANNI_STAGE_FF:
        var C = annihilation_ff_out(base, arm)
        var A = annihilation_a_mod(base)
        var B = annihilation_w_ff(base)
        if arm == ANNI_ARM_NATIVE:
            ok = _anni_matmul_stage[False, True](
                C,
                A,
                B,
                witness,
                ANNI_M,
                ANNI_FF_N,
                ANNI_DIM,
                thread_idx,
                roles.array_thread,
                roles.feeders,
            )
        else:
            ok = _anni_matmul_stage[False, False](
                C,
                A,
                B,
                witness,
                ANNI_M,
                ANNI_FF_N,
                ANNI_DIM,
                thread_idx,
                roles.array_thread,
                roles.feeders,
            )
    else:
        var C = annihilation_c_out(base, arm)
        var A = annihilation_mlp_out(base, arm)
        var B = annihilation_w_ffo(base)
        if arm == ANNI_ARM_NATIVE:
            ok = _anni_matmul_stage[True, False](
                C,
                A,
                B,
                witness,
                ANNI_M,
                ANNI_DIM,
                ANNI_HIDDEN,
                thread_idx,
                roles.array_thread,
                roles.feeders,
            )
        else:
            ok = _anni_matmul_stage[False, False](
                C,
                A,
                B,
                witness,
                ANNI_M,
                ANNI_DIM,
                ANNI_HIDDEN,
                thread_idx,
                roles.array_thread,
                roles.feeders,
            )
    if not ok or thread_idx != 0:
        return
    # Thread 0's True means every participant reached the sweep's last
    # rendezvous, so every feeder's C store retired before this read.
    if stage == ANNI_STAGE_FFO and arm == ANNI_ARM_NATIVE:
        annihilation_shim_verdict(base, witness, False)
    witness.store(ANNI_WIT_MARKER, ANNI_MARKER_DONE)


# ===--------------------------------------------------------------------------===
# Memory-transport probe — what one core can actually pull out of DDR
#
# The matmul's instruction issue accounts for ~12% of the DiT step, and the
# record cannot say what the rest is: an exposed miss latency the kernel could
# hide, or a transport ceiling no amount of hiding moves. The two are told apart
# by reading the same DDR bytes three ways in one dispatch and comparing the
# rates.
#
# - `MEMBW_MODE_LOADS` is the matmul's own access: HVX vector loads, one
#   dependent stream per thread, nothing in flight ahead of them.
# - `MEMBW_MODE_LOADS_PF` adds the `l2fetch` the matmul now issues, so the L2
#   prefetch engine runs ahead of the same stream.
# - `MEMBW_MODE_UDMA` moves the identical bytes by uDMA into VTCM — the transport
#   a VTCM-staged weight panel would use.
#
# Bytes, source and destination residency are held fixed across the three; only
# how the request is made changes. A rate that rises with the transport is
# exposed latency; a rate that does not is the ceiling.
#
# The participant mask is the second axis. Each thread streams its own fixed
# quarter of the buffer whatever the mask, so 1, 2 and 4 participants read the
# same bytes per thread and differ only in concurrency: a rate that scales with
# participants is per-thread latency, one that saturates is shared.
# ===--------------------------------------------------------------------------===

comptime MEMBW_MODE_UDMA = 0
"""Stage the slice into VTCM by linear uDMA, chunk at a time."""

comptime MEMBW_MODE_LOADS = 1
"""Read the slice with HVX vector loads and nothing else."""

comptime MEMBW_MODE_LOADS_PF = 2
"""`MEMBW_MODE_LOADS` with the next chunk `l2fetch`ed while this one is read."""

comptime MEMBW_MODE_UDMA_ONCE = 3
"""One uDMA transfer and one wait, so a refusal cannot be the submit loop.

`MEMBW_MODE_UDMA` reuses one descriptor across thousands of submissions and
waits on a `dmwait` that drains a hardware thread rather than a transfer, so a
refusal there has three candidate causes at once. This isolates the transfer
itself: one submission, out of the same large buffer, at the same address."""

comptime MEMBW_WITNESS_STRIDE = 8
"""Words a participant reports: the two 64-bit deltas, bytes, chunks, sink,
status."""

comptime MEMBW_STATUS_NO_VTCM: UInt32 = 1 << 16
"""VTCM was not granted, or the chunk does not fit the grant."""

comptime MEMBW_STATUS_TOO_SMALL: UInt32 = 1 << 17
"""The slice does not hold one chunk, so nothing was timed."""

comptime MEMBW_LANES = Int(HVX_VECTOR_SIZE) // size_of[DType.uint32]()
"""Lanes of `uint32` in an HVX vector: 32 in the 128-byte mode."""

comptime MEMBW_PF_ROW_BYTES = 4096
"""Width of one `l2fetch` descriptor row. Under the descriptor's 16-bit width
field, and a whole number of them covers any chunk this probe accepts."""


@always_inline
def _read_upcycle() -> UInt64:
    """Core cycles, from the user-readable UPCYCLE counter pair (`c15:14`)."""
    return llvm_intrinsic[
        "llvm.readcyclecounter", UInt64, has_side_effect=True
    ]()


@always_inline
def _read_utimer() -> UInt64:
    """The fixed-rate 19.2 MHz QTimer (`c31:30`), which no clock change scales.

    Read alongside UPCYCLE so a rate is reported against wall time rather than
    against a core clock that DVFS may have moved under a memory-bound load.
    """
    return inlined_assembly[
        "$0 = utimer",
        UInt64,
        constraints="=r",
        has_side_effect=True,
    ]()


@always_inline
def _membw_stream_loads[
    prefetch: Bool
](
    base: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    slice_words: Int,
    chunk_words: Int,
) -> UInt32:
    """Reads `slice_words` with HVX vector loads, returning a fold of them.

    The fold is what keeps the loads: an unread stream is dead code, and a
    kernel that was optimized down to its loop counter would report the
    transport's rate for no traffic at all.
    """
    var acc = SIMD[DType.uint32, MEMBW_LANES](0)
    var off = 0
    while off + chunk_words <= slice_words:
        comptime if prefetch:
            var next = off + chunk_words
            if next + chunk_words <= slice_words:
                _l2fetch_rows(
                    (base + next).bitcast[Scalar[GEMV_DTYPE]](),
                    MEMBW_PF_ROW_BYTES,
                    chunk_words * size_of[DType.uint32]() // MEMBW_PF_ROW_BYTES,
                )
        var i = off
        while i < off + chunk_words:
            acc ^= base.load[MEMBW_LANES](i)
            i += MEMBW_LANES
        off += chunk_words

    var sink: UInt32 = 0
    for lane in range(MEMBW_LANES):
        sink ^= acc[lane]
    return sink


@export
def mojo_npu_membw_probe(
    src: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    words: Int,
    chunk_bytes: Int,
    mode: Int,
) abi("C"):
    """Times one thread's read of a fixed DDR slice through one transport.

    Thread `t` always takes quarter `t` of `words`, whatever the participant
    mask, so the mask varies concurrency and nothing else. `chunk_bytes` is the
    uDMA transfer size, the `l2fetch` group size, and the load loop's step, so
    the three modes walk the slice in the same order at the same granularity.

    Reports, per participant: the UPCYCLE and QTimer deltas around the timed
    loop as low/high word pairs, the bytes it moved, the chunks it took, the
    fold of what it read, and a status word.
    """
    var thread_idx = _npu_thread_idx()
    var report = witness + thread_idx * MEMBW_WITNESS_STRIDE
    for j in range(MEMBW_WITNESS_STRIDE):
        report.store(j, 0)

    if thread_idx >= Int(NUM_COMPUTE_THREADS):
        return

    # A quarter each, rounded down to a whole vector so no slice starts mid-line.
    var slice_words = (
        (words // Int(NUM_COMPUTE_THREADS)) // MEMBW_LANES
    ) * MEMBW_LANES
    var chunk_words = chunk_bytes // size_of[DType.uint32]()
    if slice_words < chunk_words or chunk_words == 0:
        report.store(7, MEMBW_STATUS_TOO_SMALL)
        return
    var base = src + thread_idx * slice_words

    # Held as an address rather than a pointer because `UnsafePointer` is
    # non-nullable, and the load modes have no VTCM slice to name.
    var staged_addr = 0
    if mode == MEMBW_MODE_UDMA or mode == MEMBW_MODE_UDMA_ONCE:
        var vtcm = Int(_qshim_get_base_vtcm_addr())
        var stage_bytes = align_up(chunk_bytes, Int(HVX_VECTOR_SIZE))
        if vtcm == 0 or stage_bytes * Int(NUM_COMPUTE_THREADS) > Int(
            _qshim_get_vtcm_size()
        ):
            report.store(7, MEMBW_STATUS_NO_VTCM)
            return
        staged_addr = (
            align_up(vtcm, Int(HVX_VECTOR_SIZE)) + thread_idx * stage_bytes
        )

    var sink: UInt32 = 0
    var status: UInt32 = 0
    var chunks = 0

    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()

    if mode == MEMBW_MODE_UDMA or mode == MEMBW_MODE_UDMA_ONCE:
        var staged = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=staged_addr
        )
        var stop = slice_words if mode == MEMBW_MODE_UDMA else chunk_words
        var off = 0
        while off + chunk_words <= stop:
            var submitted: UInt32 = 0
            var handle = _qshim_linear_udma_submit(
                UInt32(thread_idx),
                (base + off).bitcast[UInt8](),
                UInt32(chunk_bytes),
                staged,
                UnsafePointer(to=submitted).as_unsafe_any_origin(),
            )
            _ = _qshim_udma_wait(handle)
            status |= submitted
            off += chunk_words
            chunks += 1
    elif mode == MEMBW_MODE_LOADS_PF:
        sink = _membw_stream_loads[True](base, slice_words, chunk_words)
        chunks = slice_words // chunk_words
    else:
        sink = _membw_stream_loads[False](base, slice_words, chunk_words)
        chunks = slice_words // chunk_words

    var ticks1 = _read_utimer()
    var cycles1 = _read_upcycle()

    if mode == MEMBW_MODE_UDMA or mode == MEMBW_MODE_UDMA_ONCE:
        # Read after the bracket: what landed in VTCM proves the transfer ran,
        # and reading it inside the loop would time a VTCM load as well.
        sink = UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin](
            unsafe_from_address=staged_addr
        ).load(0)

    var cycles = cycles1 - cycles0
    var ticks = ticks1 - ticks0
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(4, UInt32(chunks * chunk_bytes))
    report.store(5, UInt32(chunks))
    report.store(6, sink)
    report.store(7, status)


# ===--------------------------------------------------------------------------===
# Operand-port probe — what the HMX operand stream itself moves (order 159R)
#
# `membw_probe` above times HVX vector loads. That is a core-side read rate, and
# reading it as the ARRAY's operand-port rate is what order 159R exists to
# correct: no probe in any branch has ever issued a single `mxmem`, so the port
# has never been measured. This one's inner loop IS the operand stream.
#
# Target: Hexagon v81 with `+hmx`, fp16 only. The int8 forms the ISA defines do
# not assemble on this tree (`hmx_int8.mojo`'s form table), so an integer arm
# here would report the compiler's gap as the port's rate.
#
# Three properties make the number a rate rather than a plausible figure:
#
#   - The bytes come from the GEOMETRY, not from a count the loop keeps. Every
#     pass streams `2 * tiles` croutons, so `hmx_port_probe_bytes` closes the
#     denominator with nothing measured in it. It excludes the record and the
#     readout, which makes every rate a FLOOR -- and the excluded share grows as
#     `1/tiles`, so a geometry too small to keep it under 2% is REFUSED rather
#     than reported (`hmx_port_probe_min_tiles`, and `MXMEM_STATUS_GEOMETRY`
#     for what a too-small one would have done to 159R sec 7).
#   - The operands are VTCM-RESIDENT. `mxmem` requires it (order 096), and the
#     grant bounds the working set at 8 MiB, so a geometry that does not fit is
#     refused rather than run against DDR.
#   - The readout fold travels back as the sink the reader already prints. A
#     zero sink is an array that never wrote, whatever the elapsed time says --
#     the same control `membw_probe`'s sink is, for the same reason. It is a SUM
#     of the readout's bit patterns and NOT the XOR this file shipped first:
#     every element of this probe's readout is identical by construction, so an
#     XOR fold was constantly zero and the control reported failure on every
#     successful run (`hmx_port_probe_fold`).
#
# `form` selects the ADDRESSING, and it is the only difference between the two
# arms: `:deep` covers 32 croutons per packet, plain covers one, and both move
# the same bytes. So a rate difference between them is an issue-count effect and
# nothing else, which is the isolation `COLLISION_SCHEDULE_DESIGN.md` needs.
# ===--------------------------------------------------------------------------===

comptime MXMEM_FORM_DEEP = 0
"""Stream the activation with `:deep`: one packet per `HMX_MAX_DEEP_TILES`."""

comptime MXMEM_FORM_PLAIN = 1
"""Stream it plain: one packet per crouton, same bytes, more issues."""

comptime MXMEM_STATUS_NOT_BUILT: UInt32 = 1 << 16
"""This build has no array arm, so the probe modelled nothing and ran nothing.

Not a failure of the device: `SKYE_HMX_EMIT_MMA` is off, which is the default,
and the modelled arm is many orders too slow to time. A rate must never be
reported from a run carrying this bit."""

comptime MXMEM_STATUS_NO_VTCM: UInt32 = 1 << 17
"""No VTCM grant, or the geometry does not fit the one there is."""

comptime MXMEM_STATUS_NO_GRANT: UInt32 = 1 << 18
"""The HMX unit was refused. The low 16 bits carry the lock's own status."""

comptime MXMEM_STATUS_GEOMETRY: UInt32 = 1 << 19
"""The geometry cannot produce a rate this probe is willing to report.

Three ways, and all three are refusals rather than caveats because each one
turns into a plausible WRONG number instead of an obvious failure:

- `iters < 1`, so nothing is timed.
- The bytes exceed the witness word. The count rides one 32-bit word to keep
  this probe on `membw_probe`'s reader unchanged, so a geometry naming more
  than 4 GiB is refused here rather than reported modulo 2^32.
- `tiles < hmx_port_probe_min_tiles()` (29 at fp16). The denominator excludes
  the record and the readout, an understatement of `0.5625/tiles`, and below
  29 croutons that exceeds the 2% the docstring promises. It is a refusal and
  not a footnote because of what it does downstream: at 8 croutons the 7.03%
  understatement reports a true ~1050 GB/s as ~977 and fires order 159R sec 7's
  below-984 CONTRADICTION row -- the row that exists to catch a probe which is
  not on the operand path -- on an artifact of this instrument."""

comptime MXMEM_FILL_PERIOD = 7
"""Distinct fill magnitudes, coprime with the tile side and the slab pairing.

The fill is synthetic and its VALUES do not enter any claim, but a constant one
makes the readout fold the same number for every geometry, which is a control
that cannot fail. Coprime with 32 and 16 so it does not fold into the crouton's
own periods (the trap KB `kernels/hmx-crouton-tile-layout` records)."""

comptime MXMEM_FILL_STEP = 0.015625
"""`1/64`: small enough that even a 1024-tile pass stays inside fp16."""


@always_inline
def _mxmem_fill_panel[
    space: AddressSpace, //
](panel: CroutonPanelOf[HMX_DTYPE, space], tiles: Int, seed: Int):
    """Fills `tiles` croutons with a deterministic non-zero pattern.

    A slab pair is 64 contiguous fp16 under this view, so the walk is by pair
    and the store is one vector -- `hmx_attn_zero_panel`'s shape, with a value
    that varies so the readout fold is not a constant.

    Runs BEFORE the HMX unit is taken. A thread holding the array must execute
    no HVX, and this is the only vector work the probe does.
    """
    comptime PAIR_ELEMS = HMX_TILE * HMX_SPLIT
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    for t in range(tiles):
        for pair in range(SLAB_PAIRS):
            var step = Scalar[HMX_DTYPE](
                Float32((t + pair + seed) % MXMEM_FILL_PERIOD + 1)
                * MXMEM_FILL_STEP
            )
            panel.store[PAIR_ELEMS, alignment=ALIGN](
                Coord(t, pair, 0, 0), SIMD[HMX_DTYPE, PAIR_ELEMS](step)
            )


@always_inline
def _mxmem_stream_loop[
    space: AddressSpace, //, deep: Bool
](scratch: HmxScratch[space, HMX_DTYPE], tiles: Int, iters: Int):
    """Issues `iters` complete MMAs over the staged panels. THE TIMED BODY.

    Scalar by construction, which is the rule a thread holding the unit lives
    under: the loop is a counter and the chain is inline assembly.
    """
    for _ in range(iters):
        hmx_port_probe_chain[deep=deep](
            scratch.out,
            scratch.record,
            HmxOperands[space, HMX_DTYPE](scratch.act, scratch.wgt, tiles),
        )


@export
def mojo_npu_mxmem_port_probe(
    report_buffer: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    tiles: Int,
    iters: Int,
    form: Int,
) abi("C"):
    """Times `iters` fp16 MMA passes over `tiles` VTCM-resident crouton pairs.

    Reports through `membw_probe`'s witness layout so the same reader decodes
    both: the UPCYCLE and QTimer deltas around the timed loop as low/high word
    pairs, the operand bytes moved, the operand-load PACKETS issued, the fold of
    what the array wrote, and a status word.

    Runs on launch index 0 alone. The HMX unit is one resource and the grant is
    process-wide lazy state, so a second participant would be racing for it
    rather than adding concurrency.

    Args:
        report_buffer: The witness, `MEMBW_WITNESS_STRIDE` words per thread.
        tiles: Croutons of depth each pass streams, per operand.
        iters: Passes to time.
        form: `MXMEM_FORM_DEEP` or `MXMEM_FORM_PLAIN`.
    """
    var thread_idx = _npu_thread_idx()
    var report = report_buffer + thread_idx * MEMBW_WITNESS_STRIDE
    for j in range(MEMBW_WITNESS_STRIDE):
        report.store(j, 0)

    if thread_idx != 0:
        return

    comptime if not HMX_EMIT_MMA:
        report.store(7, MXMEM_STATUS_NOT_BUILT)
        return

    var bytes = hmx_port_probe_bytes[HMX_DTYPE](tiles, iters)
    if (
        tiles < hmx_port_probe_min_tiles[HMX_DTYPE]()
        or iters < 1
        or bytes > UInt64(0xFFFFFFFF)
    ):
        report.store(7, MXMEM_STATUS_GEOMETRY)
        return

    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    if not arena.usable() or hmx_scratch_bytes(tiles) > arena.budget:
        report.store(7, MXMEM_STATUS_NO_VTCM)
        return

    var scratch = hmx_scratch_in[MATMUL_HMX_SPACE](
        UnsafePointer[
            UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
        ](unsafe_from_address=arena.base),
        tiles,
    )
    _mxmem_fill_panel(scratch.act, tiles, 0)
    _mxmem_fill_panel(scratch.wgt, tiles, 3)
    # Zeroed BEFORE the run, which is what makes the fold below a witness: the
    # readout tile is VTCM this launch does not own between dispatches, so a
    # non-zero fold read out of a tile nobody zeroed would be last dispatch's.
    hmx_attn_zero_panel(scratch.out, 1)
    hmx_program_identity_record(scratch.record)

    # The grant, taken exactly as the matmul stripe takes it and held across
    # every pass: a release clears the accumulators, so cycling it per pass
    # would be timing the lock rather than the port.
    var lock_rc = external_call["qshimHmxLock", Int32]()
    if Int(lock_rc) != 0:
        report.store(7, MXMEM_STATUS_NO_GRANT | UInt32(Int(lock_rc) & 0xFFFF))
        return
    external_call["qshimHmxEnter", NoneType]()

    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()
    if form == MXMEM_FORM_PLAIN:
        _mxmem_stream_loop[deep=False](scratch, tiles, iters)
    else:
        _mxmem_stream_loop[deep=True](scratch, tiles, iters)
    var ticks1 = _read_utimer()
    var cycles1 = _read_upcycle()

    external_call["qshimHmxExit", NoneType]()
    _ = external_call["qshimHmxUnlock", Int32]()

    var issues = hmx_port_probe_issues[deep=True](tiles)
    if form == MXMEM_FORM_PLAIN:
        issues = hmx_port_probe_issues[deep=False](tiles)

    var cycles = cycles1 - cycles0
    var ticks = ticks1 - ticks0
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(4, UInt32(bytes & UInt64(0xFFFFFFFF)))
    report.store(5, UInt32(issues * iters))
    report.store(6, hmx_port_probe_fold(scratch.out))
    report.store(7, 0)


# ===--------------------------------------------------------------------------===
# `hmx_bench` — the standalone HMX bench's device entry (re-sign window item 8)
#
# Pass bodies, fills, accounting and refusal floors live in
# `hexagon_ondevice.hmx_standalone_bench`, where the host gate runs them
# modelled; this is the thin shim the `.def` row names. The two probes above
# are its parents, and the split is deliberate: modes 0-2 are
# `mxmem_port_probe`'s bracket over the module's dual-dtype arms — adding the
# iters-amortized int8 sibling order 184's harness could not express — and
# modes 3-4 are `membw_probe`'s walk in the direction it lacks, VTCM->DDR
# with per-transfer completion, by uDMA and by HVX stores. One reader decodes
# all three probes: `MEMBW_WITNESS_STRIDE` words per thread, status non-zero
# iff the case refused or a control failed.
#
# The uDMA arm carries a ring-depth knob (scalar 3, window item 9): at the
# default `depth=1` it keeps the original submit-then-wait shape verbatim —
# the pre-registered depth-1 regression arm — and at `depth > 1` it runs the
# reference consumer's push-ahead schedule (prime, pop-then-push, counted
# drain, epilogue flush) over the item-9 descriptor ring, which is DRAFT-092
# leg A' re-hosted on this body. A depth-1 row stays comparable to every
# rate quoted before the ring landed (`HMX_BENCH_NOTES.md` section 2b, delta
# finding 1); a deeper row's rate is a ring rate and says so by its depth.
# ===--------------------------------------------------------------------------===


@always_inline
def _hmx_bench_array[
    dtype: DType, form: Int
](
    report: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    k_tiles: Int,
    iters: Int,
):
    """One pure-array case: carve, stage, take the grant, bracket, report.

    `mojo_npu_mxmem_port_probe`'s body over the bench module's shared pass,
    so the fp16 rows here and selector 42's rows are the same instrument and
    the int8 row differs from them by the arm alone. The caller has already
    run the geometry gate.
    """
    comptime if not HMX_EMIT_MMA:
        report.store(7, BENCH_STATUS_NOT_BUILT)
        return

    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    if not arena.usable() or hmx_scratch_bytes[dtype](k_tiles) > arena.budget:
        report.store(7, BENCH_STATUS_NO_VTCM)
        return
    var scratch = hmx_scratch_in[MATMUL_HMX_SPACE, dtype](
        UnsafePointer[
            UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
        ](unsafe_from_address=arena.base),
        k_tiles,
    )
    bench_array_stage(scratch, k_tiles)

    # Held across every pass exactly as `mxmem_port_probe` holds it: a
    # release clears the accumulators, so cycling the grant per pass would
    # time the lock rather than the port.
    var lock_rc = external_call["qshimHmxLock", Int32]()
    if Int(lock_rc) != 0:
        report.store(7, BENCH_STATUS_NO_GRANT | UInt32(Int(lock_rc) & 0xFFFF))
        return
    external_call["qshimHmxEnter", NoneType]()

    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()
    var arm = bench_array_stream[form=form](scratch, k_tiles, iters)
    var ticks1 = _read_utimer()
    var cycles1 = _read_upcycle()

    external_call["qshimHmxExit", NoneType]()
    _ = external_call["qshimHmxUnlock", Int32]()

    # Plain addressing is one packet per crouton at either operand width;
    # only the fp16 `:deep` arm batches croutons into packets.
    var issues = k_tiles
    comptime if form == BENCH_FORM_DEEP:
        issues = hmx_port_probe_issues[deep=True](k_tiles)

    var status: UInt32 = 0
    if arm != (bench_expected_arm[dtype]() | BENCH_ARM_ISSUED):
        status = BENCH_STATUS_WRONG_ARM | (arm & 0xFFFF)

    var bytes = hmx_port_probe_bytes[dtype](k_tiles, iters)
    var cycles = cycles1 - cycles0
    var ticks = ticks1 - ticks0
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(4, UInt32(bytes & UInt64(0xFFFFFFFF)))
    report.store(5, UInt32(issues * iters))
    report.store(6, bench_readout_fold(scratch.out))
    report.store(7, status)


@always_inline
def _hmx_bench_instripe(
    sink: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    report: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    k_tiles: Int,
    iters: Int,
    depth: Int,
):
    """The in-stripe residency case: every pass's operands arrive by uDMA
    from DDR INSIDE the bracket, then the same chain the pure-array modes
    time consumes them — `_hmx_bench_array`'s instrument with the staging
    it amortizes away put back, which is the whole point.

    The DDR source is the movement modes' sink, filled here outside the
    bracket with pass-distinct crouton patterns; pass `j` stages source
    slot `j` and the walk never revisits a line. VTCM holds `depth`
    scratch sets; at depth 2 the next pass's one allowed outstanding
    transfer (order 096) is submitted before the current chain issues, so
    the row measures what the engine hides behind the MMA with ZERO
    cross-thread schedule in it. Read the three rows together: mode 0
    (staging amortized), this at depth 1 (staging serial), this at depth 2
    (staging overlapped) — `bench_compose_floor_ticks` and
    `bench_overlap_efficiency_ppm` turn them into the split the parked
    232A instrument could not deliver. The caller has already run the
    case gate.
    """
    comptime if not HMX_EMIT_MMA:
        report.store(7, BENCH_STATUS_NOT_BUILT)
        return

    var slot_bytes = bench_instripe_slot_bytes(k_tiles)
    var slot_stride = align_up(
        hmx_scratch_bytes[HMX_DTYPE](k_tiles), Int(HMX_TILE_BYTES)
    )
    var arena = StepArena[MATMUL_HMX_SPACE].at(
        Int(_qshim_get_base_vtcm_addr()), Int(_qshim_get_vtcm_size())
    )
    if not arena.usable() or depth * slot_stride > arena.budget:
        report.store(7, BENCH_STATUS_NO_VTCM)
        return

    # The DDR source: `iters` operand slots, filled with the array modes'
    # own pass-distinct patterns (the salt carries the pass index so no two
    # passes can be swapped and still agree), then never written again.
    var panel_bytes = slot_bytes // 2
    for j in range(iters):
        var slot_base = sink + j * slot_bytes
        bench_fill_fp16_croutons(
            hmx_panel[AddressSpace.GENERIC, HMX_DTYPE](
                slot_base.bitcast[Scalar[HMX_DTYPE]]().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                k_tiles,
            ),
            k_tiles,
            BENCH_ACT_SALT + j,
        )
        bench_fill_fp16_croutons(
            hmx_panel[AddressSpace.GENERIC, HMX_DTYPE](
                (slot_base + panel_bytes)
                .bitcast[Scalar[HMX_DTYPE]]()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                k_tiles,
            ),
            k_tiles,
            BENCH_WGT_SALT + j,
        )

    # `depth` scratch sets: operands land in the first `slot_bytes` of each
    # (act then wgt, adjacent — one descriptor per pass); the readout and
    # record live per set, programmed here, staged never.
    for s in range(depth):
        var scratch = hmx_scratch_in[MATMUL_HMX_SPACE, HMX_DTYPE](
            UnsafePointer[
                UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
            ](unsafe_from_address=arena.base + s * slot_stride),
            k_tiles,
        )
        bench_zero_readout(scratch.out)
        bench_array_program_record(scratch)

    var lock_rc = external_call["qshimHmxLock", Int32]()
    if Int(lock_rc) != 0:
        report.store(7, BENCH_STATUS_NO_GRANT | UInt32(Int(lock_rc) & 0xFFFF))
        return
    external_call["qshimHmxEnter", NoneType]()

    var arm: UInt32 = 0
    var status: UInt32 = 0
    var fold_slot = 0
    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()
    if depth <= 1:
        var scratch = hmx_scratch_in[MATMUL_HMX_SPACE, HMX_DTYPE](
            UnsafePointer[
                UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
            ](unsafe_from_address=arena.base),
            k_tiles,
        )
        var dst = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=arena.base
        )
        for j in range(iters):
            var submitted: UInt32 = 0
            var handle = _qshim_linear_udma_submit(
                UInt32(0),
                sink + j * slot_bytes,
                UInt32(slot_bytes),
                dst,
                UnsafePointer(to=submitted).as_unsafe_any_origin(),
            )
            status |= _qshim_udma_wait(handle)
            status |= submitted
            arm = bench_array_pass[form=BENCH_FORM_DEEP](scratch, k_tiles)
    else:
        # Prologue: slot 0 staged and awaited; steady state submits pass
        # `j+1`'s single allowed outstanding transfer BEFORE chaining pass
        # `j`, and awaits it after — the engine works behind the MMA.
        var submitted: UInt32 = 0
        var handle = _qshim_linear_udma_submit(
            UInt32(0),
            sink,
            UInt32(slot_bytes),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=arena.base
            ),
            UnsafePointer(to=submitted).as_unsafe_any_origin(),
        )
        status |= _qshim_udma_wait(handle)
        status |= submitted
        for j in range(iters):
            var pending = j + 1 < iters
            if pending:
                var next_submitted: UInt32 = 0
                _ = _qshim_linear_udma_submit(
                    UInt32(0),
                    sink + (j + 1) * slot_bytes,
                    UInt32(slot_bytes),
                    UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                        unsafe_from_address=arena.base
                        + ((j + 1) % depth) * slot_stride
                    ),
                    UnsafePointer(to=next_submitted).as_unsafe_any_origin(),
                )
                status |= next_submitted
            var scratch = hmx_scratch_in[MATMUL_HMX_SPACE, HMX_DTYPE](
                UnsafePointer[
                    UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
                ](unsafe_from_address=arena.base + (j % depth) * slot_stride),
                k_tiles,
            )
            fold_slot = j % depth
            arm = bench_array_pass[form=BENCH_FORM_DEEP](scratch, k_tiles)
            if pending:
                status |= _qshim_udma_await(UInt32(0))
    var ticks1 = _read_utimer()
    var cycles1 = _read_upcycle()

    external_call["qshimHmxExit", NoneType]()
    _ = external_call["qshimHmxUnlock", Int32]()

    if arm != (bench_expected_arm[HMX_DTYPE]() | BENCH_ARM_ISSUED):
        status |= BENCH_STATUS_WRONG_ARM | (arm & 0xFFFF)

    var fold_scratch = hmx_scratch_in[MATMUL_HMX_SPACE, HMX_DTYPE](
        UnsafePointer[
            UInt8, MutUntrackedOrigin, address_space=MATMUL_HMX_SPACE
        ](unsafe_from_address=arena.base + fold_slot * slot_stride),
        k_tiles,
    )
    var cycles = cycles1 - cycles0
    var ticks = ticks1 - ticks0
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(
        4, UInt32(bench_instripe_ddr_bytes(k_tiles, iters) & UInt64(0xFFFFFFFF))
    )
    report.store(5, UInt32(hmx_port_probe_issues[deep=True](k_tiles) * iters))
    report.store(6, bench_readout_fold(fold_scratch.out))
    report.store(7, status)


@always_inline
def _hmx_bench_move[
    udma: Bool
](
    sink: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    report: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
    k_tiles: Int,
    iters: Int,
    depth: Int,
):
    """One participant's VTCM->DDR walk, by uDMA or by HVX stores.

    `membw_probe`'s discipline in the direction it lacks: thread `t` owns
    the fixed sink region `t * chunk * iters` whatever the mask, so the mask
    varies concurrency and nothing else, and no line is written twice — a
    rate here cannot be an L2 write-hit rate re-labelled.

    `depth` (uDMA only; the case gate holds it at 1 elsewhere) is the ring
    depth of window item 9's descriptor ring. At 1 the arm keeps its
    original shape verbatim — submit one transfer, wait it out, per chunk —
    the depth-1 regression leg's arm, one variable away from the pre-ring
    baseline. Deeper, it runs the reference consumer's push-ahead
    (`matmul-ops.c:346-400` at pin `1692f9e50bb`): prime `depth` submits,
    steady-state pop-the-oldest-then-push, a counted drain, and the
    section-6 epilogue flush before returning — this launch thread is the
    port's first deep ring consumer, so it owes the drain-before-exit rule
    it was designed under. `iters == depth` runs prime-plus-drain with zero
    steady chunks: the pre-registered prologue-in-isolation row, which is
    how leg A' gets its second number without a witness-layout change.
    """
    var chunk_bytes = bench_move_chunk_bytes(k_tiles)
    var stage_bytes = align_up(chunk_bytes, Int(HVX_VECTOR_SIZE))
    var vtcm = Int(_qshim_get_base_vtcm_addr())
    if vtcm == 0 or stage_bytes * Int(NUM_COMPUTE_THREADS) > Int(
        _qshim_get_vtcm_size()
    ):
        report.store(7, BENCH_STATUS_NO_VTCM)
        return
    var staged_addr = (
        align_up(vtcm, Int(HVX_VECTOR_SIZE)) + thread_idx * stage_bytes
    )

    # The staged source, filled once, non-zero and thread-distinct: what the
    # sink read-back below proves landed.
    var staged_words = UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin](
        unsafe_from_address=staged_addr
    )
    var chunk_words = chunk_bytes // size_of[DType.uint32]()
    for i in range(chunk_words):
        staged_words.store(i, (UInt32(i) * 2654435761) ^ UInt32(thread_idx + 1))

    var region = sink + thread_idx * (chunk_bytes * iters)
    var chunks = 0
    var status: UInt32 = 0

    var cycles0 = _read_upcycle()
    var ticks0 = _read_utimer()
    comptime if udma:
        var staged = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=staged_addr
        )
        if depth <= 1:
            var off = 0
            for _ in range(iters):
                var submitted: UInt32 = 0
                var handle = _qshim_linear_udma_submit(
                    UInt32(thread_idx),
                    staged,
                    UInt32(chunk_bytes),
                    region + off,
                    UnsafePointer(to=submitted).as_unsafe_any_origin(),
                )
                _ = _qshim_udma_wait(handle)
                status |= submitted
                off += chunk_bytes
                chunks += 1
        else:
            var off = 0
            for _ in range(depth):
                var submitted: UInt32 = 0
                _ = _qshim_linear_udma_submit(
                    UInt32(thread_idx),
                    staged,
                    UInt32(chunk_bytes),
                    region + off,
                    UnsafePointer(to=submitted).as_unsafe_any_origin(),
                )
                status |= submitted
                off += chunk_bytes
                chunks += 1
            for _ in range(iters - depth):
                # The oldest transfer is the one popped; the slot it frees
                # takes the next submit. An await's 1 is a wedged engine,
                # folded into the status word so the row refuses.
                status |= _qshim_udma_await(UInt32(thread_idx))
                var submitted: UInt32 = 0
                _ = _qshim_linear_udma_submit(
                    UInt32(thread_idx),
                    staged,
                    UInt32(chunk_bytes),
                    region + off,
                    UnsafePointer(to=submitted).as_unsafe_any_origin(),
                )
                status |= submitted
                off += chunk_bytes
                chunks += 1
            for _ in range(depth):
                status |= _qshim_udma_await(UInt32(thread_idx))
    else:
        var dst_words = region.bitcast[Scalar[DType.uint32]]()
        var out = 0
        for _ in range(iters):
            var i = 0
            while i < chunk_words:
                dst_words.store(out + i, staged_words.load[MEMBW_LANES](i))
                i += MEMBW_LANES
            out += chunk_words
            chunks += 1
    var ticks1 = _read_utimer()
    var cycles1 = _read_upcycle()

    comptime if udma:
        if depth > 1:
            # Outside the bracket, deliberately: the counted drain above
            # already emptied the ring, so this prices nothing. It is the
            # drain-before-exit rule made unconditional for the port's
            # first deep consumer; a nonzero return means the bookkeeping
            # and the engine disagree, and the row refuses.
            status |= _qshim_udma_flush(UInt32(thread_idx))

    # Read after the bracket, from DDR: a sink the writes never reached
    # reads back what the host staged (the driver zeroes it), not this
    # thread's pattern.
    var sink_word = region.bitcast[Scalar[DType.uint32]]().load(0)

    var cycles = cycles1 - cycles0
    var ticks = ticks1 - ticks0
    report.store(0, UInt32(cycles & 0xFFFFFFFF))
    report.store(1, UInt32(cycles >> 32))
    report.store(2, UInt32(ticks & 0xFFFFFFFF))
    report.store(3, UInt32(ticks >> 32))
    report.store(4, UInt32(chunks * chunk_bytes))
    report.store(5, UInt32(chunks))
    report.store(6, sink_word)
    report.store(7, status)


@export
def mojo_npu_hmx_bench(
    sink: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    report_buffer: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    mode: Int,
    k_tiles: Int,
    iters: Int,
    depth: Int,
) abi("C"):
    """The standalone HMX bench: pure-array fp16/int8 and VTCM->DDR movement.

    Modes 0-2 time `iters` complete MMA passes over `k_tiles` VTCM-resident
    crouton pairs on launch index 0 alone — `mxmem_port_probe`'s bracket
    over the bench module's shared pass bodies. Modes 3-4 walk `iters`
    chunks per participant out to the DDR sink, the chunk being feeder 0's
    share of the `k_tiles` panel. Reports through `membw_probe`'s witness
    layout so one reader decodes all three probes: the two clock deltas as
    low/high pairs, the bytes the geometry names, the requests (operand
    packets, or chunks), the readout fold or sink read-back, and a status
    word that is zero iff the case delivered.

    Args:
        sink: The movement modes' DDR destination. The pure-array modes
            never touch it — they stage their own VTCM operands — but the
            wire requires every declared buffer non-empty, so the driver
            sends a token page.
        report_buffer: The witness, `MEMBW_WITNESS_STRIDE` words per thread.
        mode: `BENCH_MODE_*`.
        k_tiles: Croutons of depth per pass (modes 0-2), or the panel whose
            feeder share is the transfer grain (modes 3-4).
        iters: Passes (modes 0-2) or chunks per participant (modes 3-4).
        depth: The uDMA mode's ring depth (window item 9's knob), 1
            elsewhere; `bench_case_status` refuses anything else by name.
    """
    var thread_idx = _npu_thread_idx()
    var report = report_buffer + thread_idx * MEMBW_WITNESS_STRIDE
    for j in range(MEMBW_WITNESS_STRIDE):
        report.store(j, 0)

    var refused = bench_case_status(mode, k_tiles, iters, depth)
    if refused != 0:
        report.store(7, refused)
        return

    if mode == BENCH_MODE_UDMA_OUT or mode == BENCH_MODE_STORES_OUT:
        if thread_idx >= Int(NUM_COMPUTE_THREADS):
            return
        if mode == BENCH_MODE_UDMA_OUT:
            _hmx_bench_move[udma=True](
                sink, report, thread_idx, k_tiles, iters, depth
            )
        else:
            _hmx_bench_move[udma=False](
                sink, report, thread_idx, k_tiles, iters, depth
            )
        return

    # Launch index 0 alone: the HMX unit is one resource and the grant is
    # process-wide lazy state (`mojo_npu_mxmem_port_probe`'s rule).
    if thread_idx != 0:
        return
    if mode == BENCH_MODE_FP16_INSTRIPE:
        _hmx_bench_instripe(sink, report, k_tiles, iters, depth)
    elif mode == BENCH_MODE_INT8_PLAIN:
        _hmx_bench_array[HMX_INT8_DTYPE, BENCH_FORM_PLAIN](
            report, k_tiles, iters
        )
    elif mode == BENCH_MODE_FP16_PLAIN:
        _hmx_bench_array[HMX_DTYPE, BENCH_FORM_PLAIN](report, k_tiles, iters)
    else:
        _hmx_bench_array[HMX_DTYPE, BENCH_FORM_DEEP](report, k_tiles, iters)


# ===--------------------------------------------------------------------------===
# `sinusoid_embed` / `rope_table` — the fp32 table builders of the FLUX.2
# denoise preamble
#
# The timestep sinusoidal embedding (diffusers `get_timestep_embedding`) and the
# per-axis rope (cos, sin) table (diffusers `get_1d_rotary_pos_embed`, in the
# pair layout the signed rope kernels read), imported from
# `hexagon_ondevice.sincos_tables` where they are authored and host-validated.
#
# Both are deliberately SCALAR, which for these two kernels is a correctness
# feature: the device check demands bit-exact equality against a host reference,
# and that only holds while the two sides compile the same source. The module
# owns the argument, down to which constants are spelled to how many digits; the
# launch-striped row split and the witness came with them.
# ===--------------------------------------------------------------------------===


@export
def mojo_sinusoid_embed_f16(
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    t: UnsafePointer[Scalar[DType.float32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    time_scale: Float32,
    neg_ln_max_period: Float32,
    flip: Int,
) abi("C"):
    """Timestep sinusoidal embedding table, spread across the launch's
    threads."""
    sinusoid_embed_entry(
        output,
        t,
        witness,
        num_rows,
        half_dim,
        time_scale,
        neg_ln_max_period,
        flip,
    )


@export
def mojo_rope_table_f32(
    output: UnsafePointer[Scalar[DType.float32], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.int64], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[DType.float32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    pos_stride: Int,
) abi("C"):
    """One rope axis's fp32 (cos, sin) table, spread across the launch's
    threads."""
    rope_table_entry(
        output, positions, freqs, witness, num_rows, half_dim, pos_stride
    )


# ===------------------------------------------------------------------=== #
# The megastep walker's C seam (P5 megakernel slice 1)
# ===------------------------------------------------------------------=== #
#
# The walk itself lives in `hexagon_ondevice.megastep_walker`, imported rather
# than folded for the same reason the crouton layout is: it has a host twin,
# and a hand-copied decoder is exactly the drift a shared module prevents. Its
# `@export` sits here because this file is what the device build compiles as a
# top-level module, so this is where an `abi("C")` symbol is emitted from.
#
# Three things stay in C and are reached through `external_call`: the manifest's
# arity switch, which is an X-macro expansion of `NPUKernels.def` that Mojo has
# no preprocessor to reproduce, and the launch body, which creates QuRT threads
# on statically reserved stacks. Design section 3.2 states why.


struct _CMegastepSeam(MegastepSeam):
    """The device seam: the trampoline's own arity and launch, by symbol."""

    def __init__(out self):
        pass

    def buffer_count(self, selector: UInt32) -> Int:
        return Int(external_call["npu_megastep_buffer_count", Int32](selector))

    def scalar_count(self, selector: UInt32) -> Int:
        return Int(external_call["npu_megastep_scalar_count", Int32](selector))

    def launch(
        mut self,
        selector: UInt32,
        thread_mask: UInt32,
        args: WireBytes,
        args_bytes: Int,
        buffers: SlotBases,
        lengths: SlotLengths,
    ) -> Int32:
        return external_call["npu_run_launch", Int32](
            selector, thread_mask, args, Int32(args_bytes), buffers, lengths
        )


@export
def skye_megastep_walk_mojo(
    table_bases: TableBases,
    table_bytes: TableBytes,
    block: WireBytes,
    block_len: Int32,
    count: UInt32,
    slots: SlotBases,
    slot_lengths: SlotLengths,
    skip_compute: Int32,
) abi("C") -> Int32:
    """Walks a RUN-mode megastep block, presenting each launch in order.

    Selected by `SKYE_NPU_MOJO_WALKER`; the C walk is the default and both are
    compiled into every build, which is the project's A/B discipline. The header
    validation, the mode split, the register mode and the grant bracket stay on
    the C side with the entry point and with the session table itself -- Mojo
    has no module-scope mutable global to keep a table in, and the table is
    session state.
    """
    var seam = _CMegastepSeam()
    return megastep_walk(
        seam,
        table_bases,
        table_bytes,
        block,
        Int(block_len),
        count,
        slots,
        slot_lengths,
        skip_compute != 0,
    )
