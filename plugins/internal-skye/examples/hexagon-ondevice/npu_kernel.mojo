from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor
from std.sys import size_of

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
