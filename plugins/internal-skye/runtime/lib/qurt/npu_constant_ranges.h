//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The loader's declaration of which host memory holds init-time model
// constants.
//
// Nothing between the weights registry and `enqueueCopyToDevice` carries the
// distinction the NPU backend needs: at the copy the runtime sees a destination
// allocation, a source address and a length, and a weight's host-to-device copy
// is indistinguishable from an activation's. It is the loader that knows -- it
// read or mapped the bytes -- so the loader says so here, once per blob, before
// the model initialises. The backend then classifies a device allocation as
// init-written when a copy filling the whole allocation sourced it from a
// declared range, and only such allocations are eligible for a buffer attribute
// that assumes the host never writes them again.
//
// Declaring nothing is the safe state and the default: no allocation is
// classified, and MODULAR_NPU_WEIGHTS_NOFLUSH has nothing to act on.
//
// Plain C, so the exec drivers can call it.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_NPU_CONSTANT_RANGES_H
#define SKYE_RUNTIME_QURT_NPU_CONSTANT_RANGES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Declares `[base, base + byteSize)` as host memory holding an init-time model
/// constant. Ranges accumulate; declaring the same one twice is harmless.
///
/// Call before the copies happen -- for the whole-graph drivers, after the
/// registry blobs are loaded and before `M_initModel`. A range declared after
/// its copy classifies nothing, which is a missed optimisation and never a
/// correctness problem.
void MODULAR_NPUDeclareConstantHostRange(const void *base, size_t byteSize);

/// Drops every declared range, for a driver that releases its blobs and reloads
/// them somewhere else. Does not withdraw attributes already applied; a host
/// write to the buffer does that.
void MODULAR_NPUForgetConstantHostRanges(void);

/// Nonzero when `[base, base + byteSize)` lies entirely inside declared
/// constant memory, with `byteSize` nonzero. This is the provenance test the
/// backend applies to a copy's source; exported so a host test holds the exact
/// predicate the backend classifies with, not a restatement of it.
int MODULAR_NPUConstantHostRangeCovers(const void *base, size_t byteSize);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SKYE_RUNTIME_QURT_NPU_CONSTANT_RANGES_H
