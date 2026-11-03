//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// QAIC kernel-side unwind: per-thread `jmp_buf` slots so `abort()` (in
// std/os/os.mojo) can `longjmp` back to the entry-point wrapper (in
// std/_plugin/qaic/host/_target.mojo). Compiled to LLVM bitcode and merged
// into the std package — Mojo's stdlib build forbids module-level mutable
// vars, so the slot lives here.
//
// Per-thread indexing on the current QShim thread id is mandatory:
// cross-thread `longjmp` would resume one thread's `setjmp` on another
// thread's stack and corrupt both. The thread id comes from
// `qshimQuery(0, ...)` because the QPC Shim runtime resolves `qshimQuery`
// at JIT load but not `qshimGetThreadId`.

// Hexagon NSP supports ~6 hardware threads per core. Round up generously.
#define MAX_QAIC_THREADS 8

#define QSHIM_QUERY_THREAD_ID 0

extern unsigned int qshimQuery(unsigned int type, long long *out_val);

static unsigned char qaic_unwind_thread_id(void) {
  long long val = 0;
  qshimQuery(QSHIM_QUERY_THREAD_ID, &val);
  return (unsigned char)val;
}

static int *qaic_unwind_jmpbufs[MAX_QAIC_THREADS];

void __qaic_unwind_set(int *buf) {
  qaic_unwind_jmpbufs[qaic_unwind_thread_id()] = buf;
}

void __qaic_unwind_clear(void) {
  qaic_unwind_jmpbufs[qaic_unwind_thread_id()] = (int *)0;
}

int *__qaic_unwind_get(void) {
  return qaic_unwind_jmpbufs[qaic_unwind_thread_id()];
}
