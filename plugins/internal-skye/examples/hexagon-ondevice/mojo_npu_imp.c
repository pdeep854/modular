/* DSP-side implementation: bridges the FastRPC skel to the Mojo kernel. */
#include "mojo_npu.h"

extern void mojo_vadd_f32(const float *lhs, const float *rhs, float *result,
                          int element_count);

static int min_int(int a, int b) { return a < b ? a : b; }

int mojo_npu_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  *handle = 1; /* stateless session; any non-zero value will do */
  return 0;
}

int mojo_npu_close(remote_handle64 handle) {
  (void)handle;
  return 0;
}

AEEResult mojo_npu_vadd(remote_handle64 handle, const float *lhs, int lhs_len,
                        const float *rhs, int rhs_len, float *result,
                        int result_len) {
  (void)handle;
  /* Lengths arrive independently over the wire, so clamp rather than trust. */
  int element_count = min_int(min_int(lhs_len, rhs_len), result_len);
  mojo_vadd_f32(lhs, rhs, result, element_count);
  return 0;
}

/* The toolchain's shared-object startup references these, but the DSP
   protection domain does not export them. Nothing here unwinds across the RPC
   boundary or registers atexit handlers, so no-ops suffice. */
void __register_frame_info_bases(const void *begin, void *object, void *tbase,
                                 void *dbase) {
  (void)begin;
  (void)object;
  (void)tbase;
  (void)dbase;
}

void *__deregister_frame_info_bases(const void *begin) {
  (void)begin;
  return 0;
}

void __cxa_finalize(void *dso_handle) { (void)dso_handle; }
