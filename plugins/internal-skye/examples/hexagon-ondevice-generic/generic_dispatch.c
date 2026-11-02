/*
 * DSP-side FastRPC implementation for the generic mojo_npu interface.
 *
 * Implements both methods the skel exposes:
 *
 *   method 2  mojo_npu_vadd      legacy typed entry point kept for ABI compat
 *   method 3  mojo_npu_dispatch  generic dispatch; routes to the right Mojo
 *                                kernel by selector (see NPUKernels.def)
 *
 * The generic dispatch wire format is described in mojo_npu.idl. In brief:
 *
 *   primitive_block[0]  selector (identifies the kernel)
 *   primitive_block[1]  launch word (thread mask in [7:0])
 *   primitive_block[2]  scalar_bytes (byte length of the scalar block)
 *   primitive_block[3+2*i]  input byte length for slot i
 *   primitive_block[4+2*i]  output byte length for slot i
 *
 *   scalar_block:
 *     bytes [i*8 .. i*8+7]              scalar argument i, padded to 8 bytes
 *     bytes [(scalarCount+j)*8 .. +7]   uint64 byte offset for buffer slot j
 *
 * This file covers selector 0 (vadd, fp32).  Each additional selector added
 * to NPUKernels.def needs a matching case here and its extern declaration.
 */

#include "mojo_npu.h"
#include <string.h>

/* ---- Mojo kernel entry points (compiled for Hexagon) ---- */

extern void mojo_vadd_f32(const float *lhs, const float *rhs, float *out,
                          int n);

/* ---- Helpers ---- */

static int min_int(int a, int b) { return a < b ? a : b; }

/* Reads a native-endian uint32 from a potentially unaligned address. */
static unsigned int read_u32(const unsigned char *p) {
  unsigned int v;
  memcpy(&v, p, sizeof(v));
  return v;
}

/* Reads a native-endian uint64 from a potentially unaligned address. */
static unsigned long long read_u64(const unsigned char *p) {
  unsigned long long v;
  memcpy(&v, p, sizeof(v));
  return v;
}

/* Width of one slot in the scalar block: host packs each value into 8 bytes. */
#define NPU_SCALAR_SLOT_BYTES 8u

/* Number of buffer slots the generic interface declares. */
#define NPU_MAX_BUFFER_SLOTS 8u

/* Words in the primitive block. */
#define NPU_PRIM_WORDS 19u

/* ---- Session open / close ---- */

int mojo_npu_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  *handle = 1; /* stateless session; any non-zero value works */
  return 0;
}

int mojo_npu_close(remote_handle64 handle) {
  (void)handle;
  return 0;
}

/* ---- Method 2: legacy typed vadd ---- */

AEEResult mojo_npu_vadd(remote_handle64 handle, const float *lhs, int lhs_len,
                        const float *rhs, int rhs_len, float *out,
                        int out_len) {
  (void)handle;
  int n = min_int(min_int(lhs_len, rhs_len), out_len);
  mojo_vadd_f32(lhs, rhs, out, n);
  return 0;
}

/* ---- Method 3: generic dispatch ---- */

AEEResult mojo_npu_dispatch(
    remote_handle64 handle,
    const unsigned int *prim, int prim_len,
    const unsigned char *scalars, int scalars_len,
    const unsigned char *in0,  int in0_len,
    const unsigned char *in1,  int in1_len,
    const unsigned char *in2,  int in2_len,
    const unsigned char *in3,  int in3_len,
    const unsigned char *in4,  int in4_len,
    const unsigned char *in5,  int in5_len,
    const unsigned char *in6,  int in6_len,
    const unsigned char *in7,  int in7_len,
    unsigned char *out0,  int out0_len,
    unsigned char *out1,  int out1_len,
    unsigned char *out2,  int out2_len,
    unsigned char *out3,  int out3_len,
    unsigned char *out4,  int out4_len,
    unsigned char *out5,  int out5_len,
    unsigned char *out6,  int out6_len,
    unsigned char *out7,  int out7_len
) {
  (void)handle;
  (void)in3;  (void)in3_len;
  (void)in4;  (void)in4_len;
  (void)in5;  (void)in5_len;
  (void)in6;  (void)in6_len;
  (void)in7;  (void)in7_len;
  (void)out3; (void)out3_len;
  (void)out4; (void)out4_len;
  (void)out5; (void)out5_len;
  (void)out6; (void)out6_len;
  (void)out7; (void)out7_len;

  if ((unsigned int)prim_len < NPU_PRIM_WORDS)
    return AEE_EBADPARM;

  unsigned int selector   = prim[0];
  /* prim[1] = launch word (thread mask + timing flags), handled by the skel */
  /* prim[2] = scalar_bytes, cross-checked below */

  switch (selector) {

  /* ------------------------------------------------------------------ */
  /* selector 0: vadd  (see NPUKernels.def)                              */
  /*   buffers: lhs (slot 0), rhs (slot 1), result (slot 2)             */
  /*   scalars: n (Int32)                                                */
  /* ------------------------------------------------------------------ */
  case 0: {
    /* scalar block: 1 scalar slot + 3 buffer offset slots = 4 * 8 = 32 bytes */
    const unsigned int expected_scalar_bytes = (1u + 3u) * NPU_SCALAR_SLOT_BYTES;
    if ((unsigned int)scalars_len < expected_scalar_bytes)
      return AEE_EBADPARM;

    /* Read n from scalar slot 0 (stored as int32 in the low 4 bytes). */
    int n;
    memcpy(&n, scalars, sizeof(n));
    if (n <= 0)
      return AEE_EBADPARM;

    /* Read per-buffer byte offsets from scalar slots 1..3. */
    unsigned long long off0 = read_u64(scalars + 1u * NPU_SCALAR_SLOT_BYTES);
    unsigned long long off1 = read_u64(scalars + 2u * NPU_SCALAR_SLOT_BYTES);
    unsigned long long off2 = read_u64(scalars + 3u * NPU_SCALAR_SLOT_BYTES);

    /* Validate that offsets keep us inside the slot lengths. */
    if (off0 + (unsigned long long)(n * (int)sizeof(float)) >
        (unsigned long long)in0_len)
      return AEE_EBADPARM;
    if (off1 + (unsigned long long)(n * (int)sizeof(float)) >
        (unsigned long long)in1_len)
      return AEE_EBADPARM;
    if (off2 + (unsigned long long)(n * (int)sizeof(float)) >
        (unsigned long long)out2_len)
      return AEE_EBADPARM;

    const float *lhs    = (const float *)(in0  + off0);
    const float *rhs    = (const float *)(in1  + off1);
    float       *result = (float *)       (out2 + off2);

    (void)in2; (void)in2_len;   /* slot 2 is write-only for vadd */
    (void)out0; (void)out0_len;
    (void)out1; (void)out1_len;

    mojo_vadd_f32(lhs, rhs, result, n);
    return AEE_SUCCESS;
  }

  default:
    return AEE_ENOSUCH;
  }
}

/* ---- DSP runtime startup stubs ---- */
/*
 * The toolchain's shared-object startup references these, but the QuRT
 * protection domain does not export them.  Nothing here unwinds across the RPC
 * boundary or registers atexit handlers, so no-ops suffice.
 */
void __register_frame_info_bases(const void *begin, void *object, void *tbase,
                                 void *dbase) {
  (void)begin; (void)object; (void)tbase; (void)dbase;
}

void *__deregister_frame_info_bases(const void *begin) {
  (void)begin;
  return 0;
}

void __cxa_finalize(void *dso_handle) { (void)dso_handle; }
