// Device-side implementation of the `qshim*` services that Mojo kernels call,
// backed by QuRT and the HAP compute-resource manager instead of the JIT
// runtime used on the discrete accelerator.
//
// Kernels reach these through `std._plugin.qaic.host.qshim`, which lowers each
// wrapper to an `external_call` on the C names below. On the discrete part the
// symbols come from the vendor JIT runtime already resident in the protection
// domain; over FastRPC nothing provides them, so they link in here.
//
// Scope: VTCM acquisition, device queries, linear uDMA, and the thread barrier
// — the substrate a VTCM-staged kernel needs. Multicast, mmap, profiling and
// ECC queries are not implemented; a kernel calling one fails to link, which is
// the intended outcome rather than a silent stub.

#include <stdint.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "mojo_npu_launch.h"
#include "npu_launch_pool.h"
#include "npu_phase_probe.h"
#include "npu_sync_barrier.h"
#include "npu_udma_report.h"
#include "npu_udma_ring.h"
#include "qurt.h"
#include "udma.h"

// Mirrors `QShimThreadSyncStruct` in the stdlib overlay. Lives in the kernel's
// global data so every participating thread sees one copy.
typedef struct {
  int32_t thread_mask;
  int32_t data;
} QShimThreadSyncStruct;

typedef uint32_t QShimUDmaHandle;

#define QSHIM_INVALID_UDMA_HANDLE 0xFFFFFFFFu

// Attribute codes accepted by `qshimQuery`, matching the QSHIM_QUERY_*
// constants in the stdlib overlay.
enum {
  QSHIM_QUERY_THREAD_ID = 0,
  QSHIM_QUERY_NUM_THREADS = 1,
  QSHIM_QUERY_CORE_ID = 2,
  QSHIM_QUERY_NUM_CORES = 3,
  QSHIM_QUERY_VTCM_SIZE = 4,
  QSHIM_QUERY_NUM_HVX_UNITS = 5,
  QSHIM_QUERY_HMX_THREAD_ID = 6,
};

// ===--------------------------------------------------------------------=== //
// VTCM
// ===--------------------------------------------------------------------=== //

// Acquired once and held for the life of the protection domain. Kernels expect
// a stable base across invocations — the compiler assigns them fixed byte
// offsets from it at build time — so acquiring per call would be both slow and
// wrong if the manager handed back a different block.
static compute_res_attr_t g_vtcm_attr;
static void *g_vtcm_base = NULL;
static unsigned int g_vtcm_size = 0;
static unsigned int g_vtcm_ctx = 0;
static int g_hmx_cached_rc = -99;
// 1 only once branch A below both requests HMX AND has `acquire_cached`
// return 0; stays 0 for the plain-grant fallback and for outright failure.
// `g_hmx_cached_rc` alone conflates those latter two (-99 sentinel vs a real
// HAP error code); this is the single bit an HMX-aware caller actually needs.
static int g_hmx_ctx_armed = 0;
// P5 M2.4 order 065: the DEDICATED HMX context's own acquire rc (below,
// `hmx_ctx_acquire`), reusing the witness word/getter pair
// (`qshimHmxReleaseRc`, word 11) that carried the release half of order 064's
// stack-then-cycle attempt on the shared context. Per direct instruction to
// reuse the words rather than add new ones; the getter's NAME predates this
// repointing.
static int g_hmx_release_rc = -99;
// The dedicated context's own acquire_cached rc, reusing
// `qshimHmxBracketRc`/word 12 the same way.
static int g_hmx_bracket_rc = -99;

// One acquisition at a time, over BOTH lazy contexts in this file.
//
// The two acquires below are check-then-act over process-wide state and EVERY
// participant of a launch calls them, so on a process's first dispatch several
// threads read a null base together, all go on to `HAP_compute_res_acquire`,
// and the losers are handed NULL or a second context nothing will release.
// `_matmul_stripe_hmx`'s own comment records what that costs: a refusal that is
// thread-DIVERGENT, where some participants return while their peers wait at a
// rendezvous those participants will now never reach.
//
// NOT a `qurt_mutex_t`, though that is what this is. `qurt_mutex_*` is not on
// `npu_skel_symbol_gate.sh`'s resident-proven allowlist, and a `qurt_*` import
// this protection domain does not resolve does not fail the call -- it makes
// the whole skel unloadable, which is how `qurt_hvx_set` was lost (order 067).
// `qurt_timer_sleep` is on that list and this file already waits on it, so the
// gate is a test-and-set with a sleeping back-off. It is uncontended after the
// first dispatch of a process, the hold is bounded by the acquires' own
// timeouts, and nothing here would be improved by a fairer lock.
static int32_t g_vtcm_gate;

#define VTCM_GATE_NAP_MICROS 100u

static void vtcm_gate_lock(void) {
  for (;;) {
    int32_t expected = 0;
    if (__atomic_compare_exchange_n(&g_vtcm_gate, &expected, 1, /*weak=*/0,
                                    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
      return;
    // Sleeping rather than spinning: an NSP hardware thread spinning here is
    // one the thread holding the gate may be waiting for.
    qurt_timer_sleep(VTCM_GATE_NAP_MICROS);
  }
}

static void vtcm_gate_unlock(void) {
  __atomic_store_n(&g_vtcm_gate, 0, __ATOMIC_RELEASE);
}

// Requests the whole of the part's VTCM. Capacity is a property of the part and
// is negotiated at runtime, so it is queried rather than assumed: parts sharing
// a Hexagon revision ship different amounts.
//
// Runs under `g_vtcm_gate`, so the first-caller check below is a decision and
// not a guess.
static int vtcm_acquire_locked(void) {
  if (g_vtcm_base != NULL)
    return 0;

  unsigned int size = 0;
  if (HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL) != 0 || size == 0)
    return -1;

  if (HAP_compute_res_attr_init(&g_vtcm_attr) != 0)
    return -1;

  // Single-page mapping over the full request: page size and minimum both set
  // to the queried size, so a partial grant is refused rather than silently
  // handed back smaller than the offsets the kernel was compiled against.
  if (HAP_compute_res_attr_set_vtcm_param_v2(&g_vtcm_attr, size, size, size) !=
      0)
    return -1;

  // Arm HMX on this context: the resource manager creates the HMX enable entry
  // only when the acquire requests HMX, and under cache mode leaves it
  // armed-but-disabled until `acquire_cached` enables it (the sequence every
  // SDK HMX example uses; without HMX requested the later lock reports the
  // context as not set up for HMX, and without the cached acquire it stays
  // disabled). Both only add to the VTCM grant, so they are additive for the
  // VTCM-only kernels. Fall back to a plain grant if the HMX/cached acquisition
  // is refused, so such a kernel is never denied its scratch.
  //
  // That fallback hands back rc 0 -- the same as the armed path -- so neither
  // this function's return nor the non-null pointer `qshimGetBaseVtcmAddr()`
  // derives from it can tell "HMX-armed" from "plain VTCM" apart.
  // `qshimVtcmCtxHmxArmed()` below is the one place that can; an HMX-aware
  // caller must check it before treating a later lock refusal as informative
  // (P5 M2.4 hardening -- this fallback silently masking the distinction was
  // never observed to fire, per `qshimHmxPr42Probe` device evidence, but the
  // mask itself was still a real gap).
  // `set_serialize(0)`: ggml-hexagon sets this on the identical acquire
  // (`ggml/src/ggml-hexagon/htp/main.c:265`); this file already uses it on
  // two of its OWN probes (`qshimHmxShippingProbe`, `qshimHmxLlamacppProbe`),
  // just never on the shared context the real matmul path acquires.
  //
  // NOT copied here: their `HAP_compute_res_update_priority(rctx,
  // thread_prio + 10)` (`main.c:241`), deliberately LOWERING priority so
  // release callbacks still arrive from other sessions. `thread_prio` there
  // is a value threaded in from outside `main.c` (an IDL/init-time
  // parameter, not a call this file has any local precedent for reading) --
  // guessing the read would repeat the exact mistake `qurt_hvx_lock` just
  // was reverted for. Left for whoever can confirm the priority-query API
  // against the SDK.
  //
  // Also NOT copied: their acquire runs BEFORE the power votes
  // (`main.c:415` vs `:428+`); this shim's acquire arms after
  // `qshimHmxLlamacppProbe`'s power-vote experiment, on a separate probe
  // context. Noted, not restructured tonight.
  HAP_compute_res_attr_set_serialize(&g_vtcm_attr, 0);
  HAP_compute_res_attr_set_hmx_param(&g_vtcm_attr, 1);
  HAP_compute_res_attr_set_cache_mode(&g_vtcm_attr, 1);
  // P5 M2.4 order 065->066: 0 -> 1000000. 065's side-by-side read every
  // granting and reference path (qshimHmxPr42Probe, qshimHmxShippingProbe,
  // qshimHmxLlamacppProbe, ggml-hexagon's own vtcm_alloc) at 1000000 for
  // this exact call, and this was the one still at 0 -- the shared context
  // the real matmul dispatch acquires through was, until now, the only
  // acquire in this file that never waited. Bundled with the lock3 insertion
  // below (qshimHmxLock) as a deliberate conjunction, matching the acquire
  // cycle round's own reasoning: if the matmul's lock grants with both
  // changes present, which one was necessary is a KB note, not another
  // window.
  g_vtcm_ctx = HAP_compute_res_acquire(&g_vtcm_attr, /*timeout_us=*/1000000);
  if (g_vtcm_ctx != 0) {
    // 1000000us to match the timeout ggml-hexagon (llama.cpp's Hexagon
    // backend) uses for the identical call
    // (`ggml/src/ggml-hexagon/htp/main.c:227`, `vtcm_acquire`) -- 100x this
    // project's prior 10000us. A hardening margin, not a proven fix: this
    // project's own `qshimHmxPr42Probe` device run granted at the old value
    // (`g_hmx_cached_rc` read back 0), so a short timeout is not shown to be
    // why any specific run failed. But `HAP_compute_res_acquire_cached` can
    // legitimately block on transient CRM contention, and 10ms is a tight
    // budget to fail fast on for a call that otherwise returns in
    // microseconds.
    g_hmx_cached_rc = HAP_compute_res_acquire_cached(g_vtcm_ctx,
                                                     /*timeout_us=*/1000000);
    if (g_hmx_cached_rc != 0) {
      HAP_compute_res_release(g_vtcm_ctx);
      g_vtcm_ctx = 0;
    } else {
      g_hmx_ctx_armed = 1;
    }
  }
  if (g_vtcm_ctx == 0) {
    if (HAP_compute_res_attr_init(&g_vtcm_attr) != 0 ||
        HAP_compute_res_attr_set_vtcm_param_v2(&g_vtcm_attr, size, size,
                                               size) != 0)
      return -1;
    g_vtcm_ctx = HAP_compute_res_acquire(&g_vtcm_attr, /*timeout_us=*/0);
  }
  if (g_vtcm_ctx == 0)
    return -1;

  unsigned int granted = 0;
  if (HAP_compute_res_attr_get_vtcm_ptr_v2(&g_vtcm_attr, &g_vtcm_base,
                                           &granted) != 0 ||
      g_vtcm_base == NULL || granted < size) {
    HAP_compute_res_release(g_vtcm_ctx);
    g_vtcm_ctx = 0;
    g_vtcm_base = NULL;
    return -1;
  }

  g_vtcm_size = granted;
  return 0;
}

static int vtcm_acquire(void) {
  vtcm_gate_lock();
  int rc = vtcm_acquire_locked();
  vtcm_gate_unlock();
  return rc;
}

void *qshimGetBaseVtcmAddr(void) {
  if (vtcm_acquire() != 0)
    return NULL;
  return g_vtcm_base;
}

// P5 M2.4 order 064->065: a DEDICATED context for the HMX path, separate from
// the shared `g_vtcm_ctx` above -- 060 through 064 held M/N/K, the shape
// check, freshness, and thread identity fixed in turn and the plain lock
// still returned QURT_EINVALID on `g_vtcm_ctx`. The remaining variable this
// project has not yet isolated is the context OBJECT: `g_vtcm_ctx` is a
// SHARED handle every non-HMX VTCM consumer in this file also acquires
// through (`vtcm_acquire`, above), built once by whichever caller reaches it
// first, while the shape being copied here (`HAP_compute_res_attr_init` +
// `set_hmx_param(1)` + `set_cache_mode(1)` + `set_serialize(0)` +
// `HAP_compute_res_acquire` + `HAP_compute_res_acquire_cached`) is otherwise
// identical to `vtcm_acquire`'s own branch A.
//
// CORRECTION to this change's own originating brief: the brief attributed
// the working 058 grant to "058's granting probes build their own context"
// (`qshimHmxPr42Probe`, at the time near lines 392/540). That is not what
// `qshimHmxPr42Probe` does -- it calls `vtcm_acquire()` and explicitly
// reuses the shared context (`g_pr42_ctx = g_vtcm_ctx;`, immediately after).
// The two functions that DO build their own separate context at those line
// numbers, `qshimHmxShippingProbe` and `qshimHmxLlamacppProbe`, are both
// documented as REFUSING (rc 1), not granting -- because each omits one of
// `set_hmx_param`/`acquire_cached`, not because of context sharing. What
// their acquire step DOES establish, correctly, is that a second, VTCM-
// requesting context CAN be built and granted while `g_vtcm_ctx` is already
// live (their own `ctx == 0` acquire-failure branch was never taken) -- that
// coexistence evidence is real, the attribution to pr42 was not. This
// experiment is still worth running on that basis: `g_vtcm_ctx` is shared by
// every non-HMX caller in this process (8+ other call sites), never
// explicitly given only to HMX; a context built for HMX ALONE and touched by
// nothing else is the one combination nobody has isolated yet.
static compute_res_attr_t g_hmx_ctx_attr;
static unsigned int g_hmx_ctx = 0;
static void *g_hmx_ctx_base = NULL;
static unsigned int g_hmx_ctx_size = 0;

// Under the same gate as `vtcm_acquire_locked`, and one gate rather than two:
// both build a context over the part's whole VTCM through the same manager, so
// letting them run at once is the race with an extra step.
static int hmx_ctx_acquire_locked(void) {
  if (g_hmx_ctx != 0)
    return 0;

  unsigned int size = 0;
  if (HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL) != 0 || size == 0)
    return -1;

  if (HAP_compute_res_attr_init(&g_hmx_ctx_attr) != 0)
    return -1;
  if (HAP_compute_res_attr_set_vtcm_param_v2(&g_hmx_ctx_attr, size, size,
                                             size) != 0)
    return -1;
  HAP_compute_res_attr_set_serialize(&g_hmx_ctx_attr, 0);
  HAP_compute_res_attr_set_cache_mode(&g_hmx_ctx_attr, 1);
  HAP_compute_res_attr_set_hmx_param(&g_hmx_ctx_attr, 1);

  // Reuses `g_hmx_release_rc`/`qshimHmxReleaseRc` for THIS acquire's own rc,
  // and `g_hmx_bracket_rc`/`qshimHmxBracketRc` for the acquire_cached below --
  // same witness words 11/12 the prior (stack, then cycle) attempts used,
  // repointed at this dedicated context's own two-step acquisition rather
  // than at a release/reacquire pair on the shared one. Per direct
  // instruction to reuse those words rather than add new ones.
  unsigned int ctx = HAP_compute_res_acquire(&g_hmx_ctx_attr,
                                             /*timeout_us=*/1000000);
  g_hmx_release_rc = (ctx != 0) ? 0 : -1;
  if (ctx == 0)
    return -1;

  g_hmx_bracket_rc =
      HAP_compute_res_acquire_cached(ctx, /*timeout_us=*/1000000);
  if (g_hmx_bracket_rc != 0) {
    HAP_compute_res_release(ctx);
    return -1;
  }

  // Design choice, stated per instruction: the pr42 attribution this change
  // was meant to copy turned out not to build a separate VTCM window either
  // (see correction above), so there is no existing precedent in this file
  // for whether HMX scratch may address VTCM through a DIFFERENT context
  // than the one that mapped it. Chosen conservatively: this dedicated
  // context supplies its OWN VTCM window (queried and mapped here, same as
  // `vtcm_acquire` does for the shared one), and `_matmul_stripe_hmx`'s
  // crouton scratch now addresses THIS window, not `g_vtcm_base`. The
  // activation/weight source data still lives wherever the caller's `A`/`B`
  // pointers point (ordinary DRAM, outside either VTCM context); packing
  // that data into the crouton layout is what already crosses from DRAM into
  // VTCM, so routing it into this window instead of the shared one changes
  // the destination, not the packing step itself.
  unsigned int granted = 0;
  if (HAP_compute_res_attr_get_vtcm_ptr_v2(&g_hmx_ctx_attr, &g_hmx_ctx_base,
                                           &granted) != 0 ||
      g_hmx_ctx_base == NULL || granted < size) {
    HAP_compute_res_release(ctx);
    g_hmx_ctx_base = NULL;
    return -1;
  }

  g_hmx_ctx_size = granted;
  g_hmx_ctx = ctx;
  // P5 M2.4 order 066: does NOT set g_hmx_ctx_armed. #281 repointed that flag
  // at this context; order 065's device evidence refuted the dedicated
  // context as the differentiator (it refuses too), and qshimHmxLock now
  // locks g_vtcm_ctx again (below) -- so g_hmx_ctx_armed reverts to meaning
  // what it meant before #281: vtcm_acquire's own shared-context flag, set
  // there, not here. This function (and the rest of this dedicated-context
  // plumbing) stays in the tree unused by the active lock path, for the
  // record.
  return 0;
}

static int hmx_ctx_acquire(void) {
  vtcm_gate_lock();
  int rc = hmx_ctx_acquire_locked();
  vtcm_gate_unlock();
  return rc;
}

void *qshimGetHmxVtcmAddr(void) {
  if (hmx_ctx_acquire() != 0)
    return NULL;
  return g_hmx_ctx_base;
}

unsigned int qshimGetHmxVtcmSize(void) {
  if (hmx_ctx_acquire() != 0)
    return 0;
  return g_hmx_ctx_size;
}

// Whether the shared context `qshimGetBaseVtcmAddr`/`qshimHmxLock` operate on
// (reverted to `g_vtcm_ctx` as of order 066; #281's dedicated context was
// refuted as the differentiator by 065's device evidence) actually carries
// the HMX attribute, as opposed to `vtcm_acquire`'s plain-grant fallback. A
// caller that only checked a non-null VTCM pointer, or `qshimHmxLock`
// returning nonzero, cannot make this distinction on its own. Reads
// `g_hmx_ctx_armed`, which `vtcm_acquire` sets exactly once, on its own first
// call in the process.
int qshimVtcmCtxHmxArmed(void) { return g_hmx_ctx_armed; }

// Releases the VTCM grant. Called from the skel's close handler; kernels never
// call it.
void mojo_npu_qshim_teardown(void) {
  if (g_vtcm_ctx != 0) {
    HAP_compute_res_release(g_vtcm_ctx);
    g_vtcm_ctx = 0;
    g_vtcm_base = NULL;
    g_vtcm_size = 0;
  }
}

// ===--------------------------------------------------------------------=== //
// HMX
// ===--------------------------------------------------------------------=== //

// Release this thread's HVX context before probing for HMX, and report the
// result. A SECOND VARIANT of the probe, not part of the primary one: the
// "a thread with HVX enabled cannot have HMX enabled" note in `qurt_hmx.h:184`
// is documented under `qurt_hmx_assign` -- an `@srm`-only, thread-targeted call
// -- and NOT under the `@unsigned`-permitted `qurt_hmx_lock2`/`try_lock` that
// this code uses. Run it only if the primary probe refuses.
#ifndef SKYE_HMX_CLEAR_HVX
#define SKYE_HMX_CLEAR_HVX 0
#endif

// Escalate from the non-blocking probe to actually taking the unit and running
// HMX instructions. OFF by default and gated at run time on the probe having
// already succeeded, so no HMX instruction can execute on an ungranted unit and
// the blocking lock is never reached on a unit known to be unavailable.
#ifndef SKYE_HMX_EXECUTE
#define SKYE_HMX_EXECUTE 0
#endif

// The "hmx_busy" flag lock4 hands back: its light-weight critical section is a
// write of 1 (enter) / 0 (exit) to this location, which autonomous preemption
// watches before reassigning HMX. Null until a successful lock4, so enter/exit
// no-op until then.
static unsigned int *g_hmx_busy = NULL;

// P5 M2.4 order 066: `qshimHmxLock`'s lock3 attempt (between lock4 and the
// plain lock, matching `qshimHmxPr42Probe` line for line) and the mutex a
// success would hand back -- tracked the same way `g_hmx_busy` tracks a
// lock4 grant, so `qshimHmxUnlock` releases through the entry point that
// actually took the unit. Expected to always refuse on this manager (the KB
// already has lock3 refusing everywhere it has been tried), so this branch
// is exercised for completeness, not because success is anticipated.
static compute_res_hmx_mutex_t g_hmx_lock3_mutex;
static int g_hmx_held_by_lock3 = 0;
static int g_hmx_lock3_rc = -99;

// Return codes from the probe, each -99 until its call is actually reached, so
// "refused" is never confused with "never ran". `qurt_hmx_try_lock` and
// `qurt_hmx_lock2` document exactly three outcomes each (`qurt_hmx.h:66-77`,
// `:159-175`): QURT_EOK, QURT_EFAILED (12, wrong locking condition), and
// QURT_EVAL (1, no HMX hardware supported). Codes 13 and 95 exist in
// `qurt_error.h` but are NOT in either function's documented return set, so
// they would be off-contract rather than the expected permission answer.
static int g_hmx_try_rc = -99;
static int g_hmx_qurt_rc = -99;
static int g_hmx_held_by_qurt = 0;

// `qurt_hvx_unlock`'s status, and the HVX mode. The unlock is the load-bearing
// one -- on v65 and newer it maps to `qurt_hvx_set(QURT_HVX_RELEASE_CONTEXT)`
// (`qurt_hvx.h:173-196`) -- so its result is recorded rather than discarded.
// The mode is informational only: it reports the mode QuRT has configured, not
// whether this thread owns a context (`qurt_hvx.h:230-253`).
static int g_hvx_unlock_rc = -99;
static int g_hvx_mode = -99;

// The primary HMX probe: non-blocking, isolated, and executing no HMX.
//
// `qurt_hmx_try_lock` returns failure rather than suspending when the unit is
// unavailable (`qurt_hmx.h:159-175`), which `qurt_hmx_lock2` does NOT --
// `qurt_hmx.h:47-50` suspends the caller until the unit frees up. On a single
// poison-prone device slot an indefinite suspend is its own failure mode, and
// one that returning-before-the-HMX-instruction does not protect against, so
// the blocking form is never the first thing tried.
//
// Deliberately does NOT acquire VTCM and does NOT fall back to the resource
// manager. The QuRT path declares no VTCM dependency (`qurt_hmx.h:66-77`), and
// mixing in the manager's acquire -- which itself requests HMX via
// `set_hmx_param` -- would leave the result attributable to neither path. On
// success the unit is released immediately: this answers "would the unit be
// granted", nothing more.
int qshimHmxTryLock(void) {
#if SKYE_HMX_CLEAR_HVX
  g_hvx_mode = qurt_hvx_get_mode();
  g_hvx_unlock_rc = qurt_hvx_unlock();
#endif

  g_hmx_try_rc = qurt_hmx_try_lock();
  if (g_hmx_try_rc == 0)
    qurt_hmx_unlock2(QURT_HMX_NON_SHARED_UNLOCK);
  return g_hmx_try_rc;
}

// Steps of the manager-path probe, so a refusal names the call that produced
// it rather than collapsing into one code. Distinct from any documented
// HAP_compute_res or QuRT value, which are all >= 0.
enum {
  QSHIM_CRM_ERR_QUERY_VTCM = -101,
  QSHIM_CRM_ERR_ATTR_INIT = -102,
  QSHIM_CRM_ERR_VTCM_PARAM = -103,
  QSHIM_CRM_ERR_ACQUIRE = -104,
};

static int g_hmx_crm_acquire_rc = -99;
static int g_hmx_crm_cached_rc = -99;
static int g_hmx_crm_lock_rc = -99;
static int g_hmx_crm_lock3_rc = -99;

// The manager-path HMX probe, in the framework this part actually advertises.
//
// `HAP_compute_res_query_capability` reports 0x4000001a here -- autonomous,
// autonomous-deferred and HMX_LOCK4, with no cooperative bit. Under those
// frameworks HMX is NOT requested on the acquire attribute; it "is directly
// locked via HAP_compute_res_hmx_lock3() (or HAP_compute_res_hmx_lock4() if
// supported) using the context returned by a successful VTCM allocation"
// (`HAP_compute_res.md:117-121`). `set_hmx_param` is the cooperative idiom, so
// the sequence below omits it and follows the SDK's own autonomous-deferred
// example (`HAP_compute_res.md:285-355`): init, query, vtcm_param_v2,
// cache_mode, acquire, acquire_cached, then the lock.
//
// Runs on a context of its own and releases it before returning, so the shared
// grant `vtcm_acquire` hands the compute kernels is neither consumed nor
// disturbed by a diagnostic. Executes no HMX instruction: it answers only
// whether the unit would be granted.
int qshimHmxCrmProbe(void) {
  compute_res_attr_t attr;
  unsigned int vtcm_size = 0;

  if (HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL) != 0 ||
      vtcm_size == 0)
    return (g_hmx_crm_acquire_rc = QSHIM_CRM_ERR_QUERY_VTCM);
  if (HAP_compute_res_attr_init(&attr) != 0)
    return (g_hmx_crm_acquire_rc = QSHIM_CRM_ERR_ATTR_INIT);
  if (HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size,
                                             vtcm_size) != 0)
    return (g_hmx_crm_acquire_rc = QSHIM_CRM_ERR_VTCM_PARAM);
  HAP_compute_res_attr_set_cache_mode(&attr, 1);

  unsigned int ctx = HAP_compute_res_acquire(&attr, /*timeout_us=*/10000);
  if (ctx == 0)
    return (g_hmx_crm_acquire_rc = QSHIM_CRM_ERR_ACQUIRE);
  g_hmx_crm_acquire_rc = 0;

  g_hmx_crm_cached_rc = HAP_compute_res_acquire_cached(ctx,
                                                       /*timeout_us=*/10000);

  unsigned int *busy = NULL;
  g_hmx_crm_lock_rc = HAP_compute_res_hmx_lock4(ctx, &busy,
                                                /*timeout_us=*/10000);
  if (g_hmx_crm_lock_rc == 0) {
    HAP_compute_res_hmx_unlock4(ctx, busy);
  } else {
    // lock3 is the base autonomous entry point and lock4 only its optimized
    // critical-section form, so a lock4 that is absent or refuses does not
    // settle the manager path. Device slots are scarce enough that asking both
    // in one run is worth the few lines.
    compute_res_hmx_mutex_t mutex;
    g_hmx_crm_lock3_rc = HAP_compute_res_hmx_lock3(
        ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mutex, /*timeout_us=*/10000);
    if (g_hmx_crm_lock3_rc == 0)
      HAP_compute_res_hmx_unlock3(ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mutex);
  }

  HAP_compute_res_release(ctx);
  return g_hmx_crm_lock_rc;
}

int qshimHmxCrmAcquireRc(void) { return g_hmx_crm_acquire_rc; }
int qshimHmxCrmCachedRc(void) { return g_hmx_crm_cached_rc; }
int qshimHmxCrmLockRc(void) { return g_hmx_crm_lock_rc; }
int qshimHmxCrmLock3Rc(void) { return g_hmx_crm_lock3_rc; }

static int g_hmx_shipping_acquire_rc = -99;
static int g_hmx_shipping_lock_rc = -99;

// Registered only so the manager sees a client that can yield; the probe holds
// the context for a few microseconds and never needs to act on a request.
static int hmx_probe_release_callback(unsigned int context_id,
                                      void *client_context) {
  (void)context_id;
  (void)client_context;
  return 0;
}

// A third shape, the one a shipping open-source Hexagon backend uses.
//
// It differs from both probes above in the combination rather than in any one
// call: HMX is requested as an ATTRIBUTE of the same acquisition that allocates
// VTCM (`set_hmx_param` on the attr, the cooperative idiom), and the unit is
// then taken with the PLAIN `HAP_compute_res_hmx_lock(rctx)` -- not `lock2`,
// `lock3` or `lock4`. It also sets `serialize(0)` and registers a release
// callback, which is what tells the manager the client can yield cooperatively.
//
// Worth a device slot on its own: the two probes above cover
// autonomous-acquire + lock3/lock4 and the raw QuRT syscall, and this covers
// cooperative-acquire + plain lock, which no run here has ever tried. Executes
// no HMX instruction and releases everything it takes.
int qshimHmxShippingProbe(void) {
  compute_res_attr_t attr;
  unsigned int vtcm_size = 0;

  if (HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL) != 0 ||
      vtcm_size == 0)
    return (g_hmx_shipping_acquire_rc = QSHIM_CRM_ERR_QUERY_VTCM);
  if (HAP_compute_res_attr_init(&attr) != 0)
    return (g_hmx_shipping_acquire_rc = QSHIM_CRM_ERR_ATTR_INIT);

  HAP_compute_res_attr_set_serialize(&attr, 0);
  HAP_compute_res_attr_set_cache_mode(&attr, 1);
  if (HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size,
                                             vtcm_size) != 0)
    return (g_hmx_shipping_acquire_rc = QSHIM_CRM_ERR_VTCM_PARAM);
  HAP_compute_res_attr_set_release_callback(&attr, hmx_probe_release_callback,
                                            NULL);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);

  unsigned int ctx = HAP_compute_res_acquire(&attr, /*timeout_us=*/1000000);
  if (ctx == 0)
    return (g_hmx_shipping_acquire_rc = QSHIM_CRM_ERR_ACQUIRE);
  g_hmx_shipping_acquire_rc = 0;

  g_hmx_shipping_lock_rc = HAP_compute_res_hmx_lock(ctx);
  if (g_hmx_shipping_lock_rc == 0)
    HAP_compute_res_hmx_unlock(ctx);

  HAP_compute_res_release(ctx);
  return g_hmx_shipping_lock_rc;
}

int qshimHmxShippingAcquireRc(void) { return g_hmx_shipping_acquire_rc; }
int qshimHmxShippingLockRc(void) { return g_hmx_shipping_lock_rc; }

// `HAP_power_set` is declared by `HAP_power.h` but defined by no SDK archive:
// it resolves from the protection domain or not at all. A STRONG undefined
// reference to a name nothing exports makes the whole skel unloadable
// (AEE_EUNABLETOLOAD), which is how this project has lost a run before, so it
// is imported weakly -- an absent one reads back as a null pointer this probe
// reports, rather than a library that will not load.
int __attribute__((weak)) HAP_power_set(void *context,
                                        HAP_power_request_t *request);

static int g_hmx_power_resident = -99;
static int g_hmx_power_apptype_rc = -99;
static int g_hmx_power_dcvs_rc = -99;
static int g_hmx_power_hvx_rc = -99;
static int g_hmx_power_hmx_rc = -99;
static int g_hmx_power_hmx_v2_rc = -99;
static int g_hmx_full_acquire_rc = -99;
static int g_hmx_full_hvx_unlock_rc = -99;
static int g_hmx_full_lock_rc = -99;

// The full acquisition sequence a shipping Qualcomm-authored Hexagon backend
// performs, reproduced from its published structure.
//
// Every earlier probe here asked for the unit; none of them ever POWERED IT ON.
// That backend votes the HMX block up through `HAP_power_set` -- app type,
// DCVS, HVX rail, then `HAP_power_set_HMX_v2` at the maximum corner -- and only
// afterwards takes the unit with the plain `HAP_compute_res_hmx_lock` on the
// SAME context that allocated VTCM, from a thread holding no HVX context. A
// power-collapsed block is a plausible source of an argument-shaped refusal,
// and the power vote is the one step no run on this part has tried.
//
// It is also a probe in its own right: `HAP_power.h:525-531` says a chipset
// with no separate HMX clock rejects a corner/frequency request with
// AEE_EBADPARM, so the power manager answers the presence question through a
// third framework, independent of both the compute-resource manager and the raw
// QuRT syscall.
//
// Executes no HMX instruction, and releases everything it takes.
int qshimHmxLlamacppProbe(void) {
  g_hmx_power_resident = (HAP_power_set != 0);
  if (!g_hmx_power_resident)
    return (g_hmx_full_lock_rc = -105);

  // A stable per-client token; the power manager only uses it for identity.
  void *client = (void *)&g_hmx_power_resident;

  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    g_hmx_power_apptype_rc = HAP_power_set(client, &request);
  }

  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = FALSE;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = TRUE;
    g_hmx_power_dcvs_rc = HAP_power_set(client, &request);
  }

  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    g_hmx_power_hvx_rc = HAP_power_set(client, &request);
  }

  // Both HMX power shapes, because they fail differently: the v1 form only
  // turns the block on, while v2 additionally votes a clock corner and is
  // documented to reject a corner request on a part with no separate HMX clock.
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX;
    request.hmx.power_up = TRUE;
    g_hmx_power_hmx_rc = HAP_power_set(client, &request);
  }

  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX_v2;
    request.hmx_v2.set_power = TRUE;
    request.hmx_v2.power_up = TRUE;
    request.hmx_v2.set_clock = TRUE;
    request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    g_hmx_power_hmx_v2_rc = HAP_power_set(client, &request);
  }

  compute_res_attr_t attr;
  unsigned int vtcm_size = 0;
  if (HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL) != 0 ||
      vtcm_size == 0)
    return (g_hmx_full_acquire_rc = QSHIM_CRM_ERR_QUERY_VTCM);
  if (HAP_compute_res_attr_init(&attr) != 0)
    return (g_hmx_full_acquire_rc = QSHIM_CRM_ERR_ATTR_INIT);

  HAP_compute_res_attr_set_serialize(&attr, 0);
  HAP_compute_res_attr_set_cache_mode(&attr, 1);
  if (HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size,
                                             vtcm_size) != 0)
    return (g_hmx_full_acquire_rc = QSHIM_CRM_ERR_VTCM_PARAM);
  HAP_compute_res_attr_set_release_callback(&attr, hmx_probe_release_callback,
                                            NULL);
  HAP_compute_res_attr_set_hmx_param(&attr, 1);

  unsigned int ctx = HAP_compute_res_acquire(&attr, /*timeout_us=*/1000000);
  if (ctx == 0)
    return (g_hmx_full_acquire_rc = QSHIM_CRM_ERR_ACQUIRE);
  g_hmx_full_acquire_rc = 0;

#if SKYE_HMX_CLEAR_HVX
  // That backend's HMX thread drops its HVX context before taking the unit, and
  // a thread holding HVX cannot hold HMX, so the faithful reproduction releases
  // it here rather than relying on this thread never having taken one.
  g_hmx_full_hvx_unlock_rc = qurt_hvx_unlock();
#endif

  g_hmx_full_lock_rc = HAP_compute_res_hmx_lock(ctx);
  if (g_hmx_full_lock_rc == 0)
    HAP_compute_res_hmx_unlock(ctx);

  HAP_compute_res_release(ctx);
  return g_hmx_full_lock_rc;
}

// The session-long power vote behind MODULAR_NPU_POWER=1, held from the
// power_vote dispatch until mojo_npu_close. The shape is what the shipped
// ggml-hexagon backend pins for its whole session and never toggles around
// bursts (htp/main.c:426-505): compute client class, DCVS v3 with DCVS
// disabled and every corner at MAX with sleep disabled, the HVX rail up, and
// HMX power with its clock pinned through the v2 request. HAP_power only --
// no compute-resource acquire, no lock, no witness -- so the locked HMX
// compute surface is untouched; qshimHmxLlamacppProbe above is the probe
// that validated this exact sequence on this part.
//
// Voting twice is one vote: FastRPC opens one session per process here, but
// a second host context re-dispatches power_vote, and the power manager
// keys votes by client token, so the guard keeps the release accounting at
// one. The rcs are kept so a re-vote reports what the vote said.
static int g_npu_power_voted = 0;
static int g_npu_power_rcs[4] = {-99, -99, -99, -99};

void qshimNpuPowerVote(uint32_t *report, int word_capacity) {
  if (word_capacity > 0)
    report[0] = (HAP_power_set != 0);
  if (!HAP_power_set)
    return;

  void *client = (void *)&g_npu_power_voted;
  if (!g_npu_power_voted) {
    {
      HAP_power_request_t request;
      memset(&request, 0, sizeof(request));
      request.type = HAP_power_set_apptype;
      request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
      g_npu_power_rcs[0] = HAP_power_set(client, &request);
    }
    {
      HAP_power_request_t request;
      memset(&request, 0, sizeof(request));
      request.type = HAP_power_set_DCVS_v3;
      request.dcvs_v3.set_dcvs_enable = TRUE;
      request.dcvs_v3.dcvs_enable = FALSE;
      request.dcvs_v3.set_bus_params = TRUE;
      request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.set_core_params = TRUE;
      request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
      request.dcvs_v3.set_sleep_disable = TRUE;
      request.dcvs_v3.sleep_disable = TRUE;
      g_npu_power_rcs[1] = HAP_power_set(client, &request);
    }
    {
      HAP_power_request_t request;
      memset(&request, 0, sizeof(request));
      request.type = HAP_power_set_HVX;
      request.hvx.power_up = TRUE;
      g_npu_power_rcs[2] = HAP_power_set(client, &request);
    }
    {
      HAP_power_request_t request;
      memset(&request, 0, sizeof(request));
      request.type = HAP_power_set_HMX_v2;
      request.hmx_v2.set_power = TRUE;
      request.hmx_v2.power_up = TRUE;
      request.hmx_v2.set_clock = TRUE;
      request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
      request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
      request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
      request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
      g_npu_power_rcs[3] = HAP_power_set(client, &request);
    }
    g_npu_power_voted = 1;
  }

  for (int i = 0; i < 4 && i + 1 < word_capacity; ++i)
    report[i + 1] = (uint32_t)g_npu_power_rcs[i];
}

int qshimNpuPowerRelease(void) {
  if (!g_npu_power_voted || !HAP_power_set)
    return (HAP_power_set != 0);
  void *client = (void *)&g_npu_power_voted;
  // The reference backend never downvotes -- the votes die with its process.
  // This session's protection domain also dies with the handle, so the
  // explicit release is hygiene for the case where the process outlives the
  // session: re-enable DCVS and sleep, drop the rails. Corners are left
  // unset; re-enabled DCVS owns them again.
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX_v2;
    request.hmx_v2.set_power = TRUE;
    request.hmx_v2.power_up = FALSE;
    (void)HAP_power_set(client, &request);
  }
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = FALSE;
    (void)HAP_power_set(client, &request);
  }
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = TRUE;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = FALSE;
    (void)HAP_power_set(client, &request);
  }
  g_npu_power_voted = 0;
  return 1;
}

static unsigned int g_pr42_ctx = 0;
static int g_pr42_lock4_rc = -99;
static int g_pr42_lock3_rc = -99;
static int g_pr42_lock_rc = -99;

// The one acquisition shape the probes above could not reach, and the one this
// project's own unlanded matmul work is built on.
//
// `qshimHmxCrmProbe` calls `acquire_cached` but omits `set_hmx_param`;
// `qshimHmxShippingProbe` sets `set_hmx_param` but omits `acquire_cached`. The
// two halves were never combined, and `vtcm_acquire` above says why that
// matters: the manager creates the HMX enable entry only when the acquire
// REQUESTS HMX, and under cache mode leaves it armed-but-DISABLED until
// `acquire_cached` enables it. Either half alone therefore asks for the unit on
// a context that was never actually set up for it.
//
// This runs `vtcm_acquire` itself -- the shared grant, both halves present --
// and then asks for the unit on that context. It reports the raw context value
// and the raw `acquire_cached` status, because the interesting failure is one
// that happens BEFORE the lock.
//
// The blocking `qurt_hmx_lock2` is deliberately not used: every manager entry
// point here carries a bounded timeout, and an indefinite suspend on a single
// poison-prone device slot is its own failure mode.
int qshimHmxPr42Probe(void) {
  if (vtcm_acquire() != 0)
    return (g_pr42_lock4_rc = QSHIM_CRM_ERR_ACQUIRE);
  g_pr42_ctx = g_vtcm_ctx;

  g_pr42_lock4_rc = HAP_compute_res_hmx_lock4(g_vtcm_ctx, &g_hmx_busy,
                                              /*timeout_us=*/10000);
  if (g_pr42_lock4_rc == 0) {
    /* Released inline like the lock3 and plain-lock legs below. This leg
       once returned still holding, on a design assumption the MAC tiles
       never implemented -- they take their own lock through qshimHmxLock
       (hmx_int8_shim.c) rather than running on this grant -- so a granted
       lock4 here leaked the unit past the dispatch. Latent on this part
       (lock4 has always refused here), fixed before the power vote makes
       grants reachable. */
    HAP_compute_res_hmx_unlock4(g_vtcm_ctx, g_hmx_busy);
    g_hmx_busy = NULL;
    return 0;
  }

  compute_res_hmx_mutex_t mutex;
  g_pr42_lock3_rc = HAP_compute_res_hmx_lock3(
      g_vtcm_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &mutex, /*timeout_us=*/10000);
  if (g_pr42_lock3_rc == 0) {
    HAP_compute_res_hmx_unlock3(g_vtcm_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED,
                                &mutex);
    return 0;
  }

  g_pr42_lock_rc = HAP_compute_res_hmx_lock(g_vtcm_ctx);
  if (g_pr42_lock_rc == 0)
    HAP_compute_res_hmx_unlock(g_vtcm_ctx);
  return g_pr42_lock_rc;
}

// Nothing to drop: every leg of `qshimHmxPr42Probe` now releases inline, and
// the rc globals stay as the record of what granted. Kept exported for its
// caller in the probe kernel's no-execute branch.
void qshimHmxPr42Release(void) {}

unsigned int qshimHmxPr42Ctx(void) { return g_pr42_ctx; }
int qshimHmxPr42CachedRc(void) { return g_hmx_cached_rc; }
int qshimHmxPr42Lock4Rc(void) { return g_pr42_lock4_rc; }
int qshimHmxPr42Lock3Rc(void) { return g_pr42_lock3_rc; }
int qshimHmxPr42LockRc(void) { return g_pr42_lock_rc; }

// Runs `fn` on a freshly created QuRT thread with its own stack and joins it.
// Exists for the HMX probes: the reference backend issues every HMX
// instruction from a dedicated thread that has never executed HVX, and this
// reproduces that thread model so a refusal or a dead result on the FastRPC
// dispatch worker can be told apart from one intrinsic to the part.
static char g_fresh_thread_stack[16384] __attribute__((aligned(128)));

int qshimRunOnFreshThread(void (*fn)(void *), void *arg) {
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hmx_f16");
  qurt_thread_attr_set_stack_addr(&attr, g_fresh_thread_stack);
  qurt_thread_attr_set_stack_size(&attr, sizeof(g_fresh_thread_stack));
  qurt_thread_attr_set_priority(
      &attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_t thread;
  if (qurt_thread_create(&thread, &attr, fn, arg) != QURT_EOK)
    return -1;
  int status = 0;
  (void)qurt_thread_join(thread, &status);
  return 0;
}

int __attribute__((weak)) HAP_power_get(void *context,
                                        HAP_power_response_t *response);

static int g_hmx_clk_get_rc = -99;
static unsigned int g_hmx_clk_core_hz = 0;
static unsigned int g_hmx_clk_hmx_hz = 0;
static unsigned int g_hmx_clk_streamer_hz = 0;
static int g_hmx_clk_streamer_rc = -99;
static int g_hmx_power_streamer_rc = -99;

// Whether the power manager's AEE_SUCCESS on an HMX clock vote means anything.
//
// `HAP_power.h:529-530` says a part with no separate HMX clock rejects a
// corner request with AEE_EBADPARM, so a success reads as "the HMX rail
// exists". A power manager that rubber-stamps every rail it does not implement
// produces exactly the same observation, and that alternative has to be closed
// before the success can be quoted.
//
// Three readings do it. `HAP_power_get_clk_Freq` is the positive control: the
// Q6 core clock is unquestionably real, so a zero there means the get path
// itself is not answering and nothing below is interpretable. The HMX core
// clock is the measurement. The streamer clock is the negative control -- a
// compute DSP carries no camera streamer, so a plausible frequency there, or a
// success on a streamer VOTE, means this manager answers for rails it does not
// have and its HMX success carries no information.
int qshimHmxClockProbe(void) {
  if (!HAP_power_get || !HAP_power_set)
    return (g_hmx_clk_get_rc = -105);

  void *client = (void *)&g_hmx_clk_get_rc;

  HAP_power_response_t response;
  memset(&response, 0, sizeof(response));
  response.type = HAP_power_get_clk_Freq;
  g_hmx_clk_get_rc = HAP_power_get(client, &response);
  if (g_hmx_clk_get_rc == 0)
    g_hmx_clk_core_hz = response.clkFreqHz;

  memset(&response, 0, sizeof(response));
  response.type = HAP_power_get_hmx_core_clk_Freq;
  if (HAP_power_get(client, &response) == 0)
    g_hmx_clk_hmx_hz = response.clkFreqHz;

  memset(&response, 0, sizeof(response));
  response.type = HAP_power_get_streamer0_core_clk_Freq;
  g_hmx_clk_streamer_rc = HAP_power_get(client, &response);
  if (g_hmx_clk_streamer_rc == 0)
    g_hmx_clk_streamer_hz = response.clkFreqHz;

  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_streamer;
  request.streamer.set_streamer0_clk = TRUE;
  request.streamer.streamer0_clkFreqHz = 100000000;
  g_hmx_power_streamer_rc = HAP_power_set(client, &request);

  return g_hmx_clk_get_rc;
}

int qshimHmxClockGetRc(void) { return g_hmx_clk_get_rc; }
unsigned int qshimHmxClockCoreHz(void) { return g_hmx_clk_core_hz; }
unsigned int qshimHmxClockHmxHz(void) { return g_hmx_clk_hmx_hz; }
unsigned int qshimHmxClockStreamerHz(void) { return g_hmx_clk_streamer_hz; }
int qshimHmxClockStreamerRc(void) { return g_hmx_clk_streamer_rc; }
int qshimHmxPowerStreamerRc(void) { return g_hmx_power_streamer_rc; }

int qshimHmxPowerResident(void) { return g_hmx_power_resident; }
int qshimHmxPowerApptypeRc(void) { return g_hmx_power_apptype_rc; }
int qshimHmxPowerDcvsRc(void) { return g_hmx_power_dcvs_rc; }
int qshimHmxPowerHvxRc(void) { return g_hmx_power_hvx_rc; }
int qshimHmxPowerHmxRc(void) { return g_hmx_power_hmx_rc; }
int qshimHmxPowerHmxV2Rc(void) { return g_hmx_power_hmx_v2_rc; }
int qshimHmxFullAcquireRc(void) { return g_hmx_full_acquire_rc; }
int qshimHmxFullHvxUnlockRc(void) { return g_hmx_full_hvx_unlock_rc; }
int qshimHmxFullLockRc(void) { return g_hmx_full_lock_rc; }

// P5 M2.4 order 066: `qshimHmxLock`'s new lock3 attempt, inserted between the
// lock4 attempt and the plain lock to match `qshimHmxPr42Probe` line for
// line. -99 means `qshimHmxLock` was never reached this dispatch (the getter
// reads the same global every call, like the others in this block).
int qshimHmxLock3Rc(void) { return g_hmx_lock3_rc; }

// P5 M2.4 order 065: repointed at `hmx_ctx_acquire`'s own two-step
// acquisition (see that function, above `qshimGetHmxVtcmAddr`) rather than at
// a release/reacquire cycle on the shared context -- same witness word (11),
// per direct instruction to reuse it. Order 066: `qshimHmxLock` no longer
// calls `hmx_ctx_acquire` (reverted to `g_vtcm_ctx`, see below), and nothing
// else in the active dispatch path does either -- this now always reads its
// initial -99 sentinel, held over from #281 rather than repurposed, per
// direct instruction to keep the dump's word count stable across rounds.
int qshimHmxReleaseRc(void) { return g_hmx_release_rc; }

// The dedicated context's own `acquire_cached` rc, word 12. Same order-066
// note as `qshimHmxReleaseRc` immediately above: `hmx_ctx_acquire` is unused
// by the active lock path as of this round, so this reads -99 unless some
// other caller reaches the dedicated-context wrappers.
int qshimHmxBracketRc(void) { return g_hmx_bracket_rc; }

// Whether this build may escalate to the blocking lock and real HMX
// instructions. Reported to the kernel so a probe-only build is visible in the
// witness rather than looking like a path that silently did not run.
int qshimHmxExecuteEnabled(void) { return SKYE_HMX_EXECUTE; }

// Set while this process holds the unit, cleared by the release below. Exists
// because `HAP_compute_res_hmx_lock` takes NO timeout and blocks: if a grant is
// ever left held, the next dispatch's array thread waits on it forever, inside
// the kernel, before it has reached a rendezvous the bounded barrier could
// unwind. One launch runs at a time and the two functions below are the only
// takers, so the only party that can be holding it is a previous dispatch of
// ours, and that is recoverable rather than fatal.
static int g_hmx_held = 0;
static int g_hmx_leak_recoveries = 0;

// ===--------------------------------------------------------------------=== //
// The step-scoped grant (P5 R2)
// ===--------------------------------------------------------------------=== //

// The bracket a whole denoise step's dispatches share, and its bookkeeping.
// Closed by default, which is what makes the flag-off arm take-and-give-back
// per dispatch exactly as it always has: with no bracket open, `scope_lock` is
// an acquire and `scope_unlock` is a release, and the two functions below are
// unchanged underneath.
//
// The rule the boundary is argued from is `qurt_hmx_unlock`'s: it "unlock[s]
// the HMX unit and clear[s] its accumulators" (`qurt_hmx.h:139-143`). That sets
// a LOWER bound on the hold -- the grant must span every tile that accumulates
// together -- and today's per-dispatch hold already exceeds it, since
// `hmx_mma_panel` issues `mxclracc.hf` at the top of every output tile. What
// widening the hold buys is therefore not accumulation that would otherwise be
// lost; it is the 163 manager round trips a step that are not taken, and a
// grant that has one place to leak from instead of 163
// (`HMX_PERSISTENT_TOPOLOGY_DESIGN.md` §2).
static npu_hmx_scope_t g_hmx_scope;

static int hmx_grant_acquire(void *ctx);
static int hmx_grant_release(void *ctx);
static const npu_hmx_grant_ops_t g_hmx_grant_ops = {hmx_grant_acquire,
                                                    hmx_grant_release, 0};

// Opens the bracket. Takes nothing: the unit is per thread and this runs on the
// RPC thread, which is not the one that will issue the MMAs. The step's first
// matmul dispatch takes it on the array servicer, and the bracket is what stops
// that dispatch giving it back.
void qshimHmxScopeOpen(void) { npu_hmx_scope_step_begin(&g_hmx_scope); }

// Closes it. MUST run on the servicer, which is why `mojo_npu_imp.c` publishes
// it as a task rather than calling it from the walker.
//
// Returns what `npu_hmx_scope_step_end` returned, and the caller must not throw
// it away: nonzero means the unit was NOT given back -- either the release
// refused, or the grant is held by a thread that is not this one and no release
// this thread can issue would reach it. Either way the unit is live into the
// next dispatch, whose acquire has no timeout to recover from.
int qshimHmxScopeClose(void) {
  return npu_hmx_scope_step_end(&g_hmx_scope, g_hmx_grant_ops,
                                (uintptr_t)qurt_thread_get_id());
}

// Whether a standing grant is live, so the device A/B can tell a pooled run
// from a cycling one without consulting the build flag. Deliberately NOT a
// witness word: the witness protocol is per dispatch and is not this change's
// to extend.
int qshimHmxStandingGrant(void) { return npu_hmx_scope_standing(&g_hmx_scope); }
int qshimHmxScopeAcquires(void) { return g_hmx_scope.acquires; }
int qshimHmxScopeInherited(void) { return g_hmx_scope.inherited; }

// What the kernel calls, unchanged in name and in contract: 0 means this thread
// may issue HMX. Inside an open bracket the thread that already holds the unit
// is handed it back without a manager round trip; any other thread falls
// through to a real acquisition, because a peer handed a grant it does not hold
// would issue MMAs on an ungranted unit.
int qshimHmxLock(void) {
  return npu_hmx_scope_lock(&g_hmx_scope, g_hmx_grant_ops,
                            (uintptr_t)qurt_thread_get_id());
}

// And the matching release, which inside the bracket defers to the step's own.
int qshimHmxUnlock(void) {
  return npu_hmx_scope_unlock(&g_hmx_scope, g_hmx_grant_ops,
                              (uintptr_t)qurt_thread_get_id());
}

// Takes the HMX unit for real, for the thread that will issue the
// instructions.
//
// `qurt_hmx_lock2` was once believed unreachable from this (unsigned cdsp)
// domain: a skel referencing it failed to load with AEE_EUNABLETOLOAD, read as
// HMX being gated to a signed protection domain. That was a linkage failure,
// not a permission one -- the symbol was left as a dynamic import no resident
// library exports, while the SDK's own `libqurt.a` defines it
// (`qurt_hmx_asm.o`) as a self-contained `trap0(#0x1d)` stub. `build.sh` now
// links that object statically. Reaching the syscall is not the same as being
// granted the unit, which is what the probes above exist to establish.
static int hmx_grant_acquire(void *ctx) {
  (void)ctx;
  // A grant that outlived its dispatch. Give it back before asking for it
  // again, because asking would block forever on our own unreleased unit --
  // order 088's stall shape: a launch that never returns and so never appears
  // in the host's records at all.
  if (g_hmx_held) {
    ++g_hmx_leak_recoveries;
    (void)hmx_grant_release(0);
    g_hmx_held = 0;
  }

  // P5 M2.4 order 066: back on the shared `g_vtcm_ctx`. 065's device
  // evidence refuted the dedicated `g_hmx_ctx` (#281) as the differentiator
  // -- it refuses too -- and its own side-by-side read of this file found
  // the granting call (`qshimHmxPr42Probe`) and this one agree on all five
  // context attrs, but differ in two places that were never varied before:
  // the acquire timeout (`vtcm_acquire` now 1000000, matching), and an
  // intervening `HAP_compute_res_hmx_lock3` attempt the probe makes and this
  // function did not (below). `hmx_ctx_acquire`/`g_hmx_ctx` stay in the tree,
  // unused by this call, for the record.
  if (vtcm_acquire() != 0)
    return -1;

  // P5 M2.4 order 062: the competitor-researcher's read of ggml-hexagon
  // (llama.cpp's Hexagon backend, `.claude/agent-memory/competitor-researcher/
  // ggml-hexagon-map.md` at SHA `69bf6437914`) supersedes an earlier
  // unlock/relock-around-the-lock attempt this file carried briefly. Their
  // HVX/HMX exclusion is solved by THREAD PURITY, not by releasing and
  // retaking HVX around the lock: a dedicated worker thread runs only
  // HVX-free closures (grep-verified), and the one `qurt_hvx_unlock` in their
  // tree is an idle-spin courtesy, never on the acquire path. The matching
  // move here is at the call site, not in this function: `mojo_matmul_f16`
  // now runs the HMX stripe on its launch's structurally-idle "5th" thread
  // instead of thread 0, which is where the actual HVX/HMX exclusion risk
  // was. `g_hvx_mode` stays a pure, side-effect-free read so a witness dump
  // can confirm that thread never reports holding HVX -- unlike thread 0.
  g_hvx_mode = qurt_hvx_get_mode();

  // Also dropped per that read: `qurt_hmx_try_lock` / `unlock2` / `lock2`
  // (the "QuRT-first" preamble). ggml-hexagon calls ONLY `HAP_compute_res_*`
  // -- `qurt_hmx_*` has zero hits in their tree -- and that preamble creates
  // acquisition state (`g_hmx_held_by_qurt`, an actual QuRT-level lock) their
  // reference never creates. `qshimHmxTryRc`/`qshimHmxQurtRc` now correctly
  // read their -99 "never attempted" sentinel after a real matmul dispatch:
  // that IS the faithful-copy answer, not a regression. `qshimHmxTryLock`'s
  // own standalone probe is untouched and still exercises that path on its
  // own, separately.
  //
  // `lock4` is kept: unlike the QuRT syscalls, it IS a `HAP_compute_res_*`
  // entry point, and the KB's own device evidence already has it refusing
  // with QURT_EVAL at negligible cost on this manager, falling through to the
  // plain lock ggml-hexagon's own worker thread actually takes.
  int rc = HAP_compute_res_hmx_lock4(g_vtcm_ctx, &g_hmx_busy,
                                     /*timeout_us=*/10000);
  if (rc == 0) {
    g_hmx_held = 1;
    return 0;
  }

  // P5 M2.4 order 066: inserted to match `qshimHmxPr42Probe` line for line --
  // same context, same args, same timeout. Expected to refuse (the KB has
  // lock3 refusing everywhere it has been tried on this manager), tracked
  // for completeness in case it does not: `g_hmx_held_by_lock3` +
  // `g_hmx_lock3_mutex` so `qshimHmxUnlock` releases through the matching
  // entry point if this ever succeeds.
  g_hmx_lock3_rc = HAP_compute_res_hmx_lock3(
      g_vtcm_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &g_hmx_lock3_mutex,
      /*timeout_us=*/10000);
  if (g_hmx_lock3_rc == 0) {
    g_hmx_held_by_lock3 = 1;
    g_hmx_held = 1;
    return 0;
  }

  // The plain lock, which is the one this part actually grants, and the one
  // entry point ggml-hexagon's worker thread also uses (`hmx-queue.c:20`,
  // sticky per op-batch, no timeout, return value discarded there -- kept
  // checked and witnessed here instead, per the read's own note). Now
  // IDENTICAL to `qshimHmxPr42Probe`'s own sequence into this same call, on
  // the same context: if this still refuses, the source-level suspect list
  // (shape, freshness, thread, context object, choreography) is empty.
  g_hmx_busy = NULL;
  // Reuses `qshimHmxFullLockRc`'s global: today it also carries
  // `qshimHmxLlamacppProbe`'s own plain-lock rc, from that probe's separate,
  // self-built context. Both readings mean the same thing -- "the last plain
  // `HAP_compute_res_hmx_lock` rc this process observed" -- so a witness read
  // after a real matmul dispatch reflects THIS call, provided no probe ran
  // later in the same process.
  g_hmx_full_lock_rc = HAP_compute_res_hmx_lock(g_vtcm_ctx);
  if (g_hmx_full_lock_rc == 0)
    g_hmx_held = 1;
  return g_hmx_full_lock_rc;
}

// Whether this process believes it currently holds the unit, and how many times
// it has found the belief already true on entry (P5 M4.2 order 088). The second
// number is the one that matters: it can only be nonzero if a grant outlived
// the dispatch that took it.
int qshimHmxHeld(void) { return g_hmx_held; }
int qshimHmxLeakRecoveries(void) { return g_hmx_leak_recoveries; }

// Releases the HMX unit from the calling thread: same thread as the lock, and
// through whichever path acquired it. Also clears the accumulators, which is
// the semantics `npu_hmx_scope_unlock` above defers this call for.
static int hmx_grant_release(void *ctx) {
  (void)ctx;
  g_hmx_held = 0;
  if (g_hmx_held_by_qurt) {
    g_hmx_held_by_qurt = 0;
    return qurt_hmx_unlock2(QURT_HMX_NON_SHARED_UNLOCK);
  }
  // P5 M2.4 order 066: matches qshimHmxLock's new lock3 attempt. Mirrors
  // qshimHmxPr42Probe's own release (HAP_compute_res_hmx_unlock3 with the
  // SAME mutex object lock3 handed back) -- expected dead in practice, since
  // lock3 is not expected to grant on this manager, kept for correctness if
  // it ever does.
  if (g_hmx_held_by_lock3) {
    g_hmx_held_by_lock3 = 0;
    return HAP_compute_res_hmx_unlock3(
        g_vtcm_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED, &g_hmx_lock3_mutex);
  }
  if (g_vtcm_ctx == 0)
    return -1;
  // `unlock4` is only correct for a lock4 grant, which hands back the busy
  // flag; a null flag means the plain lock took the unit and the plain unlock
  // is the matching release. Releasing through the wrong entry point leaves
  // the unit held, which poisons the device for every later run.
  if (g_hmx_busy == NULL)
    return HAP_compute_res_hmx_unlock(g_vtcm_ctx);
  int rc = HAP_compute_res_hmx_unlock4(g_vtcm_ctx, g_hmx_busy);
  g_hmx_busy = NULL;
  return rc;
}

// The non-blocking probe's status: the one number Monday's run turns on.
int qshimHmxTryRc(void) { return g_hmx_try_rc; }

// The blocking lock's status, so the host can tell "the syscall refused" from
// "the syscall was never reached" rather than inferring it from the lock4 code.
int qshimHmxQurtRc(void) { return g_hmx_qurt_rc; }

// `qurt_hvx_unlock`'s status, and the configured HVX mode. Both -99 unless the
// HVX-release variant is compiled in.
int qshimHvxUnlockRc(void) { return g_hvx_unlock_rc; }
int qshimHvxMode(void) { return g_hvx_mode; }

// P5 M2.4 order 067: called once per launch, on the calling thread, before
// `npu_run_launch` creates any participant (`mojo_npu_imp.c`). Mechanism: on
// this SDK generation HVX is acquired IMPLICITLY, on exception, by whichever
// thread issues the first vector instruction (`qurt_hvx.h:140-171` -- every
// explicit lock/unlock/get_mode entry point is `@deprecated`, "handled
// internally by QuRT"), and nothing releases it automatically. The thread
// FastRPC hands each dispatch to (`npu_run_launch`'s participant 0) is REUSED
// across every kernel this skel dispatches, not just matmul -- if it ever ran
// an HVX-genuine kernel (softmax, groupnorm, gemv, the ordinary HVX matmul
// fallback for a non-HMX-shape) with nothing to release it after, it carries
// that context into the next dispatch's HMX attempt. The release is
// self-healing: a kernel that itself needs HVX just re-triggers the same
// implicit acquisition on its own first vector instruction, so calling this
// unconditionally, for every selector, costs nothing but the syscall.
//
// `qurt_hvx_set` (not `qurt_hvx_unlock`) is the call: the SDK marks
// `qurt_hvx_unlock` `@deprecated` with "Use the qurt_hvx_set API to release
// HVX resource" (`qurt_hvx.h:176`), and `qurt_hvx_set` itself carries no
// `@deprecated` tag (`:371-391`) -- it is the modern form the deprecated one
// points to. Its result is `unsigned int` with only the low 8 bits
// meaningful (`QURT_EOK` 0, `QURT_HVX_SET_ERROR` 0xFF, `:379-382`); the high
// bits are unspecified, so they are masked off before storing.
//
// The header documents no distinct "no context was held" code -- releasing a
// context this thread never took is not called out as a separate failure
// mode from `QURT_HVX_SET_ERROR`, so this rc is recorded for the witness and
// deliberately NOT gated on: a launch proceeds identically whichever value
// comes back, because failing a whole dispatch over a best-effort defensive
// release would be a worse failure mode than the one this guards against.
static int g_hvx_launch_release_rc = -99;

void qshimHvxReleaseAtLaunch(void) {
#if SKYE_HVX_RELEASE_VIA_UNLOCK
  /* Order 067 lane isolation, NOT part of #283: this device's QuRT does not
     resolve `qurt_hvx_set`, so a skel referencing it refuses to load at all
     (AEE_EUNABLETOLOAD) while a byte-identical kernel object loads without it.
     `qurt_hvx_unlock` is the deprecated form the SDK tells callers to replace
     with `qurt_hvx_set`, and it IS resolvable here -- so it is what this lane
     can actually run the release experiment with. */
  g_hvx_launch_release_rc = (int)(qurt_hvx_unlock() & 0xFFu);
#else
  g_hvx_launch_release_rc =
      (int)(qurt_hvx_set(QURT_HVX_RELEASE_CONTEXT) & 0xFFu);
#endif
}

int qshimHvxLaunchReleaseRc(void) { return g_hvx_launch_release_rc; }

// The acquire_cached status recorded during the VTCM/HMX acquire, surfaced
// through the kernel's witness so the host can tell an acquire problem from a
// lock problem without inference.
int qshimHmxCachedRc(void) { return g_hmx_cached_rc; }

// Enter / leave the HMX critical section: set the busy flag before the HMX
// instructions and clear it after. Autonomous preemption waits for the flag to
// clear before reassigning HMX, so the instructions must run between them.
void qshimHmxEnter(void) {
  if (g_hmx_busy)
    *g_hmx_busy = 1u;
}

void qshimHmxExit(void) {
  if (g_hmx_busy)
    *g_hmx_busy = 0u;
}

// The compute-resource preemption capability mask (bit 0 cooperative, 1
// autonomous, 2 threads-for-autonomous, 3 deferred, 4 hmx-lock4), or 0 if the
// query is unavailable. Reported through the kernel's witness so the HMX lock
// path can be matched to what the device actually implements rather than
// guessed.
unsigned int qshimHmxCaps(void) {
  unsigned int mask = 0;
  if (HAP_compute_res_query_capability(HAP_COMPUTE_RES_PREEMPTION_CAPABILITY,
                                       &mask) != 0)
    return 0;
  return mask;
}

// ===--------------------------------------------------------------------=== //
// Launch geometry
// ===--------------------------------------------------------------------=== //

// The threads of the launch in progress, slot `i` holding the QuRT id of the
// thread bound to launch index `i`. Keyed by thread id because that is the only
// identity a thread can ask for here: this shared object is loaded into a
// protection domain that gives it no thread-local storage.
//
// One launch at a time, which is what the transport delivers — an invocation
// returns only once the device is done with it — so a second concurrent
// invocation would overwrite this rather than get a table of its own.
static qurt_thread_t g_launch_threads[MOJO_NPU_MAX_LAUNCH_THREADS];
static uint32_t g_launch_count = 1;

// No QuRT thread carries this id, so an unfilled slot cannot match a live
// thread's lookup.
#define QSHIM_UNBOUND_THREAD ((qurt_thread_t)0xFFFFFFFFu)

// The launch-scoped rendezvous a kernel that splits one output across its
// participants arrives at (P5 M4.2's pipelined HMX matmul is the first).
// Reset by `mojo_npu_launch_begin` rather than by the kernel: that runs before
// any participant exists, so the reset races nothing, and a launch that was
// abandoned mid-rendezvous cannot leave the next one waiting on a generation
// that already passed.
static QShimThreadSyncStruct g_launch_sync;

// Sticky, launch-scoped: once any participant gives up on a rendezvous, every
// peer gives up too. Without it the first thread to hit the deadline would
// leave the rest waiting on an arrival that is now never coming, which is the
// wedge with extra steps.
//
// The same word is what the trampoline reports to the host
// (`mojo_npu_launch_fold_status`): a launch that gave up left part of its
// output unwritten, and one record serves both readers because there is one
// fact. Its own semantics are in `mojo_npu_launch.h`.
static npu_launch_status_t g_launch_status;

// The deadline. `NPU_SYNC_SPINS` covers the healthy case with no syscall at
// all: peers in a pipelined kernel arrive within a step, which is microseconds.
// The nap loop only ever runs when something is already wrong, and its product
// is the bound -- 2 s, which is four orders of magnitude longer than any step
// this kernel takes and four orders shorter than the 24 minutes order 084 spent
// finding out that an unbounded wait is unbounded.
//
// That premise is what order 114 falsified: the nap loop runs on EVERY barrier,
// not only when something is wrong, because nothing wakes it. So the poll
// period is a first-order term of dispatch cost, and order 115 measured the
// tuned triple below taking the matmul family 3.027x.
//
// THE TWO ARE A PAIR AND MUST MOVE TOGETHER. Their product is the deadline --
// 50 us x 40000 is the same 2 s bound the original 1000 x 2000 gave, four
// orders of magnitude longer than any step this kernel takes and four orders
// shorter than the 24 minutes order 084 spent finding out that an unbounded
// wait is unbounded. Shortening the nap alone would cut the deadline to 100 ms
// and turn the wedge detector into a flake.
//
// Still overridable from the build, so the A/B against the wakeful arm below
// can put either arm on the command line rather than in a source edit; a build
// that names one gets exactly what it names.
#ifndef NPU_SYNC_SPINS
#define NPU_SYNC_SPINS 4000u
#endif
#ifndef NPU_SYNC_NAP_MICROS
#define NPU_SYNC_NAP_MICROS 50u
#endif
#ifndef NPU_SYNC_MAX_NAPS
#define NPU_SYNC_MAX_NAPS 40000u
#endif

// The wakeful arm (P5, order 116). OFF by default and off is the shipping
// configuration: with it off this file compiles the nap barrier order 084 wrote
// and the object is byte-identical to its base. What it trades and why it is a
// flag rather than a replacement is in `npu_sync_barrier.h`'s header comment --
// briefly, the wall-clock deadline does not survive a wait with no timeout, and
// the resident-proven symbol list has no timed futex.
#ifndef SKYE_SYNC_WAKEFUL
#define SKYE_SYNC_WAKEFUL 0
#endif

#if SKYE_SYNC_WAKEFUL
// The same pair `mojo_npu_imp.c` gives the pool, for the same reason: both
// names are on `npu_skel_symbol_gate.sh`'s resident-proven allowlist because
// `qshimSyncThreads` and the pool already import them from skels that load and
// pass. This arm introduces no new QuRT symbol.
static void qshim_sync_park(volatile int32_t *word, int32_t expected,
                            void *ctx) {
  (void)ctx;
  qurt_futex_wait((void *)word, (int)expected);
}

static void qshim_sync_wake(volatile int32_t *word, int32_t count, void *ctx) {
  (void)ctx;
  qurt_futex_wake((void *)word, (int)count);
}

// The barrier hands its exit kind straight to the probe, so the two
// enumerations have to agree. Pinned rather than mapped: a renumbering on
// either side would otherwise move counts between the histogram's buckets, and
// a wrong bucket reads exactly like a finding.
_Static_assert(NPU_SYNC_EXIT_LAST_IN == NPU_WAIT_EXIT_LAST_IN &&
                   NPU_SYNC_EXIT_SPIN == NPU_WAIT_EXIT_SPIN &&
                   NPU_SYNC_EXIT_PARK == NPU_WAIT_EXIT_NAP &&
                   NPU_SYNC_EXIT_GIVE_UP == NPU_WAIT_EXIT_GIVE_UP,
               "the wakeful barrier's exit kinds must match the phase probe's");
#endif

// The skew-adaptive spin budget (P5 R1, RENDEZVOUS_DESIGN; order 227's
// mechanism). OFF by default and off is the shipping configuration: with it
// off this file compiles the fixed-budget barrier order 084 wrote — the
// additions are #if-guarded or expand to ((void)0) — and the object is
// byte-identical to its base. Policy, bounds, and the hosttest that drives
// them: `npu_sync_adapt.h`. The nap loop, its deadline pair, and the sticky
// abort are untouched on both arms; only the spin budget ahead of them moves.
#ifndef SKYE_SYNC_ADAPTIVE
#define SKYE_SYNC_ADAPTIVE 0
#endif

#if SKYE_SYNC_ADAPTIVE
#include "npu_sync_adapt.h"

// Zeroed at skel load; slots deliberately persist across launches (the
// serial-correlation carry is the win — header comment). One writer per
// slot: the thread holding that launch index, and launches are serial.
static npu_sync_adapt_t g_sync_adapt;
#endif

// The within-dispatch phase probe (P5, order 114). OFF unless the build asks,
// and off is the shipping configuration: with it off this file reads no clock,
// the barrier is byte-for-byte the function order 084 wrote, and the entry
// points below are empty bodies that exist only so a Mojo side built with its
// own flag set still links against a shim built without one. The two flags are
// set together by `build.sh HMX_PHASE_PROBE=1`; a mismatch is inert rather than
// a link error, which is the failure mode to prefer for an instrument.
//
// Design and the mechanisms it discriminates: `npu_phase_probe.h`, and
// `.agentwork/flux-npu/ARRAY_STALL_DIAGNOSIS.md` §4.
#ifndef SKYE_NPU_PHASE_PROBE
#define SKYE_NPU_PHASE_PROBE 0
#endif

#if SKYE_NPU_PHASE_PROBE
#include "npu_sync_report.h"

// Launch-scoped, and cleared by `mojo_npu_launch_begin` for exactly the reason
// `g_launch_sync` is: that runs before any participant exists, so the reset
// races nothing and an abandoned launch cannot leave residue in the next one's
// record.
static npu_phase_report_t g_phase_report;

// SESSION-scoped, deliberately NOT cleared per launch: the counters are
// monotone for the skel's lifetime, like `g_udma_report`, so a reader diffs
// rows for per-render deltas and reads the last row for the session. Order
// 243's unreadable witness is the reason this exists; the egress is the
// report slot, through `qshimPhaseCopy`.
static npu_sync_session_t g_sync_session;

// Both clocks at once, since a mark that read them at different instants would
// attribute a span to one phase in pcycles and to another in qticks -- and
// their divergence is the whole point of carrying two (`npu_phase_probe.h`).
static inline void phase_now(uint64_t *pcycle, uint64_t *qtick) {
  *pcycle = HAP_perf_get_pcycles();
  *qtick = HAP_perf_get_qtimer_count();
}
#endif

// Opens a participant's first span. Called once per thread by the kernel, after
// it knows its launch index and before its first phase.
void qshimPhaseOpen(uint32_t thread_id) {
#if SKYE_NPU_PHASE_PROBE
  uint64_t pcycle, qtick;
  phase_now(&pcycle, &qtick);
  npu_phase_open(&g_phase_report, thread_id, pcycle, qtick);
#else
  (void)thread_id;
#endif
}

// Ends the open phase and begins `phase`. One clock pair per call; the caller
// never states a duration, so the buckets partition the dispatch by
// construction.
void qshimPhaseMark(uint32_t thread_id, uint32_t phase) {
#if SKYE_NPU_PHASE_PROBE
  uint64_t pcycle, qtick;
  phase_now(&pcycle, &qtick);
  npu_phase_mark(&g_phase_report, thread_id, phase, pcycle, qtick);
#else
  (void)thread_id;
  (void)phase;
#endif
}

// Copies the records out for the launch's timing slot. Returns the bytes
// written, which is 0 when the probe is off or the slot is short -- a host that
// reads 0 knows it received nothing rather than a plausible set of zeros.
//
// The slot's size decides what travels, and every skew is graceful: the
// launch-scoped phase record goes first when it fits whole, and the 80-byte
// session sync block (`npu_sync_report.h`) goes after it -- or alone, at
// offset 0, when the host sized the slot for only it. A host that sent the
// old phase-exact slot gets exactly the old record; a host that reads a sync
// block this skel never wrote finds its own zeroes where the magic belongs.
uint32_t qshimPhaseCopy(void *dst, uint32_t bytes) {
#if SKYE_NPU_PHASE_PROBE
  uint32_t pos = 0;
  if (dst == 0)
    return 0;
  if (bytes >= (uint32_t)sizeof(g_phase_report)) {
    memcpy(dst, &g_phase_report, sizeof(g_phase_report));
    pos = (uint32_t)sizeof(g_phase_report);
  }
  if (bytes - pos >= (uint32_t)sizeof(npu_sync_report_wire_t)) {
    npu_sync_report_wire_t wire;
    uint64_t budget_hi = 0;
#if SKYE_SYNC_ADAPTIVE
    for (uint32_t t = 0; t < MOJO_NPU_MAX_LAUNCH_THREADS; ++t)
      if (g_sync_adapt.budget_hi[t] > budget_hi)
        budget_hi = g_sync_adapt.budget_hi[t];
#endif
    npu_sync_report_fill(&wire, &g_sync_session, budget_hi);
    memcpy((uint8_t *)dst + pos, &wire, sizeof(wire));
    pos += (uint32_t)sizeof(wire);
  }
  return pos;
#else
  (void)dst;
  (void)bytes;
  return 0;
#endif
}

// Arrival barrier over the launch's own participants, WITH A DEADLINE.
//
// Deliberately not `qshimSyncThreads`: that one waits on a futex forever, and
// order 084 wedged a device for 24 minutes proving that no amount of care in a
// kernel's arithmetic makes an unbounded wait safe. A rendezvous only completes
// when every participant reaches it, and whether they do depends on the launch
// machinery (a `qurt_thread_create` that failed partway leaves peers that were
// created running and a participant count that says more are coming), not on
// the kernel. So the wait is bounded and the caller is told, rather than the
// device being left to a reboot.
//
// Returns 0 when every participant arrived, nonzero when the caller should
// unwind. A nonzero return is sticky for the rest of the launch.
uint32_t qshimLaunchSyncWait(uint32_t thread_id) {
  QShimThreadSyncStruct *sync = &g_launch_sync;
#if SKYE_NPU_PHASE_PROBE
  // Which loop the wait ends in has never been counted, and the leading
  // explanation for the array's stall turns entirely on it: this poll has no
  // wake, so a participant that misses the spin window learns the gate opened
  // up to `NPU_SYNC_NAP_MICROS` late. The arithmetic has matched that since
  // order 096 and matching arithmetic is not a measurement.
  uint64_t wait_naps = 0;
  uint64_t wait_entry = HAP_perf_get_qtimer_count();
// One clock read feeding both records, so the launch-scoped histogram and the
// session-scoped counters cannot disagree about the same wait.
#define PHASE_WAIT_DONE(kind)                                                  \
  do {                                                                         \
    uint64_t wait_qticks = HAP_perf_get_qtimer_count() - wait_entry;           \
    npu_phase_wait(&g_phase_report, thread_id, (kind), wait_naps,              \
                   wait_qticks);                                               \
    npu_sync_session_note(&g_sync_session, thread_id, (kind), wait_naps,       \
                          wait_qticks);                                        \
  } while (0)
#else
  (void)thread_id;
#define PHASE_WAIT_DONE(kind) ((void)0)
#endif

  // The adapt hooks speak the probe's exit numbering by construction: the
  // literals below are 0=LAST_IN, 1=SPIN, 2=NAP, 3=GIVE_UP, the same kinds
  // PHASE_WAIT_DONE receives at each site (pinned to the wakeful barrier's
  // by the _Static_assert above when that arm is on).
#if SKYE_SYNC_ADAPTIVE
#define NPU_ADAPT_AFTER(kind)                                                  \
  npu_sync_adapt_after(&g_sync_adapt, thread_id, (kind), NPU_SYNC_SPINS)
#else
#define NPU_ADAPT_AFTER(kind) ((void)0)
#endif

  if (npu_launch_status_failed(&g_launch_status))
    return 1;

  const uint32_t participants =
      (uint32_t)__builtin_popcount((uint32_t)sync->thread_mask);
  if (participants <= 1)
    return 0;

  // A kernel naming more threads than the launch spawned is refused here rather
  // than waited on: the arrivals it is short of are never coming.
  if (participants > __atomic_load_n(&g_launch_count, __ATOMIC_ACQUIRE)) {
    npu_launch_status_fail(&g_launch_status);
    return 1;
  }

#if SKYE_SYNC_WAKEFUL
  // The state machine, the #325 gate re-check and the abort wakes all live in
  // `npu_sync_barrier.h`, where the host twin drives them. The exit codes are
  // numbered to match the probe's, so the histogram's `nap` bucket counts PARKS
  // here -- which is the comparison the device leg makes.
  npu_sync_wait_ops_t ops = {qshim_sync_park, qshim_sync_wake, 0};
#if SKYE_SYNC_ADAPTIVE
  npu_sync_wait_t wait = npu_sync_wakeful_wait(
      &sync->data, participants,
      npu_sync_adapt_budget(&g_sync_adapt, thread_id, NPU_SYNC_SPINS),
      NPU_SYNC_MAX_PARKS, &g_launch_status, ops);
#else
  npu_sync_wait_t wait =
      npu_sync_wakeful_wait(&sync->data, participants, NPU_SYNC_SPINS,
                            NPU_SYNC_MAX_PARKS, &g_launch_status, ops);
#endif
#if SKYE_NPU_PHASE_PROBE
  wait_naps = wait.parks;
#endif
  NPU_ADAPT_AFTER(wait.exit_kind);
  PHASE_WAIT_DONE(wait.exit_kind);
  return wait.give_up;
#else
  uint32_t prev =
      (uint32_t)__atomic_fetch_add(&sync->data, 1, __ATOMIC_ACQ_REL);
  uint32_t generation = prev >> 16;
  uint32_t arrived = (prev & 0xFFFFu) + 1;

  if (arrived == participants) {
    // A peer that gave up counted its arrival before it unwound, so the count
    // can reach the full width with a participant already gone. Re-read the
    // record here rather than trusting the entry check: this thread is the one
    // that releases every straggler behind it, and they would go on to read
    // output the departed peer never wrote.
    if (!npu_launch_status_gate_open(&g_launch_status))
      return 1;
    // Last in: open the gate and clear the count in one store, so no thread can
    // observe the next generation with a stale arrival count.
    __atomic_store_n(&sync->data, (int32_t)((generation + 1) << 16),
                     __ATOMIC_RELEASE);
    NPU_ADAPT_AFTER(0u);
    PHASE_WAIT_DONE(NPU_WAIT_EXIT_LAST_IN);
    return 0;
  }

#if SKYE_SYNC_ADAPTIVE
  const uint32_t spin_budget =
      npu_sync_adapt_budget(&g_sync_adapt, thread_id, NPU_SYNC_SPINS);
  for (uint32_t spin = 0; spin < spin_budget; ++spin)
#else
  for (uint32_t spin = 0; spin < NPU_SYNC_SPINS; ++spin)
#endif
    if (((uint32_t)__atomic_load_n(&sync->data, __ATOMIC_ACQUIRE) >> 16) !=
        generation) {
      NPU_ADAPT_AFTER(1u);
      PHASE_WAIT_DONE(NPU_WAIT_EXIT_SPIN);
      return 0;
    }

  for (uint32_t nap = 0; nap < NPU_SYNC_MAX_NAPS; ++nap) {
    if (((uint32_t)__atomic_load_n(&sync->data, __ATOMIC_ACQUIRE) >> 16) !=
        generation) {
      NPU_ADAPT_AFTER(2u);
      PHASE_WAIT_DONE(NPU_WAIT_EXIT_NAP);
      return 0;
    }
    if (npu_launch_status_failed(&g_launch_status)) {
      NPU_ADAPT_AFTER(3u);
      PHASE_WAIT_DONE(NPU_WAIT_EXIT_GIVE_UP);
      return 1;
    }
    qurt_timer_sleep(NPU_SYNC_NAP_MICROS);
#if SKYE_NPU_PHASE_PROBE
    wait_naps++;
#endif
  }

  npu_launch_status_fail(&g_launch_status);
  NPU_ADAPT_AFTER(3u);
  PHASE_WAIT_DONE(NPU_WAIT_EXIT_GIVE_UP);
  return 1;
#endif
}
#undef PHASE_WAIT_DONE
#undef NPU_ADAPT_AFTER

// Marks this launch failed from outside the rendezvous. The kernel's own unwind
// calls it, so the record does not depend on the barrier having been what
// refused: a kernel that gives its grant back and returns early wrote no output
// either way (`_matmul_hmx_unwind`, `npu_kernel.mojo`).
void mojo_npu_launch_fail(void) {
  npu_launch_status_fail(&g_launch_status);
#if SKYE_SYNC_WAKEFUL
  // The nap arm needed nothing here: a peer already inside the rendezvous
  // re-reads the record between naps and gives up on its own. A parked peer
  // cannot, so on this arm the thread that sets the record owes it the wake.
  // The store above is RELEASE and this follows it, so a woken peer reads the
  // failure rather than the generation that will now never advance.
  qshim_sync_wake(&g_launch_sync.data, NPU_SYNC_WAKE_ALL, 0);
#endif
}

// What the trampoline returns instead of the zero a kernel that never ran would
// otherwise report. The caller names `aborted` because the AEE numbering is the
// trampoline's and this file includes none of it.
int32_t mojo_npu_launch_fold_status(int32_t status, int32_t aborted) {
  return npu_launch_status_fold(&g_launch_status, status, aborted);
}

void mojo_npu_launch_begin(uint32_t participants) {
  if (participants == 0 || participants > MOJO_NPU_MAX_LAUNCH_THREADS)
    participants = 1;
  for (uint32_t i = 0; i < MOJO_NPU_MAX_LAUNCH_THREADS; ++i)
    g_launch_threads[i] = QSHIM_UNBOUND_THREAD;
  // The mask names every participant, because the rendezvous reads only its
  // population and refuses a mask naming more threads than the launch spawned.
  g_launch_sync.thread_mask = (int32_t)((1u << participants) - 1u);
  npu_launch_status_open(&g_launch_status);
#if SKYE_NPU_PHASE_PROBE
  // Here rather than in the kernel, for the reason the sync reset is here: this
  // runs before any participant exists, so the clear races nothing and a launch
  // abandoned mid-step cannot leave its buckets in the next one's record.
  npu_phase_reset(&g_phase_report);
#endif
  __atomic_store_n(&g_launch_sync.data, 0, __ATOMIC_RELEASE);
  // Ordered before the count so a participant that starts running early cannot
  // read a count that admits slots this has yet to clear.
  __atomic_store_n(&g_launch_count, participants, __ATOMIC_RELEASE);
}

void mojo_npu_launch_bind(uint32_t index) {
  if (index < MOJO_NPU_MAX_LAUNCH_THREADS)
    __atomic_store_n(&g_launch_threads[index], qurt_thread_get_id(),
                     __ATOMIC_RELEASE);
}

void mojo_npu_launch_end(void) {
  __atomic_store_n(&g_launch_count, 1u, __ATOMIC_RELEASE);
  for (uint32_t i = 0; i < MOJO_NPU_MAX_LAUNCH_THREADS; ++i)
    g_launch_threads[i] = QSHIM_UNBOUND_THREAD;
}

// Which participant the calling thread is. Answering 0 for a thread that bound
// nothing is deliberate rather than an error: the typed entry point calls its
// kernel without opening a launch at all, and a lone invocation is participant
// 0 by definition.
static uint32_t launch_index(void) {
  qurt_thread_t self = qurt_thread_get_id();
  uint32_t count = __atomic_load_n(&g_launch_count, __ATOMIC_ACQUIRE);
  for (uint32_t i = 0; i < count; ++i)
    if (__atomic_load_n(&g_launch_threads[i], __ATOMIC_ACQUIRE) == self)
      return i;
  return 0;
}

// ===--------------------------------------------------------------------=== //
// Device queries
// ===--------------------------------------------------------------------=== //

uint32_t qshimQuery(uint32_t attr, int64_t *out) {
  if (out == NULL)
    return 1;

  switch (attr) {
  case QSHIM_QUERY_VTCM_SIZE:
    if (vtcm_acquire() != 0)
      return 1;
    *out = (int64_t)g_vtcm_size;
    return 0;

  // A kernel's thread index is its position within the launch, 0-based, which
  // is what a kernel guarding on `thread_idx == 0` tests and what indexes the
  // per-thread DMA descriptor below. Deliberately not `qurt_thread_get_id()`,
  // which is unrelated and large enough that such a kernel silently does
  // nothing.
  //
  // A launch index rather than a hardware thread number, which is the one place
  // this transport cannot match the discrete part: participants here are QuRT
  // threads the trampoline created, and which hardware thread each is scheduled
  // onto is QuRT's to decide, not the trampoline's. A kernel that means "the
  // part's DMA thread" by index 4 gets the participant numbered 4 instead.
  case QSHIM_QUERY_THREAD_ID:
  case QSHIM_QUERY_HMX_THREAD_ID:
    *out = (int64_t)launch_index();
    return 0;

  // Participants in the launch, not hardware threads on the part, for the same
  // reason the core count below reports one: a kernel that splits its range by
  // this would leave part of the range to a thread the launch never spawned.
  case QSHIM_QUERY_NUM_THREADS:
    *out = (int64_t)__atomic_load_n(&g_launch_count, __ATOMIC_ACQUIRE);
    return 0;

  // A property of the part rather than of the launch, so this one is the
  // hardware figure.
  case QSHIM_QUERY_NUM_HVX_UNITS: {
    qurt_sysenv_max_hthreads_t hw;
    if (qurt_sysenv_get_max_hw_threads(&hw) != QURT_EOK)
      return 1;
    *out = (int64_t)hw.max_hthreads;
    return 0;
  }

  // A FastRPC session addresses one NSP, unlike the discrete part where a
  // kernel is dispatched across many. Reporting one core keeps work-splitting
  // logic that divides by the core count correct rather than idling most of
  // the range it computes.
  case QSHIM_QUERY_CORE_ID:
    *out = 0;
    return 0;
  case QSHIM_QUERY_NUM_CORES:
    *out = 1;
    return 0;

  default:
    return 1;
  }
}

// ===--------------------------------------------------------------------=== //
// Linear uDMA
// ===--------------------------------------------------------------------=== //

// One descriptor slot per participant, selected by the caller's launch index.
// The DMA engine is per hardware thread, so descriptors must not be shared
// across concurrent submitters, and a descriptor must stay resident until its
// transfer completes. Sized by the launch bound so a valid launch index is
// always a valid slot.
//
// P5 M4.3 order 102 -- THE INVARIANT ABOVE WAS HONOURED AT THE WRONG
// GRANULARITY. A type-0 descriptor is 16 bytes and this array packed them at
// that stride (UDMA_STATUS.md records it as a validated property: "participants
// 1 and 3 use descriptors that are not 32-byte aligned. Both work."), so
// participants 0 and 1 share one cache line and 2 and 3 share the next. The
// hardware's coherence granule is the LINE, not the struct, so "not shared
// across concurrent submitters" was false the moment two feeders submitted at
// once:
//
//   - a submit is a PARTIAL-LINE WRITE. `memset(d, 0, sizeof(*d))` below
//     rewrites 16 of the line's bytes; the other 16 belong to a neighbour whose
//     transfer may be in flight, and a write-back of that line carries whatever
//     copy of the neighbour's half the writer had -- including a stale
//     `dstate`, and including `src`/`dst`/`length` the engine is still reading,
//     which is exactly the residency the comment above requires;
//   - a poll then reads through the same shared line. `volatile` stops the
//     COMPILER from caching the value and does nothing about the D-cache, so
//     `qshimUDmaAwait` can observe a neighbour-resurrected `dstate` and exit
//     without ever having waited.
//
// That accounts for what the box shows and the alternatives do not: fast
// (no spinning), wrong (bytes not landed), nondeterministic (it depends on the
// interleaving of four feeders), and slightly WORSE under #313 than without it
// (the poll adds read traffic to the shared lines, widening the window) --
// order 102 measured the residue growing 1.31-1.50x rather than shrinking.
// PR #37's five-thread round trip never saw it because nothing computed
// concurrently there and no descriptor was reused back-to-back.
//
// `SKYE_UDMA_DESC_ISOLATE` gives each descriptor a line of its own. It is a
// build define rather than an environment variable because this is device C in
// a signed skel: there is no runtime channel into it that does not change the
// frozen ABI. Two skels differing by this one define are the A/B, which is the
// shape #311 and #313 already used.
//
// 128 is an upper bound over both the 32-byte L1 line and any plausible L2
// line, and it is the alignment this file already uses for a per-thread object
// (`g_fresh_thread_stack`). A smaller number would be a guess; the cost of this
// one is 640 bytes of .bss.
// P5 M4.3 order 103, ON by default since the default wave (order 104's leg
// measured it; `build.sh` always passes an explicit -D, so this fallback is
// what a compile OUTSIDE build.sh gets and the two must agree).
#ifndef SKYE_UDMA_VTCM_BYPASS
#define SKYE_UDMA_VTCM_BYPASS 1
#endif

#if SKYE_UDMA_VTCM_BYPASS
// The reference asks its queue, which carries the window it was built with
// (`dma_is_vtcm`, htp/dma-queue.h:150-152). We ask the shim's own VTCM grant,
// which is the same question against the only window this process has. A NULL
// or unacquired grant answers "not VTCM", which leaves the transfer exactly as
// it is today rather than guessing.
static int qshim_addr_is_vtcm(const void *p) {
  uintptr_t a = (uintptr_t)p;
  uintptr_t base = (uintptr_t)g_vtcm_base;
  if (base == 0 || g_vtcm_size == 0)
    return 0;
  return a >= base && a < base + (uintptr_t)g_vtcm_size;
}
#endif

#ifndef SKYE_UDMA_DESC_ISOLATE
#define SKYE_UDMA_DESC_ISOLATE 0
#endif

#if SKYE_UDMA_DESC_ISOLATE
#define QSHIM_UDMA_DESC_STRIDE 128
#else
#define QSHIM_UDMA_DESC_STRIDE ((int)sizeof(hexagon_udma_descriptor_type0_t))
#endif

typedef union {
  hexagon_udma_descriptor_type0_t desc;
  char pad[QSHIM_UDMA_DESC_STRIDE];
} QShimUDmaSlot;

// The descriptor ring, P5 M4.3 (UDMA_RING_DESIGN.md). Capacity is the
// reference's production configuration (llama.cpp htp/main.c:529); the
// state machine (npu_udma_ring.h) is host-proven and enforces power-of-two
// at init. Storage is DDR .bss: 8 x 256 x 16 B = 32 KiB at the default
// stride. 128-byte alignment keeps one thread's ring off every other
// thread's cache lines -- the cross-thread half of the order-102 class is
// structural, not probabilistic. Within a ring, packed slots are the
// reference's own shipping arrangement (order 101: bit-stable 218/218 on
// this box); SKYE_UDMA_DESC_ISOLATE still composes as the A/B if the
// resurrection class ever shows inside a ring, and an isolate build
// should lower SKYE_UDMA_RING_CAPACITY alongside (128-B slots x 256 x 8
// threads would be 256 KiB).
#ifndef SKYE_UDMA_RING_CAPACITY
#define SKYE_UDMA_RING_CAPACITY 256
#endif
typedef char qshim_udma_ring_capacity_is_pow2
    [(SKYE_UDMA_RING_CAPACITY >= 2 &&
      SKYE_UDMA_RING_CAPACITY <= NPU_UDMA_RING_MAX_CAPACITY &&
      (SKYE_UDMA_RING_CAPACITY & (SKYE_UDMA_RING_CAPACITY - 1)) == 0)
         ? 1
         : -1];

static QShimUDmaSlot g_ring_slots[MOJO_NPU_MAX_LAUNCH_THREADS]
                                 [SKYE_UDMA_RING_CAPACITY]
    __attribute__((aligned(128)));
static NpuUdmaRing g_udma_rings[MOJO_NPU_MAX_LAUNCH_THREADS];
static int g_udma_ring_armed[MOJO_NPU_MAX_LAUNCH_THREADS];

#define g_ring_desc_at(t, s) (&g_ring_slots[(t)][(s)].desc)

// Defined below, referenced by qshimUDmaWait first: file order keeps the
// old entry points at their lines for review, so the flush is declared
// here rather than moved above them.
uint32_t qshimUDmaFlush(uint32_t threadID);

// Lazy per-thread arming. Only the owner thread ever touches its ring
// (UDMA_RING_DESIGN.md section 3.2 ownership), so the armed flag needs no
// atomicity.
static NpuUdmaRing *qshim_udma_ring(uint32_t threadID) {
  NpuUdmaRing *r = &g_udma_rings[threadID];
  if (!g_udma_ring_armed[threadID]) {
    (void)npu_udma_ring_init(r, SKYE_UDMA_RING_CAPACITY);
    g_udma_ring_armed[threadID] = 1;
  }
  return r;
}

// Counters only, never the data path. Read once at session teardown; a torn
// cross-word read costs nothing because these are magnitudes, not a protocol.
static volatile NpuUdmaReport g_udma_report;
static int g_udma_report_armed;

uint32_t qshimUDmaWait(uint32_t handle) {
  // `dmwait` drains every transfer this thread started, which is coarser than
  // waiting on one handle. Conservative in the safe direction: it can only wait
  // longer than asked, never return before the named transfer landed.
  __builtin_HEXAGON_Y6_dmwait();
  // Under the ring the engine may now be idle while the bookkeeping still
  // holds the retired descriptors; reconcile so a later submit's full test
  // and the teardown census stay truthful. An invalid handle (a caller
  // that submitted with requireHandle=0) degrades to today's bare dmwait.
  if (handle < MOJO_NPU_MAX_LAUNCH_THREADS)
    return qshimUDmaFlush(handle);
  return 0;
}

// Bounded, per order 084's rule that a device kernel never waits forever. The
// bound is a wedged-engine escape, not a tuning knob: at any plausible clock it
// is far longer than the largest transfer this path issues, so exhausting it
// means the engine stopped, and returning is no worse than hanging the domain.
#define QSHIM_UDMA_COMPLETE_SPINS 0x01000000u

// Pops the ring's oldest descriptor: the FIFO wait the reference's
// dma_queue_pop performs (dma-queue.h:239-262), on our type-0 completion
// record. The authoritative check is the HEAD descriptor's own `dstate` --
// per-transfer, resident in memory, so it survives a QuRT migration (the
// order-100 lesson, kept in full: `dmwait` drains a HARDWARE thread while a
// participant is a SOFTWARE thread QuRT places where it likes, so only the
// memory-resident record is trusted, read through a volatile pointer the
// compiler cannot hoist). `dstate` is spelled as "no longer INCOMPLETE"
// rather than "== COMPLETE" so this uses only the enumerator the submit path
// below already proves is spelled that way in the SDK's `udma.h` -- this
// file cannot be compiled on the Mac seat, so a guessed second name would
// surface as a window build break; it also treats an error state as a stop
// rather than a spin. Between reads the drain instruction is `dmpoll`,
// NEVER `dmwait`: dmwait drains this hardware thread's WHOLE chain, which
// under a ring would wait on descriptors pushed after the head -- depth-1
// semantics quietly reinstated at the await point. A post-migration
// `dmpoll` against the wrong engine is harmless, because the exit condition
// is the record, not the engine. Bounded by the same wedged-engine escape
// as before (order 084's rule); the caller decides what a bound exhaustion
// poisons, and the descriptor deliberately stays in the census on a wedge
// so the teardown report names it.
static uint32_t qshim_udma_pop_one(uint32_t threadID, NpuUdmaRing *r) {
  uint32_t head = npu_udma_ring_pop_head(r);
  if (head == NPU_UDMA_RING_NO_SLOT)
    return 0;
  if (npu_udma_ring_head_is_dummy(r)) {
    // Never armed, never linked: nothing will ever write its record.
    (void)npu_udma_ring_pop_commit(r);
    return 0;
  }
  volatile hexagon_udma_descriptor_type0_t *d = g_ring_desc_at(threadID, head);
  for (uint32_t spins = 0; spins < QSHIM_UDMA_COMPLETE_SPINS; ++spins) {
    if (d->dstate != HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE) {
      // `spins == 0` means the FIRST read already found it clear, which is the
      // order-102 discriminator: a popped head whose first read is already
      // clear when the pipeline should have it in flight is a read that never
      // saw the engine.
      npu_udma_report_note_await((NpuUdmaReport *)&g_udma_report, spins,
                                 /*hit_bound=*/0);
      (void)npu_udma_ring_pop_commit(r);
      return 0;
    }
    (void)__builtin_HEXAGON_Y6_dmpoll();
  }
  npu_udma_report_note_await((NpuUdmaReport *)&g_udma_report,
                             QSHIM_UDMA_COMPLETE_SPINS, /*hit_bound=*/1);
  return 1;
}

uint32_t qshimUDmaAwait(uint32_t threadID) {
  if (threadID >= MOJO_NPU_MAX_LAUNCH_THREADS)
    return 1;
  // Pop-oldest. At the shipped kernel's one-in-flight usage this is the
  // same descriptor the old code polled, minus the leading whole-chain
  // dmwait (UDMA_RING_DESIGN.md section 5 carries the argument); an await
  // with nothing outstanding returns 0, as the idle-engine dmwait did.
  return qshim_udma_pop_one(threadID, qshim_udma_ring(threadID));
}

// Drains everything this thread has in flight. The drain-before-exit
// entry point: a deep consumer's epilogue calls this before its final
// barrier so no descriptor outlives the launch that submitted it
// (UDMA_RING_DESIGN.md section 6 -- the EBADSTATE class: an abandoned
// in-flight descriptor whose destination memory is released turns into a
// precise DMA error no user PD can read back, reported as AEE_EBADSTATE
// on a LATER invoke with nothing pointing at DMA).
uint32_t qshimUDmaFlush(uint32_t threadID) {
  if (threadID >= MOJO_NPU_MAX_LAUNCH_THREADS)
    return 1;
  NpuUdmaRing *r = qshim_udma_ring(threadID);
  while (!npu_udma_ring_empty(r)) {
    if (qshim_udma_pop_one(threadID, r) != 0)
      return 1;
  }
  return 0;
}

// The one line order 102 asked for, emitted once per session at teardown.
//
// Routed the way #295 routes the queue's import line: the string is PRE-BUILT
// and handed to the resident debug entry, because the FARF macro's inline
// vsnprintf fallback drags a heap/lock/TLS layer this protection domain does
// not provide and the skel then refuses to load. Output-neutral by
// construction -- it reads counters and writes no buffer the graph can see --
// and on a control build nothing ever calls the await, so every counter is
// zero and the line still prints, which is what makes its ABSENCE meaningful.
void qshimUDmaReport(void) {
  char message[NPU_UDMA_REPORT_MSG_BYTES];
  npu_udma_report_message(message, (int)sizeof(message),
                          (const NpuUdmaReport *)&g_udma_report,
                          SKYE_UDMA_DESC_ISOLATE);
  if (0 == (HAP_debug_v2))
    HAP_debug(message, FARF_ALWAYS_LEVEL, __FILE__, __LINE__);
  else
    HAP_debug_v2(FARF_ALWAYS_LEVEL, __FILE__, __LINE__, "%s", message);

  // One ADDITIVE census line beside the pinned await line (whose shape is a
  // contract and is not edited). `inflight` sums the armed rings' depths at
  // teardown; nonzero means a launch exited with descriptors in flight --
  // the EBADSTATE precursor (UDMA_RING_DESIGN.md section 6) counted at the
  // desk instead of diagnosed by bisection. peak/full carry the per-ring
  // maxima.
  unsigned ring_peak = 0;
  unsigned ring_full = 0;
  unsigned ring_inflight = 0;
  for (uint32_t t = 0; t < MOJO_NPU_MAX_LAUNCH_THREADS; ++t) {
    if (!g_udma_ring_armed[t])
      continue;
    const NpuUdmaRing *r = &g_udma_rings[t];
    if (r->peak_depth > ring_peak)
      ring_peak = r->peak_depth;
    if (r->full_refusals > ring_full)
      ring_full = r->full_refusals;
    ring_inflight += npu_udma_ring_depth(r);
  }
  npu_udma_report_ring_message(message, (int)sizeof(message),
                               SKYE_UDMA_RING_CAPACITY, ring_peak, ring_full,
                               ring_inflight);
  if (0 == (HAP_debug_v2))
    HAP_debug(message, FARF_ALWAYS_LEVEL, __FILE__, __LINE__);
  else
    HAP_debug_v2(FARF_ALWAYS_LEVEL, __FILE__, __LINE__, "%s", message);
}

QShimUDmaHandle qshimLinearUdmaSubmit(uint32_t threadID, void *src,
                                      uint32_t size, void *dst,
                                      void *udmaDescAttrs, int requireHandle,
                                      uint32_t *status) {
  // Ordering attributes are the caller's way of chaining transfers; honouring
  // them needs a descriptor chain this shim does not build yet, so refuse
  // rather than silently drop the ordering guarantee.
  if (udmaDescAttrs != NULL) {
    if (status != NULL)
      *status = 1;
    return QSHIM_INVALID_UDMA_HANDLE;
  }

  if (threadID >= MOJO_NPU_MAX_LAUNCH_THREADS || src == NULL || dst == NULL) {
    if (status != NULL)
      *status = 1;
    return QSHIM_INVALID_UDMA_HANDLE;
  }

  // `length` is a 24-bit field, so a longer request would be truncated into a
  // short copy that looks like it succeeded.
  if (size > 0x00FFFFFFu) {
    if (status != NULL)
      *status = 1;
    return QSHIM_INVALID_UDMA_HANDLE;
  }

  NpuUdmaRing *r = qshim_udma_ring(threadID);
  uint32_t slot = npu_udma_ring_push_begin(r);
  if (slot == NPU_UDMA_RING_NO_SLOT) {
    // Ring full: the consumer is a whole ring ahead. The oldest transfer
    // is the one it will need first, so waiting on exactly that one here
    // preserves the submit-always-lands contract without widening the
    // ABI. A wedge propagates as a refused submit rather than a hang.
    if (qshim_udma_pop_one(threadID, r) != 0 ||
        (slot = npu_udma_ring_push_begin(r)) == NPU_UDMA_RING_NO_SLOT) {
      if (status != NULL)
        *status = 1;
      return QSHIM_INVALID_UDMA_HANDLE;
    }
  }
  hexagon_udma_descriptor_type0_t *d = g_ring_desc_at(threadID, slot);
  // Zeroed whole, which leaves the L2 bypass bits clear, which is what makes
  // this a CACHED transfer -- and batched dispatch depends on that. A batch
  // runs several launches behind one FastRPC call, so the driver's per-call
  // flush and invalidate happen at the batch's edges and not between its
  // launches; a producer's output reaches its consumer only because both go
  // through the L2 the engine fetches through. Setting either bypass bit here
  // would make that false, and the symptom would be wrong numbers from a batch
  // that returned zero, not a fault. See `npu_run_batch` in mojo_npu_imp.c.
  memset(d, 0, sizeof(*d));
  d->next = NULL;
  d->length = size;
  d->desctype = HEXAGON_UDMA_DESC_DESCTYPE_TYPE0;
  d->order = HEXAGON_UDMA_DESC_ORDER_NOORDER;
  d->dstate = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE;
  d->src = src;
  d->dst = dst;

  // P5 M4.3 order 103 -- THE PAYLOAD ORDERING, taken from the reference rather
  // than reasoned out. `ggml-hexagon` runs HMX with DMA'd weights bitwise
  // stable on this exact box (order 101, 218/218), and the thing it does that
  // we did not is decide the bypass bits PER SIDE, by whether the address is
  // VTCM:
  //
  //     desc->src_bypass = dma_is_vtcm(q, dptr.src) ? 1 : q->nocache;
  //     desc->dst_bypass = dma_is_vtcm(q, dptr.dst) ? 1 : q->nocache;
  //         -- llama.cpp ggml/src/ggml-hexagon/htp/dma-queue.h:169-170
  //         at pinned SHA 687e7789271
  //
  // Ours cleared both, unconditionally, and the comment above explains why for
  // the DDR case: a batch's producer and consumer meet in the L2, so a DDR
  // destination must stay cached. That reasoning is right and is preserved
  // here -- it just never covered a VTCM destination. With `dst_bypass` clear
  // the engine's writes to VTCM go THROUGH the L2, so `dstate` can publish
  // while the payload is still posted in the fabric, and whether the array's
  // `mxmem` stream sees the bytes depends on cache pressure that varies run to
  // run. That is unstable-and-wrong, which is exactly what orders 100/102/103
  // measured, and it is the half neither #313 (ordered the flag) nor #315
  // (unshared the flag's line) touched.
  //
  // The rule is the reference's, not a variant of it: bypass IFF the address
  // is VTCM, each side independently. A DDR-to-DDR transfer is unchanged, so
  // batched dispatch keeps the property it depends on.
  //
  // Field names read from the SDK's own `udma.h` (Tools 19.0.07,
  // `libnative/include/udma.h:25-26`), not inferred: the type-0 descriptor is
  // `next` + one 32-bit bitfield word (`length:24, desctype:2, dstcomp:1,
  // srccomp:1, dstbypass:1, srcbypass:1, order:1, dstate:1`) + `src` + `dst`,
  // which is also where #315's 16-byte stride arithmetic comes from.
#if SKYE_UDMA_VTCM_BYPASS
  d->srcbypass = qshim_addr_is_vtcm(src) ? 1u : 0u;
  d->dstbypass = qshim_addr_is_vtcm(dst) ? 1u : 0u;
#endif

  uint32_t link_from = npu_udma_ring_push_commit(r, /*is_dummy=*/0);
  // The engine reads the descriptor from memory, so the writes above must not
  // be sunk past the start. A compiler barrier is the whole of what that needs:
  // stores retire in order into the L2 the engine fetches through, and
  // `dmlink` is a solo instruction the compiler cannot pack ahead of them.
  //
  // Nothing stronger, and in particular not `dmsyncht`, which is what used to
  // take the protection domain down here with AEE_EBADSTATE. `dmstart`,
  // `dmlink`, `dmpoll`, `dmwait`, `dmpause` and `dmresume` are the user-mode
  // half of this instruction class; `dmsyncht` is not, which is why QuRT
  // mediates it through `qurt_user_dma_dmsyncht` and why neither
  // `hexagon_protos.h` nor upstream LLVM carries it while all six above appear
  // in both.
  __asm__ __volatile__("" ::: "memory");
  // The reference publishes the descriptor with a real ordering instruction
  // rather than a compiler barrier alone:
  //
  //     asm volatile(" release(%0):at" : : "r"(next));
  //         -- htp/dma-queue.h:122-130, issued before BOTH dmstart and dmlink
  //
  // The release no longer rides the bypass define: under a chain the
  // publish is unconditionally load-bearing -- the ENGINE follows `next`
  // pointers through memory -- so the old conditional, an idiom borrowed
  // alongside the bypass, becomes structural here.
  __asm__ __volatile__(" release(%0):at" ::"r"(d) : "memory");
  // dmlink both appends to a live chain and starts an idle engine, which
  // is why the ring path has no dmstart at all -- the reference's own
  // shape (no dmstart appears in any dma-queue.h push path; the first
  // link lands on the zeroed tail slot the machine names). Operand order
  // is (current tail, new descriptor), the reference's `dmlink(%0, %1)`.
  __builtin_HEXAGON_Y6_dmlink(g_ring_desc_at(threadID, link_from), d);

  if (!g_udma_report_armed) {
    npu_udma_report_reset((NpuUdmaReport *)&g_udma_report);
    g_udma_report_armed = 1;
  }
  g_udma_report.submitted++;

  if (status != NULL)
    *status = 0;
  return requireHandle ? threadID : QSHIM_INVALID_UDMA_HANDLE;
}

// Argument-ABI trampoline. The Mojo `external_call` lowering does not place a
// 64-bit argument in the register pair the C ABI expects on this 32-bit target,
// so the stdlib overlay passes addresses as 32-bit values and calls this, which
// widens them using the local ABI. Mirrors the trampoline the discrete path
// injects at object-compile time; here it is linked in directly because that
// path is not used.
QShimUDmaHandle skye_udma_submit_linear_lo32(uint32_t threadID, uint32_t src,
                                             uint32_t size, uint32_t dst,
                                             void *udmaDescAttrs,
                                             int requireHandle,
                                             uint32_t *status) {
  return qshimLinearUdmaSubmit(threadID, (void *)(uintptr_t)src, size,
                               (void *)(uintptr_t)dst, udmaDescAttrs,
                               requireHandle, status);
}

// ===--------------------------------------------------------------------=== //
// Thread barrier
// ===--------------------------------------------------------------------=== //

// Arrival barrier over the threads named in `thread_mask`, using the struct's
// own `data` word as the only scratch the ABI allows. `data` packs an arrival
// count in the low 16 bits and a generation in the high 16, so the struct is
// reusable across successive barriers without a separate reset step: the last
// thread to arrive bumps the generation, which is what releases the waiters.
//
// Waiting blocks on the word rather than spinning on it. A spinning participant
// holds a hardware thread that a peer it is waiting for may need to issue from,
// and QuRT does not time-slice threads of equal priority, so a spin can deny
// the very arrival that would end it.
uint32_t qshimSyncThreads(uint32_t thread_id, QShimThreadSyncStruct *sync) {
  (void)thread_id;
  if (sync == NULL)
    return 1;

  const uint32_t participants =
      (uint32_t)__builtin_popcount((uint32_t)sync->thread_mask);
  if (participants <= 1)
    return 0;

  // A kernel naming more threads than the launch spawned is refused here rather
  // than waited on: the arrivals it is short of are never coming, and a
  // rendezvous that cannot complete takes the whole invocation with it.
  if (participants > __atomic_load_n(&g_launch_count, __ATOMIC_ACQUIRE))
    return 1;

  uint32_t prev =
      (uint32_t)__atomic_fetch_add(&sync->data, 1, __ATOMIC_ACQ_REL);
  uint32_t generation = prev >> 16;
  uint32_t arrived = (prev & 0xFFFFu) + 1;

  if (arrived == participants) {
    // Last in: open the gate and clear the count in one store, so no thread can
    // observe the next generation with a stale arrival count.
    __atomic_store_n(&sync->data, (int32_t)((generation + 1) << 16),
                     __ATOMIC_RELEASE);
    qurt_futex_wake(&sync->data, (int)(participants - 1));
    return 0;
  }

  for (;;) {
    int observed = __atomic_load_n(&sync->data, __ATOMIC_ACQUIRE);
    if (((uint32_t)observed >> 16) != generation)
      return 0;
    // Re-reading the word immediately before waiting on that same value is what
    // closes the gap: should the gate open in between, the wait finds the word
    // changed and returns rather than sleeping through the wake.
    qurt_futex_wait(&sync->data, observed);
  }
}

// Mojo's `round()` lowers to `llvm.roundeven.f32`, and this toolchain has no
// runtime for it: the v81 `G0/pic` libc.a defines roundf/rintf/nearbyintf, and
// no archive shipped with SDK 6.6.0.0 defines roundeven* at all (libm.a is a
// 728-byte stub). So the W8A8 arm's four `round()` calls in `hmx_int8.mojo`
// leave `roundevenf` undefined, and the skel's symbol gate refuses the link --
// which is what an unresolved import deserves, since on device it would surface
// as AEE_EUNABLETOLOAD and read as a signing fault.
//
// Open-coded rather than delegated to libc's `rintf`, which IS defined in that
// archive: pulling it in drags the whole C runtime behind it (sys_Mtxinit,
// sys_Tlsalloc, sys_sbrk and five more undefined imports), because the object
// carrying rintf also carries locale-aware machinery this skel must not have.
// The bit-manipulation form below has no dependencies and, unlike rint, is
// independent of the FP rounding mode rather than merely agreeing with it in
// the default one.
//
// Verified exhaustively against numpy's ties-to-even `rint` over every float32
// ulp in [0, 2^23], both signs (16777218 values, 0 mismatches), plus the
// tie/zero-sign/inf/NaN edges -- see ~/npu-lane/124-int8/roundeven_check.py.
// Compared as BIT PATTERNS, so the sign of zero is constrained too.
float roundevenf(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  const uint32_t sign = v.u & 0x80000000u;
  const uint32_t ax = v.u & 0x7fffffffu;
  const int exp = (int)(ax >> 23) - 127;

  if (exp >= 23) // already integral, or Inf/NaN: return unchanged
    return x;
  if (exp < -1) { // |x| < 0.5 rounds to a zero of the same sign
    v.u = sign;
    return v.f;
  }
  if (exp == -1) { // 0.5 <= |x| < 1: the tie goes to even, i.e. to zero
    v.u = (ax == 0x3f000000u) ? sign : (sign | 0x3f800000u);
    return v.f;
  }

  // 1 <= |x| < 2^23: the low (23 - exp) mantissa bits are the fraction, and the
  // integer's parity is the bit just above them. A carry out of the mantissa
  // lands in the exponent field, which is what IEEE-754 wants.
  const uint32_t fbits = (uint32_t)(23 - exp);
  const uint32_t frac_mask = (1u << fbits) - 1u;
  const uint32_t frac = ax & frac_mask;
  const uint32_t half = 1u << (fbits - 1u);
  uint32_t kept = ax & ~frac_mask;
  if (frac > half || (frac == half && (kept & (1u << fbits))))
    kept += (1u << fbits);
  v.u = sign | kept;
  return v.f;
}
