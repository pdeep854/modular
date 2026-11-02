"""The staging protocol's falsification test: banks, epochs and ownership.

`FEED_ARCHITECTURE.md` sec 10.2 specifies the protocol slices 3b and 4 need — a
staging line's owner, how many banks it has, and when a bank is free — and
discharges its safety argument on paper. Two independent critiques of that
document found the same joint from different sides, so a paper argument is
exactly what this file replaces: the schedule is executed here, against the two
defects those critiques named, and each defect is a MUTATION that must fail.

WHAT THIS PROVES, and the boundary is the point: it is a statement about the
SCHEDULE, not about any kernel. No staging code exists yet — that is slices 3b
and 4 — so this models the step loop those slices will run and asserts the
invariants that make their DMA safe. A protocol that fails here cannot be made
safe by writing it carefully in Mojo; a protocol that passes here still has to
be implemented to match, which is what the twin's byte-identity gate is for.

THE TWO INVARIANTS, stated the way a reader can check them against the design:

  I1  NO OVERWRITE BEFORE COMPLETION. A staging cell is written only while its
      bank is not in flight. With one outstanding descriptor per submitter a
      bank's life is fill -> arm -> drain -> await -> free, which spans two
      steps, so a single bank makes the producer of step t+1 race the transfer
      of step t.

  I2  NO REUSE BEFORE ACKNOWLEDGEMENT. A bank is re-armed only after its
      previous transfer was awaited, and every cell a step produces is drained
      exactly once. A line sized per FEEDER rather than per PANEL breaks this
      without breaking I1: the feeder overwrites its own line as it walks its
      panel share inside one step, so output is lost before any transfer sees
      it, and no bank count fixes that.

I1 is the defect one critique named; I2 is the one the other named and the first
would have missed. Both mutations are below, and both must fail.
"""

from std.collections import List

from hexagon_ondevice.matmul_kernel import hmx_work_share


comptime FEEDERS = 4
"""Threads that unpack. The array thread arms descriptors and is not one."""

comptime BANKS = 2
"""Banks the protocol calls for: the derived answer, not a chosen one.

A bank's lifetime spans two steps under one outstanding descriptor per
submitter, so two is the least depth at which the bank being filled and the
bank in flight are provably different. `BANKS_MUTANT` is the falsifying arm."""

comptime BANKS_MUTANT = 1
"""Single-buffered staging: the shape one critique attacked. Must fail I1."""


@fieldwise_init
struct StageCell(Copyable, Movable):
    """One staging line's state, stamped so a race is observable.

    `generation` is which step's output the cell holds and `drained` whether a
    transfer has consumed it. Stamping rather than counting is what lets the
    assertions name the step that collided instead of reporting a total.
    """

    var generation: Int
    var drained: Bool


@fieldwise_init
struct StageModel(Copyable, Movable):
    """The staging arena as the protocol describes it, and nothing more.

    `lines_per_bank` is `act_slots` under the specified sizing — one line per
    PANEL — and `FEEDERS` under the mutation that sizes it per feeder.
    """

    var banks: Int
    var lines_per_bank: Int
    var cells: List[StageCell]
    var in_flight: List[Int]
    """Generation each bank has a transfer posted for, or -1 for none."""

    var lost: Int
    """Cells overwritten while still holding an undrained generation."""

    var raced: Int
    """Cells written while their own bank had a transfer in flight."""


def stage_model(banks: Int, lines_per_bank: Int) -> StageModel:
    """A model with every cell empty and no transfer posted."""
    var cells = List[StageCell]()
    for _ in range(banks * lines_per_bank):
        cells.append(StageCell(generation=-1, drained=True))
    var flight = List[Int]()
    for _ in range(banks):
        flight.append(-1)
    return StageModel(
        banks=banks,
        lines_per_bank=lines_per_bank,
        cells=cells^,
        in_flight=flight^,
        lost=0,
        raced=0,
    )


def feeder_fill(
    mut model: StageModel, step: Int, feeder: Int, panels: Int, line: Int
):
    """One feeder writes one line of the bank step `step` fills.

    `line` is the cell this feeder's output lands in, and it is the whole
    difference between the specified sizing and the mutation: per PANEL the
    feeder's panels land in distinct cells, per FEEDER they land in one.
    """
    var bank = step % model.banks
    var idx = bank * model.lines_per_bank + line

    # I1: the bank must not have a transfer posted while a feeder writes it.
    if model.in_flight[bank] >= 0:
        model.raced += 1

    # I2: whatever was here must already have been drained.
    if not model.cells[idx].drained:
        model.lost += 1

    model.cells[idx] = StageCell(generation=step, drained=False)
    _ = feeder
    _ = panels


def array_arm(mut model: StageModel, step: Int):
    """The array thread posts the previous step's bank. No-op at step 0."""
    if step <= 0:
        return
    var bank = (step - 1) % model.banks
    model.in_flight[bank] = step - 1


def array_await(mut model: StageModel, step: Int) -> Int:
    """The array waits its posted transfer AFTER its MMA chain (skye 3.1).

    Returns the cells drained, so a caller can assert the step's whole output
    left rather than assuming it did.
    """
    if step <= 0:
        return 0
    var bank = (step - 1) % model.banks
    if model.in_flight[bank] < 0:
        return 0
    var generation = model.in_flight[bank]
    var drained = 0
    for line in range(model.lines_per_bank):
        var idx = bank * model.lines_per_bank + line
        if model.cells[idx].generation == generation:
            if not model.cells[idx].drained:
                model.cells[idx] = StageCell(
                    generation=generation, drained=True
                )
                drained += 1
    model.in_flight[bank] = -1
    return drained


def run_schedule[
    array_first: Bool
](banks: Int, lines_per_bank: Int, act_slots: Int, steps: Int) -> StageModel:
    """The stripe's step loop with staging, in both intra-step orders.

    `array_first` is the interleaving control: the end-of-step rendezvous orders
    steps and says NOTHING about who moves first inside one, so a protocol that
    is safe only under one order is not safe. Both orders run the same
    assertions.
    """
    var model = stage_model(banks, lines_per_bank)
    for step in range(steps):
        comptime if array_first:
            array_arm(model, step)

        for feeder in range(FEEDERS):
            var q0 = hmx_work_share(act_slots, feeder, FEEDERS)
            var q1 = hmx_work_share(act_slots, feeder + 1, FEEDERS)
            for panel in range(q0, q1):
                # Per PANEL when the arena is sized as specified; the mutation
                # collapses every panel of a feeder onto that feeder's one line.
                var line = panel if lines_per_bank >= act_slots else feeder
                feeder_fill(model, step, feeder, q1 - q0, line)

        comptime if not array_first:
            array_arm(model, step)

        _ = array_await(model, step)
    return model^


def check_protocol[array_first: Bool](act_slots: Int, steps: Int) -> Bool:
    """The specified protocol holds both invariants, in this interleaving."""
    var model = run_schedule[array_first](BANKS, act_slots, act_slots, steps)
    var ok = True
    if model.raced != 0:
        print("  FAIL I1 raced", model.raced, "act_slots", act_slots)
        ok = False
    if model.lost != 0:
        print("  FAIL I2 lost", model.lost, "act_slots", act_slots)
        ok = False
    if ok:
        print(
            "  ok   protocol act_slots=",
            act_slots,
            "array_first=",
            array_first,
        )
    return ok


def check_single_bank_fails(act_slots: Int, steps: Int) -> Bool:
    """MUTATION 1 — one bank. Must break I1, and the test fails if it does not.

    This is the control that makes `BANKS = 2` a measured requirement instead of
    a described one, and it is the defect both critiques named.

    RUN IN BOTH INTERLEAVINGS, and the reason is the protocol's own argument
    rather than thoroughness. `run_schedule`'s docstring says the end-of-step
    rendezvous orders steps and says NOTHING about who moves first inside one,
    so a protocol is only safe if it is safe under both orders — and a MUTATION
    is only refuted if it is safe under both orders too. Under `array_first =
    False` a single bank does not race at all: `array_await` clears the bank's
    in-flight generation at the end of every step, so the next step's feeders
    always find it idle. Asserting the mutation under that order alone therefore
    reports "one bank is fine", which is how this test first ran and is exactly
    the false negative a falsification test cannot afford.

    The asymmetry is a finding and is printed rather than hidden: **one bank is
    safe only if the array never arms before the feeders fill within a step.**
    Nothing in §10.2 fixes that order, so two banks are required precisely
    because the intra-step order is not ours to assume.
    """
    var early = run_schedule[True](BANKS_MUTANT, act_slots, act_slots, steps)
    var late = run_schedule[False](BANKS_MUTANT, act_slots, act_slots, steps)
    if early.raced == 0 and late.raced == 0:
        print("  FAIL mutation 1 (one bank) did not race in EITHER order")
        return False
    print(
        "  ok   mutation 1 fires: raced",
        early.raced,
        "(array first) /",
        late.raced,
        "(feeders first, safe only because await clears the bank each step)",
    )
    return True


def check_per_feeder_line_fails(act_slots: Int, steps: Int) -> Bool:
    """MUTATION 2 — one line per FEEDER instead of per PANEL. Must break I2.

    The defect a bank count does not fix: a feeder walks its whole panel share
    inside one step, so with one line it overwrites its own undrained output
    before any transfer runs. Two banks and both interleavings leave it intact,
    which is precisely why the two critiques' findings were not the same fix.
    """
    var model = run_schedule[False](BANKS, FEEDERS, act_slots, steps)
    if model.lost == 0:
        print("  FAIL mutation 2 (line per feeder) lost nothing")
        return False
    print("  ok   mutation 2 fires: lost", model.lost)
    return True


def main() raises:
    print("==== HMX staging protocol (banks, epochs, ownership) ====")
    var failures = 0

    # The six named geometries' panel shares, plus the 38-panel one the
    # critique asked for by name.
    var shapes: List[Int] = [8, 11, 32, 36, 38]
    for i in range(len(shapes)):
        if not check_protocol[False](shapes[i], 8):
            failures += 1
        if not check_protocol[True](shapes[i], 8):
            failures += 1

    if not check_single_bank_fails(38, 8):
        failures += 1
    if not check_per_feeder_line_fails(38, 8):
        failures += 1

    if failures != 0:
        print("FAILED", failures)
        raise Error("staging protocol checks failed")
    print("PASSED")
