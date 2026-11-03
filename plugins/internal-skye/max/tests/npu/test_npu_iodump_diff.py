# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The IO-dump differ names the first divergent dispatch, correctly classified.

Fixtures are records in the recorder's own schema (field-for-field with a
real dump), so a schema drift between the recorder and the differ fails here
rather than on a device day. Each divergence class the differ distinguishes
gets a fixture that differs in exactly that way -- and a fixture pair that
differs in nothing must come back clean, or the tool cries wolf.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(os.environ["NPU_IODUMP_DIFF"]).resolve()


def _launch(
    seq: int,
    name: str,
    sel: int,
    bufs: list[dict[str, object]],
    status: int = 0,
    scalars: list[str] | None = None,
) -> dict[str, object]:
    return {
        "k": "launch",
        "seq": seq,
        "name": name,
        "sel": sel,
        "mask": 1,
        "status": status,
        "scalars": scalars or ["0x400"],
        "bufs": bufs,
    }


def _slot(
    slot: int,
    alloc: int,
    off: int,
    pre: str,
    post: str,
    asz: int = 1048576,
    n: int = 512,
    pre_zero: int = 0,
    post_zero: int = 0,
) -> dict[str, object]:
    return {
        "slot": slot,
        "alloc": alloc,
        "base": "0xf00d0000",
        "asz": asz,
        "off": off,
        "n": n,
        "pre": pre,
        "preZ": pre_zero,
        "post": post,
        "postZ": post_zero,
    }


def _copy(seq: int, dst_alloc: int, dst_off: int = 0) -> dict[str, object]:
    return {
        "k": "copy_to",
        "seq": seq,
        "n": 1048576,
        "dst": "0xf00d0000",
        "dstAlloc": dst_alloc,
        "dstAsz": 1048576,
        "dstOff": dst_off,
        "src": "0xbeef0000",
        "hashed": 512,
        "srcHash": "0xaaa",
        "dstPre": "0x0",
        "dstPost": "0xaaa",
    }


def _base_run() -> list[dict[str, object]]:
    """A staging copy, then a producer/consumer pair sharing allocation 0 --
    the whole-graph chain's shape in miniature."""
    return [
        {
            "k": "meta",
            "v": 1,
            "pid": 1,
            "window": 512,
            "bytes": 0,
            "byteCap": 1,
        },
        _copy(0, dst_alloc=0),
        _launch(
            1,
            "matmul",
            24,
            [
                _slot(0, 0, 0, "0xaaa", "0xaaa"),
                _slot(1, 0, 98304, "0xbbb", "0xccc"),
            ],
        ),
        _launch(
            2,
            "rms_norm",
            5,
            [
                _slot(0, 0, 98304, "0xccc", "0xccc"),
                _slot(1, 0, 196608, "0x111", "0x222"),
            ],
        ),
    ]


def _write(tmp_path: Path, name: str, records: list[dict[str, object]]) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _run(*args: Path | str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def test_identical_runs_come_back_clean(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", _base_run())
    code, out = _run("diff", a, b)
    assert code == 0, out
    assert "identical" in out


def test_post_divergence_names_the_producing_dispatch(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.jsonl", _base_run())
    diverged = _base_run()
    diverged[2]["bufs"][1]["post"] = "0xbad"
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "produced" in out
    assert "seq 1 launch matmul" in out
    assert "this dispatch produced the first divergent bytes" in out


def test_pre_divergence_names_the_previous_toucher(tmp_path: Path) -> None:
    # The consumer's operands differ on arrival while the producer's records
    # match: something between the two dispatches changed the bytes.
    diverged = _base_run()
    diverged[3]["bufs"][0]["pre"] = "0xbad"
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "arriving" in out
    assert "seq 2 launch rms_norm" in out
    assert "Last recorded toucher" in out
    assert "seq 1 launch matmul" in out


def test_offset_change_is_the_geometry_class(tmp_path: Path) -> None:
    diverged = _base_run()
    diverged[2]["bufs"][1]["off"] = 4096
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "OPERAND GEOMETRY" in out
    assert "operand-plumbing" in out


def test_copy_alloc_ordinal_change_is_not_a_divergence(tmp_path: Path) -> None:
    """Regression for the record-156 false positive (run
    device-20260806T180821Z): the pass and restore legs' 256-byte registry
    copy differed only in the destination allocation's first-seen ordinal
    (156 vs 1) while the run's own control proved the legs' outputs
    bitwise-identical. Ordinals are address-keyed and not run-comparable for
    copies, so the differ must not flag them."""
    diverged = _base_run()
    diverged[1]["dstAlloc"] = 156
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 0, out
    assert "identical" in out


def test_copy_content_change_still_flags(tmp_path: Path) -> None:
    # Ordinal noise must not mask real bytes: same copy, ordinal differs AND
    # the destination hash differs -- the content class must still fire.
    diverged = _base_run()
    diverged[1]["dstAlloc"] = 156
    diverged[1]["dstPost"] = "0xbad"
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "CONTENT divergence" in out
    assert "produced" in out
    assert "copy_to" in out


def test_copy_size_change_is_still_geometry(tmp_path: Path) -> None:
    diverged = _base_run()
    diverged[1]["n"] = 524288
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "OPERAND GEOMETRY" in out
    assert "not comparable across runs" in out


def test_launch_alloc_relabeling_is_not_a_divergence(tmp_path: Path) -> None:
    """Launch ordinals are per-run labels too: device-20260806T180821Z's
    bitwise-identical legs swapped the pool's scratch ordinals (180/181), so
    a consistent relabeling that preserves which slots share an allocation
    must come back clean."""
    diverged = _base_run()
    for record in diverged[2:]:
        for slot in record["bufs"]:
            slot["alloc"] = 7
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 0, out
    assert "identical" in out


def test_launch_alias_pattern_change_is_geometry(tmp_path: Path) -> None:
    # What launch ordinals DO say across runs is which slots of one dispatch
    # share an allocation. The base run's launches read and write one shared
    # allocation; splitting a slot onto its own allocation is the
    # operand-plumbing signal the geometry class exists for.
    diverged = _base_run()
    diverged[2]["bufs"][1]["alloc"] = 7
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "OPERAND GEOMETRY" in out
    assert "operand-plumbing" in out


def test_different_kernel_is_the_schedule_class(tmp_path: Path) -> None:
    diverged = _base_run()
    diverged[2]["name"] = "gemv"
    diverged[2]["sel"] = 6
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", diverged)
    code, out = _run("diff", a, b)
    assert code == 1
    assert "SCHEDULE divergence" in out


def test_a_truncated_run_is_reported(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.jsonl", _base_run())
    b = _write(tmp_path, "b.jsonl", _base_run()[:-1])
    code, out = _run("diff", a, b)
    assert code == 1
    assert "SCHEDULE divergence" in out
    assert "records" in out


def test_summarize_flags_the_zero_window_signature(tmp_path: Path) -> None:
    run = _base_run()
    run[3]["bufs"][1]["post"] = "0x0"
    run[3]["bufs"][1]["postZ"] = 1
    a = _write(tmp_path, "a.jsonl", run)
    code, out = _run("summarize", a)
    assert code == 1
    assert "D2 zero-prefix signature" in out
    assert "seq 2 launch rms_norm" in out


def test_summarize_flags_a_dispatch_that_changed_nothing(
    tmp_path: Path,
) -> None:
    run = _base_run()
    run[3]["bufs"][1]["post"] = run[3]["bufs"][1]["pre"]
    a = _write(tmp_path, "a.jsonl", run)
    code, out = _run("summarize", a)
    assert code == 1
    assert "no operand window changed" in out


def test_summarize_of_a_clean_run_has_no_flags(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.jsonl", _base_run())
    code, out = _run("summarize", a)
    assert code == 0, out
    assert "no flags" in out
    assert "1 x matmul" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
