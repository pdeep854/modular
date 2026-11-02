# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests the affine guard `mo.reduce.layer_norm` dispatch applies.

The `layernorm` selector normalizes without applying an affine, so the
registration refuses weights it would otherwise drop. What the guard decides is
whether the weights it was handed are the identity, which needs no device: the
tensors it reads are host memory either way, since this transport shares its
allocations with the host process.

Each case keeps its storage alive past the call it lends a pointer to, the way
the registrations keep their witness buffers alive.
"""

from extensibility import InputTensor
from extensibility.managed_tensor_slice import _get_unknown_tensor_spec
from std.testing import assert_raises, TestSuite
from std.utils import IndexList

from hexagon_ondevice.registrations import _require_identity_affine

comptime DTYPE = DType.float16
comptime LENGTH = 8
comptime SPEC = _get_unknown_tensor_spec[DTYPE, 1]()

comptime Weight = InputTensor[static_spec=SPEC]
comptime Storage = Array[Scalar[DTYPE], LENGTH]


def _fill(mut storage: Storage, value: Scalar[DTYPE]):
    for i in range(LENGTH):
        storage[i] = value


def test_identity_gamma_dispatches() raises:
    var storage = Storage(uninitialized=True)
    _fill(storage, 1)
    var weight = Weight(storage.unsafe_ptr(), IndexList[1](LENGTH))
    _require_identity_affine(weight, 1, "gamma")
    _ = storage^


def test_identity_beta_dispatches() raises:
    var storage = Storage(uninitialized=True)
    _fill(storage, 0)
    var weight = Weight(storage.unsafe_ptr(), IndexList[1](LENGTH))
    _require_identity_affine(weight, 0, "beta")
    _ = storage^


def test_broadcast_identity_gamma_dispatches() raises:
    """A `broadcast_to` of 1.0 spans the axis through a zero stride."""
    var storage = Storage(uninitialized=True)
    _fill(storage, 1)
    var weight = Weight(
        storage.unsafe_ptr(), IndexList[1](LENGTH), IndexList[1](0)
    )
    _require_identity_affine(weight, 1, "gamma")
    _ = storage^


def test_trained_gamma_refused() raises:
    var storage = Storage(uninitialized=True)
    _fill(storage, 1)
    storage[LENGTH - 1] = 0.5
    var weight = Weight(storage.unsafe_ptr(), IndexList[1](LENGTH))
    with assert_raises(
        contains=(
            "the `layernorm` selector applies no affine, so gamma must be 1.0"
            " throughout; element 7 is 0.5"
        )
    ):
        _require_identity_affine(weight, 1, "gamma")
    _ = storage^


def test_trained_beta_refused() raises:
    var storage = Storage(uninitialized=True)
    _fill(storage, 0)
    storage[0] = -0.25
    var weight = Weight(storage.unsafe_ptr(), IndexList[1](LENGTH))
    with assert_raises(
        contains=(
            "the `layernorm` selector applies no affine, so beta must be 0.0"
            " throughout; element 0 is -0.25"
        )
    ):
        _require_identity_affine(weight, 0, "beta")
    _ = storage^


def test_broadcast_trained_gamma_refused() raises:
    """A zero stride is read, not stepped over: every element is the one value.
    """
    var storage = Storage(uninitialized=True)
    _fill(storage, 2)
    var weight = Weight(
        storage.unsafe_ptr(), IndexList[1](LENGTH), IndexList[1](0)
    )
    with assert_raises(contains="element 0 is 2.0"):
        _require_identity_affine(weight, 1, "gamma")
    _ = storage^


def test_nonfinite_gamma_refused() raises:
    """A NaN weight is not the identity, and compares unequal to it."""
    var storage = Storage(uninitialized=True)
    _fill(storage, 1)
    storage[2] = Scalar[DTYPE](FloatLiteral.nan)
    var weight = Weight(storage.unsafe_ptr(), IndexList[1](LENGTH))
    with assert_raises(contains="element 2 is nan"):
        _require_identity_affine(weight, 1, "gamma")
    _ = storage^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
