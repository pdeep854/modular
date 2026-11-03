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
"""QAIC test utilities package.

Provides composable helpers for writing concise QAIC hardware tests:

- `buffers`: Host buffer allocation with sequential, constant, or zero fills.
- `execution`: Compile-and-run wrappers for unary, binary, and reduction
  kernels.
- `verification`: Output buffer comparison using ``assert_almost_equal``.
"""

from .buffers import alloc_sequential, alloc_constant, alloc_zeros
from .execution import (
    compile_and_run_unary,
    compile_and_run_binary,
    compile_and_run_reduction,
    compile_and_run_parallel_reduction,
    MAX_NPU_WORKERS,
)
from .verification import verify_buffer_almost_equal, verify_scalar_almost_equal
from .rng import Xorshift32
from .hmx import (
    vtcm_align,
    npu_is_hmx_thread,
    cpu_matmul_f16,
    verify_matmul_f16,
    print_verify_result,
    MatmulVerifyResult,
)
from .hmx_matmul import (
    hmx_matmul_kernel,
    alloc_dev_fp16,
    compile_and_load_hmx_matmul,
    run_hmx_matmul,
)
from .qccl_runner import RankOpArgs, run_2_ranks
from .tp import (
    TPRankArgs,
    run_2_ranks_raw,
    run_n_ranks_raw,
    fill_fp16_seeded,
    extract_col_block,
)
