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
"""Dynamic loading of the QAIC bindings from MODULAR_DRIVER_PLUGINS.
"""

from std.ffi import OwnedDLHandle, _Global, _get_dylib_function
from std.os import abort, getenv


comptime QAIC_PLUGIN_LIBRARY = _Global["QAIC_PLUGIN_LIBRARY", _init_qaic_plugin]


def _init_qaic_plugin() -> OwnedDLHandle:
    """Open the QAIC plugin .so specified by MODULAR_DRIVER_PLUGINS."""
    var path = getenv("MODULAR_DRIVER_PLUGINS")
    if not path:
        abort(
            "MODULAR_DRIVER_PLUGINS environment variable is not set. "
            "Set it to the path of the QAIC plugin shared library."
        )
    try:
        return OwnedDLHandle(path=path)
    except e:
        abort(
            "Failed to load QAIC plugin from MODULAR_DRIVER_PLUGINS: "
            + String(e)
        )


@always_inline
def _get_qaic_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() -> result_type:
    """Look up `func_name` in the QAIC plugin and return it as `result_type`.

    `result_type` should be a Mojo `def(...)` type matching the C symbol's
    ABI.  Aborts if the library or symbol cannot be found.
    """
    try:
        return _get_dylib_function[
            QAIC_PLUGIN_LIBRARY(),
            func_name,
            result_type,
        ]()
    except e:
        abort(String(e))
