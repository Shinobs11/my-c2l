import torch, os
import torch.utils.benchmark as benchmark


batched_dot_src = """\
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}
"""


# PyTorch makes it easy to test our C++ implementations by providing a utility
# to JIT compile C++ source into Python extensions:
import os
from torch.utils import cpp_extension
cpp_lib = cpp_extension.load_inline(
    name='cpp_lib',
    cpp_sources=batched_dot_src,
    extra_cflags=['-O3'],
    extra_include_paths=[
        # `load_inline` needs to know where to find Pybind11 headers.
        os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    ],
    functions=['batched_dot_mul_sum_v0', 'batched_dot_mul_sum_v1']
)

# `load_inline` will create a shared object that is loaded into Python. When we collect
# instruction counts Timer will create a subprocess, so we need to re-import it. The
# import process is slightly more complicated for C extensions, but that's all we're
# doing here.
module_import_str = f"""\
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)"""

import textwrap
def pretty_print(result):
    """Import machinery for cpp_lib.so can get repetitive to look at."""
    print(repr(result).replace(textwrap.indent(module_import_str, "  "), "  import cpp_lib"))


t_baseline = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='''\
from __main__ import batched_dot_mul_sum
x = torch.randn(2, 2)''')

t0 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v0(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

t1 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v1(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

# Moving to C++ did indeed reduce overhead, but it's hard to tell which
# calling convention is more efficient. v1 (call with references) seems to
# be a bit faster, but it's within measurement error.
pretty_print(t_baseline.blocked_autorange())
pretty_print(t0.blocked_autorange())
pretty_print(t1.blocked_autorange())