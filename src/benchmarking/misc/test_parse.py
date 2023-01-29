from benchmark_parse import parseFile
import ast, inspect
import test_function
import torch.utils.benchmark as bench

#first get globals and imports
#then get function defs

members = inspect.getmembers_static(test_function)

# global_vars = vars(test_function)
# print(global_vars)

print(members)

import_members = []
function_members = []
class_members = []
global_members = []

for i, x in enumerate(members):
  if inspect.ismodule(x[1]):
    import_members.append(x)
  elif inspect.isfunction(x[1]):
    function_members.append(x)
  elif inspect.isclass(x[1]):
    class_members.append(x)
  else:
    global_members.append(x)

for f in function_members:
  bench.Timer(
    stmt=inspect.getsource(f[1])
  )






for x in members:
  print(x)