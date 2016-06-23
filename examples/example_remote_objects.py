"""Example of scattering objects onto different cluster engines.

   (First start an IPython cluster, e.g. by typing 'ipcluster start')
"""
import distob
from ipyparallel import AsyncResult

# make some arbitrary python objects:
class A(object):
    def __init__(self, p):
        self.p = p
    def f(self, x):
        return x**2 + self.p

some_numbers = range(10)
objects = [A(p) for p in some_numbers]

print('before scattering:')
for a in objects:
    print(a)

# send objects to the cluster:
objects = distob.scatter(objects)

print('after scattering:')
for a in objects:
    print(a)

# get attributes remotely:
print([a.p for a in objects])

# call methods remotely (with computation in series, same as local objects):
results = [a.f(30) for a in objects]
print(results)

# call methods remotely (with computation in parallel):
results = [a.f(30, block=False) for a in objects]
results = [r.result if isinstance(r, AsyncResult) else r for r in results]
print(results)

# another way to write that (computation in parallel):
results = distob.call_all(objects, 'f', 30)
print(results)
