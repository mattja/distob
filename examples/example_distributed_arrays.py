"""Example of scattering a numpy array across many computers on a cluster or
cloud and then using the distributed array.

   (First start an IPython cluster, e.g. by typing 'ipcluster start')
"""
import distob
import numpy as np

# scatter an array of random numbers:
ar = np.random.randn(10000, 20)
print('\nbefore scattering:\n%s' % ar)

ar = distob.scatter(ar)
print('\nafter scattering:\n%s' % ar)


# you can use the distributed array as if it were local:

print('\nmeans of each column:\n%s' % ar.mean(axis=0))

print('\nbasic slicing:\n%s' % ar[1000:1010, 5:9])

print('\nadvanced slicing:\n%s' %
      ar[np.array([20, 7, 7, 9]), np.array([1, 2, 2, 15])])

# numpy computations (and ufuncs) will automatically be done in parallel:

print('\nparallel computation with distributed arrays: ar - ar \n%s' %
      (ar - ar))

print('\nparallel computation with distributed arrays: np.exp(1.0j * ar)\n%s' %
      np.exp(1.0j * ar))


# functions that expect ordinary arrays can now compute in parallel:

from scipy.signal import decimate
vdecimate = distob.vectorize(decimate)
result = vdecimate(ar, 10, axis=0)
print('\ndecimated ar:\n%s' % result)

# another way to write that:
result = distob.apply(decimate, ar, 10, axis=0)
print('\ndecimated ar:\n%s' % result)
