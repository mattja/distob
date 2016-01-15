"""Example of scattering objects onto different cluster engines.

   (First start an IPython cluster, e.g. by typing 'ipcluster start')
"""
import distob
import numpy as np

arrays = [np.random.randn(10000) for i in xrange(20)]

arrays = distob.scatter(arrays)
#arrays = distob.async_scatter(arrays)
print('after scattering, arrays = %s' % repr(arrays))

# can still use the remote arrays as if they were local 
variances = [a.var() for a in arrays]

#distob.wait(variances)
print('result: %g' % np.sqrt(np.mean(variances)))
