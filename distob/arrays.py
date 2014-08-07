"""Remote and distributed numpy arrays
"""

from .distob import proxy_methods, Remote
import numpy as np


@proxy_methods(np.ndarray, include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__'))
class RemoteArray(Remote, object):
    """Local object representing a remote ndarray"""
    def __init__(self, obj, client):
        """Make a RemoteArray to access an existing numpy.ndarray.

        Args:
          obj (Ref or object): either a Ref reference to the (possibly remote) 
            object to be controlled, or else an actual (local) object to be 
            controlled.
          client (IPython.parallel.client)
        """
        super(RemoteArray, self).__init__(obj, client)
        descr, shape, strides, typestr = self._ref.metadata
        # implement the python array interface
        self._array_intf = {'descr': descr, 'shape': shape, 
                            'strides': strides, 'typestr': typestr, 
                            'version': 3}

    #If a local consumer wants direct data access via the python 
    #array interface, then ensure a local copy of the data is in memory
    def __get_array_intf(self):
        #print('__array_interface__ requested.')
        self._fetch()
        self._array_intf['data'] = self._obcache.__array_interface__['data']
        return self._array_intf

    # This class cannot be a subclass of ndarray because that implies that
    # the C array interface is implemented. But the C interface promises
    # consumers they can just directly read memory of the array at any time.
    # Meaning array data is forced always to be in the local computer's memory.
    # Instead we implement only the python __array_interface__. Result is
    # all existing C code that calls PyArray_FromAny() on its inputs will
    # automatically work fine with these objects as if they were ndarrays.
    __array_interface__ = property(fget=__get_array_intf)

    shape = property(fget=lambda self: self._array_intf['shape'])

    strides = property(fget=lambda self: self._array_intf['strides'])

    @classmethod
    def __pmetadata__(cls, obj):
        # obj is the real ndarray instance that we will control
        metadata = (obj.__array_interface__['descr'], obj.shape, 
                    obj.strides, obj.__array_interface__['typestr'])
        return metadata

    def __repr__(self):
        return self._cached_apply('__repr__').replace(
            'array', self.__class__.__name__, 1)

    def __len__(self):
        return self._array_intf['shape'][0]

    def __get_nbytes(self):
        if self._obcache is not None:
            return self._obcache.nbytes
        else:
            return 0

    nbytes = property(fget=__get_nbytes)

    def __array_finalize__(self, obj):
        #print('In proxy __array_finalize__, obj is type ' + str(type(obj)))
        if obj is None:
            return

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        """Route ufunc execution intelligently to local host or remote engine 
        depending on where the arguments reside.
        """
        print('In RemoteArray __numpy_ufunc__')
        print('ufunc:%s; method:%s; selfpos:%d' % (repr(ufunc), method, i))
        print('inputs:%s; kwargs:%s' % (inputs, kwargs))

        raise Error("ufunc=%s Haven't implemented ufunc support yet!" % ufunc)
        # TODO implement this!
        #return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_prepare__(self, in_arr, context=None):
        """Fetch underlying data to user's computer and apply ufunc locally.
        Only used as a fallback, for numpy versions < 1.9 which lack 
        support for the __numpy_ufunc__ mechanism. 
        """
        print('In proxy __array_prepare__')
        #TODO fetch data here
        if map(int, np.version.short_version.split('.')) < [1,9,0]:
            raise Error('Numpy version 1.9.0 or later is required!')
        else:
            raise Error('Numpy is current, but still called __array_prepare__')
        #return super(RemoteArray, self).__array_prepare__(in_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__')
        return np.ndarray.__array_wrap__(out_arr, context)


class DistArray(object):
    """Local object representing a single array distributed on multiple engines
    """
    pass
