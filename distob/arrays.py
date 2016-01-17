"""Remote and distributed numpy arrays
"""

from __future__ import absolute_import
from .distob import (proxy_methods, Remote, Ref, Error,
                     scatter, gather, _directed_scatter, _scatter_ndarray,
                     vectorize, call, methodcall, convert_result)
import numpy as np
from collections import Sequence
import numbers
import types
import warnings
import copy
import sys

# types for compatibility across python 2 and 3
_SliceType = type(slice(None))
_EllipsisType = type(Ellipsis)
_TupleType = type(())
_NewaxisType = type(np.newaxis)

try:
    string_types = basestring
except NameError:
    string_types = str


def _brief_warning(msg, stacklevel=None):
    old_format = warnings.formatwarning
    def brief_format(message, category, filename, lineno, line=None):
        return '%s [%s:%d]\n' % (msg, filename, lineno)
    warnings.formatwarning = brief_format
    warnings.warn(msg, RuntimeWarning, stacklevel + 1)
    warnings.formatwarning = old_format


@proxy_methods(np.ndarray, exclude=('dtype',), include_underscore=(
    '__getitem__', '__setitem__', '__getslice__', '__setslice__'))
class RemoteArray(Remote, object):
    """Local object representing a remote ndarray"""
    def __init__(self, obj):
        """Make a RemoteArray to access an existing numpy.ndarray.

        Args:
          obj (Ref or object): either a Ref reference to the (possibly remote) 
            object to be controlled, or else an actual (local) object to be 
            controlled.
        """
        super(RemoteArray, self).__init__(obj)
        descr, shape, strides, typestr = self._ref.metadata
        # implement the python array interface
        self._array_intf = {'descr': descr, 'shape': shape, 'strides': strides,
                            'typestr': typestr, 'version': 3}
        self.dtype = np.dtype(typestr)
        self.__engine_affinity__ = (self._id.engine, self.nbytes)

    #If a local consumer wants direct data access via the python 
    #array interface, then ensure a local copy of the data is in memory
    def __get_array_intf(self):
        if not self._obcache_current:
            msg = (u"Note: local numpy function requested local access to " +
                    "data. Fetching data..")
            _brief_warning(msg, stacklevel=4)
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

    size = property(fget=lambda self: np.prod(self._array_intf['shape']))

    itemsize = property(fget=lambda self: int(self._array_intf['typestr'][2:]))

    nbytes = property(fget=lambda self: (np.prod(self._array_intf['shape']) *
                                         int(self._array_intf['typestr'][2:])))

    ndim = property(fget=lambda self: len(self._array_intf['shape']))

    @classmethod
    def __pmetadata__(cls, obj):
        # obj is the real ndarray instance that we will control
        metadata = (obj.__array_interface__['descr'], obj.shape, 
                    obj.strides, obj.__array_interface__['typestr'])
        return metadata

    def __repr__(self):
        return methodcall(self, '__repr__').replace(
                'array', self.__class__.__name__, 1)

    def __len__(self):
        return self._array_intf['shape'][0]

    def __array_finalize__(self, obj):
        #print('In proxy __array_finalize__, obj is type ' + str(type(obj)))
        if obj is None:
            return

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        return _ufunc_dispatch(ufunc, method, i, inputs, **kwargs)

    def __array_prepare__(self, out_arr, context=None):
        """Fetch underlying data to user's computer and apply ufunc locally.
        Only used as a fallback, for numpy versions < 1.11 which lack 
        support for the __numpy_ufunc__ mechanism. 
        """
        #print('In RemoteArray __array_prepare__. context=%s' % repr(context))
        if list(map(int, np.version.short_version.split('.'))) < [1,11,0]:
            msg = (u'\nNote: Distob distributed array arithmetic and ufunc ' +
                    'support requires\nnumpy 1.11.0 or later (not yet ' +
                    'released!) Can get the latest numpy here: \n' +
                    'https://github.com/numpy/numpy/archive/master.zip\n' +
                    'Otherwise, will bring data back to the client to ' +
                    'perform the \'%s\' operation...' % context[0].__name__)
            _brief_warning(msg, stacklevel=3)
            return out_arr
        else:
            raise Error('Have numpy >=1.11 but still called __array_prepare__')

    def __array_wrap__(self, out_arr, context=None):
        #print('In RemoteArray __array_wrap__')
        return out_arr

    def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        ix = [slice(None)] * self.ndim
        ix.insert(axis, np.newaxis)
        ix = tuple(ix)
        return self[ix]

    # The following operations will be intercepted by __numpy_ufunc__()

    def __add__(self, other):
        """x.__add__(y) <==> x+y"""
        return np.add(self, other)

    def __radd__(self, other):
        """x.__radd__(y) <==> y+x"""
        return np.add(other, self)

    def __sub__(self, other):
        """x.__sub__(y) <==> x-y"""
        return np.subtract(self, other)

    def __rsub__(self, other):
        """x.__rsub__(y) <==> y-x"""
        return np.subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <==> x*y"""
        return np.multiply(self, other)

    def __rmul__(self, other):
        """x.__rmul__(y) <==> y*x"""
        return np.multiply(other, self)

    def __floordiv__(self, other):
        """x.__floordiv__(y) <==> x//y"""
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other):
        """x.__rfloordiv__(y) <==> y//x"""
        return np.floor_divide(other, self)

    def __mod__(self, other):
        """x.__mod__(y) <==> x%y"""
        return np.mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <==> y%x"""
        return np.mod(other, self)

    def __divmod__(self, other):
        """x.__divmod__(y) <==> divmod(x, y)"""
        return (np.floor_divide(self - self % other, other), self % other)

    def __rdivmod__(self, other):
        """x.__rdivmod__(y) <==> divmod(y, x)"""
        return (np.floor_divide(other - other % self, self), other % self)

    def __pow__(self, other, modulo=None):
        """x.__pow__(y[, z]) <==> pow(x, y[, z])"""
        # Deliberately match numpy behaviour of ignoring `modulo`
        return np.power(self, other)

    def __rpow__(self, other, modulo=None):
        """y.__rpow__(x[, z]) <==> pow(x, y[, z])"""
        # Deliberately match numpy behaviour of ignoring `modulo`
        return np.power(other, self)

    def __lshift__(self, other):
        """x.__lshift__(y) <==> x<<y"""
        return np.left_shift(self, other)

    def __rlshift__(self, other):
        """x.__lshift__(y) <==> y<<x"""
        return np.left_shift(other, self)

    def __rshift__(self, other):
        """x.__rshift__(y) <==> x>>y"""
        return np.right_shift(self, other)

    def __rrshift__(self, other):
        """x.__rshift__(y) <==> y>>x"""
        return np.right_shift(other, self)

    def __and__(self, other):
        """x.__and__(y) <==> x&y"""
        return np.bitwise_and(self, other)

    def __rand__(self, other):
        """x.__rand__(y) <==> y&x"""
        return np.bitwise_and(other, self)

    def __xor__(self, other):
        """x.__xor__(y) <==> x^y"""
        return np.bitwise_xor(self, other)

    def __rxor__(self, other):
        """x.__rxor__(y) <==> y^x"""
        return np.bitwise_xor(other, self)

    def __or__(self, other):
        """x.__or__(y) <==> x|y"""
        return np.bitwise_or(self, other)

    def __ror__(self, other):
        """x.__ror__(y) <==> y|x"""
        return np.bitwise_or(other, self)

    def __div__(self, other):
        """x.__div__(y) <==> x/y"""
        return np.divide(self, other)

    def __rdiv__(self, other):
        """x.__rdiv__(y) <==> y/x"""
        return np.divide(other, self)

    def __truediv__(self, other):
        """x.__truediv__(y) <==> x/y"""
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <==> y/x"""
        return np.true_divide(other, self)

# in-place operators __iadd__, __imult__ etc are not yet implemented.
# To do so will first need to implement the ufunc out= argument and update the
# location selection rules accordingly to take output location into account.
# Will also want to set self._obcache_current = False before ufunc execution.

    def __iadd__(self, other):
        """x.__add__(y) <==> x+=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __isub__(self, other):
        """x.__sub__(y) <==> x-=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __imul__(self, other):
        """x.__mul__(y) <==> x*=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ifloordiv__(self, other):
        """x.__floordiv__(y) <==> x//=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __imod__(self, other):
        """x.__mod__(y) <==> x%=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ipow__(self, other, modulo=None):
        """x.__pow__(y) <==> x**=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ilshift__(self, other):
        """x.__lshift__(y) <==> x<<=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __irshift__(self, other):
        """x.__rshift__(y) <==> x>>=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __iand__(self, other):
        """x.__and__(y) <==> x&=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ixor__(self, other):
        """x.__xor__(y) <==> x^=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ior__(self, other):
        """x.__or__(y) <==> x|=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __idiv__(self, other):
        """x.__div__(y) <==> x/=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __itruediv__(self, other):
        """x.__truediv__(y) <==> x/=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __neg__(self):
        """x.__neg__() <==> -x"""
        return np.negative(self)

    def __pos__(self):
        """x.__pos__() <==> +x"""
        return self

    def __abs__(self):
        """x.__abs__() <==> abs(x)"""
        return np.abs(self)

    def __invert__(self):
        """x.__invert__() <==> ~x"""
        return np.invert(self)

    def dot(self, other):
        """Dot product of two arrays.
        Refer to `numpy.dot` for full documentation."""
        return np.dot(self, other)


class DistArray(object):
    """Local object representing a single array distributed on multiple engines

    Currently only one of the axes will be distributed.
    """
    def __init__(self, subarrays, axis=None):
        """Make a DistArray from a list of existing remote numpy.ndarrays

        Args:
          subarrays (list of RemoteArrays, ndarrays or Refs to ndarrays):
            the subarrays (possibly remote) which form the whole array when
            concatenated. The subarrays must have the same dtype, and must
            have the same shape excepting the distributed axis.
          axis (int, optional): Position of the distributed axis, which is the 
            axis along which the subarrays will be concatenated. Default is 
            the last axis.  
        """
        from distob import engine
        self._n = len(subarrays)  # number of subarrays along distributed axis.
        if self._n < 2:
            raise ValueError('must provide more than one subarray')
        self._pref_local = True
        self._obcache = None
        self._obcache_current = False
        if axis is None:
            self._distaxis = subarrays[0].ndim - 1
        else:
            self._distaxis = axis
        # In the subarrays list, accept RemoteArray, ndarray or Ref to ndarray:
        for i, ra in enumerate(subarrays):
            if not isinstance(ra, RemoteArray):
                if not isinstance(ra, Ref):
                    ra = Ref(ra)
                RemoteClass = engine.proxy_types[ra.type]
                subarrays[i] = RemoteClass(ra)
        # Require all subarrays to resemble the first, excepting distaxis:
        descr0, subshape0, substrides0, typestr0 = subarrays[0]._ref.metadata
        shape0 = list(subshape0)
        shape0[self._distaxis] = None  # ignore this axis for shape comparison
        sublengths = [] # length of each subarray
        # si[i] is the starting index of the ith subarray for i=0..(self._n-1)
        # si[self._n] == self.shape[self._distaxis], i.e. one past the end.
        si = [0] 
        for i, ra in enumerate(subarrays):
            descr, subshape, substrides, typestr = ra._ref.metadata
            if descr != descr0 or typestr != typestr0:
                raise ValueError('subarrays must have same dtype')
            sublengths.append(subshape[self._distaxis])
            si.append(si[-1] + sublengths[-1])
            shapei = list(subshape)
            shapei[self._distaxis] = None
            if shapei != shape0:
                raise ValueError('subarrays must have the same shape except ' +
                                 'on the axis being distributed')
        self._sublengths = tuple(sublengths)
        self._si = tuple(si)
        itemsize = int(typestr0[2:])
        shape = list(subshape0)
        shape[self._distaxis] = sum(sublengths)
        shape = tuple(shape)
        # For now, report the same strides as used by the subarrays
        strides = substrides
        self.dtype = np.dtype(typestr)
        self._subarrays = subarrays
        # a surrogate ndarray to help with slicing of the distributed axis:
        self._placeholders = np.array(range(shape[self._distaxis]), dtype=int)
        # implement the python array interface
        self._array_intf = {'descr': descr, 'shape': shape, 'strides': strides,
                            'typestr': typestr, 'version': 3}
        location = ([ra._ref.id.engine for ra in subarrays], self._distaxis)
        self.__engine_affinity__ = (location, np.prod(shape)*itemsize)
        if all(ra._obcache_current for ra in self._subarrays):
            self._fetch()

    def __get_pref_local(self):
        return self._pref_local

    def __set_pref_local(self, value):
        if not isinstance(value, bool):
            raise ValueError("'prefer_local' can only be set to True or False")
        self._pref_local = value
        for ra in self._subarrays:
            ra.prefer_local = value

    prefer_local = property(fget=__get_pref_local, fset=__set_pref_local,
                            doc='whether to use locally cached results')

    def _fetch(self):
        """forces update of a local cached copy of the real object
        (regardless of the preference setting self.cache)"""
        if not self._obcache_current:
            from distob import engine
            ax = self._distaxis
            self._obcache = concatenate([ra._ob for ra in self._subarrays], ax)
            # let subarray obcaches and main obcache be views on same memory:
            for i in range(self._n):
                ix = [slice(None)] * self.ndim
                ix[ax] = slice(self._si[i], self._si[i+1])
                self._subarrays[i]._obcache = self._obcache[tuple(ix)]
            self._obcache_current = True
            # now prefer local processing:
            self.__engine_affinity__ = (
                    engine.eid, self.__engine_affinity__[1])

    def __ob(self):
        """return a copy of the real object"""
        self._fetch()
        return self._obcache

    _ob = property(fget=__ob, doc='return a local copy of the object')

    #If a local consumer asks for direct data access via the python
    #array interface, attempt to put a local copy of all the data into memory
    def __get_array_intf(self):
        #print('__array_interface__ requested.')
        self._fetch()
        self._array_intf['data'] = self._obcache.__array_interface__['data']
        self._array_intf['strides'] = \
                self._obcache.__array_interface__['strides']
        return self._array_intf

    __array_interface__ = property(fget=__get_array_intf)

    shape = property(fget=lambda self: self._array_intf['shape'])

    strides = property(fget=lambda self: self._array_intf['strides'])

    size = property(fget=lambda self: np.prod(self._array_intf['shape']))

    itemsize = property(fget=lambda self: int(self._array_intf['typestr'][2:]))

    nbytes = property(fget=lambda self: (np.prod(self._array_intf['shape']) *
                                         int(self._array_intf['typestr'][2:])))

    ndim = property(fget=lambda self: len(self._array_intf['shape']))

    def __repr__(self):
        classname = self.__class__.__name__
        if 0 in self.shape:
            return u'<%s of shape %s with axis %d distributed>' % (
                    classname, self.shape, self._distaxis)
        selectors = []
        for i in range(self.ndim):
            if self.shape[i] > 3:
                selectors.append((0, 1, -2, -1))
            else:
                selectors.append(tuple(range(self.shape[i])))
        ix = tuple(np.meshgrid(*selectors, indexing='ij'))
        corners = gather(self[ix])
        def _repr_nd(corners, shape, indent):
            """Recursively generate abbreviated text representation of array"""
            if not shape:
                if isinstance(corners, np.float):
                    return '{: .8f}'.format(corners)
                else:
                    return str(corners)
            else:
                if len(shape) > 1:
                    pre = ',' + '\n'*(len(shape)-1) + ' '*indent
                else:
                    pre = ', '
                s = '['
                n = shape[0]
                if n > 0:
                    s += _repr_nd(corners[0], shape[1:], indent+1)
                if n > 1:
                    s += pre + _repr_nd(corners[1], shape[1:], indent+1)
                if n > 4:
                    s += pre + '...'
                if n > 3:
                    s += pre + _repr_nd(corners[-2], shape[1:], indent+1)
                if n > 2:
                    s += pre + _repr_nd(corners[-1], shape[1:], indent+1)
                s += ']'
                return s
        s = u'<%s of shape %s with axis %d distributed>:\n' % (
                classname, self.shape, self._distaxis)
        indent = len(classname) + len(self.shape) - 1
        s += classname + '(' + _repr_nd(corners, self.shape, indent) + ')'
        return s

    def __len__(self):
        return self._array_intf['shape'][0]

    def __get_nbytes(self):
        return sum(ra.nbytes for ra in self._subarrays)

    nbytes = property(fget=__get_nbytes)

    ndim = property(fget=lambda self: len(self._array_intf['shape']))

    def __array_finalize__(self, obj):
        #print('In DistArray __array_finalize__, obj type: ' + str(type(obj)))
        if obj is None:
            return

    def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
        return _ufunc_dispatch(ufunc, method, i, inputs, **kwargs)

    def __array_prepare__(self, out_arr, context=None):
        """Fetch underlying data to user's computer and apply ufunc locally.
        Only used as a fallback, for numpy versions < 1.11 which lack 
        support for the __numpy_ufunc__ mechanism. 
        """
        #print('In DistArray __array_prepare__. context=%s' % repr(context))
        if list(map(int, np.version.short_version.split('.'))) < [1,11,0]:
            msg = (u'\nNote: Distob distributed array arithmetic and ufunc ' +
                    'support requires\nnumpy 1.11.0 or later (not yet ' +
                    'released!) Can get the latest numpy here: \n' +
                    'https://github.com/numpy/numpy/archive/master.zip\n' +
                    'Otherwise, will bring data back to the client to ' +
                    'perform the \'%s\' operation...' % context[0].__name__)
            _brief_warning(msg, stacklevel=3)
            return out_arr
        else:
            raise Error('Have numpy >=1.11 but still called __array_prepare__')

    def __array_wrap__(self, out_arr, context=None):
        #print('In DistArray __array_wrap__')
        return out_arr

    def _tosub(self, ix):
        """Given an integer index ix on the distributed axis, returns the pair
        (s, m) where s is the relevant subarray and m is the subindex into s.
        For example if the last axis is distributed,
          self[...,ix] == self._subarrays[s][...,m]
        """
        N = self.shape[self._distaxis]
        if ix >= N or ix < -N:
            raise IndexError(
                    'index %d out of bounds for axis %d of size %d',
                    ix, self._distaxis, N)
        if ix < 0:
            ix += N
        for s in range(0, self._n):
            if self._si[s + 1] - 1 >= ix:
                break
        m = ix - self._si[s]
        return s, m

    def _tosubs(self, ixlist):
        """Maps a list of integer indices of the DistArray to subarray indices.
        ixlist can contain repeated indices and does not need to be sorted.
        Returns pair (ss, ms) where ss is a list of subarrays and ms is a
        list of lists of subindices m (one list for each subarray s in ss).
        """
        n = len(ixlist)
        N = self.shape[self._distaxis]
        ss = []
        ms = []
        if n == 0:
            return ss, ms
        j = 0 # the position in ixlist currently being processed
        ix = ixlist[j]
        if ix >= N or ix < -N:
            raise IndexError(
                    'index %d out of bounds for axis %d of size %d',
                    ix, self._distaxis, N)
        if ix < 0:
            ix += N
        while j < n:
            for s in range(0, self._n):
                low = self._si[s]
                high = self._si[s + 1]
                if ix >= low and ix < high:
                    ss.append(s)
                    msj = [ix - low]
                    j += 1
                    while j < n:
                        ix = ixlist[j]
                        if ix >= N or ix < -N:
                            raise IndexError(
                              'index %d out of bounds for axis %d of size %d',
                              ix, self._distaxis, N)
                        if ix < 0:
                            ix += N
                        if ix < low or ix >= high:
                            break
                        msj.append(ix - low)
                        j += 1
                    ms.append(msj)
                if ix < low:
                    break
        return ss, ms

    def _tosubsj(self, ixlist):
        """Like _tosubs(), maps a list of integer indices of the DistArray to
        subarray indices, but also returns the positions processed in the
        original index list.
        ixlist can contain repeated indices and does not need to be sorted.
        Returns tuple (ss, ms, js) where ss is a list of subarrays, ms is a
        list of lists of subindices m (one list for each subarray s in ss), and
        js is a list of lists of positions in ixlist that were processed (one
        list for each subarray s in ss).
        """
        n = len(ixlist)
        N = self.shape[self._distaxis]
        ss = []
        ms = []
        js = []
        if n == 0:
            return ss, ms, js
        j = 0 # the position in ixlist currently being processed
        ix = ixlist[j]
        if ix >= N or ix < -N:
            raise IndexError(
                    'index %d out of bounds for axis %d of size %d',
                    ix, self._distaxis, N)
        if ix < 0:
            ix += N
        while j < n:
            for s in range(0, self._n):
                low = self._si[s]
                high = self._si[s + 1]
                if ix >= low and ix < high:
                    ss.append(s)
                    msj = [ix - low]
                    jsj = [j]
                    j += 1
                    while j < n:
                        ix = ixlist[j]
                        if ix >= N or ix < -N:
                            raise IndexError(
                              'index %d out of bounds for axis %d of size %d',
                              ix, self._distaxis, N)
                        if ix < 0:
                            ix += N
                        if ix < low or ix >= high:
                            break
                        msj.append(ix - low)
                        jsj.append(j)
                        j += 1
                    ms.append(msj)
                    js.append(jsj)
                if ix < low:
                    break
        return ss, ms, js

    def _tosubslices(self, sl):
        """Maps a slice object for whole array to slice objects for subarrays.
        Returns pair (ss, ms) where ss is a list of subarrays and ms is a list
        giving the slice object that should be applied to each subarray.
        """
        N = self.shape[self._distaxis]
        start, stop, step = sl.start, sl.stop, sl.step
        if step is None:
            step = 1
        ss = []
        ms = []
        if step > 0:
            if start is None:
                start = 0
            if stop is None:
                stop = N
            subs = range(0, self._n)
            for s in subs:
                low = self._si[s]
                high = self._si[s + 1]
                first = low + ((low - start) % step)
                last = high + ((high - start) % step)
                if start < high and stop > low and first < high:
                    ss.append(s)
                    substart = max(first, start) - low
                    substop = min(last, stop) - low
                    ms.append(slice(substart, substop, step))
        elif step < 0:
            if start is None:
                start = N - 1
            if stop is None:
                stop = -1
            subs = range(self._n - 1, -1, -1)
            for s in subs:
                low = self._si[s]
                high = self._si[s + 1]
                first = high + step + ((high - start) % step)
                last = low + step + ((low - start) % step)
                if start >= low and stop < high and first >= low:
                    ss.append(s)
                    substart = min(first, start) - low
                    substop = max(last + step, stop) - low
                    if substop < 0:
                        substop = None
                    ms.append(slice(substart, substop, step))
        else:
            raise ValueError('slice step cannot be zero')
        return ss, ms

    def __getitem__(self, index):
        """Slice the distributed array"""
        # To be a DistArray, must have an axis across >=2 engines. If the slice
        # result will no longer be distributed, we return a RemoteArray instead
        distaxis = self._distaxis
        if isinstance(index, np.ndarray) and index.dtype.type is np.bool_:
            raise Error('indexing by boolean array not yet implemented')
        if not isinstance(index, Sequence):
            index = (index,) + (slice(None),)*(self.ndim - 1)
        ix_types = tuple(type(x) for x in index)
        if (np.ndarray in ix_types or
                (not isinstance(index, _TupleType) and 
                    _NewaxisType not in ix_types and 
                    _EllipsisType not in ix_types and
                    _SliceType not in ix_types) or
                any(issubclass(T, Sequence) for T in ix_types)):
            basic_slicing = False
        else:
            basic_slicing = True
        # Apply any ellipses
        while _EllipsisType in ix_types:
            pos = ix_types.index(_EllipsisType)
            m = (self.ndim + ix_types.count(_NewaxisType) - len(index) + 1)
            index = index[:pos] + (slice(None),)*m + index[(pos+1):]
            ix_types = tuple(type(x) for x in index)
        # Apply any np.newaxis
        if _NewaxisType in ix_types:
            new_distaxis = distaxis
            subix = [slice(None)] * self.ndim
            while _NewaxisType in ix_types:
                pos = ix_types.index(type(np.newaxis))
                index = index[:pos] + (slice(None),) + index[(pos+1):]
                ix_types = tuple(type(x) for x in index)
                if pos <= distaxis:
                    subix[pos] = np.newaxis
                    new_distaxis += 1
                else:
                    subix[pos - 1] = np.newaxis
            new_subarrays = [ra[tuple(subix)] for ra in self._subarrays]
            return DistArray(new_subarrays, new_distaxis)[index]
        index = tuple(index) + (slice(None),)*(self.ndim - len(index))
        if len(index) > self.ndim:
            raise IndexError('too many indices for array')
        if basic_slicing:
            # separate the index acting on distributed axis from other indices
            distix = index[distaxis]
            if isinstance(distix, numbers.Number):
                # distributed axis has been sliced away: return a RemoteArray
                s, i = self._tosub(distix)
                subix = index[0:distaxis] + (i,) + index[(distaxis+1):]
                return self._subarrays[s][subix]
            else:
                result_ras = []
                ss, ms = self._tosubslices(distix)
                for s, m in zip(ss, ms):
                    subix = index[0:distaxis] + (m,) + index[(distaxis+1):]
                    result_ras.append(self._subarrays[s][subix])
                axes_removed = sum(1 for x in index[:distaxis] if isinstance(
                        x, numbers.Integral))
                new_distaxis = distaxis - axes_removed
        else:
            # advanced integer slicing
            is_fancy = tuple(not isinstance(x, _SliceType) for x in index)
            fancy_pos = tuple(i for i in range(len(index)) if is_fancy[i])
            slice_pos = tuple(i for i in range(len(index)) if not is_fancy[i])
            contiguous = (fancy_pos[-1] - fancy_pos[0] == len(fancy_pos) - 1)
            index = list(index)
            ix_arrays = [index[j] for j in fancy_pos]
            ix_arrays = np.broadcast_arrays(*ix_arrays)
            for j in range(len(fancy_pos)):
                if ix_arrays[j].shape is ():
                    ix_arrays[j] = np.expand_dims(ix_arrays[j], 0)
                index[fancy_pos[j]] = ix_arrays[j]
            index = tuple(index)
            idim = index[fancy_pos[0]].ndim # common ndim of all index arrays
            assert(idim > 0)
            distix = index[distaxis]
            otherix = index[0:distaxis] + (slice(None),) + index[(distaxis+1):]
            if not is_fancy[distaxis]:
                # fancy indexing is only being applied to non-distributed axes
                result_ras = []
                ss, ms = self._tosubslices(distix)
                for s, m in zip(ss, ms):
                    subix = index[0:distaxis] + (m,) + index[(distaxis+1):]
                    result_ras.append(self._subarrays[s][subix])
                # predict where that new axis will be in subarrays post-slicing
                if contiguous:
                    if fancy_pos[0] > distaxis:
                        new_distaxis = distaxis
                    else:
                        new_distaxis = distaxis - len(fancy_pos) + idim
                else:
                    earlier_fancy = len([i for i in fancy_pos if i < distaxis])
                    new_distaxis = distaxis - earlier_fancy + idim
            else:
                # fancy indexing is being applied to the distributed axis
                nonconstant_ix_axes = []
                for j in range(idim):
                    n = distix.shape[j]
                    if n > 1:
                        partix = np.split(distix, n, axis=j)
                        if not all(np.array_equal(
                                partix[0], partix[i]) for i in range(1, n)):
                            nonconstant_ix_axes.append(j)
                if len(nonconstant_ix_axes) <= 1:
                    # then we can apply the indexing without moving data
                    if len(nonconstant_ix_axes) is 0:
                        # implies every entry of indexing array for distaxis is
                        # the same, so all result data is on a single engine.
                        all_same_engine = True
                        iax = idim - 1
                    else:
                        # len(nonconstant_ix_axes) is 1
                        # the result will remain distributed on one output axis
                        all_same_engine = False
                        iax = nonconstant_ix_axes[0]
                    iix = [0] * idim
                    iix[iax] = slice(None)
                    iix = tuple(iix)
                    ixlist = self._placeholders[distix[iix]]
                    if contiguous:
                        new_distaxis = fancy_pos[0] + iax
                    else:
                        new_distaxis = 0 + iax
                    result_ras = []
                    ss, ms, js = self._tosubsj(ixlist)
                    shp = [1] * idim
                    for s, m, jlist in zip(ss, ms, js):
                        # Slice the original indexing arrays into smaller
                        # arrays, suitable for indexing our subarrays.
                        shp[iax] = len(m)
                        m = np.array(m).reshape(shp) # make shape broadcastable
                        sl = [slice(None)] * idim
                        sl[iax] = jlist
                        sl = tuple(sl)
                        subix = list(index)
                        for i in range(len(subix)):
                            if isinstance(subix[i], np.ndarray):
                                if i == distaxis:
                                    subix[i] = m
                                else:
                                    subix[i] = subix[i][sl]
                        subix = tuple(subix)
                        result_ras.append(self._subarrays[s][subix])
                    if all_same_engine:
                        result_ras = [expand_dims(r, new_distaxis) for
                                      r in result_ras]
                else:
                    # Have more than one nonconstant indexing axis in the
                    # indexing array that applies to our distributed axis.
                    msg = (u'The requested slicing operation requires moving '
                            'data between engines. Will fetch data locally.')
                    warnings.warn(msg, RuntimeWarning)
                    return self._ob[index]
        if len(result_ras) == 0:
            # result is an array with 0 in its shape
            subix = index[0:distaxis] + (slice(0),) + index[(distaxis+1):]
            return self._subarrays[0][subix]
        if all(not isinstance(ra, RemoteArray) or ra._id.engine == -1 for
                ra in result_ras):
            # all resulting subarrays are local: return an ordinary ndarray
            return gather(concatenate(result_ras, axis=new_distaxis))
        engines = [ra._id.engine if isinstance(ra, RemoteArray) else -1 for
                   ra in result_ras]
        if all(e == engines[0] for e in engines):
            # no longer distributed: return a RemoteArray
            return concatenate(result_ras, axis=new_distaxis)
        else:
            return DistArray(result_ras, new_distaxis)

    def __setitem__(self, index, value):
        """Assign to the sliced item"""
        raise Error(u'assigning to remote arrays via slices is '
                     'not yet implemented. stay tuned.')

    def transpose(self, *axes):
        """Returns a view of the array with axes transposed.

        For a 1-D array, this has no effect.
        For a 2-D array, this is the usual matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted

        Args:
          a (array_like): Input array.
          axes (tuple of int, optional): By default, reverse the dimensions, 
            otherwise permute the axes according to values given.
        """
        if axes is ():
            axes = None
        return transpose(self, axes)

    T = property(fget=transpose)

    @staticmethod
    def _valid_distaxis(shapes, ax):
        """`ax` is a valid candidate for a distributed axis if the given
        subarray shapes are all the same when ignoring axis `ax`"""
        compare_shapes = np.vstack(shapes)
        if ax < compare_shapes.shape[1]:
            compare_shapes[:, ax] = -1
        return np.count_nonzero(compare_shapes - compare_shapes[0]) == 0

    def _new_distaxis(self, out_shapes):
        out_ndim = len(out_shapes[0])
        in_shapes = [ra.shape for ra in self._subarrays]
#         # To choose distaxis of result array, match input shape if possible:
#         old_subshape = (in_shapes[0][0:self._distaxis] +
#                         in_shapes[0][(self._distaxis+1):])
#         res_subshape = list(out_shapes[0])
#         i = out_ndim
#         for i in range(len(old_subshape)):
#             n = old_subshape[i]
#             if n not in res_subshape:
#                 continue
#             i = res_subshape.index(n)
#             res_subshape[i] = None
#             if i >= self._distaxis:
#                 break
#             i += 1
#         if self._valid_distaxis(out_shapes, i):
#             return i
        # Find last axis with shape of the original distaxis:
        for i in range(out_ndim - 1, -1, -1):
            if (all(out_shapes[j][i] == in_shapes[j][self._distaxis] for
                    j in range(self._n)) and
                    self._valid_distaxis(out_shapes, i)):
                return i
        # Else find last axis that is length 1 on all subarrays:
        for i in range(out_ndim - 1, -1, -1):
            if (all(s[i] == 1 for s in out_shapes) and
                    self._valid_distaxis(out_shapes, i)):
                return i
        # Finally try appending a new axis at the end:
        if self._valid_distaxis(out_shapes, out_ndim):
            return out_ndim
        else:
            return None

    @classmethod
    def __distob_vectorize__(cls, f):
        """Upgrades a normal function f to act on a DistArray in parallel

        Args:
          f (callable): ordinary function which expects as its first 
            argument an array (of the same shape as our subarrays)

        Returns:
          vf (callable): new function that takes a DistArray as its first
            argument. ``vf(distarray)`` will do the computation ``f(subarray)``
            on each subarray in parallel and if possible will return another 
            DistArray. (otherwise will return a list with the result for each 
            subarray).
        """
        def vf(self, *args, **kwargs):
            kwargs = kwargs.copy()
            kwargs['block'] = False
            kwargs['prefer_local'] = False
            ars = [call(f, ra, *args, **kwargs) for ra in self._subarrays]
            results = [convert_result(ar) for ar in ars]
            if all(isinstance(r, RemoteArray) and
                   r.ndim == results[0].ndim for r in results):
                # Then we will join the results and return a DistArray.
                out_shapes = [ra.shape for ra in results]
                new_distaxis = self._new_distaxis(out_shapes)
                if new_distaxis is None:
                    return results # incompatible array shapes. return a list.
                if new_distaxis == results[0].ndim:
                    results = [r.expand_dims(new_distaxis) for r in results]
                return DistArray(results, new_distaxis)
            elif all(isinstance(r, numbers.Number) for r in results):
                return np.array(results)
            else:
                return results  # not arrays or numbers. return a list.
        if hasattr(f, '__name__'):
            vf.__name__ = 'v' + f.__name__
            f_str = f.__name__ + '()'
        else:
            f_str = 'callable'
        doc = u"""Apply %s in parallel to a DistArray\n
               Args:
                 da (DistArray)
                 other args are the same as for %s
               """ % (f_str, f_str)
        if hasattr(f, '__doc__') and f.__doc__ is not None:
            doc = doc.rstrip() + (' detailed below:\n----------\n' + f.__doc__)
        vf.__doc__ = doc
        return vf

    def __distob_gather__(self):
        return self._ob

    def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        if axis == -1:
            axis = self.ndim
        if axis <= self._distaxis:
            subaxis = axis
            new_distaxis = self._distaxis + 1
        else:
            subaxis = axis - 1
            new_distaxis = self._distaxis
        new_subarrays = [expand_dims(ra, subaxis) for ra in self._subarrays]
        return DistArray(new_subarrays, new_distaxis)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Compute the arithmetic mean along the specified axis.
        See np.mean() for details."""
        if axis == -1:
            axis = self.ndim
        if axis is None:
            results = vectorize(mean)(self, axis, dtype, keepdims=False)
            weights = self._sublengths
            res = np.average(results, axis=None, weights=weights)
            if keepdims:
                for i in range(self.ndim):
                    res = expand_dims(res, res.ndim)
        elif axis == self._distaxis:
            results = vectorize(mean)(self, axis, dtype, keepdims=True)
            results = gather(results)
            # Average manually (np.average doesn't preserve ndarray subclasses)
            weights = (np.array(self._sublengths, dtype=np.float64) /
                       sum(self._sublengths))
            ix = [slice(None)] * self.ndim
            ix[axis] = 0
            res = results[ix] * weights[0]
            for i in range(1, self._n):
                ix[axis] = i
                res = res + results[ix] * weights[i]
            if keepdims:
                res = expand_dims(res, axis)
        else:
            res = vectorize(mean)(self, axis, dtype, keepdims=False)
            if keepdims:
                res = expand_dims(res, axis)
        if out is not None:
            out[:] = res
        return res

    # The following operations will be intercepted by __numpy_ufunc__()

    def __add__(self, other):
        """x.__add__(y) <==> x+y"""
        return np.add(self, other)

    def __radd__(self, other):
        """x.__radd__(y) <==> y+x"""
        return np.add(other, self)

    def __sub__(self, other):
        """x.__sub__(y) <==> x-y"""
        return np.subtract(self, other)

    def __rsub__(self, other):
        """x.__rsub__(y) <==> y-x"""
        return np.subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <==> x*y"""
        return np.multiply(self, other)

    def __rmul__(self, other):
        """x.__rmul__(y) <==> y*x"""
        return np.multiply(other, self)

    def __floordiv__(self, other):
        """x.__floordiv__(y) <==> x//y"""
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other):
        """x.__rfloordiv__(y) <==> y//x"""
        return np.floor_divide(other, self)

    def __mod__(self, other):
        """x.__mod__(y) <==> x%y"""
        return np.mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <==> y%x"""
        return np.mod(other, self)

    def __divmod__(self, other):
        """x.__divmod__(y) <==> divmod(x, y)"""
        return (np.floor_divide(self - self % other, other), self % other)

    def __rdivmod__(self, other):
        """x.__rdivmod__(y) <==> divmod(y, x)"""
        return (np.floor_divide(other - other % self, self), other % self)

    def __pow__(self, other, modulo=None):
        """x.__pow__(y[, z]) <==> pow(x, y[, z])"""
        # Deliberately match numpy behaviour of ignoring `modulo`
        return np.power(self, other)

    def __rpow__(self, other, modulo=None):
        """y.__rpow__(x[, z]) <==> pow(x, y[, z])"""
        # Deliberately match numpy behaviour of ignoring `modulo`
        return np.power(other, self)

    def __lshift__(self, other):
        """x.__lshift__(y) <==> x<<y"""
        return np.left_shift(self, other)

    def __rlshift__(self, other):
        """x.__lshift__(y) <==> y<<x"""
        return np.left_shift(other, self)

    def __rshift__(self, other):
        """x.__rshift__(y) <==> x>>y"""
        return np.right_shift(self, other)

    def __rrshift__(self, other):
        """x.__rshift__(y) <==> y>>x"""
        return np.right_shift(other, self)

    def __and__(self, other):
        """x.__and__(y) <==> x&y"""
        return np.bitwise_and(self, other)

    def __rand__(self, other):
        """x.__rand__(y) <==> y&x"""
        return np.bitwise_and(other, self)

    def __xor__(self, other):
        """x.__xor__(y) <==> x^y"""
        return np.bitwise_xor(self, other)

    def __rxor__(self, other):
        """x.__rxor__(y) <==> y^x"""
        return np.bitwise_xor(other, self)

    def __or__(self, other):
        """x.__or__(y) <==> x|y"""
        return np.bitwise_or(self, other)

    def __ror__(self, other):
        """x.__ror__(y) <==> y|x"""
        return np.bitwise_or(other, self)

    def __div__(self, other):
        """x.__div__(y) <==> x/y"""
        return np.divide(self, other)

    def __rdiv__(self, other):
        """x.__rdiv__(y) <==> y/x"""
        return np.divide(other, self)

    def __truediv__(self, other):
        """x.__truediv__(y) <==> x/y"""
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <==> y/x"""
        return np.true_divide(other, self)

# in-place operators __iadd__, __imult__ etc are not yet implemented.
# To do so will first need to implement the ufunc out= argument and update the
# location selection rules accordingly to take output location into account.
# Will also want to set self._obcache_current = False before ufunc execution.

    def __iadd__(self, other):
        """x.__add__(y) <==> x+=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __isub__(self, other):
        """x.__sub__(y) <==> x-=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __imul__(self, other):
        """x.__mul__(y) <==> x*=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ifloordiv__(self, other):
        """x.__floordiv__(y) <==> x//=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __imod__(self, other):
        """x.__mod__(y) <==> x%=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ipow__(self, other, modulo=None):
        """x.__pow__(y) <==> x**=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ilshift__(self, other):
        """x.__lshift__(y) <==> x<<=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __irshift__(self, other):
        """x.__rshift__(y) <==> x>>=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __iand__(self, other):
        """x.__and__(y) <==> x&=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ixor__(self, other):
        """x.__xor__(y) <==> x^=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __ior__(self, other):
        """x.__or__(y) <==> x|=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __idiv__(self, other):
        """x.__div__(y) <==> x/=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __itruediv__(self, other):
        """x.__truediv__(y) <==> x/=y"""
        raise Error('distributed in-place operators not yet implemented')

    def __neg__(self):
        """x.__neg__() <==> -x"""
        return np.negative(self)

    def __pos__(self):
        """x.__pos__() <==> +x"""
        return self

    def __abs__(self):
        """x.__abs__() <==> abs(x)"""
        return np.abs(self)

    def __invert__(self):
        """x.__invert__() <==> ~x"""
        return np.invert(self)

    def dot(self, other):
        """Dot product of two arrays.
        Refer to `numpy.dot` for full documentation."""
        return np.dot(self, other)


def __print_ufunc(ufunc, method, i, inputs, **kwargs):
    print('ufunc:%s; method:%s; selfpos:%d; kwargs:%s' % (
        repr(ufunc), method, i, kwargs))
    for j, ar in enumerate(inputs):
        shape = ar.shape if hasattr(ar, 'shape') else None
        print('input %d: type=%s shape=%s' % (j, type(ar), repr(shape)))


def _rough_size(obj):
    if hasattr(obj, 'nbytes') and hasattr(type(obj), '__array_interface__'):
        return obj.nbytes
    elif (isinstance(obj, Sequence) and
          not isinstance(obj, string_types) and
          len(obj) > 0):
        # don't need accuracy, so for speed assume items roughly of equal size
        return _rough_size(obj[0]) * len(obj)
    else:
        return sys.getsizeof(obj)


def _engine_affinity(obj):
    """Which engine or engines are preferred for processing this object
    Returns: (location, weight)
      location (integer or tuple): engine id (or in the case of a distributed
      array a tuple (engine_id_list, distaxis)).
      weight(integer): Proportional to the cost of moving the object to a
        different engine. Currently just taken to be the size of data.
    """
    from distob import engine
    this_engine = engine.eid
    if isinstance(obj, numbers.Number) or obj is None:
        return (this_engine, 0)
    elif hasattr(obj, '__engine_affinity__'):
        # This case includes Remote subclasses and DistArray
        return obj.__engine_affinity__
    else:
        return (this_engine, _rough_size(obj))


def _ufunc_move_input(obj, location, bshape):
    """Copy ufunc input `obj` to new engine location(s) unless obj is scalar.

    If the input is requested to be distributed to multiple engines, this
    function will also take care of broadcasting along the distributed axis.

    If the input obj is a scalar, it will be passed through unchanged.

    Args:
      obj (array_like or scalar): one of the inputs to a ufunc
      location (integer or tuple): If an integer, this specifies a single
        engine id to which an array input should be moved. If it is a tuple,
        location[0] is a list of engine ids for distributing the array input
        and location[1] an integer indicating which axis should be distributed.
      bshape (tuple): The shape to which the input will ultimately be broadcast

    Returns:
      array_like or RemoteArray or DistArray or scalar
    """
    if (not hasattr(type(obj), '__array_interface__') and
            not isinstance(obj, Remote) and
            (isinstance(obj, string_types) or
             not isinstance(obj, Sequence))):
        # then treat it as a scalar
        return obj
    from distob import engine
    this_engine = engine.eid
    if location == this_engine:
        # move obj to the local host, if not already here
        if isinstance(obj, Remote) or isinstance(obj, DistArray):
            return gather(obj)
        else:
            return obj
    elif isinstance(location, numbers.Integral):
        # move obj to a single remote engine
        if isinstance(obj, Remote) and obj._ref.id.engine == location:
            #print('no data movement needed!')
            return obj
        obj = gather(obj)
        return _directed_scatter(obj, axis=None, destination=location)
    else:
        # location is a tuple (list of engine ids, distaxis) indicating that
        # obj should be distributed.
        engine_ids, distaxis = location
        if not isinstance(obj, DistArray):
            gather(obj)
            if isinstance(obj, Sequence):
                obj = np.array(obj)
        if obj.ndim < len(bshape):
            ix = (np.newaxis,)*(len(bshape)-obj.ndim) + (slice(None),)*obj.ndim
            obj = obj[ix]
        if (isinstance(obj, DistArray) and distaxis == obj._distaxis and
                engine_ids == [ra._ref.id.engine for ra in obj._subarrays]):
            #print('no data movement needed!')
            return obj
        obj = gather(obj)
        if obj.shape[distaxis] == 1:
            # broadcast this axis across engines
            subarrays = [_directed_scatter(obj, None, m) for m in engine_ids]
            return DistArray(subarrays, distaxis)
        else:
            return _directed_scatter(obj, distaxis, destination=engine_ids)


def _ufunc_dispatch(ufunc, method, i, inputs, **kwargs):
    """Route ufunc execution intelligently to local host or remote engine(s)
    depending on where the inputs are, to minimize the need to move data.
    Args:
      see numpy documentation for __numpy_ufunc__
    """
    #__print_ufunc(ufunc, method, i, inputs, **kwargs)
    if 'out' in kwargs and kwargs['out'] is not None:
        raise Error('for distributed ufuncs `out=` is not yet implemented')
    nin = 2 if ufunc is np.dot else ufunc.nin
    if nin is 1 and method == '__call__':
        return vectorize(ufunc.__call__)(inputs[0], **kwargs)
    elif nin is 2 and method == '__call__':
        from distob import engine
        here = engine.eid
        # Choose best location for the computation, possibly distributed:
        locs, weights = zip(*[_engine_affinity(a) for a in inputs])
        # for DistArrays, adjust preferred distaxis to account for broadcasting
        bshape = _broadcast_shape(*inputs)
        locs = list(locs)
        for i, loc in enumerate(locs):
            if isinstance(loc, _TupleType):
                num_new_axes = len(bshape) - inputs[i].ndim
                if num_new_axes > 0:
                    locs[i] = (locs[i][0], locs[i][1] + num_new_axes)
        if ufunc is np.dot:
            locs = [here if isinstance(m, _TupleType) else m for m in locs]
        if locs[0] == locs[1]:
            location = locs[0]
        else:
            # TODO: More accurately penalize the increased data movement if we
            # choose to distribute an axis that requires broadcasting.
            smallest = 0 if weights[0] <= weights[1] else 1
            largest = 1 - smallest
            if locs[0] is here or locs[1] is here:
                location = here if weights[0] == weights[1] else locs[largest]
            else:
                # Both inputs are on remote engines. With the current
                # implementation, data on one remote engine can only be moved
                # to another remote engine via the client. Cost accordingly:
                if weights[smallest]*2 < weights[largest] + weights[smallest]:
                    location = locs[largest]
                else:
                    location = here
        # Move both inputs to the chosen location:
        inputs = [_ufunc_move_input(a, location, bshape) for a in inputs]
        # Execute computation:
        if location is here:
            return ufunc.__call__(inputs[0], inputs[1], **kwargs)
        else:
            if isinstance(location, numbers.Integral):
                # location is a single remote engine
                return call(ufunc.__call__, inputs[0], inputs[1], **kwargs)
            else:
                # location is a tuple (list of engine ids, distaxis) implying
                # that the moved inputs are now distributed arrays (or scalar)
                engine_ids, distaxis = location
                n = len(engine_ids)
                is_dist = tuple(isinstance(a, DistArray) for a in inputs)
                assert(is_dist[0] or is_dist[1])
                for i in 0, 1:
                    if is_dist[i]:
                        ndim = inputs[i].ndim
                        assert(inputs[i]._distaxis == distaxis)
                        assert(inputs[i]._n == n)
                def _remote_ucall(inputs, **kwargs):
                    """(Executed on a remote or local engine) call the ufunc"""
                    return ufunc.__call__(inputs[0], inputs[1], **kwargs)
                results = []
                kwargs = kwargs.copy()
                kwargs['block'] = False
                kwargs['prefer_local'] = False
                for j in range(n):
                    subinputs = tuple(inputs[i]._subarrays[j] if 
                            is_dist[i] else inputs[i] for i in (0, 1))
                    results.append(call(_remote_ucall, subinputs, **kwargs))
                results = [convert_result(ar) for ar in results]
                return DistArray(results, distaxis)
    elif ufunc.nin > 2:
        raise Error(u'Distributing ufuncs with >2 inputs is not yet supported')
    else:
        raise Error(u'Distributed ufunc.%s() is not yet implemented' % method)


def transpose(a, axes=None):
    """Returns a view of the array with axes transposed.

    For a 1-D array, this has no effect.
    For a 2-D array, this is the usual matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted

    Args:
      a (array_like): Input array.
      axes (list of int, optional): By default, reverse the dimensions,
        otherwise permute the axes according to the values given.
    """
    if isinstance(a, np.ndarray):
        return np.transpose(a, axes)
    elif isinstance(a, RemoteArray):
        return a.transpose(*axes)
    elif isinstance(a, Remote):
        return _remote_to_array(a).transpose(*axes)
    elif isinstance(a, DistArray):
        if axes is None:
            axes = range(a.ndim - 1, -1, -1)
        axes = list(axes)
        if len(set(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")
        if sorted(axes) != list(range(a.ndim)):
            raise ValueError("axes don't match array")
        distaxis = a._distaxis
        new_distaxis = axes.index(distaxis)
        new_subarrays = [ra.transpose(*axes) for ra in a._subarrays]
        return DistArray(new_subarrays, new_distaxis)
    else:
        return np.transpose(a, axes)


def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards, until it lies in a given position.

    Args:
      a (array_like): Input array.
      axis (int): The axis to roll backwards.  The positions of the other axes 
        do not change relative to one another.
      start (int, optional): The axis is rolled until it lies before this 
        position.  The default, 0, results in a "complete" roll.

    Returns:
      res (ndarray)
    """
    if isinstance(a, np.ndarray):
        return np.rollaxis(a, axis, start)
    if axis not in range(a.ndim):
        raise ValueError(
                'rollaxis: axis (%d) must be >=0 and < %d' % (axis, a.ndim))
    if start not in range(a.ndim + 1):
        raise ValueError(
                'rollaxis: start (%d) must be >=0 and < %d' % (axis, a.ndim+1))
    axes = list(range(a.ndim))
    axes.remove(axis)
    axes.insert(start, axis)
    return transpose(a, axes)


def expand_dims(a, axis):
    """Insert a new axis, corresponding to a given position in the array shape

    Args:
      a (array_like): Input array.
      axis (int): Position (amongst axes) where new axis is to be inserted.
    """
    if hasattr(a, 'expand_dims') and hasattr(type(a), '__array_interface__'):
        return a.expand_dims(axis)
    else:
        return np.expand_dims(a, axis)


def _remote_to_array(remote):
    # Try to convert an arbitrary Remote object into a RemoteArray
    if isinstance(remote, RemoteArray):
        return remote
    else:
        return call(np.array, remote, prefer_local=False)


def concatenate(tup, axis=0):
    """Join a sequence of arrays together. 
    Will aim to join `ndarray`, `RemoteArray`, and `DistArray` without moving 
    their data, if they happen to be on different engines.

    Args:
      tup (sequence of array_like): Arrays to be concatenated. They must have
        the same shape, except in the dimension corresponding to `axis`.
      axis (int, optional): The axis along which the arrays will be joined.

    Returns: 
      res: `ndarray`, if inputs were all local
           `RemoteArray`, if inputs were all on the same remote engine
           `DistArray`, if inputs were already scattered on different engines
    """
    from distob import engine
    if len(tup) is 0:
        raise ValueError('need at least one array to concatenate')
    first = tup[0]
    others = tup[1:]
    # allow subclasses to provide their own implementations of concatenate:
    if (hasattr(first, 'concatenate') and 
            hasattr(type(first), '__array_interface__')):
        return first.concatenate(others, axis)
    # convert all arguments to arrays/RemoteArrays if they are not already:
    arrays = []
    for ar in tup:
        if isinstance(ar, DistArray):
            if axis == ar._distaxis:
                arrays.extend(ar._subarrays)
            else:
                # Since not yet implemented arrays distributed on more than
                # one axis, will fetch and re-scatter on the new axis:
                arrays.append(gather(ar))
        elif isinstance(ar, RemoteArray):
            arrays.append(ar)
        elif isinstance(ar, Remote):
            arrays.append(_remote_to_array(ar))
        elif hasattr(type(ar), '__array_interface__'):
            # then treat as a local ndarray
            arrays.append(ar)
        else:
            arrays.append(np.array(ar))
    if all(isinstance(ar, np.ndarray) for ar in arrays):
        return np.concatenate(arrays, axis)
    total_length = 0
    # validate dimensions are same, except for axis of concatenation:
    commonshape = list(arrays[0].shape)
    commonshape[axis] = None # ignore this axis for shape comparison
    for ar in arrays:
        total_length += ar.shape[axis]
        shp = list(ar.shape)
        shp[axis] = None
        if shp != commonshape:
            raise ValueError('incompatible shapes for concatenation')
    # set sensible target block size if splitting subarrays further:
    blocksize = ((total_length - 1) // engine.nengines) + 1
    rarrays = []
    for ar in arrays:
        if isinstance(ar, DistArray):
            rarrays.extend(ar._subarrays)
        elif isinstance(ar, RemoteArray):
            rarrays.append(ar)
        else:
            da = _scatter_ndarray(ar, axis, blocksize)
            for ra in da._subarrays:
                rarrays.append(ra)
            del da
    del arrays
    # At this point rarrays is a list of RemoteArray to be concatenated
    eid = rarrays[0]._id.engine
    if all(ra._id.engine == eid for ra in rarrays):
        # Arrays to be joined are all on the same engine
        if eid == engine.eid:
            # Arrays are all local
            return concatenate([gather(r) for r in rarrays], axis)
        else:
            return call(concatenate, rarrays, axis)
    else:
        # Arrays to be joined are on different engines.
        # TODO: consolidate any consecutive arrays already on same engine
        return DistArray(rarrays, axis)


def vstack(tup):
    """Stack arrays in sequence vertically (row wise), 
    handling ``RemoteArray`` and ``DistArray`` without moving data.

    Args:
      tup (sequence of array_like)

    Returns: 
      res: `ndarray`, if inputs were all local
           `RemoteArray`, if inputs were all on the same remote engine
           `DistArray`, if inputs were already scattered on different engines
    """
    # Follow numpy.vstack behavior for 1D arrays:
    arrays = list(tup)
    for i in range(len(arrays)):
        if arrays[i].ndim is 1:
            arrays[i] = arrays[i][np.newaxis, :]
    return concatenate(tup, axis=0)


def hstack(tup):
    """Stack arrays in sequence horizontally (column wise), 
    handling ``RemoteArray`` and ``DistArray`` without moving data.

    Args:
      tup (sequence of array_like)

    Returns: 
      res: `ndarray`, if inputs were all local
           `RemoteArray`, if inputs were all on the same remote engine
           `DistArray`, if inputs were already scattered on different engines
    """
    # Follow numpy.hstack behavior for 1D arrays:
    if all(ar.ndim is 1 for ar in tup):
        return concatenate(tup, axis=0)
    else:
        return concatenate(tup, axis=1)


def dstack(tup):
    """Stack arrays in sequence depth wise (along third dimension), 
    handling ``RemoteArray`` and ``DistArray`` without moving data.

    Args:
      tup (sequence of array_like)

    Returns: 
      res: `ndarray`, if inputs were all local
           `RemoteArray`, if inputs were all on the same remote engine
           `DistArray`, if inputs were already scattered on different engines
    """
    # Follow numpy.dstack behavior for 1D and 2D arrays:
    arrays = list(tup)
    for i in range(len(arrays)):
        if arrays[i].ndim is 1:
            arrays[i] = arrays[i][np.newaxis, :]
        if arrays[i].ndim is 2:
            arrays[i] = arrays[i][:, :, np.newaxis]
    return concatenate(arrays, axis=2)


def split(a, indices_or_sections, axis=0):
    if type(a) is np.ndarray: # deliberately excluding subclasses
        return np.split(a, indices_or_sections, axis)
    else:
        if not isinstance(indices_or_sections, numbers.Integral):
            raise Error('splitting by array of indices is not yet implemented')
        n = indices_or_sections
        if self.shape[axis] % n != 0:
            raise ValueError("Array split doesn't result in an equal division")
        step = self.shape[axis] / n
        pieces = []
        start = 0
        while start < self.shape[axis]:
            stop = start + step
            ix = [slice(None)] * self.ndim
            ix[axis] = slice(start, stop)
            ix = tuple(ix)
            pieces.append(self[ix])
            start += step
        return pieces


def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, axis=0)


def hsplit(a, indices_or_sections):
    return split(a, indices_or_sections, axis=1)


def dsplit(a, indices_or_sections):
    return split(a, indices_or_sections, axis=2)


def _broadcast_shape(*args):
    """Return the shape that would result from broadcasting the inputs"""
    #TODO: currently incorrect result if a Sequence is provided as an input
    shapes = [a.shape if hasattr(type(a), '__array_interface__')
              else () for a in args]
    ndim = max(len(sh) for sh in shapes) # new common ndim after broadcasting
    for i, sh in enumerate(shapes):
        if len(sh) < ndim:
            shapes[i] = (1,)*(ndim - len(sh)) + sh
    return tuple(max(sh[ax] for sh in shapes) for ax in range(ndim))


def broadcast_arrays(*args):
    if all(type(a) is np.ndarray or 
           not hasattr(type(a), '__array_interface__') for a in args):
        # deliberately excludes subclasses of ndarray
        return np.broadcast_arrays(*args)
    arrays = [a if hasattr(type(a), '__array_interface__') 
              else np.array(a) for a in args]
    shapes = [a.shape for a in arrays]
    ndim = max(len(sh) for sh in shapes) # new common ndim after broadcasting
    for i in range(len(arrays)):
        old_ndim = len(shapes[i])
        if old_ndim < ndim:
            ix = (np.newaxis,)*(ndim - old_ndim) + (slice(None),)*old_ndim
            arrays[i] = arrays[i][ix]
            shapes[i] = arrays[i].shape
    newshape = tuple(max(sh[ax] for sh in shapes) for ax in range(ndim))
    # Now broadcast arrays using fancy indexing. This is wasteful of both
    # memory and CPU, as it actually creates the redundant arrays instead of
    # making views with zero stride, but it works universally on classes that
    # implement fancy indexing. (TODO: Broadcast efficiently by having each
    # class implement its own `broaden(newshape)` method that returns views)
    for i in range(len(arrays)):
        for ax in range(ndim):
            if shapes[i][ax] == newshape[ax]:
                pass
            elif shapes[i][ax] == 1:
                ix = ((slice(None),) * ax + 
                      (np.zeros(newshape[ax], dtype=np.int64),) + 
                      (slice(None),) * (ndim - ax - 1))
                arrays[i] = arrays[i][ix]
            else:
                raise ValueError(u'shape mismatch: two or more arrays have '
                                  'incompatible dimensions on axis %d' % ax)
    return arrays


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    m : ndarray, see dtype parameter above

    Notes
    -----
    np.mean fails to pass the keepdims parameter to ndarray subclasses.
    That is the main reason we implement this function.
    """
    if (isinstance(a, np.ndarray) or
            isinstance(a, RemoteArray) or
            isinstance(a, DistArray)):
        return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    else:
        return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
