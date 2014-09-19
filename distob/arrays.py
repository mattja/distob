"""Remote and distributed numpy arrays
"""

from __future__ import absolute_import
from .distob import (proxy_methods, Remote, Ref, Error, 
                     scatter, gather, vectorize)
import numpy as np
from collections import Sequence
import numbers
import types
import warnings
import copy

# types for compatibility across python 2 and 3
_SliceType = type(slice(None))
_EllipsisType = type(Ellipsis)
_TupleType = type(())
_NewaxisType = type(np.newaxis)


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

    ndim = property(fget=lambda self: len(self._array_intf['shape']))

    size = property(fget=lambda self: np.prod(self._array_intf['shape']))

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

    def expand_dims(self, axis):
        """Insert a new axis, at a given position in the array shape
        Args:
          axis (int): Position (amongst axes) where new axis is to be inserted.
        """
        ix = [slice(None)] * self.ndim
        ix.insert(axis, np.newaxis)
        ix = tuple(ix)
        return self[ix]

    def concatenate(self, tup, axis=0):
        """Join a sequence of arrays to this one.

        Will aim to join `ndarray`, `RemoteArray`, and `DistArray` without
        moving their data if they are on different engines.

        Args: 
          tup (sequence of array_like): Arrays to be joined with this one.
            They must have the same shape as this array, except in the
            dimension corresponding to `axis`.
          axis (int, optional): The axis along which arrays will be joined.

        Returns:
          res `RemoteArray`, if arrays were all on the same engine
              `DistArray`, if inputs were on different engines
        """
        if not isinstance(tup, Sequence):
            tup = (tup,)
        if tup is (None,) or len(tup) is 0:
            return self
        tup = (self,) + tuple(tup)
        # convert all arguments to RemoteArrays if they are not already:
        arrays = []
        for ar in tup:
            if isinstance(ar, DistArray):
                if axis == ar._distaxis:
                    arrays.extend(ar._subarrays)
                else:
                    # Since not yet implemented arrays distributed on more than
                    # one axis, will fetch and re-scatter on the new axis.
                    ar = gather(ar)
                    arrays.extend(np.split(ar, ar.shape[axis], axis))
            elif isinstance(ar, RemoteArray):
                arrays.append(ar)
            elif isinstance(ar, Remote):
                arrays.append(_remote_to_array(ar))
            elif (not isinstance(ar, Remote) and 
                    not isinstance(ar, DistArray) and
                    not hasattr(type(ar), '__array_interface__')):
                arrays.append(np.array(ar))
        rarrays = scatter(arrays) # TODO: this will only work on the client
        # At this point rarrays is a list of RemoteArray to be concatenated
        refs = [ra._ref for ra in rarrays]
        eng_id = refs[0].engine_id
        if all(ref.engine_id == eng_id for ref in refs):
            # Arrays to be joined are all on the same engine
            from distob import engine
            if eng_id == engine.id:
                # All are on local host. Concatenate the local objects:
                return concatenate([engine[r.object_id] for r in refs], axis)
            else:
                # All are on the same remote host. Concatenate to a RemoteArray
                ids = [ref.object_id for ref in refs]
                dv = engine._client[eng_id]
                def remote_concat(ids, axis):
                    from distob import engine
                    ar = np.concatenate([engine[id] for id in ids], axis)
                    return Ref(ar)
                ref = dv.apply_sync(remote_concat, ids, axis)
                return RemoteArray(ref)
        else:
            # Arrays to be joined are on different engines.
            concatenation_axis_lengths = [ra.shape[axis] for ra in rarrays]
            if not all(n == 1 for n in concatenation_axis_lengths):
                #TODO: should do it remotely, using distob.split()
                msg = (u'Concatenating remote axes with lengths other than 1.'
                        'current implementation will fetch all data locally!')
                warnings.warn(msg, RuntimeWarning)
                array = concatenate(gather(rarrays), axis)
                return scatter(array, axis)
            else:
                return DistArray(rarrays, axis)


class DistArray(object):
    """Local object representing a single array distributed on multiple engines

    Currently only one of the axes will be distributed.
    """
    def __init__(self, subarrays, axis=None):
        """Make a DistArray from a list of existing remote numpy.ndarrays

        Args:
          subarrays (list of RemoteArray, or list of Ref to ndarray): 
            the subarrays (possibly remote) which form the whole array when
            concatenated. The subarrays must all have the same shape and dtype.
            Currently must have `a.shape[axis] == 1` for each subarray `a`
          axis (int, optional): Position of the distributed axis, which is the 
            axis along which the subarrays will be concatenated. Default is 
            the last axis.  
        """
        self._n = len(subarrays)  # Length of the distributed axis.
        if self._n < 2:
            raise ValueError('must provide more than one subarray')
        self._obcache = None
        self._obcache_current = False
        # In the subarrays list, accept either RemoteArray or Ref to ndarray:
        subarrays = [ra if isinstance(ra, RemoteArray)
                     else RemoteArray(ra) for ra in subarrays]
        first_ra_metadata = subarrays[0]._ref.metadata
        if not all(ra._ref.metadata == first_ra_metadata for ra in subarrays):
            raise ValueError('subarrays must have same shape, strides & dtype')
        descr, subshape, substrides, typestr = first_ra_metadata
        itemsize = int(typestr[2:])
        if axis is None:
            self._distaxis = len(subshape) - 1
        else:
            self._distaxis = axis
        # For numpy strides, `None` means assume C-style ordering:
        if substrides is None:
            substrides = tuple(int(np.product(
                    subshape[i:])*itemsize) for i in range(1, len(subshape)+1))
        shape = list(subshape)
        shape[self._distaxis] = self._n
        shape = tuple(shape)
        # For now, report the same strides as used by the subarrays
        strides = substrides
        self.dtype = np.dtype(typestr)
        self._subarrays = subarrays
        # a surrogate ndarray to help with slicing of the distributed axis:
        self._placeholders = np.array(range(len(subarrays)), dtype=int)
        # implement the python array interface
        self._array_intf = {'descr': descr, 'shape': shape, 'strides': strides,
                            'typestr': typestr, 'version': 3}

    def _fetch(self):
        """update local cached copy of the real object"""
        if not self._obcache_current:
            ax = self._distaxis
            self._obcache = concatenate([ra._ob for ra in self._subarrays], ax)
            self._obcache_current = True

    def __ob(self):
        """return a copy of the real object"""
        self._fetch()
        return self._obcache

    _ob = property(fget=__ob)

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

    size = property(fget=lambda self: np.prod(self._array_intf['shape']))

    strides = property(fget=lambda self: self._array_intf['strides'])

    def __repr__(self):
        classname = self.__class__.__name__
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
        """Route ufunc execution intelligently to local host or remote engine 
        depending on where the arguments reside.
        """
        print('In DistArray __numpy_ufunc__')
        print('ufunc:%s; method:%s; selfpos:%d' % (repr(ufunc), method, i))
        print('inputs:%s; kwargs:%s' % (inputs, kwargs))
        raise Error("ufunc=%s Haven't implemented ufunc support yet!" % ufunc)

    def __array_prepare__(self, in_arr, context=None):
        """Fetch underlying data to user's computer and apply ufunc locally.
        Only used as a fallback, for numpy versions < 1.9 which lack 
        support for the __numpy_ufunc__ mechanism. 
        """
        print('In DistArray __array_prepare__')
        #TODO fetch data here
        if map(int, np.version.short_version.split('.')) < [1,9,0]:
            raise Error('Numpy version 1.9.0 or later is required.')
        else:
            raise Error('Numpy is current, but still called __array_prepare__')

    def __array_wrap__(self, out_arr, context=None):
        print('In DistArray __array_wrap__')
        return np.ndarray.__array_wrap__(out_arr, context)

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
            ixlist = self._placeholders[distix]
            if isinstance(ixlist, numbers.Number):
                # distributed axis has been sliced away: return a RemoteArray
                subix = index[0:distaxis] + (0,) + index[(distaxis+1):]
                return self._subarrays[ixlist][subix]
            some_subarrays = [self._subarrays[i] for i in ixlist]
            subix = index[0:distaxis] + (slice(None),) + index[(distaxis+1):]
            result_ras = [ra[subix] for ra in some_subarrays]
            if len(result_ras) is 1:
                # no longer distributed: return a RemoteArray
                return result_ras[0]
            else:
                axes_removed = sum(1 for x in index[:distaxis] if isinstance(
                        x, numbers.Integral))
                new_distaxis = distaxis - axes_removed
                return DistArray(result_ras, new_distaxis)
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
                index[fancy_pos[j]] = ix_arrays[j]
            index = tuple(index)
            idim = index[fancy_pos[0]].ndim # common ndim of all index arrays
            assert(idim > 0)
            distix = index[distaxis]
            otherix = index[0:distaxis] + (slice(None),) + index[(distaxis+1):]
            if not is_fancy[distaxis]:
                # fancy indexing is only being applied to non-distributed axes
                ixlist = self._placeholders[distix]
                if isinstance(ixlist, numbers.Number):
                    # distributed axis was sliced away: return a RemoteArray
                    subix = index[0:distaxis] + (0,) + index[(distaxis+1):]
                    return self._subarrays[ixlist][subix]
                some_subarrays = [self._subarrays[i] for i in ixlist]
                # predict where that new axis will be in subarrays post-slicing
                if contiguous:
                    if fancy_pos[0] > distaxis:
                        new_distaxis = distaxis
                    else:
                        new_distaxis = distaxis - len(fancy_pos) + idim
                else:
                    earlier_fancy = len([i for i in fancy_pos if i < distaxis])
                    new_distaxis = distaxis - earlier_fancy + idim
                result_ras = [ra[otherix] for ra in some_subarrays]
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
                    some_subarrays = [self._subarrays[i] for i in ixlist]
                    if contiguous:
                        new_distaxis = fancy_pos[0] + iax
                    else:
                        new_distaxis = 0 + iax
                    result_ras = []
                    for i in range(distix.shape[iax]):
                        # Slice the original indexing arrays into smaller
                        # arrays, suitable for indexing our subarrays.
                        sl = [slice(None)] * idim
                        sl[iax] = slice(i, i+1)
                        sl = tuple(sl)
                        subix = list(index)
                        for j in range(len(subix)):
                            if isinstance(subix[j], np.ndarray):
                                subix[j] = subix[j][sl]
                                if j == distaxis:
                                    subix[j] = np.zeros_like(subix[j])
                        subix = tuple(subix)
                        result_ras.append(some_subarrays[i][subix])
                    if all_same_engine and len(result_ras) > 1:
                        rs = [expand_dims(r, new_distaxis) for r in result_ras]
                        return concatenate(rs, new_distaxis) # one RemoteArray
                else:
                    # Have more than one nonconstant indexing axis in the
                    # indexing array that applies to our distributed axis.
                    msg = (u'The requested slicing operation requires moving '
                            'data between engines. Will fetch data locally.')
                    warnings.warn(msg, RuntimeWarning)
                    return self._ob[index]
            if len(result_ras) is 1:
                # no longer distributed: return a RemoteArray
                return result_ras[0]
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
            remove_axis = ((slice(None),)*(self._distaxis) + (0,) + 
                           (slice(None),)*(self.ndim - self._distaxis - 1))
            refs = [ra[remove_axis]._ref for ra in self._subarrays]
            from distob import engine
            dv = engine._client[:]
            def remote_f(object_id, *args, **kwargs):
                import distob
                result = f(distob.engine[object_id], *args, **kwargs)
                if type(result) in distob.engine.proxy_types:
                    return Ref(result)
                else:
                    return result
            results = []
            for ref in refs:
                dv.targets = ref.engine_id
                ar = dv.apply_async(remote_f, ref.object_id, *args, **kwargs)
                results.append(ar)
            for i in range(len(results)):
                ar = results[i]
                ar.wait()
                results[i] = ar.r
                if isinstance(results[i], Ref):
                    ref = results[i]
                    RemoteClass = engine.proxy_types[ref.type]
                    results[i] = RemoteClass(ref)
            if (all(isinstance(r, RemoteArray) for r in results) and
                    all(r.shape == results[0].shape for r in results)):
                # Then we can join the results and return a DistArray.
                # To position result distaxis, match input shape where possible
                old_subshape = (self.shape[0:self._distaxis] +
                                self.shape[(self._distaxis+1):])
                res_subshape = list(results[0].shape)
                pos = len(res_subshape)
                for i in range(len(old_subshape)):
                    n = old_subshape[i]
                    if n not in res_subshape:
                        continue
                    pos = res_subshape.index(n)
                    res_subshape[pos] = None
                    if i >= self._distaxis:
                        break
                    pos += 1
                new_distaxis = pos
                results = [r.expand_dims(new_distaxis) for r in results]
                return DistArray(results, new_distaxis)
            elif all(isinstance(r, numbers.Number) for r in results):
                return np.array(results)
            else:
                return results  # list
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
        if axis <= self._distaxis:
            subaxis = axis
            new_distaxis = self._distaxis + 1
        else:
            subaxis = axis - 1
            new_distaxis = self._distaxis
        new_subarrays = [expand_dims(ra, subaxis) for ra in self._subarrays]
        return DistArray(new_subarrays, new_distaxis)

    def concatenate(self, tup, axis=0):
        """Join a sequence of arrays to this one.

        Will aim to join `ndarray`, `RemoteArray`, and `DistArray` without
        moving their data if they are on different engines.

        Args: 
          tup (sequence of array_like): Arrays to be joined with this one.
            They must have the same shape as this array, except in the
            dimension corresponding to `axis`.
          axis (int, optional): The axis along which arrays will be joined.

        Returns:
          res `RemoteArray`, if arrays were all on the same engine
              `DistArray`, if inputs were on different engines
        """
        if not isinstance(tup, Sequence):
            tup = (tup,)
        if tup is (None,) or len(tup) is 0:
            return self
        if axis == self._distaxis:
            split_self = self._subarrays
        else:
            # Since we have not yet implemented arrays distributed on more than
            # one axis, will fetch subarrays and re-scatter on the new axis.
            split_self = split(self._ob, self.shape[axis], axis)
        first = split_self[0]
        others = tuple(split_self[1:]) + tup
        first = scatter(first)
        return first.concatenate(others, axis)

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the arithmetic mean along the specified axis."""
        if axis is None:
            if out is not None:
                out[:] = vectorize(np.mean)(self, None, dtype).mean()
                return out
            else:
                return vectorize(np.mean)(self, None, dtype).mean()
        elif axis == self._distaxis:
            return np.mean(self._ob, axis, dtype, out)
        else:
            if axis > self._distaxis:
                axis -= 1
            if out is not None:
                out[:] = vectorize(np.mean)(self, axis, dtype)
                return out
            else:
                return vectorize(np.mean)(self, axis, dtype)


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
    if isinstance(a, np.ndarray):
        return np.expand_dims(a, axis)


def _remote_to_array(remote):
    # Try to convert an arbitrary Remote object into a RemoteArray
    if isinstance(remote, RemoteArray):
        return remote
    else:
        from distob import engine
        dv = engine._client[remote._ref.engine_id]
        def remote_array(object_id):
            return Ref(np.array(engine[object_id]))
        return RemoteArray(dv.apply_sync(remote_array, remote._ref.object_id))


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
    if len(tup) is 0:
        raise ValueError('need at least one array to concatenate')
    first = tup[0]
    others = tup[1:]
    if (hasattr(first, 'concatenate') and 
            hasattr(type(first), '__array_interface__')):
        # This case includes RemoteArray and DistArray
        return first.concatenate(others, axis)
    if all(isinstance(ar, np.ndarray) for ar in tup):
        return np.concatenate(tup, axis)
    if isinstance(first, Remote):
        first = _remote_to_array(first)
    else:
        first = scatter(np.array(first)) #TODO: this only works on the client
    return first.concatenate(others, axis)


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


def broadcast_arrays(*args):
    if all(type(a) is np.ndarray for a in args): # exclude subclasses
        return np.broadcast_arrays(*args)
    shapes = [a.shape for a in args]
    ndim = max(len(sh) for sh in shapes) # new common ndim after broadcasting
    arrays = list(args)
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
