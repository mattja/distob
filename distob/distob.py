# Copyright 2014 Matthew J. Aburn
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""
This module implements the core functionality of distob.

Functions:
  scatter(obj)  Distribute an object to remote iPython engines, return a proxy.
  gather(obj)  Fetch back a distributed object, making it local again.
  call_all(sequence, methodname, *args, **kwargs) Call method on each element. 

Classes:
  Remote   base class, used when creating Remote* proxy classes
  @proxy_methods(base)   class decorator for creating Remote* proxy classes
  ObjectHub  dict interface giving refs to all distributed objects cluster-wide
  ObjectEngine  dict holding the distributed objects of a single IPython engine
  Ref  reference to a (possibly remote) object
"""

from __future__ import absolute_import
from IPython import parallel
import distob
from . import _pylru
import types
import copy
import importlib
import collections


class Error(Exception):
    pass


class DistobTypeError(Error):
    pass


class DistobValueError(Error):
    pass


class DistobClusterError(Error):
    """Thrown if there is a problem using the cluster that we can't fix"""
    pass


class Ref(object):
    """Reference to a (possibly remote) object.

    a Ref can exist on the same IPython engine that owns the object, or on 
    any python instance that has an ObjectHub (such as the IPython client)

    Attributes:
      engine_id (int): The IPython engine number where the object is held. 
        -1 means that the object is held on the IPython client.
      object_id (str): An object id string that is unique cluster-wide.
      type: type of the remote object
      metadata (tuple, optional): brief extra information about the object 
        may be defined by specific Remote* classes, to help set up access.
    """
    def __init__(self, obj):
        if isinstance(obj, Ref):
            self.engine_id = obj.engine_id
            self.object_id = obj.object_id
            self.type = obj.type
            self.metadata = obj.metadata
            distob.engine.incref(self.object_id)
        else:
            self.engine_id = distob.engine.id
            self.type = type(obj)
            if self.type in distob.engine.proxy_types:
                self.metadata = \
                        distob.engine.proxy_types[self.type].__pmetadata__(obj)
            else:
                self.metadata = None
            self.object_id = '%s:%s@e%d' % (
                self.type.__name__, hex(id(obj)), self.engine_id)
            distob.engine[self.object_id] = obj

    def __getstate__(self):
        # pickled or copied Ref is included in refcount
        distob.engine.incref(self.object_id)
        return (self.engine_id, self.object_id, self.type.__name__,
                self.type.__module__, self.metadata)

    def __setstate__(self, state):
        self.engine_id,self.object_id,typename,modname,self.metadata = state
        module = importlib.import_module(modname)
        self.type = getattr(module, typename)

    def __del__(self):
        distob.engine.decref(self.object_id)

    def __repr__(self):
        return u'<Ref to %s>' % self.object_id


class ObjectEngine(dict):
    """A dict holding the distributed objects of a single IPython engine.

    It maintains cluster-wide reference counts of the objects it holds.
    There is one global instance on each engine.

    For each dict entry,
    Key is an object id string (unique cluster-wide)
    Value is the actual object that is to be accessed remotely.

    Attributes:
      id (int): IPython engine id number of this engine. -1 means this is
        an IPython client, rather than an engine.
      refcounts (dict): Key is an object id string (for an object held on
        this engine). Value is the cluster-wide count of how many Ref
        instances exist pointing to that object.
      proxy_types (dict): When an object's methods are called remotely, some
        return types should be returned by proxy instead of returning a 
        full object. This dict holds mappings {real_type: proxy_type} for 
        these types.
    """
    def __init__(self, engine_id):
        self.id = engine_id 
        self.refcounts = dict()
        self.proxy_types = dict()
        super(ObjectEngine, self).__init__()
        # TODO: The following two lines are a workaround until we can fix issue
        #       #58 in dill re transferring dynamically generated classes.
        for realtype, proxytype in ObjectHub._initial_proxy_types.items():
            self._singleeng_reg_proxy_type(realtype, proxytype)

    # TODO temporary function to be removed when we fix the problems with
    # transfering dynamically generated classes via dill (dill issue #58).
    # In the meantime, use pre-defined classes only.
    def _singleeng_reg_proxy_type(self, real_type, proxy_type):
        self.proxy_types[real_type] = proxy_type

    def incref(self, key):
        self.refcounts[key] += 1

    def decref(self, key):
        if self.refcounts[key] is 1:
            self.__delitem__(key)
        else:
            self.refcounts[key] -= 1
    
    def __setitem__(self, key, value):
        if key in self.keys():
            if value is not self[key]:
                raise DistobValueError('object id not unique: %s' % key)
            self.incref(key)
        else:
            super(ObjectEngine, self).__setitem__(key, value)
            self.refcounts[key] = 1

    def __delitem__(self, key):
        if self.refcounts[key] is 1:
            super(ObjectEngine, self).__delitem__(key)
            del self.refcounts[key]
            print('Cleaned up and deleted object %s' % key)
        else:
            self.decref(key)

    def __repr__(self):
        return '<%s instance on engine %d>:\n%s' % (
            self.__class__.__name__, self.id, repr(self.keys()))

    def _repr_pretty_(self, p, cycle):
        return '<%s instance on engine %d>:\n%s' % (
            self.__class__.__name__, self.id, p.pretty(self.keys()))


# TODO put this back inside ObjectHub after dill issue #58 is resolved.
def _remote_reg_proxy_type(real_t, proxy_t):
    distob.engine.proxy_types[real_t] = proxy_t


class ObjectHub(ObjectEngine):
    """A dict providing references to all distributed objects across the cluster

    Key is an object id string (unique cluster-wide)
    Value is the actual object (if local) or a Ref to the object (if remote)

    Operations supported are get and delete.

    Attributes:
      id (int): IPython engine id number of this engine. -1 means this is
        an IPython client, rather than an engine.
      refcounts (dict): Key is an object id string (for an object held on
        this engine). Value is the cluster-wide count of how many Ref
        instances exist pointing to that object.
      proxy_types (dict): When an object's methods are called remotely, some
        return types should be returned by proxy instead of returning a 
        full object. This dict holds mappings {real_type: proxy_type} for 
        these types.
    """
    _initial_proxy_types = dict()

    def __init__(self, engine_id, client):
        """Make an ObjectHub.
        Args:
          engine_id: -1 if this hub is on the client, otherwise this engine's id
          client: IPython.parallel.client
        """
        self._client = client
        self._dv = client.direct_view(targets='all')
        self._dv.use_dill()
        super(ObjectHub, self).__init__(engine_id)
        # TODO: restore following two lines after issue #58 in dill is fixed.
        #for realtype, proxytype in self.__class__._initial_proxy_types.items():
        #    self._runtime_reg_proxy_type(realtype, proxytype)

    @classmethod
    def register_proxy_type(cls, real_type, proxy_type):
        """Configure engines so that remote methods returning values of type
        `real_type` will instead return by proxy, as type `proxy_type`
        """
        if distob.engine is None:
            cls._initial_proxy_types[real_type] = proxy_type
        elif isinstance(distob.engine, ObjectHub):
            distob.engine._runtime_reg_proxy_type(real_type, proxy_type)
        else:
            # TODO: remove next line after issue #58 in dill is fixed.
            distob.engine._singleeng_reg_proxy_type(real_type, proxy_type)
            pass

    def _runtime_reg_proxy_type(self, real_type, proxy_type):
        #print('about to do runtime reg of %s ' % proxy_type)
        self.proxy_types[real_type] = proxy_type
        ar = self._dv.apply(_remote_reg_proxy_type, real_type, proxy_type)
        self._dv.wait(ar)
        pass

    def incref(self, key):
        engine_id = int(key[(key.rindex('@e')+2):])
        if engine_id is self.id:
            super(ObjectHub, self).incref(key)
        else:
            self._dv.execute('distob.engine.incref("%s")' % key, 
                             targets=engine_id)

    def decref(self, key):
        engine_id = int(key[(key.rindex('@e')+2):])
        if engine_id is self.id:
            super(ObjectHub, self).decref(key)
        else:
            self._dv.execute('distob.engine.decref("%s")' % key, 
                             targets=engine_id)

    def __delitem__(self, key):
        engine_id = int(key[(key.rindex('@e')+2):])
        if engine_id is self.id:
            super(ObjectHub, self).__delitem__(key)
        else:
            self._dv.execute('del distob.engine["%s"]' % key, targets=engine_id)

    def __getitem__(self, key):
        engine_id = int(key[(key.rindex('@e')+2):])
        if engine_id is self.id:
            return super(ObjectHub, self).__getitem__(key)
        else:
            def fetch_object(object_id):
                return Ref(distob.engine[object_id])
            self._dv.targets = engine_id
            res = self._dv.apply_sync(fetch_object, key)
            self._dv.targets = 'all'
            return res

    def __setitem__(self, key, value):
        engine_id = int(key[(key.rindex('@e')+2):])
        if engine_id is self.id:
            super(ObjectHub, self).__setitem__(key, value)
        else:
            raise Error('Setting remote objects via ObjectHub not implemented')

    def keys(self):
        local_keys = super(ObjectHub, self).keys()
        def fetch_keys():
            return distob.engine.keys()
        ar = self._dv.apply_async(fetch_keys)
        self._dv.wait(ar)
        remote_keys = [key for sublist in ar.r for key in sublist]
        return remote_keys + local_keys


def _remote_setup_engine(engine_id):
    if distob.engine is None:
        distob.engine = distob.ObjectEngine(engine_id)
    # TODO these imports should be unnecessary with improved deserialization
    import numpy as np
    from scipy import stats
    # TODO Using @parallel.interactive still did not import to __main__
    #      so will do it this way for now.
    import __main__
    __main__.__dict__['np'] = np
    __main__.__dict__['stats'] = stats


def _setup_engines(client=None):
    """Prepare all iPython engines for distributed object processing.

    Args:
      client (IPython.parallel.client, optional): If None, will create a client         using the default IPython profile.
    """
    if not client:
        client = parallel.Client()
    ids = client.ids
    if not ids:
        raise DistobClusterError('No IPython compute engines are available')
    dv = client[ids]
    dv.use_dill()
    with dv.sync_imports(quiet=True):
        import distob
    # create global ObjectEngine distob.engine on each engine
    ars = []
    for i in ids:
        dv.targets = i
        ars.append(dv.apply_async(_remote_setup_engine, i))
    dv.wait(ars)
    for ar in ars:
        if not ar.successful():
            raise ar.r
    # create global ObjectHub distob.engine on the client host
    if distob.engine is None:
        distob.engine = ObjectHub(-1, client)


class Remote(object):
    """Base class for Remote* proxy classes.
    An instance of a Remote subclass is a proxy designed to control
    an instance of an existing class which may be local or on a remote host.

    To make a Remote proxy class, inherit from `Remote` as first base class.
    If the instances of the new Remote* proxy class should be treated 
    exactly like instances of the controlled class, then also inherit from 
    the controlled class as the second base:

        @proxy_methods(Tree)
        class RemoteTree(Dist, Tree)

    Use the decorator @proxy_methods() to register the new Remote proxy class
    and to specify which methods/attributes of the existing class should be
    proxied. (by default all except those starting with an underscore)

    When each instance is created, it will also check for any instance-specific
    methods and attributes that exist on the controlled object.
    (For example attributes created by the __init__ of the controlled object.)
    Corresponding methods/attributes will be added to the Remote* instance to
    provide remote access to these as well.
    """
    _include_underscore = ()
    _exclude = ()

    def __init__(self, obj, client):
        """Set up the Remote* proxy object to access an already-existing object,
        which may be local or remote.

        Args:
          obj (Ref or object): either a Ref reference to the (possibly remote) 
            object to be controlled, or else an actual (local) object to be 
            controlled.
          client (IPython.parallel.client)
        """
        self._client = client
        if isinstance(obj, Ref):
            self._ref = obj
            self.is_local = (self._ref.engine_id is distob.engine.id)
        else:
            self._ref = Ref(obj)
            self.is_local = True
        if self.is_local:
            self._dv = None
            self._obcache = distob.engine[self._ref.object_id]
            self._obcache_current = True
        else:
            self._dv = client[self._ref.engine_id]
            self._dv.use_dill()
            self._obcache = None
            self._obcache_current = False
        self._id = self._ref.object_id
        #Add proxy controllers for any instance-specific methods/attributes:
        (instance_methods, instance_attribs) = self._scan_instance()
        for name, doc in instance_methods:
            def make_proxy_method(method_name, doc):
                def method(self, *args, **kwargs):
                    if self._obcache_current:
                        return apply(getattr(self._obcache, method_name),
                                     *args, **kwargs)
                    else:
                        return cls._try_cached_apply(self, method_name,
                                                     *args, **kwargs)
                method.__doc__ = doc
                method.__name__ = method_name
                #return types.MethodType(method, None, cls)
                return method
            setattr(self, name, make_proxy_method(name, doc))
        for name in instance_attribs:
            def make_property(attrib_name):
                # TODO: implement fset and fdel (requires writeback cache)
                def getter(self):
                    if self._obcache_current:
                        return getattr(self._obcache, attrib_name)
                    else:
                        return self._cached_apply('__getattribute__', 
                                                  attrib_name) 
                prop = property(fget=getter)
                return prop
            setattr(self.__class__, name, make_property(name))

    @classmethod
    def _local_scan_instance(cls, object_id, include_underscore, exclude):
        method_info = []
        attributes = []
        obj = distob.engine[object_id]
        if hasattr(obj, '__dict__'):
            for name in obj.__dict__:
                if (name not in exclude and 
                    (name[0] != '_' or 
                     include_underscore is True or
                     name in include_underscore)):
                    f = obj.__dict__[name]
                    if callable(f) and not isinstance(f, type):
                        method_info.append((name, f.__doc__))
                    else:
                        attributes.append(name)
        return (method_info, attributes)

    def _scan_instance(self):
        """get information on instance-methods/attributes of the object"""
        if not self.is_local:
            return self._dv.apply_sync(Remote._local_scan_instance,
                                       self._ref.object_id,
                                       self.__class__._include_underscore,
                                       self.__class__._exclude)

    def _apply(self, method_name, *args, **kwargs):
        """Call a method on the remote object without caching."""
        def remote_call_method(object_id, method_name, *args, **kwargs):
            obj = distob.engine[object_id]
            result = obj.__getattribute__(method_name)(*args, **kwargs)
            if type(result) in distob.engine.proxy_types:
                return Ref(result)
            else:
                return result
        r = self._dv.apply_sync(remote_call_method, self._id, 
                                method_name, *args, **kwargs)
        if isinstance(r, Ref):
            RemoteClass = distob.engine.proxy_types[r.type]
            return RemoteClass(r, self._client)
        else:
            return r

    @_pylru.lrudecorator(1000)
    def _cached_apply(self, method_name, *args, **kwargs):
        #print('cache miss. self:%s(%s); method:%s; args:%s; kwargs:%s' % (
        #    hex(id(self)), self._ref.object_id, method_name, 
        #    repr(args), repr(kwargs)))
        return self._apply(method_name, *args, **kwargs)

    def _try_cached_apply(self, method_name, *args, **kwargs):
        """Call a method on the remote object. Cache results if args hashable.
        Args:
          method_name (str): name of the remote method to call 
          *args, **kwargs: arguments to pass to the remote method
        Returns:
          ``<remote_object>.<method_name>(*args, **kwargs)``
          If the output is of a type that should be proxied, returns 
          a ``Remote*`` proxy object instead of the real object.
        """
        #print('try cached: self:%s(%s); method:%s; args:%s; kwargs:%s' % (
        #   hex(id(self)), self._ref.object_id, method_name, 
        #   repr(args), repr(kwargs)))
        #
        #For some reason immutable slice objects are not hashable in python.
        #TODO workaround this by recursively replacing any slices in dict key.
        try:
            return self._cached_apply(method_name, *args, **kwargs)
        except TypeError as te:
            if te.message[:15] == 'unhashable type':
                #print("unhashable. won't be able to cache")
                return self._apply(method_name, *args, **kwargs)
            else:
                raise

    def _apply_async(self, method_name, *args, **kwargs):
        """Call a method on the remote object without caching.
        Returns: 
          ipython.parallel.AsyncResult: async output of the remote method
        """
        def remote_call_method(object_id, method_name, *args, **kwargs):
            obj = distob.engine[object_id]
            result = obj.__getattribute__(method_name)(*args, **kwargs)
            if type(result) in distob.engine.proxy_types:
                return Ref(result)
            else:
                return result
        ar = self._dv.apply_async(remote_call_method, self._id, 
                                  method_name, *args, **kwargs)
        return ar
        if isinstance(r, Ref):
            RemoteClass = distob.engine.proxy_types[r.type]
            return RemoteClass(r, self._client)
        else:
            return r

    def _fetch(self):
        """update local cached copy of the real object"""
        if not self.is_local and not self._obcache_current:
            #print('fetching data from %s' % self._ref.object_id)
            self._obcache = self._dv['distob.engine["%s"]' % self._id]
            self._obcache_current = True

    def __ob(self):
        """return a copy of the real object"""
        self._fetch()
        return self._obcache

    _ob = property(fget=__ob)

    def __copy__(self):
        newref = copy.copy(self._ref)
        obj = self.__class__.__new__(self.__class__, newref, self._client)
        obj.__init__(newref, self._client)
        return obj

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def __pmetadata__(cls, obj):
        """Subclasses can override this method if they need Ref to provide
        more metadata on the controlled object to help configure remote access.
        Arguments:
          obj: object of type ``self._ref.type``
        """
        return None


def proxy_methods(base, include_underscore=None, exclude=None, supers=True):
    """class decorator. Modifies `Remote` subclasses to add proxy methods and
    attributes that mimic those defined in class `base`.

    Example:

        @proxy_methods(Tree)
        class RemoteTree(Remote, Tree)

    The decorator registers the new proxy class and specifies which methods
    and attributes of class `base` should be proxied via a remote call to
    a real object, and which methods/attributes should not be proxied but
    instead called directly on the instance of the proxy class.

    By default all methods and attributes of the class `base` will be
    proxied except those starting with an underscore.

    The MRO of the decorated class is respected:
    Any methods and attributes defined in the decorated class
    (or in other bases of the decorated class that do not come after `base`
     in its MRO) will override those added by this decorator,
    so that `base` is treated like a base class.

    Args:
      base (type): The class whose instances should be remotely controlled.
      include_underscore (sequence of str): Names of any methods or attributes 
        that start with an underscore but should be proxied anyway.
      exclude (sequence of str): Names of any methods or attributes that 
        should not be proxied.
      supers (bool): Proxy methods and attributes defined in superclasses 
        of ``base``, in addition to those defined directly in class ``base``
    """
    if isinstance(include_underscore, str):
        include_underscore = (include_underscore,)
    if isinstance(exclude, str):
        exclude = (exclude,)
    if not include_underscore:
        include_underscore = ()
    if not exclude:
        exclude = ()
    def rebuild_class(cls):
        # Identify any bases of cls that do not come after `base` in the list:
        bases_other = list(cls.__bases__)
        if bases_other[-1] is object:
            bases_other.pop()
        if base in bases_other:
            bases_other = bases_other[:bases_other.index(base)]
        if not issubclass(cls.__bases__[0], Remote):
            raise DistobTypeError('First base class must be subclass of Remote')
        if not issubclass(base, object):
            raise DistobTypeError('Only new-style classes currently supported')
        dct = cls.__dict__.copy()
        if cls.__doc__ is None or '\n' not in cls.__doc__:
            base_doc = base.__doc__
            if base_doc is None:
                base_doc = ''
            dct['__doc__'] = """Local object representing a remote %s
                  It can be used just like a %s object, but behind the scenes 
                  all requests are passed to a real %s object on a remote host.

                  """ % ((base.__name__,)*3) + base_doc
        newcls = type(cls.__name__, cls.__bases__, dct)
        newcls._include_underscore = include_underscore
        newcls._exclude = exclude
        if supers:
            proxied_classes = base.__mro__[:-1]
        else:
            proxied_classes = (base,)
        for c in proxied_classes:
            for name in c.__dict__:
                def mk_proxy_method(method_name, doc):
                    def method(self, *args, **kwargs):
                        if self._obcache_current:
                            return getattr(self._obcache, method_name)(
                                    *args, **kwargs)
                        else:
                            return newcls._try_cached_apply(self, method_name,
                                                            *args, **kwargs)
                    method.__doc__ = doc
                    method.__name__ = method_name
                    #return types.MethodType(method, None, newcls)
                    return method
                def mk_property(attrib_name):
                    # TODO: implement fset and fdel (requires writeback cache)
                    def getter(self):
                        if self._obcache_current:
                            return getattr(self._obcache, attrib_name)
                        else:
                            return self._cached_apply('__getattribute__',
                                                      attrib_name)
                    prop = property(fget=getter)
                    return prop
                #respect MRO: proxy an attribute only if it is not overridden
                if (name not in newcls.__dict__ and
                        all(name not in b.__dict__ 
                            for c in bases_other for b in c.mro()[:-1]) and
                        name not in newcls._exclude and
                        (name[0] != '_' or 
                         newcls._include_underscore is True or
                         name in newcls._include_underscore)):
                    f = c.__dict__[name]
                    if callable(f) and not isinstance(f, type):
                        setattr(newcls, name, mk_proxy_method(name, f.__doc__))
                    else:
                        setattr(newcls, name, mk_property(name))
        #newcls.__module__ = '__main__' # cause dill to pickle it whole
        #import __main__
        #__main__.__dict__[newcls.__name__] = newcls # for dill..
        ObjectHub.register_proxy_type(base, newcls)
        return newcls
    return rebuild_class


def _async_scatter(obj):
    """Distribute an obj or list to remote engines. 
    Return an async result or (possibly nested) lists of async results, 
    each of which is a Ref
    """
    #TODO Instead of special cases for basestring and Remote, should have a 
    #     list of types that should not be proxied, inc. basestring and Remote.
    if isinstance(obj, Remote):
        return obj
    if (isinstance(obj, collections.Sequence) and 
            not isinstance(obj, basestring)):
        ars = []
        for i in xrange(len(obj)):
            ars.append(_async_scatter(obj[i]))
        return ars
    else:
        if distob.engine is None:
            _setup_engines()
        client = distob.engine._client
        dv = distob.engine._dv
        def remote_put(obj):
            return Ref(obj)
        dv.targets = _async_scatter.next_engine
        ar_ref = dv.apply_async(remote_put, obj)
        _async_scatter.next_engine = (
                _async_scatter.next_engine + 1) % len(client)
        dv.targets = client.ids
        return ar_ref

_async_scatter.next_engine = 0


def _ars_to_proxies(ars, obj):
    """wait for async results and return proxy objects to replace obj
    Args: 
      ars: AsyncResult (or sequence of AsyncResults), each result type ``Ref``.
      obj: original object being scattered (or sequence of objects)
    Returns:
      Remote* proxy object (or sequence of them) with same interface as obj
    """
    if isinstance(ars, Remote):
        return ars
    elif isinstance(ars, collections.Sequence):
        for i in xrange(len(ars)):
            obj[i] = _ars_to_proxies(ars[i], obj[i])
        return obj
    elif isinstance(ars, parallel.AsyncResult):
        ars.wait()
        ref = ars.r
        ObClass = ref.type
        if ObClass in distob.engine.proxy_types:
            RemoteClass = distob.engine.proxy_types[ObClass]
        else:
            RemoteClass = type(
                    'Remote' + ObClass.__name__, (Remote, ObClass), dict())
            RemoteClass = proxy_methods(ObClass)(RemoteClass)
        proxy_obj = RemoteClass(ref, distob.engine._client)
        return proxy_obj
    else:
        raise DistobTypeError('Unpacking ars: unexpected type %s' % type(ars))


def scatter(obj):
    """Distribute obj or list to remote engines, returning proxy objects"""
    if isinstance(obj, Remote):
        return obj
    ars = _async_scatter(obj)
    proxy_obj = _ars_to_proxies(ars, obj)
    return proxy_obj


def gather(obj):
    """Retrieve objects that have been distributed, making them local again"""
    if not isinstance(obj, Remote) and (
            isinstance(obj, basestring) or (
                not isinstance(obj, collections.Sequence))):
        return obj
    elif not isinstance(obj, Remote):
        for i in xrange(len(obj)):
            obj[i] = gather(obj[i])
        return None
    else:
        return obj._ob


def call_all(sequence, method_name, *args, **kwargs):
    """Call a method on each element of the sequence, in parallel.
    Returns:
      list of results
    """
    # dispatch method calls asynchronously
    results = []
    for obj in sequence:
        if isinstance(obj, Remote):
            results.append(obj._apply_async(method_name, *args, **kwargs))
        else: 
            results.append(getattr(obj, method_name)(*args, **kwargs))
    # now wait for all results before returning
    for i in xrange(len(results)):
        obj = results[i]
        if isinstance(obj, parallel.AsyncResult):
            obj.wait()
            results[i] = obj.r
        if isinstance(results[i], Ref):
            ref = results[i]
            RemoteClass = distob.engine.proxy_types[ref.type]
            results[i] = RemoteClass(ref, distob.engine._client)
    return results
