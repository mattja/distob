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
  vectorize(f)  Upgrade normal function f to act in parallel on distributed
    lists or arrays
  call_all(sequence, methodname, *args, **kwargs) Call method on each element. 

Classes:
  Remote   base class, used when creating Remote* proxy classes
  @proxy_methods(base)   class decorator for creating Remote* proxy classes
  ObjectHub  dict interface giving refs to all distributed objects cluster-wide
  ObjectEngine  dict holding distributed objects of a single ipyparallel engine
  Ref  reference to a (possibly remote) object
"""

from __future__ import absolute_import
import ipyparallel
import distob
from . import _pylru
import types
import copy
import warnings
import importlib
import collections
import numbers

if distob._have_numpy:
    import numpy as np

# types for compatibility across python 2 and 3
try:
    string_types = basestring
except NameError:
    string_types = str


class Error(Exception):
    pass


class DistobTypeError(Error):
    pass


class DistobValueError(Error):
    pass


class DistobClusterError(Error):
    pass


# Cluster-wide unique identifier for an object:
Id = collections.namedtuple('Id', 'instance engine')


class Ref(object):
    """Reference to a (possibly remote) object.

    a Ref can exist on the same ipyparallel engine that holds the object, or on
    any python instance that has an ObjectHub (such as the ipyparallel client)

    Attributes:
      id (Id): An cluster-wide unique identifier for the object.
        id.instance is a number unique within a particular engine.
        id.engine is the ipyparallel engine number where the object is held,
          or a negative number if it is held on an ipyparallel client.
      type: type of the remote object
      metadata (tuple, optional): brief extra information about the object 
        may be defined by specific Remote* classes, to help set up access.
    """
    def __init__(self, obj):
        if isinstance(obj, Remote):
            obj = obj._ref
        if isinstance(obj, Ref):
            self.id = obj.id
            self.type = obj.type
            self.metadata = obj.metadata
            distob.engine.incref(self.id)
        else:
            self.type = type(obj)
            if self.type in distob.engine.proxy_types:
                ptype = distob.engine.proxy_types[self.type]
                self.metadata = ptype.__pmetadata__(obj)
            else:
                self.metadata = None
            self.id = Id(id(obj), distob.engine.eid)
            distob.engine[self.id] = obj

    def __getstate__(self):
        # pickled or copied Ref is included in refcount
        distob.engine.incref(self.id)
        return (self.id, self.type.__name__, self.type.__module__, 
                self.metadata)

    def __setstate__(self, state):
        self.id, typename, modname, self.metadata = state
        module = importlib.import_module(modname)
        self.type = getattr(module, typename)

    def __del__(self):
        distob.engine.decref(self.id)

    def __repr__(self):
        id = self.id
        typename = self.type.__name__
        return u'<Ref to %s:%s@e%d>' % (typename, id.instance, id.engine)


class ObjectEngine(dict):
    """A dict holding the distributed objects of a single ipyparallel engine.

    It maintains cluster-wide reference counts of the objects it holds.
    There is one global instance (distob.engine) on each ipyparallel engine.

    For each dict entry,
    Key is an object Id (unique cluster-wide)
    Value is the actual object that is to be accessed remotely.

    Attributes:
      eid (int): ipyparallel engine id number of this engine. A negative number
        means this is an ipyparallel client, rather than an engine.
      nengines (int): Total number of ipyparallel engines (at startup time)
      refcounts (dict): Key is an object Id (for an object held on this engine)
        Value is the cluster-wide count of how many Ref instances exist 
        pointing to that object.
      proxy_types (dict): When an object's methods are called remotely, some
        return types should be returned by proxy instead of returning a 
        full object. This dict holds mappings {real_type: proxy_type} for 
        these types.
    """
    def __init__(self, engine_id, nengines):
        self.eid = engine_id 
        self.nengines = nengines
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
            # print('Cleaned up and deleted object %s' % key)
        else:
            self.decref(key)

    def __repr__(self):
        return '<%s instance on engine %d>:\n%s' % (
            self.__class__.__name__, self.eid, repr(self.keys()))

    def _repr_pretty_(self, p, cycle):
        return '<%s instance on engine %d>:\n%s' % (
            self.__class__.__name__, self.eid, p.pretty(self.keys()))


# TODO put this back inside ObjectHub after dill issue #58 is resolved.
def _remote_reg_proxy_type(real_t, proxy_t):
    distob.engine.proxy_types[real_t] = proxy_t


class ObjectHub(ObjectEngine):
    """A dict providing references to all distributed objects across the cluster

    Key is an object Id (unique cluster-wide)
    Value is the actual object (if local) or a Ref to the object (if remote)

    Operations supported are get and delete.

    Attributes:
      eid (int): ipyparallel engine id number of this engine. A negative number
        means this is an ipyparallel client, rather than an engine.
      refcounts (dict): Key is an Id (for an object held on this engine). 
        Value is the cluster-wide count of how many Ref instances exist
        pointing to that object.
      proxy_types (dict): When an object's methods are called remotely, some
        return types should be returned by proxy instead of returning a 
        full object. This dict holds mappings {real_type: proxy_type} for 
        these types.
    """
    _initial_proxy_types = dict()

    def __init__(self, engine_id, client):
        """Make an ObjectHub.
        Args:
          engine_id: ipyparallel engine id number where this Hub is located,
            or a negative number if it is on an ipyparallel client.
          client: ipyparallel.Client
        """
        self._client = client
        self._dv = client.direct_view(targets='all')
        self._dv.use_dill()
        nengines = len(client)
        super(ObjectHub, self).__init__(engine_id, nengines)
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
        if key.engine is self.eid:
            super(ObjectHub, self).incref(key)
        else:
            def _remote_incref(key):
               distob.engine.incref(key) 
            self._dv.targets = key.engine
            self._dv.apply_sync(_remote_incref, key)
            self._dv.targets = 'all'

    def decref(self, key):
        if key.engine is self.eid:
            super(ObjectHub, self).decref(key)
        else:
            def _remote_decref(key):
               distob.engine.decref(key) 
            self._dv.targets = key.engine
            self._dv.apply_sync(_remote_decref, key)
            self._dv.targets = 'all'

    def __delitem__(self, key):
        if key.engine is self.eid:
            super(ObjectHub, self).__delitem__(key)
        else:
            def _remote_delitem(key):
               del distob.engine[key]
            self._dv.targets = key.engine
            self._dv.apply_sync(_remote_delitem, key)
            self._dv.targets = 'all'

    def __getitem__(self, key):
        if key.engine is self.eid:
            return super(ObjectHub, self).__getitem__(key)
        else:
            def fetch_object(id):
                return Ref(distob.engine[id])
            self._dv.targets = key.engine
            res = self._dv.apply_sync(fetch_object, key)
            self._dv.targets = 'all'
            return res

    def __setitem__(self, key, value):
        if key.engine is self.eid:
            super(ObjectHub, self).__setitem__(key, value)
        else:
            raise Error('Setting remote objects via ObjectHub not implemented')

    def keys(self):
        local_keys = super(ObjectHub, self).keys()
        def fetch_keys():
            from distob import engine
            return engine.keys()
        ar = self._dv.apply_async(fetch_keys)
        self._dv.wait(ar)
        remote_keys = [key for sublist in ar.r for key in sublist]
        return remote_keys + local_keys


def _remote_setup_engine(engine_id, nengines):
    """(Executed on remote engine) creates an ObjectEngine instance """
    if distob.engine is None:
        distob.engine = distob.ObjectEngine(engine_id, nengines)
    # TODO these imports should be unnecessary with improved deserialization
    import numpy as np
    from scipy import stats
    # TODO Using @ipyparallel.interactive still did not import to __main__
    #      so will do it this way for now.
    import __main__
    __main__.__dict__['np'] = np
    __main__.__dict__['stats'] = stats


def setup_engines(client=None):
    """Prepare all iPython engines for distributed object processing.

    Args:
      client (ipyparallel.Client, optional): If None, will create a client
        using the default ipyparallel profile.
    """
    if not client:
        try:
            client = ipyparallel.Client()
        except:
            raise DistobClusterError(
                u"""Could not connect to an ipyparallel cluster. Make
                 sure a cluster is started (e.g. to use the CPUs of a
                 single computer, can type 'ipcluster start')""")
    eids = client.ids
    if not eids:
        raise DistobClusterError(
                u'No ipyparallel compute engines are available')
    nengines = len(eids)
    dv = client[eids]
    dv.use_dill()
    with dv.sync_imports(quiet=True):
        import distob
    # create global ObjectEngine distob.engine on each engine
    ars = []
    for i in eids:
        dv.targets = i
        ars.append(dv.apply_async(_remote_setup_engine, i, nengines))
    dv.wait(ars)
    for ar in ars:
        if not ar.successful():
            raise ar.r
    # create global ObjectHub distob.engine on the client host
    if distob.engine is None:
        distob.engine = ObjectHub(-1, client)


def _process_args(args, kwargs, prefer_local=True, recurse=True):
    """Select local or remote execution and prepare arguments accordingly.
    Assumes any remote args have already been moved to a common engine.

    Local execution will be chosen if:
    - all args are ordinary objects or Remote instances on the local engine; or
    - the local cache of all remote args is current, and prefer_local is True.
    Otherwise, remote execution will be chosen. 

    For remote execution, replaces any remote arg with its Id.
    For local execution, replaces any remote arg with its locally cached object
    Any arguments or kwargs that are Sequences will be recursed one level deep.

    Args:
      args (list)
      kwargs (dict)
      prefer_local (bool, optional): Whether cached local results are prefered
        if available, instead of returning Remote objects. Default is True.
    """
    this_engine = distob.engine.eid
    local_args = []
    remote_args = []
    execloc = this_engine  # the chosen engine id for execution of the call
    for a in args:
        id = None
        if isinstance(a, Remote):
            id = a._ref.id
        elif isinstance(a, Ref):
            id = a.id
        elif isinstance(a, Id):
            id = a
        if id is not None:
            if id.engine is this_engine:
                local_args.append(distob.engine[id])
                remote_args.append(distob.engine[id])
            else:
                if (prefer_local and isinstance(a, Remote) and 
                        a._obcache_current):
                    local_args.append(a._obcache)
                    remote_args.append(id)
                else:
                    # will choose remote execution
                    if execloc is not this_engine and id.engine is not execloc:
                        raise DistobValueError(
                            'two remote arguments are from different engines')
                    else:
                        execloc = id.engine
                        local_args.append(None)
                        remote_args.append(id)
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types) and recurse):
            eid, ls, _ = _process_args(a, {}, prefer_local, recurse=False)
            if eid is not this_engine:
                if execloc is not this_engine and eid is not execloc:
                    raise DistobValueError(
                            'two remote arguments are from different engines')
                execloc = eid
            local_args.append(ls)
            remote_args.append(ls)
        else:
            # argument is an ordinary object
            local_args.append(a)
            remote_args.append(a)
    local_kwargs = dict()
    remote_kwargs = dict()
    for k, a in kwargs.items():
        id = None
        if isinstance(a, Remote):
            id = a._ref.id
        elif isinstance(a, Ref):
            id = a.id
        elif isinstance(a, Id):
            id = a
        if id is not None:
            if id.engine is this_engine:
                local_kwargs[k] = distob.engine[id]
                remote_kwargs[k] = distob.engine[id]
            else:
                if (prefer_local and isinstance(a, Remote) and
                        a._obcache_current):
                    local_kwargs[k] = a._obcache
                    remote_kwargs[k] = id
                else:
                    # will choose remote execution
                    if execloc is not this_engine and id.engine is not execloc:
                        raise DistobValueError(
                            'two remote arguments are from different engines')
                    else:
                        execloc = id.engine
                        local_kwargs[k] = None
                        remote_kwargs[k] = id
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types) and recurse):
            eid, ls, _ = _process_args(a, {}, prefer_local, recurse=False)
            if eid is not this_engine:
                if execloc is not this_engine and eid is not execloc:
                    raise DistobValueError(
                            'two remote arguments are from different engines')
                execloc = eid
            local_kwargs[k] = ls
            remote_kwargs[k] = ls
        else:
            # argument is an ordinary object 
            local_kwargs[k] = a
            remote_kwargs[k] = a
    if execloc is this_engine:
        return execloc, tuple(local_args), local_kwargs
    else:
        return execloc, tuple(remote_args), remote_kwargs


def _remote_call(f, *args, **kwargs):
    """(Executed on remote engine) convert Ids to real objects, call f """
    nargs = []
    for a in args:
        if isinstance(a, Id):
            nargs.append(distob.engine[a])
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types)):
            nargs.append(
                    [distob.engine[b] if isinstance(b, Id) else b for b in a])
        else: nargs.append(a)
    for k, a in kwargs.items():
        if isinstance(a, Id):
            kwargs[k] = distob.engine[a]
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types)):
            kwargs[k] = [
                    distob.engine[b] if isinstance(b, Id) else b for b in a]
    result = f(*nargs, **kwargs)
    if (isinstance(result, collections.Sequence) and
            not isinstance(result, string_types)):
        # We will return any sub-sequences by value, not recurse deeper
        results = []
        for subresult in result:
            if type(subresult) in distob.engine.proxy_types: 
                results.append(Ref(subresult))
            else:
                results.append(subresult)
        return results
    elif type(result) in distob.engine.proxy_types:
        return Ref(result)
    else:
        return result


def _uncached_call(execloc, f, *args, **kwargs):
    dv = distob.engine._dv
    dv.targets = execloc
    ar = dv.apply_async(_remote_call, f, *args, **kwargs)
    dv.targets = 'all'
    return ar


_call_cache = _pylru.lrucache(1000)


def call(f, *args, **kwargs):
    """Execute f on the arguments, either locally or remotely as appropriate.
    If there are multiple remote arguments, they must be on the same engine.
    
    kwargs:
      prefer_local (bool, optional): Whether to return cached local results if
        available, in preference to returning Remote objects. Default is True.
      block (bool, optional): Whether remote calls should be synchronous.
        If False, returned results may be AsyncResults and should be converted
        by the caller using convert_result() before use. Default is True.
    """
    this_engine = distob.engine.eid
    prefer_local = kwargs.pop('prefer_local', True)
    block = kwargs.pop('block', True)
    execloc, args, kwargs = _process_args(args, kwargs, prefer_local)
    if execloc is this_engine:
        r = f(*args, **kwargs)
    else:
        if False and prefer_local:
            # result cache disabled until issue mattja/distob#1 is fixed
            try:
                kwtuple = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
                key = (f, args, kwtuple)
                r = _call_cache[key]
            except TypeError as te:
                if te.args[0][:10] == 'unhashable':
                    #print("unhashable. won't be able to cache")
                    r = _uncached_call(execloc, f, *args, **kwargs)
                else:
                    raise
            except KeyError:
                r = _uncached_call(execloc, f, *args, **kwargs)
                if block:
                    _call_cache[key] = r.r
        else:
            r = _uncached_call(execloc, f, *args, **kwargs)
    if block:
        return convert_result(r)
    else:
        return r


def convert_result(r):
    """Waits for and converts any AsyncResults. Converts any Ref into a Remote.
    Args:
      r: can be an ordinary object, ipyparallel.AsyncResult, a Ref, or a
        Sequence of objects, AsyncResults and Refs.
    Returns: 
      either an ordinary object or a Remote instance"""
    if (isinstance(r, collections.Sequence) and
            not isinstance(r, string_types)):
        rs = []
        for subresult in r:
            rs.append(convert_result(subresult))
        return rs
    if isinstance(r, ipyparallel.AsyncResult):
        r = r.r
    if isinstance(r, Ref):
        RemoteClass = distob.engine.proxy_types[r.type]
        r = RemoteClass(r)
    return r


def _remote_methodcall(id, method_name, *args, **kwargs):
    """(Executed on remote engine) convert Ids to real objects, call method """
    obj = distob.engine[id]
    nargs = []
    for a in args:
        if isinstance(a, Id):
            nargs.append(distob.engine[a])
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types)):
            nargs.append(
                    [distob.engine[b] if isinstance(b, Id) else b for b in a])
        else: nargs.append(a)
    for k, a in kwargs.items():
        if isinstance(a, Id):
            kwargs[k] = distob.engine[a]
        elif (isinstance(a, collections.Sequence) and
                not isinstance(a, string_types)):
            kwargs[k] = [
                    distob.engine[b] if isinstance(b, Id) else b for b in a]
    result = getattr(obj, method_name)(*nargs, **kwargs)
    if (isinstance(result, collections.Sequence) and
            not isinstance(result, string_types)):
        # We will return any sub-sequences by value, not recurse deeper
        results = []
        for subresult in result:
            if type(subresult) in distob.engine.proxy_types: 
                results.append(Ref(subresult))
            else:
                results.append(subresult)
        return results
    elif type(result) in distob.engine.proxy_types:
        return Ref(result)
    else:
        return result


def _uncached_methodcall(execloc, id, method_name, *args, **kwargs):
    dv = distob.engine._dv
    dv.targets = execloc
    ar = dv.apply_async(_remote_methodcall, id, method_name, *args, **kwargs)
    dv.targets = 'all'
    return ar


def methodcall(obj, method_name, *args, **kwargs):
    """Call a method of `obj`, either locally or remotely as appropriate.
    obj may be an ordinary object, or a Remote object (or Ref or object Id)
    If there are multiple remote arguments, they must be on the same engine.
    
    kwargs:
      prefer_local (bool, optional): Whether to return cached local results if
        available, in preference to returning Remote objects. Default is True.
      block (bool, optional): Whether remote calls should be synchronous.
        If False, returned results may be AsyncResults and should be converted
        by the caller using convert_result() before use. Default is True.
    """
    this_engine = distob.engine.eid
    args = [obj] + list(args)
    prefer_local = kwargs.pop('prefer_local', None)
    if prefer_local is None:
        if isinstance(obj, Remote):
            prefer_local = obj.prefer_local
        else:
            prefer_local = True
    block = kwargs.pop('block', True)
    execloc, args, kwargs = _process_args(args, kwargs, prefer_local)
    if execloc is this_engine:
        r = getattr(args[0], method_name)(*args[1:], **kwargs)
    else:
        if False and prefer_local:
            # result cache disabled until issue mattja/distob#1 is fixed
            try:
                kwtuple = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
                key = (args[0], method_name, args, kwtuple)
                r = _call_cache[key]
            except TypeError as te:
                if te.args[0][:10] == 'unhashable':
                    #print("unhashable. won't be able to cache")
                    r = _uncached_methodcall(execloc, args[0], method_name,
                                             *args[1:], **kwargs)
                else:
                    raise
            except KeyError:
                r = _uncached_methodcall(execloc, args[0], method_name,
                                         *args[1:], **kwargs)
                if block:
                    _call_cache[key] = r.r
        else:
            r = _uncached_methodcall(execloc, args[0], method_name,
                                     *args[1:], **kwargs)
    if block:
        return convert_result(r)
    else:
        return r


def _make_proxy_method(method_name, doc=None):
    def method(self, *args, **kwargs):
        return methodcall(self, method_name, *args, **kwargs)
    method.__doc__ = doc
    method.__name__ = method_name
    return method


def _make_proxy_property(attrib_name, doc=None):
    def getter(self):
        return methodcall(self, '__getattribute__', attrib_name)
    # TODO: implement fset and fdel (requires writeback cache and locking)
    prop = property(fget=getter, doc=doc)
    return prop


def _scan_instance(obj, include_underscore, exclude):
    """(Executed on remote or local engine) Examines an object and returns info
    about any instance-specific methods or attributes.
    (For example, any attributes that were set by __init__() )

    By default, methods or attributes starting with an underscore are ignored.

    Args:
      obj (object): the object to scan. must be on this local engine.
      include_underscore (bool or sequence of str): Should methods or
        attributes that start with an underscore be proxied anyway? If a
        sequence of names is provided then methods or attributes starting with
        an underscore will only be proxied if their names are in the sequence.
      exclude (sequence of str): names of any methods or attributes that should
        not be reported.
    """
    from sys import getsizeof
    always_exclude = ('__new__', '__init__', '__getattribute__', '__class__',
                      '__reduce__', '__reduce_ex__')
    method_info = []
    attributes_info = []
    if hasattr(obj, '__dict__'):
        for name in obj.__dict__:
            if (name not in exclude and 
                name not in always_exclude and
                (name[0] != '_' or 
                 include_underscore is True or
                 name in include_underscore)):
                f = obj.__dict__[name]
                if hasattr(f, '__doc__'):
                    doc = f.__doc__
                else:
                    doc = None
                if callable(f) and not isinstance(f, type):
                    method_info.append((name, doc))
                else:
                    attributes_info.append((name, doc))
    return (method_info, attributes_info, getsizeof(obj))


class Remote(object):
    """Base class for Remote* proxy classes.
    An instance of a Remote subclass is a proxy designed to control
    an instance of an existing class which may be local or on a remote host.

    To make a Remote proxy class, inherit from `Remote` as first base class.
    If the instances of the new Remote* proxy class should be treated 
    exactly like instances of the controlled class, then also inherit from 
    the controlled class as the second base:

      @proxy_methods(Tree)
      class RemoteTree(Remote, Tree)

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

    def __init__(self, obj):
        """Set up the Remote* proxy object to access an already-existing object,
        which may be local or remote.

        Args:
          obj (Ref or object): either a Ref reference to the (possibly remote) 
            object to be controlled, or else an actual (local) object to be 
            controlled.
        """
        if distob.engine is None:
            setup_engines()
        if isinstance(obj, Ref):
            self._ref = obj
            self.is_local = (self._ref.id.engine is distob.engine.eid)
        else:
            self._ref = Ref(obj)
            self.is_local = True
        if self.is_local:
            self._dv = None
            self._obcache = distob.engine[self._ref.id]
            self._obcache_current = True
        else:
            self._dv = distob.engine._client[self._ref.id.engine]
            self._dv.use_dill()
            self._obcache = None
            self._obcache_current = False
        self._id = self._ref.id
        # preference setting: whether to give cached local results if available
        self.prefer_local = True 
        #Add proxy controllers for any instance-specific methods/attributes:
        instance_methods, instance_attribs, size = call(
                _scan_instance, self, self.__class__._include_underscore,
                self.__class__._exclude, prefer_local=False)
        for name, doc in instance_methods:
            setattr(self, name, _make_proxy_method(name, doc))
        for name, doc in instance_attribs:
            setattr(self.__class__, name, _make_proxy_property(name, doc))
        self.__engine_affinity__ = (self._ref.id.engine, size)

    def _fetch(self):
        """forces update of a local cached copy of the real object
        (regardless of the preference setting self.cache)"""
        if not self.is_local and not self._obcache_current:
            #print('fetching data from %s' % self._ref.id)
            def _remote_fetch(id):
                return distob.engine[id]
            self._obcache = self._dv.apply_sync(_remote_fetch, self._id)
            self._obcache_current = True
            self.__engine_affinity__ = (distob.engine.eid, 
                                        self.__engine_affinity__[1])

    def __ob(self):
        """return a local copy of the real object"""
        self._fetch()
        return self._obcache

    _ob = property(fget=__ob, doc='return a local copy of the object')

    def __distob_gather__(self):
        return self._ob

    def __copy__(self):
        newref = copy.copy(self._ref)
        obj = self.__class__.__new__(self.__class__, newref)
        obj.__init__(newref)
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
      include_underscore (bool or sequence of str): Should methods or
        attributes that start with an underscore be proxied anyway? If a
        sequence of names is provided then methods or attributes starting with
        an underscore will only be proxied if their names are in the sequence.
      exclude (sequence of str): Names of any methods or attributes that 
        should not be proxied.
      supers (bool): Proxy methods and attributes defined in superclasses 
        of ``base``, in addition to those defined directly in class ``base``
    """
    always_exclude = ('__new__', '__init__', '__getattribute__', '__class__',
                      '__reduce__', '__reduce_ex__')
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
                #respect MRO: proxy an attribute only if it is not overridden
                if (name not in newcls.__dict__ and
                        all(name not in b.__dict__ 
                            for c in bases_other for b in c.mro()[:-1]) and
                        name not in newcls._exclude and
                        name not in always_exclude and
                        (name[0] != '_' or 
                         newcls._include_underscore is True or
                         name in newcls._include_underscore)):
                    f = c.__dict__[name]
                    if hasattr(f, '__doc__'):
                        doc = f.__doc__
                    else:
                        doc = None
                    if callable(f) and not isinstance(f, type):
                        setattr(newcls, name, _make_proxy_method(name, doc))
                    else:
                        setattr(newcls, name, _make_proxy_property(name, doc))
        newcls.__module__ = '__main__' # cause dill to pickle it whole
        import __main__
        __main__.__dict__[newcls.__name__] = newcls # for dill..
        ObjectHub.register_proxy_type(base, newcls)
        return newcls
    return rebuild_class


def _async_scatter(obj, destination=None):
    """Distribute an obj or list to remote engines. 
    Return an async result or (possibly nested) lists of async results, 
    each of which is a Ref
    """
    #TODO Instead of special cases for strings and Remote, should have a
    #     list of types that should not be proxied, inc. strings and Remote.
    if (isinstance(obj, Remote) or 
            isinstance(obj, numbers.Number) or 
            obj is None):
        return obj
    if (isinstance(obj, collections.Sequence) and 
            not isinstance(obj, string_types)):
        ars = []
        if destination is not None:
            assert(len(destination) == len(obj))
            for i in range(len(obj)):
                ars.append(_async_scatter(obj[i], destination[i]))
        else:
            for i in range(len(obj)):
                ars.append(_async_scatter(obj[i], destination=None))
        return ars
    else:
        if distob.engine is None:
            setup_engines()
        client = distob.engine._client
        dv = distob.engine._dv
        def remote_put(obj):
            return Ref(obj)
        if destination is not None:
            assert(isinstance(destination, numbers.Integral))
            dv.targets = destination
        else:
            dv.targets = _async_scatter.next_engine
            _async_scatter.next_engine = (
                    _async_scatter.next_engine + 1) % len(client)
        ar_ref = dv.apply_async(remote_put, obj)
        dv.targets = client.ids
        return ar_ref

_async_scatter.next_engine = 0


def _ars_to_proxies(ars):
    """wait for async results and return proxy objects
    Args: 
      ars: AsyncResult (or sequence of AsyncResults), each result type ``Ref``.
    Returns:
      Remote* proxy object (or list of them)
    """
    if (isinstance(ars, Remote) or
            isinstance(ars, numbers.Number) or
            ars is None):
        return ars
    elif isinstance(ars, collections.Sequence):
        res = []
        for i in range(len(ars)):
            res.append(_ars_to_proxies(ars[i]))
        return res
    elif isinstance(ars, ipyparallel.AsyncResult):
        ref = ars.r
        ObClass = ref.type
        if ObClass in distob.engine.proxy_types:
            RemoteClass = distob.engine.proxy_types[ObClass]
        else:
            RemoteClass = type(
                    'Remote' + ObClass.__name__, (Remote, ObClass), dict())
            RemoteClass = proxy_methods(ObClass)(RemoteClass)
        proxy_obj = RemoteClass(ref)
        return proxy_obj
    else:
        raise DistobTypeError('Unpacking ars: unexpected type %s' % type(ars))


def _scatter_ndarray(ar, axis=-1, destination=None, blocksize=None):
    """Turn a numpy ndarray into a DistArray or RemoteArray
    Args:
     ar (array_like)
     axis (int, optional): specifies along which axis to split the array to 
       distribute it. The default is to split along the last axis. `None` means
       do not distribute.
     destination (int or list of int, optional): Optionally force the array to
       go to a specific engine. If an array is to be scattered along an axis, 
       this should be a list of engine ids with the same length as that axis.
     blocksize (int): Optionally control the size of intervals into which the
       distributed axis is split (the default splits the distributed axis
       evenly over all computing engines).
    """
    from .arrays import DistArray, RemoteArray
    shape = ar.shape
    ndim = len(shape)
    if axis is None:
        return _directed_scatter([ar], destination=[destination],
                                 blocksize=blocksize)[0]
    if axis < -ndim or axis > ndim - 1:
        raise DistobValueError('axis out of range')
    if axis < 0:
        axis = ndim + axis
    n = shape[axis]
    if n == 1:
        return _directed_scatter([ar], destination=[destination])[0]
    if isinstance(destination, collections.Sequence):
        ne = len(destination) # number of engines to scatter array to
    else:
        if distob.engine is None:
            setup_engines()
        ne = distob.engine.nengines # by default scatter across all engines
    if blocksize is None:
        blocksize = ((n - 1) // ne) + 1
    if blocksize > n:
        blocksize = n
    if isinstance(ar, DistArray):
        if axis == ar._distaxis:
            return ar
        else:
            raise DistobError('Currently can only scatter one axis of array')
    # Currently, if requested to scatter an array that is already Remote and
    # large, first get whole array locally, then scatter. Not really optimal.
    if isinstance(ar, RemoteArray) and n > blocksize:
        ar = ar._ob
    s = slice(None)
    subarrays = []
    low = 0
    for i in range(0, n // blocksize):
        high = low + blocksize
        index = (s,)*axis + (slice(low, high),) + (s,)*(ndim - axis - 1)
        subarrays.append(ar[index])
        low += blocksize
    if n % blocksize != 0:
        high = low + (n % blocksize)
        index = (s,)*axis + (slice(low, high),) + (s,)*(ndim - axis - 1)
        subarrays.append(ar[index])
    subarrays = _directed_scatter(subarrays, destination=destination)
    return DistArray(subarrays, axis)


def _directed_scatter(obj, axis=-1, destination=None, blocksize=None):
    """Same as scatter() but allows forcing the object to a specific engine id.
    Currently, this is intended for distob internal use only, so that ufunc
    operands can be sent to the same engine.
    Args:
      destination (int or list of int, optional): the engine id (or ids). If 
        scattering a sequence, this should be a list of engine ids with the
        same length as the sequence. If an array is to be scattered along an
        axis, this should be a list of engine ids with the same length as 
        that axis.
      blocksize (int): Optionally aim to split the distributed axis into
        intervals of approximately this length (default is number of engines 
        divided by length of distributed axis).
    """
    if hasattr(obj, '__distob_scatter__'):
        return obj.__distob_scatter__(axis, destination, blocksize)
    if distob._have_numpy and (isinstance(obj, np.ndarray) or
                        hasattr(type(obj), '__array_interface__')):
        return _scatter_ndarray(obj, axis, destination, blocksize)
    elif isinstance(obj, Remote):
        return obj
    ars = _async_scatter(obj, destination)
    proxy_obj = _ars_to_proxies(ars)
    return proxy_obj


def scatter(obj, axis=-1, blocksize=None):
    """Distribute obj or list to remote engines, returning proxy objects
    Args:
      obj: any python object, or list of objects
      axis (int, optional): Can be used if scattering a numpy array,
        specifying along which axis to split the array to distribute it. The 
        default is to split along the last axis. `None` means do not distribute
      blocksize (int, optional): Can be used if scattering a numpy array. 
        Optionally control the size of intervals into which the distributed
        axis is split (the default splits the distributed axis evenly over all
        computing engines).
    """
    if hasattr(obj, '__distob_scatter__'):
        return obj.__distob_scatter__(axis, None, blocksize)
    if distob._have_numpy and (isinstance(obj, np.ndarray) or
                        hasattr(type(obj), '__array_interface__')):
        return _scatter_ndarray(obj, axis, blocksize)
    elif isinstance(obj, Remote):
        return obj
    ars = _async_scatter(obj)
    proxy_obj = _ars_to_proxies(ars)
    return proxy_obj


def gather(obj):
    """Retrieve objects that have been distributed, making them local again"""
    if hasattr(obj, '__distob_gather__'):
        return obj.__distob_gather__()
    elif (isinstance(obj, collections.Sequence) and 
            not isinstance(obj, string_types)):
        return [gather(subobj) for subobj in obj]
    else:
        return obj


def vectorize(f):
    """Upgrade normal function f to act in parallel on distibuted lists/arrays

    Args:
      f (callable): an ordinary function which expects as its first argument a
        single object, or a numpy array of N dimensions.

    Returns:
      vf (callable): new function that takes as its first argument a list of
        objects, or a array of N+1 dimensions. ``vf()`` will do the
        computation ``f()`` on each part of the input in parallel and will
        return a list of results, or a distributed array of results.
    """
    def vf(obj, *args, **kwargs):
        # user classes can customize how to vectorize a function:
        if hasattr(obj, '__distob_vectorize__'):
            return obj.__distob_vectorize__(f)(obj, *args, **kwargs)
        if isinstance(obj, Remote):
            return call(f, obj, *args, **kwargs)
        elif distob._have_numpy and (isinstance(obj, np.ndarray) or
                 hasattr(type(obj), '__array_interface__')):
            distarray = scatter(obj, axis=-1)
            return vf(distarray, *args, **kwargs)
        elif isinstance(obj, collections.Sequence):
            inputs = scatter(obj)
            dv = distob.engine._client[:]
            kwargs = kwargs.copy()
            kwargs['block'] = False
            results = []
            for obj in inputs:
                results.append(call(f, obj, *args, **kwargs))
            for i in range(len(results)):
                results[i] = convert_result(results[i])
            return results
    if hasattr(f, '__name__'):
        vf.__name__ = 'v' + f.__name__
        f_str = f.__name__ + '()'
    else:
        f_str = 'callable'
    doc = u"""Apply %s in parallel to a list or array\n
           Args:
             obj (Sequence of objects or an array)
             other args are the same as for %s
           """ % (f_str, f_str)
    if hasattr(f, '__doc__') and f.__doc__ is not None:
        doc = doc.rstrip() + (' detailed below:\n----------\n' + f.__doc__)
    vf.__doc__ = doc
    return vf


def apply(f, obj, *args, **kwargs):
    """Apply a function in parallel to each element of the input"""
    return vectorize(f)(obj, *args, **kwargs)


def call_all(sequence, method_name, *args, **kwargs):
    """Call a method on each element of a sequence, in parallel.
    Returns:
      list of results
    """
    kwargs = kwargs.copy()
    kwargs['block'] = False
    results = []
    for obj in sequence:
        results.append(methodcall(obj, method_name, *args, **kwargs))
    for i in range(len(results)):
        results[i] = convert_result(results[i])
    return results
