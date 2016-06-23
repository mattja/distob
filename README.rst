distob
======
| Distributed computing made easier, using remote objects
|  N.B. this is a development pre-release: still a lot left to be done

Overview
--------
Distob will take your existing python objects, or a sequence of objects,
and scatter them onto many IPython parallel engines, which may be
running on a single computer or on a cluster.

In place of the original objects, proxy objects are kept on the client
computer that provide the same interface as the original objects. You
can continue to use these as if the objects were still local. All
methods are passed through to the remote objects, where computation is done.

In particular, sending numpy arrays to the cluster is supported. 

A numpy array can also be scattered across the cluster, along a particular axis. Operations on the array can then be automatically done in parallel (either using ufuncs, or by using ``vectorize()`` below)

Note: numpy 1.11.0 or later (not yet released!) is required for full functionality with distributed array arithmetic and ufuncs. You can get a development snapshot of numpy here: https://github.com/numpy/numpy/archive/master.zip

Distob is an object layer built on top of ``ipyparallel``, so it will
make use of your default IPython parallel profile. This allows different
cluster architectures, local CPUs, SSH nodes, PBS, Amazon EC2, etc.

functions
---------
| ``scatter(obj)`` Distribute any object (or list of objects) to remote iPython engines, return a proxy.
| ``gather(obj)`` Fetch back a distributed object (or list), making it local again.
|
| ``vectorize(f)`` Turn an ordinary function (that takes a single object or array) into one that acts in parallel on a scattered list or array. ``apply(f, obj)`` is the same as ``vectorize(f)(obj)``

distributed numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~
| ``scatter(a, axis=2)`` Distribute a single numpy array along axis 2, returning a DistArray.
| 
| Arithmetic operations can freely mix ordinary arrays with the new array types.
| Normal numpy ufuncs can also be used on the distributed arrays.
| Arithmetic and ufunc computations will automatically be routed to an engine, or executed in parallel on several engines, depending on where the data is. (needs numpy>=1.11.0)

| ``concatenate``, ``vstack``, ``hstack``, ``dstack``, ``expand_dims``, ``transpose``, ``rollaxis``, ``split``, ``vsplit``, ``hsplit``, ``dsplit``, ``broadcast_arrays``:
| These work like the numpy functions of the same name. But these can be used with a mix of ordinary ndarrays, RemoteArrays and DistArrays, performing array structural changes while keeping the actual data distributed across multiple engines.
| For example, stacking several RemoteArrays gives a DistArray, without needing to move data.

| The distributed arrays so far support basic indexing, slices and advanced integer indexing.

classes
-------
| ``RemoteArray`` proxy object representing a remote numpy ndarray
| ``DistArray`` a single ndarray distributed across multiple engines
| 
| ``Remote`` base class, used when auto-creating ``Remote*`` proxy classes
| ``@proxy_methods(base)`` class decorator for auto-creating ``Remote*`` proxy classes
| ``ObjectHub`` dict interface giving refs to all distributed objects cluster-wide
| ``ObjectEngine`` dict holding the distributed objects of a single IPython engine
| ``Ref`` reference to a (possibly remote) object

attributes
----------
``engine``: the ``ObjectEngine`` instance on each host (``ObjectHub`` on
the client)

TODO
----
-  Allow assignment to slices of remote arrays.

-  Properly implement caching of remote method results.

-  Auto-creation of proxy classes at runtime (depends uqfoundation/dill#58)

-  For ufunc execution, still need to implement ``reduce``, ``accumulate``, ``reduceat``, ``outer``, ``at`` methods.

-  Make proxy classes more robust, adapting ``wrapt`` (pypi.python.org/pypi/wrapt)

Thanks
------
Incorporates ``pylru.py`` by Jay Hutchinson,
http://github.com/jlhutch/pylru

``ipyparallel`` interactive parallel computing:
https://ipyparallel.readthedocs.org/

``dill`` by Mike McKerns for object serialization, see:
http://trac.mystic.cacr.caltech.edu/project/pathos
