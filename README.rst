distob
======

| Distributed computing made easier, using remote objects
|  N.B. this is a development pre-release: lots not yet working, API subject to change!

Overview
--------

Distob will take your existing python objects, or a sequence of objects,
and scatter them onto many IPython parallel engines, which may be
running on a single computer or on a cluster.

In place of the original objects, proxy objects are kept on the client
computer that provide the same interface as the original objects. You
can continue to use these as if the objects were still local. All
methods are passed through to the remote objects, where computation is
done.

In particular, sending numpy arrays to the cluster is supported. (Will
require numpy 1.9.0b1 or later for full functionality with remote
ufuncs)

A numpy array can also be scattered across the cluster, along a particular 
axis. Operations on the array will then automatically be split up and done 
in parallel (still work in progress).

Distob is an object layer built on top of IPython.parallel, so it will
make use of your default IPython parallel profile. This allows different
cluster architectures, local CPUs, SSH nodes, PBS, Amazon EC2, etc.

functions
---------

| ``scatter(obj)`` Distribute any object (or list of objects) to remote iPython engines, return a proxy.
| ``gather(obj)`` Fetch back a distributed object (or list), making it local again.

distributed numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~

| ``scatter(a, axis=2)`` Distribute a single numpy array ``a`` by splitting into pieces along axis 2, returning a DistArray.
| ``concatenate((a1, a2, ...), axis=0)`` Join a sequence of arrays together, handling multiple ``RemoteArray`` and ``DistArray`` without moving data.
| ``vstack(tup)`` Stack arrays in sequence vertically (row wise), handling ``RemoteArray`` and ``DistArray`` without moving data.
| ``hstack(tup)`` Stack arrays in sequence horizontally (column wise), handling ``RemoteArray`` and ``DistArray`` without moving data.
| ``dstack(tup)`` Stack arrays in sequence depth wise (along third dimension), handling ``RemoteArray`` and ``DistArray`` without moving data.

classes
-------

| ``RemoteArray`` proxy object representing a remote numpy ndarray
| ``DistArray`` a single ndarray distributed across multiple engines

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

-  Blocking/non-blocking proxy methods

-  Finish implementing remote ufunc support for arrays, with computation routed according to operand location.

-  Auto-creation of proxy classes at runtime (depends
   uqfoundation/dill#58)

-  Use caching only if specified for a particular method (initially
   read-only methods)

-  Make proxy classes more robust, adapting ``wrapt``
   (pypi.python.org/pypi/wrapt)

Thanks
------

Incorporates ``pylru.py`` by Jay Hutchinson,
http://github.com/jlhutch/pylru

``IPython`` parallel computing, see:
http://ipython.org/ipython-doc/dev/parallel/

``dill`` by Mike McKerns for object serialization, see:
http://trac.mystic.cacr.caltech.edu/project/pathos
