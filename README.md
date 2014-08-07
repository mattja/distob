distob
======
Distributed computing made easier, using remote objects.

Overview
--------
Distob will take your existing python objects, or a sequence of objects, 
and scatter them onto many IPython parallel engines, which may be running on
a single computer or on a cluster.

In place of the original objects, proxy objects are kept on the client
computer that provide the same interface as the original objects.
You can continue to use these as if the objects were still local. All methods
are passed through to the remote objects, where computation is done.

In particular, sending numpy arrays to the cluster is supported. 
(Will require numpy 1.9.0b1 or later for full functionality with remote ufuncs)

Distob is an object layer built on top of IPython.parallel, so it will
make use of your default IPython parallel profile. This allows different 
cluster architectures, local CPUs, SSH nodes, PBS, Amazon EC2, etc.

functions
---------
`scatter(obj)`  Distribute obj to remote iPython engines, return a proxy.  
`gather(obj)`  Fetch back a distributed object, making it local again.

classes
-------
`RemoteArray`   proxy object representing a remote numpy ndarray  

`Remote`   base class, used when creating `Remote*` proxy classes  
`@proxy_methods(base)`   class decorator for creating `Remote*` proxy classes  
`ObjectHub` dict interface giving refs to all distributed objects cluster-wide  
`ObjectEngine` dict holding the distributed objects of a single IPython engine  
`Ref`  reference to a (possibly remote) object  

attributes
----------
`engine`:  the `ObjectEngine` instance on each host (`ObjectHub` on the client)

TODO
----
* Blocking/non-blocking proxy methods

* Finish implementing remote ufunc support for arrays

* Auto-creation of proxy classes at runtime (depends uqfoundation/dill#58)

* Use caching only if specified for a particular method (initially read-only)

* Remote ufunc support with computation routed according to operand location

* Implement the DistArray class to allow a single numpy array to be spread 
  across multiple engines.

* Make proxy classes more robust, adapting `wrapt` (pypi.python.org/pypi/wrapt)

Thanks
------
Incorporates `pylru.py` by Jay Hutchinson (GPLv2+) github.com/jlhutch/pylru

`IPython` parallel computing, see: http://ipython.org/ipython-doc/dev/parallel/

`dill` by Mike McKerns for object serialization, see: http://trac.mystic.cacr.caltech.edu/project/pathos
