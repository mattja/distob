from __future__ import absolute_import
from .distob import (scatter, gather, call_all, Remote, proxy_methods,
                     ObjectHub, ObjectEngine, Ref)

engine = None
__version__ = '0.1.2'

# If numpy is available, provide RemoteArray and DistArray
try:
    from .arrays import RemoteArray, DistArray
except ImportError:
    pass
