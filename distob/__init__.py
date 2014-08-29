from __future__ import absolute_import
from .distob import (scatter, gather, call_all, Remote, proxy_methods,
                     ObjectHub, ObjectEngine, Ref)

engine = None

__version__ = '0.1.3'


# If numpy is available, provide RemoteArray and DistArray
try:
    import numpy as _np
    _have_numpy = True
except ImportError:
    pass
if _have_numpy:
    from .arrays import (RemoteArray, DistArray, transpose, rollaxis,
                         expand_dims, concatenate, vstack, hstack, dstack)
