#!/usr/bin/env python
import threading
from functools import wraps
import inspect
import warnings


__author__ = 'Darryl Oatridge'


def singleton(make_instance):
    """Pattern to support a threadsafe singleton.

    Usage: in your singleton override the parent __new__(cls) with the following
        @singleton
        def __new__(cls):
            return super().__new__(cls)

    """
    __lock = threading.Lock()
    __instance = None

    @wraps(make_instance)
    def __new__(cls, *args, **kwargs):
        nonlocal __instance

        if __instance is None:
            with __lock:
                if __instance is None:
                    __instance = make_instance(cls, *args, **kwargs)

        return __instance

    return __new__


def deprecated(reason):
    """
    Much thanks to Laurent LAPORTE for this code from Stack Overflow
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    string_types = (type(b''), type(u''))

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "The class {name} has been deprecated. ({reason})."
            else:
                fmt1 = "The function {name} has been depricated.({reason})."

            @wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "The class {name} has been depricated."
        else:
            fmt2 = "The function {name} has been depricated."

        @wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
