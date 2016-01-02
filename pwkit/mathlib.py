# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""Vectorized math functions that work on objects of any type

The basic issue is that Numpy's ufuncs can't be overridden for arbitrary
classes. We implement this feature.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

# __all__ is augmented below:
__all__ = str ('''
numpy_unary_ufuncs
numpy_binary_ufuncs
MathFunctionLibrary
NumpyFunctionLibrary
numpy_library
numpy_types
get_library_for
MathlibDelegatingObject
MathlibDelegatingWrapper
AlwaysOutFunctionLibrary
''').split ()

import abc, operator, six
from functools import partial
from six.moves import range
import numpy as np
from .oo_helpers import partialmethod


# as of 1.10:
numpy_unary_ufuncs = str ('''
abs
absolute
arccos
arccosh
arcsin
arcsinh
arctan
arctanh
bitwise_not
cbrt
ceil
conj
conjugate
cos
cosh
deg2rad
degrees
exp
exp2
expm1
fabs
floor
invert
isfinite
isinf
isnan
log
log10
log1p
log2
logical_not
negative
rad2deg
radians
reciprocal
rint
sign
signbit
sin
sinh
spacing
sqrt
square
tan
tanh
trunc
''').split ()

numpy_binary_ufuncs = str ('''
add
arctan2
bitwise_and
bitwise_or
bitwise_xor
copysign
divide
equal
floor_divide
fmax
fmin
fmod
frexp
greater
greater_equal
hypot
ldexp
left_shift
less
less_equal
logaddexp
logaddexp2
logical_and
logical_or
logical_xor
maximum
minimum
mod
modf
multiply
nextafter
not_equal
power
remainder
right_shift
subtract
true_divide
''').split ()


# TODO: additional functions
all_unary_funcs = numpy_unary_ufuncs
all_binary_funcs = numpy_binary_ufuncs




class _MathFunctionLibraryBase (object):
    def accepts (self, opname, other):
        return False


def _unimplemented_unary_func (name, x, out=None):
    raise NotImplementedError ('math function "%s" not implemented for objects of type "%s"'
                               % (name, x.__class__.__name__))

def _unimplemented_binary_func (name, x, y, out=None):
    raise NotImplementedError ('math function "%s" not implemented for objects of type "%s"'
                               % (name, x.__class__.__name__))

def _make_base_library_type ():
    items = {}

    for name in all_unary_funcs:
        items[name] = partial (_unimplemented_unary_func, name)

    for name in all_binary_funcs:
        items[name] = partial (_unimplemented_binary_func, name)

    return type (str('MathFunctionLibrary'), (_MathFunctionLibraryBase,), items)

MathFunctionLibrary = _make_base_library_type ()




numpy_types = np.ScalarType + (np.generic, np.chararray, np.ndarray, np.recarray,
                               np.ma.MaskedArray, list, tuple)

class _NumpyFunctionLibraryBase (MathFunctionLibrary):
    def accepts (self, opname, other):
        return isinstance (other, numpy_types)


def _make_numpy_library_type ():
    items = {}

    for name in numpy_unary_ufuncs:
        # Look up implementations robustly to keep compat with older Numpys.
        impl = getattr (np, name, None)
        if impl is not None:
            items[name] = impl

    for name in numpy_binary_ufuncs:
        impl = getattr (np, name, None)
        if impl is not None:
            items[name] = impl

    return type (str('NumpyFunctionLibrary'), (_NumpyFunctionLibraryBase,), items)

NumpyFunctionLibrary = _make_numpy_library_type ()

numpy_library = NumpyFunctionLibrary ()




def _try_asarray (thing):
    # This duplicates numutil._try_asarray because I think I might eventually
    # want numutil to be able to depend on mathlib.
    thing = np.asarray (thing)
    if thing.dtype.kind not in 'bifc':
        return None
    return thing


def get_library_for (x):
    # Efficiency (?): if it's a standard numpy or builtin type, delegate to
    # that ASAP.

    if isinstance (x, numpy_types):
        return numpy_library

    # If it has a _pk_mathlib_library_ property, then, well, good:

    library = getattr (x, '_pk_mathlib_library_', None)
    if library is not None:
        return library

    # If the object has a _pk_mathlib_unwrap_ function, it has been asserted
    # to be a completely stateless wrapper for some sub-object on which one
    # may do math.

    unwrap = getattr (x, '_pk_mathlib_unwrap_', None)
    if unwrap is not None:
        return get_library_for (unwrap ())

    raise ValueError ('cannot identify math function library for object '
                      '%r of type %s' % (x, x.__class__.__name__))




def _dispatch_unary_function (name, x, out=None):
    # Efficiency (?): if it's a standard numpy or builtin type, delegate to
    # that ASAP. TODO: use castable_n

    if isinstance (x, numpy_types):
        return getattr (numpy_library, name) (x, out)

    # If it has a _pk_mathlib_library_ property, it's telling us how to do
    # math on it.

    library = getattr (x, '_pk_mathlib_library_', None)
    if library is not None:
        return getattr (library, name) (x, out)

    # If the object has a _pk_mathlib_unwrap_ function, it has been asserted
    # to be a completely stateless wrapper for some sub-object on which one
    # may do math.

    unwrap = getattr (x, '_pk_mathlib_unwrap_', None)
    if unwrap is not None:
        unwrapped = unwrap ()

        if out is None:
            result = _dispatch_unary_function (name, unwrapped, out)
            return x._pk_mathlib_rewrap_ (result)

        if out is x:
            _dispatch_unary_function (name, unwrapped, unwrapped)
            return x

        # We'll just have to hope that the implementation knows what to do with
        # whatever the caller is doing.
        _dispatch_unary_function (name, unwrapped, out)
        return out

    raise ValueError ('cannot determine how to apply math function "%s" to object '
                      '%r of type %s' % (name, x, x.__class__.__name__))


def _dispatch_binary_function (name, x, y, out=None):
    # Efficiency (?): if they're both standard numpy or builtin types,
    # delegate to numpy ASAP.

    if isinstance (x, numpy_types) and isinstance (y, numpy_types):
        return getattr (numpy_library, name) (x, y, out)

    # If either object has a _pk_mathlib_library_ function, it can tell us how
    # to combine the operands.

    library = getattr (x, '_pk_mathlib_library_', None)
    if library is not None and library.accepts (name, y):
        return getattr (library, name) (x, y, out)

    library = getattr (y, '_pk_mathlib_library_', None)
    if library is not None and library.accepts (name, x):
        return getattr (library, name) (x, y, out)

    # If either object has a _pk_mathlib_unwrap_ function, it has been
    # asserted to be a completely stateless wrapper for some sub-object on
    # which one may do math.

    unwrap = getattr (x, '_pk_mathlib_unwrap_', None)
    if unwrap is not None:
        unwrapped = unwrap ()

        if out is None:
            result = _dispatch_binary_function (name, unwrapped, y, out)
            return x._pk_mathlib_rewrap_ (result)

        if out is x:
            _dispatch_binary_function (name, unwrapped, y, unwrapped)
            return x

        _dispatch_binary_function (name, unwrapped, y, out)
        return out

    unwrap = getattr (y, '_pk_mathlib_unwrap_', None)
    if unwrap is not None:
        unwrapped = unwrap ()

        if out is None:
            result = _dispatch_binary_function (name, x, unwrapped, out)
            return y._pk_mathlib_rewrap_ (result)

        if out is y:
            _dispatch_binary_function (name, x, unwrapped, unwrapped)
            return y

        _dispatch_binary_function (name, x, unwrapped, out)
        return out

    # Finally, maybe we can convert one or both of the objects to an ndarray,
    # and maybe things will work out better after we do that.

    a = _try_asarray (x)
    if a is not None and a is not x:
        return _dispatch_binary_function (name, a, y, out)

    a = _try_asarray (y)
    if a is not None and a is not y:
        return _dispatch_binary_function (name, x, a, out)

    raise ValueError ('cannot determine how to apply math function "%s" to objects '
                      '%r (type %s) and %r (type %s)' % (name, x, x.__class__.__name__,
                                                         y, y.__class__.__name__))


def _create_wrappers (namespace):
    """This function populates the global namespace with functions dispatching the
    unary and binary math functions.

    """

    for name in all_unary_funcs:
        namespace[name] = partial (_dispatch_unary_function, name)

    for name in all_binary_funcs:
        namespace[name] = partial (_dispatch_binary_function, name)

_create_wrappers (globals ())
__all__ += all_unary_funcs
__all__ += all_binary_funcs




class MathlibDelegatingObject (object):
    """Inherit from this class to delegate all math operators to the mathlib
    dispatch mechanism. You must set the :attr:`_pk_mathlib_library_`
    attribute to an instance of :class:`MathFunctionLibrary`.

    Here are math-ish functions **not** provided by this class that you may
    want to implement separately:

    __divmod__
      Division-and-modulus operator.
    __rdivmod__
      Reflected division-and-modulus operator.
    __idivmod__
      In-place division-and-modulus operator.
    __pos__
      Unary positivization operator.
    __complex__
      Convert to a complex number.
    __int__
      Convert to a (non-"long") integer.
    __long__
      Convert to a long.
    __float__
      Convert to a float.
    __index__
      Convert to an integer (int or long)

    """
    _pk_mathlib_library_ = None

    # https://docs.python.org/2/reference/datamodel.html#basic-customization

    def __dispatch_binary (self, name, other):
        return _dispatch_binary_function (name, self, other)

    __lt__ = partialmethod (__dispatch_binary, 'less')
    __le__ = partialmethod (__dispatch_binary, 'less_equal')
    __eq__ = partialmethod (__dispatch_binary, 'equal')
    __ne__ = partialmethod (__dispatch_binary, 'not_equal')
    __gt__ = partialmethod (__dispatch_binary, 'greater')
    __ge__ = partialmethod (__dispatch_binary, 'greater_equal')

    # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types

    __add__ = partialmethod (__dispatch_binary, 'add')
    __sub__ = partialmethod (__dispatch_binary, 'subtract')
    __mul__ = partialmethod (__dispatch_binary, 'multiply')
    __floordiv__ = partialmethod (__dispatch_binary, 'floor_divide')
    __mod__ = partialmethod (__dispatch_binary, 'mod')
    #__divmod__ = NotImplemented

    def __pow__ (self, other, modulo=None):
        if modulo is not None:
            raise NotImplementedError ()
        return _dispatch_binary_function ('power', self, other)

    __lshift__ = partialmethod (__dispatch_binary, 'left_shift')
    __rshift__ = partialmethod (__dispatch_binary, 'right_shift')
    __and__ = partialmethod (__dispatch_binary, 'bitwise_and')
    __xor__ = partialmethod (__dispatch_binary, 'bitwise_xor')
    __or__ = partialmethod (__dispatch_binary, 'bitwise_or')
    __div__ = partialmethod (__dispatch_binary, 'divide')
    __truediv__ = partialmethod (__dispatch_binary, 'true_divide')

    def __dispatch_binary_reflected (self, name, other):
        return _dispatch_binary_function (name, other, self)

    __radd__ = partialmethod (__dispatch_binary_reflected, 'add')
    __rsub__ = partialmethod (__dispatch_binary_reflected, 'subtract')
    __rmul__ = partialmethod (__dispatch_binary_reflected, 'multiply')
    __rdiv__ = partialmethod (__dispatch_binary_reflected, 'divide')
    __rtruediv__ = partialmethod (__dispatch_binary_reflected, 'true_divide')
    __rfloordiv__ = partialmethod (__dispatch_binary_reflected, 'floor_divide')
    __rmod__ = partialmethod (__dispatch_binary_reflected, 'mod')
    #__divmod__ = NotImplemented

    def __rpow__ (self, other, modulo=None):
        if modulo is not None:
            raise NotImplementedError ()
        return _dispatch_binary_function ('power', other, self)

    __rlshift__ = partialmethod (__dispatch_binary_reflected, 'left_shift')
    __rrshift__ = partialmethod (__dispatch_binary_reflected, 'right_shift')
    __rand__ = partialmethod (__dispatch_binary_reflected, 'bitwise_and')
    __rxor__ = partialmethod (__dispatch_binary_reflected, 'bitwise_xor')
    __ror__ = partialmethod (__dispatch_binary_reflected, 'bitwise_or')

    def __dispatch_binary_inplace (self, name, other):
        return _dispatch_binary_function (name, self, other, self)

    __iadd__ = partialmethod (__dispatch_binary_inplace, 'add')
    __isub__ = partialmethod (__dispatch_binary_inplace, 'subtract')
    __imul__ = partialmethod (__dispatch_binary_inplace, 'multiply')
    __idiv__ = partialmethod (__dispatch_binary_inplace, 'divide')
    __itruediv__ = partialmethod (__dispatch_binary_inplace, 'true_divide')
    __ifloordiv__ = partialmethod (__dispatch_binary_inplace, 'floor_divide')
    __imod__ = partialmethod (__dispatch_binary_inplace, 'mod')
    #__idivmod__ = NotImplemented

    def __ipow__ (self, other, modulo=None):
        if modulo is not None:
            raise NotImplementedError ()
        return _dispatch_binary_function ('power', self, other, self)

    __ilshift__ = partialmethod (__dispatch_binary_inplace, 'left_shift')
    __irshift__ = partialmethod (__dispatch_binary_inplace, 'right_shift')
    __iand__ = partialmethod (__dispatch_binary_inplace, 'bitwise_and')
    __ixor__ = partialmethod (__dispatch_binary_inplace, 'bitwise_xor')
    __ior__ = partialmethod (__dispatch_binary_inplace, 'bitwise_or')

    def __neg__ (self):
        return self._pk_mathlib_library_.negative (self)

    #def __pos__ (self):
    #    raise NotImplementedError

    def __abs__ (self):
        return self._pk_mathlib_library_.absolute (self)

    def __invert__ (self):
        return self._pk_mathlib_library_.bitwise_not (self)


class MathlibDelegatingWrapper (MathlibDelegatingObject):
    """This class is kind of trivial, except that it inherits from
    :class:`MathlibDelegatingObject` and so it gets all of the standard math
    operators implemented on it with correct coercion and delegation, for
    free.

    """
    def __init__ (self, data):
        self._mathlib_data = data

    @classmethod
    def _pk_mathlib_rewrap_ (cls, data):
        return cls (data)

    def _pk_mathlib_unwrap_ (self):
        return self._mathlib_data




class _AlwaysOutFunctionLibraryBase (MathFunctionLibrary):
    def __init__ (self, sub_library):
        self.sub_library = sub_library

    def accepts (self, opname, other):
        return self.sub_library.accepts (opname, other)

    def _delegate_unary (self, name, x, out=None):
        x, _, out = self.sub_library.coerce (x, None, out)
        if out is None:
            out = self.sub_library.empty_like_broadcasted (x)
        getattr (self.sub_library, name) (x, out)
        return out

    def _delegate_binary (self, name, x, y, out=None):
        x, y, out = self.sub_library.coerce (x, y, out)
        if out is None:
            out = self.sub_library.empty_like_broadcasted (x, y)
        getattr (self.sub_library, name) (x, y, out)
        return out


def _make_always_out_library_type ():
    items = {}

    for name in all_unary_funcs:
        items[name] = partialmethod (_AlwaysOutFunctionLibraryBase._delegate_unary, name)

    for name in all_binary_funcs:
        items[name] = partialmethod (_AlwaysOutFunctionLibraryBase._delegate_binary, name)

    return type (str('AlwaysOutFunctionLibrary'), (_AlwaysOutFunctionLibraryBase,), items)

AlwaysOutFunctionLibrary = _make_always_out_library_type ()
