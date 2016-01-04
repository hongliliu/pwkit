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
MathFunctionLibraryMeta
MathFunctionLibrary

NumpyFunctionLibrary
numpy_library
numpy_types

get_library_for
MathlibDelegatingObject

TidiedFunctionLibraryMeta
TidiedFunctionLibrary
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


# TODO: additional functions?
all_unary_funcs = numpy_unary_ufuncs
all_binary_funcs = numpy_binary_ufuncs



def metaclass_lookup_attribute (attrname, classdict, parents):
    """Um, surely there's a better way to do all of this? We're trying to see if a
    given name is defined anywhere in the under-construction class or its
    superclasses.

    The parent search could be better but I think should only be inefficient,
    not wrong, unless you're really going out of your way to mess with this
    code. We need to use __getattribute__ because we need to interrogate the
    partialmethod objects, which are descriptors.

    """

    attr = classdict.get (attrname)
    if attr is not None:
        return attr

    for parent in parents:
        for pp in parent.mro ():
            try:
                attr = object.__getattribute__ (pp, attrname)
                return attr
            except AttributeError:
                pass

    raise TypeError ('metaclass user must implement "%s" attribute' % attrname)


def metaclass_maybe_add_generic_impl (opname, classdict, parents, generic):
    try:
        impl = metaclass_lookup_attribute (opname, classdict, parents)
    except TypeError:
        impl = None

    if impl is not None and not getattr (impl, '_generic_impl', False):
        return # there is an implementation and it's not generic

    impl = classdict[opname] = partialmethod (generic, opname)
    impl._generic_impl = True


class MathFunctionLibraryMeta (type):
    def __new__ (cls, name, parents, dct):
        generic = metaclass_lookup_attribute ('generic_unary', dct, parents)

        for opname in all_unary_funcs:
            metaclass_maybe_add_generic_impl (opname, dct, parents, generic)

        generic = metaclass_lookup_attribute ('generic_binary', dct, parents)

        for opname in all_binary_funcs:
            metaclass_maybe_add_generic_impl (opname, dct, parents, generic)

        return super (MathFunctionLibraryMeta, cls).__new__ (cls, name, parents, dct)


class MathFunctionLibrary (six.with_metaclass (MathFunctionLibraryMeta, object)):
    def accepts (self, opname, other):
        return False

    def generic_unary (self, opname, x, out=None):
        raise NotImplementedError ('math function "%s" not implemented for objects of type "%s" in %s'
                                   % (opname, x.__class__.__name__, self))

    def generic_binary (self, opname, x, y, out=None):
        raise NotImplementedError ('math function "%s" not implemented for objects of types "%s" '
                                   'and "%s" in %s' % (opname, x.__class__.__name__,
                                                       y.__class__.__name__, self))




numpy_types = np.ScalarType + (np.generic, np.chararray, np.ndarray, np.recarray,
                               np.ma.MaskedArray, list, tuple)

class NumpyFunctionLibrary (MathFunctionLibrary):
    def accepts (self, opname, other):
        return isinstance (other, numpy_types)


def _fill_numpy_library_type ():
    items = {}

    for name in numpy_unary_ufuncs:
        # Look up implementations robustly to keep compat with older Numpys.
        impl = getattr (np, name, None)
        if impl is not None:
            setattr (NumpyFunctionLibrary, name, impl)

    for name in numpy_binary_ufuncs:
        impl = getattr (np, name, None)
        if impl is not None:
            setattr (NumpyFunctionLibrary, name, impl)

_fill_numpy_library_type ()
numpy_library = NumpyFunctionLibrary ()




def get_library_for (x, y=None):
    # Efficiency (?): if it's a standard numpy or builtin type, delegate to
    # that ASAP.

    if isinstance (x, numpy_types) and (y is None or isinstance (y, numpy_types)):
        return numpy_library

    # If either object has a _pk_mathlib_library_ function, it can tell us how
    # to combine the operands.

    library = getattr (x, '_pk_mathlib_library_', None)

    if library is not None and (y is None or library.accepts (None, y)):
        return library

    if y is None:
        raise ValueError ('cannot identify math function library for object '
                          '%r of type %s' % (x, x.__class__.__name__))

    library = getattr (y, '_pk_mathlib_library_', None)
    if library is not None and library.accepts (None, x):
        return library

    raise ValueError ('cannot identify math function library for objects '
                      '%r of type %s and %r of type %s' % (x, x.__class__.__name__,
                                                           y, y.__class__.__name__))


def _dispatch_unary_function (name, x, out=None):
    return getattr (get_library_for (x), name) (x, out)


def _dispatch_binary_function (name, x, y, out=None):
    return getattr (get_library_for (x, y), name) (x, y, out)


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

    __array_priority__ = 2000
    """This tells Numpy that our multiplication function should be used when
    evaluating, say, ``np.linspace(n) * delegating_object``. Plain ndarrays
    have priority 0; Pandas series have priority 1000.

    """
    # https://docs.python.org/2/reference/datamodel.html#basic-customization

    def __dispatch_binary (self, name, other):
        return getattr (get_library_for (self, other), name) (self, other)

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
        return getattr (get_library_for (self, other), 'power') (self, other)

    __lshift__ = partialmethod (__dispatch_binary, 'left_shift')
    __rshift__ = partialmethod (__dispatch_binary, 'right_shift')
    __and__ = partialmethod (__dispatch_binary, 'bitwise_and')
    __xor__ = partialmethod (__dispatch_binary, 'bitwise_xor')
    __or__ = partialmethod (__dispatch_binary, 'bitwise_or')
    __div__ = partialmethod (__dispatch_binary, 'divide')
    __truediv__ = partialmethod (__dispatch_binary, 'true_divide')

    def __dispatch_binary_reflected (self, name, other):
        return getattr (get_library_for (other, self), name) (other, self)

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
        return getattr (get_library_for (self, other), 'power') (other, self)

    __rlshift__ = partialmethod (__dispatch_binary_reflected, 'left_shift')
    __rrshift__ = partialmethod (__dispatch_binary_reflected, 'right_shift')
    __rand__ = partialmethod (__dispatch_binary_reflected, 'bitwise_and')
    __rxor__ = partialmethod (__dispatch_binary_reflected, 'bitwise_xor')
    __ror__ = partialmethod (__dispatch_binary_reflected, 'bitwise_or')

    def __dispatch_binary_inplace (self, name, other):
        return getattr (get_library_for (self, other), name) (self, other, self)

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
        return getattr (get_library_for (self, other), 'power') (self, other, self)

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




class TidiedFunctionLibraryMeta (MathFunctionLibraryMeta):
    # How often do you ever see metaclass inheritance? Not often.
    def __new__ (cls, name, parents, dct):
        generic = metaclass_lookup_attribute ('generic_tidy_unary', dct, parents)

        for opname in all_unary_funcs:
            metaclass_maybe_add_generic_impl ('tidy_'+opname, dct, parents, generic)

        generic = metaclass_lookup_attribute ('generic_tidy_binary', dct, parents)

        for opname in all_binary_funcs:
            metaclass_maybe_add_generic_impl ('tidy_'+opname, dct, parents, generic)

        return super (TidiedFunctionLibraryMeta, cls).__new__ (cls, name, parents, dct)


class TidiedFunctionLibrary (six.with_metaclass (TidiedFunctionLibraryMeta, MathFunctionLibrary)):
    """These function libraries need only implement "tidied" versions of the math
    functions, which can assume that the *out* argument is not None and that
    all arguments have been coerced to uniform types.

    """
    def generic_tidy_unary (self, opname, x, out):
        raise NotImplementedError ('math function "%s" not implemented for objects of type "%s" in %s'
                                   % (opname, x.__class__.__name__, self))

    def generic_tidy_binary (self, opname, x, y, out):
        raise NotImplementedError ('math function "%s" not implemented for objects of types "%s" '
                                   'and "%s" in %s' % (opname, x.__class__.__name__,
                                                       y.__class__.__name__, self))

    def coerce (self, x, y=None, out=None):
        raise NotImplementedError ()

    def empty_like_broadcasted (self, x, y=None):
        raise NotImplementedError ()

    def generic_unary (self, opname, x, out=None):
        x, _, out = self.coerce (x, None, out)
        if out is None:
            out = self.empty_like_broadcasted (x)
        return getattr (self, 'tidy_' + opname) (x, out)

    def generic_binary (self, opname, x, y, out=None):
        x, y, out = self.coerce (x, y, out)
        if out is None:
            out = self.empty_like_broadcasted (x, y)
        return getattr (self, 'tidy_' + opname) (x, y, out)
