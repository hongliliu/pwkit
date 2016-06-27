# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""The :mod:`numpy` package provides a powerful framework for manipulating and
querying N-dimensional arrays of numerical data. It provides a suite of
functions, such as :func:`numpy.shape` and :func:`numpy.multiply`, that
provide a consistent interface for performing these operations, even if these
inputs are (say) standard Python lists and not :class:`numpy.ndarray`
instances.

Unfortunately, the Numpy functions do not do a good job of handling
“array-like” Python objects that may look and behave much like N-dimensional
Numpy arrays, but do not share a common implementation. The :mod:`pwkit`
package provides two objects along these lines: :mod:`pwkit.msmt` measurement
arrays, and :mod:`pwkit.pktable` table columns. The reason for this module's
existence are the measurement array classes, which need to override the
lowest-level math operations.

The :mod:`pwkit.mathlib` package makes it so that you can use the same suite
of Numpy-like functions — the “Common Interface” — to operate on any of these
objects, or reasonable combinations of them. The same infrastructure makes it
so that you can use the standard Python mathematical operators on combinations
of these objects as well.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

# __all__ is augmented below:
__all__ = str ('''
broadcast_shapes
try_asarray
Signatures
Flags
FunctionSpecification
common_interface_functions
MathFunctionLibrary
NumpyFunctionLibrary
numpy_library
numpy_types
TidiedFunctionLibrary
MathlibDelegatingObject
''').split ()

import six
from functools import partial
from six.moves import range
import numpy as np
from .oo_helpers import partialmethod
from .simpleenum import enumeration


# A few generic helpers

def broadcast_shapes (*shapes):
    """Given a set of array shapes, compute the shape of their mutual broadcast
    using Numpy rules. Numpy can do this with actual arrays but can't do it
    with shape information only.

    We assume that the input shapes are all valid: tuples with nonnegative
    integer values.

    """
    maxlen = np.max ([len (s) for s in shapes])
    padded = np.array ([(1,) * (maxlen - len (s)) + s for s in shapes])
    result = padded[0]

    for i in range (maxlen):
        for j in range (1, len (shapes)):
            x = padded[j,i]

            if x != 1:
                if result[i] == 1:
                    result[i] = x
                elif result[i] != x:
                    raise ValueError ('shape mismatch: objects cannot be broadcast to a single shape')

    return tuple (result)


def try_asarray (thing, fail_mode='raise'):
    """Try to convert *thing* to a :class:`numpy.ndarray`.

    Unlike :func:`numpy.asarray`, this will only accept things that appear to
    contain numerical values. If *thing* appears not to be numerical, the
    behavior depends on *fail_mode*. If it is the string ``"raise"`` (the
    default), :exc:`ValueError` will be raised. If it is ``"none"``, ``None``
    will be returned. Otherwise an :exc:`Exception` will be raised complaining
    about the value given for *fail_mode*.

    In particular, we call :func:`~numpy.asarray` and examine the
    :attr:`numpy.dtype.kind` of the returned array. If it is not boolean
    (``b``), integer (``i``), floating-point (``f``), or complex (``c``), we
    consider the array not to be numerical.

    The context is that :func:`~numpy.asarray` will accept literally anything
    as an argument. If given a random Python object, it will return a scalar
    or array of dtype ``Object``, which is often not helpful.

    """
    thing = np.asarray (thing)
    if thing.dtype.kind in 'bifc':
        return thing

    if fail_mode == 'raise':
        raise ValueError ('cannot treat %r as a data array' % (thing,))
    elif fail_mode == 'none':
        return None
    raise Exception ('unrecognized try_asarray() fail_mode of %r' % (fail_mode,))


# The real meat starts with the Common Interface. This defines the set of
# operations on "array-like" objects that we support -- although
# implementations for particular objects are free to not implement some, or
# many, of the these operations. We aim to precisely match the semantics of
# Numpy whenever possible.

@enumeration
class Signatures (object):
    std_unary = 0
    """A standard unary function compatible with most of Numpy's unary universal
    functions ("ufuncs"). These have signature (x, out=None, **kwargs) and
    return one output array (that may just be *out* if it was provided).

    """
    std_binary = 1
    """A standard binary function compatible with most of Numpy's binary universal
    functions ("ufuncs"). These have signature (x, y, out=None, **kwargs) and
    return one output array (that may just be *out* if it was provided) that
    has a shape resulting from broadcasting *x* and *y*.

    """
    other_1 = 2
    """Some other non-standard signature for which only the first argument is an
    array-like object.

    """


@enumeration
class Flags (object):
    none = 0

    has_numpy_impl = 1 << 0
    """If set, there is a function in the main :mod:`numpy` module that implements
    this function with precisely the semantics that we aim to provide;
    although, of course, the numpy implementation is only guaranteed to work
    on Numpy-native types.

    """

    bool_result = 1 << 1
    """For std_unary or std_binary functions, indicates that the type of the
    result is always `numpy.bool`, regardless of the input types. Example:
    ``greater_equal``.

    """

    ints_only = 1 << 2
    """Indicates that the operation is only valid on integer inputs.

    This flag isn't currently used, but it seems good to annotate it while I'm
    paying attention to the ufunc semantics.

    """


class FunctionSpecification (object):
    name = None
    signature = None
    flags = Flags.none
    _impl = None

    def __init__ (self, name, signature, flags):
        self.name = name
        self.signature = signature
        self.flags = flags

    def __call__ (self, *args, **kwargs):
        return self._impl (*args, **kwargs)


FS = FunctionSpecification # temporary for init
s = Signatures
f = Flags

common_interface_functions = dict ((s.name, s) for s in [
    FS ('absolute',      s.std_unary,   f.has_numpy_impl),
    FS ('add',           s.std_binary,  f.has_numpy_impl),
    FS ('append',        s.other_1,     f.none),
    FS ('arccos',        s.std_unary,   f.has_numpy_impl),
    FS ('arccosh',       s.std_unary,   f.has_numpy_impl),
    FS ('arcsin',        s.std_unary,   f.has_numpy_impl),
    FS ('arcsinh',       s.std_unary,   f.has_numpy_impl),
    FS ('arctan',        s.std_unary,   f.has_numpy_impl),
    FS ('arctan2',       s.std_unary,   f.has_numpy_impl),
    FS ('arctanh',       s.std_unary,   f.has_numpy_impl),
    FS ('bitwise_and',   s.std_binary,  f.has_numpy_impl | f.ints_only),
    FS ('bitwise_or',    s.std_binary,  f.has_numpy_impl | f.ints_only),
    FS ('bitwise_xor',   s.std_binary,  f.has_numpy_impl | f.ints_only),
    FS ('broadcast_to',  s.other_1,     f.has_numpy_impl),
    FS ('cbrt',          s.std_unary,   f.has_numpy_impl),
    FS ('ceil',          s.std_unary,   f.has_numpy_impl),
    FS ('cmask',         s.other_1,     f.bool_result),
    FS ('conjugate',     s.std_unary,   f.has_numpy_impl),
    FS ('copysign',      s.std_binary,  f.has_numpy_impl),
    FS ('cos',           s.std_unary,   f.has_numpy_impl),
    FS ('cosh',          s.std_unary,   f.has_numpy_impl),
    FS ('deg2rad',       s.std_unary,   f.has_numpy_impl),
    FS ('divide',        s.std_binary,  f.has_numpy_impl),
    FS ('get_dtype',     s.other_1,     f.none),
    FS ('get_size',      s.other_1,     f.none),
    FS ('equal',         s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('exp',           s.std_unary,   f.has_numpy_impl),
    FS ('exp2',          s.std_unary,   f.has_numpy_impl),
    FS ('expm1',         s.std_unary,   f.has_numpy_impl),
    FS ('fabs',          s.std_unary,   f.has_numpy_impl),
    FS ('floor',         s.std_unary,   f.has_numpy_impl),
    FS ('floor_divide',  s.std_binary,  f.has_numpy_impl),
    FS ('fmax',          s.std_binary,  f.has_numpy_impl),
    FS ('fmin',          s.std_binary,  f.has_numpy_impl),
    FS ('fmod',          s.std_binary,  f.has_numpy_impl),
    FS ('frexp',         s.other_1,     f.has_numpy_impl),
    FS ('greater',       s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('greater_equal', s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('hypot',         s.std_binary,  f.has_numpy_impl),
    FS ('invert',        s.std_unary,   f.has_numpy_impl),
    FS ('isfinite',      s.std_unary,   f.has_numpy_impl | f.bool_result),
    FS ('isinf',         s.std_unary,   f.has_numpy_impl | f.bool_result),
    FS ('isnan',         s.std_unary,   f.has_numpy_impl | f.bool_result),
    FS ('ldexp',         s.std_binary,  f.has_numpy_impl),
    FS ('left_shift',    s.std_binary,  f.has_numpy_impl | f.ints_only),
    FS ('less',          s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('less_equal',    s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('log',           s.std_unary,   f.has_numpy_impl),
    FS ('log10',         s.std_unary,   f.has_numpy_impl),
    FS ('log1p',         s.std_unary,   f.has_numpy_impl),
    FS ('log2',          s.std_unary,   f.has_numpy_impl),
    FS ('logaddexp',     s.std_binary,  f.has_numpy_impl),
    FS ('logaddexp2',    s.std_binary,  f.has_numpy_impl),
    FS ('logical_and',   s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('logical_or',    s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('logical_not',   s.std_unary,   f.has_numpy_impl | f.bool_result),
    FS ('logical_xor',   s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('maximum',       s.std_binary,  f.has_numpy_impl),
    FS ('minimum',       s.std_binary,  f.has_numpy_impl),
    FS ('modf',          s.other_1,     f.has_numpy_impl),
    FS ('multiply',      s.std_binary,  f.has_numpy_impl),
    FS ('negative',      s.std_unary,   f.has_numpy_impl),
    FS ('nextafter',     s.std_binary,  f.has_numpy_impl),
    FS ('not_equal',     s.std_binary,  f.has_numpy_impl | f.bool_result),
    FS ('power',         s.std_binary,  f.has_numpy_impl),
    FS ('rad2deg',       s.std_unary,   f.has_numpy_impl),
    FS ('reciprocal',    s.std_unary,   f.has_numpy_impl),
    FS ('remainder',     s.std_binary,  f.has_numpy_impl),
    FS ('repvals',       s.other_1,     f.none),
    FS ('reshape',       s.other_1,     f.none),
    FS ('right_shift',   s.std_binary,  f.has_numpy_impl | f.ints_only),
    FS ('rint',          s.std_unary,   f.has_numpy_impl),
    FS ('shape',         s.other_1,     f.has_numpy_impl),
    FS ('sign',          s.std_unary,   f.has_numpy_impl),
    FS ('signbit',       s.std_unary,   f.has_numpy_impl | f.bool_result),
    FS ('sin',           s.std_unary,   f.has_numpy_impl),
    FS ('sinh',          s.std_unary,   f.has_numpy_impl),
    FS ('spacing',       s.std_unary,   f.has_numpy_impl),
    FS ('sqrt',          s.std_unary,   f.has_numpy_impl),
    FS ('square',        s.std_unary,   f.has_numpy_impl),
    FS ('subtract',      s.std_binary,  f.has_numpy_impl),
    FS ('tan',           s.std_unary,   f.has_numpy_impl),
    FS ('tanh',          s.std_unary,   f.has_numpy_impl),
    FS ('true_divide',   s.std_binary,  f.has_numpy_impl),
    FS ('trunc',         s.std_unary,   f.has_numpy_impl),
])

del FS, s, f




# Now we define the base MathFunctionLibrary that implements all of these
# functions for some input type ... as well as a few helper function.

class MathFunctionLibrary (object):
    """Instaces of this class implement the :mod:`pwkit.mathlib` API for some
    class of objects. For each free function of the “common interface”
    implemented in :mod:`pwkit.mathlib`, this class has corresponding method
    that is called. This class has a few extra methods that are not part of
    the common interface, but are needed to make the whole system work.

    """
    def __str__ (self):
        return self.__class__.__name__

    # These methods are helpers for the machinery below. Some of them are only
    # used in certain subclasses (e.g. TidiedFunctionLibrary), but they're
    # generic enough that it seems good to define them at the top of the
    # hierarchy. The common pattern here is these are functions where the
    # natural first argument is *not* an array-like object.

    def accepts (self, other):
        """Return a boolean indicating whether this math library is able to operate on
        objects having the type of *other*.

        """
        return False

    def new_empty (self, shape, dtype):
        """Return a newly-allocated array object having the specified shape and
        element dtype.

        """
        raise NotImplementedError ()

    def typeconvert (self, thing):
        """Convert *thing* to some standard array-like type.

        This will be called on all arguments to and outputs of the standard
        unary and binary functions so that the tidied versions of functions to
        not need to painstakingly special-case whether each argument is a
        Python scalar, a Numpy array, etc. Because this function is used for
        output arguments, if at all possible it should be the case that
        modifications to the the returned object propagate back to *thing*. If
        *thing* cannot sanely be converted, a `ValueError` is raised.

        """
        return thing

    # These methods implement Common Interface functions in a really generic way
    # that ought to be useful in the vast majority of cases.

    def get_dtype (self, x):
        return x.dtype

    def get_size (self, x):
        return np.prod (self.shape (x), dtype=np.int)

    def shape (self, x):
        return x.shape

    # The default implementations for the Common Interface functions are added
    # to this class in _fill_base_library_type() below.

    def _not_implemented (self, opname, x, *args, **kwargs):
        """This function is called when an unimplemented Common Interface function is
        called.

        """
        raise NotImplementedError ('math function "%s" not implemented for objects of type "%s" in %s'
                                   % (opname, x.__class__.__name__, self))


def _fill_base_library_type ():
    ni = MathFunctionLibrary._not_implemented

    for fs in six.viewvalues (common_interface_functions):
        impl = getattr (MathFunctionLibrary, fs.name, None)
        if impl is None:
            setattr (MathFunctionLibrary, fs.name, partialmethod (ni, fs.name))

_fill_base_library_type ()




# Now we implement the NumpyFunctionLibrary class that delegates almost
# everything to Numpy itself.

numpy_types = np.ScalarType + (np.generic, np.chararray, np.ndarray, np.recarray,
                               np.ma.MaskedArray, list, tuple)

class NumpyFunctionLibrary (MathFunctionLibrary):
    """This subclass is used to execute all math operations on array-like objects
    that stock Numpy can handle. Its implementations of “ufunc” functions like
    ``add`` and ``square`` delegate directly to their Numpy equivalents.
    However, it still needs to implement the methods of the Common Interface
    that aren't in Numpy.

    """
    # Mathlib machinery helpers:

    def accepts (self, other):
        return isinstance (other, numpy_types)

    def new_empty (self, shape, dtype):
        return np.empty (shape, dtype=dtype)

    def typeconvert (self, thing):
        return try_asarray (thing)

    # Implementations of the few Common Interface functions that are *not*
    # provided by Numpy itself:

    def append (self, x, y):
        return np.concatenate ((x, y))


    def cmask (self, x, welldefined=False, finite=False):
        x = np.asarray (x)
        zerod = (x.shape == ())
        x = np.atleast_1d (x)

        out = np.ones (x.shape, dtype=np.bool)

        if welldefined:
            np.logical_and (out, ~np.isnan (x), out)

        if finite:
            np.logical_and (out, np.isfinite (x), out)

        if zerod:
            return out.reshape (())
        return out


    def get_dtype (self, x):
        return np.asarray (x).dtype


    def get_size (self, x):
        return np.asarray (x).size


    def repvals (self, x):
        return np.array (x, copy=True)


    def reshape (self, x, s):
        # I'm not sure why we can't just have this be implemented with
        # np.reshape ... but I got some infinite recursion that makes me
        # believe that we can't.
        return np.asarray (x).reshape (s)

    # The implementations for the bulk of the Common Interface functions,
    # which just delegate to Numpy, are added in to the singleton instance
    # below.

numpy_library = NumpyFunctionLibrary ()

def _fill_numpy_library_instance ():
    """We allow the Numpy implementations to not be available to keep compatiblity
    with older versions of Numpy, since the list of provided ufuncs has grown
    with time.

    """
    for fs in six.viewvalues (common_interface_functions):
        if not (fs.flags & Flags.has_numpy_impl):
            continue

        impl = getattr (np, fs.name, None)
        if impl is not None:
            setattr (numpy_library, fs.name, impl)

_fill_numpy_library_instance ()




# Now we can implement _get_library_for() and the actual implementations of
# the freestanding Common Interface functions. These handle objects with
# __array__ and __array_wrap__ -- which are best handled separately -- and
# look up the right MathFunctionLibrary once all the wrapping and unwrapping
# has been dealt with.

def _get_library_for (*objects):
    """Given array-like objects, return a `MathFunctionLibrary` instance that can
    perform Common Interface math functions on all of them.

    If the inputs are scalars, instances of `numpy.generic`,
    `numpy.chararray`, `numpy.ndarray`, `numpy.recarray`,
    `numpy.ma.MaskedArray`, `list`, or `tuple`, an instance
    of`NumpyFunctionLibrary` is used, which essentially directly delegates to
    the corresponding functions in the `numpy` module.

    Otherwise, each object is checked for a ``_pk_mathlib_library_``
    attribute, which should be an instance of `MathFunctionLibrary`. The
    `~MathFunctionLibrary.accepts` method is used to check that the chosen
    library is compatible with the inputs.

    """
    # Efficiency (?): check pure-numpy case ASAP.

    if all (isinstance (o, numpy_types) for o in objects):
        return numpy_library

    for obj in objects:
        library = getattr (obj, '_pk_mathlib_library_', None)
        if library is None:
            continue

        for obj2 in objects:
            # Note that we still do this check when ``obj2 is obj``; but who cares?
            if not library.accepts (obj2):
                break
        else:
            # This path is called when we don't break out of the loop => this
            # library can handle all objects.
            return library

    # If we got here, no library dealt with everything. Boo.

    raise ValueError ('cannot identify math function library for input(s): ' +
                      ', '.join ('%s (%s)' % (o, o.__class__.__name__) for o in objects))


def _dispatch_std_unary (name, x, out=None, **kwargs):
    rewrap = lambda x: x
    unwrap = getattr (x, '__array__', None)

    if unwrap is not None:
        if out is None:
            rewrap = x.__array_wrap__
        x = unwrap ()

    if out is None:
        library = _get_library_for (x)
    else:
        unwrap = getattr (out, '__array__', None)
        if unwrap is not None:
            out = unwrap ()
        library = _get_library_for (x, out)

    return rewrap (getattr (library, name) (x, out, **kwargs))


def _dispatch_std_binary (name, x, y, out=None, **kwargs):
    """We have to be a bit careful about how we deal with wrapping and unwrapping
    arrays. Say x is a measurement and y is an ndarray. x does not provide
    __array_wrap__, but y does. If we don't pay attention to x's
    __array_priority__, we end up trying to call ndarray.__array_wrap__ on a
    Measurement, which makes it barf.

    """
    unwrap = getattr (x, '__array__', None)
    xpriority = getattr (x, '__array_priority__', 0)
    xrewrap = getattr (x, '__array_wrap__', None)

    if unwrap is not None:
        x = unwrap ()

    unwrap = getattr (y, '__array__', None)
    ypriority = getattr (y, '__array_priority__', 0)
    yrewrap = getattr (y, '__array_wrap__', None)

    if unwrap is not None:
        y = unwrap ()

    rewrap = lambda x: x

    if out is not None:
        unwrap = getattr (out, '__array__', None)
        if unwrap is not None:
            out = unwrap ()
        library = _get_library_for (x, y, out)
    else:
        library = _get_library_for (x, y)

        if xpriority >= ypriority:
            if xrewrap is not None:
                rewrap = xrewrap
        elif yrewrap is not None:
            rewrap = yrewrap

    return rewrap (getattr (library, name) (x, y, out, **kwargs))


def _dispatch_other_1 (name, x, *args, **kwargs):
    unwrap = getattr (x, '__array__', None)
    if unwrap is not None:
        x = unwrap ()

    library = _get_library_for (x)
    return getattr (library, name) (x, *args, **kwargs)


_dispatchers = {
    Signatures.std_unary: _dispatch_std_unary,
    Signatures.std_binary: _dispatch_std_binary,
    Signatures.other_1: _dispatch_other_1,
}

def _create_wrappers (namespace):
    """This function populates the global namespace with functions dispatching the
    unary and binary math functions.

    """
    for fs in six.viewvalues (common_interface_functions):
        namespace[fs.name] = fs._impl = partial (_dispatchers[fs.signature], fs.name)

_create_wrappers (globals ())
__all__ += list (six.viewkeys (common_interface_functions))


# Aliases

abs = absolute
bitwise_not = invert
conj = conjugate
degrees = rad2deg
mod = remainder
radians = deg2rad

__all__ += str('abs bitwise_not conj degrees mod radians').split ()


# These functions largely leverage the mathlib machinery, but don't map
# exactly to Common Interface functions -- because the underlying shared
# implementation is one that doesn't necessarily take an array-like as its
# first argument.

def empty_like (x, shape=None, dtype=None):
    """Return a new, uninitialized array-like object similar to *x*.

    If the *shape* or *dtype* keyword arguments are provided, they will be
    used in specifying those properties of the new object; otherwise, the
    current values of *x* will be used.

    """
    rewrap = lambda x: x
    unwrap = getattr (x, '__array__', None)

    if unwrap is not None:
        rewrap = x.__array_wrap__
        x = unwrap ()

    library = _get_library_for (x)

    if shape is None:
        shape = library.get_shape (x)
    if dtype is None:
        dtype = library.get_dtype (x)

    return rewrap (library.new_empty (shape, dtype))

__all__ += str('empty_like').split ()




# Tidied

class TidiedFunctionLibrary (MathFunctionLibrary):
    """This class makes it a lot easier to implement the Common Interface for new
    Python classes. In particular, the ways in which math functions can have
    an optional output array get very annoying to deal with, as does
    special-casing to deal with zero-dimensional arrays, which you can't
    index.

    Subclasses of this class can implement “tidy” versions of the standard
    functions that can assume the following things:

    - The output array is always provided.
    - The arguments and output have been converted to a common type.
    - The arguments and output all have the same shape.
    - The arguments and output are at least one-dimensional.

    The “tidy” version of each function should be implemented in a function
    whose name has been prefixed with ``tidy_``; e.g., ``multiply`` is
    implemented in ``tidy_multiply``. You can override the standard tidying
    effects by simply implementing a function with the un-prefixed name.

    """

    def _tidy_std_unary (self, opname, x, out=None, **kwargs):
        """The semantics of Numpy's unary ufuncs are more complicated than you might
        think. If *out* is provided, *x* and *out* must have
        broadcast-compatible shapes, but *out* must be the "larger" array in a
        broadcast-y sense. And, it is important to support the case that ``x
        is out``.

        Numpy distinguishes between "scalars" and "zero-dimensional" arrays,
        although they are very similar in many aspects. We simplify things by
        removing this distinction.

        However, we then complicate things by ensuring that the implementation
        only sees atleast-1D arrays and that the arrays have identical shapes.

        """
        x = self.typeconvert (x)
        xsh = self.shape (x)

        if out is None:
            zerod = (xsh == ())
            if zerod:
                x = self.reshape (x, (1,))

            flags = common_interface_functions[opname].flags
            if flags & Flags.bool_result:
                out_dtype = np.bool
            else:
                out_dtype = self.get_dtype (x)

            retval = out = self.new_empty (xsh, out_dtype)
            if zerod:
                retval = self.reshape (out, ())
        else:
            retval = out
            out = self.typeconvert (out)
            osh = self.shape (out)
            bsh = broadcast_shapes (xsh, osh)
            if osh != bsh:
                raise ValueError ('output parameter must have final broadcasted shape')

            zerod = (bsh == ()) # can only happen if x and out are both zero-D.

            if zerod:
                x = self.reshape (x, (1,))
                out = self.reshape (out, (1,))
            elif xsh != bsh:
                x = self.broadcast_to (x, bsh)

        getattr (self, 'tidy_' + opname) (x, out, **kwargs)
        return retval


    def _tidy_std_binary (self, opname, x, y, out=None, **kwargs):
        x = self.typeconvert (x)
        xsh = self.shape (x)
        y = self.typeconvert (y)
        ysh = self.shape (y)

        if out is None:
            bsh = broadcast_shapes (xsh, ysh)
            zerod = (bsh == ())
            if zerod:
                x = self.reshape (x, (1,))
                y = self.reshape (y, (1,))
            else:
                if xsh != bsh:
                    x = self.broadcast_to (x, bsh)
                if ysh != bsh:
                    y = self.broadcast_to (y, bsh)

            flags = common_interface_functions[opname].flags
            if flags & Flags.bool_result:
                out_dtype = np.bool
            else:
                out_dtype = np.result_type (self.get_dtype (x), self.get_dtype (y))

            retval = out = self.new_empty (bsh, out_dtype)
            if zerod:
                retval = self.reshape (out, ())
        else:
            retval = out
            out = self.typeconvert (out)
            osh = self.shape (out)
            bsh = broadcast_shapes (xsh, ysh, osh)
            if osh != bsh:
                raise ValueError ('output parameter must have final broadcasted shape')

            zerod = (bsh == ())

            if zerod:
                x = self.reshape (x, (1,))
                y = self.reshape (y, (1,))
                out = self.reshape (out, (1,))
            else:
                if xsh != bsh:
                    x = self.broadcast_to (x, bsh)
                if ysh != bsh:
                    y = self.broadcast_to (y, bsh)

        getattr (self, 'tidy_' + opname) (x, y, out, **kwargs)
        return retval

    # In principle we should also overload the implementations of any ufuncs
    # with non-standard signatures for which "tidying" makes sense ... but at
    # the moment I don't see any that fall into that description.


def _fill_tidied_library_type ():
    ni = TidiedFunctionLibrary._not_implemented

    for fs in six.viewvalues (common_interface_functions):
        if fs.signature == Signatures.std_unary:
            wrapper = partialmethod (TidiedFunctionLibrary._tidy_std_unary, fs.name)
        elif fs.signature == Signatures.std_binary:
            wrapper = partialmethod (TidiedFunctionLibrary._tidy_std_binary, fs.name)
        else:
            continue # some function like `reshape` that doesn't make sense to tidy.

        setattr (TidiedFunctionLibrary, fs.name, wrapper)
        setattr (TidiedFunctionLibrary, 'tidy_' + fs.name, partialmethod (ni, 'tidy_' + fs.name))

_fill_tidied_library_type ()




# Some sugar: the MathlibDelegatingObject class so you can delegate Python
# math operators to the dispatch mechanism.

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
        return common_interface_functions[name]._impl (self, other)

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
        return common_interface_functions['power']._impl (self, other)

    __lshift__ = partialmethod (__dispatch_binary, 'left_shift')
    __rshift__ = partialmethod (__dispatch_binary, 'right_shift')
    __and__ = partialmethod (__dispatch_binary, 'bitwise_and')
    __xor__ = partialmethod (__dispatch_binary, 'bitwise_xor')
    __or__ = partialmethod (__dispatch_binary, 'bitwise_or')
    __div__ = partialmethod (__dispatch_binary, 'divide')
    __truediv__ = partialmethod (__dispatch_binary, 'true_divide')

    def __dispatch_binary_reflected (self, name, other):
        return common_interface_functions[name]._impl (other, self)

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
        return common_interface_functions['power']._impl (other, self)

    __rlshift__ = partialmethod (__dispatch_binary_reflected, 'left_shift')
    __rrshift__ = partialmethod (__dispatch_binary_reflected, 'right_shift')
    __rand__ = partialmethod (__dispatch_binary_reflected, 'bitwise_and')
    __rxor__ = partialmethod (__dispatch_binary_reflected, 'bitwise_xor')
    __ror__ = partialmethod (__dispatch_binary_reflected, 'bitwise_or')

    def __dispatch_binary_inplace (self, name, other):
        common_interface_functions[name]._impl (self, other, self)
        return self

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
        common_interface_functions['power']._impl (self, other, self)
        return self

    __ilshift__ = partialmethod (__dispatch_binary_inplace, 'left_shift')
    __irshift__ = partialmethod (__dispatch_binary_inplace, 'right_shift')
    __iand__ = partialmethod (__dispatch_binary_inplace, 'bitwise_and')
    __ixor__ = partialmethod (__dispatch_binary_inplace, 'bitwise_xor')
    __ior__ = partialmethod (__dispatch_binary_inplace, 'bitwise_or')

    def __neg__ (self):
        return common_interface_functions['negative']._impl (self)

    #def __pos__ (self):
    #    raise NotImplementedError

    def __abs__ (self):
        return common_interface_functions['absolute']._impl (self)

    def __invert__ (self):
        return common_interface_functions['invert']._impl (self)




# Simple object for testing TidiedFunctionLibrary

class TestWrappedLibrary (TidiedFunctionLibrary):
    def accepts (self, other):
        return isinstance (other, numpy_types + (TestArrayWrapper,))

    def new_empty (self, shape, dtype):
        return TestArrayWrapper (np.empty (shape, dtype=dtype))

    def typeconvert (self, thing):
        if isinstance (thing, TestArrayWrapper):
            return thing
        return TestArrayWrapper (try_asarray (thing))

    def broadcast_to (self, x, shape):
        return TestArrayWrapper (np.broadcast_to (x.data, shape))

    def tidy_negative (self, x, out):
        print ('TN: <%s %s %s>, <%s %s %s>' % (
            x.__class__.__name__, self.shape (x), self.get_dtype (x),
            out.__class__.__name__, self.shape (out), self.get_dtype (out))
        )
        out.data[:] = -x.data

    def tidy_add (self, x, y, out):
        print ('TA: <%s %s %s>, <%s %s %s>, <%s %s %s>' % (
            x.__class__.__name__, self.shape (x), self.get_dtype (x),
            y.__class__.__name__, self.shape (y), self.get_dtype (y),
            out.__class__.__name__, self.shape (out), self.get_dtype (out))
        )
        out.data[:] = x.data + y.data


class TestArrayWrapper (MathlibDelegatingObject):
    def __init__ (self, data):
        self.data = np.asarray (data)

    def __repr__ (self):
        return repr (self.data) + ' [wrapped]'

    @property
    def dtype (self):
        return self.data.dtype

    @property
    def shape (self):
        return self.data.shape

    _pk_mathlib_library_ = TestWrappedLibrary ()
