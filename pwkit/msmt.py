# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""Math with uncertain measurements.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
Domain

Kind

MeasurementABC
MeasurementFunctionLibrary
ScalarPowFunctionLibrary

sampled_n_samples
SampledDtypeGenerator
get_sampled_dtype
Sampled
sampled_function_library

ApproximateDtypeGenerator
get_approximate_dtype
Approximate
approximate_function_library
''').split ()

import abc, operator, six
from six.moves import range
import numpy as np

from . import PKError, unicode_to_str
from .simpleenum import enumeration
from .numutil import broadcastize, try_asarray
from .mathlib import MathFunctionLibrary, MathlibDelegatingObject, AlwaysOutFunctionLibrary, numpy_types




@enumeration
class Domain (object):
    """An enumeration of possible domains that measurements may span.

    """
    anything = 0
    nonnegative = 1
    nonpositive = 2

    names = ['anything', 'nonnegative', 'nonpositive']

    @classmethod # for some reason @staticmethod isn't working?
    def normalize (cls, domain):
        """Return a verified, normalized domain. In particular we allow the string
        representations to be used, which can be convenient for code calling
        us from outside this module.

        """
        try:
            dval = Domain.names.index (domain)
        except ValueError:
            try:
                dval = int (domain)
            except (ValueError, TypeError):
                raise ValueError ('illegal measurement domain %r' % domain)

        if not (dval >= cls.anything and dval <= cls.nonpositive):
            raise ValueError ('illegal measurement domain %r' % domain)
        return dval

    negate = [anything, nonpositive, nonnegative]
    add = {
        (anything, anything): anything,
        (anything, nonnegative): anything,
        (anything, nonpositive): anything,
        (nonnegative, nonnegative): nonnegative,
        (nonnegative, nonpositive): anything,
        (nonpositive, nonpositive): nonpositive,
    }
    union = add
    mul = {
        (anything, anything): anything,
        (anything, nonnegative): anything,
        (anything, nonpositive): anything,
        (nonnegative, nonnegative): nonnegative,
        (nonnegative, nonpositive): nonpositive,
        (nonpositive, nonpositive): nonnegative,
    }


def _ordpair (v1, v2):
    """For use with Domain data tables."""
    if v1 > v2:
        return (v2, v1)
    return (v1, v2)


def _all_in_domain (data, domain):
    if domain == Domain.anything:
        return True
    if domain == Domain.nonnegative:
        return np.all (data >= 0)
    if domain == Domain.nonpositive:
        return np.all (data <= 0)




@enumeration
class Kind (object):
    undef = 0
    msmt = 1
    upper = 2
    lower = 3

    names = ['undef', 'msmt', 'upper', 'lower']

    @classmethod
    def check (cls, kinds):
        if not np.all ((kinds >= Kind.undef) & (kinds <= Kind.lower)):
            raise ValueError ('illegal measurement kind(s) %r' % kinds)
        return kinds

    negate = np.array ([undef, msmt, lower, upper])
    reciprocal = negate

    # For binary operations, we need to be able to vectorize, and it would be
    # nice to be somewhat efficient about it. So, we need to make a bunch of
    # static arrays indexed by:
    #
    #    0 - undef * undef
    #    1 - undef * msmt
    #    2 - undef * upper
    #    3 - undef * lower
    #
    #    4 - msmt  * undef  (redundant with 1)
    #    5 - msmt  * msmt
    #    6 - msmt  * upper
    #    7 - msmt  * lower
    #
    #    8 - upper * undef  (redundant with 2)
    #    9 - upper * msmt   (redundant with 6)
    #   10 - upper * upper
    #   11 - upper * lower
    #
    #   12 - lower * undef  (redundant with 3)
    #   13 - lower * msmt   (redundant with 7)
    #   14 - lower * upper  (redundant with 11)
    #   15 - lower * lower

    @classmethod
    def binop (cls, k1, k2):
        return k1 * 4 + k2

    add = np.array ([
        undef, undef, undef, undef, # undef + ...
        undef, msmt,  upper, lower, # msmt  + ...
        undef, upper, upper, undef, # upper + ...
        undef, lower, undef, lower, # lower + ...
    ])

    posmul = np.array ([
        # Multiplication, assuming both values are positive.
        undef, undef, undef, undef, # undef * ...
        undef, msmt,  upper, lower, # msmt  * ...
        undef, upper, upper, undef, # upper * ...
        undef, lower, undef, lower, # lower * ...
    ])




class MeasurementABC (six.with_metaclass (abc.ABCMeta, MathlibDelegatingObject)):
    """A an array of measurements that may be uncertain or limits.

    """
    __slots__ = ('domain', 'data', '_scalar', '_pk_mathlib_library_')

    @classmethod
    def from_other (cls, v, copy=True, domain=None):
        raise NotImplementedError ()

    @classmethod
    def _from_data (cls, domain, data):
        """This default implementation assumes that __init__ has a signature
        compatible with: __init__ (domain, None, _noalloc=False)

        """
        rv = cls (domain, None, _noalloc=True)
        rv.data = data
        rv._scalar = (rv.data.shape == ())
        if rv._scalar:
            rv.data = np.atleast_1d (rv.data)
        return rv

    def copy (self):
        return self.__class__._from_data (self.domain, self.data.copy ())


    # Basic array properties

    @property
    def shape (self):
        if self._scalar:
            return ()
        return self.data.shape

    @property
    def ndim (self):
        if self._scalar:
            return 1
        return self.data.ndim

    @property
    def size (self):
        if self._scalar:
            return 1
        return self.data.size

    def __len__ (self):
        if self._scalar:
            raise TypeError ('len() of unsized object')
        return self.data.shape[0]


    # Comparisons. We are conservative with the built-in operators.

    def __lt__ (self, other):
        raise TypeError ('Measurements do not have a well-defined "<" comparison')

    def __le__ (self, other):
        raise TypeError ('Measurements do not have a well-defined "<=" comparison')

    def __gt__ (self, other):
        raise TypeError ('Measurements do not have a well-defined ">" comparison')

    def __ge__ (self, other):
        raise TypeError ('Measurements do not have a well-defined ">=" comparison')

    def __eq__ (self, other):
        raise TypeError ('Measurements do not have a well-defined "==" comparison')

    def __ne__ (self, other):
        raise TypeError ('Measurements do not have a well-defined "!=" comparison')


    # Indexing

    def __getitem__ (self, index):
        if self._scalar:
            raise IndexError ('invalid index to scalar variable')
        return self.__class__._from_data (self.domain, self.data[index])

    def __setitem__ (self, index, value):
        if self._scalar:
            raise TypeError ('object does not support item assignment')
        value = self.__class__.from_other (value, copy=False)
        self.domain = Domain.union[_ordpair (self.domain, value.domain)]
        self.data[index] = value.data


    # Stringification

    @staticmethod
    def _str_one (datum):
        raise NotImplementedError ()


    def __unicode__ (self):
        if self._scalar:
            datum = self._str_one (self.data[0])
            return datum + ' (%s %s scalar)' % (Domain.names[self.domain],
                                                self.__class__.__name__)
        else:
            text = np.array2string (self.data, formatter={'all': self._str_one})
            return text + ' (%s %s %r-array)' % (Domain.names[self.domain],
                                                 self.__class__.__name__,
                                                 self.shape)

    __str__ = unicode_to_str

    def __repr__ (self):
        return str (self)


    @classmethod
    def parse (cls, text, domain=Domain.anything):
        raise NotImplementedError ()


class MeasurementFunctionLibrary (MathFunctionLibrary):
    """This function library implements as much basic algebra as it can
    in terms of the following fundamental operations:

    - add
    - exp
    - log
    - multiply
    - negative
    - reciprocal

    This class must be wrapped in :class:`ScalarPowFunctionLibrary` so that
    the `out` parameter always exists when these functions get called, plus
    ther `power` operation can be defined specially.

    """
    def empty_like_broadcasted (self, x, y=None):
        """Create a new empty data structure matching the shape of *x* and *y*
        broadcasted together; *y* is optional, in which case the shape should
        just be that of *x*.

        """
        raise NotImplementedError ()

    def fill_from_broadcasted (self, x, out):
        """Fill *out* with the values from *x*, broadcasting as appropriate.

        """
        raise NotImplementedError ()

    def fill_unity_like (self, x, out):
        """Fill *out* with ones, matching the (broadcasted) masking state of *x* if
        applicable.

        """
        raise NotImplementedError ()

    def coerce_one (self, x):
        """Convert an arbitrary numerical-ish object into the right type.

        """
        raise NotImplementedError ()

    def coerce (self, x, y=None, out=None):
        return self.coerce_one (x), self.coerce_one (y), self.coerce_one (out)

    def expm1 (self, x, out):
        """The whole point of this function is that non-naive implementations can
        achieve higher precision, so this naive implementation is not going to be
        that useful.

        """
        self.exp (x, out)
        self.subtract (out, 1, out)
        return out

    def log10 (self, x, out):
        self.log (x, out)
        self.true_divide (out, self.coerce_one (np.log (10)), out)
        return out

    def log1p (self, x, out):
        """The whole point of this function is that non-naive implementations can
        achieve higher precision, so this naive implementation is not going to be
        that useful.

        """
        self.add (x, 1, out)
        self.log (out, out)
        return out

    def log2 (self, x, out):
        self.log (x, out)
        self.true_divide (out, self.coerce_one (np.log (2)), out)
        return out

    def sqrt (self, x, out):
        self.power (x, 0.5, out)
        return out

    def square (self, x, out):
        self.power (x, 2, out)
        return out

    def subtract (self, x, y, out):
        # We have to allocate a temporary in case `x is out`. Or you know we could
        # just implement the function.
        temp = self.empty_like_broadcasted (y)
        self.negative (y, temp)
        self.add (x, temp, out)
        return out

    def true_divide (self, x, y, out):
        # We have to allocate a temporary in case `x is out`. Or you know we could
        # just implement the function.
        temp = self.empty_like_broadcasted (y)
        self.reciprocal (y, temp)
        self.multiply (x, temp, out)
        return out


class ScalarPowFunctionLibrary (AlwaysOutFunctionLibrary):
    def power (self, x, y, out):
        try:
            v = float (x)
            # If we didn't just raise an exception, this is the __rpow__ case.
            # You might not think that this is common functionality, but it's
            # important for undoing logarithms; i.e. 10**x.
            if out is None:
                out = self.sub_library.empty_like_broadcasted (y)
            self.multiply (y, np.log (v), out)
            self.exp (out, out)
            return out
        except TypeError:
            pass

        try:
            v = float (y)
        except TypeError:
            raise ValueError ('measurements can only be exponentiated by exact values')

        if out is None:
            out = self.sub_library.empty_like_broadcasted (x)

        if v == 0:
            self.sub_library.fill_unity_like (x, out)
            return out

        reciprocate = (v < 0)
        if reciprocate:
            v = -v

        i = int (v)

        if v != i:
            # A fractional exponent.
            self.log (x, out)
            self.multiply (out, v, out)
            self.exp (out, out)
        else:
            # An integer exponent: implement as a series of multiplies, so
            # that we can work with negative values (which the log/exp
            # approach can't). Not the most efficient, but reduces the chance
            # for bugs.
            self.sub_library.fill_from_broadcasted (x, out)
            for _ in range (i - 1):
                self.multiply (out, x, out)

        if reciprocate:
            self.reciprocal (out, out)
        return out




# A Sampled measurement is one in which we propagate uncertainties the only
# fully tractable way -- by approximating each measurement with a large number
# of samples that are then processed vectorially. It's absurdly
# memory-inefficient, but works, unlike analytic error propagation, which
# fails in many real applications.

sampled_n_samples = 1024 - 1 # each Sampled takes 1024*itemsize bytes

class SampledDtypeGenerator (object):
    def __init__ (self):
        self.cache = {}

    def __call__ (self, sample_dtype):
        prev = self.cache.get (sample_dtype)
        if prev is not None:
            return prev

        # The `str` calls are needed for Python 2.x since unicode_literals is
        # activated.
        npad = np.dtype (sample_dtype).itemsize - 1
        dtype = np.dtype ([
            (str ('kind'), np.uint8, 1),
            (str ('_pad'), np.uint8, npad),
            (str ('samples'), sample_dtype, sampled_n_samples),
        ])
        self.cache[sample_dtype] = dtype
        return dtype

get_sampled_dtype = SampledDtypeGenerator ()


class Sampled (MeasurementABC):
    """An empirical uncertain value, represented by samples.

    """
    def __init__ (self, domain, shape, sample_dtype=np.double, _noalloc=False, _noinit=False):
        self._pk_mathlib_library_ = sampled_function_library
        self.domain = Domain.normalize (domain)

        if _noalloc:
            return # caller's responsibility to set `data` and `_scalar`

        # You can't index scalar values which makes it really annoying to
        # implement a lot of our math. So we store scalars as shape (1,) and
        # fake the outer parts.

        self.data = np.empty (shape, dtype=get_sampled_dtype (sample_dtype))
        self._scalar = (self.data.shape == ())

        if self._scalar:
            self.data = np.atleast_1d (self.data)

        if _noinit:
            return # caller's responsibility to fill `data`

        self.data['kind'].fill (Kind.undef)
        self.data['samples'].fill (np.nan)


    @classmethod
    def from_other (cls, v, copy=True, domain=None):
        if isinstance (v, cls):
            if copy:
                return v.copy ()
            return v

        # TODO: handle other value types. If we're still here, we expect some
        # numerical array-like floats. Domain can only get less restrictive
        # when we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        a = try_asarray (v)
        if a is None:
            raise ValueError ('do not know how to handle operand %r' % v)

        if domain is None:
            if np.all (a >= 0):
                domain = Domain.nonnegative
            elif np.all (a <= 0):
                domain = Domain.nonpositive
            else:
                domain = Domain.anything

        return cls.from_exact_array (domain, Kind.msmt, a)

    @classmethod
    def from_exact_array (cls, domain, kind, v):
        domain = Domain.normalize (domain)
        Kind.check (kind)
        if not _all_in_domain (v, domain):
            raise ValueError ('illegal Sampled initializer: data %r do not lie in '
                              'stated domain' % v)

        v = np.asarray (v)
        r = cls (domain, v.shape)
        r.data['kind'] = kind
        r.data['samples'] = v[...,None]
        return r

    @classmethod
    def from_norm (cls, mean, std, shape=(), domain=Domain.anything):
        domain = Domain.normalize (domain)
        if std < 0:
            raise ValueError ('std must be nonnegative')

        r = cls (domain, shape)
        r.data['kind'].fill (Kind.msmt)
        r.data['samples'] = np.random.normal (mean, std, shape+(sampled_n_samples,))
        return r


    # (Additional) basic array properties

    @property
    def sample_dtype (self):
        return self.data.dtype['samples'].base


    # Stringification

    @staticmethod
    def _str_one (datum):
        k = datum['kind']

        # TODO: better control of precision

        if k == Kind.undef:
            return '?'
        elif k == Kind.lower:
            return '>%.4f' % datum['samples'][0]
        elif k == Kind.upper:
            return '<%.4f' % datum['samples'][0]

        # XXX temporary
        return '%.3fpm%.3f' % (datum['samples'].mean (), datum['samples'].std ())


    @classmethod
    def parse (cls, text, domain=Domain.anything):
        raise NotImplementedError () # TODO


class SampledFunctionLibrary (MeasurementFunctionLibrary):
    def accepts (self, opname, other):
        return isinstance (other, numpy_types + (Sampled,))


    def coerce_one (self, x):
        if x is None:
            return None

        if isinstance (x, Sampled):
            return x

        # TODO: handle other value types. If we're still here, we expect some
        # numerical array-like floats. Domain can only get less restrictive
        # when we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        a = try_asarray (x)
        if a is None:
            raise ValueError ('do not know how to handle operand %r' % x)

        if np.all (a >= 0):
            domain = Domain.nonnegative
        elif np.all (a <= 0):
            domain = Domain.nonpositive
        else:
            domain = Domain.anything

        return Sampled.from_exact_array (domain, Kind.msmt, a)


    def empty_like_broadcasted (self, x, y=None):
        if y is None:
            shape = x.shape
            sdtype = x.sample_dtype
        else:
            shape = np.broadcast (x, y).shape
            sdtype = np.result_type (x.sample_dtype, y.sample_dtype)

        return Sampled (Domain.anything, shape, sample_dtype=sdtype, _noinit=True)


    def fill_from_broadcasted (self, x, out):
        out.domain = x.domain
        out.data['kind'] = x.data['kind']
        out.data['samples'] = x.data['samples']


    def fill_unity_like (self, x, out):
        out.domain = Domain.nonnegative
        m = (x.data['kind'] != Kind.undef)
        out.data['kind'][m] = Kind.msmt
        out.data['samples'][m] = 1
        m = ~m
        out.data['kind'][m] = Kind.undef
        out.data['samples'][m] = np.nan


    # The actual core math functions

    def add (self, x, y, out):
        out.domain = Domain.add[_ordpair (x.domain, y.domain)]
        out.data['kind'] = Kind.add[Kind.binop (x.data['kind'], y.data['kind'])]
        np.add (x.data['samples'], y.data['samples'], out.data['samples'])
        return out


    def exp (self, x, out):
        out.domain = Domain.nonnegative
        out.data['kind'] = x.data['kind']
        np.exp (x.data['samples'], out.data['samples'])
        return out


    def log (self, x, out):
        # Remember that we may have `x is kind`!
        out.data['kind'] = x.data['kind']
        np.log (x.data['samples'], out.data['samples'])

        if out.domain != Domain.nonnegative:
            undef = np.any (~np.isfinite (out.data['samples']), axis=-1) | (out.data['kind'] == Kind.upper)
            out.data['kind'][undef] = Kind.undef
            out.data['samples'][undef,:] = np.nan

        out.domain = Domain.anything
        return out


    def multiply (self, x, y, out):
        out.domain = Domain.mul[_ordpair (x.domain, y.domain)]

        # There's probably a smarter way to make sure that we get the limit
        # directions correct, but this ought to at least work.

        xkind = x.data['kind'].copy ()
        xnegate = ((xkind == Kind.lower) | (xkind == Kind.upper)) & (x.data['samples'][...,0] < 0)
        xkind[xnegate] = Kind.negate[xkind[xnegate]]

        ykind = y.data['kind'].copy ()
        ynegate = ((ykind == Kind.lower) | (ykind == Kind.upper)) & (y.data['samples'][...,0] < 0)
        ykind[ynegate] = Kind.negate[ykind[ynegate]]

        out.data['kind'] = Kind.posmul[Kind.binop (xkind, ykind)]
        nnegate = xnegate ^ ynegate
        out.data['kind'][nnegate] = Kind.negate[out.data['kind'][nnegate]]

        np.multiply (x.data['samples'], y.data['samples'], out.data['samples'])
        return out


    def negative (self, x, out):
        out.domain = Domain.negate[x.domain]
        out.data['kind'] = Kind.negate[x.data['kind']]
        np.negative (x.data['samples'], out.data['samples'])
        return out


    def reciprocal (self, x, out):
        out.domain = x.domain
        out.data['kind'] = Kind.reciprocal[x.data['kind']]
        # np.reciprocal() truncates integers, which we don't want
        np.divide (1, x.data['samples'], out.data['samples'])
        return out


sampled_function_library = ScalarPowFunctionLibrary (SampledFunctionLibrary ())




# "Approximate" measurements. These do not have the absurd memory requirements
# of Sampled, but their uncertainty assessment is only ... approximate. We do
# simpleminded error propagation as a courtesy; it very easily confused and
# should not be used for careful work.

class ApproximateDtypeGenerator (object):
    def __init__ (self):
        self.cache = {}

    def __call__ (self, sample_dtype):
        prev = self.cache.get (sample_dtype)
        if prev is not None:
            return prev

        # The `str` calls are needed for Python 2.x since unicode_literals is
        # activated.
        npad = np.dtype (sample_dtype).itemsize - 1
        dtype = np.dtype ([
            (str ('kind'), np.uint8, 1),
            (str ('_pad'), np.uint8, npad),
            (str ('x'), sample_dtype, 1),
            (str ('u'), sample_dtype, 1),
        ])
        self.cache[sample_dtype] = dtype
        return dtype

get_approximate_dtype = ApproximateDtypeGenerator ()


class Approximate (MeasurementABC):
    """An approximate uncertain value, represented with a scalar uncertainty parameter.

    """
    def __init__ (self, domain, shape, sample_dtype=np.double, _noalloc=False, _noinit=False):
        self._pk_mathlib_library_ = approximate_function_library
        self.domain = Domain.normalize (domain)

        if _noalloc:
            return # caller's responsibility to set `data` and `_scalar`

        # You can't index scalar values which makes it really annoying to
        # implement a lot of our math. So we store scalars as shape (1,) and
        # fake the outer parts.

        self.data = np.empty (shape, dtype=get_approximate_dtype (sample_dtype))
        self._scalar = (self.data.shape == ())

        if self._scalar:
            self.data = np.atleast_1d (self.data)

        if _noinit:
            return # caller's responsibility to fill `data`

        self.data['kind'].fill (Kind.undef)
        self.data['x'].fill (np.nan)
        self.data['u'].fill (np.nan)


    @classmethod
    def from_other (cls, v, copy=True, domain=None):
        if isinstance (v, cls):
            if copy:
                return v.copy ()
            return v

        # TODO: handle other value types. If we're still here, we expect some
        # numerical array-like floats. Domain can only get less restrictive
        # when we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        a = try_asarray (v)
        if a is None:
            raise ValueError ('do not know how to handle operand %r' % v)

        if domain is None:
            if np.all (a >= 0):
                domain = Domain.nonnegative
            elif np.all (a <= 0):
                domain = Domain.nonpositive
            else:
                domain = Domain.anything

        return cls.from_arrays (domain, Kind.msmt, a, 0)

    @classmethod
    def from_arrays (cls, domain, kind, x, u):
        domain = Domain.normalize (domain)
        Kind.check (kind)
        if not _all_in_domain (x, domain):
            raise ValueError ('illegal Approximate x initializer: data %r do not lie in '
                              'stated domain' % x)

        x = np.asarray (x)
        r = cls (domain, x.shape)
        r.data['kind'] = kind
        r.data['x'] = x
        r.data['u'] = u

        if not _all_in_domain (r.data['u'], Domain.nonnegative):
            raise ValueError ('illegal Approximate u initializer: some values %r '
                              'are negative' % u)

        return r


    # (Additional) basic array properties

    @property
    def sample_dtype (self):
        return self.data.dtype['x']


    # Comparison helpers

    def __lt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<" comparison; use lt() with caution')

    def __le__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<=" comparison; use le() with caution')

    def __gt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">" comparison; use gt() with caution')

    def __ge__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">=" comparison; use ge() with caution')


    def lt (self, other):
        """Returns true where the values in *self* are less than *other*, following this
        definition:

        - The result for *undef* values is always false
        - The result for *msmt* values is the result of comparing the "representative" value.
        - The result for limits is true if the limit is contained entirely within
          the proposed interval.

        *other* must be a scalar.

        TODO: arguments adjusting the behavior.

        """
        rv = (((self.data['kind'] == Kind.msmt) | (self.data['kind'] == Kind.upper))
              & (self.data['x'] < other))

        if other > 0 and self.domain == Domain.nonpositive:
            # This is the only way that a lower limit can result in True in
            # the strict definition.
            rv[self.data['kind'] == Kind.lower] = True

        if self._scalar:
            rv = rv[0]
        return rv


    def gt (self, other):
        """Note that we cannot simply invert `lt` since we have to handle undefs
        properly.

        """
        rv = (((self.data['kind'] == Kind.msmt) | (self.data['kind'] == Kind.lower))
              & (self.data['x'] > other))

        if other < 0 and self.domain == Domain.nonnegative:
            rv[self.data['kind'] == Kind.upper] = True

        if self._scalar:
            rv = rv[0]
        return rv


    # Stringification

    @staticmethod
    def _str_one (datum):
        k = datum['kind']

        # TODO: better control of precision

        if k == Kind.undef:
            return '?'
        elif k == Kind.lower:
            return '>%.4f' % datum['x']
        elif k == Kind.upper:
            return '<%.4f' % datum['x']

        return '%.3fpm%.3f' % (datum['x'], datum['u'])


    @classmethod
    def parse (cls, text, domain=Domain.anything):
        """This only handles scalar values. Accepted formats are:

        "?"
        "<{float}"
        ">{float}"
        "{float}"
        "{float}pm{float}"

        where {float} stands for a floating-point number.
        """
        domain = Domain.normalize (domain)
        rv = cls (domain, ())

        if text[0] == '<':
            rv.data['kind'] = Kind.upper
            rv.data['x'] = float (text[1:])
            rv.data['u'] = 0
        elif text[0] == '>':
            rv.data['kind'] = Kind.lower
            rv.data['x'] = float (text[1:])
            rv.data['u'] = 0
        elif text == '?':
            rv.data['kind'] = Kind.undef
            rv.data['x'] = np.nan
            rv.data['u'] = np.nan
        else:
            rv.data['kind'] = Kind.msmt
            pieces = text.split ('pm', 1)
            rv.data['x'] = float (pieces[0])

            if len (pieces) == 1:
                rv.data['u'] = 0
            else:
                rv.data['u'] = float (pieces[1])

        if rv.data['kind'] != Kind.undef:
            if not _all_in_domain (rv.data['x'], rv.domain):
                raise ValueError ('value of %s is not in stated domain %s' %
                                  (text, Domain.names[domain]))
            if rv.data['u'] < 0:
                raise ValueError ('uncertainty of %s is negative' % text)

        return rv


class ApproximateFunctionLibrary (MeasurementFunctionLibrary):
    def accepts (self, opname, other):
        return isinstance (other, numpy_types + (Approximate,))


    def coerce_one (self, x):
        if x is None:
            return None

        if isinstance (x, Approximate):
            return x

        # TODO: handle other value types. If we're still here, we expect some
        # numerical array-like floats. Domain can only get less restrictive
        # when we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        a = try_asarray (x)
        if a is None:
            raise ValueError ('do not know how to handle operand %r' % x)

        if np.all (a >= 0):
            domain = Domain.nonnegative
        elif np.all (a <= 0):
            domain = Domain.nonpositive
        else:
            domain = Domain.anything

        return Approximate.from_other (a, domain=domain)


    def empty_like_broadcasted (self, x, y=None):
        if y is None:
            shape = x.shape
            sdtype = x.sample_dtype
        else:
            shape = np.broadcast (x, y).shape
            sdtype = np.result_type (x.sample_dtype, y.sample_dtype)

        return Approximate (Domain.anything, shape, sample_dtype=sdtype, _noinit=True)


    def fill_from_broadcasted (self, x, out):
        out.domain = x.domain
        out.data['kind'] = x.data['kind']
        out.data['x'] = x.data['x']
        out.data['u'] = x.data['u']


    def fill_unity_like (self, x, out):
        out.domain = Domain.nonnegative
        m = (x.data['kind'] != Kind.undef)
        out.data['kind'][m] = Kind.msmt
        out.data['x'][m] = 1
        out.data['u'][m] = 0
        m = ~m
        out.data['kind'][m] = Kind.undef
        out.data['x'][m] = np.nan
        out.data['u'][m] = np.nan


    # The actual core math functions

    def add (self, x, y, out):
        out.domain = Domain.add[_ordpair (x.domain, y.domain)]
        out.data['kind'] = Kind.add[Kind.binop (x.data['kind'], y.data['kind'])]
        np.add (x.data['x'], y.data['x'], out.data['x'])
        np.sqrt (x.data['u']**2 + y.data['x']**2, out.data['u'])
        return out


    def exp (self, x, out):
        out.domain = Domain.nonnegative
        out.data['kind'] = x.data['kind']
        np.multiply (x.data['u'], x.data['x'], out.data['u'])
        np.exp (x.data['x'], out.data['x'])
        return out


    def log (self, x, out):
        # Remember that we may have `x is kind`!
        out.data['kind'] = x.data['kind']

        if x.domain == Domain.nonnegative:
            np.divide (x.data['u'], x.data['x'], out.data['u'])
            np.log (x.data['x'], out.data['x'])
        else:
            m = (x.data['x'] <= 0) | (x.data['kind'] == Kind.upper) | (x.data['kind'] == Kind.undef)
            out.data['kind'][m] = Kind.undef
            out.data['x'][m] = np.nan
            out.data['u'][m] = np.nan
            m = ~m
            # can't use np.divide(), etc., since we're fancy-indexing:
            out.data['u'][m] = x.data['u'][m] / x.data['x'][m]
            out.data['x'][m] = np.log (x.data['x'][m])

        out.domain = Domain.anything
        return out


    def multiply (self, x, y, out):
        out.domain = Domain.mul[_ordpair (x.domain, y.domain)]

        # There's probably a smarter way to make sure that we get the limit
        # directions correct, but this ought to at least work.

        xkind = x.data['kind'].copy ()
        xnegate = (x.data['x'] < 0)
        xkind[xnegate] = Kind.negate[xkind[xnegate]]

        ykind = y.data['kind'].copy ()
        ynegate = (y.data['x'] < 0)
        ykind[ynegate] = Kind.negate[ykind[ynegate]]

        out.data['kind'] = Kind.posmul[Kind.binop (xkind, ykind)]
        nnegate = xnegate ^ ynegate
        out.data['kind'][nnegate] = Kind.negate[out.data['kind'][nnegate]]

        np.sqrt ((x.data['u'] * y.data['x'])**2 + (y.data['u'] * x.data['x'])**2, out.data['u'])
        np.multiply (x.data['x'], y.data['x'], out.data['x'])
        return out


    def negative (self, x, out):
        out.domain = Domain.negate[x.domain]
        out.data['kind'] = Kind.negate[x.data['kind']]
        np.negative (x.data['x'], out.data['x'])
        out.data['u'] = x.data['u']
        return out


    def reciprocal (self, x, out):
        out.domain = x.domain
        out.data['kind'] = Kind.reciprocal[x.data['kind']]
        # np.reciprocal() truncates integers, which we don't want
        np.divide (1, x.data['x'], out.data['x'])
        np.multiply (x.data['u'], out.data['x']**2, out.data['u'])
        return out


approximate_function_library = ScalarPowFunctionLibrary (ApproximateFunctionLibrary ())
