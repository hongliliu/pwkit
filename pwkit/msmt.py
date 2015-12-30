# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""Math with uncertain measurements.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
Domain

Kind

sampled_n_samples
SampledDtypeGenerator
get_sampled_dtype
make_sampled_data
Sampled
sampled_unary_math

ApproximateDtypeGenerator
get_approximate_dtype
make_approximate_data
Approximate
approximate_unary_math

basic_unary_math
absolute
arccos
arcsin
arctan
cos
expm1
exp
isfinite
log10
log1p
log2
log
negative
reciprocal
sin
sqrt
square
tan
''').split ()

import operator, six
from six.moves import range
import numpy as np

from . import PKError, unicode_to_str
from .simpleenum import enumeration
from .numutil import broadcastize, try_asarray


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

def make_sampled_data (kinds, samples):
    data = np.empty (kinds.shape, dtype=get_sampled_dtype (samples.dtype))
    data['kind'] = kinds
    data['samples'] = samples
    return data


class Sampled (object):
    """An empirical uncertain value, represented by samples.

    """
    __slots__ = ('domain', 'data', '_scalar')

    def __init__ (self, domain, shape_or_data=None, sample_dtype=np.double):
        self.domain = Domain.normalize (domain)

        if isinstance (shape_or_data, (tuple,) + six.integer_types):
            self.data = np.empty (shape_or_data, dtype=get_sampled_dtype (sample_dtype))
            self.data['kind'].fill (Kind.undef)
        elif isinstance (shape_or_data, (np.ndarray, np.void)):
            # Scalars end up as the `np.void` type. It's hard to check the
            # array dtype thoroughly but let's do this:
            try:
                assert shape_or_data['kind'].dtype == np.uint8
            except Exception:
                raise ValueError ('illegal Sampled initializer array %r' % shape_or_data)
            self.data = shape_or_data
        else:
            raise ValueError ('unrecognized Sampled initializer %r' % shape_or_data)

    @staticmethod
    def from_other (v, copy=True, domain=None):
        if isinstance (v, Sampled):
            if copy:
                return v.copy ()
            return v

        # TODO: handle other value types. If we're here, we just have some
        # array of floats. Domain can only get less restrictive when we do
        # math on them, so it's OK to choose the best domain we can given the
        # data we have.

        v = np.asarray (v)
        if domain is None:
            if np.all (v >= 0):
                domain = Domain.nonnegative
            elif np.all (v <= 0):
                domain = Domain.nonpositive
            else:
                domain = Domain.anything

        return Sampled.from_exact_array (domain, Kind.msmt, v)

    @staticmethod
    def from_exact_array (domain, kind, v):
        domain = Domain.normalize (domain)
        Kind.check (kind)
        if not _all_in_domain (v, domain):
            raise ValueError ('illegal Sampled initializer: data %r do not lie in '
                              'stated domain' % v)

        v = np.asarray (v)
        r = Sampled (domain, v.shape)
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

    def copy (self):
        return self.__class__ (self.domain, self.data.copy ())

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

    @property
    def sample_dtype (self):
        return self.data.dtype['samples']

    def __len__ (self):
        if self._scalar:
            raise TypeError ('len() of unsized object')
        return self.data.shape[0]

    # Math.

    def _inplace_negate (self):
        self.domain = Domain.negate[self.domain]
        self.data['kind'] = Kind.negate[self.data['kind']]
        np.negative (self.data['samples'], self.data['samples'])
        return self


    def __neg__ (self):
        rv = self.copy ()
        rv._inplace_negate ()
        return rv

    # Comparisons. We are conservative with the build-in operators, but have
    # some options that are helpful.

    def __lt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<" comparison; use lt() with caution')

    def __le__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<=" comparison; use le() with caution')

    def __gt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">" comparison; use gt() with caution')

    def __ge__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">=" comparison; use ge() with caution')

    def __eq__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "==" comparison')

    def __ne__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "!=" comparison')


    # Indexing

    def __getitem__ (self, index):
        if self._scalar:
            raise IndexError ('invalid index to scalar variable')
        return Sampled (self.domain, self.data[index])

    def __setitem__ (self, index, value):
        if self._scalar:
            raise TypeError ('object does not support item assignment')
        value = Sampled.from_other (value, copy=False)
        self.domain = Domain.union[_ordpair (self.domain, value.domain)]
        self.data[index] = value.data


    # Stringification

    @staticmethod
    def _str_one (datum):
        raise NotImplementedError ()


    def __unicode__ (self):
        if self._scalar:
            datum = Sampled._str_one (self.data)
            return datum + ' <%s %s scalar>' % (Domain.names[self.domain],
                                                self.__class__.__name__)
        else:
            text = np.array2string (self.data, formatter={'all': Sampled._str_one})
            return text + ' <%s %s %r-array>' % (Domain.names[self.domain],
                                                 self.__class__.__name__,
                                                 self.shape)

    __str__ = unicode_to_str


    def __repr__ (self):
        return str (self)


    @staticmethod
    def parse (text, domain=Domain.anything):
        raise NotImplementedError ()


def _sampled_unary_negative (q):
    return q.copy ()._inplace_negate ()


sampled_unary_math = {
    'negative': _sampled_unary_negative,
}


# "Approximate" measurements. These do not have the absurd memory requirements
# of Sampled, but their uncertainty assessment is only ... approximate. We do
# simplemended error propagation as a courtesy; it very easily confused and
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


def make_approximate_data (kind, x, u):
    data = np.empty (kind.shape, dtype=get_approximate_dtype (x.dtype))
    data['kind'] = kind
    data['x'] = x
    data['u'] = u
    return data


class Approximate (object):
    """An approximate uncertain value, represented with a scalar uncertainty parameter.

    """
    __slots__ = ('domain', 'data', '_scalar')

    def __init__ (self, domain, shape_or_data=None, sample_dtype=np.double):
        self.domain = Domain.normalize (domain)

        if isinstance (shape_or_data, (tuple,) + six.integer_types):
            self.data = np.empty (shape_or_data, dtype=get_approximate_dtype (sample_dtype))
            self.data['kind'].fill (Kind.undef)
        elif isinstance (shape_or_data, (np.ndarray, np.void)):
            # Scalars end up as the `np.void` type. It's hard to check the
            # array dtype thoroughly but let's do this:
            try:
                assert shape_or_data['kind'].dtype == np.uint8
            except Exception:
                raise ValueError ('illegal Approximate initializer array %r' % shape_or_data)
            self.data = shape_or_data
        else:
            raise ValueError ('unrecognized Approximate initializer %r' % shape_or_data)

        # You can't index scalar values which makes it really annoying to
        # implement a lot of our math. So we store scalars as shape (1,) and
        # fake the outer parts.

        self._scalar = (self.data.shape == ())

        if self._scalar:
            self.data = np.atleast_1d (self.data)

    @staticmethod
    def from_other (v, copy=True, domain=None):
        if isinstance (v, Approximate):
            if copy:
                return v.copy ()
            return v

        # TODO: handle other value types. If we're here, we just have
        # some array of floats. Domain can only get less restrictive when
        # we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        v = np.asarray (v)
        if domain is None:
            if np.all (v >= 0):
                domain = Domain.nonnegative
            elif np.all (v <= 0):
                domain = Domain.nonpositive
            else:
                domain = Domain.anything

        return Approximate.from_arrays (domain, Kind.msmt, v, 0)

    @staticmethod
    def from_arrays (domain, kind, x, u):
        domain = Domain.normalize (domain)
        Kind.check (kind)
        if not _all_in_domain (x, domain):
            raise ValueError ('illegal Approximate x initializer: data %r do not lie in '
                              'stated domain' % x)

        x = np.asarray (x)
        r = Approximate (domain, x.shape)
        r.data['kind'] = kind
        r.data['x'] = x
        r.data['u'] = u

        if not _all_in_domain (r.data['u'], Domain.nonnegative):
            raise ValueError ('illegal Approximate u initializer: some values %r '
                              'are negative' % u)

        return r

    def copy (self):
        return self.__class__ (self.domain, self.data.copy ())


    # Basic array properties

    @property
    def shape (self):
        if self._scalar:
            return ()
        return self.data.shape

    @property
    def ndim (self):
        if self._scalar:
            return 0
        return self.data.ndim

    @property
    def size (self):
        if self._scalar:
            return 1
        return self.data.size

    @property
    def sample_dtype (self):
        return self.data.dtype['x']

    def __len__ (self):
        if self._scalar:
            raise TypeError ('len() of unsized object')
        return self.data.shape[0]

    # Math.

    def _inplace_negate (self):
        self.domain = Domain.negate[self.domain]
        self.data['kind'] = Kind.negate[self.data['kind']]
        np.negative (self.data['x'], self.data['x'])
        # uncertainty unchanged.
        return self


    def _inplace_abs (self):
        self.domain = Domain.nonnegative
        np.absolute (self.data['x'], self.data['x'])
        # uncertainty unchanged
        assert False, 'figure out what to do here'
        return self


    def _inplace_reciprocate (self):
        # domain unchanged
        self.data['kind'] = Kind.reciprocal[self.data['kind']]
        # np.reciprocal() truncates integers, which we don't want
        np.divide (1, self.data['x'], self.data['x'])
        np.multiply (self.data['u'], self.data['x']**2, self.data['u'])
        return self


    def _inplace_exp (self):
        self.domain = Domain.nonnegative
        # kind is unchanged
        np.exp (self.data['x'], self.data['x'])
        np.multiply (self.data['u'], self.data['x'], self.data['u'])
        return self


    def __neg__ (self):
        rv = self.copy ()
        rv._inplace_negate ()
        return rv

    def __abs__ (self):
        rv = self.copy ()
        rv._inplace_abs ()
        return rv


    def __iadd__ (self, other):
        other = Approximate.from_other (other, copy=False)
        self.domain = Domain.add[_ordpair (self.domain, other.domain)]
        self.data['kind'] = Kind.add[Kind.binop (self.data['kind'], other.data['kind'])]
        self.data['x'] += other.data['x']
        self.data['u'] = np.sqrt (self.data['u']**2 + other.data['u']**2)
        return self

    def __add__ (self, other):
        rv = self.copy ()
        rv += other
        return rv

    __radd__ = __add__

    def __sub__ (self, other):
        return self + (-other)

    def __isub__ (self, other):
        self += (-other)
        return self

    def __rsub__ (self, other):
        return (-self) + other


    def __imul__ (self, other):
        other = Approximate.from_other (other, copy=False)
        self.domain = Domain.mul[_ordpair (self.domain, other.domain)]

        # There's probably a simpler way to make sure that we get the limit directions
        # correct, but this ought to at least work.

        skind = self.data['kind'].copy ()
        snegate = (self.data['x'] < 0)
        skind[snegate] = Kind.negate[skind[snegate]]

        okind = other.data['kind'].copy ()
        onegate = (other.data['x'] < 0)
        okind[onegate] = Kind.negate[okind[onegate]]

        self.data['kind'] = Kind.posmul[Kind.binop (skind, okind)]
        nnegate = snegate ^ onegate
        self.data['kind'][nnegate] = Kind.negate[self.data['kind'][nnegate]]

        # Besides the kinds, this is straightforward:

        self.data['x'] *= other.data['x']
        self.data['u'] = np.sqrt ((self.data['u'] * other.data['x'])**2 +
                                  (other.data['u'] * self.data['x'])**2)
        return self

    def __mul__ (self, other):
        rv = self.copy ()
        rv *= other
        return rv

    __rmul__ = __mul__

    def __itruediv__ (self, other):
        self *= other**-1
        return self

    def __truediv__ (self, other):
        rv = self.copy ()
        rv /= other
        return rv

    def __rtruediv__ (self, other):
        rv = self.copy ()
        rv._inplace_reciprocate ()
        rv *= other
        return rv

    __div__ = __truediv__
    __idiv__ = __itruediv__
    __rdiv__ = __rtruediv__


    def __ipow__ (self, other, modulo=None):
        if modulo is not None:
            raise ValueError ('powmod behavior forbidden with Approximates')

        try:
            v = float (other)
        except TypeError:
            raise ValueError ('Approximates can only be exponentiated by exact values')

        if v == 0:
            self.domain = Domain.nonnegative
            defined = (self.data['kind'] != Kind.undef)
            self.data['kind'][defined] = Kind.msmt
            self.data['x'][defined] = 1
            self.data['u'][defined] = 0
            return self

        reciprocate = (v < 0)
        if reciprocate:
            v = -v

        i = int (v)

        if v != i:
            # For us, fractional powers are only defined on positive numbers,
            # which gives us a fairly small number of valid cases to worry
            # about.
            self.data['x'] **= v
            self.data['u'] *= v * np.abs (self.data['x'])**(v - 1)

            if self.domain != Domain.nonnegative:
                undef = (self.data['kind'] == Kind.upper) | (self.data['x'] < 0)
                self.data['kind'][undef] = Kind.undef
                self.data['x'][undef] = np.nan
                self.data['u'][undef] = 0.
        else:
            # We can deal with integer exponentiation as a series of
            # multiplies. Not the most efficient, but reduces the chance for
            # bugs. However, we have to correct the uncertainty since our naive
            # uncertainty approach doesn't take into account correlations.
            orig = self.copy ()
            for _ in range (i - 1):
                self *= orig
            self.data['u'] /= np.sqrt (i)

        if reciprocate:
            self._inplace_reciprocate ()
        return self

    def __pow__ (self, other, module=None):
        rv = self.copy ()
        rv **= other
        return rv


    def __rpow__ (self, other, modulo=None):
        """You might not think that this is common functionality, but it's important
        for undoing logarithms; i.e. 10**x."""

        if modulo is not None:
            raise ValueError ('powmod behavior forbidden with Approximates')

        rv = self * np.log (other)
        rv._inplace_exp ()
        return rv


    # Comparisons. We are conservative with the build-in operators, but have
    # some options that are helpful.

    def __lt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<" comparison; use lt() with caution')

    def __le__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<=" comparison; use le() with caution')

    def __gt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">" comparison; use gt() with caution')

    def __ge__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">=" comparison; use ge() with caution')

    def __eq__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "==" comparison')

    def __ne__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "!=" comparison')

    def lt (self, other):
        """Returns true where the values in *self* are less than *other*, following this
        definition:

        - The result for *undef* values is always false
        - The result for *msmt* values is the result of comparing the "representative" value.
        - The result for limits is true if the limit is contained entirely within
          the proposed interval.

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
        """Note that cannot simply invert `lt` since we have to handle undefs properly."""
        rv = (((self.data['kind'] == Kind.msmt) | (self.data['kind'] == Kind.lower))
              & (self.data['x'] > other))

        if other < 0 and self.domain == Domain.nonnegative:
            rv[self.data['kind'] == Kind.upper] = True

        if self._scalar:
            rv = rv[0]
        return rv


    # Indexing

    def __getitem__ (self, index):
        if self._scalar:
            raise IndexError ('invalid index to scalar variable')
        return Approximate (self.domain, self.data[index])

    def __setitem__ (self, index, value):
        if self._scalar:
            raise TypeError ('object does not support item assignment')
        value = Approximate.from_other (value, copy=False)
        self.domain = Domain.union[_ordpair (self.domain, value.domain)]
        self.data[index] = value.data


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


    def __unicode__ (self):
        if self._scalar:
            datum = Approximate._str_one (self.data)
            return datum + ' <%s %s scalar>' % (Domain.names[self.domain],
                                                self.__class__.__name__)
        else:
            text = np.array2string (self.data, formatter={'all': Approximate._str_one})
            return text + ' <%s %s %r-array>' % (Domain.names[self.domain],
                                                 self.__class__.__name__,
                                                 self.shape)

    __str__ = unicode_to_str


    def __repr__ (self):
        return str (self)


    @staticmethod
    def parse (text, domain=Domain.anything):
        """This only handles scalar values. Accepted formats are:

        "?"
        "<{float}"
        ">{float}"
        "{float}"
        "{float}pm{float}"

        where {float} stands for a floating-point number.
        """
        domain = Domain.normalize (domain)
        rv = Approximate (domain, ())

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


def _approximate_unary_absolute (q):
    return q.copy ()._inplace_abs ()

def _approximate_unary_exp (q):
    return q.copy ()._inplace_exp ()

def _approximate_unary_log (q):
    domain = Domain.anything
    data = q.data.copy ()

    if q.domain == Domain.nonnegative:
        np.divide (data['u'], data['x'], data['u'])
        np.log (data['x'], data['x'])
    else:
        m = (data['x'] <= 0) | (data['kind'] == Kind.upper) | (data['kind'] == Kind.undef)
        data['kind'][m] = Kind.undef
        data['u'][m] = np.nan
        data['x'][m] = np.nan
        m = ~m
        data['u'][m] = data['u'][m] / data['x'][m]
        data['x'][m] = np.log (data['x'][m])

    return Approximate (domain, data)

def _approximate_unary_log10 (q):
    return _approximate_unary_log (q) / np.log (10)

def _approximate_unary_negative (q):
    return q.copy ()._inplace_negate ()

def _approximate_unary_reciprocal (q):
    return q.copy ()._inplace_reciprocate ()


approximate_unary_math = {
    'absolute': _approximate_unary_absolute,
    'exp': _approximate_unary_exp,
    'log10': _approximate_unary_log,
    'log': _approximate_unary_log,
    'negative': _approximate_unary_negative,
    'reciprocal': _approximate_unary_reciprocal,
}




# We want/need to provide a library of standard math functions that can
# operate on any kind of measurement -- I cannot find a way to get Numpy to
# delegate the various standard options in a vectorized manner. Rather than
# writing several large and complicated handle-anything functions, we
# implement them independently for each data type, then have generic code the
# uses the type of its argument to determine which function to invoke. This
# declutters things and also lets the implementations of math operators take
# advantage of the unary functions.

basic_unary_math = {
    'absolute': np.absolute,
    'arccos': np.arccos,
    'arcsin': np.arcsin,
    'arctan': np.arctan,
    'cos': np.cos,
    'expm1': np.expm1,
    'exp': np.exp,
    'isfinite': np.isfinite,
    'log10': np.log10,
    'log1p': np.log1p,
    'log2': np.log2,
    'log': np.log,
    'negative': np.negative,
    'reciprocal': lambda x: 1. / x, # np.reciprocal floordivs ints. I don't want that.
    'sin': np.sin,
    'sqrt': np.sqrt,
    'square': np.square,
    'tan': np.tan,
}


def _dispatch_unary_math (name, value):
    if isinstance (value, Approximate):
        table = approximate_unary_math
    elif isinstance (value, Sampled):
        table = sampled_unary_math
    elif try_asarray (value) is not None:
        table = basic_unary_math
    else:
        raise ValueError ('cannot treat %r as a numerical for %s' % (value, name))

    func = table.get (name)
    if func is None:
        raise ValueError ('no implementation of %s for %r' % (name, value))
    return func (value)


def _make_wrapped_unary_math (name):
    def unary_mathfunc (val):
        return _dispatch_unary_math (name, val)
    return unary_mathfunc


def _init_unary_math ():
    """This function creates a variety of global unary math functions, matching
    the keys in :data:`basic_unary_math`.

    """
    g = globals ()

    for name in six.iterkeys (basic_unary_math):
        g[name] = _make_wrapped_unary_math (name)

_init_unary_math ()
