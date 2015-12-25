# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""
pwkit.msmt - Working with uncertain measurements.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''uval_unary_math''').split ()

import operator, six
from six.moves import range
import numpy as np

from . import PKError, unicode_to_str
from .simpleenum import enumeration
from .numutil import broadcastize


@enumeration
class Domains (object):
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
            dval = Domains.names.index (domain)
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
    """For use with Domains data tables."""
    if v1 > v2:
        return (v2, v1)
    return (v1, v2)


def _all_in_domain (data, domain):
    if domain == Domains.anything:
        return True
    if domain == Domains.nonnegative:
        return np.all (data >= 0)
    if domain == Domains.nonpositive:
        return np.all (data <= 0)


@enumeration
class Kinds (object):
    undef = 0
    msmt = 1
    upper = 2
    lower = 3

    names = ['undef', 'msmt', 'upper', 'lower']

    @classmethod
    def check (cls, kinds):
        if not np.all ((kinds >= Kinds.undef) & (kinds <= Kinds.lower)):
            raise ValueError ('illegal Aval kind(s) %r' % kinds)
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


# Avals -- "approximate" measurements. We do very simpleminded error
# propagation as a courtesy, but it very easily confused and should not be
# used for careful work.

class AvalDtypeGenerator (object):
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

get_aval_dtype = AvalDtypeGenerator ()


def make_aval_data (kind, x, u):
    data = np.empty (kind.shape, dtype=get_aval_dtype (x.dtype))
    data['kind'] = kind
    data['x'] = x
    data['u'] = u
    return data


class Aval (object):
    """An approximate uncertain value, represented with a scalar uncertainty
    parameter.

    """
    __slots__ = ('domain', 'data')

    def __init__ (self, domain, shape_or_data=None, sample_dtype=np.double):
        self.domain = Domains.normalize (domain)

        if isinstance (shape_or_data, (tuple,) + six.integer_types):
            self.data = np.empty (shape_or_data, dtype=get_aval_dtype (sample_dtype))
            self.data['kind'].fill (Kinds.undef)
        elif isinstance (shape_or_data, (np.ndarray, np.void)):
            # Scalars end up as the `np.void` type. It's hard to check the
            # array dtype thoroughly but let's do this:
            try:
                assert shape_or_data['kind'].dtype == np.uint8
            except Exception:
                raise ValueError ('illegal Aval initializer array %r' % shape_or_data)
            self.data = shape_or_data
        else:
            raise ValueError ('unrecognized Aval initializer %r' % shape_or_data)

    @staticmethod
    def from_other (v, copy=True, domain=None):
        if isinstance (v, Aval):
            if copy:
                return v.copy ()
            return v

        # TODO: handle other value types. If we're here, we just have
        # some array of floats. Domains can only get less restrictive when
        # we do math on them, so it's OK to choose the best domain we can
        # given the data we have.

        v = np.asarray (v)
        if domain is None:
            if np.all (v >= 0):
                domain = Domains.nonnegative
            elif np.all (v <= 0):
                domain = Domains.nonpositive
            else:
                domain = Domains.anything

        return Aval.from_arrays (domain, Kinds.msmt, v, 0)

    @staticmethod
    def from_arrays (domain, kind, x, u):
        domain = Domains.normalize (domain)
        Kinds.check (kind)
        if not _all_in_domain (x, domain):
            raise ValueError ('illegal Aval x initializer: data %r do not lie in '
                              'stated domain' % x)

        x = np.asarray (x)
        r = Aval (domain, x.shape)
        r.data['kind'] = kind
        r.data['x'] = x
        r.data['u'] = u

        if not _all_in_domain (r.data['u'], Domains.nonnegative):
            raise ValueError ('illegal Aval u initializer: some values %r '
                              'are negative' % u)

        return r

    def copy (self):
        return self.__class__ (self.domain, self.data.copy ())


    # Basic array properties

    @property
    def shape (self):
        return self.data.shape

    @property
    def ndim (self):
        return self.data.ndim

    @property
    def size (self):
        return self.data.size

    @property
    def sample_dtype (self):
        return self.data.dtype['x']

    def __len__ (self):
        if not len (self.data.shape):
            raise TypeError ('len() of unsized object')
        return self.data.shape[0]


    # Math.

    def _inplace_negate (self):
        self.domain = Domains.negate[self.domain]
        self.data['kind'] = Kinds.negate[self.data['kind']]
        np.negative (self.data['x'], self.data['x'])
        # uncertainty unchanged.
        return self


    def _inplace_abs (self):
        self.domain = Domains.nonnegative
        np.absolute (self.data['x'], self.data['x'])
        # uncertainty unchanged
        assert False, 'figure out what to do here'
        return self


    def _inplace_reciprocate (self):
        # domain unchanged
        self.data['kind'] = Kinds.reciprocal[self.data['kind']]
        # np.reciprocal() truncates integers, which we don't want
        np.divide (1, self.data['x'], self.data['x'])
        np.multiply (self.data['u'], self.data['x']**2, self.data['u'])
        return self


    def _inplace_exp (self):
        self.domain = Domains.nonnegative
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
        other = Aval.from_other (other, copy=False)
        self.domain = Domains.add[_ordpair (self.domain, other.domain)]
        self.data['kind'] = Kinds.add[Kinds.binop (self.data['kind'], other.data['kind'])]
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
        other = Aval.from_other (other, copy=False)
        self.domain = Domains.mul[_ordpair (self.domain, other.domain)]
        negate = (self.data['x'] < 0) ^ (other.data['x'] < 0)
        self.data['kind'] = Kinds.posmul[Kinds.binop (self.data['kind'], other.data['kind'])]
        self.data['kind'][negate] = Kinds.negate[self.data['kind'][negate]]
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
            raise ValueError ('powmod behavior forbidden with Avals')

        try:
            v = float (other)
        except TypeError:
            raise ValueError ('Avals can only be exponentiated by exact values')

        if v == 0:
            self.domain = Domains.nonnegative
            defined = (self.data['kind'] != Kinds.undef)
            self.data['kind'][defined] = Kinds.msmt
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

            if self.domain != Domains.nonnegative:
                undef = (self.data['kind'] == Kinds.upper) | (self.data['x'] < 0)
                self.data['kind'][undef] = Kinds.undef
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
            raise ValueError ('powmod behavior forbidden with Avals')

        rv = self * np.log (other)
        rv._inplace_exp ()
        return rv


    # Unary math operations that Numpy will delegate to. These are the ones
    # listed in [1] that have a `TD(P,...)` clause (cf. [2]).
    # [1] https://github.com/numpy/numpy/blob/master/numpy/core/code_generators/generate_umath.py#L247
    # [2] https://mail.scipy.org/pipermail/numpy-discussion/2012-February/060130.html).

    def exp (self):
        rv = self.copy ()
        rv._inplace_exp ()
        return rv


    def log (self):
        domain = Domains.anything
        data = self.data.copy ()

        if self.domain == Domains.nonnegative:
            np.divide (data['u'], data['x'], data['u'])
            np.log (data['x'], data['x'])
        else:
            m = (data['x'] <= 0) | (data['kind'] == Kinds.upper) | (data['kind'] == Kinds.undef)
            data['kind'][m] = Kinds.undef
            data['u'][m] = np.nan
            data['x'][m] = np.nan
            m = ~m
            data['u'][m] = data['u'][m] / data['x'][m]
            data['x'][m] = np.log (data['x'][m])

        return Aval (domain, data)


    def log10 (self):
        return self.log () / np.log (10)


    # Indexing

    def __getitem__ (self, index):
        return Aval (self.domain, self.data[index])

    def __setitem__ (self, index, value):
        value = Aval.from_other (value, copy=False)
        self.domain = Domains.union[_ordpair (self.domain, value.domain)]
        self.data[index] = value.data


    # Stringification

    @staticmethod
    def _str_one (datum):
        k = datum['kind']

        # TODO: better control of precision

        if k == Kinds.undef:
            return '?'
        elif k == Kinds.lower:
            return '>%.4f' % datum['x']
        elif k == Kinds.upper:
            return '<%.4f' % datum['x']

        return '%.3fpm%.3f' % (datum['x'], datum['u'])


    def __unicode__ (self):
        if self.ndim == 0:
            datum = Aval._str_one (self.data)
            return datum + ' <%s %s scalar>' % (Domains.names[self.domain],
                                                self.__class__.__name__)
        else:
            text = np.array2string (self.data, formatter={'all': Aval._str_one})
            return text + ' <%s %s %r-array>' % (Domains.names[self.domain],
                                                 self.__class__.__name__,
                                                 self.shape)

    __str__ = unicode_to_str


    def __repr__ (self):
        return str (self)


    @staticmethod
    def parse (text, domain=Domains.anything):
        """This only handles scalar values. Accepted formats are:

        "?"
        "<{float}"
        ">{float}"
        "{float}"
        "{float}pm{float}"

        where {float} stands for a floating-point number.
        """
        domain = Domains.normalize (domain)
        rv = Aval (domain, ())

        if text[0] == '<':
            rv.data['kind'] = Kinds.upper
            rv.data['x'] = float (text[1:])
            rv.data['u'] = 0
        elif text[0] == '>':
            rv.data['kind'] = Kinds.lower
            rv.data['x'] = float (text[1:])
            rv.data['u'] = 0
        elif text == '?':
            rv.data['kind'] = Kinds.undef
            rv.data['x'] = np.nan
            rv.data['u'] = np.nan
        else:
            rv.data['kind'] = Kinds.msmt
            pieces = text.split ('pm', 1)
            rv.data['x'] = float (pieces[0])

            if len (pieces) == 1:
                rv.data['u'] = 0
            else:
                rv.data['u'] = float (pieces[1])

        if rv.data['kind'] != Kinds.undef:
            if not _all_in_domain (rv.data['x'], rv.domain):
                raise ValueError ('value of %s is not in stated domain %s' %
                                  (text, Domains.names[domain]))
            if rv.data['u'] < 0:
                raise ValueError ('uncertainty of %s is negative' % text)

        return rv


# Uvals -- uncertain values where the uncertainties are propagated correctly,
# simply by doing math on large arrays of samples.

n_samples = 1024 - 1 # each Uval takes 1024*itemsize bytes

class UvalDtypeGenerator (object):
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
            (str ('samples'), sample_dtype, n_samples),
        ])
        self.cache[sample_dtype] = dtype
        return dtype

get_uval_dtype = UvalDtypeGenerator ()

def make_uval_data (kinds, samples):
    data = np.empty (kinds.shape, dtype=get_uval_dtype (samples.dtype))
    data['kind'] = kinds
    data['samples'] = samples
    return data


uval_default_repval_method = 'pct'


class Uval (object):
    """An empirical uncertain value, represented by samples.

    """
    __slots__ = ('domain', 'data')

    def __init__ (self, domain, shape_or_data=None, sample_dtype=np.double):
        self.domain = Domains.normalize (domain)

        if isinstance (shape_or_data, (tuple,) + six.integer_types):
            self.data = np.empty (shape_or_data, dtype=get_uval_dtype (sample_dtype))
            self.data['kind'].fill (Kinds.undef)
        elif isinstance (shape_or_data, np.ndarray):
            # It's hard to check the array dtype thoroughly but let's do this:
            try:
                assert shape_or_data['kind'].dtype == np.uint8
            except Exception:
                raise ValueError ('illegal Uval initializer array %r' % shape_or_data)
            self.data = shape_or_data
        else:
            raise ValueError ('unrecognized Uval initializer %r' % shape_or_data)

    @staticmethod
    def from_other (v, copy=True):
        if isinstance (v, Uval):
            if copy:
                return v.copy ()
            return v

        return Uval.from_fixed (Domains.anything, Kinds.sampled, v)

    @staticmethod
    def from_fixed (domain, kind, v):
        domain = Domains.normalize (domain)
        Kinds.check (kind)
        if not _all_in_domain (v, domain):
            raise ValueError ('illegal Uval initializer: data %r do not lie in '
                              'stated domain' % v)

        v = np.asarray (v)
        r = Uval (domain, v.shape)
        r.data['kind'] = kind
        r.data['samples'] = v[...,None]
        return r

    @staticmethod
    def from_norm (mean, std, shape=(), domain=Domains.anything):
        domain = Domains.normalize (domain)
        if std < 0:
            raise ValueError ('std must be positive')

        r = Uval (domain, shape)
        r.data['kind'].fill (Kinds.sampled)
        r.data['samples'] = np.random.normal (mean, std, shape+(n_samples,))
        return r

    def copy (self):
        return self.__class__ (self.domain, self.data.copy ())

    # Basic array properties

    @property
    def shape (self):
        return self.data.shape

    @property
    def ndim (self):
        return self.data.ndim

    @property
    def size (self):
        return self.data.size

    # Math. We start with addition. It gets complicated!

    def __neg__ (self):
        return _uval_unary_negative (self)

    def __abs__ (self):
        return _uval_unary_absolute (self)


    def __add__ (self, other):
        other = Uval.from_other (other, copy=False)
        dom = Domains.add[_ordpair (self.domain, other.domain)]
        kind = Kinds.add[Kinds.binop (self.data['kind'], other.data['kind'])]
        tot = self.data['samples'] + other.data['samples']
        return Uval (dom, make_uval_data (kind, tot))

    def __iadd__ (self, other):
        other = Uval.from_other (other, copy=False)
        self.domain = Domains.add[_ordpair (self.domain, other.domain)]
        self.data['kind'] = Kinds.add[Kinds.binop (self.data['kind'], other.data['kind'])]
        self.data['samples'] += other.data['samples']
        return self

    __radd__ = __add__

    def __sub__ (self, other):
        return self + (-other)

    def __isub__ (self, other):
        self += (-other)
        return self

    __rsub__ = __sub__


    def __mul__ (self, other):
        other = Uval.from_other (other, copy=False)
        dom = Domains.mul[_ordpair (self.domain, other.domain)]



def _uval_unary_negative (v):
    r = v.copy ()
    r.domain = Domains.negate[v.domain]
    r.data['kind'] = Kinds.negate[v.data['kind']]
    np.negative (r.data['samples'], r.data['samples'])
    return r


def _uval_unary_absolute (v):
    r = v.copy ()
    r.domain = Domains.nonnegative
    np.absolute (r.data['samples'], r.data['samples'])

    assert False, 'figure out what to do here'
    ##i = np.nonzero (r.data['kind'] == Kinds.past_zero)
    ##r.data['samples'][i] = 0.
    ##r.data['kind'][i] = Kinds.to_inf

    return r
