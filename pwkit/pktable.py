# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""pwkit.pktable - I can't believe I'm writing my own table class.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''PKTable PKTableColumnABC ScalarColumn''').split ()

import abc, itertools, operator, six, types
from collections import OrderedDict
from six.moves import range
import numpy as np


def _is_sequence (thing):
    if isinstance (thing, six.string_types):
        # We never want to treat a string as a sequence of characters. Use
        # `list(strval)` if that's really what you want.
        return False

    if isinstance (thing, tuple):
        raise ValueError ('rows and columns must be indexed with lists and not '
                          'tuples, to avoid ambiguities in table slicing')

    try:
        iter (thing)
    except Exception:
        return False
    return True


def _normalize_index (idx, n, desc):
    norm_idx = idx
    if norm_idx < 0:
        norm_idx += n
    if norm_idx < 0 or norm_idx >= n:
        raise ValueError ('illegal %s index %d; there are %d %ss' % (idx, n, desc))
    return norm_idx



class PKTable (object):
    """Yet another data table class. This one lacks some sophisticated features
    and isn't speedy, but it has aggressive support for "composite" columns
    that are made out of nontrivial data structures.

    Our `shape`-type values are a bit funny. `self.shape` returns `(ncols,
    nrows)` while from a standard Numpy perspective, the ordering should be
    `(nrows, ncols)`. Unlike Pandas, `len(self)` returns the number of
    columns.

    """
    _data = None
    "An OrderedDict of columns stored by name."

    cols = None
    "A helper object for manipulating the table columns."

    rows = None
    "A helper object for manipulating the table rows."

    def __init__ (self):
        self._data = OrderedDict ()
        self.cols = _PKTableColumnsHelper (self)
        self.rows = _PKTableRowsHelper (self)


    @property
    def shape (self):
        ncol = len (self._data)
        if not ncol:
            return 0, 0
        return ncol, len (iter (six.viewvalues (self._data)).next ())


    def __len__ (self):
        return len (self._data)


    def __iter__ (self):
        """Iterate over the columns. Note that this behavior is important when writing
        `PKTable[cols,rows] = OtherPKtable`.

        """
        return six.itervalues (self._data)


    def __getitem__ (self, key):
        if not isinstance (key, tuple):
            # Non-tuple indexing is defined to be on the columns.
            return self.cols[key]

        if len (key) != 2:
            raise ValueError ('PKTables may only be indexed with length-2 tuples')

        colkey, rowkey = key

        if isinstance (rowkey, types.EllipsisType):
            return self.cols[colkey]

        if isinstance (colkey, types.EllipsisType):
            return self.rows[rowkey]

        cols, single_col_requested = self.cols._fetch_columns (colkey)
        if single_col_requested:
            # A row-filtered view of a single column. We can delegate.
            return self._data[cols[0]][rowkey]

        # A sub-table view.
        retval = self.__class__ ()
        for col in cols: # do things this way to preserve ordering
            retval._data[col] = self._data[col][rowkey]
        return retval


    def __setitem__ (self, key, value):
        if not isinstance (key, tuple):
            # Non-tuple indexing is defined to be on the columns.
            self.cols[key] = value
            return

        if len (key) != 2:
            raise ValueError ('PKTables may only be indexed with length-2 tuples')

        colkey, rowkey = key

        if isinstance (rowkey, types.EllipsisType):
            self.cols[colkey] = value
            return

        if isinstance (colkey, types.EllipsisType):
            self.rows[rowkey] = value
            return

        cols, single_col_requested = self.cols._fetch_columns (colkey)
        if single_col_requested:
            # A row-filtered view of a single column. We can delegate.
            # TODO: what if the requested column doesn't exist?
            self._data[cols[0]][rowkey] = value
            return

        # We're assigning to a sub-table.
        sentinel = object ()
        for colname, subval in itertools.izip (cols, value, fillvalue=sentinel):
            if colname is sentinel or subval is sentinel:
                raise ValueError ('disagreeing number of columns in PKTable item assignment')
            self._data[colname][rowkey] = subval


    # Stringification. TODO: a "hiding" mechanism where the user can cause
    # columns not to be shown in the stringification, for focusing on a few
    # particular columns in a large table.

    def __repr__ (self):
        if len (self) == 0:
            return '(empty PKTable)'

        if len (self._data) > 9:
            colnames = self._data.keys ()[:4] + ['...'] + self._data.keys ()[-4:]
            colobjs = self._data.values ()[:4] + [None] + self._data.values ()[-4:]
        else:
            colnames = list (self._data.keys ())
            colobjs = list (self._data.values ())

        nrow = self.shape[1]

        if nrow > 11:
            rowids = [0, 1, 2, 3, 4, None, nrow - 5, nrow - 4, nrow - 3, nrow - 2, nrow - 1]
        else:
            rowids = list (range (nrow))

        maxwidths = np.array ([len (n) for n in colnames], dtype=np.int)
        buffer = [None] * len (rowids)

        for i, rid in enumerate (rowids):
            if rid is None:
                continue

            buffer[i] = colvals = [''] * len (colobjs)

            for j, col in enumerate (colobjs):
                if col is None:
                    continue

                colvals[j] = col._repr_single_item (rid)
                maxwidths[j] = max (maxwidths[j], len (colvals[j]))

        totwidth = maxwidths.sum () + 2 * (len (colnames) - 1)

        for i in range (len (rowids)):
            if buffer[i] is None:
                buffer[i] = '(...)'.center (totwidth)
            else:
                buffer[i] = '  '.join ([buffer[i][j].rjust (maxwidths[j]) for j in range (len (colnames))])

        lines = ['  '.join ([n.rjust (maxwidths[i]) for i, n in enumerate (colnames)])]

        if not nrow:
            lines += ['(no rows)'.center (totwidth)]
        else:
            lines += ['']
            lines += buffer

        lines += ['', ('(PKTable: %d cols, %d rows)' % self.shape).rjust (totwidth)]
        return '\n'.join (lines)


class _PKTableColumnsHelper (object):
    """The object that implements the `PKTable.cols` functionality."""

    def __init__ (self, owner):
        self._owner = owner
        self._data = owner._data

    def __len__ (self):
        return len (self._data)

    def __iter__ (self):
        return iter (self._data)


    def _fetch_columns (self, key):
        """Given an indexer `key`, fetch a list of column names corresponding to the
        request. We return `(cols, single_col_requested)`, where `cols` is a
        list of specific column names and `single_col_requested` indicates if
        the key is clearly requesting a single column (to be contrasted with a
        sub-table that happens to only have one column). If the latter is the
        case, the returned `cols` will have only one element. The specified
        column names may not necessarily exist in `self._data`!

        """
        if _is_sequence (key):
            maybe_single_col = False
        else:
            maybe_single_col = True
            key = [key]

        retcols = []

        for thing in key:
            if isinstance (thing, six.integer_types):
                retcols.append (self._data.keys ()[thing])
            elif isinstance (thing, six.string_types):
                retcols.append (thing)
            elif isinstance (thing, types.SliceType):
                maybe_single_col = False
                retcols += self._data.keys ()[thing]
            else:
                raise KeyError ('unhandled PKTable column indexer %r' % (thing,))

        if len (retcols) == 1 and maybe_single_col:
            # If all the evidence is that the caller wanted to pull out a
            # single column, indicate this. Something like `t.cols[['foo']]`
            # will NOT do this, though, since the intent is pretty clearly to
            # extract a sub-table.
            return retcols, True

        return retcols, False


    def __getitem__ (self, key):
        cols, single_col_requested = self._fetch_columns (key)

        if single_col_requested:
            return self._data[cols[0]]

        retval = self._owner.__class__ ()
        for col in cols: # do things this way to preserve ordering
            retval._data[col] = self._data[col]
        return retval


    def _check_col_shape (self, colkey, colval):
        if not len (self._data):
            return # no problem if this is the first column

        ncur = len (iter (six.viewvalues (self._data)).next ())
        nnew = len (colval)

        if nnew != ncur:
            raise ValueError ('cannot add column %r of length %d as "%s"; it does '
                              'not match the table length %d' % (colval, nnew, colkey, ncur))


    def __setitem__ (self, key, value):
        retcols, single_col_requested = self._fetch_columns (key)
        if single_col_requested:
            value = [value]

        for skey, sval in itertools.izip (retcols, value):
            if isinstance (sval, PKTableColumnABC):
                self._check_col_shape (skey, sval)
                self._data[skey] = sval
                continue

            # TODO: try other classes

            pass

            # Below this point, we have a generic system for dealing with
            # input values that resemble numpy arrays. TODO: make the system
            # for recognizing these more generic and extensible.

            arrayish_factory = None
            arrayish_data = None

            from .msmt import Aval
            if isinstance (sval, Aval):
                arrayish_factory = AvalColumn._new_from_data
                arrayish_data = sval

            # Nothing specific worked. Last-ditch effort: np.asarray(). This
            # function will accept literally any argument, which is a bit
            # much, so we're a bit pickier and only take
            # bool/int/float/complex values. (Generically, it will give you an
            # array of shape () and dtype Object.) We're willing to work with
            # arrays of shape (), (n,), and (1,n), if they agree with nrows,
            # when it's known.

            if arrayish_factory is None:
                arrayish_data = np.asarray (sval)
                if arrayish_data.dtype.kind not in 'bifc':
                    raise ValueError ('unhandled PKTable column value %r for %r' % (sval, skey))

                arrayish_factory = ScalarColumn._new_from_data

            # At this point arrayish_{factory,data} have been set and we can use
            # our generic infrastructure.

            if not len (self._data):
                if arrayish_data.ndim == 0:
                    # This is actually somewhat ill-defined. Let's err on the side of caution.
                    raise ValueError ('cannot set first PKTable column to a scalar')
                elif arrayish_data.ndim == 1:
                    self._data[skey] = arrayish_factory (arrayish_data)
                elif arrayish_data.ndim == 2:
                    if arrayish_data.shape[0] != 1:
                        # Of course, we could implement this like Astropy ...
                        raise ValueError ('cannot set PKTable column to a >1D array')
                    self._data[skey] = arrayish_factory (arrayish_data[0])
                else:
                    raise ValueError ('unexpected PKTable column value %r for %r' % (sval, skey))
                continue

            nrow = self._owner.shape[1]
            if arrayish_data.ndim == 0:
                self._data[skey] = arrayish_factory (np.broadcast_to (arrayish_data), (nrow,))
            elif arrayish_data.shape == (nrow,):
                self._data[skey] = arrayish_factory (arrayish_data)
            elif arrayish_data.shape == (1,nrow):
                self._data[skey] = arrayish_factory (arrayish_data[0])
            else:
                raise ValueError ('unexpected PKTable column value %r for %r' % (sval, skey))


    def __repr__ (self):
        maxnamelen = 0
        info = []

        for name, col in six.viewitems (self._data):
            maxnamelen = max (maxnamelen, len (name))
            info.append ((name, col._coldesc_for_repr ()))

        if not len (info):
            return '(PKTable columns: none defined)'

        return '\n'.join (name.ljust (maxnamelen) + ' : ' + desc for name, desc in info)


class _PKTableRowsHelper (object):
    """The object that implements the `PKTable.rows` functionality."""

    def __init__ (self, owner):
        self._owner = owner
        self._data = owner._data

    def __repr__ (self):
        return '<PKTable rows: %d>' % len (self)

    def __len__ (self):
        if not len (self._data):
            return 0
        return len (iter (six.viewvalues (self._data)).next ())

    def __iter__ (self):
        return range (len (self))


    def _fetch_rows (self, key):
        """Given an indexer `key`, fetch a list of row indices corresponding to the
        request. We return `(rows, single_row_requested)`, where
        `single_row_requested` indicates if the key is clearly requesting a
        single row (to be contrasted with a sub-table that happens to only
        have one row). If the latter is the case, the returned `rows` will
        have only one element. Unlike the equivalent function in ColsHelper,
        existing rows must be referenced -- you cannot create rows with
        __setitem__.

        """
        if _is_sequence (key):
            maybe_single_row = False
        else:
            maybe_single_row = True
            key = [key]

        retrows = []
        n = len (self)

        for thing in key:
            if isinstance (thing, six.integer_types):
                retrows.append (_normalize_index (thing, n, 'row'))
            elif isinstance (thing, types.SliceType):
                maybe_single_row = False
                retrows += [_normalize_index (i, n, 'row')
                            for i in range (*thing.indices (n))]
            else:
                raise KeyError ('unhandled PKTable row indexer %r' % (thing,))

        if len (retrows) == 1 and maybe_single_row:
            # If all the evidence is that the caller wanted to pull out a
            # single row, indicate this. Something like `t.rows[[0]]` will NOT
            # do this, though, since the intent is pretty clearly to extract a
            # sub-table.
            return retrows, True

        return retrows, False


    def __getitem__ (self, key):
        rows, single_row_requested = self._fetch_rows (key)

        if single_row_requested:
            return _PKTableRowProxy (self._owner, rows[0])

        raise NotImplementedError ('implement row-filtered column views')


    def __setitem__ (self, key, value):
        retrows, single_row_requested = self._fetch_rows (key)
        if single_row_requested:
            value = [value]

        for skey, sval in itertools.izip (retrows, value):
            raise NotImplementedError ('hmmmmm')


class _PKTableRowProxy (object):
    """A proxy object for manipulating a specific row of a PKTable. Modifications
    of this object propagate back up to the owning table.

    """
    def __init__ (self, owner, idx):
        self._owner = owner
        self._data = owner._data
        self._idx = idx

    def __repr__ (self):
        return '<PKTable row #%d>' % self._idx

    def __len__ (self):
        return len (self._data)

    def __getitem__ (self, key):
        """We only allow scalar access here."""
        return self._data[key][self._idx]

    def __setitem__ (self, key, value):
        """We only allow scalar access here."""
        self._data[key][self._idx] = value

    def to_dict (self):
        return OrderedDict ((c, self._data[c][self._idx]) for c in six.viewkeys (self._data))

    def to_holder (self):
        from . import Holder
        return Holder (**self.to_dict ())


# Actual column types

class PKTableColumnABC (six.with_metaclass (abc.ABCMeta, object)):
    def __len__ (self):
        raise NotImplementedError ()


    def _get_index (self, idx):
        """Return the item at index `idx`, which is guaranteed to be an integer lying
        between 0 and len(self) - 1, inclusive.

        """
        raise NotImplementedError ()


    def _set_index (self, idx, value):
        """Set the item at index `idx` to `value`. The index is guaranteed to be an
        integer lying between 0 and len(self) - 1, inclusive. It's up to the
        implementation to decide what `value`s are valid.

        """
        raise NotImplementedError ()


    def __iter__ (self):
        """This is a dumb default implementation."""
        for i in range (len (self)):
            yield self._get_index (i)


    def _repr_single_item (self, idx):
        return repr (self._get_index (idx))


    def _sample_for_repr (self):
        n = len (self)

        if n < 5:
            return [self._repr_single_item (i) for i in range (n)]

        return [self._repr_single_item (0),
                self._repr_single_item (1),
                '...',
                self._repr_single_item (n - 2),
                self._repr_single_item (n - 1)]


    def _coldesc_for_repr (self):
        return self.__class__.__name__


    def __repr__ (self):
        return '[%s] (%s, %d rows)' % (', '.join (self._sample_for_repr ()),
                                       self._coldesc_for_repr (), len (self))


    def __getitem__ (self, key):
        """Get a data value (if indexed with a scalar), or a view of this column.
        Accepted indexes are scalar integers, slices, boolean vectors, and
        integer fancy-indexing vectors.

        """
        n = len (self)

        def inorm (orig_idx):
            idx = orig_idx
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise ValueError ('illegal row index %d; there are %d rows' % (orig_idx, n))
            return idx

        if isinstance (key, six.integer_types):
            return self._get_index (inorm (key))

        if isinstance (key, types.SliceType):
            return _PKTableSlicedColumnView (self, key)

        a = np.asarray (key)

        if a.dtype.kind == 'b' and a.shape == (n,):
            return _PKTableFancyIndexedColumnView (self, np.nonzero (a)[0])

        if a.dtype.kind == 'i' and a.ndim == 1:
            try:
                a = np.array ([inorm (q) for q in a])
                return _PKTableFancyIndexedColumnView (self, a)
            except ValueError:
                pass

        raise ValueError ('unhandled column indexer %r' % (key,))


    def __setitem__ (self, key, value):
        """Set a data value or values. We are lazy and so we create temporary views
        and go through __getitem__ if not indexed with a scalar.

        """
        n = len (self)

        def inorm (orig_idx):
            idx = orig_idx
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise ValueError ('illegal row index %d; there are %d rows' % (orig_idx, n))
            return idx

        if isinstance (key, six.integer_types):
            self._set_index (inorm (key), value)
            return

        # In the non-scalar case, leverage the __getitem__ views.

        view = self[key]

        for i, ival in itertools.izip (range (len (view)), value):
            view._set_index (i, ival)

        if i != len (view) - 1:
            raise ValueError ('not enough values %r to set column with indexer %r'
                              % (value, key))


class _PKTableSlicedColumnView (PKTableColumnABC):
    _parent = None
    _start = None
    _stop = None
    _stride = None

    def __init__ (self, parent, slice):
        self._parent = parent
        self._start, self._stop, self._stride = slice.indices (len (parent))

    def __len__ (self):
        return (self._stop - self._start) // self._stride

    def _get_index (self, idx):
        return self._parent._get_index (self._start + self._stride * idx)

    def _set_index (self, idx, value):
        self._parent._set_index (self._start + self._stride * idx, value)

    def _coldesc_for_repr (self):
        return 'slice view of %s' % self._parent._coldesc_for_repr ()


class _PKTableFancyIndexedColumnView (PKTableColumnABC):
    _parent = None
    _index = None

    def __init__ (self, parent, index):
        # TODO: if parent is also a fancy-index or slice column view, avoid
        # multiple layers.
        self._parent = parent
        self._index = index

    def __len__ (self):
        return len (self._index)

    def _get_index (self, idx):
        return self._parent._get_index (self._index[idx])

    def _set_index (self, idx, value):
        self._parent._set_index (self._index[idx], value)

    def _coldesc_for_repr (self):
        return 'fancy-index view of %s' % self._parent._coldesc_for_repr ()


class PKTableAlgebraColumnABC (six.with_metaclass (abc.ABCMeta, PKTableColumnABC)):
    """A column that you can do basic algebra on. Assumes that the instance has an
    attribute named `_data` on which the algebra can be performed, and that it
    has a class method _new_from_data (data) that will create a new column
    containing the specified data.

    """
    def __add__ (self, other):
        return self.__class__._new_from_data (self._data + other)

    __radd__ = __add__

    def __iadd__ (self, other):
        self._data += other
        return self

    def __sub__ (self, other):
        return self.__class__._new_from_data (self._data - other)

    def __rsub__ (self, other):
        return self.__class__._new_from_data (other - self._data)

    def __isub__ (self, other):
        self._data -= other
        return self

    def __mul__ (self, other):
        return self.__class__._new_from_data (self._data * other)

    __rmul__ = __mul__

    def __imul__ (self, other):
        self._data *= other
        return self

    def __truediv__ (self, other):
        return self.__class__._new_from_data (self._data / other)

    def __rtruediv__ (self, other):
        return self.__class__._new_from_data (other / self._data)

    def __itruediv__ (self, other):
        self._data /= other
        return self

    def __floordiv__ (self, other):
        return self.__class__._new_from_data (self._data // other)

    def __rfloordiv__ (self, other):
        return self.__class__._new_from_data (other // self._data)

    def __ifloordiv__ (self, other):
        self._data //= other
        return self

    def __pow__ (self, other, modulo=None):
        return self.__class__._new_from_data (pow (self._data, other, modulo))

    def __ipow__ (self, other):
        self._data **= other
        return self

    def __rpow__ (self, other, modulo=None):
        return self.__class__._new_from_data (pow (other, self._data, modulo))



class ScalarColumn (PKTableAlgebraColumnABC):
    _data = None
    "The actual array data."

    def __init__ (self, len, dtype=np.double, fillval=np.nan, _data=None):
        if _data is not None:
            self._data = _data
            return

        try:
            len = int (len)
        except Exception:
            raise ValueError ('ScalarColumn length must be an integer')

        self._data = np.empty (len, dtype=dtype)

        if fillval is not None:
            if self._data.dtype.kind in 'bi' and np.isnan (fillval):
                fillval = 0
            self._data.fill (fillval)


    @classmethod
    def _new_from_data (cls, data):
        return cls (None, _data=data)


    def __len__ (self):
        return len (self._data)


    def __iter__ (self):
        return iter (self._data)


    def __array__ (self, dtype=None):
        """Numpy docs: "If a class ... having the __array__ method is used as the
        output object of an ufunc, results will be written to the object
        returned by __array__. **Similar conversion is done on input
        arrays**." So, basically, this should return an equivalent Numpy array
        that math can be done on.

        """
        return self._data


    def __array_wrap__ (self, array, context=None):
        """Numpy docs: "At the end of every ufunc, this method is called on the input
        object with the highest array priority, or the output object if one
        was specified. The ufunc-computed array is passed in and whatever is
        returned is passed to the user." So, once Numpy has done math on an
        object, this is used to turn it back into our wrapper type. Columns
        are mutable containers, so we modify our contents and return self.

        """
        self._data = array
        return self


    # Indexing.

    def _get_index (self, idx):
        return self._data[idx]

    def _set_index (self, idx, value):
        self._data[idx] = value

    def __getitem__ (self, key):
        """Seeing as the ScalarColumn carries virtually no state besides the array
        data payload, I think it's OK for this to just return a numpy array.
        Any table-related routine that relies on the output of this function
        will also need to be able to accept plain-array inputs.

        """
        return self._data[key]

    def __setitem__ (self, key, value):
        self._data[key] = value


class AvalColumn (PKTableAlgebraColumnABC):
    _data = None
    "The actual Aval data."

    def __init__ (self, len, domain='anything', sample_dtype=np.double, _data=None):
        if _data is not None:
            self._data = _data
            return

        try:
            len = int (len)
        except Exception:
            raise ValueError ('AvalColumn length must be an integer')

        from .msmt import Aval
        self._data = Aval (domain, (len,), sample_dtype=sample_dtype)


    @classmethod
    def _new_from_data (cls, data):
        return cls (None, _data=data)


    def __len__ (self):
        return len (self._data)


    def __iter__ (self):
        return iter (self._data)


    def _repr_single_item (self, idx):
        from .msmt import Aval
        return Aval._str_one (self._data[idx].data)


    def _coldesc_for_repr (self):
        from .msmt import Domains

        return '%s %s' % (Domains.names[self._data.domain], self.__class__.__name__)


    # Emulating Aval attributes.

    @property
    def domain (self):
        return self._data.domain

    @domain.setter
    def domain (self, domain):
        from .msmt import Domains
        self._data.domain = Domains.normalize (domain)

    @property
    def sample_dtype (self):
        return self._data.sample_dtype


    # Indexing.

    def _get_index (self, idx):
        return self._data[idx]

    def _set_index (self, idx, value):
        self._data[idx] = value

    def __getitem__ (self, key):
        """Same line of reasoning as ScalarColumn."""
        return self._data[key]

    def __setitem__ (self, key, value):
        self._data[key] = value


class indexerproperty_Helper (object):
    def __init__ (self, instance, descriptor):
        self._instance = instance
        self._descriptor = descriptor

    def __getitem__ (self, key):
        return self._descriptor._getter (self._instance, key)

    def __setitem__ (self, key, value):
        if self._descriptor._setter is None:
            raise TypeError ('item assignment is not allowed by this property')
        return self._descriptor._setter (self._instance, key, value)

    def __delitem__ (self, key):
        if self._descriptor._deleter is None:
            raise TypeError ('item deletion is not allowed by this property')
        return self._descriptor._deleter (self._instance, key)


class indexerproperty (object):
    def __init__ (self, getter):
        self._getter = getter
        self._setter = None
        self._deleter = None

    def __get__ (self, instance, owner):
        return indexerproperty_Helper (instance, self)

    def setter (self, thesetter):
        self._setter = thesetter
        return self

    def deleter (self, thedeleter):
        self._deleter = thedeleter
        return self


class CoordColumn (PKTableColumnABC):
    """astropy.coordinates.SkyCoord instances are not mutable, so we don't use
    them for our data storage. However, we should ideally support different
    coordinate frames, etc., in the same way that it does. For now I just have
    hardcoded RA/Dec.

    """
    _data = None
    "The actual data, stored as an (nrow,2) ndarray of RA and Dec in radians."

    def __init__ (self, len, _data=None):
        if _data is not None:
            self._data = _data
            return

        try:
            len = int (len)
        except Exception:
            raise ValueError ('CoordColumn length must be an integer')

        self._data = np.empty ((len, 2))
        self._data.fill (np.nan)


    @classmethod
    def _new_from_data (cls, data):
        return cls (None, _data=data)


    def __len__ (self):
        return len (self._data)


    def __iter__ (self):
        return iter (self._data)


    def _repr_single_item (self, idx):
        ra, dec = self._data[idx]

        if not np.isfinite (ra) or not np.isfinite (dec):
            return '?'

        from .astutil import fmtradec
        return fmtradec (ra, dec)

    # Basic indexing.

    def _get_index (self, idx):
        return self._data[idx]

    def _set_index (self, idx, value):
        self._data[idx] = value

    def __getitem__ (self, key):
        """Same line of reasoning as ScalarColumn."""
        return self._data[key]

    def __setitem__ (self, key, value):
        self._data[key] = value

    # Experiment with fancy "indexer properties"

    @indexerproperty
    def radec_rad (self, idx):
        return self._data[idx]

    @radec_rad.setter
    def radec_rad (self, idx, value):
        self._data[idx] = value

    @radec_rad.deleter
    def radec_rad (self, idx):
        self._data[idx] = np.nan


    @indexerproperty
    def radec_deg (self, idx):
        from .astutil import R2D
        return self._data[idx] * R2D

    @radec_deg.setter
    def radec_deg (self, idx, value):
        from .astutil import D2R
        self._data[idx] = np.asarray (value) * D2R

    @radec_deg.deleter
    def radec_deg (self, idx):
        self._data[idx] = np.nan


    @indexerproperty
    def formatted (self, idx):
        # XXX need to handle non-scalar indices
        return self._repr_single_item (idx)

    @formatted.setter
    def formatted (self, idx, value):
        from astutil import parsehours, parsedeglat
        # XXX super not robust!
        rastr, decstr = value.split (None, 1)
        self._data[idx] = (parsehours (rastr), parsedeglat (decstr))
