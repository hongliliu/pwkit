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
        #self.cols[key] = value
        raise NotImplementedError ('TODO: table indexing with table views')


    def __repr__ (self):
        return '<PKTable: %d cols, %d rows>' % self.shape



class _PKTableColumnsHelper (object):
    """The object that implements the `PKTable.cols` functionality."""

    def __init__ (self, owner):
        self._owner = owner
        self._data = owner._data

    def __repr__ (self):
        return '<PKTable columns: %s>' % (' '.join (six.viewkeys (self._data)))

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

            # Nothing specific worked. Last-ditch effort: np.asarray(). This
            # function will accept literally any argument, which is a bit
            # much, so we're a bit pickier and only take
            # bool/int/float/complex values. (Generically, it will give you an
            # array of shape () and dtype Object.) We're willing to work with
            # arrays of shape (), (n,), and (1,n), if they agree with nrows,
            # when it's known.

            arr = np.asarray (sval)
            if arr.dtype.kind not in 'bifc':
                raise ValueError ('unhandled PKTable column value %r for %r' % (sval, skey))

            if not len (self._data):
                if arr.ndim == 0:
                    # This is actually somewhat ill-defined. Let's err on the side of caution.
                    raise ValueError ('cannot set first PKTable column to a scalar')
                elif arr.ndim == 1:
                    self._data[skey] = ScalarColumn (None, _data=arr)
                elif arr.ndim == 2:
                    if arr.shape[0] != 1:
                        # Of course, we could implement this like Astropy ...
                        raise ValueError ('cannot set PKTable column to a >1D array')
                    self._data[skey] = ScalarColumn (None, _data=arr[0])
                else:
                    raise ValueError ('unexpected PKTable column value %r for %r' % (sval, skey))
                continue

            nrow = self._owner.shape[1]
            if arr.ndim == 0:
                newcol = ScalarColumn (nrow, arr.dtype)
                newcol._data.fill (sval)
                self._data[skey] = newcol
            elif arr.shape == (nrow,):
                self._data[skey] = ScalarColumn (None, _data=arr)
            elif arr.shape == (1,nrow):
                self._data[skey] = ScalarColumn (None, _data=arr[0])
            else:
                raise ValueError ('unexpected PKTable column value %r for %r' % (sval, skey))


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


    def _sample_for_repr (self):
        n = len (self)

        if n < 5:
            return [repr (self._get_index (i)) for i in range (n)]

        return [repr (self._get_index (0)),
                repr (self._get_index (1)),
                '...',
                repr (self._get_index (n - 2)),
                repr (self._get_index (n - 1))]


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

    def __init__ (self, len, dtype=np.double, _data=None):
        if _data is not None:
            self._data = _data
            return

        try:
            len = int (len)
        except Exception:
            raise ValueError ('ScalarColumn length must be an integer')

        self._data = np.empty (len, dtype=dtype)

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


    # TODO: override __{get,set}item__ to be faster when possible and to allow
    # broadcasting.

    def _get_index (self, idx):
        return self._data[idx]

    def _set_index (self, idx, value):
        self._data[idx] = value


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


    def _sample_for_repr (self):
        from .msmt import Aval
        n = len (self)

        if n < 5:
            return [Aval._str_one (self._data[i].data) for i in range (n)]

        return [Aval._str_one (self._data[0].data),
                Aval._str_one (self._data[1].data),
                '...',
                Aval._str_one (self._data[-2].data),
                Aval._str_one (self._data[-1].data)]

    def _coldesc_for_repr (self):
        from .msmt import Domains

        return '%s %s' % (Domains.names[self._data.domain], self.__class__.__name__)


    # TODO: override __{get,set}item__ to be faster when possible and to allow
    # broadcasting.

    def _get_index (self, idx):
        return self._data[idx]

    def _set_index (self, idx, value):
        self._data[idx] = value
