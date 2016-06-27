# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Tests for pwkit.mathlib."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import testing as nt
from pwkit import mathlib as ml


def test_simple ():
    assert ml.add (1, 1) == 2
    assert ml.exp (2) == np.exp (2)


def test_broadcast_shapes ():
    assert ml.broadcast_shapes ((1,2,3,4)) == (1,2,3,4)
    assert ml.broadcast_shapes ((2,3,4), (1,1,1,1)) == (1,2,3,4)
    assert ml.broadcast_shapes ((1,0,4), (2,1,1)) == (2,0,4)
    nt.assert_raises (ValueError, lambda: ml.broadcast_shapes ((2,1), (3,1)))
    nt.assert_raises (ValueError, lambda: ml.broadcast_shapes ((2,1), (3,0)))


def test_numpy_array_meta ():
    assert ml.get_dtype (1) == np.int
    assert ml.get_dtype ([1]) == np.int
    assert ml.get_dtype ((1,)) == np.int

    assert ml.get_size (1) == 1
    assert ml.get_size ([]) == 0
    assert ml.get_size ([[1]]) == 1

    assert ml.shape (1) == ()
    assert ml.shape ([]) == (0,)
    assert ml.shape ([[1]]) == (1,1)


def test_numpy_int_core_math ():
    a = np.arange (4)

    # basic unary math
    nt.assert_array_equal (ml.square (a), a*a)

    # basic binary math, different operand types
    nt.assert_array_equal (ml.add (a, 1), a + 1)
    nt.assert_array_equal (ml.add (a, 1.5), a + 1.5)
    nt.assert_array_equal (ml.add (a, 1.5+2j), a + (1.5+2j))

    # basic output array, unary and binary
    out = np.zeros ((4,), dtype=np.int)
    ml.square (a, out)
    nt.assert_array_equal (np.square (a), out)
    ml.add (a, a, out)
    nt.assert_array_equal (2 * a, out)

    # different-shaped output array
    myout = np.zeros ((1,1,4), dtype=np.int)
    npout = np.zeros ((1,1,4), dtype=np.int)
    ml.square (a, myout)
    np.square (a, npout)
    nt.assert_array_equal (myout, npout)
    ml.add (a, a, myout)
    np.add (a, a, npout)
    nt.assert_array_equal (myout, npout)

    # different-typed output array
    myout = np.zeros ((4,), dtype=np.double)
    npout = np.zeros ((4,), dtype=np.double)
    ml.square (a, myout)
    np.square (a, npout)
    nt.assert_array_equal (myout, npout)
    ml.add (a, a, myout)
    np.add (a, a, npout)
    nt.assert_array_equal (myout, npout)

    # unary floatification
    nt.assert_array_equal (ml.exp (a), np.exp (a))

    # illegal type-unsafe output array
    out = np.zeros ((4,), dtype=np.int)
    nt.assert_raises (TypeError, lambda: np.exp (a, out))
    nt.assert_raises (TypeError, lambda: ml.exp (a, out))


def _try_different_flavors (base_array):
    yield base_array

    from pwkit.msmt import Approximate, Sampled
    yield Sampled.from_other (base_array)
    yield Approximate.from_other (base_array)

    from pwkit.pktable import ScalarColumn, MeasurementColumn
    yield ScalarColumn.new_from_data (base_array)
    yield MeasurementColumn.new_from_data (base_array)


def test_absolute ():
    a = np.array ([0, -1, 1])
    t = np.abs (a)

    for f in _try_different_flavors (a):
        nt.assert_array_equal (ml.abs (a), t)

    a = np.array ([0., -0., np.nan, np.inf, -np.inf, -5, 5.])
    t = np.abs (a)

    for f in _try_different_flavors (a):
        nt.assert_array_equal (ml.abs (a), t)
