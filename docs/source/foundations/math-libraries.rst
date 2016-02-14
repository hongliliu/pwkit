.. Copyright 2016 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

A Common Interface for math on arrays (:mod:`pwkit.mathlib`)
==============================================================================

.. automodule:: pwkit.mathlib
   :synopsis: Vectorized math functions that work on any data type.

.. currentmodule:: pwkit.mathlib


The “Common Interface”
------------------------------------------------------------------------------

The `pwkit.mathlib` module provides a set of free functions mirroring those in
the :mod:`numpy` module. Unlike those in Numpy, however, these will work
regardless of whether the input is a Numpy array-like, :class:`pwkit.msmt`
measurement type, table column, etc. When a function shares the name of a
Numpy function, the aim is that it has precisely the same semantics. Some of
the functions below are unique to ``pkwit``, though.

================  ===============
  Name              Description
================  ===============
`abs`             Alias for `absolute`.
`absolute`        Compute absolute value.
`add`             Compute sum.
`arccos`          Compute inverse cosine.
`arccosh`         Compute inverse hyperbolic cosine.
`arcsin`          Compute inverse sine.
`arcsinh`         Compute inverse hyperbolic sine.
`arctan`          Compute inverse tangent.
`arctan2`         Compute inverse tangent with separate numerator and denominator.
`arctanh`         Compute inverse hyperbolic tangent.
`asscalar`        Convert an array to a scalar, assuming it has exactly 1 value.
`atleast_1d`      Return a view of an array that is at least 1-dimensional.
`bitwise_and`     Compute bitwise AND; integer types only.
`bitwise_or`      Compute bitwise OR; integer types only.
`bitwise_not`     Alias for `invert`.
`bitwise_xor`     Compute bitwise XOR; integer types only.
`cbrt`            Compute cube root.
`ceil`            Compute ceiling.
`cmask`           Return a mask based on array contents.
`conj`            Alias for `conjugate`.
`conjugate`       Compute complex conjugates.
`copysign`        Copy sign bits from one set of numbers to another.
`cos`             Compute cosine.
`cosh`            Compute hyperbolic cosine.
`deg2rad`         Convert degrees to radians.
`degrees`         Alias for `rad2deg`.
`divide`          Compute ratio, with floor-divide semantics on integers in Python 2.
`dtype`           Get the data type of array elements.
`equal`           Test if values are equal.
`exp`             Compute the base-e exponential function.
`exp2`            Compute the base-2 exponential function.
`expm1`           Compute exp(x) - 1.
`fabs`            Compute absolute value; non-complex types only.
`floor`           Compute floor;
`floor_divide`    Compute ratio, with consistent floor-divide semantics.
`fmax`            Element-wise maximum; NaNs are ignored when possible (contra `maximum`).
`fmin`            Element-wise minimum; NaNs are ignored when possible (contra `minimum`).
`fmod`            Compute remainder of division; inverse sign semantics from `remainder`.
`frexp`           Decompose values into mantissa and twos exponent; inverse of `ldexp`.
`greater`         Test if values are greater than others.
`greater_equal`   Test if values are greater than or equal to others.
`hypot`           Compute ``sqrt(x**2 + y**2)``.
`invert`          Compute bitwise negation; integer types only.
`is_scalar`       Test if the argument array is scalar-compatible.
`isfinite`        Test if values are finite.
`isinf`           Test if values are infinite.
`isnan`           Test if values are NaN.
`ldexp`           Compute ``x * 2**y``; inverse of `frexp`
`left_shift`      Shift bits left; integer types only.
`less`            Test if values are less than others.
`less_equal`      Test if values are less than or equal to others.
`log`             Compute base-e logarithm.
`log10`           Compute base-10 logarithm.
`log1p`           Compute ``log(1 + x)``.
`log2`            Compute base-2 logarithm.
`logaddexp`       Compute ``log(exp(x) + exp(y))``.
`logaddexp2`      Compute ``log2(2**x + 2**y)``.
`logical_and`     Compute the logical AND.
`logical_or`      Compute the logical OR.
`logical_not`     Compute the logical NOT.
`logical_xor`     Compute the logical XOR.
`maximum`         Compute elementwise maximum; NaNs propagate (contra `fmax`)
`minimum`         Compute elementwise minimum; NaNs propagate (contra `fmin`)
`mod`             Alias of `remainder`.
`modf`            Separate numbers into fractional and integral parts.
`multiply`        Compute the product.
`negative`        Compute the negation.
`nextafter`       Return the next floating point number in a certain direction.
`not_equal`       Test if numbers are not equal.
`power`           Compute exponentiation.
`rad2deg`         Convert radians to degrees.
`radians`         Alias for `deg2rad`.
`reciprocal`      Compute the reciprocal; undesirable behavior for integers.
`remainder`       Compute remainder of division; inverse sign semantics from `fmod`.
`repvals`         Get “representative” scalar values from uncertain arrays.
`right_shift`     Shift bits right; integer types only.
`rint`            Round values to integers.
`shape`           Get an array’s shape.
`sign`            Compute the signum indicator.
`signbit`         Return true where input is less than zero.
`sin`             Compute sine.
`sinh`            Compute hyperbolic sine.

`spacing`         Return distance between inputs and nearest adjacent numbers in their representation.
`sqrt`            Compute square root.
`square`          Compute square.
`subtract`        Compute difference.
`tan`             Compute tangent.
`tanh`            Compute hyperbolic tangent.
`true_divide`     Compute ratio, with consistent non-floor division semantics.
`trunc`           Compute truncated (rounded-towards-zero) values.
================  ===============

Reference documentation for these functions may be found at the end of this
page.


How it works
------------------------------------------------------------------------------

The functions defined above all work by investigating the types of their
arguments and delegating the actual work to a particular implementation of the
Common Interface that’s targeted at those types. Each of these implementations
is incarnated in an instance of the `MathFunctionLibrary` class, which has one
method for each function provided in the Common Interface defined above.

.. autoclass:: MathFunctionLibrary

.. automethod:: MathFunctionLibrary.accepts
.. automethod:: MathFunctionLibrary.generic_unary
.. automethod:: MathFunctionLibrary.generic_binary

The following function is key to the dispatch process: given some input
array-like object *x*, it determines which `MathFunctionLibrary` instance
should be used.

.. autofunction:: get_library_for

The Common Interface can be difficult to implement in full generality, since
the “ufunc” features provided by Numpy have some nontrivial semantics
regarding the optional *out* parameter, behavior with regard to scalars, and
so on. The following subclass of `MathFunctionLibrary` takes care of these
generic challenges to make things easier on implementors.

.. autoclass:: TidiedFunctionLibrary
.. automethod:: TidiedFunctionLibrary.generic_tidy_unary
.. automethod:: TidiedFunctionLibrary.generic_tidy_binary
.. automethod:: TidiedFunctionLibrary.coerce
.. automethod:: TidiedFunctionLibrary.make_output_array


The Numpy function library
------------------------------------------------------------------------------

An instance of the class is used for execute all math operations on array-like
objects that stock Numpy can handle. Its implementations of “ufunc” functions
like ``add`` and ``square`` delegate directly to their Numpy equivalents.
However, it still needs to implement the methods of the Common Interface that
aren't in Numpy.

.. autoclass:: NumpyFunctionLibrary
.. automethod:: NumpyFunctionLibrary.cmask
.. automethod:: NumpyFunctionLibrary.repvals


Using :mod:`pwkit.mathlib` to implement standard math operators
------------------------------------------------------------------------------

You can use a call such as ``pwkit.mathlib.add(x, y)`` function to add
together two objects of nearly any type, but it is much nicer to be able to
write just ``x + y``. By inheriting from the following class, you can cause an
object to implement all of its operators in such a way that they delegate to
the appropriate `pwkit.mathlib` functions.

.. autoclass:: MathlibDelegatingObject


Individual function reference
------------------------------------------------------------------------------

.. These are all manually documented since their implementations are
   automatically generated within the module. Bummer.

.. function:: absolute(x, out=None)

   Compute absolute value.

.. function:: add(x, y, out=None)

   Compute sum.

.. function:: arccos(x, out=None)

   Compute inverse cosine.

.. function:: arccosh(x, out=None)

   Compute inverse hyperbolic cosine.

.. function:: arcsin(x, out=None)

   Compute inverse sine.

.. function:: arcsinh(x, out=None)

   Compute inverse hyperbolic sine.

.. function:: arctan(x, out=None)

   Compute inverse tangent.

.. function:: arctan2(x, y, out=None)

   Compute inverse tangent with separate numerator and denominator.

.. function:: arctanh(x, out=None)

   Compute inverse hyperbolic tangent.

.. function:: asscalar(x)

   Convert an array to a scalar, assuming it has exactly 1 value.

.. function:: atleast_1d(x)

   Return a view of an array that is at least 1-dimensional.

.. function:: bitwise_and(x, y, out=None)

   Compute bitwise AND; integer types only.

.. function:: bitwise_or(x, y, out=None)

   Compute bitwise OR; integer types only.

.. function:: bitwise_xor(x, y, out=None)

   Compute bitwise XOR; integer types only.

.. function:: cbrt(x, out=None)

   Compute cube root.

.. function:: ceil(x, out=None)

   Compute ceiling.

.. function:: cmask(x, out=None)

   Return a mask based on array contents.

.. function:: conjugate(x, out=None)

   Compute complex conjugates.

.. function:: copysign(x, y, out=None)

   Copy sign bits from one set of numbers to another.

.. function:: cos(x, out=None)

   Compute cosine.

.. function:: cosh(x, out=None)

   Compute hyperbolic cosine.

.. function:: deg2rad(x, out=None)

   Convert degrees to radians.

.. function:: divide(x, y, out=None)

   Compute ratio, with floor-divide semantics on integers in Python 2.

.. function:: dtype(x)

   Get the data type of array elements.

.. function:: equal(x, y, out=None)

   Test if values are equal.

.. function:: exp(x, out=None)

   Compute the base-e exponential function.

.. function:: exp2(x, out=None)

   Compute the base-2 exponential function.

.. function:: expm1(x, out=None)

   Compute exp(x) - 1.

.. function:: fabs(x, out=None)

   Compute absolute value; non-complex types only.

.. function:: floor(x, out=None)

   Compute floor;

.. function:: floor_divide(x, y, out=None)

   Compute ratio, with consistent floor-divide semantics.

.. function:: fmax(x, y, out=None)

   Element-wise maximum; NaNs are ignored when possible (contra `maximum`).

.. function:: fmin(x, y, out=None)

   Element-wise minimum; NaNs are ignored when possible (contra `minimum`).

.. function:: fmod(x, y, out=None)

   Compute remainder of division; inverse sign semantics from `remainder`.

.. function:: frexp(x, out1=None, out2=None)

   Decompose values into mantissa and twos exponent; inverse of `ldexp`.

.. function:: greater(x, y, out=None)

   Test if values are greater than others.

.. function:: greater_equal(x, y, out=None)

   Test if values are greater than or equal to others.

.. function:: hypot(x, y, out=None)

   Compute ``sqrt(x**2 + y**2)``.

.. function:: invert(x, out=None)

   Compute bitwise negation; integer types only.

.. function:: is_scalar(x, out=None)

   Test if the argument array is scalar-compatible.

.. function:: isfinite(x, out=None)

   Test if values are finite.

.. function:: isinf(x, out=None)

   Test if values are infinite.

.. function:: isnan(x, out=None)

   Test if values are NaN.

.. function:: ldexp(x, y, out=None)

   Compute ``x * 2**y``; inverse of `frexp`

.. function:: left_shift(x, y, out=None)

   Shift bits left; integer types only.

.. function:: less(x, y, out=None)

   Test if values are less than others.

.. function:: less_equal(x, y, out=None)

   Test if values are less than or equal to others.

.. function:: log(x, out=None)

   Compute base-e logarithm.

.. function:: log10(x, out=None)

   Compute base-10 logarithm.

.. function:: log1p(x, out=None)

   Compute ``log(1 + x)``.

.. function:: log2(x, out=None)

   Compute base-2 logarithm.

.. function:: logaddexp(x, y, out=None)

   Compute ``log(exp(x) + exp(y))``.

.. function:: logaddexp2(x, y, out=None)

   Compute ``log2(2**x + 2**y)``.

.. function:: logical_and(x, y, out=None)

   Compute the logical AND.

.. function:: logical_or(x, y, out=None)

   Compute the logical OR.

.. function:: logical_not(x, out=None)

   Compute the logical NOT.

.. function:: logical_xor(x, y, out=None)

   Compute the logical XOR.

.. function:: maximum(x, y, out=None)

   Compute elementwise maximum; NaNs propagate (contra `fmax`)

.. function:: minimum(x, y, out=None)

   Compute elementwise minimum; NaNs propagate (contra `fmin`)

.. function:: modf(x, out1=None, out2=None)

   Separate numbers into fractional and integral parts.

.. function:: multiply(x, y, out=None)

   Compute the product.

.. function:: negative(x, out=None)

   Compute the negation.

.. function:: nextafter(x, y, out=None)

   Return the next floating point number in a certain direction.

.. function:: not_equal(x, y, out=None)

   Test if numbers are not equal.

.. function:: power(x, y, out=None)

   Compute exponentiation.

.. function:: rad2deg(x, out=None)

   Convert radians to degrees.

.. function:: reciprocal(x, out=None)

   Compute the reciprocal; undesirable behavior for integers.

.. function:: remainder(x, y, out=None)

   Compute remainder of division; inverse sign semantics from `fmod`.

.. function:: repvals(x, out=None)

   Get “representative” scalar values from uncertain arrays.

.. function:: right_shift(x, y, out=None)

   Shift bits right; integer types only.

.. function:: rint(x, out=None)

   Round values to integers.

.. function:: shape(x)

   Get an array’s shape.

.. function:: sign(x, out=None)

   Compute the signum indicator.

.. function:: signbit(x, out=None)

   Return true where input is less than zero.

.. function:: sin(x, out=None)

   Compute sine.

.. function:: sinh(x, out=None)

   Compute hyperbolic sine.

.. function:: spacing(x, out=None)

   Return distance between inputs and nearest adjacent numbers in their representation.

.. function:: sqrt(x, out=None)

   Compute square root.

.. function:: square(x, out=None)

   Compute square.

.. function:: subtract(x, y, out=None)

   Compute difference.

.. function:: tan(x, out=None)

   Compute tangent.

.. function:: tanh(x, out=None)

   Compute hyperbolic tangent.

.. function:: true_divide(x, y, out=None)

   Compute ratio, with consistent non-floor division semantics.

.. function:: trunc(x, out=None)

   Compute truncated (rounded-towards-zero) values.
