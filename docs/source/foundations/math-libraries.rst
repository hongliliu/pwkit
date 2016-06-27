.. Copyright 2016 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

A Common Interface for math on arrays (:mod:`pwkit.mathlib`)
==============================================================================

.. automodule:: pwkit.mathlib
   :synopsis: Vectorized math functions that work on any array-like type.

.. currentmodule:: pwkit.mathlib


The “Common Interface”
------------------------------------------------------------------------------

The `pwkit.mathlib` module provides a set of free functions mirroring many of
those in the :mod:`numpy` module: the “Common Interface”. Unlike the functions
in Numpy, however, the ones in this module will work regardless of whether the
input is a Numpy array-like, :class:`pwkit.msmt` measurement type, table
column, etc. When a function shares the name of a Numpy function, the aim is
that it has precisely the same semantics. Some of the functions below are
unique to ``pkwit``, though.

================  ===============
  Name              Description
================  ===============
`absolute`        Compute absolute value.
`add`             Compute sum.
`append`          Concatenate two arrays.
`arccos`          Compute inverse cosine.
`arccosh`         Compute inverse hyperbolic cosine.
`arcsin`          Compute inverse sine.
`arcsinh`         Compute inverse hyperbolic sine.
`arctan`          Compute inverse tangent.
`arctan2`         Compute inverse tangent with separate numerator and denominator.
`arctanh`         Compute inverse hyperbolic tangent.
`bitwise_and`     Compute bitwise AND; integer types only.
`bitwise_or`      Compute bitwise OR; integer types only.
`bitwise_xor`     Compute bitwise XOR (“exclusive or”); integer types only.
`broadcast_to`    Return a view of the input with a new array shape.
`cbrt`            Compute cube root.
`ceil`            Compute ceiling.
`cmask`           Return a mask based on array contents.
`conjugate`       Compute complex conjugates.
`copysign`        Copy sign bits from one set of numbers to another.
`cos`             Compute cosine.
`cosh`            Compute hyperbolic cosine.
`deg2rad`         Convert degrees to radians.
`divide`          Compute ratio, with floor-divide semantics on integers in Python 2.
`get_dtype`       Get the data type of array elements.
`get_size`        Get the number of array elements.
`empty_like`      Return a new, uninitialized array like an existing one.
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
`logical_xor`     Compute the logical XOR (“exclusive or”).
`maximum`         Compute elementwise maximum; NaNs propagate (contra `fmax`)
`minimum`         Compute elementwise minimum; NaNs propagate (contra `fmin`)
`modf`            Separate numbers into fractional and integral parts.
`multiply`        Compute the product.
`negative`        Compute the negation.
`nextafter`       Return the next floating point number in a certain direction.
`not_equal`       Test if numbers are not equal.
`power`           Compute exponentiation.
`rad2deg`         Convert radians to degrees.
`reciprocal`      Compute the reciprocal; undesirable behavior for integers.
`remainder`       Compute remainder of division; inverse sign semantics from `fmod`.
`repvals`         Get “representative” scalar values from uncertain arrays.
`reshape`         Return a read-write view of an array with a new shape.
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

There are also a few aliases provided by Numpy that are emulated here:

================  =================
  Alias Name        Equivalent To
================  =================
`abs`             `absolute`
`bitwise_not`     `invert`
`conj`            `conjugate`
`degrees`         `rad2deg`
`mod`             `remainder`
`radians`         `deg2rad`
================  =================


How it works
------------------------------------------------------------------------------

The functions defined above all work by investigating the types of their
arguments and delegating the actual work to a particular implementation of the
Common Interface that’s targeted at those types. Each of these implementations
is incarnated in an instance of the `MathFunctionLibrary` class, which has one
method for each function provided in the Common Interface defined above.

An instance of a specialized `NumpyFunctionLibrary` class is used to execute
all math operations on array-like objects that stock Numpy can handle.

.. autoclass:: NumpyFunctionLibrary


Implementing math for your custom class
------------------------------------------------------------------------------

If you’re developing a custom array-like class that you want to do math on,
you should wire it in to the `~pwkit.mathlib` framework. This provides two key
benefits:

- The free functions in this library will automatically learn how to operate
  on your objects. This means that if you write code using the
  `~pwkit.mathlib` functions, it will automatically work on nearly *any* kind
  of array-like object.
- You can use the `MathlibDelegatingObject` mixin to instantly make your
  object correctly overload all of the standard Python operators.

“Wiring a class into the `~pwkit.mathlib` framework” requires only a small
number of conceptually simple steps:

- Write a `MathFunctionLibrary` subclass that actually implements the math.
- Set an attribute ``_pk_mathlib_library_`` on your class that points to a
  (singleton) instance of your new subclass.
- Optionally, inherit your class from the `MathlibDelegatingObject` mixin to
  overload all of Python's math operators.

In practice, your math library subclass should almost certainly inherit from
the `TidiedFunctionLibrary` class, which makes it so that you don’t have to
deal with a lot of the trickier aspects of the Numpy “ufunc” semantics.


Math function library specifics
------------------------------------------------------------------------------

.. autoclass:: MathFunctionLibrary

To implement a given math operation, simply implement a method on your
subclass having the same name and signature as the operations listed below;
although in many cases, your life will be easier if you let
`TidiedFunctionLibrary` smooth out the ufunc semantics for you.

There are also a few methods of `MathFunctionLibrary` that do not map to
Common Interface functions, but are needed to glue the system together:

.. automethod:: MathFunctionLibrary.accepts
.. automethod:: MathFunctionLibrary.new_empty
.. automethod:: MathFunctionLibrary.typeconvert

Finally:

.. autoclass:: TidiedFunctionLibrary



`MathlibDelegatingObject`
------------------------------------------------------------------------------

You can use a call such as ``pwkit.mathlib.add(x, y)`` function to add
together two objects of nearly any type, but it is much nicer to be able to
write just ``x + y``. By inheriting from the following class, you can cause an
object to implement all of its operators in such a way that they delegate to
the appropriate `pwkit.mathlib` functions.

.. autoclass:: MathlibDelegatingObject


Individual function reference for the Common Interface
------------------------------------------------------------------------------

.. These are (almost) all manually documented since their implementations are
   (almost all) automatically generated within the module. Bummer.

.. function:: absolute(x, out=None)

   Compute absolute value.

.. function:: add(x, y, out=None)

   Compute sum.

.. function:: append(x, y)

   Concatenate two arrays. The implementation of the Common Interface that is
   used is determined solely from *x*; therefore, something like
   ``mathlib.append (np.ones (4), pd.Series([2]))`` will fail.

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

.. function:: bitwise_and(x, y, out=None)

   Compute bitwise AND; integer types only.

.. function:: bitwise_or(x, y, out=None)

   Compute bitwise OR; integer types only.

.. function:: bitwise_xor(x, y, out=None)

   Compute bitwise XOR (“exclusive or”); integer types only.

.. function:: broadcast_to(x, shape)

   Return a *read-only view* of the input array *x* that has the specified
   shape. Raises `ValueError` if this is not possible. Only zero-dimensional
   arrays may be broadcast to a *shape* of ``()``.

.. function:: cbrt(x, out=None)

   Compute cube root.

.. function:: ceil(x, out=None)

   Compute ceiling.

.. function:: cmask(x, **kwargs)

   Return a mask based on array contents. The type of masking done is the
   logical “AND” of different conditions signified by boolean keyword
   arguments. The allowed keywords vary depending on the precise array type
   being used, but standardized ones are:

   ``welldefined``
     True for array elements that are not NaN.
   ``finite``
     True for array elements that are not positive or negative infinity.

.. function:: conjugate(x, out=None)

   Compute complex conjugates.

.. function:: copysign(x, y, out=None)

   Apply the sign bits of *y* to the values of *x*.

.. function:: cos(x, out=None)

   Compute cosine.

.. function:: cosh(x, out=None)

   Compute hyperbolic cosine.

.. function:: deg2rad(x, out=None)

   Convert degrees to radians.

.. function:: divide(x, y, out=None)

   Compute ratio, with floor-divide semantics on integers in Python 2.

.. function:: get_dtype(x)

   Get the data type of array elements.

.. function:: get_size(x)

   Get the number of array elements. Returns 1 for both scalars and
   zero-dimensional arrays. Returns 0 for arrays with a zero-size axis.

.. autofunction:: empty_like

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

   Compute floor.

.. function:: floor_divide(x, y, out=None)

   Compute ratio, with consistent floor-divide semantics.

.. function:: fmax(x, y, out=None)

   Element-wise maximum; NaNs are ignored when possible (contra `maximum`).

.. function:: fmin(x, y, out=None)

   Element-wise minimum; NaNs are ignored when possible (contra `minimum`).

.. function:: fmod(x, y, out=None)

   Compute remainder of division. The remainder has the same sign as the
   dividend *x*. These are the same sign semantics as the C library function
   of the same name and the MatLab ``rem`` function. These are the *opposite*
   sign semantics of the `remainder` function and the standard Python modulus
   operator ``x % y``.

.. function:: frexp(x, out1=None, out2=None)

   Decompose values into mantissa and twos exponent; inverse of `ldexp`. The
   mantissa data type is float; the exponent data type is int.

.. function:: greater(x, y, out=None)

   Test if values are greater than others.

.. function:: greater_equal(x, y, out=None)

   Test if values are greater than or equal to others.

.. function:: hypot(x, y, out=None)

   Compute ``sqrt(x**2 + y**2)``.

.. function:: invert(x, out=None)

   Compute bitwise negation; integer types only.

.. function:: isfinite(x, out=None)

   Test if values are finite.

.. function:: isinf(x, out=None)

   Test if values are infinite.

.. function:: isnan(x, out=None)

   Test if values are NaN.

.. function:: ldexp(x, y, out=None)

   Compute ``x * 2**y``; inverse of `frexp`. *y* must be integer; *x* may not
   be complex.

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

   Compute the logical XOR (“exclusive or”).

.. function:: maximum(x, y, out=None)

   Compute elementwise maximum; NaNs propagate (contra `fmax`)

.. function:: minimum(x, y, out=None)

   Compute elementwise minimum; NaNs propagate (contra `fmin`)

.. function:: modf(x, out1=None, out2=None)

   Separate numbers into fractional and integral parts. If a given input is
   negative, both the fractional and integral parts are negative. Note that
   both output arrays are returned with floating types, although the second
   output consists of values rounded to integers.

.. function:: multiply(x, y, out=None)

   Compute the product.

.. function:: negative(x, out=None)

   Compute the negation.

.. function:: nextafter(x, y, out=None)

   Return the next floating point value “after” *x*, in the sense that it is
   towards *y*.

.. function:: not_equal(x, y, out=None)

   Test if numbers are not equal.

.. function:: power(x, y, out=None)

   Compute exponentiation.

.. function:: rad2deg(x, out=None)

   Convert radians to degrees.

.. function:: reciprocal(x, out=None)

   Compute the reciprocal, ``1 / x``. The output array has the same type as
   the input, which means that this function almost surely has undesirable
   behavior for integers: any input besides 1 becomes 0.

.. function:: remainder(x, y, out=None)

   Compute remainder of division: ``x - floor (x / y) * y``. The result has
   the same sign as *y*. This has the same sign semantics as the Python
   modulus operator ``x % y``. These are the *opposite* sign semantics as the
   `fmod` function, the C library `fmod` function, and the MatLab ``rem``
   function.

.. function:: repvals(x, **kwargs)

   Get “representative” scalar values from uncertain arrays.

.. function:: reshape(x, shape)

   Return a *read-write view* of *x* having new shape *shape*. The total
   number of elements in the array must be the same, but the shape changes.
   Zero-dimensional arrays may be interconverted with other array shapes, so
   long as they have just one element.

.. function:: right_shift(x, y, out=None)

   Shift bits right; integer types only.

.. function:: rint(x, out=None)

   Round values to integers. Note that the output array is still of
   floating-point type if the input is.

.. function:: shape(x)

   Get an array’s shape. Returns ``()`` for scalars, which means that from the
   standpoint of this function they are indistinguishable from
   zero-dimensional arrays — which is not always the case in other situations.

.. function:: sign(x, out=None)

   Compute the signum indicator: -1 for inputs less than zero, 0 for inputs
   equal to zero, and +1 for inputs greater than zero. Note that the output
   data type is the same as the input data type.

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
