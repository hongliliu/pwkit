# This file combines code with varying copyrights and licenses -- see below.
# -*- mode: python; coding: utf-8 -*-

"""A collection of useful, syntactic-sugar-y functions, decorators, etc.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ('''
method_decorator
partialmethod
indexerproperty
''').split ()




# The 'method_decorator' class -- copied from the GitHub repository described
# below. A few modifications made: Override __call__ and possibly fixup.

'''
Python decorator that knows the class the decorated method is bound to.

Please see full description here:
https://github.com/denis-ryzhkov/method_decorator/blob/master/README.md

method_decorator version 0.1.3
Copyright (C) 2013 by Denis Ryzhkov <denisr@denisr.com>
MIT License, see http://opensource.org/licenses/MIT
'''

__all__ = str ('method_decorator').split ()

#### method_decorator

class method_decorator(object):

    def __init__(self, func, obj=None, cls=None, method_type='function'):
        # These defaults are OK for plain functions and will be changed by
        # __get__() for methods once a method is dot-referenced.
        self.func, self.obj, self.cls, self.method_type = func, obj, cls, method_type

    def fixup (self, newobj):
        pass

    def __get__(self, obj=None, cls=None):
        # It is executed when decorated func is referenced as a method:
        # cls.func or obj.func.

        if self.obj == obj and self.cls == cls:
            return self # Use the same instance that is already processed by
                        # previous call to this __get__().

        method_type = (
            'staticmethod' if isinstance(self.func, staticmethod) else
            'classmethod' if isinstance(self.func, classmethod) else
            'instancemethod'
            # No branch for plain function - correct method_type for it is
            # already set in __init__() defaults.
        )

        # Use specialized method_decorator (or descendant) instance, don't
        # change current instance attributes - it leads to conflicts.
        newobj = object.__getattribute__(self, '__class__')(
            # Use bound or unbound method with this underlying func.
            self.func.__get__(obj, cls), obj, cls, method_type)
        self.fixup (newobj)
        return newobj

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattribute__(self, attr_name): # Hiding traces of decoration.
        if attr_name in ('__init__', '__get__', '__call__', '__getattribute__',
                         'func', 'obj', 'cls', 'method_type', 'fixup'):
            # Our known names. '__class__' is not included because is used
            # only with explicit object.__getattribute__().
            return object.__getattribute__(self, attr_name) # Stopping recursion.

        # All other attr_names, including auto-defined by system in self, are
        # searched in decorated self.func, e.g.: __module__, __class__,
        # __name__, __doc__, im_*, func_*, etc. Raises correct AttributeError
        # if name is not found in decorated self.func.
        return getattr(self.func, attr_name)

    def __repr__(self):
        # Special case: __repr__ ignores __getattribute__.
        return self.func.__repr__()

#### test

def test():

    #### my_decorator

    class my_decorator(method_decorator):
        def __call__(self, *args, **kwargs):

            print('Calling {method_type} {method_name} from instance {instance} of class {class_name} from module {module_name} with args {args} and kwargs {kwargs}.'.format(
                method_type=self.method_type,
                method_name=self.__name__,
                instance=self.obj,
                class_name=(self.cls.__name__ if self.cls else None),
                module_name=self.__module__,
                args=args,
                kwargs=kwargs,
            ))

            return method_decorator.__call__(self, *args, **kwargs)

    #### MyClass

    class MyClass(object):

        @my_decorator
        def my_instance_method(self, arg, kwarg='default'):
            '''my_instance_method doc.'''
            return dict(arg=arg, kwarg=kwarg)

        @my_decorator
        @classmethod
        def my_class_method(cls, arg):
            return arg

        @my_decorator
        @staticmethod
        def my_static_method(arg):
            return arg

    my_class_module_name = MyClass.__module__
    my_instance = MyClass()

    #### my_plain_function

    @my_decorator
    def my_plain_function(arg):
        return arg

    #### instancemethod

    result = my_instance.my_instance_method
    assert result.method_type == 'instancemethod'
    assert result.__name__ == 'my_instance_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.__doc__ == 'my_instance_method doc.'
    assert result.im_self == my_instance
    assert result.im_class == MyClass
    assert repr(type(result.im_func)) == "<type 'function'>"
    assert result.func_defaults == ('default', )

    try:
        result.invalid
        assert False, 'Expected AttributeError'
    except AttributeError:
        pass

    result = my_instance.my_instance_method('bound', kwarg='kwarg')
    assert result == dict(arg='bound', kwarg='kwarg')

    result = my_instance.my_instance_method('bound')
    assert result == dict(arg='bound', kwarg='default')

    result = MyClass.my_instance_method
    assert result.method_type == 'instancemethod'
    assert result.__name__ == 'my_instance_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.__doc__ == 'my_instance_method doc.'
    assert result.im_self == None
    assert result.im_class == MyClass
    assert repr(type(result.im_func)) == "<type 'function'>"
    assert result.func_defaults == ('default', )

    result = MyClass.my_instance_method(MyClass(), 'unbound')
    assert result['arg'] == 'unbound'

    #### classmethod

    result = MyClass.my_class_method
    assert result.method_type == 'classmethod'
    assert result.__name__ == 'my_class_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.im_self == MyClass
    assert result.im_class == type
    assert repr(type(result.im_func)) == "<type 'function'>", type(result.im_func)

    result = MyClass.my_class_method('MyClass.my_class_method')
    assert result == 'MyClass.my_class_method'

    result = my_instance.my_class_method
    assert result.method_type == 'classmethod'
    assert result.__name__ == 'my_class_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert result.im_self == MyClass
    assert result.im_class == type
    assert repr(type(result.im_func)) == "<type 'function'>", type(result.im_func)

    result = my_instance.my_class_method('my_instance.my_class_method')
    assert result == 'my_instance.my_class_method'

    #### staticmethod

    result = MyClass.my_static_method
    assert result.method_type == 'staticmethod'
    assert result.__name__ == 'my_static_method'
    assert result.obj == None
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = MyClass.my_static_method('MyClass.my_static_method')
    assert result == 'MyClass.my_static_method'

    result = my_instance.my_static_method
    assert result.method_type == 'staticmethod'
    assert result.__name__ == 'my_static_method'
    assert result.obj == my_instance
    assert result.cls == MyClass
    assert result.cls.__name__ == 'MyClass'
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = my_instance.my_static_method('my_instance.my_static_method')
    assert result == 'my_instance.my_static_method'

    #### plain function

    result = my_plain_function
    assert result.method_type == 'function'
    assert result.__name__ == 'my_plain_function'
    assert result.obj == None
    assert result.cls == None
    assert result.__module__ == my_class_module_name
    assert not hasattr(result, 'im_self')
    assert not hasattr(result, 'im_class')
    assert not hasattr(result, 'im_func')

    result = my_plain_function('my_plain_function')
    assert result == 'my_plain_function'

    #### OK

    print('OK')

if __name__ == '__main__':
    test()




# The 'partialmethod' helper, backported from Python 3.5.0. The only change is
# one instance of 3.x-only ``cls_or_self, *args = args`` syntax. Copyright as
# described in the following comments; license is the Python license.

from functools import partial

# Python module wrapper for _functools C module
# to allow utilities written in Python to be added
# to the functools module.
# Written by Nick Coghlan <ncoghlan at gmail.com>,
# Raymond Hettinger <python at rcn.com>,
# and Łukasz Langa <lukasz at langa.pl>.
#   Copyright (C) 2006-2013 Python Software Foundation.
# See C source code for _functools credits/copyright

class partialmethod(object):
    """Method descriptor with partial application of the given arguments
    and keywords.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func, *args, **keywords):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError("{!r} is not callable or a descriptor"
                                 .format(func))

        # func could be a descriptor like classmethod which isn't callable,
        # so we can't inherit from partial (it verifies func is callable)
        if isinstance(func, partialmethod):
            # flattening is mandatory in order to place cls/self before all
            # other arguments
            # it's also more efficient since only one function will be called
            self.func = func.func
            self.args = func.args + args
            self.keywords = func.keywords.copy()
            self.keywords.update(keywords)
        else:
            self.func = func
            self.args = args
            self.keywords = keywords

    def __repr__(self):
        args = ", ".join(map(repr, self.args))
        keywords = ", ".join("{}={!r}".format(k, v)
                                 for k, v in self.keywords.items())
        format_string = "{module}.{cls}({func}, {args}, {keywords})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    func=self.func,
                                    args=args,
                                    keywords=keywords)

    def _make_unbound_method(self):
        def _method(*args, **keywords):
            call_keywords = self.keywords.copy()
            call_keywords.update(keywords)
            cls_or_self = args[0]
            rest = args[1:]
            call_args = (cls_or_self,) + self.args + tuple(rest)
            return self.func(*call_args, **call_keywords)
        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method._partialmethod = self
        return _method

    def __get__(self, obj, cls):
        get = getattr(self.func, "__get__", None)
        result = None
        if get is not None:
            new_func = get(obj, cls)
            if new_func is not self.func:
                # Assume __get__ returning something new indicates the
                # creation of an appropriate callable
                result = partial(new_func, *self.args, **self.keywords)
                try:
                    result.__self__ = new_func.__self__
                except AttributeError:
                    pass
        if result is None:
            # If the underlying descriptor didn't do anything, treat this
            # like an instance method
            result = self._make_unbound_method().__get__(obj, cls)
        return result

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)



# Code here and below is:
#
# Copyright 2015 Peter Williams and collaborators
# Licensed under the MIT License.

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
    """A decorator to make a property that is a named, virtual indexer."""

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
