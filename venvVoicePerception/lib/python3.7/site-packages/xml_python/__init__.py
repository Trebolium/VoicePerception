"""Provides the WorldBuilder class, as well as the subclass and parser
decorators."""

import sys

from inspect import isgenerator, isclass
from xml.etree.ElementTree import fromstring

from attr import attrs, attrib, Factory


class NoParent:
    def __str__(self):
        return 'Top-level node'

    def __repr__(self):
        return str(self)


no_parent = NoParent()


class XMLPythonException(Exception):
    """Base exception."""


class NoSuchParser(XMLPythonException):
    """No such parser has been defined."""


class InvalidParent(XMLPythonException):
    """An invalid parent was provided."""


class InvalidValidParent(XMLPythonException):
    """The valid_parent argument is invalid."""


@attrs
class Parser:
    """A parser which contains a function and any valid parent."""

    name = attrib()
    valid_parent = attrib()
    func = attrib()

    def __call__(self, parent, text, **kwargs):
        """Call self.func if parent is valid."""
        if isclass(self.valid_parent) or isinstance(self.valid_parent, tuple):
            vp = isinstance(parent, self.valid_parent)
        elif callable(self.valid_parent):
            vp = self.valid_parent(parent)
        else:
            vp = True
        if vp:
            return self.func(parent, text, **kwargs)
        raise InvalidParent(self, parent)


@attrs
class Builder:
    """Subclass to create your own builder, or juse instantiate and use the
    parser decorator."""

    parsers = attrib(default=Factory(dict), init=False)

    def __attrs_post_init__(self):
        """Set some defaults."""
        for name in dir(self):
            func = getattr(self, name)
            if hasattr(func, '__parser_args__'):
                args, kwargs = func.__parser_args__
                self.parser(*args, **kwargs)(func)

    def parser(self, name, valid_parent=None):
        """Decorate a function to use as a parser.

        The function must be prepared to take at least two arguments:

        A "parent" object, which is the returned value from the parent parser,
        or no_parent if this is the first parser to be called.

        Any non-XML text from the current node.

        Any attributes of the node will be used as keyword arguments.

        If valid_parent is a class, then it is checked that the parent is an
        instance of that class. If valid_parent is a function, then that
        function will be called to determine if the parent is valid."""

        def inner(func):
            self.parsers[name] = Parser(name, valid_parent, func)
            return func

        if valid_parent is None or isclass(valid_parent) or callable(
            valid_parent
        ) or isinstance(valid_parent, tuple):
            return inner
        raise InvalidValidParent(valid_parent)

    def from_args(self, default):
        """Passes either sys.argv[1], or default to self.from_filename."""
        try:
            default = sys.argv[1]
        except IndexError:
            pass  # default will simply stay the same.
        return self.from_filename(default)

    def from_filename(self, filename):
        """Opens the given filename, and passes the file object onto
        self.from_file."""
        with open(filename, 'r') as f:
            return self.from_file(f)

    def from_file(self, f):
        """Passes f.read() onto self.from_string."""
        return self.from_string(f.read())

    def from_string(self, string):
        """Given an XML string, returns an object. All tags must have a
        corrisonding parser, decorated with self.parser."""
        root = fromstring(string)
        return self.build(root, no_parent)

    def build(self, node, parent):
        """Given an XML node (such as those returned by
        xml.etree.ElementTree.fromstring), return an object."""
        name = node.tag
        text = node.text
        if name not in self.parsers:
            raise NoSuchParser(name)
        parser = self.parsers[name]
        kwargs = {
            name.replace('-', '_'): value for name, value in
            node.attrib.items()
        }
        ret = parser(parent, text, **kwargs)
        if isgenerator(ret):
            obj = next(ret)
        else:
            obj = ret
        for child in node:
            self.build(child, obj)
        if isgenerator(ret):
            try:
                next(ret)
                raise RuntimeError(
                    'Iterators can only yield once.\nOffending parser: %r' %
                    parser
                )
            except StopIteration:
                pass  # Good stuff.
        return obj


def parser(*args, **kwargs):
    """Decorate an instance method to have it automatically added as a parser.
    Takes the same arguments as the Builder.parser method."""

    def inner(func):
        func.__parser_args__ = (args, kwargs)
        return func

    return inner
