"""Provides the FlaskBuilder class."""

from functools import partial

from flask import Flask, render_template_string

from .. import Builder, NoParent, parser


class FlaskBuilder(Builder):
    """Used for building Flask apps."""

    @parser('app', valid_parent=NoParent)
    def get_app(
        self, parent, text, name=__name__, host='0.0.0.0', port='4000'
    ):
        """Returns a flask app."""
        app = Flask(name)
        app.config['HOST'] = host
        app.config['PORT'] = int(port)
        return app

    @parser('route', valid_parent=Flask)
    def get_route(self, app, text, path=None):
        """Decorates a function as a route."""
        if path is None:
            raise RuntimeError('A path must be supplied.')
        f = partial(render_template_string, text)
        f.__name__ = path.replace('/', '_')
        return app.route(path)(f)
