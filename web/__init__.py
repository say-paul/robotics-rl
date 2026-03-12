"""
Flask application factory.

create_app() builds the Flask + WebSocket app, registering route
blueprints that close over the shared SimContext and FrameBuffer.
"""

import os

from flask import Flask
from flask_sock import Sock

from .routes import register_routes
from .stream import register_stream


def create_app(ctx, frames):
    """
    Build and return a configured Flask application.

    Parameters
    ----------
    ctx : SimContext
        Shared simulation state.
    frames : FrameBuffer
        Thread-safe JPEG frame exchange.
    """
    template_dir = os.path.join(os.path.dirname(__file__), os.pardir, "templates")
    app = Flask(__name__, template_folder=os.path.abspath(template_dir))
    sock = Sock(app)

    register_routes(app, ctx)
    register_stream(app, sock, ctx, frames)

    return app
