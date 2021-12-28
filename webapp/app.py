import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dash.long_callback import DiskcacheLongCallbackManager
from uuid import uuid4
import flask
import diskcache

from layout.layout import layout

server = flask.Flask(__name__)  # define flask app.server
launch_uid = uuid4()
cache_callback = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(
    cache_callback,
    cache_by=[lambda: launch_uid],
    expire=60,
)
app = dash.Dash(
    __name__,
    # server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.COSMO],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    long_callback_manager=long_callback_manager,
)

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "simple",
        # Note that filesystem cache doesn't work on systems with ephemeral
        # filesystems like Heroku.
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
        # should be equal to maximum number of users on the app at a single time
        # higher numbers will store more data in the filesystem / redis cache
        "CACHE_THRESHOLD": 200,
    },
)
# print(layout)
app.layout = layout
server = app.server
