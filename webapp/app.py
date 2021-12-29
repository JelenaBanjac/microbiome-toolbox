import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dash.long_callback import DiskcacheLongCallbackManager
from uuid import uuid4
import flask
import diskcache
import dash_uploader as du
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

from layout.layout import layout
from environment.settings import UPLOAD_FOLDER_ROOT

server = flask.Flask(__name__)  # define flask app.server
launch_uid = uuid4()
cache_callback = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(
    cache_callback,
    cache_by=[lambda: launch_uid],
    expire=60,
)
app = DashProxy(
    __name__,
    # server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.COSMO],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    long_callback_manager=long_callback_manager,
    transforms=[MultiplexerTransform()]
)

du.configure_upload(app, UPLOAD_FOLDER_ROOT)

cache = Cache(
    app.server,
    config={
        # "CACHE_TYPE": "simple",
        # Note that filesystem cache doesn't work on systems with ephemeral
        # filesystems like Heroku.
        # "CACHE_TYPE": "filesystem",
        # "CACHE_DIR": "cache-directory",
        # should be equal to maximum number of users on the app at a single time
        # higher numbers will store more data in the filesystem / redis cache
        "CACHE_THRESHOLD": 200,
    },
)
# print(layout)
app.layout = layout
server = app.server
