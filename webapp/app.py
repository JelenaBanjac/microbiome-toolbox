import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

import flask

from layout.layout import layout

server = flask.Flask(__name__) # define flask app.server

app = dash.Dash(
    __name__,
    # server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.COSMO],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "simple",
        # Note that filesystem cache doesn't work on systems with ephemeral
        # filesystems like Heroku.
        #'CACHE_TYPE': 'filesystem',
        #'CACHE_DIR': 'cache-directory',
        # should be equal to maximum number of users on the app at a single time
        # higher numbers will store more data in the filesystem / redis cache
        "CACHE_THRESHOLD": 200,
    },
)
print(layout)
app.layout = layout
server = app.server
