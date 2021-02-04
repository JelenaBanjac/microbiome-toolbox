import dash

import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import base64
import datetime
import io
from flask_caching import Cache
import pandas as pd
import uuid
import time
import os


app_dir = os.getcwd()
filecache_dir = os.path.join(app_dir, 'cached_files')

app = dash.Dash(suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.COSMO])
#app.config.suppress_callback_exceptions = True


cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    #'CACHE_TYPE': 'filesystem',
    #'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})