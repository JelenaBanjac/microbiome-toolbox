import dash

import dash_bootstrap_components as dbc
import base64
import datetime
import io
from flask_caching import Cache
import pandas as pd
import uuid
import time
import os
import dash_uploader as du


app_dir = os.getcwd()
UPLOAD_FOLDER_ROOT = os.path.join(app_dir, 'cached_files')

app = dash.Dash(suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.COSMO])
#app.config.suppress_callback_exceptions = True
du.configure_upload(app, UPLOAD_FOLDER_ROOT)


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