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
import dash_html_components as dhc
import pathlib


app_dir = os.getcwd()
UPLOAD_FOLDER_ROOT = os.path.join(app_dir, 'cached_files')
pathlib.Path(UPLOAD_FOLDER_ROOT).mkdir(parents=True, exist_ok=True)

app = dash.Dash(suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.COSMO])
#app.config.suppress_callback_exceptions = True
du.configure_upload(app, UPLOAD_FOLDER_ROOT)
server = app.server

cache = Cache(server, config={
    'CACHE_TYPE': 'simple',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    #'CACHE_TYPE': 'filesystem',
    #'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})


# image_directory =  os.getcwd() + '/img/'
# image_filename = '/home/jelena/Desktop/microbiome-toolbox/images/loading.gif' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())
# dhc.Img(src="https://www.arcadiacars.com/static/media/loading.a74b50f6.gif")
loading_img = dhc.Div([
    dhc.Img(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/images/Ripple.gif")
])
