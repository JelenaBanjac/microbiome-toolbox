from typing import Container
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
import dash_html_components as dhc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import base64
import datetime
import io
from flask_caching import Cache
import pandas as pd
import uuid
import time
import os
import dash_uploader as du
from pathlib import Path

#from app import app, UPLOAD_FOLDER_ROOT, cache
from pages import page1, page2, page3, page4, page5, page6

# this example that adds a logo to the navbar brand
navbar = dbc.Navbar(
            dbc.Container(
                [
                    dhc.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                #dbc.Col(dhc.Img(src=PLOTLY_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Microbiome Toolbox", className="ml-2")),
                            ],
                            align="center",
                            no_gutters=True,
                        ),
                        href="#",
                    ),
                ]
            ),
            color="dark",
            dark=True,
            className="mb-5",
        )


card1 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Reference Definition & Statistics", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    #dhc.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/methods/page-1"),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card1-btn"), href='/methods/page-1'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card2 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Data Analysis & Exploration", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card2-btn"), href='/methods/page-2'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card3 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Microbiome Trajectory", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card3-btn"), href='/methods/page-3'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card4 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Bacteria Importance with Time", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card4-btn"), href='/methods/page-4'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card5 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Longitudinal Anomaly Detection", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card5-btn"), href='/methods/page-5'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card6 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Intervention Simulation", className="card-title"),
                    dhc.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card6-btn"), href='/methods/page-6'),
                ]
            ),
        ],
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)


def parse_dataset(filename):
    #content_type, content_string = content.split(',')

    #decoded = base64.b64decode(content_string)
    df = None
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = pd.read_csv(filename)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            #df = pd.read_excel(io.BytesIO(decoded))
            #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep="\t")
            df = pd.read_csv(filename, sep="\t")
    except Exception as e:
        print(e)
        return None
    print('\nFinished parsing df parse_dataset')

    return df


def write_dataframe(session_id, df):
    '''
    Write dataframe to disk, for now just as CSV
    For now do not preserve or distinguish filename;
    user has one file at once.
    '''
    print('\nCalling write_dataframe')
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.pickle")
    df.to_pickle(filename)

# cache memoize this and add timestamp as input!
@cache.memoize()
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    print('\nCalling read_dataframe')
    print('\tsession_id', session_id)
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.pickle")
    if os.path.exists(filename):
        print('\tfilename', filename)
        df = pd.read_pickle(filename)
        print('** Reading data from disk **')
    else:
        print('\tfilename not yet exists', filename)
        df = None
        print('** No data, df empty **')

    return df


def main_layout_(session_id, upload_filename):
    if not upload_filename:
        upload_filename_alert = dhc.Div([])
    else:
        upload_filename_alert = dhc.Div(dbc.Alert(f"Currently loaded file: {upload_filename}", color="info"))

    print("\nMain layout function called only with /methods")
    return dhc.Div(id="main",
                 children=[
                   dhc.Div(id='main-upload', children=[
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    dhc.H3("Upload Dataset", style={'textAlign': 'center',}),
                                    dhc.Br(),
                                    du.Upload(
                                        id='upload-data',
                                        filetypes=['csv', 'xls'],
                                        upload_id=session_id,
                                    ),
                                    dhc.Div(id="upload-infobox", children=upload_filename_alert),
                                    ]),
                                ]
                            ),
                        ], className="md-12")
                    ]),
                    dhc.Br(),
                   

                    dhc.Div(id="main-methods", children=[
                            #dcc.Location(id='url', refresh=False),
                        
                            dbc.Container([
                                dbc.Row(
                                    dbc.Col([
                                        dhc.Br(),
                                        dhc.Div(dhc.H3("Methods")),
                                        dhc.Br(),
                                        ], 
                                    className="md-12"),
                                ),
                                dbc.Row([card1, card2, card3]),
                                dbc.Row([card4, card5, card6])
                            ],
                            className="md-4",
                            )
                        ],
                        style={
                            'verticalAlign':'middle',
                            'textAlign': 'center',
                            'backgroundColor': 'rgb(255, 255, 255)', #'rgb(245, 245, 245)',
                            'position':'relative',
                            'width':'100%',
                            #'height':'100vh',
                            'bottom':'0px',
                            'left':'0px',
                            'zIndex':'1000',
                        }
                )
        ])

def layout_(session_id, upload_info):
    print("\nLayout function called from server_layout init or can be separately called")
    return dhc.Div([
        navbar,
        dcc.Location(id='url', refresh=False),
        dhc.Div(session_id, id='session-id', style={'display': 'none'}),
        dhc.Div(upload_info, id='upload-filename', style={'display': 'none'}),

        main_layout_(session_id, upload_info)
    ])

def serve_layout():
    session_id = str(uuid.uuid4())
    print("\nServe layout", session_id)

    return layout_(session_id, '')

app.layout = serve_layout



# Update the index
@app.callback(Output('main', 'children'),
              [Input('url', 'pathname')],
              [State('session-id', 'children'),
              State('upload-filename', 'children')])
def display_page(pathname, session_id, upload_filename):
    print("\nPathname", pathname, "session_id", session_id)
    if pathname == '/methods/page-1':
        return page1.layout
    elif pathname == '/methods/page-2':
        return page2.layout
    elif pathname == '/methods/page-3':
        return page3.layout
    elif pathname == '/methods/page-4':
        return page4.layout
    elif pathname == '/methods/page-5':
        return page5.layout
    elif pathname == '/methods/page-6':
        return page6.layout
    else: 
        print("\tohter path....")
        return main_layout_(session_id, upload_filename)  #app.layout  #main_layout_(None)


@app.callback(
    [Output('upload-filename', 'children'),
    Output('upload-infobox', 'children'),
    Output('card1-btn', 'disabled'),
    Output('card2-btn', 'disabled'),
    Output('card3-btn', 'disabled'),
    Output('card4-btn', 'disabled'),
    Output('card5-btn', 'disabled'),
    Output('card6-btn', 'disabled')],
    [Input('upload-data', 'isCompleted')],
    [State('upload-data', 'fileNames'),
     State('upload-data', 'upload_id'),
     State('upload-filename', 'children')])
def return_methods(iscompleted, filenames, upload_id, filename_latest):
    print("Upload callback called")
    
    upload_infobox = dhc.Div([])
    methods_disabled = True

    if filenames is not None:
        filename = filenames[0]
    elif filename_latest != '':
        filename = filename_latest
        upload_infobox = dhc.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))
        methods_disabled = False
    else:
        filename = ''
    
    
    # at the initialization of the page or when back
    if not iscompleted:
        return filename, upload_infobox, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled

    df = None
    if filenames is not None:
        if upload_id:
            root_folder = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)
        else:
            root_folder = UPLOAD_FOLDER_ROOT

        file = os.path.join(root_folder, filename)

        df = parse_dataset(file)

        write_dataframe(upload_id, df)
        upload_infobox = dhc.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))

        methods_disabled = False

    if df is None:
        upload_infobox = dhc.Div(dbc.Alert("There was an error processing this file!", color="danger"))
        methods_disabled = True

    print("filename", filename)
    print("upload_infobox", upload_infobox)
    return filename, upload_infobox, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled



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


# image_directory =  os.getcwd() + '/img/'
# image_filename = '/home/jelena/Desktop/microbiome-toolbox/images/loading.gif' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())
loading_img = dhc.Div([
    dhc.Img(src="https://www.arcadiacars.com/static/media/loading.a74b50f6.gif")
])

if __name__ == '__main__':
    app.run_server(debug=True)


