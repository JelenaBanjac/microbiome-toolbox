from typing import Container
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
import dash_html_components as html
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

from app import app, UPLOAD_FOLDER_ROOT, cache
from pages import page1, page2

# this example that adds a logo to the navbar brand
navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                #dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("ELMToolbox", className="ml-2")),
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
                    html.H4("Data Analysis & Exploration", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    #html.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/methods/page-1"),
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
                    html.H4("Healthy Reference", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("See more", outline=True, color="dark"),
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
                    html.H4("Differential Ranking", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("See more", outline=True, color="dark"),
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
                    html.H4("Microbiome Trajectory", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("See more", outline=True, color="dark"),
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
                    html.H4("Bacteria Importance with Time", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("See more", outline=True, color="dark"),
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
                    html.H4("Longitudinal Anomaly Detection", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("See more", outline=True, color="dark"),
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
        upload_filename_alert = html.Div([])
    else:
        upload_filename_alert = html.Div(dbc.Alert(f"Currently loaded file: {upload_filename}", color="info"))

    print("\nMain layout function called only with /methods")
    return html.Div(id="main",
                 children=[
                   html.Div(id='main-upload', children=[
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    html.H3("Upload Dataset", style={'textAlign': 'center',}),
                                    html.Br(),
                                    du.Upload(
                                        id='upload-data',
                                        filetypes=['csv', 'xls'],
                                        upload_id=session_id,
                                    ),
                                    html.Div(id="upload-infobox", children=upload_filename_alert),
                                    ]),
                                ]
                            ),
                        ], className="md-12")
                    ]),
                    html.Br(),
                   

                    html.Div(id="main-methods", children=[
                            #dcc.Location(id='url', refresh=False),
                        
                            dbc.Container([
                                dbc.Row(
                                    dbc.Col([
                                        html.Br(),
                                        html.Div(html.H3("Methods")),
                                        html.Br(),
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
    return html.Div([
        navbar,
        dcc.Location(id='url', refresh=False),
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        html.Div(upload_info, id='upload-filename', style={'display': 'none'}),

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
    else: 
        print("\tohter path....")
        return main_layout_(session_id, upload_filename)  #app.layout  #main_layout_(None)


@app.callback(
    [Output('upload-filename', 'children'),
    Output('upload-infobox', 'children')],
    [Input('upload-data', 'isCompleted')],
    [State('upload-data', 'fileNames'),
     State('upload-data', 'upload_id'),
     State('upload-filename', 'children')])
def return_methods(iscompleted, filenames, upload_id, filename_latest):
    print("Upload callback called")
    
    upload_infobox = html.Div([])

    if filenames is not None:
        filename = filenames[0]
    elif filename_latest != '':
        filename = filename_latest
        upload_infobox = html.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))
    else:
        filename = ''
    
    

    if not iscompleted:
        return filename, upload_infobox

    df = None
    
    if filenames is not None:
        if upload_id:
            root_folder = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)
        else:
            root_folder = UPLOAD_FOLDER_ROOT

        filename = filenames[0]
        file = os.path.join(root_folder, filename)

        df = parse_dataset(file)
        print(df)
        write_dataframe(upload_id, df)
        upload_infobox = html.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))

    if df is None:
        upload_infobox = html.Div(dbc.Alert("There was an error processing this file!", color="danger"))

    print("filename", filename)
    print("upload_infobox", upload_infobox)
    return filename, upload_infobox



if __name__ == '__main__':
    app.run_server(debug=True)


