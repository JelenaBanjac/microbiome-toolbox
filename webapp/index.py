import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
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

from app import app, filecache_dir, cache

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
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Data Analysis & Exploration", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    html.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/page-1"),
                ]
            ),
        ],
        #style={"width": "18rem"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card2 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Healthy Reference", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", outline=True, color="dark"),
                ]
            ),
        ],
        #style={"width": "18rem"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card3 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Differential Ranking", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", outline=True, color="dark"),
                ]
            ),
        ],
        #style={"width": "18rem"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card4 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Microbiome Trajectory", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", outline=True, color="dark"),
                ]
            ),
        ],
        #style={"width": "18rem"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card5 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Bacteria Importance with Time", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", outline=True, color="dark"),
                ]
            ),
        ],
        #style={"width": "18rem"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card6 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="static/img/download.svg", top=True),
            dbc.CardBody(
                [
                    html.H4("Longitudinal Anomaly Detection", className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", outline=True, color="dark"),
                ]
            ),
        ],
        style={"backgroundColor": "rgb(255, 255, 255)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)




index_page = html.Div([
    dbc.Row([
        dbc.Col(className="md-4"),
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '30%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
            ), className="md-4"
        ),
        dbc.Col(className="md-4"),
        ]
    ),
    
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
])



def parse_dataset(content, filename):
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    df = None
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            #df = pd.read_excel(io.BytesIO(decoded))
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep="\t")
    except Exception as e:
        print(e)
        return None
    print('Read dataframe:')
    print(df.columns)
    print(df)
    return df


def write_dataframe(session_id, df):
    '''
    Write dataframe to disk, for now just as CSV
    For now do not preserve or distinguish filename;
    user has one file at once.
    '''
    print('Calling write_dataframe')
    filename = os.path.join(filecache_dir, session_id)
    df.to_pickle(filename)

# cache memoize this and add timestamp as input!
@cache.memoize()
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    print('Calling read_dataframe')
    print('filecache_dir', filecache_dir)
    print(type(filecache_dir))
    print('session_id', session_id)
    filename = os.path.join(filecache_dir, session_id)
    print('filename', filename)
    df = pd.read_pickle(filename)
    # simulate reading in big data with a delay
    print('** Reading data from disk **')
    return df


def methods_container(session_id):

    df = read_dataframe(session_id, None)

    return html.Div([
                #dcc.Location(href="/methods", id='url', refresh=False),
               
                dbc.Container([
                    dbc.Row(
                        dbc.Col([
                            html.Div(html.H3("Data Loaded")),
                            html.Br(),
                            dash_table.DataTable(
                                id='upload-datatable',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.to_dict('records'),
                                style_data={
                                    'width': '{}%'.format(max(df.columns, key=len)),
                                    'minWidth': '50px',
                                    'maxWidth': '500px',
                                },
                                style_table={
                                    'height': 300, 
                                    'overflowX': 'auto'
                                }  
                            )
                            ], 
                        className="md-12"),
                    ),
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
                'backgroundColor': 'rgb(245, 245, 245)',
                'position':'relative',
                'width':'100%',
                #'height':'100vh',
                'bottom':'0px',
                'left':'0px',
                'zIndex':'1000',
            }
    )




def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div([
        navbar,
        #dcc.Location(id='url', refresh=False),
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        #html.Div(id='filecache_marker', style={'display': 'none'}),
        #index_page,
        html.Div(index_page, id='output-data-upload'),
        html.Br(),
        html.Div(id='page-content')
    ])

app.layout = serve_layout



# Update the index
# @app.callback(Output('page-content', 'children'),
#               [Input('url', 'pathname')])
# def display_page(pathname):
#     if pathname == '/page-1':
#         return page1.layout
#     elif pathname == '/page-2':
#         return page2.layout
#     else:
#         return index_page



### RETURNS METHODS ###
@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'last_modified')],
    [State('session-id', 'children')])
def return_methods(contents, filename, last_modified, session_id):

    if not contents:
        return index_page

    # write contents to file
    print('Calling save_file')
    print('New last_modified would be',last_modified)
    df = None
    if contents is not None:
        print('contents is not None')
        df = parse_dataset(contents, filename)
        write_dataframe(session_id, df)
    if df is None:
        return html.Div(dbc.Alert("There was an error processing this file!", color="danger"))
    
    return methods_container(session_id)


if __name__ == '__main__':
    app.run_server(debug=True)


