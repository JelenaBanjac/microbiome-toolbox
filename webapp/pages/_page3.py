import sys
sys.path.append("/home/jelena/Desktop/microbiome-toolbox")

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import numpy as np
import sys
import dash_table
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names

from app import app, cache, UPLOAD_FOLDER_ROOT


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Differential Ranking"),
                        dhc.Br(),
                        dhc.Div(id="page-3-main"),
                        
                        

                    ], className="md-4")
                )
            ], className="md-4",)
    ], 
    style={
        'verticalAlign':'middle',
        'textAlign': 'center',
        'backgroundColor': "rgb(255, 255, 255)", #'rgb(245, 245, 245)',
        'position':'relative',
        'width':'100%',
        #'height':'100vh',
        'bottom':'0px',
        'left':'0px',
        'zIndex':'1000',
    }
)

page_content = [
    # Loaded table
    dhc.Hr(),
    dhc.H4("Loaded data table"),
    dhc.Div(id='page-3-display-value-0'),
]

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


@app.callback(
    Output('page-3-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


@app.callback(
    Output('page-3-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
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
                        ),
                    dhc.Br(),
                    ]

    return ret_val

