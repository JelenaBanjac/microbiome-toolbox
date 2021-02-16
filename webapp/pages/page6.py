import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import numpy as np
import sys
import dash_table
#sys.path.append("C://Users//RDBanjacJe//Desktop//ELMToolBox") 
from microbiome.preprocessing import dataset_bacteria_abundances, sampling_statistics, plot_bacteria_abundance_heatmaps, plot_ultradense_longitudinal_data
from microbiome.helpers import get_bacteria_names

from app import app, cache, UPLOAD_FOLDER_ROOT


layout = html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        html.H3("Longitudinal Anomaly Detection"),
                        html.Br(),
                        html.Div(id="page-6-reloaded"),
                        
                        # Abundance plot in general
                        html.Div(id='page-6-display-value-0'),

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
    Output('page-6-reloaded', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    if df is not None:
        ret_val =  html.Div([])
    else:
        ret_val = html.Div(dbc.Alert(["You refreshed the page or were idle for too long so data. Data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))
    return ret_val


@app.callback(
    Output('page-6-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:
        ret_val =  [html.Hr(),
                    html.H4("Loaded data table"),
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
                    html.Br(),
                    ]

    return ret_val

