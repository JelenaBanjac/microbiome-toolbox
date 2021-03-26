import sys
sys.path.append("../..")

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import numpy as np
import sys
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names, two_groups_analysis


from app import app, cache, UPLOAD_FOLDER_ROOT, loading_img


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Reference Definition & Statistics"),
                        dhc.Br(),
                        dhc.Div(id="page-1-main"),
                        
                        

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
    # Important features
    dhc.Hr(),
    dhc.H4("Important Features for each of the Class"),
    dhc.Div(id='page-1-display-value-0', children=loading_img),
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
    Output('page-1-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


@app.callback(
    Output('page-1-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    df = df.convert_dtypes() 
    id_cols = list(df.columns[df.columns.str.contains("id", case=False)&(df.columns.str.len()<20)].values)
    cols_to_ignore = [ 'dataset_type', 'dataset_type_classification', 'classification_dataset_type', 'classification_label' ]  #'healthy_reference', 
    str_cols = list(set(df.columns[df.dtypes=="string"]) - set(id_cols + cols_to_ignore))
    df = pd.get_dummies(df, columns=str_cols)
    print(df.columns.values)
    
    #bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    feature_columns = set(df.columns) - set(id_cols + cols_to_ignore)
    fig1, img_src  = two_groups_analysis(df, feature_columns, references_we_compare='healthy_reference',nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="dot", show=False, website=True, layout_height=800, layout_width=1000)
    fig2           = two_groups_analysis(df, feature_columns, references_we_compare='healthy_reference', nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="hist", show=False, website=True, layout_height=800, layout_width=1000)


    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
                    dcc.Graph(figure=fig1),
                    dhc.Br(),
                    dcc.Graph(figure=fig2),
                    dhc.Br(),
                    dhc.Img(src=img_src),
                    dhc.Br(),dhc.Br(),
                    ]

    return ret_val

