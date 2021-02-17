import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import numpy as np
import sys
import dash_table
sys.path.append("C://Users//RDBanjacJe//Desktop//ELMToolBox") 
from microbiome.preprocessing import dataset_bacteria_abundances, sampling_statistics, plot_bacteria_abundance_heatmaps, plot_ultradense_longitudinal_data, plot_diversity
from microbiome.helpers import get_bacteria_names

from app import app, cache, UPLOAD_FOLDER_ROOT


layout = html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        html.H3("Data Analysis & Exploration"),
                        html.Br(),
                        html.Div(id="page-1-reloaded"),

                        # Abundance plot in general
                        html.Div(id='page-1-display-value-0'),

                        # Abundance plot in general
                        html.Div(id='page-1-display-value-1'),
                        
                        # Sampling statistics
                        html.Div(id='page-1-display-value-2'),
                        
                        # Heatmap
                        html.Div(id='page-1-display-value-3'),

                        # Shannon's diversity index and Simpson's dominace
                        html.Div(id='page-1-display-value-4'),

                        # Dense longitudinal data
                        html.Div(id='page-1-display-value-5'),
                        
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
    Output('page-1-reloaded', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    if df is not None:
        ret_val =  html.Div([])
    else:
        ret_val = html.Div(dbc.Alert(["You refreshed the page or were idle for too long so data. Data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))
    return ret_val


@app.callback(
    Output('page-1-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:
        ret_val =  [
            html.Hr(),
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
            html.Br()
            ]
    return ret_val


@app.callback(
    Output('page-1-display-value-1', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:
        
        
        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        if max(df.age_at_collection.values) < 100:
            time_unit_name="days"
            time_unit_size=1
        else:
            time_unit_name="months"
            time_unit_size=30
        
        num_cols = 3
        total_num_rows = len(bacteria_names)//num_cols+1

        fig = dataset_bacteria_abundances(df, bacteria_names, time_unit_size=time_unit_size, time_unit_name=time_unit_name, num_cols=num_cols, nice_name=nice_name, file_name=None, width=1200, height=200*total_num_rows, website=True)
        
        ret_val = [
            html.Hr(),
            html.H4("Taxa Abundances"),
            dcc.Graph(figure=fig),
            html.Br(),
        ]
    return ret_val

@app.callback(
    Output('page-1-display-value-2', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:

        if max(df.age_at_collection.values) < 100:
            time_unit_name="days"
            time_unit_size=1
        else:
            time_unit_name="months"
            time_unit_size=30

        num_sids = len(df.subjectID.unique())
        fig = sampling_statistics(df, group="group", start_age=0, limit_age=max(df.age_at_collection.values), time_unit_size=time_unit_size, time_unit_name=time_unit_name, file_name=None, height=300+5*num_sids, width=1200, website=True)

        ret_val = [
            html.Hr(),
            html.H4("Sampling Statistics"),
            dcc.Graph(figure=fig),
            html.Br(),
        ]
    return ret_val

    
@app.callback(
    Output('page-1-display-value-3', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        if max(df.age_at_collection.values) < 100:
            time_unit_name="days"
            time_unit_size=1
        else:
            time_unit_name="months"
            time_unit_size=30

        fig1, fig2 = plot_bacteria_abundance_heatmaps(df, bacteria_names=bacteria_names, short_bacteria_name=nice_name, time_unit_name=time_unit_name, time_unit_size=time_unit_size, avg_fn=np.median, fillna=False, website=True, width=1200)

        ret_val = [
            html.Hr(),
            html.H4("Taxa Abundances Histogram"),
            html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]),
            html.Br(),
        ]
    return ret_val

@app.callback(
    Output('page-1-display-value-4', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))

        if max(df.age_at_collection.values) < 100:
            time_unit_name="days"
            time_unit_size=1
        else:
            time_unit_name="months"
            time_unit_size=30
        
        fig1 = plot_diversity(df, bacteria_names, diversity="shannon", group="group", time_unit_name=time_unit_name, time_unit_size=time_unit_size, layout_height=800, layout_width=1000, website=True)
        fig2 = plot_diversity(df, bacteria_names, diversity="simpson", group="group", time_unit_name=time_unit_name, time_unit_size=time_unit_size, layout_height=800, layout_width=1000, website=True)

        ret_val = [
            html.Hr(),
            html.H4("Diversity"),
            dcc.Graph(figure=fig1),
            html.Br(),
            dcc.Graph(figure=fig2),
            html.Br(),
        ]

    return ret_val

@app.callback(
    Output('page-1-display-value-5', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = html.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)

        ret_val = [
            html.Hr(),
            html.H4("Dense Longitudinal Data"),
            dcc.Graph(figure=fig),
            html.Br(),
        ]

    return ret_val
