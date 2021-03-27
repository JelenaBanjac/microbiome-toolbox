import sys
sys.path.append("../..")
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names
from microbiome.data_analysis import *
from ipywidgets import widgets

import dash
from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)


print(dir(dhc))
layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Data Analysis & Exploration"),
                        dhc.Br(),
                        dhc.Div(id="page-2-main"),

                        
                        
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
    dhc.Div(id='page-2-display-value-0', children=loading_img),

    # Abundance plot in general
    dhc.Hr(),
    dhc.H4("Taxa Abundances"),
    dhc.Div(id='page-2-display-value-1', children=loading_img),
    
    # Sampling statistics
    dhc.Hr(),
    dhc.H4("Sampling Statistics"),
    dhc.Div(id='page-2-display-value-2', children=loading_img),
    
    # Heatmap
    dhc.Hr(),
    dhc.H4("Taxa Abundances Histogram"),
    dhc.Div(id='page-2-display-value-3', children=loading_img),

    # Shannon's diversity index and Simpson's dominace
    dhc.Hr(),
    dhc.H4("Diversity"),
    dhc.Div(id='page-2-display-value-4', children=loading_img),

    # Dense longitudinal data
    dhc.Hr(),
    dhc.H4("Dense Longitudinal Data"),
    dhc.Div(id='page-2-display-value-5', children=loading_img),

    # Embedding in 2D
    dhc.Hr(),
    dhc.H4("Embedding in 2D space"),
    dhc.Div(id='page-2-display-value-6', children=loading_img),

    # Embedding in 3D
    dhc.Hr(),
    dhc.H4("Embedding in 3D space"),
    dhc.Div(id='page-2-display-value-7', children=loading_img),

    # Embedding in 2D, interactive
    dhc.Hr(),
    dhc.H4("Embedding in 2D space - Interactive Analysis"),
    dhc.Div(id='page-2-display-value-8', children=loading_img),
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
    Output('page-2-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


@app.callback(
    Output('page-2-display-value-0', 'children'),
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
            dhc.Br()
            ]
    return ret_val


@app.callback(
    Output('page-2-display-value-1', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
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
            
            dcc.Graph(figure=fig),
            dhc.Br(),
        ]
    return ret_val

@app.callback(
    Output('page-2-display-value-2', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
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
            
            dcc.Graph(figure=fig),
            dhc.Br(),
        ]
    return ret_val

    
@app.callback(
    Output('page-2-display-value-3', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
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
            
            dhc.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]),
            dhc.Br(),
        ]
    return ret_val

@app.callback(
    Output('page-2-display-value-4', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
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
            
            dcc.Graph(figure=fig1),
            dhc.Br(),
            dcc.Graph(figure=fig2),
            dhc.Br(),
        ]

    return ret_val

@app.callback(
    Output('page-2-display-value-5', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)

        ret_val = [
            
            dcc.Graph(figure=fig),
            dhc.Br(),
        ]

    return ret_val


@app.callback(
    Output('page-2-display-value-6', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        emb = decomposition.PCA(n_components=3)
        fig = embedding(emb, df_all=df, feature_columns=bacteria_names, embedding_dimension=2, layout_settings=dict(height=600, width=600), color_column_name="group", website=True);

        #fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)

        ret_val = [
            dcc.Graph(figure=fig),
            dhc.Br(),
        ]

    return ret_val

@app.callback(
    Output('page-2-display-value-7', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        emb = decomposition.PCA(n_components=3)
        fig = embedding(emb, df_all=df, feature_columns=bacteria_names, embedding_dimension=3, layout_settings=dict(height=1000, width=1000), color_column_name="group", website=True);

        #fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)

        ret_val = [
            dcc.Graph(figure=fig),
            dhc.Br(),
        ]

    return ret_val

from IPython.display import display

@app.callback(
    Output('page-2-display-value-8', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:

        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        emb = decomposition.PCA(n_components=3)
        vbox = embeddings_interactive_selection_notebook(df_all=df, feature_columns=bacteria_names, emb=emb, layout_settings=dict(height=1000, width=1000));

        #fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)
        
        ret_val = [
            dcc.Graph(figure=vbox.children[0], id="graph"),
            #dcc.Graph(figure=vbox.children[1]),
            #fig,
            dhc.Div(id="graph-info"),
            dhc.Br(),
        ]

    return ret_val



@app.callback(Output("graph-info", "children"), 
             [Input("graph", "selectedData"), Input('session-id', 'children')],
             [State("graph", "figure")])
def update_color(selectedData, session_id, fig):
    selection = None
    # Update selection based on which event triggered the update.
    trigger = dash.callback_context.triggered[0]["prop_id"]
    # if trigger == 'graph.clickData':
    #     selection = [point["pointNumber"] for point in clickData["points"]]
    if trigger == 'graph.selectedData':
        selection = [point["pointIndex"] for point in selectedData["points"]]
    
    if selection is not None:
        # Update scatter selection
        fig["data"][0]["selectedpoints"] = selection

        df = read_dataframe(session_id, None)
        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        nice_name = lambda x: x[9:].replace("_", " ")

        t = go.FigureWidget([
            go.Table(
            header=dict(values=['sampleID','subjectID'],
                        fill = dict(color='#C2D4FF'),
                        align = ['left'] * 5),
            cells=dict(values=[df[col] for col in ['sampleID','subjectID']],
                    fill = dict(color='#F5F8FF'),
                    align = ['left'] * 5))])
        t.data[0].cells.values = [df.loc[selection][col] for col in ['sampleID','subjectID']]

        df_selected = pd.DataFrame(data={"sampleID":t.data[0].cells.values[0],
                                        "subjectID":t.data[0].cells.values[1]})

        ###
        # create new column called selected to use for reference analysis: True - selected, False - not selected
        df["selected"] = False
        df.loc[df["sampleID"].isin(df_selected["sampleID"]), "selected"] = True
        
        # plot the result of reference analysis with feature_columns_for_reference
        ffig, img_src = two_groups_analysis(df, bacteria_names, references_we_compare="selected", test_size=0.5, n_splits=5, nice_name=nice_name, style="dot", show=False, website=True, layout_height=1000, layout_width=1000, max_display=20);


        # Update parcats colors
        # new_color = np.zeros(len(fig["data"][1]["line"]["color"]), dtype='uint8')
        # new_color[selection] = 1
        # fig["data"][1]["line"]["color"] = new_color
        ret_val = [
            dcc.Graph(figure=t),
            dcc.Graph(figure=ffig),
            dhc.Br(),
            dhc.Img(src=img_src),
            dhc.Br(),
        ]
    else:
        ret_val = []
    return ret_val

