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
from elmtoolbox.preprocessing import dataset_bacteria_abundances, sampling_statistics, plot_bacteria_abundance_heatmaps, plot_ultradense_longitudinal_data
from elmtoolbox.helpers import get_bacteria_names

from app import app, cache, UPLOAD_FOLDER_ROOT

# layout = html.Div([
#     html.H3('Page 1'),
#     dcc.Dropdown(
#         id='page-1-dropdown',
#         options=[
#             {'label': 'Page 1 - {}'.format(i), 'value': i} for i in [
#                 'NYC', 'MTL', 'LA'
#             ]
#         ]
#     ),
#     html.Div(id='page-1-display-value'),
#     dcc.Link('Back', href='/methods')
# ])


# layout = html.Div([
#             #dcc.Location(id='url', refresh=False),
#             dcc.Link('Back', href='/methods'),

#             dbc.Container([
#                 dbc.Row(
#                     dbc.Col([
#                         html.Div(html.H3("Data Analysis & Exploration")),
#                         html.Br(),
#                         html.Div(id='page-1-display-value'),
#                         ], 
#                     className="md-12"),
#                 ),
#                 dbc.Row(
#                     dbc.Col([
#                         html.Br(),
#                         html.Div(html.H3("Methods")),
#                         html.Br(),
#                         ], 
#                     className="md-12"),
#                 ),
#             ],
#             className="md-4",
#             )
#         ],
#         style={
#             'verticalAlign':'middle',
#             'textAlign': 'center',
#             'backgroundColor': 'rgb(245, 245, 245)',
#             'position':'relative',
#             'width':'100%',
#             #'height':'100vh',
#             'bottom':'0px',
#             'left':'0px',
#             'zIndex':'1000',
#         }
# )


layout = html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        html.H3("Data Analysis & Exploration"),
                        html.Br(),

                        # Abundance plot in general
                        html.H4("Loaded data table"),
                        html.Div(id='page-1-display-value-0'),
                        html.Br(),

                        html.Hr(),

                        # Abundance plot in general
                        html.H4("Taxa Abundances"),
                        html.Div(id='page-1-display-value-1'),
                        html.Br(),

                        html.Hr(),

                        # Sampling statistics
                        html.H4("Sampling Statistics"),
                        html.Div(id='page-1-display-value-2'),
                        html.Br(),

                        html.Hr(),

                        # Heatmap
                        html.H4("Taxa Abundances Histogram"),
                        html.Div(id='page-1-display-value-3'),
                        html.Br(),

                        html.Hr(),

                        # Dense longitudinal data
                        html.H4("Dense Longitudinal Data"),
                        html.Div(id='page-1-display-value-4'),
                        html.Br(),

                        html.Hr(),


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
    Output('page-1-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    data_table =  dash_table.DataTable(
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
    return data_table


@app.callback(
    Output('page-1-display-value-1', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
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
    return dcc.Graph(figure=fig)

@app.callback(
    Output('page-1-display-value-2', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    if max(df.age_at_collection.values) < 100:
        time_unit_name="days"
        time_unit_size=1
    else:
        time_unit_name="months"
        time_unit_size=30

    num_sids = len(df.subjectID.unique())
    fig = sampling_statistics(df, group="group", start_age=0, limit_age=max(df.age_at_collection.values), time_unit_size=time_unit_size, time_unit_name=time_unit_name, file_name=None, height=300+5*num_sids, width=1200, website=True)
    return dcc.Graph(figure=fig)

@app.callback(
    Output('page-1-display-value-3', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    nice_name = lambda x: x[9:].replace("_", " ")

    if max(df.age_at_collection.values) < 100:
        time_unit_name="days"
        time_unit_size=1
    else:
        time_unit_name="months"
        time_unit_size=30

    fig1, fig2 = plot_bacteria_abundance_heatmaps(df, bacteria_names=bacteria_names, short_bacteria_name=nice_name, time_unit_name=time_unit_name, time_unit_size=time_unit_size, avg_fn=np.median, fillna=False, website=True, width=1200)

    return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)])

@app.callback(
    Output('page-1-display-value-4', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    nice_name = lambda x: x[9:].replace("_", " ")

    fig = plot_ultradense_longitudinal_data(df, infants_to_plot=df.subjectID.unique(), nice_name=nice_name, cols_num=15, min_days=0, max_days=max(df.age_at_collection.values), bacteria_names=bacteria_names, file_name = None, h=600, website=True)

    return dcc.Graph(figure=fig)
