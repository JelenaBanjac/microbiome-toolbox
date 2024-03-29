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
import dash_table
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names
from microbiome.postprocessing import *
from microbiome.longitudinal_anomaly_detection import *
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories
import dash_dangerously_set_inner_html

from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img, LOADING_TYPE


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Intervention Simulation"),
                        dhc.Br(),
                        dcc.Markdown('''
                        Click on one of the outliers below to see the suggestion for the intervention. 
                        The intervention simulation consists of suggesting the taxa values to change (or log-ratio values to change) in order to bring back the sample to the reference microbiome trajectory.
                        ''', style={'textAlign': 'left',}),

                        dhc.Div(id="page-6-main"),
                        # dcc.Interval(
                        #     id='page-6-main-interval-component',
                        #     interval=10000, # in milliseconds
                        #     n_intervals=0,
                        #     max_intervals=5
                        # )
                        
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
    dhc.H4("Select Outlier"),
    # dhc.Div(id='page-6-display-value-0', children=loading_img),
    # # dhc.Div(id='page-6-display-value-0-hidden', hidden=True),
    dhc.Br(),
    dcc.Loading(
        id="loading-6-0",
        children=[dhc.Div([dhc.Div(id="page-6-display-value-0")])],
        type=LOADING_TYPE,
    ),
    dhc.Br(),
]

# cache memoize this and add timestamp as input!
# @cache.memoize()
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
    Output('page-6-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content



# @app.callback(
#     Output('page-6-display-value-0', 'children'),
#     [Input('page-6-main-interval-component', 'children'),
#     Input('page-6-display-value-0-hidden', 'children')])
# def display_value(n, fig_content):
    
#     return fig_content


@app.callback(
    Output('page-6-display-value-0', 'children'),
    [Input('session-id', 'children')])
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    nice_name = lambda x: x[9:].replace("_", " ")
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        box_height = None
        units = [20, 20, 20] 
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        box_height = None
        units = [90, 90, 90, 90, 90, 90]

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

    # healthy unseen data - Test-1
    val1 = df[df.classification_dataset_type=="Test-1"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # unhealthy unseen data - Test2
    val2 =  df[df.classification_dataset_type=="Test-2"]

    #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    

    outliers = pi_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, degree=2)
    #outlier_id = outliers[0]

    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    
    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
                    dcc.Graph(figure=fig, id="graph2"),
                    dhc.Br(),dhc.Br(),
                    dhc.Div(id="graph2-info"),
                    ]

    return ret_val



@app.callback(Output("graph2-info", "children"), 
             [Input("graph2", "clickData"), Input('session-id', 'children')],
             [State("graph2", "figure")])
def update_color(clickData, session_id, fig):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    nice_name = lambda x: x[9:].replace("_", " ")
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        box_height = None
        units = [20, 20, 20] 
        limit_age = max(df.age_at_collection.values)
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        box_height = None
        units = [90, 90, 90, 90, 90, 90]

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

    # healthy unseen data - Test-1
    val1 = df[df.classification_dataset_type=="Test-1"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # unhealthy unseen data - Test2
    val2 =  df[df.classification_dataset_type=="Test-2"]

    #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    



    ret_val = []
    print("clickData", clickData)
    if clickData is not None:
        outlier_id = clickData["points"][0]["text"]
        print(outlier_id)

        fig_before, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=[outlier_id], df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
        nice_name = lambda name: " | ".join([c[3:] for c in name[9:].split("|")[-2:]])

        _val, fig1, fig2, ret_val = outlier_intervention(outlier_sampleID=outlier_id, estimator=estimator, df_all=val1, feature_columns=bacteria_names, nice_name=nice_name, max_num_bacterias_to_change=20, 
                            traj_x=traj_x, traj_mean=traj_mean, traj_pi=traj_pi, 
                            time_unit_size=1, time_unit_name="days", 
                            file_name=None, output_html=False, plot=False, normal_vals=[ False], average=np.median, std=np.std)

        fig_after, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, patent=False, highlight_outliers=[outlier_id], df_new=_val, 
                                  time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, plateau_area_start=None, longitudinal_mode="markers", longitudinal_showlegend=False, fillcolor_alpha=0.2, 
                                  img_file_name=None, website=True,
                                  num_top_bacteria=5
                                  );
        stats = dhc.Div(
            [dcc.Markdown(r) for r in ret_val.split("\n")]
        )

        if _val is not None:
            after_intervention = [
                dhc.Br(),
                dhc.H4("After Intervention"),
                dcc.Graph(figure=fig_after) if fig_after else ""
            ]
        else:
            after_intervention = []

        ret_val = [
            dhc.Hr(),
            dhc.H4("Before Intervention"),
            dcc.Graph(figure=fig_before),
            dhc.Br(),
            #dhc.P(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(ret_val.replace('\n','<br/>'))),
            stats,
            # dcc.Graph(figure=fig1),
            # dhc.Br(),
            # dcc.Graph(figure=fig2),
            *after_intervention,
            dhc.Br(),
            dhc.Br(),]

    else:
        ret_val = []


    return ret_val
