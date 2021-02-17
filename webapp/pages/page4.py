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
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories

from microbiome.variables import *

from app import app, cache, UPLOAD_FOLDER_ROOT


layout = html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        html.H3("Microbiome Trajectory"),
                        html.Br(),
                        html.Div(id="page-4-reloaded"),
                        
                        # Abundance plot in general
                        html.Div(id='page-4-display-value-0'),

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
    Output('page-4-reloaded', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)

    if df is not None:
        ret_val =  html.Div([])
    else:
        ret_val = html.Div(dbc.Alert(["You refreshed the page or were idle for too long so data. Data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))
    return ret_val


@app.callback(
    Output('page-4-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 45
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2, file_name=None)

    # # healthy unseen data - Test-1
    # val1 = df[df.classification_dataset_type=="Test-1"]
    # # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    # other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # # unhealthy unseen data - Test2
    # val2 =  df[df.classification_dataset_type=="Test-2"]
     # healthy unseen data - Test-1
    val1 = df[df.dataset_type=="Validation"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.dataset_type=="Test"]
    # unhealthy unseen data - Test2
    #val2 =  df[df.classification_dataset_type=="Test-2"]

    fig1,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group=None, nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);

    fig2,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group="group", linear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);

    fig3,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group="group", nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start,  time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);

    fig4,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=other, group=None, nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);

    fig5 = plot_2_trajectories(estimator, val1, other, feature_cols=bacteria_names, degree=2, plateau_area_start=plateau_area_start, limit_age=limit_age, start_age=0, time_unit_size=time_unit_size, time_unit_name=time_unit_name, linear_pval=True, nonlinear_pval=False, img_file_name=None, website=True)

    fig6 = plot_2_trajectories(estimator, val1, other, feature_cols=bacteria_names, degree=2, plateau_area_start=plateau_area_start, limit_age=limit_age, start_age=0, time_unit_size=time_unit_size, time_unit_name=time_unit_name, linear_pval=False, nonlinear_pval=True, img_file_name=None, website=True)

    ret_val = html.Div([])
    if df is not None:
        ret_val =  [html.Hr(),
                    html.H4("Only Trajectory Line"),
                    dcc.Graph(figure=fig1),
                    html.Br(),
                    html.Hr(),
                    html.H4("Universality: Linear Difference between Group Trajectories"),
                    dcc.Graph(figure=fig2),
                    html.Br(),
                    html.Hr(),
                    html.H4("Universality: Nonlinear Difference between Group Trajectories"),
                    dcc.Graph(figure=fig3),
                    html.Br(),
                    html.Hr(),
                    html.H4("Healthy vs. Non-healthy Longitudinal Trajectories"),
                    dcc.Graph(figure=fig4),
                    html.Br(),
                    html.Hr(),
                    html.H4("Differentiation: Linear Healthy vs. Non-healthy Difference Between Trajectories"),
                    dcc.Graph(figure=fig5),
                    html.Br(),
                    html.Hr(),
                    html.H4("Differentiation: Nonlinear (Spline) Healthy vs. Non-healthy Difference Between Trajectories"),
                    dcc.Graph(figure=fig6),
                    html.Br(),
                    ]

    return ret_val

