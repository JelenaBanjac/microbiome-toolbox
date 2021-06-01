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
from microbiome.helpers import get_bacteria_names
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories

from microbiome.variables import *

from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img, INTERVAL, MAX_INTERVALS


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Microbiome Trajectory"),
                        dhc.Br(),
                        dcc.Markdown('''
                        * Data handling (hunting for the plateau of performance reached, so we can use less number of features):  
                            - top K important features selection based on the smallest MAE error (i.e. how does trajectory performance looks like when only working with top 5 or top 10 bacteria used for the model)  
                            - remove near zero variance features  
                            - remove correlated features  
                        * Microbiome Trajectory - all the combinations below
                            - only mean line  
                            - only line with prediction interval and confidence interval  
                            - line with samples  
                            - longitudinal data, every subject  
                            - coloring per group (e.g. per country)  
                            - red-color dots or stars-shapes are outliers  
                        * Measuring the trajectory performance (all before plateau area):  
                            - MAE  (goal: *smaller*)  
                            - R^2 score (goal: *bigger*), percent of variance captured  
                            - Pearson correlation (MMI, age_at_collection)  
                            - Prediction Interval (PI) - mean and median = prediction interval 95% = the interval in which we expect the healthy reference to fall in (goal: *smaller*)  
                            - Standard deviation of the error  
                            - Visual check
                        * Testing difference between different trajectories using linear regression statistical analysis and spline:  
                            - Testing **universality** across different groups  
                            - Testing **differentiation** of 2 trajectories (e.g. healthy vs. non-healthy) - spline p-values, linear regression p-values  
                        ''', style={'textAlign': 'left',}),

                        dhc.Div(id="page-3-main"),
                        dcc.Interval(
                            id='page-3-main-interval-component',
                            interval=10000, # in milliseconds
                            n_intervals=0,  # start counter
                            max_intervals=30  #MAX_INTERVALS
                        ),
                        dhc.Div(id="test"),
                        
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
    
    dhc.Hr(),
    dhc.H4("Only Trajectory Line"),
    dhc.Br(),
    #dhc.Div(id='page-3-display-value-0', children=loading_img),
    #dhc.Div(id='page-3-display-value-0-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-0",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-0-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Longitudinal Subject's Data"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-1', children=loading_img),
    # dhc.Div(id='page-3-display-value-1-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-1",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-1-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Universality: Linear Difference between Group Trajectories"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-2', children=loading_img),
    # dhc.Div(id='page-3-display-value-2-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-2",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-2-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Universality: Nonlinear Difference between Group Trajectories"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-3', children=loading_img),
    # dhc.Div(id='page-3-display-value-3-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-3",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-3-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Reference vs. Non-reference Longitudinal Trajectories"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-4', children=loading_img),
    # dhc.Div(id='page-3-display-value-4-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-4",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-4-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Differentiation: Linear Reference vs. Non-reference Difference Between Trajectories"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-5', children=loading_img),
    # dhc.Div(id='page-3-display-value-5-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-5",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-5-hidden")])],
        type="circle",
    ),

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Differentiation: Nonlinear (Spline) Reference vs. Non-reference Difference Between Trajectories"),
    dhc.Br(),
    # dhc.Div(id='page-3-display-value-6', children=loading_img),
    # dhc.Div(id='page-3-display-value-6-hidden', hidden=True),
    dcc.Loading(
        id="loading-3-6",
        children=[dhc.Div([dhc.Div(id="page-3-display-value-6-hidden")])],
        type="circle",
    ),

    dhc.Br(),
]



# cache memoize this and add timestamp as input!
# @cache.memoize()
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.pickle")
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print('** Reading data from disk **')
    else:
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

# @app.callback(
#    [#Output('page-3-display-value-0', 'children'),
#     Output('page-3-display-value-1', 'children'),
#     Output('page-3-display-value-2', 'children'),
#     Output('page-3-display-value-3', 'children'),
#     Output('page-3-display-value-4', 'children'),
#     Output('page-3-display-value-5', 'children'),
#     Output('page-3-display-value-6', 'children')],
#    [#Input('page-3-main-interval-component', 'n_intervals'),
#     State('page-3-display-value-0-hidden', 'children'),
#     State('page-3-display-value-1-hidden', 'children'),
#     State('page-3-display-value-2-hidden', 'children'),
#     State('page-3-display-value-3-hidden', 'children'),
#     State('page-3-display-value-4-hidden', 'children'),
#     State('page-3-display-value-5-hidden', 'children'),
#     State('page-3-display-value-6-hidden', 'children')])
# def display_value(n, c0, c1, c2, c3, c4, c5, c6):
#     print("=====")
#     print("interval:", n)
#     print(c0)
#     print("=====")
#     return c0, c1, c2, c3, c4, c5, c6


@app.callback(
    Output('test', 'children'),
    Input('page-3-main-interval-component', 'n_intervals'))
def display_value(n_intervals):
    print("n_intervals", n_intervals)
    return ""


@app.callback(
    Output('page-3-display-value-0-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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
    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig1),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val


@app.callback(
    Output('page-3-display-value-1-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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

        fig7,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group=None, nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True, longitudinal_mode="markers+lines");

    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig7),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val


@app.callback(
    Output('page-3-display-value-2-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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


        fig2,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group="group", linear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);
    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig2),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val



@app.callback(
    Output('page-3-display-value-3-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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

        fig3,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group="group", nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start,  time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);
    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig3),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val



@app.callback(
    Output('page-3-display-value-4-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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


        fig4,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=other, group=None, nonlinear_difference=True, start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True);
    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig4),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val



@app.callback(
    Output('page-3-display-value-5-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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

        fig5 = plot_2_trajectories(estimator, val1, other, feature_cols=bacteria_names, degree=2, plateau_area_start=plateau_area_start, limit_age=limit_age, start_age=0, time_unit_size=time_unit_size, time_unit_name=time_unit_name, linear_pval=True, nonlinear_pval=False, img_file_name=None, website=True)
    except Exception as e:
        df = None

    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig5),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val



@app.callback(
    Output('page-3-display-value-6-hidden', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    if max(df.age_at_collection.values) < 100:
        plateau_area_start=None #45
        time_unit_size=1
        time_unit_name="days"
        limit_age = 60
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        limit_age = 750

    try:
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


        fig6 = plot_2_trajectories(estimator, val1, other, feature_cols=bacteria_names, degree=2, plateau_area_start=plateau_area_start, limit_age=limit_age, start_age=0, time_unit_size=time_unit_size, time_unit_name=time_unit_name, linear_pval=False, nonlinear_pval=True, img_file_name=None, website=True)
    except Exception as e:
        df = None

    
    if df is not None:
        ret_val =  [
            dcc.Graph(figure=fig6),
        ]
    else:
        ret_val = dhc.Div([])

    return ret_val