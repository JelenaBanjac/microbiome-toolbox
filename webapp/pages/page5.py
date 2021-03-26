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
#sys.path.append("C://Users//RDBanjacJe//Desktop//ELMToolBox") 
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names
from microbiome.postprocessing import *
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories
from microbiome.longitudinal_anomaly_detection import *

from app import app, cache, UPLOAD_FOLDER_ROOT, loading_img


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Longitudinal Anomaly Detection"),
                        dhc.Br(),
                        dhc.Div(id="page-5-main"),
                        
                        
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
    # Outside Prediction Interval
    dhc.Hr(),dhc.Br(),
    dhc.H4("Outside Prediction Interval"),
    dhc.Div(id='page-5-display-value-0', children=loading_img),
    dhc.Br(),

    # Low Pass Filter Longitudinal Anomaly Detection
    dhc.Hr(),dhc.Br(),
    dhc.H4("Low Pass Filter Longitudinal Anomaly Detection"),
    dhc.Div(id='page-5-display-value-1', children=loading_img),
    dhc.Br(),
    
    # Isolation Forest Longitudinal Anomaly Detection
    dhc.Hr(),dhc.Br(),
    dhc.H4("Isolation Forest Longitudinal Anomaly Detection"),
    dhc.Div(id='page-5-display-value-2', children=loading_img),
    dhc.Br(),
    
    # All outliers analysis
    dhc.Hr(),dhc.Br(),
    dhc.H4("Outliers analysis"),
    dhc.Div(id='page-5-display-value-3', children=loading_img),
    dhc.Br(),
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
    Output('page-5-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


@app.callback(
    Output('page-5-display-value-0', 'children'),
    Input('session-id', 'children'))
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
    outlier_id = outliers[0]

    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    

    # ret_val = dhc.Div([])
    # if df is not None:
    #     ret_val =  [
    #                 dcc.Graph(figure=fig),
    #                 dhc.Br(),
    #                 dhc.Br(),
    #                 ]

    return dcc.Graph(figure=fig)



@app.callback(
    Output('page-5-display-value-1', 'children'),
    Input('session-id', 'children'))
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
        window = 10
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]
        num_of_stds=1.5
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        box_height = None
        units = [90, 90, 90, 90, 90, 90]
        window = 100
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]
        num_of_stds=1.5

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

    # healthy unseen data - Test-1
    val1 = df[df.classification_dataset_type=="Test-1"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # unhealthy unseen data - Test2
    val2 =  df[df.classification_dataset_type=="Test-2"]

    #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    

    outliers = lpf_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, 
                                    num_of_stds=num_of_stds, window=window)

    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    

    # ret_val = dhc.Div([])
    # if df is not None:
    #     ret_val =  [
    #                 dcc.Graph(figure=fig),
    #                 dhc.Br(),
    #                 dhc.Br(),
    #                 ]

    return dcc.Graph(figure=fig)



@app.callback(
    Output('page-5-display-value-2', 'children'),
    Input('session-id', 'children'))
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
        window = 10
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        box_height = None
        units = [90, 90, 90, 90, 90, 90]
        window = 100
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

    # healthy unseen data - Test-1
    val1 = df[df.classification_dataset_type=="Test-1"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # unhealthy unseen data - Test2
    val2 =  df[df.classification_dataset_type=="Test-2"]

    #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    

    outliers = if_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, 
                                    outliers_fraction=outliers_fraction, window=window, anomaly_columns=anomaly_columns)

    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    

    # ret_val = dhc.Div([])
    # if df is not None:
    #     ret_val =  [
    #                 dcc.Graph(figure=fig),
    #                 dhc.Br(),
    #                 dhc.Br(),
    #                 ]

    return dcc.Graph(figure=fig)



@app.callback(
    Output('page-5-display-value-3', 'children'),
    Input('session-id', 'children'))
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
        window = 10
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]
        num_of_stds=1.5
    else:
        plateau_area_start=None  #700
        time_unit_size=30
        time_unit_name="months"
        box_height = None
        units = [90, 90, 90, 90, 90, 90]
        window = 100
        outliers_fraction = 0.1
        anomaly_columns=["y_pred_zscore"]
        num_of_stds=1.5

    estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

    # healthy unseen data - Test-1
    val1 = df[df.classification_dataset_type=="Test-1"]
    # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
    other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
    # unhealthy unseen data - Test2
    val2 =  df[df.classification_dataset_type=="Test-2"]

    #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    

    outliers1 = pi_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, degree=2)
    outliers2 = lpf_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, 
                                    num_of_stds=num_of_stds, window=window)
    outliers3 = if_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, 
                                    outliers_fraction=outliers_fraction, window=window, anomaly_columns=anomaly_columns)

    all_outliers = list(set(outliers1+outliers2+outliers3))


    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=all_outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    
    # create new column called selected to use for reference analysis: True - selected, False - not selected
    val1["selected"] = False
    val1.loc[val1["sampleID"].isin(all_outliers), "selected"] = True

    ffig, img_src = two_groups_analysis(val1, bacteria_names, references_we_compare="selected", test_size=0.5, n_splits=2, nice_name=nice_name, style="dot", 
                                        show=False, website=True, layout_height=1000, layout_width=1000, max_display=50);
    
    
    # ret_val = dhc.Div([])
    # if df is not None:
    #     ret_val =  [
    #                 dcc.Graph(figure=fig),
    #                 dhc.Br(),
    #                 dhc.Br(),
    #                 ]

    ret_val = [
        dcc.Graph(figure=fig),
        dhc.Br(),dhc.Br(),
        dhc.H4("Statistics"),
        dcc.Graph(figure=ffig),
        dhc.Br(),
        dhc.Img(src=img_src),
        dhc.Br(),dhc.Br(),
    ]

    return ret_val

