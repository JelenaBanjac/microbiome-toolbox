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
from microbiome.helpers import get_bacteria_names
from microbiome.postprocessing import outlier_intervention, plot_importance_boxplots_over_age
from microbiome.trajectory import train
from microbiome.variables import Regressor, param_grid, parameters
from celery.result import AsyncResult
from tasks import query_mt_60, celery_app
from index import app, UPLOAD_FOLDER_ROOT, loading_img
import gc


def slogger(origin, message):
    """Log a message in the Terminal
    Args:
        str: The origin of the message, e.g. the name of a function
        str: The message itself, e.g. 'Query the database'
    Returns:
        None
    """
    print('\033[94m[SLOG] \u001b[36m|  \033[1m\u001b[33m{} \u001b[0m{}'.format(origin.upper(), message))
    sys.stdout.flush()


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
    # dhc.Div(id='page-6-display-value-0-hidden', hidden=True),
    dhc.Div(id='task-id-6-0', children='none', hidden=True),                
    dhc.Div(id='task-status-6-0', children='task-status-6-0', hidden=True),                
    dcc.Interval(id='task-interval-6-0', interval=250, n_intervals=0),
    dhc.Div(id='spinner-6-0', children=loading_img),
    dhc.Div(id='page-6-display-value-0'),
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
    del df
    gc.collect()
    return page_content


# Don't touch this:
@app.callback(Output(f'task-interval-6-0', 'interval'),
            [Input(f'task-id-6-0', 'children'),
            Input(f'task-status-6-0', 'children')])
def toggle_interval_speed(task_id, task_status):
    """This callback is triggered by changes in task-id and task-status divs.  It switches the 
    page refresh interval to fast (1 sec) if a task is running, or slow (24 hours) if a task is 
    pending or complete."""
    if task_id == 'none':
        slogger('toggle_interval_speed', 'no task-id --> slow refresh')
        return 24*60*60*1000
    if task_id != 'none' and (task_status in ['SUCCESS', 'FAILURE']):
        slogger('toggle_interval_speed', 'task-id is {} and status is {} --> slow refresh'.format(task_id, task_status))
        return 24*60*60*1000
    else:
        slogger('toggle_interval_speed', 'task-id is {} and status is {} --> fast refresh'.format(task_id, task_status))
        return 1000


# Don't touch this:
@app.callback(Output(f'spinner-6-0', 'hidden'),
            [Input(f'task-interval-6-0', 'n_intervals'),
            Input(f'task-status-6-0', 'children')])
def show_hide_spinner(n_intervals, task_status):
    """This callback is triggered by then Interval clock and checks the task progress
    via the invisible div 'task-status'.  If a task is running it will show the spinner,
    otherwise it will be hidden."""
    if task_status == 'PROGRESS':
        slogger('show_hide_spinner', 'show spinner')
        return False
    else:
        slogger('show_hide_spinner', 'hide spinner because task_status={}'.format(task_status))
        return True


# Don't touch this:
@app.callback(Output(f'task-status-6-0', 'children'),
            [Input(f'task-interval-6-0', 'n_intervals'),
            Input(f'task-id-6-0', 'children')])
def update_task_status(n_intervals, task_id):
    """This callback is triggered by the Interval clock and task-id .  It checks the task
    status in Celery and returns the status to an invisible div"""
    return str(AsyncResult(task_id, app=celery_app).state)


@app.callback(
    Output(f'page-6-display-value-0', 'children'),
    [Input(f'task-status-6-0', 'children')],
    [State(f'task-id-6-0', 'children')])
def display_value(task_status, task_id):
    if task_status == 'SUCCESS':
        # Fetch results from Celery and forget the task
        slogger('get_results', 'retrieve results for task-id {} from Celery'.format(task_id))
        result = AsyncResult(task_id, app=celery_app).result    # fetch results
        forget = AsyncResult(task_id, app=celery_app).forget()  # delete from Celery
        # Display a message if their were no hits
        if result == [{}]:
            return ["We couldn\'t find any results.  Try broadening your search."]
        # Otherwise return the populated DataTable
        return result

    else:
        # don't display any results
        return []

@app.callback(
    Output(f'task-id-6-0', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-6-0', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = query_mt_60.apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)




# @app.callback(
#     Output('page-6-display-value-0-hidden', 'children'),
#     [Input('session-id', 'children')])
# def display_value(session_id):
#     df = read_dataframe(session_id, None)
#     bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
#     nice_name = lambda x: x[9:].replace("_", " ")
#     if max(df.age_at_collection.values) < 100:
#         plateau_area_start=None #45
#         time_unit_size=1
#         time_unit_name="days"
#         box_height = None
#         units = [20, 20, 20] 
#     else:
#         plateau_area_start=None  #700
#         time_unit_size=30
#         time_unit_name="months"
#         box_height = None
#         units = [90, 90, 90, 90, 90, 90]

#     estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor, parameters=parameters, param_grid=param_grid, n_splits=2)

#     # healthy unseen data - Test-1
#     val1 = df[df.classification_dataset_type=="Test-1"]
#     # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
#     other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
#     # unhealthy unseen data - Test2
#     val2 =  df[df.classification_dataset_type=="Test-2"]

#     #fig, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, units=units, start_age=0, patent=False, highlight_outliers=False, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, box_height=box_height, file_name=None, plateau_area_start=None, longitudinal_mode=None, longitudinal_showlegend=False, fillcolor_alpha=0.2, website=True);
    

#     outliers = pi_anomaly_detection(estimator=estimator, df_all=val1, feature_columns=bacteria_names, degree=2)
#     #outlier_id = outliers[0]

#     fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
#                                                                 units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
#                                                                 box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
#                                                                 fillcolor_alpha=0.2, website=True);
    
#     ret_val = dhc.Div([])
#     if df is not None:
#         ret_val =  [
#                     dcc.Graph(figure=fig, id="graph2"),
#                     dhc.Br(),dhc.Br(),
#                     dhc.Div(id="graph2-info"),
#                     ]

#     return ret_val



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

