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
from microbiome.variables import *
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories
from microbiome.postprocessing import plot_importance_boxplots_over_age
from celery.result import AsyncResult
from tasks import *
from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Bacteria Importance with Time"),
                        dhc.Br(),
                        dcc.Markdown('''
                        * Importance of different bacteria and their abundances across time boxes (reference data) - resolution (i.e. box width) can be chosen (in the dashboard it is fixed)
                            - bacteria height represents its importance in that time block
                            - values in the box represent average ± STD of its abundance in that time block (Note: we tested mean, geometric mean, and robust mean, and median represented data the best for our data. In this toolbox, we have a support for any *average function* a user may want.)
                            - total height of the box is fixed in all time blocks
                            - can choose number of important bacteria for time interval
                        * Importance of different bacteria and their abundances across time boxes on non-healthy data (but model trained on healthy samples)
                        ''', style={'textAlign': 'left',}),
                        dcc.Markdown("The examples that are not in the dashboard can be found in the `microbiome-toolbox` repository.", style={'textAlign': 'left',}),
                        
                        dhc.Br(),
                        dhc.Div(id="page-4-main"),
                        
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
    # Abundance plot in general
    # dhc.Div(id='page-4-display-value-0', children=loading_img),
    dhc.Div(id='task-id-4-0', children='none', hidden=True),                
    dhc.Div(id='task-status-4-0', children='task-status-4-0', hidden=True),                
    dcc.Interval(id='task-interval-4-0', interval=250, n_intervals=0),
    dhc.Div(id='spinner-4-0', children=loading_img),
    dhc.Div(id='page-4-display-value-0'),
    
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
    Output('page-4-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


# Don't touch this:
@app.callback(Output(f'task-interval-4-0', 'interval'),
            [Input(f'task-id-4-0', 'children'),
            Input(f'task-status-4-0', 'children')])
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
@app.callback(Output(f'spinner-4-0', 'hidden'),
            [Input(f'task-interval-4-0', 'n_intervals'),
            Input(f'task-status-4-0', 'children')])
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
@app.callback(Output(f'task-status-4-0', 'children'),
            [Input(f'task-interval-4-0', 'n_intervals'),
            Input(f'task-id-4-0', 'children')])
def update_task_status(n_intervals, task_id):
    """This callback is triggered by the Interval clock and task-id .  It checks the task
    status in Celery and returns the status to an invisible div"""
    return str(AsyncResult(task_id, app=celery_app).state)


@app.callback(
    Output(f'page-4-display-value-0', 'children'),
    [Input(f'task-status-4-0', 'children')],
    [State(f'task-id-4-0', 'children')])
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
    Output(f'task-id-4-0', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-4-0', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_40").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

