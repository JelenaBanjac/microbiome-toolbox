import sys
sys.path.append("../..")
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import sys
from celery.result import AsyncResult
from tasks import *

from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img

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
    dhc.Div(id='task-id-3-0', children='none', hidden=True),                
    dhc.Div(id='task-status-3-0', children='task-status-3-0', hidden=True),                
    dcc.Interval(id='task-interval-3-0', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-0', children=loading_img),
    dhc.Div(id='page-3-display-value-0'),
    
    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Longitudinal Subject's Data"),
    dhc.Br(),
    dhc.Div(id='task-id-3-1', children='none', hidden=True),                
    dhc.Div(id='task-status-3-1', children='task-status-3-1', hidden=True),                
    dcc.Interval(id='task-interval-3-1', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-1', children=loading_img),
    dhc.Div(id='page-3-display-value-1'),
    

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Universality: Linear Difference between Group Trajectories"),
    dhc.Br(),
    dhc.Div(id='task-id-3-2', children='none', hidden=True),                
    dhc.Div(id='task-status-3-2', children='task-status-3-2', hidden=True),                
    dcc.Interval(id='task-interval-3-2', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-2', children=loading_img),
    dhc.Div(id='page-3-display-value-2'),
    

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Universality: Nonlinear Difference between Group Trajectories"),
    dhc.Br(),
    dhc.Div(id='task-id-3-3', children='none', hidden=True),                
    dhc.Div(id='task-status-3-3', children='task-status-3-3', hidden=True),                
    dcc.Interval(id='task-interval-3-3', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-3', children=loading_img),
    dhc.Div(id='page-3-display-value-3'),
    
    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Reference vs. Non-reference Longitudinal Trajectories"),
    dhc.Br(),
    dhc.Div(id='task-id-3-4', children='none', hidden=True),                
    dhc.Div(id='task-status-3-4', children='task-status-3-4', hidden=True),                
    dcc.Interval(id='task-interval-3-4', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-4', children=loading_img),
    dhc.Div(id='page-3-display-value-4'),
    

    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Differentiation: Linear Reference vs. Non-reference Difference Between Trajectories"),
    dhc.Br(),
    dhc.Div(id='task-id-3-5', children='none', hidden=True),                
    dhc.Div(id='task-status-3-5', children='task-status-3-5', hidden=True),                
    dcc.Interval(id='task-interval-3-5', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-5', children=loading_img),
    dhc.Div(id='page-3-display-value-5'),
    
    dhc.Br(),
    dhc.Hr(),
    dhc.H4("Differentiation: Nonlinear (Spline) Reference vs. Non-reference Difference Between Trajectories"),
    dhc.Br(),
    dhc.Div(id='task-id-3-6', children='none', hidden=True),                
    dhc.Div(id='task-status-3-6', children='task-status-3-6', hidden=True),                
    dcc.Interval(id='task-interval-3-6', interval=250, n_intervals=0),
    dhc.Div(id='spinner-3-6', children=loading_img),
    dhc.Div(id='page-3-display-value-6'),
    
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


for idx in range(7):
    # Don't touch this:
    @app.callback(Output(f'task-interval-3-{idx}', 'interval'),
                [Input(f'task-id-3-{idx}', 'children'),
                Input(f'task-status-3-{idx}', 'children')])
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
    @app.callback(Output(f'spinner-3-{idx}', 'hidden'),
                [Input(f'task-interval-3-{idx}', 'n_intervals'),
                Input(f'task-status-3-{idx}', 'children')])
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
    @app.callback(Output(f'task-status-3-{idx}', 'children'),
                [Input(f'task-interval-3-{idx}', 'n_intervals'),
                Input(f'task-id-3-{idx}', 'children')])
    def update_task_status(n_intervals, task_id):
        """This callback is triggered by the Interval clock and task-id .  It checks the task
        status in Celery and returns the status to an invisible div"""
        return str(AsyncResult(task_id, app=celery_app).state)


    @app.callback(
        Output(f'page-3-display-value-{idx}', 'children'),
        [Input(f'task-status-3-{idx}', 'children')],
        [State(f'task-id-3-{idx}', 'children')])
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
    Output(f'task-id-3-0', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-0', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_30").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-1', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-1', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_31").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-2', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-2', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_32").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-3', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-3', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_33").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-4', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-4', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_34").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-5', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-5', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_35").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-3-6', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-3-6', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_36").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, plot=3{} task-id={}'.format(idx, task.id))
    return str(task.id)

