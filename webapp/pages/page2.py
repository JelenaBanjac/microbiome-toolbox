import sys
sys.path.append("../..")
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import dash
from microbiome.data_preparation import *
from microbiome.data_analysis import *
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

                        dhc.H3("Data Analysis & Exploration"),
                        dhc.Br(),

                        dcc.Markdown('''
                        Some of the methods for data analysis and exploration provided are:
                        - Sampling statistics
                        - Heatmap of taxa abundances w.r.t. time
                        - Taxa abundance errorbars
                        - Dense longitudinal data
                        - Shannon diversity index and Simpson dominance index
                        - Embeddings (different algorithms that we used in 2D and 3D space) with interactive selection and reference analysis.
                        ''', style={'textAlign': 'left',}),

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
    # dhc.Div(id='page-2-display-value-0', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-0', children='none', hidden=True),                
    dhc.Div(id='task-status-2-0', children='task-status-2-0', hidden=True),                
    dcc.Interval(id='task-interval-2-0', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-0', children=loading_img),
    dhc.Div(id='page-2-display-value-0'),
    
    
    # Abundance plot in general
    dhc.Hr(),
    dhc.H4("Taxa Abundances"),
    # dhc.Div(id='page-2-display-value-1', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-1', children='none', hidden=True),                
    dhc.Div(id='task-status-2-1', children='task-status-2-1', hidden=True),                
    dcc.Interval(id='task-interval-2-1', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-1', children=loading_img),
    dhc.Div(id='page-2-display-value-1'),
    
    # Sampling statistics
    dhc.Hr(),
    dhc.H4("Sampling Statistics"),
    # dhc.Div(id='page-2-display-value-2', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-2', children='none', hidden=True),                
    dhc.Div(id='task-status-2-2', children='task-status-2-2', hidden=True),                
    dcc.Interval(id='task-interval-2-2', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-2', children=loading_img),
    dhc.Div(id='page-2-display-value-2'),
    
    # Heatmap
    dhc.Hr(),
    dhc.H4("Taxa Abundances Heatmap"),
    # dhc.Div(id='page-2-display-value-3', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-3', children='none', hidden=True),                
    dhc.Div(id='task-status-2-3', children='task-status-2-3', hidden=True),                
    dcc.Interval(id='task-interval-2-3', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-3', children=loading_img),
    dhc.Div(id='page-2-display-value-3'),
    
    # Shannon's diversity index and Simpson's dominace
    # dhc.Hr(),
    # dhc.H4("Diversity"),
    # dhc.Div(id='page-2-display-value-4', children=loading_img),

    # Dense longitudinal data
    dhc.Hr(),
    dhc.H4("Dense Longitudinal Data"),
    # dhc.Div(id='page-2-display-value-5', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-4', children='none', hidden=True),                
    dhc.Div(id='task-status-2-4', children='task-status-2-4', hidden=True),                
    dcc.Interval(id='task-interval-2-4', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-4', children=loading_img),
    dhc.Div(id='page-2-display-value-4'),
    
    # Embedding in 2D
    dhc.Hr(),
    dhc.H4("Embedding in 2D space"),
    # dhc.Div(id='page-2-display-value-6', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-5', children='none', hidden=True),                
    dhc.Div(id='task-status-2-5', children='task-status-2-5', hidden=True),                
    dcc.Interval(id='task-interval-2-5', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-5', children=loading_img),
    dhc.Div(id='page-2-display-value-5'),
    
    # Embedding in 3D
    dhc.Hr(),
    dhc.H4("Embedding in 3D space"),
    # dhc.Div(id='page-2-display-value-7', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-6', children='none', hidden=True),                
    dhc.Div(id='task-status-2-6', children='task-status-2-6', hidden=True),                
    dcc.Interval(id='task-interval-2-6', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-6', children=loading_img),
    dhc.Div(id='page-2-display-value-6'),
    
    # Embedding in 2D, interactive
    dhc.Hr(),
    dhc.H4("Embedding in 2D space - Interactive Analysis"),
    dcc.Markdown('''To use an interactive option:   
    - click on `Lasso Select` on the plot toolbox,  
    - select the samples you want to group,  
    - wait for the explainatory information to load (with confusion matrix).  
    ''', style={'textAlign': 'left',}),
    # dhc.Div(id='page-2-display-value-8', children=loading_img),
    dhc.Br(),
    dhc.Div(id='task-id-2-7', children='none', hidden=True),                
    dhc.Div(id='task-status-2-7', children='task-status-2-7', hidden=True),                
    dcc.Interval(id='task-interval-2-7', interval=250, n_intervals=0),
    dhc.Div(id='spinner-2-7', children=loading_img),
    dhc.Div(id='page-2-display-value-7'),
    
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
    Output('page-2-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content



for idx in range(8):
    # Don't touch this:
    @app.callback(Output(f'task-interval-2-{idx}', 'interval'),
                [Input(f'task-id-2-{idx}', 'children'),
                Input(f'task-status-2-{idx}', 'children')])
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
    @app.callback(Output(f'spinner-2-{idx}', 'hidden'),
                [Input(f'task-interval-2-{idx}', 'n_intervals'),
                Input(f'task-status-2-{idx}', 'children')])
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
    @app.callback(Output(f'task-status-2-{idx}', 'children'),
                [Input(f'task-interval-2-{idx}', 'n_intervals'),
                Input(f'task-id-2-{idx}', 'children')])
    def update_task_status(n_intervals, task_id):
        """This callback is triggered by the Interval clock and task-id .  It checks the task
        status in Celery and returns the status to an invisible div"""
        return str(AsyncResult(task_id, app=celery_app).state)


    @app.callback(
        Output(f'page-2-display-value-{idx}', 'children'),
        [Input(f'task-status-2-{idx}', 'children')],
        [State(f'task-id-2-{idx}', 'children')])
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
    Output(f'task-id-2-0', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-0', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_20").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-1', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-1', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_21").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-2', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-2', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_22").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-3', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-3', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_23").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-4', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-4', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_24").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-5', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-5', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_25").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-6', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-6', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_26").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)

@app.callback(
    Output(f'task-id-2-7', 'children'),
    [Input(f'session-id', 'children')],
    [State(f'task-id-2-7', 'children')])
def start_task_callback(session_id, task_id):
    # Don't touch this:
    slogger('start_task_callback', 'task_id={}, session_id={}'.format(task_id, session_id))

    # Put search function in the queue and return task id
    # (arguments must always be passed as a list)
    slogger('start_task_callback', 'query accepted and applying to Celery')
    
    task = eval(f"query_mt_27").apply_async([session_id])
    # don't touch this:
    slogger('start_Task_callback', 'query is on Celery, task-id={}'.format(task.id))
    return str(task.id)


##############################3


@app.callback(Output("graph-info", "children"), 
             [Input("graph", "selectedData"), 
             Input('session-id', 'children')],
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
        ffig, img_src, stats = two_groups_analysis(df, bacteria_names, references_we_compare="selected", test_size=0.5, n_splits=5, nice_name=nice_name, style="dot", show=False, website=True, layout_height=1000, layout_width=1000, max_display=20);

        stats = stats.split("\n")   #style={ 'verticalAlign':'left', 'textAlign': 'left',}
        stats_div = dhc.Div(children=[
            dhc.Br(), 
            dhc.H5("Groups discrimination performance results"), 
            dhc.Br(),dhc.Br(),
            dhc.P("The ideal separation between two groups (reference vs. non-reference) will have 100% of values detected on the second diagonal. This would mean that the two groups can be easily separated knowing their taxa abundamces and metadata information."),]+
            [dcc.Markdown(r) for r in stats]) 
        img_src.update_layout(height=400, width=400)
        confusion_matrix = dcc.Graph(figure=img_src)

        statistics_part = dbc.Container(
            dbc.Row([
                dbc.Col(stats_div),
                dbc.Col(confusion_matrix)
            ])
        )

        # Update parcats colors
        # new_color = np.zeros(len(fig["data"][1]["line"]["color"]), dtype='uint8')
        # new_color[selection] = 1
        # fig["data"][1]["line"]["color"] = new_color
        ret_val = [
            dcc.Graph(figure=t),
            dcc.Graph(figure=ffig),
            dhc.Br(),
            statistics_part,
            dhc.Br(),
        ]
    else:
        ret_val = []
    return ret_val

