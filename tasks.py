import sys
sys.path.append("../..")
from microbiome.data_preparation import *
import time
import os
import config
from index import UPLOAD_FOLDER_ROOT
import pandas as pd

from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names
from microbiome.trajectory import plot_trajectory, train, plot_2_trajectories
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
import sys
from microbiome.postprocessing import *
from microbiome.data_preparation import *
from microbiome.longitudinal_anomaly_detection import *
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names
from microbiome.data_analysis import *
from sklearn import decomposition
import dash_table

# This is a very simple function for logging messages in a Terminal in near-realtime from a web application

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

# Initialize Celery - you don't need to change anything here:
from celery import Celery
#redis_url = "redis://0.0.0.0:6379" # os.environ['REDIS_URL']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

slogger('tasks.py', 'declare celery_app: redis_url={}'.format(redis_url))
celery_app = Celery('query', backend=redis_url, broker=redis_url, 
    accept_content=['pickle'], task_serializer='pickle', result_serializer='pickle')
slogger('tasks.py', 'celery_app declared successfully')

# This is the function that will be run by Celery
# You need to change the function declaration to include all the
# arguments that the app will pass to the function:

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

@celery_app.task(bind=True, serializer='pickle')
def query_mt_30(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(
        df, bacteria_fun=lambda x: x.startswith("bacteria_"))

    if max(df.age_at_collection.values) < 100:
        plateau_area_start = None  # 45
        time_unit_size = 1
        time_unit_name = "days"
        limit_age = 60
    else:
        plateau_area_start = None  # 700
        time_unit_size = 30
        time_unit_name = "months"
        limit_age = 750

    try:
        estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor,
                          parameters=parameters, param_grid=param_grid, n_splits=2, file_name=None)

        # # healthy unseen data - Test-1
        # val1 = df[df.classification_dataset_type=="Test-1"]
        # # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
        # other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
        # # unhealthy unseen data - Test2
        # val2 =  df[df.classification_dataset_type=="Test-2"]
        # healthy unseen data - Test-1
        val1 = df[df.dataset_type == "Validation"]
        # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
        other = df[df.dataset_type == "Test"]
        # unhealthy unseen data - Test2
        #val2 =  df[df.classification_dataset_type=="Test-2"]

        fig1,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group=None, nonlinear_difference=True,
                                                    start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True)
    except Exception as e:
        df = None

    if df is not None:
        ret_val = [
            dcc.Graph(figure=fig1),
        ]
    else:
        ret_val = dhc.Div([])
    
    print(ret_val)
    # Return results for display
    slogger('query', 'return results 3.0.')
    return ret_val



@celery_app.task(bind=True, serializer='pickle')
def query_mt_31(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

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
    
    # Return results for display
    slogger('query', 'return results  3.1.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_32(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

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
    
    # Return results for display
    slogger('query', 'return results  3.2.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_33(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

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

    # Return results for display
    slogger('query', 'return results 3.3.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_34(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

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

    # Return results for display
    slogger('query', 'return results 3.4.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_35(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

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

    # Return results for display
    slogger('query', 'return results 3.5.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_36(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results 3.6.')
    return ret_val

#####################################


@celery_app.task(bind=True, serializer='pickle')
def query_mt_60(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val

########################################


# @celery_app.task(bind=True, serializer='pickle')
# def query_mt_20(self, session_id):
#     task_id = self.request.id
#     slogger('query', 'query in progress, task_id={}'.format(task_id))
#     # Don't touch this:
#     self.update_state(state='PROGRESS')
#     # a short dwell is necessary for other async processes to catch-up
#     time.sleep(1.5)

#     # read arguments
#     slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
#     # Change all of this to whatever you want:
#     df = read_dataframe(session_id, None)

#     ret_val = dhc.Div([])
#     if df is not None:
#         ret_val =  [
            
#             dash_table.DataTable(
#                         id='upload-datatable',
#                         columns=[{"name": i, "id": i} for i in df.columns],
#                         data=df.to_dict('records'),
#                         style_data={
#                             'width': '{}%'.format(max(df.columns, key=len)),
#                             'minWidth': '50px',
#                             'maxWidth': '500px',
#                         },
#                         style_table={
#                             'height': 300, 
#                             'overflowX': 'auto'
#                         }  
#                     ),
#             dhc.Br()
#             ]

#     # Return results for display
#     slogger('query', 'return results query_mt_20')
#     return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_20(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
    print(f"session_id --- cekery app {session_id} {UPLOAD_FOLDER_ROOT}")

    # Change all of this to whatever you want:

    df = read_dataframe(session_id, None)
    bacteria_names = get_bacteria_names(
        df, bacteria_fun=lambda x: x.startswith("bacteria_"))

    if max(df.age_at_collection.values) < 100:
        plateau_area_start = None  # 45
        time_unit_size = 1
        time_unit_name = "days"
        limit_age = 60
    else:
        plateau_area_start = None  # 700
        time_unit_size = 30
        time_unit_name = "months"
        limit_age = 750

    try:
        estimator = train(df, feature_cols=bacteria_names, Regressor=Regressor,
                          parameters=parameters, param_grid=param_grid, n_splits=2, file_name=None)

        # # healthy unseen data - Test-1
        # val1 = df[df.classification_dataset_type=="Test-1"]
        # # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
        # other = df[df.classification_dataset_type.isin(["Train-2","Test-2"])]
        # # unhealthy unseen data - Test2
        # val2 =  df[df.classification_dataset_type=="Test-2"]
        # healthy unseen data - Test-1
        val1 = df[df.dataset_type == "Validation"]
        # unhealthy unseen data - Test2 & unhealthy seen data - Train-2
        other = df[df.dataset_type == "Test"]
        # unhealthy unseen data - Test2
        #val2 =  df[df.classification_dataset_type=="Test-2"]

        fig1,  mae, r2, pi_median = plot_trajectory(estimator=estimator, df=val1, feature_cols=bacteria_names, df_other=None, group=None, nonlinear_difference=True,
                                                    start_age=0, limit_age=limit_age, plateau_area_start=plateau_area_start, time_unit_size=time_unit_size, time_unit_name=time_unit_name, website=True)
    except Exception as e:
        df = None

    if df is not None:
        ret_val = [
            dcc.Graph(figure=fig1),
        ]
    else:
        ret_val = dhc.Div([])
    
    print(ret_val)
    # Return results for display
    slogger('query', 'return results 3.0.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_21(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
    df = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:
        
        
        bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
        print("BACTERIA", bacteria_names)
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

    # Return results for display
    slogger('query', 'return results query_mt_21')
    return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_22(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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
    # Return results for display
    slogger('query', 'return results query_mt_22')
    return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_23(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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
    # Return results for display
    slogger('query', 'return results query_mt_23')
    return ret_val

# sklearn bio is not supported on Heroku
# @celery_app.task(bind=True, serializer='pickle')
# def query_mt_24(self, session_id):
#     task_id = self.request.id
#     slogger('query', 'query in progress, task_id={}'.format(task_id))
#     # Don't touch this:
#     self.update_state(state='PROGRESS')
#     # a short dwell is necessary for other async processes to catch-up
#     time.sleep(1.5)

#     # read arguments
#     slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
#     # Change all of this to whatever you want:
#      df = read_dataframe(session_id, None)

#     ret_val = dhc.Div([])
#     if df is not None:

#         bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))

#         if max(df.age_at_collection.values) < 100:
#             time_unit_name="days"
#             time_unit_size=1
#         else:
#             time_unit_name="months"
#             time_unit_size=30
        
#         fig1 = plot_diversity(df, bacteria_names, diversity="shannon", group="group", time_unit_name=time_unit_name, time_unit_size=time_unit_size, layout_height=800, layout_width=1000, website=True)
#         fig2 = plot_diversity(df, bacteria_names, diversity="simpson", group="group", time_unit_name=time_unit_name, time_unit_size=time_unit_size, layout_height=800, layout_width=1000, website=True)

#         ret_val = [
            
#             dcc.Graph(figure=fig1),
#             dhc.Br(),
#             dcc.Graph(figure=fig2),
#             dhc.Br(),
#         ]

#     # Return results for display
#     slogger('query', 'return results 6.0.')
#     return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_24(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results query_mt_25')
    return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_25(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results query_mt_26')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_26(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results query_mt_27')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_27(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    # Return results for display
    slogger('query', 'return results query_mt_28')
    return ret_val



######################################
@celery_app.task(bind=True, serializer='pickle')
def query_mt_40(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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
    
    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=None, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    
    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [dhc.Hr(),
                    #dhc.H4("Important Bacteria w.r.t. Time"),
                    dcc.Graph(figure=fig),
                    dhc.Br(),
                    ]
    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val

######################################
@celery_app.task(bind=True, serializer='pickle')
def query_mt_50(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    fig, traj_x, traj_pi, traj_mean = plot_importance_boxplots_over_age(estimator, val1, bacteria_names, nice_name=nice_name, 
                                                                units=units, patent=False, highlight_outliers=outliers, df_new=None, time_unit_size=time_unit_size, time_unit_name=time_unit_name, 
                                                                box_height=box_height, plateau_area_start=plateau_area_start, longitudinal_mode="markers", longitudinal_showlegend=False, 
                                                                fillcolor_alpha=0.2, website=True);
    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
                    dcc.Graph(figure=fig),
                    ]
    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_51(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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
    
    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
                    dcc.Graph(figure=fig),
                    ]
    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val

@celery_app.task(bind=True, serializer='pickle')
def query_mt_52(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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
    
    ret_val = dhc.Div([])
    if df is not None:
        ret_val =  [
                    dcc.Graph(figure=fig),
                    ]
    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val


@celery_app.task(bind=True, serializer='pickle')
def query_mt_53(self, session_id):
    task_id = self.request.id
    slogger('query', 'query in progress, task_id={}'.format(task_id))
    # Don't touch this:
    self.update_state(state='PROGRESS')
    # a short dwell is necessary for other async processes to catch-up
    time.sleep(1.5)

    # read arguments
    slogger('query', 'query in progress, task_id={}, session_id={}'.format(task_id, session_id))
   
    # Change all of this to whatever you want:
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

    ffig, img_src, stats = two_groups_analysis(val1, bacteria_names, references_we_compare="selected", test_size=0.5, n_splits=2, nice_name=nice_name, style="dot", 
                                        show=False, website=True, layout_height=1000, layout_width=1000, max_display=50);
    
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
        statistics_part,
        dhc.Br(),dhc.Br(),
    ]
    # Return results for display
    slogger('query', 'return results 6.0.')
    return ret_val