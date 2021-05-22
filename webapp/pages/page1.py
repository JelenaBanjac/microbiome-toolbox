import sys

from dash_core_components.Markdown import Markdown
sys.path.append("/home/jelena/Desktop/microbiome-toolbox")

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as dhc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import numpy as np
import sys
from microbiome.data_preparation import *
from microbiome.helpers import get_bacteria_names, two_groups_analysis
import dash_table

from index import app, cache, UPLOAD_FOLDER_ROOT, loading_img


layout = dhc.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dcc.Link('Back', href='/'),

                        dhc.H3("Reference Definition & Statistics"),
                        dhc.Br(),

                        dcc.Markdown('''
                        There are two ways to define the reference set in the dataset:  
                            1. _predefined by user (on raw data)_: all samples that belong to the reference are specified by user in the uploaded dataset (with the `True` value in the `reference_group` column). 
                            Other samples are considered to be non-reference samples.  
                            2. _unsupervised anomaly detection (on raw data)_ where we don't feed the algorithm about our true differentiation:
                            Performs novelty and outlier detection -- use the user's reference definition as a start and decide whether a new observation from other belongs to the reference or not. 
                            For the metric we use [Bray-Curtis distance](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity).
                        The column for this property is called `reference_group` and it contails only `True`/`False` values.

                        Bellow we also analyse the features important in each of the groups. 
                        To find the features that differentiate the two groups (reference vs non-reference group), we train the binary classification model (using supervised ensemble methods `XGBClassifier` or `RandomForestClassifier`) with confusion matrix.
                        The confusion matrix enables the insight on how good the separation between the two groups is.
                        ''', style={'textAlign': 'left',}),

                        dhc.Br(),
                        dhc.Div(id="page-1-main"),
                        
                        

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
    # Important features
    dhc.Hr(),
    dhc.H4("Table of Reference Group Samples"),
    dhc.Div(id='page-1-display-value-0', children=loading_img),

    dhc.Hr(),
    dhc.H4("(1) Rest of the samples put into Non-Reference Group"),
    dhc.Div(id='page-1-display-value-1', children=loading_img),

    dhc.Hr(),
    dhc.H4("(2) Novelty detection with respect to defined Reference Group"),
    dhc.Div(id='page-1-display-value-2', children=loading_img),
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

        df.sampleID = df.sampleID.astype(str)
        df.subjectID = df.subjectID.astype(str)

        df = df.fillna(0)
        df = df.convert_dtypes()

        feature_columns = df.columns[df.columns.str.startswith("bacteria_")].tolist()
        metadata_columns = df.columns[df.columns.str.startswith("meta_")].tolist()
        id_columns = df.columns[df.columns.str.startswith("id_")].tolist()
        other_columns = df.columns[(~df.columns.str.startswith("bacteria_"))&(~df.columns.str.startswith("meta_"))&(~df.columns.str.startswith("id_"))].tolist()

        str_cols_meta = list(set(df[metadata_columns].columns[df[metadata_columns].dtypes=="string"]))
        obj_cols_meta = list(set(df[metadata_columns].columns[df[metadata_columns].dtypes=="object"]))
        df = pd.get_dummies(df, columns=str_cols_meta+obj_cols_meta)

        # update
        feature_columns = df.columns[df.columns.str.startswith("bacteria_")].tolist()
        metadata_columns = df.columns[df.columns.str.startswith("meta_")].tolist()
        id_columns = df.columns[df.columns.str.startswith("id_")].tolist()
        other_columns = df.columns[(~df.columns.str.startswith("bacteria_"))&(~df.columns.str.startswith("meta_"))&(~df.columns.str.startswith("id_"))].tolist()

        #meta_and_feature_columns = feature_columns+metadata_columns

    else:
        print('\tfilename not yet exists', filename)
        df = None
        feature_columns = []
        metadata_columns = []
        id_columns = []
        other_columns = []
        print('** No data, df empty **')

    return df, feature_columns, metadata_columns, id_columns, other_columns

@app.callback(
    Output('page-1-main', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df = read_dataframe(session_id, None)
    
    if df is None:
        return dhc.Div(dbc.Alert(["You refreshed the page or were idle for too long so data got lost. Please go ", dcc.Link('back', href='/'), " and upload again."], color="warning"))

    return page_content


@app.callback(
    Output('page-1-display-value-0', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df, *_ = read_dataframe(session_id, None)

    ret_val = dhc.Div([])
    if df is not None:
        df_ref = df[df.reference_group==True]
        
        ret_val =  [
            
            dash_table.DataTable(
                        id='upload-datatable',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df_ref.to_dict('records'),
                        style_data={
                            'width': '{}%'.format(max(df.columns, key=len)),
                            'minWidth': '50px',
                            'maxWidth': '500px',
                        },
                        style_table={
                            'height': 300, 
                            'overflowX': 'auto'
                        }  
                    ),
            dhc.Br()
            ]
    return ret_val


@app.callback(
    Output('page-1-display-value-1', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df, feature_columns, metadata_columns, id_columns, other_columns = read_dataframe(session_id, None)
    # feature_columns = df.columns[df.columns.str.startswith("bacteria_")].tolist()
    # metadata_columns = df.columns[df.columns.str.startswith("meta_")].tolist()
    # id_columns = df.columns[df.columns.str.startswith("id_")].tolist()
    # other_columns = df.columns[(~df.columns.str.startswith("bacteria_"))&(~df.columns.str.startswith("meta_"))&(~df.columns.str.startswith("id_"))].tolist()

    # df = df.convert_dtypes() 
    # id_cols = list(df.columns[df.columns.str.contains("id", case=False)&(df.columns.str.len()<20)].values)
    # cols_to_ignore = [ 'dataset_type', 'dataset_type_classification', 'classification_dataset_type', 'classification_label' ]  
    # str_cols = list(set(df.columns[df.dtypes=="string"]) - set(id_cols + cols_to_ignore))
    # df = pd.get_dummies(df, columns=str_cols)
    # print(df.columns.values)
    
    #bacteria_names = get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("bacteria_"))
    
    #feature_columns = set(df.columns) - set(id_cols + cols_to_ignore)
    print("feature_columns", feature_columns)

    # reference_group column needs 2 values: True and False
    # here we populate all non-True to False
    references_we_compare = 'reference_group'
    df[references_we_compare] = df[references_we_compare].apply(lambda x: True if str(x)=='True' else False)
    before_ref = df[references_we_compare].value_counts()

    fig1, img_src, stats  = two_groups_analysis(df, feature_columns, references_we_compare=references_we_compare,nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="dot", show=False, website=True, layout_height=800, layout_width=1000)
    fig2, _, _            = two_groups_analysis(df, feature_columns, references_we_compare=references_we_compare, nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="hist", show=False, website=True, layout_height=800, layout_width=1000)

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

    ret_val = []
    if df is not None:
        ret_val =  [
            dhc.P("Reference vs. Non-reference count: "+str(before_ref)),
            dhc.Br(),
            dcc.Graph(figure=fig1),
            dhc.Br(),
            dcc.Graph(figure=fig2),
            dhc.Br(),
            statistics_part,
            dhc.Br(),dhc.Br(),
        ]

    return ret_val


@app.callback(
    Output('page-1-display-value-2', 'children'),
    Input('session-id', 'children'))
def display_value(session_id):
    df, feature_columns, metadata_columns, id_columns, other_columns = read_dataframe(session_id, None)
    meta_and_feature_columns = feature_columns + metadata_columns

    print("feature_columns", len(feature_columns))
    print("metadata_columns", len(metadata_columns))
    print("meta_and_feature_columns", len(meta_and_feature_columns))
    
    # reference_group column needs 2 values: True and False
    # here we populate all non-True to False
    references_we_compare = 'reference_group'
    df[references_we_compare] = df[references_we_compare].apply(lambda x: True if str(x)=='True' else False)
    before_ref = df[references_we_compare].value_counts()

    #[25, 30, 35, 40]
    df_stats, fig = gridsearch_novelty_detection_parameters(df, "n_neighbors", [1, 2, 3], feature_columns, metadata_columns, meta_and_feature_columns, num_runs=3, website=True, layout_settings=None)
    df_stats_best = df_stats.groupby(["parameter", "columns_type"]).agg(np.mean)
    #best_params = df_stats_best.iloc[df_stats_best[['accuracy']].idxmax()]
    best_param, best_features_type = int(df_stats_best[['accuracy']].idxmax()['accuracy'][0]), df_stats_best[['accuracy']].idxmax()['accuracy'][1]

    best_features = metadata_columns if best_features_type=="meta" else feature_columns if best_features_type=="taxa" else meta_and_feature_columns 
    best_params = dict(metric='braycurtis', n_neighbors=best_param)
    df = df.assign(reference_group=update_reference_group_with_novelty_detection(df, meta_and_feature_columns, local_outlier_factor_settings=best_params))
    after_ref = df[references_we_compare].value_counts()

    fig1, img_src, stats  = two_groups_analysis(df, best_features, references_we_compare=references_we_compare,nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="dot", show=False, website=True, layout_height=800, layout_width=1000)
    fig2, _, _            = two_groups_analysis(df, best_features, references_we_compare=references_we_compare, nice_name=lambda x: x[9:] if x.startswith("bacteria_") else x, 
                                        style="hist", show=False, website=True, layout_height=800, layout_width=1000)

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


    ret_val = []
    if df is not None:
        ret_val =  [
                    dhc.P("Reference vs. Non-reference count before novelty detection:"+str(before_ref)),
                    dhc.P("Reference vs. Non-reference count after novelty detection:"+str(after_ref)),
                    dhc.Br(),
                    dcc.Graph(figure=fig),
                    dhc.Br(),
                    dhc.P(f"The model with the best accuracy was chosen: n_neighbors={best_param} and features type {best_features_type}"),
                    dhc.Br(),
                    dcc.Graph(figure=fig1),
                    dhc.Br(),
                    dcc.Graph(figure=fig2),
                    dhc.Br(),
                    statistics_part,
                    dhc.Br(),dhc.Br(),
                    ]

    return ret_val