from typing import Container
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
import dash_html_components as dhc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import base64
import datetime
import io
from flask_caching import Cache
import pandas as pd
import uuid
import time
import os
import dash_uploader as du
from pathlib import Path
import math
import numpy as np

from index import server, app, cache, UPLOAD_FOLDER_ROOT, loading_img

from webapp.pages import page1, page2, page3, page4, page5, page6

DF_DEFAULT = pd.read_csv('https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/notebooks/Mouse_16S/INPUT_FILES/website_mousedata_default.csv', sep=",")
FILE_NAME_DEFAULT = "website_mousedata_default.csv"

# this example that adds a logo to the navbar brand
navbar = dbc.Navbar(
            dbc.Container(
                [
                    dhc.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                #dbc.Col(dhc.Img(src=PLOTLY_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Microbiome Toolbox", className="ml-2")),
                            ],
                            align="center",
                            no_gutters=True,
                        ),
                        href="#",
                    ),
                ]
            ),
            color="dark",
            dark=True,
            className="mb-5",
        )


card1 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Reference Definition & Statistics", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    #dhc.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/methods/page-1"),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card1-btn"), href='/methods/page-1'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card2 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Data Analysis & Exploration", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card2-btn"), href='/methods/page-2'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card3 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Microbiome Trajectory", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card3-btn"), href='/methods/page-3'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card4 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Bacteria Importance with Time", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card4-btn"), href='/methods/page-4'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card5 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Longitudinal Anomaly Detection", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card5-btn"), href='/methods/page-5'),
                ]
            ),
        ],
        #style={"width": "18rem"},
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)

card6 = dbc.Col(
    dbc.Card(
        [
            dbc.CardImg(src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg", top=True),
            dbc.CardBody(
                [
                    dhc.H4("Intervention Simulation", className="card-title"),
                    # dhc.P(
                    #     "Some quick example text to build on the card title and "
                    #     "make up the bulk of the card's content.",
                    #     className="card-text",
                    # ),
                    dcc.Link(dbc.Button("See more", outline=True, color="dark", id="card6-btn"), href='/methods/page-6'),
                ]
            ),
        ],
        style={"backgroundColor": "rgb(240, 240, 240)"},
        className="mb-4 box-shadow"
    ),
    className="md-4"
)


def parse_dataset(filename):
    #content_type, content_string = content.split(',')

    #decoded = base64.b64decode(content_string)
    df = None
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = pd.read_csv(filename)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            #df = pd.read_excel(io.BytesIO(decoded))
            #df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep="\t")
            df = pd.read_csv(filename, sep="\t")
            if len(df.columns)==1:
                df = pd.read_csv(filename, sep=",")
                if len(df.columns)==1:
                    raise Exception("Not a good file separator")
    except Exception as e:
        print(e)
        return None
    print('\nFinished parsing df parse_dataset')

    return df




def main_layout_(session_id, upload_filename):
    if not upload_filename:
        upload_filename_alert = dhc.Div([])
    else:
        upload_filename_alert = dhc.Div(dbc.Alert(f"Currently loaded file: {upload_filename}", color="info"))

    print("\nMain layout function called only with /methods")
    return dhc.Div(id="main",
                 children=[
                   dhc.Div(id='main-upload', children=[
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    dhc.H3("Upload Dataset", style={'textAlign': 'center',}),
                                    dhc.Br(),
                                    dhc.P(["The Microbiome Toolbox implements methods that can be used for microbiome dataset analysis and microbiome trajectory prediction. The dashboard offers a wide variety of interactive visualizations.\
                                           If you are just interested in seeing what methods are coved, you can use a demo dataset (mouse data) which enables the toolbox options below (by pressing the button).\
                                           You can also upload your own dataset (by clicking or drag-and-dropping the file into the area below).", 
                                           dhc.Br(),dhc.Br(),
                                           "In order for the methods to work, make sure the uploaded dataset has the following columns:",
                                    ]),
                                    dcc.Markdown('''
                                        * `sampleID`: a unique dataset identifier, the ID of a sample,
                                        * `subjecID`: an identifier of the subject (i.e. mouse name),
                                        * `age_at_collection`: the time at which the sample was collected (in days),
                                        * `group`: the groups that are going to be compared, e.g. different `country` that sample comes from,
                                        * `meta_*`: prefix for metadata columns, e.g. `csection` becomes `meta_csection`, etc.
                                        * `id_*`: prefix for other ID columns (don't prefix `sampleID` nor `subjecID`)
                                        * `reference_group`: with `True`/`False` values, examples of a reference vs. non-reference
                                        * all other columns left should be bacteria names which will be automatically prefixed with `bacteria_*` after the upload.
                                    '''),
                                    # dhc.P([
                                    #        "add the prefix 'meta__' for the metadata columns, and `id__` for the ID columns. The rest of the columns will be considered to be bacteria abundance columns automaticaly after the upload.",
                                    #        dhc.Br(),
                                    #        "When data table is loaded, you can remove or modify the column names, and edit table values.", 
                                    #        dhc.Br(),
                                    # ]),
                                    # dhc.P(["Or use the mouse dataset to see how the Microbiome Toolbox works, which can be downloaded from this ",
                                    #         dhc.A("link.", href="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/notebooks/Mouse_16S/INPUT_FILES/website_mousedata.xls", target='_blank'),
                                    #         "or click on this button to load it now ",
                                            
                                    #     ], style={'textAlign': 'center',}),
                                    dhc.P([dbc.Button("load demo mosue data", outline=True, color="dark", id='upload-default-data', n_clicks=0),], style={'textAlign': 'center',}),
                                    #
                                    dhc.P("or", style={'textAlign': 'center',}),
                                    du.Upload(
                                        id='upload-data',
                                        filetypes=['csv', 'xls'],
                                        upload_id=session_id,
                                    ),
                                    dhc.Div(id="upload-infobox", children=upload_filename_alert),
                                    dhc.Br(),
                                    dhc.Div(id='upload-datatable-div'),
                                    dhc.Br()
                                    ]),
                            ]),
                            
                        ], className="md-12"),
                    ]),
                    dhc.Br(),
                   

                    dhc.Div(id="main-methods", children=[
                            #dcc.Location(id='url', refresh=False),
                        
                            dbc.Container([
                                dbc.Row(
                                    dbc.Col([
                                        dhc.Br(),
                                        dhc.Div(dhc.H3("Methods")),
                                        dhc.Br(),
                                        ], 
                                    className="md-12"),
                                ),
                                dbc.Row([card1, card2, card3]),
                                dbc.Row([card4, card5, card6])
                            ],
                            className="md-4",
                            )
                        ],
                        style={
                            'verticalAlign':'middle',
                            'textAlign': 'center',
                            'backgroundColor': 'rgb(255, 255, 255)', #'rgb(245, 245, 245)',
                            'position':'relative',
                            'width':'100%',
                            #'height':'100vh',
                            'bottom':'0px',
                            'left':'0px',
                            'zIndex':'1000',
                        }
                )
        ])

def layout_(session_id, upload_info):
    print("\nLayout function called from server_layout init or can be separately called")
    return dhc.Div([
        navbar,
        dcc.Location(id='url', refresh=False),
        dhc.Div(session_id, id='session-id', style={'display': 'none'}),
        dhc.Div(upload_info, id='upload-filename', style={'display': 'none'}),

        main_layout_(session_id, upload_info)
    ])

def serve_layout():
    session_id = str(uuid.uuid4())
    print("\nServe layout", session_id)

    return layout_(session_id, '')

app.layout = serve_layout

def check_dataframe_validity(df):
    print("\tcolumns:",df.columns.tolist())
    ret_val_err = ""
    if "sampleID" not in df.columns.tolist():
        ret_val_err += "sampleID\n"
    if "subjectID" not in df.columns.tolist():
        ret_val_err += "subjectID\n"
    if "age_at_collection" not in df.columns.tolist():
        ret_val_err += "age_at_collection\n"
    if "group" not in df.columns.tolist():
        ret_val_err += "group\n"
    if "reference_group" not in df.columns.tolist():
        ret_val_err += "reference_group\n"

    return ret_val_err


def write_dataframe(session_id, df):
    '''
    Write dataframe to disk, for now just as CSV
    For now do not preserve or distinguish filename;
    user has one file at once.
    '''
    
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.pickle")
    print('\nCalling write_dataframe', filename)
    df.to_pickle(filename)

def write_logbacteria(session_id, bacteria):
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.txt")
    if bacteria:
        with open(filename, "w") as text_file:
            text_file.write(bacteria)

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

# @cache.memoize()
def read_logbacteria(session_id, timestamp):
    filename = os.path.join(UPLOAD_FOLDER_ROOT, f"{session_id}.txt")
    
    data = None
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = file.read().replace('\n', '')
    print("read logbacteria", data)
    return data 


# Update the index
@app.callback(Output('main', 'children'),
              [Input('url', 'pathname')],
              [State('session-id', 'children'),
              State('upload-filename', 'children')])
def display_page(pathname, session_id, upload_filename):
    print("\nPathname", pathname, "session_id", session_id)
    if pathname == '/methods/page-1':
        return page1.layout
    elif pathname == '/methods/page-2':
        return page2.layout
    elif pathname == '/methods/page-3':
        return page3.layout
    elif pathname == '/methods/page-4':
        return page4.layout
    elif pathname == '/methods/page-5':
        return page5.layout
    elif pathname == '/methods/page-6':
        return page6.layout
    else: 
        print("\tohter path....")
        return main_layout_(session_id, upload_filename)  #app.layout  #main_layout_(None)

def fix_zeros(row, feature_columns):
    for c in feature_columns:
        row[c] = 1e-10 if row[c]==0.0 or row[c]<1e-10 else row[c]
    return row

@app.callback(Output('upload-datatable2', 'data'),
              Input('bacteria-log-ratio', 'value'),
              State('session-id', 'children'))
def display_output(log_ratio_bacteria, session_id):
    print(f"Selected log-bacteria: {log_ratio_bacteria}")
    
    df = read_dataframe(f"{session_id}_original", None)
    feature_columns = df.columns[df.columns.str.startswith("bacteria_")].tolist()
    df = df.apply(lambda row: fix_zeros(row, feature_columns), axis=1)

    if log_ratio_bacteria is not None:
        for c in feature_columns:
            if c != log_ratio_bacteria:
                df[c] = df.apply(lambda row: math.log2(row[c]/row[log_ratio_bacteria]), axis=1)

        # remove reference, since these are abundances
        df = df.drop(columns=log_ratio_bacteria, axis=1)
        
    write_logbacteria(session_id, log_ratio_bacteria)
    write_dataframe(session_id, df)
    print("Stored log-bacteria:", read_logbacteria(session_id, None))

    return df.to_dict('records')


# def train_val_test_split(df, test_size):
#     df["dataset_type"] = "Test"

#     for g in df.group.unique():
#         df_ref_g   = df[(df.healthy_reference==True)&(df.group==g)]
#         train_idx_g, val_idx_g = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7).split(df_ref_g, groups=df_ref_g["subjectID"]))
#         df.loc[df_ref_g.iloc[train_idx_g].index, "dataset_type"] = "Train"
#         df.loc[df_ref_g.iloc[val_idx_g].index, "dataset_type"] = "Validation"

# def train_test_split_per_reference(df, test_size):
#     train_idx1, test_idx1 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df.healthy_reference==True].index, groups=df[df.healthy_reference==True]['subjectID']))
#     train_idx2, test_idx2 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df.healthy_reference==False].index, groups=df[df.healthy_reference==False]['subjectID']))

#     df.loc[df[df.healthy_reference==True].iloc[train_idx1].index, "classification_dataset_type"] = "Train-1"
#     df.loc[df[df.healthy_reference==True].iloc[test_idx1].index, "classification_dataset_type"] = "Test-1"
#     df.loc[df[df.healthy_reference==False].iloc[train_idx2].index, "classification_dataset_type"] = "Train-2"
#     df.loc[df[df.healthy_reference==False].iloc[test_idx2].index, "classification_dataset_type"] = "Test-2"

#     df.loc[df.healthy_reference==True, "classification_label"] = 1
#     df.loc[df.healthy_reference==False, "classification_label"] = -1


@app.callback(
    [Output('upload-filename', 'children'),
    Output('upload-infobox', 'children'),
    Output('card1-btn', 'disabled'),
    Output('card2-btn', 'disabled'),
    Output('card3-btn', 'disabled'),
    Output('card4-btn', 'disabled'),
    Output('card5-btn', 'disabled'),
    Output('card6-btn', 'disabled'),
    Output('upload-datatable-div', 'children'),],
    [Input('upload-data', 'isCompleted'),
     Input('upload-default-data', 'n_clicks')],
    [State('session-id', 'children'),
     State('upload-data', 'fileNames'),
     State('upload-data', 'upload_id'),
     State('upload-filename', 'children')])
def return_methods(iscompleted, default_data_clicked, session_id, filenames, upload_id, filename_latest):
    print("Upload callback called")
    print(default_data_clicked)
    filename = ''
    ret_val_err = ''

    if default_data_clicked > 0:
        print("load mouse data")
        filename = FILE_NAME_DEFAULT
        iscompleted = True
        upload_id = session_id
    
    upload_infobox = dhc.Div([])
    upload_datatable = dash_table.DataTable(id='upload-datatable2')
    methods_disabled = True

    if filenames is not None:
        if filename == '':
            print("filename", filename, "filenames", filenames)
            filename = filenames[0]

    elif filename_latest != '':
        filename = filename_latest
        upload_infobox = dhc.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))
        methods_disabled = False
        df_original = read_dataframe(f"{session_id}_original", None)
        df = read_dataframe(session_id, None)
        log_ratio_bacterias = [b for b in df_original.columns[df_original.columns.str.startswith("bacteria_")] ]
        logbacteria = read_logbacteria(session_id, None)
        
        testset_size = len(df[df.reference_group!=True])/len(df)
        val_size_default = 0.2
        classification_split = 0.2
        reference_groups = ["user defined", "novelty detection algorithm decision"]
        reference_group = reference_groups[0]

        table = dash_table.DataTable(
            id='upload-datatable2',
            columns=[{
                "name": i, 
                "id": i,
                'deletable': True,
                'renamable': True
                } for i in df_original.columns],
            data=df.to_dict('records'),
            style_data={
                'width': '{}%'.format(max(df_original.columns, key=len)),
                'minWidth': '50px',
                'maxWidth': '500px',
            },
            style_table={
                'height': 300, 
                'overflowX': 'auto'
            },
            editable=True, 
            export_format='xlsx',
            export_headers='display',
            merge_duplicate_headers=True
        )

        log_ratio_choice = dcc.Dropdown(
            id='bacteria-log-ratio',
            optionHeight=20,
            options=[ {'label': b, "value": b} for b in log_ratio_bacterias],
            searchable=True,
            clearable=True,
            placeholder="[optional] select a bacteria for log-ratio",
            value=logbacteria
        )

        reference_group_choice = dcc.Dropdown(
            id='reference-group',
            optionHeight=20,
            options=[ {'label': rg, "value": rg} for rg in reference_groups],
            searchable=True,
            clearable=True,
            placeholder="reference group",
            value=reference_group
        )

        train_test_split_choice = dcc.Dropdown(
            id='train-val-test-split',
            optionHeight=20,
            options=[ {'label': f"{(1-r-testset_size)*100:.0f}%-{r*100:.0f}%-{testset_size*100:.0f}%", 
            "value": f"{(1-r-testset_size)*100:.0f}%-{r*100:.0f}%-{testset_size*100:.0f}%"} for r in np.arange(0.0, 1.0-testset_size, 0.2)],
            searchable=True,
            clearable=True,
            placeholder="train-validation-test split",
            value=f"{(1-val_size_default-testset_size)*100:.0f}%-{val_size_default*100:.0f}%-{testset_size*100:.0f}%"
        )

        classification_split_choice = dcc.Dropdown(
            id='classification-split',
            optionHeight=20,
            options=[ {'label': f"reference: {(1-testset_size)*(1-r)*100:.0f}%-{(1-testset_size)*r*100:.0f}% & non-reference: {testset_size*(1-r)*100:.0f}%-{testset_size*r*100:.0f}%", 
            "value": f"reference: {(1-testset_size)*(1-r)*100:.0f}%-{(1-testset_size)*r*100:.0f}% & non-reference: {testset_size*(1-r)*100:.0f}%-{testset_size*r*100:.0f}%" } for r in np.arange(0.0, 1.0, 0.2)],
            searchable=True,
            clearable=True,
            placeholder="reference vs. non-reference train-test split",
            value=f"reference: {(1-testset_size)*(1-classification_split)*100:.0f}%-{(1-testset_size)*classification_split*100:.0f}% & non-reference: {testset_size*(1-classification_split)*100:.0f}%-{testset_size*classification_split*100:.0f}%"
        )

        upload_datatable = dhc.Div([
            
            table,
            log_ratio_choice,
            reference_group_choice,
            train_test_split_choice,
            classification_split_choice,
        ], style={"height": 530})

    
    
    # at the initialization of the page or when back
    if not iscompleted:
        return filename, upload_infobox, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, upload_datatable

    df = None
    if filename is not None:
        if filename not in FILE_NAME_DEFAULT:

            if upload_id:
                root_folder = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)
            else:
                root_folder = UPLOAD_FOLDER_ROOT

            file = os.path.join(root_folder, filename)

            df = parse_dataset(file)

            
        else:
            df = DF_DEFAULT.copy()

        ret_val_err = check_dataframe_validity(df)
        if ret_val_err != "":
            upload_infobox = dhc.Div([dbc.Alert("There was an error processing this file! Missing columns: " + ret_val_err.replace('\n', ', ')[:-2], color="danger")])
            methods_disabled = True

        else:
            specific_columns = ["sampleID", "subjectID", "group", "age_at_collection", "reference_group",
                                'dataset_type', 'dataset_type_classification', 'classification_dataset_type', 'classification_label']
            #metadata_columns = df.columns[df.columns.str.startswith("meta_")].tolist()
            #id_columns = df.columns[df.columns.str.startswith("id_")].tolist()
            other_columns = df.columns[(~df.columns.isin(specific_columns))&(~df.columns.str.startswith("meta_"))&(~df.columns.str.startswith("id_"))].tolist()
            df = df.rename(mapper={k:f"bacteria_{k}" for k in other_columns}, axis=1)

            write_dataframe(f"{upload_id}_original", df)
            upload_infobox = dhc.Div(dbc.Alert(f"Currently loaded file: {filename}", color="info"))

            methods_disabled = False
            log_ratio_bacterias = [b for b in df.columns[df.columns.str.startswith("bacteria_")] ]
            logbacteria = read_logbacteria(session_id, None)

            testset_size = len(df[df.reference_group!=True])/len(df)
            val_size_default = 0.2
            classification_split = 0.2
            reference_groups = ["user defined", "novelty detection algorithm decision"]
            reference_group = reference_groups[0]

            table = dash_table.DataTable(
                id='upload-datatable2',
                columns=[{
                    "name": i, 
                    "id": i,
                    'deletable': True,
                    'renamable': True
                    } for i in df.columns],
                data=df.to_dict('records'),
                style_data={
                    'width': '{}%'.format(max(df.columns, key=len)),
                    'minWidth': '50px',
                    'maxWidth': '500px',
                },
                style_table={
                    'height': 300, 
                    'overflowX': 'auto'
                },
                editable=True, 
                export_format='xlsx',
                export_headers='display',
                merge_duplicate_headers=True
            )

            log_ratio_choice = dcc.Dropdown(
                id='bacteria-log-ratio',
                optionHeight=20,
                options=[ {'label': b, "value": b} for b in log_ratio_bacterias],
                searchable=True,
                clearable=True,
                placeholder="[optional] select a bacteria for log-ratio",
                value=logbacteria
            )

            reference_group_choice = dcc.Dropdown(
                id='reference-group',
                optionHeight=20,
                options=[ {'label': rg, "value": rg} for rg in reference_groups],
                searchable=True,
                clearable=True,
                placeholder="reference group",
                value=reference_group
            )

            train_test_split_choice = dcc.Dropdown(
                id='train-val-test-split',
                optionHeight=20,
                options=[ {'label': f"{(1-r-testset_size)*100:.0f}%-{r*100:.0f}%-{testset_size*100:.0f}%", 
                "value": f"{(1-r-testset_size)*100:.0f}%-{r*100:.0f}%-{testset_size*100:.0f}%"} for r in np.arange(0.0, 1.0-testset_size, 0.2)],
                searchable=True,
                clearable=True,
                placeholder="train-validation-test split",
                value=f"{(1-val_size_default-testset_size)*100:.0f}%-{val_size_default*100:.0f}%-{testset_size*100:.0f}%"
            )

            classification_split_choice = dcc.Dropdown(
                id='classification-split',
                optionHeight=20,
                options=[ {'label': f"reference: {(1-testset_size)*(1-r)*100:.0f}%-{(1-testset_size)*r*100:.0f}% & non-reference: {testset_size*(1-r)*100:.0f}%-{testset_size*r*100:.0f}%", 
                "value": f"reference: {(1-testset_size)*(1-r)*100:.0f}%-{(1-testset_size)*r*100:.0f}% & non-reference: {testset_size*(1-r)*100:.0f}%-{testset_size*r*100:.0f}%" } for r in np.arange(0.0, 1.0, 0.2)],
                searchable=True,
                clearable=True,
                placeholder="reference vs. non-reference train-test split",
                value=f"reference: {(1-testset_size)*(1-classification_split)*100:.0f}%-{(1-testset_size)*classification_split*100:.0f}% & non-reference: {testset_size*(1-classification_split)*100:.0f}%-{testset_size*classification_split*100:.0f}%"
            )

            upload_datatable = dhc.Div([
                
                table,
                log_ratio_choice,
                reference_group_choice,
                train_test_split_choice,
                classification_split_choice,
            ], style={"height": 530})


    if df is None:
        upload_infobox = dhc.Div(dbc.Alert("There was an error processing this file!", color="danger"))
        methods_disabled = True

    print("filename", filename)
    print("upload_infobox", upload_infobox)
    return filename, upload_infobox, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, methods_disabled, upload_datatable



if __name__ == '__main__':
    app.run_server(debug=True, port=8081)
    # app.run_server(debug=False,
    #             host=os.getenv("HOST", "0.0.0.0"),
    #             port=os.getenv("PORT", "5000"))


