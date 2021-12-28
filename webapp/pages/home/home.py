import dash_bootstrap_components as dbc
from dash import dcc
# from dash import html as dhc
from dash import html as dhc
import dash_uploader as du
from dash import dash_table
from dash_uploader import upload
import uuid
from utils.constants import home_location, page1_location, page2_location, page3_location, page4_location, page5_location, page6_location
from microbiome.enumerations import Normalization, ReferenceGroup, TimeUnit, AnomalyType, FeatureExtraction, FeatureColumnsType



def serve_upload(session_id):
    upload = [
        dhc.H3(
            "Dataset upload",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        dhc.P(
            [
                "The Microbiome Toolbox implements methods that can be used for microbiome dataset analysis and microbiome trajectory prediction. The dashboard offers a wide variety of interactive visualizations.\
                    If you are just interested in seeing what methods are coved, you can use a demo dataset (mouse data) which enables the toolbox options below (by pressing the button).\
                    You can also upload your own dataset (by clicking or drag-and-dropping the file into the area below).",
                dhc.Br(),
                dhc.Br(),
                "In order for the methods to work, make sure the uploaded dataset has the following columns:",
            ]
        ),
        dcc.Markdown(
            """
                * `sampleID`: a unique dataset identifier, the ID of a sample,
                * `subjectID`: an identifier of the subject (i.e. mouse name),
                * `age_at_collection`: the time at which the sample was collected (in days),
                * `group`: the groups that are going to be compared, e.g. different `country` that sample comes from,
                * `meta_*`: prefix for metadata columns, e.g. `c-section` becomes `meta_csection`, etc.
                * `id_*`: prefix for other ID columns (don't prefix `sampleID` nor `subjectID`)
                * `reference_group`: with `True`/`False` values, examples of a reference vs. non-reference
                * all other columns left should be bacteria names which will be automatically prefixed with `bacteria_*` after the upload.
            """
        ),
        dcc.Markdown(
            "More methods and specific details of method implementations can be seen in the Github repository [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox)."
        ),
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(dhc.Div(dhc.P("Demo datasets:")), width=3),
                        dbc.Col([
                            dhc.Div(dbc.Button(
                                "Mouse data",
                                outline=True,
                                color="dark",
                                id="button-mouse-data",
                                n_clicks=0,
                            )),
                            dhc.Div(dhc.P()),
                            dhc.Div(dbc.Button(
                                "Human data",
                                outline=True,
                                color="dark",
                                id="button-human-data",
                                n_clicks=0,
                            ))
                            ], width=6
                        ),
                    ]),
                dhc.Br(),
                dbc.Row([
                        dbc.Col(dhc.Div(dhc.P("Custom datasets:")), width=3),
                        dbc.Col([
                            dhc.Div(dbc.Button(
                                "Custom data",
                                outline=True,
                                color="dark",
                                id="button-custom-data",
                                n_clicks=0,
                                disabled=True,
                            )),
                            dhc.Div(dhc.P()),
                            dhc.Div(du.Upload(
                                id="upload-data",
                                filetypes=["csv", "xls"],
                                upload_id=session_id,
                            ))
                            ], width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row([
                    dbc.Col(width=3),
                    dbc.Col(
                        dcc.Loading(id="loading-boxes", children=[
                            dhc.Div(id="upload-infobox"),
                            dhc.Div(id="upload-errorbox"),
                        ], type="default"), width=6),
                ]),
            ],
            className="md-12",
            style={"height": 250},
        ),
        dhc.Br(),
        
    ]
    return upload


def serve_settings():
    
    # reference_groups = ["user defined", "novelty detection algorithm decision"]

    table = dash_table.DataTable(
        id='upload-datatable',
        # style_data={
        #     'width': f'{max(df_dummy.columns, key=len)}%',
        #     'minWidth': '50px',
        #     'maxWidth': '500px',
        # },
        style_table={
            'height': 300, 
            'overflowX': 'auto'
        },
        style_cell={
            'height': 'auto',
            # all three widths are needed
            'minWidth': '200px', 
            # 'width': f'{max(df_dummy.columns, key=len)}%',
            'maxWidth': '200px',
            'whiteSpace': 'normal'
        },
        # Style headers with a dotted underline to indicate a tooltip
        style_header={
            'textDecoration': 'underline',
            'textDecorationStyle': 'dotted',
        },
        editable=True, 
        export_format='xlsx',
        export_headers='display',
        merge_duplicate_headers=True,
        tooltip_delay=0,
        tooltip_duration=None
    )

    
    feature_columns_choice = dcc.Dropdown(
        id='settings-feature-columns-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in FeatureColumnsType],
        searchable=True,
        clearable=True,
        placeholder="select feature columns",
        # value=FeatureColumnsType.BACTERIA.name,
        value=None,
    )

    reference_group_choice = dcc.Dropdown(
        id='settings-reference-group-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in ReferenceGroup],
        searchable=True,
        clearable=True,
        placeholder="select reference group",
        # value=ReferenceGroup.USER_DEFINED.name,
        value=None,
    )

    time_unit_choice = dcc.Dropdown(
        id='settings-time-unit-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in TimeUnit],
        searchable=True,
        clearable=True,
        placeholder="select time unit",
        # value=TimeUnit.DAY.name,
        value=None,
    )

    

    normalized_choice = dcc.Dropdown(
        id='settings-normalized-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in Normalization],
        searchable=True,
        clearable=True,
        placeholder="select normalization",
        # value=Normalization.NON_NORMALIZED.name,
        value=None,
    )

    anomaly_type_choice = dcc.Dropdown(
        id='settings-anomaly-type-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in AnomalyType],
        searchable=True,
        clearable=True,
        placeholder="select anomaly type",
        # value=AnomalyType.PREDICTION_INTERVAL.name,
        value=None,
    )

    feature_extraction_choice = dcc.Dropdown(
        id='settings-feature-extraction-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in FeatureExtraction],
        searchable=True,
        clearable=True,
        placeholder="select feature extraction",
        # value=FeatureExtraction.NONE.name,
        value=None,
    )

    log_ratio_bacteria_choice = dcc.Dropdown(
        id='settings-log-ratio-bacteria-choice',
        optionHeight=20,
        # options=[ {'label': b, "value": b} for b in log_ratio_bacterias],
        searchable=True,
        clearable=True,
        placeholder="[optional] select a bacteria for log-ratio",
        value=None,
    )

    file_name = dhc.Div(id="upload-file-name")
    number_of_samples = dhc.Div(id="upload-number-of-samples")
    number_of_subjects = dhc.Div(id="upload-number-of-subjects")
    unique_groups = dhc.Div(id="upload-unique-groups")
    number_of_reference_samples = dhc.Div(id="upload-number-of-reference-samples")
    differentiation_score = dhc.Div(id="upload-differentiation-score")

    settings = [
        dhc.Br(),
        dhc.Br(),
        dhc.Br(),
        dhc.H3(
            "Dataset settings",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        # dcc.Markdown(f"Currently loaded file: `{filename}`"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col("Loaded file: ", width=3),
                        dbc.Col(file_name, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of samples:", width=3),
                        dbc.Col(number_of_samples, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of subjects:", width=3),
                        dbc.Col(number_of_subjects, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Unique groups:", width=3),
                        dbc.Col(unique_groups, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of reference samples:", width=3),
                        dbc.Col(number_of_reference_samples, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Differentiation score:", width=3),
                        dbc.Col(differentiation_score, width=6),
                    ]
                ),
                dhc.Br(),
            ],
            className="md-12",
            # style={"height": 250},
        ),

        table,
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col("Feature columns:", width=3),
                        dbc.Col(feature_columns_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Reference group:", width=3),
                        dbc.Col(reference_group_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Time unit:", width=3),
                        dbc.Col(time_unit_choice, width=6),
                    ]
                ),
                dhc.Br(),
                
                dbc.Row(
                    [
                        dbc.Col("Normalization:", width=3),
                        dbc.Col(normalized_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Anomaly type:", width=3),
                        dbc.Col(anomaly_type_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Feature extraction:", width=3),
                        dbc.Col(feature_extraction_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Log-ratio bacteria:", width=3),
                        dbc.Col(log_ratio_bacteria_choice, width=6),
                    ]
                ),
                dhc.Br(),
            ],
            className="md-12",
            # style={"height": 250},
        ),
    ]
    # settings = dcc.Loading(
    #     id="loading-dataset-settings",
    #     children=settings,
    # )
    return settings

def serve_methods():

    card1 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Reference Definition & Statistics", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        # dhc.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/methods/page-1"),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card1-btn"
                            ),
                            href=page1_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card2 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Data Analysis & Exploration", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card2-btn"
                            ),
                            href=page2_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card3 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Microbiome Trajectory", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card3-btn"
                            ),
                            href=page3_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card4 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Bacteria Importance with Time", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card4-btn"
                            ),
                            href=page4_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card5 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Longitudinal Anomaly Detection", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card5-btn"
                            ),
                            href=page5_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card6 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Intervention Simulation", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card6-btn"
                            ),
                            href=page6_location,
                        ),
                    ]
                ),
            ],
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    methods = [
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dhc.Br(),
                            dhc.Div(dhc.H3("Dataset methods")),
                            dhc.Br(),
                        ],
                        className="md-12",
                    ),
                ),
                dbc.Row([card1, card2, card3]),
                dbc.Row([card4, card5, card6]),
            ],
            className="md-4",
        )
    ]

    return methods

def serve_layout():
    session_id = str(uuid.uuid4())

    upload = serve_upload(session_id)
    settings = serve_settings()
    methods = serve_methods()

    layout = [
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    [
                        dcc.Store(data=session_id, id='session-id'),
                        dhc.Br(),
                        dhc.Div(children=upload, id="dataset-upload"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Div(children=settings, id="dataset-settings"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Div(
                            children=methods,
                            id="dataset-methods",
                            style={
                                "verticalAlign": "middle",
                                "textAlign": "center",
                                "backgroundColor": "rgb(255, 255, 255)",
                                "position": "relative",
                                "width": "100%",
                                #'height':'100vh',
                                "bottom": "0px",
                                "left": "0px",
                                "zIndex": "1000",
                            },
                        ),
                        dhc.Br(),
                    ]
                ),
            ),
            className="md-12",
        ),
        dhc.Br(),
        dhc.Br(),
    ]

    return layout

layout = serve_layout()

# from dash import html as dhc
# from dash import dcc

# layout = dhc.Div([
#     dcc.Link('Go to Page 1', href='/page-1'),
#     dhc.Br(),
#     dcc.Link('Go to Page 2', href='/page-2'),
# ])
