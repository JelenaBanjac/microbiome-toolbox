import dash_bootstrap_components as dbc
from dash import dcc
import dash_html_components as dhc
import dash_uploader as du
import dash_table
from dash_uploader import upload
import uuid
from utils.constants import home_location, page1_location, page2_location, page3_location, page4_location, page5_location, page6_location


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
                * `subjecID`: an identifier of the subject (i.e. mouse name),
                * `age_at_collection`: the time at which the sample was collected (in days),
                * `group`: the groups that are going to be compared, e.g. different `country` that sample comes from,
                * `meta_*`: prefix for metadata columns, e.g. `csection` becomes `meta_csection`, etc.
                * `id_*`: prefix for other ID columns (don't prefix `sampleID` nor `subjecID`)
                * `reference_group`: with `True`/`False` values, examples of a reference vs. non-reference
                * all other columns left should be bacteria names which will be automatically prefixed with `bacteria_*` after the upload.
            """
        ),
        dcc.Markdown(
            "More methods and specific details of method implementations can be seen in the Github repository [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox)."
        ),
        dhc.Br(),
        dhc.P(
            [
                dbc.Button(
                    "mouse data",
                    outline=True,
                    color="dark",
                    id="upload-default-data",
                    n_clicks=0,
                ),
            ],
            style={
                "textAlign": "center",
            },
        ),
        dhc.P(
            "or",
            style={
                "textAlign": "center",
            },
        ),
        du.Upload(
            id="upload-data",
            filetypes=["csv", "xls"],
            upload_id=session_id,
        ),
        dhc.Div(id="upload-infobox"),
    ]
    return upload


def serve_settings():

    import pandas as pd
    # reference_groups = ["user defined", "novelty detection algorithm decision"]
    df_dummy = pd.DataFrame(data={
        "sampleID": [""]*10,
        "subjectID": [""]*10,
        "age_at_collection": [""]*10,
        "reference_group": [""]*10,
        "group": [""]*10,
        "bacteria_1": [""]*10,
        "bacteria_2": [""]*10,
        "bacteria_3": [""]*10,
        "meta_1": [""]*10,
        "meta_2": [""]*10,
    })

    table = dash_table.DataTable(
        id='upload-datatable2',
        columns=[{
            "name": i, 
            "id": i,
            'deletable': True,
            'renamable': True
            } for i in df_dummy.columns],
        data=df_dummy.to_dict('records'),
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
            'width': f'{max(df_dummy.columns, key=len)}%',
            'maxWidth': '200px',
            'whiteSpace': 'normal'
        },
        editable=True, 
        export_format='xlsx',
        export_headers='display',
        merge_duplicate_headers=True
    )

    log_ratio_choice = dcc.Dropdown(
        id='bacteria-log-ratio',
        optionHeight=20,
        # options=[ {'label': b, "value": b} for b in log_ratio_bacterias],
        searchable=True,
        clearable=True,
        placeholder="[optional] select a bacteria for log-ratio",
        # value=settings["log_ratio_bacteria"]
    )

    reference_group_choice = dcc.Dropdown(
        id='reference-group',
        optionHeight=20,
        # options=[ {'label': rg, "value": rg} for rg in reference_groups],
        searchable=True,
        clearable=True,
        placeholder="reference group",
        # value=settings["ref_group_choice"]
    )


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
        table,
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col("Log-ratio bacteria for denominator:", className="md-3"),
                        dbc.Col(log_ratio_choice, className="md-9"),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col("Reference group choice:", className="md-3"),
                        dbc.Col(reference_group_choice, className="md-9"),
                    ]
                ),
            ],
            className="md-12",
            style={"height": 250},
        ),
    ]
    return settings

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

def serve_layout():
    session_id = str(uuid.uuid4())

    upload = serve_upload(session_id)
    settings = serve_settings()

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

# import dash_html_components as dhc
# from dash import dcc

# layout = dhc.Div([
#     dcc.Link('Go to Page 1', href='/page-1'),
#     dhc.Br(),
#     dcc.Link('Go to Page 2', href='/page-2'),
# ])
