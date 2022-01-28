# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from utils.constants import methods_location

from microbiome.enumerations import EmbeddingModelType

embedding_methods = [
    {"label": "PCA", "value": "pca"},
    {"label": "UMAP", "value": "umap"},
    {"label": "t-SNE", "value": "tsne"},
    {"label": "Isomap", "value": "isomap"},
]

layout = dhc.Div(
    id="page-1-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Data Analysis & Exploration",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                        Some of the methods for data analysis and exploration provided are:
                        - Sampling statistics
                        - Heatmap of taxa abundances w.r.t. time
                        - Taxa abundance errorbars
                        - Dense longitudinal data
                        - Shannon diversity index and Simpson dominance index
                        - Embeddings (different algorithms that we used in 2D and 3D space) with interactive selection and reference analysis.
                        """,
                                style={
                                    "textAlign": "left",
                                },
                            ),
                            # # Loaded table
                            # dhc.Hr(),
                            # dhc.H4("Loaded data table"),
                            # dhc.Br(),
                            # # dhc.Div(id='page-1-display-value-0', children=loading_img),
                            # # dhc.Div(id='page-1-display-value-0-hidden', hidden=True),
                            # dcc.Loading(
                            #     id="loading-1-0",
                            #     children=[dhc.Div([dhc.Div(id='page-1-display-value-0'),])],
                            #     type="default",
                            # ),
                            # dhc.Br(),
                            dhc.Br(),
                            # Abundance plot in general
                            dhc.Hr(),
                            dhc.Br(),
                            dhc.H4("Taxa Abundances"),
                            
                            dhc.Br(),
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Button(
                                                    "Refresh",
                                                    outline=True,
                                                    color="dark",
                                                    id="button-refresh-abundances",
                                                    n_clicks=0,
                                                ),
                                                dhc.I(title="Refresh plot if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                                                ], width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=10),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Number of columns: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="abundances-number-of-columns",
                                                    type="number",
                                                    min=1,
                                                    max=15,
                                                    step=1,
                                                    value=3,
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("X-axis Δtick: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="xaxis-delta-tick-abundances",
                                                    type="number",
                                                    min=1,
                                                    # max=20,
                                                    step=1,
                                                    # value=1,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Y-axis Δtick: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="yaxis-delta-tick-abundances",
                                                    type="number",
                                                    min=1,
                                                    # max=20,
                                                    step=1,
                                                    # value=1,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Row height: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="height-abundances",
                                                    type="number",
                                                    # min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=200,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Figure width: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="width-abundances",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=1200,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                    
                                ]
                            ),
                            dcc.Loading(
                                id="loading-1-1",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-1-display-value-1"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            # # Sampling statistics
                            # dhc.Hr(),
                            # dhc.H4("Sampling Statistics"),
                            # dhc.Br(),
                            # # dhc.Div(id='page-1-display-value-2', children=loading_img),
                            # # dhc.Div(id='page-1-display-value-1-hidden', hidden=True),
                            # dcc.Loading(
                            #     id="loading-1-2",
                            #     children=[dhc.Div([dhc.Div(id='page-1-display-value-2'),])],
                            #     type="default",
                            # ),
                            # dhc.Br(),
                            # Heatmap
                            dhc.Br(),
                            dhc.Hr(),
                            dhc.Br(),
                            dhc.H4("Taxa Abundances Heatmap"),
                            dhc.Br(),
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Button(
                                                    "Refresh",
                                                    outline=True,
                                                    color="dark",
                                                    id="button-refresh-heatmap",
                                                    n_clicks=0,
                                                ),
                                                dhc.I(title="Refresh plot if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                                                ], width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=10),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Values: ", width=2),
                                            dbc.Col(
                                                dcc.RadioItems(
                                                    id="heatmap-relative-absolute-values",
                                                    options=[
                                                        {'label': 'Relative', 'value': 'relative'},
                                                        {'label': 'Absolute', 'value': 'absolute'},
                                                    ],
                                                    value='relative',
                                                    # labelStyle={'display': 'inline-block'}
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Empty cells: ", width=2),
                                            dbc.Col(
                                                dcc.RadioItems(
                                                    id="heatmap-fillna-dropna",
                                                    options=[
                                                        {'label': 'fill', 'value': "fill"},
                                                        {'label': 'drop', 'value': "drop"},
                                                        {'label': 'none', 'value': "none"},
                                                    ],
                                                    value="none",
                                                    # labelStyle={'display': 'inline-block'}
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Average function: ", width=2),
                                            dbc.Col(
                                                dcc.RadioItems(
                                                    id="heatmap-avg-fn",
                                                    options=[
                                                        {'label': 'Median', 'value': "median"},
                                                        {'label': 'Mean', 'value': "mean"},
                                                    ],
                                                    value="median",
                                                    # labelStyle={'display': 'inline-block'}
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Row height: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="height-heatmap",
                                                    type="number",
                                                    # min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=20,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Figure width: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="width-heatmap",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=1200,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dhc.Br(),
                            dcc.Loading(
                                id="loading-1-2",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-1-display-value-2"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            # # Shannon's diversity index and Simpson's dominace
                            # dhc.Hr(),
                            # dhc.H4("Diversity"),
                            # dhc.Div(id='page-1-display-value-4', children=loading_img),
                            # Dense longitudinal data
                            dhc.Hr(),
                            dhc.H4("Dense Longitudinal Data"),
                            dhc.Br(),
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Button(
                                                    "Refresh",
                                                    outline=True,
                                                    color="dark",
                                                    id="button-refresh-longitudinal-stack",
                                                    n_clicks=0,
                                                ),
                                                dhc.I(title="Refresh plot if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                                                ], width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=10),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Number of columns: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="longitudinal-number-of-columns",
                                                    type="number",
                                                    min=1,
                                                    max=15,
                                                    step=1,
                                                    value=6,
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Number of bacteria: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="longitudinal-number-of-bacteria",
                                                    type="number",
                                                    min=1,
                                                    max=50,
                                                    step=1,
                                                    value=20,
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("X-axis Δtick: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="xaxis-delta-tick-longitudinal-stack",
                                                    type="number",
                                                    min=1,
                                                    # max=20,
                                                    step=1,
                                                    # value=1,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Y-axis Δtick: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="yaxis-delta-tick-longitudinal-stack",
                                                    type="number",
                                                    min=1,
                                                    # max=20,
                                                    step=1,
                                                    # value=1,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Figure height: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="height-longitudinal-stack",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=900,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Figure width: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="width-longitudinal-stack",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=1200,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Color palette name: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="color-palette-name-longitudinal-stack",
                                                    type="text",
                                                    min=1,
                                                    max=50,
                                                    step=1,
                                                    value="tab20",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                ]
                            ),
                            dcc.Loading(
                                id="loading-1-3",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-1-display-value-3"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            # Embedding in 2D
                            dhc.Hr(),
                            dhc.H4("Embedding in 2D space"),
                            dhc.Br(),
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Button(
                                                    "Refresh",
                                                    outline=True,
                                                    color="dark",
                                                    id="button-refresh-embedding",
                                                    n_clicks=0,
                                                ),
                                                dhc.I(title="Refresh plot if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                                                ], width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=10),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Embedding dimension: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="embedding-dimension",
                                                    type="number",
                                                    min=2,
                                                    max=3,
                                                    step=1,
                                                    value=2,
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Embedding method: ", width=2),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="embedding-method-type",
                                                    optionHeight=20,
                                                    options=[
                                                        {"label": e.name, "value": e.name}
                                                        for e in EmbeddingModelType
                                                    ],
                                                    searchable=True,
                                                    clearable=True,
                                                    placeholder="select anomaly type",
                                                    value=EmbeddingModelType.PCA.name,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Figure height: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="height-embedding",
                                                    type="number",
                                                    min=100,
                                                    # max=20,
                                                    step=1,
                                                    value=800,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Figure width: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="width-embedding",
                                                    type="number",
                                                    min=100,
                                                    # max=20,
                                                    step=1,
                                                    value=800,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        dbc.Col(
                                            dcc.Loading(
                                                id="loading-1-4",
                                                children=[
                                                    dhc.Div(
                                                        [
                                                            dhc.Div(
                                                                id="page-1-display-value-4"
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                type="default",
                                            ),
                                            width=12,
                                        ),
                                    ),
                                ]
                            ),
                            dhc.Br(),
                            dhc.Br(),
                            
                            # Embedding in 2D, interactive
                            dhc.Hr(),
                            dhc.H4("Embedding in 2D space - Interactive Analysis"),
                            dhc.Br(),
                            
                            dbc.Container(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Button(
                                                    "Refresh",
                                                    outline=True,
                                                    color="dark",
                                                    id="button-refresh-embedding-interactive",
                                                    n_clicks=0,
                                                ),
                                                dhc.I(title="Refresh plot if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                                                ], width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=10),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Embedding method: ", width=2),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="embedding-method-type-interactive",
                                                    optionHeight=20,
                                                    options=[
                                                        {"label": e.name, "value": e.name}
                                                        for e in EmbeddingModelType
                                                    ],
                                                    searchable=True,
                                                    clearable=True,
                                                    placeholder="select anomaly type",
                                                    value=EmbeddingModelType.PCA.name,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=8),
                                        ]
                                    ),
                                    dhc.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col("Figure height: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="height-embedding-interactive",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=800,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(dhc.P(), width=1),
                                            dbc.Col("Figure width: ", width=2),
                                            dbc.Col(
                                                dcc.Input(
                                                    id="width-embedding-interactive",
                                                    type="number",
                                                    min=500,
                                                    # max=20,
                                                    step=1,
                                                    value=800,
                                                    persistence=True,
                                                    persistence_type="session",
                                                ),
                                                width=2,
                                            ),
                                        ]
                                    ),
                                ]
                            ),

                            dhc.Br(),
                            dcc.Loading(
                                id="loading-1-5",
                                children=[
                                    dhc.Div(
                                        [
                                            dcc.Markdown(
                                                """To use an interactive option:   
                                - click on `Lasso Select` on the plot toolbox,  
                                - select the samples you want to group,  
                                - wait for the explainatory information to load (with confusion matrix).  
                                """,
                                                style={
                                                    "textAlign": "left",
                                                },
                                            ),
                                            dhc.Div(id="page-1-display-value-5"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            dcc.Link(
                                "Back", 
                                href=methods_location,
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dhc.Br(),
                            # dcc.Interval(
                            #     id='page-1-main-interval-component',
                            #     interval=INTERVAL, # in milliseconds
                            #     n_intervals=0,
                            #     max_intervals=MAX_INTERVALS
                            # )
                        ],
                        className="md-4",
                    )
                )
            ],
            className="md-4",
        )
    ],
)
