# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc

layout = dhc.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href="/"),
                            dhc.H3("Data Analysis & Exploration"),
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
                            # # dhc.Div(id='page-2-display-value-0', children=loading_img),
                            # # dhc.Div(id='page-2-display-value-0-hidden', hidden=True),
                            # dcc.Loading(
                            #     id="loading-2-0",
                            #     children=[dhc.Div([dhc.Div(id='page-2-display-value-0'),])],
                            #     type="default",
                            # ),
                            # dhc.Br(),
                            dhc.Br(),
                            # Abundance plot in general
                            dhc.Hr(),
                            dhc.Br(),
                            dhc.H4("Taxa Abundances"),
                            dhc.Br(),
                            # dcc.Input(id='number-of-columns', type='number', min=2, max=10, step=1, value=3),
                            # dhc.Div(id='page-2-display-value-1', children=loading_img),
                            # dhc.Div(id='page-2-display-value-1-hidden', hidden=True),
                            dcc.Loading(
                                id="loading-2-1",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-2-display-value-1"),
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
                            # # dhc.Div(id='page-2-display-value-2', children=loading_img),
                            # # dhc.Div(id='page-2-display-value-2-hidden', hidden=True),
                            # dcc.Loading(
                            #     id="loading-2-2",
                            #     children=[dhc.Div([dhc.Div(id='page-2-display-value-2'),])],
                            #     type="default",
                            # ),
                            # dhc.Br(),
                            # Heatmap
                            dhc.Br(),
                            dhc.Hr(),
                            dhc.Br(),
                            dhc.H4("Taxa Abundances Heatmap"),
                            dhc.Br(),
                            # dhc.Div(id='page-2-display-value-3', children=loading_img),
                            # dhc.Div(id='page-2-display-value-3-hidden', hidden=True),
                            dcc.Loading(
                                id="loading-2-2",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-2-display-value-2"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            # # Shannon's diversity index and Simpson's dominace
                            # dhc.Hr(),
                            # dhc.H4("Diversity"),
                            # dhc.Div(id='page-2-display-value-4', children=loading_img),
                            # Dense longitudinal data
                            dhc.Hr(),
                            dhc.H4("Dense Longitudinal Data"),
                            dhc.Br(),
                            # dhc.Div(id='page-2-display-value-5', children=loading_img),
                            # dhc.Div(id='page-2-display-value-5-hidden', hidden=True),
                            dcc.Loading(
                                id="loading-2-3",
                                children=[
                                    dhc.Div(
                                        [
                                            dhc.Div(id="page-2-display-value-3"),
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
                            # dhc.Div(id='page-2-display-value-6', children=loading_img),
                            # dhc.Div(id='page-2-display-value-6-hidden', hidden=True),
                            dbc.Container(
                                [
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
                                            dbc.Col(
                                                dcc.Loading(
                                                    id="loading-2-4",
                                                    children=[
                                                        dhc.Div(
                                                            [
                                                                dhc.Div(
                                                                    id="page-2-display-value-4"
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    type="default",
                                                ),
                                                width=8,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dhc.Br(),
                            dhc.Br(),
                            # # Embedding in 3D
                            # dhc.Hr(),
                            # dhc.H4("Embedding in 3D space"),
                            # dhc.Br(),
                            # # dhc.Div(id='page-2-display-value-7', children=loading_img),
                            # # dhc.Div(id='page-2-display-value-7-hidden', hidden=True),
                            # dcc.Loading(
                            #     id="loading-2-7",
                            #     children=[dhc.Div([dhc.Div(id='page-2-display-value-7'),])],
                            #     type="default",
                            # ),
                            # dhc.Br(),
                            # Embedding in 2D, interactive
                            dhc.Hr(),
                            dhc.H4("Embedding in 2D space - Interactive Analysis"),
                            dhc.Br(),
                            # dhc.Div(id='page-2-display-value-8', children=loading_img),
                            # dhc.Div(id='page-2-display-value-8-hidden', hidden=True),
                            dhc.Br(),
                            dcc.Loading(
                                id="loading-2-5",
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
                                            dhc.Div(id="page-2-display-value-5"),
                                        ]
                                    )
                                ],
                                type="default",
                            ),
                            dhc.Br(),
                            # dcc.Interval(
                            #     id='page-2-main-interval-component',
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
    style={
        "verticalAlign": "middle",
        "textAlign": "center",
        "backgroundColor": "rgb(255, 255, 255)",  #'rgb(245, 245, 245)',
        "position": "relative",
        "width": "100%",
        #'height':'100vh',
        "bottom": "0px",
        "left": "0px",
        "zIndex": "1000",
    },
)
