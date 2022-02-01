# from dash import html as dhc
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
from utils.constants import methods_location

from microbiome.enumerations import AnomalyType

layout = dhc.Div(
    id="page-5-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Anomaly Detection",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                                There is a support for detecting anomalies in three different ways:  
                                1. `PREDICTION_INTERVAL`: samples outside the PI are considered to be anomalies,   
                                2. `LOW_PASS_FILTER`: the samples passing 2 standard deviations of the mean are considered to be anomalies,  
                                3. `ISOLATION_FOREST`: unsupervised anomaly detection algorithm on longitudinal data and get what samples are anomalous.   
                                """,
                            ),
                            dcc.Markdown(
                                """
                                By default, anomaly is a sample that is outside the prediction interval of microbiome trajectory that is built on reference samples.
                                The anomaly type can be chosen on Home page under the Trajectory settings or below. If anomaly type is chosen below, it will not affect the dataset available in other sections (methods).
                                """,
                            ),
                            dcc.Markdown(
                                """ 
                                Some of the available plot options:
                                - if plot is not loaded, click Refresh button,
                                - hovering above the plot shows more information on the samples,
                                - clicking on the labels on the legend can show/hide the clicked item from the plot,
                                - reset plot to initial state is enabled by clicking Home option or Refresh button,
                                - plots can be downloaded in SVG format.
                                """
                            ),
                            dcc.Markdown(
                                """
                                The examples that are not in the dashboard can be found in the [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox) repository.
                                """,
                            ),
                        ]
                    )
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-anomalies",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-anomalies",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=2,
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
                        dbc.Col("X-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="xaxis-delta-tick-anomalies",
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
                                id="yaxis-delta-tick-anomalies",
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
                                id="height-anomalies",
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
                                id="width-anomalies",
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
                        dbc.Col("Select anomaly type: ", width=2),
                        dbc.Col(
                            dcc.Dropdown(
                                id="anomaly-type-selection",
                                optionHeight=20,
                                options=[
                                    {"label": e.name, "value": e.name}
                                    for e in AnomalyType
                                ],
                                searchable=True,
                                clearable=True,
                                placeholder="select anomaly type",
                                value=None,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=4,
                        ),
                        dbc.Col(dhc.P(), width=6),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-5-1",
                            children=dhc.Div(id="page-5-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
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
            ]
        )
    ],
)
