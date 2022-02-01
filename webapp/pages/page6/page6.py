# from dash import html as dhc
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
from utils.constants import methods_location

layout = dhc.Div(
    id="page-6-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Intervention Simulation",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                                Intervention simulation is a technique we propose to use in order to return an outlier back to the reference trajectory. 
                                This is done by modifying the taxa of the outlier with values of the top important bacteria from reference samples.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                Click on one of the outliers below to see the suggestion for the intervention. The intervention simulation consists of suggesting the taxa values to change (or log-ratio values to change) in order to bring back the sample to the reference microbiome trajectory.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                If an outlier is not back on the reference trajectory, possible reasons are:
                                - the time block in which the outlier is located is not wide or small enough,
                                - the reference samples have samples that should not be considered reference samples,
                                - the number of top important bacteria to consider is not high enough to help an outlier to become a reference sample.
                                """,
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
                                    id="button-refresh-intervention",
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
                        dbc.Col("Assumed anomaly type: ", width=2),
                        dbc.Col(
                            dhc.Div(id="anomaly-type-intervention"),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-intervention",
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
                                id="xaxis-delta-tick-intervention",
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
                                id="yaxis-delta-tick-intervention",
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
                                id="height-intervention",
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
                                id="width-intervention",
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
                        dbc.Col("Number of top bacteria: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="number-of-top-bacteria-intervention",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
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
                        dbc.Col("Number of time blocks: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="number-of-timeblocks-intervention",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=5,
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
                        dbc.Col("Time block ranges: ", width=2),
                        dbc.Col(
                            dcc.RangeSlider(
                                id="time-block-ranges-intervention",
                                min=0,
                                # max=30,
                                value=[8, 10, 15, 17, 20],
                                pushable=1,
                                allowCross=False,
                                tooltip={"placement": "bottom", "always_visible": True},
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=8,
                        ),
                        dbc.Col(dhc.P(), width=2),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-6-1",
                            children=dhc.Div(id="page-6-display-value-1"),
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
