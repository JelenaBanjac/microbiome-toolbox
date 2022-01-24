# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
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
                            Click on one of the outliers below to see the suggestion for the intervention. 
                            The intervention simulation consists of suggesting the taxa values to change (or log-ratio values to change) in order to bring back the sample to the reference microbiome trajectory.
                            """,
                                style={
                                    "textAlign": "left",
                                },
                            ),
                        ]
                    )
                ),
                dhc.Br(),
                # dhc.Hr(),
                # dhc.Br(),
                # dhc.H3(
                #     "Reference trajectory",
                #     style={
                #         "textAlign": "center",
                #     },
                # ),
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
                                # value=[8, 10, 15, 17, 20],
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
