# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from utils.constants import home_location

layout = dhc.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=home_location),
                            dhc.H3(
                                "Bacteria Importance with Time",
                                style={
                                    "textAlign": "center",
                                }
                            ),
                            dhc.Br(),
                            dcc.Markdown('''
                            * Importance of different bacteria and their abundances across time boxes (reference data) - resolution (i.e. box width) can be chosen (in the dashboard it is fixed)
                                - bacteria height represents its importance in that time block
                                - values in the box represent average ± STD of its abundance in that time block (Note: we tested mean, geometric mean, and robust mean, and median represented data the best for our data. In this toolbox, we have a support for any *average function* a user may want.)
                                - total height of the box is fixed in all time blocks
                                - can choose number of important bacteria for time interval
                            * Importance of different bacteria and their abundances across time boxes on non-healthy data (but model trained on healthy samples)
                            ''', style={'textAlign': 'left',}),
                            dcc.Markdown("The examples that are not in the dashboard can be found in the `microbiome-toolbox` repository.", style={'textAlign': 'left',}),
                            
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
                            dhc.Div(id="anomaly-type-timeblocks"),
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
                                id="polynomial-degree-timeblocks",
                                type="number",
                                min=1,
                                max=20,
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
                        dbc.Col("Number of top bacteria: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="number-of-top-bacteria",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
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
                                id="number-of-timeblocks",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=5,
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
                                id="time-block-ranges",
                                min=0,
                                # max=30,
                                # value=[8, 10, 15, 17, 20],
                                pushable=1,
                                allowCross=False,
                                tooltip={"placement": "bottom", "always_visible": True},
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
                            id="loading-4-1",
                            children=dhc.Div(id="page-4-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                
            ]
        )
    ],
    # style={
    #     "verticalAlign": "middle",
    #     "textAlign": "center",
    #     "backgroundColor": "rgb(255, 255, 255)",  #'rgb(245, 245, 245)',
    #     "position": "relative",
    #     "width": "100%",
    #     #'height':'100vh',
    #     "bottom": "0px",
    #     "left": "0px",
    #     "zIndex": "1000",
    # },
)
