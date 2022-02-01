# from dash import html as dhc
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
from utils.constants import methods_location

layout = dhc.Div(
    id="page-4-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Bacteria Importance with Time",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                                This type of the plot is useful if we are interested to see what bacteria is important in what time block for a microbiome trajectory that is built on the reference samples.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                Importance of different bacteria and their abundances across time blocks:   
                                - bacteria importance are stacked vertically where the size of the each bacteria sub-block represents its importance in that time block,  
                                - values in the box represent mean and standard deviation of its abundance in that time block (Note: we tested mean, geometric mean, and robust mean, and median represented data the best for our data. In this toolbox, we have a support for any *average function* a user may want),  
                                - total height of the box is fixed in all time blocks,  
                                - can choose number of important bacteria for time interval.  
                                """
                            ),
                            dcc.Markdown(
                                """
                                Some of the available plot options:  
                                - if plot is not loaded, click Refresh button,  
                                - hovering above the plot shows more information on the samples,  
                                - clicking on the labels on the legend can show/hide the clicked item from the plot,  
                                - reset plot to initial state is enabled by clicking Home option or Refresh button,  
                                - plots can be downloaded in SVG format,  
                                - time block widths can be modified by sliding the circles in the Time block ranges option.  
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
                                    id="button-refresh-timeblocks",
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
                                id="xaxis-delta-tick-timeblocks",
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
                                id="yaxis-delta-tick-timeblocks",
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
                                id="height-timeblocks",
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
                                id="width-timeblocks",
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
                                id="number-of-top-bacteria",
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
                                id="number-of-timeblocks",
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
                                id="time-block-ranges",
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
                            id="loading-4-1",
                            children=dhc.Div(id="page-4-display-value-1"),
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
