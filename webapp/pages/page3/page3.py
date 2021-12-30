# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from utils.constants import home_location

layout = dhc.Div(
    id="page-3-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=home_location),
                            dhc.H3(
                                "Microbiome Trajectory",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                            * Data handling (hunting for the plateau of performance reached, so we can use less number of features):  
                                - top K important features selection based on the smallest MAE error (i.e. how does trajectory performance looks like when only working with top 5 or top 10 bacteria used for the model)  
                                - remove near zero variance features  
                                - remove correlated features  
                            * Microbiome Trajectory - all the combinations below
                                - only mean line  
                                - only line with prediction interval and confidence interval  
                                - line with samples  
                                - longitudinal data, every subject  
                                - coloring per group (e.g. per country)  
                                - red-color dots or stars-shapes are outliers  
                            * Measuring the trajectory performance (all before plateau area):  
                                - MAE  (goal: *smaller*)  
                                - R^2 score (goal: *bigger*), percent of variance captured  
                                - Pearson correlation (MMI, age_at_collection)  
                                - Prediction Interval (PI) - mean and median = prediction interval 95% = the interval in which we expect the healthy reference to fall in (goal: *smaller*)  
                                - Standard deviation of the error  
                                - Visual check
                            * Testing difference between different trajectories using linear regression statistical analysis and spline:  
                                - Testing **universality** across different groups  
                                - Testing **differentiation** of 2 trajectories (e.g. healthy vs. non-healthy) - spline p-values, linear regression p-values  
                            """,
                                style={
                                    "textAlign": "left",
                                },
                            ),
                        ]
                    )
                ),
                # Reference trajectory
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3(
                    "Feature extraction",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                # dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                # dbc.Row(
                #     [
                #         dbc.Col("Polynomial degree: ", width=2),
                #         dbc.Col(
                #             dcc.Input(
                #                 id="polynomial-degree-reference-trajectory",
                #                 type="number",
                #                 min=1,
                #                 max=20,
                #                 step=1,
                #                 value=1,
                #             ),
                #             width=2,
                #         ),
                #         dbc.Col(dhc.P(), width=8),
                #     ]
                # ),
                # dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-0",
                            children=dhc.Div(id="page-3-display-value-0"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Reference trajectory
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3(
                    "Reference trajectory",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-reference-trajectory",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-1",
                            children=dhc.Div(id="page-3-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Reference groups
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3(
                    "Reference groups",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-reference-groups",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-2",
                            children=dhc.Div(id="page-3-display-value-2"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Groups
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3(
                    "Groups",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-groups",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-3",
                            children=dhc.Div(id="page-3-display-value-3"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Longitudinal information
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3(
                    "Longitudinal information",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-longitudinal",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-4",
                            children=dhc.Div(id="page-3-display-value-4"),
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
)
