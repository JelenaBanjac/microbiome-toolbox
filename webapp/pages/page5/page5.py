# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
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
                            * Detecting the outlier depending on outlier definition (in healthy reference data)
                                - *Prediction Interval:* outside 95% prediction interval (healthy trajectory interval)
                                - Other ways of detecting outliers - *longitudinal anomaly detection* - implemented rolling average:
                                    - *z-score on the trajectory*: Anomaly detection with z-scores and ones that pass 2xSTD
                                    - *Isolation Forest (IF):* unsupervised anomaly detection algorithm on longitudinal data and get what samples are anomalous
                            * Exploring trajectory outliers and finding the *commonalities - reference analysis*:
                                - Looking across different outliers, are there common features that are *off/FALSE* in most of these? Gives a common intervention angle
                                - Build a supervised model (`XGBoost`) and get the `SHAP` values in order to explain the anomalies.
                            * What do we do with these outliers that are detected in a healthy reference?
                                - Returning the (one) outlier back to the healthy region by changing the bacteria abundances that are not in normal range (healthy reference data)
                                - Remove these outliers and retrain the model with updated reference dataset
                            * Importance of different bacteria and their abundances across time boxes on non-healthy data (but model trained on healthy samples).
                            """,
                                style={
                                    "textAlign": "left",
                                },
                            ),
                            dcc.Markdown(
                                "The examples that are not in the dashboard can be found in the `microbiome-toolbox` repository.",
                                style={
                                    "textAlign": "left",
                                },
                            ),
                        ]
                    )
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Button(
                                "Refresh",
                                outline=True,
                                color="dark",
                                id="button-refresh-anomalies",
                                n_clicks=0,
                            ),
                            dhc.I(title="Refresh plots if not loaded", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            ], width=2,
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
