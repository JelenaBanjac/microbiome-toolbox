import traceback

import dash_bootstrap_components as dbc
from app import app
from dash import dcc
from dash import html as dhc
from dash.dependencies import Input, Output
from pages.home.home_data import get_trajectory

from microbiome.enumerations import AnomalyType


@app.callback(
    Output("anomaly-type-selection", "value"),
    Input("microbiome-trajectory-location", "data"),
)
def display_value(trajectory_path):
    anomaly_type = None
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        anomaly_type = trajectory.anomaly_type.name
    return anomaly_type


@app.callback(
    Output("page-5-display-value-1", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("anomaly-type-selection", "value"),
    Input("polynomial-degree-anomalies", "value"),
    Input("height-anomalies", "value"),
    Input("width-anomalies", "value"),
    Input("xaxis-delta-tick-anomalies", "value"),
    Input("yaxis-delta-tick-anomalies", "value"),
    Input("button-refresh-anomalies", "n_clicks"),
)
def display_value(
    trajectory_path, anomaly_type, degree, height, width, x_delta, y_delta, n_clicks
):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            layout_settings = dict(
                height=height,
                width=width,
            )
            xaxis_settings = dict(
                dtick=x_delta,
            )
            yaxis_settings = dict(
                dtick=y_delta,
            )

            if anomaly_type is None:
                anomaly_type_enum = trajectory.anomaly_type
            else:
                anomaly_type_enum = AnomalyType[anomaly_type]

            result = trajectory.plot_anomalies(
                anomaly_type=anomaly_type_enum,
                degree=degree,
                layout_settings=layout_settings,
                xaxis_settings=xaxis_settings,
                yaxis_settings=yaxis_settings,
            )

            results = [
                dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
                dhc.Br(),
                dcc.Graph(figure=result["fig"], config=result["config"]),
            ]
        except Exception as e:
            results = dbc.Alert(
                children=[
                    dcc.Markdown("Microbiome trajectory error: " + str(e)),
                    dcc.Markdown(traceback.format_exc()),
                    dcc.Markdown(
                        "Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."
                    ),
                ],
                color="danger",
            )
    return results
