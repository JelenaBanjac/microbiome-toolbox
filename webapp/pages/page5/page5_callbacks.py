from dash.dependencies import Input, Output
from app import app
from pages.home.home_data import get_trajectory
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from microbiome.enumerations import AnomalyType
import traceback


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
)
def display_value(trajectory_path, anomaly_type, degree):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            if anomaly_type is None:
                anomaly_type_enum = trajectory.anomaly_type
            else:
                anomaly_type_enum = AnomalyType[anomaly_type]

            result = trajectory.plot_anomalies(
                anomaly_type=anomaly_type_enum, degree=degree
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an [email](msjelenabanjac@gmail.com)."),
                ],
                color="danger",
            )
    return results
