from dash.dependencies import Input, Output, State
from app import app
from pages.home.home_data import get_trajectory
from dash import dcc
from dash import html as dhc
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np


@app.callback(
    Output("time-block-ranges", "value"),
    Output("time-block-ranges", "max"),
    Input("microbiome-trajectory-location", "data"),
    Input("number-of-timeblocks", "value"),
)
def display_value(trajectory_path, number_od_timeblocks):
    maximum = None
    values = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)

        maximum = trajectory.y.max()
        values = np.linspace(0, maximum, number_od_timeblocks)

    print("values", values, "\n\n")
    return values, maximum


@app.callback(
    Output("page-4-display-value-1", "children"),
    Output("anomaly-type-timeblocks", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("time-block-ranges", "value"),
    Input("number-of-top-bacteria", "value"),
    Input("polynomial-degree-timeblocks", "value"),
)
def display_value(trajectory_path, time_block_ranges, num_top_bacteria, degree):
    results = []
    anomaly_type = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        anomaly_type = trajectory.anomaly_type.name

        result = trajectory.plot_timeboxes(
            layout_settings=dict(hoverdistance=None),
            time_block_ranges=time_block_ranges,
            num_top_bacteria=num_top_bacteria,
            degree=degree,
        )

        results = [
            dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
            dhc.Br(),
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results, anomaly_type
