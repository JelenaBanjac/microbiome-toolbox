import traceback

import dash_bootstrap_components as dbc
import numpy as np
from app import app
from dash import dcc
from dash import html as dhc
from dash.dependencies import Input, Output
from pages.home.home_data import get_trajectory


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
    Input("height-timeblocks", "value"),
    Input("width-timeblocks", "value"),
    Input("xaxis-delta-tick-timeblocks", "value"),
    Input("yaxis-delta-tick-timeblocks", "value"),
    Input("button-refresh-timeblocks", "n_clicks"),
)
def display_value(
    trajectory_path,
    time_block_ranges,
    num_top_bacteria,
    degree,
    height,
    width,
    x_delta,
    y_delta,
    n_clicks,
):
    results = []
    anomaly_type = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)
            anomaly_type = trajectory.anomaly_type.name

            layout_settings = dict(
                height=height,
                width=width,
                hoverdistance=None,
            )
            xaxis_settings = dict(
                dtick=x_delta,
            )
            yaxis_settings = dict(
                dtick=y_delta,
            )

            result = trajectory.plot_timeboxes(
                time_block_ranges=time_block_ranges,
                num_top_bacteria=num_top_bacteria,
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
    return results, anomaly_type
