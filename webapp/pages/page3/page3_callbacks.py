from dash.dependencies import Input, Output
from app import app
from pages.home.home_data import get_trajectory
from dash import dcc
from dash import html as dhc
import traceback
import dash_bootstrap_components as dbc


@app.callback(
    Output("page-3-display-value-0", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("button-refresh-feature-extraction", "n_clicks"),
)
def display_value(trajectory_path, n_clicks):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            if trajectory.feature_columns_plot is None:
                results = [
                    dcc.Markdown("Feature extraction has not been used."),
                    dcc.Markdown(
                        "Select one option from the dropdown menu in Trajectory Settings section of home page."
                    ),
                ]
            else:
                results = [
                    dcc.Markdown(
                        trajectory.feature_columns_plot_ret_val, dangerously_allow_html=True
                    ),
                    dhc.Br(),
                    dcc.Graph(
                        figure=trajectory.feature_columns_plot,
                        config=trajectory.feature_columns_plot_config,
                    ),
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


@app.callback(
    Output("page-3-display-value-1", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-trajectory", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            result = trajectory.plot_reference_trajectory(degree=degree)

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


@app.callback(
    Output("page-3-display-value-2", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-groups", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            if len(trajectory.dataset.df.reference_group.unique()) != 2:
                results = [
                    dcc.Markdown("Reference groups are not available."),
                    dcc.Markdown(
                        "Dataset contains only samples from a reference. Please modify the reference_group (with True, False values) column in your dataset to correspond to the reference groups."
                    ),
                ]
            else:
                result = trajectory.plot_reference_groups(degree=degree)

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


@app.callback(
    Output("page-3-display-value-3", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-groups", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            result = trajectory.plot_groups(degree=degree)

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


@app.callback(
    Output("page-3-display-value-4", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-longitudinal", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            result = trajectory.plot_animated_longitudinal_information(degree=degree)

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
