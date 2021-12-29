from dash.dependencies import Input, Output, State
from app import app
from pages.home.home_data import get_trajectory
from dash import dcc
from dash import html as dhc
import dash
import dash_bootstrap_components as dbc
import pandas as pd


@app.callback(
    Output('page-3-display-value-0', 'children'),
    Input("microbiome-trajectory-location", "data"),
)
def display_value(trajectory_path):
    results = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
    
        if trajectory.feature_columns_plot is None:
            results = [
                dcc.Markdown("Feature extraction has not been used."),
                dcc.Markdown("Select one option from the dropdown menu in Trajectory Settings section of home page."),
            ]
        else:
            results = [
                dcc.Markdown(trajectory.feature_columns_plot_ret_val, dangerously_allow_html=True),
                dhc.Br(),
                dcc.Graph(figure=trajectory.feature_columns_plot, config=trajectory.feature_columns_plot_config),
            ]
    return results

@app.callback(
    Output('page-3-display-value-1', 'children'),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-trajectory", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        
        result = trajectory.plot_reference_trajectory(degree=degree)

        results = [
            dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
            dhc.Br(),
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results

@app.callback(
    Output('page-3-display-value-2', 'children'),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-groups", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        
        result = trajectory.plot_reference_groups(degree=degree)

        results = [
            dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
            dhc.Br(),
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results


@app.callback(
    Output('page-3-display-value-3', 'children'),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-groups", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        
        result = trajectory.plot_groups(degree=degree)

        results = [
            dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
            dhc.Br(),
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results

@app.callback(
    Output('page-3-display-value-4', 'children'),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-longitudinal", "value"),
)
def display_value(trajectory_path, degree):
    results = []
    if trajectory_path:
        trajectory = get_trajectory(trajectory_path)
        
        result = trajectory.plot_animated_longitudinal_information(degree=degree)

        results = [
            dcc.Markdown(result["ret_val"], dangerously_allow_html=True),
            dhc.Br(),
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results
