from dash.dependencies import Input, Output
from app import app
from pages.home.home_data import get_trajectory
from dash import dcc
from dash import html as dhc
import traceback
import dash_bootstrap_components as dbc
from dash import dash_table


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
                    dcc.Markdown("Feature extraction has not been used. All features in the dataset are used to build the trajectory."),
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results


@app.callback(
    Output("page-3-display-value-0-table", "children"),
    Input("microbiome-trajectory-location", "data"),
    # Input("button-refresh-feature-extraction", "n_clicks"),
)
def display_value(trajectory_path):
    results = []
    if trajectory_path:
        try:
            trajectory = get_trajectory(trajectory_path)

            data = trajectory.feature_importance.to_dict("records")
            columns = [
                {"name": i, "id": i, "deletable": True, "renamable": True} for i in trajectory.feature_importance.columns
            ]
            tooltip_header = {i: i for i in trajectory.feature_importance.columns}
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in trajectory.feature_importance.to_dict('records')
            ]

            table = dash_table.DataTable(
                id='datatable-important-features',
                # style_data={
                #     'width': f'{max(df_dummy.columns, key=len)}%',
                #     'minWidth': '50px',
                #     'maxWidth': '500px',
                # },
                style_table={
                    'height': 300, 
                    'overflowX': 'auto'
                },
                style_cell={
                    'height': 'auto',
                    # all three widths are needed
                    'minWidth': '200px', 
                    # 'width': f'{max(df_dummy.columns, key=len)}%',
                    'maxWidth': '200px',
                    'whiteSpace': 'normal'
                },
                # Style headers with a dotted underline to indicate a tooltip
                style_header={
                    'textDecoration': 'underline',
                    'textDecorationStyle': 'dotted',
                },
                editable=True, 
                export_format='csv',
                export_headers='display',
                merge_duplicate_headers=True,
                tooltip_delay=0,
                tooltip_duration=None,
                columns=columns,
                tooltip_header=tooltip_header,
                data=data,
                tooltip_data=tooltip_data,
            )

            results = [
                dhc.H3(
                    "Feature importance",
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                table,
                dhc.Br(),
            ]
            
        except Exception as e:
            results = dbc.Alert(
                children=[
                    dcc.Markdown("Microbiome trajectory error: " + str(e)),
                    dcc.Markdown(traceback.format_exc()),
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results

@app.callback(
    Output("page-3-display-value-1", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-trajectory", "value"),
    Input("height-reference-trajectory", "value"),
    Input("width-reference-trajectory", "value"),
    Input("xaxis-delta-tick-reference-trajectory", "value"),
    Input("yaxis-delta-tick-reference-trajectory", "value"),
)
def display_value(trajectory_path, degree, height, width, x_delta, y_delta):
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
            result = trajectory.plot_reference_trajectory(
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results


@app.callback(
    Output("page-3-display-value-2", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-reference-groups", "value"),
    Input("height-reference-groups", "value"),
    Input("width-reference-groups", "value"),
    Input("xaxis-delta-tick-reference-groups", "value"),
    Input("yaxis-delta-tick-reference-groups", "value"),
)
def display_value(trajectory_path, degree, height, width, x_delta, y_delta):
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

                result = trajectory.plot_reference_groups(
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results


@app.callback(
    Output("page-3-display-value-3", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-groups", "value"),
    Input("height-groups", "value"),
    Input("width-groups", "value"),
    Input("xaxis-delta-tick-groups", "value"),
    Input("yaxis-delta-tick-groups", "value"),
)
def display_value(trajectory_path, degree, height, width, x_delta, y_delta):
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

            result = trajectory.plot_groups(
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results


@app.callback(
    Output("page-3-display-value-4", "children"),
    Input("microbiome-trajectory-location", "data"),
    Input("polynomial-degree-longitudinal", "value"),
    Input("height-longitudinal", "value"),
    Input("width-longitudinal", "value"),
    Input("xaxis-delta-tick-longitudinal", "value"),
    Input("yaxis-delta-tick-longitudinal", "value"),
)
def display_value(trajectory_path, degree, height, width, x_delta, y_delta):
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


            result = trajectory.plot_animated_longitudinal_information(
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
                    dcc.Markdown("Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."),
                ],
                color="danger",
            )
    return results
