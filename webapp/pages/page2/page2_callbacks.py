from dash.dependencies import Input, Output, State
from app import app
from pages.home.home_data import get_dataset
from dash import dcc
from dash import html as dhc
import dash
import dash_bootstrap_components as dbc
import traceback
from microbiome.enumerations import EmbeddingModelType
import numpy as np

@app.callback(
    Output("page-2-display-value-1", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("abundances-number-of-columns", "value"),
    Input("height-abundances", "value"),
    Input("width-abundances", "value"),
    Input("xaxis-delta-tick-abundances", "value"),
    Input("yaxis-delta-tick-abundances", "value"),
)
def display_value(dataset_path, number_of_columns, height_row, width, x_delta, y_delta):
    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)

            number_of_rows = len(dataset.bacteria_columns) // number_of_columns + 1

            layout_settings = dict(
                height=height_row*number_of_rows,
                width=width,
            )
            xaxis_settings = dict(
                dtick=x_delta,
            )
            yaxis_settings = dict(
                dtick=y_delta,
            )

            result = dataset.plot_bacteria_abundances(
                number_of_columns=number_of_columns,
                layout_settings=layout_settings,
                xaxis_settings=xaxis_settings,
                yaxis_settings=yaxis_settings,
            )

            results = [
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
    Output("page-2-display-value-2", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("heatmap-relative-absolute-values", "value"),
    Input("heatmap-fillna-dropna", "value"),
    Input("heatmap-avg-fn", "value"),
    Input("height-heatmap", "value"),
    Input("width-heatmap", "value"),
)
def display_value(dataset_path, relative_values, empty_cells, avg_fn, height_row, width):
    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)

            number_of_bacteria = len(dataset.bacteria_columns)

            layout_settings = dict(
                height=height_row*number_of_bacteria,
                width=width,
            )

            result = dataset.plot_bacteria_abundance_heatmaps(
                relative_values=relative_values=="relative",
                fillna=empty_cells=="fill",
                dropna=empty_cells=="drop",
                avg_fn=np.median if avg_fn=="median" else np.mean,
                layout_settings=layout_settings,
            )

            results = [
                dcc.Graph(figure=result["fig"], config=result["config"]),
            ]
        except Exception as e:
            results = [
                dbc.Alert(
                    children=[
                        dcc.Markdown("Dataset/Plot error: " + str(e)),
                        dcc.Markdown(traceback.format_exc()),
                        dcc.Markdown(
                            "Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an email to <msjelenabanjac@gmail.com>."
                        ),
                    ],
                    color="danger",
                )
            ]
    return results


@app.callback(
    Output("page-2-display-value-3", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("longitudinal-number-of-columns", "value"),
    Input("longitudinal-number-of-bacteria", "value"),
    Input("height-longitudinal-stack", "value"),
    Input("width-longitudinal-stack", "value"),
    Input("xaxis-delta-tick-longitudinal-stack", "value"),
    Input("yaxis-delta-tick-longitudinal-stack", "value"),
)
def display_value(dataset_path, number_of_columns, number_of_bacteria, height, width, x_delta, y_delta):
    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)

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

            result = dataset.plot_ultradense_longitudinal_data(
                number_of_columns=number_of_columns,
                number_of_bacteria=number_of_bacteria,
                layout_settings=layout_settings,
                xaxis_settings=xaxis_settings,
                yaxis_settings=yaxis_settings,
            )

            results = [
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
    Output("page-2-display-value-4", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("embedding-dimension", "value"),
    Input("embedding-method-type", "value"),
    Input("height-embedding", "value"),
    Input("width-embedding", "value"),
)
def display_value(dataset_path, embedding_dimension, embedding_method_type, height, width):
    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)

            layout_settings = dict(
                height=height,
                width=width,
            )

            embedding_model = EmbeddingModelType[embedding_method_type].value(n_components=embedding_dimension)

            result = dataset.embedding_to_latent_space(
                embedding_dimension=embedding_dimension,
                embedding_model=embedding_model,
                layout_settings=layout_settings,
            )

            results = [
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
    Output("page-2-display-value-5", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("embedding-method-type-interactive", "value"),
    Input("height-embedding-interactive", "value"),
    Input("width-embedding-interactive", "value"),
)
def display_value(dataset_path, embedding_method_type, height, width):
    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)

            layout_settings = dict(
                height=height,
                width=width,
            )

            embedding_model = EmbeddingModelType[embedding_method_type].value(n_components=2)

            result = dataset.embeddings_interactive_selection_notebook(
                embedding_model=embedding_model,
                layout_settings=layout_settings,
            )
            vbox = result["vbox"]
            config = result["config"]

            results = [
                dcc.Graph(figure=vbox.children[0], id="interactive-embeddings", config=config),
                dhc.Div(id="interactive-embeddings-info"),
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
    Output("interactive-embeddings-info", "children"),
    [
        Input("interactive-embeddings", "selectedData"),
        Input("microbiome-dataset-location", "data"),
    ],
    [State("interactive-embeddings", "figure")],
)
def display_value(selectedData, dataset_path, fig):
    import plotly.graph_objects as go

    results = []
    if dataset_path:
        try:
            dataset = get_dataset(dataset_path)
            selection = None
            # Update selection based on which event triggered the update.
            trigger = dash.callback_context.triggered[0]["prop_id"]
            # if trigger == 'graph.clickData':
            #     selection = [point["pointNumber"] for point in clickData["points"]]
            if trigger == "interactive-embeddings.selectedData":
                selection = [point["pointIndex"] for point in selectedData["points"]]

            if selection is not None:
                # Update scatter selection
                fig["data"][0]["selectedpoints"] = selection

                # Create a table FigureWidget that updates on selection from points in the scatter plot of f
                t = go.FigureWidget(
                    [
                        go.Table(
                            header=dict(
                                values=["sampleID", "subjectID"],
                                fill=dict(color="#C2D4FF"),
                                align=["left"] * 5,
                            ),
                            cells=dict(
                                values=[
                                    dataset.df[col] for col in ["sampleID", "subjectID"]
                                ],
                                fill=dict(color="#F5F8FF"),
                                align=["left"] * 5,
                            ),
                        )
                    ]
                )

                results = dataset.selection_embeddings(t)(
                    None, fig["data"][0]["selectedpoints"], None
                )

                ffig = results["fig"]
                img_src = results["img_src"]
                acccuracy = results["accuracy"]
                f1score = results["f1score"]
                config = results["config"]
                img_src.update_layout(height=400, width=400)
                confusion_matrix = dcc.Graph(figure=img_src)

                results = [
                    dcc.Graph(figure=t),
                    dcc.Graph(figure=ffig, config=config),
                    dhc.Br(),
                    dbc.Container(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dhc.Br(),
                                        dhc.H5("Groups discrimination performance results"),
                                        dhc.Br(),
                                        dhc.Br(),
                                        dhc.P(
                                            "The ideal separation between two groups (reference vs. non-reference) will have 100% of values detected on the second diagonal. This would mean that the two groups can be easily separated knowing their taxa abundamces and metadata information."
                                        ),
                                        dhc.P(f"Accuracy: {acccuracy:.2f}"),
                                        dhc.P(f"F1-score: {f1score:.2f}"),
                                    ]
                                ),
                                dbc.Col(confusion_matrix),
                            ]
                        )
                    ),
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
