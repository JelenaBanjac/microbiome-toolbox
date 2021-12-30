from dash.dependencies import Input, Output, State
from app import app
from pages.home.home_data import get_dataset
from dash import dcc
from dash import html as dhc
import dash
import dash_bootstrap_components as dbc
import traceback


@app.callback(
    Output("page-2-display-value-1", "children"),
    Input("microbiome-dataset-location", "data"),
)
def display_value(dataset_path):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        result = dataset.plot_bacteria_abundances()

        results = [
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results


@app.callback(
    Output("page-2-display-value-2", "children"),
    Input("microbiome-dataset-location", "data"),
)
def display_value(dataset_path):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        try:
            result = dataset.plot_bacteria_abundance_heatmaps()

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
                            "Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an [email](msjelenabanjac@gmail.com)."
                        ),
                    ],
                    color="danger",
                )
            ]
    return results


@app.callback(
    Output("page-2-display-value-3", "children"),
    Input("microbiome-dataset-location", "data"),
)
def display_value(dataset_path):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        result = dataset.plot_ultradense_longitudinal_data()

        results = [
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results


@app.callback(
    Output("page-2-display-value-4", "children"),
    Input("microbiome-dataset-location", "data"),
    Input("embedding-dimension", "value"),
)
def display_value(dataset_path, embedding_dimension):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        result = dataset.embedding_to_latent_space(
            embedding_dimension=embedding_dimension
        )

        results = [
            dcc.Graph(figure=result["fig"], config=result["config"]),
        ]
    return results


@app.callback(
    Output("page-2-display-value-5", "children"),
    Input("microbiome-dataset-location", "data"),
)
def display_value(dataset_path):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        vbox = dataset.embeddings_interactive_selection_notebook()

        results = [
            dcc.Graph(figure=vbox.children[0], id="interactive-embeddings"),
            dhc.Div(id="interactive-embeddings-info"),
            dhc.Br(),
        ]
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
            img_src.update_layout(height=400, width=400)
            confusion_matrix = dcc.Graph(figure=img_src)

            results = [
                dcc.Graph(figure=t),
                dcc.Graph(figure=ffig),
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
    return results
