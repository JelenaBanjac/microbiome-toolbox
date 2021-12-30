from dash.dependencies import Input, Output
from app import app
from pages.home.home_data import get_dataset
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc


@app.callback(
    Output("page-1-display-value-1", "children"),
    Input("microbiome-dataset-location", "data"),
)
def display_value(dataset_path):
    results = []
    if dataset_path:
        dataset = get_dataset(dataset_path)

        if dataset.reference_group_plot is None:
            results = [
                dcc.Markdown("Reference groups have not been defined yet."),
            ]
        else:
            img_src = dataset.reference_group_img_src
            acccuracy = dataset.reference_group_accuracy
            f1score = dataset.reference_group_f1score
            img_src.update_layout(height=400, width=400)
            confusion_matrix = dcc.Graph(figure=img_src)

            results = [
                dcc.Graph(
                    figure=dataset.reference_group_plot,
                    config=dataset.reference_group_config,
                ),
                dhc.Br(),
                dbc.Container(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dhc.Br(),
                                    dhc.Br(),
                                    dcc.Markdown(
                                        f"Reference groups differentiation can be seen below."
                                    ),
                                    dcc.Markdown(
                                        "The ideal separation between two groups (reference vs. non-reference) will have 100% of values detected on the second diagonal. This would mean that the two groups can be easily separated knowing their taxa abundamces and metadata information."
                                    ),
                                    dcc.Markdown(f"Accuracy: {acccuracy*100:.2f}"),
                                    dcc.Markdown(f"F1-score: {f1score:.2f}"),
                                ]
                            ),
                            dbc.Col(confusion_matrix),
                        ]
                    )
                ),
                dhc.Br(),
            ]
    return results
