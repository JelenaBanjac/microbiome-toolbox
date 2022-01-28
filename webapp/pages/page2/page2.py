# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from utils.constants import methods_location

layout = dhc.Div(
    id="page-2-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Reference Definition",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                                There are two ways to define the reference set in the dataset:
                                    1. _predefined by user (on raw data)_: all samples that belong to the reference are specified by user in the uploaded dataset (samples where `reference_group==True`). Other samples are considered to be non-reference samples. If uploaded dataset does not have `reference_group` column, it will be created automatically with all `True` values. This means that all samples will be considered as reference samples. 
                                    2. _unsupervised anomaly detection (on raw data)_ performs novelty and outlier detection. The algorithm uses the user's reference definition as a start (samples where `reference_group==True`) and decides whether a new observation from unlabeled samples belong to the reference or not. For the metric we use [Bray-Curtis distance](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity).
                                
                                Below we also analyse the features important in each of the groups. To find the features that differentiate the two groups (reference vs. non-reference group), we train the binary classification model (using supervised ensemble methods `XGBClassifier` or `RandomForestClassifier`) with confusion matrix. The confusion matrix enables the insight on how good the separation between the two groups is.

                                To find the features that differentiate the two groups (reference vs non-reference group), we train the binary classification model (using supervised ensemble methods `XGBClassifier` or `RandomForestClassifier`) with confusion matrix.
                                The confusion matrix enables the insight on how good the separation between the two groups is.
                                """
                            ),
                            dcc.Markdown(
                                """
                                The examples that are not in the dashboard can be found in the [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox) repository.
                                """,
                            ),
                        ]
                    )
                ),
                dhc.Br(),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Refresh",
                            outline=True,
                            color="dark",
                            id="button-refresh-reference-statistics",
                            n_clicks=0,
                        ),
                        width=12,
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-2-1",
                            children=dhc.Div(id="page-2-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                dcc.Link(
                    "Back", 
                    href=methods_location,
                    style={
                        "textAlign": "center",
                    },
                ),
                dhc.Br(),
                dhc.Br(),
            ]
        )
    ],
)
