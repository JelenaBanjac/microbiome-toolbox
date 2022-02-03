# from dash import html as dhc
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
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
                                - _predefined by user_ (i.e. `USER_DEFINED`): all samples that belong to the reference are specified by user in the uploaded dataset (samples where `reference_group==True`). Other samples are considered to be non-reference samples. If uploaded dataset does not have `reference_group` column, it will be created automatically with all `True` values. This means that all samples will be considered as reference samples.  
                                - _unsupervised anomaly detection_ (i.e. `NOVELTY_DETECTION`) performs novelty and outlier detection. The algorithm uses the user's reference definition as a start (samples where `reference_group==True`) and decides whether a new observation from unlabeled samples belong to the reference or not. We use [`LocalOutlierFactor`](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html) method with default [Bray-Curtis distance](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity) metric and 2 neighbors. These parameters can be modified on the Home page. The features that are used for this model are specified under Dataset Settings feature columns option on Home page (when novelty detection option is selected). These features are not necessary matching the feature columns used for building the microbiome trajectory. If dashboard user selects novelty detection option, it might not yield the optimal results for default parameters (fixed parameters cannot generalize accross different datasets). Hence, we suggest to play with the parameters (metric and number of neighbors). 
                                """
                            ),
                            dcc.Markdown(
                                """
                                Below we also analyse the features important in each of the groups. To find the features that differentiate the two groups (reference vs. non-reference group), we train the binary classification model [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and perform cross validation with [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). The parameters we use for the classifier are `n_estimators=140` and `max_samples=0.8`, and they are fixed. To change these values for parameters, you would need to play with the toolbox locally. The confusion matrix enables the insight on how good the separation between the two groups is.
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
