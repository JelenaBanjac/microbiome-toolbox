# from dash import html as dhc
from dash import dcc
from dash import html as dhc
import dash_bootstrap_components as dbc
from utils.constants import home_location

layout = dhc.Div(
    id="page-1-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=home_location),
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
                                1. _predefined by user (on raw data)_: all samples that belong to the reference are specified by user in the uploaded dataset (with the `True` value in the `reference_group` column). 
                                Other samples are considered to be non-reference samples.  
                                2. _unsupervised anomaly detection (on raw data)_ where we don't feed the algorithm about our true differentiation:
                                Performs novelty and outlier detection -- use the user's reference definition as a start and decide whether a new observation from other belongs to the reference or not. 
                                For the metric we use [Bray-Curtis distance](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity).
                            The column for this property is called `reference_group` and it contails only `True`/`False` values.

                            Bellow we also analyse the features important in each of the groups. 
                            To find the features that differentiate the two groups (reference vs non-reference group), we train the binary classification model (using supervised ensemble methods `XGBClassifier` or `RandomForestClassifier`) with confusion matrix.
                            The confusion matrix enables the insight on how good the separation between the two groups is.
                            """,
                                style={
                                    "textAlign": "left",
                                },
                            ),
                        ]
                    )
                ),
                dhc.Br(),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-1-1",
                            children=dhc.Div(id="page-1-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
            ]
        )
    ],
)
