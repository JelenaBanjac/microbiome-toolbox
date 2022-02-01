# from dash import html as dhc
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
from utils.constants import methods_location

layout = dhc.Div(
    id="page-3-layout",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Link("Back", href=methods_location),
                            dhc.H3(
                                "Microbiome Trajectory",
                                style={
                                    "textAlign": "center",
                                },
                            ),
                            dhc.Br(),
                            dcc.Markdown(
                                """
                                Microbiome trajectory is often used in the microbiome research as a visualization showing the microbiome development with time.
                                The reference samples are the samples that are used to build the microbiome trajectory.
                                Using machine learning algorithms, we predict a Microbiome Maturation Index (MMI) as a function of time for each sample.
                                A smooth fit is then used to obtain the trajectory.
                                With the visualizations below we hope to discover the microbiome trajectory of a given dataset.
                                """,
                            ),
                            # dcc.Markdown(
                            #     """
                            #     The microbiome trajectory is influenced by many factors. 
                            #     For example, nutrition is a critical modulator of the microbiome since it can affect the microbiome present in the subjects.
                            #     We expect to see the trajectory where the only linear part is the initial part until some point (in human microbiome it is usually until 2 years).
                            #     After that, the trajectory plateaus.
                            #     This means that it becomes much harder to determine infant's age based on its microbiota status if the infants is older than 2 years.
                            #     """,
                            # ),
                            dcc.Markdown(
                                """
                                Available techniques to decrease the size of the model while still keeping its performance are:
                                - top K important features selection based on the smallest MAE error (i.e. how does trajectory performance looks like when only working with top 5 or top 10 bacteria used for the model),
                                - remove near zero variance features,
                                - remove correlated features.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                To make sure the performance is still good, we show the plots in Feature extraction part.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                Microbiome trajectory plots contain following information:
                                - only mean line,
                                - only line with prediction interval and confidence interval,
                                - line with samples,
                                - longitudinal data, every subject,
                                - coloring per group (e.g. per country),
                                - red-color dots or stars-shapes are outliers.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                Measuring the trajectory performance (all before plateau area):
                                - MAE (goal: smaller),
                                - R^2 score (goal: bigger), percent of variance captured,
                                - Pearson correlation (MMI, age_at_collection),
                                - Prediction Interval (PI) is prediction interval 95%, the interval in which we expect the healthy reference to fall in (goal: smaller),
                                - standard deviation of the error,
                                - visual check.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                We used two different statistical analysis tools that are used to compare the significant difference between two trajectories:  
                                - _Splinectomy longitudinal statistical analysis tools_: The 3 methods used are translated from R to Python and accommodated for our project. The original package is called [`splinectomeR`](https://github.com/RRShieldsCutler/splinectomeR), implemented in R. For more details please check [Shields-Cutler et al. SplinectomeR Enables Group Comparisons in Longitudinal Microbiome Studies](https://www.frontiersin.org/articles/10.3389/fmicb.2018.00785/full). In short, the test compares whether the area between two polynomial lines is significantly different. Our trajectory lines are the polynomial lines with degrees 2 or 3.  
                                - _Linear regression statistical analysis tools_: A statistical method based on comparing the two linear lines (line y=k*x+n). To compare two linear lines, we compare the significant difference in the two coefficients that represent the line k (slope) and n (y-axis intersection).  
                                """,
                            ),
                            dcc.Markdown(
                                """
                                Some of the available plot options:
                                - if plot is not loaded, click Refresh button,
                                - hovering above the plot shows more information on the samples,
                                - clicking on the labels on the legend can show/hide the clicked item from the plot,
                                - reset plot to initial state is enabled by clicking Home option or Refresh button,
                                - plots can be downloaded in SVG format,
                                - a p-value less than 0.05 (typically ≤ 0.05) is statistically significant (i.e. lines are different). A p-value higher than 0.05 (> 0.05) is not statistically significant and indicates strong evidence for the null hypothesis (H0: two lines have similar slope/intersection/etc.). This means we retain the null hypothesis and reject the alternative hypothesis.
                                """,
                            ),
                            dcc.Markdown(
                                """
                                The examples that are not in the dashboard can be found in the [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox) repository.
                                """,
                            ),
                        ]
                    )
                ),
                # Reference trajectory
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3("Feature extraction"),
                dhc.Br(),
                dcc.Markdown(
                    """
                    Metrics used to evaluate the performances of different model sizes are:  
                    - [`mean_squared_error`](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)  
                    - [`R2`](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)  
                    """,
                ),
                dcc.Markdown(
                    """
                    The x-axis shows the number of features used to train the model, y-axis shows the performance value.
                    Below we see the feature importance for the microbiome trajectory (sorted from the most important to the least important feature).
                    """,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-feature-extraction",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-0",
                            children=dhc.Div(id="page-3-display-value-0"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(dhc.P(), width=3),
                        dbc.Col(
                            dcc.Loading(
                                id="loading-3-0-table",
                                children=dhc.Div(id="page-3-display-value-0-table"),
                                type="default",
                            ),
                            width=6,
                        ),
                        dbc.Col(dhc.P(), width=3),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                # Reference trajectory
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3("Reference trajectory"),
                dhc.Br(),
                dcc.Markdown(
                    """
                    The microbiome trajectory is built on the reference samples.
                    The plot below shows reference samples, its prediction and confidence intervals with mean.
                    """,
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-reference-trajectory",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-reference-trajectory",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("X-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="xaxis-delta-tick-reference-trajectory",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Y-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="yaxis-delta-tick-reference-trajectory",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Figure height: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="height-reference-trajectory",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=900,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Figure width: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="width-reference-trajectory",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=1200,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-1",
                            children=dhc.Div(id="page-3-display-value-1"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Reference groups
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3("Reference groups"),
                dhc.Br(),
                dcc.Markdown(
                    """
                    If dataset has reference and non-reference samples, both lines will be visualized separately with their corresponding samples, prediction and confidence intervals with mean.
                    """,
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-reference-groups",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-reference-groups",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("X-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="xaxis-delta-tick-reference-groups",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Y-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="yaxis-delta-tick-reference-groups",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Figure height: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="height-reference-groups",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=900,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Figure width: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="width-reference-groups",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=1200,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-2",
                            children=dhc.Div(id="page-3-display-value-2"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Groups
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3("Groups"),
                dhc.Br(),
                dcc.Markdown(
                    """
                    If dataset has several groups, all lines will be visualized separately with their corresponding samples, prediction and confidence intervals with mean.
                    """,
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-groups",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-groups",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("X-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="xaxis-delta-tick-groups",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Y-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="yaxis-delta-tick-groups",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Figure height: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="height-groups",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=900,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Figure width: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="width-groups",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=1200,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-3",
                            children=dhc.Div(id="page-3-display-value-3"),
                            type="default",
                        ),
                        width=12,
                    ),
                ),
                dhc.Br(),
                dhc.Br(),
                # Longitudinal information
                dhc.Br(),
                dhc.Hr(),
                dhc.Br(),
                dhc.H3("Longitudinal information"),
                dhc.Br(),
                dcc.Markdown(
                    """
                    Animated longitudinal information of reference samples.
                    """,
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Refresh",
                                    outline=True,
                                    color="dark",
                                    id="button-refresh-longitudinal",
                                    n_clicks=0,
                                ),
                                dhc.I(
                                    title="Refresh plots if not loaded",
                                    className="fa fa-info-circle",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=10),
                    ]
                ),
                dhc.Br(),
                dcc.Markdown("<b>Plot settings</b>", dangerously_allow_html=True),
                dbc.Row(
                    [
                        dbc.Col("Polynomial degree: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="polynomial-degree-longitudinal",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=8),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("X-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="xaxis-delta-tick-longitudinal",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Y-axis Δtick: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="yaxis-delta-tick-longitudinal",
                                type="number",
                                min=1,
                                # max=20,
                                step=1,
                                # value=1,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Figure height: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="height-longitudinal",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=900,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                        dbc.Col(dhc.P(), width=1),
                        dbc.Col("Figure width: ", width=2),
                        dbc.Col(
                            dcc.Input(
                                id="width-longitudinal",
                                type="number",
                                min=500,
                                # max=20,
                                step=1,
                                value=1200,
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=2,
                        ),
                    ]
                ),
                dhc.Br(),
                dhc.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Loading(
                            id="loading-3-4",
                            children=dhc.Div(id="page-3-display-value-4"),
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
