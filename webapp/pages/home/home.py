import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
import dash_uploader as du
from dash import dash_table
from dash_uploader import upload
import uuid
from utils.constants import home_location, page1_location, page2_location, page3_location, page4_location, page5_location, page6_location
from microbiome.enumerations import Normalization, ReferenceGroup, TimeUnit, AnomalyType, FeatureExtraction, FeatureColumnsType


mouse_data_modal = dbc.Modal(
    [
        dbc.ModalHeader("Mouse dataset references"),
        dbc.ModalBody([
            dcc.Markdown("""
                Relevant description of data: All mice were fed the same low-fat, plant polysaccharide–rich diet for the first 21 days of the study. At this point 6 of the mice were then switched to a high-fat, high-sugar “Western” diet. The subsequent changes in the microbial community were then observed over a follow-up of roughly 60 days.
            """),
            dcc.Markdown("""
                Humanized gnotobiotic mouse gut [2]: Twelve germ-free adult male C57BL/6J mice were
                fed a low-fat, plant polysaccharide-rich diet. Each mouse was gavaged with healthy adult
                human fecal material. Following the fecal transplant, mice remained on the low-fat, plant
                polysacchaaride-rich diet for four weeks, following which a subset of 6 were switched to a
                high-fat and high-sugar diet for eight weeks. Fecal samples for each mouse went through
                PCR amplification of the bacterial 16S rRNA gene V2 region weekly. Details of experimental protocols and further details of the data can be found in Turnbaugh et. al.
                Sequences and further information can be found at: http://gordonlab.wustl.edu/
                TurnbaughSE_10_09/STM_2009.html
            """),
            dhc.Br(),
            dhc.Img(src="https://www.ncbi.nlm.nih.gov/pmc/articles/instance/2894525/bin/nihms209492f1.jpg",width="100%"),
            dcc.Markdown("""
                **Design of human microbiota transplant experiments** (A) The initial (first-generation) humanization procedure, including the diet shift. Brown arrows indicate fecal collection time points. (B) Reciprocal microbiota transplantations. Microbiota from first-generation humanized mice fed LF/PP or Western diets were transferred to LF/PP or Western diet-fed germ-free recipients. (C) Colonization of germ-free mice starting with a frozen human fecal sample. (D) Characterization of the postnatal assembly and daily variation of the humanized mouse gut microbiota. (E) Sampling of the humanized mouse gut microbiota along the length of the gastrointestinal tract. [Source](https://pubmed.ncbi.nlm.nih.gov/20368178/#&gid=article-figures&pid=fig-1-uid-0)
            """),
            dhc.Br(),
            dcc.Markdown("""
               References:  
               (1) [Joseph Nathaniel Paulson, 2016, metagenomeSeq: Statistical analysis for sparse high-throughput sequencing](https://bioconductor.org/packages/release/bioc/vignettes/metagenomeSeq/inst/doc/metagenomeSeq.pdf).  
               (2) Package page: [`metagenomeSeq`](https://bioconductor.org/packages/release/bioc/html/metagenomeSeq.html).  
               (3) [Turnbaugh PJ, Ridaura VK, Faith JJ, Rey FE, Knight R, Gordon JI. The effect of diet on the human gut microbiome: a metagenomic analysis in humanized gnotobiotic mice](https://pubmed.ncbi.nlm.nih.gov/20368178/).  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="mouse-data-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="mouse-data-modal",
    scrollable=True,
    is_open=False,
)

human_data_modal = dbc.Modal(
    [
        dbc.ModalHeader("Human infants dataset references"),
        dbc.ModalBody([
            dcc.Markdown("""
                Dataset we used is taken from (3) and only around 80 samples are selected to be used on the web dashboard due to its size.
            """),
            dcc.Markdown("""
               References:  
               (1) [Subramanian et al. Persistent Gut Microbiota Immaturity in Malnourished Bangladeshi Children](https://gordonlab.wustl.edu/supplemental-data/supplemental-data-portal/subramanian-et-al-2014/), raw data.  
               (2) [The effects of exclusive breastfeeding on infant gut microbiota: a meta-analysis across populations](https://zenodo.org/record/1304367#.YfMX3ITMIkl), some processing included on raw data.  
               (3) [Meta-analysis of effects of exclusive breastfeeding on infant gut microbiota across populations dataset](https://github.com/nhanhocu/metamicrobiome_breastfeeding)
               (4) [Ho NT, et al., Meta-analysis of effects of exclusive breastfeeding on infant gut microbiota across populations](https://www.nature.com/articles/s41467-018-06473-x).  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="human-data-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="human-data-modal",
    scrollable=True,
    is_open=False,
)

custom_data_modal = dbc.Modal(
    [
        dbc.ModalHeader("Rules for custom dataset"),
        dbc.ModalBody([
            dcc.Markdown(
            """
            In order for the methods to work, make sure the uploaded dataset has the following columns:
            """),
            dcc.Markdown("""
                * `sampleID`: a unique dataset identifier, the ID of a sample,
                * `subjectID`: an identifier of the subject (i.e. mouse name),
                * `age_at_collection`: the time at which the sample was collected, should be in DAYS,
                * all other required columns in the dataset should be bacteria names which will be automatically prefixed with bacteria_* after the upload.
            """),
            dcc.Markdown("""
            Optional columns:
            """),
            dcc.Markdown("""
                * `reference_group`: with `True`/`False` values (e.g. `True` is a healthy sample, `False` is a non-healthy sample); if this column is not specified, it will be automatically created with all True values, therefore, all samples will belong to one reference group,
                * `group`: the groups that are going to be compared (e.g. `country`); if this column is not specified, we won’t have the visualization of different groups separately,
                * `meta_*`: prefix for metadata columns (e.g. `c-section` becomes `meta_csection`, etc.),
                * `id_*`: prefix for other ID columns (don't prefix `sampleID` nor `subjectID`).

            """
            ),
            dcc.Markdown("""
                **Important**: the uploaded dataset should be in a csv file.
            """),


        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="custom-data-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="custom-data-modal",
    scrollable=True,
    is_open=False,
)

differentiation_score_modal = dbc.Modal(
    [
        dbc.ModalHeader("Differentiation score"),
        dbc.ModalBody([
            dcc.Markdown("""
               Differentiation score tells us how good the samples from a reference group are separable from the samples from the non-reference group.
               The measure we use is [F1-score](https://deepai.org/machine-learning-glossary-and-terms/f-score#:~:text=The%20F%2Dscore%2C%20also%20called,positive'%20or%20'negative'.) since the underlaying model is a binary classifier.
               Under the hood, we train a binary classifier to differentiate between two groups of samples (reference and non-reference). The result of this classification is the F1-score. 
            """),
            dcc.Markdown("""
               Higher F1-score (closer to 1) means that the samples from the reference group are more likely to be separable from the samples from the non-reference group. Low values of F1-score (closer to 0) indicate that the samples from the reference group are less likely to be separable from the samples from the non-reference group.
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="differentiation-score-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="differentiation-score-modal",
    scrollable=True,
    is_open=False,
)

feature_columns_dataset_modal = dbc.Modal(
    [
        dbc.ModalHeader("Feature columns"),
        dbc.ModalBody([
            dcc.Markdown("""
               The selection of this option is important only if user has selected the `NOVELTY_DETECTION` in Reference group options.
               Main assumption: the samples that do not belong to the reference group are considered anomaly. We assume that the reference samples are the majority sample representation of the dataset.
               On the other side, the non-reference samples are the minority and considered/assumed to be anomaly of the dataset.
               Therefore, we use the novelty detection algorithm as an anomaly detection algorithm. More concretely, we use Local Outlier Factor method (LOF).
            """),
            dcc.Markdown("""
                The novelty detection algorithm works the following way: take all the samples where `reference_group==True`, and find the samples that are the most similar to them and re-label them to be `True`. 
                The remaining set of samples is labeled `False`, i.e., they are considered as anomalies compared to the reference. 
            """),
            dcc.Markdown("""
               Available options:  
               1. `BACTERIA`: the features are only bacteria abundance information,  
               2. `METADATA`: the features are only metadata information,  
               3. `BACTERIA_AND_METADATA`: all features are used for novelty detection,  
            """),
            dcc.Markdown("""
                Important: if your dataset does not have the `reference_group` column, there is no point in using `NOVELTY_DETECTION`. Select `USER_DEFINED` option instead.
            """),
            dcc.Markdown("""
               References:  
               (1) Novelty and Outlier detection in [sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-and-outlier-detection).  
               (2) Local outlier factor [LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor).  
               (3) [Breunig et al. 2000, LOF: identifying density-based local outliers](https://dl.acm.org/doi/10.1145/335191.335388).  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="feature-columns-dataset-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="feature-columns-dataset-modal",
    scrollable=True,
    is_open=False,
)

reference_group_modal = dbc.Modal(
    [
        dbc.ModalHeader("Reference group"),
        dbc.ModalBody([
            dcc.Markdown("""
                The reference group is the group of samples that are going to be used as a reference for the microbiome trajectory creation.
                The reference group is defined in `reference_group` column with `True` and `False` values.
                The column is not mandatory, but if it is not specified, all samples will automatically be labeled to belong to the reference group (i.e. `reference_group=True`).
                Information on reference group split analysis can be seen in Reference Definition card.
            """),
            dcc.Markdown("""
                There are two options for the reference group:  
                1. `USER_DEFINED`: the user will specify the reference group in the `reference_group` column,  
                2. `NOVELTY_DETECTION`: the reference group is automatically determined by the novelty detection algorithm.  
            """),
            dcc.Markdown("""
                This can be finetuned further by specifying the feature columns to be used for novelty detection (above drop down option).
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="reference-group-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="reference-group-modal",
    scrollable=True,
    is_open=False,
)

time_unit_modal = dbc.Modal(
    [
        dbc.ModalHeader("Time unit"),
        dbc.ModalBody([
            dcc.Markdown("""
                While it is mandatory to upload dataset with time format in DAYS, you can specify the format it will be visualized in futher data and trajectory analysis.
            """),
            dcc.Markdown("""
                Available time units:  
                1. `DAYS`: the time unit is in days,  
                2. `MONTHS`: the time unit is in months,  
                3. `YEARS`: the time unit is in years.  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="time-unit-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="time-unit-modal",
    scrollable=True,
    is_open=False,
)

normalization_modal = dbc.Modal(
    [
        dbc.ModalHeader("Data normalization"),
        dbc.ModalBody([
            dcc.Markdown("""
                Data normalization is a very important part of data preparation.
                It is important to normalize the data to have a mean of 0 and a standard deviation of 1.
                The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. 
                For machine learning, every dataset does not require normalization. 
                It is required only when features have different ranges.
                Therefore, we offer the option to normalize the data.
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="normalization-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="normalization-modal",
    scrollable=True,
    is_open=False,
)

log_ratio_modal = dbc.Modal(
    [
        dbc.ModalHeader("Log-ratio bacteria"),
        dbc.ModalBody([
            dcc.Markdown("""
            By default, we use bacteria abundances as the features.
            However, if you want to use log-ratio of bacteria abundances w.r.t. the chosen bacteria, select one of the drop down bacteria.
            By choosing one of the bacteria, your whole dataset will be transformed to use log-ratio of bacteria abundances w.r.t. the chosen bacteria.
            Log-ratio is a way to normalize the data.
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="log-ratio-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="log-ratio-modal",
    scrollable=True,
    is_open=False,
)

feature_columns_trajectory_modal = dbc.Modal(
    [
        dbc.ModalHeader("Feature columns"),
        dbc.ModalBody([
            dcc.Markdown("""
                The feature columns that are used to build a microbiome trajectory.
            """),
            dcc.Markdown("""
               Available options:  
               1. `BACTERIA`: the features are only bacteria abundance information,  
               2. `METADATA`: the features are only metadata information,  
               3. `BACTERIA_AND_METADATA`: all features are used for building the trajectory.  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="feature-columns-trajectory-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="feature-columns-trajectory-modal",
    scrollable=True,
    is_open=False,
)

anomaly_type_modal = dbc.Modal(
    [
        dbc.ModalHeader("Anomaly type"),
        dbc.ModalBody([
            dcc.Markdown("""
                The default option is to consider anomaly all the samples that are outside the prediction interval of all the reference samples trajectory.
                Information on detected anomalies can be seen in Anomaly Detection card.
            """),
            dcc.Markdown("""
               Available options:  
               1. `PREDICTION_INTERVAL`: samples outside the PI are considered to be anomalies,  
               2. `LOW_PASS_FILTER`: the samples passing 2 standard deviations of the mean are considered to be anomalies,,  
               3. `ISOLATION_FOREST`: algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.  
            """),
            dcc.Markdown("""
               References:  
               (1) Isolation forest [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).   
                (2) Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on.  
                (3) Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation-based anomaly detection.” ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012)  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="anomaly-type-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="anomaly-type-modal",
    scrollable=True,
    is_open=False,
)

feature_extraction_modal = dbc.Modal(
    [
        dbc.ModalHeader("Feature extraction"),
        dbc.ModalBody([
            dcc.Markdown("""
                This option specifies additional filtering for columns that are used for microbiome trajectory. 
                Information on the performance can be seen in Microbiome Trajectory card.
            """),
            dcc.Markdown("""
               Available options:  
               1. `NONE`: there is no feature extraction, all feature columns are used,  
               2. `NEAR_ZERO_VARIANCE`: remove features with near zero variance,   
               3. `CORRELATION`: remove correlated features.   
               4. `TOP_K_IMPORTANT`: use only top k important features to build microbiome trajectory.  
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="feature-extraction-close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="feature-extraction-modal",
    scrollable=True,
    is_open=False,
)


def serve_upload(session_id):
    upload = [
        dhc.H3(
            "Dataset upload",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        dcc.Markdown(
            """The Microbiome Toolbox implements methods that can be used for microbiome dataset analysis and microbiome trajectory prediction. The dashboard offers a wide variety of interactive visualizations.\
            If you are just interested in seeing what methods are covered, you can use demo datasets (mouse data, human infants data) which enables the toolbox options below (by clicking the button).\
            You can also upload your own dataset (by clicking or drag-and-dropping the file into the area below). More methods and specific details of method implementations can be seen in the Github repository [`microbiome-toolbox`](https://github.com/JelenaBanjac/microbiome-toolbox).
        """
        ),
        dhc.Br(),
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(dhc.Div(dhc.P("Demo datasets:")), width=3),
                        dbc.Col([
                            dbc.Button(
                                "Mouse data",
                                outline=True,
                                color="dark",
                                id="button-mouse-data",
                                n_clicks=0,
                            ),
                            dhc.I(title="More information", id="mouse-data-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            mouse_data_modal,
                            ], width=3
                            
                        ),
                        
                        dbc.Col(dhc.P(), width=6),
                    ]),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(dhc.P(), width=3),
                        dbc.Col([
                            dbc.Button(
                                "Human data",
                                outline=True,
                                color="dark",
                                id="button-human-data",
                                n_clicks=0,
                            ),
                            dhc.I(title="More information", id="human-data-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            human_data_modal,
                            ], width=3
                        ),
                        dbc.Col(dhc.P(), width=6),
                    ]),
                dhc.Br(),
                dbc.Row([
                        dbc.Col(dhc.P("Custom datasets:"), width=3),
                        dbc.Col([
                            dbc.Button(
                                "Custom data",
                                outline=True,
                                color="dark",
                                id="button-custom-data",
                                n_clicks=0,
                                disabled=True,
                            ),
                            dhc.I(title="More information", id="custom-data-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            custom_data_modal,
                            dhc.Div(dhc.Small("Upload your CSV dataset to enable the button."), id="small-upload-info-text", hidden=False),
                        ]),
                ]),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(dhc.P(), width=3),
                        dbc.Col([
                            dhc.Div(dcc.Store(id="upload-data-file-path"), hidden=True),
                            dhc.Div(du.Upload(
                                id="upload-data",
                                filetypes=["csv"],
                                upload_id=session_id,
                            ))
                            ], width=6
                        ),
                        dbc.Col(dhc.P(), width=6),
                    ]),
                dhc.Br(),
                dbc.Row([
                    dbc.Col(width=3),
                    dbc.Col(
                        dcc.Loading(id="loading-boxes", children=dhc.Div(id="upload-infobox"), type="default"), width=6),
                ]),
                # dbc.Row([
                #     dbc.Col(width=3),
                #     dbc.Col(
                #         dcc.Loading(id="loading-boxes", children=dhc.Div(id="dataset-errors"), type="default"), width=6),
                # ]),
            ],
            className="md-12",
            # style={"height": 250},
        ),
        dhc.Br(),
        
    ]
    return upload

def serve_dataset_table():
    file_name = dhc.Div(id="upload-file-name")
    number_of_samples = dhc.Div(id="upload-number-of-samples")
    number_of_subjects = dhc.Div(id="upload-number-of-subjects")
    unique_groups = dhc.Div(id="upload-unique-groups")
    number_of_reference_samples = dhc.Div(id="upload-number-of-reference-samples")
    differentiation_score = dhc.Div(id="upload-differentiation-score")
    table = dash_table.DataTable(
        id='upload-datatable',
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
        tooltip_duration=None
    )

    dataset_table = [
        dhc.Br(),
        dhc.Br(),
        dhc.Br(),
        dhc.H3(
            "Dataset table",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        # dcc.Markdown(f"Currently loaded file: `{filename}`"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col("Loaded file: ", width=3),
                        dbc.Col(file_name, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of samples:", width=3),
                        dbc.Col(number_of_samples, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of subjects:", width=3),
                        dbc.Col(number_of_subjects, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Unique groups:", width=3),
                        dbc.Col(unique_groups, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col("Number of reference samples:", width=3),
                        dbc.Col(number_of_reference_samples, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            "Differentiation score:",
                            dhc.I(title="More information", id="differentiation-score-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            differentiation_score_modal,
                        
                        ], width=3),
                        dbc.Col(differentiation_score, width=6),
                    ]
                ),
                dhc.Br(),
            ],
            className="md-12",
            # style={"height": 250},
        ),

        dcc.Loading(table, id="upload-datatable-loading", type="default"),
        dhc.Br(),
    ]

    return dataset_table

def serve_dataset_settings():
    
    # reference_groups = ["user defined", "novelty detection algorithm decision"]
    feature_columns_choice = dcc.Dropdown(
        id='settings-feature-columns-choice-dataset',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in FeatureColumnsType],
        searchable=True,
        clearable=True,
        placeholder="select feature columns",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    reference_group_choice = dcc.Dropdown(
        id='settings-reference-group-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in ReferenceGroup],
        searchable=True,
        clearable=True,
        placeholder="select reference group",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    time_unit_choice = dcc.Dropdown(
        id='settings-time-unit-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in TimeUnit],
        searchable=True,
        clearable=True,
        placeholder="select time unit",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    

    normalized_choice = dcc.Dropdown(
        id='settings-normalized-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in Normalization],
        searchable=True,
        clearable=True,
        placeholder="select normalization",
        value=None,
        persistence=True,
        persistence_type="session",
    )


    log_ratio_bacteria_choice = dcc.Dropdown(
        id='settings-log-ratio-bacteria-choice',
        optionHeight=20,
        searchable=True,
        clearable=True,
        placeholder="[optional] select a bacteria for log-ratio",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    

    settings = [
        
        dhc.Br(),
        dhc.Br(),
        dhc.Br(),
        dhc.H3(
            "Dataset settings",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col([
                            "Feature columns (for the novelty detection):",
                            dhc.I(title="More information", id="feature-columns-dataset-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            feature_columns_dataset_modal,
                        ], width=3),
                        dbc.Col(feature_columns_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            "Reference group:",
                            dhc.I(title="More information", id="reference-group-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            reference_group_modal,
                        ], width=3),
                        dbc.Col(reference_group_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            "Time unit:",
                            dhc.I(title="More information", id="time-unit-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            time_unit_modal,
                        ], width=3),
                        dbc.Col(time_unit_choice, width=6),
                    ]
                ),
                dhc.Br(),
                
                dbc.Row(
                    [
                        dbc.Col([
                            "Normalization:",
                            dhc.I(title="More information", id="normalization-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            normalization_modal,
                        ], width=3),
                        dbc.Col(normalized_choice, width=6),
                    ]
                ),
                
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            "Log-ratio bacteria:",
                            dhc.I(title="More information", id="log-ratio-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            log_ratio_modal,
                        ], width=3),
                        dbc.Col(log_ratio_bacteria_choice, width=6),
                    ]
                ),
                dhc.Br(),

                dbc.Row(
                    [
                        dbc.Col(dhc.P(), width=3),
                        dbc.Col(
                            [dhc.Div(dbc.Button(
                                "Update dataset",
                                outline=True,
                                color="dark",
                                id="button-dataset-settings-update",
                                n_clicks=0,
                                disabled=True,
                            )),
                            dhc.Br(),
                            dhc.Br(),
                            dcc.Loading(id="loading-boxes-dataset", children=[
                                dhc.Div(id="dataset-settings-infobox"),
                            ], type="default")
                            ]
                            , width=6),
                    ]
                ),
                dhc.Br(),
            ],
            className="md-12",
            # style={"height": 250},
        ),
    ]
    # settings = dcc.Loading(
    #     id="loading-dataset-settings",
    #     children=settings,
    # )
    return settings

def serve_trajectory_settings():
    feature_columns_choice = dcc.Dropdown(
        id='settings-feature-columns-choice-trajectory',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in FeatureColumnsType],
        searchable=True,
        clearable=True,
        placeholder="select feature columns",
        value=None,
        persistence=True,
        persistence_type="session",
    )
    # time_unit_choice = dcc.Dropdown(
    #     id='settings-time-unit-choice-trajectory',
    #     optionHeight=20,
    #     options=[ {'label': e.name, "value": e.name} for e in TimeUnit],
    #     searchable=True,
    #     clearable=True,
    #     placeholder="select time unit",
    #     # value=TimeUnit.DAY.name,
    #     value=None,
    # )

    anomaly_type_choice = dcc.Dropdown(
        id='settings-anomaly-type-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in AnomalyType],
        searchable=True,
        clearable=True,
        placeholder="select anomaly type",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    feature_extraction_choice = dcc.Dropdown(
        id='settings-feature-extraction-choice',
        optionHeight=20,
        options=[ {'label': e.name, "value": e.name} for e in FeatureExtraction],
        searchable=True,
        clearable=True,
        placeholder="select feature extraction",
        value=None,
        persistence=True,
        persistence_type="session",
    )

    settings = [
        
        dhc.Br(),
        dhc.Br(),
        dhc.Br(),
        dhc.H3(
            "Trajectory settings",
            style={
                "textAlign": "center",
            },
        ),
        dhc.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col([
                            "Feature columns (for the trajectory model):",
                            dhc.I(title="More information", id="feature-columns-trajectory-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            feature_columns_trajectory_modal,
                        ], width=3),
                        dbc.Col(feature_columns_choice, width=6),
                    ]
                ),
                dhc.Br(),
                
                # dbc.Row(
                #     [
                #         dbc.Col("Time unit (for plots):", width=3),
                #         dbc.Col(time_unit_choice, width=6),
                #     ]
                # ),
                # dhc.Br(),
                
                
                dbc.Row(
                    [
                        dbc.Col([
                            "Anomaly type:",
                            dhc.I(title="More information", id="anomaly-type-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            anomaly_type_modal,
                        ], width=3),
                        dbc.Col(anomaly_type_choice, width=6),
                    ]
                ),
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col([
                            "Feature extraction:",
                            dhc.I(title="More information", id="feature-extraction-info", className="fa fa-info-circle", style={"marginLeft": "10px"}),
                            feature_extraction_modal,
                        ], width=3),
                        dbc.Col(feature_extraction_choice, width=6),
                    ]
                ),
                
                dhc.Br(),
                dbc.Row(
                    [
                        dbc.Col(dhc.P(), width=3),
                        dbc.Col(
                            [dhc.Div(dbc.Button(
                                "Update trajectory",
                                outline=True,
                                color="dark",
                                id="button-trajectory-settings-update",
                                n_clicks=0,
                                disabled=True,
                            )),
                            dhc.Br(),
                            dhc.Br(),
                            dcc.Loading(id="loading-boxes-trajectory", children=[
                                dhc.Div(id="trajectory-settings-infobox"),
                            ], type="default")
                            ]
                            , width=6),
                    ]
                ),
                dhc.Br(),
            ],
            className="md-12",
            # style={"height": 250},
        ),
    ]

    return settings

def serve_methods():

    card1 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Data Analysis & Exploration", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        # dhc.A(dbc.Button("Go somewhere", outline=True, color="dark", id="card1-btn"), href="/methods/page-1"),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-1-btn", disabled=True,
                            ),
                            href=page1_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card2 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Reference Definition", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-2-btn", disabled=True,
                            ),
                            href=page2_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card3 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Microbiome Trajectory", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-3-btn", disabled=True,
                            ),
                            href=page3_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card4 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Bacteria Importance with Time", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-4-btn", disabled=True,
                            ),
                            href=page4_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card5 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Anomaly Detection", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-5-btn", disabled=True,
                            ),
                            href=page5_location,
                        ),
                    ]
                ),
            ],
            # style={"width": "18rem"},
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    card6 = dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(
                    src="https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/webapp/static/img/data_analysis2.jpg",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        dhc.H4("Intervention Simulation", className="card-title"),
                        # dhc.P(
                        #     "Some quick example text to build on the card title and "
                        #     "make up the bulk of the card's content.",
                        #     className="card-text",
                        # ),
                        dcc.Link(
                            dbc.Button(
                                "See more", outline=True, color="dark", id="card-6-btn", disabled=True,
                            ),
                            href=page6_location,
                        ),
                    ]
                ),
            ],
            style={"backgroundColor": "rgb(240, 240, 240)"},
            className="mb-4 box-shadow",
        ),
        className="md-4",
    )

    methods = [
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dhc.Br(),
                            dhc.Div(dhc.H3("Trajectory methods")),
                            dhc.Br(),
                        ],
                        className="md-12",
                    ),
                ),
                dbc.Row([card1, card2, card3]),
                dbc.Row([card4, card5, card6]),
            ],
            className="md-4",
        )
    ]

    return methods

def serve_layout():
    session_id = str(uuid.uuid4())

    upload = serve_upload(session_id)
    dataset_table = serve_dataset_table()
    dataset_settings = serve_dataset_settings()
    trajectory_settings = serve_trajectory_settings()
    methods = serve_methods()

    layout = dhc.Div(id="home-layout", children=[
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    [
                        dcc.Store(data=session_id, id='session-id'),
                        dhc.Br(),
                        dhc.Div(children=upload, id="dataset-upload"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Br(),
                        dhc.Div(children=dataset_table, id="dataset-table"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Br(),
                        dhc.Div(children=dataset_settings, id="dataset-settings"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Br(),
                        dhc.Div(children=trajectory_settings, id="trajectory-settings"),
                        dhc.Br(),
                        dhc.Br(),
                        dhc.Hr(),
                        dhc.Div(
                            children=methods,
                            id="dataset-methods",
                            style={
                                "verticalAlign": "middle",
                                "textAlign": "center",
                                "backgroundColor": "rgb(255, 255, 255)",
                                "position": "relative",
                                "width": "100%",
                                #'height':'100vh',
                                "bottom": "0px",
                                "left": "0px",
                                "zIndex": "1000",
                            },
                        ),
                        dhc.Br(),
                    ]
                ),
            ),
            className="md-12",
        ),
        dhc.Br(),
        dhc.Br(),
    ])

    return layout

layout = serve_layout()

# from dash import html as dhc
# from dash import dcc

# layout = dhc.Div([
#     dcc.Link('Go to Page 1', href='/page-1'),
#     dhc.Br(),
#     dcc.Link('Go to Page 2', href='/page-2'),
# ])
