from dash.dependencies import Input, Output, State
from app import app, cache
import pandas as pd
from microbiome.dataset import MicrobiomeDataset
import dash_bootstrap_components as dbc

from microbiome.enumerations import *
from microbiome.trajectory import MicrobiomeTrajectory

# def parse_contents(contents, filename, date):
#     # content_type, content_string = contents.split(',')

#     # decoded = base64.b64decode(content_string)
#     # try:
#     #     if 'csv' in filename:
#     #         # Assume that the user uploaded a CSV file
#     #         df = pd.read_csv(
#     #             io.StringIO(decoded.decode('utf-8')))
#     #     elif 'xls' in filename:
#     #         # Assume that the user uploaded an excel file
#     #         df = pd.read_excel(io.BytesIO(decoded))
#     # except Exception as e:
#     #     print(e)
#     #     return html.Div([
#     #         'There was an error processing this file.'
#     #     ])

#     return "Hello"

# @app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'),
#               State('upload-data', 'filename'),
#               State('upload-data', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children


def parse_dataset(filename):
    df = None
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(filename)
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_csv(filename, sep="\t")
            if len(df.columns) == 1:
                df = pd.read_csv(filename, sep=",")
                if len(df.columns) == 1:
                    raise Exception("Not a good file separator")
    except Exception as e:
        print(e)
        return None
    print("\nFinished parsing df parse_dataset")

    return df


# @cache.memoize(timeout=3600)
# def query_data(file_name):
#     if file_name != "":
#         dataset = MicrobiomeDataset(file_name=file_name)
#         df = dataset.df
#     else:
#         df = pd.DataFrame(
#             data={
#                 "sampleID": [""] * 10,
#                 "subjectID": [""] * 10,
#                 "age_at_collection": [""] * 10,
#                 "reference_group": [""] * 10,
#                 "group": [""] * 10,
#                 "bacteria_1": [""] * 10,
#                 "bacteria_2": [""] * 10,
#                 "bacteria_3": [""] * 10,
#                 "meta_1": [""] * 10,
#                 "meta_2": [""] * 10,
#             }
#         )

#     return df.to_json(date_format="iso", orient="split")


# def dataframe(file_name):
#     return pd.read_json(query_data(file_name), orient="split")



@app.callback(
    [
        # Table content
        Output("upload-datatable", "data"),
        Output("upload-datatable", "columns"),
        Output("upload-datatable", "tooltip_header"),
        # Update infobox or errorbox when dataset is loaded 
        Output("upload-infobox", "children"),
        Output("upload-errorbox", "children"),
        # Write dataset count stats
        Output("upload-file-name", "children"),
        Output("upload-number-of-samples", "children"),
        Output("upload-number-of-subjects", "children"),
        Output("upload-unique-groups", "children"),
        Output("upload-number-of-reference-samples", "children"),
        Output("upload-differentiation-score", "children"),
        # Update dropdown options for log-bacteria
        Output("settings-log-ratio-bacteria-choice", "options"),

        Output("settings-reference-group-choice", "value"),
        Output("settings-time-unit-choice", "value"),
        Output("settings-feature-columns-choice", "value"),
        Output("settings-normalized-choice", "value"),
        Output("settings-anomaly-type-choice", "value"),
        Output("settings-feature-extraction-choice", "value"),
        Output("settings-log-ratio-bacteria-choice", "value"),
    ],
    [
        Input("button-mouse-data", "n_clicks_timestamp"),
        Input("button-human-data", "n_clicks_timestamp"),
        Input("settings-reference-group-choice", "value"),
        Input("settings-time-unit-choice", "value"),
        Input("settings-feature-columns-choice", "value"),
        Input("settings-normalized-choice", "value"),
        Input("settings-anomaly-type-choice", "value"),
        Input("settings-feature-extraction-choice", "value"),
        Input("settings-log-ratio-bacteria-choice", "value"),
    ],
)
def update_table(
    n_click_timestamp_mouse,
    n_click_timestamp_human,
    reference_group,
    time_unit,
    feature_columns,
    normalized,
    anomaly_type,
    feature_extraction,
    log_ratio_bacteria,
):
    n_click_timestamp_mouse = n_click_timestamp_mouse or 0
    n_click_timestamp_human = n_click_timestamp_human or 0

    infobox = ""
    errorbox = ""
    file_name = ""
    number_of_samples = ""
    number_of_subjects = ""
    unique_groups = ""
    number_of_reference_samples = ""
    differentiation_score = ""

    log_ratio_bacteria_choices = []

    # df = dataframe(file_name)
    dataset = None
    df = pd.DataFrame(
        data={
            "sampleID": [""] * 10,
            "subjectID": [""] * 10,
            "age_at_collection": [""] * 10,
            "reference_group": [""] * 10,
            "group": [""] * 10,
            "bacteria_1": [""] * 10,
            "bacteria_2": [""] * 10,
            "bacteria_3": [""] * 10,
            "meta_1": [""] * 10,
            "meta_2": [""] * 10,
        }
    )

    if n_click_timestamp_mouse or n_click_timestamp_human:
        if n_click_timestamp_mouse > n_click_timestamp_human:
            file_name = "mouse_data"
        else:
            file_name = "human_data"
        # df = dataframe(file_name)
        try:
            dataset = MicrobiomeDataset(file_name=file_name)
        except Exception as e:
            errorbox = dbc.Alert(e, color="danger")
            infobox = ""

    if dataset is not None:
        

        reference_group = reference_group or ReferenceGroup.USER_DEFINED.name
        time_unit = time_unit or TimeUnit.DAY.name
        feature_columns = feature_columns or FeatureColumnsType.BACTERIA.name
        normalized = normalized or Normalization.NON_NORMALIZED.name
        anomaly_type = anomaly_type or AnomalyType.PREDICTION_INTERVAL.name
        feature_extraction = feature_extraction or FeatureExtraction.NONE.name

        # dataset.time_unit = TimeUnit[time_unit]
        dataset.time_unit = TimeUnit[time_unit]
        
        dataset.feature_columns = FeatureColumnsType[feature_columns]
        
        dataset.reference_group_choice = ReferenceGroup[reference_group]
        
        dataset.log_ratio_bacteria = log_ratio_bacteria
        
        dataset.normalized = Normalization[normalized]
        
        # anomaly_type = AnomalyType[anomaly_type]
        # feature_extraction = FeatureExtraction[feature_extraction]

        trajectory = MicrobiomeTrajectory(
            dataset,
            feature_columns=FeatureColumnsType[feature_columns],
            feature_extraction=FeatureExtraction[feature_extraction],
            time_unit=TimeUnit[time_unit],
            train_indices=None,
        )
        log_ratio_bacteria_choices = [ {'label': e, "value": e} for e in dataset.bacteria_columns] 
        df = trajectory.dataset.df.copy(deep=True)
        
        infobox = dbc.Alert("Successfully loaded data", color="success")
        number_of_samples = len(dataset.df)
        number_of_subjects = len(dataset.df.subjectID.unique())
        unique_groups = ", ".join(list(dataset.df.group.unique()))
        number_of_reference_samples = sum(dataset.df.reference_group == True)
        differentiation_score = dataset.differentiation_score

    columns = [
        {"name": i, "id": i, "deletable": True, "renamable": True} for i in df.columns
    ]
    tooltip_header = {i: i for i in df.columns}

    # TODO:
    # - trajectory separate from dataset
    # - upload dataset
    # - successfully loaded data

    return (
        df.to_dict("records"),
        columns,
        tooltip_header,
        infobox,
        errorbox,
        file_name,
        number_of_samples,
        number_of_subjects,
        unique_groups,
        number_of_reference_samples,
        differentiation_score,

        log_ratio_bacteria_choices,

        reference_group,
        time_unit,
        feature_columns,
        normalized,
        anomaly_type,
        feature_extraction,
        log_ratio_bacteria,
        
    )
