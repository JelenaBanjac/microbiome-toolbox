# from dash.dependencies import Input, Output, State
from app import app
import pandas as pd
from microbiome.dataset import MicrobiomeDataset
import dash_bootstrap_components as dbc
import dash_uploader as du
from microbiome.enumerations import *
from microbiome.trajectory import MicrobiomeTrajectory
import dash
from environment.settings import UPLOAD_FOLDER_ROOT
from dash_extensions.enrich import Output, Input, State
import pathlib
from .home_data import get_dataset, set_dataset, set_trajectory
import traceback
from dash import dcc


@du.callback(
    [
        Output("button-custom-data", "disabled"),
        Output("upload-data-file-path", "data"),
        # Output("upload-infobox", "children"),
        Output("small-upload-info-text", "hidden"),
    ],
    id="upload-data",
)
def upload_file(file_name):
    print("\n\nUploaded data...\n\n")
    print(file_name)

    return False, file_name[0], True


@app.callback(
    [
        # store dataset and trajectory in session
        Output("microbiome-dataset-location", "data"),
        Output("microbiome-trajectory-location", "data"),
        # error box
        Output("upload-infobox", "children"),
        # 3 dataset buttons
        Output("button-mouse-data", "outline"),
        Output("button-human-data", "outline"),
        Output("button-custom-data", "outline"),
        # 2 buttons for settings
        Output("button-dataset-settings-update", "disabled"),
        Output("button-trajectory-settings-update", "disabled"),
        # 6 buttons for methods
        Output("card-1-btn", "disabled"),
        Output("card-2-btn", "disabled"),
        Output("card-3-btn", "disabled"),
        Output("card-4-btn", "disabled"),
        Output("card-5-btn", "disabled"),
        Output("card-6-btn", "disabled"),
        # Write dataset count stats
        Output("upload-file-name", "children"),
        Output("upload-number-of-samples", "children"),
        Output("upload-number-of-subjects", "children"),
        Output("upload-unique-groups", "children"),
        Output("upload-number-of-reference-samples", "children"),
        Output("upload-differentiation-score", "children"),
        # Update dropdown options for dataset settings
        Output("settings-feature-columns-choice-dataset", "value"),
        Output("settings-reference-group-choice", "value"),
        Output("settings-time-unit-choice", "value"),
        Output("settings-normalized-choice", "value"),
        Output("settings-log-ratio-bacteria-choice", "value"),
        Output("settings-log-ratio-bacteria-choice", "options"),
        # Update dropdown options for trajectory settings
        Output("settings-feature-columns-choice-trajectory", "value"),
        Output("settings-anomaly-type-choice", "value"),
        Output("settings-feature-extraction-choice", "value"),
        # Disable reference group choice if references are all samples
        Output("settings-reference-group-choice", "disabled"),
        Output("card-2-btn", "disabled"),
    ],
    [
        Input("button-mouse-data", "n_clicks"),
        Input("button-human-data", "n_clicks"),
        Input("button-custom-data", "n_clicks"),
    ],
    [
        State("upload-data-file-path", "data"),
        State("session-id", "data"),
    ],
)
def dataset_buttons_click(
    button_mouse_dataset_clicks,
    button_human_dataset_clicks,
    button_custom_dataset_clicks,
    upload_data_file_path,
    session_id,
):
    ctx = dash.callback_context

    button_id = None
    file_name = ""
    dataset = None
    dataset_path = ""
    trajectory_path = ""
    infobox = ""

    number_of_samples = ""
    number_of_subjects = ""
    unique_groups = ""
    number_of_reference_samples = ""
    differentiation_score = ""

    button_mouse_data_outline = True
    button_human_data_outline = True
    button_custom_data_outline = True
    dataset_settings_disabled = True
    trajectory_settings_disabled = True
    card_1_btn_disabled = True
    card_2_btn_disabled = True
    card_3_btn_disabled = True
    card_4_btn_disabled = True
    card_5_btn_disabled = True
    card_6_btn_disabled = True

    reference_group = None
    time_unit = None
    feature_columns = None
    normalized = None
    log_ratio_bacteria = None
    feature_columns_trajectory = None
    anomaly_type = None
    feature_extraction = None
    log_ratio_bacteria_choices = []

    two_references_not_available = False

    if (
        button_mouse_dataset_clicks > 0
        or button_human_dataset_clicks > 0
        or button_custom_dataset_clicks > 0
    ) and ctx.triggered:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print("button_id", button_id)

        if button_id == "button-mouse-data":
            file_name = "mouse_data"
            button_mouse_data_outline = False
            # dataset_id = join(UPLOAD_FOLDER_ROOT, session_id, "default_mouse_dataset.xls")
        elif button_id == "button-human-data":
            file_name = "human_data"
            button_human_data_outline = False
            # dataset_id = join(UPLOAD_FOLDER_ROOT, session_id, "default_human_dataset.csv")
        elif button_id == "button-custom-data":
            file_name = upload_data_file_path
            button_custom_data_outline = False

        try:

            path_dir = pathlib.Path(UPLOAD_FOLDER_ROOT) / session_id
            path_dir.mkdir(parents=True, exist_ok=True)

            dataset = MicrobiomeDataset(file_name=file_name)
            dataset_path = str(path_dir / f"{button_id}-dataset.pickle")
            set_dataset(dataset, dataset_path)

            dataset_settings_disabled = False

        except Exception as e:
            infobox = dbc.Alert(
                children=[
                    dcc.Markdown("Microbiome dataset error: " + str(e)),
                    dcc.Markdown(traceback.format_exc()),
                    dcc.Markdown(
                        "Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an [email](msjelenabanjac@gmail.com)."
                    ),
                ],
                color="danger",
            )

    if dataset is not None:

        reference_group = ReferenceGroup.USER_DEFINED.name
        time_unit = TimeUnit.DAY.name
        feature_columns = FeatureColumnsType.BACTERIA.name
        normalized = Normalization.NON_NORMALIZED.name
        log_ratio_bacteria = None
        feature_columns_trajectory = FeatureColumnsType.BACTERIA.name
        anomaly_type = AnomalyType.PREDICTION_INTERVAL.name
        feature_extraction = FeatureExtraction.NONE.name

        dataset.time_unit = TimeUnit[time_unit]
        dataset.feature_columns = FeatureColumnsType[feature_columns]
        dataset.reference_group_choice = ReferenceGroup[reference_group]
        dataset.log_ratio_bacteria = log_ratio_bacteria
        dataset.normalized = Normalization[normalized]

        infobox = [dbc.Alert("Successfully loaded data", color="success")]
        if len(dataset.df.reference_group.unique()) != 2:
            two_references_not_available = True
            note_on_reference_group = dbc.Alert(
                "Column `reference_group` is not defined. All samples are therefore considered reference.",
                color="warning", duration=2000,
            )
            infobox.append(note_on_reference_group)

        log_ratio_bacteria_choices = [
            {"label": e, "value": e} for e in dataset.bacteria_columns
        ]

        number_of_samples = len(dataset.df)
        number_of_subjects = len(dataset.df.subjectID.unique())
        unique_groups = ", ".join(list(dataset.df.group.unique()))
        number_of_reference_samples = sum(dataset.df.reference_group == True)
        differentiation_score = dataset.differentiation_score

        try:
            trajectory = MicrobiomeTrajectory(
                dataset,
                feature_columns=FeatureColumnsType[feature_columns_trajectory],
                feature_extraction=FeatureExtraction[feature_extraction],
                time_unit=TimeUnit[time_unit],
                anomaly_type=AnomalyType[anomaly_type],
                train_indices=None,
            )
            trajectory_path = str(path_dir / f"{button_id}-trajectory.pickle")
            set_trajectory(trajectory, trajectory_path)

            trajectory_settings_disabled = False

            card_1_btn_disabled = False
            card_2_btn_disabled = False
            card_3_btn_disabled = False
            card_4_btn_disabled = False
            card_5_btn_disabled = False
            card_6_btn_disabled = False
        except Exception as e:
            infobox = dbc.Alert(
                children=[
                    dcc.Markdown("Microbiome trajectory error: " + str(e)),
                    dcc.Markdown(traceback.format_exc()),
                    dcc.Markdown(
                        "Open an [issue on GitHub](https://github.com/JelenaBanjac/microbiome-toolbox/issues) or send an [email](msjelenabanjac@gmail.com)."
                    ),
                ],
                color="danger",
            )

    print("file_name", file_name)
    print("button_id", button_id)
    print("session_id", session_id)
    print("dataset_path", dataset_path)

    return (
        dataset_path,
        trajectory_path,
        infobox,
        # 3 buttons
        button_mouse_data_outline,
        button_human_data_outline,
        button_custom_data_outline,
        # 2 buttons
        dataset_settings_disabled,
        trajectory_settings_disabled,
        # 6 buttons
        card_1_btn_disabled,
        card_2_btn_disabled,
        card_3_btn_disabled,
        card_4_btn_disabled,
        card_5_btn_disabled,
        card_6_btn_disabled,
        # Write dataset count stats
        file_name,
        number_of_samples,
        number_of_subjects,
        unique_groups,
        number_of_reference_samples,
        differentiation_score,
        # Update dropdown options for dataset settings
        feature_columns,
        reference_group,
        time_unit,
        normalized,
        log_ratio_bacteria,
        log_ratio_bacteria_choices,
        # Update dropdown options for trajectory settings
        feature_columns_trajectory,
        anomaly_type,
        feature_extraction,
        # Disable
        two_references_not_available,
        two_references_not_available,
    )


@app.callback(
    [
        Output("microbiome-dataset-location", "data"),
        Output("dataset-settings-infobox", "children"),
        # Write dataset count stats
        Output("upload-number-of-reference-samples", "children"),
        Output("upload-differentiation-score", "children"),
    ],
    [
        Input("button-dataset-settings-update", "n_clicks"),
    ],
    [
        State("microbiome-dataset-location", "data"),
        # Get dropdown options for log-bacteria
        State("settings-feature-columns-choice-dataset", "value"),
        State("settings-reference-group-choice", "value"),
        State("settings-time-unit-choice", "value"),
        State("settings-normalized-choice", "value"),
        State("settings-log-ratio-bacteria-choice", "value"),
    ],
)
def update_dataset(
    button_dataset_settings_update_clicks,
    dataset_path,
    feature_columns,
    reference_group,
    time_unit,
    normalized,
    log_ratio_bacteria,
):
    infobox = ""
    number_of_reference_samples = ""
    differentiation_score = ""

    if dataset_path is not None:
        dataset = get_dataset(dataset_path)

    if button_dataset_settings_update_clicks > 0:
        feature_columns = feature_columns or FeatureColumnsType.BACTERIA.name
        reference_group = reference_group or ReferenceGroup.USER_DEFINED.name
        time_unit = time_unit or TimeUnit.DAY.name
        normalized = normalized or Normalization.NON_NORMALIZED.name
        log_ratio_bacteria = log_ratio_bacteria or None

        dataset.time_unit = TimeUnit[time_unit]
        dataset.feature_columns = FeatureColumnsType[feature_columns]
        dataset.reference_group_choice = ReferenceGroup[reference_group]
        dataset.log_ratio_bacteria = log_ratio_bacteria
        dataset.normalized = Normalization[normalized]

        infobox = dbc.Alert("Dataset updated", color="success", duration=2000)
        number_of_reference_samples = sum(dataset.df.reference_group == True)
        differentiation_score = dataset.differentiation_score

        set_dataset(dataset, dataset_path)

    return dataset_path, infobox, number_of_reference_samples, differentiation_score


@app.callback(
    [
        Output("microbiome-trajectory-location", "data"),
        Output("trajectory-settings-infobox", "children"),
    ],
    [
        Input("button-trajectory-settings-update", "n_clicks"),
        Input("dataset-settings-infobox", "children"),
    ],
    [
        State("microbiome-trajectory-location", "data"),
        State("microbiome-dataset-location", "data"),
        # Get dropdown options for trajectory settings
        State("settings-feature-columns-choice-trajectory", "value"),
        State("settings-anomaly-type-choice", "value"),
        State("settings-feature-extraction-choice", "value"),
    ],
)
def update_trajectory(
    button_trajectory_settings_update_clicks,
    infobox_dataset,
    trajectory_path,
    dataset_path,
    feature_columns_trajectory,
    anomaly_type,
    feature_extraction,
):
    infobox = ""

    if trajectory_path is not None:
        if dataset_path:
            dataset = get_dataset(dataset_path)
            if button_trajectory_settings_update_clicks > 0:
                feature_columns_trajectory = (
                    feature_columns_trajectory or FeatureColumnsType.BACTERIA.name
                )
                anomaly_type = anomaly_type or ReferenceGroup.USER_DEFINED.name
                feature_extraction = feature_extraction or TimeUnit.DAY.name

                trajectory = MicrobiomeTrajectory(
                    dataset,
                    feature_columns=FeatureColumnsType[feature_columns_trajectory],
                    feature_extraction=FeatureExtraction[feature_extraction],
                    time_unit=dataset.time_unit,
                    anomaly_type=AnomalyType[anomaly_type],
                    train_indices=None,
                )

                infobox = dbc.Alert(
                    "Trajectory updated", color="success", duration=2000
                )

                set_trajectory(trajectory, trajectory_path)

    return trajectory_path, infobox


@app.callback(
    [  # Table content
        Output("upload-datatable", "data"),
        Output("upload-datatable", "columns"),
        Output("upload-datatable", "tooltip_header"),
    ],
    [
        Input("microbiome-dataset-location", "data"),
    ],
)
def update_table(dataset_path):
    print("dataset_path", dataset_path)
    if dataset_path:
        dataset = get_dataset(dataset_path)
        # trajectory = get_trajectory(dataset_id)
        df = dataset.df
    else:
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
    columns = [
        {"name": i, "id": i, "deletable": True, "renamable": True} for i in df.columns
    ]
    tooltip_header = {i: i for i in df.columns}

    return (
        df.to_dict("records"),
        columns,
        tooltip_header,
    )
