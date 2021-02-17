from pickle import FALSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import seaborn as sns
from catboost import Pool
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
import scipy.stats as stats
import scipy as sp
import pathlib
import shap
from microbiome.variables import *
from microbiome.helpers import df2vectors
from microbiome.statistical_analysis import *
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations 
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV


def topK_important_features(k, estimator, df, feature_cols, n_splits, estimator_for_fit, save=False, file_name=None):
    """Get top k important features for the estimator

    k: int
        Number of important features we want to have in the model.
    estimator:
        Model for the trajectory.
    df: pd.DataFrame
        Dataset on which we want to find the top important features.
    feature_cols: list
        List of bacteria for the estimator.
    n_splits: int
        Number of split for the group cross validation.
    estimator_for_fit:
        Object to use to fit the data (same as the estimator)
    save: bool
        If True, the search for the important features will be performed and sabed in the end. Otherwise, it will read the important features from some 
        of the previous runs from a file.
    file_name: str
        Name of the file to save data in or read from.

    Returns
    -------
    important_features: list
        List of important features.
    """
    if file_name is None:
        raise Exception("You should specify a file you wish to save the list of important features")

    # f'OUTPUT_FILES/important_features_{DATASET_LEVEL}_{SAVE_IDENTIFIER}.txt'
    if save:
        X, y = df2vectors(df, feature_cols)
        
        shap_values = estimator.get_feature_importance(Pool(X, y), type="ShapValues")
        feature_importance = pd.DataFrame(list(zip(feature_cols, np.abs(shap_values).mean(0))), columns=['bacteria_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        important_features = list(feature_importance.bacteria_name.values)

        features_num = []
        accuracy_means = []
        accuracy_stds = []
        maes = []
        r2s = []

        i = 1
        while i <  len(important_features[:k])+1:
            done = 0
            X, y = df2vectors(df, important_features[:i])
            while done < 3:
                try:
                    scores = cross_val_score(estimator_for_fit, X, y, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
                    y_pred = cross_val_predict(estimator_for_fit, X, y, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
                    accuracy_means.append(scores.mean())
                    accuracy_stds.append(scores.std())
                    maes.append(np.mean(abs(y_pred - y)))
                    r2s.append(r2_score(y, y_pred))
                    features_num.append(i)
                    i+=2
                    done = 4
                except Exception as e:
                    print(e)
                    done += 1


        fig = make_subplots(rows=1, cols=3)
        
        i = np.argmin(maes)
        fig.add_trace(go.Scatter(
                x=features_num,
                y=maes,
                name="MAE",
                hovertemplate =None),
                row=1, col=1)
        fig.add_trace(go.Scatter(
                x=[features_num[i]],
                y=[maes[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=1)
        fig.update_xaxes(title="Number of important features", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1) 
        fig.update_yaxes(title="MAEs", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1)
                
        j = np.argmax(accuracy_means)
        fig.add_trace(go.Scatter(
                x=features_num,
                y=accuracy_means,
                name="Accuracy",
                hovertemplate = None,
                error_y=dict(
                        type='data', 
                        array=accuracy_stds,
                        visible=True)),
                row=1, col=2)
        fig.add_trace(go.Scatter(
                x=[features_num[i]],
                y=[accuracy_means[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=2)
        fig.update_xaxes(title="Number of important features", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2) 
        fig.update_yaxes(title="Accuracy", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2)
        
        fig.add_trace(go.Scatter(
                x=features_num,
                y=r2s,
                name="R-square",
                hovertemplate = None),
                row=1, col=3)
        fig.add_trace(go.Scatter(
                x=[features_num[i]],
                y=[r2s[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=True,
                hovertemplate =None),
                row=1, col=3)
        fig.update_xaxes(title="Number of important features", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3)  
        fig.update_yaxes(title="R-squared", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3)  

        fig.update_layout(height=500, width=1200, 
                        #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, b=0, pad=0),
                        title_text="Top Important Features",
                        hovermode="x")
        fig.show()
        
        # excract only top important features
        important_features = important_features[:features_num[i]]

        with open(file_name, 'w') as f:
            f.writelines(f"{feature}\n" for feature in important_features)

    else:
        important_features = []

        with open(file_name, 'r') as f:
            filecontents = f.readlines()

            for line in filecontents:
                important_features.append(line[:-1])
                
    return important_features

def remove_nzv(save, df, feature_cols, n_splits, estimator_for_fit, nzv_thresholds=None):
    feature_cols = np.array(feature_cols)
    """Remove features with near-zero-variance"""
    if save:
        accuracy_means = []
        accuracy_stds = []
        maes = []
        r2s = []
        features_num = []
        
        for threshold in nzv_thresholds:
            # filter near zero variance features
            constant_filter = VarianceThreshold(threshold=threshold)
            constant_filter.fit(df[feature_cols])
            idx = np.where(constant_filter.get_support())[0]
            constant_columns = [column for column in feature_cols if column not in feature_cols[idx]]
            features_num.append(len(feature_cols)-len([x for x in constant_columns if x.startswith('k__')] ))
            feature_cols_new = list(set(feature_cols) - set(constant_columns))

            X, y = df2vectors(df, feature_cols_new)

            scores = cross_val_score(estimator_for_fit, X, y, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
            accuracy_means.append(scores.mean())
            accuracy_stds.append(scores.std())

            y_pred = cross_val_predict(estimator_for_fit, X, y, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
            maes.append(np.mean(abs(y_pred - y)))
            r2s.append(r2_score(y, y_pred))

        fig = make_subplots(rows=1, cols=3)
        
        i = np.argmin(maes)
        fig.add_trace(go.Scatter(
                x=nzv_thresholds,
                y=maes,
                name="MAE",
                hovertemplate =None),
                row=1, col=1)
        fig.add_trace(go.Scatter(
                x=[nzv_thresholds[i]],
                y=[maes[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=1)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1) 
        fig.update_yaxes(title="MAEs", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1) 
                
        j = np.argmax(accuracy_means)
        fig.add_trace(go.Scatter(
                x=nzv_thresholds,
                y=accuracy_means,
                name="Accuracy",
                hovertemplate = None,
                error_y=dict(
                        type='data', 
                        array=accuracy_stds,
                        visible=True)),
                row=1, col=2)
        fig.add_trace(go.Scatter(
                x=[nzv_thresholds[i]],
                y=[accuracy_means[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=2)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2) 
        fig.update_yaxes(title="Accuracy", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2)  
        
        fig.add_trace(go.Scatter(
                x=nzv_thresholds,
                y=r2s,
                name="R-square",
                hovertemplate = None),
                row=1, col=3)
        fig.add_trace(go.Scatter(
                x=[nzv_thresholds[i]],
                y=[r2s[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=True,
                hovertemplate =None),
                row=1, col=3)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3) 
        fig.update_yaxes(title="R-squared", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3) 

        fig.update_layout(height=500, width=1200, 
                        #paper_bgcolor="white",
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, b=0, pad=0),
                        title_text="Near Zero Variance",
                        hovermode="x")
        fig.show()

        nzv_threshold = nzv_thresholds[j]
    else:
        # 26-Nov-2020
        nzv_threshold = 0.01
        
    constant_filter = VarianceThreshold(threshold=nzv_threshold)
    constant_filter.fit(df[feature_cols])
    feature_cols = np.array(feature_cols)
    constant_columns = [column for column in feature_cols if column not in feature_cols[np.where(constant_filter.get_support())[0]]]
    feature_cols_new = list(set(feature_cols) - set(constant_columns))
    print(f"Number of features left after removing features with variance {nzv_threshold} or smaller: {len(feature_cols_new)}/{len(feature_cols)}")

    return feature_cols_new

def remove_correlated(save, df, feature_cols, n_splits, estimator_for_fit, correlation_thresholds=None):
    """"Remove correlated values"""
    # the higher the number the less it is removing
    if save:
        #correlation_thresholds = [0.0, 0.1, 0.3, 0.8]
        accuracy_means = []
        accuracy_stds = []
        maes = []
        r2s = []
        features_num = []

        for threshold in correlation_thresholds:
            correlated_features = set()
            correlation_matrix = df[feature_cols].corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > threshold:
                        colname = correlation_matrix.columns[i]
                        correlated_features.add(colname)
                        
            feature_cols_ncorr = list(set(feature_cols)-correlated_features)
            features_num.append(len(feature_cols_ncorr))

            X_healthy, y_healthy = df2vectors(df, feature_cols_ncorr)
            
            scores = cross_val_score(estimator_for_fit, X_healthy, y_healthy, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
            accuracy_means.append(scores.mean())
            accuracy_stds.append(scores.std())

            y_healthy_pred = cross_val_predict(estimator_for_fit, X_healthy, y_healthy, groups=df.subjectID.values, cv=GroupKFold(n_splits), verbose=0)
            maes.append(np.mean(abs(y_healthy_pred - y_healthy)))
            r2s.append(r2_score(y_healthy, y_healthy_pred))

        fig = make_subplots(rows=1, cols=3)
        
        i = np.argmin(maes)
        fig.add_trace(go.Scatter(
                x=correlation_thresholds,
                y=maes,
                name="MAE",
                hovertemplate =None),
                row=1, col=1)
        fig.add_trace(go.Scatter(
                x=[correlation_thresholds[i]],
                y=[maes[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=1)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1) 
        fig.update_yaxes(title="MAEs", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=1)  
                
        j = np.argmax(accuracy_means)
        fig.add_trace(go.Scatter(
                x=correlation_thresholds,
                y=accuracy_means,
                name="Accuracy",
                hovertemplate = None,
                error_y=dict(
                        type='data',
                        array=accuracy_stds,
                        visible=True)),
                row=1, col=2)
        fig.add_trace(go.Scatter(
                x=[correlation_thresholds[i]],
                y=[accuracy_means[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=False,
                hovertemplate =None),
                row=1, col=2)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2) 
        fig.update_yaxes(title="Accuracy", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=2) 
        
        fig.add_trace(go.Scatter(
                x=correlation_thresholds,
                y=r2s,
                name="R-square",
                hovertemplate = None),
                row=1, col=3)
        fig.add_trace(go.Scatter(
                x=[correlation_thresholds[i]],
                y=[r2s[i]],
                mode="markers",
                marker_symbol="star",
                marker_size=15,
                marker_color="green",
                name="optimal MAE",
                showlegend=True,
                hovertemplate =None),
                row=1, col=3)
        fig.update_xaxes(title="NZV thresholds", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3) 
        fig.update_yaxes(title="R-squared", showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', row=1, col=3)  

        fig.update_layout(height=500, width=1200, 
                        #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, b=0, pad=0),
                        title_text="Correlation",
                        hovermode="x")
        fig.show()

        correlation_threshold = correlation_thresholds[j]
    else:
        correlation_threshold = 1.0

    correlated_features = set()
    correlation_matrix = df[feature_cols].corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    feature_cols_new = list(set(feature_cols)-correlated_features)
    
    print(f"Number of features left after removing features with correlation {correlation_threshold}: {len(feature_cols_new)}/{len(feature_cols)}")

    return feature_cols_new

def train(df, feature_cols, Regressor, parameters, param_grid, n_splits, file_name=None):
    """f"{PIPELINE_DIRECTORY}/model_NoTopImportant"""
    train1 = df[(df.healthy_reference==True)&(df.dataset_type=="Train")]

    X_train1, y_train1 = df2vectors(train1, feature_cols)

    rfr = Regressor(**parameters)
    gkf = list(GroupKFold(n_splits=n_splits).split(X_train1, y_train1, groups=train1["subjectID"].values))

    search = GridSearchCV(rfr, param_grid, cv=gkf)
    search.fit(X_train1, y_train1)

    estimator = search.best_estimator_
    if file_name:
        estimator.save_model(file_name)

    return estimator

def get_pvalue_regliner(df, group):
    _df = df.copy(deep=False)

    group_values = _df[group].unique()

    assert len(group_values) == 2, "the dataframe in statistical analysis needs to have only 2 unique groups to compare"

    df_stats = pd.DataFrame(data={"Input": list(_df.y.values),
                                  "Output": list(_df.y_pred.values),
                                  "Condition": list(_df[group].values)})

    return regliner(df_stats, {group_values[0]: 0, group_values[1]: 1})

def get_pvalue_permuspliner(df, group, degree=2):
    _df = df.copy(deep=False)

    group_values = _df[group].unique()

    assert len(group_values) == 2, "the dataframe in statistical analysis needs to have only 2 unique groups to compare"

    df_stats = pd.DataFrame(data={"Input": list(_df.y.values),
                                  "Output": list(_df.y_pred.values),
                                  "Condition":list(_df[group].values),
                                  "sampleID": list(_df["sampleID"].values)})

    result = permuspliner(df_stats, xvar="Input", yvar="Output", category="Condition", degree = degree, cases="sampleID", groups = group_values, perms = 500, test_direction = 'more', ints = 1000, quiet = True)

    return result["pval"]

def plot_trajectory(estimator, df, feature_cols, df_other=None, group=None, linear_difference=None, nonlinear_difference=None, 
                    plateau_area_start=None, limit_age=1200, start_age=0, 
                    time_unit_size=1, time_unit_name="days", img_file_name=None, 
                    degree=2, longitudinal_mode=None, longitudinal_showlegend=True,
                    patent=False, layout_settings=None, website=False, highlight_outliers=None, df_new=None,
                    plot_CI=False, plot_PI=True):
    """Trajectory line with performance stats and many other settings

    estimator: sklearn models, CatBoostRegressor, etc.
        Model for the trajectory line.
    df: pd.DataFrame
        Reference dataset, validation data.
    feature_cols: list
        Feature columns needed for the estimator of the trajectory.
    df_other: pd.DataFrame
        Other dataset (not the reference)
    degree: int
        The degree for the polyfit line (trajectory).
    traj_color: str
        Color for the trajectory line. Usually green, but if it is a plot for IP, we need gray.
    group: bool
        To plot the colors based on the group.
    linear_difference: bool
        If True, plot the linear lines for each of the countries and calculate the significant difference between
        these linear lines.
    nonlinear_difference: bool
        If True, plot the spline lines for each of the countries and calculate the significant difference between
        these splines.
    plateau_area_start: int
        The day when the plateau starts (shade that area gray then). Usuallyafter 2-3 year, i.e. 720+ days.
    limit_age: int
        Days when to end the trajectory (used to cut the first few days and last days for infants)
    start_age: int
        Days when to start the trajectory (used to cut the first few days and last days for infants)
    time_unit_name: str
        Name of the time unit (e.g. month, year, etc.)
    time_unit_size: int
        Number of days in a new time definition (e.g. if we want to deal with months, then time_unit_in_days=30, for the year, time_unit_in_days=365)
    img_file_name: str
        Name of the file where to save the plot of trajectory.
    nboot: int
        Number of bootstrap samples from the data.

    Returns
    -------
    fig: plotly object
        To continue furhter plots in case needed.
    outliers: list
        List of outliers.
    mae: float
        The MAE error.
    r2: float
        The R^2 metric.
    pi_median: float
        Prediction interval median value across the ages.
    """
    df = df.sort_values(by="age_at_collection")
    
    fig = go.Figure()

    limit_age_max = int(max(df["age_at_collection"]))+1

    layout_settings_default = dict(
        height=900, 
        width=1000,
        barmode='stack', 
        uniformtext=dict(mode="hide", minsize=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',  
        margin=dict(l=0, r=0, b=0, pad=0),
        title_text="Microbiome Trajectory"
    )

    if patent:
        traj_color = "0,0,0"
        colors = px.colors.sequential.Greys
        outlier_color = '0,0,0'
        outlier_size = 15
        marker_outlier=dict(size=25, color=f'rgba({outlier_color},0.95)', symbol="star-open", line_width=4)
        layout_settings_default["height"]=900
        layout_settings_default["width"]=1100
        layout_settings_default["font"] = dict(
                #family="Courier New, monospace",
                size=20,
                #color="RebeccaPurple"
            )
    else:
        traj_color = "26,150,65"
        colors = px.colors.qualitative.Plotly
        outlier_color = '255,0,0'
        outlier_size = 15  
        marker_outlier=dict(size=20, color=f'rgba({outlier_color},0.95)', symbol="star-open", line_width=4)  
        
    if layout_settings is None:
        layout_settings = {}
    layout_settings_final = {**layout_settings_default, **layout_settings}
    
    if plateau_area_start is not None:
        # shaded area where plateau is expected
        if plateau_area_start/time_unit_size < limit_age/time_unit_size:
            _x = np.linspace(plateau_area_start, limit_age_max, 10)/time_unit_size
            fig.add_trace(go.Scatter(
                x=list(_x)+list(_x[::-1]),
                y=list(np.zeros(10))+list(np.ones(10)*limit_age/time_unit_size+1),
                fill='toself',
                fillcolor='rgba(220,220,220,0.5)',
                line_color='rgba(220,220,220,0.5)',
                showlegend=True,
                name=f"sample at time > {plateau_area_start} days",
            ))
    
    if group is not None:
        longitudinal_showlegend = False

    fig, ret_val, mae, r2, pi_median, _, _, _ = plot_1_trajectory(fig, estimator, df, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color=traj_color, traj_label="reference", 
                                                                            plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, longitudinal_mode=longitudinal_mode, 
                                                                            longitudinal_showlegend=longitudinal_showlegend, highlight_outliers=highlight_outliers, marker_outlier=marker_outlier, df_new=df_new,
                                                                            plot_CI=plot_CI, plot_PI=plot_PI)
    
    X, y = df2vectors(df, feature_cols)
    y_pred = estimator.predict(X)
    
    y = np.array(y)/time_unit_size
    y_pred = np.array(y_pred)/time_unit_size
    
    df["y"] = y
    df["y_pred"] = y_pred
    
    equation = lambda a, b: np.polyval(a, b) 
        
    # Data
    if group is not None:
        colors_rgb = [tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for h in colors]
        
        # linear difference between the groups
        if linear_difference:
            
            for i, group_trace in enumerate(df[group].unique()):
                idx = np.where(df[group]==group_trace)[0]
                _df = df[df[group]==group_trace]
                xdata = np.linspace(0, limit_age_max//time_unit_size+1, limit_age_max//time_unit_size+1)

                # lines
                _p, _cov = np.polyfit(y[idx], y_pred[idx], 1, cov=True)    
                fig.add_trace(go.Scatter(
                    x=xdata,
                    y=equation(_p, xdata),
                    mode="lines",
                    line = dict(width=3, dash='dash', color=colors[i]),
                    marker=dict(size=10, color=colors[i]),
                    showlegend=True,
                    legendgroup=group_trace,
                    name=f"{group}={group_trace}",
                    text=list(_df["sampleID"].values), 
                    hovertemplate = f'<b>Group ({group}): {group_trace}</b><br>',
                    hoveron="points"
                ))
                # points    
                fig.add_trace(go.Scatter(
                    x=y[idx],
                    y= y_pred[idx],
                    mode="markers",
                    line = dict(width=3, dash='dash', color=colors[i]),
                    marker=dict(size=10, color=colors[i]),
                    showlegend=True,
                    legendgroup=group_trace,
                    name=f"sample with {group}={group_trace}",
                    text=list(_df["sampleID"].values), 
                    hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                    f'<b>Group ({group}): {group_trace}</b><br>'+
                                    '<b>SampleID</b>: %{text}<br>'+
                                    '<b>Age</b>: %{x:.2f}'+
                                    '<br><b>MMI</b>: %{y}<br>',
                    hoveron="points"
                ))
            
            group_vals = df[group].unique()
            comb = combinations(group_vals, 2)
            ret_val += "<b>Linear p-value (k, n)</b>:"
            for c in list(comb):
                _df = df[(df[group].isin(c))&(df.age_at_collection<limit_age)]
                pval_k, pval_n = get_pvalue_regliner(_df, group)
                ret_val += f"<br>{group} {c[0]} vs. {c[1]}: ({pval_k:.3f}, {pval_n:.3f})"
        
        # non-linear difference between the groups (splines)
        elif nonlinear_difference:
            for i, group_trace in enumerate(df[group].unique()):
                idx = np.where(df[group]==group_trace)[0]
                _df = df[df[group]==group_trace]
                
                xdata = y[idx]
                ydata = y_pred[idx]
                _p, _cov = np.polyfit(xdata, ydata, degree, cov=True)    
                _y_model = equation(_p, ydata)                             # model using the fit parameters; NOTE: parameters here are coefficients

                # Statistics
                n = ydata.size                                             # number of observations
                m = _p.size                                                # number of parameters
                dof = n - m                                                # degrees of freedom
                t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands

                # Estimates of Error in Data/Model
                resid = ydata - _y_model                           
                chi2 = np.sum((resid / _y_model)**2)                       # chi-squared; estimates error in data
                chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
                s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

                x2 = np.linspace(0, limit_age_max/time_unit_size, limit_age_max+1) #np.linspace(np.min(x), np.max(x), 100)
                y2 = equation(_p, x2)
                pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(y))**2 / np.sum((y - np.mean(y))**2))   
                
                # mean prediction
                fig.add_trace(go.Scatter(
                    x=x2, y=y2,
                    line_color=colors[i],
                    name=f"trajectory for {group}={group_trace}",
                    legendgroup=group_trace,
                ))
                # prediction interval
                fig.add_trace(go.Scatter(
                    x=list(x2)+list(x2[::-1]),
                    y=list(y2-pi)+list(y2+pi)[::-1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(colors_rgb[i])+[0.15])}',
                    line_color=f'rgba{tuple(list(colors_rgb[i])+[0.25])}',
                    legendgroup=group_trace, 
                    showlegend=False,
                    name="95% Prediction Interval"
                ))
                # points    
                fig.add_trace(go.Scatter(
                    x=y[idx],
                    y= y_pred[idx],
                    mode="markers",
                    line = dict(width=3, dash='dash', color=colors[i]),
                    marker=dict(size=10, color=colors[i]),
                    showlegend=True,
                    legendgroup=group_trace,
                    name=f"sample with {group}={group_trace}",
                    text=list(_df["sampleID"].values), 
                    hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                    f'<b>Group ({group}): {group_trace}</b><br>'+
                                    '<b>SampleID</b>: %{text}<br>'+
                                    '<b>Age</b>: %{x:.2f}'+
                                    '<br><b>MMI</b>: %{y}<br>',
                    hoveron="points"
                )) 
            
            group_vals = df[group].unique()
            comb = combinations(group_vals, 2)
            ret_val += "<b>Nonlinear p-value</b>:"
            for c in list(comb):
                _df = df[(df[group].isin(c))]  #&(df.age_at_collection<limit_age)&(df.age_at_collection>start_age)
                error_cnt = 3
                while error_cnt > 0:
                    try:
                        pval = get_pvalue_permuspliner(_df, group, degree=degree)
                        error_cnt = -1
                    except:
                        error_cnt -= 1
                if error_cnt == -1:
                    ret_val += f"<br>{group} {c[0]} vs. {c[1]}: {pval:.3f}"
                else:
                    ret_val += f"<br>{group} {c[0]} vs. {c[1]}: not available"

                    
        else:
            # longitudinal, but color based on the group
            for i, group_trace in enumerate(df[group].unique()):
                idx = np.where(df[group]==group_trace)[0]
                _df = df[df[group]==group_trace]
                color = colors[i]
                for j, trace in enumerate(_df["subjectID"].unique()):
                    idx2 = np.where(_df["subjectID"]==trace)[0]
                    fig.add_trace(go.Scatter(
                        x=y[idx][idx2],
                        y=y_pred[idx][idx2],
                        mode="lines+markers",
                        line = dict(width=3, dash='dash', color=color),
                        marker=dict(size=10, color=color),
                        showlegend=True if j == 0 else False,
                        legendgroup=group_trace,
                        name=group_trace,
                        text=list(_df["sampleID"].values[idx2]), 
                        hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                        f'<b>Group ({group}): {group_trace}</b><br>'+
                                        '<b>SampleID</b>: %{text}<br>'+
                                        '<b>Age</b>: %{x:.2f}'+
                                        '<br><b>MMI</b>: %{y}<br>',
                        hoveron="points"
                    ))
      
    ###########################################################################
    X_other, y_other, y_other_pred = None, None, None
    if df_other is not None:
        
        df_other = df_other.sort_values(by="age_at_collection")

        X_other, y_other = df2vectors(df_other, feature_cols)
        y_other_pred = estimator.predict(X_other)
        
        y_other = np.array(y_other)/time_unit_size
        y_other_pred = np.array(y_other_pred)/time_unit_size
        
        for trace in df_other["subjectID"].unique():
            idx = np.where(df_other["subjectID"].values==trace)[0]

            fig.add_trace(go.Scatter(
                x=y_other[idx],
                y=y_other_pred[idx],
                mode=longitudinal_mode,
                line=dict(width=3, dash='dash'),
                marker=dict(size=outlier_size, color=f'rgba({outlier_color},0.95)'),
                showlegend=longitudinal_showlegend,
                name=trace,
                text=list(df_other["sampleID"].values[idx]), 
                hovertemplate = '<b>Other sample</b><br><br>'+
                                '<b>SampleID</b>: %{text}<br>'+
                                '<b>Age</b>: %{x:.2f}<br>'+
                                '<b>MMI</b>: %{y}<br>',
                hoveron="points"
            ))


    fig.update_xaxes(title=f"Age [{time_unit_name}]", range=(start_age//time_unit_size, limit_age//time_unit_size), 
                    tick0=start_age//time_unit_size, dtick=2, 
                    showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
    fig.update_yaxes(title=f"Microbiome Maturation Index [{time_unit_name}]", range=(start_age//time_unit_size, limit_age//time_unit_size), 
                    tick0=start_age//time_unit_size, dtick=2, 
                    showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
    
    fig.update_layout(**layout_settings_final)
    
    if not patent:
        fig.update_layout(go.Layout(
            annotations=[
                go.layout.Annotation(
                    text=ret_val,
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=.99,
                    y=.01,
                    bordercolor='black',
                    bgcolor='white',
                    borderwidth=0.5,
                    borderpad=8
                )
            ]
        ))
    else:
        print(ret_val)
    
    if img_file_name:
        pathlib.Path('/'.join(img_file_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    
        if "html" in img_file_name.lower(): 
            fig.write_html(img_file_name)
        elif any(extension in img_file_name for extension in ["png", "jpeg", "webp", "svg", "pdf", "eps" ]):
            fig.write_image(img_file_name)
        else:
            raise Exception(f"Extension {img_file_name.split('.')[-1]} is not implemented")
    

    
    if not website:
        fig.show()
    
    return fig, mae, r2, pi_median




def plot_1_trajectory(fig, estimator, df, bacteria_names, limit_age, time_unit_size, time_unit_name, traj_label, plateau_area_start, traj_color, limit_age_max, df_new=None, degree=2, longitudinal_mode=None, longitudinal_showlegend=True, fillcolor_alpha=0.3, highlight_outliers=None, marker_outlier=None, plot_CI=False, plot_PI=True): 
    """
    longitudinal_mode: str
        How a longitudinal data is plotted: markers+lines, lines, markers, etc.
    Reference:
    - https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    """
    if df_new is None:
        df_new = df.copy()
    
    df = df.sort_values(by="age_at_collection")
    X, y = df2vectors(df, bacteria_names)
    y_pred = estimator.predict(X)
    y = np.array(y)/time_unit_size
    y_pred = np.array(y_pred)/time_unit_size
    
    df_new = df_new.sort_values(by="age_at_collection")
    X_new, y_new = df2vectors(df_new, bacteria_names)
    y_pred_new = estimator.predict(X_new)
    y_new = np.array(y_new)/time_unit_size
    y_pred_new = np.array(y_pred_new)/time_unit_size

    mae   = round(np.mean(abs(y_pred - y)), 2)
    r2    = r2_score(y, y_pred)
    coeff = stats.pearsonr(y_pred, y)
    
    # performace calculated until limit_age
    idx       = np.where(y < limit_age/time_unit_size)[0]
    mae_idx   = round(np.mean(abs(y_pred[idx] - y[idx])), 2)
    r2_idx    = r2_score(y[idx], y_pred[idx])
    coeff_idx = stats.pearsonr(y_pred[idx], y[idx])
    ret_val = "<b>Performance Information</b><br>"
    ret_val += f'MAE: {mae_idx}<br>'
    ret_val += f'R^2: {r2_idx:.3f}<br>'
    ret_val += f"Pearson: {coeff_idx[0]:.3f}, 2-tailed p-value: {coeff_idx[1]:.2e}<br>"
    
    if plateau_area_start is not None:
        idx       = np.where(y < plateau_area_start/time_unit_size)[0]
        mae_idx   = round(np.mean(abs(y_pred[idx] - y[idx])), 2)
        r2_idx    = r2_score(y[idx], y_pred[idx])
        coeff_idx = stats.pearsonr(y_pred[idx], y[idx])
    
        ret_val += f"<b>Performance Information < {plateau_area_start} days</b><br>"
        ret_val += f'MAE: {mae_idx}<br>'
        ret_val += f'R^2: {r2_idx:.3f}<br>'
        ret_val += f"Pearson: {coeff_idx[0]:.3f}, 2-tailed p-value: {coeff_idx[1]:.2e}<br>"

        
        
    # Plot data
    equation = lambda a, b: np.polyval(a, b) 

    # Stats
    p, cov = np.polyfit(y, y_pred, degree, cov=True)           # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, y)                                   # model using the fit parameters; NOTE: parameters here are coefficients

    # Statistics
    n = y_pred.size                                            # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y_pred - y_model                           
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

    # Confidence Interval (select one)
    if plot_CI:
        nboot = 500
        bootindex = sp.random.randint
        
        for b in range(nboot):
            resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
            # Make coeffs of for polys
            pc = np.polyfit(y, y_pred + resamp_resid, degree)                   
            # Plot bootstrap cluster
            idx = np.argsort(y)
            _x = y[idx]
            _y = np.polyval(pc, _x)
            
            if b < nboot-1:
                fig.add_trace(go.Scatter(
                    x=_x,
                    y=_y,
                    mode="lines",
                    line = dict(width=5),
                    fillcolor=f'rgba({traj_color},{3.0 / float(nboot):.2f})',
                    line_color=f'rgba({traj_color},{3.0 / float(nboot):.2f})',
                    showlegend=False,
                    name="Confidence Interval",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=_x,
                    y=_y,
                    mode="lines",
                    line = dict(width=5),
                    fillcolor=f'rgba({traj_color},{0.1:.2f})',
                    line_color=f'rgba({traj_color},{0.1:.2f})',
                    showlegend=True,
                    name="95% Confidence Interval",
                    legendgroup="trajectory"
                ))

    x2 = np.linspace(0, limit_age_max/time_unit_size, limit_age_max+1) #np.linspace(np.min(x), np.max(x), 100)
    y2 = equation(p, x2)

    # Prediction Interval
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(y))**2 / np.sum((y - np.mean(y))**2))   
    pi_mean       = np.mean(pi)
    pi_median     = np.median(pi)
    
    ret_val += "<b>Statistics</b><br>"
    ret_val += f"Chi^2: {chi2:.2f}<br>"
    ret_val += f"Reduced chi^2: {chi2_red:.2f}<br>"
    ret_val += f"Standard deviation of the error: {s_err:.2f}<br>"
    ret_val += f"Prediction interval:<br> mean={pi_mean:.2f}, median={pi_median:.2f}<br>"
    
    if plateau_area_start is not None:
        idx           = np.where(x2 < plateau_area_start)[0]
        pi_mean_idx   = np.mean(pi[idx])
        pi_median_idx = np.median(pi[idx])
        ret_val += f"Prediction interval < {plateau_area_start} {time_unit_name}:<br> mean={pi_mean_idx:.2f}, median={pi_median_idx:.2f}<br>"

    # mean prediction
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        line_color=f'rgba({traj_color},1.)',
        name=f"{traj_label.title()} trajectory",
        legendgroup="trajectory"
    ))
    # prediction interval
    if plot_PI:
        fig.add_trace(go.Scatter(
            x=list(x2)+list(x2[::-1]),
            y=list(y2-pi)+list(y2+pi)[::-1],
            fill='toself',
            fillcolor=f'rgba({traj_color},{fillcolor_alpha})',
            line_color=f'rgba({traj_color},{fillcolor_alpha+0.2})',
            showlegend=True,
            name="95% Prediction Interval",
            legendgroup="trajectory"
        ))
        

    if highlight_outliers is not None:
        idx = np.where(df_new["sampleID"].astype(str).isin(highlight_outliers))[0]
        fig.add_trace(go.Scatter(
            x=y_new[idx],
            y=y_pred_new[idx],
            mode="markers",
            marker=marker_outlier,
            showlegend=True,
            name="Outliers",
            text=list(df_new["sampleID"].values[idx]), 
            hovertemplate = '<b>Healthy reference sample outside the healthy region</b><br><br>'+
                            '<b>SampleID</b>: %{text}<br>'+
                            '<b>Age</b>: %{x:.2f}'+
                            '<br><b>MMI</b>: %{y}<br>',
            hoveron="points"
        ))
    
    if longitudinal_mode is not None:
        if longitudinal_mode == "markers":
            fig.add_trace(go.Scatter(
                    x=y_new,
                    y=y_pred_new,
                    mode=longitudinal_mode,
                    line=dict(width=3, dash='dash', color=f'rgba({traj_color},0.65)'),
                    marker=dict(size=10, color=f'rgba({traj_color},0.65)'),
                    showlegend=True,
                    name="Healthy samples",
                    text=list(df_new["sampleID"].values), 
                    hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                    '<b>SampleID</b>: %{text}<br>'+
                                    '<b>Age</b>: %{x:.2f}'+
                                    '<br><b>MMI</b>: %{y}<br>',
                    hoveron="points"
                ))
        else:
            # longitudinal - line per subject
            for trace in df_new["subjectID"].unique():
                idx = np.where(df_new["subjectID"]==trace)[0]
                fig.add_trace(go.Scatter(
                    x=y_new[idx],
                    y=y_pred_new[idx],
                    mode=longitudinal_mode,
                    line=dict(width=3, dash='dash', color=f'rgba({traj_color},0.65)'),
                    marker=dict(size=10, color=f'rgba({traj_color},0.65)'),
                    showlegend=longitudinal_showlegend,
                    name=trace,
                    text=list(df_new["sampleID"].values[idx]), 
                    hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                    '<b>SampleID</b>: %{text}<br>'+
                                    '<b>Age</b>: %{x:.2f}'+
                                    '<br><b>MMI</b>: %{y}<br>',
                    hoveron="points"
                ))
    
    
    return fig, ret_val, mae, r2, pi_median, x2, pi, y2

def plot_2_trajectories(estimator_ref, val1, val2, feature_cols, degree=2, plateau_area_start=2, limit_age=1200, start_age=0, time_unit_size=1, time_unit_name="days", 
                        linear_pval=False, nonlinear_pval=False, img_file_name=None, longitudinal_mode="markers+lines", 
                        website=False, plot_CI=False, plot_PI=True, layout_settings=None):
    val1 = val1.sort_values(by="age_at_collection")
    X1, y1 = df2vectors(val1, feature_cols)
    y_pred1 = estimator_ref.predict(X1)
    sid1 = val1["sampleID"].values
    
    val2 = val2.sort_values(by="age_at_collection")
    X2, y2 = df2vectors(val2, feature_cols)
    y_pred2 = estimator_ref.predict(X2)
    sid2 = val2["sampleID"].values
    

    fig = go.Figure()
    ret_val = ""
    
    limit_age_max1 = int(max(val1["age_at_collection"]))+1
    limit_age_max2 = int(max(val2["age_at_collection"]))+1
    limit_age_max = max(limit_age_max1, limit_age_max2)

    if plateau_area_start is not None:
        # shaded area where plateau is expected
        if plateau_area_start//time_unit_size+1 < limit_age//time_unit_size+1:
            _x = np.linspace(plateau_area_start, limit_age_max, 10)//time_unit_size+1
            fig.add_trace(go.Scatter(
                x=list(_x)+list(_x[::-1]),
                y=list(np.zeros(10))+list(np.ones(10)*limit_age//time_unit_size+1),
                fill='toself',
                fillcolor='rgba(220,220,220,0.5)',
                line_color='rgba(220,220,220,0.5)',
                showlegend=True,
                name=f"sample at time > {plateau_area_start} days",
            ))
    else:
        plateau_area_start = limit_age
    
    
    fig, ret_val1, _, _, _, _, _, _ = plot_1_trajectory(fig, estimator_ref, val1, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color="0,0,255", traj_label="reference", 
                                                            plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, longitudinal_mode=longitudinal_mode, longitudinal_showlegend=False,
                                                            plot_CI=plot_CI, plot_PI=plot_PI)
    fig, ret_val2, _, _, _, _, _, _ = plot_1_trajectory(fig, estimator_ref, val2, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color="255,0,0", traj_label="other", 
                                                            plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, longitudinal_mode=longitudinal_mode, longitudinal_showlegend=False,
                                                            plot_CI=plot_CI, plot_PI=plot_PI)

              
    # dataframe will be used for linear and nonlinear p-value calculation
    df = pd.DataFrame(data={"y":np.concatenate([y1, y2]), "y_pred":np.concatenate([y_pred1, y_pred2]),
                            "diftimeunit_y":np.concatenate([y1, y2]), "diftimeunit_y_pred":np.concatenate([y_pred1, y_pred2]),
                            "sampleID":list(sid1)+list(sid2),
                            "label": ["reference"]*len(y1) + ["other"]*len(y2)})

    if linear_pval:
        df = df[(df.y<plateau_area_start) & (start_age<df.y)]
        
        equation = lambda a, b: np.polyval(a, b) 
        xx = np.linspace(0, plateau_area_start, 10)/time_unit_size+1

        
        # lines
        p1, _cov = np.polyfit(df[df.label=="reference"].y.values//time_unit_size+1, df[df.label=="reference"].y_pred.values//time_unit_size+1, 1, cov=True)  
        fig.add_trace(go.Scatter(
            x=xx,
            y=equation(p1, xx),
            mode="lines",
            line = dict(width=5, dash='dash', color="rgba(0,0,255,1.0)"),
            marker=dict(size=10, color='red'),
            showlegend=True,
            name='Reference linear line',
        ))
        
        p2, _cov = np.polyfit(df[df.label=="other"].y.values//time_unit_size+1, df[df.label=="other"].y_pred.values//time_unit_size+1, 1, cov=True)         
        fig.add_trace(go.Scatter(
            x=xx,
            y=equation(p2, xx),
            mode="lines",
            line = dict(width=5, dash='dash', color='red'),
            marker=dict(size=10, color="rgba(255,0,0,1.0)"),
            showlegend=True,
            name='Other linear line',
        ))
        pval_k, pval_n = get_pvalue_regliner(df, group="label")
        ret_val += f"<b>Linear lines difference:</b><br>p = {pval_k:.3f}, {pval_n:.3f}"

    if nonlinear_pval:
        df = df[(df.y<limit_age) & (start_age<df.y)]
        
        pval = get_pvalue_permuspliner(df, group="label", degree=degree)
        
        ret_val += f"<b>Splines difference:</b><br>p = {pval:.3f}"
        
    fig.update_xaxes(title=f"Age [{time_unit_name}]", range=(start_age/time_unit_size, limit_age/time_unit_size), 
                     tick0=start_age/time_unit_size, dtick=2, showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
    fig.update_yaxes(title=f"Microbiome Maturation Index [{time_unit_name}]", range=(start_age/time_unit_size, limit_age/time_unit_size), 
                     tick0=start_age/time_unit_size, dtick=2, showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
    

    layout_settings_default = dict(
        height=900, 
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0, r=0, b=0, pad=0),
        title_text="Microbiome Trajectory"
    )

    if layout_settings is None:
        layout_settings = {}
    layout_settings_final = {**layout_settings_default, **layout_settings}
    
    fig.update_layout(**layout_settings_final)
    
    fig.update_layout(go.Layout(
        annotations=[
            go.layout.Annotation(
                text=ret_val1,
                font = dict(size = 10),
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.5,
                y=.01,
                bordercolor='black',
                bgcolor='rgba(0,0,220,0.5)',
                borderwidth=0.5,
                borderpad=8
            ),
            go.layout.Annotation(
                text=ret_val2,
                font = dict(size = 10),
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.0,
                y=.01,
                bordercolor='black',
                bgcolor='rgba(220,0,0,0.5)',
                borderwidth=0.5,
                borderpad=8
            ),
            go.layout.Annotation(
                text=ret_val,
                font = dict(size = 15),
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=.01,
                y=.99,
                bordercolor='black',
                bgcolor='white',
                borderwidth=0.5,
                borderpad=8
            )
        ]
    ), xaxis=dict(domain=[0.1, 0.1]))
    
    if img_file_name:
        fig.write_html(img_file_name)

    if not website:
        fig.show()
    
    return fig


