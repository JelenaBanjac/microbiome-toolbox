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
from elmtoolbox.variables import *
from elmtoolbox.helpers import df2vectors
from elmtoolbox.statistical_analysis import *
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations 


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

        fig, axs = plt.subplots(1, 3, figsize=(25, 7))

        axs[0].plot(features_num, maes, marker="o")
        i = np.argmin(maes)
        axs[0].plot(features_num[i], maes[i], marker="*", color="green", markersize=15)
        axs[0].set_xlabel("Number of important features");axs[0].set_ylabel("MAEs")

        axs[1].errorbar(features_num, accuracy_means, accuracy_stds,  marker='^')
        j = np.argmax(accuracy_means)
        axs[1].plot(features_num[i], accuracy_means[i], marker="*", color="green", markersize=15)
        axs[1].plot(features_num[j], accuracy_means[j], marker="*", color="red", markersize=15)
        axs[1].set_xlabel("Number of important features");axs[1].set_ylabel("Accuracy")

        axs[2].plot(features_num, r2s, marker="o")
        k = np.argmax(r2s)
        axs[2].plot(features_num[i], r2s[i], marker="*", color="green", markersize=15)
        axs[2].plot(features_num[k], r2s[k], marker="*", color="red", markersize=15)
        axs[2].set_xlabel("Number of important features");axs[2].set_ylabel("R-squared")

        plt.show()

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

        fig, axs = plt.subplots(1, 4, figsize=(25, 7))

        axs[0].errorbar(nzv_thresholds, accuracy_means, accuracy_stds,  marker='^')
        i = np.argmax(accuracy_means)
        axs[0].plot(nzv_thresholds[i], accuracy_means[i], marker="*", color="red", markersize=15)
        axs[0].set_xlabel("NZV thresholds");axs[0].set_ylabel("Accuracy")
        axs[0].set_xscale('log')


        axs[1].plot(nzv_thresholds, maes, marker="o")
        j = np.argmin(maes)
        axs[1].plot(nzv_thresholds[j], maes[j], marker="*", color="red", markersize=15)
        axs[1].set_xlabel("NZV thresholds");axs[1].set_ylabel("MAEs")
        axs[1].set_xscale('log')


        axs[2].plot(nzv_thresholds, r2s, marker="o")
        k = np.argmax(r2s)
        axs[2].plot(nzv_thresholds[k], r2s[k], marker="*", color="red", markersize=15)
        axs[2].set_xlabel("NZV thresholds");axs[2].set_ylabel("R-squared")
        axs[2].set_xscale('log')

        axs[3].plot(nzv_thresholds, features_num, marker="o")
        axs[3].set_xlabel("NZV thresholds");axs[3].set_ylabel("Number of features")
        axs[3].set_xscale('log')

        plt.show()

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

        fig, axs = plt.subplots(1, 4, figsize=(25, 7))

        axs[0].errorbar(correlation_thresholds, accuracy_means, accuracy_stds,  marker='^')
        i = np.argmax(accuracy_means)
        axs[0].plot(correlation_thresholds[i], accuracy_means[i], marker="*", color="red", markersize=15)
        axs[0].set_xlabel("correlation thresholds");axs[0].set_ylabel("Accuracy")

        axs[1].plot(correlation_thresholds, maes, marker="o")
        j = np.argmin(maes)
        axs[1].plot(correlation_thresholds[j], maes[j], marker="*", color="red", markersize=15)
        axs[1].set_xlabel("correlation thresholds");axs[1].set_ylabel("MAEs")

        axs[2].plot(correlation_thresholds, r2s, marker="o")
        k = np.argmax(r2s)
        axs[2].plot(correlation_thresholds[k], r2s[k], marker="*", color="red", markersize=15)
        axs[2].set_xlabel("correlation thresholds");axs[2].set_ylabel("R-squared")

        axs[3].plot(correlation_thresholds, features_num, marker="o")
        axs[3].set_xlabel("correlation thresholds");axs[3].set_ylabel("Number of features")

        plt.show()

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

    feature_cols = list(set(feature_cols)-correlated_features)

    return feature_cols


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
                    degree=2, nboot=None,
                    patent=False):
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
    
    fig, ret_val, outliers, mae, r2, pi_median = plot_1_trajectory(fig, estimator, df, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color="26,150,65", traj_label="reference", plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, nboot=nboot, longitudinal=group is None)
    
    X, y = df2vectors(df, feature_cols)
    y_pred = estimator.predict(X)
    
    y = np.array(y)/time_unit_size
    y_pred = np.array(y_pred)/time_unit_size
    
    df["y"] = y
    df["y_pred"] = y_pred
    
    equation = lambda a, b: np.polyval(a, b) 
        
    # Data
    if group is not None:
        
        colors = px.colors.qualitative.Plotly
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
                _df = df[df[group].isin(c)]
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
                ))
                # prediction interval
                fig.add_trace(go.Scatter(
                    x=list(x2)+list(x2[::-1]),
                    y=list(y2-pi)+list(y2+pi)[::-1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(colors_rgb[i])+[0.15])}',
                    line_color=f'rgba{tuple(list(colors_rgb[i])+[0.25])}', #f'rgba({traj_color},0.25)',
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
                _df = df[df[group].isin(c)]
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
                for trace in _df["subjectID"].unique():
                    idx2 = np.where(_df["subjectID"]==trace)[0]
                    fig.add_trace(go.Scatter(
                        x=y[idx][idx2],
                        y=y_pred[idx][idx2],
                        mode="lines+markers",
                        line = dict(width=3, dash='dash', color=color),
                        marker=dict(size=10, color=color),
                        showlegend=True,
                        name=trace,
                        text=list(_df["sampleID"].values[idx2]), 
                        hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                        f'<b>Group ({group}): {group_trace}</b><br>'+
                                        '<b>SampleID</b>: %{text}<br>'+
                                        '<b>Age</b>: %{x:.2f}'+
                                        '<br><b>MMI</b>: %{y}<br>',
                        hoveron="points"
                    ))
      

    X_other, y_other, y_other_pred = None, None, None
    if df_other is not None:
        df_other = df_other.sort_values(by="age_at_collection")

        X_other, y_other = df2vectors(df_other, feature_cols)
        y_other_pred = estimator.predict(X_other)
        
        for trace in df_other["subjectID"].unique():
            idx = np.where(df_other["subjectID"]==trace)[0]
            fig.add_trace(go.Scatter(
                x=y_other.values[idx],
                y=y_other_pred[idx],
                mode="lines+markers",
                line = dict(width=3, dash='dash'),
                marker=dict(size=10, color='rgba(255,0,0,0.75)'),
                showlegend=True,
                name=trace,
                text=list(df["sampleID"].values[idx]), 
                hovertemplate = '<b>Other sample</b><br><br>'+
                                '<b>SampleID</b>: %{text}<br>'+
                                '<b>Age</b>: %{x:.2f}<br>'+
                                '<b>MMI</b>: %{y}<br>',
                hoveron="points"
            ))


    fig.update_xaxes(title=f"Age [{time_unit_name}]", range=(start_age//time_unit_size, limit_age//time_unit_size), 
                    tick0=start_age//time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray') 
    fig.update_yaxes(title=f"Microbiome Maturation Index [{time_unit_name}]", range=(start_age//time_unit_size, limit_age//time_unit_size), 
                    tick0=start_age//time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray')  
    
    fig.update_layout(height=1000, width=1100, 
                      #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=0, r=0, b=0, pad=0),
                      title_text="Microbiome Trajectory")
    
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
    
    if img_file_name:
        fig.write_html(img_file_name)
    fig.show()
    
    return fig, outliers, mae, r2, pi_median




def plot_1_trajectory(fig, estimator, df, bacteria_names, limit_age, time_unit_size, time_unit_name, traj_label, plateau_area_start, traj_color, limit_age_max, degree=2, nboot=50, longitudinal=True): 
    df = df.sort_values(by="age_at_collection")

    X, y = df2vectors(df, bacteria_names)
    y_pred = estimator.predict(X)
    
    y = np.array(y)/time_unit_size
    y_pred = np.array(y_pred)/time_unit_size
    
    mae   = round(np.mean(abs(y_pred - y)), 2)
    r2    = r2_score(y, y_pred)
    coeff = stats.pearsonr(y_pred, y)
    
    ret_val = "<b>Performance Information</b><br>"
    ret_val += f'MAE: {mae}<br>'
    ret_val += f'R^2: {r2:.3f}<br>'
    ret_val += f"Pearson: {coeff[0]:.3f}, 2-tailed p-value: {coeff[1]:.2e}<br>"
    
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
    if nboot:
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
                    name="95% Confidence Limit",
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
    ))
    # prediction interval
    fig.add_trace(go.Scatter(
        x=list(x2)+list(x2[::-1]),
        y=list(y2-pi)+list(y2+pi)[::-1],
        fill='toself',
        fillcolor=f'rgba({traj_color},0.15)',
        line_color=f'rgba({traj_color},0.25)',
        showlegend=False,
        name="95% Prediction Interval",
    ))
    
    
    outliers = set()
    for i in range(len(y)):
        idx = int(x2[int(y[i])])
        lowlim = (y2-pi)[idx]
        uplim  = (y2+pi)[idx]
        if y_pred[i] > uplim or y_pred[i] < lowlim:
            outliers.add(df.iloc[i]["subjectID"])
    
    if longitudinal:
        # longitudinal - line per subject
        for trace in df["subjectID"].unique():
            idx = np.where(df["subjectID"]==trace)[0]
            fig.add_trace(go.Scatter(
                x=y[idx],
                y=y_pred[idx],
                mode="markers+lines",
                line = dict(width=3, dash='dash', color=f'rgba({traj_color},0.35)'),
                marker=dict(size=10, color=f'rgba({traj_color},0.35)'),
                showlegend=True,
                name=trace,
                text=list(df["sampleID"].values[idx]), 
                hovertemplate = '<b>Healthy reference sample</b><br><br>'+
                                '<b>SampleID</b>: %{text}<br>'+
                                '<b>Age</b>: %{x:.2f}'+
                                '<br><b>MMI</b>: %{y}<br>',
                hoveron="points"
            ))
    
    
    return fig, ret_val, outliers, mae, r2, pi_median

def plot_2_trajectories(estimator_ref, val1, val2, feature_cols, degree=2, plateau_area_start=2, limit_age=1200, start_age=0, time_unit_size=1, time_unit_name="days", title=None, everytick=False, linear_pval=False, nonlinear_pval=False, img_file_name=None, nboot=None):
    val1 = val1.sort_values(by="age_at_collection")
    val2 = val2.sort_values(by="age_at_collection")

    X1, y1 = df2vectors(val1, feature_cols)
    y_pred1 = estimator_ref.predict(X1)
    sid1 = val1["sampleID"].values

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
    
    
    fig, ret_val1, _, _, _, _ = plot_1_trajectory(fig, estimator_ref, val1, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color="0,0,255", traj_label="reference", plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, nboot=nboot)
    fig, ret_val2, _, _, _, _ = plot_1_trajectory(fig, estimator_ref, val2, feature_cols, limit_age, time_unit_size, time_unit_name, traj_color="255,0,0", traj_label="other", plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, nboot=nboot)

              
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
                     tick0=start_age/time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray') 
    fig.update_yaxes(title=f"Microbiome Maturation Index [{time_unit_name}]", range=(start_age/time_unit_size, limit_age/time_unit_size), 
                     tick0=start_age/time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray')  
    

    fig.update_layout(height=1000, width=1100, 
                      #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=0, r=0, b=0, pad=0),
                      title_text="Microbiome Trajectory")
    
    fig.update_layout(go.Layout(
        annotations=[
            go.layout.Annotation(
                text=ret_val1,
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
    fig.show()



def plot_longitudinal_data(estimator, df, feature_cols, outliers, df_other=None, zscore_above=None, zscore_below=None, density_plot=False, traj_color="green", title=None, longitudinal_dir=None):
    """Longitudinal plot

    Lines connecting the samples of the same infant. Plotted as a trajectory per infant.
    """
    X, y = df2vectors(df, feature_cols)
    y_pred = estimator.predict(X)

    X_other, y_other, y_other_pred = None, None, None
    if df_other:
        X_other, y_other = df2vectors(df_other, feature_cols)
        y_other_pred = estimator.predict(X_other)
        
    plt.rc('legend', fontsize=11)
    
    yr_limit = 2
    degree = 2
    nboot = None  #500
    
    mae   = round(np.mean(abs(y_pred - y)), 2)
    r2    = r2_score(y, y_pred)
    coeff = stats.pearsonr(y_pred, y.values)
    idx       = np.where(y.values < yr_limit*365)[0]
    mae_idx   = round(np.mean(abs(y_pred[idx] - y.values[idx])), 2)
    r2_idx    = r2_score(y.values[idx], y_pred[idx])
    coeff_idx = stats.pearsonr(y_pred[idx], y.values[idx])

    # Plot data
    equation = lambda a, b: np.polyval(a, b) 

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

    # Plotting --------------------------------------------------------------------
    subjectIDs = df.subjectID.unique()
    
    for sid_main in subjectIDs:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, 1200);ax.set_ylim(0, 1200)
        ax.fill_between(np.linspace(yr_limit*365, 5*365, 10), np.zeros(10), np.ones(10)*1200, alpha=0.3, color="gray", label=f"infant sample older than {yr_limit} yrs")

        # Borders
        ax.spines["top"].set_color("0.5")
        ax.spines["bottom"].set_color("0.5")
        ax.spines["left"].set_color("0.5")
        ax.spines["right"].set_color("0.5")
        ax.get_xaxis().set_tick_params(direction="out")
        ax.get_yaxis().set_tick_params(direction="out")
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left() 

        # Labels
        if title:
            plt.title(title, fontsize="16", fontweight="bold")
        plt.xlabel(f"Age [days]")
        plt.ylabel(f"Microbiome Maturation Index [days]")
        #plt.xlim(np.min(x) - 1, np.max(x) + 1)

        sns.scatterplot(x=y, y=y_pred, ax=ax, label="Healthy samples", color=traj_color)


        x2 = np.linspace(0, 1200, 1200) #np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Confidence Interval (select one)
        # sns.regplot(x=x, y=y_model, color ='red', label="Prediction fit", order=degree)
        if nboot:
            bootindex = sp.random.randint
            for _ in range(nboot):
                resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
                # Make coeffs of for polys
                pc = np.polyfit(y, y_pred + resamp_resid, degree)                   
                # Plot bootstrap cluster
                ax.plot(y, np.polyval(pc, y), "g-", linewidth=2, alpha=3.0 / float(nboot))

        # Fit   
        #sns.lineplot(x=x, y=y_model, color="green", style=True, linewidth=5, alpha=0.5, label="Fit", ax=ax)  
        sns.lineplot(x=x2, y=y2, color=traj_color, style=True, linewidth=5, alpha=0.5, label="Healthy reference", ax=ax)  

        # Prediction Interval
        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(y))**2 / np.sum((y - np.mean(y))**2))   
        sns.lineplot(x=x2, y=y2 - pi, dashes=[(2,2)], style=True, color=traj_color, label="95% Prediction Interval", ax=ax)
        sns.lineplot(x=x2, y=y2 + pi, dashes=[(2,2)], style=True, color=traj_color, ax=ax, label=None)

        pi_mean       = np.mean(pi)
        pi_median     = np.median(pi)
        idx           = np.where(x2 < yr_limit*365)[0]
        pi_mean_idx   = np.mean(pi[idx])
        pi_median_idx = np.median(pi[idx])

        if df_other is not None:
            ax.scatter(y_other, y_other_pred, c="red", label="Unhealthy sample", alpha=alpha, marker="x")
            p_fin, cov_fin = np.polyfit(y_other, y_other_pred, degree, cov=True)                
            y_model_fin = equation(p_fin, y_other)   
            sns.regplot(x=y_other, y=y_model_fin, color="red", label="Unhealthy prediction fit", order=degree, scatter=False)

            if zscore_above is not None:
                unhealthy_with_zscore_above = np.where(df_other.subjectID.isin(zscore_above))[0]
                ax.scatter(np.array(y_other)[unhealthy_with_zscore_above], np.array(y_other_pred)[unhealthy_with_zscore_above], label="with z-score > +2std", alpha=0.4, facecolors='none', edgecolors='#8a0031', lw=4, marker="8", s=130)
            if zscore_below is not None:
                unhealthy_with_zscore_below = np.where(df_other.subjectID.isin(zscore_below))[0]
                ax.scatter(np.array(y_other)[unhealthy_with_zscore_below], np.array(y_other_pred)[unhealthy_with_zscore_below], label="with z-score < -2std", alpha=0.4, facecolors='none', edgecolors='blue', lw=4, marker="s", s=130)


        
        idx = np.where(x2 < yr_limit*365)[0]

        prefix = None
        # Data
        for sid in set(subjectIDs)-set(outliers):
            y_subj = y[df.subjectID==sid].values
            idx_sorted = np.argsort(y_subj)
            y_pred_subj = y_pred[df.subjectID==sid]
            sns.lineplot(x=y_subj[idx_sorted], y=y_pred_subj[idx_sorted], ax=ax, marker="o", color="green", alpha=0.3)
            
            if sid == sid_main:
                sns.lineplot(x=y_subj[idx_sorted], y=y_pred_subj[idx_sorted], ax=ax, marker="o", color="green", alpha=1.0, markersize=15, linewidth=5)
                ax.text(0.2, 0.8, f"Subject ID: {sid}", transform=ax.transAxes, fontsize=15, verticalalignment='center', bbox=dict(boxstyle='round', facecolor="w", edgecolor="k", alpha=0.5))
                prefix = "_inside"

        # Data
        for sid in outliers:
            y_subj = y[df.subjectID==sid].values
            idx_sorted = np.argsort(y_subj)
            y_pred_subj = y_pred[df.subjectID==sid]
            sns.lineplot(x=y_subj[idx_sorted], y=y_pred_subj[idx_sorted], ax=ax, marker="o", color="red", alpha=0.3)
            
            if sid == sid_main:
                sns.lineplot(x=y_subj[idx_sorted], y=y_pred_subj[idx_sorted], ax=ax, marker="o", color="red", alpha=1.0, markersize=15, linewidth=5)
                ax.text(0.2, 0.8, f"Subject ID: {sid}", transform=ax.transAxes, fontsize=15, verticalalignment='center', bbox=dict(boxstyle='round', facecolor="w", edgecolor="k", alpha=0.5))
                prefix = "_outside"
        
        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        #handles = list(np.array(handles)[np.array([0,2,5,6,7,8])])
        #labels = list(np.array(labels)[np.array([0,2,5,6,7,8])])
        #display = (0,2,5,6,7,8,9, 10)
        display = np.where(np.array(labels)!='True')[0]
        anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
        legend = plt.legend(
            [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
            [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
            loc=3, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
        )  
        frame = legend.get_frame().set_edgecolor("0.5")
        plt.tight_layout()
        if longitudinal_dir:
            pathlib.Path(longitudinal_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{longitudinal_dir}/{prefix}_{sid_main}.png")
            plt.clf()
            plt.close()
        else:
            plt.show()


def plot_shap_abundances_and_ratio(estimator, train, important_features, bacteria_name, short_bacteria_name_fn, file_name, threshold = 0.5):
    """ Plot the shap values and abundances and ratios

    Plot that is used when working with ratios. We wanted to see plots side-by-side that represent the abundance of bacteria. On the other plot
    the ratio of these two bacterias. We wanted to understand better this mapping of abundances to ratios. At the end we concluded that we want 
    to use the bacteria in the numerator that has the least number of crossings when we look at the 2 abundances.
    """
    sns.set_style("whitegrid")
    _X_train, _y_train = df2vectors(train, important_features)
    _y_train = np.array(_y_train.values)//30
    #print(_y_train.shape, _X_train.shape)
    #df_abundance ={"X_train":X_train, "y_train":np.array(y_train.values)//30}
    
    train["month"] = train["age_at_collection"]//30
    train_mon = train[important_features+[bacteria_name, "month"]].groupby("month").agg(np.mean).reset_index()
    X_train, y_train = df2vectors(train_mon, important_features, time_col="month")
    X_features_and_age = np.hstack((X_train, np.array(y_train).reshape(-1, 1)))
    
    important_features_short = list(map(short_bacteria_name_fn, important_features)) #list(map(lambda x: f"log of ratio {x.split('g__')[1]}/{bacteria_name.split('g__')[1]}" if len( x.split("g__"))>1 else 'Other' ,important_features))
    
    plt.rc('axes', labelsize= 14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize= 12)
    plt.rc('legend', fontsize= 13)

    fig, axs = plt.subplots(len(important_features), 3, figsize=(30,len(important_features)*7))
    
    for i in range(len(important_features)):
        name = important_features[i]
        name_short = important_features_short[i]
        
        #idx_same = np.where(np.isclose(X_train[:,i], 0.0, atol=threshold))[0]
        
        train_bacteria_name = train[[f"abundance_{bacteria_name}", "month"]].groupby("month").agg(np.mean)
        train_name          = train[[f"abundance_{name}", "month"]].groupby("month").agg(np.mean)
        months              = train_bacteria_name.reset_index()["month"].values
        diff_ref_and_taxa   = train_bacteria_name.values.reshape(-1) - train_name.values.reshape(-1)
        diff_ref_and_taxa   = -diff_ref_and_taxa if diff_ref_and_taxa[0]<0 else diff_ref_and_taxa
        signchange =  ((np.roll(np.sign(diff_ref_and_taxa), 1) - np.sign(diff_ref_and_taxa)) != 0).astype(int) 
        signchange[0] = 0
        idx_cross = np.where(signchange==1)[0]

        # shap plot
        #X_features_and_age = np.hstack((X_train, np.array(y_train).reshape(-1, 1)))
        idx_of_age = X_features_and_age.shape[1]-1
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_features_and_age)
        inds = shap.approximate_interactions(idx_of_age, shap_values, X_features_and_age)
        shap.dependence_plot(i, shap_values, X_features_and_age, feature_names=important_features_short+["Age at collection [month]"], interaction_index=inds[idx_of_age], ax=axs[i][0], show=False, dot_size=30)
        axs[i][0].plot(np.zeros(10), np.linspace(np.min(shap_values[:,i]), np.max(shap_values[:,i]), 10), lw=10, alpha=0.5, color="orange")
        axs[i][0].plot(X_features_and_age[:,i][idx_cross],shap_values[:,i][idx_cross], marker="X", markersize=15, color="red", lw=0, alpha=0.7)
        
        sns.pointplot(x=train["age_at_collection"]//30, y=train[f"abundance_{name}"], capsize=.2, alpha=0.4, ax=axs[i][1], color="green", label=name_short)
        sns.pointplot(x=train["age_at_collection"]//30, y=train[f"abundance_{bacteria_name}"], capsize=.2, alpha=0.4, ax=axs[i][1], color="blue", label=f"{bacteria_name.split('g__')[1]} (reference)")
        #axs[i][1].plot(months[idx_same], train_name.values.reshape(-1)[idx_same], marker="*", markersize=10, color="orange", lw=0, label="same")
        axs[i][1].plot(months[idx_cross]-0.5, train_name.values.reshape(-1)[idx_cross], marker="X", markersize=15, color="red", lw=0, alpha=1.0, label="crossed")
        axs[i][1].set_ylabel("Abundance")
        axs[i][1].set_xlabel("Age at collection [month]")

        
        sns.pointplot(x=_y_train, y=_X_train[:,i], capsize=.2, alpha=0.4, ax=axs[i][2], color="gray")
        #x = (np.array(y_train)//30).flatten()
        #y = X_train[:,i].flatten()
        #axs[i][2].plot(x[idx_same], y[idx_same], marker="*", markersize=10, color="orange", lw=0)
        axs[i][2].plot(np.linspace(0, 38, 10), np.zeros(10), lw=10, alpha=0.5, color="orange")
        axs[i][2].plot(months[idx_cross]-0.5, np.zeros(len(idx_cross)), alpha=0.7, marker="X", markersize=15, color="red", lw=0)
        axs[i][2].set_ylabel(important_features_short[i])
        axs[i][2].set_xlabel("Age at collection [month]")
        

        custom_lines = [Line2D([0], [0], color="green", ls="-", marker="o", lw=4),
                        Line2D([0], [0], color="blue", ls="-", marker="o", lw=4),
                        #Line2D([0], [0], color="orange", marker="*", markersize=10, ls="--", lw=0),
                        Line2D([0], [0], color="red", marker="X", markersize=10, ls="--", lw=0),
                        Line2D([0], [0], color="orange", ls="-", alpha=0.4, lw=10)]
        #print(name_short)
        axs[i][2].legend(custom_lines, [important_features[i].split('g__')[1], f"{bacteria_name.split('g__')[1]} (reference)", "crossed", "zero log-ratio"], loc="upper left", bbox_to_anchor=(1, 1))

    #print("/".join(file_name.split("/")[:-1]))
    pathlib.Path("/".join(file_name.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

