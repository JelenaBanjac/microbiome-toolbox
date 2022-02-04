import copy
import itertools
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy as sp
import scipy.stats as stats
import shap
from natsort import natsorted
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler

from microbiome.enumerations import (
    AnomalyType,
    FeatureColumnsType,
    FeatureExtraction,
    TimeUnit,
)
from microbiome.statistical_analysis import permuspliner, regliner

RANDOM_STATE = 42


class MicrobiomeTrajectory:
    def __init__(
        self,
        dataset,
        feature_columns,
        feature_extraction=FeatureExtraction.NONE,
        time_unit=TimeUnit.DAY,
        anomaly_type=AnomalyType.PREDICTION_INTERVAL,
        train_indices=None,
    ):
        self.dataset = copy.deepcopy(dataset)
        # initialize feature columns (only once, in constructor!)
        if isinstance(feature_columns, (list, np.ndarray)):
            self._feature_columns = feature_columns
        elif feature_columns == FeatureColumnsType.BACTERIA:
            self._feature_columns = self.dataset.bacteria_columns
        elif feature_columns == FeatureColumnsType.METADATA:
            self._feature_columns = self.dataset.metadata_columns
        elif feature_columns == FeatureColumnsType.BACTERIA_AND_METADATA:
            self._feature_columns = self.dataset.bacteria_and_metadata_columns

        self.axis_settings_default = dict(
            tick0=0,
            mirror=True,
            # dtick=2,
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
            autorange=True,
        )

        results = self.get_less_feature_columns(plot=True, technique=feature_extraction)
        self.feature_columns = results["feature_columns"]
        self.feature_importance = results["feature_importance"]
        self.feature_columns_plot = results["fig"]
        self.feature_columns_plot_ret_val = results["ret_val"]
        self.feature_columns_plot_config = results["config"]

        self.dataset.time_unit = time_unit
        self.anomaly_type = anomaly_type

        # TODO: handle less features here too!
        if train_indices is None:
            train_indices = (self.dataset.df.reference_group == True).values

        df_train = self.dataset.df.iloc[train_indices]
        df_test = self.dataset.df.iloc[~train_indices]

        X_train = df_train[self.feature_columns].values
        y_train = df_train.age_at_collection.values

        groups_train = list(df_train.subjectID.values)

        # estimator = RandomForestRegressor(random_state=RANDOM_STATE)
        # estimator.fit(X, y)
        parameters_gridsearch = {"n_estimators": [50, 100, 150]}
        rfr = RandomForestRegressor(random_state=RANDOM_STATE)
        n_splits = 5 if len(np.unique(groups_train)) > 5 else len(np.unique(groups_train))
        gkf = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups_train))
        search = GridSearchCV(rfr, parameters_gridsearch, cv=gkf)
        search.fit(X_train, y_train)
        self.estimator = search.best_estimator_

        y_train_pred = self.estimator.predict(X_train).astype(float)
        self.dataset.df.loc[df_train.index, "MMI"] = y_train_pred

        if len(df_test) > 0:
            X_test = df_test[self.feature_columns].values
            y_test = df_test.age_at_collection.values
            y_test_pred = self.estimator.predict(X_test).astype(float)
            self.dataset.df.loc[df_test.index, "MMI"] = y_test_pred

        self.reference_groups = self.dataset.df.reference_group.values.astype(bool)
        self.X = self.dataset.df[self.feature_columns]
        self.y = self.dataset.df.age_at_collection.values.astype(float)
        self.y_pred = self.dataset.df.MMI.values.astype(float)
        self.sample_ids = self.dataset.df.sampleID.values.astype(str)
        self.subject_ids = self.dataset.df.subjectID.values.astype(str)
        self.groups = self.dataset.df.group.values.astype(str)
        self.x = np.linspace(np.min(self.y), np.max(self.y), 100)

        self.layout_settings_default = dict(
            height=900,
            width=1200,
            plot_bgcolor="rgba(255,255,255,255)",
            paper_bgcolor="rgba(255,255,255,255)",
            margin=dict(l=70, r=70, t=70, b=70),
            font=dict(size=17),
            hoverdistance=-1,
            legend=dict(
                x=1.01,
                y=1,
            ),
        )

        self.color_reference = "26,150,65"
        self.color_non_reference = "255,150,65"
        self.color_anomaly = "255,0,0"
        colors = px.colors.qualitative.Plotly
        colors_rgb = [
            tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) for h in colors
        ]
        self.colors_rgb = [str(x)[1:-1] for x in colors_rgb]

        self.bacteria_colors = {}
        self.group_colors = {}
        self.subject_colors = {}
        self.palette = itertools.cycle(self.colors_rgb)

    def get_less_feature_columns(
        self,
        plot=False,
        technique=FeatureExtraction.NONE,
        thresholds=None,
        layout_settings=None,
        xaxis_settings=None,
        yaxis_settings=None,
    ):
        """Extract just few feature columns to reduce the number of features wlog.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the performance statistics, by default False.
            If True, the plot with MAEs and R2s w.r.t. the number of features used is returned.
        technique : FeatureExtraction, str, optional
            Technique used to reduce number of features, by default FeatureExtraction.NONE.
            Possible values: NONE, NEAR_ZERO_VARIANCE, CORRELATION, TOP_K_IMPORTANT
        thresholds : list, optional
            Threshold values to be tested for performance, by default None.
        layout_settings : dict, optional
            Layout settings fot the plotly, by default None.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
            - feature_columns: list of feature columns found used 3 reduction techniques.
            - fig : figure object of performances with different number of feature columns.
            - config : plotly config object.
        """
        fig = None
        config = None
        ret_val = None

        if technique == FeatureExtraction.NONE:
            feature_columns = self._feature_columns
        else:
            X = self.dataset.df[self._feature_columns].values
            y = self.dataset.df.age_at_collection.values
            groups = self.dataset.df.subjectID.values

            estimator = RandomForestRegressor(random_state=RANDOM_STATE)
            estimator.fit(X, y)

            scorer_list = ["r2", "neg_mean_squared_error"]
            scores = defaultdict(list)

            # get thresholds
            if thresholds is None:
                if technique == FeatureExtraction.TOP_K_IMPORTANT:
                    feature_importances = np.array(estimator.feature_importances_)
                    feature_importance = pd.DataFrame(
                        list(zip(self._feature_columns, feature_importances)),
                        columns=["feature_name", "feature_importance"],
                    )
                    feature_importance.sort_values(
                        by=["feature_importance"], ascending=False, inplace=True
                    )
                    important_features = feature_importance.feature_name.values
                    thresholds = np.linspace(1, 50, num=10, dtype=int)
                elif technique == FeatureExtraction.NEAR_ZERO_VARIANCE:
                    variances_ = np.nanvar(X, axis=0)
                    thresholds = np.linspace(
                        0, np.max(variances_), num=10, endpoint=False
                    )
                elif technique == FeatureExtraction.CORRELATION:
                    thresholds = np.linspace(0, 1, num=10, endpoint=False)

            estimator = RandomForestRegressor(random_state=RANDOM_STATE)
            for threshold in thresholds:
                if technique == FeatureExtraction.TOP_K_IMPORTANT:
                    feature_columns = important_features[:threshold]
                elif technique == FeatureExtraction.NEAR_ZERO_VARIANCE:
                    X = self.dataset.df[self._feature_columns].values
                    constant_filter = VarianceThreshold(threshold=threshold)
                    constant_filter.fit(X)
                    idx = np.where(constant_filter.get_support())[0]
                    feature_columns = self._feature_columns[idx]
                elif technique == FeatureExtraction.CORRELATION:
                    correlation_matrix = (
                        self.dataset.df[self._feature_columns].corr().abs()
                    )
                    upper = correlation_matrix.where(
                        np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool)
                    )
                    correlated_features = [
                        column
                        for column in upper.columns
                        if any(upper[column] > threshold)
                    ]
                    idx = np.isin(
                        self._feature_columns, correlated_features, invert=True
                    )
                    feature_columns = self._feature_columns[idx]

                X = self.dataset.df[feature_columns].values
                for scorer in scorer_list:
                    cv_scores = cross_val_score(
                        estimator,
                        X,
                        y,
                        scoring=scorer,
                        groups=groups,
                        cv=GroupShuffleSplit(random_state=RANDOM_STATE),
                    )
                    scores["score"].extend(cv_scores)
                    scores["scorer"].extend([scorer] * len(cv_scores))
                    scores["features_number"].extend(
                        [len(feature_columns)] * len(cv_scores)
                    )

                    scores["threshold"].extend([threshold] * len(cv_scores))

            df_scores = pd.DataFrame(data=scores)
            r2_mean = (
                df_scores[df_scores.scorer == "r2"]
                .groupby(by="features_number")
                .mean()["score"]
            )
            i = np.argmax(r2_mean.values)

            # excract only top important features
            thresholds = (
                df_scores[df_scores.scorer == "r2"]
                .groupby(by="features_number")
                .first()["threshold"]
                .values
            )
            threshold = thresholds[i]

            # get feature_columns
            if technique == FeatureExtraction.TOP_K_IMPORTANT:
                feature_columns = important_features[:threshold]
            elif technique == FeatureExtraction.NEAR_ZERO_VARIANCE:
                X = self.dataset.df[self._feature_columns].values
                constant_filter = VarianceThreshold(threshold=threshold)
                constant_filter.fit(X)
                idx = np.where(constant_filter.get_support())[0]
                feature_columns = self._feature_columns[idx]
            elif technique == FeatureExtraction.CORRELATION:
                correlation_matrix = self.dataset.df[self._feature_columns].corr().abs()
                upper = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool)
                )
                correlated_features = [
                    column for column in upper.columns if any(upper[column] > threshold)
                ]
                idx = np.isin(self._feature_columns, correlated_features, invert=True)
                feature_columns = self._feature_columns[idx]

            if plot:
                layout_settings_default = dict(
                    height=500,
                    width=400 * len(scorer_list),
                    plot_bgcolor="rgba(255,255,255,255)",
                    paper_bgcolor="rgba(255,255,255,255)",
                    margin=dict(l=0, r=0, b=0, pad=0),
                    title_text=f"{technique.name.title()} features",
                    font=dict(size=12),
                    hovermode="x",
                )
                if layout_settings is None:
                    layout_settings = {}
                layout_settings_final = {**layout_settings_default, **layout_settings}
                if xaxis_settings is None:
                    xaxis_settings = {}
                if yaxis_settings is None:
                    yaxis_settings = {}
                xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
                yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

                fig = make_subplots(
                    rows=1, cols=len(scorer_list), horizontal_spacing=0.2
                )
                ret_val = "<b>Performance Information</b><br>"

                for col, scorer in enumerate(scorer_list, start=1):
                    df_scorer = df_scores[df_scores.scorer == scorer]
                    scorer_mean = df_scorer.groupby(by="features_number").mean()[
                        "score"
                    ]
                    scorer_std = df_scorer.groupby(by="features_number").std()["score"]
                    thresholds = df_scorer.groupby(by="features_number").first()[
                        "threshold"
                    ]
                    hovertemplate = (
                        "Features number: %{x}<br>R2 Score: %{y}<br>"
                        + "Threshold: %{text}"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=scorer_mean.index,
                            y=scorer_mean.values,
                            error_y=dict(
                                type="data",
                                array=scorer_std.values,
                                visible=True,
                            ),
                            name=scorer,
                            hovertemplate=hovertemplate,
                            text=thresholds,
                        ),
                        row=1,
                        col=col,
                    )

                    rangey = (0, 1) if scorer == "r2" else None
                    showlegend = scorer == "r2"

                    ret_val += (
                        f"<b>{scorer.title()}</b><br>"
                        + f"Performance: {scorer_mean.values[i]:.3f}±{scorer_std.values[i]:.3f}<br>"
                        + f"Features number: {scorer_mean.index[i]}<br>"
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[scorer_mean.index[i]],
                            y=[scorer_mean.values[i]],
                            mode="markers",
                            marker_symbol="star",
                            marker_size=15,
                            marker_color="green",
                            name="optimal r2",
                            showlegend=showlegend,
                            hovertemplate="%{y:.0%}",
                        ),
                        row=1,
                        col=col,
                    )
                    fig.update_xaxes(
                        title="Number of features",
                        row=1,
                        col=col,
                        **xaxis_settings_final,
                    )
                    fig.update_yaxes(
                        title=scorer,
                        row=1,
                        col=col,
                        # range=rangey,
                        **yaxis_settings_final,
                    )

                fig.update_layout(**layout_settings_final)
                config = {
                    "toImageButtonOptions": {
                        "format": "svg",  # one of png, svg, jpeg, webp
                        "filename": "bacteria_abundances",
                        "height": layout_settings_final["height"],
                        "width": layout_settings_final["width"],
                        "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
                    }
                }

        X = self.dataset.df[feature_columns].values
        y = self.dataset.df.age_at_collection.values
        groups = self.dataset.df.subjectID.values

        estimator = RandomForestRegressor(random_state=RANDOM_STATE)
        estimator.fit(X, y)

        feature_importances = np.array(estimator.feature_importances_)
        feature_importance = pd.DataFrame(
            list(zip(feature_columns, feature_importances)),
            columns=["feature_name", "feature_importance"],
        )
        feature_importance.sort_values(
            by=["feature_importance"], ascending=False, inplace=True
        )

        results = {
            "fig": fig,
            "config": config,
            "ret_val": ret_val,
            "feature_columns": list(feature_columns),
            "feature_importance": feature_importance,
        }

        return results

    def get_prediction_and_confidence_interval(self, indices, x=None, degree=2):
        """Get prediction and confidence interval for samples indicated by indices.

        Parameters
        ----------
        indices : list
            Indices of samples that will be used to get the prediction and confidence interval.
        x : numpy.array
            Different sampling points for the age_at_collection values. By default it is None and
            the give age_at_collection values are used (vector y).
        degree : int
            Degree of the polynomial used to fit the data. By default it is 2.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - 'prediction_interval' : dict
                    Prediction interval for each sample sampled at x given by the mean and +- bounds.
                - 'confidence_interval' : dict
                    Confidence interval for each sample sampled at x given by the mean and +- bounds.

        Attributes
        ----------
        q : float
            Confidence level for the confidence interval. By default it is 0.975.
        p : numpy.array
            Polynomial coefficients, highest power first.
        y_model : numpy.array
            Model of the polynomial fit to the data.
        n : int
            Number of samples (observations).
        m : int
            Number of features (parameters).
        dof : int
            Degrees of freedom.
        t : float
            Student's t value. Used for CI and PI bands.
        residual : numpy.array
            Residuals of the polynomial fit. Estimates the error in the data.
        s_err : numpy.array
            Standard deviation of the error.
        y2 : numpy.array
            Prediction interval mean.
        pi : numpy.array
            Prediction interval bound.
        n_boot : int
            Number of bootstrap samples.
        y3 : numpy.array
            Confidence interval mean of every bootstraped version.
        """
        if x is None:
            x = copy.deepcopy(self.y)
            x = x[indices]
        y = self.y[indices]
        y_pred = self.y_pred[indices]

        # Stats
        p = np.polyfit(y, y_pred, degree)
        y_model = np.polyval(p, y)

        # Statistics
        q = 0.975
        n = y_pred.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(q, n - m)
        residual = y_pred - y_model
        # chi2 = np.sum((residual / y_model)**2)  # chi-squared; estimates error in data
        # chi2_red = chi2 / dof                   # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(residual ** 2) / dof)

        # Prediction Interval
        y2 = np.polyval(p, x)
        pi = (
            t
            * s_err
            * np.sqrt(1 + 1 / n + (x - np.mean(y)) ** 2 / np.sum((y - np.mean(y)) ** 2))
        )

        # Confidence Interval
        n_boot = 500
        bootindex = sp.random.randint
        y3 = np.empty((n_boot, len(x)))
        for b in range(n_boot):
            resamp_residual = residual[bootindex(0, len(residual) - 1, len(residual))]
            # make coefficients of for polys
            pc = np.polyfit(y, y_pred + resamp_residual, degree)
            # collect bootstrap cluster
            y3[b] = np.polyval(pc, x)

        results = {
            "prediction_interval": {
                "x": x,
                "mean": y2,
                "bound": pi,
            },
            "confidence_interval": {
                "x": x,
                "mean": np.mean(y3, axis=0),
                "bound": 1.96 * np.std(y3, axis=0),
            },
        }
        return results

    def get_anomalies(self, anomaly_type, indices):
        """Get anomalies for samples indicated by indices.

        Parameters
        ----------
        anomaly_type : str
            Type of anomaly to be detected.
        indices : list
            Indices of samples that will be used to identify anomalous samples among them.
            E.g. indices of all healthy samples where we are interested to find the anomalies.

        Returns
        -------
        anomalies : numpy.array
            Indices of the anomalous samples. Indices are relative to the original data.
            Possible values are: True - anomaly, False - non-anomaly, None - other samples that were
            not indicated with indices argument.

        Notes
        -----
        We use three different types to detect anomalies:
            1. Prediction interval
            2. Low-pass filter
            3. Isolation forest
        """
        y_pred = self.y_pred[indices]

        if anomaly_type == AnomalyType.PREDICTION_INTERVAL:
            degree = 3

            results = self.get_prediction_and_confidence_interval(
                indices=indices, degree=degree
            )

            # pi_x = results["prediction_interval"]["x"] == y
            pi_mean = results["prediction_interval"]["mean"]
            pi_bound = results["prediction_interval"]["bound"]

            # highlight_outliers
            anomaly = np.logical_or(
                np.less(y_pred, pi_mean - pi_bound),
                np.greater(y_pred, pi_mean + pi_bound),
            )

        elif anomaly_type == AnomalyType.LOW_PASS_FILTER:
            window = 10
            number_of_stds = 2

            y_pred_rolling_avg = (
                pd.Series(y_pred)
                .rolling(window=window, min_periods=1, center=True)
                .mean()
            )
            y_pred_rolling_std = (
                pd.Series(y_pred)
                .rolling(window=window, min_periods=1, center=True)
                .std(skipna=True)
            )
            anomaly = abs(y_pred - y_pred_rolling_avg) > (
                number_of_stds * y_pred_rolling_std
            )
        elif anomaly_type == AnomalyType.ISOLATION_FOREST:
            outliers_fraction = 0.1
            window = 5

            y_pred_rolling_avg = (
                pd.Series(y_pred)
                .rolling(window=window, min_periods=1, center=True)
                .mean()
            )
            y_pred_zscore = y_pred - y_pred_rolling_avg

            # Subset the dataframe by desired columns
            dataframe_filtered_columns = pd.DataFrame(
                data={
                    "y_pred": y_pred,
                    "y_pred_zscore": y_pred_zscore,
                }
            )

            # Scale the column that we want to flag for anomalies
            np_scaled = StandardScaler().fit_transform(dataframe_filtered_columns)
            scaled_time_series = pd.DataFrame(np_scaled)

            # train isolation forest
            model = IsolationForest(contamination=outliers_fraction)
            model.fit(scaled_time_series)

            # generate column for Isolation Forest-detected anomalies
            anomaly = model.predict(scaled_time_series)
            anomaly[anomaly == 1] = False  # inliner
            anomaly[anomaly == -1] = True  # outliner
            anomaly = anomaly.astype(bool)

        anomalies = np.empty(len(self.y_pred))
        anomalies[indices] = anomaly

        return anomalies

    def get_pvalue_linear(self, column, indices):
        """Get p-value for linear lines.

        Parameters
        ----------
        indices : list
            Indices of samples that contain only 2 groups that will be compared.

        Returns
        -------
        result["k"] : float
            P-value for linear line slopes.
        result["n"] : float
            P-value for linear line y-intercepts.

        Notes
        -----
        The cutoff for significance used is an alpha of 0.05.
        If the p-value is less than 0.05, we reject the null hypothesis that
        there's no difference between the means and conclude that a significant
        difference does exist.
        Below 0.05, significant. Over 0.05, not significant.
        """
        y = self.y[indices]
        y_pred = self.y_pred[indices]
        groups = getattr(self, column)[indices]

        group_values = np.unique(groups)
        print("group_values", group_values)

        assert len(group_values) == 2, f"Needs to have only 2 unique groups to compare, has {group_values}"

        df_stats = pd.DataFrame(
            data={
                "Input": list(y),
                "Output": list(y_pred),
                "Condition": list(groups),
            }
        )

        result = regliner(df_stats, {group_values[0]: 0, group_values[1]: 1})
        return result["k"], result["n"]

    def get_pvalue_spline(self, column, indices, degree=2):
        """Get p-value for spline lines.

        Parameters
        ----------
        indices : list
            Indices of samples that contain only 2 groups that will be compared.
        degree : int
            The degree of the polynomial.

        Returns
        -------
        result["pval"] : float
            P-value for spline lines.

        Notes
        -----
        The cutoff for significance used is an alpha of 0.05.
        If the p-value is less than 0.05, we reject the null hypothesis that
        there's no difference between the means and conclude that a significant
        difference does exist.
        Below 0.05, significant. Over 0.05, not significant.
        """
        y = self.y[indices]
        y_pred = self.y_pred[indices]
        groups = getattr(self, column)[indices]
        sample_ids = self.sample_ids[indices]

        group_values = np.unique(groups)

        assert len(group_values) == 2, f"Needs to have only 2 unique groups to compare, has {group_values}"

        df_stats = pd.DataFrame(
            data={
                "Input": list(y),
                "Output": list(y_pred),
                "Condition": list(groups),
                "sampleID": list(sample_ids),
            }
        )

        result = permuspliner(
            df_stats,
            xvar="Input",
            yvar="Output",
            category="Condition",
            degree=degree,
            cases="sampleID",
            groups=group_values,
            perms=500,
            test_direction="more",
            ints=1000,
            quiet=True,
        )

        return result["pval"]

    def get_top_bacteria_in_time(self, indices, time_current, time_delta):
        """Get top important bacteria for indicated time block.

        Time block is the time interval representing the different bacteria importance
        with their corresponding average and stds.

        Parameters
        ----------
        indices : list
            Indices of samples that will be used to calculate the time block
            averages. E.g. healthy samples without anomalies.
        time_current : float
            Current time point. Time point from which to collect samples into
            time block.
        time_delta : float
            Time delta from current time point. Time delta during which to collect.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
            - "feature_importance" : pandas.DataFrame
                DataFrame with bacteria in the rows, and their importance, average and std in the columns.
            - "df_time_block" : pandas.DataFrame
                DataFrame with all data used to calculate the time block bacterias importance and averages.
        """
        indices_timebox = (time_current <= self.y) & (
            self.y < time_current + time_delta
        )
        i = 0
        while len(indices_timebox) == 0:
            indices_timebox = (time_current - (2 * i - 1) * time_delta <= self.y) & (
                self.y < time_current + (1 + i) * time_delta
            )
            i += 0.1

        indices = np.logical_and(indices_timebox, indices)

        df_time_block = self.dataset.df.iloc[indices][
            self.feature_columns + ["age_at_collection", "MMI"]
        ]

        # average, std, of samples in the time block for a given bacteria
        avgs = df_time_block[self.feature_columns].mean(axis=0)
        stds = df_time_block[self.feature_columns].std(axis=0)
        shap_values = shap.TreeExplainer(self.estimator).shap_values(
            pd.DataFrame([avgs])
        )
        feature_importance = pd.DataFrame(
            data={
                "importance": np.abs(shap_values).mean(0),
                "feature_avg": avgs,
                "feature_std": stds,
            }
        )
        feature_importance.sort_values(by=["importance"], ascending=True, inplace=True)

        results = {
            "feature_importance": feature_importance,
            "df_time_block": df_time_block,
        }

        return results

    def get_bacteria_colors(self, val):
        if not self.bacteria_colors.get(val, None):
            self.bacteria_colors[val] = next(self.palette)
        return self.bacteria_colors[val]

    def get_subject_colors(self, val):
        if not self.subject_colors.get(val, None):
            self.subject_colors[val] = next(self.palette)
        return self.subject_colors[val]

    def get_group_colors(self, val):
        if not self.group_colors.get(val, None):
            self.group_colors[val] = next(self.palette)
        return self.group_colors[val]

    def get_intervention_point(self, X_i, y_i, time_block_ranges, indices):
        """Get intervention point.

        X_i : numpy.array
            Features of i-th sample.
        y_i : float
            Age at collection of i-th sample.
        time_block_ranges : list
            Units or time block sizes.
        indices : list
            Indices of samples that will be used to calculate the time block
            averages. E.g. healthy samples without anomalies.
        """
        normal_vals = [False]

        current_time = time_block_ranges[0]
        for time in time_block_ranges[1:]:
            time_delta = time - current_time
            if current_time <= y_i < current_time + time_delta:
                break

            current_time += time_delta

        shap_values = shap.TreeExplainer(self.estimator).shap_values(
            pd.DataFrame([X_i])
        )

        feature_importance_outlier = pd.DataFrame(
            data={
                "importance_outlier": np.abs(shap_values).mean(0),
                "outlier": X_i,
            },
            index=self.feature_columns,
        )
        feature_importance_outlier.sort_values(
            by=["importance_outlier"], ascending=True, inplace=True
        )

        results = self.get_top_bacteria_in_time(
            indices, time_current=current_time, time_delta=time_delta
        )
        feature_importance = results["feature_importance"]

        feature_importance = feature_importance.join(feature_importance_outlier)
        feature_importance["normal"] = feature_importance.apply(
            lambda x: True
            if x["feature_avg"] - x["feature_std"]
            < x["outlier"]
            < x["feature_avg"] + x["feature_std"]
            else False,
            axis=1,
        )
        feature_importance.sort_values(
            by=["importance_outlier", "importance"], ascending=False, inplace=True
        )
        feature_importance["change"] = feature_importance.apply(
            lambda row: row["feature_avg"] if row["normal"] in normal_vals else None,
            axis=1,
        )
        feature_importance = feature_importance[feature_importance["change"].notna()]

        return feature_importance

    def add_timeboxes(self, fig, time_block_ranges, indices, num_top_bacteria=5):
        """Add time boxes to the figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the time boxes will be added.
        time_block_ranges : list
            Units or time block sizes.
        indices : list
            Indices of samples that will be used to calculate the time block
            averages. E.g. healthy samples without anomalies.
        num_top_bacteria : int
            Number of top bacteria to be added to the figure per time block.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added time boxes.
        """
        current_time = time_block_ranges[0]
        box_height = self._get_axis_max_limit(fig, "y") // 3

        legend_labels = []
        first_iteration = True
        for time in time_block_ranges[1:]:
            time_delta = time - current_time
            box_width = 0.9 * time_delta
            results = self.get_top_bacteria_in_time(indices, current_time, time_delta)
            feature_importance = results["feature_importance"]
            df_time_block = results["df_time_block"]
            x = current_time + time_delta / 2
            y = df_time_block["MMI"].median()
            number_of_samples = len(df_time_block)

            if number_of_samples > 0:

                ratios = feature_importance["importance"].values[-num_top_bacteria:]
                bacteria_names = feature_importance.index.values[-num_top_bacteria:]
                ratios /= sum(ratios)
                if x is not None:
                    y -= box_height / 2
                    x -= box_width / 2
                    y = max(y, 0)
                    # x = max(x, 0)

                    for j in range(num_top_bacteria):
                        bacteria_name = bacteria_names[j]
                        # text mean and std for this bacteria inside one boxplot
                        bacteria_avg = feature_importance["feature_avg"][bacteria_name]
                        bacteria_std = feature_importance["feature_std"][bacteria_name]

                        height = ratios[j] * box_height
                        if bacteria_name not in legend_labels:
                            legend_labels.append(bacteria_name)
                            showlegend = True
                        else:
                            showlegend = False

                        bar_trace = dict(
                            x=[x],
                            y=[height],
                            base=y,
                            offset=0,
                            width=box_width,
                            marker_color=f"rgba({self.get_bacteria_colors(bacteria_name)},0.8)",
                            legendgroup=bacteria_name,
                            showlegend=showlegend,
                            name=self.dataset.nice_name(bacteria_name),
                            hovertemplate="<br>".join(
                                [
                                    f"<b>bacteria: {self.dataset.nice_name(bacteria_name)}</b>",
                                    f"avg ± std: {bacteria_avg:.5f} ± {bacteria_std:.5f}",
                                    f"importance: {ratios[j]*100:.2f}%",
                                    f"# samples: {number_of_samples}",
                                ]
                            ),
                        )
                        if first_iteration:
                            bar_trace = {
                                "legendgroup": "<b>Important bacteria</b>",
                                "legendgrouptitle_text": "<b>Important bacteria</b>",
                                **bar_trace,
                            }
                            first_iteration = False
                        fig.add_trace(go.Bar(bar_trace))

                        y += height

            current_time += time_delta

        return fig

    def add_intervention(self, fig, time_block_ranges, indices):
        """Add intervention to the figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the intervention will be added.
        time_block_ranges : list
            Units or time block sizes.
        indices : list
            Indices of samples that will be used to calculate the time block
            averages. E.g. healthy samples without anomalies.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added intervention.
        """
        number_of_changes = 5

        # click anomaly to intervene
        # our custom event handler
        def _update_trace(trace, points, selector):
            results = self.selection_update_trace(
                fig, time_block_ranges, number_of_changes, indices
            )(trace, points, selector)

        # we need to add the on_click event to each trace separately
        for i in range(len(fig.data)):
            fig.data[i].on_click(_update_trace)

        return fig

    def selection_update_trace(
        self, fig, time_block_ranges, number_of_changes, indices
    ):
        # this list stores the points which were clicked on
        # in all but one trace they are empty
        # if len(points.point_inds) == 0:
        #     return
        if isinstance(fig, dict):
            fig = go.FigureWidget(fig)

        # traces = len(fig.data)

        def _inner(trace, points, selector):
            ret_val = ""
            config = None

            if trace["name"] == "Samples":
                try:
                    if hasattr(points, "point_inds"):
                        idx = points.point_inds[0]
                    else:
                        idx = points
                except:
                    pass

                y = trace["x"]
                y_pred = trace["y"]
                subject_ids = [x[0] for x in list(trace["customdata"])]
                sample_ids = [x[1] for x in list(trace["customdata"])]
                X = pd.DataFrame(
                    [x[2] for x in list(trace["customdata"])],
                    columns=self.feature_columns,
                )

                X_i = X.iloc[idx]
                y_i = y[idx]

                feature_importance = self.get_intervention_point(
                    X_i,
                    y_i,
                    time_block_ranges,
                    indices,
                )

                ret_val = "<b>Intervention bacteria "
                ret_val += f"for Subject {subject_ids[idx]}, Sample {sample_ids[idx]}</b><br><br>"
                ret_val += "<br>".join(
                    [
                        f"<b>{self.dataset.nice_name(x[0])}</b>: {x[1]:.5f} → {x[2]:.5f}"
                        for x in feature_importance[["outlier", "change"]]
                        .reset_index()
                        .values[:number_of_changes]
                    ]
                )

                change = feature_importance.iloc[:number_of_changes]["change"].to_dict()
                for feature in change:
                    X_i[feature] = change[feature]
                X_i = X_i.values.reshape(1, -1)
                sample_pred_new = self.estimator.predict(X_i)

                sample_ids_new = np.append(sample_ids, sample_ids[idx])
                if len(sample_ids_new) == 1 + len(set(sample_ids_new)):
                    y_new = np.append(y, y[idx])
                    y_pred_new = np.append(y_pred, sample_pred_new[0])
                    subject_ids_new = np.append(subject_ids, subject_ids[idx])
                    X_all = np.append(X.values, X_i, axis=0)

                    colors = [trace["marker"]["line"]["color"]] * len(y) + ["gray"]
                    widths = [trace["marker"]["line"]["width"]] * len(y) + [5]
                    sizes = [trace["marker"]["size"]] * len(y) + [20]
                else:
                    y_new = np.append(y[:-1], y[idx])
                    y_pred_new = np.append(y_pred[:-1], sample_pred_new[0])
                    subject_ids_new = np.append(subject_ids[:-1], subject_ids[idx])
                    X_all = np.append(X.values[:-1], X_i, axis=0)
                    sample_ids_new = np.append(sample_ids[:-1], sample_ids[idx])

                    colors = trace["marker"]["line"]["color"]
                    sizes = trace["marker"]["size"]

                    widths = trace["marker"]["line"]["width"]
                    fig.data = fig.data[:-1]
                customdata = [
                    (a, b, x) for a, b, x in zip(subject_ids_new, sample_ids_new, X_all)
                ]

                with fig.batch_update():
                    trace["marker"]["line"]["color"] = colors
                    trace["marker"]["size"] = sizes

                    trace["marker"]["line"]["width"] = widths
                    trace["marker"]["line"]["color"] = "black"
                    trace["customdata"] = customdata

                    trace["x"] = y_new
                    trace["y"] = y_pred_new

                    y_pred_delta = y_pred_new[-1] - y_pred[idx]  # * fig.layout.height
                    print(y_pred_delta)

                    fig.add_trace(
                        go.Scatter(
                            x=[y_new[-1], y_new[-1]],
                            y=[y_pred[idx], y_pred_new[-1]],
                            line_color="rgba(0,0,0,1)",
                            name="Intervention",
                            hoveron="points",
                            mode="lines+markers",
                            marker=dict(size=10, line=dict(width=2)),
                            marker_symbol="circle-open",
                            line={"width": 3},
                        )
                    )

                config = {
                    "toImageButtonOptions": {
                        "format": "svg",  # one of png, svg, jpeg, webp
                        "filename": "embedding_to_latent_space",
                        "height": fig["layout"]["height"],
                        "width": fig["layout"]["width"],
                        "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
                    }
                }

            results = {
                "fig": fig,
                "trace": trace,
                "ret_val": ret_val,
                "config": config,
            }
            return results

        return _inner

    def add_longitudinal(self, fig, indices):
        """Add longitudinal analysis per SubjectID to figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the longitudinal analysis will be added.
        indices : list
            Indices of samples from which the longitudinal analysis will be
            calculated for every subjectID.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added longitudinal analysis.
        """
        default_linewidth = 3
        highlighted_linewidth_delta = 5
        number_of_figures_start = len(fig.data)

        subject_ids = self.subject_ids[indices]
        y = self.y[indices]
        y_pred = self.y_pred[indices]

        first_iteration = True
        for subject_id in natsorted(np.unique(subject_ids)):
            indices = subject_ids == subject_id
            y_subject = y[indices]
            y_pred_subject = y_pred[indices]

            scatter_trace = dict(
                x=y_subject,
                y=y_pred_subject,
                line_color="rgba(0,0,0,0.5)",
                name=subject_id,
                hoveron="points",
                mode="lines+markers",
                marker=dict(size=10, line=dict(width=2)),
                marker_symbol="circle-open",
                line={"width": default_linewidth, "dash": "dashdot"},
                visible="legendonly",
            )
            if first_iteration:
                scatter_trace = {
                    "legendgroup": "<b>Longitudinal subjects</b>",
                    "legendgrouptitle_text": "<b>Longitudinal subjects</b>",
                    **scatter_trace,
                }
                first_iteration = False
            fig.add_trace(scatter_trace)

        number_of_figures_end = len(fig.data) - number_of_figures_start
        # our custom event handler
        def _update_trace(trace, points, selector):
            # this list stores the points which were clicked on
            # in all but one trace they are empty
            if len(points.point_inds) == 0:
                return

            for i in range(number_of_figures_start, number_of_figures_end):
                fig.data[i]["line"][
                    "width"
                ] = default_linewidth + highlighted_linewidth_delta * (
                    i == points.trace_index
                )

        # we need to add the on_click event to each trace separately
        for i in range(number_of_figures_start, number_of_figures_end):
            fig.data[i].on_click(_update_trace)

        return fig

    def add_trajectory(self, fig, indices, name, color, degree, **kwargs):
        """Plot trajectory mean and intervals.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the trajectory will be added.
        indices : list
            Indices of samples from which the trajectory will be calculated.
        name : str
            Name of the trajectory.
        color : str
            Color of the trajectory.
        degree : int
            Degree of the polynomial used for the trajectory.
        **kwargs : dict
            Additional arguments for the plotly.graph_objects.Scatter object.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added trajectory.
        """
        results = self.get_prediction_and_confidence_interval(
            indices=indices,
            degree=degree,
        )

        pi_x = results["prediction_interval"]["x"]
        pi_mean = results["prediction_interval"]["mean"]
        pi_bound = results["prediction_interval"]["bound"]

        ci_x = results["confidence_interval"]["x"]
        ci_mean = results["confidence_interval"]["mean"]
        ci_bound = results["confidence_interval"]["bound"]

        # prediction interval
        fig.add_trace(
            go.Scatter(
                x=list(pi_x) + list(pi_x[::-1]),
                y=list(pi_mean - pi_bound) + list(pi_mean + pi_bound)[::-1],
                fill="toself",
                fillcolor=f"rgba({color},0.3)",
                line_color=f"rgba({color},0.5)",
                showlegend=True,
                name="95% Prediction Interval",
                hoveron="points",
                legendgroup=f"<b>{name}</b>",
                legendgrouptitle_text=f"<b>{name}</b>",
                **kwargs,
            )
        )
        # confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(ci_x) + list(ci_x[::-1]),
                y=list(ci_mean - ci_bound) + list(ci_mean + ci_bound)[::-1],
                fill="toself",
                fillcolor=f"rgba({color},0.6)",
                line_color=f"rgba({color},0.8)",
                showlegend=True,
                name="95% Confidence Interval",
                hoveron="points",
                **kwargs,
            )
        )
        # mean prediction
        fig.add_trace(
            go.Scatter(
                x=pi_x,
                y=pi_mean,
                line_color=f"rgba({color},1.0)",
                name="Trajectory mean",
                hoveron="points",
                **kwargs,
            )
        )

        return fig

    def add_samples(self, fig, indices, color, **kwargs):
        """Add samples to figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the samples will be added.
        indices : list
            Indices of samples from which the samples will be calculated.
        color : str
            Color of the samples.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added samples.
        """
        y = self.y[indices]
        y_pred = self.y_pred[indices]
        sample_ids = self.sample_ids[indices]
        subject_ids = self.subject_ids[indices]
        X = self.X.values[indices]

        # samples
        fig.add_trace(
            go.Scatter(
                x=y,
                y=y_pred,
                line_color=f"rgba({color},0.8)",
                name="Samples",
                mode="markers",
                marker=dict(size=8, line=dict(width=2, color="rgba(0.0,0.0,0.0,0.8)")),
                hoveron="points",
                customdata=[(a, b, x) for a, b, x in zip(subject_ids, sample_ids, X)],
                hovertemplate="Age: %{x:.2f}<br>MMI: %{y:.2f}"
                + "<br>Subject: %{customdata[0]}<br>Sample: %{customdata[1]}",
                **kwargs,
            )
        )
        return fig

    def add_anomalies(self, fig, color, anomaly_type, indices):
        """Add anomalies to figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to which the anomalies will be added.
        color : str
            Color of the anomalies.
        anomaly_type : str
            Type of the anomalies.
        indices : list
            Indices of samples from which the anomalies will be calculated.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with added anomalies.
        """
        anomalies = self.get_anomalies(anomaly_type=anomaly_type, indices=indices)

        anomaly_indices = anomalies == True
        y = self.y[anomaly_indices]
        y_pred = self.y_pred[anomaly_indices]
        sample_ids = self.sample_ids[anomaly_indices]
        subject_ids = self.subject_ids[anomaly_indices]

        fig.add_trace(
            go.Scatter(
                x=y,
                y=y_pred,
                line_color=f"rgba({color},0.7)",
                name="Anomaly",
                mode="markers",
                marker=dict(size=20, line=dict(width=5)),
                marker_symbol="cross-open",
                hoveron="points",
                customdata=[(a, b) for a, b in zip(subject_ids, sample_ids)],
                hovertemplate="Age: %{x:.2f}<br>MMI: %{y:.2f}"
                + "<br>Subject: %{customdata[0]}<br>Sample: %{customdata[1]}",
            )
        )

        return fig

    def _get_axis_max_limit(self, fig, axis_name):
        maxs = []
        for trace_data in fig.data:
            if len(trace_data[axis_name]) > 0:
                maxs.append(max(trace_data[axis_name]))
        return max(maxs)

    def plot_reference_trajectory(
        self,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        degree=2,
    ):
        """Plot reference samples, fit line, its CI and PI, and longitudinal data per subject.

        Parameters
        ----------
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        degree : int
            Degree of the polynomial fit.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        indices = self.reference_groups == True

        ret_val = "<b>Performance Information</b><br>"
        ret_val += (
            f"MAE: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        )
        ret_val += f"R^2: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()

        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Reference",
            color=self.color_reference,
            degree=degree,
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_reference,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "x")),
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "y")),
            **yaxis_settings_final,
        )
        fig.update_layout(
            title_text="Microbiome trajectory with reference samples",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_reference_trajectory",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_reference_groups(
        self,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        degree=2,
    ):
        """Plot the reference vs non-reference trajectory.

        Parameters
        ----------
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        degree : int
            Degree of the polynomial fit.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        assert len(self.dataset.df.reference_group.unique()) == 2, "Reference groups is available, but no samples for non-reference."
        
        ret_val = "<b>Performance Information</b><br>"
        indices = self.reference_groups == True
        ret_val += f"MAE reference: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        ret_val += (
            f"R^2 reference: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"
        )
        indices = self.reference_groups != True
        ret_val += f"MAE non-reference: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        ret_val += f"R^2 non-reference: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"

        indices = (self.reference_groups == True) | (self.reference_groups != True)
        print("indices", len(self.dataset.df), len(indices))
        if degree == 1:
            ret_val += "<b>Linear p-value (k, n)</b>:"
            pval_k, pval_n = self.get_pvalue_linear(column="reference_groups", indices=indices)
            ret_val += f"<br>Reference vs. Non-reference: ({pval_k:.3f}, {pval_n:.3f})"
        else:
            ret_val += "<b>Spline p-value</b>:"
            pval = self.get_pvalue_spline(column="reference_groups", indices=indices, degree=degree)
            ret_val += f"<br>Reference vs. Non-reference: {pval:.3f}"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()

        # plot reference trajectory
        indices = self.reference_groups == True
        
        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Reference",
            color=self.color_reference,
            degree=degree,
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_reference,
        )

        # plot non-reference trajectory
        indices = self.reference_groups != True
        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Non-reference",
            color=self.color_non_reference,
            degree=degree,
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_non_reference,
        )
        # plot longitudinal
        fig = self.add_longitudinal(fig, indices=indices)

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "x")),
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "y")),
            **yaxis_settings_final,
        )

        fig.update_layout(
            title_text="Microbiome trajectory - trajectory per reference group",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_trajectory_per_reference_group",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_groups(
        self,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        degree=2,
    ):
        """Plot the trajectory per group.

        Parameters
        ----------
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        degree : int
            Degree of the polynomial fit.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        assert len(self.dataset.df.group.unique()) >= 2, "There are no at least 2 groups to compare."
        group_vals = np.unique(self.groups)
        ret_val = "<b>Performance Information</b><br>"
        for group in group_vals:
            indices = self.groups == group
            ret_val += f"MAE {group}: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
            ret_val += f"R^2 {group}: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"
        
        comb = combinations(group_vals, 2)
        for c in list(comb):
            indices = (self.groups == c[0]) | (self.groups == c[1])
            if degree == 1:
                ret_val += "<b>Linear p-value (k, n)</b>:"
                pval_k, pval_n = self.get_pvalue_linear(
                    column="groups",
                    indices=indices,
                )
                ret_val += f"<br>{c[0]} vs. {c[1]}: ({pval_k:.3f}, {pval_n:.3f})"
            else:
                ret_val += "<b>Spline p-value</b>:"
                pval = self.get_pvalue_spline(
                    column="groups",
                    indices=indices,
                    degree=degree,
                )
                ret_val += f"<br>{c[0]} vs. {c[1]}: {pval:.3f}"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()
        for i, group in enumerate(natsorted(np.unique(self.groups))):
            indices = self.groups == group

            fig = self.add_trajectory(
                fig=fig,
                indices=indices,
                name=group,
                color=self.colors_rgb[i],
                degree=degree,
            )
            fig = self.add_samples(
                fig=fig,
                indices=indices,
                color=self.colors_rgb[i],
            )

        indices = range(len(self.dataset.df))
        fig = self.add_longitudinal(
            fig=fig,
            indices=indices,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "x")),
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "y")),
            **yaxis_settings_final,
        )

        fig.update_layout(
            title_text="Microbiome trajectory - trajectory per group",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_trajectory_per_group",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_anomalies(
        self,
        anomaly_type=AnomalyType.PREDICTION_INTERVAL,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        degree=2,
    ):
        """Plot anomalies.

        Parameters
        ----------
        anomaly_type : AnomalyType
            Type of anomaly to plot.
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        degree : int
            Degree of the polynomial fit.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        indices = self.reference_groups == True

        ret_val = "<b>Performance Information</b><br>"
        ret_val += (
            f"MAE: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        )
        ret_val += f"R^2: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()

        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Reference",
            color=self.color_reference,
            degree=degree,
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_reference,
        )

        fig = self.add_anomalies(
            fig=fig,
            color=self.color_anomaly,
            anomaly_type=anomaly_type,
            indices=indices,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "x")),
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            # range=(0.0, self._get_axis_max_limit(fig, "y")),
            **yaxis_settings_final,
        )
        fig.update_layout(
            title_text="Microbiome trajectory - trajectory per reference group",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_reference_trajectory",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_timeboxes(
        self,
        time_block_ranges,
        anomaly_type=None,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        num_top_bacteria=5,
        degree=2,
    ):
        """Plot importance time boxes for reference non-anomalous samples.

        Parameters
        ----------
        time_block_ranges : list
            List of time units to plot.
        anomaly_type : AnomalyType
            Type of anomaly to plot.
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        degree : int
            Degree of the polynomial fit.
        num_top_bacteria : int
            Number of top bacteria to plot per block.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        indices = self.reference_groups == True  # & (self.anomaly == False)
        anomaly_type = anomaly_type or self.anomaly_type

        ret_val = "<b>Performance Information</b><br>"
        ret_val += (
            f"MAE: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        )
        ret_val += f"R^2: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()

        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Reference",
            color=self.color_reference,
            degree=degree,
            hoverinfo="skip",
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_reference,
            hoverinfo="skip",
        )

        anomalies = self.get_anomalies(anomaly_type=anomaly_type, indices=indices)
        indices = np.logical_and(indices, anomalies == False)

        fig = self.add_timeboxes(
            fig,
            time_block_ranges,
            indices=indices,
            num_top_bacteria=num_top_bacteria,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            **yaxis_settings_final,
        )
        fig.update_layout(
            title_text="Microbiome trajectory - importance timeboxes",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_importance_timeboxes",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_intervention(
        self,
        time_block_ranges,
        anomaly_type=None,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        num_top_bacteria=5,
        degree=2,
    ):
        """Plot interactive sample intervention that returns samples back to the non-anomalous referene samples.

        Parameters
        ----------
        time_block_ranges : list
            List of time units to plot.
        anomaly_type : AnomalyType
            Type of anomaly to plot.
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.
        num_top_bacteria : int
            Number of top bacteria to plot per block.
        degree : int
            Degree of the polynomial fit.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the reference samples, fit line, its CI and PI, and longitudinal data per subject.
                - "ret_val" : str
                    String with the informational text on performance.
                - "config" : dict
                    Dictionary for the plotly export image options.
        """
        indices = self.reference_groups == True
        anomaly_type = anomaly_type or self.anomaly_type

        ret_val = "<b>Performance Information</b><br>"
        ret_val += (
            f"MAE: {mean_absolute_error(self.y[indices], self.y_pred[indices]):.3f}<br>"
        )
        ret_val += f"R^2: {r2_score(self.y[indices], self.y_pred[indices]):.3f}<br>"

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        fig = go.FigureWidget()

        fig = self.add_trajectory(
            fig=fig,
            indices=indices,
            name="Reference",
            color=self.color_reference,
            degree=degree,
        )
        fig = self.add_samples(
            fig=fig,
            indices=indices,
            color=self.color_reference,
        )

        anomalies = self.get_anomalies(anomaly_type=anomaly_type, indices=indices)
        intervention_indice = np.logical_and(indices, anomalies == False)

        fig = self.add_timeboxes(
            fig,
            time_block_ranges,
            indices=intervention_indice,
            num_top_bacteria=num_top_bacteria,
        )

        fig = self.add_intervention(
            fig=fig,
            time_block_ranges=time_block_ranges,
            indices=intervention_indice,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            **yaxis_settings_final,
        )
        fig.update_layout(
            title_text="Microbiome trajectory - trajectory per reference group",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "microbiome_reference_trajectory",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": ret_val, "config": config}

        return results

    def plot_animated_longitudinal_information(
        self,
        xaxis_settings=None,
        yaxis_settings=None,
        layout_settings=None,
        degree=2,
    ):
        """Plot animated longitudinal information of reference samples.

        Parameters
        ----------
        xaxis_settings : dict
            Settings for xaxis.
        yaxis_settings : dict
            Settings for yaxis.
        layout_settings : dict
            Settings for layout.

        Returns
        -------
        results : dict
            Dictionary with the following keys:
                - "fig" : plotly.graph_objects.Figure
                    Figure with the interactive longitudinal information.
        """
        indices = self.reference_groups == True

        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**self.layout_settings_default, **layout_settings}
        if xaxis_settings is None:
            xaxis_settings = {}
        if yaxis_settings is None:
            yaxis_settings = {}
        xaxis_settings_final = {**self.axis_settings_default, **xaxis_settings}
        yaxis_settings_final = {**self.axis_settings_default, **yaxis_settings}

        frames = []

        trajectory = self.add_trajectory(
            fig=go.FigureWidget(),
            indices=indices,
            name="Reference",
            color="220,220,220",
            degree=degree,
        ).data

        subject_ids = self.subject_ids[indices]
        sample_ids = self.sample_ids[indices]
        y = self.y[indices]
        y_pred = self.y_pred[indices]

        first_iteration = True
        for age_at_collection in np.unique(y):
            data = [*trajectory]
            indices_time = y <= age_at_collection

            for subject_id in natsorted(np.unique(subject_ids)):
                indices_subject = list(subject_ids[indices_time] == subject_id)

                y_subject = y[indices_time][indices_subject]
                y_pred_subject = y_pred[indices_time][indices_subject]
                sample_ids_subject = sample_ids[indices_time][indices_subject]

                scatter_trace = dict(
                    x=y_subject,
                    y=y_pred_subject,
                    line_color=f"rgba({self.get_subject_colors(subject_id)},0.9)",
                    name=subject_id,
                    mode="lines+markers",
                    marker=dict(size=10, line=dict(width=3)),
                    marker_symbol="circle-open",
                    hoveron="points",
                    customdata=sample_ids_subject,
                    hovertemplate="Age: %{x:.2f}<br>MMI: %{y:.2f}"
                    + f"<br>Subject: {subject_id}<br>"
                    + "Sample: %{customdata}",
                )
                if first_iteration:
                    scatter_trace = {
                        "legendgroup": "<b>Longitudinal subjects</b>",
                        "legendgrouptitle_text": "<b>Longitudinal subjects</b>",
                        **scatter_trace,
                    }
                    first_iteration = False
                data.append(scatter_trace)
            frames.append(go.Frame(data=data))

        fig = go.Figure(
            data=frames[0]["data"],
            layout=go.Layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[dict(label="Play", method="animate", args=[None])],
                    )
                ]
            ),
            frames=frames,
        )

        fig.update_xaxes(
            title=f"Age at collection [{self.dataset.time_unit.name}]",
            **xaxis_settings_final,
        )
        fig.update_yaxes(
            title=f"Microbiome Maturation Index [{self.dataset.time_unit.name}]",
            **yaxis_settings_final,
        )
        fig.update_layout(
            title_text="Animated longitudinal information",
            **layout_settings_final,
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "animation_longitudinal_information",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "ret_val": None, "config": config}

        return results
