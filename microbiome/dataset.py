import math
import os
import pickle
import itertools

import numpy as np
import pandas as pd
from microbiome.enumerations import Normalization, ReferenceGroup, TimeUnit, FeatureColumnsType
from collections import namedtuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ipywidgets import VBox
from sklearn.model_selection import cross_val_score, cross_val_predict
import re


class MicrobiomeDataset:

    # initial columns, necessary for the toolbox to work
    specific_columns = [
        "sampleID",
        "subjectID",
        "group",
        "age_at_collection",
        "reference_group",
    ]
    # new columns automatically created
    # Note: e.g. reference group = healthy samples
    # non reference group = non healthy samples
    # group = country
    future_columns = []

    def __init__(self, file_name=None, feature_columns=FeatureColumnsType.BACTERIA):

        if file_name == "mouse_data":
            # file_name = "https://raw.githubusercontent.com/JelenaBanjac/microbiome-toolbox/main/notebooks/Mouse_16S/INPUT_FILES/website_mousedata_default.csv"
            # TODO: remove two classification columns
            file_name = "/home/jelena/Desktop/microbiome2021/ssh/microbiome-toolbox/notebooks/Mouse_16S/INPUT_FILES/website_mousedata.xls"
        elif file_name == "human_data":
            # file_name = "INPUT_FILES/website_humandata.xls"
            file_name = "/home/jelena/Desktop/microbiome2021/ssh/microbiome-toolbox/notebooks/Human_Subramanian/INPUT_FILES/subramanian_et_al_l2_ELM_website.csv"

        # create dataframe regardless of delimiter (sep)
        self.df = pd.read_csv(file_name, sep=None, engine="python")

        # collect all columns that are missing
        missing_columns = []

        for col in self.specific_columns:
            if col not in self.df.columns:
                if col == "reference_group":
                    # if this column doesn't exist, create it s.t.
                    # all samples are in reference by default
                    self.df["reference_group"] = True
                else:
                    missing_columns.append(col)

        # check if there is at least one column with bacteria
        bacteria_columns = self.df.columns[
            (~self.df.columns.isin(self.specific_columns))
            & (~self.df.columns.isin(self.future_columns))
            & (~self.df.columns.str.startswith("meta_"))
            & (~self.df.columns.str.startswith("id_"))
        ].tolist()
        if len(bacteria_columns) > 0:
            # if there are bacteria columns, make sure these columns have `bacteria_` prefix
            self.df = self.df.rename(
                mapper={
                    k: f"bacteria_{k}"
                    for k in bacteria_columns
                    if not k.startswith("bacteria_")
                },
                axis=1,
            )
            self.df = self.df.apply(lambda row: self._fix_zeros(row), axis=1)

        else:
            missing_columns.append("bacteria_*")

        if len(missing_columns) > 0:
            raise Exception(
                "There was an error processing this file! Missing columns: "
                + ",".join(missing_columns)
            )

        # preprocess df
        self.df.sampleID = self.df.sampleID.astype(str)
        self.df.subjectID = self.df.subjectID.astype(str)
        self.df.reference_group = self.df.reference_group.fillna(False).astype(bool)

        self.df = self.df.convert_dtypes()

        # convert metadata string columns to categorical
        for column in self.metadata_columns:
            self.df[column] = self.df[column].astype("category")

        # dummy encoding of metadata columns (string and object type)
        self.df = pd.get_dummies(self.df, dummy_na=True)
        self.df = self.df.fillna(0)
        self.df = self.df.sort_values(by="age_at_collection", ignore_index=True)

        # initialize trajectory values
        self.df["MMI"] = 0

        # initialize feature columns (only once, in constructor!)
        if isinstance(feature_columns, (list, np.ndarray)):
            self._feature_columns = feature_columns
        elif feature_columns == FeatureColumnsType.BACTERIA:
            self._feature_columns = self.bacteria_columns
        elif feature_columns == FeatureColumnsType.METADATA:
            self._feature_columns = self.metadata_columns
        elif feature_columns == FeatureColumnsType.BACTERIA_AND_METADATA:
            self._feature_columns = self.bacteria_and_metadata_columns

        # save initial
        self.__age_at_collection = self.df.age_at_collection.to_numpy()
        self.__reference_group = self.df.reference_group.to_numpy()
        self.__df = self.df.copy(deep=True)

        # normalization by log-ratio
        self.log_ratio_bacteria = None
        # normalization by mean and std (along columns)
        self.normalized = Normalization.NON_NORMALIZED

        self.time_unit = TimeUnit.DAY
        self.reference_group_choice = ReferenceGroup.USER_DEFINED
        self.nice_name = (
            lambda x: re.sub(" +", "|", re.sub("[kpcofgs]__|\.|_", " ", x[9:]).strip())
            if x.startswith("bacteria_")
            else x
        )

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
                # traceorder='normal',
            ),
            # annotations=[go.layout.Annotation(
            #     text=ret_val,
            #     align='left',
            #     showarrow=False,
            #     xref='paper',
            #     yref='paper',
            #     x=1.53,
            #     y=1,
            #     width=330,
            #     bordercolor='black',
            #     bgcolor='white',
            #     borderwidth=0.5,
            #     borderpad=8,
            # )]
        )

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
        )


    def __str__(self):
        ret_val = ""

        # dataframe size
        ret_val += "### Dataset size\n"
        ret_val += f"#samples = {self.df.shape[0]}, #columns = {self.df.shape[1]}\n"

        # counts per column type
        ret_val += "### Counts per column type\n"
        ret_val += f"Number of bacteria columns: {len(self.bacteria_columns)}\n"
        ret_val += f"Number of metadata columns: {len(self.metadata_columns)}\n"
        ret_val += f"Number of ID columns: {len(self.id_columns)}\n"
        ret_val += f"Number of other columns: {len(self.other_columns)}\n"

        # count reference
        ret_val += "### Reference group\n"
        ref_counts = self.df.reference_group.value_counts()
        ret_val += f"Reference group count vs. non-reference: {ref_counts[True]} vs. {ref_counts[False]}\n"

        return ret_val

    def _fix_zeros(self, row):
        for col in self.bacteria_columns:
            try:
                row[col] = 1e-10 if row[col] == 0.0 or row[col] < 1e-10 else row[col]
            except TypeError as e:
                raise Exception(
                    f"Check yout bacteria columns, {col[9:]} doesn't have a numerical values."
                ) from e
        return row

    @property
    def time_unit(self):
        return self._time_unit

    @time_unit.setter
    def time_unit(self, val):
        """Modify age_at_collection column"""
        if val not in TimeUnit:
            raise ValueError(f"There is no value {val} in TimeUnit enumeration class!")
        self._time_unit = val
        self.df.age_at_collection = self.__age_at_collection / val.value
        self.df.MMI = self.df.MMI / val.value

    @property
    def nice_name(self):
        return self._nice_name

    @nice_name.setter
    def nice_name(self, val):
        self._nice_name = val

    @property
    def reference_group_choice(self):
        return self._reference_group_choice

    @reference_group_choice.setter
    def reference_group_choice(self, val):
        """Modify reference_group column"""
        if val not in ReferenceGroup:
            raise ValueError(
                f"There is no value {val} in ReferenceGroup enumeration calss!"
            )
        self._reference_group_choice = val

        self.df.reference_group = self.__reference_group
        # TODO: feature_columns can be 3 cases
        X = self.df[self.feature_columns].values
        y = self.df.reference_group.values
        groups = self.df.subjectID.values

        if val == ReferenceGroup.NOVELTY_DETECTION:
            reference_group = self._find_best_novelty_reference_group(X, y, groups)
        elif val == ReferenceGroup.USER_DEFINED:
            reference_group = self.__reference_group
        else:
            raise NotImplementedError(f"Not implemented yet {val}, {type(val)}!")

        self.df.reference_group = reference_group

        # get latest values for y and groups
        y = self.df.reference_group.values
        # groups = self.df.subjectID.values
        results = two_groups_differentiation(X, y, groups)
        self._differentiation_score = results["f1score"]

    def _find_best_novelty_reference_group(self, X, y, groups, novelty_settings=None):

        if novelty_settings is None:
            c = Counter(groups)
            n_neighbors_min = 1
            n_neighbors_max = c.most_common(1)[0][1]
            n_neighbors_list = set(
                [
                    n_neighbors_min,
                    (n_neighbors_min + n_neighbors_max) // 2 + 1,
                    n_neighbors_max,
                ]
            )

            novelty_settings = []
            for n_neighbors in n_neighbors_list:
                novelty_settings.append({"n_neighbors": n_neighbors})

        f1score_best = 0
        settings_best = None
        reference_group_best = None
        for settings in novelty_settings:
            # initialize reference group with the given start values
            reference_group = copy.deepcopy(y)
            # novelty detection
            settings_final = {
                "metric": "braycurtis",
                "n_neighbors": 2,
                **settings,
            }

            X_train = X[reference_group == True]
            X_test = X[reference_group == False]

            # find outliers (ones that shall not be in the reference)
            lof = LocalOutlierFactor(novelty=True, **settings_final)
            lof.fit(X_train)
            y_test = lof.predict(X_test)

            # modify reference
            reference_group[reference_group == False] = y_test == 1

            # calculate new f1score
            results = two_groups_differentiation(
                X, copy.deepcopy(reference_group), groups
            )
            f1score = results["f1score"]
            print(f"Novelty settings: {settings_final}")
            print(f"Novelty f1score: {f1score}")
            print(pd.Series(reference_group).value_counts())
            if not all(reference_group):
                if f1score > f1score_best:
                    f1score_best = f1score
                    settings_best = settings
                    reference_group_best = copy.deepcopy(reference_group)
        print(f"BEST Novelty settings: {settings_best}")
        print(f"BEST Novelty f1score: {f1score_best}")
        return reference_group_best

    @property
    def differentiation_score(self):
        return self._differentiation_score

    @property
    def feature_columns(self):
        if self.log_ratio_bacteria is not None:
            feature_columns = self._feature_columns[
                self._feature_columns != self.log_ratio_bacteria
            ]
        else:
            feature_columns = self._feature_columns
        return feature_columns

    @feature_columns.setter
    def feature_columns(self, val):
        if isinstance(val, list):
            self._feature_columns = val
        elif isinstance(val, FeatureColumnsType):
            if val == FeatureColumnsType.BACTERIA:
                self._feature_columns = self.bacteria_columns
            elif val == FeatureColumnsType.METADATA:
                self._feature_columns = self.metadata_columns
            elif val == FeatureColumnsType.BACTERIA_AND_METADATA:
                self._feature_columns = self.bacteria_and_metadata_columns
        # but also update novelty detection result if it was used
        # self.reference_group_choice = self._reference_group_choice
        

    @property
    def bacteria_columns(self):
        return self.df.columns[self.df.columns.str.startswith("bacteria_")].to_numpy()

    @property
    def metadata_columns(self):
        return self.df.columns[self.df.columns.str.startswith("meta_")].to_numpy()

    @property
    def id_columns(self):
        return self.df.columns[self.df.columns.str.startswith("id_")].to_numpy()

    @property
    def other_columns(self):
        return self.df.columns[
            (~self.df.columns.str.startswith("bacteria_"))
            & (~self.df.columns.str.startswith("meta_"))
            & (~self.df.columns.str.startswith("id_"))
        ].to_numpy()

    @property
    def bacteria_and_metadata_columns(self):
        return np.concatenate((self.bacteria_columns, self.metadata_columns))

    @property
    def log_ratio_bacteria(self):
        return self._log_ratio_bacteria

    @log_ratio_bacteria.setter
    def log_ratio_bacteria(self, val):
        bacteria_columns_all = self.__df.columns[self.df.columns.str.startswith("bacteria_")].to_numpy()
        self._log_ratio_bacteria = val
        features = self.__df[bacteria_columns_all].values
        if val is not None:
            self.normalized = Normalization.NON_NORMALIZED
            
            for i, c in enumerate(bacteria_columns_all):
                self.df.loc[:, c] = features[:, i]
                if c != val:
                    self.df[c] = self.df.apply(
                        lambda row: math.log2(row[c] / row[val]), axis=1
                    )
            
            # remove reference, since these are abundances
            self.df = self.df.drop(columns=val, axis=1)
            
        else:
            
            for i, feature_column in enumerate(bacteria_columns_all):
                self.df.loc[:, feature_column] = features[:, i]
            # self.df = self.__df.copy(deep=True)

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, val):
        self._normalized = val
        features = self.__df[self.feature_columns].values
        
        if val == Normalization.NORMALIZED:
            from sklearn.preprocessing import normalize
            features = normalize(features, axis=0)
        
        for i, feature_column in enumerate(self.feature_columns):
            self.df.loc[:, feature_column] = features[:, i]


    def set_log_ratio_bacteria_with_least_crossings(self):
        def _crossings(x, y1, y2, degree=5):
            """
            To calculate crossing points between two abundance curves,
            we do the following:
            - approximate bacteria abundance over time with a curve,
            - get the indices of coordinates where two curves intersect.
            """
            f1 = np.poly1d(np.polyfit(x, y1, degree))
            f2 = np.poly1d(np.polyfit(x, y2, degree))

            x_new = np.linspace(min(x), max(x))
            # approximated bacteria abundances y1_new, y2_new
            y1_new = f1(x_new)
            y2_new = f2(x_new)

            # crossings
            idx = np.argwhere(np.diff(np.sign(y1_new - y2_new))).flatten()

            return idx

        from collections import defaultdict
        from itertools import product

        # abundance approximations
        x = self.__df.age_at_collection.tolist()

        number_of_crossings = defaultdict(lambda: 0)
        for b1, b2 in product(self.bacteria_columns, self.bacteria_columns):
            if b1 != b2:
                # bacteria abundances y1, y2
                y1 = self.__df[b1].tolist()
                y2 = self.__df[b2].tolist()
                number_of_crossings[b1] += len(_crossings(x, y1, y2, degree=5))

        # bacteria with the least number of crossings
        self.log_ratio_bacteria = min(number_of_crossings, key=number_of_crossings.get)

    def write_data(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read_data(filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                df = pickle.loads(f.read())
        else:
            df = None

        return df

    def z_score(self, column_name, sample_size=30):
        # TODO: above and below zscore values
        data = self.df[[column_name, "age_at_collection"]]
        data.age_at_collection //= sample_size
        from scipy.stats import zmap

        column_zscore = data.apply(
            lambda x: zmap(
                x[column_name],
                data[data.age_at_collection == x.age_at_collection][
                    column_name
                ].tolist(),
            )[0],
            axis=1,
        ).tolist()
        return column_zscore

    def plot_bacteria_abundances(self, number_of_columns=3, layout_settings=None):
        number_of_columns = 3
        number_of_rows = len(self.bacteria_columns) // number_of_columns + 1

        layout_settings_default = dict(
            height=number_of_rows * 200,
            width=1500,
            plot_bgcolor="rgba(255,255,255,255)",
            paper_bgcolor="rgba(255,255,255,255)",
            margin=dict(l=0, r=0, b=0, pad=0),
            title_text="Bacteria Abundances in the Dataset",
            font=dict(size=10),
            yaxis=dict(position=0.0),
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        fig = make_subplots(
            rows=number_of_rows,
            cols=number_of_columns,
            horizontal_spacing=0.1,
        )
        # y_max = self.df[self.bacteria_columns].values.max()+1
        # y_min = self.df[self.bacteria_columns].values.min()-1
        for idx, bacteria_name in enumerate(self.bacteria_columns):
            df = self.df[["age_at_collection", bacteria_name]]
            fig.add_trace(
                go.Scatter(
                    x=df.groupby(by="age_at_collection")
                    .agg(np.mean)[bacteria_name]
                    .index,
                    y=df.groupby(by="age_at_collection")
                    .agg(np.mean)[bacteria_name]
                    .values,
                    error_y=dict(
                        type="data",  # value of error bar given in data coordinates
                        array=df.groupby(by="age_at_collection")
                        .agg(np.std)[bacteria_name]
                        .values,
                        visible=True,
                    ),
                    name=self.nice_name(bacteria_name),
                    hovertemplate="Abundance: %{y:.2f}"
                    + f"<br>{self.time_unit.name}:"
                    + " %{x}",
                ),
                row=idx // number_of_columns + 1,
                col=idx % number_of_columns + 1,
            )
            fig.update_xaxes(
                title=f"Age at collection [{self.time_unit.name}]",
                row=idx // number_of_columns + 1,
                col=idx % number_of_columns + 1,
                gridcolor="lightgrey",
                showspikes=True,
                spikecolor="gray",
                zeroline=True,
                zerolinecolor="lightgrey",
            )
            fig.update_yaxes(
                title="Abundance value",
                row=idx // number_of_columns + 1,
                col=idx % number_of_columns + 1,
                # range=(y_min, y_max),
                gridcolor="lightgrey",
                showspikes=True,
                spikecolor="gray",
                zeroline=True,
                zerolinecolor="lightgrey",
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
        results = {"fig": fig, "config": config}

        return results

    def plot_bacteria_abundance_heatmaps(
        self,
        relative_values=False,
        fillna=False,
        avg_fn=np.median,
        layout_settings=None,
    ):
        layout_settings_default = dict(
            height=20 * len(self.bacteria_columns),
            width=1500,
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        # extract just the important columns for the heatmap
        df = self.df[self.bacteria_columns + ["subjectID", "age_at_collection"]]
        # replace long bacteria names with nice names
        df = df.rename({b: self.nice_name(b) for b in self.bacteria_columns}, axis=1)
        # update the bacteria_names with short names
        bacteria_names = np.array(list(map(self.nice_name, self.bacteria_columns)))

        def fill_collected(row):
            """Ïf we use bigger time units, then we need to find a median when collapsing all the samples in that time interval into one box in the heatmap"""
            val = df[df["age_at_collection"] == row["age_at_collection"]][
                row["bacteria_name"]
            ].values
            row["bacteria_value"] = avg_fn(val) if len(val) != 0 else np.nan
            return row

        x, y = np.meshgrid(bacteria_names, range(int(max(df.age_at_collection)) + 1))

        df_heatmap = pd.DataFrame(
            data={
                "bacteria_name": x.flatten(),
                "age_at_collection": y.flatten(),
                "bacteria_value": np.nan,
            }
        )
        df_heatmap = df_heatmap.sort_values(by=["bacteria_name", "age_at_collection"])
        df_heatmap = df_heatmap.fillna(0)
        df_heatmap = df_heatmap.apply(lambda row: fill_collected(row), axis=1)

        # create new column bacteria_name_cat in order to sort dataframe by bacteria importance
        df_heatmap["bacteria_name_cat"] = pd.Categorical(
            df_heatmap["bacteria_name"],
            categories=bacteria_names,  # order of bacteria is imposed by the list
            ordered=True,
        )
        df_heatmap = df_heatmap.sort_values("bacteria_name_cat")
        df_heatmap = df_heatmap[df_heatmap.age_at_collection > 0]

        if relative_values:
            X = df_heatmap.bacteria_value.values
            X = X.reshape(len(bacteria_names), -1)
            xmin = np.nanmin(X, axis=1).reshape(len(bacteria_names), 1)
            xmax = np.nanmax(X, axis=1).reshape(len(bacteria_names), 1)
            X_std = (X - xmin) / (xmax - xmin + 1e-10)
            df_heatmap.bacteria_value = X_std.flatten()

        # plot top absolute
        df_heatmap = df_heatmap[
            ["age_at_collection", "bacteria_name_cat", "bacteria_value"]
        ]
        df_heatmap_pivot = df_heatmap.pivot(
            "bacteria_name_cat", "age_at_collection", "bacteria_value"
        )

        if fillna:
            df_heatmap_pivot = df_heatmap_pivot.fillna(0)

        fig = px.imshow(
            df_heatmap_pivot.values,
            labels=dict(
                x=f"Age [{self.time_unit.name}]", y="Bacteria Name", color="Abundance"
            ),
            x=df_heatmap_pivot.columns,
            y=df_heatmap_pivot.index,
            title="Absolute abundances",
            **layout_settings_final,
        )
        fig.update_xaxes(side="bottom")

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "bacteria_abundances_heatmap",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        results = {"fig": fig, "config": config}

        return results

    def plot_ultradense_longitudinal_data(
        self,
        layout_settings=None,
        number_of_columns=6,
        number_of_bacteria=20,
        color_palette="tab20",
    ):
        number_of_rows = len(self.df.subjectID.unique()) // number_of_columns + int(
            len(self.df.subjectID.unique()) % number_of_columns > 0
        )

        # plotly settings
        layout_settings_default = dict(
            height=350 * number_of_rows,
            width=1500,
            plot_bgcolor="rgba(0,0,0,0)",
            title_text="Ultradense Longitudinal Data",
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        # limit to plot 20 bacteria
        bacteria_names = self.bacteria_columns[:number_of_bacteria]

        cmap = plt.cm.get_cmap(color_palette, len(bacteria_names))
        colors_dict = dict([(b, cmap(i)) for i, b in enumerate(bacteria_names)])

        fig = make_subplots(
            rows=number_of_rows,
            cols=number_of_columns,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.01,
        )

        for idx, infant in enumerate(self.df.subjectID.unique()):
            i, j = (
                idx // number_of_columns,
                idx % number_of_columns,
            )

            df1 = self.df.reset_index()
            df1 = df1[df1.subjectID == infant].sort_values("age_at_collection")

            if len(df1) == 1:

                for b in bacteria_names:

                    fig.add_trace(
                        go.Scatter(
                            x=list(df1.age_at_collection.values),
                            y=list(df1[b].values) * 2,
                            text=list(map(lambda x: self.nice_name(x), bacteria_names)),
                            mode="lines",
                            marker_color=f"rgba{colors_dict[b]}",
                            name=self.nice_name(b),
                            legendgroup=self.nice_name(b),
                            showlegend=True if idx == 0 else False,
                            stackgroup="one",  # define stack group
                            hovertemplate=f"Taxa: {self.nice_name(b)}"
                            + "<br>Abundance: %{y:.2f}"
                            + f"<br>{self.time_unit.name}:"
                            + " %{x}",
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

                    fig.update_xaxes(title=infant, row=i + 1, col=j + 1)

            else:
                for b in bacteria_names:
                    fig.add_trace(
                        go.Scatter(
                            x=list(df1.age_at_collection.values),
                            y=list(df1[b].values),
                            text=list(map(lambda x: self.nice_name(x), bacteria_names)),
                            mode="lines",
                            marker_color=f"rgba{colors_dict[b]}",
                            name=self.nice_name(b),
                            legendgroup=self.nice_name(b),
                            showlegend=True if idx == 0 else False,
                            stackgroup="one",
                            hovertemplate=f"Taxa: {self.nice_name(b)}"
                            + "<br>Abundance: %{y:.2f}"
                            + f"<br>{self.time_unit.name}:"
                            + " %{x}",
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

                    fig.update_xaxes(title=infant, row=i + 1, col=j + 1)

        fig.update_layout(**layout_settings_final)

        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=8, color="#000000")

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "bacteria_abundances_longitudinal",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        results = {"fig": fig, "config": config}

        return results

    def embedding_to_latent_space(
        self,
        color_column_name=None,
        embedding_model=PCA(n_components=3),
        embedding_dimension=2,
        layout_settings=None,
    ):
        fig = go.Figure()

        color_column_name = color_column_name or "group"
        subjectIDs = np.array(self.df.subjectID.tolist())
        sampleIDs = np.array(self.df.sampleID.tolist())
        X = self.df[self.feature_columns].values
        X_emb = embedding_model.fit_transform(X)

        if isinstance(embedding_model, PCA):
            xaxis_label = f"PC1 - {embedding_model.explained_variance_ratio_[0]*100:.1f}% explained variance"
            yaxis_label = f"PC2 - {embedding_model.explained_variance_ratio_[1]*100:.1f}% explained variance"
        else:
            xaxis_label = "1st dimension"
            yaxis_label = "2nd dimension"

        # plotly settings
        layout_settings_default = dict(
            height=600,
            width=600,
            barmode="stack",
            uniformtext=dict(mode="hide", minsize=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, b=0, pad=0),
            title_text=f"Embedding in {embedding_dimension}D space",
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    align="right",
                    valign="top",
                    text="colored by " + color_column_name,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                )
            ],
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        if embedding_dimension == 2:
            if color_column_name:
                for g in self.df[color_column_name].unique():
                    idx = np.where(np.array(self.df[color_column_name].values) == g)[0]
                    fig.add_trace(
                        go.Scatter(
                            x=X_emb[:, 0][idx],
                            y=X_emb[:, 1][idx],
                            name=str(g) or "NaN",
                            mode="markers",
                            text=[
                                f"<b>SampleID</b> {i}<br><b>SubjectID</b> {j}<br>"
                                for i, j in zip(sampleIDs[idx], subjectIDs[idx])
                            ],
                            hovertemplate="%{text}"
                            + f"<b>Group ({color_column_name}): {g}</b><br>"
                            + "<b>x</b>: %{x:.2f}<br>"
                            + "<b>y</b>: %{y:.2f}<br>",
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X_emb[:, 0],
                        y=X_emb[:, 1],
                        # name=sampleIDs,
                        mode="markers",
                        text=[
                            f"<b>SampleID</b> {i}<br><b>SubjectID</b> {j}<br>"
                            for i, j in zip(sampleIDs, subjectIDs)
                        ],
                        hovertemplate="%{text}"
                        + "<b>x</b>: %{x:.2f}<br>"
                        + "<b>y</b>: %{y:.2f}<br>",
                    )
                )
        elif embedding_dimension == 3:
            if color_column_name:
                for g in self.df[color_column_name].unique():
                    idx = np.where(np.array(self.df[color_column_name].values) == g)[0]

                    fig.add_trace(
                        go.Scatter3d(
                            x=X_emb[:, 0][idx],
                            y=X_emb[:, 1][idx],
                            z=X_emb[:, 2][idx],
                            name=str(g) or "NaN",
                            mode="markers",
                            marker=dict(size=5, opacity=0.8),
                            text=[
                                f"<b>SampleID</b> {i}<br><b>SubjectID</b> {j}<br>"
                                for i, j in zip(sampleIDs[idx], subjectIDs[idx])
                            ],
                            hovertemplate="%{text}"
                            + f"<b>Group ({color_column_name}): {g}</b><br>"
                            + "<b>x</b>: %{x:.2f}<br>"
                            + "<b>y</b>: %{y:.2f}<br>",
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=X_emb[:, 0],
                        y=X_emb[:, 1],
                        z=X_emb[:, 2],
                        mode="markers",
                        marker=dict(size=5, opacity=0.9),
                        text=[
                            f"<b>SampleID</b> {i}<br><b>SubjectID</b> {j}<br>"
                            for i, j in zip(sampleIDs, subjectIDs)
                        ],
                        hovertemplate="%{text}"
                        + "<b>x</b>: %{x:.2f}<br>"
                        + "<b>y</b>: %{y:.2f}<br>",
                    )
                )
        else:
            raise NotImplemented(
                f"Dimension {embedding_dimension} not supported for visualization :)"
            )

        fig.update_xaxes(
            title=xaxis_label,
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )
        fig.update_yaxes(
            title=yaxis_label,
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )

        fig.update_layout(**layout_settings_final)

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "embedding_to_latent_space",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

        results = {"fig": fig, "config": config}

        return results

    def embeddings_interactive_selection_notebook(
        self,
        embedding_model=PCA(n_components=2),
        layout_settings=None,
        file_name=None,
    ):
        fig = go.Figure()

        subjectIDs = np.array(self.df.subjectID.tolist())
        sampleIDs = np.array(self.df.sampleID.tolist())
        X = self.df[self.feature_columns].values
        X_emb = embedding_model.fit_transform(X)

        if isinstance(embedding_model, PCA):
            xaxis_label = f"PC1 - {embedding_model.explained_variance_ratio_[0]*100:.1f}% explained variance"
            yaxis_label = f"PC2 - {embedding_model.explained_variance_ratio_[1]*100:.1f}% explained variance"
        else:
            xaxis_label = "1st dimension"
            yaxis_label = "2nd dimension"

        layout_settings_default = dict(
            height=600,
            width=900,
            barmode="stack",
            uniformtext=dict(mode="hide", minsize=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, b=0, pad=0),
            title_text=f"Embedding in 2D space",
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    align="right",
                    valign="top",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                )
            ],
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        fig.add_trace(
            go.Scatter(
                x=X_emb[:, 0],
                y=X_emb[:, 1],
                # name=sampleIDs,
                mode="markers",
                text=[
                    f"<b>SampleID</b> {i}<br><b>SubjectID</b> {j}<br>"
                    for i, j in zip(sampleIDs, subjectIDs)
                ],
                hovertemplate="%{text}"
                + "<b>x</b>: %{x:.2f}<br>"
                + "<b>y</b>: %{y:.2f}<br>",
            )
        )

        fig.update_xaxes(
            title=xaxis_label,
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )
        fig.update_yaxes(
            title=yaxis_label,
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )

        fig.update_layout(**layout_settings_final)

        f = go.FigureWidget(fig)

        # Create a table FigureWidget that updates on selection from points in the scatter plot of f
        t = go.FigureWidget(
            [
                go.Table(
                    header=dict(
                        values=["sampleID", "subjectID"],
                        fill=dict(color="#C2D4FF"),
                        align=["left"] * 5,
                    ),
                    cells=dict(
                        values=[self.df[col] for col in ["sampleID", "subjectID"]],
                        fill=dict(color="#F5F8FF"),
                        align=["left"] * 5,
                    ),
                )
            ]
        )

        def selection_fn(trace, points, selector):
            t.data[0].cells.values = [
                self.df.loc[points.point_inds][col] for col in ["sampleID", "subjectID"]
            ]

            df_selected = pd.DataFrame(
                data={
                    "sampleID": t.data[0].cells.values[0],
                    "subjectID": t.data[0].cells.values[1],
                }
            )
            df_selected.to_csv(file_name, index=False)
            if file_name:
                print("Saved to:", file_name)
            else:
                print(
                    "Selection file not saved. Specify file_name if you want to save."
                )

            plt.clf()
            # create new column called selected to use for reference analysis: True - selected, False - not selected
            self.df["selected"] = False
            self.df.loc[
                self.df["sampleID"].isin(df_selected["sampleID"]), "selected"
            ] = True
            groups = self.df.subjectID.values

            results = two_groups_differentiation(
                self.df[self.feature_columns],
                self.df["selected"],
                groups,
                nice_name=self.nice_name,
                plot=True,
            )
            # plot the result of reference analysis with feature_columns_for_reference
            results["fig"].show(config=results["config"])
            results["img_src"].show()

        f.data[0].on_selection(selection_fn)

        if "selected" in self.df.columns:
            self.df.drop(columns="selected", inplace=True)

        plt.close("all")

        # Put everything together
        return VBox((f, t))


### HELPERS for datasets?
from sklearn.metrics import confusion_matrix
from plotly.tools import mpl_to_plotly
import shap
from collections import Counter
import copy

RANDOM_STATE = 42


def two_groups_differentiation(
    X, y, groups, nice_name=lambda x: x, settings=None, plot=False, layout_settings=None
):
    if isinstance(X, pd.DataFrame):
        x_columns = X.columns
        X = X.values

    if isinstance(y, pd.Series):
        y_column = y.name
        y = y.values.astype(int)
    else:
        y = y.astype(int)

    if isinstance(groups, pd.Series):
        groups = groups.values

    if settings is None:
        settings = {}
    settings_final = {
        "n_estimators": 140,
        "max_samples": 0.8,
        "random_state": RANDOM_STATE,
        **settings,
    }

    rfc = RandomForestClassifier(**settings_final)
    f1_scores = cross_val_score(
        rfc,
        X,
        y,
        scoring="f1",
        groups=groups,
        cv=GroupShuffleSplit(random_state=RANDOM_STATE),
    )
    accuracy_scores = cross_val_score(
        rfc,
        X,
        y,
        scoring="accuracy",
        groups=groups,
        cv=GroupShuffleSplit(random_state=RANDOM_STATE),
    )

    rfc.fit(X, y)

    fig, config, img_src = None, None, None
    if plot:
        max_display = 20
        style = "dot"
        layout_settings_default = dict(
            height=1000,
            width=800,
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, b=0, pad=0),
            title_text="Classification Important Features",
        )
        if layout_settings is None:
            layout_settings = {}
        layout_settings_final = {**layout_settings_default, **layout_settings}

        shap.initjs()

        explainer = shap.TreeExplainer(rfc)
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots()

        shap.summary_plot(
            shap_values[0] if style == "dot" else shap_values,
            features=X,
            class_names=[y_column, f"non-{y_column}"],
            show=False,
            max_display=max_display,
        )

        max_limit = max(20, min(max_display, len(x_columns)))

        fig = mpl_to_plotly(fig)

        fig.update_xaxes(
            title="SHAP value (impact on model output)",
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )

        fig.update_yaxes(
            title="Features",
            tickmode="array",
            tickvals=list(range(0, max_limit)),
            ticktext=list(
                map(
                    nice_name,
                    list(
                        np.array(x_columns)[np.argsort(np.abs(shap_values[0]).mean(0))]
                    )[::-1][:max_limit][::-1],
                )
            ),
            showline=True,
            linecolor="lightgrey",
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="lightgrey",
            showspikes=True,
            spikecolor="gray",
        )
        fig.update_layout(**layout_settings_final)

        y_pred = rfc.predict(X)
        cm = confusion_matrix(y_pred, y)

        img_src = plot_confusion_matrix(
            cm, [f"non-{y_column}", y_column], "Confusion matrix"
        )

        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "embedding_interactive",
                "height": layout_settings_final["height"],
                "width": layout_settings_final["width"],
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }

    results = {
        "accuracy": np.mean(accuracy_scores),
        "f1score": np.mean(f1_scores),
        "fig": fig,
        "config": config,
        "img_src": img_src,
    }
    return results


def plot_confusion_matrix(cm, classes, title):
    # cm : confusion matrix list(list)
    # classes : name of the data list(str)
    # title : title for the heatmap

    data = go.Heatmap(
        z=cm,
        y=classes,
        x=classes,
        colorscale=["#ffffff", "#F52757"],
    )

    annotations = []

    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fmt = ".1%"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        annotations.append(
            {
                "x": classes[i],
                "y": classes[j],
                "font": {
                    "color": "white" if cm[i, j] > thresh else "black",
                    "size": 20,
                },
                "text": format(cm[i, j], fmt),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            }
        )

    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
        "height": 500,
        "width": 500,
    }
    fig = go.Figure(data=data, layout=layout)
    return fig
