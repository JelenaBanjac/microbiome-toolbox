import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
import seaborn as sns
from microbiome.helpers import two_groups_analysis, df2vectors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import math
import pathlib
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import pandas as pd
import seaborn as sns
import gc


def zscore_analysis(df_all, z_col_name, hue_col, cross_limit=2, plot=True):
    df = df_all.copy()
    df["month"] = df.age_at_collection//30
    
    outliers_above, outliers_below = [], []
    
    df_mean_std = df[["month", z_col_name]].groupby(["month"]).agg([np.mean, np.std]).reset_index()

    def fun(row):
        mean = df_mean_std[df_mean_std.month==row["month"]][[(z_col_name, 'mean')]].values[0][0]
        std  = df_mean_std[df_mean_std.month==row["month"]][[(z_col_name, 'std')]].values[0][0]
        row[f"z_{z_col_name}"] = (row[z_col_name] - mean)/std
        return row

    df = df.apply(lambda row: fun(row), axis=1)
    df = df.sort_values(by=["month"])
    
    # plot for individual subjects
    fig, ax = plt.subplots(figsize=(20,10))

    x_lim = 39
    y_lim = 6

    ax.plot(np.linspace(0, x_lim, num=x_lim), np.zeros(x_lim), c="k", lw=2)
    for i in range(1, y_lim):
        ax.plot(np.linspace(0, x_lim, num=x_lim), np.ones(x_lim)*i, c="gray", lw=2, linestyle="--")
        ax.plot(np.linspace(0, x_lim, num=x_lim), -np.ones(x_lim)*i, c="gray", lw=2, linestyle="--")

    for s in df.subjectID.unique():
        df_subj = df[df.subjectID==s]
        #df_subj = df_subj[(df_subj.z_weight_growth_pace_during_three_years.max()<=cross_limit)&(df_subj.z_weight_growth_pace_during_three_years.min()>=-cross_limit)]
        
        if -cross_limit <= df_subj[f"z_{z_col_name}"].max()<=cross_limit and \
            -cross_limit <= df_subj[f"z_{z_col_name}"].min()<=cross_limit: 
            ax.plot(df_subj.month, df_subj[f"z_{z_col_name}"], marker="o", label=s)
        else:
            if df_subj[f"z_{z_col_name}"].max() < -cross_limit or \
                df_subj[f"z_{z_col_name}"].min() < -cross_limit:
                outliers_below.append(s)
            elif df_subj[f"z_{z_col_name}"].max() > cross_limit or \
                df_subj[f"z_{z_col_name}"].min() > cross_limit:
                outliers_above.append(s)
            else:
                pass
                #print("Check this case?")
    
    ax.set_ylim((-y_lim, y_lim));ax.set_xlim((0, x_lim))
    ax.legend(ncol=12, bbox_to_anchor=(1., -.05))
    
    # plot mean+std of z-score
    fig, ax = plt.subplots(figsize=(20,10))

    x_lim = 39
    y_lim = 3

    ax.plot(np.linspace(0, x_lim, num=x_lim), np.zeros(x_lim), c="k", lw=2)
    for i in range(1, y_lim):
        ax.plot(np.linspace(0, x_lim, num=x_lim), np.ones(x_lim)*i, c="gray", lw=2, linestyle="--")
        ax.plot(np.linspace(0, x_lim, num=x_lim), -np.ones(x_lim)*i, c="gray", lw=2, linestyle="--")
    ax = sns.pointplot(x="month", y=f"z_{z_col_name}", hue=hue_col, data=df, ax=ax, capsize=.2)
    #ax.scatter(df_subj.age_at_collection, df_subj.waz_last)
    ax.set_ylim((-y_lim, y_lim));ax.set_xlim((0, x_lim));

    if plot:
        plt.show()
    plt.clf()
    del df
    gc.collect()
    return outliers_below, outliers_above


n_splits = 5
test_size = 0.5

Regressor = CatBoostRegressor
parameters = {"loss_function": "MAE",
              "random_state": 42,
              "allow_writing_files": True,
             "verbose":False}
param_grid = { 'learning_rate': [0.5, 0.1],
               #"depth": [5, 10],
               #"iterations": [100, 500, 1000]
             }

def find_best_reference_with_least_crossings(df, feature_columns, nice_name=lambda x: x, file_directory=None):
    results = pd.DataFrame({
        "bacteria_name":[], 
        "total_num_of_crossings":[],
        "total_num_of_crossings_smooth":[], 
        "MAE":[],
        "R2":[]})
    for b_ref in feature_columns:
        total_num_of_crossings1, total_num_of_crossings2, mae, r2 = plot_shap_abundances_and_ratio(df, feature_columns, b_ref, nice_name, Regressor=Regressor, parameters=parameters, n_splits=n_splits, 
                                                                                                    file_name=f"{file_directory}/{b_ref.replace(';','_')}.png", plot=False);
        results = results.append({
            "bacteria_name":b_ref, 
            "total_num_of_crossings":total_num_of_crossings1,
            "total_num_of_crossings_smooth":total_num_of_crossings2,
            "MAE":mae,
            "r2":r2
        }, ignore_index=True)
        print(b_ref, "-->", total_num_of_crossings1, total_num_of_crossings2)
    
    file_name = f"{file_directory}/crossings.xls" or "crossings.xls"
    results.to_csv(file_name, sep="\t", index=False)
    plt.clf()
    del df
    gc.collect()
    return results
    

def plot_shap_abundances_and_ratio(df, important_features, bacteria_name, nice_name, file_name, Regressor=Regressor, parameters=parameters, n_splits=n_splits, plot=False, patent=False, time_unit_size=30, time_unit_name="months"):
    """ Plot the shap values and abundances and ratios

    Plot that is used when working with ratios. We wanted to see plots side-by-side that represent the abundance of bacteria. On the other plot
    the ratio of these two bacterias. We wanted to understand better this mapping of abundances to ratios. At the end we concluded that we want 
    to use the bacteria in the numerator that has the least number of crossings when we look at the 2 abundances.
    
    color1 = other abundance
    color2 = abundance reference
    color3 = crossing
    color4 = zero line
    colorbg = bg color
    """
    if not patent:
        color1 = "green"
        color2 = "blue"
        color3 = "red"
        color4 = "orange"
        color5 = "gray"
        colorbg = "white"
    else:
        color1 = "black"
        color2 = "white"
        color3 = "black"
        color4 = "white"
        color5 = "gray"
        colorbg = "lightgray"
    
    train = df.copy()
    
    sns.set_style("whitegrid")
         

    train[time_unit_name] = train["age_at_collection"]//time_unit_size

    plt.rc('axes', labelsize= 14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize= 8)
    plt.rc('legend', fontsize= 13)
    num_ticks = len(train[time_unit_name].unique())//2+1

    fig, axs = plt.subplots(len(important_features), 4, figsize=(40,len(important_features)*7))
    
    train_bacteria_name = train[[f"{bacteria_name}", time_unit_name]].groupby(time_unit_name).agg(np.mean).reset_index().sort_values(by=time_unit_name)
    months              = train_bacteria_name[time_unit_name].values
    total_num_of_crossings1, total_num_of_crossings2 = 0, 0
    for i in range(len(important_features)):
        
        name = important_features[i]
        
        # skip if numerator and denominator are same bacteria
        if name != bacteria_name:
            
            train_name          = train[[f"{name}", time_unit_name]].groupby(time_unit_name).agg(np.mean).reset_index()
            
            diff_ref_and_taxa1   = train_bacteria_name[f"{bacteria_name}"].values.reshape(-1) - train_name[f"{name}"].values.reshape(-1)
            diff_ref_and_taxa1   = -diff_ref_and_taxa1 if diff_ref_and_taxa1[0]<0 else diff_ref_and_taxa1
            signchange1 =  ((np.roll(np.sign(diff_ref_and_taxa1), 1) - np.sign(diff_ref_and_taxa1)) != 0).astype(int) 
            signchange1[0] = 0
            idx_cross1 = np.where(signchange1==1)[0]

            # plot abundances
            sns.pointplot(x=train[time_unit_name], y=train[f"{name}"].values, capsize=.2, alpha=0.4, ax=axs[i][0], color=color1, label=nice_name(name), markers="D")
            sns.pointplot(x=train[time_unit_name], y=train[f"{bacteria_name}"].values, capsize=.2, alpha=0.4, ax=axs[i][0], color=color2, label=f"{nice_name(bacteria_name)} (reference)", markers="*", size=220)
            axs[i][0].plot(months[idx_cross1], train_name[f"{name}"].values.reshape(-1)[idx_cross1], marker="X", markersize=15, color=color3, lw=0, alpha=1.0, label="crossed")
            axs[i][0].set_ylabel("Abundance")
            axs[i][0].set_xlabel(f"Age at collection [{time_unit_name}]")
            axs[i][0].set(facecolor=colorbg)
            axs[i][0].xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
            
            x = np.arange(0, max(months)+1)
            p1, _ = np.polyfit(months, train_name[f"{name}"].values, 5, cov=True)    
            y1 = np.polyval(p1, x)
            y1[(y1==0.0)|(y1<1e-10)] = 1e-10
            p2, _ = np.polyfit(months, train_bacteria_name[f"{bacteria_name}"].values, 5, cov=True)    
            y2 = np.polyval(p2, x)
            y2[(y2==0.0)|(y2<1e-10)] = 1e-10

            diff_ref_and_taxa2   = y2 - y1
            diff_ref_and_taxa2   = -diff_ref_and_taxa2 if diff_ref_and_taxa2[0]<0 else diff_ref_and_taxa2
            signchange2 =  ((np.roll(np.sign(diff_ref_and_taxa2), 1) - np.sign(diff_ref_and_taxa2)) != 0).astype(int) 
            signchange2[0] = 0
            idx_cross2 = np.where(signchange2==1)[0]
            
            # plot smoothed abundances
            sns.pointplot(x=x, y=y1, capsize=.2, alpha=0.4, ax=axs[i][1], color=color1, label=nice_name(name), markers="D")
            sns.pointplot(x=x, y=y2, capsize=.2, alpha=0.4, ax=axs[i][1], color=color2, label=f"{nice_name(bacteria_name)} (reference)", markers="*", size=220)
            axs[i][1].plot(x[idx_cross2]-0.5, y1[idx_cross2], marker="X", markersize=15, color=color3, lw=0, alpha=1.0, label="crossed")
            axs[i][1].set_ylabel("Abundance")
            axs[i][1].set_xlabel(f"Age at collection [{time_unit_name}]")
            axs[i][1].set(facecolor=colorbg)
            axs[i][1].xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
            
            # plot log-raios of the 2 bacteria abundances
            sns.pointplot(x=train[time_unit_name], y=[math.log2(x/y) for (x,y) in zip(train[f"{name}"].values, train[f"{bacteria_name}"].values)], capsize=.2, alpha=0.4, ax=axs[i][2], color=color5)
            axs[i][2].plot(np.linspace(0, 38, 10), np.zeros(10), lw=10, alpha=0.5, color=color4)
            axs[i][2].plot(months[idx_cross1]-0.5, np.zeros(len(idx_cross1)), alpha=0.7, marker="X", markersize=15, color=color3, lw=0)
            axs[i][2].set_ylabel("Log-Ratio")
            axs[i][2].set_xlabel(f"Age at collection [{time_unit_name}]")
            axs[i][2].set(facecolor=colorbg)
            axs[i][2].xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
            
            # plot smoothed log-ratio of the 2 bacteria abundances
            #print([(k, j) for (k,j) in zip(y1, y2)])
            sns.pointplot(x=x, y=[math.log2(k/j) for (k,j) in zip(y1, y2)], capsize=.2, alpha=0.4, ax=axs[i][3], color=color5)
            axs[i][3].plot(np.linspace(0, 38, 10), np.zeros(10), lw=10, alpha=0.5, color=color4)
            axs[i][3].plot(x[idx_cross2]-0.5, np.zeros(len(idx_cross2)), alpha=0.7, marker="X", markersize=15, color=color3, lw=0)
            axs[i][3].set_ylabel("Log-Ratio")
            axs[i][3].set_xlabel(f"Age at collection [{time_unit_name}]")
            axs[i][3].set(facecolor=colorbg)
            axs[i][3].xaxis.set_major_locator(plt.MaxNLocator(num_ticks))

            custom_lines = [Line2D([0], [0], color=color1, ls="-", marker="o", lw=4),
                            Line2D([0], [0], color=color2, ls="-", marker="o", lw=4),
                            Line2D([0], [0], color=color3, marker="X", markersize=10, ls="--", lw=0),
                            Line2D([0], [0], color=color4, ls="-", alpha=0.4, lw=10)]
            axs[i][3].legend(custom_lines, [nice_name(important_features[i]), 
                                            f"{nice_name(bacteria_name)} (reference)", 
                                            "crossed", 
                                            "zero log-ratio"], loc="upper left", bbox_to_anchor=(1, 1))

            total_num_of_crossings1 += len(idx_cross1)
            total_num_of_crossings2 += len(idx_cross2)
   
    train[f"abundance_{bacteria_name}"] = train[bacteria_name].copy()
    for c in set(important_features):
        if c != bacteria_name:
            train[f"abundance_{c}"] = train[c].copy()
            train[c] = train.apply(lambda row: math.log2(row[f"abundance_{c}"]/row[f"abundance_{bacteria_name}"]), axis=1)
            
    _X_train, _y_train = df2vectors(train, important_features)
    rfr = Regressor(**parameters)
    gkf = list(GroupKFold(n_splits=n_splits).split(_X_train, _y_train, groups=train.subjectID.values))
    search = GridSearchCV(rfr, param_grid, cv=gkf)
    search.fit(_X_train, _y_train)
    estimator = search.best_estimator_
    _y_train_pred = estimator.predict(_X_train)
    _y_train = np.array(_y_train.values)/time_unit_size
    _y_train_pred = np.array(_y_train_pred)/time_unit_size
    mae   = round(np.mean(abs(_y_train_pred - _y_train)), 2)
    r2    = r2_score(_y_train, _y_train_pred) 
    
    pathlib.Path("/".join(file_name.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name)
    plt.tight_layout()
    if plot:
        plt.show()
    plt.clf()
    plt.close()

    del df
    gc.collect()

    return total_num_of_crossings1, total_num_of_crossings2, mae, r2


from sklearn.neighbors import LocalOutlierFactor

def update_reference_group_with_novelty_detection(df_all, feature_columns, local_outlier_factor_settings=dict(metric='braycurtis', n_neighbors=2)):
    """
    Features with bacteria only and Bray-Curtis distance as a metric.

    The number of neighbors considered (parameter n_neighbors) is typically set 1) greater than the minimum number of samples a cluster has to contain, so that other samples can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by samples that can potentially be local outliers. In practice, such informations are generally not available, so we take it to be 5. Information on other algorithms used: https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-with-lof
    """
    df = df_all.copy()
    
    X_train = df[df["reference_group"]==True][feature_columns].values

    lof = LocalOutlierFactor(novelty=True, **local_outlier_factor_settings)
    lof.fit(X_train)
    
    X_test = df[df["reference_group"]==False][feature_columns].values
    y_test = lof.predict(X_test)
    
    df.loc[df["reference_group"]==False, "reference_group"] = y_test==1
    
    plt.clf()
    del df, df_all
    gc.collect()
    return df["reference_group"].values


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def gridsearch_novelty_detection_parameters(df_all, parameter_name, parameter_vals, feature_columns, metadata_columns, meta_and_feature_columns, num_runs=5, website=False, layout_settings=None):
    df = df_all.copy()
    
    df_stats = pd.DataFrame(data={"parameter":[], "run":[], "num_ref":[], "num_nonref":[], "accuracy":[], "columns_type":[]})

    # after modification
    df["reference_group"].value_counts()
    
    d = {}
    for n in parameter_vals:
        d["parameter"] = n

        settings = dict(metric='braycurtis')
        settings[parameter_name] = n

        # temp: reference_group1
        df = df.assign(reference_group1=update_reference_group_with_novelty_detection(df, meta_and_feature_columns, local_outlier_factor_settings=settings))

        # after modification
        d["num_ref"] = df["reference_group1"].value_counts()[True]
        d["num_nonref"] = df["reference_group1"].value_counts()[False]

        for i in range(num_runs):
            d["run"] = i
            output1 = two_groups_analysis(df, feature_columns, references_we_compare="reference_group1", nice_name=lambda x: x, style="dot", show=False, website=False);
            d["columns_type"] = "taxa"
            d["accuracy"] = output1["accuracy"]
            df_stats = df_stats.append(d, ignore_index=True)
            output2 = two_groups_analysis(df, metadata_columns, references_we_compare="reference_group1", nice_name=lambda x: x, style="dot", show=False, website=False);
            d["columns_type"] = "meta"
            d["accuracy"] = output2["accuracy"]
            df_stats = df_stats.append(d, ignore_index=True)
            output3 = two_groups_analysis(df, meta_and_feature_columns, references_we_compare="reference_group1", nice_name=lambda x: x, style="dot", show=False, website=False);
            d["columns_type"] = "meta and taxa"
            d["accuracy"] = output3["accuracy"]
            df_stats = df_stats.append(d, ignore_index=True)

    num_cols = 3
    fig = make_subplots(rows=1, cols=num_cols, horizontal_spacing=0.1, subplot_titles=df_stats.columns_type.unique())

    layout_settings_default = dict(
        height=400, 
        width=1100, 
        paper_bgcolor="white",
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0, r=0, b=0, pad=0),
        title_text="Novelty Algorithm Performance measure for different value of parameter",
        font=dict(size=10),
        yaxis=dict(position=0.0)
    )

    if layout_settings is None:
        layout_settings = {}
    layout_settings_final = {**layout_settings_default, **layout_settings}
    
    for idx, ct in enumerate(df_stats.columns_type.unique()):
        df_stats1 = df_stats[df_stats.columns_type==ct]
        fig.add_trace(go.Scatter(
                x=df_stats1.groupby(by="parameter").agg(np.median)["accuracy"].index,
                y=df_stats1.groupby(by="parameter").agg(np.median)["accuracy"],
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=df_stats1.groupby(by="parameter").agg(np.std)["accuracy"],
                    visible=True), name=ct
            ), row=1, col=idx%num_cols+1)
        fig.update_xaxes(title="parameter", row=1, col=idx%num_cols+1)  # gridcolor='lightgrey'
        fig.update_yaxes(title="accuracy", row=1, col=idx%num_cols+1)  # gridcolor='lightgrey'

    fig.update_layout()
    if not website:
        fig.show()

    plt.clf()
    del df, df_all
    gc.collect()
        
    return df_stats, fig