import seaborn as sns
from microbiome.helpers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import math
import pathlib
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor
import pandas as pd

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

    return total_num_of_crossings1, total_num_of_crossings2, mae, r2