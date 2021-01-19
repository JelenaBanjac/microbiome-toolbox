
import matplotlib.pyplot as plt 
import shap
import numpy as np 
import pandas as pd 
import itertools
import math 
import seaborn as sns
import scipy.stats as stats
import pathlib
from matplotlib.patches import Patch
import scipy as sp
from elmtoolbox.helpers import df2vectors
from matplotlib.ticker import MaxNLocator


def shap_important_bacteria_ratios_with_age(estimator, X_train, y_train, important_features, short_bacteria_name_fn, file_name):
    """Dependency plot: y-axis = shap value, x-axis = bacteria abundance/ratio, color = age"""
    ncols = 5
    fig, axs = plt.subplots(len(important_features)//ncols+1, ncols, figsize=(42,(len(important_features)//ncols+1)*5)) 

    important_features_short = list(map(lambda x: short_bacteria_name_fn(x), important_features))

    X_features_and_age = np.hstack((X_train, np.array(y_train).reshape(-1, 1)))
    idx_of_age = X_features_and_age.shape[1]-1
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_features_and_age)
    inds = shap.approximate_interactions(idx_of_age, shap_values, X_features_and_age)
    
    for i in range(len(important_features_short)):
        shap.dependence_plot(i, shap_values, X_features_and_age, feature_names=important_features_short+["Age at collection [day]"], interaction_index=inds[idx_of_age], ax=axs[i//ncols,i%ncols], show=False, dot_size=20, cmap=None)
        bact_min = np.min(X_features_and_age[:,i])
        bact_max = np.max(X_features_and_age[:,i])
        axs[i//ncols,i%ncols].plot(np.linspace(bact_min, bact_max, 10), np.zeros(10), lw=4, alpha=0.5, color="orange")
        if i%ncols == 0:
            axs[i//ncols,i%ncols].set_ylabel("SHAP value")
        else:
            axs[i//ncols,i%ncols].set_ylabel("")
        if i != 0:
            fig.axes[-1].remove()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def get_top_bacteria_in_time(estimator, df_all, top_bacteria, days_start, days_number_1unit, num_top_bacteria, average=np.median, std=np.std):  # scipy.stats.hmean
    """
    e.g.
    time_type = "year", days_number_1unit = 12*30
    """
    df = df_all.copy()
    
    if "y" not in df.columns:
        X, y = df2vectors(df, top_bacteria)
        y_pred = estimator.predict(X)
        df["y"] = y
        df["y_pred"] = y_pred

    df_time = df[(days_start<df.y)&(df.y<days_start+days_number_1unit)]
    df_time = df_time[top_bacteria+["age_at_collection", "y", "y_pred"]]
    
    xpos, bottom, bacteria_name, ratios, means, stds = None, None, None, None, None, None
    feature_importance = None
    if df_time.shape[0]!=0:
        means = pd.Series(data=dict([(c, average(df_time[c].values)) for c in df_time.columns])) #df_time.mean()
        stds  = pd.Series(data=dict([(c, std(df_time[c].values)) for c in df_time.columns])) #df_time.std()   
        medians = df_time.median()
        X, y = df2vectors(pd.DataFrame(dict(means), index=[0]), top_bacteria)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        feature_importance = pd.DataFrame(list(zip(top_bacteria, np.abs(shap_values).mean(0))), columns=['bacteria_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=True,inplace=True)

        # location of boxplots (medians)
        xpos = (days_start+days_number_1unit/2)//30 +1
        bottom = medians["y_pred"]//30 + 1
        ratios = feature_importance["feature_importance_vals"].values[-num_top_bacteria:]
        bacteria_name = feature_importance["bacteria_name"].values[-num_top_bacteria:]
        ratios /= sum(ratios)
         
        feature_importance["bacteria_avg"] = feature_importance.apply(lambda x: means[x["bacteria_name"]], axis=1)
        feature_importance["bacteria_std"] = feature_importance.apply(lambda x: stds[x["bacteria_name"]], axis=1)
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=True,inplace=True)
    return xpos, bottom, bacteria_name, ratios, means, stds, feature_importance, len(df_time)


def plot_importance_boxplots_over_age(estimator, df_all, top_bacteria, file_name, bacteria_short_fn, forpatent=False, plot_outliers=None, plot_sample=None, df_new=None, figsize=(25, 25)):
    """ Importance boxplot over age on the trajectory
    
    The most important method after discovering trajectory. It will tell you what are the bacteria that are the most important for each age interval.

    Parameters
    ----------
    estimator: sklearn model, CatBoostRegressor, etc.
        Model used to create the trajectory.    
    df_all: pd.DataFrame
        Dataset containing everything.
    top_bacteria: list
        List of the top important bacteria that are used for prediction in our model.
    file_name: str
        File name where to save the final figure with boxplots.
    bacteria_short_fn: callable
        Function for shortening the bacteria names since they are too long.
    forpatent: bool
        If it is True, it will plot black and white figure since patents are done that way.
    plot_outliers: bool
        If True, all samples crossing the prediction interval pi will be colored red and displayed as an outlier.
    plot_sample: int
        The index of the sample from the dataframe that we want to accent (bold red).
    df_new: pd.Dataframe
        Dataframe with an outlier now being healthy (after intervention). 
        In order not to recalculate the new trajectory with this "moved" outlier, we use old dataset for the trajectory, and use this dataframe 
        with new modified sample that is now considered healthy. 
    figsize: tuple
        Figure size.

    Returns
    -------
    y2: list
        Mean line of the trajectory. Used later to see how far the outliers are from this line +- pi.    
    pi: float
        Prediction interval size. It is the line surrounding the mean line of trajectory y2, and containing 95% of predcition.
    """
    sns.set_style("whitegrid")
    plt.rc('font', size=15)          
    plt.rc('axes', labelsize= 15)  
    plt.rc('xtick', labelsize=15)       
    plt.rc('ytick', labelsize=15)   
    plt.rc('legend', fontsize=15) 
    
    df = df_all.copy()

    X, y = df2vectors(df, top_bacteria)
    y_pred = estimator.predict(X)
    
    df["y"] = y
    df["y_pred"] = y_pred
    
    if df_new is not None:
        X, y = df2vectors(df_new, top_bacteria)
        y_pred = estimator.predict(X)

        df_new["y"] = y
        df_new["y_pred"] = y_pred

    colors = {}
    ypos_text_margin = 1.5
    text_fontsize_samples = 15
    str_fmt = ".5f"
    scale = 5
    latest_day = 0
    
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_ylim((-2, 13)); ax.set_xlim((-1, 13))
    ax.set_ylim((0, 36)); ax.set_xlim((0, 36))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
    ax.set_xlabel("Age [month]", fontdict={"fontsize":text_fontsize_samples});
    ax.set_ylabel("Microbiome Maturation Index [month]", fontdict={"fontsize":text_fontsize_samples})
    if forpatent:
        palette = itertools.cycle(sns.color_palette("light:gray", 19))
        traj_color = "k"
        boxplot_alpha = 1.0
        num_top_bacteria = 5
        outlier_color = "gray"
    else:
        palette = itertools.cycle(sns.color_palette("Paired").as_hex())
        traj_color = "g"
        boxplot_alpha = 0.8
        num_top_bacteria = 5
        outlier_color = "red"

    # months
    width=0.9
    days_number_1unit=30
    for i in range(7):
        xpos, bottom, bacteria_name, ratios, bacteria_means, bacteria_stds, _, samples_num = get_top_bacteria_in_time(estimator, df, top_bacteria, days_start=latest_day, days_number_1unit=days_number_1unit, num_top_bacteria=num_top_bacteria)
        if xpos is not None:
            bottom -= scale/2 
            ypos = 0

            for j in range(len(ratios)):
                # text mean and std for this bacteria inside one boxplot
                bacteria_mean = bacteria_means[bacteria_name[j]]
                bacteria_std  = bacteria_stds[bacteria_name[j]]
                bacteria_std = '' if math.isnan(bacteria_std) else f"\n±{bacteria_std:{str_fmt}}"

                if not colors.get(bacteria_name[j], None):
                    colors[bacteria_name[j]] = next(palette)


                height = ratios[j]*scale
                ax.bar(xpos, height, width, bottom=bottom, color=colors[bacteria_name[j]], alpha=boxplot_alpha)
                bottom += height
                ypos = bottom - height/2
                #ax.text(xpos, ypos, f"{bacteria_mean:{str_fmt}}{bacteria_std}", ha='center', fontdict={"fontsize":10})
                #ax.text(xpos, ypos, bacteria_short_fn(bacteria_name[j]), ha='center', fontdict={"fontsize":10})

            latest_day += days_number_1unit
            
            # number of samples above boxplot
            ax.text(xpos, ypos+ypos_text_margin, f"{samples_num}", ha='center', fontdict={"fontsize":text_fontsize_samples}) 

    # trimester
    width=3*width
    days_number_1unit=3*30
    for i in range(6):
        xpos, bottom, bacteria_name, ratios, bacteria_means, bacteria_stds, _, samples_num = get_top_bacteria_in_time(estimator, df, top_bacteria, days_start=latest_day, days_number_1unit=days_number_1unit, num_top_bacteria=num_top_bacteria)
        if xpos is not None:
            bottom -= scale/2 
            ypos = 0

            for j in range(len(ratios)):
                # text mean and std for this bacteria inside one boxplot
                bacteria_mean = bacteria_means[bacteria_name[j]]
                bacteria_std  = bacteria_stds[bacteria_name[j]]
                bacteria_std = '' if math.isnan(bacteria_std) else f"±{bacteria_std:{str_fmt}}"

                if not colors.get(bacteria_name[j], None):
                    colors[bacteria_name[j]] = next(palette)


                height = ratios[j]*scale
                ax.bar(xpos, height, width, bottom=bottom, color=colors[bacteria_name[j]], alpha=boxplot_alpha)
                bottom += height
                ypos = bottom - height/2
                #ax.text(xpos, ypos, f"{bacteria_mean:{str_fmt}}{bacteria_std}", ha='center', fontdict={"fontsize":12})
                #ax.text(xpos, ypos, bacteria_short_fn(bacteria_name[j]), ha='center', fontdict={"fontsize":12})

            latest_day += days_number_1unit
            # number of samples above boxplot
            ax.text(xpos, ypos+ypos_text_margin, f"{samples_num}", ha='center', fontdict={"fontsize":text_fontsize_samples}) 

        
    # year
    width=3.5*width
    days_number_1unit=12*30
    for i in range(1):
        xpos, bottom, bacteria_name, ratios, bacteria_means, bacteria_stds, _, samples_num = get_top_bacteria_in_time(estimator, df, top_bacteria, days_start=latest_day, days_number_1unit=days_number_1unit, num_top_bacteria=num_top_bacteria)
        if xpos is not None:
            bottom -= scale/2 
            ypos = 0

            for j in range(len(ratios)):
                # text mean and std for this bacteria inside one boxplot
                bacteria_mean = bacteria_means[bacteria_name[j]]
                bacteria_std  = bacteria_stds[bacteria_name[j]]
                bacteria_std = '' if math.isnan(bacteria_std) else f"±{bacteria_std:{str_fmt}}"
                
                if not colors.get(bacteria_name[j], None):
                    colors[bacteria_name[j]] = next(palette)


                height = ratios[j]*scale
                ax.bar(xpos, height, width, bottom=bottom, color=colors[bacteria_name[j]], alpha=boxplot_alpha)
                bottom += height
                ypos = bottom - height/2
                #ax.text(xpos, ypos, f"{bacteria_mean:{str_fmt}}{bacteria_std}", ha='center', fontdict={"fontsize":13})
                #ax.text(xpos, ypos, bacteria_short_fn(bacteria_name[j]), ha='center', fontdict={"fontsize":13})

            latest_day += days_number_1unit
            # number of samples above boxplot
            ax.text(xpos, ypos+ypos_text_margin, f"{samples_num}", ha='center', fontdict={"fontsize":text_fontsize_samples}) 

    
    # prediction interval with the mean prediction
    yr_limit=2
    degree=2
    nboot=None
    y = df["y"]/30  # in months, x-axis
    y_pred = df["y_pred"]/30 # in months, y-axis

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

    # Data
    if df_new is not None:
        sns.scatterplot(x=df_new["y"]/30, y=df_new["y_pred"]/30, ax=ax, label="Healthy samples", color=traj_color)  # , color="#b9cfe7" marker="o", size=5
    else:
        sns.scatterplot(x=df["y"]/30, y=df["y_pred"]/30, ax=ax, label="Healthy samples", color=traj_color)  # , color="#b9cfe7" marker="o", size=50

    x2 = np.linspace(0, 40, 40) #np.linspace(np.min(x), np.max(x), 100)
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
    
    _x2 = x2.copy() 
    _y  = y.values.copy()
    _y_pred = y_pred.values.copy()
    if plot_outliers:
        # plot outliers
        for i in range(len(_y)):
            idx    = int(x2[int(_y[i])])
            lowlim = (y2-pi)[idx]+0.2
            uplim  = (y2+pi)[idx] #-0.2

            if _y_pred[i]>uplim:
                ax.text(_y[i], _y_pred[i]+0.2, df.iloc[i].sampleID, ha='center', fontdict={"fontsize":12}) 
                ax.scatter(_y[i], _y_pred[i], alpha=0.8, facecolors='none', edgecolors=outlier_color, lw=3, marker="o", s=120)
            elif _y_pred[i]<lowlim:
                ax.text(_y[i], _y_pred[i]+0.2, df.iloc[i].sampleID, ha='center', fontdict={"fontsize":12}) 
                ax.scatter(_y[i], _y_pred[i], alpha=0.8, facecolors='none', edgecolors=outlier_color, lw=3, marker="o", s=120)
    
    if plot_sample is not None:
        if df_new is not None:
            _x2 = x2.copy() 
            _y  = (df_new["y"]/30).values.copy()
            _y_pred = (df_new["y_pred"]/30).values.copy()
        else:
            _x2 = x2.copy() 
            _y  = y.values.copy()
            _y_pred = y_pred.values.copy()
        ax.text(_y[plot_sample], _y_pred[plot_sample]+0.3, df.iloc[plot_sample].sampleID, ha='center', fontdict={"fontsize":12}) 
        ax.plot(_y[plot_sample], _y_pred[plot_sample], alpha=1, markerfacecolor=traj_color, markeredgecolor=outlier_color, markeredgewidth=6, marker="o", markersize=14)
        
    custom_patches = []
    custom_names = []
    for k, v in colors.items():
        custom_patches.append(Patch(facecolor=v, edgecolor=v, label=k))
        custom_names.append(bacteria_short_fn(k))
    plt.legend(custom_patches, custom_names, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    pathlib.Path('/'.join(file_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name)
    plt.xlim(0, latest_day//30+1);plt.ylim(0, latest_day//30+1)
    plt.show()
    
    return  y2, pi