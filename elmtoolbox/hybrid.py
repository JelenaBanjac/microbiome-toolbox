import numpy as np
from elmtoolbox.helpers import df2vectors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.stats as stats
import scipy as sp
from matplotlib.ticker import MaxNLocator
import pandas as pd
from elmtoolbox.statistical_analysis import *


def plot_trajectory(estimator, df, bacteria_names, ax, limit_age, time_unit_size, time_unit_name, traj_color, traj_label, yr_limit, limit_age_max):
    degree = 2
    nboot = None 
    
    X, y = df2vectors(df, bacteria_names)
    y_pred = estimator.predict(X)
    
    y = np.array(y)/time_unit_size
    y_pred = np.array(y_pred)/time_unit_size
    
    mae   = round(np.mean(abs(y_pred - y)), 2)
    r2    = r2_score(y, y_pred)
    coeff = stats.pearsonr(y_pred, y)
    idx       = np.where(y < yr_limit*365/time_unit_size)[0]
    mae_idx   = round(np.mean(abs(y_pred[idx] - y[idx])), 2)
    r2_idx    = r2_score(y[idx], y_pred[idx])
    coeff_idx = stats.pearsonr(y_pred[idx], y[idx])

    print("\n", "-"*5, "Performance Information - ", traj_label, "-"*5)
    print(f'MAE: {mae} {time_unit_name}')
    print(f'R^2: {r2:.3f}')
    print(f"Pearson correlation coeff: {coeff[0]:.3f}, 2-tailed p-value: {coeff[1]:.2e}\n")
    print("-"*5, "Performance Information", f"< {yr_limit}yr", "-"*5)
    print(f'MAE: {mae_idx} {time_unit_name}')
    print(f'R^2: {r2_idx:.3f}')
    print(f"Pearson correlation coeff: {coeff_idx[0]:.3f}, 2-tailed p-value: {coeff_idx[1]:.2e}\n")

    # Plot data
    equation = lambda a, b: np.polyval(a, b) 

    # Plotting --------------------------------------------------------------------
    sns.scatterplot(x=y, y=y_pred, ax=ax, label="Healthy samples", color=traj_color)

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
    #     sns.regplot(x=x, y=y_model, color ='red', label="Prediction fit", order=degree)
    if nboot:
        bootindex = sp.random.randint
        for _ in range(nboot):
            resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
            # Make coeffs of for polys
            pc = np.polyfit(y, y_pred + resamp_resid, degree)                   
            # Plot bootstrap cluster
            ax.plot(y, np.polyval(pc, y), "g-", linewidth=2, alpha=3.0 / float(nboot))

    x2 = np.linspace(0, limit_age_max//time_unit_size+1, limit_age_max//time_unit_size+1) #np.linspace(np.min(x), np.max(x), 100)
    y2 = equation(p, x2)

    # Fit   
    #sns.lineplot(x=x, y=y_model, color="green", style=True, linewidth=5, alpha=0.5, label="Fit", ax=ax)  
    sns.lineplot(x=x2, y=y2, color=traj_color, style=True, linewidth=5, alpha=0.5, ax=ax, label=traj_label)  

    # Prediction Interval
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(y))**2 / np.sum((y - np.mean(y))**2))   
    #sns.lineplot(x=x2, y=y2 - pi, dashes=[(2,2)], style=True, color=traj_color, label="95% Prediction Interval", ax=ax)
    #sns.lineplot(x=x2, y=y2 + pi, dashes=[(2,2)], style=True, color=traj_color, ax=ax, label=None)
    ax.fill_between(x2, y2-pi, y2+pi, alpha=0.15, color=traj_color, label=f"95% Prediction Interval")

    pi_mean       = np.mean(pi)
    pi_median     = np.median(pi)
    idx           = np.where(x2 < yr_limit*365)[0]
    pi_mean_idx   = np.mean(pi[idx])
    pi_median_idx = np.median(pi[idx])
    print("-"*5, f"Statistics", "-"*5)
    print(f"Chi^2: {chi2:.2f}")
    print(f"Reduced chi^2: {chi2_red:.2f}")
    print(f"Standard deviation of the error: {s_err:.2f}")
    print(f"Prediction interval: mean={pi_mean:.2f}, median={pi_median:.2f}")
    print(f"Prediction interval < {yr_limit}yrs : mean={pi_mean_idx:.2f}, median={pi_median_idx:.2f}")
    
    return ax


def plot_2_trajectories(estimator, estimator_all, bacteria_names, bacteria_names_all, df, degree=2, yr_limit=2, limit_age=1200, start_age=0, time_unit_size=1, time_unit_name="days", title=None, everytick=False, linear_pval=False, nonlinear_pval=False):

    X1, y1 = df2vectors(df, bacteria_names_all)
    y_pred1 = estimator_all.predict(X1)
    X2, y2 = df2vectors(df, bacteria_names)
    y_pred2 = estimator.predict(X2)

    df["y1"] = y1/time_unit_size+1
    df["y_pred1"] = y_pred1/time_unit_size+1
    df["y2"] = y2/time_unit_size+1
    df["y_pred2"] = y_pred2/time_unit_size+1

    def get_pvalue_regliner(df):
        _df = df.copy(deep=False)
       
        df_stats = pd.DataFrame(data={"Input": list(_df.y1.values) + list(_df.y2.values),
                                      "Output": list(_df.y_pred1.values) + list(_df.y_pred2.values),
                                      "Condition": ["all"]*len(_df.y1.values) + ["hybrid"]*len(_df.y2.values)})

        return regliner(df_stats, {"all": 0, "hybrid": 1})

    def get_pvalue_permuspliner(df):
        _df = df.copy(deep=False)
        subjectID1 = list(_df.subjectID.values)
        subjectID2 = list(map(lambda x: f"{x}_hybrid", subjectID1))
        
        df_stats = pd.DataFrame(data={"Input": list(_df.y1.values) + list(_df.y2.values),
                                      "Output": list(_df.y_pred1.values) + list(_df.y_pred2.values),
                                      "Condition": ["all"]*len(_df.y1.values) + ["hybrid"]*len(_df.y2.values),
                                      "SubjectID": subjectID1 + subjectID2})

        result = permuspliner(df_stats, xvar="Input", yvar="Output", category="Condition", degree = degree, cases="SubjectID", groups = ["all", "hybrid"], perms = 500, test_direction = 'more', ints = 1000, quiet = True)
    
        return result["pval"]

    limit_age_max = 1200

    #df = df[df.age_at_collection<limit_age]

    plt.rc('axes', labelsize= 15)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    fig, ax = plt.subplots(figsize=(12, 12))
    #ax.set_xlim(start_age//time_unit_size, limit_age//time_unit_size+1);ax.set_ylim(start_age//time_unit_size, limit_age//time_unit_size+1)
    if yr_limit*365//time_unit_size+1 < limit_age//time_unit_size+1:
        ax.fill_between(np.linspace(yr_limit*365, 5*365, 10)//time_unit_size+1, np.zeros(10), np.ones(10)*limit_age, alpha=0.3, color="gray", label=f"infant sample older than {yr_limit} yrs")

    ax = plot_trajectory(estimator, df, bacteria_names, ax, limit_age, time_unit_size, time_unit_name, traj_color="blue", traj_label="selected bacteria (sum)", yr_limit=yr_limit, limit_age_max=limit_age_max)
    ax = plot_trajectory(estimator_all, df, bacteria_names_all, ax, limit_age, time_unit_size, time_unit_name, traj_color="red", traj_label="all bacteria", yr_limit=yr_limit, limit_age_max=limit_age_max)

    # Labels
    if title:
        plt.title(title, fontsize="16", fontweight="bold")
    plt.xlabel(f"Age [{time_unit_name}]")
    plt.ylabel(f"Microbiome Maturation Index [{time_unit_name}]")
    plt.xlim(start_age//time_unit_size+1, limit_age//time_unit_size+1);plt.ylim(start_age//time_unit_size+1, limit_age//time_unit_size+1)

    if linear_pval:
        equation = lambda a, b: np.polyval(a, b) 
        xx = np.linspace(0, yr_limit*365, 10)/time_unit_size+1
        
        p1, _cov = np.polyfit(df.y1.values, df.y_pred1.values, 1, cov=True)  
        #sns.lineplot(x="x", y="y", data=dict(x=df.y1.values, y=equation(_p, df.y1.values)), linestyle="--", lw=4, color="red", ax=ax)
        ax.plot(xx, equation(p1, xx), ls="--", lw=4, c="red")
        
        p2, _cov = np.polyfit(df.y2.values, df.y_pred2.values, 1, cov=True)         
        #sns.lineplot(x="x", y="y", data=dict(x=df.y2.values, y=equation(_p, df.y2.values)), linestyle="--", lw=4, color="blue", ax=ax)
        ax.plot(xx, equation(p2, xx), ls="--", lw=4, c="blue")

        pval_k, pval_n = get_pvalue_regliner(df[df.age_at_collection<yr_limit*365])
        ax.text(0.1, 0.9, "p-value (k, n):\n$p = "+f"{pval_k:.3f}, {pval_n:.3f}$", 
                transform=ax.transAxes, fontsize=15, verticalalignment='center', bbox=dict(boxstyle='round', facecolor="w", edgecolor="k", alpha=0.5))
        
    if nonlinear_pval:
        pval = get_pvalue_permuspliner(df[df.age_at_collection<limit_age*365])
        ax.text(0.1, 0.9, "$p = "+f"{pval:.3f}$", 
                transform=ax.transAxes, fontsize=15, verticalalignment='center', bbox=dict(boxstyle='round', facecolor="w", edgecolor="k", alpha=0.5))

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    display = np.where(np.array(labels)!='True')[0]
    anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
    legend = plt.legend(
        [handle for i, handle in enumerate(handles) if i in display], #+ [anyArtist],
        [label for i, label in enumerate(labels) if i in display], # + ["95% Confidence Limits"],
        loc=3, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
    )  
    frame = legend.get_frame().set_edgecolor("0.5")
    if everytick:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=limit_age_max//time_unit_size+1, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=limit_age_max//time_unit_size+1, integer=True))
    plt.tight_layout()
    plt.show()
