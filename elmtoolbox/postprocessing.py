
import matplotlib.pyplot as plt 
import shap
import numpy as np 
import pandas as pd 
import itertools
import seaborn as sns
import scipy.stats as stats
import math 
import pathlib
from matplotlib.patches import Patch
import scipy as sp
from elmtoolbox.helpers import df2vectors
from matplotlib.ticker import MaxNLocator
from elmtoolbox.trajectory import plot_1_trajectory
import plotly.graph_objects as go


def get_top_bacteria_in_time(estimator, df_all, top_bacteria, days_start, days_number_1unit, num_top_bacteria, average=np.median, std=np.std, time_unit_size=1, time_unit_name="days"):  # scipy.stats.hmean
    """
    e.g.
    time_type = "year", days_number_1unit = 12*30
    
    Parameters
    ----------
    num_top_bacteria: int
        Number of top important bacteria to be ploted in this time block.
    """
    df = df_all.copy()
    
    if "y" not in df.columns:
        X, y = df2vectors(df, top_bacteria)
        y_pred = estimator.predict(X)
        df["y"] = y
        df["y_pred"] = y_pred

    # from main df, extract only the time block we are interested in
    df_time_block = df[(days_start<df.y)&(df.y<days_start+days_number_1unit)]
    # and now exctract only the columns we need
    df_time_block = df_time_block[top_bacteria+["age_at_collection", "y", "y_pred"]]
    
    xpos, bottom, bacteria_name, ratios, avgs, stds = None, None, None, None, None, None
    feature_importance = None
    if df_time_block.shape[0]!=0:
        # average, std, and median of samples in the time block for a given bacteria (and age, y, y_pred)
        avgs    = pd.Series(data=dict([(c, average(df_time_block[c].values)) for c in df_time_block.columns])) 
        stds    = pd.Series(data=dict([(c, std(df_time_block[c].values)) for c in df_time_block.columns]))
         
        # for this time block average, determine what are the most important features based on SHAP
        X, y = df2vectors(pd.DataFrame(dict(avgs), index=[0]), top_bacteria)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        feature_importance = pd.DataFrame(list(zip(top_bacteria, np.abs(shap_values).mean(0))), columns=['bacteria_name','feature_importance_vals'])
        #feature_importance.sort_values(by=['feature_importance_vals'], ascending=True,inplace=True)
        # besides bacteria name and its importance, add its average and std for this time block
        feature_importance["bacteria_avg"] = feature_importance.apply(lambda x: avgs[x["bacteria_name"]], axis=1)
        feature_importance["bacteria_std"] = feature_importance.apply(lambda x: stds[x["bacteria_name"]], axis=1)
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=True,inplace=True)

        # location of boxplots (medians)
        xpos   = (days_start+days_number_1unit/2)/time_unit_size
        bottom = df_time_block.median()["y_pred"]/time_unit_size
        ratios = feature_importance["feature_importance_vals"].values[-num_top_bacteria:]
        bacteria_name = feature_importance["bacteria_name"].values[-num_top_bacteria:]
        ratios /= sum(ratios)
        
    return xpos, bottom, bacteria_name, ratios, avgs, stds, feature_importance, len(df_time_block)

def plot_importance_boxplots_over_age(estimator, df_all, top_bacteria, nice_name, units, start_age=0, limit_age=None, patent=False, highlight_outliers=None, df_new=None, time_unit_size=1, time_unit_name="days", box_height=10, file_name=None, plateau_area_start=None, longitudinal_mode="markers+lines", longitudinal_showlegend=True, fillcolor_alpha=0.3, layout_height=900, layout_width=1000):
    
    df = df_all.copy()
    
    # if there are samples after units, add the last box at the end that would fill in 
    limit_age_max = limit_age or max(df.age_at_collection.astype(int))+1
    if sum(units) < limit_age_max:
        units.append(limit_age_max-sum(units))
    if not plateau_area_start:
        plateau_area_start = limit_age_max
    
    X, y = df2vectors(df, top_bacteria)
    y_pred = estimator.predict(X)

    df["y"] = y
    df["y_pred"] = y_pred

    if df_new is not None:
        X, y = df2vectors(df_new, top_bacteria)
        y_pred = estimator.predict(X)

        df_new["y"] = y
        df_new["y_pred"] = y_pred


    if patent:
        palette = itertools.cycle(sns.color_palette("Greys", n_colors=11).as_hex())
        boxplot_alpha = 1.0
        num_top_bacteria = 5
        outlier_color = "0,0,0"
        traj_color = "0,0,0"
    else:
        palette = itertools.cycle(sns.color_palette("Paired").as_hex())
        boxplot_alpha = 0.8
        num_top_bacteria = 5
        outlier_color = "255,0,0"
        traj_color = "26,150,65" 
        
    colors = {}
    ypos_text_margin = 1.5
    text_fontsize_samples = 15
    str_fmt = ".5f"
    latest_day = start_age
    legend_vals = []

    fig = go.Figure()  
    fig, ret_val, outliers, mae, r2, pi_median, traj_pi, traj_mean = plot_1_trajectory(fig, estimator, df, top_bacteria, limit_age_max, time_unit_size, time_unit_name, traj_color=traj_color, traj_label="reference", 
                                                                   plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, nboot=None, longitudinal_mode=longitudinal_mode, 
                                                                   longitudinal_showlegend=longitudinal_showlegend, fillcolor_alpha=fillcolor_alpha, highlight_outliers=highlight_outliers,
                                                                  outlier_color=outlier_color, df_new=df_new)


    for days_number_1unit in units:

        box_width=0.9 * days_number_1unit/time_unit_size

        xpos, bottom, bacteria_name, ratios, bacteria_avgs, bacteria_stds, _, samples_num = get_top_bacteria_in_time(estimator, df, top_bacteria, days_start=latest_day, days_number_1unit=days_number_1unit, num_top_bacteria=num_top_bacteria, time_unit_size=time_unit_size, time_unit_name=time_unit_name)
        if xpos is not None:
            bottom -= box_height/2
            xpos   -= box_width/2

            for j in range(len(ratios)):
                # text mean and std for this bacteria inside one boxplot
                bacteria_avg = bacteria_avgs[bacteria_name[j]]
                bacteria_std  = bacteria_stds[bacteria_name[j]]

                if not colors.get(bacteria_name[j], None):
                    colors[bacteria_name[j]] = next(palette)

                height = ratios[j]*box_height
                legendgroup = bacteria_name[j]
                if legendgroup not in legend_vals:
                    legend_vals.append(legendgroup)
                    showlegend = True
                else:
                    showlegend = False
                fig.add_trace(go.Bar(x=[xpos], y=[height],   #red
                        base=bottom,
                        offset=0,
                        width=box_width,
                        marker_color=colors[bacteria_name[j]],  # color dep on bacteria
                        legendgroup=legendgroup,
                        showlegend=showlegend,
                        name=nice_name(bacteria_name[j]),
                        hovertemplate="<br>".join([
                        f"<b>bacteria: {nice_name(bacteria_name[j])}</b>",
                        f"avg ± std: {bacteria_avg:{str_fmt}} ± {bacteria_std:{str_fmt}}",
                        f"importance: {ratios[j]*100:.2f}%",
                        f"# samples: {samples_num}",
                    ])))

                bottom += height

        latest_day += days_number_1unit

    
    fig.update_xaxes( 
        title=f"Age [{time_unit_name}]",
        range=(0, latest_day/time_unit_size),
        tick0=start_age/time_unit_size, dtick=2,
        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey'
    )
    fig.update_yaxes(title=f"Microbiome Maturation Index [{time_unit_name}]",
                     tick0=start_age/time_unit_size, dtick=2,
                     range=(0, latest_day/time_unit_size), 
                     showline=True, linecolor='lightgrey', gridcolor='lightgrey') 
    fig.update_layout(height=layout_height, width=layout_width,
                      barmode='stack', uniformtext=dict(mode="hide", minsize=10),
                      plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=0, r=0, b=0, pad=0),
                      title_text="Trajectory time blocks with top important bacteria")
    if file_name:
        fig.write_html(file_name)
    fig.show()
    
    return traj_pi, traj_mean