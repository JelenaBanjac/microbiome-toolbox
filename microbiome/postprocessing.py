import matplotlib
matplotlib.use('agg')
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
from microbiome.helpers import df2vectors
from microbiome.trajectory import plot_1_trajectory
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

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

def plot_importance_boxplots_over_age(estimator, df_all, top_bacteria, nice_name, units, num_top_bacteria=5, start_age=0, limit_age=None, patent=False, highlight_outliers=None, df_new=None, 
                                        time_unit_size=1, time_unit_name="days", box_height=None, img_file_name=None, plateau_area_start=None, longitudinal_mode="markers+lines", longitudinal_showlegend=True, 
                                        fillcolor_alpha=0.3, layout_settings=None, website=False, PI_percentage=90, dtick=2):
    
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

    layout_settings_default = dict(
        height=900, 
        width=1100,
        barmode='stack', 
        uniformtext=dict(mode="hide", minsize=10),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0, r=0, b=0, pad=0),
        title_text="Trajectory time blocks with top important bacteria"
    )

    if patent:
        palette = itertools.cycle(sns.color_palette("Greys", n_colors=11).as_hex())
        boxplot_alpha = 1.0
        outlier_color = "0,0,0"
        traj_color = "0,0,0"
        marker_outlier=dict(size=25, color=f'rgba({outlier_color},0.95)', symbol="star-open", line_width=4)
        layout_settings_default["font"] = dict(
                #family="Courier New, monospace",
                size=20,
                #color="RebeccaPurple"
            )
    else:
        palette = itertools.cycle(sns.color_palette("Paired").as_hex())
        boxplot_alpha = 0.8
        outlier_color = "255,0,0"
        traj_color = "26,150,65" 
        marker_outlier=dict(size=25, color=f'rgba({outlier_color},0.95)', symbol="star-open", line_width=4)    

    if layout_settings is None:
        layout_settings = {}
    layout_settings_final = {**layout_settings_default, **layout_settings}
        
    colors = {}
    ypos_text_margin = 1.5
    text_fontsize_samples = 15
    str_fmt = ".5f"
    latest_day = start_age
    legend_vals = []

    fig = go.Figure()  
    fig, ret_val, mae, r2, pi_median, traj_x, traj_pi, traj_mean = plot_1_trajectory(fig, estimator, df, top_bacteria, limit_age_max, time_unit_size, time_unit_name, traj_color=traj_color, traj_label="reference", 
                                                                   plateau_area_start=plateau_area_start, limit_age_max=limit_age_max, longitudinal_mode=longitudinal_mode, 
                                                                   longitudinal_showlegend=longitudinal_showlegend, fillcolor_alpha=fillcolor_alpha, highlight_outliers=highlight_outliers,
                                                                  marker_outlier=marker_outlier, df_new=df_new, plot_CI=False, plot_PI=True, PI_percentage=PI_percentage)

    if box_height is None:
        box_height = 2*np.median(traj_pi)
        print("box_height", box_height)

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
        tick0=start_age/time_unit_size, dtick=dtick,
        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey')
    fig.update_yaxes(
        title=f"Microbiome Maturation Index [{time_unit_name}]",
        tick0=start_age/time_unit_size, dtick=dtick,
        range=(0, latest_day/time_unit_size), 
        showline=True, linecolor='lightgrey', gridcolor='lightgrey') 

    fig.update_layout(**layout_settings_final)

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
    
    return fig, traj_x, traj_pi, traj_mean


def outlier_intervention(outlier_sampleID, estimator, df_all, feature_columns, nice_name, max_num_bacterias_to_change, traj_x, traj_mean, traj_pi, time_unit_size=30, time_unit_name="months", plot=False, file_name=None, output_html=False, normal_vals=[False],average=np.median, std=np.std):
    
    def prediction_interval_with_direction(x, traj_mean, traj_pi, direction):
        # direction=-1 for lower limit, +1 for upper
        p, _ = np.polyfit(traj_x, traj_mean+direction*traj_pi, 2, cov=True)    
        y = np.polyval(p, x) 
        return y
    
    ### Outlier importances before ###
    df_all = df_all.copy()
    # chosing one outlier
    df_all = df_all.reset_index(drop=True)
    i = df_all.loc[df_all.sampleID==outlier_sampleID].index[0]
    
    X, y = df2vectors(df_all, feature_columns)
    y_pred = estimator.predict(X)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    # margin will be used as a +- range we want to put the outlier in
    # for e.g. outlier has 15 months on x-axis, so in order to align it to healthy region on y-axis, we need to collect the samples in the region 15+-margin months and do the stats on these healthy subset of samples.
    # if outlier has age_at_collection < 7 months, put +-1 month in MMI, etc.
    margin = 1*30/time_unit_size if int(round(y[i]/time_unit_size)) < 7*30/time_unit_size else 2*30/time_unit_size if  7*30/time_unit_size <= int(round(y[i]/time_unit_size)) < 27*30/time_unit_size else int(round(np.median(traj_pi)))
    #print(margin)

    shift_direction = -1 if y_pred[i]>y[i] else +1
    additional_push = int(shift_direction*np.median(traj_pi)/2)
    left  = max(int(round(y[i]/time_unit_size - margin)) + additional_push, 0)
    right = max(int(round(y[i]/time_unit_size + margin)) + additional_push,0) + margin
    print(left, right)

    left_pred  = max(int(round(prediction_interval_with_direction(right, traj_mean, traj_pi, direction=-1))), 0)
    right_pred     = int(round(prediction_interval_with_direction(left, traj_mean, traj_pi, direction=+1)))
    #print(left_pred, right_pred)

    ret_val1 = "\n--- Before Intervention ---\n"
    ret_val1 += f"age_at_collection = {y[i]:.2f} days [{int(round(y[i]/time_unit_size))} {time_unit_name}]\n"
    ret_val1 += f"microbiota_age = {y_pred[i]:.2f} days [{int(round(y_pred[i]/time_unit_size))} {time_unit_name}]\n"
    ret_val1 += f"put in prediction interval (y-axis) between {left_pred} [{left_pred*time_unit_size} days] and {right_pred} [{right_pred*time_unit_size} days] {time_unit_name} to make it healthy\n"
    ret_val1 += f"do the stats on interval (x-axis) between {left} [{left*time_unit_size} days] and {right} [{right*time_unit_size} days] {time_unit_name}\n"
    if not output_html:
        print(ret_val1)

    # shorten bacteria names just for the force plot below
    feature_columns_short = list(map(nice_name, feature_columns))
    if plot:
        shap.force_plot(explainer.expected_value, shap_values[i], features=X[i], feature_names=feature_columns_short, text_rotation=90, matplotlib=True)
        plt.show()
    else:
        fig1, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[i], features=X[i], feature_names=feature_columns_short, text_rotation=90, matplotlib=plot)
        
        fig1 =  mpl_to_plotly(fig1)
            
        fig1.update_xaxes(title="SHAP value (impact on model output)", 
                        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
        fig1.update_yaxes(title="Features", 
                            tickmode='array',
                            tickvals=list(range(0, min(20, len(feature_columns_short)))),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                            #ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))][::-1])),
                        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
        fig1.update_layout(#height=layout_height, width=layout_width,
                        #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, b=0, pad=0),
                        title_text="Classification Important Features")

    ### Intervention Table ###
    # get importances of outlier
    feature_importance_outlier = pd.DataFrame(list(zip(feature_columns, np.abs(shap_values).mean(0))), columns=['bacteria_name','feature_importance_outlier'])
    feature_importance_outlier.sort_values(by=['feature_importance_outlier'], ascending=True, inplace=True)

    # get importances of the trajectory
    _, _, _, _, _, _, feature_importance, _ = get_top_bacteria_in_time(estimator, df_all, feature_columns, 
                                                                       days_start=left*time_unit_size, 
                                                                       days_number_1unit=(left+right)*time_unit_size, 
                                                                       num_top_bacteria=5, 
                                                                       average=average, std=std)
    print(feature_importance)
    #display(feature_importance)
    # merge two importances
    feature_importance["outlier"] = feature_importance.apply(lambda x: df_all.iloc[i][x["bacteria_name"]], axis=1)
    feature_importance["normal"] = feature_importance.apply(lambda x: True if x["bacteria_avg"]-x["bacteria_std"] < x["outlier"] < x["bacteria_avg"]+x["bacteria_std"] else False, axis=1)
    feature_importance = pd.merge(feature_importance, feature_importance_outlier, on="bacteria_name")
    feature_importance.sort_values(by=['feature_importance_outlier', 'feature_importance_vals'], ascending=False,inplace=True)
    if file_name:
        feature_importance.to_csv(file_name, index=False)
    
    
    # change the most important bacteria value
    # Faecalibacterium is the first important bacteria that is outside the healthy range
    df_all_updated = df_all.copy()

    #bacterias_to_change = feature_importance.iloc[:5].bacteria_name.values  #
    #bacterias_to_change = list(set(list(feature_importance[feature_importance.normal==False].bacteria_name.values) + list( feature_importance.iloc[:5].bacteria_name.values)))
    bacterias_to_change = feature_importance[feature_importance.normal.isin(normal_vals)].bacteria_name.values[:max_num_bacterias_to_change]

    ret_val2 = "\n--- Intervention bacteria ---\n"
    for bacteria_to_change in bacterias_to_change:
        ret_val2 += f"\n{nice_name(bacteria_to_change)}\n"
        ret_val2 +=f"before = {df_all_updated.at[i, bacteria_to_change]}\n"
        df_all_updated.at[i, bacteria_to_change] = feature_importance[feature_importance.bacteria_name==bacteria_to_change]["bacteria_avg"].values[0]
        ret_val2 +=f"after = {df_all_updated.at[i, bacteria_to_change]}\n"
    if not output_html:
        print(ret_val2)

    X, y = df2vectors(df_all_updated, feature_columns)
    y_pred = estimator.predict(X)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    margin = 1*30/time_unit_size if int(round(y[i]/time_unit_size)) < 7*30/time_unit_size else 2*30/time_unit_size if  7*30/time_unit_size <= int(round(y[i]/time_unit_size)) < 27*30/time_unit_size else int(round(np.median(traj_pi)))
    
    shift_direction = -1 if y_pred[i]>y[i] else +1
    additional_push = shift_direction*np.median(traj_pi)/2
    left  = max(int(round(y[i]/time_unit_size - margin)) + additional_push, 0)
    right = max(int(round(y[i]/time_unit_size + margin)) + additional_push,0) + margin
    print(left, right)
    
    left_pred  = max(int(round(prediction_interval_with_direction(right, traj_mean, traj_pi, direction=-1))), 0)
    right_pred     = int(round(prediction_interval_with_direction(left, traj_mean, traj_pi, direction=+1)))
    
    ret_val3 = "\n--- After Intervention ---\n"
    ret_val3 += f"age_at_collection = {y[i]:.2f} days [{int(round(y[i]/time_unit_size))} {time_unit_name}]\n"
    ret_val3 += f"microbiota_age = {y_pred[i]:.2f} days [{int(round(y_pred[i]/time_unit_size))} {time_unit_name}]\n"
    ret_val3 += f"put in prediction interval (y-axis) between {left_pred} [{left_pred*time_unit_size} days] and {right_pred} [{right_pred*time_unit_size} days] months to make it healthy\n"
    ret_val3 += f"do the stats on interval (x-axis) between {left} [{left*time_unit_size} days] and {right} [{right*time_unit_size} days] {time_unit_name}\n"
    if not output_html:
        print(ret_val3)
    if plot:
        shap.force_plot(explainer.expected_value, shap_values[i], features=X[i], feature_names=feature_columns_short, text_rotation=90, matplotlib=True)
        plt.show()
    else:

        fig2, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[i], features=X[i], feature_names=feature_columns_short, text_rotation=90, matplotlib=plot)
        
        fig2 =  mpl_to_plotly(fig2)
            
        fig2.update_xaxes(title="SHAP value (impact on model output)", 
                        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
        fig2.update_yaxes(title="Features", 
                            tickmode='array',
                            tickvals=list(range(0, min(20, len(feature_columns_short)))),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                            #ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))][::-1])),
                        showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
        fig2.update_layout(#height=layout_height, width=layout_width,
                        #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, b=0, pad=0),
                        title_text="Classification Important Features")
    
    
    return df_all_updated, fig1, fig2, ret_val1+ret_val2+ret_val3