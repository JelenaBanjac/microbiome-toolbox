import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from elmtoolbox.variables import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


### PLOTLY ###
def dataset_bacteria_abundances(df, bacteria_names, time, num_cols, nice_name=lambda x: x, file_name=None):
    """ Plot dataset's bacteria abundances
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing at least all bacteria names and the time axis
    bacteria_names: list
        List of bacteria names for which we want to see abundances
    time: str
        The name of the column that is our time axis
    num_cols: int
        The number of columns in the figure
    nice_name: callable
        Function that converts ugly string bacteria name to a readable one
    file_name: str
        The HTML file to store the interactive plot
    """
    num_rows = len(bacteria_names)//num_cols+1

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for idx, bacteria_name in enumerate(bacteria_names):
        fig.add_trace(go.Scatter(
                x=df.groupby(by=time).agg(np.mean)[bacteria_name].index,
                y=df.groupby(by=time).agg(np.mean)[bacteria_name],
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=df.groupby(by=time).agg(np.std)[bacteria_name],
                    visible=True), name=nice_name(bacteria_name)
            ), row=idx//num_cols+1, col=idx%num_cols+1)
        fig.update_xaxes(title="Time", row=idx//num_cols+1, col=idx%num_cols+1)  # gridcolor='lightgrey'
        fig.update_yaxes(title=nice_name(bacteria_name), row=idx//num_cols+1, col=idx%num_cols+1)  # gridcolor='lightgrey'

    fig.update_layout(height=1900, width=1500, 
                      paper_bgcolor="white",#'rgba(0,0,0,0)', 
                      #plot_bgcolor='rgba(0,0,0,0)', 
                       margin=dict(l=0, r=0, b=0, pad=0),
                      title_text="Bacteria Abundances in the Dataset")
    if file_name:
        fig.write_html(file_name)
    fig.show()



def sampling_statistics(df, train_subjectIDs, val_subjectIDs, test_subjectIDs, group, start_age=0, limit_age=1200, time_unit_size=1, time_unit_name="days", file_name=None):
    
    def fill_in_data_type(row):
        if row["subjectID"] in train_subjectIDs: 
            row["data_type"] = "train"  #"blue" #1 
            row["order"] =  0
        elif row["subjectID"] in val_subjectIDs: 
            row["data_type"] = "validation"  #"green" #2 
            row["order"] =  1
        elif row["subjectID"] in test_subjectIDs: 
            row["data_type"] = "test"  #"red" #3 
            row["order"] =  2
        else:
            raise Exception(f"Please assign the type to all the subjectID! Here, {row['subjectID']} is not assigned to any of train, test, val datasets")
        return row

    colors = ["0,0,255", "255,0,0", "0,255,0"]

    if group is not None:

        num_cols = len(df[group].unique())
        fig = make_subplots(rows=1, cols=num_cols, subplot_titles=[f"{group}={s}" for s in df[group].unique()])

        for col, g in enumerate(df[group].unique(), 1):
            df1 = df[df[group]==g]
            df1 = df1[[ "subjectID", "age_at_collection", "sampleID"]].sort_values(by=["subjectID", "age_at_collection"])
            df1 = df1.apply(lambda row: fill_in_data_type(row), axis=1)
            df1 = df1.sort_values(by=["order"])

            # longitudinal - line per subject
            for sid in df1["subjectID"].unique():
                idx = np.where(df1["subjectID"]==sid)[0]

                for i, dt in enumerate(df1["data_type"].unique()):

                    idx2 = np.where(df1.iloc[idx]["data_type"]==dt)[0]
                    traj_color = colors[i]
                    fig.add_trace(go.Scatter(
                        x=df1.age_at_collection.values[idx][idx2],
                        y=[sid]*len(idx2),
                        mode="markers+lines",
                        line = dict(width=3, dash='solid', color=f'rgba({traj_color},0.35)'),
                        marker=dict(size=10, color=f'rgba({traj_color},0.35)'),
                        showlegend=True,
                        name=sid,
                        text=list(df1["sampleID"].values[idx]), 
                        hovertemplate = f'<b>{dt.title()} sample</b><br><br>'+
                                        '<b>SampleID</b>: %{text}<br>'+
                                        '<b>SubjectID</b>: %{y}<br>'+
                                        '<b>Age</b>: %{x:.2f}<br>'
                                        ,
                        hoveron="points"
                    ), row=1, col=col)

            fig.update_xaxes(title=f"Age [{time_unit_name}]", range=(start_age/time_unit_size-1, limit_age/time_unit_size+1), 
                            tick0=start_age/time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray', row=1, col=col) 
            fig.update_yaxes(title=f"Subject ID ", gridcolor='lightgrey', showspikes=True, spikecolor='gray', row=1, col=col)  


        fig.update_layout(height=500, width=600*num_cols, 
                          #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0, r=0, b=0, pad=0),
                          title_text="Sampling statistics")
        fig.show()
    else:

        fig = go.Figure()

        df1 = df[[ "subjectID", "age_at_collection", "sampleID"]].sort_values(by=["subjectID", "age_at_collection"])
        df1 = df1.apply(lambda row: fill_in_data_type(row), axis=1)
        df1 = df1.sort_values(by=["order"])

        # longitudinal - line per subject
        for sid in df1["subjectID"].unique():
            idx = np.where(df1["subjectID"]==sid)[0]

            for i, dt in enumerate(df1["data_type"].unique()):
                idx2 = np.where(df1.iloc[idx]["data_type"]==dt)[0]
                traj_color = colors[i]
                fig.add_trace(go.Scatter(
                    x=df1.age_at_collection.values[idx][idx2],
                    y=[sid]*len(idx2),
                    mode="markers+lines",
                    line = dict(width=3, dash='solid', color=f'rgba({traj_color},0.35)'),
                    marker=dict(size=10, color=f'rgba({traj_color},0.35)'),
                    showlegend=True,
                    name=sid,
                    text=list(df1["sampleID"].values[idx]), 
                    hovertemplate = f'<b>{dt.title()} sample</b><br><br>'+
                                    '<b>SampleID</b>: %{text}<br>'+
                                    '<b>SubjectID</b>: %{y}<br>'+
                                    '<b>Age</b>: %{x:.2f}<br>'
                                    ,
                    hoveron="points"
                ))

        fig.update_xaxes(title=f"Age [{time_unit_name}]", range=(start_age/time_unit_size-1, limit_age/time_unit_size+1), 
                        tick0=start_age/time_unit_size, dtick=round(2/time_unit_size, 1), gridcolor='lightgrey', showspikes=True, spikecolor='gray') 
        fig.update_yaxes(title=f"Subject ID ", gridcolor='lightgrey', showspikes=True, spikecolor='gray')  


        fig.update_layout(height=500, width=650, 
                          #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0, r=0, b=0, pad=0),
                          title_text="Sampling statistics")
        fig.show()

    if file_name:
        fig.write_html(file_name)



def plot_bacteria_abundance_heatmaps(df, bacteria_names, short_bacteria_name=lambda x: x, time_unit_name="days", time_unit_size=1, avg_fn=np.median, fillna=False):
    """ Plot bacteria abundances over time 
    
    Plot 2 plots:
    - average absolute abundances of bacteria with time
    - average relative (relative to bacteria observed) abundance of bacteria with time (and plot will be filled with more colors than the one above :) )

    Parameters
    ----------
    df_all: pd.DataFrame
        Dataset with everything.
    bacteria_names: list
        Names of all the bacteria in the dataset.
    short_bacteria_name: callable
        Function that wills shorten the very long bacteria name. Default function is identity: the same name as given without shortening.
    time_unit_name: str
        Name of the time unit (e.g. month, year, etc.)
    time_unit_size: int
        Number of days in a new time definition (e.g. if we want to deal with months, then time_unit_in_days=30, for the year, time_unit_in_days=365)
    avg_fn: callable
        Function for averaging. We are used to mean, but sometimes median best represents the real situation. By default, median is used.

    Examples
    --------
    >> df_all = ... # dataset we want to use in the heatmap (we use the validation dataset)
    >> bacteria_names = ... # list of 16s bacteria we are interested in plotting in the heatmap (all bacteria we have specified in df_all)
    >> short_bacteria_name = lambda name: " | ".join([c[3:] for c in name.split(";")[-2:]])  # shorteing 16s data
    >> plot_bacteria_abundance_heatmaps(df_all=df_all, top_bacteria=important_features_reorder, bacteria_names=bacteria_names, short_bacteria_name=short_bacteria_name, time_unit_name="month", time_unit_in_days=30)
    """    
    
    # Plot months
    _df = df.copy()
     # create the time unit that will be on the x-axis
    _df["time_unit"] = _df.age_at_collection//time_unit_size
    # extract just the important columns for the heatmap
    _df = _df[list(bacteria_names)+["subjectID", "time_unit"]]
    # replace long bacteria names with nice names
    _df = _df.rename(dict([(b, short_bacteria_name(b)) for b in bacteria_names]), axis=1)
    # update the bacteria_names with short names
    bacteria_names = np.array(list(map(lambda x: short_bacteria_name(x), bacteria_names)))

    def fill_collected(row):
        """Ïf we use bigger time units, then we need to find a median when collapsing all the samples in that time interval into one box in the heatmap"""
        val = _df[_df["time_unit"]==row["time_unit"]][row["bacteria_name"]].values
        row["bacteria_value"] = avg_fn(val) if len(val)!=0 else np.nan
        return row

    x, y = np.meshgrid(bacteria_names, range(int(max(_df["time_unit"]))+1))

    df_heatmap = pd.DataFrame(data={"bacteria_name":  x.flatten(), 
                                    "time_unit":      y.flatten(), 
                                    "bacteria_value": np.nan})
    df_heatmap = df_heatmap.sort_values(by=["bacteria_name", "time_unit"])
    df_heatmap = df_heatmap.fillna(0)
    df_heatmap = df_heatmap.apply(lambda row: fill_collected(row), axis=1)
    
    # create new column bacteria_name_cat in order to sort dataframe by bacteria importance
    df_heatmap['bacteria_name_cat'] = pd.Categorical(
        df_heatmap['bacteria_name'], 
        categories=bacteria_names, # order of bacteria is imposed by the list
        ordered=True
    )
    df_heatmap = df_heatmap.sort_values('bacteria_name_cat')
    df_heatmap = df_heatmap[df_heatmap.time_unit > 0]

    # plot top absolute
    df_heatmap = df_heatmap[["time_unit", "bacteria_name_cat", "bacteria_value"]]
    df_heatmap_pivot = df_heatmap.pivot("bacteria_name_cat", "time_unit", "bacteria_value")
        
    if fillna:
        df_heatmap_pivot = df_heatmap_pivot.fillna(0)

    fig = px.imshow(df_heatmap_pivot.values,
                    labels=dict(x=f"Age [{time_unit_name}]", y="Bacteria Name", color="Abundance"),
                    x=df_heatmap_pivot.columns,
                    y=df_heatmap_pivot.index,
                    title="Absolute abundances",
                    height=20*len(bacteria_names), width=20*len(max(df_heatmap_pivot.index, key=len))+20*max(df_heatmap_pivot.columns)
                   )
    fig.update_xaxes(side="bottom")
    fig.show()

    #######
    X = df_heatmap.bacteria_value.values 
    X = X.reshape(len(bacteria_names), -1)
    xmin = np.nanmin(X, axis=1).reshape(len(bacteria_names), 1)
    xmax = np.nanmax(X, axis=1).reshape(len(bacteria_names), 1)
    X_std = (X - xmin) / (xmax - xmin+1e-10) 
    df_heatmap.bacteria_value = X_std.flatten()
    df_heatmap_relative = df_heatmap[["time_unit", "bacteria_name_cat", "bacteria_value"]]
    df_heatmap_relative_pivot = df_heatmap_relative.pivot("bacteria_name_cat", "time_unit", "bacteria_value")
    
    if fillna:
        df_heatmap_relative_pivot = df_heatmap_relative_pivot.fillna(0)

    fig = px.imshow(df_heatmap_relative_pivot.values,
                    labels=dict(x=f"Age [{time_unit_name}]", y="Bacteria Name", color="Abundance"),
                    x=df_heatmap_relative_pivot.columns,
                    y=df_heatmap_relative_pivot.index,
                    title="Relative abundances",
                    height=20*len(bacteria_names), width=20*len(max(df_heatmap_pivot.index, key=len))+20*max(df_heatmap_pivot.columns)
                   )
    fig.update_xaxes(side="bottom")
    fig.show()



### MATPLOTLIB ###
def sampling_statistics_matplotlib(df, train_subjectIDs, val_subjectIDs, test_subjectIDs, vertical=True, file_name=None):
    """ Plot data split to train-validation-test sets

    Plot data split to train-validation-test sets across each of the countries. This plot is used to help us understand the 
    sampling frequency for every subject (infant) in each of the subsets across all the countries. Usually used at the begining 
    of the pipeline to gain more insight in the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataset containing all data that we want to split into train-validation-test sets.
    train_subjectIDs: list, pd.Series
        Subject IDs (infant IDs) that are in the training set.
    val_subjectIDs: list, pd.Series
        Subject IDs (infant IDs) that are in the validation set.
    test_subjectIDs: list, pd.Series
        Subject IDs (infant IDs) that are in the test set.
    vertical: bool
        The direction countries are ordered. Default is True. Otherwise, it will be horizontal plot.
    file_name: str
        The name of the file to save the plot (.pdf extension for the highest quality). Default None is not saving the plot.

    Examples
    --------
    >> plot_train_val_test_sampling(df, train.subjectID.unique(), val.subjectID.unique(), test.subjectID.unique(), vertical=False, file_name=f"{PIPELINE_DIRECTORY}/images/sampling_stats.pdf")

    """
    def fill_in_data_type(row):
        if row["subjectID"] in train_subjectIDs: 
            row["data_type"] = "train"  #"blue" #1 
            row["order"] =  0
        elif row["subjectID"] in val_subjectIDs: 
            row["data_type"] = "validation"  #"green" #2 
            row["order"] =  1
        elif row["subjectID"] in test_subjectIDs: 
            row["data_type"] = "test"  #"red" #3 
            row["order"] =  2
        return row
    
    sns.set_style("whitegrid")
    colors = ["blue", "red", "green"]
    palette = sns.set_palette(sns.color_palette(colors))
    margin = 0.75
    text_margin = 1.11
    axis_fontsize = 15
    legend_fontsize = 15
    
    plt.rc('axes', labelsize= axis_fontsize)    
    plt.rc('xtick', labelsize= axis_fontsize)   
    plt.rc('legend', fontsize= legend_fontsize) 

    mgs_marker_size = 130
    marker_size = 8
    
    if vertical:
        df1 = df[["subjectID", "age_at_collection"]].sort_values(by=["subjectID", "age_at_collection"])
        df1 = df1.apply(lambda row: fill_in_data_type(row), axis=1)
        df1["month"] = df1.age_at_collection//30
        df1_fin = df1[df1.subjectID.str.startswith("E")]
        df1_fin = df1_fin.sort_values(by=["order"])
        df1_rus = df1[df1.subjectID.str.startswith("P")]
        df1_rus = df1_rus.sort_values(by=["order"])
        df1_est = df1[df1.subjectID.str.startswith("T")]
        df1_est = df1_est.sort_values(by=["order"])
        df1 = pd.concat([df1_fin, df1_rus, df1_est])

        # https://python-graph-gallery.com/43-use-categorical-variable-to-color-scatterplot-seaborn/
        plt.figure(figsize=(10,50))
        g = sns.lineplot(x="month", y="subjectID", hue="data_type", units="subjectID", estimator=None, data=df1, legend=True, marker="o", palette=palette)
        g.set(xlim=(0,max(df1.month)))

        plt.fill_between(np.linspace(0, max(df1.month), 10), 10*[-0.5], 10*[len(df1_fin.subjectID.unique())-0.5], alpha=0.1, color=col_fin, label="Finland")
        plt.fill_between(np.linspace(0, max(df1.month), 10), 10*[-0.5+len(df1_fin.subjectID.unique())], 10*[len(df1_fin.subjectID.unique()) + len(df1_rus.subjectID.unique())-0.5], alpha=0.1, color=col_rus, label="Russia")
        plt.fill_between(np.linspace(0, max(df1.month), 10), 10*[-0.5+len(df1_fin.subjectID.unique()) + len(df1_rus.subjectID.unique()) ], 10*[len(df1_fin.subjectID.unique()) + len(df1_rus.subjectID.unique()) +len(df1_est.subjectID.unique())-0.5], alpha=0.1, color=col_est, label="Estonia")

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(file_name)
        plt.show()
    else:      
        df1 = df[["subjectID", "age_at_collection", "sampleID"]].sort_values(by=["subjectID", "age_at_collection"])
        df1 = df1.apply(lambda row: fill_in_data_type(row), axis=1)
        df1["month"] = df1.age_at_collection//30
        
        df1_fin = df1[df1.subjectID.str.startswith("E")]
        df1_fin = df1_fin.sort_values(by=["order"])
        df1_rus = df1[df1.subjectID.str.startswith("P")]
        df1_rus = df1_rus.sort_values(by=["order"])
        df1_est = df1[df1.subjectID.str.startswith("T")]
        df1_est = df1_est.sort_values(by=["order"])
        df1 = pd.concat([df1_fin, df1_rus, df1_est])

        # https://python-graph-gallery.com/43-use-categorical-variable-to-color-scatterplot-seaborn/
        fig, axs = plt.subplots(1, 3, figsize=(35,15))
        sns.lineplot(x="month", y="subjectID", hue="data_type", units="subjectID", estimator=None, data=df1_fin, legend=True, marker="o", markersize=marker_size, palette=palette, ax=axs[0])
        sns.lineplot(x="month", y="subjectID", hue="data_type", units="subjectID", estimator=None, data=df1_rus, legend=True, marker="o", markersize=marker_size, palette=palette, ax=axs[1])
        sns.lineplot(x="month", y="subjectID", hue="data_type", units="subjectID", estimator=None, data=df1_est, legend=True, marker="o", markersize=marker_size, palette=palette, ax=axs[2])
        
        # place a text box in upper left in axes coords
        axs[0].text(0.01, 1.02, "Finland", transform=axs[0].transAxes, fontsize=22, verticalalignment='center', bbox=dict(boxstyle='round', facecolor=col_fin, alpha=0.5))
        axs[1].text(text_margin, 1.02, "Russia", transform=axs[0].transAxes, fontsize=22, verticalalignment='center', bbox=dict(boxstyle='round', facecolor=col_rus, alpha=0.5))
        axs[2].text(2*text_margin-0.005, 1.02, "Estonia", transform=axs[0].transAxes, fontsize=22, verticalalignment='center', bbox=dict(boxstyle='round', facecolor=col_est, alpha=0.5))
        #axs[0].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[-margin], 10*[len(df1_fin.subjectID.unique())-1+margin], alpha=0.1, color=col_fin)
        axs[0].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_fin.subjectID.unique())-1+margin], 10*[len(df1_fin.subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==2].subjectID.unique())-1+margin], alpha=0.1, color="green")
        axs[0].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_fin.subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==2].subjectID.unique())-1+margin], 10*[len(df1_fin.subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==2].subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==1].subjectID.unique())-0.15], alpha=0.1, color="red")
        axs[0].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_fin.subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==2].subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==1].subjectID.unique())-0.15], 10*[len(df1_fin.subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==2].subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==1].subjectID.unique())-1+margin-len(df1_fin[df1_fin.order==0].subjectID.unique())-1+margin], alpha=0.1, color="blue")
        #axs[1].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[-margin], 10*[len(df1_rus.subjectID.unique())-1+margin], alpha=0.1, color=col_rus)
        axs[1].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_rus.subjectID.unique())-1+margin], 10*[len(df1_rus.subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==2].subjectID.unique())-1+margin], alpha=0.1, color="green")
        axs[1].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_rus.subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==2].subjectID.unique())-1+margin], 10*[len(df1_rus.subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==2].subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==1].subjectID.unique())-1+margin], alpha=0.1, color="red")
        axs[1].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_rus.subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==2].subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==1].subjectID.unique())-1+margin], 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==2].subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==1].subjectID.unique())-1+margin-len(df1_rus[df1_rus.order==0].subjectID.unique())-1+margin], alpha=0.1, color="blue")
        #axs[2].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[-margin], 10*[len(df1_est.subjectID.unique())-1+margin], alpha=0.1, color=col_est)
        axs[2].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_est.subjectID.unique())-1+margin], 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_est[df1_est.order==2].subjectID.unique())-1+margin], alpha=0.1, color="green")
        axs[2].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_est[df1_est.order==2].subjectID.unique())-1+margin], 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_est[df1_est.order==2].subjectID.unique())-1+margin-len(df1_est[df1_est.order==1].subjectID.unique())-1+margin], alpha=0.1, color="red")
        axs[2].fill_between(np.linspace(-0.5, max(df1.month)+0.5, 10), 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_est[df1_est.order==2].subjectID.unique())-1+margin-len(df1_est[df1_est.order==1].subjectID.unique())-1+margin], 10*[len(df1_est.subjectID.unique())-1+margin-len(df1_est[df1_est.order==2].subjectID.unique())-1+margin-len(df1_est[df1_est.order==1].subjectID.unique())-1+margin-len(df1_est[df1_est.order==0].subjectID.unique())-1+margin], alpha=0.1, color="blue")
       
        
        axs[0].set_xlim((-0.5,max(df1.month)+0.5));axs[0].set_ylim((-margin, len(df1_fin.subjectID.unique())-1+margin))
        axs[1].set_xlim((-0.5,max(df1.month)+0.5));axs[1].set_ylim((-margin, len(df1_rus.subjectID.unique())-1+margin))
        axs[2].set_xlim((-0.5,max(df1.month)+0.5));axs[2].set_ylim((-margin, len(df1_est.subjectID.unique())-1+margin))
        axs[0].legend(bbox_to_anchor=(1.0, 1.04), ncol=4)
        axs[1].legend(bbox_to_anchor=(1.0, 1.04), ncol=4)
        axs[2].legend(bbox_to_anchor=(1.0, 1.04), ncol=4)
        axs[0].set_ylabel("Subject ID");axs[0].set_xlabel("Age at collection [months]")
        axs[1].set_ylabel("");axs[1].set_xlabel("Age at collection [months]")
        axs[2].set_ylabel("");axs[2].set_xlabel("Age at collection [months]")
        
        plt.tight_layout()
        plt.savefig(file_name)
        plt.show()

