import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, sem
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, spearmanr, pearsonr, t
from skbio.stats.composition import multiplicative_replacement
from scipy.stats import ttest_ind
import plotly.express as px


def plot_abundances_and_log(b1, b2, rmp_data, nice_name):
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 13))
    sns.pointplot(x="label", y='bacteria1', hue="time", data=rmp_data, ax=axs[0])
    axs[0].set_ylabel(nice_name(b1))
    sns.pointplot(x="label", y='bacteria2', hue="time", data=rmp_data, ax=axs[1])
    axs[1].set_ylabel(nice_name(b2))
    sns.pointplot(x="label", y='log(bacteria1/bacteria2)', hue="time", data=rmp_data, ax=axs[2])
    axs[2].set_ylabel(r'$\log( \frac{' + nice_name(b1) + '}{' + nice_name(b2) + '})$')
    axs[1].get_legend().remove();axs[2].get_legend().remove()
    axs[0].set_xlabel("");axs[1].set_xlabel("");axs[2].set_xlabel("");
    
    plt.show()
    

def plot_trajectory(ax, df, name, title='', left_title='', ypad=0, logscale=False, norm1=False):

    subs = df['time'].value_counts().index
    
    for i, sub in enumerate(np.sort(subs)):
        subdf = df.loc[df['time'] == sub]
        subdf = subdf.sort_values(by='label').iloc[::-1]

        x = np.arange(2)
        y = subdf.groupby('label').mean()[name][::-1]
        
        if norm1:
            y = y / y[0]
        else:
            y = y - y[0]
        
        e = subdf.groupby('label').agg(sem)[name][::-1]
        ax.errorbar(x, y, yerr=e, label=sub, marker='o')

    ax.set_xticks([0, 1])            
    ax.set_xticklabels(y.index)

    if logscale:
        ax.set_yscale('log')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    for x in ax.get_xticklabels():
        x.set_fontsize(12)

    ax.set_ylabel(left_title, fontsize=14, labelpad=ypad, rotation=0)

    for y in ax.get_yticklabels():
        y.set_fontsize(12)   
        

def plot_abundance_and_ratio(b1, b2, table, metadata, nice_name):
    # TODO: make time column general, to support
    rmp_table = table.apply(lambda x: x * 100 / x.sum(), axis=1)
    rmp_table = pd.DataFrame(data=multiplicative_replacement(rmp_table.values), columns=rmp_table.columns, index=rmp_table.index)
    
    print(f"Bacteria 1: {b1}\nBacteria 2: {b2}")

    rmp_data = pd.DataFrame({
        'log(bacteria1/bacteria2)': np.log10(rmp_table[b1]) - np.log10(rmp_table[b2]),
        'bacteria1': rmp_table[b1],
        'bacteria2': rmp_table[b2],
        'label': metadata['label'],
        'time': metadata['time']    
    })

    plot_abundances_and_log(b1, b2, rmp_data, nice_name)

    num_taxa = 'bacteria1'
    denom_taxa = 'bacteria2'
    ypad_scalar = 4

    fig, axes = plt.subplots(3, 1, figsize=(3, 8))
    plot_trajectory(axes[0], rmp_data, num_taxa, 
                    logscale=False, left_title=r"$\it{%s}$" % nice_name(b1), ypad=len(nice_name(b1))*ypad_scalar)
    plot_trajectory(axes[1], rmp_data, denom_taxa,
                    logscale=False, left_title=r"$\it{%s}$" % nice_name(b2), ypad=len(nice_name(b2))*ypad_scalar)
    plot_trajectory(axes[2], rmp_data, 'log(' + num_taxa + '/' + denom_taxa + ')', 
                    left_title=r'$\log( \frac{' + nice_name(b1) + '}{' + nice_name(b2) + '})$', ypad=max(len(nice_name(b1)), len(nice_name(b1)))*ypad_scalar)

    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=1, fancybox=True, shadow=True)
    axes[2].set_xlabel('% rel')

    # Test if bacteria (numerator, denominator, log-ratio) is changing in the data?
    # numerator taxa - bacteria 1
    subrmp_data = rmp_data.dropna()
    statistic, pval = ttest_ind(np.array(subrmp_data.loc[subrmp_data['label'] == 'reference', num_taxa]),
                                np.array(subrmp_data.loc[subrmp_data['label'] == 'other', num_taxa]))
    print("statistic =", statistic, "\np-value =",pval)
    print("The means are NOT significantly different." if pval>0.05 else "The means are significantly different.")

    # denominator taxa - bacteria 2
    subrmp_data = rmp_data.dropna()
    statistic, pval = ttest_ind(np.array(subrmp_data.loc[subrmp_data['label'] == 'reference', denom_taxa]),
                                np.array(subrmp_data.loc[subrmp_data['label'] == 'other', denom_taxa]))
    print("statistic =", statistic, "\np-value =",pval)
    print("The means are NOT significantly different." if pval>0.05 else "The means are significantly different.")

    # log ration of two taxa
    subrmp_data = rmp_data.dropna()
    statistic, pval = ttest_ind(
        np.array(subrmp_data.loc[subrmp_data['label'] == 'reference', 
                              'log(' + num_taxa + '/' + denom_taxa + ')']),
        np.array(subrmp_data.loc[subrmp_data['label'] == 'other', 
                              'log(' + num_taxa + '/' + denom_taxa + ')']))
    print("statistic =", statistic, "\np-value =",pval)
    print("The means are NOT significantly different." if pval>0.05 else "The means are significantly different.")
    
def differential_ranking_with_hybrid_matplotlib(beta, df_hybrid_and_all, reference_column):
    _df = pd.merge(df_hybrid_and_all.set_index("bacteria_name"), beta, left_index=True, right_index=True)
    _df = _df.sort_values(by=reference_column)
    x = np.arange(_df.shape[0])

    fig = plt.figure(figsize=(25, 10))
    
    for b in _df.hybrid_bacteria_name.unique():
        idx = np.where(_df.hybrid_bacteria_name==b)[0]
        color = _df.iloc[idx].drop_duplicates().color.values[0]
        plt.bar(x[idx], _df.iloc[idx][reference_column].values, 0.8, alpha=0.2 if len(b)==0 else 0.8, edgecolor=color, facecolor=color, lw=1, label="%s"%b[0:].replace("_", " ") if len(b)!=0 else None)  #label="%s" % genus[3:].replace("_", " "),
    
    plt.xlabel('species');plt.xticks([]);plt.ylabel(r'$\log (\frac{referemce}{others}) + K$', fontsize=20, labelpad=90, rotation=0);
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(x, np.array([0]*len(x)), c='k', lw=1)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    
def differential_ranking_with_hybrid_big_matplotlib(beta, df_hybrid_and_all, reference_column):
    _df = pd.merge(df_hybrid_and_all.set_index("bacteria_name"), beta, left_index=True, right_index=True)
    _df = _df.sort_values(by=reference_column)

    # Plot the figure.
    plt.figure(figsize=(30, 100))
    plt.rc('ytick', labelsize=15)

    ax =  _df[reference_column].plot(kind='barh', color=_df.color)
    
    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Create annotation
        plt.annotate(
            f"{x_value:.2e}",                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha,
            size=15)                      # Horizontally align label differently for
                                        # positive and negative values.

    ax.set_xlabel(r'$\log (\frac{referemce}{others}) + K$')
    ax.set_ylabel('Bacteria') 
    ax.set_yticklabels(_df.index)
    
    plt.show()

    
def differential_ranking_with_hybrid(beta, df_hybrid_and_all, reference_column, height=600, width=1600, file_name=None):
    # reference_column="label[T.reference]"
    _df = pd.merge(df_hybrid_and_all.set_index("bacteria_name"), beta, left_index=True, right_index=True)
    _df = _df.sort_values(by=reference_column)
    x = np.arange(_df.shape[0])
    
    fig = go.Figure()    
    fig.add_trace(go.Bar(x=x, y=_df[reference_column].values, marker_color=_df.color,
                 customdata=[(x, y) for x, y in zip(_df.index, _df.hybrid_bacteria_name)],
                 hovertemplate = '<b>Taxa: %{customdata[0]}</b><br><br>'+
                                '<b>Log-ratio</b>: %{y}<br>'+
                                '<b>Hybrid class</b>: %{customdata[1]}<br>'))
    
    fig.update_xaxes(title="Species", gridcolor='lightgrey', ticks="", showticklabels=False) 
    fig.update_yaxes(title=r'$\log (\frac{referemce}{others}) + K$', gridcolor='lightgrey')  


    fig.update_layout(height=height, width=width, 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=0, r=0, b=0, pad=0),
                      title_text="Songbird's Multinomial Regression")
    if file_name:
        fig.write_html(file_name)

    fig.show()