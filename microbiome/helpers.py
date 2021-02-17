import io
import base64
import html
from shap.plots._force_matplotlib import draw_additive_plot
import shap


def df2vectors(_df, feature_cols=None):
    """Function that converts data frame to corresponding feature vector X and the label vector y
    
    This function is mostly used to convert pandas dataframe objects to lists, and then propagated to
    different ML methods (i.e. CatBoost, RandomForest) in order to perform the prediction.

    Parameters
    ----------
    _df: pd.DataFrame
        Data frame that we want to convert to a feature vector X with its corresponding labels y.
    feature_cols: list
        Column names that represent the features that will be used in a feature vector X (e.g. bacteria names, and/or meta data). There
        should not be any column that is ID (e.g. sample id, etc.)
    
    Returns
    -------
    X: list of lists
        Feature matrix with dimension (n_samples, n_features). 
    y: list
        Label vector with dimension (n_samples,)
    """
    
    # labels, what we want to predict -> age of the infant
    y = _df["age_at_collection"]
    
    _df = _df[feature_cols]
    
    # features, what is given to us -> bacteria info
    X = _df.values

    return X, y


def get_bacteria_names(df, bacteria_fun=lambda x: x.startswith("k__")):
    return list(df.columns[df.columns.map(bacteria_fun)])


import shap
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def two_groups_analysis(df_all, feature_cols, references_we_compare, test_size=0.5, n_splits=5, nice_name=lambda x: x, style="dot", show=False, website=True, layout_height=1000, layout_width=1000, max_display=20):
    """Style can be dot or hist"""
    shap.initjs() 
    
    df = df_all.copy()

    df["dataset_type_classification"] = ""
    if len(df[df[references_we_compare]==True])>0:
        train_idx1, test_idx1 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df[references_we_compare]==True].index, groups=df[df[references_we_compare]==True]['subjectID']))
        df.loc[df[df[references_we_compare]==True].iloc[train_idx1].index, "classification_dataset_type"] = "Train-1"
        df.loc[df[df[references_we_compare]==True].iloc[test_idx1].index, "classification_dataset_type"] = "Test-1"

    if len(df[df[references_we_compare]==False])>0:
        train_idx2, test_idx2 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df[references_we_compare]==False].index, groups=df[df[references_we_compare]==False]['subjectID']))
        df.loc[df[df[references_we_compare]==False].iloc[train_idx2].index, "classification_dataset_type"] = "Train-2"
        df.loc[df[df[references_we_compare]==False].iloc[test_idx2].index, "classification_dataset_type"] = "Test-2"

        
    df.loc[df[references_we_compare]==True, "classification_label"] = 1
    df.loc[df[references_we_compare]==False, "classification_label"] = -1

    df_train = df[df.classification_dataset_type.str.startswith("Train")]
    df_test = df[df.classification_dataset_type.str.startswith("Test")]
    
    X_train = df_train[feature_cols]
    y_train = df_train.classification_label.values.astype('int')
    X_test = df_test[feature_cols]
    y_test = df_test.classification_label
    
    m = RandomForestClassifier(n_estimators=140, random_state=0, max_samples=0.8)
    m.fit(X_train, y_train)
    
    fig, ax = plt.subplots()
    
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X_train)
    
    feature_names = list(map(nice_name, X_train.columns))
    if not website:
        sns.set_style("whitegrid")
        
        if style == "dot":
            shap.summary_plot(shap_values[0], features=X_train, feature_names=feature_names, show=True, max_display=max_display)
        elif style == "hist":
            shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, class_names=["reference", "other"], show=True, max_display=max_display)


        return list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1]
    else:
        if style == "dot":
            shap.summary_plot(shap_values[0], features=X_train, show=show, max_display=max_display)

            fig =  mpl_to_plotly(fig)
            
            fig.update_xaxes(title="SHAP value (impact on model output)", 
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
            fig.update_yaxes(title="Features", 
                             tickmode='array',
                             tickvals=list(range(0, min(20, len(feature_names)))),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                             ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))][::-1])),
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
            fig.update_layout(height=layout_height, width=layout_width,
                            #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(l=0, r=0, b=0, pad=0),
                            title_text="Classification Important Features")
            
            #fig.update_traces(marker=dict(color="red"), selector=dict(type="scatter", mode="marker"))
#             fig.for_each_trace(
#                 lambda trace: trace.update(marker_symbol="square") if trace.name == "trace 39" else (),
#             )
            return fig
        elif style == "hist":
            shap.summary_plot(shap_values, features=X_train, class_names=["reference", "other"], show=show, max_display=max_display)
            
            fig =  mpl_to_plotly(fig)
            
            fig.update_xaxes(title="SHAP value (impact on model output)", 
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
            fig.update_yaxes(title="Features", 
                             tickmode='array',
                             tickvals=list(range(0, min(20, len(feature_names)))),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                             ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))][::-1])),
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
            fig.update_layout(height=layout_height, width=layout_width,
                            #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(l=0, r=0, b=0, pad=0),
                            title_text="Classification Important Features")
            return fig