import matplotlib
matplotlib.use('agg')
import base64
import shap
import pandas as pd
import plotly.graph_objects as go
import gc
import shap
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
from io import BytesIO


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




def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

def _plot_confusion_matrix(cm, title, classes=['False', 'True'],
                          cmap=plt.cm.Blues, save=False, saveas="MyFigure.png", website=False):
    
    # print Confusion matrix with blue gradient colours
    plt.rcParams.update({'font.size': 15})
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1%'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save:
        plt.savefig(saveas, dpi=100)

    if not website:
        plt.show()
    else:
        return fig_to_uri(plt)

def plot_confusion_matrix(cm, classes, title):
    # cm : confusion matrix list(list)
    # classes : name of the data list(str)
    # title : title for the heatmap
    

    data = go.Heatmap(z=cm, y=classes, x=classes, colorscale= ["#ffffff", "#F52757"],)

    annotations = []

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.1%'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        annotations.append(
                {
                    "x": classes[i],
                    "y": classes[j],
                    "font": {"color": "white" if cm[i, j] > thresh else "black", "size":20},
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
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig

def two_groups_analysis(df_all, feature_cols, references_we_compare, test_size=0.5, n_splits=5, nice_name=lambda x: x, style="dot", show=False, website=True, layout_height=1000, layout_width=1000, max_display=20):
    """Style can be dot or hist"""
    if show: 
        print("two_groups_analysis")
        shap.initjs() 
    
    df = df_all.copy()
    
    # df1 = df[df[references_we_compare].astype(str)=='True']
    # df2 = df[df[references_we_compare].astype(str)!='True']
    df1 = df[df[references_we_compare]==True]
    df2 = df[df[references_we_compare]==False]
    print(references_we_compare)
    print(df[references_we_compare].dtype)
    print(df[references_we_compare].unique())
    print(len(df), len(df1), len(df2))
    
    min_samples = df[references_we_compare].value_counts(sort=True).values[-1]
    min_group   = df[references_we_compare].value_counts(sort=True).index[-1]
    
    #print(min_samples/len(df))
    if min_group==True:
        df2 = df2.sample(n=int(min_samples))  #+(len(df)-min_samples)*0.1)
    else:
        df1 = df1.sample(n=int(min_samples))
    
    df = pd.concat([df1, df2])

    df["dataset_type_classification"] = ""
    if len(df[df[references_we_compare]==True])>0:
        train_idx1, test_idx1 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df[references_we_compare]==True].index, groups=df[df[references_we_compare]==True]['subjectID']))
        df.loc[df[df[references_we_compare]==True].iloc[train_idx1].index, "classification_dataset_type"] = "Train-1"
        df.loc[df[df[references_we_compare]==True].iloc[test_idx1].index, "classification_dataset_type"] = "Test-1"

    if len(df[df[references_we_compare]==False])>0:
        train_idx2, test_idx2 = next(GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=7).split(df[df[references_we_compare]==False].index, groups=df[df[references_we_compare]==False]['subjectID']))
        df.loc[df[df[references_we_compare]==False].iloc[train_idx2].index, "classification_dataset_type"] = "Train-2"
        df.loc[df[df[references_we_compare]==False].iloc[test_idx2].index, "classification_dataset_type"] = "Test-2"

    #print(df.classification_dataset_type.value_counts())
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

    y_test_pred = m.predict(X_test)

    if isinstance(y_test, pd.Series):
        y_test = np.array(y_test.values, dtype=type(y_test_pred[0]))

    cm_test = confusion_matrix(y_test_pred, y_test)
    acc = 100*(cm_test[0][0]+cm_test[1][1]) / (sum(cm_test[0]) + sum(cm_test[1]))

    max_limit = max(20, min(max_display, len(feature_names)))

    ret_val = f'Total OTHER detected in test set: **{cm_test[1][1]:.2f} / {cm_test[1][1]+cm_test[1][0]:.2f}**\n'
    ret_val += f'Total REFERENCE transactions detected in test set: **{cm_test[0][0]:.2f} / {cm_test[0][1]+cm_test[0][0]:.2f}**\n'
    ret_val += f'Probability to detect a OTHER in the test set: **{cm_test[1][1]/(cm_test[1][1]+cm_test[1][0]):.2f}**\n'
    ret_val += f'Probability to detect a REFERENCE in the test set: **{cm_test[0][0]/(cm_test[0][1]+cm_test[0][0]):.2f}**\n'
    ret_val += f"Accuracy on the test set: **{acc:.2f}%** \n"


    if not website:
        sns.set_style("whitegrid")
        
        # if style == "dot":
        #     shap.summary_plot(shap_values[0], features=X_train, feature_names=feature_names, show=show, max_display=max_display)
        # elif style == "hist":
        #     shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, class_names=["reference", "other"], show=show, max_display=max_display)

        # if show:
        #     plot_confusion_matrix(m, X_test, y_test)  
        #     plt.show() 
        #     _plot_confusion_matrix(cm_test,"Confusion matrix", ['False', 'True'])

        #     print(ret_val)

        output = dict(top_features_list=list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1],
                      accuracy=acc)
        del df
        gc.collect()
        return output
    else:
        if style == "dot":
            shap.summary_plot(shap_values[0], features=X_train, show=show, max_display=max_display)

            fig =  mpl_to_plotly(fig)
            
            fig.update_xaxes(title="SHAP value (impact on model output)", 
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
            fig.update_yaxes(title="Features", 
                             tickmode='array',
                             tickvals=list(range(0, max_limit)),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                             ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:max_limit][::-1])),
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
            fig.update_layout(height=layout_height, width=layout_width,
                            #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(l=0, r=0, b=0, pad=0),
                            title_text="Classification Important Features")
            

            img_src = plot_confusion_matrix(cm_test, ['other', 'reference'], "Confusion matrix")
            plt.clf()
            del df
            gc.collect()
            return fig, img_src, ret_val

        elif style == "hist":
            shap.summary_plot(shap_values, features=X_train, class_names=["reference", "other"], show=show, max_display=max_display)
            
            fig =  mpl_to_plotly(fig)
            
            fig.update_xaxes(title="SHAP value (impact on model output)", 
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray') 
            fig.update_yaxes(title="Features", 
                             tickmode='array',
                             tickvals=list(range(0, max_limit)),  #list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:min(20, len(feature_names))], # 
                             ticktext=list(map(nice_name, list(X_train.columns[np.argsort(np.abs(shap_values[0]).mean(0))])[::-1][:max_limit][::-1])),
                            showline=True, linecolor='lightgrey', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', showspikes=True, spikecolor='gray')  
            fig.update_layout(height=layout_height, width=layout_width,
                            #paper_bgcolor="white",#'rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(l=0, r=0, b=0, pad=0),
                            title_text="Classification Important Features")

            img_src = plot_confusion_matrix(cm_test, ['other', 'reference'], "Confusion matrix")
            plt.clf()
            del df
            gc.collect()
            return fig, img_src, ret_val