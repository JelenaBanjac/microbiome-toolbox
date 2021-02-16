
import shap
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np

def reference_analysis(df_all, feature_cols, nice_name=lambda x: x, style="dot", show=False, website=True, layout_height=1000, layout_width=1000, max_display=20):
    """Style can be dot or hist"""
    shap.initjs() 
    
    df = df_all.copy()
    
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