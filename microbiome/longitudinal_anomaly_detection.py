from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import svm
from microbiome.helpers import df2vectors
import pandas as pd
import numpy as np
import scipy.stats as stats


def pi_anomaly_detection(estimator, df, feature_columns, degree, df_new=None):
    """
    Prediction Interval anomaly detection. We consider samples outside prediction interval to be outliers.
    
    Parameters
    ----------
    estimator: sklearn mordl, etc.
        Estimator used for microbiome trajectory.
    df: pd.DataFrame
        Dataframe containing data
    feature_columns: float
        List of taxa on which the model/estimator was trained on (usualy)
    degree: int
        Degree of the microbiome trajectory poly line.
    
    Returns
    -------
    outliers :list
        List of sampleIDs that are anomalous.
    """
    def _inner_pi_anomaly_detection(df_model, df_model_new, degree):
        """ Brain of this anomaly detection
        Parameters
        ----------
        df_model: pd.DataFrame
            Dataframe containing x-axis (age_at_collection, which we call y), y-axis (MMI, which we call y_pred), and unique sample IDs (sampleID)
        degree: float
            Polynomial degree of this microbiome trajectory
            
        Returns
        -------
        outliers :list
            List of sampleIDs that are anomalous.
        """
        limit_age_max = int(max(df_model["y"]))+1

        equation = lambda a, b: np.polyval(a, b) 

        # Stats
        p, cov = np.polyfit(df_model["y"].values, df_model["y_pred"].values, degree, cov=True)           # parameters and covariance from of the fit of 1-D polynom.
        y_model = equation(p, df_model["y"].values)                                   # model using the fit parameters; NOTE: parameters here are coefficients

        # Statistics
        n = df_model["y_pred"].values.size                                            # number of observations
        m = p.size                                                 # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands

        # Estimates of Error in Data/Model
        resid = df_model["y_pred"].values - y_model                           
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
    
        x2 = np.linspace(0, limit_age_max, limit_age_max+1) #np.linspace(np.min(x), np.max(x), 100)
        y2 = equation(p, x2)

        # Prediction Interval
        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(df_model["y"].values))**2 / np.sum((df_model["y"].values - np.mean(df_model["y"].values))**2))   
        pi_mean       = np.mean(pi)
        pi_median     = np.median(pi)
        
        # highlight_outliers - list of outliers to highlight, or True or False to plot all
        p_lower, _ = np.polyfit(x2, y2-pi, degree, cov=True)
        p_upper, _ = np.polyfit(x2, y2+pi, degree, cov=True)  
        outliers = set()
        for i in range(len(df_model_new["y"].values)):
            if df_model["y_pred"].values[i] > equation(p_upper, df_model_new["y"].values[i]) or df_model["y_pred"].values[i] < equation(p_lower, df_model_new["y"].values[i]):
                outliers.add(str(df_model_new.iloc[i]["sampleID"]))

        
        return list(outliers)
    
    df = df.sort_values(by="age_at_collection")
    X, y = df2vectors(df, feature_columns)
    y_pred = estimator.predict(X)
    
    if df_new is None:
        df_new = df.copy()
        
    df_new = df_new.sort_values(by="age_at_collection")
    X_new, y_new = df2vectors(df_new, feature_columns)
    y_pred_new = estimator.predict(X_new)
    
    df_model = pd.DataFrame(data={"y":y, "y_pred":y_pred, 'sampleID': df.sampleID})
    df_model_new = pd.DataFrame(data={"y":y_new, "y_pred":y_pred_new, 'sampleID': df_new.sampleID})
    
    outliers = _inner_pi_anomaly_detection(df_model, df_model_new, degree)
        
    return outliers    



def lpf_anomaly_detection(estimator, df, feature_columns, num_of_stds, window):
    """
    Low-pass filter (LPF) to detect anomalies in a time series. We unite all longitudinal data into one signal. 
    If we had more smaples per subjects, we could do this lpf anomaly detection for every subject separately.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing data
    num_of_stds: float
        Number of standard deviations of z-score that will be considered anomalous if higher.
    window: int
        Number of observations used for calculating the statistic.
    
    Returns
    -------
    outliers :list
        List of sampleIDs that are anomalous.
    """
    def _inner_lpf_anomaly_detection(df, num_of_stds, window):
        """ Brain of this anomaly detection
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing x-axis (age_at_collection, which we call y), y-axis (MMI, which we call y_pred), and unique sample IDs (sampleID)
        num_of_stds: float
            Number of standard deviations of z-score that will be considered anomalous if higher.
        window: int
            Number of observations used for calculating the statistic.
            
        Returns
        -------
        outliers :list
            List of sampleIDs that are anomalous.
        """
        df = df.sort_values(by="y")
        df["y_pred_rolling_avg"]=df["y_pred"].rolling(window=window, min_periods=1, center=True).mean()
        df["y_pred_rolling_std"]=df["y_pred"].rolling(window=window, min_periods=1, center=True).std(skipna=True)
        df["lpf_anomaly"]= abs(df["y_pred"]-df["y_pred_rolling_avg"])>(num_of_stds*df["y_pred_rolling_std"])
        return df[df["lpf_anomaly"]==True].sampleID.values.tolist()
    
    X, y = df2vectors(df, feature_columns)
    y_pred = estimator.predict(X)
    df_model = pd.DataFrame(data={"y":y, "y_pred":y_pred, 'sampleID': df.sampleID})
    
    outliers = _inner_lpf_anomaly_detection(df_model, num_of_stds, window)
        
    return outliers    


def if_anomaly_detection(estimator, df, feature_columns, outliers_fraction, window, anomaly_columns):
    """
    Isolation Forest (IF) to detect anomalies in a time series. We unite all longitudinal data into one signal. 
    If we had more smaples per subjects, we could do this lpf anomaly detection for every subject separately.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing data
    num_of_stds: float
        Number of standard deviations of z-score that will be considered anomalous if higher.
    window: int
        Number of observations used for calculating the statistic.
    anomaly_column: list
        Two options: `y_pred`, which ignores the timebox, and `y_pred_zcore`, or both.
    
    Returns
    -------
    outliers :list
        List of sampleIDs that are anomalous.
    """
    def _inner_if_anomaly_detection(df, outliers_fraction, anomaly_columns):
        """ Brain of this anomaly detection
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing x-axis (age_at_collection, which we call y), y-axis (MMI, which we call y_pred), and unique sample IDs (sampleID)
        """
        # Subset the dataframe by desired columns
        dataframe_filtered_columns = df[anomaly_columns]
        
        # Scale the column that we want to flag for anomalies
        min_max_scaler = StandardScaler()
        np_scaled = min_max_scaler.fit_transform(dataframe_filtered_columns)
        scaled_time_series = pd.DataFrame(np_scaled)
        
        # train isolation forest 
        model =  IsolationForest(contamination=outliers_fraction)
        model.fit(scaled_time_series)
        
        #Generate column for Isolation Forest-detected anomalies
        df["if_anomaly"] = model.predict(scaled_time_series)
        df["if_anomaly"] = df["if_anomaly"].map( {1: False, -1: True} )
        return df[df["if_anomaly"]==True].sampleID.values.tolist()
    
    X, y = df2vectors(df, feature_columns)
    y_pred = estimator.predict(X)
    df_model = pd.DataFrame(data={"y":y, "y_pred":y_pred, 'sampleID': df.sampleID})
    df_model = df_model.sort_values(by="y")
    df_model["y_pred_rolling_avg"] = df_model["y_pred"].rolling(window=window, min_periods=1, center=True).mean()
    df_model["y_pred_zscore"]= df_model["y_pred"]-df_model["y_pred_rolling_avg"]
    
    outliers = _inner_if_anomaly_detection(df_model, outliers_fraction, anomaly_columns=anomaly_columns)
        
    return outliers    



def svm_anomaly_detection(estimator, df, feature_columns, outliers_fraction, window, anomaly_columns):
    """
    Isolation Forest (IF) to detect anomalies in a time series. We unite all longitudinal data into one signal. 
    If we had more smaples per subjects, we could do this lpf anomaly detection for every subject separately.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing data
    num_of_stds: float
        Number of standard deviations of z-score that will be considered anomalous if higher.
    window: int
        Number of observations used for calculating the statistic.
    anomaly_columns: list
        Two options: `y_pred`, which ignores the timebox, and `y_pred_zcore`, or both.
    
    Returns
    -------
    outliers :list
        List of sampleIDs that are anomalous.
    """
    def _inner_svm_anomaly_detection(df, outliers_fraction, anomaly_columns):
        """ Brain of this anomaly detection
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing x-axis (age_at_collection, which we call y), y-axis (MMI, which we call y_pred), and unique sample IDs (sampleID)
        """
        # Subset the dataframe by desired columns
        dataframe_filtered_columns = df[anomaly_columns]
        
        # Scale the column that we want to flag for anomalies
        min_max_scaler = StandardScaler()
        np_scaled = min_max_scaler.fit_transform(dataframe_filtered_columns)
        scaled_dataframe = pd.DataFrame(np_scaled)
        
        # Remove any NaN from the dataframe
        scaled_dataframe =scaled_dataframe.dropna()
        
        # Train the One Class SVM
        model = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
        model.fit(scaled_dataframe)
        
        # Generate column for detected anomalies
        scaled_dataframe["svm_anomaly"] = pd.Series(model.predict(scaled_dataframe)).map({1: False, -1: True})
        df["svm_anomaly"] = scaled_dataframe["svm_anomaly"]
        return df[df["svm_anomaly"]==True].sampleID.values.tolist()
    
    X, y = df2vectors(df, feature_columns)
    y_pred = estimator.predict(X)
    
    df_model = pd.DataFrame(data={"y":y, "y_pred":y_pred, 'sampleID': df.sampleID})
    df_model = df_model.sort_values(by="y")
    df_model["y_pred_rolling_avg"] = df_model["y_pred"].rolling(window=window, min_periods=1, center=True).mean()
    df_model["y_pred_zscore"]= df_model["y_pred"]-df_model["y_pred_rolling_avg"]
    
    outliers = _inner_svm_anomaly_detection(df_model, outliers_fraction, anomaly_columns=anomaly_columns)
        
    return outliers    