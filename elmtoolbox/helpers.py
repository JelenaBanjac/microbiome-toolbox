


def df2vectors(_df, feature_cols=None, time_col="age_at_collection"):
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
    time_col: str
        Time column (label) is explicitly specified in case we want to handle it in months or years, and not in days (as in age_at_collection).

    Returns
    -------
    X: list of lists
        Feature matrix with dimension (n_samples, n_features). 
    y: list
        Label vector with dimension (n_samples,)
    """
    
    # labels, what we want to predict -> age of the infant
    y = _df[time_col]
    
    _df = _df[feature_cols]
    
    # features, what is given to us -> bacteria info
    X = _df.values

    return X, y