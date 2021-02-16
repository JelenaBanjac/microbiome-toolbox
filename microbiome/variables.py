# country colors
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

col_rus = "#E3211B"
col_est = "#0095C8"
col_fin = "#00A24A"
alpha   = 0.7

Regressor = CatBoostRegressor
parameters = {"loss_function": "MAE",
              "random_state": 42,
              "allow_writing_files": True,
             "verbose":False}
param_grid = { 'learning_rate': [0.5, 0.1],
               #"depth": [5, 10],
               #"iterations": [100, 500, 1000]
             }