# country colors
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor

col_rus = "#E3211B"
col_est = "#0095C8"
col_fin = "#00A24A"
alpha   = 0.7

n_splits = 5
test_size = 0.5

Regressor = RandomForestRegressor
parameters = {#"loss_function": "MAE",
                "random_state": 42,}
param_grid = { 'n_estimators': [50, 100] }

# Regressor = CatBoostRegressor
# parameters = {"loss_function": "MAE",
#               "random_state": 42,
#              "verbose":False}
# param_grid = { 'learning_rate': [0.5, 0.1],
#                #"depth": [5, 10],
#                #"iterations": [100, 500, 1000]
#              }