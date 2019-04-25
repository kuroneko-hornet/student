import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import sys

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from functools import partial
import optuna


def preprocess(train_x, test):

    concat_data = pd.concat([train_x, test], axis=0)

    # only 'object' data type
    concat_data =  data.select_dtypes(include=['object'])
    concat_data = concat_data.fillna("missing")
    concat_data = pd.get_dummies(concat_data)

    # except for 'object'
    number_data = data.select_dtypes(exclude=['object'])
    number_data = number_data.fillna(-999)

    attribute_data = pd.concat([number_data, concat_data], axis=1)
    attribute_data = attribute_data.drop(['Id'], axis=1)

    train_X = attribute_data.iloc[:train_x.shape[0], :]
    test_X = attribute_data.iloc[train_x.shape[0]:, :]

    return train_X, test_X


def objective(df_x, df_y, test_x, test_y, trial):
    
    (test_x, x, test_y, y) = train_test_split(test_x, test_y, test_size = 0.5, random_state = None)

    #目的関数
    params = {
        'num_leaves' : trial.suggest_int('num_leaves',2,8),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.01,0.2), 
        'n_estimators' : trial.suggest_int('n_estimators', 50, 5000),
        'num_iteration' : trial.suggest_int('num_iteration', 50, 150),
        'feature_fraction' : trial.suggest_uniform('feature_fraction',0.1,0.9),
        'bagging_fraction' : trial.suggest_uniform('bagging_fraction',0.1,0.9),
        'bagging_freq' : trial.suggest_int('bagging_freq', 0, 10),
        'max_bin' : trial.suggest_int('max_bin', 150, 250),
        'bagging_seed' : trial.suggest_int('bagging_seed', 7, 7),
        'feature_fraction_seed' : trial.suggest_int('feature_fraction_seed', 7, 7),
        'verbose' : trial.suggest_int('verbose', -1,-1)
    }
    lightgbm = LGBMRegressor(**params)

    lightgbm.fit(df_x, df_y)

    answer = lightgbm.predict(test_x)
    score = np.sqrt(mean_squared_log_error(answer,test_y))
    return score



    '''
    target = 'SalePrice'
    train_original['MSSubClass'] = train_original['MSSubClass'].apply(str)
    train_original = train_original[~((train_original['GrLivArea'] > 4000))]
    train_original = train_original[~((train_original['GrLivArea'] > 4000) & (train_original['SalePrice'] < 300000))]
    train_original = train_original[~(train_original['LotFrontage'] > 300)]
    train_original = train_original[~(train_original['LotArea'] > 100000)]
    train_original = train_original[~((train_original['OverallQual'] == 10) & (train_original['SalePrice'] < 200000))]
    train_original = train_original[~((train_original['OverallCond'] == 2) & (train_original['SalePrice'] > 300000))]
    train_original = train_original[~((train_original['OverallCond'] > 4) & (train_original['SalePrice'] > 700000))]
    train_original = train_original[~((train_original['YearBuilt'] < 1900) & (train_original['SalePrice'] > 200000))]
    train_original = train_original[~((train_original['YearRemodAdd'] < 2010) & (train_original['SalePrice'] > 600000))]
    train_original = train_original[~(train_original['BsmtFinSF1'] > 5000)]
    train_original = train_original[~(train_original['TotalBsmtSF'] > 5000)]
    '''

def main(train_original, test_original):

    target = 'quality'

    split_train_

    train_y = train_original[target]

    train_x, df_test = preprocess(train_original.drop([target], axis = 1), test_original)
    (train_x, train_t_x, train_y, train_t_y) =\
        train_test_split(df_train, df_y, test_size = 0.2, random_state = None)
    # # make verifivation data
    # (train_t_x, x, train_t_y, y) = train_test_split(train_t_x, train_t_y, test_size = 0.5, random_state = None)

    print(train_x.shape)
    print(train_t_x.shape)
    print(train_y.shape)
    print(train_t_y.shape)

    use_optuna = False

    if isUsing_optuna:
        obj_f = partial(objective, train_x, train_y, train_t_x, train_t_y)
        # セッション作成
        study = optuna.create_study()
        # 回数
        study.optimize(obj_f, n_trials=30)
        params = study.best_params
    else:
        params = {
            "num_leaves":4,
            "learning_rate":0.01, 
            "n_estimators":5000,
            "max_bin":200, 
            "bagging_fraction":0.75,
            "bagging_freq":5, 
            "bagging_seed":7,
            "feature_fraction":0.2,
            "feature_fraction_seed":7,
            "verbose":-1
        }

    lightgbm = LGBMRegressor(**params)
    lightgbm.fit(train_x, train_y)

    train_answer = lightgbm.predict(train_x)
    print('train score :',np.sqrt(mean_squared_log_error(train_answer,train_y)))
    train_t_answer = lightgbm.predict(train_t_x)
    print('train_t score :',np.sqrt(mean_squared_log_error(train_t_answer,train_t_y)))
    t_answer = lightgbm.predict(train_t_x)
    print('test score :',np.sqrt(mean_squared_log_error(t_answer,train_t_y)))
    

    # sub = pd.read_csv("sample_submission.csv")
    # sub["SalePrice"] = lightgbm.predict(df_test)
    # return sub



if __name__ == '__main__':
    target = 'quality'
    original = pd.read_csv("kaggle/wine_quality/dataframe/winequality-red.csv")
    train_original, test_original = original[:1000], original[1000:]
    test_original, test_answer = test_original.drop([target], axis=1), test_original[target]

    # print(train_original.head())
    # print(test_original.head())

    #answer = main(train_original, test_original)

    #answer.to_csv("lgbm.csv", index=False)
    '''
    f, ax = plt.subplots(figsize=[7,10])
    lgb.plot_importance(lightgbm, max_num_features=85, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')
    '''



