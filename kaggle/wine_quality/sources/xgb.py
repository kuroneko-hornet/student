'''
winequality-red.csvに対してxgboost, optunaを適用
'''

from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def div_max(df,column):
    df[column] = df[column]/df[column].max()
    return df

def preprocess(df):
    df = df.drop(['citric acid','free sulfur dioxide','fixed acidity','alcohol'],axis=1)
    df[target_column] = df[target_column] /10
    return df

def train(df,max_depth,learning_rate,num_round,gamma,min_childe_weigh,colsample_bytree,alpha,how_objective):
    train_x = df.drop(target_column,axis=1)
    train_y = df[target_column]
    dtrain = xgb.DMatrix(train_x,label=train_y)
    param = { 'max_depth':max_depth,'learning_rate':learning_rate,'objective':how_objective,'gamma':gamma,'min_childe_weigh':min_childe_weigh,'colsample_bytree':colsample_bytree,'alpha':alpha,'silent':1}
    bst = xgb.train(param,dtrain,num_round)
    return bst

def predict(bst,df):
    return bst.predict(xgb.DMatrix(df))*10

def objective(df, df_test, y, how_objective, trial):
    #目的関数
    max_depth = trial.suggest_int('max_depth',1,30)
    learning_rate = trial.suggest_uniform('learning_rate',0.0,1.0)
    round_num = trial.suggest_int('round_num',1,30)
    gamma = trial.suggest_uniform('gamma',0.0,10.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree',0.0,1.0)
    min_childe_weigh = trial.suggest_uniform('min_childe_weigh',0.0,10.0)
    alpha = trial.suggest_uniform('alpha',0.0,10.0)
    bst = train(df,max_depth,learning_rate,round_num,gamma,min_childe_weigh,colsample_bytree,alpha,how_objective)
    answer = predict(bst,df_test).round().astype(int)
    score = accuracy_score(answer,y.astype(int))
    return 1.0 - score

def main():
    global target_column
    df_original = pd.read_csv("/Users/pc1013/Desktop/first/XGBoost_optuna/df/winequality-red.csv")
    df = preprocess(df_original)
    (df_train, df_test) = train_test_split(df, test_size = 0.1, random_state = 666)
    y = df_test[target_column]*10
    df_test = df_test.drop(target_column,axis=1)

    how_objective = 'reg:logistic'
    #optunaの前処理
    obj_f = partial(objective, df_train, df_test, y, how_objective)
    #セッション作成
    study = optuna.create_study()
    #回数
    study.optimize(obj_f, n_trials=200)

    max_depth = study.best_params['max_depth']
    learning_rate = study.best_params['learning_rate']
    round_num = study.best_params['round_num']
    colsample_bytree = study.best_params['colsample_bytree']
    min_childe_weigh = study.best_params['min_childe_weigh']
    gamma = study.best_params['gamma']
    alpha = study.best_params['alpha']

    bst = train(df_train,max_depth,learning_rate,round_num,gamma,min_childe_weigh,colsample_bytree,alpha,how_objective)
    print('\nparams :',study.best_params)
    answer = predict(bst,df_train.drop(target_column,axis=1)).round().astype(int)
    print('train score :',accuracy_score(answer,df_train[target_column]*10))
    answer = predict(bst,df_test).round().astype(int)
    xgb.plot_importance(bst)
    print('test score :',accuracy_score(answer,y))

if __name__ == '__main__':
    target_column = 'quality'
    main()
