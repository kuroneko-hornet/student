#optuna
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess(df):
    df['Fare'] = df ['Fare'].fillna(df['Fare'].mean())
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    #df['Embarked'] = df['Embarked'].map( {'S':0,'C':1,'Q':2,'Unknown':3} ).astype(int)
    df = df.drop(['Cabin','Name','PassengerId','Ticket','Embarked'],axis=1)
    return df

def train(df,max_depth,learning_rate,num_round,gamma,min_childe_weigh,colsample_bytree,alpha):
    train_x = df.drop('Survived',axis=1)
    train_y = df.Survived
    dtrain = xgb.DMatrix(train_x,label=train_y)
    param = { 'max_depth':max_depth,'learning_rate':learning_rate,'objective':'reg:logistic','gamma':gamma,'min_childe_weigh':min_childe_weigh,'colsample_bytree':colsample_bytree,'alpha':alpha}
    bst = xgb.train(param,dtrain,num_round)
    return bst

def predict(bst,df):
    return bst.predict(xgb.DMatrix(df))

def objective(df, df_test, y,trial):
    #目的関数
    max_depth = trial.suggest_int('max_depth',1,30)
    learning_rate = trial.suggest_uniform('learning_rate',0.0,1.0)
    round_num = trial.suggest_int('round_num',1,30)
    gamma = trial.suggest_uniform('gamma',0.0,1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree',0.0,1.0)
    min_childe_weigh = trial.suggest_uniform('min_childe_weigh',0.0,1.0)
    alpha = trial.suggest_uniform('alpha',0.0,1.0)
    bst = train(df,max_depth,learning_rate,round_num,gamma,min_childe_weigh,colsample_bytree,alpha)
    answer = predict(bst,df_test).round().astype(int)
    score = accuracy_score(answer.round(),y)
    return 1.0 - score

def main():
    df_original = pd.read_csv("/Users/pc1013/Desktop/first/XGBoost_optuna/df/train.csv")
    df_test = preprocess(df_original.tail(100))
    df_train = preprocess(df_original.head(791))
    y = df_test['Survived']
    df_test = df_test.drop('Survived',axis=1)
    
    #optunaの前処理
    obj_f = partial(objective, df_train, df_test, y)
    #セッション作成
    study = optuna.create_study()
    #回数
    
    study.optimize(obj_f, n_trials=100)
    
    max_depth = study.best_params['max_depth']
    learning_rate = study.best_params['learning_rate']
    round_num = study.best_params['round_num']
    subsample = 1
    colsample_bytree = study.best_params['colsample_bytree']
    min_childe_weigh = study.best_params['min_childe_weigh']
    gamma = study.best_params['gamma']
    alpha = study.best_params['alpha']
    lamb = 1
    
    
    bst = train(df_train,max_depth,learning_rate,round_num,gamma,min_childe_weigh,colsample_bytree,alpha)
    print('\nparams :',study.best_params)
    answer = predict(bst,df_train.drop('Survived',axis=1)).round().astype(int)
    print('train score :',accuracy_score(answer.round(),df_train['Survived']))
    answer = predict(bst,df_test).round().astype(int)
    xgb.plot_importance(bst)
    print('test score :',accuracy_score(answer.round(),y))

if __name__ == '__main__':
    main()

