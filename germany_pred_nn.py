import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import check_array

from sklearn import metrics
#import seaborn as sns


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def make_train_test():
    df_all = pd.read_csv('germany_2016_with_features.csv')
    #df_all = df_all.loc[df_all['dt']>'2016-01-01 23:00:00']

    df_train, df_test = train_test_split(df_all, test_size=0.3)

    return df_train, df_test


def preprocessing(df):
    #df = df_train
    #df = df_test

    predictors = ['v1', 'v2', 'v_50m', 'h1', 'h2', 'z0', 'SWTDN', 'SWGDN', 'T', 'rho', 'p', 'day_of_year', 'month', 'hr_of_day', 'day_of_month']

    X = df[predictors]
    y = df.iloc[:,1:2]

    features = X.columns.values
    n_cols = X.shape[1]

    #standardize the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, y, n_cols, features


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def make_model():

    model = Sequential()
    model.add(Dense(15, input_dim= 15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mse', optimizer='adam', metrics=['mape', rmse, coeff_determination])

    model.summary()

    return model

def predictor():

    history = model.fit(X_train, y_train, validation_split=0.3, epochs=300, batch_size=32, verbose=2)

    ax = plt.plot(history.history['rmse'])
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    sum = sum(history.history['rmse'])
    len = len(history.history['rmse'])
    rmse = sum(history.history['rmse']) / len(history.history['rmse'])

    ax = plt.plot(history.history['coeff_determination'])
    plt.ylabel('R squared')
    plt.xlabel('Epoch')
    r2 = sum(history.history['coeff_determination']) / len(history.history['coeff_determination'])


    ax = plt.plot(history.history['mean_absolute_percentage_error'])
    plt.ylabel('MAPE')
    plt.xlabel('Epoch')
    mape = sum(history.history['mean_absolute_percentage_error']) / len(history.history['mean_absolute_percentage_error'])

    return history

def test_settings(model):

    # define the grid search parameters
    batch_size = [10, 20, 32, 40, 60]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def k_cross_val(model, X_train, y_train):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    #scores = sklearn.model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=['r2', 'mean_squared_error'])
    mse_kfold_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    print(mse_kfold_scores)

    return mse_kfold_scores

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def do_it():
    df_train, df_test = make_train_test()
    X_train, y_train, n_cols, features = preprocessing(df=df_train)
    X_test, y_test, n_cols, features = preprocessing(df=df_test)

    model = KerasRegressor(build_fn=make_model, epochs=100, batch_size=20, verbose=2)

    test_settings(model)

    mse_kfold_scores = k_cross_val(model, X_train, y_train)

    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)
    r2 = r2_score(y_test, pred_y)
    mse = mean_squared_error(y_test, pred_y)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, pred_y)

    print("KERAS: R2 : {0:f}, MSE : {1:f}".format(r2, rmse))

    plt.scatter(y=pred_y, x=y_test, alpha=0.5)
    plt.title("comparison of actual and predicted wind generation (MWh)")
    plt.xlabel("actual wind generation (MWh)")
    plt.ylabel("predicted wind generation (MWh)")
    plt.show()

    plt.scatter(y=pred_y, x=rmse, alpha=0.5)


    df_results = pd.DataFrame(y_test)
    df_results['wind_pred'] = pred_y
    df_results['abs_error'] = abs(df_results['DE_wind_generation_actual'] - df_results['wind_pred'])
    df_results['perc_error'] = df_results['abs_error'] / abs(df_results['DE_wind_generation_actual'])

    df_results['perc_error'].mean()

    plt.scatter(x=df['v1'], y = df['DE_wind_generation_actual'], alpha=0.5)
    plt.scatter(x=df['v2'], y = df['DE_wind_generation_actual'], alpha=0.5)
    plt.scatter(x=df['v_50m'], y = df['DE_wind_generation_actual'], alpha=0.5)
    plt.legend(['wind speed at displacement height +2m','wind speed at displacement height +10m', 'wind speed at 50m above ground'])
    plt.xlabel('wind speed (m/s)')
    plt.ylabel('actual wind generation (MW)')




'''
def build_NN(model, X_train, y_train):
    history = model(X_train, y_train, epochs=150, batch_size=5, verbose=1, validation_split=0.2)

    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=fdfdfa
    plt.show()





    # model.fit(X_train, y_train, batch_size=10, epochs=100)
    model.evaluate(X_test, y_test)[1]
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy * 100))
'''