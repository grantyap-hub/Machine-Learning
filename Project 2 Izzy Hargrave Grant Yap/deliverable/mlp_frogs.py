from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from keras.optimizers import Adam
from load_newts import load_newts
from load_newts import eval_kfold_stub
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np


def train_model(x_vals, y_vals, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=56, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_vals, y_vals, epochs=1000, batch_size=100, verbose=1, use_multiprocessing=True, workers=-1)
    loss, accuracy = model.evaluate(x_vals, y_vals, verbose=1)
    print('Model Loss: %.2f, Accuracy: %.2f' % ((loss * 100), (accuracy * 100)))
    return model


# returns average f1-score over five cross-validation cycles
def cross_validate(x_vals, y_vals, hidden_units):
    kf = KFold(n_splits=5, shuffle=True)
    score = 0
    for train_idxs, test_idxs in kf.split(x_vals, y_vals):
        xtrain_this_fold = x_vals.loc[train_idxs]
        xtest_this_fold = x_vals.loc[test_idxs]
        ytrain_this_fold = y_vals.loc[train_idxs]
        ytest_this_fold = y_vals.loc[test_idxs]

        # train a model on this fold
        model = train_model(xtrain_this_fold, ytrain_this_fold, hidden_units)

        # test the model on this fold
        score += f1_score(ytest_this_fold, np.round(model.predict(xtest_this_fold), decimals=0))
    average_score = score / 5
    return average_score

def sweep_units(x_vals, y_vals, hidden_units):
    f1_sweep_results = np.zeros([1, hidden_units])
    for i in range(1, hidden_units):
        f1_sweep_results[0, i] = cross_validate(x_vals, y_vals, i)
    return f1_sweep_results


# main method
x_vals, y_vals = load_newts("frogs.csv", do_min_max=True)
print(y_vals)
print(type(y_vals))
print(cross_validate(x_vals, y_vals, 30))

# results = sweep_units(x_vals, y_vals, 30)
# print(results)
