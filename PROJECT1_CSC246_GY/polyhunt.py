# Grant Yap
# Email: gyap@u.rochester.edu

# m - integer - polynomial order (or maximum in autofit mode)

# gamma - float - regularization constant (use a default of 0)

# trainPath - string - a filepath to the training data

# modelOutput - string - a filepath where the best fit parameters will be saved, if this
# is not supplied, then you do not have to output any model parameters

# autofit - boolean - a flag which when supplied engages the order sweeping loop, when
# this flag is false (or not supplied) you should simply fit a polynomial of the given order
# and parameters. In either case, save the best fit model to the file specified by the
# modelOutput path, and print the RMSE/order information to the screen for the TA
# to read.

# info - boolean - if this flag is set, the program should print your name and contact
# information (of all members, if working in a team)

# OPTIONAL
# numFolds - the number of folds to use for cross validation
# devPath - a path to a held-out data set
# debug - a flag to turn on printing of extra information (for debugging)
# ... plus any others you wish.


import argparse
import string
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import random
import collections
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
                                'Add a polynomial order (m), gamma(g) '\
                                'trainpath, modelOutput, autofit, and/or info')
parser.add_argument('m', metavar='M', type = int, help = 'polynomial order (or maximum in autofit mode)')

parser.add_argument('gamma', metavar='g', type = float, help = 'regularization constant (use a default of 0)', default = 0)
parser.add_argument('trainPath', metavar='train', type = str, help = 'filepath to training data')
parser.add_argument('modelOutput', metavar='model', type = str, help = 'filepath where best fit is saved')
parser.add_argument('--autofit', action = 'store_true', default = False, help = 'a flag which when supplied engages the' + \
'order sweeping loop, when this flag is false (or not supplied) you should simply fit a polynomial of the given' + \
'order and parameters.')
parser.add_argument('--info', action = 'store_true', default = False, help = 'if this flag is set, the program should print' + \
'your name and contact information (of all members, if working in a team).')
parser.add_argument('--plotting', action = 'store_true', default = False, help = 'if this flag is set, the program should plot' + \
'a graph')
parser.add_argument('--folds', default = 5, type = int, help = 'if this flag is set, the program should create a' + \
'number of folds (default at 5)')
parser.add_argument('--trials', default = 5, type = int, help = 'if this flag is set, the program should create a' + \
'number of trials (default at 5)')

args = parser.parse_args()
# print(args.gamma)
# m
# g
# trainpath
# modeloutput
# autofit
# info
# plotting




df = pd.read_csv(args.trainPath, header = None, names = ['colA', 'colB'])
df_small = df.loc[:,['colA','colB']].dropna()
df_small = df_small.sample(frac=1)
x = df_small.colA.values.T
y = df_small.colB.values.T

def plotting(x, y, m, xlabel, ylabel, polynomial_degree):
    plt.figure()
    plt.plot(x, y, 'x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Vs f{xlabel}')
    xfine = np.linspace(np.min(x), np.max(x), 10*len(x))
    G = makePhi(xfine, polynomial_degree)
    yfine = np.dot(G, m)
    plt.plot(xfine, yfine, 'r--')
    plt.show()

def kfoldValidationAttempt(x, y, k, order, g):
    splitL = int(len(x)/k)
    curr_RMSE = []
    models = []
    for i in range(1, k+1):
        splitter = splitL*i
        if(i == 1):
            x_train = x[splitter:]
            x_test = x[:splitter]
            y_train = y[splitter:]
            y_test = y[:splitter]
        elif(i == k):
            splitter -= splitL
            x_train = x[:splitter]
            x_test = x[splitter:]
            y_train = y[:splitter]
            y_test = y[splitter:]
        else:
            correct_interval = splitter - splitL
            x_train = np.append(x[:correct_interval], x[splitter:])
            x_test = x[correct_interval:splitter]
            y_train = np.append(y[:correct_interval], y[splitter:])
            y_test = y[correct_interval:splitter]
        model = linearInversion(x_train, y_train, order, g)
        ypredict = predictedY(makePhi(x_test, order), model)
        curr_RMSE.append(RMSE(ypredict, y_test))
    return sum(curr_RMSE)/k


# make phi
def makePhi(x, polynomial_degree):
    G = np.zeros((len(x), polynomial_degree + 1))
    for power in range(polynomial_degree+1):
        G[:,power] = x**power
    return G

# make predicted Y
def predictedY(G, model):
    return np.dot(G, model)

# make regularized least squares regression model
def linearInversion(x, y, polynomial_degree, lam):
    G = makePhi(x, polynomial_degree)
    Gt = G.T
    GtG = np.dot(Gt, G)
    regularization = lam * np.eye(polynomial_degree + 1)
    GtGinv = np.linalg.inv(GtG + regularization)
    m = np.dot(np.dot(GtGinv, Gt), y)
    return m

# find Root Mean Square Error
def RMSE(Ypredict, y):
    total = 0
    for i in range(len(y)):
        total += (y[i] - Ypredict[i])**2
    total /= len(y)
    return math.sqrt(total)

# AutoFit loop going from 1 to 20 polynomial order
def autofit(x, y, folds, g):
    models = []
    for i in range(1,20):
        models.append(kfoldValidationAttempt(x, y, folds, i, g))
    output = min(models)
    return (models.index(output) + 1)

# Find average model vector
# def getAverageModelVector(order):
#     return averages[order]


order = args.m
gamma = args.gamma
counter = []
folds = args.folds
if args.autofit:
    for i in range(args.trials):
        df_small = df_small.sample(frac=1)
        x = df_small.colA.values.T
        y = df_small.colB.values.T
        counter.append(autofit(x, y, folds, gamma))
    correctOrder = max(counter, key = counter.count)
    model = linearInversion(x, y, correctOrder, gamma)
    print("Model Vector: ", model)
    print("Order: ", correctOrder)
    print("RMSE: ", kfoldValidationAttempt(x,y,folds, correctOrder, gamma))
    if args.plotting:
        plotting(x, y, model, 'weight', 'height', correctOrder)
    np.savetxt(args.modelOutput, model, header = "m = %d\ngamma = %f"%(correctOrder, gamma))
else:
    model = linearInversion(x, y, order, gamma)
    print("Model Vector: ", model)
    print("Order: ", order)
    print("RMSE: ", RMSE(predictedY(makePhi(x, order), model), y))
    if args.plotting:
        plotting(x, y, model, 'weight', 'height', order)
    np.savetxt(args.modelOutput, model, header = "m = %d\ngamma = %f"%(order, gamma))
if args.info:
    print("Name: Grant Yap\nEmail: gyap@u.rochester.edu")


np.loadtxt(args.modelOutput)


