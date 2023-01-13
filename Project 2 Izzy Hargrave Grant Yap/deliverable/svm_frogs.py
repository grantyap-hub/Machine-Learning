import sklearn.metrics

from load_newts import load_newts
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV, StratifiedShuffleSplit
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# PARAMETERS FOR KERNEL SVM:
# GAMMA= 0.0000000001
# C = 0.01

# PARAMETERS FOR LINEAR SVM:
# C = 0.01

class my_svc:
    def linear_svc(self, filename):
        x_vals, y_vals = load_newts(filename, do_min_max=True)
        linear_classifer = make_pipeline(LinearSVC(dual=False))
        kf = KFold(n_splits=5, shuffle=True)
        f1_kfold_scores = np.zeros([1, 5])
        accuracy_kfold_scores = np.zeros([1, 5])
        index = 0

        for train_idxs, test_idxs in kf.split(x_vals, y_vals):
            xtrain_this_fold = x_vals.loc[train_idxs]
            xtest_this_fold = x_vals.loc[test_idxs]
            ytrain_this_fold = y_vals.loc[train_idxs]
            ytest_this_fold = y_vals.loc[test_idxs]

            # train a model on this fold
            linear_classifer.fit(xtrain_this_fold, ytrain_this_fold)

            # test the model on this fold
            f1_kfold_scores[0, index] = f1_score(ytest_this_fold, linear_classifer.predict(xtest_this_fold))
            accuracy_kfold_scores[0, index] = f1_score(ytest_this_fold, linear_classifer.predict(xtest_this_fold))
            index += 1

        linear_classifer.fit(x_vals, y_vals)
        accuracy = f1_score(y_vals, linear_classifer.predict(x_vals))
        return accuracy_kfold_scores.mean(), accuracy

    # TODO: this should be a lot better than it is, the linear model should not be better
    def kernel_svc(self, filename):
        x_vals, y_vals = load_newts(filename, do_min_max=True)
        kernel_classifier = make_pipeline(SVC(C=50, cache_size=1000))
        kf = KFold(n_splits=5, shuffle=True)
        accuracy_kfold_scores = np.zeros([1, 5])
        index = 0

        for train_idxs, test_idxs in kf.split(x_vals, y_vals):
            xtrain_this_fold = x_vals.loc[train_idxs]
            xtest_this_fold = x_vals.loc[test_idxs]
            ytrain_this_fold = y_vals.loc[train_idxs]
            ytest_this_fold = y_vals.loc[test_idxs]

            # train a model on this fold
            kernel_classifier.fit(xtrain_this_fold, ytrain_this_fold)

            # test the model on this fold
            accuracy_kfold_scores[0, index] = f1_score(ytest_this_fold, kernel_classifier.predict(xtest_this_fold))
            index += 1

        kernel_classifier.fit(x_vals, y_vals)

        full_accuracy = f1_score(y_vals, kernel_classifier.predict(x_vals))
        return accuracy_kfold_scores.mean(), full_accuracy


# main method
this_svc = my_svc()

linear_f1_score, linear_full_score = this_svc.linear_svc("frogs.csv")
print("LINEAR SVM\nk-fold f1 score: " + str(linear_f1_score) + '\nfull f1 score: ' + str(linear_full_score))

kernel_f1_score, kernel_full_score = this_svc.kernel_svc("frogs.csv")
print("\nKERNEL SVM\nk-fold f1 score: " + str(kernel_f1_score) + '\nfull f1 score: ' + str(kernel_full_score))
