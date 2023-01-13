from load_newts import load_newts
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV, StratifiedShuffleSplit, ParameterGrid
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import matplotlib.pyplot
from pandas import DataFrame

x_vals, y_vals = load_newts("frogs.csv", do_min_max=True)
this_svc = svm.SVC()
this_linear_svc = LinearSVC(dual=False)

plt = matplotlib.pyplot
linear_plt = matplotlib.pyplot
figure, axes = plt.subplots(2, 2)
l_figure, l_axes = linear_plt.subplots(1, 2)

c_bounds = np.logspace(-2, 2, 4)
gamma_bounds = np.logspace(-11, 0, 12)
c_parameters = dict(C=c_bounds)
k_fold = KFold(n_splits=5)

# linear c search (k-fold)
cl_search = GridSearchCV(this_linear_svc, refit=True, cv=k_fold, param_grid=c_parameters, n_jobs=-1, scoring='f1')
cl_search.fit(x_vals, y_vals)
cl_results = DataFrame(cl_search.cv_results_)
print(c_bounds)
l_axes[0].plot(c_bounds, cl_results['mean_test_score'])
l_axes[0].set_title("LinearSVM C Against F1 Score (K-Fold)")
l_axes[0].set_xlabel("C")
l_axes[0].set_ylabel("F1 Score")

# linear C search (full)
clk_search = GridSearchCV(this_linear_svc, refit=True, cv=k_fold, param_grid=c_parameters, n_jobs=-1, scoring='f1')
clk_search.fit(x_vals, y_vals)
clk_results = DataFrame(clk_search.cv_results_)
l_axes[1].plot(c_bounds, clk_results['mean_test_score'].to_numpy())
l_axes[1].set_title("LinearSVM C Against F1 Score (full)")
l_axes[1].set_xlabel("C")
l_axes[0].set_ylabel("F1 Score")

# C search (k-fold)
ck_search = GridSearchCV(this_svc, refit=True, cv=k_fold, param_grid=c_parameters, n_jobs=-1, scoring='f1')
ck_search.fit(x_vals, y_vals)

ck_results = DataFrame(ck_search.cv_results_)
axes[0, 0].plot(c_bounds, ck_results['mean_test_score'].to_numpy())
axes[0, 0].set_title("C Against F1 Score (K-Fold)")
axes[0, 0].set_xlabel("C")
axes[0, 0].set_ylabel("F1 Score")
print("C search k fold")
print(ck_search.best_params_)

# C search (full)
cf_search = GridSearchCV(this_svc, param_grid=c_parameters, n_jobs=-1, scoring='f1')
cf_search.fit(x_vals, y_vals)

cf_results = DataFrame(cf_search.cv_results_)
axes[0, 1].plot(c_bounds, cf_results['mean_test_score'].to_numpy())
axes[0, 1].set_title("C Against F1 Score (Full)")
axes[0, 1].set_xlabel("C")
axes[0, 1].set_ylabel("F1 Score")
print("C search full")
print(cf_search.best_params_)

# gamma search (k-fold)
gamma_parameters = dict(gamma=gamma_bounds)

gk_search = GridSearchCV(this_svc, cv=k_fold, param_grid=gamma_parameters, n_jobs=-1, scoring='f1')
gk_search.fit(x_vals, y_vals)

gk_results = DataFrame(gk_search.cv_results_)
axes[1, 0].plot(gamma_bounds, gk_results['mean_test_score'].to_numpy())
axes[1, 0].set_title("Gamma Against F1 Score (K-Fold)")
axes[1, 0].set_xlabel("Gamma")
axes[1, 0].set_ylabel("F1 Score")
print("gamma search k fold")
print(gk_search.best_params_)

# gamma search (full)
gf_search = GridSearchCV(this_svc, param_grid=gamma_parameters, n_jobs=-1, scoring='f1')
gf_search.fit(x_vals, y_vals)

gf_results = DataFrame(gf_search.cv_results_)
axes[1, 1].plot(gamma_bounds, gf_results['mean_test_score'].to_numpy())
axes[1, 1].set_title("Gamma Against F1 Score (Full)")
axes[1, 1].set_xlabel("Gamma")
axes[1, 1].set_ylabel("F1 Score")
print("gamma search full")
print(gf_search.best_params_)

plt.show()
linear_plt.show()