import threading

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV,RandomizedSearchCV,  cross_val_score, cross_val_predict, LearningCurveDisplay, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, RocCurveDisplay
from sklearn.neural_network import MLPClassifier

from scipy.stats import randint, uniform

from xgboost import XGBClassifier

import itertools
import copy
import torch
from torch.autograd import Variable

import warnings

# Silence pesky deprecation warnings from sklearn
warnings.filterwarnings(module='sklearn*', action='ignore')
sns.set_palette('deep')
random_state = 1

show_plots = False

def report(results, n_top=3, limit=3):

    best_param = []

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        if limit is not None:
            candidates = candidates[:limit]
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]:.4f} (std: {results['std_test_score'][candidate]:.4f})")
            print(f"Parameters: {results['params'][candidate]}")

            if i == 1:
                best_param = results['params'][candidate]

    print("\n\n")

    return best_param


def feature_importance_graph(data, model):

    g = sns.barplot(x=model.feature_importances_, y=data.columns, orient='h')
    _ = g.set_xlabel('Relative importance')
    _ = g.set_ylabel('Features')
    _ = g.set_title('Feature Importance')

    # plt.savefig('age_predict_relative_importance')
    plt.show()

#Ground truth comparision
def check_accuracy(prediction):
    survived = pd.read_csv("/Users/ykamoji/Documents/Semester1/COMPSCI_589/titanic_survival_prediction/titanic/" + 'test_v2.csv')[['PassengerId', 'Survived']]
    return (prediction == survived['Survived']).sum()/len(survived)

from packages.data_preprocessing import preprocess_dataset
from packages.TreeClassifer import  tree_ensemble_model
from packages.PolynomialClassifier import polynomial_ensemble_model
from packages.NN import model_tuning_NN, model_tuning_mlp

