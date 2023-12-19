import numpy as np
import pandas as pd
import time
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
import threading
from packages import show_plots, RocCurveDisplay,preprocess_dataset, tree_ensemble_model, polynomial_ensemble_model, model_tuning_NN, model_tuning_mlp

datapath = "/Users/ykamoji/Documents/Semester1/COMPSCI_589/titanic_survival_prediction/titanic/"



def load_data(file):
    return pd.read_csv(datapath+file)

def process_data(data):
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))

    return x, y


if __name__ == '__main__':
    train = load_data('train.csv')
    test = load_data('test.csv')
    x_train, y_train, x_test = preprocess_dataset(train, test)

    print("\n\n------Tree based approach modeling------\n\n")
    tree_model = tree_ensemble_model(x_train.copy(), y_train.copy(), x_test.copy())

    print("\n\n------Polynomial based modeling------\n\n")
    poly_model = polynomial_ensemble_model(x_train.copy(), y_train.copy(), x_test.copy())

    print("\n\n------Nural Network based modeling------\n\n")
    model_tuning_NN(x_train.copy(),y_train.copy(), x_test.copy())

    print("\n\n------MLP based modeling------\n\n")
    nn_model = model_tuning_mlp(x_train.copy(), y_train.copy(), x_test.copy())

    if show_plots:
        fig, ax = plt.subplots(figsize=(6, 6))

        for index, model in enumerate([tree_model, poly_model, nn_model]):
            RocCurveDisplay.from_estimator(model, x_train, y_train, alpha=1, lw=1, color=['b','g','r'][index], ax=ax)
        plt.title('ROC curves')
        plt.savefig('Model_comparison.png')
        plt.show()



