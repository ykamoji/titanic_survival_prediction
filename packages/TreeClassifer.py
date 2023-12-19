import matplotlib.pyplot as plt

from packages import *
def model_tuning_trees(modelName, X_train, Y_train):

    modelMap = {
        'RandomForest' : {
            'model': RandomForestClassifier,
            'params':{
                'max_depth': list(range(2, 8)),
                'max_features': list(range(5, 15))
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier,
            'params': {
                'max_depth': list(range(2, 8)),
                'max_features': list(range(5, 15))
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesClassifier,
            'params': {
                'max_depth': list(range(2, 8)),
                'max_features': list(range(5, 15))
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier,
            'params': {
                'learning_rate': [0.001, 0.01, 0.1]
            }
        },
        'XGBoost': {
            'model': XGBClassifier,
            'params': {
                'max_depth': list(range(2, 8)),
                'learning_rate': [0.001, 0.01, 0.1]
            }
        },
        'GradientBoost': {
            'model': GradientBoostingClassifier,
            'params': {
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': list(range(2, 8)),
                'max_features': list(range(5, 15))
            }

        }
    }

    print(f"Model {modelName}\n")

    Model = modelMap[modelName]['model']
    rand_param = modelMap[modelName]['params']

    model = Model(random_state=random_state)

    search = GridSearchCV(model, param_grid=rand_param, cv=5, n_jobs=-1, verbose=1)

    search.fit(X_train, Y_train)


    best_param = report(search.cv_results_)

    best_model = Model(random_state=random_state, **best_param)
    best_model.fit(X_train, Y_train)
    # print(f"\nFit accuracy for {modelName} : {best_model.score(X_train, Y_train) * 100:.2f}\n")

    return best_model

def tree_ensemble_model(x_train, y_train, x_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = column_or_1d(y_train, warn=False)


    k_folds = 3
    n_estimators = 200

    classifiers = {
        'name':['DecisionTree','RandomForest','ExtraTrees','AdaBoost','XGBoost','GradientBoost'],
        'models':[
            DecisionTreeClassifier(random_state=random_state),
            RandomForestClassifier(random_state=random_state, n_estimators=n_estimators),
            ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators),
            AdaBoostClassifier(random_state=random_state, n_estimators=n_estimators),
            XGBClassifier(random_state=random_state, n_estimators=n_estimators),
            GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators)
        ],
        'scores': [],
        'acc_mean': [],
        'acc_std': []
    }

    best_model = None
    best_score = 0.0
    # Run cross-validation and store the scores
    for model in classifiers['models']:
        score = cross_val_score(model, x_train, y_train, cv=k_folds, n_jobs=-1)
        classifiers['scores'].append(score)
        classifiers['acc_mean'].append(score.mean())
        classifiers['acc_std'].append(score.std())
        if float(score.mean()) > best_score:
            best_score = float(score.mean())
            best_model = model


    # Create a nice table with the results
    classifiers_df = pd.DataFrame({
        'Model Name': classifiers['name'],
        'Accuracy': classifiers['acc_mean'],
        'Std': classifiers['acc_std']
    }, columns=['Model Name', 'Accuracy', 'Std']).set_index('Model Name')

    print(classifiers_df.sort_values('Accuracy', ascending=False))
    print("\n")

    print(f"\nBest model is {type(best_model).__name__} with high cross validation score of {best_score:.3f} \n")

    best_model.fit(x_train, y_train)
    y_scores = best_model.predict_proba(x_train)
    y_scores = y_scores[:, 1]

    precision, recall, threshold = precision_recall_curve(y_train, y_scores)

    if show_plots:
        def plot_precision_and_recall(precision, recall, threshold):
            plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
            plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
            plt.xlabel("threshold", fontsize=19)
            plt.legend(loc="upper right", fontsize=19)
            plt.ylim([0, 1])

        plt.figure(figsize=(14, 7))
        plot_precision_and_recall(precision, recall, threshold)
        # plt.savefig(f'{type(best_model).__name__}_learning_curve.png')
        plt.show()


    for model in classifiers['name']:
         model_tuning_trees(model, x_train, y_train)

    # submission = x_test.copy()
    # submission = pd.DataFrame(submission)
    # submission['Survived'] = best_model.predict(submission)
    # drop_columns = list(submission.columns)
    # drop_columns.remove('Survived')
    # submission.drop(drop_columns, axis=1, inplace=True)
    # submission.to_csv(
    #     path_or_buf='/Users/ykamoji/Documents/Semester1/COMPSCI_589/titanic_survival_prediction/titanic/' + str(
    #         type(best_model).__name__) + '_submission.csv')

    return best_model