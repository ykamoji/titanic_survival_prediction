from packages import *
def model_tuning_polynomial(modelName, X_train, Y_train):
    modelMap = {
        'LogReg': {
            'model': LogisticRegression,
            'params': {
                'penalty':[None, 'l1', 'l2','elasticnet'],
            }
        },
        'SVC': {
            'model': SVC,
            'params': {
                'gamma': [0.01, 1, 2],
                'degree': [2,3]
            }
        }
    }

    print(f"Model {modelName}\n")

    Model = modelMap[modelName]['model']
    rand_param = modelMap[modelName]['params']

    if modelName == 'SVC':
        model = SVC(random_state=random_state, kernel='rbf')
    else:
        model = Model(random_state=random_state)

    if modelName == 'SVC':
        search = RandomizedSearchCV(model, param_distributions=rand_param, cv=5, n_jobs=-1, verbose=1)
    else:
        search = GridSearchCV(model, param_grid=rand_param, cv=5, n_jobs=-1, verbose=1)

    search.fit(X_train, Y_train)

    best_param = report(search.cv_results_)

    if modelName == 'SVC':
        best_model = SVC(random_state=random_state, kernel='rbf',C=0.1,**best_param)
    else:
        best_model = Model(random_state=random_state, **best_param)

    best_model.fit(X_train, Y_train)
    # print(f"\nFit accuracy for {modelName} : {best_model.score(X_train, Y_train) * 100:.2f}\n")

    return best_model

def polynomial_ensemble_model(x_train, y_train, x_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = column_or_1d(y_train, warn=False)

    k_folds = 3

    classifiers = {
        'name': ['LogReg', 'SVC'],
        'models': [
            LogisticRegression(random_state=random_state),
            SVC(random_state=random_state)
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

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    if show_plots:
        LearningCurveDisplay.from_estimator(best_model, **common_params,)
        plt.legend()
        # plt.savefig(f'{best_model}_learning_curve.png')
        plt.show()


    for model in classifiers['name']:
        model_tuning_polynomial(model, x_train, y_train)

    best_model.fit(x_train, y_train)

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