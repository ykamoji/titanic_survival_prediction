from packages import *
def titanic_net(d_in, d_hidden, n_hidden, d_out):
    if d_in < 1 or d_hidden < 1 or d_out < 1:
        raise ValueError("expected layer dimensions to be equal or greater than 1")
    if n_hidden < 0:
        raise ValueError("expected number of hidden layers to be equal or greater than 0")

        # If the number of hidden layers is 0 we have a single-layer network
    if n_hidden == 0:
        return torch.nn.Linear(d_in, d_out)

    # Number of hidden layers is greater than 0
    # Define the 3 main blocks
    first_hlayer = [torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU()]
    hlayer = [torch.nn.Linear(d_hidden, d_hidden), torch.nn.ReLU()]
    output_layer = [torch.nn.Linear(d_hidden, d_out)]

    # Build the model
    layers = torch.nn.ModuleList()

    # First hidden layer
    layers.extend(first_hlayer)

    # Remaining hidden layers
    # Subtract 1 to account for the previous layer
    for i in range(n_hidden - 1):
        layers.extend(hlayer)

    # Output layer
    layers.extend(output_layer)

    return torch.nn.Sequential(*layers)


def fit(model, X, y, epochs=250, optim='adam', lr=0.001, verbose=0):
    # Optimizer argument validation
    valid_optims = ['sgd', 'rmsprop', 'adam']
    optim = optim.lower()
    if optim.lower() not in valid_optims:
        raise ValueError("invalid optimizer got '{0}' and expect one of {1}"
                         .format(optim, valid_optims))

    # Define the loss function - we are dealing with a classification task with two classes
    # binary cross-entropy (BCE) is, therefore, the most appropriate loss function.
    # Within BCE we can use BCELoss or BCEWithLogitsLoss. The latter is more stable, so we'll
    # use that one. It expects logits, not predictions, which is why our output layer doesn't
    # have an activation function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Define the optimization algorithm
    optim = optim.lower()
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for t in range(epochs):
        # Forward pass: The model will return the logits, not predictions
        logits = model(X)

        # Compute loss from logits
        loss = loss_fn(logits, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # We can get the tensor of predictions by applying the sigmoid nonlinearity
        pred = torch.sigmoid(logits)

        # Compute training accuracy
        acc = torch.eq(y, pred.round_()).cpu().float().mean().item()

        if verbose > 2:
            print("Epoch {0:>{2}}/{1}: Loss={3:.4f}, Accuracy={4:.4f}"
                  .format(t + 1, epochs, len(str(epochs)), loss.item(), acc))

    if verbose > 2:
        print("Training complete! Loss={0:.4f}, Accuracy={1:.4f}".format(loss.item(), acc))

    return {'loss': loss.item(), 'acc': acc}


def custom_cross_val_score(model, X, y, cv=3, epochs=250, optim='adam', lr=0.001, use_cuda=True, verbose=0):
    # Generate indices to split data into training and validation set
    kfolds = KFold(cv).split(X)

    # For each fold, train the network and evaluate the accuracy on the validation set
    score = []
    for fold, (train_idx, val_idx) in enumerate(kfolds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Convert the training data to Variables
        X_train = Variable(torch.Tensor(X_train), requires_grad=True)
        y_train = Variable(torch.Tensor(y_train), requires_grad=False).unsqueeze_(-1)
        X_val = Variable(torch.Tensor(X_val), requires_grad=False)
        y_val = Variable(torch.Tensor(y_val), requires_grad=False).unsqueeze_(-1)

        # Clone the original model so we always start the training from an untrained model
        model_train = copy.deepcopy(model)

        # Move model and tensors to CUDA if use_cuda is True
        if (use_cuda):
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_val = X_val.cuda()
            y_val = y_val.cuda()
            model_train = model_train.cuda()

        # Train the network
        metrics = fit(model_train, X_train, y_train, epochs=epochs, optim=optim,
                      lr=lr, verbose=verbose)

        # Predict for validation samples
        y_val_pred = torch.sigmoid(model_train(X_val))
        acc = torch.eq(y_val, y_val_pred.round_()).cpu().float().mean().item()
        score.append(acc)

        if verbose > 2:
            print("Fold {0:>{2}}/{1}: Validation accuracy={3:.4f}"
                  .format(fold + 1, cv, len(str(cv)), acc))

    return score


def titanic_net_grid_search(X_train, Y_train, param_grid, n_folds, cv=3, epochs=250, use_cuda=False, verbose=0):
    # Cartesian product of a dictionary of lists
    # Source: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    grid = list((dict(zip(param_grid, param))
                 for param in itertools.product(*param_grid.values())))

    n_candidates = len(grid)
    if verbose > 0:
        print(f"Fitting {n_folds} folds for each of {n_candidates} candidates, totaling {n_folds * n_candidates} fits\n")

    # Do cross-validation for each combination of the hyperparameters in grid_param
    best_params = None
    best_model = None
    best_score = 0
    for candidate, params in enumerate(grid):
        if verbose == 1:
            progress = "Candidate {0:>{2}}/{1}".format(candidate + 1, n_candidates,
                                                       len(str(n_candidates)))
            print(progress, end="\r")
        elif verbose > 1:
            print("Candidate", candidate + 1)
            print("Parameters: {}".format(params))

        # Model parameters and creation
        d_in = X_train.shape[-1]
        d_hidden = params['d_hidden']
        n_hidden = params['n_hidden']
        d_out = 1
        model = titanic_net(d_in, d_hidden, n_hidden, d_out)

        # Cross-validation
        cv_score = custom_cross_val_score(model, X_train, Y_train, cv=n_folds, epochs=epochs,
                                   use_cuda=use_cuda, verbose=verbose)
        cv_mean_acc = np.mean(cv_score)
        if verbose > 1:
            print("Mean CV accuracy: {0:.4f}".format(cv_mean_acc))
            print()

        # Check if this  is the best model; if so, store it
        if cv_mean_acc > best_score:
            best_params = params
            best_model = model
            best_score = cv_mean_acc

    if verbose > 0:
        if verbose == 1:
            print()
        print("Best model")
        print("Parameters: {}".format(best_params))
        print("Mean CV accuracy: {0:.4f}".format(best_score))

    return {'best_model': best_model, 'best_params': best_params, 'best_score': best_score}


def formatter(x_train, y_train, x_test):
    x_train = np.array(x_train.copy())
    y_train = np.array(y_train.copy())
    x_test = np.array(x_test.copy())

    y_train = y_train.reshape(1, 891)[0]

    return x_train, y_train, x_test


def model_tuning_NN(x_train, y_train, x_test):

    x_train_torch, y_train_torch, x_test_torch = formatter(x_train, y_train, x_test)

    # Number of folds
    n_folds = 10

    # Grid search
    grid = {
        'n_hidden': [0, 3, 7, 10],
        'd_hidden': [3, 7, 10],
        'lr': [0.001, 0.01],
        'optim': ['Adam']
    }
    best_candidate = titanic_net_grid_search(x_train_torch, y_train_torch, grid, cv=n_folds,
                                             epochs=500, verbose=2, n_folds=n_folds)

    # Our best network
    best_model = best_candidate['best_model']

    X_train_t = Variable(torch.Tensor(x_train_torch), requires_grad=True)
    y_train_t = Variable(torch.Tensor(y_train_torch), requires_grad=False).unsqueeze_(-1)
    X_test_t = Variable(torch.Tensor(x_test_torch), requires_grad=False)

    # Train the best model
    best_params = best_candidate["best_params"]

    _ = fit(best_model, X_train_t, y_train_t, epochs=500, optim=best_params['optim'],
            lr=best_params['lr'])


    predictions = torch.sigmoid(best_model(X_train_t)).data.round_().numpy().flatten()

    if show_plots:
        sns.heatmap(confusion_matrix(y_train, predictions), annot=True, fmt='g')
        # plt.savefig(f'NN_confusion_matrix.png')
        plt.show()

    prediction = torch.sigmoid(best_model(X_test_t)).data.round_().numpy().flatten()

    # submission = x_test.copy()
    # submission = pd.DataFrame(submission)
    # submission['Survived'] = prediction
    # drop_columns = list(submission.columns)
    # drop_columns.remove('Survived')
    # submission.drop(drop_columns, axis=1, inplace=True)
    # submission.to_csv(
    #     path_or_buf='/Users/ykamoji/Documents/Semester1/COMPSCI_589/titanic_survival_prediction/titanic/' + "NN" + '_submission.csv')

def model_tuning_mlp(x_train, y_train, x_test):

    x_train, y_train, x_test = formatter(x_train, y_train, x_test)

    classifiers = {
        'name': ['MLP'],
        'models': [MLPClassifier(random_state=random_state,max_iter=500)],
        'scores': [],
        'acc_mean': [],
        'acc_std': []
    }

    for model in classifiers['models']:
        score = cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1,)
        classifiers['scores'].append(score)
        classifiers['acc_mean'].append(score.mean())
        classifiers['acc_std'].append(score.std())


    # Create a nice table with the results
    classifiers_df = pd.DataFrame({
        'Model Name': classifiers['name'],
        'Accuracy': classifiers['acc_mean'],
        'Std': classifiers['acc_std']
    }, columns=['Model Name', 'Accuracy', 'Std']).set_index('Model Name')

    print(classifiers_df)
    print("\n")

    print(f"Model MLP\n")

    model = MLPClassifier(random_state=random_state, max_iter=1000)

    rand_param={
        'hidden_layer_sizes':list(range(0,15)),
        'activation': ['tanh', 'relu'],
    }

    search = GridSearchCV(model, param_grid=rand_param, cv=5, n_jobs=-1, verbose=1)

    search.fit(x_train, y_train)

    best_param = report(search.cv_results_)

    best_model = MLPClassifier(random_state=random_state, max_iter=1000, **best_param)
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






