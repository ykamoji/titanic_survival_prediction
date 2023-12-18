from packages import *
def plot_custom_graph(data, x, hue=None):
    """
    Creates the bar chart of the X feature ordered by hue
    :param data:
    :param x:
    :param hue:
    :return:
    """
    ax = sns.countplot(data, x=x, hue=hue)
    ax.set(ylim=(0, 600))
    for index, values in enumerate(ax.containers):
        label = [f"{round((label / len(data))* 100,2)} %" for label in values.datavalues]
        ax.bar_label(ax.containers[index], labels=label)

    # plt.savefig(f"{x}{'_'+str(order) if order else ''}")
    plt.show()

def plot_grid(data, x, hue):
    """
    Creates the Grid chart of the X feature ordered by hue
    :param data:
    :param x:
    :param hue:
    :return:
    """
    with sns.axes_style("darkgrid"):
        g = sns.FacetGrid(data, hue=hue, height=5, aspect=2.5)
        g.map(sns.kdeplot, x, fill=True)
        g.add_legend()
        g.set(xticks=np.arange(0, data[x].max() + 1, 5), xlim=(0, data[x].max()))

    # plt.savefig(f"{x}{'_'+str(order) if order else ''}")
    plt.show()

def prepare_title(dataset):
    """

    :param dataset:
    :return:
    """

    # Get the titles of the passengers
    dataset['Title'] = dataset['Name'].apply(lambda x: x[x.find(', ') + 2:x.find('.')])

    top_titles = dataset[['Title', 'Name']].groupby(['Title']).agg('count').sort_values(by='Name', ascending=False)

    # print(top_titles)

    dataset['Title'] = dataset['Name'].apply(lambda x: x[x.find(', ') + 2:x.find('.')])
    dataset['Title'].replace(to_replace=['Mlle', 'Ms'], value='Miss', inplace=True)
    dataset['Title'].replace(to_replace='Mme', value='Mrs', inplace=True)
    dataset['Title'].replace(to_replace=['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',
                                         'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'],
                             value='Rare', inplace=True)
    return dataset

def model_tuning_age(modelName, X_train, Y_train):

    if modelName == "RandomForestClassifier" :

        # model_rf = RandomForestClassifier(random_state=random_state)
        # rand_param = {
        #     'n_estimators': range(0,100,10),
        #     'max_depth': list(range(2,8)),
        #     'max_features': list(range(5, 15)),
        #     'min_samples_split': list(range(2, 10))
        # }
        # rf_search = GridSearchCV(model_rf, param_grid=rand_param, cv=5, n_jobs=-1, verbose=1)
        #
        # rf_search.fit(X_train, Y_train)
        #
        # best_param = report(rf_search.cv_results_)

        best_param = {'max_depth': 7, 'max_features': 5, 'min_samples_split': 4, 'n_estimators': 10}

        # best_model = RandomForestClassifier(random_state=random_state, **best_param)
        # best_model.fit(X_train,Y_train)
        # print(f"Fit accuracy : {best_model.score(X_train,Y_train)*100:.2f}")

        return RandomForestClassifier(random_state=random_state, **best_param)

def kmeans_age(data, n_clusters):
    """
    Run K means for Age to get age labels
    :param data:
    :param n_clusters:
    :return age bands:
    """
    k_means_data = data.copy()
    k_means_data = k_means_data[['Age', 'Survived']].dropna()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(k_means_data)
    labels = kmeans.labels_

    if show_plots:
        sns.swarmplot(data=k_means_data, y='Age', hue='Survived', s=2.5)
        # plt.savefig('Kmean_age_before_clustering.png')
        plt.show()

    if show_plots:
        sns.swarmplot(data = k_means_data, y='Age', x = labels, hue='Survived', s=2.5)
        # plt.savefig('Kmean_age_after_clustering.png')
        plt.show()

    k_means_data['AgeCluster'] = labels
    age_bands = [0] + sorted(k_means_data.groupby('AgeCluster')['Age'].max().tolist())[:-1] + [np.inf]
    return age_bands

def age_classifers(X_train, Y_train):
    """

    :param X_train:
    :param Y_train:
    :return:
    """
    logreg = LogisticRegression(random_state=random_state)
    logreg_scores = cross_val_score(logreg, X_train, Y_train, cv=10)

    knn = KNeighborsClassifier()
    knn_scores = cross_val_score(knn, X_train, Y_train, cv=10)

    tree = DecisionTreeClassifier(random_state=random_state)
    tree_scores = cross_val_score(tree, X_train, Y_train, cv=10)

    rf = RandomForestClassifier(random_state=random_state)
    rf_score = cross_val_score(rf, X_train, Y_train, cv=10)

    results = pd.DataFrame({
        'Model': ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier','RandomForestClassifier'],
        'Score': [logreg_scores.mean(), knn_scores.mean(), tree_scores.mean(), rf_score.mean()]})

    results = results.sort_values(by='Score', ascending=False)

    #results = results.set_index('Score')
    #print(results))

    model = results.max()['Model']

    # print(f"\n{model} gives the best cross validation score {max(results['Score'])*100:.2f} % for predicting age.\n")

    return model

def predict_age(age_predict_data):
    """

    :param data:
    :return:
    """

    # age_predict_data['IsMale'] = age_predict_data['Sex'].astype('category').cat.codes

    age_predict_data = pd.get_dummies(age_predict_data, columns=['Title','Sex', "Embarked"])

    age_predict_data.drop(["Ticket", "Cabin","Name"], axis=1, inplace=True)

    age_predict_data = age_predict_data.rename(columns={"Title_Master": "Master", "Title_Miss": "Miss","Title_Mr":"Mr",
                                                        "Title_Mrs":"Mrs", "Title_Rare":"Rare"})

    age_predict_data = age_predict_data.rename(columns={"Sex_female": "female", "Sex_male": "male"})

    age_predict_data = age_predict_data.rename(columns={"Embarked_C": "E_C", "Embarked_Q": "E_Q","Embarked_S": "E_S"})

    # print(f"\nFinal columns after encoding: {', '.join(list(age_predict_data.columns))}")

    X_train = age_predict_data.loc[age_predict_data['AgeBand'] != -1].drop(columns='AgeBand')
    Y_train = age_predict_data['AgeBand'].loc[age_predict_data['AgeBand'] != -1]


    model = age_classifers(X_train, Y_train)

    model = model_tuning_age(model, X_train, Y_train)

    model.fit(X_train, Y_train)
    # feature_importance_graph(X_train, model)

    # Get all rows with unknown age bands (-1) for the test set
    X = age_predict_data.loc[age_predict_data['AgeBand'] == -1].drop(columns='AgeBand')

    return model.predict(X)

def fill_missing_age(train, test, dataset):
    """

    :param train:
    :param test:
    :param dataset:
    :return:
    """

    ## Step 1: Run K means clustering to get age bands only on train
    age_bands = kmeans_age(train.copy(), n_clusters=4)

    ## Step 2: Use the age bands to create a new feature 'Age Band'
    dataset['AgeBand'] = pd.cut(dataset['Age'], age_bands)
    dataset.drop(["Age"], axis=1, inplace=True)

    ## We don't do one-hot encoding here since age bands contain information about survivability and hence should
    ## have relative ordering.
    dataset['AgeBand'] = dataset['AgeBand'].astype('category').cat.codes

    ## Step 3: Predict the missing values for 'Age Band' using a classification algorithm
    predicted_age_band = predict_age(dataset)

    dataset.loc[dataset['AgeBand'] == -1, 'AgeBand'] = predicted_age_band

    train['AgeBand'] = dataset.iloc[:len(train), -1,]
    test['AgeBand'] = dataset.iloc[len(train):, -1]

    return train, test

def embarked_analysis(train):
    """

    :param data:
    :return:
    """
    if show_plots:
        _ = sns.catplot(x='Embarked', col='Pclass', row='Sex', data=train, kind='count')
    # plt.savefig('embarked_pclass_sex_count')
        plt.show()

    embarked_corr = (train[['Survived', 'Embarked', 'Pclass', 'Sex']].groupby(['Embarked', 'Pclass', 'Sex'])
                     .agg(['count', 'sum', 'mean']))
    embarked_corr.columns = embarked_corr.columns.droplevel(0)
    embarked_corr.columns = ['Total', 'Survived', 'Rate']

    # print(embarked_corr)

def fare_analysis(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    fare_dataset = pd.concat([train, test], sort=True)
    num_fare_bins = 3


    fare_dataset['TicketFreq'] = fare_dataset.groupby('Ticket')['Ticket'].transform('count')
    fare_dataset['PassengerFare'] = fare_dataset['Fare'] / fare_dataset['TicketFreq']

    fare_dataset['FareBand'], fare_bins = pd.qcut(fare_dataset['PassengerFare'], num_fare_bins, retbins=True)

    if show_plots:
        _ = sns.countplot(x='FareBand', hue='Survived', data=fare_dataset)
        _ = plt.xticks(rotation=0, ha='center')
        # plt.savefig('Fare_band_counts')

        g = sns.catplot(x='FareBand', col='Pclass', data=fare_dataset, kind='count')
        _ = g.set_xticklabels(rotation=0, ha='center')
        # plt.savefig('FareBand_pclass_count')

        _ = sns.catplot(x='FareBand', col='Pclass', row='AgeBand', data=fare_dataset, kind='count')

        # plt.savefig('FareBand_pclass_ageband_count')
        plt.show()

    band = pd.Interval(left=7.775, right=13.0)
    mask = (fare_dataset['FareBand'] == band) & (fare_dataset['Pclass'] != 1)

    # print(f"\n\n{fare_dataset.loc[mask, ['FareBand', 'Pclass', 'PassengerFare']].groupby(['FareBand', 'Pclass']).agg(['mean']).dropna()}")

    band1 = pd.Interval(left=-0.001, right=7.775)
    band2 = pd.Interval(left=7.775, right=13.0)
    mask = ((fare_dataset['FareBand'] == band1) | (fare_dataset['FareBand'] == band2)) & (fare_dataset['Pclass'] == 3)

    # print(f"\n\n{fare_dataset.loc[mask, ['FareBand', 'Pclass', 'Survived']].groupby(['FareBand', 'Pclass']).agg(['mean']).dropna()}")

def passenger_groups_feature(train, test):
    """

    :param train:
    :param test:
    :return:
    """

    train_len = len(train)

    groups_dataset = pd.concat([train, test], sort=True)

    surname = groups_dataset['Name'].apply(lambda x: x[:x.find(',')])
    ticket = groups_dataset['Ticket'].apply(lambda x: x[:-1])

    groups_dataset['SPTE'] = (surname.astype(str) + '-' + groups_dataset['Pclass'].astype(str) + '-'
                       + ticket.astype(str) + '-' + groups_dataset['Embarked'].astype(str))

    def spte_group_lebeler(group):
        group_elements = groups_dataset.loc[groups_dataset['SPTE'] == group, 'PassengerId']
        if len(group_elements) == 1:
            return 0
        else:
            return group_elements.min()

    groups_dataset['GroupId'] = groups_dataset['SPTE'].apply(spte_group_lebeler)
    groups_dataset.drop(columns='SPTE', inplace=True)

    def ticket_group_labeler(group):
        unique_groups = group.unique()
        if len(unique_groups) == 1:
            return unique_groups[0]
        elif len(unique_groups) == 2 and min(unique_groups) == 0:
            return groups_dataset.loc[group.index, 'PassengerId'].min()
        else:
            raise ValueError("Found conflict between SPTE and ticket grouping:\n\n{}".format(groups_dataset.loc[group.index]))

    groups_dataset['GroupId'] = groups_dataset.groupby('Ticket')['GroupId'].transform(ticket_group_labeler)

    groups_dataset['GroupSize'] = groups_dataset.groupby('GroupId')['GroupId'].transform('count')
    groups_dataset.loc[groups_dataset['GroupId'] == 0, 'GroupSize'] = 1

    # InGroup is 1 for groups with more than one member
    groups_dataset['InGroup'] = (groups_dataset['GroupSize'] > 1).astype(int)

    # Add to the train and test datasets
    train['InGroup'] = groups_dataset.iloc[:train_len, -1]
    test['InGroup'] = groups_dataset.iloc[train_len:, -1].reset_index(drop=True)

    if show_plots:
        _ = sns.countplot(x='InGroup', hue='Survived', data=train)
        # plt.savefig('Ingroup_count')
        plt.show()

    # print(train.groupby('InGroup')['Survived'].mean())

    wcg = women_child_group(groups_dataset)

    # Add to the train and test datasets
    train['InWcg'] = wcg.iloc[:len(train), -1]
    test['InWcg'] = wcg.iloc[len(train):, -1].reset_index(drop=True)

    wcsg = women_child_survived_group(groups_dataset)

    # Add to the train and test datasets
    train['WcgAllSurvived'] = wcsg.iloc[:train_len, -1]
    test['WcgAllSurvived'] = wcsg.iloc[train_len:, -1].reset_index(drop=True)

    wcdg = women_child_all_died_group(groups_dataset)

    # Add to the train and test datasets
    train['WcgAllDied'] = wcdg.iloc[:train_len, -1]
    test['WcgAllDied'] = wcdg.iloc[train_len:, -1].reset_index(drop=True)

    return train, test

def women_child_group(groups_dataset):
    """

    :param groups_dataset:
    :return:
    """

    groups_dataset['Title'] = groups_dataset['Name'].apply(lambda x: x[x.find(', ') + 2:x.find('.')])

    # Create a mask to account only for females or boys in groups
    mask = (groups_dataset['GroupId'] != 0) & ((groups_dataset['Title'] == 'Master') | (groups_dataset['Sex'] == 'female'))

    # Get the number of females and boys in each group, discard groups with only one member
    wcg_groups = groups_dataset.loc[mask, 'GroupId'].value_counts()
    wcg_groups = wcg_groups[wcg_groups > 1]

    # Update the mask to discard groups with only one female or boy
    mask = mask & (groups_dataset['GroupId'].isin(wcg_groups.index))

    # Create the new feature using the updated mask
    groups_dataset['InWcg'] = 0
    groups_dataset.loc[mask, 'InWcg'] = 1

    # print("Number of woman-child-groups found:", len(wcg_groups))
    # print("Number of passengers in woman-child-groups:", len(groups_dataset.loc[groups_dataset['InWcg'] == 1]))

    return groups_dataset

def women_child_survived_group(groups_dataset):
    """

    :param groups_dataset:
    :return:
    """

    grouped_means = groups_dataset.loc[groups_dataset['InWcg'] == 1].groupby('GroupId')['Survived'].mean()

    # Create a new column with mean survival rates based on 'GroupId'
    groups_dataset['WcgAllSurvived'] = groups_dataset['GroupId'].map(grouped_means)

    # Replace NaN values with 0
    groups_dataset['WcgAllSurvived'].fillna(0, inplace=True)

    # Convert to integer
    groups_dataset['WcgAllSurvived'] = groups_dataset['WcgAllSurvived'].astype(int)

    return groups_dataset

def women_child_all_died_group(groups_dataset):
    """

    :param groups_dataset:
    :return:
    """
    grouped_means = groups_dataset.loc[groups_dataset['InWcg'] == 1].groupby('GroupId')['Survived'].mean()

    # Create a new column with mean survival rates based on 'GroupId'
    groups_dataset['WcgAllDied'] = groups_dataset['GroupId'].map(grouped_means)

    # Replace NaN values with 0
    groups_dataset['WcgAllDied'].fillna(0, inplace=True)

    # Convert to integer
    groups_dataset['WcgAllDied'] = groups_dataset['WcgAllDied'].astype(int)

    return groups_dataset

def preprocess_dataset(train, test):
    print('\n\n----- Starting data preprocessing ------\n')

    ### taking a copy of the original train & test sets.
    processed_x_train = pd.DataFrame()
    processed_y_train = pd.DataFrame()
    processed_x_test = pd.DataFrame()

    ### train.head() to view the feature structures.

    ### Combining the train & test set to view the dataset statistics.
    dataset = pd.concat([train, test], sort=False).drop(columns='Survived')
    # print(pd.DataFrame({'Na': dataset.isna().sum(), '%': round(dataset.isna().sum() / len(dataset),4)*100}))

    ### Plots to check relationship of feature to predict Survival : Sex, Pclass, & Embarked.
    if show_plots:
        for feature, hue in [('Survived',None),('Sex','Survived'),('Pclass','Survived'),('Embarked','Survived')]:
            plot_custom_graph(train, x=feature, hue=hue)

    for feature in ['Sex', 'Pclass']:
        processed_x_train[feature] = train[feature]
        processed_x_test[feature] = test[feature]
        if feature == 'Sex':
            processed_x_train['IsMale'] = pd.get_dummies(processed_x_train['Sex'], drop_first=True)
            processed_x_train['IsMale'] = processed_x_train['IsMale'].astype('category').cat.codes
            processed_x_train.drop(["Sex"], axis=1, inplace=True)

            processed_x_test['IsMale'] = pd.get_dummies(processed_x_test['Sex'], drop_first=True)
            processed_x_test['IsMale'] = processed_x_test['IsMale'].astype('category').cat.codes
            processed_x_test.drop(["Sex"], axis=1, inplace=True)


    ### Name analysis
    dataset = prepare_title(dataset)

    ## Based on analysis not include Title
    # processed_x_train['Title'] = dataset.iloc[:len(train), -1, ]
    # processed_x_test['Title'] = dataset.iloc[len(train):, -1]
    #
    # processed_x_train = pd.get_dummies(processed_x_train, columns=['Title'])
    # processed_x_test = pd.get_dummies(processed_x_test, columns=['Title'])


    # dataset.drop(["Name"], axis=1, inplace=True)

    ### Add missing Fare
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    ### Age analysis.
    if show_plots:
        plot_grid(train, x='Age', hue='Survived')

    ### Fill in missing Age for both train & test
    train_age, test_age = fill_missing_age(train, test, dataset)

    processed_x_train['AgeBand'] = train_age['AgeBand']
    processed_x_test['AgeBand'] = test_age['AgeBand']

    #print(dataset['AgeBand'].isna().sum())

    ### Embarked analysis
    embarked_analysis(train.copy())
    # processed_x_train['Embarked'] = train_age['Embarked']
    # processed_x_test['Embarked'] = test_age['Embarked']

    # processed_x_train = pd.get_dummies(processed_x_train, columns=['Embarked'])
    # processed_x_test = pd.get_dummies(processed_x_test, columns=['Embarked'])

    ## Based on analysis not include Embarked


    ### Fare analysis
    fare_analysis(train.copy(), test.copy())

    # processed_x_train['Fare'] = train['Fare']
    # processed_x_train['Fare'].fillna(dataset['Fare'].mean(), inplace=True)
    #
    # processed_x_test['Fare'] = test['Fare']
    # processed_x_test['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    ## Based on analysis not include Fare

    ### New groups feature based on SibSp and Parch
    train_groups, test_groups = passenger_groups_feature(train.copy(), test.copy())

    for feature in ['InGroup', 'InWcg','WcgAllSurvived','WcgAllDied']:
        processed_x_train[feature] = train_groups[feature]
        processed_x_test[feature] = test_groups[feature]

    processed_y_train['Survived'] = train['Survived']

    print('\n\n----- Completed data preprocessing ------\n')

    print(f"X training set shape = {processed_x_train.shape}")

    print(f"Y training set shape = {processed_y_train.shape}")

    print(f"test set shape = {processed_x_test.shape}")

    return processed_x_train, processed_y_train, processed_x_test







