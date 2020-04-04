def genetic_algorithms_methods(df, X, y):

    x_train = X
    y_train = y

    from genetic_selection import GeneticSelectionCV

    # import your preferred ml model.
    from sklearn.ensemble import RandomForestClassifier

    # build the model with your preferred hyperparameters.
    model = RandomForestClassifier(n_estimators=100)

    # create the GeneticSelection search with the different parameters available.
    selection = GeneticSelectionCV(model,
                                   cv=5,
                                   scoring="accuracy",
                                   max_features=None,
                                   n_population=120,
                                   crossover_proba=0.5,
                                   mutation_proba=0.2,
                                   n_generations=50,
                                   crossover_independent_proba=0.5,
                                   mutation_independent_proba=0.05,
                                   n_gen_no_change=10,
                                   n_jobs=-1)

    # fit the GA search to our data.
    selection = selection.fit(x_train, y_train)

    # print the results.
    print("GeneticSelection: ", selection.support_)

    res = []
    res.append(('genetic', 'GeneticSelectionCV', selection.support_))
    return res
