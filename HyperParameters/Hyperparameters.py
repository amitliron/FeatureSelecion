import pandas as pd
import numpy as py

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from skopt import BayesSearchCV

from Dataset.CreateRandomDataset import generate_dataset
from sklearn.model_selection import train_test_split, GridSearchCV


def create_model(optimizer='adam', learn_rate=0.01, dropout_rate=0.0, neurons=1):

    model = Sequential()
    model.add(Dense(neurons, input_dim=40, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():


    df, X, Y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    model = KerasClassifier(build_fn=create_model, verbose=0)

    batch_size = [10, 20]
    epochs = [10, 30]
    optimizer = ['SGD', 'Adam']
    learn_rate = [0.01, 0.3]
    dropout_rate = [0.0, 0.1, 0.3]
    neurons = [25, 30]
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, dropout_rate=dropout_rate, neurons=neurons)

    #grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #grid_result = grid.fit(X_train, y_train)

    #hp = BayesSearchCV(estimator=model, fit_params=param_grid,  cv=3)
    hp = BayesSearchCV(model, param_grid, cv=3)
    hp.fit(X_train, y_train)

    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("finished")


if __name__ == "__main__":
    main()