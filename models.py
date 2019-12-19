from sklearn import neighbors
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

import tensorflow as tf
from tensorflow import keras


def knn(X, Y, seed):
    """Return a K-NN model, fitted with the best parameters found using cross-validation and grid search."""
    model = neighbors.KNeighborsClassifier(algorithm='auto')
    param_grid = {'n_neighbors': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37], 'weights': ['uniform', 'distance']}
    
    # Grid search on the parameters, to find the best score.
    k = 3
    split = StratifiedShuffleSplit(n_splits=k, random_state=seed)
    search = GridSearchCV(model, param_grid, cv=split, scoring="f1")
    search.fit(X,Y)

    score = search.best_score_
    model = search.best_estimator_
    print("score={}\nModel: {}".format(score, model))
    
    return model


def neural_net(X, Y):
    """Return a trained Neural Network."""
    model = keras.Sequential([
            keras.layers.Dense(32, input_dim=len(X[0])),
            keras.layers.Activation("relu"),
            keras.layers.Dense(32),
            keras.layers.Activation("relu"),
            keras.layers.Dense(128),
            keras.layers.Activation("relu"),
            keras.layers.Dense(128),
            keras.layers.Activation("relu"),
            keras.layers.Dense(32),
            keras.layers.Activation("relu"),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    # Use the adam optimizer
    adam = keras.optimizers.Adam(lr=0.01)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    # Training
    model.fit(X, Y, epochs=20, validation_split=0.1)

    return model
    
