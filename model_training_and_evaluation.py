from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, plot_confusion_matrix, precision_recall_curve, fbeta_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from itertools import product

from wordcloud import WordCloud

import time

# Define a function to look for the best model among a parameters grid
def fit_model_GridSearchCV(_model, X_, _y, param_grid, cv=5, scoring='accuracy'):    
    """
    Find the best parameters for a model in a parametere grid by fitting the models to train data and cross validate it
    Return the best possible fitted model
    -------------------------------------------------------------------------------
    Parameters:
        - model: scikit-learn like model
        - X: train data (np.array)
        - y: train labels (np.array)
        - param_grid: grid of parameters to try and iterate
        - n_iter: number of parameter combinations to try

    Returns:
        - best_model: a dictionary that contains the best model and the obtained score by that model
    """

    print('----- TAINING {} WITH {} ROWS AND {} FEATURES -----'.format(_model, X_.shape[0], X_.shape[1]))
    # Fit to data and find the best parameters for this estimator
    grid_search = GridSearchCV(
                            estimator=_model, 
                            param_grid=param_grid,
                            scoring=scoring,
                            cv=cv, n_jobs=-1, verbose=3)
    grid_search.fit(X_, _y)
    _best_model = grid_search.best_estimator_

    print("--------- BEST MODEL PARAMETERS: {} ---------".format(grid_search.best_params_))
    
    print("--------- BEST MODEL TRAIN {} SCORE ON CROSS VALIDATION: {} ---------".format(scoring, grid_search.best_score_))

    # Return best model for future use
    return _best_model


def print_best_model_variables(_model, _names, _max_words=40, _title=''):
    """
    If the model has a coefficients attribute, the best 25 features are displayed in a barplot
    -------------------------------------------------------------------------------
    Parameters:
        - model: scikit-learn like fit model on data
    """
    if hasattr(_model, 'coef_'):
        coeffs = pd.DataFrame(
                    zip(_names, _model.coef_[0]),
                    columns=['word', 'coef']
                    )
        # coeffs = coeffs.sort_values(by=['coef'], ascending=False)
            # NOTE: negative coefficients are terms that do not increase the likelyhood of the ouput being true
                # so they are discarded
        wc = WordCloud(width=1200, height=400, max_words=_max_words)
        wc = wc.generate_from_frequencies(dict(zip(coeffs.word, coeffs.coef)))

        _ = plt.figure(figsize=(10,6))
        plt.imshow(wc, interpolation='bilinear', aspect='auto')
        
        if _title != '': plt.title(_title, fontsize=20)
        
        plt.show()

        return wc
        # print("MOST RELEVANT FEATURES IN MODEL:")
        # print(coeffs.head(40))
        
def evaluate_model_test(_model, X_, _y, _labels, _scoring_beta=1):
    """
    Evaluate performance of the model on the test data
    -------------------------------------------------------------------------------
    Parameters:
        - _model: scikit-learn like fit model on data
        - _scoring_beta: beta value for sklearn fbeta_score function
        - X_: test features
        - _y: test labels
    """
    print('----- EVALUATING {} WITH {} ROWS AND {} FEATURES -----'.format(_model, X_.shape[0], X_.shape[1]))
    # Calculate class predictions with normal threshold
    test_predictions = _model.predict(X_)

    # Calculate probabilities output by the _model to increase ROC curve detail
    test_probabilities = _model.predict_proba(X_)[:, 1]

    # Generate confusion matrix
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    _ = plot_confusion_matrix(_model, X_, _y,
                                    normalize=None,
                                    ax=ax[0])
    ax[0].set_title('Confusion matrix')
    

    # Plot ROC curve and calculate AUC
    false_positive_rate, true_positive_rate, _ = roc_curve(_y, test_probabilities)


    print('--------- f{}_score: {} ---------'.format(_scoring_beta, fbeta_score(_y, test_predictions, beta=_scoring_beta)))

    ax[1].set_title('Receiver Operating Characteristic')
    ax[1].plot(false_positive_rate, true_positive_rate)
    ax[1].plot([0, 1], ls="--")
    ax[1].plot([0, 0], [1, 0] , c=".7"), ax[1].plot([1, 1] , c=".7")
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('False Positive Rate')

    precision, recall, _ = precision_recall_curve(_y, test_probabilities)

    ax[2].set_title('Precision-Recall curve')
    ax[2].plot(precision, recall)
    ax[2].plot([0, 1], ls="--")
    ax[2].plot([0, 0], [1, 0] , c=".7"), ax[2].plot([1, 1] , c=".7")



    # Print classification report and confussion matrix
    print('---------- CLASSIFICATION REPORT ---------')

    print(classification_report(_y, test_predictions, target_names=_labels))
    
    return classification_report(_y, test_predictions, target_names=_labels, output_dict=True)

# Define a function to look for the best model among a parameters grid
def cross_val_score_modify_training_data(_model, _X, _y, _cv=5, _validation_size = 0.15, _scoring_beta=1, _function=None, _function_args={}):    
    """
    Performs cross validation of a model on a training set, applying a function to the training set on every iteration. 
    If no _function is passed, it performs normal cross validation

    It validates using fscore, with a passed _scoring_beta value. 1 is balanced between false positives and false negatives. 0.5 prioritized not havinf false positives. 2 prioritizes not having false negatives.
    
    (https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)
    -------------------------------------------------------------------------------
    Parameters:
        - _model: scikit-learn like model
        - _X (dataframe): train data (np.array)
        - _y (array): train labels (np.array)
        - _cv (int): number of cross validation iterations
        - _validation_size (float): size of the validation set
        - _scoring_beta: beta value for sklearn fbeta_score function
        - _function (a function object to be applied to all rows of _X): function to be applied to the training data on each cross validation iteration
        - _function_args (dict): parameters of the _function in a dictionary format

    Returns:
        - mean_score: average score on cross validation using the _scoring metric
    """
    cross_validation_scores = []
    # print('-- Starting cross validation with {} --'.format(_function_args))
    for _ in range(0, _cv): # Perform cross validation with random data chunk selection from training data
        # print('---- Starting CV {}. Train test split ---'.format(i))
        # Train and validation data split
        _X_train, X_val, _y_train, y_val = train_test_split(_X, _y, test_size=_validation_size)

        # print('---- CV {}. Applying function ---'.format(i))
        # Apply _function to train data if a it is passed, with the provided functino args
        if _function:
            _X_train, _y_train = _function(_X_train, _y_train, **_function_args)    # Call the function unpacking argument dictionary
        # print('---- CV {}. Fitting model ---'.format(i))
        # Fit the model on training data
        _model = _model.fit(_X_train, _y_train)   
        
        # print('---- CV {}. Validating model ---'.format(i))

        # Predict labels of validation data and calculate score with passed scoring function
        vals_predictions = _model.predict(X_val)    
        score = fbeta_score(y_val, vals_predictions, beta=_scoring_beta)
        cross_validation_scores.append(score)
    
    # Calculate and return average cross validation score
    mean_score = sum(cross_validation_scores) / len(cross_validation_scores)

    # Fit a model with all the data
    
    # print('returnng {} mean score'.format(mean_score))
    return(mean_score)

def grid_search_cv_modify_training_data(_model, _model_grid, _X, _y, _cv=5, _validation_size = 0.15, _scoring_beta=1, _function=None, _function_grid={}, _function_args={}, _plot_scores=False):
    """
    Performs grid search over a grid of parameters for an ML model and another grid of parameters for a function applied to training data in order to augment it
    It uses a custom cross validation function that only applies the function to the training data and validates on clean data
    -------------------------------------------------------------------------------
    Parameters:
        - _model: scikit-learn like model
        - _model_grid: dictionary of parameters to perform gird search on the _model
        - _X (dataframe): train data (np.array)
        - _y (array): train labels (np.array)
        - _cv (int): number of cross validation iterations
        - _validation_size (float): size of the validation set
        - _scoring_beta: beta value for sklearn fbeta_score function
        - _function (a function object to be applied to all rows of _X): function to be applied to the training data on each cross validation iteration
        - _function_grid: dictionary of parameters to perform gird search on the _function
        - _function_args (dict): parameters of the _function in a dictionary format

    Returns:
        - best_model: a dictionaty that contains the results with the best model found by performing the grid search over the _model and _function
            + _best_model'
            + _best_model_params
            + _best_function_params
            + _best_score
    """
    # Get all combinations of parameters in grid
    model_keys = sorted(_model_grid)
    model_param_list = [dict(zip(model_keys, combination)) for combination in product(*(_model_grid[Name] for Name in model_keys))]
    
    # Get all combinations of parameters in grid
    function_keys = sorted(_function_grid)
    function_param_list = [dict(zip(function_keys, combination)) for combination in product(*(_function_grid[Name] for Name in function_keys))]

    total_fits = len(model_param_list) * len(function_param_list) * _cv
    model_params, function_params, function_params_disp, scores = [], [], [], []

    start = time.time()

    print('- Starting grid search, totalling {} jobs -'.format(total_fits))
    for model_param_combination in model_param_list:

        # Instantiate a model with the given param combination in the iteration
        model = _model.set_params(**model_param_combination)
        for function_param_combination in function_param_list:
            # Insert the function params in the grid search into the _function_args
            _function_args.update(function_param_combination)
            
            model_score = cross_val_score_modify_training_data(_model=model, 
                                                                _X=_X, _y=_y,
                                                                _cv=_cv, _validation_size = _validation_size, _scoring_beta=_scoring_beta, 
                                                                _function=_function, _function_args=_function_args)
            model_params.append(model_param_combination)
            function_params.append(_function_args.copy())
            function_params_disp.append(function_param_combination)
            scores.append(model_score)
            
            # print progress
            if len(scores)*_cv%5==0: 
                elapsed_time = time.time() - start
                print('------ Fitted {} jobs out of {}. Elapsed {} ------'.format(len(scores)*_cv, total_fits, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 
            

    elapsed_time = time.time() - start
    print('--- Ending grid search, totalling {} jobs. Elapsed {} ---'.format(total_fits, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 
    best_index = scores.index(max(scores))

    if _plot_scores:
        graph_labels = [str(model_param)+' '+str(function_param) for model_param, function_param in zip(model_params, function_params_disp)]

        scores, graph_labels = zip(*sorted(zip(scores, graph_labels)))

        ind = np.arange(len(graph_labels))
        width = 0.8
        fig, ax = plt.subplots(figsize=(12, int(len(graph_labels)/3)))
        ax.barh(y=ind, width=scores, height=width, label='Scores')
        ax.set_yticks(ind)
        ax.set_yticklabels(graph_labels)
        ax.set_xlabel('score')
        ax.set_ylabel('{<model parameters>}{<function parameters>}')
        for i, v in enumerate(scores):
            ax.text(v + max(scores)/100, i - .1, '{:.3f}'.format(v), fontweight='bold', fontsize=10)
        fig.suptitle('Grid search results')


    return {'_best_model_params': model_params[best_index], '_best_function_params': function_params[best_index], '_best_score': scores[best_index]}