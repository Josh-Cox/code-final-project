# ---------------- IMPORTS ---------------- #

import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
import os
import copy

from joblib import dump, load
from halo import Halo

from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from prediction import is_legal, is_legal_start
from pca import make_dir


# ---------------- GLOBALS ---------------- #

DATA_PREFIX = '../data/'
MODEL_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'

def train_models(model_name, comps_start, comps_end, hyper=False):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also used for hyperparameter training (--hyper)

    :param model_name: name of the model ('dt', 'gb', 'ebm)
    :param comps_start: number of components to train model 1 with)
    :param comps_end: number of components to train model 2 with)
    :param hyper, default=False: whether to tune hyperparmeters
    """
    
    # check if given folder exists
    if not os.path.isdir(f"{PCA_PREFIX}/{comps_start}_{comps_end}"):
        print("ERROR: Folder not found")
        exit()
                
    # ---------------- GET DATA FROM FILES ---------------- #
    
    X_train_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/X_train_start.csv')
    X_train_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/X_train_end.csv')
    X_val_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/X_val_start.csv')
    X_val_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/X_val_end.csv')
    y_train_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/y_train_start.csv')
    y_train_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/y_train_end.csv')
    y_val_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/y_val_start.csv')
    y_val_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/y_val_end.csv')
    boards_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/boards_start.csv')
    boards_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/boards_end.csv')

    global train_boards_start
    train_boards_start = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/train_boards_start.csv')
    global train_boards_end
    train_boards_end = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/train_boards_end.csv')
    global start_squares
    start_squares = list(y_train_start['start_square'])
    
    combined_boards_start = pd.concat([train_boards_start, boards_start], ignore_index=True)
    combined_boards_end = pd.concat([train_boards_end, boards_end], ignore_index=True)
    
    
    # ---------------- CREATE DIRECTORIES ---------------- #

    model_path = f'{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}'
    make_dir(model_path)

    # ---------------- SAVE PCA & SCALER TO FILE ---------------- #
    
    pca_start = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/pca_start.joblib')
    pca_end = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/pca_end.joblib')
    scaler_start = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/scaler_start.joblib')
    scaler_end = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/scaler_end.joblib')
    dump(pca_start, model_path + '/pca_start.joblib')
    dump(pca_end, model_path + '/pca_end.joblib')
    dump(scaler_start, model_path + '/scaler_start.joblib')
    dump(scaler_end, model_path + '/scaler_end.joblib')

   # ---------------- DATA ENCODING ---------------- #

    # keep original versions for combining models later
    X_val_end_og = copy.deepcopy(X_val_end)
    
    # perform scaling
    X_val_end = scaler_end.transform(X_val_end)
    
    # perform pca
    X_val_end = pca_end.transform(X_val_end)
    
    # convert to dataframe
    X_val_end = pd.DataFrame(X_val_end, columns=[f'PC{i+1}' for i in range(int(comps_end))])
   
    # change shape
    y_train_start = np.ravel(y_train_start)
    y_train_end = np.ravel(y_train_end)

    # create encoders
    le_start = LabelEncoder()
    le_end = LabelEncoder()
    
    # combined data for label encoder
    combined_data_start = np.concatenate([y_train_start, np.ravel(y_val_start)])
    combined_data_end = np.concatenate([y_train_end, np.ravel(y_val_end)])

    # fit encoder
    le_start.fit(combined_data_start)
    le_end.fit(combined_data_end)
    
    # encode
    y_train_start = le_start.transform(y_train_start)
    y_train_end = le_end.transform(y_train_end)
    
    # combine data
    combined_y_start = le_start.transform(combined_data_start)
    combined_y_end = le_end.transform(combined_data_end)
    combined_X_start = pd.concat([X_train_start, X_val_start], ignore_index=True)
    combined_X_end = pd.concat([X_train_end, X_val_end], ignore_index=True)
    
    # save label encoder classes for later use
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_start.npy", le_start.classes_)
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_end.npy", le_end.classes_)
    
    # ---------------- TRAINING MODEL ---------------- #
    
    # start time of training
    start_time = time.time()
    
    # initialise models
    model_start, model_end = None, None
    
    # if tuning hyperparameters
    if hyper: 
        tune_hyper(combined_X_start, combined_X_end, combined_y_start, combined_y_end, model_name)
        exit()
    else:
        # create and train correct model
        match model_name:
            case 'dt':
                start_params = {
                    'criterion': 'gini',
                    'max_depth': 11,
                    'max_features': 47,
                    'min_samples_leaf': 20,
                    'min_samples_split': 2,
                    'random_state': 42,
                }
                end_params = {
                    'criterion': 'gini',
                    'max_depth': 9,
                    'max_features': 15,
                    'min_samples_leaf': 2,
                    'min_samples_split': 2,
                    'random_state': 42,
                }
                
                model_start = DecisionTreeClassifier(**start_params)
                model_end = DecisionTreeClassifier(**end_params)
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)
            case 'gb':
                start_params = {
                    'learning_rate': 0.010388437424247943,
                    'max_depth': 17,
                    'min_child_weight': 3,
                    'gamma': 3.350987217555605,
                    'random_state': 42,
                    'enable_categorical': True,
                }
                end_params = {
                    'learning_rate': 0.012081004159771277,
                    'max_depth': 8,
                    'min_child_weight': 1,
                    'gamma': 5.778223716405115,
                    'random_state': 42,
                    'enable_categorical': True,
                }
                model_start = xgb.XGBClassifier(**start_params)
                model_end = xgb.XGBClassifier(**end_params)
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)
            case 'ebm':
                start_params = {
                    'cyclic_progress': 0.5,
                    'greedy_ratio': 0.0,
                    'interactions': 0,
                    'learning_rate': 0.0025,
                    'max_bins': 16384,
                    'max_leaves': 3,
                    'min_samples_leaf': 3,
                    'n_jobs': -2,
                    'random_state': 42,
                    'smoothing_rounds': 50,
                    'validation_size': 0.15,
                }
                
                end_params = {
                    'cyclic_progress': 0.5,
                    'greedy_ratio': 1.75,
                    'interactions': 0,
                    'learning_rate': 0.02,
                    'max_leaves': 4,
                    'min_samples_leaf': 2,
                    'n_jobs': -2,
                    'random_state': 42,
                    'smoothing_rounds': 0,
                    'validation_size': 0.2,
                    'max_bins': 16384
                }
                model_start = ExplainableBoostingClassifier(**start_params) 
                model_end = ExplainableBoostingClassifier(**end_params) 
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)

    
    # save model to file
    dump(model_start, model_path + '/model_start.joblib')
    dump(model_end, model_path + '/model_end.joblib')
    
    # test on validation sets
    pred_start = model_start.predict(X_val_start)
    
    # decode predictions
    pred_start = le_start.inverse_transform(pred_start)

    # convert to lists
    boards_start = list(boards_start['board_pos'])
    boards_end = list(boards_end['board_pos'])
    y_val_start = list(y_val_start['start_square'])
    y_val_end = list(y_val_end['end_square'])
    
    # create legal predictions lists
    filtered_pred_start = []
    filtered_val_start = []
    filtered_pred_end = []
    filtered_val_end = []
    
    # filter illegal moves
    for i in range(len(pred_start)):
        # check is legal (non-empty square)
        if is_legal_start(boards_start[i], pred_start[i]):
            filtered_pred_start.append(pred_start[i])
            filtered_val_start.append(y_val_start[i])
        
    
    # make predictions    
    pred_end = model_end.predict(X_val_end)
    # decode predictions
    pred_end = le_end.inverse_transform(pred_end)
    
    for i in range(len(pred_end)):
        # convert to move
        pred_move = str(y_val_start[i]) + str(pred_end[i])
        # check is legal
        if is_legal(boards_end[i], pred_move):
            filtered_pred_end.append(pred_end[i])
            filtered_val_end.append(y_val_end[i])


    # create data for overall accuracy evaluation
    X_val_end_og = X_val_end_og.drop(columns=['start_square'])
    X_val_end_og['start_square'] = pred_start
    
    # scale and perform pca
    X_val_end_og = scaler_end.transform(X_val_end_og)
    X_val_end_og = pca_end.transform(X_val_end_og)
    
    # make predictions
    preds = model_end.predict(X_val_end_og)
    
    # decode predictions
    preds = le_end.inverse_transform(preds)
    
    # create legal predictions lists
    filtered_preds = []
    filtered_acc = []
            
    # loop through predictions, keeping any legal moves (not just correct ones, legally allowed moves in the current board position)
    for i in range(len(pred_end)):
        pred_move = str(pred_start[i]) + str(preds[i])
        if is_legal(boards_end[i], pred_move):
            filtered_preds.append(int(pred_move))
            filtered_acc.append(int(str(y_val_start[i]) + str(y_val_end[i])))
    
    # individual accuracy scores
    acc_start = accuracy_score(filtered_val_start, filtered_pred_start)
    acc_end = accuracy_score(filtered_val_end, filtered_pred_end)
    
    # overall scores
    acc_overall = accuracy_score(filtered_acc, filtered_preds)
    f1 = f1_score(filtered_acc, filtered_preds, average='weighted')
    precision = precision_score(filtered_acc, filtered_preds, average='weighted', zero_division=0)
    recall = recall_score(filtered_acc, filtered_preds, average='weighted', zero_division=0)
        
    # end time for training model
    end_time = time.time()
    
    print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time - start_time} ---')

    # output scores
    print(f"\nModel 1 Score: {acc_start}\nModel 2 Score: {acc_end}\nCombined Model Score: {acc_overall}")
    
    print(f"F1-Score: {f1}\nPrecision: {precision}\nRecall: {recall}")
    
    # save scores to file
    with open(f"{model_path}/results.txt", "w") as f:
        f.write(f"Accuracy: {acc_overall}\nF1-Score: {f1}\nPrecision: {precision}\nRecall: {recall}")
        
    return acc_overall

def tune_hyper(X_start, X_end, y_start, y_end, model_name):
    """
    Trains models with different hyper parameters and records accuracy
    
    :param X_start: input model 1 training data
    :param X_end: input model 2 training data
    :param y_start: output model 1 training data
    :param y_end: output model 2 training data
    :param model_name: name of the model
    """
    
    # ---------------- DEFINE MODEL SPECIFIC PARAMETERS ---------------- #
    
    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': Integer(1, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(2, 20),
        'max_features': Integer(2, 60),
    }
    
    gb_params = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'max_depth': Integer(1, 20),
        'min_child_weight': Integer(1, 20),
        'gamma': Real(0, 10, 'uniform'),
        'enable_categorical': [True],
    }
    
    ebm_params = {
        'validation_size': [0.1, 0.15, 0.2],
        'learning_rate': [0.02, 0.01, 0.005, 0.0025],
        'greedy_ratio': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0],
        'cyclic_progress': [0.0, 0.5, 1.0],
        'smoothing_rounds': [0, 50, 100, 200, 500, 1000, 2000, 4000],
        'min_samples_leaf': Integer(2, 4),
        'max_leaves': Integer(3, 4),
        'random_state': [42],
        'n_jobs': [-2],
        'interactions': [0],
        }
    
    # ---------------- RUN PARAMETER TUNING ---------------- #

    # define number of loops (for progress bar)
    print("\nTuning Hyperparameters...")

    # use correct model
    match model_name:
        case 'dt': 
            opt_start = BayesSearchCV(DecisionTreeClassifier(), dt_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
            opt_end = BayesSearchCV(DecisionTreeClassifier(), dt_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
        case 'gb': 
            opt_start = BayesSearchCV(xgb.XGBClassifier(), gb_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
            opt_end = BayesSearchCV(xgb.XGBClassifier(), gb_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
            
        case 'ebm':
            opt_start = BayesSearchCV(ExplainableBoostingClassifier() , ebm_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
            opt_end = BayesSearchCV(ExplainableBoostingClassifier() , ebm_params, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=3)
    
    # ---------------- TRAIN MODELS AND SAVE BEST ---------------- #

    opt_start.fit(X_start, y_start)
    best_params_start = opt_start.best_params_
    best_score_start = opt_start.best_score_
    
    # write to files
    with open(f"../hyperparameters/{model_name}_start.txt", "a") as f:
        f.write(f"MODEL: {model_name}\n")
        f.write(f"\nParameters Tested: {gb_params}\nBest Parameters: {best_params_start}\nBest Mean CV Score: {best_score_start}\n--------------------")         
        
    # print results     
    print(f"\nModel 1 Best Hyperparameters:\n{best_params_start}")
    print(f"\nModel 1 Mean CV Score: {best_score_start}\n")

    opt_end.fit(X_end, y_end)
    best_params_end = opt_end.best_params_
    best_score_end = opt_end.best_score_

    # write to files
    with open(f"../hyperparameters/{model_name}_end.txt", "a") as f:
        f.write(f"MODEL: {model_name}\n")
        f.write(f"\nParameters Tested: {gb_params}\nBest Parameters: {best_params_end}\nBest Mean CV Score: {best_score_end}\n--------------------")         

    # print results     
    print(f"\nModel 2 Best Hyperparameters:\n{best_params_end}")
    print(f"\nModel 2 Mean CV Score: {best_score_end}\n")

def train_params(model, X_train, y_train, X_val, y_val, val_boards):
    """
    Trains the given model and returns its accuracy
    
    :param model: model to train
    :param X_train: training input data for model
    :param y_train: training output data for model
    :param X_val: validation input data for model
    :param y_val: validation output data for model
    :param val_boards: board positions for validation data (filtering illegal moves)
    
    :return: accuracy of the model
    """
    
    # train the model
    model.fit(X_train, y_train)
                
    # evaluate
    val_preds = model.predict(X_val)
    
    # setup label encoder
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    
    # decode predictions
    val_preds = le.inverse_transform(val_preds)

    # create legal predictions lists
    filtered_y_pred = []
    filtered_y_val = []
    
    # filter illegal moves
    for i in range(len(val_preds)):
        if is_legal(val_boards[i], val_preds[i]):
            filtered_y_pred.append(val_preds[i])
            filtered_y_val.append(y_val[i])
    
    # get model accuracy
    val_accuracy = accuracy_score(filtered_y_val, filtered_y_pred)
    print("Score:", val_accuracy)
    
    return val_accuracy
        
def train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end):
    """
    Trains the given model
    
    :param X_train_start: training input data for model 1
    :param X_train_end: training input data for model 2
    :param y_train_start: training output data for model 1
    :param y_train_end: training output data for model 2
    :param model_start: model 1 to train
    :param model_end: model 2 to train
    
    :return: the trained models
    """
    
    # train models with spinners
    with Halo(text=f'Training first model', color='grey', spinner="dots3"):
        model_start.fit(X_train_start, y_train_start)
    with Halo(text=f'Training second model', color='grey', spinner="dots3"):
        model_end.fit(X_train_end, y_train_end)
    
    return model_start, model_end

def main():
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', required=True, help='Number of components to train model 1 with (folder must exist under "PCA/")')
    parser.add_argument('--comps_2', required=True, help='Number of components to train model 2 with (folder must exist under "PCA/")')
    parser.add_argument('--hyper', action='store_true', required=False, help="Whether to perform hyperparameter tuning")
    args = parser.parse_args()
    
    # ---------------- TRAIN MODEL ---------------- #
    
    train_models(args.model, args.comps_1, args.comps_2, args.hyper)
    
if __name__ == '__main__':
    main()