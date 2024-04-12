# ---------------- IMPORTS ---------------- #

import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
import chess.pgn
import os
import pickle

from alive_progress import alive_bar
from joblib import dump, load
from halo import Halo
from tqdm import tqdm

from interpret.glassbox import ExplainableBoostingClassifier
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from prediction import make_predictions, is_legal, is_legal_start, decode_std
from pca import make_dir


# ---------------- GLOBALS ---------------- #

set_visualize_provider(InlineProvider())

DATA_PREFIX = '../data/'
MODEL_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'

def train_models(model_name, comps_start, comps_end, batch=None, hyper=False, encoding_method='std'):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also used for hyperparameter training (--hyper)

    :param model_name: name of the model ('dt', 'gb', 'ebm)
    :param folder: name of folder for number of components (e.g. 2 or 10)
    :param batch, default=None: whether to use batch training (if yes then number of batches)
    :param test, default=False: whether to use train_test_split and run model predictions
    :param hyper, default=False: whether to tune hyperparmeters
    :param encoding_method, default='std': encoding method for 'next_move_encoded'
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
    
    # ---------------- CREATE DIRECTORIES ---------------- #
    
    if batch:
        model_path = f'{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}_batch'
    else:
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
    # X_val_start_og = X_val_start
    X_val_end_og = X_val_end
    
    # perform scaling
    # X_val_start = scaler_start.transform(X_val_start)
    X_val_end = scaler_end.transform(X_val_end)
    
    # perform pca
    # X_val_start = pca_start.transform(X_val_start)
    X_val_end = pca_end.transform(X_val_end)
    
    # convert to dataframe
    # X_val_start = pd.DataFrame(X_val_start, columns=[f'PC{i+1}' for i in range(comps_start)])
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
    
    
    # save label encoder classes for later use
    # np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_start.npy", le_start.classes_)
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_start.npy", le_start.classes_)
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_end.npy", le_end.classes_)
    
    # ---------------- TRAINING MODEL ---------------- #
    
    # start time of training
    start_time = time.time()
    
    # initialise model
    model_start, model_end = None, None
    
    # if tuning hyperparameters
    if hyper: 
        tune_hyper(X_train_start, y_train_start, X_val_start, y_val_start, train_boards_start, model_name)
        exit()
    else:
        # create and train correct model
        match model_name:
            case 'dt':
                model_start = DecisionTreeClassifier(random_state=42)
                model_end = DecisionTreeClassifier(random_state=42)
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)
            case 'gb':
                start_params = {
                    'learning_rate': 0.0012,
                    'max_depth': 8,
                    'min_child_weight': 2,
                    'random_state': 42,
                    'enable_categorical': True,
                }
                # start_params = {
                #     'learning_rate': 0.3,
                #     'max_depth': 6,
                #     'min_child_weight': 1,
                # }
                # end_params = {
                #     'learning_rate': 0.3,
                #     'max_depth': 6,
                #     'min_child_weight': 1,
                # }
                end_params = {
                    'learning_rate': 0.041,
                    'max_depth': 2,
                    'min_child_weight': 12,
                    'random_state': 42,
                    'enable_categorical': True,
                }
                model_start = xgb.XGBClassifier(**start_params)
                model_end = xgb.XGBClassifier(**end_params)
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)
            case 'ebm':
                if batch: 
                    pass
                    # num_batches = batch 

                    # # calculate the batch size
                    # batch_size = len(X_train) // num_batches
                    # remainder = len(X_train) % num_batches
                    # if remainder != 0:
                    #     batch_size += 1

                    # # create model
                    # model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0)

                    # for i in range(num_batches):
                    #     batch_start = time.time()
                        
                    #     start_idx = i * batch_size
                    #     end_idx = min((i + 1) * batch_size, len(X_train))
                        
                    #     X_batch = X_train[start_idx:end_idx]
                    #     y_batch = y_train[start_idx:end_idx]

                    #     # train model
                    #     with Halo(text=f'Training', color='grey', spinner="dots3"):
                    #         model.fit(X_batch, y_batch)
                        
                    #     batch_end = time.time()
                    #     print(f'\n--- BATCH {i} COMPLETED ---')
                    #     print(f'\n--- TIME ELAPSED: {batch_end - batch_start} ---')
                else:
                    model_start = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
                    model_end = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
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
        # print(f"\nPred: {pred_move}  Move: {str(y_start[i]) + str(y_end[i])} \nBoard\n{chess.Board(boards[i])}")
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
    
    # save scores to file
    with open(f"{model_path}/results.txt", "w") as f:
        f.write(f"Accuracy: {acc_overall}\nF1-Score: {f1}\nPrecision: {precision}\nRecall: {recall}")
        
    return acc_overall
    
def score_function_start(model, X, y):
    
    # predict
    preds = model.predict(X)
    
    # convert to lists
    global train_boards_start
    boards = list(train_boards_start['board_pos'])

    # create legal predictions lists
    filtered_pred = []
    filtered_acc = []

    # filter illegal moves
    for i in range(len(preds)):
        # check is legal (non-empty square)
        if is_legal_start(boards[i], preds[i]):
            filtered_pred.append(preds[i])
            filtered_acc.append(y[i])
    
    # return accuracy
    return accuracy_score(filtered_acc, filtered_pred)

def score_function_end(model, X, y):
    
    # predict
    preds = model.predict(X)
    
    # convert to lists
    global train_boards_end
    global start_squares
    boards = list(train_boards_end['board_pos'])

    # create legal predictions lists
    filtered_pred = []
    filtered_acc = []

    # filter illegal moves
    for i in range(len(preds)):
        # create move
        pred_move = str(start_squares[i]) + str(preds[i])
        # check is legal
        if is_legal(boards[i], pred_move):
            filtered_pred.append(preds[i])
            filtered_acc.append(y[i])
    
    # return accuracy
    acc =  accuracy_score(filtered_acc, filtered_pred)
    return acc
        

def tune_hyper(X_train, y_train, X_val, y_val, train_boards_start, model_name):
    """
    Trains models with different hyper parameters and records accuracy
    
    :param X_train: input training data
    :param y_trai: output training data
    :param X_val: input validation data
    :param y_val: output validation data
    :param val_boards: boards of the validation data
    :param model_name: name of the model
    """
    
    # ---------------- DEFINE MODEL SPECIFIC PARAMETERS ---------------- #
    dt_params = {}
    
    gb_params_start = {
        'learning_rate': [0.3, 0.0012],
        'max_depth': [6, 8],
        'min_child_weight': [1, 2],
    }
    gb_params_end = {
        'learning_rate': [0.041],
        'max_depth': [2],
        'min_child_weight': [12],
    }
    
    gb_bayes_params = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'max_depth': Integer(1, 20),
        'min_child_weight': Integer(1, 20),
        'n_estimators': Integer(10, 1000),
        'gamma': Real(0, 10, 'uniform')
    }
    
    ebm_params = {}
    
    # ---------------- DEFINE VARIABLES ---------------- #
    
    # convert to array
    # val_boards = list(val_boards['board_pos'])
    y_val = list(y_val['start_square'])


    # ---------------- WIPE OUTPUT FILE ---------------- #
    
    # output_path_start = f"../hyperparameters/{model_name}_start.txt"
    # output_path_end = f"../hyperparameters/{model_name}_end.txt"
    # if os.path.exists(output_path_start):
    #     os.remove(output_path_start)
    # if os.path.exists(output_path_end):
    #     os.remove(output_path_end)
        
    # f_start = open(output_path_start, "w")
    # f_start.close()
    # f_end = open(output_path_end, "w")
    # f_end.close()
    
    # ---------------- RUN PARAMETER TUNING ---------------- #
    
    # define number of loops (for progress bar)
    print("\nTuning Hyperparameters...")
    
    # initialise model and grid search
    model_start, model_end = None, None
    grid_search_start, grid_search_end = None, None

    # use correct model
    match model_name:
        case 'dt': 
            model_start = DecisionTreeClassifier(random_state=42)
            model_end = DecisionTreeClassifier(random_state=42)
            grid_search_start = GridSearchCV(model_start, dt_params, cv=5, verbose=2, scoring=score_function_start)
            grid_search_end = GridSearchCV(model_end, dt_params, cv=5, verbose=2, scoring=score_function_end)
        case 'gb': 
            # opt_start = BayesSearchCV(xgb.XGBClassifier(enable_categorical=True), gb_bayes_params, n_iter=50, cv=5, random_state=42, verbose=3, scoring=score_function_start)
            # opt_end = BayesSearchCV(xgb.XGBClassifier(enable_categorical=True), gb_bayes_params, n_iter=50, cv=5, random_state=42, verbose=3, scoring=score_function_end)
            model_start = xgb.XGBClassifier(random_state=42, enable_categorical=True)
            model_end = xgb.XGBClassifier(random_state=42, enable_categorical=True)
            grid_search_start = GridSearchCV(model_start, gb_params_start, cv=5, verbose=2, scoring=score_function_start)
            grid_search_end = GridSearchCV(model_end, gb_params_end, cv=5, verbose=2, scoring=score_function_end)
            
        case 'ebm':
            model_start = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
            model_end = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
            grid_search_start = GridSearchCV(model_start, ebm_params, cv=5, verbose=2, scoring=score_function_start)
            grid_search_end = GridSearchCV(model_end, ebm_params, cv=5, verbose=2, scoring=score_function_end)
    
    # ---------------- TRAIN MODELS AND SAVE BEST ---------------- #

    # opt_start.fit(X_train, y_train)
    # best_params_start = opt_start.best_params_
    # best_score_start = opt_start.best_score_
    grid_search_start.fit(X_train, y_train)
    best_params_start = grid_search_start.best_params_
    best_score_start = grid_search_start.best_score_
    
    # write to files
    with open(f"../hyperparameters/{model_name}_start.txt", "a") as f:
        f.write(f"\nParameters Tested: {gb_params_start}\nBest Parameters: {best_params_start}\nBest Mean CV Score: {best_score_start}\n--------------------")         
        
    # print results     
    print(f"\nModel 1 Best Hyperparameters:\n{best_params_start}")
    print(f"\nModel 1 Mean CV Score: {best_score_start}\n")

    # opt_end.fit(X_train, y_train)
    # best_params_end = opt_end.best_params_
    # best_score_end = opt_end.best_score_
    grid_search_end.fit(X_train, y_train)
    best_params_end = grid_search_end.best_params_
    best_score_end = grid_search_end.best_score_

    # write to files
    with open(f"../hyperparameters/{model_name}_end.txt", "a") as f:
        f.write(f"\nParameters Tested: {gb_params_end}\nBest Parameters: {best_params_end}\nBest Mean CV Score: {best_score_end}\n--------------------")         

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
    
    :param X_train: training input data for model
    :param y_train: training output data for model
    :param model: the model to train
    
    :return: the trained model
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
    parser.add_argument('--batch', type=int, help='Number of batches train model in (Used if model is crashing)')
    parser.add_argument('--hyper', action='store_true', required=False, help="Whether to perform hyperparameter tuning")
    args = parser.parse_args()
    
    # ---------------- DEFINE VARIABLES ---------------- #
    
    encoding_method = "std"
    
    # ---------------- TRAIN MODEL ---------------- #
    
    train_models(args.model, args.comps_1, args.comps_2, args.batch, args.hyper)
    
if __name__ == '__main__':
    main()