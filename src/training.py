# ---------------- IMPORTS ---------------- #

import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
import chess.pgn
import os

from alive_progress import alive_bar
from joblib import dump, load
from halo import Halo
from tqdm import tqdm

from interpret.glassbox import ExplainableBoostingClassifier
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
    start_squares = pd.read_csv(f'{PCA_PREFIX}{comps_start}_{comps_end}/start_squares.csv')

   # ---------------- DATA ENCODING ---------------- #
   
    # change shape
    y_train_start = np.ravel(y_train_start)
    y_train_end = np.ravel(y_train_end)
    # create encoders
    le_start = LabelEncoder()
    le_end = LabelEncoder()
    # fit encoder
    le_start.fit(y_train_start)
    le_end.fit(y_train_end)
    # encode
    y_train_start = le_start.transform(y_train_start)
    y_train_end = le_end.transform(y_train_end)
    
    
    # ---------------- CREATE DIRECTORIES ---------------- #
    if batch:
        model_path = f'{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}_batch'
    else:
        model_path = f'{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}'
        
    make_dir(model_path)
    
    # save classes for later use
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_start.npy", le_start.classes_)
    np.save(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_end.npy", le_end.classes_)
        
    # ---------------- SAVE PCA & SCALER TO FILE ---------------- #
    
    pca_start = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/pca_start.joblib')
    pca_end = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/pca_end.joblib')
    scaler_start = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/scaler_start.joblib')
    scaler_end = load(f'{PCA_PREFIX}{comps_start}_{comps_end}/scaler_end.joblib')
    dump(pca_start, model_path + '/pca_start.joblib')
    dump(pca_end, model_path + '/pca_end.joblib')
    dump(scaler_start, model_path + '/scaler_start.joblib')
    dump(scaler_end, model_path + '/scaler_end.joblib')
    
    # ---------------- TRAINING MODEL ---------------- #
    
    # start time of training
    start_time = time.time()
    
    # initialise model
    model = None
    
    # if tuning hyperparameters
    if True == False: pass # TODO: hyper: tune_hyper(X_train, y_train, X_val, y_val, val_boards, model_name)
    else:
        # create and train correct model
        match model_name:
            # case 'dt':
            #     model = DecisionTreeClassifier(random_state=42)
            #     model = train_model(X_train, y_train, model)
            case 'gb':
                model_start = xgb.XGBClassifier(random_state=42, enable_categorical=True)
                model_end = xgb.XGBClassifier(random_state=42, enable_categorical=True)
                model_start, model_end = train_model(X_train_start, X_train_end, y_train_start, y_train_end, model_start, model_end)
            # case 'ebm':
            #     if batch: 
            #         num_batches = batch 

            #         # calculate the batch size
            #         batch_size = len(X_train) // num_batches
            #         remainder = len(X_train) % num_batches
            #         if remainder != 0:
            #             batch_size += 1

            #         # create model
            #         model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0)

            #         for i in range(num_batches):
            #             batch_start = time.time()
                        
            #             start_idx = i * batch_size
            #             end_idx = min((i + 1) * batch_size, len(X_train))
                        
            #             X_batch = X_train[start_idx:end_idx]
            #             y_batch = y_train[start_idx:end_idx]

            #             # train model
            #             with Halo(text=f'Training', color='grey', spinner="dots3"):
            #                 model.fit(X_batch, y_batch)
                        
            #             batch_end = time.time()
            #             print(f'\n--- BATCH {i} COMPLETED ---')
            #             print(f'\n--- TIME ELAPSED: {batch_end - batch_start} ---')
            #     else:
            #         model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
            #         model = train_model(X_train, y_train, model)
    
    # save model to file
    dump(model_start, model_path + '/model_start.joblib')
    dump(model_end, model_path + '/model_end.joblib')
    
    # test on validation sets
    pred_start = model_start.predict(X_val_start)
    
    # setup label encoder
    le_start = LabelEncoder()
    le_end = LabelEncoder()
    le_start.classes_ = np.load(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_start.npy")
    le_end.classes_ = np.load(f"{MODEL_PREFIX}{str(model_name)}/{str(model_name)}_{comps_start}_{comps_end}/classes_end.npy")
    
    # decode predictions
    pred_start = le_start.inverse_transform(pred_start)

    # convert to lists
    boards_start = list(boards_start['board_pos'])
    boards_end = list(boards_end['board_pos'])
    y_val_start = list(y_val_start['start_square'])
    y_val_end = list(y_val_end['end_square'])
    start_squares = list(start_squares['start_square'])

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
        pred_move = str(start_squares[i]) + str(pred_end[i])
        # check is legal
        if is_legal(boards_end[i], pred_move):
            filtered_pred_end.append(pred_end[i])
            filtered_val_end.append(y_val_end[i])

    
    # scores
    acc_start = accuracy_score(filtered_val_start, filtered_pred_start)
    acc_end = accuracy_score(filtered_val_end, filtered_pred_end)
    
    # output scores
    print(f"Model 1 Score: {acc_start}\nModel 2 Score: {acc_end}\n")
        
    # end time for training model
    end_time = time.time()
    
    print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time - start_time} ---\n')
    
def tune_hyper(X_train, y_train, X_val, y_val, val_boards, model_name):
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
    
    gb_params = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
    }
    
    ebm_params = {}
    
    # ---------------- DEFINE VARIABLES ---------------- #

    # define best accuracy and model
    best_model = None
    best_accuracy = 0
    
    # convert to array
    val_boards = list(val_boards['board_pos'])
    y_val = list(y_val['next_move_encoded'])


    # ---------------- WIPE OUTPUT FILE ---------------- #
    
    output_path = f"../hyperparameters/{model_name}.txt"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    f = open(output_path, "w")
    f.close()
    
    # ---------------- RUN PARAMETER TUNING ---------------- #
    
    # define number of loops (for progress bar)
    print("\nTuning Hyperparameters...")

    # use correct model
    match model_name:
        case 'dt': 
            # TODO: hyperparameter tuning
            model = DecisionTreeClassifier(random_state=42)
        case 'gb': 
            # find number of loops (for progress bar)
            num_loops = 1
            for key in gb_params:
                num_loops *= len(gb_params[key])
            
            # try all combinations
            with alive_bar(num_loops, bar="classic2", stats=False, spinner=None) as bar:
                for learning_rate in gb_params['learning_rate']:
                    for max_depth in gb_params['max_depth']:
                        for n_estimators in gb_params['n_estimators']:
                            for min_child_weight in gb_params['min_child_weight']:
                                # create model with hyperparameters
                                model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                                            min_child_weight=min_child_weight, random_state=42, enable_categorical=True)
                                # get accuracy of model
                                val_accuracy = train_params(model, X_train, y_train, X_val, y_val, val_boards)
                                
                                # write to file
                                with open(f"../hyperparameters/{model_name}.txt", "a") as f:
                                    f.write(f"""\nPARAMS:\nlearning_rate: {learning_rate}
max_depth: {max_depth}\nn_estimators: {n_estimators}
min_child_weight: {min_child_weight}\n\nAccuracy: {val_accuracy}\n--------------------""")
                                
                                # check if better than current best model
                                if val_accuracy > best_accuracy:
                                    best_accuracy = val_accuracy
                                    best_model = model
                                
                                # increment progress bar
                                bar()
            
            # print results     
            print("Best Hyperparameters:\n")
            best_params = best_model.get_params()
            for param in best_params:
                if param in gb_params:
                    print(f"{param}: {best_params[param]}") 
            print("\n Accuracy:", best_accuracy)
            
        case 'ebm':
            # TODO: hyperparameter tuning
            # create model
            model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
  
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