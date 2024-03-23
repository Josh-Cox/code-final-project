# ---------------- IMPORTS ---------------- #

import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import time
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

from prediction import make_predictions, is_legal
from pca import make_dir


# ---------------- GLOBALS ---------------- #

# ebm
set_visualize_provider(InlineProvider())

# file paths
DATA_PREFIX = '../data/'
MODEL_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'

def train_models(model_name, folder, batch=None, test=False, hyper=False, encoding_method='std'):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also used for hyperparameter training (--hyper)

    :param model_name: name of the model ('dt', 'gb', 'ebm)
    :param folder: name of folder for number of components (e.g. 2 or 10)
    :param batch, default=None: whether to use batch training (if yes then number of batches)
    :param test, default=False: whether to use train_test_split and run model predictions
    :param hyper, default=False: whether to tune hyperparmeters
    :param encoding_method, default='std': encoding method for 'next_move_encoded'
    """
    
    # if train test split used then run predictions
    # if test:
    #     make_predictions(str(model_name), str(folder), "test", batch)
    #     exit()
    
    # check if given folder exists
    if not os.path.isdir(str(PCA_PREFIX + folder)):
        print("ERROR: Folder not found")
        exit()
                
    # ---------------- GET DATA FROM FILES ---------------- #
    
    X_train = pd.read_csv(PCA_PREFIX + f'{folder}/X_pca_train.csv')
    X_val = pd.read_csv(PCA_PREFIX + f'{folder}/X_pca_val.csv')
    y_train = pd.read_csv(PCA_PREFIX + f'{folder}/y_train.csv')
    y_val = pd.read_csv(PCA_PREFIX + f'{folder}/y_val.csv')
    val_boards = pd.read_csv(PCA_PREFIX + f'{folder}/val_boards.csv')

   # ---------------- DATA ENCODING ---------------- #
   
    # change shape
    y_train = np.ravel(y_train)
    # create encoder
    le = LabelEncoder()
    # fit encoder
    le.fit(y_train)
    # save classes for later use
    np.save('classes.npy', le.classes_)
    # encode
    y_train = le.transform(y_train)
    
    
    # ---------------- CREATE DIRECTORIES ---------------- #
    if batch:
        model_path = f'{MODEL_PREFIX}/{str(model_name)}/{str(model_name)}_{folder}_batch'
    else:
        model_path = f'{MODEL_PREFIX}/{str(model_name)}/{str(model_name)}_{folder}'
        
    make_dir(model_path)
        
    # ---------------- SAVE PCA & SCALER TO FILE ---------------- #
    
    pca = load(f'{PCA_PREFIX}/{folder}/pca.joblib')
    scaler = load(f'{PCA_PREFIX}/{folder}/scaler.joblib')
    dump(pca, model_path + '/pca.joblib')
    dump(scaler, model_path + '/scaler.joblib')
    
    # start time of training
    start_time = time.time()
    
    # initialise model
    model = None
    
    # if tuning hyperparameters
    if hyper: tune_hyper(X_train, y_train, X_val, y_val, val_boards, model_name)
    else:
        # create and train correct model
        match model_name:
            case 'dt':
                model = DecisionTreeClassifier(random_state=42)
                model = train_model(X_train, y_train, model)
            case 'gb':
                model = xgb.XGBClassifier(random_state=42, enable_categorical=True)
                model = train_model(X_train, y_train, model)
            case 'ebm':
                if batch: 
                    num_batches = batch 

                    # calculate the batch size
                    batch_size = len(X_train) // num_batches
                    remainder = len(X_train) % num_batches
                    if remainder != 0:
                        batch_size += 1

                    # create model
                    model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0)

                    for i in range(num_batches):
                        batch_start = time.time()
                        
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(X_train))
                        
                        X_batch = X_train[start_idx:end_idx]
                        y_batch = y_train[start_idx:end_idx]

                        # train model
                        with Halo(text=f'Training', color='grey', spinner="dots3"):
                            model.fit(X_batch, y_batch)
                        
                        batch_end = time.time()
                        print(f'\n--- BATCH {i} COMPLETED ---')
                        print(f'\n--- TIME ELAPSED: {batch_end - batch_start} ---')
                else:
                    model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
                    model = train_model(X_train, y_train, model)
    
    # save model to file
    dump(model, model_path + '/model.joblib')
        
    # end time for training model
    end_time = time.time()
    
    print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time - start_time} ---\n')
    
    # if train test split used then run predictions
    if test:
        make_predictions(str(model_name), str(folder), "test", batch)
    
def tune_hyper(X_train, y_train, X_val, y_val, val_boards, model_name):
    
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

    match model_name:
        case 'dt': 
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
                                model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                                            min_child_weight=min_child_weight, random_state=42, enable_categorical=True)
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
                                
                                # progress bar
                                bar()
            
            # print results     
            print("Best Hyperparameters:\n")
            best_params = best_model.get_params()
            for param in best_params:
                if param in gb_params:
                    print(f"{param}: {best_params[param]}") 
            print("\n Accuracy:", best_accuracy)
            
        case 'ebm':
            model = ExplainableBoostingClassifier(random_state=42, n_jobs=-2, interactions=0) 
  
def train_params(model, X_train, y_train, X_val, y_val, val_boards):
    """
    Trains the given model and returns its accuracy
    
    :param model: model to train
    :param X_train: training input data for model
    :param y_train: training output data for model
    :param X_val: validation input data for model
    :param y_val: validation output data for model
    :param val_boards: board positions for filtering legal moves
    
    :return: accuracy of the model
    """
    model.fit(X_train, y_train)
                
    # evaluate
    val_preds = model.predict(X_val)
    
    # setup label encoder
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    
    val_preds = le.inverse_transform(val_preds)

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
        
def train_model(X_train, y_train, model):
    """
    Trains the given model
    
    :param X_train: training input data for model
    :param y_train: training output data for model
    :param model: the model to train
    
    :return: the trained model
    """
    # train model
    with Halo(text=f'Training', color='grey', spinner="dots3"):
        model.fit(X_train, y_train) 
        return model

def main():
    
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--n_comps', required=True, help='Number of components to train with (folder must exist under "PCA/")')
    parser.add_argument('--batch', type=int, help='Number of batches train model in (Used if model is crashing)')
    parser.add_argument('--test', action='store_true', required=False, help="Whether to use train_test_split ('test' must be used in pca)")
    parser.add_argument('--hyper', action='store_true', required=False, help="Whether to perform hyperparameter tuning")
    args = parser.parse_args()
    
    # ---------------- DEFINE VARIABLES ---------------- #
    
    encoding_method = "std"
    
    # ---------------- TRAIN MODEL ---------------- #
    
    train_models(args.model, args.n_comps, args.batch, args.test, args.hyper)
    
if __name__ == '__main__':
    main()