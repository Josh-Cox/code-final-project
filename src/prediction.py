# Imports
from preprocessing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap
import xgboost as xgb
import argparse
import os

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# path to model files (train_test_split and model joblibs)
PATH_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'
DATA_PREFIX = '../data/'

# global table to decode chess moves
DECODING_TABLE = {
    '1': 'a',
    '2': 'b',
    '3': 'c',
    '4': 'd',
    '5': 'e',
    '6': 'f',
    '7': 'g',
    '8': 'h',
}

# promotion hashmap
PROMOTION_DECODING_TABLE = {
    '1': 'r',
    '2': 'n',
    '3': 'b',
    '4': 'q'
}


def make_predictions_multi(boards):
        
    # get test data from train_test_split
    X_test = pd.read_csv(PATH_PREFIX + 'split_test_data.csv')
    
    # load trained model from file
    model = load(PATH_PREFIX + 'gb.joblib')
    
    test_data = X_test.to_numpy()
    X_single_pos = test_data[0]
    print(X_single_pos)
    X_single_pos = X_single_pos.reshape(1, -1)
    
    # get legal moves for board position
    # legal_moves = X_board_pos.legal_moves
    
    # make predictions with probabilities
    predicted_probs_list = model.predict_proba(X_single_pos)
    
    # Assuming you have a list of moves for each output column
    moves_lists = [model.estimators_[i].classes_ for i in range(len(predicted_probs_list))]

    # Group predicted moves for each row
    grouped_moves = [[] for _ in range(len(X_single_pos))]

    # Iterate over each output column
    for i, predicted_probs in enumerate(predicted_probs_list):
        # Get the moves for the current output column
        moves_list = moves_lists[i]
        
        # Filter predictions by legality and add to the corresponding row
        for row_index, probs in enumerate(predicted_probs):
            legal_moves = [moves_list[j] for j in range(len(probs))]
            grouped_moves[row_index].extend(legal_moves)

    # Print grouped predicted moves for each row
    print("Grouped Predicted Moves:")
    for row_moves in grouped_moves:
        print(row_moves)
    
def make_predictions(model_name, n_components, pred_type, batch, files=None):
    """
    Makes predictions using trained models and test data

    :param X_test: test data for predictions
    :param board: board positions to check for legal moves
    """
    
    # path to model folder
    if batch:
        model_path = f'{PATH_PREFIX}/{str(model_name)}/{str(model_name)}_{str(n_components)}_batch/'
    else:
        model_path = f'{PATH_PREFIX}/{str(model_name)}/{str(model_name)}_{str(n_components)}/'
        
    # check if model exists
    if not os.path.isfile(f'{model_path}/model.joblib'):
        print("ERROR: No trained model exists. Train using python training.py")
        exit()
    
    if pred_type == "test":
        # get test data from train_test_split
        X_test = pd.read_csv(PCA_PREFIX + str(n_components) + '/X_pca_test.csv')
        y_test = pd.read_csv(PCA_PREFIX + str(n_components) + '/y_test.csv')
        boards = pd.read_csv(PCA_PREFIX + str(n_components) + '/Boards.csv')
    else:
        # check if given filename exists
        if not os.path.isfile(f'{DATA_PREFIX}/{files}.csv'):
            print("ERROR: File not found")
            exit()
            
        pca = load(model_path + '/pca.joblib')
        scaler = load(model_path + '/scaler.joblib')
    
        input_data = pd.read_csv(f'{DATA_PREFIX}/{files}.csv')
            
        # get input data and board
        X = input_data.drop(columns=['board_pos', 'next_move_encoded'])
        y_test = input_data[['next_move_encoded']]
        boards = input_data[['board_pos']]
        
        # scale and perform pca
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        
        # get correct number of components
        X_test = X_pca[:, :n_components]
            
        
    print("\n--- MAKING PREDICTIONS ---")
    
    # start time for predicting
    start_time = time.time()
    
    # load trained model from file
    model = load(f'{model_path}/model.joblib')

    # make predictions with probabilities
    y_pred = model.predict(X_test)
    
    # end time for predicting
    end_time = time.time()
    
    print("\n--- FINISHED PREDICTIONS ---")
    print(f'\n--- TIME ELAPSED: {end_time - start_time} ---\n')
    
    # setup label encoder
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    
    y_pred = le.inverse_transform(y_pred)
    
    filtered_y_pred = []
    filtered_y_test = []
    filtered_boards = []
    
    boards = list(boards['board_pos'])
    y_test = list(y_test['next_move_encoded'])
            
    for i in range(len(y_pred)):
        if is_legal(boards[i], y_pred[i]):
            filtered_y_pred.append(y_pred[i])
            filtered_y_test.append(y_test[i])
            filtered_boards.append(boards[i])
         
    precision = precision_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
    recall = recall_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(filtered_y_test, filtered_y_pred, average='weighted')
    accuracy = accuracy_score(filtered_y_test, filtered_y_pred)

    # write to file
    with open(f'{model_path}/results.txt', 'w') as f:
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1-Score: {f1:.2f}\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        
    # print results
    print(f'Precision: {precision:.2f}\n')
    print(f'Recall: {recall:.2f}\n')
    print(f'F1-Score: {f1:.2f}\n')
    print(f'Accuracy: {accuracy:.2f}\n')
            
    # UI_loop(filtered_boards, filtered_y_pred, filtered_y_test)
        
def single_prediction(input_file, model_name, folder, batch):
    
    # load trained model from file
    if batch:
        model_path = f'{PATH_PREFIX}/{model_name}/{model_name}_{folder}_batch'
    else:
        model_path = f'{PATH_PREFIX}/{model_name}/{model_name}_{folder}'
    
    model = load(model_path + '/model.joblib')
    pca = load(model_path + '/pca.joblib')
    scaler = load(model_path + '/scaler.joblib')

    # check if given filename exists
    if not os.path.isfile(DATA_PREFIX + input_file + '.csv'):
        print("ERROR: File not found")
        exit()
    
    input_data = pd.read_csv(DATA_PREFIX + input_file + '.csv')
    
    # get input data and board
    X = input_data.drop(columns=['board_pos'])
    board = input_data[['board_pos']].iloc[0][0]
    
    # scale and perform pca
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # get correct number of components
    X_pca = X_pca[:, :folder]
    
    # make predictions
    y_probs = model.predict_proba(X_pca)
    
    # setup label encoder
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    
    # decode labels
    decoded_labels = le.inverse_transform(model.classes_)
    
    print("Predicted Moves:\n")
    
    # whether a legal prediction has been found
    preds = []
    
    # loop through probabilities
    for probs in y_probs:
        
        # sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_classes = decoded_labels[sorted_indices]
        
        # for each label and probability
        for label, prob in zip(sorted_classes, sorted_probs):
            # print if it is legal
            if is_legal(board, label):
                preds.append((decode_std(label), prob))
                
    # print best prediction (if no legal predictions they choose random from legal moves list)
    if preds == []:
        # fallback_move = random.choice(legal_moves)
        print(f"Model couldn't provide a valid prediction")
    else:
        print(f"Model predicted: {preds[0][0]}")
      
def is_legal(board, move, method="std"):
    """
    Checks if a move is legal for a given board

    :param board: board position as FEN
    :param move: move to check
    :param method: method of encoding used ('hash' or 'std')
    
    :returns: True or False if move is legal or not respectively
    """
        
    # check encoding method used
    if method == "std":
        
        # convert to string
        move = str(move)
        
        # check correct length
        if len(move) < 4 or len(move) > 5:
            return False
        
        # check contains wrong chars
        s = "12345678"
        for char in move:
            if char not in s:
                return False

        # check for null move E.g. c2c2
        if move[0] == move[2] and move[1] == move[3]: return False

    
        
        # if move includes a promotion
        if len(move) >= 5:
            
            # check correct char
            if move[4] not in "1234":
                return False
            
            
        # decode (only decoding 1st and 3rd chars as numbers aren't changed E.g. e5d7 == 5547)
        new_str = decode_std(move)
            
        # convert move to python-chess move format
        new_move = chess.Move.from_uci(new_str)
        
        # return True if move is in set of legal moves, else return False
        return new_move in chess.Board(board).legal_moves
        
def decode_std(move):
    move = str(move)
    
    res = DECODING_TABLE[move[0]] + move[1] + DECODING_TABLE[move[2]] + move[3]
    
    if len(move) > 4:
        res += PROMOTION_DECODING_TABLE[move[4]]
        
    return res
    
def display_single_result(predictions):
    
    # print all predicted moves and their probabilities
    print("Predicted Moves:")
    for move, prob in predictions:
        print(f"Move: {decode_std(move)}, Probability: {prob}")
    
    # print best prediction (if no legal predictions they choose random from legal moves list)
    if not predictions:
        # fallback_move = random.choice(legal_moves)
        print(f"Model couldn't provide a valid prediction")
    else:
        best_move, _ = max(predictions, key=lambda x: x[1])
        print(f"Model predicted: {decode_std(best_move)}")
        
    # # plot the importance
    # plt.figure(figsize=(10, 6))
    # plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    # plt.xlabel('Importance')
    # plt.ylabel('Feature')
    # plt.title('Feature Importance')
    # plt.show() 

def UI_loop(boards, y_pred, y_test):
    pos = int(input("Please enter a number: "))
    while pos != -1:
        print(chess.Board(boards[pos]))
        print("Predicted:", decode_std(y_pred[pos]))
        print("Actual:", decode_std(y_test[pos]))
        pos = int(input("Please enter a number: "))
    
def main():
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('type', choices=['multiple', 'single'], help="Type of prediction")
    parser.add_argument('--model', choices=['ebm', 'gb'], required=True, help="Model file to predict with")
    parser.add_argument('--n_comps', type=int, required=True, help='Number of components used to train model')
    parser.add_argument('--input', type=str, required=False, help='Input file for prediction')
    parser.add_argument('--batch', action='store_true', help='If the model was batch trained')
    args = parser.parse_args()
    
    # check required arguments are present
    if args.type == 'multiple':
        make_predictions(args.model, args.n_comps, "prediction", args.batch, args.input)
    elif args.type == 'single':
        if args.i is None:
            print("Single input must be specified (-i)")
            exit()
        else:
            single_prediction(args.input, args.model, args.n_comps, args.batch)
    
    # make predictions
    
    # interpret the model
    # interpret_model()
    
    # display predictions
    # display_single_result(preds)

    
if __name__ == '__main__':
    main()