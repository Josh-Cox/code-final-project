# ---------------- IMPORTS ---------------- #

import pandas as pd
import numpy as np
import time
import argparse
import os

from halo import Halo
from joblib import load

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import *

# ---------------- GLOBALS ---------------- #

PATH_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'
DATA_PREFIX = '../data/'

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

PROMOTION_DECODING_TABLE = {
    '1': 'r',
    '2': 'n',
    '3': 'b',
    '4': 'q'
}

SQUARES = {
    "A1": chess.A1, "A2": chess.A2, "A3": chess.A3, "A4": chess.A4, "A5": chess.A5, "A6": chess.A6, "A7": chess.A7, "A8": chess.A8,
    "B1": chess.B1, "B2": chess.B2, "B3": chess.B3, "B4": chess.B4, "B5": chess.B5, "B6": chess.B6, "B7": chess.B7, "B8": chess.B8,
    "C1": chess.C1, "C2": chess.C2, "C3": chess.C3, "C4": chess.C4, "C5": chess.C5, "C6": chess.C6, "C7": chess.C7, "C8": chess.C8,
    "D1": chess.D1, "D2": chess.D2, "D3": chess.D3, "D4": chess.D4, "D5": chess.D5, "D6": chess.D6, "D7": chess.D7, "D8": chess.D8,
    "E1": chess.E1, "E2": chess.E2, "E3": chess.E3, "E4": chess.E4, "E5": chess.E5, "E6": chess.E6, "E7": chess.E7, "E8": chess.E8,
    "F1": chess.F1, "F2": chess.F2, "F3": chess.F3, "F4": chess.F4, "F5": chess.F5, "F6": chess.F6, "F7": chess.F7, "F8": chess.F8,
    "G1": chess.G1, "G2": chess.G2, "G3": chess.G3, "G4": chess.G4, "G5": chess.G5, "G6": chess.G6, "G7": chess.G7, "G8": chess.G8,
    "H1": chess.H1, "H2": chess.H2, "H3": chess.H3, "H4": chess.H4, "H5": chess.H5, "H6": chess.H6, "H7": chess.H7, "H8": chess.H8
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
    
def make_predictions(model_name, comps_1, comps_2, batch, files=None):
    """
    Makes predictions using trained models

    :param model_name: name of the trained model
    :param n_components: number of components used to train model (needs to be same for input data)
    :param pred_type: prediction type ('multiple' or 'single')
    :param batch: whether the model was batch trained (for filename)
    :param files: filename of input position (not neede if using test set in training.py)
    """
    
    # ---------------- SETUP VARIABLES ---------------- #

    X_test = None
    y_test = None
    boards = None
    
    # path to model folder
    if batch:
        model_path = f'{PATH_PREFIX}/{str(model_name)}/{str(model_name)}_{str(comps_1)}_{str(comps_2)}_batch/'
    else:
        model_path = f'{PATH_PREFIX}/{str(model_name)}/{str(model_name)}_{str(comps_1)}_{str(comps_2)}/'
        
    # check if model exists
    if not os.path.isfile(f'{model_path}/model_start.joblib') or not os.path.isfile(f'{model_path}/model_end.joblib'):
        print("ERROR: No trained model exists. Train using python training.py")
        exit()

    # check if given filename exists
    if not os.path.isfile(f'{DATA_PREFIX}/{files}.csv'):
        print("ERROR: File not found")
        exit()
    
    # load pca and scaler
    pca_start = load(model_path + '/pca_start.joblib')
    pca_end = load(model_path + '/pca_end.joblib')
    scaler_start = load(model_path + '/scaler_start.joblib')
    scaler_end = load(model_path + '/scaler_end.joblib')
    
    # setup label encoders
    le_start = LabelEncoder()
    le_end = LabelEncoder()
    le_start.classes_ = np.load(model_path + 'classes_start.npy')
    le_end.classes_ = np.load(model_path + 'classes_end.npy')
    
    

    # get input data
    input_data = pd.read_csv(f'{DATA_PREFIX}/{files}.csv')
        
    # get input data and board
    X_start = input_data.drop(columns=['board_pos', 'end_square', 'start_square'])
    X_end = input_data.drop(columns=['board_pos', 'end_square', 'start_square'])
    y_start = input_data[['start_square']]
    y_end = input_data[['end_square']]
    boards = input_data[['board_pos']]
    
    # scale and perform pca
    X_start = scaler_start.transform(X_start)
    X_start = pca_start.transform(X_start)
            
    
    # ---------------- PREDICTING START SQUARE ---------------- #
    
    # start time for predicting
    start_time = time.time()
    
    # load trained model from file
    model_start = load(f'{model_path}/model_start.joblib')
    
    # initialise pred_start
    pred_start = None
    
    # make predictions with probabilities
    with Halo(text=f'Predicting starting squares', color='grey', spinner="dots3"):
        pred_start = model_start.predict(X_start)
    

    # decode predictions
    pred_start_decoded = le_start.inverse_transform(pred_start)

    # convert to lists
    boards = list(boards['board_pos'])
    y_start = list(y_start['start_square'])
    y_end = list(y_end['end_square'])
    
    # create legal predictions lists
    filtered_pred_start = []
    filtered_val_start = []
    
    # filter illegal moves
    for i in range(len(pred_start_decoded)):
        # check is legal (non-empty square)
        if is_legal_start(boards[i], pred_start_decoded[i]):
            filtered_pred_start.append(pred_start_decoded[i])
            filtered_val_start.append(y_start[i])
            
    # ---------------- CREARTE DATA FOR MODEL 2 ---------------- #

    X_end['start_square'] = pred_start_decoded
    
    # scale and perform pca
    X_end = scaler_end.transform(X_end)
    X_end = pca_end.transform(X_end)

    # ---------------- PREDICTING END SQUARE ---------------- #
    
    # start time for predicting
    start_time = time.time()
    
    # load trained model from file
    model_end = load(f'{model_path}/model_end.joblib')
    
    # make predictions with probabilities
    with Halo(text=f'Predicting ending squares', color='grey', spinner="dots3"):
        pred_end = model_end.predict(X_end)
        
    # end time for predicting
    end_time = time.time()
    
    print("\n--- FINISHED PREDICTIONS ---")
    print(f'\n--- TIME ELAPSED: {end_time - start_time} ---\n')
    
    # ---------------- DECODE & FILTER PREDICTIONS ---------------- #

    
    # decode predictions
    pred_end = le_end.inverse_transform(pred_end)
    
    filtered_pred_end = []
    filtered_val_end = []
        
    for i in range(len(pred_end)):
        # convert to move
        pred_move = str(y_start[i]) + str(pred_end[i])
        # check is legal
        if is_legal(boards[i], pred_move):
            filtered_pred_end.append(pred_end[i])
            filtered_val_end.append(y_end[i])
    
    # create legal predictions lists
    filtered_preds = []
    filtered_acc = []
            
    # loop through predictions, keeping any legal moves (not just correct ones, legally allowed moves in the current board position)
    for i in range(len(pred_end)):
        pred_move = str(pred_start_decoded[i]) + str(pred_end[i])
        # print(f"\nPred: {pred_move}  Move: {str(y_start[i]) + str(y_end[i])} \nBoard\n{chess.Board(boards[i])}")
        if is_legal(boards[i], pred_move):
            filtered_preds.append(int(pred_move))
            filtered_acc.append(int(str(y_start[i]) + str(y_end[i])))
            

    # ---------------- SCORING ---------------- #
    
    precision = precision_score(filtered_acc, filtered_preds, average='weighted', zero_division=0)
    recall = recall_score(filtered_acc, filtered_preds, average='weighted', zero_division=0)
    f1 = f1_score(filtered_acc, filtered_preds, average='weighted')
    accuracy = accuracy_score(filtered_acc, filtered_preds)

    # ---------------- OUTPUT SCORES ---------------- #
    
    # save scores to text file
    with open(f'{model_path}/results.txt', 'w') as f:
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1-Score: {f1:.2f}\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        
    # print results
    print(f"\nStart Square Prediction Accuracy: {accuracy_score(filtered_val_start, filtered_pred_start)}")
    print(f"End Square Prediction Accuracy: {accuracy_score(filtered_val_end, filtered_pred_end)}")
    print("\nMove Prediction Scores:\n")
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
            
    # UI_loop(filtered_boards, filtered_y_pred, filtered_y_test)
        
def single_prediction(input_file, model_name, n_components, batch):
    """
    Genreate predictions and probabilities for a single input position
    
    :param input_file: filename of input position
    :param model_name: name of the model to use
    :param n_components: number of PCA components used to train model
    :param batch: whether the model was batch trained
    """
    
    # ---------------- SETUP VARIBALES ---------------- #
    
    # load trained model from file
    if batch:
        model_path = f'{PATH_PREFIX}/{model_name}/{model_name}_{n_components}_batch'
    else:
        model_path = f'{PATH_PREFIX}/{model_name}/{model_name}_{n_components}'
    
    model = load(model_path + '/model.joblib')

    # load scaler and PCA
    pca = load(model_path + '/pca.joblib')
    scaler = load(model_path + '/scaler.joblib')

    # check if given filename exists
    if not os.path.isfile(DATA_PREFIX + input_file + '.csv'):
        print("ERROR: File not found")
        exit()
    
    # get input data
    input_data = pd.read_csv(DATA_PREFIX + input_file + '.csv')
    
    # get input data and board
    X = input_data.drop(columns=['board_pos'])
    board = input_data[['board_pos']].iloc[0][0]
    
    # scale and perform pca
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # get correct number of components
    X_pca = X_pca[:, :n_components]
    
    # ---------------- PREDICTING ---------------- # 
    
    # make predictions
    y_probs = model.predict_proba(X_pca)
    
    # setup label encoder
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    
    # decode labels
    decoded_labels = le.inverse_transform(model.classes_)
    
    print("Predicted Moves:\n")
    
    # list of legal predictions
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
        # TODO: Add a fallback to pick a random move from legal moves list
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
    
    :return: True or False if move is legal or not
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
    
def is_legal_start(board, square):
    
    # check correct length
    if len(str(square)) != 2:
        return False

    # check contains wrong chars
    s = "12345678"
    for char in str(square):
        if char not in s:
            return False

    board = chess.Board()
    
    square = decode_start(square).upper()
    square = SQUARES[square]
    
    if board.piece_at(square) is None:
        return False
    else:
        return True

def decode_std(move):
    """
    Decodes a move from numbers to characters

    :param move: move to decode
    
    :return: decoded move
    """
    
    # convert move to string
    move = str(move)
    
    # convert first and third characters using one-hot encoding
    res = DECODING_TABLE[move[0]] + move[1] + DECODING_TABLE[move[2]] + move[3]
    
    # if move has promotion
    if len(move) > 4:
        # decode promotion value
        res += PROMOTION_DECODING_TABLE[move[4]]
        
    return res

def decode_start(square):
    """
    Decodes a square from numbers to characters

    :param sqaure: sqaure to decode
    
    :return: decoded sqaure
    """
    
    s = str(square)
    res = DECODING_TABLE[s[0]] + s[1]
    
    return res

def display_single_result(predictions):
    """
    Displays a single prediction result
    
    :param predictions: predictions to display
    """
    
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
    """
    Command line interface for the user to traverse all of the predicted moves, along with the board and the actual moves
    
    :param boards: list of board that predictions were made on
    :param y_pred: predicted moves
    :param y_test: actual moves
    """
    
    # ask user for prediction number
    pos = int(input("Please enter a number: "))
    # loop until user inputs -1
    while pos != -1:
        # print board, prediction and actual move
        print(chess.Board(boards[pos]))
        print("Predicted:", decode_std(y_pred[pos]))
        print("Actual:", decode_std(y_test[pos]))
        pos = int(input("Please enter a number: "))
    
def main():
    
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('type', choices=['multiple', 'single'], help="Type of prediction")
    parser.add_argument('--model', choices=['ebm', 'gb', 'dt'], required=True, help="Model file to predict with")
    parser.add_argument('--comps_1', type=int, required=True, help='Number of components used to train model 1')
    parser.add_argument('--comps_2', type=int, required=True, help='Number of components used to train model 2')
    parser.add_argument('--input', type=str, required=True, help='Input file for prediction')
    parser.add_argument('--batch', action='store_true', help='If the model was batch trained')
    args = parser.parse_args()
    
    
    # check required arguments are present
    if args.type == 'multiple':
        make_predictions(args.model, args.comps_1, args.comps_2, args.batch, args.input)
    elif args.type == 'single':
        if args.i is None:
            print("Single input must be specified (--input)")
            exit()
        else:
            single_prediction(args.input, args.model, args.n_comps, args.batch)

    
if __name__ == '__main__':
    main()