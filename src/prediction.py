# Imports
from preprocessing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

PATH_PREFIX = '../models/'
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
    
def make_predictions():
    """
    Makes predictions using trained models and test data

    :param X_test: test data for predictions
    :param board: board positions to check for legal moves
    """
    
    # get test data from train_test_split
    X_test = pd.read_csv(PATH_PREFIX + 'X_test.csv')
    y_train = pd.read_csv(PATH_PREFIX + 'y_train.csv')
    y_test = pd.read_csv(PATH_PREFIX + 'y_test.csv')
    boards = pd.read_csv(PATH_PREFIX + 'Boards.csv')
    
    # load trained model from file
    model = load(PATH_PREFIX + 'gb.joblib')
    
    # # extract features from test data
    # feature_names = X_test.columns
    
    # # feature importance
    # feature_importances = model.feature_importances_
    
    # # df of features and their importance
    # feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    
    # # sort the df by importance
    # feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # make predictions with probabilities
    y_pred = model.predict(X_test)
    
    # Decode
    le = LabelEncoder()
    le.fit(y_train)
    
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
         

    # print(accuracy_score(y_test, filtered_y_pred))
    # print(f1_score(y_test, filtered_y_pred, average='weighted'))
    
    UI_loop(filtered_boards, filtered_y_pred, filtered_y_test)
    
    return y_pred
    
def single_prediction(pos):
    
    # load trained model from file
    model = load(PATH_PREFIX + 'gb.joblib')
    
    # get training data to fit the encoder
    y_train = pd.read_csv(PATH_PREFIX + 'y_train.csv')
    
    X_board_pos = pos[0]
    
    # get legal moves for board position
    legal_moves = X_board_pos.legal_moves
    
    # make predictions with probabilities
    predicted_probs = model.predict_proba(pos)
    
    # Decode
    le = LabelEncoder()
    le.fit(y_train)
    
    move_classes = le.inverse_transform(model.classes_)
        
    filtered_preds = [(move, prob) for move, prob in zip(move_classes, predicted_probs) if is_legal(X_board_pos, move)]
    
    return filtered_preds
    
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

    
        # decode (only decoding 1st and 3rd chars as numbers aren't changed E.g. e5d7 == 5547)
        new_str = decode_std(move)
        
        # if move includes a promotion
        if len(move) >= 5:
            
            # check correct char
            if move[4] not in "1234":
                return False
            
            # promotion hashmap
            promotion_dict = {
                '1': 'r',
                '2': 'n',
                '3': 'b',
                '4': 'q'
            }
            
            # encode last char
            new_str += promotion_dict[move[4]]
            
        # convert move to python-chess move format
        new_move = chess.Move.from_uci(new_str)
        
        # return True if move is in set of legal moves, else return False
        return new_move in chess.Board(board).legal_moves
        
def decode_std(move):
    move = str(move)

    res = DECODING_TABLE[move[0]] + move[1] + DECODING_TABLE[move[2]] + move[3]
        
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
    
    # make predictions
    preds = make_predictions()
    
    # display predictions
    # display_single_result(preds)

    
if __name__ == '__main__':
    main()