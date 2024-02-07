# Imports
from preprocessing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support


# ----------------------------------------------------------------
    
def train_models(df):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also saves X_test to csv file.

    :param df: the dataframe to train the models on
    """
    
    # create input and output features
    X = df.drop(columns=['board_pos', 'next_move_encoded'])
    y = df['next_move_encoded']
    
    # split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # create model
    gb = GradientBoostingClassifier(n_estimators=1000, min_samples_split=2, max_features=5, random_state=42)
    
    # fit model
    gb.fit(X_train, y_train)
    
    # save model to file
    dump(gb, './data/gb.joblib')
    
    # save test data to csv
    X_test.to_csv('./data/split_test_data.csv', index=False)
    
def make_predictions(boards):
    """
    Makes predictions using trained models and test data

    :param X_test: test data for predictions
    :param board: board positions to check for legal moves
    """
    
    # get test data from train_test_split
    X_test = pd.read_csv('./data/split_test_data.csv')
    
    # load trained model from file
    model = load('./data/gb.joblib')
    
    # extract features from test data
    feature_names = X_test.columns
    
    # feature importance
    feature_importances = model.feature_importances_
    
    # df of features and their importance
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    
    # sort the df by importance
    feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # plot the importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()
    
    # TODO -- Currently only predicting one position -> Loop through all positions and save predicitons for each
    
    # transform test data and board position
    test_data = X_test.to_numpy()
    X_single_pos = test_data[0]
    X_single_pos = X_single_pos.reshape(1, -1)
    X_board_pos = chess.Board(boards.iloc[0])
    
    # get legal moves for board position
    legal_moves = X_board_pos.legal_moves
    
    # make predictions with probabilities
    predicted_probs = model.predict_proba(X_single_pos)[0]
    
    # filter predictions by legality
    filtered_preds = [(move, prob) for move, prob in zip(model.classes_, predicted_probs) if check_if_legal(X_board_pos, move)]
    
    # print all predicted moves and their probabilities
    print("Predicted Moves:")
    for move, prob in filtered_preds:
        print(f"Move: {move}, Probability: {prob}")
    
    # print best prediction (if no legal predictions they choose random from legal moves list)
    if not filtered_preds:
        fallback_move = random.choice(legal_moves)
        print(f"Model couldn't provide a valid prediction. Using fallback move: {fallback_move}")
    else:
        best_move, _ = max(filtered_preds, key=lambda x: x[1])
        print(f"Model predicted: {best_move}")
    
def check_if_legal(board, move, method="std"):
    """
    Checks if a move is legal for a given board

    :param board: board position as FEN
    :param move: move to check
    :param method: method of encoding used ('hash' or 'std')
    
    :returns: True or False if move is legal or not respectively
    """
    
    # check encoding method used
    if method == "hash":
        pass
    else:
        # create hashmaps
        mapping_dict = {
            '1': 'a',
            '2': 'b',
            '3': 'c',
            '4': 'd',
            '5': 'e',
            '6': 'f',
            '7': 'g',
            '8': 'h',
        }
    
        
        # convert move to str and decode (only decoding 1st and 3rd chars as numbers aren't changed E.g. e5d7 == 5547)
        move = str(move)
        new_str = mapping_dict[move[0]] + move[1] + mapping_dict[move[2]] + move[3]
        
        # if move includes a promotion
        if len(move) >= 5:
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
        move = chess.Move.from_uci(new_str)
        
        # return True if move is in set of legal moves, else return False
        return move in board.legal_moves
         
def preprocess_data(filename):
    """
    Preprocess the dataframe and save to csv file
    
    :param filename: name of file that contains chess games as PGNs
    """
    
    # get file path
    path = "./data/" + filename
    # generate dataframe from games
    df = generate_df(path, "std", "std")
    # save to csv file
    df.to_csv('./data/games.csv', index=False)
    
def main():
    # # preprocess the dataframe
    preprocess_data('lichess-2023-11')
    
    # grab df from games.csv
    df = pd.read_csv('./data/games.csv')
    
    # # train the models    
    train_models(df)
    
    # grab board positions (for checking legal move)
    boards = df['board_pos']
    
    # make predictions
    make_predictions(boards)

    
if __name__ == '__main__':
    main()