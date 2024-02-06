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
    
def train_models(X_train, y_train):
    
    # # Create model
    gb = GradientBoostingClassifier(n_estimators=1000, min_samples_split=2, max_features=5, random_state=42)
    
    # Fit model
    gb.fit(X_train, y_train)
    
    dump(gb, 'gb.joblib')
    
    # np.savetxt('test.csv', y_test, delimiter=',')
    # np.savetxt('pred.csv', y_pred, delimiter=',')
    # np.savetxt('classes.csv', gb.classes_, delimiter=',')
    
def make_predictions(X_test):
    
    model = load('gb.joblib')
    
    test_data = X_test.drop(columns=['legal_moves']).to_numpy()
    legal_moves = X_test['legal_moves'].to_numpy()

    X_single_pos = test_data[0]
    X_single_pos = X_single_pos.reshape(1, -1)
    
    predicted_probs = model.predict_proba(X_single_pos)[0]
    X_legal_moves = legal_moves[0]
    
    filtered_preds = [(move, prob) for move, prob in zip(model.classes_, predicted_probs) if move in legal_moves]
    
    if not filtered_preds:
        fallback_move = random.choice(legal_moves)
        print(f"Model couldn't provide a valid prediction. Using fallback move: {fallback_move}")
    else:
        best_move, _ = max(filtered_preds, key=lambda x: x[1])
        print(f"Model predicted: {best_move}")
    
    
    
    
def preprocess_data(filename):
    path = "../data/" + filename
    df = generate_df(path)
    df.to_csv('games.csv', index=False)
    

def main():
    # preprocess_data('lichess-2023-11')
    
    # grab df from games.csv
    df = pd.read_csv('games.csv')
    
    # Create input and output features
    X = df.drop(columns=['next_move'])
    y = df['next_move']
    
    # Split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = X_train.drop(columns=['legal_moves'])
    
    # train_models(X_train, y_train)
    
    make_predictions(X_test)

    
if __name__ == '__main__':
    main()