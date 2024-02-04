# Imports
from preprocessing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


def main():
    df = generate_df()

    X = df.drop(columns=['next_move'])
    y = df['next_move']
    
    # Split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # # Create model
    gb = GradientBoostingClassifier(n_estimators=1000, min_samples_split=2, max_features=2, random_state=42)
    
    # Fit model
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    
    # Visualise
    plt.scatter(y_test, y_pred, color="cadetblue", label="Predicted vs Actual")
    plt.title("GB")
    plt.xlabel("Prediction Data")
    plt.ylabel("Actual Data")
    plt.show()
    
if __name__ == '__main__':
    main()