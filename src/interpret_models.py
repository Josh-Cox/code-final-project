import shap
import pandas as pd
from joblib import load
import os
import argparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from joblib import load

PATH_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'
DATA_PREFIX = '../data/'

def interpret_model(model, df_pca, y_train, pca, scaler):   
    pass

    
def main():
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', required=True, help='Number of components to train model 1 with (folder must exist under "PCA/")')
    parser.add_argument('--comps_2', required=True, help='Number of components to train model 2 with (folder must exist under "PCA/")')
    args = parser.parse_args()

    model_path = f"{PATH_PREFIX}{args.model}/{args.model}_{args.comps_1}_{args.comps_2}/"
    pca_path = f"{PCA_PREFIX}{args.comps_1}_{args.comps_2}/"
    pca_start = load(pca_path + 'pca_start.joblib')
    scaler_start = load(pca_path + 'scaler_start.joblib')

    # check if model exists
    if not os.path.isfile(f'{model_path}model_start.joblib') or not os.path.isfile(f'{model_path}model_end.joblib'):
        print("ERROR: No trained model exists. Train using python training.py")
        exit()

    
    # load trained model from file
    model = load(model_path + 'model_start.joblib')
    X_test = pd.read_csv(pca_path + 'X_val_start.csv')
    y_train = pd.read_csv(pca_path + 'y_val_start.csv')
    interpret_model(model, X_test, y_train, pca_start, scaler_start)

if __name__ == "__main__":
    main()