import argparse
import os
import pandas as pd
from training import train_models
from pca import split_data, make_dir

import os
import argparse
import pandas as pd

from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- GLOBALS ---------------- #

DATA_PREFIX = '../data/'
FEATURE_PREFIX = '../feature_importance/'
RESULTS_PREFIX = '../results/'
PCA_PREFIX = '../PCA/'

def create_pca(df, num_comps_start, num_comps_end):
    """
    Perform scaling and PCA on data
    
    :param df: dataframe to use
    :num_comps_start: list of component values to use for model 1
    :num_comps_end: list of component values to use for model 2
    
    :return: saves pca values to file
    """
    
    # ---------------- DEFINE VARIABLES ---------------- #
    
    # scaler
    scaler_start = StandardScaler()
    scaler_end = StandardScaler()
    
    # pca
    pca_start = PCA(n_components=num_comps_start)
    pca_end = PCA(n_components=num_comps_end)
    
    # get values for both models
    X_train_start, X_val_start, y_train_start, y_val_start, X_train_end, X_val_end, y_train_end, y_val_end, boards_start, boards_end, train_boards_start, train_boards_end = split_data(df)
    
    # save start square column for model 2 testing
    start_squares = X_val_end[['start_square']]

    # ---------------- SCALING & PCA ---------------- #
    
    # apply scaling
    X_train_start = scaler_start.fit_transform(X_train_start)
    X_train_end = scaler_end.fit_transform(X_train_end)
    X_val_start = scaler_start.transform(X_val_start)
    
    # apply pca
    pca_start = pca_start.fit(X_train_start)
    pca_end = pca_end.fit(X_train_end)
    
    pca_train_start = pca_start.transform(X_train_start)
    pca_val_start = pca_start.transform(X_val_start)
    pca_train_end = pca_end.transform(X_train_end)

    # if folder doesn't exist then create
    pca_path = f'{PCA_PREFIX}/{str(num_comps_start)}_{str(num_comps_end)}/'
    make_dir(pca_path)
    
    # save starting squares for model 2 testing
    start_squares.to_csv(pca_path + f'start_squares.csv', index=False)
    
    # check number given is valid, if not set to max or min
    if num_comps_start > pca_start.n_components_:
        print(f'ERROR: Number of components invalid.\nSetting to MAX={pca_start.n_components_}')
        num_comps_start = pca_start.n_components_
    elif num_comps_end > pca_end.n_components_:
        print(f'ERROR: Number of components invalid.\nSetting to MAX={pca_end.n_components_}')
        num_comps_end = pca_end.n_components_
    elif num_comps_start <= 0:
        print(f'ERROR: Number of components invalid.\nSetting to MIN=1')
        num_comps_start = 1
    elif num_comps_end <= 0:
        print(f'ERROR: Number of components invalid.\nSetting to MIN=1')
        num_comps_end = 1

    # save boards to file
    train_boards_start.to_csv(pca_path + 'train_boards_start.csv', index=False)
    train_boards_end.to_csv(pca_path + 'train_boards_end.csv', index=False)
    boards_start.to_csv(pca_path + 'boards_start.csv', index=False)
    boards_end.to_csv(pca_path + 'boards_end.csv', index=False)
        
    # ---------------- CREATE DATAFRAME ---------------- #
    
    # generate dataframe
    df_train_start = pd.DataFrame(pca_train_start, columns=[f'PC{i+1}' for i in range(num_comps_start)])
    df_train_end = pd.DataFrame(pca_train_end, columns=[f'PC{i+1}' for i in range(num_comps_end)])
    
    df_val_start = pd.DataFrame(pca_val_start, columns=[f'PC{i+1}' for i in range(num_comps_start)])
    
    # ---------------- SAVE VARIABLES TO FILE ---------------- #
    
    # Save to csv files for future use (predictions)
    y_train_start.to_csv(pca_path + 'y_train_start.csv', index=False)
    y_train_end.to_csv(pca_path + 'y_train_end.csv', index=False)
    y_val_start.to_csv(pca_path + 'y_val_start.csv', index=False)
    y_val_end.to_csv(pca_path + 'y_val_end.csv', index=False)
    df_train_start.to_csv(pca_path + f'X_train_start.csv', index=False)
    df_train_end.to_csv(pca_path + f'X_train_end.csv', index=False)
    df_val_start.to_csv(pca_path + f'X_val_start.csv', index=False)
    X_val_end.to_csv(pca_path + f'X_val_end.csv', index=False)
    
    # Save PCA and Scaler for future predictions
    dump(pca_start, pca_path + 'pca_start.joblib')
    dump(scaler_start, pca_path + 'scaler_start.joblib')
    dump(pca_end, pca_path + 'pca_end.joblib')
    dump(scaler_end, pca_path + 'scaler_end.joblib')

def main():

    # ---------------- ARGUMENT HANDLING ---------------- #
    parser = argparse.ArgumentParser(description="Run PCA range on certain models")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', nargs='+', required=False, help='List of component values for model 1 (E.g. 1 2 3)')
    parser.add_argument('--comps_2', nargs='+', required=False, help='List of component values for model 2 (E.g. 1 2 3)')
    parser.add_argument('--file', type=str, required=True, help="CSV file to use (excluding extensions)")
    args = parser.parse_args()
    
    # check if given filename exists
    if not os.path.isfile(DATA_PREFIX + args.file + '.csv'):
        print("ERROR: File not found")
        exit()
    
    if args.n_comps and (args.comps_1 is not None or args.comps_2):
        print("n_comps can't be used if comps_1 or comps_2 are used.")
        exit()
    elif (args.comps_1 is None and args.comps_2) or (args.comps_1 and args.comps_2 is None):
        print("If n_comps is not used, both comps_1 and comps_2 must be specified.")
        exit()
        
    # create dataframe
    df = pd.read_csv(DATA_PREFIX + args.file + '.csv')
    
    # change datatypes to category where applicable
    df['start_square'] = df['start_square'].astype('category')
    df['end_square'] = df['end_square'].astype('category')
    df['turn'] = df['turn'].astype('category')
    
    # dict of accuray and PCA number
    acc_dict = {}

    # loop through and train each model on different pca component numbers
    if args.n_comps:
        for num in args.n_comps:
            create_pca(df, int(num), int(num))
            acc_dict[f"{num}_{num}"] = train_models(args.model, int(num), int(num))
            
    else:
        for num1 in args.comps_1:
            for num2 in args.comps_2:
                create_pca(df, int(num1), int(num2))
                acc_dict[f"{num1}_{num2}"] = train_models(args.model, int(num1), int(num2))
            
    # get the best accuracy
    best_acc, best_num = 0, ""
    for key, item in acc_dict.items():
        if item > best_acc:
            best_acc = item
            best_num = key
            
    print(f"\nBest Accuracy: {best_acc} with {best_num} PCA components")
    
if __name__ == '__main__':
    main()