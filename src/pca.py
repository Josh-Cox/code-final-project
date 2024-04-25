# ---------------- IMPORTS ---------------- #

import os
import argparse
import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ---------------- GLOBALS ---------------- #

DATA_PREFIX = '../data/'
FEATURE_PREFIX = '../feature_importance/'
RESULTS_PREFIX = '../results/'
PCA_PREFIX = '../PCA/'

ALL_FEATURES = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']
    
def make_dir(path):
    """
    Checks if a directory exists, if not then creates
    
    :param path: directory path
    """
    
    if not os.path.isdir(str(path)):
        os.makedirs(str(path))

def split_data(df):
    """
    Split the dataframe into train and validation (also test if needed)
    
    :param df: dataframe to split
    
    :return: train, validation and test sets
    """
    
    # ---------------- DEFINE X AND Y ---------------- #

    X_start = df.drop(columns=['start_square', 'end_square'])
    y_start = df[['start_square']]
    
    X_end = df.drop(columns=['end_square'])
    y_end = df[['end_square']]
    
    # ---------------- SPLIT DATA ---------------- #
    
    # training and validation    
    X_train_start, X_val_start, y_train_start, y_val_start = train_test_split(X_start, y_start, test_size=0.2, random_state=42)
    X_train_end, X_val_end, y_train_end, y_val_end = train_test_split(X_end, y_end, test_size=0.2, random_state=42)
    
    # drop board position column
    boards_start = X_val_start[['board_pos']]
    boards_end = X_val_end[['board_pos']]
    train_boards_start = X_train_start[['board_pos']]
    train_boards_end = X_train_end[['board_pos']]
    X_train_start = X_train_start.drop(columns=['board_pos'])
    X_val_start = X_val_start.drop(columns=['board_pos'])
    X_train_end = X_train_end.drop(columns=['board_pos'])
    X_val_end = X_val_end.drop(columns=['board_pos'])
    
    return X_train_start, X_val_start, y_train_start, y_val_start, X_train_end, X_val_end, y_train_end, y_val_end, boards_start, boards_end, train_boards_start,train_boards_end
    
def plot_pca(pca_start, pca_end, plot_type, per_var_start, per_var_end, prop_var_start, prop_var_end, labels_start, labels_end):
    """
    Plots the pca on differnt graphs to visualize variance of components
    
    :param pca_start: the model 1 pca to plot
    :param pca_end: the model 2 pca to plot
    :param plot_type: the type of plot ('bar' or 'elbow')
    :param per_var_start: explained variance for model 1 as a percentage
    :param per_var_end: explained variance for model 2 as a percentage
    :param prop_var_start: explained variance for model 1
    :param prop_var_end: explained variance for model 2
    :param labels_start: labels for model 1 graph
    :param labels_end: labels for model 2 graph

    :return: shows a plot of PCA components
    """
    
    # ---------------- PLOT PCA ---------------- #
    
    # check plot if wanted
    if plot_type is not None:
        # determine which plot
        if plot_type == 'bar':
            # plot variance
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # first subplot
            axs[0].bar(x=range(1, len(per_var_start)+1), height=per_var_start, tick_label=labels_start)
            axs[0].set_ylabel("Percentage of Explained Variance")
            axs[0].set_xlabel("Principal Component")
            axs[0].set_title("Scree Plot 1")
            
            # second subplots
            axs[1].bar(x=range(1, len(per_var_end)+1), height=per_var_end, tick_label=labels_end)
            axs[1].set_ylabel("Percentage of Explained Variance")
            axs[1].set_xlabel("Principal Component")
            axs[1].set_title("Scree Plot 2")
            
            #show
            plt.show()
        else:
            # plot PCA subplots
            PC_number_start = np.arange(pca_start.n_components_) + 1
            PC_number_end = np.arange(pca_end.n_components_) + 1
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # subplot 1
            axs[0].plot(PC_number_start, prop_var_start, 'ro-')
            axs[0].set_title("Scree Plot 1")
            axs[0].set_xlabel("Component Number")
            axs[0].set_ylabel("Proportion of Variance")
            axs[0].grid()
            
            # subplot 2
            axs[1].plot(PC_number_end, prop_var_end, 'ro-')
            axs[1].set_title("Scree Plot 2")
            axs[1].set_xlabel("Component Number")
            axs[1].set_ylabel("Proportion of Variance")
            axs[1].grid()
            
            # show
            plt.show()
             
def pca_analysis(df, plot_type):
    """
    Perform scaling and PCA on data
    
    :param df: dataframe to use
    :plot_type: the type of plot ('bar' or 'elbow')

    :return: performs PCA on data and saves to file
    """
    
    # ---------------- DEFINE VARIABLES ---------------- #
    
    # remove null values
    df.dropna(inplace=True)
    
    # scaler
    scaler_start = StandardScaler()
    scaler_end = StandardScaler()
    # PCA
    pca_start_vis = PCA(0.95) # retain 95% of variance
    pca_end_vis = PCA(0.95) # retain 95% of variance
    
    # get values for both models
    X_train_start, X_val_start, y_train_start, y_val_start, X_train_end, X_val_end, y_train_end, y_val_end, boards_start, boards_end, train_boards_start, train_boards_end = split_data(df)
    
    # save input data before pca and scaling
    X_val_start_og = copy.deepcopy(X_val_start)
    X_val_end_og = copy.deepcopy(X_val_end)
    
    # ---------------- SCALING & PCA ---------------- #
    
    # apply scaling
    X_train_start = scaler_start.fit_transform(X_train_start)
    X_train_end = scaler_end.fit_transform(X_train_end)
    X_val_start = scaler_start.transform(X_val_start)
    
    # apply pca
    pca_start_vis.fit(X_train_start)
    pca_end_vis.fit(X_train_end)
    
    
    # ---------------- PLOT PCA ---------------- #
    
    # variance for bar and elbow plot
    variance_percent_start = np.round(pca_start_vis.explained_variance_ratio_ * 100, decimals=1)
    variance_percent_end = np.round(pca_end_vis.explained_variance_ratio_ * 100, decimals=1)
    
    variance_start = pca_start_vis.explained_variance_ratio_
    variance_end = pca_end_vis.explained_variance_ratio_
    
    # labels for plot
    labels_start = ['PC' + str(x) for x in range(1, len(variance_percent_start)+1)]
    labels_end = ['PC' + str(x) for x in range(1, len(variance_percent_end)+1)]
    
    # plot the pca
    if plot_type:
        plot_pca(pca_start_vis, pca_end_vis, plot_type, variance_percent_start, variance_percent_end, variance_start, variance_end, labels_start, labels_end)
    
    
    # ---------------- GET N COMPONENTS FROM USER ---------------- #
    
    # number of components to use
    num_comps_start = input(f'\nEnter the number of components to save for plot 1 (MAX={pca_start_vis.n_components_}): ')
    num_comps_start = int(num_comps_start)
    num_comps_end = input(f'\nEnter the number of components to save for plot 2 (MAX={pca_end_vis.n_components_}): ')
    num_comps_end = int(num_comps_end)
    
    # if folder doesn't exist then create
    pca_path = f'{PCA_PREFIX}/{str(num_comps_start)}_{str(num_comps_end)}/'
    make_dir(pca_path)
    
    # check number given is valid, if not set to max or min
    if num_comps_start > pca_start_vis.n_components_:
        print(f'ERROR: Number of components invalid.\nSetting to MAX={pca_start_vis.n_components_}')
        num_comps_start = pca_start_vis.n_components_
    elif num_comps_end > pca_end_vis.n_components_:
        print(f'ERROR: Number of components invalid.\nSetting to MAX={pca_end_vis.n_components_}')
        num_comps_end = pca_end_vis.n_components_
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
    
    # save input data before pca and scaling
    X_val_start_og.to_csv(pca_path + 'X_test_start_og.csv', index=False)
    X_val_end_og.to_csv(pca_path + 'X_test_end_og.csv', index=False)
        
    # create PCA with correct number of components
    pca_start = PCA(n_components=num_comps_start)
    pca_end = PCA(n_components=num_comps_end)
    pca_start = pca_start.fit(X_train_start)
    pca_end = pca_end.fit(X_train_end)
    
    # Save PCA and Scaler for future predictions
    dump(pca_start, pca_path + 'pca_start.joblib')
    dump(scaler_start, pca_path + 'scaler_start.joblib')
    dump(pca_end, pca_path + 'pca_end.joblib')
    dump(scaler_end, pca_path + 'scaler_end.joblib')

    pca_train_start = pca_start.transform(X_train_start)
    pca_val_start = pca_start.transform(X_val_start)
    pca_train_end = pca_end.transform(X_train_end)
    
    # ---------------- CREATE DATAFRAME ---------------- #
    
    # generate dataframe
    df_train_start = pd.DataFrame(pca_train_start, columns=[f'PC{i+1}' for i in range(num_comps_start)])
    df_train_end = pd.DataFrame(pca_train_end, columns=[f'PC{i+1}' for i in range(num_comps_end)])
    
    df_val_start = pd.DataFrame(pca_val_start, columns=[f'PC{i+1}' for i in range(num_comps_start)])
    # df_val_end = pd.DataFrame(pca_val_end, columns=[f'PC{i+1}' for i in range(num_comps_end)])
    
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

def main():
    
    # ---------------- ARGUMENT HANDLING ---------------- #
    parser = argparse.ArgumentParser(description="Which functions/plots to run on")
    parser.add_argument('--plot', choices=['bar', 'elbow'], required=False, help="PCA plot type")
    parser.add_argument('--file', type=str, required=True, help="CSV file to use (excluding extensions)")
    args = parser.parse_args()
    
    # ---------------- INITIALISE VALUES ---------------- #
    
    # check if given filename exists
    if not os.path.isfile(DATA_PREFIX + args.file + '.csv'):
        print("ERROR: File not found")
        exit()
    
    # create dataframe
    df = pd.read_csv(DATA_PREFIX + args.file + '.csv')
            
    # change datatypes to category where applicable
    df['start_square'] = df['start_square'].astype('category')
    df['end_square'] = df['end_square'].astype('category')
    df['turn'] = df['turn'].astype('category')
    
    # call pca with given plot and test flags
    pca_analysis(df, args.plot)

    
if __name__ == '__main__':
    main()