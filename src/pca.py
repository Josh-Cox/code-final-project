# ---------------- IMPORTS ---------------- #

import os
import time
import argparse
import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
from joblib import dump, load
import xgboost as xgb
from prediction import is_legal

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

# ---------------- GLOBALS ---------------- #

set_visualize_provider(InlineProvider())


DATA_PREFIX = '../data/'
FEATURE_PREFIX = '../feature_importance/'
RESULTS_PREFIX = '../results/'
PCA_PREFIX = '../PCA/'


ALL_FEATURES = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']

# def train_model(df, model, features_to_use, encoding_method='std'):
#     """
#     Trains the model with the given dataframe and saves it to a .joblib file. Also saves train_test_split data

#     :param df: the dataframe to train the models on
#     :param model: name of the model (used for files) ['gb', 'dt', 'ebm']
#     :param features_to_use: features from the df to use
#     :param encoding_method, default 'std': encoding method for next_move ['std', 'vector', 'binary']
#     """
    
#     # ---------------- DEFINE VARIBALES ---------------- #
    
#     # include board position
#     features_to_use.append('board_pos')
    
#     # drop all non needed features
#     features_to_drop = [x for x in ALL_FEATURES if x not in features_to_use]    
            
#     # Get features
#     X = df.drop(columns=features_to_drop)
#     y = df['next_move_encoded']

#     # split into test and train data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # separate boards from X_test
#     Boards = X_test['board_pos']
#     X_test = X_test.drop(columns=['board_pos'])
#     X_train = X_train.drop(columns=['board_pos'])
    
#     # get suffix for file names
#     name = ""
#     for feature in ALL_FEATURES:
#         if feature in features_to_use:
#             name += '-' + str(feature)
        
#     # if folder doesn't exist then create
#     data_path = FEATURE_PREFIX + str(model) + '/data'
#     if not os.path.isdir(str(data_path)):
#         os.makedirs(str(data_path))
        
#     # if folder doesn't exist then create
#     model_path = FEATURE_PREFIX + str(model) + '/model'
#     if not os.path.isdir(str(model_path)):
#         os.makedirs(str(model_path))
        
#     # if folder doesn't exist then create
#     results_path = FEATURE_PREFIX + str(model) + '/results'
#     if not os.path.isdir(str(results_path)):
#         os.makedirs(str(results_path))
    
#     # Save to csv files for future use
#     X_test.to_csv(data_path + '/X_test' + name + '.csv', index=False)
#     y_test.to_csv(data_path + '/y_test' + name + '.csv', index=False)
#     y_train.to_csv(data_path + '/y_train' + name + '.csv', index=False)
#     X_train.to_csv(data_path + '/X_train' + name + '.csv', index=False)
#     Boards.to_csv(data_path + '/Boards' + name + '.csv', index=False)
    
#     # Encoding the data
#     le = LabelEncoder()
#     y_train = le.fit_transform(y_train)
    
#     print("\n--- TRAINING MODEL ---\n")
    
#     seed = 42
#     np.random.seed(seed)
    
#     start_time = time.time()
    
#     if model == 'gb':
#         # create model
#         gb = xgb.XGBClassifier(random_state=seed, enable_categorical=True)
        
#         # fit model
#         gb.fit(X_train, y_train)
        
#         # save model to file
#         dump(gb, model_path + '/' + name + '.joblib')
#     elif model == 'ebm':
#         # create model
#         ebm = ExplainableBoostingClassifier(random_state=seed, interactions=0)
        
#         # fit model
#         ebm.fit(X_train, y_train)
        
#         # save model to file
#         dump(ebm, model_path + '/' + name + '.joblib')
        
#     end_time = time.time()
    
#     print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time-start_time} ---\n')
#     print("\n--- TESTING MODEL ---\n")
    
#     # test the model
#     test_model(model_path, data_path, results_path, name, features_to_use)

# def test_model(model_path, data_path, results_path, name, corr_feat):
#     """
#     Makes predictions using trained models and test data

#     :param model_path: path to model folder
#     :param data_path: path to data folder
#     :param name: name of the file (appended features used)
#     :param corr_feat: list of features to test correlation of e.g. ['w_rating', 'b_rating'] (Must NOT be in features_to_drop)
#     """
    
#     # get test data from train_test_split
#     X_test = pd.read_csv(data_path + '/X_test' + name + '.csv')
#     X_train = pd.read_csv(data_path + '/X_train' + name + '.csv')
#     y_train = pd.read_csv(data_path + '/y_train' + name + '.csv')
#     y_test = pd.read_csv(data_path + '/y_test' + name + '.csv')
#     boards = pd.read_csv(data_path + '/Boards' + name + '.csv')
    
#     # load trained model from file
#     model = load(model_path + '/' + name + '.joblib')
    
#     # make predictions with probabilities
#     y_pred = model.predict(X_test)
    
#     # Decode
#     le = LabelEncoder()
#     le.fit(y_train.values.ravel())
    
#     y_pred = le.inverse_transform(y_pred)
    
#     filtered_y_pred = []
#     filtered_y_test = []
#     filtered_boards = []
    
#     boards = list(boards['board_pos'])
#     y_test = list(y_test['next_move_encoded'])
            
#     # filter all illegal predictions
#     for i in range(len(y_pred)):
#         if is_legal(boards[i], y_pred[i]):
#             filtered_y_pred.append(y_pred[i])
#             filtered_y_test.append(y_test[i])
#             filtered_boards.append(boards[i])
            
#     # convert to 1D array
#     filtered_y_test = np.array(filtered_y_test)
#     filtered_y_pred = np.array(filtered_y_pred)
    
#     # evaluate model
#     precision = precision_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
#     recall = recall_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
#     f1 = f1_score(filtered_y_test, filtered_y_pred, average='weighted')
#     accuracy = accuracy_score(filtered_y_test, filtered_y_pred)
    
#     # write to file
#     with open(results_path + '/' + name + '.txt', 'w') as f:
#         f.write(f'Precision: {precision:.2f}\n')
#         f.write(f'Recall: {recall:.2f}\n')
#         f.write(f'F1-Score: {f1:.2f}\n')
#         f.write(f'Accuracy: {accuracy:.2f}\n')
        
#         # check correlation values are in training data
#         valid = True
#         if corr_feat != None:
#             for feat in corr_feat:
#                 if feat not in X_train:
#                     valid = False
                    
#         # get correlation of new feature 
#         if valid:
#             new_df = X_train[corr_feat].merge(y_train, left_index=True, right_index=True)
#             corr_train = new_df.corr()
#             f.write(f'Correlation of {corr_feat}: \n{corr_train}\n')
            
#     print(f'Accuracy: {accuracy}\n')
#     print(f'Precision: {precision}\n')
#     print(f'Recall: {recall}\n')
#     print(f'F1-Score: {f1}\n')
    
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
    :param test: whether to create a test set
    
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
    
    :param pca: the pca to plot
    :param plot_type: the type of plot ('bar' or 'elbow')
    :param per_var: TODO
    :param prop_var: TODO
    :labels: labels for the graph
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
             
def pca_analysis(df, plot_type, test=False):
    """
    Perform scaling and PCA on data
    
    :param df: dataframe to use
    :plot_type: the type of plot ('bar' or 'elbow')
    :param test: whether to use a test set
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
    #TODO
    per_var_start = np.round(pca_start_vis.explained_variance_ratio_ * 100, decimals=1)
    per_var_end = np.round(pca_end_vis.explained_variance_ratio_ * 100, decimals=1)
    
    prop_var_start = pca_start_vis.explained_variance_ratio_
    prop_var_end = pca_end_vis.explained_variance_ratio_
    
    # labels for plot
    labels_start = ['PC' + str(x) for x in range(1, len(per_var_start)+1)]
    labels_end = ['PC' + str(x) for x in range(1, len(per_var_end)+1)]
    
    # plot the pca
    if plot_type:
        plot_pca(pca_start_vis, pca_end_vis, plot_type, per_var_start, per_var_end, prop_var_start, prop_var_end, labels_start, labels_end)
    
    
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
    # parser.add_argument('-e', choices=['std', 'binary', 'vector'], default='std')
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