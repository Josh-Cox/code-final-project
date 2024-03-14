import os
import time
import argparse
import pandas as pd
import numpy as np

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

set_visualize_provider(InlineProvider())


DATA_PREFIX = '../data/'
MODEL_PREFIX = '../feature_importance/'
RESULTS_PREFIX = '../results/'

ALL_FEATURES = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']

def train_model(df, model, features_to_use, encoding_method='std'):
    """
    Trains the model with the given dataframe and saves it to a .joblib file. Also saves train_test_split data

    :param df: the dataframe to train the models on
    :param model: name of the model (used for files) ['gb', 'dt', 'ebm']
    :param features_to_drop: features to drop from the df
    :param encoding_method, default 'std': encoding method for next_move ['std', 'vector', 'binary']
    :param corr_feat: feature to test correlation of e.g. 'turn' (Must NOT be in features_to_drop)
    """
    
    # include board pos
    features_to_use.append('board_pos')
    
    features_to_drop = [x for x in ALL_FEATURES if x not in features_to_use]    
            
    # Get features
    X = df.drop(columns=features_to_drop)
    y = df['next_move_encoded']

    # split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # separate boards from X_test
    Boards = X_test['board_pos']
    X_test = X_test.drop(columns=['board_pos'])
    X_train = X_train.drop(columns=['board_pos'])
    
    # get suffix for file names
    name = ""
    for feature in ALL_FEATURES:
        if feature in features_to_use:
            name += '-' + str(feature)
        
    # if folder doesn't exist then create
    data_path = MODEL_PREFIX + str(model) + '/data'
    if not os.path.isdir(str(data_path)):
        os.makedirs(str(data_path))
        
    # if folder doesn't exist then create
    model_path = MODEL_PREFIX + str(model) + '/model'
    if not os.path.isdir(str(model_path)):
        os.makedirs(str(model_path))
        
    # if folder doesn't exist then create
    results_path = MODEL_PREFIX + str(model) + '/results'
    if not os.path.isdir(str(results_path)):
        os.makedirs(str(results_path))
    
    # Save to csv files for future use
    X_test.to_csv(data_path + '/X_test' + name + '.csv', index=False)
    y_test.to_csv(data_path + '/y_test' + name + '.csv', index=False)
    y_train.to_csv(data_path + '/y_train' + name + '.csv', index=False)
    X_train.to_csv(data_path + '/X_train' + name + '.csv', index=False)
    Boards.to_csv(data_path + '/Boards' + name + '.csv', index=False)
    
    # Encoding the data
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    print("\n--- TRAINING MODEL ---\n")
    
    seed = 42
    np.random.seed(seed)
    
    start_time = time.time()
    
    if model == 'gb':
        # create model
        gb = xgb.XGBClassifier(random_state=seed, enable_categorical=True)
        
        # fit model
        gb.fit(X_train, y_train)
        
        # save model to file
        dump(gb, model_path + '/' + name + '.joblib')
    elif model == 'ebm':
        # create model
        ebm = ExplainableBoostingClassifier(random_state=seed, interactions=0)
        
        # fit model
        ebm.fit(X_train, y_train)
        
        # save model to file
        dump(ebm, model_path + '/' + name + '.joblib')
        
    end_time = time.time()
    
    print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time-start_time} ---\n')
    print("\n--- TESTING MODEL ---\n")
    
    # test the model
    test_model(model_path, data_path, results_path, name, features_to_use)

def test_model(model_path, data_path, results_path, name, corr_feat):
    """
    Makes predictions using trained models and test data

    :param model_path: path to model folder
    :param data_path: path to data folder
    :param name: name of the file (appended features used)
    :param corr_feat: list of features to test correlation of e.g. ['w_rating', 'b_rating'] (Must NOT be in features_to_drop)
    """
    
    # get test data from train_test_split
    X_test = pd.read_csv(data_path + '/X_test' + name + '.csv')
    X_train = pd.read_csv(data_path + '/X_train' + name + '.csv')
    y_train = pd.read_csv(data_path + '/y_train' + name + '.csv')
    y_test = pd.read_csv(data_path + '/y_test' + name + '.csv')
    boards = pd.read_csv(data_path + '/Boards' + name + '.csv')
    
    # load trained model from file
    model = load(model_path + '/' + name + '.joblib')
    
    # make predictions with probabilities
    y_pred = model.predict(X_test)
    
    # Decode
    le = LabelEncoder()
    le.fit(y_train.values.ravel())
    
    y_pred = le.inverse_transform(y_pred)
    
    filtered_y_pred = []
    filtered_y_test = []
    filtered_boards = []
    
    boards = list(boards['board_pos'])
    y_test = list(y_test['next_move_encoded'])
            
    # filter all illegal predictions
    for i in range(len(y_pred)):
        if is_legal(boards[i], y_pred[i]):
            filtered_y_pred.append(y_pred[i])
            filtered_y_test.append(y_test[i])
            filtered_boards.append(boards[i])
            
    # convert to 1D array
    filtered_y_test = np.array(filtered_y_test)
    filtered_y_pred = np.array(filtered_y_pred)
    
    # evaluate model
    precision = precision_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
    recall = recall_score(filtered_y_test, filtered_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(filtered_y_test, filtered_y_pred, average='weighted')
    accuracy = accuracy_score(filtered_y_test, filtered_y_pred)
    
    # write to file
    with open(results_path + '/' + name + '.txt', 'w') as f:
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1-Score: {f1:.2f}\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        
        # check correlation values are in training data
        valid = True
        if corr_feat != None:
            for feat in corr_feat:
                if feat not in X_train:
                    valid = False
                    
        # get correlation of new feature 
        if valid:
            new_df = X_train[corr_feat].merge(y_train, left_index=True, right_index=True)
            corr_train = new_df.corr()
            f.write(f'Correlation of {corr_feat}: \n{corr_train}\n')
            
    print(f'Accuracy: {accuracy}\n')
    print(f'Precision: {precision}\n')
    print(f'Recall: {recall}\n')
    print(f'F1-Score: {f1}\n')
    
def pca_analysis(df):
    
    df = df.drop(columns=['board_pos', 'next_move_encoded'])
    board_features = df.iloc[:, -64:]
    
    additional_features = df[['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']]
    
    # standardise
    scaler_board = StandardScaler()
    board_features_scaled = scaler_board.fit_transform(board_features)
    additional_features_scaled = scaler_board.fit_transform(additional_features)
    # scaled_df = scaler_board.fit_transform(df)
    
    X_scaled = pd.DataFrame(data=np.hstack((board_features_scaled, additional_features_scaled)), columns=board_features.columns.tolist() + additional_features.columns.tolist())
    
    # apply PCA
    pca = PCA(0.95) # retain 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    prop_var = pca.explained_variance_ratio_
    
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
            
    # if folder doesn't exist then create
    pca_path = MODEL_PREFIX
    if not os.path.isdir(str(pca_path)):
        os.makedirs(str(pca_path))
    
    # write to file
    with open(pca_path + '/pca.txt', 'w') as f:
        for component_idx, component in enumerate(pca.components_):
            top_feature_indices = component.argsort()[-5:][::-1] 

            f.write(f"\nTop features for Principal Component {component_idx + 1}:\n")
            for feature_idx in top_feature_indices:
                f.write(f"Feature {feature_idx}: {df.columns[feature_idx]}\n")
        
    # plot variance
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("Scree Plot")
    plt.show()
    
    # plot PCA
    # PC_number = np.arange(pca.n_components_) + 1
    # plt.figure(figsize=(10, 6))
    # plt.plot(PC_number, prop_var, 'ro-')
    # plt.title("Scree Plot (Elbow Method)")
    # plt.xlabel("Component Number")
    # plt.ylabel("Proportion of Variance")
    # plt.grid()
    # plt.show()

def main():
    
    # --- ARGUMENT HANDLING ---
    parser = argparse.ArgumentParser(description="Which functions to run on which model")
    parser.add_argument('-f', choices=['pca', 'feat_select'], required=True)
    # parser.add_argument('-m', choices=['gb', 'ebm'], required=False)
    # parser.add_argument('-e', choices=['std', 'binary', 'vector'], default='std')
    args = parser.parse_args()

        
    # check that a model is specified if feat_select is chosen
    if args.f == 'feat_select' and args.m == None:
        print("Please specify a model with '-m'")
        exit()
        
    # check command line arguments
    if args.f == "pca":
        df = pd.read_csv(DATA_PREFIX + 'lichess-2023-11-100k.csv')
            
        # change datatypes to category where applicable
        df['next_move_encoded'] = df['next_move_encoded'].astype('category')
        df['turn'] = df['turn'].astype('category')
        
        pca_analysis(df)
    else:
        user_features = input("""
Please enter the numbers of the features you want to use (e.g. 1467)\n
w_rating:  1\n
b_rating:  2\n
w_central: 3\n
b_central: 4\n
w_safety:  5\n
b_safety:  6\n
turn:      7\n\n-> """)
        
        
        # list of features to use
        features = ['w_rating', 'b_rating', 'w_central', 'b_central', 'w_safety', 'b_safety', 'turn']
        features_to_use = []
        
        # extract user picked features
        user_features = str(user_features)
        
        # add feature if it is valid
        for num in user_features:
            if int(num) not in [1, 2, 3, 4, 5, 6, 7]:
                print("Number invalid")
                exit()
            features_to_use.append(features[int(num)-1])
                
        # set encoding method
        encoding_method = args.e
        
        # grab relative df from csv
        if args.m == "ebm":
            df = pd.read_csv(DATA_PREFIX + 'lichess-2023-11-1k.csv')
        elif args.m == "gb":
            df = pd.read_csv(DATA_PREFIX + 'lichess-2023-11-100k.csv')
            
        # change datatypes to category where applicable
        df['next_move_encoded'] = df['next_move_encoded'].astype('category')
        df['turn'] = df['turn'].astype('category')
        
        
        # train the model - remember to add corr_feat parameter if correlation evaluation wanted (see function docstring)
        train_model(df, args.m, features_to_use, encoding_method)

    # speed_test(df)

    
if __name__ == '__main__':
    main()