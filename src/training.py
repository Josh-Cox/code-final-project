from preprocessing import *
from joblib import dump
from tqdm import tqdm
import os
import time
import argparse

from progress.spinner import MoonSpinner

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

set_visualize_provider(InlineProvider())

DATA_PREFIX = '../data/'
MODEL_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'

def train_models(model, folder, encoding_method='std'):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also saves X_test to csv file.

    :param df: the dataframe to train the models on
    :param model: name of model to train {'gb', 'ebm', 'dt'}
    :param encoding_method, default='std': encoding method to use for next move
    """
    
    # create input and output features
    # if encoding_method == 'std': 
    #     X = df.drop(columns=['next_move_encoded'])
    #     y = df['next_move_encoded']
    # elif encoding_method == 'vector':
    #     # grouping certain columns
    #     additional_cols = ['columns_moved', 'ranks_moved']
    #     start_pos_cols = [col for col in df.columns if len(col) == 1]
        
    #     # get columns to keep and drop
    #     cols_to_drop = additional_cols + ['board_pos','promote_q', 'promote_r', 'promote_n', 'promote_b'] + start_pos_cols
    #     cols_to_keep = start_pos_cols + additional_cols
        
    #     X = df.drop(columns=cols_to_drop)
        
    #     y = df[cols_to_keep].copy()
    
    # check if given folder exists
    if not os.path.isdir(str(PCA_PREFIX + folder)):
        print("ERROR: Folder not found")
        exit()
                
    # get test data from train_test_split
    X_test = pd.read_csv(PCA_PREFIX + f'{folder}/X_pca_test.csv')
    X_train = pd.read_csv(PCA_PREFIX + f'{folder}/X_pca_train.csv')
    y_train = pd.read_csv(PCA_PREFIX + f'{folder}/y_train.csv')
    y_test = pd.read_csv(PCA_PREFIX + f'{folder}/y_test.csv')
    boards = pd.read_csv(PCA_PREFIX + f'{folder}/Boards.csv')
    
    # if folder doesn't exist then create
    model_path = MODEL_PREFIX + str(model) + '/model/'
    if not os.path.isdir(str(model_path)):
        os.makedirs(str(model_path))
    
    # unique_classes = y_train.apply(lambda col: len(col.unique()))
    # columns_with_multiple_classes = unique_classes[unique_classes > 1].index
    
    # y_train_filtered = y_train[columns_with_multiple_classes]
    
    # model = MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
    
    # model.fit(X_train, y_train_filtered)
    
    # Encoding the data
    le = LabelEncoder()
    le.fit(y_train)
    np.save('classes.npy', le.classes_)
    y_train = le.transform(y_train)
    
    # start time of training
    start_time = time.time()

    print(f'\n--- TRAINING {model.upper()} ---')
    
    # set random seed for models
    seed = 42
    np.random.seed(seed)
    
    # create, train and save model
    if model == 'gb':
        # create model
        gb = xgb.XGBClassifier(random_state=seed, enable_categorical=True)
        
        # Train model with progress bar
        gb.fit(X_train, y_train)
            
        # save model to file
        dump(gb, model_path + f'gb_{folder}.joblib')
    elif model == 'ebm':
        # create model
        ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1, interactions=0)
        
        # train model
        ebm.fit(X_train, y_train)
                
        # save model to file
        dump(ebm, model_path + f'ebm_{folder}.joblib')
    
    # end time for training model
    end_time = time.time()
    
    print(f'\n--- FINISHED TRAINING ---\n\n--- TIME ELAPSED: {end_time - start_time} ---\n')
    
def main():
    
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('-m', choices=['gb', 'ebm'], required=True, help='Model Selection')
    parser.add_argument('-n', required=True, help='Number of components to train with (E.g. "10", "20") (Folder must exist under "PCA/")')
    args = parser.parse_args()
    
    # set encoding method
    encoding_method = "std"
    
    # # grab df from csv
    # df = pd.read_csv(file_path)
    # df['next_move_encoded'] = df['next_move_encoded'].astype('category')
    # df['turn'] = df['turn'].astype('category')
    
    # TODO: Remove this in preprocessing
    # df = df.drop(columns=['next_move'])
    
    # train the models
    train_models(args.m, args.n)
    
if __name__ == '__main__':
    main()