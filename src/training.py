from preprocessing import *
from joblib import dump
from tqdm import tqdm
import os

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

def train_models(df, model, encoding_method='std'):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also saves X_test to csv file.

    :param df: the dataframe to train the models on
    :param model: name of model to train {'gb', 'ebm', 'dt'}
    :param encoding_method, default='std': encoding method to use for next move
    """
    
    # create input and output features
    if encoding_method == 'std': 
        # TODO: remove 'next_move' in preprocessing
        X = df.drop(columns=['next_move_encoded'])
        y = df['next_move_encoded']
    elif encoding_method == 'vector':
        # grouping certain columns
        additional_cols = ['columns_moved', 'ranks_moved']
        start_pos_cols = [col for col in df.columns if len(col) == 1]
        
        # get columns to keep and drop
        cols_to_drop = additional_cols + ['board_pos','promote_q', 'promote_r', 'promote_n', 'promote_b'] + start_pos_cols
        cols_to_keep = start_pos_cols + additional_cols
        
        X = df.drop(columns=cols_to_drop)
        
        y = df[cols_to_keep].copy()
                
    
    # split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # separate boards from X_test
    Boards = X_test['board_pos']
    X_test = X_test.drop(columns=['board_pos'])
    X_train = X_train.drop(columns=['board_pos'])
    
    # if folder doesn't exist then create
    model_path = MODEL_PREFIX + str(model)
    if not os.path.isdir(str(model_path)):
        os.makedirs(str(model_path))
    
    # Save to csv files for future use
    X_test.to_csv(MODEL_PREFIX + str(model) + '/X_test.csv', index=False)
    y_test.to_csv(MODEL_PREFIX + str(model) + '/y_test.csv', index=False)
    y_train.to_csv(MODEL_PREFIX + str(model) + '/y_train.csv', index=False)
    X_train.to_csv(MODEL_PREFIX + str(model) + '/X_train.csv', index=False)
    Boards.to_csv(MODEL_PREFIX + str(model) + '/Boards.csv', index=False)
    
    
    # unique_classes = y_train.apply(lambda col: len(col.unique()))
    # columns_with_multiple_classes = unique_classes[unique_classes > 1].index
    
    # y_train_filtered = y_train[columns_with_multiple_classes]
    
    # model = MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
    
    # model.fit(X_train, y_train_filtered)
    
    # Encoding the data
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    
    print("\nNOW TRAINING MODEL\n")

    # create, train and save model
    if model == 'gb':
        # create model
        gb = xgb.XGBClassifier(random_state=42, enable_categorical=True)
        
        # Train model with progress bar
        gb.fit(X_train, y_train)
              
        # save model to file
        dump(gb, MODEL_PREFIX + str(model) + '/gb.joblib')
    elif model == 'ebm':
        # create model
        ebm = ExplainableBoostingClassifier(n_jobs=1)
        
        # train model (with progress bar)
        ebm.fit(X_train, y_train)
                
        # save model to file
        dump(ebm, MODEL_PREFIX + str(model) + '/ebm.joblib')
    
        
def main():
    
    # set encoding method
    encoding_method = "std"
    
    # grab df from games.csv
    df = pd.read_csv(DATA_PREFIX + 'tiny_test.csv')
    df['next_move_encoded'] = df['next_move_encoded'].astype('category')
    df['turn'] = df['turn'].astype('category')
    
    # list of features to drop
    # OPTIONS: 'w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn'
    features_to_drop = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']
    
    # train the models
    train_models(df, 'gb')
    
if __name__ == '__main__':
    main()