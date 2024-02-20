from preprocessing import *
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

DATA_PREFIX = '../data/'
MODEL_PREFIX = '../models/'

def train_models(df, encoding_method='std'):
    """
    Trains the models with the given dataframe and saves it to a .joblib file. Also saves X_test to csv file.

    :param df: the dataframe to train the models on
    """
    
    # create input and output features
    if encoding_method == 'std': 
        X = df.drop(columns=['board_pos', 'next_move_encoded'])
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
    
    unique_classes = y_train.apply(lambda col: len(col.unique()))
    columns_with_multiple_classes = unique_classes[unique_classes > 1].index
    
    y_train_filtered = y_train[columns_with_multiple_classes]
    
    model = MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
    model.fit(X_train, y_train_filtered)
    
    # create model
    # gb = GradientBoostingClassifier(n_estimators=1000, min_samples_split=2, max_features=5, random_state=42)
    
    # fit model
    # gb.fit(X_train, y_train)
    
    # save model to file
    dump(model, MODEL_PREFIX + 'gb.joblib')
    
    # save test data to csv
    X_test.to_csv(MODEL_PREFIX + 'split_test_data.csv', index=False)
    y_test.to_csv(MODEL_PREFIX + 'split_answer_data.csv', index=False)
    y_train_filtered.to_csv(MODEL_PREFIX + 'split_answer_filt_data.csv', index=False)
    
def main():
    
    encoding_method = "vector"
    # preprocess the dataframe
    preprocess_data('lichess-2023-11', 'std', encoding_method)
    
    # grab df from games.csv
    df = pd.read_csv(MODEL_PREFIX + 'games.csv')
    
    # train the models    
    train_models(df, encoding_method)
    
if __name__ == '__main__':
    main()