from training import *
from prediction import *
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

DATA_PREFIX = '../data/'
MODEL_PREFIX = '../feature_importance/'
RESULTS_PREFIX = '../results/'

ALL_FEATURES = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn']

def train_model(df, model, features_to_drop, corr_feat=None, encoding_method='std'):
    """
    Trains the model with the given dataframe and saves it to a .joblib file. Also saves train_test_split data

    :param df: the dataframe to train the models on
    :param model: name of the model (used for files) ['gb', 'dt', 'ebm']
    :param features_to_drop: features to drop from the df
    :param encoding_method, default 'std': encoding method for next_move ['std', 'vector', 'binary']
    :param corr_feat: feature to test correlation of e.g. 'turn' (Must NOT be in features_to_drop)
    """
            
    # TODO: remove 'next_move' in preprocessing
    X = df.drop(columns=['next_move_encoded', 'next_move'])
    y = df['next_move_encoded']

    # remove any unwanted columns
    for column in X.columns:
        if column in features_to_drop:
            X = X.drop(columns=[column])

    # split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # separate boards from X_test
    Boards = X_test['board_pos']
    X_test = X_test.drop(columns=['board_pos'])
    X_train = X_train.drop(columns=['board_pos'])
    
    # get suffix for file names
    name = ""
    for feature in ALL_FEATURES:
        if feature not in features_to_drop:
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
    
    # create model
    gb = xgb.XGBClassifier(random_state=42, enable_categorical=True)

    print("\nTRAINING MODEL\n")
    
    # fit model
    gb.fit(X_train, y_train)
    
    # save model to file
    dump(gb, model_path + '/' + name + '.joblib')
    
    print("\nTESTING MODEL\n")

    # test the model
    test_model(model_path, data_path, results_path, name, corr_feat)

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
            f.write(f'Correlation of {corr_feat}: {corr_train}\n')


def main():  
    # set encoding method
    encoding_method = "std"
    
    # grab df from games.csv
    df = pd.read_csv(DATA_PREFIX + 'lichess-2023-11-100k.csv')
    # change datatypes to category where applicable
    df['next_move_encoded'] = df['next_move_encoded'].astype('category')
    df['turn'] = df['turn'].astype('category')
    
    # list of features to drop
    # OPTIONS: 'w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn'
    features_to_drop = ['w_safety', 'b_safety', 'w_rating', 'b_rating', 'turn']
    
    # train the model - remember to add corr_feat parameter if correlation evaluation wanted (see function docstring)
    train_model(df, 'gb', features_to_drop, ['w_central', 'b_central'])

    
if __name__ == '__main__':
    main()