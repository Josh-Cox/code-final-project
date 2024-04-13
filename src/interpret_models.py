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
from prediction import decode_std
import seaborn as sns

from interpret import show
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

PATH_PREFIX = '../models/'
PCA_PREFIX = '../PCA/'
DATA_PREFIX = '../data/'

def interpret_model(model, X_test_pca, y_test, pca, scaler, X_test):
    
    # get feature column names (excluding squares)
    column_names = pd.read_csv('../data/lichess-2023-11-10k.csv').drop(columns=['board_pos', 'start_square', 'end_square']).columns.tolist()
    column_names = column_names[:7]
    
    # add column name for summed square shap values
    column_names.append('square_summary')

    # create explainer
    explainer = shap.TreeExplainer(model)

    # get shap values from explainer
    shap_values_pca = explainer.shap_values(X_test_pca)

    # reverse the PCA to get original features
    shap_values_scaled = pca.inverse_transform(shap_values_pca)
    
    # create empty list for formatted set of shap values

    # ---------------- ADD SQUARE SHAP VALUES ---------------- #
    
    all_shap = [0] * len(shap_values_scaled)
    
    # add all of the square values to one number to more easily visualise
    for k in range(len(shap_values_scaled)):
        values = scaler.inverse_transform(shap_values_scaled[k])
        all_values = [0] * len(values)
        for i in range(len(values)):
            temp_arr = [0] * 8
            temp = 0
            # loop through each set of values
            for j in range(71):
                # put first 7 values in new array
                if j < 7:
                    temp_arr[j] = values[i][j]
                # sum last 64 squares to one values and put into new array
                elif j == 70:
                    temp += values[i][j]
                    temp_arr[7] = temp
            # add new array to main array
            all_values[i] = temp_arr
        all_shap[k] = all_values
            
    all_shap = np.array(all_shap)
    # convert to numpy array so summary_plot can run .shape
    
    # get the first set of shap values
    shap_values = scaler.inverse_transform(shap_values_scaled[0])
    
    # plot the shap values
    # shap.summary_plot(shap_values_scaled[0], X_test.values, feature_names=X_test.columns.tolist())

    # shap.waterfall_plot(shap.Explanation(values=all_shap[0][0], base_values=explainer.expected_value[0], data=X_test.iloc[0], feature_names=column_names))
    # shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=X_test.iloc[0], feature_names=X_test.columns.tolist()))


    # shap.force_plot(explainer.expected_value[0], shap_values_scaled[0][0], X_test.values[0], feature_names=X_test.columns.tolist(), matplotlib=True)
    # shap.plots.waterfall(shap_values[0])

def interpret_ebm(pca_start, scaler_start, comps_1, comps_2):
    
    # calculate loadings of original features on PCA components
    
    original_features = [
    'w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating',
    'turn', 'square_0', 'square_1', 'square_2', 'square_3', 'square_4',
    'square_5', 'square_6', 'square_7', 'square_8', 'square_9', 'square_10',
    'square_11', 'square_12', 'square_13', 'square_14', 'square_15', 'square_16',
    'square_17', 'square_18', 'square_19', 'square_20', 'square_21', 'square_22',
    'square_23', 'square_24', 'square_25', 'square_26', 'square_27', 'square_28',
    'square_29', 'square_30', 'square_31', 'square_32', 'square_33', 'square_34',
    'square_35', 'square_36', 'square_37', 'square_38', 'square_39', 'square_40',
    'square_41', 'square_42', 'square_43', 'square_44', 'square_45', 'square_46',
    'square_47', 'square_48', 'square_49', 'square_50', 'square_51', 'square_52',
    'square_53', 'square_54', 'square_55', 'square_56', 'square_57', 'square_58',
    'square_59', 'square_60', 'square_61', 'square_62', 'square_63'
    ]
    
    collapsed_features = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating',
    'turn', 'sum_squares', 'mean_squares']
    
    start_loadings = np.array([[0.0] * len(original_features) for _ in range(comps_1)])
    
    # get loading for each feature of each PCA component
    for i in range(comps_1):
        for j in range(len(original_features)):
            start_loadings[i][j] = pca_start.components_[i][j]
    
    # new loadings arr
    new_start_loadings = np.array([[0.0] * 9 for _ in range(comps_1)])
    
    # sum/mean the square positions
    for i in range(comps_1):
        temp_sum = 0.0
        for j in range(len(original_features)):
            if j < 7:
                new_start_loadings[i][j] = start_loadings[i][j]
            elif j == 63:
                temp_sum += start_loadings[i][j]
                new_start_loadings[i][7] = temp_sum
                new_start_loadings[i][8] = temp_sum / 64
            else:
                temp_sum += start_loadings[i][j]
                


    # ---------------- PLOT VALUES ---------------- #

    # heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(new_start_loadings, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=collapsed_features, yticklabels=["PC " + str(i) for i in range(1, new_start_loadings.shape[0] + 1)])
    plt.title('PCA Loadings Heatmap')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    plt.show()
                
                
            
    

    
def main():
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', required=True, help='Number of components to train model 1 with (folder must exist under "PCA/")')
    parser.add_argument('--comps_2', required=True, help='Number of components to train model 2 with (folder must exist under "PCA/")')
    args = parser.parse_args()
    
    model_path = f"{PATH_PREFIX}{args.model}/{args.model}_{args.comps_1}_{args.comps_2}/"
    model = load(model_path + 'model_start.joblib')
    pca_path = f"{PCA_PREFIX}{args.comps_1}_{args.comps_2}/"
    pca_start = load(pca_path + 'pca_start.joblib')
    scaler_start = load(pca_path + 'scaler_start.joblib')
    
    if args.model == 'ebm':
        interpret_ebm(pca_start,scaler_start, int(args.comps_1), int(args.comps_2))
        exit()
        


    # check if model exists
    if not os.path.isfile(f'{model_path}model_start.joblib') or not os.path.isfile(f'{model_path}model_end.joblib'):
        print("ERROR: No trained model exists. Train using python training.py")
        exit()

    
    # load trained model from file
    model = load(model_path + 'model_start.joblib')
    X_val = pd.read_csv(pca_path + 'X_val_start.csv')
    X_test = pd.read_csv(pca_path + 'X_test_start_og.csv')
    y_test = pd.read_csv(pca_path + 'y_val_start.csv')
    interpret_model(model, X_val, y_test, pca_start, scaler_start, X_test)


if __name__ == "__main__":
    main()