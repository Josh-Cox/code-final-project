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
PLOT_PREFIX = '../plots/'

def create_pca_columns(pca):
    """
    Create a column names list for each PCA component
    
    :param pca: PCA
    
    :return: list of PCA component numbers
    """
    num_components = pca.components_.shape[0]
    comps_arr = []
    for i in range(num_components):
        comps_arr.append(f"PC{i+1}")
        
    return comps_arr

def combine_squares(shap_scaled, model):
    """
    Sums all of the square SHAP values (easier to visualise the impact of the board position as a whole)
    
    :param shap_scaled: the scaled SHAP values
    
    :return: SHAP values with squares summed to one column
    """
    
    if model == 'start':
        num = 7
    elif model == 'end':
        num = 8
    
    # create array for summe shap values
    all_shap = [0] * len(shap_scaled)
    
    # add all of the square values to one number to more easily visualise
    for k in range(len(shap_scaled)):
        values = shap_scaled[k]
        all_values = [0] * len(values)
        for i in range(len(values)):
            temp_arr = [0] * (num + 1)
            temp = 0
            # loop through each set of values
            for j in range(64 + num):
                # put first 7 values in new array
                if j < num:
                    temp_arr[j] = values[i][j]
                # sum last 64 squares to one values and put into new arr
                elif j == 63 + num:
                    temp += values[i][j]
                    temp_arr[num] = temp
            # add new array to main array
            all_values[i] = temp_arr
        all_shap[k] = all_values 
    
    return np.array(all_shap)

def interpret_tree(model_start, model_end, X_test_pca_start, X_test_pca_end, pca_start, pca_end, X_test_start, X_test_end, plot, plot_type, model_name):
    
    # ---------------- ORIGINAL COLUMN NAMES ---------------- #
    
    # get feature column names (excluding squares)
    column_names_start = pd.read_csv('../data/lichess-2023-11-10k.csv').drop(columns=['board_pos', 'start_square', 'end_square']).columns.tolist()
    column_names_end = pd.read_csv('../data/lichess-2023-11-10k.csv').drop(columns=['board_pos', 'end_square']).columns.tolist()
    column_names_start = column_names_start[:7]
    column_names_end = column_names_end[:8]
    
    # add column name for summed square shap values
    column_names_start.append('square_summary')
    column_names_end.append('square_summary')
    
    # ---------------- PCA COLUMN NAMES ---------------- #
    
    start_comps = create_pca_columns(pca_start)
    end_comps = create_pca_columns(pca_end)
    
    # ---------------- GET SHAP VALUES ---------------- #

    # create explainer
    explainer_start = shap.TreeExplainer(model_start)
    explainer_end = shap.TreeExplainer(model_end)

    # get shap values from explainer
    shap_pca_start = explainer_start.shap_values(X_test_pca_start)
    shap_pca_end = explainer_end.shap_values(X_test_pca_end)

    # reverse the PCA to get original features
    shap_scaled_start = pca_start.inverse_transform(shap_pca_start)
    shap_scaled_end = pca_end.inverse_transform(shap_pca_end)

    # ---------------- COMBINE SQUARE VALUES ---------------- #
    
    all_shap_start = combine_squares(shap_scaled_start, 'start')
    all_shap_end = combine_squares(shap_scaled_end, 'start')
    
    # ---------------- PLOTS ---------------- #
    
    if plot == 'summary':
        # save plots        
        save_plot(shap_scaled_start, X_test_start, X_test_start.columns.tolist(), 'model_1', 'summary', model_name)
        save_plot(shap_scaled_end, X_test_end, X_test_end.columns.tolist(), 'model_2', 'summary', model_name)

    elif plot == 'waterfall':
        # save plots
        if plot_type == 'original':    
            save_plot(all_shap_start, X_test_start, X_test_start.columns.tolist(), 'model_1', 'waterfall', model_name, explainer_start)
            save_plot(all_shap_end, X_test_end, X_test_end.columns.tolist(), 'model_2', 'waterfall', model_name, explainer_end)
        elif plot_type == 'pca':
            save_plot(shap_pca_start, X_test_start, start_comps, 'model_1_pca', 'waterfall', model_name, explainer_start)
            save_plot(shap_pca_end, X_test_end, end_comps, 'model_2_pca', 'waterfall', model_name, explainer_end)
            
    elif plot == 'force':
        # save plots
        save_plot(shap_scaled_start, X_test_start, X_test_start.columns.tolist(), 'model_1', 'force', explainer_start)
        save_plot(shap_scaled_end, X_test_end, X_test_end.columns.tolist(), 'model_2', 'force', explainer_end)
        
    # display saved plots as subplot
    display_plots(plot, plot_type, model_name)

def save_plot(shap_values, X_test, column_names, model_number, plot, model_name, explainer=None):
    """
    Creates and saves SHAP plots
    
    :param shap_values: SHAP values to use in plot
    :param X_test: X_test values to use in plot
    :param column_names: feature names to use in plot
    :param model_name: name of the model (1 or 2)
    :param plot: type of plot {'summary', 'waterfall', 'force'}
    :param explainer: if an explainer is needed for shap plot
    
    :return: saves plots to png files
    """

    # create the specified plot
    if plot == 'summary':
        shap.summary_plot(shap_values[0], X_test.values, feature_names=column_names, show=False)
    elif plot == 'waterfall':
        shap.waterfall_plot(shap.Explanation(values=shap_values[0][0], base_values=explainer.expected_value[0], data=X_test.iloc[0], feature_names=column_names), show=False)
    elif plot == 'force':
        shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.values[0], feature_names=column_names, matplotlib=True, show=False)


    # set title and save to file
    plt.title(model_number)
    plt.tight_layout()
    plt.savefig(PLOT_PREFIX + f'{model_name}_{plot}_{model_number}.png')
    plt.close()
        
def display_plots(plot, plot_type, model_name):
    """
    Displays saved plots
    
    :param plot: type of plot {'summary', 'waterfall', 'force'}
    :param plot_type: type of plot data {'original', 'pca'}
    
    :return: displays the saved plots as a subplot
    """
    # create subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # get plots file paths
    if plot_type == 'original':
        plot_1 = plt.imread(PLOT_PREFIX + f'{model_name}_{plot}_model_1.png')
        plot_2 = plt.imread(PLOT_PREFIX + f'{model_name}_{plot}_model_2.png')
    elif plot_type == 'pca':
        plot_1 = plt.imread(PLOT_PREFIX + f'{model_name}_{plot}_model_1_pca.png')
        plot_2 = plt.imread(PLOT_PREFIX + f'{model_name}_{plot}_model_2_pca.png')
    
    # get plots from files and set titles
    axes[0].imshow(plot_1)
    axes[0].axis('off')
    axes[0].set_title('Model 1')
    
    axes[1].imshow(plot_2)
    axes[1].axis('off')
    axes[1].set_title('Model 2')
    
    # show subplot
    plt.tight_layout()
    plt.show()

def pca_loadings(pca_start, pca_end, comps_1, comps_2):
    """
    Get the loadings of the original features for each PCA component
    
    :param pca_start: PCA for model 1
    :param pca_end: PCA for model 2
    :param comps_1: number of components for model 1
    :param comps_1: number of components for model 2
    
    :return: display specified plot of loadings
    """
    
    # original features to map to
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
    
    # features with square positions as sum and mean
    collapsed_features = ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating',
    'turn', 'sum_squares', 'mean_squares']
    
    # create arrays for pca loadings
    start_loadings = np.array([[0.0] * len(original_features) for _ in range(comps_1)])
    end_loadings = np.array([[0.0] * len(original_features) for _ in range(comps_1)])
    
    # get start loading for each feature of each PCA component
    for i in range(comps_1):
        for j in range(len(original_features)):
            start_loadings[i][j] = pca_start.components_[i][j]
            
    # get end loading for each feature of each PCA component
    for i in range(comps_2):
        for j in range(len(original_features)):
            end_loadings[i][j] = pca_end.components_[i][j]
    
    # new loadings array
    new_start_loadings = np.array([[0.0] * 9 for _ in range(comps_1)])
    new_end_loadings = np.array([[0.0] * 9 for _ in range(comps_2)])
    
    # sum/mean the square positions for model 1
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

    # sum/mean the square positions for model 2
    for i in range(comps_2):
        temp_sum = 0.0
        for j in range(len(original_features)):
            if j < 7:
                new_end_loadings[i][j] = end_loadings[i][j]
            elif j == 63:
                temp_sum += start_loadings[i][j]
                new_end_loadings[i][7] = temp_sum
                new_end_loadings[i][8] = temp_sum / 64
            else:
                temp_sum += end_loadings[i][j]
                

    # ---------------- PLOT VALUES ---------------- #

    # TODO: Add other diagrams
    # TODO: Add option to save to file

    # heatmap model 1
    plt.figure(figsize=(10, 6))
    sns.heatmap(new_start_loadings, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=collapsed_features, yticklabels=["PC " + str(i) for i in range(1, new_start_loadings.shape[0] + 1)])
    plt.title('PCA Loadings Heatmap For Model 1')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    plt.show()
    
    # heatmap model 2
    plt.figure(figsize=(10, 6))
    sns.heatmap(new_end_loadings, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=collapsed_features, yticklabels=["PC " + str(i) for i in range(1, new_end_loadings.shape[0] + 1)])
    plt.title('PCA Loadings Heatmap For Model 2')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    plt.show()
    
def main():
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', required=True, help='Number of components to train model 1 with (folder must exist under "PCA/")')
    parser.add_argument('--comps_2', required=True, help='Number of components to train model 2 with (folder must exist under "PCA/")')
    parser.add_argument('--plot', choices=['summary', 'waterfall', 'force'], required=True, help="SHAP plot type")
    parser.add_argument('--plot_type', choices=['original', 'pca'], required=True, help="Features to plot on the graph")
    args = parser.parse_args()
    
    # create model path
    model_path = f"{PATH_PREFIX}{args.model}/{args.model}_{args.comps_1}_{args.comps_2}/"
    
    # check if model exists
    if not os.path.isfile(f'{model_path}model_start.joblib') or not os.path.isfile(f'{model_path}model_end.joblib'):
        print("ERROR: No trained model exists. Train using python training.py")
        exit()
    
    # load models, scalers and pca
    pca_path = f"{PCA_PREFIX}{args.comps_1}_{args.comps_2}/"
    model_start = load(model_path + 'model_start.joblib')
    model_end = load(model_path + 'model_end.joblib')
    pca_start = load(pca_path + 'pca_start.joblib')
    pca_end = load(pca_path + 'pca_end.joblib')
    scaler_end = load(pca_path + 'scaler_end.joblib')
    
    if args.model == 'ebm':
        pca_loadings(pca_start, pca_end, int(args.comps_1), int(args.comps_2))
    elif args.model == 'gb' or args.model == 'dt':
        # load trained model from file
        X_val_start = pd.read_csv(pca_path + 'X_val_start.csv')
        X_val_end = pd.read_csv(pca_path + 'X_val_end.csv')
        X_val_end = scaler_end.transform(X_val_end)
        X_val_end = pca_end.transform(X_val_end)
        X_test_start = pd.read_csv(pca_path + 'X_test_start_og.csv')
        X_test_end = pd.read_csv(pca_path + 'X_test_end_og.csv')
        interpret_tree(model_start, model_end, X_val_start, X_val_end, pca_start, pca_end, X_test_start, X_test_end, args.plot, args.plot_type, args.model)


if __name__ == "__main__":
    main()