import shap
import pandas as pd
from joblib import load
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
import seaborn as sns
from halo import Halo
from pca import make_dir

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
    
    :param pca: PCA object
    
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
    :param model: whether its the first or second model
    
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
                if j < 7:
                    temp_arr[j] = values[i][j]
                # sum next 64 squares to one value and put into new arr
                elif j == 70:
                    temp += values[i][j]
                    temp_arr[7] = temp
                # if start_square exists then add to array
                elif j == 71:
                    temp_arr[8] = values[i][j]
                else:
                    temp += values[i][j]
                    
            # add new array to main array
            all_values[i] = temp_arr
        all_shap[k] = all_values 
    
    return np.array(all_shap)

def interpret_tree(model_start, model_end, X_test_pca_start, X_test_pca_end, pca_start, pca_end, X_test_start, X_test_end, plot, plot_type, model_name, suffix):
    """
    Creates SHAP plots from given model and data
    
    :param model_start: first model
    :param model_end: second model
    :param X_test_pca_start: first set of PCA values
    :param X_test_pca_end: second set of PCA values
    :param pca_start: first PCA object
    :param pca_end: second PCA object
    :param X_test_start: first set of original values
    :param X_test_end: second set of original values
    :param plot: type of plot {'all', 'summary', 'waterfall', 'force', 'bar'}
    :param plot_type: type of data to plot {'pca', 'original'}
    :param model_name: name of model {'dt', 'gb'}
    :param suffix: optional suffix for filenames
    
    :return: plots and saves SHAP plots
    """

    # ---------------- ORIGINAL COLUMN NAMES ---------------- #
    
    # get feature column names (excluding squares)
    column_names_start = pd.read_csv('../data/lichess-2023-11-10k.csv').drop(columns=['board_pos', 'start_square', 'end_square']).columns.tolist()
    column_names_end = pd.read_csv('../data/lichess-2023-11-10k.csv').drop(columns=['board_pos', 'end_square']).columns.tolist()
    column_names_start = column_names_start[:7]
    column_names_end = column_names_end[:7]
    
    # add column name for summed square shap values
    column_names_start.append('square_summary')
    column_names_end.append('square_summary')
    column_names_end.append('start_square')
    
    # ---------------- ORIGINAL DATA ---------------- #
    
    X_test_combined_start = X_test_start[['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn', 'square_0']]
    X_test_combined_end = X_test_end[['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn', 'square_0', 'start_square']]
    
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
    all_shap_end = combine_squares(shap_scaled_end, 'end')
    
    
    # ---------------- PLOTS ---------------- #
    
    
    match plot:
        case 'all':
            # save all plots
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', plot='summary', model_name=model_name, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', plot='summary', model_name=model_name, suffix=suffix)
            
            save_plot(shap_values=all_shap_start, X_test=X_test_combined_start, column_names=column_names_start, model_number='model_1', plot='waterfall', model_name=model_name, explainer=explainer_start, suffix=suffix)
            save_plot(shap_values=all_shap_end, X_test=X_test_combined_end, column_names=column_names_end, model_number='model_2', plot='waterfall', model_name=model_name, explainer=explainer_end, suffix=suffix)
                
            save_plot(shap_values=shap_pca_start, X_test=X_test_start, column_names=start_comps, model_number='model_1_pca', plot='waterfall', model_name=model_name, explainer=explainer_start, suffix=suffix)
            save_plot(shap_values=shap_pca_end, X_test=X_test_end, column_names=end_comps, model_number='model_2_pca', plot='waterfall', model_name=model_name, explainer=explainer_end, suffix=suffix)
            
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', model_name=model_name, plot='force', explainer=explainer_start, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', model_name=model_name, plot='force', explainer=explainer_end, suffix=suffix)
            
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', plot='bar', model_name=model_name, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', plot='bar', model_name=model_name, suffix=suffix)
            
        case 'summary':
            # save plots 
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', plot='summary', model_name=model_name, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', plot='summary', model_name=model_name, suffix=suffix)

        case 'waterfall':
            # save plots
            if plot_type == 'original':    
                save_plot(shap_values=all_shap_start, X_test=X_test_combined_start, column_names=column_names_start, model_number='model_1', plot='waterfall', model_name=model_name, explainer=explainer_start, suffix=suffix)
                save_plot(shap_values=all_shap_end, X_test=X_test_combined_end, column_names=column_names_end, model_number='model_2', plot='waterfall', model_name=model_name, explainer=explainer_end, suffix=suffix)
            elif plot_type == 'pca':
                save_plot(shap_values=shap_pca_start, X_test=X_test_start, column_names=start_comps, model_number='model_1_pca', plot='waterfall', model_name=model_name, explainer=explainer_start, suffix=suffix)
                save_plot(shap_values=shap_pca_end, X_test=X_test_end, column_names=end_comps, model_number='model_2_pca', plot='waterfall', model_name=model_name, explainer=explainer_end, suffix=suffix)
        case 'force':
            # save plots
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', model_name=model_name, plot='force', explainer=explainer_start, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', model_name=model_name, plot='force', explainer=explainer_end, suffix=suffix)
        case 'bar':
            save_plot(shap_values=shap_scaled_start, X_test=X_test_start, column_names=X_test_start.columns.tolist(), model_number='model_1', plot='bar', model_name=model_name, suffix=suffix)
            save_plot(shap_values=shap_scaled_end, X_test=X_test_end, column_names=X_test_end.columns.tolist(), model_number='model_2', plot='bar', model_name=model_name, suffix=suffix)

    # display saved plots as subplot
    if plot != 'all':
        display_plots(plot, plot_type, model_name)
    
def save_plot(shap_values, X_test, column_names, model_number, plot, model_name, suffix, explainer=None):
    """
    Creates and saves SHAP plots
    
    :param shap_values: SHAP values to use in plot
    :param X_test: X_test values to use in plot
    :param column_names: feature names to use in plot
    :param model_nunmber: name of the model (1 or 2)
    :param plot: type of plot {'summary', 'waterfall', 'force'}
    :param model_name: name of the model ({'dt', 'gb', 'ebm'})
    :param suffix: optional suffix to add to filename when saving
    :param explainer: if an explainer is needed for shap plot
    
    :return: saves plots to png files
    """
    
    # create folder if doesn't exist
    plot_path = f'{PLOT_PREFIX}{model_name}/{plot}/'
    make_dir(plot_path)
    
    
    # create the specified plot
    match plot:
        case 'summary':
            shap.summary_plot(shap_values[0], X_test.values, feature_names=column_names, show=False)
        case 'waterfall':
            shap.waterfall_plot(shap.Explanation(values=shap_values[0][0], base_values=explainer.expected_value[0], data=X_test.iloc[0], feature_names=column_names), show=False)
        case 'force':
            shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.values[0], feature_names=column_names, matplotlib=True, show=False)
        case 'bar':
            shap.summary_plot(shap_values[0], X_test.values, feature_names=column_names, plot_type='bar', show=False)


    # set title and save to file
    plt.title(model_number)
    plt.tight_layout()
    if suffix:
        plt.savefig(plot_path + f'{model_number}_{suffix}.png')
    else:
        plt.savefig(plot_path + f'{model_number}.png')
    plt.close()
        
def display_plots(plot, plot_type, model_name, suffix):
    """
    Displays saved plots
    
    :param plot: type of plot {'summary', 'waterfall', 'force'}
    :param plot_type: type of plot data {'original', 'pca'}
    :parma model_name: name of the model {'dt', 'gb', 'ebm'}
    :param suffix: optional suffix to add to filename when saving   
    
    :return: displays the saved plots as a subplot
    """
    
    plot_path = f'{PLOT_PREFIX}{model_name}/{plot}/'
    
    # create subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # get plots file paths
    if plot_type == 'pca':
        if suffix:
            plot_1 = plt.imread(plot_path + f'model_1_pca_{suffix}.png')
            plot_2 = plt.imread(plot_path + f'model_2_pca_{suffix}.png')
        else:
            plot_1 = plt.imread(plot_path + 'model_1_pca.png')
            plot_2 = plt.imread(plot_path + 'model_2_pca.png')
    else:
        if suffix:
            plot_1 = plt.imread(plot_path + f'model_1_{suffix}.png')
            plot_2 = plt.imread(plot_path + f'model_2_{suffix}.png')
        else:
            plot_1 = plt.imread(plot_path + 'model_1.png')
            plot_2 = plt.imread(plot_path + 'model_2.png')

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

def pca_loadings(pca_start, pca_end, comps_1, comps_2, suffix):
    """
    Get the loadings of the original features for each PCA component
    
    :param pca_start: PCA for model 1
    :param pca_end: PCA for model 2
    :param comps_1: number of components for model 1
    :param comps_1: number of components for model 2
    :param suffix: optional suffix to add to filename when saving
    
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

    # create folder if doesn't exist
    plot_path = f'{PLOT_PREFIX}PCA/'
    make_dir(plot_path)
    
    # set title and save to file
    plt.title('Model 1 PCA Loadings')
    plt.tight_layout()
    if suffix:
        plt.savefig(plot_path + f'loadings_{comps_1}_{comps_2}_model1_{suffix}.png')
    else:
        plt.savefig(plot_path + f'loadings_{comps_1}_{comps_2}_model1.png')

    plt.show()
    
    # heatmap model 2
    plt.figure(figsize=(10, 6))
    sns.heatmap(new_end_loadings, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=collapsed_features, yticklabels=["PC " + str(i) for i in range(1, new_end_loadings.shape[0] + 1)])
    plt.title('PCA Loadings Heatmap For Model 2')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')

    # set title and save to file
    plt.title('Model 2 PCA Loadings')
    plt.tight_layout()
    if suffix:
        plt.savefig(plot_path + f'loadings_{comps_1}_{comps_2}_model2_{suffix}.png')
    else:
        plt.savefig(plot_path + f'loadings_{comps_1}_{comps_2}_model2.png')

    plt.show()
    
def interpret_ebm(comps_1, comps_2, num_inputs, files):
    """
    Interprets EBM model using InterpretML Explainer
    
    :param comps_1: number of components to use for model 1
    :param comps_2: number of components to use for model 2
    :param num_inputs: number of inputs to interpret (no larger than file size)
    :param files: list of files to use
    
    :return: plots explanations for EBM model
    """

    # loop through each file
    for num in files:

        # get filename
        filename = f'lichess-{num}-{num+200}-25k'

        # get data
        X_start, X_end, model_start, model_end = get_data(filename, comps_1, comps_2)

        # get explainations
        values_start = model_start.explain_local(X_start[:num_inputs])
        values_end = model_end.explain_local(X_end[:num_inputs])

        # get mean values
        mean_start = get_mean_values(values_start, num_inputs)
        mean_end = get_mean_values(values_end, num_inputs)

        # plot values
        plot_values(mean_start, mean_end, str(num))

def get_data(input_file, comps_1, comps_2):
    """
    Gets all of the needed data from files and preprares.

    :param input_file: file to make predictions with
    :param comps_1: number of components to use for model 1
    :param comps_2: number of components to use for model 2

    :return: relevent input data and models
    """
    # get models
    model_start = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/model_start.joblib')
    model_end = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/model_end.joblib')
    
    # get pca and scalers
    scaler_start = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/scaler_start.joblib')
    scaler_end = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/scaler_end.joblib')
    pca_start = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/pca_start.joblib')
    pca_end = load(f'{PATH_PREFIX}ebm/ebm_{comps_1}_{comps_2}/pca_end.joblib')
    
    # get input data
    input_data = pd.read_csv(f'{DATA_PREFIX}{input_file}.csv')
    
    X_start = input_data.drop(columns=['board_pos', 'end_square', 'start_square'])
    y_start = input_data[['start_square']]
    X_end = input_data.drop(columns=['board_pos', 'end_square'])
    y_start = input_data[['end_square']]
    
    # scale and perform pca
    X_start_scaled = scaler_start.transform(X_start)
    X_end_scaled = scaler_end.transform(X_end)
    X_start_pca = pca_start.transform(X_start_scaled)
    X_end_pca = pca_end.transform(X_end_scaled)

    return X_start_pca, X_end_pca, model_start, model_end

def get_mean_values(values, num_inputs):
    """
    Gets the mean values from all of the PCA values given by the explainer

    :param values: values given by explainer
    :param num_inputs: number of inputs to interpret (no larger than file size)

    :return: mean values
    """
    total_pca = {}
    
    for i in range(0, num_inputs):
        item = values.data(key=i)
        for key, value in enumerate(item['scores']):
            temp_total = 0
            for number in value:
                temp_total += number
            total_pca[f'PC{key+1}'] = total_pca.get(f'PC{key+1}', 0) + temp_total
            
    # loop through dividing by num_inputs (to get mean)
    for key in total_pca:
        total_pca[key] = total_pca.get(key, 0) / 10
    
    return total_pca

def plot_values(mean_start, mean_end, suffix):
    """
    Plots the mean values on a bar chart

    :param mean_start: mean values for model 1
    :param mean_end: mean values for model 2
    :param suffix: optional suffix to add to filename when saving

    :return: plot the mean values and save plot to file
    """

    # create folder if doesn't exist
    plot_path = f'{PLOT_PREFIX}ebm/'
    make_dir(plot_path)

    # get values and axis labels
    labels_start = list(mean_start.keys())
    labels_end = list(mean_end.keys())
    mean_values_start = list(mean_start.values())
    mean_values_end = list(mean_end.values())
    
    # create plot
    plt.figure(figsize=(10, 6)) 
    plt.bar(labels_start, mean_values_start)
    
    plt.xlabel('PCA Component') 
    plt.ylabel('Mean Value')  
    plt.title('Mean Impact Values for PCA Components Model 1')  
    plt.savefig(f'{plot_path}/impact_1_{suffix}.png')
    plt.close()
    
    # create plot
    plt.figure(figsize=(10, 6)) 
    plt.bar(labels_end, mean_values_end)
    
    plt.xlabel('PCA Component') 
    plt.ylabel('Mean Value')  
    plt.title('Mean Impact Values for PCA Components Model 2')  
    plt.savefig(f'{plot_path}/impact_2_{suffix}.png')
    plt.close()

def main(args):
    
    # ---------------- ARGUMENT HANDLING ---------------- #
    
    if args.model == 'dt' or args.model == 'gb':
        if not args.input:
            print("'--input' must be specified for GB or DT.")
            exit()
        elif not args.plot:
            print("'--plot' must be specified for GB or DT.")
            exit()
        elif args.plot == 'waterfall' and args.plot_type is None:
            print("'--plot_type' must be specified for a waterfall plot {'original', 'pca'}.")
            exit()
    
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
    scaler_start = load(pca_path + 'scaler_start.joblib')
    scaler_end = load(pca_path + 'scaler_end.joblib')
            
    if args.model == 'dt' or args.model == 'gb':
        # check if given filename exists
        if not os.path.isfile(f'{DATA_PREFIX}/{args.input}.csv'):
            print("ERROR: No CSV file exists. Use preprocessing.py to create one.")
            exit()
            
        # get input data
        input_data = pd.read_csv(f'{DATA_PREFIX}/{args.input}.csv')
        # get input data and board
        X_start = input_data.drop(columns=['board_pos', 'end_square', 'start_square'])
        X_end = input_data.drop(columns=['board_pos', 'end_square'])
    
        # scale and perform pca
        X_start_scaled = scaler_start.transform(X_start)
        X_end_scaled = scaler_end.transform(X_end)
        X_start_pca = pca_start.transform(X_start_scaled)
        X_end_pca = pca_end.transform(X_end_scaled)


    if args.model == 'ebm':
        # define the number of inputs to interpret (must be no larger than amount of inputs in given files)
        num_inputs = 25000

        # define different files to use (expected format of file 'lichess-{num}-{num + 200}-{num_inputs}k)
        # Example file names: [lichess-1100-1300-50k, lichess-1500-1700-50k] --> 'files' list should be [1100, 1500]

        files = [900, 1100, 1300, 1500, 1700, 1900]

        with Halo(text=f'Interpreting predictions', color='grey', spinner="dots3"):
            interpret_ebm(args.comps_1, args.comps_2, num_inputs, files)

    elif args.model == 'gb' or args.model == 'dt':
        with Halo(text=f'Interpreting predictions', color='grey', spinner="dots3"):
            interpret_tree(model_start, model_end, X_start_pca, X_end_pca, pca_start, pca_end, X_start, X_end, args.plot, args.plot_type, args.model, args.suffix)
    elif args.model == 'pca':
        with Halo(text=f'Interpreting PCA loadings', color='grey', spinner="dots3"):
            pca_loadings(pca_start, pca_end, int(args.comps_1), int(args.comps_2), args.suffix)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('--model', choices=['dt', 'gb', 'ebm'], required=True, help='Model to train')
    parser.add_argument('--comps_1', required=True, help='Number of components to train model 1 with (folder must exist under "PCA/")')
    parser.add_argument('--comps_2', required=True, help='Number of components to train model 2 with (folder must exist under "PCA/")')
    parser.add_argument('--input', type=str, required=False, help='Input file for prediction')
    parser.add_argument('--plot', choices=['all', 'summary', 'waterfall', 'force', 'bar'], required=False, help="SHAP plot type")
    parser.add_argument('--plot_type', choices=['original', 'pca'], required=False, help="Features to plot on the graph")
    parser.add_argument('--suffix', type=str, required=False, help="Optional suffix to add to filenames")
    args = parser.parse_args()
    main(args)