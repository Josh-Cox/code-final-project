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

def main():
    ['w_safety', 'b_safety', 'w_central', 'b_central', 'w_rating', 'b_rating', 'turn', 'square_0', 
     'square_1', 'square_2', 'square_3', 'square_4', 'square_5', 'square_6', 'square_7', 'square_8', 
     'square_9', 'square_10', 'square_11', 'square_12', 'square_13', 'square_14', 'square_15', 'square_16', 
     'square_17', 'square_18', 'square_19', 'square_20', 'square_21', 'square_22', 'square_23', 'square_24', 
     'square_25', 'square_26', 'square_27', 'square_28', 'square_29', 'square_30', 'square_31', 'square_32',
     'square_33', 'square_34', 'square_35', 'square_36', 'square_37', 'square_38', 'square_39', 'square_40', 
     'square_41', 'square_42', 'square_43', 'square_44', 'square_45', 'square_46', 'square_47', 'square_48', 
     'square_49', 'square_50', 'square_51', 'square_52', 'square_53', 'square_54', 'square_55', 'square_56', 
     'square_57', 'square_58', 'square_59', 'square_60', 'square_61', 'square_62', 'square_63']
    df = pd.read_csv('../data/lichess-2023-11-50k.csv').drop(columns=['board_pos', 'start_square', 'end_square'])
    # print(df.shape)
    # exit()
    # d = {'col1': [1, 2, 3], 'col2': [1, 6, 2]}
    # df = pd.DataFrame(data=d)
    scaler = StandardScaler()
    pca = PCA(n_components=5)
    df_scaled = scaler.fit_transform(df)
    df_pca = pca.fit_transform(df_scaled)
    dump(pca, '../data/pca.joblib')
    dump(scaler, '../data/scaler.joblib')
    print(df_pca)
    return df_pca
    
def test(df, pca, scaler):
    print(df.shape)
    columns = pd.read_csv('../data/lichess-2023-11-50k.csv').drop(columns=['board_pos', 'start_square', 'end_square']).columns.tolist()
    # Inverse transform PCA data
    df_approx_scaled = pca.inverse_transform(df)

    # Inverse scale the data
    df_approx = scaler.inverse_transform(df_approx_scaled)

    # Creating DataFrame from the approximation
    df_approx_df = pd.DataFrame(df_approx, columns=columns)

    print(df_approx_df)

if __name__ == '__main__':
    df = main()
    pca = load('../data/pca.joblib')
    scaler = load('../data/scaler.joblib')
    test(df, pca, scaler)