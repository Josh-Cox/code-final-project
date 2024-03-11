import shap
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

PATH_PREFIX = '../models/'

def interpret_model(model, X_test, y_train):    
    # decode
    le = LabelEncoder()
    le.fit(y_train)
    decoded_class_names = le.classes_
    
    # Use the TreeExplainer to explain XGBoost predictions
    explainer = shap.Explainer(model)

    # Get SHAP values for a specific instance (e.g., the first test instance)
    shap_values = explainer.shap_values(X_test.iloc[0])

    # Visualize the feature importance using a summary plot
    shap.summary_plot(shap_values, X_test.iloc[0])
    
    # Choose an index (e.g., 0) for a specific instance in X_test
    sample_index = 0

    # Individual prediction plot
    shap.force_plot(explainer.expected_value, shap_values[sample_index, :], X_test.iloc[sample_index, :])

    
def main():
    model_name = 'gb'
    models_path = PATH_PREFIX + model_name + '/'
    
    # load trained model from file
    model = load(models_path + model_name + '.joblib')
    X_test = pd.read_csv(models_path + 'X_test.csv')
    y_train = pd.read_csv(models_path + 'y_train.csv')
    interpret_model(model, X_test, y_train)

if __name__ == "__main__":
    main()