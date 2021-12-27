'''
This module contains the functions necessary to create a model for churn
prediction. That includes EDA, data engineering, training and testing of
the models and visualization of the results.

Author: Rodrigo da Matta Bastos.
Date: 23/12/2021.
'''

import joblib

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from constants import (DATABASE_FILE, IMAGES_EDA_PATH, IMAGES_RESULTS_PATH,
                       MODELS_PATH, CATEGORICAL_COLUMNS, KEEP_COLUMNS,
                       RANDOM_STATE, PARAM_GRID, CATEGORICAL_COLUMNS_EDA,
                       NUMERICAL_COLUMNS_EDA)

sns.set()


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    Input:
            pth: A path to the csv.
    Output:
            df: Pandas dataframe.
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df, categorical_columns, numerical_columns):
    '''
    Perform eda on df and save figures to images folder.
    Input:
            df: Pandas dataframe.
            categorical_columns: Categorical columns that will be used
            to generate histograms on the EDA.
            numerical_columns: Numerical columns that will be used to
            generate distribution plots on the EDA.

    Output:
            None.
    '''

    for column in categorical_columns:
        plt.figure(figsize=(20, 10))
        df[column].hist()
        plt.savefig(
            IMAGES_EDA_PATH +
            'Cat_' +
            column +
            '.png',
            bbox_inches='tight')

    for column in numerical_columns:
        plt.figure(figsize=(20, 10))
        sns.distplot(df[column])
        plt.savefig(
            IMAGES_EDA_PATH +
            'Num_' +
            column +
            '.png',
            bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(IMAGES_EDA_PATH + 'Correlation.png', bbox_inches='tight')


def encoder_helper(df, categorical_columns, response):
    '''
    Helper function to turn each categorical column into a new column
    with propotion of churn for each category.

    Input:
            df: Pandas dataframe.
            categorical_columns: List of columns that contain
            categorical features.
            response: String of response name [optional argument that
            could be used for naming variables or index y column].

    Output:
            new_df: Pandas dataframe with columns encoded.
    '''
    new_df = df.copy()
    for column in categorical_columns:
        encoded_values = new_df.groupby(column).mean()[response].to_dict()
        new_df[column + '_' + response] = new_df[column].map(encoded_values)
    return new_df


def perform_feature_engineering(df, response="Churn"):
    '''
    Perform feature engineering on the dataframe and splits dataset into
    train and test.
    Input:
              df: Pandas dataframe.
              response: String of response name [optional argument that
              could be used for naming variables or index y column].

    Output:
              X_train: X training data.
              X_test: X testing data.
              y_train: y training data.
              y_test: y testing data.
    '''

    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df = encoder_helper(df, CATEGORICAL_COLUMNS, response)

    y = df[response]

    X = pd.DataFrame()
    X[KEEP_COLUMNS] = df[KEEP_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    '''
    Produces classification report for training and testing results and
    stores report as image in images folder.
    Input:
            y_train: training response values.
            y_test:  test response values.
            y_train_preds_lr: training predictions from logistic
            regression.
            y_train_preds_rf: training predictions from random forest.
            y_test_preds_lr: test predictions from logistic regression.
            y_test_preds_rf: test predictions from random forest.

    Output:
             None.
    '''

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        IMAGES_RESULTS_PATH +
        'Random_Forest_Report.png',
        bbox_inches='tight')

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(IMAGES_RESULTS_PATH + 'Logistic_Regression_Report.png',
                bbox_inches='tight')


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances in pth. Both the model
    feature importances and the shap importance plot are created.
    Input:
            model: Model object containing feature_importances_.
            X_data: Pandas dataframe of X values.
            output_pth: Path of the folder to store the figures.

    Output:
             None.
    '''

    plt.figure(figsize=(8, 9))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth + 'Shap_Explainer.png', bbox_inches='tight')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    importance_indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature
    # importances
    feature_names = [X_data.columns[i] for i in importance_indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[importance_indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), feature_names, rotation=90)
    plt.savefig(output_pth + 'Feature_Importances.png', bbox_inches='tight')


def roc_auc_plot(
        random_forest_model,
        logistic_regression_model,
        X_data,
        y_data):
    '''
    Creates and stores the roc auc plot for both logistic regression and
    random forest models.
    Input:
            random_forest_model: Trained random forest model.
            logistic_regression_model: Trained logistic regression
            model.
            X_data: Pandas dataframe of X values.
            y_data: Pandas dataframe of y values.

    Output:
             None.
    '''

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        random_forest_model, X_data, y_data, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(
        logistic_regression_model, X_data, y_data, ax=ax, alpha=0.8)
    plt.savefig(IMAGES_RESULTS_PATH + 'Roc_Curve.png', bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models.
    Input:
              X_train: X training data.
              X_test: X testing data.
              y_train: y training data.
              y_test: y testing data.
    Output:
              None.
    '''

    random_forest_model = RandomForestClassifier(random_state=RANDOM_STATE)
    logistic_regression_model = LogisticRegression()

    grid_search_cv = GridSearchCV(estimator=random_forest_model,
                                  param_grid=PARAM_GRID, cv=5)
    grid_search_cv.fit(X_train, y_train)

    logistic_regression_model.fit(X_train, y_train)

    joblib.dump(grid_search_cv.best_estimator_, MODELS_PATH + 'rfc_model.pkl')
    joblib.dump(logistic_regression_model, MODELS_PATH + 'logistic_model.pkl')

    y_train_preds_rf = grid_search_cv.best_estimator_.predict(X_train)
    y_test_preds_rf = grid_search_cv.best_estimator_.predict(X_test)

    y_train_preds_lr = logistic_regression_model.predict(X_train)
    y_test_preds_lr = logistic_regression_model.predict(X_test)

    roc_auc_plot(grid_search_cv.best_estimator_, logistic_regression_model,
                 X_test, y_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(grid_search_cv.best_estimator_, X_test,
                            IMAGES_RESULTS_PATH)


if __name__ == '__main__':

    DF = import_data(DATABASE_FILE)
    perform_eda(DF, CATEGORICAL_COLUMNS_EDA, NUMERICAL_COLUMNS_EDA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DF)
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
