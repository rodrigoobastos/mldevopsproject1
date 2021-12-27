'''
This file is used to test the functions of churn_library module.

Author: Rodrigo da Matta Bastos.
Date: 23/12/2021.
'''

import os
import logging

import numpy as np

from constants import (DATABASE_FILE, IMAGES_EDA_PATH, IMAGES_RESULTS_PATH,
                       MODELS_PATH, KEEP_COLUMNS, CATEGORICAL_COLUMNS_TEST,
                       NUMERICAL_COLUMNS_TEST, CATEGORY_LIST_TEST)
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    Test data import.
    '''
    try:
        df = import_data(DATABASE_FILE)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and \
            columns")
        raise err


def test_eda(perform_eda):
    '''
    Test perform eda function.
    '''
    df = cls.import_data(DATABASE_FILE)
    try:
        perform_eda(df, CATEGORICAL_COLUMNS_TEST, NUMERICAL_COLUMNS_TEST)
        logging.info("Testing perform_eda: SUCCESS")
        assert os.path.isfile(IMAGES_EDA_PATH + 'Cat_Marital_Status.png')
        os.remove(IMAGES_EDA_PATH + 'Cat_Marital_Status.png')
        assert os.path.isfile(IMAGES_EDA_PATH + 'Num_Total_Trans_Ct.png')
        os.remove(IMAGES_EDA_PATH + 'Num_Total_Trans_Ct.png')
        assert os.path.isfile(IMAGES_EDA_PATH + 'Correlation.png')
        os.remove(IMAGES_EDA_PATH + 'Correlation.png')
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The function doesn't appear to have created \
             the EDA files")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper.
    '''
    df = cls.import_data(DATABASE_FILE)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    try:
        new_df = encoder_helper(df, CATEGORY_LIST_TEST, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")

        assert all(new_df['Gender'].value_counts().values ==
                   new_df['Gender_Churn'].value_counts().values)
        assert all(new_df['Education_Level'].value_counts(
        ).values == new_df['Education_Level'].value_counts().values)
        assert all(new_df['Marital_Status'].value_counts(
        ).values == new_df['Marital_Status'].value_counts().values)
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The mean enconded columns values doesn't \
             have the same distribution of the categories")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering.
    '''

    df = cls.import_data(DATABASE_FILE)
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")

        assert all(X_test.columns.values == np.array(KEEP_COLUMNS))
        assert all(X_train.columns.values == np.array(KEEP_COLUMNS))
        assert y_test.shape[0] == X_test.shape[0]
        assert y_train.shape[0] == X_train.shape[0]
        assert X_train.shape[0] > X_test.shape[0]
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: There are missing columns \
            in X or one of the variables has the wrong number of rows")
        raise err


def test_train_models(train_models):
    '''
    Test train_models.
    '''

    df = cls.import_data(DATABASE_FILE)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")

        assert os.path.isfile(IMAGES_RESULTS_PATH + 'Roc_Curve.png')
        os.remove(IMAGES_RESULTS_PATH + 'Roc_Curve.png')
        assert os.path.isfile(IMAGES_RESULTS_PATH + 'Shap_Explainer.png')
        os.remove(IMAGES_RESULTS_PATH + 'Shap_Explainer.png')
        assert os.path.isfile(IMAGES_RESULTS_PATH + 'Feature_Importances.png')
        os.remove(IMAGES_RESULTS_PATH + 'Feature_Importances.png')
        assert os.path.isfile(IMAGES_RESULTS_PATH + 'Random_Forest_Report.png')
        os.remove(IMAGES_RESULTS_PATH + 'Random_Forest_Report.png')
        assert os.path.isfile(
            IMAGES_RESULTS_PATH +
            'Logistic_Regression_Report.png')
        os.remove(IMAGES_RESULTS_PATH + 'Logistic_Regression_Report.png')

    except AssertionError as err:
        logging.error("Testing train_models: There are missing image files")
        raise err

    try:
        assert os.path.isfile(MODELS_PATH + 'rfc_model.pkl')
        os.remove(MODELS_PATH + 'rfc_model.pkl')
        assert os.path.isfile(MODELS_PATH + 'logistic_model.pkl')
        os.remove(MODELS_PATH + 'logistic_model.pkl')

    except AssertionError as err:
        logging.error(
            "Testing train_models: The models pickle file are missing")
        raise err


if __name__ == '__main__':
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
