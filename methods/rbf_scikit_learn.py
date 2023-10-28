from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from utils.dataset_handle import pre_processing_dataframe
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def rbf_sklearn(credit_card_df: pd.DataFrame):

    try:
        print('RBF SKLEARN\nProcessing data')
        x_data, y_data = pre_processing_dataframe(credit_card_df)
        print(len(x_data))

        print('Splitting data')
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
        start_time = time.time()
        print('Training model')

        kernel = 1.0 * RBF(1.0)
        rbf_model = GaussianProcessClassifier(kernel=kernel, copy_X_train=False, n_jobs=-1)
        rbf_model.fit(x_train, y_train)

        predictions = rbf_model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.2f}')

        print('\n\n##########Evaluating###########\n\n')
        print(f'\nTotal elapsed time: {(time.time() - start_time) / 60} minutes')
        print(f'\nConfusion matrix:\n {confusion_matrix(y_test, predictions)}')
        print(f'\nClassification Report:\n {classification_report(y_test, predictions)}')
    except Exception as e:
        print(f'Error: {e}')
