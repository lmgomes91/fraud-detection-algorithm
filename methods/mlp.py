import pandas

from utils.dataset_handle import pre_processing_dataframe
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # for classification
from sklearn.metrics import confusion_matrix, classification_report


def mlp(credit_card_df: pandas.DataFrame):

    print('Processing data')
    x_data, y_data = pre_processing_dataframe(credit_card_df)

    print('Splitting data')
    x_train, x_test = train_test_split(x_data, test_size=0.3, random_state=42)
    y_train, y_test = train_test_split(y_data, test_size=0.3, random_state=42)

    print('Training model')
    # Create an MLP Classifier model
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    # Fit the model to the training data
    mlp_classifier.fit(x_train, y_train)
    # Prediction
    y_pred = mlp_classifier.predict(x_test)

    print('\n\n##########Evaluating###########\n\n')
    print(f'Accuracy: {mlp_classifier.score(x_test, y_test)}')
    print(f'\nConfusion matrix:\n {confusion_matrix(y_test, y_pred)}')
    print(f'\nClassification Report:\n {classification_report(y_test, y_pred)}')
