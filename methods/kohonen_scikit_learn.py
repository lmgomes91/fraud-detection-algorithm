import pandas as pd
from minisom import MiniSom
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from utils.dataset_handle import pre_processing_dataframe
import time
import numpy as np


def classify(som: MiniSom, data, class_assignments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assignments # noqa
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def kohonen_sklearn(credit_card_df: pd.DataFrame):
    try:
        print('RBF SKLEARN\nProcessing data')
        x_data, y_data = pre_processing_dataframe(credit_card_df)
        print(len(x_data))

        print('Splitting data')
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
        start_time = time.time()

        print('Training model')
        som = MiniSom(45, 45, sigma=3, learning_rate=0.5,
                      neighborhood_function='triangle', random_seed=10, input_len=x_data.shape[1])
        som.pca_weights_init(x_train)
        som.train_random(x_train, 5000, verbose=True)
        class_assignments = som.labels_map(x_train, y_train)

        predictions = classify(som, x_test, class_assignments)

        accuracy = accuracy_score(predictions, y_test)
        print(f'Accuracy: {accuracy:.2f}')

        print('\n\n##########Evaluating###########\n\n')
        print(f'\nTotal elapsed time: {(time.time() - start_time) / 60} minutes')
        print(f'\nConfusion matrix:\n {confusion_matrix(y_test, predictions)}')
        print(f'\nClassification Report:\n {classification_report(y_test, predictions)}')
    except Exception as e:
        print(f'Error: {e}')
