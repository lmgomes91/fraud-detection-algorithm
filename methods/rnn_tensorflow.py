import time
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils.dataset_handle import pre_processing_dataframe


def rnn_tensorflow(credit_card_df: pandas.DataFrame):
    try:
        print('RNN TENSORFLOW\nProcessing data')
        x_data, y_data = pre_processing_dataframe(credit_card_df)
        print(len(x_data))

        print('Splitting data')
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

        start_time = time.time()
        print('Training model')
        model = keras.Sequential([
            keras.layers.Dense(units=64, activation='relu',    name='input_layer', input_shape=(30,),),
            keras.layers.Dense(units=32, activation='relu',    name='hidden_layer_1'),
            keras.layers.Dense(units=32, activation='relu',    name='hidden_layer_2'),
            keras.layers.Dense(units=16, activation='relu',    name='hidden_layer_3'),
            keras.layers.Dense(units=1,  activation='sigmoid', name='output_layer')
            # Single output unit for binary classification
        ])

        model.compile(
            optimizer='SGD',
            loss='binary_crossentropy',  # noqa Use binary_crossentropy for binary classification
            metrics=['accuracy']
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=1000,
            validation_data=(x_train, y_train),
            verbose=False
        )

        print(f'\nElapsed time in training: {(time.time() - start_time)/60} minutes')

        predictions = model.predict(x_test)
        print('\n\n##########Evaluating###########\n\n')
        print(f'\nTotal elapsed time: {(time.time() - start_time)/60} minutes')
        print(f'\nHistory: {history.history}')
        print(f"\nconfusion_matrix: {tf.math.confusion_matrix(y_test, predictions, num_classes=2)}")
    except Exception as e:
        raise RuntimeError(f'Error on tensorflow mlp: {e}') from e
