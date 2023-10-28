import pandas
import time
from sklearn.model_selection import train_test_split
from utils.dataset_handle import pre_processing_dataframe
from tensorflow import keras
import tensorflow as tf


def lstm_tensorflow(credit_card_df: pandas.DataFrame):
    print('LSTM TENSORFLOW\nProcessing data')
    x_data, y_data = pre_processing_dataframe(credit_card_df)

    print('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    start_time = time.time()

    print(x_train.shape)
    model = keras.Sequential([
        keras.layers.Input(shape=x_train.shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # noqa
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_train, y_train))

    print(f'\nElapsed time in training: {(time.time() - start_time) / 60} minutes')

    predictions = model.predict(x_test)

    print('\n\n##########Evaluating###########\n\n')
    print(f'\nTotal elapsed time: {(time.time() - start_time) / 60} minutes')
    print(f'\nHistory: {history.history}')
    print(f"\nconfusion_matrix: {tf.math.confusion_matrix(y_test, predictions, num_classes=2)}")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)
