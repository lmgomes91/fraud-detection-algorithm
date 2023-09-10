import time
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils.dataset_handle import pre_processing_dataframe


def convert_dataframe_to_tensors(credit_card_df: pandas.DataFrame):
    print('Processing data')
    x_data, y_data = pre_processing_dataframe(credit_card_df)

    print('Splitting data')
    x_train, x_test = train_test_split(x_data, test_size=0.3, random_state=42)
    y_train, y_test = train_test_split(y_data, test_size=0.3, random_state=42)

    return (
        tf.convert_to_tensor(x_train),
        tf.convert_to_tensor(x_test),
        tf.convert_to_tensor(y_train),
        tf.convert_to_tensor(y_test)
    )


def mlp_tensorflow(credit_card_df: pandas.DataFrame):
    print('Converting data into tensors')
    x_train, x_test, y_train, y_test = convert_dataframe_to_tensors(credit_card_df)
    print('Converted tensors')

    start_time = time.time()
    print('Training model')
    model = keras.Sequential([
        keras.layers.Dense(units=64, activation='relu', input_shape=(31,), name='input_layer'),
        keras.layers.Dense(units=32, activation='relu', name='hidden_layer_1'),
        keras.layers.Dense(units=1, activation='sigmoid', name='output_layer')
        # Single output unit for binary classification
    ])

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',  # Use binary_crossentropy for binary classification
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=1000,
        validation_data=(x_test, y_test),
        verbose=False
    )
    print(f'\nHistory: {history}')
    print(f'\nElapsed time in training: {time.time() - start_time} seconds')

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy}")

    # predictions = model.predict(new_data)
