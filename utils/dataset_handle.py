import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler


def pre_processing_dataframe(dataframe: pd.DataFrame) -> tuple[numpy.ndarray, numpy.ndarray]:
    try:
        dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
        dataframe['Class'] = dataframe['Class'].str.replace("'", '')
        dataframe['Class'] = dataframe['Class'].astype(int)

        output_class = dataframe.pop('Class').to_numpy()

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(dataframe)

        return normalized_data, output_class
    except Exception as e:
        print(f'Error when pre process the dataframe: {e}')

