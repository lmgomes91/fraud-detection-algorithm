import pandas as pd
import logging


def open_csv_as_dataframe(path_file: str, file_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(f'{path_file}/{file_name}.csv', index_col=None)
    except Exception as e:
        logging.error(f'Error when try to create csv dataframe: {e}')
