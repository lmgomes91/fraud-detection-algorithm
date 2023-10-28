import argparse

from methods.kohonen_scikit_learn import kohonen_sklearn
from methods.lstm_tensorflow import lstm_tensorflow
from methods.rnn_tensorflow import rnn_tensorflow
from methods.rbf_scikit_learn import rbf_sklearn
from utils.csv_handle import open_csv_as_dataframe
from methods.mlp_scikit_learn import mlp_sklearn


def main(main_args: argparse.Namespace) -> None:
    credit_card_df = open_csv_as_dataframe('./dataset', 'CreditCard')

    match main_args.method:
        case 'kohonen':
            kohonen_sklearn(credit_card_df)
        case 'lstm':
            raise "Not supported yet"
            lstm_tensorflow(credit_card_df)
        case 'mlp':
            mlp_sklearn(credit_card_df)
        case 'rbf':
            rbf_sklearn(credit_card_df)
        case 'rnn':
            rnn_tensorflow(credit_card_df)
        case _:
            raise f'No method matched for {main_args.method}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter to set what method will run")
    parser.add_argument('--method', type=str, help="First argument")
    args = parser.parse_args()
    main(args)
