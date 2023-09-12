from methods.mlp_tensorflow import mlp_tensorflow
from utils.csv_handle import open_csv_as_dataframe
from methods.mlp_skitlearn import mlp_sklearn


def main() -> None:
    credit_card_df = open_csv_as_dataframe('./dataset', 'CreditCard')
    mlp_sklearn(credit_card_df)
    # mlp_tensorflow(credit_card_df)


if __name__ == '__main__':
    main()
