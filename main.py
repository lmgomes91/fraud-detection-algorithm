from utils.csv_handle import open_csv_as_dataframe
from methods.mlp import mlp


def main() -> None:
    credit_card_df = open_csv_as_dataframe('./dataset', 'CreditCard')
    mlp(credit_card_df)


if __name__ == '__main__':
    main()
