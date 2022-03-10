import pandas as pd


def clean_data(data_path: str):
    data = pd.read_csv(data_path, sep='\t', quotechar='|')
    data['arguments'] = data['arguments'].str.replace(r'^V', '_', regex=True)
    data.to_csv(data_path, sep='\t', quotechar='|')
    print('DONE')


if __name__ == "__main__":
    train_path = "cleaned_data/clean_train_arguments.tsv"
    test_path = "cleaned_data/clean_test_arguments.tsv"
    clean_data(train_path)
    clean_data(test_path)


