""" Create srl_data with each predicate """

import pandas as pd


def extended_dataset_seperate_predicates(data_path: str, output_path: str):
    data = pd.read_csv(data_path, sep='\t', quotechar='|', header=None)

    data[12] = data[12].fillna("_")
    df_temp = data.iloc[:, [0, 2]]
    df_temp.columns = ['sentence_no', 'tokens']
    df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()

    final_data = pd.DataFrame()

    argument_no = 0
    for i in range(len(df_temp)):
        sentence = df_temp['tokens'][i]
        number = df_temp['sentence_no'][i]
        frozen_df = data[data.iloc[:, 0] == number]
        predicate_count = 0
        for k in range(len(frozen_df)):

            predicate_info = frozen_df[11].iloc[k]

            if predicate_info != '_':
                argument_no += 1
                predicate_count += 1
                predicate_arguments = frozen_df[11 + predicate_count]
                tmp_df = frozen_df.iloc[:, 0:10]
                tmp_df['predicate'] = '_'
                tmp_df['predicate'].iloc[k] = predicate_info
                tmp_df['arguments'] = list(predicate_arguments)
                tmp_df['sentence_no'] = number
                tmp_df['argument_number'] = argument_no
                final_data = pd.concat([final_data, tmp_df])

        if predicate_count == 0:
            predicate_arguments = frozen_df[11 + predicate_count]
            tmp_df = frozen_df.iloc[:, 0:10]
            tmp_df['predicate'] = '_'
            tmp_df['sentence_no'] = number
            tmp_df['arguments'] = list(predicate_arguments)
            tmp_df['argument_number'] = argument_no
            final_data = pd.concat([final_data, tmp_df])

    def predicate(row):
        if row["predicate"] == "_":
            val = 0
        else:
            val = 1
        return val

    def arguments(row):
        if row["arguments"] == "_":
            val = 0
        else:
            val = 1
        return val

    final_data['arguments'] = final_data['arguments'].str.replace(r'^V', '_', regex=True)

    final_data['gold_predicate_binary'] = final_data.apply(predicate, axis=1)
    final_data['gold_arguments_binary'] = final_data.apply(arguments, axis=1)
    final_data.to_csv(output_path, sep='\t', quotechar='|', index=False)
    print('DONE')


if __name__ == "__main__":
    extended_dataset_seperate_predicates("../cleaned_data/clean_raw_train_data.tsv",
                                        "cleaned_data/final_train.tsv")
    extended_dataset_seperate_predicates("../cleaned_data/clean_raw_test_data.tsv",
                                         "cleaned_data/final_test.tsv")

