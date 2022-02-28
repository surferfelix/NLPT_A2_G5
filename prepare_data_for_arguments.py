""" Create data with each predicate """

import pandas as pd

#input_path = "cleaned_data/clean_raw_train_data.tsv"
input_path = "cleaned_data/clean_raw_test_data.tsv"

data = pd.read_csv(input_path, sep='\t', quotechar='|', header=None)

df_temp = data.iloc[:, [0, 2]]
df_temp.columns = ['sentence_no', 'tokens']
df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()


final_data = pd.DataFrame()

argument_no = 0
for i in range(len(df_temp)):
    print(i)
    sentence = df_temp['tokens'][i]
    number = df_temp['sentence_no'][i]
    frozen_df = data[data.iloc[:, 0] == number]
    predicate_count = 0
    for k in range(len(frozen_df)):

        predicate_info = frozen_df[11].iloc[k]

        if predicate_info != '_':
            argument_no += 1
            predicate_count += 1
            predicate_arguments = frozen_df[11+predicate_count]
            tmp_df = frozen_df.iloc[:, 0:10]
            tmp_df['predicate'] = '_'
            tmp_df['predicate'].iloc[k] = predicate_info
            tmp_df['arguments'] = list(predicate_arguments)
            tmp_df['sentence_no'] = number
            tmp_df['argument_number'] = argument_no
            final_data = pd.concat([final_data, tmp_df])


final_data.to_csv("cleaned_data/clean_test_arguments.tsv", sep='\t', quotechar='|')
print('stop')






