import pandas as pd
import json
from typing import List
import csv


def preprocessing_raw_data(raw, file_type):
    '''Exports preprocessed raw srl_data file'''

    container = []  # List of lists
    with open(raw, encoding="utf-8") as raw_file:
        in_raw = csv.reader(raw_file, delimiter='\t', quotechar='|')
        for row in in_raw:
            if row:
                if row[0].startswith('#'):
                    continue
                else:
                    container.append(row)

    with open('../srl_data/clean_raw_' + file_type + '_data.tsv', 'w', encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter='\t', quotechar='|')
        # Getting the max line size
        padding_limit = max([len(line) for line in container])
        sentence_nr = []
        line_iter = 0
        for line in container:
            while len(line) != padding_limit:
                line.append('_')
            try:
                if int(line[0]) == 1:  # New line
                    line_iter += 1
            except ValueError:  # Some lines have periods in the index
                line_iter += 1
            writer.writerow([line_iter] + line)


def change_value(preprocessed_data, file_type):
    df = pd.read_csv('../srl_data/clean_raw_' + file_type + '_data.tsv', sep='\t', header=None, encoding="utf-8",
                     keep_default_na=False, quotechar='|', skip_blank_lines=False)

    length = len(df.iloc[0].tolist())
    for i in range(length - 11):
        row = i + 11
        df.iloc[:, row] = df.iloc[:, row].apply(lambda x: 'O' if x in ['', '_'] else x).tolist()
        if row >= 12:
            df.iloc[:, row] = df.iloc[:, row].apply(lambda x: "B-" + x if x
                                                                          not in ['', 'O'] else x).tolist()
    return df


def from_csv_to_json(df, file_type):
    schemas = []  # set up container

    # group sentence from sentences
    sentences = df.groupby(0)
    sum_length = 0
    for sentence in sentences:
        flag = 1
        num, df = sentence
        df[1] = df[1].astype('int64')
        for i in range(len(df)):
            verbs = df[11].iloc[i]
            if verbs == "O":
                continue

            # sent_number = df[df.iloc[:, 1]]
            if flag == 1:
                V = "V"

            else:
                V = "_"

            flag += 1

            schema = {
                'sep_words': df.iloc[:, 2].tolist(),
                'BIO': df.iloc[:, 12].tolist(),
                'pred_sense': [i, df.iloc[i, 11], V, df.iloc[i, 5]]
            }

            schemas.append(schema)

            # int(schema)
    with open("../srl_data/srl_univprop_en." + file_type + ".jsonl", 'w') as f:
        for entry in schemas:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    file_type = ['train', 'dev']
    for file in file_type:
        path = "../srl_data/srl_univprop_en." + file + ".conll"
        preprocessed_data = preprocessing_raw_data(path, file)
        df = change_value(preprocessed_data, file)
        from_csv_to_json(df, file)
