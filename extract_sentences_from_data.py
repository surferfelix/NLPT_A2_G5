import csv


def read_file(path: str) -> list:
    '''Reads a SRL CoNLL tsv file and extracts all sentences'''
    all_sentences = list()
    with open(path) as file:
        infile = csv.reader(file, delimiter='\t', quotechar='|')
        for row in infile:
            if row:
                if row[0].startswith('# text ='):
                    all_sentences.append(row[0][7::])
    with open(f"{path.rstrip('.conllu')}_full_sentences.tsv", 'w') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|')
        writer.writerow(['Sentence'])
        for sentence in all_sentences:
            writer.writerow([sentence])


def main(path: str):
    ans = read_file(path)
    print(ans)


if __name__ == '__main__':
    path = 'raw_data/en_ewt-up-train.conllu'
    main(path)
