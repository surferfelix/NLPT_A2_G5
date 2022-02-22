import csv
import os
import sys
from nltk import pos_tag, WordNetLemmatizer
from typing import List
from utils import CONFIG


# inspired by https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment3/CRF.py, extract_sents_from_conll
def extract_tokenized_sentences(file_path: str) -> List[List[str]]:
    """Extract a list of tokens for every sentence from the corpus and return it."""
    tokenized_sentences = []
    sentence_tokens = []

    with open(file_path, 'r', encoding='utf8') as infile:
        filereader = csv.reader(infile, delimiter='\t', quotechar='\\')
        for row in filereader:
            if not row:  # empty line, end of sentence
                tokenized_sentences.append(sentence_tokens)
                sentence_tokens = []
            else:
                token = row[3]
                sentence_tokens.append(token)

    tokenized_sentences.append(sentence_tokens)  # append the last sentence; there is no empty line at the end of file

    return tokenized_sentences


# inspired by
# https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
def transform_pen_treebank_pos_tag(pen_treebank_pos_tag: str) -> str:
    """Transform a TreeBank PoS tag into a PoS tag that can be passed to WordNet Lemmatizer and return it."""
    # valid options for WordNetLemmatizer PoS tags  are “n” for nouns, “v” for verbs, “a” for adjectives, “r” for
    # adverbs and “s” for satellite adjectives, https://www.nltk.org/api/nltk.stem.wordnet.html

    tag = pen_treebank_pos_tag[0].lower()  # only look at the first letter, it already has all the info we need
    if tag == 'j':  # adjective
        return 'a'
    elif tag in ['n', 'r', 'v']:  # noun, adverb, verb
        return tag
    else:
        return 'n'  # WordNetLemmatizer assigns 'n' by default to anything that is not a noun, adjective, adverb or verb


def lemmatize_pos_tagged_sentences(pos_tagged_sentences: List[List[tuple]]) -> List[List[str]]:
    """Extract a list of lemmas for every sentence from a list of POS tagged sentence tokens and return it."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []

    for pos_tagged_sentence in pos_tagged_sentences:
        sentence_lemmas = []
        for token, tag in pos_tagged_sentence:
            sentence_lemmas.append(lemmatizer.lemmatize(token, transform_pen_treebank_pos_tag(tag)))
        lemmatized_sentences.append(sentence_lemmas)

    return lemmatized_sentences


def generate_preprocessed_file(infile_path: str, outfile_path: str, lemmatized_sentences: List[List[str]],
                               pos_tagged_sentences: List[List[tuple]]) -> None:
    """Generate a new file containing information obtained through preprocessing."""
    with open(outfile_path, 'w',  newline='', encoding='utf8') as outfile:
        filewriter = csv.writer(outfile, delimiter='\t', quotechar='\\')

        with open(infile_path, 'r', encoding='utf8') as infile:
            filereader = csv.reader(infile, delimiter='\t', quotechar='\\')

            sentence_index = 0
            token_index = 0
            for inrow in filereader:
                if not inrow:  # empty line
                    filewriter.writerow(inrow)  # write an empty row to output file
                    sentence_index += 1
                    token_index = 0
                else:
                    outrow = inrow[:-1] + [lemmatized_sentences[sentence_index][token_index],
                                           pos_tagged_sentences[sentence_index][token_index][1]] + [inrow[-1]]
                    filewriter.writerow(outrow)
                    token_index += 1


def main(paths=None) -> None:
    """Preprocess input file and save a preprocessed version of it."""
    if not paths:  # if no paths are passed to the function
        paths = sys.argv[1:]

    if not paths:  # if no paths are passed to the function through the command line
        paths = [CONFIG['train_path'], CONFIG['dev_path']]

    for path in paths:
        print(f'Preprocessing {os.path.basename(path)}')
        tokenized_sentences = extract_tokenized_sentences(path)
        pos_tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
        lemmatized_sentences = lemmatize_pos_tagged_sentences(pos_tagged_sentences)

        preprocessed_path = path.replace('.txt', '_preprocessed.txt')
        generate_preprocessed_file(path, preprocessed_path, lemmatized_sentences, pos_tagged_sentences)


if __name__ == '__main__':
    main()
