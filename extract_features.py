import pandas as pd
import csv
from gensim.models import Word2Vec, KeyedVectors
import spacy

def preprocessing_raw_data(raw):
    '''Exports preprocessed raw data file'''
    container = [] # List of lists 
    with open(raw) as raw_file:
        in_raw = csv.reader(raw_file, delimiter = '\t', quotechar = '|')
        for row in in_raw:
            if row:
                if row[0].startswith('#'):
                    continue
                else:
                    container.append(row)

    with open('cleaned_data/clean_raw_test_data.tsv', 'w') as output_file:
        writer = csv.writer(output_file, delimiter = '\t', quotechar = '|')
        # Getting the max line size
        padding_limit = max([len(line) for line in container])
        sentence_nr = []
        line_iter = 0
        for line in container:
            while len(line) != padding_limit:
                line.append('_')
            try:
                if int(line[0]) == 1: # New line
                    line_iter += 1
            except ValueError: # Some lines have periods in the index
                line_iter +=1
            writer.writerow([line_iter]+line)

def fetch_tokens_from_data():
    pass

def initialise_spacy():
    '''Will initialise the Spacy NLP object'''
    nlp = spacy.load('en_core_web_sm')
    return nlp

def get_tokens(doc):
    return [token for token in doc]

def get_embedding_representation_of_token(tokens: list, embeddingmodel = '', dimensions = 100) -> list:
    ''' Function to get the embedding representation of a token if this exists
    :param tokens: spacy doc object for the token
    :param embeddingmodel: a loaded w2v style pre-trained embedding model
    :dimensions: can be adjusted if not using a 100 dimension embeddingmodel
    '''
    vector_reps = []
    if not embeddingmodel:
        tokens = [token.text for token in tokens]
        language_model = Word2Vec(tokens, min_count=200) # Change min_count for smaller datasets (needs to be proportional)
        language_model = language_model.wv
        if dimensions != 100:
            print('WARNING: No pre-trained embedding models only support 100 dimensions...\n')
            print('\nSetting dimensions to 100')
            dimensions= 100
    elif embeddingmodel:
        language_model = embeddingmodel
    for token in tokens:
        if token in language_model:
            vector = language_model[token]
        else:
            vector = [0] * dimensions
        vector_reps.append(vector)
    return vector_reps

def extract_features(input_data):
    '''Extracts the tokens, lemmas, and heads from the data
    :type input_data: a pd.DataFrame object with 'Sentence' header'''
    tokens = []
    sentences = input_data['Sentence']
    heads = []
    nlp = initialise_spacy()
    for sentence in sentences:
        doc = nlp(sentence)
        tokens_in_sentence = get_tokens(doc)
        for token in tokens_in_sentence:
            tokens.append(token)
    heads = [token.head.text for token in tokens]
    lemmas = [token.lemma_ for token in tokens]
    return tokens, lemmas, heads

def write_feature_out(tokens, lemmas, heads, embedding_model):
    '''Takes the features as input and writes a tsv file
    :param tokens: output of extract_features function
    :param lemmas: the lemmatized tokens, also output of extract_features function
    :param heads: the heads of the sentences, also output of extract_features function
    :embedding_model: a loaded w2v embedding_model
    '''
    tokens = [token.text for token in tokens] # Need to conv for embedding loading
    embeddings = get_embedding_representation_of_token(tokens, embedding_model)
    df= pd.DataFrame({'Tokens': tokens, 'Lemmas': lemmas, 'Heads': heads, 'Embeddings': embeddings})
    df.to_csv('processed_data/feature_file.tsv', sep = '\t', quotechar = '|')

def main(input_data, embedding_model):
    tokens, lemmas, heads = extract_features(input_data)
    write_feature_out(tokens, lemmas, heads, embedding_model)

if __name__ == '__main__':
    input_data = 'raw_data/en_ewt-up-test.conllu'
    preprocessing_raw_data(input_data)
    # input_data = pd.read_csv("cleaned_data/en_ewp-up-train_clean_sentences.conllu", sep='\t')
    # path_to_emb = '' # Add path to embedding model here
    # print('Loading Embeddings')
    # loaded_embeddings = KeyedVectors.load_word2vec_format(path_to_emb)
    # print('Embeddings loaded...')
    # print('Iterating over data..')
    # main(input_data, loaded_embeddings)
    # print('Done')