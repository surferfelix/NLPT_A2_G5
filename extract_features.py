import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import spacy


def initialise_spacy():
    nlp = spacy.load('en_core_web_sm')
    return nlp

def get_tokens(doc):
    return [token for token in doc]

def get_embedding_representation_of_token(tokens: list, embeddingmodel = '', dimensions = 100) -> list:
    '''
    :param tokens: spacy doc object for the token'''
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

def extract_features(input_data, embedding_model = ''):
    tokens = []
    sentences = input_data['Sentence']
    nlp = initialise_spacy()
    for sentence in sentences:
        doc = nlp(sentence)
        tokens_in_sentence = get_tokens(doc)
        for token in tokens_in_sentence:
            tokens.append(token)
    lemmas = [token.lemma_ for token in tokens]
    embeddings = get_embedding_representation_of_token(tokens)
    df= pd.DataFrame({'Tokens': tokens, 'Lemmas': lemmas, 'embeddings': embeddings})
    df.to_csv('processed_data/feature_file.tsv', sep = '\t', quotechar = '|')

def main(input_data: str, embedding_model: str):
    extract_features(input_data, embedding_model)

if __name__ == '__main__':
    input_data = pd.read_csv("cleaned_data/en_ewp-up-train_clean_sentences.conllu", sep='\t')
    path_to_emb = '/Volumes/Samsung_T5/Text_Mining/Models/enwiki_20180420_100d.txt'
    print('Loading Embeddings')
    loaded_embeddings = KeyedVectors.load_word2vec_format(path_to_emb)
    print('Embeddings loaded...')
    main(input_data, loaded_embeddings)