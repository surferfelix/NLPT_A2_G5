
import pandas as pd
import csv
from gensim.models import Word2Vec, KeyedVectors
import spacy
from spacy.tokens import Doc


def preprocessing_raw_data(raw): # No longer using this function; done by Alicja now
    '''Exports preprocessed raw data file'''
    container = []  # List of lists
    with open(raw) as raw_file:
        in_raw = csv.reader(raw_file, delimiter='\t', quotechar='|')
        for row in in_raw:
            if row:
                if row[0].startswith('#'):
                    continue
                else:
                    container.append(row)

    with open('cleaned_data/clean_raw_test_data.tsv', 'w') as output_file:
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


def fetch_tokens_from_data(input):
    '''Returns a list of lists, where inner list represents sentence'''
    tokens = pd.read_csv(input, sep='\t', quotechar='|', header=None).iloc[:, 2]
    sen_nrs = pd.read_csv(input, sep='\t', quotechar='|', header=None).iloc[:, 1]
    start_sen_index = 1
    sentence_holder = []
    container = []
    for sen, token in zip(sen_nrs, tokens):
        if int(sen) != start_sen_index:
            sentence_holder.append(token)
        else:
            start_sen_index += 1
            container.append(sentence_holder)
            sentence_holder.clear()
            sentence_holder.append(token)
    return container


def initialise_spacy():
    '''Will initialise the Spacy NLP object'''
    nlp = spacy.load('en_core_web_sm')
    return nlp


def get_tokens(doc):
    return [token for token in doc]


def get_embedding_representation_of_token(tokens: list, embeddingmodel='', dimensions=100) -> list:
    ''' Function to get the embedding representation of a token if this exists
    :param tokens: spacy doc object for the token
    :param embeddingmodel: a loaded w2v style pre-trained embedding model
    :dimensions: can be adjusted if not using a 100 dimension embeddingmodel
    '''
    vector_reps = []
    if not embeddingmodel:
        tokens = [token.text for token in tokens]
        language_model = Word2Vec(tokens,
                                  min_count=200)  # Change min_count for smaller datasets (needs to be proportional)
        language_model = language_model.wv
        if dimensions != 100:
            print('WARNING: No pre-trained embedding models only support 100 dimensions...\n')
            print('\nSetting dimensions to 100')
            dimensions = 100
    elif embeddingmodel:
        language_model = embeddingmodel
    for token in tokens:
        if token in language_model:
            vector = language_model[token]
        else:
            vector = [0] * dimensions
        vector_reps.append(vector)
    return vector_reps

def extract_constituencies(input):
    '''Will retrieve constituents for each text part in list
    :param input: takes a spacy doc object as 
    return: returns a container with the constituents after parsing'''
    #inspired by [https://github.com/nikitakit/self-attentive-parser][16-02-2022]
    for i in input:
        print(i)
    # nlp = initialise_spacy()
    # container = []
    # doc = nlp(input[0])
    # for sent in list(doc.sents):
    #     container.append(sent._.parse_string)
    # return container

def extract_features(input_data):
    '''Extracts the tokens, lemmas, and heads from the data
    :type input_data: a pd.DataFrame object
    '''
    def custom_tokenizer(text): # https://stackoverflow.com/questions/53594690/is-it-possible-to-use-spacy-with-already-tokenized-input
        if text in token_dict:
            return Doc(nlp.vocab, token_dict[text])
        else:
            raise ValueError('No tokenization available for input')
    
    token_dict = {}
    tokens = []
    heads = []
    lemmas = []
        
    df = pd.read_csv(input_data, sep='\t', quotechar='|', header=None)
    df_temp = df.iloc[:, [12, 2]]
    df_temp.columns = ['sentence_no', 'tokens']
    df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()

    nlp = initialise_spacy()

    nlp.tokenizer = custom_tokenizer
    sentences = list(df_temp['tokens'])
    
    for sentence in sentences: # Initializing tokenizer dict
        full_text = ' '.join(sentence)
        token_dict[full_text] = sentence

    for sentence in sentences: # Need to do this twice since now the tokenizer dict is initialized
        doc = nlp(' '.join(sentence))
        tokens_in_sentence = get_tokens(doc)
        extract_constituencies(doc)
    #     for token in doc:
    #         tokens.append(token)
    #         heads.append(token.head.text)
    #         lemmas.append(token.lemma_)

    # return tokens, lemmas, heads


def write_feature_out(tokens, lemmas, heads, embedding_model, input_path):
    '''Takes the features as input and writes a tsv file
    :param tokens: output of extract_features function
    :param lemmas: the lemmatized tokens, also output of extract_features function
    :param heads: the heads of the sentences, also output of extract_features function
    :embedding_model: a loaded w2v embedding_model
    '''
    # tokens = [token.text for token in tokens]  # Need to conv for embedding loading
    # embeddings = get_embedding_representation_of_token(tokens, embedding_model)
    # df = pd.DataFrame([*zip(tokens, lemmas, heads, embeddings)])
    print(tokens, lemmas, heads)
    df = pd.DataFrame(*[zip(tokens, lemmas, heads)])
    old_df = pd.read_csv(input_path, sep='\t', quotechar='|', header = None)
    big_df = pd.concat([df, old_df], ignore_index=True, axis=1)
    big_df.to_csv('processed_data/feature_file.tsv', sep='\t', quotechar='|', header = None)


def create_feature_files(input_data, loaded_embeddings):
    embedding_model = loaded_embeddings
    extract_features(input_data) # tokens, lemmas, heads = 
    # write_feature_out(tokens, lemmas, heads, embedding_model, input_data)


if __name__ == '__main__':
    input_data = "cleaned_data/final_train.tsv"
    path_to_emb = 'wiki_embeddings.txt'  # Add path to embedding model here
    print('Loading Embeddings')
    # loaded_embeddings = KeyedVectors.load_word2vec_format(path_to_emb)
    print('Embeddings loaded...')
    print('Iterating over data..')
    loaded_embeddings = ''
    create_feature_files(input_data, loaded_embeddings)
    print('Done')
