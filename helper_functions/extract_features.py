import nltk
import pandas as pd
import csv
from gensim.models import Word2Vec, KeyedVectors
import spacy
from spacy.tokens import Doc
import stanza


def preprocessing_raw_data(raw):  # No longer using this function; done by Alicja now
    '''Exports preprocessed raw srl_data file'''
    container = []  # List of lists
    with open(raw) as raw_file:
        in_raw = csv.reader(raw_file, delimiter='\t', quotechar='|')
        for row in in_raw:
            if row:
                if row[0].startswith('#'):
                    continue
                else:
                    container.append(row)

    with open('../cleaned_data/clean_raw_test_data.tsv', 'w') as output_file:
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


def initialize_stanza():  # In this project we use Stanza for constituency extraction
    nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, constituency', tokenize_pretokenized=True,
                          tokenize_no_ssplit=True)
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


def extract_features(input_data):
    '''Extracts the tokens, lemmas, and heads from the srl_data
    :type input_data: a pd.DataFrame object
    '''

    def custom_tokenizer(
            text):  # https://stackoverflow.com/questions/53594690/is-it-possible-to-use-spacy-with-already-tokenized-input
        if text in token_dict:
            return Doc(nlp.vocab, token_dict[text])
        else:
            raise ValueError('No tokenization available for input')

    token_dict = {}
    tokens = []
    heads = []
    lemmas = []
    named_entities = []
    sentence_for_token = []

    df = pd.read_csv(input_data, sep='\t', quotechar='|')
    df_temp = df.iloc[:, [0, 2]]
    df_temp.columns = ['sentence_no', 'tokens']
    df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()

    nlp = initialise_spacy()
    nlp.tokenizer = custom_tokenizer
    sentences = list(df_temp['tokens'])
    complete_stanza_input = []
    for sentence in sentences:  # Initializing tokenizer dict
        full_text = ' '.join(sentence)
        token_dict[full_text] = sentence
    for sentence in sentences:  # Need to do this twice since now the tokenizer dict is initialized
        complete_stanza_input.append(sentence)
        doc = nlp(' '.join(sentence))
        tokens_in_sentence = get_tokens(doc)
        for token in doc:
            sentence_for_token.append(sentence)
            tokens.append(token)
            heads.append(token.head.text)
            lemmas.append(token.lemma_)
            ne = token.ent_type_
            if ne == "":
                ne = "NONE"
            named_entities.append(ne)

    return tokens, lemmas, heads, named_entities, complete_stanza_input, sentence_for_token


def extract_features_from_file(old_df) -> list:
    '''Extracts old features from file
    :param inputfile: pd Dataframe with old pos tags'''
    pos_tags = old_df['4'].tolist()  # 4 includes the POS tags in the main file
    prevs = []
    nexts = []
    for index, tag in enumerate(pos_tags):
        # Previous POS
        try:
            prev_pos = pos_tags[index - 1]
        except IndexError:
            prev_pos = '_'
        prevs.append(prev_pos)
        # Next POS
        try:
            next_pos = pos_tags[index + 1]
        except IndexError:
            next_pos = '_'
        nexts.append(next_pos)
    return prevs, nexts


def get_stanza_constituents(complete_stanza_input):
    path_labels = []
    nlp2 = initialize_stanza()
    doc_stanza = nlp2(complete_stanza_input)
    doc_sentences = list(doc_stanza.sentences)
    # for each sentence in the text, get the tree paths for the tokens in the sentence
    for i in range(len(doc_sentences)):
        if i != 0:  # Skip header
            get_stanza_paths([], doc_sentences[i].constituency.children[0], path_labels)
    return path_labels


# Function taken from previous assignment and adjusted: https://github.com/efemeryds/NLP-technology-assignment-1/blob/main/code/extract_token_features.py
def get_stanza_paths(path_list, node, overarching_list):
    """
    Function that creates a constituency tree path for each word in text.
    """
    # check whether there is a syntax tree
    if path_list is None:
        return
        # if so, append current label
    path_list.append(node.label)
    # once you get to leaf, append path of the leaf
    if len(node.children) == 0:
        # exclude the leaf/word itself and add to overarching list
        overarching_list.append(path_list)
        # stop function
        return overarching_list
    for n in node.children:
        # all children need to have same subpath, which is why .copy() is needed
        # keep getting paths until leaf is reached
        get_stanza_paths(path_list.copy(), n, overarching_list)


def write_feature_out(tokens: list, lemmas: list, heads: list, named_entities: list, constituencies: list,
                      embedding_model, input_path: str, sentences_for_token):
    '''Takes the features as input and writes a tsv file
    :param tokens: output of extract_features function
    :param lemmas: the lemmatized tokens, also output of extract_features function
    :param heads: the heads of the sentences, also output of extract_features function
    :param constituencies: The path in the constituency tree to the specific token
    :embedding_model: a loaded w2v embedding_model
    '''
    tokens = [token.text for token in tokens]  # Need to conv for embedding loading
    # embeddings = get_embedding_representation_of_token(tokens, embedding_model)
    old_df = pd.read_csv(input_path, sep='\t', quotechar='|')
    prev_pos, next_pos = extract_features_from_file(old_df)
    everything = [tokens, lemmas, heads, prev_pos, next_pos]
    for index, i in enumerate(everything):
        if len(i) != len(tokens):
            print(index, len(i))
    df = pd.DataFrame({'tokens': tokens, 'lemmas': lemmas, 'heads': heads, 'named_entities': named_entities,
                       'constituencies': constituencies, 'sentences_for_token': sentences_for_token,
                       'Prev_pos': prev_pos, 'Next_pos': next_pos})
    big_df = df.merge(old_df, how='inner', left_index=True, right_index=True)
    write_path = input_path.split('/')[-1].rstrip('.tsv') + '_with_feature' + '.tsv'
    big_df.to_csv(f"feature_data/{write_path}", sep='\t', quotechar='|', index=False)


def create_feature_files(input_data, loaded_embeddings=''):
    embedding_model = loaded_embeddings
    tokens, lemmas, heads, named_entities, complete_stanza_input, sentences_for_token = extract_features(input_data)
    # DUE TO BUG COULD NOT GET WORKING: WAITING FOR RESOLVING STANZA ISSUE OR 'KEY' PARAMETER IMPLEMENTATION
    constituencies = ['not_working' for token in tokens]
    # get_stanza_constituents(complete_stanza_input)
    # for index, (tok, cons) in enumerate(zip(tokens, constituencies)):
    #     if tok.text != cons[-1]:
    #         print(tok, cons)
    #         print('Stanza tokenization alignment issue, adding _ to attempt srl_data alignment')
    #         constituencies.insert(index, ['_'])
    #         break
    write_feature_out(tokens, lemmas, heads, named_entities, constituencies, embedding_model, input_data,
                      sentences_for_token)

def get_embedding_pipe(file, embedding_model_path=''):
    '''Reads a tsv file and goes through tokens to match these in embedding model,
    make sure to set dimensions variable if model not == 100d
    :param file: file to load tokens from'''
    infile = pd.read_csv(file, sep = '\t', quotechar = '|')
    print('Loading Embeddings')
    loaded_embeddings = KeyedVectors.load_word2vec_format(embedding_model_path)
    print('Embeddings loaded...')
    tokens = infile["tokens"].tolist()
    vector_reps = get_embedding_representation_of_token(file, loaded_embeddings)
    return vector_reps

if __name__ == '__main__':

    data_paths = ["../cleaned_data/mini_final_train.tsv", "../cleaned_data/mini_final_test.tsv"]
    path_to_emb = '../wiki_embeddings.txt'  # Add path to embedding model here
    for input_data in data_paths:
        print(f'Starting run for {input_data}')
        print('Iterating over srl_data..')
        create_feature_files(input_data)
    print('Done')
