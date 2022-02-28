""" Code source: https://github.com/Dimev/Spacy-SVO-extraction"""

import spacy
import pandas as pd
import spacy
from spacy.tokens import Doc

# nlp = spacy.load("en_core_web_sm")

# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
# tags that define wether the word is wh-
WH_WORDS = {"WP", "WP$", "WRB"}


# extract the subject, object and verb from the input
def extract_svo(doc):
    sub = []
    at = []
    ve = []
    for token in doc:
        # is this a verb?
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
    return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(at).strip().lower()


# wether the doc is a question, as well as the wh-word if any
def is_question(doc):
    # is the first token a verb?
    if len(doc) > 0 and doc[0].pos_ == "VERB":
        return True, ""
    # go over all words
    for token in doc:
        # is it a wh- word?
        if token.tag_ in WH_WORDS:
            return True, token.text.lower()
    return False, ""


def initialise_spacy():
    ''' Will initialise the Spacy NLP object '''
    nlp = spacy.load('en_core_web_sm')
    return nlp


def get_sentence_predicates_and_arguments(input_path: str):
    subject_list = []
    predicate_list = []
    arguments_list = []
    tokens = []
    token_dict = {}

    df = pd.read_csv(input_path, sep='\t', quotechar='|', header=None)
    df_temp = df.iloc[:, [0, 2]]
    df_temp.columns = ['sentence_no', 'tokens']
    df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()

    nlp = initialise_spacy()
    sentences = list(df_temp['tokens'])

    # for i in range(len(sentences)):
    #     sentence = sentences[i]
    #     # sentence = [x for x in sentence if x]
    #     doc = Doc(nlp.vocab, words=sentence)
    for sentence in sentences: # Initializing tokenizer dict
        full_text = ' '.join(sentence)
        token_dict[full_text] = sentence

    for sentence in sentences: # Need to do this twice since now the tokenizer dict is initialized
        doc = nlp(' '.join(sentence))

        subject, verb, attribute = extract_svo(doc)
        subject_list.append([subject])    
        predicate_list.append([verb])
        arguments_list.append([attribute])
        tokens.append(sentence)

    data = pd.DataFrame({"tokens": tokens})

    data['subject'] = subject_list
    data['predicate'] = predicate_list
    data['arguments'] = arguments_list
    return data


def create_tokens_predicate_dataframe(data):
    predicate_argument_list = []
    for j in range(len(data)):
        for k in range(len(data['tokens'][j])):
            subjects = data['subject'][j][0]
            predicate = data['predicate'][j][0]
            arguments = data['arguments'][j][0]
            token = data['tokens'][j][k]
            if token in predicate:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 1, "if_argument": 0})
            elif token in arguments:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 0, "if_argument": 1})
            elif token in subjects:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 0, "if_argument": 1})
            else:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 0, "if_argument": 0})

    predicate_df = pd.DataFrame(predicate_argument_list)
    return predicate_df


# gather the user input and gather the info

if __name__ == "__main__":
    input_data = "cleaned_data/clean_raw_train_data.tsv"
    final_data = get_sentence_predicates_and_arguments(input_data)

    predicate_argument_df = create_tokens_predicate_dataframe(final_data)
    predicate_argument_df.to_csv("processed_data/pred_arg_labels_0.csv")

