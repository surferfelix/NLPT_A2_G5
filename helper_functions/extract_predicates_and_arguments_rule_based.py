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
        if token.pos_ == "VERB":
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
    nlp = spacy.blank('en')
    return nlp


def get_sentence_predicates_and_arguments(input_path: str):
    subject_list = []
    predicate_list = []
    arguments_list = []
    tokens = []

    df = pd.read_csv(input_path, sep='\t', quotechar='|')
    df_temp = df.iloc[:, [0, 2]]
    df_temp.columns = ['sentence_no', 'tokens']
    df_temp = df_temp.groupby('sentence_no')['tokens'].apply(list).reset_index()

    nlp = initialise_spacy()
    sentences = list(df_temp['tokens'])

    for i in range(len(sentences)):
        sentence = sentences[i]
        # sentence = [x for x in sentence if x]
        doc = Doc(nlp.vocab, words=sentence)

        subject, verb, attribute = extract_svo(doc)
        subject_list.append([subject])
        predicate_list.append([verb])
        arguments_list.append([attribute])
        tokens.append(sentence)

    data = pd.DataFrame({"tokens": tokens})

    data['subject'] = subject_list
    data['predicate'] = predicate_list
    data['arguments'] = arguments_list
    return data, df


def create_tokens_predicate_dataframe(input_path: str):
    processed_data, raw_data = get_sentence_predicates_and_arguments(input_path)
    predicate_argument_list = []
    for j in range(len(processed_data)):
        for k in range(len(processed_data['tokens'][j])):
            predicate = processed_data['predicate'][j][0]
            arguments = processed_data['arguments'][j][0]
            token = processed_data['tokens'][j][k]
            if token in predicate:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 1, "if_argument": 0})
            elif token in arguments:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 0, "if_argument": 1})
            else:
                predicate_argument_list.append({"number": j+1, "token": token, "if_predicate": 0, "if_argument": 0})

    final_df = pd.DataFrame(predicate_argument_list)
    final_df['gold_predicate_binary'] = list(raw_data['gold_predicate_binary'])
    final_df['gold_arguments_binary'] = list(raw_data['gold_arguments_binary'])

    predicate_final_data = final_df[['if_predicate', 'gold_predicate_binary']]
    argument_final_data = final_df[['if_argument', 'gold_arguments_binary']]

    predicate_final_data.columns = ['predict', 'gold']
    argument_final_data.columns = ['predict', 'gold']

    predicate_final_data.to_csv("../output/rule_arg_identification.tsv", sep='\t', quotechar='|', index=False)
    argument_final_data.to_csv("../output/rule_pred_identification.tsv", sep='\t', quotechar='|', index=False)

    return predicate_final_data, argument_final_data


# gather the user input and gather the info

if __name__ == "__main__":
    input_data = "../cleaned_data/final_test.tsv"
    predicate_final, argument_final = create_tokens_predicate_dataframe(input_data)


