""" Code source: https://github.com/Dimev/Spacy-SVO-extraction"""

import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

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


def get_sentence_predicates_and_arguments(raw_data):
    data = raw_data.copy()

    subject_list = []
    predicate_list = []
    arguments_list = []
    token_list = []
    for i in range(len(data['Sentence'])):
        sentence = data['Sentence'][i]
        doc = nlp(sentence)
        subject, verb, attribute = extract_svo(doc)

        subject_list.append([subject])
        predicate_list.append([verb])
        arguments_list.append([attribute])
        tokens = [token.text for token in doc]
        token_list.append(tokens)

    data['subject'] = subject_list
    data['predicate'] = predicate_list
    data['arguments'] = arguments_list
    data['tokens'] = token_list
    return data


def create_tokens_predicate_dataframe(data):
    predicate_argument_list = []
    for j in range(len(data)):
        for k in range(len(data['tokens'][j])):
            predicate = data['predicate'][j][0]
            arguments = data['arguments'][j][0]
            token = data['tokens'][j][k]
            if token in predicate:
                predicate_argument_list.append({"number": j, "token": token, "if_predicate": 1, "if_argument": 0})
            elif token in arguments:
                predicate_argument_list.append({"number": j, "token": token, "if_predicate": 0, "if_argument": 1})
            else:
                predicate_argument_list.append({"number": j, "token": token, "if_predicate": 0, "if_argument": 0})

    predicate_df = pd.DataFrame(predicate_argument_list)
    return predicate_df


# gather the user input and gather the info

if __name__ == "__main__":
    input_data = pd.read_csv("cleaned_data/en_ewp-up-train_clean_sentences.conllu", sep='\t')

    final_data = get_sentence_predicates_and_arguments(input_data)
    final_data[['Sentence', 'tokens']].to_csv("cleaned_data/clean_sentences.csv")

    predicate_argument_df = create_tokens_predicate_dataframe(final_data)
    predicate_argument_df.to_csv("processed_data/pred_arg_labels_0.csv")

    clean_tokens = predicate_argument_df[['number', 'token']]
    clean_tokens.to_csv("cleaned_data/clean_tokens.csv")
