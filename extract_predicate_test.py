
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


# gather the user input and gather the info

if __name__ == "__main__":

    raw_data = pd.read_csv("cleaned_data/en_ewt-up-trai_full_sentences.tsv", sep='\t')

    raw_data['Sentence'] = raw_data['Sentence'].map(lambda x: x.lstrip('= '))

    final_data = raw_data.copy()

    subject_list = []
    predicate_list = []
    arguments_list = []
    token_list = []
    for i in range(len(final_data['Sentence'])):
        sentence = final_data['Sentence'][i]
        doc = nlp(sentence)
        subject, verb, attribute = extract_svo(doc)

        subject_list.append([subject])
        predicate_list.append([verb])
        arguments_list.append([attribute])
        tokens = [token.text for token in doc]
        token_list.append(tokens)
        # print out the pos and deps
        # for token in doc:
        #     print("Token {} POS: {}, dep: {}".format(token.text, token.pos_, token.dep_))

    final_data['subject'] = subject_list
    final_data['predicate'] = predicate_list
    final_data['arguments'] = arguments_list
    final_data['tokens'] = token_list

    final_data.to_csv("simplest_extraction.csv")

    token_list = [val for sublist in list(final_data['tokens']) for val in sublist]

    token_df = pd.DataFrame({"tokens": token_list})

    print("stop")

