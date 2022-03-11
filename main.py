""" file that carries out your entire experiment (feature extraction, training, testing) using command line arguments
for potential parameters (e.g. filepaths) + all of your other scripts. """

from helper_functions.extract_features import create_feature_files
from helper_functions.extract_predicates_and_arguments_rule_based import create_tokens_predicate_dataframe
from helper_functions.proper_evaluation import run_evaluation
from helper_functions.new_SVM import run_model
import sys

if_rule_based = sys.argv[1]  # 'yes', or 'no'
if_mini_version = sys.argv[2]  # 'yes' or 'no'
if_with_embeddings = sys.argv[3] # 'yes' or 'no'

try:
    embedding_path = sys.argv[4]
except:
    pass

print(if_rule_based)

""" Feature extraction """

if if_mini_version == "yes":
    train_path = "mini_final_train_with_feature.tsv"
    test_path = "mini_final_te_with_feature.tsv"
    feature_train = "mini_final_train.tsv"
    rule_path = "mini_final_test.tsv"
else:
    train_path = "final_train_with_feature.tsv"
    test_path = "final_te_with_feature.tsv"
    feature_train = "final_train.tsv"
    rule_path = "final_test.tsv"

# saves feature to the 'feature_data/feature_file.tsv'
input_data = f"../cleaned_data/{feature_train}"
create_feature_files(input_data)

""" Predicates and arguments identification """
if if_rule_based == 'yes':
    name = "rule_based"
    print("You chose predicates and arguments identification with RULE BASED method ")
    create_tokens_predicate_dataframe(f"../cleaned_data/{rule_path}")
    arg_identification = "rule_arg_identification"
    pred_identification = "rule_pred_identification"

else:
    if if_with_embeddings == 'yes':
        name = "svm"
        print("You chose predicates and arguments identification with SVM method ")
        print("Using embedding feature..")
        folder_name_feature = "../feature_data"
        run_model(f"{folder_name_feature}/{train_path}", f"{folder_name_feature}/{test_path}", ['tokens', 'lemmas', 'heads', 'named_entities', 'sentences_for_token', 'Prev_pos',
                                   'Next_pos', '4', '5', '6', '8', '9', 'embedding'],
                  'gold_predicate_binary', 'emb_pred_identification')
        run_model(f"{folder_name_feature}/{train_path}", f"{folder_name_feature}/{test_path}", ['tokens', 'lemmas', 'heads', 'named_entities', 'sentences_for_token', 'Prev_pos',
                                   'Next_pos', '4', '5', '6', '8', '9', 'embedding'],
                  'gold_arguments_binary', 'emb_arg_identification', embedding_path)
        arg_identification = "emb_arg_identification"
        pred_identification = "emb_pred_identification"

    else:
        name = "svm"
        print("You chose predicates and arguments identification with SVM method ")
        folder_name_feature = "../feature_data"
        run_model(f"{folder_name_feature}/{train_path}", f"{folder_name_feature}/{test_path}", [], 'gold_predicate_binary', 'pred_identification')
        run_model(f"{folder_name_feature}/{train_path}", f"{folder_name_feature}/{test_path}", [], 'gold_arguments_binary', 'arg_identification')
        arg_identification = "arg_identification"
        pred_identification = "pred_identification"

""" Arguments classification """
print("Running arguments classification with SVM method..")

if if_with_embeddings == "yes":
    print("Using embedding feature..")
    pred_svm_path = 'emb_argument_classification'
    run_model(f"../feature_data/{train_path}", f"../feature_data/{test_path}", [], 'arguments',
              pred_svm_path, embedding_path)

else:
    pred_svm_path = "argument_classification"
    run_model(f"../feature_data/{train_path}", f"../feature_data/{test_path}", [], 'arguments', pred_svm_path)

""" Evaluation """
folder_name = "../output"


print("Evaluating arguments identification..")
run_evaluation(f"{folder_name}/{arg_identification}.tsv")

print("Evaluating predicates identification..")
run_evaluation(f"{folder_name}/{pred_identification}.tsv")

print("Evaluating arguments classification..")
run_evaluation(f"{folder_name}/{pred_svm_path}.tsv")
