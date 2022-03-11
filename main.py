""" file that carries out your entire experiment (feature extraction, training, testing) using command line arguments
for potential parameters (e.g. filepaths) + all of your other scripts. """

from helper_functions.extract_features import create_feature_files
from helper_functions.extract_predicates_and_arguments_rule_based import create_tokens_predicate_dataframe
from helper_functions.proper_evaluation import run_evaluation
from helper_functions.new_SVM import run_model
import sys


if_rule_based = sys.argv[0] # 'yes', or 'no'
test_data_path = sys.argv[1] # string path to the test file, eg 'cleaned_data/final_test.tsv


""" Feature extraction """

# saves feature to the 'feature_data/feature_file.tsv'
input_data = "cleaned_data/final_train.tsv"
create_feature_files(input_data)


""" Predicates and arguments identification """
if if_rule_based == 'yes':
    print("You chose predicates and arguments identification with RULE BASED method ")
    create_tokens_predicate_dataframe(test_data_path)
    arg_identification = "rule_arg_identification"
    pred_identification = "rule_pred_identification"

else:
    train_path = 'feature_data/mini_final_train_with_features.tsv'
    test_path = 'feature_data/mini_final_te_with_features.tsv'

    print("You chose predicates and arguments identification with SVM method ")

    run_model(train_path, test_path, [], 'gold_predicate_binary', 'pred_identification')
    run_model(train_path, test_path, [], 'gold_arguments_binary', 'arg_identification')
    arg_identification = "arg_identification"
    pred_identification = "pred_identification"


""" Arguments classification """
print("Running arguments classification with SVM method..")

train_path = 'feature_data/mini_final_train_with_features.tsv'
test_path = 'feature_data/mini_final_te_with_features.tsv'

# TODO: add classification model


""" Evaluation """
folder_name = "output"
pred_svm_path = "pred_identification"

print("Evaluating arguments identification..")
run_evaluation(f"{folder_name}/{arg_identification}")

print("Evaluating predicates identification..")
run_evaluation(f"{folder_name}/{pred_identification}")

print("Evaluating arguments classification..")
run_evaluation(f"{folder_name}/{pred_svm_path}")










