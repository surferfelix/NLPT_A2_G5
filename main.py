""" file that carries out your entire experiment (feature extraction, training, testing) using command line arguments
for potential parameters (e.g. filepaths) + all of your other scripts. """

from helper_functions.extract_features import create_feature_files
from helper_functions.extract_predicates_and_arguments_rule_based import create_tokens_predicate_dataframe
from helper_functions.proper_evaluation import run_evaluation
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
    # TODO: add SVM for identification
    print("You chose predicates and arguments identification with SVM method ")
    arg_identification = "arg_identification"
    pred_identification = "pred_identification"



""" Arguments classification """
print("Runs arguments classification with SVM method")
# add SVM for classification


""" Evaluation """
folder_name = "output"
pred_svm_path = "pred_identification"

print("Evaluate arguments identification")
run_evaluation(f"{folder_name}/{arg_identification}")

print("Evaluate predicates identification")
run_evaluation(f"{folder_name}/{pred_identification}")

print("Evaluate arguments classification")
run_evaluation(f"{folder_name}/{pred_svm_path}")










