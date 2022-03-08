""" file that carries out your entire experiment (feature extraction,
training, testing) using command line arguments for potential parameters
(e.g. filepaths) + all of your other scripts. """
import pandas as pd
from extract_features import create_feature_files
import sys


if_rule_based = sys.argv[0]



""" Feature extraction """

# saves feature to the 'processed_data/feature_file.tsv'
input_data = "cleaned_data/final_train.tsv"
create_feature_files(input_data)

""" Get predicate labels with simple rules """


""" Evaluate the results """



""" Get arguments labels with simple rules """


""" Evaluate the results """



""" Get predicate labels with SVM """


""" Evaluate the results """



""" Get arguments labels with SVM """


""" Evaluate the results """





""" Get arguments labels with SVM """




""" Evaluate the results """










