""" file that carries out your entire experiment (feature extraction,
training, testing) using command line arguments for potential parameters
(e.g. filepaths) + all of your other scripts. """
import pandas as pd
from extract_features import create_feature_files

""" 1) Feature extraction """

# saves feature to the 'processed_data/feature_file.tsv'
# one needs to have proper word embeddings model downloaded to make it work
create_feature_files()

""" 2) Get predicate labels with simple rules """


""" 3) Get predicate labels with SVM """


""" 4) Get arguments labels with SVM """


""" 5) Evaluate the results """

""" a) rule-based predicates """


""" b) svm-based predicates """


""" c) svm-based arguments """


