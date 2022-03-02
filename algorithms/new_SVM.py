import csv
import sys
import os
import pandas as pd
from tabulate import tabulate
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix



def extract_features_and_labels(file_path, selected_features, label):
    """Extract a set of features and gold labels from file."""

    features = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as infile:
        # restval specifies value to be used for missing values
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='|')
        for row in reader:
            feature_dict = {}
            for feature_name in selected_features:
                if row[feature_name]:  # if there is a value for this feature
                    feature_dict[feature_name] = row[feature_name]
            features.append(feature_dict)
            labels.append(row[label])

    return features, labels

def create_classifier(train_features, train_labels):
    """Vectorize features and create classifier from training data."""

    classifier = LinearSVC(random_state=42)
    vec = DictVectorizer()
    train_features_vectorized = vec.fit_transform(train_features)
    print("training... this will take some time")
    classifier.fit(train_features_vectorized, train_labels)
        
    return classifier, vec


def get_predicted_and_gold_labels(test_path, vectorizer, classifier, selected_features, label):
    """Vectorize test features and get predictions."""

    # we use the same function as above (guarantees features have the same name and form)
    test_features, gold_labels = extract_features_and_labels(test_path, selected_features, label)
    
    # we need to use the same fitting as before, so now we only transform the current features according to this
    # mapping (using only transform)
    test_features_vectorized = vectorizer.transform(test_features)
    print(f'Using vectors of dimensionality {test_features_vectorized.shape[1]}')
    predictions = classifier.predict(test_features_vectorized)

    return predictions, gold_labels

def run_classifier_and_return_predictions_and_gold(train_path, test_path, selected_features, label):
    """Run classifier and get predictions using default parameters or cross validation."""

    train_features, train_labels = extract_features_and_labels(train_path, selected_features, label)

   
    classifier, vectorizer = create_classifier(train_features, train_labels)

    predictions, gold_labels = get_predicted_and_gold_labels(test_path, vectorizer, classifier, 
                                                             selected_features, label)

    return predictions, gold_labels

def generate_confusion_matrix(predictions, gold_labels):
    """Generate a confusion matrix."""

    labels = sorted(set(gold_labels))
    cf_matrix = confusion_matrix(gold_labels, predictions, labels=labels)
    # transform confusion matrix into a dataframe
    df_cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    return df_cf_matrix


def calculate_precision_recall_f1_score(predictions, gold_labels, digits=3):
    """Calculate evaluation metrics."""

    # get the report in dictionary form
    report = classification_report(gold_labels, predictions, zero_division=0, output_dict=True)
    # remove unwanted metrics
    report.pop('accuracy')
    report.pop('weighted avg')
    # transform dictionary into a dataframe and round the results
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(digits)
    df_report['support'] = df_report['support'].astype(int)

    return df_report 

def evaluate_classifier(predictions, gold_labels, selected_features, name):
    """Produce full evaluation of classifier."""

    print(f"Evaluating {name.replace('_', ' ')} with {', '.join(selected_features)} as features:")

    cf_matrix = generate_confusion_matrix(predictions, gold_labels)
    report = calculate_precision_recall_f1_score(predictions, gold_labels)

    print(cf_matrix)
    # print(cf_matrix.to_latex())  # print and paste to Overleaf

    print(report)
    # print(report.to_latex())  # print and paste to Overleaf

def run_and_evaluate_a_system(train_path, test_path, selected_features, name, label):
    """Run full classification and evaluation of a system."""

    predictions, gold_labels = run_classifier_and_return_predictions_and_gold(train_path, test_path, 
                                                                              selected_features, label)
         
    
    print(f"Running {name.replace('_', ' ')}")

    
    evaluate_classifier(predictions, gold_labels, selected_features, name)

################################################################
# change the paths for different tasks 

paths = ['../cleaned_data/clean_train_arguments.tsv',
                '../cleaned_data/clean_test_arguments.tsv']

# change the features for different tasks 

selected_features = [ '2', '3', '4','5', '6', '7', '8', '9', 'sentence_no']

def svm_for_argument_classification(paths, selected_features, label = 'arguments'):       

    train_path = paths[0]
    test_path = paths[1]

    name = "argument_classification_SVM"
    label = "arguments"
    run_and_evaluate_a_system(train_path, test_path, selected_features, name, label)
    
def svm_for_predicate_identification(paths, selected_features, label = 'predicate'):       
 

    train_path = paths[0]
    test_path = paths[1]

    name = "predicate_identification_SVM"
    run_and_evaluate_a_system(train_path, test_path, selected_features, name, label)
    
def svm_for_argument_identification(paths, selected_features, label = 'arguments'):       

    train_path = paths[0]
    test_path = paths[1]

    name = "argument_identification_SVM"
    run_and_evaluate_a_system(train_path, test_path, selected_features, name, label)

# svm_for_argument_classification(paths, selected_features, label = 'arguments')