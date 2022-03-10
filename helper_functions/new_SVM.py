import csv
import sys
import os
import pandas as pd
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
                if row[feature_name]:  # if this feature exists
                    feature_dict[feature_name] = row[feature_name]
            features.append(feature_dict)
            labels.append(row[label])

    return features, labels

def create_classifier(train_features, train_labels):
    """Vectorize features and create classifier from training srl_data."""

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

def run_classifier_and_return_predictions_and_gold(train_path, test_path, selected_features, label, name):
    """Run classifier and get predictions using default parameters or cross validation."""

    train_features, train_labels = extract_features_and_labels(train_path, selected_features, label) # train_features, train_labels = 

    classifier, vectorizer = create_classifier(train_features, train_labels)

    predictions, gold_labels = get_predicted_and_gold_labels(test_path, vectorizer, classifier, 
                                                             selected_features, label)

    
    list_dict = {'predict':predictions, 'gold':gold_labels} 
    df = pd.DataFrame(list_dict) 
    df.to_csv("../output/"+name+'.csv', sep='\t', quotechar='|', index=False)


def main(paths=None) -> None:
    """Preprocess input file and save a preprocessed version of it."""
    #if not paths:  # if no paths are passed to the function
    #    paths = sys.argv[1:]

    if not paths:  # if no paths are passed to the function through the command line
        
        paths = ['../feature_data/mini_final_train_with_feature.tsv', # FeatureFile
                        '../feature_data/mini_final_te_with_feature.tsv']  # Automatic Evaluation

    # change the features for different tasks 
    # 0 = token, 1 = spacy_lemma, 2 = head, 3= path, 4 = '?', 5 = '?', 6 = POS, 7 = type 8 = ?, 
    # 15 = GOLD PRED, # 16 = GOLD ARG
    selected_features = [ '0','2','4', '6', '7', '8']
    train_path = paths[0]
    test_path = paths[1]

    labels_name = {"15":"pred_identification", 
               "16":"arg_identification"}
    
    for label, name in labels_name.items():
        print(f'running {name}')
        run_classifier_and_return_predictions_and_gold(train_path, test_path, selected_features, 
                                                       label, name)   

if __name__ == '__main__':
    main()