import csv
import sys
import scipy.stats
import sklearn_crfsuite
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from utils import evaluate_classifier, write_predictions_to_file, CONFIG

# based on https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment3/CRF.py


def token2features(sentence, i, selected_features):
    """Extract features from token level tokenization."""

    features = {'bias': 1.0}

    for feature in selected_features:
        value = sentence[i][feature]

        if value:  # if there is a value for this feature
            features[feature] = value

    if i == 0:
        features['BOS'] = True
    elif i == len(sentence) - 1:
        features['EOS'] = True

    return features


def sent2features(sent, selected_features):
    """Extract features from sentence level tokenization."""

    return [token2features(sent, i, selected_features) for i in range(len(sent))]


def sent2labels(sent):
    """Get gold labels from sentence level tokenization."""

    return [d['gold_label'] for d in sent]


def extract_sents_from_file(file_path):
    """Read in file and recompose sentences from tokens."""

    sents = []
    current_sent = []

    with open(file_path, 'r', encoding='utf8') as infile:
        reader = csv.DictReader(infile, restval='', delimiter='\t', quotechar='\\')
        for row in reader:
            if row['next_lemma'] == 'eos':
                current_sent.append(row)
                sents.append(current_sent)
                current_sent = []
            else:
                current_sent.append(row)

    return sents


def train_crf_model(X_train, y_train):
    """Create classifier with default parameters."""

    classifier = sklearn_crfsuite.CRF()  # use the default parameters

    classifier.fit(X_train, y_train)

    return classifier


def train_custom_crf_model(X_train, y_train):
    """Manually create a classifier using the best parameters obtained through CV."""

    classifier = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                      c1=0.18164392998437914,
                                      c2=0.005138865750125312,
                                      max_iterations=100,
                                      all_possible_transitions=True)

    classifier.fit(X_train, y_train)

    return classifier


def train_crf_model_using_cross_validation(X_train, y_train):
    """Create classifier using cross validation."""

    print("Running cross validation, this will take a while and you should get some Future Warnings")

    # part of the code below taken from
    # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#hyperparameter-optimization

    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               max_iterations=100,
                               all_possible_transitions=True)

    params_space = {'c1': scipy.stats.expon(scale=0.5),
                    'c2': scipy.stats.expon(scale=0.05)}

    # # we can't use the same scoring as for SVM because labels are represented differently > as a list of lists, 1 list
    # # of labels per sentence
    f1_scorer = make_scorer(metrics.flat_f1_score, average='macro')

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=5,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer,
                            random_state=42)

    try:
        rs.fit(X_train, y_train)

    except AttributeError:  # https://github.com/TeamHG-Memex/sklearn-crfsuite/issues/60
        print("You have to use sklearn version lower than 0.24 to be able to run cross-validation.")
        print("You can install it by typing \'pip install -U 'scikit-learn<0.24'\' into your terminal.")

    else:
        print(f'Done! Best parameters: {rs.best_params_}')
        print(f'Best result on the training set: {round(rs.best_score_, 3)} macro avg f1-score')

    return rs.best_estimator_


def create_crf_model(train_path, selected_features, cross_validation=False, custom=False):
    """Create classifier using default parameters or cross validation."""

    train_sents = extract_sents_from_file(train_path)
    X_train = [sent2features(s, selected_features) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    if cross_validation:
        crf = train_crf_model_using_cross_validation(X_train, y_train)
    elif custom:
        crf = train_custom_crf_model(X_train, y_train)
    else:
        crf = train_crf_model(X_train, y_train)

    return crf


def run_crf_model(crf, test_path, selected_features):
    """Extract features from test set, run classifier and get predictions."""

    test_sents = extract_sents_from_file(test_path)
    X_test = [sent2features(s, selected_features) for s in test_sents]
    y_pred = crf.predict(X_test)

    return y_pred


def train_and_run_crf_model(train_path, test_path, selected_features, cross_validation=False, custom=False):
    """Run classifier and get predictions."""

    crf = create_crf_model(train_path, selected_features, cross_validation, custom)

    pred_labels = run_crf_model(crf, test_path, selected_features)

    predictions = []
    for pred in pred_labels:
        predictions += pred

    return predictions


def run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name, cross_validation=False, custom=False):
    """Run and evaluate a system using default parameters or cross validation and write predictions to file."""

    predictions = train_and_run_crf_model(train_path, test_path, selected_features, cross_validation, custom)

    if cross_validation:
        print(f"Running {name.replace('_', ' ')} with best parameters")
    else:
        print(f"Running {name.replace('_', ' ')}")

    write_predictions_to_file(test_path, selected_features, predictions, name)

    df = pd.read_csv(test_path, encoding='utf-8', sep='\t', keep_default_na=False,
                     quotechar='\\', skip_blank_lines=False)

    gold_labels = df['gold_label'].to_list()
    evaluate_classifier(predictions, gold_labels, selected_features, name)


def main(paths=None):
    """Run system with full set of features using default parameters
        and/or cross validation"""

    if not paths:  # if no paths are passed to the function
        paths = sys.argv[1:]

    if not paths:  # if no paths are passed to the function through the command line
        paths = [CONFIG['train_path'].replace('.txt', '_features.txt'),
                 CONFIG['dev_path'].replace('.txt', '_features.txt')]

    train_path, test_path = paths

    # use the full set of features
    name = "system3_CRF"
    selected_features = ['lemma', 'prev_lemma', 'next_lemma', 'pos_category', 'is_single_cue', 'has_affix', 'affix',
                         'base_is_word', 'base']
    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name)

    # implement basic cross-validation in combination with the system using all features
    name = "system4_CRF"
    run_and_evaluate_a_crf_system(train_path, test_path, selected_features, name, cross_validation=True)


if __name__ == '__main__':
    main()