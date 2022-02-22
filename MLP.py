import gensim
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV

# get inspired by the machine learning course
def extract_word_embedding(token, word_embedding_model):
 
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def combine_embeddings(data, word_embedding_model):
   
 
    embeddings = []
    for token, prev_token, next_token, lemma in zip(data['token'], data['prev_token'], data['next_token'], data['lemma']):

        # Extract embeddings for all token features
        token_vector = extract_word_embedding(token, word_embedding_model)
        prev_token_vector = extract_word_embedding(prev_token, word_embedding_model)
        next_token_vector = extract_word_embedding(next_token, word_embedding_model)
        lemma_vector = extract_word_embedding(lemma, word_embedding_model)

        # Concatenate the embeddings
        embeddings.append(np.concatenate((token_vector, prev_token_vector, next_token_vector, lemma_vector)))

    return embeddings


def make_sparse_features(data, feature_names):
    """
    Transforms traditional features into one-hot-encoded vectors
    :param data: a pandas dataframe
    :param feature_names: a list containing the header names of the traditional features
    :type data: pandas.core.frame.DataFrame
    :type feature_names: list
    :returns a vector representation of the traditional features
    :rtype: list
    """

    sparse_features = []
    for i in range(len(data)):

        # Prepare feature dictionary for each sample
        feature_dict = defaultdict(str)

        # Add feature values to dictionary
        for feature in feature_names:
            value = data[feature][i]
            feature_dict[feature] = value

        # Append all sample feature dictionaries
        sparse_features.append(feature_dict)

    return sparse_features


def combine_features(sparse, dense):
    """
    Combines sparse (one-hot-encoded) and dense (e.g. word embeddings) features
    into a combined feature set.
    :param sparse: one-hot representations of the traditional features
    :param dense: word embeddings of the token features
    :type sparse: list
    :type dense: list
    :returns a vector representation of all features combined
    :rtype: list
    """
    # Prepare containers
    combined_vectors = []
    sparse = np.array(sparse.toarray())

    # Concatanate vectors for each sample
    for index, vector in enumerate(sparse):
        combined_vector = np.concatenate((vector, dense[index]))
        combined_vectors.append(combined_vector)

    return combined_vectors


def train_classifier(x_train, y_train):
  
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=500, random_state=1)
    clf.fit(x_train, y_train)
    return clf


def load_data_embeddings(training_data_path, test_data_path, embedding_model_path):

    print('Loading word embedding model...')
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_model_path, binary=True)
    print('Done loading word embedding model')

    training = pd.read_csv(training_data_path, encoding='utf-8', sep='\t')
    training_labels = training['gold_label']

    test = pd.read_csv(test_data_path, encoding='utf-8', sep='\t')
    test_labels = test['gold_label']

    return training, training_labels, test, test_labels, embedding_model

def run_classifier(training, training_labels, test, word_embedding_model, sparse):
   
    # Extract embeddings for token, prev_token and next_token
    embeddings = combine_embeddings(training, word_embedding_model)

    # Extract and vectorize one-hot features
    sparse_features = make_sparse_features(training, sparse)
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(sparse_features)

    # Combine both kind of features into training data
    training_data = combine_features(sparse_vectors, embeddings)

    # Train network
    print("Training classifier...")
    clf = train_classifier(training_data, training_labels)
    print("Done training classifier")

    # Extract embeddings for token, prev_token and next_token from test data
    embeddings = combine_embeddings(test, word_embedding_model)

    # Extract and vectorize one-hot features for test data
    sparse_features = make_sparse_features(test, sparse)
    sparse_vectors = vec.transform(sparse_features)

    test_data = combine_features(sparse_vectors, embeddings)

    return clf, test_data


def evaluation(test_labels, prediction):
    """
    Function to print f-score, precision and recall for each class of test data.
    Also prints confusion matrix
    :param test_labels: the test labels from the test set
    :param prediction: the prediction of the trained classifier on the test set
    :type test_labels: list
    :type prediction: list
    """
    metrics = classification_report(test_labels, prediction, digits=3)
    print(metrics)

    # Confusion matrix
    data = {'Gold': test_labels, 'Predicted': prediction}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    print()




    # Train classifiers
    for features in sparse:
        clf, test_data = run_classifier(training, training_labels, test, word_embedding_model, features)

        # Make prediction
        prediction = clf.predict(test_data)

        # Print evaluation
        print('-------------------------------------------------------')
        print("Evaluation of MLP system with the following sparse features:")
        print(features)
        evaluation(test_labels, prediction)


if __name__ == '__main__':
    main()