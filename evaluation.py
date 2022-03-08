import sklearn

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