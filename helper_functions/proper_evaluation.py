import csv
import sys
import os
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix


def read_file(file):
    df = pd.read_csv(file, sep=',', encoding="utf-8",
                     keep_default_na=False, quotechar='|', skip_blank_lines=False)
    predictions = df['predict']
    gold_labels = df['gold']

    return gold_labels, predictions


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


def evaluate_classifier(file, name):
    """Produce full evaluation of classifier."""
    gold_labels, predictions = read_file(file)
    print(f"Evaluating {name.replace('_', ' ')}: ")
    print()
    cf_matrix = generate_confusion_matrix(predictions, gold_labels)
    report = calculate_precision_recall_f1_score(predictions, gold_labels)

    print(cf_matrix)
    # print(cf_matrix.to_latex())  # print and paste to Overleaf

    print(report)
    # print(report.to_latex())  # print and paste to Overleaf
    cf_matrix.to_csv(f"evaluations/{name}.csv")


def run_evaluation(str_path: str):
    name = os.path.basename(str_path).replace('.csv', '')
    evaluate_classifier(str_path, name)


if __name__ == '__main__':
    selected_files = ['../output/arg_classification.csv', '../output/pred_identification.csv',
                      '../output/arg_identification.csv']
    run_evaluation(selected_files[0])
