from typing import Dict

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from os import mkdir
from os.path import isdir

from pathlib import Path


def create_performance_scenario_dict(actual, predicted_probabilities: np.array) -> Dict[np.array, np.array]:
    """
    This function creates a performance scenario dictionary given an array of actual values and model
    predicted probability values.
    Returns:

    """
    return {
        'actual': np.array(actual),
        'predicted_probabilities': np.array(predicted_probabilities)
    }


def random_model() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities where there are true positives, false positives,
    true negatives, and false negatives, all of equal proportion. This represents a fully random model.
    Returns: dictionary containing the model result.

    """
    a = [0, 1, 0, 1]
    p = [1.0, 0.0, 0.0, 1.0]

    # The line below returns a dictionary of "actual" and "predicted" values, representing a perfect model.
    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def gradient_confidence_perfect_model() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities that is perfect (no false positives or false
    negatives). The model has a gradient about the predictions but is always > 0.5 in the positive case and < 0.5 in
    the negative case.
    Returns: dictionary containing the model result.

    """

    a = ([1] * 5) + ([0] * 5)
    p = [0.6, 0.7, 0.8, 0.9, 1] + [0.0, 0.1, 0.2, 0.3, 0.4]

    # The line below returns a dictionary of "actual" and "predicted" values, representing a perfect model.
    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def wild_roc_curve() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities that approximates something one might find in
    the wild.
    Returns: dictionary containing the model result.

    """

    a = ([1] * 100) + ([0] * 100)
    p = list(np.concatenate(
        [np.random.normal(loc=0.9, scale=1.0, size=100),
         np.random.normal(loc=0.1, scale=1.0, size=100)], axis=0)
    )

    # The line below returns a dictionary of "actual" and "predicted" values, representing a perfect model.
    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def full_confidence_found_a_needle_in_the_haystack() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities that represents a model being able to find the
    single needle in the haystack.
    Returns: dictionary containing the model result.

    """
    a = [1] + ([0] * 99)
    p = [1.0] + ([0.0] * 99)

    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def full_confidence_found_all_the_hay_but_lost_the_needle_in_the_haystack() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities that represents a model that is unable to find
    the single needle in the haystack
    Returns: dictionary containing the model result.

    """
    a = [1] + ([0] * 99)
    p = [0.0] + ([0.0] * 99)

    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def full_confidence_mistaken_hay_for_needle() -> Dict[np.array, np.array]:
    """
    This function creates an example actual and predicted_probabilities that represents a model that mistakes the
    hay for the needle and vice versa.
    Returns: dictionary containing the model result.

    """
    a = [1] + ([0] * 99)
    p = [0.0] + ([1.0] * 99)

    return create_performance_scenario_dict(actual=a, predicted_probabilities=p)


def plot_roc_curve(experiment_name: str, data: Dict[np.array, np.array], report_path: Path) -> None:
    fpr, tpr, _ = roc_curve(data['actual'], data['predicted_probabilities'], drop_intermediate=False)
    # Generate the area under the curve
    try:
        # Try catch except to catch when only one class is present in the actual labels. AUC cannot be computed
        auc = roc_auc_score(data['actual'], data['predicted_probabilities'])
    except ValueError:
        auc = 'Undefined (only one class present in truth values)'
    # Plot the ROC curve and add the label for the area under the curve.
    plt.plot(fpr, tpr, color='red', label="auc=" + str(auc), marker='.')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Random Model')
    plt.xlabel("False Positives / (False Positives + True Negatives) = False Positive Rate")
    plt.ylabel("True Positives / (True Positives + False Negatives) = True Positive Rate")
    plt.legend(loc=4)
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    # Save the figure
    plt.savefig(Path(report_path, f'roc_curve_{experiment_name}.png'))
    # Clear the figure
    plt.clf()
    # Close the plot
    plt.close()


def plot_precision_recall_curve(experiment_name: str, data: Dict[np.array, np.array], report_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(data['actual'], data['predicted_probabilities'])
    # Generate the area under the curve
    try:
        # Try catch except to catch when only one class is present in the actual labels. AUC cannot be computed
        aps = average_precision_score(data['actual'], data['predicted_probabilities'])
    except ValueError:
        aps = 'Undefined (only one class present in truth values)'
    # Plot the ROC curve and add the label for the area under the curve.
    plt.plot(recall, precision, color='red', label=f"average precision score = {aps}", marker='.')
    random_model = len(data['actual'][data['actual'] == 1]) / len(data['actual'])
    plt.plot([0, 1], [random_model, random_model], linestyle='--', color='darkblue', label='Random Model')
    plt.xlabel('True Positives / (True Positives + False Positives) = Recall')
    plt.ylabel('True Positives / (True Positives + False Negatives) = Precision')
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    plt.legend(loc=4)
    # Save the figure
    plt.savefig(Path(report_path, f'precision_recall_curve_{experiment_name}.png'))
    # Clear the figure
    plt.clf()
    # Close the plot
    plt.close()


def main():
    # Create the different scenarios to analyze the ROC curves for.
    rm = random_model()
    gradient_fcpmr = gradient_confidence_perfect_model()
    wroc = wild_roc_curve()
    fanith = full_confidence_found_a_needle_in_the_haystack()
    fathbltnith = full_confidence_found_all_the_hay_but_lost_the_needle_in_the_haystack()
    mhfn = full_confidence_mistaken_hay_for_needle()

    # Create the report directory
    report_path = Path('reports')
    if not isdir(report_path):
        mkdir(report_path)

    # Generate graphs for each of the ROC scenarios
    for key, value in {'random_model': rm,
                       'gradient_confidence_perfect_model': gradient_fcpmr,
                       'wild_roc_curve': wroc,
                       'full_confidence_found_a_needle_in_the_haystack': fanith,
                       'full_confidence_found_all_the_hay_but_lost_the_needle_in_the_haystack': fathbltnith,
                       'full_confidence_mistaken_hay_for_needle': mhfn
                       }.items():
        # Generate the roc curve.
        plot_roc_curve(experiment_name=key, data=value, report_path=report_path)
        # Generate the precision recall curve.
        plot_precision_recall_curve(experiment_name=key, data=value, report_path=report_path)


if __name__ == '__main__':
    main()
