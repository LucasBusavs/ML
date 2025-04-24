from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import make_scorer
from tabulate import tabulate


def pipeline_score(*args, weights=None, verbosity=False):
    """
    Function to calculate a custom score for ML models.
    The score is a weighted sum of accuracy, precision, recall, and F-score.
    The weights for each metric can be customized using the 'weights' parameter.

    The default weights are:
    - accuracy: 1
    - precision: 1
    - recall: 1
    - f_score: 1

    Can be called in two ways:
    1. pipeline_score(y_true, y_pred)  -> For manual evaluations
    2. pipeline_score(estimator, X, y) -> For GridSearchCV and RandomSearchCV

    Parameters
    ----------
    args : tuple
        - (y_true, y_pred): True labels and predicted labels.
        - (estimator, X, y): Estimator, feature set, and true labels.
    weights : dict, optional
        Custom weights for accuracy, precision, recall, and F-score.
    verbosity : bool, optional
        If True, prints the detailed metrics for each class.

    Returns
    -------
    score: float
        The calculated score based on the provided weights and metrics.
    """

    # GridSearchCV & RandomSearchCV
    if len(args) == 3:
        estimator, X, y_true = args
        y_pred = estimator.predict(X)
    elif len(args) == 2:
        y_true, y_pred = args
    else:
        raise ValueError(
            "Invalid number of arguments. Expected 2 or 3 arguments."
            " (y_true, y_pred) or (estimator, X, y_true)."
        )

    default_weights = {
        'accuracy': 1,
        'precision': 1,
        'f_score': 1,
        'recall': 1,
    }
    beta = 1

    if weights:
        default_weights.update(weights)
        if (default_weights['precision'] > default_weights['recall']):
            beta = 2
        elif (default_weights['precision'] < default_weights['recall']):
            beta = 0.5

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, beta=beta, average=None, zero_division=0
    )

    # Print metrics if verbosity is enabled
    if verbosity:
        print(f"Accuracy: {accuracy:.2f}\n")

        table_data = [
            [f"Class {i}", f"{precision[i]:.2f}",
                f"{recall[i]:.2f}", f"{fscore[i]:.2f}", support[i]]
            for i in range(len(precision))
        ]
        headers = ["Class", "Precision", "Recall", "F-score", "Support"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    score = default_weights['accuracy'] * accuracy
    for i in range(len(precision)):
        score += default_weights['precision'] * precision[i]
        score += default_weights['recall'] * recall[i]
        score += default_weights['f_score'] * fscore[i]
    return float(score)


def pipeline_score_scorer(weights=None, verbosity=False):
    """
    Function to create a custom scorer for use with GridSearchCV or RandomizedSearchCV.

    Parameters
    ----------
    weights : dict, optional
        Custom weights for accuracy, precision, recall, and F-score.
    verbosity : bool, optional
        If True, prints the detailed metrics for each class.

    Returns
    -------
    scorer: callable
        A custom scoring function that can be used with GridSearchCV or RandomizedSearchCV.
    """
    return make_scorer(pipeline_score, weights=weights, verbosity=verbosity)
