from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate

def pipeline_score(y_true, y_pred, weights=None):

    default_weights = {
        'accuracy': 1,
        'precision': 1,
        'f_score': 1,
        'recall': 1,
    }
    beta = 1

    if weights:
        default_weights.update(weights)
        if(default_weights['precision'] > default_weights['recall']):
            beta = 2
        elif(default_weights['precision'] < default_weights['recall']):
            beta = 0.5

    # Calcula as métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print(f"Accuracy: {accuracy:.2f}\n")

    table_data = [
        [f"Classe {i}", f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{fscore[i]:.2f}", support[i]]
        for i in range(len(precision))
    ]
    headers = ["Classe", "Precision", "Recall", "F-score", "Support"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    score = default_weights['accuracy'] * accuracy
    for i in range(len(precision)):
        score += default_weights['precision'] * precision[i]
        score += default_weights['recall'] * recall[i]
        score += default_weights['f_score'] * fscore[i]
    return score