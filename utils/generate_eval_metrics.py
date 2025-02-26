
from matplotlib import pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

# Create a function that will automatically provide the eval metrics spcified below for a model

def generate_eval_metrics(actual, predicted):
    
    # Confusion Matrix

    actual = y_test
    predicted = y_pred

    cm_result  = confusion_matrix(actual, predicted)

    custom_labels = ['Malignant', 'Benign'] # M (Malignant) = Unlabelled Instance (0), B (Benign) = Positive Instance (1)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_result, display_labels = custom_labels)

    cm_display.plot()

    print("Confusion Matrix:")
    plt.show()

    # Accuracy
    accuracy = accuracy_score(actual, predicted)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Precision
    precision = precision_score(actual, predicted)
    print(f"Precision: {precision:.2f}")

    # Recall
    recall = recall_score(actual, predicted)
    print(f"Recall: {recall:.2f}")

    # F1 Score
    f1 = f1_score(actual, predicted)
    print(f"F1 Score: {f1:.2f}")