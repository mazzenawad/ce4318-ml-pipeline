from pyexpat import model
from itertools import cycle
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from robustness_check import save_figure_to_output
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_absolute_error

def evaluate_model(model, X_test, y_test, output_dir):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

def evaluate_ordinal_performance(y_true, y_pred, output_dir):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    qwk_str = f"Quadratic Weighted Kappa: {qwk:.4f}"
    print(qwk_str)
    
    mae = mean_absolute_error(y_true, y_pred)
    mae_str = f"Mean Absolute Error: {mae:.4f}"
    print(mae_str)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'ordinal_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("--- Ordinal Evaluation Metrics ---\n\n")
        f.write(f"{qwk_str}\n")
        f.write(f"{mae_str}\n\n")
        f.write("Note: A lower MAE indicates the model's errors are mostly 'off-by-one'.\n")
    
    print(f"Text report saved to: {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure("Ordinal_Confusion_Matrix", figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix: Are errors "off-by-one"?')
    plt.ylabel('True Condition Rating')
    plt.xlabel('Predicted Condition Rating')
    fig = plt.gcf()
    save_figure_to_output(fig, 'confusion_matrix2.png') 
    plt.show()
    plt.close()

def ROC_AUC_multiclass(model, X_test, y_test, output_dir):
    y_pred_proba = model.predict_proba(X_test)
    roc_auc = roc_auc_score(
        y_test, 
        y_pred_proba, 
        multi_class="ovr", 
        average="weighted"
    )
    roc_str = f"Weighted ROC-AUC Score: {roc_auc:.4f}"
    print(roc_str)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'roc_auc_report.txt'), 'w') as f:
        f.write(roc_str + "\n")
def plot_multiclass_roc_curve(model, X_test, y_test, output_dir):
    y_pred_proba = model.predict_proba(X_test)
    classes = model.classes_
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    fpr = dict()
    tpr = dict()
    class_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        class_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure("Multiclass_ROC_Curve", figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'darkorange', 'purple'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Rating {classes[i]} ROC (area = {class_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve: Underground Pipe Condition Prediction')
    plt.legend(loc="lower right")
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.gcf()
    save_figure_to_output(fig, 'multiclass_roc_curve.png')
    plt.show()
    plt.close()