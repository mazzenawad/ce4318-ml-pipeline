from sklearn.metrics import classification_report, accuracy_score
import os
from robustness_check import save_figure_to_output
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_absolute_error

def evaluate_model(model, X_test, y_test, output_dir):
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    # Print to console
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

def evaluate_ordinal_performance(y_true, y_pred, output_dir):
    # 1. Quadratic Weighted Kappa (QWK)
    # 1.0 is perfect agreement, 0.0 is random chance. Penalizes big misses heavily.
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    qwk_str = f"Quadratic Weighted Kappa: {qwk:.4f}"
    print(qwk_str)
    
    # 2. Mean Absolute Error (MAE)
    # Tells you the average "off-by-X" error (e.g., MAE of 0.8 means it usually guesses within 1 rating level)
    mae = mean_absolute_error(y_true, y_pred)
    mae_str = f"Mean Absolute Error: {mae:.4f}"
    print(mae_str)

    # --- SAVE TEXT REPORT ---
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'ordinal_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("--- Ordinal Evaluation Metrics ---\n\n")
        f.write(f"{qwk_str}\n")
        f.write(f"{mae_str}\n\n")
        f.write("Note: A lower MAE indicates the model's errors are mostly 'off-by-one'.\n")
    
    print(f"Text report saved to: {report_path}")

    # 3. The Confusion Matrix (Visual Check)
    cm = confusion_matrix(y_true, y_pred)
    
    # Give the figure a unique ID to prevent blank-figure overlapping issues
    plt.figure("Ordinal_Confusion_Matrix", figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix: Are errors "off-by-one"?')
    plt.ylabel('True Condition Rating')
    plt.xlabel('Predicted Condition Rating')
    
    # Save the figure using your helper function
    fig = plt.gcf()
    save_figure_to_output(fig, 'confusion_matrix2.png') 
    
    plt.show()
    plt.close() 