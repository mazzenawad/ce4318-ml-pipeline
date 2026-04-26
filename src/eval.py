from sklearn.metrics import classification_report, accuracy_score
import os

def evaluate_model(model, X_test, y_test, output_dir):
    """Generates evaluation metrics and saves them to a text file."""
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