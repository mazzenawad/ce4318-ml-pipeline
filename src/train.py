import xgboost as xgb
import joblib
import os

def train_model(X_train, y_train, random_seed=42):
    """Trains an XGBoost Classifier on the provided data."""
    # Instantiating basic parameters (Hyperparameter tuning could be added here)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=random_seed
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    return model

def save_model(model, preprocessor, output_path):
    """Saves the trained model and preprocessor as a combined pipeline if desired, or separately."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({'model': model, 'preprocessor': preprocessor}, output_path)
    print(f"Model saved to {output_path}")