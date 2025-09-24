import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- Load and Preprocess Data ---
def train_and_save_model():
    """
    Trains a RandomForestClassifier model on the HR Attrition data
    and saves the model and feature columns to .pkl files.
    """
    try:
        df = pd.read_csv('HR-Employee-Attrition.csv')
    except FileNotFoundError:
        print("Error: The file 'HR-Employee-Attrition.csv' was not found.")
        print("Please ensure 'HR-Employee-Attrition.csv' is in the same directory.")
        return

    # Drop columns with constant values as they don't provide predictive power
    df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18'])

    # Feature Engineering
    df['MonthlyIncomeLPA'] = df['MonthlyIncome'] / 100000
    df = df.drop(columns=['MonthlyIncome']) 

    # Separate features (X) and target (y)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Convert 'Attrition' to binary (Yes=1, No=0)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Automatically identify all categorical columns (object type)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical features for the model
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Train the RandomForestClassifier model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    # Save the trained model and the feature names to .pkl files
    print("Saving the trained model and feature list...")
    joblib.dump(model, 'attrition_model.pkl')
    joblib.dump(X_encoded.columns.tolist(), 'model_features.pkl')

    print("Model training complete!")
    print("Files 'attrition_model.pkl' and 'model_features.pkl' have been created.")

if __name__ == "__main__":
    train_and_save_model()