import joblib
import numpy as np
import pandas as pd

def preprocess_input(inputs, feature_names, scaler):
    """
    Preprocess user inputs using the scaler and align them with the feature set.
    """
    # Create a DataFrame to align input features with the training set
    input_df = pd.DataFrame([inputs], columns=feature_names)

    # Fill missing features with 0 (e.g., one-hot encoded categories not present in input)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale numerical features
    scaled_inputs = scaler.transform(input_df)
    return scaled_inputs

def load_model_and_scaler():
    """Load the saved model, scaler, and feature names."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.txt", "r") as f:
        feature_names = f.read().splitlines()
    return model, scaler, feature_names

def main():
    # Load model, scaler, and feature names
    model, scaler, feature_names = load_model_and_scaler()

    # Prompt user for input
    print("Enter the following details for prediction:")
    age = float(input("Age: "))
    sex = int(input("Sex (0 = Female, 1 = Male): "))
    cp = int(input("Chest Pain Type (0-3): "))
    trestbps = float(input("Resting Blood Pressure (in mm Hg): "))
    chol = float(input("Cholesterol Level (mg/dl): "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes): "))
    restecg = int(input("Resting ECG Results (0-2): "))
    thalach = float(input("Maximum Heart Rate Achieved: "))
    exang = int(input("Exercise Induced Angina (0 = No, 1 = Yes): "))
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = int(input("Slope of the Peak Exercise ST Segment (0-2): "))
    ca = int(input("Number of Major Vessels (0-3): "))
    thal = int(input("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect): "))

    # User inputs mapped to feature names
    user_inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Preprocess inputs
    processed_inputs = preprocess_input(user_inputs, feature_names, scaler)

    # Make prediction
    prediction = model.predict(processed_inputs)
    prediction_proba = model.predict_proba(processed_inputs)

    # Display the result
    if prediction[0] == 1:
        print(f"The model predicts: High risk of heart disease (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        print(f"The model predicts: Low risk of heart disease (Confidence: {prediction_proba[0][0]*100:.2f}%)")

if __name__ == "__main__":
    main()
