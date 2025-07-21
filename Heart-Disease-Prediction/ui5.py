
import tkinter as tk
from tkinter import messagebox
from matplotlib import pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os


def preprocess_input(inputs, feature_names, scaler):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    scaled_inputs = scaler.transform(input_df)
    return scaled_inputs


def load_model_and_scaler():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.txt", "r") as f:
        feature_names = f.read().splitlines()
    return model, scaler, feature_names


def show_graph(prediction_proba):
    categories = ["Low Risk", "High Risk"]
    probabilities = [prediction_proba[0][0], prediction_proba[0][1]]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, probabilities, color=["green", "red"])
    plt.title("Heart Disease Prediction Probabilities")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def predict_heart_disease():
    try:
        age = float(entry_age.get())
        sex = int(var_sex.get())
        cp = int(var_cp.get())
        trestbps = float(entry_trestbps.get())
        chol = float(entry_chol.get())
        fbs = int(var_fbs.get())
        restecg = int(var_restecg.get())
        thalach = float(entry_thalach.get())
        exang = int(var_exang.get())
        oldpeak = float(entry_oldpeak.get())
        slope = int(var_slope.get())
        ca = int(entry_ca.get())
        thal = int(var_thal.get())

        model, scaler, feature_names = load_model_and_scaler()
        user_inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        processed_inputs = preprocess_input(user_inputs, feature_names, scaler)

        prediction = model.predict(processed_inputs)
        prediction_proba = model.predict_proba(processed_inputs)

        if prediction[0] == 1:
            messagebox.showinfo("Prediction Result",
                                f"High risk of heart disease (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        else:
            messagebox.showinfo("Prediction Result",
                                f"Low risk of heart disease (Confidence: {prediction_proba[0][0]*100:.2f}%)")

        show_graph(prediction_proba)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def clear_inputs():
    entry_age.delete(0, tk.END)
    var_sex.set("0")
    var_cp.set("0")
    entry_trestbps.delete(0, tk.END)
    entry_chol.delete(0, tk.END)
    var_fbs.set("0")
    var_restecg.set("0")
    entry_thalach.delete(0, tk.END)
    var_exang.set("0")
    entry_oldpeak.delete(0, tk.END)
    var_slope.set("0")
    entry_ca.delete(0, tk.END)
    var_thal.set("0")


def save_inputs():
    try:
        user_data = {
            "Age": entry_age.get(),
            "Sex": var_sex.get(),
            "ChestPainType": var_cp.get(),
            "RestingBP": entry_trestbps.get(),
            "Cholesterol": entry_chol.get(),
            "FastingBS": var_fbs.get(),
            "RestingECG": var_restecg.get(),
            "MaxHR": entry_thalach.get(),
            "ExerciseAngina": var_exang.get(),
            "Oldpeak": entry_oldpeak.get(),
            "Slope": var_slope.get(),
            "CA": entry_ca.get(),
            "Thal": var_thal.get(),
        }
        df = pd.DataFrame([user_data])
        df.to_csv("user_inputs.csv", mode='a', index=False, header=not os.path.exists("user_inputs.csv"))
        messagebox.showinfo("Success", "Inputs saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Could not save inputs: {str(e)}")


root = tk.Tk()
root.title("Heart Disease Prediction")

tk.Label(root, text="Age:").grid(row=0, column=0, sticky="e")
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sex (0=Female, 1=Male):").grid(row=1, column=0, sticky="e")
var_sex = tk.StringVar(value="0")
tk.OptionMenu(root, var_sex, "0", "1").grid(row=1, column=1)

tk.Label(root, text="Chest Pain Type (0-3):").grid(row=2, column=0, sticky="e")
var_cp = tk.StringVar(value="0")
tk.OptionMenu(root, var_cp, "0", "1", "2", "3").grid(row=2, column=1)

tk.Label(root, text="Resting Blood Pressure:").grid(row=3, column=0, sticky="e")
entry_trestbps = tk.Entry(root)
entry_trestbps.grid(row=3, column=1)

tk.Label(root, text="Cholesterol Level:").grid(row=4, column=0, sticky="e")
entry_chol = tk.Entry(root)
entry_chol.grid(row=4, column=1)

tk.Label(root, text="Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes):").grid(row=5, column=0, sticky="e")
var_fbs = tk.StringVar(value="0")
tk.OptionMenu(root, var_fbs, "0", "1").grid(row=5, column=1)

tk.Label(root, text="Resting ECG Results (0-2):").grid(row=6, column=0, sticky="e")
var_restecg = tk.StringVar(value="0")
tk.OptionMenu(root, var_restecg, "0", "1", "2").grid(row=6, column=1)

tk.Label(root, text="Maximum Heart Rate Achieved:").grid(row=7, column=0, sticky="e")
entry_thalach = tk.Entry(root)
entry_thalach.grid(row=7, column=1)

tk.Label(root, text="Exercise Induced Angina (0=No, 1=Yes):").grid(row=8, column=0, sticky="e")
var_exang = tk.StringVar(value="0")
tk.OptionMenu(root, var_exang, "0", "1").grid(row=8, column=1)

tk.Label(root, text="ST Depression Induced by Exercise:").grid(row=9, column=0, sticky="e")
entry_oldpeak = tk.Entry(root)
entry_oldpeak.grid(row=9, column=1)

tk.Label(root, text="Slope of the Peak Exercise ST Segment (0-2):").grid(row=10, column=0, sticky="e")
var_slope = tk.StringVar(value="0")
tk.OptionMenu(root, var_slope, "0", "1", "2").grid(row=10, column=1)

tk.Label(root, text="Number of Major Vessels (0-3):").grid(row=11, column=0, sticky="e")
entry_ca = tk.Entry(root)
entry_ca.grid(row=11, column=1)

tk.Label(root, text="Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect):").grid(row=12, column=0, sticky="e")
var_thal = tk.StringVar(value="0")
tk.OptionMenu(root, var_thal, "0", "1", "2").grid(row=12, column=1)

tk.Button(root, text="Predict", command=predict_heart_disease).grid(row=13, column=0)
tk.Button(root, text="Clear", command=clear_inputs).grid(row=13, column=1)
tk.Button(root, text="Save", command=save_inputs).grid(row=13, column=2)

root.mainloop()
