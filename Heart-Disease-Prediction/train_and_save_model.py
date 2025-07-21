import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv("dataset.csv")

# Define features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

# Save model, scaler, and feature names
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("feature_names.txt", "w") as f:
    f.write("\n".join(X.columns))
