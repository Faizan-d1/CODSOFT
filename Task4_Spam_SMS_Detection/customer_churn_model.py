# ----------------------------------------
# Task 3: Customer Churn Prediction
# Dataset: Churn_Modelling.csv
# ----------------------------------------

# Step 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------------------
# Step 2: Load Dataset
# ----------------------------------------
data_path = 'Churn_Modelling.csv'  # Update if your path is different
df = pd.read_csv(data_path)

print("Dataset preview:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nTarget distribution (Exited):")
print(df['Exited'].value_counts())

# ----------------------------------------
# Step 3: Preprocessing
# ----------------------------------------

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Encode categorical features
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------
# Step 4: Train Model
# ----------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------
# Step 5: Evaluate Model
# ----------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ----------------------------------------
# Step 6: Save Model
# ----------------------------------------
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'churn_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved successfully at: {model_path}")
