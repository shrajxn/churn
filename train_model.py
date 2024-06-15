import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Load data
file_path = r'C:\Users\Shrajan\OneDrive\Desktop\flask_ml_project\churn\Churn-Data.csv'
  # Adjust if your dataset is in a different location
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Drop 'cID' column if it exists
if 'cID' in df.columns:
    df.drop('cID', axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Encode target variable
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model
model_path = 'churn_model.pkl'  # Adjust as needed
joblib.dump(model, model_path)

# Save the label encoders
label_encoders_path = 'label_encoders.pkl'  # Adjust as needed
joblib.dump(label_encoders, label_encoders_path)
