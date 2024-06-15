from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Ensure the correct path to the model and encoders
model_path = 'churn_model.pkl'
label_encoders_path = 'label_encoders.pkl'

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

# Check if the label encoders file exists
if os.path.exists(label_encoders_path):
    label_encoders = joblib.load(label_encoders_path)
else:
    raise FileNotFoundError(f"Label encoders file '{label_encoders_path}' not found.")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features', {})  # Get features dictionary from JSON data
        
        # Ensure feature order for consistent transformation
        ordered_features = ['tenure', 'Contract', 'TotalCharges']
        
        # Prepare feature array for prediction
        feature_array = []
        for feature in ordered_features:
            feature_array.append(features.get(feature, None))
        
        # Apply label encoding to categorical features
        for column, encoder in label_encoders.items():
            feature_array[ordered_features.index(column)] = encoder.transform([feature_array[ordered_features.index(column)]])[0]
        
        # Convert to numpy array for prediction
        features_np = np.array([feature_array])

        # Predict using the model
        prediction = model.predict(features_np)

        # Return the prediction as JSON response
        return jsonify({'churn_prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
