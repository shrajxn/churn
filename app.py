from flask import Flask, request, jsonify, render_template
import joblib as jb
import numpy as np
import Constants

app = Flask(__name__)
model_path = Constants.model_path
label_encoders_path = Constants.label_encoders_path

model = jb.load(model_path)
label_encoders = jb.load(label_encoders_path)

@app.route('/')
def index():
    return render_template('index.html', message=None)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            tenure = request.form.get('tenure')
            contract = request.form.get('contract')
            total_charges = request.form.get('total_charges')

            ordered_features = ['tenure', 'Contract', 'TotalCharges']

            feature_array = []
            features = {'tenure': tenure, 'Contract': contract, 'TotalCharges': total_charges}
            for feature in ordered_features:
                feature_array.append(features.get(feature, None))
            for column, encoder in label_encoders.items():
                feature_array[ordered_features.index(column)] = encoder.transform([feature_array[ordered_features.index(column)]])[0]

            features_np = np.array([feature_array])

            prediction = model.predict(features_np)

            return render_template('predict.html', message=f'Churn Prediction: {int(prediction[0])}')
        except Exception as e:
            return render_template('predict.html', message=f'Error: {str(e)}'), 400
    else:
        return render_template('predict.html', message=None)

if __name__ == '__main__':
    app.run(debug=True)


