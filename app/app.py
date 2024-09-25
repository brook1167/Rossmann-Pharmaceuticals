import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('../models/model_25-09-2024-13-07-08-459997.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Ensure that the input data is in the correct format
        input_data = data['input']  # Adjust this according to your model input format
        
        # Make a prediction
        prediction = model.predict([input_data])  # Wrap in a list if a single sample is provided

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
