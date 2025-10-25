from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

try:
    model = joblib.load("student_performance_pipeline6.joblib")
    print("✅ Model loaded successfully!")

    
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None



app = Flask(__name__)
@app.route('/')
def home():
    
    return render_template('index.html')


@app.route('/predict' , methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        features_list = data.get('features')

        if not features_list or len(features_list) != 6:
            return jsonify({'error': 'Invalid input. Expected 6 features.'}), 400
        

        features = np.array(features_list, dtype=float)
        
        features_2d = features.reshape(1, -1)

        prediction = model.predict(features_2d)
        prediction_value = prediction[0]
        if prediction_value > 20.0:
            prediction_value = 20.0
        if prediction_value < 0.0:
            prediction_value = 0.0

        return jsonify({'prediction': prediction_value})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500


if __name__ == "__main__":
    app.run(debug=True)