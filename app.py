from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

try:
    model = joblib.load("student_performance_pipeline.joblib")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
all_features = {
    'Hours_Studied': 19.975329,
    'Attendance': 79.977448,
    'Parental_Involvement': 1.086423,
    'Access_to_Resources': 1.100197,
    'Extracurricular_Activities': 0.596035,
    'Sleep_Hours': 7.029060,
    'Previous_Scores': 75.070531,
    'Motivation_Level': 0.906463,
    'Internet_Access': 0.924474,
    'Tutoring_Sessions': 1.493719,
    'Family_Income': 0.787649,
    'Teacher_Quality': 1.195247,
    'School_Type': 0.304071,
    'Physical_Activity': 2.967610,
    'Learning_Disabilities': 0.105191,
    'Parental_Education_Level': 0.696080,
    'Distance_from_Home': 0.501589,
    'Gender': 0.422733,
    'Exam_Score': 13.447132,
    'Peer_Negative': 0.208415,
    'Peer_Neutral': 0.392311,
    'Peer_Positive': 0.399273,
    'Study_Attendance': 1596.890571,
    'Sleep_Efficiency': 0.374639,
    'Physical_Impact': 20.858937
}
app = Flask(__name__)


@app.route('/predict')
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500




    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)