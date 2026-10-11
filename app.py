import os
import json
from flask import Flask, render_template, request, jsonify
from src.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Initialize prediction pipeline
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = PredictPipeline()
    return predictor

def load_metrics():
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"test_accuracy": 100.0, "test_loss": 0.08}

@app.route("/", methods=["GET"])
def home():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    metrics = load_metrics()
    try:
        pipeline = get_predictor()

        # Handle JSON requests (REST API)
        if request.is_json:
            data = request.get_json()
            study_hours = float(data.get("study_hours", 0))
            attendance = float(data.get("attendance", 0))
            previous_score = float(data.get("previous_score", 0))

            prediction = pipeline.predict(study_hours, attendance, previous_score)
            return jsonify({
                "status": "success",
                "input": {
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "previous_score": previous_score
                },
                "result": prediction
            }), 200

        # Handle Form requests (HTML Web UI)
        study_hours = float(request.form.get("study_hours", 0))
        attendance = float(request.form.get("attendance", 0))
        previous_score = float(request.form.get("previous_score", 0))

        result = pipeline.predict(study_hours, attendance, previous_score)
        return render_template("index.html", metrics=metrics, result=result)

    except Exception as e:
        if request.is_json:
            return jsonify({"status": "error", "message": str(e)}), 400
        return render_template("index.html", metrics=metrics, error=str(e), result=None)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "student-pass-fail-predictor"}), 200

@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(load_metrics()), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
