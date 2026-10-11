# Student Pass/Fail Prediction - MLOps Pipeline 🎓🚀

An end-to-end Machine Learning Operations (MLOps) project implementing a **Neural Network (Multi-Layer Perceptron)** for predicting student pass/fail outcomes, complete with automated CI/CD and deployment on **Render** via **GitHub**.

---

## 🌟 MLOps Architecture

```
                                [ students.csv ]
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │  Step 1: Data Ingestion & Validation  │
                   │         (src/data_ingestion.py)       │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Step 2: Transformation & Normalization│
                   │       (src/data_transformation.py)    │
                   │         → outputs: scaler.pkl         │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    Step 3: Neural Network Training    │
                   │          (src/model_trainer.py)       │
                   │      3 -> 8 (ReLU) -> 1 (Sigmoid)     │
                   │   → outputs: model.keras, metrics.json│
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    Step 4: Continuous Integration     │
                   │      (.github/workflows/mlops_ci.yml) │
                   │      Automated Tests (tests/)         │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Step 5: Web Serving & Cloud Deployment│
                   │        (Flask / Gunicorn on Render)   │
                   │        Real-time Web UI & REST API    │
                   └───────────────────────────────────────┘
```

---

## 📁 Project Structure

```
student_pass_fail_prediction/
│
├── .github/
│   └── workflows/
│       └── mlops_ci.yml           # CI/CD: Automated Pipeline Run & Tests
├── data/
│   └── raw/
│       └── students.csv           # Raw student dataset
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py          # Data ingestion & schema boundary validation
│   ├── data_transformation.py     # StandardScaler normalization & train/test split
│   ├── model_trainer.py           # Keras Neural Network builder, training & metrics
│   └── predict_pipeline.py        # Dual-engine inference pipeline (Keras + lightweight weights)
├── artifacts/
│   ├── scaler.pkl                 # Fitted StandardScaler artifact
│   ├── model.keras                # Trained Keras Neural Network model
│   ├── weights.npz                # Saved network weights (Render memory-optimized)
│   ├── metrics.json               # Tracked model evaluation metrics
│   ├── train.csv                  # Ingested training split (80%)
│   └── test.csv                   # Ingested testing split (20%)
├── templates/
│   └── index.html                 # Responsive web UI for live testing
├── tests/
│   └── test_pipeline.py           # Automated unit & integration tests
├── app.py                         # Production Flask web application (Web UI + REST API)
├── run_pipeline.py                # MLOps pipeline orchestrator
├── render.yaml                    # Render Blueprint configuration
├── Procfile                       # Gunicorn web process definition for Render
├── runtime.txt                    # Python 3.10 runtime declaration
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Complete project documentation
```

---

## 🧠 Neural Network Model Summary

- **Input Features (3):** `StudyHours`, `Attendance`, `PreviousScore`
- **Hidden Layer (1):** 8 Neurons, `ReLU` activation
- **Output Layer (1):** 1 Neuron, `Sigmoid` activation (Outputs Pass probability between 0 and 1)
- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `Adam` (learning rate = 0.01)
- **Epochs:** 100
- **Test Accuracy:** 100.0%

---

## 💻 Local Execution

### 1. Run the Entire MLOps Pipeline
Executes Ingestion $\rightarrow$ Transformation $\rightarrow$ Training $\rightarrow$ Evaluation:
```bash
python run_pipeline.py
```

### 2. Run Automated Tests
```bash
python -m unittest discover tests -v
```

### 3. Start the Web App Locally
```bash
python app.py
```
Open your browser at `http://localhost:5000` to interact with the prediction UI.

### 4. REST API Endpoint
You can also send POST requests to `/predict`:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"study_hours": 7.0, "attendance": 85.0, "previous_score": 75.0}'
```

---

## 🚀 How to Push to GitHub and Deploy to Render

### Step 1: Push Code to GitHub

1. Open PowerShell or Command Prompt in this folder:
   ```bash
   cd "C:\Users\Lavudya Manasa\Desktop\student_pass_fail_prediction"
   ```

2. Initialize Git and configure your identity:
   ```bash
   git init
   git config user.name "Your Name"
   git config user.email "your_email@example.com"
   ```

3. Stage and commit all files:
   ```bash
   git add .
   git commit -m "feat: complete MLOps pipeline and Render deployment setup"
   ```

4. Create a new repository on [GitHub](https://github.com/new) named `student-pass-fail-predictor`.

5. Link and push to GitHub:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/student-pass-fail-predictor.git
   git push -u origin main
   ```

---

### Step 2: Deploy on Render

1. Log in to [Render](https://dashboard.render.com/).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your GitHub account.
4. Select your repository: `student-pass-fail-predictor`.
5. Fill in the following settings (Render may detect these automatically via `render.yaml`):
   - **Name:** `student-pass-fail-predictor`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** `Free`
6. Click **Deploy Web Service**!
7. Render will build and deploy your project. Once live, you will get a public URL like:
   `https://student-pass-fail-predictor.onrender.com`
