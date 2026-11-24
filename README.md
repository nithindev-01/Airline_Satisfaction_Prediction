# ✈️ Airline Passenger Satisfaction Prediction

### An End-to-End Machine Learning Application with Optuna, MLflow, and Streamlit

Predicting airline passenger satisfaction using machine learning, automated hyperparameter optimization, and experiment tracking — packaged into an interactive web application.

---

## 📌 Project Overview

Passenger satisfaction plays an important role in the airline industry's ability to improve service quality, customer experience, and customer retention.

This project develops an end-to-end machine learning solution to predict whether an airline passenger is **Satisfied** or **Neutral or Dissatisfied** based on passenger demographics, travel information, and in-flight service experiences.

The project goes beyond traditional model training by implementing an automated machine learning workflow that includes:

- Data preprocessing and feature engineering.
- Training and comparison of multiple classification algorithms.
- Hyperparameter optimization using **Optuna**.
- Experiment tracking and model management using **MLflow**.
- Model evaluation using classification metrics.
- Deployment through an interactive **Streamlit** application.

The final application allows users to enter passenger information and receive a satisfaction prediction with its corresponding probability.

---

## 🎯 Project Objectives

- Understand the factors that influence airline passenger satisfaction.
- Build reliable classification models using real-world passenger data.
- Compare multiple machine learning algorithms.
- Optimize model performance using Optuna.
- Track experiments, parameters, and evaluation metrics using MLflow.
- Select and save the best-performing model.
- Develop an interactive application for real-time predictions.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, CatBoost |
| Hyperparameter Optimization | Optuna |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 🤖 Machine Learning Models

The following classification algorithms are trained and evaluated:

1. **Logistic Regression** – A baseline classification algorithm.
2. **Decision Tree** – Learns decision rules from the training data.
3. **Random Forest** – Combines multiple decision trees to improve predictive performance.
4. **XGBoost** – Uses gradient boosting to build powerful predictive models.
5. **CatBoost** – A gradient boosting algorithm designed to handle categorical features effectively.

### Model Selection

Models are compared using cross-validation ROC-AUC scores.

The best-performing model is selected based on its validation performance and subsequently evaluated on the test dataset.

---

## ⚙️ Machine Learning Workflow

The project follows a structured machine learning pipeline:

```text
                Dataset
                   |
                   v
          Data Preprocessing
                   |
                   v
           Feature Engineering
                   |
                   v
          Train-Test Evaluation
                   |
                   v
          Model Development
                   |
                   v
       Hyperparameter Optimization
               (Optuna)
                   |
                   v
          Cross Validation
                   |
                   v
         Experiment Tracking
               (MLflow)
                   |
                   v
          Model Comparison
                   |
                   v
          Best Model Selection
                   |
                   v
          Model Serialization
                   |
                   v
       Interactive Streamlit App
                   |
                   v
          Passenger Prediction
```

---

## 🔍 Hyperparameter Optimization with Optuna

Instead of manually selecting model hyperparameters, this project uses **Optuna** to automate the optimization process.

For each machine learning model:

- 20 optimization trials are performed.
- Different hyperparameter combinations are explored.
- Stratified K-Fold cross-validation is used.
- ROC-AUC is used as the optimization objective.
- The best hyperparameters are selected based on validation performance.

This approach helps identify suitable model configurations while making the model comparison process more systematic.

---

## 📊 Experiment Tracking with MLflow

MLflow is integrated to track machine learning experiments and manage model artifacts.

For every experiment, relevant information is recorded, including:

- Model name.
- Hyperparameter configurations.
- Cross-validation performance.
- Evaluation metrics.
- Trained model artifacts.

### MLflow Workflow

1. Create an MLflow experiment.
2. Start a parent run for model optimization.
3. Record individual Optuna trial information.
4. Log the best hyperparameters and evaluation metrics.
5. Save the selected model as an MLflow artifact.
6. Compare experiment results through the MLflow UI.

To launch the MLflow dashboard locally:

```bash
mlflow ui
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

---

## 📈 Model Performance

The following results were obtained from the selected CatBoost model.

### Best Model: CatBoost

<details>
<summary><strong>View Evaluation Metrics</strong></summary>

| Evaluation Metric | Score |
|---|---:|
| Cross-Validation ROC-AUC | 0.9953 |
| Test Accuracy | 96.40% |
| Test Precision | 97.32% |
| Test Recall | 94.40% |
| Test F1-Score | 95.84% |
| Test ROC-AUC | 0.9955 |

</details>

### Performance Highlights

- Achieved **96.40% test accuracy**.
- Achieved **0.9955 test ROC-AUC**.
- Demonstrated strong classification performance.
- Selected CatBoost based on cross-validation ROC-AUC.

*Note: These are the recorded results from the current experiment. Performance may vary if the dataset, preprocessing, or training configuration changes.*

---

## 🌐 Streamlit Web Application

The trained model is integrated into an interactive Streamlit application.

Users can enter passenger details and receive predictions without needing to execute Python scripts or understand the underlying machine learning implementation.

### Application Features

- Professional and interactive user interface.
- Passenger information input.
- Real-time satisfaction prediction.
- Predicted satisfaction category.
- Satisfaction probability display.
- Information about the machine learning model.
- Instructions explaining how to use the application.

### Live Application

🚀 **[Open the Streamlit Application](YOUR_STREAMLIT_APP_URL)**

*Replace the placeholder with your deployed Streamlit application URL.*

---

## 📂 Project Structure

```text
airline-passenger-satisfaction/
│
├── app.py
│
├── train.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── artifacts/
│   ├── best_model.joblib
│   ├── input_schema.json
│   └── winner.json
│
└── .gitignore
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application for passenger satisfaction predictions. |
| `train.py` | Complete model training, optimization, evaluation, and experiment tracking workflow. |
| `requirements.txt` | Python dependencies required to run the project. |
| `data/` | Training and testing datasets. |
| `artifacts/` | Saved model and supporting prediction metadata. |
| `README.md` | Project documentation. |

*The dataset and local MLflow database do not need to be uploaded to GitHub for the deployed prediction application if the saved model artifacts are available.*

---

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/airline-passenger-satisfaction.git
```

### 2. Navigate to the Project Directory

```bash
cd airline-passenger-satisfaction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

Make sure the required model artifacts are available inside the `artifacts/` directory.

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🧪 Run the Model Training Pipeline

To execute the complete machine learning workflow:

1. Place the training and testing datasets inside the `data/` directory.
2. Ensure the required dependencies are installed.
3. Execute the training script.

```bash
python train.py
```

The training pipeline performs:

- Data preprocessing.
- Model training.
- Optuna hyperparameter optimization.
- Cross-validation.
- Model evaluation.
- MLflow experiment tracking.
- Best model selection and artifact saving.

**Note:** Model training is performed locally. The deployed Streamlit application uses the saved model artifacts and does not need to retrain the models every time a user opens the application.

---

## ☁️ Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select **Create app**.
5. Choose your repository and the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Once deployment is complete, Streamlit provides a public URL that can be shared with others.

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Building an end-to-end classification workflow.
- Comparing different machine learning algorithms.
- Applying cross-validation for reliable model evaluation.
- Automating hyperparameter tuning using Optuna.
- Tracking experiments and model artifacts using MLflow.
- Evaluating models using multiple classification metrics.
- Saving and reusing trained machine learning pipelines.
- Developing interactive ML applications using Streamlit.
- Preparing machine learning projects for deployment.

---

## 🔮 Future Improvements

Potential enhancements for this project include:

- Adding model explainability using SHAP.
- Providing feature importance visualizations.
- Monitoring prediction patterns and model performance.
- Implementing automated model retraining.
- Integrating a cloud-based MLflow tracking server.
- Adding data drift detection.
- Developing a REST API for model predictions.

---

## 👨‍💻 Author

**Vanga Nithin Dev**

Computer Science Engineering Graduate | Aspiring Machine Learning Engineer

Interested in Machine Learning, Data Science, and building practical AI applications.

- **GitHub:** [Your GitHub Profile](https://github.com/YOUR_USERNAME)
- **LinkedIn:** [Your LinkedIn Profile](YOUR_LINKEDIN_URL)

---

## ⭐ Support

If you find this project useful or interesting, consider giving this repository a ⭐ on GitHub.

Your feedback and suggestions are always welcome!

---

*Built with Python, Machine Learning, Optuna, MLflow, and Streamlit.*
