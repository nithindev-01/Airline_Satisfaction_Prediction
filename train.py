"""
Single training + testing entry point.

Workflow:
1. Load train.csv and test.csv.
2. Run baseline cross-validation for five classifiers.
3. Tune each classifier with Optuna (20 trials by default).
4. Track baseline, trials, best parameters and metrics in MLflow.
5. Select the winner using training CV ROC-AUC only.
6. Fit only the winner on all training rows.
7. Evaluate the winner once on the untouched test set.
8. Save the fitted pipeline and Streamlit input schema under artifacts/.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "satisfaction"
DROP_COLUMNS = ["Unnamed: 0", "id"]
SEED = 42
N_TRIALS = 20
CV_FOLDS = 3
EXPERIMENT_NAME = "airline-passenger-satisfaction"
TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "airline-satisfaction-classifier"

# Five models: SVC, ExtraTrees and HistGradientBoosting are not included.
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "catboost",
]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def encode_target(values: pd.Series) -> pd.Series:
    labels = values.astype("string").str.strip().str.lower()
    unique = set(labels.dropna().unique())
    if unique.issubset({"satisfied", "neutral or dissatisfied"}):
        return (labels == "satisfied").astype("int8")
    if unique.issubset({"0", "1"}):
        return labels.astype("int8")
    raise ValueError(
        "Unexpected target labels. Expected 'satisfied' and "
        "'neutral or dissatisfied', or numeric 0/1. "
        f"Found: {sorted(map(str, unique))}"
    )


def load_data(data_dir: Path):
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Put the Airline Passenger Satisfaction train.csv and test.csv "
            f"files inside: {data_dir}"
        )

    train_df = clean_frame(pd.read_csv(train_path))
    test_df = clean_frame(pd.read_csv(test_path))
    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError(f"Both CSV files must contain '{TARGET}'.")

    y_train = encode_target(train_df.pop(TARGET))
    y_test = encode_target(test_df.pop(TARGET))

    missing = sorted(set(train_df.columns) - set(test_df.columns))
    if missing:
        raise ValueError(f"Test CSV is missing feature columns: {missing}")
    extra = sorted(set(test_df.columns) - set(train_df.columns))
    if extra:
        test_df = test_df.drop(columns=extra)
    test_df = test_df.reindex(columns=train_df.columns)
    return train_df, y_train, test_df, y_test


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical),
    ])


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if name == "decision_tree":
        return {
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 35),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.25, log=True
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 10.0
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 20.0, log=True
            ),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 700, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.25, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 20.0, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
        }
    raise ValueError(f"Unknown model: {name}")


def build_classifier(name: str, params: dict):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000, solver="liblinear", random_state=SEED, **params
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=SEED, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=SEED, n_jobs=-1, **params
        )
    if name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    if name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, params: dict, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("classifier", build_classifier(name, params)),
    ])


def make_input_schema(X: pd.DataFrame) -> dict:
    """Metadata used by Streamlit to create safe, type-aware input controls."""
    schema = {"features": []}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce")
            median = float(clean.median()) if clean.notna().any() else 0.0
            low = float(clean.min()) if clean.notna().any() else 0.0
            high = float(clean.max()) if clean.notna().any() else 1.0
            if low == high:
                low, high = low - 1.0, high + 1.0
            schema["features"].append({
                "name": col,
                "type": "numeric",
                "min": low,
                "max": high,
                "default": min(max(median, low), high),
                "integer": pd.api.types.is_integer_dtype(series),
            })
        else:
            values = (
                series.dropna().astype(str).value_counts().head(50).index.tolist()
            )
            if not values:
                values = ["Unknown"]
            schema["features"].append({
                "name": col,
                "type": "categorical",
                "options": values,
                "default": values[0],
            })
    return schema


def log_baselines(X, y, cv):
    rows = []
    with mlflow.start_run(run_name="baseline_comparison", nested=True):
        for name in MODEL_NAMES:
            model = build_pipeline(name, {}, X)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="roc_auc", n_jobs=1
            )
            mean_score = float(scores.mean())
            std_score = float(scores.std())
            mlflow.log_metric(f"{name}_baseline_roc_auc", mean_score)
            rows.append({
                "model": name,
                "baseline_cv_roc_auc": mean_score,
                "baseline_cv_std": std_score,
            })
            print(f"Baseline {name}: {mean_score:.5f} (+/- {std_score:.5f})")
    return pd.DataFrame(rows)


def tune_model(name, X, y, cv, n_trials):
    with mlflow.start_run(run_name=f"tuning_{name}", nested=True):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("run_type", "optuna_model_study")

        def objective(trial):
            params = suggest_params(name, trial)
            pipeline = build_pipeline(name, params, X)
            scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
            )
            mean_score = float(np.mean(scores))
            with mlflow.start_run(
                run_name=f"{name}_trial_{trial.number:02d}", nested=True
            ):
                mlflow.set_tag("model_name", name)
                mlflow.set_tag("run_type", "optuna_trial")
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", mean_score)
                mlflow.log_metric("cv_roc_auc_std", float(np.std(scores)))
            return mean_score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name=name,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_params({
            f"best_{key}": value for key, value in study.best_params.items()
        })
        mlflow.log_metric("best_cv_roc_auc", float(study.best_value))
        mlflow.log_metric("optuna_trials", len(study.trials))
        mlflow.log_dict(study.best_params, "best_parameters.json")

    return study


def evaluate_and_save(model, X_train, X_test, y_test, winner_name, best_params, cv_score):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "best_model.joblib")
    (ARTIFACT_DIR / "input_schema.json").write_text(
        json.dumps(make_input_schema(X_train), indent=2),
        encoding="utf-8",
    )
    winner = {
        "model_name": winner_name,
        "best_params": best_params,
        "cv_roc_auc": float(cv_score),
        "test_metrics": metrics,
    }
    (ARTIFACT_DIR / "winner.json").write_text(
        json.dumps(winner, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "classification_report.txt").write_text(
        classification_report(
            y_test, predictions,
            target_names=["Neutral or Dissatisfied", "Satisfied"],
            zero_division=0,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Neutral/Dissatisfied", "Satisfied"],
        cmap="Blues", ax=ax,
    )
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    return metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    X_train, y_train, X_test, y_test = load_data(args.data_dir)
    print(f"Training data: {X_train.shape}")
    print(f"Testing data:  {X_test.shape}")
    print(f"Optuna trials per model: {args.n_trials}")
    print(f"Total Optuna trials: {args.n_trials * len(MODEL_NAMES)}")

    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=SEED
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    leaderboard = []
    with mlflow.start_run(run_name="airline_classification_experiment"):
        mlflow.log_params({
            "models": ",".join(MODEL_NAMES),
            "n_trials_per_model": args.n_trials,
            "total_optuna_trials": args.n_trials * len(MODEL_NAMES),
            "cv_folds": args.cv_folds,
            "optimization_metric": "roc_auc",
            "random_state": SEED,
        })

        baseline_df = log_baselines(X_train, y_train, cv)
        ARTIFACT_DIR.mkdir(exist_ok=True)
        baseline_df.to_csv(ARTIFACT_DIR / "baseline_comparison.csv", index=False)
        mlflow.log_artifact(str(ARTIFACT_DIR / "baseline_comparison.csv"))

        for name in MODEL_NAMES:
            print(f"\n--- Optuna tuning: {name} ---")
            study = tune_model(name, X_train, y_train, cv, args.n_trials)
            leaderboard.append({
                "model": name,
                "best_cv_roc_auc": float(study.best_value),
                "best_params": study.best_params,
            })
            print(f"Best CV ROC-AUC: {study.best_value:.5f}")
            print(f"Best parameters: {study.best_params}")

        leaderboard = sorted(
            leaderboard, key=lambda row: row["best_cv_roc_auc"], reverse=True
        )
        leaderboard_df = pd.DataFrame([
            {
                "model": row["model"],
                "best_cv_roc_auc": row["best_cv_roc_auc"],
                "best_params": json.dumps(row["best_params"]),
            }
            for row in leaderboard
        ])
        leaderboard_df.to_csv(ARTIFACT_DIR / "model_leaderboard.csv", index=False)
        mlflow.log_artifact(str(ARTIFACT_DIR / "model_leaderboard.csv"))

        # Winner selection uses training CV only; test data is not consulted.
        winner = leaderboard[0]
        winner_name = winner["model"]
        best_params = winner["best_params"]
        print(f"\nSelected winner: {winner_name}")

        final_model = build_pipeline(winner_name, best_params, X_train)
        final_model.fit(X_train, y_train)

        test_metrics = evaluate_and_save(
            final_model, X_train, X_test, y_test,
            winner_name, best_params, winner["best_cv_roc_auc"]
        )
        print("\nHeld-out test metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.5f}")

        with mlflow.start_run(run_name=f"final_winner_{winner_name}", nested=True):
            mlflow.set_tag("run_type", "final_winner")
            mlflow.set_tag("selected_model", winner_name)
            mlflow.log_params({
                "selected_model": winner_name,
                **{f"best_{key}": value for key, value in best_params.items()},
            })
            mlflow.log_metric("best_cv_roc_auc", winner["best_cv_roc_auc"])
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            for filename in [
                "winner.json", "classification_report.txt",
                "confusion_matrix.png", "roc_curve.png",
                "input_schema.json", "model_leaderboard.csv",
            ]:
                mlflow.log_artifact(str(ARTIFACT_DIR / filename))

           
# Use predictions for a signature that matches the model's predict() output
            signature = infer_signature(
             X_train.head(5),
             final_model.predict(X_train.head(5)),
            )

            logged = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=X_train.head(3),
)

            try:
                registered = mlflow.register_model(
                    logged.model_uri, REGISTERED_MODEL_NAME
                )
                print(
                    f"MLflow registered {REGISTERED_MODEL_NAME}, "
                    f"version {registered.version}"
                )
            except Exception as exc:
                print(f"Registry registration warning: {exc}")

        print("\nModel leaderboard (training CV ROC-AUC):")
        print(leaderboard_df[["model", "best_cv_roc_auc"]].to_string(index=False))
        print("\nSaved deployment files in the artifacts/ folder.")


if __name__ == "__main__":
    main()
