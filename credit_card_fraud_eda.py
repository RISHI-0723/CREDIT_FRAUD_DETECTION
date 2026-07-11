"""Credit Card Fraud Detection - Data Loading and Exploratory Data Analysis.

This script loads the creditcard.csv dataset, performs initial EDA,
checks missing values and duplicate rows, removes duplicates, and visualizes
class distribution for the target column 'Class'.
All matplotlib figures are saved to the 'plots/' directory (no GUI windows).
A results metadata JSON is saved alongside the model for the Streamlit app.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Directory for saved plots ────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig_name: str) -> None:
    """Save the current matplotlib figure and close it."""
    plt.savefig(PLOTS_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load the dataset into a pandas DataFrame."""
    if file_path is None:
        file_path = str(Path(__file__).resolve().parent / "creditcard.csv")

    if not file_path:
        raise ValueError("File path cannot be empty.")

    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset file not found: {file_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset: {exc}") from exc


# ── EDA ───────────────────────────────────────────────────────────────────────

def perform_eda(df: pd.DataFrame) -> None:
    """Display basic exploratory data analysis information for the dataset."""
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe(include='all').T)

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame and return the cleaned version."""
    try:
        duplicate_rows_before = df.duplicated().sum()
        print(f"\nNumber of duplicate rows before removal: {duplicate_rows_before}")

        df_cleaned = df.drop_duplicates().reset_index(drop=True)

        duplicate_rows_after = df_cleaned.duplicated().sum()
        print(f"Number of duplicate rows after removal: {duplicate_rows_after}")

        print("\nDataset shape before removing duplicates:")
        print(df.shape)

        print("\nDataset shape after removing duplicates:")
        print(df_cleaned.shape)

        return df_cleaned
    except Exception as exc:
        raise RuntimeError(f"Failed to remove duplicates: {exc}") from exc


def visualize_class_distribution(df: pd.DataFrame) -> None:
    """Visualize the target column distribution for fraud detection classes."""
    if "Class" not in df.columns:
        raise KeyError("Target column 'Class' not found in the dataset.")

    class_counts = df["Class"].value_counts()
    class_percentages = df["Class"].value_counts(normalize=True) * 100

    print("\nClass counts:")
    print(class_counts)

    print("\nClass percentages:")
    print(class_percentages)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(x="Class", data=df, ax=axes[0], palette=["#4C78A8", "#F58518"])
    axes[0].set_title("Count Plot of Class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    labels = ["Genuine Transaction", "Fraudulent Transaction"]
    axes[1].pie(
        class_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4C78A8", "#F58518"],
    )
    axes[1].set_title("Class Distribution")

    plt.tight_layout()
    _save("class_distribution.png")   # ← was plt.show()


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    if "Class" not in df.columns:
        raise KeyError("Target column 'Class' not found in the dataset.")

    try:
        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print("\nTrain-test split completed.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test
    except Exception as exc:
        raise RuntimeError(f"Failed to split the data: {exc}") from exc


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to the training data to balance the minority class."""
    try:
        print("\nClass distribution before SMOTE:")
        print(y_train.value_counts())

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print("\nClass distribution after SMOTE:")
        print(y_train_balanced.value_counts())

        print(f"\nX_train shape after SMOTE: {X_train_balanced.shape}")
        print(f"y_train shape after SMOTE: {y_train_balanced.shape}")

        return X_train_balanced, y_train_balanced
    except Exception as exc:
        raise RuntimeError(f"Failed to apply SMOTE: {exc}") from exc


def visualize_balancing(y_train_before: pd.Series, y_train_after: pd.Series) -> None:
    """Create a grouped bar chart comparing class distribution before and after SMOTE."""
    try:
        before_counts = y_train_before.value_counts().sort_index()
        after_counts = y_train_after.value_counts().sort_index()

        balance_df = pd.DataFrame(
            {
                "Before SMOTE": before_counts.reindex([0, 1], fill_value=0).values,
                "After SMOTE": after_counts.reindex([0, 1], fill_value=0).values,
            },
            index=[0, 1],
        )

        balance_df.index.name = "Class"
        balance_df.plot(kind="bar", figsize=(8, 4), color=["#4C78A8", "#F58518"])

        plt.title("Class Distribution Before and After SMOTE")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.legend(title="Distribution")

        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt="%d", padding=3)

        plt.tight_layout()
        _save("smote_balancing.png")   # ← was plt.show()
    except Exception as exc:
        raise RuntimeError(f"Failed to visualize balancing results: {exc}") from exc


# ── Model training ────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    """Train a logistic regression classifier on the balanced training data."""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index,
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        print("✓ Logistic Regression trained successfully.")
        return model, scaler
    except Exception as exc:
        raise RuntimeError(f"Failed to train Logistic Regression: {exc}") from exc


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series
) -> DecisionTreeClassifier:
    """Train a decision tree classifier on the balanced training data."""
    try:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("✓ Decision Tree trained successfully.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to train Decision Tree: {exc}") from exc


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Train a random forest classifier on the balanced training data."""
    try:
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=50,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print("✓ Random Forest trained successfully.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to train Random Forest: {exc}") from exc


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, dict[str, object]]:
    """Train all requested classification models on the balanced training data."""
    try:
        logistic_model, scaler = train_logistic_regression(X_train, y_train)
        decision_tree_model = train_decision_tree(X_train, y_train)
        random_forest_model = train_random_forest(X_train, y_train)

        models = {
            "Logistic Regression": {
                "model": logistic_model,
                "scaler": scaler,
            },
            "Decision Tree": {
                "model": decision_tree_model,
                "scaler": None,
            },
            "Random Forest": {
                "model": random_forest_model,
                "scaler": None,
            },
        }
        print("\nAll requested models trained successfully.")
        return models
    except Exception as exc:
        raise RuntimeError(f"Failed to train all models: {exc}") from exc


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model_info: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, object]:
    """Evaluate a trained model on the original test data."""
    try:
        model = model_info["model"]

        if model_name == "Logistic Regression":
            scaler = model_info["scaler"]
            X_test_scaled = scaler.transform(X_test)
            X_test_for_model = pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index,
            )
        else:
            X_test_for_model = X_test

        predictions = model.predict(X_test_for_model)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, predictions)

        print(f"\n{'=' * 40}")
        print(f"{model_name} Results")
        print(f"{'=' * 40}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Predictions": predictions,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to evaluate {model_name}: {exc}") from exc


def plot_confusion_matrix(
    y_test: pd.Series,
    predictions: np.ndarray,
    model_name: str,
) -> None:
    """Plot and save a confusion matrix heatmap for a trained model."""
    try:
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        _save(f"confusion_matrix_{safe_name}.png")   # ← was plt.show()
    except Exception as exc:
        raise RuntimeError(f"Failed to plot confusion matrix for {model_name}: {exc}") from exc


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_models(results: list[dict[str, object]]) -> pd.DataFrame:
    """Compare evaluation results across all models in a DataFrame."""
    try:
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.drop(columns=["Predictions"], errors="ignore")
        comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False)
        print("\nModel Comparison Table")
        print(comparison_df)
        return comparison_df
    except Exception as exc:
        raise RuntimeError(f"Failed to compare models: {exc}") from exc


def display_best_model(comparison_df: pd.DataFrame) -> None:
    """Identify and display the best-performing model based on F1-score."""
    try:
        best_row = comparison_df.iloc[0]
        best_model = best_row["Model"]
        best_f1 = best_row["F1-Score"]

        print(f"\n{'=' * 36}")
        print("Best Performing Model")
        print(f"{'=' * 36}")
        print(f"Model Name: {best_model}")
        print(f"F1-score: {best_f1:.4f}")
        print(
            "Reason: This model achieved the highest F1-score while "
            "maintaining a strong balance between Precision and Recall, "
            "making it the most suitable model for fraud detection."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to identify the best model: {exc}") from exc


# ── Save best model ───────────────────────────────────────────────────────────

def save_best_model(best_model: object) -> Path:
    """Save the selected best model safely using joblib."""
    try:
        model_file = Path(__file__).resolve().parent / "fraud_detection_model.pkl"

        if isinstance(best_model, dict):
            model_obj = best_model.get("model")
            scaler = best_model.get("scaler")

            if scaler is not None:
                scaler_file = Path(__file__).resolve().parent / "scaler.pkl"
                joblib.dump(scaler, scaler_file)
                print("Scaler saved successfully.")
                print("Location:")
                print(scaler_file)
        else:
            model_obj = best_model

        joblib.dump(model_obj, model_file)
        print("Best model saved successfully.")
        print("Location:")
        print(model_file)
        return model_file
    except Exception as exc:
        raise RuntimeError(f"Failed to save the best model: {exc}") from exc


# ── Feature-importance plot ───────────────────────────────────────────────────

def plot_feature_importance(rf_model: RandomForestClassifier, feature_names: list[str]) -> None:
    """Save a Top-10 feature importance bar chart for the Random Forest model."""
    try:
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_importances, y=top_features, palette="Blues_r")
        plt.title("Top 10 Feature Importances — Random Forest")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        _save("feature_importance.png")
        print("✓ Feature importance plot saved.")
    except Exception as exc:
        raise RuntimeError(f"Failed to plot feature importance: {exc}") from exc


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_transaction(model: object, transaction: object) -> str:
    """Predict whether a single transaction is fraudulent or genuine."""
    try:
        transaction_array = np.asarray(transaction).reshape(1, -1)
        model_obj = model
        scaler = None

        if isinstance(model, dict):
            model_obj = model.get("model")
            scaler = model.get("scaler")

        if scaler is not None:
            transaction_array = scaler.transform(transaction_array)

        prediction = model_obj.predict(transaction_array)[0]
        return "Fraudulent Transaction" if int(prediction) == 1 else "Genuine Transaction"
    except Exception as exc:
        raise RuntimeError(f"Failed to predict the transaction: {exc}") from exc


# ── Results metadata ──────────────────────────────────────────────────────────

def save_results_metadata(
    df: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Persist key results to results_metadata.json for the Streamlit app."""
    class_counts = df_cleaned["Class"].value_counts().to_dict()
    metadata = {
        "raw_shape": list(df.shape),
        "cleaned_shape": list(df_cleaned.shape),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_count": int(class_counts.get(1, 0)),
        "genuine_count": int(class_counts.get(0, 0)),
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "model_results": comparison_df.to_dict(orient="records"),
        "best_model": str(comparison_df.iloc[0]["Model"]),
        "best_f1": float(comparison_df.iloc[0]["F1-Score"]),
    }

    out_path = Path(__file__).resolve().parent / "results_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Results metadata saved to: {out_path}")


# ── Project summary ───────────────────────────────────────────────────────────

def project_summary() -> None:
    """Print a complete summary of the fraud detection project workflow."""
    try:
        print(f"\n{'=' * 36}")
        print("PROJECT SUMMARY")
        print(f"{'=' * 36}")
        print("Dataset Loaded")
        print("EDA Completed")
        print("Missing Values Checked")
        print("Duplicates Removed")
        print("SMOTE Applied")
        print("Models Trained")
        print("Models Evaluated")
        print("Best Model Selected")
        print("Model Saved Successfully")
    except Exception as exc:
        raise RuntimeError(f"Failed to print project summary: {exc}") from exc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full workflow: EDA → preprocessing → SMOTE → training → evaluation → saving."""
    try:
        df = load_data()
        perform_eda(df)
        df_cleaned = remove_duplicates(df)
        visualize_class_distribution(df_cleaned)

        X_train, X_test, y_train, y_test = split_data(df_cleaned)
        X_train_smote, y_train_smote = apply_smote(X_train, y_train)
        visualize_balancing(y_train, y_train_smote)
        models = train_all_models(X_train_smote, y_train_smote)

        results = []
        for model_name, model_info in models.items():
            result = evaluate_model(model_info, X_test, y_test, model_name)
            plot_confusion_matrix(y_test, result["Predictions"], model_name)
            results.append(result)

        comparison_df = compare_models(results)
        display_best_model(comparison_df)

        # ── Save feature importance for Random Forest ──────────────────────
        rf_info = models.get("Random Forest")
        if rf_info is not None:
            plot_feature_importance(rf_info["model"], X_train.columns.tolist())

        # ── Save best model ────────────────────────────────────────────────
        best_model_name = comparison_df.iloc[0]["Model"]
        best_model_info = None
        for name, model_info in models.items():
            if name == best_model_name:
                best_model_info = model_info
                break

        if best_model_info is not None:
            save_best_model(best_model_info)
            sample_transaction = X_test.iloc[0]
            prediction = predict_transaction(best_model_info, sample_transaction)
            print("\nSample Prediction:")
            print(prediction)

        # ── Persist metadata for Streamlit ─────────────────────────────────
        save_results_metadata(df, df_cleaned, comparison_df, X_train.columns.tolist())

        project_summary()
    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()