from __future__ import annotations
import os
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_slug: str = "mirichoi0218/insurance") -> pd.DataFrame:
    """Download and load the dataset into a pandas DataFrame."""
    download_dir = kagglehub.dataset_download(dataset_slug)
    csv_file = os.path.join(download_dir, "insurance.csv")
    if not os.path.exists(csv_file):
        csv_file = "insurance.csv"
    return pd.read_csv(csv_file)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove redundant duplicate records from the dataset."""
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_rows - len(df_cleaned)
    if dropped > 0:
        print(f"[INFO] Removed {dropped} duplicate row(s). Remaining records: {len(df_cleaned)}.")
    return df_cleaned


def export_eda_visualizations(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Generate and save exploratory data analysis figures."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["charges"], kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Distribusi Biaya Medis (Charges)", fontweight="bold")
    axes[0].set_xlabel("Charges ($)")

    sns.histplot(df["age"], kde=True, ax=axes[1], color="#3d8b37")
    axes[1].set_title("Distribusi Usia (Age)", fontweight="bold")
    axes[1].set_xlabel("Age")

    sns.histplot(df["bmi"], kde=True, ax=axes[2], color="#d95f02")
    axes[2].set_title("Distribusi BMI", fontweight="bold")
    axes[2].set_xlabel("BMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_univariate_distribution.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = {"yes": "#e41a1c", "no": "#377eb8"}
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette=palette, alpha=0.7, ax=axes[0])
    axes[0].set_title("Usia vs Biaya (Kategori Perokok)", fontweight="bold")

    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", palette=palette, alpha=0.7, ax=axes[1])
    axes[1].set_title("BMI vs Biaya (Kategori Perokok)", fontweight="bold")

    sns.boxplot(data=df, x="smoker", y="charges", hue="smoker", palette=palette, legend=False, ax=axes[2])
    axes[2].set_title("Biaya berdasarkan Status Merokok", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_bivariate_relationships.png"), dpi=300)
    plt.close()


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "charges",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "figures",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Encode categorical variables, split train/test sets, and scale numeric attributes."""
    df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriks Korelasi Fitur Terhadap Target (Charges)", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    numeric_features = ["age", "bmi", "children"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, and R-squared regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Any]:
    """Train baseline algorithms, perform hyperparameter tuning, and tabulate evaluation metrics."""
    model_candidates = {
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
    }

    records = []
    for name, model in model_candidates.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_m = compute_metrics(y_train, train_pred)
        test_m = compute_metrics(y_test, test_pred)

        records.append({
            "Model": name,
            "Train MSE": train_m["mse"], "Test MSE": test_m["mse"],
            "Train RMSE": train_m["rmse"], "Test RMSE": test_m["rmse"],
            "Train MAE": train_m["mae"], "Test MAE": test_m["mae"],
            "Train R2": train_m["r2"], "Test R2": test_m["r2"],
        })

    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }
    grid_search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_

    tuned_train_pred = best_gb.predict(X_train)
    tuned_test_pred = best_gb.predict(X_test)
    tuned_train_m = compute_metrics(y_train, tuned_train_pred)
    tuned_test_m = compute_metrics(y_test, tuned_test_pred)

    records.append({
        "Model": "Gradient Boosting (Tuned)",
        "Train MSE": tuned_train_m["mse"], "Test MSE": tuned_test_m["mse"],
        "Train RMSE": tuned_train_m["rmse"], "Test RMSE": tuned_test_m["rmse"],
        "Train MAE": tuned_train_m["mae"], "Test MAE": tuned_test_m["mae"],
        "Train R2": tuned_train_m["r2"], "Test R2": tuned_test_m["r2"],
    })

    eval_df = pd.DataFrame(records)
    return eval_df, best_gb


def export_performance_visualizations(
    eval_df: pd.DataFrame,
    best_model: Any,
    feature_names: list[str],
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: str = "figures",
) -> None:
    """Save model performance comparisons, actual vs. predicted plot, and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eval_df.plot(x="Model", y=["Train RMSE", "Test RMSE"], kind="bar", ax=axes[0], colormap="viridis")
    axes[0].set_title("Perbandingan RMSE (Train vs Test)", fontweight="bold")
    axes[0].set_ylabel("RMSE ($)")
    axes[0].set_xticklabels(eval_df["Model"], rotation=15)

    eval_df.plot(x="Model", y=["Train R2", "Test R2"], kind="bar", ax=axes[1], colormap="magma")
    axes[1].set_title("Perbandingan R-Squared (Train vs Test)", fontweight="bold")
    axes[1].set_ylabel("R2 Score")
    axes[1].set_xticklabels(eval_df["Model"], rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_metrics.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#2b5c8f", edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Garis Identitas Ideal (y = x)")
    plt.title("Nilai Aktual vs Prediksi (Gradient Boosting Tuned)", fontweight="bold")
    plt.xlabel("Nilai Aktual Charges ($)")
    plt.ylabel("Nilai Prediksi Charges ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    feat_imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    feat_imp.plot(kind="barh", color="#2ca02c")
    plt.title("Tingkat Kepentingan Fitur (Feature Importance) - GB Tuned", fontweight="bold")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()


def main() -> None:
    """Execution entry point."""
    print("[INFO] Loading dataset via kagglehub...")
    df = load_dataset()
    print(f"[INFO] Initial dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_dataset(df)

    print("[INFO] Generating EDA figures...")
    export_eda_visualizations(df)

    print("[INFO] Preparing features and splitting dataset...")
    X_train, X_test, y_train, y_test, _ = prepare_features(df)
    print(f"[INFO] Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("[INFO] Training baseline models and performing hyperparameter tuning...")
    eval_df, best_gb = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    print("\n[EVALUATION METRICS]")
    print(eval_df.to_string(index=False))

    y_test_pred = best_gb.predict(X_test)
    export_performance_visualizations(eval_df, best_gb, X_train.columns.tolist(), y_test, y_test_pred)

    best_metrics = eval_df.loc[eval_df["Model"] == "Gradient Boosting (Tuned)"].iloc[0]
    print(f"\n[INFO] Best Model: {best_metrics['Model']}")
    print(f"[INFO] Test RMSE: ${best_metrics['Test RMSE']:.2f}, Test R2: {best_metrics['Test R2']:.6f}")
    print("[INFO] Execution completed successfully. Artifacts saved to 'figures/'.")


if __name__ == "__main__":
    main()
