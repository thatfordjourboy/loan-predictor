from __future__ import annotations
import time
from time import perf_counter
from typing import Any, Dict, Tuple, Optional

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.exceptions import UndefinedMetricWarning

# Silence undefined-metric spam during CV when a fold predicts one class
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------- UI: Dataset info ----------
def show_dataset_info(df: pd.DataFrame) -> None:
    total_records = f"{df.shape[0]:,}"
    total_features = df.shape[1]
    missing_values = f"{df.isnull().sum().sum():,}"
    default_rate = f"{(df['Default'] == 1).mean() * 100:.1f}%" if "Default" in df.columns else "N/A"

    st.markdown(
        f"""
        <div style="border: 1px solid #e6e6e6; border-radius: 12px; padding: 24px; margin-bottom: 20px;">
            <h4 style="margin-bottom: 5px;">📂 Dataset Information</h4>
            <p style="margin-top: 0px; color: #666;">Loan Default Prediction Dataset</p>
            <div style="background: #f9f9f9; padding: 15px 20px; border-radius: 10px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <strong style="font-size: 16px;">Loan_default.csv</strong><br>
                    <span style="color: #777; font-size: 13px;">A dataset for classifying loan default risk using applicant profiles</span>
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{total_records}</div>
                    <div style="font-size: 13px; color: #666;">Total Records</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{total_features}</div>
                    <div style="font-size: 13px; color: #666;">Features</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{missing_values}</div>
                    <div style="font-size: 13px; color: #666;">Missing Values</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{default_rate}</div>
                    <div style="font-size: 13px; color: #666;">Default Rate</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Dataset Preview")
    st.caption(f"Showing the complete dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True, height=400)

    with st.expander("🔍 Column Details"):
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2),
            }
        )
        st.dataframe(col_info, use_container_width=True)


# ---------- Preprocessing ----------
def _make_onehot_dense() -> OneHotEncoder:
    """Return a dense-output OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def run_preprocessing(df: pd.DataFrame) -> None:
    """
    Fit transformers on training data, transform train/test, and stash in session_state.
    Also store medians/mins/maxs for form defaults and ranges.
    """
    if st.session_state.get("preprocessing_done"):
        st.info("Preprocessing has already been done.")
        return

    data = df.copy()

    # Drop ID + target for X; keep y
    X = data.drop(columns=["LoanID", "Default"], errors="ignore")
    y = data["Default"]

    # Column groups
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Numeric distribution info for realistic defaults on the form
    num_medians = X_train[numeric_cols].median(numeric_only=True).to_dict()
    num_mins = X_train[numeric_cols].min(numeric_only=True).to_dict()
    num_maxs = X_train[numeric_cols].max(numeric_only=True).to_dict()

    # Transformers
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _make_onehot_dense())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit + transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    st.session_state.update(
        {
            "preprocessing_done": True,
            "preprocessor": preprocessor,
            "X_train": X_train_transformed,
            "X_test": X_test_transformed,
            "y_train": y_train,
            "y_test": y_test,
            "medians": num_medians,
            "mins": num_mins,
            "maxs": num_maxs,
            "target_names": ["No Default", "Default"],
        }
    )

    st.success("✅ Preprocessing completed successfully.")


def render_preprocessing_steps() -> None:
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-title">⚙️ Preprocessing Pipeline</div>
            <div class="pipeline-subtitle">Step-by-step data transformation process</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">1</div>
                    <div class="step-title">Missing Value Treatment</div>
                    <div class="status-pill">✔ completed</div>
                </div>
                <div class="step-subtitle">Handle missing values using appropriate imputation strategies</div>
                <div class="checkmark">✅ Median imputation for numerical features</div>
                <div class="checkmark">✅ Mode imputation for categorical features</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">2</div>
                    <div class="step-title">Feature Scaling & Encoding</div>
                    <div class="status-pill">✔ completed</div>
                </div>
                <div class="step-subtitle">Standardize and encode feature types for modeling</div>
                <div class="checkmark">✅ Standard scaling for numerical features</div>
                <div class="checkmark">✅ One-hot encoding for categorical features</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">3</div>
                    <div class="step-title">Train-Test Split</div>
                    <div class="status-pill">✔ completed</div>
                </div>
                <div class="step-subtitle">Split dataset into separate training and testing subsets</div>
                <div class="checkmark">✅ Stratified 80% train / 20% test</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")

    if X_train is not None and X_test is not None and y_train is not None:
        st.markdown(
            f"""
                <div class="step-card summary-box">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">✅ Summary</div>
                    <div><strong>X_train shape:</strong> {X_train.shape}</div>
                    <div><strong>X_test shape:</strong> {X_test.shape}</div>
                </div>
            </div> <!-- close pipeline container -->
            """,
            unsafe_allow_html=True,
        )

        # y_train distribution
        dist_df = y_train.value_counts(normalize=True).reset_index()
        dist_df.columns = ["Class", "Proportion"]
        dist_df["Class"] = dist_df["Class"].map({0: "No Default (0)", 1: "Default (1)"})
        st.markdown("**y_train distribution:**")
        st.table(dist_df)
    else:
        st.warning("No training/test data found in session state.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Threshold / score utilities ----------
def _safe_scores(model, X):
    """Return scores in [0,1] where higher = risk(Default=1)."""
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            raw = model.decision_function(X)
            rmin, rmax = float(raw.min()), float(raw.max())
            return (raw - rmin) / (rmax - rmin + 1e-9)
        except Exception:
            return None

def _approval_curve(y_true, scores, thr_grid):
    """For each threshold, compute approval rate (scores<thr) and default rate among approved."""
    out = []
    for thr in thr_grid:
        approved = scores < thr
        approved_cnt = int(approved.sum())
        appr_rate = approved_cnt / len(scores)
        def_rate = np.nan if approved_cnt == 0 else float(((y_true == 1) & approved).sum()) / approved_cnt
        out.append((thr, appr_rate, def_rate))
    return pd.DataFrame(out, columns=["Threshold", "ApprovalRate", "DefaultRateAmongApproved"])

def _recommend_threshold_by_cap(y_true, scores, max_default_rate_among_approved: float = 0.08):
    grid = np.linspace(0.05, 0.95, 91)
    curve = _approval_curve(y_true, scores, grid)
    feasible = curve[curve["DefaultRateAmongApproved"] <= max_default_rate_among_approved].copy()
    if feasible.empty:
        feasible = curve.dropna(subset=["DefaultRateAmongApproved"]).copy()
        if feasible.empty:
            return 0.50, None
        idx = (feasible["DefaultRateAmongApproved"] - max_default_rate_among_approved).abs().idxmin()
        return float(feasible.loc[idx, "Threshold"]), feasible
    idx = feasible.sort_values(["ApprovalRate", "DefaultRateAmongApproved"], ascending=[False, True]).index[0]
    return float(feasible.loc[idx, "Threshold"]), feasible


# ---------- Model training & evaluation ----------
def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: np.ndarray | list,
    on_step=None,
    use_class_weight: bool = True,
    cv_folds: int = 3,
    refit_metric: str = "f1",
    n_iter_per_model: int = 6,
    target_names: Optional[list[str]] = None,
    calibrate_proba: bool = True,
) -> Tuple[
    Dict[str, Any], str, Any, np.ndarray, np.ndarray, Optional[pd.DataFrame], Optional[str], Optional[float]
]:
    """Train Decision Tree & Random Forest via randomized CV; evaluate on test set."""

    def step(label: str):
        if on_step:
            on_step(f"▶️ {label}…")
        return (label, perf_counter())

    def done(token):
        label, t0 = token
        if on_step:
            on_step(f"✅ {label} ({perf_counter() - t0:.2f}s)")

    results_dict: Dict[str, Any] = {}
    best_name: Optional[str] = None
    best_model: Optional[Any] = None
    y_pred_best: Optional[np.ndarray] = None
    cm_best: Optional[np.ndarray] = None
    feat_imp_df: Optional[pd.DataFrame] = None
    clf_report_text: Optional[str] = None
    infer_time_ms_per_sample: Optional[float] = None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    dt_param_dist = {
        "max_depth": [3, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        **({"class_weight": [None, "balanced"]} if use_class_weight else {}),
    }
    rf_param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        **({"class_weight": [None, "balanced"]} if use_class_weight else {}),
    }

    model_specs = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42), dt_param_dist),
        ("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_dist),
    ]

    scoring = {
        "f1":        make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score, zero_division=0),
        "roc_auc":   "roc_auc",
    }

    tok = step("Randomized CV on training data")
    for label, base_model, param_dist in model_specs:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter_per_model,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=42,
            return_train_score=False,
        )

        t0 = perf_counter()
        search.fit(X_train, y_train)
        cv_fit_time = perf_counter() - t0

        best_est = search.best_estimator_

        # Probability calibration improves threshold tuning and displayed probabilities
        calibrated_est = best_est
        if calibrate_proba:
            try:
                method = "isotonic" if len(y_train) >= 2000 else "sigmoid"
                calibrated_est = CalibratedClassifierCV(base_estimator=best_est, method=method, cv=3)
                calibrated_est.fit(X_train, y_train)
            except Exception:
                calibrated_est = best_est  # fallback

        # Honest hold-out evaluation
        y_pred = calibrated_est.predict(X_test)
        try:
            y_scores = calibrated_est.predict_proba(X_test)[:, 1]
        except Exception:
            y_scores = None

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_scores) if y_scores is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        # Inference timing (ms/sample)
        t_inf0 = perf_counter()
        _ = calibrated_est.predict(X_test)
        per_sample_ms = ((perf_counter() - t_inf0) / len(X_test)) * 1000

        name = f"{label} (best={search.best_params_})"
        results_dict[name] = {
            "model": calibrated_est,
            "best_params": search.best_params_,
            "cv_fit_time": cv_fit_time,
            "cv_best_score_refit_metric": search.best_score_,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "y_pred": y_pred,
            "conf_matrix": cm,
            "inference_ms_per_sample": per_sample_ms,
        }

        if (best_model is None) or (test_f1 > results_dict.get(best_name, {}).get("test_f1", -1)):
            best_name = name
            best_model = calibrated_est
            y_pred_best = y_pred
            cm_best = cm
            infer_time_ms_per_sample = per_sample_ms

            try:
                importances = getattr(search.best_estimator_, "feature_importances_", None)
                if importances is not None:
                    feat_imp_df = (
                        pd.DataFrame({"Feature": list(feature_names), "Importance": importances})
                        .sort_values("Importance", ascending=False)
                        .reset_index(drop=True)
                    )
                else:
                    feat_imp_df = None
            except Exception:
                feat_imp_df = None

            clf_report_text = classification_report(
                y_test, y_pred_best,
                target_names=target_names or ["0", "1"],
                digits=3,
                zero_division=0,
            )

    done(tok)
    tok = step("Packaging results"); time.sleep(0.02); done(tok)

    return (
        results_dict,
        best_name or "",
        best_model,
        y_pred_best,
        cm_best,
        feat_imp_df,
        clf_report_text,
        infer_time_ms_per_sample,
    )
