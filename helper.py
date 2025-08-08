import time
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def show_dataset_info(df: pd.DataFrame) -> None:
    total_records = f"{df.shape[0]:,}"
    total_features = df.shape[1]
    missing_values = f"{df.isnull().sum().sum():,}"
    default_rate = (
        f"{(df['Default'] == 1).mean() * 100:.1f}%" if 'Default' in df.columns else "N/A"
    )

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

    # Display the full dataset with scrolling capability
    st.dataframe(
        df,
        use_container_width=True,
        height=400,  # Set a fixed height to enable scrolling
    )

    # Column information
    with st.expander("🔍 Column Details"):
        col_info = pd.DataFrame(
            {
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            }
        )
        st.dataframe(col_info, use_container_width=True)


def run_preprocessing(df: pd.DataFrame) -> None:
    if not st.session_state.get("preprocessing_done"):
        # Make a safe working copy of the dataset
        data = df.copy()

        # Drop label and ID columns
        X = data.drop(columns=["LoanID", "Default"], errors="ignore")
        y = data["Default"]

        # Identify feature types
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include="object").columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define preprocessing pipeline
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        # Fit and transform
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Store in session state
        st.session_state.update(
            {
                "preprocessing_done": True,
                "preprocessor": preprocessor,
                "X_train": X_train_transformed,
                "X_test": X_test_transformed,
                "y_train": y_train,
                "y_test": y_test,
            }
        )

        st.success("✅ Preprocessing completed successfully.")
    else:
        st.info("Preprocessing has already been done.")


def render_preprocessing_steps() -> None:
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-title">⚙️ Preprocessing Pipeline</div>
            <div class="pipeline-subtitle">Step-by-step data transformation process</div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1: Missing value treatment
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

    # Step 2: Feature scaling and encoding
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

    # Step 3: Train-test split
    st.markdown(
        """
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">3</div>
                    <div class="step-title">Train-Test Split</div>
                    <div class="status-pill">✔ completed</div>
                </div>
                <div class="step-subtitle">Split dataset into separate training and testing subsets</div>
                <div class="checkmark">✅ Split into 80% train / 20% test</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary box
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")

    if X_train is not None and X_test is not None and y_train is not None:
        st.markdown(
            """
                <div class="step-card summary-box">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">✅ Summary</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
                    <div><strong>X_train shape:</strong> {X_train.shape}</div>
                    <div><strong>X_test shape:</strong> {X_test.shape}</div>
                </div>
            </div> <!-- close pipeline container -->
            """,
            unsafe_allow_html=True,
        )

        # Display y_train distribution
        dist_df = y_train.value_counts(normalize=True).reset_index()
        dist_df.columns = ["Class", "Proportion"]
        dist_df["Class"] = dist_df["Class"].apply(lambda x: f"Class {x}")
        st.markdown("**y_train distribution:**")
        st.table(dist_df)
    else:
        st.warning("No training/test data found in session state.")
        st.markdown("</div>", unsafe_allow_html=True)


def train_and_evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
) -> tuple:
    results_dict = {}
    best_model_name = None
    best_model_object = None
    y_pred_best = None
    conf_matrix_best = None
    feature_importance_df = None
    best_f1_score = -1

    # Hyperparameter grids
    dt_params = {
        "max_depth": [4, 6, 8],
        "min_samples_split": [5, 10],
    }
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 8],
    }

    # Train Decision Tree models
    for max_depth in dt_params["max_depth"]:
        for min_split in dt_params["min_samples_split"]:
            name = f"Decision Tree (depth={max_depth}, split={min_split})"
            model = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_split=min_split, random_state=42
            )

            start_time = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start_time

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            try:
                y_scores = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_scores = model.decision_function(X_test)

            results_dict[name] = {
                "model": model,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, zero_division=0),
                "recall": recall_score(y_test, y_test_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
                "auc": roc_auc_score(y_test, y_scores),
                "training_time": duration,
                "y_pred": y_test_pred,
                "conf_matrix": confusion_matrix(y_test, y_test_pred),
            }

            if results_dict[name]["f1_score"] > best_f1_score:
                best_model_name = name
                best_model_object = model
                y_pred_best = y_test_pred
                conf_matrix_best = results_dict[name]["conf_matrix"]
                best_f1_score = results_dict[name]["f1_score"]

                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_importance_df = (
                        pd.DataFrame({"Feature": feature_names, "Importance": importance})
                        .sort_values(by="Importance", ascending=False)
                        .reset_index(drop=True)
                    )

    # Train Random Forest models
    for n_estimators in rf_params["n_estimators"]:
        for max_depth in rf_params["max_depth"]:
            name = f"Random Forest (n={n_estimators}, depth={max_depth})"
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            start_time = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start_time

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            try:
                y_scores = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_scores = model.decision_function(X_test)

            results_dict[name] = {
                "model": model,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, zero_division=0),
                "recall": recall_score(y_test, y_test_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
                "auc": roc_auc_score(y_test, y_scores),
                "training_time": duration,
                "y_pred": y_test_pred,
                "conf_matrix": confusion_matrix(y_test, y_test_pred),
            }

            if results_dict[name]["f1_score"] > best_f1_score:
                best_model_name = name
                best_model_object = model
                y_pred_best = y_test_pred
                conf_matrix_best = results_dict[name]["conf_matrix"]
                best_f1_score = results_dict[name]["f1_score"]

                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_importance_df = (
                        pd.DataFrame({"Feature": feature_names, "Importance": importance})
                        .sort_values(by="Importance", ascending=False)
                        .reset_index(drop=True)
                    )

    return (
        results_dict,
        best_model_name,
        best_model_object,
        y_pred_best,
        conf_matrix_best,
        feature_importance_df,
    )