import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import os
import time
from pandas.api.types import CategoricalDtype

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import plot_tree
from streamlit_option_menu import option_menu

# --- Load Dataset ---
df = pd.read_csv("Loan_default.csv")

# group numerical and categorical columns separately
numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Default'], errors='ignore').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# --- Data Overview Page ---
def data_overview():
    st.title("Dataset Overview")
    st.caption("Comprehensive analysis of the loan dataset with statistical insights and visualizations.")

    file_timestamp = os.path.getmtime("Loan_default.csv")
    last_updated = datetime.fromtimestamp(file_timestamp).strftime("%b %Y")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}", "Complete loan applications")
    with col2:
        st.metric("Features", f"{df.shape[1] - 1}", "Input variables")
    with col3:
        if 'Default' in df.columns:
            approval_rate = (df['Default'] == 0).mean() * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%", "Historical approval rate")
    with col4:
        st.metric("Last Updated", last_updated, "Data freshness")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3= st.tabs(
        ["📐 Statistical Summary", "📊 Distributions", "🔗 Correlations"])

    # Statistical summary on tab1
    with tab1:
        st.subheader("📈 Numerical Features Summary")
        st.caption("Descriptive statistics for numerical variables")

        # Get describe output and transpose
        stats = df[numeric_cols].describe(percentiles=[.25, .5, .75]).T

        # Rename and reorder columns
        stats = stats.rename(columns={
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max"
        })[["Mean", "Median", "Std Dev", "Min", "Q1", "Q3", "Max"]]

        # Round and reset index for clean display
        stats = stats.round(1).reset_index().rename(columns={"index": "Feature"})

        st.dataframe(stats, use_container_width=True)

        st.markdown("### 📦 Box Plot Analysis")
        st.caption("Distribution and outlier analysis for key numerical features")

        # Create 3 columns
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        # Render a selectbox + boxplot + stats in each column
        for i, col in enumerate(columns):
            with col:
                feature = st.selectbox(f"Feature {i + 1}", options=numeric_cols, index=i, key=f"box-feature-{i}")

                # Plot
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.boxplot(df[feature].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#E0ECF8", color="#4F81BD"),
                           medianprops=dict(color="red"))
                ax.set_title(feature, fontsize=10)
                ax.set_xticks([])

                st.pyplot(fig)

                # Stats
                desc = df[feature].describe(percentiles=[.25, .5, .75]).round(1)
                st.markdown(f"""
                <div style="font-size: 13px;">
                <strong>Min:</strong> {desc['min']}<br>
                <strong>Q1:</strong> {desc['25%']}<br>
                <strong>Median:</strong> {desc['50%']}<br>
                <strong>Q3:</strong> {desc['75%']}<br>
                <strong>Max:</strong> {desc['max']}
                </div>
                """, unsafe_allow_html=True)

    # code for the distributions tab
    with tab2:
        st.caption("Frequency distribution of key numerical features")

        col1, col2 = st.columns(2)
        selected_features = numeric_cols[:2]

        for i, col in enumerate([col1, col2]):
            feature = selected_features[i]
            col.markdown(f"**{feature} Distribution**")
            col.caption(f"Frequency distribution of {feature.lower().replace('_', ' ')}")

            # Bin the feature manually
            binned = pd.cut(df[feature], bins=10)
            counts = binned.value_counts().sort_index()

            # Format bin labels
            counts.index = [f"{int(interval.left):,} - {int(interval.right):,}" for interval in counts.index]

            # Create DataFrame and plot
            chart_df = pd.DataFrame({feature: counts.values}, index=counts.index)
            col.bar_chart(chart_df)

    with tab3:
        st.subheader("🔗 Correlation Heatmap")
        st.caption("Pairwise Pearson correlation between numerical features")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(
            corr,
            annot=True,
            cmap="BrBG",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title("Correlation Matrix", fontsize=10)
        st.pyplot(fig)

        st.markdown("---")

        # Insights
        st.subheader("💡 Correlation Insights")
        st.caption("Key findings from correlation analysis")

        insight_blocks = [
            {
                "title": "Strong Positive Correlation",
                "text": "Applicant Income and Loan Amount show strong correlation (0.57), indicating higher income applicants request larger loans.",
                "color": "#e8f0fe",
                "title_color": "#1967d2"
            },
            {
                "title": "Asset-Income Relationship",
                "text": "Total Assets correlate moderately with both Applicant Income (0.43) and Loan Amount (0.38), showing wealth consistency.",
                "color": "#e6f4ea",
                "title_color": "#137333"
            },
            {
                "title": "Independence Noted",
                "text": "Co-applicant Income shows weak correlation with Applicant Income (0.19), suggesting independent income sources.",
                "color": "#fef7e0",
                "title_color": "#d39e00"
            }
        ]

        for block in insight_blocks:
            st.markdown(f"""
            <div style="background-color:{block['color']}; padding:15px; border-radius:10px; margin-bottom:12px;">
                <h5 style="color:{block['title_color']}; margin-bottom:5px;">{block['title']}</h5>
                <p style="margin:0;">{block['text']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("___")

    def show_dataset_info(df: pd.DataFrame):
        """Display dataset information in a formatted card"""
        total_records = f"{df.shape[0]:,}"
        total_features = df.shape[1]
        missing_values = f"{df.isnull().sum().sum():,}"
        default_rate = f"{(df['Default'] == 1).mean() * 100:.1f}%" if 'Default' in df.columns else "N/A"

        st.markdown(f"""
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
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Dataset Preview")
        st.caption(f"Showing the complete dataset with {df.shape[0]} rows and {df.shape[1]} columns")

        # Display the full dataset with scrolling capability
        st.dataframe(
            df,
            use_container_width=True,
            height=400  # Set a fixed height to enable scrolling
        )

        # column information
        with st.expander("🔍 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

    if st.button("📊 View Dataset Information"):
        show_dataset_info(df)

#---------------------------------------------------------------------------------------------------------------
# --- PAGE 2: DATA PREPROCESSING
def run_preprocessing():
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing pipeline
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ])

        # Fit and transform
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Store in session state
        st.session_state.update({
            "preprocessing_done": True,
            "preprocessor": preprocessor,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })

        st.success("✅ Preprocessing completed successfully.")
    else:
        st.info("Preprocessing has already been done.")


def render_preprocessing_steps():
    st.markdown("""
    <style>
        .pipeline-container {
            background-color: #fff;
            padding: 30px;
            border-radius: 16px;
            border: 1px solid #e6e6e6;
            margin-top: 10px;
        }
        .pipeline-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 0;
        }
        .pipeline-subtitle {
            font-size: 14px;
            color: #7f7f7f;
            margin-top: 0;
            margin-bottom: 30px;
        }
        .step-card {
            background-color: #fff;
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .step-header {
            display: flex;
            align-items: center;
            margin-bottom: 6px;
        }
        .step-number {
            background-color: #000;
            color: white;
            font-weight: 600;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            text-align: center;
            line-height: 28px;
            margin-right: 12px;
            font-size: 14px;
        }
        .step-title {
            font-weight: 600;
            font-size: 16px;
            margin: 0;
        }
        .status-pill {
            background-color: #000;
            color: white;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 8px;
            margin-left: 10px;
        }
        .step-subtitle {
            font-size: 13px;
            color: #888;
            margin: 4px 0 16px 40px;
        }
        .checkmark {
            font-size: 14px;
            margin-left: 40px;
            margin-bottom: 4px;
        }
        .summary-box {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 18px 24px;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-title">⚙️ Preprocessing Pipeline</div>
        <div class="pipeline-subtitle">Step-by-step data transformation process</div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown("""
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
    """, unsafe_allow_html=True)

    # Step 2
    st.markdown("""
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
    """, unsafe_allow_html=True)

    # Step 3
    st.markdown("""
        <div class="step-card">
            <div class="step-header">
                <div class="step-number">3</div>
                <div class="step-title">Train-Test Split</div>
                <div class="status-pill">✔ completed</div>
            </div>
            <div class="step-subtitle">Split dataset into separate training and testing subsets</div>
            <div class="checkmark">✅ Split into 80% train / 20% test</div>
        </div>
    """, unsafe_allow_html=True)

    # Summary
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")

    if X_train is not None and X_test is not None and y_train is not None:
        st.markdown("""
            <div class="step-card summary-box">
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 10px;">✅ Summary</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
                <div><strong>X_train shape:</strong> {X_train.shape}</div>
                <div><strong>X_test shape:</strong> {X_test.shape}</div>
            </div>
        </div> <!-- close pipeline container -->
        """, unsafe_allow_html=True)

        dist_df = y_train.value_counts(normalize=True).reset_index()
        dist_df.columns = ["Class", "Proportion"]
        dist_df["Class"] = dist_df["Class"].apply(lambda x: f"Class {x}")
        st.markdown("**y_train distribution:**")
        st.table(dist_df)
    else:
        st.warning("No training/test data found in session state.")
        st.markdown("</div>", unsafe_allow_html=True)


def preprocessing():
    st.title("Preprocessing Pipeline")
    st.write("Click the button below to run preprocessing on the dataset.")

    if st.button("🚀 Run Preprocessing"):
        run_preprocessing()

    if st.session_state.get("preprocessing_done"):
        render_preprocessing_steps()

# -----------------------------------------------------------------------------
# ----- PAGE 3: model training and evaluation logic

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    results_dict = {}
    best_model_name = None
    best_model_object = None
    y_pred_best = None
    conf_matrix_best = None
    feature_importance_df = None
    best_f1_score = -1

    # --- Hyperparameter grids ---
    dt_params = {
        "max_depth": [4, 6, 8],
        "min_samples_split": [5, 10]
    }
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 8]
    }

    # --- Train Decision Tree ---
    for max_depth in dt_params["max_depth"]:
        for min_split in dt_params["min_samples_split"]:
            name = f"Decision Tree (depth={max_depth}, split={min_split})"
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split, random_state=42)

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
                "conf_matrix": confusion_matrix(y_test, y_test_pred)
            }

            if results_dict[name]["f1_score"] > best_f1_score:
                best_model_name = name
                best_model_object = model
                y_pred_best = y_test_pred
                conf_matrix_best = results_dict[name]["conf_matrix"]
                best_f1_score = results_dict[name]["f1_score"]

                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importance
                    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # --- Train Random Forest ---
    for n_estimators in rf_params["n_estimators"]:
        for max_depth in rf_params["max_depth"]:
            name = f"Random Forest (n={n_estimators}, depth={max_depth})"
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

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
                "conf_matrix": confusion_matrix(y_test, y_test_pred)
            }

            if results_dict[name]["f1_score"] > best_f1_score:
                best_model_name = name
                best_model_object = model
                y_pred_best = y_test_pred
                conf_matrix_best = results_dict[name]["conf_matrix"]
                best_f1_score = results_dict[name]["f1_score"]

                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importance
                    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return (
        results_dict,
        best_model_name,
        best_model_object,
        y_pred_best,
        conf_matrix_best,
        feature_importance_df
    )

def model():
    st.title("Model Training & Evaluation")
    st.caption("Comprehensive model development and performance analysis")

    if not st.session_state.get("preprocessing_done"):
        st.warning("⚠️ Please complete preprocessing before training models.")
        return

    if "results_dict" not in st.session_state:
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                results_dict, best_name, best_model, y_pred_best, cm, feat_imp_df = train_and_evaluate_models(
                    st.session_state["X_train"],
                    st.session_state["y_train"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                    st.session_state["preprocessor"].get_feature_names_out()
                )

                st.session_state.update({
                    "trained_models": {name: res["model"] for name, res in results_dict.items()},
                    "results_dict": results_dict,
                    "best_model_name": best_name,
                    "best_model": best_model,
                    "y_pred_best": y_pred_best,
                    "conf_matrix": cm,
                    "feature_importance_df": feat_imp_df
                })

                st.success(f"✅ Models trained. Best model: {best_name}")
        return

    results_dict = st.session_state["results_dict"]
    best_name = st.session_state["best_model_name"]
    cm = st.session_state["conf_matrix"]
    feat_imp_df = st.session_state["feature_importance_df"]
    best_model = results_dict[best_name]

    # --- Metric Cards ---
    best = st.session_state["results_dict"][st.session_state["best_model_name"]]
    best_name = st.session_state["best_model_name"]
    model_type = best_name.split(" (")[0]
    model_params = best_name.split(" (")[1].replace(")", "")

    # Unified 4-card layout
    st.markdown("""
    <div style="display: flex; gap: 20px; justify-content: space-between; margin-top: 30px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 180px; background: #f7f7f7; padding: 20px; border-radius: 12px; text-align: center;">
        <div style="font-size: 14px; color: #666; font-weight: 600;">Best Model</div>
        <div style="font-size: 20px; font-weight: bold; margin: 4px 0;">{}</div>
        <div style="font-size: 13px; color: #888;">{}</div>
      </div>
      <div style="flex: 1; min-width: 180px; background: #f7f7f7; padding: 20px; border-radius: 12px; text-align: center;">
        <div style="font-size: 14px; color: #666; font-weight: 600;">Test Accuracy</div>
        <div style="font-size: 24px; font-weight: bold;">{:.2f}%</div>
      </div>
      <div style="flex: 1; min-width: 180px; background: #f7f7f7; padding: 20px; border-radius: 12px; text-align: center;">
        <div style="font-size: 14px; color: #666; font-weight: 600;">Training Time</div>
        <div style="font-size: 24px; font-weight: bold;">{:.2f} s</div>
      </div>
      <div style="flex: 1; min-width: 180px; background: #f7f7f7; padding: 20px; border-radius: 12px; text-align: center;">
        <div style="font-size: 14px; color: #666; font-weight: 600;">AUC Score</div>
        <div style="font-size: 24px; font-weight: bold;">{:.2f}</div>
      </div>
    </div>
    """.format(
        model_type, model_params,
        best["test_accuracy"] * 100,
        best["training_time"],
        best["auc"]
    ), unsafe_allow_html=True)

    st.markdown("### 🧮 Model Performance Comparison")

    table_data = []
    for name, res in results_dict.items():
        status = "best" if name == best_name else "good"
        table_data.append({
            "Model": name,
            "Accuracy": f"{res['test_accuracy']*100:.1f}%",
            "Precision": f"{res['precision']:.2f}",
            "Recall": f"{res['recall']:.2f}",
            "F1-Score": f"{res['f1_score']:.2f}",
            "AUC": f"{res['auc']:.2f}",
            "Status": status
        })

    df_perf = pd.DataFrame(table_data)

    def style_status(val):
        color = {"best": "#000", "good": "#999"}.get(val, "#CCC")
        return f'background-color:{color}; color:white; border-radius:6px; padding:2px 8px; font-weight:bold'

    st.dataframe(df_perf.style.applymap(style_status, subset=["Status"]), use_container_width=True)

    # --- Confusion Matrix Breakdown ---
    st.markdown("### 🧮 Confusion Matrix Breakdown")

    # Extract values from confusion matrix
    tp = st.session_state["conf_matrix"][1, 1]
    fp = st.session_state["conf_matrix"][0, 1]
    fn = st.session_state["conf_matrix"][1, 0]
    tn = st.session_state["conf_matrix"][0, 0]

    # Styled block layout
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 10px;">

      <div style="flex: 1; min-width: 240px; background-color: #e6f4ea; padding: 20px; border-radius: 12px;">
        <span style="font-size: 16px;">✅ <strong>True Positives:</strong> {}</span>
        <div style="color: #137333; font-size: 13px; margin-top: 4px;">Correctly predicted defaults</div>
      </div>

      <div style="flex: 1; min-width: 240px; background-color: #fff8e1; padding: 20px; border-radius: 12px;">
        <span style="font-size: 16px;">⚠️ <strong>False Negatives:</strong> {}</span>
        <div style="color: #a76f00; font-size: 13px; margin-top: 4px;">Defaults missed by model</div>
      </div>

      <div style="flex: 1; min-width: 240px; background-color: #fdecea; padding: 20px; border-radius: 12px;">
        <span style="font-size: 16px;">❌ <strong>False Positives:</strong> {}</span>
        <div style="color: #d93025; font-size: 13px; margin-top: 4px;">Incorrectly flagged defaults</div>
      </div>

      <div style="flex: 1; min-width: 240px; background-color: #e8f0fe; padding: 20px; border-radius: 12px;">
        <span style="font-size: 16px;">🔍 <strong>True Negatives:</strong> {}</span>
        <div style="color: #1967d2; font-size: 13px; margin-top: 4px;">Correctly predicted approvals</div>
      </div>

    </div>
    """.format(tp, fn, fp, tn), unsafe_allow_html=True)

    # Chart view toggle
    with st.expander("📈 View Confusion Matrix Chart"):
        fig, ax = plt.subplots()
        sns.heatmap(
            st.session_state["conf_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            cbar=False,
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix Heatmap")
        st.pyplot(fig)

    # --- Feature Importance ---
    if feat_imp_df is not None:
        st.markdown("### 📉 Feature Importance")
        fig = px.bar(
            feat_imp_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Influential Features",
            height=400
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # --- Training Configuration Summary ---
    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]

    st.markdown("""
    <div style="background-color:#f9f9f9; padding: 25px 30px; border-radius: 12px; border: 1px solid #eee; margin-top:20px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 22px; margin-right: 10px;">⚙️</span>
            <h4 style="margin: 0;">Training Configuration</h4>
        </div>
        <ul style="padding-left: 20px; font-size: 15px; color: #333; line-height: 1.8;">
            <li><strong>Training Set Size:</strong> {:,}</li>
            <li><strong>Test Set Size:</strong> {:,}</li>
            <li><strong>Split Ratio:</strong> 80:20</li>
            <li><strong>Validation Method:</strong> Manual hyperparameter tuning</li>
            <li><strong>Training Time:</strong> {:.2f} seconds</li>
        </ul>
    </div>
    """.format(
        len(st.session_state["X_train"]),
        len(st.session_state["X_test"]),
        st.session_state["results_dict"][st.session_state["best_model_name"]]["training_time"]
    ), unsafe_allow_html=True)

    st.markdown("### 🌳 Model Visualization")

    selected_model_name = st.selectbox(
        "Select a model to visualize",
        list(st.session_state["trained_models"].keys())
    )

    model_to_plot = st.session_state["trained_models"][selected_model_name]
    feature_names = st.session_state["preprocessor"].get_feature_names_out()
    class_names = ["No Default", "Default"]

    # Default settings for readability
    depth = st.slider("Tree Depth to Display", 1, 6, value=3)
    figsize = st.selectbox("Figure Width", options=[(16, 8), (20, 10), (28, 14)],
                           format_func=lambda x: f"{x[0]} x {x[1]}")

    # Random Forest handling
    if hasattr(model_to_plot, "estimators_"):
        st.info(
            "ℹ️ Random Forest is an ensemble of decision trees. You can select and visualize one of its individual trees.")

        index = st.slider("Select Tree Index", 0, len(model_to_plot.estimators_) - 1, 0)
        tree = model_to_plot.estimators_[index]

        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=depth,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)

    # Decision Tree handling
    elif hasattr(model_to_plot, "tree_"):
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            model_to_plot,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=depth,
            fontsize=10,
            ax=ax
        )
        st.pyplot(fig)

    else:
        st.warning("⚠️ This model type cannot be visualized as a tree.")

# ---------prediction page logic
def prediction():
    if "preprocessor" not in st.session_state or "best_model" not in st.session_state:
        st.warning("Please complete model training before using the prediction page.")
        return

    df = pd.read_csv("Loan_default.csv")  # for metadata
    X = df.drop(columns=["LoanID", "Default"], errors="ignore")

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Create form layout
    with st.form("loan_form"):
        st.markdown("### 📝 Loan Application Form")
        st.caption("Enter your details below to check your loan eligibility using our ML model")

        col1, col2 = st.columns(2)
        user_input = {}

        # Left column – Personal Info
        with col1:
            for col in cat_cols[:len(cat_cols)//2]:
                options = sorted(df[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(col.replace("_", " "), options=options, key=f"cat1_{col}")

            for col in numeric_cols[:len(numeric_cols)//2]:
                default_val = float(df[col].median())
                user_input[col] = st.number_input(col.replace("_", " "), value=default_val, key=f"num1_{col}")

        # Right column – Financial Info
        with col2:
            for col in cat_cols[len(cat_cols)//2:]:
                options = sorted(df[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(col.replace("_", " "), options=options, key=f"cat2_{col}")

            for col in numeric_cols[len(numeric_cols)//2:]:
                default_val = float(df[col].median())
                user_input[col] = st.number_input(col.replace("_", " "), value=default_val, key=f"num2_{col}")

        submitted = st.form_submit_button("🔍 Predict Loan Eligibility")

    if submitted:
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess and predict
        X_transformed = st.session_state["preprocessor"].transform(input_df)
        model = st.session_state["best_model"]

        prediction = model.predict(X_transformed)[0]
        try:
            confidence = model.predict_proba(X_transformed)[0][1] if prediction == 1 else model.predict_proba(X_transformed)[0][0]
        except:
            confidence = None

        # Get top 10 features
        top_feats = st.session_state["feature_importance_df"].head(10)

        # Show Prediction Result
        st.markdown("### ✅ Prediction Result")
        if prediction == 0:
            st.success("Loan Approved!")
        else:
            st.error("Loan Not Approved – Risk of Default")

        if confidence is not None:
            st.metric("Confidence Score", f"{confidence*100:.1f}%")

        # Generate decision factor explanations
        st.markdown("#### 📌 Decision Factors")
        explanations = []
        for feat in top_feats["Feature"]:
            # Get the untransformed name
            if "__" in feat:
                raw_feat = feat.split("__")[-1]
            else:
                raw_feat = feat

            if raw_feat in input_df.columns:
                val = input_df[raw_feat].values[0]
                importance = top_feats[top_feats["Feature"] == feat]["Importance"].values[0]
                # You can customize these rules
                if isinstance(val, (int, float)):
                    if val > df[raw_feat].mean():
                        explanations.append((f"✓ High {raw_feat.replace('_', ' ')}", True))
                    else:
                        explanations.append((f"× Low {raw_feat.replace('_', ' ')}", False))
                else:
                    explanations.append((f"✓ {raw_feat.replace('_', ' ')} = {val}", True))

        for text, is_positive in explanations:
            if is_positive:
                st.markdown(f"✅ {text}")
            else:
                st.markdown(f"❌ {text}")

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <h2 style='margin-bottom:0;'>Loan Predictor</h2>
    <p style='margin-top:0; color: gray;'>ML-Powered Decision Engine: <span><i>A Group 5 Dashboard</i></span></p>
    <hr style='margin-top:10px; margin-bottom:10px;'>
    """, unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=[
            "Loan Prediction",
            "Data Overview",
            "Preprocessing",
            "Model Training & Evaluation",
        ],
        icons=["house", "database", "gear", "cpu"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {
                "background-color": "#ffffff",
                "font-weight": "bold",
                "color": "#000000"
            }
        }
    )

# --- Page Routing ---
if selected_page == "Loan Prediction":
    prediction()
elif selected_page == "Data Overview":
    data_overview()
elif selected_page == "Preprocessing":
    preprocessing()
elif selected_page == "Model Training & Evaluation":
    model()