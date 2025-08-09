import os
from datetime import datetime
from treeinterpreter import treeinterpreter as ti

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px

# Import helper functions
from helper import show_dataset_info, run_preprocessing, render_preprocessing_steps, train_and_evaluate_models

def load_css(css_file: str) -> None:
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{css_file}' not found. Styles may not apply correctly.")

st.set_page_config(layout="wide",page_title="Loan Predict@G5", initial_sidebar_state="auto")

# --- Load Dataset ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data("Loan_default.csv")

numeric_cols = (df.select_dtypes(include=["int64", "float64"]).drop(columns=["Default"], errors="ignore").columns.tolist())
cat_cols = df.select_dtypes(include="object").columns.tolist()


def data_overview() -> None:
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
        if "Default" in df.columns:
            approval_rate = (df["Default"] == 0).mean() * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%", "Historical approval rate")
    with col4:
        st.metric("Last Updated", last_updated, "Data freshness")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📐 Statistical Summary", "📊 Distributions", "🔗 Correlations"])

    # Statistical summary on tab1
    with tab1:
        st.subheader("📈 Numerical Features Summary")
        st.caption("Descriptive statistics for numerical variables")

        # Get describe output and transpose
        stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T

        # Rename and reorder columns
        stats = stats.rename(
            columns={
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Min",
                "25%": "Q1",
                "50%": "Median",
                "75%": "Q3",
                "max": "Max",
            }
        )[["Mean", "Median", "Std Dev", "Min", "Q1", "Q3", "Max"]]

        # Round and reset index for clean display
        stats = stats.round(1).reset_index().rename(columns={"index": "Feature"})

        st.dataframe(stats, use_container_width=True)

        st.markdown("### 📦 Box Plot Analysis")
        st.caption("Distribution and outlier analysis for key numerical features")

        # Create 3 columns
        col1b, col2b, col3b = st.columns(3)
        columns = [col1b, col2b, col3b]

        # Render a selectbox + boxplot + stats in each column
        for i, col in enumerate(columns):
            with col:
                feature = st.selectbox(
                    f"Feature {i + 1}", options=numeric_cols, index=i, key=f"box-feature-{i}"
                )

                # Plot
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.boxplot(
                    df[feature].dropna(),
                    vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="#E0ECF8", color="#4F81BD"),
                    medianprops=dict(color="red"),
                )
                ax.set_title(feature, fontsize=10)
                ax.set_xticks([])

                st.pyplot(fig)

                # Stats
                desc = df[feature].describe(percentiles=[0.25, 0.5, 0.75]).round(1)
                st.markdown(
                    f"""
                    <div style="font-size: 13px;">
                    <strong>Min:</strong> {desc['min']}<br>
                    <strong>Q1:</strong> {desc['25%']}<br>
                    <strong>Median:</strong> {desc['50%']}<br>
                    <strong>Q3:</strong> {desc['75%']}<br>
                    <strong>Max:</strong> {desc['max']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Distributions tab
    with tab2:
        st.caption("Frequency distribution of key numerical features")

        col1d, col2d = st.columns(2)
        selected_features = numeric_cols[:2]

        for i, col in enumerate([col1d, col2d]):
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

    # Correlations tab
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
            ax=ax,
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
                "text": "There are no strong positive correlations in this dataset because the absolute values are all very close to zero. This suggests that none of the numerical features vary strongly together. In practical terms, features like Income and LoanAmount or CreditScore and LoanTerm are essentially independent in this dataset.",
                "color": "#e8f0fe",
                "title_color": "#1967d2",
            },
            {
                "title": "Asset-Income Relationship",
                "text": "If we treat Income as a proxy for assets, the correlation values between Income and all other variables are near zero (e.g., Income–LoanAmount ≈ –0.00087, Income–MonthsEmployed ≈ 0.0027). This implies that higher income does not consistently align with larger loan amounts or longer employment durations in this data — at least not in a linear sense.",
                "color": "#e6f4ea",
                "title_color": "#137333",
            },
            {
                "title": "Independence Noted",
                "text": "The consistently low correlation magnitudes (close to 0) between most pairs, such as CreditScore and Income (≈ –0.0014) or Age and LoanAmount (≈ –0.0022), indicate that these variables behave independently. This suggests that each feature may contribute unique, uncorrelated information to the model, which can be useful for machine learning algorithms that benefit from independent predictors.",
                "color": "#fef7e0",
                "title_color": "#d39e00",
            },
        ]

        for block in insight_blocks:
            st.markdown(
                f"""
                <div style="background-color:{block['color']}; padding:15px; border-radius:10px; margin-bottom:12px;">
                    <h5 style="color:{block['title_color']}; margin-bottom:5px;">{block['title']}</h5>
                    <p style="margin:0;">{block['text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("___")

    # Show dataset info button
    if st.button("📊 View Dataset Information"):
        show_dataset_info(df)


def preprocessing_page() -> None:
    st.title("Preprocessing Pipeline")
    st.write("Click the button below to run preprocessing on the dataset.")

    if st.button("🚀 Run Preprocessing"):
        run_preprocessing(df)

    if st.session_state.get("preprocessing_done"):
        render_preprocessing_steps()


def model_page() -> None:
    st.title("Model Training & Evaluation")
    st.caption("Comprehensive model development and performance analysis")

    # Ensure preprocessing has been run
    if not st.session_state.get("preprocessing_done"):
        st.warning("⚠️ Please complete preprocessing before training models.")
        return

    # Trigger training if not already in session state
    if "results_dict" not in st.session_state:
        if st.button("🚀 Train Models", type="primary"):
            with st.status("Training models...", expanded=True) as status:
                def log(msg: str):
                    status.write(msg)  # each stage & model shows with elapsed seconds

                results_dict, best_name, best_model, y_pred_best, cm, feat_imp_df = train_and_evaluate_models(
                    st.session_state["X_train"],
                    st.session_state["y_train"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                    st.session_state["preprocessor"].get_feature_names_out(),
                    on_step=log,  # 👈 feed progress messages into the status box
                )

                st.session_state.update({
                    "trained_models": {name: res["model"] for name, res in results_dict.items()},
                    "results_dict": results_dict,
                    "best_model_name": best_name,
                    "best_model": best_model,
                    "y_pred_best": y_pred_best,
                    "conf_matrix": cm,
                    "feature_importance_df": feat_imp_df,
                })

                status.update(label=f"✅ Model training complete! Best: {best_name}", state="complete", expanded=False)
                st.success(f"✅ Models trained. Best model: {best_name}")

        return

    # Extract results from session state
    results_dict = st.session_state["results_dict"]
    best_name = st.session_state["best_model_name"]
    cm = st.session_state["conf_matrix"]
    feat_imp_df = st.session_state["feature_importance_df"]
    best = results_dict[best_name]

    model_type = best_name.split(" (")[0]
    model_params = best_name.split(" (")[1].replace(")", "")

    # Metric cards
    st.markdown(
        """
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
            model_type,
            model_params,
            best["test_accuracy"] * 100,
            best["training_time"],
            best["auc"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 🧮 Model Performance Comparison")

    # Build performance table
    table_data = []
    for name, res in results_dict.items():
        status = "best" if name == best_name else "good"
        table_data.append(
            {
                "Model": name,
                "Accuracy": f"{res['test_accuracy'] * 100:.1f}%",
                "Precision": f"{res['precision']:.2f}",
                "Recall": f"{res['recall']:.2f}",
                "F1-Score": f"{res['f1_score']:.2f}",
                "AUC": f"{res['auc']:.2f}",
                "Status": status,
            }
        )

    df_perf = pd.DataFrame(table_data)

    def style_status(val):
        color = {"best": "#000", "good": "#999"}.get(val, "#CCC")
        return f'background-color:{color}; color:white; border-radius:6px; padding:2px 8px; font-weight:bold'

    st.dataframe(
        df_perf.style.applymap(style_status, subset=["Status"]), use_container_width=True
    )

    # Confusion matrix breakdown
    st.markdown("### 🧮 Confusion Matrix Breakdown")

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    st.markdown(
        """
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
    """.format(tp, fn, fp, tn),
        unsafe_allow_html=True,
    )

    # Chart view toggle
    with st.expander("📈 View Confusion Matrix Chart"):
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix Heatmap")
        st.pyplot(fig)

    # Feature importance
    if feat_imp_df is not None:
        st.markdown("### 📉 Feature Importance")
        fig = px.bar(
            feat_imp_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Influential Features",
            height=400,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Training configuration summary
    st.markdown(
        """
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
            st.session_state["results_dict"][st.session_state["best_model_name"]]["training_time"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### 🌳 Model Visualization")

    selected_model_name = st.selectbox(
        "Select a model to visualize", list(st.session_state["trained_models"].keys())
    )

    model_to_plot = st.session_state["trained_models"][selected_model_name]
    try:
        feature_names = st.session_state["preprocessor"].get_feature_names_out()
    except:
        feature_names = list(range(st.session_state["X_train"].shape[1]))

    class_names = ["No Default", "Default"]

    # Default settings for readability
    depth = st.slider("Tree Depth to Display", 1, 6, value=3)
    figsize = st.selectbox(
        "Figure Width",
        options=[(16, 8), (20, 10), (28, 14)],
        format_func=lambda x: f"{x[0]} x {x[1]}",
    )

    # Random Forest handling
    if hasattr(model_to_plot, "estimators_"):
        st.info(
            "ℹ️ Random Forest is an ensemble of decision trees. You can select and visualize one of its individual trees."
        )

        index = st.slider(
            "Select Tree Index", 0, len(model_to_plot.estimators_) - 1, 0
        )
        tree = model_to_plot.estimators_[index]

        fig, ax = plt.subplots(figsize=figsize)
        from sklearn.tree import plot_tree as sk_plot_tree  # local import to avoid circular

        sk_plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=depth,
            fontsize=10,
            ax=ax,
        )
        st.pyplot(fig)

    # Decision Tree handling
    elif hasattr(model_to_plot, "tree_"):
        fig, ax = plt.subplots(figsize=figsize)
        from sklearn.tree import plot_tree as sk_plot_tree  # local import to avoid circular

        sk_plot_tree(
            model_to_plot,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=depth,
            fontsize=10,
            ax=ax,
        )
        st.pyplot(fig)
    else:
        st.warning("⚠️ This model type cannot be visualized as a tree.")


def prediction():
    if "preprocessor" not in st.session_state or "trained_models" not in st.session_state:
        st.warning("⚠️ Please train at least one model before using the prediction page.")
        return

    # Load schema
    df_meta = pd.read_csv("Loan_default.csv")
    X_meta = df_meta.drop(columns=["LoanID", "Default"], errors="ignore")
    numeric_cols = X_meta.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_meta.select_dtypes(include="object").columns.tolist()

    # User-friendly labels
    friendly_labels = {
        "Income": "Monthly Income (local currency)",
        "CreditScore": "Credit Score",
        "LoanAmount": "Loan Amount Requested",
        "LoanTerm": "Loan Term (months)",
        "Age": "Applicant Age",
        "MonthsEmployed": "Months Employed",
        "Gender": "Gender",
        "EducationLevel": "Education Level",
        "MaritalStatus": "Marital Status",
        "EmploymentType": "Employment Type",
        "PropertyArea": "Property Area",
        "DTIRatio": "Debt-to-Income Ratio",
        "NumCreditLines": "Number of Credit Lines",
        "InterestRate": "Interest Rate",
        "HasDependents": "Has Dependents",
        "HasCoSigner": "Has Co-Signer",
        "LoanPurpose": "Loan Purpose"
    }

    with st.form("loan_form"):
        st.markdown("<h2 class='section-title'>📝 Loan Application Form</h2>", unsafe_allow_html=True)
        st.caption("Select a model, then fill in your details to get a prediction.")

        # Model selection
        model_choice = st.selectbox(
            "Select a model for detailed analysis",
            options=list(st.session_state["trained_models"].keys()),
            index=0
        )

        col1, col2 = st.columns(2)
        user_input = {}

        with col1:
            for col in cat_cols[: len(cat_cols) // 2]:
                label = friendly_labels.get(col, col.replace("_", " "))
                options = sorted(df_meta[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(label, options=options, key=f"cat1_{col}")
            for col in numeric_cols[: len(numeric_cols) // 2]:
                label = friendly_labels.get(col, col.replace("_", " "))
                user_input[col] = st.number_input(label, placeholder=f"Enter {label.lower()}", key=f"num1_{col}")

        with col2:
            for col in cat_cols[len(cat_cols) // 2:]:
                label = friendly_labels.get(col, col.replace("_", " "))
                options = sorted(df_meta[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(label, options=options, key=f"cat2_{col}")
            for col in numeric_cols[len(numeric_cols) // 2:]:
                label = friendly_labels.get(col, col.replace("_", " "))
                user_input[col] = st.number_input(label, placeholder=f"Enter {label.lower()}", key=f"num2_{col}")

        submitted = st.form_submit_button("🔍 Predict", use_container_width=False)

    if not submitted:
        return

    # Transform input
    input_df = pd.DataFrame([user_input])
    X_trans = st.session_state["preprocessor"].transform(input_df)
    try:
        X_dense = X_trans.toarray()
    except AttributeError:
        X_dense = np.asarray(X_trans)

    feature_names = st.session_state["preprocessor"].get_feature_names_out()

    # --- Probability comparison for all models ---
    results_table = []
    for name, mdl in st.session_state["trained_models"].items():
        proba = mdl.predict_proba(X_dense)[0]
        approved_prob = proba[0]
        risk_prob = proba[1]
        pred_class = mdl.predict(X_dense)[0]
        results_table.append({
            "Model": name,
            "Approval Probability": f"{approved_prob*100:.1f}%",
            "Risk Probability": f"{risk_prob*100:.1f}%",
            "Predicted Class": "Approved" if pred_class == 0 else "Not Approved"
        })

    st.markdown("## 📊 Model Prediction Comparison")
    st.caption("See how each trained model scores this application.")
    st.dataframe(pd.DataFrame(results_table), use_container_width=True)

    # --- Detailed breakdown for selected model ---
    model = st.session_state["trained_models"][model_choice]
    pred_label = int(model.predict(X_dense)[0])
    confidence = float(model.predict_proba(X_dense)[0][pred_label])

    st.markdown("## 🎯 Selected Model Result")
    if pred_label == 0:
        st.markdown(f"<div class='result-card approved'>Loan Approved ✅<br><span>{confidence*100:.1f}% confidence</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-card declined'>Loan Not Approved – Risk of Default ⚠️<br><span>{confidence*100:.1f}% confidence</span></div>", unsafe_allow_html=True)

    # Feature contributions chart
    prediction_vec, bias, contributions = ti.predict(model, X_dense)
    pred_idx = list(model.classes_).index(pred_label)
    contrib_for_pred = contributions[0, :, pred_idx]

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contrib_for_pred,
        "AbsContribution": np.abs(contrib_for_pred),
    }).sort_values("AbsContribution", ascending=False)
    top_k = contrib_df.head(10).drop(columns=["AbsContribution"])

    st.markdown("## 📌 Top Factors (Selected Model)")
    st.caption("Positive values push **towards** this outcome; negative values push **against** it.")
    fig = px.bar(
        top_k.sort_values("Contribution"),
        x="Contribution",
        y="Feature",
        orientation="h",
        height=420,
        color="Contribution",
        color_continuous_scale=["#d73027", "#1a9850"]
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

def main() -> None:
    """Main application logic: handles page navigation and styling."""
    # Inject custom CSS
    load_css("styles.css")

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
            <h2 style='margin-bottom:0;'>Loan Predictor</h2>
            <p style='margin-top:0; color: gray;'>ML-Powered Decision Engine: <span><i>A Group 5 Dashboard</i></span></p>
            <hr style='margin-top:10px; margin-bottom:10px;'>
            """,
            unsafe_allow_html=True,
        )

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
            default_index=1,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "black", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {
                    "background-color": "#ffffff",
                    "font-weight": "bold",
                    "color": "#000000",
                },
            },
        )

    # Page routing
    if selected_page == "Loan Prediction":
        prediction()
    elif selected_page == "Data Overview":
        data_overview()
    elif selected_page == "Preprocessing":
        preprocessing_page()
    elif selected_page == "Model Training & Evaluation":
        model_page()

if __name__ == "__main__":
    main()

st.sidebar.markdown("___")
st.sidebar.selectbox("Tap to View All Group Members",
    ("Eleazer F. Quayson (22253333)", "Priscilla D. Gborbitey (22253220)",
     "Magdalene Arhin (22253225)", "Anna E.A Creppy (11410565)",
     "Raymond Tetteh - 22255065", "Samuel K. Tuffour (22253144"))

