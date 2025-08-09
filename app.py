import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

# tree viz & models
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    average_precision_score, roc_curve, precision_recall_curve
)

# --- helper functions
from helper import (
    show_dataset_info,
    run_preprocessing,
    render_preprocessing_steps,
    train_and_evaluate_models,
    _safe_scores,
    _approval_curve,
    _recommend_threshold_by_cap,
)

import logging, sys, traceback
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
import streamlit as st
st.set_option("client.showErrorDetails", True)


# ----------------- page config -----------------
st.set_page_config(layout="wide", page_title="Loan Predict@G5", initial_sidebar_state="auto")

def load_css(css_file: str) -> None:
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{css_file}' not found. Styles may not apply correctly.")

# ----------------- data cache -----------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data("Loan_default.csv")

# --- global policy state ---
if "policy_threshold" not in st.session_state:
    st.session_state["policy_threshold"] = 0.50  # risk-side threshold (approve if risk < threshold)
if "policy_mode" not in st.session_state:
    st.session_state["policy_mode"] = "business"   # "business" or "manual"
if "policy_appetite" not in st.session_state:
    st.session_state["policy_appetite"] = "Balanced (risk vs volume)"

# global type lists (excluding target)
numeric_cols_global = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Default"], errors="ignore").columns.tolist()
cat_cols_global = df.select_dtypes(include="object").columns.tolist()

# ----------------- pages -----------------
def data_overview() -> None:
    st.title("Dataset Overview")
    st.caption("Comprehensive analysis of the loan dataset with statistical insights and visualizations.")

    try:
        file_timestamp = os.path.getmtime("Loan_default.csv")
        last_updated = datetime.fromtimestamp(file_timestamp).strftime("%b %Y")
    except Exception:
        last_updated = "N/A"

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
    tab1, tab2, tab3 = st.tabs(["📐 Statistical Summary", "📊 Distributions", "🔗 Correlations"])

    with tab1:
        st.subheader("📈 Numerical Features Summary")
        stats = df[numeric_cols_global].describe(percentiles=[0.25, 0.5, 0.75]).T.rename(
            columns={"mean": "Mean","std": "Std Dev","min": "Min","25%": "Q1","50%": "Median","75%": "Q3","max": "Max"}
        )[["Mean","Median","Std Dev","Min","Q1","Q3","Max"]].round(1).reset_index().rename(columns={"index": "Feature"})
        st.dataframe(stats, use_container_width=True)

        st.markdown("### 📦 Box Plot Analysis")
        col1b, col2b, col3b = st.columns(3)
        for i, col in enumerate([col1b, col2b, col3b]):
            with col:
                feature = st.selectbox(f"Feature {i+1}", options=numeric_cols_global, index=min(i, len(numeric_cols_global)-1), key=f"box-feature-{i}")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.boxplot(df[feature].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#E0ECF8", color="#4F81BD"),
                           medianprops=dict(color="red"))
                ax.set_title(feature, fontsize=10); ax.set_xticks([])
                st.pyplot(fig)
                desc = df[feature].describe(percentiles=[0.25,0.5,0.75]).round(1)
                st.markdown(
                    f"<div style='font-size: 13px;'><strong>Min:</strong> {desc['min']}<br>"
                    f"<strong>Q1:</strong> {desc['25%']}<br><strong>Median:</strong> {desc['50%']}<br>"
                    f"<strong>Q3:</strong> {desc['75%']}<br><strong>Max:</strong> {desc['max']}</div>",
                    unsafe_allow_html=True
                )

    with tab2:
        st.caption("Frequency distribution of key numerical features")
        c1, c2 = st.columns(2)
        selected_features = numeric_cols_global[:2] if len(numeric_cols_global) >= 2 else numeric_cols_global
        for i, col in enumerate([c1, c2][:len(selected_features)]):
            feature = selected_features[i]
            col.markdown(f"**{feature} Distribution**")
            binned = pd.cut(df[feature], bins=10)
            counts = binned.value_counts().sort_index()
            counts.index = [f"{int(iv.left):,} - {int(iv.right):,}" for iv in counts.index]
            chart_df = pd.DataFrame({feature: counts.values}, index=counts.index)
            col.bar_chart(chart_df)

    with tab3:
        st.subheader("🔗 Correlation Heatmap")
        corr = df[numeric_cols_global].corr()
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(corr, annot=True, cmap="BrBG", linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=10)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("💡 Correlation Insights")
        st.caption("Key findings from correlation analysis")
        blocks = [
            ("Strong Positive Correlation", "There are no strong positive correlations; values are close to zero, suggesting independence.", "#e8f0fe", "#1967d2"),
            ("Asset-Income Relationship", "Income shows near-zero linear correlation with other features, implying weak linear relationships.", "#e6f4ea", "#137333"),
            ("Independence Noted", "Low magnitudes across the board indicate features contribute largely independent information.", "#fef7e0", "#d39e00"),
        ]
        for t, txt, bg, tc in blocks:
            st.markdown(
                f"<div style='background:{bg}; padding:15px; border-radius:10px; margin-bottom:12px;'>"
                f"<h5 style='color:{tc}; margin-bottom:5px;'>{t}</h5><p style='margin:0;'>{txt}</p></div>",
                unsafe_allow_html=True
            )

    st.markdown("___")
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

    if not st.session_state.get("preprocessing_done"):
        st.warning("⚠️ Please complete preprocessing before training models.")
        return

    # ---- trigger training ----
    if "results_dict" not in st.session_state:
        if st.button("🚀 Train Models", type="primary"):
            with st.status("Training models...", expanded=True) as status:
                def log(msg: str): status.write(msg)
                results = train_and_evaluate_models(
                    st.session_state["X_train"], st.session_state["y_train"],
                    st.session_state["X_test"],  st.session_state["y_test"],
                    st.session_state["preprocessor"].get_feature_names_out(),
                    on_step=log, target_names=st.session_state.get("target_names", None),
                )
                (results_dict, best_name, best_model, y_pred_best, cm, feat_imp_df,
                 clf_report_text, infer_ms) = results

                if clf_report_text:
                    st.subheader("Classification Report (Best Model)")
                    st.text(clf_report_text)
                if infer_ms is not None:
                    st.caption(f"Inference speed ≈ {infer_ms:.2f} ms per sample")

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

    # ---- display trained results ----
    results_dict   = st.session_state["results_dict"]
    best_name      = st.session_state["best_model_name"]
    best_result    = results_dict[best_name]
    best_model     = st.session_state["best_model"]
    X_test         = st.session_state["X_test"]
    y_test         = st.session_state["y_test"]
    cm_stored      = st.session_state["conf_matrix"]
    feat_imp_df    = st.session_state["feature_importance_df"]

    test_acc_stored = best_result.get("test_accuracy", 0.0)
    test_auc_stored = best_result.get("test_auc", best_result.get("auc", 0.0))
    search_time_s   = best_result.get("cv_fit_time", best_result.get("training_time", 0.0))

    # Scores for threshold tuning (safe) — higher = risk(Default=1)
    y_scores = _safe_scores(best_model, X_test)

    # Historical approval rate from dataset (class 0 = approved)
    try:
        historical_approval_rate = (df["Default"] == 0).mean() * 100 if "Default" in df.columns else None
    except Exception:
        try:
            hist_df = pd.read_csv("Loan_default.csv")
            historical_approval_rate = (hist_df["Default"] == 0).mean() * 100 if "Default" in hist_df.columns else None
        except Exception:
            historical_approval_rate = None

    # ===== Threshold Tuning + Business Mode (no key collisions) =====
    st.markdown("#### Threshold Tuning")

    # Build curve df if scores exist (used by Business Mode + plot)
    thr_df = None
    curve_df = None
    if y_scores is not None:
        thr_grid = np.linspace(0.05, 0.95, 91)
        curve_df = _approval_curve(y_test, y_scores, thr_grid)
        curve_df["ApprovalRatePct"] = (curve_df["ApprovalRate"] * 100).round(2)
        thr_df = curve_df[["Threshold", "ApprovalRate"]].rename(
            columns={"ApprovalRate": "Model Approval Rate"}
        ).reset_index(drop=True)

    # --- Business Mode toggle (separate from manual slider key)
    if y_scores is None:
        st.toggle("Use Business Mode to set the approval threshold", value=False, disabled=True)
        st.caption("Business Mode unavailable (model does not expose scores).")
        use_business_mode = False
    else:
        use_business_mode = st.toggle(
            "Use Business Mode to set the approval threshold",
            value=(st.session_state.get("policy_mode", "business") == "business"),
            help=("Automatically pick a policy threshold from your risk appetite "
                  "using the model's test-set behavior.")
        )

    # Compute final policy_threshold (either from business mode or manual slider)
    if use_business_mode and thr_df is not None:
        TARGETS = {
            "Aggressive (grow volume)": 0.60,   # ≥ 60% approvals
            "Balanced (risk vs volume)": 0.45,  # ≥ 45% approvals
            "Conservative (protect book)": 0.30 # ≥ 30% approvals
        }
        appetite = st.radio(
            "Risk appetite",
            list(TARGETS.keys()),
            index=["Aggressive (grow volume)","Balanced (risk vs volume)","Conservative (protect book)"].index(
                st.session_state.get("policy_appetite","Balanced (risk vs volume)")
            ),
            horizontal=True
        )
        st.session_state["policy_appetite"] = appetite
        target = TARGETS[appetite]

        def _pick_threshold_for_target(df_: pd.DataFrame, target_rate: float) -> float:
            cand = df_[df_["Model Approval Rate"] >= target_rate]
            if len(cand):
                return float(cand.sort_values("Threshold").iloc[0]["Threshold"])
            return float(df_.sort_values("Model Approval Rate", ascending=False).iloc[0]["Threshold"])

        policy_threshold = float(_pick_threshold_for_target(thr_df, target))

        # persist non-widget state
        st.session_state["policy_threshold"] = float(policy_threshold)
        st.session_state["policy_mode"] = "business"

        colA, colB, colC = st.columns(3)
        with colA: st.metric("Selected appetite", appetite.split(" (")[0])
        with colB: st.metric("Target approval ≥", f"{target*100:.0f}%")
        with colC: st.metric("Recommended threshold", f"{policy_threshold:.2f}")

        st.caption("Business Mode picks the smallest threshold whose backtested approval rate meets your target.")
    else:
        # Manual mode slider—DIFFERENT key to avoid session/key collisions
        policy_threshold = st.slider(
            "Decision threshold for 'Default' (positive class)",
            min_value=0.05, max_value=0.95, step=0.01,
            value=float(st.session_state.get("policy_threshold", 0.50)),
            key="manual_threshold",
            help="Lower threshold approves more loans (risk score < threshold ⇒ approve)."
        )
        st.session_state["policy_threshold"] = float(policy_threshold)
        st.session_state["policy_mode"] = "manual"

    # Predictions at current threshold
    if y_scores is not None:
        y_pred_thr = (y_scores >= st.session_state["policy_threshold"]).astype(int)
        approval_rate_now = float((y_scores < st.session_state["policy_threshold"]).mean() * 100)
    else:
        y_pred_thr = st.session_state.get("y_pred_best", best_result.get("y_pred"))
        approval_rate_now = None

    # Metrics @ threshold (AUC from scores if available, else stored)
    acc  = accuracy_score(y_test, y_pred_thr)
    bacc = balanced_accuracy_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    rec  = recall_score(y_test, y_pred_thr, zero_division=0)
    f1   = f1_score(y_test, y_pred_thr, zero_division=0)
    auc  = roc_auc_score(y_test, y_scores) if y_scores is not None else test_auc_stored
    ap   = average_precision_score(y_test, y_scores) if y_scores is not None else None
    cm   = confusion_matrix(y_test, y_pred_thr)

    model_type = best_name.split(" (")[0]

    # KPI cards
    pt = float(st.session_state["policy_threshold"])
    st.markdown(
        f"""
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top: 10px;">
          <div style="flex:1; min-width:220px; background:#f7f7f7; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:14px; color:#666; font-weight:600;">Best Model</div>
            <div style="font-size:20px; font-weight:bold; margin:6px 0;">{model_type}</div>
            <div style="font-size:12px; color:#999;">CV Search Time: {search_time_s:.2f} s</div>
          </div>
          <div style="flex:1; min-width:220px; background:#f7f7f7; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:14px; color:#666; font-weight:600;">Accuracy @ {pt:.2f}</div>
            <div style="font-size:24px; font-weight:bold;">{acc*100:.2f}%</div>
            <div style="font-size:12px; color:#999;">Stored test acc: {test_acc_stored*100:.2f}%</div>
          </div>
          <div style="flex:1; min-width:220px; background:#f7f7f7; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:14px; color:#666; font-weight:600;">F1 (Default) @ {pt:.2f}</div>
            <div style="font-size:24px; font-weight:bold;">{f1:.2f}</div>
            <div style="font-size:12px; color:#999;">Precision {prec:.2f} • Recall {rec:.2f}</div>
          </div>
          <div style="flex:1; min-width:220px; background:#f7f7f7; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:14px; color:#666; font-weight:600;">AUC / PR-AUC</div>
            <div style="font-size:24px; font-weight:bold;">{auc:.2f} / {('-' if ap is None else f"{ap:.2f}")}</div>
            <div style="font-size:12px; color:#999;">Balanced Acc {bacc:.2f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if approval_rate_now is not None and historical_approval_rate is not None:
        st.caption(f"**Current policy approval (model): {approval_rate_now:.1f}%** vs **Historical: {historical_approval_rate:.1f}%**")

    # ===== Approval-rate curve & marker =====
    if y_scores is not None and curve_df is not None:
        fig_curve = px.line(
            curve_df, x="Threshold", y="ApprovalRatePct",
            title="Model‑Predicted Approval Rate vs Policy Threshold",
            labels={"ApprovalRatePct": "Approval Rate (%)"}
        )
        fig_curve.add_vline(x=pt, line_dash="dash",
                            annotation_text=f"Current {pt:.2f}",
                            annotation_position="top left")
        if historical_approval_rate is not None:
            fig_curve.add_hline(y=historical_approval_rate, line_dash="dot",
                                annotation_text=f"Historical {historical_approval_rate:.1f}%",
                                annotation_position="bottom right")
        st.plotly_chart(fig_curve, use_container_width=True)

        # quick default-rate readout at current threshold
        now_row = curve_df.iloc[(curve_df["Threshold"] - pt).abs().idxmin()]
        now_def_rate = now_row["DefaultRateAmongApproved"]
        if not np.isnan(now_def_rate):
            st.caption(
                f"At threshold **{pt:.2f}**: "
                f"Approval ≈ **{(now_row['ApprovalRate']*100):.1f}%**, "
                f"Default among approved ≈ **{now_def_rate*100:.1f}%**."
            )
    else:
        st.info("Approval‑rate curve & Business Mode unavailable (model does not expose scores).")

    # ===== Confusion matrix & curves =====
    st.markdown("### 🧮 Confusion Matrix") # for current threshold

    try:
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        # fallback in case shape is unexpected
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    def _fmt(n):
        try:
            return f"{int(n):,}"
        except Exception:
            return str(n)

    #  colored cards + short explanations (loan context)
    cards_html = f"""
    <style>
    .cm-grid {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .cm-card {{ flex: 1; min-width: 220px; border-radius: 12px; padding: 16px; border: 1px solid rgba(0,0,0,.06); }}
    .cm-title {{ margin: 0; font-size: 15px; font-weight: 700; }}
    .cm-value {{ font-size: 28px; font-weight: 800; margin: 6px 0; }}
    .cm-desc {{ font-size: 13px; margin: 0; opacity: .9; }}
    </style>

    <div class="cm-grid">
      <div class="cm-card" style="background:#dcfce7; border-color:#16a34a22;">
        <div class="cm-title">✅ True Positives (Default caught)</div>
        <div class="cm-value">{_fmt(tp)}</div>
        <p class="cm-desc">Predicted <b>Default</b> and borrower actually defaulted — loss avoided.</p>
      </div>

      <div class="cm-card" style="background:#fef9c3; border-color:#f59e0b22;">
        <div class="cm-title">⚠️ False Negatives (Missed risk)</div>
        <div class="cm-value">{_fmt(fn)}</div>
        <p class="cm-desc">Predicted <b>Safe</b> but borrower defaulted — costly misses.</p>
      </div>

      <div class="cm-card" style="background:#fee2e2; border-color:#ef444422;">
        <div class="cm-title">❌ False Positives (Over‑rejects)</div>
        <div class="cm-value">{_fmt(fp)}</div>
        <p class="cm-desc">Predicted <b>Default</b> but borrower was actually safe — revenue left on the table.</p>
      </div>

      <div class="cm-card" style="background:#e0f2fe; border-color:#3b82f622;">
        <div class="cm-title">🔍 True Negatives (Good approvals)</div>
        <div class="cm-value">{_fmt(tn)}</div>
        <p class="cm-desc">Predicted <b>Safe</b> and borrower was safe — optimal approvals.</p>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # quick rates to give more context
    total = tn + fp + fn + tp
    if total > 0:
        st.caption(
            f"TPR/Recall: {(tp / (tp + fn)) * 100:.1f}% • "
            f"FPR: {(fp / (fp + tn)) * 100:.1f}% • "
            f"Approval rate @ threshold: "
            f"{((y_scores < policy_threshold).mean() * 100 if y_scores is not None else float('nan')):.1f}%"
        )

    # keep the detailed heatmap & curves in an expander
    with st.expander("📈 View Confusion Matrix Heatmap / ROC / PR"):
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            cbar=False, ax=ax
        )
        ax.set_xlabel("Predicted");
        ax.set_ylabel("Actual");
        ax.set_title(f"Confusion Matrix @ {policy_threshold:.2f}")
        st.pyplot(fig)

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_scores)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr);
            ax2.plot([0, 1], [0, 1], "--")
            ax2.set_xlabel("FPR");
            ax2.set_ylabel("TPR");
            ax2.set_title("ROC Curve")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            ax3.plot(rec_curve, prec_curve)
            ax3.set_xlabel("Recall");
            ax3.set_ylabel("Precision");
            ax3.set_title("Precision–Recall Curve")
            st.pyplot(fig3)

    st.markdown("___")

    # ===== Model performance comparison table (stored test metrics) =====
    st.markdown("### 🧮 Model Performance Comparison")
    rows = []
    for name, res in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{res.get('test_accuracy', 0)*100:.1f}%",
            "Precision": f"{res.get('test_precision', res.get('precision', 0)):.2f}",
            "Recall": f"{res.get('test_recall', res.get('recall', 0)):.2f}",
            "F1-Score": f"{res.get('test_f1', res.get('f1_score', 0)):.2f}",
            "AUC": f"{res.get('test_auc', res.get('auc', 0)):.2f}",
            "Status": "best" if name == best_name else "good",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ===== Feature importance =====
    if feat_imp_df is not None and len(feat_imp_df):
        st.markdown("### 📉 Feature Importance (Top 10)")
        fig_imp = px.bar(feat_imp_df.head(10), x="Importance", y="Feature",
                         orientation="h", height=420, title="Most Influential Features")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("___")

    # ===== Tree visualization =====
    st.markdown("### 🌳 Model Visualization")

    # Pick which trained model to visualize
    model_names = list(st.session_state["trained_models"].keys())
    selected_model_name = st.selectbox("Select a model to visualize", model_names)
    model_to_plot = st.session_state["trained_models"][selected_model_name]

    # 2) Feature & class names (robust fallback)
    try:
        raw_feature_names = list(st.session_state["preprocessor"].get_feature_names_out())
    except Exception:
        n_features = st.session_state["X_train"].shape[1]
        raw_feature_names = [f"f{i}" for i in range(n_features)]

    # Tidy up long names like "cat__Gender_Male" → "Gender_Male"
    def _clean_name(s: str) -> str:
        for pat in ("num__", "cat__", "remainder__", "preprocessor__"):
            s = s.replace(pat, "")
        return s

    feature_names = [_clean_name(x) for x in raw_feature_names]
    class_names = ["No Default", "Default"]

    # 3) Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        depth = st.slider("Max depth to display", 1, 8, value=3)
    with c2:
        figsize = st.selectbox("Figure size", options=[(16, 8), (20, 10), (28, 14)],
                               format_func=lambda x: f"{x[0]} × {x[1]}")
    with c3:
        show_impurity = st.checkbox("Show impurity", value=False)

    show_proportion = st.checkbox("Show class proportions", value=True, help="Displays class percentages per node")
    show_text_rules = st.checkbox("Show text rules (compact)", value=False, help="Readable if the tree depth is small")

    # 4) Choose tree (for forests) or single DT
    from sklearn.tree import plot_tree as sk_plot_tree
    from sklearn.tree import export_text

    def _plot_single_tree(estimator, title: str):
        # Align feature name length with the estimator
        n = getattr(estimator, "n_features_in_", len(feature_names))
        used_names = feature_names if len(feature_names) == n else [f"f{i}" for i in range(n)]

        fig, ax = plt.subplots(figsize=figsize)
        sk_plot_tree(
            estimator,
            feature_names=used_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=depth,
            impurity=show_impurity,
            proportion=show_proportion,
            fontsize=10,
            ax=ax,
        )
        ax.set_title(title)
        st.pyplot(fig, use_container_width=True)

        if show_text_rules and depth <= 5:  # keep text view readable
            st.caption("Compact text rules (truncated by the selected max depth):")
            try:
                rules = export_text(estimator, feature_names=used_names, max_depth=depth)
                st.code(rules, language="text")
            except Exception:
                st.info("Rules view unavailable for this estimator.")

    if isinstance(model_to_plot, RandomForestClassifier):
        if not getattr(model_to_plot, "estimators_", None):
            st.warning("This Random Forest has no fitted trees to visualize.")
        else:
            # Helpful selectors for forests
            trees = model_to_plot.estimators_
            idx = st.slider("Select tree index", 0, len(trees) - 1, 0)
            # Optional: pick a tree with median depth
            with st.expander("Auto-select helpers"):
                max_depths = [getattr(t.tree_, "max_depth", None) for t in trees]
                if any(d is not None for d in max_depths):
                    md_idx = int(np.argsort([d if d is not None else -1 for d in max_depths])[len(trees) // 2])
                    if st.button(f"Use median-depth tree (index {md_idx}, depth≈{max_depths[md_idx]})"):
                        idx = md_idx

            tree = trees[idx]
            _plot_single_tree(tree, f"Random Forest — Tree #{idx}")
    else:
        if isinstance(model_to_plot, DecisionTreeClassifier):
            _plot_single_tree(model_to_plot, "Decision Tree")
        else:
            st.warning("⚠️ This model type cannot be visualized as a tree.")


def prediction():
    if "preprocessor" not in st.session_state or "trained_models" not in st.session_state:
        st.warning("⚠️ Please train at least one model before using the prediction page.")
        return

    # --- Current threshold from session (risk side)
    current_thr = float(st.session_state.get("policy_threshold", 0.50))

    st.markdown("<h2 class='section-title'>📝 Loan Application Form</h2>", unsafe_allow_html=True)
    st.caption("This page applies the current policy threshold from the Model page.")
    st.metric("Current Policy Threshold (risk)", f"{current_thr:.2f}")

    # Schema for form
    df_meta = pd.read_csv("Loan_default.csv")
    X_meta = df_meta.drop(columns=["LoanID", "Default"], errors="ignore")
    numeric_cols = X_meta.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_meta.select_dtypes(include="object").columns.tolist()

    # friendly labels
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
        "LoanPurpose": "Loan Purpose",
        "HasMortgage": "HasMortgage",
    }

    # saved stats from preprocessing (medians for defaults, min/max for guardrails)
    med = st.session_state.get("medians", {})
    mins = st.session_state.get("mins", {})
    maxs = st.session_state.get("maxs", {})

    def find_ood(numeric_row: dict, mins: dict, maxs: dict):
        flags = []
        for k, v in numeric_row.items():
            if k in mins and k in maxs:
                try:
                    if (float(v) < float(mins[k])) or (float(v) > float(maxs[k])):
                        flags.append(k)
                except Exception:
                    pass
        return flags

    # Local utility: safe probabilities built on helper._safe_scores
    def _safe_probas(model, X):
        """
        Returns (approved_prob, risk_prob) with risk_prob in [0,1].
        Prefers predict_proba; otherwise builds from _safe_scores.
        If all fails, returns (None, None).
        """
        try:
            proba = model.predict_proba(X)[0]
            return float(proba[0]), float(proba[1])  # [Approved, Default]
        except Exception:
            scores = _safe_scores(model, X)
            if scores is not None:
                risk = float(scores[0])
                return float(1.0 - risk), risk
        return None, None

    with st.form("loan_form"):
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
                if not options:
                    options = ["N/A"]
                user_input[col] = st.selectbox(label, options=options, key=f"cat1_{col}")

            for col in numeric_cols[: len(numeric_cols) // 2]:
                label = friendly_labels.get(col, col.replace("_", " "))
                val = float(med.get(col, 0.0))
                lo = float(mins.get(col, 0.0))
                hi = float(maxs.get(col, val if val > 0 else 1.0))
                user_input[col] = st.number_input(label, value=val, min_value=lo, max_value=hi, step=1.0,
                                                  key=f"num1_{col}")

        with col2:
            for col in cat_cols[len(cat_cols) // 2:]:
                label = friendly_labels.get(col, col.replace("_", " "))
                options = sorted(df_meta[col].dropna().unique().tolist())
                if not options:
                    options = ["N/A"]
                user_input[col] = st.selectbox(label, options=options, key=f"cat2_{col}")

            for col in numeric_cols[len(numeric_cols) // 2:]:
                label = friendly_labels.get(col, col.replace("_", " "))
                val = float(med.get(col, 0.0))
                lo = float(mins.get(col, 0.0))
                hi = float(maxs.get(col, val if val > 0 else 1.0))
                user_input[col] = st.number_input(label, value=val, min_value=lo, max_value=hi, step=1.0,
                                                  key=f"num2_{col}")

        submitted = st.form_submit_button("🔍 Predict", use_container_width=False)

    if not submitted:
        return

    # OOD warning (soft)
    numeric_subset = {k: v for k, v in user_input.items() if k in mins}
    ood = find_ood({k: user_input[k] for k in numeric_subset}, mins, maxs)
    if ood:
        st.warning(
            "Some values are **outside the training range**: " + ", ".join(ood) + ". Prediction may be less reliable.")

    # Transform input
    input_df = pd.DataFrame([user_input])
    X_trans = st.session_state["preprocessor"].transform(input_df)
    try:
        X_dense = X_trans.toarray()
    except AttributeError:
        X_dense = np.asarray(X_trans)
    feature_names = st.session_state["preprocessor"].get_feature_names_out()

    # Compare across all models (uses safe probas)
    results_table = []
    for name, mdl in st.session_state["trained_models"].items():
        approved_prob, risk_prob = _safe_probas(mdl, X_dense)
        if approved_prob is None:  # fallback to hard class if everything else fails
            try:
                pred_class = int(mdl.predict(X_dense)[0])
                results_table.append({
                    "Model": name,
                    "Approval Probability": "n/a",
                    "Risk Probability": "n/a",
                    "Predicted Class": "Approved" if pred_class == 0 else "Not Approved"
                })
                continue
            except Exception:
                results_table.append({
                    "Model": name,
                    "Approval Probability": "n/a",
                    "Risk Probability": "n/a",
                    "Predicted Class": "n/a"
                })
                continue

        pred_class = 0 if (risk_prob < current_thr) else 1
        results_table.append({
            "Model": name,
            "Approval Probability": f"{approved_prob * 100:.1f}%",
            "Risk Probability": f"{risk_prob * 100:.1f}%",
            "Predicted Class": "Approved" if pred_class == 0 else "Not Approved"
        })

    st.markdown("## 📊 Model Prediction Comparison")
    st.caption("All models scored on this application using the current policy threshold.")
    st.dataframe(pd.DataFrame(results_table), use_container_width=True)

    # Selected model decision (policy applies here)
    model = st.session_state["trained_models"][model_choice]
    approved_prob, risk_prob = _safe_probas(model, X_dense)

    if approved_prob is None:  # extreme fallback: use hard class only
        pred_class = int(model.predict(X_dense)[0])
        final_label = pred_class  # 0 approved, 1 not approved
        confidence = None
    else:
        final_label = 0 if (risk_prob < current_thr) else 1
        confidence = (1.0 - risk_prob) if final_label == 0 else risk_prob

    st.markdown("## 🎯 Selected Model Result")
    if final_label == 0:
        conf_txt = "" if confidence is None else f"<br><span>{confidence * 100:.1f}% confidence (policy risk < {current_thr:.2f})</span>"
        st.markdown(f"<div class='result-card approved'>Loan Approved ✅{conf_txt}</div>", unsafe_allow_html=True)
    else:
        conf_txt = "" if confidence is None else f"<br><span>{confidence * 100:.1f}% confidence (policy risk ≥ {current_thr:.2f})</span>"
        st.markdown(f"<div class='result-card declined'>Loan Not Approved – Risk of Default ⚠️{conf_txt}</div>",
                    unsafe_allow_html=True)

    # Contributions (tree-only)
    is_tree_model = isinstance(model, (DecisionTreeClassifier, RandomForestClassifier))
    if is_tree_model:
        pred_idx = 0 if final_label == 0 else 1
        try:
            prediction_vec, bias, contributions = ti.predict(model, X_dense)
            contrib_for_pred = contributions[0, :, pred_idx]
            contrib_df = pd.DataFrame({
                "Feature": feature_names,
                "Contribution": contrib_for_pred,
                "AbsContribution": np.abs(contrib_for_pred),
            }).sort_values("AbsContribution", ascending=False)
            top_k = contrib_df.head(10).drop(columns=["AbsContribution"])

            st.markdown("## 📌 Top Factors (Selected Model)")
            st.caption("Positive values push **towards** this outcome; negative values push **against** it.")
            fig = px.bar(top_k.sort_values("Contribution"), x="Contribution", y="Feature",
                         orientation="h", height=420, color="Contribution",
                         color_continuous_scale=["#d73027", "#1a9850"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Contribution chart unavailable for this model configuration.")
    else:
        st.info("Feature contribution chart is available for tree models.")

# ----------------- router -----------------
def main() -> None:
    load_css("styles.css")
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
            options=["Loan Prediction", "Data Overview", "Preprocessing", "Model Training & Evaluation"],
            icons=["house", "database", "gear", "cpu"],
            menu_icon=None, default_index=1,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "black", "font-size": "18px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ffffff", "font-weight": "bold", "color": "#000000"},
            },
        )

    if selected_page == "Loan Prediction":
        prediction()
    elif selected_page == "Data Overview":
        data_overview()
    elif selected_page == "Preprocessing":
        preprocessing_page()
    elif selected_page == "Model Training & Evaluation":
        model_page()

    # Team list footer
    st.sidebar.markdown("___")
    st.sidebar.selectbox(
        "Tap to View All Group Members",
        (
            "Eleazer F. Quayson (22253333)", "Priscilla D. Gborbitey (22253220)",
            "Magdalene Arhin (22253225)", "Anna E.A Creppy (11410565)",
            "Raymond Tetteh - 22255065", "Samuel K. Tuffour (22253144)"
        )
    )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Unhandled exception occurred. See details below (temporary debug).")
        st.code("".join(traceback.format_exc()), language="python")
