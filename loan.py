import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score

df = pd.read_csv("Loan_default.csv")

# --------------------- PAGE FUNCTIONS --------------------- #

def home():
    st.header("Data Overview")

    if st.checkbox("View Raw Dataset"):
        st.write(df)

    if st.checkbox("Show Descriptive Statistics"):
        st.write(df.describe())

    if 'Default' in df.columns:
        st.subheader("Loan Status Distribution")
        st.bar_chart(df['Default'].value_counts())

    st.subheader("Correlation Heatmap")
    col_nums = df.select_dtypes(include=np.number)
    st.write("Final Numeric Columns:", col_nums.columns.tolist())

    if not col_nums.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(col_nums.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric values identified")

    st.subheader("Feature Distributions")
    num_cols = col_nums.columns.tolist()
    selected_feature = st.selectbox("Select a feature", num_cols)

    if selected_feature:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig_hist = px.histogram(df, x=selected_feature, nbins=10,
                                    title=f"{selected_feature} Distribution", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df, y=selected_feature,
                             title=f"{selected_feature} Spread & Outliers", template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

# --------------------- PREPROCESSING --------------------- #

@st.cache_data
def preprocess_data(df_raw):
    df_clean = df_raw.copy()

    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).drop(columns=['Default'], errors='ignore').columns.tolist()
    cat_cols = df_clean.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return {
        "Pipeline": full_pipeline,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols
    }

def preprocessing():
    st.header("Data Preprocessing")

    if st.checkbox("Show Data Info"):
        docker = io.StringIO()
        df.info(buf=docker)
        st.text(docker.getvalue())

    with st.spinner("Hold on while we preprocess the data..."):
        results = preprocess_data(df)

    st.markdown("### Feature Overview")
    st.markdown(f"**Numeric Columns:** {results['numeric_cols']}")
    st.markdown(f"**Categorical Columns:** {results['categorical_cols']}")

    st.success("Preprocessing pipeline successfully created. Model awaits training!")

# --------------------- MODEL & CONCLUSION PLACEHOLDERS --------------------- #

def model():
    st.header("Train & Evaluate Models")

# --------------------- LOAN PREDICTION LOGIC --------------------- #
def prediction():
    st.header("Predict Loan Eligibility")

# --------------------- PROJECT CONCLUSION --------------------- #
def conclusion():
    st.header("Insights & Conclusion")

# --------------------- PAGE ROUTING --------------------- #

# Group pages logically
prediction_page = {'Predict Loan Eligibility': prediction}
other_pages = {
    'Data Overview': home,
    'Data Preprocessing': preprocessing,
    'Train & Evaluate Models': model,
    'Insights & Conclusion': conclusion
}

# Sidebar UI
with st.sidebar:
    st.markdown("### Group 5 - SML@2025")
    selected_main = st.radio("Start Here", list(prediction_page.keys()))

    st.markdown("___")
    st.markdown("### Explore the Project")
    selected_subpage = st.selectbox("More Sections", ["None"] + list(other_pages.keys()))

# Control which page is rendered
if selected_subpage != "None":
    other_pages[selected_subpage]()  # Show selected subpage only
else:
    prediction_page[selected_main]()  # Default view
