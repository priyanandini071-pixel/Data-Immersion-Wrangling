import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title=" Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("📊 Data Immersion & Wrangling Dashboard")
st.markdown("### Task 1 - Data Access, Profiling, Cleaning & Preparation")

# Load Dataset
try:
    df = pd.read_excel("ApexPlanet_DataAnalytics_Dataset.xlsx")
    st.success("✅ Dataset Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Dataset: {e}")
    st.stop()

# Dataset Preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10))

# Dataset Information
st.subheader("📌 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Duplicate Rows", df.duplicated().sum())

# Data Dictionary Creation
st.subheader("📖 Data Dictionary")

data_dictionary = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values,
    "Unique Values": [df[col].nunique() for col in df.columns]
})

st.dataframe(data_dictionary)

# Save Data Dictionary
data_dictionary.to_csv(
    "Data_Dictionary.csv",
    index=False
)

# Download Data Dictionary
dict_csv = data_dictionary.to_csv(index=False)

st.download_button(
    label="⬇ Download Data Dictionary",
    data=dict_csv,
    file_name="Data_Dictionary.csv",
    mime="text/csv"
)

# Missing Value Analysis
st.subheader("⚠ Missing Value Analysis")

missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_values)

# Missing Value Visualization
fig1 = plt.figure(figsize=(8, 4))

plt.bar(
    missing_values["Column"],
    missing_values["Missing Values"]
)

plt.xticks(rotation=45)
plt.ylabel("Count")
plt.title("Missing Values by Column")
plt.tight_layout()

st.pyplot(fig1)

# Duplicate Record Analysis
st.subheader("🔁 Duplicate Analysis")

duplicate_count = df.duplicated().sum()

st.metric(
    "Duplicate Records Found",
    duplicate_count
)

# Data Cleaning and Preprocessing
st.subheader("🧹 Data Cleaning")

cleaned_df = df.copy()

# Remove Duplicate Rows
cleaned_df.drop_duplicates(inplace=True)

# Missing Value Handling
for col in cleaned_df.columns:

    if cleaned_df[col].dtype == "object":

        cleaned_df[col] = cleaned_df[col].fillna("Unknown")

        cleaned_df[col] = (
            cleaned_df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

    else:

        cleaned_df[col] = cleaned_df[col].fillna(
            cleaned_df[col].median()
        )

st.success("✅ Missing Values Handled Successfully")

# Date Standardization
st.subheader("📅 Date Standardization")

date_columns = []

for col in cleaned_df.columns:

    if "date" in col.lower():

        try:
            cleaned_df[col] = pd.to_datetime(
                cleaned_df[col],
                errors="coerce"
            )

            date_columns.append(col)

        except:
            pass

if date_columns:
    st.success(
        f"✅ Date Columns Standardized: {', '.join(date_columns)}"
    )
else:
    st.info("No Date Columns Found")

# Feature Engineering
st.subheader("⚙ Feature Engineering")

# Create Order Month Feature
if "Order_Date" in cleaned_df.columns:

    cleaned_df["Order_Month"] = (
        cleaned_df["Order_Date"]
        .dt.month_name()
    )

    st.success("✅ Order_Month Created")

# Create Sales Per Unit Feature
if (
    "Total_Sales" in cleaned_df.columns and
    "Quantity" in cleaned_df.columns
):

    cleaned_df["Sales_Per_Unit"] = (
        cleaned_df["Total_Sales"] /
        cleaned_df["Quantity"]
    )

    st.success("✅ Sales_Per_Unit Created")

# Outlier Detection
st.subheader("🚨 Outlier Detection")

numeric_cols = cleaned_df.select_dtypes(
    include=np.number
).columns

if len(numeric_cols) > 0:

    selected_col = st.selectbox(
        "Select Numeric Column",
        numeric_cols
    )

    Q1 = cleaned_df[selected_col].quantile(0.25)
    Q3 = cleaned_df[selected_col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = cleaned_df[
        (cleaned_df[selected_col] < lower) |
        (cleaned_df[selected_col] > upper)
    ]

    st.write(
        f"Number of Outliers: {len(outliers)}"
    )

    fig2 = plt.figure(figsize=(6, 4))

    plt.boxplot(cleaned_df[selected_col])

    plt.title(
        f"Outlier Detection - {selected_col}"
    )

    st.pyplot(fig2)

# Correlation Analysis
st.subheader("🔗 Correlation Matrix")

corr_matrix = cleaned_df.corr(
    numeric_only=True
)

st.dataframe(corr_matrix)

# Analysis Ready Dataset
st.subheader("✅ Analysis Ready Dataset")

st.dataframe(cleaned_df.head(10))

# Export Cleaned Dataset
cleaned_df.to_csv(
    "Cleaned_Dataset.csv",
    index=False
)

# Download Cleaned Dataset
cleaned_csv = cleaned_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Cleaned Dataset",
    data=cleaned_csv,
    file_name="Cleaned_Dataset.csv",
    mime="text/csv"
)

# Project Summary
st.subheader("📋 Project Summary")

st.success(
    "🎉 Data profiling, cleaning, transformation, feature engineering, and dataset preparation completed successfully."
)