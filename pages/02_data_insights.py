import streamlit as st
from utils.data_loader import load_all_data, check_data_available

st.set_page_config(
    page_title="Data Insights - Matatu ML",
    page_icon="📊",
    layout="wide"
)

st.header("📊 Data Insights")

# Load data
data = load_all_data()

if check_data_available(data, 'predictions'):
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(data['predictions']))
    with col2:
        st.metric("Features", data['predictions'].shape[1])
    with col3:
        st.metric("Data Points Available", f"{len(data['predictions']):,}")
    
    st.divider()
    
    st.subheader("Data Sample")
    st.write("First 10 rows of the prediction dataset:")
    st.dataframe(data['predictions'].head(10), use_container_width=True)
    
    st.divider()
    
    st.subheader("Statistical Summary")
    st.write("Summary statistics for numerical columns:")
    st.dataframe(data['predictions'].describe(), use_container_width=True)
    
    st.divider()
    
    st.subheader("Data Types")
    st.write("Data types for each column:")
    st.dataframe(data['predictions'].dtypes, use_container_width=True)
    
    # Missing values check
    st.subheader("Missing Values")
    missing = data['predictions'].isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values detected in the dataset!")
    else:
        st.dataframe(missing[missing > 0], use_container_width=True)
    
    # Column information
    with st.expander("📋 Column Descriptions"):
        st.markdown("""
        The dataset contains the following columns:
        
        - **Prediction columns**: Model predictions for demand/pricing
        - **Actual values**: Ground truth values from test set
        - **Features**: Route ID, time of day, day of week, season, etc.
        
        Each row represents a specific prediction instance with associated features.
        """)
else:
    st.warning("⚠️ Predictions data not available. Please ensure `predictions_test.csv` exists in the repository.")
