import streamlit as st
from utils.data_loader import load_all_data, check_data_available

st.set_page_config(
    page_title="Predictions - Matatu ML",
    page_icon="🔮",
    layout="wide"
)

st.header("🔮 Predictions & Results")

# Load data
data = load_all_data()

if check_data_available(data, 'predictions'):
    st.subheader("Test Set Predictions")
    
    # Display predictions
    st.write("Model predictions on the test dataset (first 20 rows):")
    st.dataframe(data['predictions'].head(20), use_container_width=True)
    
    st.divider()
    
    # Prediction statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(data['predictions'].columns) > 0:
            col_name = data['predictions'].columns[0]
            st.metric(
                col_name.replace('_', ' ').title(),
                f"{data['predictions'][col_name].mean():.2f}"
            )
    
    with col2:
        if len(data['predictions'].columns) > 1:
            col_name = data['predictions'].columns[1]
            st.metric(
                col_name.replace('_', ' ').title(),
                f"{data['predictions'][col_name].mean():.2f}"
            )
    
    with col3:
        if len(data['predictions'].columns) > 2:
            col_name = data['predictions'].columns[2]
            st.metric(
                col_name.replace('_', ' ').title(),
                f"{data['predictions'][col_name].mean():.2f}"
            )
    
    st.divider()
    
    # Download predictions
    st.subheader("📥 Download Results")
    
    csv = data['predictions'].to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="predictions_test.csv",
        mime="text/csv",
        help="Click to download the full prediction results"
    )
    
    st.divider()
    
    # Prediction analysis
    with st.expander("📊 Prediction Analysis"):
        st.markdown("""
        ### Understanding the Predictions
        
        The predictions represent the model's output for:
        - **Demand forecasting**: Predicted number of passengers
        - **Pricing recommendations**: Suggested optimal prices
        - **Confidence scores**: Model's confidence in each prediction
        
        ### Accuracy Metrics
        
        - **Mean Absolute Error (MAE)**: Average deviation from actual values
        - **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
        - **R² Score**: Explains the proportion of variance (higher is better)
        
        ### How to Use Predictions
        
        1. Use demand predictions for capacity planning
        2. Apply pricing recommendations to optimize revenue
        3. Monitor prediction accuracy over time
        4. Retrain models when accuracy degrades
        """)
else:
    st.warning("⚠️ Predictions data not available. Please ensure `predictions_test.csv` exists.")
