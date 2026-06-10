import streamlit as st
import plotly.express as px
import pandas as pd
from utils.data_loader import load_all_data, check_data_available

st.set_page_config(
    page_title="Model Performance - Matatu ML",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 Model Performance")

# Load data
data = load_all_data()

st.markdown("""
### Models Developed

1. **Traditional ML Models** - Random Forest and Gradient Boosting
2. **XGBoost Model** - Enhanced gradient boosting with hyperparameter tuning
3. **Neural Networks** - Deep learning approach for demand prediction

### Evaluation Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes larger errors
- **R² Score** - Proportion of variance explained (0-1 scale)
""")

st.divider()

if check_data_available(data, 'ab_comparison'):
    st.subheader("Model Comparison Results")
    
    # Display the comparison table
    st.dataframe(data['ab_comparison'], use_container_width=True)
    
    # Create visualization if there are multiple columns
    if len(data['ab_comparison'].columns) > 1:
        st.subheader("Performance Visualization")
        
        try:
            # Prepare data for visualization - select numeric columns only
            df_viz = data['ab_comparison'].copy()
            
            # Convert numeric columns
            numeric_cols = []
            first_col = df_viz.columns[0]
            
            for col in df_viz.columns[1:]:
                try:
                    df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
                    numeric_cols.append(col)
                except:
                    pass
            
            # Only plot if we have numeric data
            if numeric_cols and len(numeric_cols) > 0:
                fig = px.bar(
                    df_viz[[first_col] + numeric_cols],
                    x=first_col,
                    y=numeric_cols,
                    title="Model Performance Comparison",
                    barmode="group",
                    labels={"value": "Score", "variable": "Metric"}
                )
                fig.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric data available for visualization. Check the data format.")
                
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")
            st.info("Displaying data as table instead.")
    
    # Model insights
    st.divider()
    
    with st.expander("📈 Model Insights"):
        st.markdown("""
        ### Key Findings
        
        - **Best Performing Model**: XGBoost achieved the highest R² score
        - **Error Analysis**: Average prediction error within acceptable range
        - **Model Stability**: Neural networks showed good generalization
        
        ### Recommendations
        
        1. Use XGBoost for production predictions
        2. Ensemble approach: Combine predictions from multiple models
        3. Regular retraining with new data to maintain accuracy
        """)
else:
    st.info("📊 Model comparison data not available. Please ensure `ab_comparison.csv` exists.")

st.divider()

with st.expander("🔧 Model Development Process"):
    st.markdown("""
    ### Development Pipeline
    
    1. **Data Preparation**
       - Data cleaning and validation
       - Feature engineering
       - Train/test split
    
    2. **Model Training**
       - Hyperparameter tuning
       - Cross-validation
       - Performance optimization
    
    3. **Model Evaluation**
       - Testing on unseen data
       - Error analysis
       - Comparison with baselines
    
    4. **Model Selection**
       - Best model identification
       - Production deployment
       - Performance monitoring
    """)
