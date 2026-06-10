import streamlit as st

st.set_page_config(
    page_title="Home - Matatu ML",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.header("🏠 Home")

st.markdown("""
## Welcome to the Matatu ML Dashboard

This dashboard presents a **machine learning system** developed to predict demand and optimize pricing for matatu (mini-bus) services in Kenya.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>📊 Data Analysis</h4>
        <p>Comprehensive exploratory data analysis of matatu operations, including demand patterns, pricing trends, and route performance.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🤖 ML Models</h4>
        <p>Advanced machine learning models including XGBoost and neural networks trained on real operational data.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>💰 Pricing Strategy</h4>
        <p>Surge pricing analysis and revenue optimization strategies based on demand forecasting.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("📋 Project Overview")

st.markdown("""
### What is Matatu ML?

Matatu ML is a final-year university project that develops a comprehensive machine learning system for:

1. **Demand Prediction** - Forecasting passenger demand across different routes and times
2. **Pricing Optimization** - Recommending optimal pricing strategies to maximize revenue
3. **Surge Analysis** - Identifying high-value surge pricing opportunities

### Key Achievements

✅ Built multiple ML models with high prediction accuracy  
✅ Identified profitable surge pricing opportunities  
✅ Created an interactive dashboard for insights  
✅ Provided actionable recommendations for operators  

### Data & Methods

- **Dataset**: Real-world matatu operational data from Kenya
- **Features**: Route info, time, demand patterns, pricing, passenger behavior
- **Models**: Random Forest, XGBoost, Neural Networks
- **Metrics**: MAE, RMSE, R² Score
""")

st.divider()

st.subheader("🚀 Getting Started")

st.markdown("""
**Explore the dashboard using the navigation menu:**

1. **📊 Data Insights** - Analyze the dataset and view statistics
2. **🤖 Model Performance** - Compare different ML models
3. **🔮 Predictions** - View model predictions on test data
4. **⚡ Surge Analysis** - Explore surge pricing opportunities
5. **ℹ️ About** - Learn more about the project
""")

st.info("💡 Tip: Click on the menu items in the left sidebar to navigate between pages.")
