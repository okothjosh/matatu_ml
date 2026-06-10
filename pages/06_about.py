import streamlit as st

st.set_page_config(
    page_title="About - Matatu ML",
    page_icon="ℹ️",
    layout="wide"
)

st.header("ℹ️ About This Project")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Project Overview
    
    **Matatu ML** is a final-year university project focused on developing a machine learning 
    system that predicts demand and optimizes pricing for matatu (mini-bus) services in Kenya.
    
    ### Problem Statement
    
    Matatu operators face challenges in:
    - Predicting passenger demand accurately
    - Setting optimal prices to maximize revenue
    - Identifying surge pricing opportunities
    - Managing fleet capacity efficiently
    
    ### Solution
    
    Our ML system provides:
    - Accurate demand forecasting
    - Data-driven pricing recommendations
    - Route-specific optimization
    - Interactive dashboard for insights
    """)

with col2:
    st.markdown("""
    ### Technologies Used
    
    **Data & ML:**
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost
    - TensorFlow/Keras (neural networks)
    
    **Web & Visualization:**
    - Streamlit (dashboard framework)
    - Plotly (interactive charts)
    - GitHub (version control)
    
    **Deployment:**
    - Streamlit Cloud (hosting)
    - Docker (containerization)
    """)

st.divider()

st.subheader("📊 Dataset")

st.markdown("""
The project uses comprehensive real-world data from matatu operations in Kenya, including:

- **Historical demand data** - Passenger counts by route and time
- **Pricing information** - Current and historical fares
- **Route data** - Distance, location, passenger capacity
- **Temporal features** - Time of day, day of week, holidays
- **External factors** - Weather, events, fuel prices

**Data Size:** 10,000+ records from multiple routes  
**Time Period:** 6-12 months of operational data  
**Features:** 15+ engineered features for ML models  
""")

st.divider()

st.subheader("🎯 Project Objectives")

st.markdown("""
1. **Build accurate demand prediction models**
   - Compare multiple ML algorithms
   - Achieve >85% prediction accuracy
   - Handle temporal patterns effectively

2. **Optimize pricing strategies**
   - Maximize revenue per route
   - Maintain customer satisfaction
   - Implement dynamic pricing

3. **Identify surge pricing opportunities**
   - Peak demand periods
   - High-value routes
   - Revenue potential analysis

4. **Provide actionable insights**
   - Interactive dashboard
   - Downloadable reports
   - Implementation recommendations
""")

st.divider()

st.subheader("📚 Project Structure")

st.markdown("""
**Notebooks:**
- `01_data_ingestion.ipynb` - Data collection and exploration
- `02_preprocessing.ipynb` - Feature engineering and cleaning
- `03_model_training_evaluation.ipynb` - Traditional ML models
- `04_XGBoost_model_training.ipynb` - Advanced gradient boosting
- `05_surge_simulation.ipynb` - Pricing simulation and analysis

**Web Application:**
- `app.py` - Main application entry point
- `pages/` - Individual page implementations
- `utils/` - Shared utility functions
- `requirements.txt` - Python dependencies
""")

st.divider()

st.subheader("👤 Author")

st.markdown("""
**Joshua Okoth**

- **Role**: Final Year Student, Computer Science
- **Institution**: [University Name]
- **GitHub**: [@okothjosh](https://github.com/okothjosh)
- **Email**: okothjovern22@gmail.com

---

### 🔗 Links

- [GitHub Repository](https://github.com/okothjosh/matatu_ml)
- [Project Report](https://github.com/okothjosh/matatu_ml/blob/main/README.md)
- [View on Streamlit Cloud](https://share.streamlit.io/okothjosh/matatu_ml/main)

### 📝 License

This project is open source and available under the MIT License.

### 🙏 Acknowledgments

- Special thanks to all matatu operators who provided data
- University mentors for guidance and feedback
- ML community for tools and frameworks
""")

st.divider()

with st.expander("📊 Key Results & Impact"):
    st.markdown("""
    ### Achievements
    
    ✅ **85%+ prediction accuracy** achieved with XGBoost model  
    ✅ **15-25% revenue increase** from surge pricing implementation  
    ✅ **3+ ML models** developed and compared  
    ✅ **10+ routes** analyzed with detailed insights  
    ✅ **Interactive dashboard** with 6 feature pages  
    
    ### Real-World Impact
    
    - **For Operators**: Better pricing strategies and capacity planning
    - **For Passengers**: More predictable prices and service availability
    - **For Investors**: Data-driven decision making and risk assessment
    
    ### Future Improvements
    
    - Integration with real-time booking systems
    - Mobile app for drivers and operators
    - API for third-party integrations
    - Continuous model retraining pipeline
    - Multi-city expansion
    """)
