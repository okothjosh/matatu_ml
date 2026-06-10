import streamlit as st
import plotly.express as px
from utils.data_loader import load_all_data, check_data_available

st.set_page_config(
    page_title="Surge Analysis - Matatu ML",
    page_icon="⚡",
    layout="wide"
)

st.header("⚡ Surge Pricing Analysis")

# Load data
data = load_all_data()

st.markdown("""
### Understanding Surge Pricing

Surge pricing is a dynamic pricing strategy that adjusts prices based on demand:
- **High demand periods**: Prices increase to maximize revenue
- **Low demand periods**: Prices decrease to encourage bookings
- **Peak hours**: Special rates during rush hours (morning, evening)
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    if check_data_available(data, 'surge_ranking'):
        st.subheader("Routes by Surge Potential")
        
        try:
            surge_data = data['surge_ranking'].head(15)
            
            if len(surge_data.columns) > 1:
                fig = px.bar(
                    surge_data,
                    x=surge_data.columns[1],
                    y=surge_data.columns[0],
                    title="Top 15 Routes by Surge Multiplier",
                    labels={
                        surge_data.columns[1]: "Surge Multiplier",
                        surge_data.columns[0]: "Route"
                    },
                    orientation="h"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(surge_data, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualization: {e}")
            st.dataframe(surge_data, use_container_width=True)
    else:
        st.info("Surge ranking data not available.")

with col2:
    if check_data_available(data, 'surge_multipliers'):
        st.subheader("Surge Multiplier Distribution")
        
        try:
            fig = px.histogram(
                data['surge_multipliers'],
                nbins=30,
                title="Distribution of Surge Multipliers",
                labels={"value": "Surge Multiplier", "count": "Frequency"}
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualization: {e}")
    else:
        st.info("Surge multiplier data not available.")

st.divider()

# Simulation results
if check_data_available(data, 'simulation'):
    st.subheader("Simulation Results")
    
    st.write("Simulation outcomes from surge pricing scenarios (first 10 results):")
    st.dataframe(data['simulation'].head(10), use_container_width=True)
    
    st.divider()
    
    # Revenue comparison
    try:
        if len(data['simulation'].columns) > 1:
            fig = px.scatter(
                data['simulation'],
                x=data['simulation'].columns[0],
                y=data['simulation'].columns[1],
                title="Simulation Outcomes",
                size=data['simulation'].columns[2] if len(data['simulation'].columns) > 2 else None,
                hover_data=data['simulation'].columns.tolist()
            )
            fig.update_layout(height=500, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create visualization: {e}")
else:
    st.info("Simulation data not available.")

st.divider()

with st.expander("💡 Surge Pricing Strategy"):
    st.markdown("""
    ### Implementation Strategy
    
    1. **Demand Forecasting**
       - Predict passenger demand for each route and time
       - Identify peak and off-peak periods
    
    2. **Dynamic Pricing**
       - Set base fares for each route
       - Apply surge multipliers during high-demand periods
       - Optimize for revenue and customer satisfaction
    
    3. **Route-Specific Optimization**
       - Different routes have different demand patterns
       - Customize surge strategies per route
       - Consider customer elasticity
    
    4. **Revenue Impact**
       - Estimated revenue increase: 15-25% from surge pricing
       - Improved capacity utilization
       - Better customer experience through availability
    
    ### Key Metrics
    
    - **Surge Multiplier**: Price increase factor (1.0 = base price)
    - **Peak Hours**: Times with highest demand
    - **Revenue Potential**: Expected revenue increase
    """)
