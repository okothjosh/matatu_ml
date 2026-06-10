import streamlit as st
import plotly.express as px
import pandas as pd
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
            st.warning(f"Could not create chart. Displaying as table instead.")
            if check_data_available(data, 'surge_ranking'):
                st.dataframe(data['surge_ranking'].head(15), use_container_width=True)
    else:
        st.info("Surge ranking data not available.")

with col2:
    if check_data_available(data, 'surge_multipliers'):
        st.subheader("Surge Multiplier Distribution")
        
        try:
            # Check if data is numeric
            surge_mult = data['surge_multipliers'].copy()
            
            # Try to convert to numeric if needed
            if len(surge_mult.columns) > 0:
                col_name = surge_mult.columns[0]
                try:
                    surge_mult[col_name] = pd.to_numeric(surge_mult[col_name], errors='coerce')
                    
                    fig = px.histogram(
                        surge_mult,
                        x=col_name,
                        nbins=30,
                        title="Distribution of Surge Multipliers",
                        labels={col_name: "Surge Multiplier", "count": "Frequency"}
                    )
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Surge multiplier data format not suitable for histogram. Displaying as table:")
                    st.dataframe(surge_mult.head(20), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")
            st.info("Displaying raw data instead:")
            st.dataframe(data['surge_multipliers'].head(20), use_container_width=True)
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
            sim_data = data['simulation'].copy()
            x_col = sim_data.columns[0]
            y_col = sim_data.columns[1]
            
            # Try to convert to numeric
            try:
                sim_data[y_col] = pd.to_numeric(sim_data[y_col], errors='coerce')
                
                # Remove rows with NaN values
                sim_data = sim_data.dropna(subset=[y_col])
                
                if len(sim_data) > 0:
                    fig = px.scatter(
                        sim_data,
                        x=x_col,
                        y=y_col,
                        title="Simulation Outcomes",
                        hover_data=sim_data.columns.tolist()
                    )
                    fig.update_layout(height=500, hovermode="closest")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid numeric data for scatter plot.")
            except:
                st.info("Simulation data format not suitable for scatter plot.")
                
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
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
