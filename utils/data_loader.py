import streamlit as st
import pandas as pd

@st.cache_data
def load_all_data():
    """
    Load all CSV files needed for the dashboard.
    Uses Streamlit's cache to avoid reloading on every page refresh.
    """
    data = {}
    
    try:
        data['predictions'] = pd.read_csv('predictions_test.csv')
    except FileNotFoundError:
        data['predictions'] = pd.DataFrame()
    
    try:
        data['simulation'] = pd.read_csv('simulation_results.csv')
    except FileNotFoundError:
        data['simulation'] = pd.DataFrame()
    
    try:
        data['surge_ranking'] = pd.read_csv('route_surge_ranking_with_names.csv')
    except FileNotFoundError:
        data['surge_ranking'] = pd.DataFrame()
    
    try:
        data['surge_multipliers'] = pd.read_csv('surge_multipliers.csv')
    except FileNotFoundError:
        data['surge_multipliers'] = pd.DataFrame()
    
    try:
        data['ab_comparison'] = pd.read_csv('ab_comparison.csv')
    except FileNotFoundError:
        data['ab_comparison'] = pd.DataFrame()
    
    return data

def check_data_available(data, key):
    """
    Check if a specific dataset is available and not empty.
    """
    return key in data and not data[key].empty
