"""Main Streamlit dashboard for Monte Carlo studies sequential testing."""

import streamlit as st


config_page = st.Page("01_configure.py", title="Configuration", icon=":material/settings:")
error_analysis_page = st.Page(
    "02_error_analysis.py", title="Error Analysis", icon=":material/monitoring:")

pg = st.navigation(
     {
            "Specifications": [config_page],
            "Results": [error_analysis_page],
        }
)
st.set_page_config( 
    page_title="Sequential Testing Monte Carlo study",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)
pg.run()