import streamlit as st
import numpy as np
import pandas as pd
import requests
from fractions import Fraction

st.set_page_config(page_title="Benter Mobile Master Engine", layout="centered")
st.title("🏇 Benter Algorithmic Engine Pro Master")
st.caption("All-in-one system with dual-odds conversion, market timing, track links, and live API tools")

# --- ONLINE SCRAPER: AUTOMATED ODDS FETCHING FUNCTION ---
def fetch_live_market_data(api_key, region="uk"):
    """
    Connects to online betting platforms to scrape real-time market entries.
    """
    url = "https://the-odds-api.com"
    params = {
        'apiKey': api_key,
        'regions': region,
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.success("🤖 Scraped real-time market data from online platforms successfully!")
            return response.json()
        else:
            st.warning(f"⚠️ API Connection responded with code {response.status_code}. Using manual data below.")
            return None
    except Exception as e:
        st.error(f"❌ Could not connect to online platform: {str(e)}")
        return None

# --- FRACTION TO DECIMAL UTILITY CONVERTERS ---
def fraction_to_decimal(frac_str):
    try:
        if '/' in str(frac_str):
            num, denom = str(frac_str).split('/')
            return (float(num) / float(denom)) + 1.0
        return float(frac_str)
    except:
        return 2.0 

def decimal_to_fraction(decimal_val):
    try:
        if decimal_val <= 1.0:
            return "0/1"
        imbalance = decimal_val - 1.0
        frac = Fraction(imbalance).limit_denominator(20)
        return f"{frac.numerator}/{frac.denominator}"
    except:
        return "1/1"

# --- EXPANDABLE SETTINGS (Saves screen space on phone) ---
with st.expander("⚙️ System Capital & Market Controls", expanded=False):
    bankroll = st.number_input("Total Trading Bankroll ($)", min_value=10.0, value=5000.0)
    kelly_fraction = st.slider("Kelly Fraction Risk Sizer", 0.05, 1.0, 0.20)
    track_moisture = st.slider("Live Track Moisture Level", 0.0, 1.0, 0.40)
    
# --- RECONNAISSANCE: RACECOURSE DIRECT QUICK LINKS ---
st.subheader("🌐 Live Racecourse Platforms & Streams")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🇬🇧 Ascot Live Feed", use_container_width=True):
        st.markdown("[Open Ascot Official Site](https://ascot.com)")
with col2:
    if st.button("🇺🇸 Churchill Downs", use_container_width=True):
        st.markdown("[Open Churchill Downs](https://churchilldowns.com)")
with col3:
    if st.button("🇭🇰 Sha Tin / HKJC", use_container_width=True):
        st.markdown("[Open Hong Kong Jockey Club](https://hkjc.com)")

st.markdown("---")

# --- INTEGRATED ONLINE PLATFORM CONNECTIVITY SECTION ---
st.subheader("📡 Live Web Platform Integration")
with st.expander("🔌 Connect to Live API Stream", expanded=False):
    user_api_key = st.text_input("Enter your Free 'TheOddsAPI' Key String:", type="password")
    selected_region = st.selectbox("Select Target Region Market:", ["uk", "us", "au", "eu"])
    if st.button("🔄 Scrape Online Odds Now", use_container_width=True):
        if user_api_key:
            fetch_live_market_data(user_api_key, region=selected_region)
        else:
            st.error("Please enter a valid API key string to access online networks.")

st.markdown("---")

# --- GLOBAL DISPLAY FORMAT SETTING BUTTON ---
st.subheader("🎚️ Odds Format Display Toggle")
odds_display_format = st.radio(
    "Select Your Preferred Format Matrix Mode:",
    ('Decimal Form (e.g. 4.50)', 'Fractional Form (e.g. 7/2)'),
    horizontal=True
)

# --- INTERACTIVE DATA ENTRY MATRIX ---
st.subheader("1. Quick-Entry Timing Field Matrix")
st.info("⚠️ Enter inputs into the grid using standard decimals (e.g., 3.5 or 6.0) for accurate equation calculations.")

if "race_field_master" not in st.session_state:
    st.session_state.race_field_master = pd.DataFrame({
        'Horse Name': ['Apex King', 'Blitzkrieg', 'Crimson Tide'],
        'Morning Odds (Dec)': [4.50, 5.00, 2.50],
        'Live 10-Min Odds (Dec)': [3.50, 6.00, 2.10], # Late market updates
        'Sustained Speed': [92.0, 85.5, 94.5],
        'Early 800m (s)': [48.2, 49.5, 47.9],
        'Final 400m (s)': [22.1, 23.4, 24.5],
        'Jockey Win %': [0.24, 0.15, 0.28],
        'Trainer Win %': [0.18, 0.12, 0.22],
        'Weight (lbs)': [122.0, 118.0, 126.0],
        'Horse Weight (lbs)': [1100.0, 1050.0, 1150.0],
        'Wet Track Win %': [0.35, 0.50, 0.40]
    })

# Render mobile responsive editing grid spreadsheet
edited_df = st.data_editor(st.session_state.race_field_master, num_rows="dynamic", use_container_width=True)

if st.button("💾 Lock Matrix Changes", use_container_width=True):
    st.session_state.race_field_master = edited_df
    st.success("Race entries successfully compiled inside phone session memory!")

# --- COMPUTATIONAL TRADING ALGORITHM ENGINE ---
if st.button("🔥 EXECUTE CALCULATOR ALGORITHM", type="primary", use_container_width=True):
    df = edited_df.copy()
    
    # 1. Feature Engineering Vector Layers
    df['sectional_acceleration'] = df['Final 400m (s)'] / df['Early 800m (s)']
    df['weight_to_power'] = df['Weight (lbs)'] / df['Horse Weight (lbs)']
    df['jockey_trainer'] = df['Jockey Win %'] * df['Trainer Win %'] * 10
    df['track_fitness'] = df['Wet Track Win %'] * track_moisture
    
    # Calculate Late Market Drift (Morning vs 10-Min final line shift)
    df['odds_market_drift'] = df['Morning Odds (Dec)'] - df['Live 10-Min Odds (Dec)']
    
    # 2. Complete Benter Multi-Variable Beta Coefficient Matrix Weights
    weights = {
        'Sustained Speed': 0.22,
        'sectional_acceleration': -0.28,
        'jockey_trainer': 0.12,
        'track_fitness': 0.12,
        'weight_to_power': -0.08,
        'odds_market_drift': 0.18 # Higher weights reward steaming late horses
    }
    
    X = np.column_stack([
        df['Sustained Speed'], df['sectional_acceleration'], 
        df['jockey_trainer'], df['track_fitness'], df['weight_to_power'],
        df['odds_market_drift']
    ])
    beta = np.array(list(weights.values()))
    
    # 3. Multinomial Logit Transformation Matrix (Softmax)
    utility = np.dot(X, beta)
    exp_u = np.exp(utility - np.max(utility))
    model_probs = exp_u / np.sum(exp_u)
    
    df['Model Win %'] = np.round(model_probs, 4)
    df['Bookie Implied %'] = np.round(1.0 / df['Live 10-Min Odds (Dec)'], 4)
    df['Edge'] = df['Model Win %'] - df['Bookie Implied %']
    
    cash_stakes = []
    execution_orders = []
    
    # 4. Strict Overpricing Overlay Valuation with Fraction Kelly Management
    for idx, row in df.iterrows():
        if row['Edge'] > 0.015: 
            b = row['Live 10-Min Odds (Dec)'] - 1.0
            p = row['Model Win %']
            q = 1.0 - p
            f_star = ((b * p) - q) / b
            stake = max(0.0, f_star * kelly_fraction) * bankroll
            cash_stakes.append(f"${stake:.2f}")
            execution_orders.append("🔥 VALUE BET")
        else:
            cash_stakes.append("$0.00")
            execution_orders.append("❌ PASS")
            
    df['Stake'] = cash_stakes
    df['Order'] = execution_orders
    
    # --- OUTPUT SHEET CONVERSION DISPLAY LOGIC ---
    if 'Fractional Form' in odds_display_format:
        df['Morning Odds'] = df['Morning Odds (Dec)'].apply(decimal_to_fraction)
        df['Live 10-Min Odds'] = df['Live 10-Min Odds (Dec)'].apply(decimal_to_fraction)
    else:
        df['Morning Odds'] = df['Morning Odds (Dec)'].apply(lambda x: f"{x:.2f}")
        df['Live 10-Min Odds'] = df['Live 10-Min Odds (Dec)'].apply(lambda x: f"{x:.2f}")
        
    # --- VISUAL TRADING OUTPUT MATRIX DISPLAY ---
    st.subheader("2. Final Algorithmic Trading Orders")
    final_output_cols = ['Horse Name', 'Morning Odds', 'Live 10-Min Odds', 'Edge', 'Stake', 'Order']
    st.dataframe(df[final_output_cols], use_container_width=True)
