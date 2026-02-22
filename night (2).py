import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
import datetime

st.set_page_config(
    page_title="Crop Cast | Futuristic Price Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Syncopate:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Animated Border */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020c1b);
        color: #e6f1ff;
        font-family: 'Inter', sans-serif;
        border: 4px solid #00ffff;
        box-shadow: inset 0 0 20px #00ffff, 0 0 20px #00ffff;
        margin: 0;
        height: 100vh;
        animation: borderGlow 4s infinite alternate;
    }

    @keyframes borderGlow {
        from { box-shadow: inset 0 0 10px #00ffff, 0 0 10px #00ffff; border-color: #00ffff; }
        to { box-shadow: inset 0 0 25px #00e5ff, 0 0 35px #00e5ff; border-color: #00e5ff; }
    }

    /* Hero Section Styling */
    .hero-container {
        text-align: center;
        padding: 80px 20px 20px 20px;
    }

    .app-title-main {
        font-family: 'Syncopate', sans-serif;
        font-size: 8rem;
        background: linear-gradient(180deg, #ffffff 0%, #00ffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 25px;
        font-weight: 700;
        filter: drop-shadow(0 0 30px rgba(0, 255, 255, 0.5));
        line-height: 1.2;
    }

    .team-attribution {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        color: #00ffff;
        letter-spacing: 8px;
        margin-bottom: 60px;
        opacity: 0.8;
        text-transform: uppercase;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 25px;
        margin-bottom: 20px;
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .glass-card:hover {
        border: 1px solid rgba(0, 255, 255, 0.8) !important;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.3);
    }

    .value-text {
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }

    .stButton > button {
        background: rgba(0, 255, 255, 0.05) !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        font-family: 'Orbitron', sans-serif !important;
        padding: 0.8rem 2rem !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        border-radius: 10px !important;
        transition: all 0.4s !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: #00ffff !important;
        color: #020c1b !important;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.6) !important;
    }
    
    .section-header {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        letter-spacing: 4px;
        margin-top: 20px;
        margin-bottom: 30px;
    }

    /* Remove unnecessary padding from main block */
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

if 'view' not in st.session_state:
    st.session_state.view = 'home'

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tensor.csv")
        df.columns = df.columns.str.strip().str.lower()
        df['ds'] = pd.to_datetime(df['month'].astype(str) + '-01')
        df['y'] = df['avg_max_price']
        return df
    except:
        dates = pd.date_range(start="2022-01-01", periods=36, freq='MS')
        crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane']
        data_frames = []
        for crop in crops:
            data_frames.append(pd.DataFrame({
                'month': [d.strftime('%Y-%m') for d in dates],
                'commodity_name': [crop]*36,
                'ds': dates,
                'y': np.random.randint(1800, 3500, 36) + np.linspace(0, 500, 36),
                'rainfall': np.random.randint(50, 200, 36),
                'temperature': np.random.randint(15, 38, 36)
            }))
        return pd.concat(data_frames)

def build_hybrid_model(df_crop):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_crop[['ds', 'y']])
    prophet_pred = m.predict(df_crop[['ds']])['yhat']
    residuals = df_crop['y'].values - prophet_pred.values
    xgb_features = pd.DataFrame({'month_num': df_crop['ds'].dt.month, 'year_num': df_crop['ds'].dt.year})
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    xgb_model.fit(xgb_features, residuals)
    return m, xgb_model

def get_agro_metrics(crop):
    agro_data = {
        'Wheat': {'temp': '15-24°C', 'rain': '450-650mm', 'soil': 'Loamy/Clayey'},
        'Rice': {'temp': '21-37°C', 'rain': '1500-2000mm', 'soil': 'Alluvial/Clayey'},
        'Cotton': {'temp': '21-30°C', 'rain': '500-1000mm', 'soil': 'Black Soil'},
        'Sugarcane': {'temp': '20-35°C', 'rain': '750-1200mm', 'soil': 'Sandy Loam'}
    }
    return agro_data.get(crop, {'temp': '22-28°C', 'rain': '800mm', 'soil': 'Well-drained'})

# View Logic
if st.session_state.view == 'home':
    st.markdown('<div class="hero-container"><h1 class="app-title-main">CROP CAST</h1><p class="team-attribution">engineered by team tensor</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    images = [
        "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1592982537447-7440770cbfc9?auto=format&fit=crop&w=800&q=80"
    ]
    for i, col in enumerate([col1, col2, col3]):
        with col: st.markdown(f'<img src="{images[i]}" style="width:100%; height:250px; object-fit:cover; border-radius:15px; border:1px solid #00ffff;">', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("INITIALIZE SYSTEM"):
            st.session_state.view = 'dashboard'
            st.rerun()

else:
    st.sidebar.markdown("<h2 style='color:#00ffff; font-family:Orbitron; letter-spacing:2px;'>CROP CAST</h2>", unsafe_allow_html=True)
    menu = st.sidebar.radio("Navigation Hub", ["Price Forecaster", "Historical Ledger", "About Team Tensor"])
    if st.sidebar.button("← TERMINATE SESSION"):
        st.session_state.view = 'home'
        st.rerun()

    data = load_data()
    available_crops = data['commodity_name'].unique()

    if menu == "Price Forecaster":
        st.markdown("<h2 class='section-header'>// CROP PRICE FORECASTER</h2>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 0.8])
        with c1: target_date = st.date_input("Target Forecast Timeline", datetime.date(2025, 12, 1))
        with c2: selected_crop = st.selectbox("Select Target Commodity", available_crops)
        with c3: 
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("EXECUTE ANALYTICS")

        if predict_btn:
            crop_df = data[data['commodity_name'] == selected_crop].sort_values('ds')
            p_model, x_model = build_hybrid_model(crop_df)
            f_ds = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
            p_val = p_model.predict(f_ds)['yhat'].iloc[0]
            x_input = pd.DataFrame({'month_num': [target_date.month], 'year_num': [target_date.year]})
            final_price = p_val + x_model.predict(x_input)[0]
            agro = get_agro_metrics(selected_crop)
            
            res_cols = st.columns(2)
            metrics = [
                ("Predicted Valuation", f"₹{final_price:,.2f}", "Per Quintal"),
                ("Thermal Index", agro['temp'], "Optimal Range"),
                ("Hydraulic Metric", agro['rain'], "Annual Accumulation"),
                ("Substrate", agro['soil'], "Sustainability Factor")
            ]
            for i, m in enumerate(metrics):
                with res_cols[i % 2]:
                    st.markdown(f'<div class="glass-card"><p style="color:#8892b0; font-size:0.8rem; letter-spacing:2px;">{m[0]}</p><p class="value-text">{m[1]}</p><p style="color:#8892b0; font-size:0.7rem;">{m[2]}</p></div>', unsafe_allow_html=True)

    elif menu == "Historical Ledger":
        st.markdown("<h2 class='section-header'>// HISTORICAL LEDGER</h2>", unsafe_allow_html=True)
        selected_crop = st.selectbox("Filter Ledger by Commodity", available_crops)
        crop_df = data[data['commodity_name'] == selected_crop].sort_values('ds')
        
        tab1, tab2 = st.tabs(["Market Trajectory", " Neural Data Matrix"])
        with tab1: st.line_chart(crop_df.set_index('ds')['y'], color="#00ffff")
        with tab2: st.dataframe(crop_df[['month', 'y', 'rainfall', 'temperature']], use_container_width=True)

    elif menu == "About Team Tensor":
        st.markdown("<h2 class='section-header'>// SYSTEM INFORMATION & CREATORS</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color:#00ffff; font-family:Orbitron;">THE ARCHITECTS</h3>
                    <p style="color:#e6f1ff; font-size:1.1rem; line-height:1.8;">
                        • <b>ISHAN MISHRA</b><br>
                        • <b>DEEPAK KUMAR MISHRA</b>  <br>
                        • <b>SRIYA DASH</b> <br>
                        • <b>ABHISHEK SADANGI</b> 
                    </p>
                    <hr style="border-color:rgba(0,255,255,0.2)">
                    <p style="font-size:0.9rem; opacity:0.8;">Specialized in hybrid time-series forecasting and neural interfaces.</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Transmit Query")
            with st.form("query_form"):
                st.text_input("User Identification")
                st.text_area("Inquiry Details")
                if st.form_submit_button("SEND DATA"): st.success("Packet received.")
            st.markdown('</div>', unsafe_allow_html=True)