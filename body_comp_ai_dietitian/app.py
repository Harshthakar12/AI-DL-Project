import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from data_generator import generate_sample_fitness_data
from lstm_model import WeightTrajectoryPredictor
from ai_dietitian import generate_dietitian_assessment, chat_with_dietitian_response

# Page Configuration
st.set_page_config(
    page_title="AI Body Composition & Caloric Trajectory Forecaster",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("🤖 LLM API Settings")
    provider = st.selectbox(
        "AI Provider",
        ["Google Gemini", "OpenAI", "Demo / Simulated"],
        index=0,
        help="Select which LLM to power your AI Dietitian."
    )
    
    api_key = ""
    model_name = ""
    if provider != "Demo / Simulated":
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            placeholder="Paste your API key here...",
            help="Your API key stays local and is never stored permanently."
        )
        if "Gemini" in provider:
            model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-1.5-pro"], index=0)
        else:
            model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    else:
        st.info("💡 Running in Demo Mode with instant simulated nutritionist assessments.")

    st.markdown("---")
    st.subheader("👤 User Physical Profile")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        age = st.number_input("Age", 18, 90, 28)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col_u2:
        height = st.number_input("Height (cm)", 120, 230, 178)
        current_weight = st.number_input("Current Wt (kg)", 40.0, 200.0, 84.5, step=0.5)
        
    goal_weight = st.number_input("Goal Weight (kg)", 40.0, 200.0, 78.0, step=0.5)
    
    # Calculate approximate BMR/TDEE
    # Mifflin-St Jeor formula
    if gender == "Male":
        bmr = 10 * current_weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * current_weight + 6.25 * height - 5 * age - 161
    est_tdee = int(bmr * 1.45) # moderately active baseline
    
    st.caption(f"Estimated Baseline TDEE: **~{est_tdee} kcal/day**")

    st.markdown("---")
    st.subheader("🎯 Target Daily Protocol")
    planned_calories = st.slider("Target Daily Intake (kcal)", 1200, 4000, 1950, step=50)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        protein = st.number_input("Protein (g)", 50, 300, 165)
    with c_m2:
        carbs = st.number_input("Carbs (g)", 20, 500, 180)
    with c_m3:
        fat = st.number_input("Fat (g)", 20, 150, 55)
        
    macro_cals = protein * 4 + carbs * 4 + fat * 9
    st.caption(f"Macros sum: **{macro_cals} kcal** (vs. {planned_calories} target)")
    
    planned_steps = st.slider("Daily Step Target", 3000, 25000, 9500, step=500)
    forecast_days = st.slider("Forecast Horizon (Days)", 14, 60, 30, step=7)
    
    st.markdown("---")
    if st.button("🔄 Regenerate Fitness History Data", use_container_width=True):
        st.session_state.pop("fitness_df", None)
        st.session_state.pop("lstm_predictor", None)
        st.rerun()

# ----------------- SESSION STATE -----------------
if "fitness_df" not in st.session_state:
    with st.spinner("Generating historical sequence data..."):
        st.session_state.fitness_df = generate_sample_fitness_data(
            start_weight_kg=current_weight + 2.5,
            target_weight_kg=goal_weight,
            days=60,
            base_tdee=est_tdee,
            avg_intake=planned_calories + 200
        )

if "lstm_predictor" not in st.session_state:
    with st.spinner("Training PyTorch LSTM on sequence data..."):
        predictor = WeightTrajectoryPredictor(seq_len=7)
        predictor.train(st.session_state.fitness_df, epochs=80)
        st.session_state.lstm_predictor = predictor

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am Dr. Elena Vance, your AI Sports Dietitian. I've reviewed your biometric history and LSTM projection. How can I assist with your nutrition, meals, or macro targets today?"}
    ]

df_history = st.session_state.fitness_df
predictor = st.session_state.lstm_predictor

# Compute Forecast
with st.spinner("Calculating forward LSTM trajectory..."):
    df_forecast = predictor.forecast_future(
        historical_df=df_history,
        future_days=forecast_days,
        planned_daily_intake=planned_calories,
        planned_tdee=est_tdee,
        planned_protein=protein,
        planned_carbs=carbs,
        planned_fat=fat,
        planned_steps=planned_steps
    )

latest_actual_weight = round(df_history['Weight_kg'].iloc[-1], 2)
projected_final_weight = round(df_forecast['Forecasted_Weight_kg'].iloc[-1], 2)
net_change = round(projected_final_weight - latest_actual_weight, 2)
net_caloric_balance = planned_calories - est_tdee

# ----------------- MAIN HEADER -----------------
st.title("🥗 Body Composition & Caloric Trajectory Forecaster")
st.markdown("Combines **PyTorch LSTM sequence forecasting**, **LangChain Clinical Nutrition Agent**, and an interactive dashboard.")

# Top Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Current Weight</div>
        <div class="metric-val">{latest_actual_weight} kg</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">{forecast_days}-Day LSTM Forecast</div>
        <div class="metric-val">{projected_final_weight} kg</div>
    </div>""", unsafe_allow_html=True)
with m3:
    delta_color = "#10b981" if net_change < 0 else "#f59e0b"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Projected Δ</div>
        <div class="metric-val" style="color: {delta_color}">{'+' if net_change > 0 else ''}{net_change} kg</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Daily Energy Balance</div>
        <div class="metric-val">{'+' if net_caloric_balance > 0 else ''}{net_caloric_balance} kcal</div>
    </div>""", unsafe_allow_html=True)
with m5:
    gap = round(latest_actual_weight - goal_weight, 2)
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Remaining to Goal</div>
        <div class="metric-val">{abs(gap)} kg {'to lose' if gap > 0 else 'to gain'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab_trajectory, tab_dietitian, tab_chat, tab_data = st.tabs([
    "📈 LSTM Trajectory Forecast",
    "🩺 AI Dietitian Analysis",
    "💬 Interactive Dietitian Chat",
    "📋 Historical Data & Logs"
])

# ----- TAB 1: TRAJECTORY & LSTM FORECAST -----
with tab_trajectory:
    st.subheader(f"📊 Weight Projection ({forecast_days}-Day Horizon)")
    
    # Create Plotly Chart
    fig = go.Figure()
    
    # 1. Historical Actual Weights (scatter points)
    fig.add_trace(go.Scatter(
        x=df_history['Date'],
        y=df_history['Weight_kg'],
        mode='markers',
        name='Daily Scale Weight',
        marker=dict(color='#94a3b8', size=5, opacity=0.7)
    ))
    
    # 2. Historical 7-day rolling trend
    df_history['Rolling_Avg'] = df_history['Weight_kg'].rolling(window=7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_history['Date'],
        y=df_history['Rolling_Avg'],
        mode='lines',
        name='7-Day Rolling Average',
        line=dict(color='#38bdf8', width=2.5)
    ))
    
    # 3. Fluctuation Confidence Interval (Water retention bound)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast['Date'], df_forecast['Date'][::-1]]),
        y=pd.concat([df_forecast['Upper_Bound_kg'], df_forecast['Lower_Bound_kg'][::-1]]),
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Water Weight Fluctuation Zone'
    ))
    
    # 4. LSTM Future Forecast Line
    # Bridge the last historical point with the first forecast point
    bridge_dates = [df_history['Date'].iloc[-1]] + df_forecast['Date'].tolist()
    bridge_weights = [df_history['Weight_kg'].iloc[-1]] + df_forecast['Forecasted_Weight_kg'].tolist()
    
    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_weights,
        mode='lines+markers',
        name='LSTM Trajectory Forecast',
        line=dict(color='#10b981', width=3, dash='dash'),
        marker=dict(size=4)
    ))
    
    # 5. Goal Line
    all_dates = list(df_history['Date']) + list(df_forecast['Date'])
    fig.add_trace(go.Scatter(
        x=[all_dates[0], all_dates[-1]],
        y=[goal_weight, goal_weight],
        mode='lines',
        name=f'Goal Target ({goal_weight} kg)',
        line=dict(color='#f43f5e', width=2, dash='dot')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.6)',
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date", showgrid=True, gridcolor='#334155'),
        yaxis=dict(title="Weight (kg)", showgrid=True, gridcolor='#334155')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Secondary Charts: Calories and Steps
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.caption("Historical Energy Deficit / Surplus (kcal/day)")
        fig_cal = go.Figure()
        colors = ['#10b981' if val < 0 else '#f59e0b' for val in df_history['Net_Calories']]
        fig_cal.add_trace(go.Bar(
            x=df_history['Date'],
            y=df_history['Net_Calories'],
            marker_color=colors,
            name="Net Balance"
        ))
        fig_cal.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Net kcal", showgrid=True, gridcolor='#334155')
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        
    with c_sub2:
        st.caption("Daily Steps Tracking")
        fig_steps = go.Figure()
        fig_steps.add_trace(go.Scatter(
            x=df_history['Date'],
            y=df_history['Steps'],
            line=dict(color='#a855f7', width=2),
            fill='tozeroy',
            fillcolor='rgba(168, 85, 247, 0.1)',
            name="Steps"
        ))
        fig_steps.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Steps", showgrid=True, gridcolor='#334155')
        )
        st.plotly_chart(fig_steps, use_container_width=True)

# ----- TAB 2: AI DIETITIAN ANALYSIS -----
with tab_dietitian:
    st.subheader("🩺 Clinical Nutrition & Recomposition Analysis")
    st.markdown("Dr. Elena Vance uses **LangChain** to synthesize your physical profile, target intake, and the **LSTM sequence forecast**.")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("✨ Generate / Refresh Assessment", type="primary", use_container_width=True)
    with col_info:
        if provider != "Demo / Simulated" and not api_key:
            st.warning("⚠️ No API Key entered in sidebar. Running in simulated demo mode.")
            
    # User Profile dict for LangChain
    profile_data = {
        "current_weight": latest_actual_weight,
        "goal_weight": goal_weight,
        "height": height,
        "age": age,
        "gender": gender,
        "activity_level": "Moderately Active"
    }
    
    # LSTM Summary dict for LangChain
    lstm_summary = {
        "final_weight": projected_final_weight,
        "total_change": net_change,
        "daily_net": net_caloric_balance,
        "planned_calories": planned_calories,
        "planned_protein": protein,
        "planned_carbs": carbs,
        "planned_fat": fat,
        "planned_steps": planned_steps,
        "confidence_band": 0.85
    }
    
    if run_analysis or "dietitian_report" not in st.session_state:
        with st.spinner("Dr. Vance is analyzing your trajectory and synthesizing recommendations..."):
            report = generate_dietitian_assessment(
                user_profile=profile_data,
                lstm_forecast_summary=lstm_summary,
                provider=provider,
                api_key=api_key,
                model_name=model_name
            )
            st.session_state.dietitian_report = report
            
    st.markdown(st.session_state.dietitian_report)

# ----- TAB 3: INTERACTIVE CHAT -----
with tab_chat:
    st.subheader("💬 Ask Your AI Sports Dietitian")
    st.caption("Ask questions about recipes, meal timing, hunger management, or how to hit your macros.")
    
    # Display message history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_prompt = st.chat_input("E.g., What should I eat before high-intensity training?")
    if user_prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        context_str = f"""
        User Stats: Current Wt: {latest_actual_weight}kg, Goal: {goal_weight}kg.
        Plan: {planned_calories} kcal ({protein}g P, {carbs}g C, {fat}g F), {planned_steps} steps.
        LSTM Forecast: Projected {projected_final_weight}kg in {forecast_days} days.
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chat_with_dietitian_response(
                    messages=st.session_state.chat_messages,
                    user_context=context_str,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name
                )
                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ----- TAB 4: HISTORICAL DATA & LOGS -----
with tab_data:
    st.subheader("📋 Raw Sequence Telemetry Data")
    st.dataframe(df_history, use_container_width=True)
    
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Historical Log (CSV)",
        data=csv_data,
        file_name="fitness_trajectory_data.csv",
        mime="text/csv"
    )
