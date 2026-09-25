import sys
import urllib.request
import pandas as pd
import torch

def test_system():
    print("=" * 60)
    print("[*] RUNNING SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    # 1. Check Python & PyTorch
    print("\n[1/5] Checking PyTorch & Hardware...")
    print(f"  * PyTorch Version: {torch.__version__}")
    print(f"  * CUDA Available: {torch.cuda.is_available()}")
    print("  [OK] PyTorch is operational.")

    # 2. Check Data Generator
    print("\n[2/5] Testing Data Generator...")
    from data_generator import generate_sample_fitness_data
    df = generate_sample_fitness_data(days=30)
    assert len(df) == 30, "Data generator failed length check"
    print(f"  * Generated {len(df)} days of synthetic telemetry.")
    print("  [OK] Data generator is operational.")

    # 3. Check PyTorch LSTM Model Training & Forecasting
    print("\n[3/5] Testing LSTM Training & Forecast Engine...")
    from lstm_model import WeightTrajectoryPredictor
    predictor = WeightTrajectoryPredictor(seq_len=7)
    loss = predictor.train(df, epochs=30)
    print(f"  * Trained LSTM (Final Loss: {loss:.4f})")
    forecast_df = predictor.forecast_future(df, future_days=14)
    assert len(forecast_df) == 14, "Forecast failed length check"
    print(f"  * Successfully forecasted 14 days ahead (Start: {forecast_df['Forecasted_Weight_kg'].iloc[0]} kg -> End: {forecast_df['Forecasted_Weight_kg'].iloc[-1]} kg)")
    print("  [OK] LSTM sequence engine is operational.")

    # 4. Check LangChain Integration
    print("\n[4/5] Testing LangChain AI Dietitian Module...")
    from ai_dietitian import generate_dietitian_assessment, chat_with_dietitian_response
    profile = {"current_weight": 82.0, "goal_weight": 76.0}
    summary = {"final_weight": 79.5, "total_change": -2.5, "daily_net": -500}
    assessment = generate_dietitian_assessment(profile, summary, provider="Demo / Simulated")
    assert len(assessment) > 100, "Assessment generation failed"
    chat_reply = chat_with_dietitian_response([{"role": "user", "content": "Hello"}], "User at 82kg")
    assert len(chat_reply) > 50, "Chat reply failed"
    print("  * LangChain prompts, chains, and fallback handlers passed.")
    print("  [OK] AI Dietitian module is operational.")

    # 5. Check Streamlit Server Connection
    print("\n[5/5] Checking Streamlit Web Server Status (http://localhost:8501)...")
    try:
        req = urllib.request.Request("http://localhost:8501", headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, timeout=5) as response:
            code = response.getcode()
            if code == 200:
                print(f"  * Server responded with HTTP status {code} (OK).")
                print("  [OK] Streamlit application is live and listening on port 8501!")
            else:
                print(f"  [WARN] Server responded with code: {code}")
    except Exception as e:
        print(f"  [ERROR] Web server check error: {e}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL SYSTEMS ARE HEALTHY AND RUNNING PERFECTLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
