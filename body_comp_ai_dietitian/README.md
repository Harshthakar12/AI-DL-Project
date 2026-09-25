# 🥗 AI Body Composition & Caloric Trajectory Forecaster

A hybrid AI fitness & nutrition application integrating:
1. **PyTorch LSTM Neural Network:** Models daily sequential caloric/macronutrient balance, activity steps, and water weight fluctuations to forecast multi-week weight trajectory.
2. **LangChain & LLMs (Gemini / OpenAI):** Clinical Sports Dietitian agent (`Dr. Elena Vance`) evaluating trajectory feasibility, macro distribution, and crafting 7-day meal plans.
3. **Interactive Streamlit Web Dashboard:** Interactive Plotly charts, water weight fluctuation confidence zones, macro calculators, and conversational AI chat.

---

## 🏗️ Architecture

```
[Daily Fitness Logs: Weight, Calories, Macros, Steps]
                     │
                     ▼
        [PyTorch LSTM Neural Network]
                     │
                     ▼
         [Forward Sequence Forecast]
                     │
                     ▼
       [LangChain Prompt & LLM Agent] ◄── User Physical Profile & Goals
                     │
                     ▼
       [Streamlit Interactive Frontend] ──► Charts, Meal Plans, & Chat
```

---

## 🚀 How to Run the App



. Run the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

 Open your browser at `http://localhost:8501`.

---

## 🔑 Key Features
* **Multi-Horizon Forecasting:** Predict weight outcomes over 14 to 60 days based on customized daily caloric targets.
* **Water Weight Fluctuation Zone:** Visualizes natural fluid retention spikes (from carbs and sodium) so users don't confuse water retention with fat gain.
* **Clinical Dietitian Agent:** Powered by LangChain, offering scientific critiques, meal plan blueprints, and interactive Q&A.
* **Flexible LLM Provider:** Switch between Google Gemini, OpenAI, or instant simulated Demo Mode.
