from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

def get_llm(provider: str, api_key: str, model_name: str = None):
    """
    Initializes the appropriate LangChain Chat Model based on provider and API key.
    """
    if not api_key:
        return None
        
    provider = provider.lower()
    if "gemini" in provider or "google" in provider:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_name or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.4
        )
    elif "openai" in provider:
        from langchain_openai import ChatOpenAI
        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.4
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def generate_dietitian_assessment(
    user_profile: Dict[str, Any],
    lstm_forecast_summary: Dict[str, Any],
    provider: str = "Demo / Simulated",
    api_key: str = "",
    model_name: str = ""
) -> str:
    """
    Uses LangChain to produce a clinical nutrition analysis and actionable recommendations
    based on user goals and the LSTM model's weight forecast.
    """
    # Check if real API key is present
    if api_key and provider != "Demo / Simulated":
        try:
            llm = get_llm(provider, api_key, model_name)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are Dr. Elena Vance, an elite Sports Dietitian and PhD Exercise Physiologist.
You specialize in body recomposition, evidence-based energy balance, and macronutrient periodization.

Analyze the user's historical fitness data and the LSTM neural network's forecasted trajectory.
Provide a clear, empowering, professional, and science-backed critique with the following sections:

### 1. 📊 Trajectory & Reality Check
Evaluate the LSTM's projected rate of change compared to the user's goal. Note whether the rate of weight loss/gain is safe and realistic (e.g. 0.5-1% body weight per week to preserve lean mass).

### 2. 🥩 Macronutrient & Caloric Periodization
Critique the proposed caloric deficit/surplus and macros (protein, carbs, fats). Recommend specific grams and explain the physiological rationale (e.g. muscle protein synthesis, glycogen replenishment, hormone balance).

### 3. 🍽️ 7-Day Precision Meal Plan Blueprint
Outline a structured, nutrient-dense daily meal framework with whole foods, timing suggestions (pre/post-workout), and hydration recommendations.

### 4. ⚠️ Potential Pitfalls & Water Weight Fluctuations
Explain how sodium, stress (cortisol), and carbohydrate intake might cause transient water weight spikes that diverge from the LSTM trend line, and how to stay psychologically resilient.
"""),
                ("human", """User Profile:
- Current Weight: {current_weight} kg
- Goal Weight: {goal_weight} kg
- Height: {height} cm
- Age: {age}, Gender: {gender}
- Activity Level: {activity_level}
- Target Daily Calories: {planned_calories} kcal
- Target Macros: {protein}g Protein | {carbs}g Carbs | {fat}g Fat
- Daily Step Target: {steps} steps

LSTM 30-Day Forecast Data:
- Projected Final Weight in 30 Days: {forecasted_final_weight} kg
- Total Projected Net Weight Change: {projected_change} kg ({change_direction})
- Estimated Calorie Deficit/Surplus: {daily_net} kcal/day
- Fluctuation Confidence Interval: +/- {confidence_band} kg

Please provide your comprehensive nutritional and trajectory assessment!""")
            ])
            
            chain = prompt_template | llm
            response = chain.invoke({
                "current_weight": user_profile.get("current_weight", 80),
                "goal_weight": user_profile.get("goal_weight", 75),
                "height": user_profile.get("height", 178),
                "age": user_profile.get("age", 28),
                "gender": user_profile.get("gender", "Male"),
                "activity_level": user_profile.get("activity_level", "Moderate"),
                "planned_calories": lstm_forecast_summary.get("planned_calories", 2000),
                "protein": lstm_forecast_summary.get("planned_protein", 160),
                "carbs": lstm_forecast_summary.get("planned_carbs", 180),
                "fat": lstm_forecast_summary.get("planned_fat", 60),
                "steps": lstm_forecast_summary.get("planned_steps", 10000),
                "forecasted_final_weight": lstm_forecast_summary.get("final_weight", 78.5),
                "projected_change": lstm_forecast_summary.get("total_change", -2.5),
                "change_direction": "Loss" if lstm_forecast_summary.get("total_change", 0) < 0 else "Gain",
                "daily_net": lstm_forecast_summary.get("daily_net", -500),
                "confidence_band": lstm_forecast_summary.get("confidence_band", 0.8)
            })
            return response.content
        except Exception as e:
            return f"⚠️ Error querying LLM: {str(e)}\n\nFalling back to simulated assessment below:\n\n" + _generate_mock_assessment(user_profile, lstm_forecast_summary)
            
    # Fallback / Demo Mode assessment
    return _generate_mock_assessment(user_profile, lstm_forecast_summary)

def _generate_mock_assessment(user_profile: Dict[str, Any], lstm_summary: Dict[str, Any]) -> str:
    curr_w = user_profile.get("current_weight", 82.5)
    goal_w = user_profile.get("goal_weight", 77.0)
    final_w = lstm_summary.get("final_weight", 80.1)
    tot_change = round(final_w - curr_w, 2)
    daily_net = lstm_summary.get("daily_net", -500)
    
    status = "on a sustainable fat loss pace" if tot_change < 0 else "in a progressive caloric surplus"
    
    return f"""### 1. 📊 Trajectory & Reality Check
* **Current Weight:** {curr_w} kg ➔ **LSTM 30-Day Projected:** {final_w} kg (Net Δ: **{tot_change} kg**)
* **Goal Target:** {goal_w} kg
* **Assessment:** The LSTM sequence forecast indicates you are **{status}**. A deficit of approximately **{abs(daily_net)} kcal/day** targets ~0.5 kg of fat loss per week, which sits in the gold-standard recommendation (0.5%–1.0% body weight/week) for preserving lean muscle mass and resting metabolic rate.

### 2. 🥩 Macronutrient & Caloric Periodization
* **Protein ({lstm_summary.get('planned_protein', 160)}g):** Excellent target (~2.0g/kg bodyweight). Keeps muscle protein synthesis (MPS) elevated while in a deficit.
* **Carbohydrates ({lstm_summary.get('planned_carbs', 180)}g):** Sufficient to replenish muscle glycogen for high-intensity resistance training sessions and prevent thyroid/leptin down-regulation.
* **Fats ({lstm_summary.get('planned_fat', 55)}g):** Provides essential fatty acids and supports hormone production (testosterone/estrogen synthesis) without spilling over into excess caloric storage.

### 3. 🍽️ 7-Day Precision Meal Plan Blueprint
* **Breakfast (Energy & Satiety):**
  * 3 whole eggs or egg whites scramble with spinach & mushrooms
  * 60g rolled oats with 1 scoop whey isolate + cinnamon & berries
* **Lunch (Sustained Glycogen):**
  * 180g grilled chicken breast or baked tofu
  * 150g steamed jasmine rice or sweet potato
  * Generous side of steamed broccoli & zucchini with 1 tsp olive oil
* **Pre/Post Workout Snack:**
  * 1 medium banana + 150g non-fat Greek yogurt or rice cakes with whey shake
* **Dinner (Recovery & Micronutrients):**
  * 200g wild salmon or lean 93/7 beef patty
  * Large rainbow salad with leafy greens, cucumber, and balsamic vinaigrette
* **Hydration:** Aim for 3.0 to 3.5 Liters of water daily.

### 4. ⚠️ Potential Pitfalls & Water Weight Fluctuations
* The LSTM model projects a smoothed metabolic trend, but your morning scale will fluctuate by **±0.6 to 1.2 kg** due to glycogen bound water (1g carb binds ~3g water) and sodium shifts.
* Do not panic if the scale jumps after a high-carb meal; measure your 7-day rolling average to verify true fat loss progress!
"""

def chat_with_dietitian_response(
    messages: List[Dict[str, str]],
    user_context: str,
    provider: str = "Demo / Simulated",
    api_key: str = "",
    model_name: str = ""
) -> str:
    """
    Handles interactive back-and-forth chat between the user and the AI Dietitian with memory.
    """
    if api_key and provider != "Demo / Simulated":
        try:
            llm = get_llm(provider, api_key, model_name)
            langchain_messages = [
                SystemMessage(content=f"""You are Dr. Elena Vance, an elite sports dietitian and nutrition coach.
You have access to the user's ongoing physical stats, diet logs, and LSTM trajectory forecast:
{user_context}

Be encouraging, scientifically precise, and practical. Offer concrete food swaps, recipe ideas, macro adjustments, and psychological tips.""")
            ]
            for m in messages:
                if m["role"] == "user":
                    langchain_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=m["content"]))
                    
            response = llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}. (Check your API Key or quota)."
            
    # Fallback simulated response
    last_user_query = messages[-1]["content"] if messages else ""
    return f"**Dr. Elena Vance (AI Dietitian - Demo Mode):**\n\nRegarding: *\"{last_user_query}\"*\n\nBased on your current LSTM caloric projection, remember that energy balance is the primary driver of body composition. If you're experiencing hunger spikes, increase fibrous vegetables (broccoli, leafy greens) and ensure you are hitting at least 30g of protein in your first meal to stabilize ghrelin (hunger hormone) throughout the day!\n\n*(To activate live personalized generative replies, enter your OpenAI or Google Gemini API key in the sidebar).* "
