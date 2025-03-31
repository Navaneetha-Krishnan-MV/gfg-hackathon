# --- START OF FILE app_ml_no_vectordb.py ---
import traceback
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
# Core Langchain imports remain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.chat_models import ChatOllama # Use community for Ollama
# Removed Embedding/VectorDB imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage # HumanMessage might not be needed if not directly used
from typing import Dict, List, Any
import re
import joblib

app = Flask(__name__)

# --- Configuration ---
MODEL_NAME = "gemma3:1b"
# EMBEDDING_MODEL_NAME no longer needed
FARMER_CSV = 'farmer_advisor_dataset.csv'
MARKET_CSV = 'market_researcher_dataset.csv'
# VECTOR_DB config removed
YIELD_MODEL_FILE = 'crop_yield_prediction_model.joblib'
SUSTAINABILITY_MODEL_FILE = 'sustainability_score_prediction_model.joblib'

# --- Load Datasets ---
try:
    farmer_df = pd.read_csv(FARMER_CSV)
    market_df = pd.read_csv(MARKET_CSV)
    print("Datasets loaded successfully.")
    # Basic Cleaning & Default Calculation (for ML input)
    farmer_df.dropna(subset=['Crop_Yield_ton', 'Sustainability_Score'], inplace=True)
    numeric_cols_farmer = farmer_df.select_dtypes(include=np.number).columns
    for col in numeric_cols_farmer:
        if farmer_df[col].isnull().any():
            median_val = farmer_df[col].median()
            farmer_df[col].fillna(median_val, inplace=True)
    DEFAULT_FERTILIZER = farmer_df['Fertilizer_Usage_kg'].mean()
    DEFAULT_PESTICIDE = farmer_df['Pesticide_Usage_kg'].mean()
    print(f"Calculated defaults: Fertilizer={DEFAULT_FERTILIZER:.2f}, Pesticide={DEFAULT_PESTICIDE:.2f}")

    # Column checks can remain for safety
    required_farmer_cols = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Farm_ID']
    required_market_cols = ['Product', 'Market_Price_per_ton', 'Demand_Index', 'Supply_Index', 'Weather_Impact_Score', 'Consumer_Trend_Index']
    if not all(col in farmer_df.columns for col in required_farmer_cols):
        print(f"Warning: Farmer dataset might be missing expected columns.")
    if not all(col in market_df.columns for col in required_market_cols):
         print(f"Warning: Market dataset might be missing expected columns.")

except FileNotFoundError as e:
    print(f"Error loading dataset: {e}. Make sure CSV files are present.")
    exit()
except Exception as e:
    print(f"An error occurred during dataset loading/processing: {e}")
    exit()

# --- Load ML Models ---
yield_model = None
sustainability_model = None
YIELD_MODEL_FEATURES = [
    'Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
    'Crop_Type', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg'
]
SUSTAINABILITY_MODEL_FEATURES = [
    'Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
    'Crop_Type', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg',
    'Crop_Yield_ton'
]
try:
    yield_model = joblib.load(YIELD_MODEL_FILE)
    print(f"ML model '{YIELD_MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Yield ML model file '{YIELD_MODEL_FILE}' not found. ML predictions unavailable.")
except Exception as e:
    print(f"Error loading yield ML model: {e}. ML predictions unavailable.")
try:
    sustainability_model = joblib.load(SUSTAINABILITY_MODEL_FILE)
    print(f"ML model '{SUSTAINABILITY_MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Sustainability ML model file '{SUSTAINABILITY_MODEL_FILE}' not found. ML predictions unavailable.")
except Exception as e:
    print(f"Error loading sustainability ML model: {e}. ML predictions unavailable.")

# --- Initialize LLM ---
try:
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    print(f"Initialized ChatOllama with model: {MODEL_NAME}")
except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    exit()

# --- Vector DB Setup Removed ---

# Helper function to safely extract numeric parameters from query string
def extract_param(query: str, param_name: str, default: float) -> float:
    match = re.search(f"{param_name}=([+-]?\\d*\\.?\\d+)", query, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return default
    return default

# --- Simplified Agent Tools (No Vector DB Context) ---

def analyze_soil_conditions_simplified(query: str) -> str:
    """Analyzes soil conditions based *only* on parameters provided in the query (like soil_ph, soil_moisture) and general agricultural knowledge. Recommends suitable crops, amendments, and soil health practices."""
    print(f"Soil Analysis Tool (Simplified) triggered with query: {query[:100]}...")

    # Extract parameters for the prompt context
    soil_ph = extract_param(query, 'soil_ph', 7.0) # Provide default if not found
    soil_moisture = extract_param(query, 'soil_moisture', 50.0) # Provide default
    print(f"Extracted params for soil analysis: pH={soil_ph}, Moisture={soil_moisture}")

    prompt = PromptTemplate(
        input_variables=["soil_ph", "soil_moisture", "query"],
        template="""
        You are an agricultural soil expert. Based *only* on the provided parameters and the user's query, use your general knowledge to analyze the soil conditions and provide specific recommendations.

        Input Parameters:
        - Soil pH: {soil_ph}
        - Soil Moisture: {soil_moisture}%

        User Query: {query}

        Your Analysis Should Address:
        1. Suitable crops considering the provided pH and moisture levels, based on general crop requirements.
        2. Potential soil amendments needed (e.g., lime for low pH, organic matter for moisture retention) based on the parameters.
        3. General best practices for maintaining soil health relevant to the potential issues indicated by the parameters.

        Clearly state that your analysis is based on general knowledge and the provided parameters, not specific local soil samples or detailed context.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        # Pass extracted parameters to the chain
        response = chain.run(soil_ph=soil_ph, soil_moisture=soil_moisture, query=query)
        print("Soil Analysis Tool (Simplified) finished.")
        return response
    except Exception as e:
        print(f"Error in Soil Analysis Tool (Simplified) LLM chain: {e}")
        return f"Error during simplified soil analysis: {e}"

def predict_crop_yield_stats_only(query: str) -> str:
    """Predicts potential crop yield based *only on simple statistical averages* from the historical dataset, considering soil conditions and weather factors mentioned (like soil_ph, soil_moisture, temperature, rainfall). Provides a discussion based on these stats."""
    print(f"Yield Prediction Tool (Stats Based) triggered with query: {query[:100]}...")

    # Extract parameters safely
    soil_ph = extract_param(query, 'soil_ph', 7.0)
    soil_moisture = extract_param(query, 'soil_moisture', 50.0)
    temperature = extract_param(query, 'temperature', 25.0)
    rainfall = extract_param(query, 'rainfall', 200.0)
    print(f"Extracted params for STATISTICAL yield prediction: pH={soil_ph}, Moisture={soil_moisture}, Temp={temperature}, Rainfall={rainfall}")

    # Simple statistical prediction based on DataFrame filtering (as before)
    ph_tolerance = 0.5
    moisture_tolerance = 10
    temp_tolerance = 5
    rainfall_tolerance = 50

    similar_conditions = farmer_df[
        (farmer_df['Soil_pH'].between(soil_ph - ph_tolerance, soil_ph + ph_tolerance)) &
        (farmer_df['Soil_Moisture'].between(soil_moisture - moisture_tolerance, soil_moisture + moisture_tolerance)) &
        (farmer_df['Temperature_C'].between(temperature - temp_tolerance, temperature + temp_tolerance)) &
        (farmer_df['Rainfall_mm'].between(rainfall - rainfall_tolerance, rainfall + rainfall_tolerance))
    ]

    statistical_prediction = "Insufficient similar historical data for a statistical yield estimate with the given parameters."
    if not similar_conditions.empty:
        avg_yield = similar_conditions['Crop_Yield_ton'].mean()
        top_crops = similar_conditions.groupby('Crop_Type')['Crop_Yield_ton'].mean().sort_values(ascending=False).head(3)
        # ... (rest of statistical prediction string generation is the same) ...
        if not np.isnan(avg_yield) and not top_crops.empty:
            top_crops_str = "\n".join([f"- {crop}: {yield_val:.2f} tons/acre (avg in similar conditions)" for crop, yield_val in top_crops.items()])
            statistical_prediction = f"Based on historical data with similar conditions (pH±{ph_tolerance}, Moist±{moisture_tolerance}%, Temp±{temp_tolerance}°C, Rain±{rainfall_tolerance}mm):\n" \
                                     f"- Average yield across suitable crops in these conditions: {avg_yield:.2f} tons/acre\n" \
                                     f"- Top performing crops in these conditions:\n{top_crops_str}"
        elif not np.isnan(avg_yield):
             statistical_prediction = f"Based on historical data with similar conditions:\n- Average yield across suitable crops: {avg_yield:.2f} tons/acre (No specific crop breakdown available)."
        elif not top_crops.empty:
             top_crops_str = "\n".join([f"- {crop}: {yield_val:.2f} tons/acre (avg in similar conditions)" for crop, yield_val in top_crops.items()])
             statistical_prediction = f"Based on historical data with similar conditions:\n- Top performing crops in these conditions:\n{top_crops_str} (Overall average yield calculation not possible)."

    print(f"Statistical Prediction: {statistical_prediction}")

    # Use LLM to discuss the statistical prediction
    prompt = PromptTemplate(
        input_variables=["statistical_prediction", "query"],
        template="""
        You are an agricultural yield prediction analyst. Discuss the following statistical yield prediction derived from historical data, in the context of the user's query.

        Statistical Prediction based on Historical Averages:
        ---
        {statistical_prediction}
        ---

        User Query: {query}

        Your Discussion Should:
        1. Explain the statistical prediction clearly.
        2. Discuss potential yield ranges based *only* on these historical averages.
        3. Mention factors from the query (pH, moisture, temp, rain) and how they likely influenced this statistical outcome.
        4. Briefly suggest general ways yield could be optimized, acknowledging this is based on averages, not detailed context.
        5. State the limitations (based only on historical averages, not detailed real-time data or deep context).
        (Note: A separate ML prediction might provide a different estimate).
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(statistical_prediction=statistical_prediction, query=query)
        print("Yield Prediction Tool (Stats Based) finished.")
        return response
    except Exception as e:
        print(f"Error in Yield Prediction Tool (Stats Based) LLM chain: {e}")
        return f"Error during statistical yield discussion: {e}"

def forecast_weather_impact(query: str) -> str:
    # This tool remains largely the same as it didn't rely on farmer/market vector dbs
    """Forecasts potential weather pattern impacts on crops using general knowledge and market dataset stats, considering general risk and mitigation. Use this for weather effects/risk assessment."""
    print(f"Weather Impact Tool triggered with query: {query[:100]}...")
    if market_df.empty or 'Weather_Impact_Score' not in market_df.columns:
         return "Market data with weather impact information is not available."

    location_match = re.search(r"location=([\w\s-]+)", query, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else "the specified region"
    print(f"Location context for weather impact: {location}")

    weather_impact_stats = market_df['Weather_Impact_Score'].describe().to_dict()
    weather_data_summary = f"General Weather Impact Context (Market Data Scores):\n" \
                           f"- Avg Score: {weather_impact_stats.get('mean', 'N/A'):.2f}, Std Dev: {weather_impact_stats.get('std', 'N/A'):.2f}, Range: {weather_impact_stats.get('min', 'N/A'):.2f}-{weather_impact_stats.get('max', 'N/A'):.2f}\n" \
                           f"(Note: Score interpretation depends on context, higher might mean more disruption)."
    print(f"Weather Impact Summary: {weather_data_summary}")

    prompt = PromptTemplate(
        input_variables=["weather_data_summary", "location", "query"],
        template="""
        You are a meteorological expert specializing in agricultural weather impacts. Based on general weather knowledge and the summary statistics provided, provide an analysis for the user's query about {location}.

        General Weather Impact Context (Historical Market Data Scores):
        ---
        {weather_data_summary}
        ---

        User Query: {query}

        Your analysis for {location} should include:
        1. General discussion of how typical weather patterns might affect major crop types.
        2. General risk assessment for potential extreme weather events.
        3. General recommendations for weather-adaptive farming strategies.
        4. General ways to mitigate potential negative weather impacts.

        Acknowledge you lack a live forecast. Base response on general principles and the provided stats.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(weather_data_summary=weather_data_summary, location=location, query=query)
        print("Weather Impact Tool finished.")
        return response
    except Exception as e:
        print(f"Error in Weather Impact Tool LLM chain: {e}")
        return f"Error during weather impact analysis: {e}"

def recommend_water_management_simplified(query: str) -> str:
    """Recommends optimal water management and irrigation strategies based *only* on parameters in the query (soil_moisture, rainfall) and general agricultural/hydrology knowledge."""
    print(f"Water Management Tool (Simplified) triggered with query: {query[:100]}...")

    # Extract parameters safely
    soil_moisture = extract_param(query, 'soil_moisture', 50.0)
    rainfall = extract_param(query, 'rainfall', 0.0)
    print(f"Extracted params for water management: Moisture={soil_moisture}, Rainfall={rainfall}")

    prompt = PromptTemplate(
        input_variables=["soil_moisture", "rainfall", "query"],
        template="""
        You are a water management and irrigation expert. Based *only* on the provided parameters and the user query, use your general knowledge to provide water management recommendations.

        Input Conditions Provided:
        - Soil Moisture: {soil_moisture}%
        - Expected Rainfall (if specified): {rainfall}mm

        User Query: {query}

        Provide detailed recommendations covering:
        1. General optimal irrigation scheduling and methods considering the soil moisture and rainfall.
        2. General water conservation techniques suitable for such conditions (e.g., mulching, drip irrigation).
        3. Potential drainage considerations based on the parameters.
        4. How these practices generally relate to sustainability.

        State that recommendations are based on general principles and the inputs, not specific field data.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(soil_moisture=soil_moisture, rainfall=rainfall, query=query)
        print("Water Management Tool (Simplified) finished.")
        return response
    except Exception as e:
        print(f"Error in Water Management Tool (Simplified) LLM chain: {e}")
        return f"Error during simplified water management recommendation: {e}"

def analyze_market_trends_stats_only(query: str) -> str:
    """Analyzes market trends, pricing, supply/demand using *only* statistics calculated from the market dataset. Can focus on a specific crop if 'crop=<crop_name>' is in the query."""
    print(f"Market Analysis Tool (Stats Based) triggered with query: {query[:100]}...")

    # Extract specific crop if mentioned
    crop_match = re.search(r"crop=([\w\s]+)", query, re.IGNORECASE)
    crop_type = crop_match.group(1).strip() if crop_match else None
    print(f"Crop specified for market analysis: {crop_type}")

    market_summary = "No specific market summary generated."
    # ...(market summary generation logic based on market_df remains the same)...
    if market_df.empty: market_summary = "Market data frame is empty. Cannot generate summary."
    elif crop_type:
        # Analyze specific crop
        crop_market_data = market_df[market_df['Product'].str.contains(crop_type, case=False, na=False)]
        if not crop_market_data.empty:
            avg_price = crop_market_data['Market_Price_per_ton'].mean()
            avg_demand = crop_market_data['Demand_Index'].mean()
            avg_supply = crop_market_data['Supply_Index'].mean()
            avg_consumer_trend = crop_market_data['Consumer_Trend_Index'].mean()
            market_summary = f"""Market Summary for '{crop_type}' (Dataset Averages):
            - Avg Price: ${avg_price:.2f}/ton, Avg Demand Idx: {avg_demand:.2f}, Avg Supply Idx: {avg_supply:.2f}, Avg Consumer Trend: {avg_consumer_trend:.2f}"""
        else: market_summary = f"No specific market data found for a product containing the name '{crop_type}' in the dataset."
    else:
        # General market overview if no specific crop
        try:
            numeric_cols = ['Market_Price_per_ton', 'Demand_Index', 'Supply_Index']
            temp_market_df = market_df.copy()
            for col in numeric_cols:
                 if temp_market_df[col].dtype not in ['int64', 'float64']: temp_market_df[col] = pd.to_numeric(temp_market_df[col], errors='coerce')
            temp_market_df.dropna(subset=numeric_cols, inplace=True)
            if not temp_market_df.empty:
                top_products_price = temp_market_df.groupby('Product')['Market_Price_per_ton'].mean().sort_values(ascending=False).head(3)
                top_products_demand = temp_market_df.groupby('Product')['Demand_Index'].mean().sort_values(ascending=False).head(3)
                market_summary = f"""General Market Overview (Dataset Averages - Top 3):
                Highest Priced: {top_products_price.to_string()}
                Highest Demand Idx: {top_products_demand.to_string()}"""
            else: market_summary = "Could not generate general market overview after data cleaning."
        except Exception as e: market_summary = f"Error generating general market overview: {e}"
    print(f"Market Summary generated: {market_summary}")

    # Use LLM to discuss the market summary
    prompt = PromptTemplate(
        input_variables=["market_summary", "query"],
        template="""
        You are an agricultural market analyst expert. Based *only* on the provided market summary statistics derived from a historical dataset, analyze the market situation relevant to the user's query.

        Market Summary Statistics (Historical Dataset Averages):
        ---
        {market_summary}
        ---

        User Query: {query}

        Provide a market analysis based *only* on interpreting these statistics:
        1. Interpretation of price, supply, demand based on the summary. Focus on specific crop if mentioned in summary.
        2. Potential opportunities or challenges hinted at by the averages (e.g., high demand vs high supply).
        3. Strategic considerations based *only* on these numbers.
        4. Potential risks evident from the averages (e.g., low price, high competition).

        Clearly state that this analysis is based *only* on historical averages from the dataset, not live data or external context.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(market_summary=market_summary, query=query)
        print("Market Analysis Tool (Stats Based) finished.")
        return response
    except Exception as e:
        print(f"Error in Market Analysis Tool (Stats Based) LLM chain: {e}")
        return f"Error during market stats analysis: {e}"

def recommend_sustainable_practices_simplified(query: str) -> str:
    """Recommends sustainable farming practices based *only* on general knowledge and potentially referencing average statistics calculated from the farmer dataset (like avg fertilizer/pesticide use, avg sustainability score)."""
    print(f"Sustainability Tool (Simplified) triggered with query: {query[:100]}...")

    # Calculate average sustainability metrics from the farmer dataframe (as before)
    sustainability_metrics_summary = "Could not calculate sustainability metrics from farmer data."
    # ...(calculation logic remains the same)...
    if not farmer_df.empty and all(col in farmer_df.columns for col in ['Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Sustainability_Score']):
        try:
            avg_fertilizer = pd.to_numeric(farmer_df['Fertilizer_Usage_kg'], errors='coerce').mean()
            avg_pesticide = pd.to_numeric(farmer_df['Pesticide_Usage_kg'], errors='coerce').mean()
            avg_sustainability = pd.to_numeric(farmer_df['Sustainability_Score'], errors='coerce').mean()
            sustainability_metrics_summary = f"Dataset Sustainability Averages:\n" \
                                             f"- Avg Fertilizer: {avg_fertilizer:.2f} kg, Avg Pesticide: {avg_pesticide:.2f} kg, Avg Score: {avg_sustainability:.2f}"
        except Exception as e: sustainability_metrics_summary = f"Error calculating metrics: {e}"
    print(f"Sustainability Metrics Summary: {sustainability_metrics_summary}")

    # Find examples of top sustainable farms from the dataframe (as before)
    top_farms_summary = "Could not identify top sustainable farms from data."
    # ...(calculation logic remains the same)...
    if not farmer_df.empty and 'Sustainability_Score' in farmer_df.columns:
         try:
             temp_farmer_df = farmer_df.copy()
             temp_farmer_df['Sustainability_Score_Num'] = pd.to_numeric(temp_farmer_df['Sustainability_Score'], errors='coerce')
             temp_farmer_df.dropna(subset=['Sustainability_Score_Num'], inplace=True)
             if not temp_farmer_df.empty:
                 top_sustainable = temp_farmer_df.sort_values('Sustainability_Score_Num', ascending=False).head(3)
                 cols_to_show = ['Farm_ID', 'Crop_Type', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Sustainability_Score']
                 cols_present = [col for col in cols_to_show if col in top_sustainable.columns]
                 if cols_present: top_farms_summary = f"Examples of Top Performing Sustainable Farms (Dataset Avg):\n{top_sustainable[cols_present].to_string(index=False)}"
                 else: top_farms_summary = "Top farms found, but cols missing."
             else: top_farms_summary = "No valid scores found."
         except Exception as e: top_farms_summary = f"Error identifying top farms: {e}"
    print(f"Top Farms Summary: {top_farms_summary}")

    # Use LLM with stats/examples and general knowledge
    prompt = PromptTemplate(
        input_variables=["sustainability_metrics_summary", "top_farms_summary", "query"],
        template="""
        You are an expert in sustainable agriculture. Based on general agricultural knowledge, the user's query, and the provided *summary statistics* from a historical dataset, provide recommendations for sustainable practices.

        General Sustainability Metrics & Examples from Dataset (Historical Averages):
        ---
        {sustainability_metrics_summary}
        {top_farms_summary}
        ---

        User Query: {query}

        Provide comprehensive recommendations for sustainable practices, using your general knowledge and referencing the dataset averages where relevant:
        1. Suggest ways to reduce chemical inputs, possibly comparing to the dataset averages.
        2. Recommend general practices for improving soil health and biodiversity.
        3. Discuss general water conservation techniques.
        4. Mention general strategies to improve sustainability, potentially aiming for better-than-average scores based on the dataset stats.
        5. Briefly touch upon general climate resilience practices.

        Focus on actionable advice. Clearly state that recommendations are based on general principles and historical dataset averages, not detailed site-specific context.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(
            sustainability_metrics_summary=sustainability_metrics_summary,
            top_farms_summary=top_farms_summary,
            query=query
        )
        print("Sustainability Tool (Simplified) finished.")
        return response
    except Exception as e:
        print(f"Error in Sustainability Tool (Simplified) LLM chain: {e}")
        return f"Error during simplified sustainability recommendation: {e}"


# --- Create Tools for Agents (Using Simplified Functions) ---
farmer_tools = [
    Tool( name="SoilAnalysisTool", func=analyze_soil_conditions_simplified, description="Analyzes soil based on input parameters (pH, moisture) and general knowledge."),
    Tool( name="CropYieldDiscussionTool", func=predict_crop_yield_stats_only, description="Discusses potential yield based ONLY on historical statistical averages from the dataset and input parameters."),
    Tool( name="SustainabilityPracticesTool", func=recommend_sustainable_practices_simplified, description="Recommends sustainable techniques based on general knowledge and dataset averages.")
]

weather_tools = [
    Tool(name="WeatherForecastImpactTool", func=forecast_weather_impact, description="Analyzes potential weather impacts, risks, and adaptive strategies using general knowledge and market stats."),
    Tool(name="WaterManagementTool", func=recommend_water_management_simplified, description="Recommends water/irrigation strategies based on input parameters (moisture, rainfall) and general knowledge.")
]

market_tools = [
    Tool(name="MarketAnalysisTool", func=analyze_market_trends_stats_only, description="Analyzes market trends using ONLY statistics calculated from the market dataset (optional specific crop)."),
    Tool( name="SustainabilityPracticesLinkTool", func=recommend_sustainable_practices_simplified, description="Finds sustainable practices based on general knowledge/dataset averages, potentially linking to marketability.")
]

# --- Initialize Agents ---
AGENT_TYPE = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
def create_agent(tools: List[Tool], system_message: str):
    # ... (Agent creation logic remains the same) ...
    if not tools:
        print(f"Warning: No tools provided for agent: {system_message[:50]}...")
        return None
    agent_kwargs = {"system_message": SystemMessage(content=system_message)}
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(
        tools, llm, agent=AGENT_TYPE, verbose=True, memory=memory,
        agent_kwargs=agent_kwargs, handle_parsing_errors="Check output format!", max_iterations=5 )

farmer_agent = create_agent(farmer_tools, "Farmer Advisor: Analyze farm conditions (soil params, yield stats), recommend practices using general knowledge.") if farmer_tools else None
weather_agent = create_agent(weather_tools, "Weather Specialist: Analyze weather impacts, recommend water management using general knowledge and stats.") if weather_tools else None
market_agent = create_agent(market_tools, "Market Researcher: Analyze market trends (dataset stats only), discuss sustainability market links using general knowledge.") if market_tools else None
print(f"Agents initialized: Farmer={bool(farmer_agent)}, Weather={bool(weather_agent)}, Market={bool(market_agent)}")


# --- Expert Coordinator (Updated Prompt - Less emphasis on agent's context) ---
coordinator_prompt_template = """
You are the AgriCoordinator, an expert agricultural advisor synthesizing insights from specialist agents AND specific ML predictions to provide comprehensive, actionable, and sustainable farming recommendations based primarily on the user's input parameters and general knowledge.

Your Task: Integrate the findings from the Farmer Advisor, Weather Specialist, Market Researcher (which are based on parameters, stats, and general knowledge) and the provided ML predictions regarding the user's query. Create a unified recommendation.

User's Original Query:
{query}

Insights Received from Agents (Based on parameters, stats, general knowledge):

--- FARMER ADVISOR INSIGHTS ---
{farmer_insights}
---------------------------------

--- WEATHER SPECIALIST INSIGHTS ---
{weather_insights}
-----------------------------------

--- MARKET RESEARCHER INSIGHTS ---
{market_insights}
------------------------------------

--- MACHINE LEARNING PREDICTIONS (Numeric Estimates) ---
Predicted Yield (ML Model): {ml_yield_prediction} tons/acre
Predicted Sustainability Score (ML Model): {ml_sustainability_prediction}
(Note: Use 'N/A' if prediction unavailable. Assumes default inputs if not specified.)
---------------------------------------------------------

Synthesized Recommendation:

Based on the agents' analyses AND the ML predictions, provide a final, integrated recommendation for the user's query ({query}). Your recommendation must:
1.  **Identify Optimal Crop Choices:** Recommend crops considering parameters (soil, weather from query), market stats (Market Agent), AND the ML yield prediction. Justify choices.
2.  **Outline Sustainable Practices:** Detail actionable sustainable practices based on agent suggestions and general knowledge. Connect these to the potential ML sustainability score.
3.  **Action Plan & Priorities:** Provide a clear, prioritized list of actions.
4.  **Risk Assessment & Mitigation:** Summarize risks identified by agents (weather, market stats) and suggest mitigation.
5.  **Holistic Conclusion:** Summarize how the plan balances sustainability (using ML score as a benchmark) with farmer goals (yield/profitability, using ML yield as benchmark).

Acknowledge that agent insights are now based on general knowledge/stats due to lack of deep context. Rely heavily on the ML predictions for quantitative estimates.
"""

coordinator_prompt = PromptTemplate(
    input_variables=["query", "farmer_insights", "weather_insights", "market_insights", "ml_yield_prediction", "ml_sustainability_prediction"],
    template=coordinator_prompt_template
)
coordinator_chain = LLMChain(llm=llm, prompt=coordinator_prompt)

def run_coordinator(query: str, agents_data: Dict[str, Any], ml_predictions: Dict[str, Any]) -> str:
    # ... (Coordinator execution logic remains the same) ...
    print("Running AgriCoordinator to synthesize insights and ML predictions...")
    try:
        response = coordinator_chain.run(
            query=query,
            farmer_insights=agents_data.get('farmer', 'No insights available from Farmer Advisor.'),
            weather_insights=agents_data.get('weather', 'No insights available from Weather Specialist.'),
            market_insights=agents_data.get('market', 'No insights available from Market Researcher.'),
            ml_yield_prediction=ml_predictions.get('yield', 'N/A'),
            ml_sustainability_prediction=ml_predictions.get('sustainability', 'N/A')
        )
        print("AgriCoordinator finished.")
        return response
    except Exception as e:
        print(f"Error running coordinator chain: {e}")
        return f"Error synthesizing recommendations: {e}"


# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health_check():
    print("--- HEALTH CHECK ROUTE ENTERED ---", flush=True) # Add entry log
    try:
        # --- Original health check logic ---
        active_agents = []
        if farmer_agent: active_agents.append('farmer')
        if weather_agent: active_agents.append('weather')
        if market_agent: active_agents.append('market')
        active_agents.append('coordinator')

        ml_model_status = {
            'yield_model': 'Loaded' if yield_model else 'Unavailable',
            'sustainability_model': 'Loaded' if sustainability_model else 'Unavailable'
        }
        # --- End original logic ---

        response_data = {
            'status': 'healthy',
            'message': 'API is running.',
            'model': MODEL_NAME,
            'active_agents': active_agents,
            'ml_model_status': ml_model_status
        }
        print("--- HEALTH CHECK SUCCESS ---", flush=True)
        return jsonify(response_data)

    except Exception as e:
        print(f"!!! ERROR IN /api/health: {e}", flush=True) # Print error
        traceback.print_exc() # Print full traceback TO THE FLASK CONSOLE
        print("--- HEALTH CHECK ERROR EXIT ---", flush=True)
        # Return a generic 500, but the console log is key
        return jsonify({'error': f'Internal server error in health check'}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_crops_and_practices():
    print("--- RECOMMEND ROUTE ENTERED ---", flush=True) # Add entry log
    try:
        # --- Original recommend logic ---
        data = request.json
        if not data or 'query' not in data:
            print("--- RECOMMEND ERROR: Missing query ---", flush=True)
            return jsonify({'error': "Missing 'query' in request body"}), 400

        query = data.get('query', '')
        print(f"\n--- New Request Received --- \nQuery: {query}")
        print(f"Input Data: {data}")

        # Parameter Extraction and Validation
        required_params = ['soil_ph', 'soil_moisture', 'temperature', 'rainfall']
        formatted_query_parts = [query]
        missing_params = []
        param_values = {}
        for param in required_params:
            if param in data:
                try:
                    value = float(data[param])
                    formatted_query_parts.append(f"{param}={value}")
                    param_values[param] = value
                except (ValueError, TypeError):
                    print(f"--- RECOMMEND ERROR: Invalid param type for {param} ---", flush=True)
                    return jsonify({'error': f"Invalid non-numeric value for '{param}'"}), 400
            else:
                missing_params.append(param)
        if missing_params:
            print(f"--- RECOMMEND ERROR: Missing params {missing_params} ---", flush=True)
            return jsonify({'error': f"Missing required parameters: {', '.join(missing_params)}",'required_params': required_params}), 400
        optional_params = ['location', 'crop']
        crop_type_input = 'Unknown'
        for param in optional_params:
            if param in data and data[param]:
                formatted_query_parts.append(f"{param}={data[param]}")
                param_values[param] = data[param]
                if param == 'crop': crop_type_input = data[param]
        formatted_query = ", ".join(formatted_query_parts)
        print(f"Formatted Query for Agents: {formatted_query}")

        # ML Model Prediction
        ml_yield_prediction = None
        ml_sustainability_prediction = None
        ml_predictions_for_coordinator = {'yield': 'N/A', 'sustainability': 'N/A'}
        # ...(rest of ML prediction logic remains the same)...
        if yield_model:
            try:
                input_data_yield = {k: [param_values.get(k.lower(), np.nan)] for k in YIELD_MODEL_FEATURES if k!= 'Crop_Type' and k!= 'Fertilizer_Usage_kg' and k!= 'Pesticide_Usage_kg'}
                input_data_yield['Crop_Type'] = [crop_type_input]
                input_data_yield['Fertilizer_Usage_kg'] = [float(data.get('fertilizer_usage', DEFAULT_FERTILIZER))]
                input_data_yield['Pesticide_Usage_kg'] = [float(data.get('pesticide_usage', DEFAULT_PESTICIDE))]
                for feature in YIELD_MODEL_FEATURES:
                    if feature not in input_data_yield:
                        if feature.lower() in param_values: input_data_yield[feature] = [param_values[feature.lower()]]
                        else: input_data_yield[feature] = [np.nan]; print(f"Warning: Feature '{feature}' missing for yield prediction, using NaN.")
                predict_df_yield = pd.DataFrame(input_data_yield)[YIELD_MODEL_FEATURES]
                print(f"Predicting yield with input: {predict_df_yield.to_dict('records')}")
                if predict_df_yield.isnull().values.any(): print("Warning: NaN values present in input for yield prediction.")
                prediction = yield_model.predict(predict_df_yield)[0]
                ml_yield_prediction = round(float(prediction), 2)
                ml_predictions_for_coordinator['yield'] = f"{ml_yield_prediction:.2f}"
                print(f"ML Yield Prediction successful: {ml_yield_prediction}")
            except KeyError as e: print(f"Error during ML yield prediction (KeyError): {e}")
            except Exception as e: print(f"Error during ML yield prediction: {e}")
        else: print("Yield ML model not loaded.")

        if sustainability_model and ml_yield_prediction is not None:
            try:
                input_data_sustainability = {k: [param_values.get(k.lower(), np.nan)] for k in SUSTAINABILITY_MODEL_FEATURES if k!= 'Crop_Type' and k!= 'Fertilizer_Usage_kg' and k!= 'Pesticide_Usage_kg' and k!= 'Crop_Yield_ton'}
                input_data_sustainability['Crop_Type'] = [crop_type_input]
                input_data_sustainability['Fertilizer_Usage_kg'] = [float(data.get('fertilizer_usage', DEFAULT_FERTILIZER))]
                input_data_sustainability['Pesticide_Usage_kg'] = [float(data.get('pesticide_usage', DEFAULT_PESTICIDE))]
                input_data_sustainability['Crop_Yield_ton'] = [ml_yield_prediction]
                for feature in SUSTAINABILITY_MODEL_FEATURES:
                     if feature not in input_data_sustainability:
                         if feature.lower() in param_values: input_data_sustainability[feature] = [param_values[feature.lower()]]
                         else: input_data_sustainability[feature] = [np.nan]; print(f"Warning: Feature '{feature}' missing for sustainability prediction, using NaN.")
                predict_df_sustainability = pd.DataFrame(input_data_sustainability)[SUSTAINABILITY_MODEL_FEATURES]
                print(f"Predicting sustainability with input: {predict_df_sustainability.to_dict('records')}")
                if predict_df_sustainability.isnull().values.any(): print("Warning: NaN values present in input for sustainability prediction.")
                prediction = sustainability_model.predict(predict_df_sustainability)[0]
                ml_sustainability_prediction = round(float(prediction), 2)
                ml_predictions_for_coordinator['sustainability'] = f"{ml_sustainability_prediction:.2f}"
                print(f"ML Sustainability Prediction successful: {ml_sustainability_prediction}")
            except KeyError as e: print(f"Error during ML sustainability prediction (KeyError): {e}")
            except Exception as e: print(f"Error during ML sustainability prediction: {e}")
        else: print("Sustainability ML model not loaded or yield prediction failed.")


        # Run Specialized Agents
        agents_data = {}
        agent_list = {'farmer': farmer_agent, 'weather': weather_agent, 'market': market_agent}
        for name, agent_instance in agent_list.items():
            print(f"\n--- Running {name.capitalize()} Agent ---")
            if agent_instance:
                try: agents_data[name] = agent_instance.run(formatted_query)
                except Exception as e: print(f"Error running {name.capitalize()} Agent: {e}"); agents_data[name] = f"Error: {e}"
            else: agents_data[name] = f"{name.capitalize()} agent is not available."
            print(f"--- {name.capitalize()} Agent Finished ---")

        # Coordinate the Responses
        print("\n--- Running Coordinator ---")
        final_recommendation = run_coordinator(query=query, agents_data=agents_data, ml_predictions=ml_predictions_for_coordinator )
        print("--- Coordinator Finished ---")

        # Post-processing & Response Formatting
        # ...(post-processing logic remains the same)...
        recommended_crops = []
        try:
            possible_crops = list(set(farmer_df['Crop_Type'].astype(str).unique()))
            recommendation_lower = final_recommendation.lower()
            for crop in possible_crops:
                if re.search(r'\b' + re.escape(crop.lower()) + r'\b', recommendation_lower):
                    if crop not in recommended_crops: recommended_crops.append(crop)
        except Exception as e: print(f"Error during heuristic crop extraction: {e}"); recommended_crops = []
        print(f"Extracted Recommended Crops (heuristic): {recommended_crops}")
        sustainability_data = {}
        market_data = {}
        if not farmer_df.empty:
            for crop in recommended_crops:
                try:
                    crop_data = farmer_df[farmer_df['Crop_Type'].str.contains(crop, case=False, na=False)]
                    if not crop_data.empty:
                        sustainability_data[crop] = {'avg_sustainability_score': round(pd.to_numeric(crop_data['Sustainability_Score'], errors='coerce').mean(), 2),'avg_yield_ton': round(pd.to_numeric(crop_data['Crop_Yield_ton'], errors='coerce').mean(), 2),'avg_fertilizer_kg': round(pd.to_numeric(crop_data['Fertilizer_Usage_kg'], errors='coerce').mean(), 2),'avg_pesticide_kg': round(pd.to_numeric(crop_data['Pesticide_Usage_kg'], errors='coerce').mean(), 2)}
                        sustainability_data[crop] = {k: (v if not pd.isna(v) else None) for k, v in sustainability_data[crop].items()}
                except Exception as e: print(f"Error calculating sustainability stats for {crop}: {e}"); sustainability_data[crop] = {'error': 'Stats calculation failed'}
        if not market_df.empty:
            for crop in recommended_crops:
                try:
                    crop_market = market_df[market_df['Product'].str.contains(crop, case=False, na=False)]
                    if not crop_market.empty:
                        market_data[crop] = {'avg_price_per_ton': round(pd.to_numeric(crop_market['Market_Price_per_ton'], errors='coerce').mean(), 2),'avg_demand_index': round(pd.to_numeric(crop_market['Demand_Index'], errors='coerce').mean(), 2),'avg_supply_index': round(pd.to_numeric(crop_market['Supply_Index'], errors='coerce').mean(), 2),'avg_consumer_trend_index': round(pd.to_numeric(crop_market['Consumer_Trend_Index'], errors='coerce').mean(), 2)}
                        market_data[crop] = {k: (v if not pd.isna(v) else None) for k, v in market_data[crop].items()}
                except Exception as e: print(f"Error calculating market stats for {crop}: {e}"); market_data[crop] = {'error': 'Stats calculation failed'}


        response = {
            'final_recommendation': final_recommendation,
            'ml_yield_prediction_tons_acre': ml_yield_prediction,
            'ml_sustainability_score_prediction': ml_sustainability_prediction,
            'recommended_crops_heuristic': recommended_crops,
            'sustainability_data_avg': sustainability_data,
            'market_data_avg': market_data,
            'agent_insights': agents_data
        }
        # --- End original logic ---

        print("--- RECOMMEND SUCCESS ---", flush=True)
        return jsonify(response)

    except Exception as e:
        print(f"!!! ERROR IN /api/recommend: {e}", flush=True) # Print error
        traceback.print_exc() # Print full traceback TO THE FLASK CONSOLE
        print("--- RECOMMEND ERROR EXIT ---", flush=True)
         # Return a generic 500, but the console log is key
        return jsonify({'error': f'Internal server error during recommendation'}), 500

# --- Optional Direct Routes Removed as they relied on specific old tool functions ---

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask application with ML integration (No Vector DB)...")
    app.run(debug=True, host='0.0.0.0', port=5000)

# --- END OF FILE app_ml_no_vectordb.py ---