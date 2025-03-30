import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOllama
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, List, Any

app = Flask(__name__)

# Load datasets
farmer_df = pd.read_csv('farmer_advisor_dataset.csv')
market_df = pd.read_csv('market_researcher_dataset.csv')

# Initialize Gemma 3 model from Ollama
llm = ChatOllama(model="gemma3:1b ", temperature=0.1)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Set up vector databases for each dataset
def setup_vector_db(file_path, collection_name):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    return db

farmer_db = setup_vector_db('farmer_advisor_dataset.csv', 'farmer_data')
market_db = setup_vector_db('market_researcher_dataset.csv', 'market_data')

# Farmer Agent Tools
def analyze_soil_conditions(query: str) -> str:
    """Analyze soil conditions based on pH, moisture and other factors to recommend optimal crops."""
    docs = farmer_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="""
        You are an agricultural soil expert. Based on the following data:
        
        {context}
        
        {query}
        
        Analyze the soil conditions and provide specific recommendations regarding:
        1. Suitable crops based on soil pH and moisture
        2. Necessary soil amendments if needed
        3. Best practices for maintaining soil health
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, query=query)
    return response

def predict_crop_yield(query: str) -> str:
    """Predict crop yield based on historical data, soil conditions, and weather patterns."""
    docs = farmer_db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Extract data for machine learning prediction
    soil_ph = float(query.split('soil_ph=')[1].split(',')[0]) if 'soil_ph=' in query else 7.0
    soil_moisture = float(query.split('soil_moisture=')[1].split(',')[0]) if 'soil_moisture=' in query else 50.0
    temperature = float(query.split('temperature=')[1].split(',')[0]) if 'temperature=' in query else 25.0
    rainfall = float(query.split('rainfall=')[1].split(',')[0]) if 'rainfall=' in query else 200.0
    
    # Simple yield prediction based on averages from similar farms
    similar_conditions = farmer_df[
        (farmer_df['Soil_pH'].between(soil_ph-0.5, soil_ph+0.5)) &
        (farmer_df['Soil_Moisture'].between(soil_moisture-10, soil_moisture+10)) &
        (farmer_df['Temperature_C'].between(temperature-5, temperature+5)) &
        (farmer_df['Rainfall_mm'].between(rainfall-50, rainfall+50))
    ]
    
    if len(similar_conditions) > 0:
        avg_yield = similar_conditions['Crop_Yield_ton'].mean()
        top_crops = similar_conditions.groupby('Crop_Type')['Crop_Yield_ton'].mean().sort_values(ascending=False).head(3)
        top_crops_str = "\n".join([f"{crop}: {yield_val:.2f} tons/acre" for crop, yield_val in top_crops.items()])
        prediction = f"Predicted average yield: {avg_yield:.2f} tons/acre\nTop performing crops:\n{top_crops_str}"
    else:
        prediction = "Insufficient data for prediction with these exact parameters. Consider broadening your search."
    
    prompt = PromptTemplate(
        input_variables=["context", "prediction", "query"],
        template="""
        You are an agricultural yield prediction expert. Based on the following data:
        
        {context}
        
        And our initial statistical prediction:
        {prediction}
        
        Please analyze the query: {query}
        
        Provide a comprehensive yield prediction that includes:
        1. Expected yield ranges for suitable crops
        2. Factors that may impact yield positively or negatively
        3. Recommendations to optimize yield
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, prediction=prediction, query=query)
    return response

# Weather Station Agent Tools
def forecast_weather_impact(query: str) -> str:
    """Forecast weather patterns and their potential impact on various crops."""
    # Extract location data if available
    location = query.split('location=')[1].split(',')[0] if 'location=' in query else "unknown"
    
    # Get weather impact scores from market data
    weather_impacts = market_df['Weather_Impact_Score'].describe().to_dict()
    
    prompt = PromptTemplate(
        input_variables=["weather_data", "query"],
        template="""
        You are a meteorological expert specializing in agricultural weather forecasting.
        
        Weather impact statistics: {weather_data}
        
        For the query: {query}
        
        Provide a detailed analysis of:
        1. How current and forecasted weather patterns will affect different crop types
        2. Risk assessment for extreme weather events
        3. Recommendations for weather-adaptive farming strategies
        4. How to mitigate potential negative weather impacts
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(weather_data=str(weather_impacts), query=query)
    return response

def recommend_water_management(query: str) -> str:
    """Recommend optimal water management strategies based on crop type and conditions."""
    docs = farmer_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Extract water-related data
    soil_moisture = float(query.split('soil_moisture=')[1].split(',')[0]) if 'soil_moisture=' in query else 50.0
    rainfall = float(query.split('rainfall=')[1].split(',')[0]) if 'rainfall=' in query else 200.0
    
    prompt = PromptTemplate(
        input_variables=["context", "soil_moisture", "rainfall", "query"],
        template="""
        You are a water management and irrigation expert. Based on:
        
        Reference data: {context}
        Current soil moisture: {soil_moisture}%
        Expected rainfall: {rainfall}mm
        
        And the query: {query}
        
        Provide detailed recommendations for:
        1. Optimal irrigation scheduling and methods
        2. Water conservation techniques appropriate for the conditions
        3. Drainage requirements if applicable
        4. Sustainable water harvesting possibilities
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, soil_moisture=soil_moisture, rainfall=rainfall, query=query)
    return response

# Market Researcher Agent Tools
def analyze_market_trends(query: str) -> str:
    """Analyze current market trends and pricing for various agricultural products."""
    docs = market_db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Extract specific crop if mentioned
    crop_type = query.split('crop=')[1].split(',')[0] if 'crop=' in query else None
    
    if crop_type:
        crop_market_data = market_df[market_df['Product'] == crop_type]
        if len(crop_market_data) > 0:
            avg_price = crop_market_data['Market_Price_per_ton'].mean()
            demand_index = crop_market_data['Demand_Index'].mean()
            supply_index = crop_market_data['Supply_Index'].mean()
            consumer_trend = crop_market_data['Consumer_Trend_Index'].mean()
            
            market_summary = f"""
            Market summary for {crop_type}:
            - Average price: ${avg_price:.2f} per ton
            - Demand index: {demand_index:.2f} (higher is stronger demand)
            - Supply index: {supply_index:.2f} (higher is greater supply)
            - Consumer trend index: {consumer_trend:.2f} (higher indicates growing popularity)
            """
        else:
            market_summary = f"No specific market data available for {crop_type}."
    else:
        # General market overview
        top_products = market_df.groupby('Product')['Market_Price_per_ton'].mean().sort_values(ascending=False).head(5)
        top_demand = market_df.groupby('Product')['Demand_Index'].mean().sort_values(ascending=False).head(5)
        
        market_summary = f"""
        Top 5 highest-priced products:
        {top_products.to_string()}
        
        Top 5 products by demand:
        {top_demand.to_string()}
        """
    
    prompt = PromptTemplate(
        input_variables=["context", "market_summary", "query"],
        template="""
        You are an agricultural market analyst expert. Based on:
        
        Reference data: {context}
        
        Market summary: {market_summary}
        
        For the query: {query}
        
        Provide a detailed market analysis including:
        1. Price trends and projections
        2. Supply and demand dynamics
        3. Market opportunities for farmers
        4. Strategic recommendations for crop selection based on market conditions
        5. Potential risks and how to mitigate them
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, market_summary=market_summary, query=query)
    return response

def recommend_sustainable_practices(query: str) -> str:
    """Recommend sustainable farming practices to improve sustainability score."""
    docs = farmer_db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Calculate average sustainability metrics
    avg_fertilizer = farmer_df['Fertilizer_Usage_kg'].mean()
    avg_pesticide = farmer_df['Pesticide_Usage_kg'].mean()
    avg_sustainability = farmer_df['Sustainability_Score'].mean()
    
    # Find top sustainable farms
    top_sustainable = farmer_df.sort_values('Sustainability_Score', ascending=False).head(5)
    top_sustainable_summary = top_sustainable[['Farm_ID', 'Crop_Type', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Sustainability_Score']].to_string()
    
    prompt = PromptTemplate(
        input_variables=["context", "sustainability_metrics", "top_farms", "query"],
        template="""
        You are an expert in sustainable agriculture and agroecology. Based on:
        
        Reference data: {context}
        
        Sustainability metrics:
        - Average fertilizer usage: {sustainability_metrics[0]} kg
        - Average pesticide usage: {sustainability_metrics[1]} kg
        - Average sustainability score: {sustainability_metrics[2]}
        
        Top performing sustainable farms:
        {top_farms}
        
        For the query: {query}
        
        Provide comprehensive recommendations for:
        1. Reducing chemical inputs while maintaining productivity
        2. Implementing regenerative agriculture practices
        3. Improving biodiversity and ecosystem services
        4. Water conservation techniques
        5. Reducing carbon footprint and improving climate resilience
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(
        context=context, 
        sustainability_metrics=[avg_fertilizer, avg_pesticide, avg_sustainability], 
        top_farms=top_sustainable_summary, 
        query=query
    )
    return response

# Create tools for each agent
farmer_tools = [
    Tool(
        name="SoilAnalysisTool",
        func=analyze_soil_conditions,
        description="Analyze soil conditions to recommend optimal crops"
    ),
    Tool(
        name="CropYieldPredictionTool",
        func=predict_crop_yield,
        description="Predict crop yield based on conditions"
    )
]

weather_tools = [
    Tool(
        name="WeatherForecastTool",
        func=forecast_weather_impact,
        description="Forecast weather impacts on crops"
    ),
    Tool(
        name="WaterManagementTool",
        func=recommend_water_management,
        description="Recommend water management strategies"
    )
]

market_tools = [
    Tool(
        name="MarketAnalysisTool",
        func=analyze_market_trends,
        description="Analyze market trends for agricultural products"
    ),
    Tool(
        name="SustainabilityPracticesTool",
        func=recommend_sustainable_practices,
        description="Recommend sustainable farming practices"
    )
]

# Initialize agents
farmer_agent = initialize_agent(
    farmer_tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=True
)

weather_agent = initialize_agent(
    weather_tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=True
)

market_agent = initialize_agent(
    market_tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=True
)

# Initialize expert coordinator agent
def run_coordinator(query: str, agents_data: Dict) -> str:
    """Coordinator agent that orchestrates the other agents and synthesizes their inputs."""
    
    coordinator_system_message = """
    You are the AgriCoordinator, an expert agricultural advisor that coordinates multiple specialized agents
    to provide comprehensive, data-driven farming recommendations. Your goal is to synthesize inputs from 
    farmer advisors, weather specialists, and market researchers to create optimal sustainable farming plans.
    
    For each query:
    1. Determine which specialized agents should be consulted
    2. Evaluate and integrate their recommendations
    3. Resolve any conflicts between recommendations
    4. Prioritize sustainability and economic viability
    5. Provide a final recommendation that balances all factors
    """
    
    coordinator_prompt = PromptTemplate(
        input_variables=["farmer_insights", "weather_insights", "market_insights", "query"],
        template="""
        Based on the following specialized insights:
        
        FARMER ADVISOR INSIGHTS:
        {farmer_insights}
        
        WEATHER SPECIALIST INSIGHTS:
        {weather_insights}
        
        MARKET RESEARCHER INSIGHTS:
        {market_insights}
        
        And considering the original query: {query}
        
        Provide a comprehensive, integrated recommendation that:
        1. Identifies the optimal crop choices considering all factors
        2. Outlines specific sustainable farming practices tailored to the situation
        3. Provides a clear implementation plan with prioritized actions
        4. Addresses potential risks and mitigation strategies
        5. Balances environmental sustainability with economic viability
        
        Your recommendation should be specific, actionable, and data-driven.
        """
    )
    
    coordinator_chain = LLMChain(llm=llm, prompt=coordinator_prompt)
    
    response = coordinator_chain.run(
        farmer_insights=agents_data.get('farmer', 'No farmer insights available'),
        weather_insights=agents_data.get('weather', 'No weather insights available'),
        market_insights=agents_data.get('market', 'No market insights available'),
        query=query
    )
    
    return response

# API Routes
@app.route('/api/recommend', methods=['POST'])
def recommend_crops():
    data = request.json
    query = data.get('query', '')
    
    # Check for required parameters
    required_params = ['soil_ph', 'soil_moisture', 'temperature', 'rainfall']
    missing_params = [param for param in required_params if param not in data]
    
    if missing_params:
        return jsonify({
            'error': f"Missing required parameters: {', '.join(missing_params)}",
            'required_params': required_params
        }), 400
    
    # Format query with parameters
    formatted_query = f"{query}, soil_ph={data['soil_ph']}, soil_moisture={data['soil_moisture']}, " \
                      f"temperature={data['temperature']}, rainfall={data['rainfall']}"
    
    if 'location' in data:
        formatted_query += f", location={data['location']}"
    
    if 'crop' in data:
        formatted_query += f", crop={data['crop']}"
    
    # Run specialized agents in parallel (in a production environment)
    # For simplicity, we're running them sequentially here
    farmer_response = farmer_agent.run(formatted_query)
    weather_response = weather_agent.run(formatted_query)
    market_response = market_agent.run(formatted_query)
    
    # Coordinate the responses
    agents_data = {
        'farmer': farmer_response,
        'weather': weather_response,
        'market': market_response
    }
    
    final_recommendation = run_coordinator(formatted_query, agents_data)
    
    # Extract recommended crops from final recommendation
    # This is a simple extraction that would be more sophisticated in production
    recommended_crops = []
    for line in final_recommendation.split('\n'):
        if 'recommend' in line.lower() and any(crop.lower() in line.lower() for crop in set(farmer_df['Crop_Type'])):
            for crop in set(farmer_df['Crop_Type']):
                if crop.lower() in line.lower():
                    recommended_crops.append(crop)
    
    if not recommended_crops:
        # Fallback to find any mentioned crops
        for crop in set(farmer_df['Crop_Type']):
            if crop.lower() in final_recommendation.lower():
                recommended_crops.append(crop)
    
    # Calculate sustainability scores for recommended crops
    sustainability_data = {}
    for crop in recommended_crops:
        crop_data = farmer_df[farmer_df['Crop_Type'] == crop]
        if len(crop_data) > 0:
            avg_score = crop_data['Sustainability_Score'].mean()
            sustainability_data[crop] = {
                'sustainability_score': round(avg_score, 2),
                'avg_yield': round(crop_data['Crop_Yield_ton'].mean(), 2),
                'avg_fertilizer': round(crop_data['Fertilizer_Usage_kg'].mean(), 2),
                'avg_pesticide': round(crop_data['Pesticide_Usage_kg'].mean(), 2)
            }
    
    # Get market data for recommended crops
    market_data = {}
    for crop in recommended_crops:
        crop_market = market_df[market_df['Product'] == crop]
        if len(crop_market) > 0:
            market_data[crop] = {
                'avg_price': round(crop_market['Market_Price_per_ton'].mean(), 2),
                'demand_index': round(crop_market['Demand_Index'].mean(), 2),
                'supply_index': round(crop_market['Supply_Index'].mean(), 2),
                'consumer_trend': round(crop_market['Consumer_Trend_Index'].mean(), 2)
            }
    
    response = {
        'recommendation': final_recommendation,
        'recommended_crops': recommended_crops,
        'sustainability_data': sustainability_data,
        'market_data': market_data,
        'agent_insights': {
            'farmer_insights': farmer_response,
            'weather_insights': weather_response,
            'market_insights': market_response
        }
    }
    
    return jsonify(response)

@app.route('/api/analyze_soil', methods=['POST'])
def analyze_soil():
    data = request.json
    query = data.get('query', '')
    
    if 'soil_ph' in data and 'soil_moisture' in data:
        query += f", soil_ph={data['soil_ph']}, soil_moisture={data['soil_moisture']}"
    
    response = analyze_soil_conditions(query)
    return jsonify({'analysis': response})

@app.route('/api/forecast_weather', methods=['POST'])
def forecast_weather():
    data = request.json
    query = data.get('query', '')
    
    if 'location' in data:
        query += f", location={data['location']}"
    
    response = forecast_weather_impact(query)
    return jsonify({'forecast': response})

@app.route('/api/market_analysis', methods=['POST'])
def market_analysis():
    data = request.json
    query = data.get('query', '')
    
    if 'crop' in data:
        query += f", crop={data['crop']}"
    
    response = analyze_market_trends(query)
    return jsonify({'analysis': response})

@app.route('/api/sustainability', methods=['POST'])
def sustainability():
    data = request.json
    query = data.get('query', '')
    
    response = recommend_sustainable_practices(query)
    return jsonify({'recommendations': response})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'agents': ['farmer', 'weather', 'market', 'coordinator']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)