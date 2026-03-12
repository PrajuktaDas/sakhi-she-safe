# SAKHI-SHE SAFE  
AI-Powered Women's Safety Intelligence System

Live Demo: https://sakhi-she-safe.onrender.com

---

## Overview

SAKHI-SHE SAFE is an AI-powered safety intelligence platform designed to transform crime data into actionable safety insights.

The system analyzes historical crime patterns and environmental indicators to help individuals, city authorities, and policymakers make safer mobility decisions.

By combining machine learning, spatial analysis, and decision intelligence, the platform can identify high-risk zones, recommend safer travel strategies, and provide insights that can help authorities allocate surveillance and policing resources more effectively.

The project aims to contribute to safer urban mobility, particularly for women.

---

## Why This Project Matters

Urban safety remains one of the most critical challenges in modern cities. Traditional navigation systems optimize for distance and speed but ignore safety indicators such as crime patterns, lighting conditions, and police presence.

SAKHI-SHE SAFE introduces a safety-first approach to urban mobility.

The platform can assist:

Citizens  
by recommending safer travel times and transport options.

City Authorities  
by highlighting high-risk zones that may require additional surveillance or infrastructure improvements.

Policy Makers  
by providing data-driven insights into crime distribution and safety patterns.

By converting raw crime data into meaningful intelligence, the system helps move toward proactive safety management rather than reactive crime response.

---

## Alignment with the MANAV Vision

The project aligns with the broader goals of the MANAV initiative highlighted in the Global Partnership Summit held in New Delhi.

MANAV emphasizes the use of technology, artificial intelligence, and data-driven solutions to enhance societal well-being and public safety.

SAKHI-SHE SAFE supports this vision by demonstrating how artificial intelligence can be applied to real-world societal challenges.

The platform illustrates how data science can help:

Improve public safety planning  
Enable intelligent urban infrastructure decisions  
Promote safer environments for women and vulnerable communities  

The project represents a practical example of how AI can contribute to human-centric innovation and social impact.

---

## Key Features

### Location Safety Check
Uses a trained Random Forest machine learning model to estimate safety risk for specific city and time conditions.

### Travel Recommendation
Analyzes environmental indicators such as lighting conditions, police presence, and population density to recommend safer transportation options.

### Best Time to Travel
Identifies safer travel windows by analyzing crime frequency across different times of the day.

### Safe Route Planner
Uses graph-based algorithms to identify routes with lower risk indicators.

### Crime Risk Heatmap
Visualizes crime intensity geographically using color gradients to highlight potential high-risk zones.

### Crime Trend Analysis
Displays historical trends in crime severity to help identify emerging patterns.

### Community Safety Feedback
Allows users to share safety experiences, enabling community-driven intelligence.

### Emergency Alert Simulation
Simulates emergency alert workflows to demonstrate how real-time alert systems could function.

---

## How This System Can Support Authorities

Beyond individual users, the platform can assist urban authorities by identifying potential target zones that may require increased monitoring.

Using spatial crime analysis, the system can highlight areas where:

Crime frequency is consistently higher  
Environmental safety indicators are weaker  
Risk patterns show temporal clustering

These insights can help authorities deploy targeted interventions such as:

Improved street lighting  
Increased police patrols  
Additional CCTV surveillance  
Community safety initiatives

This transforms raw crime data into actionable urban safety intelligence.

---

## System Architecture

Crime Dataset  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Machine Learning Model (Random Forest)  
↓  
Safety Risk Prediction  
↓  
Decision Intelligence Modules  
    Time Analyzer  
    Transport Recommendation Engine  
    Route Optimization System  
↓  
Streamlit Web Application

---

## Technology Stack

Programming Language  
Python

Machine Learning  
Scikit-learn

Data Processing  
Pandas  
NumPy

Visualization  
Plotly  
Streamlit

Graph Algorithms  
NetworkX

Model Persistence  
Joblib

---

## Project Structure

sakhi-she-safe/

app.py  
README.md  
requirements.txt  

data/  
    raw/  
    processed/  
    feedback/  

models/  
    safety_risk_rf_model.pkl  
    model_features.pkl  

src/  
    data_preprocessing.py  
    feature_engineering.py  
    clustering.py  
    risk_scoring.py  
    model_training.py  
    recommendation_engine.py  
    route_optimizer.py  
    time_analyzer.py  

---

## Installation

Clone the repository

git clone https://github.com/PrajuktaDas/sakhi-she-safe.git

Navigate to the project folder

cd sakhi-she-safe

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py

---

## Future Improvements

Integration with real-time crime datasets  
GPS-based safety navigation  
Mobile application version  
Integration with smart city infrastructure  
Real-time emergency response systems  

---

## Author

Prajukta Das
