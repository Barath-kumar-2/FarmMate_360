# FarmMate_360
FarmMate 360 – Smart Crop Decision Support System
Overview

FarmMate 360 is a data-driven decision support system designed to help farmers select the most suitable crops based on soil conditions, weather parameters, and seasonal factors.

Instead of predicting a single crop, the system generates a ranked list of crops along with reasoning and estimated profitability, enabling more informed agricultural decisions.

Key Features
Crop recommendation using a Random Forest model
Ranked output of multiple crops (Top-N recommendations)
Profit estimation using basic market-based adjustments
Explainability using feature importance
Integration of soil, weather, and seasonal data
Data preprocessing and missing value handling
Tech Stack
Language: Python
Libraries: scikit-learn, pandas, numpy
Tools: Jupyter Notebook / VS Code
Dataset Features

The model uses the following inputs:

Soil Parameters

Nitrogen (N)
Phosphorus (P)
Potassium (K)
Soil moisture

Weather Parameters

Temperature
Rainfall
Humidity

Seasonal Information

Kharif
Rabi
Zaid
System Workflow
Data Collection
Soil, weather, and seasonal inputs are collected
Preprocessing
Missing values handled and features prepared
Model Training
Random Forest model trained on crop dataset
Prediction
Model outputs probability scores for each crop
Ranking
Crops sorted based on suitability scores
Business Logic Layer
Profit estimation adjusted using basic supply-demand assumptions
Explainability
Feature importance highlights key influencing factors
