# Integrated Soil-Microclimate Analysis for Automated Irrigation ML

This directory contains scripts and data files for processing and integrating soil quality data with microclimate variables to create comprehensive features for an automated irrigation ML model for root vegetables in northwestern Iraq.

## Overview of Contents

- `CalculateMicroclimateParameters.py` - Script for processing microclimate data and calculating irrigation-relevant parameters
- `IntegrateSoilAndMicroclimate.py` - Script for integrating soil quality data with microclimate variables 
- `data/` - Processed microclimate data
- `integrated_data/` - Integrated soil-climate datasets for ML
- `plots/` - Visualizations of microclimate parameters
- `integrated_plots/` - Visualizations of integrated soil-climate data

## Data Processing Workflow

### 1. Microclimate Data Processing

The `CalculateMicroclimateParameters.py` script processes the raw microclimate data (`MicroclimateDataSubset.csv`) to calculate:

- Reference evapotranspiration (ET0) using the FAO-56 Penman-Monteith equation
- Crop coefficients (Kc) for root vegetables based on growing season
- Crop evapotranspiration (ETc = Kc × ET0)
- Effective rainfall and net irrigation requirements
- Water deficit and aridity indices

The processed microclimate data is saved to `data/processed_microclimate.csv`.

### 2. Soil-Microclimate Integration

The `IntegrateSoilAndMicroclimate.py` script integrates the cleaned soil quality data (`SoilQuality_ML_Ready.csv`) with the processed microclimate data to:

- Calculate soil-climate interaction metrics
- Estimate irrigation requirements specific to soil types
- Create comprehensive features for ML models
- Visualize the relationships between soil properties and climate variables

Key soil-climate interaction metrics include:

- **Moisture Stress Index**: Indicates water stress based on climate and soil water capacity
- **Infiltration Efficiency**: Represents how effectively rainfall infiltrates different soil types
- **Evaporation Susceptibility**: Indicates vulnerability to water loss through evaporation
- **Irrigation Efficiency**: Measures how efficiently irrigation water can be used 
- **Days to Depletion**: Estimates days until available soil water is depleted
- **Irrigation Interval**: Recommended interval between irrigation events
- **Irrigation Amount**: Recommended amount per irrigation event

## Key Files Generated

### Data Files

- `processed_microclimate.csv` - Enhanced microclimate dataset with calculated parameters
- `soil_climate_integrated.csv` - Complete integrated dataset with all soil and climate variables
- `irrigation_ml_features.csv` - ML-ready dataset with selected features for irrigation modeling

### Visualization Files

- Monthly temperature, precipitation, and evapotranspiration plots
- Soil texture-specific irrigation parameters across seasons
- Correlation matrices of key soil-climate variables
- Irrigation requirement analyses by soil type and season

## Usage for ML Model Development

The `irrigation_ml_features.csv` file contains all the necessary features for developing ML models to predict:

1. Optimal irrigation scheduling (timing)
2. Precise irrigation amounts
3. Water-use efficiency optimization
4. Drought stress prediction
5. Yield potential based on soil-climate conditions

The data is structured with soil properties prefixed with `soil_`, climate variables prefixed with `climate_`, and calculated interaction metrics provided as direct columns.

## Methodology Notes

- Root vegetable crop coefficients are based on FAO-56 standards
- Irrigation parameters are calculated based on a 30 cm effective root zone
- Water requirements assume 50% management allowable depletion
- Soil texture classifications are based on USDA soil texture triangle
- Evapotranspiration calculations follow the standardized FAO Penman-Monteith method 