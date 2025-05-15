import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Create output directory for plots
os.makedirs('microclimate_analysis/plots', exist_ok=True)
os.makedirs('microclimate_analysis/data', exist_ok=True)

print("Reading microclimate data...")
# Load microclimate data
microclimate_df = pd.read_csv('MicroclimateDataSubset.csv')

# Convert datetime string to datetime object
microclimate_df['datetime'] = pd.to_datetime(microclimate_df['datetime'])
microclimate_df['month'] = microclimate_df['datetime'].dt.month
microclimate_df['day'] = microclimate_df['datetime'].dt.day
microclimate_df['year'] = microclimate_df['datetime'].dt.year

# Calculate mean temperature in Celsius
microclimate_df['tempC'] = (microclimate_df['temp'] - 32) * 5/9
microclimate_df['tempmaxC'] = (microclimate_df['tempmax'] - 32) * 5/9
microclimate_df['tempminC'] = (microclimate_df['tempmin'] - 32) * 5/9
microclimate_df['tempmeanC'] = (microclimate_df['tempmaxC']+microclimate_df['tempminC'])/2
microclimate_df['dewC'] = (microclimate_df['dew'] - 32) * 5/9

#convert precip from inches to mm
microclimate_df['precip']=microclimate_df['precip']*25.4

# Calculate vapor pressure (kPa) based on dew point (Allen et al., 1998)
microclimate_df['actual_vapor_pressure'] = 0.6108 * np.exp((17.27 * microclimate_df['dewC']) / (microclimate_df['dewC'] + 237.3))

# Calculate saturation vapor pressure (kPa) based on temperature
microclimate_df['es_max'] = 0.6108 * np.exp((17.27 * microclimate_df['tempmaxC']) / (microclimate_df['tempmaxC'] + 237.3))
microclimate_df['es_min'] = 0.6108 * np.exp((17.27 * microclimate_df['tempminC']) / (microclimate_df['tempminC'] + 237.3))
microclimate_df['es'] = (microclimate_df['es_max'] + microclimate_df['es_min']) / 2

# Calculate vapor pressure deficit (kPa)
microclimate_df['vpd'] = microclimate_df['es'] - microclimate_df['actual_vapor_pressure']

# Calculate solar radiation from given values in MJ/m²/day
solar_radiation_MJ = microclimate_df['solarenergy']

# Convert wind speed from mph to m/s
microclimate_df['wind_2m'] = ((microclimate_df['windspeed']/3.6)*4.87)/np.log(67.8*6-5.42)+(microclimate_df['windspeed'] * 0.44704)

# ---------------------------------------
# Calculate reference evapotranspiration
# ---------------------------------------

def calculate_ET0_FAO56(df):
    """
    Calculate reference evapotranspiration (ET0) using the FAO Penman-Monteith equation
    This is the FAO-56 version standardized for a hypothetical grass reference crop
    """
    # Constants
    albedo = 0.23  # Albedo for grass reference crop
    G = 0  # Soil heat flux density (MJ/m²/day), assumed negligible for daily calculations
    sigma = 4.903e-9  # Stefan-Boltzmann constant (MJ/K⁴/m²/day)
    
    # Extract necessary variables
    Tmean = df['tempC']  # Mean temperature (°C)
    u2 = df['wind_2m']  # Wind speed at 2m height (m/s)
    es = df['es']  # Saturation vapor pressure (kPa)
    ea = df['actual_vapor_pressure']  # Actual vapor pressure (kPa)
    Rs = df['solarenergy']  # Solar radiation (MJ/m²/day)
    sealevelpressure = df['sealevelpressure']


    # Calculate psychrometric constant (kPa/°C)
    # Assume atmospheric pressure based on elevation (approximate for northwestern Iraq - 300m)
    gamma = (0.665*10**(-3))*(sealevelpressure/10)

    # Calculate slope of saturation vapor pressure curve (kPa/°C)
    delta = (4098 * (0.6108 * np.exp((17.27 * Tmean) / (Tmean + 237.3)))) / ((Tmean + 237.3) ** 2)
    
    # Calculate net radiation
    # Estimate clear-sky solar radiation (Rs0)
    Ra = 30  # Approximate extraterrestrial radiation for the region (MJ/m²/day)
    Rs0 = (0.75 + 2e-5 * 300) * Ra  # Rs0 = (0.75 + 2×10⁻⁵z) × Ra
    
    # Calculate net shortwave radiation (MJ/m²/day)
    Rns = (1 - albedo) * Rs
    
    # Calculate clear-sky emissivity
    ratio = np.minimum(Rs / Rs0, 1.0)
    fcd = 1.35 * ratio - 0.35
    
    # Actual vapor pressure influence on net longwave radiation
    ea_term = 0.34 - 0.14 * np.sqrt(ea)
    
    # Temperature in Kelvin for net longwave radiation
    Tmax_K = df['tempmaxC'] + 273.16
    Tmin_K = df['tempminC'] + 273.16
    
    # Calculate net longwave radiation (MJ/m²/day)
    Rnl = sigma * fcd * ea_term * ((Tmax_K**4 + Tmin_K**4) / 2)
    
    # Calculate net radiation (MJ/m²/day)
    Rn = Rns - Rnl
    
    # Calculate denominator and numerator terms for ET0
    temp_term = 900 / (Tmean + 273) * u2
    
    # Calculate reference evapotranspiration (mm/day)
    numerator = 0.408 * delta * (Rn - G) + gamma * temp_term * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    ET0 = numerator / denominator
    
    return ET0

# Calculate reference evapotranspiration (ET0)
microclimate_df['ET0'] = calculate_ET0_FAO56(microclimate_df)

# ----------------------------------
# Calculate crop water requirements
# ----------------------------------

# Define typical Kc values for root vegetables (FAO-56 values)
# Initial, mid, and late season crop coefficients
kc_init = 0.5   # Initial stage
kc_mid = 1.05   # Mid-season stage
kc_end = 0.9    # Late season stage

# Create a function to calculate Kc based on day of year
def calculate_Kc(row):
    month = row['month']
    day = row['day']
    
    # Simple determination for northwestern Iraq with typical root vegetable growing season
    # Assume planting in October (month 10) and harvesting in March-April (month 3-4)
    
    if month in [10, 11]:  # Initial stage (Oct-Nov)
        return kc_init
    elif month in [12, 1]:  # Mid-season (Dec-Jan)
        return kc_mid
    elif month in [2, 3, 4]:  # Late season (Feb-Apr)
        return kc_end
    else:  # Not main growing season
        return 0  # No irrigation during off-season

# Apply the function to calculate Kc
microclimate_df['Kc'] = microclimate_df.apply(calculate_Kc, axis=1)

# Calculate crop evapotranspiration (ETc = Kc × ET0)
microclimate_df['ETc'] = microclimate_df['Kc'] * microclimate_df['ET0']

# --------------------------------
# Calculate irrigation parameters
# --------------------------------

# Calculate effective rainfall (simplified approach)
def effective_rainfall(precip):
    """
    Simplified USDA Soil Conservation Service formula
    Effective rainfall is the amount that's actually available to the crop
    """
    if precip <= 0:
        return 0
    elif precip <= 25:
        return precip * (125 - 0.2 * precip) / 125
    else:
        return 125/3 + 0.1 * precip
    
# Apply effective rainfall calculation
microclimate_df['effective_precip'] = microclimate_df['precip'].apply(effective_rainfall)

# Calculate net irrigation requirement (NIR = ETc - effective rainfall)
microclimate_df['NIR'] = np.maximum(0, microclimate_df['ETc'] - microclimate_df['effective_precip'])

# Calculate reference soil moisture depletion
# Assume a representative available water capacity (AWC) for the region
# This is a simplified approach - in a real model, this would be based on actual soil data
AWC = 160  # mm/m (representative for loamy soils)
root_depth = 0.4  # m (typical for root vegetables)
management_allowable_depletion = 0.5  # Standard MAD for most crops
total_available_water = AWC * root_depth  # mm
readily_available_water = total_available_water * management_allowable_depletion  # mm

microclimate_df['TAW'] = total_available_water
microclimate_df['RAW'] = readily_available_water

# -----------------------------
# Calculate drought indicators
# -----------------------------

# Aridity Index: Ratio of precipitation to potential evapotranspiration
microclimate_df['aridity_index'] = microclimate_df['precip'] / microclimate_df['ET0']

# Water Deficit: Difference between potential and actual evapotranspiration
# For a simplified model, assume actual ET is limited by water availability
microclimate_df['water_deficit'] = np.maximum(0, microclimate_df['ETc'] - microclimate_df['precip'])

# ----------------------------
# Save processed data to file
# ----------------------------

print("Saving processed microclimate data...")
microclimate_df.to_csv('microclimate_analysis/data/processed_microclimate.csv', index=False)

# --------------------------
# Create visualizations
# --------------------------

print("Creating visualizations...")

# Plot monthly averages of key parameters

# 1. Get mean of all variables (except precip) for each month across all years
monthly_means = microclimate_df.groupby('month').agg({
    'tempmeanC': 'mean',
    'humidity': 'mean',
    'ET0': 'mean',
    'ETc': 'mean',
    'NIR': 'mean',
    'water_deficit': 'mean'
})

monthly_data = microclimate_df.groupby('month').agg({
    'tempmeanC': 'mean',
    'humidity': 'mean',
    'precip': 'sum',
    'ET0': 'mean',
    'ETc': 'mean',
    'NIR': 'mean',
    'water_deficit': 'mean'
}).reset_index()

# 2. For precip, first sum by year and month, then average across years for each month
monthly_precip = (microclimate_df.groupby(['year', 'month'])['precip'].sum().groupby('month').mean().to_frame('precip'))
print(monthly_precip.head())

# 3. Combine into one DataFrame
monthly_data = monthly_means.join(monthly_precip).reset_index()

# Plot monthly temperature
plt.figure(figsize=(10, 6))
plt.plot(monthly_data['month'], monthly_data['tempC'], 'o-', color='red')
plt.title('Average Monthly Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 13))
plt.savefig('microclimate_analysis/plots/monthly_temperature.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot monthly ET0 and precipitation
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.bar(monthly_data['month'], monthly_data['precip'], color='blue', alpha=0.5, label='Precipitation')
ax2.plot(monthly_data['month'], monthly_data['ET0'], 'o-', color='red', label='Reference ET')
ax1.set_xlabel('Month')
ax1.set_ylabel('Precipitation (mm)', color='blue')
ax2.set_ylabel('Reference ET (mm/day)', color='red')
ax1.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', colors='red')
plt.title('Monthly Precipitation vs Reference Evapotranspiration')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
plt.savefig('microclimate_analysis/plots/precip_vs_et0.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot irrigation requirement
plt.figure(figsize=(10, 6))
plt.bar(monthly_data['month'], monthly_data['NIR'], color='green', alpha=0.7)
plt.title('Monthly Net Irrigation Requirement')
plt.xlabel('Month')
plt.ylabel('NIR (mm/day)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 13))
plt.savefig('microclimate_analysis/plots/monthly_nir.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot water deficit
plt.figure(figsize=(10, 6))
plt.bar(monthly_data['month'], monthly_data['water_deficit'], color='brown', alpha=0.7)
plt.title('Monthly Water Deficit')
plt.xlabel('Month')
plt.ylabel('Water Deficit (mm/day)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 13))
plt.savefig('microclimate_analysis/plots/monthly_water_deficit.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot relationship between temperature and ET0
plt.figure(figsize=(10, 6))
plt.scatter(microclimate_df['tempC'], microclimate_df['ET0'], alpha=0.5)
plt.title('Relationship Between Temperature and Reference ET')
plt.xlabel('Temperature (°C)')
plt.ylabel('Reference ET (mm/day)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('microclimate_analysis/plots/temp_vs_et0.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot correlation heatmap of key variables
corr_columns = ['tempC', 'humidity', 'precip', 'windspeed_ms', 'vpd', 'ET0', 'ETc', 'NIR', 'water_deficit']
corr_matrix = microclimate_df[corr_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Irrigation Parameters')
plt.tight_layout()
plt.savefig('microclimate_analysis/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete! Results saved to microclimate_analysis/ directory.") 
