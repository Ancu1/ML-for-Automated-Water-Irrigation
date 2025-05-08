import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Get absolute path of the workspace directory
workspace_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(f"Workspace directory: {workspace_dir}")

# Create output directories with absolute paths
output_data_dir = os.path.join(workspace_dir, 'microclimate_analysis', 'integrated_data')
output_plots_dir = os.path.join(workspace_dir, 'microclimate_analysis', 'integrated_plots')

os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_plots_dir, exist_ok=True)

print(f"Output data directory: {output_data_dir}")
print(f"Output plots directory: {output_plots_dir}")

print("Loading and integrating soil and microclimate data...")

# Load the soil quality data with absolute path
soil_data_path = os.path.join(workspace_dir, 'SoilQuality_ML_Ready.csv')
soil_data = pd.read_csv(soil_data_path)
print(f"Loaded soil data from: {soil_data_path}")

# Load the processed microclimate data with absolute path
microclimate_data_path = os.path.join(workspace_dir, 'microclimate_analysis', 'data', 'processed_microclimate.csv')
microclimate_data = pd.read_csv(microclimate_data_path)
print(f"Loaded microclimate data from: {microclimate_data_path}")

# Convert datetime column in microclimate data
microclimate_data['datetime'] = pd.to_datetime(microclimate_data['datetime'])

# Aggregate microclimate data to monthly averages for integration with soil data
monthly_climate = microclimate_data.groupby(microclimate_data['datetime'].dt.month).agg({
    'tempC': 'mean',
    'humidity': 'mean',
    'precip': 'sum',
    'windspeed_ms': 'mean',
    'ET0': 'mean',
    'ETc': 'mean',
    'NIR': 'mean',
    'water_deficit': 'mean',
    'vpd': 'mean',
    'actual_vapor_pressure': 'mean',
    'aridity_index': 'mean'
}).reset_index()
monthly_climate.rename(columns={'datetime': 'month'}, inplace=True)

# Create a function to calculate soil-climate interaction metrics
def calculate_interaction_metrics(soil_row, climate_month_data):
    """
    Calculate interaction metrics between soil properties and climate variables
    for a specific month's climate data and soil sample
    """
    # Extract relevant soil properties
    # Using actual column names from SoilQuality_ML_Ready.csv
    sand = soil_row['2000-63 µm [%]']  # Sand content %
    clay = soil_row['2-0.063 µm [%]']  # Clay content %
    silt = soil_row['63-2 µm [%]']  # Silt content %
    bd = soil_row['Bulk Density [g/cm³]']  # Bulk density (g/cm³)
    om = soil_row['TOC [%]']  # Using TOC (Total Organic Carbon) as proxy for Organic Matter
    awc = soil_row['Available Water Capacity [m³/m³]']  # Available water capacity
    hc = soil_row['Hydraulic Conductivity [cm/hr]']  # Hydraulic conductivity (cm/hr)
    ir = soil_row['Infiltration Rate [cm/hr]']  # Infiltration rate (mm/hr)
    
    # Extract climate properties for this month
    temp = climate_month_data['tempC']  # Temperature (°C)
    humidity = climate_month_data['humidity']  # Relative humidity (%)
    precip = climate_month_data['precip']  # Monthly precipitation (mm)
    wind = climate_month_data['windspeed_ms']  # Wind speed (m/s)
    et0 = climate_month_data['ET0']  # Reference evapotranspiration (mm/day)
    vpd = climate_month_data['vpd']  # Vapor pressure deficit (kPa)
    
    # Calculate interaction metrics
    
    # Convert AWC from m³/m³ to percentage for calculations
    awc_pct = awc * 100
    
    # 1. Soil Moisture Stress Index
    # Higher values indicate more moisture stress (0-1 scale)
    if precip > 0:
        moisture_stress = np.clip(1 - (precip / (et0 * 30 * 1.5)), 0, 1) * (1 - awc)
    else:
        moisture_stress = 1 * (1 - awc)
    
    # 2. Infiltration Efficiency
    # How effectively rainfall can infiltrate the soil
    # Higher values mean better infiltration (0-1 scale)
    if precip > 0:
        max_intensity = precip / 4  # Simple estimation of maximum rainfall intensity
        infiltration_efficiency = np.clip(ir / max_intensity, 0, 1)
    else:
        infiltration_efficiency = 1  # No rain, so no infiltration issues
    
    # 3. Evaporation Susceptibility
    # Higher values mean more susceptible to evaporation (0-1 scale)
    evaporation_susceptibility = (sand/100) * vpd / 5  # Normalize by typical max VPD of 5 kPa
    evaporation_susceptibility = np.clip(evaporation_susceptibility, 0, 1)
    
    # 4. Drainage Factor
    # Higher values mean more water lost to drainage (0-1 scale)
    drainage_factor = np.clip(hc / 10, 0, 1)  # Normalize by a high HC value of 10 cm/hr
    
    # 5. Irrigation Efficiency Factor
    # How efficiently irrigation water can be used (0-1 scale, higher is better)
    irrigation_efficiency = awc * (1 - drainage_factor) * (1 - evaporation_susceptibility)
    irrigation_efficiency = np.clip(irrigation_efficiency, 0, 1)
    
    # 6. Root Zone Water Capacity (mm)
    # Estimate for a 30 cm root zone (convert m³/m³ to mm)
    root_zone_water_capacity = awc * 300  # 30 cm = 300 mm, and AWC is already in m³/m³
    
    # 7. Days to Depletion
    # Estimate days until soil water is depleted without rainfall
    if et0 > 0:
        days_to_depletion = root_zone_water_capacity / et0
    else:
        days_to_depletion = 30  # Arbitrary high value if ET0 is zero
    
    # 8. Irrigation Interval (days)
    # Recommended interval between irrigation events
    if et0 > 0:
        irrigation_interval = (root_zone_water_capacity * 0.5) / et0  # Allow 50% depletion
    else:
        irrigation_interval = 30  # Arbitrary high value if ET0 is zero
    
    # 9. Irrigation Amount (mm)
    # Recommended amount per irrigation event
    irrigation_amount = root_zone_water_capacity * 0.5  # Refill 50% depletion
    
    return {
        'moisture_stress': moisture_stress,
        'infiltration_efficiency': infiltration_efficiency,
        'evaporation_susceptibility': evaporation_susceptibility,
        'drainage_factor': drainage_factor,
        'irrigation_efficiency': irrigation_efficiency,
        'root_zone_water_capacity': root_zone_water_capacity,
        'days_to_depletion': days_to_depletion,
        'irrigation_interval': irrigation_interval,
        'irrigation_amount': irrigation_amount
    }

# Create a comprehensive dataset with monthly soil-climate interactions
print("Calculating soil-climate interactions...")

# Initialize list to store results
integrated_data = []

# For each soil sample, calculate interaction metrics for each month
for _, soil_row in soil_data.iterrows():
    sample_id = soil_row['Lab label']  # Use 'Lab label' instead of 'SampleID'
    
    for _, climate_row in monthly_climate.iterrows():
        month = climate_row['month']
        
        # Create a dictionary with soil properties
        row_dict = {
            'SampleID': sample_id,  # Keep the key as 'SampleID' for consistency in output
            'Month': month
        }
        
        # Add all soil properties
        for col in soil_data.columns:
            if col != 'Lab label':  # Skip the ID column
                row_dict[f'soil_{col}'] = soil_row[col]
        
        # Add all climate properties
        for col in monthly_climate.columns:
            if col != 'month':
                row_dict[f'climate_{col}'] = climate_row[col]
        
        # Calculate and add interaction metrics
        interaction_metrics = calculate_interaction_metrics(soil_row, climate_row)
        for metric, value in interaction_metrics.items():
            row_dict[metric] = value
        
        integrated_data.append(row_dict)

# Convert to DataFrame
integrated_df = pd.DataFrame(integrated_data)

# Save the integrated dataset
print("Saving integrated dataset...")
integrated_data_file = os.path.join(output_data_dir, 'soil_climate_integrated.csv')
integrated_df.to_csv(integrated_data_file, index=False)
print(f"Saved integrated dataset to: {integrated_data_file}")

# Create visualization subsets for key variables
print("Creating visualizations of integrated data...")

# Plot key parameters across months by soil texture
# Calculate average metrics by soil texture and month
soil_data['texture_class'] = soil_data.apply(
    lambda row: 'Sandy' if row['2000-63 µm [%]'] > 0.6 else 'Clayey' if row['2-0.063 µm [%]'] > 0.3 else 'Loamy', 
    axis=1
)

# Add texture class to integrated dataset
texture_mapping = dict(zip(soil_data['Lab label'], soil_data['texture_class']))  # Use 'Lab label' here
integrated_df['texture_class'] = integrated_df['SampleID'].map(texture_mapping)

# Get average metrics by texture class and month
texture_month_avg = integrated_df.groupby(['texture_class', 'Month']).agg({
    'moisture_stress': 'mean',
    'irrigation_efficiency': 'mean',
    'days_to_depletion': 'mean',
    'irrigation_interval': 'mean',
    'irrigation_amount': 'mean'
}).reset_index()

# Plot irrigation efficiency by soil texture across months
plt.figure(figsize=(12, 8))
for texture in texture_month_avg['texture_class'].unique():
    subset = texture_month_avg[texture_month_avg['texture_class'] == texture]
    plt.plot(subset['Month'], subset['irrigation_efficiency'], 'o-', label=texture)

plt.title('Irrigation Efficiency by Soil Texture Across Months')
plt.xlabel('Month')
plt.ylabel('Irrigation Efficiency (0-1)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(range(1, 13))
plot_file = os.path.join(output_plots_dir, 'irrigation_efficiency_by_texture.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to: {plot_file}")

# Plot moisture stress by soil texture across months
plt.figure(figsize=(12, 8))
for texture in texture_month_avg['texture_class'].unique():
    subset = texture_month_avg[texture_month_avg['texture_class'] == texture]
    plt.plot(subset['Month'], subset['moisture_stress'], 'o-', label=texture)

plt.title('Moisture Stress by Soil Texture Across Months')
plt.xlabel('Month')
plt.ylabel('Moisture Stress Index (0-1)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(range(1, 13))
plot_file = os.path.join(output_plots_dir, 'moisture_stress_by_texture.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to: {plot_file}")

# Plot recommended irrigation interval across months by soil texture
plt.figure(figsize=(12, 8))
for texture in texture_month_avg['texture_class'].unique():
    subset = texture_month_avg[texture_month_avg['texture_class'] == texture]
    plt.plot(subset['Month'], subset['irrigation_interval'], 'o-', label=texture)

plt.title('Recommended Irrigation Interval by Soil Texture Across Months')
plt.xlabel('Month')
plt.ylabel('Irrigation Interval (days)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(range(1, 13))
plot_file = os.path.join(output_plots_dir, 'irrigation_interval_by_texture.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to: {plot_file}")

# Plot irrigation amount needed by soil texture
plt.figure(figsize=(12, 8))
for texture in texture_month_avg['texture_class'].unique():
    subset = texture_month_avg[texture_month_avg['texture_class'] == texture]
    plt.plot(subset['Month'], subset['irrigation_amount'], 'o-', label=texture)

plt.title('Recommended Irrigation Amount by Soil Texture Across Months')
plt.xlabel('Month')
plt.ylabel('Irrigation Amount (mm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(range(1, 13))
plot_file = os.path.join(output_plots_dir, 'irrigation_amount_by_texture.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to: {plot_file}")

# Create a heatmap of correlations between key integrated variables
key_columns = [
    'soil_2000-63 µm [%]', 'soil_2-0.063 µm [%]', 'soil_TOC [%]', 'soil_Available Water Capacity [m³/m³]',
    'soil_Hydraulic Conductivity [cm/hr]', 'soil_Infiltration Rate [cm/hr]', 'climate_tempC', 
    'climate_humidity', 'climate_precip', 'climate_ET0', 'climate_NIR',
    'moisture_stress', 'irrigation_efficiency', 'days_to_depletion', 'irrigation_interval'
]

correlation_df = integrated_df[key_columns].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Integrated Soil-Climate Variables')
plt.tight_layout()
plot_file = os.path.join(output_plots_dir, 'integrated_correlation_matrix.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved correlation matrix to: {plot_file}")

# Create a subset for ML model training
print("Creating ML-ready integrated dataset...")

# Select relevant features for irrigation ML model
ml_features = [
    'Month', 
    'soil_2000-63 µm [%]', 'soil_63-2 µm [%]', 'soil_2-0.063 µm [%]', 'soil_TOC [%]', 
    'soil_pH', 'soil_CEC [cmol/kg]', 'soil_Available Water Capacity [m³/m³]',
    'soil_Hydraulic Conductivity [cm/hr]', 'soil_Infiltration Rate [cm/hr]',
    'climate_tempC', 'climate_humidity', 'climate_precip', 'climate_ET0',
    'climate_vpd', 'climate_NIR', 'climate_water_deficit',
    'moisture_stress', 'infiltration_efficiency', 'evaporation_susceptibility',
    'drainage_factor', 'irrigation_efficiency', 'root_zone_water_capacity',
    'days_to_depletion', 'irrigation_interval', 'irrigation_amount'
]

# Create the ML-ready dataset
ml_ready_df = integrated_df[ml_features]

# Save the ML-ready dataset
ml_file = os.path.join(output_data_dir, 'irrigation_ml_features.csv')
ml_ready_df.to_csv(ml_file, index=False)
print(f"Saved ML-ready dataset to: {ml_file}")

print("Integration complete! Results saved to microclimate_analysis/integrated_data/ directory.")
print("Visualization plots saved to microclimate_analysis/integrated_plots/ directory.") 