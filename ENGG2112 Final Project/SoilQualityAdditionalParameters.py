import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    # Get current working directory for file paths
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Load the original dataset
    print("Loading original dataset...")
    try:
        input_file = os.path.join(current_dir, 'SoilQualityDataset.csv')
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    
    # Create new dataframe with same structure for additional parameters
    df_add = pd.DataFrame()
    
    # Keep track of original columns
    original_columns = list(df.columns)
    
    # Copy the original columns to maintain structure
    for col in original_columns:
        df_add[col] = df[col]
    
    # Calculate Bulk Density (BD) in g/cm³
    # Formula: BD = (1.55 - 0.0046 × %clay - 0.263 × %organic_carbon)
    # Reference: Saxton & Rawls (2006)
    print("Calculating Bulk Density...")
    # Use the clay content (2-0.063 µm [%]) and TOC (%)
    clay_percent = df['2-0.063 µm [%]']
    organic_carbon = df['TOC [%]']
    
    # Replace any NaN values with reasonable defaults
    clay_percent = clay_percent.fillna(df['2-0.063 µm [%]'].mean())
    organic_carbon = organic_carbon.fillna(df['TOC [%]'].mean())
    
    df_add['Bulk Density [g/cm³]'] = 1.55 - (0.0046 * clay_percent) - (0.263 * organic_carbon)
    # Ensure values are in reasonable range (1.0 to 1.8 g/cm³)
    df_add['Bulk Density [g/cm³]'] = df_add['Bulk Density [g/cm³]'].clip(1.0, 1.8)
    
    # Calculate Field Capacity (FC) and Wilting Point (WP)
    # FC = 0.2576 - 0.002×%sand + 0.0036×%clay + 0.0299×%OM
    # WP = 0.026 + 0.005×%clay + 0.0158×%OM
    # Reference: Rawls et al. (1982)
    print("Calculating Field Capacity and Wilting Point...")
    
    sand_percent = df['2000-63 µm [%]']  # Sand fraction
    # Convert TOC to OM (organic matter) using typical conversion factor of 1.724
    organic_matter = organic_carbon * 1.724
    
    sand_percent = sand_percent.fillna(df['2000-63 µm [%]'].mean())
    
    # Calculate as volumetric water content (m³/m³)
    df_add['Field Capacity [m³/m³]'] = 0.2576 - (0.002 * sand_percent) + (0.0036 * clay_percent) + (0.0299 * organic_matter)
    df_add['Wilting Point [m³/m³]'] = 0.026 + (0.005 * clay_percent) + (0.0158 * organic_matter)
    
    # Ensure values are in reasonable range (0.1 to 0.5 for FC, 0.01 to 0.25 for WP)
    df_add['Field Capacity [m³/m³]'] = df_add['Field Capacity [m³/m³]'].clip(0.1, 0.5)
    df_add['Wilting Point [m³/m³]'] = df_add['Wilting Point [m³/m³]'].clip(0.01, 0.25)
    
    # Calculate Available Water Capacity (AWC)
    # Formula: θAWC = θFC - θPWP
    # Reference: Gupta & Larson (1979)
    print("Calculating Available Water Capacity...")
    df_add['Available Water Capacity [m³/m³]'] = df_add['Field Capacity [m³/m³]'] - df_add['Wilting Point [m³/m³]']
    
    # Calculate Hydraulic Conductivity
    # Ks = 2.54 × 10^(−0.6 + 0.012×%sand − 0.0064×%clay) in cm/hr
    # Reference: Cosby et al. (1984)
    print("Calculating Hydraulic Conductivity...")
    df_add['Hydraulic Conductivity [cm/hr]'] = 2.54 * (10 ** (-0.6 + (0.012 * sand_percent) - (0.0064 * clay_percent)))
    # Clip to reasonable range
    df_add['Hydraulic Conductivity [cm/hr]'] = df_add['Hydraulic Conductivity [cm/hr]'].clip(0.01, 100)
    
    # Calculate Cation Exchange Capacity (CEC)
    # CEC = (0.1 × %clay × (pH - 4.5)) + (4.5 × %organic_matter)
    # Reference: Brady & Weil (2008)
    print("Calculating Cation Exchange Capacity...")
    ph_values = df['pH']
    ph_values = ph_values.fillna(df['pH'].mean())
    
    df_add['CEC [cmol/kg]'] = (0.1 * clay_percent * (ph_values - 4.5)) + (4.5 * organic_matter)
    df_add['CEC [cmol/kg]'] = df_add['CEC [cmol/kg]'].clip(0, 50)
    
    # Calculate Infiltration Rate based on texture and bulk density
    # This is a simplified approach based on USDA-NRCS tables
    # Reference: Rawls et al. (1993)
    print("Estimating Infiltration Rate...")
    # Simplified calculation based on sand and clay percentages
    df_add['Infiltration Rate [cm/hr]'] = (0.5 * sand_percent / clay_percent).clip(0.1, 25)
    
    # Calculate Salinity Class based on EC
    # Using standard USDA classification
    print("Determining Salinity Class...")
    ec_values = df['EC [µS/cm]']
    # Convert µS/cm to dS/m
    ec_ds_m = ec_values / 1000
    
    # Create salinity class column
    salinity_class = []
    for ec in ec_ds_m:
        if pd.isna(ec):
            salinity_class.append(np.nan)
        elif ec < 2:
            salinity_class.append("Non-saline")
        elif ec < 4:
            salinity_class.append("Slightly saline")
        elif ec < 8:
            salinity_class.append("Moderately saline")
        elif ec < 16:
            salinity_class.append("Strongly saline")
        else:
            salinity_class.append("Very strongly saline")
    
    df_add['Salinity Class'] = salinity_class
    
    # Define all the new parameters
    new_params = [
        'Bulk Density [g/cm³]',
        'Field Capacity [m³/m³]',
        'Wilting Point [m³/m³]',
        'Available Water Capacity [m³/m³]',
        'Hydraulic Conductivity [cm/hr]',
        'CEC [cmol/kg]',
        'Infiltration Rate [cm/hr]',
        'Salinity Class'
    ]
    
    # Verify that all calculated parameters are in the dataframe
    for param in new_params:
        if param not in df_add.columns:
            print(f"WARNING: {param} is missing from the dataframe!")
    
    # Save the new dataset with full path
    print("Saving new dataset...")
    output_file = os.path.join(current_dir, 'SoilQualityAdditions.csv')
    df_add.to_csv(output_file, index=False)
    
    # Verify file was created
    if os.path.exists(output_file):
        print(f"Complete! File saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"ERROR: Failed to create file at {output_file}")
    
    # Print column names in the final CSV to confirm
    print("\nColumns in the final CSV file:")
    print(", ".join(df_add.columns))
    
    # Print sample data for new parameters
    print("\nSample data for new parameters (first 3 rows):")
    sample_data = df_add[new_params].head(3)
    print(sample_data)
    
    # Print summary of new parameters
    print("\nSummary of calculated parameters:")
    for col in new_params:
        if col in df_add.columns:  # Make sure column exists
            # Check if the column contains numeric data
            if pd.api.types.is_numeric_dtype(df_add[col]):
                print(f"{col}: Mean = {df_add[col].mean():.4f}, Min = {df_add[col].min():.4f}, Max = {df_add[col].max():.4f}")
            else:
                # For non-numeric columns (like Salinity Class), show value counts
                value_counts = df_add[col].value_counts(dropna=False)
                print(f"{col} distribution:")
                for value, count in value_counts.items():
                    value_str = 'NaN' if pd.isna(value) else value
                    print(f"  {value_str}: {count} samples")
        else:
            print(f"Column {col} not found in the dataset")
    
    # Also save a version with only the new parameters for easier viewing
    new_only_file = os.path.join(current_dir, 'SoilQualityNewParametersOnly.csv')
    df_new_only = df_add[new_params].copy()
    df_new_only.to_csv(new_only_file, index=False)
    
    # Verify second file was created
    if os.path.exists(new_only_file):
        print(f"\nAlso saved new parameters only to: {new_only_file}")
        print(f"File size: {os.path.getsize(new_only_file)} bytes")
    else:
        print(f"\nERROR: Failed to create new parameters file at {new_only_file}")

if __name__ == "__main__":
    main() 