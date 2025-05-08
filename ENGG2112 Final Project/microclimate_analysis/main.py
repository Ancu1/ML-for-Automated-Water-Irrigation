#!/usr/bin/env python3
import os
import subprocess
import time
import sys

def run_script(script_name, description):
    """Run a Python script and display its progress"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_name], 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        
        # Print the output
        print(result.stdout)
        
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print(result.stderr)
            
        end_time = time.time()
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run {script_name}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main function to run the entire data processing pipeline"""
    print("\n" + "="*40)
    print("SOIL-MICROCLIMATE INTEGRATION PIPELINE")
    print("For Automated Irrigation ML Model")
    print("="*40 + "\n")
    
    # Ensure necessary directories exist
    os.makedirs('microclimate_analysis/data', exist_ok=True)
    os.makedirs('microclimate_analysis/plots', exist_ok=True)
    os.makedirs('microclimate_analysis/integrated_data', exist_ok=True)
    os.makedirs('microclimate_analysis/integrated_plots', exist_ok=True)
    
    # Check that required input files exist
    required_files = [
        'MicroclimateDataSubset.csv',
        'SoilQuality_ML_Ready.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: The following required input files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure these files are present before running the pipeline.")
        return False
    
    # Run the microclimate data processing script
    if not run_script("microclimate_analysis/CalculateMicroclimateParameters.py", 
                     "Processing Microclimate Data"):
        return False
    
    # Run the integration script
    if not run_script("microclimate_analysis/IntegrateSoilAndMicroclimate.py",
                     "Integrating Soil and Microclimate Data"):
        return False
    
    print("\n" + "="*40)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*40)
    
    # List the output files
    print("\nGenerated files:")
    
    output_files = [
        "microclimate_analysis/data/processed_microclimate.csv",
        "microclimate_analysis/integrated_data/soil_climate_integrated.csv",
        "microclimate_analysis/integrated_data/irrigation_ml_features.csv"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # Size in KB
            print(f"  - {file} ({size:.1f} KB)")
    
    # List visualization files
    print("\nGenerated visualizations:")
    
    viz_dirs = [
        "microclimate_analysis/plots",
        "microclimate_analysis/integrated_plots"
    ]
    
    for directory in viz_dirs:
        if os.path.exists(directory):
            print(f"\nIn {directory}:")
            for viz_file in os.listdir(directory):
                if viz_file.endswith('.png'):
                    print(f"  - {viz_file}")
    
    print("\nThe prepared dataset for ML model development is:")
    print("  microclimate_analysis/integrated_data/irrigation_ml_features.csv")
    
    print("\nRefer to microclimate_analysis/README.md for detailed documentation.")
    return True

if __name__ == "__main__":
    main() 