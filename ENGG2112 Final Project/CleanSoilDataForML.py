import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import os

def clean_data_for_ml():
    print("Starting soil quality data cleanup for ML...")
    
    # Load the dataset with all parameters
    try:
        df = pd.read_csv('SoilQualityAdditions.csv')
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Make a copy to preserve the original
    df_clean = df.copy()
    
    # ====== STEP 1: Initial Exploration ======
    print("\n=== Data Exploration ===")
    
    # Check for missing values
    missing_values = df_clean.isnull().sum()
    print(f"Columns with missing values:\n{missing_values[missing_values > 0]}")
    
    # Check data types
    print("\nData types:")
    print(df_clean.dtypes.value_counts())
    
    # ====== STEP 2: Type Conversion ======
    print("\n=== Converting Data Types ===")
    
    # Convert any object columns that should be numeric
    numeric_columns = []
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Skip date columns and categorical columns
            if 'Date' in col or 'Class' in col or 'Event' in col or 'label' in col:
                continue
            
            # Try to convert to numeric
            try:
                # Replace common non-numeric values
                df_clean[col] = df_clean[col].replace('<0.05', '0.025')  # Replace less than values with half
                df_clean[col] = df_clean[col].replace('<0.03', '0.015')
                
                # Convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col])
                numeric_columns.append(col)
                print(f"Converted {col} to numeric")
            except:
                print(f"Could not convert {col} to numeric")
                
    # ====== STEP 3: Handle Missing Values ======
    print("\n=== Handling Missing Values ===")
    
    # Identify columns with significant missing data (>30%)
    missing_percent = df_clean.isnull().mean() * 100
    high_missing = missing_percent[missing_percent > 30].index.tolist()
    
    if high_missing:
        print(f"Columns with >30% missing values: {high_missing}")
        # Consider dropping these columns
        df_clean = df_clean.drop(columns=high_missing)
        print(f"Dropped {len(high_missing)} columns with high missing values")
    
    # For remaining numerical columns with missing values, impute with median
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Impute numeric columns
    if numeric_cols.any():
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        print(f"Imputed missing values in {len(numeric_cols)} numeric columns using median")
    
    # For categorical columns, impute with most frequent value
    if categorical_cols.any():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
        print(f"Imputed missing values in {len(categorical_cols)} categorical columns using most frequent value")
    
    # ====== STEP 4: Handle Outliers ======
    print("\n=== Handling Outliers ===")
    
    # Identify and cap outliers in numerical columns (using 1.5 * IQR method)
    for col in numeric_cols:
        if col in df_clean.columns:  # Check if column wasn't dropped
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers > 0:
                print(f"Found {outliers} outliers in {col}")
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in {col}")
    
    # ====== STEP 5: Feature Encoding ======
    print("\n=== Encoding Categorical Features ===")
    
    # Identify categorical columns that need encoding (exclude date columns and ID-like columns)
    cat_cols_to_encode = [col for col in categorical_cols 
                          if 'Date' not in col and 'Event' not in col and 'label' not in col]
    
    if cat_cols_to_encode:
        print(f"Encoding categorical columns: {cat_cols_to_encode}")
        
        # Use pandas get_dummies for one-hot encoding (simpler than sklearn's OneHotEncoder for this case)
        df_encoded = pd.get_dummies(df_clean, columns=cat_cols_to_encode, drop_first=True)
        print(f"Encoded {len(cat_cols_to_encode)} categorical columns")
        
        # Update dataframe
        df_clean = df_encoded
    
    # ====== STEP 6: Feature Scaling ======
    print("\n=== Scaling Features ===")
    
    # Create a separate dataframe for scaling (don't scale ID columns or target variables)
    # Assuming all numeric columns except ID-like columns and date columns should be scaled
    cols_to_scale = [col for col in df_clean.select_dtypes(include=['float64', 'int64']).columns
                     if 'label' not in col and 'Event' not in col]
    
    if cols_to_scale:
        # Store original values before scaling (for reference)
        df_original = df_clean[cols_to_scale].copy()
        
        # Apply standard scaling
        scaler = StandardScaler()
        df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
        print(f"Scaled {len(cols_to_scale)} numeric features")
        
        # Save the scaler parameters for later use in predictions
        scaling_params = pd.DataFrame({
            'feature': cols_to_scale,
            'mean': scaler.mean_,
            'scale': scaler.scale_
        })
        scaling_params.to_csv('scaling_parameters.csv', index=False)
        print("Saved scaling parameters for future use")
    
    # ====== STEP 7: Feature Selection ======
    print("\n=== Feature Selection ===")
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)  # Features with variance < 0.01 will be removed
    
    # Get feature names that would be selected (without actually transforming)
    feature_mask = selector.fit(df_clean[cols_to_scale]).get_support()
    selected_features = [cols_to_scale[i] for i in range(len(cols_to_scale)) if feature_mask[i]]
    
    # Report on low variance features
    low_var_features = [cols_to_scale[i] for i in range(len(cols_to_scale)) if not feature_mask[i]]
    if low_var_features:
        print(f"Low variance features that could be removed: {low_var_features}")
    
    # Keep all features for now, but flag the low variance ones
    print(f"Retained {len(selected_features)} out of {len(cols_to_scale)} features after variance checking")
    
    # ====== STEP 8: Save Clean Dataset ======
    print("\n=== Saving Clean Dataset ===")
    
    # Save the cleaned dataset
    df_clean.to_csv('SoilQuality_ML_Ready.csv', index=False)
    print(f"Saved cleaned dataset with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
    
    # Create a simple HTML report with key findings
    report_content = f"""
    <html>
    <head>
        <title>Soil Quality Data Cleaning Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Soil Quality Dataset Cleaning Report</h1>
        
        <h2>Dataset Overview</h2>
        <p>Original dimensions: {df.shape[0]} rows × {df.shape[1]} columns</p>
        <p>Cleaned dimensions: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns</p>
        
        <h2>Cleaning Steps Performed</h2>
        <ul>
            <li>Converted string values to numeric where possible</li>
            <li>Handled missing values through median/mode imputation</li>
            <li>Capped outliers using the 1.5 × IQR method</li>
            <li>Encoded categorical variables</li>
            <li>Standardized numerical features</li>
            <li>Identified low variance features</li>
        </ul>
        
        <h2>Feature Engineering Recommendations</h2>
        <p>Consider creating interaction terms between soil texture components and water-related parameters.</p>
        <p>For root vegetable irrigation, pay special attention to Available Water Capacity, which directly impacts irrigation needs.</p>
        
        <h2>ML Model Recommendations</h2>
        <p>Consider using ensemble methods like Random Forest or Gradient Boosting as they often perform well on soil-related data.</p>
        <p>For irrigation prediction, regression models would be appropriate.</p>
    </body>
    </html>
    """
    
    with open('data_cleaning_report.html', 'w') as f:
        f.write(report_content)
    print("Created data cleaning report")
    
    return df_clean

if __name__ == "__main__":
    df_clean = clean_data_for_ml()
    print("\nData cleaning complete!") 