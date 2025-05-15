## Drainage Prediction Model

This model determines whether drainage is needed based on environmental and soil conditions, so that irrigation is not applied when the field is already too saturated.

### 1. Dataset
- **Source:** [IoT Agriculture 2024 dataset](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024)
- **Focus location:** Tikrit, Iraq (Lat: 34.5716°, Lon: 43.6866°)

### 2. Preprocessing
- Converted `date` to datetime format and extracted time features
- Computed `nutrient_score = (N + P + K) / 765`
- Integrated soil properties from [SoilGrids.org](https://soilgrids.org):
  - `clay_pct = 34.6%`, `sand_pct = 17.9%`, `silt_pct = 47.5%`
  - `bulk_density_gcm3 = 1.43`, `ph_water = 7.8`

### 3. Model
- Trained a **Random Forest classifier**
- Input features:
  - `water_level`, `tempreature`, `humidity`, `nutrient_score`
  - `clay_pct`, `sand_pct`, `silt_pct`, `bulk_density_gcm3`, `ph_water`
- Output: `Predicted_Drainage` (binary classification)

### 4. Evaluation
- Performance metrics and classification report
- Simulation check with edge cases (dry, critical, flooded)
- SHAP analysis for feature interpretability
- Temporal validation over recent weeks

### 5. Output Files
- `drainage_predictions.csv`: Cleaned dataset with raw inputs + model output
- `drainage_ML_original.csv`: ML-ready with original units (for tree-based models)
- `drainage_ML_standardised.csv`: Scaled version (for SVM, ANN, etc.)

### 6. Integration
This model output is now ready to be combined with:
- `ETc` from the evapotranspiration model (Sub-D)
- `AWC` from the soil capacity model (Sub-C)
- Into the team's final **irrigation decision model**
