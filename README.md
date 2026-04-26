# ce4318-ml-pipeline
# Underground Pipe Condition Predictor

## Project Overview
Urban infrastructure decay is a pressing challenge for city planners and civil engineers. This project develops a predictive machine learning pipeline to classify the structural condition of underground pipes (on a 1-to-5 degradation rating). By analyzing physical and environmental factors, this tool allows municipalities to proactively prioritize maintenance and prevent critical infrastructure failures.

## Data
The project uses synthetic infrastructure data simulating real-world municipal environments.
* **Source:** `pipe_condition_class_synthetic.csv`
* **Target Variable:** `Condition Rating` (Ordinal classification 1 to 5)
* **Numerical Features:** `Age`, `Diameter`, `Slope`, `Depth`, `Length`, `Soil PH`
* **Categorical Features:** `Material` (e.g., PVC, VCP, RC), `Soil Type` (e.g., Clay, Sand, Loam), `Road Type`

## Workflow

```text
[Raw Data] -> [Preprocessing] -> [Class Balancing] -> [Model Training] -> [Evaluation]
                 (Scaling/           (SMOTE)           (XGBoost)        (Metrics/Visuals)
                  Encoding)