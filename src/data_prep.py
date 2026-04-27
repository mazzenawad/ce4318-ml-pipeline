import pandas as pd
from param import random_seed
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import os

def load_and_preprocess_data(raw_data_path, random_seed=42):
    df = pd.read_csv(raw_data_path)
    df['Age_x_Soil_PH'] = df['Age'] * df['Soil PH']

    x = df.drop('Condition Rating', axis = 1)
    y = df['Condition Rating']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    #Split Categorical and Numerical Features (Seperate heavily skewed from standard)
    skewed_num_cols = ['Diameter', 
                        'Slope', 
                        'Depth', 
                        'Length']
    standard_num_cols = ['Age', 
                        'Soil PH',
                        'Age_x_Soil_PH'
                        ]
    categorical_cols = ['Material',
                        'Road Type', 
                        'Soil Type'
                        ]
    skewed_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
    ('scaler', RobustScaler())
    ])

    standard_num_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('skewed', skewed_pipeline, skewed_num_cols),
        ('standard', standard_num_pipeline, standard_num_cols),
        ('cat', categorical_pipeline, categorical_cols)
        ], verbose_feature_names_out=False
    )

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=random_seed, stratify=y_encoded)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Balance training data with SMOTE
    smote = SMOTE(random_state=random_seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor