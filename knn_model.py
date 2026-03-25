import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv('knn_regression_dataset.csv')

# Features and target
features = ['age', 'income', 'loan_amount', 'credit_score', 'city', 'employment_type']
X = df[features]
y = df['target']

# Preprocessing pipeline
numeric_features = ['age', 'income', 'loan_amount', 'credit_score']
categorical_features = ['city', 'employment_type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder())
])

# Split categorical for LabelEncoder (fits on train only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit encoders on train data
city_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
X_train['city_encoded'] = city_encoder.fit_transform(X_train['city'].fillna('missing'))
X_train['employment_encoded'] = employment_encoder.fit_transform(X_train['employment_type'].fillna('missing'))

X_test['city_encoded'] = city_encoder.transform(X_test['city'].fillna('missing'))
X_test['employment_encoded'] = employment_encoder.transform(X_test['employment_type'].fillna('missing'))

# Prepare final features (numeric + encoded categoricals)
final_features = ['age', 'income', 'loan_amount', 'credit_score', 'city_encoded', 'employment_encoded']
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

numeric_imputer = SimpleImputer(strategy='median')
X_train_final[numeric_features] = numeric_imputer.fit_transform(X_train_final[numeric_features])
X_test_final[numeric_features] = numeric_imputer.transform(X_test_final[numeric_features])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# Train model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_final, y_train)
test_score = model.score(X_test_final, y_test)
print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# Save model, encoders, imputer, scaler
joblib.dump(model, 'knn_model.pkl')
joblib.dump({
    'city_encoder': city_encoder, 
    'employment_encoder': employment_encoder,
    'numeric_imputer': numeric_imputer,
    'scaler': scaler
}, 'preprocessors.pkl')
print("Model and preprocessors saved!")

if __name__ == "__main__":
    print("Run this script to train and save the KNN model.")

