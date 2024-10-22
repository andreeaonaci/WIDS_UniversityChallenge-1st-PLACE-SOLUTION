import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
import numpy as np
import random

# Generate a random number for the random state
RandomNr = random.randint(0, 2048)
print(f"Random state for train_test_split: {RandomNr}")

# Load the training dataset into a DataFrame
train_df = pd.read_csv('/kaggle/working/output.csv')

# Separate features (X) and target variable (y)
X = train_df.iloc[:, 1:-1]  # The first column is the ID and the last column is the target
y = train_df.iloc[:, -1]

# Load the test dataset into a DataFrame
test_df = pd.read_csv('/kaggle/input/widsdata/test.csv')

# Identify numeric and non-numeric columns
numeric_cols = X.select_dtypes(include=['number']).columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns

# Preprocess numerical columns
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocess categorical columns with target encoding
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='novalue')),
    ('target_encoder', TargetEncoder())
])

# Combine numerical and categorical preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, non_numeric_cols)
    ])

# Create XGBoost model pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(colsample_bytree=1, colsample_bylevel=0.90))
])

# Reduce the number of parameter combinations
param_grid = {
    'xgb__learning_rate': [0.047], # 0.0337
#     'xgb__max_depth': [4,3],
    'xgb__n_estimators': [100],
    'xgb__min_child_weight': [5],

}

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RandomNr)

# Reduce the number of cross-validation folds
grid_search = GridSearchCV(estimator=xgb_pipeline, param_grid=param_grid, cv=9, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Calculate Mean Squared Error on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = pow(mse_val, 0.5)

print(f'Best Parameters: {best_params}')
print(f'Mean Squared Error on the validation set: {mse_val}')
print(f'Root Mean Squared Error on the validation set: {rmse_val}')
print(f'Best Negative Mean Squared Error: {best_score}')

# Add the new column to test_df
# test_df['new_column'] = new_column

# Make predictions on the test set using the best model
print("Making predictions on the test set...")
predictions_test = best_model.predict(test_df.iloc[:, 1:])  # Assuming the first column is the ID
print("Predictions completed.")

# Write predicted numbers to a text file using index as IDs
print("Writing rounded predictions to file...")
with open('SampleSolution.csv', 'w') as file:
    file.write("patient_id,treatment_pd\n")
    for index, prediction in zip(test_df.iloc[:, 0], predictions_test):  # Assuming the first column is the ID
        file.write(f'{index},{int(round(prediction))}\n')
print("Writing to file completed.")
