import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer

# Load the training dataset into a DataFrame
train_df = pd.read_csv('train.csv')

# Separate features (X) and target variable (y)
X = train_df.iloc[:, 1:-1]  # Assuming the first column is the ID and the last column is the target
y = train_df.iloc[:, -1]

# Load the test dataset into a DataFrame
test_df = pd.read_csv('test.csv')

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
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
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
    ('xgb', XGBRegressor(random_state=69, colsample_bytree=1.0, colsample_bylevel=0.75))
])

# Reduce the number of parameter combinations
param_grid = {
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 5, 7, 9],
    'xgb__min_child_weight': [1, 3, 5, 7],
    'xgb__gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'xgb__subsample': [0.5, 0.7, 0.9],
    'xgb__colsample_bytree': [0.5, 0.7, 0.9],
}

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Reduce the number of cross-validation folds
grid_search = GridSearchCV(estimator=xgb_pipeline, param_grid=param_grid, cv=4, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)

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


# Make predictions on the test set using the best model
print("Making predictions on the test set...")
predictions_test = best_model.predict(test_df.iloc[:, 1:])  # Assuming the first column is the ID
print("Predictions completed.")

# Write predicted numbers to a text file using index as IDs
print("Writing rounded predictions to file...")
with open('SampleSolution.txt', 'w') as file:
    for index, prediction in zip(test_df.iloc[:, 0], predictions_test):  # Assuming the first column is the ID
        file.write(f'{index},{int(round(prediction))}\n')
print("Writing to file completed.")
