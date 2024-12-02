# Re-importing required modules and reloading data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Reloading datasets
train_data_path = '/mnt/data/Train_dataset_(1)_(1).csv'
test_data_path = '/mnt/data/Test_dataset_(1)_(1).csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Standardizing column names again for consistency
train_data.columns = train_data.columns.str.strip().str.lower().str.replace(' ', '_')
test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_')

# Align column names due to discrepancies
train_data.columns = train_data.columns.str.replace('endorsed_by', 'endoresed_by')

# Handle missing values and preprocess
train_data = train_data.fillna(train_data.median(numeric_only=True))
test_data = test_data.fillna(test_data.median(numeric_only=True))
train_data['restaurant_age'] = 2024 - pd.to_datetime(train_data['opening_day_of_restaurant'], errors='coerce').dt.year
test_data['restaurant_age'] = 2024 - pd.to_datetime(test_data['opening_day_of_restaurant'], errors='coerce').dt.year
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# Encoding categorical data
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Splitting data for training and testing
X = train_data.drop(columns=['annual_turnover'], errors='ignore')
y = train_data['annual_turnover']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
et_param_grid = rf_param_grid

# RandomizedSearchCV for Random Forest
rf_random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                      param_distributions=rf_param_grid,
                                      n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
rf_random_search.fit(X_train, y_train)

# RandomizedSearchCV for Extra Trees
et_random_search = RandomizedSearchCV(ExtraTreesRegressor(random_state=42),
                                      param_distributions=et_param_grid,
                                      n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
et_random_search.fit(X_train, y_train)

# Best models
best_rf_model = rf_random_search.best_estimator_
best_et_model = et_random_search.best_estimator_

# Predictions with the tuned models
best_rf_preds = best_rf_model.predict(X_valid)
best_et_preds = best_et_model.predict(X_valid)

# Calculate RMSE for tuned models
best_rf_rmse = np.sqrt(mean_squared_error(y_valid, best_rf_preds))
best_et_rmse = np.sqrt(mean_squared_error(y_valid, best_et_preds))

# Final predictions for the test set using the best Extra Trees model
final_predictions = best_et_model.predict(test_data.drop(columns=['annual_turnover'], errors='ignore'))

# Prepare the results as per the required format
submission = pd.DataFrame({
    'Registration Number': test_data['registration_number'],
    'Annual Turnover': final_predictions
})

# Save to CSV
submission_file_path = '/mnt/data/Restaurant_Turnover_Predictions.csv'
submission.to_csv(submission_file_path, index=False)

best_rf_rmse, best_et_rmse, submission_file_path
