import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the processed data
data = pd.read_csv('../Data/processed_boston.csv')

# Define features and target variable
X = data.drop('medv', axis=1)
y = np.log(data['medv'] + 1)  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Ridge model and hyperparameter grid
ridge = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform GridSearchCV to find the best alpha
grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Get the best model
best_model = grid.best_estimator_
print("Best alpha:", grid.best_params_['alpha'])

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R² Score:", r2)

import os
import joblib

# Ensure the directory exists for saving the model
output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the best model
joblib.dump(best_model, os.path.join(output_dir, 'ridge_best_model.pkl'))

# Save the best model
joblib.dump(best_model, 'models/ridge_best_model.pkl')
print("Model saved to models/ridge_best_model.pkl")
