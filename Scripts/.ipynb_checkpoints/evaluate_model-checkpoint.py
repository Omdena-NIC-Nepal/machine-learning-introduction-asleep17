import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the processed data 
data = pd.read_csv('../Data/processed_boston.csv')

# Define features and target variable
X = data.drop('medv', axis=1)
y = np.log(data['medv'] + 1)  # Log-transform the target variable

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model (Ridge regression model or any model trained on the log-transformed target)
best_model = joblib.load('models/ridge_best_model.pkl')

# Evaluate the model
y_pred = best_model.predict(X_test)

# Reverse the log transformation for evaluation metrics
y_test_original = np.exp(y_test) - 1
y_pred_original = np.exp(y_pred) - 1

# Calculate Mean Squared Error (MSE) and R-squared (R²) for the original scale
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

# Print evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Residuals Plot (in original scale)
residuals = y_test_original - y_pred_original

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_original, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values (Original Scale)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Actual vs Predicted Plot (in original scale)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, color='blue', alpha=0.5)
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--')
plt.title('Actual vs Predicted Values (Original Scale)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
