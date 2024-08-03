import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from calculate_dt import calculate_dt

# File paths
train_file_path = 'Train_data_original.xlsx'
test_file_path = 'Test_data.xlsx'
train_additional_features_path = 'Train_Variants_with_Total_Energy.xlsx'
test_additional_features_path = 'Test_Variants_with_Total_Energy.xlsx'
train_summary_file_path = 'Scores_results_for_each_matrix.xlsx'
test_summary_file_path = 'test_Scores_results_for_each_matrix.xlsx'

# Read the Excel files
train_features = pd.read_excel(train_file_path, sheet_name='Features')
test_features = pd.read_excel(test_file_path, sheet_name='Features')
train_additional_features = pd.read_excel(train_additional_features_path)
test_additional_features = pd.read_excel(test_additional_features_path)
train_summary_df = pd.read_excel(train_summary_file_path, sheet_name='Summary')
test_summary_df = pd.read_excel(test_summary_file_path, sheet_name='Summary')

# Extract the motif columns and the Variant number
train_motif_columns = train_summary_df.filter(regex='^Motif')
train_motif_columns['Variant number'] = train_summary_df['Variant number']
test_motif_columns = test_summary_df.filter(regex='^Motif')
test_motif_columns['Variant number'] = test_summary_df['Variant number']

# Merge the motif columns into the additional features DataFrame
train_additional_features = pd.merge(train_additional_features, train_motif_columns, on='Variant number')
test_additional_features = pd.merge(test_additional_features, test_motif_columns, on='Variant number')

# Load the additional sheets for the luminescence data
train_data = pd.ExcelFile(train_file_path)
test_data = pd.ExcelFile(test_file_path)
train_df_without_DNT = pd.read_excel(train_data, sheet_name='luminescence without DNT')
train_df_with_DNT = pd.read_excel(train_data, sheet_name='luminescence with DNT')

# Calculate Dt and avg_dt for training data
Dt_train, avg_dt_train = calculate_dt(train_df_without_DNT, train_df_with_DNT)

# Create a combined dictionary for easy access
train_targets = {'Dt': Dt_train, 'avg_dt': avg_dt_train}

# Extract the target values into DataFrame
y_train = pd.DataFrame({'Variant number': list(train_targets['Dt'].keys()),
                        'Dt': [value[0] for value in train_targets['Dt'].values()],
                        'Dt_avg': list(train_targets['avg_dt'].values())})

# Merge the additional features with the training and testing data
train_features = pd.merge(train_features, train_additional_features, on='Variant number')
test_features = pd.merge(test_features, test_additional_features, on='Variant number')

# Ensure the IDs in the features match those in the targets
X_train = train_features.set_index('Variant number')
y_train = y_train.set_index('Variant number')

# Step 2: Split the training data into training and validation sets
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize each column separately using training data parameters
scalers = {}
for column in X_train.columns:
    scaler = StandardScaler()
    X_train_split[column] = scaler.fit_transform(X_train_split[[column]])
    X_valid_split[column] = scaler.transform(X_valid_split[[column]])
    X_train[column] = scaler.fit_transform(X_train[[column]])
    test_features[column] = scaler.transform(test_features[[column]])
    scalers[column] = scaler

# Function to evaluate and train model
def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_valid_pred)
    r2 = r2_score(y_valid, y_valid_pred)
    return model, mse, r2

# Define the models
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100)
}

# Train and evaluate each model for both Dt and Dt_avg
results = {}
for target in ['Dt', 'Dt_avg']:
    results[target] = {}
    for model_name, model in models.items():
        trained_model, mse_valid, r2_valid = evaluate_model(model, X_train_split, y_train_split[target], X_valid_split, y_valid_split[target])
        results[target][model_name] = {"model": trained_model, "mse": mse_valid, "r2": r2_valid}
        print(f"Target: {target}, Model: {model_name}, MSE: {mse_valid}, R^2: {r2_valid}")

# Choose the best model for each target based on MSE
best_models = {}
for target in ['Dt', 'Dt_avg']:
    best_model_name = min(results[target], key=lambda k: results[target][k]["mse"])
    best_models[target] = results[target][best_model_name]["model"]
    print(f"Best Model for {target}: {best_model_name}, MSE: {results[target][best_model_name]['mse']}, R^2: {results[target][best_model_name]['r2']}")

# Make predictions on the testing data with the best models
X_test_scaled = test_features.set_index('Variant number')
y_test_pred_dt = best_models['Dt'].predict(X_test_scaled)
y_test_pred_dt_avg = best_models['Dt_avg'].predict(X_test_scaled)

# Convert predictions to DataFrame
y_test_pred_df = pd.DataFrame({'Predicted Dt': y_test_pred_dt, 'Predicted Dt_avg': y_test_pred_dt_avg}, index=test_features['Variant number'])
print("Predictions for the test data using the best models:")
print(y_test_pred_df)

# Save the predictions
y_test_pred_df.to_csv('test_predictions.csv')

# Optional: Save the best models
import joblib
for target, model in best_models.items():
    model_path = f'best_model_{target}.pkl'
    joblib.dump(model, model_path)
    print(f'Best model for {target} saved to {model_path}')
