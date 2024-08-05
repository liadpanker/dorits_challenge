import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from calculate_dt import calculate_dt
import time
import random
from calculate_mutation import process_and_generate_features
import joblib
import json

# Parameter to decide whether to load parameters from JSON or run iterations
load_from_json = True  # Set to False to run iterations instead

# File paths
train_file_path = 'Train_data_original.xlsx'  # Replace with the path to your training Excel file
test_file_path = 'Test_data.xlsx'  # Replace with the path to your testing Excel file
train_summary_file_path = 'Scores_results_for_each_matrix.xlsx'  # Path to the provided Excel file
test_summary_file_path = 'test_Scores_results_for_each_matrix.xlsx'  # Path to the provided Excel file

# Read the original training and testing Excel files
train_features = pd.read_excel(train_file_path, sheet_name='Features')
test_features = pd.read_excel(test_file_path, sheet_name='Features')
train_summary_df = pd.read_excel(train_summary_file_path, sheet_name='Summary')
test_summary_df = pd.read_excel(test_summary_file_path, sheet_name='Summary')

# Load the additional feature files
train_additional_files = [
    "Train_Variants_with_Energy_Window40_Sum40.xlsx",
]

test_additional_files = [
    "Test_Variants_with_Energy_Window40_Sum40.xlsx"
]

# Read and merge additional features
for file in train_additional_files:
    additional_features = pd.read_excel(file)
    train_features = train_features.merge(additional_features, on="Variant number")

for file in test_additional_files:
    additional_features = pd.read_excel(file)
    test_features = test_features.merge(additional_features, on="Variant number")

# Merge the training data with the summary data
train_features = train_features.merge(train_summary_df, on="Variant number")
test_features = test_features.merge(test_summary_df, on="Variant number")

# Mutations:
train_features_df, test_features_df = process_and_generate_features(train_file_path, test_file_path)
train_features = pd.merge(train_features, train_features_df, on='Variant number')
test_features = pd.merge(test_features, test_features_df, on='Variant number')

# Load specific sheets into DataFrames
train_df_without_DNT = pd.read_excel(train_file_path, sheet_name='luminescence without DNT')
train_df_with_DNT = pd.read_excel(train_file_path, sheet_name='luminescence with DNT')
# Calculate Dt and avg_dt for training data
Dt_train, avg_dt_train = calculate_dt(train_df_without_DNT, train_df_with_DNT)
# Create a combined dictionary for easy access
train_targets = {'Dt': Dt_train, 'avg_dt': avg_dt_train}

# Extract the target values into DataFrame
y_train = pd.DataFrame({'Variant number': list(train_targets['Dt'].keys()),
                        'Dt': [value[0] for value in train_targets['Dt'].values()],
                        'Dt_avg': list(train_targets['avg_dt'].values())})

# Ensure the IDs in the features match those in the targets
X_train = train_features.set_index('Variant number')
y_train = y_train.set_index('Variant number')

X_train = X_train.drop(['Changed codons', 'Folding energy window 1', 'Folding energy window 2',
                        'Sum Window 0 from 10', 'Sum Window 1 from 10'], axis=1)

# Parameters for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

if load_from_json:
    # Load the best parameters and settings from the JSON file
    with open('best_model_details_0.7789494322903365_0.7574289778187829.json', 'r') as f:
        best_model_details = json.load(f)

    best_score = best_model_details['best_score']
    best_params = best_model_details['best_params']
    best_random_state = best_model_details['best_random_state']
    best_test_size = best_model_details['best_test_size']
    best_features = best_model_details['best_features']

    print(f"Loaded best model details from 'best_model_details.json'")
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    print(f"Best random state: {best_random_state}")
    print(f"Best test size: {best_test_size}")
    print(f"Best features: {best_features}")



else:
    # Initialize variables to store the best score and parameters
    best_score = float('-inf')
    best_params = None
    best_random_state = -1
    start_time = time.time()

    # Iterate over different subsets of features and randomize random_state
    for iteration in range(900):  # Number of iterations
        random_state = random.randint(1000, 1000000000)

        # Randomly select a subset of features
        selected_features = X_train.columns.tolist()
        random.shuffle(selected_features)
        selected_features = selected_features[:int(1 * len(selected_features))]

        # Initialize GridSearchCV
        grid_search = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid, cv=5, n_jobs=-1, verbose=2)

        for test_size in [0.4]:
            X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train[selected_features],
                                                                                          y_train, test_size=test_size,
                                                                                          random_state=random_state)

            grid_search.fit(X_train_split, y_train_split)
            score = grid_search.best_score_

            if score > best_score:
                best_score = score
                best_params = grid_search.best_params_
                best_test_size = test_size
                best_features = selected_features
                best_random_state = random_state

    elapsed_time = time.time() - start_time
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    print(f"Elapsed time for the loop: {elapsed_time:.2f} seconds")

    # Save the best parameters and settings to a JSON file
    best_model_details = {
        'best_score': best_score,
        'best_params': best_params,
        'best_random_state': best_random_state,
        'best_test_size': best_test_size,
        'best_features': best_features
    }

    with open('best_model_details.json', 'w') as f:
        json.dump(best_model_details, f)

    print(f"Best model details saved to 'best_model_details.json'")

# Split the data using the best random state and test size
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
     X_train[best_features], y_train, test_size=best_test_size, random_state=best_random_state)

# Train the final model with the best parameters and feature set
model = RandomForestRegressor(**best_params, random_state=best_random_state)
X_train_selected = X_train[best_features]
model.fit(X_train_selected, y_train)

# Make predictions on the validation data
y_valid_pred = model.predict(X_valid_split[best_features])
y_valid_pred_df = pd.DataFrame(y_valid_pred, columns=['Predicted Dt', 'Predicted Dt_avg'], index=X_valid_split.index)

# Evaluate the final model
mse_valid = mean_squared_error(y_valid_split, y_valid_pred_df, multioutput='raw_values')
r2_valid = r2_score(y_valid_split, y_valid_pred_df, multioutput='raw_values')

spearman_corr_dt, _ = spearmanr(y_valid_split['Dt'], y_valid_pred_df['Predicted Dt'])
spearman_corr_dt_avg, _ = spearmanr(y_valid_split['Dt_avg'], y_valid_pred_df['Predicted Dt_avg'])

print(f'Mean Squared Error for each target on validation data: {mse_valid}')
print(f'R^2 Score for each target on validation data: {r2_valid}')
print(f"Spearman's rank correlation coefficient for Dt: {spearman_corr_dt}")
print(f"Spearman's rank correlation coefficient for Dt_avg: {spearman_corr_dt_avg}")
print(f"Best random state: {best_random_state}")

# Train the model on the entire training data and make predictions on the test data
model.fit(X_train_selected, y_train)

# Ensure test data has the same order of features
X_test = test_features.set_index('Variant number')[best_features]

# Make predictions on the testing data
y_test_pred = model.predict(X_test)

# Convert predictions to DataFrame
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['Predicted Dt', 'Predicted Dt_avg'], index=X_test.index)
print("Predictions for the test data:")
print(y_test_pred_df)

# Convert predictions to DataFrame
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['Predicted Dt', 'Predicted Dt_avg'],
                              index=test_features['Variant number'])
# Add the Variant number column to the predictions DataFrame
y_test_pred_df['Variant number'] = y_test_pred_df.index

# Reorder the columns to [name, dt_average, DT_max]
y_test_pred_df = y_test_pred_df[['Variant number', 'Predicted Dt_avg', 'Predicted Dt']]

# Rename the columns to match the required names
y_test_pred_df.columns = ['name', 'dt_average', 'DT_max']

# Save the predictions to an Excel file
predictions_file_path = 'test_predictions.xlsx'
y_test_pred_df.to_excel(predictions_file_path, index=False)
print(f"Predictions saved to {predictions_file_path}")