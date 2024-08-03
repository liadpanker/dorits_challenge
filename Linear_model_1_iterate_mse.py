import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from calculate_dt import calculate_dt
from calculate_mutation import process_and_generate_features
import time
import joblib

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

additional_features = None
# Read and merge additional features
for file in train_additional_files:
    additional_features = pd.read_excel(file)
    train_features = train_features.merge(additional_features, on="Variant number")

additional_features = None
# Read and merge additional features
for file in test_additional_files:
    additional_features = pd.read_excel(file)
    test_features = test_features.merge(additional_features, on="Variant number")

# Extract the motif columns and the Variant number
train_motif_columns = train_summary_df.filter(regex='^Motif')  # Assuming columns start with 'motif'
train_motif_columns['Variant number'] = train_summary_df['Variant number']
test_motif_columns = test_summary_df.filter(regex='^Motif')  # Assuming columns start with 'motif'
test_motif_columns['Variant number'] = test_summary_df['Variant number']

# Merge the motif columns into the additional features DataFrame
train_features = pd.merge(train_features, train_motif_columns, on='Variant number')
test_features = pd.merge(test_features, test_motif_columns, on='Variant number')

# Mutations:
train_features_df, test_features_df = process_and_generate_features(train_file_path, test_file_path)
train_features = pd.merge(train_features, train_features_df, on='Variant number')
test_features = pd.merge(test_features, test_features_df, on='Variant number')

# Load the additional sheets for the luminescence data
train_data = pd.ExcelFile(train_file_path)
test_data = pd.ExcelFile(test_file_path)

# Load specific sheets into DataFrames
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

# Ensure the IDs in the features match those in the targets
X_train = train_features.set_index('Variant number')
y_train = y_train.set_index('Variant number')

X_train = X_train.drop('Changed codons', axis=1)
X_train = X_train.drop('Folding energy window 1', axis=1)
X_train = X_train.drop('Folding energy window 2', axis=1)
X_train = X_train.drop('Sum Window 0 from 10', axis=1)
X_train = X_train.drop('Sum Window 1 from 10', axis=1)

# Initialize lists to keep track of selected features and their MSE scores
selected_features = []
mse_scores = []

# Step 1: Start with running the model with each feature individually
remaining_features = list(X_train.columns)

# Parameter X (number of iterations)
X = 50000
# Initialize best parameters
best_test_size = None
best_random_state = None
best_mse = float('inf')
best_feature_mse = float('inf')
# Start time measurement
start_time = time.time()

for i in range(X):
    test_size = 0.4
    random_state = np.random.randint(1, 10000)
    current_selected_features = []
    current_remaining_features = remaining_features.copy()
    current_mse_scores = []

    while current_remaining_features:
        best_feature = None
        for feature in current_remaining_features:
            current_features = current_selected_features + [feature]
            X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
                X_train[current_features], y_train, test_size=test_size, random_state=random_state)

            model = LinearRegression()
            model.fit(X_train_split, y_train_split)

            y_valid_pred = model.predict(X_valid_split)
            y_valid_pred_df = pd.DataFrame(y_valid_pred, columns=['Predicted Dt', 'Predicted Dt_avg'],
                                           index=X_valid_split.index)

            mse_dt = mean_squared_error(y_valid_split['Dt'], y_valid_pred_df['Predicted Dt'])
            mse_dt_avg = mean_squared_error(y_valid_split['Dt_avg'], y_valid_pred_df['Predicted Dt_avg'])

            avg_mse = (mse_dt + mse_dt_avg) / 2

            if avg_mse < best_feature_mse:
                best_feature_mse = avg_mse
                best_feature = feature

        if best_feature:
            current_selected_features.append(best_feature)
            current_remaining_features.remove(best_feature)
            current_mse_scores.append(best_feature_mse)
        else:
            break

    # Update the best parameters if current iteration is better
    if best_feature_mse < best_mse:
        best_mse = best_feature_mse
        selected_features = current_selected_features.copy()
        mse_scores = current_mse_scores.copy()
        best_test_size = test_size
        best_random_state = random_state

# End time measurement
end_time = time.time()
elapsed_time = end_time - start_time

# Print the final selected features and elapsed time
print("Selected features based on Mean Squared Error:")
print(selected_features)

print(f"Best test size: {best_test_size}")
print(f"Best random state: {best_random_state}")
print(f"Elapsed time for the loop: {elapsed_time:.2f} seconds")

# Final model training with the selected features
X_train_selected = X_train[selected_features]
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train_selected, y_train, test_size=best_test_size, random_state=best_random_state)

model = LinearRegression()
model.fit(X_train_split, y_train_split)

y_valid_pred = model.predict(X_valid_split)
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

# Now, train the model on the entire training data and make predictions on the test data
model.fit(X_train_selected, y_train)
selected_features = selected_features + ['Variant number']
# Make predictions on the testing data
y_test_pred = model.predict(test_features[selected_features].set_index('Variant number'))

# Convert predictions to DataFrame
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['Predicted Dt', 'Predicted Dt_avg'],
                              index=test_features['Variant number'])
print("Predictions for the test data:")
print(y_test_pred_df)

# Optional: Display the coefficients
coefficients = pd.DataFrame(model.coef_, columns=selected_features.remove('Variant number'), index=['Dt', 'Dt_avg'])
print(coefficients)

# Save the model
model_path = 'linear_regression_model_mse_iterate.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")