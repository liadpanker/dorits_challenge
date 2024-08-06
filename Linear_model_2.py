import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from calculate_dt import calculate_dt
from calculate_mutation import process_and_generate_features
import numpy as np
import random
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
    "Train_Variants_with_Energy_Window40_Sum40.xlsx"
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

# Remove unwanted features
unwanted_features = ['Changed codons', 'Folding energy window 1', 'Folding energy window 2',
                     'Sum Window 0 from 10', 'Sum Window 1 from 10']
X_train = X_train.drop(columns=unwanted_features, errors='ignore')

# Initialize best parameters
best_test_size = 0.4
best_random_state = 485575
selected_features = ['mut_288-312', 'Sum Window 8 from 10', 'Sum Window 6 from 10', 'CAI', 'mut_357-364', 'mut_246-279', 'mut_384-393']

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
model.fit(X_train, y_train)

# Make predictions on the testing data
test_features = test_features.drop(columns=unwanted_features, errors='ignore')
y_test_pred = model.predict(test_features.set_index('Variant number'))

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
predictions_file_path = 'LR_test_predictions.xlsx'
y_test_pred_df.to_excel(predictions_file_path, index=False)
print(f"Predictions saved to {predictions_file_path}")

# Save the model
model_path = 'LR_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")