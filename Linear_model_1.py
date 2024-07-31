import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from calculate_dt import calculate_dt

# File paths
train_file_path = 'Train_data_original.xlsx'  # Replace with the path to your training Excel file
test_file_path = 'Test_data.xlsx'    # Replace with the path to your testing Excel file
train_additional_features_path = 'Train_Variants_with_Total_Energy.xlsx'
test_additional_features_path = 'Test_Variants_with_Total_Energy.xlsx'
train_summary_file_path = 'Scores_results_for_each_matrix.xlsx'  # Path to the provided Excel file
test_summary_file_path = 'test_Scores_results_for_each_matrix.xlsx'  # Path to the provided Excel file

# Read the Excel files
train_features = pd.read_excel(train_file_path, sheet_name='Features')
test_features = pd.read_excel(test_file_path, sheet_name='Features')
train_additional_features = pd.read_excel(train_additional_features_path)
test_additional_features = pd.read_excel(test_additional_features_path)
train_summary_df = pd.read_excel(train_summary_file_path, sheet_name='Summary')
test_summary_df = pd.read_excel(test_summary_file_path, sheet_name='Summary')

# Extract the motif columns and the Variant number
train_motif_columns = train_summary_df.filter(regex='^Motif')  # Assuming columns start with 'motif'
train_motif_columns['Variant number'] = train_summary_df['Variant number']
test_motif_columns = test_summary_df.filter(regex='^Motif')  # Assuming columns start with 'motif'
test_motif_columns['Variant number'] = test_summary_df['Variant number']

# Merge the motif columns into the additional features DataFrame
train_additional_features = pd.merge(train_additional_features, train_motif_columns, on='Variant number')
test_additional_features = pd.merge(test_additional_features, test_motif_columns, on='Variant number')

# Display the first few rows of the dataframes
print("Training Features:")
print(train_features.head())
print("\nTesting Features:")
print(test_features.head())

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

# Step 3: Build and train the linear regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Step 4: Make predictions on the validation data
y_valid_pred = model.predict(X_valid_split)

# Convert predictions to DataFrame
y_valid_pred_df = pd.DataFrame(y_valid_pred, columns=['Predicted Dt', 'Predicted Dt_avg'], index=X_valid_split.index)

# Step 5: Evaluate the model on the validation data
mse_valid = mean_squared_error(y_valid_split, y_valid_pred_df, multioutput='raw_values')
r2_valid = r2_score(y_valid_split, y_valid_pred_df, multioutput='raw_values')

print(f'Mean Squared Error for each target on validation data: {mse_valid}')
print(f'R^2 Score for each target on validation data: {r2_valid}')

# Now, train the model on the entire training data and make predictions on the test data
model.fit(X_train, y_train)

# Make predictions on the testing data
X_test_scaled = test_features.set_index('Variant number')  # Ensure the IDs in the features match
y_test_pred = model.predict(X_test_scaled)

# Convert predictions to DataFrame
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['Predicted Dt', 'Predicted Dt_avg'], index=test_features['Variant number'])
print("Predictions for the test data:")
print(y_test_pred_df)

# Optional: Display the coefficients
coefficients = pd.DataFrame(model.coef_, columns=X_train.columns, index=['Dt', 'Dt_avg'])
print(coefficients)
