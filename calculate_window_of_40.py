from seqfold import dg
import pandas as pd
import time

def calculate_total_energy(sequence, temp=37.0):
    """
    Calculate the folding energy for the entire given sequence.

    Parameters:
    - sequence: The RNA or DNA sequence (string).
    - temp: The temperature for the energy calculation (default is 37.0).

    Returns:
    - The folding energy for the sequence.
    """
    return dg(sequence, temp=temp)

def calculate_energies_for_windows(file_path, sheet_name, window_size=40, windows_to_sum=20, output_file_path='output.xlsx'):
    """
    Calculate the sum of folding energies for each window of a given size
    and save the results to an Excel file.

    Parameters:
    - file_path: Path to the Excel file containing variant data.
    - sheet_name: The sheet name containing the sequence data.
    - window_size: The size of each window (default is 40).
    - windows_to_sum: The number of windows to sum (default is 20).
    - output_file_path: The path to save the output Excel file.
    """
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Load the data from the specified sheet
    variants_data = pd.read_excel(xls, sheet_name=sheet_name)

    # Clean up the sequences
    variants_data['Variant sequence'] = variants_data['Variant sequence'].str.replace("'", "")

    # Initialize list to store results
    results = []

    # Loop through each sequence in the data
    start_time = time.time()
    for idx, sequence in enumerate(variants_data['Variant sequence']):
        print(f'Processing Variant {variants_data["Variant number"][idx]}')
        # Calculate energy for each window
        energies = []
        for i in range(0, len(sequence) - window_size + 1, 1):
            window_sequence = sequence[i:i + window_size]
            energy = calculate_total_energy(window_sequence)
            energies.append(energy)

        # Sum every specified number of windows
        sum_energies = [
            sum(energies[i:i + windows_to_sum])
            for i in range(0, len(energies), windows_to_sum)
        ]
        # Add variant number and results to the list
        results.append([variants_data['Variant number'][idx]] + sum_energies)
    # Create a new DataFrame for the results
    results_df = pd.DataFrame(results)
    results_df.columns = ['Variant'] + [f'Sum Window {i}' for i in range(results_df.shape[1] - 1)]

    # Save the results to a new Excel file
    results_df.to_excel(output_file_path, index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken for calculation: {execution_time} seconds")

# Set parameters
window_size = 40
windows_to_sum = 20

# Process the train data
train_file_path = 'Train_data_original.xlsx'  # Update this path if needed
calculate_energies_for_windows(train_file_path, 'Variants data', window_size, windows_to_sum, 'Train_Variants_with_Total_Energy.xlsx')

# Process the test data
test_file_path = 'Test_data.xlsx'  # Update this path if needed
calculate_energies_for_windows(test_file_path, 'Variants data', window_size, windows_to_sum, 'Test_Variants_with_Total_Energy.xlsx')
