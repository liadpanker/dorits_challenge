import pandas as pd
import numpy as np

# Load the Excel file
file_path = 'PSSM.xlsx'
xls = pd.ExcelFile(file_path)

# Load all PSSM sheets
pssm_sheets = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

# Load the variants data (assuming the variants are in a sheet named 'Variants')
variants_data = pd.read_excel('Train_data_original.xlsx', sheet_name='Variants data')

# Function to calculate the score of a subsequence using a PSSM
def calculate_pssm_score(subsequence, pssm):
    score = 1.0
    for i, nucleotide in enumerate(subsequence):
        if nucleotide == 'A':
            score *= pssm.iloc[0, i+1]
        elif nucleotide == 'C':
            score *= pssm.iloc[1, i+1]
        elif nucleotide == 'G':
            score *= pssm.iloc[2, i+1]
        elif nucleotide == 'T':
            score *= pssm.iloc[3, i+1]
    return score

# Initialize a list to store the results
results = []

# Iterate over each PSSM matrix
for motif_name, pssm in pssm_sheets.items():
    window_size = pssm.shape[1] - 1

    # Iterate over each variant
    for idx, row in variants_data.iterrows():
        sequence = row['Variant sequence']

        # Slide the window across the sequence
        for i in range(len(sequence) - window_size + 1):
            subsequence = sequence[i:i + window_size]
            score = calculate_pssm_score(subsequence, pssm)
            results.append({
                'Variant number': row['Variant number'],
                'Motif': motif_name,
                'Start position': i,
                'Subsequence': subsequence,
                'Score': score
            })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new Excel file, each motif in a separate sheet
with pd.ExcelWriter('Total_Scores_results.xlsx') as writer:
    for motif_name in results_df['Motif'].unique():
        motif_results = results_df[results_df['Motif'] == motif_name]
        motif_results.to_excel(writer, sheet_name=motif_name, index=False)

# Display the results
print(results_df)
print(results_df.head())
