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

# Initialize a dictionary to store the total scores for each variant
total_scores = {}

# Initialize a list to store the results
results = []

# Iterate over each PSSM matrix
for motif_name, pssm in pssm_sheets.items():
    window_size = pssm.shape[1] - 1

    # Iterate over each variant
    for idx, row in variants_data.iterrows():
        sequence = row['Variant sequence']
        variant_number = row['Variant number']
        variant_total_score = 0.0

        # Slide the window across the sequence
        for i in range(len(sequence) - window_size + 1):
            subsequence = sequence[i:i + window_size]
            score = calculate_pssm_score(subsequence, pssm)
            variant_total_score += score
            results.append({
                'Variant number': variant_number,
                'Motif': motif_name,
                'Start position': i,
                'Subsequence': subsequence,
                'Score': score
            })

        # Add the total score for this variant to the total_scores dictionary
        if variant_number in total_scores:
            total_scores[variant_number] += variant_total_score
        else:
            total_scores[variant_number] = variant_total_score

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Convert the total scores dictionary to a DataFrame
total_scores_df = pd.DataFrame(list(total_scores.items()), columns=['Variant number', 'Total Score'])

# Display the results
print(results_df)
print(results_df.head())

print(total_scores_df)
print(total_scores_df.head())

