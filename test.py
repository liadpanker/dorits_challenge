import pandas as pd
import numpy as np

# Load the Excel file
file_path = 'PSSM.xlsx'
xls = pd.ExcelFile(file_path)

# Load all PSSM sheets
pssm_sheets = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

# Load the variants data (assuming the variants are in a sheet named 'Variants')
variants_data = pd.read_excel('Test_data.xlsx', sheet_name='Variants data')

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

# Function to calculate the numeric score for each variant
def calculate_numeric_score(motif_name, score):
    score_dict = {
        'Motif 1': [(0.000000000000000083, 1)],
        'Motif 2': [(0.00002401, 1)],
        'Motif 3': [(1.6807E-06, 1)],
        'Motif 4': [(7.87693E-14, 1)],
        'Motif 5': [(1.30689E-10, 1)],
        'Motif 6': [(3.08753E-09, 1)],
        'Motif 7': [(5.33471E-10, 1)],
        'Motif 8': [(1.87715E-10, 1)],
        'Motif 9': [(1.49131E-10, 1)],
        'Motif 11': [(3.39097E-12, 1)],
        'Motif 12': [(7.24151E-12, 1)]
    }
    thresholds = score_dict.get(motif_name, [])
    numeric_score = 0
    for threshold, value in thresholds:
        if score > threshold:
            numeric_score += value
    return numeric_score

# Initialize a list to store the results
results = []
variant_motif_sums = {}

# Iterate over each PSSM matrix
for motif_name, pssm in pssm_sheets.items():
    window_size = pssm.shape[1] - 1

    # Iterate over each variant
    for idx, row in variants_data.iterrows():
        sequence = row['Variant sequence']
        sequence = sequence.strip("'")

        total_numeric_score = 0

        # Slide the window across the sequence
        for i in range(len(sequence) - window_size + 1):
            subsequence = sequence[i:i + window_size]
            score = calculate_pssm_score(subsequence, pssm)
            numeric_score = calculate_numeric_score(motif_name, score)
            total_numeric_score += numeric_score
            results.append({
                'Variant number': row['Variant number'],
                'Motif': motif_name,
                'Start position': i,
                'Subsequence': subsequence,
                'Score': score,
                'Numeric Score': numeric_score
            })

        if row['Variant number'] not in variant_motif_sums:
            variant_motif_sums[row['Variant number']] = {}
        if motif_name not in variant_motif_sums[row['Variant number']]:
            variant_motif_sums[row['Variant number']][motif_name] = 0
        variant_motif_sums[row['Variant number']][motif_name] += total_numeric_score

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Create a summary DataFrame for the variant-motif sums
summary_data = []
for variant_number, motifs in variant_motif_sums.items():
    row = {'Variant number': variant_number}
    row.update(motifs)
    summary_data.append(row)
summary_df = pd.DataFrame(summary_data).fillna(0)

# Dynamically determine the motif columns present in the data
motif_columns = [col for col in summary_df.columns if col != 'Variant number']
summary_df = summary_df[['Variant number'] + sorted(motif_columns)]

# Save the results to a new Excel file, each motif in a separate sheet
with pd.ExcelWriter('test_Scores_results_for_each_matrix.xlsx') as writer:
    for motif_name in results_df['Motif'].unique():
        motif_results = results_df[results_df['Motif'] == motif_name]
        motif_results.to_excel(writer, sheet_name=motif_name, index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

# Display the results
print(results_df)
print(results_df.head())
print(summary_df)
print(summary_df.head())
