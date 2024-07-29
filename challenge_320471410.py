import pandas as pd
from seqfold import dg
import numpy as np

# Load the Excel file
train_data = 'Train_data_original.xlsx'
xls = pd.ExcelFile(train_data)
PSSM = 'PSSM.xlsx'


# Load the specific sheets into DataFrames
df_without_DNT = pd.read_excel(xls, sheet_name='luminescence without DNT')
df_with_DNT = pd.read_excel(xls, sheet_name='luminescence with DNT')
df_variants = pd.read_excel(xls, sheet_name='Variants data')
control_sequence = df_variants[df_variants['Variant number'] == 1]['Variant sequence'].values[0].strip("'")
# Load all PSSM sheets
pssm_sheets = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

# Function to create luminescence dictionary
def create_luminescence_dict(df):
    luminescence_dict = {}
    for index, row in df.iterrows():
        variant_number = row['Variant number']
        luminescence_values = row.drop('Variant number').tolist()
        luminescence_dict[variant_number] = luminescence_values
    return luminescence_dict


# Calculateing the different features
# Function to calculate GC content
def calculate_gc_content(sequence):
    """The percentage of guanine (G) and cytosine (C) nucleotides in the DNA sequence"""
    sequence = sequence.strip("'")
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    gc_content = ((g_count + c_count) / len(sequence)) * 100
    return gc_content


# Function to calculate mutations compared to control
def calculate_mutations(variant_sequence, control_sequence):
    variant_sequence = variant_sequence.strip("'")  # Remove leading and trailing single quotes
    control_sequence = control_sequence.strip("'")
    mutations = []
    for i, (var_nuc, ctrl_nuc) in enumerate(zip(variant_sequence, control_sequence)):
        if var_nuc != ctrl_nuc:
            mutations.append((i, var_nuc, ctrl_nuc))
    mutation_count = len(mutations)
    return mutation_count, mutations


def calculate_total_energy(sequence, window_size, temp=37.0):
    """
    Calculate the total folding energy for the given sequence by summing the energies
    """
    total_energy = 0
    # Loop through the sequence in steps of 1 to get overlapping windows
    for i in range(len(sequence) - window_size + 1):
        window_seq = sequence[i:i + window_size]
        energy = dg(window_seq, temp=temp)
        total_energy += energy

    return total_energy


def energy_diff_compare_to_control(variants_data, control_sequence, window_size=None, temp=37.0):
    control_energy = calculate_total_energy(control_sequence, window_size=window_size, temp=temp)
    comparison_data = []

    for i, row in variants_data.iterrows():
        sequence = row['Variant sequence']
        variant_number = row['Variant number']
        variant_energy = calculate_total_energy(sequence, window_size=window_size, temp=temp)
        energy_difference = variant_energy - control_energy
        comparison_data.append((variant_number, variant_energy, control_energy, energy_difference))

    comparison_df = pd.DataFrame(comparison_data, columns=['Variant number', 'Variant Energy', 'Control Energy',
                                                           'Energy Difference'])
    return comparison_df


def extract_cai_tai_to_dataframe(file_path, sheet_name='Features'):
    """
    Extracts CAI values from the specified sheet of the given Excel file and saves them into a DataFrame.
    """
    # Load the Excel file
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract the CAI values for each variant into a dictionary
    cai_tai_df = data[['Variant number', 'CAI', 'tAI']]

    return cai_tai_df


# Function to calculate the score of a subsequence using a PSSM
def calculate_pssm_score(subsequence, pssm):
    score = 1.0
    for i, nucleotide in enumerate(subsequence):
        if nucleotide == 'A':
            score *= pssm.iloc[0, i]
        elif nucleotide == 'C':
            score *= pssm.iloc[1, i]
        elif nucleotide == 'G':
            score *= pssm.iloc[2, i]
        elif nucleotide == 'T':
            score *= pssm.iloc[3, i]
    return score

#~~~~~~~~~~~~~~~calculation of dt, dt avg, and Dt ~~~~~~~~~~~~~~~~~~~~#
# Create dictionaries for both sheets


luminescence_without_DNT_dict = create_luminescence_dict(df_without_DNT)
luminescence_with_DNT_dict = create_luminescence_dict(df_with_DNT)

# Create the dt dictionary by subtracting the values
dt_dict = {}
for variant in luminescence_with_DNT_dict:
    if variant in luminescence_without_DNT_dict:
        dt_dict[variant] = [with_DNT - without_DNT for with_DNT, without_DNT in zip(luminescence_with_DNT_dict[variant], luminescence_without_DNT_dict[variant])]


avg_dt = {variant: sum(values) / len(values) for variant, values in dt_dict.items()}

# Calculate the maximal value Dt for each variant and the corresponding time
Dt = {}
for variant, values in dt_dict.items():
    max_value = max(values)
    max_index = values.index(max_value)  # Index of the maximum value
    time_of_max_value = df_with_DNT.columns[max_index + 1]  # Adjust for the 0-based index and column shift
    Dt[variant] = (max_value, time_of_max_value)
print(Dt)



#~~~~~~~~~~~~~~~calculate the features for each variant~~~~~~~~~~~~~~~#
#calculate the GC content for each variant
df_variants = pd.read_excel(xls, sheet_name='Variants data')
gc_content_dict = {}
for index, row in df_variants.iterrows():
    variant_number = row['Variant number']
    sequence = row['Variant sequence']  # Assuming the sequence is in a column named 'Sequence'
    gc_content = calculate_gc_content(sequence)
    gc_content_dict[variant_number] = gc_content


# Calculate mutations for each variant
mutations_dict = {}
for index, row in df_variants.iterrows():
    variant_number = row['Variant number']
    sequence = row['Variant sequence']  # Extracting and cleaning the sequence
    mutation_count, mutations = calculate_mutations(sequence, control_sequence)
    mutations_dict[variant_number] = {'count': mutation_count, 'locations': mutations}


# Print the GC content dictionary to verify
print("GC content dictionary:")
print(list(gc_content_dict.items())[:5])

# Print the mutations dictionary to verify
print("\nMutations dictionary:")
for variant, details in list(mutations_dict.items())[:4]:
    print(f"Variant {variant}:")
    print(f"  Number of mutations: {details['count']}")
    for mutation in details['locations']:
        print(f"  Position {mutation[0]}: Variant nucleotide = {mutation[1]}, Control nucleotide = {mutation[2]}")
print(len(control_sequence))

df_variants['Variant sequence'] = df_variants['Variant sequence'].str.replace("'", "")

# # Print the energy dictionary to verify
# df_variants['Total Energy'] = df_variants['Variant sequence'].apply(lambda seq: calculate_total_energy(seq, window_size=40))
# df_variants['Total Energy'] = df_variants['Variant sequence'].apply(lambda seq: calculate_total_energy(seq, window_size=None))

# comparison_full = energy_diff_compare_to_control(df_variants, control_sequence, window_size=None)



#Extracts CAI values from the given Excel file

# Usage
file_path = 'Train_data.xlsx'
sheet_name = 'Features'
cai_tai_df = extract_cai_tai_to_dataframe(file_path, sheet_name)

# Display the resulting DataFrame
print(cai_tai_df.head())

