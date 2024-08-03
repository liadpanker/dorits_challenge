# calculate_mutation.py
import pandas as pd

def calculate_mutations(variant_sequence, control_sequence):
    variant_sequence = variant_sequence.strip("'")
    control_sequence = control_sequence.strip("'")
    mutations = []
    for i, (var_nuc, ctrl_nuc) in enumerate(zip(variant_sequence, control_sequence)):
        if var_nuc != ctrl_nuc:
            mutations.append(i)  # Only store the position of the mutation
    mutation_count = len(mutations)
    return mutation_count, mutations

def create_feature_columns(df, features_col, features_threshold):
    # Initialize an empty DataFrame to store the features
    features_df = pd.DataFrame(index=df['Variant number'], columns=features_col)
    features_df = features_df.fillna(0)  # Initialize all counts to 0

    for index, row in df.iterrows():
        variant_number = row['Variant number']
        mutation_positions = row['Mutation Positions']
        for pos in mutation_positions:
            for i, threshold in enumerate(features_threshold):
                if pos <= threshold:
                    features_df.at[variant_number, features_col[i]] += 1
                    break

    return features_df

def process_and_generate_features(train_data_path, test_data_path):
    # Define feature columns and thresholds
    features_col = ["mut_0-170", "mut_171-178", "mut_179-220", "mut_221-224", "mut_225-229", "mut_230-232", "mut_233-242","mut_243_245","mut_246-279", "mut_280-287", "mut_288-312", "mut_313-321", "mut_322-336", "mut_337-354", "mut_355-356", "mut_357-364", "mut_365-371", "mut_372-381", "mut_382-383", "mut_384-393", "mut_394-399", "mut_400_461"]
    features_threshold = [170, 178, 220, 224, 229, 232, 242, 245, 279, 287, 312, 321, 336, 354, 356, 364, 371, 381, 383,393, 399, 460]

    # Load the Excel file
    xls = pd.ExcelFile(train_data_path)
    xls_test = pd.ExcelFile(test_data_path)

    # Load the specific sheets into DataFrames
    df_variants = pd.read_excel(xls, sheet_name='Variants data')
    df_variants_test = pd.read_excel(xls_test, sheet_name='Variants data')
    control_sequence = df_variants[df_variants['Variant number'] == 1]['Variant sequence'].values[0].strip("'")

    # Calculate mutations for each variant in train
    mutations_dict = {}
    for index, row in df_variants.iterrows():
        variant_number = row['Variant number']
        sequence = row['Variant sequence']
        mutation_count, mutations = calculate_mutations(sequence, control_sequence)
        mutations_dict[variant_number] = {'count': mutation_count, 'locations': mutations}

    # Calculate mutations for each variant in test
    mutations_dict_test = {}
    for index, row in df_variants_test.iterrows():
        variant_number = row['Variant number']
        sequence = row['Variant sequence']
        mutation_count, mutations = calculate_mutations(sequence, control_sequence)
        mutations_dict_test[variant_number] = {'count': mutation_count, 'locations': mutations}

    # Save mutation_count and mutations to an Excel file for train
    mutation_count_df = pd.DataFrame({
        'Variant number': list(mutations_dict.keys()),
        'Mutation Count': [details['count'] for details in mutations_dict.values()],
        'Mutation Positions': [details['locations'] for details in mutations_dict.values()]
    })

    # Create the feature columns for train data
    train_features_df = create_feature_columns(mutation_count_df, features_col, features_threshold)
    train_features_df.index.name = 'Variant number'

    # Save mutation_count and mutations to an Excel file for test
    mutation_count_df_test = pd.DataFrame({
        'Variant number': list(mutations_dict_test.keys()),
        'Mutation Count': [details['count'] for details in mutations_dict_test.values()],
        'Mutation Positions': [details['locations'] for details in mutations_dict_test.values()]
    })

    test_features_df = create_feature_columns(mutation_count_df_test, features_col, features_threshold)
    test_features_df.index.name = 'Variant number'

    return train_features_df, test_features_df
