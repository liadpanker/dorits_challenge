# calculate_dt.py

import pandas as pd

def create_luminescence_dict(df):
    luminescence_dict = {}
    for index, row in df.iterrows():
        variant_number = row['Variant number']
        luminescence_values = row.drop('Variant number').tolist()
        luminescence_dict[variant_number] = luminescence_values
    return luminescence_dict

def calculate_dt(df_without_DNT, df_with_DNT):
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
    return Dt, avg_dt
