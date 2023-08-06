import os
import pandas as pd

def process_diabetes_metadata(directory):
    # Read the CSV file
    samples = pd.read_csv(directory)
    # drop 'Adaptive ID' that are NaN
    samples = samples.dropna(subset=['Adaptive ID'])
    # make sure that if the 'Adaptive ID' is a number, it wil be a whole number
    samples['Adaptive ID'] = samples['Adaptive ID'].astype(str).str.replace('.0', '')
    # Create a 'Patient ID' column
    samples['Patient ID'] = samples.index // 4
    # Group by 'Patient ID' and join the 'Adaptive ID' values within each group
    samples['Joined Adaptive ID'] = samples.groupby('Patient ID')['Adaptive ID'].transform(lambda x: '_'.join(map(str, x)))
    # Pivot the DataFrame
    samples_pivot = samples.pivot(index='Patient ID', columns='Sample Timepoint')
    # Flatten the multi-level column index into a single level
    samples_pivot.columns = [' '.join(str(col) for col in cols) for cols in samples_pivot.columns]
    # Rename the joined ID's column and drop the old ones
    samples_pivot['Joined Adaptive ID'] = samples_pivot['Joined Adaptive ID 1.0']
    samples_pivot = samples_pivot.drop(columns=[col for col in samples_pivot.columns if 'Joined Adaptive ID' in col and col != 'Joined Adaptive ID'])
    # Drop all "Unnamed" columns
    samples_pivot = samples_pivot.loc[:, ~samples_pivot.columns.str.contains('^Unnamed')]
    return samples_pivot

# create a patient data dictionary where the key is the patient ID and the value is a list of dataframes (samples)
def join_data(df, DATA_FOLDER_DIR):
    column_to_use = ['sample_name', 'locus', 'sample_catalog_tags', 'sample_tags',
                     'v_family', 'v_gene', 'j_family', 'j_gene', 'bio_identity', 'cdr3_amino_acid']
    patient_data = {}
    for index, row in df.iterrows():
        file_names = [f"{ID}_TCRB.tsv" for ID in row['Joined Adaptive ID'].split('_')]
        dataframes = []
        for file_name in file_names:
            file_path = os.path.join(DATA_FOLDER_DIR, file_name)
            if os.path.exists(file_path):
                df_file = pd.read_csv(file_path, sep='\t', usecols=column_to_use)
                df_file = df_file.dropna(subset=['cdr3_amino_acid'])
                # drop duplicate CD3R sequences
                df_file = df_file.drop_duplicates(subset=['cdr3_amino_acid'])
                df_file['Patient ID'] = index
                dataframes.append(df_file)
        if dataframes:
            patient_data[index] = dataframes
    return patient_data


# add metadata to the dataframe
def add_patient_metadata(patient_data, metadata_df):
    for patient_id, df in patient_data.items():
        metadata_row = metadata_df[metadata_df['Patient ID'] == patient_id]
        for column in metadata_row.columns:
            df[column] = metadata_row.iloc[0][column]
    return patient_data

# divide sample_catalog_tags column into sub column
def parse_tags(row):
    tags_dict = {}
    tags = row.split(',')
    for tag in tags:
        key, value = tag.split(':')
        tags_dict[key] = value
    return pd.Series(tags_dict)

from tqdm import tqdm
def apply_parse_tags(cases_dataframes_dict):
    for patient_id in tqdm(cases_dataframes_dict.keys()):
        dataframes_list = cases_dataframes_dict[patient_id]
        for i, df in enumerate(dataframes_list):
            dataframes_list[i] = df.join(df['sample_catalog_tags'].apply(du.parse_tags))
            # change cdr3_amino_acid column name to Sequences
            dataframes_list[i].rename(columns={'cdr3_amino_acid': 'Sequences'}, inplace=True)
            # change j_gene column to be all strings
            dataframes_list[i]['j_gene'] = dataframes_list[i]['j_gene'].astype(str)
    return cases_dataframes_dict


