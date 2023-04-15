import sys
import tqdm
import pandas as pd

# read csv file that contains the barcodes
def read_barcodes(file_path):
    # read csv file
    df = pd.read_csv(file_path)
    # creat column with barcode names without '-1' at the end
    df['barcode_name'] = df['barcode'].str[:-2]
    # make list of barcode names
    barcode_list = df['barcode_name'].tolist()
    df_labels = df[['barcode_name','chain', 'cdr3']]
    return df_labels

# extract sequences from fasta file, take only the sequences that the first 16 nucleotides match the barcodes and take
# the corresponding sequence from the second fasta file
from collections import defaultdict
def extract_sequences(file_path1, file_path2, df_labels):
    # read fasta file
    file1 = open(file_path1, "r")
    # read fasta file
    file2 = open(file_path2, "r")
    # create dataframe with the barcodes and the corresponding sequences
    df = pd.DataFrame(columns=['barcode', 'sequence'])
    df_labels_output = pd.DataFrame(columns=['barcode', 'cdr3'])
    # create a new df from df_labels with only 'TRB' chain and cdr3 is not None
    df_labels_TRB = df_labels[df_labels['chain'] == 'TRB']
    df_labels_TRB = df_labels_TRB[df_labels_TRB['cdr3'] != 'None']

    barcode_list = df_labels[df_labels['chain'] == 'TRB']['barcode_name'].tolist()
    # barcode_list = df_labels['barcode_name'].tolist()
    barcode_sequence_dict = defaultdict(list)
    label_dict = defaultdict(list)
    # parse both fasta files at the same time and show progress b
    for line1, line2 in tqdm.tqdm(zip(file1, file2)):
        # if line1 does not start with >
        if line1[0] != ">":
            line1_barcode = line1[:16]
            # check if the first 16 nucleotides match the barcodes
            if line1_barcode in barcode_list:
                # add the barcode and the corresponding sequence to the dictionary
                barcode_sequence_dict[line1_barcode].append(line2)
                # add the barcode and the corresponding cdr3 label to the dictionary
                label_dict[line1_barcode].append(df_labels_TRB[df_labels_TRB['barcode_name'] == line1_barcode]['cdr3'])

    # add the barcodes and the corresponding sequences to the dataframe
    for key, value in barcode_sequence_dict.items():
        df = df.append({'barcode': key, 'sequence': value}, ignore_index=True)
    for key, value in label_dict.items():
        df_labels_output = df_labels_output.append({'barcode': key, 'cdr3': value}, ignore_index=True)

    print("len(barcode_sequence_dict): ", len(barcode_sequence_dict))
    print("len(label_dict): ", len(label_dict))

    file1.close()
    file2.close()
    return df, df_labels

def main():
    file_path1 = sys.argv[1] # fasta file with the barcodes
    file_path2 = sys.argv[2] # fasta file with the sequences
    file_path3 = sys.argv[3] # csv file with the barcodes
    output_file_name = sys.argv[4]
    output_file_name_labels = sys.argv[5]
    df_labels = read_barcodes(file_path3)
    df, df_labels = extract_sequences(file_path1, file_path2, df_labels)
    print("done")
    # csv file with the barcodes and the corresponding sequences
    df.to_csv(output_file_name)
    df_labels.to_csv(output_file_name_labels)
    

if __name__ == "__main__":
    main()
