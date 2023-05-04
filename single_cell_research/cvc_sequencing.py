import sys
import pandas as pd
import tqdm
import cvc.featurization as ft
import torch
import random

# covert fasta file into as pandas dataframe with barcodes and sequences
def fasta_to_df(file_path):
    # read in the fasta file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # remove the new line characters
    # lines = [line.strip() for line in lines]
    # get the barcodes and sequences
    barcodes = []
    sequences = []
    for i, line in enumerate(lines):
        # the barcode are the lines that start with '>'
        if line.startswith('>'):
            barcodes.append(line[1:])
            # the sequence is the next line
            sequences.append(lines[i+1])
        else:
            sequences.append(line)
    # create a dataframe with the barcodes and sequences
    df = pd.DataFrame({'barcode': barcodes, 'sequence': sequences})
    return df

# read csv file that contains the barcodes
def read_barcodes(file_path):
    # read csv file
    df = pd.read_csv(file_path)
    # creat column with barcode names without '-1' at the end
    df['barcode_name'] = df['barcode'].str[:-2]
    # make list of barcode names
    barcode_list = df['barcode_name'].tolist()
    # make df with barcode_name, chain and cdr3 columns
    df_labels = df[['barcode_name', 'chain', 'cdr3']]
    return barcode_list, df_labels

# extract sequences from fasta file and take only the sequences that the first 16 nucleotides match the barcodes and take
# the corresponding sequence from the second fasta file
def extract_sequences(file_path1, file_path2, df_labels):
    # read fasta file
    file1 = open(file_path1, "r")
    # read fasta file
    file2 = open(file_path2, "r")
    # create dataframe with the barcodes and the corresponding sequences
    df = pd.DataFrame(columns=['barcode', 'sequence'])
    barcode_list = df_labels['barcode_name'].tolist()
    barcode_sequence_dict = {}
    # parse both fasta files at the same time and show progress b
    for line1, line2 in tqdm.tqdm(zip(file1, file2)):
        # if line1 does not start with >
        if line1[0] != ">":
            # check if the first 16 nucleotides match the barcodes
            if line1[:16] in barcode_list:
                # add the barcode and the corresponding sequence to the dictionary
                # if the barcode is already in the dictionary, add the sequence to the list of sequences
                if line1[:16] in barcode_sequence_dict:
                    barcode_sequence_dict[line1[:16]].append(line2)
                else:
                    barcode_sequence_dict[line1[:16]] = [line2]

    # add the barcodes and the corresponding sequences to the dataframe
    for key, value in barcode_sequence_dict.items():
        df = df.append({'barcode': key, 'sequence': value}, ignore_index=True)
    file1.close()
    file2.close()
    return df

# convert nucleotide sequence to amino acid sequence
def nucleotide_to_aa(seq):
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'.', 'TAG':'.',
        'TGC':'C', 'TGT':'C', 'TGA':'.', 'TGG':'W',
    }
    protein =""
    # if seq is not multiple of 3, pad with * at the end
    if len(seq)%3 != 0:
        seq = seq + "*"*(3-len(seq)%3)
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        # if codon is not in table, replace with _
        if codon not in table:
            protein += "."
        else:
            protein+= table[codon]
    return protein


# read csv file nad split sequence column (if its a list of multiple sequences per barcode)
def split_df(file_path):
    # read csv file that contains the barcodes and the corresponding sequences
    df = pd.read_csv(file_path)
    # if in the sequence column there are multiple sequences per barcode, split the column into multiple rows
    df = df.set_index(['barcode']).sequence.str.split(',', expand=True).stack().reset_index().rename(columns={0:'sequence'})
    # remove \\n from the end of the sequence
    df['sequence'] = df['sequence'].str.replace(r'\\n', '')
    df['sequence'] = df['sequence'].apply(lambda x : x[2:-2])
    return df

def add_kmers(df):
    for offset in [0,1,2]:
        # for each nucleotide sequence in the 'sequence' column, convert to amino acid sequence
        df[f'aa_sequence_offset_{offset}'] = df['sequence'].apply(lambda seq: nucleotide_to_aa(seq[offset:]))

        cut_sizes_list = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        # for each of the cut sizes in the list above create a column with the corresponding kmers of the amino acid sequence
        for cut_size in cut_sizes_list:
            df[f'kmer_{cut_size}_offset_{offset}'] = df[f'aa_sequence_offset_{offset}'].apply(lambda x: [x[i:i+cut_size] for i in range(len(x) - cut_size + 1)])

        # list of kmer lists in df
        df[f'kmer_list_offset_{offset}'] = df[[f'kmer_{cut_size}_offset_{offset}'
                                               for cut_size in cut_sizes_list]].values.tolist()

    return df


def calc_pseudo_perplexity(model, tokenizer, seq, device):
    # Tokenize input
    tokens = tokenizer(ft.insert_whitespace(seq), return_tensors="pt").input_ids.to(device)
    N = len(seq)
    mask_token = tokenizer.mask_token_id

    # Create a matrix of masked 1 out for each token
    # transform tokens from (1, N) to (N, N)
    tokens_mat = tokens.repeat(N, 1)
    # place mask token in each row
    tokens_mat[:, 1:-1][torch.arange(N), torch.arange(N)] = mask_token

    # Get the logits for each token
    with torch.no_grad():
        logits = model(tokens_mat).logits # [N, N+2, token_vocab_size]

    # Get the log probabilities for each token
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # now calc pseudo perplexity
    log_probs_at_mask = log_probs[:, 1:-1, :][torch.arange(N), torch.arange(N)]

    # Get the real log prob for each token
    one_hot = torch.nn.functional.one_hot(tokens[0, 1:-1], num_classes=log_probs_at_mask.shape[-1])

    # Get cross entropy loss
    loss = -torch.sum(one_hot * log_probs_at_mask, dim=-1)
    loss = loss.mean()
    ppl = torch.exp(loss)
    return ppl.item()


SEP='|'
def create_tcr_seqs(tcr_seqs_df, max_len=120, column_to_concat='cdr3', barcodes_column='barcode_unique'):
    tcr_seqs = []
    tcr_seqs_df = tcr_seqs_df.sort_values(by=[barcodes_column])
    barcodes = tcr_seqs_df[barcodes_column].values
    tcrs = tcr_seqs_df[column_to_concat].values
    start_idx = 0
    for i in tqdm.tqdm(range(1, len(barcodes))):
        if barcodes[i] != barcodes[i - 1]:
            end_idx = i
            tcr = SEP.join(tcrs[start_idx:end_idx])
            if len(tcr) <= max_len:
                tcr_seqs.append(tcr)
            start_idx = i
    end_idx = len(barcodes)
    tcr = SEP.join(tcrs[start_idx:end_idx])
    if len(tcr) <= max_len:
        tcr_seqs.append(tcr)
    return tcr_seqs


def concatenate_labels(tcr_seqs_df, max_len=120, label_column='MAIT_cell', barcodes_column='barcode'):
    # concatenated TCR sequences
    tcr_seqs = create_tcr_seqs(tcr_seqs_df, max_len, column_to_concat='cdr3', barcodes_column=barcodes_column)
    # concatenated labels
    label_seqs = create_tcr_seqs(tcr_seqs_df, max_len, column_to_concat=label_column, barcodes_column=barcodes_column)
    # Initialize lists to store the concatenated TCR sequences and their corresponding labels
    concat_tcrs = []
    labels = []
    # Iterate through the TCR sequences and label sequences
    for tcr, label_seq in zip(tcr_seqs, label_seqs):
        # Check if "MAIT cell" is in the label_seq
        if "MAIT_cell" in label_seq.split(SEP):
            label = "MAIT_cell"
        else:
            label = "non-MAIT_cell"
        # Append the TCR sequence and its corresponding label
        concat_tcrs.append(tcr)
        labels.append(label)
    # Create a dataframe with the concatenated TCR sequences and their labels
    df = pd.DataFrame({'cdr3': concat_tcrs, label_column: labels})
    return df

def random_concat_tcr_seqs(tcr_seqs_df, max_len=120, column_to_concat='cdr3'):
    tcr_seqs = []
    barcodes = tcr_seqs_df['barcode_unique'].unique()
    for i in range(0, len(barcodes), 2):
        if i + 1 >= len(barcodes):
            break
        barcode1 = barcodes[i]
        barcode2 = barcodes[i+1]
        tcrs1 = tcr_seqs_df[tcr_seqs_df['barcode_unique'] == barcode1][column_to_concat].values
        tcrs2 = tcr_seqs_df[tcr_seqs_df['barcode_unique'] == barcode2][column_to_concat].values
        if len(tcrs1) > 0 and len(tcrs2) > 0:
            if len(tcrs2) > 1:
                tcr = SEP.join([tcrs1[0], tcrs2[1]])
            else:
                tcr = SEP.join([tcrs1[0], tcrs2[0]])
            if len(tcr) <= max_len:
                tcr_seqs.append(tcr)
    # create dataframe
    df = pd.DataFrame({
        'barcode1': barcodes[::2][:(len(tcr_seqs))],
        'barcode2': barcodes[1::2][:(len(tcr_seqs))],
        'concatenated_seq': tcr_seqs
    })
    return df


def main():
    file_path1 = sys.argv[1] # fasta file with the sequences
    file_path2 = sys.argv[2] # fasta file with the barcodes
    file_path3 = sys.argv[3] # csv file with the barcodes
    output_file_name = sys.argv[4]
    barcode_list, df_labels = read_barcodes(file_path3)
    df = extract_sequences(file_path1, file_path2, df_labels)
    # create a csv file with the barcodes and the corresponding sequences
    df.to_csv(output_file_name, index=False)
    # export labels to csv file
    label_file_name = output_file_name.split('.')[0] + '_labels.csv'
    df_labels.to_csv(label_file_name, index=False)

if __name__ == "__main__":
    main()



