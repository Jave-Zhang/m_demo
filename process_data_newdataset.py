import numpy as np
import os
import subprocess
import collections
import pickle as cPickle
import random
import sys

def one_hot(seq):
    """
    Convert RNA sequence to one-hot encoding
    
    Args:
        seq (str): RNA sequence
        
    Returns:
        numpy.ndarray: One-hot encoded sequence
    """
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
            [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
            in RNN_seq])

    return feat

def clean_pair(pair_list, seq):
    """
    Clean the base pair list by removing invalid pairs
    
    Args:
        pair_list (list): List of base pairs
        seq (str): RNA sequence
        
    Returns:
        list: Cleaned list of base pairs
    """
    valid_pairs = []
    for item in pair_list:
        if seq[item[0]] == 'A' and seq[item[1]] == 'U':
            valid_pairs.append(item)
        elif seq[item[0]] == 'C' and seq[item[1]] == 'G':
            valid_pairs.append(item)
        elif seq[item[0]] == 'U' and seq[item[1]] == 'A':
            valid_pairs.append(item)
        elif seq[item[0]] == 'G' and seq[item[1]] == 'C':
            valid_pairs.append(item)
        else:
            print('%s+%s' % (seq[item[0]], seq[item[1]]))
    
    return valid_pairs

if __name__=='__main__':
    # Set input data directory and output paths
    data_dir = '/home/j611/dataset/RNAStructureData/bpRNAnew_dataset/bpRNAnew.nr500.canonicals/'
    output_dir = "/home/j611/dataset/lc/UFold-main/UFold-main/data/"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(4)
    
    # Get list of all files in the directory
    all_files = os.listdir(data_dir)
    random.shuffle(all_files)
    
    # Define named tuple for RNA structure data
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    
    all_files_list = []
    skip_count = 0
    
    # Process each file
    for index, item_file in enumerate(all_files):
        # Extract sequence
        t0 = subprocess.getstatusoutput(f'awk \'{{print $2}}\' {data_dir}{item_file}')
        if t0[0] == 0:
            seq = ''.join(t0[1].split('\n'))
            try:
                one_hot_matrix = one_hot(seq.upper())
            except:
                continue
        
        # Extract indices
        t1 = subprocess.getstatusoutput(f'awk \'{{print $1}}\' {data_dir}{item_file}')
        t2 = subprocess.getstatusoutput(f'awk \'{{print $3}}\' {data_dir}{item_file}')
        
        # Create pair list
        if t1[0] == 0 and t2[0] == 0:
            pair_dict_all_list = [[int(item_tmp)-1, int(t2[1].split('\n')[index_tmp])-1] 
                                  for index_tmp, item_tmp in enumerate(t1[1].split('\n')) 
                                  if int(t2[1].split('\n')[index_tmp]) != 0]
        else:
            pair_dict_all_list = []
        
        seq_name = item_file
        seq_len = len(seq)
        
        # Create pair dictionary (only pairs where i < j)
        pair_dict_all = dict([item for item in pair_dict_all_list if item[0] < item[1]])
        
        # Log progress
        if index % 1000 == 0:
            print('Current processing %d/%d' % (index+1, len(all_files)))
        
        # Process sequences of valid length (up to 600)
        if seq_len > 0 and seq_len <= 600:
            # Create secondary structure label
            ss_label = np.zeros((seq_len, 3), dtype=int)
            ss_label[[*pair_dict_all.keys()], ] = [0, 1, 0]
            ss_label[[*pair_dict_all.values()], ] = [0, 0, 1]
            ss_label[np.where(np.sum(ss_label, axis=1) <= 0)[0], ] = [1, 0, 0]
            
            # Pad sequence to fixed length (600)
            one_hot_matrix_600 = np.zeros((600, 4))
            one_hot_matrix_600[:seq_len, ] = one_hot_matrix
            ss_label_600 = np.zeros((600, 3), dtype=int)
            ss_label_600[:seq_len, ] = ss_label
            ss_label_600[np.where(np.sum(ss_label_600, axis=1) <= 0)[0], ] = [1, 0, 0]
            
            # Create data sample
            sample_tmp = RNA_SS_data(seq=one_hot_matrix_600, 
                                     ss_label=ss_label_600, 
                                     length=seq_len, 
                                     name=seq_name, 
                                     pairs=pair_dict_all_list)
            all_files_list.append(sample_tmp)

    print(f"Total processed samples: {len(all_files_list)}")
    
    # Save processed data
    output_file = os.path.join(output_dir, "bpRNA_1m.cPickle")
    cPickle.dump(all_files_list, open(output_file, "wb"))
    print(f"Data saved to {output_file}")
