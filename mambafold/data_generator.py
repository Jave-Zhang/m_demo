import numpy as np
import os
import _pickle as cPickle
import collections
from mambafold.utils import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch
from itertools import permutations, product
import pdb
from collections import defaultdict


perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

AUGC_BREAK_PRIOR = torch.tensor([
    # A   U   G   C  N
    [0, 2, 0, 0, 0], # A
    [2, 0, 0.8, 0, 0], # U
    [0, 0.8, 0, 3, 0], # G
    [0, 0, 3, 0, 0], # C
    [0, 0, 0, 0, 0]  # N
])

        
class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s' % self.split), 'rb') as f:
            self.data = cPickle.load(f,encoding='iso-8859-1')
        if self.upsampling:
            self.data = self.upsampling_data_new()
        print(self.data[:1])
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])

    def upsampling_data_new(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('_')[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index])
        final_d_list= list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300-d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                print('processing PDB seq...')
                index = np.random.choice(d.shape[0], d.shape[0]*4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def next_batch_SL(self, batch_size):
        p = Pool()
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[bp:bp + batch_size]
        data_seq = self.data_x[bp:bp + batch_size]
        data_pairs = self.pairs[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0
        contact = np.array(list(map(self.pairs2map, data_pairs)))
        matrix_rep = np.zeros(contact.shape)
        yield contact, data_seq, matrix_rep

    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

    def get_one_sample_long(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = np.nonzero(self.data_x[index].sum(axis=1))[0].max()
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq

    def get_one_sample_cdp(self, index):
        data_seq = self.data_x[index]
        data_label = self.data_y[index]

        return data_seq, data_label


class RNASSDataGenerator_input(object):
    def __init__(self,data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
                    'seq ss_label length name pairs')
        input_file = open(os.path.join(data_dir, '%s.txt' % self.split),'r').readlines()
        self.data_name = np.array([itm.strip()[1:] for itm in input_file if itm.startswith('>')])
        self.seq = [itm.strip().upper().replace('T','U') for itm in input_file if itm.upper().startswith(('A','U','C','G','T'))]
        self.len = len(self.seq)
        self.seq_length = np.array([len(item) for item in self.seq])
        self.data_x = np.array([self.one_hot_600(item) for item in self.seq])
        self.seq_max_len = 600
        self.data_y = self.data_x

    def one_hot_600(self,seq_item):
        RNN_seq = seq_item
        BASES = 'AUCG'
        bases = np.array([base for base in BASES])
        feat = np.concatenate(
                [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
                in RNN_seq])
        if len(seq_item) <= 600:
            one_hot_matrix_600 = np.zeros((600,4))
        else:
            one_hot_matrix_600 = np.zeros((600,4))
        one_hot_matrix_600[:len(seq_item),] = feat
        return one_hot_matrix_600

    def get_one_sample(self, index):
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_name = self.data_name[index]
        return data_seq, data_len, data_name


# Base dataset class
class Dataset(data.Dataset):
    """Base dataset class for RNA secondary structure prediction"""
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample(index)


class Dataset_Augmented(data.Dataset):
    """
    Dataset class for processing RNA data.
    Augments data_x with rows based on AUGC_BREAK_PRIOR matrix.
    """
    def __init__(self, data_list):
        """
        Initialize the dataset.

        Parameters:
            data_list: List of RNA data generator objects.
        """
        'Initialization'
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        """
        Return total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.data.len

    def __getitem__(self, index):
        """
        Get a single sample with augmented data.

        Parameters:
            index (int): Sample index.

        Returns:
            tuple: Tuple containing augmented data, labels, sequence length, etc.
        """
        # Get original data
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        cons_mat = np.zeros((1,l,l))
        cons_mat[0,:data_len,:data_len] = constraint_adjacency_matrix(data_seq[:data_len,])
        return contact[:l, :l],data_fcn_1,cons_mat, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Augmented1(data.Dataset):
    """
    Dataset class for processing RNA data.
    Simplified version of Dataset_Augmented.
    """
    def __init__(self, data_list):
        """
        Initialize the dataset.

        Parameters:
            data_list: List of RNA data generator objects.
        """
        'Initialization'
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        """
        Return total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.data.len

    def __getitem__(self, index):
        """
        Get a single sample with basic processing.

        Parameters:
            index (int): Sample index.

        Returns:
            tuple: Tuple containing processed data.
        """
        # Get original data
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        return contact[:l, :l], matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Augmented_test(data.Dataset):
    """
    Dataset class for testing RNA data processing.
    Generates additional features for evaluation.
    """
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def __getitem__(self, index):
        """
        Get a single sample with test-specific processing.

        Parameters:
            index (int): Sample index.

        Returns:
            tuple: Tuple containing processed data with additional features.
        """
        # Get original data
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        data_nc = np.zeros((10, l, l))
        feature = np.zeros((8,l,l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(bool)
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        cons_mat = np.zeros((1,l,l))
        cons_mat[0,:data_len,:data_len] = constraint_adjacency_matrix(data_seq[:data_len,])
        data_fcn_2 = np.concatenate((cons_mat,data_fcn_1),axis=0) 
        return contact[:l, :l], cons_mat,data_fcn_1,matrix_rep, data_len, data_seq[:l], data_name, data_nc,l


class Dataset_Augmented_test1(data.Dataset):
    """
    Simplified test dataset for RNA data.
    """
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def __getitem__(self, index):
        """
        Get a single sample with minimal processing for testing.

        Parameters:
            index (int): Sample index.

        Returns:
            tuple: Tuple containing essential data for evaluation.
        """
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len,80)
        data_nc = np.zeros((10, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(bool)
        return contact[:l, :l],matrix_rep, data_len, data_seq[:l], data_name, data_nc,l


class Dataset_cdp(data.Dataset):
    """Dataset class for CDP processing"""
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample_cdp(index)


class LongRNASSDataset(data.Dataset):
    """Dataset class for handling long RNA sequences"""
    def __init__(self, data, max_length=2400, device=None):
        self.data = data
        self.max_length = max_length
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.data.len

    def __getitem__(self, index):
        # Get original data
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        
        # Process long sequences
        l = self.get_effective_length(data_len)
        data_fcn = self.compute_features(data_seq, l)
        
        # Process contact matrix
        contact_adj = np.zeros((l, l))
        contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        
        cons_mat = np.zeros((1,l,l))
        cons_mat[0,:data_len,:data_len] = constraint_adjacency_matrix(data_seq[:data_len,])
        
        # Return processed data
        return contact_adj, cons_mat,data_fcn_1 , matrix_rep, data_len, data_seq[:l], data_name

    def get_effective_length(self, data_len):
        # Calculate effective length, ensuring it's a multiple of 16
        if data_len > self.max_length:
            return (((self.max_length - 1) // 16) + 1) * 16
        else:
            return (((data_len - 1) // 16) + 1) * 16

    def compute_features(self, data_seq, l):
        # Compute feature matrices
        data_fcn = np.zeros((16, l, l))
        for n, (i, j) in enumerate(product(range(4), range(4))):
            data_fcn[n, :l, :l] = np.matmul(
                data_seq[:l, i].reshape(-1, 1),
                data_seq[:l, j].reshape(1, -1)
            )
        return data_fcn


def get_cut_len(data_len, set_len):
    """
    Calculate appropriate length for processing, ensuring it's a multiple of 16.
    
    Parameters:
        data_len (int): Original sequence length
        set_len (int): Minimum length to consider
        
    Returns:
        int: Adjusted length (multiple of 16 or set_len if sequence is short)
    """
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


def z_mask(seq_len):
    """
    Create an upper triangular mask starting from diagonal+2.
    
    Parameters:
        seq_len (int): Length of sequence
        
    Returns:
        numpy.ndarray: Upper triangular mask
    """
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)


def l_mask(inp, seq_len):
    """
    Create a mask that excludes positions with -1 values in input.
    
    Parameters:
        inp: Input data
        seq_len (int): Length of sequence
        
    Returns:
        numpy.ndarray: Masked matrix
    """
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)


def creatmat(data, device=None):
    """
    Create base pairing probability matrix.
    
    Parameters:
        data: RNA sequence in one-hot encoding format
        device: Computation device (CPU or GPU)
        
    Returns:
        numpy.ndarray: Base pairing probability matrix
    """
    if device==None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        # Convert one-hot encoding to nucleotide sequence
        sequence = []
        for d in data:
            try:
                # Find index with value 1 and map to corresponding nucleotide
                nucleotide = 'AUCG'[list(d).index(1)]
                sequence.append(nucleotide)
            except ValueError:
                # If no 1 in vector, add default value 'N'
                sequence.append('N')
        data = ''.join(sequence)
        paired = defaultdict(int, {'AU':2, 'UA':2, 'GC':3, 'CG':3, 'UG':0.8, 'GU':0.8})

        mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
        n = len(data)

        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing=None)
        t = torch.arange(30).to(device)
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
        m1 = m1.float()
        m1 *= torch.exp(-0.5*t*t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
        to0indices = t[None,None,:]>first0[:,:,None]
        m1[to0indices] = 0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
        t = t.float()
        m2 = m2.float()
        m2 *= torch.exp(-0.5*t*t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
        m2[to0indices] = 0
        m2 = m2.sum(dim=2)
        m2[m1==0] = 0

        return (m1+m2).to(torch.device('cpu'))


def constraint_adjacency_matrix(one_hot_sequence):
    """
    Generate constraint adjacency matrix based on base pairing rules.
    
    Parameters:
        one_hot_sequence (numpy.ndarray): One-hot encoded sequence matrix, shape (L, 4)
                                          where L is sequence length.
    
    Returns:
        numpy.ndarray: Adjacency matrix of shape (L, L) indicating which positions
                       can potentially form base pairs.
    """
    L = one_hot_sequence.shape[0]  # Sequence length
    adj_matrix = np.zeros((L, L), dtype=int)  # Initialize adjacency matrix

    # Define one-hot encoding for bases
    A = np.array([1, 0, 0, 0])
    U = np.array([0, 1, 0, 0])
    C = np.array([0, 0, 1, 0])
    G = np.array([0, 0, 0, 1])
    N = np.array([0, 0, 0, 0])

    # Define base pairing rules
    pairing_rules = {
        'A': {'U'},
        'U': {'A', 'G'},
        'G': {'C', 'U'},
        'C': {'G'}
    }

    # Convert one-hot encoding to base character
    def one_hot_to_base(one_hot):
        if np.array_equal(one_hot, A):
            return 'A'
        elif np.array_equal(one_hot, U):
            return 'U'
        elif np.array_equal(one_hot, C):
            return 'C'
        elif np.array_equal(one_hot, G):
            return 'G'
        elif np.array_equal(one_hot, N):
            return 'N'
        else:
            raise ValueError("Invalid one-hot encoding.")

    # Check all possible site pairs
    for i in range(L):
        for j in range(i + 1, L):
            # Check if distance between sites is greater than 4
            if abs(i - j) > 4:
                # Get bases at current positions
                base_i = one_hot_to_base(one_hot_sequence[i])
                base_j = one_hot_to_base(one_hot_sequence[j])

                # Check if they satisfy pairing rules
                if base_j in pairing_rules.get(base_i, set()):
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Symmetric matrix

    return adj_matrix

