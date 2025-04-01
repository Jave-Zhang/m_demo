import _pickle as pickle
import sys
import os
import torch
import torch.optim as optim
from torch.utils import data

# from Network import GRmambafold as MambaFold
# from Network import TransformerFold as MambaFold
# from Network import MambaFold_DynamicDP as MambaFold
from Network import MambaFold
# from Network import S4FoldEfficient as MambaFold
from mambafold.utils import *
from mambafold.config import process_config

import time
from mambafold.data_generator import RNASSDataGenerator

from mambafold.data_generator import Dataset_Augmented_test1 as Dataset_energy
import collections

args = get_args()
if args.nc:
    from mambafold.postprocess import postprocess_new_nc as postprocess
else:
    from mambafold.postprocess import postprocess_new as postprocess




def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation

def check_pseudoknot(connection_matrix: torch.Tensor) -> bool:
    """
    判断连接矩阵中是否存在伪结。
    计算方法:
    1. 统计所有连接的前后位置, 利用上三角矩阵
    2. 按照前位置排序, 得到排序后的前位置和后位置
    3. 计算后位置的差分, 如果存在大于0的差分, 则存在伪结。
    
    参数:
        connection_matrix (np.ndarray): 一个二维数组，表示RNA的二级结构连接关系。
    
    返回:
        bool: 如果存在伪结，返回True；否则返回False。
    """
    contacts = connection_matrix.triu(1).nonzero()
    master_sort = contacts[:, 0].sort().indices
    servant_sort = contacts[master_sort, 1]
    diff = servant_sort.diff()
    condition = (diff > 0).any()
    return condition


def mask_of_long_range_contacts(contacts: torch.Tensor, long_range_ratio=0.5, postive_mask=True):
    """
    获取长程连接的mask。
    计算方法:
    直接选取上三角矩阵的右上角的一部分。
    """
    seq_len = contacts.size(-1)
    long_range_len = int(seq_len * long_range_ratio)
    mask = contacts.triu(seq_len - long_range_len) > 0
    if not postive_mask:
        mask = ~mask
    return mask




def model_eval_all_test(contact_net,test_generator):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    
    # 记录评估开始时的显存使用情况
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f'初始显存占用: {start_mem:.2f} MB')
    
    for contacts, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_ori_batch= torch.Tensor(seq_ori.float()).to(device)
        # cons_mat_batch = torch.Tensor(cons_mat.float()).to(device)
        # feature_batch = torch.Tensor(feature.float()).to(device)
        nc_map_nc = nc_map.float() * contacts
        if seq_lens.item() > 1500:
            continue
        if not check_pseudoknot(connection_matrix=contacts_batch[0]):
            continue
        if batch_n%1000==0:
            print('Batch number: ', batch_n)
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                print(f'当前显存占用: {current_mem:.2f} MB, 峰值显存: {max_mem:.2f} MB')
        
        batch_n += 1
        
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())

        tik = time.time()
        
        with torch.no_grad():
            pred_contacts = contact_net(seq_ori_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        
        run_time.append(t0)
        
        # long_range_mask = mask_of_long_range_contacts(contacts_batch[0])
        # contacts_batch[0] = contacts_batch[0] * long_range_mask
        # map_no_train = map_no_train * long_range_mask        
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        
        if nc_map_nc.sum() != 0:
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                nc_map_nc.cpu().float()[i]), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    
    # 显示最终的显存统计
    if torch.cuda.is_available():
        final_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print('显存统计:')
        print(f'- 最终显存占用: {final_mem:.2f} MB')
        print(f'- 峰值显存占用: {peak_mem:.2f} MB')
    
    print('Average time costing: ', np.average(run_time))
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


def report_gpu_memory(location="未指定"):
    """报告GPU显存使用情况的详细信息"""
    if torch.cuda.is_available():
        # 当前分配的显存
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # 当前缓存的显存
        cached_memory = torch.cuda.memory_reserved() / (1024 * 1024)
        # 峰值分配显存
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        # 峰值缓存显存
        peak_cached = torch.cuda.max_memory_reserved() / (1024 * 1024)
        
        # 获取显卡信息
        device_name = torch.cuda.get_device_name(1)
        device_properties = torch.cuda.get_device_properties(1)
        total_memory = device_properties.total_memory / (1024 * 1024)
        
        print(f"\n========== GPU内存使用报告 ({location}) ==========")
        print(f"设备: {device_name}")
        print(f"显存总量: {total_memory:.2f} MB")
        print(f"当前分配显存: {current_memory:.2f} MB ({current_memory/total_memory*100:.2f}%)")
        print(f"当前缓存显存: {cached_memory:.2f} MB ({cached_memory/total_memory*100:.2f}%)")
        print(f"峰值分配显存: {peak_memory:.2f} MB ({peak_memory/total_memory*100:.2f}%)")
        print(f"峰值缓存显存: {peak_cached:.2f} MB ({peak_cached/total_memory*100:.2f}%)")
        print("=================================================\n")
        
        return {
            "current_memory": current_memory,
            "cached_memory": cached_memory,
            "peak_memory": peak_memory,
            "peak_cached": peak_cached,
            "total_memory": total_memory
        }
    else:
        print("CUDA不可用，无法获取GPU内存信息")
        return None

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(1)
    
    # 初始化显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        report_gpu_memory("程序初始化")
    
    config_file = args.config
    test_file = args.test_files
    
    config = process_config(config_file)
    # MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold-bak/models_trans/bpRNA-1m-mambaFold/mamba_fold_dp_bpRNA-1m-mambaFold_3_0.17960700392723083.pt'
    # MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold-bak/models_gru/bpRNA-1m-mambaFold/mamba_fold_gru_bpRNA-1m-mambaFold_93_0.05006871372461319.pt'
    # MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold/models/lncRNA/mamba_fold_lncRNA_73_0.008355999334860012.pt'
    MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold/models/crossfamily_merged_am1/mamba_fold_crossfamily_merged_am1_9_0.09464814988072931.pt'
    # MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold-bak/models_s4/bpRNA-1m-mambaFold/s4_fold_bpRNA-1m-mambaFold_3_0.13126571476459503.pt'
    # MODEL_SAVED = '/home/j611/Project/jkj/python_code/mambafold/models/crossfamily_merged_am/mamba_fold_crossfamily_merged_am_75_0.06288257578560054.pt'
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    epoches_first = config.epoches_first
        
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    seed_torch()
    
    # if test_file == 'RNAStralign' or test_file == 'ArchiveII':
    #     test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    # else:
    #     test_data = RNASSDataGenerator('datanew/',test_file+'.cPickle')
    # train_data_list = []
    test_data = RNASSDataGenerator('/home/j611/Project/jkj/python_code/mambafold/data','bpRNAnew_dataset.cPickle')
    # train_data_list.append(data_input)

    seq_len = test_data.data_y.shape[-2]
    print('Max seq length ', seq_len)
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    test_set = Dataset_energy(test_data)
    test_generator = data.DataLoader(test_set, **params)
            
    contact_net = MambaFold()
    
    report_gpu_memory("模型创建后")
    
    print('==========Start Loading==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:1'))
    
    report_gpu_memory("模型加载后")
    
    print('==========Finish Loading==========')
    
    contact_net.to(device)
    
    report_gpu_memory("模型迁移到GPU后")
    
    model_eval_all_test(contact_net,test_generator)
    
    # 程序结束前报告最终显存使用情况
    report_gpu_memory("程序结束")

import multiprocessing
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()






