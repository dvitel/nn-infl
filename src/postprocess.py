from collections import defaultdict
from functools import partial
import json
import os
from matplotlib.ticker import MultipleLocator
import pandas as pd
import re
from typing import Optional
from tabulate import tabulate

import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy import stats
import torch
from datasets import load_from_disk
from scipy import stats as sci_stats
from sklearn.preprocessing import StandardScaler

# from pyclustering.cluster.xmeans import xmeans
from sklearn.metrics import silhouette_samples
# import warnings
# np.warnings = warnings

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    

benchmark = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

from cifar import DatasetSplits

def draw_mislabel_detection_rate(task="mrpc", res_folder = "./data/", datasets_folder = "./data/",  
                                    module_pattern='', out = "./data/mdr/qnli.png", title = "All modules"):
    # res_pattern = re.compile(res_filer)
    res_path = os.path.join(res_folder, task)
    all_influences = defaultdict(list)
    if module_pattern == '':
        ptrn = None 
    else:
        ptrn = re.compile(module_pattern)
    seeds = defaultdict(list)
    for file_name in os.listdir(res_path):
        name_parts = file_name.split('.')[0].split('_')
        method_name = name_parts[1]
        seed = int(name_parts[-1])
        seeds[method_name].append(seed)
        tensor_dict = torch.load(os.path.join(res_path, file_name))
        if ptrn is None:
            all_influences[method_name].append(tensor_dict[''])
        else:
            module_influences = None
            for module_name, influences in tensor_dict.items():
                if ptrn.match(module_name):
                    print(f"{module_name}")
                    if module_influences is None: 
                        module_influences = influences
                    else:
                        module_influences += influences
            all_influences[method_name].append(module_influences)
    data_detection_per_method = {}
    datasets_path = os.path.join(datasets_folder, task)
    for method_name, tensors in all_influences.items():
        method_seeds = seeds[method_name]
        for infls_tensor, seed in zip(tensors, method_seeds):
            infls = infls_tensor.numpy()
            dataset = load_from_disk(os.path.join(datasets_path, f"d_{task}_{seed}"))
            noises = np.array(dataset['train']['noise'])
            noise_index = np.where(noises)[0]
            n_train = len(infls)
            detection_rate_list=[]
            low_quality_to_high_quality=np.argsort(infls)[::-1]
            for ind in range(1, len(low_quality_to_high_quality)+1):
                detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
                detection_rate = 100*len(detected_samples)/len(noise_index)
                detection_rate_list.append(detection_rate)
            data_detection_per_method.setdefault(method_name, []).append(detection_rate_list)
    method_names = sorted(data_detection_per_method.keys())
    plt.ioff()
    for method in method_names:
        detection_rate_lists = data_detection_per_method[method]
        drls = np.array(detection_rate_lists)
        drl = np.mean(drls, axis=0)
        confidence_level = 0.95
        degrees_freedom = drls.shape[0] - 1
        sample_standard_error = stats.sem(drls, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, drl, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        xs = 100*np.arange(n_train)/n_train
        plt.plot(xs, drl, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=30, color='r', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{(task.upper())}, {title}', fontsize=15)
    plt.tight_layout()
    plt.savefig(out)  
    plt.clf()  


def draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",  
                                    out = "./data/mdr/cifar.png", module_name = '', step = 500, title = "All modules", task = "cifar"):
    ''' For cifar and resnet '''
    # res_pattern = re.compile(res_filer)
    all_influences = defaultdict(list)
    # seeds = defaultdict(list)
    for file_name in os.listdir(infl_folder):
        name_parts = file_name.split('.')[0].split('_')
        method_name = name_parts[1]
        seed = int(name_parts[-1])
        # seeds[method_name].append(seed)
        tensor_dict = torch.load(os.path.join(infl_folder, file_name))
        t = tensor_dict[module_name]
        if not(torch.any(torch.isnan(t))):
            all_influences[method_name].append(t)
        else:
            print(f"Found nan in {file_name}")
    data_detection_per_method = {}
    dataset_splits: DatasetSplits = torch.load(dataset_file, weights_only = False)
    for method_name, tensors in all_influences.items():
        # method_seeds = seeds[method_name]
        for infls_tensor in tensors:
            infls = infls_tensor.to('cpu').numpy()
            noises = (dataset_splits.noisy_labels["worst"] != dataset_splits.train_clean_labels).to('cpu').numpy()
            noise_index = np.where(noises)[0]
            n_train = len(infls)
            detection_rate_list=[]
            low_quality_to_high_quality=np.argsort(infls)[::-1]
            for ind in range(1, len(low_quality_to_high_quality)+1, step):
                detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
                detection_rate = 100*len(detected_samples)/len(noise_index)
                detection_rate_list.append(detection_rate)
            data_detection_per_method.setdefault(method_name, []).append(detection_rate_list)
    method_names = sorted(data_detection_per_method.keys())
    plt.ioff()
    for method in method_names:
        detection_rate_lists = data_detection_per_method[method]
        drls = np.array(detection_rate_lists)
        drl = np.mean(drls, axis=0)
        confidence_level = 0.95
        degrees_freedom = drls.shape[0] - 1
        sample_standard_error = stats.sem(drls, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, drl, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        xs = 100*np.arange(0, n_train, step)/ n_train
        plt.plot(xs, drl, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=30, color='r', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{(task.upper())}, {title}', fontsize=15)
    plt.tight_layout()
    plt.savefig(out)  
    plt.clf()  

def list_modules(infl_file: str, module_pattern = ''):
    tensor_dict = torch.load(infl_file)
    selected = []
    if module_pattern == '':
        for module_name, infls in tensor_dict.items():
            selected.append((module_name, infls))
    else:
        pattern = re.compile(module_pattern)
        for module_name, infls in tensor_dict.items():
            if pattern.match(module_name): 
                selected.append((module_name, infls))
    for module_name, infls in selected:
        print(f"{module_name}")

def compute_ndr_histogram(scores: torch.Tensor, noise_mask_cpu: torch.Tensor, bins = 10):
    ''' scores is 3-D tensor: agg_method * modules_id * train_sample_scores 
        we compute histograms for each agg_method * modules_id
    '''
    histograms = torch.zeros((scores.shape[0], scores.shape[1], bins), dtype=torch.float, device = "cpu")     

    quantiles = torch.linspace(0, 1, bins + 1, device = scores.device)

    bin_edges = torch.quantile(scores, quantiles, dim=-1)
    bin_ranges = bin_edges[-1] - bin_edges[0]
    bin_sizes = (bin_edges[1:] - bin_edges[:-1]) / bin_ranges

    for method_id in range(scores.shape[0]):
        for module_id in range(scores.shape[1]):
            one_scores = scores[method_id, module_id].cpu()
            one_bin_edges = bin_edges[:, method_id, module_id].cpu()
            # NOTE: histogram does not work on CUDA
            hist, _ = torch.histogram(one_scores, bins = one_bin_edges, weight=noise_mask_cpu, density=False)
            histograms[method_id, module_id] = hist

    histograms = histograms / torch.sum(noise_mask_cpu)

    bin_sizes_p = bin_sizes.permute(1, 2, 0).cpu()

    return histograms, bin_sizes_p

def compute_histograms(infl_matrix: torch.Tensor, noise_mask: torch.Tensor, bins = 10):
    ''' infl_matrix is 2-D tensor train_sample * infl_scores,
         infl_matrix.t() is infl_sample * infl_scores 
    '''
    infl_matrix_t = infl_matrix.t()
    histograms = torch.zeros((infl_matrix_t.shape[0], bins), dtype=torch.float, device = infl_matrix.device)     
    bin_sizes_all = torch.zeros((infl_matrix_t.shape[0], bins), dtype=torch.float, device = infl_matrix.device)     

    quantiles = torch.linspace(0, 1, bins + 1, device = infl_matrix.device)
    for infl_id in range(infl_matrix_t.shape[0]):
        bin_edges = torch.quantile(infl_matrix_t[infl_id], quantiles)
        # count_in_bin = np.sum((module_interactions[infl_id] >= bin_edges[0]) & (module_interactions[infl_id] <= bin_edges[1]))
        hist, _ = torch.histogram(infl_matrix_t[infl_id], bins = bin_edges, weight=noise_mask, density=False)
        histograms[infl_id] = hist
        bin_sizes_all[infl_id] = bin_edges[1:] - bin_edges[:-1]

    mean_histogram = histograms.mean(dim=0)
    return histograms, mean_histogram, bin_sizes_all


# NOTE: Obserted effect: noisy group for some layers has less % (on average) of codirected infl samples than clean group 
def dir_majority_indicator(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' Returns 1 for each train sample that has majority of codirected influences (positive influences)'''
    same_dir_mask = inf_matrix > 0
    if additional_mask is not None:
        same_dir_mask &= additional_mask
        half_infl = additional_mask.sum(dim=0) // 2
    else:
        half_infl = same_dir_mask.shape[0] // 2
    maj = torch.sum(same_dir_mask, dim=0) > half_infl
    return maj

# scores - bigger is better (more influential example).
def mean_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = 0
        inf_matrix = inf_matrix_clone
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = inf_matrix.sum(dim=0) / denoms
    return res

def mean_matrix_score(int_matrix: torch.Tensor, *, use_correct = None, correct_infl_preds, trim_ratio = None, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    if use_correct is not None:
        if use_correct:
            int_matrix = int_matrix[:, :, correct_infl_preds]
        else:
            int_matrix = int_matrix[:, :, ~correct_infl_preds]
    if trim_ratio is not None:
        start_id = round(int_matrix.shape[-1] * trim_ratio / 2)
        end_id = int_matrix.shape[-1] - start_id
        int_matrix_sorted, int_matrix_sorted_ids = torch.sort(int_matrix, dim = -1)
        scores = torch.mean(int_matrix_sorted[:, :, start_id:end_id], dim=(-2,-1))
        del int_matrix_sorted, int_matrix_sorted_ids
    else:  
        scores = int_matrix.mean(dim=(-2,-1))

    # mean_histogram, histograms, _ = compute_histograms(int_matrix, noise_mask)
    # mean_histogram2, _, _ = compute_histograms(int_matrix, noise_mask, bins = 100)
    return scores

def median_matrix_score(int_matrix: torch.Tensor, *, noise_mask, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    scores = int_matrix.median(dim=(-2,-1))

    # mean_histogram, histograms, _ = compute_histograms(int_matrix, noise_mask)
    # mean_histogram2, _, _ = compute_histograms(int_matrix, noise_mask, bins = 100)
    return scores    

# common set on missed influence samples
# def csmi_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
#                               trainset_labels: torch.Tensor, inflset_labels: torch.Tensor,
#                               inflset_logits: torch.Tensor,
#                               same_class = False, descending = False, vote_ratio = 0.3, noise_ratio = 0.3, **_) -> torch.Tensor:
#     ''' 3-D tensor: train sample * module * infl sample '''
    
#     global_total_int_matrix = int_matrix.mean(dim=-2)

#     infl_predictions = torch.argmax(inflset_logits, dim = -1)

#     unique_classes = torch.unique(trainset_labels)

#     infl_groups = {}

#     for label in unique_classes:
#         infl1 = torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0]
#         infl2 = torch.where((inflset_labels == label) & (inflset_labels != infl_predictions))[0]
#         infl3 = torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0]
#         infl4 = torch.where((inflset_labels != label) & (inflset_labels != infl_predictions))[0]

#         infl_groups[label] = (infl1, infl2, infl3, infl4)        


#     train_ids = []
#     infl_ids = []
#     for label in unique_classes:
#         infl1 = torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0]
#         infl2 = torch.where((inflset_labels == label) & (inflset_labels != infl_predictions))[0]
#         infl3 = torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0]
#         infl4 = torch.where((inflset_labels != label) & (inflset_labels != infl_predictions))[0]

#         label_train_ids = torch.where(trainset_labels == label)[0]
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl1)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl2)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl3)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl4)                        
#         # if same_class:
#         #     infl_ids.append(torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0])
#         # else:
#         #     infl_ids.append(torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0])
#     scores = torch.zeros(global_total_int_matrix.shape[0], dtype = torch.float, device = global_total_int_matrix.device)
#     for one_train_ids, one_infl_ids in zip(train_ids, infl_ids):
#         total_int_matrix = global_total_int_matrix[one_train_ids][:, one_infl_ids]
#         test_ids_ordered = total_int_matrix.argsort(dim=0, descending = descending)
#         # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
#         noise_size = int(noise_ratio * total_int_matrix.shape[0])
#         mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
#         col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
#         for t in range(total_int_matrix.shape[0]):
#             mask[test_ids_ordered[t], col_range] = True 
#             votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
#             commonset = torch.where(votes >= vote_ratio)[0]
#             if len(commonset) >= noise_size:
#                 break
#         one_scores = 1.0 - votes

#         selected_ids = torch.argsort(one_scores)[:noise_size]
#         noise_ndr1 = noise_mask[selected_ids].sum().item() / 900
#         print(noise_ndr1)
#         # selected_ids = torch.asgsort(one_scores)[noise_size:]
#         # noise_ndr2 = noise_mask[selected_ids].sum().item()
#         # print(noise_ndr2)
#         scores[one_train_ids] = one_scores
#     # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
#     # scores[commonset] = -1.0     
#     return scores    

def commonsubset_matrix_score(int_matrix: torch.Tensor, *, 
                              trainset_labels: torch.Tensor, inflset_labels: torch.Tensor,
                              same_class = False, descending = False, vote_ratio = 0.4, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    
    global_total_int_matrix = int_matrix.mean(dim=-2)

    unique_classes = torch.unique(trainset_labels)
    train_ids = []
    infl_ids = []
    for label in unique_classes:
        train_ids.append(torch.where(trainset_labels == label)[0])
        if same_class:
            infl_ids.append(torch.where(inflset_labels == label)[0])
        else:
            infl_ids.append(torch.where(inflset_labels != label)[0])
    scores = torch.zeros(global_total_int_matrix.shape[0], dtype = torch.float, device = global_total_int_matrix.device)
    for one_train_ids, one_infl_ids in zip(train_ids, infl_ids):
        total_int_matrix = global_total_int_matrix[one_train_ids][:, one_infl_ids]
        test_ids_ordered = total_int_matrix.argsort(dim=0, descending = descending)
        # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
        noise_size = int(noise_ratio * total_int_matrix.shape[0])
        mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
        col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
        for t in range(total_int_matrix.shape[0]):
            mask[test_ids_ordered[t], col_range] = True 
            votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
            commonset = torch.where(votes >= vote_ratio)[0]
            if len(commonset) >= noise_size:
                break
        one_scores = 1.0 - votes
        scores[one_train_ids] = one_scores
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores    

def commonset_matrix_score(int_matrix: torch.Tensor, *, vote_ratio = 0.2, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.mean(dim=-2)
    test_ids_ordered = total_int_matrix.argsort(dim=0)
    # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
    noise_size = int(noise_ratio * total_int_matrix.shape[0])
    mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
    col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
    for t in range(total_int_matrix.shape[0]):
        mask[test_ids_ordered[t], col_range] = True 
        votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
        commonset = torch.where(votes >= vote_ratio)[0]
        if len(commonset) >= noise_size:
            break
    scores = 1.0 - votes
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores

def mean_min_matrix_score(int_matrix: torch.Tensor, *, min_ratio = None, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    if min_ratio is not None:
        infl_threshold = int(min_ratio * total_int_matrix.shape[-1])
        infl_set_ids = torch.argsort(total_int_matrix, dim = -1)[:, :infl_threshold]
        selected_infl_scores = total_int_matrix.gather(1, infl_set_ids)
        scores = torch.mean(selected_infl_scores, dim = -1)
    else:
        scores = torch.min(total_int_matrix, dim = -1).values
    return scores

def cset_matrix_score(int_matrix: torch.Tensor, *, vote_ratio = 0.2, noise_ratio = 0.3, 
                                                    both_sides = False, **_) -> torch.Tensor:    
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    votes = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    vote_threshold = int(total_int_matrix.shape[-1] * vote_ratio)
    noise_size = int(noise_ratio * total_int_matrix.shape[0])
    test_ids_ordered = total_int_matrix.argsort(dim=0) 
    for t in range(total_int_matrix.shape[0]):
        votes[test_ids_ordered[t]] += 1 
        if both_sides:
            votes[test_ids_ordered[-t-1]] += 1
        enough_votes = torch.sum(votes >= vote_threshold)
        if enough_votes >= noise_size:
            break
    scores = 1.0 - votes / total_int_matrix.shape[-1]
    return scores

def maj_matrix_score(int_matrix: torch.Tensor, *, vote_ratio = 0.5, noise_ratio = 0.3, **_) -> torch.Tensor:    
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    votes = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    vote_threshold = int(total_int_matrix.shape[-1] * vote_ratio)
    noise_size = int(noise_ratio * total_int_matrix.shape[0])
    test_ids_ordered = total_int_matrix.argsort(dim=0)
    voters = torch.zeros(total_int_matrix.shape[-1], dtype = torch.int64, device = total_int_matrix.device)
    voter_ids = torch.arange(total_int_matrix.shape[-1], device = total_int_matrix.device)
    while True:
        cur_test_ids = torch.gather(test_ids_ordered, 0, voters.unsqueeze(0)).view(-1)
        min_val_voter_id = torch.argmin(total_int_matrix[cur_test_ids, voter_ids])
        cur_test_id = test_ids_ordered[voters[min_val_voter_id]]
        votes[cur_test_id] += 1
        voters[min_val_voter_id] += 1
        enough_votes = torch.sum(votes >= vote_threshold)
        if enough_votes >= noise_size:
            break
    scores = 1.0 - votes / total_int_matrix.shape[-1]
    return scores

# def get_pareto_front_indexes_neg(int_matrix: torch.Tensor) -> np.ndarray:
#     ''' Get the pareto front from a population. 
#         NOTE: greater is better here. Invert your fitness if it is the opposite.
#     '''
#     # mask = np.ones(int_matrix.shape[0], dtype=bool)
#     # mask[exclude_indexes] = False
#     # index_remap = np.where(mask)[0]
#     domination_matrix = torch.all(int_matrix[:, None] >= int_matrix, dim=2) & torch.any(int_matrix[:, None] > int_matrix, axis=2)
#     indexes = torch.where(~torch.any(domination_matrix, axis=1))[0]
#     return indexes

def pareto_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.mean(dim=-2)
    total_mean = total_int_matrix.mean(dim=-1)
    binary_mask = total_int_matrix < total_mean[:, None]
    # total_int_matrix = total_int_matrix[:, :10]
    scores = torch.zeros(total_int_matrix.shape[0], dtype=total_int_matrix.dtype, device = total_int_matrix.device)
    dom_ids = get_pareto_front_indexes_neg(total_int_matrix)
    scores[dom_ids] -= 1
    # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
    # noise_size = int(noise_ratio * total_int_matrix.shape[0])
    # mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
    # col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
    # for t in range(total_int_matrix.shape[0]):
    #     mask[test_ids_ordered[t], col_range] = True 
    #     votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
    #     commonset = torch.where(votes >= vote_ratio)[0]
    #     if len(commonset) >= noise_size:
    #         break
    # scores = 1.0 - votes
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores    

def confident_matrix_score(int_matrix: torch.Tensor, *, base_method_fn = mean_matrix_score, 
                            trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, 
                            inflset_logits: torch.Tensor, n_confident = 50, **_) -> torch.Tensor:    
    ''' 3-D tensor: train sample * module * infl sample '''
    # TODO: for multiclass we should use maximal non-golden conterpart for confidence
    logit_dist = inflset_logits[torch.arange(inflset_logits.shape[0]), inflset_labels] - inflset_logits[torch.arange(inflset_logits.shape[0]), 1 - inflset_labels]
    num_bigger = torch.sum(logit_dist > 0)
    n_confident_local = min(num_bigger.item(), n_confident)
    infl_ids = torch.argsort(logit_dist, descending=True)[:n_confident_local]
    selected_int_matrix = int_matrix[:, :, infl_ids]
    del logit_dist
    scores = base_method_fn(selected_int_matrix, trainset_labels = trainset_labels, inflset_labels = inflset_labels[infl_ids], inflset_logits = inflset_logits[infl_ids])
    del infl_ids
    return scores


def median_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample ''' 
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    scores = torch.median(total_int_matrix, dim = -1).values
    return scores

def vote_matrix_score(rank_matrix: torch.Tensor, *, chunk_size = 10000, filter_perc = 0.3, **_) -> torch.Tensor:
    # total_rank_matrix = rank_matrix.view(rank_matrix.shape[0], -1)
    votes = torch.zeros(rank_matrix.shape[0], dtype = torch.float, device = rank_matrix.device)
    filter_threshold = round(filter_perc * rank_matrix.shape[-1])
    for (ranks_view, votes_view) in zip(torch.split(rank_matrix, chunk_size, dim = 0),
                                                torch.split(votes, chunk_size, dim = 0)):    
        votes_view[:] = torch.sum(ranks_view >= filter_threshold, dim=(-2, -1), dtype=torch.float)
    return votes

def min_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    scores = torch.min(int_matrix.view(int_matrix.shape[0], -1), dim = -1).values
    return scores

def max_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    scores = torch.max(int_matrix.view(int_matrix.shape[0], -1), dim = -1).values
    return scores

def rank_matrix_score(int_matrix: torch.Tensor, *, 
                        use_correct = None, correct_infl_preds, 
                        chunk_size = 10000, rank_score_fn = mean_matrix_score, **_):
    ''' 3-D tensor: train sample * module * infl sample '''
    if use_correct is not None:
        if use_correct:
            int_matrix = int_matrix[:, :, correct_infl_preds]
        else:
            int_matrix = int_matrix[:, :, ~correct_infl_preds]    
    
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    ranks = torch.zeros_like(total_int_matrix, dtype=torch.float, device = total_int_matrix.device)
    rank_ranges = torch.arange(total_int_matrix.shape[0], device = total_int_matrix.device, dtype=torch.float).view(-1, 1).repeat(1, chunk_size)
    for (int_matrix_view, ranks_view) in zip(torch.split(total_int_matrix, chunk_size, dim = 1),
                                                torch.split(ranks, chunk_size, dim = 1)):
        sort_indexes = torch.argsort(int_matrix_view, dim = 0)
        # rank_range += 1.0
        # int_matrix_view = int_matrix.view(int_matrix.shape[0], -1)        
        # ranks[sort_indexes] = rank_range
        ranks_view.scatter_(0, sort_indexes, rank_ranges)
        # for module_id in range(int_matrix.shape[1]):
        #     for val_id in range(int_matrix.shape[2]):
        #         ranks[sort_indexes[val_id, module_id], module_id, val_id] = rank_range
        del sort_indexes
        # del sort_indexes, rank_range
    del rank_ranges

    # noisy_ranks = ranks[noise_mask] / (int_matrix.shape[0] - 1)
    # mean_ranker_noise_scores = torch.mean(noisy_ranks, dim=0) 
    # good_rankers_mask = mean_ranker_noise_scores < 0.3
    # good_rankers_mask_sz = good_rankers_mask.sum()
    # # good_rankers_mask = (mean_ranker_noise_scores.view(-1, int_matrix.shape[-1]).mean(dim=0, dtype=torch.float)) < 0.5
    # # match_labels = trainset_labels == inflset_labels
    # # missmatch_labels = trainset_labels != inflset_labels

    # pred_labels = torch.argmax(inflset_logits, dim = -1)
    # correct_mask = inflset_labels == pred_labels

    # good_and_correct = good_rankers_mask.view(-1, correct_mask.shape[0]) & correct_mask.unsqueeze(0)
    # good_and_correct_counts = torch.sum(good_and_correct, dim=1) / good_rankers_mask_sz  #per layer
    # discovered_count = torch.mean(good_and_correct_counts, dtype=torch.float)

    # good_and_incorrect = good_rankers_mask.view(-1, correct_mask.shape[0]) & (~correct_mask).unsqueeze(0)
    # good_and_incorrect_counts = torch.sum(good_and_incorrect, dim=1) / good_rankers_mask_sz  #per layer
    # discovered_count2 = torch.mean(good_and_incorrect_counts, dtype=torch.float)


    scores = rank_score_fn(ranks.view(int_matrix.shape), correct_infl_preds = correct_infl_preds)
    del ranks
    return scores

def draw_clusters(X, y, G, module_id = 0):
    import umap

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X_umap = umap_model.fit_transform(X)

    distinct_y = np.unique(y)

    # colors = ['#03fcc2', '#ed91e4', '#03fcc2', '#ed91e4', '#f04354', '#696667']

    legends = []
    plt.ioff()
    plt.clf()

    for label_id, label in enumerate(distinct_y):

        X0 = X_umap[(G == 0) & (y == label)]
        X1 = X_umap[(G == 1) & (y == label)]
        # color = colors[label_id]

        h = plt.scatter(X0[:, 0], X0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        plt.scatter(X1[:, 0], X1[:, 1], marker="x", s=20, alpha = 1, color=h.get_facecolor()[0])
        legends.append(f"Cluster {label}, noise")
    
    plt.legend(legends)
    plt.savefig("tmp.pdf")
    plt.savefig(f"data/clusters/l-{module_id}.pdf")
    pass

def draw_clusters2(X1, y1, G1, X2, y2, G2, module_id = 0):

    plt.ioff()
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    import umap

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X1_umap = umap_model.fit_transform(X1)

    distinct_y = np.unique(y1)

    legends = []
    for label_id, label in enumerate(distinct_y):

        xx0 = X1_umap[(G1 == 0) & (y1 == label)]
        xx1 = X1_umap[(G1 == 1) & (y1 == label)]
        # color = colors[label_id]

        ax[0].scatter(xx0[:, 0], xx0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        ax[0].scatter(xx1[:, 0], xx1[:, 1], marker="x", s=20, alpha = 1)
        legends.append(f"Cluster {label}, noise")
    
    ax[0].legend(legends)

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X2_umap = umap_model.fit_transform(X2)

    distinct_y = np.unique(y2)

    legends = []
    for label_id, label in enumerate(distinct_y):

        xx0 = X2_umap[(G2 == 0) & (y2 == label)]
        xx1 = X2_umap[(G2 == 1) & (y2 == label)]
        # color = colors[label_id]

        ax[1].scatter(xx0[:, 0], xx0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        ax[1].scatter(xx1[:, 0], xx1[:, 1], marker="x", s=20, alpha = 1) # color=h.get_facecolor()[0])
        legends.append(f"Cluster {label}, noise")
    
    ax[1].legend(legends)


    plt.savefig("tmp.pdf")
    plt.savefig(f"data/clusters/l-{module_id}.pdf")
    plt.close(fig)
    pass

def distance_to_centroid(X, _):
    centroid = np.mean(X, axis = 0)
    d = np.linalg.norm(X - centroid, axis=1)
    max_d = np.max(d)
    return 1 - d / max_d


def cluster_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
                         trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, 
                         measure_clusters = silhouette_samples, kmax = 4, **_):
    ''' 3-D tensor: train sample * module * infl sample '''

    int_matrix_X = int_matrix.float().cpu().numpy()
    # same_class = trainset_labels[:, None] == inflset_labels[None, :]
    labels = trainset_labels.cpu().numpy()
    # labels = same_class.to(torch.int)
    # labels[:, trainset_labels == 1] += 2
    # labels = labels.cpu().numpy()
    noise_mask_np = noise_mask.cpu().numpy()
    scores = []
    thr = 1350
    noise_size = 900
    ndrs = []
    for module_id in range(int_matrix.shape[1]):
        # random_state = np.random.randint(0, 100000)
        module_int_matrix_X = int_matrix_X[:, module_id]
        X_normalized = StandardScaler().fit_transform(module_int_matrix_X)
        # X_normalized = module_int_matrix_X
        # xmeans_instance = KMeans(n_clusters = kmax, random_state = random_state).fit(X_normalized)
        # # xmeans_instance.process()
        # # clusters = xmeans_instance.get_clusters()
        # # labels = np.zeros(int_matrix.shape[0], dtype=int)
        # # for cluster_id, smaple_ids in enumerate(clusters):
        # #     labels[smaple_ids] = cluster_id
        # labels = xmeans_instance.labels_


        draw_clusters(X_normalized, labels, noise_mask_np, module_id = module_id)

        scores_by_clusters = measure_clusters(X_normalized, labels)
        idxs = np.argsort(scores_by_clusters)
        start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
        end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
        print(f"Module {module_id} start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")
        ndrs.append({"module_id": module_id, "start_noise_30": start_noise_30, "end_noise_30": end_noise_30})
        scores.append(scores_by_clusters)

    df = pd.DataFrame(ndrs)
    df.to_csv("data/clusters/ndrs.csv")
    mean_scores = np.mean(scores, axis = 0)
    idxs = np.argsort(mean_scores)
    start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
    end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
    print(f"Total: start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")

    return mean_scores

def cluster_matrix_score2(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
                         trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, 
                         measure_clusters = distance_to_centroid, kmax = 4, **_):
    ''' 3-D tensor: train sample * module * infl sample '''

    int_matrix_X = int_matrix.float().cpu().numpy()
    train_labels = trainset_labels.cpu().numpy()
    infl_labels = inflset_labels.cpu().numpy()
    all_labels = np.unique(train_labels)
    train_ids = []
    infl_ids = []
    for label in all_labels:
        train_ids.append(np.where(train_labels == label)[0])
        infl_ids.append(np.where(infl_labels == label)[0])
    # same_class = trainset_labels[:, None] == inflset_labels[None, :]
    # labels = same_class.to(torch.int)
    # labels[:, trainset_labels == 1] += 2
    # labels = labels.cpu().numpy()
    noise_mask_np = noise_mask.cpu().numpy()
    scores = []
    thr = 1350
    noise_size = 900
    ndrs = []
    for module_id in range(int_matrix.shape[1]):
        # random_state = np.random.randint(0, 100000)
        module_int_matrix_X = int_matrix_X[:, module_id]
        X_normalized = StandardScaler().fit_transform(module_int_matrix_X)
        # X_normalized = module_int_matrix_X
        # xmeans_instance = KMeans(n_clusters = kmax, random_state = random_state).fit(X_normalized)
        # # xmeans_instance.process()
        # # clusters = xmeans_instance.get_clusters()
        # # labels = np.zeros(int_matrix.shape[0], dtype=int)
        # # for cluster_id, smaple_ids in enumerate(clusters):
        # #     labels[smaple_ids] = cluster_id
        # labels = xmeans_instance.labels_

        X1 = X_normalized[train_ids[0]][:, infl_ids[0]]
        y1 = train_labels[train_ids[0]]
        G1 = noise_mask_np[train_ids[0]]

        X2 = X_normalized[train_ids[1]][:, infl_ids[1]]
        y2 = train_labels[train_ids[1]]
        G2 = noise_mask_np[train_ids[1]]
        all_scores_by_clusters = np.zeros(int_matrix_X.shape[0], dtype = float)

        draw_clusters2(X1, y1, G1, X2, y2, G2, module_id = module_id)

        scores_by_clusters = measure_clusters(X1, y1)
        all_scores_by_clusters[train_ids[0]] = scores_by_clusters

        scores_by_clusters = measure_clusters(X2, y2)
        all_scores_by_clusters[train_ids[1]] = scores_by_clusters 

        idxs = np.argsort(all_scores_by_clusters)
        start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
        end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
        print(f"Module {module_id} start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")
        ndrs.append({"module_id": module_id, "start_noise_30": start_noise_30, "end_noise_30": end_noise_30})
        scores.append(all_scores_by_clusters)

    df = pd.DataFrame(ndrs)
    df.to_csv("data/clusters/ndrs.csv")
    mean_scores = np.mean(scores, axis = 0)
    idxs = np.argsort(mean_scores)
    start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
    end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
    print(f"Total: start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")

    return mean_scores


def dir_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    same_dir_mask = int_matrix > 0
    one_scores = same_dir_mask.float()
    scores = torch.mean(one_scores, dim=(-2, -1))
    del same_dir_mask, one_scores
    return scores


def median_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = torch.nan
        inf_matrix = inf_matrix_clone
    res = torch.nanmedian(inf_matrix, dim=0)[0]
    return res

def mean_dir_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    same_dir_mask = inf_matrix > 0
    if additional_mask is not None:
        same_dir_mask &= additional_mask
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = same_dir_mask.sum(dim=0).float() / denoms
    return res

def mean_on_preds(inf_matrix: torch.Tensor, *, mask_cache, base_dir, task, seed, **_) -> torch.Tensor:
    key = (base_dir, task, seed)
    if key not in mask_cache:
        logits_path = os.path.join(base_dir, f"m_b_{task}_{seed}", "infl_logits.pt")
        ds_path = os.path.join(base_dir, f"d_{task}_{seed}")
        ds = datasets.load_from_disk(ds_path)
        inflset_labels = torch.tensor(ds["infl"]['labels'], device = "cuda")
        inflset_logits = torch.load(logits_path)
        inflset_preds = torch.argmax(inflset_logits, dim=1)
        inflset_correct = inflset_preds == inflset_labels
        mask_cache[key] = inflset_correct
    inflset_correct = mask_cache[key]
    filtered_infl_matrix = inf_matrix[inflset_correct]
    return mean_score(filtered_infl_matrix)

def get_simple_module_name(module_name, *, replace_name = {'layer': 'L', 'classifier': 'CL', 'word_embeddings': 'WE', 'lora_A': 'A', 'lora_B': 'B' }, simple_module_names = {}):
    if module_name not in simple_module_names:
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
        simple_module_name = ' '.join(simple_name_parts)
        simple_module_names[module_name] = simple_module_name   
    return simple_module_names[module_name] 

layer_regex = re.compile(r'layer\.(.+?)\.')
def get_simple_module_and_layer_name(module_name: str, *, replace_name = {'layer': 'L', 'classifier': 'CL', 'word_embeddings': 'WE', 'lora_A': 'A', 'lora_B': 'B' }, simple_module_names = {}):
    if module_name not in simple_module_names:
        module_layer_match = layer_regex.search(module_name)
        if module_layer_match:
            layer_name = module_layer_match.group(1)
        elif (".classifier." in module_name) or (module_name == 'classifier'):
            layer_name = "CL"
        elif (".word_embeddings." in module_name) or (module_name == 'word_embeddings'):
            layer_name = "WE"
        else:
            layer_name = ""
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'weight', 'attention', 'self', 'modules_to_save']]
        num_to_skip = 0 
        if layer_name not in [ 'CL', 'WE', '']:
            num_to_skip = 2
        elif layer_name == 'CL':
            num_to_skip = 1
        simple_module_name = ' '.join(simple_name_parts[num_to_skip:])
        simple_module_names[module_name] = (layer_name, simple_module_name)
    return simple_module_names[module_name] 

def get_simple_agg_name(agg_method):
    score_names = {mean_score: "mean", mean_dir_score: "dir", median_score: "median", mean_rank_score: "rank"} #, mean_on_preds_fn: "meanpred"}
    return score_names.get(agg_method, agg_method.__name__)

def get_simple_infl_name(infl_method):
    infl_method_names = {"cos": "cos", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf", "datainf0": "datainf0", "datainf_one": "datainf1"}    
    return infl_method_names.get(infl_method, infl_method)


def split_dict(d: dict, filter_fn:callable):
    d1 = {}
    d2 = {}
    for k, v in d.items():
        if filter_fn(k):
            d1[k] = v
        else:
            d2[k] = v
    return d1, d2

def get_avg_ranks(setup_score_values: dict[str, list[float]], *, ascending = True) -> dict[str, float]:
    ''' ranks setups, assumes that same trials are given as list, index of trial defines same trial accross setups  '''
    setup_names = list(setup_score_values.keys())
    total_measures = np.array([setup_score_values[setup_name] for setup_name in setup_names])
    df = pd.DataFrame(total_measures.T)
    # sort_indeces = np.argsort(total_measures_per_trial, axis = 1)
    ranks = df.rank(axis = 1, method="min", ascending=ascending)
    mean_ranks = ranks.mean(axis = 0)
    rank_dict = {setup_name: mean_rank for setup_name, mean_rank in zip(setup_names, mean_ranks)}
    return rank_dict


def load_ds_info(task_in_dir: str):
    noise_list_dict = {}
    trainset_labels_dict = {}
    inflset_labels_dict = {}
    for file_name in os.listdir(task_in_dir):
        if not file_name.startswith("d_"):
            continue
        ds_file_parts = file_name.split('_')
        *_, run_id = ds_file_parts
        ds_path = os.path.join(task_in_dir, file_name)
        ds = datasets.load_from_disk(ds_path)
        trainset = ds['train']
        noise_list_dict[run_id] = trainset['noise']
        trainset_labels_dict[run_id] = trainset['labels']
        inflset = ds['infl']
        inflset_labels_dict[run_id] = inflset['labels']
    return noise_list_dict, trainset_labels_dict, inflset_labels_dict

def load_m_info(task_in_dir: str, m_prefix: str):
    ''' Loads infl samples logits on the specified chehckpoint '''
    inflset_logits = {}
    for file_name in os.listdir(task_in_dir):
        if not file_name.startswith(m_prefix):
            continue
        m_file_parts = file_name.split('_')
        *_, run_id = m_file_parts
        logits_m_path = os.path.join(task_in_dir, file_name, "infl_logits.pt")
        logits = torch.load(logits_m_path)
        inflset_logits[run_id] = logits
    return inflset_logits

def get_cancellation_metrics_table(base_dir_path: str, task='qnli',
                                    module_groups_regex: dict[str, str] = {}):
    cancel_abs_per_module = defaultdict(list)
    cancel_norm_per_module = defaultdict(list)    
    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}
    task_in_dir = os.path.join(base_dir_path, task)
    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith("c_"):
            continue
        config_path = os.path.join(task_in_dir, file_name_with_ext)
        with open(config_path, 'r') as f:
            config = json.load(f)
        cur_cancel_abs = config['finetune']['cancel_abs']
        cur_cancel_norm = config['finetune']['cancel_norm']
        for module_name, score in cur_cancel_abs.items():
            cancel_abs_per_module[module_name].append(score)
        for module_name, score in cur_cancel_norm.items():
            cancel_norm_per_module[module_name].append(score)

    module_names = list(cancel_abs_per_module.keys())
    for pattern_name, pattern in module_groups_patterns.items():
        group_cancel_abs = []
        group_cancel_norm = []
        for module_name in module_names:
            if pattern.match(module_name):
                group_cancel_abs.append(cancel_abs_per_module[module_name])
                group_cancel_norm.append(cancel_norm_per_module[module_name])
        cancel_abs_per_module[pattern_name] = np.array(group_cancel_abs).mean(axis = 0)
        cancel_norm_per_module[pattern_name] = np.array(group_cancel_norm).mean(axis = 0)

    cancel_abs_per_group, cancel_abs_per_orig_module = split_dict(cancel_abs_per_module, lambda k: (k in module_groups_patterns) or (k == 'total'))
    cancel_norm_per_group, cancel_norm_per_orig_module = split_dict(cancel_norm_per_module, lambda k: (k in module_groups_patterns) or (k == 'total'))
    cancel_abs_module_ranks = get_avg_ranks(cancel_abs_per_orig_module)
    cancel_abs_group_ranks = get_avg_ranks(cancel_abs_per_group)
    cancel_norm_module_ranks = get_avg_ranks(cancel_norm_per_orig_module)
    cancel_norm_group_ranks = get_avg_ranks(cancel_norm_per_group)
    avg_cancel_abs_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_abs_per_module.items()}
    avg_cancel_norm_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_norm_per_module.items()}

    rows = []
    for module_name in module_names:
        cancel_abs_mean, cancel_abs_std = avg_cancel_abs_per_module[module_name]
        cancel_norm_mean, cancel_norm_std = avg_cancel_norm_per_module[module_name]
        cancel_abs_module_rank = cancel_abs_module_ranks.get(module_name, pd.NA)
        cancel_abs_group_rank = cancel_abs_group_ranks.get(module_name, pd.NA)
        cancel_norm_module_rank = cancel_norm_module_ranks.get(module_name, pd.NA)
        cancel_norm_group_rank = cancel_norm_group_ranks.get(module_name, pd.NA)
        cancel_row_data = {
            "cancel_norm_module_rank": cancel_norm_module_rank,
            "cancel_norm_group_rank": cancel_norm_group_rank,
            "cancel_norm_mean": cancel_norm_mean,
            "cancel_norm_std": cancel_norm_std,
            "cancel_abs_module_rank": cancel_abs_module_rank,
            "cancel_abs_group_rank": cancel_abs_group_rank,
            "cancel_abs_mean": cancel_abs_mean,
            "cancel_abs_std": cancel_abs_std                
        }
        rows.append(cancel_row_data)
    df = pd.DataFrame(rows)
    return df

def output_table(df: pd.DataFrame, base_path: str, task: str):
    ndr_stats_file_path = os.path.join(base_path, f"{task}_ndr_stats.csv")
    
    df.to_csv(ndr_stats_file_path, index = False)

    stats_file_path = os.path.join(base_path, f"{task}_ndr_stats_simple.txt")
    with open(stats_file_path, "w") as stats_file:
        print(tabulate(df, headers = 'keys', tablefmt="github", floatfmt=".3f", showindex=True), file = stats_file)
    pass 

# RQ1: what checkpoint it is better to use with infl methods to detect noise 
def get_df_from_file(metric_file: str, module_groups):    

    # file_name = os.path.basename(metric_file).split(".")[0]
    with open(metric_file, 'r') as f:
        runs = [json.loads(l) for l in f.readlines()]

    rows = []
    rand_accuracies = {}
    for r in runs: 

        accuracy0 = r["first_finetune"]["accuracy"]
        accuracy = r["accuracy"]
        best_accuracy0 = np.max(accuracy0)
        best_accuracy = np.max(accuracy)
        accuracy_delta = best_accuracy - best_accuracy0

        infl_accuracy0 = r["first_finetune"]["infl_accuracy"]
        infl_accuracy = r["infl_accuracy"]
        best_infl_accuracy0 = np.max(infl_accuracy0)
        best_infl_accuracy = np.max(infl_accuracy)
        infl_accuracy_delta = best_infl_accuracy - best_infl_accuracy0

        infl_loss0 = r["first_finetune"]["infl_loss"]
        infl_loss = r["infl_loss"]
        min_infl_loss0 = np.min(infl_loss0)
        min_infl_loss = np.min(infl_loss)        
        infl_loss_delta = min_infl_loss - min_infl_loss0

        logits_change = r["logits_change"]

        auc_ndr = r["auc_ndr"]
        filtered = r["filtered"]

        noise_curve = r["ndr_curve"]
        num_noise = round(0.2 * len(noise_curve))
        ideal_area = num_noise / 2 + (len(noise_curve) - num_noise)        
        auc_ndr2 = sum((noise_curve[i] + noise_curve[i + 1]) / 2 for i in range(len(noise_curve) - 1)) / ideal_area

        task = r["config"]["task"]
        infl_method = r["config"]["infl_method"]
        seed = r["config"]["seed"]
        filter_perc = r["config"]["filter_perc"]

        index_30 = round(filter_perc * len(noise_curve))
        noise_30  = noise_curve[index_30]

        assert np.allclose(auc_ndr, auc_ndr2, atol=1e-2)
        assert np.allclose(noise_30, filtered, atol=1e-2)

        if infl_method == "rand":
            rand_accuracies[(task, filter_perc, seed)] = (best_accuracy, best_infl_accuracy)

        module_name = r["config"]["module_name"]
        modules_pattern = module_groups.get(module_name, None)

        if modules_pattern is None:
            cancel = 0
        else:
            cancel_map = r["first_finetune"]["cancel_norm"]

            cancel = 0
            cancel_count = 0
            for m in cancel_map.keys():
                if modules_pattern.match(m):
                    cancel += cancel_map[m]
                    cancel_count ++ 1 
            
            cancel = 0 if cancel_count == 0 else (cancel / cancel_count)

        row = {
               "task": task,
               "filter_perc": filter_perc,
               "infl_method": infl_method,
               "agg_method": r["config"]["agg_method"],
               "module": module_name,
               
               "best_accuracy_delta": accuracy_delta,
               "best_infl_accuracy_delta": infl_accuracy_delta,
                "infl_loss_delta": infl_loss_delta,
               "logits_change": logits_change,
               "noise_30": filtered,
               "auc_ndr": auc_ndr,

               "best_accuracy_0": best_accuracy0,
               "best_accuracy_1": best_accuracy,

               "best_infl_accuracy_0": best_infl_accuracy0,
               "best_infl_accuracy_1": best_infl_accuracy,

                "infl_loss_0": min_infl_loss0,
                'infl_loss_1': min_infl_loss,
                "cancel": cancel,

               "seed": seed

               }
        rows.append(row)

    for r in rows:
        key = (r["task"], r["filter_perc"], r["seed"])
        rand_accuracy, rand_infl_accuracy = rand_accuracies[key]
        accuracy_rand_delta = r["best_accuracy_1"] - rand_accuracy
        infl_accuracy_rand_delta = r["best_infl_accuracy_1"] - rand_infl_accuracy
        r["accuracy_rand_delta"] = accuracy_rand_delta
        r["infl_accuracy_rand_delta"] = infl_accuracy_rand_delta

    df = pd.DataFrame(rows)

    return df

def get_agg_df(df: DataFrame, key_columns = ['task', 'filter_perc', 'infl_method', 'agg_method', 'module'],
                drop_columns=['seed'], selected_columns=None, agg_types = ['mean', 'std']):
    if selected_columns is None:
        grouped = df.drop(columns=drop_columns).groupby(key_columns)
    else:
        grouped = df[key_columns + selected_columns].drop(columns=drop_columns).groupby(key_columns)
    aggregated = grouped.agg(agg_types).reset_index()
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    aggregated_sorted = aggregated.sort_values(by='best_accuracy_delta_mean', ascending=False)
    std_columns = [col for col in aggregated_sorted.columns if col.endswith('std')]
    other_columns = [col for col in aggregated_sorted.columns if not col.endswith('std')]

    # Reorder the columns
    aggregated_sorted = aggregated_sorted[other_columns + std_columns]    
    return aggregated_sorted

def get_all_df(base_path = "data/roberta/filter-30", datasets = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"],
                tmp_file = "tmp.pkl", module_groups = {}):
    # if os.path.exists(tmp_file):
    #     df1 = pd.read_pickle(tmp_file)
    # else:
    plain_dataframes = []
    for dataset in datasets:
        file = os.path.join(base_path, f"{dataset}-bl.jsonlist")
        df = get_df_from_file(file, module_groups)
        plain_dataframes.append(df)
    
    df1 = pd.concat(plain_dataframes, ignore_index=True)
    # df1.to_pickle(tmp_file)
    return df1

def rank_checkpoints(base_path = "data/roberta-2", metric = "noise_30",
                        datasets = ["mrpc", "qnli", "sst2", "qqp"],
                        checkpoints = ["b", "l", "bl"]):
    df1 = get_all_df(base_path = base_path, datasets = datasets,
                        checkpoints = checkpoints)
    key_columns = ["checkpoint", "task", "infl_method", "agg_method", "module", "seed"]

    df2 = df1[(df1["agg_method"] != "")]
    df3 = df2[key_columns + [metric]]
    df4 = df3.pivot(index=["task", "infl_method", "agg_method", "module", "seed"], columns="checkpoint", values=metric)
    df4 = df4[~df4.isna().any(axis=1)]
    df5: DataFrame = df4[df4[checkpoints].gt(0).any(axis=1)]

    stats_data = df5.T.to_numpy()

    friedman_res = sci_stats.friedmanchisquare(*stats_data)

    import scikit_posthocs as sci_posthocs

    nemenyi_res = sci_posthocs.posthoc_nemenyi_friedman(stats_data.T) 
    print(f"\n----------------------------------")
    print(f"Friedman: {friedman_res}")
    from tabulate import tabulate
    names = ["b", "bl", "l"]
    rows = []
    for i in range(len(names)):
        row = []
        row.append(names[i])
        for j in range(len(names)):
            row.append(nemenyi_res[i][j])
        rows.append(row)
    print(tabulate(rows, headers=["", *names], tablefmt="grid", numalign="center", stralign="center"))


    # df5.groupby(["task", "infl_method", "agg_method", "module"]).agg(["mean"]).groupby(["task", "infl_method", "agg_method"]).agg(["max"]).reset_index()    
    mean_ranks = df5.rank(axis=1, ascending=False).mean(axis=0)
    mean_metric = df5.mean(axis=0)
    print(mean_ranks)
    print(mean_metric)
    # score = df6.groupby(["task"]).agg(["mean"]).reset_index()
    return mean_ranks, mean_metric


def estimate_ndr(folder:str, prefix: str):
    noise_30 = {}
    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(folder)
    for file_name in os.listdir(folder):
        if not file_name.startswith(prefix):
            continue 
        run_id = file_name.split(".")[0].split("_")[-1]
        noise_list = noise_list_dict[run_id]
        noise_mask = torch.tensor(noise_list, device = "cpu")
        total_noise = noise_mask.sum()
        path = os.path.join(folder, file_name)
        score_dict = torch.load(path)
        for key, scores in score_dict.items():
            train_idxs = torch.argsort(scores)
            filter_len = int(0.3*len(train_idxs))
            train_idxs_left = train_idxs[:filter_len].cpu().numpy()            
            noise_tensor = noise_mask[train_idxs_left]            
            num_noise = noise_tensor.sum()
            ndr = num_noise / total_noise
            noise_30.setdefault(key, []).append(ndr)
        pass
    rows = []
    for (infl_method, agg_method, module), ndrs in noise_30.items():
        rows.append({
            "infl_method": infl_method, "agg_method": agg_method,
            "module": module, "ndr_30_mean": np.mean(ndrs), "ndr_30_std": np.std(ndrs)
        })
    df = DataFrame(rows)
    df = df.sort_values(by='ndr_30_mean', ascending=False)
    pass 

def create_dataset_table():

    with open("data/roberta/groups.json", "r") as f:
        module_patterns = json.load(f)
        
    module_patterns = {m: re.compile(p) for m, p in module_patterns.items()}

    datasets = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]
    for dataset in datasets:    
        df = get_all_df(datasets = [dataset], module_groups=module_patterns)
        sorted_df = get_agg_df(df)
        # sorted_df2 = sorted_df.reset_index()

        sorted_df2 = sorted_df.sort_values(by="best_accuracy_1_mean", ascending=False)

        columns_to_include = ["best_accuracy_1", "best_infl_accuracy_1", "accuracy_rand_delta", "noise_30", "auc_ndr", "infl_loss_delta", "cancel", "logits_change"]
        rows = []
        for row in sorted_df2.to_dict(orient="records"):
            new_row = []
            infl_method = row["infl_method"]
            module = row["module"]
            new_row.append(infl_method)
            new_row.append(module)
            for c in columns_to_include:
                v_mean = row[c + "_mean"]
                v_std = row[c + "_std"]
                if c in ["best_accuracy_1", "best_infl_accuracy_1", "accuracy_rand_delta", "noise_30", "auc_ndr"]:
                    v_mean = round(v_mean * 100) 
                    v_std = round(v_std * 100)
                else:
                    if v_mean == 0 and c == "cancel":
                        v_mean = ""
                        v_std = "0"
                    else:
                        v_mean = round(v_mean * 100) / 100
                        v_mean = str(v_mean)
                        if v_mean.startswith("0."):
                            v_mean = "." + v_mean[2:]
                        if v_mean.startswith("-0."):
                            v_mean = "-." + v_mean[3:]
                        if  v_std == 0:
                            v_std = "0"
                        else:
                            v_std = round(v_std * 100) / 100
                            v_std = str(v_std)
                            if v_std.startswith("0."):
                                v_std = "." + v_std[2:] 
                if (v_std == 0) or (v_std == "0"):
                    new_row.append(f"{v_mean}")
                else:
                    new_row.append(f"{v_mean} $\pm$ {v_std}")
            rows.append(new_row)

        column_names = ["method", "layers", "$Acc_t$", "$Acc_v$", "$\Delta Acc^{rnd}_t$", "$NDR_{30}$", "$AUC_{NDR}$", "$\Delta loss_v$", "Cancel", "$\Delta logits_t $"]
        with open(f"{dataset}.tex", "w") as stats_file:
            s = tabulate(rows, headers = column_names, showindex=False, tablefmt="latex", floatfmt=".3f")
            s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "$hf^{10}_{we}$").replace("hf\_we\_", "$hf_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("llllllllll", "ll|cccccccc")
            print(s, file = stats_file)
    pass

def run_friedman_tests(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30"):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx

    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    metrics_per_ds['rank'] = ranks.mean(axis=1).round(1)
    metrics_per_ds['rank_std'] = ranks.std(axis=1).round(1)
    
    metrics_per_ds = metrics_per_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True]).drop(columns=["rank", 'rank_std'])    

    methods = [ (m if m in ['denoise', 'rand'] else f'{m}, all') if l == "" else f"{m}, {l}" for m, _, l in metrics_per_ds.index.to_list()]

    stats_data = metrics_per_ds.dropna(axis=1, how='any').to_numpy()

    friedman_res = sci_stats.friedmanchisquare(*stats_data)

    import scikit_posthocs as sci_posthocs

    nemenyi_res = sci_posthocs.posthoc_nemenyi_friedman(stats_data.T) 
    print(f"\n----------------------------------")
    print(f"Friedman {metric_name}: {friedman_res}")
    from tabulate import tabulate
    rows = []
    for i in range(len(methods)):
        row = []
        row.append(i+1)
        for j in range(len(methods)):
            # pvalue = round(nemenyi_res[i][j] * 100) / 100
            pvalue = nemenyi_res[i][j]
            pvalue_exp = int(f"{pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
            if pvalue_exp == 0:
                row.append("")
            elif pvalue <= 0.05:
                row.append(f"\\textbf{{{pvalue_exp}}}")
            else:
                row.append(f"{pvalue_exp}")
        rows.append(row)

    with open(f"{out_folder}/{metric_name}-friedman.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *[i + 1 for i in range(len(methods))]], showindex=False, tablefmt="latex", numalign="left", stralign="left")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(f"% Friedman {metric_name}: {friedman_res}", file = stats_file)
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_wilcoxon_tests(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30"):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx

    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    metrics_per_ds['rank'] = ranks.mean(axis=1).round(1)
    metrics_per_ds['rank_std'] = ranks.std(axis=1).round(1)
    
    metrics_per_ds = metrics_per_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True]).drop(columns=["rank", 'rank_std'])    

    methods = [ (m if m in ['denoise', 'rand'] else f'{m}, all') if l == "" else f"{m}, {l}" for m, _, l in metrics_per_ds.index.to_list()]

    stats_data = metrics_per_ds.dropna(axis=1, how='any').to_numpy()

    wilcoxon_res = {}
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            w = sci_stats.wilcoxon(stats_data[i], stats_data[j])
            wilcoxon_res.setdefault(i, {}).setdefault(j, w.pvalue)
            wilcoxon_res.setdefault(j, {}).setdefault(i, w.pvalue)

    from tabulate import tabulate
    rows = []
    for i in range(len(methods)):
        row = []
        row.append(i+1)
        for j in range(len(methods)):
            # pvalue = round(nemenyi_res[i][j] * 100) / 100
            if i == j:
                pvalue = 1.0
            else:
                pvalue = wilcoxon_res[i][j]
            pvalue_exp = int(f"{pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
            if pvalue_exp == 0:
                row.append("")
            elif pvalue <= 0.05:
                row.append(f"\\textbf{{{pvalue_exp}}}")
            else:
                row.append(f"{pvalue_exp}")
        rows.append(row)

    with open(f"{out_folder}/{metric_name}-wilcoxon.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *[i + 1 for i in range(len(methods))]], showindex=False, tablefmt="latex", numalign="left", stralign="left")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_spearman_tests(metric_name: str = "best_accuracy_1", ndr_delta = None,
                        out_folder = "data/roberta/filter-30", suffix=""):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, 'noise_30'])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    noise_30_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == 'noise_30']}
    common_columns = set.intersection(metric_columns, noise_30_columns)

    rho = {}
    pvalues = {}
    for ds, seed in common_columns:
        series1 = metrics_per_ds.loc[:, (metric_name, ds, seed)].to_numpy()
        series2 = metrics_per_ds.loc[:, ('noise_30', ds, seed)].rank(method="average").to_numpy()
        if ndr_delta is not None:
            ndr_idxs = np.argsort(series2)[::-1]
            selected_idxs = [ndr_idxs[0]] 
            prev_ndr = series2[selected_idxs[-1]]
            for i in range(1, len(ndr_idxs)):
                cur_ndr = series2[ndr_idxs[i]]
                if  prev_ndr - cur_ndr >= ndr_delta:
                    selected_idxs.append(ndr_idxs[i])
                    prev_ndr = cur_ndr
            series1 = series1[selected_idxs]
            series2 = series2[selected_idxs]

        s = sci_stats.spearmanr(series1, series2)
        rho.setdefault(ds, []).append(s.correlation)
        pvalues.setdefault(ds, []).append(s.pvalue)

    rows = []
    # header_row = [""]
    # for d in datasets:
    #     header_row.append(d)
    # header_row.append("Total")
    # rows.append(header_row)

    rho_row = ["Spearman $\\rho$"]
    for d in datasets:
        ds_rho = rho[d]
        mean_rho = round(np.mean(ds_rho) * 10) / 10
        std_rho = round(np.std(ds_rho) * 10) / 10
        rho_row.append(f"{mean_rho} $\pm$ {std_rho}")
    all_rho_plain = [v for vl in rho.values() for v in vl]
    rho_mean = np.mean(all_rho_plain)
    rho_std = np.std(all_rho_plain)
    rho_mean = round(np.mean(rho_mean) * 10) / 10
    rho_std = round(np.std(rho_std) * 10) / 10
    rho_row.append(f"{mean_rho} $\pm$ {std_rho}")
    rows.append(rho_row)

    pvalue_row = ["p-value"]
    for d in datasets:
        ds_pvalues = pvalues[d]
        r = sci_stats.combine_pvalues(ds_pvalues)
        # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
        pvalue_row.append(f"{r.pvalue:.0e}")
    
    r = sci_stats.combine_pvalues([v for vl in pvalues.values() for v in vl])
    # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
    pvalue_row.append(f"{r.pvalue:.0e}")
    rows.append(pvalue_row)

    with open(f"{out_folder}/{metric_name}-spearman{suffix}.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *datasets, "Total"], showindex=False, tablefmt="latex", numalign="center", stralign="center")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_concat_spearman_test(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30", suffix=""):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, 'noise_30'])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    noise_30_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == 'noise_30']}
    common_columns = set.intersection(metric_columns, noise_30_columns)

    ds_series1 = {}
    ds_series2 = {}

    for ds, seed in common_columns:
        series1 = metrics_per_ds.loc[:, (metric_name, ds, seed)].to_numpy()
        series2 = metrics_per_ds.loc[:, ('noise_30', ds, seed)].to_numpy()   
        ds_series1.setdefault(ds, []).append(series1) 
        ds_series2.setdefault(ds, []).append(series2)


    for ds in datasets:
        ds_series1[ds] = np.concatenate(ds_series1[ds])
        ds_series2[ds] = np.concatenate(ds_series2[ds])

    rho = {}
    pvalues = {}
    for ds in datasets:
        series1 = ds_series1[ds]
        series2 = ds_series2[ds]
        s = sci_stats.spearmanr(series1, series2)
        rho[ds] = s.correlation
        pvalues[ds] = s.pvalue

    return rho, pvalues 

def run_best_spearman_test(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30", suffix=""):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, 'noise_30'])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_df = metrics_per_ds.loc[:, metric_name].groupby(level=0, axis=1).max()
    noise_df = metrics_per_ds.loc[:, 'noise_30'].groupby(level=0, axis=1).max()

    rho = {}
    pvalues = {}
    for ds in datasets:
        series1 = metric_df[ds]
        series2 = noise_df[ds]
        s = sci_stats.spearmanr(series1, series2)
        rho[ds] = s.correlation
        pvalues[ds] = s.pvalue

    return rho, pvalues 

def create_tun2_metric_table(metric_name: str = "best_accuracy_1", prec = 2, 
                        out_folder = "data/roberta/filter-30",
                        highlight_max = True, ds_ranks = False, mul = 100,
                        with_row_id = False):
    with open("data/roberta/groups.json", "r") as f:
        module_patterns = json.load(f)
        
    module_patterns = {m: re.compile(p) for m, p in module_patterns.items()}

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets, module_groups=module_patterns)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx
    # metrics_per_ds = metrics_per_ds.round(prec)

    # ranks = metrics_per_ds.round(prec).rank(axis = 0, ascending=False, method="average") # method="dense")
    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    suffix = ""
    if ds_ranks:
        # ranks.columns = idx
        # metrics_per_ds = ranks.groupby(level=0, axis=1).mean()
        metrics_per_ds = ranks
        suffix = "-rank"


    # metrics_per_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-all.csv")

    metric_by_ds_mean = metrics_per_ds.groupby(level=0, axis=1).mean()
    metric_by_ds_std = metrics_per_ds.groupby(level=0, axis=1).std()
    metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
    metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
    metric_by_ds['rank'] = ranks.mean(axis=1)
    metric_by_ds['rank_std'] = ranks.std(axis=1)
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True])

    metric_by_ds = metric_by_ds[[*[de for d in datasets for de in [d, d + "_std"]], "rank", "rank_std"]]
    # metric_by_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-avg.csv")

    filtered_df = metric_by_ds.loc[metric_by_ds.index.get_level_values('infl_method') != 'denoise']
    if highlight_max:
        values_to_highlight = filtered_df[datasets].to_numpy().max(axis=0)
    else:
        values_to_highlight = filtered_df[datasets].to_numpy().min(axis=0)
    
    values_to_highlight = [round(v * (10 ** prec) * mul) / (10 ** prec) for v in values_to_highlight]

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        method = row["infl_method"]        
        layer = row["module"]
        if with_row_id:
            new_row["id"] = (row_id + 1)
        method_rename = {"datainf": "DataInf", "hf": "TracIn", "cos": "Cosine", "denoise": "Full", "rand": "Random", "hf_we_": "TracIn$_{we}$", "hf_we_topk_10": "TracIn$^{10}_{we}$"}
        new_row["Method"] = method_rename[method]
        new_row["Layer"] = "all" if (method not in ["denoise", "rand"]) and layer == "" else layer 
        for did, d in enumerate([*datasets, "rank"]):
            should_highlight = False
            if d == "rank":
                m = round(row[d] * 10) / 10
                m_std = round(row[d + "_std"] * 10) / 10
            else:
                m = round(row[d] * (10 ** prec) * mul) / (10 ** prec)
                if method != "denoise":
                    should_highlight = m == values_to_highlight[did]
                m_std = round(row[d + "_std"] * (10 ** prec) * mul) / (10 ** prec)
            m = f"{m:.1f}"
            m_std = f"{m_std:.1f}"

            d_name = "Rank" if d == "rank" else d.upper()

            if should_highlight:
                new_row[d_name] = f"\\textbf{{{m}}} {{\\footnotesize $\pm$ {m_std} }}"
            else:
                new_row[d_name] = f"{m} {{\\footnotesize $\pm$ {m_std}}}"
        rows.append(new_row)

    with open(f"{out_folder}/{metric_name}{suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass

# def draw_tun2_metric(base_path: str, task:str, metric_name: str, selected_methods: dict = {}, 
#                  draw_diff = False, suffix = ""):
#     infile = os.path.join(base_path, f"{task}-bl.jsonlist")
#     with open(infile, 'r') as f:
#         json_lines = f.readlines()
#     all_metrics = [json.loads(l) for l in json_lines]
#     method_metrics = defaultdict(list)
#     for metrics in all_metrics:
#         infl_method = metrics['config']['infl_method']
#         agg_method = metrics['config']['agg_method']                
#         module_name = metrics['config']['module_name']
#         key = (infl_method, agg_method, module_name)
#         if draw_diff:
#             before_metric_values = metrics['first_finetune'][metric_name] 
#             after_metric_values = metrics[metric_name]
#             metric_values = [a - b for a, b in zip(after_metric_values, before_metric_values)]
#         else:
#             metric_values = metrics[metric_name]        
#         if key not in selected_methods:
#             continue
#         method_metrics[key].append(metric_values)

#     # max_sz = max(len(l) for l in method_metrics.values())

#     # method_metrics_flat = {k: [v3 for v2 in v for v3 in (v2[:10] + [np.nan] * max(0, 10 - len(v2)))] for k, v in method_metrics.items()}
#     # method_metric_ranks = get_avg_ranks(method_metrics_flat)

#     # setup_names = list(setup_score_values.keys())

#     # method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
#     method_names = list(selected_methods.keys())
#     plt.ioff()

#     handles_ = []

#     for i, (infl_method, agg_method, module_name) in enumerate(method_names):
#         metrics = method_metrics[(infl_method, agg_method, module_name)]
#         metric_values = np.array(metrics) * 100
#         mean = np.mean(metric_values, axis=0)
#         confidence_level = 0.95
#         degrees_freedom = metric_values.shape[0] - 1
#         sample_standard_error = stats.sem(metric_values, axis=0)
#         confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
#         min_v = confidence_interval[0]
#         max_v = confidence_interval[1]
#         method_settings = selected_methods[(infl_method, agg_method, module_name)]
#         # default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
#         default_args = dict(marker='o', markersize=0, linewidth=0.9, color = method_settings['color'])
#         draw_full = False
#         shift = ((i - len(method_metrics) // 2) * 0.025)
#         if infl_method == 'denoise':
#             default_args['linestyle']='--'
#             default_args['markersize'] = 0
#             draw_full = True 
#             shift = 0
#         if infl_method == 'rand':
#             default_args['linestyle']='-.'
#             default_args['markersize'] = 0
#             draw_full = True 
#             shift = 0
#         # xs = np.arange(len(mean)) + 1
#         xs = np.arange(len(mean)) + 1 + shift # Shift x-coordinates slightly
#         line = plt.plot(xs, mean, zorder=1, **default_args)
#         if draw_full:
#             plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
#         else:
#             plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], fmt='none', ecolor=line[0].get_color(), capsize=1, linewidth=0.5, zorder=0)
#         handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))
    
#     handles_.sort(key = lambda x: x[2])
#     ordered_handles = [h[0] for h in handles_]
#     ordered_labels = [h[1] for h in handles_]
#     plt.xlabel('Epoch', fontsize=20)
#     plt.ylabel('Accuracy, \\%', fontsize=20)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(ordered_handles, ordered_labels, fontsize=15)
#     # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
#     plt.title(f'{task.upper()}', fontsize=20)
#     plt.tight_layout()
#     outfile = os.path.join(base_path, "plots", f"{task}-{metric_name}-{suffix}.pdf")
#     plt.savefig(outfile)  
#     plt.clf()  

def draw_all_tun2_metric(base_path: str, tasks = benchmark, metric_name: str = "accuracy", figsize = (8, 3),
                        selected_methods: dict = {}, num_in_row = 4, draw_diff = False, suffix = ""):
    
    tasks_metrics = {}
    for task in tasks:
        infile = os.path.join(base_path, f"{task}-bl.jsonlist")
        with open(infile, 'r') as f:
            json_lines = f.readlines()
        all_metrics = [json.loads(l) for l in json_lines]
        method_metrics = defaultdict(list)
        for metrics in all_metrics:
            infl_method = metrics['config']['infl_method']
            agg_method = metrics['config']['agg_method']                
            module_name = metrics['config']['module_name']
            key = (infl_method, agg_method, module_name)
            if draw_diff:
                before_metric_values = metrics['first_finetune'][metric_name] 
                after_metric_values = metrics[metric_name]
                metric_values = [a - b for a, b in zip(after_metric_values, before_metric_values)]
            else:
                metric_values = metrics[metric_name]        
            if key not in selected_methods:
                continue
            method_metrics[key].append(metric_values)
        tasks_metrics[task] = method_metrics

    # max_sz = max(len(l) for l in method_metrics.values())

    # method_metrics_flat = {k: [v3 for v2 in v for v3 in (v2[:10] + [np.nan] * max(0, 10 - len(v2)))] for k, v in method_metrics.items()}
    # method_metric_ranks = get_avg_ranks(method_metrics_flat)

    # setup_names = list(setup_score_values.keys())

    # method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
    plt.ioff()

    method_names = list(selected_methods.keys())

    import math
    num_rows = math.ceil(len(benchmark) / num_in_row)

    fig, axes = plt.subplots(num_rows, num_in_row, figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=-0.1)
    handles_ = []

    for i, task in enumerate(benchmark):
        ax = axes[i // num_in_row, i % num_in_row]
        ax.set_title(task.upper(), fontsize=10, pad=2)

        method_metrics = tasks_metrics[task]

        for key in method_names:
            metrics = method_metrics[key]
            metric_values = np.array(metrics) * 100
            mean = np.mean(metric_values, axis=0)
            confidence_level = 0.95
            degrees_freedom = metric_values.shape[0] - 1
            sample_standard_error = stats.sem(metric_values, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            method_settings = selected_methods[key]
            # default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
            default_args = dict(marker='o', markersize=0, linewidth=1.5, color = method_settings['color'])
            # shift = ((i - len(method_metrics) // 2) * 0.025)
            show_err = True 
            if key[0] == 'denoise':
                default_args['linestyle']='--'
                default_args['linewidth']=0.9
                default_args['markersize'] = 0
                show_err = False 
            if key[0] == 'rand':
                default_args['linestyle']='-.'
                default_args['linewidth']=0.9
                default_args['markersize'] = 0
                show_err = False
            # xs = np.arange(len(mean)) + 1
            xs = np.arange(len(mean)) + 1
            line = ax.plot(xs, mean, zorder=1, **default_args)
            if show_err:
                ax.fill_between(xs, min_v, max_v, alpha=.1, color = line[0].get_color(), linewidth=0)
            if i == 0:
                handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))

        ax.set_xticks([2,4,6,8,10])
        if (i // num_in_row) != (num_rows - 1):  # Not in the last row
            ax.set_xticklabels([])
        else:            
            ax.set_xticklabels([2,4,6,8,10])
            ax.xaxis.set_tick_params(pad=1)
        ax.yaxis.set_tick_params(pad=1)

        if (i // num_in_row) == (num_rows - 1):
            ax.set_xlabel('Epoch', fontsize=10, labelpad = 2)

        if (i % num_in_row) == 0:
            ax.set_ylabel('Accuracy \\%', fontsize=10, labelpad = 2)

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
          
    
    handles_.sort(key = lambda x: x[2])
    ordered_handles = [h[0] for h in handles_]
    ordered_labels = [h[1] for h in handles_]
    # plt.xlabel('Epoch', fontsize=20)
    # plt.ylabel('Accuracy, \\%', fontsize=20)
    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
                ncol=len(ordered_handles),  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, -0.01)  # Adjust position (centered below the grid)
        )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)
    fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)
    outfile = os.path.join(base_path, "plots", f"fig-{metric_name}-{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  

agg_methods = {
    "mean": mean_matrix_score,
    "mean-c": partial(mean_matrix_score, use_correct = True),
    # "mean-i": partial(mean_matrix_score, use_correct = False),
    "rank": rank_matrix_score,
    "rank-c": partial(rank_matrix_score, use_correct = True), 
    # "rank-i": partial(rank_matrix_score, use_correct = False),
    "vote": partial(rank_matrix_score, rank_score_fn = vote_matrix_score),
    "vote-c": partial(rank_matrix_score, rank_score_fn = vote_matrix_score, use_correct = True),

    "rmin": partial(rank_matrix_score, rank_score_fn = min_matrix_score),
    "rmin-c": partial(rank_matrix_score, rank_score_fn = min_matrix_score, use_correct = True),


    # "median": median_matrix_score,
    # "maj": maj_matrix_score,
    # "cset": cset_matrix_score,
    # "cset-2": partial(cset_matrix_score, both_sides = True),
    # "min": mean_min_matrix_score,
    # "min-20": partial(mean_min_matrix_score, min_ratio=0.2),
    
    # "cmean": confident_matrix_score,
    
    # "crank": partial(confident_matrix_score, base_method_fn=rank_matrix_score),
    # # "ccset": partial(confident_matrix_score, base_method_fn=commonset_matrix_score),
    # # "ccsset": partial(confident_matrix_score, n_confident = 100, base_method_fn=partial(commonsubset_matrix_score, vote_ratio = 0.3)),
    # "dir": dir_matrix_score,
    # "cset": partial(commonset_matrix_score, vote_ratio = 0.2),
    # "csset": partial(commonsubset_matrix_score, vote_ratio = 0.3),
    # # "csmi": partial(csmi_matrix_score, vote_ratio = 0.5, descending = False)


    # "mean_10": partial(mean_matrix_score, trim_ratio=0.1),
    # "mean_50": partial(mean_matrix_score, trim_ratio=0.5),
    # "commonset1": partial(commonset_matrix_score, vote_ratio = 0.1),
    # "commonset-80": partial(commonset_matrix_score, vote_ratio = 0.8),
    # "commonset3": partial(commonset_matrix_score, vote_ratio = 0.3),
    # "commonset4": partial(commonset_matrix_score, vote_ratio = 0.4),
    # "commonsubset1": partial(commonsubset_matrix_score, vote_ratio = 0.1, descending = False),
    # "commonsubset2": partial(commonsubset_matrix_score, vote_ratio = 0.2, descending = False),
    # "commonsubset3": partial(commonsubset_matrix_score, vote_ratio = 0.3, descending = False),
    # "commonsubset-60": partial(commonsubset_matrix_score, vote_ratio = 0.6, descending = False),
    # "commonsubset-40r": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = True),
    # "commonsubset-40rr": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = False),
    # "silhouette": cluster_matrix_score
}

def compute_ndr_metrics_table(base_dir_path: str, task='qnli', 
                                    group_file: str = "./groups.json",
                                    infl_methods = ['hf'],
                                    agg_method_names: list[str] = ["mean"],
                                    include_total = True, levels = [30],
                                    m_prefix = "m_bl", i_prefix="i_bl", ndr_prefix = "ndr_bl",
                                    save_df = True, device = "cuda",
                                    noise_hist_bins = 10):

    if group_file != '' and group_file is not None:
        import re
        with open(os.path.join(base_dir_path, group_file), "r") as f:
            module_groups_regex = json.load(f)
        module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}
    else:
        module_groups_patterns = {}

    task_in_dir = os.path.join(base_dir_path, task)

    metric_by_method = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of dict of metrics
    # f30_score_by_infl = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
    # curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(task_in_dir)

    inflset_logits_dict = load_m_info(task_in_dir, m_prefix)

    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith(i_prefix):
            continue
        file_name = file_name_with_ext.split('.')[0]
        fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
        *method_parts, _, run_id_str = fine_name_parts
        # run_id = int(run_id_str)
        infl_method = '_'.join(method_parts)
        if infl_method not in infl_methods:
            continue
        file_path = os.path.join(task_in_dir, file_name_with_ext)
        matrix_dict = torch.load(file_path)
        module_names = list(matrix_dict.keys())
        module_and_group_names = list(module_names)
        # transpose all matrices 
        for module_name in module_names:
            matrix_dict[module_name] = matrix_dict[module_name].t().to(device) # first dim is train_sample now and second is infl val sample
        if len(module_names) == 0:
            continue
        all_interactions = torch.stack([matrix_dict[module_name] for module_name in module_names], dim = 1)
        for int_matrix in matrix_dict.values():
            del int_matrix
        # now the dimensions is train_sample * module * infl_val_sample

        # create module views
        module_interactions = [all_interactions[:, module_id:module_id+1] for module_id in range(len(module_names))]
        group_modules = defaultdict(list)
        for group_name, pattern in module_groups_patterns.items():
            for module_id, module_name in enumerate(module_names):
                if pattern.match(module_name):
                    group_modules[group_name].append(module_id)

        group_names = list(group_modules.keys())
        module_and_group_names.extend(group_names)
        for group_name in group_names:
            group_module_ids = torch.tensor(group_modules[group_name], device = device)
            module_interactions.append(all_interactions[:, group_module_ids])
            del group_module_ids

        if include_total:
            module_interactions.insert(0, all_interactions)
            module_and_group_names.insert(0, 'total')
        
        noise_list = noise_list_dict[run_id_str]
        num_noise = sum(noise_list)
        num_clean = len(noise_list) - num_noise
        filter_idxs = torch.tensor([ round(level * len(noise_list) / 100) for level in levels], device = device)
        # first_30_idx = round(filter_perc * len(noise_list))
        ideal_area = num_noise / 2 + num_clean

        noise_mask = torch.tensor(noise_list, device = device)
        noise_mask_cpu = noise_mask.cpu().float()
        trainset_labels = torch.tensor(trainset_labels_dict[run_id_str], device = device)
        inflset_labels = torch.tensor(inflset_labels_dict[run_id_str], device = device)
        inflset_logits = inflset_logits_dict[run_id_str].to(device)
        inflset_preds = torch.argmax(inflset_logits, dim = -1)
        correct_infl_preds = inflset_preds == inflset_labels
        
        scores = torch.zeros((len(agg_method_names), len(module_interactions), all_interactions.shape[0]), device = device)

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for matrix_id, inf_matrix in enumerate(module_interactions):
                agg_method_fn = agg_methods[agg_method_name] 
                new_scores = agg_method_fn(inf_matrix, noise_mask = noise_mask, 
                                       trainset_labels = trainset_labels, inflset_labels = inflset_labels, 
                                       inflset_logits = inflset_logits, 
                                       correct_infl_preds = correct_infl_preds, task = task, run_id = run_id_str)
                scores[agg_method_id, matrix_id] = new_scores
                del new_scores
                torch.cuda.empty_cache() 

        if noise_hist_bins is not None:
            hists, bin_sizes = compute_ndr_histogram(scores, noise_mask_cpu, bins = noise_hist_bins)            
        else:
            hists = None
            bin_sizes = None
                
        del trainset_labels, inflset_labels
                
        train_ids = torch.argsort(scores, dim = -1)

        noise_tensor = noise_mask[train_ids]

        del noise_mask, noise_mask_cpu

        noise_detection_curves = torch.cumsum(noise_tensor, dim = -1, dtype = torch.float)
        noise_detection_curves /= num_noise

        auc_ndrs = noise_detection_curves.sum(dim = -1) 
        auc_ndrs /= ideal_area

        ndr_at_levels = noise_detection_curves[:, :, filter_idxs]

        auc_ndrs_cpu = auc_ndrs.cpu()
        ndr_at_levels_cpu = ndr_at_levels.cpu()
        del auc_ndrs, ndr_at_levels, noise_detection_curves, scores, train_ids

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for module_id, module_name in enumerate(module_and_group_names):
                one_metrics = {level: ndr_at_levels_cpu[agg_method_id, module_id, level_i].item()  for level_i, level in enumerate(levels) }
                one_metrics["auc_ndr"] = auc_ndrs_cpu[agg_method_id, module_id].item()                
                if hists is not None:
                    for bin_id in range(noise_hist_bins):
                        one_metrics[f"hist_y_{bin_id}"] = hists[agg_method_id, module_id, bin_id].item()
                        one_metrics[f"hist_x_{bin_id}"] = bin_sizes[agg_method_id, module_id, bin_id].item()
                metric_by_method[(infl_method, agg_method_name, module_name)].append(one_metrics)
        torch.cuda.empty_cache() 

    # first_30_mean_std = {key: (np.mean(first_30_values), np.std(first_30_values)) for key, first_30_values in f30_score_by_infl.items() }
    # first_30_ranks = get_avg_ranks(f30_score_by_infl, ascending=False)

    # auc_rocs_mean_std = {key: (np.mean(auc_ndr), np.std(auc_ndr)) for key, auc_ndr in auc_ndr_by_infl.items() }
    # auc_ndr_ranks = get_avg_ranks(auc_ndr_by_infl, ascending=False)

    # sorted_method_keys = sorted(first_30_ranks.keys(), key = lambda x: (first_30_ranks[x], x))

    rows = []
    # for sk in sorted_method_keys:
    for key, metrics_list_of_dict in metric_by_method.items():
        infl_method, agg_method_name, module_name = key
        if module_name in module_groups_patterns or module_name == 'total':
            module_layer, module_simple_name = module_name, "all"
        else:
            module_layer, module_simple_name = get_simple_module_and_layer_name(module_name)
        for run_id, metrics_dict in enumerate(metrics_list_of_dict):
            # auc_ndr = auc_ndr_by_infl[key][run_id]
            # f30_mean, f30_std = first_30_mean_std[sk]
            # auc_ndr_mean, auc_ndr_std = auc_rocs_mean_std[sk]
            # f30_rank = first_30_ranks[sk]
            # auc_ndr_rank = auc_ndr_ranks[sk]
            row_data = {
                "task": task,
                "infl": infl_method,
                "agg": agg_method_name,
                "layer": module_layer,
                "module": module_simple_name,
                "run_id": run_id,
                **metrics_dict
            }
            rows.append(row_data)

    df = pd.DataFrame(rows).set_index(["task", "infl", "agg", "layer", "module", "run_id"])    

    if save_df:
        outfile = os.path.join(base_dir_path, f"{ndr_prefix}_{task}.pcl")
        df.to_pickle(outfile)
        print(f"Saved ndr metrics to {outfile}")

    return df

def process_ndr_table(base_path: str, tasks: list[str] = benchmark, output_ranks = False, with_row_id = False,
                        metric_name = 30, ndr_prefix = "ndr_bl", layers = None,
                        best_group_by = None, custom_suffix = ""): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    dfs = [ pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl")) for task in tasks ]
    df = pd.concat(dfs, ignore_index=False)

    if layers is not None:
        df = df.loc[df.index.get_level_values('layer').isin(layers)]
        df = df.loc[df.index.get_level_values('module') == "all"]
        pass

    metric_df = df.reset_index().pivot(index=["infl", "agg", "layer", "module"], columns=["task", "run_id"], values=[metric_name])

    ranks = metric_df.rank(axis = 0, ascending=False, method="average")
    
    suffix = ""
    if output_ranks:
        metric_df = ranks
        suffix = "-rank"

    metric_by_ds_mean = metric_df.groupby(level=1, axis=1).mean()
    metric_by_ds_std = metric_df.groupby(level=1, axis=1).std()
    metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
    metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
    rank_columns = ["rank", "rank_std"]
    metric_by_ds["rank"] = ranks.mean(axis=1)
    metric_by_ds["rank_std"] = ranks.std(axis=1)

    if best_group_by is not None:
        best_idxs = metric_by_ds.groupby(level=best_group_by, axis=0)['rank'].idxmin()
        metric_by_ds = metric_by_ds.loc[best_idxs]
        metric_df = metric_df.loc[best_idxs]
        ranks = metric_df.rank(axis = 0, ascending=False, method="average")
        metric_by_ds["rank"] = ranks.mean(axis=1)
        metric_by_ds["rank_std"] = ranks.std(axis=1)
    
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True])

    metric_by_ds = metric_by_ds[[*[de for d in tasks for de in [d, d + "_std"]], *rank_columns]]
    # metric_by_ds.to_csv(f"{base_path}/{metric_name}{suffix}-avg.csv")

    values_to_highlight = metric_by_ds[tasks].to_numpy().max(axis=0)

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        infl_method = row["infl"]
        agg_method = row["agg"]
        layer = row["layer"]
        module = row["module"].replace("embed_tokens", "WE")
        if with_row_id:
            new_row["id"] = (row_id + 1)
        new_row["infl"] = infl_method
        new_row["agg"] = agg_method
        find_layer = re.match(r"layers\s(\d+)\s", module)
        if find_layer is not None:
            layer = find_layer.group(1)
            module = module.replace(find_layer.group(0), "")
        new_row["layer"] = "WE" if module == "WE" else layer
        new_row["module"] = module
        for did, d in enumerate([*tasks, "rank"]):
            should_highlight = False
            if d == "rank":
                m = round(row[d] * 10) / 10
                m_std = round(row[d + "_std"] * 10) / 10
            else:
                should_highlight = row[d] == values_to_highlight[did]
                m = round(row[d] * 1000) / 10
                m_std = round(row[d + "_std"] * 1000) / 10
            m = str(m).rstrip("0").rstrip(".").lstrip("0").replace("-0.", "-.")
            m_std = str(m_std).rstrip("0").rstrip(".").lstrip("0")

            if should_highlight:
                if m_std == "":
                    new_row[d] = f"\\textbf{{{m}}}"
                else:
                    new_row[d] = f"\\textbf{{{m}}} $\pm$ {m_std}"
            else:
                if m_std == "":
                    new_row[d] = m
                else:
                    new_row[d] = f"{m} $\pm$ {m_std}"
        rows.append(new_row)

    with open(f"{base_path}/ndr-{metric_name}{suffix}{custom_suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")\
            .replace("_10", "$^{10}$").replace("_50", "$^{50}$")\
            .replace("commonset", "cset").replace("-20", "$^{20}$").replace("-30", "$^{30}$")\
            .replace("commonsubset", "csset").replace("out_proj", "proj")\
            .replace('self_attn ', '')\
            .replace('v_proj', 'value').replace('q_proj', 'query')
        print(s, file = stats_file)

    pass


def process_ndr_table2(base_path: str, tasks: list[str],
                        infl_methods = [ 'datainf', 'hf', 'cos'],
                        layers = [], agg_name = "mean",
                        ndr_prefix = "ndr_bl", num_bins = 5, 
                        hist_w_delta = 1, hist_w = 10, hist_w_min = 0.5,
                        hist_gap = 0.2,
                        suffix = ""): 
    
    task_rows = []

    for task in tasks:

        df = pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl"))

        full_infl_methods = [mx for m in infl_methods for mg in [["hf_we_", "hf_we_topk_10", "hf"] if m == "hf" else [m]] for mx in mg]

        selected_df = df.loc[task, full_infl_methods, agg_name, layers, 'all']
        df = selected_df.reset_index().drop(columns=['task', 'agg', 'module'])
        df2 = df.groupby(["infl", "layer"]).mean()

        max_noise_filtering = df2[[c for c in df2.columns if type(c) is str and c.startswith("hist_y_") ]].max().max()
        

        datainf_ids = [('datainf', l) for l in layers if l != 'WE']
        hf_ids = [('hf_we_', 'WE'), ('hf_we_topk_10', 'WE'), *(('hf', l) for l in layers)]
        cos_ids = [('cos', l) for l in layers]
        
        datainf_df = df2.loc[datainf_ids]
        hf_df = df2.loc[hf_ids]
        cos_df = df2.loc[cos_ids]

        method_names = ['DataInf', 'TracIn', 'Cosine']
        subtables = []    

        for m_df_i, m_df in enumerate([datainf_df, hf_df, cos_df]):

            head_row = f"{task.upper()} & \multicolumn{{3}}{{l}}{{{method_names[m_df_i]}}}"

            rows = []

            for row_id, row in enumerate(m_df.reset_index().to_dict(orient="records")):
                layer = row['layer']
                if layer == 'WE' and m_df_i == 1 and row_id == 0:
                    layer = "TracIn$_{we}$"
                if layer == 'WE' and m_df_i == 1 and row_id == 1:
                    layer = "TracIn$^{10}_{we}$"
                ndr_30 = round(row[30] * 100)
                auc_ndr = row["auc_ndr"]
                hist_y = []
                hist_x = []
                for i in range(num_bins):
                    y = row[f"hist_y_{i}"]
                    x = row[f"hist_x_{i}"]
                    hist_y.append(y)
                    hist_x.append(x)

                min_x = np.min(hist_x)
                max_x = np.max(hist_x)
                if min_x == max_x: 
                    ws = np.ones_like(hist_x)
                ws = (np.array(hist_x) - min_x) * hist_w_delta / (max_x - min_x) + hist_w_min
                ws_sum = np.sum(ws)
                ws = ws / ws_sum * hist_w
                hs = np.array(hist_y)
                # max_h = np.max(hs)
                # hs = hs / max_noise_filtering * 100
                hs = hs * 100 / max_noise_filtering
                bars = []
                cur_x = 0

                max_density = np.max(hs / np.array(hist_x))

                for i, (w, y) in enumerate(zip(ws, hs)):
                    density = round(60 + ((y / hist_x[i]) / max_density) * 40)
                    density = 100 if density > 95 else density
                    bars.append(f"\\fill[gray!{density}] ({cur_x},0) rectangle +({w},{y});")
                    cur_x += w 
                    cur_x += hist_gap
                # bars.append(f"\\path [draw=white] (0, 0) -- ({cur_x - hist_gap}, 0);")
                all_bars = "\n".join(bars)
                tikz_bars = f"\\begin{{tikzpicture}}[x=3pt,y=0.15pt,baseline=(current bounding box.center)]\n{all_bars}\n\\end{{tikzpicture}}\n"
                line = f"{layer} & {ndr_30:.0f} & {auc_ndr:.2f} & {tikz_bars}"
                rows.append(line)

            subtables.extend([head_row, *rows])
            # subtable = head_row + "\\\\ \\hline\n" + "\\\\\n".join(rows) + "\\\\ \\hline\n"
            # subtables.append(subtable)

        header_main = "Layer & NDR & AUC & Noise"
        full_table = [header_main, *subtables]
        task_rows.append(full_table)

    all_rows = [ " & ".join(cols) for cols in zip(*task_rows) ]
    head_of_heads, others = all_rows[0], all_rows[1:]

    full_table = head_of_heads + "\\\\ \\hline\n" + "\\\\\n".join(others) + "\\\\ \\hline\n"

    with open(f"{base_path}/ndr2-{suffix}.tex", "w") as stats_file:
        print(full_table, file = stats_file)

    pass

def draw_one_noise_distr(ax, plot_data, bar_w_delta = 0.1, bar_w = 10, min_w = 0.5, max_y = 100, auc_ndr_max = 1):
    auc_ndr = plot_data["auc_ndr"]
    ndr_30 = plot_data[30]
    ys = plot_data[[c for c in plot_data.index if type(c) is str and c.startswith("hist_y_")]].to_numpy()
    ws = plot_data[[c for c in plot_data.index if type(c) is str and c.startswith("hist_x_")]].to_numpy()    
    ws *= bar_w
    ws = np.where(ws < min_w, min_w, ws)
    ws_sum = np.sum(ws)
    ws = ws / ws_sum * bar_w
    xs = np.insert(np.cumsum(ws), 0, 0)[:-1]
    xs += np.arange(len(xs)) * bar_w_delta
    ys *= 100
    ax.set_ylim(0, max_y)
    ax.bar(xs, ys, ws, align='edge', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    auc_ndr_str = f"{auc_ndr:.2f}".lstrip("0")
    if auc_ndr == auc_ndr_max:
        auc_ndr_str = f"\\textbf{{{auc_ndr_str}}}"
    ax.text(0.98, 0.95, f"AUC={auc_ndr_str}", ha='right', va='top', fontsize=6, transform=ax.transAxes, color='black')

    pass

def reset_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    pass

def draw_noise_distr(base_path: str, tasks: list[str] = benchmark,
                        infl_methods = [ 'datainf', 'hf', 'cos'],
                        layers = [], agg_name = "mean", figsize = (8, 8),
                        ndr_prefix = "ndr_bl", num_bins = 10, 
                        suffix = ""): 
    ''' Draws hists: y-ax is dataset, method, x-ax - layers '''

    full_infl_methods = [mx for m in infl_methods for mg in [["hf_we_", "hf_we_topk_10", "hf"] if m == "hf" else [m]] for mx in mg]

    dfs = [ pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl")) for task in tasks ]
    df = pd.concat(dfs, ignore_index=False)

    df = df.loc[tasks, full_infl_methods, agg_name, layers, 'all']
    df = df[[30, "auc_ndr", *(f"hist_x_{i}" for i in range(num_bins) ), *(f"hist_y_{i}" for i in range(num_bins) )]]

    df = df.reset_index().drop(columns=['agg', 'module', 'run_id'])

    grouped_df = df.groupby(["task", "infl", "layer"])
    mean_df = grouped_df.mean()
    min_df = grouped_df.min()
    max_df = grouped_df.max()

    plt.ioff()
    simple_methods = ['datainf', 'hf', 'cos']
    if 'WE' in layers:
        column_methods = [
            ['', '', '', *(['datainf'] * (len(layers) - 1))], #datainf
            ['hf_we_', 'hf_we_topk_10', *(['hf'] * len(layers))], #hf
            ['', '', *(['cos'] * len(layers))], #cos
        ]
        column_layers = ['WE', 'WE', *layers]
    else:
        column_methods = [
            ['datainf'] * len(layers),
            ['hf'] * len(layers),
            ['cos'] * len(layers)
        ]
        column_layers = layers
    fig, axes = plt.subplots(len(tasks) * len(column_methods), len(column_layers), figsize=figsize)

    for i, task in enumerate(tasks):
        task_df = mean_df.loc[task]
        ys_max = task_df[[c for c in task_df.columns if type(c) is str and c.startswith("hist_y_")]].max().max() * 100
        ys_max += 5
        for j, methods in enumerate(column_methods):
            mthd = simple_methods[j]
            if mthd == 'hf':
                mthd = ['hf_we_', 'hf_we_topk_10', 'hf']
            else:
                mthd = [mthd]
            auc_ndr_max = mean_df.loc[task].loc[mthd]['auc_ndr'].to_numpy().max()
            for k, (method, layer) in enumerate(zip(methods, column_layers)):
                ax = axes[i * len(column_methods) + j, k]
                if method == '':
                    reset_plot(ax)
                    continue
                plot_data = mean_df.loc[task, method, layer]
                draw_one_noise_distr(ax, plot_data, max_y=ys_max, auc_ndr_max = auc_ndr_max)
    
    # fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
    #             ncol=len(ordered_handles),  # Arrange all legend items in one row
    #             bbox_to_anchor=(0.5, -0.01)  # Adjust position (centered below the grid)
    #     )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)
    # fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)

    fig_delta = 0.02
    fig_sz = 1 - fig_delta * 2

    for i in range(1, len(tasks)):
        y_pos = fig_delta + fig_sz * (i / len(tasks))
        fig.add_artist(plt.Line2D([0, 1], [y_pos, y_pos], color='black', linewidth=1))

    for i, task in enumerate(tasks):    
        # y_start = i / len(tasks)
        y_middle = (1 - fig_delta) - ((i + 0.5) / len(tasks)) * fig_sz
        y_start = (1 - fig_delta) - (i / len(tasks)) * fig_sz
        y_width = fig_sz / len(tasks)
        fig.text(0.00, y_middle, f'{task.upper()}', ha='left', va='center', rotation='vertical', fontsize=10)
        for j, method in enumerate(['DataInf', 'TracIn', 'Cosine']):
            m_width = y_width / 3
            m_middle = y_start - (j + 0.5) * m_width
            fig.text(1, m_middle, f'{method}', ha='right', va='center', rotation=270, fontsize=6)

            if j != 0:
                y_line = y_start - j * m_width

                for i in range(1, len(tasks)):
                    fig.add_artist(plt.Line2D([fig_delta, 1], [y_line, y_line], color='gray', linewidth=0.5, linestyle='--'))

    layer_names = ['TracIn$_{we}$','TracIn$^{10}_{we}$', *layers]
    for i, layer_name in enumerate(layer_names):
        
        w = fig_sz / len(layer_names)
        w_middle = fig_delta + (i + 0.5) * w
        fig.text(w_middle, 1, layer_name, ha='center', va='top', fontsize=8)
        fig.text(w_middle, 0.00, layer_name, ha='center', va='bottom', fontsize=8)
        if i > 0:
            x_line = fig_delta + i * w
            fig.add_artist(plt.Line2D([x_line, x_line], [0, 1], color='gray', linewidth=0.5, linestyle='--'))


    # fig.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
    fig.tight_layout(pad = 0, rect=(fig_delta, fig_delta, 1 - fig_delta, 1 - fig_delta))
    outfile = os.path.join(base_path, "plots", f"noise-{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  

    pass



def draw_ndr_curve(ys: np.ndarray, xs:np.ndarray, ylegend:list[str], title:str, outfile: str,
                    xaxis_line = 30, noise_ratio = 20): 
    ''' 
        ys - 3d, method * levels * run_ids
        xs - measure points, len(xs) == len(values)
        len(ylegend) == len(method)

        xs and ys are in range [0, 100]
    '''
    plt.ioff()
    xs = np.concatenate([[0], xs, [100]])
    for method_id, method in enumerate(ylegend):
        y = ys[method_id].T
        y_mean = np.nanmean(y, axis=0)
        confidence_level = 0.95
        degrees_freedom = y.shape[0] - 1
        sample_standard_error = stats.sem(y, axis=0, nan_policy = 'omit')
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, y_mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        y_mean = np.concatenate([[0], y_mean, [100]])
        min_v = np.concatenate([[0], min_v, [100]])
        max_v = np.concatenate([[0], max_v, [100]])
        plt.plot(xs, y_mean, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=xaxis_line, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=0.5)
    # best ndr asymptote
    plt.plot([0, noise_ratio], [0, 100], color='gray', linestyle='--', linewidth=0.5)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)', fontsize=20)
    plt.ylabel('Detection Rate (\\%)', fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.clf()      

def draw_ndr_curves(base_path: str, tasks: list[str] = benchmark, levels = [5,10,15,20,25,30,35,40,45,50,60,70,80,90], 
                        ndr_prefix = "ndr_bl",
                        selected_methods: dict[str, str] = {
                            "hf:mean:0:query A": "hf, mean, L0, query A", 
                            "hf:mean:total:all": "hf, mean, total",  
                            "datainf:mean:total:all": "datainf, mean, total",
                        }): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    selected_methods = {tuple(m.split(":")):v  for m, v in selected_methods.items()}
    selected_method_ids = list(selected_methods.keys())

    for task in tasks:

        df = pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl"))
        df = df.reset_index().drop(columns=["task", "auc_ndr"])
        metric_df = df.pivot(index=["infl", "agg", "layer", "module"], columns=["run_id"], values=levels)
        selected_df = metric_df.loc[selected_method_ids]
        ys = selected_df.to_numpy().reshape(len(selected_method_ids), len(levels), -1) * 100.0

        outfile = os.path.join(base_path, "plots", f"ndr-curve-{task}.pdf")
        draw_ndr_curve(ys, levels, list(selected_methods.values()), f"{task.upper()}", outfile)


    pass

def draw_tun2_bar_metric(base_path: str, tasks:list[str] = benchmark, metric_name: str = "accuracy",
                         selected_methods: dict = {}, from_method = None, suffix = "", title = None):
    all_task_means = []
    all_task_std = []
    categories = [v for k, v in selected_methods.items() if k != from_method] 
    for task in tasks:
        task_means = []
        task_std = []
        infile = os.path.join(base_path, f"{task}-bl.jsonlist")
        with open(infile, 'r') as f:
            json_lines = f.readlines()
        all_metrics = [json.loads(l) for l in json_lines]
        method_metrics = defaultdict(list)
        for metrics in all_metrics:
            infl_method = metrics['config']['infl_method']
            agg_method = metrics['config']['agg_method']                
            module_name = metrics['config']['module_name']
            key = (infl_method, agg_method, module_name)
            if key not in selected_methods:
                continue

            # before_metric_values = metrics['first_finetune'][metric_name] 
            # after_metric_values = metrics[metric_name]
            # metric_value = np.max([a - b for a, b in zip(after_metric_values, before_metric_values)])

            before_metric_value = np.max(metrics['first_finetune'][metric_name] )
            after_metric_value = np.max(metrics[metric_name])
            metric_value = after_metric_value - before_metric_value

            method_metrics[key].append(metric_value)

        delta = None 
        if from_method is not None:
            delta = np.array(method_metrics[from_method])

        for method_id, key in enumerate(selected_methods.keys()):
            if key == from_method:
                continue
            metrics = method_metrics[key]
            if delta is not None:
                metric_values = (np.array(metrics) - delta[:len(metrics)]) * 100
            else:
                metric_values = np.array(metrics) * 100
            # mean = np.mean(metric_values)            
            # std = np.std(metric_values)
            mean = np.mean(metric_values)            
            std = 0

            task_means.append(mean)
            task_std.append(std)
        all_task_means.append(task_means)
        all_task_std.append(task_std)

    all_task_means = np.array(all_task_means)
    all_task_std = np.array(all_task_std)

    xs = np.arange(len(tasks)) * 2.5

    bar_width = 0.3

    plt.ioff()

    for i, category in enumerate(categories):
        plt.bar(xs + i * bar_width, 
                all_task_means[:, i], 
                # yerr=all_task_std[:, i], 
                width=bar_width, 
                # error_kw=dict(lw=0.5, capsize=1, capthick=1),
                label=category)

    # plt.xlabel('Datasets', fontsize=20)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.ylabel('$\\Delta$ Acc, \\%', fontsize=15)
    plt.xticks(xs + bar_width * (len(categories) - 1) / 2, tasks, rotation=45, fontsize=10)  # Center group labels
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    if title:
        plt.title(title, fontsize=15)
    # plt.tight_layout()
    outfile = os.path.join(base_path, "plots", f"{metric_name}{suffix}-diffs.pdf")
    plt.savefig(outfile)  
    plt.clf()  

def draw_all_tun2_box_metric(base_path: str, tasks:list[str] = benchmark, metric_name: str = "best_accuracy_1", figsize = (8, 5),
                             agg_method = "mean", num_in_rows = 4,
                             layers: list[str] = [], suffix = ""):

    df = get_all_df(base_path, tasks)
    df = df[df["agg_method"] == agg_method].drop(columns=["agg_method"])
    metrics_per_ds = df.pivot(index=["task", "seed"], columns=["infl_method", "module"], values=metric_name)
    metrics_per_ds = metrics_per_ds * 100.0
    # metric_by_ds_mean = metrics_per_ds.groupby(level=0, axis=1).mean() * 100

    # datainf_df = metrics_per_ds.loc[('datainf', agg_method, layers)]
    # hf0_df = metrics_per_ds.loc[('hf_we_', agg_method, ['WE'])]
    # hf1_df = metrics_per_ds.loc[('hf_we_topk_10', agg_method, ['WE'])]
    # hf2_df = metrics_per_ds.loc[('hf', agg_method, layers)]
    # hf_df = pd.concat([hf0_df, hf1_df, hf2_df], axis=0)
    # cos_df = metrics_per_ds.loc[('cos', agg_method, layers)]
    # pass

    # datainf_data = datainf_df.to_numpy()
    # import math 
    # datainf_min = math.floor(datainf_data.min() / 5) * 5
    # datainf_max = math.ceil(datainf_data.max() / 5) * 5
    # hf_data = hf_df.to_numpy()
    # hf_min = math.floor(hf_data.min() / 5) * 5
    # hf_max = math.ceil(hf_data.max() / 5) * 5
    # cos_data = cos_df.to_numpy()
    # cos_min = math.floor(cos_data.min() / 5) * 5
    # cos_max = math.ceil(cos_data.max() / 5) * 5

    # categories_list = [[v for k, v in ms.items()] for ms in selected_methods] 
        

    # bar_width = 0.3

    # xs = 3 * (np.arange(len(tasks)) + 1)
    plt.ioff()    

    # methods = [datainf_data, hf_data, cos_data] 
    # methods_name = ['DataInf', 'TracIn', 'Cosine']
    # method_mins = [datainf_min, hf_min, cos_min]
    # method_maxs = [datainf_max, hf_max, cos_max]
    
    import math
    num_rows = math.ceil(len(tasks) / num_in_rows)
    fig, axes = plt.subplots(num_rows, num_in_rows, figsize=figsize)

    width = 1
    for i, task in enumerate(tasks):
        ax = axes[i // num_in_rows, i % num_in_rows]
        ax.set_title(task.upper(), fontsize=10, pad = 2)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        # first column 
        # if i % num_in_rows == 0:
        #     ax.set_ylabel('Accuracy \\%', fontsize=15, labelpad = 2)
        
        data = metrics_per_ds.loc[task].T

        datainf_np = data.loc[[('datainf', l) for l in layers if l != 'WE']].to_numpy().tolist()
        datainf_np = [[y for y in d if not np.isnan(y)] for d in datainf_np]
        hf0_np = data.loc[('hf_we_', 'WE')].to_numpy().tolist()
        hf1_np = data.loc[('hf_we_topk_10', 'WE')].to_numpy().tolist()
        hf2_np = data.loc[[('hf', l) for l in layers]].to_numpy().tolist()
        hf_np = [hf0_np, hf1_np, *hf2_np]
        hf_np = [[y for y in d if not np.isnan(y)] for d in hf_np]
        cos_np = data.loc[[('cos', l) for l in layers]].to_numpy().tolist()
        cos_np = [[y for y in d if not np.isnan(y)] for d in cos_np]

        colors = plt.cm.Set3.colors 

        def set_props(bplot, delta):
            for i, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(colors[(i + delta) % len(colors)])
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)
            for whisker in bplot['whiskers']:
                whisker.set_linewidth(0.5)
                whisker.set_color('black')
            for cap in bplot['caps']:
                cap.set_linewidth(0.5)
                cap.set_color('black')

            for line in bplot['medians']:
                line.set_linewidth(0)
                line.set_color('black')                

        bplot1 = ax.boxplot(datainf_np, positions=np.arange(1, len(datainf_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot1, 3)

        ax.plot(np.arange(1, len(datainf_np) +  1), [np.mean(d) for d in datainf_np], color='black', linestyle='--', linewidth=0.5)

        ax.axvline(x=len(datainf_np) + 1.5, color='gray', linestyle='--', linewidth=0.5)

        bplot2 = ax.boxplot(hf_np, positions= len(datainf_np) + 2 + np.arange(1, len(hf_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot2, 0)

        ax.plot(len(datainf_np) + 2 + np.arange(1, len(hf_np) +  1), [np.mean(d) for d in hf_np], color='black', linestyle='--', linewidth=0.5)

        ax.axvline(x=len(datainf_np) + 2 + len(hf_np) + 1.5, color='gray', linestyle='--', linewidth=0.5)

        bplot3 = ax.boxplot(cos_np, positions= len(datainf_np) + 2 + len(hf_np) + 2 + np.arange(1, len(cos_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot3, 2)

        ax.plot(len(datainf_np) + 2 + len(hf_np) + 2 + np.arange(1, len(cos_np) +  1), [np.mean(d) for d in cos_np], color='black', linestyle='--', linewidth=0.5)

        ax.set_xticks([(len(datainf_np) + 1) / 2, len(datainf_np) + 2 + (len(hf_np) + 1) / 2, len(datainf_np) + 2 + len(hf_np) + 2 + (len(cos_np) + 1) / 2])
        ax.set_xticklabels([])

        # last row 
        if i // num_in_rows == num_rows - 1:
            ax.set_xticklabels(['DataInf', 'TracIn', 'Cosine'], fontsize=8)

        ax.yaxis.set_tick_params(pad=1)
        

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)            

        

    # for i, category in enumerate(categories):
    #     plt.bar(xs + i * bar_width, 
    #             all_task_means[:, i], 
    #             # yerr=all_task_std[:, i], 
    #             width=bar_width, 
    #             # error_kw=dict(lw=0.5, capsize=1, capthick=1),
    #             label=category)

    # plt.xlabel('Datasets', fontsize=20)
    # plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # plt.ylabel('$\\Delta$ Acc, \\%', fontsize=15)

    # fig.legend(['', '', *layers], loc='lower center', fontsize=10,
    #             ncol=len(layers),  # Arrange all legend items in one row
    #             bbox_to_anchor=(0.5, -0.01)  # Adjust position (centered below the grid)
    #     )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)

    colors = [colors[i] for i in range(2 + len(layers))]
    labels = ['TracIn$_{we}$', 'TracIn$^{10}_{we}$', *layers]

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=title) for color, title in zip(colors, labels)]

    fig.legend(handles = legend_handles, loc='lower center', fontsize=10,
                ncol=len(legend_handles),  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, -0.01),
                columnspacing = 0.75  # Adjust position (centered below the grid)
        )

    fig.text(0.01, 0.5, 'Accuracy \\%', ha='center', va='center', rotation='vertical', fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)

    # plt.tight_layout()
    outfile = os.path.join(base_path, "plots", f"box-{metric_name}{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  
    pass

def where_is_the_noise(base_dir_path: str, task: str, infl_method: str, 
                       module_pattern: str, bins = 10,
                       i_prefix = 'i_b', m_prefix = "m_b", device = "cuda"):
    ''' Works with infl matrix, pick corresponding layers, does simple aggregation 
        and builds the histograms of noise distribution for each val sample '''
    module_pattern = re.compile(module_pattern)
    task_in_dir = os.path.join(base_dir_path, task)

    histograms_by_run_ids = defaultdict(list) # run_id -> histogram

    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(task_in_dir)

    inflset_logits_dict = load_m_info(task_in_dir, m_prefix)

    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith(i_prefix):
            continue
        file_name = file_name_with_ext.split('.')[0]
        fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
        *method_parts, _, run_id_str = fine_name_parts
        # run_id = int(run_id_str)
        file_infl_method = '_'.join(method_parts)
        if file_infl_method != infl_method:
            continue
        file_path = os.path.join(task_in_dir, file_name_with_ext)
        matrix_dict = torch.load(file_path)
        file_module_names = list(matrix_dict.keys())
        filtered_module_names = [name for name in file_module_names if module_pattern.match(name)]
        if len(filtered_module_names) == 0:
            continue
        # transpose all matrices 
        filtered_matrix_dict = {}
        for module_name in filtered_module_names:
            filtered_matrix_dict[module_name] = matrix_dict[module_name].to(dtype=torch.float, device = device) # first dim is train_sample now and second is infl val sample
        del matrix_dict
        all_interactions = torch.stack([filtered_matrix_dict[module_name] for module_name in filtered_module_names], dim = 1)
        for int_matrix in filtered_matrix_dict.values():
            del int_matrix
        # now the dimensions is train_sample * module * infl_val_sample

        # create module views
        module_interactions = torch.mean(all_interactions, dim = 1).numpy()
        del all_interactions
        
        # infl_predictions = torch.argmax(inflset_logits_dict[run_id_str], dim=-1).to(device = device).numpy()
        ilogits = inflset_logits_dict[run_id_str].to(dtype=torch.float, device = device).numpy()
        infl_labels = np.array(inflset_labels_dict[run_id_str])
        infl_predictions1 = (ilogits[:, 1] > ilogits[:, 0]).astype(int)
        gold_logits = ilogits[np.arange(ilogits.shape[0]), infl_labels]
        logit_dist = gold_logits - ilogits[np.arange(ilogits.shape[0]), 1 - infl_labels]
        n_confident = min(50, np.sum(logit_dist > 0))
        most_confident_infl_ids = np.argsort(logit_dist)[-n_confident:]
        best_logit_dist = logit_dist[most_confident_infl_ids]
        # trainset_labels = np.array(trainset_labels_dict[run_id_str])
        infl_error_mask1 = (infl_predictions1 != infl_labels)
        # if mid_ground < 1 - probably the NN is badly trained
        infl_error_mask2 = np.ones_like(infl_error_mask1)
        infl_error_mask2[most_confident_infl_ids] = False
        # train_mask = (trainset_labels != infl_labels)
        noise_list = noise_list_dict[run_id_str]
        noise_mask = np.array(noise_list, dtype=float)
        histograms = np.zeros((module_interactions.shape[0], bins), dtype=float)     
        bin_sizes_all = np.zeros((module_interactions.shape[0], bins), dtype=float)     

        for infl_id in range(module_interactions.shape[0]):
            bin_edges = np.quantile(module_interactions[infl_id], np.linspace(0, 1, bins + 1))
            count_in_bin = np.sum((module_interactions[infl_id] >= bin_edges[0]) & (module_interactions[infl_id] <= bin_edges[1]))
            hist, _ = np.histogram(module_interactions[infl_id], bins = bin_edges, weights=noise_mask, density=False)
            histograms[infl_id] = hist
            bin_sizes_all[infl_id] = bin_edges[1:] - bin_edges[:-1]

        hist_mask = ~(histograms[:, 0] > 2 * histograms[:, -1])
        ratio_in_errors1 = np.sum(hist_mask & infl_error_mask1) / np.sum(hist_mask)
        ratio_in_errors2 = np.sum(hist_mask & infl_error_mask2) / np.sum(hist_mask)
        sel_infl_pred = ilogits[hist_mask]
        sel_infl_labels = infl_labels[hist_mask]
        sel_infl_pred_no_err = ilogits[hist_mask & ~infl_error_mask1]
        sel_infl_labels_no_err = infl_labels[hist_mask & ~infl_error_mask1]
        hist_no_err = histograms[hist_mask & ~infl_error_mask1]
        bin_sizes_no_err = bin_sizes_all[hist_mask & ~infl_error_mask1]
        print(f"Ratios {ratio_in_errors1:.2f} {ratio_in_errors2:.2f} {np.sum(~infl_error_mask2)}")
        histograms_by_run_ids[run_id_str].append(histograms)

        torch.cuda.empty_cache() 

    return histograms_by_run_ids

    # first_30_mean_std = {key: (np.mean(first_30_values), np.std(first_30_values)) for key, first_30_values in f30_score_by_infl.items() }
    # first_30_ranks = get_avg_ranks(f30_score_by_infl, ascending=False)

    # auc_rocs_mean_std = {key: (np.mean(auc_ndr), np.std(auc_ndr)) for key, auc_ndr in auc_ndr_by_infl.items() }
    # auc_ndr_ranks = get_avg_ranks(auc_ndr_by_infl, ascending=False)

    # sorted_method_keys = sorted(first_30_ranks.keys(), key = lambda x: (first_30_ranks[x], x))



if __name__ == "__main__":

    # base_path = "data/roberta"
    base_path = "data/llama-0"
    # base_path = "data/mistral"
    group_file = "./groups.json"

    # create_tun2_metric_table(metric_name="noise_30", ds_ranks=False, mul = 100, highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec = 1)
    pass

    # create_tun2_metric_table(metric_name="best_accuracy_1", ds_ranks=False, mul = 100, 
    #                          highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec=1)
    pass

    # run_friedman_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_wilcoxon_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_best_spearman_test(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # where_is_the_noise(base_path, task='qnli', infl_method='datainf', 
    #                    module_pattern=".*\\.layers\\.([4-7])\\..*\\.lora_(A)\\..*",
    #                 #    module_pattern=".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",
    #                    device = 'cpu')
    # pass
    # agg_method_names = ["mean", "rank", "rmin", "vote"]
    agg_method_names = ["rank", "rank-c", "mean", "mean-c", "vote", "vote-c", "rmin", "rmin-c"]
    # dss = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]
    dss = ["mrpc", "qnli", "sst2", "qqp"]
    # dss = ["mrpc"]
    # for ds in dss:
    #     compute_ndr_metrics_table(base_path, task=ds, 
    #                             group_file=group_file, levels=[5,10,15,20,25,30,35,40,45,50,60,70,80,90],
    #                             infl_methods = ['hf', 'cos', 'datainf', 'hf_we_', 'hf_we_topk_10'],
    #                             agg_method_names=agg_method_names)

    roberta_layers = ['WE', '00-05', '06-11', '12-17', '18-23', 'CL']
    llama_layers = ['WE', '00-03', '04-07', '08-11', '12-15', 'CL']
    mistral_layers = ['WE', '00-07', '08-15', '16-23', '24-31', 'CL']
    process_ndr_table(base_path, tasks=dss, with_row_id=False, custom_suffix = "-best", 
                      best_group_by=["infl", "agg"], layers=llama_layers)
    process_ndr_table(base_path, tasks=dss, with_row_id=False,
                        layers=llama_layers)
    # draw_noise_distr(base_path, tasks=benchmark, layers = roberta_layers, suffix="2")
    pass

    # roberta_selected_methods = {
    #         "datainf:commonset-20:CL:all": "datainf, cset$^{20}$, CL", 
    #         "hf:commonset-20:total:all": "hf, cset$^{20}$",
    #         "cos:rank:23:value B": "cos, rank, L23, value B", 
    #         # "hf:mean:total:all": "hf, mean, total",  
    #     }
    # llama_selected_methods = {
    #         "cos:rank::layers 9 self_attn v_proj B": "cos, rank, L8, value B", 
    #         "datainf:rank::layers 7 self_attn v_proj B": "datainf, rank, L7, value B",
    #         "hf:rank:04-07:all": "hf, rank, L4-7", 
    #         # "hf:mean:total:all": "hf, mean, total",  
    #     }    
    # draw_ndr_curves(base_path, selected_methods=llama_selected_methods)
    # pass 

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods={
    #                         # ('denoise', '', ''): "denoise",
    #                         ('hf_we_', 'mean', 'WE'): "hf$_{we}$",
    #                         # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #                         ('hf', 'mean', 'WE'): "hf, WE",
    #                         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                         ('cos', 'mean', '12-17'): "cos, 12-17",
    #                         ('datainf', 'mean', '18-23'): "datainf, 18-23",
    #                         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                         ('rand', '', ''): "rand",
    #                     },
    #                     from_method=("rand", "", ""),
    #                     suffix = "-rand",
    #                     title = "Accuracy difference from random")

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods={
    #                         # ('denoise', '', ''): "denoise",
    #                         ('hf_we_', 'mean', 'WE'): "hf$_{we}$",
    #                         # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #                         ('hf', 'mean', 'WE'): "hf, WE",
    #                         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                         ('cos', 'mean', '12-17'): "cos, 12-17",
    #                         ('datainf', 'mean', '18-23'): "datainf, 18-23",
    #                         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                         ('rand', '', ''): "rand",
    #                     },
    #                     # from_method=("rand", "", ""),
    #                     # suffix = "-rand",
    #                     title = "Accuracy difference from first finetune")    

    roberta_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
        ('hf', 'mean', 'WE'): {'color': 'blue', 'legend_name': 'TracIn, WE', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '12-17'): {'color': 'green', 'legend_name': 'Cosine, 12-17', 'legend_order': 5},
        ('datainf', 'mean', '18-23'): {'color': 'red', 'legend_name': 'DataInf, 18-23', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},
    }

    llama_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): "hf$^{10}_{we}$",
        ('hf', 'mean', '04-07'): {'color': 'blue', 'legend_name': 'TracIn, 04-07', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '04-07'): {'color': 'green', 'legend_name': 'Cosine, 04-07', 'legend_order': 5},
        ('datainf', 'mean', '04-07'): {'color': 'red', 'legend_name': 'DataInf, 04-07', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},
    }

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods=llama_selected_methods,
    #                     from_method=("rand", "", ""),
    #                     suffix = "-rand",
    #                     title = "Accuracy difference from random")

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods=llama_selected_methods,
    #                     # from_method=("rand", "", ""),
    #                     # suffix = "-rand",
    #                     title = "Accuracy difference from first finetune")        
    # pass

    
    mistral_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): "hf$^{10}_{we}$",
        ('hf', 'mean', '00-07'): {'color': 'blue', 'legend_name': 'TracIn, 00-07', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '08-15'): {'color': 'green', 'legend_name': 'Cosine, 08-15', 'legend_order': 5},
        ('datainf', 'mean', '08-15'): {'color': 'red', 'legend_name': 'DataInf, 08-15', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},        
    }

    # draw_all_tun2_metric(base_path, selected_methods=llama_selected_methods)
    # roberta_layers = ['WE', '00-05', '06-11', '12-17', '18-23', 'CL']
    llama_layers = ['WE', '00-03', '04-07', '08-11', '12-15', 'CL']
    # mistral_layers = ['WE', '00-07', '08-15', '16-23', '24-31', 'CL']
    # draw_all_tun2_box_metric(base_path, layers = llama_layers)
    pass


    # for task in benchmark:
    #     draw_tun2_metric(base_path, task, "accuracy", suffix = "best", selected_methods={
    #         ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': -1},
    #         # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #         ('hf_we_topk_10', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #         ('hf', 'mean', '04-07'): {'color': 'blue', 'legend_name': 'hf, 04-07', 'legend_order': 2},
    #         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #         ('cos', 'mean', '04-07'): {'color': 'green', 'legend_name': 'cos, 04-07', 'legend_order': 5},
    #         ('datainf', 'mean', '00-03'): {'color': 'red', 'legend_name': 'datainf, 00-03', 'legend_order': 6},
    #         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #         ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': 8},
    #     })    
    pass
    
