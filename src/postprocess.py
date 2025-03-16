from collections import defaultdict
import json
import os
import pickle
import re
from typing import Optional

import datasets
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy import stats
import torch
from datasets import load_from_disk

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    

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


def draw_ft2_metric(task: str, infile: str, outfile: str, metric = 'accuracy', influence_method = "", module_pattern_to_name = {}, allowed_filter_methods = ['rand', 'denoise', 'none']):
    with open(infile, 'r') as f:
        json_lines = f.readlines()
    all_metrics = [json.loads(l) for l in json_lines]
    method_metrics = defaultdict(list)
    for metrics in all_metrics:
        metric_values = metrics['metrics'][metric]
        infl_method = metrics['finetune2']['infl_method']
        filter_method = metrics['finetune2']['filter_method']                
        if (influence_method != "" and influence_method != infl_method) and (filter_method not in allowed_filter_methods):
            continue
        module_pattern = metrics['finetune2']['module_pattern']
        if module_pattern not in module_pattern_to_name:
            continue
        module_pattern = module_pattern_to_name[module_pattern]
        is_infl = False
        if filter_method  == 'infl':
            filter_method = infl_method
            is_infl = True
        if module_pattern != "" and is_infl:
            filter_method = f'{filter_method}, {module_pattern}'
        method_metrics[filter_method].append(metric_values)

    method_names = sorted(method_metrics.keys())
    plt.ioff()
    for method in method_names:
        metrics = method_metrics[method]
        metric_values = np.array(metrics) * 100
        mean = np.mean(metric_values, axis=0)
        confidence_level = 0.95
        degrees_freedom = metric_values.shape[0] - 1
        sample_standard_error = stats.sem(metric_values, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        default_args = dict(marker='o', markersize=5, linewidth=1)
        if method == 'denoise':
            default_args = dict(linewidth=1, color='darkgray', linestyle='--')
        if method == 'rand':
            default_args = dict(linewidth=1, color='gray', linestyle='-.')
        line = plt.plot(np.arange(1, 11), mean, label=method, **default_args)
        plt.fill_between(np.arange(1, 11), min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, %')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{task.upper()} 70% filtered finetuning', fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile)  
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
def mean_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = 0
        inf_matrix = inf_matrix_clone
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = inf_matrix.sum(dim=0) / denoms
    return res

def median_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = torch.nan
        inf_matrix = inf_matrix_clone
    res = torch.nanmedian(inf_matrix, dim=0)[0]
    return res

def mean_dir_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    same_dir_mask = inf_matrix > 0
    if additional_mask is not None:
        same_dir_mask &= additional_mask
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = same_dir_mask.sum(dim=0).float() / denoms
    return res

def compute_noise_detection_metrics(base_dir_path: str, 
                                    m_prefix="m_b", i_prefix="i_b", 
                                    plot_title = "", cache_file = "",
                                    out_chart_file = ""):

    # loading dataset data for noise:
    score_fns = [mean_dir_score, mean_score, median_score] #[mean_score] #, median_score, mean_dir_score]

    score_names = {mean_score: "mean", mean_dir_score: "meandir", median_score: "median"}

    infl_method_names = {"cos": "cos", "cov": "cov", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf"}

    replace_name = {'layer': 'L', 'classifier': 'C', 'word_embeddings': 'WE' }

    # glob_mask = torch.zeros((len(infl), len(trainset)), dtype = torch.bool, device = device)

    # here we deal with binary confusion matrix
    # i0 = torch.where(infl_labels == 0)[0]
    # i1 = torch.where(infl_labels == 1)[0]
    # t0_n = torch.where((trainset_labels == 0) & noise_mask)[0]
    # t0_c = torch.where((trainset_labels == 0) & ~noise_mask)[0]
    # t1_n = torch.where((trainset_labels == 1) & noise_mask)[0]
    # t1_c = torch.where((trainset_labels == 1) & ~noise_mask)[0]

    all_dict_loaded = False
    if cache_file != "": # loading all post computed metrics from cache file for visualization
        try:
            all_dict = torch.load(cache_file)
            auc_rocs_by_methods_and_layer = all_dict['auc_rocs_by_methods_and_layer']
            first_30_score_by_methods_and_layer = all_dict['first_30_score_by_methods_and_layer']
            curves_by_methods_and_layer = all_dict['curves_by_methods_and_layer']
            auc_rocs_mean_std = all_dict['auc_rocs_mean_std']
            curves_mean_confidence_interval = all_dict['curves_mean_confidence_interval']
            sorted_method_keys = all_dict['sorted_method_keys']
            n_train = all_dict['n_train']
            num_noise = all_dict['num_noise']
            all_dict_loaded = True
        except:
            all_dict_loaded = False 
    if not all_dict_loaded:

        auc_rocs_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
        first_30_score_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
        curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

        noise_list_dict = {} # by run_id
        for file_name in os.listdir(base_dir_path):
            if not file_name.startswith("d_"):
                continue
            ds_file_parts = file_name.split('_')
            *_, run_id = ds_file_parts
            ds_path = os.path.join(base_dir_path, file_name)
            ds = datasets.load_from_disk(ds_path)
            trainset = ds['train']
            # noise_mask = torch.tensor(trainset['noise'], device = device)
            noise_list_dict[run_id] = trainset['noise']

        n_train = len(noise_list_dict['0'])
        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith(i_prefix):
                continue
            file_name = file_name_with_ext.split('.')[0]
            fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
            *method_parts, _, run_id_str = fine_name_parts
            infl_method = '_'.join(method_parts)
            file_path = os.path.join(base_dir_path, file_name_with_ext)
            matrix_dict = torch.load(file_path)
            if len(matrix_dict) > 1:
                total_layer_matrix = None 
                for module_name, inf_matrix in matrix_dict.items():
                    if total_layer_matrix is None:
                        total_layer_matrix = torch.clone(inf_matrix)
                    else:
                        total_layer_matrix += inf_matrix
                matrix_dict['total'] = total_layer_matrix
            noise_list = noise_list_dict[run_id_str]
            num_noise = sum(noise_list)
            num_clean = len(noise_list) - num_noise
            first_30_idx = round(0.3 * len(noise_list))
            ideal_area = num_noise / 2 + num_clean
            for module_name, inf_matrix in matrix_dict.items():
                for agg_method in score_fns:
                    scores = agg_method(inf_matrix)
                    train_ids = torch.argsort(scores).tolist()
                    noise_perc_curve = []
                    noise_count = 0
                    for train_id in train_ids:
                        if noise_list[train_id]:
                            noise_count += 1
                        noise_perc_curve.append(noise_count / num_noise)
                    first_30 = noise_perc_curve[first_30_idx]
                    auc_roc = sum((noise_perc_curve[i] + noise_perc_curve[i + 1]) / 2 for i in range(len(noise_perc_curve) - 1)) / ideal_area
                    auc_rocs_by_methods_and_layer[(infl_method, agg_method, module_name)].append(auc_roc)
                    curves_by_methods_and_layer[(infl_method, agg_method, module_name)].append(noise_perc_curve)
                    first_30_score_by_methods_and_layer[(infl_method, agg_method, module_name)].append(first_30)

        auc_rocs_mean_std = {}

        for key, auc_rocs in auc_rocs_by_methods_and_layer.items():
            mean = np.mean(auc_rocs)
            std = np.std(auc_rocs)
            auc_rocs_mean_std[key] = (mean, std)

        curves_mean_confidence_interval = {}

        for key, curves in curves_by_methods_and_layer.items():
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            confidence_level = 0.95
            degrees_freedom = curves.shape[0] - 1
            sample_standard_error = stats.sem(curves, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            curves_mean_confidence_interval[key] = (mean * 100, min_v * 100, max_v * 100)

        sorted_method_keys = sorted(auc_rocs_mean_std.keys(), key = lambda x: auc_rocs_mean_std[x][0], reverse=True)

        if cache_file != "":
            all_dict = {'auc_rocs_by_methods_and_layer': auc_rocs_by_methods_and_layer,
                        'first_30_score_by_methods_and_layer': first_30_score_by_methods_and_layer,
                        'curves_by_methods_and_layer': curves_by_methods_and_layer,
                        "auc_rocs_mean_std": auc_rocs_mean_std,
                        "curves_mean_confidence_interval": curves_mean_confidence_interval,
                        "sorted_method_keys": sorted_method_keys,
                        "n_train": n_train, "num_noise": num_noise}
            torch.save(all_dict, cache_file)


    # first_5 = sorted_method_keys[:5]

    # pick top of each excluding total
    selected_infl_methods = ['cos', 'cov', 'hf', 'hf_we_', 'hf_we_topk_10', 'datainf']
    tops = []
    for infl_method in selected_infl_methods:
        infl_key = next((key for key in sorted_method_keys if key[0] == infl_method and key[1] == mean_score and key[2] != 'total'), None)
        if infl_key:
            tops.append(infl_key)

    selected_keys = tops #first_5

    xs = 100*np.arange(n_train)/n_train
    simple_module_names = {}
    plt.ioff()
    for key in selected_keys:
        module_name = key[2]
        if module_name not in simple_module_names:
            simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
            simple_module_name = ' '.join(simple_name_parts)
            simple_module_names[module_name] = simple_module_name
        simple_module_name = simple_module_names[module_name]
        label = f"{infl_method_names[key[0]]}, {score_names[key[1]]}, {simple_module_name}"
        curve_means, curve_mins, curve_maxs = curves_mean_confidence_interval[key]
        p = plt.plot(xs, curve_means, label=label, linewidth=1)
        plt.fill_between(xs, curve_mins, curve_maxs, alpha=.1, linewidth=0, color = p[0].get_color())
     
    best_xs =100*np.arange(num_noise)/ n_train
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / n_train for cnt in range(n_train) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / n_train, color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)')
    plt.ylabel('Detection Rate (\\%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(plot_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_chart_file)  
    plt.clf()  

    # perc = DataFrame(rows, columns = ['module', *column_names])
    # print(perc)

    # perc.to_csv(f'tmp.csv', index = False)

    pass



def compute_noise_detection_metrics_per_sample(base_dir_path: str, 
                                    m_prefix="m_b", i_prefix="i_b", 
                                    plot_title = "", cache_file = "",
                                    out_chart_file = ""):   
    ''' In contrast to compute_noise_detection_metrics, uses each infl sample for scoring and filtering and then avg '''

    infl_method_names = {"cos": "cos", "cov": "cov", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf"}

    replace_name = {'layer': 'L', 'classifier': 'C', 'word_embeddings': 'WE' }


    all_dict_loaded = False
    if cache_file != "": # loading all post computed metrics from cache file for visualization
        try:
            all_dict = torch.load(cache_file)
            auc_ndr_by_methods_and_layer = all_dict['auc_rocs_by_methods_and_layer']
            # first_30_score_by_methods_and_layer = all_dict['first_30_score_by_methods_and_layer']
            curves_by_methods_and_layer = all_dict['curves_by_methods_and_layer']
            auc_ndr_mean_std = all_dict['auc_rocs_mean_std']
            curves_mean_confidence_interval = all_dict['curves_mean_confidence_interval']
            sorted_method_keys = all_dict['sorted_method_keys']
            n_train = all_dict['n_train']
            num_noise = all_dict['num_noise']
            all_dict_loaded = True
        except:
            all_dict_loaded = False 
    if not all_dict_loaded:

        auc_ndr_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
        # first_30_score_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
        curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

        noise_list_dict = {} # by run_id
        for file_name in os.listdir(base_dir_path):
            if not file_name.startswith("d_"):
                continue
            ds_file_parts = file_name.split('_')
            *_, run_id = ds_file_parts
            ds_path = os.path.join(base_dir_path, file_name)
            ds = datasets.load_from_disk(ds_path)
            trainset = ds['train']
            # noise_mask = torch.tensor(trainset['noise'], device = device)
            noise_list_dict[run_id] = trainset['noise']

        n_train = len(noise_list_dict['0'])
        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith(i_prefix):
                continue
            file_name = file_name_with_ext.split('.')[0]
            fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
            *method_parts, _, run_id_str = fine_name_parts
            infl_method = '_'.join(method_parts)
            file_path = os.path.join(base_dir_path, file_name_with_ext)
            matrix_dict = torch.load(file_path)
            noise_list = noise_list_dict[run_id_str]
            noise_one_tensor = torch.tensor(noise_list, dtype = torch.float, device = "cuda")
            num_noise = sum(noise_list)
            num_clean = len(noise_list) - num_noise
            # first_30_idx = round(0.3 * len(noise_list))
            ideal_area = num_noise / 2 + num_clean
            for module_name, inf_matrix in matrix_dict.items():
                train_ids = torch.argsort(inf_matrix, dim = 1)
                noise_tensor = noise_one_tensor[train_ids]

                noise_perc_curve = torch.cumsum(noise_tensor, dim = 1)
                noise_perc_curve /= num_noise

                auc_ndr = torch.sum((noise_perc_curve[:, :-1] + noise_perc_curve[:, 1:]) / 2, dim = 1) / ideal_area

                auc_ndr_mean = auc_ndr.mean().item()
                auc_ndr_by_methods_and_layer[(infl_method, module_name)].append(auc_ndr_mean)
                noise_perc_curve_mean = noise_perc_curve.mean(dim=0).cpu().numpy()
                curves_by_methods_and_layer[(infl_method, module_name)].append(noise_perc_curve_mean)

                del train_ids, noise_tensor, noise_perc_curve, auc_ndr

        auc_ndr_mean_std = {}

        for key, auc_ndr in auc_ndr_by_methods_and_layer.items():
            mean = np.mean(auc_ndr)
            std = np.std(auc_ndr)
            auc_ndr_mean_std[key] = (mean, std)

        curves_mean_confidence_interval = {}

        for key, curves in curves_by_methods_and_layer.items():
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            confidence_level = 0.95
            degrees_freedom = curves.shape[0] - 1
            sample_standard_error = stats.sem(curves, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            curves_mean_confidence_interval[key] = (mean * 100, min_v * 100, max_v * 100)

        sorted_method_keys = sorted(auc_ndr_mean_std.keys(), key = lambda x: auc_ndr_mean_std[x][0], reverse=True)

        if cache_file != "":
            all_dict = {'auc_rocs_by_methods_and_layer': auc_ndr_by_methods_and_layer,
                        # 'first_30_score_by_methods_and_layer': first_30_score_by_methods_and_layer,
                        'curves_by_methods_and_layer': curves_by_methods_and_layer,
                        "auc_rocs_mean_std": auc_ndr_mean_std,
                        "curves_mean_confidence_interval": curves_mean_confidence_interval,
                        "sorted_method_keys": sorted_method_keys,
                        "n_train": n_train, "num_noise": num_noise}
            torch.save(all_dict, cache_file)


    # first_5 = sorted_method_keys[:5]

    # pick top of each excluding total
    selected_infl_methods = ['cos', 'cov', 'hf', 'hf_we_', 'hf_we_topk_10', 'datainf']
    tops = []
    for infl_method in selected_infl_methods:
        infl_key = next((key for key in sorted_method_keys if key[0] == infl_method), None)
        if infl_key:
            tops.append(infl_key)

    selected_keys = tops #first_5

    xs = 100*np.arange(n_train)/n_train
    simple_module_names = {}
    plt.ioff()
    for key in selected_keys:
        module_name = key[1]
        if module_name not in simple_module_names:
            simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
            simple_module_name = ' '.join(simple_name_parts)
            simple_module_names[module_name] = simple_module_name
        simple_module_name = simple_module_names[module_name]
        label = f"{infl_method_names[key[0]]}, {simple_module_name}"
        curve_means, curve_mins, curve_maxs = curves_mean_confidence_interval[key]
        p = plt.plot(xs, curve_means, label=label, linewidth=1)
        plt.fill_between(xs, curve_mins, curve_maxs, alpha=.1, linewidth=0, color = p[0].get_color())
     
    best_xs =100*np.arange(num_noise)/ n_train
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / n_train for cnt in range(n_train) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / n_train, color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)')
    plt.ylabel('Detection Rate (\\%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(plot_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_chart_file)  
    plt.clf()  

    # perc = DataFrame(rows, columns = ['module', *column_names])
    # print(perc)

    # perc.to_csv(f'tmp.csv', index = False)

    pass


if __name__ == "__main__":

    compute_noise_detection_metrics_per_sample('./data/roberta-infl-matrix-with-we/qnli', 
                                    plot_title = "QNLI on Roberta-large with WE",
                                    cache_file="./data/roberta-infl-matrix-with-we/qnli/postprocess-per-sample.pt",
                                    out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/qnli/best_modules_per_sample.pdf")

    compute_noise_detection_metrics_per_sample('./data/roberta-infl-matrix-with-we/mrpc', 
                                    plot_title = "MRPC on Roberta-large with WE",
                                    cache_file="./data/roberta-infl-matrix-with-we/mrpc/postprocess-per-sample.pt",
                                    out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/mrpc/best_modules_per_sample.pdf")

    compute_noise_detection_metrics_per_sample('./data/roberta-infl-matrix-with-we/sst2', 
                                    plot_title = "SST2 on Roberta-large with WE",
                                    cache_file="./data/roberta-infl-matrix-with-we/sst2/postprocess-per-sample.pt",
                                    out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/sst2/best_modules_per_sample.pdf")

    compute_noise_detection_metrics_per_sample('./data/roberta-infl-matrix-with-we/qqp', 
                                    plot_title = "QQP on Roberta-large with WE",
                                    cache_file="./data/roberta-infl-matrix-with-we/qqp/postprocess-per-sample.pt",
                                    out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/qqp/best_modules_per_sample.pdf")


    # compute_noise_detection_metrics('./data/roberta-infl-matrix-with-we/qnli', 
    #                                 plot_title = "QNLI on Roberta-large with WE",
    #                                 cache_file="./data/roberta-infl-matrix-with-we/qnli/postprocess.pt",
    #                                 out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/qnli/best_modules_on_mean.pdf")

    # compute_noise_detection_metrics('./data/roberta-infl-matrix-with-we/mrpc', 
    #                                 plot_title = "MRPC on Roberta-large with WE",
    #                                 cache_file="./data/roberta-infl-matrix-with-we/mrpc/postprocess.pt",
    #                                 out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/mrpc/best_modules_on_mean.pdf")

    # compute_noise_detection_metrics('./data/roberta-infl-matrix-with-we/sst2', 
    #                                 plot_title = "SST2 on Roberta-large with WE",
    #                                 cache_file="./data/roberta-infl-matrix-with-we/sst2/postprocess.pt",
    #                                 out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/sst2/best_modules_on_mean.pdf")

    # compute_noise_detection_metrics('./data/roberta-infl-matrix-with-we/qqp', 
    #                                 plot_title = "QQP on Roberta-large with WE",
    #                                 cache_file="./data/roberta-infl-matrix-with-we/qqp/postprocess.pt",
    #                                 out_chart_file="./data/roberta-infl-matrix-with-we/postprocess/qqp/best_modules_on_mean.pdf")


    # draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",
    #                                 out = "./data/mdr/cifar.png", module_name = '')

    # draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",
    #                                 out = "./data/mdr/cifar-layer4.png", module_name = 'layer4\\..*', title='Layer 4, worst noise', task="cifar10")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr/{ds}.png")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.[0-7]\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-0/{ds}.png", title="Layers 0-7")

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.([8-9]|1[0-5])\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-1/{ds}.png", title="Layers 8-15")

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-2/{ds}.png", title="Layers 16-23")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.classifier\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-3/{ds}.png", title="Classifier")        

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_ft2_metric(task = ds, infile = f'./data/ft2-infl/{ds}.jsonlist', outfile = f'./data/accuracy/{ds}.png', metric = 'accuracy')
    # list_modules('./data/infl/mrpc/i_datainf_mrpc_0.pt', '') #'.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*')
    # draw_curve(res_filer='infl_qnli_', out = "./data/auc/qnli.png")    
    # draw_curve(task="qnli", res_folder = "./data/self-infl", module_pattern = '', datasets_folder = './data/datasets', out = "./data/auc/qnli.png")
    
    # module_pattern_to_name = {".*\\.layer\\.(1[6-9]|2[0-3])\\..*\\.lora_(A|B)\\..*": "last 8", 
    #                           "": "all", 
    #                           ".*\\.classifier\\..*": "classifier", 
    #                           ".*\\.layer\\.([0-9]|1[0-5])\\..*\\.lora_(A|B)\\..*": "first 16", 
    #                           "rand": "rand", 
    #                           ".*\\.layer\\.[0-8]\\..*\\.lora_(A|B)\\..*": "first 9"}
    # for d in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     for m in ['datainf', 'hf', 'lissa']:
    #         draw_ft2_metric(d, infile = f'./data/ft2-infl/{d}.jsonlist', outfile = f'./data/accuracy/{m}/{d}.png', 
    #                         influence_method = m, metric = 'accuracy', module_pattern_to_name = module_pattern_to_name)
    
    # module_pattern_to_name = {".*\\.layer\\.(1[6-9]|2[0-3])\\..*\\.lora_(A|B)\\..*": "last 8", "": "all"}
    # for d in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_ft2_metric(infile = f'./data/ft2-infl/{d}.jsonlist', outfile = f'./data/accuracy/{d}.png', 
    #                     influence_method = "lissa", metric = 'accuracy', module_pattern_to_name = module_pattern_to_name)
