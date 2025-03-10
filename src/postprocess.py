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

def compute_noise_detection_metrics(infl_file_name: str, ds_file_name: str, device = 'cuda'):    
    matrix_dict = torch.load(infl_file_name)
    ds = datasets.load_from_disk(ds_file_name)
    trainset = ds['train']
    trainset_labels = torch.tensor(trainset['labels'], device = device)
    noise_mask = torch.tensor(trainset['noise'], device = device)
    noise_list = trainset['noise']
    infl = ds['infl']
    infl_labels = torch.tensor(infl['labels'], device = device)
    num_noise = noise_mask.sum().item()
    num_clean = len(trainset) - num_noise
    num_infl = len(infl)

    class_match_mask = infl_labels.unsqueeze(1) == trainset_labels.unsqueeze(0)    

    score_fns = [mean_score, median_score, mean_dir_score]

    # glob_mask = torch.zeros((len(infl), len(trainset)), dtype = torch.bool, device = device)

    # here we deal with binary confusion matrix
    # i0 = torch.where(infl_labels == 0)[0]
    # i1 = torch.where(infl_labels == 1)[0]
    # t0_n = torch.where((trainset_labels == 0) & noise_mask)[0]
    # t0_c = torch.where((trainset_labels == 0) & ~noise_mask)[0]
    # t1_n = torch.where((trainset_labels == 1) & noise_mask)[0]
    # t1_c = torch.where((trainset_labels == 1) & ~noise_mask)[0]

    column_names = [fn.__name__ for fn in score_fns] #["maj_acc", "maj_same_class_acc", "maj_diff_class_acc"] #['noise_maj', 'clean_maj', 'noise_maj_same_class', 'clean_maj_same_class', 'noise_maj_diff_class', 'clean_maj_diff_class']
    rows = []
    ideal_area = num_noise / 2 + num_clean
    baseline_auc_roc = ((num_clean + num_noise) / 2) / ideal_area
    curves = {}
    total_int_matrix = torch.zeros((len(infl), len(trainset)), dtype=torch.float, device = device)
    module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    for module_name, int_matrix in matrix_dict.items():
        if not any(f in module_name for f in module_filter):
            total_int_matrix += int_matrix
    matrix_dict['total'] = total_int_matrix
    
    for module_name, int_matrix in matrix_dict.items():

        # module_metrics = {}
        auc_rocs = []
        curve_list = curves.setdefault(module_name, [])

        for score_fn in score_fns: 

            scores = score_fn(int_matrix, additional_mask=~class_match_mask)
            
            train_ids = torch.argsort(scores).tolist()

            noise_perc = []
            noise_count = 0
            for train_id in train_ids:
                if noise_list[train_id]:
                    noise_count += 1
                noise_perc.append(noise_count / num_noise)
            auc_roc = sum((noise_perc[i] + noise_perc[i + 1]) / 2 for i in range(len(noise_perc) - 1)) / ideal_area
            curve_list.append(noise_perc)

            auc_rocs.append(auc_roc)
        rows.append([module_name, *auc_rocs])

            # module_metrics.setdefault(score_fn.__name__, {"auc_roc": auc_roc, "curve": noise_perc})


        # maj = dir_majority_indicator(int_matrix)
        # noise_maj = maj[noise_mask].float().mean().item()
        # clean_maj = maj[~noise_mask].float().mean().item()

        # maj_same_class = dir_majority_indicator(int_matrix, additional_mask = class_match_mask)
        # noise_maj_same_class = maj_same_class[noise_mask].float().mean().item()
        # clean_maj_same_class = maj_same_class[~noise_mask].float().mean().item()

        # maj_diff_class = dir_majority_indicator(int_matrix, additional_mask = ~class_match_mask)
        # noise_maj_diff_class = maj_diff_class[noise_mask].float().mean().item()
        # clean_maj_diff_class = maj_diff_class[~noise_mask].float().mean().item()
    
        # rows.append([module_name, noise_maj, clean_maj, noise_maj_same_class, clean_maj_same_class, noise_maj_diff_class, clean_maj_diff_class])
        # rows.append([module_name, maj_acc, maj_same_class_acc, maj_diff_class_acc])

    modules_sorted = sorted(rows, key = lambda x: max(x[1:]), reverse=True)

    best_3_module_names = [mn[0] for mn in modules_sorted[:3]]

    plt.ioff()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    xs = 100*np.arange(len(trainset))/ len(trainset)
    replace_name = {'layer': 'L', 'classifier': 'C', 'word_embeddings': 'WE' }
    for mid, module_name in enumerate(best_3_module_names):
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]

        # detection_rate_lists = data_detection_per_method[method]
        # drls = np.array(detection_rate_lists)
        # drl = np.mean(drls, axis=0)
        # confidence_level = 0.95
        # degrees_freedom = drls.shape[0] - 1
        # sample_standard_error = stats.sem(drls, axis=0)
        # confidence_interval = stats.t.interval(confidence_level, degrees_freedom, drl, sample_standard_error)
        # min_v = confidence_interval[0]
        # max_v = confidence_interval[1]
        ys = curves[module_name]
        linestyles = ['-', '--', '-.']
        simple_module_name = ' '.join(simple_name_parts)
        for curve_id, (curve, score_fn) in enumerate(zip(ys, score_fns)):
            plt.plot(xs, [ c * 100 for c in curve], label=f"{simple_module_name} {score_fn.__name__}", linestyle=linestyles[curve_id], color = colors[mid], linewidth = 1.5)
        # plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    best_xs =100*np.arange(num_noise)/ len(trainset)
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / len(trainset) for cnt in range(len(trainset)) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / len(trainset), color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    # plt.title(f'{(task.upper())}, {title}', fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig('tmp.png')  
    plt.clf()  




    perc = DataFrame(rows, columns = ['module', *column_names])
    print(perc)

    perc.to_csv(f'tmp.csv', index = False)

    pass

if __name__ == "__main__":

    compute_noise_detection_metrics('./data/infl-matrix/qnli/i_hf_qnli_0.pt', './data/infl-matrix/qnli/d_qnli_0')
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
