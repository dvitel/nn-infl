from functools import partial
import json
import os
import re
import sys
from typing import Optional
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, AutoModel,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

def dropout_forward_hook(module: torch.nn.Module, input, output, p = 0.05):
    result = torch.nn.functional.dropout(output, p=p, training=module.training)
    return result

def unfreeze_modules(model: torch.nn.Module, unfreeze_modules_regex: Optional[str] = None):
    if unfreeze_modules_regex:
        unfreeze_pattern = re.compile(unfreeze_modules_regex)
        for module_name, module_params in model.named_parameters():
            if unfreeze_pattern.match(module_name):
                module_params.requires_grad = True
                module_true_name = ".".join(module_name.split(".")[:-1])
                submodule = model.get_submodule(module_true_name)
                submodule.register_forward_hook(dropout_forward_hook)

def vocab_remap_forward_pre_hook(module: torch.nn.Module, input):
    ''' This hook is used to remap input ids '''
    # if hasattr(module, 'vocab_mapping'):
    mapping = module.vocab_mapping
    if mapping.device != input[0].device:
        module.vocab_mapping = mapping.to(input[0].device)
        mapping = module.vocab_mapping
    updated_input = mapping[input[0]]
    return (updated_input,*input[1:])
    # return input

def build_LORA_model(model_name_or_path, target_modules, low_rank, unfreeze_modules_regex: Optional[str] = None,
                        all_token_ids: Optional[torch.Tensor] = None, mapping_tensor: Optional[torch.Tensor] = None,
                        pad_token_id = None):
    '''
    unfreeze_modules - list of additional modules to unfreeze
    all_token_ids - to resize embeddings
    '''

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                    return_dict=True,
                                                                    torch_dtype = torch.bfloat16,
                                                                    offload_folder = os.path.join(os.environ['HF_HOME'], ".offload"),
                                                                    offload_state_dict = True)
    model.config.use_cache = False
    model.config.pad_token_id = pad_token_id
        
    peft_config = LoraConfig(task_type="SEQ_CLS",
                                inference_mode=False, 
                                target_modules=target_modules,
                                r=low_rank,
                                lora_alpha=low_rank, 
                                lora_dropout=0.05)
    model = get_peft_model(model, peft_config)

    if all_token_ids is not None:
        orig_embeddings = model.get_input_embeddings()
        new_embedding_matrix = orig_embeddings.weight[all_token_ids]
        new_pad_token_id = mapping_tensor[pad_token_id]
        new_embedding_layer = torch.nn.Embedding.from_pretrained(new_embedding_matrix, freeze=False, padding_idx=new_pad_token_id,
                                                                 max_norm = orig_embeddings.max_norm,
                                                                 norm_type = orig_embeddings.norm_type,
                                                                 scale_grad_by_freq = orig_embeddings.scale_grad_by_freq,
                                                                 sparse = orig_embeddings.sparse)
        new_embedding_layer.vocab_mapping = mapping_tensor
        new_embedding_layer.register_forward_pre_hook(vocab_remap_forward_pre_hook)
        model.set_input_embeddings(new_embedding_layer)
        # model.tie_weights() - does not work - resort to save our custom embeddings

    unfreeze_modules(model, unfreeze_modules_regex)

    model.print_trainable_parameters()
    
    infos = []
    for module_name, p in model.named_parameters():
        size_gb = p.numel() * (torch.finfo(p.dtype).bits // 8) / 1024 ** 3
        shape = [d for d in p.shape]
        tunable = p.requires_grad
        infos.append({"name": module_name, "numel": p.numel(), "size_gb": size_gb, "shape": shape, 'tunable': tunable})
    num_tunable_p, num_all_p = model.get_nb_trainable_parameters()
    model_info_table = tabulate(infos, headers="keys", tablefmt="github", floatfmt=".3f", showindex=True)
    model_info_str = "[%s] Num trainable parameters: %d, num all parameters: %d, %.2f" % (model_name_or_path, num_tunable_p, num_all_p, round(num_tunable_p * 100 / num_all_p))
    model_info_str += "\n" + model_info_table
    return model, model_info_str

def load_pretrained_LORA_model(model_name_or_path, unfreeze_modules_regex: Optional[str] = None):
    '''
    This function loads a pre-trained model.
    '''
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                    return_dict=True,
                                                                    torch_dtype = torch.bfloat16,
                                                                    offload_folder = os.path.join(os.environ['HF_HOME'], ".offload"),
                                                                    offload_state_dict = True)                                                                    
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=True)
    # if os.path.exists(f"{model_name_or_path}/emb.pt"):
    emb_params = torch.load(f"{model_name_or_path}/emb.pt")
    embeddings = emb_params.pop('weight')
    mapping = emb_params.pop('mapping')
    model.config.pad_token_id = emb_params['padding_idx']
    del emb_params['padding_idx']
    emb = torch.nn.Embedding.from_pretrained(embeddings, padding_idx = model.config.pad_token_id, **emb_params)
    emb.vocab_mapping = mapping
    emb.register_forward_pre_hook(vocab_remap_forward_pre_hook)
    model.set_input_embeddings(emb)

    unfreeze_modules(model, unfreeze_modules_regex)
    
    model.print_trainable_parameters()
    return model

def save_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    model.save_pretrained(checkpoint_path)
    emb = model.get_input_embeddings()
    emb_data = dict(weight = emb.weight.cpu().detach(), 
                    padding_idx=(emb.padding_idx if type(emb.padding_idx) == int else emb.padding_idx.item()),
                    max_norm = emb.max_norm,
                    norm_type = emb.norm_type,
                    scale_grad_by_freq = emb.scale_grad_by_freq,
                    sparse = emb.sparse,
                    mapping = emb.vocab_mapping.cpu())
    torch.save(emb_data, f"{checkpoint_path}/emb.pt")    

def train_LORA_model(model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        infl_dataloader: Optional[DataLoader],
        device="cuda",
        num_epochs=10,
        lr=1e-4,
        compute_cancellation = False,
        compute_gold_val_predictions = False,
        best_checkpoint_path = None,
        last_checkpoint_path = None,
        best_loss_model_path = None):
    '''
    This function fine-tunes a model for GLUE classification tasks. 
    For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.

    Params compute_cancellation and compute_gold_val_predictions are used for metrics from:
        https://proceedings.neurips.cc/paper_files/paper/2022/file/d07022783ff6f7bf7a288c207b7dcbd1-Paper-Conference.pdf
    '''
    # metric = evaluate.load("glue", task)
    accuracy_metric = evaluate.load("accuracy")
    # train_accuracy_metric = evaluate.load("accuracy")
    infl_accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.1*(len(train_dataloader)*num_epochs),
        num_training_steps=(len(train_dataloader)*num_epochs),
    )

    model.to(device)
    eval_metrics = {}
    weights_delta = {}
    weights_delta_abs = {}
    weights_delta_norms = {}
    grads_abs = {}
    grads_norms = {}
    current_lr = lr
    cancel_norm = {}
    cancel_abs = {}
    gold_val_predictions = []
    best_infl_loss = np.inf
    best_infl_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        if (epoch == (num_epochs - 1)) and compute_cancellation: 
            weights_before = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights_before[name] = param.clone().detach()
            current_lr = lr_scheduler.get_last_lr()[0]
            cancellation_data_loader = DataLoader(train_dataloader.dataset, batch_size=1, shuffle=False, collate_fn=train_dataloader.collate_fn)
            for step, batch in enumerate(tqdm(cancellation_data_loader)):
                optimizer.zero_grad()
                batch.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if name in grads_abs:
                            grads_abs[name] += torch.abs(param.grad)
                        else:
                            grads_abs[name] = torch.abs(param.grad)
                        if name in grads_norms:
                            grads_norms[name] += torch.norm(param.grad.view(-1))
                        else:
                            grads_norms[name] = torch.norm(param.grad.view(-1))
            optimizer.zero_grad()

        # NEXT is for debugging embedding shrinking
        # emb.weight.grad[~torch.isin(torch.arange(emb.weight.shape[0], device='cuda'), emb.vocab_mapping[batch['input_ids']].unique())]
        # emb = model.get_input_embeddings()
        # torch.all(emb.weight.grad[~torch.isin(torch.arange(emb.weight.shape[0], device='cuda'), emb.vocab_mapping[batch['input_ids']].unique())] == 0)

        train_loss = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()            
            lr_scheduler.step()
            optimizer.zero_grad()

        if (epoch == (num_epochs - 1)) and compute_cancellation: 
            weights_after = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights_after[name] = param.clone().detach()

            for name in weights_before.keys():            
                weights_delta[name] = weights_after[name] - weights_before[name]
                weights_delta_abs[name] = torch.abs(weights_delta[name])
                # weights_delta_abs[name] += 1e-20 # to avoid division by zero
                weights_delta_norms[name] = torch.norm(weights_delta[name].view(-1))
            for name in grads_abs.keys():
                grads_abs[name] *= current_lr   
            for name in grads_norms.keys():
                grads_norms[name] *= current_lr                     
            for name in grads_abs.keys():
                # cancel_abs[name] = torch.mean(grads_abs[name] / weights_delta_abs[name]).item()
                updated_grads = (grads_abs[name] > 0).view(-1)
                grads_to_consider = (grads_abs[name] / weights_delta_abs[name]).view(-1)[updated_grads]
                cancel_abs[name] = torch.median(grads_to_consider).item()
            for name in grads_norms.keys():
                cancel_norm[name] = (grads_norms[name] / weights_delta_norms[name]).item()
            del weights_before, weights_after, weights_delta, grads_abs, weights_delta_abs, weights_delta_norms

        model.eval()
        infl_loss = []
        infl_logits = None
        if infl_dataloader is not None:
            infl_shift = 0
            for step, batch in enumerate(tqdm(infl_dataloader)):
                batch.to(device)
                with torch.no_grad():
                    outputs = model(**batch)
                infl_loss.append(outputs.loss.item())
                if infl_logits is None:
                    infl_logits = torch.zeros((len(infl_dataloader.dataset), model.config.num_labels), device=outputs.logits.device, dtype = outputs.logits.dtype)
                batch_size = outputs.logits.shape[0]
                infl_logits[infl_shift:infl_shift+batch_size] = outputs.logits
                infl_shift += batch_size
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                infl_accuracy_metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

        # for step, batch in enumerate(tqdm(train_dataloader)):
        #     batch.to(device)
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     predictions = outputs.logits.argmax(dim=-1)
        #     predictions, references = predictions, batch["labels"]
        #     train_accuracy_metric.add_batch(
        #         predictions=predictions,
        #         references=references,
        #     )                

        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            accuracy_metric.add_batch(
                predictions=predictions,
                references=references,
            )
            f1_metric.add_batch(
                predictions=predictions,
                references=references,
            )

            if (epoch == (num_epochs - 1)) and compute_gold_val_predictions:
                batch_gold_preds = outputs.logits[torch.arange(outputs.logits.shape[0]), references].cpu().tolist()
                gold_val_predictions.extend(batch_gold_preds)

        # eval_metric = metric.compute()
        # train_accuracy = train_accuracy_metric.compute()["accuracy"]
        accuracy = accuracy_metric.compute()
        f1 = f1_metric.compute()
        metrics = {"train_loss": np.mean(train_loss), **accuracy, **f1}
        if len(infl_loss) > 0:
            infl_accuracy = infl_accuracy_metric.compute()["accuracy"]
            infl_loss_value = np.mean(infl_loss)            
        else: 
            infl_loss_value = None
            infl_accuracy = None
        if (best_checkpoint_path is not None) and (infl_accuracy is not None) and ((infl_accuracy > best_infl_accuracy) or ((infl_accuracy == best_infl_accuracy) and (infl_loss_value < best_infl_loss))):
            save_checkpoint(model, best_checkpoint_path)
            if infl_logits is not None:
                torch.save(infl_logits, f"{best_checkpoint_path}/infl_logits.pt")
        if (best_loss_model_path is not None) and (infl_loss_value is not None) and (infl_loss_value <= best_infl_loss):
            save_checkpoint(model, best_loss_model_path)
            if infl_logits is not None:
                torch.save(infl_logits, f"{best_loss_model_path}/infl_logits.pt")
        if (infl_loss_value is not None) and (best_infl_loss > infl_loss_value):
            best_infl_loss = infl_loss_value
        if (infl_accuracy is not None) and (best_infl_accuracy < infl_accuracy):
            best_infl_accuracy = infl_accuracy
        if infl_accuracy is not None:            
            metrics.update(best_infl_loss = best_infl_loss, infl_loss = infl_loss_value, best_infl_accuracy = best_infl_accuracy, infl_accuracy = infl_accuracy)
        print(f"Epoch {(epoch+1)}:", metrics)
        for key, item in metrics.items():
            eval_metrics.setdefault(key, []).append(item)
    if len(cancel_norm) > 0:
        eval_metrics["cancel_norm"] = cancel_norm
    if len(cancel_abs) > 0:
        eval_metrics["cancel_abs"] = cancel_abs        
    if len(gold_val_predictions) > 0:
        eval_metrics["gold_val_predictions"] = gold_val_predictions
    if last_checkpoint_path is not None:
        save_checkpoint(model, last_checkpoint_path)
        if infl_logits is not None:
            infl_logits_path = f"{last_checkpoint_path}/infl_logits.pt"
            torch.save(infl_logits, infl_logits_path)
    return eval_metrics

def compute_grads(model, dataloader, device="cuda", bring_to_cpu=False):
    ''' Builds tensor of grads, collected accross the model '''
    module_grads = {}
    num_samples = len(dataloader)
    model.to(device)
    model.eval() # avoid dropout and batchnorm
    module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    # module_filter = ['modules_to_save.default.out_proj.weight']
    for k, v in model.named_parameters():
        if any(f in k for f in module_filter):
            grad = torch.empty((num_samples, v.numel()), device=device)
            module_grads[k] = grad
            v.requires_grad = True
        else:
            v.requires_grad = False
            pass         
    # collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    # dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)        
    for step, batch in enumerate(tqdm(dataloader)):
        model.zero_grad() # zeroing out gradient
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        for k, v in model.named_parameters():
            if k in module_grads:
                module_grads[k][step] = v.grad.view(-1)
            else:
                pass
    if bring_to_cpu:
        for k, v in module_grads.items():
            module_grads[k] = v.cpu()
            del v
    return module_grads
    
    # def compute_gradient_old(self, tokenized_datasets, collate_fn):
    #     train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
    #     val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
    #     # Compute the gradient
    #     self.model.eval()
    #     tr_grad_dict = {}
    #     for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             elif 'lora_B' in k:
    #                 # first index of shape indicates low-rank
    #                 grad_dict[k]=v.grad.cpu().T
    #             elif 'modules_to_save.default.out_proj.weight' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             else:
    #                 pass
    #         tr_grad_dict[step]=grad_dict
    #         del grad_dict
            
    #     val_grad_dict = {}
    #     for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             elif 'lora_B' in k:
    #                 # first index of shape indicates low-rank
    #                 grad_dict[k]=v.grad.cpu().T
    #             elif 'modules_to_save.default.out_proj.weight' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             else:
    #                 pass
    #         val_grad_dict[step]=grad_dict    
    #         del grad_dict
            
    #     return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                project_path,
                dataset_name='math_with_reason',
                device="cuda"):
        self.base_path = base_path
        self.project_path = project_path
        self.adapter_path = f"{self.project_path}/models/math_with_reason_13bf"
        self.dataset_name = dataset_name
        self.device=device
        self.load_pretrained_network()
        self.load_datasets()

    def load_pretrained_network(self):
        # setup tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        base_model = LlamaForCausalLM.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
        )

        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    def load_datasets(self):
        self.train_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_train.hf")
        self.validation_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_test.hf")

    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            x["prompt"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        if 'with_reason' in self.dataset_name:
            column_list=["text", "answer", "variation", "prompt", "reason"]
        else:
            column_list=["text", "answer", "variation", "prompt"]

        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        return tokenized_datasets, collate_fn

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict

