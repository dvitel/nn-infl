import re
from typing import Optional
from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, AutoModel,
    get_linear_schedule_with_warmup,
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

    # def __init__(self, 
    #             model_name_or_path="roberta-large",
    #             target_modules=["value"],
    #             train_dataloader=None,
    #             eval_dataloader=None,
    #             device="cuda",
    #             num_epochs=10,
    #             lr=3e-4,
    #             low_rank=2,
    #             task="mrpc"):
    #     self.model_name_or_path=model_name_or_path
    #     self.target_modules=target_modules
    #     self.train_dataloader=train_dataloader
    #     self.eval_dataloader=eval_dataloader
    #     self.device=device
    #     self.num_epochs=num_epochs
    #     self.lr=lr
    #     self.task=task
    #     self.low_rank=low_rank

def unfreeze_modules(model, unfreeze_modules_regex: Optional[str] = None):
    if unfreeze_modules_regex:
        unfreeze_pattern = re.compile(unfreeze_modules_regex)
        for module_name, module_params in model.named_parameters():
            if unfreeze_pattern.match(module_name):
                module_params.requires_grad = True
        
def build_LORA_model(model_name_or_path, target_modules, low_rank, unfreeze_modules_regex: Optional[str] = None):
    '''
    This function fine-tunes a model for classification tasks. 
    For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.

    unfreeze_modules - list of additional modules to unfreeze
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                    return_dict=True)
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
        
    peft_config = LoraConfig(task_type="SEQ_CLS",
                                inference_mode=False, 
                                target_modules=target_modules,
                                r=low_rank,
                                lora_alpha=low_rank, 
                                lora_dropout=0.05)
    model = get_peft_model(model, peft_config)

    unfreeze_modules(model, unfreeze_modules_regex)

    model.print_trainable_parameters()

    return model

def load_pretrained_LORA_model(model_name_or_path, unfreeze_modules_regex: Optional[str] = None):
    '''
    This function loads a pre-trained model.
    '''
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=True)
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id

    unfreeze_modules(model, unfreeze_modules_regex)
    
    model.print_trainable_parameters()
    return model

def train_LORA_model(model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device="cuda",
        num_epochs=10,
        lr=3e-4,
        task="mrpc",
        compute_cancellation = False,
        compute_gold_val_predictions = False):
    '''
    This function fine-tunes a model for GLUE classification tasks. 
    For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.

    Params compute_cancellation and compute_gold_val_predictions are used for metrics from:
        https://proceedings.neurips.cc/paper_files/paper/2022/file/d07022783ff6f7bf7a288c207b7dcbd1-Paper-Conference.pdf
    '''
    metric = evaluate.load("glue", task)
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06*(len(train_dataloader)*num_epochs),
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

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
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
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

            if (epoch == (num_epochs - 1)) and compute_gold_val_predictions:
                batch_gold_preds = outputs.logits[torch.arange(outputs.logits.shape[0]), references].cpu().tolist()
                gold_val_predictions.extend(batch_gold_preds)

        eval_metric = metric.compute()
        print(f"Epoch {(epoch+1)}:", eval_metric)
        for key, item in eval_metric.items():
            eval_metrics.setdefault(key, []).append(item)
    if len(cancel_norm) > 0:
        eval_metrics["cancel_norm"] = cancel_norm
    if len(cancel_abs) > 0:
        eval_metrics["cancel_abs"] = cancel_abs        
    if len(gold_val_predictions) > 0:
        eval_metrics["gold_val_predictions"] = gold_val_predictions
    return eval_metrics

def compute_grads(model, dataloader, device="cuda", bring_to_cpu=False):
    ''' Builds tensor of grads, collected accross the model '''
    module_grads = {}
    num_samples = len(dataloader)
    model.to(device)
    model.eval() # avoid dropout and batchnorm
    module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    for k, v in model.named_parameters():
        if any(f in k for f in module_filter):
            grad = torch.empty((num_samples, v.numel()), device=device)
            module_grads[k] = grad
        else:
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

