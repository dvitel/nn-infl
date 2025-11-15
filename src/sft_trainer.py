from dataclasses import dataclass, field
import os
from random import random
from typing import Optional
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser
from trl import SFTTrainer, SFTConfig
from lora_model import load_tokenizer

seed = int(os.environ.get("INFL_SEED", 0))
cwd = os.environ.get("INFL_CWD", "./data/dev")

print(f"Setting random seed {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tqdm.pandas()

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=128, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    m_prefix: Optional[str] = field(default="checkpoint", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})
    # save_steps: Optional[int] = field(
    #     default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    # )
    save_total_limit: Optional[int] = field(default=5, metadata={"help": "Limits total number of checkpoints."})
    lora_targets: Optional[str] = field(default="q_proj,v_proj", metadata={"help": "Comma separated list of PEFT modules to unfreeze"})    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print(f"Loading data {script_args.dataset_name}...")

dataset = load_from_disk(script_args.dataset_name) # should be _train from datasetss

dataset = dataset.remove_columns(["answer", "prompt"])

tokenizer = load_tokenizer(script_args.model_name)

# collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt", max_length=script_args.seq_length)  

# train_dataloader = DataLoader(dataset,
#                                 shuffle=True,
#                                 collate_fn=collator,
#                                 batch_size=script_args.batch_size) 

# all_token_ids, mapping_tensor = present_token_ids(train_dataloader)

lora_targets = script_args.lora_targets.split(",") if script_args.lora_targets else ["q_proj", "v_proj"]
# model, model_info = build_causal_LORA_model(
#             model_name_or_path=script_args.model_name, 
#             target_modules=lora_targets,
#             low_rank=script_args.peft_lora_r, 
#             lora_alpha=script_args.peft_lora_alpha,
#             unfreeze_modules_regex=None,
#             all_token_ids=None, #all_token_ids, 
#             mapping_tensor=None, #mapping_tensor, 
#             pad_token_id=tokenizer.pad_token_id,
#             quantization_config=quantization_config,
#         )

print(f"Loading model {script_args.model_name}...")

model = AutoModelForCausalLM.from_pretrained(script_args.model_name,
            return_dict=True,
            # quantization_config = quantization_config,
            device_map="auto",
            dtype = torch.bfloat16,
            offload_folder = os.path.join(os.environ['HF_HOME'], ".offload"),
            offload_state_dict = True)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
    
peft_config = LoraConfig(task_type="CAUSAL_LM",
                            inference_mode=False, 
                            target_modules=lora_targets,
                            r=script_args.peft_lora_r,
                            lora_alpha=script_args.peft_lora_alpha,                                 
                            lora_dropout=0.05)


output_dir = os.path.join(cwd, f"{script_args.m_prefix}_{seed}")

# Step 3: Define the training arguments
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    # max_steps=script_args.max_steps,
    # save_steps=script_args.save_steps,
    save_total_limit=5, 
    max_length=script_args.seq_length,
    dataset_text_field=script_args.dataset_text_field,
    report_to="none", 
    # no_cuda=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    # optim="adamw_torch_fused"
)
# import pdb; pdb.set_trace()

print(f"Training...")

trainer.train()

# Step 6: Save the model
# import pdb; pdb.set_trace()
print(f"Saving to {output_dir}...")

trainer.save_model(output_dir)
pass
