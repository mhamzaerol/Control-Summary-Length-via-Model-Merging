import sys
sys.path.append('..')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv
import os
import argparse

from src.utils import load_yaml_as_namespace, manage_exp_dirs, DebugDataCollator, DebugTrainer, PrintModelOutputCallback
from src.data import create_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

load_dotenv()
hf_access_token = os.getenv('HF_ACCESS_TOKEN')

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_path", type=str, default="experiments/configs/example.yaml")
args = argparser.parse_args()

config = load_yaml_as_namespace(args.config_path)
config = manage_exp_dirs(config)

# TODO: Incorporate the seed from the config file

torch_dtype = torch.bfloat16 if config.exp.device == 'cuda' else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    config.model.tokenizer_name,
    token=hf_access_token,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.model.model_name,
    token=hf_access_token,
    torch_dtype=torch_dtype,
    max_length=config.data.max_length,
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=config.peft.r, 
    lora_alpha=config.peft.lora_alpha, 
    lora_dropout=config.peft.lora_dropout
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(config.exp.device)

data = create_dataset(config)
data.prepare(tokenizer)

training_args = TrainingArguments(
    **vars(config.training)
)

if config.exp.debug:
    collator_class = DebugDataCollator
    trainer_class = DebugTrainer
else:
    collator_class = DataCollatorForSeq2Seq
    trainer_class = Trainer

data_collator = collator_class(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    return_tensors="pt"
)

trainer = trainer_class(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data.get_split("train"),
    eval_dataset=data.get_split("val"),
)

if config.exp.debug or config.exp.sample_logging:
    trainer.add_callback(PrintModelOutputCallback(
        tokenizer=tokenizer,
        raw_debug_dataset=data.get_raw_split("test"),
        num_samples=5,
        config=config
    ))

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

if data.get_split("test") is not None:
    trainer.eval_dataset = data.get_split("test")
    test_results = trainer.evaluate()
    print(test_results)
