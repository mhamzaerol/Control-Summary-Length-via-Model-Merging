import sys
sys.path.append('..')

import torch
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm
import os
from src.utils import load_yaml_as_namespace, update_config_with_exp_dir
from src.data import create_dataset
from peft import PeftModel
from src.utils import get_assistant_output_raw
from tqdm import tqdm
import pickle
import json
import argparse
import numpy as np


load_dotenv()
hf_access_token = os.getenv('HF_ACCESS_TOKEN')

argparser = argparse.ArgumentParser()
argparser.add_argument("--short_config_path", type=str, default="experiments/configs/merged_short.yaml", help="Path to the config file for the short model")
argparser.add_argument("--long_config_path", type=str, default="experiments/configs/merged_long.yaml", help="Path to the config file for the long model")
argparser.add_argument("--load_step", type=int, default=300, help="Step number of the model to load")
argparser.add_argument("--which_model", type=str, default='both', choices=['short', 'long', 'both'], help="Which model to load the adapter from. Both means two adapters will be loaded and merged. Otherwise, only one adapter will be loaded.")
argparser.add_argument("--save_dir", type=str, default='experiments/files_short_long', help="Directory to save the results")
argparser.add_argument("--combination_type", type=str, default='linear', choices=['linear', 'cat', 'ties', 'dare_ties', 'dare_linear', 'svd'], help="Type of combination to use when merging two adapters")
argparser.add_argument("--extrapolate", action='store_true', help="Whether to extrapolate the results (i.e. use weights outside of [0, 1])")
argparser.add_argument("--eval_on_subset", action='store_true', help="Whether to evaluate on a subset of the test data")
argparser.add_argument("--run_left_end", action='store_true', help="Whether to run the left end of the extrapolation (i.e. w = 1.0)")
argparser.add_argument("--run_right_end", action='store_true', help="Whether to run the right end of the extrapolation (i.e. w = 0.0)")
argparser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing results")

args = argparser.parse_args()

short_config = load_yaml_as_namespace(args.short_config_path)
long_config = load_yaml_as_namespace(args.long_config_path)

short_config = update_config_with_exp_dir(short_config)
long_config = update_config_with_exp_dir(long_config)

torch_dtype = torch.bfloat16 if short_config.exp.device == 'cuda' else torch.float32

short_model_load_step = args.load_step
long_model_load_step = args.load_step

short_model_load_path = os.path.join(short_config.training.output_dir, f'checkpoint-{short_model_load_step}')
long_model_load_path = os.path.join(long_config.training.output_dir, f'checkpoint-{long_model_load_step}')

short_tokenizer = AutoTokenizer.from_pretrained(short_model_load_path)
long_tokenizer = AutoTokenizer.from_pretrained(long_model_load_path)

short_data = create_dataset(short_config)
long_data = create_dataset(long_config)

short_data.prepare(short_tokenizer)
long_data.prepare(long_tokenizer)

short_test_data = short_data.get_raw_split('test')
long_test_data = long_data.get_raw_split('test')

final_test_data = concatenate_datasets([short_test_data, long_test_data])

if args.eval_on_subset:
    final_test_data = final_test_data.select(range(16))

save_path = args.save_dir + f'_{args.which_model}'
os.makedirs(save_path, exist_ok=True)

ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

if args.extrapolate:
    ratios = [1.5, 1.4, 1.3, 1.2, 1.1] + ratios + [-0.1, -0.2, -0.3, -0.4, -0.5]
    ratios = ratios[::-1]

def save_current_results(save_dir, ios, cur_gen_lens):
    with open(f'{save_dir}/cur_gen_lens.pkl', 'wb') as f:
        pickle.dump(cur_gen_lens, f)
    
    ios = [{'prompt': i[0], 'generated': i[1], 'target': i[2], 'idx': i[3]} for i in ios]
    with open(f'{save_dir}/ios.json', 'w') as f:
        json.dump(ios, f)


for w in tqdm(ratios):
    SS = 4

    if w == 1.0 and not args.run_left_end:
        continue
    if w == 0.0 and not args.run_right_end:
        continue
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        short_config.model.model_name,
        token=hf_access_token,
        torch_dtype=torch_dtype,
        max_length=short_config.data.max_length,
    )
    base_model.to(short_config.exp.device)
    base_model.eval()

    if args.which_model in ['short', 'long']:
        # Single adapter mode: choose the adapter and corresponding weight
        adapter_name = f"{args.which_model}_adapter"
        # Choose the weight: w
        scale_weight = w
        # Load the appropriate adapter (using the proper load path)
        adapter_load_path = short_model_load_path if args.which_model == 'short' else long_model_load_path
        peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_load_path,
            adapter_name=adapter_name
        )
        
        # Scale the adapter parameters by the chosen weight
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                module.lora_A[adapter_name].weight.data *= np.sqrt(np.abs(scale_weight)) * (1 if scale_weight > 0 else -1)
            if hasattr(module, "lora_B") and adapter_name in module.lora_B:
                module.lora_B[adapter_name].weight.data *= np.sqrt(np.abs(scale_weight))
            if hasattr(module, "lora_embedding_A") and adapter_name in module.lora_embedding_A:
                module.lora_embedding_A[adapter_name].data *= np.sqrt(np.abs(scale_weight)) * (1 if scale_weight > 0 else -1)
            if hasattr(module, "lora_embedding_B") and adapter_name in module.lora_embedding_B:
                module.lora_embedding_B[adapter_name].data *= np.sqrt(np.abs(scale_weight))
        
        peft_model.set_adapter(adapter_name)
        if args.which_model == 'short':
            peft_model.to(short_config.exp.device)
        else:
            peft_model.to(long_config.exp.device)
        peft_model.eval()

    else:
        # Two-adapter merging (as in your original code)
        peft_model = PeftModel.from_pretrained(
            base_model,
            short_model_load_path,
            adapter_name="short_adapter"
        )
        peft_model.load_adapter(long_model_load_path, adapter_name="long_adapter")
        
        adapters = ["short_adapter", "long_adapter"]
        weights = [w, 1 - w]
        
        if args.combination_type != 'cat':
            for adapter_name, weight_val in zip(adapters, weights):
                if weight_val < 0:
                    for name, module in peft_model.named_modules():
                        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                            module.lora_A[adapter_name].weight.data *= -1
                        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
                            module.lora_B[adapter_name].weight.data *= -1
                        if hasattr(module, "lora_embedding_A") and adapter_name in module.lora_embedding_A:
                            module.lora_embedding_A[adapter_name].data *= -1
                        if hasattr(module, "lora_embedding_B") and adapter_name in module.lora_embedding_B:
                            module.lora_embedding_B[adapter_name].data *= -1
            final_weights = [abs(weight_val) for weight_val in weights]
        else:
            final_weights = weights
        
        peft_model.add_weighted_adapter(
            adapters=adapters,
            weights=final_weights,
            adapter_name="merged_adapter",
            combination_type=args.combination_type,
            density=1.0
        )
        peft_model.set_adapter("merged_adapter")
        peft_model.to(short_config.exp.device)
        peft_model.eval()

    del base_model


    save_dir = f'{save_path}/{args.combination_type}/{int(w * 100)}'
    # check if save_dir exists
    if os.path.exists(save_dir) and not args.overwrite:
        print(f'Skipping {save_dir} as it already exists')
        del peft_model
        continue
    os.makedirs(save_dir, exist_ok=True)
    ios = []
    cur_gen_lens = []
    for idx, sample in enumerate(tqdm(final_test_data)):
        prompt_text = sample['_input']
        wint_model_input = short_tokenizer(prompt_text, return_tensors='pt')
        
        MARGIN = 256
        if wint_model_input['input_ids'].shape[1] > short_config.data.max_length - MARGIN:
            print(f'Skipping due to length: {wint_model_input["input_ids"].shape[1]}')
            continue
        
        wint_model_input.to(short_config.exp.device)
        with torch.no_grad():
            wint_model_output = peft_model.generate(
                **wint_model_input,
                max_length=long_config.data.max_length,
                eos_token_id=long_tokenizer.eos_token_id,
                pad_token_id=long_tokenizer.eos_token_id,
            )
        gen = get_assistant_output_raw(short_tokenizer.decode(wint_model_output[0]))
        cur_gen_lens.append(
            (len(gen.split('<|eot_id|>')[0]), idx)
        )
        target_text = get_assistant_output_raw(sample['_target'])

        ios.append((prompt_text, gen, target_text, idx))

        if idx >= SS - 1:
            save_current_results(save_dir, ios, cur_gen_lens)
            SS *= 2

    save_current_results(save_dir, ios, cur_gen_lens)

    del peft_model
