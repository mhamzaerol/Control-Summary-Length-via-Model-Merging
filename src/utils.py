from transformers import TrainerCallback
import random
import torch
import yaml
from types import SimpleNamespace
import os
from transformers import DataCollatorForSeq2Seq, Trainer

def get_assistant_output_raw(raw_output):
    return raw_output.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()

class PrintModelOutputCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_debug_dataset, config, num_samples=5):
        """
        `raw_eval_dataset` should be a version of your eval data that contains the original 'document' (prompt)
        """
        self.tokenizer = tokenizer
        self.raw_debug_dataset = raw_debug_dataset
        self.num_samples = num_samples
        self.config = config

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model", None)
        if model is None and "trainer" in kwargs:
            model = kwargs["trainer"].model
        if model is None:
            print("No model found in callback!")
            return

        model.eval()
        sample_indices = random.sample(range(len(self.raw_debug_dataset)), 2 * self.num_samples)
        cnt_ran = 0
        print("\n--- Sample Outputs During Evaluation ---")
        for idx in sample_indices:
            if cnt_ran == self.num_samples:
                break
            # Get the prompt text from the raw dataset
            prompt_text = self.raw_debug_dataset[idx]['_input']
            # Tokenize only the prompt
            MARGIN = 256
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            if inputs['input_ids'].shape[1] > self.config.data.max_length - MARGIN:
                continue 
                
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, 
                    max_length=self.config.data.max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            output_text = self.tokenizer.decode(gen_ids[0])
            model_output = get_assistant_output_raw(output_text)
            print("-" * 150)
            print(f"\n\033[1mSample:\033[0m {idx}")
            print(f"\033[1mOutput:\033[0m {model_output}")
            print(f"\033[1mGold:\033[0m {get_assistant_output_raw(self.raw_debug_dataset[idx]['_target'])}\n")
            del inputs, gen_ids
            torch.cuda.empty_cache()
            cnt_ran += 1
        print("=" * 150)


class DebugTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        
        model.train()

        import ipdb; ipdb.set_trace()

        # Run the model forward pass on the current batch.
        outputs = model(**inputs)
        
        # Assuming the model returns a ModelOutput or dict with a 'logits' key.
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]
        
        print("Model output logits shape:", logits.shape)

        # Get the loss from the model outputs.
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[1]
        return loss.detach()


class DebugDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, **kwargs):
        self.__tokenizer=kwargs['tokenizer']
        super().__init__(**kwargs)
    def __call__(self, features):
        batch = super().__call__(features)
        
        print("Batch input_ids shape:", batch["input_ids"].shape)
        print("Batch labels shape:", batch["labels"].shape)
        print("Sample input_ids (first example):", batch["input_ids"][0])
        print("Sample labels (first example):", batch["labels"][0])
        print("Sample input_ids decoded", self.__tokenizer.decode(batch["input_ids"][0]))
        
        # Set a breakpoint.
        ipdb.set_trace()
        return batch


def make_max_length(dataset, max_length):
    def repeat_until_length(text, length):
        while len(text) < length:
            text += text
        return text[:length]

    dataset = dataset.map(
        lambda x: {
            **x,
            '_input': repeat_until_length(x['_input'], max_length * 10)
        },
        num_proc=4,
        load_from_cache_file=False,
    )
    return dataset

def dict_to_namespace(d):
    """Recursively converts a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def load_yaml_as_namespace(file_path):
    """Loads a YAML file and converts it into a nested SimpleNamespace."""
    with open(file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return dict_to_namespace(yaml_data)

def make_exp_dir(config):
    exp_dir = f'experiments/runs/{config.exp.exp_name}/'
    if os.path.exists(exp_dir):
        response = input(f"Experiment {config.exp.exp_name} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit()
        else:
            print("Continuing...")
            # remove the dir
            os.system(f"rm -r {exp_dir}")
    # create the exp_dir
    os.makedirs(exp_dir)

def update_config_with_exp_dir(config):
    exp_dir = f'experiments/runs/{config.exp.exp_name}/'
    if hasattr(config, 'training'):
        if hasattr(config.training, 'output_dir'):
            config.training.output_dir = exp_dir + config.training.output_dir
        if hasattr(config.training, 'logging_dir'):
            config.training.logging_dir = exp_dir + config.training.logging_dir
    return config

def manage_exp_dirs(config):
    make_exp_dir(config)
    config = update_config_with_exp_dir(config)
    return config