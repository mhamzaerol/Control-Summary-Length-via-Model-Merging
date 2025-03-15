from typing import Optional, Dict, Any
from transformers import PreTrainedTokenizerBase    
from datasets import load_dataset, DatasetDict
import os


def convert_to_chat_template(dataset, tokenizer):
    return dataset.map(
        lambda x: {
            **x,
            '_input': tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': f'Summarize the following text:\n{x["_input"]}'
                    }
                ], 
                tokenize=False, 
                add_generation_prompt=True, 
            ),
            '_target': tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': f'Summarize the following text:\n{x["_input"]}'
                    },
                    {
                        'role': 'assistant',
                        'content': x['_target']
                    }
                ], 
                tokenize=False,
            )
        },
        num_proc=4
    )

def preprocess_function(example, tokenizer, config):
    input = example['_input']
    target = example['_target']
    model_input = tokenizer(target, truncation=True, max_length=config.data.max_length)
    
    label = model_input["input_ids"].copy()
    prompt_ids = tokenizer(input, truncation=True, max_length=config.data.max_length)["input_ids"]
    prompt_length = len(prompt_ids)
    label[:prompt_length] = [-100] * prompt_length

    model_input["labels"] = label
    return model_input

class BaseDataset:
    def __init__(self, config):
        self.config = config
        self.load()
        self.process()
    
    def load(self):
        raise NotImplementedError

    def process(self):
        
        def merge_columns(example, input_columns, target_columns):
            # get the non-null input column
            example["_input"] = None
            for input_column in input_columns:
                if input_column in example and example[input_column] is not None:
                    example["_input"] = example[input_column]
                    break
            # get the non-null target column
            example["_target"] = None
            for target_column in target_columns:
                if target_column in example and example[target_column] is not None:
                    example["_target"] = example[target_column]
                    break
            return example
        
        good_columns = self.config.data.input_columns + self.config.data.target_columns
        for split in self.splits:
            id = getattr(self.config.data, split).id
            bad_columns = [col for col in self.dataset[id].column_names if col not in good_columns]
            self.dataset[id] = self.dataset[id].remove_columns(bad_columns)
            self.dataset[id] = self.dataset[id].map(
                lambda x: merge_columns(x, self.config.data.input_columns, self.config.data.target_columns)
            )
            existing_good_columns = [col for col in good_columns if col in self.dataset[id].column_names]
            self.dataset[id] = self.dataset[id].remove_columns(existing_good_columns)

    def load_from_hf(self):
        print(f"Loading dataset: {self.config.data.dataset_name} from Hugging Face...")
        self.dataset = load_dataset(self.config.data.dataset_name, trust_remote_code=True)
        
        self.splits = []
        for split in ["train", "val", "test"]:
            if hasattr(self.config.data, split):
                self.splits.append(split)
                split_config = getattr(self.config.data, split)
                id = split_config.id
                if hasattr(split_config, "subset_size"):
                    self.dataset[id] = self.dataset[id].select(range(split_config.subset_size))

    def load_from_disk(self):
        print(f"Loading dataset: {self.config.data.dataset_name} from disk...")
        dataset_path = f'datasets/{self.config.data.dataset_name}'
        self.splits = []
        self.dataset = {}
        for split in ["train", "val", "test"]:
            if hasattr(self.config.data, split):
                self.splits.append(split)
                split_config = getattr(self.config.data, split)
                id = split_config.id

                if not os.path.exists(f'{dataset_path}/{id}'):
                    continue
                
                variant = self.config.exp.variant
                if not os.path.exists(f'{dataset_path}/{id}/{variant}.jsonl'):
                    raise FileNotFoundError(f'File {dataset_path}/{id}/{variant}.jsonl not found')
                
                self.dataset[id] = load_dataset("json", data_files=f'{dataset_path}/{id}/{variant}.jsonl')

                if hasattr(split_config, "subset_size"):
                    self.dataset[id]['train'] = self.dataset[id]['train'].select(range(split_config.subset_size))
        self.dataset = DatasetDict({
            getattr(self.config.data, split).id: self.dataset[getattr(self.config.data, split).id]['train']
            for split in self.splits
        })

    def get_split(self, split):
        if split not in self.splits:
            return None
        return self.dataset[getattr(self.config.data, split).id]

    def get_raw_split(self, split):
        if split not in self.splits:
            return None
        return self.raw_dataset[getattr(self.config.data, split).id]
    
    def prepare(self, tokenizer):
        self.dataset = convert_to_chat_template(self.dataset, tokenizer)
        # copy the dataset to store the raw version
        self.raw_dataset = self.dataset.copy()
        self.dataset = self.dataset.map(
            lambda x: preprocess_function(x, tokenizer, self.config),
            num_proc=4,
            remove_columns=self.dataset[getattr(self.config.data, self.splits[0]).id].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

class DatasetHF(BaseDataset):
    def load(self):
        self.load_from_hf()

class DatasetDisk(BaseDataset):
    def load(self):
        self.load_from_disk()

def create_dataset(config):
    if config.data.load_hf:
        return DatasetHF(config)
    else:
        return DatasetDisk(config)