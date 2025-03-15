import argparse

from datasets import load_dataset
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

xsum_ds = load_dataset("EdinburghNLP/xsum")
cnndm_ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

SHUFFLE_SEED = 0
OOD_FILTER_LENGTH = 512
SHORT_LENGTH = 128
# To create shortest dataset
# SHORT_LENGTH = 64

LONG_LENGTH = 256
LONG_MAP_LENGTH = 320
TRAIN_ITEM_COUNT = 2000
VALID_ITEM_COUNT = 250
TEST_ITEM_COUNT = 250

client = OpenAI(
    api_key=openai_api_key
)


def short_openai_api(text, length):
    print("*********** LLM CALL")
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content":
             f"""Promptly summarize the following text to under {length} characters.

Text: {text}

Print out the reduced summary only, which should be under {length} characters."""
         }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    output = completion.choices[0].message.content
    # if len(output) > SHORT_LENGTH:
    #     raise ValueError(f"Output Too Long: {output}")
    # Length control honor system
    print(f"Original: {text}")
    print(f"Output: {output}")
    return output


def all_map_func(x, sum_key):
    x[sum_key] = x[sum_key].replace("\n", "").replace(" .", ". ")
    return x


def short_map_func(x, sum_key):
    if len(x[sum_key]) < SHORT_LENGTH:
        return x
    x[sum_key] = short_openai_api(x[sum_key], SHORT_LENGTH)
    return x


def long_map_func(x, sum_key):
    if len(x[sum_key]) >= LONG_MAP_LENGTH:
        x[sum_key] = short_openai_api(x[sum_key], LONG_MAP_LENGTH)
    return x


def rand_select(path, ds, doc_key, sum_key, split, count):
    ds = ds[split]
    ds = ds.shuffle(seed=SHUFFLE_SEED)
    ds = ds.filter(lambda x: len(x[sum_key]) < OOD_FILTER_LENGTH)
    ds = ds.map(lambda x: all_map_func(x, sum_key))
    ds = ds.map(lambda x: all_map_func(x, doc_key))

    short_ds = ds.select([i for i in range(min(len(ds), count))])
    short_ds = short_ds.map(lambda x: short_map_func(x, sum_key))

    long_ds = ds.filter(lambda x: len(x[sum_key]) > LONG_LENGTH)
    long_ds = long_ds.select([i for i in range(min(len(long_ds), count))])
    # To create long low-variance dataset, uncomment below.
    # Run src/load_256_320.py to further reduce variance of dataset.
    # long_ds = long_ds.map(lambda x: long_map_func(x, sum_key))

    path = path + "/" + split
    short_path = path + "/short.jsonl"
    long_path = path + "/long.jsonl"
    gold_path = path + "/gold.jsonl"
    short_ds.to_json(short_path)
    long_ds.to_json(long_path)
    ds.to_json(gold_path)


def run(path):
    xsum_path = path + "/xsum"
    rand_select(xsum_path, xsum_ds, "document", "summary", "train", TRAIN_ITEM_COUNT)
    rand_select(xsum_path, xsum_ds, "document", "summary", "validation", VALID_ITEM_COUNT)
    rand_select(xsum_path, xsum_ds, "document", "summary", "test", TEST_ITEM_COUNT)

    cnn_path = path + "/cnndm"
    rand_select(cnn_path, cnndm_ds, "article", "highlights", "train", TRAIN_ITEM_COUNT)
    rand_select(cnn_path, cnndm_ds, "article", "highlights", "validation", VALID_ITEM_COUNT)
    rand_select(cnn_path, cnndm_ds, "article", "highlights", "test", TEST_ITEM_COUNT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    run(args.path)
