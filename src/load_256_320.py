import argparse
import json
import os

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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

def process(path, to_path):
    long_path = path + "/long.jsonl"
    long_256_path = path + "/long_256_320.jsonl"

    long_256 = [line for line in open(long_256_path, "r").readlines()]

    with open(to_path, 'w') as file:
        for i, line in enumerate(open(long_path, "r").readlines()):
            line_256 = long_256[i]
            item_256 = json.loads(line_256)
            item = json.loads(line)
            key = None
            if "summary" in item:
                key = "summary"
            elif "highlights" in item:
                key = "highlights"
            else:
                continue
            summary_256 = item_256[key]
            summary = item[key]
            tries = 0
            while len(summary_256) < 256 and tries < 3:
                print(f"Rewrite {tries}: {summary_256}")
                summary_256 = short_openai_api(summary, 320)
                tries += 1
            item[key] = summary_256
            json.dump(item, file)
            file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("to_path")
    args = parser.parse_args()
    # run(args.path)
    process(args.path, args.to_path)
