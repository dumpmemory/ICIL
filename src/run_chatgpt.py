import glob
import json
import openai
import tqdm
import os
import random
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
from itertools import chain
import time
import sys
import spacy
import numpy as np
import random
import re


openai.api_key=os.environ["openai_key"]

@dataclass
class GPT3Arguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/gpt3/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    gpt3_temperature: float = field(
        default=0, metadata={"help": "the temprature of GPT3."}
    )
    gpt3_top_p: float = field(
        default=1, metadata={"help": "the top_p parameter of GPT3."}
    )
    engine: str = field(
        default="text-davinci-001", metadata={"help": "the openai GPT3 engine to use."}
    )
    icil: bool = field(
        default=False, metadata={"help": "Either you will use ICIL or not."}
    )
    demo_path: str = field(
        default="", metadata={"help": "Path to file containing demo extracted by clustering."}
    )
    adaptive: bool = field(
        default=False, metadata={"help": "Adaptively change the number of demos. This takes effect only when demo_path is not None"}
    )


if __name__ == "__main__":
    random.seed(123)

    ### WARNING: YOU MAY WANT TO EMPTY THE BELOW FILES. ###
    parser = HfArgumentParser((GPT3Arguments,))
    args, = parser.parse_args_into_dataclasses()
    
    ### Add Positive Demontrastions if original dataset has less than the specified amount.
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    ## Start of Main Code. ##
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data_collator = DataCollatorForNI(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True,
        icil=args.icil, # added
        demo_path=args.demo_path,
        adaptive=args.adaptive,
        chatGPT=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "gpt3_run_config.json"), "a") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    existing_tasks = []

    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["gpt3_input"]] = request_info["gpt3_response"]
                
                # IF already in ... exclude that.
                existing_tasks.append(request_info['id'])

    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "a") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            
            if example['id'] in existing_tasks:
                continue

            encoded_example = data_collator([example])
            example["gpt3_input"] = encoded_example["inputs"][0].strip()
            example["gpt3_target"] = encoded_example["labels"][0].strip()
            length_issue = False

            if example["gpt3_input"] in existing_requests:
                response = existing_requests[example["gpt3_input"]]
            else:
                # call GPT-3 API until result is provided and then return it
                response = None
                received = False   

                while not received:
                    try:
                        # MAKE SURE INPUT IS NOT CUT:
                        if example["gpt3_input"][-7:].strip() != "Output:":
                            length_issue = True
                            break

                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages = [{"role": "user", "content": example["gpt3_input"]}],
                            #   prompt=source,
                            temperature=args.gpt3_temperature,
                            max_tokens=128,
                            top_p=args.gpt3_top_p,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=["\n\n"]
                        )

                        received = True
                    except:
                        error = sys.exc_info()[0]
                        if error == openai.error.InvalidRequestError: 
                            # something is wrong: e.g. prompt too long
                            print(f"InvalidRequestError\nPrompt passed in:{example['gpt3_input']}\n\n")
                            assert False
                        print("API error:", error)
                        time.sleep(2)

            if length_issue:
                print("[Length issue] Skipping...")
                length_issue = False
                continue

            example["gpt3_response"] = response
            # print(response)
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["gpt3_response"]["choices"][0]["message"]["content"].strip().split(".")[0]
            print(example["prediction"])
            fout.write(json.dumps(example) + "\n")