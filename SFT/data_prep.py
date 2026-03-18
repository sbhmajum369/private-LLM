"""
Copyright (c) 2026 Subhranil Majumder 
Licensed under the GPL-3.0 license
See LICENSE file in the project root for full license information 
""" 

""" Data Processing """ 

import os 
from datasets import load_dataset
from transformers import AutoTokenizer 
from pprint import pprint 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from huggingface_hub import snapshot_download 


Tested_Models = ["mistralai/Mistral-7B-v0.3", "mistralai/Ministral-3-3B-Base-2512", "Qwen/Qwen2.5-7B", "LiquidAI/LFM2.5-1.2B-Base"] 
local_model_path = os.path.abspath(f"./pre-trained_models") 
MODEL_NAME = Tested_Models[3] 
model_src_path = os.path.join(local_model_path, MODEL_NAME) 


def tokenizer_prep_Mistral():
    tokenizer = AutoTokenizer.from_pretrained(model_src_path) 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    return tokenizer 

def tokenizer_prep_ChatML():
    tokenizer = AutoTokenizer.from_pretrained(model_src_path) 

    tokenizer.chat_template = (
        "{% if not add_generation_prompt is defined %}"
        "{% set add_generation_prompt = false %}"
        "{% endif %}"
        "{% for message in messages %}"
        "<|im_start|>{{message['role']}} \n"
        "{{message['content']}} \n"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    ) 
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}) 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    return tokenizer 


def chatml_conversation_format(example, tokenizer):
    role_mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    
    standardized_messages = [
        {
            "role": role_mapping.get(msg["from"], msg["from"]), 
            "content": msg["value"]
        }
        for msg in example["conversations"]
    ] 
    
    text = tokenizer.apply_chat_template(
        standardized_messages, 
        tokenize=False, 
        add_generation_prompt=False
    ) 

    return {"text": text}

def mistral_conversation_format(example, tokenizer):
    role_mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    
    prompt = "" 
    system_content = "" 

    for msg in example["conversations"]:
        role = role_mapping.get(msg["from"], msg["from"])
        content = msg["value"] 
        # print(role) 

        ## NEW 
        if role == 'system':
            # Store system content to prepend to the first user message
            system_content = f"{content}\n"
        elif role == 'user':
            # Prepend system content only to the first user message
            combined_content = f"{system_content}{content}" if system_content else f"{content}"
            prompt += f"[INST] {combined_content} [/INST]"
            system_content = "" # Clear after first use 
        elif role == 'assistant':
            prompt += f"{content}{tokenizer.eos_token}" 
    
    return {"text": prompt} 


def mistral_nonChatML(dataset_train, dataset_eval, save_path):
    tokenizer = tokenizer_prep_Mistral() 
    
    train_data = dataset_train.map(mistral_conversation_format, remove_columns=dataset_train.column_names, batched=False, fn_kwargs={"tokenizer": tokenizer}) 
    eval_data = dataset_eval.map(mistral_conversation_format, remove_columns=dataset_eval.column_names, batched=False, fn_kwargs={"tokenizer": tokenizer}) 

    train_data = train_data.filter(lambda x: x["text"] is not None) 
    eval_data = eval_data.filter(lambda x: x["text"] is not None) 

    print(train_data[-2]["text"]) 

    train_data.save_to_disk(os.path.join(save_path, "train")) 
    eval_data.save_to_disk(os.path.join(save_path, "eval")) 

    print(f"\nData processed and Saved. \nTrain len: {len(train_data)}, Eval len: {len(eval_data)}") 

    return 

def chatml_format(dataset_train, dataset_eval, save_path):
    tokenizer = tokenizer_prep_ChatML() 
        
    train_data = dataset_train.map(chatml_conversation_format, remove_columns=dataset_train.column_names, batched=False, fn_kwargs={"tokenizer": tokenizer}) 
    eval_data = dataset_eval.map(chatml_conversation_format, remove_columns=dataset_eval.column_names, batched=False, fn_kwargs={"tokenizer": tokenizer}) 
     
    train_data = train_data.filter(lambda x: x["text"] is not None) 
    eval_data = eval_data.filter(lambda x: x["text"] is not None) 

    print(train_data[-2]["text"])

    train_data.save_to_disk(os.path.join(save_path, "train")) 
    eval_data.save_to_disk(os.path.join(save_path, "eval")) 

    print(f"\nData processed and Saved. \nTrain len: {len(train_data)}, Eval len: {len(eval_data)}") 

    return 


def main():
    ## Model download 
    os.makedirs(local_model_path, exist_ok=True) 

    if not os.path.isdir(model_src_path):
        try:
            snapshot_download(
                repo_id= MODEL_NAME,
                local_dir=os.path.join(local_model_path, MODEL_NAME), 
            ) 
            print(f"\nModel saved at {model_src_path}")
        except Exception as ex:
            print(ex)
            return 
    elif len(os.listdir(model_src_path)) == 0:
        try:
            snapshot_download(
                repo_id= MODEL_NAME,
                local_dir=os.path.join(local_model_path, MODEL_NAME), 
            ) 
            print(f"\nModel saved at {model_src_path}")
        except Exception as ex:
            print(ex)
            return 
    else:
        print("DIrectory not Empty\n")
    
    ## Dataset prep 
    dataset = load_dataset("mlabonne/FineTome-100k", split="train") 
    dataset_save_path = os.path.abspath(f"dataset/dataset_test") 
    
    dataset_tr = dataset #.select(range(50000)) 
    dataset_ev = dataset.select(range(5000)) 

    print(f"\n Loading Tokenizer for: {MODEL_NAME}\n") 

    ## Apply conversation formatting 
    # Mistral variants 
    # mistral_nonChatML(dataset_tr, dataset_ev, dataset_save_path) 

    # Qwen, LFM other ChatML format variants 
    chatml_format(dataset_tr, dataset_ev, dataset_save_path) 

    return 


if __name__ == "__main__":
    main() 


