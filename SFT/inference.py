"""
Copyright (c) 2026 Subhranil Majumder 
Licensed under the GPL-3.0 license
See LICENSE file in the project root for full license information 
""" 

""" Inference """ 

import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mistral3ForConditionalGeneration 
from peft import PeftModel 
from datasets import load_from_disk 


Tested_Models = ["mistralai/Mistral-7B-v0.3", "mistralai/Ministral-3-3B-Base-2512", "Qwen/Qwen2.5-7B", "LiquidAI/LFM2.5-1.2B-Base"] 

local_model_path = os.path.abspath(f"./pre-trained_models") 

MODEL_NAME = Tested_Models[1] 
BASE_MODEL = os.path.join(local_model_path, MODEL_NAME) 

ADAPTER_PATH = "./trained_models/Ministral3_3B_adapter_1" 
FINAL_SAVE_PATH = "./trained_models/Ministral3_3B_IFT_1" 

DEVICE = "auto" #"cuda:0" if torch.cuda.is_available() else "cpu" 


def merge_save(quantized_4bit:bool=True):
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) 

    if quantized_4bit:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map = DEVICE, #"auto",
            # dtype=torch.float16,
        ) 
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16, # or torch.float16
            device_map = DEVICE, # "cuda:0",
            low_cpu_mem_usage=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    base_model.resize_token_embeddings(len(tokenizer))  

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) 
    merged_model = model.merge_and_unload() 

    # merged_model.config.pad_token_id = tokenizer.pad_token_id

    merged_model.save_pretrained(
        FINAL_SAVE_PATH, 
        safe_serialization=True, 
        max_shard_size="2GB"
    )

    ## Save the Tokenizer (Critical for the chat_template to exist in the folder)
    tokenizer.save_pretrained(FINAL_SAVE_PATH) 

    return 

def merge_save_mistral(quantized_4bit:bool=True):
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) 

    if quantized_4bit:
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            BASE_MODEL, 
            device_map=DEVICE, 
            quantization_config=bnb_config
        )
    else:
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16, # or torch.float16
            device_map = DEVICE, # "cuda:0",
            low_cpu_mem_usage=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    base_model.resize_token_embeddings(len(tokenizer))  

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) 
    merged_model = model.merge_and_unload() 

    # merged_model.config.pad_token_id = tokenizer.pad_token_id

    merged_model.save_pretrained(
        FINAL_SAVE_PATH, 
        safe_serialization=True, 
        max_shard_size="2GB"
    )

    ## Save the Tokenizer (Critical for the chat_template to exist in the folder)
    tokenizer.save_pretrained(FINAL_SAVE_PATH) 

    return 


def load_model_and_tokenizer(quantized_4bit:bool=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) 

    tokenizer = AutoTokenizer.from_pretrained(FINAL_SAVE_PATH)

    if quantized_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            FINAL_SAVE_PATH,
            quantization_config = bnb_config, 
            # dtype=torch.bfloat16,
            device_map = DEVICE, #"auto",
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            FINAL_SAVE_PATH,
            # quantization_config = bnb_config, 
            # dtype=torch.bfloat16,
            device_map = DEVICE,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
        ) 
    model.config.use_cache = False

    return model, tokenizer 

def load_model_and_tokenizer_mistral(quantized_4bit:bool=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) 

    tokenizer = AutoTokenizer.from_pretrained(FINAL_SAVE_PATH)

    if quantized_4bit:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            FINAL_SAVE_PATH,
            quantization_config = bnb_config, 
            # dtype=torch.bfloat16,
            device_map = DEVICE, #"auto",
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
        )
    else:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            FINAL_SAVE_PATH, 
            device_map = DEVICE,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
        ) 
    model.config.use_cache = False 

    return model, tokenizer 


def inference(model, tokenizer, query:str=""):
    if len(query) == 0:
        query = "Create a python function to calculate the sum of a sequence of integers. here are the inputs [1, 2, 3, 4, 5]" 
    

    # Build a chat prompt manually 
    system_prompt = "You are a helpful question answering assistant, who always provide the best factually correct answer" 

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ] 

    ## Apply Template 
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    ) 
    
    ## troubleshooting 
    print(prompt)
     
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 

    # Best Config temp=0.1 and min_p=0.1 for repeatable predictable answers 
    ## Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.3,
            # top_p=0.99,
            min_p=0.10,
            # use_cache=True,
            # repetition_penalty = 1.1,
        )
    
    # print("\nRaw O/P => ", tokenizer.decode(outputs[0]))

    # Decode and skip the prompt 
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip() 

    print(f"\nUser Query: {query}")
    print(" Assistant:", response) 

    return response 

def inference_mistral(model, tokenizer, query:str=""):
    if len(query) == 0:
        query = "Create a python function to calculate the sum of a sequence of integers. here are the inputs [1, 2, 3, 4, 5]" 
    
    # Build a chat prompt manually 
    system_prompt = "You are a helpful question answering assistant, who always provide the best factually correct answer" 

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ] 

    ## Apply Template  
    has_system = any(m["role"] == "system" for m in messages)

    if has_system:
        system_prompt = next(m["content"] for m in messages if m["role"] == "system")
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        prompt = (
            f"<s>###system:\n{system_prompt}\n"
            f"###user:\n{user_prompt}\n"
            f"###assistant:\n"
        )
    else:
        user_prompt = next(m["content"] for m in messages if m["role"] == "user")

        prompt = (
            f"<s>###user:\n{user_prompt}\n"
            f"###assistant:\n"
        ) 

    ## troubleshooting 
    print(prompt) 
     
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("</s>") 
    ]

    # Best Config temp=0.1 and min_p=0.1 for repeatable predictable answers 
    ## Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.3,
            # top_p=0.95,
            min_p=0.10,
            # use_cache=True,
            eos_token_id = stop_ids,
            repetition_penalty = 1.1,
        )
    
    # print("\nRaw O/P => ", tokenizer.decode(outputs[0]))

    # Decode and skip the prompt 
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip() 

    print(f"\nUser Query: {query}")
    print(" Assistant:", response) 

    return response 


def main():
    ################### 

    ## Adapter Testing 
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # ) 

    # tokenizer_lora = AutoTokenizer.from_pretrained(ADAPTER_PATH) 

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     # dtype=torch.bfloat16,
    #     quantization_config=bnb_config,
    #     device_map = "auto", #"cuda:0",
    # ) 

    # # base_model = Mistral3ForConditionalGeneration.from_pretrained(
    # #     BASE_MODEL, 
    # #     device_map="auto", 
    # #     quantization_config=bnb_config
    # # ) 
    
    # if tokenizer_lora.pad_token is None:
    #     tokenizer_lora.pad_token = tokenizer_lora.eos_token 
    
    # base_model.resize_token_embeddings(len(tokenizer_lora)) 

    # lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) 

    ################### 
    
    ## Merging and Saving 
    
    # # merge_save() 
    # merge_save_mistral()
    # torch.cuda.empty_cache() 
    # print("\nLoRA adapter merging complete!\n") 
    # return 

    ## Inference 
    dataset_eval = load_from_disk("dataset/dataset_RL_1/rl_eval") 
    eval_samples = dataset_eval.select(range(5)) 

    # model_4bit, tokenizer = load_model_and_tokenizer(False) 
    model_4bit, tokenizer = load_model_and_tokenizer_mistral(False) 
    
    questions = [
        "Say the words 'doughnot', 'apple' and 'excalibar'. Don't provide any explanation.",
        "What is (2.5+2.3)x4=? Break it down into individual steps to reach the final answer",
        "Design an algorithm for sorting a list of integers, in python, with lowest time complexity",
        "Write a highly effecient sorting algorithm in Python that has lowest time complexity. Use a random number generator to generate >100 number in a list to test it.",
        "Continue the fibonacci sequence after this, until you reach a 3-digit number.\n 1, 1, 2, 3, 5, 8, ...", 
        "Capital of Netherlands? What is it's population?", 
        "what is/are the official language(s) of India?",
        "If it is 3:15 PM, what is the angle between the hour and minute hand? (Eg. when it's 3:00 AM/PM the angle is 90 degrees)",
        "A book has 200 pages. How many times does the digit '9' appear? Show in details"
    ] 

    # for q in questions:
    #     user_msg = questions[4] 
    #     # inference(lora_model, tokenizer_lora, user_msg) 
    #     inference(model_4bit, tokenizer, user_msg) 
    #     print("\n============") 
    
    user_msg = questions[2] 
    # inference(lora_model, tokenizer_lora, user_msg) 
    inference_mistral(model_4bit, tokenizer, user_msg) # As of my last update in 2023, the population of Amsterdam is approximately 846,000 people

    return 


if __name__ == "__main__":
    main()



