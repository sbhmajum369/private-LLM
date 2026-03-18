"""
Copyright (c) 2026 Subhranil Majumder 
Licensed under the GPL-3.0 license
See LICENSE file in the project root for full license information 
""" 

""" Base model Supervised Fine-tuning from scratch, for Instruction-following tasks """

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from datasets import load_from_disk, load_dataset 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mistral3ForConditionalGeneration 
from peft import LoraConfig 
from trl import SFTTrainer, SFTConfig, clone_chat_template 

from huggingface_hub import snapshot_download 



Tested_Models = ["mistralai/Mistral-7B-v0.3", "mistralai/Ministral-3-3B-Base-2512", "Qwen/Qwen2.5-7B", "LiquidAI/LFM2.5-1.2B-Base"] 
local_model_path = os.path.abspath(f"./pre-trained_models") 
MODEL_NAME = Tested_Models[1] 
OUTPUT_DIR = "./trained_models/Ministral3_3B_adapter_1" 

context_length = 1024 #2048 

############ 


def tokenizer_prep_Mistral(tokenizer):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    return tokenizer 

def tokenizer_prep_ChatML(tokenizer):

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


def main():
    torch.cuda.empty_cache() 

    ## Model download 
    os.makedirs(local_model_path, exist_ok=True) 
    
    model_src_path = os.path.join(local_model_path, MODEL_NAME) 

    if (len(os.listdir(model_src_path)) == 0) or (not os.path.isdir(model_src_path)):
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
        print("\nBase Model detected!") 
        
    
    train_ds = load_from_disk("dataset/dataset_ministral/train") 
    # eval_ds = load_from_disk("dataset/dataset_ministral/eval")  
    
    print(f"Model: {MODEL_NAME}\n") 

    ## Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_src_path) 

    ## Choose the proper tokenizer format 
    # tokenizer = tokenizer_prep_Mistral(tokenizer) 
    tokenizer = tokenizer_prep_ChatML(tokenizer)

    ## Quantization config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) 
    
    ## Load 4-bit quantized model 
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_src_path, #MODEL_NAME,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     # trust_remote_code=True,
    #     # attn_implementation = "sdpa",  # sdpa (packing=False) flash_attention_3
    # ) 

    ## For Ministral variants, load 4-bit quantized model 
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_src_path, 
        device_map="auto", 
        quantization_config=bnb_config
    ) 

    ## For full SFT using un-quantized model 
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_src_path,
    #     dtype=torch.bfloat16,
    #     device_map="auto"
    # ) 
    
    
    model.resize_token_embeddings(len(tokenizer)) 
    model.config.eos_token_id = tokenizer.eos_token  
    model.config.pad_token_id = tokenizer.pad_token_id 

    ## verifying final data 
    # print("\n", train_ds[-1]["text"], "\n") 
    # tokens = tokenizer(train_ds[-1]["text"]) 
    # print(tokenizer.decode(tokens["input_ids"]), "\n") 
    # return 

    model.config.use_cache = False 

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        # target_modules = "all-linear", 
        lora_dropout = 0.05,
        # use_rslora=False,
        # use_dora=True,
        bias="none",
        task_type="CAUSAL_LM",
    ) 

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        # dataset_num_proc=4,
        learning_rate = 3e-4, # 3e-4 
        logging_steps = 5,
        num_train_epochs = -1,   # OR -1, OR 1 
        max_steps = 150,         # OR -1, OR 300 
        # eval_strategy="steps",
        # eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        warmup_steps=10,
        lr_scheduler_type="cosine",     # cosine_with_restarts | cosine 
        # load_best_model_at_end = True,
        bf16 = True,  
        optim="adamw_8bit", # adamw_8bit | paged_adamw_8bit 
        max_length = context_length,
        packing=False,
        report_to="none",  # report_to="tensorboard" 
        completion_only_loss=True,
        dataset_text_field="text",
    ) 
    
    trainer = SFTTrainer(
        model=model,
        train_dataset = train_ds, 
        # eval_dataset = eval_ds, 
        peft_config = peft_config,
        processing_class = tokenizer,
        args=training_args, 
    )

    # Printing model param 
    trainer.model.print_trainable_parameters() 

    ## Training start 
    if len(os.listdir(OUTPUT_DIR)) > 2:
        trainer.train(resume_from_checkpoint=True) 
    else:
        trainer.train() 
    
    ## Saving adapter and tokenizer 
    trainer.save_model(OUTPUT_DIR) 
    tokenizer.save_pretrained(OUTPUT_DIR) 

    return 


if __name__ == "__main__":
    main() 


