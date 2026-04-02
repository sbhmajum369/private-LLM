""" Merging and Inference """ 

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from peft import PeftModel 


BASE_MODEL = "./trained_models_SFT/Qwen25_3B_IFT_2" 
ADAPTER_PATH = "./trained_models_DPO/Qwen25_3B_DPO_adapter_4" 
model_id = "./trained_models_DPO/Qwen25_3B_DPO_merged_4" 


## MERGE ## 
def merge_adapter():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) 

    base_sft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,   # or torch.float16
        low_cpu_mem_usage=True,
        device_map = "auto",
    ) 

    model = PeftModel.from_pretrained(base_sft_model, ADAPTER_PATH) 
    merged_model = model.merge_and_unload() 

    merged_model.save_pretrained(
        model_id, 
        safe_serialization=True, 
        max_shard_size="2GB",
        save_original_format=False,   # key change
    ) 
    ## Save the Tokenizer (Critical for the chat_template to exist in the folder)
    tokenizer.save_pretrained(model_id) 

    return 

def inference(usr_query:str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype="bfloat16",
        # attn_implementation="flash_attention_2" <- uncomment on compatible GPU
    ) 

    # Generate answer 
    system_prompt_1 = "You are a precise question answering assistant, who always provide the best factually correct answer" 
    system_prompt_2 = "You are an independent thinker who provides precise and factually correct answer always." 

    message = [
        {"role": "system", "content": system_prompt_1},
        {"role": "user", "content": usr_query}
    ]

    inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            # top_p=0.95,
            min_p=0.10,
            repetition_penalty=1.1,
            max_new_tokens=1024,
        ) 

    response = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False).strip() 

    print(f"\nUser Query: {usr_query}")
    print(f"Response: {response}") 

    return 


def main():
    # prompt = "How many types of RL algo are there?" 
    prompt = "Reply these 'doughnut', 'aaple', 'strawberry' and 'excalibar' exactly as they are. Don't provide any explanation." 
    # prompt = "Could you elaborate on the potential psychological impacts of adhering to the philosophy of money as a good servant but a bad master, particularly in relation to stress and overall mental health?" 
    # prompt = "Can you elaborate on the role of neurotransmitters in stress-induced insomnia and how this can potentially be managed or treated?" 

    merge_adapter() 
    torch.cuda.empty_cache() 

    inference(prompt)

    return 


if __name__ == "__main__":
   main() 
