""" Merging and Inference """ 

import torch 
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration 
from peft import PeftModel 


BASE_MODEL = "./trained_models_SFT/Ministral3_3B_IFT_2" 
ADAPTER_PATH = "./trained_models_DPO/Ministral3_3B_DPO_adapter_1"
model_id = "./trained_models_DPO/Ministral3_3B_DPO_merged_1" 


## MERGE ## 
def merge_adapter():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH) 
    base_sft_model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        device_map = "auto",
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2"
    ) 

    model = PeftModel.from_pretrained(base_sft_model, ADAPTER_PATH) 
    merged_model = model.merge_and_unload() 

    merged_model.save_pretrained(
        model_id, 
        safe_serialization=True, 
        max_shard_size="2GB",
        save_original_format=False,   # key change
    )
    ## Save the Tokenizer 
    tokenizer.save_pretrained(model_id) 

    return 

def inference(usr_query:str):
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, 
        device_map = "auto",
        dtype="bfloat16",
        # attn_implementation="flash_attention_2"
    ) 

    # Generate answer 
    # system_prompt_1 = "You are a helpful and precise question answering assistant, who always provide the best factually correct answer" 
    system_prompt_1 = "You are an intelligent and precise question answering assistant, who always provide the best factually correct answer. Only generate the option. No Explanation." 
    system_prompt_2 = "You are an independent thinker who provides precise and factually correct answer always. Only generate the option. No Explanation." 

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
    prompt = "How many types of RL algo are there?" #"What are the top LLMs and their respective architectures?" 
    # prompt = "Reply the words 'doughnut', 'APLE', 'STRBRY' and 'excalibar'. Don't provide any explanation." 
    # prompt = "Could you elaborate on the potential psychological impacts of adhering to the philosophy of money as a good servant but a bad master, particularly in relation to stress and overall mental health?" 
    # prompt = "How would the society of Luminestra handle conflicts and disagreements that arise from the participatory democratic governance model?" 

    merge_adapter() 
    torch.cuda.empty_cache() 

    inference(prompt) 

    return 


if __name__ == "__main__":
   main() 

