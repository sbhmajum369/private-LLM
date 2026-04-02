""" Processing a Multi-COnversation dataset into a single-turn dataset """

import os 
from datasets import load_dataset, load_from_disk 
import torch 
from transformers import AutoTokenizer 
from pprint import pprint 


## For Single-turn conversation 

def is_valid_pair(chosen_msgs, rejected_msgs):
   if not chosen_msgs or not rejected_msgs:
      return False
   if len(chosen_msgs) != len(rejected_msgs):
      return False

   # roles must match turn-by-turn
   for c, r in zip(chosen_msgs, rejected_msgs):
      if c["role"] != r["role"]:
         return False

   return True

def to_chat_format(example):
    return {
        "prompt": [
            {"role": "user", "content": example["prompt"].split("<|im_start|>user \n")[-1].split("<|im_end|>")[0].strip()}
        ],
        "chosen": [{"role": "assistant", "content": example["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": example["rejected"].strip()}],
    }

def extract_turns(example, tokenizer): 
   all_prompts = []
   all_chosen = []
   all_rejected = [] 

   # chosen_conv = example["chosen"]      # ✅ this is already ONE conversation
   # rejected_conv = example["rejected"]

   for chosen_conv, rejected_conv in zip(example["chosen"], example["rejected"]):
      # --- validation ---
      if not chosen_conv or not rejected_conv:
         return {"prompt": [], "chosen": [], "rejected": []}

      if len(chosen_conv) != len(rejected_conv):
         return {"prompt": [], "chosen": [], "rejected": []}
      
      # print("\n", len(chosen_conv), len(rejected_conv))
      # print(chosen_conv, "\n") 
      # print(rejected_conv, "\n") 

      # iterate over turns
      for i in range(0, len(chosen_conv) - 1, 2):
         # print(i)
         c_user = chosen_conv[i]
         c_assistant = chosen_conv[i + 1]

         r_user = rejected_conv[i]
         r_assistant = rejected_conv[i + 1]

         # ensure correct roles
         if c_user["role"] != "user" or c_assistant["role"] != "assistant":
            continue

         if r_user["role"] != "user" or r_assistant["role"] != "assistant":
            continue

         # ensure same prompt
         if c_user["content"].strip() != r_user["content"].strip():
            continue

         prompt_text = c_user["content"].strip()
         chosen_resp = c_assistant["content"].strip()
         rejected_resp = r_assistant["content"].strip()

         # skip invalid
         if not prompt_text or not chosen_resp or not rejected_resp:
            continue

         if chosen_resp == rejected_resp:
            continue 

         # format ONLY prompt
         prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
         )

         all_prompts.append(prompt)
         all_chosen.append(chosen_resp)
         all_rejected.append(rejected_resp)
   
   # print(f"\n{len(all_chosen)}, {len(all_rejected)}, {len(all_prompts)}") 

   return {
      "prompt": all_prompts,
      "chosen": all_chosen,
      "rejected": all_rejected,
   }


def main():
   torch.cuda.empty_cache() 

   model_name = "./trained_models_SFT/Qwen25_3B_IFT_1" 
   
   # number of samples are directly proportional to the training time. 
   dataset_dpo = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train") 

   dataset_dpo = dataset_dpo.filter(lambda r: r["source"] != "toxic-dpo-v0.2") 
   print(f"Dataset Len: {len(dataset_dpo)}") 

   dataset_dpo = dataset_dpo.select(range(5000)) 
   print(f"Selected Dataset Len: {len(dataset_dpo)}") 
   # pprint(dataset_dpo[:2])

   ##### 
   print("📚 Loading tokenizer...") 
   tokenizer = AutoTokenizer.from_pretrained(model_name) 
   if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token 
   tokenizer.padding_side = "right" 
   

   ##  Single-turn 
   dataset = dataset_dpo.map(lambda x: extract_turns(x, tokenizer), remove_columns=dataset_dpo.column_names, batched=True) 
   # remove empty 
   dataset = dataset.filter(lambda x: x is not None and len(x["chosen"]) > 0) 
   dataset = dataset.map(to_chat_format)
   
   # print("\n After sample >>") 
   # for data in dataset:
   #    pprint(data["prompt"]) 
   # pprint(dataset[0]["prompt"]) 
   # pprint(dataset[0]["chosen"]) 

   # print(type(dataset[0]["prompt"])) 

   # pprint(dataset[0]["chosen"]) 
   # pprint(dataset[0]["rejected"]) 
   print(len(dataset)) 
   # pprint(dataset[:1]) 

   dataset.to_json("./dataset/orpo-dpo_mix_5k.jsonl") 

   return 


if __name__ == "__main__":
   main() 

