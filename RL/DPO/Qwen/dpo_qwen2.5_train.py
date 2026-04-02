""" Training DPO """
import os 
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset 
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from trl import DPOConfig, DPOTrainer 
from peft import LoraConfig 
from pprint import pprint 
import json 


model_name = "./trained_models_SFT/Qwen25_3B_IFT_2" 
OUTPUT_DIR = "./trained_models_DPO/Qwen25_3B_DPO_adapter_4" 
LOG_DIR = "./trained_models_DPO/Qwen25_3B_DPO_adapter_4/logs" 
EVAL = False 


QWEN_CHAT_TEMPLATE = (
   "{%- if not add_generation_prompt is defined -%}"
   "{%- set add_generation_prompt = false -%}"
   "{%- endif -%}"
   "{%- for message in messages -%}"
   "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
   "{%- endfor -%}"
   "{%- if add_generation_prompt -%}"
   "{{'<|im_start|>assistant\n'}}"
   "{%- endif -%}"
) 


def main():
   torch.cuda.empty_cache() 
   # torch.set_num_threads(1)

   dataset = load_dataset("json", data_files="dataset/orpo-dpo_mix_5k.jsonl", split="train")

   dataset_dpo_split = dataset.train_test_split(test_size=0.3, seed=42) 
   train_dataset_dpo, eval_dataset_dpo = dataset_dpo_split['train'], dataset_dpo_split['test'] 
   print(f"\n Train Data len: {len(train_dataset_dpo)}") 

   # pprint(train_dataset_dpo[0]["prompt"]) 
   # pprint(train_dataset_dpo[0]["chosen"]) 
   # pprint(train_dataset_dpo[1]["prompt"]) 
   # pprint(train_dataset_dpo[1]["chosen"]) 

   # return 

   print("📚 Loading tokenizer...") 
   tokenizer = AutoTokenizer.from_pretrained(model_name) 
   tokenizer.chat_template = QWEN_CHAT_TEMPLATE 

   if tokenizer.pad_token is None:
      tokenizer.pad_token = "<|endoftext|>"
   
   tokenizer.padding_side = "right" 

   # tokenizer.eos_token = "<|im_end|>"
   # tokenizer.pad_token = "<|endoftext|>"
   
   print("🧠 Loading model...") 
   # bnb_config = BitsAndBytesConfig(
   #    load_in_4bit=True,
   #    bnb_4bit_compute_dtype=torch.bfloat16,
   #    bnb_4bit_quant_type="nf4",
   #    bnb_4bit_use_double_quant=True,
   # ) 

   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      # quantization_config=bnb_config,
      low_cpu_mem_usage=True,
      # dtype= torch.bfloat16,
   ) 
   
   model.config.use_cache = False 
   
   lora_config = LoraConfig(
      r=32,
      lora_alpha=64, 
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
      # target_modules = "all-linear", 
      lora_dropout = 0.03,
      bias="none",
      task_type="CAUSAL_LM",
   ) 

   # DPO Training configuration 
   dpo_config = DPOConfig(
      output_dir=OUTPUT_DIR,
      num_train_epochs = -1, 
      max_steps = 120, 
      per_device_train_batch_size=2,
      gradient_accumulation_steps=4,
      gradient_checkpointing=True,
      learning_rate=3e-5, # recommended: 5e-6 
      lr_scheduler_type="cosine", 
      optim="adamw_8bit",
      beta=0.2,
      # loss_type="sigmoid", 
      # loss_type=["sigmoid", "bco_pair", "sft"], 
      # loss_weights=[0.8, 0.2, 0.8],
      warmup_steps=5,
      logging_steps=5,
      save_steps=20,
      max_length=1024,
      precompute_ref_log_probs=True,
      bf16 = torch.cuda.is_bf16_supported(),
      # dataloader_num_workers=0,
      logging_dir=LOG_DIR,
      report_to="tensorboard",
      seed=35,
   ) 

   ## Saving configs 
   with open(f"{OUTPUT_DIR}/dpo_config.json", "w") as f:
      json.dump(dpo_config.to_dict(), f, indent=2) 

   # Create DPO trainer
   print("🏗️ Creating DPO trainer...")
   dpo_trainer = DPOTrainer(
      model=model,
      args=dpo_config,
      train_dataset=train_dataset_dpo,
      # eval_dataset=eval_dataset_dpo,
      processing_class=tokenizer,
      peft_config=lora_config
   ) 

   # Printing model param 
   dpo_trainer.model.print_trainable_parameters()

   
   if len(os.listdir(OUTPUT_DIR)) > 2:
      dpo_trainer.train(resume_from_checkpoint=True) 
   else:
      dpo_trainer.train() 
   
   dpo_trainer.accelerator.print("✅ Training completed.")

   if EVAL:
      metrics = dpo_trainer.evaluate() 
      dpo_trainer.log_metrics("eval", metrics) 
      dpo_trainer.save_metrics("eval", metrics) 
   
   # Save the DPO model
   dpo_trainer.save_model(OUTPUT_DIR) 
   tokenizer.save_pretrained(OUTPUT_DIR) 
 
   return 


if __name__ == "__main__":
   main() 




