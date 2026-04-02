""" Training DPO """
import os 
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset, load_from_disk 
import torch 
from transformers import AutoTokenizer, BitsAndBytesConfig, Mistral3ForConditionalGeneration 
from trl import DPOConfig, DPOTrainer 
from peft import LoraConfig 
from pprint import pprint 
import json 


model_name = "./trained_models_SFT/Ministral3_3B_IFT_2" 
OUTPUT_DIR = "./trained_models_DPO/Ministral3_3B_DPO_adapter_3" 
LOG_DIR = "./trained_models_DPO/Ministral3_3B_DPO_adapter_3/logs" 
EVAL = False 

MISTRAL_CHAT_TEMPLATE = (
   "{{ bos_token }}"
   "{%- for message in messages -%}"
   "{%- if message['role'] == 'system' -%}"
   "{{ '[SYSTEM_PROMPT] ' + message['content'] + ' [/SYSTEM_PROMPT]' }}"
   "{%- elif message['role'] == 'user' -%}"
   "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
   "{%- elif message['role'] == 'assistant' -%}"
   "{{ ' ' + message['content'] + eos_token }}"
   "{%- endif -%}"
   "{%- endfor -%}"
) 


def main():
   torch.cuda.empty_cache() 

   dataset = load_dataset("json", data_files="dataset/orpo-dpo_mix_5k.jsonl", split="train")

   dataset_dpo_split = dataset.train_test_split(test_size=0.3, seed=42) 
   train_dataset_dpo, eval_dataset_dpo = dataset_dpo_split['train'], dataset_dpo_split['test'] 
   print(f"\n Train Data len: {len(train_dataset_dpo)}") 

   print("📚 Loading tokenizer...") 
   tokenizer = AutoTokenizer.from_pretrained(model_name) 
   tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE 

   if tokenizer.pad_token is None:
      tokenizer.pad_token = "<pad>" 
   
   tokenizer.padding_side = "right"

   print("🧠 Loading model...") 
   ## Optional 
   # bnb_config = BitsAndBytesConfig(
   #    load_in_4bit=True,
   #    bnb_4bit_compute_dtype=torch.bfloat16,
   #    bnb_4bit_quant_type="nf4",
   #    bnb_4bit_use_double_quant=True,
   # ) 

   model = Mistral3ForConditionalGeneration.from_pretrained(
      model_name, 
      device_map = "auto",
      low_cpu_mem_usage=True,
      # attn_implementation="flash_attention_2"
   ) 
   
   model.config.use_cache = False 

   ## Only for MoE models 
   model.config.output_router_logits = True 
   model.config.router_aux_loss_coef = 0.001 

   lora_config = LoraConfig(
      r=16,
      lora_alpha=32, 
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
      # target_modules = "all-linear", 
      lora_dropout = 0.05,
      bias="none",
      task_type="CAUSAL_LM",
   ) 

   # DPO Training configuration 
   dpo_config = DPOConfig(
      output_dir=OUTPUT_DIR,
      num_train_epochs = -1, 
      max_steps = 150, 
      per_device_train_batch_size=2,
      gradient_accumulation_steps=4,
      gradient_checkpointing=True,
      learning_rate=3e-6,  # recommended: 5e-6
      lr_scheduler_type="cosine", 
      optim="adamw_8bit",  # paged_adamw_8bit 
      adam_beta2 = 0.95,
      beta= 0.1,
      max_grad_norm=0.5,
      # loss_type= "robust", # robust | ipo | sigmoid 
      # label_smoothing = 0.05,
      # loss_type = ["sigmoid", "sft"],
      # loss_weights = [0.8, 0.2],
      warmup_ratio=0.05,
      logging_steps=5,
      save_steps=20,
      max_length=1024,
      precompute_ref_log_probs=True,
      bf16 = torch.cuda.is_bf16_supported(),
      # dataloader_num_workers=0,
      logging_dir=LOG_DIR,
      report_to="tensorboard",
      seed=40,
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




