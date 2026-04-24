# Private-LLM  

Train your own LLM with custom instructions using **HuggingFace** and open-weight base models.  
This is a ***POST-TRAINING*** pipeline for custom model tuning, which can be used to validate different LLM architectures with proprietary data.  


Each SECTION includes data_prep, training and inference scripts for models tested.  

*`flash_attention`* installation is optional. Doesn't provide much benefit for training in Windows env.  


## Installation and setup  

Tested with **Python 3.12.x**.  

1. `venv_init.bat`  
2. Activate *venv* and run `deps_install.bat` (use the same sequence for linux kernel)  


## Post-training (*wLoRA*)  

With an open-source BASE model from HuggingFace and your own data custom train your model following these 2 stages: ***SFT*** and ***RL***.  

It turns out you can use __LoRA__ in both stages to preserve base knowledge and create a fine-tuned version, mitigating catastrophic forgetting.  

Following is an optimal strategy:  
* Use a balanced diverse dataset (100k-200k) to QLoRA SFT a base model. Use this merged model as a baseline and create domain targetted LoRA adapters.  
* Use LoRA or full-BFLoat Reinforcement Learning, to optimize each adapters.  

** __*Another Option*__:  

* Merge SFT adapters for different domain into a single larger BASE version. Apply RL on it (+LoRA) with balanced dataset to create the *v1* Foundation Model.  
* Use this **v1** as a base model to further instruction-finetune using SFT for a custom application (proprietary or not).  



### Training Steps:  

For SFT and RL an off-the-shelf dataset from HuggingFace has been used with 2 types of data format: *ChatML* and *Mistral*.  

***Supervised Fine-tuning***:  

1. `SFT/data_prep.py`: Select the base model, paths and CHAT format. Add your custom data schema processing code here.  
2. `SFT/train.py`: Repeat the BASE model and CHAT format selection, update the training configs, dest path and RUN.  
3. `SFT/inference.py`: Select between Mistral and non-Mistral variants and *Merge* the adapter with the base model. Then **Load** and **Run**.  

***Re-inforcement Learning***:  

I have only validated Qwen2.5 and Ministral models as they lie on different ends of the spectrum (MoE and non-MoE).  

For DPO:  
1. __Data Processing__ (`RL/DPO/dpo_data_prep.py`): It download a multi-turn conversation dataset and converts it into single-turn version with `prompt`, `chosen` and `rejected` columns, as DPO needs them.  
2. __Training__: Individual training file inside `RL/DPO/.../...*train.py`.  
3. __Inference__(and merging): For merging and running test on RL adapters with base SFT model, use `RL/DPO/.../merge_and_inference_*.py`  


~~***GRPO and RLOO will be added in the future***~~  


## Notes  

Use `min_p` and `temp` hyperparams for best and repeatable inference output.  
For **Mistral**, use `AutoModelForCausalLM` and for **Ministral** variants use `Mistral3ForConditionalGeneration`.  

### Tested models  
* `mistralai/Mistral-7B-v0.3`  
* `mistralai/Ministral-3-3B-Base-2512` (*prefereed*)  
* `Qwen/Qwen2.5-7B` (*preferred*)  
* `LiquidAI/LFM2.5-1.2B-Base`

During SFT, ***Qwen models converge faster and also learns better***.  

~~***Next Version v0.2.0 includes auto-training and fine-tuning on local coversation data.***~~  

_______ 

***Please Leave a :star: if you like it***  

___________  

