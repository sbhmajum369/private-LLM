# Private-LLM  

Train your own LLM with custom instructions (Query-Response pair) using HuggingFace and open-weight base models.  

**SFT** folder includes data_prep, training and inference scripts for models tested.  

For testing, a demo dataset is loaded and processed using ChatML/Mistral format.  


*`flash_attention`* installation is optional. Doesn't provide much benefit for training in Windows env.  


## Installation and setup  

Tested with Python 3.12.x  

1. `venv_init.bat`  
2. Activate *venv* and run `deps_install.bat` (use the same sequence for linux kernel)  


## Training and Inference  

1. `data_prep.py`: Select the base model, paths and CHAT format. Add your custom data schema processing code here.  
2. `train.py`: Repeat the BASE model and CHAT format selection, update the training configs, dest path and RUN.  
3. `inference.py`: Select between Mistral and non-Mistral variants and *Merge* the adapter with the base model. Then **Load** and **Run**.  

Use `min_p` and `temp` hyperparams for best and repeatable inference output.  

### Tested models  

* `mistralai/Mistral-7B-v0.3`  
* `mistralai/Ministral-3-3B-Base-2512`  
* `Qwen/Qwen2.5-7B`  
* `LiquidAI/LFM2.5-1.2B-Base`


***Qwen models converge faster and also learns better***  


____  

__RL coming soon__ 

