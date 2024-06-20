import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

HF_CACHE_DIR = "/home/master/13/hhhsu/models"
os.environ['HF_HOME'] = HF_CACHE_DIR

MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"

model_config = AutoConfig.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True, 
    #quantization_config=quantization_config,
    config=model_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

### Freezing the original weights ###
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


