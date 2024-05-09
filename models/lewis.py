import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch
import gc

import re
import sys
import subprocess

from models.base import Model, mod1000
import pandas as pd
import os
from datasets import MATH

from tqdm import tqdm

from sympy import symbols, simplify, Eq, solve
from latex2sympy2 import latex2sympy

class Lewis(Model):
    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.USE_PAST_KEY = True

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        transformers.set_seed(42)

        # what device??
        torch.cuda.empty_cache()
        gc.collect()

        self.MODEL_PATH = "/kaggle/input/deepseek-math"#"/kaggle/input/gemma/transformers/7b-it/1"
    DEEP = True

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.gradient_checkpointing = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    device_map = self._get_device_map(??)

    if QUANT:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="sequential",
            torch_dtype="auto",
            trust_remote_code=True, 
            quantization_config=quantization_config,
            config=config
        )
    else:  
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            #quantization_config=quantization_config,
            config=config
        )
    
    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype='auto',
    device_map=device_map,
)

    


    stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output"] #,  
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    model.dtype, model.hf_device_map


    def _get_device_map(device1, device2=None):
        '''
        :device1:, :device2: cuda device indices
        '''
        if device2 is None:
            device2 = device1

        device_map = [
            ('model.embed_tokens', device1),
            ('model.layers.0', device1),
            ('model.layers.1', device1),
            ('model.layers.2', device1),
            ('model.layers.3', device1),
            ('model.layers.4', device1),
            ('model.layers.5', device1),
            ('model.layers.6', device1),
            ('model.layers.7', device1),
            ('model.layers.8', device1),
            ('model.layers.9', device1),
            ('model.layers.10', device1),
            ('model.layers.11', device1),
            ('model.layers.12', device1),
            ('model.layers.13', device1),
            ('model.layers.14', device1),
            ('model.layers.15', device1),
            ('model.layers.16', device1),
            ('model.layers.17', device1),
            ('model.layers.18', device1),
            ('model.layers.19', device1),
            ('model.layers.20', device1),
            ('model.layers.21', device1),
            ('model.layers.22', device2),
            ('model.layers.23', device2),
            ('model.layers.24', device2),
            ('model.layers.25', device2),
            ('model.layers.26', device2),
            ('model.layers.27', device2),
            ('model.layers.28', device2),
            ('model.layers.29', device2),
            ('model.norm', device2),
            ('lm_head', device2)]

        device_map = {ii: jj for (ii, jj) in device_map}
        return device_map

# Example usage
device_map = generate_device_map(0, 1)
print(device_map)



class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop,last_token)):
                return True
        return False


def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)

def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""

def process_code(code, return_shell_output=False):
    
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)

    if return_shell_output:
        code = code.replace('\n', '\n    ')
            # Add a try...except block
        code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
    
    if not return_shell_output:
        print(code)
    with open('code.py', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        print(shell_output)
        if return_shell_output:
            if return_value=='FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value+='\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value))) % 1000
    except Exception as e:
        print(e,'shell_output')
        code_output = -1
    
    if return_shell_output:
        if code_output==-1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    
    
    return code_output


def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)

        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]

        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        
        else:
            result_output = round(float(eval(result_output))) % 1000
    
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output



    

    
