import torch
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import gc
from flask import Flask, request

from utils.lewis import *

MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DeepseekMath:
    def __init__(self, model_path, devices, quant:bool = False):
        self.model_path = model_path
        self.quant = quant
        self.devices = devices

        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config.gradient_checkpointing = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.USE_PAST_KEY = True
        self.DEEP = True

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        transformers.set_seed(42)

        torch.cuda.empty_cache()
        gc.collect()

        print("Loading DeepseekMath Model ...")
        print(f"    devices: {self.devices}")
        print(f"    model path: {self.model_path}")

        if self.quant:
            self.device_map = "sequential"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype="auto",
                trust_remote_code=True, 
                quantization_config=quantization_config,
                config=self.config
            )
        else:
            self.device_map = self._get_device_map(devices)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype="auto",
                trust_remote_code=True,
                #quantization_config=quantization_config,
                config=self.config
            )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype='auto',
            device_map=self.device_map,
        )

        print("\nModel setup done.")
        print(f"    model.dtype: {self.model.dtype}")
        print(f"    model.hf_device_map: {self.model.hf_device_map}")

    def _get_device_map(devices):
        '''
        :device1:, :device2: cuda device indices
        '''
        if len(devices) > 2:
            raise Exception("Does not support device count > 2.")
        else:
            device1 = devices[0]
            device2 = devices[1] if len(devices) == 2 else device1

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

    def generate(self, input_text, *args, **kwargs):
        model_inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        generation_output = self.model.generate(**model_inputs, *args, **kwargs)
        output_ids = generation_output[0]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    '''
    def predict(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_tensor, max_new_tokens=512)

        return self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)'''

# app = Flask(__name__)
# model = Deepseek(MODEL_NAME, DEVICE)

# @app.route("/", methods=['POST'])
# def receive():
#     prompt = request.values.get('prompt')
#     response = model.predict(prompt)
#     return {'response': response}

if __name__ == '__main__':
    model_path = 
    model = DeepseekMath()

